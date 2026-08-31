"""Compare two arbitrary ComfyUI values with a synchronized preview UI."""

from __future__ import annotations

import difflib
import io as py_io
import json
import logging
import math
import sys
import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing_extensions import override

from comfy_api.latest import ComfyExtension, Input, InputImpl, io


_CATEGORY = "😺dzNodes/CineStyle"
_CACHE_NAMESPACE = "compare_any"
_MAX_PREVIEW_PIXELS = 1_000_000
_MAX_PREVIEW_FPS = 25.0
_MAX_TEXT_CHARS = 200_000
_MAX_JSON_DEPTH = 32
_MAX_REPR_CHARS = 1_000
_COMPARE_ROUTE_REGISTERED = False
_COMPARE_STORE = None
_COMPARE_PROGRESS: dict[str, dict[str, Any]] = {}
_COMPARE_PROGRESS_LOCK = threading.RLock()
_LOGGER = logging.getLogger(__name__)


def _set_progress(node_id: Any, progress: Any, message: str, info: dict[str, Any] | None = None, status: str = "running") -> None:
    key = str(node_id or "").strip()
    if not key:
        return
    try:
        value = max(0, min(100, int(round(float(progress)))))
    except (TypeError, ValueError, OverflowError):
        value = 0
    payload = {
        "status": str(status or "running"),
        "progress": value,
        "message": str(message or "Preparing comparison cache"),
        "updated": time.time(),
    }
    if info:
        payload["info"] = dict(info)
    should_log = False
    with _COMPARE_PROGRESS_LOCK:
        previous = _COMPARE_PROGRESS.get(key) or {}
        _COMPARE_PROGRESS[key] = payload
        previous_progress = int(previous.get("progress", -1) or -1)
        should_log = (
            str(previous.get("status") or "") != payload["status"]
            or value >= 100
            or value - previous_progress >= 5
        )
    if should_log:
        log_info = dict(info or {})
        for field in ("source_a", "source_b"):
            nested = log_info.get(field)
            if isinstance(nested, dict):
                log_info[field] = {
                    name: nested.get(name)
                    for name in ("kind", "frames", "fps", "duration", "width", "height", "has_audio", "completed", "total")
                    if name in nested
                }
        try:
            info_text = json.dumps(log_info, ensure_ascii=False, separators=(",", ":"), default=str)
        except (TypeError, ValueError):
            info_text = str(log_info)
        if len(info_text) > 1000:
            info_text = f"{info_text[:1000]}..."
        _LOGGER.info("[CS Compare Any][%s] %s (%d%%)%s", key, payload["message"], value, f" info={info_text}" if info_text else "")


class _CacheProgress:
    def __init__(self, node_id: str, start: float, span: float, total: int, message: str, info: dict[str, Any]):
        self.node_id = str(node_id)
        self.start = float(start)
        self.span = max(0.0, float(span))
        self.total = max(1, int(total))
        self.count = 0
        self.message = message
        self.info = info

    def update(self, amount: int = 1) -> None:
        self.count = min(self.total, self.count + max(1, int(amount or 1)))
        _set_progress(
            self.node_id,
            self.start + self.span * self.count / self.total,
            self.message,
            {**self.info, "completed": self.count, "total": self.total},
        )


def _safe_fps(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return 24.0
    return result if math.isfinite(result) and result > 0 else 24.0


def _safe_type_name(value: Any) -> str:
    value_type = type(value)
    module = str(getattr(value_type, "__module__", "") or "")
    name = str(getattr(value_type, "__qualname__", value_type.__name__) or value_type.__name__)
    return name if module in {"", "builtins"} else f"{module}.{name}"


def _rewind_source(source: Any) -> Any:
    if hasattr(source, "seek"):
        source.seek(0)
    return source


def _stream_rotation(stream: av.VideoStream) -> int:
    try:
        return int(round(float(stream.metadata.get("rotate", 0)) / 90.0)) % 4
    except (TypeError, ValueError):
        return 0


def _preview_dimensions(width: int, height: int) -> tuple[int, int]:
    width = max(1, int(width))
    height = max(1, int(height))
    scale = min(1.0, math.sqrt(_MAX_PREVIEW_PIXELS / float(width * height)))
    result_width = max(2, int(math.floor(width * scale)))
    result_height = max(2, int(math.floor(height * scale)))
    return result_width, result_height


def _resize_rgb_array(array: np.ndarray) -> np.ndarray:
    height, width = int(array.shape[0]), int(array.shape[1])
    target_width, target_height = _preview_dimensions(width, height)
    if (target_width, target_height) == (width, height):
        return np.ascontiguousarray(array, dtype=np.uint8)
    image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")
    return np.ascontiguousarray(np.asarray(image.resize((target_width, target_height), Image.Resampling.BILINEAR), dtype=np.uint8))


def _to_uint8(array: Any) -> np.ndarray:
    """Convert ComfyUI tensor-like image data to contiguous RGB uint8 frames."""
    if isinstance(array, torch.Tensor):
        value = array.detach()
        input_was_float = torch.is_floating_point(value)
        if value.ndim != 4:
            raise ValueError("IMAGE data must have shape [batch, height, width, channels].")
        if value.shape[-1] == 1:
            value = value.expand(*value.shape[:-1], 3)
        elif value.shape[-1] < 3:
            raise ValueError("IMAGE data must have one, three, or four channels.")
        else:
            value = value[..., :3]
        target_width, target_height = _preview_dimensions(int(value.shape[2]), int(value.shape[1]))
        if (target_width, target_height) != (int(value.shape[2]), int(value.shape[1])):
            value = F.interpolate(
                value.to(dtype=torch.float32).movedim(-1, 1),
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            ).movedim(1, -1)
        value = value.to(device="cpu").numpy()
    else:
        value = np.asarray(array)
    if value.ndim != 4:
        raise ValueError("IMAGE data must have shape [batch, height, width, channels].")
    if value.shape[0] == 0 or value.shape[1] <= 0 or value.shape[2] <= 0:
        raise ValueError("IMAGE data contains no frames.")
    if value.shape[-1] == 1:
        value = np.repeat(value, 3, axis=-1)
    elif value.shape[-1] < 3:
        raise ValueError("IMAGE data must have one, three, or four channels.")
    else:
        value = value[..., :3]
    if np.issubdtype(value.dtype, np.floating) and (not isinstance(array, torch.Tensor) or input_was_float):
        value = np.clip(value, 0.0, 1.0) * 255.0
    else:
        value = np.clip(value, 0, 255)
    result = np.ascontiguousarray(np.rint(value), dtype=np.uint8)
    if not isinstance(array, torch.Tensor):
        target_width, target_height = _preview_dimensions(int(result.shape[2]), int(result.shape[1]))
        if (target_width, target_height) != (int(result.shape[2]), int(result.shape[1])):
            result = np.stack([_resize_rgb_array(frame) for frame in result], axis=0)
    return result


def _mask_to_uint8(value: torch.Tensor) -> np.ndarray:
    if value.ndim != 3:
        raise ValueError("MASK data must have shape [batch, height, width].")
    if value.shape[0] == 0 or value.shape[1] <= 0 or value.shape[2] <= 0:
        raise ValueError("MASK data contains no frames.")
    tensor = value.detach().to(dtype=torch.float32)
    target_width, target_height = _preview_dimensions(int(tensor.shape[2]), int(tensor.shape[1]))
    if (target_width, target_height) != (int(tensor.shape[2]), int(tensor.shape[1])):
        tensor = F.interpolate(tensor.unsqueeze(1), size=(target_height, target_width), mode="nearest").squeeze(1)
    array = tensor.to(device="cpu").numpy()
    array = np.clip(array, 0.0, 1.0) * 255.0
    array = np.rint(array).astype(np.uint8, copy=False)[..., None]
    return np.ascontiguousarray(np.repeat(array, 3, axis=-1))


@dataclass
class _Media:
    kind: str
    frames: np.ndarray
    fps: float
    audio: dict[str, Any] | None = None

    @property
    def width(self) -> int:
        return int(self.frames.shape[2])

    @property
    def height(self) -> int:
        return int(self.frames.shape[1])

    @property
    def count(self) -> int:
        return int(self.frames.shape[0])

    @property
    def duration(self) -> float:
        return self.count / _safe_fps(self.fps)


def _normalise_audio(audio: Any) -> dict[str, Any] | None:
    if not isinstance(audio, dict) or not isinstance(audio.get("waveform"), torch.Tensor):
        return None
    waveform = audio["waveform"]
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    try:
        sample_rate = int(audio.get("sample_rate", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        return None
    if waveform.ndim != 3 or waveform.numel() == 0 or sample_rate <= 0:
        return None
    waveform = waveform[:1, :2].detach().to(device="cpu", dtype=torch.float32).contiguous()
    return {"waveform": waveform, "sample_rate": sample_rate}


def _decode_stream_audio(source: Any, start_time: float, duration: float) -> dict[str, Any] | None:
    chunks: list[np.ndarray] = []
    with av.open(_rewind_source(source), mode="r") as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        sample_rate = int(getattr(stream, "rate", 0) or getattr(stream.codec_context, "sample_rate", 0) or 48000)
        channels = int(getattr(stream.codec_context, "channels", 0) or 2)
        layout = "mono" if channels <= 1 else "stereo"
        resampler = av.audio.resampler.AudioResampler(format="fltp", layout=layout, rate=sample_rate)
        for decoded in container.decode(stream):
            for converted in resampler.resample(decoded):
                values = converted.to_ndarray()
                if values.ndim == 1:
                    values = values[None, :]
                if values.ndim == 2 and values.shape[1] > 0:
                    chunks.append(np.asarray(values[:2], dtype=np.float32))
        for converted in resampler.resample(None):
            values = converted.to_ndarray()
            if values.ndim == 1:
                values = values[None, :]
            if values.ndim == 2 and values.shape[1] > 0:
                chunks.append(np.asarray(values[:2], dtype=np.float32))
    if not chunks:
        return None
    waveform = np.concatenate(chunks, axis=1)
    start_sample = max(0, int(round(float(start_time) * sample_rate)))
    if duration > 0:
        end_sample = max(start_sample, int(round((float(start_time) + float(duration)) * sample_rate)))
        waveform = waveform[:, start_sample:end_sample]
    else:
        waveform = waveform[:, start_sample:]
    if waveform.shape[1] == 0:
        return None
    return {"waveform": torch.from_numpy(np.ascontiguousarray(waveform[None, :2])), "sample_rate": sample_rate}


def _decode_video(value: Input.Video) -> _Media:
    if isinstance(value, InputImpl.VideoFromComponents):
        components = value.get_components()
        images = components.images
        if not isinstance(images, torch.Tensor):
            raise ValueError("VIDEO components contain no image tensor.")
        frames = _to_uint8(images)
        return _Media("VIDEO", frames, _safe_fps(components.frame_rate), _normalise_audio(components.audio))

    source = value.get_stream_source()
    start_time, requested_duration = value.get_active_trim_window()
    start_time = max(0.0, float(start_time or 0.0))
    requested_duration = max(0.0, float(requested_duration or 0.0))
    decoded_frames: list[np.ndarray] = []
    with av.open(_rewind_source(source), mode="r") as container:
        if not container.streams.video:
            raise ValueError("VIDEO contains no decodable video stream.")
        stream = container.streams.video[0]
        fps = _safe_fps(stream.average_rate)
        interval = 1.0 / fps
        end_time = start_time + requested_duration if requested_duration else None
        if start_time > 0:
            try:
                container.seek(int(start_time / (stream.time_base or Fraction(1, 1))), stream=stream, backward=True)
            except (av.error.FFmpegError, ValueError, ZeroDivisionError):
                pass
        fallback_index = 0
        for decoded in container.decode(stream):
            if decoded.pts is not None and stream.time_base is not None:
                frame_time = float(decoded.pts * stream.time_base)
            else:
                frame_time = start_time + fallback_index * interval
            fallback_index += 1
            if frame_time + interval * 0.5 < start_time:
                continue
            if end_time is not None and frame_time >= end_time:
                break
            array = decoded.to_ndarray(format="rgb24")
            rotation = int(round(float(getattr(decoded, "rotation", 0) or 0) / 90.0)) % 4
            if not rotation:
                rotation = _stream_rotation(stream)
            if rotation:
                array = np.rot90(array, k=rotation, axes=(0, 1)).copy()
            decoded_frames.append(_resize_rgb_array(array[..., :3]))
    if not decoded_frames:
        raise ValueError("VIDEO contains no frames in the active trim window.")
    try:
        audio = _decode_stream_audio(source, start_time, len(decoded_frames) / fps)
    except (OSError, ValueError, RuntimeError, av.error.FFmpegError):
        audio = None
    return _Media("VIDEO", np.stack(decoded_frames, axis=0), fps, audio)


def _classify(value: Any) -> str:
    if isinstance(value, Input.Video):
        return "VIDEO"
    if isinstance(value, torch.Tensor):
        if value.ndim == 4 and int(value.shape[-1]) in {1, 3, 4}:
            return "IMAGE"
        if value.ndim == 3:
            return "MASK"
        return "UNSUPPORTED"
    if isinstance(value, str):
        return "STRING"
    if isinstance(value, bool):
        return "BOOL"
    if isinstance(value, int):
        return "INT"
    if isinstance(value, float):
        return "FLOAT"
    if isinstance(value, (list, tuple)):
        return "LIST"
    if isinstance(value, dict):
        if ("waveform" in value or "sample_rate" in value) and isinstance(value.get("waveform"), torch.Tensor):
            return "UNSUPPORTED"
        if "samples" in value and isinstance(value.get("samples"), torch.Tensor):
            return "UNSUPPORTED"
        return "DICT"
    return "UNSUPPORTED"


def _upstream_output_is_list(node_cls: type[io.ComfyNode], input_name: str) -> bool:
    prompt = getattr(getattr(node_cls, "hidden", None), "prompt", None)
    node_id = str(getattr(getattr(node_cls, "hidden", None), "unique_id", "") or "")
    if not isinstance(prompt, dict) or not node_id:
        return False
    current = prompt.get(node_id) or prompt.get(str(node_id))
    if not isinstance(current, dict):
        return False
    link = (current.get("inputs") or {}).get(input_name)
    if not isinstance(link, (list, tuple)) or len(link) < 2:
        return False
    upstream = prompt.get(str(link[0])) or prompt.get(link[0])
    if not isinstance(upstream, dict):
        return False
    try:
        import nodes

        upstream_class = nodes.NODE_CLASS_MAPPINGS.get(str(upstream.get("class_type") or ""))
        if upstream_class is None:
            return False
        output_index = int(link[1])
        output_is_list = getattr(upstream_class, "OUTPUT_IS_LIST", ())
        return output_index < len(output_is_list) and bool(output_is_list[output_index])
    except (AttributeError, IndexError, TypeError, ValueError):
        return False


def _unwrap_input(value: Any, declared_list: bool) -> Any:
    """INPUT_IS_LIST wraps normal values once; preserve real/output lists."""
    if declared_list:
        return list(value) if isinstance(value, (list, tuple)) else [value]
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _safe_jsonable(value: Any, depth: int = 0, seen: set[int] | None = None) -> Any:
    if depth > _MAX_JSON_DEPTH:
        return "<max depth>"
    if seen is None:
        seen = set()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    identity = id(value)
    if identity in seen:
        return "<circular reference>"
    if isinstance(value, (list, tuple)):
        seen.add(identity)
        result = [_safe_jsonable(item, depth + 1, seen) for item in value]
        seen.discard(identity)
        return result
    if isinstance(value, dict):
        seen.add(identity)
        result = {str(key): _safe_jsonable(item, depth + 1, seen) for key, item in value.items()}
        seen.discard(identity)
        return result
    if isinstance(value, torch.Tensor):
        return f"<torch.Tensor shape={tuple(int(item) for item in value.shape)} dtype={value.dtype}>"
    text = repr(value)
    return text if len(text) <= _MAX_REPR_CHARS else f"{text[:_MAX_REPR_CHARS]}..."


def _serialise_value(value: Any, kind: str) -> str:
    if kind == "STRING":
        text = value
    elif kind in {"INT", "FLOAT", "BOOL"}:
        text = str(value)
    else:
        text = json.dumps(_safe_jsonable(value), ensure_ascii=False, indent=2, sort_keys=kind == "DICT")
    text = str(text)
    if len(text) <= _MAX_TEXT_CHARS:
        return text
    omitted = len(text) - _MAX_TEXT_CHARS
    return f"{text[:_MAX_TEXT_CHARS]}\n\n... truncated {omitted} characters"


def _inline_parts(left: str, right: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    matcher = difflib.SequenceMatcher(a=left, b=right, autojunk=False)
    left_parts: list[dict[str, str]] = []
    right_parts: list[dict[str, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            left_parts.append({"kind": "equal", "text": left[i1:i2]})
            right_parts.append({"kind": "equal", "text": right[j1:j2]})
        else:
            if i1 != i2:
                left_parts.append({"kind": "changed", "text": left[i1:i2]})
            if j1 != j2:
                right_parts.append({"kind": "changed", "text": right[j1:j2]})
    return left_parts, right_parts


def _diff_lines(left: str, right: str) -> list[dict[str, Any]]:
    left_lines = left.splitlines()
    right_lines = right.splitlines()
    if not left_lines:
        left_lines = [""]
    if not right_lines:
        right_lines = [""]
    matcher = difflib.SequenceMatcher(a=left_lines, b=right_lines, autojunk=False)
    rows: list[dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for line_left, line_right in zip(left_lines[i1:i2], right_lines[j1:j2]):
                parts_left, parts_right = _inline_parts(line_left, line_right)
                rows.append({"op": "equal", "a": line_left, "b": line_right, "a_parts": parts_left, "b_parts": parts_right})
        elif tag == "delete":
            for line_left in left_lines[i1:i2]:
                rows.append({"op": "delete", "a": line_left, "b": "", "a_parts": [{"kind": "changed", "text": line_left}], "b_parts": []})
        elif tag == "insert":
            for line_right in right_lines[j1:j2]:
                rows.append({"op": "insert", "a": "", "b": line_right, "a_parts": [], "b_parts": [{"kind": "changed", "text": line_right}]})
        else:
            count = max(i2 - i1, j2 - j1)
            for offset in range(count):
                line_left = left_lines[i1 + offset] if i1 + offset < i2 else ""
                line_right = right_lines[j1 + offset] if j1 + offset < j2 else ""
                parts_left, parts_right = _inline_parts(line_left, line_right)
                rows.append({"op": "replace", "a": line_left, "b": line_right, "a_parts": parts_left, "b_parts": parts_right})
    return rows


def _preview_cache_module() -> Any:
    package = __name__.rsplit(".", 1)[0]
    return sys.modules.get(f"{package}._py_preview_cache")


def _compare_store() -> Any:
    global _COMPARE_STORE
    if _COMPARE_STORE is None:
        module = _preview_cache_module()
        if module is None or not hasattr(module, "PreviewCacheStore"):
            raise RuntimeError("The shared preview cache is unavailable.")
        _COMPARE_STORE = module.PreviewCacheStore(_CACHE_NAMESPACE, max_entries=16, max_bytes=8 * 1024**3)
    return _COMPARE_STORE


def _normalise_pair(media_a: _Media, media_b: _Media) -> tuple[np.ndarray, np.ndarray, float, int]:
    timeline_fps = min(_MAX_PREVIEW_FPS, max(_safe_fps(media_a.fps), _safe_fps(media_b.fps)))
    total_frames = max(1, int(math.ceil(max(media_a.duration, media_b.duration) * timeline_fps - 1e-9)))

    def fill(media: _Media) -> np.ndarray:
        result = np.zeros((total_frames, media.height, media.width, 3), dtype=np.uint8)
        for index in range(total_frames):
            timestamp = index / timeline_fps
            if timestamp < media.duration - 1e-9:
                source_index = min(media.count - 1, max(0, int(math.floor(timestamp * media.fps + 1e-9))))
                result[index] = media.frames[source_index]
        return result

    return fill(media_a), fill(media_b), timeline_fps, total_frames


def _source_info(media: _Media) -> dict[str, Any]:
    audio = media.audio or {}
    waveform = audio.get("waveform") if isinstance(audio, dict) else None
    return {
        "kind": media.kind,
        "frames": media.count,
        "fps": _safe_fps(media.fps),
        "duration": media.duration,
        "width": media.width,
        "height": media.height,
        "has_audio": isinstance(waveform, torch.Tensor) and waveform.numel() > 0,
        "audio_sample_rate": int(audio.get("sample_rate", 0) or 0) if isinstance(audio, dict) else 0,
        "audio_channels": int(waveform.shape[1]) if isinstance(waveform, torch.Tensor) and waveform.ndim == 3 else 0,
    }


def _media_value(value: Any, kind: str) -> _Media:
    if kind == "VIDEO":
        return _decode_video(value)
    if kind == "IMAGE":
        return _Media(kind, _to_uint8(value), 1.0)
    if kind == "MASK":
        return _Media(kind, _mask_to_uint8(value), 1.0)
    raise ValueError(f"Unsupported media type: {kind}")


def _png_response(array: np.ndarray) -> Any:
    from aiohttp import web

    buffer = py_io.BytesIO()
    Image.fromarray(np.asarray(array[0], dtype=np.uint8), mode="RGB").save(buffer, format="PNG", optimize=False)
    return web.Response(body=buffer.getvalue(), content_type="image/png", headers={"Cache-Control": "no-store"})


async def _compare_any_frame_route(request: Any) -> Any:
    from aiohttp import web

    try:
        token = str(request.query.get("token") or "")
        index = int(request.query.get("frame", 0))
        entry = _compare_store().get_token(token)
        if entry is None:
            return web.json_response({"error": "Compare Any preview cache not found."}, status=404)
        frames = np.load(str(entry["frames_path"]), mmap_mode="r", allow_pickle=False)
        index = max(0, min(index, int(frames.shape[0]) - 1))
        return _png_response(np.array(frames[index : index + 1], copy=True))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def _compare_any_progress_route(request: Any) -> Any:
    from aiohttp import web

    node_id = str(request.query.get("node_id") or "").strip()
    with _COMPARE_PROGRESS_LOCK:
        progress = dict(_COMPARE_PROGRESS.get(node_id) or {})
    if not progress:
        return web.json_response({"status": "idle", "progress": 0, "message": "Preparing comparison cache"})
    return web.json_response(progress)


async def _compare_any_video_route(request: Any) -> Any:
    from aiohttp import web

    entry = _compare_store().get_token(request.query.get("token", ""))
    path = Path(str((entry or {}).get("video_path") or (entry or {}).get("path") or "")) if entry else None
    if entry is None or path is None or not path.is_file():
        return web.json_response({"error": "Compare Any preview video not found."}, status=404)
    return web.FileResponse(path=path, headers={"Content-Type": "video/mp4", "Cache-Control": "no-store"})


class CSCompareAny(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CS_Compare_Any",
            display_name="CS Compare Any",
            category=_CATEGORY,
            essentials_category="Utilities",
            description="Compare two values of the same ComfyUI type with synchronized media and text diff previews.",
            search_aliases=["compare any", "diff any", "compare image", "compare video"],
            inputs=[
                io.AnyType.Input("source_a", tooltip="First value to compare."),
                io.AnyType.Input("source_b", tooltip="Second value to compare."),
            ],
            hidden=[io.Hidden.unique_id, io.Hidden.prompt],
            outputs=[],
            is_input_list=True,
            is_output_node=True,
        )

    @classmethod
    def execute(cls, source_a: Any, source_b: Any) -> io.NodeOutput:
        node_id = str(getattr(getattr(cls, "hidden", None), "unique_id", "") or "compare")
        declared_a = _upstream_output_is_list(cls, "source_a")
        declared_b = _upstream_output_is_list(cls, "source_b")
        value_a = _unwrap_input(source_a, declared_a)
        value_b = _unwrap_input(source_b, declared_b)
        kind_a = _classify(value_a)
        kind_b = _classify(value_b)
        payload: dict[str, Any]
        _set_progress(node_id, 0, "Preparing comparison", {"type_a": kind_a, "type_b": kind_b})
        try:
            if kind_a != kind_b or kind_a == "UNSUPPORTED":
                detail = "source_a and source-b must be the same type"
                if kind_a == kind_b == "UNSUPPORTED":
                    detail = "The connected values are not supported by CS Compare Any."
                payload = {
                    "version": 1,
                    "mode": "error",
                    "type_a": kind_a,
                    "type_b": kind_b,
                    "error": detail,
                }
                _set_progress(node_id, 100, "Comparison ready", {"mode": "error", "type_a": kind_a, "type_b": kind_b}, status="ready")
            elif kind_a in {"VIDEO", "IMAGE", "MASK"}:
                _set_progress(node_id, 5, "Decoding source A", {"media_kind": kind_a})
                media_a = _media_value(value_a, kind_a)
                _set_progress(node_id, 30, "Decoding source B", {"media_kind": kind_a, "source_a": _source_info(media_a)})
                media_b = _media_value(value_b, kind_b)
                _set_progress(node_id, 52, "Normalizing synchronized timeline", {"source_a": _source_info(media_a), "source_b": _source_info(media_b)})
                frames_a, frames_b, fps, total_frames = _normalise_pair(media_a, media_b)
                store = _compare_store()
                store.clear_node(node_id)
                info_a = {**_source_info(media_a), "timeline_frames": total_frames, "timeline_fps": fps}
                info_b = {**_source_info(media_b), "timeline_frames": total_frames, "timeline_fps": fps}
                entry_a = store.put(
                    node_id,
                    frames_a,
                    fps,
                    variant="a",
                    encode_video=True,
                    info=info_a,
                    audio=media_a.audio,
                    progress=_CacheProgress(node_id, 55, 20, total_frames, "Encoding source A preview", info_a),
                )
                _set_progress(node_id, 76, "Encoding source B preview", {"source_a": info_a, "source_b": info_b})
                entry_b = store.put(
                    node_id,
                    frames_b,
                    fps,
                    variant="b",
                    encode_video=True,
                    info=info_b,
                    audio=media_b.audio,
                    progress=_CacheProgress(node_id, 76, 20, total_frames, "Encoding source B preview", info_b),
                )
                payload = {
                    "version": 1,
                    "mode": "media",
                    "media_kind": kind_a,
                    "timeline": {"frames": total_frames, "fps": fps, "duration": total_frames / fps},
                    "sources": {
                        "a": {**info_a, "token": str(entry_a.get("token") or ""), "video_url": f"/cinestyle/compare-any-video?token={entry_a.get('token')}"},
                        "b": {**info_b, "token": str(entry_b.get("token") or ""), "video_url": f"/cinestyle/compare-any-video?token={entry_b.get('token')}"},
                    },
                }
                _set_progress(node_id, 100, "Comparison cache ready", {"mode": "media", "media_kind": kind_a, "timeline_frames": total_frames}, status="ready")
            else:
                text_a = _serialise_value(value_a, kind_a)
                text_b = _serialise_value(value_b, kind_b)
                rows = _diff_lines(text_a, text_b)
                payload = {
                    "version": 1,
                    "mode": "diff",
                    "diff_kind": kind_a,
                    "type_a": kind_a,
                    "type_b": kind_b,
                    "a_text": text_a,
                    "b_text": text_b,
                    "rows": rows,
                }
                _set_progress(node_id, 100, "Comparison ready", {"mode": "diff", "diff_kind": kind_a, "rows": len(rows)}, status="ready")
        except Exception as exc:
            payload = {
                "version": 1,
                "mode": "error",
                "type_a": kind_a,
                "type_b": kind_b,
                "error": str(exc),
            }
            _set_progress(node_id, 100, str(exc), {"mode": "error", "type_a": kind_a, "type_b": kind_b, "error": str(exc)[:500]}, status="failed")
        return io.NodeOutput(ui={"compare_any": (payload,)})


class CompareAnyExtension(ComfyExtension):
    @override
    async def on_load(self) -> None:
        global _COMPARE_ROUTE_REGISTERED
        if _COMPARE_ROUTE_REGISTERED:
            return
        from server import PromptServer

        server_instance = getattr(PromptServer, "instance", None)
        if server_instance is not None:
            server_instance.routes.get("/cinestyle/compare-any-frame")(_compare_any_frame_route)
            server_instance.routes.get("/cinestyle/compare-any-progress")(_compare_any_progress_route)
            server_instance.routes.get("/cinestyle/compare-any-video")(_compare_any_video_route)
            _COMPARE_ROUTE_REGISTERED = True

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CSCompareAny]


async def comfy_entrypoint() -> CompareAnyExtension:
    return CompareAnyExtension()
