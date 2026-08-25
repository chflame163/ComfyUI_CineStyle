"""Subtitle timeline and burn-in node for standard ComfyUI VIDEO values."""

from __future__ import annotations

import re
import sys
import json
import hashlib
import importlib.util
import logging
import os
from fractions import Fraction
from urllib.parse import quote, unquote
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - ComfyUI normally provides tqdm
    tqdm = None

import folder_paths
from comfy_api.latest import ComfyExtension, InputImpl, Types, io


_CATEGORY = "😺dzNodes/CineStyle/Video"
_ROUTE_REGISTERED = False
_TIME_RE = re.compile(r"^(\d+):(\d{2}):(\d{2})[,.](\d{3})$")
_SUBTITLE_SRT_CACHE: dict[str, dict[str, str]] = {}
_SUBTITLE_PROXY_RENDER_SIZE: dict[str, tuple[int, int]] = {}
_SUBTITLE_SOURCE_INFO: dict[str, dict[str, Any]] = {}
_PREVIEW_CACHE_STORE = None
_SUBTITLE_LOGGER = logging.getLogger("CineStyleVideoSubtitle")


def _subtitle_info(message: str, *args: Any) -> None:
    _SUBTITLE_LOGGER.info("[CS Video Subtitle] " + message, *args)


def _preview_cache_store():
    global _PREVIEW_CACHE_STORE
    if _PREVIEW_CACHE_STORE is None:
        package = __name__.rsplit(".", 1)[0]
        module = sys.modules.get(f"{package}._py_preview_cache")
        if module is None:
            raise RuntimeError("CineStyle preview cache module is unavailable.")
        _PREVIEW_CACHE_STORE = module.PreviewCacheStore("video_subtitle")
    return _PREVIEW_CACHE_STORE


class _SubtitleProgress:
    """Emit the same console-friendly progress style as the video nodes."""

    def __init__(self, total: int, description: str = "frame processing"):
        self.bar = None
        if tqdm is not None:
            self.bar = tqdm(
                total=max(1, int(total)),
                desc=f"[INFO] [CS Video Subtitle] {description}",
                unit="frame",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                mininterval=0.1,
                dynamic_ncols=True,
                leave=True,
            )

    def update(self, amount: int = 1) -> None:
        if self.bar is not None:
            self.bar.update(max(0, int(amount)))

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()


def _fonts_root() -> Path:
    return Path(folder_paths.models_dir) / "fonts"


def _font_files() -> list[str]:
    renderer = _renderer_module()
    return renderer.font_files(_fonts_root())


def _renderer_module():
    module = sys.modules.get(f"{__package__}._py_subtitle_renderer")
    if module is not None:
        return module
    module = next(
        (candidate for name, candidate in sys.modules.items() if name.endswith("._py_subtitle_renderer")),
        None,
    )
    if module is not None:
        return module
    renderer_path = Path(__file__).with_name("subtitle_renderer.py")
    spec = importlib.util.spec_from_file_location("_cinestyle_subtitle_renderer", renderer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load subtitle renderer from {renderer_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_time(value: str) -> float:
    match = _TIME_RE.match(value.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value!r}")
    hours, minutes, seconds, milliseconds = (int(item) for item in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Invalid SRT timestamp: {value!r}")
    return hours * 3600.0 + minutes * 60.0 + seconds + milliseconds / 1000.0


def parse_srt(text: str) -> list[dict[str, Any]]:
    """Parse SRT cues while tolerating BOM, blank lines and cue settings."""
    source = str(text or "").replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", source.strip()) if source.strip() else []
    cues: list[dict[str, Any]] = []
    for block in blocks:
        lines = [line.rstrip() for line in block.split("\n")]
        if not lines:
            continue
        if "-->" not in lines[0]:
            lines = lines[1:]
        if not lines or "-->" not in lines[0]:
            continue
        start_text, end_text = [part.strip().split(" ", 1)[0] for part in lines[0].split("-->", 1)]
        start = _parse_time(start_text)
        end = _parse_time(end_text)
        value = "\n".join(lines[1:]).strip()
        if value and end > start:
            cues.append({"id": len(cues) + 1, "start": start, "end": end, "text": value})
    return cues


def _coerce_cues(srt: str, edited_srt: str = "") -> list[dict[str, Any]]:
    return parse_srt(edited_srt.strip() or srt)


def _format_srt_time(seconds: float) -> str:
    milliseconds = max(0, int(round(float(seconds) * 1000.0)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds_value, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds_value:02d},{millis:03d}"


def _cues_to_srt(cues: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, cue in enumerate(cues, 1):
        text = str(cue.get("text", "")).strip()
        start = float(cue.get("start", 0.0))
        end = float(cue.get("end", start + 0.05))
        if text and end > start:
            blocks.append(f"{index}\n{_format_srt_time(start)} --> {_format_srt_time(end)}\n{text}")
    return "\n\n".join(blocks) + ("\n\n" if blocks else "")


def _srt_source_hash(value: str) -> str:
    return hashlib.sha1(str(value or "").encode("utf-8")).hexdigest()


def _coerce_srt_input(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 1 and isinstance(value[0], str):
        return value[0]
    raise ValueError("srt input must be a text value containing valid SRT cues.")


def _safe_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = float(value)
        if not np.isfinite(number):
            raise ValueError
        return max(minimum, min(maximum, int(round(number))))
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_float(value: Any, default: float, minimum: float, maximum: float, decimals: int | None = None) -> float:
    try:
        number = float(value)
        if not np.isfinite(number):
            raise ValueError
        number = max(minimum, min(maximum, number))
        return round(number, decimals) if decimals is not None else number
    except (TypeError, ValueError, OverflowError):
        return default


def _normalize_hex(value: Any, default: str) -> str:
    text = str(value or "").strip().upper()
    return text if re.fullmatch(r"#[0-9A-F]{6}", text) else default


def _clear_video_caches(node_id: Any) -> None:
    key = str(node_id or "").strip()
    if not key:
        return
    _preview_cache_store().clear_node(key)


def _source_filename_from_metadata(metadata: dict[str, Any]) -> str:
    for name in ("source_filename", "filename", "source_file", "video_file", "video_path", "path", "source"):
        value = metadata.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _video_source_filename(video: Any, metadata: dict[str, Any]) -> str:
    value = _source_filename_from_metadata(metadata)
    if value:
        return value
    getter = getattr(video, "get_stream_source", None)
    if callable(getter):
        try:
            source = getter()
            if isinstance(source, (str, os.PathLike)) and str(source).strip():
                return str(source).strip()
        except Exception:
            pass
    return ""


def _source_info_for_node(node_id: Any) -> dict[str, Any]:
    return _SUBTITLE_SOURCE_INFO.get(str(node_id or "").strip(), {})


def _adapt_source_frames(array: np.ndarray, source_fps: float, source_info: dict[str, Any]) -> tuple[np.ndarray, float]:
    metadata = source_info.get("metadata") if isinstance(source_info.get("metadata"), dict) else {}
    frame_count = int(array.shape[0])
    start_frame = 0
    end_frame = frame_count - 1
    has_frame_window = "start_frame" in metadata or "end_frame" in metadata
    if not has_frame_window:
        try:
            trim_start = float(metadata.get("active_start_seconds", 0.0) or 0.0)
            trim_duration = float(metadata.get("active_duration_seconds", 0.0) or 0.0)
            if trim_start > 0:
                start_frame = max(0, min(frame_count - 1, int(round(trim_start * source_fps))))
            if trim_duration > 0:
                end_frame = min(frame_count - 1, start_frame + max(1, int(round(trim_duration * source_fps))) - 1)
        except (TypeError, ValueError, OverflowError):
            pass
    try:
        if has_frame_window:
            start_frame = max(0, min(frame_count - 1, int(float(metadata.get("start_frame", 0) or 0))))
    except (TypeError, ValueError, OverflowError):
        start_frame = 0
    try:
        if has_frame_window:
            requested_end = int(float(metadata.get("end_frame", -1) or -1))
            if requested_end >= 0:
                end_frame = min(frame_count - 1, requested_end)
    except (TypeError, ValueError, OverflowError):
        end_frame = frame_count - 1
    if end_frame < start_frame:
        end_frame = start_frame
    array = array[start_frame:end_frame + 1]

    target_fps = source_fps
    try:
        candidate_fps = float(metadata.get("loaded_fps", source_fps) or source_fps)
        if np.isfinite(candidate_fps) and candidate_fps > 0:
            target_fps = candidate_fps
    except (TypeError, ValueError, OverflowError):
        pass
    if array.shape[0] > 1 and abs(source_fps - target_fps) >= 1e-6:
        output_count = max(1, int(round(array.shape[0] * target_fps / source_fps)))
        indices = np.rint(np.linspace(0, array.shape[0] - 1, output_count)).astype(np.int64)
        array = array[indices]

    try:
        target_width = int(float(metadata.get("loaded_width", 0) or 0))
        target_height = int(float(metadata.get("loaded_height", 0) or 0))
    except (TypeError, ValueError, OverflowError):
        target_width = target_height = 0
    if target_width > 0 and target_height > 0 and (array.shape[2] != target_width or array.shape[1] != target_height):
        resized = []
        for frame in array:
            resized.append(
                av.VideoFrame.from_ndarray(frame, format="rgb24")
                .reformat(width=target_width, height=target_height, format="rgb24")
                .to_ndarray(format="rgb24")
            )
        array = np.stack(resized, axis=0).astype(np.uint8, copy=False)
    return array, target_fps


def _extract_video_frames(video: Any) -> tuple[np.ndarray, float, dict[str, Any], Any]:
    components = video.get_components()
    images = components.images
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] == 0:
        raise ValueError("VIDEO contains no decodable frames.")
    fps = float(components.frame_rate)
    safe_fps = fps if np.isfinite(fps) and fps > 0 else 24.0
    frames = (
        images[..., :3]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .contiguous()
        .numpy()
    )
    info = {
        "frames": int(frames.shape[0]),
        "width": int(frames.shape[2]),
        "height": int(frames.shape[1]),
        "fps": safe_fps,
        "duration": float(frames.shape[0]) / safe_fps,
    }
    return frames, safe_fps, info, getattr(components, "audio", None)


def _cache_video_source(
    video: Any,
    node_id: Any,
    variant: str,
    cache_label: str,
    render_size: tuple[int, int] | None = None,
    encode_video: bool = True,
) -> bool:
    """Write an independent frame/video cache for one subtitle node source."""
    if video is None or not node_id:
        return False
    if render_size is not None and variant == "preview":
        _SUBTITLE_PROXY_RENDER_SIZE[str(node_id)] = (int(render_size[0]), int(render_size[1]))
    try:
        frames, safe_fps, info, audio = _extract_video_frames(video)
        return _store_frame_cache(frames, safe_fps, info, node_id, variant, cache_label, encode_video, audio)
    except Exception as exc:
        _subtitle_info("%s cache unavailable: %s", cache_label, exc)
    return False


def _prepare_audio(audio: Any) -> dict[str, Any] | None:
    if not isinstance(audio, dict):
        return None
    waveform = audio.get("waveform")
    if not isinstance(waveform, torch.Tensor) or waveform.numel() == 0:
        return None
    try:
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 3 or waveform.shape[1] <= 0 or waveform.shape[2] <= 0:
            return None
        sample_rate = int(audio.get("sample_rate", 0) or 0)
        if sample_rate <= 0:
            return None
        return {
            "waveform": waveform.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            "sample_rate": sample_rate,
        }
    except (TypeError, ValueError):
        return None


def _store_frame_cache(
    frames: np.ndarray,
    safe_fps: float,
    info: dict[str, Any],
    node_id: Any,
    variant: str,
    cache_label: str,
    encode_video: bool = True,
    audio: Any = None,
) -> bool:
    entry = _preview_cache_store().put(
        node_id,
        frames,
        safe_fps,
        variant=variant,
        encode_video=encode_video,
        info=info,
        audio=_prepare_audio(audio) if not encode_video else None,
    )
    _subtitle_info(
        "%s cache ready: frames=%d, size=%dx%d, fps=%.3f",
        cache_label,
        info["frames"],
        info["width"],
        info["height"],
        safe_fps,
    )
    return entry is not None


def _resolve_video_file(filename: str) -> Path:
    value = str(filename or "").strip()
    if not value:
        raise FileNotFoundError(value)
    if folder_paths.exists_annotated_filepath(value):
        return Path(folder_paths.get_annotated_filepath(value)).resolve()
    candidate = Path(os.path.abspath(os.path.expandvars(os.path.expanduser(value))))
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(value)


def _cache_main_video_from_file(filename: str, node_id: Any) -> bool:
    """Decode the connected source file when an editor requests a deferred cache."""
    if not filename or not node_id:
        return False
    try:
        path = _resolve_video_file(filename)
        with av.open(str(path), mode="r") as container:
            if not container.streams.video:
                raise ValueError("No video stream found.")
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.guessed_rate or Fraction(24, 1)
            safe_fps = float(Fraction(rate))
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
        if not frames:
            raise ValueError("Video contains no decodable frames.")
        array = np.stack(frames, axis=0).astype(np.uint8, copy=False)
        source_info = _source_info_for_node(node_id)
        array, output_fps = _adapt_source_frames(array, safe_fps, source_info)
        audio = None
        try:
            with av.open(str(path), mode="r") as container:
                if container.streams.audio:
                    audio_stream = container.streams.audio[0]
                    chunks = []
                    for frame in container.decode(audio_stream):
                        chunk = frame.to_ndarray()
                        if chunk.ndim == 1:
                            chunk = chunk[None, :]
                        chunks.append(chunk)
                    if chunks:
                        sample_rate = int(audio_stream.codec_context.sample_rate or 0)
                        if sample_rate > 0:
                            audio = {
                                "waveform": torch.from_numpy(np.concatenate(chunks, axis=1)).unsqueeze(0),
                                "sample_rate": sample_rate,
                            }
        except Exception as exc:
            _subtitle_info("source audio cache unavailable: %s", exc)
        info = {
            "frames": int(array.shape[0]),
            "width": int(array.shape[2]),
            "height": int(array.shape[1]),
            "fps": output_fps if np.isfinite(output_fps) and output_fps > 0 else 24.0,
            "duration": float(array.shape[0]) / (output_fps if np.isfinite(output_fps) and output_fps > 0 else 24.0),
        }
        return _store_frame_cache(
            array,
            float(info["fps"]),
            info,
            node_id,
            "preview",
            "preview-file",
            encode_video=False,
            audio=audio,
        )
    except Exception as exc:
        _subtitle_info("main video file cache unavailable: %s", exc)
        return False


def _cache_proxy_preview(
    proxy_video: Any,
    node_id: Any,
    render_size: tuple[int, int] | None = None,
) -> bool:
    return _cache_video_source(proxy_video, node_id, "preview", "preview", render_size)


def _cache_main_video(video: Any, node_id: Any) -> bool:
    return _cache_video_source(video, node_id, "main", "main", encode_video=False)


def _ensure_preview_video(entry: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    """Lazily encode a video stream when a recovered frame cache needs one."""
    if Path(str(entry.get("video_path") or entry.get("path") or "")).is_file():
        return entry
    progress = _SubtitleProgress(int(entry.get("info", {}).get("frames", 1) or 1), "preview video encoding")
    try:
        entry = _preview_cache_store().ensure_video(entry, progress=progress)
        if not entry:
            return None
        _subtitle_info("preview video rebuilt from main video frames")
        return entry
    except Exception as exc:
        _subtitle_info("preview video rebuild unavailable: %s", exc)
        return None
    finally:
        progress.close()


def _preview_cache_entry(node_id: str) -> dict[str, Any] | None:
    """Return a readable preview cache, rebuilding it from the main video cache when needed."""
    entry = _preview_cache_store().get_node(node_id, "preview")
    try:
        if entry:
            frames = np.load(str(entry["frames_path"]), mmap_mode="r", allow_pickle=False)
            if frames.ndim == 4 and frames.shape[0] > 0:
                return entry
    except (OSError, ValueError, KeyError):
        pass
    main_entry = _preview_cache_store().get_node(node_id, "main")
    try:
        if main_entry:
            frames = np.load(str(main_entry["frames_path"]), mmap_mode="r", allow_pickle=False)
            if frames.ndim == 4 and frames.shape[0] > 0:
                _subtitle_info("preview frame cache rebuilt from main video cache")
                return main_entry
    except (OSError, ValueError, KeyError):
        pass
    return None


def _preview_entry_for_request(node_id: str, video_filename: str = "") -> dict[str, Any] | None:
    entry = _preview_cache_entry(node_id)
    if entry is not None:
        return entry
    source_info = _source_info_for_node(node_id)
    candidates = []
    for candidate in (video_filename, source_info.get("source_filename", "")):
        value = str(candidate or "").strip()
        if value and value not in candidates:
            candidates.append(value)
    for candidate in candidates:
        if _cache_main_video_from_file(candidate, node_id):
            _subtitle_info("preview cache created on demand from source video: %s", candidate)
            return _preview_cache_entry(node_id)
    return None

class CSVideoSubtitle(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        fonts = _font_files()
        return io.Schema(
            node_id="CS_Video_Subtitle",
            display_name="CS Video Subtitle",
            category=_CATEGORY,
            essentials_category="Video Tools",
            search_aliases=["SRT", "subtitle timeline", "字幕轨道", "burn subtitles"],
            description="Burn an editable SRT subtitle track onto a standard ComfyUI VIDEO.",
            inputs=[
                io.Video.Input("video", tooltip="Connect any compatible VIDEO output."),
                io.Video.Input("proxy_video", optional=True, tooltip="Optional compatible proxy VIDEO output for Edit Timeline preview."),
                io.String.Input("srt", force_input=True, tooltip="Connect any STRING output containing valid SRT cues."),
                io.String.Input(
                    "edited_srt",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Persisted SRT text edited in Edit Timeline. When non-empty, it overrides the connected SRT.",
                ),
                io.Int.Input("preview_in", default=0, min=0, max=10000000, step=1, advanced=True),
                io.Int.Input("preview_out", default=-1, min=-1, max=10000000, step=1, advanced=True),
                io.Combo.Input("font", options=fonts or [""], default=fonts[0] if fonts else "", advanced=True),
                io.Int.Input("font_size", default=30, min=8, max=100, step=1, advanced=True),
                io.String.Input("primary_color", default="#FFFFFF", advanced=True),
                io.String.Input("secondary_color", default="#FF0000", advanced=True),
                io.Boolean.Input("gradient", default=False, advanced=True),
                io.Combo.Input("text_align", options=["left", "center", "right"], default="center", advanced=True),
                io.Boolean.Input("italic", default=False, advanced=True),
                io.Int.Input("letter_spacing", default=0, min=-10, max=50, step=1, advanced=True),
                io.Float.Input("position_x", default=0.5, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Float.Input("position_y", default=0.88, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Int.Input("outline_size", default=2, min=0, max=20, step=1, advanced=True),
                io.String.Input("outline_color", default="#000000", advanced=True),
                io.Int.Input("shadow_size", default=3, min=0, max=20, step=1, advanced=True),
                io.String.Input("shadow_color", default="#000000", advanced=True),
            ],
            outputs=[io.Video.Output("video"), io.String.Output("srt", display_name="SRT")],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        video: Any,
        proxy_video: Any = None,
        srt: Any = "",
        edited_srt: str = "",
        preview_in: int = 0,
        preview_out: int = -1,
        font: str = "",
        font_size: int = 30,
        primary_color: str = "#FFFFFF",
        secondary_color: str = "#FF0000",
        gradient: bool = False,
        text_align: str = "center",
        italic: bool = False,
        letter_spacing: int = 0,
        position_x: float = 0.5,
        position_y: float = 0.88,
        outline_size: int = 2,
        outline_color: str = "#000000",
        shadow_size: int = 3,
        shadow_color: str = "#000000",
    ) -> io.NodeOutput:
        if not hasattr(video, "get_components"):
            raise ValueError("video input is not a compatible VIDEO value.")
        _subtitle_info("start")
        _subtitle_info("stage 1/6: validating video and subtitle inputs")
        preview_in = _safe_int(preview_in, 0, 0, 10000000)
        preview_out = _safe_int(preview_out, -1, -1, 10000000)
        font_size = _safe_int(font_size, 30, 8, 100)
        outline_size = _safe_int(outline_size, 2, 0, 20)
        shadow_size = _safe_int(shadow_size, 3, 0, 20)
        letter_spacing = _safe_int(letter_spacing, 0, -10, 50)
        position_x = _safe_float(position_x, 0.5, 0.0, 1.0, 2)
        position_y = _safe_float(position_y, 0.88, 0.0, 1.0, 2)
        primary_color = _normalize_hex(primary_color, "#FFFFFF")
        secondary_color = _normalize_hex(secondary_color, "#FF0000")
        srt = _coerce_srt_input(srt)
        if not srt.strip():
            raise ValueError("Connect an external SRT source to the srt input.")
        edited_srt = str(edited_srt or "")
        raw_node_id = getattr(getattr(cls, "hidden", None), "unique_id", None)
        cache_key = str(raw_node_id or "").strip()
        node_id = cache_key or None
        source_hash = _srt_source_hash(srt)
        cached_srt = _SUBTITLE_SRT_CACHE.get(cache_key) if cache_key else None
        cached_edited_srt = (
            str(cached_srt.get("srt", ""))
            if cached_srt and (not cached_srt.get("source_hash") or cached_srt.get("source_hash") == source_hash)
            else ""
        )
        edited_srt = edited_srt.strip() or cached_edited_srt or str(srt)
        if cache_key:
            _SUBTITLE_SRT_CACHE[cache_key] = {"source_hash": source_hash, "srt": edited_srt, "node_id": cache_key}
        components = video.get_components()
        source_metadata = dict(components.metadata or {})
        source_filename = _video_source_filename(video, source_metadata)
        if source_filename and "source_filename" not in source_metadata:
            source_metadata["source_filename"] = source_filename
        active_trim = getattr(video, "get_active_trim_window", None)
        if callable(active_trim):
            try:
                trim_start, trim_duration = active_trim()
                if float(trim_start or 0) > 0:
                    source_metadata["active_start_seconds"] = float(trim_start)
                if float(trim_duration or 0) > 0:
                    source_metadata["active_duration_seconds"] = float(trim_duration)
            except (TypeError, ValueError, OverflowError):
                pass
        images = components.images
        if images.ndim != 4 or images.shape[0] == 0:
            raise ValueError("VIDEO contains no decodable frames.")
        frame_rate = float(components.frame_rate)
        if frame_rate <= 0:
            raise ValueError("VIDEO frame rate must be positive.")
        _subtitle_info(
            "input ready: frames=%d, size=%dx%d, fps=%.3f",
            int(images.shape[0]),
            int(images.shape[2]),
            int(images.shape[1]),
            frame_rate,
        )
        render_size = (int(images.shape[2]), int(images.shape[1]))
        if node_id:
            _SUBTITLE_PROXY_RENDER_SIZE[str(node_id)] = render_size
        if node_id:
            _clear_video_caches(node_id)
            _SUBTITLE_SOURCE_INFO[str(node_id)] = {
                "source_filename": source_filename,
                "metadata": {
                    key: source_metadata[key]
                    for key in (
                        "source_filename", "filename", "source_file", "video_file", "video_path", "path", "source",
                        "source_fps", "start_frame", "end_frame", "loaded_fps",
                        "loaded_width", "loaded_height", "active_start_seconds", "active_duration_seconds",
                    )
                    if key in source_metadata
                },
                "frames": int(images.shape[0]),
                "width": int(images.shape[2]),
                "height": int(images.shape[1]),
                "fps": frame_rate,
            }
        _subtitle_info(
            "stage 2/6: recording source info; preview cache deferred%s",
            f" ({source_filename})" if source_filename else " (source filename unavailable)",
        )
        cache_entry = _SUBTITLE_SRT_CACHE.get(cache_key, {}) if cache_key else {}
        cached_edited_srt = (
            str(cache_entry.get("srt", ""))
            if cache_entry and (not cache_entry.get("source_hash") or cache_entry.get("source_hash") == _srt_source_hash(srt))
            else ""
        )
        edited_srt = edited_srt.strip() or cached_edited_srt or str(srt)
        _subtitle_info("stage 3/6: parsing subtitle cues")
        cues = _coerce_cues(srt, edited_srt)
        source_offset = 0.0
        if source_metadata.get("source_fps"):
            source_offset = float(source_metadata.get("start_frame", 0) or 0) / float(source_metadata["source_fps"])
        if source_offset:
            cues = [
                {**cue, "start": max(0.0, float(cue["start"]) - source_offset), "end": max(0.0, float(cue["end"]) - source_offset)}
                for cue in cues
            ]
            cues = [cue for cue in cues if cue["end"] > cue["start"]]
        style = {
            "font": font,
            "font_size": font_size,
            "primary_color": primary_color,
            "secondary_color": secondary_color,
            "gradient": gradient,
            "text_align": text_align,
            "italic": italic,
            "letter_spacing": letter_spacing,
            "position_x": position_x,
            "position_y": position_y,
            "outline_size": outline_size,
            "outline_color": outline_color,
            "shadow_size": shadow_size,
            "shadow_color": shadow_color,
        }
        _subtitle_info("subtitle cues ready: count=%d", len(cues))
        _subtitle_info("stage 4/6: rendering subtitles onto %d frames", int(images.shape[0]))
        rendered = []
        progress = _SubtitleProgress(int(images.shape[0]))
        try:
            for index, frame in enumerate(images):
                time_seconds = index / frame_rate
                active = [cue for cue in cues if float(cue["start"]) <= time_seconds < float(cue["end"])]
                rendered.append(
                    _renderer_module().render_frame(frame, active, style, _fonts_root())
                    if active
                    else frame[..., :3].detach().cpu().float()
                )
                progress.update()
        finally:
            progress.close()
        _subtitle_info("frame rendering complete: %d frames", len(rendered))
        output_images = torch.stack(rendered, dim=0).clamp(0, 1)
        _subtitle_info("stage 5/6: assembling output video")
        metadata = source_metadata
        edited_srt = _cues_to_srt(cues)
        metadata["cinestyle_subtitles"] = {"cue_count": len(cues), "style": style}
        try:
            output_components = Types.VideoComponents(
                images=output_images,
                audio=components.audio,
                frame_rate=components.frame_rate,
                metadata=metadata,
            )
        except TypeError:
            output_components = Types.VideoComponents(
                images=output_images,
                audio=components.audio,
                frame_rate=components.frame_rate,
            )
        _subtitle_info("stage 6/6: complete, output frames=%d", int(output_images.shape[0]))
        return io.NodeOutput(InputImpl.VideoFromComponents(output_components), edited_srt)


async def _fonts_route(request):
    from aiohttp import web

    return web.json_response({"fonts": _font_files()})


async def _font_file_route(request):
    from aiohttp import web

    relative = unquote(request.match_info.get("font", ""))
    root = _fonts_root().resolve()
    try:
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise FileNotFoundError(relative)
    except (OSError, ValueError):
        return web.json_response({"error": "font not found"}, status=404)
    return web.FileResponse(path=path, headers={"Cache-Control": "public,max-age=86400"})


async def _subtitle_srt_cache_route(request):
    from aiohttp import web

    node_id = str(request.query.get("node_id", "")).strip()
    value = _SUBTITLE_SRT_CACHE.get(node_id)
    if value is None or value.get("node_id", node_id) != node_id:
        return web.json_response({"error": "No saved Timeline SRT for this node."}, status=404)
    return web.json_response(value)


async def _subtitle_srt_cache_update_route(request):
    from aiohttp import web

    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON payload."}, status=400)
    node_id = str(payload.get("node_id", "")).strip()
    source_hash = str(payload.get("source_hash", "")).strip()
    edited_srt = str(payload.get("srt", ""))
    current = _SUBTITLE_SRT_CACHE.get(node_id)
    if not node_id:
        return web.json_response({"error": "Missing subtitle node id."}, status=400)
    if not parse_srt(edited_srt):
        return web.json_response({"error": "Edited SRT contains no valid cues."}, status=400)
    if current and current.get("node_id") not in (None, node_id):
        return web.json_response({"error": "The saved SRT belongs to another subtitle node."}, status=409)
    if current and source_hash and current.get("source_hash") and source_hash != current.get("source_hash"):
        return web.json_response({"error": "The external SRT changed. Reload the current SRT before applying."}, status=409)
    if current is None:
        current = {"source_hash": source_hash, "srt": edited_srt, "node_id": node_id}
        _SUBTITLE_SRT_CACHE[node_id] = current
    else:
        current["source_hash"] = source_hash or current.get("source_hash", "")
        current["srt"] = edited_srt
    return web.json_response({"ok": True})


async def _subtitle_preview_route(request):
    from aiohttp import web

    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON payload."}, status=400)
    node_id = str(payload.get("node_id", "")).strip()
    if not node_id:
        return web.json_response({"error": "Missing subtitle node id."}, status=400)
    entry = _preview_entry_for_request(node_id, str(payload.get("video_filename", "")).strip())
    if not entry:
        return web.json_response({"error": "Source video file unavailable. Run CS Video Subtitle once after restoring the source video."}, status=404)
    try:
        frames = np.load(str(entry["frames_path"]), mmap_mode="r", allow_pickle=False)
        frame_index = max(0, min(int(payload.get("frame", 0)), int(frames.shape[0]) - 1))
        proxy_height, proxy_width = int(frames.shape[1]), int(frames.shape[2])
        fps = float(entry.get("info", {}).get("fps", 24.0) or 24.0)
        current_time = frame_index / max(0.001, fps)
        cues = payload.get("cues") if isinstance(payload.get("cues"), list) else []
        active = [
            cue for cue in cues
            if isinstance(cue, dict)
            and float(cue.get("start", 0.0)) <= current_time < float(cue.get("end", 0.0))
            and str(cue.get("text", "")).strip()
        ]
        style = dict(payload.get("style") or {})
        target_width, target_height = _SUBTITLE_PROXY_RENDER_SIZE.get(node_id, (proxy_width, proxy_height))
        scale = min(proxy_width / max(1, target_width), proxy_height / max(1, target_height))
        if abs(scale - 1.0) > 1e-6:
            for key in ("font_size", "outline_size", "shadow_size"):
                if key in style:
                    style[key] = float(style[key]) * scale
        renderer = _renderer_module()
        body, bounds = renderer.render_overlay_png_with_bounds(proxy_width, proxy_height, active, style, _fonts_root())
        headers = {"Cache-Control": "no-store"}
        if bounds:
            headers["X-CineStyle-Subtitle-Bounds"] = json.dumps(
                {"left": bounds[0], "top": bounds[1], "right": bounds[2], "bottom": bounds[3], "width": bounds[2] - bounds[0], "height": bounds[3] - bounds[1]},
                separators=(",", ":"),
            )
        return web.Response(body=body, content_type="image/png", headers=headers)
    except (OSError, ValueError, KeyError, IndexError) as exc:
        return web.json_response({"error": f"Unable to render subtitle preview: {exc}"}, status=500)


async def _subtitle_preview_info_route(request):
    from aiohttp import web

    node_id = str(request.query.get("node_id", "")).strip()
    entry = _preview_entry_for_request(node_id, str(request.query.get("video_filename", "")).strip())
    entry = _ensure_preview_video(entry, node_id) if entry else None
    video_path = Path(str(entry.get("video_path", ""))) if entry else None
    if not entry or video_path is None or not video_path.is_file():
        return web.json_response({"error": "Source video file unavailable. Run CS Video Subtitle once after restoring the source video."}, status=404)
    return web.json_response(
        {
            "video_url": f"/cinestyle/video-subtitle-preview-video?node_id={quote(node_id)}",
            "info": dict(entry.get("info") or {}),
            "label": "Subtitle preview cache generated on demand",
        }
    )


async def _subtitle_preview_video_route(request):
    from aiohttp import web

    node_id = str(request.query.get("node_id", "")).strip()
    entry = _preview_entry_for_request(node_id, str(request.query.get("video_filename", "")).strip())
    entry = _ensure_preview_video(entry, node_id) if entry else None
    video_path = Path(str(entry.get("video_path", ""))) if entry else None
    if not entry or video_path is None or not video_path.is_file():
        return web.json_response({"error": "Source video file unavailable. Subtitle preview cache cannot be generated."}, status=404)
    return web.FileResponse(
        path=video_path,
        headers={"Content-Type": "video/mp4", "Cache-Control": "no-store"},
    )


class CineStyleVideoSubtitleExtension(ComfyExtension):
    async def on_load(self) -> None:
        global _ROUTE_REGISTERED
        if _ROUTE_REGISTERED:
            return
        from server import PromptServer

        server_instance = getattr(PromptServer, "instance", None)
        if server_instance is not None:
            server_instance.routes.get("/cinestyle/fonts")(_fonts_route)
            server_instance.routes.get("/cinestyle/font/{font:.*}")(_font_file_route)
            server_instance.routes.get("/cinestyle/video-subtitle-srt-cache")(_subtitle_srt_cache_route)
            server_instance.routes.post("/cinestyle/video-subtitle-srt-cache")(_subtitle_srt_cache_update_route)
            server_instance.routes.get("/cinestyle/video-subtitle-preview-info")(_subtitle_preview_info_route)
            server_instance.routes.get("/cinestyle/video-subtitle-preview-video")(_subtitle_preview_video_route)
            server_instance.routes.post("/cinestyle/video-subtitle-preview")(_subtitle_preview_route)
            _ROUTE_REGISTERED = True

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CSVideoSubtitle]


async def comfy_entrypoint() -> CineStyleVideoSubtitleExtension:
    return CineStyleVideoSubtitleExtension()


WEB_DIRECTORY = "./web"
