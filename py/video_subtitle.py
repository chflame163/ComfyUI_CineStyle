"""Subtitle timeline and burn-in node for standard ComfyUI VIDEO values."""

from __future__ import annotations

import re
import sys
import json
import hashlib
import importlib.util
import logging
import threading
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
_SUBTITLE_SOURCE_METADATA: dict[str, dict[str, Any]] = {}
_SUBTITLE_LAZY_CACHE_LOCK = threading.RLock()
_SUBTITLE_TIMELINE_OPEN: set[str] = set()
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
    # Preserve loader metadata so a downstream timeline can distinguish a
    # trimmed VIDEO from the original source file when it builds a preview.
    metadata = getattr(components, "metadata", None)
    if isinstance(metadata, dict):
        for key in (
            "source_filename", "source_fps", "source_frame_count", "source_duration",
            "source_width", "source_height", "start_frame", "end_frame", "loaded_fps",
            "loaded_frame_count", "loaded_duration", "loaded_width", "loaded_height",
        ):
            if key in metadata:
                info[key] = metadata[key]
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
    entry = _preview_cache_store().put_preview(
        node_id,
        frames,
        safe_fps,
        proxy=str(variant or "").lower() in {"proxy", "preview"},
        encode_video=encode_video,
        info=info,
        audio=_prepare_audio(audio),
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


def _cache_proxy_preview(
    proxy_video: Any,
    node_id: Any,
    render_size: tuple[int, int] | None = None,
) -> bool:
    return _cache_video_source(proxy_video, node_id, "preview", "preview", render_size)


def _cache_main_video(video: Any, node_id: Any) -> bool:
    return _cache_video_source(video, node_id, "main", "main", encode_video=True)


def _preview_cache_entry(node_id: str) -> dict[str, Any] | None:
    """Return a readable preview cache, falling back to the node's main cache."""
    entry = _preview_cache_store().get_preview(node_id)
    try:
        if entry:
            frames = np.load(str(entry["frames_path"]), mmap_mode="r", allow_pickle=False)
            if frames.ndim == 4 and frames.shape[0] > 0:
                return entry
    except (OSError, ValueError, KeyError):
        pass
    main_entry = _preview_cache_store().get_preview_variant(node_id, proxy=False)
    try:
        if main_entry:
            frames = np.load(str(main_entry["frames_path"]), mmap_mode="r", allow_pickle=False)
            if frames.ndim == 4 and frames.shape[0] > 0:
                _subtitle_info("preview frame cache rebuilt from main video cache")
                return main_entry
    except (OSError, ValueError, KeyError):
        pass
    return None


def _preview_entry_for_request(
    node_id: str,
    video_filename: str = "",
    trim_metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    entry = _preview_cache_entry(node_id)
    if entry is None:
        return entry
    info = dict(entry.get("info") or {})
    if video_filename and info.get("source_filename") and str(info.get("source_filename")).strip() != str(video_filename).strip():
        return None
    if not trim_metadata:
        return entry
    requested_start = trim_metadata.get("start_frame")
    requested_end = trim_metadata.get("end_frame")
    requested_fps = trim_metadata.get("loaded_fps")
    if requested_start is not None:
        try:
            if int(info.get("start_frame")) != int(requested_start):
                return None
        except (TypeError, ValueError):
            return None
    if requested_end is not None:
        try:
            requested_end = int(requested_end)
        except (TypeError, ValueError):
            return None
    if requested_end is not None:
        try:
            expected_end = requested_end
            if requested_end < 0:
                source_count = int(info.get("source_frame_count") or 0)
                expected_end = source_count - 1 if source_count > 0 else None
            if expected_end is not None and int(info.get("end_frame")) != expected_end:
                return None
        except (TypeError, ValueError):
            return None
    if requested_fps is not None:
        try:
            requested_fps = float(requested_fps)
            if requested_fps > 0 and abs(float(info.get("loaded_fps")) - requested_fps) > 1e-4:
                return None
        except (TypeError, ValueError):
            return None
    return entry


def _subtitle_proxy_dimensions(width: int, height: int, target_pixels: int = 800_000) -> tuple[int, int]:
    width = max(2, int(width))
    height = max(2, int(height))
    if width * height <= target_pixels:
        return width - width % 2, height - height % 2
    scale = (target_pixels / float(width * height)) ** 0.5
    proxy_width = max(2, int(width * scale) // 2 * 2)
    proxy_height = max(2, int(height * scale) // 2 * 2)
    return proxy_width, proxy_height


def _audio_from_video_file(source_path: str, start_seconds: float, duration: float) -> dict[str, Any] | None:
    """Decode only the audio samples that belong to a lazy preview range."""
    try:
        start_seconds = max(0.0, float(start_seconds))
        duration = max(0.0, float(duration))
    except (TypeError, ValueError, OverflowError):
        return None
    if duration <= 0.0:
        return None
    try:
        with av.open(source_path, mode="r") as container:
            if not container.streams.audio:
                return None
            stream = container.streams.audio[0]
            sample_rate = int(stream.codec_context.sample_rate or stream.rate or 0)
            if sample_rate <= 0:
                return None
            start_sample = int(round(start_seconds * sample_rate))
            end_sample = max(start_sample + 1, int(round((start_seconds + duration) * sample_rate)))
            cursor = 0
            chunks: list[np.ndarray] = []
            for audio_frame in container.decode(stream):
                try:
                    samples = audio_frame.to_ndarray(format="fltp")
                except Exception:
                    samples = audio_frame.to_ndarray()
                samples = np.asarray(samples, dtype=np.float32)
                if samples.ndim == 1:
                    samples = samples[None, :]
                elif samples.ndim > 2:
                    samples = samples.reshape(samples.shape[0], -1)
                if samples.ndim != 2 or samples.shape[1] <= 0:
                    continue
                frame_start = cursor
                frame_end = cursor + int(samples.shape[1])
                cursor = frame_end
                if frame_end <= start_sample:
                    continue
                if frame_start >= end_sample:
                    break
                left = max(0, start_sample - frame_start)
                right = min(samples.shape[1], end_sample - frame_start)
                if right > left:
                    chunks.append(np.ascontiguousarray(samples[:, left:right]))
            if not chunks:
                return None
            waveform = torch.from_numpy(np.ascontiguousarray(np.concatenate(chunks, axis=1))).unsqueeze(0)
            return {"waveform": waveform, "sample_rate": sample_rate}
    except (OSError, ValueError, av.error.FFmpegError):
        return None


def _lazy_cache_trimmed_preview(
    node_id: str,
    video_filename: str,
    requested_metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build a lightweight trimmed preview when execution skipped cache generation."""
    key = str(node_id or "").strip()
    source = str(video_filename or "").strip()
    if not key or not source:
        return None
    with _SUBTITLE_LAZY_CACHE_LOCK:
        existing = _preview_entry_for_request(key, source, requested_metadata)
        if existing is not None:
            existing_info = dict(existing.get("info") or {})
            if existing.get("audio") is not None or "has_audio" in existing_info:
                return existing
        metadata = dict(_SUBTITLE_SOURCE_METADATA.get(key) or {})
        if metadata.get("source_filename") and str(metadata.get("source_filename")).strip() != source:
            metadata = {}
        if requested_metadata:
            metadata.update({name: value for name, value in requested_metadata.items() if value is not None})
        try:
            source_path = _preview_cache_store()._resolve_file(source)
            source_fps = float(metadata.get("source_fps") or 0.0)
            source_count = int(metadata.get("source_frame_count") or 0)
            source_width = int(metadata.get("source_width") or 0)
            source_height = int(metadata.get("source_height") or 0)
            if source_fps <= 0 or source_count <= 0 or source_width <= 0 or source_height <= 0:
                with av.open(source_path, mode="r") as container:
                    if not container.streams.video:
                        return None
                    stream = container.streams.video[0]
                    rate = stream.average_rate or stream.guessed_rate
                    source_fps = source_fps if source_fps > 0 else float(rate or 24.0)
                    source_width = source_width or int(stream.width or 0)
                    source_height = source_height or int(stream.height or 0)
                    source_count = source_count or int(stream.frames or 0)
                    if source_count <= 0 and container.duration and source_fps > 0:
                        source_count = max(1, int(round(float(container.duration / av.time_base) * source_fps)))
            if source_fps <= 0 or source_count <= 0 or source_width <= 0 or source_height <= 0:
                return None
            target_fps = float(metadata.get("loaded_fps") or source_fps)
            start = max(0, min(int(metadata.get("start_frame") or 0), source_count - 1))
            requested_end = metadata.get("end_frame")
            end = source_count - 1 if requested_end is None or int(requested_end) < 0 else min(int(requested_end), source_count - 1)
            end = max(start, end)
            loaded_width = int(metadata.get("loaded_width") or source_width)
            loaded_height = int(metadata.get("loaded_height") or source_height)
            if loaded_width <= 0 or loaded_height <= 0 or source_fps <= 0 or target_fps <= 0:
                return None
            proxy_width, proxy_height = _subtitle_proxy_dimensions(loaded_width, loaded_height)
            output_count = max(1, int(round((end - start + 1) * target_fps / source_fps)))
            requested = np.rint(np.linspace(start, end, output_count)).astype(np.int64)
            frames: list[np.ndarray] = []
            requested_index = 0
            with av.open(source_path, mode="r") as container:
                if not container.streams.video:
                    return None
                stream = container.streams.video[0]
                for source_index, decoded in enumerate(container.decode(stream)):
                    if requested_index >= len(requested):
                        break
                    if source_index < int(requested[requested_index]):
                        continue
                    while requested_index < len(requested) and int(requested[requested_index]) == source_index:
                        frames.append(decoded.reformat(width=proxy_width, height=proxy_height, format="rgb24").to_ndarray())
                        requested_index += 1
            if not frames:
                return None
            preview_audio = _audio_from_video_file(
                source_path,
                start / source_fps,
                len(frames) / target_fps,
            )
            info = {
                "source_filename": source,
                "source_fps": source_fps,
                "source_frame_count": source_count,
                "source_width": source_width,
                "source_height": source_height,
                "start_frame": start,
                "end_frame": end,
                "loaded_fps": target_fps,
                "loaded_frame_count": len(frames),
                "loaded_width": loaded_width,
                "loaded_height": loaded_height,
                "has_audio": bool(preview_audio),
            }
            _SUBTITLE_PROXY_RENDER_SIZE[key] = (loaded_width, loaded_height)
            return _preview_cache_store().put_preview(
                key,
                np.stack(frames, axis=0),
                target_fps,
                proxy=True,
                encode_video=True,
                info=info,
                audio=preview_audio,
            )
        except (OSError, ValueError, KeyError, IndexError, av.error.FFmpegError) as exc:
            _subtitle_info("lazy trimmed preview unavailable: %s", exc)
            return None


def _downsample_waveform_peaks(values: Any, target: int = 1600) -> list[float]:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return []
    array = np.nan_to_num(np.abs(array), nan=0.0, posinf=1.0, neginf=0.0)
    target = max(1, min(int(target), int(array.size)))
    edges = np.linspace(0, int(array.size), target + 1, dtype=np.int64)
    peaks = []
    for index in range(target):
        start, end = int(edges[index]), int(edges[index + 1])
        peaks.append(float(np.max(array[start:end])) if end > start else 0.0)
    peak_max = max(peaks, default=0.0)
    if peak_max > 0.0:
        peaks = [peak / peak_max for peak in peaks]
    return [max(0.0, min(1.0, peak)) for peak in peaks]


def _waveform_from_video_file(source: str) -> tuple[list[float], float]:
    if not source:
        return [], 0.0
    try:
        source_path = _preview_cache_store()._resolve_file(source)
        with av.open(source_path, mode="r") as container:
            if not container.streams.audio:
                return [], 0.0
            stream = container.streams.audio[0]
            sample_rate = int(stream.codec_context.sample_rate or stream.rate or 0)
            if sample_rate <= 0:
                return [], 0.0
            frame_peaks: list[float] = []
            sample_count = 0
            for audio_frame in container.decode(stream):
                try:
                    samples = audio_frame.to_ndarray(format="fltp")
                except Exception:
                    samples = audio_frame.to_ndarray()
                samples = np.asarray(samples, dtype=np.float32)
                if samples.ndim == 1:
                    samples = samples[None, :]
                elif samples.ndim > 2:
                    samples = samples.reshape(samples.shape[0], -1)
                if samples.ndim != 2 or samples.shape[1] == 0:
                    continue
                mono = samples.mean(axis=0)
                frame_peaks.append(float(np.max(np.abs(mono))))
                sample_count += int(mono.size)
            duration = float(sample_count) / sample_rate if sample_count else 0.0
            return _downsample_waveform_peaks(frame_peaks), duration
    except Exception as exc:
        _subtitle_info("audio waveform unavailable: %s", exc)
        return [], 0.0

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
                io.Int.Input("font_size", default=30, min=8, max=200, step=1, advanced=True),
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
        font_size = _safe_int(font_size, 30, 8, 200)
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
            _SUBTITLE_SOURCE_METADATA[str(node_id)] = dict(source_metadata)
        if node_id:
            _clear_video_caches(node_id)
        if node_id and str(node_id) in _SUBTITLE_TIMELINE_OPEN:
            _subtitle_info("stage 2/6: preparing preview caches")
            main_cached = _cache_main_video(video, node_id)
            if proxy_video is None:
                if main_cached:
                    _subtitle_info("preview uses the main video cache")
            else:
                cached_preview = _cache_proxy_preview(proxy_video, node_id, render_size=render_size)
                if not cached_preview and main_cached:
                    _subtitle_info("proxy cache failed; preview uses the main video cache")
        else:
            _subtitle_info("stage 2/6: preview cache generation skipped (Edit Timeline is closed)")
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


async def _subtitle_timeline_state_route(request):
    from aiohttp import web

    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON payload."}, status=400)
    node_id = str(payload.get("node_id", "")).strip()
    if not node_id:
        return web.json_response({"error": "Missing subtitle node id."}, status=400)
    if bool(payload.get("open")):
        _SUBTITLE_TIMELINE_OPEN.add(node_id)
    else:
        _SUBTITLE_TIMELINE_OPEN.discard(node_id)
    return web.json_response({"ok": True})


def _trim_metadata_from_values(values) -> dict[str, Any]:
    """Read optional upstream CS Load Video range hints from request values."""
    metadata: dict[str, Any] = {}
    for name in ("start_frame", "end_frame"):
        value = values.get(name)
        if value is None or str(value).strip() == "":
            continue
        try:
            metadata[name] = int(float(value))
        except (TypeError, ValueError, OverflowError):
            continue
    value = values.get("loaded_fps")
    if value is not None and str(value).strip() != "":
        try:
            loaded_fps = float(value)
            if np.isfinite(loaded_fps) and loaded_fps > 0:
                metadata["loaded_fps"] = loaded_fps
        except (TypeError, ValueError, OverflowError):
            pass
    return metadata


def _trim_metadata_from_request(request) -> dict[str, Any]:
    return _trim_metadata_from_values(request.query)


async def _subtitle_preview_route(request):
    from aiohttp import web

    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON payload."}, status=400)
    node_id = str(payload.get("node_id", "")).strip()
    if not node_id:
        return web.json_response({"error": "Missing subtitle node id."}, status=400)
    video_filename = str(payload.get("video_filename", "")).strip()
    trim_metadata = _trim_metadata_from_values(payload)
    entry = _preview_entry_for_request(node_id, video_filename, trim_metadata)
    if entry is None and video_filename and trim_metadata:
        entry = _lazy_cache_trimmed_preview(node_id, video_filename, trim_metadata)
    try:
        frame_index = max(0, int(payload.get("frame", 0)))
        source_frame_index = max(0, int(payload.get("source_frame", frame_index)))
        if entry:
            frames = np.load(str(entry["frames_path"]), mmap_mode="r", allow_pickle=False)
            frame_index = min(frame_index, int(frames.shape[0]) - 1)
            proxy_height, proxy_width = int(frames.shape[1]), int(frames.shape[2])
            fps = float(entry.get("info", {}).get("fps", 24.0) or 24.0)
        else:
            proxy_width = max(0, int(payload.get("preview_width", 0) or 0))
            proxy_height = max(0, int(payload.get("preview_height", 0) or 0))
            fps = float(payload.get("preview_fps", 0) or 0)
            if proxy_width <= 0 or proxy_height <= 0:
                source = str(payload.get("video_filename", "")).strip()
                if not source:
                    return web.json_response({"error": "Subtitle preview cache is unavailable. Run CS Video Subtitle once or provide a source video."}, status=404)
                frame = _preview_cache_store().decode_frame({"video": source, "source_kind": "video"}, source_frame_index)
                proxy_height, proxy_width = int(frame.shape[1]), int(frame.shape[2])
                if fps <= 0:
                    fps = 24.0
                    try:
                        source_path = _preview_cache_store()._resolve_file(source)
                        with av.open(source_path, mode="r") as source_container:
                            source_rate = source_container.streams.video[0].average_rate or source_container.streams.video[0].guessed_rate
                            if source_rate:
                                fps = float(source_rate)
                    except Exception:
                        pass
            if fps <= 0:
                fps = 24.0
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


async def _subtitle_waveform_route(request):
    import asyncio
    from aiohttp import web

    node_id = str(request.query.get("node_id", "")).strip()
    filename = str(request.query.get("video_filename", "")).strip()
    trim_metadata = _trim_metadata_from_request(request)
    entry = _preview_entry_for_request(node_id, filename, trim_metadata) if node_id else None
    if entry is None and node_id and filename and (trim_metadata or node_id in _SUBTITLE_SOURCE_METADATA):
        entry = _lazy_cache_trimmed_preview(node_id, filename, trim_metadata)
    # Decode the exact media file used by the <video> element. This keeps the
    # waveform and playback on one source, including codec/container timing.
    preview_path = Path(str(entry.get("video_path") or entry.get("path") or "")) if entry else None
    if preview_path is not None and preview_path.is_file():
        peaks, duration = await asyncio.to_thread(_waveform_from_video_file, str(preview_path))
    elif filename:
        peaks, duration = await asyncio.to_thread(_waveform_from_video_file, filename)
    else:
        peaks, duration = [], 0.0
    if entry:
        duration = float((entry.get("info") or {}).get("duration", duration) or duration)
    return web.json_response(
        {"peaks": peaks, "duration": max(0.0, duration), "has_audio": bool(peaks)},
        headers={"Cache-Control": "no-store"},
    )


async def _subtitle_preview_info_route(request):
    from aiohttp import web

    node_id = str(request.query.get("node_id", "")).strip()
    filename = str(request.query.get("video_filename", "")).strip()
    trim_metadata = _trim_metadata_from_request(request)
    entry = _preview_entry_for_request(node_id, filename, trim_metadata)
    if entry is None and (trim_metadata or node_id in _SUBTITLE_SOURCE_METADATA):
        entry = _lazy_cache_trimmed_preview(node_id, filename, trim_metadata)
    video_path = Path(str(entry.get("video_path", ""))) if entry else None
    if not entry or video_path is None or not video_path.is_file():
        return web.json_response({"error": "Subtitle preview cache is unavailable. Run CS Video Subtitle once to build it."}, status=404)
    return web.json_response(
        {
            "video_url": f"/cinestyle/video-subtitle-preview-video?node_id={quote(node_id)}",
            "info": dict(entry.get("info") or {}),
            "label": "Subtitle preview cache from the last workflow run",
        }
    )


async def _subtitle_preview_video_route(request):
    from aiohttp import web

    node_id = str(request.query.get("node_id", "")).strip()
    entry = _preview_entry_for_request(node_id)
    video_path = Path(str(entry.get("video_path", ""))) if entry else None
    if not entry or video_path is None or not video_path.is_file():
        return web.json_response({"error": "Subtitle preview cache is unavailable."}, status=404)
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
            server_instance.routes.post("/cinestyle/video-subtitle-timeline-state")(_subtitle_timeline_state_route)
            server_instance.routes.get("/cinestyle/video-subtitle-preview-info")(_subtitle_preview_info_route)
            server_instance.routes.get("/cinestyle/video-subtitle-preview-video")(_subtitle_preview_video_route)
            server_instance.routes.get("/cinestyle/video-subtitle-waveform")(_subtitle_waveform_route)
            server_instance.routes.post("/cinestyle/video-subtitle-preview")(_subtitle_preview_route)
            _ROUTE_REGISTERED = True

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CSVideoSubtitle]


async def comfy_entrypoint() -> CineStyleVideoSubtitleExtension:
    return CineStyleVideoSubtitleExtension()


WEB_DIRECTORY = "./web"
