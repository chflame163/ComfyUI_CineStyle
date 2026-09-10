"""Two-track timeline editing for CineStyle.

The node in this module deliberately keeps the editable timeline as JSON.  A
timeline is a list of source ranges placed on a two-track video/audio canvas;
the same descriptor is used by the browser preview and by the final render.
The renderer is intentionally dependency-light (PyTorch + NumPy) so it can be
used with any standard ComfyUI ``VIDEO`` value, including values which do not
have an originating file name.

The browser side of the node can use the small HTTP helpers at the end of this
file.  They are optional: rendering a node does not require a running HTTP
server, and cache failures never change the final output.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import threading
import time
import tempfile
import uuid
import shlex
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - ComfyUI normally provides tqdm
    tqdm = None

try:  # ``av`` is already a CineStyle dependency, but keep import optional for tests.
    import av
except Exception:  # pragma: no cover - unit-test environments may omit PyAV
    av = None

try:
    from aiohttp import web
except Exception:  # pragma: no cover - importing the node must remain cheap
    web = None

import folder_paths
from comfy_api.latest import ComfyExtension, InputImpl, Types, io


_LOGGER = logging.getLogger("CineStyleVideoTimelineEdit")
_CATEGORY = "😺dzNodes/CineStyle/Video"
_NODE_ID = "CS_Video_Timeline_Edit"
_SCHEMA_VERSION = 1
_TRANSFORM_VERSION = 1
_RENDER_CACHE_VERSION = 2
_SOURCE_FINGERPRINT_VERSION = 2
_DEFAULT_FPS = 24.0
_DEFAULT_AUDIO_RATE = 48000
_DEFAULT_AUDIO_CHANNELS = 2
_ANSI_GREEN = "\033[32m"
_ANSI_RESET = "\033[0m"
_SCHEMA_MAX_DIMENSION = 1_048_576
_SCHEMA_MAX_MULTIPLE = 65_536
_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".avif"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv"}

_ROUTE_REGISTERED = False
_CACHE_STORE: Any = None
_STATE_LOCK = threading.RLock()
_TIMELINE_STATES: dict[str, dict[str, Any]] = {}
_STATE_REVISIONS: dict[str, int] = {}
_HISTORY_LOCK = threading.RLock()
_TIMELINE_HISTORIES: dict[str, dict[str, Any]] = {}
_PREVIEW_WARNINGS: dict[str, str] = {}
_PROXY_JOBS: dict[str, dict[str, Any]] = {}
_PROXY_JOBS_LOCK = threading.RLock()
_SHOT_CACHE: dict[str, list[dict[str, int]]] = {}
_SHOT_CACHE_LOCK = threading.RLock()
_PROXY_JOB_TTL_SECONDS = 3600.0


def _timeline_info(message: str, *args: Any) -> None:
    """Keep CS Video Timeline status lines aligned with CS Load Video."""
    _LOGGER.info("[CS Video Timeline Edit] " + message, *args)


class _TimelineProgress:
    """Emit a throttled tqdm-style progress bar for frame rendering."""

    def __init__(self, total: int, description: str = "rendering frames"):
        self.bar = None
        if tqdm is not None:
            self.bar = tqdm(
                total=max(1, int(total)),
                desc=f"{_ANSI_GREEN}[INFO]{_ANSI_RESET} [CS Video Timeline Edit] {description}",
                unit="frame",
                bar_format=(
                    "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}]"
                ),
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


# ---------------------------------------------------------------------------
# Small, deterministic coercion helpers


def _safe_int(value: Any, default: int = 0, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        if isinstance(value, bool):
            result = int(value)
        elif isinstance(value, str) and not value.strip():
            result = int(default)
        else:
            result = int(float(value))
    except (TypeError, ValueError, OverflowError):
        result = int(default)
    if minimum is not None:
        result = max(int(minimum), result)
    if maximum is not None:
        result = min(int(maximum), result)
    return result


def _safe_float(value: Any, default: float = 0.0, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        result = float(default)
    if minimum is not None:
        result = max(float(minimum), result)
    if maximum is not None:
        result = min(float(maximum), result)
    return result


def _coerce_fps(value: Any, default: float = 24.0) -> float:
    """Read a CFR rate from floats, Fractions, or ``"num/den"`` strings."""
    try:
        if isinstance(value, str) and "/" in value:
            value = float(Fraction(value.strip()))
        result = float(value)
        if not math.isfinite(result) or result <= 0:
            raise ValueError
        return result
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return float(default)


def _ceil_multiple(value: Any, multiple: Any) -> int:
    """Round a dimension *up* to the selected positive integer multiple."""
    amount = max(1, _safe_int(value, 1, 1))
    step = max(1, _safe_int(multiple, 1, 1))
    return max(step, int(math.ceil(amount / step) * step))


def _round_half_up(value: Any) -> int:
    """Round a finite non-negative dimension like JavaScript ``Math.round``."""
    try:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        return 0
    return int(math.floor(number + 0.5))


def _normalise_hex(value: Any, default: str = "#000000") -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"#[0-9a-fA-F]{3}", text):
        text = "#" + "".join(ch * 2 for ch in text[1:])
    if not _HEX_RE.fullmatch(text):
        return default.upper()
    return text.upper()


def _hex_rgb(value: Any) -> tuple[float, float, float]:
    text = _normalise_hex(value)
    return tuple(int(text[index : index + 2], 16) / 255.0 for index in (1, 3, 5))  # type: ignore[return-value]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _parse_json(value: Any, name: str = "timeline_json") -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and not value.strip():
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON: {exc.msg}.") from exc
    if decoded is None:
        return {}
    if not isinstance(decoded, Mapping):
        raise ValueError(f"{name} must contain a JSON object.")
    return dict(decoded)


def _first(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _normalise_track(value: Any, default: int = 0) -> int:
    if isinstance(value, str):
        text = value.strip().lower().replace("_", "-")
        if text in {"upper", "top", "video-upper", "audio-upper", "1"} or "upper" in text or "top" in text:
            return 1
        if text in {"lower", "bottom", "video-lower", "audio-lower", "0"} or "lower" in text or "bottom" in text:
            return 0
    try:
        return 1 if int(float(value)) > 0 else 0
    except (TypeError, ValueError, OverflowError):
        return 1 if int(default) > 0 else 0


def _track_index(value: Any, default: int = 0, *, allow_none: bool = False) -> int:
    """Parse a video/audio track label while preserving an explicit ``none``."""
    if allow_none and isinstance(value, str):
        text = value.strip().lower().replace("_", "-")
        if text in {"none", "mute", "muted", "audio", "audio-only", "-1"}:
            return -1
    try:
        if allow_none and int(float(value)) < 0:
            return -1
    except (TypeError, ValueError, OverflowError):
        pass
    return _normalise_track(value, default)


def _normalise_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "on", "1"}:
            return True
        if text in {"false", "no", "off", "0", ""}:
            return False
    return bool(value)


def _video_metadata(video: Any) -> dict[str, Any]:
    """Collect runtime/component metadata without relying on VIDEO internals."""
    metadata: dict[str, Any] = {}
    runtime = getattr(video, "_cinestyle_runtime_metadata", None)
    if isinstance(runtime, Mapping):
        metadata.update(runtime)
    try:
        components = video.get_components() if video is not None and hasattr(video, "get_components") else None
        component_metadata = getattr(components, "metadata", None) if components is not None else None
        if isinstance(component_metadata, Mapping):
            # Component metadata is the canonical value when both layers are
            # present; this also mirrors ComfyUI's VideoFromComponents model.
            metadata.update(component_metadata)
    except Exception:
        pass
    # The current ComfyUI ``VideoFromComponents.get_components()`` wrapper
    # intentionally omits metadata.  Recover the original dataclass metadata
    # when it is still attached privately; this keeps source identity and
    # loader window information available to downstream nodes without relying
    # on a ComfyUI implementation detail being public API.
    if video is not None:
        try:
            for value in vars(video).values():
                candidate = getattr(value, "metadata", None)
                if isinstance(candidate, Mapping):
                    # Fill fields omitted by the public wrapper (notably
                    # loader window/VFR flags) without overriding metadata
                    # already exposed by the VIDEO/components API.
                    for key, item in candidate.items():
                        metadata.setdefault(str(key), item)
        except (TypeError, AttributeError):
            pass
    return metadata


def _file_source_fingerprint(metadata: Mapping[str, Any]) -> str:
    """Recreate the loader-preview file fingerprint when a filename is known."""
    source_filename = str(metadata.get("source_filename") or metadata.get("filename") or "").strip()
    annotated = re.match(r"^(.*)\s+\[(?:input|output|temp)\]$", source_filename, flags=re.IGNORECASE)
    if annotated:
        source_filename = annotated.group(1).strip()
    if not source_filename:
        return ""
    try:
        if folder_paths.exists_annotated_filepath(source_filename):
            path = Path(folder_paths.get_annotated_filepath(source_filename)).resolve()
        else:
            path = Path(os.path.expandvars(os.path.expanduser(source_filename))).resolve()
        if not path.is_file():
            return ""
        stat = path.stat()
        payload = f"{path}|{stat.st_size}|{stat.st_mtime_ns}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()
    except (OSError, TypeError, ValueError):
        return ""


def _sampled_content_fingerprint(images: torch.Tensor | None) -> str:
    """Compute a bounded, metadata-independent content key for a VIDEO batch."""
    if images is None:
        return ""
    try:
        if not isinstance(images, torch.Tensor):
            images = torch.as_tensor(images)
        # Hash a bounded set of frames for speed while retaining shape and
        # dtype identity.  Converting a 4K/long VIDEO tensor wholesale to CPU
        # here would briefly duplicate hundreds of megabytes solely for a key.
        source = images.detach()
        shape = tuple(int(v) for v in source.shape)
        digest = hashlib.sha256()
        digest.update(str(shape).encode("ascii"))
        digest.update(str(source.dtype).encode("ascii"))
        if source.ndim >= 1 and shape[0] > 0:
            # Small/medium clips are cheap enough to hash completely, which
            # makes source-change detection exact for the normal editor use
            # case.  Long clips remain bounded (32 spread-out samples).
            total_bytes = int(source.numel()) * max(1, int(source.element_size()))
            sample_count = (
                shape[0]
                if shape[0] <= 64 or total_bytes <= 64 * 1024 * 1024
                else min(32, shape[0])
            )
            indices = sorted({int(round(i * (shape[0] - 1) / max(1, sample_count - 1))) for i in range(sample_count)})
            for index in indices:
                frame = source[index].to(device="cpu", dtype=torch.float32).contiguous()
                raw = memoryview(frame.numpy()).cast("B")
                stride = max(1, len(raw) // (256 * 1024))
                digest.update(str(index).encode("ascii"))
                digest.update(raw[::stride])
        return digest.hexdigest()
    except Exception:
        return ""


def _source_fingerprint(
    video: Any,
    images: torch.Tensor | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Return a stable source key for preview caches and persisted state.

    Explicit upstream identities are returned verbatim.  The previous
    implementation *prefixed* an existing fingerprint with sampled pixels,
    which made a Timeline output's fingerprint change every time it was fed
    into another Timeline node.  Keeping the identity and sampled fallback
    namespaces separate makes chained nodes and persisted timelines stable.
    """
    metadata = dict(metadata) if isinstance(metadata, Mapping) else _video_metadata(video)
    # A Timeline output is a derived VIDEO.  Its inherited original filename
    # and source fingerprint identify provenance, but they must not be used as
    # the cache key for a downstream Timeline node whose actual pixels may
    # have been moved, cropped, or transformed.
    if metadata.get("timeline_version") is not None or metadata.get("timeline_json"):
        derived = _sampled_content_fingerprint(images)
        if derived:
            return derived
    file_identity = _file_source_fingerprint(metadata)
    if file_identity:
        # A filename is stronger than an inherited upstream label: chained
        # nodes often carry the original ``source_fingerprint`` forward, and
        # the file stat lets us still notice that the underlying source was
        # replaced.
        return file_identity
    explicit = str(
        metadata.get("source_fingerprint")
        or metadata.get("input_signature")
        or metadata.get("loader_signature")
        or ""
    ).strip()
    # For a generic VIDEO, the pixels delivered to this node are the actual
    # source.  Prefer their content key over an inherited upstream label:
    # intermediate nodes may preserve the original filename/fingerprint even
    # after changing frames (resize, trim, grade, or timeline composition).
    # File-backed identities were handled above and remain authoritative.
    content = _sampled_content_fingerprint(images)
    if content:
        return content
    explicit_version = _safe_int(metadata.get("source_fingerprint_version", 0), 0, 0)
    if explicit and explicit_version >= _SOURCE_FINGERPRINT_VERSION:
        return explicit
    return explicit


def _source_identity_hint(video: Any, metadata: Mapping[str, Any] | None = None) -> str:
    """Return an upstream-provided identity suitable for UI migration checks."""
    metadata = dict(metadata) if isinstance(metadata, Mapping) else _video_metadata(video)
    file_identity = _file_source_fingerprint(metadata)
    if file_identity:
        return file_identity
    for key in ("source_identity", "source_fingerprint", "input_signature", "loader_signature"):
        if metadata.get(key):
            return str(metadata[key])
    # No filename or explicit identity is available.  The caller can still
    # use the sampled content key as a cache key, but it is not safe to compare
    # it against a browser-provided file identity for migration purposes.
    return ""


# ---------------------------------------------------------------------------
# Timeline normalisation


def _transform_descriptor(value: Any) -> dict[str, Any]:
    """Normalise the permissive browser transform shape to a stable schema."""
    source: Mapping[str, Any]
    if isinstance(value, Mapping):
        source = value
    else:
        source = {}

    translation = _first(source, "translation", "translate", "position", default=None)
    tx: Any = _first(source, "translate_x", "translation_x", "x", "offset_x", default=None)
    ty: Any = _first(source, "translate_y", "translation_y", "y", "offset_y", default=None)
    if isinstance(translation, Mapping):
        tx = _first(translation, "x", "translate_x", default=tx)
        ty = _first(translation, "y", "translate_y", default=ty)
    elif isinstance(translation, Sequence) and not isinstance(translation, (str, bytes)) and len(translation) >= 2:
        tx = translation[0] if tx is None else tx
        ty = translation[1] if ty is None else ty
    tx = _safe_float(tx, 0.0)
    ty = _safe_float(ty, 0.0)
    unit = str(_first(source, "translation_unit", "position_unit", default="auto") or "auto").lower()
    # ``x``/``y`` in the timeline UI are normally normalised to canvas units;
    # explicit pixel fields remain unambiguous.
    if unit not in {"normalized", "normalised", "pixel", "pixels"}:
        unit = "normalized" if (abs(tx) <= 1.0 and abs(ty) <= 1.0) else "pixel"
    if unit in {"normalised"}:
        unit = "normalized"
    if unit == "pixels":
        unit = "pixel"

    scale_value = _first(source, "scale", "zoom", default=1.0)
    if isinstance(scale_value, Sequence) and not isinstance(scale_value, (str, bytes)):
        sx = _safe_float(scale_value[0] if len(scale_value) else 1.0, 1.0)
        sy = _safe_float(scale_value[1] if len(scale_value) > 1 else sx, sx)
    elif isinstance(scale_value, Mapping):
        sx = _safe_float(_first(scale_value, "x", "scale_x", default=1.0), 1.0)
        sy = _safe_float(_first(scale_value, "y", "scale_y", default=sx), sx)
    else:
        sx = _safe_float(_first(source, "scale_x", default=scale_value), 1.0)
        sy = _safe_float(_first(source, "scale_y", default=scale_value), sx)
    # A few timeline clients send percentages (100 = 100%).
    if abs(sx) > 20.0:
        sx /= 100.0
    if abs(sy) > 20.0:
        sy /= 100.0
    sx = abs(sx) if abs(sx) > 1e-6 else 1.0
    sy = abs(sy) if abs(sy) > 1e-6 else 1.0
    sx = max(0.1, min(4.0, sx))
    sy = max(0.1, min(4.0, sy))

    rotation = max(-90.0, min(90.0, _safe_float(_first(source, "rotation", "rotate", "angle", "rotation_degrees", default=0.0), 0.0)))
    flip_x = _normalise_bool(_first(source, "flip_x", "flipX", "flip_h", "flipH", "mirror_x", "mirrorX", "mirror_horizontal", "mirrorHorizontal", "horizontal_flip", default=False))
    flip_y = _normalise_bool(_first(source, "flip_y", "flipY", "flip_v", "flipV", "mirror_y", "mirrorY", "mirror_vertical", "mirrorVertical", "vertical_flip", default=False))
    # ``mirror`` may be a direction or a pair of booleans.
    mirror = _first(source, "mirror", default=None)
    if isinstance(mirror, str):
        mirror_text = mirror.lower().strip().replace("_", "-")
        flip_x = flip_x or mirror_text in {"x", "h", "horizontal", "left-right", "leftright", "both", "xy"}
        flip_y = flip_y or mirror_text in {"y", "v", "vertical", "up-down", "updown", "both", "xy"}
    elif isinstance(mirror, Sequence) and not isinstance(mirror, (str, bytes)):
        if len(mirror) > 0:
            flip_x = flip_x or _normalise_bool(mirror[0])
        if len(mirror) > 1:
            flip_y = flip_y or _normalise_bool(mirror[1])

    return {
        "version": _TRANSFORM_VERSION,
        "scale_x": float(sx),
        "scale_y": float(sy),
        "rotation": float(rotation),
        "translate_x": float(tx),
        "translate_y": float(ty),
        "translation_unit": unit,
        "flip_x": bool(flip_x),
        "flip_y": bool(flip_y),
    }


def _clip_from_raw(raw: Any, index: int, source_frames: int, fps: float, default_audio: bool = True) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    source_start_value = _first(raw, "source_start", "sourceStart", "source_in", "sourceIn", "in_frame", "inFrame", "start_frame", default=0)
    source_start = _safe_int(source_start_value, 0, 0)
    source_end_value = _first(raw, "source_end", "sourceEnd", "source_out", "sourceOut", "out_frame", "outFrame", "end_frame", default=None)
    duration_value = _first(raw, "duration_frames", "durationFrames", "frame_count", "frames", default=None)
    if source_end_value is None:
        if duration_value is None:
            source_end = source_frames
        else:
            source_end = source_start + max(0, _safe_int(duration_value, 0))
    else:
        source_end = _safe_int(source_end_value, source_frames)
    source_start = min(max(0, source_start), max(0, source_frames))
    # A client may explicitly mark an inclusive source end.  Internally all
    # ranges are half-open, so convert it before clipping.
    if _normalise_bool(_first(raw, "source_end_inclusive", "sourceEndInclusive", "inclusive", default=False)):
        source_end += 1
    source_end = min(max(source_start, source_end), max(0, source_frames))
    if source_end <= source_start:
        return None

    timeline_start = _safe_int(
        _first(raw, "timeline_start", "timelineStart", "start", "start_frame_timeline", "at_frame", "atFrame", default=0),
        0,
        0,
    )
    timeline_end_value = _first(raw, "timeline_end", "timelineEnd", "end", "end_frame_timeline", "timelineEndFrame", default=None)
    timeline_duration = _first(raw, "timeline_duration_frames", "timelineDurationFrames", "length_frames", "length", default=None)
    source_length = source_end - source_start
    if timeline_end_value is None:
        timeline_length = source_length if timeline_duration is None else min(source_length, max(0, _safe_int(timeline_duration, source_length)))
        timeline_end = timeline_start + timeline_length
    else:
        timeline_end = timeline_start + max(0, _safe_int(timeline_end_value, timeline_start) - timeline_start)
        if _normalise_bool(_first(raw, "timeline_end_inclusive", "timelineEndInclusive", default=False)):
            timeline_end += 1
        # A clip cannot be stretched beyond its source range.  Trim the right
        # edge while retaining its requested timeline start.
        timeline_end = min(timeline_end, timeline_start + source_length)
    if timeline_end <= timeline_start:
        return None

    video_track_value = _first(raw, "video_track", "videoTrack", "track", "track_index", "video_layer", default=0)
    explicit_video_track = False
    try:
        explicit_video_track = int(float(video_track_value)) < 0
    except (TypeError, ValueError, OverflowError):
        explicit_video_track = str(video_track_value or "").strip().lower() in {"none", "audio", "audio-only", "-1"}
    audio_track_value = _first(raw, "audio_track", "audioTrack", "audio_layer", default=0 if default_audio else -1)
    audio_enabled = not _normalise_bool(_first(raw, "mute", "muted", "audio_disabled", default=False))
    if _first(raw, "audio_enabled", "audioEnabled", default=None) is not None:
        audio_enabled = _normalise_bool(_first(raw, "audio_enabled", "audioEnabled"), default_audio)
    try:
        audio_track = -1 if int(float(audio_track_value)) < 0 else _normalise_track(audio_track_value)
    except (TypeError, ValueError, OverflowError):
        text = str(audio_track_value or "").lower()
        audio_track = -1 if text in {"none", "mute", "muted", "-1"} else _normalise_track(audio_track_value)
    if not audio_enabled:
        audio_track = -1

    clip_id = str(_first(raw, "id", "clip_id", "clipId", default=f"clip-{index + 1}") or f"clip-{index + 1}")
    linked = _normalise_bool(_first(raw, "linked", "av_linked", "link_audio", "linkAudio", default=True), True)
    # A clip explicitly placed on the audio-only collection has no picture to
    # follow, so never mark it as an A/V-linked item in the canonical schema.
    if explicit_video_track:
        linked = False
    return {
        "id": clip_id,
        "source_start": int(source_start),
        "source_end": int(source_end),
        "timeline_start": int(timeline_start),
        "timeline_end": int(timeline_end),
        "video_track": -1 if explicit_video_track else _normalise_track(video_track_value),
        "audio_track": int(audio_track),
        "linked": bool(linked),
        "transform": _transform_descriptor(_first(raw, "transform", "transforms", default=raw)),
        "enabled": (
            _normalise_bool(raw.get("enabled"), True)
            if "enabled" in raw
            else not _normalise_bool(raw.get("disabled"), False)
        ),
        "order": _safe_int(_first(raw, "order", "z", "placement_order", default=index), index),
    }


def _default_clip(source_frames: int) -> dict[str, Any]:
    return {
        "id": "clip-1",
        "source_start": 0,
        "source_end": int(max(0, source_frames)),
        "timeline_start": 0,
        "timeline_end": int(max(0, source_frames)),
        "video_track": 0,
        "audio_track": 0,
        "linked": True,
        "transform": _transform_descriptor({}),
        "enabled": True,
        "order": 0,
    }


def normalise_timeline(value: Any, source_frames: int, fps: float, *, default_if_empty: bool = True) -> dict[str, Any]:
    """Return a canonical, frame-based timeline descriptor.

    Both snake_case and the camelCase names emitted by the JavaScript timeline
    are accepted.  All ranges are half-open ``[start, end)`` intervals.
    """
    source_frames = max(0, _safe_int(source_frames, 0, 0))
    fps = _coerce_fps(fps, _DEFAULT_FPS)
    raw = _parse_json(value)
    # Presence of an audio-only collection is also an explicit timeline.  In
    # that shape the caller intentionally supplies no video clips; inserting
    # the default lower-track source video would unexpectedly reintroduce it.
    has_explicit_clips = any(
        key in raw
        for key in ("clips", "video_clips", "videoClips", "audio_clips", "audioClips")
    )
    clips_raw = raw.get("clips")
    if clips_raw is None:
        # Accept the convenient per-track shape used by early UI prototypes.
        clips_raw = []
        video_aliases = ("video_clips", "videoClips") if has_explicit_clips else ("video_clips", "videoClips", "items", "segments")
        for key in video_aliases:
            candidate = raw.get(key)
            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
                clips_raw.extend(candidate)
                # When both naming conventions are present, the canonical
                # snake_case collection wins instead of duplicating every clip.
                if has_explicit_clips:
                    break
            elif has_explicit_clips and key in raw:
                break
        if not has_explicit_clips:
            for track_key, track_value in (("lower", raw.get("lower_track")), ("upper", raw.get("upper_track"))):
                if isinstance(track_value, Sequence) and not isinstance(track_value, (str, bytes)):
                    for item in track_value:
                        if isinstance(item, Mapping):
                            clips_raw.append({**item, "video_track": 1 if track_key == "upper" else 0})
    if isinstance(clips_raw, Mapping):
        clips_raw = list(clips_raw.values())
    if not isinstance(clips_raw, Sequence) or isinstance(clips_raw, (str, bytes)):
        clips_raw = []

    clips: list[dict[str, Any]] = []
    for index, item in enumerate(clips_raw):
        clip = _clip_from_raw(item, index, source_frames, fps, default_audio=True)
        if clip is not None:
            clips.append(clip)
    # An omitted ``clips`` key means the input is placed on the lower tracks by
    # default.  An explicit ``clips: []`` is a deliberate empty timeline and
    # must remain empty (the UI uses this while clearing a sequence).
    if not clips and default_if_empty and source_frames > 0 and not has_explicit_clips:
        clips = [_default_clip(source_frames)]

    # Separate audio-only clips are useful when the UI unlinks A/V.  They use
    # the same source range fields but have no video contribution.
    audio_raw = raw.get("audio_clips", raw.get("audioClips", []))
    audio_clips: list[dict[str, Any]] = []
    if isinstance(audio_raw, Mapping):
        audio_raw = list(audio_raw.values())
    if isinstance(audio_raw, Sequence) and not isinstance(audio_raw, (str, bytes)):
        for index, item in enumerate(audio_raw):
            clip = _clip_from_raw(item, index, source_frames, fps, default_audio=True)
            if clip is not None:
                clip["video_track"] = -1
                clip["linked"] = False
                audio_clips.append(clip)

    # The timeline extent is defined by the furthest end of every serialized
    # clip. Disabled clips contribute no pixels/audio, but they still retain
    # their edit-space extent (useful for an explicitly cleared timeline
    # sentinel). A serialized duration is only a compatibility hint when the
    # collection has no clips at all.
    clip_ends = [
        int(c["timeline_end"])
        for c in clips + audio_clips
    ]
    explicit_duration = _first(raw, "duration_frames", "durationFrames", "timeline_duration_frames", "timelineDurationFrames", default=None)
    if clip_ends:
        duration = max(clip_ends)
    elif explicit_duration is not None:
        duration = max(0, _safe_int(explicit_duration, 0, 0))
    else:
        duration = 0

    in_frame = _safe_int(_first(raw, "in_frame", "inFrame", "preview_in", "previewIn", "in_point", "inPoint", default=0), 0, 0)
    out_raw = _first(raw, "out_frame", "outFrame", "preview_out", "previewOut", "out_point", "outPoint", default=-1)
    out_frame = -1 if _safe_int(out_raw, -1) < 0 else _safe_int(out_raw, -1, 0)
    # A standard VIDEO cannot represent an empty In/Out interval.  Keep In
    # on the last valid frame whenever a non-empty timeline is available.
    in_frame = min(in_frame, max(0, duration - 1)) if duration > 0 else 0
    if out_frame < 0:
        out_frame = duration
    else:
        out_frame = min(max(in_frame, out_frame), duration)
    if duration > 0 and out_frame <= in_frame:
        # A malformed range should still render one frame rather than produce
        # a tensor with an invalid shape.
        out_frame = min(duration, in_frame + 1)

    threshold = _safe_float(_first(raw, "threshold", "shot_detect_threshold", "Threshold", default=0.5), 0.5, 0.0, 1.0)
    min_scene_seconds = _safe_float(_first(raw, "min_scene_seconds", "shot_detect_min_scene_sec", "MinSceneSeconds", "minSceneSeconds", default=0.0), 0.0, 0.0)
    raw_output = raw.get("output") if isinstance(raw.get("output"), Mapping) else {}
    output_fit = str(
        _first(
            raw_output,
            "fit_mode",
            "fitMode",
            "output_fit_mode",
            "outputFitMode",
            default=_first(raw, "fit_mode", "fitMode", "output_fit_mode", "outputFitMode", default="letterbox"),
        )
        or "letterbox"
    ).strip().lower()
    if output_fit not in {"letterbox", "crop", "fill"}:
        output_fit = "letterbox"
    output_width_raw = _first(raw_output, "width", "output_width", "outputWidth", default=None)
    if output_width_raw is None or _safe_int(output_width_raw, 0, 0) <= 0:
        output_width_raw = _first(raw, "width", "output_width", "outputWidth", default=0)
    output_height_raw = _first(raw_output, "height", "output_height", "outputHeight", default=None)
    if output_height_raw is None or _safe_int(output_height_raw, 0, 0) <= 0:
        output_height_raw = _first(raw, "height", "output_height", "outputHeight", default=0)
    output_multiple_raw = _first(raw_output, "multiple", "output_multiple", "outputMultiple", default=None)
    if output_multiple_raw is None or _safe_int(output_multiple_raw, 0, 0) <= 0:
        output_multiple_raw = _first(raw, "multiple", "output_multiple", "outputMultiple", default=32)
    output_fill_raw = _first(raw_output, "fill_color", "fillColor", "output_fill_color", "outputFillColor", default=None)
    if output_fill_raw is None or not str(output_fill_raw).strip():
        output_fill_raw = _first(raw, "fill_color", "fillColor", "output_fill_color", "outputFillColor", default="#000000")
    output = {
        "width": -1 if _safe_int(output_width_raw, -1) <= 0 else _safe_int(output_width_raw, -1, 1),
        "height": -1 if _safe_int(output_height_raw, -1) <= 0 else _safe_int(output_height_raw, -1, 1),
        "multiple": max(1, _safe_int(output_multiple_raw, 32, 1)),
        "fit_mode": output_fit,
        "fill_color": _normalise_hex(output_fill_raw),
    }
    result = {
        "version": _SCHEMA_VERSION,
        "transform_version": _TRANSFORM_VERSION,
        "fps": float(fps),
        "duration_frames": int(duration),
        "in_frame": int(in_frame),
        "out_frame": int(out_frame),
        "clips": clips,
        "audio_clips": audio_clips,
        "threshold": float(threshold),
        "min_scene_seconds": float(min_scene_seconds),
        "output": output,
    }
    # Preserve an optional source identity without allowing arbitrary nested
    # objects to make cache keys unstable.
    source_id = _first(raw, "source_fingerprint", "sourceFingerprint", "input_signature", default="")
    if source_id:
        result["source_fingerprint"] = str(source_id)
    source_fingerprint_kind = str(_first(raw, "source_fingerprint_kind", "sourceFingerprintKind", default="") or "").strip().lower()
    if source_fingerprint_kind in {"file", "content", "metadata", "unknown"}:
        result["source_fingerprint_kind"] = source_fingerprint_kind
    source_fingerprint_version = _first(raw, "source_fingerprint_version", "sourceFingerprintVersion", default=None)
    if source_fingerprint_version is not None:
        result["source_fingerprint_version"] = _safe_int(source_fingerprint_version, 1, 1)
    source_identity = _first(raw, "source_identity", "sourceIdentity", default="")
    if source_identity:
        result["source_identity"] = str(source_identity)
    source_frame_count = _first(raw, "source_frame_count", "sourceFrameCount", default=None)
    if source_frame_count is not None:
        result["source_frame_count"] = max(0, _safe_int(source_frame_count, source_frames, 0))
    source_fps = _first(raw, "source_fps", "sourceFps", default=None)
    if source_fps is not None:
        result["source_fps"] = _safe_float(source_fps, fps, 0.001)
    source_start_frame = _first(raw, "source_start_frame", "sourceStartFrame", default=None)
    if source_start_frame is not None:
        result["source_start_frame"] = max(0, _safe_int(source_start_frame, 0, 0))
    source_end_frame = _first(raw, "source_end_frame", "sourceEndFrame", default=None)
    if source_end_frame is not None:
        result["source_end_frame"] = max(0, _safe_int(source_end_frame, source_frames - 1, 0))
    return result


def canonical_timeline_json(value: Any, source_frames: int, fps: float) -> str:
    return _canonical_json(normalise_timeline(value, source_frames, fps))


# US-spelling aliases are kept for integrations and small third-party tests.
normalize_timeline = normalise_timeline


# ---------------------------------------------------------------------------
# Frame and affine rendering


def _fit_dimensions(source_w: int, source_h: int, width: Any, height: Any, multiple: Any) -> tuple[int, int]:
    sw, sh = max(1, _safe_int(source_w, 1, 1)), max(1, _safe_int(source_h, 1, 1))
    # ``-1`` means use the source side.  ``0`` remains accepted as a legacy
    # alias for old workflow JSON and is treated identically.
    requested_w, requested_h = _safe_int(width, -1), _safe_int(height, -1)
    requested_w = requested_w if requested_w > 0 else -1
    requested_h = requested_h if requested_h > 0 else -1
    if requested_w < 0 and requested_h < 0:
        target_w, target_h = sw, sh
    elif requested_w > 0 and requested_h < 0:
        target_w = requested_w
        target_h = max(1, _round_half_up(target_w * sh / sw))
    elif requested_h > 0 and requested_w < 0:
        target_h = requested_h
        target_w = max(1, _round_half_up(target_h * sw / sh))
    else:
        target_w, target_h = requested_w, requested_h
    return _ceil_multiple(target_w, multiple), _ceil_multiple(target_h, multiple)


def _fill_tensor(device: torch.device, dtype: torch.dtype, height: int, width: int, color: tuple[float, float, float]) -> torch.Tensor:
    # Materialise the expansion: letterbox fitting writes the resized image
    # into this canvas and an expanded zero-stride view is not writable.
    return torch.tensor(color, device=device, dtype=dtype).view(1, 3, 1, 1).expand(1, 3, height, width).clone()


def _resize_chw(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if image.shape[-2:] == (height, width):
        return image
    return F.interpolate(image, size=(height, width), mode="bilinear", align_corners=False)


def _normalise_image_tensor(images: torch.Tensor) -> torch.Tensor:
    # ComfyUI IMAGE/VIDEO tensors are normally float32 in [0, 1].  Avoid an
    # ``amax`` over an entire 4K/long source here: that reduction synchronises
    # a GPU and can briefly dominate an otherwise low-resolution preview.  A
    # small sample is enough to recognise the uint8-style float batches that
    # standalone callers occasionally provide.
    value = images.float()
    try:
        if not torch.is_floating_point(images):
            value = value / 255.0
        elif value.numel():
            flat = value.detach().reshape(-1)
            stride = max(1, int(flat.numel() // 4096))
            sample_max = float(flat[::stride].amax().item())
            if sample_max > 1.5:
                value = value / 255.0
    except Exception:
        # Keep the standard [0,1] interpretation when a device/backend does
        # not support the inexpensive sampling path.
        pass
    return value.clamp(0.0, 1.0)


def _fit_frame_rgba(
    image_hwc: torch.Tensor,
    width: int,
    height: int,
    fit_mode: str,
    color: tuple[float, float, float],
    *,
    normalized: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit one frame and return RGB plus an occupancy mask.

    Letterbox padding is deliberate clip content and is therefore opaque in
    the mask: the configured fill colour must remain visible as the clip's
    letterbox bars even when that clip is placed on the upper video track.
    Pixels outside a transformed clip's bounds are still transparent after
    the affine step, so moving/scaling/rotating an upper clip can reveal the
    lower track around the transformed canvas.
    """
    width = max(1, _safe_int(width, 1, 1))
    height = max(1, _safe_int(height, 1, 1))
    frame = image_hwc[..., :3].float().clamp(0.0, 1.0) if normalized else _normalise_image_tensor(image_hwc[..., :3])
    chw = frame.permute(2, 0, 1).unsqueeze(0)
    src_h, src_w = int(chw.shape[-2]), int(chw.shape[-1])
    mode = str(fit_mode or "letterbox").strip().lower()
    if mode not in {"letterbox", "crop", "fill"}:
        mode = "letterbox"
    if mode == "fill":
        return _resize_chw(chw, height, width), torch.ones((1, 1, height, width), device=chw.device, dtype=chw.dtype)
    if mode == "letterbox":
        scale = min(width / max(1, src_w), height / max(1, src_h))
    else:  # crop: cover then center crop
        scale = max(width / max(1, src_w), height / max(1, src_h))
    fit_w = max(1, _round_half_up(src_w * scale))
    fit_h = max(1, _round_half_up(src_h * scale))
    resized = _resize_chw(chw, fit_h, fit_w)
    if mode == "letterbox":
        # Rounding a fitted side can overshoot the canvas by one pixel.  Clamp
        # before slicing so unusual aspect ratios never trigger a shape error.
        if fit_w > width or fit_h > height:
            clamp_scale = min(width / max(1, src_w), height / max(1, src_h))
            fit_w = max(1, min(width, _round_half_up(src_w * clamp_scale)))
            fit_h = max(1, min(height, _round_half_up(src_h * clamp_scale)))
            resized = _resize_chw(chw, fit_h, fit_w)
        canvas = _fill_tensor(chw.device, resized.dtype, height, width, color)
        # The bars are part of the fitted clip canvas (rule: fill colour is
        # used for letterbox bars), so they intentionally cover lower-track
        # content when this clip is composited above it.
        alpha = torch.ones((1, 1, height, width), device=chw.device, dtype=resized.dtype)
        left = max(0, (width - fit_w) // 2)
        top = max(0, (height - fit_h) // 2)
        canvas[..., top : top + fit_h, left : left + fit_w] = resized
        return canvas, alpha
    left = max(0, min(fit_w - width, (fit_w - width) // 2))
    top = max(0, min(fit_h - height, (fit_h - height) // 2))
    return resized[..., top : top + height, left : left + width], torch.ones((1, 1, height, width), device=chw.device, dtype=resized.dtype)


def _fit_frame(image_hwc: torch.Tensor, width: int, height: int, fit_mode: str, color: tuple[float, float, float]) -> torch.Tensor:
    """Fit one RGB HWC frame to an opaque output canvas."""
    return _fit_frame_rgba(image_hwc, width, height, fit_mode, color)[0]


def _affine_frame_rgba(
    base: torch.Tensor,
    transform: Mapping[str, Any],
    base_alpha: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one affine transform and return RGB plus an occupancy/alpha mask.

    VIDEO values are RGB, but retaining an internal mask is important for the
    two-track compositor: a scaled/moved upper-track clip should reveal the
    lower track outside its transformed bounds instead of painting the whole
    canvas with the fill colour.  ``base_alpha`` carries the fitted clip
    occupancy (including opaque letterbox bars).  The mask is discarded when
    the final RGB batch is emitted.
    """
    if base.ndim != 4 or base.shape[1] < 3:
        alpha = base_alpha if base_alpha is not None else torch.ones((base.shape[0], 1, base.shape[-2], base.shape[-1]), device=base.device, dtype=base.dtype)
        return base, alpha
    height, width = int(base.shape[-2]), int(base.shape[-1])
    t = _transform_descriptor(transform)
    sx = float(t.get("scale_x", 1.0))
    sy = float(t.get("scale_y", 1.0))
    angle = math.radians(float(t.get("rotation", 0.0)))
    tx = float(t.get("translate_x", 0.0))
    ty = float(t.get("translate_y", 0.0))
    if str(t.get("translation_unit", "pixel")) == "normalized":
        tx *= width
        ty *= height
    if t.get("flip_x"):
        sx = -sx
    if t.get("flip_y"):
        sy = -sy
    if abs(sx - 1.0) < 1e-7 and abs(sy - 1.0) < 1e-7 and abs(angle) < 1e-7 and abs(tx) < 1e-7 and abs(ty) < 1e-7:
        alpha = base_alpha if base_alpha is not None else torch.ones((base.shape[0], 1, height, width), device=base.device, dtype=base.dtype)
        return base, alpha

    # Forward pixel transform: center -> scale/mirror -> rotation -> offset.
    c, s = math.cos(angle), math.sin(angle)
    cx, cy = (width - 1) * 0.5, (height - 1) * 0.5
    forward = torch.tensor(
        [
            [c * sx, -s * sy, cx + tx - c * sx * cx + s * sy * cy],
            [s * sx, c * sy, cy + ty - s * sx * cx - c * sy * cy],
            [0.0, 0.0, 1.0],
        ],
        device=base.device,
        dtype=torch.float32,
    )
    try:
        inverse = torch.linalg.inv(forward)
    except RuntimeError:
        alpha = base_alpha if base_alpha is not None else torch.ones((base.shape[0], 1, height, width), device=base.device, dtype=base.dtype)
        return base, alpha
    yy, xx = torch.meshgrid(
        torch.arange(height, device=base.device, dtype=torch.float32),
        torch.arange(width, device=base.device, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(xx)
    coords = torch.stack((xx, yy, ones), dim=0).reshape(3, -1)
    source = inverse @ coords
    src_x = source[0].reshape(height, width)
    src_y = source[1].reshape(height, width)
    # align_corners=False maps pixel centres as (p + .5) / size.
    grid = torch.stack(((src_x + 0.5) * 2.0 / width - 1.0, (src_y + 0.5) * 2.0 / height - 1.0), dim=-1).unsqueeze(0)
    occupancy = torch.ones((1, 1, height, width), device=base.device, dtype=base.dtype)
    occupancy = F.grid_sample(occupancy, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if base_alpha is None:
        sampled = F.grid_sample(base, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        alpha = occupancy
    else:
        # Interpolate premultiplied RGB so transformed occupancy edges remain
        # stable when a clip is composited over another track.
        sampled_alpha = F.grid_sample(base_alpha, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled_premultiplied = F.grid_sample(base * base_alpha, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled = (sampled_premultiplied / sampled_alpha.clamp_min(1e-6)).clamp(0.0, 1.0)
        alpha = sampled_alpha * occupancy
    # Avoid tiny negative/over-one values at interpolation boundaries.
    return sampled, alpha.clamp(0.0, 1.0)


def _affine_frame(base: torch.Tensor, transform: Mapping[str, Any], color: tuple[float, float, float]) -> torch.Tensor:
    """Apply an affine transform and flatten transparent bounds to *color*.

    This compatibility helper retains the original RGB-only contract for
    callers outside the compositor.  ``render_timeline_frames`` uses the
    mask-aware variant above so lower-track pixels remain visible around a
    transformed upper clip.
    """
    sampled, alpha = _affine_frame_rgba(base, transform)
    fill = _fill_tensor(base.device, sampled.dtype, int(base.shape[-2]), int(base.shape[-1]), color)
    return sampled * alpha + fill * (1.0 - alpha)


def _render_clip_frame(source_images: torch.Tensor, clip: Mapping[str, Any], timeline_frame: int, width: int, height: int, fit_mode: str, color: tuple[float, float, float]) -> torch.Tensor | None:
    bounded_clip = _clip_with_source_bound(clip, int(source_images.shape[0]))
    if bounded_clip is None or not _normalise_bool(bounded_clip.get("enabled", True), True):
        return None
    clip = bounded_clip
    start, end = _safe_int(clip.get("timeline_start", 0), 0), _safe_int(clip.get("timeline_end", 0), 0)
    if timeline_frame < start or timeline_frame >= end:
        return None
    source_index = _safe_int(clip.get("source_start", 0), 0) + (timeline_frame - start)
    if source_index < 0 or source_index >= int(source_images.shape[0]):
        return None
    base = _fit_frame(source_images[source_index], width, height, fit_mode, color)
    transformed = _affine_frame(base, clip.get("transform") if isinstance(clip.get("transform"), Mapping) else {}, color)
    return transformed[0].permute(1, 2, 0).clamp(0.0, 1.0)


def _render_clip_rgba(
    source_images: torch.Tensor,
    clip: Mapping[str, Any],
    timeline_frame: int,
    width: int,
    height: int,
    fit_mode: str,
    color: tuple[float, float, float],
    *,
    normalized: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Render one active clip as HWC RGB plus a HxW occupancy mask."""
    bounded_clip = _clip_with_source_bound(clip, int(source_images.shape[0]))
    if bounded_clip is None or not _normalise_bool(bounded_clip.get("enabled", True), True):
        return None
    clip = bounded_clip
    start, end = _safe_int(clip.get("timeline_start", 0), 0), _safe_int(clip.get("timeline_end", 0), 0)
    if timeline_frame < start or timeline_frame >= end:
        return None
    source_index = _safe_int(clip.get("source_start", 0), 0) + (timeline_frame - start)
    if source_index < 0 or source_index >= int(source_images.shape[0]):
        return None
    base, base_alpha = _fit_frame_rgba(source_images[source_index], width, height, fit_mode, color, normalized=normalized)
    transformed, alpha = _affine_frame_rgba(
        base,
        clip.get("transform") if isinstance(clip.get("transform"), Mapping) else {},
        base_alpha,
    )
    return transformed[0].permute(1, 2, 0).clamp(0.0, 1.0), alpha[0, 0].clamp(0.0, 1.0)


def _active_clip(clips: Sequence[Mapping[str, Any]], frame: int, track: int) -> Mapping[str, Any] | None:
    candidates = [
        (index, clip)
        for index, clip in enumerate(clips)
        if _normalise_bool(clip.get("enabled", True), True)
        # Hand-authored descriptors may omit ``video_track``; canonical
        # normalisation treats that as the default lower track.
        and _track_index(clip.get("video_track", 0), 0, allow_none=True) == int(track)
        and _safe_int(clip.get("timeline_start", 0), 0) <= frame < _safe_int(clip.get("timeline_end", 0), 0)
    ]
    if not candidates:
        return None
    # A clip's ``order`` is its placement sequence.  It deliberately comes
    # before the timeline start: a clip placed earlier may start later in time
    # but must still lose to a clip placed afterwards when their occupied
    # ranges overlap.  The list position makes hand-authored descriptors with
    # duplicate/missing order values deterministic.
    _, selected = max(
        candidates,
        key=lambda item: (
            _safe_int(item[1].get("order", 0), 0),
            int(item[0]),
            _safe_int(item[1].get("timeline_start", 0), 0),
        ),
    )
    return selected


def _active_clips(clips: Sequence[Mapping[str, Any]], frame: int, track: int) -> list[Mapping[str, Any]]:
    """Return all active clips on a track in compositing order.

    Later placement order is composited over earlier placement order. Keeping
    the full stack (rather than selecting one winner up front) means a moved,
    scaled, or rotated later clip can reveal the earlier clip around its
    transformed bounds while still fully covering it when opaque.
    """
    candidates = [
        (index, clip)
        for index, clip in enumerate(clips)
        if _normalise_bool(clip.get("enabled", True), True)
        and _track_index(clip.get("video_track", 0), 0, allow_none=True) == int(track)
        and _safe_int(clip.get("timeline_start", 0), 0) <= frame < _safe_int(clip.get("timeline_end", 0), 0)
    ]
    candidates.sort(
        key=lambda item: (
            _safe_int(item[1].get("order", 0), 0),
            int(item[0]),
            _safe_int(item[1].get("timeline_start", 0), 0),
        )
    )
    return [clip for _, clip in candidates]


def _clip_with_source_bound(clip: Mapping[str, Any], source_frames: int) -> dict[str, Any] | None:
    """Clamp a clip's usable timeline extent to the available source frames.

    Canonical descriptors are already bounded by ``_clip_from_raw``.  The
    renderer also accepts hand-authored mappings directly, so enforce the same
    no-extension rule at the final sampling boundary instead of emitting a
    trailing gap for an invalid overlong range.
    """
    try:
        source_count = max(0, int(source_frames))
        source_start = _safe_int(clip.get("source_start", 0), 0, 0, source_count)
        timeline_start = _safe_int(clip.get("timeline_start", 0), 0, 0)
        timeline_end = _safe_int(clip.get("timeline_end", timeline_start), timeline_start, timeline_start)
        declared_source_end = min(
            source_count,
            max(source_start, _safe_int(clip.get("source_end", source_count), source_count)),
        )
        bounded_end = min(
            timeline_end,
            timeline_start + max(0, declared_source_end - source_start),
        )
        if bounded_end <= timeline_start:
            return None
        result = dict(clip)
        result["source_start"] = source_start
        result["source_end"] = declared_source_end
        result["timeline_start"] = timeline_start
        result["timeline_end"] = bounded_end
        return result
    except (TypeError, ValueError, OverflowError):
        return None


def _timeline_collection(value: Any) -> list[Any]:
    """Return a timeline collection as a list for tolerant API callers."""
    if isinstance(value, Mapping):
        return list(value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _normalise_render_collection(
    value: Any,
    source_frames: int,
    fps: float,
    *,
    audio_only: bool = False,
) -> list[dict[str, Any]]:
    """Canonicalise a collection before the low-level render helpers use it."""
    result: list[dict[str, Any]] = []
    for index, item in enumerate(_timeline_collection(value)):
        clip = _clip_from_raw(item, index, source_frames, fps, default_audio=True)
        if clip is None:
            continue
        if audio_only:
            clip["video_track"] = -1
            clip["linked"] = False
        result.append(clip)
    return result


def render_timeline_frames(
    source_images: torch.Tensor,
    timeline: Mapping[str, Any],
    width: int,
    height: int,
    fit_mode: str = "letterbox",
    fill_color: str = "#000000",
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
    progress: Any = None,
) -> torch.Tensor:
    """Render a timeline to an ``[frames,height,width,3]`` float tensor."""
    if not isinstance(timeline, Mapping):
        timeline = {}
    width = max(1, _safe_int(width, 1, 1))
    height = max(1, _safe_int(height, 1, 1))
    if not isinstance(source_images, torch.Tensor):
        source_images = torch.as_tensor(source_images)
    if source_images.ndim == 3:
        source_images = source_images.unsqueeze(0)
    if source_images.ndim != 4 or source_images.shape[0] <= 0 or source_images.shape[1] <= 0 or source_images.shape[2] <= 0 or source_images.shape[-1] < 3:
        raise ValueError("VIDEO frames must have shape [frames, height, width, 3 or 4].")
    source_images = _normalise_image_tensor(source_images[..., :3]).detach()
    video_collection = timeline.get("clips")
    if video_collection is None:
        video_collection = timeline.get(
            "video_clips",
            timeline.get("videoClips", timeline.get("items", timeline.get("segments", []))),
        )
    if not _timeline_collection(video_collection) and not any(
        key in timeline for key in ("clips", "video_clips", "videoClips", "items", "segments")
    ):
        track_items: list[Any] = []
        for track_key, track_value in (("lower", timeline.get("lower_track", timeline.get("lowerTrack"))), ("upper", timeline.get("upper_track", timeline.get("upperTrack")))):
            for item in _timeline_collection(track_value):
                if isinstance(item, Mapping):
                    track_items.append({**item, "video_track": 1 if track_key == "upper" else 0})
        video_collection = track_items
    render_fps = _coerce_fps(timeline.get("fps", timeline.get("frame_rate", _DEFAULT_FPS)), _DEFAULT_FPS)
    clips = _normalise_render_collection(video_collection, int(source_images.shape[0]), render_fps)
    audio_only = _normalise_render_collection(
        timeline.get("audio_clips", timeline.get("audioClips", [])),
        int(source_images.shape[0]),
        render_fps,
        audio_only=True,
    )
    if not clips and not audio_only and not any(
        key in timeline for key in ("clips", "video_clips", "videoClips", "audio_clips", "audioClips")
    ):
        # Match ``normalise_timeline`` for callers that invoke the renderer
        # directly with the compatibility shorthand ``{}``.
        clips = [_default_clip(int(source_images.shape[0]))]
    # The timeline extent includes every serialized item. In particular an
    # unlinked audio clip can extend beyond the last video clip; dropping it
    # here would make the returned VIDEO/AUDIO lengths diverge.
    clip_ends = [
        _safe_int(clip.get("timeline_end", 0), 0)
        for clip in (*clips, *audio_only)
    ]
    # Keep the renderer consistent with ``normalise_timeline``: populated
    # timelines end at the furthest serialized clip, while an explicit
    # duration is retained only for an empty timeline.
    if clip_ends:
        duration = max(clip_ends)
    else:
        if "duration_frames" in timeline or "durationFrames" in timeline:
            duration = max(
                0,
                _safe_int(
                    timeline.get("duration_frames", timeline.get("durationFrames", 0)) or 0,
                    0,
                ),
            )
        elif any(key in timeline for key in ("clips", "video_clips", "videoClips", "audio_clips", "audioClips")):
            # An explicit empty collection means an intentionally blank
            # timeline.  Do not silently restore the source-duration default.
            duration = 0
        else:
            # A bare descriptor is the compatibility shorthand for placing
            # the complete source on the lower track.
            duration = int(source_images.shape[0])
    begin = _safe_int(
        (timeline.get("in_frame", timeline.get("inFrame", 0)) if start_frame is None else start_frame),
        0,
    )
    finish = _safe_int(
        (timeline.get("out_frame", timeline.get("outFrame", duration)) if end_frame is None else end_frame),
        duration,
    )
    begin = max(0, min(begin, duration))
    finish = max(begin, min(finish if finish >= 0 else duration, duration))
    count = finish - begin
    color = _hex_rgb(fill_color)
    mode = str(fit_mode or "letterbox").strip().lower()
    if mode not in {"letterbox", "crop", "fill"}:
        mode = "letterbox"
    device = source_images.device
    output = torch.empty((count, int(height), int(width), 3), dtype=torch.float32, device=device)
    for output_index, timeline_frame in enumerate(range(begin, finish)):
        # Start every frame with the configured timeline fill colour.  Each
        # active clip is then composited with its transformed occupancy mask;
        # this preserves lower-track pixels around a moved/rotated/scaled
        # upper clip while still making true timeline gaps fully filled.
        rendered = torch.empty((int(height), int(width), 3), dtype=torch.float32, device=device)
        rendered[..., 0] = color[0]
        rendered[..., 1] = color[1]
        rendered[..., 2] = color[2]
        for lower in _active_clips(clips, timeline_frame, 0):
            lower_rendered = _render_clip_rgba(source_images, lower, timeline_frame, width, height, mode, color, normalized=True)
            if lower_rendered is not None:
                lower_rgb, lower_alpha = lower_rendered
                rendered = lower_rgb * lower_alpha[..., None] + rendered * (1.0 - lower_alpha[..., None])
        for upper in _active_clips(clips, timeline_frame, 1):
            upper_rendered = _render_clip_rgba(source_images, upper, timeline_frame, width, height, mode, color, normalized=True)
            if upper_rendered is not None:
                upper_rgb, upper_alpha = upper_rendered
                rendered = upper_rgb * upper_alpha[..., None] + rendered * (1.0 - upper_alpha[..., None])
        output[output_index] = rendered
        if progress is not None:
            progress.update()
    return output


# ---------------------------------------------------------------------------
# Audio timeline rendering


def _prepare_audio(audio: Any) -> tuple[torch.Tensor | None, int, int]:
    if not isinstance(audio, Mapping) or not isinstance(audio.get("waveform"), torch.Tensor):
        return None, _DEFAULT_AUDIO_RATE, _DEFAULT_AUDIO_CHANNELS
    waveform = audio["waveform"]
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 3 or waveform.shape[0] <= 0 or waveform.shape[-1] < 0 or waveform.shape[1] <= 0:
        return None, _DEFAULT_AUDIO_RATE, _DEFAULT_AUDIO_CHANNELS
    rate = _safe_int(audio.get("sample_rate", _DEFAULT_AUDIO_RATE), _DEFAULT_AUDIO_RATE, 1)
    channels = int(waveform.shape[1])
    return waveform[0].detach().to(device="cpu", dtype=torch.float32).contiguous(), rate, channels


def _aligned_source_audio(video: Any, components: Any) -> Any:
    """Align audio from a trimmed upstream VIDEO to its frame zero.

    Current ComfyUI ``VideoFromFile`` already trims audio with the selected
    video window.  Some older producers, however, carried the original full
    waveform beside an already-trimmed frame batch.  Avoid double-trimming by
    applying an inferred ``start_frame`` offset only when the waveform is long
    enough to contain both that offset and the complete selected duration.
    An explicit ``audio_start_seconds`` remains authoritative.
    """
    audio = getattr(components, "audio", None)
    if not isinstance(audio, Mapping) or not isinstance(audio.get("waveform"), torch.Tensor):
        return audio
    metadata: dict[str, Any] = _video_metadata(video)
    if _normalise_bool(metadata.get("audio_already_trimmed", False), False):
        return audio
    try:
        rate = _safe_int(audio.get("sample_rate", _DEFAULT_AUDIO_RATE), _DEFAULT_AUDIO_RATE, 1)
        explicit_offset = "audio_start_seconds" in metadata
        start_seconds = _safe_float(metadata.get("audio_start_seconds", 0.0), 0.0, 0.0)
        if start_seconds <= 0 and not explicit_offset:
            start_frame = _safe_int(metadata.get("start_frame", 0), 0, 0)
            source_fps = _coerce_fps(metadata.get("source_fps", metadata.get("fps", 0.0)), 0.0)
            if start_frame > 0 and source_fps > 0 and (
                "source_filename" in metadata or "loader_id" in metadata or "source_frame_count" in metadata
            ):
                start_seconds = start_frame / source_fps
        if start_seconds <= 0:
            return audio
        waveform = audio["waveform"]
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 3:
            return audio
        waveform_duration = int(waveform.shape[-1]) / float(rate)
        selected_duration = _safe_float(
            metadata.get("loaded_duration", metadata.get("duration", 0.0)),
            0.0,
            0.0,
        )
        if selected_duration <= 0:
            images = getattr(components, "images", None)
            loaded_fps = _coerce_fps(metadata.get("loaded_fps", metadata.get("fps", 0.0)), 0.0)
            if isinstance(images, torch.Tensor) and images.ndim >= 1 and loaded_fps > 0:
                selected_duration = int(images.shape[0]) / loaded_fps
        tolerance = max(0.05, 2.0 / max(_safe_float(metadata.get("source_fps", 0.0), 0.0, 0.001), 0.001))
        if not explicit_offset and selected_duration > 0 and waveform_duration < start_seconds + selected_duration - tolerance:
            # The audio length already matches the selected frame batch, so an
            # inferred source-frame offset would remove valid samples twice.
            return audio
        offset = min(int(round(start_seconds * rate)), int(waveform.shape[-1]))
        aligned = waveform[..., offset:]
        if selected_duration > 0:
            aligned = aligned[..., : max(1, int(math.ceil(selected_duration * rate - 1e-9)))]
        return {"waveform": aligned.contiguous(), "sample_rate": rate}
    except (TypeError, ValueError, OverflowError):
        return audio


def render_timeline_audio(
    source_audio: Any,
    timeline: Mapping[str, Any],
    fps: float,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> dict[str, Any]:
    """Mix linked clips on two audio tracks and return standard AUDIO data."""
    if not isinstance(timeline, Mapping):
        timeline = {}
    source, sample_rate, channels = _prepare_audio(source_audio)
    video_collection = timeline.get("clips")
    if video_collection is None:
        video_collection = timeline.get(
            "video_clips",
            timeline.get("videoClips", timeline.get("items", timeline.get("segments", []))),
        )
    if not _timeline_collection(video_collection) and not any(
        key in timeline for key in ("clips", "video_clips", "videoClips", "items", "segments")
    ):
        track_items: list[Any] = []
        for track_key, track_value in (("lower", timeline.get("lower_track", timeline.get("lowerTrack"))), ("upper", timeline.get("upper_track", timeline.get("upperTrack")))):
            for item in _timeline_collection(track_value):
                if isinstance(item, Mapping):
                    track_items.append({**item, "video_track": 1 if track_key == "upper" else 0})
        video_collection = track_items
    render_fps = _coerce_fps(timeline.get("fps", timeline.get("frame_rate", fps)), fps)
    raw_video_items = [item for item in _timeline_collection(video_collection) if isinstance(item, Mapping)]
    raw_audio_items = [
        item
        for item in _timeline_collection(timeline.get("audio_clips", timeline.get("audioClips", [])))
        if isinstance(item, Mapping)
    ]
    raw_items: list[Mapping[str, Any]] = [*raw_video_items, *raw_audio_items]
    # Prefer the declared source range in the timeline.  If a hand-authored
    # descriptor omits it, infer a conservative bound from the available
    # waveform rather than permitting an audio clip to stretch indefinitely.
    # ``source_frame_count`` is authoritative when present: a malformed or
    # hand-authored clip must not regain samples past the connected VIDEO's
    # source range merely because its ``source_end`` is too large.  Without
    # that metadata, infer a conservative bound from the descriptors and the
    # available waveform so direct helper callers still get finite clips.
    declared_source_frames = _safe_int(
        timeline.get("source_frame_count", timeline.get("sourceFrameCount", 0)),
        0,
        0,
    )
    if declared_source_frames <= 0:
        declared_source_frames = max(
            (_safe_int(_first(item, "source_end", "sourceEnd", "source_out", "sourceOut", "out_frame", "outFrame", "end_frame", default=0), 0, 0) for item in raw_items),
            default=0,
        )
        if source is not None and source.numel() > 0:
            waveform_frames = max(1, int(math.ceil(source.shape[-1] / max(sample_rate, 1) * max(render_fps, 1e-6))))
            declared_source_frames = max(declared_source_frames, waveform_frames)
    normalise_source_frames = max(1, declared_source_frames)
    normalised_items: list[Mapping[str, Any]] = []
    for index, item in enumerate(raw_video_items):
        clip = _clip_from_raw(item, index, normalise_source_frames, render_fps, default_audio=True)
        if clip is not None:
            normalised_items.append(clip)
    for index, item in enumerate(raw_audio_items):
        clip = _clip_from_raw(item, index, normalise_source_frames, render_fps, default_audio=True)
        if clip is not None:
            clip["video_track"] = -1
            clip["linked"] = False
            normalised_items.append(clip)
    raw_items = normalised_items
    if declared_source_frames <= 0 and raw_items:
        declared_source_frames = normalise_source_frames
    if not raw_items and not any(key in timeline for key in ("clips", "video_clips", "videoClips", "audio_clips", "audioClips")):
        # A bare descriptor means “play the connected source unchanged,” just
        # like ``normalise_timeline({})``.  Recreate its default linked item so
        # direct audio-helper callers receive the source waveform as well.
        default_frames = declared_source_frames or _safe_int(
            timeline.get("duration_frames", timeline.get("durationFrames", 0)),
            0,
            0,
        )
        if default_frames > 0:
            raw_items = [_default_clip(default_frames)]
    bounded_items = [
        bounded
        for item in raw_items
        for bounded in [_clip_with_source_bound(item, declared_source_frames)]
        if bounded is not None
    ] if declared_source_frames > 0 else raw_items
    clips: list[Mapping[str, Any]] = []
    for item in bounded_items:
        if isinstance(item, Mapping) and _track_index(item.get("audio_track", -1), -1, allow_none=True) >= 0 and _normalise_bool(item.get("enabled", True), True):
            clips.append(item)
    # Audio must have the same timeline extent as the rendered VIDEO, even
    # when the furthest video clip is muted/unlinked (or there is no source
    # waveform at all).  Looking only at ``clips`` with an enabled audio
    # track would truncate the required trailing silence in those cases.
    all_timeline_items = bounded_items
    timeline_ends = [
        _safe_int(item.get("timeline_end", 0), 0)
        for item in all_timeline_items
    ]
    duration = max(
        timeline_ends
        if timeline_ends
        else [_safe_int(timeline.get("duration_frames", timeline.get("durationFrames", 0)), 0)]
    )
    if duration <= 0:
        explicit_collections = any(key in timeline for key in ("clips", "video_clips", "videoClips", "audio_clips", "audioClips"))
        if not explicit_collections and source is not None and source.numel() > 0:
            duration = max(1, int(math.ceil(source.shape[-1] / max(sample_rate, 1) * max(render_fps, 1e-6) - 1e-9)))
        else:
            duration = max(
                _safe_int(timeline.get("out_frame", timeline.get("outFrame", 0)) or 0, 0),
                1,
            )
    begin = max(
        0,
        _safe_int(
            (timeline.get("in_frame", timeline.get("inFrame", 0)) if start_frame is None else start_frame),
            0,
        ),
    )
    finish_raw = (timeline.get("out_frame", timeline.get("outFrame", duration)) if end_frame is None else end_frame)
    finish = duration if finish_raw is None or _safe_int(finish_raw, -1) < 0 else min(duration, max(begin, _safe_int(finish_raw, duration)))
    begin = min(begin, duration)
    if finish <= begin:
        finish = min(duration, begin + 1)
    output_samples = max(1, int(math.ceil((finish - begin) / max(render_fps, 1e-6) * sample_rate - 1e-9)))
    mixed = torch.zeros((channels, output_samples), dtype=torch.float32)
    if source is not None and source.numel() > 0:
        for clip in clips:
            clip_start = _safe_int(clip.get("timeline_start", 0), 0)
            clip_end = _safe_int(clip.get("timeline_end", clip_start), clip_start)
            if clip_end <= begin or clip_start >= finish:
                continue
            source_start_frame = _safe_int(clip.get("source_start", 0), 0)
            # Map each clip's frame interval into output sample coordinates.
            overlap_start = max(begin, clip_start)
            overlap_end = min(finish, clip_end)
            out_start = int(round((overlap_start - begin) / max(render_fps, 1e-6) * sample_rate))
            out_end = int(round((overlap_end - begin) / max(render_fps, 1e-6) * sample_rate))
            src_start = int(round((source_start_frame + overlap_start - clip_start) / max(render_fps, 1e-6) * sample_rate))
            length = min(out_end - out_start, source.shape[-1] - src_start)
            if length <= 0 or out_start >= output_samples:
                continue
            out_start = max(0, out_start)
            length = min(length, output_samples - out_start)
            if length <= 0:
                continue
            # Channel conversion is deliberately deterministic: duplicate mono
            # to stereo or truncate/replicate to the requested source layout.
            src = source[:, max(0, src_start) : max(0, src_start) + length]
            if src.shape[0] != channels:
                if src.shape[0] == 1:
                    src = src.expand(channels, -1)
                elif channels == 1:
                    src = src.mean(dim=0, keepdim=True)
                elif src.shape[0] > channels:
                    src = src[:channels]
                else:
                    src = torch.cat([src, src[-1:].expand(channels - src.shape[0], -1)], dim=0)
            mixed[:, out_start : out_start + src.shape[-1]] += src
    # Audio tracks intentionally sum, then clamp to avoid integer/codec
    # overflow.  This preserves relative gain while making the output safe.
    mixed = mixed.clamp(-1.0, 1.0).unsqueeze(0)
    return {"waveform": mixed, "sample_rate": int(sample_rate)}


# Small integration aliases used by early timeline prototypes and external
# workflow helpers.  The canonical public names remain the explicit
# ``*_frames``/``*_audio`` variants above.
render_timeline = render_timeline_frames
render_timeline_video = render_timeline_frames
mix_timeline_audio = render_timeline_audio


# ---------------------------------------------------------------------------
# Shot detection (lightweight fallback used by the Timeline button)


def predictions_to_scenes(predictions: Sequence[float], threshold: float = 0.5) -> list[tuple[int, int]]:
    """Convert TransNet-style boundary probabilities to inclusive scenes.

    This mirrors ``video_assemble.scene_split_transnetv2`` (strict ``>``
    threshold and inclusive end indices).  The timeline API converts the
    resulting ranges to half-open intervals before creating clips.
    """
    values: list[float] = []
    for item in predictions:
        if isinstance(item, np.ndarray) and item.ndim > 0:
            item = item.flat[0] if item.size else 0.0
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            item = item[0] if item else 0.0
        try:
            values.append(float(item))
        except (TypeError, ValueError, OverflowError):
            values.append(0.0)
    if not values:
        return []
    cutoff = _safe_float(threshold, 0.5, 0.0, 1.0)
    binary = [1 if value > cutoff else 0 for value in values]
    scenes: list[tuple[int, int]] = []
    previous = 0
    start = 0
    for index, current in enumerate(binary):
        if previous == 1 and current == 0:
            start = index
        if previous == 0 and current == 1 and index != 0:
            scenes.append((start, index))
        previous = current
    if binary[-1] == 0:
        scenes.append((start, len(binary) - 1))
    if not scenes:
        scenes.append((0, len(binary) - 1))
    return scenes


def merge_short_scenes(scenes: Sequence[tuple[int, int]], min_scene_seconds: float, fps: float) -> list[tuple[int, int]]:
    """Merge scenes shorter than the configured duration (inclusive ranges)."""
    minimum_seconds = _safe_float(min_scene_seconds, 0.0, 0.0)
    safe_fps = _coerce_fps(fps, _DEFAULT_FPS)
    if minimum_seconds <= 0 or len(scenes) <= 1:
        return [(int(start), int(end)) for start, end in scenes]
    minimum = max(1, int(math.ceil(minimum_seconds * safe_fps)))
    merged = [[int(start), int(end)] for start, end in scenes]
    index = 0
    while index < len(merged):
        start, end = merged[index]
        if end - start + 1 >= minimum or len(merged) == 1:
            index += 1
            continue
        if index == 0:
            merged[1][0] = start
            del merged[0]
        else:
            merged[index - 1][1] = end
            del merged[index]
            index -= 1
    return [(start, end) for start, end in merged]


def detect_shots(
    images: torch.Tensor,
    fps: float,
    threshold: float = 0.5,
    min_scene_seconds: float = 0.0,
    predictions: Sequence[float] | None = None,
) -> list[dict[str, int]]:
    """Detect cuts with frame differences and merge short scenes.

    The optional TransNetV2 integration can feed its probabilities into this
    same post-processing contract.  Keeping the fallback here makes the UI
    useful for ordinary VIDEO tensors and keeps the node import-independent.
    """
    if not isinstance(images, torch.Tensor):
        images = torch.as_tensor(images)
    if images.ndim == 3:
        images = images.unsqueeze(0)
    count = int(images.shape[0]) if images.ndim >= 1 else 0
    if count <= 0:
        return []
    if count == 1:
        return [{"start": 0, "end": 1}]
    if predictions is not None:
        # Detector sidecars occasionally contain one prediction per decoded
        # frame minus a trailing batch, or a few extra padded values.  Align
        # them to the actual VIDEO frame count before applying the external
        # scene-splitting contract so returned ranges always cover valid local
        # source frames.
        values: list[float] = []
        for item in predictions:
            # TransNet sidecars commonly contain two columns (single-frame
            # and many-hot probabilities).  The scene splitter uses the first
            # column; accept both that shape and a plain 1-D sequence here.
            if isinstance(item, np.ndarray) and item.ndim > 0:
                item = item.flat[0] if item.size else 0.0
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                item = item[0] if item else 0.0
            try:
                values.append(float(item))
            except (TypeError, ValueError, OverflowError):
                values.append(0.0)
        if len(values) < count:
            values.extend([values[-1] if values else 0.0] * (count - len(values)))
        elif len(values) > count:
            values = values[:count]
        scenes_inclusive = merge_short_scenes(predictions_to_scenes(values, threshold), min_scene_seconds, fps)
        # Convert the external detector's inclusive end to the canonical
        # half-open range used by the timeline editor.
        return [{"start": int(start), "end": min(count, int(end) + 1)} for start, end in scenes_inclusive if end >= start]

    # Lightweight fallback when TransNetV2 is not installed: mean absolute RGB
    # change between adjacent frames.  Keep the same post-processing contract.
    sample = _normalise_image_tensor(images[..., :3]).detach().to(device="cpu", dtype=torch.float32)
    differences = (sample[1:] - sample[:-1]).abs().mean(dim=(1, 2, 3)).tolist()
    cut_threshold = _safe_float(threshold, 0.5, 0.0, 1.0)
    # TransNet's threshold is a probability in [0,1], while a raw RGB mean
    # difference is scene/content dependent.  Normalize against the observed
    # robust range so the same control remains meaningful in the fallback;
    # constant-motion clips produce no artificial cuts.
    if differences:
        values = np.asarray(differences, dtype=np.float32)
        baseline = float(np.median(values))
        peak = float(values.max())
        span = max(peak - baseline, 1e-8)
        if peak > 1e-8 and peak - baseline <= 1e-8:
            # A tiny synthetic test clip (or a hard-cut sequence of equally
            # different colour cards) can have identical non-zero deltas.  Do
            # not erase every boundary merely because robust normalisation has
            # no spread; scale against the observed peak instead.
            scores = (values / peak).clip(0.0, 1.0)
        else:
            scores = ((values - baseline) / span).clip(0.0, 1.0)
    else:
        scores = np.zeros((0,), dtype=np.float32)
    boundaries = [0] + [index + 1 for index, value in enumerate(scores.tolist()) if float(value) > cut_threshold] + [count]
    boundaries = sorted(set(max(0, min(count, int(value))) for value in boundaries))
    raw = [(boundaries[index], boundaries[index + 1] - 1) for index in range(len(boundaries) - 1) if boundaries[index + 1] > boundaries[index]]
    scenes_inclusive = merge_short_scenes(raw, min_scene_seconds, fps)
    return [{"start": int(start), "end": min(count, int(end) + 1)} for start, end in scenes_inclusive if end >= start]


# Compatibility alias for integrations that use a more descriptive name.
detect_video_shots = detect_shots


def _transnet_configuration() -> tuple[list[str], Path | None] | None:
    """Locate the optional TransNetV2 runtime used by ``video_assemble``.

    TransNetV2 depends on TensorFlow, which is intentionally not imported into
    the ComfyUI process.  If an external executable/script is present we run
    it in a subprocess; otherwise callers transparently use the deterministic
    frame-difference fallback above.  Environment variables make the setup
    portable while the user's referenced ``video_assemble`` checkout remains
    the first Windows default.
    """
    env_exec = str(os.environ.get("CINESTYLE_TRANSNET_EXEC", "")).strip()
    env_python = str(os.environ.get("CINESTYLE_TRANSNET_PYTHON", "")).strip()
    env_script = str(os.environ.get("CINESTYLE_TRANSNET_SCRIPT", "")).strip()
    env_weights = str(os.environ.get("CINESTYLE_TRANSNET_WEIGHTS", "")).strip()
    roots: list[Path] = []
    for value in (
        os.environ.get("CINESTYLE_VIDEO_ASSEMBLE_ROOT", ""),
        r"E:\work\yangbin\video_assemble",
        # Keep the historical spelling used in the original task notes as a
        # fallback for deployments that retain that directory layout.
        r"E:\work\yangbin\video\_assemble",
        str(Path(__file__).resolve().parents[1] / "video_assemble"),
    ):
        if value:
            candidate = Path(value).expanduser()
            if candidate not in roots:
                roots.append(candidate)

    weights: Path | None = Path(env_weights).expanduser() if env_weights else None
    if weights is not None and not weights.is_dir():
        weights = None
    if weights is None:
        for root in roots:
            for candidate in (
                root / "TransNetV2" / "inference" / "transnetv2-weights",
                root / "transnetv2" / "inference" / "transnetv2-weights",
            ):
                if candidate.is_dir():
                    weights = candidate
                    break
            if weights is not None:
                break

    if env_exec:
        try:
            command = shlex.split(env_exec, posix=(os.name != "nt"))
        except ValueError:
            command = []
        if command:
            return command, weights

    if env_python and env_script:
        python_path, script_path = Path(env_python).expanduser(), Path(env_script).expanduser()
        if python_path.is_file() and script_path.is_file():
            return [str(python_path), str(script_path)], weights

    direct = shutil.which("transnetv2_predict")
    if direct:
        return [direct], weights

    for root in roots:
        script = root / "TransNetV2" / "inference" / "transnetv2.py"
        if not script.is_file():
            script = root / "transnetv2" / "inference" / "transnetv2.py"
        if not script.is_file():
            continue
        python_candidates = (
            root / ".venv" / "Scripts" / "python.exe",
            root / ".venv" / "bin" / "python",
        )
        for python_path in python_candidates:
            if python_path.is_file():
                return [str(python_path), str(script)], weights

    # A current-process TensorFlow install is uncommon but valid (for example
    # a Linux deployment with the official package installed globally).
    if env_script and Path(env_script).is_file():
        try:
            import importlib.util

            if importlib.util.find_spec("tensorflow") is not None:
                return [sys.executable, str(Path(env_script).expanduser())], weights
        except Exception:
            pass
    return None


def _write_transnet_video(frames: torch.Tensor, fps: float, path: Path) -> None:
    """Write a small CFR RGB proxy suitable for the official detector."""
    if av is None:
        raise RuntimeError("PyAV is unavailable for TransNetV2 detection.")
    value = _normalise_image_tensor(frames[..., :3]).detach().to(device="cpu")
    array = value.mul(255.0).round().to(torch.uint8).numpy()
    if array.ndim != 4 or array.shape[0] <= 0:
        raise ValueError("No frames are available for shot detection.")
    height, width = int(array.shape[1]), int(array.shape[2])
    encoded_width = max(2, width + (width % 2))
    encoded_height = max(2, height + (height % 2))
    rate = Fraction(max(float(fps), 0.001)).limit_denominator(1000)
    with av.open(str(path), mode="w", format="mp4") as container:
        try:
            stream = container.add_stream("libx264", rate=rate)
            stream.options = {"preset": "ultrafast", "crf": "28"}
        except Exception:
            stream = container.add_stream("mpeg4", rate=rate)
        stream.width, stream.height, stream.pix_fmt = encoded_width, encoded_height, "yuv420p"
        for frame_array in array:
            if encoded_width != width or encoded_height != height:
                frame_array = np.pad(
                    frame_array,
                    ((0, encoded_height - height), (0, encoded_width - width), (0, 0)),
                    mode="edge",
                )
            video_frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _transnet_predictions(images: torch.Tensor, fps: float) -> list[float] | None:
    """Return official TransNetV2 single-frame probabilities when available."""
    configuration = _transnet_configuration()
    if configuration is None:
        return None
    command, weights = configuration
    try:
        with tempfile.TemporaryDirectory(prefix="cinestyle_transnet_") as temporary:
            video_path = Path(temporary) / "source.mp4"
            _write_transnet_video(images, fps, video_path)
            run_command = [*command]
            # The official script accepts --weights; custom executables may not.
            if weights is not None and run_command and "transnetv2.py" in run_command[-1].lower():
                run_command.extend(["--weights", str(weights)])
            run_command.append(str(video_path))
            completed = subprocess.run(
                run_command,
                cwd=str(Path(command[-1]).parent) if len(command) > 1 else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=900,
                check=False,
            )
            prediction_path = Path(f"{video_path}.predictions.txt")
            if completed.returncode != 0 or not prediction_path.is_file():
                _LOGGER.info("TransNetV2 unavailable/failed (code=%s): %s", completed.returncode, (completed.stdout or "")[-500:])
                return None
            predictions: list[float] = []
            for line in prediction_path.read_text(encoding="utf-8", errors="replace").splitlines():
                fields = line.strip().split()
                if not fields:
                    continue
                try:
                    predictions.append(float(fields[0]))
                except ValueError:
                    continue
            if not predictions:
                return None
            count = int(images.shape[0])
            if len(predictions) < count:
                predictions.extend([predictions[-1]] * (count - len(predictions)))
            return predictions[:count]
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        _LOGGER.info("TransNetV2 detection skipped: %s", exc)
        return None


def detect_shots_preferred(
    images: torch.Tensor,
    fps: float,
    threshold: float = 0.5,
    min_scene_seconds: float = 0.0,
) -> tuple[list[dict[str, int]], str]:
    """Run TransNetV2 when configured, with a deterministic local fallback."""
    if not isinstance(images, torch.Tensor):
        images = torch.as_tensor(images)
    if images.ndim == 3:
        images = images.unsqueeze(0)
    source_key = hashlib.sha256(
        _canonical_json(
            {
                "source": _source_fingerprint(None, images),
                "fps": round(float(fps), 6),
                "threshold": round(float(threshold), 6),
                "min_scene_seconds": round(float(min_scene_seconds), 6),
            }
        ).encode("utf-8")
    ).hexdigest()
    with _SHOT_CACHE_LOCK:
        cached = _SHOT_CACHE.get(source_key)
    if cached is not None:
        return [dict(item) for item in cached], "cached"
    predictions = _transnet_predictions(images, fps)
    if predictions is not None:
        shots = detect_shots(images, fps, threshold, min_scene_seconds, predictions=predictions)
        detector = "transnetv2"
    else:
        shots = detect_shots(images, fps, threshold, min_scene_seconds)
        detector = "frame-difference-fallback"
    with _SHOT_CACHE_LOCK:
        _SHOT_CACHE[source_key] = [dict(item) for item in shots]
        # Keep the in-memory cache bounded even when a user repeatedly probes
        # many source clips.
        while len(_SHOT_CACHE) > 64:
            _SHOT_CACHE.pop(next(iter(_SHOT_CACHE)))
    return shots, detector


# ---------------------------------------------------------------------------
# Preview/state cache helpers


def _cache_store() -> Any:
    global _CACHE_STORE
    if _CACHE_STORE is None:
        package = __name__.rsplit(".", 1)[0]
        module = None
        for module_name in (f"{package}._py_preview_cache", "py.preview_cache", "preview_cache"):
            try:
                module = __import__(module_name, fromlist=["*"])
                break
            except Exception:
                continue
        if module is None:
            raise RuntimeError("CineStyle preview cache module is unavailable.")
        # Keep several edit revisions plus the original source variant alive;
        # a user can scrub/preview many unsaved layouts before applying one.
        _CACHE_STORE = module.PreviewCacheStore("video_time_edit", max_entries=32, max_bytes=4 * 1024**3)
    return _CACHE_STORE


def _node_id() -> str:
    value = getattr(getattr(CSVideoTimelineEdit, "hidden", None), "unique_id", None)
    return str(value or "").strip()


def _state_root() -> Path | None:
    try:
        root = Path(folder_paths.get_temp_directory()) / "cinestyle" / "video_time_edit_state"
        root.mkdir(parents=True, exist_ok=True)
        return root
    except Exception:
        return None


def _state_path(node_id: Any) -> Path | None:
    key = str(node_id or "").strip()
    root = _state_root()
    if not key or root is None:
        return None
    return root / f"{hashlib.sha1(key.encode('utf-8')).hexdigest()}.json"


def _save_state(node_id: Any, value: Mapping[str, Any], revision: Any = None) -> bool:
    key = str(node_id or "").strip()
    if not key:
        return False
    requested_revision = _safe_int(revision, 0, 0) if revision is not None else 0
    with _STATE_LOCK:
        previous_revision = int(_STATE_REVISIONS.get(key, 0) or 0)
        if requested_revision and requested_revision <= previous_revision:
            return False
        next_revision = max(previous_revision, requested_revision) + (0 if requested_revision else 1)
        _STATE_REVISIONS[key] = next_revision
        payload = {**_json_safe(dict(value)), "version": _SCHEMA_VERSION, "revision": next_revision}
        _TIMELINE_STATES[key] = dict(payload)
    path = _state_path(key)
    if path is None:
        return True
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    # Keep the manifest replacement ordered with the in-memory revision.  A
    # slow filesystem write must not let an older request replace a newer
    # timeline on disk after both requests have been accepted.
    with _STATE_LOCK:
        # A newer request may have committed while this call was preparing its
        # temporary file.  Never let the older request replace the newer
        # manifest on disk; the in-memory state and revision are authoritative.
        current = _TIMELINE_STATES.get(key)
        if not isinstance(current, Mapping) or _safe_int(current.get("revision", 0), 0, 0) != next_revision:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            return False
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=True, separators=(",", ":")), encoding="utf-8")
            temporary.replace(path)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return True


def _load_state(node_id: Any) -> dict[str, Any] | None:
    key = str(node_id or "").strip()
    if not key:
        return None
    with _STATE_LOCK:
        cached = _TIMELINE_STATES.get(key)
    if cached:
        return dict(cached)
    path = _state_path(key)
    if path is None or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, Mapping) and int(value.get("version", 0)) == _SCHEMA_VERSION:
            with _STATE_LOCK:
                _TIMELINE_STATES[key] = dict(value)
                _STATE_REVISIONS[key] = _safe_int(value.get("revision", 0), 0, 0)
            return dict(value)
    except (OSError, ValueError, TypeError):
        pass
    return None


def _history_root() -> Path | None:
    try:
        root = Path(folder_paths.get_temp_directory()) / "cinestyle" / "video_time_edit_history"
        root.mkdir(parents=True, exist_ok=True)
        return root
    except Exception:
        return None


def _history_path(node_id: Any) -> Path | None:
    key = str(node_id or "").strip()
    root = _history_root()
    if not key or root is None:
        return None
    return root / f"{hashlib.sha1(key.encode('utf-8')).hexdigest()}.json"


def _history_source_key(timeline: Mapping[str, Any]) -> str:
    for name in ("source_identity", "source_fingerprint", "input_signature"):
        value = str(timeline.get(name) or "").strip()
        if value:
            return value
    return ""


def _load_history(node_id: Any) -> dict[str, Any]:
    key = str(node_id or "").strip()
    with _HISTORY_LOCK:
        cached = _TIMELINE_HISTORIES.get(key)
        if isinstance(cached, Mapping):
            return dict(cached)
        path = _history_path(key)
        if path is not None and path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                entries = value.get("entries") if isinstance(value, Mapping) else None
                if isinstance(entries, list):
                    history = {
                        "version": 1,
                        "cursor": _safe_int(value.get("cursor", len(entries) - 1), len(entries) - 1, -1, len(entries) - 1),
                        "entries": [dict(item) for item in entries if isinstance(item, Mapping) and isinstance(item.get("timeline"), Mapping)],
                    }
                    history["cursor"] = min(int(history["cursor"]), len(history["entries"]) - 1)
                    _TIMELINE_HISTORIES[key] = history
                    return dict(history)
            except (OSError, ValueError, TypeError):
                pass
        history = {"version": 1, "cursor": -1, "entries": []}
        _TIMELINE_HISTORIES[key] = history
        return dict(history)


def _write_history(node_id: Any, history: Mapping[str, Any]) -> None:
    key = str(node_id or "").strip()
    safe_history = _json_safe(dict(history))
    with _HISTORY_LOCK:
        _TIMELINE_HISTORIES[key] = dict(safe_history)
        path = _history_path(key)
        if path is None:
            return
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(safe_history, ensure_ascii=True, separators=(",", ":")), encoding="utf-8")
            temporary.replace(path)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _history_result(history: Mapping[str, Any]) -> dict[str, Any]:
    entries = history.get("entries") if isinstance(history.get("entries"), list) else []
    cursor = _safe_int(history.get("cursor", -1), -1, -1, len(entries) - 1)
    entry = entries[cursor] if 0 <= cursor < len(entries) and isinstance(entries[cursor], Mapping) else None
    return {
        "ok": True,
        "cursor": cursor,
        "count": len(entries),
        "can_undo": cursor > 0,
        "can_redo": 0 <= cursor < len(entries) - 1,
        "timeline": dict(entry.get("timeline") or {}) if entry else None,
        "label": str(entry.get("label") or "") if entry else "",
    }


def _update_history(
    node_id: Any,
    action: str,
    timeline: Mapping[str, Any] | None = None,
    label: str = "",
    target_cursor: Any = None,
) -> dict[str, Any]:
    key = str(node_id or "").strip()
    action = str(action or "record").strip().lower()
    with _HISTORY_LOCK:
        history = _load_history(key)
        entries = list(history.get("entries") or [])
        cursor = _safe_int(history.get("cursor", -1), -1, -1, len(entries) - 1)

        if action in {"init", "record", "reset"}:
            if not isinstance(timeline, Mapping):
                raise ValueError("A timeline is required when recording history.")
            snapshot = _json_safe(dict(timeline))
            signature = _canonical_json(snapshot)
            if action == "reset":
                _TIMELINE_HISTORIES.pop(key, None)
                path = _history_path(key)
                if path is not None:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass
                return _history_result({"version": 1, "cursor": -1, "entries": []})
            if action == "init" and entries:
                current_timeline = entries[cursor].get("timeline") if 0 <= cursor < len(entries) else {}
                current_source = _history_source_key(current_timeline) if isinstance(current_timeline, Mapping) else ""
                incoming_source = _history_source_key(snapshot)
                if current_source and incoming_source and current_source != incoming_source:
                    entries = []
                    cursor = -1
            current_signature = ""
            if 0 <= cursor < len(entries) and isinstance(entries[cursor], Mapping):
                current_signature = _canonical_json(entries[cursor].get("timeline") or {})
            if current_signature != signature:
                entries = entries[: cursor + 1]
                entries.append({
                    "label": str(label or ("Open editor" if action == "init" else "Timeline edit")),
                    "created_at": time.time(),
                    "timeline": snapshot,
                })
                cursor = len(entries) - 1
        elif action == "undo":
            cursor = max(0, cursor - 1) if entries else -1
        elif action == "redo":
            cursor = min(len(entries) - 1, cursor + 1) if entries else -1
        elif action == "seek":
            cursor = _safe_int(target_cursor, cursor, -1, len(entries) - 1)
        else:
            raise ValueError(f"Unsupported history action: {action}")

        history = {"version": 1, "cursor": cursor, "entries": entries}
        _write_history(key, history)
        return _history_result(history)


def _timeline_cache_key(source_key: str, timeline: Mapping[str, Any], width: int, height: int, fit_mode: str, color: str, fps: float, proxy: bool) -> str:
    payload = {
        "version": _SCHEMA_VERSION,
        "render_version": _RENDER_CACHE_VERSION,
        "source": str(source_key or ""),
        "timeline": timeline,
        "width": int(width),
        "height": int(height),
        "fit_mode": str(fit_mode),
        "fill_color": str(color),
        "fps": float(fps),
        "proxy": bool(proxy),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _proxy_dimensions(width: int, height: int, limit: int = 640) -> tuple[int, int]:
    safe_width = max(1, _safe_int(width, 1, 1))
    safe_height = max(1, _safe_int(height, 1, 1))
    safe_limit = max(1, _safe_int(limit, 640, 1))
    scale = min(1.0, float(safe_limit) / safe_width, float(safe_limit) / safe_height)
    return max(1, _round_half_up(safe_width * scale)), max(1, _round_half_up(safe_height * scale))


def _resolve_output_options(payload: Mapping[str, Any], output_config: Mapping[str, Any] | None = None) -> tuple[Any, Any, Any, str, str]:
    """Merge neutral preview controls with the canonical timeline output."""
    saved = output_config if isinstance(output_config, Mapping) else {}
    raw_width = _first(payload, "width", "output_width", "outputWidth", default=-1)
    raw_height = _first(payload, "height", "output_height", "outputHeight", default=-1)
    raw_multiple = _first(payload, "multiple", "output_multiple", "outputMultiple", default=32)
    raw_fit = str(_first(payload, "fit_mode", "fitMode", "output_fit_mode", "outputFitMode", default="letterbox") or "letterbox").strip().lower()
    raw_color = _first(payload, "fill_color", "fillColor", "output_fill_color", "outputFillColor", default="#000000")
    width = raw_width if _safe_int(raw_width, -1) > 0 else _first(saved, "width", "output_width", "outputWidth", default=raw_width)
    height = raw_height if _safe_int(raw_height, -1) > 0 else _first(saved, "height", "output_height", "outputHeight", default=raw_height)
    saved_multiple = _first(saved, "multiple", "output_multiple", "outputMultiple", default=None)
    multiple = raw_multiple if _safe_int(raw_multiple, 32, 1) != 32 or not saved_multiple else saved_multiple
    saved_fit = _first(saved, "fit_mode", "fitMode", "output_fit_mode", "outputFitMode", default=None)
    saved_color = _first(saved, "fill_color", "fillColor", "output_fill_color", "outputFillColor", default=None)
    fit_mode = saved_fit if raw_fit == "letterbox" and saved_fit else raw_fit
    color = saved_color if _normalise_hex(raw_color) == "#000000" and saved_color else raw_color
    return width, height, multiple, str(fit_mode), _normalise_hex(color)


def _resize_batch(images: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if images.shape[1:3] == (height, width):
        return images
    return F.interpolate(images.permute(0, 3, 1, 2).float(), size=(height, width), mode="bilinear", align_corners=False).permute(0, 2, 3, 1).clamp(0.0, 1.0)


def _downscale_preview_batch(images: torch.Tensor, limit: int = 640) -> torch.Tensor:
    """Bound direct-file preview memory while leaving cached VIDEO paths intact."""
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] <= 0:
        return images
    source_height, source_width = int(images.shape[1]), int(images.shape[2])
    scale = min(1.0, float(limit) / max(1, source_width), float(limit) / max(1, source_height))
    if scale >= 1.0:
        return images
    target_width = max(1, _round_half_up(source_width * scale))
    target_height = max(1, _round_half_up(source_height * scale))
    return _resize_batch(images, target_width, target_height)


def _cache_proxy(node_id: str, frames: torch.Tensor, audio: Any, fps: float, info: Mapping[str, Any], cache_key: str) -> dict[str, Any] | None:
    if not isinstance(frames, torch.Tensor):
        frames = torch.as_tensor(frames)
    if not node_id or frames.ndim != 4 or frames.shape[0] <= 0:
        return None
    try:
        entry = _cache_store().put_preview(
            node_id,
            frames,
            fps,
            proxy=True,
            cache_fingerprint=cache_key,
            encode_video=True,
            info=dict(info),
            audio=audio,
            force=False,
        )
        if isinstance(entry, dict):
            # Playback reads the encoded MP4; retaining a full-length waveform
            # in the long-lived proxy index needlessly doubles memory use.
            entry["audio"] = None
        _PREVIEW_WARNINGS.pop(str(node_id), None)
        return entry
    except Exception as exc:
        # Some PyAV/FFmpeg builds reject unusual source sample rates (synthetic
        # VIDEO tests may use a tiny rate such as 2 Hz).  The proxy remains a
        # useful visual cache in that case; retry without an audio stream while
        # preserving the final node's full AUDIO output.
        if audio is not None:
            try:
                fallback_info = dict(info)
                fallback_info["proxy_audio_warning"] = str(exc)
                entry = _cache_store().put_preview(
                    node_id,
                    frames,
                    fps,
                    proxy=True,
                    cache_fingerprint=cache_key,
                    encode_video=True,
                    info=fallback_info,
                    audio=None,
                    force=False,
                )
                if isinstance(entry, dict):
                    entry["audio"] = None
                _PREVIEW_WARNINGS.pop(str(node_id), None)
                return entry
            except Exception:
                pass
        _LOGGER.warning("Timeline proxy cache unavailable: %s", exc)
        _PREVIEW_WARNINGS[node_id] = str(exc)
        return None


def _cache_source_proxy(node_id: str, frames: torch.Tensor, fps: float, info: Mapping[str, Any], cache_key: str, audio: Any = None) -> dict[str, Any] | None:
    """Keep a small original-source cache for shot detection and frame probes.

    The source variant is encoded as a low-resolution MP4 as well as an NPY
    frame store.  Encoding here means a generic VIDEO can later provide both
    visual and audio proxy playback without retaining the full waveform in the
    long-lived Python cache entry.
    """
    if not isinstance(frames, torch.Tensor):
        frames = torch.as_tensor(frames)
    if not node_id or frames.ndim != 4 or frames.shape[0] <= 0:
        return None
    try:
        entry = _cache_store().put(
            node_id,
            frames,
            fps,
            variant="source",
            cache_fingerprint=cache_key,
            encode_video=True,
            info=dict(info),
            audio=audio,
            force=False,
        )
        # The encoded source MP4 is now self-contained; do not keep a second
        # in-memory copy of a potentially hour-long waveform on the entry.
        if isinstance(entry, dict):
            entry["audio"] = None
        return entry
    except Exception as exc:
        if audio is not None:
            try:
                entry = _cache_store().put(
                    node_id,
                    frames,
                    fps,
                    variant="source",
                    cache_fingerprint=cache_key,
                    encode_video=True,
                    info={**dict(info), "source_audio_warning": str(exc)},
                    audio=None,
                    force=False,
                )
                if isinstance(entry, dict):
                    entry["audio"] = None
                return entry
            except Exception:
                pass
        _LOGGER.debug("Timeline source cache unavailable: %s", exc)
        return None


def _cache_wait_input(
    node_id: Any,
    prompt: Any,
    frames: torch.Tensor,
    fps: float,
    info: Mapping[str, Any] | None = None,
    audio: Any = None,
) -> dict[str, Any] | None:
    """Populate the shared wait-input cache used by CineStyle preview UIs.

    The timeline owns an additional source variant for exact frame scrubbing,
    but publishing the same input through the common chain cache keeps this
    node compatible with the existing Selector/Subtitle/Grade preview flow.
    Cache failures are deliberately non-fatal: the node-owned cache remains a
    valid fallback and final rendering never depends on preview media.
    """
    key = str(node_id or "").strip()
    if not key or not isinstance(frames, torch.Tensor) or frames.ndim != 4 or frames.shape[0] <= 0:
        return None
    package = __name__.rsplit(".", 1)[0]
    module = sys.modules.get(f"{package}._py_preview_cache")
    if module is None:
        try:
            module = __import__(f"{package}._py_preview_cache", fromlist=["*"])
        except Exception:
            module = None
    if module is None:
        return None
    try:
        chain = module.build_input_chain(prompt, key, ("video",))
        if chain is None:
            return None
        cache_info = dict(info or {})
        cache_info.update({
            "producer_node_id": key,
            "producer_node_type": _NODE_ID,
            "source_fingerprint": str(cache_info.get("source_fingerprint") or ""),
        })
        entry = module.get_wait_input_cache_store().put_chain(
            chain,
            frames[..., :3],
            fps,
            info=cache_info,
            # ``PreviewCacheStore`` performs its own AUDIO normalisation and
            # expects the standard ``{"waveform", "sample_rate"}`` mapping.
            # Passing the internal ``(tensor, rate, channels)`` tuple returned
            # by ``_prepare_audio`` would silently drop audio from the shared
            # wait-input cache.
            audio=audio,
            force=True,
        )
        if isinstance(entry, dict):
            entry["audio"] = None
        return entry
    except Exception as exc:
        _LOGGER.debug("Timeline wait-input cache unavailable: %s", exc)
        return None


def _proxy_request_key(payload: Mapping[str, Any]) -> str:
    """Build a stable key for an asynchronously rendered timeline proxy."""
    descriptor = payload.get("timeline_json", payload.get("timeline", {}))
    if descriptor is None or (isinstance(descriptor, str) and not descriptor.strip()):
        descriptor = payload.get("timeline", {})
    if isinstance(descriptor, str):
        try:
            descriptor = _parse_json(descriptor)
        except ValueError:
            descriptor = {"raw": descriptor}
    video_filename = str(payload.get("video_filename", payload.get("video", ""))).strip()
    file_fingerprint = _file_source_fingerprint({"source_filename": video_filename}) if video_filename else ""
    key_payload = {
        "version": _SCHEMA_VERSION,
        "render_version": _RENDER_CACHE_VERSION,
        "node_id": str(payload.get("node_id", "")),
        "source_token": str(payload.get("source_token", "")),
        "source_cache_key": str(payload.get("source_cache_key", "")),
        "source_identity": str(payload.get("source_identity", "")),
        "source_fingerprint_kind": str(payload.get("source_fingerprint_kind", "")),
        "source_frame_count": _safe_int(payload.get("source_frame_count", payload.get("source_frames", 0)), 0, 0),
        "video_filename": video_filename,
        "video_file_fingerprint": file_fingerprint,
        "source_start_frame": _safe_int(payload.get("source_start_frame", 0), 0, 0),
        "source_end_frame": _safe_int(payload.get("source_end_frame", -1), -1),
        "source_target_fps": _safe_float(payload.get("source_target_fps", 0), 0.0, 0.0),
        "source_output_width": _safe_int(payload.get("source_output_width", 0), 0, 0),
        "source_output_height": _safe_int(payload.get("source_output_height", 0), 0, 0),
        "source_output_multiple": _safe_int(payload.get("source_output_multiple", 1), 1, 1),
        "timeline": descriptor,
        "width": _safe_int(payload.get("width", 0), 0, 0),
        "height": _safe_int(payload.get("height", 0), 0, 0),
        "multiple": _safe_int(payload.get("multiple", 32), 32, 1),
        "fit_mode": str(payload.get("fit_mode", "letterbox")),
        "fill_color": _normalise_hex(payload.get("fill_color", "#000000")),
    }
    return hashlib.sha256(_canonical_json(key_payload).encode("utf-8")).hexdigest()


def _proxy_job_public(job: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "job_key": str(job.get("job_key", "")),
        "status": str(job.get("status", "queued")),
        "progress": int(job.get("progress", 0) or 0),
        "stage": str(job.get("stage", "queued")),
    }
    for key in ("error", "token", "video_url", "info"):
        if key in job and job.get(key) is not None:
            result[key] = _json_safe(job[key])
    return result


def _proxy_job_media_available(job: Mapping[str, Any] | None) -> bool:
    """Check that a cached ready-job token still points at a live media file."""
    if not isinstance(job, Mapping) or str(job.get("status", "")) != "ready":
        return False
    token = str(job.get("token") or "").strip()
    if not token:
        return False
    try:
        entry = _cache_store().get_token(token)
        path = Path(str((entry or {}).get("video_path") or (entry or {}).get("path") or "")) if entry else None
        return entry is not None and path is not None and path.is_file()
    except Exception:
        return False


def _prune_proxy_jobs(now: float | None = None) -> None:
    current = float(now if now is not None else time.time())
    with _PROXY_JOBS_LOCK:
        stale = [
            key
            for key, job in _PROXY_JOBS.items()
            if str(job.get("status")) in {"ready", "failed", "cancelled"}
            and current - float(job.get("created", current) or current) > _PROXY_JOB_TTL_SECONDS
        ]
        for key in stale:
            _PROXY_JOBS.pop(key, None)
        # Protect against a client generating many unique timelines in one
        # session even before the TTL expires.
        if len(_PROXY_JOBS) > 128:
            ordered = sorted(_PROXY_JOBS.items(), key=lambda item: float(item[1].get("created", 0) or 0))
            for key, _ in ordered[: len(_PROXY_JOBS) - 128]:
                _PROXY_JOBS.pop(key, None)


def _decode_preview_audio(payload: Mapping[str, Any]) -> Any:
    """Best-effort audio extraction for a proxy source.

    NPY source caches intentionally contain video frames only.  If the token or
    filename resolves to an encoded media file, decode and resample it to a
    standard float waveform; final node execution always uses the original
    VIDEO components and is unaffected by this preview-only helper.
    """
    if av is None:
        return None
    token = str(payload.get("source_token") or payload.get("token") or "").strip()
    entry = None
    if token:
        try:
            entry = _cache_store().get_token(token)
        except Exception:
            entry = None
        if entry is None:
            package = __name__.rsplit(".", 1)[0]
            for module_name, getter in (
                (f"{package}._py_loader_preview_cache", "get_loader_preview_cache"),
                (f"{package}._py_preview_cache", "get_wait_input_cache_store"),
            ):
                try:
                    module = __import__(module_name, fromlist=["*"])
                    store = getattr(module, getter)()
                    entry = store.entry_for_token(token) if token.startswith("loader_preview:") else store.get_token(token)
                    if entry is not None:
                        break
                except Exception:
                    continue
    source = str(payload.get("video_filename") or payload.get("video") or "").strip()
    path = Path(str((entry or {}).get("video_path") or (entry or {}).get("path") or "")) if entry else None
    if path is None or not path.is_file():
        if source:
            try:
                path = Path(folder_paths.get_annotated_filepath(source)) if folder_paths.exists_annotated_filepath(source) else Path(os.path.abspath(source))
            except Exception:
                path = None
    cached_audio = (entry or {}).get("audio") if isinstance(entry, Mapping) else None
    if isinstance(cached_audio, Mapping) and isinstance(cached_audio.get("waveform"), torch.Tensor):
        return cached_audio
    if path is None or not path.is_file():
        return None
    try:
        with av.open(str(path), mode="r") as container:
            if not container.streams.audio:
                return None
            stream = container.streams.audio[0]
            video_stream = container.streams.video[0] if container.streams.video else None
            video_rate = video_stream.average_rate or video_stream.guessed_rate if video_stream is not None else None
            source_fps = float(Fraction(video_rate)) if video_rate else _DEFAULT_FPS
            rate = _safe_int(getattr(stream, "rate", 0) or 48000, 48000, 1)
            channels = max(1, int(getattr(stream, "channels", 2) or 2))
            chunks: list[torch.Tensor] = []
            for audio_frame in container.decode(stream):
                try:
                    array = audio_frame.to_ndarray(format="fltp")
                except TypeError:
                    array = audio_frame.to_ndarray()
                value = np.asarray(array, dtype=np.float32)
                if value.ndim == 1:
                    value = value[None, :]
                if value.ndim == 2 and value.shape[1] > 0:
                    chunks.append(torch.from_numpy(np.ascontiguousarray(value)))
            if not chunks:
                return None
            waveform = torch.cat(chunks, dim=1).unsqueeze(0)
            # Encoded loader/wait-input caches already contain the selected
            # VIDEO window.  Only apply the frame window to a direct filename
            # fallback, where the file is still the full source.
            if entry is None:
                start_frame = max(0, _safe_int(payload.get("source_start_frame", 0), 0, 0))
                requested_end = _safe_int(payload.get("source_end_frame", -1), -1)
                if start_frame > 0 or requested_end >= 0:
                    duration_frames = int(round((float(container.duration) / av.time_base) * source_fps)) if container.duration else 0
                    if duration_frames <= 0 and video_stream is not None:
                        duration_frames = int(video_stream.frames or 0)
                    if duration_frames > 0:
                        end_frame = duration_frames - 1 if requested_end < 0 else min(requested_end, duration_frames - 1)
                        end_frame = max(start_frame, end_frame)
                        target_fps = _safe_float(payload.get("source_target_fps", 0), 0.0, 0.0)
                        frame_duration = (
                            max(1, int(round((end_frame - start_frame + 1) * target_fps / max(source_fps, 1e-6)))) / max(target_fps, 1e-6)
                            if target_fps > 0 and abs(target_fps - source_fps) > 1e-6
                            else (end_frame - start_frame + 1) / max(source_fps, 1e-6)
                        )
                        first_sample = min(int(round(start_frame / max(source_fps, 1e-6) * rate)), int(waveform.shape[-1]))
                        last_sample = min(int(round((start_frame / max(source_fps, 1e-6) + frame_duration) * rate)), int(waveform.shape[-1]))
                        waveform = waveform[..., first_sample:max(first_sample + 1, last_sample)]
            return {"waveform": waveform, "sample_rate": rate}
    except Exception:
        return None


def _build_proxy_job(job_key: str, payload: Mapping[str, Any]) -> None:
    """Worker body for the non-blocking proxy render endpoint."""
    started_at = time.perf_counter()
    def update(**values: Any) -> None:
        with _PROXY_JOBS_LOCK:
            job = _PROXY_JOBS.get(job_key)
            if job is not None:
                job.update(values)

    try:
        _timeline_info("preview proxy start")
        _timeline_info("preview proxy stage 1/4: decoding source")
        update(status="running", progress=5, stage="decoding")
        if str(payload.get("source_token") or "").strip() and _preview_entry_for_token(payload)[0] is None and not str(payload.get("video_filename") or "").strip():
            raise RuntimeError("The supplied source preview token does not match this VIDEO.")
        source_result = _decode_preview_source(payload)
        if source_result is None:
            raise RuntimeError("Timeline source cache is unavailable. Run the node once or provide a source video.")
        source_images, fps = source_result
        source_frames = int(source_images.shape[0])
        _timeline_info("preview proxy source ready: frames=%d; fps=%.3f", source_frames, fps)
        timeline_raw = payload.get("timeline_json")
        if timeline_raw is None or (isinstance(timeline_raw, str) and not timeline_raw.strip()):
            timeline_raw = payload.get("timeline", {})
        timeline = normalise_timeline(timeline_raw, source_frames, fps)
        if int(timeline.get("duration_frames", 0)) <= 0:
            timeline["duration_frames"] = 1
            timeline["in_frame"] = 0
            timeline["out_frame"] = 1
        output_config = timeline.get("output") if isinstance(timeline.get("output"), Mapping) else {}
        proxy_source_width, proxy_source_height = _preview_source_dimensions(
            payload,
            {},
            source_images,
        )
        effective_width, effective_height, effective_multiple, fit_mode, color = _resolve_output_options(payload, output_config)
        width, height = _fit_dimensions(
            proxy_source_width,
            proxy_source_height,
            effective_width,
            effective_height,
            effective_multiple,
        )
        fit_mode = str(fit_mode or "letterbox").lower()
        if fit_mode not in {"letterbox", "crop", "fill"}:
            fit_mode = "letterbox"
        color = _normalise_hex(color)
        proxy_width, proxy_height = _proxy_dimensions(width, height)
        _timeline_info("preview proxy stage 2/4: rendering %d frames at %dx%d", int(timeline.get("out_frame", 0)) - int(timeline.get("in_frame", 0)), proxy_width, proxy_height)
        update(progress=20, stage="rendering")
        frame_progress = _TimelineProgress(
            int(timeline.get("out_frame", 0)) - int(timeline.get("in_frame", 0)),
            "preview frame rendering",
        )
        try:
            frames = render_timeline_frames(
                source_images,
                timeline,
                proxy_width,
                proxy_height,
                fit_mode=fit_mode,
                fill_color=color,
                progress=frame_progress,
            )
        finally:
            frame_progress.close()
        _timeline_info("preview proxy frame rendering complete: %d frames", int(frames.shape[0]))
        _timeline_info("preview proxy stage 3/4: rendering timeline audio")
        update(progress=78, stage="encoding")
        source_audio = _decode_preview_audio(payload)
        # Apply the same clip ranges, In/Out window, and sample-level mixing
        # used by final execution.  Passing the raw source waveform directly
        # would make proxy playback audibly disagree with the rendered node
        # whenever a clip is moved, trimmed, muted, or duplicated.
        audio = render_timeline_audio(source_audio, timeline, fps)
        _timeline_info("preview proxy audio processing complete: %s", "available" if audio is not None else "none")
        node_id = str(payload.get("node_id", "")).strip()
        if not node_id:
            raise RuntimeError("A node_id is required to store a timeline proxy.")
        _timeline_info("preview proxy stage 4/4: encoding preview cache")
        source_key = _source_fingerprint(None, source_images)
        source_identity = str(payload.get("source_identity") or "").strip()
        source_fingerprint_kind = str(payload.get("source_fingerprint_kind") or "").strip().lower()
        if source_fingerprint_kind not in {"file", "content"}:
            source_fingerprint_kind = "file" if source_identity and len(source_identity) == 40 else "content"
        # Keep the proxy key in the same identity namespace as the exact
        # preview request.  A file-backed source token may carry a loader SHA1
        # while ``source_images`` itself is content-hashed; using both avoids
        # collisions between two windows of the same file and between cache
        # restarts that retain only encoded source media.
        proxy_source_key = hashlib.sha256(
            _canonical_json(
                {
                    "content": source_key,
                    "identity": source_identity,
                    "kind": source_fingerprint_kind,
                    "frames": source_frames,
                    "start": _safe_int(payload.get("source_start_frame", 0), 0, 0),
                    "end": _safe_int(payload.get("source_end_frame", -1), -1),
                    "target_fps": _safe_float(payload.get("source_target_fps", 0), 0.0, 0.0),
                }
            ).encode("utf-8")
        ).hexdigest()
        cache_key = _timeline_cache_key(proxy_source_key, timeline, proxy_width, proxy_height, fit_mode, color, fps, True)
        info = {
            "proxy": True,
            "cache_fingerprint": cache_key,
            "timeline_json": _canonical_json(timeline),
            "timeline_in_frame": int(timeline.get("in_frame", 0)),
            "timeline_out_frame": int(timeline.get("out_frame", timeline.get("duration_frames", 0))),
            "timeline_duration_frames": int(timeline.get("duration_frames", 0)),
            "width": int(proxy_width),
            "height": int(proxy_height),
            "source_width": int(proxy_source_width),
            "source_height": int(proxy_source_height),
            "fps": float(fps),
            "source_frame_count": source_frames,
            "source_fingerprint": proxy_source_key,
            "source_content_fingerprint": source_key,
            "source_fingerprint_kind": source_fingerprint_kind,
            "source_identity": source_identity,
            "proxy_audio_approximate": str(payload.get("source_token", "")).startswith(("loader_preview:", "wait_input:")),
        }
        entry = _cache_proxy(node_id, frames, audio, fps, info, cache_key)
        if entry is None:
            raise RuntimeError("Unable to store the timeline proxy video.")
        token = str(entry.get("token") or "")
        update(
            status="ready",
            progress=100,
            stage="ready",
            token=token,
            video_url=f"/cinestyle/video-time-edit-preview-video?token={token}",
            info=dict(entry.get("info") or info),
        )
        _timeline_info("preview proxy complete; elapsed=%.2fs", time.perf_counter() - started_at)
    except Exception as exc:
        _timeline_info("preview proxy failed: %s", exc)
        _LOGGER.warning("Timeline proxy job failed: %s", exc)
        update(status="failed", progress=100, stage="failed", error=str(exc))


async def _timeline_proxy_route(request: Any) -> Any:
    if web is None:
        return None
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON payload."}, status=400)
    if not isinstance(payload, Mapping):
        return web.json_response({"error": "JSON object required."}, status=400)
    _prune_proxy_jobs()
    node_id = str(payload.get("node_id", "")).strip()
    if not node_id:
        return web.json_response({"error": "Missing node_id."}, status=400)
    job_key = _proxy_request_key(payload)
    with _PROXY_JOBS_LOCK:
        job = _PROXY_JOBS.get(job_key)
        if job is None or job.get("status") in {"failed", "cancelled"} or (job.get("status") == "ready" and not _proxy_job_media_available(job)):
            job = {"job_key": job_key, "status": "queued", "progress": 0, "stage": "queued", "created": time.time()}
            _PROXY_JOBS[job_key] = job
            threading.Thread(target=_build_proxy_job, args=(job_key, dict(payload)), daemon=True).start()
        result = _proxy_job_public(job)
    return web.json_response(result)


async def _timeline_proxy_progress_route(request: Any) -> Any:
    if web is None:
        return None
    job_key = str(request.query.get("job_key", "")).strip()
    if not job_key:
        return web.json_response({"error": "Missing job_key."}, status=400)
    _prune_proxy_jobs()
    with _PROXY_JOBS_LOCK:
        job = _PROXY_JOBS.get(job_key)
        result = _proxy_job_public(job) if job is not None else {"job_key": job_key, "status": "missing", "progress": 0, "stage": "missing"}
    return web.json_response(result)
# ---------------------------------------------------------------------------
# ComfyUI node


class CSVideoTimelineEdit(io.ComfyNode):
    """Edit a standard VIDEO on two video and two audio tracks."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=_NODE_ID,
            display_name="CS Video Timeline Edit",
            category=_CATEGORY,
            essentials_category="Video Tools",
            search_aliases=["timeline edit", "multi track video", "shot editor", "video timeline"],
            description=(
                "Edit a VIDEO on two video and two audio tracks. The timeline "
                "JSON is persisted by the node; preview uses a low-resolution "
                "proxy and execution renders the selected In/Out range. Input "
                "must be CFR."
            ),
            inputs=[
                io.Video.Input("video", tooltip="Connect any standard ComfyUI VIDEO."),
                io.String.Input(
                    "timeline_json",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Persisted timeline JSON written by Edit Timeline.",
                ),
                io.Int.Input("width", default=-1, min=-1, max=_SCHEMA_MAX_DIMENSION, step=1, advanced=True, tooltip="Output canvas width; -1 uses the input source width. Positive values are rounded up to the selected multiple."),
                io.Int.Input("height", default=-1, min=-1, max=_SCHEMA_MAX_DIMENSION, step=1, advanced=True, tooltip="Output canvas height; -1 uses the input source height. Positive values are rounded up to the selected multiple."),
                io.Int.Input("multiple", default=32, min=1, max=_SCHEMA_MAX_MULTIPLE, step=1, advanced=True, tooltip="Round each output side up to this integer multiple."),
                io.Combo.Input("fit_mode", options=["letterbox", "crop", "fill"], default="letterbox", advanced=True, tooltip="Fit each clip to the output canvas."),
                io.String.Input("fill_color", default="#000000", advanced=True, tooltip="Hex color for letterbox bars and empty timeline regions."),
                io.Int.Input("in_frame", display_name="in_frame", default=0, min=0, max=10000000, step=1, advanced=True, tooltip="Timeline In frame; also trims final output."),
                io.Int.Input("out_frame", display_name="out_frame", default=-1, min=-1, max=10000000, step=1, advanced=True, tooltip="Timeline Out frame (exclusive); -1 uses the timeline end."),
                io.Float.Input("shot_detect_threshold", default=0.5, min=0.0, max=1.0, step=0.01, advanced=True, tooltip="Shot detection threshold."),
                io.Float.Input("shot_detect_min_scene_sec", default=0.0, min=0.0, max=60.0, step=0.01, advanced=True, tooltip="Minimum shot duration used by automatic detection."),
                io.Boolean.Input("wait_for_input_cache", default=False, advanced=True, tooltip="Build the timeline preview cache and pause execution."),
            ],
            outputs=[
                # Match CS Load Video's standard VIDEO socket label/type.
                io.Video.Output(),
                io.Image.Output(display_name="IMAGE"),
                io.Int.Output(display_name="frame_count"),
                io.Audio.Output(display_name="audio"),
                io.Dict.Output(display_name="video_info"),
                io.Float.Output(display_name="fps"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        video: Any,
        timeline_json: Any = "",
        width: int = -1,
        height: int = -1,
        multiple: int = 32,
        fit_mode: str = "letterbox",
        fill_color: str = "#000000",
        in_frame: int = 0,
        out_frame: int = -1,
        shot_detect_threshold: float = 0.5,
        shot_detect_min_scene_sec: float = 0.0,
        wait_for_input_cache: bool = False,
        timeline: Any = None,
        edited_timeline: Any = None,
    ) -> io.NodeOutput:
        started_at = time.perf_counter()
        _timeline_info("start")
        _timeline_info("stage 1/7: validating VIDEO input")
        if video is None or not hasattr(video, "get_components"):
            raise ValueError("video input is not a compatible VIDEO value.")
        components = video.get_components()
        source_metadata: dict[str, Any] = _video_metadata(video)
        for metadata_candidate in (source_metadata,):
            if isinstance(metadata_candidate, Mapping):
                if _normalise_bool(metadata_candidate.get("vfr", metadata_candidate.get("is_vfr", False)), False) or (
                    ("cfr" in metadata_candidate or "is_cfr" in metadata_candidate)
                    and not _normalise_bool(metadata_candidate.get("cfr", metadata_candidate.get("is_cfr", True)), True)
                ):
                    raise ValueError("VFR VIDEO input is not supported; provide a CFR video.")
                metadata_rate = metadata_candidate.get("frame_rate", metadata_candidate.get("fps"))
                if isinstance(metadata_rate, (list, tuple)) or (isinstance(metadata_rate, np.ndarray) and metadata_rate.ndim > 0):
                    raise ValueError("VFR VIDEO input is not supported; provide a CFR video.")
        source_audio = _aligned_source_audio(video, components)
        images = getattr(components, "images", None)
        if not isinstance(images, torch.Tensor):
            images = torch.as_tensor(images)
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4 or images.shape[0] <= 0 or images.shape[1] <= 0 or images.shape[2] <= 0 or images.shape[-1] < 3:
            raise ValueError("VIDEO contains no decodable RGB frames.")
        images = _normalise_image_tensor(images[..., :3])
        source_frames, source_height, source_width = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
        rate_value = components.frame_rate
        if isinstance(rate_value, (list, tuple)) or (isinstance(rate_value, np.ndarray) and rate_value.ndim > 0):
            raise ValueError("VFR frame-rate data is not supported.")
        fps = _coerce_fps(rate_value, _DEFAULT_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("VIDEO frame rate must be positive (CFR input is required).")
        _timeline_info(
            "source ready: frames=%d; size=%dx%d; fps=%.3f; audio=%s",
            source_frames,
            source_width,
            source_height,
            fps,
            "available" if source_audio is not None else "none",
        )
        _timeline_info("stage 2/7: preparing timeline and output settings")
        color = _normalise_hex(fill_color)
        fit_mode = str(fit_mode or "letterbox").strip().lower()
        if fit_mode not in {"letterbox", "crop", "fill"}:
            fit_mode = "letterbox"
        multiple = max(1, _safe_int(multiple, 32, 1))
        width = -1 if _safe_int(width, -1) <= 0 else _safe_int(width, -1, 1)
        height = -1 if _safe_int(height, -1) <= 0 else _safe_int(height, -1, 1)
        output_width, output_height = _fit_dimensions(source_width, source_height, width, height, multiple)
        current_source_key = _source_fingerprint(video, images, source_metadata)
        current_fingerprint_kind = (
            "content"
            if source_metadata.get("timeline_version") is not None or source_metadata.get("timeline_json")
            else (
                "file"
                if _file_source_fingerprint(source_metadata)
                or str(source_metadata.get("source_fingerprint_kind") or "").strip().lower() == "file"
                else "content"
            )
        )
        # A loader can expose a different source window while keeping the
        # same file, frame count, and FPS.  Record that window when reliable
        # loader/file metadata is present so old frame indices are not
        # silently applied to a different part of the source.
        source_window_reliable = bool(
            source_metadata.get("source_filename")
            or source_metadata.get("loader_id")
            or source_metadata.get("source_start_frame") is not None
            or source_metadata.get("source_end_frame") is not None
        )
        current_source_start = _safe_int(
            source_metadata.get("source_start_frame", source_metadata.get("start_frame", 0)),
            0,
            0,
        ) if source_window_reliable else None
        raw_source_end = source_metadata.get("source_end_frame", source_metadata.get("end_frame", source_frames - 1))
        current_source_end = (
            max(0, source_frames - 1)
            if _safe_int(raw_source_end, source_frames - 1) < 0
            else _safe_int(raw_source_end, source_frames - 1, 0)
        ) if source_window_reliable else None

        # Node widgets take precedence over stale values in a saved descriptor
        # for the explicit In/Out controls; all other timeline state lives in
        # the canonical JSON.  When a workflow was saved before the JSON
        # widget existed, recover the last editor state from the temp manifest.
        node_id = str(getattr(getattr(cls, "hidden", None), "unique_id", "") or "").strip()
        current_identity_available = bool(_source_identity_hint(video, source_metadata))
        timeline_input = timeline_json
        if timeline_input is None or str(timeline_input).strip() == "":
            timeline_input = timeline if timeline is not None else edited_timeline
        if (timeline_input is None or str(timeline_input).strip() == "") and node_id:
            persisted = _load_state(node_id)
            persisted_source = str((persisted or {}).get("source_fingerprint") or "") if isinstance(persisted, Mapping) else ""
            persisted_kind = str((persisted or {}).get("source_fingerprint_kind") or "").strip().lower() if isinstance(persisted, Mapping) else ""
            comparable_persisted = (
                not persisted_source
                or not current_source_key
                or persisted_source == current_source_key
                or (
                    persisted_kind
                    and persisted_kind != current_fingerprint_kind
                    and not current_identity_available
                )
                or (
                    not persisted_kind
                    and len(persisted_source) != len(current_source_key)
                )
            )
            if persisted and comparable_persisted:
                timeline_input = persisted
        timeline = normalise_timeline(timeline_input, source_frames, fps)
        declared_source = str(timeline.get("source_fingerprint") or "").strip()
        declared_identity = str(timeline.get("source_identity") or "").strip()
        current_identity_hint = _source_identity_hint(video, source_metadata)
        declared_fingerprint_kind = str(timeline.get("source_fingerprint_kind") or "").strip().lower()
        if declared_fingerprint_kind not in {"file", "content"}:
            # Legacy descriptors did not record the namespace.  A reliable
            # file identity makes the distinction explicit; otherwise only
            # compare same-length content hashes to avoid treating a browser
            # file SHA-1 as a tensor SHA-256 mismatch.
            declared_fingerprint_kind = (
                current_fingerprint_kind
                if declared_source and len(declared_source) == len(current_source_key)
                else "unknown"
            )
        # A generic upstream VIDEO may have no file identity at execution time,
        # while the browser has learned one from a loader/wait-input cache.  Do
        # not treat that incomparable cache label as a source mismatch.  The
        # sampled content fingerprint below remains the authoritative check.
        current_identity = current_identity_hint or declared_identity or current_source_key
        declared_count = _safe_int(timeline.get("source_frame_count", 0), 0, 0)
        declared_fps = _safe_float(timeline.get("source_fps", 0.0), 0.0, 0.0)
        declared_source_start = timeline.get("source_start_frame")
        declared_source_end = timeline.get("source_end_frame")
        source_mismatch = bool(
            (declared_identity and current_identity_hint and declared_identity != current_identity_hint)
            or (
                declared_source
                and current_source_key
                and int(timeline.get("source_fingerprint_version", 0) or 0) >= _SOURCE_FINGERPRINT_VERSION
                and declared_fingerprint_kind == current_fingerprint_kind
                and declared_source != current_source_key
            )
            or (declared_count and declared_count != source_frames)
            or (declared_fps and abs(declared_fps - fps) > 1e-3)
            or (
                current_source_start is not None
                and declared_source_start is not None
                and _safe_int(declared_source_start, -1, -1) != current_source_start
            )
            or (
                current_source_end is not None
                and declared_source_end is not None
                and _safe_int(declared_source_end, -1, -1) != current_source_end
            )
        )
        if source_mismatch:
            # Clip frame indices belong to the source they were authored
            # against.  Silently clamping them onto a newly connected VIDEO
            # can produce a plausible-looking but entirely wrong edit.  Start
            # from the documented default placement and keep only node-level
            # display settings; the user can then re-run shot detection.
            preserved_output = timeline.get("output")
            timeline = normalise_timeline({}, source_frames, fps)
            if isinstance(preserved_output, Mapping):
                timeline["output"] = dict(preserved_output)
            timeline["source_mismatch"] = True
        saved_output = timeline.get("output") if isinstance(timeline.get("output"), Mapping) else {}
        # A timeline JSON copied between workflows may carry its canvas
        # settings even when the newer widgets are still at their defaults.
        # Treat neutral widget values as fallbacks, while explicit non-neutral
        # widget values remain authoritative.
        effective_width = width if _safe_int(width, 0, 0) > 0 else saved_output.get("width", width)
        effective_height = height if _safe_int(height, 0, 0) > 0 else saved_output.get("height", height)
        effective_multiple = multiple if _safe_int(multiple, 32, 1) != 32 or not saved_output.get("multiple") else saved_output.get("multiple", multiple)
        effective_fit_mode = fit_mode
        if str(fit_mode or "letterbox").strip().lower() == "letterbox" and saved_output.get("fit_mode"):
            effective_fit_mode = saved_output.get("fit_mode")
        effective_fill_color = fill_color
        if _normalise_hex(fill_color) == "#000000" and saved_output.get("fill_color"):
            effective_fill_color = saved_output.get("fill_color")
        effective_multiple = max(1, _safe_int(effective_multiple, 32, 1))
        effective_fit_mode = str(effective_fit_mode or "letterbox").strip().lower()
        if effective_fit_mode not in {"letterbox", "crop", "fill"}:
            effective_fit_mode = "letterbox"
        effective_fill_color = _normalise_hex(effective_fill_color)
        output_width, output_height = _fit_dimensions(source_width, source_height, effective_width, effective_height, effective_multiple)
        width, height, multiple, fit_mode, color = effective_width, effective_height, effective_multiple, effective_fit_mode, effective_fill_color
        timeline["threshold"] = _safe_float(shot_detect_threshold, timeline.get("threshold", 0.5), 0.0, 1.0)
        timeline["min_scene_seconds"] = _safe_float(shot_detect_min_scene_sec, timeline.get("min_scene_seconds", 0.0), 0.0)
        timeline["output"] = {
            "width": int(output_width),
            "height": int(output_height),
            "multiple": max(1, _safe_int(multiple, 1, 1)),
            "fit_mode": str(fit_mode or "letterbox"),
            "fill_color": color,
        }
        preview_in_value = _safe_int(in_frame, 0, 0)
        preview_out_value = _safe_int(out_frame, -1)
        if preview_in_value != 0 or preview_out_value != -1:
            duration_limit = int(timeline.get("duration_frames", 0))
            timeline["in_frame"] = max(0, min(preview_in_value, max(0, duration_limit - 1) if duration_limit > 0 else 0))
            out_value = preview_out_value
            timeline["out_frame"] = duration_limit if out_value < 0 else min(max(timeline["in_frame"] + 1, out_value), duration_limit)
        else:
            # Keep JSON In/Out but ensure they are valid after source changes.
            duration_limit = int(timeline.get("duration_frames", 0))
            timeline["in_frame"] = max(0, min(int(timeline.get("in_frame", 0)), max(0, duration_limit - 1) if duration_limit > 0 else 0))
            timeline["out_frame"] = min(max(timeline["in_frame"], int(timeline.get("out_frame", timeline.get("duration_frames", 0)))), int(timeline.get("duration_frames", 0)))
        if int(timeline.get("duration_frames", 0)) > 0 and int(timeline.get("out_frame", 0)) <= int(timeline.get("in_frame", 0)):
            timeline["out_frame"] = min(int(timeline["duration_frames"]), int(timeline["in_frame"]) + 1)
        # A standard VIDEO cannot carry a zero-frame batch.  An explicitly
        # cleared timeline therefore becomes one fill-color frame; this keeps
        # the empty-timeline rule useful while preserving a valid ComfyUI
        # value.  Normal timelines (including gaps between clips) are not
        # affected.
        if int(timeline.get("duration_frames", 0)) <= 0:
            timeline["duration_frames"] = 1
            timeline["in_frame"] = 0
            timeline["out_frame"] = 1
        _timeline_info(
            "timeline ready: duration=%d frames; in=%d; out=%d; output=%dx%d; fit=%s",
            int(timeline.get("duration_frames", 0)),
            int(timeline.get("in_frame", 0)),
            int(timeline.get("out_frame", 0)),
            output_width,
            output_height,
            fit_mode,
        )
        timeline["source_fingerprint"] = current_source_key
        timeline["source_fingerprint_kind"] = current_fingerprint_kind
        timeline["source_fingerprint_version"] = _SOURCE_FINGERPRINT_VERSION
        timeline["source_identity"] = current_identity
        timeline["source_frame_count"] = source_frames
        timeline["source_fps"] = float(fps)
        if current_source_start is not None:
            timeline["source_start_frame"] = int(current_source_start)
        if current_source_end is not None:
            timeline["source_end_frame"] = int(current_source_end)
        canonical = _canonical_json(timeline)
        _timeline_info("stage 3/7: preparing source preview cache")
        if node_id:
            _save_state(node_id, timeline)
            try:
                source_proxy_w, source_proxy_h = _proxy_dimensions(source_width, source_height)
                source_proxy = _resize_batch(images, source_proxy_w, source_proxy_h)
                # Include the connected VIDEO window and loader output shape
                # in the source-cache identity.  The same file can be loaded
                # with different start/end/FPS/resize settings; reusing one
                # source proxy for all of those variants would make the
                # timeline editor show stale frames or wrong shot boundaries.
                source_cache_key = hashlib.sha256(
                    _canonical_json(
                        {
                            "source": timeline["source_fingerprint"],
                            "source_identity": timeline.get("source_identity", ""),
                            "source_frame_count": source_frames,
                            "source_fps": fps,
                            "source_start_frame": current_source_start,
                            "source_end_frame": current_source_end,
                            "width": source_width,
                            "height": source_height,
                            "proxy_width": source_proxy_w,
                            "proxy_height": source_proxy_h,
                        }
                    ).encode("utf-8")
                ).hexdigest()
                source_entry = _cache_source_proxy(
                    node_id,
                    source_proxy,
                    fps,
                    {
                        "source": True,
                        "source_fingerprint": timeline["source_fingerprint"],
                        "source_fingerprint_kind": timeline.get("source_fingerprint_kind", "content"),
                        "source_identity": timeline.get("source_identity", ""),
                        "source_width": source_width,
                        "source_height": source_height,
                        "source_frame_count": source_frames,
                        "source_fps": fps,
                        "source_start_frame": current_source_start,
                        "source_end_frame": current_source_end,
                        "width": source_proxy_w,
                        "height": source_proxy_h,
                        "fps": fps,
                    },
                    source_cache_key,
                    audio=source_audio,
                )
                if source_entry is not None:
                    _timeline_info("source preview cache ready: %dx%d", source_proxy_w, source_proxy_h)
                else:
                    _timeline_info("source preview cache unavailable")
            except Exception as exc:
                _LOGGER.debug("Timeline source proxy generation skipped: %s", exc)
                _timeline_info("source preview cache unavailable: %s", exc)
        else:
            _timeline_info("source preview cache skipped: node id unavailable")

        # ``wait_for_input_cache`` is the explicit preview-build mode used by
        # the timeline window.  Render only the small proxy and interrupt
        # before materialising the full-resolution output tensor.
        if _normalise_bool(wait_for_input_cache, False):
            _timeline_info("stage 4/7: building preview cache before wait_for_input_cache interrupt")
            # Publish a bounded source proxy through the shared chain cache so
            # the same input can be consumed by the existing CineStyle preview
            # helpers.  The node-owned source/proxy entries above remain the
            # primary exact-frame path for this editor.
            if node_id:
                wait_frames = _downscale_preview_batch(images)
                _timeline_info("wait input cache: source frames prepared at %dx%d", int(wait_frames.shape[2]), int(wait_frames.shape[1]))
                _cache_wait_input(
                    node_id,
                    getattr(getattr(cls, "hidden", None), "prompt", None),
                    wait_frames,
                    fps,
                    {
                        "source_fingerprint": timeline.get("source_fingerprint", ""),
                        "source_fingerprint_kind": timeline.get("source_fingerprint_kind", "content"),
                        "source_identity": timeline.get("source_identity", ""),
                        "source_width": source_width,
                        "source_height": source_height,
                        "source_frame_count": source_frames,
                        "source_fps": fps,
                        "loaded_width": int(wait_frames.shape[2]) if isinstance(wait_frames, torch.Tensor) and wait_frames.ndim >= 3 else source_width,
                        "loaded_height": int(wait_frames.shape[1]) if isinstance(wait_frames, torch.Tensor) and wait_frames.ndim >= 3 else source_height,
                        "loaded_frame_count": source_frames,
                        "loaded_fps": fps,
                    },
                    source_audio,
                )
                _timeline_info("wait input cache ready")
            try:
                from comfy.model_management import InterruptProcessingException
            except (ImportError, AttributeError):
                InterruptProcessingException = None
            if InterruptProcessingException is not None:
                try:
                    proxy_w, proxy_h = _proxy_dimensions(output_width, output_height)
                    frame_progress = _TimelineProgress(
                        int(timeline.get("out_frame", 0)) - int(timeline.get("in_frame", 0)),
                        "preview frame rendering",
                    )
                    try:
                        proxy_images = render_timeline_frames(
                            images,
                            timeline,
                            proxy_w,
                            proxy_h,
                            fit_mode=fit_mode,
                            fill_color=color,
                            progress=frame_progress,
                        )
                    finally:
                        frame_progress.close()
                    proxy_audio = render_timeline_audio(source_audio, timeline, fps)
                    cache_key = _timeline_cache_key(timeline["source_fingerprint"], timeline, proxy_w, proxy_h, fit_mode, color, fps, True)
                    if node_id:
                        _cache_proxy(node_id, proxy_images, proxy_audio, fps, {"proxy": True, "cache_fingerprint": cache_key, "timeline_json": canonical, "width": proxy_w, "height": proxy_h}, cache_key)
                finally:
                    raise InterruptProcessingException()

        timeline_frame_count = int(timeline.get("out_frame", 0)) - int(timeline.get("in_frame", 0))
        _timeline_info("stage 4/7: rendering %d timeline frames at %dx%d", timeline_frame_count, output_width, output_height)
        frame_progress = _TimelineProgress(timeline_frame_count)
        try:
            output_images = render_timeline_frames(
                images,
                timeline,
                output_width,
                output_height,
                fit_mode=fit_mode,
                fill_color=color,
                progress=frame_progress,
            )
        finally:
            frame_progress.close()
        _timeline_info("frame rendering complete: %d frames", int(output_images.shape[0]))
        _timeline_info("stage 5/7: rendering timeline audio")
        output_audio = render_timeline_audio(
            source_audio,
            timeline,
            fps,
        )
        _timeline_info("audio processing complete: %s", "available" if output_audio is not None else "none")
        _timeline_info("stage 6/7: assembling output VIDEO and metadata")

        metadata = dict(source_metadata)
        try:
            audio_waveform = output_audio.get("waveform") if isinstance(output_audio, Mapping) else None
            audio_channels = int(audio_waveform.shape[1]) if isinstance(audio_waveform, torch.Tensor) and audio_waveform.ndim == 3 else 0
            audio_samples = int(audio_waveform.shape[-1]) if isinstance(audio_waveform, torch.Tensor) and audio_waveform.ndim >= 1 else 0
        except (AttributeError, TypeError, ValueError):
            audio_channels, audio_samples = 0, 0
        metadata.update(
            {
                "source_width": source_width,
                "source_height": source_height,
                "source_frame_count": source_frames,
                "source_fps": fps,
                "source_start_frame": int(current_source_start) if current_source_start is not None else None,
                "source_end_frame": int(current_source_end) if current_source_end is not None else None,
                "source_identity": current_identity,
                "source_fingerprint": current_source_key,
                "source_fingerprint_kind": current_fingerprint_kind,
                "source_fingerprint_version": _SOURCE_FINGERPRINT_VERSION,
                "timeline_version": _SCHEMA_VERSION,
                "timeline_json": canonical,
                "timeline_duration_frames": int(timeline.get("duration_frames", 0)),
                "timeline_in_frame": int(timeline.get("in_frame", 0)),
                "timeline_out_frame": int(timeline.get("out_frame", 0)),
                "loaded_width": output_width,
                "loaded_height": output_height,
                "loaded_frame_count": int(output_images.shape[0]),
                "frame_count": int(output_images.shape[0]),
                "loaded_duration": float(output_images.shape[0] / fps),
                # Keep the same convenient aliases exposed by CS Load Video;
                # downstream preview widgets commonly read either spelling.
                "width": output_width,
                "height": output_height,
                "output_width": output_width,
                "output_height": output_height,
                "frames": int(output_images.shape[0]),
                "fps": float(fps),
                "frame_rate": float(fps),
                "duration": float(output_images.shape[0] / fps),
                "loaded_fps": float(fps),
                "source_duration": _safe_float(metadata.get("source_duration", source_frames / fps), source_frames / fps, 0.0),
                "start_frame": _safe_int(metadata.get("start_frame", 0), 0, 0),
                "end_frame": _safe_int(metadata.get("end_frame", max(0, source_frames - 1)), max(0, source_frames - 1), 0),
                "source_filename": str(metadata.get("source_filename", "") or ""),
                "loader_id": str(metadata.get("loader_id", "") or ""),
                "audio_format": metadata.get("audio_format"),
                "has_audio": bool(audio_channels and audio_samples),
                "audio_sample_rate": int(output_audio.get("sample_rate", _DEFAULT_AUDIO_RATE)) if isinstance(output_audio, Mapping) else _DEFAULT_AUDIO_RATE,
                "audio_channels": audio_channels,
                "audio_sample_count": audio_samples,
                "audio_already_trimmed": True,
                "fit_mode": str(fit_mode),
                "fill_color": color,
                "multiple": max(1, _safe_int(multiple, 1, 1)),
                "cfr": True,
                "is_cfr": True,
                # Standard ComfyUI VIDEO carries RGB frames here.  The renderer
                # keeps an internal occupancy mask so transformed upper-track
                # content covers the lower track only inside its transformed
                # bounds; unoccupied canvas pixels remain visible below (or
                # use the configured fill colour when both tracks are empty).
                "track_compositing": "upper_over_lower_with_transform_mask",
                "transform_order": "fit_then_scale_flip_rotate_translate",
                "audio_compositing": "two_track_sum_clamped",
                "empty_timeline_fill": color,
                "timeline_source_mismatch": bool(source_mismatch),
            }
        )
        try:
            output_components = Types.VideoComponents(
                images=output_images,
                audio=output_audio,
                frame_rate=Fraction(fps).limit_denominator(1000),
                metadata=metadata,
            )
        except TypeError:
            output_components = Types.VideoComponents(
                images=output_images,
                audio=output_audio,
                frame_rate=Fraction(fps).limit_denominator(1000),
            )
        output_video = InputImpl.VideoFromComponents(output_components)
        try:
            output_video._cinestyle_runtime_metadata = dict(metadata)
            output_video._cinestyle_timeline_json = canonical
        except (AttributeError, TypeError):
            pass

        # A proxy is a derived resource and must not affect node execution if
        # encoding is unavailable.  Build it only when a node id is available.
        _timeline_info("stage 7/7: storing output preview cache")
        if node_id:
            try:
                proxy_w, proxy_h = _proxy_dimensions(output_width, output_height)
                proxy_images = _resize_batch(output_images, proxy_w, proxy_h)
                proxy_audio = output_audio
                cache_key = _timeline_cache_key(timeline["source_fingerprint"], timeline, proxy_w, proxy_h, fit_mode, color, fps, True)
                proxy_info = {
                    **metadata,
                    "proxy": True,
                    "cache_fingerprint": cache_key,
                    "width": int(proxy_w),
                    "height": int(proxy_h),
                    "loaded_width": int(proxy_w),
                    "loaded_height": int(proxy_h),
                    "loaded_frame_count": int(proxy_images.shape[0]),
                    "frame_count": int(proxy_images.shape[0]),
                }
                _cache_proxy(node_id, proxy_images, proxy_audio, fps, proxy_info, cache_key)
            except Exception as exc:
                _LOGGER.debug("Timeline proxy generation skipped: %s", exc)
                _timeline_info("output preview cache unavailable: %s", exc)
        else:
            _timeline_info("output preview cache skipped: node id unavailable")

        info = metadata
        _timeline_info(
            "stage 7/7: complete, output frames=%d; elapsed=%.2fs",
            int(output_images.shape[0]),
            time.perf_counter() - started_at,
        )
        return io.NodeOutput(output_video, output_images, int(output_images.shape[0]), output_audio, info, float(fps))

    @classmethod
    def fingerprint_inputs(cls, video: Any, timeline_json: Any = "", **kwargs: Any) -> str:
        try:
            components = video.get_components()
            images = components.images
            fps = _coerce_fps(components.frame_rate, _DEFAULT_FPS)
            metadata = _video_metadata(video)
            payload = {"source": _source_fingerprint(video, images, metadata), "timeline": canonical_timeline_json(timeline_json, int(images.shape[0]), fps), **{str(k): _json_safe(v) for k, v in kwargs.items()}}
            return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        except Exception:
            return ""

# Friendly aliases for code which used the shorter class name during early
# development.  The registered node remains the stable ``CS_Video_Timeline_Edit``.
CSVideoTimeEdit = CSVideoTimelineEdit
CSVideoTimeline = CSVideoTimelineEdit


# ---------------------------------------------------------------------------
# HTTP endpoints consumed by the Edit Timeline window


def _entry_for_node(node_id: Any, cache_key: str = "") -> dict[str, Any] | None:
    try:
        return _cache_store().get_preview_variant(str(node_id or ""), proxy=True, cache_fingerprint=str(cache_key or ""))
    except Exception:
        return None


def _source_entry_for_node(node_id: Any, cache_key: str = "") -> dict[str, Any] | None:
    try:
        store = _cache_store()
        if cache_key:
            # ``PreviewCacheStore.get_preview_variant`` only knows main/proxy;
            # source entries are addressed directly and checked by metadata.
            with store.lock:
                candidates = [
                    item
                    for item in store.entries.values()
                    if str(item.get("node_id") or "") == str(node_id or "")
                    and str(item.get("variant") or "") == "source"
                    and str((item.get("info") or {}).get("cache_fingerprint") or "") == str(cache_key)
                ]
                if candidates:
                    return max(candidates, key=lambda item: float(item.get("created") or 0.0))
            # An explicit cache key is a source identity, not merely a hint.
            # Falling back to a newer source entry here could render a frame
            # from a different VIDEO after the cache has been refreshed.
            return None
        base_key = store._base_key(str(node_id or ""), "source")
        with store.lock:
            latest_key = store.latest.get(base_key)
            latest = store.entries.get(latest_key) if latest_key else None
            if latest is not None:
                return latest
            candidates = [
                item
                for item in store.entries.values()
                if str(item.get("node_id") or "") == str(node_id or "")
                and str(item.get("variant") or "") == "source"
            ]
            return max(candidates, key=lambda item: float(item.get("created") or 0.0)) if candidates else None
    except Exception:
        return None


async def _timeline_source_info_route(request: Any) -> Any:
    """Expose the low-resolution source cache used by generic VIDEO inputs.

    A VIDEO produced by an arbitrary upstream node often has no filename that
    the browser can seek.  ``execute`` stores a small source-frame variant;
    this endpoint turns it into an on-demand MP4 (through the shared cache)
    and returns both its token and metadata.  The endpoint is intentionally
    read-only from the user's perspective and is safe to call repeatedly.
    """
    if web is None:
        return None
    node_id = str(request.query.get("node_id", "")).strip()
    cache_key = str(request.query.get("cache_key", "")).strip()
    if not node_id:
        return web.json_response({"error": "Missing node_id."}, status=400)
    entry = _source_entry_for_node(node_id, cache_key)
    if entry is None:
        return web.json_response({"error": "Timeline source cache is unavailable. Run the node once."}, status=404)
    try:
        entry = await asyncio.to_thread(_cache_store().ensure_video, entry)
    except Exception as exc:
        _LOGGER.debug("Unable to encode timeline source cache: %s", exc)
        entry = None
    path = Path(str((entry or {}).get("video_path") or (entry or {}).get("path") or "")) if entry else None
    if entry is None or path is None or not path.is_file():
        return web.json_response({"error": "Timeline source cache video is unavailable."}, status=404)
    token = str(entry.get("token") or "")
    return web.json_response(
        {
            "token": token,
            "video_url": f"/cinestyle/video-time-edit-source-video?token={token}",
            "info": dict(entry.get("info") or {}),
            "source_cache_key": str((entry.get("info") or {}).get("cache_fingerprint") or ""),
            "label": "Timeline source cache",
        }
    )


async def _timeline_source_video_route(request: Any) -> Any:
    if web is None:
        return None
    token = str(request.query.get("token", "")).strip()
    try:
        entry = _cache_store().get_token(token)
    except Exception:
        entry = None
    path = Path(str((entry or {}).get("video_path") or (entry or {}).get("path") or "")) if entry else None
    if entry is not None and (path is None or not path.is_file()):
        try:
            entry = await asyncio.to_thread(_cache_store().ensure_video, entry)
            path = Path(str((entry or {}).get("video_path") or (entry or {}).get("path") or "")) if entry else None
        except Exception:
            path = None
    if entry is None or path is None or not path.is_file():
        return web.json_response({"error": "Timeline source cache video not found."}, status=404)
    return web.FileResponse(path=path, headers={"Content-Type": "video/mp4", "Cache-Control": "no-store"})


async def _timeline_state_route(request: Any) -> Any:
    if web is None:
        return None
    # GET requests carry the node id in the query string and must not attempt
    # to parse an absent JSON body (aiohttp otherwise raises a 400 error).
    if str(getattr(request, "method", "GET") or "GET").upper() == "GET":
        node_id = str(request.query.get("node_id", "")).strip()
        if not node_id:
            return web.json_response({"error": "Missing node_id."}, status=400)
        value = _load_state(node_id)
        if value is None:
            return web.json_response({"error": "Timeline state not found."}, status=404)
        return web.json_response(value)
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON payload."}, status=400)
    node_id = str(payload.get("node_id", "")).strip() if isinstance(payload, Mapping) else ""
    if not node_id:
        return web.json_response({"error": "Missing node_id."}, status=400)
    timeline = payload.get("timeline", payload.get("timeline_json", payload)) if isinstance(payload, Mapping) else {}
    try:
        # Route-side validation is structural; source frame count can be filled
        # in by the next node execution when it is not supplied yet.
        parsed = _parse_json(timeline) if not isinstance(timeline, Mapping) else dict(timeline)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    accepted = _save_state(node_id, parsed, payload.get("revision") if isinstance(payload, Mapping) else None)
    if not accepted:
        return web.json_response({"ok": False, "version": _SCHEMA_VERSION, "stale": True}, status=409)
    return web.json_response({"ok": True, "version": _SCHEMA_VERSION})


async def _timeline_history_route(request: Any) -> Any:
    if web is None:
        return None
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON payload."}, status=400)
    if not isinstance(payload, Mapping):
        return web.json_response({"error": "JSON object required."}, status=400)
    node_id = str(payload.get("node_id", "")).strip()
    if not node_id:
        return web.json_response({"error": "Missing node_id."}, status=400)
    action = str(payload.get("action", "record") or "record").strip().lower()
    timeline_value = payload.get("timeline")
    try:
        timeline = None
        if action in {"init", "record", "reset"}:
            timeline = _parse_json(timeline_value) if not isinstance(timeline_value, Mapping) else dict(timeline_value)
        result = _update_history(node_id, action, timeline, str(payload.get("label", "")), payload.get("cursor"))
        return web.json_response(result)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def _timeline_preview_info_route(request: Any) -> Any:
    if web is None:
        return None
    node_id = str(request.query.get("node_id", "")).strip()
    cache_key = str(request.query.get("cache_key", "")).strip()
    entry = _entry_for_node(node_id, cache_key)
    expected_identity = str(request.query.get("source_identity", "")).strip()
    if entry is not None and expected_identity:
        entry_info = entry.get("info") if isinstance(entry.get("info"), Mapping) else {}
        actual_identity = str(
            entry_info.get("source_identity")
            or entry_info.get("source_fingerprint")
            or entry_info.get("input_signature")
            or ""
        ).strip()
        if actual_identity and actual_identity != expected_identity:
            entry = None
    if entry is None:
        warning = _PREVIEW_WARNINGS.get(node_id)
        payload: dict[str, Any] = {"error": "Timeline preview cache is unavailable. Run the node once."}
        if warning:
            payload["warning"] = warning
        return web.json_response(payload, status=404)
    path = Path(str(entry.get("video_path") or entry.get("path") or ""))
    if not path.is_file():
        try:
            entry = await asyncio.to_thread(_cache_store().ensure_video, entry)
        except Exception:
            entry = None
    if not entry:
        return web.json_response({"error": "Timeline preview video is unavailable."}, status=404)
    token = str(entry.get("token") or "")
    return web.json_response(
        {
            "token": token,
            "video_url": f"/cinestyle/video-time-edit-preview-video?token={token}",
            "info": dict(entry.get("info") or {}),
            "label": "Timeline proxy preview",
        }
    )


async def _timeline_preview_video_route(request: Any) -> Any:
    if web is None:
        return None
    token = str(request.query.get("token", "")).strip()
    try:
        entry = _cache_store().get_token(token)
    except Exception:
        entry = None
    path = Path(str((entry or {}).get("video_path") or (entry or {}).get("path") or "")) if entry else None
    if entry is None or path is None or not path.is_file():
        return web.json_response({"error": "Timeline preview video not found."}, status=404)
    return web.FileResponse(path=path, headers={"Content-Type": "video/mp4", "Cache-Control": "no-store"})


def _png_bytes(frame: torch.Tensor) -> bytes:
    from PIL import Image
    import io as py_io

    array = frame.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0).mul(255).round().to(torch.uint8).numpy()
    image = Image.fromarray(np.ascontiguousarray(array[..., :3]), mode="RGB")
    output = py_io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _decode_preview_source(payload: Mapping[str, Any]) -> tuple[torch.Tensor, float] | None:
    """Load source frames for an on-demand preview request.

    Cache tokens are preferred.  The file fallback is intentionally simple and
    deterministic; normal playback uses the encoded proxy route instead.
    """
    token = str(payload.get("source_token") or payload.get("token") or "").strip()

    def frames_from_entry(entry: Mapping[str, Any] | None, *, loader_store: Any = None) -> tuple[torch.Tensor, float] | None:
        """Read either an NPY cache entry or an encoded cache video."""
        if not entry:
            return None
        info = entry.get("info") if isinstance(entry.get("info"), Mapping) else {}
        fps = _safe_float(info.get("fps", info.get("loaded_fps", _DEFAULT_FPS)), _DEFAULT_FPS, 0.001)
        frames_path = Path(str(entry.get("frames_path") or ""))
        if frames_path.is_file():
            try:
                array = np.load(str(frames_path), mmap_mode="r", allow_pickle=False)
                if array.ndim == 4 and array.shape[0] > 0:
                    return _downscale_preview_batch(torch.from_numpy(np.array(array, copy=True)).float().div_(255.0)), fps
            except (OSError, ValueError, TypeError):
                pass
        # LoaderPreviewCache keeps only the encoded MP4.  Decode it once here;
        # exact single-frame requests can use its own random-seek helper, but a
        # timeline frame may refer to either source track, so the full proxy is
        # the portable fallback.
        video_path = Path(str(entry.get("video_path") or entry.get("path") or ""))
        if not video_path.is_file() and loader_store is not None:
            try:
                video_path = Path(str(entry.get("video_path") or ""))
            except Exception:
                video_path = Path("")
        if video_path.is_file() and av is not None:
            try:
                decoded: list[np.ndarray] = []
                with av.open(str(video_path), mode="r") as container:
                    stream = container.streams.video[0] if container.streams.video else None
                    if stream is None:
                        return None
                    rate = stream.average_rate or stream.guessed_rate
                    if rate:
                        fps = _safe_float(float(Fraction(rate)), fps, 0.001)
                    for decoded_frame in container.decode(stream):
                        decoded.append(decoded_frame.to_ndarray(format="rgb24"))
                if decoded:
                    return _downscale_preview_batch(torch.from_numpy(np.stack(decoded)).float().div_(255.0)), fps
            except Exception:
                pass
        return None
    if not token:
        # The node stores a low-resolution original-source variant alongside
        # the rendered timeline proxy.  This lets the editor request a frame
        # for unsaved edits without requiring a file path or a second VIDEO
        # transport.
        try:
            source_entry = _source_entry_for_node(str(payload.get("node_id", "")).strip(), str(payload.get("source_cache_key", "")).strip())
            loaded = frames_from_entry(source_entry)
            if loaded is not None:
                return loaded
        except Exception:
            pass
    if token:
        # Node-owned timeline cache.
        try:
            entry = _cache_store().get_token(token)
            loaded = frames_from_entry(entry)
            if loaded is not None:
                return loaded
        except Exception:
            pass
        # Shared loader/wait-input caches expose compatible ``get_token`` APIs.
        package = __name__.rsplit(".", 1)[0]
        for module_name, getter in (
            (f"{package}._py_loader_preview_cache", "get_loader_preview_cache"),
            (f"{package}._py_preview_cache", "get_wait_input_cache_store"),
        ):
            try:
                module = __import__(module_name, fromlist=["*"])
                store = getattr(module, getter)()
                if token.startswith("loader_preview:"):
                    entry = store.entry_for_token(token)
                else:
                    entry = store.get_token(token)
                loaded = frames_from_entry(entry, loader_store=store)
                if loaded is not None:
                    return loaded
            except Exception:
                continue
    source = str(payload.get("video_filename") or payload.get("video") or "").strip()
    if not source or av is None:
        return None
    try:
        path = folder_paths.get_annotated_filepath(source) if folder_paths.exists_annotated_filepath(source) else os.path.abspath(source)
        decoded: list[np.ndarray] = []
        with av.open(path, mode="r") as container:
            if not container.streams.video:
                return None
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.guessed_rate
            fps = float(Fraction(rate)) if rate else _DEFAULT_FPS
            for frame in container.decode(stream):
                decoded.append(frame.to_ndarray(format="rgb24"))
        if not decoded:
            return None
        source_array = np.stack(decoded)
        # A direct CS Load Video fallback must expose the same local frame
        # coordinate system as the connected VIDEO.  The shared loader cache
        # normally does this for us; when it is unavailable, reproduce its
        # selected inclusive range and CFR resampling here.
        start = max(0, min(_safe_int(payload.get("source_start_frame", 0), 0, 0), source_array.shape[0] - 1))
        requested_end = _safe_int(payload.get("source_end_frame", -1), -1)
        end = source_array.shape[0] - 1 if requested_end < 0 else min(requested_end, source_array.shape[0] - 1)
        end = max(start, end)
        selected = source_array[start : end + 1]
        target_fps = _safe_float(payload.get("source_target_fps", 0), 0.0, 0.0)
        if target_fps > 0 and abs(target_fps - fps) > 1e-6:
            if selected.shape[0] > 1:
                loaded_count = max(1, int(round(selected.shape[0] * target_fps / max(fps, 1e-6))))
                indices = np.rint(np.linspace(0, selected.shape[0] - 1, loaded_count)).astype(np.int64)
                selected = selected[indices]
            fps = target_fps
        preview_tensor = torch.from_numpy(np.ascontiguousarray(selected)).float().div_(255.0)
        return _downscale_preview_batch(preview_tensor), fps
    except Exception:
        return None


def _preview_entry_for_token(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, Any]:
    """Resolve a preview token without decoding its media payload."""
    expected_identity = str(payload.get("source_identity") or "").strip()
    expected_count = _safe_int(payload.get("source_frames", payload.get("source_frame_count", 0)), 0, 0)

    def accepted(entry: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if not entry:
            return None
        info = entry.get("info") if isinstance(entry.get("info"), Mapping) else {}
        actual_identity = str(
            info.get("source_identity") or info.get("source_fingerprint") or info.get("input_signature") or ""
        ).strip()
        actual_count = _safe_int(info.get("frames", info.get("loaded_frame_count", info.get("source_frame_count", 0))), 0, 0)
        if expected_identity and actual_identity and expected_identity != actual_identity:
            return None
        if expected_count > 0 and actual_count > 0 and expected_count != actual_count:
            return None
        return entry

    token = str(payload.get("source_token") or payload.get("token") or "").strip()
    if token:
        try:
            entry = _cache_store().get_token(token)
            entry = accepted(entry)
            if entry is not None:
                return entry, _cache_store()
        except Exception:
            pass
        package = __name__.rsplit(".", 1)[0]
        for module_name, getter in (
            (f"{package}._py_loader_preview_cache", "get_loader_preview_cache"),
            (f"{package}._py_preview_cache", "get_wait_input_cache_store"),
        ):
            try:
                module = __import__(module_name, fromlist=["*"])
                store = getattr(module, getter)()
                entry = store.entry_for_token(token) if token.startswith("loader_preview:") else store.get_token(token)
                entry = accepted(entry)
                if entry is not None:
                    return entry, store
            except Exception:
                continue
    node_id = str(payload.get("node_id", "")).strip()
    if node_id:
        entry = _source_entry_for_node(node_id, str(payload.get("source_cache_key", "")).strip())
        entry = accepted(entry)
        if entry is not None:
            try:
                return entry, _cache_store()
            except Exception:
                return entry, None
    return None, None


def _read_preview_frame(entry: Mapping[str, Any] | None, store: Any, frame_index: int) -> tuple[torch.Tensor, float] | None:
    """Read one cached source frame, keeping long-video previews bounded."""
    if not entry:
        return None
    info = entry.get("info") if isinstance(entry.get("info"), Mapping) else {}
    fps = _safe_float(info.get("fps", info.get("loaded_fps", _DEFAULT_FPS)), _DEFAULT_FPS, 0.001)
    target = max(0, int(frame_index))
    frames_path = Path(str(entry.get("frames_path") or ""))
    if frames_path.is_file():
        try:
            array = np.load(str(frames_path), mmap_mode="r", allow_pickle=False)
            if array.ndim != 4 or array.shape[0] <= 0:
                return None
            target = min(target, int(array.shape[0]) - 1)
            frame = torch.from_numpy(np.array(array[target], copy=True)).float().div_(255.0)
            return frame, fps
        except (OSError, ValueError, TypeError):
            pass
    token = str(entry.get("token") or "")
    if store is not None and token.startswith("loader_preview:"):
        try:
            # LoaderPreviewCacheStore accepts a bare token, whereas the
            # generic PreviewCacheStore API accepts a request payload.  Keep
            # both signatures working so exact scrubbing stays on the cache's
            # random-frame path instead of decoding the MP4.
            try:
                frame = store.decode_frame(token, target)
            except (AttributeError, TypeError):
                frame = store.decode_frame({"source_token": token}, target)
            if isinstance(frame, torch.Tensor):
                if frame.ndim == 4:
                    frame = frame[0]
                return _normalise_image_tensor(frame), fps
        except Exception:
            pass
    video_path = Path(str(entry.get("video_path") or entry.get("path") or ""))
    if video_path.is_file() and av is not None:
        try:
            with av.open(str(video_path), mode="r") as container:
                stream = container.streams.video[0] if container.streams.video else None
                if stream is None:
                    return None
                rate = stream.average_rate or stream.guessed_rate
                if rate:
                    fps = _safe_float(float(Fraction(rate)), fps, 0.001)
                # Seeking to a nearby keyframe avoids decoding the entire MP4
                # for every pointer move.  The fallback loop is bounded by the
                # GOP rather than the full clip in normal H.264 proxies.
                try:
                    container.seek(int(target / fps * av.time_base), stream=stream, any_frame=False, backward=True)
                except Exception:
                    pass
                target_time = target / max(fps, 0.001)
                for decoded in container.decode(stream):
                    try:
                        decoded_time = float(decoded.pts * decoded.time_base) if decoded.pts is not None and decoded.time_base is not None else None
                    except Exception:
                        decoded_time = None
                    if decoded_time is not None and decoded_time + (0.5 / max(fps, 0.001)) >= target_time:
                        return torch.from_numpy(decoded.to_ndarray(format="rgb24")).float().div_(255.0), fps
                    if decoded_time is None and target <= 0:
                        return torch.from_numpy(decoded.to_ndarray(format="rgb24")).float().div_(255.0), fps
            # A few containers omit PTS values.  Fall back to a bounded
            # sequential decode so the frame request still succeeds.
            with av.open(str(video_path), mode="r") as fallback_container:
                fallback_stream = fallback_container.streams.video[0] if fallback_container.streams.video else None
                if fallback_stream is not None:
                    for index, decoded in enumerate(fallback_container.decode(fallback_stream)):
                        if index >= target:
                            return torch.from_numpy(decoded.to_ndarray(format="rgb24")).float().div_(255.0), fps
        except Exception:
            return None
    return None


def _preview_source_dimensions(payload: Mapping[str, Any], info: Mapping[str, Any], frames: torch.Tensor | None = None) -> tuple[int, int]:
    """Resolve dimensions of the connected VIDEO for an exact preview.

    Loader preview metadata distinguishes original-file dimensions from the
    resized dimensions delivered by CS Load Video.  Prefer the delivered
    canvas, while generic VIDEO caches use ``source_width/source_height`` as
    their tensor dimensions.
    """
    # ``loaded_width/height`` are the dimensions actually delivered by a
    # shared loader cache.  They take precedence over the original loader
    # widget hints; otherwise a cached, already-resized source could be
    # resized a second time during exact-frame preview.
    loaded_width = _safe_int(info.get("loaded_width", 0), 0, 0)
    loaded_height = _safe_int(info.get("loaded_height", 0), 0, 0)
    if loaded_width > 0 and loaded_height > 0:
        return loaded_width, loaded_height

    output_hint_width = _safe_int(payload.get("source_output_width", 0), 0, 0)
    output_hint_height = _safe_int(payload.get("source_output_height", 0), 0, 0)
    if output_hint_width > 0 or output_hint_height > 0:
        base_width = _safe_int(
            info.get("source_width", info.get("width", payload.get("source_width", 1))),
            1,
            1,
        )
        base_height = _safe_int(
            info.get("source_height", info.get("height", payload.get("source_height", 1))),
            1,
            1,
        )
        return _fit_dimensions(
            base_width,
            base_height,
            output_hint_width,
            output_hint_height,
            _safe_int(payload.get("source_output_multiple", 1), 1, 1),
        )
    hinted_width = _safe_int(payload.get("source_width", 0), 0, 0)
    hinted_height = _safe_int(payload.get("source_height", 0), 0, 0)
    if hinted_width > 0 and hinted_height > 0:
        return hinted_width, hinted_height
    content_width = _safe_int(info.get("content_width", 0), 0, 0)
    content_height = _safe_int(info.get("content_height", 0), 0, 0)
    if content_width > 0 and content_height > 0:
        return content_width, content_height
    if _normalise_bool(info.get("source", False), False):
        preview_width = _safe_int(info.get("width", 0), 0, 0)
        preview_height = _safe_int(info.get("height", 0), 0, 0)
        if preview_width > 0 and preview_height > 0:
            return preview_width, preview_height
    width = _safe_int(info.get("source_width", info.get("width", 0)), 0, 0)
    height = _safe_int(info.get("source_height", info.get("height", 0)), 0, 0)
    if width > 0 and height > 0:
        return width, height
    if isinstance(frames, torch.Tensor) and frames.ndim >= 3:
        return int(frames.shape[2]), int(frames.shape[1])
    return 1, 1


def _decode_video_file_frames(path_value: str, payload: Mapping[str, Any] | None = None) -> tuple[torch.Tensor, float]:
    """Decode a video file to a small RGB batch for shot detection."""
    if av is None:
        raise RuntimeError("PyAV is unavailable for shot detection.")
    decoded: list[np.ndarray] = []
    with av.open(path_value, mode="r") as container:
        stream = container.streams.video[0] if container.streams.video else None
        if stream is None:
            return torch.empty((0, 1, 1, 3)), _DEFAULT_FPS
        rate = stream.average_rate or stream.guessed_rate
        fps = float(Fraction(rate)) if rate else _DEFAULT_FPS
        # Detection does not need full source resolution.  Keeping the longest
        # side near 640 substantially reduces the temporary batch while
        # preserving TransNet/fallback cut features.
        source_width, source_height = int(stream.width or 0), int(stream.height or 0)
        scale = min(1.0, 640.0 / max(1, source_width), 640.0 / max(1, source_height))
        target_width = max(1, int(round(source_width * scale))) if source_width else 640
        target_height = max(1, int(round(source_height * scale))) if source_height else 360
        for item in container.decode(stream):
            decoded.append(item.reformat(width=target_width, height=target_height, format="rgb24").to_ndarray())
    if not decoded:
        return torch.empty((0, 1, 1, 3)), fps
    array = np.stack(decoded)
    request = payload if isinstance(payload, Mapping) else {}
    start_value = _safe_int(request.get("source_start_frame", 0), 0, 0)
    end_value = _safe_int(request.get("source_end_frame", -1), -1)
    if start_value > 0 or end_value >= 0:
        start_value = min(start_value, array.shape[0] - 1)
        end_value = array.shape[0] - 1 if end_value < 0 else min(end_value, array.shape[0] - 1)
        end_value = max(start_value, end_value)
        array = array[start_value : end_value + 1]
    target_fps = _safe_float(request.get("source_target_fps", 0), 0.0, 0.0)
    if target_fps > 0 and abs(target_fps - fps) > 1e-6:
        if array.shape[0] > 1:
            loaded_count = max(1, int(round(array.shape[0] * target_fps / max(fps, 1e-6))))
            indices = np.rint(np.linspace(0, array.shape[0] - 1, loaded_count)).astype(np.int64)
            array = array[indices]
        fps = target_fps
    return torch.from_numpy(np.ascontiguousarray(array)).float().div_(255.0), fps


def _load_cached_preview_batch(entry: Mapping[str, Any]) -> tuple[torch.Tensor, float]:
    path = Path(str(entry.get("frames_path") or ""))
    if not path.is_file():
        raise FileNotFoundError("Cached source frames are unavailable.")
    array = np.load(str(path), mmap_mode="r", allow_pickle=False)
    if array.ndim != 4 or array.shape[0] <= 0:
        return torch.empty((0, 1, 1, 3)), _DEFAULT_FPS
    fps = _safe_float((entry.get("info") or {}).get("fps", _DEFAULT_FPS), _DEFAULT_FPS, 0.001)
    return torch.from_numpy(np.array(array, copy=True)).float().div_(255.0), fps


def _render_cached_timeline_frame(payload: Mapping[str, Any], frame_index: int, raw_timeline: Any) -> tuple[torch.Tensor, float] | None:
    """Render one timeline frame by reading only its active source frames."""
    entry, store = _preview_entry_for_token(payload)
    if entry is None:
        return None
    info = entry.get("info") if isinstance(entry.get("info"), Mapping) else {}
    source_count = _safe_int(info.get("frames", info.get("loaded_frame_count", 0)), 0, 0)
    frames_path = Path(str(entry.get("frames_path") or ""))
    if source_count <= 0 and frames_path.is_file():
        try:
            source_count = int(np.load(str(frames_path), mmap_mode="r", allow_pickle=False).shape[0])
        except Exception:
            source_count = 0
    if source_count <= 0:
        source_count = max(1, _safe_int(payload.get("source_frames", 1), 1, 1))
    fps = _safe_float(info.get("fps", info.get("loaded_fps", _DEFAULT_FPS)), _DEFAULT_FPS, 0.001)
    timeline = normalise_timeline(raw_timeline, source_count, fps)
    if int(timeline.get("duration_frames", 0)) <= 0:
        timeline["duration_frames"] = 1
        timeline["in_frame"] = 0
        timeline["out_frame"] = 1
    frame = min(max(0, int(frame_index)), max(0, int(timeline.get("duration_frames", 1)) - 1))
    output_config = timeline.get("output") if isinstance(timeline.get("output"), Mapping) else {}
    source_width, source_height = _preview_source_dimensions(payload, info)
    effective_width, effective_height, effective_multiple, mode, effective_color = _resolve_output_options(payload, output_config)
    width, height = _fit_dimensions(
        source_width,
        source_height,
        effective_width,
        effective_height,
        effective_multiple,
    )
    # Scrubbing is an interactive operation.  Keep exact-frame responses in
    # the same bounded proxy domain as playback; final node execution still
    # renders the requested full-resolution canvas.
    width, height = _proxy_dimensions(width, height)
    mode = str(mode or "letterbox").lower()
    if mode not in {"letterbox", "crop", "fill"}:
        mode = "letterbox"
    color = _hex_rgb(effective_color)
    clips = [item for item in _timeline_collection(timeline.get("clips")) if isinstance(item, Mapping)]
    rendered = torch.empty((height, width, 3), dtype=torch.float32)
    rendered[..., 0], rendered[..., 1], rendered[..., 2] = color
    for clip in _active_clips(clips, frame, 0) + _active_clips(clips, frame, 1):
        source_index = int(clip.get("source_start", 0)) + frame - int(clip.get("timeline_start", 0))
        source_frame = _read_preview_frame(entry, store, source_index)
        if source_frame is None:
            continue
        source_tensor, detected_fps = source_frame
        fps = detected_fps
        local_clip = dict(clip)
        local_clip.update({"source_start": 0, "source_end": 1, "timeline_start": frame, "timeline_end": frame + 1})
        rgba = _render_clip_rgba(source_tensor.unsqueeze(0), local_clip, frame, width, height, mode, color, normalized=True)
        if rgba is None:
            continue
        rgb, alpha = rgba
        rendered = rgb * alpha[..., None] + rendered * (1.0 - alpha[..., None])
    return rendered, fps


async def _timeline_preview_frame_route(request: Any) -> Any:
    if web is None:
        return None
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON payload."}, status=400)
    node_id = str(payload.get("node_id", "")).strip() if isinstance(payload, Mapping) else ""
    frame_index = max(0, _safe_int(payload.get("frame", 0) if isinstance(payload, Mapping) else 0, 0))
    has_timeline = isinstance(payload, Mapping) and any(
        key in payload for key in ("timeline", "timeline_json", "clips", "video_clips", "videoClips", "audio_clips", "audioClips")
    )
    persisted_timeline = _load_state(node_id) if node_id else None
    if not has_timeline and persisted_timeline:
        # The editor often sends only ``node_id`` and the playhead.  Re-render
        # from the latest persisted descriptor so transform edits are visible
        # immediately instead of showing an old proxy frame.
        has_timeline = True
    # Keep the rendered proxy as a fallback even when a caller supplies an
    # unsaved timeline.  The source variant may have been evicted after a
    # restart, in which case returning the last proxy frame is preferable to a
    # hard 404.
    entry = _entry_for_node(node_id, str(payload.get("cache_key", "")) if isinstance(payload, Mapping) else "")
    if isinstance(payload, Mapping) and any(
        key in payload for key in ("timeline", "timeline_json", "clips", "video_clips", "videoClips", "audio_clips", "audioClips")
    ):
        raw_timeline = payload.get("timeline_json")
        if raw_timeline is None or (isinstance(raw_timeline, str) and not raw_timeline.strip()):
            raw_timeline = payload.get("timeline", payload)
    else:
        raw_timeline = persisted_timeline or {}
    if has_timeline:
        if isinstance(payload, Mapping) and str(payload.get("source_token") or payload.get("token") or "").strip():
            checked_entry, _ = _preview_entry_for_token(payload)
            if checked_entry is None and (
                str(payload.get("source_identity") or "").strip()
                or _safe_int(payload.get("source_frames", payload.get("source_frame_count", 0)), 0, 0) > 0
            ) and not str(payload.get("video_filename") or "").strip():
                return web.json_response({"error": "The supplied source preview token does not match this VIDEO."}, status=409)
        # The common scrub/edit path needs at most one source frame per active
        # video track.  Read those frames directly from the mmap/cache instead
        # of copying or decoding the entire source batch on every pointer move.
        try:
            single = await asyncio.to_thread(_render_cached_timeline_frame, payload if isinstance(payload, Mapping) else {}, frame_index, raw_timeline)
            if single is not None:
                rendered_frame, _ = single
                return web.Response(body=_png_bytes(rendered_frame), content_type="image/png", headers={"Cache-Control": "no-store"})
        except Exception as exc:
            _LOGGER.debug("Single-frame timeline cache path unavailable: %s", exc)
    try:
        if entry is not None and Path(str(entry.get("frames_path") or "")).is_file():
            array = np.load(str(entry["frames_path"]), mmap_mode="r", allow_pickle=False)
            if array.ndim != 4 or array.shape[0] == 0:
                raise ValueError("Cached timeline frames are empty.")
            proxy_info = entry.get("info") if isinstance(entry.get("info"), Mapping) else {}
            proxy_start = _safe_int(proxy_info.get("timeline_in_frame", proxy_info.get("in_frame", 0)), 0, 0)
            local_index = max(0, frame_index - proxy_start)
            local_index = min(local_index, int(array.shape[0]) - 1)
            frame = torch.from_numpy(np.array(array[local_index], copy=True)).float().div_(255.0)
            return web.Response(body=_png_bytes(frame), content_type="image/png", headers={"Cache-Control": "no-store"})
    except (OSError, ValueError, KeyError) as exc:
        return web.json_response({"error": str(exc)}, status=500)
    # A request can render a single frame before the proxy has finished.  The
    # supplied timeline is canonicalised against the decoded source and then
    # narrowed to a one-frame In/Out window.
    source_result = await asyncio.to_thread(_decode_preview_source, payload if isinstance(payload, Mapping) else {})
    if source_result is None:
        return web.json_response({"error": "Timeline preview cache is unavailable."}, status=404)
    source_frames, fps = source_result
    source_count = int(source_frames.shape[0])
    try:
        timeline = normalise_timeline(raw_timeline, source_count, fps)
        if int(timeline.get("duration_frames", 0)) <= 0:
            timeline["duration_frames"] = 1
            timeline["in_frame"] = 0
            timeline["out_frame"] = 1
        requested = min(max(0, frame_index), max(0, int(timeline.get("duration_frames", 0)) - 1))
        timeline["in_frame"] = requested
        timeline["out_frame"] = requested + 1
        output_config = timeline.get("output") if isinstance(timeline.get("output"), Mapping) else {}
        # ``source_width``/``source_height`` are optional preview hints.  A
        # browser may send zero for an unknown value; in that case use the
        # decoded cache dimensions rather than accidentally reducing the
        # canvas to the 1-pixel minimum accepted by the coercion helpers.
        preview_source_width, preview_source_height = _preview_source_dimensions(
            payload if isinstance(payload, Mapping) else {},
            {},
            source_frames,
        )
        effective_width, effective_height, effective_multiple, preview_fit_mode, preview_color = _resolve_output_options(
            payload if isinstance(payload, Mapping) else {},
            output_config,
        )
        output_width, output_height = _fit_dimensions(
            preview_source_width,
            preview_source_height,
            effective_width,
            effective_height,
            effective_multiple,
        )
        output_width, output_height = _proxy_dimensions(output_width, output_height)
        rendered = render_timeline_frames(
            source_frames,
            timeline,
            output_width,
            output_height,
            fit_mode=preview_fit_mode,
            fill_color=preview_color,
        )
        if rendered.shape[0] == 0:
            raise ValueError("Requested timeline frame is outside the timeline.")
        return web.Response(body=_png_bytes(rendered[0]), content_type="image/png", headers={"Cache-Control": "no-store"})
    except (OSError, ValueError, KeyError, IndexError) as exc:
        return web.json_response({"error": str(exc)}, status=500)


async def _timeline_shot_detect_route(request: Any) -> Any:
    if web is None:
        return None
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON payload."}, status=400)
    if not isinstance(payload, Mapping):
        return web.json_response({"error": "JSON object required."}, status=400)
    node_id = str(payload.get("node_id", "")).strip()
    requested_source_cache_key = str(payload.get("source_cache_key", "")).strip()
    entry = _source_entry_for_node(node_id, requested_source_cache_key) if node_id else None
    if entry is None and node_id and not requested_source_cache_key:
        entry = _source_entry_for_node(node_id)
    try:
        source_token = str(payload.get("source_token") or "").strip()
        if source_token:
            token_entry, _ = _preview_entry_for_token(payload)
            if token_entry is None and not str(payload.get("video_filename") or "").strip():
                return web.json_response({"error": "The supplied source preview token does not match this VIDEO."}, status=409)
            decoded = await asyncio.to_thread(_decode_preview_source, payload)
            if decoded is not None:
                frames, fps = decoded
            elif str(payload.get("video_filename") or "").strip():
                source = str(payload.get("video_filename") or "").strip()
                path = folder_paths.get_annotated_filepath(source) if folder_paths.exists_annotated_filepath(source) else os.path.abspath(source)
                frames, fps = await asyncio.to_thread(_decode_video_file_frames, path, payload)
            else:
                return web.json_response({"error": "The supplied source preview token is unavailable."}, status=404)
        elif entry is not None and Path(str(entry.get("frames_path") or "")).is_file():
            frames, fps = await asyncio.to_thread(_load_cached_preview_batch, entry)
        else:
            source = str(payload.get("video_filename", "")).strip()
            if not source:
                return web.json_response({"error": "Provide a cached node or video_filename."}, status=400)
            path = folder_paths.get_annotated_filepath(source) if folder_paths.exists_annotated_filepath(source) else os.path.abspath(source)
            # Decode off the aiohttp event loop; TransNet and the fallback can
            # take seconds on a long source and should not freeze the editor.
            frames, fps = await asyncio.to_thread(_decode_video_file_frames, path, payload)
        supplied_predictions = payload.get("predictions")
        if not isinstance(supplied_predictions, Sequence) or isinstance(supplied_predictions, (str, bytes)):
            supplied_predictions = None
        threshold = _safe_float(payload.get("shot_detect_threshold", payload.get("threshold", 0.5)), 0.5, 0.0, 1.0)
        min_scene_seconds = _safe_float(payload.get("shot_detect_min_scene_sec", payload.get("min_scene_seconds", 0.0)), 0.0, 0.0)
        if supplied_predictions is not None:
            shots = detect_shots(frames, fps, threshold, min_scene_seconds, predictions=supplied_predictions)
            detector = "transnetv2-predictions"
        else:
            # TensorFlow/TransNetV2 runs outside the event loop and falls back
            # to frame differences when the optional runtime is absent.
            shots, detector = await asyncio.to_thread(
                detect_shots_preferred,
                frames,
                fps,
                threshold,
                min_scene_seconds,
            )
        return web.json_response(
            {
                "shots": shots,
                "fps": fps,
                "frames": int(frames.shape[0]),
                "detector": detector,
                "shot_detect_threshold": threshold,
                "shot_detect_min_scene_sec": min_scene_seconds,
            }
        )
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


# Route aliases retained for clients that used the shorter prototype names.
_timeline_preview_route = _timeline_preview_frame_route
_timeline_shot_route = _timeline_shot_detect_route


class CineStyleVideoTimelineEditExtension(ComfyExtension):
    async def on_load(self) -> None:
        global _ROUTE_REGISTERED
        if _ROUTE_REGISTERED:
            return
        try:
            from server import PromptServer

            server_instance = getattr(PromptServer, "instance", None)
        except Exception:
            server_instance = None
        if server_instance is None:
            return
        if server_instance is not None:
            server_instance.routes.post("/cinestyle/video-time-edit-state")(_timeline_state_route)
            server_instance.routes.get("/cinestyle/video-time-edit-state")(_timeline_state_route)
            server_instance.routes.post("/cinestyle/video-time-edit-history")(_timeline_history_route)
            server_instance.routes.get("/cinestyle/video-time-edit-source-info")(_timeline_source_info_route)
            server_instance.routes.get("/cinestyle/video-time-edit-source-video")(_timeline_source_video_route)
            server_instance.routes.get("/cinestyle/video-time-edit-preview-info")(_timeline_preview_info_route)
            server_instance.routes.get("/cinestyle/video-time-edit-preview-video")(_timeline_preview_video_route)
            server_instance.routes.post("/cinestyle/video-time-edit-proxy")(_timeline_proxy_route)
            server_instance.routes.get("/cinestyle/video-time-edit-proxy-progress")(_timeline_proxy_progress_route)
            server_instance.routes.post("/cinestyle/video-time-edit-preview")(_timeline_preview_frame_route)
            server_instance.routes.post("/cinestyle/video-time-edit-shot-detect")(_timeline_shot_detect_route)
        _ROUTE_REGISTERED = True

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CSVideoTimelineEdit]


async def comfy_entrypoint() -> CineStyleVideoTimelineEditExtension:
    return CineStyleVideoTimelineEditExtension()


# Compatibility for older custom-node scanners.
NODE_CLASS_MAPPINGS = {_NODE_ID: CSVideoTimelineEdit}
NODE_DISPLAY_NAME_MAPPINGS = {_NODE_ID: "CS Video Timeline Edit"}
WEB_DIRECTORY = "./web"
