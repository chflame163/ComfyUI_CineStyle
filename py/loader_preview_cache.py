"""Shared preview-only MP4 cache for CS Load Video outputs.

The cache is intentionally separate from node-owned processing caches.  It is
keyed by the loader node id plus an effective-input signature, and the media is
never used as a source for final node execution.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch

import folder_paths


_CACHE_VERSION = 1
_MAX_CACHE_PIXELS = 1_000_000
_MIN_BITRATE = 3_000_000
_AUDIO_RATE = 16_000
_MAX_ENTRIES = 32
_MAX_BYTES = 4 * 1024**3
_BUILD_SLOTS = threading.BoundedSemaphore(2)
_TOKEN_PREFIX = "loader_preview:"
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".avif"}
_LOGGER = logging.getLogger("CineStyleLoaderPreview")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else float(default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _round_multiple(value: float, multiple: int) -> int:
    multiple = max(1, int(multiple))
    return max(multiple, int(math.floor(float(value) / multiple + 0.5)) * multiple)


def _fingerprint(path: str) -> str:
    stat = os.stat(path)
    payload = f"{Path(path).resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _resolve_file(value: str) -> str:
    source = str(value or "").strip()
    if not source:
        raise FileNotFoundError(source)
    if folder_paths.exists_annotated_filepath(source):
        return str(Path(folder_paths.get_annotated_filepath(source)).resolve())
    path = Path(os.path.expandvars(os.path.expanduser(source))).resolve()
    if path.is_file():
        return str(path)
    raise FileNotFoundError(source)


def _probe(path: str) -> dict[str, Any]:
    with av.open(path, mode="r") as container:
        if not container.streams.video:
            raise ValueError("Source contains no video stream.")
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate or Fraction(24, 1)
        fps = float(Fraction(rate))
        frames = int(stream.frames or 0)
        duration = float(container.duration / av.time_base) if container.duration else 0.0
        if frames <= 0 and duration > 0 and fps > 0:
            frames = max(1, int(round(duration * fps)))
        if duration <= 0 and frames > 0 and fps > 0:
            duration = frames / fps
        return {
            "source_width": int(stream.width or 0),
            "source_height": int(stream.height or 0),
            "source_fps": fps,
            "source_frame_count": frames,
            "source_duration": duration,
            "has_audio": bool(container.streams.audio),
            "audio_format": container.streams.audio[0].codec.name if container.streams.audio and container.streams.audio[0].codec else None,
        }


def aspect_locked_dimensions(
    source_width: Any,
    source_height: Any,
    requested_width: Any = 0,
    requested_height: Any = 0,
    multiple: Any = 32,
) -> tuple[int, int]:
    source_width = max(1, _safe_int(source_width, 1))
    source_height = max(1, _safe_int(source_height, 1))
    aspect = source_width / float(source_height)
    multiple = max(1, _safe_int(multiple, 32))
    requested_width = _safe_int(requested_width, 0)
    requested_height = _safe_int(requested_height, 0)
    if requested_width > 0:
        desired_width = float(requested_width)
        desired_height = desired_width / aspect
    elif requested_height > 0:
        desired_height = float(requested_height)
        desired_width = desired_height * aspect
    else:
        desired_width = float(source_width)
        desired_height = float(source_height)

    # Width and height are rounded together. Evaluating the neighbouring
    # multiples keeps the final ratio closer to the source than independently
    # rounding width first and deriving height afterwards.
    width_candidates = {
        _round_multiple(desired_width, multiple),
        max(multiple, int(math.floor(desired_width / multiple)) * multiple),
        max(multiple, int(math.ceil(desired_width / multiple)) * multiple),
    }
    height_candidates = {
        _round_multiple(desired_height, multiple),
        max(multiple, int(math.floor(desired_height / multiple)) * multiple),
        max(multiple, int(math.ceil(desired_height / multiple)) * multiple),
    }
    width, height = min(
        ((candidate_width, candidate_height) for candidate_width in width_candidates for candidate_height in height_candidates),
        key=lambda pair: (
            abs((pair[0] / float(pair[1])) - aspect),
            abs(pair[0] - desired_width) / max(1.0, desired_width)
            + abs(pair[1] - desired_height) / max(1.0, desired_height),
        ),
    )
    # The backend owns the aspect-ratio rule.  A stale/hand-edited workflow
    # cannot request an independently stretched canvas.
    return max(1, width), max(1, height)


def _effective_dimensions(probe: dict[str, Any], payload: dict[str, Any]) -> tuple[int, int]:
    return aspect_locked_dimensions(
        probe.get("source_width"),
        probe.get("source_height"),
        payload.get("width"),
        payload.get("height"),
        payload.get("multiple"),
    )


def _cache_dimensions(width: int, height: int) -> tuple[int, int, int, int]:
    """Return content and even encoded dimensions for a half-size/1MP proxy."""
    area = max(1, int(width) * int(height))
    scale = 1.0 if area <= _MAX_CACHE_PIXELS else min(0.5, math.sqrt(_MAX_CACHE_PIXELS / float(area)))
    target_width = max(2.0, width * scale)
    target_height = max(2.0, height * scale)
    target_aspect = float(width) / max(1.0, float(height))

    def even_candidates(value: float) -> set[int]:
        lower = max(2, int(math.floor(value / 2.0)) * 2)
        return {lower, lower + 2}

    candidates = [
        (candidate_width, candidate_height)
        for candidate_width in even_candidates(target_width)
        for candidate_height in even_candidates(target_height)
    ]
    bounded = [pair for pair in candidates if pair[0] * pair[1] <= _MAX_CACHE_PIXELS]
    if bounded:
        candidates = bounded
    encoded_width, encoded_height = min(
        candidates,
        key=lambda pair: (
            abs((pair[0] / float(pair[1])) - target_aspect),
            max(0, pair[0] * pair[1] - _MAX_CACHE_PIXELS) / float(_MAX_CACHE_PIXELS),
            abs(pair[0] - target_width) + abs(pair[1] - target_height),
        ),
    )
    # Content is encoded at the same even dimensions, so no implicit edge
    # padding changes the advertised aspect ratio.
    return encoded_width, encoded_height, encoded_width, encoded_height


def _normalise_request(payload: dict[str, Any]) -> dict[str, Any]:
    loader_id = str(payload.get("loader_id") or "").strip()
    filename = str(payload.get("video") or payload.get("filename") or "").strip()
    if not loader_id or not filename:
        raise ValueError("loader_id and video are required.")
    source_path = _resolve_file(filename)
    probe = _probe(source_path)
    source_fps = max(0.001, _safe_float(probe.get("source_fps"), 24.0))
    source_count = max(1, _safe_int(probe.get("source_frame_count"), 1))
    start = max(0, min(_safe_int(payload.get("start_frame"), 0), source_count - 1))
    requested_end = _safe_int(payload.get("end_frame"), -1)
    end = source_count - 1 if requested_end < 0 else min(requested_end, source_count - 1)
    end = max(start, end)
    target_fps = _safe_float(payload.get("fps"), 0.0)
    target_fps = target_fps if target_fps > 0 else source_fps
    loaded_width, loaded_height = _effective_dimensions(probe, payload)
    loaded_count = max(1, int(round((end - start + 1) * target_fps / source_fps)))
    cache_width, cache_height, encoded_width, encoded_height = _cache_dimensions(loaded_width, loaded_height)
    source_fingerprint = _fingerprint(source_path)
    signature_payload = {
        "version": _CACHE_VERSION,
        "source_fingerprint": source_fingerprint,
        "start_frame": start,
        "end_frame": end,
        "loaded_fps": round(target_fps, 6),
        "loaded_width": loaded_width,
        "loaded_height": loaded_height,
        "loaded_frame_count": loaded_count,
        "cache_width": cache_width,
        "cache_height": cache_height,
        "encoded_width": encoded_width,
        "encoded_height": encoded_height,
        "audio_rate": _AUDIO_RATE if probe.get("has_audio") else 0,
    }
    signature = hashlib.sha1(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "loader_id": loader_id,
        "filename": filename,
        "source_path": source_path,
        "source_fingerprint": source_fingerprint,
        "source_fps": source_fps,
        "source_frame_count": source_count,
        "source_width": int(probe["source_width"]),
        "source_height": int(probe["source_height"]),
        "source_duration": float(probe["source_duration"]),
        "start_frame": start,
        "end_frame": end,
        "loaded_fps": target_fps,
        "loaded_frame_count": loaded_count,
        "loaded_width": loaded_width,
        "loaded_height": loaded_height,
        "cache_width": cache_width,
        "cache_height": cache_height,
        "encoded_width": encoded_width,
        "encoded_height": encoded_height,
        "multiple": max(1, _safe_int(payload.get("multiple"), 32)),
        "has_audio": bool(probe.get("has_audio")),
        "audio_rate": _AUDIO_RATE if probe.get("has_audio") else 0,
        "signature": signature,
        "cache_version": _CACHE_VERSION,
    }


def _audio_preview(path: str, start_seconds: float, duration: float) -> np.ndarray | None:
    if duration <= 0:
        return None
    try:
        with av.open(path, mode="r") as container:
            if not container.streams.audio:
                return None
            stream = container.streams.audio[0]
            resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=_AUDIO_RATE)
            chunks: list[np.ndarray] = []
            for frame in container.decode(stream):
                for converted in resampler.resample(frame):
                    array = converted.to_ndarray(format="fltp")
                    array = np.asarray(array, dtype=np.float32)
                    if array.ndim == 1:
                        array = array[None, :]
                    if array.ndim != 2 or array.shape[1] <= 0:
                        continue
                    chunks.append(array[:1])
            tail = resampler.resample(None)
            for converted in tail:
                array = np.asarray(converted.to_ndarray(format="fltp"), dtype=np.float32)
                if array.ndim == 1:
                    array = array[None, :]
                if array.ndim == 2 and array.shape[1] > 0:
                    chunks.append(array[:1])
            if not chunks:
                return None
            values = np.concatenate(chunks, axis=1)
            begin = max(0, int(round(start_seconds * _AUDIO_RATE)))
            end = min(values.shape[1], begin + max(1, int(round(duration * _AUDIO_RATE))))
            if end <= begin:
                return None
            return np.ascontiguousarray(values[:, begin:end])
    except (OSError, TypeError, AttributeError, ValueError, av.error.FFmpegError):
        return None


class LoaderPreviewCache:
    def __init__(self) -> None:
        self.root = Path(folder_paths.get_temp_directory()) / "cinestyle_loader_preview"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.entries: dict[tuple[str, str], dict[str, Any]] = {}
        self.jobs: dict[tuple[str, str], dict[str, Any]] = {}

    @staticmethod
    def _safe_loader_id(loader_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(loader_id or "").strip()) or "loader"

    def _directory(self, loader_id: str) -> Path:
        path = self.root / self._safe_loader_id(loader_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _manifest_path(self, request: dict[str, Any]) -> Path:
        return self._directory(request["loader_id"]) / f"{request['signature']}.json"

    def _entry_from_manifest(self, path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            entry = dict(value) if isinstance(value, dict) else None
            if not entry or not Path(str(entry.get("video_path") or "")).is_file():
                return None
            return entry
        except (OSError, ValueError, TypeError):
            return None

    def _find_ready(self, request: dict[str, Any]) -> dict[str, Any] | None:
        key = (request["loader_id"], request["signature"])
        with self.lock:
            entry = self.entries.get(key)
        if entry is not None and Path(str(entry.get("video_path") or "")).is_file():
            return entry
        manifest = self._manifest_path(request)
        entry = self._entry_from_manifest(manifest)
        if entry is not None:
            with self.lock:
                self.entries[key] = entry
        return entry

    def entry_for_signature(self, loader_id: str, signature: str) -> dict[str, Any] | None:
        loader_key = str(loader_id or "").strip()
        signature_key = str(signature or "").strip()
        if not loader_key or not signature_key:
            return None
        key = (loader_key, signature_key)
        with self.lock:
            entry = self.entries.get(key)
        if entry is not None and Path(str(entry.get("video_path") or "")).is_file():
            return entry
        manifest = self.root / self._safe_loader_id(loader_key) / f"{signature_key}.json"
        entry = self._entry_from_manifest(manifest)
        if entry is not None:
            with self.lock:
                self.entries[key] = entry
        return entry

    @staticmethod
    def _public_entry(entry: dict[str, Any], *, status: str = "ready", progress: int = 100, stage: str = "ready") -> dict[str, Any]:
        info = dict(entry.get("info") or {})
        token = str(entry.get("token") or "")
        return {
            "status": status,
            "progress": int(progress),
            "stage": stage,
            "loader_id": str(entry.get("loader_id") or ""),
            "signature": str(entry.get("signature") or ""),
            "token": token,
            "video_url": f"/cinestyle/loader-preview-cache-video?token={token}" if token else "",
            "info": info,
        }

    def ensure(self, payload: dict[str, Any], *, start_build: bool = True) -> dict[str, Any]:
        request = _normalise_request(payload)
        ready = self._find_ready(request)
        if ready is not None:
            return self._public_entry(ready)
        key = (request["loader_id"], request["signature"])
        if not start_build:
            return {
                "status": "missing",
                "progress": 0,
                "stage": "missing",
                "loader_id": request["loader_id"],
                "signature": request["signature"],
                "info": self._request_info(request),
            }
        with self.lock:
            job = self.jobs.get(key)
            if job is None or job.get("status") in {"failed", "cancelled"}:
                job = {
                    "status": "queued",
                    "progress": 0,
                    "stage": "queued",
                    "loader_id": request["loader_id"],
                    "signature": request["signature"],
                    "info": self._request_info(request),
                }
                self.jobs[key] = job
                threading.Thread(target=self._build, args=(request, key), daemon=True).start()
        return dict(job)

    @staticmethod
    def _request_info(request: dict[str, Any]) -> dict[str, Any]:
        return {
            "source_filename": request["filename"],
            "source_fingerprint": request["source_fingerprint"],
            "source_fps": request["source_fps"],
            "source_frame_count": request["source_frame_count"],
            "source_width": request["source_width"],
            "source_height": request["source_height"],
            "start_frame": request["start_frame"],
            "end_frame": request["end_frame"],
            "loaded_fps": request["loaded_fps"],
            "loaded_frame_count": request["loaded_frame_count"],
            "loaded_width": request["loaded_width"],
            "loaded_height": request["loaded_height"],
            "width": request["encoded_width"],
            "height": request["encoded_height"],
            "content_width": request["cache_width"],
            "content_height": request["cache_height"],
            "frames": request["loaded_frame_count"],
            "fps": request["loaded_fps"],
            "duration": request["loaded_frame_count"] / max(0.001, request["loaded_fps"]),
            "cache_width": request["cache_width"],
            "cache_height": request["cache_height"],
            "encoded_width": request["encoded_width"],
            "encoded_height": request["encoded_height"],
            "preview_only": True,
            "audio_preview_rate": request["audio_rate"],
            "has_audio": request["has_audio"],
        }

    def _set_job(self, key: tuple[str, str], **values: Any) -> None:
        with self.lock:
            self.jobs.setdefault(key, {}).update(values)

    def _build(self, request: dict[str, Any], key: tuple[str, str]) -> None:
        acquired = False
        try:
            self._set_job(key, status="queued", stage="waiting", progress=0)
            _BUILD_SLOTS.acquire()
            acquired = True
            _LOGGER.info(
                "[CS Load Video] preview cache start: loader=%s signature=%s source=%s",
                request["loader_id"], request["signature"], request["filename"],
            )
            self._set_job(key, status="running", stage="decoding", progress=1)
            indices = np.rint(np.linspace(request["start_frame"], request["end_frame"], request["loaded_frame_count"])).astype(np.int64)
            frames: list[np.ndarray] = []
            with av.open(request["source_path"], mode="r") as container:
                stream = container.streams.video[0]
                requested_index = 0
                for source_index, decoded in enumerate(container.decode(stream)):
                    while requested_index < len(indices) and int(indices[requested_index]) == source_index:
                        frames.append(
                            decoded.reformat(
                                width=request["cache_width"],
                                height=request["cache_height"],
                                format="rgb24",
                            ).to_ndarray()
                        )
                        requested_index += 1
                    if requested_index >= len(indices):
                        break
                    if source_index % 4 == 0:
                        progress = min(70, int(5 + source_index * 65 / max(1, request["end_frame"] + 1)))
                        self._set_job(key, progress=progress, stage="decoding")
            if not frames:
                raise ValueError("No preview frames were decoded.")
            self._set_job(key, status="running", stage="audio", progress=72)
            audio = _audio_preview(
                request["source_path"],
                request["start_frame"] / request["source_fps"],
                len(frames) / request["loaded_fps"],
            ) if request["has_audio"] else None
            _LOGGER.info(
                "[CS Load Video] preview cache audio: loader=%s %s",
                request["loader_id"], "mono/16k AAC" if audio is not None else "none",
            )
            self._set_job(key, status="running", stage="encoding", progress=80)
            directory = self._directory(request["loader_id"])
            final_path = directory / f"{request['signature']}.mp4"
            temporary = directory / f".{request['signature']}.{uuid.uuid4().hex}.tmp.mp4"
            self._encode_video(temporary, np.stack(frames, axis=0), request, audio, key)
            os.replace(temporary, final_path)
            actual_info = self._request_info(request)
            actual_info["frames"] = len(frames)
            actual_info["loaded_frame_count"] = len(frames)
            actual_info["duration"] = len(frames) / max(0.001, request["loaded_fps"])
            entry = {
                "token": f"{_TOKEN_PREFIX}{uuid.uuid4().hex}",
                "loader_id": request["loader_id"],
                "signature": request["signature"],
                "video_path": str(final_path),
                "created": time.time(),
                "info": actual_info,
            }
            entry["info"]["audio_format"] = "aac" if audio is not None else None
            manifest = self._manifest_path(request)
            temporary_manifest = manifest.with_name(f".{manifest.name}.{uuid.uuid4().hex}.tmp")
            temporary_manifest.write_text(json.dumps(entry, ensure_ascii=True, separators=(",", ":")), encoding="utf-8")
            os.replace(temporary_manifest, manifest)
            with self.lock:
                self.entries[key] = entry
            self._set_job(key, **self._public_entry(entry))
            _LOGGER.info(
                "[CS Load Video] preview cache ready: loader=%s signature=%s size=%dx%d frames=%d fps=%.3f",
                request["loader_id"], request["signature"], request["encoded_width"], request["encoded_height"],
                request["loaded_frame_count"], request["loaded_fps"],
            )
            self._evict()
        except Exception as exc:
            for candidate in self._directory(request["loader_id"]).glob(f".{request['signature']}.*.tmp*"):
                try:
                    candidate.unlink(missing_ok=True)
                except OSError:
                    pass
            self._set_job(key, status="failed", progress=0, stage="failed", error=str(exc), info=self._request_info(request))
            _LOGGER.exception(
                "[CS Load Video] preview cache failed: loader=%s signature=%s",
                request.get("loader_id"), request.get("signature"),
            )
        finally:
            if acquired:
                _BUILD_SLOTS.release()

    def _encode_video(self, path: Path, frames: np.ndarray, request: dict[str, Any], audio: np.ndarray | None, key: tuple[str, str]) -> None:
        rate = Fraction(request["loaded_fps"]).limit_denominator(1000)
        with av.open(str(path), mode="w", format="mp4", options={"movflags": "+faststart"}) as container:
            stream = container.add_stream("libx264", rate=rate)
            stream.options = {"preset": "ultrafast", "tune": "zerolatency"}
            stream.width = request["encoded_width"]
            stream.height = request["encoded_height"]
            stream.pix_fmt = "yuv420p"
            stream.bit_rate = _MIN_BITRATE
            stream.codec_context.max_b_frames = 0
            audio_stream = None
            if audio is not None and audio.size > 0:
                audio_stream = container.add_stream("aac", rate=_AUDIO_RATE, layout="mono", bit_rate=64000)
            total = max(1, len(frames))
            for index, array in enumerate(frames):
                if array.shape[0] != request["encoded_height"] or array.shape[1] != request["encoded_width"]:
                    array = np.pad(
                        array,
                        ((0, request["encoded_height"] - array.shape[0]), (0, request["encoded_width"] - array.shape[1]), (0, 0)),
                        mode="edge",
                    )
                frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
                self._set_job(key, progress=min(98, 80 + int(index * 17 / total)), stage="encoding")
            for packet in stream.encode():
                container.mux(packet)
            if audio_stream is not None and audio is not None:
                audio_frame = av.AudioFrame.from_ndarray(audio, format="fltp", layout="mono")
                audio_frame.sample_rate = _AUDIO_RATE
                for packet in audio_stream.encode(audio_frame):
                    container.mux(packet)
                for packet in audio_stream.encode():
                    container.mux(packet)

    def progress(self, loader_id: str, signature: str) -> dict[str, Any]:
        request_key = (str(loader_id or "").strip(), str(signature or "").strip())
        with self.lock:
            job = self.jobs.get(request_key)
        if job is not None:
            return dict(job)
        return {"status": "missing", "progress": 0, "stage": "missing", "loader_id": request_key[0], "signature": request_key[1]}

    def entry_for_token(self, token: str) -> dict[str, Any] | None:
        value = str(token or "").strip()
        if not value.startswith(_TOKEN_PREFIX):
            return None
        with self.lock:
            for entry in self.entries.values():
                if entry.get("token") == value:
                    return entry
        for manifest in self.root.glob("*/[0-9a-f]*.json"):
            entry = self._entry_from_manifest(manifest)
            if entry is not None and entry.get("token") == value:
                return entry
        return None

    def decode_frame(self, token: str, frame_index: int) -> torch.Tensor:
        entry = self.entry_for_token(token)
        if entry is None:
            raise ValueError("Loader preview cache is unavailable.")
        target = max(0, int(frame_index))
        with av.open(str(entry["video_path"]), mode="r") as container:
            stream = container.streams.video[0]
            for index, decoded in enumerate(container.decode(stream)):
                if index == target:
                    array = decoded.to_ndarray(format="rgb24")
                    return torch.from_numpy(array).unsqueeze(0).to(torch.float32).div_(255.0)
        raise ValueError(f"Frame {target} is outside the loader preview cache.")

    def _evict(self) -> None:
        manifests = sorted(self.root.glob("*/[0-9a-f]*.json"), key=lambda item: item.stat().st_mtime if item.exists() else 0)
        total = 0
        valid: list[tuple[Path, Path, int]] = []
        for manifest in manifests:
            entry = self._entry_from_manifest(manifest)
            if entry is None:
                continue
            video_path = Path(str(entry["video_path"]))
            size = video_path.stat().st_size + manifest.stat().st_size
            total += size
            valid.append((manifest, video_path, size))
        while len(valid) > _MAX_ENTRIES or total > _MAX_BYTES:
            manifest, video_path, size = valid.pop(0)
            total -= size
            try:
                video_path.unlink(missing_ok=True)
                manifest.unlink(missing_ok=True)
            except OSError:
                pass


_STORE: LoaderPreviewCache | None = None


def get_loader_preview_cache() -> LoaderPreviewCache:
    global _STORE
    if _STORE is None:
        _STORE = LoaderPreviewCache()
    return _STORE


async def loader_preview_ensure_route(request):
    from aiohttp import web

    try:
        payload = await request.json()
        raw_start_build = payload.get("start_build", True)
        start_build = (
            str(raw_start_build).strip().lower() not in {"0", "false", "no", "off"}
            if isinstance(raw_start_build, str)
            else bool(raw_start_build)
        )
        return web.json_response(get_loader_preview_cache().ensure(payload, start_build=start_build))
    except (OSError, ValueError, KeyError, TypeError, av.error.FFmpegError) as exc:
        return web.json_response({"status": "failed", "error": str(exc)}, status=400)


async def loader_preview_progress_route(request):
    from aiohttp import web

    loader_id = str(request.query.get("loader_id") or "").strip()
    signature = str(request.query.get("signature") or "").strip()
    return web.json_response(get_loader_preview_cache().progress(loader_id, signature))


async def loader_preview_video_route(request):
    from aiohttp import web

    entry = get_loader_preview_cache().entry_for_token(request.query.get("token", ""))
    path = Path(str((entry or {}).get("video_path") or "")) if entry else None
    if path is None or not path.is_file():
        return web.json_response({"error": "Loader preview cache not found."}, status=404)
    return web.FileResponse(path=path, headers={"Content-Type": "video/mp4", "Cache-Control": "no-store"})
