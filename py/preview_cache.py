"""Shared frame-cache primitives used by CineStyle preview nodes.

The store is deliberately stateful per namespace. Nodes may share this code,
but never share an index, token, or cache file namespace.
"""

from __future__ import annotations

import math
import os
import tempfile
import threading
import time
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from PIL import Image


class PreviewCacheStore:
    def __init__(self, namespace: str, root: str | os.PathLike | None = None, max_entries: int = 8, max_bytes: int = 4 * 1024**3):
        self.namespace = str(namespace).strip() or "preview"
        base = Path(root) if root is not None else Path(tempfile.gettempdir()) / "cinestyle_preview_cache"
        self.root = base / self.namespace
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_entries = max(1, int(max_entries))
        self.max_bytes = max(1, int(max_bytes))
        self.entries: dict[str, dict[str, Any]] = {}
        self.lock = threading.RLock()

    def _key(self, node_id: Any, variant: str = "") -> str:
        key = str(node_id or "").strip()
        suffix = str(variant or "").strip()
        return f"{key}:{suffix}" if suffix else key

    def _token(self) -> str:
        return f"{self.namespace}:{uuid.uuid4().hex}"

    @staticmethod
    def _as_uint8(frames: Any) -> np.ndarray:
        if isinstance(frames, torch.Tensor):
            value = frames[..., :3].detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).contiguous().numpy()
        else:
            value = np.asarray(frames)
            if value.ndim != 4:
                raise ValueError("Preview frames must have shape [frames, height, width, channels].")
            value = value[..., :3]
            if np.issubdtype(value.dtype, np.floating):
                value = np.clip(value, 0.0, 1.0) * 255.0
            value = np.asarray(np.rint(value), dtype=np.uint8)
        if value.ndim != 4 or value.shape[0] == 0 or value.shape[-1] < 3:
            raise ValueError("Preview frames contain no usable RGB frames.")
        return np.ascontiguousarray(value[..., :3])

    @staticmethod
    def _safe_fps(fps: Any) -> float:
        try:
            value = float(fps)
            if math.isfinite(value) and value > 0:
                return value
        except (TypeError, ValueError, OverflowError):
            pass
        return 24.0

    @staticmethod
    def _remove_files(entry: dict[str, Any] | None) -> None:
        if not entry:
            return
        for name in ("path", "video_path", "frames_path"):
            try:
                Path(str(entry.get(name) or "")).unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _prepare_audio(audio: Any) -> dict[str, Any] | None:
        if not isinstance(audio, dict) or not isinstance(audio.get("waveform"), torch.Tensor):
            return None
        waveform = audio["waveform"]
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        try:
            sample_rate = int(audio.get("sample_rate", 0) or 0)
        except (TypeError, ValueError):
            return None
        if waveform.ndim != 3 or waveform.numel() == 0 or sample_rate <= 0:
            return None
        return {"waveform": waveform.detach().to(device="cpu", dtype=torch.float32).contiguous(), "sample_rate": sample_rate}

    def _encode_video(self, path: Path, frames: np.ndarray, fps: float, audio: Any = None, progress: Any = None) -> None:
        height, width = map(int, frames.shape[1:3])
        encoded_width = width + (width % 2)
        encoded_height = height + (height % 2)
        rate = Fraction(self._safe_fps(fps)).limit_denominator(1000)
        with av.open(str(path), mode="w", format="mp4") as container:
            try:
                stream = container.add_stream("libx264", rate=rate)
                stream.options = {"preset": "ultrafast", "crf": "20"}
            except (av.error.FFmpegError, ValueError):
                stream = container.add_stream("mpeg4", rate=rate)
            stream.width = encoded_width
            stream.height = encoded_height
            stream.pix_fmt = "yuv420p"
            audio_stream = None
            prepared_audio = self._prepare_audio(audio)
            if prepared_audio is not None:
                channels = int(prepared_audio["waveform"].shape[1])
                layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(channels, "stereo")
                audio_stream = container.add_stream("aac", rate=int(prepared_audio["sample_rate"]), layout=layout)
            for array in frames:
                if encoded_width != width or encoded_height != height:
                    array = np.pad(array, ((0, encoded_height - height), (0, encoded_width - width), (0, 0)), mode="edge")
                video_frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)
                if progress is not None:
                    progress.update()
            for packet in stream.encode():
                container.mux(packet)
            if audio_stream is not None and prepared_audio is not None:
                audio_frame = av.AudioFrame.from_ndarray(
                    prepared_audio["waveform"][0].numpy(),
                    format="fltp",
                    layout=audio_stream.layout.name,
                )
                audio_frame.sample_rate = int(prepared_audio["sample_rate"])
                for packet in audio_stream.encode(audio_frame):
                    container.mux(packet)
                for packet in audio_stream.encode():
                    container.mux(packet)

    def _evict(self) -> None:
        evicted: list[dict[str, Any]] = []
        with self.lock:
            while len(self.entries) > self.max_entries or (len(self.entries) > 1 and sum(int(item.get("size_bytes") or 0) for item in self.entries.values()) > self.max_bytes):
                key, entry = next(iter(self.entries.items()))
                self.entries.pop(key, None)
                evicted.append(entry)
        for entry in evicted:
            self._remove_files(entry)

    def put(self, node_id: Any, frames: Any, fps: Any, *, variant: str = "", encode_video: bool = True, info: dict[str, Any] | None = None, audio: Any = None, progress: Any = None) -> dict[str, Any]:
        key = self._key(node_id, variant)
        if not key:
            raise ValueError("Preview cache node_id is required.")
        array = self._as_uint8(frames)
        safe_fps = self._safe_fps(fps)
        token = self._token()
        frames_path = self.root / f"{uuid.uuid4().hex}.npy"
        video_path = self.root / f"{uuid.uuid4().hex}.mp4" if encode_video else None
        np.save(frames_path, array, allow_pickle=False)
        try:
            if video_path is not None:
                self._encode_video(video_path, array, safe_fps, audio=audio, progress=progress)
        except Exception:
            frames_path.unlink(missing_ok=True)
            if video_path is not None:
                video_path.unlink(missing_ok=True)
            raise
        base_info = {
            "frames": int(array.shape[0]),
            "width": int(array.shape[2]),
            "height": int(array.shape[1]),
            "fps": safe_fps,
            "duration": float(array.shape[0]) / safe_fps,
            "audio_format": None,
        }
        if info:
            base_info.update(info)
        entry = {
            "token": token,
            "namespace": self.namespace,
            "node_id": str(node_id),
            "variant": str(variant or ""),
            "path": str(video_path) if video_path is not None else "",
            "video_path": str(video_path) if video_path is not None else "",
            "frames_path": str(frames_path),
            "created": time.time(),
            "size_bytes": int(frames_path.stat().st_size + (video_path.stat().st_size if video_path is not None else 0)),
            "info": base_info,
            "audio": audio,
        }
        with self.lock:
            previous = self.entries.pop(key, None)
            self.entries[key] = entry
        self._remove_files(previous)
        self._evict()
        return entry

    def get_node(self, node_id: Any, variant: str = "") -> dict[str, Any] | None:
        with self.lock:
            return self.entries.get(self._key(node_id, variant))

    def get_token(self, token: Any) -> dict[str, Any] | None:
        resolved = str(token or "").strip()
        if not resolved or not resolved.startswith(f"{self.namespace}:"):
            return None
        with self.lock:
            return next((entry for entry in self.entries.values() if entry.get("token") == resolved), None)

    def remove(self, node_id: Any, variant: str = "") -> None:
        with self.lock:
            entry = self.entries.pop(self._key(node_id, variant), None)
        self._remove_files(entry)

    def clear_node(self, node_id: Any) -> None:
        key = str(node_id or "").strip()
        with self.lock:
            keys = [item for item, entry in self.entries.items() if entry.get("node_id") == key]
            removed = [self.entries.pop(item, None) for item in keys]
        for entry in removed:
            self._remove_files(entry)

    def ensure_video(self, entry: dict[str, Any], progress: Any = None) -> dict[str, Any] | None:
        if not entry:
            return None
        video_path = Path(str(entry.get("video_path") or entry.get("path") or ""))
        if video_path.is_file():
            return entry
        frames_path = Path(str(entry.get("frames_path") or ""))
        try:
            frames = np.load(str(frames_path), mmap_mode="r", allow_pickle=False)
            if frames.ndim != 4 or frames.shape[0] == 0:
                return None
            video_path = self.root / f"{uuid.uuid4().hex}.mp4"
            self._encode_video(video_path, np.asarray(frames), float(entry.get("info", {}).get("fps", 24.0) or 24.0), audio=entry.get("audio"), progress=progress)
            entry["path"] = str(video_path)
            entry["video_path"] = str(video_path)
            entry["size_bytes"] = int(frames_path.stat().st_size + video_path.stat().st_size)
            return entry
        except (OSError, ValueError, KeyError, av.error.FFmpegError):
            return None

    def read_frame(self, token: Any, frame_index: int) -> np.ndarray:
        entry = self.get_token(token)
        if entry is None:
            raise ValueError(f"Preview cache token is unavailable for namespace '{self.namespace}'.")
        try:
            frames = np.load(str(entry["frames_path"]), mmap_mode="r", allow_pickle=False)
            target = int(frame_index)
            if target < 0 or target >= int(frames.shape[0]):
                raise ValueError(f"Frame {target} is outside the cached input.")
            return np.array(frames[target:target + 1], copy=True)
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError("Cached preview frames are unavailable.") from exc

    @staticmethod
    def _resolve_file(value: str) -> str:
        import folder_paths
        source = str(value or "").strip()
        if folder_paths.exists_annotated_filepath(source):
            return folder_paths.get_annotated_filepath(source)
        path = Path(os.path.expandvars(os.path.expanduser(source))).resolve()
        if path.is_file():
            return str(path)
        raise ValueError(f"Preview source file not found: {source}")

    def decode_frame(self, payload: dict[str, Any], frame_index: int) -> torch.Tensor:
        token = str(payload.get("source_token") or "").strip()
        if token:
            frame = self.read_frame(token, frame_index)
            return torch.from_numpy(frame).to(torch.float32).div_(255.0)
        source = str(payload.get("video") or "").strip()
        if not source:
            raise ValueError(f"Preview cache token is required for namespace '{self.namespace}'.")
        source_path = self._resolve_file(source)
        if str(payload.get("source_kind") or "").lower() == "image" or Path(source_path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".avif"}:
            if int(frame_index) != 0:
                raise ValueError("An image source contains one frame.")
            with Image.open(source_path) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
            return torch.from_numpy(array).unsqueeze(0).to(torch.float32).div_(255.0)
        target = max(0, int(frame_index))
        with av.open(source_path, mode="r") as container:
            if not container.streams.video:
                raise ValueError("Preview source contains no video stream.")
            for index, decoded in enumerate(container.decode(container.streams.video[0])):
                if index == target:
                    array = decoded.to_ndarray(format="rgb24")
                    return torch.from_numpy(array).unsqueeze(0).to(torch.float32).div_(255.0)
        raise ValueError(f"Frame {target} is outside the preview source.")
