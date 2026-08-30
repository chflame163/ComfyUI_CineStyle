"""Shared frame-cache primitives used by CineStyle preview nodes.

The store is deliberately stateful per namespace. Node-owned stores keep their
indexes and tokens isolated; the explicit ``wait_input`` store is shared by
the preview nodes and keyed by an upstream input-chain fingerprint.
"""

from __future__ import annotations

import math
import hashlib
import json
import os
import re
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


_WAIT_INPUT_CACHE_VERSION = 1
_WAIT_INPUT_CACHE_NAMESPACE = "wait_input"
_WAIT_INPUT_CACHE_STORE = None
_WAIT_INPUT_CACHE_STORE_LOCK = threading.RLock()
_WAIT_INPUT_CACHE_ROUTES_REGISTERED = False


def _prompt_node(prompt: Any, node_id: Any) -> dict[str, Any] | None:
    if not isinstance(prompt, dict):
        return None
    return prompt.get(str(node_id)) or prompt.get(node_id)


def _prompt_link(value: Any) -> tuple[str, Any] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    if not isinstance(value[0], (str, int)) or isinstance(value[0], bool):
        return None
    upstream_id = str(value[0] or "").strip()
    if not upstream_id:
        return None
    try:
        output_slot: Any = int(value[1])
    except (TypeError, ValueError, OverflowError):
        output_slot = str(value[1] or "").strip()
    return upstream_id, output_slot


def _prompt_link_for(prompt: Any, value: Any) -> tuple[str, Any] | None:
    link = _prompt_link(value)
    if link is None or _prompt_node(prompt, link[0]) is None:
        return None
    return link


def _normalise_input_chain(chain: Any) -> dict[str, Any] | None:
    if not isinstance(chain, dict):
        return None
    # The consumer widget name is intentionally not part of the source key:
    # IMAGE and VIDEO consumers must be able to reuse the same upstream cache.
    input_name = "media"
    roots: list[dict[str, Any]] = []
    for value in chain.get("roots") or []:
        if not isinstance(value, dict):
            continue
        node_id = str(value.get("node_id") or "").strip()
        if not node_id:
            continue
        try:
            output_slot: Any = int(value.get("output_slot", 0))
        except (TypeError, ValueError, OverflowError):
            output_slot = str(value.get("output_slot") or "").strip()
        roots.append({"input_name": "media", "node_id": node_id, "output_slot": output_slot})
    roots.sort(key=lambda item: (item["input_name"], item["node_id"], str(item["output_slot"])))
    nodes = sorted({str(value).strip() for value in chain.get("nodes") or [] if str(value).strip()})
    edges: list[dict[str, Any]] = []
    for value in chain.get("edges") or []:
        if not isinstance(value, dict):
            continue
        node_id = str(value.get("node_id") or "").strip()
        upstream_id = str(value.get("upstream_node_id") or "").strip()
        if not node_id or not upstream_id:
            continue
        try:
            output_slot: Any = int(value.get("output_slot", 0))
        except (TypeError, ValueError, OverflowError):
            output_slot = str(value.get("output_slot") or "").strip()
        edges.append({
            "node_id": node_id,
            "input_name": str(value.get("input_name") or ""),
            "upstream_node_id": upstream_id,
            "output_slot": output_slot,
        })
    edges.sort(key=lambda item: (item["node_id"], item["input_name"], item["upstream_node_id"], str(item["output_slot"])))
    return {
        "version": _WAIT_INPUT_CACHE_VERSION,
        "input_name": input_name,
        "roots": roots,
        "nodes": nodes,
        "edges": edges,
        "complete": bool(chain.get("complete", True)) and bool(input_name and roots and nodes),
    }


def input_chain_fingerprint(chain: Any) -> str:
    """Hash a canonical graph path used by wait-for-input preview caches."""
    normalised = _normalise_input_chain(chain)
    if normalised is None or not normalised["complete"]:
        return ""
    payload = json.dumps(normalised, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_input_chain(prompt: Any, node_id: Any, input_names: tuple[str, ...] | list[str]) -> dict[str, Any] | None:
    """Collect the selected input node and every linked upstream node."""
    target = _prompt_node(prompt, node_id)
    if not isinstance(target, dict):
        return None
    inputs = target.get("inputs") if isinstance(target.get("inputs"), dict) else {}
    root = None
    selected_name = ""
    for name in input_names:
        link = _prompt_link_for(prompt, inputs.get(name))
        if link is not None:
            selected_name = str(name)
            root = link
            break
    if root is None:
        return None
    roots = [{"input_name": "media", "node_id": root[0], "output_slot": root[1]}]
    queue = [root[0]]
    visited: set[str] = set()
    nodes: set[str] = set()
    edges: list[dict[str, Any]] = []
    complete = True
    while queue:
        current_id = str(queue.pop(0))
        if current_id in visited:
            continue
        visited.add(current_id)
        nodes.add(current_id)
        current = _prompt_node(prompt, current_id)
        if not isinstance(current, dict):
            complete = False
            continue
        current_inputs = current.get("inputs") if isinstance(current.get("inputs"), dict) else {}
        for input_name in sorted(current_inputs, key=str):
            link = _prompt_link_for(prompt, current_inputs.get(input_name))
            if link is None:
                continue
            edges.append({
                "node_id": current_id,
                "input_name": str(input_name),
                "upstream_node_id": link[0],
                "output_slot": link[1],
            })
            queue.append(link[0])
    return _normalise_input_chain({
        "input_name": "media",
        "roots": roots,
        "nodes": sorted(nodes),
        "edges": edges,
        "complete": complete,
    })


class PreviewCacheStore:
    def __init__(self, namespace: str, root: str | os.PathLike | None = None, max_entries: int = 8, max_bytes: int = 4 * 1024**3):
        self.namespace = str(namespace).strip() or "preview"
        base = Path(root) if root is not None else Path(tempfile.gettempdir()) / "cinestyle_preview_cache"
        self.root = base / self.namespace
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_entries = max(1, int(max_entries))
        self.max_bytes = max(1, int(max_bytes))
        self.entries: dict[str, dict[str, Any]] = {}
        self.latest: dict[str, str] = {}
        self.lock = threading.RLock()

    def _base_key(self, node_id: Any, variant: str = "") -> str:
        key = str(node_id or "").strip()
        suffix = str(variant or "").strip()
        return f"{key}:{suffix}" if suffix else key

    def _key(self, node_id: Any, variant: str = "", fingerprint: str = "") -> str:
        base = self._base_key(node_id, variant)
        suffix = str(fingerprint or "").strip()
        return f"{base}:{suffix}" if suffix else base

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
    def _frame_fingerprint(frames: np.ndarray, fps: float) -> str:
        digest = hashlib.sha256()
        digest.update(str(tuple(int(value) for value in frames.shape)).encode("ascii"))
        digest.update(b"|")
        digest.update(f"{float(fps):.6f}".encode("ascii"))
        raw = memoryview(frames).cast("B")
        chunk_size = 8 * 1024 * 1024
        for start in range(0, len(raw), chunk_size):
            digest.update(raw[start : start + chunk_size])
        return digest.hexdigest()

    @classmethod
    def _audio_fingerprint(cls, audio: Any) -> str:
        prepared = cls._prepare_audio(audio)
        if prepared is None:
            return "none"
        waveform = prepared["waveform"].numpy()
        digest = hashlib.sha256()
        digest.update(str(tuple(int(value) for value in waveform.shape)).encode("ascii"))
        digest.update(b"|")
        digest.update(str(int(prepared["sample_rate"])).encode("ascii"))
        raw = memoryview(np.ascontiguousarray(waveform)).cast("B")
        chunk_size = 8 * 1024 * 1024
        for start in range(0, len(raw), chunk_size):
            digest.update(raw[start : start + chunk_size])
        return digest.hexdigest()

    @staticmethod
    def fingerprint_value(value: Any) -> str:
        """Return a stable content fingerprint for auxiliary tensor data."""
        if isinstance(value, torch.Tensor):
            array = value.detach().to(device="cpu").contiguous().numpy()
        else:
            array = np.ascontiguousarray(np.asarray(value))
        digest = hashlib.sha256()
        digest.update(str(tuple(int(item) for item in array.shape)).encode("ascii"))
        digest.update(b"|")
        digest.update(str(array.dtype).encode("ascii"))
        raw = memoryview(array).cast("B")
        chunk_size = 8 * 1024 * 1024
        for start in range(0, len(raw), chunk_size):
            digest.update(raw[start : start + chunk_size])
        return digest.hexdigest()

    def _refresh_latest(self, base_key: str) -> None:
        candidates = [
            (key, entry)
            for key, entry in self.entries.items()
            if self._base_key(entry.get("node_id"), entry.get("variant")) == base_key
        ]
        if candidates:
            latest_key, _ = max(candidates, key=lambda item: float(item[1].get("created") or 0.0))
            self.latest[base_key] = latest_key
        else:
            self.latest.pop(base_key, None)

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
                base_key = self._base_key(entry.get("node_id"), entry.get("variant"))
                if self.latest.get(base_key) == key:
                    self._refresh_latest(base_key)
                evicted.append(entry)
        for entry in evicted:
            self._remove_files(entry)

    def put(
        self,
        node_id: Any,
        frames: Any,
        fps: Any,
        *,
        variant: str = "",
        cache_fingerprint: str = "",
        encode_video: bool = True,
        info: dict[str, Any] | None = None,
        audio: Any = None,
        progress: Any = None,
        force: bool = False,
    ) -> dict[str, Any]:
        base_key = self._base_key(node_id, variant)
        if not base_key:
            raise ValueError("Preview cache node_id is required.")
        array = self._as_uint8(frames)
        safe_fps = self._safe_fps(fps)
        content_fingerprint = self._frame_fingerprint(array, safe_fps)
        audio_fingerprint = self._audio_fingerprint(audio)
        content_fingerprint = hashlib.sha256(
            f"{content_fingerprint}:{audio_fingerprint}".encode("ascii")
        ).hexdigest()
        semantic_fingerprint = str(cache_fingerprint or "").strip()
        fingerprint = (
            f"{semantic_fingerprint}:{content_fingerprint}"
            if semantic_fingerprint
            else content_fingerprint
        )
        key = self._key(node_id, variant, fingerprint)
        with self.lock:
            existing = self.entries.get(key)
            if existing is not None and not force:
                frames_ready = Path(str(existing.get("frames_path") or "")).is_file()
                video_path = Path(str(existing.get("video_path") or existing.get("path") or ""))
                video_ready = not encode_video or video_path.is_file()
                if frames_ready and video_ready:
                    self.latest[base_key] = key
                    return existing
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
        base_info["content_fingerprint"] = content_fingerprint
        if semantic_fingerprint:
            base_info["cache_fingerprint"] = semantic_fingerprint
        entry = {
            "token": token,
            "namespace": self.namespace,
            "node_id": str(node_id),
            "variant": str(variant or ""),
            "fingerprint": fingerprint,
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
            self.latest[base_key] = key
        self._remove_files(previous)
        self._evict()
        return entry

    def put_preview(
        self,
        node_id: Any,
        frames: Any,
        fps: Any,
        *,
        proxy: bool = False,
        cache_fingerprint: str = "",
        encode_video: bool = True,
        info: dict[str, Any] | None = None,
        audio: Any = None,
        progress: Any = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Store a node-owned preview using the shared main/proxy variants."""
        return self.put(
            node_id,
            frames,
            fps,
            variant="proxy" if proxy else "main",
            cache_fingerprint=cache_fingerprint,
            encode_video=encode_video,
            info=info,
            audio=audio,
            progress=progress,
            force=force,
        )

    def get_node(self, node_id: Any, variant: str = "") -> dict[str, Any] | None:
        with self.lock:
            return self.entries.get(self._key(node_id, variant))

    def get_preview_variant(
        self,
        node_id: Any,
        *,
        proxy: bool = False,
        cache_fingerprint: str = "",
    ) -> dict[str, Any] | None:
        """Return one canonical preview variant, with legacy-key compatibility."""
        variant = "proxy" if proxy else "main"
        if cache_fingerprint:
            with self.lock:
                entry = self.entries.get(self._key(node_id, variant, cache_fingerprint))
                if entry is None:
                    candidates = [
                        candidate
                        for candidate in self.entries.values()
                        if self._base_key(candidate.get("node_id"), candidate.get("variant"))
                        == self._base_key(node_id, variant)
                        and str((candidate.get("info") or {}).get("cache_fingerprint") or "")
                        == str(cache_fingerprint)
                    ]
                    entry = max(candidates, key=lambda item: float(item.get("created") or 0.0)) if candidates else None
        else:
            base_key = self._base_key(node_id, variant)
            with self.lock:
                latest_key = self.latest.get(base_key)
                entry = self.entries.get(latest_key) if latest_key else None
                if entry is None:
                    entry = self.entries.get(base_key)
        if entry is not None:
            return entry
        # Older nodes stored proxy data by appending ':proxy' to node_id.
        if proxy:
            entry = self.get_node(f"{str(node_id or '').strip()}:proxy")
            if entry is not None:
                return entry
            # Subtitle caches used the name 'preview' before the shared API.
            return self.get_node(node_id, "preview")
        entry = self.get_node(node_id, "main")
        return entry if entry is not None else self.get_node(node_id)

    def get_preview(self, node_id: Any) -> dict[str, Any] | None:
        """Return a node's proxy preview, falling back to its main preview."""
        return self.get_preview_variant(node_id, proxy=True) or self.get_preview_variant(node_id, proxy=False)

    def get_token(self, token: Any) -> dict[str, Any] | None:
        resolved = str(token or "").strip()
        if not resolved or not resolved.startswith(f"{self.namespace}:"):
            return None
        with self.lock:
            return next((entry for entry in self.entries.values() if entry.get("token") == resolved), None)

    def remove(self, node_id: Any, variant: str = "") -> None:
        with self.lock:
            base_key = self._base_key(node_id, variant)
            key = self.latest.pop(base_key, None) or base_key
            entry = self.entries.pop(key, None)
            if key != base_key and entry is None:
                entry = self.entries.pop(base_key, None)
            self._refresh_latest(base_key)
        self._remove_files(entry)

    def clear_node(self, node_id: Any) -> None:
        key = str(node_id or "").strip()
        with self.lock:
            keys = [item for item, entry in self.entries.items() if entry.get("node_id") == key]
            removed = [self.entries.pop(item, None) for item in keys]
            for variant in {str(entry.get("variant") or "") for entry in removed if entry}:
                self.latest.pop(self._base_key(key, variant), None)
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


class WaitInputCacheStore(PreviewCacheStore):
    """Persistent, source-chain keyed cache shared by the preview nodes."""

    def __init__(self, root: str | os.PathLike | None = None):
        super().__init__(
            _WAIT_INPUT_CACHE_NAMESPACE,
            root=root,
            max_entries=32,
            max_bytes=4 * 1024**3,
        )
        self.manifest_root = self.root / "manifests"
        self.manifest_root.mkdir(parents=True, exist_ok=True)

    def _manifest_path(self, fingerprint: str) -> Path:
        value = str(fingerprint or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("Invalid wait input cache fingerprint.")
        return self.manifest_root / f"{value}.json"

    @staticmethod
    def _manifest_safe(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): WaitInputCacheStore._manifest_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [WaitInputCacheStore._manifest_safe(item) for item in value]
        return str(value)

    @staticmethod
    def _entry_files_ready(entry: dict[str, Any] | None) -> bool:
        if not entry:
            return False
        try:
            return (
                Path(str(entry.get("frames_path") or "")).is_file()
                and Path(str(entry.get("video_path") or entry.get("path") or "")).is_file()
            )
        except (OSError, TypeError, ValueError):
            return False

    def _write_manifest(self, fingerprint: str, entry: dict[str, Any]) -> None:
        path = self._manifest_path(fingerprint)
        payload = {
            "version": _WAIT_INPUT_CACHE_VERSION,
            "fingerprint": str(fingerprint),
            "entry": {
                **dict(entry),
                # Waveforms are not JSON serialisable. The encoded video is
                # already complete, so a restart does not need the samples.
                "audio": None,
            },
        }
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(self._manifest_safe(payload), ensure_ascii=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _load_manifest(self, fingerprint: str) -> dict[str, Any] | None:
        path = self._manifest_path(fingerprint)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or int(payload.get("version", 0)) != _WAIT_INPUT_CACHE_VERSION:
                return None
            entry = payload.get("entry")
            if not isinstance(entry, dict) or str(payload.get("fingerprint") or "") != str(fingerprint):
                return None
            if not self._entry_files_ready(entry):
                path.unlink(missing_ok=True)
                return None
            entry = dict(entry)
            entry["namespace"] = self.namespace
            entry["audio"] = None
            key = self._key(entry.get("node_id"), entry.get("variant"), entry.get("fingerprint"))
            with self.lock:
                self.entries[key] = entry
                self.latest[self._base_key(entry.get("node_id"), entry.get("variant"))] = key
            return entry
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

    def get_chain(self, chain: Any) -> dict[str, Any] | None:
        fingerprint = input_chain_fingerprint(chain) if isinstance(chain, dict) else str(chain or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            return None
        base_key = self._base_key(fingerprint, "main")
        with self.lock:
            key = self.latest.get(base_key)
            entry = self.entries.get(key) if key else None
        if self._entry_files_ready(entry):
            return entry
        return self._load_manifest(fingerprint)

    def _drop_entry(self, entry: dict[str, Any] | None) -> None:
        if not entry:
            return
        key = self._key(entry.get("node_id"), entry.get("variant"), entry.get("fingerprint"))
        base_key = self._base_key(entry.get("node_id"), entry.get("variant"))
        with self.lock:
            current = self.entries.get(key)
            if current is not None and str(current.get("token")) == str(entry.get("token")):
                self.entries.pop(key, None)
            if self.latest.get(base_key) == key:
                self._refresh_latest(base_key)
        self._remove_files(entry)

    def put_chain(
        self,
        chain: dict[str, Any],
        frames: Any,
        fps: Any,
        *,
        info: dict[str, Any] | None = None,
        audio: Any = None,
        progress: Any = None,
        force: bool = False,
    ) -> dict[str, Any]:
        normalised = _normalise_input_chain(chain)
        fingerprint = input_chain_fingerprint(normalised)
        if normalised is None or not fingerprint:
            raise ValueError("A complete input chain is required for the wait input cache.")
        previous = self.get_chain(fingerprint)
        cache_info = dict(info or {})
        cache_info.update(
            {
                "wait_input_cache": True,
                "source_chain_fingerprint": fingerprint,
                "source_chain": normalised,
            }
        )
        entry = super().put(
            fingerprint,
            frames,
            fps,
            variant="main",
            encode_video=True,
            info=cache_info,
            audio=audio,
            progress=progress,
            force=force,
        )
        if previous is not None and str(previous.get("token")) != str(entry.get("token")):
            self._drop_entry(previous)
        self._write_manifest(fingerprint, entry)
        return entry


def get_wait_input_cache_store() -> WaitInputCacheStore:
    global _WAIT_INPUT_CACHE_STORE
    with _WAIT_INPUT_CACHE_STORE_LOCK:
        if _WAIT_INPUT_CACHE_STORE is None:
            _WAIT_INPUT_CACHE_STORE = WaitInputCacheStore()
        return _WAIT_INPUT_CACHE_STORE


async def _wait_input_cache_info_route(request: Any) -> Any:
    from aiohttp import web

    try:
        payload = await request.json()
        chain = _normalise_input_chain(payload.get("chain") if isinstance(payload, dict) else None)
        if chain is None or not chain["complete"]:
            return web.json_response({"error": "Run ComfyUI once to generate preview cache."}, status=404)
        entry = get_wait_input_cache_store().get_chain(chain)
        if entry is None:
            return web.json_response({"error": "Run ComfyUI once to generate preview cache."}, status=404)
        entry = get_wait_input_cache_store().ensure_video(entry)
        path = Path(str((entry or {}).get("video_path") or (entry or {}).get("path") or "")) if entry else None
        if entry is None or path is None or not path.is_file():
            return web.json_response({"error": "Run ComfyUI once to generate preview cache."}, status=404)
        token = str(entry.get("token") or "")
        return web.json_response(
            {
                "token": token,
                "label": "Shared preview cache from input chain",
                "video_url": f"/cinestyle/wait-input-cache-video?token={token}",
                "info": dict(entry.get("info") or {}),
                "source_chain_fingerprint": str((entry.get("info") or {}).get("source_chain_fingerprint") or ""),
            }
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def _wait_input_cache_video_route(request: Any) -> Any:
    from aiohttp import web

    entry = get_wait_input_cache_store().get_token(request.query.get("token", ""))
    if entry is None:
        return web.json_response({"error": "Wait input preview cache not found."}, status=404)
    path = Path(str(entry.get("video_path") or entry.get("path") or ""))
    if not path.is_file():
        return web.json_response({"error": "Wait input preview cache video not found."}, status=404)
    return web.FileResponse(path=path, headers={"Content-Type": "video/mp4", "Cache-Control": "no-store"})


def register_wait_input_cache_routes(server_instance: Any) -> None:
    global _WAIT_INPUT_CACHE_ROUTES_REGISTERED
    if _WAIT_INPUT_CACHE_ROUTES_REGISTERED or server_instance is None:
        return
    server_instance.routes.post("/cinestyle/wait-input-cache")(_wait_input_cache_info_route)
    server_instance.routes.get("/cinestyle/wait-input-cache-video")(_wait_input_cache_video_route)
    _WAIT_INPUT_CACHE_ROUTES_REGISTERED = True
