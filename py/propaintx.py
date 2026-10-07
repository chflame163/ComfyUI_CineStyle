"""ProPainterX video object removal for ComfyUI.

Frames plus a per-frame MASK go in, the masked regions are erased and filled, and the
result is pasted back onto the source resolution. The engine is ProPainterX, the fork
Sammie-Roto 2 ships: upstream ProPainter weights with MemFOF in place of RAFT for the
optical-flow stage, which is what makes Full HD removal fit on a 24 GB card.

The model code is bundled in ``propaintx_vendor/propaintx`` with package-relative
imports, so it never competes for the top-level ``model`` name that the stock
ComfyUI_ProPainter_Nodes pack also owns.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import torch

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


NODE_ID = "CS_ProPainterX_Inpaint"
_CATEGORY = "😺dzNodes/CineStyle"
_LOGGER = logging.getLogger("CineStyleProPainterX")

# Official sources. The two .pth files are the upstream author's own release assets and
# carry the same md5s Sammie-Roto pins; the MemFOF repo is the checkpoint the MemFOF
# project (msu-video-group/memfof) names in its README and model card, the one it
# recommends for real-world video.
_GITHUB_RELEASE = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0"
_MEMFOF_REPO = "egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH"
_DEFAULT_HF_ENDPOINT = "https://huggingface.co"
WEIGHT_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "path": "ProPainter.pth",
        "kind": "github",
        "url": f"{_GITHUB_RELEASE}/ProPainter.pth",
        "md5": "83e3941395917f6c1943dcf2f7655454",
        "size": 157780510,
    },
    {
        "path": "recurrent_flow_completion.pth",
        "kind": "github",
        "url": f"{_GITHUB_RELEASE}/recurrent_flow_completion.pth",
        "md5": "2879dbdd08fa50c656ff3ff1659dd660",
        "size": 20348681,
    },
    {
        "path": "memfof/config.json",
        "kind": "hf",
        "repo": _MEMFOF_REPO,
        "file": "config.json",
        "md5": "232de6070af4973acdc5d2b6788581d8",
        "size": 157,
    },
    {
        "path": "memfof/model.safetensors",
        "kind": "hf",
        "repo": _MEMFOF_REPO,
        "file": "model.safetensors",
        "md5": "17ca039b45c73b392ab77e3053fde7db",
        "size": 303172564,
    },
)

# ProPainterX reports progress across four stages: optical flow, flow completion,
# image propagation and the transformer.
_STAGE_COUNT = 4

# Decisions the node makes for itself instead of exposing another dozen inputs: run at
# the source size, let one pass cover as many frames as the free VRAM affords, and
# always paste the repaired region back onto the source resolution.
_WORKING_RESOLUTION = 0
_GROW_MASK = 0
_VRAM_FRACTION = 0.8
_ENCODER_CHUNK_SIZE = 10
_FP16 = True
_DEFAULT_BATCH_OVERLAP = 4
# Edge treatment when the repaired region has to be pasted back onto a larger source
# frame. Only reachable when the source is not a multiple of 8, since inference then
# runs slightly smaller than the source.
_COMPOSITE_GROW = 21
_COMPOSITE_FEATHER = 10

# Peak VRAM of one frame inside the transformer stage, measured at a 1024x576
# reference frame. Deliberately conservative: under-estimating turns a slow run into a
# CUDA OOM halfway through, and this node is meant to share a card with other jobs.
_FRAME_PEAK_BYTES_AT_REF = 96 * 1024**2
_REF_PIXELS = 1024 * 576
_GPU_MEMORY_RESERVE_BYTES = 384 * 1024**2
_MAX_BATCH_FRAMES = 512

# ProPainterX's encoder/decoder chain needs dimensions divisible by this.
_ALIGNMENT = 8
# Matches scipy.ndimage.binary_dilation's default connectivity-1 element, which is what
# ProPainterX's own read_mask() dilates with.
_CROSS_KERNEL = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

_MODEL_LOCK = threading.RLock()
_DOWNLOAD_LOCK = threading.Lock()
_PIPELINES: dict[tuple, Any] = {}


def _info(message: str) -> None:
    """Emit the readable stage prefix the other video nodes use."""
    _LOGGER.info("[CS ProPainterX Inpaint] %s", message)


def _check_interrupt() -> None:
    checker = getattr(comfy.model_management, "throw_exception_if_processing_interrupted", None)
    if callable(checker):
        checker()


def _interrupted() -> bool:
    try:
        _check_interrupt()
    except Exception:
        return True
    return False


# ------------------------------------------------------------------ weights

def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_root(explicit: str = "") -> Path:
    """Where the ProPainterX weights live.

    ``<models>/ProPainter`` by default, which is a directory of its own: the stock
    ComfyUI_ProPainter_Nodes pack keeps its weights inside its own folder, so the two
    never collide.
    """
    value = str(explicit or "").strip()
    if value:
        return Path(os.path.abspath(os.path.expanduser(value)))
    env = (os.environ.get("PROPAINTER_X_MODEL_DIR") or "").strip()
    if env:
        return Path(os.path.abspath(os.path.expanduser(env)))
    try:
        roots = folder_paths.get_folder_paths("propainter")
    except Exception:
        roots = []
    if roots:
        return Path(roots[0])
    return Path(folder_paths.models_dir) / "ProPainter"


def _target_path(root: Path, relative: str) -> Path:
    return root.joinpath(*relative.split("/"))


def _missing_weights(root: Path) -> list[dict[str, Any]]:
    """Sources whose file is absent, truncated, or has the wrong checksum."""
    missing: list[dict[str, Any]] = []
    for source in WEIGHT_SOURCES:
        target = _target_path(root, str(source["path"]))
        if not target.is_file() or target.stat().st_size <= 0:
            missing.append(source)
            continue
        expected = str(source.get("md5") or "")
        if expected and _md5(target) != expected:
            _info(f"{source['path']} exists but its checksum is wrong; re-downloading")
            target.unlink(missing_ok=True)
            missing.append(source)
    return missing


def _candidate_urls(source: dict[str, Any]) -> list[str]:
    """Direct URL first, then any mirror configured for restricted networks."""
    if source["kind"] == "github":
        urls = [str(source["url"])]
        mirror = (os.environ.get("PROPAINTER_X_GITHUB_MIRROR") or "").strip().rstrip("/")
        if mirror:
            urls.append(f"{mirror}/{source['url']}")
        return urls

    endpoint = (
        (os.environ.get("PROPAINTER_X_HF_ENDPOINT") or os.environ.get("HF_ENDPOINT") or "")
        .strip()
        .rstrip("/")
        or _DEFAULT_HF_ENDPOINT
    )
    path = f"{source['repo']}/resolve/main/{source['file']}"
    urls = [f"{endpoint}/{path}"]
    if endpoint != _DEFAULT_HF_ENDPOINT:
        urls.append(f"{_DEFAULT_HF_ENDPOINT}/{path}")
    return urls


def _download(source: dict[str, Any], root: Path) -> Path:
    target = _target_path(root, str(source["path"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".download")
    expected = int(source.get("size") or 0)
    errors: list[str] = []

    for url in _candidate_urls(source):
        done = 0
        next_mark = 16 * 1024 * 1024
        try:
            _info(f"downloading {source['path']} from {url}")
            with urlopen(Request(url, headers={"User-Agent": "ComfyUI_CineStyle"}), timeout=60) as response:
                with partial.open("wb") as handle:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
                        done += len(block)
                        if done >= next_mark:
                            _info(f"  {source['path']}: {done / 1048576:.1f}/{expected / 1048576:.1f} MiB")
                            next_mark += 16 * 1024 * 1024
            if expected and done != expected:
                raise RuntimeError(f"got {done} bytes, expected {expected}")
            digest = str(source.get("md5") or "")
            if digest and _md5(partial) != digest:
                raise RuntimeError("checksum does not match the official release")
            os.replace(partial, target)
            _info(f"  saved {source['path']} ({target.stat().st_size / 1048576:.1f} MiB)")
            return target
        except (OSError, RuntimeError, URLError, ValueError) as exc:
            errors.append(f"{url}: {exc}")
            partial.unlink(missing_ok=True)

    raise RuntimeError(
        f"Unable to download {source['path']}. Place it at {target} by hand. Attempts:\n    "
        + "\n    ".join(errors)
    )


def _ensure_weights(root: Path) -> None:
    missing = _missing_weights(root)
    if not missing:
        return
    total_mb = sum(int(source.get("size") or 0) for source in missing) / 1048576
    _info(f"{len(missing)} weight file(s) missing under {root}; fetching {total_mb:.0f} MiB")
    with _DOWNLOAD_LOCK:
        for source in missing:
            if not _target_path(root, str(source["path"])).is_file():
                _download(source, root)
    _info("all weights ready")


def _weight_paths(root: Path) -> dict[str, str]:
    return {
        "propainter": str(_target_path(root, "ProPainter.pth")),
        "flowcomp": str(_target_path(root, "recurrent_flow_completion.pth")),
        "memfof": str(_target_path(root, "memfof")),
    }


# ------------------------------------------------------------------ device and memory

def _device(value: str) -> torch.device:
    choice = str(value or "auto")
    if choice == "cpu":
        return torch.device("cpu")
    if choice.startswith("gpu"):
        try:
            index = int(choice[3:])
            if torch.cuda.is_available() and 0 <= index < torch.cuda.device_count():
                return torch.device(f"cuda:{index}")
        except (TypeError, ValueError):
            pass
        _LOGGER.warning(
            "[CS ProPainterX Inpaint] invalid device %r; falling back to ComfyUI auto device.", choice
        )
    result = comfy.model_management.get_torch_device()
    return result if isinstance(result, torch.device) else torch.device(result)


def _free_vram_bytes(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    try:
        free, _total = torch.cuda.mem_get_info(device)
        return int(free)
    except (RuntimeError, TypeError, ValueError, AttributeError):
        getter = getattr(comfy.model_management, "get_free_memory", None)
        if callable(getter):
            try:
                return int(float(getter(device)) * 1024 * 1024)
            except (RuntimeError, TypeError, ValueError):
                pass
    return None


def _usable_bytes(free_bytes: int) -> int:
    fraction = max(0.05, min(0.95, _VRAM_FRACTION))
    return max(0, min(int(free_bytes * fraction), int(free_bytes) - _GPU_MEMORY_RESERVE_BYTES))


def _frame_peak_bytes(height: int, width: int) -> float:
    scale = max(0.125, (height * width) / float(_REF_PIXELS))
    return _FRAME_PEAK_BYTES_AT_REF * scale


def _batch_frames(total: int, height: int, width: int, free_bytes: int | None) -> int:
    """How many frames one inference pass should cover, given the free VRAM."""
    total = max(1, int(total))
    if free_bytes is None:
        # CPU: VRAM is not the constraint, so only bound the host-side buffering.
        return min(total, _MAX_BATCH_FRAMES)
    usable = _usable_bytes(free_bytes)
    per_frame = max(1.0, _frame_peak_bytes(height, width))
    # No lower floor on purpose: if the budget only affords one frame, forcing four
    # just moves the OOM from planning time into the middle of a run.
    return max(1, min(total, _MAX_BATCH_FRAMES, int(usable // per_frame)))


def _transformer_subvideo_length(
    subvideo_length: int,
    neighbor_length: int,
    ref_stride: int,
    height: int,
    width: int,
    free_bytes: int | None,
) -> int:
    """Shrink ``subvideo_length`` when the transformer stage would not fit in VRAM.

    That stage holds ``neighbor_length`` neighbours plus one reference frame per
    ``ref_stride`` frames of the current sub-video, so this is the knob that actually
    decides peak VRAM; total clip length does not.
    """
    subvideo = max(1, int(subvideo_length))
    neighbor = max(2, int(neighbor_length))
    stride = max(1, int(ref_stride))
    if free_bytes is None:
        return subvideo

    usable = _usable_bytes(free_bytes)
    per_frame = max(1.0, _frame_peak_bytes(height, width))
    allowed = max(1, int(usable // per_frame))
    if neighbor + max(1, subvideo // stride) <= allowed:
        return subvideo
    room_for_refs = max(1, allowed - neighbor)
    shrunk = max(2 * stride, (room_for_refs // stride) * stride)
    return min(subvideo, shrunk)


def _clear_cache() -> None:
    softer = getattr(comfy.model_management, "soft_empty_cache", None)
    if callable(softer):
        try:
            softer()
            return
        except (RuntimeError, TypeError, ValueError):
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ frame plumbing

def _frames_from_images(images: torch.Tensor) -> list[np.ndarray]:
    """ComfyUI IMAGE ``[B,H,W,C]`` in 0..1 -> list of HxWx3 uint8 RGB."""
    value = images.unsqueeze(0) if images.ndim == 3 else images
    if value.ndim != 4 or value.shape[-1] < 3:
        raise ValueError(f"image must be an IMAGE batch of shape [B,H,W,C]; got {tuple(images.shape)}")
    rgb = value[..., :3].to(device="cpu", dtype=torch.float32)
    rgb = rgb.mul(255.0).add(0.5).clamp(0.0, 255.0).to(torch.uint8).numpy()
    return [np.ascontiguousarray(frame) for frame in rgb]


def _masks_from_tensor(mask: torch.Tensor, frame_count: int) -> list[np.ndarray]:
    """ComfyUI MASK -> list of HxW uint8 {0,255}; a single mask broadcasts to all frames."""
    value = mask
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim == 4 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 3:
        raise ValueError(f"mask must have shape [H,W] or [B,H,W]; got {tuple(mask.shape)}")

    count = int(value.shape[0])
    if count == 1 and frame_count > 1:
        value = value.expand(frame_count, -1, -1)
    elif count != frame_count:
        raise ValueError(
            f"mask batch ({count}) matches neither one shared mask nor the frame count ({frame_count})"
        )

    planes = value.to(device="cpu", dtype=torch.float32).numpy()
    return [np.ascontiguousarray(np.where(plane > 0.5, 255, 0).astype(np.uint8)) for plane in planes]


def _working_size(height: int, width: int, max_side: int) -> tuple[int, int]:
    """Cap the shorter side, then floor both dimensions to a multiple of 8."""
    scale = 1.0
    if max_side > 0 and min(height, width) > max_side:
        scale = float(max_side) / float(min(height, width))
    new_h = (int(height * scale) // _ALIGNMENT) * _ALIGNMENT
    new_w = (int(width * scale) // _ALIGNMENT) * _ALIGNMENT
    return max(_ALIGNMENT, new_h), max(_ALIGNMENT, new_w)


def _resize_stack(frames: list[np.ndarray], size: tuple[int, int], mask: bool = False) -> list[np.ndarray]:
    height, width = size
    interpolation = cv2.INTER_NEAREST if mask else cv2.INTER_AREA
    output = []
    for array in frames:
        if (array.shape[0], array.shape[1]) == (height, width):
            output.append(array)
        else:
            output.append(np.ascontiguousarray(cv2.resize(array, (width, height), interpolation=interpolation)))
    return output


def _grow_masks(masks: list[np.ndarray], grow: int) -> list[np.ndarray]:
    if grow <= 0:
        return masks
    return [np.ascontiguousarray(cv2.dilate(item, _CROSS_KERNEL, iterations=int(grow))) for item in masks]


def _composite_over_original(
    processed: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    grow: int = 0,
    feather: int = 0,
) -> np.ndarray:
    """Paste the inpainted region back onto the full-resolution frame.

    ProPainterX runs at a reduced working size, so only the hole comes from the
    processed frame and everything else keeps the original detail.
    """
    height, width = original.shape[:2]
    if processed.shape[:2] != (height, width):
        processed = cv2.resize(processed, (width, height), interpolation=cv2.INTER_LINEAR)
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    if grow > 0:
        mask = cv2.dilate(mask, _CROSS_KERNEL, iterations=int(grow))
    if feather > 0:
        ksize = int(feather) * 2 + 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    blended = processed.astype(np.float32) * alpha + original.astype(np.float32) * (1.0 - alpha)
    return np.clip(np.rint(blended), 0.0, 255.0).astype(np.uint8)


def _images_from_frames(frames: list[np.ndarray]) -> torch.Tensor:
    stacked = np.stack(frames).astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(stacked))


# ------------------------------------------------------------------ windowing

def _window_starts(total: int, window_frames: int, overlap_frames: int) -> list[int]:
    """Global start index of every window, with no gap in coverage.

    Windows advance by ``window_frames - overlap_frames``. When the last window would
    only add a handful of new frames it slides back to land on the final frame instead
    of running a short pass, which is the same layout CS MatAnyone2 uses for its
    multi-anchor segments.
    """
    if total <= 0:
        return []
    window = max(1, min(int(window_frames), total))
    overlap = max(0, min(int(overlap_frames), window - 1))
    if window >= total:
        return [0]

    stride = max(1, window - overlap)
    starts = [0]
    while starts[-1] + window < total:
        next_start = starts[-1] + stride
        if next_start + window >= total:
            final_start = max(0, total - window)
            if len(starts) >= 2 and final_start > starts[-1] and final_start < starts[-2] + window:
                starts[-1] = final_start
                break
            next_start = final_start
        if next_start <= starts[-1]:
            break
        starts.append(next_start)
    return starts


def _window_weights(length: int, start: int, total: int, overlap_frames: int) -> list[float]:
    """Linear fade-in / fade-out weights for one window's frames."""
    weights = [1.0] * length
    end = start + length
    ramp = min(max(0, int(overlap_frames)), length)
    if ramp <= 0:
        return weights
    if start > 0:
        for index in range(ramp):
            weights[index] *= float(index + 1) / float(ramp + 1)
    if end < total:
        for index in range(length - ramp, length):
            weights[index] *= float(length - index) / float(ramp + 1)
    return weights


class _OverlapBlender:
    """Combine the frames each window produced into one sequence.

    ProPainterX infers every window on its own, so a seam can appear where two windows
    meet. Frames inside an overlap are averaged with the ramp above to hide it. Only
    frames that actually receive more than one contribution are held in float32, which
    keeps host memory flat for long clips.
    """

    def __init__(self, total: int, overlap_frames: int) -> None:
        self.total = int(total)
        self.overlap = max(0, int(overlap_frames))
        self._single: list[Any] = [None] * self.total
        self._mixed: dict[int, list[Any]] = {}

    def add_window(self, start: int, frames: list[np.ndarray]) -> None:
        weights = _window_weights(len(frames), start, self.total, self.overlap)
        for offset, frame in enumerate(frames):
            index = start + offset
            weight = weights[offset]
            contribution = frame.astype(np.float32) * weight
            if index in self._mixed:
                accumulator, total_weight = self._mixed[index]
                self._mixed[index] = [accumulator + contribution, total_weight + weight]
            elif self._single[index] is not None:
                base = self._single[index].astype(np.float32)
                self._single[index] = None
                self._mixed[index] = [base + contribution, 1.0 + weight]
            elif weight >= 1.0:
                self._single[index] = frame
            else:
                self._mixed[index] = [contribution, weight]

    def finish(self, fallback_frames: list[np.ndarray]) -> list[np.ndarray]:
        for index, (accumulator, total_weight) in self._mixed.items():
            if total_weight > 0:
                self._single[index] = np.clip(
                    np.rint(accumulator / total_weight), 0.0, 255.0
                ).astype(np.uint8)
        return [
            frame if frame is not None else fallback_frames[index]
            for index, frame in enumerate(self._single)
        ]


# ------------------------------------------------------------------ runtime

def _add_bundled_runtime() -> None:
    vendor = Path(__file__).resolve().parent / "propaintx_vendor"
    if (vendor / "propaintx" / "pipeline.py").is_file():
        value = str(vendor)
        if value not in sys.path:
            sys.path.insert(0, value)


def _pipeline_class() -> Any:
    _add_bundled_runtime()
    try:
        module = importlib.import_module("propaintx.pipeline")
    except ImportError as exc:
        raise RuntimeError(
            "ProPainterX runtime is unavailable. Reinstall ComfyUI_CineStyle with its "
            "bundled runtime and dependencies, then restart ComfyUI."
        ) from exc
    return module.ProPainterXPipeline, module.CancelledError


def _get_pipeline(weights: dict[str, str], device: torch.device, fp16: bool, encoder_chunk_size: int) -> Any:
    pipeline_class, _ = _pipeline_class()
    key = (
        tuple(sorted(weights.items())),
        device.type,
        device.index,
        bool(fp16),
        int(encoder_chunk_size),
    )
    with _MODEL_LOCK:
        pipeline = _PIPELINES.get(key)
        if pipeline is not None:
            return pipeline
        _unload_pipelines()
        _info(f"loading ProPainterX on {device} (fp16={fp16})")
        pipeline = pipeline_class(
            device=device,
            propainter_ckpt=weights["propainter"],
            flowcomp_ckpt=weights["flowcomp"],
            memfof_model_dir=weights["memfof"],
            fp16=fp16,
            clear_cache=_clear_cache,
        )
        pipeline.load()
        _PIPELINES[key] = pipeline
        _info("models loaded")
        return pipeline


def _unload_pipelines() -> None:
    if not _PIPELINES:
        return
    for pipeline in list(_PIPELINES.values()):
        try:
            pipeline.unload()
        except (RuntimeError, AttributeError, TypeError) as exc:
            _LOGGER.warning("[CS ProPainterX Inpaint] unload failed: %s", exc)
    _PIPELINES.clear()
    _clear_cache()
    _info("released models")


class _Progress:
    """Guard around comfy.utils.ProgressBar so reporting cannot break a run."""

    def __init__(self, total_steps: int) -> None:
        self.total = max(1, int(total_steps))
        self._bar = None
        try:
            self._bar = comfy.utils.ProgressBar(self.total)
        except (RuntimeError, TypeError, ValueError):
            self._bar = None

    def set(self, value: float) -> None:
        if self._bar is None:
            return
        try:
            self._bar.update_absolute(int(value), self.total)
        except (RuntimeError, TypeError, ValueError):
            self._bar = None


# ------------------------------------------------------------------ node

class CSProPainterXInpaint(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        devices = ["auto", "cpu"]
        if torch.cuda.is_available():
            devices.extend(f"gpu{index}" for index in range(torch.cuda.device_count()))
        return io.Schema(
            node_id=NODE_ID,
            display_name="CS ProPainterX Inpaint",
            category=_CATEGORY,
            essentials_category="Video Tools",
            search_aliases=[
                "propainterx",
                "video inpainting",
                "object removal",
                "erase object",
                "memfof",
            ],
            description=(
                "Erase masked regions from a video sequence with ProPainterX and paste the "
                "result back onto the source frames. Inference runs at the source size, one "
                "pass covers as many frames as the free VRAM affords, and long clips are run "
                "in overlapping passes that are cross-faded together."
            ),
            inputs=[
                io.Image.Input("image", tooltip="Video frames as a ComfyUI IMAGE batch."),
                io.Mask.Input(
                    "mask",
                    tooltip="Matched per-frame masks marking what to remove; one shared mask broadcasts to every frame.",
                ),
                io.Combo.Input("device", options=devices, default="auto", advanced=True),
                io.Int.Input(
                    "subvideo_length",
                    display_name="Sub-video length",
                    default=80,
                    min=8,
                    max=300,
                    step=1,
                    advanced=True,
                    tooltip=(
                        "Frames the transformer stage reasons over at once. This is what sets peak "
                        "VRAM, and the node lowers it automatically when memory is tight."
                    ),
                ),
                io.Int.Input("neighbor_length", display_name="Neighbor length", default=10, min=2, max=100, step=2, advanced=True),
                io.Int.Input("ref_stride", display_name="Reference stride", default=10, min=1, max=100, step=1, advanced=True),
                io.Int.Input(
                    "mask_dilation",
                    display_name="Mask dilation",
                    default=4,
                    min=0,
                    max=64,
                    step=1,
                    advanced=True,
                    tooltip="ProPainterX's own dilation of the removal region.",
                ),
                io.Boolean.Input(
                    "force_unload_model",
                    display_name="Force unload model",
                    default=False,
                    advanced=True,
                    tooltip="Release the models when this run ends so other nodes can take the VRAM back.",
                ),
                io.Int.Input(
                    "batch_overlap",
                    display_name="Batch overlap",
                    default=_DEFAULT_BATCH_OVERLAP,
                    min=0,
                    max=64,
                    step=1,
                    tooltip="Frames two consecutive passes share; they are cross-faded to hide the seam.",
                ),
            ],
            outputs=[io.Image.Output("image", display_name="IMAGE")],
        )

    @classmethod
    @torch.inference_mode()
    def execute(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor,
        device: str = "auto",
        subvideo_length: int = 80,
        neighbor_length: int = 10,
        ref_stride: int = 10,
        mask_dilation: int = 4,
        force_unload_model: bool = False,
        batch_overlap: int = _DEFAULT_BATCH_OVERLAP,
    ) -> io.NodeOutput:
        frames_source = image.unsqueeze(0) if image.ndim == 3 else image
        total = int(frames_source.shape[0])
        if total < 1:
            raise ValueError("image batch is empty")

        original_frames = _frames_from_images(frames_source)
        height, width = original_frames[0].shape[:2]
        work_h, work_w = _working_size(height, width, _WORKING_RESOLUTION)
        needs_resize = (work_h, work_w) != (height, width)
        frames = _resize_stack(original_frames, (work_h, work_w)) if needs_resize else original_frames

        # Masks are binarised once at source size: the pipeline gets the grown,
        # downscaled copy and the full-resolution composite gets its own copy.
        masks_full = _masks_from_tensor(mask, total)
        masks_work = (
            _resize_stack(masks_full, (work_h, work_w), mask=True) if needs_resize else masks_full
        )

        compute_device = _device(device)
        use_fp16 = _FP16 and compute_device.type == "cuda"
        free_bytes = _free_vram_bytes(compute_device)

        window = _batch_frames(total, work_h, work_w, free_bytes)
        overlap = max(0, min(int(batch_overlap), window - 1))
        starts = _window_starts(total, window, overlap)
        subvideo = _transformer_subvideo_length(
            subvideo_length, neighbor_length, ref_stride, work_h, work_w, free_bytes
        )
        if subvideo != int(subvideo_length):
            _info(f"VRAM is tight: sub_video_length {int(subvideo_length)} -> {subvideo}")

        free_text = "cpu" if free_bytes is None else f"{free_bytes / 1048576:.0f} MiB free"
        _info(
            f"frames={total} source={width}x{height} working={work_w}x{work_h} device={compute_device} "
            f"({free_text}); passes={len(starts)} batch={window} overlap={overlap} "
            f"sub_video_length={subvideo} neighbor_length={int(neighbor_length)} "
            f"ref_stride={int(ref_stride)} fp16={use_fp16} vram_fraction={_VRAM_FRACTION}"
        )

        root = _model_root()
        _ensure_weights(root)
        weights = _weight_paths(root)
        pipeline = _get_pipeline(weights, compute_device, use_fp16, _ENCODER_CHUNK_SIZE)
        for name, value in (
            ("SUBVIDEO_LENGTH", subvideo),
            ("NEIGHBOR_LENGTH", max(2, int(neighbor_length))),
            ("REF_STRIDE", max(1, int(ref_stride))),
            ("MASK_DILATION", max(0, int(mask_dilation))),
            ("ENCODER_CHUNK_SIZE", _ENCODER_CHUNK_SIZE),
        ):
            setattr(pipeline, name, int(value))

        _, cancelled = _pipeline_class()
        progress = _Progress(len(starts) * _STAGE_COUNT)
        blender = _OverlapBlender(total, overlap)

        for pass_index, start in enumerate(starts):
            end = min(total, start + window)
            base_step = pass_index * _STAGE_COUNT

            def report(stage_idx: int, _stage_count: int, _label: str, done: int, stage_total: int,
                       _base: int = base_step) -> None:
                fraction = 0.0 if not stage_total else min(1.0, float(done) / float(stage_total))
                progress.set(_base + min(int(stage_idx), _STAGE_COUNT - 1) + fraction)

            try:
                outputs = pipeline.run(
                    frames[start:end],
                    masks_work[start:end],
                    on_progress=report,
                    on_frame_done=None,
                    should_cancel=_interrupted,
                )
            except cancelled:
                _info("cancelled by user")
                _check_interrupt()
                raise

            expected = end - start
            if not outputs or len(outputs) != expected:
                raise RuntimeError(
                    f"ProPainterX returned {0 if not outputs else len(outputs)} frames for a {expected} frame window"
                )
            blender.add_window(start, outputs)
            outputs = None
            _clear_cache()

        blended = blender.finish(frames)
        # At source size there is nothing to paste back onto, so the composite only
        # runs when inference actually happened at a different resolution.
        if needs_resize:
            composite_masks = _grow_masks(masks_full, _COMPOSITE_GROW)
            results = [
                _composite_over_original(
                    blended[index],
                    original_frames[index],
                    composite_masks[index],
                    grow=0,
                    feather=_COMPOSITE_FEATHER,
                )
                for index in range(total)
            ]
        else:
            results = blended

        output = _images_from_frames(results)
        _info(f"done: {output.shape[0]} frames at {output.shape[2]}x{output.shape[1]}")
        if force_unload_model:
            with _MODEL_LOCK:
                _unload_pipelines()
        return io.NodeOutput(output)


class ProPainterXExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CSProPainterXInpaint]


async def comfy_entrypoint() -> ProPainterXExtension:
    return ProPainterXExtension()
