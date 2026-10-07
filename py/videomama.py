"""VideoMaMa alpha matting for ComfyUI.

A per-frame coarse binary MASK plus the matching frames go in, and a soft 8-bit
video matte comes back out as a ComfyUI MASK batch. The engine is the Sammie-Roto 2
fork of VideoMaMa's inference pipeline, which never loads the SVD CLIP image encoder
that the upstream code only runs in order to zero its output away.

Long clips are the whole reason this node exists. VideoMaMa is a single-step SVD UNet
whose temporal attention grows superlinearly with frame count, so feeding it a whole
clip at once OOMs well before a 24 GB card is full. Instead one pass covers as many
frames as the free VRAM affords at the current working size, and consecutive passes
share a short seam whose leading frames are re-predicted from the previous pass's own
soft output. That feedback, not just cross-fading, is what keeps the seam invisible.

Inference also runs inside the union bounding box of the masks, so a subject that
fills a third of the frame costs a fraction of the pixels, which is what lets a pass
cover more frames on the same card.

The frame order and frame count of the output match the input exactly, at the input
resolution. Nothing is dropped, reordered, padded or trimmed.
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

try:
    import tqdm
except ImportError:  # ComfyUI always ships it; this only keeps the node loadable without
    tqdm = None

NODE_ID = "CS_VideoMaMa"
_CATEGORY = "😺dzNodes/CineStyle"
_LOGGER = logging.getLogger("CineStyleVideoMaMa")

# Official sources. The UNet is the VideoMaMa authors' own release; the VAE comes from
# the SVD base model it was fine-tuned from, and is the fp16 file because that is the
# only VAE weight the official VideoMaMa layout carries. Sizes and md5s match the pins
# Sammie-Roto 2 ships.
_HF_VIDEO_MAMA = "SammyLim/VideoMaMa"
_HF_SVD = "stabilityai/stable-video-diffusion-img2vid-xt"
_DEFAULT_HF_ENDPOINT = "https://huggingface.co"
WEIGHT_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "path": "unet/config.json",
        "repo": _HF_VIDEO_MAMA,
        "file": "unet/config.json",
        "md5": "732de45e5120d83002ba6c79ae69be34",
        "size": 937,
    },
    {
        "path": "unet/diffusion_pytorch_model.safetensors",
        "repo": _HF_VIDEO_MAMA,
        "file": "unet/diffusion_pytorch_model.safetensors",
        "md5": "c8d457d4d5eb90f274bd441df60c8e47",
        "size": 6098728544,
    },
    {
        "path": "vae/config.json",
        "repo": _HF_SVD,
        "file": "vae/config.json",
        "md5": "1137f303186f6cfaaf75fbb12f9b0967",
        "size": 607,
    },
    {
        "path": "vae/diffusion_pytorch_model.fp16.safetensors",
        "repo": _HF_SVD,
        "file": "vae/diffusion_pytorch_model.fp16.safetensors",
        "md5": "46a0af9a794fb405221988a7e2b1396b",
        "size": 195531910,
    },
)

# Peak VRAM of one frame inside the UNet pass, at a 1024x576 reference frame.
# Derived rather than guessed: SAMMatte's planner settles on 15 frames per pass at this
# size on a 24 GB RTX 3090, and its README measures the whole refinement at 10-14 GiB
# with about 4.3 GiB of weights resident, which puts the activation cost near 0.4-0.65
# GiB per frame. SAMMatte plans against a flat 1 GiB per frame and so halves its own
# batches; this takes the top of the measured range and leaves the rest to the OOM retry
# below. Raise it if a card still OOMs on a later pass.
_FRAME_PEAK_BYTES_AT_REF = 640 * 1024**2
_REF_PIXELS = 1024 * 576
_GPU_MEMORY_RESERVE_BYTES = 2048 * 1024**2
# The flat reserve above is capped by this share of whatever is actually free, so a card
# that is nearly full still gets a workable batch instead of one frame per pass.
_GPU_MEMORY_RESERVE_RATIO = 0.15
_VRAM_FRACTION = 0.90
_MAX_BATCH_FRAMES = 512
# A pass that runs out of memory is retried from the top of the clip at half the batch,
# a bounded number of times. Restarting rather than resuming keeps the seam chain
# consistent, because each pass's conditioning depends on the one before it.
_MAX_ATTEMPTS = 3

# The UNet is trained on 25-frame clips (``num_frames`` in its config), so asking the
# auto planner for more than that buys diminishing temporal context at superlinear
# memory cost. Raise it by hand if a shot really needs a longer window.
_TRAINED_WINDOW_FRAMES = 25

# SVD frame-rate conditioning, not the clip's real frame rate. Every VideoMaMa reference
# implementation ships 7 and the model is trained at 7, so it is fixed here rather than
# exposed as a knob that would only ever be set wrong.
_FPS = 7

# Behaviour fixed by design so the node only exposes the knobs worth reaching for. Each
# one is still a module constant rather than a literal, so tuning them is a one line edit.
_DEFAULT_MAX_RESOLUTION = 1280
_SOFT_ALPHA_FEEDBACK = True
_CROP_TO_MASK_REGION = True
_CONSOLE_PROGRESS = True
_FP16 = True
_NOISE_AUG_STRENGTH = 0.0
# The temporal VAE decoder only smooths across as many frames as it is handed at once.
_VAE_DECODE_CHUNK_SIZE = 8
# 0 sizes every pass from the free VRAM; the ceiling is the 25 frame training window.
_WINDOW_FRAMES = 0

_DEFAULT_BATCH_OVERLAP = 2
_ALIGNMENT = 8

# An overlap close to the pass length is self-defeating: passes advance by
# ``window - overlap``, so overlap == window - 1 means one new frame per pass and a 350
# frame clip turns into 350 UNet forwards. Half the window is the widest seam that still
# advances at least as many new frames as it re-predicts. It is deliberately generous: a
# 4 frame seam on a 10 frame pass is a real quality choice worth 32% more passes, while a
# 9 frame seam on that same pass is just waste.
_MAX_OVERLAP_DIVISOR = 2

# IMAGE and MASK arrive as float32 batches, and converting a whole clip of them at once
# costs four bytes per channel per frame on top of the source tensor: a 300 frame 1080p
# clip peaks at over 7 GiB of host RAM before a single frame is inferred. Converting in
# slices keeps that peak flat regardless of clip length.
_CONVERT_CHUNK_FRAMES = 32

# Padding around the union mask bounding box, mirroring Sammie-Roto 2's
# compute_mask_bounding_box: a fraction of the cropped region itself, floored so a
# small subject still gets enough context.
_ROI_BUFFER_RATIO = 0.10
_ROI_BUFFER_MIN_PX = 32
# Below this share of the frame in both dimensions, cropping cannot save enough work to
# be worth the aspect-ratio change.
_ROI_SKIP_RATIO = 0.90

_MODEL_LOCK = threading.RLock()
_DOWNLOAD_LOCK = threading.Lock()
_PIPELINES: dict[tuple, Any] = {}


def _info(message: str) -> None:
    """Emit the readable stage prefix the other video nodes use."""
    _LOGGER.info("[CS VideoMaMa] %s", message)


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

def _model_root() -> Path:
    """Where the VideoMaMa weights live: ``<models>/videomama``.

    The layout under it is the diffusers one the official repos use, ``unet/`` and
    ``vae/`` side by side, which is also what Sammie-Roto 2 keeps under its
    ``checkpoints/videomama``.
    """
    env = (os.environ.get("VIDEOMAMA_MODEL_DIR") or "").strip()
    if env:
        return Path(os.path.abspath(os.path.expanduser(env)))
    try:
        roots = folder_paths.get_folder_paths("videomama")
    except Exception:
        roots = []
    if roots:
        return Path(roots[0])
    return Path(folder_paths.models_dir) / "videomama"


def _register_model_path() -> None:
    try:
        folder_paths.add_model_folder_path(
            "videomama", os.path.join(folder_paths.models_dir, "videomama")
        )
    except Exception:
        _LOGGER.debug("[CS VideoMaMa] could not register the videomama model folder", exc_info=True)


_register_model_path()


def _target_path(root: Path, relative: str) -> Path:
    return root.joinpath(*relative.split("/"))


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    """Official endpoint first, then any mirror configured for restricted networks."""
    path = f"{source['repo']}/resolve/main/{source['file']}"
    mirror = (
        (os.environ.get("VIDEOMAMA_HF_ENDPOINT") or os.environ.get("HF_ENDPOINT") or "")
        .strip()
        .rstrip("/")
    )
    urls = []
    if mirror and mirror != _DEFAULT_HF_ENDPOINT:
        urls.append(f"{mirror}/{path}")
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
        next_mark = 256 * 1024 * 1024
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
                            next_mark += 256 * 1024 * 1024
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
        _LOGGER.warning("[CS VideoMaMa] invalid device %r; falling back to ComfyUI auto device.", choice)
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
    # Holding back a flat 2 GiB is right when the card is mostly free and self-defeating
    # when it is nearly full: at 3 GiB free it leaves room for a single frame per pass,
    # which turns a long clip into hundreds of passes. Scale the floor with what is
    # actually available and let the OOM retry absorb the difference.
    reserve = min(_GPU_MEMORY_RESERVE_BYTES, int(free_bytes * _GPU_MEMORY_RESERVE_RATIO))
    return max(0, min(int(free_bytes * fraction), int(free_bytes) - reserve))


def _is_oom(exc: BaseException) -> bool:
    """True only for allocation failures, so a retry never hides a real error."""
    oom = getattr(torch.cuda, "OutOfMemoryError", ())
    if oom and isinstance(exc, oom):
        return True
    return "out of memory" in str(exc).lower()


def _effective_overlap(requested: int, window: int) -> int:
    """Seam width actually usable for a given pass length."""
    window = max(1, int(window))
    requested = max(0, int(requested))
    if window <= 1:
        return 0
    return min(requested, window // _MAX_OVERLAP_DIVISOR, window - 1)


def _frame_peak_bytes(height: int, width: int) -> float:
    scale = max(0.125, (height * width) / float(_REF_PIXELS))
    return _FRAME_PEAK_BYTES_AT_REF * scale


def _batch_frames(
    total: int,
    height: int,
    width: int,
    free_bytes: int | None,
    override: int,
    ceiling: int,
) -> int:
    """How many frames one UNet pass should cover.

    ``override`` wins when it is positive. Otherwise the budget decides, capped at
    ``ceiling`` because the model was trained on a 25-frame window and temporal
    attention cost grows faster than linearly past it.
    """
    total = max(1, int(total))
    ceiling = max(1, min(int(ceiling), _MAX_BATCH_FRAMES))
    if override > 0:
        return max(1, min(total, int(override)))
    if free_bytes is None:
        # CPU: VRAM is not the constraint, so only bound the host-side buffering.
        return min(total, ceiling, _MAX_BATCH_FRAMES)
    usable = _usable_bytes(free_bytes)
    per_frame = max(1.0, _frame_peak_bytes(height, width))
    # No lower floor on purpose: if the budget only affords one frame, forcing more
    # just moves the OOM from planning time into the middle of a run.
    return max(1, min(total, ceiling, int(usable // per_frame)))


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
    out: list[np.ndarray] = []
    step = max(1, _CONVERT_CHUNK_FRAMES)
    for begin in range(0, int(value.shape[0]), step):
        piece = value[begin: begin + step, ..., :3].to(device="cpu", dtype=torch.float32)
        piece = piece.mul(255.0).add(0.5).clamp(0.0, 255.0).to(torch.uint8).numpy()
        out.extend(np.ascontiguousarray(frame) for frame in piece)
    return out


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

    out: list[np.ndarray] = []
    step = max(1, _CONVERT_CHUNK_FRAMES)
    for begin in range(0, int(value.shape[0]), step):
        piece = value[begin: begin + step].to(device="cpu", dtype=torch.float32).numpy()
        out.extend(
            np.ascontiguousarray(np.where(plane > 0.5, 255, 0).astype(np.uint8)) for plane in piece
        )
    return out


def _mask_roi(masks: list[np.ndarray]) -> tuple[int, int, int, int] | None:
    """Union bounding box of every mask, padded and snapped outward to a multiple of 8.

    One rect is computed for the whole clip and reused for every pass, so the model
    always sees the same spatial region and the seams stay comparable. Returns ``None``
    when the box already covers almost the whole frame, or when no frame has a single
    set pixel.
    """
    if not masks:
        return None
    height, width = masks[0].shape[:2]
    rows = np.zeros(height, dtype=bool)
    cols = np.zeros(width, dtype=bool)
    for mask in masks:
        rows |= np.any(mask > 0, axis=1)
        cols |= np.any(mask > 0, axis=0)
    if not rows.any() or not cols.any():
        return None

    y1 = int(np.argmax(rows))
    y2 = int(height - 1 - np.argmax(rows[::-1]))
    x1 = int(np.argmax(cols))
    x2 = int(width - 1 - np.argmax(cols[::-1]))

    crop_w = x2 - x1 + 1
    crop_h = y2 - y1 + 1
    if crop_w >= width * _ROI_SKIP_RATIO and crop_h >= height * _ROI_SKIP_RATIO:
        return None

    pad_x = max(_ROI_BUFFER_MIN_PX, int(crop_w * _ROI_BUFFER_RATIO))
    pad_y = max(_ROI_BUFFER_MIN_PX, int(crop_h * _ROI_BUFFER_RATIO))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width - 1, x2 + pad_x)
    y2 = min(height - 1, y2 + pad_y)

    x1 = (x1 // _ALIGNMENT) * _ALIGNMENT
    y1 = (y1 // _ALIGNMENT) * _ALIGNMENT
    x2 = min(width - 1, ((x2 + _ALIGNMENT - 1) // _ALIGNMENT) * _ALIGNMENT)
    y2 = min(height - 1, ((y2 + _ALIGNMENT - 1) // _ALIGNMENT) * _ALIGNMENT)

    # The 32 px floor per side swallows the whole frame on small sources, and a crop that
    # covers everything is pure overhead: the model sees the same pixels either way.
    if x1 == 0 and y1 == 0 and x2 >= width - 1 and y2 >= height - 1:
        return None

    return (x1, y1, x2, y2)


def _apply_crop(image: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = rect
    return np.ascontiguousarray(image[y1: y2 + 1, x1: x2 + 1])


def _expand_to_full(image: np.ndarray, rect: tuple[int, int, int, int], width: int, height: int) -> np.ndarray:
    """Paste a cropped matte back onto a black canvas at the source resolution."""
    x1, y1, x2, y2 = rect
    canvas = np.zeros((height, width), dtype=image.dtype)
    canvas[y1: y2 + 1, x1: x2 + 1] = image
    return canvas


def _working_size(height: int, width: int, max_side: int) -> tuple[int, int]:
    """Cap the longest side, then floor both dimensions to a multiple of 8.

    VideoMaMa is an SVD derivative whose reference frame is 1024x576 landscape, and the
    VRAM estimate below is anchored there, so the longest side is the axis worth
    capping. Capping the shorter side instead, which is what the ProPainterX node does,
    would let a 1920x1080 clip through at 1816x1024: three times the pixels, and a
    three times smaller batch for the same card.
    """
    scale = 1.0
    if max_side > 0 and max(height, width) > max_side:
        scale = float(max_side) / float(max(height, width))
    new_h = (int(height * scale) // _ALIGNMENT) * _ALIGNMENT
    new_w = (int(width * scale) // _ALIGNMENT) * _ALIGNMENT
    return max(_ALIGNMENT, new_h), max(_ALIGNMENT, new_w)


def _resize_stack(items: list[np.ndarray], size: tuple[int, int], mask: bool = False) -> list[np.ndarray]:
    height, width = size
    interpolation = cv2.INTER_NEAREST if mask else cv2.INTER_AREA
    out = []
    for item in items:
        if (item.shape[0], item.shape[1]) == (height, width):
            out.append(item)
        else:
            out.append(np.ascontiguousarray(cv2.resize(item, (width, height), interpolation=interpolation)))
    return out


def _degenerate(plane: np.ndarray) -> int | None:
    """0 or 255 when a frame needs no model at all, otherwise ``None``.

    Empty and fully-covered frames are a large share of real rotoscoping work, and
    VideoMaMa has nothing useful to add to them.
    """
    if not plane.any():
        return 0
    if not (plane <= 127).any():
        return 255
    return None


def _alpha_stack(planes: list[np.ndarray]) -> torch.Tensor:
    """uint8 HxW planes -> ComfyUI MASK ``[B,H,W]`` float32 in 0..1."""
    stacked = np.stack(planes)
    stacked = stacked.astype(np.float32)
    stacked /= 255.0
    return torch.from_numpy(np.ascontiguousarray(stacked))


# ------------------------------------------------------------------ windowing

def _window_starts(total: int, window_frames: int, overlap_frames: int) -> list[int]:
    """Global start index of every pass, with no gap in coverage.

    Passes advance by ``window_frames - overlap_frames``. When the last pass would only
    add a handful of new frames it slides back to land on the final frame instead of
    running a short pass, because a 6-frame tail is far from the 25-frame window the
    UNet was trained on.
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


class _MatteSequence:
    """Fold each pass's output into one continuous matte sequence.

    Two mechanisms keep the seams invisible, both taken from Sammie-Roto 2:

    * Soft alpha feedback. The previous pass's own soft mattes for the seam frames are
      fed back in as the *mask conditioning* of the next pass, so the model is steered
      toward its earlier answer instead of being asked cold. This is why a seam of two
      frames is enough here where a blind re-run needs four.
    * A linear cross-fade across the seam, weighted ``(k + 1) / (shared + 1)`` toward
      the newer prediction.

    Seam frames are matched by absolute frame index rather than by position in the
    window. That matters once the final pass slides back: the shared run can be shorter
    than the requested overlap, and a positional slice would silently skip a frame and
    leave a hole.
    """

    def __init__(self, total: int, overlap_frames: int, feedback: bool = True) -> None:
        self.total = int(total)
        self.overlap = max(0, int(overlap_frames))
        self.feedback = bool(feedback)
        self._committed: list[np.ndarray | None] = [None] * self.total
        self._tail_work: dict[int, np.ndarray] = {}
        self._tail_roi: dict[int, np.ndarray] = {}

    def conditioning_for(self, start: int, end: int, fallback: list[np.ndarray]) -> list[np.ndarray]:
        """Mask conditioning for one pass: prior soft mattes where they exist."""
        planes = list(fallback[start:end])
        if not self.feedback or self.overlap <= 0:
            return planes
        for offset in range(len(planes)):
            carried = self._tail_work.get(start + offset)
            if carried is not None:
                planes[offset] = carried
        return planes

    def _shared_run(self, start: int, end: int) -> int:
        """Length of the already-committed run this pass re-predicts, counted from ``start``."""
        if self.overlap <= 0:
            return 0
        shared = 0
        while start + shared < end and (start + shared) in self._tail_roi:
            shared += 1
        return min(shared, self.overlap)

    def add_pass(self, start: int, end: int, work_planes: list[np.ndarray], restore) -> None:
        """Commit one pass. ``restore`` resizes a working-size plane back to the crop rect."""
        length = len(work_planes)
        if length != end - start:
            raise RuntimeError(f"VideoMaMa returned {length} frames for a {end - start} frame window")

        roi_planes = [restore(plane) for plane in work_planes]
        shared = self._shared_run(start, end)

        # Cross-fade the seam: the newer prediction gains weight as it moves away from
        # the seam's leading edge.
        if self.overlap > 0:
            for offset in range(shared):
                previous = self._tail_roi.get(start + offset)
                if previous is None:
                    continue
                weight = float(offset + 1) / float(shared + 1)
                blended = (1.0 - weight) * previous.astype(np.float32)
                blended += weight * roi_planes[offset].astype(np.float32)
                self._committed[start + offset] = np.clip(
                    np.rint(blended), 0.0, 255.0
                ).astype(np.uint8)

        # The seam frames of a non-first pass are warm-up context only; their already
        # committed values are better, so the fresh prediction for them is dropped.
        for offset in range(shared, length):
            self._committed[start + offset] = roi_planes[offset]

        # Carry the tail forward for the next pass. Only frames this pass actually
        # committed qualify: when a slid-back pass shares more frames than it newly
        # commits, the extra leading frames were cross-faded above and must not be
        # offered to the next pass as if they were a clean prediction.
        self._tail_work = {}
        self._tail_roi = {}
        tail_start = max(shared, length - self.overlap)
        for offset in range(tail_start, length):
            self._tail_work[start + offset] = work_planes[offset]
            self._tail_roi[start + offset] = roi_planes[offset]

    def finish(self, fallback: list[np.ndarray]) -> list[np.ndarray]:
        """Every frame must hold a matte; anything missing falls back to its input mask."""
        return [
            plane if plane is not None else fallback[index]
            for index, plane in enumerate(self._committed)
        ]


# ------------------------------------------------------------------ runtime

def _add_bundled_runtime() -> None:
    vendor = Path(__file__).resolve().parent / "videomama_vendor"
    if (vendor / "videomama" / "pipeline.py").is_file():
        value = str(vendor)
        if value not in sys.path:
            sys.path.insert(0, value)


def _pipeline_class() -> Any:
    _add_bundled_runtime()
    try:
        module = importlib.import_module("videomama.pipeline")
    except ImportError as exc:
        raise RuntimeError(
            "The VideoMaMa runtime is unavailable. Reinstall ComfyUI_CineStyle with its "
            "bundled runtime and its diffusers dependency, then restart ComfyUI."
        ) from exc
    return module.VideoMaMaPipeline, module.CancelledError


def _get_pipeline(model_dir: str, device: torch.device, fp16: bool, decode_chunk: int, cpu_offload: bool) -> Any:
    pipeline_class, _ = _pipeline_class()
    key = (
        os.path.abspath(model_dir),
        device.type,
        device.index,
        bool(fp16),
        int(decode_chunk),
        bool(cpu_offload),
    )
    with _MODEL_LOCK:
        pipeline = _PIPELINES.get(key)
        if pipeline is not None:
            return pipeline
        # These are plain diffusers modules, not ComfyUI model patches, so ComfyUI's own
        # memory manager will neither track them nor evict them to make room. Clearing
        # what it does hold first is the only way to avoid starving this node.
        if device.type == "cuda":
            try:
                comfy.model_management.unload_all_models()
            except (RuntimeError, TypeError, ValueError):
                pass
            _clear_cache()
        _unload_pipelines()
        _info(f"loading VideoMaMa on {device} (fp16={fp16}, decode_chunk={decode_chunk})")
        pipeline = pipeline_class(
            model_dir=model_dir,
            device=device,
            fp16=fp16,
            vae_decode_chunk_size=decode_chunk,
            clear_cache=_clear_cache,
        )
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
            _LOGGER.warning("[CS VideoMaMa] unload failed: %s", exc)
    _PIPELINES.clear()
    _clear_cache()
    _info("released models")


class _Progress:
    """Console and UI progress reporting.

    ComfyUI's own bar only reaches the web UI, so a long run looks frozen in the
    terminal even while the UNet is working. Both backends are optional and neither may
    break a run: tqdm is missing in some headless setups, and ComfyUI's bar is missing
    when a node is driven from a script.
    """

    def __init__(self, total_steps: int, label: str = "", console: bool = True) -> None:
        self.total = max(1, int(total_steps))
        self._shown = 0.0
        self._bar = None
        self._console = None
        try:
            self._bar = comfy.utils.ProgressBar(self.total)
        except (RuntimeError, TypeError, ValueError):
            self._bar = None
        if console and tqdm is not None:
            try:
                self._console = tqdm.tqdm(
                    total=self.total,
                    desc=label or "CS VideoMaMa",
                    unit="pass",
                    # Whole passes only: the fractional position is useful for the web bar
                    # but reads as noise in a terminal, and seconds per pass is the number
                    # that actually tells you how long the clip will take.
                    bar_format=(
                        "{l_bar}{bar}| {n:3.0f}/{total:3.0f} passes "
                        "[{elapsed}<{remaining}, {rate_noinv_fmt}]"
                    ),
                    dynamic_ncols=True,
                    leave=True,
                )
            except Exception:
                self._console = None

    def set(self, value: float, label: str = "") -> None:
        # Never rewind. An OOM retry restarts the pass list, and a bar that jumps back
        # reads like a hang; each attempt still gets its own fresh bar.
        value = max(0.0, min(float(value), float(self.total)))
        if value < self._shown:
            value = self._shown
        if self._console is not None:
            try:
                if label:
                    self._console.set_description(label, refresh=False)
                self._console.update(value - self._shown)
            except Exception:
                self._console = None
        self._shown = value
        if self._bar is not None:
            try:
                self._bar.update_absolute(int(value), self.total)
            except (RuntimeError, TypeError, ValueError):
                self._bar = None

    def close(self) -> None:
        if self._console is not None:
            try:
                self._console.close()
            except Exception:
                pass
        self._console = None


# ------------------------------------------------------------------ node

class CSVideoMaMa(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        devices = ["auto", "cpu"]
        if torch.cuda.is_available():
            devices.extend(f"gpu{index}" for index in range(torch.cuda.device_count()))
        return io.Schema(
            node_id=NODE_ID,
            display_name="CS VideoMaMa",
            category=_CATEGORY,
            essentials_category="Video Tools",
            search_aliases=[
                "videomama",
                "matting",
                "alpha matte",
                "rotoscope",
                "mask refine",
                "svd",
            ],
            description=(
                "Turn coarse per-frame binary masks into soft video mattes with VideoMaMa. "
                "Inference runs inside the mask bounding box at a reduced working size, one "
                "pass covers as many frames as the free VRAM affords, and passes are joined "
                "through a short seam that feeds each pass the previous pass's own soft "
                "output. Output keeps the input frame count, order and resolution."
            ),
            inputs=[
                io.Combo.Input(
                    "device",
                    options=devices,
                    default="auto",
                    tooltip="auto follows whatever device ComfyUI is currently using.",
                ),
                io.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip=(
                        "Drives both the UNet noise and the VAE posterior sample, so the same "
                        "seed reproduces a clip bit for bit. The global RNG is snapshotted and "
                        "restored, so other nodes in the workflow are unaffected."
                    ),
                ),
                io.Image.Input("image", tooltip="Video frames as a ComfyUI IMAGE batch."),
                io.Mask.Input(
                    "mask",
                    tooltip=(
                        "Matched per-frame coarse masks; one shared mask broadcasts to every "
                        "frame. Binarised at 0.5, so feathered edges are discarded."
                    ),
                ),
                io.Int.Input(
                    "batch_overlap",
                    display_name="Batch overlap",
                    default=_DEFAULT_BATCH_OVERLAP,
                    min=0,
                    max=32,
                    step=1,
                    tooltip=(
                        "Frames two consecutive passes share. They are re-predicted from the "
                        "previous pass's soft alpha, then cross-faded; 2 is enough because of "
                        "that feedback. 0 disables both and leaves visible seams. Capped at half "
                        "the pass length: a wider seam re-predicts more than half of every pass, "
                        "which costs more than it buys."
                    ),
                ),
                io.Int.Input(
                    "max_resolution",
                    display_name="Max working resolution",
                    default=_DEFAULT_MAX_RESOLUTION,
                    min=256,
                    max=2048,
                    step=8,
                    advanced=True,
                    tooltip=(
                        "Cap on the longest side of the region actually fed to the model. This "
                        "is the main speed control, since pixel count decides how many frames "
                        "fit in one pass. Output always returns to the source resolution."
                    ),
                ),
                io.Int.Input(
                    "motion_bucket_id",
                    display_name="Motion bucket",
                    default=127,
                    min=1,
                    max=255,
                    advanced=True,
                    tooltip=(
                        "SVD motion conditioning, not the clip's real motion. Lower (50-100) = "
                        "subtle, higher (150-200) = dynamic. Leave at 127 unless edges flicker."
                    ),
                ),
                io.Boolean.Input(
                    "cpu_offload",
                    display_name="CPU offload weights",
                    default=False,
                    advanced=True,
                    tooltip=(
                        "Keep weights in RAM and move them per stage. Saves VRAM but costs a "
                        "transfer per pass, a poor trade once a clip needs many passes."
                    ),
                ),
                io.Boolean.Input(
                    "force_unload_model",
                    display_name="Force unload model",
                    default=False,
                    advanced=True,
                    tooltip=(
                        "Release the models when this run ends so other nodes can take the VRAM "
                        "back. ComfyUI does not track these diffusers modules by itself."
                    ),
                ),
            ],
            outputs=[io.Mask.Output("mask", display_name="MASK")],
        )

    @classmethod
    @torch.inference_mode()
    def execute(
        cls,
        # image and mask carry no default because ComfyUI always supplies them, which is
        # also why they lead the list even though the schema shows device and seed first.
        image: torch.Tensor,
        mask: torch.Tensor,
        device: str = "auto",
        seed: int = 42,
        batch_overlap: int = _DEFAULT_BATCH_OVERLAP,
        max_resolution: int = _DEFAULT_MAX_RESOLUTION,
        motion_bucket_id: int = 127,
        cpu_offload: bool = False,
        force_unload_model: bool = False,
    ) -> io.NodeOutput:
        frames_source = image.unsqueeze(0) if image.ndim == 3 else image
        total = int(frames_source.shape[0])
        if total < 1:
            raise ValueError("image batch is empty")

        original_frames = _frames_from_images(frames_source)
        height, width = original_frames[0].shape[:2]
        masks_full = _masks_from_tensor(mask, total)
        if (masks_full[0].shape[0], masks_full[0].shape[1]) != (height, width):
            raise ValueError(
                f"mask is {masks_full[0].shape[1]}x{masks_full[0].shape[0]} but image is "
                f"{width}x{height}"
            )

        # Frames that are already all background or all foreground never need the model.
        fixed = [_degenerate(plane) for plane in masks_full]
        needs_model = [index for index, value in enumerate(fixed) if value is None]
        if not needs_model:
            _info(f"all {total} frames are empty or fully covered; skipping VideoMaMa")
            return io.NodeOutput(
                _alpha_stack([np.full((height, width), fixed[index], dtype=np.uint8) for index in range(total)])
            )

        rect = (
            _mask_roi([masks_full[index] for index in needs_model]) if _CROP_TO_MASK_REGION else None
        )
        if rect is None:
            region_frames = original_frames
            region_masks = masks_full
            region_w, region_h = width, height
        else:
            region_frames = [_apply_crop(frame, rect) for frame in original_frames]
            region_masks = [_apply_crop(plane, rect) for plane in masks_full]
            region_h, region_w = region_frames[0].shape[:2]

        work_h, work_w = _working_size(region_h, region_w, max_resolution)
        needs_resize = (work_h, work_w) != (region_h, region_w)
        work_frames = _resize_stack(region_frames, (work_h, work_w)) if needs_resize else region_frames
        work_masks = _resize_stack(region_masks, (work_h, work_w), mask=True) if needs_resize else region_masks

        compute_device = _device(device)
        use_fp16 = _FP16 and compute_device.type == "cuda"

        root = _model_root()
        _ensure_weights(root)
        pipeline = _get_pipeline(
            str(root), compute_device, use_fp16, _VAE_DECODE_CHUNK_SIZE, bool(cpu_offload)
        )
        cancelled_error = _pipeline_class()[1]

        # Measured after the weights are resident: planning against the free memory that
        # existed before the load is how a run OOMs partway through.
        free_bytes = _free_vram_bytes(compute_device)
        window = _batch_frames(
            total, work_h, work_w, free_bytes, _WINDOW_FRAMES, _TRAINED_WINDOW_FRAMES
        )

        free_text = "cpu" if free_bytes is None else f"{free_bytes / 1048576:.0f} MiB free"
        region_text = "full frame" if rect is None else f"crop {region_w}x{region_h} at {rect[0]},{rect[1]}"

        # The UNet noise comes from a generator seeded with ``seed``, but the VAE
        # posterior sample draws from the global RNG, which the seed would otherwise not
        # reach. Snapshot, reseed for this clip, and hand the state back afterwards so the
        # next node in the workflow sees exactly what it would have without this one.
        rng_cpu = torch.random.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state_all() if compute_device.type == "cuda" else None
        torch.manual_seed(int(seed))
        if rng_cuda is not None:
            torch.cuda.manual_seed_all(int(seed))

        def restore(plane: np.ndarray) -> np.ndarray:
            if (plane.shape[0], plane.shape[1]) == (region_h, region_w):
                return plane
            return np.ascontiguousarray(
                cv2.resize(plane, (region_w, region_h), interpolation=cv2.INTER_LINEAR)
            )

        progress: Any = None
        try:
            for attempt in range(_MAX_ATTEMPTS):
                overlap = _effective_overlap(batch_overlap, window)
                if attempt == 0 and overlap != int(batch_overlap):
                    # Say it out loud: a knob silently doing nothing, or silently
                    # multiplying the pass count, is otherwise invisible in the output.
                    if window <= 1:
                        reason = "one frame per pass leaves no seam to blend"
                    else:
                        reason = (
                            f"{_effective_overlap(1 << 30, window)} is the ceiling for a "
                            f"{window} frame pass; a wider seam re-predicts more than half of "
                            "each pass, which costs more than it buys"
                        )
                    _info(f"batch_overlap {int(batch_overlap)} reduced to {overlap}: {reason}")
                starts = _window_starts(total, window, overlap)
                if attempt == 0:
                    _info(
                        f"frames={total} source={width}x{height} {region_text} working={work_w}x{work_h} "
                        f"device={compute_device} ({free_text}); passes={len(starts)} batch={window} "
                        f"overlap={overlap} feedback={_SOFT_ALPHA_FEEDBACK and overlap > 0} "
                        f"decode_chunk={_VAE_DECODE_CHUNK_SIZE} seed={int(seed)} "
                        f"fp16={use_fp16} vram_fraction={_VRAM_FRACTION}"
                    )
                else:
                    _info(f"out of memory mid-clip: restarting the run at {window} frames per pass")

                if progress is not None:
                    progress.close()
                sequence = _MatteSequence(total, overlap, feedback=_SOFT_ALPHA_FEEDBACK and overlap > 0)
                progress = _Progress(
                    len(starts),
                    label=f"CS VideoMaMa {work_w}x{work_h} {total}f",
                    console=_CONSOLE_PROGRESS,
                )
                try:
                    for pass_index, start in enumerate(starts):
                        end = min(total, start + window)
                        _check_interrupt()

                        # Whole-pass shortcut: a run of empty or fully covered frames costs
                        # a full UNet forward for an answer that is already known.
                        if all(fixed[index] is not None for index in range(start, end)):
                            planes = [
                                np.full((work_h, work_w), fixed[index], dtype=np.uint8)
                                for index in range(start, end)
                            ]
                        else:
                            conditioning = sequence.conditioning_for(start, end, work_masks)

                            def on_progress(
                                _label: str,
                                done: int,
                                stage_total: int,
                                base: float = float(pass_index),
                                first: int = start,
                                last: int = end - 1,
                                passes: int = len(starts),
                                idx: int = pass_index,
                            ) -> None:
                                fraction = 0.0 if not stage_total else min(1.0, float(done) / float(stage_total))
                                progress.set(
                                    base + fraction,
                                    f"pass {idx + 1}/{passes} · frames {first}-{last}",
                                )

                            try:
                                planes = pipeline.run(
                                    cond_frames=work_frames[start:end],
                                    mask_frames=conditioning,
                                    seed=int(seed),
                                    fps=_FPS,
                                    motion_bucket_id=int(motion_bucket_id),
                                    noise_aug_strength=_NOISE_AUG_STRENGTH,
                                    on_progress=on_progress,
                                    should_cancel=_interrupted,
                                )
                            except cancelled_error:
                                _info("cancelled by user")
                                _check_interrupt()
                                raise

                            # Per-frame overrides still apply inside a mixed pass.
                            for offset, index in enumerate(range(start, end)):
                                if fixed[index] is not None and offset < len(planes):
                                    planes[offset] = np.full(
                                        (work_h, work_w), fixed[index], dtype=np.uint8
                                    )

                        sequence.add_pass(start, end, planes, restore)
                        planes = None
                        progress.set(
                            pass_index + 1,
                            f"pass {pass_index + 1}/{len(starts)} · frames {start}-{end - 1} done",
                        )
                        # No cache flush here. Every pass has the same shape, so the
                        # allocator reuses its blocks between them; flushing forces a full
                        # device sync plus fresh cudaMalloc on the next pass, and on a long
                        # clip that overhead is paid hundreds of times for nothing.
                except RuntimeError as exc:
                    if not _is_oom(exc) or window <= 1 or attempt + 1 >= _MAX_ATTEMPTS:
                        raise
                    _clear_cache()
                    window = max(1, window // 2)
                    continue
                break
            # Once for the whole run, after the last pass, to hand the working set back.
            _clear_cache()
        finally:
            if progress is not None:
                progress.close()
            if force_unload_model:
                with _MODEL_LOCK:
                    _unload_pipelines()
            torch.random.set_rng_state(rng_cpu)
            if rng_cuda is not None:
                try:
                    torch.cuda.set_rng_state_all(rng_cuda)
                except (RuntimeError, TypeError, ValueError):
                    pass

        blended = sequence.finish(region_masks)
        if len(blended) != total:
            raise RuntimeError(f"built {len(blended)} mattes for a {total} frame clip")

        if rect is None:
            mattes = blended
        else:
            mattes = [_expand_to_full(plane, rect, width, height) for plane in blended]

        # A fully covered frame is foreground across the whole frame, not just inside the
        # crop rect, so its constant is applied after the paste back rather than before.
        for index, value in enumerate(fixed):
            if value is not None:
                mattes[index] = np.full((height, width), value, dtype=np.uint8)

        output = _alpha_stack(mattes)
        _info(f"done: {output.shape[0]} mattes at {output.shape[2]}x{output.shape[1]}")
        return io.NodeOutput(output)


class VideoMaMaExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CSVideoMaMa]


async def comfy_entrypoint() -> VideoMaMaExtension:
    return VideoMaMaExtension()
