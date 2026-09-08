"""Adaptive multi-anchor MatAnyone2 video matting for ComfyUI."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import importlib
import io as py_io
import json
import logging
import math
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from aiohttp import web
from PIL import Image
from typing_extensions import override
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - ComfyUI normally bundles tqdm
    tqdm = None

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, io


NODE_ID = "CS_MatAnyone2"
MODEL_FILENAME = "matanyone2.pth"
MODEL_URL = "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"
MODEL_MD5 = "b1d3cfbb7596ecf3b88391198427ca95"
_CATEGORY = "😺dzNodes/CineStyle/Video"
_LOGGER = logging.getLogger("CineStyleMatAnyone2")
_ROUTES_REGISTERED = False
_MODEL_LOCK = threading.RLock()
_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_DOWNLOAD_LOCK = threading.Lock()
_CACHE_LOCK = threading.RLock()
_MASK_CACHE: dict[str, dict[str, Any]] = {}
_PREVIEW_STORE = None
_CACHE_LIMIT = 2


def _matanyone_info(message: str) -> None:
    """Emit the same readable stage prefix used by the video nodes."""
    # ComfyUI's logger renders the INFO level tag in green; keep the node
    # name in the message so it follows in the normal white console colour,
    # matching CS Load Video.
    _LOGGER.info("[CS MatAnyone2] %s", message)


try:
    folder_paths.add_model_folder_path("matanyone", os.path.join(folder_paths.models_dir, "matanyone"))
except Exception:
    pass


def _preview_store():
    global _PREVIEW_STORE
    if _PREVIEW_STORE is None:
        package = __name__.rsplit(".", 1)[0]
        module = sys.modules.get(f"{package}._py_preview_cache")
        if module is None:
            raise RuntimeError("CineStyle preview cache module is unavailable.")
        _PREVIEW_STORE = module.PreviewCacheStore(
            "matanyone2",
            max_entries=_CACHE_LIMIT,
            max_bytes=4 * 1024**3,
        )
    return _PREVIEW_STORE


def _cache_root() -> Path:
    root = Path(folder_paths.get_temp_directory()) / "cinestyle_preview_cache" / "matanyone2_masks"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_options() -> list[str]:
    try:
        names = [name for name in folder_paths.get_filename_list("matanyone") if name.lower().endswith((".pth", ".pt"))]
    except Exception:
        names = []
    return names or [MODEL_FILENAME]


def _model_path(filename: str) -> Path:
    name = str(filename or MODEL_FILENAME).strip() or MODEL_FILENAME
    existing = folder_paths.get_full_path("matanyone", name)
    if existing:
        return Path(existing)
    if Path(name).name != MODEL_FILENAME:
        raise FileNotFoundError(f"MatAnyone2 checkpoint not found: {name}")
    try:
        roots = folder_paths.get_folder_paths("matanyone")
    except Exception:
        roots = []
    target = Path(roots[0] if roots else Path(folder_paths.models_dir) / "matanyone") / MODEL_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    with _DOWNLOAD_LOCK:
        if target.is_file() and target.stat().st_size > 0:
            return target
        partial = target.with_suffix(target.suffix + ".download")
        _matanyone_info(f"downloading checkpoint: {target}")
        try:
            torch.hub.download_url_to_file(MODEL_URL, str(partial), progress=True)
            if not partial.is_file() or partial.stat().st_size <= 0:
                raise RuntimeError("downloaded checkpoint is empty")
            if _md5(partial) != MODEL_MD5:
                raise RuntimeError("downloaded checkpoint checksum does not match the official release")
            os.replace(partial, target)
        except Exception as exc:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"Unable to download MatAnyone2. Download {MODEL_URL} manually and place it at {target}. "
                f"Original error: {exc}"
            ) from exc
    return target


def _add_bundled_runtime() -> None:
    local_vendor = Path(__file__).resolve().parent / "matanyone2_vendor"
    candidates = [local_vendor]
    for parent in Path(__file__).resolve().parents:
        if parent.name.lower() == "custom_nodes" and parent.is_dir():
            candidates.extend(parent.glob("*/vendor/MatAnyone2"))
    for candidate in candidates:
        if (candidate / "matanyone2" / "model" / "matanyone2.py").is_file():
            value = str(candidate)
            if value not in sys.path:
                sys.path.insert(0, value)
            return


def _matanyone_imports():
    _add_bundled_runtime()
    try:
        inference_module = importlib.import_module("matanyone2.inference.inference_core")
        model_module = importlib.import_module("matanyone2.model.matanyone2")
        from hydra import compose, initialize_config_module
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf, open_dict
    except ImportError as exc:
        raise RuntimeError(
            "MatAnyone2 runtime is unavailable. Reinstall ComfyUI_CineStyle with its bundled runtime "
            "and dependencies, then restart ComfyUI."
        ) from exc
    return (
        inference_module.InferenceCore,
        model_module.MatAnyone2,
        compose,
        initialize_config_module,
        GlobalHydra,
        OmegaConf,
        open_dict,
    )


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
        _LOGGER.warning("[CS MatAnyone2] invalid device %r; falling back to ComfyUI auto device.", choice)
    result = comfy.model_management.get_torch_device()
    return result if isinstance(result, torch.device) else torch.device(result)


def _check_interrupt() -> None:
    checker = getattr(comfy.model_management, "throw_exception_if_processing_interrupted", None)
    if callable(checker):
        checker()


def _load_model(path: Path, device: torch.device) -> Any:
    key = (str(path.resolve()), str(device))
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            cached.to(device).eval()
            return cached

        _, model_class, compose, initialize_config_module, global_hydra, _, open_dict = _matanyone_imports()
        # The upstream model module keeps a module-level device used by
        # encode_image() for its normalization buffers. Keep it aligned with
        # the device selected by this node (notably for explicit CPU mode).
        try:
            importlib.import_module("matanyone2.model.matanyone2").device = device
        except Exception:
            pass
        hydra = global_hydra.instance()
        if hydra.is_initialized():
            hydra.clear()
        try:
            initialize_config_module(
                version_base="1.3.2",
                config_module="matanyone2.config",
                job_name="cinestyle_matanyone2",
            )
            cfg = compose(config_name="eval_matanyone_config")
            with open_dict(cfg):
                cfg.weights = str(path)
                cfg.max_internal_size = -1
                cfg.model.pretrained_resnet = False
            model = model_class(cfg, single_object=True).to(device).eval()
            try:
                state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
            except TypeError:
                state_dict = torch.load(str(path), map_location="cpu")
            except Exception:
                # Older releases may contain pickled containers not accepted
                # by the restricted loader; the checkpoint is user-selected
                # and has already passed the optional checksum validation.
                state_dict = torch.load(str(path), map_location="cpu", weights_only=False)
            model.load_weights(state_dict)
            _MODEL_CACHE[key] = model
            return model
        finally:
            if hydra.is_initialized():
                hydra.clear()


def _unload_model(path: Path, device: torch.device, model: Any) -> None:
    key = (str(path.resolve()), str(device))
    with _MODEL_LOCK:
        _MODEL_CACHE.pop(key, None)
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    empty_cache = getattr(comfy.model_management, "soft_empty_cache", None)
    if callable(empty_cache):
        empty_cache()


def _normalise_images(images: Any) -> torch.Tensor:
    if not isinstance(images, torch.Tensor):
        raise ValueError("image must be a ComfyUI IMAGE tensor.")
    value = images.unsqueeze(0) if images.ndim == 3 else images
    if value.ndim != 4 or value.shape[-1] < 3 or value.shape[0] < 1:
        raise ValueError("image must have shape [frames, height, width, 3 or 4].")
    return value[..., :3].detach().to(device="cpu", dtype=torch.float32).nan_to_num().clamp_(0.0, 1.0).contiguous()


def _normalise_masks(mask: Any, frames: int, height: int, width: int) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise ValueError("mask must be a ComfyUI MASK tensor.")
    value = mask
    if value.ndim == 2:
        value = value.unsqueeze(0)
    elif value.ndim == 4:
        # Accept both common layouts emitted by ComfyUI/custom nodes:
        # [frames, 1, height, width] and [frames, height, width, 1].
        # Resolve degenerate one-pixel dimensions from the expected spatial
        # size before falling back to channel-position heuristics.
        nchw_match = value.shape[1] == 1 and tuple(value.shape[2:]) == (height, width)
        nhwc_match = value.shape[-1] == 1 and tuple(value.shape[1:3]) == (height, width)
        if nchw_match and not nhwc_match:
            value = value[:, 0]
        elif nhwc_match and not nchw_match:
            value = value[..., 0]
        elif value.shape[-1] == 1:
            value = value[..., 0]
        elif value.shape[1] == 1:
            value = value[:, 0]
        else:
            raise ValueError("mask must have one channel when a 4-D tensor is provided.")
    if value.ndim != 3 or value.shape[0] < 1:
        raise ValueError("mask must have shape [H,W], [frames,H,W], or [frames,H,W,1].")
    value = value.detach().to(device="cpu", dtype=torch.float32).nan_to_num().clamp_(0.0, 1.0)
    if value.shape[0] != frames:
        _LOGGER.warning(
            "[CS MatAnyone2] mask frame count %d does not match image frame count %d; "
            "resampling masks by nearest frame index.",
            int(value.shape[0]),
            int(frames),
        )
        indices = torch.linspace(0, value.shape[0] - 1, frames).round().long()
        value = value.index_select(0, indices)
    if tuple(value.shape[-2:]) != (height, width):
        _LOGGER.warning(
            "[CS MatAnyone2] mask size %dx%d does not match image size %dx%d; resizing masks.",
            int(value.shape[-1]),
            int(value.shape[-2]),
            int(width),
            int(height),
        )
        value = F.interpolate(value.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False)[:, 0]
    return value.contiguous().clamp_(0.0, 1.0)


def _target_size(height: int, width: int, max_megapixels: float) -> tuple[int, int]:
    try:
        megapixels = float(max_megapixels)
    except (TypeError, ValueError):
        megapixels = 2.1
    if not math.isfinite(megapixels):
        megapixels = 2.1
    limit = max(0.05, megapixels) * 1_000_000.0
    pixels = height * width
    if pixels <= limit:
        return height, width
    scale = math.sqrt(limit / pixels)
    return max(16, int(round(height * scale))), max(16, int(round(width * scale)))


def _lanczos_resize_images(images: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if tuple(images.shape[1:3]) == (height, width):
        return images
    output = torch.empty((images.shape[0], height, width, images.shape[-1]), dtype=torch.float32)
    for index in range(images.shape[0]):
        array = images[index].numpy()
        resized = cv2.resize(array, (width, height), interpolation=cv2.INTER_LANCZOS4)
        output[index].copy_(torch.from_numpy(np.ascontiguousarray(resized)))
    return output.clamp_(0.0, 1.0)


def _resize_masks(masks: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if tuple(masks.shape[1:3]) == (height, width):
        return masks
    output = torch.empty((masks.shape[0], height, width), dtype=torch.float32)
    for index in range(masks.shape[0]):
        # Lanczos is used for RGB frames.  Alpha/seed masks use area when
        # reducing and linear interpolation when restoring size so a hard
        # edge does not acquire ringing halos that would affect matting.
        interpolation = cv2.INTER_AREA if height < masks.shape[1] or width < masks.shape[2] else cv2.INTER_LINEAR
        resized = cv2.resize(masks[index].numpy(), (width, height), interpolation=interpolation)
        output[index].copy_(torch.from_numpy(np.ascontiguousarray(resized)))
    return output.clamp_(0.0, 1.0)


def _parse_anchors(value: Any, frame_count: int) -> list[int]:
    if isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        text = str(value or "").strip()
        if not text:
            raw = [0]
        else:
            try:
                parsed = json.loads(text)
                raw = parsed if isinstance(parsed, list) else [parsed]
            except json.JSONDecodeError:
                raw = [item for item in text.replace(";", ",").replace(" ", ",").split(",") if item]
    try:
        anchors = sorted({int(item) for item in raw})
    except (TypeError, ValueError) as exc:
        _LOGGER.warning("[CS MatAnyone2] invalid anchor_frames %r; falling back to frame 0.", value)
        anchors = [0]
    if not anchors:
        anchors = [0]
    if frame_count <= 1:
        return [0]
    # Anchor spacing/limit are analysis controls only. User-authored anchors
    # are retained, while malformed/out-of-range values are made usable by
    # clamping them to the current frame range instead of aborting execution.
    clamped = sorted({max(0, min(frame_count - 1, item)) for item in anchors}) or [0]
    if clamped != anchors:
        _LOGGER.warning("[CS MatAnyone2] anchor_frames %s were clamped to frame range 0..%d: %s", anchors, frame_count - 1, clamped)
    return clamped


def _effective_overlap(anchors: list[int], overlap: int) -> int:
    """Clamp overlap to a safe seam width without rejecting user anchors."""
    try:
        requested = max(0, int(overlap))
    except (TypeError, ValueError, OverflowError):
        requested = 0
    if len(anchors) < 2:
        return requested
    shortest = min(right - left for left, right in zip(anchors, anchors[1:]))
    # Adjacent windows must retain at least one non-overlap frame.  This keeps
    # the cosine seam well-defined even when anchors are intentionally close.
    safe_max = max(0, (shortest - 1) // 2)
    if requested > safe_max:
        _LOGGER.info(
            "[CS MatAnyone2] overlap=%d exceeds safe value %d for anchor spacing; using %d.",
            requested,
            safe_max,
            safe_max,
        )
    return min(requested, safe_max)


def _seed_mask(mask: torch.Tensor, threshold: float, morphology: int) -> torch.Tensor:
    binary = (mask >= float(threshold)).numpy().astype(np.uint8)
    radius = abs(int(morphology))
    if radius > 0:
        size = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        binary = cv2.dilate(binary, kernel) if morphology > 0 else cv2.erode(binary, kernel)
    return torch.from_numpy(binary.astype(np.float32) * 255.0)


def _runtime_cfg(
    model: Any,
    memory_interval: int,
    memory_frames: int,
    use_long_term: bool,
    height: int,
    width: int,
) -> Any:
    _, _, _, _, _, omega_conf, open_dict = _matanyone_imports()
    cfg = omega_conf.create(omega_conf.to_container(model.cfg, resolve=False))
    with open_dict(cfg):
        cfg.mem_every = max(1, int(memory_interval))
        cfg.max_mem_frames = max(2, int(memory_frames))
        cfg.use_long_term = bool(use_long_term)
        cfg.max_internal_size = -1
        if bool(use_long_term):
            # MemoryManager subtracts one frame from these values because the
            # first frame is permanent. Keep at least one working frame and
            # one consolidation candidate even when the user selects the
            # smallest UI value (otherwise its top-k prototype reduction can
            # receive an empty/undersized candidate tensor).
            lt_max_frames = max(3, int(memory_frames))
            lt_min_frames = max(2, min(lt_max_frames - 1, int(memory_frames) - 1))
            cfg.long_term.max_mem_frames = lt_max_frames
            cfg.long_term.min_mem_frames = lt_min_frames
            # The default 128 prototypes exceed the available token count in
            # small previews; consolidation calls torch.topk directly.
            spatial_tokens = max(1, math.ceil(height / 16) * math.ceil(width / 16))
            cfg.long_term.num_prototypes = min(
                max(1, int(cfg.long_term.num_prototypes)), spatial_tokens
            )
        # The reference config uses top_k=30.  At the first propagated frame
        # the memory contains only one frame, so the available token count is
        # the spatial token count (not ``spatial_tokens * memory_frames``).
        # Clamp to that first-frame capacity; otherwise small previews (for
        # example 32x32 => four tokens) fail in torch.topk before memory can
        # grow to its configured frame budget.
        spatial_tokens = max(1, math.ceil(height / 16) * math.ceil(width / 16))
        cfg.top_k = min(max(1, int(cfg.top_k)), spatial_tokens)
    return cfg


def _new_processor(model: Any, cfg: Any, device: torch.device) -> Any:
    inference_core, *_ = _matanyone_imports()
    try:
        return inference_core(model, cfg=cfg, device=device)
    except TypeError:
        return inference_core(model, cfg=cfg)


def _alpha_from_output(processor: Any, output: torch.Tensor) -> torch.Tensor:
    alpha = processor.output_prob_to_mask(output)
    while alpha.ndim > 2:
        alpha = alpha[0]
    return alpha.detach().to(device="cpu", dtype=torch.float32).clamp_(0.0, 1.0)


@torch.inference_mode()
def _run_direction(
    model: Any,
    cfg: Any,
    frames: torch.Tensor,
    seed: torch.Tensor,
    device: torch.device,
    warmup: int,
    progress: Any,
) -> torch.Tensor:
    processor = _new_processor(model, cfg, device)
    autocast = torch.amp.autocast(device_type="cuda", enabled=True) if device.type == "cuda" else contextlib.nullcontext()
    with autocast:
        anchor = frames[0].movedim(-1, 0).to(device=device, dtype=torch.float32)
        seed = seed.to(device=device, dtype=torch.float32)
        output = processor.step(anchor, seed, objects=[1])
        output = processor.step(anchor, first_frame_pred=True)
        # Match the official warmup convention: one first-frame prediction
        # after seed injection plus ``warmup`` repeated predictions.
        for _ in range(max(1, int(warmup))):
            output = processor.step(anchor, first_frame_pred=True)
        result = torch.empty((frames.shape[0], frames.shape[1], frames.shape[2]), dtype=torch.float32)
        result[0] = _alpha_from_output(processor, output)
        progress.update(1)
        for index in range(1, frames.shape[0]):
            _check_interrupt()
            frame = frames[index].movedim(-1, 0).to(device=device, dtype=torch.float32)
            # Keep the final frame on the normal memory-update path.  The
            # upstream ``process_video`` loop leaves ``end=False`` for every
            # frame; using ``end=True`` here would skip the last memory update
            # and makes this segmented pass behave differently from the
            # reference implementation.
            output = processor.step(frame)
            result[index] = _alpha_from_output(processor, output)
            progress.update(1)
    del processor
    return result


class _Progress:
    def __init__(self, total: int):
        self.total = max(1, int(total))
        self.value = 0
        self.backend = comfy.utils.ProgressBar(self.total)
        self.bar = (
            tqdm(
                total=self.total,
                desc="[INFO] [CS MatAnyone2] frame processing",
                unit="frame",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                mininterval=0.1,
                dynamic_ncols=True,
                leave=True,
            )
            if tqdm is not None
            else None
        )

    def update(self, amount: int = 1) -> None:
        step = max(0, int(amount))
        self.value = min(self.total, self.value + step)
        self.backend.update_absolute(self.value, self.total)
        if self.bar is not None:
            self.bar.update(step)

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()


def _anchor_windows(anchors: list[int], frame_count: int, overlap: int) -> list[tuple[int, int]]:
    windows = []
    for index, anchor in enumerate(anchors):
        left_mid = 0 if index == 0 else (anchors[index - 1] + anchor) // 2
        right_mid = frame_count - 1 if index == len(anchors) - 1 else (anchor + anchors[index + 1]) // 2
        left = 0 if index == 0 else max(0, left_mid - overlap)
        right = frame_count - 1 if index == len(anchors) - 1 else min(frame_count - 1, right_mid + overlap)
        windows.append((left, right))
    return windows


def _anchor_weight(position: int, anchor_index: int, anchors: list[int], overlap: int) -> float:
    """Return a complementary half-cosine weight at an adjacent-anchor seam."""
    if overlap <= 0:
        return 1.0
    weight = 1.0
    anchor = anchors[anchor_index]
    if anchor_index > 0:
        seam = (anchors[anchor_index - 1] + anchor) / 2.0
        start = seam - overlap
        end = seam + overlap
        if start <= position <= end:
            u = (position - start) / max(1.0, end - start)
            # This is the right-hand anchor at the seam: its confidence
            # rises from zero to one as we move away from the previous
            # anchor.  The previous anchor uses the complementary descending
            # half-cosine below.
            weight *= 0.5 * (1.0 - math.cos(math.pi * u))
    if anchor_index + 1 < len(anchors):
        seam = (anchor + anchors[anchor_index + 1]) / 2.0
        start = seam - overlap
        end = seam + overlap
        if start <= position <= end:
            u = (position - start) / max(1.0, end - start)
            weight *= 0.5 * (1.0 + math.cos(math.pi * u))
    return max(1e-4, weight)


@torch.inference_mode()
def _matte(
    model: Any,
    frames: torch.Tensor,
    masks: torch.Tensor,
    anchors: list[int],
    device: torch.device,
    warmup: int,
    threshold: float,
    morphology: int,
    overlap: int,
    memory_interval: int,
    memory_frames: int,
    use_long_term: bool,
) -> torch.Tensor:
    count, height, width = frames.shape[:3]
    windows = _anchor_windows(anchors, count, overlap)
    total = sum(
        (right - anchor + 1) + (anchor - left + 1 if left < anchor else 0)
        for (left, right), anchor in zip(windows, anchors)
    )
    progress = _Progress(total)
    cfg = _runtime_cfg(model, memory_interval, memory_frames, use_long_term, height, width)
    weighted = torch.zeros((count, height, width), dtype=torch.float32)
    weights = torch.zeros(count, dtype=torch.float32)
    canonical: dict[int, torch.Tensor] = {}
    try:
        for anchor_index, (anchor, (left, right)) in enumerate(zip(anchors, windows)):
            _matanyone_info(f"anchor {anchor_index + 1}/{len(anchors)} frame {anchor}: forward {anchor}->{right}")
            seed = _seed_mask(masks[anchor], threshold, morphology)
            if not bool((seed > 0).any()):
                raise ValueError(
                    f"Anchor frame {anchor} has an empty seed mask after threshold/morphology. "
                    "Choose a different anchor or adjust mask_threshold/seed_morphology."
                )
            forward = _run_direction(model, cfg, frames[anchor : right + 1], seed, device, warmup, progress)
            canonical[anchor] = forward[0].clone()
            backward = None
            if left < anchor:
                _matanyone_info(f"anchor {anchor_index + 1}/{len(anchors)} frame {anchor}: backward {anchor}->{left}")
                backward = _run_direction(model, cfg, frames[left : anchor + 1].flip(0), seed, device, warmup, progress).flip(0)

            for frame_index in range(left, right + 1):
                alpha = backward[frame_index - left] if frame_index < anchor and backward is not None else forward[frame_index - anchor]
                confidence = _anchor_weight(frame_index, anchor_index, anchors, overlap)
                weighted[frame_index].add_(alpha, alpha=confidence)
                weights[frame_index] += confidence
            del forward, backward

        _matanyone_info("cosine overlap blending")
        output = weighted / weights.clamp_min(1e-6).view(-1, 1, 1)
        for anchor, alpha in canonical.items():
            output[anchor] = alpha
        return output.clamp_(0.0, 1.0)
    finally:
        progress.close()


def _mask_cache_put(node_id: Any, masks: torch.Tensor) -> dict[str, Any]:
    key = str(node_id or "").strip()
    if not key:
        raise ValueError("MatAnyone2 preview cache requires a node id.")
    array = masks.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0).mul(255).round().to(torch.uint8).numpy()
    path = _cache_root() / f"{uuid.uuid4().hex}.npy"
    np.save(path, array, allow_pickle=False)
    entry = {"token": f"matanyone2-mask:{uuid.uuid4().hex}", "path": str(path), "shape": tuple(array.shape), "created": time.time()}
    with _CACHE_LOCK:
        previous = _MASK_CACHE.pop(key, None)
        _MASK_CACHE[key] = entry
        while len(_MASK_CACHE) > _CACHE_LIMIT:
            old_key, old = next(iter(_MASK_CACHE.items()))
            _MASK_CACHE.pop(old_key, None)
            Path(str(old.get("path") or "")).unlink(missing_ok=True)
    if previous:
        Path(str(previous.get("path") or "")).unlink(missing_ok=True)
    return entry


def _cache_inputs(node_id: Any, images: torch.Tensor, masks: torch.Tensor, fps: float = 24.0) -> None:
    preview_height, preview_width = _target_size(int(images.shape[1]), int(images.shape[2]), 1.0)
    preview_images = _lanczos_resize_images(images, preview_height, preview_width)
    preview_masks = _resize_masks(masks, preview_height, preview_width)
    _preview_store().put_preview(node_id, preview_images, fps, encode_video=False, force=True)
    _mask_cache_put(node_id, preview_masks)


def _cache_wait_input(node_id: Any, prompt: Any, images: torch.Tensor) -> None:
    package = __name__.rsplit(".", 1)[0]
    module = sys.modules.get(f"{package}._py_preview_cache")
    if module is None:
        return
    chain = module.build_input_chain(prompt, node_id, ("image",))
    if chain is not None:
        module.get_wait_input_cache_store().put_chain(chain, images, 24.0, force=True)


def _cached_pair(node_id: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    image_entry = _preview_store().get_preview_variant(node_id, proxy=False)
    with _CACHE_LOCK:
        mask_entry = _MASK_CACHE.get(str(node_id or "").strip())
    if image_entry is None or mask_entry is None:
        raise ValueError("No Matte Preview cache is available. Enable wait for input cache and run the workflow once.")
    if not Path(str(mask_entry.get("path") or "")).is_file():
        raise ValueError("The Matte Preview mask cache expired. Run the workflow again.")
    return image_entry, mask_entry


def _preview_data_url(image: np.ndarray, mask: np.ndarray) -> str:
    rgb = image.astype(np.float32) / 255.0
    alpha = np.clip(mask.astype(np.float32) / 255.0, 0.0, 1.0)[..., None] * 0.48
    color = np.asarray([0.31, 0.76, 0.93], dtype=np.float32)
    result = np.clip(rgb * (1.0 - alpha) + color * alpha, 0.0, 1.0)
    encoded = Image.fromarray(np.rint(result * 255).astype(np.uint8), mode="RGB")
    buffer = py_io.BytesIO()
    encoded.save(buffer, format="JPEG", quality=91, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _analysis_features(mask_frames: np.ndarray, stride: int, threshold: float) -> tuple[list[int], np.ndarray]:
    frame_count, height, width = mask_frames.shape
    indices = list(range(0, frame_count, max(1, int(stride))))
    if indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    scale = min(1.0, 320.0 / max(height, width))
    ah, aw = max(24, int(round(height * scale))), max(24, int(round(width * scale)))
    binary = []
    for index in indices:
        small = cv2.resize(mask_frames[index], (aw, ah), interpolation=cv2.INTER_AREA)
        binary.append(small >= int(round(threshold * 255.0)))
    scores = np.zeros(len(indices), dtype=np.float32)
    for pos in range(1, len(indices)):
        previous, current = binary[pos - 1], binary[pos]
        intersection = np.logical_and(previous, current).sum()
        union = np.logical_or(previous, current).sum()
        iou_error = 1.0 - (intersection / union if union else 1.0)
        area_a, area_b = float(previous.mean()), float(current.mean())
        area_error = min(1.0, abs(math.log((area_b + 1e-4) / (area_a + 1e-4))) / 1.5)
        edge_a = cv2.morphologyEx(previous.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        edge_b = cv2.morphologyEx(current.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        edge_union = np.logical_or(edge_a, edge_b).sum()
        edge_error = np.logical_xor(edge_a, edge_b).sum() / edge_union if edge_union else 0.0
        quality = 1.0 if 0.001 <= area_b <= 0.98 else 0.2
        scores[pos] = quality * (0.55 * iou_error + 0.25 * area_error + 0.20 * edge_error)
    return indices, scores


def _select_anchors(
    indices: list[int],
    scores: np.ndarray,
    frame_count: int,
    min_spacing: int,
    hysteresis: int,
    limit: int,
    sensitivity: float,
    valid_frames: set[int] | None = None,
) -> list[int]:
    valid = [frame for frame in indices if valid_frames is None or frame in valid_frames]
    # Preview analysis should not propose an empty seed frame.  Execution
    # still keeps the documented no-preview default of frame 0; this fallback
    # only affects automatically generated candidates.
    selected = [valid[0] if valid else 0]
    if limit <= 1:
        return selected
    radius = max(0, int(math.ceil(hysteresis / max(1, indices[1] - indices[0])))) if len(indices) > 1 else 0
    radius = min(radius, max(0, len(scores) // 2))
    if radius:
        kernel = np.ones(radius * 2 + 1, dtype=np.float32)
        smooth = np.convolve(scores, kernel / kernel.sum(), mode="same")
    else:
        smooth = scores
    candidates = []
    for pos in range(1, len(indices) - 1):
        if valid_frames is not None and indices[pos] not in valid_frames:
            continue
        if smooth[pos] >= sensitivity and smooth[pos] >= smooth[pos - 1] and smooth[pos] >= smooth[pos + 1]:
            candidates.append((float(smooth[pos]), indices[pos]))
    candidates.sort(reverse=True)
    for _, frame in candidates:
        if len(selected) >= limit:
            break
        if frame not in selected and all(abs(frame - existing) >= min_spacing for existing in selected):
            selected.append(frame)
    last = frame_count - 1
    if (
        len(selected) < limit
        and (valid_frames is None or last in valid_frames)
        and last >= min_spacing
        and all(abs(last - item) >= min_spacing for item in selected)
    ):
        selected.append(last)
    return sorted(selected)


async def _cache_route(request: web.Request) -> web.Response:
    try:
        image_entry, mask_entry = _cached_pair(request.query.get("node_id", ""))
        info = dict(image_entry.get("info") or {})
        info["mask_frames"] = int(mask_entry["shape"][0])
        return web.json_response({"info": info, "token": image_entry.get("token", "")})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=404)


async def _frame_route(request: web.Request) -> web.Response:
    try:
        image_entry, mask_entry = _cached_pair(request.query.get("node_id", ""))
        frame = max(0, min(int(request.query.get("frame", 0)), int(mask_entry["shape"][0]) - 1))
        images = np.load(str(image_entry["frames_path"]), mmap_mode="r", allow_pickle=False)
        masks = np.load(str(mask_entry["path"]), mmap_mode="r", allow_pickle=False)
        return web.json_response({"frame": frame, "image": _preview_data_url(np.asarray(images[frame]), np.asarray(masks[frame]))})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def _analysis_route(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
        _, mask_entry = _cached_pair(payload.get("node_id", ""))
        masks = np.load(str(mask_entry["path"]), mmap_mode="r", allow_pickle=False)
        minimum = max(1, int(payload.get("anchor_min_spacing", 48)))
        requested_hysteresis = max(1, int(payload.get("anchor_hysteresis", 3)))
        # Hysteresis is an analysis smoothing radius, not an execution
        # constraint. Keep it in a numerically useful range automatically.
        hysteresis = min(requested_hysteresis, max(1, minimum // 2))
        limit = max(1, int(payload.get("anchor_limit", 12)))
        stride = max(1, int(payload.get("analysis_stride", 1)))
        sensitivity = min(1.0, max(0.0, float(payload.get("anchor_sensitivity", 0.35))))
        threshold = min(1.0, max(0.0, float(payload.get("mask_threshold", 0.5))))
        indices, scores = _analysis_features(masks, stride, threshold)
        binary_area = np.asarray(masks >= int(round(threshold * 255.0))).reshape(len(masks), -1).mean(axis=1)
        valid_frames = {int(frame) for frame in indices if 0.001 <= float(binary_area[frame]) <= 0.98}
        anchors = _select_anchors(indices, scores, len(masks), minimum, hysteresis, limit, sensitivity, valid_frames)
        candidates = [
            {"frame": int(frame), "score": round(float(score), 5)}
            for frame, score in zip(indices, scores)
            if frame in valid_frames and score >= sensitivity * 0.65
        ]
        primary = max(
            ((float(scores[pos]), int(frame)) for pos, frame in enumerate(indices) if frame in valid_frames),
            default=(0.0, int(anchors[0] if anchors else 0)),
        )[1]
        return web.json_response(
            {
                "anchors": anchors,
                "primary_anchor": primary,
                "candidates": candidates,
                "frame_count": int(len(masks)),
                "effective": {
                    "anchor_min_spacing": minimum,
                    "anchor_hysteresis": hysteresis,
                    "anchor_limit": limit,
                },
            }
        )
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)


class CSMatAnyone2(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        devices = ["auto", "cpu"]
        if torch.cuda.is_available():
            devices.extend(f"gpu{index}" for index in range(torch.cuda.device_count()))
        return io.Schema(
            node_id=NODE_ID,
            display_name="CS MatAnyone2",
            category=_CATEGORY,
            essentials_category="Video Tools",
            search_aliases=["matanyone2", "video matte", "adaptive anchor", "alpha matte"],
            description="Create a single-person or union alpha matte from a matched IMAGE and coarse MASK sequence.",
            inputs=[
                io.Image.Input("image", tooltip="Video frames as a ComfyUI IMAGE batch."),
                io.Mask.Input("mask", tooltip="Matched per-frame coarse masks from SAM3 or another segmenter."),
                io.Float.Input("max_megapixels", display_name="Max inference size (MPixels)", default=2.1, min=0.1, max=64.0, step=0.1),
                io.String.Input("anchor_frames", display_name="Anchor frames", default="0", tooltip="JSON list or comma-separated local frame numbers."),
                io.Int.Input("anchor_min_spacing", display_name="Anchor minimum spacing", default=48, min=1, max=100000, step=1),
                io.Int.Input("anchor_hysteresis", display_name="Anchor hysteresis", default=3, min=1, max=1000, step=1),
                io.Int.Input("anchor_limit", display_name="Anchor limit", default=12, min=1, max=128, step=1),
                io.Int.Input("overlap", default=12, min=0, max=10000, step=1),
                io.Int.Input("analysis_stride", display_name="Analysis stride", default=1, min=1, max=120, step=1, advanced=True),
                io.Float.Input("anchor_sensitivity", display_name="Anchor sensitivity", default=0.35, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Float.Input("mask_threshold", display_name="Mask threshold", default=0.5, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Int.Input("seed_morphology", display_name="Seed morphology", default=0, min=-64, max=64, step=1, advanced=True, tooltip="Negative erodes; positive dilates anchor masks."),
                io.Int.Input("warmup", default=10, min=1, max=50, step=1, advanced=True),
                io.Int.Input("memory_interval", display_name="Memory interval", default=5, min=1, max=100, step=1, advanced=True),
                io.Int.Input("memory_frames", display_name="Memory frames", default=5, min=2, max=50, step=1, advanced=True),
                io.Boolean.Input("use_long_term", display_name="Use long-term memory", default=False, advanced=True),
                io.Combo.Input("device", options=devices, default="auto", advanced=True),
                io.Combo.Input("model_file", display_name="Model file", options=_model_options(), default=MODEL_FILENAME, advanced=True),
                io.Boolean.Input("auto_unload_model", display_name="Auto unload model", default=True, advanced=True),
                io.Boolean.Input("wait_for_input_cache", display_name="wait for input cache", default=False, advanced=True),
            ],
            outputs=[
                io.Mask.Output("mask", display_name="MASK"),
                io.Dict.Output("info", display_name="info"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    @torch.inference_mode()
    def execute(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor,
        max_megapixels: float = 2.1,
        anchor_frames: str = "0",
        anchor_min_spacing: int = 48,
        anchor_hysteresis: int = 3,
        anchor_limit: int = 12,
        overlap: int = 12,
        analysis_stride: int = 1,
        anchor_sensitivity: float = 0.35,
        mask_threshold: float = 0.5,
        seed_morphology: int = 0,
        warmup: int = 10,
        memory_interval: int = 5,
        memory_frames: int = 5,
        use_long_term: bool = False,
        device: str = "auto",
        model_file: str = MODEL_FILENAME,
        auto_unload_model: bool = True,
        wait_for_input_cache: bool = False,
    ) -> io.NodeOutput:
        # Automatic candidate selection is intentionally performed by Matte
        # Preview. A queued execution consumes the explicit anchor list; when
        # the preview has never been opened, the default remains frame 0.
        del analysis_stride, anchor_sensitivity
        started = time.perf_counter()
        _matanyone_info("start")
        images = _normalise_images(image)
        frame_count, source_height, source_width = map(int, images.shape[:3])
        masks = _normalise_masks(mask, frame_count, source_height, source_width)
        _matanyone_info(f"input ready: frames={frame_count}, size={source_width}x{source_height}")
        node_id = getattr(getattr(cls, "hidden", None), "unique_id", "")
        prompt = getattr(getattr(cls, "hidden", None), "prompt", None)
        _cache_inputs(node_id, images, masks)
        if wait_for_input_cache:
            _matanyone_info("input cached; waiting for Matte Preview")
            _cache_wait_input(node_id, prompt, images)
            from comfy.model_management import InterruptProcessingException

            raise InterruptProcessingException()

        anchors = _parse_anchors(anchor_frames, frame_count)
        # Spacing, hysteresis and limit guide preview candidate generation;
        # they deliberately do not reject user-authored anchors at execute
        # time.  Only the overlap width needs a runtime safety clamp because
        # it directly controls adjacent propagation windows.
        effective_overlap = _effective_overlap(anchors, overlap)
        infer_height, infer_width = _target_size(source_height, source_width, max_megapixels)
        _matanyone_info(f"preparing inference: size={infer_width}x{infer_height}, anchors={anchors}")
        inference_images = _lanczos_resize_images(images, infer_height, infer_width)
        inference_masks = _resize_masks(masks, infer_height, infer_width)
        target_device = _device(device)
        checkpoint = _model_path(model_file)
        model = _load_model(checkpoint, target_device)
        _matanyone_info(f"model ready: device={target_device}")
        try:
            alpha = _matte(
                model,
                inference_images,
                inference_masks,
                anchors,
                target_device,
                max(1, int(warmup)),
                min(1.0, max(0.0, float(mask_threshold))),
                max(-64, min(64, int(seed_morphology))),
                effective_overlap,
                max(1, int(memory_interval)),
                max(2, int(memory_frames)),
                bool(use_long_term),
            )
            alpha = _resize_masks(alpha, source_height, source_width)
        finally:
            if auto_unload_model:
                _unload_model(checkpoint, target_device, model)
        info = {
            "node": "CS MatAnyone2",
            "status": "complete",
            "frames": frame_count,
            "source_width": source_width,
            "source_height": source_height,
            "inference_width": infer_width,
            "inference_height": infer_height,
            "anchors": anchors,
            "overlap": effective_overlap,
            "device": str(target_device),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        _matanyone_info(f"complete: frames={frame_count}, anchors={anchors}, elapsed={info['elapsed_seconds']:.2f}s")
        return io.NodeOutput(alpha.contiguous(), info)


class MatAnyone2Extension(ComfyExtension):
    @override
    async def on_load(self) -> None:
        global _ROUTES_REGISTERED
        if _ROUTES_REGISTERED:
            return
        from server import PromptServer

        instance = getattr(PromptServer, "instance", None)
        if instance is not None:
            instance.routes.get("/cinestyle/matanyone2-cache")(_cache_route)
            instance.routes.get("/cinestyle/matanyone2-frame")(_frame_route)
            instance.routes.post("/cinestyle/matanyone2-analyze")(_analysis_route)
            _ROUTES_REGISTERED = True

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CSMatAnyone2]


async def comfy_entrypoint() -> MatAnyone2Extension:
    return MatAnyone2Extension()
