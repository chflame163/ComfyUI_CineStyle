"""GPU accelerated single-layer image compositor for CineStyle.

The node deliberately keeps the editable state small: position, size and
rotation are interpolated keyframes, while blend mode and opacity are fixed
for the complete batch.  IMAGE batches are broadcast using the usual ComfyUI
single-frame convention; a warning is emitted whenever broadcasting is used.
"""

from __future__ import annotations

import io as py_io
import json
import logging
import math
import sys
from typing import Any, Mapping

import torch
import torch.nn.functional as F
import numpy as np
from aiohttp import web
from PIL import Image
from typing_extensions import override

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - ComfyUI normally provides tqdm
    tqdm = None

from comfy_api.latest import ComfyExtension, io


_NODE_ID = "CS_Image_Composite"
_CATEGORY = "😺dzNodes/CineStyle"
_LOGGER = logging.getLogger("CineStyleImageComposite")
_CACHE_STORE = None
_ROUTES_REGISTERED = False
_MASK_CACHE: dict[str, torch.Tensor] = {}
_BLEND_MODES = (
    "normal", "dissolve", "darken", "multiply", "color burn", "linear burn",
    "darker color", "lighten", "screen", "color dodge", "linear dodge(add)",
    "lighter color", "dodge", "overlay", "soft light", "hard light", "vivid light",
    "linear light", "pin light", "hard mix", "difference", "exclusion", "subtract",
    "divide", "hue", "saturation", "color", "luminosity", "grain extract", "grain merge",
)
_EPS = 1.0e-6
_GPU_MEMORY_FRACTION = 0.40
_GPU_MEMORY_RESERVE = 512 * 1024 * 1024
_MAX_GPU_BATCH = 32
_CPU_BATCH = 4
_ANSI_GREEN = "\033[32m"
_ANSI_RESET = "\033[0m"


def _composite_info(message: str, *args: Any) -> None:
    """Keep CS Image Composite status lines aligned with CineStyle video nodes."""
    _LOGGER.info("[CS Image Composite] " + message, *args)


class _CompositeProgress:
    """Emit a tqdm-style progress bar for final frame compositing."""

    def __init__(self, total: int, description: str = "compositing frames"):
        self.bar = None
        if tqdm is not None:
            self.bar = tqdm(
                total=max(1, int(total)),
                desc=f"{_ANSI_GREEN}[INFO]{_ANSI_RESET} [CS Image Composite] {description}",
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


def _cache_store():
    global _CACHE_STORE
    if _CACHE_STORE is None:
        package = __name__.rsplit(".", 1)[0]
        module = sys.modules.get(f"{package}._py_preview_cache")
        if module is None:
            raise RuntimeError("CineStyle preview cache module is unavailable.")
        _CACHE_STORE = module.PreviewCacheStore("image_composite")
    return _CACHE_STORE


def _loader_preview_cache():
    """Return the shared CS Load Video preview cache, when available."""
    package = __name__.rsplit(".", 1)[0]
    module = sys.modules.get(f"{package}._py_loader_preview_cache")
    return module.get_loader_preview_cache() if module is not None else None


def _wait_input_cache_store():
    """Return the shared wait-input cache, when the preview package is loaded."""
    package = __name__.rsplit(".", 1)[0]
    module = sys.modules.get(f"{package}._py_preview_cache")
    return module.get_wait_input_cache_store() if module is not None else None


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _integer(value: Any, default: int = 0) -> int:
    try:
        number = int(round(float(value)))
        return number
    except (TypeError, ValueError, OverflowError):
        return default


def _normalise_image(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a ComfyUI IMAGE tensor.")
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.shape[-1] < 3:
        raise ValueError(f"{name} must have shape [batch,height,width,3 or 4].")
    if any(int(size) <= 0 for size in value.shape[:3]):
        raise ValueError(f"{name} must contain non-empty frames.")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} contains non-finite pixel values.")
    return value.float().clamp(0.0, 1.0)


def _normalise_mask(value: Any, name: str, batch: int, height: int, width: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a ComfyUI MASK tensor.")
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim == 4:
        value = value[..., 0]
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [batch,height,width].")
    if int(value.shape[0]) <= 0:
        raise ValueError(f"{name} must contain at least one frame.")
    value = value.float().clamp(0.0, 1.0)
    if value.shape[1:3] != (height, width):
        value = F.interpolate(value.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False)[:, 0]
    return value


def _broadcast_batch(value: torch.Tensor, count: int, label: str, *, warn: bool = True) -> torch.Tensor:
    current = int(value.shape[0])
    if current == count:
        return value
    if warn:
        _LOGGER.warning("[CS Image Composite] %s batch=%d broadcast to background batch=%d", label, current, count)
    # Repeat-and-truncate is the same deterministic convention used by
    # ComfyUI's repeat_to_batch_size helper. It supports a short animated
    # layer as well as the common static (batch=1) layer case.
    repeats = math.ceil(count / max(1, current))
    return value.repeat((repeats, *([1] * (value.ndim - 1))))[:count]


def _timeline_keyframes(value: Any, count: int, canvas_w: int, canvas_h: int, layer_w: int, layer_h: int) -> tuple[dict[str, float], list[dict[str, float]]]:
    # Scale values are relative to a letterbox-fit layer.  This keeps 1.0
    # meaningful regardless of the source and background aspect ratios.
    fit = min(canvas_w / max(1.0, layer_w), canvas_h / max(1.0, layer_h))
    base_width = max(1.0e-4, layer_w * fit / max(1.0, canvas_w))
    base_height = max(1.0e-4, layer_h * fit / max(1.0, canvas_h))
    default = {
        "x": 0.5,
        "y": 0.5,
        "width": base_width,
        "height": base_height,
        "rotation": 0.0,
    }
    try:
        raw = json.loads(value) if isinstance(value, str) and value.strip() else (value or {})
    except (TypeError, ValueError):
        raw = {}
    if not isinstance(raw, Mapping):
        raw = {}
    raw_default = raw.get("default") if isinstance(raw.get("default"), Mapping) else {}
    modern_scale = "scale_x" in raw_default or "scale_y" in raw_default or int(raw.get("version", 0) or 0) >= 2
    for key in default:
        if key in raw_default:
            default[key] = _finite(raw_default[key], default[key])
    if modern_scale:
        default["width"] = max(1.0e-4, _finite(raw_default.get("scale_x"), 1.0) * base_width)
        default["height"] = max(1.0e-4, _finite(raw_default.get("scale_y"), 1.0) * base_height)
    default["width"] = max(1.0e-4, default["width"])
    default["height"] = max(1.0e-4, default["height"])
    frames: list[dict[str, float]] = []
    # Version 2 stores one shared transform record.  Keep parsing v1 records
    # for backwards compatibility, but never let stale keyframes from a v2
    # payload reintroduce frame-dependent transforms.
    raw_frames = raw.get("keyframes") if not modern_scale and isinstance(raw.get("keyframes"), list) else []
    for item in raw_frames:
        if not isinstance(item, Mapping):
            continue
        frame = max(0, min(max(0, count - 1), _integer(item.get("frame"), 0)))
        keyframe = {key: _finite(item.get(key), default[key]) for key in default}
        if modern_scale:
            keyframe["width"] = _finite(item.get("scale_x"), 1.0) * base_width
            keyframe["height"] = _finite(item.get("scale_y"), 1.0) * base_height
        keyframe["frame"] = float(frame)
        keyframe["width"] = max(1.0e-4, keyframe["width"])
        keyframe["height"] = max(1.0e-4, keyframe["height"])
        frames.append(keyframe)
    frames.sort(key=lambda item: int(item["frame"]))
    return default, frames


def _interpolated_transform(default: Mapping[str, float], keyframes: list[dict[str, float]], frame: int) -> dict[str, float]:
    if not keyframes:
        return dict(default)
    if frame <= keyframes[0]["frame"]:
        result = dict(keyframes[0])
        result.pop("frame", None)
        return result
    if frame >= keyframes[-1]["frame"]:
        result = dict(keyframes[-1])
        result.pop("frame", None)
        return result
    left, right = keyframes[0], keyframes[-1]
    for index in range(1, len(keyframes)):
        if frame <= keyframes[index]["frame"]:
            left, right = keyframes[index - 1], keyframes[index]
            break
    span = max(1.0, right["frame"] - left["frame"])
    amount = (frame - left["frame"]) / span
    # Smoothstep keeps the position, size and rotation continuous at a
    # keyframe instead of changing velocity abruptly when the next keyframe
    # becomes active.
    amount = amount * amount * (3.0 - 2.0 * amount)
    result = {}
    for key in ("x", "y", "width", "height"):
        result[key] = left[key] + (right[key] - left[key]) * amount
    # Interpolate along the shortest angular path.
    delta = (right["rotation"] - left["rotation"] + 180.0) % 360.0 - 180.0
    result["rotation"] = left["rotation"] + delta * amount
    return result


def _device(source: torch.Tensor) -> torch.device:
    if torch.cuda.is_available():
        try:
            import comfy.model_management as model_management

            candidate = model_management.get_torch_device()
            return candidate if isinstance(candidate, torch.device) else torch.device(candidate)
        except Exception:
            return torch.device(f"cuda:{torch.cuda.current_device()}")
    return source.device if isinstance(source, torch.Tensor) else torch.device("cpu")


def _batch_size(background: torch.Tensor, device: torch.device) -> int:
    total = int(background.shape[0])
    if total <= 1:
        return 1
    if device.type != "cuda":
        return min(total, _CPU_BATCH)
    pixels = max(1, int(background.shape[1]) * int(background.shape[2]))
    estimated = pixels * 4 * 4 * 12
    try:
        free, _ = torch.cuda.mem_get_info(device)
        budget = max(estimated, int(max(0, free - _GPU_MEMORY_RESERVE) * _GPU_MEMORY_FRACTION))
        return max(1, min(total, _MAX_GPU_BATCH, budget // estimated))
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return 1


def _hsv(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum, index = rgb.max(dim=-1)
    minimum = rgb.min(dim=-1).values
    delta = maximum - minimum
    safe = delta.clamp_min(_EPS)
    r, g, b = rgb.unbind(-1)
    h = torch.zeros_like(maximum)
    h = torch.where(index == 0, ((g - b) / safe) % 6.0, h)
    h = torch.where(index == 1, (b - r) / safe + 2.0, h)
    h = torch.where(index == 2, (r - g) / safe + 4.0, h)
    h = torch.where(delta > _EPS, h / 6.0, torch.zeros_like(h))
    s = delta / maximum.clamp_min(_EPS)
    s = torch.where(maximum > _EPS, s, torch.zeros_like(s))
    return h % 1.0, s, maximum


def _hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    h6 = (h % 1.0) * 6.0
    i = torch.floor(h6).long() % 6
    f = h6 - torch.floor(h6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    choices = (
        torch.stack((v, t, p), -1), torch.stack((q, v, p), -1),
        torch.stack((p, v, t), -1), torch.stack((p, q, v), -1),
        torch.stack((t, p, v), -1), torch.stack((v, p, q), -1),
    )
    result = choices[0]
    for index in range(1, 6):
        result = torch.where((i == index).unsqueeze(-1), choices[index], result)
    return result


def _blend_rgb(mode: str, backdrop: torch.Tensor, layer: torch.Tensor, opacity: float) -> torch.Tensor:
    mode = str(mode or "normal")
    amount = max(0.0, min(1.0, float(opacity)))
    if mode == "dissolve":
        # V2's dissolve is stochastic.  A coordinate-derived noise field keeps
        # preview and final execution stable without introducing a new seed
        # widget, while retaining the same coverage interpretation.
        height, width = backdrop.shape[-3], backdrop.shape[-2]
        yy, xx = torch.meshgrid(torch.arange(height, device=backdrop.device), torch.arange(width, device=backdrop.device), indexing="ij")
        noise = torch.remainder(torch.sin(xx.float() * 12.9898 + yy.float() * 78.233) * 43758.5453, 1.0).unsqueeze(-1)
        return torch.where(noise < amount, layer, backdrop)
    if mode == "normal":
        return torch.lerp(backdrop, layer, amount)
    if mode == "multiply":
        formula = backdrop * layer
    elif mode == "darken":
        formula = torch.minimum(backdrop, layer)
    elif mode == "lighten":
        formula = torch.maximum(backdrop, layer)
    elif mode == "darker color" or mode == "lighter color":
        _, _, value_backdrop = _hsv(backdrop)
        _, _, value_layer = _hsv(layer)
        choose_layer = value_layer < value_backdrop if mode == "darker color" else value_layer > value_backdrop
        formula = torch.where(choose_layer.unsqueeze(-1), layer, backdrop)
    elif mode == "dodge":
        formula = (backdrop / (1 - layer).clamp_min(_EPS)).clamp(0.0, 1.0)
    elif mode == "screen":
        formula = 1 - (1 - backdrop) * (1 - layer)
    elif mode == "overlay":
        formula = torch.where(backdrop < 0.5, 2 * backdrop * layer, 1 - 2 * (1 - backdrop) * (1 - layer))
    elif mode == "hard light":
        formula = torch.where(layer < 0.5, 2 * backdrop * layer, 1 - 2 * (1 - backdrop) * (1 - layer))
    elif mode == "soft light":
        formula = (1 - backdrop) * backdrop * layer + backdrop * (1 - (1 - backdrop) * (1 - layer))
    elif mode == "difference":
        formula = (backdrop - layer).abs()
    elif mode == "exclusion":
        formula = backdrop + layer - 2 * backdrop * layer
    elif mode == "linear dodge(add)":
        formula = backdrop + layer
    elif mode == "linear burn":
        formula = backdrop + layer - 1
    elif mode == "subtract":
        formula = backdrop - layer
    elif mode == "divide":
        formula = backdrop / layer.clamp_min(_EPS)
    elif mode == "color dodge":
        formula = backdrop / (1 - layer).clamp_min(_EPS)
    elif mode == "color burn":
        formula = 1 - (1 - backdrop) / layer.clamp_min(_EPS)
    elif mode == "linear light":
        formula = backdrop + 2 * layer - 1
    elif mode == "vivid light":
        formula = torch.where(layer <= 0.5, backdrop / (1 - 2 * layer).clamp_min(_EPS), 1 - (1 - backdrop) / (2 * layer - 0.5).clamp_min(_EPS))
    elif mode == "pin light":
        formula = torch.where(layer <= 0.5, torch.minimum(backdrop, 2 * layer), torch.maximum(backdrop, 2 * (layer - 0.5)))
    elif mode == "hard mix":
        formula = (backdrop + layer >= 1.0).to(backdrop.dtype)
    elif mode == "grain extract":
        formula = backdrop - layer + 0.5
    elif mode == "grain merge":
        formula = backdrop + layer - 0.5
    elif mode in {"hue", "saturation", "color", "luminosity"}:
        h_i, s_i, v_i = _hsv(backdrop)
        h_l, s_l, v_l = _hsv(layer)
        if mode == "hue":
            formula = _hsv_to_rgb(torch.lerp(h_i, h_l, amount), s_i, v_i)
        elif mode == "saturation":
            formula = _hsv_to_rgb(h_i, torch.lerp(s_i, s_l, amount), v_i)
        elif mode == "color":
            formula = _hsv_to_rgb(torch.lerp(h_i, h_l, amount), torch.lerp(s_i, s_l, amount), v_i)
        else:
            formula = _hsv_to_rgb(h_i, s_i, torch.lerp(v_i, v_l, amount))
        return formula.clamp(0.0, 1.0)
    else:
        formula = layer
    return torch.lerp(backdrop, formula, amount).clamp(0.0, 1.0)


def _transform_grid(
    count: int,
    canvas_h: int,
    canvas_w: int,
    layer_h: int,
    layer_w: int,
    transforms: list[dict[str, float]],
    device: torch.device,
) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(canvas_h, device=device, dtype=torch.float32),
        torch.arange(canvas_w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(xx)
    coords = torch.stack((xx, yy, ones), dim=0).reshape(3, -1)
    matrices = []
    for item in transforms:
        tw = max(1.0, item["width"] * canvas_w)
        th = max(1.0, item["height"] * canvas_h)
        cx, cy = item["x"] * canvas_w, item["y"] * canvas_h
        angle = math.radians(item["rotation"])
        c, s = math.cos(angle), math.sin(angle)
        sx, sy = tw / max(1, layer_w), th / max(1, layer_h)
        src_cx, src_cy = (layer_w - 1) * 0.5, (layer_h - 1) * 0.5
        forward = torch.tensor(
            [[c * sx, -s * sy, cx - c * sx * src_cx + s * sy * src_cy],
             [s * sx, c * sy, cy - s * sx * src_cx - c * sy * src_cy],
             [0.0, 0.0, 1.0]], device=device, dtype=torch.float32,
        )
        matrices.append(torch.linalg.inv(forward))
    inverse = torch.stack(matrices)
    source = inverse @ coords.unsqueeze(0).expand(count, -1, -1)
    src_x, src_y = source[:, 0].reshape(count, canvas_h, canvas_w), source[:, 1].reshape(count, canvas_h, canvas_w)
    return torch.stack(((src_x + 0.5) * 2.0 / layer_w - 1.0, (src_y + 0.5) * 2.0 / layer_h - 1.0), dim=-1)


@torch.inference_mode()
def _composite_batch(
    background: torch.Tensor,
    layer: torch.Tensor,
    layer_mask: torch.Tensor | None,
    blend_mode: str,
    opacity: float,
    timeline_json: Any,
    *,
    output_device: torch.device | str | None = None,
    warn_broadcast: bool = True,
    progress: _CompositeProgress | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    background = _normalise_image(background, "background_image")
    layer = _normalise_image(layer, "layer_image")
    count, canvas_h, canvas_w = int(background.shape[0]), int(background.shape[1]), int(background.shape[2])
    layer_count, layer_h, layer_w = int(layer.shape[0]), int(layer.shape[1]), int(layer.shape[2])
    layer = _broadcast_batch(layer, count, "layer_image", warn=warn_broadcast)
    if layer_mask is not None:
        # A static layer may be paired with a frame-aligned mask, so validate
        # against the resolved output batch rather than the original layer
        # batch before broadcasting.  Layer masks use the compositor
        # convention: white (1) is visible and black (0) is hidden.
        mask = _normalise_mask(layer_mask, "layer_mask", max(count, layer_count), layer_h, layer_w)
        mask = _broadcast_batch(mask, count, "layer_mask", warn=warn_broadcast)
    else:
        mask = None
    device = _device(background)
    output_device = torch.device(output_device) if output_device is not None else background.device
    result_store = torch.empty((count, canvas_h, canvas_w, 3), device="cpu", dtype=torch.float32)
    mask_store = torch.empty((count, canvas_h, canvas_w), device="cpu", dtype=torch.float32)
    default, keyframes = _timeline_keyframes(timeline_json, count, canvas_w, canvas_h, layer_w, layer_h)
    batch = _batch_size(background, device)
    start = 0
    while start < count:
        end = min(count, start + batch)
        try:
            indices = list(range(start, end))
            bg = background[start:end, ..., :3].to(device=device, dtype=torch.float32, non_blocking=True)
            alpha_bg = background[start:end, ..., 3].to(device=device, dtype=torch.float32, non_blocking=True) if background.shape[-1] >= 4 else torch.ones((end - start, canvas_h, canvas_w), device=device)
            lay = layer[start:end, ..., :3].to(device=device, dtype=torch.float32, non_blocking=True)
            alpha_layer = layer[start:end, ..., 3].to(device=device, dtype=torch.float32, non_blocking=True) if layer.shape[-1] >= 4 else torch.ones((end - start, layer_h, layer_w), device=device)
            if mask is not None:
                alpha_layer = alpha_layer * mask[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
            transforms = [_interpolated_transform(default, keyframes, index) for index in indices]
            grid = _transform_grid(end - start, canvas_h, canvas_w, layer_h, layer_w, transforms, device)
            lay_canvas = F.grid_sample(lay.movedim(-1, 1), grid, mode="bilinear", padding_mode="zeros", align_corners=False).movedim(1, -1)
            alpha_canvas = F.grid_sample(alpha_layer.unsqueeze(1), grid, mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0]
            # Opacity is part of the blend operation, as in LayerStyle V2;
            # keep the geometric/alpha coverage separate for the output mask.
            source_alpha = alpha_canvas.clamp(0.0, 1.0)
            blended = _blend_rgb(blend_mode, bg, lay_canvas, opacity).clamp(0.0, 1.0)
            out_alpha = source_alpha + alpha_bg * (1.0 - source_alpha)
            out_premult = blended * source_alpha.unsqueeze(-1) + bg * alpha_bg.unsqueeze(-1) * (1.0 - source_alpha.unsqueeze(-1))
            out = torch.where(out_alpha.unsqueeze(-1) > _EPS, out_premult / out_alpha.unsqueeze(-1).clamp_min(_EPS), torch.zeros_like(out_premult)).clamp(0.0, 1.0)
            result_store[start:end].copy_(out.detach().to("cpu", dtype=torch.float32))
            # ComfyUI's standard MASK convention is 1 = transparent.  The
            # output therefore describes transparency of the final composite;
            # an opaque background correctly yields an all-zero mask.
            mask_store[start:end].copy_((1.0 - out_alpha).clamp(0.0, 1.0).detach().to("cpu", dtype=torch.float32))
            if progress is not None:
                progress.update(end - start)
            del bg, lay, lay_canvas, alpha_canvas, grid, out
            start = end
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or batch <= 1:
                raise
            batch = max(1, batch // 2)
            _LOGGER.warning("[CS Image Composite] CUDA OOM; retrying with batch=%d", batch)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return result_store.to(output_device), mask_store.to(output_device)


def _png_bytes(frame: torch.Tensor) -> bytes:
    array = frame.detach().to("cpu", dtype=torch.float32).clamp(0.0, 1.0).mul(255).round().to(torch.uint8).numpy()
    buffer = py_io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def _entry_for_token(token: str) -> dict[str, Any] | None:
    value = str(token or "").strip()
    if value.startswith("loader_preview:"):
        cache = _loader_preview_cache()
        return cache.entry_for_token(value) if cache is not None else None
    if value.startswith("wait_input:"):
        cache = _wait_input_cache_store()
        return cache.get_token(value) if cache is not None else None
    return _cache_store().get_token(value)


def _decode_token(token: str, frame: int) -> torch.Tensor:
    value = str(token or "").strip()
    entry = _entry_for_token(value)
    if entry is None:
        raise ValueError("Preview cache token is unavailable.")
    info = dict(entry.get("info") or {})
    count = max(1, _integer(info.get("frames"), 1))
    target = 0 if count <= 1 else max(0, frame) % count
    if value.startswith("loader_preview:"):
        cache = _loader_preview_cache()
        if cache is None:
            raise ValueError("The shared loader preview cache is unavailable.")
        return cache.decode_frame(value, target)
    if value.startswith("wait_input:"):
        cache = _wait_input_cache_store()
        if cache is None:
            raise ValueError("The shared wait input preview cache is unavailable.")
        return cache.decode_frame({"source_token": value}, target)
    return _cache_store().decode_frame({"source_token": value}, target)


def _decode_preview_input(payload: Mapping[str, Any], prefix: str, frame: int) -> torch.Tensor:
    token = str(payload.get(f"{prefix}_token") or "").strip()
    if token:
        return _decode_token(token, frame)
    filename = str(payload.get(f"{prefix}_file") or "").strip()
    if not filename:
        raise ValueError(f"{prefix}_token or {prefix}_file is required.")
    kind = str(payload.get(f"{prefix}_kind") or "image").strip().lower()
    if kind == "image" and prefix in {"background", "layer"}:
        path = _cache_store()._resolve_file(filename)
        with Image.open(path) as image:
            rgba = np.asarray(image.convert("RGBA"), dtype="float32") / 255.0
        return torch.from_numpy(np.ascontiguousarray(rgba)).unsqueeze(0)
    if prefix == "mask":
        path = _cache_store()._resolve_file(filename)
        if kind != "image":
            raise ValueError("Direct layer_mask preview currently supports image files only.")
        channel = str(payload.get("mask_channel") or payload.get("layer_channel") or "alpha").strip().lower()
        with Image.open(path) as image:
            rgba = image.convert("RGBA")
            if channel in {"red", "r"}:
                selected = rgba.getchannel("R")
            elif channel in {"green", "g"}:
                selected = rgba.getchannel("G")
            elif channel in {"blue", "b"}:
                selected = rgba.getchannel("B")
            else:
                selected = rgba.getchannel("A")
            tensor = torch.from_numpy(np.asarray(selected, dtype="float32")).div_(255.0)
        return tensor.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, 3)
    return _cache_store().decode_frame({"video": filename, "source_kind": kind}, max(0, frame))


def _input_chain(prompt: Any, node_id: str, input_name: str) -> dict[str, Any] | None:
    package = __name__.rsplit(".", 1)[0]
    module = sys.modules.get(f"{package}._py_preview_cache")
    if module is None:
        return None
    return module.build_input_chain(prompt, node_id, (input_name,))


def _connected_mask_channel(prompt: Any, node_id: str) -> str:
    """Return the channel selected by a directly connected LoadImageMask."""
    if not isinstance(prompt, Mapping):
        return ""
    node = prompt.get(str(node_id)) or prompt.get(node_id)
    inputs = node.get("inputs") if isinstance(node, Mapping) and isinstance(node.get("inputs"), Mapping) else {}
    link = inputs.get("layer_mask")
    if not isinstance(link, (list, tuple)) or not link:
        return ""
    upstream = prompt.get(str(link[0])) or prompt.get(link[0])
    if not isinstance(upstream, Mapping):
        return ""
    class_type = str(upstream.get("class_type") or "").lower()
    if "loadimagemask" not in class_type and "image_as_mask" not in class_type and "load_image_mask" not in class_type:
        return ""
    values = upstream.get("inputs") if isinstance(upstream.get("inputs"), Mapping) else {}
    return str(values.get("channel") or "").strip().lower()


def _cache_wait_input(node_id: str, prompt: Any, input_name: str, frames: torch.Tensor) -> None:
    package = __name__.rsplit(".", 1)[0]
    module = sys.modules.get(f"{package}._py_preview_cache")
    chain = _input_chain(prompt, node_id, input_name)
    if module is None or chain is None:
        return
    value = frames
    if input_name == "layer_mask":
        value = _normalise_mask(value, input_name, int(value.shape[0]), int(value.shape[-2]), int(value.shape[-1])).unsqueeze(-1).expand(-1, -1, -1, 3)
    else:
        value = _normalise_image(value, input_name)
    try:
        module.get_wait_input_cache_store().put_chain(
            chain,
            value[..., :3],
            24.0,
            info={
                "producer_node_id": node_id,
                "producer_node_type": _NODE_ID,
                "producer_input_name": input_name,
            },
            force=True,
        )
    except Exception as exc:
        _LOGGER.warning("[CS Image Composite] wait input cache failed for %s: %s", input_name, exc)


async def _cache_info_route(request: web.Request) -> web.Response:
    node_id = str(request.query.get("node_id") or "").strip()
    if not node_id:
        return web.json_response({"error": "node_id is required."}, status=400)
    store = _cache_store()
    response: dict[str, Any] = {}
    for name in ("background", "layer", "mask"):
        entry = store.get_preview_variant(f"{node_id}:{name}", proxy=False)
        if entry is not None:
            info = dict(entry.get("info") or {})
            response[name] = {"token": str(entry.get("token") or ""), "info": info, "kind": "image" if int(info.get("frames") or 1) == 1 else "video"}
    if not response.get("background") or not response.get("layer"):
        return web.json_response({"error": "Run CS Image Composite once to cache its inputs."}, status=404)
    return web.json_response(response)


async def _preview_route(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
        if not isinstance(payload, Mapping):
            raise ValueError("Preview payload must be an object.")
        frame = max(0, _integer(payload.get("frame"), 0))
        bg_token = str(payload.get("background_token") or "")
        layer_token = str(payload.get("layer_token") or "")
        mask_token = str(payload.get("mask_token") or "")
        background = _decode_preview_input(payload, "background", frame)
        layer = _decode_preview_input(payload, "layer", frame)
        mask = _decode_preview_input(payload, "mask", frame)[..., 0] if mask_token or payload.get("mask_file") else None
        # The preview decodes one frame at a time, so evaluate the requested
        # timeline frame before handing a one-frame descriptor to the renderer.
        timeline_value = {
            "version": 2,
            "sync": bool(payload.get("sync_scale", True)),
            "default": {
                "x": _finite(payload.get("x"), 0.5),
                "y": _finite(payload.get("y"), 0.5),
                "scale_x": max(0.0001, _finite(payload.get("scale_x"), 1.0)),
                "scale_y": max(0.0001, _finite(payload.get("scale_y"), 1.0)),
                "rotation": _finite(payload.get("rotation"), 0.0),
            },
        }
        background_entry = _entry_for_token(bg_token) or {}
        background_count = max(
            1,
            _integer(
                payload.get("background_frames"),
                _integer((background_entry.get("info") or {}).get("frames"), frame + 1),
            ),
        )
        default, keyframes = _timeline_keyframes(
            timeline_value,
            background_count,
            int(background.shape[2]),
            int(background.shape[1]),
            int(layer.shape[2]),
            int(layer.shape[1]),
        )
        preview_transform = _interpolated_transform(default, keyframes, frame)
        preview_timeline = json.dumps({"default": preview_transform}, separators=(",", ":"))
        output, _ = _composite_batch(background, layer, mask, str(payload.get("blend_mode") or "normal"), _finite(payload.get("opacity"), 100.0) / 100.0, preview_timeline, output_device="cpu", warn_broadcast=False)
        return web.Response(body=_png_bytes(output[0]), content_type="image/png", headers={"Cache-Control": "no-store"})
    except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def _frame_route(request: web.Request) -> web.Response:
    """Return one cached source frame for the browser's local transform preview."""
    try:
        token = str(request.query.get("token") or "").strip()
        filename = str(request.query.get("file") or "").strip()
        frame = max(0, _integer(request.query.get("frame"), 0))
        if token:
            source = _decode_token(token, frame)
        elif filename:
            source = _cache_store().decode_frame({"video": filename, "source_kind": str(request.query.get("kind") or "image")}, frame)
        else:
            raise ValueError("token or file is required.")
        return web.Response(body=_png_bytes(source[0]), content_type="image/png", headers={"Cache-Control": "no-store"})
    except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)


class CSImageComposite(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=_NODE_ID,
            display_name="CS Image Composite",
            category=_CATEGORY,
            essentials_category="Image Effects",
            search_aliases=["image composite", "layer composite", "blend layer", "overlay image"],
            description="Composite one IMAGE layer over a background IMAGE batch with GPU-accelerated transforms and LayerStyle blend modes.",
            inputs=[
                io.Image.Input("background_image", tooltip="Background IMAGE batch and output canvas."),
                io.Image.Input("layer_image", tooltip="Layer IMAGE; a single frame broadcasts to the background batch."),
                io.Mask.Input("layer_mask", optional=True, tooltip="Layer coverage mask: white (1) is visible and black (0) is hidden."),
                io.Float.Input("x", display_name="X", default=0.5, min=-10.0, max=10.0, step=0.001, tooltip="Layer center X, normalized to the background canvas."),
                io.Float.Input("y", display_name="Y", default=0.5, min=-10.0, max=10.0, step=0.001, tooltip="Layer center Y, normalized to the background canvas."),
                io.Float.Input("scale_x", display_name="Scale X", default=1.0, min=0.0001, max=10.0, step=0.001, tooltip="Layer horizontal scale relative to letterbox-fit size."),
                io.Float.Input("scale_y", display_name="Scale Y", default=1.0, min=0.0001, max=10.0, step=0.001, tooltip="Layer vertical scale relative to letterbox-fit size."),
                io.Boolean.Input("sync_scale", display_name="Sync Scale", default=True, tooltip="Keep Scale X and Scale Y linked."),
                io.Float.Input("rotation", display_name="Rotation", default=0.0, min=-360.0, max=360.0, step=0.1, tooltip="Layer rotation in degrees."),
                io.Int.Input("opacity", default=100, min=0, max=100, step=1, tooltip="Layer opacity, fixed for the complete batch."),
                io.Combo.Input("blend_mode", options=list(_BLEND_MODES), default="normal", tooltip="LayerStyle ImageBlendAdvance V2 blend mode; fixed for the complete batch."),
                io.Boolean.Input("wait_for_input_cache", default=False, advanced=True, tooltip="Cache both image inputs and interrupt execution so Edit Timeline can inspect generated tensors."),
            ],
            outputs=[
                io.Image.Output("image", display_name="image", tooltip="Composited RGB image."),
                io.Mask.Output("composit_mask", display_name="composit_mask", tooltip="Standard transparent MASK: 1 means transparent. With an opaque background this is normally all zeros."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    @torch.no_grad()
    def execute(
        cls,
        background_image: torch.Tensor,
        layer_image: torch.Tensor,
        layer_mask: torch.Tensor | None = None,
        x: float = 0.5,
        y: float = 0.5,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        sync_scale: bool = True,
        rotation: float = 0.0,
        opacity: int = 100,
        blend_mode: str = "normal",
        wait_for_input_cache: bool = False,
    ) -> io.NodeOutput:
        _composite_info("start: blend_mode=%s opacity=%d%%", blend_mode, _integer(opacity, 100))
        if blend_mode not in _BLEND_MODES:
            raise ValueError(f"blend_mode must be one of: {', '.join(_BLEND_MODES)}")
        opacity = max(0, min(100, _integer(opacity, 100)))
        transform_json = {
            "version": 2,
            "sync": bool(sync_scale),
            "default": {
                "x": _finite(x, 0.5),
                "y": _finite(y, 0.5),
                "scale_x": max(0.0001, _finite(scale_x, 1.0)),
                "scale_y": max(0.0001, _finite(scale_x if sync_scale else scale_y, 1.0)),
                "rotation": _finite(rotation, 0.0),
            },
        }
        _composite_info("stage 1/4: normalizing IMAGE and MASK inputs")
        background_value = _normalise_image(background_image, "background_image")
        layer_value = _normalise_image(layer_image, "layer_image")
        _composite_info(
            "inputs ready: background=%d frames %dx%d, layer=%d frames %dx%d%s",
            int(background_value.shape[0]),
            int(background_value.shape[2]),
            int(background_value.shape[1]),
            int(layer_value.shape[0]),
            int(layer_value.shape[2]),
            int(layer_value.shape[1]),
            ", mask connected" if layer_mask is not None else ", no mask",
        )
        node_id = str(getattr(getattr(cls, "hidden", None), "unique_id", "") or "").strip()
        prompt = getattr(getattr(cls, "hidden", None), "prompt", None)
        effective_layer_mask = layer_mask
        # ComfyUI's LoadImageMask returns 1 - alpha for its "alpha" option.
        # The composite node intentionally exposes the more intuitive
        # coverage convention (white visible, black hidden), so undo that
        # one producer-specific inversion while leaving RGB channel masks
        # untouched.
        if layer_mask is not None and _connected_mask_channel(prompt, node_id) == "alpha":
            effective_layer_mask = 1.0 - layer_mask
        if node_id:
            _composite_info("stage 2/4: updating preview cache")
            store = _cache_store()
            try:
                store.put_preview(f"{node_id}:background", background_value[..., :3], 24.0, encode_video=False, info={"input_name": "background_image"}, force=True)
                store.put_preview(f"{node_id}:layer", layer_value[..., :3], 24.0, encode_video=False, info={"input_name": "layer_image"}, force=True)
                normalised = None
                if effective_layer_mask is not None:
                    normalised = _normalise_mask(effective_layer_mask, "layer_mask", max(int(background_value.shape[0]), int(layer_value.shape[0])), int(layer_value.shape[1]), int(layer_value.shape[2]))
                # PreviewCacheStore intentionally stores RGB frames.  Preserve
                # a fourth-channel IMAGE alpha (and combine it with an
                # explicit coverage mask) in the dedicated mask cache so the
                # browser preview matches execution.
                alpha_coverage = layer_value[..., 3] if layer_value.shape[-1] >= 4 else None
                if normalised is not None or alpha_coverage is not None:
                    effective_coverage = normalised if normalised is not None else alpha_coverage
                    if normalised is not None and alpha_coverage is not None:
                        effective_coverage = normalised * alpha_coverage
                    _MASK_CACHE[node_id] = effective_coverage.detach().to("cpu", dtype=torch.float32).contiguous()
                    mask_rgb = effective_coverage.unsqueeze(-1).expand(-1, -1, -1, 3)
                    store.put_preview(f"{node_id}:mask", mask_rgb, 24.0, encode_video=False, info={"input_name": "layer_mask"}, force=True)
                else:
                    _MASK_CACHE.pop(node_id, None)
                    store.remove(f"{node_id}:mask", "main")
            except Exception as exc:
                _LOGGER.warning("[CS Image Composite] preview cache unavailable: %s", exc)
        if wait_for_input_cache:
            _composite_info("stage 3/4: storing input cache and interrupting for timeline preview")
            _cache_wait_input(node_id, prompt, "background_image", background_image)
            _cache_wait_input(node_id, prompt, "layer_image", layer_image)
            if effective_layer_mask is not None:
                _cache_wait_input(node_id, prompt, "layer_mask", effective_layer_mask)
            from comfy.model_management import InterruptProcessingException

            raise InterruptProcessingException()
        _composite_info("stage 3/4: compositing frames on GPU")
        progress = _CompositeProgress(int(background_value.shape[0]))
        try:
            image, composit_mask = _composite_batch(
                background_value,
                layer_value,
                effective_layer_mask,
                blend_mode,
                opacity / 100.0,
                transform_json,
                output_device="cpu",
                progress=progress,
            )
        finally:
            progress.close()
        _composite_info("stage 4/4: complete, output frames=%d", int(image.shape[0]))
        return io.NodeOutput(image, composit_mask)


class ImageCompositeExtension(ComfyExtension):
    @override
    async def on_load(self) -> None:
        global _ROUTES_REGISTERED
        if _ROUTES_REGISTERED:
            return
        try:
            from server import PromptServer

            server = getattr(PromptServer, "instance", None)
        except Exception:
            server = None
        if server is not None:
            server.routes.get("/cinestyle/image-composite-cache")(_cache_info_route)
            server.routes.post("/cinestyle/image-composite-preview")(_preview_route)
            server.routes.get("/cinestyle/image-composite-frame")(_frame_route)
            _ROUTES_REGISTERED = True

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CSImageComposite]


async def comfy_entrypoint() -> ImageCompositeExtension:
    return ImageCompositeExtension()


NODE_CLASS_MAPPINGS = {_NODE_ID: CSImageComposite}
NODE_DISPLAY_NAME_MAPPINGS = {_NODE_ID: "CS Image Composite"}
