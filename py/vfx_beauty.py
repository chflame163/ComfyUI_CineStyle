
from __future__ import annotations

import math
import gc
import importlib.util
import sys
import base64
import io as py_io
import logging
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from aiohttp import web
from PIL import Image
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


# 1920x1080 is a typical source size for this node.  The expensive blur passes
# run at this long-side limit; the final grain and colour pass run at source
# resolution so the input detail is not permanently softened.  Keeping this
# below 1024 is important for video batches: all expensive kernels scale with
# proxy pixel count, not with the final output size.
_PROXY_LONG_SIDE = 640
_COLOUR_LONG_SIDE = 512
_COLOUR_SAMPLE_FRAMES = 16
_BISE_NET_SKIN_CLASS = 1
_AUTO_COLOUR_FALLBACK = (0.5294118, 0.3803922, 0.3294118)
_EPS = 1.0e-6
_VFX_MASK_CACHE: dict[str, torch.Tensor | None] = {}
_VFX_COLOUR_CACHE: dict[str, torch.Tensor] = {}
_VFX_PROXY_PRESENT: dict[str, bool] = {}
_VFX_CACHE_LOCK = threading.RLock()
_VFX_ROUTE_REGISTERED = False
_BISE_NET_DOWNLOAD_LOCK = threading.Lock()
_BISE_NET_HF_URL = "https://huggingface.co/jellyhe/parsing_bisenet.pth/resolve/main/parsing_bisenet.pth"
_BISE_NET_OFFICIAL_URL = "https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth"
_LOGGER = logging.getLogger("CineStyleVFXBeauty")
_PREVIEW_CACHE_STORE = None
_BEAUTY_GPU_MEMORY_FRACTION = 0.45
_BEAUTY_MEMORY_RESERVE_BYTES = 512 * 1024 * 1024
_BEAUTY_MAX_GPU_BATCH = 32
_BEAUTY_CPU_BATCH = 8


def _preview_cache_store():
    global _PREVIEW_CACHE_STORE
    if _PREVIEW_CACHE_STORE is None:
        package = __name__.rsplit(".", 1)[0]
        module = sys.modules.get(f"{package}._py_preview_cache")
        if module is None:
            raise RuntimeError("CineStyle preview cache module is unavailable.")
        _PREVIEW_CACHE_STORE = module.PreviewCacheStore("vfx_beauty")
    return _PREVIEW_CACHE_STORE


def _loader_preview_cache():
    package = __name__.rsplit(".", 1)[0]
    module = sys.modules.get(f"{package}._py_loader_preview_cache")
    return module.get_loader_preview_cache() if module is not None else None


def _loader_id_from_video(video: Any) -> str:
    if video is None or not hasattr(video, "get_components"):
        return ""
    try:
        components = video.get_components()
        metadata = dict(getattr(video, "_cinestyle_runtime_metadata", None) or {})
        metadata.update(dict(getattr(components, "metadata", None) or {}))
        return str(metadata.get("loader_id") or "").strip()
    except (AttributeError, TypeError, ValueError):
        return ""


def _loader_id_from_prompt(prompt: Any, node_id: Any) -> str:
    """Find CS Load Video when VFX receives only the loader's IMAGE output."""
    if not isinstance(prompt, dict):
        return ""
    node = prompt.get(str(node_id)) or prompt.get(node_id)
    if not isinstance(node, dict):
        return ""
    inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
    pending: list[Any] = [value for name, value in inputs.items() if "video" in str(name).lower() or "image" in str(name).lower()]
    visited: set[str] = set()
    while pending:
        link = pending.pop(0)
        if not isinstance(link, (list, tuple)) or len(link) < 2:
            continue
        upstream_id = str(link[0])
        if upstream_id in visited:
            continue
        visited.add(upstream_id)
        upstream = prompt.get(upstream_id) or prompt.get(link[0])
        if not isinstance(upstream, dict):
            continue
        class_type = str(upstream.get("class_type") or "")
        if class_type == "CS_Load_Video" or class_type.endswith(".CS_Load_Video") or class_type.endswith("::CS_Load_Video"):
            return upstream_id
        upstream_inputs = upstream.get("inputs") if isinstance(upstream.get("inputs"), dict) else {}
        pending.extend(value for value in upstream_inputs.values() if isinstance(value, (list, tuple)) and len(value) >= 2)
    return ""


def _console_info(node_id: Any, stage: str, detail: str = "") -> None:
    suffix = f" | {detail}" if detail else ""
    node_detail = f"node={node_id} | " if str(node_id or "").strip() else ""
    _LOGGER.info("[CS VFX Beauty] %s%s%s", node_detail, stage, suffix)


class _BeautyProgress:
    """Console stage logger for the bounded Beauty batch pipeline."""

    def __init__(self, node_id: Any, frame_total: int):
        self.node_id = node_id
        self.frame_total = max(1, int(frame_total))
    def info(self, stage: str, detail: str = "") -> None:
        _console_info(self.node_id, stage, detail)

    def close_line(self) -> None:
        return None

    def stage(self, percent: float, stage: str, detail: str = "") -> None:
        self.info(stage, detail)


def _parse_hex_colour(value: Any, name: str = "colour") -> torch.Tensor | None:
    """Parse ``auto`` or a strict six-digit ``#RRGGBB`` colour."""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be 'auto' or a Hex colour in the form #RRGGBB.")
    text = value.strip()
    if text.lower() == "auto":
        return None
    if len(text) != 7 or text[0] != "#":
        raise ValueError(f"{name} must be 'auto' or a Hex colour in the form #RRGGBB.")
    try:
        channels = [int(text[index : index + 2], 16) / 255.0 for index in (1, 3, 5)]
    except ValueError as exc:
        raise ValueError(f"{name} must be 'auto' or a Hex colour in the form #RRGGBB.") from exc
    return torch.tensor(channels, dtype=torch.float32)


def _parse_vec3(value: Any, default: tuple[float, float, float], name: str) -> torch.Tensor:
    """Parse a vec3 widget value from a string, sequence, or scalar."""
    if value is None:
        values = default
    elif isinstance(value, torch.Tensor):
        values = tuple(float(item) for item in value.detach().flatten().cpu().tolist())
    elif isinstance(value, str):
        text = value.strip().replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        text = text.replace(";", ",")
        try:
            values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must contain three comma-separated numbers.") from exc
    elif isinstance(value, Sequence):
        try:
            values = tuple(float(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain three numeric values.") from exc
    else:
        raise ValueError(f"{name} must contain three comma-separated numbers.")
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values.")
    return torch.tensor(values, dtype=torch.float32)


def _bisenet_weights_path() -> Path:
    try:
        import folder_paths

        candidate = Path(folder_paths.models_dir) / "facexlib" / "parsing_bisenet.pth"
    except Exception:
        candidate = Path(__file__).resolve().parents[1] / "models" / "facexlib" / "parsing_bisenet.pth"
    if candidate.is_file() and candidate.stat().st_size > 0:
        return candidate
    with _BISE_NET_DOWNLOAD_LOCK:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
        candidate.parent.mkdir(parents=True, exist_ok=True)
        temporary = candidate.with_name(candidate.name + ".download")
        errors: list[str] = []
        for url in (_BISE_NET_OFFICIAL_URL, _BISE_NET_HF_URL):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-CineStyle/1.0"})
                with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                if temporary.stat().st_size <= 0:
                    raise OSError("downloaded file is empty")
                temporary.replace(candidate)
                print(f"[CineStyle] Downloaded BiSeNet weights to {candidate}.")
                return candidate
            except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
                errors.append(f"{url}: {exc}")
                temporary.unlink(missing_ok=True)
    raise FileNotFoundError(
        "BiSeNet weights are unavailable and automatic download failed. "
        f"Place parsing_bisenet.pth at {candidate}. Download errors: {'; '.join(errors)}"
    )


def _bisenet_type():
    module_name = "_cinestyle_face_parsing_bisenet"
    module = sys.modules.get(module_name)
    if module is None:
        module_path = Path(__file__).with_name("face_parsing_bisenet.py")
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load BiSeNet implementation from {module_path}.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.BiSeNet


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load a local state dict across Torch versions before ``weights_only``."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # ``weights_only`` was added after the oldest Torch versions still
        # encountered in ComfyUI installations.  This path is only used for a
        # local, trusted state-dict file.
        return torch.load(path, map_location="cpu")


@torch.no_grad()
def _bisenet_skin_mask(images: torch.Tensor, progress: _BeautyProgress | None = None) -> torch.Tensor:
    """Generate a temporary full-frame skin mask at the colour proxy size."""
    model = None
    try:
        if progress is not None:
            progress.info("load BiSeNet", f"frames={images.shape[0]}, size={images.shape[2]}x{images.shape[1]}")
        model = _bisenet_type()(num_class=19)
        state = _load_state_dict(_bisenet_weights_path())
        model.load_state_dict(state, strict=True)
        model.eval().to(images.device)
        network_input = images.movedim(-1, 1).clamp(0.0, 1.0)
        network_input = (network_input - 0.5) / 0.5
        masks: list[torch.Tensor] = []
        for start in range(0, network_input.shape[0], 2):
            logits = model(network_input[start : start + 2])[0]
            masks.append(logits.argmax(dim=1).eq(_BISE_NET_SKIN_CLASS).float())
        if progress is not None:
            progress.info("BiSeNet complete", "skin mask generated; temporary model will be released")
        return torch.cat(masks, dim=0)
    finally:
        del model
        gc.collect()
        if images.device.type == "cuda":
            torch.cuda.empty_cache()
        if progress is not None:
            progress.info("release BiSeNet", "temporary model and CUDA cache released")


def _as_rgb(images: torch.Tensor) -> torch.Tensor:
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError("front must be an IMAGE tensor with shape [batch, height, width, 3 or 4].")
    if images.shape[0] == 0 or images.shape[1] == 0 or images.shape[2] == 0:
        raise ValueError("front must contain at least one non-empty image.")
    return images[..., :3].to(dtype=torch.float32)


def _preferred_device(fallback: torch.device) -> torch.device:
    """Use ComfyUI's active compute device when IMAGE arrived on the CPU."""
    try:
        import comfy.model_management as model_management

        device = model_management.get_torch_device()
        if isinstance(device, torch.device):
            return device
        return torch.device(device)
    except Exception:
        return fallback


def _resize_bhwc(images: torch.Tensor, height: int, width: int, mode: str = "bilinear") -> torch.Tensor:
    if images.shape[1] == height and images.shape[2] == width:
        return images
    chw = images.movedim(-1, 1)
    if mode == "nearest":
        resized = F.interpolate(chw, size=(height, width), mode=mode)
    else:
        resized = F.interpolate(chw, size=(height, width), mode=mode, align_corners=False)
    return resized.movedim(1, -1)


def _axis_kernel(image: torch.Tensor, kernel: torch.Tensor, axis: str, edge: str) -> torch.Tensor:
    """Apply a separable kernel with one grouped CUDA convolution.

    The previous implementation launched one tensor-indexing operation for
    every radius offset.  Grouped convolution keeps the same batched BHWC
    contract while letting cuDNN process all channels and frames together.
    """
    if kernel.numel() <= 1:
        return image
    channels = int(image.shape[-1])
    radius = int(kernel.numel() // 2)
    x = image.movedim(-1, 1)
    if axis == "x":
        weight = kernel.to(device=x.device, dtype=x.dtype).view(1, 1, 1, -1)
    else:
        weight = kernel.to(device=x.device, dtype=x.dtype).view(1, 1, -1, 1)
    weight = weight.expand(channels, 1, *weight.shape[-2:]).contiguous()
    x = _pad_axis_bchw(x, radius, axis, edge)
    return F.conv2d(x, weight, groups=channels).movedim(1, -1)


def _pad_axis_bchw(image: torch.Tensor, radius: int, axis: str, edge: str) -> torch.Tensor:
    """Pad one BCHW axis, including circular padding larger than the image."""
    if radius <= 0:
        return image
    if axis == "x":
        size = int(image.shape[-1])
        if edge == "repeat":
            indices = torch.arange(-radius, size + radius, device=image.device).remainder(size)
        else:
            indices = torch.arange(-radius, size + radius, device=image.device).clamp(0, size - 1)
        return image.index_select(-1, indices)
    size = int(image.shape[-2])
    if edge == "repeat":
        indices = torch.arange(-radius, size + radius, device=image.device).remainder(size)
    else:
        indices = torch.arange(-radius, size + radius, device=image.device).clamp(0, size - 1)
    return image.index_select(-2, indices)


def _triangular_kernel(radius: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(0.0, float(radius))
    radius_int = int(radius)
    if radius_int <= 0:
        return torch.ones(1, device=device, dtype=dtype)
    offsets = torch.arange(-radius_int, radius_int + 1, device=device, dtype=dtype)
    kernel = 1.0 - offsets.abs() / float(radius)
    return kernel / kernel.sum().clamp_min(_EPS)


def _gaussian_kernel(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sigma = max(0.0, float(sigma))
    radius = int(sigma * 3.0)
    if radius <= 0:
        return torch.ones(1, device=device, dtype=dtype)
    positions = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (positions / float(sigma)).square())
    return kernel / kernel.sum().clamp_min(_EPS)


def _prepare_matte(matte: torch.Tensor | None, batch: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    if matte is None:
        return torch.ones((batch, height, width), device=device, dtype=torch.float32)
    if not isinstance(matte, torch.Tensor):
        raise ValueError("matte must be an IMAGE or MASK tensor.")
    if matte.ndim == 4:
        if matte.shape[-1] < 1:
            raise ValueError("matte must have at least one channel.")
        value = matte[..., :1]
    elif matte.ndim == 3:
        value = matte.unsqueeze(-1)
    else:
        raise ValueError("matte must have shape [batch, height, width] or [batch, height, width, channels].")
    value = value.to(device=device, dtype=torch.float32)
    if value.shape[0] not in (1, batch):
        raise ValueError("matte batch size must be 1 or match front.")
    value = _resize_bhwc(value, height, width)
    if value.shape[0] == 1 and batch != 1:
        value = value.expand(batch, -1, -1, -1)
    return value[..., 0]


def _triangular_blur(image: torch.Tensor, radius: float, axis: str, edge: str) -> torch.Tensor:
    """Match the XML shader's linear triangular blur kernel."""
    kernel = _triangular_kernel(radius, image.device, image.dtype)
    return _axis_kernel(image, kernel, axis, edge)


def _edge_preserving_blur(image: torch.Tensor, sigma: float, threshold: float, axis: str) -> torch.Tensor:
    """Port the adaptive Gaussian blur used by CROK Beauty passes 6/7/16/17."""
    sigma = max(0.0, float(sigma))
    if sigma <= _EPS:
        return image
    support = int(sigma * 3.0)
    if support <= 0:
        return image

    # A 601-tap window at the XML maximum would be needlessly large for a
    # video batch.  Use a Gaussian convolution plus a local edge gate in that
    # regime; this keeps the control responsive without allocating a huge
    # [B,H,W,K,C] temporary tensor.
    batch, height, width, channels = map(int, image.shape)
    chunk_size = 2 if image.device.type == "cuda" else 1
    estimated_elements = min(batch, chunk_size) * height * width * (2 * support + 1) * channels
    if support > 64 or estimated_elements > 450_000_000:
        blurred = _axis_kernel(image, _gaussian_kernel(sigma, image.device, image.dtype), axis, "repeat")
        distance = (blurred - image).square().sum(dim=-1, keepdim=True).sqrt() / math.sqrt(3.0)
        gate = torch.exp(-distance * max(0.0, float(threshold)))
        return image + (blurred - image) * gate

    # The shader forces neighbour alpha to 1 and computes colour distance in
    # RGB, so keeping this helper RGB-only reproduces the effective operation.
    pi = math.pi
    gaussian0 = 1.0 / (math.sqrt(2.0 * pi) * sigma)
    gaussian_step = math.exp(-0.5 / (sigma * sigma))
    rgb_hyp = math.sqrt(3.0)
    threshold = max(0.0, float(threshold))

    # Build all integer neighbours in one unfold operation instead of issuing
    # two index_select calls for every radius value.  The batch is chunked to
    # keep the temporary [B,H,W,2*support,C] tensor bounded for video batches.
    x = image.movedim(-1, 1)
    batch = int(x.shape[0])
    chunk_size = 2 if x.device.type == "cuda" else 1
    output = torch.empty_like(image)
    kernel_size = 2 * support + 1
    for start in range(0, batch, chunk_size):
        end = min(batch, start + chunk_size)
        part = x[start:end]
        if axis == "x":
            padded = _pad_axis_bchw(part, support, "x", "repeat")
            windows = padded.unfold(3, kernel_size, 1)
        else:
            padded = _pad_axis_bchw(part, support, "y", "repeat")
            windows = padded.unfold(2, kernel_size, 1)
        windows = windows.permute(0, 2, 3, 4, 1).contiguous()  # B,H,W,K,C
        center = windows[..., support, :]
        neighbours = torch.cat((windows[..., :support, :], windows[..., support + 1 :, :]), dim=-2)

        offsets = torch.arange(1, support + 1, device=image.device, dtype=image.dtype)
        coefficients = float(gaussian0) * torch.exp(-0.5 * (offsets / float(sigma)).square())
        coefficients = torch.cat((coefficients.flip(0), coefficients), dim=0)
        coefficients = coefficients.view(1, 1, 1, -1, 1)

        distance = (neighbours - center.unsqueeze(-2)).square().sum(dim=-1, keepdim=True).sqrt() / rgb_hyp
        # The source calls pow() before clamp().  Clamping the base first
        # avoids NaNs for non-integer threshold values while preserving the
        # intended 0.001 minimum contribution.
        factor = (1.0 - distance).clamp(0.0, 1.0).pow(threshold).clamp(0.001, 1.0)
        weighted = coefficients * factor
        result = center * float(gaussian0) + (neighbours * weighted).sum(dim=-2)
        energy = float(gaussian0) + weighted.sum(dim=-2)
        output[start:end] = result / energy.clamp_min(_EPS)
    return output


def _rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    maximum = rgb.max(dim=-1).values
    minimum = rgb.min(dim=-1).values
    delta = maximum - minimum
    hue = torch.zeros_like(maximum)
    safe_delta = delta.clamp_min(_EPS)

    red = maximum.eq(rgb[..., 0])
    green = maximum.eq(rgb[..., 1])
    hue = torch.where(red, (rgb[..., 1] - rgb[..., 2]) / safe_delta, hue)
    hue = torch.where(green & ~red, 2.0 + (rgb[..., 2] - rgb[..., 0]) / safe_delta, hue)
    hue = torch.where(~red & ~green, 4.0 + (rgb[..., 0] - rgb[..., 1]) / safe_delta, hue)
    hue = torch.remainder(hue / 6.0, 1.0)
    saturation = torch.where(maximum > minimum, delta / maximum.clamp_min(_EPS), torch.zeros_like(delta))
    return torch.stack((hue, saturation, maximum), dim=-1)


def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    hue, saturation, value = hsv.unbind(dim=-1)
    h = torch.remainder(hue, 1.0) * 6.0
    index = torch.floor(h).to(torch.int64)
    fraction = h - torch.floor(h)
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * fraction)
    t = value * (1.0 - saturation * (1.0 - fraction))
    choices = torch.stack(
        (
            torch.stack((value, t, p), dim=-1),
            torch.stack((q, value, p), dim=-1),
            torch.stack((p, value, t), dim=-1),
            torch.stack((p, q, value), dim=-1),
            torch.stack((t, p, value), dim=-1),
            torch.stack((value, p, q), dim=-1),
        ),
        dim=-2,
    )
    return choices.gather(-2, index.unsqueeze(-1).unsqueeze(-1).expand(*index.shape, 1, 3)).squeeze(-2)


def _circular_hue_peak(values: torch.Tensor) -> torch.Tensor:
    bins = 180
    indices = (values.clamp(0.0, 1.0 - _EPS) * bins).to(torch.int64)
    histogram = torch.bincount(indices, minlength=bins)
    peak = (histogram.argmax().to(values.dtype) + 0.5) / bins
    distance = torch.remainder(values - peak + 0.5, 1.0) - 0.5
    local = values[distance.abs() <= 0.15]
    if local.numel() == 0:
        local = values
    angles = local * (2.0 * math.pi)
    return torch.remainder(torch.atan2(torch.sin(angles).mean(), torch.cos(angles).mean()) / (2.0 * math.pi), 1.0)


def _estimate_colour_from_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    alpha: torch.Tensor,
    fallback: torch.Tensor,
) -> torch.Tensor:
    """Estimate one clip-stable RGB colour from masked, valid skin pixels."""
    batch = int(image.shape[0])
    sample_count = min(batch, _COLOUR_SAMPLE_FRAMES)
    indices = torch.linspace(0, batch - 1, sample_count).round().to(torch.int64).unique().to(image.device)
    image = image.index_select(0, indices)
    mask = mask.index_select(0, indices)
    alpha = alpha.index_select(0, indices)
    hsv = _rgb_to_hsv(image)
    frame_stats: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for frame_index in range(image.shape[0]):
        frame = hsv[frame_index]
        valid = (
            (mask[frame_index] > 0.5)
            & (alpha[frame_index] > 0.01)
            & torch.isfinite(frame).all(dim=-1)
            & (frame[..., 2] > 0.15)
            & (frame[..., 2] < 0.90)
            & (frame[..., 1] > 0.05)
        )
        # Histogram/quantile support is inconsistent on older MPS builds and
        # some non-CUDA backends.  The candidate set is small enough that
        # doing these statistics on CPU does not affect the GPU-heavy path.
        pixels = frame[valid].detach().to(device="cpu", dtype=torch.float32)
        if pixels.shape[0] < 32:
            continue
        frame_stats.append(
            (
                _circular_hue_peak(pixels[:, 0]),
                torch.quantile(pixels[:, 1], 0.5),
                torch.quantile(pixels[:, 2], 0.5),
            )
        )
    if not frame_stats:
        return fallback.to(device=image.device, dtype=image.dtype)

    hues = torch.stack([item[0] for item in frame_stats])
    saturations = torch.stack([item[1] for item in frame_stats])
    values = torch.stack([item[2] for item in frame_stats])
    # Unwrap hue around the first sample, smooth it across sampled frames, and
    # take a median so one lighting change cannot move the whole clip's key.
    unwrapped = [hues[0]]
    for current in hues[1:]:
        delta = torch.remainder(current - unwrapped[-1] + 0.5, 1.0) - 0.5
        unwrapped.append(unwrapped[-1] + delta)
    smooth_hues = [unwrapped[0]]
    for current in unwrapped[1:]:
        smooth_hues.append(0.75 * smooth_hues[-1] + 0.25 * current)
    hue = torch.remainder(torch.stack(smooth_hues).median(), 1.0)
    saturation = saturations.median()
    value = values.median()
    return _hsv_to_rgb(torch.stack((hue, saturation, value)).view(1, 1, 1, 3)).view(3).to(
        device=image.device, dtype=image.dtype
    )


def _estimate_clip_colour(
    source: torch.Tensor,
    external: torch.Tensor | None,
    alpha: torch.Tensor,
    progress: _BeautyProgress | None = None,
) -> torch.Tensor:
    """Estimate a clip-stable key colour using an optional mask or BiSeNet."""
    _, height, width, _ = source.shape
    colour_height, colour_width = height, width
    if max(height, width) > _COLOUR_LONG_SIDE:
        colour_scale = _COLOUR_LONG_SIDE / float(max(height, width))
        colour_height = max(1, int(round(height * colour_scale)))
        colour_width = max(1, int(round(width * colour_scale)))
    colour_image = _resize_bhwc(source, colour_height, colour_width)
    colour_alpha = _resize_bhwc(alpha.unsqueeze(-1), colour_height, colour_width)[..., 0]
    if external is not None:
        colour_mask = _resize_bhwc(external.unsqueeze(-1), colour_height, colour_width)[..., 0]
    else:
        colour_mask = _bisenet_skin_mask(colour_image, progress=progress)
    fallback = source.new_tensor(_AUTO_COLOUR_FALLBACK)
    colour = _estimate_colour_from_mask(colour_image, colour_mask, colour_alpha, fallback)
    del colour_image, colour_mask, colour_alpha
    return colour


def _skin_matte(image: torch.Tensor, colour: torch.Tensor, weights: torch.Tensor, external: torch.Tensor | None) -> torch.Tensor:
    if external is not None:
        return external.clamp(0.0, 1.0)
    hsv = _rgb_to_hsv(image)
    target = _rgb_to_hsv(colour.to(device=image.device, dtype=image.dtype).view(1, 1, 1, 3))
    distance = (weights.to(device=image.device, dtype=image.dtype).view(1, 1, 1, 3) * (target - hsv)).square().sum(dim=-1).sqrt()
    matte = 1.0 - (3.0 * distance - 1.5).clamp(0.0, 1.0)
    return matte.clamp(0.0, 1.0)


def _lighten(source: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
    return torch.maximum(source, destination)


def _darker_color(source: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
    return torch.where(
        source.sum(dim=-1, keepdim=True) < destination.sum(dim=-1, keepdim=True),
        source,
        destination,
    )


def _overlay(source: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
    low = 2.0 * source * destination
    high = 1.0 - 2.0 * (1.0 - source) * (1.0 - destination)
    return torch.where(destination < 0.5, low, high)


def _rgb_to_yuv(rgb: torch.Tensor) -> torch.Tensor:
    # Rec.601-style full-range YUV.  Matchbox provides this as an API call;
    # keeping it local makes the Torch port deterministic and self-contained.
    r, g, b = rgb.unbind(dim=-1)
    y = 0.29900 * r + 0.58700 * g + 0.11400 * b
    u = -0.14713 * r - 0.28886 * g + 0.43600 * b
    v = 0.61500 * r - 0.51499 * g - 0.10001 * b
    return torch.stack((y, u, v), dim=-1)


def _yuv_to_rgb(yuv: torch.Tensor) -> torch.Tensor:
    y, u, v = yuv.unbind(dim=-1)
    r = y + 1.13983 * v
    g = y - 0.39465 * u - 0.58060 * v
    b = y + 2.03211 * u
    return torch.stack((r, g, b), dim=-1)


def _hueshift(rgb: torch.Tensor, hue_amount: float, saturation: float) -> torch.Tensor:
    yuv = _rgb_to_yuv(rgb)
    radians = math.radians(float(hue_amount))
    cosine = math.cos(radians)
    sine = math.sin(radians)
    scale = float(saturation) * 0.01
    u, v = yuv[..., 1], yuv[..., 2]
    rotated = torch.stack(
        (
            yuv[..., 0],
            scale * cosine * u - sine * v,
            sine * u + scale * cosine * v,
        ),
        dim=-1,
    )
    return _yuv_to_rgb(rotated)


def _highpass(original: torch.Tensor, cleaned: torch.Tensor, strength: float) -> torch.Tensor:
    channels = int(original.shape[-1])
    x = original.movedim(-1, 1)
    kernel = original.new_tensor(
        [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]
    ).view(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous()
    padded = _pad_axis_bchw(_pad_axis_bchw(x, 1, "x", "repeat"), 1, "y", "repeat")
    response = F.conv2d(padded, kernel, groups=channels).movedim(1, -1)
    key = cleaned[..., 3:4]
    return response * float(strength) * key * 0.1 + 0.5


def _compute_proxy(
    original: torch.Tensor,
    external_matte: torch.Tensor | None,
    colour: torch.Tensor,
    weights: torch.Tensor,
    blur_m: float,
    sigma: float,
    threshold: float,
    r_spots_blend: float,
    r_h_blend: float,
    strength: float,
    blur_h: float,
    blur_s: float,
    o_amount: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate passes 1-17 and return cleaned skin, matte, and degrained RGB."""
    # Pass 3: chroma key or external matte.
    matte = _skin_matte(original, colour, weights, external_matte)

    # Passes 4-5: triangular matte softening, clamp-to-edge sampling.
    matte_image = matte.unsqueeze(-1)
    matte_image = _triangular_blur(matte_image, blur_m, "y", "clamp")
    matte_image = _triangular_blur(matte_image, blur_m, "x", "clamp")
    matte = matte_image[..., 0].clamp(0.0, 1.0)

    # Passes 6-7: edge-preserving skin blur.
    dollface = _edge_preserving_blur(original, sigma, threshold, "x")
    dollface = _edge_preserving_blur(dollface, sigma, threshold, "y")

    # Pass 8: remove dark spots and highlights.
    c = _lighten(dollface, original)
    c = original + (c - original) * float(r_spots_blend)
    c = matte.unsqueeze(-1) * c + (1.0 - matte.unsqueeze(-1)) * original

    diff = (dollface - original).abs()
    diff = diff @ original.new_tensor([0.2125, 0.7154, 0.0721])
    diff = diff.clamp(0.0, 1.0)
    diff = (1.0 - diff) * matte
    darkened = _darker_color(dollface, c)
    darkened = c + (darkened - c) * float(r_h_blend)
    cleaned = diff.unsqueeze(-1) * darkened + (1.0 - diff.unsqueeze(-1)) * c

    cleaned_rgba = torch.cat((cleaned, matte.unsqueeze(-1)), dim=-1)

    # Passes 9-11: high-pass extraction and triangular softening.
    highpass = _highpass(original, cleaned_rgba, strength)
    highpass = _triangular_blur(highpass, blur_h, "y", "repeat")
    highpass = _triangular_blur(highpass, blur_h, "x", "repeat")

    # Pass 12: overlay softened high-frequency detail.
    beauty = _overlay(cleaned, highpass[..., :3])
    beauty_rgba = torch.cat((beauty, matte.unsqueeze(-1)), dim=-1)

    # Passes 13-14: shine blur.
    shine = _triangular_blur(beauty_rgba, blur_s, "y", "repeat")
    shine = _triangular_blur(shine, blur_s, "x", "repeat")

    # Pass 15: restore shine.
    shine_comp = _overlay(beauty, shine[..., :3])
    shine_comp = beauty + (shine_comp - beauty) * (float(o_amount) * matte.unsqueeze(-1))
    cleaned_skin = torch.cat((shine_comp, matte.unsqueeze(-1)), dim=-1)

    # Passes 16-17: fixed degrain blur.
    degrain = _edge_preserving_blur(original, 2.0, 100.0, "x")
    degrain = _edge_preserving_blur(degrain, 2.0, 100.0, "y")
    return cleaned_skin, matte, degrain


def _run_beauty(
    image: torch.Tensor,
    matte_input: torch.Tensor | None,
    colour: torch.Tensor | None,
    weights: torch.Tensor,
    blur_m: float,
    sigma: float,
    threshold: float,
    r_spots_blend: float,
    r_h_blend: float,
    strength: float,
    blur_h: float,
    blur_s: float,
    o_amount: float,
    sat_amount: float,
    hue_amount: float,
    progress: _BeautyProgress | None = None,
) -> torch.Tensor:
    source_device = image.device if isinstance(image, torch.Tensor) else torch.device("cpu")
    source = _as_rgb(image).to(device=_preferred_device(source_device), non_blocking=True)
    batch, height, width, _ = source.shape
    device = source.device
    if progress is not None:
        progress.info("prepare input", f"frames={batch}, size={width}x{height}, device={device}")
    if isinstance(image, torch.Tensor) and image.shape[-1] >= 4:
        alpha = image[..., 3].to(device=device, dtype=torch.float32, non_blocking=True)
    else:
        alpha = torch.ones((batch, height, width), device=device, dtype=torch.float32)
    external = _prepare_matte(matte_input, batch, height, width, device) if matte_input is not None else None

    if colour is None:
        # Colour estimation is deliberately independent from the beauty proxy.
        # It uses a 512-long-side clip sample and is therefore stable across
        # all frames without allocating a full-resolution histogram.
        if progress is not None:
            progress.info("estimate colour", "auto mode; using mask" if external is not None else "auto mode; running BiSeNet")
        colour = _estimate_clip_colour(source, external, alpha, progress=progress)
    else:
        if progress is not None:
            progress.info("use fixed colour", "skipping automatic colour estimation")
        colour = colour.to(device=device, dtype=source.dtype)

    longest = max(height, width)
    proxy_height, proxy_width = height, width
    if longest > _PROXY_LONG_SIDE:
        scale = _PROXY_LONG_SIDE / float(longest)
        proxy_height = max(1, int(round(height * scale)))
        proxy_width = max(1, int(round(width * scale)))

    proxy_source = _resize_bhwc(source, proxy_height, proxy_width)
    proxy_external = None if external is None else _resize_bhwc(external.unsqueeze(-1), proxy_height, proxy_width)[..., 0]
    scale_x = proxy_width / float(width)
    scale_y = proxy_height / float(height)
    # Shader radii are pixel distances.  Scale them with the proxy so their
    # physical size remains consistent after the proxy result is upsampled.
    proxy_blur_m = float(blur_m) * (scale_x + scale_y) * 0.5
    proxy_sigma = float(sigma) * (scale_x + scale_y) * 0.5
    proxy_blur_h = float(blur_h) * (scale_x + scale_y) * 0.5
    proxy_blur_s = float(blur_s) * (scale_x + scale_y) * 0.5

    if progress is not None:
        progress.info("proxy setup", f"proxy={proxy_width}x{proxy_height}")

    if progress is not None:
        progress.info("run proxy passes", "passes 1-17; internal Torch kernels do not expose per-frame callbacks")
    cleaned_proxy, matte_proxy, degrain_proxy = _compute_proxy(
        proxy_source,
        proxy_external,
        colour,
        weights,
        proxy_blur_m,
        proxy_sigma,
        threshold,
        r_spots_blend,
        r_h_blend,
        strength,
        proxy_blur_h,
        proxy_blur_s,
        o_amount,
    )
    if progress is not None:
        progress.info("proxy passes complete")

    cleaned = _resize_bhwc(cleaned_proxy, height, width)
    degrain = _resize_bhwc(degrain_proxy, height, width)
    matte = cleaned[..., 3:4].clamp(0.0, 1.0)

    # Pass 18: regrain at source resolution.  This keeps high-frequency input
    # detail even when the beauty and blur stages used a proxy resolution.
    grain = source - degrain
    result = grain + cleaned[..., :3]
    result = matte * result + (1.0 - matte) * cleaned[..., :3]

    # Pass 19: Matchbox's YUV-plane hue/saturation transform, masked to skin.
    shifted = _hueshift(result, hue_amount, sat_amount)
    result = matte * shifted + (1.0 - matte) * result
    result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    output = torch.cat((result, matte), dim=-1)
    if progress is not None:
        progress.info("final colour pass", "regrain and YUV hue/saturation adjustment complete")
        progress.info("complete", f"frames={batch}")
    return output.to(device=source_device, non_blocking=True)


def _sample_indices(batch: int, limit: int, device: torch.device | str = "cpu") -> torch.Tensor:
    count = min(max(1, int(batch)), max(1, int(limit)))
    return torch.linspace(0, batch - 1, count, device=device).round().to(torch.int64).unique()


def _slice_matte_input(
    matte: torch.Tensor | None,
    indices: torch.Tensor,
    total_batch: int,
) -> torch.Tensor | None:
    """Select only the frames needed by a batch without moving the full mask."""
    if matte is None:
        return None
    if not isinstance(matte, torch.Tensor):
        raise ValueError("matte must be an IMAGE or MASK tensor.")
    if matte.ndim not in (3, 4):
        raise ValueError("matte must have shape [batch, height, width] or [batch, height, width, channels].")
    if matte.shape[0] == 1:
        return matte
    if matte.shape[0] != total_batch:
        raise ValueError("matte batch size must be 1 or match front.")
    return matte.index_select(0, indices.to(device=matte.device))


def _estimate_video_colour(
    image: torch.Tensor,
    matte_input: torch.Tensor | None,
    progress: _BeautyProgress | None = None,
) -> torch.Tensor:
    """Estimate one clip colour while transferring only sampled frames to the compute device."""
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError("front must be an IMAGE tensor with shape [batch, height, width, 3 or 4].")
    batch, height, width = map(int, image.shape[:3])
    if batch == 0 or height == 0 or width == 0:
        raise ValueError("front must contain at least one non-empty image.")
    indices = _sample_indices(batch, _COLOUR_SAMPLE_FRAMES, device=image.device)
    sample_height, sample_width = height, width
    if max(height, width) > _COLOUR_LONG_SIDE:
        colour_scale = _COLOUR_LONG_SIDE / float(max(height, width))
        sample_height = max(1, int(round(height * colour_scale)))
        sample_width = max(1, int(round(width * colour_scale)))
    compute_device = _preferred_device(image.device)
    sampled_source = _resize_bhwc(
        _as_rgb(image.index_select(0, indices)), sample_height, sample_width
    ).to(device=compute_device, non_blocking=True)
    if isinstance(image, torch.Tensor) and image.shape[-1] >= 4:
        sampled_alpha = _resize_bhwc(
            image[..., 3].index_select(0, indices).unsqueeze(-1), sample_height, sample_width
        )[..., 0].to(device=compute_device, dtype=torch.float32, non_blocking=True)
    else:
        sampled_alpha = torch.ones((indices.numel(), sample_height, sample_width), device=compute_device, dtype=torch.float32)
    sampled_matte = _slice_matte_input(matte_input, indices, batch)
    if sampled_matte is None:
        external = None
    else:
        sampled_matte = _prepare_matte(
            sampled_matte,
            int(indices.numel()),
            height,
            width,
            sampled_matte.device,
        )
        external = _resize_bhwc(sampled_matte.unsqueeze(-1), sample_height, sample_width)[..., 0]
        external = external.to(device=compute_device, dtype=torch.float32, non_blocking=True)
    try:
        return _estimate_clip_colour(sampled_source, external, sampled_alpha, progress=progress)
    finally:
        del sampled_source, sampled_alpha, sampled_matte, external
        if compute_device.type == "cuda":
            torch.cuda.empty_cache()


def _beauty_batch_size(image: torch.Tensor, total_batch: int, progress: _BeautyProgress | None = None) -> int:
    """Choose a conservative frame batch from current free GPU memory."""
    if total_batch <= 1:
        return 1
    source_device = image.device if isinstance(image, torch.Tensor) else torch.device("cpu")
    compute_device = _preferred_device(source_device)
    height, width = int(image.shape[1]), int(image.shape[2])
    longest = max(height, width)
    if longest > _PROXY_LONG_SIDE:
        scale = _PROXY_LONG_SIDE / float(longest)
        proxy_height = max(1, int(round(height * scale)))
        proxy_width = max(1, int(round(width * scale)))
    else:
        proxy_height, proxy_width = height, width

    # Account for the source frame, resized proxy, convolution workspaces and
    # several same-size intermediates.  This is intentionally conservative;
    # the OOM retry below adapts to drivers with unusually large workspaces.
    source_bytes = height * width * 3 * 4
    proxy_bytes = proxy_height * proxy_width * 4 * 4
    estimated_per_frame = max(source_bytes * 3, proxy_bytes * 24)
    if compute_device.type == "cuda":
        try:
            free_bytes, _ = torch.cuda.mem_get_info(compute_device)
            budget = max(
                estimated_per_frame,
                int((free_bytes - _BEAUTY_MEMORY_RESERVE_BYTES) * _BEAUTY_GPU_MEMORY_FRACTION),
            )
            selected = max(1, min(_BEAUTY_MAX_GPU_BATCH, int(budget // estimated_per_frame)))
            if progress is not None:
                progress.info(
                    "batch plan",
                    f"GPU free={free_bytes / 1024**3:.2f} GiB; estimated={estimated_per_frame / 1024**2:.1f} MiB/frame; batch={selected}",
                )
            return min(total_batch, selected)
        except (RuntimeError, AttributeError, TypeError):
            pass
    selected = min(total_batch, _BEAUTY_CPU_BATCH)
    if progress is not None:
        progress.info("batch plan", f"device={compute_device}; batch={selected}")
    return max(1, selected)


def _run_beauty_batched(
    image: torch.Tensor,
    matte_input: torch.Tensor | None,
    colour: torch.Tensor | None,
    weights: torch.Tensor,
    blur_m: float,
    sigma: float,
    threshold: float,
    r_spots_blend: float,
    r_h_blend: float,
    strength: float,
    blur_h: float,
    blur_s: float,
    o_amount: float,
    sat_amount: float,
    hue_amount: float,
    progress: _BeautyProgress | None = None,
) -> torch.Tensor:
    """Run the unchanged Beauty pipeline in bounded frame batches."""
    total_batch = int(image.shape[0])
    batch_size = _beauty_batch_size(image, total_batch, progress=progress)
    output_store = torch.empty(
        (total_batch, int(image.shape[1]), int(image.shape[2]), 4),
        device="cpu",
        dtype=torch.float32,
    )
    start = 0
    while start < total_batch:
        end = min(total_batch, start + batch_size)
        frame_indices = torch.arange(start, end, device=image.device)
        image_chunk = image.index_select(0, frame_indices)
        matte_chunk = _slice_matte_input(matte_input, frame_indices, total_batch)
        try:
            if progress is not None:
                progress.info("process batch", f"frames={start + 1}-{end}/{total_batch}; batch={end - start}")
            output_chunk = _run_beauty(
                image_chunk,
                matte_chunk,
                colour,
                weights,
                blur_m,
                sigma,
                threshold,
                r_spots_blend,
                r_h_blend,
                strength,
                blur_h,
                blur_s,
                o_amount,
                sat_amount,
                hue_amount,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            del image_chunk, matte_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if batch_size <= 1:
                raise
            batch_size = max(1, batch_size // 2)
            if progress is not None:
                progress.info("reduce batch", f"CUDA OOM; retrying with batch={batch_size}")
            continue
        output_store[start:end].copy_(output_chunk.to(device="cpu", dtype=torch.float32), non_blocking=True)
        del output_chunk, image_chunk, matte_chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        start = end
    return output_store


def _cache_vfx_input(
    node_id: Any,
    prompt: Any,
    image: torch.Tensor,
    mask: torch.Tensor | None,
    proxy_video: Any = None,
) -> None:
    """Make the last node input available to the browser preview dialog."""
    key = str(node_id or "").strip()
    if not key or not isinstance(image, torch.Tensor) or image.ndim != 4:
        return
    loader_origin = _loader_id_from_video(proxy_video) or _loader_id_from_prompt(prompt, node_id)
    proxy_present = False
    try:
        fps = 24.0
        if isinstance(prompt, dict):
            pending = [prompt.get(str(node_id)) or prompt.get(node_id)]
            visited = set()
            while pending and fps == 24.0:
                current = pending.pop(0)
                if not isinstance(current, dict):
                    continue
                inputs = current.get("inputs") if isinstance(current.get("inputs"), dict) else {}
                for name in ("fps", "frame_rate", "target_fps"):
                    try:
                        candidate = float(inputs.get(name))
                        if math.isfinite(candidate) and candidate > 0:
                            fps = candidate
                            break
                    except (TypeError, ValueError, OverflowError):
                        pass
                for value in inputs.values():
                    if isinstance(value, (list, tuple)) and len(value) >= 2 and str(value[0]) not in visited:
                        visited.add(str(value[0]))
                        pending.append(prompt.get(str(value[0])) or prompt.get(value[0]))
        # CS Load Video owns the shared preview MP4. Keep a frame-only local
        # entry for optional colour estimation, but avoid encoding it again.
        _preview_cache_store().put_preview(key, image, fps, encode_video=not loader_origin)
        if loader_origin:
            _console_info(key, "shared loader preview", f"loader={loader_origin}")
    except Exception as exc:
        print(f"[CineStyle] VFX Beauty preview cache failed for node {key}: {exc}")
    try:
        proxy_images = proxy_video.get_components().images if proxy_video is not None else None
        if isinstance(proxy_images, torch.Tensor) and proxy_images.ndim == 4 and proxy_images.shape[0] > 0:
            try:
                proxy_fps = float(proxy_video.get_components().frame_rate)
            except Exception:
                proxy_fps = 24.0
            if loader_origin:
                _console_info(key, "proxy preview uses shared loader cache", f"loader={loader_origin}")
            else:
                proxy_present = _preview_cache_store().put_preview(key, proxy_images, proxy_fps, proxy=True, encode_video=True) is not None
    except Exception as exc:
        print(f"[CineStyle] VFX Beauty proxy cache failed for node {key}: {exc}")
    cached_mask = None
    if isinstance(mask, torch.Tensor) and mask.ndim >= 3:
        cached_mask = mask.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if cached_mask.ndim == 4:
            cached_mask = cached_mask[..., 0]
    with _VFX_CACHE_LOCK:
        _VFX_MASK_CACHE[key] = cached_mask
        _VFX_COLOUR_CACHE.pop(key, None)
        _VFX_PROXY_PRESENT[key] = proxy_present


def _preview_cache_entry(node_id: str) -> dict[str, Any] | None:
    try:
        return _preview_cache_store().get_preview_variant(node_id, proxy=False)
    except Exception:
        return None


def _preview_proxy_cache_entry(node_id: str) -> dict[str, Any] | None:
    with _VFX_CACHE_LOCK:
        if not _VFX_PROXY_PRESENT.get(node_id, False):
            return None
    try:
        return _preview_cache_store().get_preview_variant(node_id, proxy=True)
    except Exception:
        return None


def _preview_mask_frame_index(node_id: str, frame_index: int, source_token: str) -> int:
    """Map a proxy frame to the corresponding original mask frame."""
    try:
        source_entry = (
            _loader_preview_cache().entry_for_token(source_token)
            if source_token.startswith("loader_preview:") and _loader_preview_cache() is not None
            else _preview_cache_store().get_token(source_token)
        )
        original_entry = _preview_cache_store().get_preview_variant(node_id, proxy=False)
        source_count = int((source_entry or {}).get("info", {}).get("frames") or 0)
        original_count = int((original_entry or {}).get("info", {}).get("frames") or 0)
        if source_count > 1 and original_count > 1 and source_count != original_count:
            return int(round(frame_index * (original_count - 1) / (source_count - 1)))
    except Exception:
        pass
    return frame_index


def _preview_mask_for_node(node_id: str, frame_index: int, height: int, width: int) -> torch.Tensor | None:
    with _VFX_CACHE_LOCK:
        mask = _VFX_MASK_CACHE.get(node_id)
    if mask is None or mask.ndim != 3 or frame_index < 0 or (frame_index >= mask.shape[0] and mask.shape[0] != 1):
        return None
    frame = mask[0:1] if mask.shape[0] == 1 else mask[frame_index : frame_index + 1]
    if frame.shape[1:3] != (height, width):
        frame = _resize_bhwc(frame.unsqueeze(-1), height, width, mode="nearest")[..., 0]
    return frame


def _preview_colour(
    node_id: str,
    requested: Any,
    current_frame: torch.Tensor,
    current_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, str]:
    parsed = _parse_hex_colour(requested)
    if parsed is not None:
        return parsed, str(requested).strip().upper()
    with _VFX_CACHE_LOCK:
        cached = _VFX_COLOUR_CACHE.get(node_id)
    if cached is not None:
        return cached, "#%02X%02X%02X" % tuple(int(round(float(v) * 255.0)) for v in cached)

    clip = current_frame
    external = current_mask
    entry = _preview_cache_entry(node_id)
    if entry is not None:
        try:
            frames = np.load(str(entry["frames_path"]), mmap_mode="r", allow_pickle=False)
            clip = torch.from_numpy(np.array(frames, copy=True)).to(torch.float32).div_(255.0)
            with _VFX_CACHE_LOCK:
                all_mask = _VFX_MASK_CACHE.get(node_id)
            if all_mask is not None and all_mask.ndim == 3 and all_mask.shape[0] in (1, clip.shape[0]):
                external = all_mask if all_mask.shape[0] == clip.shape[0] else all_mask.expand(clip.shape[0], -1, -1)
            else:
                external = None
        except (OSError, KeyError, ValueError):
            clip = current_frame
            external = current_mask
    alpha = torch.ones(clip.shape[:3], dtype=torch.float32, device=clip.device)
    colour = _estimate_clip_colour(clip, external, alpha).detach().to(device="cpu", dtype=torch.float32)
    with _VFX_CACHE_LOCK:
        _VFX_COLOUR_CACHE[node_id] = colour
    return colour, "#%02X%02X%02X" % tuple(int(round(float(v) * 255.0)) for v in colour)


def _encode_preview_png(image: torch.Tensor) -> str:
    frame = image[0] if image.ndim == 4 else image
    array = (
        frame[..., :3]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    buffer = py_io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG", optimize=False)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _preview_float(payload: dict[str, Any], name: str, default: float) -> float:
    value = payload.get(name, default)
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


async def _vfx_beauty_cache_info_route(request: web.Request) -> web.Response:
    node_id = str(request.query.get("node_id") or "").strip()
    entry = _preview_cache_store().get_preview(node_id)
    if entry is None:
        return web.json_response({"error": "Run the workflow once to cache the VFX Beauty input."}, status=404)
    info = dict(entry.get("info") or {})
    with _VFX_CACHE_LOCK:
        has_mask = node_id in _VFX_MASK_CACHE and _VFX_MASK_CACHE[node_id] is not None
        uses_proxy = bool(entry and entry.get("variant") == "proxy")
    return web.json_response(
        {
            "token": str(entry.get("token") or ""),
            "label": "Proxy input from the last workflow run" if uses_proxy else "Cached input from the last workflow run",
            "video_url": f"/cinestyle/vfx-beauty-cache-video?token={entry.get('token')}",
            "info": info,
            "has_mask": has_mask,
            "uses_proxy": uses_proxy,
        }
    )


async def _vfx_beauty_cache_video_route(request: web.Request) -> web.StreamResponse:
    from aiohttp import web

    entry = _preview_cache_store().get_token(request.query.get("token", ""))
    path = Path(str((entry or {}).get("video_path") or (entry or {}).get("path") or "")) if entry else None
    if entry is None or path is None or not path.is_file():
        return web.json_response({"error": "VFX Beauty preview cache not found."}, status=404)
    return web.FileResponse(path=path, headers={"Content-Type": "video/mp4", "Cache-Control": "no-store"})


async def _vfx_beauty_preview_route(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Preview payload must be an object.")
        node_id = str(payload.get("node_id") or "").strip()
        if not node_id:
            raise ValueError("node_id is required.")
        frame_index = max(0, int(payload.get("frame", 0)))
        source_token = str(payload.get("source_token") or "")
        if source_token.startswith("loader_preview:"):
            cache = _loader_preview_cache()
            if cache is None:
                raise ValueError("The shared loader preview cache is unavailable.")
            frame = cache.decode_frame(source_token, frame_index)
        else:
            frame = _preview_cache_store().decode_frame(payload, frame_index)
        height, width = int(frame.shape[1]), int(frame.shape[2])
        mask_index = _preview_mask_frame_index(node_id, frame_index, str(payload.get("source_token") or ""))
        mask = _preview_mask_for_node(node_id, mask_index, height, width)
        colour, colour_label = _preview_colour(node_id, payload.get("colour", "auto"), frame, mask)
        output = _run_beauty(
            frame,
            mask,
            colour,
            _parse_vec3(payload.get("weights", "6.0, 0.0, 3.0"), (6.0, 0.0, 3.0), "weights"),
            _preview_float(payload, "blur_m", 10.0),
            _preview_float(payload, "sigma", 10.0),
            _preview_float(payload, "threshold", 15.0),
            _preview_float(payload, "r_spots_blend", 0.8),
            _preview_float(payload, "r_h_blend", 0.5),
            _preview_float(payload, "strength", 0.0),
            _preview_float(payload, "blur_h", 0.0),
            _preview_float(payload, "blur_s", 30.0),
            _preview_float(payload, "o_amount", 0.2),
            _preview_float(payload, "sat_amount", 100.0),
            _preview_float(payload, "hue_amount", 0.0),
        )
        return web.json_response(
            {
                "frame": int(payload.get("local_frame", frame_index)),
                "colour": colour_label,
                "original": _encode_preview_png(frame),
                "preview": _encode_preview_png(output),
            }
        )
    except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)


class CSVFXBeauty(io.ComfyNode):
    """Torch port of the 19-pass Matchbox ``crok_beauty`` shader."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CS_VFX_Beauty",
            display_name="CS VFX Beauty",
            category="😺dzNodes/CineStyle",
            essentials_category="Image Effects",
            search_aliases=["crok beauty", "skin beauty", "dollface", "matchbox beauty"],
            description="Torch port of the Matchbox CROK Beauty skin cleanup pipeline.",
            inputs=[
                io.Image.Input("image", tooltip="Original foreground image."),
                io.Mask.Input("mask", optional=True, tooltip="Optional skin mask. When connected, it is used automatically for the beauty region and colour estimation."),
                io.Video.Input("proxy_video", optional=True, tooltip="Optional proxy VIDEO used only by VFX Preview. Rendering still uses image."),
                io.String.Input("colour", default="auto", tooltip="Use auto for clip skin-colour estimation, or enter a #RRGGBB value to bypass automatic estimation."),
                io.String.Input("weights", default="6.0, 0.0, 3.0", tooltip="HSV keying weights vec3."),
                io.Float.Input("blur_m", default=10.0, min=0.0, max=100.0, step=0.01, display_name="Soften"),
                io.Float.Input("sigma", default=10.0, min=0.0, max=100.0, step=0.01, display_name="Amount"),
                io.Float.Input("threshold", default=15.0, min=0.0, max=100.0, step=0.01, display_name="Preserve Edges"),
                io.Float.Input("r_spots_blend", default=0.8, min=0.0, max=1.0, step=0.001, display_name="Dark Spots"),
                io.Float.Input("r_h_blend", default=0.5, min=0.0, max=1.0, step=0.001, display_name="Highlights"),
                io.Float.Input("strength", default=0.0, min=0.0, max=10.0, step=0.01, display_name="Restore Detail"),
                io.Float.Input("blur_h", default=0.0, min=0.0, max=50.0, step=0.01, display_name="Detail Soften"),
                io.Float.Input("blur_s", default=30.0, min=0.0, max=100.0, step=0.01, display_name="Blur Shine"),
                io.Float.Input("o_amount", default=0.2, min=0.0, max=1.0, step=0.001, display_name="Shine Amount"),
                io.Float.Input("sat_amount", default=100.0, min=0.0, max=300.0, step=0.1, display_name="Saturation"),
                io.Float.Input("hue_amount", default=0.0, min=-360.0, max=360.0, step=0.01, display_name="Hue Shift"),
            ],
            outputs=[
                io.Image.Output("image", display_name="IMAGE"),
                io.Mask.Output("mask", display_name="MASK"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    @torch.no_grad()
    def execute(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor | None = None,
        proxy_video: Any = None,
        colour: Any = "auto",
        weights: Any = "6.0, 0.0, 3.0",
        blur_m: float = 10.0,
        sigma: float = 10.0,
        threshold: float = 15.0,
        r_spots_blend: float = 0.8,
        r_h_blend: float = 0.5,
        strength: float = 0.0,
        blur_h: float = 0.0,
        blur_s: float = 30.0,
        o_amount: float = 0.2,
        sat_amount: float = 100.0,
        hue_amount: float = 0.0,
    ) -> io.NodeOutput:
        node_id = getattr(cls, "hidden", None) and getattr(cls.hidden, "unique_id", "")
        prompt = getattr(cls, "hidden", None) and getattr(cls.hidden, "prompt", None)
        frame_total = int(image.shape[0]) if isinstance(image, torch.Tensor) and image.ndim >= 1 else 1
        progress = _BeautyProgress(node_id, frame_total)
        progress.info("start", f"frames={frame_total}")
        try:
            progress.info("cache input", "storing frames and optional mask for VFX Preview")
            _cache_vfx_input(node_id, prompt, image, mask, proxy_video)
            progress.info("cache input complete")

            colour_tensor = _parse_hex_colour(colour)
            weights_tensor = _parse_vec3(weights, (6.0, 0.0, 3.0), "weights")
            progress.info("validate parameters", f"colour={'auto' if colour_tensor is None else 'fixed'}")
            if colour_tensor is None:
                progress.info("estimate colour", f"sampling up to {_COLOUR_SAMPLE_FRAMES} frames for the whole clip")
                colour_tensor = _estimate_video_colour(image, mask, progress=progress).detach().to(device="cpu", dtype=torch.float32)
            output = _run_beauty_batched(
                image,
                mask,
                colour_tensor,
                weights_tensor,
                float(blur_m),
                float(sigma),
                float(threshold),
                float(r_spots_blend),
                float(r_h_blend),
                float(strength),
                float(blur_h),
                float(blur_s),
                float(o_amount),
                float(sat_amount),
                float(hue_amount),
                progress=progress,
            )
            progress.info("outputs", "returning RGB IMAGE and skin Matte MASK")
            return io.NodeOutput(output[..., :3], output[..., 3])
        except Exception:
            progress.info("failed", "see traceback for the exception details")
            raise
        finally:
            progress.close_line()


class VFXBeautyExtension(ComfyExtension):
    @override
    async def on_load(self) -> None:
        global _VFX_ROUTE_REGISTERED
        if _VFX_ROUTE_REGISTERED:
            return
        from server import PromptServer

        server_instance = getattr(PromptServer, "instance", None)
        if server_instance is not None:
            server_instance.routes.get("/cinestyle/vfx-beauty-cache")(_vfx_beauty_cache_info_route)
            server_instance.routes.get("/cinestyle/vfx-beauty-cache-video")(_vfx_beauty_cache_video_route)
            server_instance.routes.post("/cinestyle/vfx-beauty-preview")(_vfx_beauty_preview_route)
            _VFX_ROUTE_REGISTERED = True

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CSVFXBeauty]


async def comfy_entrypoint() -> VFXBeautyExtension:
    return VFXBeautyExtension()
