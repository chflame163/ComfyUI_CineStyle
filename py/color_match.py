"""Reference based color matching for ComfyUI IMAGE batches.

The node deliberately performs a global, point-wise transform.  It never
resamples or spatially filters the input frames, so texture and edge detail
remain in the source image.  Statistics are fitted once for the whole batch
and the fitted transform is then applied in GPU-sized chunks.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - ComfyUI normally provides tqdm
    tqdm = None


_LOGGER = logging.getLogger("CineStyleColorMatch")
_CATEGORY = "😺dzNodes/CineStyle"
_ANSI_GREEN = "\033[32m"
_ANSI_RESET = "\033[0m"
_EPS = 1.0e-6
_GPU_MEMORY_RESERVE_BYTES = 512 * 1024 * 1024
_GPU_MEMORY_FRACTION = 0.40
_MAX_GPU_BATCH = 64
_CPU_BATCH = 8
_STAT_SAMPLE_PIXELS = 32768
_PDF_SAMPLE_PIXELS = 8192
_REFERENCE_STAT_MAX_DIM = 2048
_PDF_ITERATIONS = 3
_PDF_DIRECTIONS = 6

_METHODS = ("Reinhard", "LHM", "PCCM", "PDF", "Optimal Transport")
_COLOR_SPACES = ("Lab", "OKLab")


# sRGB D65 matrices.  Oklab is defined from linear sRGB, not gamma encoded RGB.
_SRGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
_XYZ_TO_SRGB = (
    (3.2404542, -1.5371385, -0.4985314),
    (-0.9692660, 1.8760108, 0.0415560),
    (0.0556434, -0.2040259, 1.0572252),
)
_D65 = (0.95047, 1.0, 1.08883)
_OKLAB_M1 = (
    (0.4122214708, 0.5363325363, 0.0514459929),
    (0.2119034982, 0.6806995451, 0.1073969566),
    (0.0883024619, 0.2817188376, 0.6299787005),
)
_OKLAB_M2 = (
    (0.2104542553, 0.7936177850, -0.0040720468),
    (1.9779984951, -2.4285922050, 0.4505937099),
    (0.0259040371, 0.7827717662, -0.8086757660),
)
_OKLAB_M1_INV = (
    (4.0767416621, -3.3077115913, 0.2309699292),
    (-1.2684380046, 2.6097574011, -0.3413193965),
    (-0.0041960863, -0.7034186147, 1.7076147010),
)
_OKLAB_M2_INV = (
    (1.0, 0.3963377774, 0.2158037573),
    (1.0, -0.1055613458, -0.0638541728),
    (1.0, -0.0894841775, -1.2914855480),
)


@dataclass
class _TransferModel:
    method: str
    mean_source: torch.Tensor
    mean_target: torch.Tensor
    matrix: torch.Tensor | None = None
    pdf_steps: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...] = ()
    candidate_l_mean: torch.Tensor | None = None
    candidate_l_std: torch.Tensor | None = None
    source_l_mean: torch.Tensor | None = None
    source_l_std: torch.Tensor | None = None


def _match_info(message: str, *args: Any) -> None:
    """Keep CS Color Match status lines aligned with the video nodes."""
    _LOGGER.info("[CS Color Match] " + message, *args)


class _MatchProgress:
    """Emit a throttled tqdm-style progress bar for frame rendering."""

    def __init__(self, total: int, description: str = "rendering frames"):
        self.bar = None
        if tqdm is not None:
            self.bar = tqdm(
                total=max(1, int(total)),
                desc=f"{_ANSI_GREEN}[INFO]{_ANSI_RESET} [CS Color Match] {description}",
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


def _matrix(values: tuple[tuple[float, ...], ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(values, device=device, dtype=dtype)


def _as_rgb(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a ComfyUI IMAGE tensor.")
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.shape[-1] < 3:
        raise ValueError(f"{name} must have shape [batch, height, width, 3 or 4].")
    if any(int(size) <= 0 for size in value.shape[:3]):
        raise ValueError(f"{name} must contain non-empty frames.")
    if not torch.isfinite(value[..., :3]).all().item():
        raise ValueError(f"{name} contains non-finite pixel values.")
    return value[..., :3].float().clamp(0.0, 1.0)


def _first_image(value: torch.Tensor, name: str) -> torch.Tensor:
    """Select the first reference frame before validating/converting the batch."""
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a ComfyUI IMAGE tensor.")
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.shape[-1] < 3:
        raise ValueError(f"{name} must have shape [batch, height, width, 3 or 4].")
    if int(value.shape[0]) <= 0 or any(int(size) <= 0 for size in value.shape[1:3]):
        raise ValueError(f"{name} must contain a non-empty image.")
    return value[:1]


def _preferred_device(fallback: torch.device) -> torch.device:
    try:
        import comfy.model_management as model_management

        device = model_management.get_torch_device()
        return device if isinstance(device, torch.device) else torch.device(device)
    except Exception:
        return fallback


def _batch_size(image: torch.Tensor, device: torch.device, method: str) -> int:
    total = int(image.shape[0])
    if total <= 1:
        return 1
    height, width = int(image.shape[1]), int(image.shape[2])
    method_factor = 26 if method == "PDF" else 16
    estimated_per_frame = max(1, height * width * 3 * 4 * method_factor)
    if device.type == "cuda":
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
            budget = max(
                estimated_per_frame,
                int((free_bytes - _GPU_MEMORY_RESERVE_BYTES) * _GPU_MEMORY_FRACTION),
            )
            return max(1, min(total, _MAX_GPU_BATCH, budget // estimated_per_frame))
        except (RuntimeError, AttributeError, TypeError):
            pass
    return max(1, min(total, _CPU_BATCH))


def _srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    rgb = rgb.clamp(0.0, 1.0)
    return torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055).clamp_min(0.0).pow(2.4),
    )


def _linear_to_srgb(rgb: torch.Tensor) -> torch.Tensor:
    rgb = rgb.clamp_min(0.0)
    return torch.where(
        rgb <= 0.0031308,
        rgb * 12.92,
        1.055 * rgb.clamp_min(0.0).pow(1.0 / 2.4) - 0.055,
    )


def _rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    device, dtype = rgb.device, rgb.dtype
    xyz = _srgb_to_linear(rgb) @ _matrix(_SRGB_TO_XYZ, device, dtype).T
    xyz = xyz / _matrix((_D65,), device, dtype)
    delta = 6.0 / 29.0
    threshold = delta**3
    positive_xyz = xyz.clamp_min(0.0)
    f = torch.where(
        xyz > threshold,
        positive_xyz.pow(1.0 / 3.0),
        xyz / (3.0 * delta * delta) + 4.0 / 29.0,
    )
    fx, fy, fz = f.unbind(dim=-1)
    return torch.stack((116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)), dim=-1)


def _lab_to_rgb(values: torch.Tensor) -> torch.Tensor:
    device, dtype = values.device, values.dtype
    L, a, b = values.unbind(dim=-1)
    delta = 6.0 / 29.0
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    def inverse_f(value: torch.Tensor) -> torch.Tensor:
        return torch.where(value > delta, value.pow(3), (value - 16.0 / 116.0) * 3.0 * delta * delta)

    xyz = torch.stack((inverse_f(fx), inverse_f(fy), inverse_f(fz)), dim=-1)
    xyz = xyz * _matrix((_D65,), device, dtype)
    linear = xyz @ _matrix(_XYZ_TO_SRGB, device, dtype).T
    return _linear_to_srgb(linear)


def _rgb_to_oklab(rgb: torch.Tensor) -> torch.Tensor:
    device, dtype = rgb.device, rgb.dtype
    linear = _srgb_to_linear(rgb)
    lms = linear @ _matrix(_OKLAB_M1, device, dtype).T
    lms = lms.sign() * lms.abs().pow(1.0 / 3.0)
    return lms @ _matrix(_OKLAB_M2, device, dtype).T


def _oklab_to_rgb(values: torch.Tensor) -> torch.Tensor:
    device, dtype = values.device, values.dtype
    lms_cbrt = values @ _matrix(_OKLAB_M2_INV, device, dtype).T
    lms = lms_cbrt.pow(3)
    linear = lms @ _matrix(_OKLAB_M1_INV, device, dtype).T
    return _linear_to_srgb(linear)


def _to_space(rgb: torch.Tensor, color_space: str) -> torch.Tensor:
    if color_space == "Lab":
        return _rgb_to_lab(rgb)
    # OKLab statistics are fitted in Cartesian a/b coordinates; polar hue and
    # chroma controls are applied after the transfer.
    return _rgb_to_oklab(rgb)


def _from_space(values: torch.Tensor, color_space: str) -> torch.Tensor:
    if color_space == "Lab":
        return _lab_to_rgb(values)
    return _oklab_to_rgb(values)


def _sample_pixels(image: torch.Tensor, max_pixels: int) -> torch.Tensor:
    total, height, width = map(int, image.shape[:3])
    frame_count = min(total, 24)
    frame_indices = torch.linspace(0, total - 1, frame_count, device=image.device).round().long()
    spatial_stride = max(1, int(math.ceil(math.sqrt((frame_count * height * width) / max_pixels))))
    selected = image.index_select(0, frame_indices)[:, ::spatial_stride, ::spatial_stride, :3]
    points = selected.reshape(-1, 3)
    if points.shape[0] > max_pixels:
        indices = torch.linspace(0, points.shape[0] - 1, max_pixels, device=points.device).round().long()
        points = points.index_select(0, indices)
    return points.contiguous()


def _resize_reference_for_stats(image: torch.Tensor) -> torch.Tensor:
    """Downscale only the reference copy used for statistics."""
    height, width = int(image.shape[1]), int(image.shape[2])
    longest = max(height, width)
    if longest <= _REFERENCE_STAT_MAX_DIM:
        return image
    scale = _REFERENCE_STAT_MAX_DIM / float(longest)
    resized_height = max(1, int(round(height * scale)))
    resized_width = max(1, int(round(width * scale)))
    resized = F.interpolate(
        image.movedim(-1, 1),
        size=(resized_height, resized_width),
        mode="area",
    )
    return resized.movedim(1, -1).contiguous()


def _covariance(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = points.mean(dim=0)
    centered = points - mean
    divisor = max(1, int(points.shape[0]) - 1)
    covariance = centered.T @ centered / float(divisor)
    scale = covariance.diag().mean().clamp_min(_EPS)
    covariance = (covariance + covariance.T) * 0.5
    covariance = covariance + torch.eye(3, device=points.device, dtype=points.dtype) * (scale * 1.0e-5 + _EPS)
    return mean, covariance


def _eigh_canonical(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values, vectors = torch.linalg.eigh(matrix)
    # Eigenvector signs are arbitrary.  A deterministic sign convention keeps
    # PCCM stable when a covariance matrix is nearly unchanged.
    for column in range(vectors.shape[-1]):
        vector = vectors[:, column]
        pivot = int(vector.abs().argmax().item())
        if vector[pivot].item() < 0.0:
            vectors[:, column] = -vector
    return values.clamp_min(_EPS), vectors


def _matrix_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    values, vectors = _eigh_canonical(matrix)
    return (vectors * values.sqrt()) @ vectors.T


def _matrix_invsqrt(matrix: torch.Tensor) -> torch.Tensor:
    values, vectors = _eigh_canonical(matrix)
    return (vectors * values.rsqrt()) @ vectors.T


def _fit_affine(method: str, source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source_mean, source_cov = _covariance(source)
    target_mean, target_cov = _covariance(target)
    if method == "Reinhard":
        source_std = source_cov.diag().sqrt()
        target_std = target_cov.diag().sqrt()
        matrix = torch.diag(target_std / source_std.clamp_min(_EPS))
    elif method == "LHM":
        matrix = _matrix_sqrt(target_cov) @ _matrix_invsqrt(source_cov)
    elif method == "PCCM":
        source_values, source_vectors = _eigh_canonical(source_cov)
        target_values, target_vectors = _eigh_canonical(target_cov)
        matrix = (
            target_vectors
            * target_values.sqrt()
        ) @ torch.diag(source_values.rsqrt()) @ source_vectors.T
    elif method == "Optimal Transport":
        source_sqrt = _matrix_sqrt(source_cov)
        source_invsqrt = _matrix_invsqrt(source_cov)
        middle = source_sqrt @ target_cov @ source_sqrt
        matrix = source_invsqrt @ _matrix_sqrt(middle) @ source_invsqrt
    else:
        raise ValueError(f"Unsupported affine color transfer method: {method}")
    matrix = torch.where(torch.isfinite(matrix), matrix, torch.eye(3, device=matrix.device, dtype=matrix.dtype))
    return source_mean, target_mean, matrix, target_cov


def _quantile_map(values: torch.Tensor, source_sorted: torch.Tensor, target_sorted: torch.Tensor) -> torch.Tensor:
    if source_sorted.numel() <= 1 or target_sorted.numel() <= 1:
        return values + (target_sorted.mean() - source_sorted.mean())
    positions = torch.searchsorted(source_sorted, values.contiguous(), right=True)
    lower = (positions - 1).clamp(0, source_sorted.numel() - 1)
    upper = positions.clamp(0, source_sorted.numel() - 1)
    lower_value = source_sorted[lower]
    upper_value = source_sorted[upper]
    fraction = ((values - lower_value) / (upper_value - lower_value).clamp_min(_EPS)).clamp(0.0, 1.0)
    source_quantile = (lower.to(values.dtype) + fraction) / float(source_sorted.numel() - 1)
    target_position = source_quantile * float(target_sorted.numel() - 1)
    target_lower = target_position.floor().long().clamp(0, target_sorted.numel() - 1)
    target_upper = target_position.ceil().long().clamp(0, target_sorted.numel() - 1)
    target_fraction = target_position - target_lower.to(values.dtype)
    return torch.lerp(target_sorted[target_lower], target_sorted[target_upper], target_fraction)


def _make_directions(dim: int, count: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(137)
    random = torch.randn((count, dim), generator=generator, dtype=torch.float32)
    random = random / random.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    axes = torch.eye(dim, dtype=torch.float32)
    directions = torch.cat((axes, random), dim=0).to(device=device, dtype=dtype)
    return directions / directions.norm(dim=-1, keepdim=True).clamp_min(_EPS)


def _fit_pdf(source: torch.Tensor, target: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
    current = source.clone()
    directions = _make_directions(3, _PDF_DIRECTIONS, source.device, source.dtype)
    steps: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for _ in range(_PDF_ITERATIONS):
        for direction in directions:
            source_projection = current @ direction
            target_projection = target @ direction
            source_sorted = torch.sort(source_projection).values
            target_sorted = torch.sort(target_projection).values
            mapped = _quantile_map(source_projection, source_sorted, target_sorted)
            current = current + (mapped - source_projection).unsqueeze(-1) * direction
            steps.append((direction, source_sorted, target_sorted))
    return tuple(steps)


def _apply_model(points: torch.Tensor, model: _TransferModel) -> torch.Tensor:
    flat = points.reshape(-1, 3)
    if model.method == "PDF":
        result = flat
        for direction, source_sorted, target_sorted in model.pdf_steps:
            projection = result @ direction
            mapped = _quantile_map(projection, source_sorted, target_sorted)
            result = result + (mapped - projection).unsqueeze(-1) * direction
        return result.reshape_as(points)
    if model.matrix is None:
        return points
    return ((flat - model.mean_source) @ model.matrix.T + model.mean_target).reshape_as(points)


def _fit_model(source: torch.Tensor, target: torch.Tensor, method: str) -> _TransferModel:
    sample_limit = _PDF_SAMPLE_PIXELS if method == "PDF" else _STAT_SAMPLE_PIXELS
    if source.shape[0] > sample_limit:
        source = source[:sample_limit]
    if target.shape[0] > sample_limit:
        target = target[:sample_limit]
    source_mean, source_cov = _covariance(source)
    target_mean, _ = _covariance(target)
    if method == "PDF":
        steps = _fit_pdf(source, target)
        mapped_sample = _apply_model(source, _TransferModel(method, source_mean, target_mean, pdf_steps=steps))
        matrix = None
    else:
        source_mean, target_mean, matrix, _ = _fit_affine(method, source, target)
        mapped_sample = _apply_model(source, _TransferModel(method, source_mean, target_mean, matrix=matrix))
        steps = ()
    source_l = source[:, 0]
    mapped_l = mapped_sample[:, 0]
    strength_source = source_l.mean()
    strength_std = source_l.std(unbiased=False).clamp_min(_EPS)
    candidate_l_mean = mapped_l.mean()
    candidate_l_std = mapped_l.std(unbiased=False).clamp_min(_EPS)
    return _TransferModel(
        method=method,
        mean_source=source_mean,
        mean_target=target_mean,
        matrix=matrix,
        pdf_steps=steps,
        candidate_l_mean=candidate_l_mean,
        candidate_l_std=candidate_l_std,
        source_l_mean=strength_source,
        source_l_std=strength_std,
    )


def _gamut_map(values: torch.Tensor, color_space: str) -> torch.Tensor:
    values = values.clone()
    if color_space == "Lab":
        values[..., 0] = values[..., 0].clamp(0.0, 100.0)
    else:
        values[..., 0] = values[..., 0].clamp(0.0, 1.0)
    linear = _space_to_linear(values, color_space)
    bad = (~torch.isfinite(linear).all(dim=-1)) | (linear < 0.0).any(dim=-1) | (linear > 1.0).any(dim=-1)
    if bad.any().item():
        flat = values.reshape(-1, 3)
        bad_indices = bad.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        selected = flat.index_select(0, bad_indices)
        chroma = selected[:, 1:].norm(dim=-1)
        unit = selected[:, 1:] / chroma.unsqueeze(-1).clamp_min(_EPS)
        low = torch.zeros_like(chroma)
        high = chroma
        for _ in range(9):
            middle = (low + high) * 0.5
            trial = torch.cat((selected[:, :1], unit * middle.unsqueeze(-1)), dim=-1)
            trial_linear = _space_to_linear(trial, color_space)
            inside = torch.isfinite(trial_linear).all(dim=-1) & (trial_linear >= 0.0).all(dim=-1) & (trial_linear <= 1.0).all(dim=-1)
            low = torch.where(inside, middle, low)
            high = torch.where(inside, high, middle)
        selected[:, 1:] = unit * low.unsqueeze(-1)
        flat.index_copy_(0, bad_indices, selected)
        values = flat.reshape_as(values)
    result = _linear_to_srgb(_space_to_linear(values, color_space))
    return torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _space_to_linear(values: torch.Tensor, color_space: str) -> torch.Tensor:
    if color_space == "Lab":
        device, dtype = values.device, values.dtype
        L, a, b = values.unbind(dim=-1)
        delta = 6.0 / 29.0
        fy = (L + 16.0) / 116.0
        fx = fy + a / 500.0
        fz = fy - b / 200.0

        def inverse_f(value: torch.Tensor) -> torch.Tensor:
            return torch.where(value > delta, value.pow(3), (value - 16.0 / 116.0) * 3.0 * delta * delta)

        xyz = torch.stack((inverse_f(fx), inverse_f(fy), inverse_f(fz)), dim=-1)
        xyz = xyz * _matrix((_D65,), device, dtype)
        return xyz @ _matrix(_XYZ_TO_SRGB, device, dtype).T
    device, dtype = values.device, values.dtype
    lms_cbrt = values @ _matrix(_OKLAB_M2_INV, device, dtype).T
    lms = lms_cbrt.pow(3)
    return lms @ _matrix(_OKLAB_M1_INV, device, dtype).T


def _apply_controls(
    source: torch.Tensor,
    mapped: torch.Tensor,
    model: _TransferModel,
    chroma_floor: float,
    match_strength: float,
    preserve_luminance: float,
    preserve_contrast: float,
    preserve_saturation: float,
    hue_strength: float,
    chroma_strength: float,
) -> torch.Tensor:
    source_l = source[..., 0]
    target_l = mapped[..., 0]
    if model.candidate_l_mean is not None and model.candidate_l_std is not None:
        contrast_l = (
            (target_l - model.candidate_l_mean)
            * model.source_l_std.clamp_min(_EPS)
            / model.candidate_l_std.clamp_min(_EPS)
            + model.source_l_mean
        )
        target_l = torch.lerp(target_l, contrast_l, float(preserve_contrast))
    # Match Strength is the final source-to-target interpolation for L/h/C.
    output_l = torch.lerp(source_l, target_l, float(match_strength))
    output_l = torch.lerp(output_l, source_l, float(preserve_luminance))

    source_ab = source[..., 1:]
    mapped_ab = mapped[..., 1:]
    source_chroma = source_ab.norm(dim=-1)
    mapped_chroma = mapped_ab.norm(dim=-1)
    source_hue = torch.atan2(source_ab[..., 1], source_ab[..., 0])
    mapped_hue = torch.atan2(mapped_ab[..., 1], mapped_ab[..., 0])
    hue_delta = torch.atan2(torch.sin(mapped_hue - source_hue), torch.cos(mapped_hue - source_hue))
    chroma_gate = (source_chroma / max(_EPS, chroma_floor)).clamp(0.0, 1.0)
    hue_alpha = float(match_strength) * float(hue_strength) * chroma_gate
    output_hue = source_hue + hue_delta * hue_alpha
    target_chroma = source_chroma + float(chroma_strength) * (mapped_chroma - source_chroma)
    output_chroma = source_chroma + float(match_strength) * (target_chroma - source_chroma)
    output_chroma = torch.lerp(output_chroma, source_chroma, float(preserve_saturation)).clamp_min(0.0)
    output_ab = torch.stack((output_chroma * output_hue.cos(), output_chroma * output_hue.sin()), dim=-1)
    return torch.cat((output_l.unsqueeze(-1), output_ab), dim=-1)


@torch.no_grad()
def _render(
    image: torch.Tensor,
    reference_image: torch.Tensor,
    method: str,
    color_space: str,
    match_strength: float,
    preserve_luminance: float,
    preserve_contrast: float,
    preserve_saturation: float,
    hue_strength: float,
    chroma_strength: float,
    progress: _MatchProgress | None = None,
) -> torch.Tensor:
    _match_info("stage 1/5: validating input tensors")
    rgb = _as_rgb(image, "image")
    reference = _as_rgb(_first_image(reference_image, "reference_image"), "reference_image")
    _match_info(
        "source ready: frames=%d; size=%dx%d; reference size=%dx%d",
        int(rgb.shape[0]),
        int(rgb.shape[2]),
        int(rgb.shape[1]),
        int(reference.shape[2]),
        int(reference.shape[1]),
    )

    _match_info("stage 2/5: preparing reference statistics")
    reference_stats = _resize_reference_for_stats(reference)
    _match_info(
        "reference statistics input: original=%dx%d; sampled=%dx%d; max_pixels=%d",
        int(reference.shape[2]),
        int(reference.shape[1]),
        int(reference_stats.shape[2]),
        int(reference_stats.shape[1]),
        _PDF_SAMPLE_PIXELS if method == "PDF" else _STAT_SAMPLE_PIXELS,
    )
    total, height, width = map(int, rgb.shape[:3])
    device = _preferred_device(rgb.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = rgb.device
    if match_strength <= _EPS:
        _match_info("stage 3/5: match strength is zero; transform fitting skipped")
        _match_info("stage 4/5: rendering skipped; returning source frames")
        if progress is not None:
            progress.update(total)
        return rgb.detach().to(device="cpu", dtype=torch.float32).contiguous()

    sample_limit = _PDF_SAMPLE_PIXELS if method == "PDF" else _STAT_SAMPLE_PIXELS
    source_sample_rgb = _sample_pixels(rgb, sample_limit).to(device=device, dtype=torch.float32)
    target_sample_rgb = _sample_pixels(reference_stats, sample_limit).to(device=device, dtype=torch.float32)
    source_sample = _to_space(source_sample_rgb, color_space).reshape(-1, 3)
    target_sample = _to_space(target_sample_rgb, color_space).reshape(-1, 3)
    _match_info(
        "stage 3/5: fitting %s transform in %s; source_samples=%d; reference_samples=%d",
        method,
        color_space,
        int(source_sample.shape[0]),
        int(target_sample.shape[0]),
    )
    model = _fit_model(source_sample, target_sample, method)
    batch_size = _batch_size(rgb, device, method)
    _match_info(
        "render setup: method=%s; color_space=%s; frames=%d; frame_batch=%d; device=%s",
        method,
        color_space,
        total,
        batch_size,
        device,
    )

    _match_info("stage 4/5: rendering frames")
    output = torch.empty((total, height, width, 3), device="cpu", dtype=torch.float32)
    start = 0
    while start < total:
        end = min(total, start + batch_size)
        source_chunk: torch.Tensor | None = None
        try:
            source_chunk = rgb[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
            source_space = _to_space(source_chunk, color_space)
            mapped_space = _apply_model(source_space, model)
            controlled_space = _apply_controls(
                source_space,
                mapped_space,
                model,
                2.0 if color_space == "Lab" else 0.02,
                match_strength,
                preserve_luminance,
                preserve_contrast,
                preserve_saturation,
                hue_strength,
                chroma_strength,
            )
            rendered = _gamut_map(controlled_space, color_space)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or batch_size <= 1:
                raise
            if source_chunk is not None:
                del source_chunk
            if device.type == "cuda":
                torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            _LOGGER.warning("[CS Color Match] CUDA OOM; retrying with batch=%d", batch_size)
            continue
        output[start:end].copy_(rendered.to(device="cpu", dtype=torch.float32), non_blocking=True)
        processed = end - start
        del source_chunk, source_space, mapped_space, controlled_space, rendered
        start = end
        if progress is not None:
            progress.update(processed)
    return output


def _parameter(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return parsed


class CSColorMatch(io.ComfyNode):
    """Transfer a reference image color style to an IMAGE frame batch."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CS_Color_Match",
            display_name="CS Color Match",
            category=_CATEGORY,
            essentials_category="Image Effects",
            search_aliases=["color match", "colour match", "reference color", "video color transfer"],
            description=(
                "Match an IMAGE batch to the color style of a single reference image "
                "while preserving source luminance, contrast, saturation, and detail."
            ),
            inputs=[
                io.Image.Input("image", tooltip="Source IMAGE batch; a batch is processed in its original frame order."),
                io.Image.Input("reference_image", tooltip="Reference IMAGE; only the first image is used."),
                io.Combo.Input("method", options=list(_METHODS), default="Optimal Transport", tooltip="Global color transfer method."),
                io.Combo.Input("color_space", display_name="Color Space", options=list(_COLOR_SPACES), default="OKLab", tooltip="Color space used for statistics and transfer."),
                io.Float.Input("match_strength", display_name="Match Strength", default=0.75, min=0.0, max=1.0, step=0.01, tooltip="Overall interpolation from the source to the matched lightness, hue, and chroma target."),
                io.Float.Input("preserve_luminance", display_name="Preserve Luminance", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="0 to 1, where 1 keeps source perceptual lightness."),
                io.Float.Input("preserve_contrast", display_name="Preserve Contrast", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="0 to 1, where 1 restores source luminance contrast."),
                io.Float.Input("preserve_saturation", display_name="Preserve Saturation", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="0 to 1, where 1 keeps source chroma."),
                io.Float.Input("hue_strength", display_name="Hue Strength", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("chroma_strength", display_name="Chroma Strength", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Image.Output("image", display_name="IMAGE")],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    @torch.no_grad()
    def execute(
        cls,
        image: torch.Tensor,
        reference_image: torch.Tensor,
        method: str = "Optimal Transport",
        color_space: str = "OKLab",
        match_strength: float = 0.75,
        preserve_luminance: float = 1.0,
        preserve_contrast: float = 1.0,
        preserve_saturation: float = 0.0,
        hue_strength: float = 1.0,
        chroma_strength: float = 1.0,
    ) -> io.NodeOutput:
        started_at = time.perf_counter()
        if method not in _METHODS:
            raise ValueError(f"method must be one of: {', '.join(_METHODS)}")
        if color_space not in _COLOR_SPACES:
            raise ValueError(f"color_space must be one of: {', '.join(_COLOR_SPACES)}")
        parameters = {
            "match_strength": _parameter(match_strength, "match_strength"),
            "preserve_luminance": _parameter(preserve_luminance, "preserve_luminance"),
            "preserve_contrast": _parameter(preserve_contrast, "preserve_contrast"),
            "preserve_saturation": _parameter(preserve_saturation, "preserve_saturation"),
            "hue_strength": _parameter(hue_strength, "hue_strength"),
            "chroma_strength": _parameter(chroma_strength, "chroma_strength"),
        }
        _match_info("start: method=%s; color_space=%s", method, color_space)
        progress_total = int(image.shape[0]) if isinstance(image, torch.Tensor) and image.ndim >= 1 else 1
        progress = _MatchProgress(progress_total)
        try:
            output = _render(image, reference_image, method, color_space, progress=progress, **parameters)
        finally:
            progress.close()
        _match_info("stage 5/5: complete; frames=%d; elapsed=%.2fs", int(output.shape[0]), time.perf_counter() - started_at)
        return io.NodeOutput(output)


class ColorMatchExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CSColorMatch]


async def comfy_entrypoint() -> ColorMatchExtension:
    return ColorMatchExtension()
