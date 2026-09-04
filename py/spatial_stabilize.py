"""Mask-driven spatial stabilization and reversible local restoration."""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn.functional as F
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - ComfyUI normally provides tqdm
    tqdm = None


_LOGGER = logging.getLogger(__name__)
_CATEGORY = "😺dzNodes/CineStyle/Video"
_DATA_TYPE = "CS_SPATIAL_STABLE_DATA"
_DATA_VERSION = 1
_BBOX_THRESHOLD = 0.5
_ANSI_GREEN = "\033[32m"
_ANSI_RESET = "\033[0m"


SPATIAL_STABLE_DATA = io.Custom(_DATA_TYPE)


def _spatial_info(node_name: str, message: str, *args: Any) -> None:
    """Keep status lines consistent with the other CineStyle video nodes."""
    _LOGGER.info(f"[{node_name}] {message}", *args)


class _SpatialProgress:
    """Emit a console-friendly tqdm frame progress bar when available."""

    def __init__(self, node_name: str, total: int, description: str):
        self.bar = None
        if tqdm is not None:
            self.bar = tqdm(
                total=max(1, int(total)),
                desc=f"{_ANSI_GREEN}[INFO]{_ANSI_RESET} [{node_name}] {description}",
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


def _normalise_image(image: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise ValueError(f"{name} must be a ComfyUI IMAGE tensor.")
    if image.ndim != 4:
        raise ValueError(f"{name} must have shape [batch, height, width, channels].")
    if image.shape[0] < 1 or image.shape[1] < 1 or image.shape[2] < 1:
        raise ValueError(f"{name} dimensions must be non-empty.")
    if image.shape[-1] < 3:
        raise ValueError(f"{name} must contain at least three colour channels.")
    if not bool(torch.isfinite(image).all().item()):
        raise ValueError(f"{name} must contain only finite values.")
    return image[..., :3]


def _normalise_mask_shape(mask: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise ValueError(f"{name} must be a ComfyUI MASK tensor.")
    if mask.ndim == 2:
        value = mask.unsqueeze(0)
    elif mask.ndim == 3:
        value = mask
    elif mask.ndim == 4 and mask.shape[1] == 1:
        value = mask[:, 0]
    elif mask.ndim == 4 and mask.shape[-1] == 1:
        value = mask[..., 0]
    else:
        raise ValueError(
            f"{name} must have shape [H,W], [B,H,W], [B,1,H,W], or [B,H,W,1]."
        )
    if value.shape[-2] < 1 or value.shape[-1] < 1:
        raise ValueError(f"{name} dimensions must be non-empty.")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values.")
    return value


def _prepare_stabilize_mask(
    mask: torch.Tensor,
    frame_count: int,
    height: int,
    width: int,
) -> torch.Tensor:
    value = _normalise_mask_shape(mask, "mask")
    if tuple(value.shape[-2:]) != (height, width):
        raise ValueError(
            f"mask spatial size must match image: expected {width}x{height}, "
            f"received {int(value.shape[-1])}x{int(value.shape[-2])}."
        )
    return value[:frame_count]


def _prepare_restore_mask(
    mask: torch.Tensor,
    frame_count: int,
    height: int,
    width: int,
) -> torch.Tensor:
    value = _normalise_mask_shape(mask, "mask")
    if int(value.shape[0]) != frame_count:
        raise ValueError(
            f"restore mask batch must contain {frame_count} frames; received {int(value.shape[0])}."
        )
    if tuple(value.shape[-2:]) != (height, width):
        raise ValueError(
            f"restore mask size must be {width}x{height}; "
            f"received {int(value.shape[-1])}x{int(value.shape[-2])}."
        )
    return value


def _compute_device(source_device: torch.device) -> torch.device:
    try:
        import comfy.model_management as model_management

        device = torch.device(model_management.get_torch_device())
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
        device = source_device if source_device.type != "cpu" else torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def _is_cuda_oom(exc: RuntimeError) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _mask_frame(
    mask: torch.Tensor,
    index: int,
    height: int,
    width: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    if index < int(mask.shape[0]):
        value = mask[index]
        if device is not None:
            value = value.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        else:
            value = value.to(dtype=torch.float32)
        return value.clamp(0.0, 1.0)
    return torch.zeros((height, width), device=device or torch.device("cpu"), dtype=torch.float32)


def _gaussian_kernel_1d(
    sigma: float,
    radius: int,
    device: torch.device,
) -> torch.Tensor | None:
    if sigma <= 0.0 or radius <= 0:
        return None
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (coordinates / sigma).square())
    return kernel / kernel.sum()


def _blur_mask_for_bbox(
    mask: torch.Tensor,
    horizontal_kernel: torch.Tensor | None,
    vertical_kernel: torch.Tensor | None,
) -> torch.Tensor:
    if horizontal_kernel is None and vertical_kernel is None:
        return mask
    value = mask.to(dtype=torch.float32).clamp(0.0, 1.0).unsqueeze(0).unsqueeze(0)
    if horizontal_kernel is not None:
        radius = int(horizontal_kernel.numel() // 2)
        value = F.conv2d(
            F.pad(value, (radius, radius, 0, 0), mode="replicate"),
            horizontal_kernel.view(1, 1, 1, -1),
        )
    if vertical_kernel is not None:
        radius = int(vertical_kernel.numel() // 2)
        value = F.conv2d(
            F.pad(value, (0, 0, radius, radius), mode="replicate"),
            vertical_kernel.view(1, 1, -1, 1),
        )
    return value[0, 0]


def _bbox_observations(
    mask: torch.Tensor,
    frame_count: int,
    height: int,
    width: int,
    mask_blur_sigma: float,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    sigma = float(mask_blur_sigma)
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError("mask_blur_sigma must be a finite value greater than or equal to zero.")
    radius = int(math.ceil(3.0 * sigma)) if sigma > 0.0 else 0
    horizontal_kernel = _gaussian_kernel_1d(sigma, min(radius, max(0, width - 1)), mask.device)
    vertical_kernel = _gaussian_kernel_1d(sigma, min(radius, max(0, height - 1)), mask.device)
    progress = _SpatialProgress("CS Spatial Stabilize", frame_count, "analysing mask frames")
    try:
        for index in range(frame_count):
            if index >= int(mask.shape[0]):
                observations.append({"status": "empty", "bbox": None})
                progress.update()
                continue

            bbox_mask = _blur_mask_for_bbox(
                mask[index],
                horizontal_kernel,
                vertical_kernel,
            )
            foreground = bbox_mask > _BBOX_THRESHOLD
            if not bool(foreground.any().item()):
                observations.append({"status": "empty", "bbox": None})
                progress.update()
                continue

            rows = torch.where(foreground.any(dim=1))[0]
            columns = torch.where(foreground.any(dim=0))[0]
            y_min = int(rows[0].item())
            y_max = int(rows[-1].item())
            x_min = int(columns[0].item())
            x_max = int(columns[-1].item())
            box_width = x_max - x_min + 1
            box_height = y_max - y_min + 1
            touches_edge = (
                x_min == 0 or y_min == 0 or x_max == width - 1 or y_max == height - 1
            )
            observations.append(
                {
                    "status": "out_of_frame" if touches_edge else "valid",
                    "bbox": (x_min, y_min, x_max, y_max),
                    "center_x": (x_min + x_max) * 0.5,
                    "center_y": (y_min + y_max) * 0.5,
                    "width": box_width,
                    "height": box_height,
                    "area": box_width * box_height,
                }
            )
            progress.update()
    finally:
        progress.close()
    return observations


def _fill_time_series(
    values: list[float | None],
) -> list[float]:
    known = [index for index, value in enumerate(values) if value is not None]
    if not known:
        raise ValueError("Cannot interpolate a time series without any observations.")

    transformed = [None if value is None else float(value) for value in values]
    result = [0.0] * len(values)
    first = known[0]
    last = known[-1]
    first_value = float(transformed[first])
    last_value = float(transformed[last])
    for index in range(0, first + 1):
        result[index] = first_value
    for index in range(last, len(values)):
        result[index] = last_value

    for left, right in zip(known, known[1:]):
        left_value = float(transformed[left])
        right_value = float(transformed[right])
        span = right - left
        for index in range(left, right + 1):
            amount = (index - left) / span
            result[index] = left_value + (right_value - left_value) * amount

    return result


def _moving_average(values: list[float], window: int) -> list[float]:
    """Return a centred moving average with continuously constrained endpoints."""
    window = max(1, int(window))
    if window == 1 or len(values) <= 1:
        return list(values)

    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + float(value))

    left_radius = (window - 1) // 2
    right_radius = window // 2
    result: list[float] = []
    for index in range(len(values)):
        start = max(0, index - left_radius)
        end = min(len(values), index + right_radius + 1)
        result.append((prefix[end] - prefix[start]) / (end - start))
    # Constrain both endpoints without replacing them abruptly. The correction
    # is distributed linearly through the sequence so adjacent frames remain
    # continuous while the first and last solved transforms stay exact.
    first_correction = float(values[0]) - result[0]
    last_correction = float(values[-1]) - result[-1]
    last_index = len(values) - 1
    return [
        value
        + first_correction * (1.0 - index / last_index)
        + last_correction * (index / last_index)
        for index, value in enumerate(result)
    ]


def _ceil_multiple(value: float, multiple: int) -> int:
    multiple = max(1, int(multiple))
    return max(multiple, int(math.ceil(float(value) / multiple)) * multiple)


def _analyse_stabilization(
    observations: list[dict[str, Any]],
    multiple: int,
    crop_margin_percent: float,
    average_frames: int,
    mask_blur_sigma: float,
    source_width: int,
    source_height: int,
) -> dict[str, Any]:
    positioned = [item for item in observations if item["status"] != "empty"]
    valid = [(index, item) for index, item in enumerate(observations) if item["status"] == "valid"]
    if not positioned:
        raise ValueError("All mask frames are empty; spatial stabilization needs at least one non-empty mask.")
    if not valid:
        raise ValueError(
            "No fully visible mask frame is available; at least one mask must not touch any image edge "
            "to establish the anchor scale."
        )

    anchor_index, anchor = max(valid, key=lambda pair: int(pair[1]["area"]))
    anchor_area = float(anchor["area"])

    center_x = _fill_time_series(
        [None if item["status"] == "empty" else float(item["center_x"]) for item in observations]
    )
    center_y = _fill_time_series(
        [None if item["status"] == "empty" else float(item["center_y"]) for item in observations]
    )
    scale_observations: list[float | None] = []
    for item in observations:
        if item["status"] == "valid":
            scale_observations.append(math.sqrt(anchor_area / float(item["area"])))
        else:
            scale_observations.append(None)
    scales = _fill_time_series(scale_observations)
    # Keep the crop envelope tied to the unsmoothed stable BBoxes. The moving
    # average is a temporal de-jitter operation and must not change output size.
    crop_scales = list(scales)
    average_frames = max(1, int(average_frames))
    center_x = _moving_average(center_x, average_frames)
    center_y = _moving_average(center_y, average_frames)
    scales = _moving_average(scales, average_frames)

    stable_widths = [
        float(item["width"]) * crop_scales[index]
        for index, item in enumerate(observations)
        if item["status"] != "empty"
    ]
    stable_heights = [
        float(item["height"]) * crop_scales[index]
        for index, item in enumerate(observations)
        if item["status"] != "empty"
    ]
    maximum_width = max(stable_widths)
    maximum_height = max(stable_heights)
    margin = max(0.0, float(crop_margin_percent)) / 100.0
    crop_width = _ceil_multiple(
        maximum_width + 2.0 * multiple + 2.0 * maximum_width * margin,
        multiple,
    )
    crop_height = _ceil_multiple(
        maximum_height + 2.0 * multiple + 2.0 * maximum_height * margin,
        multiple,
    )
    crop_center_x = (crop_width - 1) * 0.5
    crop_center_y = (crop_height - 1) * 0.5

    forward_matrices = torch.empty((len(observations), 3, 3), dtype=torch.float64, device="cpu")
    inverse_matrices = torch.empty_like(forward_matrices)
    for index, (x, y, scale) in enumerate(zip(center_x, center_y, scales)):
        tx = crop_center_x - scale * x
        ty = crop_center_y - scale * y
        forward_matrices[index] = torch.tensor(
            ((scale, 0.0, tx), (0.0, scale, ty), (0.0, 0.0, 1.0)),
            dtype=torch.float64,
        )
        inverse_scale = 1.0 / scale
        inverse_matrices[index] = torch.tensor(
            (
                (inverse_scale, 0.0, -tx * inverse_scale),
                (0.0, inverse_scale, -ty * inverse_scale),
                (0.0, 0.0, 1.0),
            ),
            dtype=torch.float64,
        )

    return {
        "type": _DATA_TYPE,
        "version": _DATA_VERSION,
        "frame_count": len(observations),
        "source_width": int(source_width),
        "source_height": int(source_height),
        "crop_width": crop_width,
        "crop_height": crop_height,
        "multiple": int(multiple),
        "crop_margin_percent_per_side": float(crop_margin_percent),
        "crop_geometry_source": "unsmoothed_stable_bbox",
        "average_frames": average_frames,
        "mask_blur_sigma": float(mask_blur_sigma),
        "bbox_threshold": _BBOX_THRESHOLD,
        "anchor_index": anchor_index,
        "anchor_bbox": tuple(anchor["bbox"]),
        "anchor_area": anchor_area,
        "frame_status": tuple(item["status"] for item in observations),
        "source_centers": torch.tensor(list(zip(center_x, center_y)), dtype=torch.float64),
        "scales": torch.tensor(scales, dtype=torch.float64),
        "observed_bboxes": tuple(
            None if item["bbox"] is None else tuple(item["bbox"])
            for item in observations
        ),
        "forward_matrices": forward_matrices,
        "inverse_matrices": inverse_matrices,
        "coordinate_convention": "pixel_centers_align_corners_false",
        "padding_mode": "zeros",
    }


def _pixel_grid(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width, device=device, dtype=dtype)
    return torch.meshgrid(y, x, indexing="ij")


def _normalise_grid(
    x: torch.Tensor,
    y: torch.Tensor,
    input_width: int,
    input_height: int,
) -> torch.Tensor:
    normal_x = (2.0 * x + 1.0) / float(input_width) - 1.0
    normal_y = (2.0 * y + 1.0) / float(input_height) - 1.0
    return torch.stack((normal_x, normal_y), dim=-1).unsqueeze(0)


def _sample_stable_frame(
    image: torch.Tensor,
    mask: torch.Tensor,
    center_x: float,
    center_y: float,
    scale: float,
    crop_height: int,
    crop_width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_height, source_width = int(image.shape[0]), int(image.shape[1])
    target_y, target_x = _pixel_grid(crop_height, crop_width, device, torch.float32)
    crop_center_x = (crop_width - 1) * 0.5
    crop_center_y = (crop_height - 1) * 0.5
    source_x = (target_x - crop_center_x) / scale + center_x
    source_y = (target_y - crop_center_y) / scale + center_y
    grid = _normalise_grid(source_x, source_y, source_width, source_height)

    image_chw = image.to(
        device=device,
        dtype=torch.float32,
        non_blocking=device.type == "cuda",
    ).movedim(-1, 0).unsqueeze(0)
    mask_chw = mask.to(
        device=device,
        dtype=torch.float32,
        non_blocking=device.type == "cuda",
    ).unsqueeze(0).unsqueeze(0)
    sampled = F.grid_sample(
        torch.cat((image_chw, mask_chw), dim=1),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled[:, :3].movedim(1, -1)[0], sampled[:, 3]


@torch.inference_mode()
def spatial_stabilize(
    image: torch.Tensor,
    mask: torch.Tensor,
    multiple: int,
    crop_margin_percent: float,
    average_frames: int,
    mask_blur_sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    _spatial_info("CS Spatial Stabilize", "stage 1/4: validating IMAGE and MASK inputs")
    source = _normalise_image(image, "image")
    frame_count, source_height, source_width = map(int, source.shape[:3])
    prepared_mask = _prepare_stabilize_mask(mask, frame_count, source_height, source_width)
    _spatial_info(
        "CS Spatial Stabilize",
        "input frames: %d; source=%dx%d; mask frames=%d",
        frame_count,
        source_width,
        source_height,
        int(prepared_mask.shape[0]),
    )
    _spatial_info(
        "CS Spatial Stabilize",
        "stage 2/4: applying mask pre-blur and analysing frame states and BBoxes",
    )
    observations = _bbox_observations(
        prepared_mask,
        frame_count,
        source_height,
        source_width,
        max(0.0, float(mask_blur_sigma)),
    )
    status_counts = {
        status: sum(item["status"] == status for item in observations)
        for status in ("valid", "out_of_frame", "empty")
    }
    _spatial_info(
        "CS Spatial Stabilize",
        "mask analysis complete: valid=%d; out_of_frame=%d; empty=%d",
        status_counts["valid"],
        status_counts["out_of_frame"],
        status_counts["empty"],
    )
    _spatial_info("CS Spatial Stabilize", "stage 3/4: calculating anchor, smoothing and crop geometry")
    stable_data = _analyse_stabilization(
        observations,
        max(1, int(multiple)),
        max(0.0, float(crop_margin_percent)),
        max(1, int(average_frames)),
        max(0.0, float(mask_blur_sigma)),
        source_width,
        source_height,
    )

    crop_height = int(stable_data["crop_height"])
    crop_width = int(stable_data["crop_width"])
    centers = stable_data["source_centers"]
    scales = stable_data["scales"]
    output_images = torch.empty(
        (frame_count, crop_height, crop_width, 3),
        dtype=torch.float32,
        device="cpu",
    )
    output_masks = torch.empty(
        (frame_count, crop_height, crop_width),
        dtype=torch.float32,
        device="cpu",
    )
    device = _compute_device(source.device)

    _spatial_info(
        "CS Spatial Stabilize",
        "geometry ready: crop=%dx%d; anchor=%d; average_frames=%d; mask_blur_sigma=%.2f; device=%s",
        crop_width,
        crop_height,
        int(stable_data["anchor_index"]),
        int(stable_data["average_frames"]),
        float(stable_data["mask_blur_sigma"]),
        device,
    )
    _spatial_info(
        "CS Spatial Stabilize",
        "stage 4/4: sampling stabilized frames and masks",
    )
    _spatial_info(
        "CS Spatial Stabilize",
        "frames=%d; source=%dx%d; crop=%dx%d; anchor=%d; device=%s",
        frame_count,
        source_width,
        source_height,
        crop_width,
        crop_height,
        int(stable_data["anchor_index"]),
        device,
    )
    progress = _SpatialProgress("CS Spatial Stabilize", frame_count, "stabilizing frames")

    try:
        index = 0
        while index < frame_count:
            source_frame = mask_frame = sampled_image = sampled_mask = None
            try:
                source_frame = source[index]
                mask_frame = _mask_frame(
                    prepared_mask,
                    index,
                    source_height,
                    source_width,
                    device=device,
                )
                sampled_image, sampled_mask = _sample_stable_frame(
                    source_frame,
                    mask_frame,
                    float(centers[index, 0]),
                    float(centers[index, 1]),
                    float(scales[index]),
                    crop_height,
                    crop_width,
                    device,
                )
                output_images[index].copy_(sampled_image.to(device="cpu").clamp(0.0, 1.0))
                output_masks[index].copy_(sampled_mask[0].to(device="cpu").clamp(0.0, 1.0))
                progress.update()
                index += 1
            except RuntimeError as exc:
                if device.type != "cuda" or not _is_cuda_oom(exc):
                    raise
                source_frame = mask_frame = sampled_image = sampled_mask = None
                torch.cuda.empty_cache()
                device = torch.device("cpu")
                _LOGGER.warning("[CS Spatial Stabilize] one frame did not fit VRAM; continuing on CPU")
            finally:
                source_frame = mask_frame = sampled_image = sampled_mask = None
    finally:
        progress.close()

    _spatial_info(
        "CS Spatial Stabilize",
        "complete: output=%d frames at %dx%d",
        frame_count,
        crop_width,
        crop_height,
    )
    return output_images, output_masks, stable_data


def _validate_stable_data(
    stable_data: Any,
    frame_count: int,
    source_height: int,
    source_width: int,
    crop_height: int,
    crop_width: int,
) -> torch.Tensor:
    if not isinstance(stable_data, dict) or stable_data.get("type") != _DATA_TYPE:
        raise ValueError("stable_data must come from CS Spatial Stabilize.")
    if int(stable_data.get("version", -1)) != _DATA_VERSION:
        raise ValueError(f"Unsupported stable_data version: {stable_data.get('version')!r}.")

    expected = {
        "frame_count": frame_count,
        "source_height": source_height,
        "source_width": source_width,
        "crop_height": crop_height,
        "crop_width": crop_width,
    }
    for field, value in expected.items():
        if int(stable_data.get(field, -1)) != value:
            raise ValueError(
                f"stable_data {field} is {stable_data.get(field)!r}, but the connected tensors require {value}."
            )
    if stable_data.get("coordinate_convention") != "pixel_centers_align_corners_false":
        raise ValueError("stable_data uses an unsupported coordinate convention.")

    matrices = stable_data.get("forward_matrices")
    if not isinstance(matrices, torch.Tensor) or tuple(matrices.shape) != (frame_count, 3, 3):
        raise ValueError("stable_data forward_matrices are missing or malformed.")
    matrices = matrices.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(matrices).all().item()):
        raise ValueError("stable_data forward_matrices must contain only finite values.")
    if not bool((matrices[:, 0, 0] > 0).all().item()):
        raise ValueError("stable_data contains a non-positive scale.")
    return matrices


def _feather_alpha(
    height: int,
    width: int,
    soft_border: int,
    device: torch.device,
) -> torch.Tensor:
    if soft_border <= 0:
        return torch.ones((height, width), device=device, dtype=torch.float32)
    y = torch.arange(height, device=device, dtype=torch.float32)
    x = torch.arange(width, device=device, dtype=torch.float32)
    distance_y = torch.minimum(y, (height - 1) - y).unsqueeze(1)
    distance_x = torch.minimum(x, (width - 1) - x).unsqueeze(0)
    return (torch.minimum(distance_y, distance_x) / float(soft_border)).clamp(0.0, 1.0)


def _restore_roi(
    matrix: torch.Tensor,
    source_height: int,
    source_width: int,
    crop_height: int,
    crop_width: int,
) -> tuple[int, int, int, int]:
    scale = float(matrix[0, 0])
    tx = float(matrix[0, 2])
    ty = float(matrix[1, 2])
    left = (-0.5 - tx) / scale
    right = (crop_width - 0.5 - tx) / scale
    top = (-0.5 - ty) / scale
    bottom = (crop_height - 0.5 - ty) / scale
    x0 = max(0, int(math.floor(min(left, right))) - 1)
    x1 = min(source_width, int(math.ceil(max(left, right))) + 2)
    y0 = max(0, int(math.floor(min(top, bottom))) - 1)
    y1 = min(source_height, int(math.ceil(max(top, bottom))) + 2)
    return x0, y0, x1, y1


def _sample_restore_frame(
    local_image: torch.Tensor,
    local_alpha: torch.Tensor,
    matrix: torch.Tensor,
    roi: tuple[int, int, int, int],
    crop_height: int,
    crop_width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x0, y0, x1, y1 = roi
    target_y, target_x = _pixel_grid(y1 - y0, x1 - x0, device, torch.float32)
    target_x = target_x + x0
    target_y = target_y + y0
    scale = float(matrix[0, 0])
    crop_x = target_x * scale + float(matrix[0, 2])
    crop_y = target_y * scale + float(matrix[1, 2])
    grid = _normalise_grid(crop_x, crop_y, crop_width, crop_height)

    image_chw = local_image.to(
        device=device,
        dtype=torch.float32,
        non_blocking=device.type == "cuda",
    ).movedim(-1, 0).unsqueeze(0)
    alpha_chw = local_alpha.to(
        device=device,
        dtype=torch.float32,
        non_blocking=device.type == "cuda",
    ).clamp(0.0, 1.0).unsqueeze(0).unsqueeze(0)
    premultiplied = image_chw * alpha_chw
    sampled = F.grid_sample(
        torch.cat((premultiplied, alpha_chw), dim=1),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled[:, :3].movedim(1, -1)[0], sampled[:, 3, :, :][0]


@torch.inference_mode()
def spatial_restore(
    source_image: torch.Tensor,
    stabilized_image: torch.Tensor,
    stable_data: Any,
    mask: torch.Tensor | None,
    soft_border: int,
) -> torch.Tensor:
    _spatial_info("CS Spatial Restore", "stage 1/3: validating source, local frames and stable data")
    source = _normalise_image(source_image, "source_image")
    local = _normalise_image(stabilized_image, "stabilized_image")
    frame_count, source_height, source_width = map(int, source.shape[:3])
    local_frames, crop_height, crop_width = map(int, local.shape[:3])
    if local_frames != frame_count:
        raise ValueError(
            f"stabilized_image batch must contain {frame_count} frames; received {local_frames}."
        )

    _spatial_info("CS Spatial Restore", "stage 2/3: preparing reverse sampling and composite mask")
    matrices = _validate_stable_data(
        stable_data,
        frame_count,
        source_height,
        source_width,
        crop_height,
        crop_width,
    )
    prepared_mask = None
    if mask is not None:
        prepared_mask = _prepare_restore_mask(mask, frame_count, crop_height, crop_width)

    output = source.detach().to(device="cpu", dtype=torch.float32).clone().clamp(0.0, 1.0)
    device = _compute_device(local.device)
    feather = None if prepared_mask is not None else _feather_alpha(
        crop_height,
        crop_width,
        max(0, int(soft_border)),
        device,
    )
    _spatial_info(
        "CS Spatial Restore",
        "input frames=%d; source=%dx%d; crop=%dx%d; mask=%s; soft_border=%d; device=%s",
        frame_count,
        source_width,
        source_height,
        crop_width,
        crop_height,
        prepared_mask is not None,
        max(0, int(soft_border)),
        device,
    )
    _spatial_info("CS Spatial Restore", "stage 3/3: restoring local frames to source positions")
    progress = _SpatialProgress("CS Spatial Restore", frame_count, "restoring frames")

    try:
        index = 0
        while index < frame_count:
            alpha = restored_rgb = restored_alpha = None
            try:
                matrix = matrices[index]
                roi = _restore_roi(
                    matrix,
                    source_height,
                    source_width,
                    crop_height,
                    crop_width,
                )
                x0, y0, x1, y1 = roi
                if x0 >= x1 or y0 >= y1:
                    progress.update()
                    index += 1
                    continue
                alpha = (
                    _mask_frame(prepared_mask, index, crop_height, crop_width, device=device)
                    if prepared_mask is not None
                    else feather
                )
                restored_rgb, restored_alpha = _sample_restore_frame(
                    local[index],
                    alpha,
                    matrix,
                    roi,
                    crop_height,
                    crop_width,
                    device,
                )
                rgb_cpu = restored_rgb.to(device="cpu").clamp(0.0, 1.0)
                alpha_cpu = restored_alpha.to(device="cpu").clamp(0.0, 1.0).unsqueeze(-1)
                source_roi = output[index, y0:y1, x0:x1]
                output[index, y0:y1, x0:x1] = source_roi * (1.0 - alpha_cpu) + rgb_cpu
                progress.update()
                index += 1
            except RuntimeError as exc:
                if device.type != "cuda" or not _is_cuda_oom(exc):
                    raise
                alpha = restored_rgb = restored_alpha = feather = None
                torch.cuda.empty_cache()
                device = torch.device("cpu")
                if prepared_mask is None:
                    feather = _feather_alpha(crop_height, crop_width, max(0, int(soft_border)), device)
                _LOGGER.warning("[CS Spatial Restore] one frame did not fit VRAM; continuing on CPU")
            finally:
                alpha = restored_rgb = restored_alpha = None
    finally:
        progress.close()

    _spatial_info(
        "CS Spatial Restore",
        "complete: restored %d frames to %dx%d",
        frame_count,
        source_width,
        source_height,
    )
    return output.clamp(0.0, 1.0)


class CSSpatialStabilize(io.ComfyNode):
    """Stabilize a masked subject's position and equal-area size into a fixed crop."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CS_Spatial_Stabilize",
            display_name="CS Spatial Stabilize",
            category=_CATEGORY,
            essentials_category="Video Tools",
            search_aliases=["mask stabilize", "spatial stabilize", "stable crop", "video crop tracking"],
            description=(
                "Centers and equal-area scales a masked subject across an IMAGE batch, then returns "
                "a fixed multiple-aligned crop and reversible transform data."
            ),
            inputs=[
                io.Image.Input("image", tooltip="Source video frames as a standard ComfyUI IMAGE batch."),
                io.Mask.Input(
                    "mask",
                    tooltip=(
                        "Corresponding MASK batch. Frames beyond the IMAGE batch are discarded; "
                        "missing frames are treated as empty masks."
                    ),
                ),
                io.Int.Input(
                    "multiple",
                    default=32,
                    min=1,
                    max=1024,
                    step=1,
                    tooltip="Crop width and height are rounded upward to this multiple.",
                ),
                io.Float.Input(
                    "crop_margin_percent",
                    display_name="Crop Margin Per Side (%)",
                    default=30.0,
                    min=0.0,
                    max=500.0,
                    step=0.1,
                    tooltip=(
                        "Extra margin on each side as a percentage of the maximum stabilized BBox; "
                        "one multiple per side is always added first."
                    ),
                ),
                io.Int.Input(
                    "average_frames",
                    display_name="Average Frames",
                    default=8,
                    min=1,
                    max=999,
                    step=1,
                    tooltip=(
                        "Centred moving-average window applied to the interpolated X, Y, and Scale "
                        "values while the first and last frames remain locked; 1 disables smoothing."
                    ),
                ),
                io.Float.Input(
                    "mask_blur_sigma",
                    display_name="Mask Blur Sigma",
                    default=6.0,
                    min=0.0,
                    max=256.0,
                    step=0.1,
                    tooltip=(
                        "Gaussian sigma applied only before the > 0.5 BBox threshold. "
                        "This suppresses isolated mask pixels; 0 disables pre-blur."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output("image", display_name="IMAGE"),
                SPATIAL_STABLE_DATA.Output("stable_data", display_name="STABLE_DATA"),
                io.Mask.Output("mask", display_name="MASK"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor,
        multiple: int = 32,
        crop_margin_percent: float = 30.0,
        average_frames: int = 8,
        mask_blur_sigma: float = 6.0,
    ) -> io.NodeOutput:
        stable_image, stable_mask, stable_data = spatial_stabilize(
            image,
            mask,
            int(multiple),
            float(crop_margin_percent),
            int(average_frames),
            float(mask_blur_sigma),
        )
        return io.NodeOutput(stable_image, stable_data, stable_mask)


class CSSpatialRestore(io.ComfyNode):
    """Restore a processed stabilized crop into its original frame positions."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CS_Spatial_Restore",
            display_name="CS Spatial Restore",
            category=_CATEGORY,
            essentials_category="Video Tools",
            search_aliases=[
                "mask restore",
                "spatial restore",
                "restore crop",
                "video patch restore",
            ],
            description=(
                "Uses CS Spatial Stabilize transform data to composite a processed local IMAGE batch "
                "back into the original source frames."
            ),
            inputs=[
                io.Image.Input(
                    "source_image",
                    tooltip="The same source IMAGE batch connected to CS Spatial Stabilize.",
                ),
                io.Image.Input(
                    "stabilized_image",
                    tooltip="Processed local IMAGE batch with unchanged crop size and frame count.",
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip=(
                        "Optional local MASK batch corresponding to stabilized_image. When connected, "
                        "it controls the restored composite and Soft Border is ignored."
                    ),
                ),
                SPATIAL_STABLE_DATA.Input(
                    "stable_data",
                    tooltip="Transform data produced by the matching CS Spatial Stabilize execution.",
                ),
                io.Int.Input(
                    "soft_border",
                    display_name="Soft Border (crop px)",
                    default=32,
                    min=0,
                    max=4096,
                    step=1,
                    tooltip=(
                        "Feather width measured in stabilized crop pixels. Used only when no mask is connected."
                    ),
                ),
            ],
            outputs=[io.Image.Output("image", display_name="IMAGE")],
        )

    @classmethod
    def execute(
        cls,
        source_image: torch.Tensor,
        stabilized_image: torch.Tensor,
        mask: torch.Tensor | None = None,
        stable_data: Any = None,
        soft_border: int = 32,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            spatial_restore(
                source_image,
                stabilized_image,
                stable_data,
                mask,
                int(soft_border),
            )
        )


class SpatialStabilizeExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CSSpatialStabilize, CSSpatialRestore]


async def comfy_entrypoint() -> SpatialStabilizeExtension:
    return SpatialStabilizeExtension()
