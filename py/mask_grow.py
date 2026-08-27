"""Exact Euclidean mask growth for image and video mask batches."""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


_LOGGER = logging.getLogger(__name__)
_CATEGORY = "😺dzNodes/CineStyle"
_BINARY_THRESHOLD = 128.0 / 255.0
_GPU_MEMORY_FRACTION = 0.70
_GPU_MEMORY_RESERVE_BYTES = 768 * 1024**2
_MAX_GPU_BATCH = 64
_CPU_BATCH = 1


def _normalise_mask(mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise ValueError("mask must be a ComfyUI MASK tensor.")
    if mask.ndim == 2:
        value = mask.unsqueeze(0)
    elif mask.ndim == 3:
        value = mask
    elif mask.ndim == 4 and mask.shape[1] == 1:
        value = mask[:, 0]
    elif mask.ndim == 4 and mask.shape[-1] == 1:
        value = mask[..., 0]
    else:
        raise ValueError("mask must have shape [H,W], [B,H,W], [B,1,H,W], or [B,H,W,1].")
    if value.shape[0] < 1 or value.shape[-2] < 1 or value.shape[-1] < 1:
        raise ValueError("mask dimensions must be non-empty.")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError("mask must contain only finite values.")
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


def _estimated_bytes_per_frame(height: int, width: int, preserve_soft_edges: bool) -> int:
    # Includes the uploaded float mask, output, prefix sums or pooling workspace,
    # and temporary row results. OOM retry handles backend-specific workspaces.
    bytes_per_pixel = 56 if preserve_soft_edges else 36
    return max(1, height * width * bytes_per_pixel)


def _batch_size(mask: torch.Tensor, device: torch.device, preserve_soft_edges: bool) -> int:
    total, height, width = map(int, mask.shape)
    if total <= 1:
        return 1
    if device.type == "cuda":
        estimated = _estimated_bytes_per_frame(height, width, preserve_soft_edges)
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
            available = max(0, int(free_bytes) - _GPU_MEMORY_RESERVE_BYTES)
            budget = max(estimated, int(available * _GPU_MEMORY_FRACTION))
            return max(1, min(total, _MAX_GPU_BATCH, budget // estimated))
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return 1
    return max(1, min(total, _CPU_BATCH))


def _row_slices(height: int, offset: int) -> tuple[int, int, int, int] | None:
    if offset >= 0:
        target_start, target_end = 0, height - offset
        source_start, source_end = offset, height
    else:
        target_start, target_end = -offset, height
        source_start, source_end = 0, height + offset
    if target_start >= target_end:
        return None
    return target_start, target_end, source_start, source_end


def _binary_disk_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Dilate a Bx1xHxW binary mask by an exact discrete Euclidean disk."""
    if radius <= 0:
        return mask.clone()
    batch, channels, height, width = map(int, mask.shape)
    if channels != 1:
        raise ValueError("internal mask tensor must have one channel.")

    prefix = F.pad(torch.cumsum(mask.to(torch.int32), dim=-1, dtype=torch.int32), (1, 0))
    output = torch.zeros((batch, 1, height, width), device=mask.device, dtype=torch.bool)
    x = torch.arange(width, device=mask.device)
    horizontal_indices: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    radius_squared = radius * radius

    for y_offset in range(min(radius, height - 1) + 1):
        x_radius = min(math.isqrt(radius_squared - y_offset * y_offset), width - 1)
        indices = horizontal_indices.get(x_radius)
        if indices is None:
            indices = (
                (x - x_radius).clamp(0, width),
                (x + x_radius + 1).clamp(0, width),
            )
            horizontal_indices[x_radius] = indices
        left, right = indices
        offsets = (0,) if y_offset == 0 else (-y_offset, y_offset)
        for offset in offsets:
            slices = _row_slices(height, offset)
            if slices is None:
                continue
            target_start, target_end, source_start, source_end = slices
            rows = prefix[:, :, source_start:source_end]
            hits = rows.index_select(-1, right) - rows.index_select(-1, left) > 0
            output[:, :, target_start:target_end] |= hits
    return output


def _soft_disk_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Apply grayscale dilation with the same exact Euclidean disk."""
    if radius <= 0:
        return mask.clone()
    batch, channels, height, width = map(int, mask.shape)
    if channels != 1:
        raise ValueError("internal mask tensor must have one channel.")

    output = torch.zeros_like(mask)
    radius_squared = radius * radius
    for y_offset in range(min(radius, height - 1) + 1):
        x_radius = min(math.isqrt(radius_squared - y_offset * y_offset), width - 1)
        kernel_size = x_radius * 2 + 1
        offsets = (0,) if y_offset == 0 else (-y_offset, y_offset)
        for offset in offsets:
            slices = _row_slices(height, offset)
            if slices is None:
                continue
            target_start, target_end, source_start, source_end = slices
            rows = mask[:, :, source_start:source_end].reshape(-1, 1, width)
            pooled = F.max_pool1d(
                rows,
                kernel_size=kernel_size,
                stride=1,
                padding=x_radius,
            ).reshape(batch, 1, target_end - target_start, width)
            output[:, :, target_start:target_end] = torch.maximum(
                output[:, :, target_start:target_end],
                pooled,
            )
    return output


def _grow_chunk(mask: torch.Tensor, grow: int, preserve_soft_edges: bool) -> torch.Tensor:
    radius = abs(int(grow))
    if preserve_soft_edges:
        value = mask.clamp(0.0, 1.0)
        if grow >= 0:
            return _soft_disk_dilate(value, radius)
        return 1.0 - _soft_disk_dilate(1.0 - value, radius)

    binary = mask.clamp(0.0, 1.0) >= _BINARY_THRESHOLD
    if grow >= 0:
        return _binary_disk_dilate(binary, radius).to(torch.float32)
    return (~_binary_disk_dilate(~binary, radius)).to(torch.float32)


def _is_cuda_oom(exc: RuntimeError) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


@torch.inference_mode()
def grow_mask_batch(mask: torch.Tensor, grow: int, preserve_soft_edges: bool) -> torch.Tensor:
    """Grow a standard MASK tensor in VRAM-bounded frame batches."""
    normalised = _normalise_mask(mask)
    total, height, width = map(int, normalised.shape)
    device = _compute_device(normalised.device)
    batch_size = _batch_size(normalised, device, preserve_soft_edges)
    output = torch.empty((total, height, width), device="cpu", dtype=torch.float32)
    _LOGGER.info(
        "[CS Mask Grow] frames=%d; size=%dx%d; grow=%d; soft=%s; device=%s; batch=%d",
        total,
        width,
        height,
        int(grow),
        bool(preserve_soft_edges),
        device,
        batch_size,
    )

    start = 0
    while start < total:
        end = min(total, start + batch_size)
        source = result = None
        try:
            source = normalised[start:end].to(
                device=device,
                dtype=torch.float32,
                non_blocking=device.type == "cuda",
            ).unsqueeze(1)
            result = _grow_chunk(source, int(grow), bool(preserve_soft_edges))
            output[start:end].copy_(result[:, 0].to(device="cpu", dtype=torch.float32))
            start = end
        except RuntimeError as exc:
            if device.type != "cuda" or not _is_cuda_oom(exc):
                raise
            source = result = None
            torch.cuda.empty_cache()
            if batch_size > 1:
                batch_size = max(1, batch_size // 2)
                _LOGGER.warning("[CS Mask Grow] CUDA OOM; retrying with batch=%d", batch_size)
                continue
            device = torch.device("cpu")
            batch_size = _CPU_BATCH
            _LOGGER.warning("[CS Mask Grow] one frame does not fit VRAM; continuing on CPU")
        finally:
            source = result = None
    return output


class CSMaskGrow(io.ComfyNode):
    """Grow or shrink mask batches with an isotropic Euclidean contour."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CS_Mask_Grow",
            display_name="CS Mask Grow",
            category=_CATEGORY,
            essentials_category="Mask",
            search_aliases=["grow mask", "expand mask", "shrink mask", "euclidean mask", "video mask"],
            description="Exact Euclidean mask grow/shrink with automatic VRAM-bounded video batching.",
            inputs=[
                io.Mask.Input("mask", tooltip="Standard ComfyUI MASK tensor; video frames are processed as a batch."),
                io.Int.Input("grow", default=0, min=-4096, max=4096, step=1, tooltip="Positive values grow outward; negative values shrink inward, in pixels."),
                io.Boolean.Input(
                    "preserve_soft_edges",
                    display_name="Preserve Soft Edges",
                    default=False,
                    tooltip="Keep grayscale alpha transitions. When disabled, input is binarized at gray value 128 before processing.",
                ),
            ],
            outputs=[io.Mask.Output("mask", display_name="MASK")],
        )

    @classmethod
    @torch.inference_mode()
    def execute(
        cls,
        mask: torch.Tensor,
        grow: int = 0,
        preserve_soft_edges: bool = False,
    ) -> io.NodeOutput:
        return io.NodeOutput(grow_mask_batch(mask, int(grow), bool(preserve_soft_edges)))


class MaskGrowExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CSMaskGrow]


async def comfy_entrypoint() -> MaskGrowExtension:
    return MaskGrowExtension()
