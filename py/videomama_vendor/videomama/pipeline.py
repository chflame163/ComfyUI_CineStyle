"""VideoMaMa single-step video matting inference.

This is the engine the CS VideoMaMa node drives: the Sammie-Roto 2 fork of the
upstream VideoMaMa ``VideoInferencePipeline``, not the stock one. The difference
matters for VRAM. The upstream pipeline loads the SVD CLIP image encoder, runs it
on the first frame, then throws the result away as ``torch.zeros_like(...)``; this
builds the zero cross-attention conditioning directly from
``unet.config.cross_attention_dim`` and never loads the encoder at all, which keeps
about 1.2 GiB of weights off the card and removes a wasted forward per pass.

It also encodes and decodes the VAE in chunks, frees each stage's tensors before
the next one runs, and hands back plain numpy planes instead of PIL images.

Output is one 8-bit alpha plane per input frame. The upstream code averages the
decoded RGB channels and replicates that value across three channels; averaging
three identical channels returns the same number, so returning the mean directly
is lossless and skips a copy.
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Callable, Sequence

import numpy as np
import torch
from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel

_LOGGER = logging.getLogger("CineStyleVideoMaMa")


class CancelledError(RuntimeError):
    """Raised when the caller asks for the run to stop between chunks."""


class VideoMaMaPipeline:
    """Single-step VideoMaMa inference: coarse binary masks in, soft mattes out."""

    def __init__(
        self,
        model_dir: str,
        device: torch.device,
        fp16: bool = True,
        vae_encode_chunk_size: int = 1,
        vae_decode_chunk_size: int = 4,
        attention_mode: str = "auto",
        enable_vae_slicing: bool = True,
        cpu_offload: bool = False,
        clear_cache: Callable[[], None] | None = None,
    ) -> None:
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.weight_dtype = torch.float16 if fp16 else torch.float32
        self.vae_encode_chunk_size = max(1, int(vae_encode_chunk_size))
        self.vae_decode_chunk_size = max(1, int(vae_decode_chunk_size))
        self.cpu_offload = bool(cpu_offload) and self.device.type == "cuda"
        self._clear_cache = clear_cache or self._default_clear_cache

        # The official VideoMaMa release ships the UNet as fp32 and the SVD VAE as an
        # fp16-only file, so the VAE is asked for the fp16 variant first and falls back
        # to whatever is on disk.
        self.vae = _load_vae(model_dir, self.weight_dtype)
        self.unet = UNetSpatioTemporalConditionModel.from_pretrained(
            model_dir, subfolder="unet", torch_dtype=self.weight_dtype
        )
        self.vae.eval()
        self.unet.eval()

        _apply_attention_optimization(self.unet, attention_mode)

        if enable_vae_slicing:
            try:
                self.vae.enable_slicing()
            except (AttributeError, NotImplementedError):
                pass

        target = "cpu" if self.cpu_offload else self.device
        self.vae.to(target)
        self.unet.to(target)

    # ------------------------------------------------------------------ lifecycle

    def unload(self) -> None:
        self.vae = None
        self.unet = None
        gc.collect()
        self._clear_cache()

    def _default_clear_cache(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ inference

    @torch.inference_mode()
    def run(
        self,
        cond_frames: Sequence[np.ndarray],
        mask_frames: Sequence[np.ndarray],
        seed: int = 42,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.0,
        on_progress: Callable[[str, int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[np.ndarray]:
        """Return one uint8 alpha plane per frame, at the size the frames came in at."""
        if len(cond_frames) != len(mask_frames) or not len(cond_frames):
            raise ValueError("cond_frames and mask_frames must match and be non-empty")

        def tick(label: str, done: int, total: int) -> None:
            if should_cancel is not None and should_cancel():
                raise CancelledError("cancelled")
            if on_progress is not None:
                on_progress(label, done, total)

        cond = self._to_video_tensor(cond_frames).to(self.device)
        mask = self._to_video_tensor(mask_frames).to(self.device)

        # Cross-attention conditioning is trained as zeros, so build it directly and
        # skip the SVD image encoder entirely.
        encoder_hidden_states = torch.zeros(
            (1, 1, self.unet.config.cross_attention_dim),
            dtype=self.weight_dtype,
            device=self.device,
        )

        if self.cpu_offload:
            self.vae.to(self.device)

        total = len(cond_frames)
        tick("encoding", 0, total)
        cond_latents = self._encode(cond, self.vae_encode_chunk_size, "encoding", tick)
        mask_latents = self._encode(mask, self.vae_encode_chunk_size, "encoding", tick)
        del cond, mask
        self._clear_cache()

        if self.cpu_offload:
            self.vae.to("cpu")
            self.unet.to(self.device)

        tick("diffusing", 0, total)
        # A CPU generator keeps the noise identical across devices, which matters when a
        # workflow is re-run on a different card.
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        noisy = torch.randn(
            cond_latents.shape, generator=generator, dtype=self.weight_dtype
        ).to(self.device)
        timesteps = torch.full((1,), 1.0, device=self.device, dtype=torch.long)
        added_time_ids = self._add_time_ids(fps, motion_bucket_id, noise_aug_strength)

        unet_input = torch.cat([noisy, cond_latents, mask_latents], dim=2)
        del noisy, cond_latents, mask_latents
        self._clear_cache()

        pred = self.unet(
            unet_input, timesteps, encoder_hidden_states, added_time_ids=added_time_ids
        ).sample
        del unet_input, encoder_hidden_states, added_time_ids, timesteps
        if self.cpu_offload:
            self.unet.to("cpu")
            self.vae.to(self.device)
        self._clear_cache()

        tick("decoding", 0, total)
        pred = (1.0 / self.vae.config.scaling_factor) * pred.squeeze(0)
        chunks: list[torch.Tensor] = []
        step = min(self.vae_decode_chunk_size, pred.shape[0])
        for index in range(0, pred.shape[0], step):
            tick("decoding", index, total)
            piece = pred[index: index + step]
            # The temporal decoder's convolutions only see what is inside this call, so
            # a larger chunk here is what buys VAE-level temporal smoothing.
            chunks.append(self.vae.decode(piece, num_frames=piece.shape[0]).sample.cpu())
        del pred
        if self.cpu_offload:
            self.vae.to("cpu")
        self._clear_cache()

        video = torch.cat(chunks, dim=0)
        del chunks
        alpha = (video / 2.0 + 0.5).clamp(0.0, 1.0).mean(dim=1)
        alpha = alpha.mul(255.0).round().clamp(0.0, 255.0).to(torch.uint8).numpy()
        tick("decoding", total, total)
        return [np.ascontiguousarray(plane) for plane in alpha]

    # ------------------------------------------------------------------ tensor plumbing

    def _to_video_tensor(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        """Frames in ``{uint8}`` -> ``(1, F, 3, H, W)`` float in -1..1, in weight dtype."""
        planes = []
        for frame in frames:
            array = np.asarray(frame)
            if array.ndim == 2:
                array = np.repeat(array[..., None], 3, axis=-1)
            elif array.shape[-1] == 1:
                array = np.repeat(array, 3, axis=-1)
            elif array.shape[-1] != 3:
                raise ValueError(f"expected HxW or HxWx3 frames, got {array.shape}")
            planes.append(np.ascontiguousarray(array.transpose(2, 0, 1)))
        stacked = np.stack(planes).astype(np.float32) / 255.0
        tensor = torch.from_numpy(stacked).unsqueeze(0).mul(2.0).sub(1.0)
        return tensor.to(dtype=self.weight_dtype)

    def _encode(
        self,
        video: torch.Tensor,
        chunk: int,
        label: str,
        tick: Callable[[str, int, int], None],
    ) -> torch.Tensor:
        """VAE-encode a ``(B, F, C, H, W)`` video in chunks along the frame axis.

        Returns the raw posterior sample. The upstream pipeline multiplies by
        ``vae.config.scaling_factor`` here and divides by it again in the caller, which
        cancels out; skipping the pair keeps the conditioning in the same space the UNet
        was trained in and costs one rounding step less.
        """
        batch, frames = video.shape[0], video.shape[1]
        flat = video.reshape(batch * frames, *video.shape[2:])
        out: list[torch.Tensor] = []
        for index in range(0, flat.shape[0], max(1, int(chunk))):
            tick(label, index, flat.shape[0])
            piece = flat[index: index + chunk]
            out.append(self.vae.encode(piece).latent_dist.sample())
        latents = torch.cat(out, dim=0)
        del out
        latents = latents.reshape(batch, frames, *latents.shape[1:])
        return latents

    def _add_time_ids(self, fps: int, motion_bucket_id: int, noise_aug_strength: float) -> torch.Tensor:
        values = [fps, motion_bucket_id, noise_aug_strength]
        passed = self.unet.config.addition_time_embed_dim * len(values)
        expected = self.unet.add_embedding.linear_1.in_features
        if expected != passed:
            raise ValueError(
                f"UNet expects an added time embedding of length {expected}, got {passed}."
            )
        return torch.tensor([values], dtype=self.weight_dtype, device=self.device)


def _load_vae(model_dir: str, dtype: torch.dtype) -> AutoencoderKLTemporalDecoder:
    """Prefer the fp16 VAE weights, which is the only file the official layout carries."""
    try:
        return AutoencoderKLTemporalDecoder.from_pretrained(
            model_dir, subfolder="vae", variant="fp16", torch_dtype=dtype
        )
    except (OSError, ValueError):
        _LOGGER.info("[CS VideoMaMa] no fp16 VAE variant on disk, loading the default file")
        return AutoencoderKLTemporalDecoder.from_pretrained(
            model_dir, subfolder="vae", torch_dtype=dtype
        )


def _apply_attention_optimization(unet: UNetSpatioTemporalConditionModel, mode: str) -> None:
    """Memory-efficient attention, best available first. Silent when the backend is absent."""
    if mode == "none":
        return
    if mode in ("auto", "xformers"):
        try:
            import xformers  # noqa: F401

            unet.enable_xformers_memory_efficient_attention()
            return
        except Exception:
            if mode == "xformers":
                _LOGGER.info("[CS VideoMaMa] xformers unavailable, falling back to SDPA")
    if mode in ("auto", "sdpa"):
        try:
            from diffusers.models.attention_processor import AttnProcessor2_0

            unet.set_attn_processor(AttnProcessor2_0())
        except Exception:
            _LOGGER.info("[CS VideoMaMa] SDPA processor unavailable, using diffusers default")


__all__ = ["VideoMaMaPipeline", "CancelledError"]
