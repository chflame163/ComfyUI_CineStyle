"""CineStyle video input and output nodes for ComfyUI."""

from __future__ import annotations

import os
import math
import json
import hashlib
import asyncio
import threading
import logging
import time
import sys
from fractions import Fraction
from typing import Any

import av
import torch
import torch.nn.functional as F
from aiohttp import web
from typing_extensions import override

import folder_paths
from comfy_api.latest import ComfyExtension, Input, InputImpl, Types, io, ui

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - portable ComfyUI normally provides tqdm
    tqdm = None


_ROUTE_REGISTERED = False
_PROXY_TOTAL_PIXELS = 1_000_000
_PROXY_PROGRESS: dict[str, dict[str, Any]] = {}
_PROXY_PROGRESS_LOCK = threading.Lock()
_LOGGER = logging.getLogger("CineStyleVideoTimeline")


def _video_files() -> list[str]:
    input_dir = folder_paths.get_input_directory()
    files = [
        name
        for name in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, name))
    ]
    return sorted(folder_paths.filter_files_content_types(files, ["video"]))


def _resize_frames(images: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if images.shape[2] == width and images.shape[1] == height:
        return images
    if images.shape[0] == 0:
        return images.new_zeros((0, height, width, images.shape[-1]))
    frames = images.movedim(-1, 1).float()
    frames = F.interpolate(frames, size=(height, width), mode="bilinear", align_corners=False)
    return frames.movedim(1, -1).clamp(0.0, 1.0)


def _round_to_multiple(value: int | float, multiple: int) -> int:
    """Round a dimension to the nearest positive multiple."""
    multiple = max(1, int(multiple))
    rounded = int(math.floor(float(value) / multiple + 0.5)) * multiple
    return max(multiple, rounded)


def _resample_frames(images: torch.Tensor, source_fps: float, target_fps: float) -> torch.Tensor:
    if images.shape[0] <= 1 or target_fps <= 0 or abs(source_fps - target_fps) < 1e-6:
        return images
    output_count = max(1, int(round(images.shape[0] * target_fps / source_fps)))
    indices = torch.linspace(0, images.shape[0] - 1, output_count, device=images.device)
    return images.index_select(0, indices.round().long())


def _trim_audio(audio: dict[str, Any] | None, start_seconds: float, duration: float) -> dict[str, Any] | None:
    if audio is None:
        return None
    waveform = audio["waveform"]
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 3:
        raise ValueError("Video audio must have shape [batch, channels, samples]")
    sample_rate = int(audio["sample_rate"])
    start_sample = max(0, int(round(start_seconds * sample_rate)))
    end_sample = max(start_sample, int(round((start_seconds + duration) * sample_rate)))
    return {
        "waveform": waveform[..., start_sample:end_sample],
        "sample_rate": sample_rate,
    }


def _normalize_proxy_threshold(value: float | int | str | None) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        threshold = 2.1
    if not math.isfinite(threshold) or threshold <= 0:
        threshold = 2.1
    return max(0.1, min(1000.0, threshold))


def _normalize_proxy_size(value: float | int | str | None) -> float:
    try:
        size = float(value)
    except (TypeError, ValueError):
        size = 0.8
    if not math.isfinite(size) or size <= 0:
        size = 0.8
    return max(0.1, min(1000.0, size))


def _read_video_info(filename: str, proxy_threshold: float = 2.1, proxy_size: float = 0.8) -> dict[str, Any]:
    path = _resolve_video_path(filename)
    with av.open(path, mode="r") as container:
        if not container.streams.video:
            raise ValueError(f"No video stream found in '{filename}'")
        stream = container.streams.video[0]
        audio_stream = container.streams.audio[0] if container.streams.audio else None
        width = int(stream.width or 0)
        height = int(stream.height or 0)
        rate = stream.average_rate or stream.guessed_rate or Fraction(24, 1)
        fps = float(Fraction(rate))
        frames = int(stream.frames or 0)
        duration = float(container.duration / av.time_base) if container.duration else 0.0
        if frames <= 0 and duration > 0:
            frames = max(1, int(round(duration * fps)))
        if duration <= 0 and frames > 0:
            duration = frames / fps
    threshold = _normalize_proxy_threshold(proxy_threshold)
    size = _normalize_proxy_size(proxy_size)
    threshold_pixels = threshold * 1_000_000
    proxy_pixels = size * 1_000_000
    proxy_width, proxy_height = _proxy_dimensions(width, height, proxy_pixels)
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
        "duration": duration,
        "audio_format": audio_stream.codec.name if audio_stream and audio_stream.codec else None,
        "proxy_threshold": threshold,
        "proxy_size": size,
        "proxy_required": width * height > threshold_pixels,
        "proxy_width": proxy_width,
        "proxy_height": proxy_height,
    }


def _proxy_dimensions(width: int, height: int, target_pixels: float = _PROXY_TOTAL_PIXELS) -> tuple[int, int]:
    """Return even dimensions whose area is at most the configured preview target."""
    if width <= 0 or height <= 0 or width * height <= target_pixels:
        return max(2, width - width % 2), max(2, height - height % 2)
    scale = math.sqrt(target_pixels / float(width * height))
    proxy_width = max(2, int(width * scale) // 2 * 2)
    proxy_height = max(2, int(height * scale) // 2 * 2)
    while proxy_width * proxy_height > target_pixels:
        if proxy_width >= proxy_height:
            proxy_width -= 2
        else:
            proxy_height -= 2
    return proxy_width, proxy_height


def _proxy_cache_path(source_path: str, proxy_width: int, proxy_height: int) -> str:
    stat = os.stat(source_path)
    cache_key = hashlib.sha1(
        f"{source_path}|{stat.st_size}|{stat.st_mtime_ns}|{proxy_width}x{proxy_height}".encode("utf-8")
    ).hexdigest()
    cache_dir = os.path.join(folder_paths.get_temp_directory(), "cinestyle_proxy")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{cache_key}_{proxy_width}x{proxy_height}.mp4")


def _create_proxy_video(
    source_path: str,
    proxy_path: str,
    proxy_width: int,
    proxy_height: int,
    progress_key: str | None = None,
) -> None:
    temporary_path = f"{proxy_path}.{os.getpid()}.mp4"
    progress_bar = None
    try:
        with av.open(source_path, mode="r") as source:
            video_stream = source.streams.video[0]
            audio_stream = source.streams.audio[0] if source.streams.audio else None
            total_frames = int(video_stream.frames or 0)
            if total_frames <= 0:
                duration = float(source.duration / av.time_base) if source.duration else 0.0
                rate = float(Fraction(video_stream.average_rate or video_stream.guessed_rate or Fraction(24, 1)))
                total_frames = max(1, int(round(duration * rate))) if duration > 0 else 0
            processed_frames = 0
            if progress_key and tqdm is not None:
                progress_bar = tqdm(
                    total=total_frames or None,
                    desc="[INFO] [CS Load Video] proxy frame processing",
                    unit="frame",
                    file=sys.stderr,
                    dynamic_ncols=True,
                    mininterval=0.25,
                    leave=True,
                )
            if progress_key:
                with _PROXY_PROGRESS_LOCK:
                    _PROXY_PROGRESS[progress_key] = {"progress": 1, "running": True}
            frame_rate = video_stream.average_rate or video_stream.guessed_rate or Fraction(24, 1)
            with av.open(temporary_path, mode="w", options={"movflags": "use_metadata_tags+faststart"}) as output:
                output_video = output.add_stream("h264", rate=Fraction(frame_rate))
                output_video.width = proxy_width
                output_video.height = proxy_height
                output_video.pix_fmt = "yuv420p"
                output_video.bit_rate = 2_000_000
                output_video.codec_context.max_b_frames = 0

                output_audio = None
                audio_resampler = None
                if audio_stream is not None and audio_stream.codec_context is not None:
                    sample_rate = int(audio_stream.codec_context.sample_rate or 48000)
                    channels = int(audio_stream.codec_context.channels or 2)
                    layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(channels, "stereo")
                    output_audio = output.add_stream("aac", rate=sample_rate, layout=layout)
                    audio_resampler = av.audio.resampler.AudioResampler(
                        format="fltp", layout=layout, rate=sample_rate
                    )

                streams = (video_stream,) if audio_stream is None else (video_stream, audio_stream)
                for packet in source.demux(*streams):
                    if packet.stream == video_stream:
                        for frame in packet.decode():
                            preview_frame = frame.reformat(
                                width=proxy_width, height=proxy_height, format="yuv420p"
                            )
                            for encoded in output_video.encode(preview_frame):
                                output.mux(encoded)
                            processed_frames += 1
                            if progress_bar is not None:
                                progress_bar.update(1)
                            if progress_key and (processed_frames == 1 or processed_frames % 3 == 0):
                                progress = 5 if total_frames <= 0 else min(95, 5 + int(processed_frames * 90 / total_frames))
                                with _PROXY_PROGRESS_LOCK:
                                    _PROXY_PROGRESS[progress_key] = {"progress": progress, "running": True}
                    elif output_audio is not None and audio_resampler is not None and packet.stream == audio_stream:
                        for frame in packet.decode():
                            for resampled in audio_resampler.resample(frame):
                                for encoded in output_audio.encode(resampled):
                                    output.mux(encoded)

                for encoded in output_video.encode():
                    output.mux(encoded)
                if output_audio is not None and audio_resampler is not None:
                    for resampled in audio_resampler.resample(None):
                        for encoded in output_audio.encode(resampled):
                            output.mux(encoded)
                    for encoded in output_audio.encode():
                        output.mux(encoded)
        os.replace(temporary_path, proxy_path)
        if progress_bar is not None:
            progress_bar.close()
        _LOGGER.info("[CS Load Video] proxy generation complete: %s", proxy_path)
        if progress_key:
            with _PROXY_PROGRESS_LOCK:
                _PROXY_PROGRESS[progress_key] = {"progress": 100, "running": False}
    except Exception:
        if progress_bar is not None:
            progress_bar.close()
        try:
            os.remove(temporary_path)
        except OSError:
            pass
        if progress_key:
            with _PROXY_PROGRESS_LOCK:
                _PROXY_PROGRESS[progress_key] = {"progress": 0, "running": False, "error": "proxy generation failed"}
        raise


def _ensure_proxy_video(source_path: str, proxy_threshold: float, proxy_size: float) -> str:
    """Return the source or a cached proxy path according to the preview settings."""
    info = _read_video_info(source_path, proxy_threshold, proxy_size)
    if not info["proxy_required"]:
        return source_path
    proxy_path = _proxy_cache_path(source_path, info["proxy_width"], info["proxy_height"])
    if not os.path.isfile(proxy_path) or os.path.getsize(proxy_path) == 0:
        _create_proxy_video(source_path, proxy_path, info["proxy_width"], info["proxy_height"])
    return proxy_path


def _resolve_video_path(filename: str) -> str:
    value = str(filename or "").strip()
    if not value:
        raise FileNotFoundError(value)
    if folder_paths.exists_annotated_filepath(value):
        return folder_paths.get_annotated_filepath(value)
    candidate = os.path.abspath(os.path.expandvars(os.path.expanduser(value)))
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(value)


async def _video_info_route(request: web.Request) -> web.Response:
    filename = request.query.get("filename", "")
    if not filename:
        return web.json_response({"error": "filename is required"}, status=400)
    try:
        threshold = request.query.get("proxy_threshold", "2.1")
        size = request.query.get("proxy_size", "0.8")
        return web.json_response(_read_video_info(filename, threshold, size))
    except FileNotFoundError:
        return web.json_response({"error": "video file not found"}, status=404)
    except (OSError, ValueError, av.error.FFmpegError) as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def _video_source_route(request: web.Request) -> web.StreamResponse:
    try:
        path = _resolve_video_path(request.query.get("filename", ""))
    except FileNotFoundError:
        return web.json_response({"error": "video file not found"}, status=404)
    return web.FileResponse(path=path, headers={"Cache-Control": "no-store"})


async def _video_proxy_route(request: web.Request) -> web.StreamResponse:
    try:
        source_path = _resolve_video_path(request.query.get("filename", ""))
        threshold = request.query.get("proxy_threshold", "2.1")
        size = request.query.get("proxy_size", "0.8")
        info = _read_video_info(request.query.get("filename", ""), threshold, size)
        if not info["proxy_required"]:
            return web.FileResponse(path=source_path, headers={"Cache-Control": "no-store"})
        proxy_path = _proxy_cache_path(source_path, info["proxy_width"], info["proxy_height"])
        with _PROXY_PROGRESS_LOCK:
            state = _PROXY_PROGRESS.get(proxy_path)
            running = bool(state and state.get("running"))
            if not running and (not os.path.isfile(proxy_path) or os.path.getsize(proxy_path) == 0):
                _PROXY_PROGRESS[proxy_path] = {"progress": 1, "running": True}
        if not os.path.isfile(proxy_path) or os.path.getsize(proxy_path) == 0:
            if not running:
                await asyncio.to_thread(_create_proxy_video, source_path, proxy_path, info["proxy_width"], info["proxy_height"], proxy_path)
            else:
                while True:
                    await asyncio.sleep(0.1)
                    if os.path.isfile(proxy_path) and os.path.getsize(proxy_path) > 0:
                        break
                    with _PROXY_PROGRESS_LOCK:
                        state = _PROXY_PROGRESS.get(proxy_path, {})
                    if state.get("error"):
                        raise RuntimeError(state["error"])
        return web.FileResponse(path=proxy_path, headers={"Cache-Control": "public,max-age=86400"})
    except FileNotFoundError:
        return web.json_response({"error": "video file not found"}, status=404)
    except (OSError, ValueError, av.error.FFmpegError) as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def _video_proxy_progress_route(request: web.Request) -> web.Response:
    try:
        source_path = _resolve_video_path(request.query.get("filename", ""))
        threshold = request.query.get("proxy_threshold", "2.1")
        size = request.query.get("proxy_size", "0.8")
        info = _read_video_info(request.query.get("filename", ""), threshold, size)
        if not info["proxy_required"]:
            return web.json_response({"progress": 100, "done": True})
        proxy_path = _proxy_cache_path(source_path, info["proxy_width"], info["proxy_height"])
        if os.path.isfile(proxy_path) and os.path.getsize(proxy_path) > 0:
            return web.json_response({"progress": 100, "done": True})
        with _PROXY_PROGRESS_LOCK:
            state = dict(_PROXY_PROGRESS.get(proxy_path, {"progress": 0, "running": False}))
        return web.json_response({
            "progress": int(state.get("progress", 0)),
            "done": not bool(state.get("running", False)) and bool(state.get("error")),
            "error": state.get("error"),
        })
    except FileNotFoundError:
        return web.json_response({"error": "video file not found"}, status=404)
    except (OSError, ValueError, av.error.FFmpegError) as exc:
        return web.json_response({"error": str(exc)}, status=400)


class CSLoadVideo(io.ComfyNode):
    """Load an uploaded video and expose a frame-range editing workflow."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CS_Load_Video",
            display_name="CS Load Video",
            search_aliases=["video timeline", "load video upload", "trim video frames"],
            category="😺dzNodes/CineStyle",
            essentials_category="Video Tools",
            description=(
                "Loads an uploaded video as a ComfyUI IMAGE batch. "
                "Use Edit Timeline to choose the frame range, output size, and FPS."
            ),
            inputs=[
                io.Combo.Input(
                    "video",
                    options=_video_files(),
                    upload=io.UploadType.video,
                    tooltip="Video file in the ComfyUI input directory.",
                ),
                io.Boolean.Input(
                    "keep_aspect_ratio",
                    default=True,
                    advanced=True,
                    tooltip="Keep the source aspect ratio when output dimensions are edited.",
                ),
                io.Int.Input(
                    "multiple",
                    default=32,
                    min=1,
                    max=1024,
                    step=1,
                    advanced=True,
                    tooltip="Round output dimensions to the nearest multiple.",
                ),
                io.Int.Input(
                    "start_frame",
                    default=0,
                    min=0,
                    max=10000000,
                    step=1,
                    advanced=True,
                    tooltip="Inclusive first frame. Edit Timeline is the preferred control.",
                ),
                io.Int.Input(
                    "end_frame",
                    default=-1,
                    min=-1,
                    max=10000000,
                    step=1,
                    advanced=True,
                    tooltip="Inclusive last frame. -1 uses the final source frame.",
                ),
                io.Int.Input(
                    "width",
                    default=0,
                    min=0,
                    max=8192,
                    step=2,
                    advanced=True,
                    tooltip="Output width. 0 keeps the source width.",
                ),
                io.Int.Input(
                    "height",
                    default=0,
                    min=0,
                    max=8192,
                    step=2,
                    advanced=True,
                    tooltip="Output height. 0 keeps the source height.",
                ),
                io.Float.Input(
                    "fps",
                    default=0.0,
                    min=0.0,
                    max=240.0,
                    step=0.01,
                    advanced=True,
                    tooltip="Output frame rate. 0 keeps the source frame rate.",
                ),
                io.Float.Input(
                    "proxy_threshold",
                    default=2.1,
                    min=0.1,
                    max=1000.0,
                    step=0.1,
                    round=0.1,
                    advanced=True,
                    tooltip="Generate a preview proxy only when the source exceeds this total pixel count, in MPixels.",
                ),
                io.Float.Input(
                    "proxy_size",
                    default=0.8,
                    min=0.1,
                    max=1000.0,
                    step=0.1,
                    round=0.1,
                    advanced=True,
                    tooltip="Target total pixel count for the generated preview proxy, in MPixels.",
                ),
            ],
            outputs=[
                io.Video.Output(),
                io.Image.Output(display_name="IMAGE"),
                io.Int.Output(display_name="frame_count"),
                io.Audio.Output(display_name="audio"),
                io.Dict.Output(display_name="video_info"),
                io.AnyType.Output(display_name="proxy_video"),
            ],
        )

    @classmethod
    def execute(
        cls,
        video: str,
        keep_aspect_ratio: bool,
        multiple: int,
        start_frame: int,
        end_frame: int,
        width: int,
        height: int,
        fps: float,
        proxy_threshold: float = 2.1,
        proxy_size: float = 0.8,
    ) -> io.NodeOutput:
        _LOGGER.info("[CS Load Video] start: %s", video)
        if not folder_paths.exists_annotated_filepath(video):
            raise ValueError(f"Invalid video file: {video}")

        source_path = folder_paths.get_annotated_filepath(video)
        source_info = _read_video_info(video)
        source_fps = float(source_info["fps"])
        source_rate = Fraction(str(source_fps)).limit_denominator(100000)
        source_count = int(source_info["frames"])
        if source_count <= 0 and source_info["duration"] > 0:
            source_count = max(1, int(round(source_info["duration"] * source_fps)))
        if source_count <= 0 or source_fps <= 0:
            raise ValueError(f"Video contains no usable frame metadata: {video}")

        _LOGGER.info("[CS Load Video] stage 1/7: decoding selected source frames")
        decode_started_at = time.perf_counter()
        start = max(0, min(int(start_frame), source_count - 1))
        end = source_count - 1 if int(end_frame) < 0 else min(int(end_frame), source_count - 1)
        if end < start:
            raise ValueError(f"end_frame ({end}) must be greater than or equal to start_frame ({start})")
        selected_frame_count = end - start + 1
        start_time = float(Fraction(start, 1) / source_rate)
        selected_duration = float(Fraction(selected_frame_count, 1) / source_rate)
        _LOGGER.info(
            "[CS Load Video] source trim window: frames %d-%d -> %.6fs + %.6fs",
            start,
            end,
            start_time,
            selected_duration,
        )
        source = InputImpl.VideoFromFile(source_path, start_time=start_time, duration=selected_duration)
        components = source.get_components()
        images = components.images
        if images.ndim != 4 or images.shape[-1] not in (3, 4):
            raise ValueError("Decoded video frames must have shape [frames, height, width, 3 or 4]")
        if images.shape[0] == 0:
            raise ValueError(f"Video contains no decodable frames: {video}")
        _LOGGER.info("[CS Load Video] decoding source frames complete in %.2fs", time.perf_counter() - decode_started_at)
        _LOGGER.info(
            "[CS Load Video] selected source decoded: %d frames (requested %d), %dx%d, %.3f fps",
            images.shape[0],
            selected_frame_count,
            int(images.shape[2]),
            int(images.shape[1]),
            source_fps,
        )

        _LOGGER.info("[CS Load Video] stage 2/7: selected frame window already applied (%d frames)", images.shape[0])
        selected = images
        target_fps = source_fps if fps <= 0 else float(fps)
        _LOGGER.info("[CS Load Video] stage 3/7: FPS processing %.3f -> %.3f", source_fps, target_fps)
        selected = _resample_frames(selected, source_fps, target_fps)
        _LOGGER.info("[CS Load Video] FPS processing complete: %d frames", selected.shape[0])

        multiple = max(1, int(multiple))
        source_width = int(selected.shape[2])
        source_height = int(selected.shape[1])
        aspect_ratio = source_width / source_height
        if keep_aspect_ratio:
            if int(width) > 0:
                output_width = _round_to_multiple(width, multiple)
                output_height = _round_to_multiple(output_width / aspect_ratio, multiple)
            elif int(height) > 0:
                output_height = _round_to_multiple(height, multiple)
                output_width = _round_to_multiple(output_height * aspect_ratio, multiple)
            else:
                output_width = _round_to_multiple(source_width, multiple)
                output_height = _round_to_multiple(output_width / aspect_ratio, multiple)
        else:
            output_width = _round_to_multiple(source_width, multiple) if int(width) <= 0 else int(width)
            output_height = _round_to_multiple(source_height, multiple) if int(height) <= 0 else int(height)
        if output_width < 1 or output_height < 1:
            raise ValueError("Output width and height must be positive")
        _LOGGER.info("[CS Load Video] stage 4/7: resizing frames to %dx%d", output_width, output_height)
        selected = _resize_frames(selected, output_width, output_height)
        _LOGGER.info("[CS Load Video] frame resizing complete: %d frames", selected.shape[0])

        _LOGGER.info("[CS Load Video] stage 5/7: trimming audio")
        audio = components.audio
        _LOGGER.info("[CS Load Video] audio processing: 100%% (%s)", "available" if audio is not None else "none")
        normalized_proxy_threshold = _normalize_proxy_threshold(proxy_threshold)
        normalized_proxy_size = _normalize_proxy_size(proxy_size)
        proxy_threshold_pixels = normalized_proxy_threshold * 1_000_000
        proxy_required = output_width * output_height > proxy_threshold_pixels
        _LOGGER.info(
            "[CS Load Video] stage 6/7: proxy decision output=%dx%d, threshold=%.1f MPixels -> %s",
            output_width,
            output_height,
            normalized_proxy_threshold,
            "generate" if proxy_required else "reuse video object",
        )
        info = {
            "source_filename": str(video),
            "source_fps": source_fps,
            "source_frame_count": source_count,
            "source_duration": source_count / source_fps,
            "source_width": int(images.shape[2]),
            "source_height": int(images.shape[1]),
            "start_frame": start,
            "end_frame": end,
            "loaded_fps": target_fps,
            "loaded_frame_count": int(selected.shape[0]),
            "loaded_duration": float(selected.shape[0] / target_fps),
            "loaded_width": output_width,
            "loaded_height": output_height,
            "keep_aspect_ratio": bool(keep_aspect_ratio),
            "multiple": multiple,
            "proxy_threshold": normalized_proxy_threshold,
            "proxy_size": normalized_proxy_size,
            "proxy_video": proxy_required,
        }
        video_images = selected[..., :3] if selected.shape[-1] == 4 else selected
        output_frame_rate = Fraction(target_fps).limit_denominator(1000)
        output_video = InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=video_images,
                audio=audio,
                frame_rate=output_frame_rate,
                metadata=info,
            )
        )
        # ComfyUI's VideoFromComponents currently drops ``metadata`` when a
        # downstream node calls get_components(). Keep a CineStyle-private
        # descriptor on the VIDEO object so subtitle preview recovery can
        # still identify the source range and actual output canvas.
        try:
            output_video._cinestyle_runtime_metadata = dict(info)
        except (AttributeError, TypeError):
            pass
        if proxy_required:
            proxy_width, proxy_height = _proxy_dimensions(
                output_width,
                output_height,
                normalized_proxy_size * 1_000_000,
            )
            proxy_images = _resize_frames(video_images, proxy_width, proxy_height)
            _LOGGER.info("[CS Load Video] proxy frame resizing complete: %d frames", proxy_images.shape[0])
            proxy_video = InputImpl.VideoFromComponents(
                Types.VideoComponents(
                    images=proxy_images,
                    audio=audio,
                    frame_rate=output_frame_rate,
                    metadata=info,
                )
            )
            try:
                proxy_video._cinestyle_runtime_metadata = dict(info)
            except (AttributeError, TypeError):
                pass
        else:
            proxy_video = output_video
        _LOGGER.info("[CS Load Video] stage 7/7: complete, output frames=%d", selected.shape[0])
        return io.NodeOutput(output_video, selected, int(selected.shape[0]), audio, info, proxy_video)

    @classmethod
    def fingerprint_inputs(cls, video: str, **kwargs: Any) -> float:
        path = folder_paths.get_annotated_filepath(video)
        return os.path.getmtime(path)

    @classmethod
    def validate_inputs(cls, video: str, **kwargs: Any) -> str | bool:
        if not folder_paths.exists_annotated_filepath(video):
            return f"Invalid video file: {video}"
        return True


def _metadata_for_save(cls: type[io.ComfyNode], video: Input.Video, save_metadata: bool) -> dict[str, Any] | None:
    if not save_metadata:
        return None

    components = video.get_components()
    metadata: dict[str, Any] = dict(components.metadata or {})
    if cls.hidden.extra_pnginfo is not None:
        metadata.update(cls.hidden.extra_pnginfo)
    if cls.hidden.prompt is not None:
        metadata["prompt"] = cls.hidden.prompt
    return metadata or None


def _write_h264(
    video: Input.Video,
    path: str,
    metadata: dict[str, Any] | None,
    bitrate_mbps: float,
) -> None:
    components = video.get_components()
    images = components.images
    if images.ndim != 4 or images.shape[-1] not in (3, 4):
        raise ValueError("Video frames must have shape [frames, height, width, 3 or 4]")
    if images.shape[0] == 0:
        raise ValueError("Cannot save a video with no frames")

    height, width = int(images.shape[1]), int(images.shape[2])
    if width % 2 or height % 2:
        raise ValueError(f"H.264 output requires even dimensions, got {width}x{height}")

    rate = Fraction(components.frame_rate).limit_denominator(1000)
    with av.open(path, mode="w", options={"movflags": "use_metadata_tags+faststart"}) as container:
        if metadata:
            for key, value in metadata.items():
                container.metadata[str(key)] = value if isinstance(value, str) else json.dumps(value)

        stream = container.add_stream("h264", rate=rate)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.bit_rate = int(round(float(bitrate_mbps) * 1_000_000))
        stream.codec_context.max_b_frames = 0

        audio_stream = None
        audio = components.audio
        waveform = None
        if audio is not None:
            waveform = audio["waveform"]
            if waveform.ndim == 2:
                waveform = waveform.unsqueeze(0)
            if waveform.ndim == 3 and waveform.shape[-1] > 0:
                sample_rate = int(audio["sample_rate"])
                channels = int(waveform.shape[1])
                layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(channels, "stereo")
                audio_stream = container.add_stream("aac", rate=sample_rate, layout=layout)

        for image in images:
            frame = av.VideoFrame.from_ndarray(
                torch.clamp(image[..., :3] * 255, min=0, max=255)
                .to(device=torch.device("cpu"), dtype=torch.uint8)
                .numpy(),
                format="rgb24",
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

        if audio_stream is not None and waveform is not None:
            audio_frame = av.AudioFrame.from_ndarray(
                waveform[0].float().cpu().contiguous().numpy(),
                format="fltp",
                layout=audio_stream.layout.name,
            )
            audio_frame.sample_rate = int(audio["sample_rate"])
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)
            for packet in audio_stream.encode():
                container.mux(packet)


class CSSaveVideo(io.ComfyNode):
    """Save a ComfyUI VIDEO with optional metadata and H.264 bitrate control."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CS_Save_Video",
            search_aliases=["save video cinestyle", "export video bitrate", "h264 bitrate"],
            display_name="CS Save Video",
            category="😺dzNodes/CineStyle",
            essentials_category="Video Tools",
            description="Save a VIDEO with an optional metadata flag and explicit H.264 bitrate.",
            inputs=[
                io.Video.Input("video", tooltip="The video to save."),
                io.String.Input(
                    "filename_prefix",
                    default="video/ComfyUI",
                    tooltip="Output filename prefix; date and node widget formatting are supported.",
                ),
                io.Combo.Input(
                    "format",
                    options=Types.VideoContainer.as_input(),
                    default="auto",
                    tooltip="The container format used for the output video.",
                ),
                io.DynamicCombo.Input(
                    "codec",
                    options=[
                        io.DynamicCombo.Option(
                            "h264",
                            [
                                io.Float.Input(
                                    "bitrate",
                                    display_name="H.264 bitrate (Mbps)",
                                    default=8.0,
                                    min=1.0,
                                    max=160.0,
                                    step=0.1,
                                    round=0.1,
                                    tooltip="H.264 target bitrate in Mbps. Official guidance spans about 1.0 Mbps for low resolution to 160.0 Mbps for 8K high-frame-rate video.",
                                ),
                            ],
                        ),
                        io.DynamicCombo.Option("auto", []),
                    ],
                    tooltip="The video codec. H.264 is the default and exposes a target bitrate control.",
                ),
                io.Boolean.Input(
                    "save_metadata",
                    default=False,
                    tooltip="When enabled, write workflow and source metadata like the official Save Video node.",
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[io.Video.Output("video")],
        )

    @classmethod
    def execute(
        cls,
        video: Input.Video,
        filename_prefix: str,
        format: str,
        codec: io.DynamicCombo.Type,
        save_metadata: bool,
    ) -> io.NodeOutput:
        codec_name = codec.get("codec", "h264") if isinstance(codec, dict) else str(codec or "h264")
        bitrate_mbps = float(codec.get("bitrate", 8.0)) if isinstance(codec, dict) else 8.0
        width, height = video.get_dimensions()
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            width,
            height,
        )
        extension = Types.VideoContainer.get_extension(format)
        file = f"{filename}_{counter:05}_.{extension}"
        output_path = os.path.join(full_output_folder, file)
        metadata = _metadata_for_save(cls, video, bool(save_metadata))

        if codec_name == "h264":
            _write_h264(video, output_path, metadata, bitrate_mbps)
        elif codec_name == "auto":
            if save_metadata:
                video.save_to(output_path, format=Types.VideoContainer(format), codec="auto", metadata=metadata)
            else:
                _write_h264(video, output_path, None, bitrate_mbps)
        else:
            raise ValueError(f"Unsupported video codec: {codec_name}")

        return io.NodeOutput(video, ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]))


class CineStyleVideoExtension(ComfyExtension):
    @override
    async def on_load(self) -> None:
        global _ROUTE_REGISTERED
        if _ROUTE_REGISTERED:
            return
        from server import PromptServer

        server_instance = getattr(PromptServer, "instance", None)
        if server_instance is not None:
            server_instance.routes.get("/cinestyle/video-info")(_video_info_route)
            server_instance.routes.get("/cinestyle/video-source")(_video_source_route)
            server_instance.routes.get("/cinestyle/video-proxy")(_video_proxy_route)
            server_instance.routes.get("/cinestyle/video-proxy-progress")(_video_proxy_progress_route)
            _ROUTE_REGISTERED = True

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CSLoadVideo, CSSaveVideo]


async def comfy_entrypoint() -> CineStyleVideoExtension:
    return CineStyleVideoExtension()


WEB_DIRECTORY = "./web"
