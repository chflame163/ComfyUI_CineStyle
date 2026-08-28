"""Preview arbitrary ComfyUI values as media and structured debug text."""

from __future__ import annotations

import math
import numbers
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
import torch.nn.functional as F
from typing_extensions import override

import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, Input, InputImpl, io, ui


_CATEGORY = "😺dzNodes/CineStyle"
_MAX_PREVIEW_PIXELS = 1_000_000
_MAX_PREVIEW_FPS = 25.0
_MAX_TEXT_CHARS = 200_000
_MAX_DEBUG_ITEMS = 200
_WAVEFORM_BARS = 160
_VIDEO_CACHE_SUBFOLDER = "cinestyle_preview_any"


def _safe_type_name(value: Any) -> str:
    value_type = type(value)
    module = str(getattr(value_type, "__module__", "") or "")
    name = str(getattr(value_type, "__qualname__", value_type.__name__) or value_type.__name__)
    return name if module in {"", "builtins"} else f"{module}.{name}"


def _shape_text(value: torch.Tensor) -> str:
    return "[" + ", ".join(str(int(item)) for item in value.shape) + "]"


def _aspect_ratio(width: int, height: int) -> str:
    divisor = math.gcd(max(1, int(width)), max(1, int(height)))
    return f"{width // divisor}:{height // divisor} ({width / max(1, height):.4f})"


def _format_duration(seconds: float) -> str:
    safe = max(0.0, float(seconds))
    hours = int(safe // 3600)
    minutes = int((safe % 3600) // 60)
    remainder = safe % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remainder:06.3f} ({safe:.3f} s)"
    return f"{minutes:02d}:{remainder:06.3f} ({safe:.3f} s)"


def _truncate_text(value: str) -> str:
    if len(value) <= _MAX_TEXT_CHARS:
        return value
    omitted = len(value) - _MAX_TEXT_CHARS
    return f"{value[:_MAX_TEXT_CHARS]}\n\n... truncated {omitted} characters"


def _debug_number(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Number):
        return str(value)
    return None


def _debug_list(value: list[Any]) -> list[str]:
    lines = ["Type: LIST", f"Length: {len(value)}"]
    for index, item in enumerate(value[:_MAX_DEBUG_ITEMS]):
        number = _debug_number(item)
        lines.append(f"[{index}] {number}" if number is not None else f"[{index}] <{_safe_type_name(item)}>")
    if len(value) > _MAX_DEBUG_ITEMS:
        lines.append(f"... {len(value) - _MAX_DEBUG_ITEMS} more items")
    return lines


def _debug_dict(value: dict[Any, Any]) -> list[str]:
    lines = ["Type: DICT", f"Length: {len(value)}"]
    for index, (key, item) in enumerate(value.items()):
        if index >= _MAX_DEBUG_ITEMS:
            lines.append(f"... {len(value) - _MAX_DEBUG_ITEMS} more items")
            break
        if isinstance(key, (str, int, float, bool)):
            key_text = repr(key)
        else:
            key_text = f"<{_safe_type_name(key)}>"
        number = _debug_number(item)
        lines.append(f"{key_text}: {number}" if number is not None else f"{key_text}: <{_safe_type_name(item)}>")
    return lines


def _tensor_statistics(value: torch.Tensor) -> str:
    flat = value.detach().reshape(-1)
    if flat.numel() == 0:
        return "empty"
    if flat.numel() > 1_000_000:
        step = max(1, flat.numel() // 1_000_000)
        flat = flat[::step]
    sample = flat.to(dtype=torch.float32)
    finite = sample[torch.isfinite(sample)]
    if finite.numel() == 0:
        return "no finite values"
    return f"min={finite.min().item():.6g}, max={finite.max().item():.6g}, mean={finite.mean().item():.6g}, std={finite.std(correction=0).item():.6g}"


def _preview_dimensions(width: int, height: int) -> tuple[int, int]:
    width = max(1, int(width))
    height = max(1, int(height))
    scale = min(1.0, math.sqrt(_MAX_PREVIEW_PIXELS / float(width * height)))
    preview_width = max(2, int(math.floor(width * scale)))
    preview_height = max(2, int(math.floor(height * scale)))
    preview_width -= preview_width % 2
    preview_height -= preview_height % 2
    while preview_width * preview_height > _MAX_PREVIEW_PIXELS:
        if preview_width >= preview_height and preview_width > 2:
            preview_width -= 2
        elif preview_height > 2:
            preview_height -= 2
        else:
            break
    return preview_width, preview_height


def _safe_fps(value: Any) -> float:
    try:
        fps = float(value)
    except (TypeError, ValueError, OverflowError):
        return 24.0
    return fps if math.isfinite(fps) and fps > 0 else 24.0


def _rewind_source(source: Any) -> Any:
    if hasattr(source, "seek"):
        source.seek(0)
    return source


def _stream_rotation(stream: av.VideoStream) -> int:
    try:
        return int(round(float(stream.metadata.get("rotate", 0)) / 90.0)) % 4
    except (TypeError, ValueError):
        return 0


def _display_dimensions(stream: av.VideoStream) -> tuple[int, int]:
    width, height = int(stream.width), int(stream.height)
    return (height, width) if _stream_rotation(stream) % 2 else (width, height)


def _audio_stream_info(stream: av.AudioStream | None) -> dict[str, Any]:
    if stream is None or stream.codec_context is None:
        return {"has_audio": False}
    context = stream.codec_context
    codec = context.codec.name if context.codec is not None else None
    sample_format = context.format.name if context.format is not None else None
    layout = context.layout.name if context.layout is not None else None
    return {
        "has_audio": True,
        "audio_codec": codec,
        "audio_sample_format": sample_format,
        "audio_sample_rate": int(context.sample_rate or 0),
        "audio_channels": int(context.channels or 0),
        "audio_layout": layout,
        "audio_stream_index": int(stream.index),
    }


def _first_audio_stream(container: av.InputContainer) -> av.AudioStream | None:
    return next((stream for stream in reversed(container.streams.audio) if stream.codec_context is not None), None)


def _add_video_stream(container: av.OutputContainer, fps: float, width: int, height: int) -> av.VideoStream:
    rate = Fraction(fps).limit_denominator(1000)
    stream = container.add_stream("libx264", rate=rate)
    stream.options = {"preset": "ultrafast", "crf": "23"}
    stream.codec_context.max_b_frames = 0
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    return stream


def _add_audio_stream(container: av.OutputContainer, sample_rate: int, channels: int) -> tuple[av.AudioStream, str]:
    layout = "mono" if channels <= 1 else "stereo"
    stream = container.add_stream("aac", rate=sample_rate, layout=layout)
    stream.bit_rate = 128_000
    return stream, layout


def _mux_video_array(
    container: av.OutputContainer,
    stream: av.VideoStream,
    array: np.ndarray,
    index: int,
    fps: float,
    width: int,
    height: int,
) -> None:
    frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(array), format="rgb24")
    frame = frame.reformat(width=width, height=height, format="yuv420p")
    frame.pts = index
    frame.time_base = Fraction(1, 1) / Fraction(fps).limit_denominator(1000)
    for packet in stream.encode(frame):
        container.mux(packet)


def _write_audio_chunk(
    container: av.OutputContainer,
    stream: av.AudioStream,
    layout: str,
    sample_rate: int,
    values: np.ndarray,
    pts: int,
) -> int:
    if values.ndim != 2 or values.shape[1] == 0:
        return pts
    values = np.ascontiguousarray(values, dtype=np.float32)
    frame = av.AudioFrame.from_ndarray(values, format="fltp", layout=layout)
    frame.sample_rate = sample_rate
    frame.pts = pts
    frame.time_base = Fraction(1, sample_rate)
    for packet in stream.encode(frame):
        container.mux(packet)
    return pts + int(values.shape[1])


def _tensor_audio(audio: Any) -> tuple[torch.Tensor, int, int] | None:
    if not isinstance(audio, dict) or not isinstance(audio.get("waveform"), torch.Tensor):
        return None
    waveform = audio["waveform"]
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    try:
        sample_rate = int(audio.get("sample_rate", 0) or 0)
    except (TypeError, ValueError):
        return None
    if waveform.ndim != 3 or waveform.numel() == 0 or sample_rate <= 0:
        return None
    channels = int(waveform.shape[1])
    values = waveform[0].detach().to(device="cpu", dtype=torch.float32)
    if channels > 2:
        values = values[:2]
        channels = 2
    return values.contiguous(), sample_rate, channels


def _encode_tensor_audio(
    container: av.OutputContainer,
    stream: av.AudioStream,
    layout: str,
    values: torch.Tensor,
    sample_rate: int,
    duration: float,
) -> None:
    sample_count = min(int(values.shape[-1]), max(0, int(round(duration * sample_rate))))
    chunk_size = max(sample_rate, sample_rate * 10)
    pts = 0
    for start in range(0, sample_count, chunk_size):
        chunk = values[:, start : min(sample_count, start + chunk_size)].numpy()
        pts = _write_audio_chunk(container, stream, layout, sample_rate, chunk, pts)
    for packet in stream.encode():
        container.mux(packet)


def _source_audio_stream(container: av.InputContainer, stream_index: int) -> av.AudioStream | None:
    return next((stream for stream in container.streams.audio if int(stream.index) == int(stream_index)), None)


def _encode_source_audio(
    source: Any,
    source_stream_index: int,
    output: av.OutputContainer,
    output_stream: av.AudioStream,
    layout: str,
    sample_rate: int,
    start_time: float,
    duration: float,
) -> bool:
    limit = max(0, int(round(duration * sample_rate)))
    if limit == 0:
        return False
    written = 0
    wrote_audio = False
    with av.open(_rewind_source(source), mode="r") as input_container:
        input_stream = _source_audio_stream(input_container, source_stream_index)
        if input_stream is None:
            return False
        if start_time > 0:
            try:
                input_container.seek(int(start_time * av.time_base), backward=True)
            except (av.error.FFmpegError, ValueError):
                pass
        resampler = av.audio.resampler.AudioResampler(format="fltp", layout=layout, rate=sample_rate)

        def consume(converted: av.AudioFrame, fallback_offset: int) -> int:
            nonlocal written, wrote_audio
            values = converted.to_ndarray()
            if values.ndim != 2 or values.shape[1] == 0:
                return fallback_offset
            if converted.pts is not None and converted.time_base is not None:
                offset = int(round((float(converted.pts * converted.time_base) - start_time) * sample_rate))
            else:
                offset = fallback_offset
            if offset + values.shape[1] <= 0 or offset >= limit:
                return max(fallback_offset, offset + int(values.shape[1]))
            if offset < 0:
                values = values[:, -offset:]
                offset = 0
            if offset > written:
                silence_count = min(offset - written, limit - written)
                if silence_count > 0:
                    silence = np.zeros((1 if layout == "mono" else 2, silence_count), dtype=np.float32)
                    written = _write_audio_chunk(output, output_stream, layout, sample_rate, silence, written)
            if offset < written:
                overlap = written - offset
                if overlap >= values.shape[1]:
                    return max(fallback_offset, offset + int(values.shape[1]))
                values = values[:, overlap:]
            values = values[:, : max(0, limit - written)]
            if values.shape[1] > 0:
                written = _write_audio_chunk(output, output_stream, layout, sample_rate, values, written)
                wrote_audio = True
            return max(fallback_offset, offset + int(values.shape[1]))

        fallback = 0
        for decoded in input_container.decode(input_stream):
            for converted in resampler.resample(decoded):
                fallback = consume(converted, fallback)
                if written >= limit:
                    break
            if written >= limit:
                break
        if written < limit:
            for converted in resampler.resample(None):
                fallback = consume(converted, fallback)
                if written >= limit:
                    break
    for packet in output_stream.encode():
        output.mux(packet)
    return wrote_audio


@dataclass
class _VideoCacheEntry:
    key: str
    path: Path
    created: float
    size: int


class _PreviewVideoCache:
    def __init__(self, max_entries: int = 24, max_bytes: int = 8 * 1024**3):
        self.root = Path(folder_paths.get_temp_directory()) / _VIDEO_CACHE_SUBFOLDER
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.entries: dict[str, _VideoCacheEntry] = {}
        self.lock = threading.RLock()

    def new_paths(self, node_id: str) -> tuple[Path, Path]:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(node_id or "preview")) or "preview"
        name = f"{safe}_{uuid.uuid4().hex}.mp4"
        final_path = self.root / name
        return final_path.with_name(f".{name}.tmp.mp4"), final_path

    def register(self, key: str, path: Path) -> None:
        entry = _VideoCacheEntry(key, path, time.time(), int(path.stat().st_size))
        with self.lock:
            previous = self.entries.pop(key, None)
            self.entries[key] = entry
            ordered = list(self.entries.values())
            total = sum(item.size for item in ordered)
            removed: list[_VideoCacheEntry] = []
            while len(ordered) > self.max_entries or (len(ordered) > 1 and total > self.max_bytes):
                oldest = ordered.pop(0)
                self.entries.pop(oldest.key, None)
                total -= oldest.size
                removed.append(oldest)
        for item in [previous, *removed]:
            if item is not None and item.path != path:
                try:
                    item.path.unlink(missing_ok=True)
                except OSError:
                    pass

    def clear_node(self, node_id: str) -> None:
        prefix = f"{node_id}:"
        with self.lock:
            keys = [key for key in self.entries if key.startswith(prefix)]
            removed = [self.entries.pop(key) for key in keys]
        for item in removed:
            try:
                item.path.unlink(missing_ok=True)
            except OSError:
                pass


_VIDEO_CACHE: _PreviewVideoCache | None = None


def _video_cache() -> _PreviewVideoCache:
    global _VIDEO_CACHE
    if _VIDEO_CACHE is None:
        _VIDEO_CACHE = _PreviewVideoCache()
    return _VIDEO_CACHE


def _video_info_lines(info: dict[str, Any]) -> list[str]:
    lines = [
        "Type: VIDEO",
        f"Original size: {info['original_width']} x {info['original_height']}",
        f"Preview size: {info['preview_width']} x {info['preview_height']}",
        f"Aspect ratio: {_aspect_ratio(info['original_width'], info['original_height'])}",
        f"Original frame rate: {info['original_fps']:.4f} fps",
        f"Preview frame rate: {info['preview_fps']:.4f} fps",
        f"Original frames: {info['original_frames']}",
        f"Preview frames: {info['preview_frames']}",
        f"Original duration: {_format_duration(info['original_duration'])}",
        f"Preview duration: {_format_duration(info['preview_duration'])}",
        f"Container: {info.get('container') or 'tensor components'}",
        f"Video codec: {info.get('video_codec') or 'tensor RGB'}",
        f"Colour space: {info.get('colour_space') or 'unknown'}",
        f"Bit depth: {info.get('bit_depth') or 'unknown'}",
    ]
    if info.get("has_audio"):
        source_audio = info.get("audio_codec") or "PCM tensor"
        sample_format = info.get("audio_sample_format") or "float tensor"
        lines.extend(
            [
                f"Audio: {source_audio}, {sample_format}, {info.get('audio_sample_rate', 0)} Hz, "
                f"{info.get('audio_channels', 0)} channel(s), {info.get('audio_layout') or 'unknown layout'}",
                f"Preview audio: {'AAC' if info.get('preview_has_audio') else 'unavailable'}",
            ]
        )
    else:
        lines.append("Audio: none")
    return lines


def _encode_components_video(video: InputImpl.VideoFromComponents, path: Path) -> dict[str, Any]:
    components = video.get_components()
    images = components.images
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] == 0:
        raise ValueError("VIDEO components contain no image frames")
    original_frames = int(images.shape[0])
    original_height, original_width = int(images.shape[1]), int(images.shape[2])
    original_fps = _safe_fps(components.frame_rate)
    preview_fps = min(original_fps, _MAX_PREVIEW_FPS)
    original_duration = original_frames / original_fps
    preview_frames = max(1, int(math.ceil(original_duration * preview_fps - 1e-9)))
    indices = torch.linspace(0, original_frames - 1, preview_frames).round().to(torch.long).tolist()
    preview_width, preview_height = _preview_dimensions(original_width, original_height)
    prepared_audio = _tensor_audio(components.audio)
    preview_duration = preview_frames / preview_fps
    progress = comfy.utils.ProgressBar(preview_frames)

    with av.open(str(path), mode="w", format="mp4", options={"movflags": "+faststart"}) as output:
        video_stream = _add_video_stream(output, preview_fps, preview_width, preview_height)
        audio_stream = None
        audio_layout = None
        if prepared_audio is not None:
            _, sample_rate, channels = prepared_audio
            audio_stream, audio_layout = _add_audio_stream(output, sample_rate, channels)
        for output_index, source_index in enumerate(indices):
            frame = images[source_index]
            if frame.shape[-1] == 1:
                frame = frame.expand(*frame.shape[:-1], 3)
            frame = frame[..., :3].detach().to(dtype=torch.float32).clamp(0.0, 1.0)
            if tuple(frame.shape[:2]) != (preview_height, preview_width):
                frame = F.interpolate(
                    frame.movedim(-1, 0).unsqueeze(0),
                    size=(preview_height, preview_width),
                    mode="bilinear",
                    align_corners=False,
                )[0].movedim(0, -1)
            array = frame.mul(255.0).round().to(device="cpu", dtype=torch.uint8).numpy()
            _mux_video_array(output, video_stream, array, output_index, preview_fps, preview_width, preview_height)
            progress.update(1)
        for packet in video_stream.encode():
            output.mux(packet)
        if audio_stream is not None and prepared_audio is not None and audio_layout is not None:
            values, sample_rate, _ = prepared_audio
            _encode_tensor_audio(output, audio_stream, audio_layout, values, sample_rate, preview_duration)

    return {
        "original_width": original_width,
        "original_height": original_height,
        "preview_width": preview_width,
        "preview_height": preview_height,
        "original_fps": original_fps,
        "preview_fps": preview_fps,
        "original_frames": original_frames,
        "preview_frames": preview_frames,
        "original_duration": original_duration,
        "preview_duration": preview_duration,
        "container": None,
        "video_codec": None,
        "colour_space": video.get_color_space(),
        "bit_depth": video.get_bit_depth(),
        "has_audio": prepared_audio is not None,
        "audio_codec": None,
        "audio_sample_format": str(prepared_audio[0].dtype).replace("torch.", "") if prepared_audio else None,
        "audio_sample_rate": prepared_audio[1] if prepared_audio else 0,
        "audio_channels": prepared_audio[2] if prepared_audio else 0,
        "audio_layout": "mono" if prepared_audio and prepared_audio[2] == 1 else "stereo" if prepared_audio else None,
        "preview_has_audio": prepared_audio is not None,
    }


def _encode_stream_video(video: Input.Video, path: Path) -> dict[str, Any]:
    source = video.get_stream_source()
    start_time, requested_duration = video.get_active_trim_window()
    start_time = max(0.0, float(start_time or 0.0))
    requested_duration = max(0.0, float(requested_duration or 0.0))

    with av.open(_rewind_source(source), mode="r") as input_container:
        if not input_container.streams.video:
            raise ValueError("VIDEO contains no decodable video stream")
        input_stream = input_container.streams.video[0]
        original_width, original_height = _display_dimensions(input_stream)
        original_fps = _safe_fps(input_stream.average_rate)
        preview_fps = min(original_fps, _MAX_PREVIEW_FPS)
        preview_width, preview_height = _preview_dimensions(original_width, original_height)
        raw_duration = 0.0
        if input_container.duration:
            raw_duration = float(input_container.duration / av.time_base)
        elif input_stream.duration is not None and input_stream.time_base is not None:
            raw_duration = float(input_stream.duration * input_stream.time_base)
        duration_from_start = max(0.0, raw_duration - start_time) if raw_duration else 0.0
        expected_duration = min(requested_duration, duration_from_start) if requested_duration and duration_from_start else requested_duration or duration_from_start
        audio_info = _audio_stream_info(_first_audio_stream(input_container))
        container_name = input_container.format.name if input_container.format else None
        video_codec = input_stream.codec.name if input_stream.codec is not None else None
        if isinstance(source, (str, os.PathLike)):
            input_bit_depth = video.get_bit_depth()
            colour_space = video.get_color_space()
        else:
            input_bit_depth = max(
                (int(component.bits) for component in (input_stream.format.components if input_stream.format else [])),
                default=8,
            )
            colour_space = "auto"
        expected_source_frames = int(math.ceil(expected_duration * original_fps)) if expected_duration else int(input_stream.frames or 0)
        progress = comfy.utils.ProgressBar(max(1, expected_source_frames))

        with av.open(str(path), mode="w", format="mp4", options={"movflags": "+faststart"}) as output:
            output_video = _add_video_stream(output, preview_fps, preview_width, preview_height)
            output_audio = None
            output_audio_layout = None
            if audio_info.get("has_audio") and audio_info.get("audio_sample_rate", 0) > 0:
                output_audio, output_audio_layout = _add_audio_stream(
                    output,
                    int(audio_info["audio_sample_rate"]),
                    int(audio_info.get("audio_channels", 1) or 1),
                )
            if start_time > 0:
                try:
                    input_container.seek(int(start_time / input_stream.time_base), stream=input_stream, backward=True)
                except (av.error.FFmpegError, ValueError):
                    pass

            source_frames = 0
            output_frames = 0
            next_output_time = 0.0
            last_array = None
            last_local_time = 0.0
            fallback_frame = 0
            source_interval = 1.0 / original_fps
            end_time = start_time + requested_duration if requested_duration else None
            for decoded in input_container.decode(input_stream):
                if decoded.pts is not None and input_stream.time_base is not None:
                    frame_time = float(decoded.pts * input_stream.time_base)
                else:
                    frame_time = start_time + fallback_frame * source_interval
                fallback_frame += 1
                if frame_time + source_interval * 0.5 < start_time:
                    continue
                if end_time is not None and frame_time >= end_time:
                    break
                local_time = max(0.0, frame_time - start_time)
                source_frames += 1
                progress.update(1)
                rotation = int(round(float(getattr(decoded, "rotation", 0) or 0) / 90.0)) % 4
                if rotation == 0:
                    rotation = _stream_rotation(input_stream)
                array = decoded.to_ndarray(format="rgb24")
                if rotation:
                    array = np.rot90(array, k=rotation, axes=(0, 1)).copy()
                while next_output_time <= local_time + source_interval * 0.5:
                    _mux_video_array(output, output_video, array, output_frames, preview_fps, preview_width, preview_height)
                    output_frames += 1
                    next_output_time = output_frames / preview_fps
                last_array = array
                last_local_time = local_time

            if source_frames == 0 or last_array is None:
                raise ValueError("VIDEO contains no frames in the active trim window")
            if expected_duration <= 0:
                expected_duration = last_local_time + source_interval
            expected_output_frames = max(1, int(math.ceil(expected_duration * preview_fps - 1e-9)))
            while output_frames < expected_output_frames:
                _mux_video_array(output, output_video, last_array, output_frames, preview_fps, preview_width, preview_height)
                output_frames += 1
            for packet in output_video.encode():
                output.mux(packet)

            preview_duration = output_frames / preview_fps
            preview_has_audio = False
            if output_audio is not None and output_audio_layout is not None:
                preview_has_audio = _encode_source_audio(
                    source,
                    int(audio_info["audio_stream_index"]),
                    output,
                    output_audio,
                    output_audio_layout,
                    int(audio_info["audio_sample_rate"]),
                    start_time,
                    preview_duration,
                )

    return {
        "original_width": original_width,
        "original_height": original_height,
        "preview_width": preview_width,
        "preview_height": preview_height,
        "original_fps": original_fps,
        "preview_fps": preview_fps,
        "original_frames": source_frames,
        "preview_frames": output_frames,
        "original_duration": expected_duration,
        "preview_duration": preview_duration,
        "container": container_name,
        "video_codec": video_codec,
        "colour_space": colour_space,
        "bit_depth": input_bit_depth,
        **audio_info,
        "preview_has_audio": preview_has_audio,
    }


def _create_video_preview(video: Input.Video, node_id: str, item_index: int) -> tuple[list[dict[str, Any]], list[str]]:
    cache = _video_cache()
    temporary_path, final_path = cache.new_paths(node_id)
    try:
        if isinstance(video, InputImpl.VideoFromComponents):
            info = _encode_components_video(video, temporary_path)
        else:
            info = _encode_stream_video(video, temporary_path)
        os.replace(temporary_path, final_path)
        cache.register(f"{node_id}:{item_index}", final_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        raise
    saved = ui.SavedResult(final_path.name, _VIDEO_CACHE_SUBFOLDER, io.FolderType.temp)
    return ui.PreviewVideo([saved]).as_dict()["images"], _video_info_lines(info)


def _audio_info_lines(audio: dict[str, Any]) -> list[str]:
    waveform = audio["waveform"]
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    sample_rate = int(audio["sample_rate"])
    samples = int(waveform.shape[-1])
    channels = int(waveform.shape[1])
    batch = int(waveform.shape[0])
    sample = waveform.detach().to(dtype=torch.float32)
    peak = sample.abs().max().item() if sample.numel() else 0.0
    rms = sample.square().mean().sqrt().item() if sample.numel() else 0.0
    return [
        "Type: AUDIO",
        f"Waveform shape: {_shape_text(waveform)}",
        f"Dtype: {str(waveform.dtype).replace('torch.', '')}",
        f"Device: {waveform.device}",
        f"Batch: {batch}",
        f"Channels: {channels}",
        f"Sample rate: {sample_rate} Hz",
        f"Samples per channel: {samples}",
        f"Duration: {_format_duration(samples / sample_rate)}",
        f"Peak amplitude: {peak:.6g}",
        f"RMS: {rms:.6g}",
        "Source format: float PCM tensor",
        "Preview format: FLAC",
    ]


def _audio_waveform(audio: dict[str, Any]) -> list[float]:
    waveform = audio["waveform"]
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    mono = waveform[0].detach().to(dtype=torch.float32).abs().mean(dim=0)
    if mono.numel() == 0:
        return []
    bars = F.adaptive_max_pool1d(mono.reshape(1, 1, -1), _WAVEFORM_BARS).flatten()
    peak = bars.max()
    if torch.isfinite(peak) and peak.item() > 0:
        bars = bars / peak
    return bars.clamp(0.0, 1.0).to(device="cpu").tolist()


def _audio_for_preview(audio: dict[str, Any]) -> dict[str, Any]:
    waveform = audio["waveform"]
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[1] > 2:
        waveform = waveform[:, :2]
    return {"waveform": waveform, "sample_rate": int(audio["sample_rate"])}


def _valid_audio(value: Any) -> bool:
    if not isinstance(value, dict) or "waveform" not in value or "sample_rate" not in value:
        return False
    waveform = value.get("waveform")
    try:
        sample_rate = int(value.get("sample_rate", 0) or 0)
    except (TypeError, ValueError):
        return False
    return isinstance(waveform, torch.Tensor) and waveform.ndim in {2, 3} and waveform.numel() > 0 and sample_rate > 0


def _valid_latent(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("samples"), torch.Tensor) and value["samples"].ndim in {4, 5}


def _latent_info_lines(value: dict[str, Any]) -> list[str]:
    samples = value["samples"]
    lines = [
        "Type: LATENT",
        f"Samples shape: {_shape_text(samples)}",
        f"Dtype: {str(samples.dtype).replace('torch.', '')}",
        f"Device: {samples.device}",
        f"Batch: {int(samples.shape[0])}",
        f"Channels: {int(samples.shape[1])}",
        f"Dimensions: {'video/3D' if samples.ndim == 5 else 'image/2D'}",
        f"Statistics: {_tensor_statistics(samples)}",
        f"Keys: {', '.join(str(key) for key in value.keys())}",
    ]
    noise_mask = value.get("noise_mask")
    if isinstance(noise_mask, torch.Tensor):
        lines.append(f"Noise mask shape: {_shape_text(noise_mask)}")
    batch_index = value.get("batch_index")
    if isinstance(batch_index, list):
        lines.append(f"Batch index length: {len(batch_index)}")
    if "type" in value:
        lines.append(f"Latent type tag: {value['type']}")
    return lines


def _image_info_lines(value: torch.Tensor, kind: str) -> list[str]:
    if kind == "MASK":
        batch, height, width = map(int, value.shape)
        channels = 1
        mode = "L"
    else:
        batch, height, width, channels = map(int, value.shape)
        mode = {1: "L", 3: "RGB", 4: "RGBA"}.get(channels, f"{channels} channels")
    return [
        f"Type: {kind}",
        f"Tensor shape: {_shape_text(value)}",
        f"Image size: {width} x {height}",
        f"Batch: {batch}",
        f"Channels: {channels}",
        f"Channel mode: {mode}",
        f"Dtype: {str(value.dtype).replace('torch.', '')}",
        f"Device: {value.device}",
    ]


@dataclass
class _ItemPreview:
    kind: str
    lines: list[str]
    images: list[dict[str, Any]] = field(default_factory=list)
    audio: list[dict[str, Any]] = field(default_factory=list)
    waveform: list[float] = field(default_factory=list)


class _PreviewContext:
    def __init__(self, node_cls: type[io.ComfyNode], node_id: str):
        self.node_cls = node_cls
        self.node_id = node_id

    def preview(self, value: Any, item_index: int, allow_video: bool, allow_audio: bool) -> _ItemPreview:
        if isinstance(value, Input.Video):
            if not allow_video:
                return _ItemPreview("video", ["Type: VIDEO", "Preview omitted: only the first VIDEO item is displayed."])
            try:
                images, lines = _create_video_preview(value, self.node_id, item_index)
                return _ItemPreview("video", lines, images=images)
            except Exception as exc:
                return _ItemPreview("error", ["Unable to parse data type", "Type: VIDEO", f"Error: {exc}"])

        if isinstance(value, torch.Tensor):
            try:
                if value.ndim == 4 and int(value.shape[-1]) in {1, 3, 4}:
                    display = value.expand(*value.shape[:-1], 3) if int(value.shape[-1]) == 1 else value
                    images = ui.PreviewImage(display, cls=self.node_cls).as_dict()["images"]
                    return _ItemPreview("image", _image_info_lines(value, "IMAGE"), images=images)
                if value.ndim == 3:
                    images = ui.PreviewMask(value, cls=self.node_cls).as_dict()["images"]
                    return _ItemPreview("image", _image_info_lines(value, "MASK"), images=images)
            except Exception as exc:
                return _ItemPreview("error", ["Unable to parse data type", f"Python type: {_safe_type_name(value)}", f"Error: {exc}"])
            return _ItemPreview(
                "error",
                ["Unable to parse data type", f"Python type: {_safe_type_name(value)}", f"Tensor shape: {_shape_text(value)}"],
            )

        if isinstance(value, dict) and ("waveform" in value or "sample_rate" in value):
            if not _valid_audio(value):
                return _ItemPreview("error", ["Unable to parse data type", "Candidate type: AUDIO", f"Python type: {_safe_type_name(value)}"])
            if not allow_audio:
                return _ItemPreview("audio", ["Type: AUDIO", "Preview omitted: only the first AUDIO item is displayed."])
            try:
                audio = ui.PreviewAudio(_audio_for_preview(value), cls=self.node_cls).as_dict()["audio"]
                return _ItemPreview("audio", _audio_info_lines(value), audio=audio, waveform=_audio_waveform(value))
            except Exception as exc:
                return _ItemPreview("error", ["Unable to parse data type", "Candidate type: AUDIO", f"Error: {exc}"])

        if isinstance(value, dict) and "samples" in value:
            if not _valid_latent(value):
                return _ItemPreview("error", ["Unable to parse data type", "Candidate type: LATENT", f"Python type: {_safe_type_name(value)}"])
            return _ItemPreview("latent", _latent_info_lines(value))

        if isinstance(value, str):
            return _ItemPreview("text", ["Type: STRING", "Value:", _truncate_text(value)])
        if isinstance(value, bool):
            return _ItemPreview("text", ["Type: BOOL", f"Value: {value}"])
        if isinstance(value, int):
            return _ItemPreview("text", ["Type: INT", f"Value: {value}"])
        if isinstance(value, float):
            return _ItemPreview("text", ["Type: FLOAT", f"Value: {value}"])
        if isinstance(value, list):
            return _ItemPreview("text", _debug_list(value))
        if isinstance(value, dict):
            return _ItemPreview("text", _debug_dict(value))
        if value is None:
            return _ItemPreview("unsupported", ["Data type: NoneType", "Preview unavailable."])
        return _ItemPreview("unsupported", [f"Data type: {_safe_type_name(value)}", "Preview unavailable."])


def _upstream_output_is_list(node_cls: type[io.ComfyNode]) -> bool:
    prompt = getattr(getattr(node_cls, "hidden", None), "prompt", None)
    node_id = str(getattr(getattr(node_cls, "hidden", None), "unique_id", "") or "")
    if not isinstance(prompt, dict) or not node_id:
        return False
    current = prompt.get(node_id)
    if not isinstance(current, dict):
        return False
    link = (current.get("inputs") or {}).get("source")
    if not isinstance(link, (list, tuple)) or len(link) < 2:
        return False
    upstream = prompt.get(str(link[0]))
    if not isinstance(upstream, dict):
        return False
    try:
        import nodes

        upstream_class = nodes.NODE_CLASS_MAPPINGS.get(str(upstream.get("class_type") or ""))
        if upstream_class is None:
            return False
        output_index = int(link[1])
        output_is_list = getattr(upstream_class, "OUTPUT_IS_LIST", ())
        return output_index < len(output_is_list) and bool(output_is_list[output_index])
    except (AttributeError, IndexError, TypeError, ValueError):
        return False


class CSPreviewAny(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CS_Preview_Any",
            display_name="CS Preview Any",
            category=_CATEGORY,
            essentials_category="Utilities",
            description="Preview supported media and inspect arbitrary ComfyUI values; latent values are shown as metadata only.",
            search_aliases=["preview any", "inspect any", "debug any", "show any"],
            inputs=[
                io.AnyType.Input("source", tooltip="Any ComfyUI value, including OUTPUT_IS_LIST values."),
            ],
            hidden=[io.Hidden.unique_id, io.Hidden.prompt],
            outputs=[],
            is_input_list=True,
            is_output_node=True,
        )

    @classmethod
    def execute(cls, source: list[Any]) -> io.NodeOutput:
        node_id = str(getattr(cls.hidden, "unique_id", "") or "preview")
        _video_cache().clear_node(node_id)
        context = _PreviewContext(cls, node_id)
        declared_list = _upstream_output_is_list(cls)
        is_output_list = declared_list or len(source) != 1
        values = list(source) if is_output_list else [source[0] if source else None]

        previews: list[_ItemPreview] = []
        video_seen = False
        audio_seen = False
        for index, value in enumerate(values):
            preview = context.preview(value, index, allow_video=not video_seen, allow_audio=not audio_seen)
            if isinstance(value, Input.Video):
                video_seen = True
            if _valid_audio(value):
                audio_seen = True
            previews.append(preview)

        if is_output_list:
            text_lines = ["Input container: OUTPUT_IS_LIST", f"Item count: {len(values)}"]
            for index, preview in enumerate(previews):
                text_lines.extend(["", f"--- Item {index} ---", *preview.lines])
        else:
            text_lines = [*previews[0].lines]

        video_preview = next((item for item in previews if item.kind == "video" and item.images), None)
        audio_preview = next((item for item in previews if item.kind == "audio" and item.audio), None)
        image_previews = [image for item in previews if item.kind == "image" for image in item.images]
        if video_preview is not None:
            kind = "video"
            images = video_preview.images
            audios: list[dict[str, Any]] = []
            waveform: list[float] = []
        elif audio_preview is not None:
            kind = "audio"
            images = []
            audios = audio_preview.audio[:1]
            waveform = audio_preview.waveform
        elif image_previews:
            kind = "image"
            images = image_previews
            audios = []
            waveform = []
        else:
            kind = "none"
            images = []
            audios = []
            waveform = []

        text = _truncate_text("\n".join(text_lines))
        payload = {
            "kind": kind,
            "is_list": is_output_list,
            "item_count": len(values),
            "waveform": waveform,
        }
        ui_payload: dict[str, Any] = {
            "images": images,
            "audio": audios,
            "text": (text,),
            "preview_any": (payload,),
        }
        # Older ComfyUI frontends treated the presence of ``animated`` as
        # enough to select the video component. Omit it for still images,
        # masks, audio, and text instead of sending ``[false]``.
        if kind == "video":
            ui_payload["animated"] = (True,)
        return io.NodeOutput(
            ui=ui_payload
        )


class CineStylePreviewAnyExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CSPreviewAny]


async def comfy_entrypoint() -> CineStylePreviewAnyExtension:
    return CineStylePreviewAnyExtension()
