"""CineStyle video input and output nodes for ComfyUI."""

from __future__ import annotations

import os
import math
import json
from fractions import Fraction
from typing import Any

import av
import torch
import torch.nn.functional as F
from aiohttp import web
from typing_extensions import override

import folder_paths
from comfy_api.latest import ComfyExtension, Input, InputImpl, Types, io, ui


_ROUTE_REGISTERED = False


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


def _read_video_info(filename: str) -> dict[str, Any]:
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
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
        "duration": duration,
        "audio_format": audio_stream.codec.name if audio_stream and audio_stream.codec else None,
    }


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
        return web.json_response(_read_video_info(filename))
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
            ],
            outputs=[
                io.Video.Output(),
                io.Image.Output(display_name="IMAGE"),
                io.Int.Output(display_name="frame_count"),
                io.Audio.Output(display_name="audio"),
                io.Dict.Output(display_name="video_info"),
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
    ) -> io.NodeOutput:
        if not folder_paths.exists_annotated_filepath(video):
            raise ValueError(f"Invalid video file: {video}")

        source = InputImpl.VideoFromFile(folder_paths.get_annotated_filepath(video))
        components = source.get_components()
        images = components.images
        if images.ndim != 4 or images.shape[-1] not in (3, 4):
            raise ValueError("Decoded video frames must have shape [frames, height, width, 3 or 4]")
        source_fps = float(components.frame_rate)
        source_count = int(images.shape[0])
        if source_count == 0:
            raise ValueError(f"Video contains no decodable frames: {video}")

        start = max(0, min(int(start_frame), source_count - 1))
        end = source_count - 1 if int(end_frame) < 0 else min(int(end_frame), source_count - 1)
        if end < start:
            raise ValueError(f"end_frame ({end}) must be greater than or equal to start_frame ({start})")

        selected = images[start : end + 1]
        selected_duration = selected.shape[0] / source_fps
        target_fps = source_fps if fps <= 0 else float(fps)
        selected = _resample_frames(selected, source_fps, target_fps)

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
        selected = _resize_frames(selected, output_width, output_height)

        audio = _trim_audio(components.audio, start / source_fps, selected_duration)
        info = {
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
        }
        video_images = selected[..., :3] if selected.shape[-1] == 4 else selected
        output_video = InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=video_images,
                audio=audio,
                frame_rate=Fraction(target_fps).limit_denominator(1000),
            )
        )
        return io.NodeOutput(output_video, selected, int(selected.shape[0]), audio, info)

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
            _ROUTE_REGISTERED = True

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CSLoadVideo, CSSaveVideo]


async def comfy_entrypoint() -> CineStyleVideoExtension:
    return CineStyleVideoExtension()


WEB_DIRECTORY = "./web"
