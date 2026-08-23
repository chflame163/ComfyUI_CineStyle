"""Subtitle timeline and burn-in node for standard ComfyUI VIDEO values."""

from __future__ import annotations

import json
import os
import re
from urllib.parse import unquote
from pathlib import Path
from typing import Any

import numpy as np
import torch

import folder_paths
from comfy_api.latest import ComfyExtension, Input, InputImpl, Types, io


_CATEGORY = "😺dzNodes/CineStyle/Video"
_ROUTE_REGISTERED = False
_TIME_RE = re.compile(r"^(\d+):(\d{2}):(\d{2})[,.](\d{3})$")


def _video_files() -> list[str]:
    input_dir = folder_paths.get_input_directory()
    files = [
        name
        for name in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, name))
    ]
    return sorted(folder_paths.filter_files_content_types(files, ["video"]))


def _fonts_root() -> Path:
    return Path(folder_paths.models_dir) / "fonts"


def _font_files() -> list[str]:
    root = _fonts_root()
    if not root.is_dir():
        return []
    return sorted(
        str(path.relative_to(root)).replace("\\", "/")
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".ttf", ".otf", ".ttc"}
    )


def _parse_time(value: str) -> float:
    match = _TIME_RE.match(value.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value!r}")
    hours, minutes, seconds, milliseconds = (int(item) for item in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Invalid SRT timestamp: {value!r}")
    return hours * 3600.0 + minutes * 60.0 + seconds + milliseconds / 1000.0


def parse_srt(text: str) -> list[dict[str, Any]]:
    """Parse SRT cues while tolerating BOM, blank lines and cue settings."""
    source = str(text or "").replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", source.strip()) if source.strip() else []
    cues: list[dict[str, Any]] = []
    for block in blocks:
        lines = [line.rstrip() for line in block.split("\n")]
        if not lines:
            continue
        if "-->" not in lines[0]:
            lines = lines[1:]
        if not lines or "-->" not in lines[0]:
            continue
        start_text, end_text = [part.strip().split(" ", 1)[0] for part in lines[0].split("-->", 1)]
        start = _parse_time(start_text)
        end = _parse_time(end_text)
        value = "\n".join(lines[1:]).strip()
        if value and end > start:
            cues.append({"id": len(cues) + 1, "start": start, "end": end, "text": value})
    return cues


def _coerce_cues(srt: str, subtitle_data: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(subtitle_data or "[]")
        if isinstance(parsed, list):
            result = []
            for index, item in enumerate(parsed, 1):
                if not isinstance(item, dict):
                    continue
                start = float(item.get("start", 0.0))
                end = float(item.get("end", 0.0))
                value = str(item.get("text", "")).strip()
                if value and end > start:
                    result.append({"id": item.get("id", index), "start": start, "end": end, "text": value})
            if result:
                return result
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return parse_srt(srt)


def _parse_colour(value: str, fallback: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    text = str(value or "").strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) == 6:
        try:
            return (*bytes.fromhex(text), 255)
        except ValueError:
            pass
    if len(text) == 8:
        try:
            return tuple(bytes.fromhex(text))  # type: ignore[return-value]
        except ValueError:
            pass
    return fallback


def _resolve_font(value: str):
    from PIL import ImageFont

    candidate = str(value or "").strip()
    if candidate:
        root = _fonts_root().resolve()
        path = Path(candidate)
        if not path.is_absolute():
            path = root / path
        try:
            path = path.resolve()
            if root == path or root not in path.parents:
                path = root / Path(candidate).name
            if path.is_file():
                return path
        except OSError:
            pass
    fonts = _font_files()
    return (_fonts_root() / fonts[0]) if fonts else None


def _draw_subtitle(frame: torch.Tensor, active: list[dict[str, Any]], style: dict[str, Any]) -> torch.Tensor:
    from PIL import Image, ImageDraw, ImageFont

    height, width = int(frame.shape[0]), int(frame.shape[1])
    image = Image.fromarray(
        (frame[..., :3].clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()),
        mode="RGB",
    ).convert("RGBA")
    font_path = _resolve_font(str(style.get("font_family", "")))
    font_size = max(8, min(256, int(float(style.get("font_size", 48) or 48))))
    try:
        font = ImageFont.truetype(str(font_path), font_size) if font_path else ImageFont.load_default()
    except (OSError, ValueError):
        font = ImageFont.load_default()

    fill_1 = _parse_colour(style.get("fill_color_1", "#FFFFFF"), (255, 255, 255, 255))
    fill_2 = _parse_colour(style.get("fill_color_2", ""), fill_1)
    outline_color = _parse_colour(style.get("outline_color", "#000000"), (0, 0, 0, 255))
    shadow_color = _parse_colour(style.get("shadow_color", "#000000"), (0, 0, 0, 190))
    outline_size = max(0, min(32, int(float(style.get("outline_size", 2) or 0))))
    shadow_size = max(0, min(32, int(float(style.get("shadow_size", 3) or 0))))
    align = str(style.get("text_align", "center")).lower()
    pos_x = max(0.0, min(1.0, float(style.get("position_x", 0.5) or 0.5)))
    pos_y = max(0.0, min(1.0, float(style.get("position_y", 0.88) or 0.88)))
    max_width = max(80, int(width * 0.9))

    draw_cues = [{"text": "\n".join(str(item.get("text", "")).strip() for item in active)}] if active else []
    for cue in draw_cues:
        text = str(cue.get("text", "")).strip()
        if not text:
            continue
        lines: list[str] = []
        for source_line in text.splitlines() or [text]:
            words = list(source_line) if any("\u4e00" <= char <= "\u9fff" for char in source_line) else source_line.split(" ")
            line = ""
            for word in words:
                candidate = f"{line}{word}" if not line or any("\u4e00" <= char <= "\u9fff" for char in source_line) else f"{line} {word}".strip()
                bbox = ImageDraw.Draw(image).textbbox((0, 0), candidate, font=font, stroke_width=outline_size)
                if line and bbox[2] - bbox[0] > max_width:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            if line:
                lines.append(line)
        if not lines:
            continue
        line_boxes = [ImageDraw.Draw(image).textbbox((0, 0), line, font=font, stroke_width=outline_size) for line in lines]
        text_width = max(box[2] - box[0] for box in line_boxes)
        line_height = max(box[3] - box[1] for box in line_boxes) + max(2, font_size // 8)
        text_height = line_height * len(lines)
        anchor_x = int(width * pos_x)
        top = int(height * pos_y - text_height)
        if align == "left":
            left = anchor_x
        elif align == "right":
            left = anchor_x - text_width
        else:
            left = anchor_x - text_width // 2
        left = max(0, min(width - text_width, left))
        top = max(0, min(height - text_height, top))

        stroke_mask = Image.new("L", (width, height), 0)
        fill_mask = Image.new("L", (width, height), 0)
        stroke_draw = ImageDraw.Draw(stroke_mask)
        fill_draw = ImageDraw.Draw(fill_mask)
        for line_index, line in enumerate(lines):
            line_box = line_boxes[line_index]
            line_width = line_box[2] - line_box[0]
            line_left = left if align == "left" else left + (text_width - line_width if align == "right" else (text_width - line_width) // 2)
            baseline = top + line_index * line_height - line_box[1]
            stroke_draw.text((line_left, baseline), line, font=font, fill=255, stroke_width=outline_size, stroke_fill=255)
            fill_draw.text((line_left, baseline), line, font=font, fill=255)

        if shadow_size:
            shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            shadow_mask = Image.new("L", (width, height), 0)
            shadow_mask.paste(stroke_mask, (shadow_size, shadow_size))
            shadow.paste(shadow_color, mask=shadow_mask)
            image.alpha_composite(shadow)
        if outline_size:
            outline = Image.new("RGBA", (width, height), outline_color)
            image.alpha_composite(Image.composite(outline, Image.new("RGBA", (width, height)), stroke_mask))

        fill = Image.new("RGBA", (width, height), fill_1)
        if bool(style.get("gradient", False)) and fill_1 != fill_2:
            gradient = Image.new("RGBA", (1, max(1, text_height)))
            pixels = gradient.load()
            for y in range(max(1, text_height)):
                ratio = y / max(1, text_height - 1)
                pixels[0, y] = tuple(int(fill_1[channel] * (1 - ratio) + fill_2[channel] * ratio) for channel in range(4))
            fill = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            fill.paste(gradient.resize((width, height)), (0, 0))
        image.alpha_composite(Image.composite(fill, Image.new("RGBA", (width, height)), fill_mask))
    return torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0)


class CSVideoSubtitleTrack(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        videos = _video_files()
        return io.Schema(
            node_id="CS_Video_Subtitle_Track",
            display_name="CS Video Subtitle Track",
            category=_CATEGORY,
            essentials_category="Video Tools",
            search_aliases=["SRT", "subtitle timeline", "字幕轨道", "burn subtitles"],
            description="Burn an editable SRT subtitle track onto a standard ComfyUI VIDEO.",
            inputs=[
                io.Video.Input("video", tooltip="Standard ComfyUI VIDEO input."),
                io.String.Input("srt", multiline=True, default="", dynamic_prompts=False, tooltip="SRT subtitle text."),
                io.Combo.Input("video_file", options=videos or [""], upload=io.UploadType.video, optional=True, advanced=True, tooltip="Optional source file used only for timeline proxy preview."),
                io.Int.Input("start_frame", default=0, min=0, max=10000000, step=1, advanced=True),
                io.Int.Input("end_frame", default=-1, min=-1, max=10000000, step=1, advanced=True),
                io.String.Input("subtitle_data", multiline=True, default="[]", advanced=True, tooltip="Timeline cue JSON saved by Edit Timeline."),
                io.String.Input("font_family", default="", advanced=True),
                io.Int.Input("font_size", default=48, min=8, max=256, step=1, advanced=True),
                io.String.Input("fill_color_1", default="#FFFFFF", advanced=True),
                io.String.Input("fill_color_2", default="#FFFFFF", advanced=True),
                io.Boolean.Input("gradient", default=False, advanced=True),
                io.Combo.Input("text_align", options=["left", "center", "right"], default="center", advanced=True),
                io.Float.Input("position_x", default=0.5, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Float.Input("position_y", default=0.88, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Int.Input("outline_size", default=2, min=0, max=32, step=1, advanced=True),
                io.String.Input("outline_color", default="#000000", advanced=True),
                io.Int.Input("shadow_size", default=3, min=0, max=32, step=1, advanced=True),
                io.String.Input("shadow_color", default="#000000", advanced=True),
            ],
            outputs=[io.Video.Output("video"), io.Dict.Output("subtitle_info")],
        )

    @classmethod
    def execute(
        cls,
        video: Input.Video,
        srt: str,
        video_file: str = "",
        start_frame: int = 0,
        end_frame: int = -1,
        subtitle_data: str = "[]",
        font_family: str = "",
        font_size: int = 48,
        fill_color_1: str = "#FFFFFF",
        fill_color_2: str = "#FFFFFF",
        gradient: bool = False,
        text_align: str = "center",
        position_x: float = 0.5,
        position_y: float = 0.88,
        outline_size: int = 2,
        outline_color: str = "#000000",
        shadow_size: int = 3,
        shadow_color: str = "#000000",
    ) -> io.NodeOutput:
        components = video.get_components()
        images = components.images
        if images.ndim != 4 or images.shape[0] == 0:
            raise ValueError("VIDEO contains no decodable frames.")
        frame_rate = float(components.frame_rate)
        if frame_rate <= 0:
            raise ValueError("VIDEO frame rate must be positive.")
        cues = _coerce_cues(srt, subtitle_data)
        source_metadata = dict(components.metadata or {})
        source_offset = 0.0
        if source_metadata.get("source_fps"):
            source_offset = float(source_metadata.get("start_frame", 0) or 0) / float(source_metadata["source_fps"])
        if source_offset:
            cues = [
                {**cue, "start": max(0.0, float(cue["start"]) - source_offset), "end": max(0.0, float(cue["end"]) - source_offset)}
                for cue in cues
            ]
            cues = [cue for cue in cues if cue["end"] > cue["start"]]
        style = {
            "font_family": font_family,
            "font_size": font_size,
            "fill_color_1": fill_color_1,
            "fill_color_2": fill_color_2,
            "gradient": gradient,
            "text_align": text_align,
            "position_x": position_x,
            "position_y": position_y,
            "outline_size": outline_size,
            "outline_color": outline_color,
            "shadow_size": shadow_size,
            "shadow_color": shadow_color,
        }
        rendered = []
        for index, frame in enumerate(images):
            time_seconds = index / frame_rate
            active = [cue for cue in cues if float(cue["start"]) <= time_seconds < float(cue["end"])]
            rendered.append(_draw_subtitle(frame, active, style) if active else frame[..., :3].detach().cpu().float())
        output_images = torch.stack(rendered, dim=0).clamp(0, 1)
        metadata = source_metadata
        metadata["cinestyle_subtitles"] = {"cue_count": len(cues), "source": video_file or None, "style": style}
        try:
            output_components = Types.VideoComponents(
                images=output_images,
                audio=components.audio,
                frame_rate=components.frame_rate,
                metadata=metadata,
            )
        except TypeError:
            output_components = Types.VideoComponents(
                images=output_images,
                audio=components.audio,
                frame_rate=components.frame_rate,
            )
        info = {"cue_count": len(cues), "frame_count": int(images.shape[0]), "fps": frame_rate, "video_file": video_file}
        return io.NodeOutput(InputImpl.VideoFromComponents(output_components), info)


async def _fonts_route(request):
    from aiohttp import web

    return web.json_response({"fonts": _font_files()})


async def _font_file_route(request):
    from aiohttp import web

    relative = unquote(request.match_info.get("font", ""))
    root = _fonts_root().resolve()
    try:
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise FileNotFoundError(relative)
    except (OSError, ValueError):
        return web.json_response({"error": "font not found"}, status=404)
    return web.FileResponse(path=path, headers={"Cache-Control": "public,max-age=86400"})


class CineStyleVideoSubtitleExtension(ComfyExtension):
    async def on_load(self) -> None:
        global _ROUTE_REGISTERED
        if _ROUTE_REGISTERED:
            return
        from server import PromptServer

        server_instance = getattr(PromptServer, "instance", None)
        if server_instance is not None:
            server_instance.routes.get("/cinestyle/fonts")(_fonts_route)
            server_instance.routes.get("/cinestyle/font/{font:.*}")(_font_file_route)
            _ROUTE_REGISTERED = True

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CSVideoSubtitleTrack]


async def comfy_entrypoint() -> CineStyleVideoSubtitleExtension:
    return CineStyleVideoSubtitleExtension()


WEB_DIRECTORY = "./web"
