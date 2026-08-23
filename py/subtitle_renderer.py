"""Shared Pillow/FreeType subtitle renderer for final output and preview."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


_FONT_SUFFIXES = {".ttf", ".otf", ".ttc"}


def font_files(fonts_root: Path) -> list[str]:
    root = Path(fonts_root)
    if not root.is_dir():
        return []
    return sorted(
        str(path.relative_to(root)).replace("\\", "/")
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _FONT_SUFFIXES
    )


def resolve_font(value: str, fonts_root: Path):
    from PIL import ImageFont

    root = Path(fonts_root).resolve()
    candidate = str(value or "").strip()
    if candidate:
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
    fonts = font_files(root)
    return (root / fonts[0]) if fonts else None


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


def _safe_number(value: Any, fallback: float) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else fallback
    except (TypeError, ValueError, OverflowError):
        return fallback


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _text_bbox(draw, text: str, font, spacing: int, stroke_width: int):
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    if spacing == 0 or len(text) < 2:
        return bbox
    width = 0
    for char in text:
        char_box = draw.textbbox((0, 0), char, font=font, stroke_width=0)
        width += char_box[2] - char_box[0]
    width += spacing * (len(text) - 1)
    return (bbox[0], bbox[1], bbox[0] + max(0, width + stroke_width * 2), bbox[3])


def _draw_spaced_text(draw, xy, text: str, font, fill, spacing: int, stroke_width: int = 0, stroke_fill=None):
    if spacing == 0:
        draw.text(xy, text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
        return
    x, y = xy
    for index, char in enumerate(text):
        draw.text((x, y), char, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
        char_box = draw.textbbox((0, 0), char, font=font, stroke_width=0)
        x += char_box[2] - char_box[0]
        if index + 1 < len(text):
            x += spacing


def _draw_text_layer(image, active: list[dict[str, Any]], style: dict[str, Any], fonts_root: Path, supersample: bool = True):
    from PIL import Image, ImageDraw, ImageFont

    height, width = image.height, image.width
    if supersample and width > 0 and height > 0:
        scale = 2
        enlarged = image.resize((width * scale, height * scale), Image.Resampling.BICUBIC)
        scaled_style = dict(style)
        for key in ("font_size", "outline_size", "shadow_size", "letter_spacing"):
            scaled_style[key] = _safe_number(style.get(key), 0) * scale
        rendered = _draw_text_layer(enlarged, active, scaled_style, fonts_root, supersample=False)
        return rendered.resize((width, height), Image.Resampling.LANCZOS)
    font_path = resolve_font(str(style.get("font", "")), fonts_root)
    font_size = max(8, min(100, int(round(_safe_number(style.get("font_size", 30), 30)))))
    try:
        font = ImageFont.truetype(str(font_path), font_size) if font_path else ImageFont.load_default()
    except (OSError, ValueError):
        font = ImageFont.load_default()

    fill_1 = _parse_colour(style.get("primary_color", "#FFFFFF"), (255, 255, 255, 255))
    fill_2 = _parse_colour(style.get("secondary_color", "#FF0000"), fill_1)
    outline_color = _parse_colour(style.get("outline_color", "#000000"), (0, 0, 0, 255))
    shadow_color = _parse_colour(style.get("shadow_color", "#000000"), (0, 0, 0, 190))
    outline_size = max(0, min(20, int(round(_safe_number(style.get("outline_size", 2), 2)))))
    shadow_size = max(0, min(20, int(round(_safe_number(style.get("shadow_size", 3), 3)))))
    spacing = max(-10, min(50, int(round(_safe_number(style.get("letter_spacing", 0), 0)))))
    align = str(style.get("text_align", "center")).lower()
    if align not in {"left", "center", "right"}:
        align = "center"
    pos_x = max(0.0, min(1.0, _safe_number(style.get("position_x", 0.5), 0.5)))
    pos_y = max(0.0, min(1.0, _safe_number(style.get("position_y", 0.88), 0.88)))
    draw = ImageDraw.Draw(image)

    draw_cues = [{"text": "\n".join(str(item.get("text", "")).strip() for item in active)}] if active else []
    for cue in draw_cues:
        text = " ".join(str(cue.get("text", "")).replace("\r", "\n").split())
        if not text:
            continue
        lines = [text]

        line_boxes = [_text_bbox(draw, line, font, spacing, outline_size) for line in lines]
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
        top = max(0, min(height - text_height, top))

        stroke_mask = Image.new("L", (width, height), 0)
        fill_mask = Image.new("L", (width, height), 0)
        stroke_draw = ImageDraw.Draw(stroke_mask)
        fill_draw = ImageDraw.Draw(fill_mask)
        for line_index, line in enumerate(lines):
            line_box = line_boxes[line_index]
            line_width = line_box[2] - line_box[0]
            if align == "right":
                line_left = left + text_width - line_width
            elif align == "center":
                line_left = left + (text_width - line_width) // 2
            else:
                line_left = left
            baseline = top + line_index * line_height - line_box[1]
            _draw_spaced_text(stroke_draw, (line_left, baseline), line, font, 255, spacing, outline_size, 255)
            _draw_spaced_text(fill_draw, (line_left, baseline), line, font, 255, spacing)

        if _as_bool(style.get("italic", False)):
            shear = 0.20
            matrix = (1, shear, -shear * top, 0, 1, 0)
            resample = Image.Resampling.BICUBIC
            stroke_mask = stroke_mask.transform((width, height), Image.Transform.AFFINE, matrix, resample=resample)
            fill_mask = fill_mask.transform((width, height), Image.Transform.AFFINE, matrix, resample=resample)

        if shadow_size:
            shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            shadow_mask = Image.new("L", (width, height), 0)
            shadow_mask.paste(stroke_mask, (shadow_size, shadow_size))
            shadow.paste(shadow_color, mask=shadow_mask)
            image.alpha_composite(shadow)
        if outline_size:
            outline = Image.new("RGBA", (width, height), outline_color)
            image.alpha_composite(Image.composite(outline, Image.new("RGBA", (width, height)), stroke_mask))

        if _as_bool(style.get("gradient", False)) and fill_1 != fill_2:
            fill = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            gradient_draw = ImageDraw.Draw(fill)
            for y in range(max(1, text_height)):
                ratio = y / max(1, text_height - 1)
                colour = tuple(int(fill_1[channel] * (1 - ratio) + fill_2[channel] * ratio) for channel in range(4))
                gradient_draw.line((0, top + y, width, top + y), fill=colour)
        else:
            fill = Image.new("RGBA", (width, height), fill_1)
        image.alpha_composite(Image.composite(fill, Image.new("RGBA", (width, height)), fill_mask))
    return image


def render_frame(frame: torch.Tensor, active: list[dict[str, Any]], style: dict[str, Any], fonts_root: Path) -> torch.Tensor:
    from PIL import Image

    image = Image.fromarray(
        (frame[..., :3].clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()),
        mode="RGB",
    ).convert("RGBA")
    image = _draw_text_layer(image, active, style, fonts_root)
    return torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0)


def render_overlay_png_with_bounds(width: int, height: int, active: list[dict[str, Any]], style: dict[str, Any], fonts_root: Path) -> tuple[bytes, tuple[int, int, int, int] | None]:
    from io import BytesIO
    from PIL import Image

    image = Image.new("RGBA", (int(width), int(height)), (0, 0, 0, 0))
    image = _draw_text_layer(image, active, style, fonts_root)
    bounds = image.getchannel("A").getbbox()
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue(), bounds


def render_overlay_png(width: int, height: int, active: list[dict[str, Any]], style: dict[str, Any], fonts_root: Path) -> bytes:
    body, _ = render_overlay_png_with_bounds(width, height, active, style, fonts_root)
    return body


__all__ = ["font_files", "render_frame", "render_overlay_png", "render_overlay_png_with_bounds", "resolve_font"]
