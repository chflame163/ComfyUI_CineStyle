

from __future__ import annotations

import base64
import io as py_io
import json
import logging
import math
import sys
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from aiohttp import web
from PIL import Image
from typing_extensions import override

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - ComfyUI normally provides tqdm
    tqdm = None

from comfy_api.latest import ComfyExtension, io


_EPS = 1.0e-6
_LUT_SIZE = 4096
_NO_EXTERNAL_LUT = "None"
_MAX_CUBE_1D_SIZE = 65536
_MAX_CUBE_3D_SIZE = 128
_CUBE_PIXEL_CHUNK = 1 << 20
_GPU_MEMORY_FRACTION = 0.5
_GPU_MEMORY_RESERVE_BYTES = 512 * 1024 * 1024
_MAX_GPU_BATCH = 64
_CPU_BATCH = 16
_CATEGORY = "😺dzNodes/CineStyle"
_LOGGER = logging.getLogger("CineStyleColorGrade")
_CACHE_LOCK = threading.RLock()
_MASK_CACHE: dict[str, torch.Tensor | None] = {}
_MASK_CACHE_BY_TOKEN: dict[str, torch.Tensor | None] = {}
_PREVIEW_CACHE_STORE = None
_ROUTES_REGISTERED = False
_ANSI_GREEN = "\033[32m"
_ANSI_RESET = "\033[0m"

_DEFAULT_CURVES = {
    "version": 1,
    "domain": [0.0, 1.0],
    "rgb": [[0.0, 0.0], [1.0, 1.0]],
    "r": [[0.0, 0.0], [1.0, 1.0]],
    "g": [[0.0, 0.0], [1.0, 1.0]],
    "b": [[0.0, 0.0], [1.0, 1.0]],
}
_DEFAULT_CURVES_JSON = json.dumps(_DEFAULT_CURVES, separators=(",", ":"))


def _grade_info(message: str, *args: Any) -> None:
    """Keep status lines consistent with the other CineStyle video nodes."""
    _LOGGER.info("[CS Color Grade] " + message, *args)


class _GradeProgress:
    """Emit a single throttled tqdm-style frame progress bar when available."""

    def __init__(self, total: int, description: str = "frame processing"):
        self.bar = None
        if tqdm is not None:
            self.bar = tqdm(
                total=max(1, int(total)),
                desc=f"{_ANSI_GREEN}[INFO]{_ANSI_RESET} [CS Color Grade] {description}",
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


def _lut_root() -> Path:
    import folder_paths

    return Path(folder_paths.models_dir).resolve() / "luts"


def _lut_files() -> list[str]:
    root = _lut_root()
    if not root.is_dir():
        return [_NO_EXTERNAL_LUT]
    files = sorted(
        (
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() == ".cube"
        ),
        key=str.casefold,
    )
    return [_NO_EXTERNAL_LUT, *files]


def _resolve_lut_path(value: Any) -> Path | None:
    name = str(value or "").strip().replace("\\", "/")
    if not name or name.casefold() == _NO_EXTERNAL_LUT.casefold():
        return None
    relative = Path(name)
    if relative.is_absolute() or relative.suffix.lower() != ".cube" or ".." in relative.parts:
        raise ValueError("lut must name a .cube file inside ComfyUI/models/luts.")
    root = _lut_root()
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise ValueError("lut must stay inside ComfyUI/models/luts.")
    if not path.is_file():
        raise ValueError(f"LUT file not found in ComfyUI/models/luts: {name}")
    return path


@dataclass(frozen=True)
class _CubeLUT:
    one_d: torch.Tensor | None
    one_d_min: tuple[float, float, float]
    one_d_max: tuple[float, float, float]
    three_d: torch.Tensor | None
    three_d_min: tuple[float, float, float]
    three_d_max: tuple[float, float, float]


def _parse_cube_vector(tokens: Sequence[str], name: str, line_number: int) -> tuple[float, float, float]:
    if len(tokens) != 3:
        raise ValueError(f"{name} must contain three values (line {line_number}).")
    values = tuple(_finite_float(item, f"{name}[{index}]") for index, item in enumerate(tokens))
    return values  # type: ignore[return-value]


def _parse_cube_range(
    tokens: Sequence[str], name: str, line_number: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if len(tokens) != 2:
        raise ValueError(f"{name} must contain min and max (line {line_number}).")
    low = _finite_float(tokens[0], f"{name}.min")
    high = _finite_float(tokens[1], f"{name}.max")
    if high <= low:
        raise ValueError(f"{name} max must be greater than min (line {line_number}).")
    return (low, low, low), (high, high, high)  # type: ignore[return-value]


@lru_cache(maxsize=64)
def _parse_cube_cached(path_string: str, size: int, mtime_ns: int) -> _CubeLUT:
    del size, mtime_ns  # Cache key invalidates the parsed result when the file changes.
    path = Path(path_string)
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Unable to read LUT file: {path.name}") from exc

    one_d_size: int | None = None
    three_d_size: int | None = None
    domain_min = (0.0, 0.0, 0.0)
    domain_max = (1.0, 1.0, 1.0)
    one_d_min = domain_min
    one_d_max = domain_max
    three_d_min = domain_min
    three_d_max = domain_max
    one_d_range_set = False
    three_d_range_set = False
    rows: list[tuple[float, float, float]] = []
    directive_names = {
        "TITLE", "DOMAIN_MIN", "DOMAIN_MAX", "LUT_1D_SIZE", "LUT_3D_SIZE",
        "LUT_1D_INPUT_RANGE", "LUT_3D_INPUT_RANGE",
    }
    for line_number, raw_line in enumerate(lines, 1):
        content = raw_line.split("#", 1)[0].strip()
        if not content:
            continue
        tokens = content.split()
        key = tokens[0].upper()
        if key == "TITLE":
            continue
        if key == "DOMAIN_MIN":
            domain_min = _parse_cube_vector(tokens[1:], key, line_number)
            continue
        if key == "DOMAIN_MAX":
            domain_max = _parse_cube_vector(tokens[1:], key, line_number)
            continue
        if key == "LUT_1D_SIZE" or key == "LUT_3D_SIZE":
            if len(tokens) != 2:
                raise ValueError(f"{key} must contain one integer (line {line_number}).")
            try:
                parsed_size = int(tokens[1])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must contain one integer (line {line_number}).") from exc
            limit = _MAX_CUBE_1D_SIZE if key == "LUT_1D_SIZE" else _MAX_CUBE_3D_SIZE
            if parsed_size < 2 or parsed_size > limit:
                raise ValueError(f"{key} must be between 2 and {limit}.")
            if key == "LUT_1D_SIZE":
                one_d_size = parsed_size
            else:
                three_d_size = parsed_size
            continue
        if key == "LUT_1D_INPUT_RANGE" or key == "LUT_3D_INPUT_RANGE":
            low, high = _parse_cube_range(tokens[1:], key, line_number)
            if key == "LUT_1D_INPUT_RANGE":
                one_d_min, one_d_max = low, high
                one_d_range_set = True
            else:
                three_d_min, three_d_max = low, high
                three_d_range_set = True
            continue
        try:
            values = tuple(_finite_float(item, "LUT value") for item in tokens)
        except ValueError:
            # Preserve compatibility with harmless, vendor-specific metadata directives.
            if key.isalpha() or key in directive_names:
                continue
            raise ValueError(f"Invalid LUT data (line {line_number}).")
        if len(values) != 3:
            raise ValueError(f"LUT data must contain three values (line {line_number}).")
        rows.append(values)  # type: ignore[arg-type]

    if one_d_size is None and three_d_size is None:
        raise ValueError(f"{path.name} does not declare LUT_1D_SIZE or LUT_3D_SIZE.")
    if domain_max[0] <= domain_min[0] or domain_max[1] <= domain_min[1] or domain_max[2] <= domain_min[2]:
        raise ValueError("DOMAIN_MAX must be greater than DOMAIN_MIN for every channel.")
    if not one_d_range_set:
        one_d_min, one_d_max = domain_min, domain_max
    if not three_d_range_set:
        three_d_min, three_d_max = domain_min, domain_max
    offset = 0
    one_d = None
    if one_d_size is not None:
        expected = one_d_size
        if len(rows) < expected:
            raise ValueError(f"{path.name} contains {len(rows)} LUT rows; expected at least {expected}.")
        one_d = torch.tensor(rows[:expected], dtype=torch.float32)
        offset = expected
    three_d = None
    if three_d_size is not None:
        expected = three_d_size ** 3
        if len(rows) - offset != expected:
            raise ValueError(f"{path.name} contains {len(rows) - offset} 3D LUT rows; expected {expected}.")
        # .cube stores red fastest, then green, then blue: [blue, green, red, channel].
        three_d = torch.tensor(rows[offset:], dtype=torch.float32).reshape(three_d_size, three_d_size, three_d_size, 3)
    elif len(rows) != offset:
        raise ValueError(f"{path.name} contains unexpected extra LUT rows.")
    return _CubeLUT(one_d, one_d_min, one_d_max, three_d, three_d_min, three_d_max)


def _load_cube(value: Any) -> _CubeLUT | None:
    path = _resolve_lut_path(value)
    if path is None:
        return None
    try:
        stat = path.stat()
        return _parse_cube_cached(str(path), int(stat.st_size), int(stat.st_mtime_ns))
    except OSError as exc:
        raise ValueError(f"Unable to stat LUT file: {path.name}") from exc


def _cube_on_device(cube: _CubeLUT | None, device: torch.device) -> _CubeLUT | None:
    if cube is None:
        return None
    return _CubeLUT(
        cube.one_d.to(device=device, non_blocking=True) if cube.one_d is not None else None,
        cube.one_d_min,
        cube.one_d_max,
        cube.three_d.to(device=device, non_blocking=True) if cube.three_d is not None else None,
        cube.three_d_min,
        cube.three_d_max,
    )


def _preview_cache_store():
    global _PREVIEW_CACHE_STORE
    if _PREVIEW_CACHE_STORE is None:
        package = __name__.rsplit(".", 1)[0]
        module = sys.modules.get(f"{package}._py_preview_cache")
        if module is None:
            raise RuntimeError("CineStyle preview cache module is unavailable.")
        _PREVIEW_CACHE_STORE = module.PreviewCacheStore("color_grade")
    return _PREVIEW_CACHE_STORE


def _loader_preview_cache():
    package = __name__.rsplit(".", 1)[0]
    module = sys.modules.get(f"{package}._py_loader_preview_cache")
    return module.get_loader_preview_cache() if module is not None else None


def _cache_wait_input(node_id: Any, prompt: Any, image: torch.Tensor) -> dict[str, Any] | None:
    package = __name__.rsplit(".", 1)[0]
    module = sys.modules.get(f"{package}._py_preview_cache")
    if module is None:
        return None
    chain = module.build_input_chain(prompt, node_id, ("image",))
    if chain is None:
        return None
    try:
        return module.get_wait_input_cache_store().put_chain(
            chain,
            image[..., :3],
            _source_fps_from_prompt(prompt, node_id),
            info={
                "producer_node_id": str(node_id or ""),
                "producer_node_type": "CS_Color_Grade",
            },
            force=True,
        )
    except Exception as exc:
        _LOGGER.warning("[CS Color Grade] wait input cache failed: %s", exc)
        return None


def _loader_id_from_prompt(prompt: Any, node_id: Any) -> str:
    """Find a CS Load Video upstream when the node receives only IMAGE."""
    if not isinstance(prompt, dict):
        return ""
    node = prompt.get(str(node_id)) or prompt.get(node_id)
    if not isinstance(node, dict):
        return ""
    inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
    pending = [
        value
        for name, value in inputs.items()
        if "image" in str(name).lower() or "video" in str(name).lower()
    ]
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
        if class_type == "CS_Load_Video" or class_type.endswith("::CS_Load_Video"):
            return upstream_id
        upstream_inputs = upstream.get("inputs") if isinstance(upstream.get("inputs"), dict) else {}
        pending.extend(
            value
            for value in upstream_inputs.values()
            if isinstance(value, (list, tuple)) and len(value) >= 2
        )
    return ""


def _source_fps_from_prompt(prompt: Any, node_id: Any) -> float:
    if not isinstance(prompt, dict):
        return 24.0
    pending = [prompt.get(str(node_id)) or prompt.get(node_id)]
    visited: set[str] = set()
    while pending:
        current = pending.pop(0)
        if not isinstance(current, dict):
            continue
        inputs = current.get("inputs") if isinstance(current.get("inputs"), dict) else {}
        for name in ("fps", "frame_rate", "target_fps"):
            try:
                candidate = float(inputs.get(name))
                if math.isfinite(candidate) and candidate > 0:
                    return candidate
            except (TypeError, ValueError, OverflowError):
                pass
        for value in inputs.values():
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                continue
            upstream_id = str(value[0])
            if upstream_id in visited:
                continue
            visited.add(upstream_id)
            pending.append(prompt.get(upstream_id) or prompt.get(value[0]))
    return 24.0


def _as_rgb(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError("image must have shape [batch, height, width, 3 or 4].")
    if any(int(size) <= 0 for size in image.shape[:3]):
        raise ValueError("image must contain at least one non-empty frame.")
    return image[..., :3]


def _preferred_device(fallback: torch.device) -> torch.device:
    try:
        import comfy.model_management as model_management

        device = model_management.get_torch_device()
        return device if isinstance(device, torch.device) else torch.device(device)
    except Exception:
        return fallback


def _finite_float(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum:g}.")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum:g}.")
    return result


def _parse_vec3(value: Any, default: tuple[float, float, float], name: str) -> tuple[float, float, float]:
    if value is None:
        values: Sequence[Any] = default
    elif isinstance(value, torch.Tensor):
        values = value.detach().flatten().cpu().tolist()
    elif isinstance(value, str):
        text = value.strip()
        try:
            parsed = json.loads(text)
            values = parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            values = [part.strip() for part in text.strip("[]() ").replace(";", ",").split(",")]
    elif isinstance(value, Sequence):
        values = value
    else:
        raise ValueError(f"{name} must contain three numeric values.")
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values.")
    result = tuple(_finite_float(item, f"{name}[{index}]") for index, item in enumerate(values))
    return result  # type: ignore[return-value]


def _canonical_point(value: Any, channel: str, index: int) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"curves.{channel}[{index}] must be [x, y].")
    x = _finite_float(value[0], f"curves.{channel}[{index}].x")
    y = _finite_float(value[1], f"curves.{channel}[{index}].y")
    if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
        raise ValueError(f"curves.{channel}[{index}] must stay within 0..1.")
    return (round(x, 6), round(y, 6))


def _parse_curves(value: Any) -> dict[str, tuple[tuple[float, float], ...]]:
    if value in (None, ""):
        parsed: Any = _DEFAULT_CURVES
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"curves must be valid JSON: {exc.msg}.") from exc
    else:
        parsed = value
    if not isinstance(parsed, dict):
        raise ValueError("curves must be a JSON object.")
    result: dict[str, tuple[tuple[float, float], ...]] = {}
    for channel in ("rgb", "r", "g", "b"):
        raw_points = parsed.get(channel, _DEFAULT_CURVES[channel])
        if not isinstance(raw_points, list) or len(raw_points) < 2:
            raise ValueError(f"curves.{channel} must contain at least two points.")
        points = sorted(
            (_canonical_point(point, channel, index) for index, point in enumerate(raw_points)),
            key=lambda point: point[0],
        )
        for left, right in zip(points, points[1:]):
            if right[0] - left[0] <= _EPS:
                raise ValueError(f"curves.{channel} x coordinates must be strictly increasing.")
        result[channel] = tuple(points)
    return result


def _is_identity_curve(points: tuple[tuple[float, float], ...]) -> bool:
    return (
        abs(points[0][0]) <= _EPS
        and abs(points[0][1]) <= _EPS
        and abs(points[-1][0] - 1.0) <= _EPS
        and abs(points[-1][1] - 1.0) <= _EPS
        and all(abs(x - y) <= _EPS for x, y in points)
    )


def _endpoint_slope(h0: float, h1: float, delta0: float, delta1: float) -> float:
    slope = ((2.0 * h0 + h1) * delta0 - h0 * delta1) / (h0 + h1)
    if slope * delta0 <= 0.0:
        return 0.0
    if delta0 * delta1 < 0.0 and abs(slope) > abs(3.0 * delta0):
        return 3.0 * delta0
    return slope


def _pchip_slopes(points: tuple[tuple[float, float], ...]) -> tuple[float, ...]:
    count = len(points)
    h = [points[index + 1][0] - points[index][0] for index in range(count - 1)]
    delta = [(points[index + 1][1] - points[index][1]) / h[index] for index in range(count - 1)]
    if count == 2:
        return (delta[0], delta[0])
    slopes = [0.0] * count
    slopes[0] = _endpoint_slope(h[0], h[1], delta[0], delta[1])
    slopes[-1] = _endpoint_slope(h[-1], h[-2], delta[-1], delta[-2])
    for index in range(1, count - 1):
        if delta[index - 1] == 0.0 or delta[index] == 0.0 or delta[index - 1] * delta[index] < 0.0:
            slopes[index] = 0.0
            continue
        weight1 = 2.0 * h[index] + h[index - 1]
        weight2 = h[index] + 2.0 * h[index - 1]
        slopes[index] = (weight1 + weight2) / (
            weight1 / delta[index - 1] + weight2 / delta[index]
        )
    return tuple(slopes)


@lru_cache(maxsize=256)
def _curve_lut(points: tuple[tuple[float, float], ...]) -> torch.Tensor:
    """Build a monotone PCHIP LUT, using endpoint values outside point range."""
    slopes = _pchip_slopes(points)
    samples = torch.linspace(0.0, 1.0, _LUT_SIZE, dtype=torch.float64)
    output = torch.empty_like(samples)
    output[samples <= points[0][0]] = points[0][1]
    output[samples >= points[-1][0]] = points[-1][1]
    for index in range(len(points) - 1):
        x0, y0 = points[index]
        x1, y1 = points[index + 1]
        if index == len(points) - 2:
            selected = (samples >= x0) & (samples <= x1)
        else:
            selected = (samples >= x0) & (samples < x1)
        t = (samples[selected] - x0) / (x1 - x0)
        t2 = t * t
        t3 = t2 * t
        h = x1 - x0
        output[selected] = (
            (2.0 * t3 - 3.0 * t2 + 1.0) * y0
            + (t3 - 2.0 * t2 + t) * h * slopes[index]
            + (-2.0 * t3 + 3.0 * t2) * y1
            + (t3 - t2) * h * slopes[index + 1]
        )
    return output.clamp(0.0, 1.0).to(torch.float32)


def _curve_luts(
    curves: dict[str, tuple[tuple[float, float], ...]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        channel: _curve_lut(points).to(device=device, non_blocking=True)
        for channel, points in curves.items()
        if not _is_identity_curve(points)
    }


def _apply_lut(values: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    position = values.clamp(0.0, 1.0) * float(_LUT_SIZE - 1)
    lower = position.floor().to(torch.long)
    upper = (lower + 1).clamp_max(_LUT_SIZE - 1)
    fraction = position - lower.to(position.dtype)
    return torch.lerp(lut[lower], lut[upper], fraction)


def _cube_interp_1d(
    values: torch.Tensor,
    table: torch.Tensor,
    input_min: tuple[float, float, float],
    input_max: tuple[float, float, float],
) -> torch.Tensor:
    minimum = torch.tensor(input_min, device=values.device, dtype=values.dtype)
    span = torch.tensor(input_max, device=values.device, dtype=values.dtype) - minimum
    position = ((values - minimum) / span).clamp(0.0, 1.0) * float(table.shape[0] - 1)
    lower = position.floor().to(torch.long)
    upper = (lower + 1).clamp_max(table.shape[0] - 1)
    fraction = position - lower.to(position.dtype)
    table_channels = table.transpose(0, 1)
    flat_lower = table_channels.gather(1, lower.reshape(-1, 3).transpose(0, 1))
    flat_upper = table_channels.gather(1, upper.reshape(-1, 3).transpose(0, 1))
    lower_value = flat_lower.transpose(0, 1).reshape_as(values)
    upper_value = flat_upper.transpose(0, 1).reshape_as(values)
    return torch.lerp(lower_value, upper_value, fraction)


def _cube_interp_3d(
    values: torch.Tensor,
    table: torch.Tensor,
    input_min: tuple[float, float, float],
    input_max: tuple[float, float, float],
) -> torch.Tensor:
    minimum = torch.tensor(input_min, device=values.device, dtype=values.dtype)
    span = torch.tensor(input_max, device=values.device, dtype=values.dtype) - minimum
    position = ((values - minimum) / span).clamp(0.0, 1.0) * float(table.shape[0] - 1)
    lower = position.floor().to(torch.long)
    upper = (lower + 1).clamp_max(table.shape[0] - 1)
    fraction = position - lower.to(position.dtype)
    flat_lower = lower.reshape(-1, 3)
    flat_upper = upper.reshape(-1, 3)
    flat_fraction = fraction.reshape(-1, 3)
    red0, green0, blue0 = flat_lower.unbind(dim=1)
    red1, green1, blue1 = flat_upper.unbind(dim=1)
    fr, fg, fb = flat_fraction.unbind(dim=1)
    c000 = table[blue0, green0, red0]
    c100 = table[blue0, green0, red1]
    c010 = table[blue0, green1, red0]
    c110 = table[blue0, green1, red1]
    c001 = table[blue1, green0, red0]
    c101 = table[blue1, green0, red1]
    c011 = table[blue1, green1, red0]
    c111 = table[blue1, green1, red1]
    c00 = torch.lerp(c000, c100, fr.unsqueeze(-1))
    c10 = torch.lerp(c010, c110, fr.unsqueeze(-1))
    c01 = torch.lerp(c001, c101, fr.unsqueeze(-1))
    c11 = torch.lerp(c011, c111, fr.unsqueeze(-1))
    c0 = torch.lerp(c00, c10, fg.unsqueeze(-1))
    c1 = torch.lerp(c01, c11, fg.unsqueeze(-1))
    return torch.lerp(c0, c1, fb.unsqueeze(-1)).reshape_as(values)


def _apply_cube(values: torch.Tensor, cube: _CubeLUT | None) -> torch.Tensor:
    if cube is None:
        return values
    result = values
    if cube.one_d is not None:
        result = _cube_interp_1d(result, cube.one_d, cube.one_d_min, cube.one_d_max)
    if cube.three_d is not None:
        shape = result.shape
        flat = result.reshape(-1, 3)
        if flat.shape[0] > _CUBE_PIXEL_CHUNK:
            result = torch.cat(
                [
                    _cube_interp_3d(
                        flat[start : start + _CUBE_PIXEL_CHUNK],
                        cube.three_d,
                        cube.three_d_min,
                        cube.three_d_max,
                    )
                    for start in range(0, flat.shape[0], _CUBE_PIXEL_CHUNK)
                ],
                dim=0,
            ).reshape(shape)
        else:
            result = _cube_interp_3d(result, cube.three_d, cube.three_d_min, cube.three_d_max)
    return result


def _white_point_from_hex(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, str):
        raise ValueError("white_point must be a HEX colour in the form #RRGGBB.")
    text = value.strip()
    if len(text) != 7 or not text.startswith("#"):
        raise ValueError("white_point must be a HEX colour in the form #RRGGBB.")
    try:
        encoded = tuple(int(text[index : index + 2], 16) / 255.0 for index in (1, 3, 5))
    except ValueError as exc:
        raise ValueError("white_point must be a HEX colour in the form #RRGGBB.") from exc

    def linearise(channel: float) -> float:
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    protected = tuple(max(_EPS, linearise(channel)) for channel in encoded)
    mean = sum(protected) / 3.0
    if not math.isfinite(mean) or mean <= _EPS:
        raise ValueError("white_point produced an invalid linear Rec.709 value.")
    result = tuple(max(_EPS, channel / mean) for channel in protected)
    if not all(math.isfinite(channel) and channel >= _EPS for channel in result):
        raise ValueError("white_point produced a zero or non-finite channel.")
    return result  # type: ignore[return-value]


def _validated_parameters(
    white_point: Any,
    color_temperature: Any,
    tint: Any,
    offset: Any,
    multiply: Any,
    gamma: Any,
    brightness: Any,
    contrast: Any,
    saturation: Any,
    lut_strength: Any,
    rgb_offset: Any,
    rgb_multiply: Any,
    rgb_gamma: Any,
) -> dict[str, Any]:
    result = {
        "white_point": _white_point_from_hex(white_point),
        "color_temperature": _finite_float(color_temperature, "color_temperature", minimum=-1.0, maximum=1.0),
        "tint": _finite_float(tint, "tint", minimum=-1.0, maximum=1.0),
        "offset": _finite_float(offset, "offset", minimum=-1.0, maximum=1.0),
        "multiply": _finite_float(multiply, "multiply", minimum=0.0, maximum=2.0),
        "gamma": _finite_float(gamma, "gamma", minimum=_EPS, maximum=10.0),
        "brightness": _finite_float(brightness, "brightness", minimum=-1.0, maximum=1.0),
        "contrast": _finite_float(contrast, "contrast", minimum=-1.0, maximum=1.0),
        "saturation": _finite_float(saturation, "saturation", minimum=0.0, maximum=10.0),
        "lut_strength": _finite_float(lut_strength, "lut_strength", minimum=0.0, maximum=1.0),
        "rgb_offset": _parse_vec3(rgb_offset, (0.0, 0.0, 0.0), "rgb_offset"),
        "rgb_multiply": _parse_vec3(rgb_multiply, (1.0, 1.0, 1.0), "rgb_multiply"),
        "rgb_gamma": _parse_vec3(rgb_gamma, (1.0, 1.0, 1.0), "rgb_gamma"),
    }
    if any(value < _EPS for value in result["rgb_gamma"]):
        raise ValueError(f"every rgb_gamma channel must be at least {_EPS:g}.")
    return result


def _white_balance(
    white_point: tuple[float, float, float],
    color_temperature: float,
    tint: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    white = torch.tensor(white_point, device=device, dtype=dtype)
    white = white + torch.tensor(
        (
            -tint / 3.0 - color_temperature * 0.5,
            tint * 2.0 / 3.0,
            -tint / 3.0 + color_temperature * 0.5,
        ),
        device=device,
        dtype=dtype,
    )
    if not torch.isfinite(white).all().item():
        raise ValueError("the final white point contains a non-finite value.")
    if (white.abs() < _EPS).any().item():
        raise ValueError("the final white-point channels must not be zero.")
    mean = white.mean()
    if not torch.isfinite(mean).item() or abs(float(mean.item())) < _EPS:
        raise ValueError("the final white-point sum must not be zero.")
    return mean / white


def _signed_pow(value: torch.Tensor, exponent: torch.Tensor | float) -> torch.Tensor:
    return value.sign() * value.abs().pow(exponent)


def _grade_chunk(
    source: torch.Tensor,
    parameters: dict[str, Any],
    luts: dict[str, torch.Tensor],
    cube: _CubeLUT | None = None,
) -> torch.Tensor:
    result = source * _white_balance(
        parameters["white_point"],
        parameters["color_temperature"],
        parameters["tint"],
        source.device,
        source.dtype,
    )
    rgb_offset = torch.tensor(parameters["rgb_offset"], device=source.device, dtype=source.dtype)
    rgb_multiply = torch.tensor(parameters["rgb_multiply"], device=source.device, dtype=source.dtype)
    rgb_gamma = torch.tensor(parameters["rgb_gamma"], device=source.device, dtype=source.dtype)
    result = result + parameters["offset"] + rgb_offset
    result = result * parameters["multiply"] * rgb_multiply
    result = _signed_pow(result, 1.0 / parameters["gamma"])
    result = _signed_pow(result, 1.0 / rgb_gamma)
    contrast_factor = 1.0 + parameters["contrast"]
    result = result * contrast_factor + 0.5 * (1.0 - contrast_factor) + parameters["brightness"]
    luma = (
        result[..., 0] * 0.2126
        + result[..., 1] * 0.7152
        + result[..., 2] * 0.0722
    ).unsqueeze(-1)
    result = luma + parameters["saturation"] * (result - luma)
    if "rgb" in luts:
        result = _apply_lut(result, luts["rgb"])
    channels = []
    for index, channel in enumerate(("r", "g", "b")):
        values = result[..., index]
        channels.append(_apply_lut(values, luts[channel]) if channel in luts else values)
    before_external_lut = torch.stack(channels, dim=-1)
    if cube is None or parameters["lut_strength"] <= 0.0:
        return before_external_lut
    after_external_lut = _apply_cube(before_external_lut, cube)
    if parameters["lut_strength"] >= 1.0:
        return after_external_lut
    return torch.lerp(before_external_lut, after_external_lut, parameters["lut_strength"])


def _normalise_mask(mask: torch.Tensor | None, batch: int) -> torch.Tensor | None:
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        raise ValueError("mask must be a ComfyUI MASK tensor.")
    if mask.ndim == 2:
        value = mask.unsqueeze(0)
    elif mask.ndim == 3:
        value = mask
    elif mask.ndim == 4 and mask.shape[-1] >= 1:
        value = mask[..., 0]
    else:
        raise ValueError("mask must have shape [H,W], [B,H,W], or [B,H,W,1].")
    if value.shape[0] not in (1, batch):
        raise ValueError("mask batch size must be 1 or match image batch size.")
    if not torch.isfinite(value).all().item():
        raise ValueError("mask must contain only finite values.")
    return value


def _mask_chunk(
    mask: torch.Tensor | None,
    start: int,
    end: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor | None:
    if mask is None:
        return None
    selected = mask if mask.shape[0] == 1 else mask[start:end]
    selected = selected.to(device=device, dtype=torch.float32, non_blocking=True).unsqueeze(1)
    if selected.shape[-2:] != (height, width):
        selected = F.interpolate(selected, size=(height, width), mode="bilinear", align_corners=False)
    if selected.shape[0] == 1 and end - start > 1:
        selected = selected.expand(end - start, -1, -1, -1)
    return selected[:, 0].clamp(0.0, 1.0)


def _batch_size(image: torch.Tensor, device: torch.device) -> int:
    total = int(image.shape[0])
    if total <= 1:
        return 1
    height, width = int(image.shape[1]), int(image.shape[2])
    estimated_per_frame = max(1, height * width * 3 * 4 * 10)
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


@torch.no_grad()
def _run_color_grade(
    image: torch.Tensor,
    mask: torch.Tensor | None,
    parameters: dict[str, Any],
    curves: dict[str, tuple[tuple[float, float], ...]],
    lut: Any = _NO_EXTERNAL_LUT,
    progress: _GradeProgress | None = None,
) -> torch.Tensor:
    rgb = _as_rgb(image)
    total, height, width = map(int, rgb.shape[:3])
    normalised_mask = _normalise_mask(mask, total)
    device = _preferred_device(rgb.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = rgb.device
    luts = _curve_luts(curves, device)
    cube = _cube_on_device(_load_cube(lut), device)
    batch_size = _batch_size(rgb, device)
    _grade_info(
        "render setup: device=%s; frames=%d; frame_batch=%d; external_lut=%s; lut_strength=%.3f; active_curves=%s",
        device,
        total,
        batch_size,
        str(lut or _NO_EXTERNAL_LUT),
        parameters["lut_strength"],
        ",".join(sorted(luts)) if luts else "none",
    )
    output = torch.empty((total, height, width, 3), device="cpu", dtype=torch.float32)
    start = 0
    while start < total:
        end = min(total, start + batch_size)
        source = rgb[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        effect_mask = _mask_chunk(normalised_mask, start, end, height, width, device)
        try:
            graded = _grade_chunk(source, parameters, luts, cube)
            if effect_mask is not None:
                graded = torch.lerp(source, graded, effect_mask.unsqueeze(-1))
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or batch_size <= 1:
                raise
            del source, effect_mask
            if device.type == "cuda":
                torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            _LOGGER.warning("[CS Color Grade] CUDA OOM; retrying with batch=%d", batch_size)
            continue
        output[start:end].copy_(graded.to(device="cpu", dtype=torch.float32), non_blocking=True)
        processed = end - start
        del source, effect_mask, graded
        start = end
        if progress is not None:
            progress.update(processed)
    return output


def _cache_input(node_id: Any, prompt: Any, image: torch.Tensor, mask: torch.Tensor | None) -> None:
    key = str(node_id or "").strip()
    if not key or not isinstance(image, torch.Tensor) or image.ndim != 4:
        return
    loader_origin = _loader_id_from_prompt(prompt, node_id)
    cached_mask = None
    if isinstance(mask, torch.Tensor):
        try:
            cached_mask = _normalise_mask(mask, int(image.shape[0]))
            if cached_mask is not None:
                cached_mask = cached_mask.detach().to(device="cpu", dtype=torch.float32).contiguous()
        except ValueError:
            cached_mask = None
    entry = None
    try:
        cache_fingerprint = ""
        if cached_mask is not None:
            cache_fingerprint = f"mask:{_preview_cache_store().fingerprint_value(cached_mask)}"
        entry = _preview_cache_store().put_preview(
            key,
            image[..., :3],
            _source_fps_from_prompt(prompt, node_id),
            cache_fingerprint=cache_fingerprint,
            encode_video=not loader_origin,
        )
        if loader_origin:
            _LOGGER.info("[CS Color Grade] using shared loader preview cache: loader=%s", loader_origin)
    except Exception as exc:
        _LOGGER.warning("[CS Color Grade] preview cache failed for node %s: %s", key, exc)
    with _CACHE_LOCK:
        _MASK_CACHE[key] = cached_mask
        token = str((entry or {}).get("token") or "").strip()
        if token:
            _MASK_CACHE_BY_TOKEN[token] = cached_mask
            while len(_MASK_CACHE_BY_TOKEN) > 64:
                _MASK_CACHE_BY_TOKEN.pop(next(iter(_MASK_CACHE_BY_TOKEN)))


def _preview_mask_frame_index(node_id: str, frame_index: int, source_token: str) -> int:
    try:
        loader_cache = _loader_preview_cache()
        wait_cache_module = sys.modules.get(f"{__name__.rsplit('.', 1)[0]}._py_preview_cache")
        source_entry = (
            loader_cache.entry_for_token(source_token)
            if source_token.startswith("loader_preview:") and loader_cache is not None
            else wait_cache_module.get_wait_input_cache_store().get_token(source_token)
            if source_token.startswith("wait_input:") and wait_cache_module is not None
            else _preview_cache_store().get_token(source_token)
        )
        original_entry = (
            source_entry
            if source_entry is not None and not source_token.startswith("loader_preview:")
            else _preview_cache_store().get_preview_variant(node_id, proxy=False)
        )
        source_count = int((source_entry or {}).get("info", {}).get("frames") or 0)
        original_count = int((original_entry or {}).get("info", {}).get("frames") or 0)
        if source_count > 1 and original_count > 1 and source_count != original_count:
            return int(round(frame_index * (original_count - 1) / (source_count - 1)))
    except Exception:
        pass
    return frame_index


def _preview_mask(
    node_id: str,
    frame_index: int,
    height: int,
    width: int,
    source_token: str = "",
) -> torch.Tensor | None:
    with _CACHE_LOCK:
        mask = _MASK_CACHE_BY_TOKEN.get(source_token) if source_token else None
        if source_token and source_token not in _MASK_CACHE_BY_TOKEN:
            mask = _MASK_CACHE.get(node_id)
        elif not source_token:
            mask = _MASK_CACHE.get(node_id)
    if mask is None or mask.ndim != 3 or frame_index < 0:
        return None
    if frame_index >= mask.shape[0] and mask.shape[0] != 1:
        return None
    frame = mask[0:1] if mask.shape[0] == 1 else mask[frame_index : frame_index + 1]
    if frame.shape[1:3] != (height, width):
        frame = F.interpolate(
            frame.unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
    return frame


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
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


async def _cache_info_route(request: web.Request) -> web.Response:
    node_id = str(request.query.get("node_id") or "").strip()
    entry = _preview_cache_store().get_preview(node_id)
    if entry is None:
        return web.json_response(
            {"error": "Run CS Color Grade once to cache its connected input."},
            status=404,
        )
    with _CACHE_LOCK:
        has_mask = _MASK_CACHE.get(node_id) is not None
    return web.json_response(
        {
            "token": str(entry.get("token") or ""),
            "label": "Cached CS Color Grade input",
            "video_url": f"/cinestyle/color-grade-cache-video?token={entry.get('token')}",
            "info": dict(entry.get("info") or {}),
            "has_mask": has_mask,
        }
    )


async def _cache_video_route(request: web.Request) -> web.StreamResponse:
    entry = _preview_cache_store().get_token(request.query.get("token", ""))
    path = Path(str((entry or {}).get("video_path") or (entry or {}).get("path") or "")) if entry else None
    if entry is None or path is None or not path.is_file():
        return web.json_response({"error": "CS Color Grade preview cache not found."}, status=404)
    return web.FileResponse(path=path, headers={"Content-Type": "video/mp4", "Cache-Control": "no-store"})


async def _preview_route(request: web.Request) -> web.Response:
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
        elif source_token.startswith("wait_input:"):
            package = __name__.rsplit(".", 1)[0]
            module = sys.modules.get(f"{package}._py_preview_cache")
            if module is None:
                raise ValueError("The shared wait input preview cache is unavailable.")
            frame = module.get_wait_input_cache_store().decode_frame(payload, frame_index)
        else:
            frame = _preview_cache_store().decode_frame(payload, frame_index)
        height, width = int(frame.shape[1]), int(frame.shape[2])
        mask_index = _preview_mask_frame_index(node_id, frame_index, source_token)
        mask = _preview_mask(node_id, mask_index, height, width, source_token)
        parameters = _validated_parameters(
            payload.get("white_point", "#FFFFFF"),
            payload.get("color_temperature", 0.0),
            payload.get("tint", 0.0),
            payload.get("offset", 0.0),
            payload.get("multiply", 1.0),
            payload.get("gamma", 1.0),
            payload.get("brightness", 0.0),
            payload.get("contrast", 0.0),
            payload.get("saturation", 1.0),
            payload.get("lut_strength", 1.0),
            payload.get("rgb_offset", [0.0, 0.0, 0.0]),
            payload.get("rgb_multiply", [1.0, 1.0, 1.0]),
            payload.get("rgb_gamma", [1.0, 1.0, 1.0]),
        )
        curves = _parse_curves(payload.get("curves", _DEFAULT_CURVES))
        output = _run_color_grade(frame, mask, parameters, curves, payload.get("lut", _NO_EXTERNAL_LUT))
        return web.json_response(
            {
                "frame": int(payload.get("local_frame", frame_index)),
                "original": _encode_preview_png(frame),
                "preview": _encode_preview_png(output),
            }
        )
    except (ValueError, TypeError, KeyError, IndexError, RuntimeError, json.JSONDecodeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)


class CSColorGrade(io.ComfyNode):
    """Torch port of AFX Grade with RGB white balance and channel curves."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CS_Color_Grade",
            display_name="CS Color Grade",
            category=_CATEGORY,
            essentials_category="Image Effects",
            search_aliases=["color grade", "colour grade", "AFX Grade", "RGB curves", "LUT", "cube LUT"],
            description="RGB white balance, AFX/Nuke-style grade controls, RGB channel controls, and PCHIP curves.",
            inputs=[
                io.Image.Input("image", tooltip="Standard ComfyUI IMAGE. RGBA inputs ignore the fourth channel."),
                io.Mask.Input("mask", optional=True, tooltip="Optional effect mask; black keeps the source and white applies the grade."),
                io.Combo.Input(
                    "lut",
                    display_name="Load LUT",
                    options=_lut_files(),
                    default=_NO_EXTERNAL_LUT,
                    tooltip="Optional .cube LUT from ComfyUI/models/luts; applied after the built-in curves.",
                ),
                io.String.Input(
                    "white_point",
                    display_name="White Point",
                    default="#FFFFFF",
                    tooltip="sRGB HEX white point converted to linear Rec.709 and normalized by channel mean.",
                ),
                io.Float.Input("color_temperature", display_name="Color Temperature", default=0.0, min=-1.0, max=1.0, step=0.001),
                io.Float.Input("tint", default=0.0, min=-1.0, max=1.0, step=0.001),
                io.Float.Input("offset", default=0.0, min=-1.0, max=1.0, step=0.001),
                io.Float.Input("multiply", default=1.0, min=0.0, max=2.0, step=0.001),
                io.Float.Input("gamma", default=1.0, min=_EPS, max=10.0, step=0.001),
                io.Float.Input("brightness", default=0.0, min=-1.0, max=1.0, step=0.001),
                io.Float.Input("contrast", default=0.0, min=-1.0, max=1.0, step=0.001),
                io.Float.Input("saturation", default=1.0, min=0.0, max=10.0, step=0.001),
                io.String.Input(
                    "rgb_offset",
                    display_name="RGB Offset [R,G,B]",
                    default="[0.0,0.0,0.0]",
                    advanced=True,
                ),
                io.String.Input(
                    "rgb_multiply",
                    display_name="RGB Multiply [R,G,B]",
                    default="[1.0,1.0,1.0]",
                    advanced=True,
                ),
                io.String.Input(
                    "rgb_gamma",
                    display_name="RGB Gamma [R,G,B]",
                    default="[1.0,1.0,1.0]",
                    advanced=True,
                ),
                io.String.Input(
                    "curves",
                    default=_DEFAULT_CURVES_JSON,
                    multiline=True,
                    advanced=True,
                    tooltip="Versioned JSON control points for RGB, R, G, and B PCHIP curves.",
                ),
                io.Float.Input(
                    "lut_strength",
                    display_name="LUT Strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    tooltip="Blend between the image before the external LUT (0) and the full LUT result (1).",
                ),
                io.Boolean.Input(
                    "wait_for_input_cache",
                    display_name="wait for input cache",
                    default=False,
                    advanced=True,
                    tooltip="Interrupt execution when this node is reached after caching its input.",
                ),
            ],
            outputs=[io.Image.Output("image", display_name="IMAGE")],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    @torch.no_grad()
    def execute(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor | None = None,
        wait_for_input_cache: bool = False,
        lut: str = _NO_EXTERNAL_LUT,
        lut_strength: float = 1.0,
        white_point: str = "#FFFFFF",
        color_temperature: float = 0.0,
        tint: float = 0.0,
        offset: float = 0.0,
        multiply: float = 1.0,
        gamma: float = 1.0,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 1.0,
        rgb_offset: Any = "[0.0,0.0,0.0]",
        rgb_multiply: Any = "[1.0,1.0,1.0]",
        rgb_gamma: Any = "[1.0,1.0,1.0]",
        curves: Any = _DEFAULT_CURVES_JSON,
    ) -> io.NodeOutput:
        started_at = time.perf_counter()
        node_id = getattr(getattr(cls, "hidden", None), "unique_id", "")
        prompt = getattr(getattr(cls, "hidden", None), "prompt", None)
        _grade_info("start: frames=%d", int(image.shape[0]))
        _grade_info("stage 1/5: caching source preview")
        _cache_input(node_id, prompt, image, mask)
        if bool(wait_for_input_cache):
            _cache_wait_input(node_id, prompt, image)
            from comfy.model_management import InterruptProcessingException

            raise InterruptProcessingException()
        _grade_info(
            "source ready: frames=%d; size=%dx%d; mask=%s",
            int(image.shape[0]),
            int(image.shape[2]),
            int(image.shape[1]),
            "available" if mask is not None else "none",
        )
        _grade_info("stage 2/5: validating grade parameters")
        parameters = _validated_parameters(
            white_point,
            color_temperature,
            tint,
            offset,
            multiply,
            gamma,
            brightness,
            contrast,
            saturation,
            lut_strength,
            rgb_offset,
            rgb_multiply,
            rgb_gamma,
        )
        _grade_info("stage 3/5: preparing curves and external LUT")
        parsed_curves = _parse_curves(curves)
        active_curves = tuple(
            channel for channel, points in parsed_curves.items() if not _is_identity_curve(points)
        )
        _grade_info(
            "grade setup ready: active_curves=%s; external_lut=%s; lut_strength=%.3f",
            ",".join(active_curves) if active_curves else "none",
            str(lut or _NO_EXTERNAL_LUT),
            lut_strength,
        )
        _grade_info("stage 4/5: rendering frames")
        progress = _GradeProgress(int(image.shape[0]))
        try:
            output = _run_color_grade(image, mask, parameters, parsed_curves, lut, progress)
        finally:
            progress.close()
        _grade_info(
            "stage 5/5: complete, output frames=%d; elapsed=%.2fs",
            int(output.shape[0]),
            time.perf_counter() - started_at,
        )
        return io.NodeOutput(output)


class ColorGradeExtension(ComfyExtension):
    @override
    async def on_load(self) -> None:
        global _ROUTES_REGISTERED
        if _ROUTES_REGISTERED:
            return
        from server import PromptServer

        server_instance = getattr(PromptServer, "instance", None)
        if server_instance is not None:
            server_instance.routes.get("/cinestyle/color-grade-cache")(_cache_info_route)
            server_instance.routes.get("/cinestyle/color-grade-cache-video")(_cache_video_route)
            server_instance.routes.post("/cinestyle/color-grade-preview")(_preview_route)
            _ROUTES_REGISTERED = True

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CSColorGrade]


async def comfy_entrypoint() -> ColorGradeExtension:
    return ColorGradeExtension()
