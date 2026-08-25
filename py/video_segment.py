"""Interactive multi-object SAM3.1 and SeC-4B video segmentation."""

from __future__ import annotations

import json
import math
import base64
import io as py_io
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import logging
import importlib
import warnings
from fractions import Fraction
import uuid
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
import torch.nn.functional as F
from aiohttp import web
from PIL import Image
from typing_extensions import override

import comfy.model_management
import comfy.model_detection
import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, io
from tqdm import tqdm


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"Importing from timm\.models\.layers is deprecated.*",
)


NODE_ID = "CS_Video_Segment_SAM3"
PROMPT_VERSION = 2
_PROPAGATION_OPTIONS = ["both", "forward", "backward"]
_PREVIEW_ROUTE_REGISTERED = False
_LAST_MODEL: Any = None
# SAM3.1 preview models are keyed by source path. ComfyUI's ModelPatcher
# remains responsible for GPU residency; this cache only avoids loading a
# second Python model object when Preview is clicked repeatedly.
_PREVIEW_MODEL_CACHE: dict[str, Any] = {}
_SAM3_CLIP_CACHE: dict[str, Any] = {}
_LAST_MODEL_SOURCE: dict[str, Any] | None = None
_SEC_MODEL_REGISTRY: dict[str, Any] = {}
_SEC_MODEL_NODE_TOKENS: dict[str, str] = {}
_SEC_MODEL_LOCK = threading.RLock()
_SEC_MODEL_LOAD_LOCK = threading.Lock()
_SEC_MODEL_DOWNLOAD_LOCK = threading.Lock()
_SEC_PREVIEW_MODEL_TOKEN: str | None = None
_SEC_PACKAGE_PATH = Path(__file__).resolve().parent / "sec_inference"
_SEC_CONFIG_PATH = Path(__file__).resolve().parent / "sec_configs"
_SEC_MODEL_CONFIG_PATH = Path(__file__).resolve().parent / "sec_model_config"
_SEC_MODEL_FOLDER = "sec_models"
_SEC_WEIGHT_SPECS = {
    "SeC-4B-bf16.safetensors": (torch.bfloat16, "https://huggingface.co/VeryAladeen/Sec-4B/resolve/main/SeC-4B-bf16.safetensors"),
    "SeC-4B-fp16.safetensors": (torch.float16, "https://huggingface.co/VeryAladeen/Sec-4B/resolve/main/SeC-4B-fp16.safetensors"),
}
_SEC_DEFAULT_WEIGHT_FILENAME = "SeC-4B-bf16.safetensors"
_SELECTOR_CACHE_LIMIT = 8
_SELECTOR_CACHE_MAX_BYTES = 4 * 1024**3
_SEGMENT_LOGGER = logging.getLogger("CineStyleVideoSegment")
_NESTED_TQDM_LOCK = threading.Lock()
_PREVIEW_CACHE_STORE = None


def _preview_cache_store():
    global _PREVIEW_CACHE_STORE
    if _PREVIEW_CACHE_STORE is None:
        package = __name__.rsplit(".", 1)[0]
        module = sys.modules.get(f"{package}._py_preview_cache")
        if module is None:
            raise RuntimeError("CineStyle preview cache module is unavailable.")
        _PREVIEW_CACHE_STORE = module.PreviewCacheStore("video_segment", max_entries=_SELECTOR_CACHE_LIMIT, max_bytes=_SELECTOR_CACHE_MAX_BYTES)
    return _PREVIEW_CACHE_STORE


def _no_nested_tqdm(iterable: Any, *args: Any, **kwargs: Any) -> Any:
    return iterable


class _NestedTqdmSilencer:
    """Temporarily hide progress bars emitted inside the bundled model kernels."""

    def __init__(self, module_names: tuple[str, ...]):
        self.module_names = module_names
        self._patched: list[tuple[Any, Any]] = []
        self._locked = False

    def start(self) -> None:
        _NESTED_TQDM_LOCK.acquire()
        self._locked = True
        for module_name in self.module_names:
            module = sys.modules.get(module_name)
            if module is None:
                try:
                    module = importlib.import_module(module_name)
                except Exception:
                    continue
            if module is None or not hasattr(module, "tqdm"):
                continue
            self._patched.append((module, module.tqdm))
            module.tqdm = _no_nested_tqdm

    def stop(self) -> None:
        for module, original in reversed(self._patched):
            module.tqdm = original
        self._patched.clear()
        if self._locked:
            self._locked = False
            _NESTED_TQDM_LOCK.release()


def _segment_info(node_name: str, message: str) -> None:
    _SEGMENT_LOGGER.info("[%s] %s", node_name, message)


class _SegmentProgress:
    """Forward ComfyUI progress updates while emitting throttled tqdm-style logs."""

    def __init__(self, node_name: str, total: int, backend: Any = None):
        self.node_name = node_name
        self.total = max(1, int(total))
        self.backend = backend
        self.bar = tqdm(
            total=self.total,
            desc=f"[INFO] [{node_name}] frame processing",
            unit="frame",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            mininterval=0.1,
            dynamic_ncols=True,
            leave=True,
        )

    def update(self, amount: int = 1) -> None:
        step = max(0, int(amount))
        if self.backend is not None:
            self.backend.update(step)
        self.bar.update(step)

    def close(self) -> None:
        self.bar.close()


def _segment_expected_frames(frame_count: int, anchor: int, direction: str, limit: int | None = None) -> int:
    direction = "both" if direction == "bidirectional" else direction
    cap = None if limit is None or int(limit) < 0 else max(1, int(limit))
    total = 0
    if direction in {"both", "forward"} and anchor + 1 < frame_count:
        total += min(cap, frame_count - anchor) if cap is not None else frame_count - anchor
    if direction in {"both", "backward"} and anchor > 0:
        total += min(cap, anchor + 1) if cap is not None else anchor + 1
    return max(1, total)

try:
    folder_paths.add_model_folder_path("sams", os.path.join(folder_paths.models_dir, "sams"))
except Exception:
    pass


def _video_files() -> list[str]:
    import os

    input_dir = folder_paths.get_input_directory()
    names = [
        name
        for name in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, name))
    ]
    return sorted(folder_paths.filter_files_content_types(names, ["video"]))


def _resolve_video_path(video: str) -> str:
    value = str(video or "").strip()
    if not value:
        raise ValueError("Choose a video file before requesting a preview.")
    if folder_paths.exists_annotated_filepath(value):
        return folder_paths.get_annotated_filepath(value)
    candidate = Path(os.path.expandvars(os.path.expanduser(value))).resolve()
    if candidate.is_file():
        return str(candidate)
    raise ValueError(f"Video file not found: {value}")


def _decode_video(video: str) -> torch.Tensor:
    from comfy_api.latest import InputImpl

    components = InputImpl.VideoFromFile(
        _resolve_video_path(video)
    ).get_components()
    images = components.images
    if not isinstance(images, torch.Tensor):
        raise ValueError("Video decoder did not return a tensor of frames.")
    return images


def _decode_video_frame(video: str, frame_index: int) -> torch.Tensor:
    path = _resolve_video_path(video)
    target = max(0, int(frame_index))
    with av.open(path, mode="r") as container:
        if not container.streams.video:
            raise ValueError("The selected file contains no video stream.")
        stream = container.streams.video[0]
        for index, decoded in enumerate(container.decode(stream)):
            if index == target:
                array = decoded.to_ndarray(format="rgb24")
                return torch.from_numpy(array).unsqueeze(0).float().div_(255.0)
    raise ValueError(f"Frame {target} is outside the selected video.")


def _looks_like_image_file(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if " [" in text:
        text = text.split(" [", 1)[0]
    return text.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".avif"))


def _resolve_image_path(image: str) -> str:
    value = str(image or "").strip()
    if not value:
        raise ValueError("Choose an image file before requesting a preview.")
    if folder_paths.exists_annotated_filepath(value):
        return folder_paths.get_annotated_filepath(value)
    candidate = Path(os.path.expandvars(os.path.expanduser(value))).resolve()
    if candidate.is_file():
        return str(candidate)
    raise ValueError(f"Image file not found: {value}")


def _decode_image_frame(image: str, frame_index: int) -> torch.Tensor:
    if int(frame_index) != 0:
        raise ValueError("A Load Image source contains one frame.")
    with Image.open(_resolve_image_path(image)) as decoded:
        array = np.asarray(decoded.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).unsqueeze(0).float().div_(255.0)


def _prompt_node(prompt: Any, node_id: Any) -> dict[str, Any] | None:
    if not isinstance(prompt, dict):
        return None
    return prompt.get(str(node_id)) or prompt.get(node_id)


def _looks_like_video_file(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if " [" in text:
        text = text.split(" [", 1)[0]
    return text.endswith((".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv"))


def _prompt_has_file_video_source(prompt: Any, node_id: Any) -> bool:
    node = _prompt_node(prompt, node_id)
    if not node:
        return False
    inputs = node.get("inputs") if isinstance(node, dict) else None
    pending: list[Any] = []
    if isinstance(inputs, dict):
        pending.extend(inputs.get(name) for name in ("images", "video_input"))
    visited: set[str] = set()
    while pending:
        link = pending.pop(0)
        if not isinstance(link, (list, tuple)) or len(link) < 2:
            continue
        upstream_id = str(link[0])
        if upstream_id in visited:
            continue
        visited.add(upstream_id)
        upstream = _prompt_node(prompt, upstream_id)
        if not isinstance(upstream, dict):
            continue
        class_type = str(upstream.get("class_type") or "")
        upstream_inputs = upstream.get("inputs")
        if not isinstance(upstream_inputs, dict):
            continue
        if re.search(r"load.*video|video.*load", class_type, re.IGNORECASE):
            return True
        for name, value in upstream_inputs.items():
            if any(token in str(name).lower() for token in ("video", "file", "filename", "path")):
                if _looks_like_video_file(value):
                    return True
        pending.extend(
            value
            for value in upstream_inputs.values()
            if isinstance(value, (list, tuple)) and len(value) >= 2
        )
    return False


def _prompt_selector_fps(prompt: Any, node_id: Any) -> float | None:
    node = _prompt_node(prompt, node_id)
    if not node:
        return None
    inputs = node.get("inputs") if isinstance(node, dict) else None
    pending: list[Any] = []
    if isinstance(inputs, dict):
        pending.extend(inputs.get(name) for name in ("images", "video_input"))
    visited: set[str] = set()
    while pending:
        link = pending.pop(0)
        if not isinstance(link, (list, tuple)) or len(link) < 2:
            continue
        upstream_id = str(link[0])
        if upstream_id in visited:
            continue
        visited.add(upstream_id)
        upstream = _prompt_node(prompt, upstream_id)
        if not isinstance(upstream, dict):
            continue
        upstream_inputs = upstream.get("inputs")
        if not isinstance(upstream_inputs, dict):
            continue
        for name in ("fps", "frame_rate", "target_fps"):
            value = upstream_inputs.get(name)
            if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0:
                return float(value)
        pending.extend(
            value
            for value in upstream_inputs.values()
            if isinstance(value, (list, tuple)) and len(value) >= 2
        )
    return None


def _video_input_fps(video_input: Any, prompt: Any = None, node_id: Any = None) -> float:
    if video_input is not None:
        try:
            fps = float(video_input.get_components().frame_rate)
            if math.isfinite(fps) and fps > 0:
                return fps
        except Exception:
            pass
    return _prompt_selector_fps(prompt, node_id) or 24.0


def _cache_selector_input(node_id: Any, images: torch.Tensor, fps: float) -> str | None:
    key = str(node_id or "").strip()
    if not key or not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] == 0:
        return None
    safe_fps = float(fps) if math.isfinite(float(fps)) and float(fps) > 0 else 24.0
    try:
        entry = _preview_cache_store().put(key, images, safe_fps, encode_video=True)
    except Exception as exc:
        print(f"[CineStyle] Selector input cache failed for node {key}: {exc}")
        return None
    print(
        f"[CineStyle] Cached selector input for node {key}: "
        f"{entry['info']['frames']} frames at {safe_fps:.3f} fps."
    )
    return str(entry["token"])


def _selector_cache_for_node(node_id: Any) -> dict[str, Any] | None:
    return _preview_cache_store().get_node(node_id)


def _selector_cache_for_token(token: Any) -> dict[str, Any] | None:
    return _preview_cache_store().get_token(token)


def _decode_selector_frame(payload: dict[str, Any], frame_index: int) -> torch.Tensor:
    token = str(payload.get("source_token") or "").strip()
    if not token:
        source = str(payload.get("video") or "")
        if str(payload.get("source_kind") or "").lower() == "image" or _looks_like_image_file(source):
            return _decode_image_frame(source, frame_index)
        return _decode_video_frame(source, frame_index)
    entry = _selector_cache_for_token(token)
    if entry is None:
        raise ValueError("The cached Selector input is no longer available. Run the workflow once again.")
    try:
        frames = np.load(str(entry["frames_path"]), mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError("The cached Selector frames are unavailable. Run the workflow once again.") from exc
    target = int(frame_index)
    if target < 0 or target >= int(frames.shape[0]):
        raise ValueError(f"Frame {target} is outside the cached input.")
    frame = np.array(frames[target : target + 1], copy=True)
    return torch.from_numpy(frame).to(torch.float32).div_(255.0)


async def _selector_cache_info_route(request: web.Request) -> web.Response:
    requested_node_id = str(request.query.get("node_id", ""))
    entry = _selector_cache_for_node(requested_node_id)
    if entry is None:
        return web.json_response({"error": "No cached Selector input."}, status=404)
    token = str(entry["token"])
    return web.json_response(
        {
            "token": token,
            "label": "Proxy input from the last workflow run" if requested_node_id.endswith(":proxy") else "Cached input from the last workflow run",
            "video_url": f"/cinestyle/video-selector-cache-video?token={token}",
            "info": entry["info"],
        }
    )


async def _selector_cache_video_route(request: web.Request) -> web.StreamResponse:
    entry = _selector_cache_for_token(request.query.get("token", ""))
    if entry is None or not Path(str(entry.get("path") or "")).is_file():
        return web.json_response({"error": "Cached Selector video not found."}, status=404)
    return web.FileResponse(
        path=str(entry["path"]),
        headers={"Cache-Control": "no-store"},
    )


def _parse_json(value: str | None, name: str) -> Any:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON: {exc.msg}.") from exc


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _pixel_coordinate(value: Any, size: int, name: str) -> float:
    result = _number(value, name)
    # The selector writes normalized coordinates. Pixel coordinates are also
    # accepted so values copied from the official SAM3 Detect node work too.
    if 0.0 <= result <= 1.0:
        result *= size
    return max(0.0, min(float(size), result))


def _parse_points(value: str | None, width: int, height: int) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    raw = _parse_json(value, "points")
    if raw is None:
        return [], []
    if isinstance(raw, dict):
        raw = raw.get("points", [raw])
    if not isinstance(raw, list):
        raise ValueError("points must be a JSON list of {x, y, label} objects.")

    positive: list[dict[str, float]] = []
    negative: list[dict[str, float]] = []
    for index, item in enumerate(raw):
        if isinstance(item, dict):
            if "x" not in item or "y" not in item:
                raise ValueError(f"points[{index}] needs x and y.")
            label = item.get("label", 1)
            if isinstance(label, str):
                label = 0 if label.lower() in {"negative", "neg", "background", "0"} else 1
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            label = item[2] if len(item) >= 3 else 1
            item = {"x": item[0], "y": item[1]}
        else:
            raise ValueError(f"points[{index}] must be an object or [x, y, label].")

        point = {
            "x": _pixel_coordinate(item["x"], width, f"points[{index}].x"),
            "y": _pixel_coordinate(item["y"], height, f"points[{index}].y"),
        }
        (negative if int(label) == 0 else positive).append(point)

    if not positive and not negative:
        raise ValueError("points must contain at least one point.")
    return positive, negative


def _parse_bbox(value: str | None, width: int, height: int) -> dict[str, float]:
    raw = _parse_json(value, "bbox")
    if raw is None:
        raise ValueError("bbox mode requires a JSON box.")
    if isinstance(raw, list):
        if len(raw) == 1 and isinstance(raw[0], (dict, list)):
            raw = raw[0]
        elif len(raw) >= 4:
            raw = {"x": raw[0], "y": raw[1], "w": raw[2], "h": raw[3]}
    if not isinstance(raw, dict):
        raise ValueError("bbox must be {x, y, w, h} or [x, y, w, h].")

    x = _pixel_coordinate(raw.get("x", 0), width, "bbox.x")
    y = _pixel_coordinate(raw.get("y", 0), height, "bbox.y")
    width_value = raw.get("w", raw.get("width"))
    height_value = raw.get("h", raw.get("height"))
    if width_value is None or height_value is None:
        raise ValueError("bbox needs w/h (or width/height).")
    box_width = _number(width_value, "bbox.w")
    box_height = _number(height_value, "bbox.h")
    if 0.0 <= box_width <= 1.0:
        box_width *= width
    if 0.0 <= box_height <= 1.0:
        box_height *= height
    box_width = min(float(width) - x, box_width)
    box_height = min(float(height) - y, box_height)
    if box_width <= 0 or box_height <= 0:
        raise ValueError("bbox must have positive width and height inside the frame.")
    return {"x": x, "y": y, "width": box_width, "height": box_height}


def _decode_prompt_mask(value: Any, width: int, height: int) -> torch.Tensor | None:
    """Decode a Selector PNG mask into a soft ``[H, W]`` tensor."""
    if not value:
        return None
    if isinstance(value, dict):
        encoded = value.get("data") or value.get("png") or value.get("base64")
    else:
        encoded = value
    text = str(encoded or "").strip()
    if not text:
        return None
    if "," in text and text.lower().startswith("data:"):
        text = text.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(text, validate=True)
        with Image.open(py_io.BytesIO(image_bytes)) as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    except Exception as exc:
        raise ValueError("prompt_data contains an invalid mask PNG.") from exc
    alpha = rgba[..., 3]
    if alpha.size == 0 or float(alpha.max()) <= 0.0:
        alpha = rgba[..., :3].mean(axis=-1)
    mask = torch.from_numpy(alpha.copy()).float()
    if tuple(mask.shape) != (height, width):
        mask = F.interpolate(mask[None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    if not bool((mask > 0.01).any()):
        return None
    return mask.clamp_(0.0, 1.0)


def _parse_prompt_data(value: str | None, width: int, height: int) -> list[dict[str, Any]]:
    """Parse the Selector's versioned per-object mask/box/point protocol."""
    raw = _parse_json(value, "prompt_data")
    if raw is None:
        raise ValueError("prompt_data is empty. Open the Selector and define at least one object.")
    objects = raw.get("objects") if isinstance(raw, dict) else raw
    if not isinstance(objects, list):
        raise ValueError("prompt_data must contain an objects list.")
    parsed_objects: list[dict[str, Any]] = []
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            raise ValueError(f"prompt_data.objects[{index}] must be an object.")
        raw_points = item.get("points") or []
        points: list[dict[str, float | int]] = []
        if raw_points:
            positive, negative = _parse_points(json.dumps(raw_points), width, height)
            points = [{**point, "label": 1} for point in positive] + [{**point, "label": 0} for point in negative]
        box = None
        raw_box = item.get("bbox") or item.get("box")
        if raw_box:
            box = _parse_bbox(json.dumps(raw_box), width, height)
        mask = _decode_prompt_mask(item.get("mask"), width, height)
        text = str(item.get("text") or item.get("semantic") or "").strip()
        if not text and not points and box is None and mask is None:
            raise ValueError(f"prompt_data.objects[{index}] has no semantic, mask, bbox, or point prompt.")
        parsed_objects.append({"text": text, "points": points, "bbox": box, "mask": mask})
    if not parsed_objects:
        raise ValueError("prompt_data must contain at least one prompted object.")
    return parsed_objects


def _sam3_mask_logits(mask: torch.Tensor, device: Any, dtype: torch.dtype) -> torch.Tensor:
    """Convert a ComfyUI 0..1 brush mask into a coarse SAM3 prompt logit."""
    # SAM3 receives decoder-style logits rather than a binary output mask.
    return torch.logit(mask.to(device=device, dtype=dtype).clamp(0.05, 0.95))[None, None]


def _preview_model(source: Any) -> Any:
    global _LAST_MODEL_SOURCE
    if isinstance(source, dict) and source.get("name"):
        _LAST_MODEL_SOURCE = dict(source)
    if not isinstance(source, dict):
        return _LAST_MODEL
    kind = str(source.get("kind") or "")
    name = str(source.get("name") or "")
    if not name:
        return _LAST_MODEL
    if kind == "checkpoint":
        path = folder_paths.get_full_path_or_raise("checkpoints", name)
        model = _PREVIEW_MODEL_CACHE.get(path)
        if model is None:
            model, clip, _, _ = comfy.sd.load_checkpoint_guess_config(
                path,
                output_vae=False,
                output_clip=True,
                output_clipvision=False,
                output_model=True,
            )
            if model is None:
                raise ValueError(f"Unable to load SAM3 checkpoint: {name}")
            _PREVIEW_MODEL_CACHE[path] = model
            if clip is not None:
                _SAM3_CLIP_CACHE[path] = clip
        return model
    if kind == "diffusion_model":
        path = folder_paths.get_full_path_or_raise("diffusion_models", name)
        model = _PREVIEW_MODEL_CACHE.get(path)
        if model is None:
            model = comfy.sd.load_diffusion_model(path)
            _PREVIEW_MODEL_CACHE[path] = model
        return model
    return _LAST_MODEL


def _sam3_clip_from_path(path: str) -> Any:
    """Load the SAM3 text encoder paired with a checkpoint, once per path."""
    clip = _SAM3_CLIP_CACHE.get(path)
    if clip is not None:
        return clip
    state_dict, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    prefix = comfy.model_detection.unet_prefix_from_state_dict(state_dict)
    config = comfy.model_detection.model_config_from_unet(state_dict, prefix, metadata=metadata)
    if config is None:
        raise ValueError(f"Unable to identify SAM3 checkpoint: {Path(path).name}")
    # The official ComfyUI SAM3 config stashes its embedded CLIP weights while
    # normalizing the model state dict. This avoids constructing a duplicate
    # full SAM3 model just to obtain the text encoder.
    config.process_unet_state_dict(state_dict)
    clip_sd = config.process_clip_state_dict({})
    if not clip_sd:
        raise ValueError(f"SAM3 checkpoint has no text encoder: {Path(path).name}")
    clip_target = config.clip_target(state_dict=clip_sd)
    if clip_target is None:
        raise ValueError(f"SAM3 checkpoint has no compatible text encoder: {Path(path).name}")
    clip = comfy.sd.CLIP(
        clip_target,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        tokenizer_data=clip_sd,
        parameters=comfy.utils.calculate_parameters(clip_sd),
        state_dict=clip_sd,
    )
    _SAM3_CLIP_CACHE[path] = clip
    return clip


def _sam3_clip_from_model(model: Any) -> Any:
    """Build the paired Comfy CLIP from the SAM3 model's stashed text weights."""
    cache_key = f"model:{id(model)}"
    if cache_key in _SAM3_CLIP_CACHE:
        return _SAM3_CLIP_CACHE[cache_key]
    base = getattr(model, "model", None)
    config = getattr(base, "model_config", None)
    stash = getattr(config, "_clip_stash", None)
    if not stash:
        return None
    clip_sd = config.process_clip_state_dict({})
    if not clip_sd:
        return None
    clip_target = config.clip_target(state_dict=clip_sd)
    if clip_target is None:
        return None
    clip = comfy.sd.CLIP(
        clip_target,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        tokenizer_data=clip_sd,
        parameters=comfy.utils.calculate_parameters(clip_sd),
        state_dict=clip_sd,
    )
    _SAM3_CLIP_CACHE[cache_key] = clip
    return clip


def _sam3_text_embeddings(model: Any, text: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Encode a Selector semantic prompt with ComfyUI's official SAM3 CLIP."""
    attached = None
    getter = getattr(model, "get_attachment", None)
    if callable(getter):
        attached = getter("sam3_clip")
    clip = attached or _sam3_clip_from_model(model)
    if clip is None and _LAST_MODEL_SOURCE and _LAST_MODEL_SOURCE.get("kind") == "checkpoint":
        path = folder_paths.get_full_path_or_raise("checkpoints", str(_LAST_MODEL_SOURCE.get("name")))
        clip = _sam3_clip_from_path(path)
    if clip is None:
        candidates = [
            name for name in folder_paths.get_filename_list("checkpoints")
            if "sam3" in str(name).lower()
        ]
        if not candidates:
            raise ValueError("SAM3 Semantic needs a SAM3 checkpoint with its official CLIP text encoder.")
        clip = _sam3_clip_from_path(folder_paths.get_full_path_or_raise("checkpoints", candidates[0]))
    setter = getattr(model, "set_attachments", None)
    if callable(setter):
        setter("sam3_clip", clip)
    encoded = clip.encode_from_tokens(clip.tokenize(text), return_dict=True)
    cond = encoded.get("cond")
    if cond is None:
        raise ValueError("SAM3 CLIP did not return text conditioning.")
    attention = encoded.get("attention_mask")
    return cond, attention


def _sam3_anchor_masks(model: Any, image: torch.Tensor, prompt_data: str | None) -> torch.Tensor:
    """Run SAM3.1's official model kernel with one mixed prompt per object."""
    _, height, width, _ = image.shape
    prompts = _parse_prompt_data(prompt_data, width, height)
    comfy.model_management.load_model_gpu(model)
    device = comfy.model_management.get_torch_device()
    dtype = model.model.get_dtype()
    sam3_model = model.model.diffusion_model
    image_in = comfy.utils.common_upscale(
        image[..., :3].movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled"
    ).to(device=device, dtype=dtype)
    masks: list[torch.Tensor] = []
    for prompt in prompts:
        points = prompt["points"]
        point_inputs = None
        if points:
            point_inputs = {
                "point_coords": torch.tensor(
                    [[[point["x"] / width * 1008, point["y"] / height * 1008] for point in points]],
                    dtype=dtype,
                    device=device,
                ),
                "point_labels": torch.tensor(
                    [[int(point["label"]) for point in points]],
                    dtype=torch.int32,
                    device=device,
                ),
            }
        box_inputs = None
        if prompt["bbox"] is not None:
            box = prompt["bbox"]
            box_inputs = torch.tensor(
                [[
                    [box["x"] / width * 1008, box["y"] / height * 1008],
                    [(box["x"] + box["width"]) / width * 1008, (box["y"] + box["height"]) / height * 1008],
                ]],
                dtype=dtype,
                device=device,
            )
        mask_inputs = None
        if prompt["mask"] is not None:
            mask_inputs = _sam3_mask_logits(prompt["mask"], device, dtype)

        with torch.no_grad():
            if prompt["text"]:
                text_embeddings, text_mask = _sam3_text_embeddings(model, prompt["text"])
                # The official detector consumes normalized cxcywh boxes.  Keep
                # point prompts on the interactive decoder path below, then use
                # the highest-scoring text detection as this object's anchor.
                detector_boxes = None
                if prompt["bbox"] is not None:
                    box = prompt["bbox"]
                    detector_boxes = torch.tensor([[
                        [(box["x"] + box["width"] / 2) / width,
                         (box["y"] + box["height"] / 2) / height,
                         box["width"] / width,
                         box["height"] / height]
                    ]], dtype=dtype, device=device)
                detected = sam3_model(
                    image_in,
                    text_embeddings=text_embeddings.to(device=device, dtype=dtype),
                    text_mask=text_mask.to(device=device) if text_mask is not None else None,
                    boxes=detector_boxes,
                    threshold=0.0,
                    orig_size=(height, width),
                )
                scores = detected.get("scores")
                detected_masks = detected.get("masks")
                if detected_masks is None or detected_masks.numel() == 0:
                    raise ValueError(f"SAM3 returned no detection for semantic prompt: {prompt['text']}")
                score_row = scores[0] if scores is not None and scores.numel() else None
                best = int(score_row.argmax().item()) if score_row is not None else 0
                coarse = detected_masks[0, best:best + 1].unsqueeze(1)
                if point_inputs is not None or box_inputs is not None or mask_inputs is not None:
                    coarse_1008 = F.interpolate(coarse, size=(1008, 1008), mode="bilinear", align_corners=False)
                    if mask_inputs is not None:
                        coarse_1008 = (coarse_1008 + F.interpolate(mask_inputs, size=(1008, 1008), mode="bilinear", align_corners=False)) * 0.5
                    mask_logits = sam3_model.forward_segment(
                        image_in,
                        point_inputs=point_inputs,
                        box_inputs=box_inputs,
                        mask_inputs=coarse_1008,
                    )
                    mask_logits = sam3_model.forward_segment(
                        image_in,
                        point_inputs=point_inputs,
                        box_inputs=box_inputs,
                        mask_inputs=mask_logits,
                    )
                else:
                    mask_logits = coarse
            else:
                mask_logits = sam3_model.forward_segment(
                    image_in,
                    point_inputs=point_inputs,
                    box_inputs=box_inputs,
                    mask_inputs=mask_inputs,
                )
                # Match the official Detect node's default interactive refinement pass.
                mask_logits = sam3_model.forward_segment(
                    image_in,
                    point_inputs=point_inputs,
                    box_inputs=box_inputs,
                    mask_inputs=mask_logits,
                )
        resized = F.interpolate(mask_logits, size=(height, width), mode="bilinear", align_corners=False)
        masks.append((resized[0, 0] > 0).float().to("cpu"))
    return torch.stack(masks, dim=0)


def _preview_data_url(frame: torch.Tensor, mask: torch.Tensor) -> str:
    source = frame[..., :3].to("cpu", dtype=torch.float32).clamp(0.0, 1.0)
    alpha = mask.to("cpu", dtype=torch.float32).clamp(0.0, 1.0).unsqueeze(-1) * 0.52
    color = source.new_tensor([0.20, 0.77, 0.71])
    composite = source * (1.0 - alpha) + color * alpha
    array = (composite * 255.0).round().to(torch.uint8).numpy()
    image = Image.fromarray(array, mode="RGB")
    buffer = py_io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/png;base64," + encoded


async def _video_segment_preview_route(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
        frame_index = max(0, int(payload.get("frame", 0)))
        frame = _decode_selector_frame(payload, frame_index)

        # ``SAM3_Detect`` updates a ComfyUI ProgressBar even when it is
        # invoked outside the normal prompt queue.  Some portable builds do
        # not create ``last_prompt_id`` until the first queued prompt runs;
        # initialize the optional state so the global progress hook remains
        # usable for an interactive preview request.
        from server import PromptServer

        server_instance = getattr(PromptServer, "instance", None)
        if server_instance is not None and not hasattr(server_instance, "last_prompt_id"):
            server_instance.last_prompt_id = "cinestyle-preview"

        model = _preview_model(payload.get("model_source"))
        if model is None:
            raise ValueError(
                "Connect a CheckpointLoaderSimple or Load Diffusion Model node to MODEL, "
                "or run this SAM3 node once before using Preview."
            )
        masks = _sam3_anchor_masks(model, frame, payload.get("prompt_data"))
        mask = masks.amax(dim=0).to("cpu").float().clamp_(0.0, 1.0)
        return web.json_response(
            {
                "frame": frame_index,
                "image": _preview_data_url(frame[0], mask),
                "mask_area": float((mask > 0.5).float().mean().item()),
            }
        )
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)


def _sec_model_roots() -> list[str]:
    try:
        roots = folder_paths.get_folder_paths("sams")
    except KeyError:
        roots = []
    return [str(root) for root in (list(roots) or [os.path.join(folder_paths.models_dir, "sams")])]


def _sec_sync_model_folder() -> None:
    roots = [os.path.join(root, "SeC-4B") for root in _sec_model_roots()]
    extensions = set(getattr(folder_paths, "supported_pt_extensions", {".safetensors", ".bin", ".pth"}))
    folder_paths.folder_names_and_paths[_SEC_MODEL_FOLDER] = (roots, extensions)


def _sec_model_file_options() -> list[str]:
    """Return supported single-file names using ComfyUI's standard model listing."""
    _sec_sync_model_folder()
    try:
        files = folder_paths.get_filename_list(_SEC_MODEL_FOLDER)
    except (KeyError, OSError):
        files = []
    supported_names = set(_SEC_WEIGHT_SPECS)
    return [name for name in files if Path(name).name in supported_names]


def _sec_download_weights(target: Path, url: str) -> None:
    import urllib.request

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.download")
    print(f"[CineStyle] SeC-4B weights not found; downloading {target.name} to {target}.")
    print(f"[CineStyle] Download source: {url}")
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "CineStyle-ComfyUI/1.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as handle:
            expected = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            next_report = 256 * 1024 * 1024
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    if expected:
                        print(f"[CineStyle] SeC-4B download: {downloaded / expected:.0%}")
                    else:
                        print(f"[CineStyle] SeC-4B downloaded: {downloaded / (1024 ** 3):.2f} GiB")
                    next_report += 256 * 1024 * 1024
        if not partial.is_file() or partial.stat().st_size <= 0:
            raise RuntimeError("the downloaded file is empty")
        os.replace(str(partial), str(target))
    except Exception as exc:
        try:
            partial.unlink(missing_ok=True)
        except Exception:
            pass
        raise RuntimeError(
            f"Unable to download {target.name} automatically. "
            f"Download it manually from {url} and place it at {target}. "
            f"Original error: {exc}"
        ) from exc


def _sec_weight_dtype(weight_path: str) -> torch.dtype:
    filename = Path(weight_path).name.lower()
    if filename == "sec-4b-fp16.safetensors":
        return torch.float16
    if filename == "sec-4b-bf16.safetensors":
        return torch.bfloat16
    raise ValueError(
        "Unsupported SeC weight file. Choose SeC-4B-bf16.safetensors or SeC-4B-fp16.safetensors."
    )


def _sec_default_weight_filename() -> str:
    options = _sec_model_file_options()
    for filename in _SEC_WEIGHT_SPECS:
        if filename in options:
            return filename
    return _SEC_DEFAULT_WEIGHT_FILENAME


def _sec_weight_path(filename: str | None = None) -> str:
    filename = str(filename or _sec_default_weight_filename())
    basename = Path(filename).name
    spec = _SEC_WEIGHT_SPECS.get(basename)
    if spec is None:
        allowed = ", ".join(_SEC_WEIGHT_SPECS)
        raise ValueError(f"Unsupported SeC weight file {filename!r}. Supported files: {allowed}.")
    _, url = spec
    _sec_sync_model_folder()
    existing = folder_paths.get_full_path(_SEC_MODEL_FOLDER, filename)
    if existing:
        return existing
    roots = _sec_model_roots()
    target = Path(roots[0]) / "SeC-4B" / basename
    with _SEC_MODEL_DOWNLOAD_LOCK:
        if not target.is_file() or target.stat().st_size <= 0:
            _sec_download_weights(target, url)
    return str(target)


def _sec_model_config_path() -> str:
    required = ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
    missing = [name for name in required if not (_SEC_MODEL_CONFIG_PATH / name).is_file()]
    if missing:
        raise RuntimeError(
            "CineStyle SeC model configuration is incomplete: "
            + ", ".join(missing)
        )
    return str(_SEC_MODEL_CONFIG_PATH)


def _sec_imports() -> tuple[Any, Any, Any]:
    package_root = str(_SEC_PACKAGE_PATH.parent)
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    import warnings

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r"Importing from timm\.models\.layers is deprecated.*",
    )
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        logging.getLogger("transformers").setLevel(logging.ERROR)
    except Exception:
        pass
    from sec_inference.configuration_sec import SeCConfig
    from sec_inference.modeling_sec import SeCModel
    from transformers import AutoTokenizer

    return SeCConfig, SeCModel, AutoTokenizer


def _sec_install_dtype_hooks(model: Any) -> None:
    def dtype_conversion_hook(module, args, kwargs):
        try:
            module_dtype = next(module.parameters(recurse=False)).dtype
        except StopIteration:
            return args, kwargs
        except Exception:
            return args, kwargs
        if isinstance(module, torch.nn.Embedding):
            return args, kwargs

        def convert(value):
            if not isinstance(value, torch.Tensor) or value.dtype in {
                torch.long, torch.int, torch.int32, torch.int64,
            } or value.dtype == module_dtype:
                return value
            return value.to(dtype=module_dtype)

        return tuple(convert(value) for value in args), {
            key: convert(value) for key, value in kwargs.items()
        }

    if getattr(model, "_cinestyle_dtype_hooks", False):
        return
    for module in model.modules():
        if any(True for _ in module.parameters(recurse=False)):
            module.register_forward_pre_hook(dtype_conversion_hook, with_kwargs=True)
    model._cinestyle_dtype_hooks = True


def _sec_create_model(
    weight_path: str,
    torch_dtype: torch.dtype,
    device: str,
    use_flash_attn: bool,
    allow_mask_overlap: bool,
) -> Any:
    SeCConfig, SeCModel, AutoTokenizer = _sec_imports()
    config_path = _sec_model_config_path()
    config = SeCConfig.from_pretrained(config_path)
    config.hydra_overrides_extra = [
        f"++model.non_overlap_masks={'false' if allow_mask_overlap else 'true'}"
    ]

    try:
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
    except ImportError:
        init_empty_weights = None
        set_module_tensor_to_device = None

    from safetensors.torch import load_file

    if init_empty_weights is not None and set_module_tensor_to_device is not None:
        with init_empty_weights():
            model = SeCModel(config, use_flash_attn=use_flash_attn)
        state_dict = load_file(weight_path, device="cpu")
        try:
            for name, value in state_dict.items():
                set_module_tensor_to_device(model, name, device="cpu", value=value)
        finally:
            del state_dict
    else:
        model = SeCModel(config, use_flash_attn=use_flash_attn)
        state_dict = load_file(weight_path, device="cpu")
        try:
            model.load_state_dict(state_dict, strict=True)
        finally:
            del state_dict

    model = model.eval().to(device=device, dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(config_path, trust_remote_code=True)
    model.preparing_for_generation(tokenizer=tokenizer, torch_dtype=torch_dtype)
    if use_flash_attn and device.startswith("cuda:"):
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            print("[CineStyle] SeC flash attention is unavailable; using standard attention.")
    if device.startswith("cuda:") and torch_dtype != torch.float32:
        _sec_install_dtype_hooks(model)
    model._sec_loading_metadata = {
        "weight_path": weight_path,
        "torch_dtype": torch_dtype,
        "device": device,
        "use_flash_attn": use_flash_attn,
        "allow_mask_overlap": allow_mask_overlap,
    }
    model._sec_unloaded = False
    return model


def _sec_reload_model(model: Any) -> Any:
    metadata = getattr(model, "_sec_loading_metadata", None)
    if not metadata:
        raise RuntimeError("SeC model has been unloaded and has no reload metadata.")
    fresh = _sec_create_model(**metadata)
    model.__dict__.update(fresh.__dict__)
    model._sec_loading_metadata = metadata
    model._sec_unloaded = False
    del fresh
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def _sec_ensure_loaded(model: Any) -> Any:
    if getattr(model, "_sec_unloaded", False):
        return _sec_reload_model(model)
    return model


def _sec_unload_model(model: Any) -> None:
    if getattr(model, "_sec_unloaded", False):
        return
    for component in ("vision_model", "language_model", "grounding_encoder", "tokenizer"):
        value = getattr(model, component, None)
        if value is not None:
            try:
                if hasattr(value, "cpu"):
                    value.cpu()
                delattr(model, component)
            except Exception:
                pass
    model._sec_unloaded = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sec_executing_node_id() -> str | None:
    """Return the Comfy node currently executing, when supported by this build."""
    try:
        from comfy_execution.utils import get_executing_context

        context = get_executing_context()
        node_id = getattr(context, "node_id", None)
        return str(node_id) if node_id is not None else None
    except Exception:
        # Older ComfyUI builds do not expose an execution context. The global
        # latest-token fallback still keeps Preview functional there.
        return None


def _sec_register_model(model: Any, node_id: str | None = None) -> str:
    token = getattr(model, "_cinestyle_sec_token", None)
    if not token:
        token = f"sec-{uuid.uuid4().hex}"
        model._cinestyle_sec_token = token
    node_id = node_id or _sec_executing_node_id()
    with _SEC_MODEL_LOCK:
        _SEC_MODEL_REGISTRY[token] = model
        if node_id:
            _SEC_MODEL_NODE_TOKENS[node_id] = token
        global _SEC_PREVIEW_MODEL_TOKEN
        _SEC_PREVIEW_MODEL_TOKEN = token
    return token


def _sec_default_model_settings() -> tuple[torch.dtype, str, bool, bool]:
    """Return the same defaults used by the SeC Loader for cold Preview."""
    if torch.cuda.is_available():
        return torch.bfloat16, "cuda:0", True, True
    return torch.float32, "cpu", False, True


def _sec_cold_load_model() -> Any:
    """Load the default SeC model once when Preview has no Loader token."""
    with _SEC_MODEL_LOAD_LOCK:
        # Another request may have populated the registry while this request
        # was waiting for the load lock.
        with _SEC_MODEL_LOCK:
            latest = _SEC_PREVIEW_MODEL_TOKEN
            existing = _SEC_MODEL_REGISTRY.get(latest) if latest else None
        if existing is not None:
            return _sec_ensure_loaded(existing)

        _, device, use_flash_attn, allow_mask_overlap = _sec_default_model_settings()
        weight_path = _sec_weight_path()
        torch_dtype = _sec_weight_dtype(weight_path) if device != "cpu" else torch.float32
        print(
            f"[CineStyle] SeC Preview has no registered Loader model; "
            f"cold-loading default SeC-4B {Path(weight_path).stem.removeprefix('SeC-4B-').upper()} weights from {weight_path} on {device}."
        )
        model = _sec_create_model(
            weight_path,
            torch_dtype,
            device,
            use_flash_attn,
            allow_mask_overlap,
        )
        model._cinestyle_sec_cache_key = (
            weight_path,
            str(torch_dtype),
            device,
            bool(use_flash_attn),
            bool(allow_mask_overlap),
        )
        token = _sec_register_model(model)
        print(f"[CineStyle] SeC-4B cold Preview model ready; token={token}")
        return model


def _sec_model_for_token(token: str | None) -> Any:
    with _SEC_MODEL_LOCK:
        resolved = str(token or "") or _SEC_PREVIEW_MODEL_TOKEN
        model = _SEC_MODEL_REGISTRY.get(resolved) if resolved else None
    if model is None:
        if token and str(token).strip():
            raise ValueError("The requested SeC model token is no longer available.")
        return _sec_cold_load_model()
    return _sec_ensure_loaded(model)


def _sec_model_registry_response(loader_node_id: str | None = None) -> web.Response:
    with _SEC_MODEL_LOCK:
        models = [
            {
                "token": token,
                "loaded": not bool(getattr(model, "_sec_unloaded", False)),
                "device": str(getattr(model, "_sec_loading_metadata", {}).get("device", "")),
                "node_id": next((node for node, mapped in _SEC_MODEL_NODE_TOKENS.items() if mapped == token), None),
            }
            for token, model in _SEC_MODEL_REGISTRY.items()
        ]
        if loader_node_id:
            # Newer ComfyUI builds expose the executing node context, so a
            # connected Selector must use that Loader's token. On older builds
            # without context support, a single registered model is unambiguous.
            latest = _SEC_MODEL_NODE_TOKENS.get(str(loader_node_id))
            if latest is None and len(_SEC_MODEL_REGISTRY) == 1:
                latest = _SEC_PREVIEW_MODEL_TOKEN
        else:
            latest = _SEC_PREVIEW_MODEL_TOKEN
    return web.json_response({"models": models, "latest": latest, "loader_node_id": loader_node_id})


def _sec_frame_dir(images: torch.Tensor) -> str:
    temp_dir = tempfile.mkdtemp(prefix="cinestyle_sec_")
    for index, image in enumerate(images):
        array = (image[..., :3].to("cpu", dtype=torch.float32).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(array, mode="RGB").save(os.path.join(temp_dir, f"{index:05d}.jpg"), "JPEG", quality=95)
    return temp_dir


def _sec_mask_2d(mask: Any) -> np.ndarray:
    """Reduce a SeC mask/logit result to one object's HxW mask."""
    array = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    while array.ndim > 2:
        # SeC returns an object dimension (and some model versions add a
        # singleton mask dimension). This node tracks one object, so select it.
        array = array[0]
    if array.ndim != 2:
        raise ValueError("SeC returned a mask with an unsupported shape.")
    return array


def _sec_mask_for_object(mask: Any, object_ids: Any, object_id: int) -> np.ndarray:
    """Select one object's mask from SeC's consolidated multi-object output."""
    array = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    ids = list(object_ids or [])
    try:
        object_index = ids.index(object_id)
    except ValueError:
        object_index = 0
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim == 3:
        array = array[object_index if array.shape[0] > object_index else 0]
    while array.ndim > 2:
        array = array[0]
    if array.ndim != 2:
        raise ValueError("SeC returned a mask with an unsupported shape.")
    return array


def _sec_add_prompt(model: Any, state: dict[str, Any], frame_index: int, object_id: int, points, labels, box, init_mask):
    if init_mask is not None:
        # The bundled SeC/SAM2 configs enable mask-as-output for video
        # conditioning. Temporarily disable that shortcut on the anchor so a
        # rough brush mask is actually passed through the SAM decoder.
        encoder = model.grounding_encoder
        previous_mask_mode = getattr(encoder, "use_mask_input_as_output_without_sam", None)
        if previous_mask_mode is not None:
            encoder.use_mask_input_as_output_without_sam = False
        try:
            _, object_ids, mask_logits = encoder.add_new_mask(
                inference_state=state,
                frame_idx=frame_index,
                obj_id=object_id,
                mask=_sec_mask_2d(init_mask),
            )
        finally:
            if previous_mask_mode is not None:
                encoder.use_mask_input_as_output_without_sam = previous_mask_mode
        init_mask = _sec_mask_for_object(mask_logits > 0.0, object_ids, object_id)
    if points is not None or box is not None:
        _, object_ids, logits = model.grounding_encoder.add_new_points_or_box(
            inference_state=state,
            frame_idx=frame_index,
            obj_id=object_id,
            points=points,
            labels=labels,
            box=box,
        )
        init_mask = _sec_mask_for_object(logits > 0.0, object_ids, object_id)
    return init_mask


def _sec_anchor_preview(model: Any, frame: torch.Tensor, prompt_data: str | None) -> torch.Tensor:
    height, width = map(int, frame.shape[1:3])
    prompts = _parse_prompt_data(prompt_data, width, height)
    if any(prompt["text"] for prompt in prompts):
        raise ValueError("SeC-4B does not support Semantic prompts. Use Point, BBox, or Draw Mask.")
    temp_dir = _sec_frame_dir(frame)
    try:
        states = getattr(model.grounding_encoder, "_states", None)
        if hasattr(states, "clear"):
            states.clear()
        state = model.grounding_encoder.init_state(video_path=temp_dir, offload_video_to_cpu=False, offload_state_to_cpu=False)
        model.grounding_encoder.reset_state(state)
        masks = []
        for object_index, prompt in enumerate(prompts, start=1):
            point_values = prompt["points"]
            point_array = np.asarray([[item["x"], item["y"]] for item in point_values], dtype=np.float32) if point_values else None
            labels = np.asarray([int(item["label"]) for item in point_values], dtype=np.int32) if point_values else None
            box = prompt["bbox"]
            box_array = None if box is None else np.asarray(
                [box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]],
                dtype=np.float32,
            )
            init_mask = _sec_add_prompt(
                model,
                state,
                0,
                object_index,
                point_array,
                labels,
                box_array,
                prompt["mask"].numpy() if isinstance(prompt["mask"], torch.Tensor) else prompt["mask"],
            )
            if init_mask is None:
                raise ValueError(f"SeC did not produce a mask for object {object_index}.")
            masks.append(torch.from_numpy((_sec_mask_2d(init_mask) > 0).astype(np.float32)))
        return torch.stack(masks, dim=0).amax(dim=0)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


async def _sec_video_segment_models_route(request: web.Request) -> web.Response:
    loader_node_id = request.query.get("loader_node_id")
    with _SEC_MODEL_LOCK:
        if loader_node_id:
            has_latest = bool(
                _SEC_MODEL_NODE_TOKENS.get(str(loader_node_id))
                or (len(_SEC_MODEL_REGISTRY) == 1 and _SEC_PREVIEW_MODEL_TOKEN)
            )
        else:
            has_latest = bool(_SEC_PREVIEW_MODEL_TOKEN)
    if not has_latest:
        print(
            "[CineStyle] SeC Preview requested without a registered model; "
            "default cold loading will be used."
        )
    return _sec_model_registry_response(loader_node_id)


async def _sec_video_segment_preview_route(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
        frame_index = max(0, int(payload.get("frame", 0)))
        frame = _decode_selector_frame(payload, frame_index)
        model = _sec_model_for_token(payload.get("model_token"))
        mask = _sec_anchor_preview(model, frame, payload.get("prompt_data"))
        return web.json_response({
            "frame": frame_index,
            "image": _preview_data_url(frame[0], mask),
            "mask_area": float((mask > 0.5).float().mean().item()),
        })
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)


SEC_MODEL = io.Custom("SEC_MODEL")


class CSSeCModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        devices = ["auto", "cpu"]
        if torch.cuda.is_available():
            devices.extend(f"gpu{index}" for index in range(torch.cuda.device_count()))
        return io.Schema(
            node_id="CS_SeC_ModelLoader",
            display_name="CS SeC-4B Model Loader",
            category="😺dzNodes/CineStyle/Video",
            search_aliases=["sec", "segment concept", "seC-4B", "video segmentation"],
            inputs=[
                io.Combo.Input(
                    "model_file",
                    options=_sec_model_file_options() or list(_SEC_WEIGHT_SPECS),
                    default=_sec_default_weight_filename(),
                    tooltip="SeC single-file weights found in ComfyUI/models/sams/SeC-4B.",
                ),
                io.Combo.Input("device", options=devices, default="auto"),
                io.Boolean.Input("use_flash_attn", default=True, advanced=True),
                io.Boolean.Input("allow_mask_overlap", default=True, advanced=True),
            ],
            outputs=[SEC_MODEL.Output("model", display_name="SEC_MODEL")],
        )

    @classmethod
    def execute(cls, model_file=_SEC_DEFAULT_WEIGHT_FILENAME, device="auto", use_flash_attn=True, allow_mask_overlap=True) -> io.NodeOutput:
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif device.startswith("gpu"):
            device = f"cuda:{int(device[3:])}"
        weight_path = _sec_weight_path(model_file)
        dtype = _sec_weight_dtype(weight_path) if device != "cpu" else torch.float32
        if device == "cpu":
            use_flash_attn = False
        key = (weight_path, str(dtype), device, bool(use_flash_attn), bool(allow_mask_overlap))
        with _SEC_MODEL_LOCK:
            model = next((candidate for candidate in _SEC_MODEL_REGISTRY.values() if getattr(candidate, "_cinestyle_sec_cache_key", None) == key), None)
        if model is None:
            precision = Path(weight_path).stem.removeprefix("SeC-4B-").upper()
            print(f"[CineStyle] Loading SeC-4B {precision} weights from {weight_path} on {device}.")
            model = _sec_create_model(weight_path, dtype, device, bool(use_flash_attn), bool(allow_mask_overlap))
            model._cinestyle_sec_cache_key = key
        else:
            model = _sec_ensure_loaded(model)
        token = _sec_register_model(model)
        print(f"[CineStyle] SeC-4B ready; preview token={token}")
        return io.NodeOutput(model)


class CSVideoSegmentSeC(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CS_Video_Segment_SeC",
            display_name="CS Video Segment (SeC-4B)",
            category="😺dzNodes/CineStyle/Video",
            search_aliases=["sec video", "segment concept", "longsam2", "video mask"],
            inputs=[
                SEC_MODEL.Input("model", tooltip="Loaded SeC-4B model."),
                io.Image.Input("images", optional=True, tooltip="Video frames as an IMAGE batch."),
                io.Video.Input("video_input", optional=True, tooltip="Optional VIDEO input."),
                io.Video.Input("proxy_video", optional=True, tooltip="Optional VIDEO used only by Selector preview."),
                io.Int.Input("anchor_frame", default=0, min=0, max=10000000, step=1),
                io.String.Input("prompt_data", default='{"version":2,"objects":[]}', multiline=True, optional=True, tooltip="Selector multi-object mask, bbox, and point prompts."),
                io.Combo.Input("tracking_direction", options=["forward", "backward", "bidirectional"], default="bidirectional", advanced=True),
                io.Int.Input("max_frames_to_track", default=-1, min=-1, max=10000000, step=1, advanced=True),
                io.Int.Input("mllm_memory_size", default=12, min=1, max=20, step=1, advanced=True),
                io.Boolean.Input("offload_video_to_cpu", default=False, advanced=True),
                io.Boolean.Input("auto_unload_model", default=True, advanced=True),
            ],
            outputs=[
                io.Mask.Output("mask", display_name="MASK"),
                io.Mask.Output("anchor_mask", display_name="anchor_mask"),
                io.Dict.Output("video_info", display_name="video_info"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: Any,
        images: torch.Tensor | None = None,
        video_input: Any = None,
        proxy_video: Any = None,
        anchor_frame: int = 0,
        prompt_data: str = '{"version":2,"objects":[]}',
        tracking_direction: str = "bidirectional",
        max_frames_to_track: int = -1,
        mllm_memory_size: int = 12,
        offload_video_to_cpu: bool = False,
        auto_unload_model: bool = True,
    ) -> io.NodeOutput:
        node_name = "CS Video Segment (SeC-4B)"
        _segment_info(node_name, "start")
        model = _sec_ensure_loaded(model)
        _segment_info(node_name, "model ready")
        if images is None and video_input is not None:
            images = video_input.get_components().images
        if images is None:
            raise ValueError("Connect CS Load Video to images or video_input.")
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] < 3:
            raise ValueError("images must have shape [frames, height, width, 3 or 4].")
        images = images[..., :3].to("cpu", dtype=torch.float32).clamp_(0, 1)
        _segment_info(node_name, f"input ready: frames={images.shape[0]}, size={images.shape[2]}x{images.shape[1]}")
        if not _prompt_has_file_video_source(cls.hidden.prompt, cls.hidden.unique_id):
            _cache_selector_input(
                cls.hidden.unique_id,
                images,
                _video_input_fps(video_input, cls.hidden.prompt, cls.hidden.unique_id),
            )
        if proxy_video is not None:
            try:
                proxy_images = proxy_video.get_components().images
                if isinstance(proxy_images, torch.Tensor) and proxy_images.ndim == 4 and proxy_images.shape[0] > 0:
                    _cache_selector_input(
                        f"{cls.hidden.unique_id}:proxy",
                        proxy_images,
                        _video_input_fps(proxy_video, cls.hidden.prompt, cls.hidden.unique_id),
                    )
                    _segment_info(node_name, f"proxy preview cached: frames={proxy_images.shape[0]}, size={proxy_images.shape[2]}x{proxy_images.shape[1]}")
                else:
                    _segment_info(node_name, "proxy_video ignored: no valid IMAGE frames")
            except Exception as exc:
                _segment_info(node_name, f"proxy preview cache unavailable: {exc}")
        frame_count, height, width = map(int, images.shape[:3])
        anchor = int(anchor_frame)
        if not 0 <= anchor < frame_count:
            raise ValueError(f"anchor_frame must be between 0 and {frame_count - 1}.")
        if tracking_direction not in {"forward", "backward", "bidirectional"}:
            raise ValueError("tracking_direction must be forward, backward, or bidirectional.")
        prompts = _parse_prompt_data(prompt_data, width, height)
        _segment_info(node_name, f"prompts parsed: objects={len(prompts)}, anchor={anchor}")
        if any(prompt["text"] for prompt in prompts):
            raise ValueError("SeC-4B does not support Semantic prompts. Use Point, BBox, or Draw Mask.")

        temp_dir = _sec_frame_dir(images)
        _segment_info(node_name, "temporary frame sequence prepared")
        state = None
        nested_tqdm = _NestedTqdmSilencer(
            (
                "sec_inference.modeling_sec",
                "sec_inference.sam2_video_predictor",
                "sec_inference.sam2.sam2_video_predictor",
                "sec_inference.sam2.utils.misc",
            )
        )
        nested_tqdm.start()
        try:
            state = model.grounding_encoder.init_state(
                video_path=temp_dir,
                offload_video_to_cpu=bool(offload_video_to_cpu),
                offload_state_to_cpu=str(model._sec_loading_metadata.get("device")) == "cpu",
            )
            model.grounding_encoder.reset_state(state)
            _segment_info(node_name, "video tracking state initialized")
            object_masks: list[np.ndarray] = []

            def add_all_prompts() -> list[np.ndarray]:
                added: list[np.ndarray] = []
                for object_index, prompt in enumerate(prompts, start=1):
                    point_values = prompt["points"]
                    point_array = np.asarray([[item["x"], item["y"]] for item in point_values], dtype=np.float32) if point_values else None
                    labels = np.asarray([int(item["label"]) for item in point_values], dtype=np.int32) if point_values else None
                    box = prompt["bbox"]
                    box_array = None if box is None else np.asarray(
                        [box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]],
                        dtype=np.float32,
                    )
                    init_mask = _sec_add_prompt(
                        model,
                        state,
                        anchor,
                        object_index,
                        point_array,
                        labels,
                        box_array,
                        prompt["mask"].numpy() if isinstance(prompt["mask"], torch.Tensor) else prompt["mask"],
                    )
                    if init_mask is None:
                        raise RuntimeError(f"SeC did not produce an initial mask for object {object_index}.")
                    added.append(_sec_mask_2d(init_mask))
                return added

            object_masks = add_all_prompts()
            _segment_info(node_name, f"anchor prompts applied: objects={len(object_masks)}")
            initial_union = torch.from_numpy(np.asarray(object_masks).astype(np.float32).max(axis=0))
            limit = frame_count if int(max_frames_to_track) < 0 else max(1, int(max_frames_to_track))
            segments: dict[int, torch.Tensor] = {}
            progress_total = _segment_expected_frames(frame_count, anchor, tracking_direction, limit)
            if tracking_direction == "bidirectional":
                _segment_info(node_name, "propagating forward and backward")
            else:
                _segment_info(node_name, f"propagating {tracking_direction}")
            progress = _SegmentProgress(node_name, progress_total)

            def collect(reverse: bool):
                for frame_index, object_ids, mask_logits in model.propagate_in_video(
                    state,
                    start_frame_idx=anchor,
                    max_frame_num_to_track=limit,
                    reverse=reverse,
                    init_mask=initial_union.numpy(),
                    tokenizer=None,
                    mllm_memory_size=max(1, int(mllm_memory_size)),
                ):
                    union = (mask_logits > 0.0).any(dim=0).to("cpu", dtype=torch.float32)
                    segments[int(frame_index)] = union
                    progress.update()

            if tracking_direction in {"forward", "bidirectional"}:
                collect(False)
            if tracking_direction == "bidirectional":
                model.grounding_encoder.reset_state(state)
                object_masks = add_all_prompts()
                collect(True)
            elif tracking_direction == "backward":
                collect(True)
            progress.close()

            output = torch.zeros(frame_count, height, width, dtype=torch.float32)
            for frame_index, mask in segments.items():
                if 0 <= frame_index < frame_count:
                    output[frame_index] = mask
            output[anchor] = initial_union
            info = {
                "frame_count": frame_count,
                "height": height,
                "width": width,
                "anchor_frame": anchor,
                "tracking_direction": tracking_direction,
                "object_count": len(prompts),
            }
            _segment_info(node_name, f"complete: frames={frame_count}, object_count={len(prompts)}")
            return io.NodeOutput(output, initial_union, info)
        finally:
            nested_tqdm.stop()
            try:
                if state is not None:
                    model.grounding_encoder.reset_state(state)
            except Exception:
                pass
            shutil.rmtree(temp_dir, ignore_errors=True)
            if auto_unload_model:
                _sec_unload_model(model)
                _segment_info(node_name, "model unloaded")
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()


def _unpack_track(result: dict[str, Any], height: int, width: int) -> torch.Tensor | None:
    packed = result.get("packed_masks")
    if packed is None:
        return None
    from comfy.ldm.sam3.tracker import unpack_masks

    unpacked = unpack_masks(packed).float()  # [T, N_obj, Hm, Wm]
    if unpacked.ndim != 4:
        return None
    frames, objects = unpacked.shape[:2]
    resized = F.interpolate(
        unpacked.reshape(frames * objects, 1, *unpacked.shape[-2:]),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(frames, objects, height, width).amax(dim=1)


def _propagate(
    model: Any,
    images: torch.Tensor,
    anchor_masks: torch.Tensor,
    anchor_frame: int,
    direction: str,
    pbar: Any,
    max_objects: int,
) -> torch.Tensor:
    """Propagate from the anchor in one or both temporal directions."""
    frame_count, height, width = images.shape[:3]
    output = torch.zeros(frame_count, height, width, dtype=torch.float32)
    output[anchor_frame] = anchor_masks.amax(dim=0).to("cpu").float()

    comfy.model_management.load_model_gpu(model)
    device = comfy.model_management.get_torch_device()
    dtype = model.model.get_dtype()
    sam3_model = model.model.diffusion_model
    frames_chw = images[..., :3].movedim(-1, 1)

    def run_sequence(sequence: torch.Tensor) -> torch.Tensor | None:
        with torch.no_grad():
            result = sam3_model.forward_video(
                images=sequence,
                initial_masks=anchor_masks,
                pbar=pbar,
                text_prompts=None,
                max_objects=max_objects,
                target_device=device,
                target_dtype=dtype,
            )
        return _unpack_track(result, height, width)

    if direction in {"both", "forward"} and anchor_frame + 1 < frame_count:
        forward = run_sequence(frames_chw[anchor_frame:])
        if forward is not None:
            output[anchor_frame:] = forward

    if direction in {"both", "backward"} and anchor_frame > 0:
        backward = run_sequence(frames_chw[: anchor_frame + 1].flip(0))
        if backward is not None:
            chronological = backward.flip(0)
            output[: anchor_frame + 1] = chronological

    output[anchor_frame] = anchor_masks.amax(dim=0).to("cpu").float()
    return output.clamp_(0.0, 1.0)


class CSVideoSegmentSAM3(io.ComfyNode):
    """Select an object on any video frame and propagate its SAM3.1 mask."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=NODE_ID,
            display_name="CS Video Segment (SAM3.1)",
            category="😺dzNodes/CineStyle/Video",
            essentials_category="Video Tools",
            search_aliases=["video segment", "sam3.1", "sam3 video", "propagate mask"],
            description=(
                "Define semantic, mask, bounding-box, and positive/negative point prompts "
                "on any video frame, then propagate them in both directions."
            ),
            inputs=[
                io.Model.Input("model", tooltip="Official ComfyUI SAM3/SAM3.1 model."),
                io.Image.Input(
                    "images",
                    optional=True,
                    tooltip="Video frames as an IMAGE batch. Connect CS Load Video for Selector input.",
                ),
                io.Video.Input(
                    "video_input",
                    optional=True,
                    tooltip="Optional VIDEO input from CS Load Video.",
                ),
                io.Video.Input(
                    "proxy_video",
                    optional=True,
                    tooltip="Optional VIDEO used only by Selector preview.",
                ),
                io.Int.Input(
                    "anchor_frame",
                    default=0,
                    min=0,
                    max=10000000,
                    step=1,
                    tooltip="Frame where the selector prompt is defined.",
                ),
                io.String.Input(
                    "prompt_data",
                    default='{"version":2,"objects":[]}',
                    multiline=True,
                    optional=True,
                    tooltip="Selector multi-object semantic, mask, bbox, and point prompts.",
                ),
                io.Combo.Input(
                    "propagation_direction",
                    options=_PROPAGATION_OPTIONS,
                    default="both",
                    advanced=True,
                    tooltip="Propagate toward both sides of the anchor or only one side.",
                ),
                io.Int.Input(
                    "max_objects",
                    default=16,
                    min=1,
                    max=64,
                    step=1,
                    advanced=True,
                    tooltip="Maximum SAM3.1 multiplex object slots.",
                ),
            ],
            outputs=[
                io.Mask.Output("mask", display_name="MASK"),
                io.Mask.Output("anchor_mask", display_name="anchor_mask"),
                io.Dict.Output("video_info", display_name="video_info"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: Any,
        images: torch.Tensor | None = None,
        video_input: Any = None,
        proxy_video: Any = None,
        anchor_frame: int = 0,
        prompt_data: str = '{"version":2,"objects":[]}',
        propagation_direction: str = "both",
        max_objects: int = 16,
    ) -> io.NodeOutput:
        global _LAST_MODEL
        node_name = "CS Video Segment (SAM3.1)"
        _segment_info(node_name, "start")
        _LAST_MODEL = model
        _segment_info(node_name, "model ready")
        if images is None and video_input is not None:
            images = video_input.get_components().images
        if images is None:
            raise ValueError("Connect CS Load Video to images or video_input.")
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] < 3:
            raise ValueError("images must have shape [frames, height, width, 3 or 4].")
        if images.shape[0] == 0:
            raise ValueError("The video contains no frames.")

        images = images[..., :3].to(device="cpu", dtype=torch.float32).clamp_(0.0, 1.0)
        _segment_info(node_name, f"input ready: frames={images.shape[0]}, size={images.shape[2]}x{images.shape[1]}")
        if not _prompt_has_file_video_source(cls.hidden.prompt, cls.hidden.unique_id):
            _cache_selector_input(
                cls.hidden.unique_id,
                images,
                _video_input_fps(video_input, cls.hidden.prompt, cls.hidden.unique_id),
            )
        if proxy_video is not None:
            try:
                proxy_images = proxy_video.get_components().images
                if isinstance(proxy_images, torch.Tensor) and proxy_images.ndim == 4 and proxy_images.shape[0] > 0:
                    _cache_selector_input(
                        f"{cls.hidden.unique_id}:proxy",
                        proxy_images,
                        _video_input_fps(proxy_video, cls.hidden.prompt, cls.hidden.unique_id),
                    )
                    _segment_info(node_name, f"proxy preview cached: frames={proxy_images.shape[0]}, size={proxy_images.shape[2]}x{proxy_images.shape[1]}")
                else:
                    _segment_info(node_name, "proxy_video ignored: no valid IMAGE frames")
            except Exception as exc:
                _segment_info(node_name, f"proxy preview cache unavailable: {exc}")
        frame_count, height, width = map(int, images.shape[:3])
        anchor = int(anchor_frame)
        if anchor < 0 or anchor >= frame_count:
            raise ValueError(f"anchor_frame must be between 0 and {frame_count - 1}.")
        if propagation_direction not in _PROPAGATION_OPTIONS:
            raise ValueError(f"propagation_direction must be one of {_PROPAGATION_OPTIONS}.")

        # SAM3.1's multiplex tracker has sixteen object slots. Keep the
        # public control compatible with broader SAM3 workflows, but never
        # pass more than the architectural cap to the tracker.
        object_limit = min(16, max(1, int(max_objects)))

        anchor_mask_objects = _sam3_anchor_masks(model, images[anchor : anchor + 1], prompt_data)
        _segment_info(node_name, f"anchor prompts segmented: objects={anchor_mask_objects.shape[0]}")
        if anchor_mask_objects.shape[0] > object_limit:
            anchor_mask_objects = anchor_mask_objects[:object_limit]
        anchor_mask = anchor_mask_objects.amax(dim=0).to("cpu").float().clamp_(0.0, 1.0)

        progress_total = _segment_expected_frames(frame_count, anchor, propagation_direction)
        backend_pbar = comfy.utils.ProgressBar(progress_total)
        _segment_info(node_name, f"propagating masks: direction={propagation_direction}")
        pbar = _SegmentProgress(node_name, progress_total, backend_pbar)
        nested_tqdm = _NestedTqdmSilencer(("comfy.ldm.sam3.tracker",))
        nested_tqdm.start()
        try:
            mask = _propagate(
                model,
                images,
                anchor_mask_objects,
                anchor,
                propagation_direction,
                pbar,
                object_limit,
            )
        finally:
            nested_tqdm.stop()
            pbar.close()
        info = {
            "frame_count": frame_count,
            "height": height,
            "width": width,
            "anchor_frame": anchor,
            "propagation_direction": propagation_direction,
            "prompt_version": PROMPT_VERSION,
            "object_count": int(anchor_mask_objects.shape[0]),
        }
        _segment_info(node_name, f"complete: frames={frame_count}, object_count={int(anchor_mask_objects.shape[0])}")
        return io.NodeOutput(mask, anchor_mask, info)


class VideoSegmentExtension(ComfyExtension):
    @override
    async def on_load(self) -> None:
        global _PREVIEW_ROUTE_REGISTERED
        if _PREVIEW_ROUTE_REGISTERED:
            return
        from server import PromptServer

        server_instance = getattr(PromptServer, "instance", None)
        if server_instance is not None:
            server_instance.routes.post("/cinestyle/video-segment-preview")(
                _video_segment_preview_route
            )
            server_instance.routes.get("/cinestyle/sec-models")(
                _sec_video_segment_models_route
            )
            server_instance.routes.post("/cinestyle/sec-video-segment-preview")(
                _sec_video_segment_preview_route
            )
            server_instance.routes.get("/cinestyle/video-selector-cache")(
                _selector_cache_info_route
            )
            server_instance.routes.get("/cinestyle/video-selector-cache-video")(
                _selector_cache_video_route
            )
            _PREVIEW_ROUTE_REGISTERED = True

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CSVideoSegmentSAM3, CSSeCModelLoader, CSVideoSegmentSeC]


async def comfy_entrypoint() -> VideoSegmentExtension:
    return VideoSegmentExtension()
