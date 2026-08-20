"""Interactive SAM 3.1 video segmentation for CineStyle.

The node deliberately delegates the anchor-frame segmentation to ComfyUI's
official ``SAM3_Detect`` node.  Video propagation uses the same official SAM3
model's ``forward_video`` implementation; the only extra work here is running
it from an arbitrary anchor frame in both temporal directions.
"""

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
import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, io


NODE_ID = "CS_Video_Segment_SAM3"
_VIDEO_MODE_OPTIONS = ["semantic", "points", "bbox"]
_PROPAGATION_OPTIONS = ["both", "forward", "backward"]
_PREVIEW_ROUTE_REGISTERED = False
_LAST_MODEL: Any = None
_SEMANTIC_CLIP_CACHE: dict[str, Any] = {}
# SAM3.1 preview models are keyed by source path. ComfyUI's ModelPatcher
# remains responsible for GPU residency; this cache only avoids loading a
# second Python model object when Preview is clicked repeatedly.
_PREVIEW_MODEL_CACHE: dict[str, Any] = {}
_SEC_MODEL_REGISTRY: dict[str, Any] = {}
_SEC_MODEL_NODE_TOKENS: dict[str, str] = {}
_SEC_MODEL_LOCK = threading.RLock()
_SEC_MODEL_LOAD_LOCK = threading.Lock()
_SEC_PREVIEW_MODEL_TOKEN: str | None = None
_SEC_PACKAGE_PATH = Path(__file__).resolve().parent / "sec_inference"
_SEC_CONFIG_PATH = Path(__file__).resolve().parent / "sec_configs"
_SELECTOR_CACHE_ROOT = Path(tempfile.gettempdir()) / "cinestyle_selector_cache"
_SELECTOR_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
_SELECTOR_INPUT_CACHE: dict[str, dict[str, Any]] = {}
_SELECTOR_CACHE_LOCK = threading.RLock()
_SELECTOR_CACHE_LIMIT = 8
_SELECTOR_CACHE_MAX_BYTES = 4 * 1024**3

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


def _encode_selector_cache(path: Path, frames: torch.Tensor, fps: float) -> None:
    height, width = map(int, frames.shape[1:3])
    encoded_width = width + (width % 2)
    encoded_height = height + (height % 2)
    rate = Fraction(float(fps)).limit_denominator(1000)
    with av.open(str(path), mode="w", format="mp4") as container:
        try:
            stream = container.add_stream("libx264", rate=rate)
            stream.options = {"preset": "ultrafast", "crf": "20"}
        except (av.error.FFmpegError, ValueError):
            stream = container.add_stream("mpeg4", rate=rate)
        stream.width = encoded_width
        stream.height = encoded_height
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            array = frame.numpy()
            if encoded_width != width or encoded_height != height:
                array = np.pad(
                    array,
                    ((0, encoded_height - height), (0, encoded_width - width), (0, 0)),
                    mode="edge",
                )
            video_frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _remove_selector_cache_entry(entry: dict[str, Any] | None) -> None:
    if not entry:
        return
    for name in ("path", "frames_path"):
        try:
            Path(str(entry.get(name) or "")).unlink(missing_ok=True)
        except OSError:
            pass


def _cache_selector_input(node_id: Any, images: torch.Tensor, fps: float) -> str | None:
    key = str(node_id or "").strip()
    if not key or not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] == 0:
        return None
    frames = (
        images[..., :3]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .contiguous()
    )
    safe_fps = float(fps) if math.isfinite(float(fps)) and float(fps) > 0 else 24.0
    token = uuid.uuid4().hex
    path = _SELECTOR_CACHE_ROOT / f"{token}.mp4"
    frames_path = _SELECTOR_CACHE_ROOT / f"{token}.npy"
    try:
        np.save(frames_path, frames.numpy(), allow_pickle=False)
        _encode_selector_cache(path, frames, safe_fps)
    except Exception as exc:
        path.unlink(missing_ok=True)
        frames_path.unlink(missing_ok=True)
        print(f"[CineStyle] Selector input cache failed for node {key}: {exc}")
        return None

    entry = {
        "token": token,
        "node_id": key,
        "path": str(path),
        "frames_path": str(frames_path),
        "created": time.time(),
        "size_bytes": int(path.stat().st_size + frames_path.stat().st_size),
        "info": {
            "frames": int(frames.shape[0]),
            "width": int(frames.shape[2]),
            "height": int(frames.shape[1]),
            "fps": safe_fps,
            "duration": float(frames.shape[0]) / safe_fps,
            "audio_format": None,
        },
    }
    evicted: list[dict[str, Any]] = []
    with _SELECTOR_CACHE_LOCK:
        previous = _SELECTOR_INPUT_CACHE.pop(key, None)
        if previous:
            evicted.append(previous)
        _SELECTOR_INPUT_CACHE[key] = entry
        while (
            len(_SELECTOR_INPUT_CACHE) > _SELECTOR_CACHE_LIMIT
            or (
                len(_SELECTOR_INPUT_CACHE) > 1
                and sum(int(item.get("size_bytes") or 0) for item in _SELECTOR_INPUT_CACHE.values())
                > _SELECTOR_CACHE_MAX_BYTES
            )
        ):
            _, oldest = next(iter(_SELECTOR_INPUT_CACHE.items()))
            _SELECTOR_INPUT_CACHE.pop(str(oldest["node_id"]), None)
            evicted.append(oldest)
    for stale in evicted:
        _remove_selector_cache_entry(stale)
    print(
        f"[CineStyle] Cached selector input for node {key}: "
        f"{frames.shape[0]} frames at {safe_fps:.3f} fps."
    )
    return token


def _selector_cache_for_node(node_id: Any) -> dict[str, Any] | None:
    with _SELECTOR_CACHE_LOCK:
        return _SELECTOR_INPUT_CACHE.get(str(node_id or "").strip())


def _selector_cache_for_token(token: Any) -> dict[str, Any] | None:
    resolved = str(token or "").strip()
    if not resolved:
        return None
    with _SELECTOR_CACHE_LOCK:
        return next(
            (entry for entry in _SELECTOR_INPUT_CACHE.values() if entry.get("token") == resolved),
            None,
        )


def _decode_selector_frame(payload: dict[str, Any], frame_index: int) -> torch.Tensor:
    token = str(payload.get("source_token") or "").strip()
    if not token:
        return _decode_video_frame(str(payload.get("video") or ""), frame_index)
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
    entry = _selector_cache_for_node(request.query.get("node_id", ""))
    if entry is None:
        return web.json_response({"error": "No cached Selector input."}, status=404)
    token = str(entry["token"])
    return web.json_response(
        {
            "token": token,
            "label": "Cached input from the last workflow run",
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


def _model_checkpoint_path(model: Any) -> str | None:
    """Get the source checkpoint from ComfyUI's model reload metadata."""
    cached = getattr(model, "cached_patcher_init", None)
    if not cached or len(cached) < 2:
        return None
    factory, args = cached[:2]
    name = getattr(factory, "__name__", "")
    if "load_checkpoint" not in name or not args:
        return None
    path = args[0]
    return str(path) if path else None


def _semantic_conditioning(model: Any, prompt: str) -> Any:
    """Encode a semantic prompt using the CLIP bundled in the SAM3 checkpoint."""
    prompt = str(prompt or "").strip()
    if not prompt:
        raise ValueError("Semantic mode requires a semantic_prompt.")
    checkpoint = _model_checkpoint_path(model)
    if not checkpoint:
        raise ValueError(
            "Semantic mode needs a SAM3 checkpoint loaded by CheckpointLoaderSimple; "
            "the connected MODEL does not expose its text encoder source."
        )
    clip = _SEMANTIC_CLIP_CACHE.get(checkpoint)
    if clip is None:
        _, clip, _, _ = comfy.sd.load_checkpoint_guess_config(
            checkpoint,
            output_vae=False,
            output_clip=True,
            output_clipvision=False,
            output_model=False,
        )
        if clip is None:
            raise ValueError("The SAM3 checkpoint does not contain a text encoder.")
        _SEMANTIC_CLIP_CACHE[checkpoint] = clip
    return clip.encode_from_tokens_scheduled(clip.tokenize(prompt), show_pbar=False)


def _preview_model(source: Any) -> Any:
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
                _SEMANTIC_CLIP_CACHE[path] = clip
        return model
    if kind == "diffusion_model":
        path = folder_paths.get_full_path_or_raise("diffusion_models", name)
        model = _PREVIEW_MODEL_CACHE.get(path)
        if model is None:
            model = comfy.sd.load_diffusion_model(path)
            _PREVIEW_MODEL_CACHE[path] = model
        return model
    return _LAST_MODEL


def _anchor_masks(
    model: Any,
    image: torch.Tensor,
    mode: str,
    semantic_prompt: str,
    points: str | None,
    bbox: str | None,
    threshold: float,
) -> torch.Tensor:
    """Run the official SAM3 Detect node on one frame and return [N, H, W]."""
    from comfy_extras.nodes_sam3 import SAM3_Detect

    _, height, width, _ = image.shape
    positive_coords: list[dict[str, float]] = []
    negative_coords: list[dict[str, float]] = []
    boxes = None
    conditioning = None
    if mode == "semantic":
        conditioning = _semantic_conditioning(model, semantic_prompt)
    elif mode == "points":
        positive_coords, negative_coords = _parse_points(points, width, height)
    elif mode == "bbox":
        boxes = [_parse_bbox(bbox, width, height)]
    else:
        raise ValueError(f"Unsupported selection mode: {mode}")

    result = SAM3_Detect.execute(
        model=model,
        image=image,
        conditioning=conditioning if mode == "semantic" else None,
        bboxes=boxes,
        positive_coords=json.dumps(positive_coords),
        negative_coords=json.dumps(negative_coords),
        threshold=float(threshold),
        # No external mask refinement is part of this node.  The official
        # detector still performs its normal prompt decoding.
        refine_iterations=0,
        individual_masks=True,
    )
    masks = result[0]
    if not isinstance(masks, torch.Tensor) or masks.ndim != 3 or masks.shape[0] == 0:
        raise ValueError("SAM3 Detect did not find a segmentation object on the anchor frame.")
    return masks.float()


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
        masks = _anchor_masks(
            model,
            frame,
            str(payload.get("mode") or "points"),
            str(payload.get("semantic_prompt") or ""),
            payload.get("points", "[]"),
            payload.get("bbox", "{}"),
            float(payload.get("threshold", 0.5)),
        )
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


def _sec_model_path() -> str:
    try:
        roots = folder_paths.get_folder_paths("sams")
    except KeyError:
        roots = []
    roots = list(roots) or [os.path.join(folder_paths.models_dir, "sams")]
    for root in roots:
        candidate = os.path.join(root, "SeC-4B")
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "config.json")):
            return candidate
    raise ValueError("SeC-4B was not found in models/sams/SeC-4B.")


def _sec_imports() -> tuple[Any, Any, Any]:
    package_root = str(_SEC_PACKAGE_PATH.parent)
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
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
    model_path: str,
    torch_dtype: torch.dtype,
    device: str,
    use_flash_attn: bool,
    allow_mask_overlap: bool,
) -> Any:
    SeCConfig, SeCModel, AutoTokenizer = _sec_imports()
    config = SeCConfig.from_pretrained(model_path)
    config.hydra_overrides_extra = [
        f"++model.non_overlap_masks={'false' if allow_mask_overlap else 'true'}"
    ]
    load_kwargs: dict[str, Any] = {
        "config": config,
        "torch_dtype": torch_dtype,
        "use_flash_attn": use_flash_attn,
        "low_cpu_mem_usage": True,
    }
    if device.startswith("cuda:"):
        load_kwargs["device_map"] = {"": device}
    model = SeCModel.from_pretrained(
        model_path,
        _fast_init=False,
        **load_kwargs,
    ).eval()
    if device.startswith("cuda:") and torch_dtype != torch.float32:
        model = model.to(dtype=torch_dtype)
    else:
        model = model.to(device=device, dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.preparing_for_generation(tokenizer=tokenizer, torch_dtype=torch_dtype)
    if use_flash_attn and device.startswith("cuda:"):
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            print("[CineStyle] SeC flash attention is unavailable; using standard attention.")
    if device.startswith("cuda:") and torch_dtype != torch.float32:
        _sec_install_dtype_hooks(model)
    model._sec_loading_metadata = {
        "model_path": model_path,
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

        torch_dtype, device, use_flash_attn, allow_mask_overlap = _sec_default_model_settings()
        model_path = _sec_model_path()
        print(
            f"[CineStyle] SeC Preview has no registered Loader model; "
            f"cold-loading default SeC-4B from {model_path} on {device}."
        )
        model = _sec_create_model(
            model_path,
            torch_dtype,
            device,
            use_flash_attn,
            allow_mask_overlap,
        )
        model._cinestyle_sec_cache_key = (
            model_path,
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


def _sec_prompt_arrays(
    points: str | None,
    bbox: str | None,
    width: int,
    height: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    positive, negative = [], []
    if points and str(points).strip():
        parsed = _parse_json(points, "points")
        raw_points = parsed.get("points") if isinstance(parsed, dict) else parsed
        if raw_points:
            positive, negative = _parse_points(points, width, height)
    point_values = positive + negative
    point_array = np.asarray([[item["x"], item["y"]] for item in point_values], dtype=np.float32) if point_values else None
    labels = np.asarray([1] * len(positive) + [0] * len(negative), dtype=np.int32) if point_values else None
    box = None
    if bbox and str(bbox).strip():
        parsed_box = _parse_json(bbox, "bbox")
        if parsed_box:
            box = _parse_bbox(bbox, width, height)
    box_array = None if box is None else np.asarray([box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]], dtype=np.float32)
    return point_array, labels, box_array


def _sec_mask_2d(mask: Any) -> np.ndarray:
    """Reduce a SeC mask/logit result to the single selected object's HxW mask."""
    array = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    while array.ndim > 2:
        # SeC returns an object dimension (and some model versions add a
        # singleton mask dimension). This node tracks one object, so select it.
        array = array[0]
    if array.ndim != 2:
        raise ValueError("SeC returned a mask with an unsupported shape.")
    return array


def _sec_add_prompt(model: Any, state: dict[str, Any], frame_index: int, object_id: int, points, labels, box, init_mask):
    if init_mask is not None:
        model.grounding_encoder.add_new_mask(
            inference_state=state,
            frame_idx=frame_index,
            obj_id=object_id,
            mask=_sec_mask_2d(init_mask),
        )
    if points is not None or box is not None:
        _, object_ids, logits = model.grounding_encoder.add_new_points_or_box(
            inference_state=state,
            frame_idx=frame_index,
            obj_id=object_id,
            points=points,
            labels=labels,
            box=box,
        )
        init_mask = _sec_mask_2d(logits > 0.0)
    return init_mask


def _sec_anchor_preview(model: Any, frame: torch.Tensor, mode: str, points: str, bbox: str) -> torch.Tensor:
    height, width = map(int, frame.shape[1:3])
    point_array, labels, box_array = _sec_prompt_arrays(points if mode == "points" else "", bbox if mode == "bbox" else "", width, height)
    if point_array is None and box_array is None:
        raise ValueError("Add at least one point or draw a bounding box before Preview.")
    temp_dir = _sec_frame_dir(frame)
    try:
        states = getattr(model.grounding_encoder, "_states", None)
        if hasattr(states, "clear"):
            states.clear()
        state = model.grounding_encoder.init_state(video_path=temp_dir, offload_video_to_cpu=False, offload_state_to_cpu=False)
        model.grounding_encoder.reset_state(state)
        init_mask = _sec_add_prompt(model, state, 0, 1, point_array, labels, box_array, None)
        if init_mask is None:
            raise ValueError("SeC did not produce a mask for the supplied prompt.")
        return torch.from_numpy((_sec_mask_2d(init_mask) > 0).astype(np.float32))
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
        mask = _sec_anchor_preview(
            model,
            frame,
            str(payload.get("mode") or "points"),
            str(payload.get("points") or "[]"),
            str(payload.get("bbox") or "{}"),
        )
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
                io.Combo.Input("torch_dtype", options=["bfloat16", "float16", "float32"], default="bfloat16"),
                io.Combo.Input("device", options=devices, default="auto"),
                io.Boolean.Input("use_flash_attn", default=True, advanced=True),
                io.Boolean.Input("allow_mask_overlap", default=True, advanced=True),
            ],
            outputs=[SEC_MODEL.Output("model", display_name="SEC_MODEL")],
        )

    @classmethod
    def execute(cls, torch_dtype="bfloat16", device="auto", use_flash_attn=True, allow_mask_overlap=True) -> io.NodeOutput:
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif device.startswith("gpu"):
            device = f"cuda:{int(device[3:])}"
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[str(torch_dtype)]
        if device == "cpu":
            dtype = torch.float32
            use_flash_attn = False
        model_path = _sec_model_path()
        key = (model_path, str(dtype), device, bool(use_flash_attn), bool(allow_mask_overlap))
        with _SEC_MODEL_LOCK:
            model = next((candidate for candidate in _SEC_MODEL_REGISTRY.values() if getattr(candidate, "_cinestyle_sec_cache_key", None) == key), None)
        if model is None:
            print(f"[CineStyle] Loading SeC-4B from {model_path} on {device}.")
            model = _sec_create_model(model_path, dtype, device, bool(use_flash_attn), bool(allow_mask_overlap))
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
                io.Combo.Input("selection_mode", options=["points", "bbox"], default="points"),
                io.Int.Input("anchor_frame", default=0, min=0, max=10000000, step=1),
                io.String.Input("points", default="[]", multiline=True, optional=True),
                io.String.Input("bbox", default="{}", optional=True),
                io.Mask.Input("input_mask", optional=True, tooltip="Optional initial mask at the anchor frame."),
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
        selection_mode: str = "points",
        anchor_frame: int = 0,
        points: str = "[]",
        bbox: str = "{}",
        input_mask: torch.Tensor | None = None,
        tracking_direction: str = "bidirectional",
        max_frames_to_track: int = -1,
        mllm_memory_size: int = 12,
        offload_video_to_cpu: bool = False,
        auto_unload_model: bool = True,
    ) -> io.NodeOutput:
        model = _sec_ensure_loaded(model)
        if images is None and video_input is not None:
            images = video_input.get_components().images
        if images is None:
            raise ValueError("Connect CS Load Video to images or video_input.")
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] < 3:
            raise ValueError("images must have shape [frames, height, width, 3 or 4].")
        images = images[..., :3].to("cpu", dtype=torch.float32).clamp_(0, 1)
        if not _prompt_has_file_video_source(cls.hidden.prompt, cls.hidden.unique_id):
            _cache_selector_input(
                cls.hidden.unique_id,
                images,
                _video_input_fps(video_input, cls.hidden.prompt, cls.hidden.unique_id),
            )
        frame_count, height, width = map(int, images.shape[:3])
        anchor = int(anchor_frame)
        if not 0 <= anchor < frame_count:
            raise ValueError(f"anchor_frame must be between 0 and {frame_count - 1}.")
        if selection_mode not in {"points", "bbox"}:
            raise ValueError("selection_mode must be points or bbox.")
        if tracking_direction not in {"forward", "backward", "bidirectional"}:
            raise ValueError("tracking_direction must be forward, backward, or bidirectional.")
        if selection_mode == "points":
            point_array, labels, box_array = _sec_prompt_arrays(points, "", width, height)
        else:
            point_array, labels, box_array = _sec_prompt_arrays("", bbox, width, height)
        init_mask = None
        if input_mask is not None:
            mask_value = input_mask
            if mask_value.ndim == 3 and mask_value.shape[0] == frame_count:
                mask_value = mask_value[anchor]
            elif mask_value.ndim == 3:
                mask_value = mask_value[0]
            if mask_value.ndim != 2:
                raise ValueError("input_mask must be [H,W] or [frames,H,W].")
            init_mask = (mask_value.detach().cpu().numpy() > 0.5).astype(np.bool_)
            # Treat an empty placeholder mask like a disconnected optional
            # input. This keeps stale workflow links from overriding point or
            # box prompts with an all-zero mask.
            if not init_mask.any():
                init_mask = None
        if point_array is None and box_array is None and init_mask is None:
            raise ValueError("No selection prompt was provided. Click Apply to node after placing points or drawing a bounding box, or connect input_mask.")

        temp_dir = _sec_frame_dir(images)
        state = None
        try:
            state = model.grounding_encoder.init_state(
                video_path=temp_dir,
                offload_video_to_cpu=bool(offload_video_to_cpu),
                offload_state_to_cpu=str(model._sec_loading_metadata.get("device")) == "cpu",
            )
            model.grounding_encoder.reset_state(state)
            init_mask = _sec_add_prompt(model, state, anchor, 1, point_array, labels, box_array, init_mask)
            if init_mask is None:
                raise RuntimeError("SeC did not produce an initial mask.")
            initial_union = torch.from_numpy((_sec_mask_2d(init_mask) > 0).astype(np.float32))
            limit = frame_count if int(max_frames_to_track) < 0 else max(1, int(max_frames_to_track))
            segments: dict[int, torch.Tensor] = {}

            def collect(reverse: bool):
                for frame_index, object_ids, mask_logits in model.propagate_in_video(
                    state,
                    start_frame_idx=anchor,
                    max_frame_num_to_track=limit,
                    reverse=reverse,
                    init_mask=init_mask,
                    tokenizer=None,
                    mllm_memory_size=max(1, int(mllm_memory_size)),
                ):
                    union = (mask_logits > 0.0).any(dim=0).to("cpu", dtype=torch.float32)
                    segments[int(frame_index)] = union

            if tracking_direction in {"forward", "bidirectional"}:
                collect(False)
            if tracking_direction == "bidirectional":
                model.grounding_encoder.reset_state(state)
                _sec_add_prompt(model, state, anchor, 1, point_array, labels, box_array, init_mask)
                collect(True)
            elif tracking_direction == "backward":
                collect(True)

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
                "selection_mode": selection_mode,
                "tracking_direction": tracking_direction,
                "object_count": 1,
            }
            return io.NodeOutput(output, initial_union, info)
        finally:
            try:
                if state is not None:
                    model.grounding_encoder.reset_state(state)
            except Exception:
                pass
            shutil.rmtree(temp_dir, ignore_errors=True)
            if auto_unload_model:
                _sec_unload_model(model)
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
                "Select a semantic object, points, or a hand-drawn box on an arbitrary "
                "video frame, then propagate the mask through the video."
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
                io.Combo.Input(
                    "selection_mode",
                    options=_VIDEO_MODE_OPTIONS,
                    default="points",
                    tooltip="Prompt type used on the anchor frame.",
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
                    "semantic_prompt",
                    default="",
                    placeholder="person, hair, dress",
                    optional=True,
                    tooltip="Semantic object description. The node uses the text encoder bundled in the SAM3 checkpoint.",
                ),
                io.String.Input(
                    "points",
                    default="[]",
                    multiline=True,
                    optional=True,
                    tooltip="Selector data: [{\"x\":0.5,\"y\":0.5,\"label\":1}]. Coordinates may be normalized or pixels.",
                ),
                io.String.Input(
                    "bbox",
                    default="{}",
                    optional=True,
                    tooltip="Selector data: {\"x\":0.2,\"y\":0.2,\"w\":0.4,\"h\":0.5}. Coordinates may be normalized or pixels.",
                ),
                io.Float.Input(
                    "threshold",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                    tooltip="SAM3 semantic detection threshold.",
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
        selection_mode: str = "points",
        anchor_frame: int = 0,
        semantic_prompt: str = "",
        points: str = "[]",
        bbox: str = "{}",
        threshold: float = 0.5,
        propagation_direction: str = "both",
        max_objects: int = 16,
    ) -> io.NodeOutput:
        global _LAST_MODEL
        _LAST_MODEL = model
        if images is None and video_input is not None:
            images = video_input.get_components().images
        if images is None:
            raise ValueError("Connect CS Load Video to images or video_input.")
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] < 3:
            raise ValueError("images must have shape [frames, height, width, 3 or 4].")
        if images.shape[0] == 0:
            raise ValueError("The video contains no frames.")

        images = images[..., :3].to(device="cpu", dtype=torch.float32).clamp_(0.0, 1.0)
        if not _prompt_has_file_video_source(cls.hidden.prompt, cls.hidden.unique_id):
            _cache_selector_input(
                cls.hidden.unique_id,
                images,
                _video_input_fps(video_input, cls.hidden.prompt, cls.hidden.unique_id),
            )
        frame_count, height, width = map(int, images.shape[:3])
        anchor = int(anchor_frame)
        if anchor < 0 or anchor >= frame_count:
            raise ValueError(f"anchor_frame must be between 0 and {frame_count - 1}.")
        if selection_mode not in _VIDEO_MODE_OPTIONS:
            raise ValueError(f"selection_mode must be one of {_VIDEO_MODE_OPTIONS}.")
        if propagation_direction not in _PROPAGATION_OPTIONS:
            raise ValueError(f"propagation_direction must be one of {_PROPAGATION_OPTIONS}.")

        # SAM3.1's multiplex tracker has sixteen object slots. Keep the
        # public control compatible with broader SAM3 workflows, but never
        # pass more than the architectural cap to the tracker.
        object_limit = min(16, max(1, int(max_objects)))

        anchor_mask_objects = _anchor_masks(
            model,
            images[anchor : anchor + 1],
            selection_mode,
            semantic_prompt,
            points,
            bbox,
            float(threshold),
        )
        if anchor_mask_objects.shape[0] > object_limit:
            anchor_mask_objects = anchor_mask_objects[:object_limit]
        anchor_mask = anchor_mask_objects.amax(dim=0).to("cpu").float().clamp_(0.0, 1.0)

        pbar = comfy.utils.ProgressBar(max(1, frame_count * 2))
        mask = _propagate(
            model,
            images,
            anchor_mask_objects,
            anchor,
            propagation_direction,
            pbar,
            object_limit,
        )
        info = {
            "frame_count": frame_count,
            "height": height,
            "width": width,
            "anchor_frame": anchor,
            "selection_mode": selection_mode,
            "semantic_prompt": str(semantic_prompt or ""),
            "propagation_direction": propagation_direction,
            "object_count": int(anchor_mask_objects.shape[0]),
        }
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
