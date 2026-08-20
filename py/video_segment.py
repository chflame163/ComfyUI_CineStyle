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
from typing import Any

import av
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
_PREVIEW_MODEL_CACHE: dict[str, Any] = {}


def _video_files() -> list[str]:
    import os

    input_dir = folder_paths.get_input_directory()
    names = [
        name
        for name in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, name))
    ]
    return sorted(folder_paths.filter_files_content_types(names, ["video"]))


def _decode_video(video: str) -> torch.Tensor:
    if not video or not folder_paths.exists_annotated_filepath(video):
        raise ValueError("Choose a video file or connect an IMAGE video batch.")
    from comfy_api.latest import InputImpl

    components = InputImpl.VideoFromFile(
        folder_paths.get_annotated_filepath(video)
    ).get_components()
    images = components.images
    if not isinstance(images, torch.Tensor):
        raise ValueError("Video decoder did not return a tensor of frames.")
    return images


def _decode_video_frame(video: str, frame_index: int) -> torch.Tensor:
    if not video or not folder_paths.exists_annotated_filepath(video):
        raise ValueError("Choose a video file before requesting a preview.")
    path = folder_paths.get_annotated_filepath(video)
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
        frame = _decode_video_frame(str(payload.get("video") or ""), frame_index)

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
                io.Combo.Input(
                    "video",
                    options=_video_files(),
                    optional=True,
                    upload=io.UploadType.video,
                    tooltip="Video used by the selector and as a fallback when images is not connected.",
                ),
                io.Image.Input(
                    "images",
                    optional=True,
                    tooltip="Video frames as an IMAGE batch. Connect CS Load Video when available.",
                ),
                io.Video.Input(
                    "video_input",
                    optional=True,
                    tooltip="Optional standard VIDEO input. Its decoded frames are used when images is not connected.",
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
        video: str | None = None,
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
            images = _decode_video(video or "")
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] < 3:
            raise ValueError("images must have shape [frames, height, width, 3 or 4].")
        if images.shape[0] == 0:
            raise ValueError("The video contains no frames.")

        images = images[..., :3].to(device="cpu", dtype=torch.float32).clamp_(0.0, 1.0)
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
            _PREVIEW_ROUTE_REGISTERED = True

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CSVideoSegmentSAM3]


async def comfy_entrypoint() -> VideoSegmentExtension:
    return VideoSegmentExtension()
