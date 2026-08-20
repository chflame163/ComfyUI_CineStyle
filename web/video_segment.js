import { app } from "../../../scripts/app.js";
import { registerVideoSelector } from "./video_selector.js";

const NODE_ID = "CS_Video_Segment_SAM3";

function widget(node, name) {
    return node.widgets?.find((item) => item.name === name);
}

function connectedModelSource(node) {
    const input = node.inputs?.find((item) => item.name === "model");
    const graph = node.graph || app.graph;
    const link = input?.link == null ? null : graph?.links?.[input.link];
    const originId = link?.origin_id ?? link?.originId;
    const origin = originId == null ? null : graph?.getNodeById?.(originId);
    if (!origin) return null;
    if (origin.type === "CheckpointLoaderSimple" || origin.type === "CheckpointLoader") {
        return { kind: "checkpoint", name: String(widget(origin, "ckpt_name")?.value || "") };
    }
    if (origin.type === "UNETLoader") {
        return { kind: "diffusion_model", name: String(widget(origin, "unet_name")?.value || "") };
    }
    return null;
}

registerVideoSelector({
    nodeId: NODE_ID,
    extensionName: "CineStyle.VideoSegmentSAM3",
    title: "SAM3.1 Video Selector",
    previewRoute: "/cinestyle/video-segment-preview",
    previewLabel: "Running SAM3.1 on this frame...",
    semantic: true,
    modes: [
        { value: "points", label: "Points" },
        { value: "bbox", label: "Bounding box" },
        { value: "semantic", label: "Semantic" },
    ],
    note: { semantic: "Enter the object description, then preview the current frame." },
    removeInputs: ["clip", "conditioning"],
    removeWidgets: ["video"],
    preview: async ({ node, filename, previewFrame, mode, semanticPrompt, points, box, fetchPreview }) => fetchPreview({
        video: filename,
        frame: previewFrame,
        mode,
        semantic_prompt: semanticPrompt,
        points: JSON.stringify(points),
        bbox: JSON.stringify(box || {}),
        threshold: Number(widget(node, "threshold")?.value ?? 0.5),
        model_source: connectedModelSource(node),
    }),
    apply: ({ node, frame, mode, semanticPrompt, points, box, setWidgetValue }) => {
        setWidgetValue(node, "selection_mode", mode);
        setWidgetValue(node, "anchor_frame", frame);
        setWidgetValue(node, "semantic_prompt", semanticPrompt);
        setWidgetValue(node, "points", JSON.stringify(points));
        setWidgetValue(node, "bbox", JSON.stringify(box || {}));
    },
});
