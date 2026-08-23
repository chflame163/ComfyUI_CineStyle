import { app } from "../../../scripts/app.js";
import { registerVideoSelector } from "./video_selector_multi.js";

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
    widgets: { prompt: "prompt_data" },
    videoInputs: ["images", "video_input"],
    note: {},
    removeInputs: ["clip", "conditioning"],
    removeWidgets: ["video", "selection_mode", "semantic_prompt", "points", "bbox", "threshold"],
    preview: async ({ node, filename, previewFrame, promptData, fetchPreview }) => fetchPreview({
        video: filename,
        frame: previewFrame,
        prompt_data: promptData,
        model_source: connectedModelSource(node),
    }),
    apply: ({ node, frame, promptData, setWidgetValue }) => {
        setWidgetValue(node, "anchor_frame", frame);
        setWidgetValue(node, "prompt_data", promptData);
    },
});
