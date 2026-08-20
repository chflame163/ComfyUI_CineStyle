import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js";
import { registerVideoSelector } from "./video_selector.js";

const NODE_ID = "CS_Video_Segment_SeC";

function widget(node, name) {
    return node.widgets?.find((item) => item.name === name);
}

function connectedLoaderNodeId(node) {
    const input = node.inputs?.find((item) => item.name === "model");
    const graph = node.graph || app.graph;
    const link = input?.link == null ? null : graph?.links?.[input.link];
    const originId = link?.origin_id ?? link?.originId;
    const origin = originId == null ? null : graph?.getNodeById?.(originId);
    if (!origin || origin.type !== "CS_SeC_ModelLoader") return "";
    return String(origin.id ?? originId);
}

async function modelToken(node) {
    const loaderNodeId = connectedLoaderNodeId(node);
    const query = loaderNodeId ? `?loader_node_id=${encodeURIComponent(loaderNodeId)}` : "";
    const response = await api.fetchApi(`/cinestyle/sec-models${query}`);
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Unable to query SeC model registry");
    // A registered Loader token is preferred. A null token tells the Preview
    // route to cold-load the default SeC-4B configuration and register it.
    return result.latest || null;
}

registerVideoSelector({
    nodeId: NODE_ID,
    extensionName: "CineStyle.VideoSegmentSeC",
    title: "SeC-4B Video Selector",
    previewRoute: "/cinestyle/sec-video-segment-preview",
    previewLabel: "Loading SeC-4B if needed and running...",
    semantic: false,
    modes: [
        { value: "points", label: "Points" },
        { value: "bbox", label: "Bounding box" },
    ],
    removeWidgets: ["video"],
    note: {},
    preview: async ({ node, filename, previewFrame, mode, points, box, fetchPreview }) => fetchPreview({
        video: filename,
        frame: previewFrame,
        mode,
        points: JSON.stringify(points),
        bbox: JSON.stringify(box || {}),
        model_token: await modelToken(node),
    }),
    apply: ({ node, frame, mode, points, box, setWidgetValue }) => {
        setWidgetValue(node, "selection_mode", mode);
        setWidgetValue(node, "anchor_frame", frame);
        setWidgetValue(node, "points", JSON.stringify(points));
        setWidgetValue(node, "bbox", JSON.stringify(box || {}));
    },
});
