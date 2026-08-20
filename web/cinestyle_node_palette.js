import { app } from "../../../scripts/app.js";

const NODE_PREFIX = "CS_";
const GREEN_GRAY_TITLE = "rgba(67, 105, 82, 0.85)";

app.registerExtension({
    name: "CineStyle.NodePalette",
    nodeCreated(node) {
        const comfyClass = node.comfyClass || node.constructor?.comfyClass || "";
        if (!comfyClass.startsWith(NODE_PREFIX)) return;
        node.color = GREEN_GRAY_TITLE;
    },
});
