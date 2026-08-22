import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import {
    connectedVideoSource,
    fetchCachedSource,
    fetchInfo,
    prepareInputTimeline,
    sourceFrameForLocal,
} from "./video_selector_multi.js";

const STYLE_ID = "cinestyle-vfx-beauty-style";

const PARAMS = [
    ["blur_m", "Soften", 0, 100, 10, 0.01, "Softens the keyed skin matte."],
    ["sigma", "Amount", 0, 100, 10, 0.01, "Controls the edge-preserving skin blur."],
    ["threshold", "Preserve Edges", 0, 100, 15, 0.01, "Keeps stronger edges out of the blur."],
    ["r_spots_blend", "Dark Spots", 0, 1, 0.8, 0.001, "Blends away dark spots inside the matte."],
    ["r_h_blend", "Highlights", 0, 1, 0.5, 0.001, "Blends the highlight recovery pass."],
    ["strength", "Restore Detail", 0, 10, 0, 0.01, "Adds back fine facial detail."],
    ["blur_h", "Detail Soften", 0, 50, 0, 0.01, "Softens the restored high-frequency detail."],
    ["blur_s", "Blur Shine", 0, 100, 30, 0.01, "Controls the shine blur radius."],
    ["o_amount", "Shine Amount", 0, 1, 0.2, 0.001, "Sets the recovered shine strength."],
    ["sat_amount", "Saturation", 0, 1000, 100, 0.1, "Scales the final skin saturation."],
    ["hue_amount", "Hue Shift", -360, 360, 0, 0.01, "Rotates the final skin hue in degrees."],
];

function addStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .cs-vfx-dialog{width:min(1180px,96vw);max-width:none;max-height:94vh;overflow:auto;padding:0;border:1px solid #343943;border-radius:10px;background:#17191e;color:#e6e9ef;box-shadow:0 22px 80px #000b}
      .cs-vfx-dialog::backdrop{background:#050609b8}
      .cs-vfx-shell{display:grid;gap:12px;padding:16px;font:13px/1.35 system-ui,sans-serif}
      .cs-vfx-head,.cs-vfx-row,.cs-vfx-actions{display:flex;align-items:center;gap:9px;flex-wrap:wrap}
      .cs-vfx-head{justify-content:space-between}.cs-vfx-title{margin:0;font-size:17px}.cs-vfx-muted,.cs-vfx-status{color:#9da5b4}
      .cs-vfx-button{min-height:31px;border:1px solid #424956;border-radius:5px;padding:6px 10px;background:#20232a;color:#f2f4f7;cursor:pointer}
      .cs-vfx-button:hover{border-color:#6aa9df}.cs-vfx-button.active{background:#317ec4;border-color:#6db6ee}.cs-vfx-close{font-size:18px;padding:3px 10px}
      .cs-vfx-view-grid{display:grid;grid-template-columns:1fr 1fr;gap:9px;min-height:260px}
      .cs-vfx-viewport{position:relative;min-height:260px;overflow:hidden;background:#08090b;border:1px solid #343943;border-radius:6px}
      .cs-vfx-viewport-label{position:absolute;z-index:2;top:8px;left:9px;padding:3px 6px;border-radius:4px;background:#111419c9;color:#cbd2dc;font-size:12px}
      .cs-vfx-viewport img{display:block;width:100%;height:100%;min-height:260px;object-fit:contain;transform-origin:center center;transition:transform .08s linear}
      .cs-vfx-zoom{display:flex;gap:5px;align-items:center}.cs-vfx-zoom .cs-vfx-button{min-height:27px;padding:4px 8px}
      .cs-vfx-controls{display:grid;grid-template-columns:auto minmax(100px,1fr) 74px auto;align-items:center;gap:8px}
      .cs-vfx-step-buttons{display:flex;gap:5px}.cs-vfx-step-buttons .cs-vfx-button{width:38px;padding-inline:0}
      .cs-vfx-controls input[type=range]{width:100%}.cs-vfx-frame-input{width:74px;min-height:29px;border:1px solid #424956;border-radius:5px;padding:4px 7px;background:#111419;color:#f2f4f7}
      .cs-vfx-param-grid{display:grid;grid-template-columns:repeat(3,minmax(180px,1fr));gap:10px 14px;border-top:1px solid #343943;padding-top:12px}
      .cs-vfx-param{display:grid;grid-template-columns:minmax(92px,auto) 1fr 54px;align-items:center;gap:7px}.cs-vfx-param label{color:#d9dee6}.cs-vfx-param input[type=range]{width:100%}
      .cs-vfx-param output{color:#f7b955;text-align:right;font-variant-numeric:tabular-nums}.cs-vfx-help{grid-column:1/-1;color:#8993a2;font-size:11px;margin-top:-5px}
      .cs-vfx-text{min-height:29px;border:1px solid #424956;border-radius:5px;padding:5px 8px;background:#111419;color:#f2f4f7}.cs-vfx-text:focus,.cs-vfx-frame-input:focus{outline:1px solid #55a9f5;border-color:#55a9f5}
      .cs-vfx-inline{display:grid;grid-template-columns:auto minmax(120px,180px) auto minmax(170px,240px);gap:8px;align-items:center}.cs-vfx-inline label{color:#9da5b4}
      .cs-vfx-actions{justify-content:flex-end}.cs-vfx-status{flex:1;min-width:120px}.cs-vfx-error{color:#ff939b}
      @media(max-width:760px){.cs-vfx-view-grid{grid-template-columns:1fr}.cs-vfx-param-grid{grid-template-columns:1fr}.cs-vfx-controls{grid-template-columns:auto 1fr auto}.cs-vfx-frame-count{grid-column:1/-1}.cs-vfx-inline{grid-template-columns:auto 1fr}}
    `;
    document.head.append(style);
}

function widget(node, name) { return node.widgets?.find((item) => item.name === name); }
function valueOf(node, name, fallback) { const value = widget(node, name)?.value; return value == null ? fallback : value; }
function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function setZoom(dialog, zoom) {
    const value = clamp(Number(zoom) || 1, 0.25, 4);
    dialog._vfxZoom = value;
    dialog.querySelectorAll(".cs-vfx-image").forEach((image) => { image.style.transform = `scale(${value})`; });
    dialog.querySelector(".cs-vfx-zoom-value").textContent = `${Math.round(value * 100)}%`;
}

async function openPreview(node) {
    addStyles();
    let dialog = null;
    const status = (message, error = false) => {
        const element = dialog?.querySelector(".cs-vfx-status");
        if (element) { element.textContent = message; element.classList.toggle("cs-vfx-error", error); }
    };
    let source = null;
    try { source = await fetchCachedSource(node); } catch (error) { status(error.message, true); return; }
    if (!source) source = connectedVideoSource(node, ["image", "images", "video_input"]);
    if (!source) { app.canvas?.prompt?.("Run the workflow once to cache the connected image/video input.", ""); return; }
    if (!source.token && !source.filename) { app.canvas?.prompt?.("No previewable input source was found.", ""); return; }
    let info = source.info;
    if (!info) {
        try { info = prepareInputTimeline(source, await fetchInfo(source.filename)); } catch (error) { app.canvas?.prompt?.(error.message, ""); return; }
    }
    dialog = document.createElement("dialog");
    dialog.className = "cs-vfx-dialog";
    dialog._vfxZoom = 1;
    dialog.innerHTML = `<div class="cs-vfx-shell">
      <div class="cs-vfx-head"><div><h2 class="cs-vfx-title">VFX Preview</h2><div class="cs-vfx-muted cs-vfx-file"></div></div><button class="cs-vfx-button cs-vfx-close" type="button">&times;</button></div>
      <div class="cs-vfx-view-grid"><div class="cs-vfx-viewport"><span class="cs-vfx-viewport-label">Original</span><img class="cs-vfx-image cs-vfx-original" alt="Original frame"></div><div class="cs-vfx-viewport"><span class="cs-vfx-viewport-label">VFX Beauty</span><img class="cs-vfx-image cs-vfx-result" alt="VFX Beauty preview"></div></div>
      <div class="cs-vfx-row"><div class="cs-vfx-zoom"><span class="cs-vfx-muted">Zoom</span><button class="cs-vfx-button" data-zoom="0.5" type="button">50%</button><button class="cs-vfx-button" data-zoom="1" type="button">100%</button><button class="cs-vfx-button" data-zoom="2" type="button">200%</button><button class="cs-vfx-button" data-zoom="fit" type="button">Fit</button><span class="cs-vfx-muted cs-vfx-zoom-value">100%</span></div></div>
      <div class="cs-vfx-controls"><div class="cs-vfx-step-buttons"><button class="cs-vfx-button cs-vfx-prev" type="button">|&lt;</button><button class="cs-vfx-button cs-vfx-next" type="button">&gt;|</button></div><input class="cs-vfx-timeline" type="range" min="0" max="0" step="1" value="0"><input class="cs-vfx-frame-input" type="number" min="0" max="0" step="1" value="0"><span class="cs-vfx-frame-count cs-vfx-muted">0 / 0</span></div>
      <div class="cs-vfx-inline"><label for="cs-vfx-colour">Colour</label><input id="cs-vfx-colour" class="cs-vfx-text cs-vfx-colour" value="${String(valueOf(node, "colour", "auto")).replace(/"/g, "&quot;")}"><label for="cs-vfx-weights">Weights</label><input id="cs-vfx-weights" class="cs-vfx-text cs-vfx-weights" value="${String(valueOf(node, "weights", "6.0, 0.0, 3.0")).replace(/"/g, "&quot;")}"></div>
      <div class="cs-vfx-param-grid">${PARAMS.map(([name, label, min, max, defaultValue, step, help]) => `<div class="cs-vfx-param"><label for="cs-vfx-${name}">${label}</label><input id="cs-vfx-${name}" data-param="${name}" type="range" min="${min}" max="${max}" step="${step}" value="${valueOf(node, name, defaultValue)}"><output data-output="${name}">${valueOf(node, name, defaultValue)}</output><div class="cs-vfx-help">${help}</div></div>`).join("")}</div>
      <div class="cs-vfx-actions"><span class="cs-vfx-status">Loading preview...</span><button class="cs-vfx-button cs-vfx-cancel" type="button">Close</button><button class="cs-vfx-button active cs-vfx-apply" type="button">Apply to Node</button></div>
    </div>`;
    document.body.append(dialog);
    dialog.querySelector(".cs-vfx-file").textContent = `${source.label || source.filename || "Cached input"} · ${info.frames || 1} frames`;
    const maxFrame = Math.max(0, Number(info.frames || 1) - 1);
    const timeline = dialog.querySelector(".cs-vfx-timeline");
    const frameInput = dialog.querySelector(".cs-vfx-frame-input");
    timeline.max = String(maxFrame); frameInput.max = String(maxFrame);
    let frame = 0; let requestSerial = 0; let timer = null;
    const setFrame = (next) => { frame = clamp(Math.round(Number(next) || 0), 0, maxFrame); timeline.value = String(frame); frameInput.value = String(frame); dialog.querySelector(".cs-vfx-frame-count").textContent = `${frame} / ${maxFrame}`; schedulePreview(); };
    const payload = () => {
        const result = { node_id: String(node.id), source_token: source.token || "", video: source.filename || "", frame: source.token ? frame : sourceFrameForLocal(info, frame), local_frame: frame, colour: dialog.querySelector(".cs-vfx-colour").value.trim(), weights: dialog.querySelector(".cs-vfx-weights").value.trim() };
        dialog.querySelectorAll("[data-param]").forEach((input) => { result[input.dataset.param] = Number(input.value); });
        return result;
    };
    async function preview() {
        const serial = ++requestSerial; status(`Rendering frame ${frame}...`);
        try {
            const response = await api.fetchApi("/cinestyle/vfx-beauty-preview", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload()) });
            const result = await response.json(); if (!response.ok) throw new Error(result.error || "Preview failed"); if (serial !== requestSerial) return;
            dialog.querySelector(".cs-vfx-original").src = result.original; dialog.querySelector(".cs-vfx-result").src = result.preview; status(`Frame ${result.frame} · colour ${result.colour}`);
        } catch (error) { if (serial === requestSerial) status(error.message, true); }
    }
    function schedulePreview() { clearTimeout(timer); timer = setTimeout(preview, 100); }
    timeline.addEventListener("input", () => setFrame(timeline.value)); frameInput.addEventListener("change", () => setFrame(frameInput.value));
    dialog.querySelector(".cs-vfx-prev").addEventListener("click", () => setFrame(frame - 1)); dialog.querySelector(".cs-vfx-next").addEventListener("click", () => setFrame(frame + 1));
    dialog.querySelectorAll("[data-param]").forEach((input) => input.addEventListener("input", () => { dialog.querySelector(`[data-output="${input.dataset.param}"]`).value = input.value; schedulePreview(); }));
    dialog.querySelectorAll(".cs-vfx-colour,.cs-vfx-weights").forEach((input) => input.addEventListener("change", schedulePreview));
    dialog.querySelectorAll("[data-zoom]").forEach((button) => button.addEventListener("click", () => setZoom(dialog, button.dataset.zoom === "fit" ? 1 : Number(button.dataset.zoom))));
    dialog.querySelectorAll(".cs-vfx-viewport").forEach((viewport) => viewport.addEventListener("wheel", (event) => { event.preventDefault(); setZoom(dialog, dialog._vfxZoom + (event.deltaY < 0 ? 0.1 : -0.1)); }, { passive: false }));
    const close = () => { clearTimeout(timer); dialog.close(); dialog.remove(); }; dialog.querySelector(".cs-vfx-close").addEventListener("click", close); dialog.querySelector(".cs-vfx-cancel").addEventListener("click", close); dialog.addEventListener("cancel", close);
    dialog.querySelector(".cs-vfx-apply").addEventListener("click", () => {
        const setValue = (name, value) => { const target = widget(node, name); if (target) { target.value = value; target.callback?.(value); } };
        setValue("colour", dialog.querySelector(".cs-vfx-colour").value.trim());
        setValue("weights", dialog.querySelector(".cs-vfx-weights").value.trim());
        dialog.querySelectorAll("[data-param]").forEach((input) => setValue(input.dataset.param, Number(input.value)));
        node.graph?.setDirtyCanvas(true, true); close();
    });
    dialog.showModal(); setZoom(dialog, 1); setFrame(0);
}

app.registerExtension({
    name: "CineStyle.VFXBeautyPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "CS_VFX_Beauty") return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () { original?.apply(this, arguments); const button = this.addWidget("button", "VFX Preview", "", () => openPreview(this)); button.name = "VFX Preview"; button.label = "VFX Preview"; button.options = { ...(button.options || {}), serialize: false }; this.setSize?.([390, Math.max(420, this.computeSize?.()[1] || 420)]); };
    },
    loadedGraphNode(node) { if (node?.type !== "CS_VFX_Beauty") return; node.setSize?.([node.size?.[0] || 390, Math.max(420, node.computeSize?.()[1] || node.size?.[1] || 420)]); },
});
