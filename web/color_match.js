import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import {
    connectedInputChain,
    connectedVideoSource,
    ensureLoaderPreviewSource,
    fetchInfo,
    prepareInputTimeline,
    sourceFrameForLocal,
    fetchWaitInputCache,
} from "./video_selector_multi.js";

const NODE_ID = "CS_Color_Match";
const STYLE_ID = "cinestyle-color-match-style";
const METHODS = ["Reinhard", "LHM", "PCCM", "PDF", "Optimal Transport"];
const COLOR_SPACES = ["Lab", "OKLab"];
const PARAMS = [
    ["match_strength", "Match Strength", 0.75],
    ["preserve_luminance", "Preserve Luminance", 1],
    ["preserve_contrast", "Preserve Contrast", 1],
    ["preserve_saturation", "Preserve Saturation", 0],
    ["hue_strength", "Hue Strength", 1],
    ["chroma_strength", "Chroma Strength", 1],
];

function addStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .cs-match-dialog{width:min(1000px,96vw);max-width:none;max-height:95vh;overflow:auto;padding:0;border:1px solid #353a43;border-radius:8px;background:#17191e;color:#e8ebef;box-shadow:0 22px 80px #000b}
      .cs-match-dialog::backdrop{background:#050609b8}
      .cs-match-shell{display:grid;gap:12px;padding:16px;font:13px/1.35 system-ui,sans-serif}
      .cs-match-head,.cs-match-actions,.cs-match-zoom{display:flex;align-items:center;gap:7px;flex-wrap:wrap}
      .cs-match-head{justify-content:space-between}.cs-match-title{margin:0;font-size:17px;letter-spacing:0}.cs-match-muted,.cs-match-status{color:#9da5b4}
      .cs-match-button{min-height:30px;border:1px solid #454c57;border-radius:5px;padding:5px 9px;background:#22252c;color:#f2f4f7;cursor:pointer}
      .cs-match-button:hover{border-color:#79aee0}.cs-match-button.active{border-color:#74b8f0;background:#2879b8}.cs-match-close{font-size:18px;padding:2px 9px}
      .cs-match-preview-wrap{position:relative;display:flex;justify-content:center;min-height:320px}
      .cs-match-cache-loading{position:absolute;z-index:10;inset:0;display:flex;align-items:center;justify-content:center;padding:18px;background:#08090be8;color:#dce7f3;text-align:center}.cs-match-cache-loading[hidden]{display:none}
      .cs-match-viewport{position:relative;width:min(900px,100%);height:clamp(320px,44vh,560px);overflow:hidden;border:1px solid #343943;border-radius:6px;background:#08090b;--compare-position:0%;cursor:default;touch-action:none;user-select:none}
      .cs-match-viewport.pan-ready{cursor:grab}.cs-match-viewport.pan-active{cursor:grabbing}.cs-match-viewport-label{position:absolute;z-index:4;top:8px;left:9px;padding:3px 6px;border-radius:4px;background:#111419cf;color:#d8dde5;font-size:12px}
      .cs-match-image{position:absolute;inset:0;display:block;width:100%;height:100%;object-fit:contain;transform-origin:center center;transition:transform .08s linear;pointer-events:none;user-select:none;-webkit-user-drag:none}
      .cs-match-original-clip{position:absolute;inset:0;z-index:2;overflow:hidden;pointer-events:none;clip-path:inset(0 calc(100% - var(--compare-position)) 0 0)}
      .cs-match-divider{position:absolute;z-index:3;top:0;bottom:0;left:var(--compare-position);width:2px;transform:translateX(-1px);background:#f3f5f7;box-shadow:0 0 0 1px #1118;cursor:ew-resize;touch-action:none}
      .cs-match-divider::before{content:"";position:absolute;top:50%;left:50%;width:24px;height:24px;transform:translate(-50%,-50%);border:2px solid #f2f4f7;border-radius:50%;background:#20232a;box-shadow:0 2px 8px #000b}
      .cs-match-divider::after{content:"\u2194";position:absolute;top:50%;left:50%;transform:translate(-50%,-53%);font-size:14px;line-height:1;color:#f2f4f7}
      .cs-match-zoom{justify-content:center}.cs-match-zoom .cs-match-button{min-height:27px;padding:3px 8px}
      .cs-match-timeline{display:grid;grid-template-columns:auto minmax(120px,1fr) 76px auto;align-items:center;gap:8px}.cs-match-step{display:flex;gap:5px}.cs-match-step .cs-match-button{width:38px;padding-inline:0}
      .cs-match-number,.cs-match-select{min-height:29px;box-sizing:border-box;border:1px solid #454c57;border-radius:4px;padding:4px 7px;background:#101216;color:#f2f4f7;font-variant-numeric:tabular-nums}.cs-match-frame{width:76px}
      .cs-match-lower{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:16px;border-top:1px solid #343943;padding-top:14px;align-items:start}
      .cs-match-options{display:grid;grid-template-columns:max-content 146px;align-items:center;column-gap:8px;row-gap:12px}.cs-match-option{display:contents}.cs-match-option label{text-align:right;white-space:nowrap}.cs-match-select{width:146px;max-width:100%}
      .cs-match-params{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));column-gap:14px}.cs-match-param-column{display:grid;grid-template-columns:max-content minmax(80px,1fr) 68px 30px;align-items:center;column-gap:6px;row-gap:10px}.cs-match-param{display:contents}.cs-match-param label{padding-right:calc(2ch - 6px);text-align:right;white-space:nowrap}.cs-match-param input[type=range]{width:100%;min-width:0}.cs-match-param .cs-match-number{width:68px}.cs-match-reset{width:30px;padding:2px;font-size:15px}
      .cs-match-actions{justify-content:flex-end}.cs-match-status{flex:1;min-width:160px}.cs-match-error{color:#ff939b}
      @media(max-width:920px){.cs-match-lower{grid-template-columns:1fr}.cs-match-params{grid-template-columns:1fr;row-gap:10px}.cs-match-options{justify-content:start}}
      @media(max-width:620px){.cs-match-timeline{grid-template-columns:auto 1fr 76px}.cs-match-frame-count{grid-column:1/-1}.cs-match-preview-wrap{min-height:280px}.cs-match-viewport{height:330px}.cs-match-param-column{grid-template-columns:max-content minmax(64px,1fr) 68px 30px}}
    `;
    document.head.append(style);
}

function widget(node, name) { return node.widgets?.find((item) => item.name === name); }
function valueOf(node, name, fallback) { const value = widget(node, name)?.value; return value == null ? fallback : value; }
function comboOptions(node, name, fallback) {
    const options = widget(node, name)?.options;
    const values = Array.isArray(options?.values) ? options.values : (Array.isArray(options) ? options : fallback);
    return values.map(String);
}
function escapeHtml(value) { return String(value).replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character])); }
function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }

async function fetchCachedInputs(node) {
    const nodeId = String(node?.id ?? "").trim();
    if (!nodeId) return {};
    const response = await api.fetchApi(`/cinestyle/color-match-cache?${new URLSearchParams({ node_id: nodeId, t: String(Date.now()) })}`);
    if (response.status === 404) return {};
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Unable to read CS Color Match preview cache");
    const convert = (entry) => {
        if (!entry) return null;
        const info = entry.info || {};
        return {
            filename: "",
            kind: entry.kind || (Number(info.frames || 1) > 1 ? "video" : "image"),
            label: String(entry.label || "Cached CS Color Match input"),
            url: entry.video_url ? api.apiURL(entry.video_url) : "",
            token: String(entry.token || ""),
            info,
            startFrame: 0,
            endFrame: Math.max(0, Number(info.frames || 1) - 1),
            targetFps: Number(info.fps || 24),
        };
    };
    return { source: convert(result.source), reference: convert(result.reference) };
}

async function resolveInput(node, inputName, cached, setLoading) {
    const upstream = connectedVideoSource(node, [inputName]);
    const chain = connectedInputChain(node, [inputName]);
    let source = chain ? await fetchWaitInputCache(chain).catch(() => null) : null;
    if (!source) source = cached || upstream;
    if (!source) throw new Error(`Run ComfyUI once to cache the connected ${inputName} input.`);
    if (source.loaderId && !source.token) {
        source = await ensureLoaderPreviewSource(source, {
            onProgress: (progress) => setLoading(`Preparing ${inputName} cache, please wait ${progress}%`),
        });
    }
    if (!source.token && !source.filename) throw new Error(`No previewable ${inputName} source was found.`);
    source.info = source.info || prepareInputTimeline(source, await fetchInfo(source.filename));
    return source;
}

function clampPan(dialog) {
    const pan = dialog._matchPan || { x: 0, y: 0 };
    const viewport = dialog.querySelector(".cs-match-viewport");
    const image = dialog.querySelector(".cs-match-result");
    const zoom = Number(dialog._matchZoom || 1);
    const maxX = Math.max(0, (image.clientWidth * zoom - viewport.clientWidth) / 2);
    const maxY = Math.max(0, (image.clientHeight * zoom - viewport.clientHeight) / 2);
    dialog._matchPan = { x: clamp(pan.x, -maxX, maxX), y: clamp(pan.y, -maxY, maxY) };
}

function applyViewportTransform(dialog) {
    const pan = dialog._matchPan || { x: 0, y: 0 };
    const zoom = Number(dialog._matchZoom || 1);
    dialog.querySelectorAll(".cs-match-image").forEach((image) => { image.style.transform = `translate(${pan.x}px,${pan.y}px) scale(${zoom})`; });
    dialog.querySelector(".cs-match-viewport")?.classList.toggle("pan-ready", zoom > 1);
    dialog.querySelector(".cs-match-zoom-value").textContent = `${Math.round(zoom * 100)}%`;
}

function setZoom(dialog, zoom) {
    dialog._matchZoom = clamp(Number(zoom) || 1, 0.5, 4);
    if (dialog._matchZoom <= 1) dialog._matchPan = { x: 0, y: 0 };
    clampPan(dialog);
    applyViewportTransform(dialog);
}

function setCompare(dialog, percent) {
    const value = clamp(Number(percent) || 0, 0, 100);
    dialog.querySelector(".cs-match-viewport")?.style.setProperty("--compare-position", `${value}%`);
}

function setNodeValue(node, name, value) {
    const target = widget(node, name);
    if (!target) return;
    const index = node.widgets?.indexOf(target) ?? -1;
    target.value = value;
    target.callback?.(value);
    target.value = value;
    if (index >= 0 && Array.isArray(node.widgets_values)) node.widgets_values[index] = value;
}

async function openPreview(node) {
    addStyles();
    const methodOptions = comboOptions(node, "method", METHODS);
    const colorOptions = comboOptions(node, "color_space", COLOR_SPACES);
    const method = String(valueOf(node, "method", "Optimal Transport"));
    const colorSpace = String(valueOf(node, "color_space", "OKLab"));
    const paramMarkup = [PARAMS.slice(0, 3), PARAMS.slice(3)].map((column) => `<div class="cs-match-param-column">${column.map(([name, label, defaultValue]) => `<div class="cs-match-param"><label for="cs-match-${name}">${label}</label><input id="cs-match-${name}" data-match-param="${name}" type="range" min="0" max="1" step="0.01" value="${valueOf(node, name, defaultValue)}"><input class="cs-match-number" data-match-number="${name}" type="number" min="0" max="1" step="0.01" value="${valueOf(node, name, defaultValue)}"><button class="cs-match-button cs-match-reset" data-match-reset="${name}" type="button" title="Reset ${label}">&#8634;</button></div>`).join("")}</div>`).join("");
    const dialog = document.createElement("dialog");
    dialog.className = "cs-match-dialog";
    dialog.innerHTML = `<div class="cs-match-shell"><div class="cs-match-head"><div><h2 class="cs-match-title">Match Preview</h2><div class="cs-match-muted cs-match-file"></div></div><button class="cs-match-button cs-match-close" type="button">&times;</button></div><div class="cs-match-preview-wrap"><div class="cs-match-viewport"><span class="cs-match-viewport-label">Result / Original</span><img class="cs-match-image cs-match-result" draggable="false" alt="Matched preview"><div class="cs-match-original-clip"><img class="cs-match-image cs-match-original" draggable="false" alt="Original comparison"></div><div class="cs-match-divider" title="Drag to compare Original and Result"></div></div><div class="cs-match-cache-loading" role="status">Preparing cache, please wait 0%</div></div><div class="cs-match-zoom"><span class="cs-match-muted">Zoom</span><button class="cs-match-button" data-match-zoom="0.5" type="button">50%</button><button class="cs-match-button" data-match-zoom="1" type="button">100%</button><button class="cs-match-button" data-match-zoom="2" type="button">200%</button><button class="cs-match-button" data-match-zoom="fit" type="button">Fit</button><span class="cs-match-muted cs-match-zoom-value">100%</span></div><div class="cs-match-timeline"><div class="cs-match-step"><button class="cs-match-button cs-match-prev" type="button">|&lt;</button><button class="cs-match-button cs-match-next" type="button">&gt;|</button></div><input class="cs-match-timeline-range" type="range" min="0" max="0" step="1" value="0"><input class="cs-match-number cs-match-frame" type="number" min="0" max="0" step="1" value="0"><span class="cs-match-muted cs-match-frame-count">0 / 0</span></div><div class="cs-match-lower"><section class="cs-match-options"><div class="cs-match-option"><label for="cs-match-method">Method</label><select id="cs-match-method" class="cs-match-select">${methodOptions.map((option) => `<option value="${escapeHtml(option)}"${option === method ? " selected" : ""}>${escapeHtml(option)}</option>`).join("")}</select></div><div class="cs-match-option"><label for="cs-match-space">Color Space</label><select id="cs-match-space" class="cs-match-select">${colorOptions.map((option) => `<option value="${escapeHtml(option)}"${option === colorSpace ? " selected" : ""}>${escapeHtml(option)}</option>`).join("")}</select></div></section><section class="cs-match-params">${paramMarkup}</section></div><div class="cs-match-actions"><span class="cs-match-status">Loading preview...</span><button class="cs-match-button cs-match-reset-all" type="button">Reset All</button><button class="cs-match-button cs-match-cancel" type="button">Close</button><button class="cs-match-button active cs-match-apply" type="button">Apply to Node</button></div></div>`;
    document.body.append(dialog);
    const loading = dialog.querySelector(".cs-match-cache-loading");
    const statusElement = dialog.querySelector(".cs-match-status");
    let closed = false; let timer = null; let resizeObserver = null;
    const status = (message, error = false) => { statusElement.textContent = message; statusElement.classList.toggle("cs-match-error", error); };
    const setLoading = (message, visible = true) => { loading.textContent = message; loading.hidden = !visible; };
    const earlyClose = () => { closed = true; dialog.close(); dialog.remove(); };
    dialog.querySelector(".cs-match-close").addEventListener("click", earlyClose);
    dialog.querySelector(".cs-match-cancel").addEventListener("click", earlyClose);
    dialog.addEventListener("cancel", earlyClose);
    dialog.showModal();

    let source; let reference;
    try {
        const cached = await fetchCachedInputs(node);
        source = await resolveInput(node, "image", cached.source, setLoading);
        reference = await resolveInput(node, "reference_image", cached.reference, setLoading);
    } catch (error) {
        if (!closed) setLoading(error?.message || "Unable to prepare preview cache");
        return;
    }
    if (closed) return;
    dialog.querySelector(".cs-match-close").removeEventListener("click", earlyClose);
    dialog.querySelector(".cs-match-cancel").removeEventListener("click", earlyClose);
    dialog.removeEventListener("cancel", earlyClose);
    setLoading("Preparing cache, please wait 100%", false);
    const info = source.info || {};
    dialog.querySelector(".cs-match-file").textContent = `${source.label || source.filename || "Cached input"} · ${info.frames || 1} frames`;
    const timeline = dialog.querySelector(".cs-match-timeline-range");
    const frameInput = dialog.querySelector(".cs-match-frame");
    const maxFrame = Math.max(0, Number(info.frames || 1) - 1);
    timeline.max = String(maxFrame); frameInput.max = String(maxFrame);
    let frame = 0; let requestSerial = 0; let compareDragging = false;
    const scalarValues = () => Object.fromEntries(PARAMS.map(([name]) => [name, Number(dialog.querySelector(`[data-match-param="${name}"]`).value)]));
    const payload = () => ({
        node_id: String(node.id),
        source_kind: source.kind || "video",
        source_token: source.token || "",
        source_file: source.filename || "",
        source_info: info,
        reference_kind: reference.kind || "image",
        reference_token: reference.token || "",
        reference_file: reference.filename || "",
        frame: source.token ? frame : sourceFrameForLocal(info, frame),
        local_frame: frame,
        method: dialog.querySelector("#cs-match-method").value,
        color_space: dialog.querySelector("#cs-match-space").value,
        ...scalarValues(),
    });
    async function preview() {
        const serial = ++requestSerial;
        status(`Rendering frame ${frame}...`);
        try {
            const response = await api.fetchApi("/cinestyle/color-match-preview", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload()) });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || "Preview failed");
            if (serial !== requestSerial) return;
            dialog.querySelector(".cs-match-original").src = result.original;
            dialog.querySelector(".cs-match-result").src = result.preview;
            status(`Frame ${result.frame}`);
        } catch (error) {
            if (serial === requestSerial) status(error.message, true);
        }
    }
    const schedulePreview = () => { clearTimeout(timer); timer = setTimeout(preview, 100); };
    const setFrame = (value) => {
        frame = clamp(Math.round(Number(value) || 0), 0, maxFrame);
        timeline.value = String(frame); frameInput.value = String(frame);
        dialog.querySelector(".cs-match-frame-count").textContent = `${frame} / ${maxFrame}`;
        schedulePreview();
    };
    timeline.addEventListener("input", () => setFrame(timeline.value));
    frameInput.addEventListener("change", () => setFrame(frameInput.value));
    dialog.querySelector(".cs-match-prev").addEventListener("click", () => setFrame(frame - 1));
    dialog.querySelector(".cs-match-next").addEventListener("click", () => setFrame(frame + 1));
    dialog.querySelectorAll("[data-match-param]").forEach((range) => range.addEventListener("input", () => { dialog.querySelector(`[data-match-number="${range.dataset.matchParam}"]`).value = range.value; schedulePreview(); }));
    dialog.querySelectorAll("[data-match-number]").forEach((number) => number.addEventListener("input", () => { const value = Number(number.value); if (!Number.isFinite(value)) return; const range = dialog.querySelector(`[data-match-param="${number.dataset.matchNumber}"]`); range.value = String(clamp(value, 0, 1)); schedulePreview(); }));
    dialog.querySelectorAll("[data-match-reset]").forEach((button) => button.addEventListener("click", () => { const definition = PARAMS.find(([name]) => name === button.dataset.matchReset); const value = definition[2]; dialog.querySelector(`[data-match-param="${definition[0]}"]`).value = String(value); dialog.querySelector(`[data-match-number="${definition[0]}"]`).value = String(value); schedulePreview(); }));
    dialog.querySelectorAll(".cs-match-select").forEach((select) => select.addEventListener("change", schedulePreview));

    const viewport = dialog.querySelector(".cs-match-viewport");
    const divider = dialog.querySelector(".cs-match-divider");
    const updateCompare = (event) => { const rect = viewport.getBoundingClientRect(); setCompare(dialog, (event.clientX - rect.left) / Math.max(1, rect.width) * 100); };
    divider.addEventListener("pointerdown", (event) => { if (event.button !== 0) return; event.preventDefault(); event.stopPropagation(); compareDragging = true; divider.setPointerCapture?.(event.pointerId); updateCompare(event); });
    divider.addEventListener("pointermove", (event) => { if (compareDragging) updateCompare(event); });
    const stopCompare = (event) => { compareDragging = false; divider.releasePointerCapture?.(event.pointerId); };
    divider.addEventListener("pointerup", stopCompare); divider.addEventListener("pointercancel", stopCompare);
    dialog.querySelectorAll("[data-match-zoom]").forEach((button) => button.addEventListener("click", () => setZoom(dialog, button.dataset.matchZoom === "fit" ? 1 : Number(button.dataset.matchZoom))));
    dialog.querySelectorAll(".cs-match-image").forEach((image) => image.addEventListener("load", () => { clampPan(dialog); applyViewportTransform(dialog); }));
    viewport.addEventListener("wheel", (event) => { event.preventDefault(); setZoom(dialog, dialog._matchZoom + (event.deltaY < 0 ? 0.1 : -0.1)); }, { passive: false });
    viewport.addEventListener("pointerdown", (event) => {
        if (dialog._matchZoom <= 1 || event.button !== 0 || event.target === divider) return;
        event.preventDefault(); viewport.setPointerCapture?.(event.pointerId); viewport.classList.add("pan-active");
        dialog._matchPanDrag = { pointerId: event.pointerId, startX: event.clientX, startY: event.clientY, origin: { ...dialog._matchPan } };
    });
    viewport.addEventListener("pointermove", (event) => { const drag = dialog._matchPanDrag; if (!drag || drag.pointerId !== event.pointerId) return; event.preventDefault(); dialog._matchPan = { x: drag.origin.x + event.clientX - drag.startX, y: drag.origin.y + event.clientY - drag.startY }; clampPan(dialog); applyViewportTransform(dialog); });
    const stopPan = (event) => { if (!dialog._matchPanDrag || dialog._matchPanDrag.pointerId !== event.pointerId) return; dialog._matchPanDrag = null; viewport.releasePointerCapture?.(event.pointerId); viewport.classList.remove("pan-active"); };
    viewport.addEventListener("pointerup", stopPan); viewport.addEventListener("pointercancel", stopPan); viewport.addEventListener("lostpointercapture", stopPan);
    resizeObserver = new ResizeObserver(() => { clampPan(dialog); applyViewportTransform(dialog); });
    resizeObserver.observe(dialog.querySelector(".cs-match-preview-wrap"));

    const close = () => { closed = true; clearTimeout(timer); resizeObserver.disconnect(); dialog.close(); dialog.remove(); };
    dialog.querySelector(".cs-match-close").addEventListener("click", close);
    dialog.querySelector(".cs-match-cancel").addEventListener("click", close);
    dialog.addEventListener("cancel", close);
    dialog.querySelector(".cs-match-reset-all").addEventListener("click", () => {
        dialog.querySelector("#cs-match-method").value = "Optimal Transport";
        dialog.querySelector("#cs-match-space").value = "OKLab";
        for (const [name, , defaultValue] of PARAMS) {
            dialog.querySelector(`[data-match-param="${name}"]`).value = String(defaultValue);
            dialog.querySelector(`[data-match-number="${name}"]`).value = String(defaultValue);
        }
        schedulePreview();
    });
    dialog.querySelector(".cs-match-apply").addEventListener("click", () => {
        setNodeValue(node, "method", dialog.querySelector("#cs-match-method").value);
        setNodeValue(node, "color_space", dialog.querySelector("#cs-match-space").value);
        for (const [name] of PARAMS) setNodeValue(node, name, Number(dialog.querySelector(`[data-match-param="${name}"]`).value));
        node.graph?.setDirtyCanvas(true, true);
        close();
    });
    setCompare(dialog, 0); setZoom(dialog, 1); setFrame(0);
}

app.registerExtension({
    name: "CineStyle.ColorMatchPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_ID) return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            original?.apply(this, arguments);
            const button = this.addWidget("button", "Match Preview", "", () => openPreview(this));
            button.name = "Match Preview";
            button.label = "Match Preview";
            button.options = { ...(button.options || {}), serialize: false };
            this.setSize?.([430, this.computeSize?.()[1] || this.size?.[1] || 320]);
        };
    },
    loadedGraphNode(node) {
        if (node?.type !== NODE_ID) return;
        node.setSize?.([node.size?.[0] || 430, node.computeSize?.()[1] || node.size?.[1] || 320]);
    },
});
