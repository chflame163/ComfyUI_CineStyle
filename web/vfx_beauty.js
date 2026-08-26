import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import {
    connectedVideoSource,
    ensureLoaderPreviewSource,
    fetchInfo,
    prepareInputTimeline,
    sourceFrameForLocal,
} from "./video_selector_multi.js";

const STYLE_ID = "cinestyle-vfx-beauty-style";

async function fetchBeautyCachedSource(node) {
    const nodeId = String(node?.id ?? "").trim();
    if (!nodeId) return null;
    const response = await api.fetchApi(`/cinestyle/vfx-beauty-cache?${new URLSearchParams({ node_id: nodeId, t: String(Date.now()) })}`);
    if (response.status === 404) return null;
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Unable to read VFX Beauty preview cache");
    const info = result.info || {};
    return {
        filename: "",
        kind: "video",
        label: String(result.label || "Cached VFX Beauty preview input"),
        url: api.apiURL(result.video_url),
        token: String(result.token || ""),
        info,
        startFrame: 0,
        endFrame: Math.max(0, Number(info.frames || 1) - 1),
        targetFps: Number(info.fps || 24),
        usesProxy: Boolean(result.uses_proxy),
    };
}

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
    ["sat_amount", "Saturation", 0, 300, 100, 0.1, "Scales the final skin saturation."],
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
      .cs-vfx-viewport{position:relative;min-height:260px;overflow:hidden;background:#08090b;border:1px solid #343943;border-radius:6px;cursor:default}.cs-vfx-viewport.pan-ready{cursor:grab}.cs-vfx-viewport.pan-active{cursor:grabbing}
      .cs-vfx-viewport-label{position:absolute;z-index:2;top:8px;left:9px;padding:3px 6px;border-radius:4px;background:#111419c9;color:#cbd2dc;font-size:12px}
      .cs-vfx-viewport img{display:block;width:100%;height:100%;min-height:260px;object-fit:contain;transform-origin:center center;transition:transform .08s linear}
      .cs-vfx-compare-original-clip{position:absolute;inset:0;z-index:1;overflow:hidden;pointer-events:none;clip-path:inset(0 calc(100% - var(--compare-position)) 0 0)}.cs-vfx-compare-original-clip img{position:absolute;inset:0;pointer-events:none}
      .cs-vfx-compare-divider{position:absolute;z-index:3;top:0;bottom:0;left:var(--compare-position);width:2px;transform:translateX(-1px);background:#f2f4f7;box-shadow:0 0 0 1px #11141980;cursor:ew-resize;touch-action:none}.cs-vfx-compare-divider::before{content:"";position:absolute;top:50%;left:50%;width:24px;height:24px;transform:translate(-50%,-50%);border:2px solid #f2f4f7;border-radius:50%;background:#20232a;box-shadow:0 2px 8px #000b}.cs-vfx-compare-divider::after{content:"↔";position:absolute;top:50%;left:50%;transform:translate(-50%,-53%);color:#f2f4f7;font-size:14px;line-height:1}
      .cs-vfx-zoom{display:flex;gap:5px;align-items:center}.cs-vfx-zoom .cs-vfx-button{min-height:27px;padding:4px 8px}
      .cs-vfx-controls{display:grid;grid-template-columns:auto minmax(100px,1fr) 74px auto;align-items:center;gap:8px}
      .cs-vfx-step-buttons{display:flex;gap:5px}.cs-vfx-step-buttons .cs-vfx-button{width:38px;padding-inline:0}
      .cs-vfx-controls input[type=range]{width:100%}.cs-vfx-frame-input{width:74px;min-height:29px;border:1px solid #424956;border-radius:5px;padding:4px 7px;background:#111419;color:#f2f4f7}
      .cs-vfx-param-grid{display:grid;grid-template-columns:repeat(3,minmax(180px,1fr));gap:10px 14px;border-top:1px solid #343943;padding-top:12px}
      .cs-vfx-param{display:grid;grid-template-columns:minmax(92px,auto) 1fr 54px 29px;align-items:center;gap:7px}.cs-vfx-param label{color:#d9dee6}.cs-vfx-param input[type=range]{width:100%}
      .cs-vfx-param output{color:#f7b955;text-align:right;font-variant-numeric:tabular-nums}.cs-vfx-help{grid-column:1/-1;color:#8993a2;font-size:11px;margin-top:-5px}
      .cs-vfx-reset{width:29px;min-height:27px;padding:3px;font-size:15px;line-height:1}
      .cs-vfx-text{min-height:29px;border:1px solid #424956;border-radius:5px;padding:5px 8px;background:#111419;color:#f2f4f7}.cs-vfx-text:focus,.cs-vfx-frame-input:focus{outline:1px solid #55a9f5;border-color:#55a9f5}
      .cs-vfx-input-grid{display:grid;grid-template-columns:minmax(350px,1.15fr) minmax(290px,.85fr);gap:12px}.cs-vfx-setting{display:grid;gap:5px}.cs-vfx-setting-row{display:flex;align-items:center;gap:6px}.cs-vfx-setting-row label{color:#d9dee6;white-space:nowrap}.cs-vfx-setting-row .cs-vfx-text{flex:1;min-width:110px}
      .cs-vfx-setting-help{color:#8993a2;font-size:11px;padding-left:1px}.cs-vfx-swatch{width:34px;height:29px;box-sizing:border-box;border:1px solid #424956;border-radius:5px;padding:2px;background:#20232a;cursor:pointer}.cs-vfx-colour-value{width:63px;color:#f7b955;font-variant-numeric:tabular-nums}
      .cs-vfx-weight-setting{display:grid;gap:6px}.cs-vfx-weight-grid{display:grid;grid-template-columns:repeat(3,minmax(90px,1fr));gap:8px}.cs-vfx-weight-field{display:grid;gap:4px}.cs-vfx-weight-field label{color:#d9dee6}.cs-vfx-weight-field input{width:100%;box-sizing:border-box}.cs-vfx-weight-help{color:#8993a2;font-size:11px}
      .cs-vfx-actions{justify-content:flex-end}.cs-vfx-status{flex:1;min-width:120px}.cs-vfx-error{color:#ff939b}
      @media(max-width:760px){.cs-vfx-view-grid{grid-template-columns:1fr}.cs-vfx-param-grid,.cs-vfx-input-grid{grid-template-columns:1fr}.cs-vfx-controls{grid-template-columns:auto 1fr auto}.cs-vfx-frame-count{grid-column:1/-1}}
    `;
    document.head.append(style);
}

function widget(node, name) { return node.widgets?.find((item) => item.name === name); }
function valueOf(node, name, fallback) { const value = widget(node, name)?.value; return value == null ? fallback : value; }
function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function parseWeights(value) {
    const defaults = [6.0, 0.0, 3.0];
    const parts = String(value ?? "").split(",").map((part) => Number(part.trim()));
    return defaults.map((fallback, index) => Number.isFinite(parts[index]) ? parts[index] : fallback);
}
function clampPan(dialog) {
    const pan = dialog._vfxPan || { x: 0, y: 0 };
    const viewport = dialog.querySelector(".cs-vfx-viewport");
    const image = dialog.querySelector(".cs-vfx-image");
    const zoom = Number(dialog._vfxZoom || 1);
    if (!viewport || !image || zoom <= 1) { dialog._vfxPan = { x: 0, y: 0 }; return; }
    const maxX = Math.max(0, (image.clientWidth * zoom - viewport.clientWidth) * 0.5);
    const maxY = Math.max(0, (image.clientHeight * zoom - viewport.clientHeight) * 0.5);
    dialog._vfxPan = { x: clamp(pan.x, -maxX, maxX), y: clamp(pan.y, -maxY, maxY) };
}
function applyViewportTransform(dialog) {
    const pan = dialog._vfxPan || { x: 0, y: 0 };
    const zoom = Number(dialog._vfxZoom || 1);
    dialog.querySelectorAll(".cs-vfx-image").forEach((image) => { image.style.transform = `translate3d(${pan.x}px, ${pan.y}px, 0) scale(${zoom})`; });
    dialog.querySelectorAll(".cs-vfx-viewport").forEach((viewport) => viewport.classList.toggle("pan-ready", zoom > 1));
}
function setComparePosition(dialog, percent) {
    const value = clamp(Number(percent) || 0, 0, 100);
    dialog._vfxCompare = value;
    const compare = dialog.querySelector(".cs-vfx-compare");
    if (compare) compare.style.setProperty("--compare-position", `${value}%`);
}
function setZoom(dialog, zoom) {
    const value = clamp(Number(zoom) || 1, 0.25, 4);
    dialog._vfxZoom = value;
    if (value <= 1) dialog._vfxPan = { x: 0, y: 0 };
    clampPan(dialog); applyViewportTransform(dialog);
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
    let cachedSource = null;
    try { cachedSource = await fetchBeautyCachedSource(node); } catch (error) { app.canvas?.prompt?.(error.message, ""); return; }
    const upstreamSource = connectedVideoSource(node, ["proxy_video"]) || connectedVideoSource(node, ["image", "images", "video_input"]);
    source = upstreamSource?.loaderId ? upstreamSource : cachedSource;
    if (!source) source = upstreamSource;
    if (!source) { app.canvas?.prompt?.("Run this VFX node once to cache its own connected image/video input.", ""); return; }
    if (source.loaderId && !source.token) {
        try { source = await ensureLoaderPreviewSource(source); } catch (error) { app.canvas?.prompt?.(error.message, ""); return; }
    }
    if (!source.token && !source.filename) { app.canvas?.prompt?.("No previewable input source was found.", ""); return; }
    let info = source.info;
    if (!info) {
        try { info = prepareInputTimeline(source, await fetchInfo(source.filename)); } catch (error) { app.canvas?.prompt?.(error.message, ""); return; }
    }
    const initialWeights = parseWeights(valueOf(node, "weights", "6.0, 0.0, 3.0"));
    dialog = document.createElement("dialog");
    dialog.className = "cs-vfx-dialog";
    dialog._vfxZoom = 1;
    dialog._vfxPan = { x: 0, y: 0 };
    dialog._vfxCompare = 0;
    dialog.innerHTML = `<div class="cs-vfx-shell">
      <div class="cs-vfx-head"><div><h2 class="cs-vfx-title">VFX Preview</h2><div class="cs-vfx-muted cs-vfx-file"></div></div><button class="cs-vfx-button cs-vfx-close" type="button">&times;</button></div>
      <div class="cs-vfx-view-grid"><div class="cs-vfx-viewport"><span class="cs-vfx-viewport-label">Original</span><img class="cs-vfx-image cs-vfx-original" alt="Original frame"></div><div class="cs-vfx-viewport cs-vfx-compare"><span class="cs-vfx-viewport-label">Result</span><img class="cs-vfx-image cs-vfx-result" alt="Result preview"><div class="cs-vfx-compare-original-clip"><img class="cs-vfx-image cs-vfx-compare-original" alt="Original comparison"></div><div class="cs-vfx-compare-divider" title="Drag to compare Original and Result"><span></span></div></div></div>
      <div class="cs-vfx-row"><div class="cs-vfx-zoom"><span class="cs-vfx-muted">Zoom</span><button class="cs-vfx-button" data-zoom="0.5" type="button">50%</button><button class="cs-vfx-button" data-zoom="1" type="button">100%</button><button class="cs-vfx-button" data-zoom="2" type="button">200%</button><button class="cs-vfx-button" data-zoom="fit" type="button">Fit</button><span class="cs-vfx-muted cs-vfx-zoom-value">100%</span></div></div>
      <div class="cs-vfx-controls"><div class="cs-vfx-step-buttons"><button class="cs-vfx-button cs-vfx-prev" type="button">|&lt;</button><button class="cs-vfx-button cs-vfx-next" type="button">&gt;|</button></div><input class="cs-vfx-timeline" type="range" min="0" max="0" step="1" value="0"><input class="cs-vfx-frame-input" type="number" min="0" max="0" step="1" value="0"><span class="cs-vfx-frame-count cs-vfx-muted">0 / 0</span></div>
      <div class="cs-vfx-input-grid">
        <div class="cs-vfx-setting"><div class="cs-vfx-setting-row"><label for="cs-vfx-colour">Colour</label><input id="cs-vfx-colour" class="cs-vfx-text cs-vfx-colour" value="${String(valueOf(node, "colour", "auto")).replace(/"/g, "&quot;")}"><input class="cs-vfx-swatch" type="color" value="#878787" title="Choose an RGB colour" aria-label="Choose an RGB colour"><span class="cs-vfx-colour-value">--</span></div><div class="cs-vfx-setting-help">Use auto for clip colour detection or enter #RRGGBB for a fixed target. Default: auto.</div></div>
        <div class="cs-vfx-weight-setting"><div class="cs-vfx-weight-grid"><div class="cs-vfx-weight-field"><label for="cs-vfx-weight-h">Hue</label><input id="cs-vfx-weight-h" class="cs-vfx-text cs-vfx-weight" data-weight="0" type="number" step="0.01" value="${initialWeights[0]}"><span class="cs-vfx-weight-help">Hue sensitivity</span></div><div class="cs-vfx-weight-field"><label for="cs-vfx-weight-s">Saturation</label><input id="cs-vfx-weight-s" class="cs-vfx-text cs-vfx-weight" data-weight="1" type="number" step="0.01" value="${initialWeights[1]}"><span class="cs-vfx-weight-help">Saturation sensitivity</span></div><div class="cs-vfx-weight-field"><label for="cs-vfx-weight-v">Value</label><input id="cs-vfx-weight-v" class="cs-vfx-text cs-vfx-weight" data-weight="2" type="number" step="0.01" value="${initialWeights[2]}"><span class="cs-vfx-weight-help">Brightness sensitivity</span></div></div><div class="cs-vfx-setting-help">HSV key weights. Default: 6.0, 0.0, 3.0.</div></div>
      </div>
      <div class="cs-vfx-param-grid">${PARAMS.map(([name, label, min, max, defaultValue, step, help]) => `<div class="cs-vfx-param"><label for="cs-vfx-${name}">${label}</label><input id="cs-vfx-${name}" data-param="${name}" type="range" min="${min}" max="${max}" step="${step}" value="${valueOf(node, name, defaultValue)}"><output data-output="${name}">${valueOf(node, name, defaultValue)}</output><button class="cs-vfx-button cs-vfx-reset" data-reset="${name}" type="button" title="Reset ${label} to ${defaultValue}" aria-label="Reset ${label} to default">&#8634;</button><div class="cs-vfx-help">${help} Default: ${defaultValue}.</div></div>`).join("")}</div>
      <div class="cs-vfx-actions"><span class="cs-vfx-status">Loading preview...</span><button class="cs-vfx-button cs-vfx-cancel" type="button">Close</button><button class="cs-vfx-button active cs-vfx-apply" type="button">Apply to Node</button></div>
    </div>`;
    document.body.append(dialog);
    setComparePosition(dialog, 0);
    dialog.querySelector(".cs-vfx-file").textContent = `${source.label || source.filename || "Cached input"} · ${info.frames || 1} frames`;
    const maxFrame = Math.max(0, Number(info.frames || 1) - 1);
    const timeline = dialog.querySelector(".cs-vfx-timeline");
    const frameInput = dialog.querySelector(".cs-vfx-frame-input");
    const colourInput = dialog.querySelector(".cs-vfx-colour");
    const colourSwatch = dialog.querySelector(".cs-vfx-swatch");
    const colourValue = dialog.querySelector(".cs-vfx-colour-value");
    const compareViewport = dialog.querySelector(".cs-vfx-compare");
    const compareDivider = dialog.querySelector(".cs-vfx-compare-divider");
    const normalizeHex = (value) => {
        const text = String(value || "").trim();
        return /^#[0-9a-f]{6}$/i.test(text) ? text.toUpperCase() : null;
    };
    const setColourDisplay = (value, updateInput = false) => {
        const hex = normalizeHex(value);
        if (!hex) { colourValue.textContent = "--"; return null; }
        colourSwatch.value = hex;
        colourValue.textContent = hex;
        if (updateInput) colourInput.value = hex;
        return hex;
    };
    setColourDisplay(colourInput.value) || setColourDisplay("#878787");
    timeline.max = String(maxFrame); frameInput.max = String(maxFrame);
    let frame = 0; let requestSerial = 0; let timer = null; let colourChanged = false;
    let compareDragging = false;
    const setFrame = (next) => { frame = clamp(Math.round(Number(next) || 0), 0, maxFrame); timeline.value = String(frame); frameInput.value = String(frame); dialog.querySelector(".cs-vfx-frame-count").textContent = `${frame} / ${maxFrame}`; schedulePreview(); };
    const weightsValue = () => Array.from(dialog.querySelectorAll("[data-weight]")).map((input) => input.value.trim()).join(", ");
    const payload = () => {
        const result = { node_id: String(node.id), source_kind: source.kind || "video", source_token: source.token || "", video: source.filename || "", frame: source.token ? frame : sourceFrameForLocal(info, frame), local_frame: frame, colour: colourInput.value.trim(), weights: weightsValue() };
        dialog.querySelectorAll("[data-param]").forEach((input) => { result[input.dataset.param] = Number(input.value); });
        return result;
    };
    async function preview() {
        const serial = ++requestSerial; status(`Rendering frame ${frame}...`);
        try {
            const response = await api.fetchApi("/cinestyle/vfx-beauty-preview", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload()) });
            const result = await response.json(); if (!response.ok) throw new Error(result.error || "Preview failed"); if (serial !== requestSerial) return;
            dialog.querySelector(".cs-vfx-original").src = result.original; dialog.querySelector(".cs-vfx-compare-original").src = result.original; dialog.querySelector(".cs-vfx-result").src = result.preview; status(`Frame ${result.frame} · colour ${result.colour}`);
            setColourDisplay(result.colour);
        } catch (error) { if (serial === requestSerial) status(error.message, true); }
    }
    function schedulePreview() { clearTimeout(timer); timer = setTimeout(preview, 100); }
    timeline.addEventListener("input", () => setFrame(timeline.value)); frameInput.addEventListener("change", () => setFrame(frameInput.value));
    dialog.querySelector(".cs-vfx-prev").addEventListener("click", () => setFrame(frame - 1)); dialog.querySelector(".cs-vfx-next").addEventListener("click", () => setFrame(frame + 1));
    dialog.querySelectorAll("[data-param]").forEach((input) => input.addEventListener("input", () => { dialog.querySelector(`[data-output="${input.dataset.param}"]`).value = input.value; schedulePreview(); }));
    dialog.querySelectorAll("[data-reset]").forEach((button) => button.addEventListener("click", () => {
        const definition = PARAMS.find(([name]) => name === button.dataset.reset);
        const input = dialog.querySelector(`[data-param="${button.dataset.reset}"]`);
        if (!definition || !input) return;
        input.value = String(definition[4]);
        dialog.querySelector(`[data-output="${button.dataset.reset}"]`).value = input.value;
        schedulePreview();
    }));
    colourInput.addEventListener("change", () => { colourChanged = true; setColourDisplay(colourInput.value, true); schedulePreview(); });
    colourInput.addEventListener("input", () => { colourChanged = true; setColourDisplay(colourInput.value, true); });
    colourSwatch.addEventListener("input", () => { colourChanged = true; setColourDisplay(colourSwatch.value, true); schedulePreview(); });
    dialog.querySelectorAll("[data-weight]").forEach((input) => input.addEventListener("input", schedulePreview));
    const updateCompareFromEvent = (event) => {
        if (!compareViewport) return;
        const rect = compareViewport.getBoundingClientRect();
        setComparePosition(dialog, ((event.clientX - rect.left) / Math.max(1, rect.width)) * 100);
    };
    compareDivider?.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        event.preventDefault(); event.stopPropagation(); compareDragging = true; compareDivider.setPointerCapture?.(event.pointerId); updateCompareFromEvent(event);
    });
    compareDivider?.addEventListener("pointermove", (event) => { if (compareDragging) updateCompareFromEvent(event); });
    const stopCompareDrag = (event) => { if (!compareDragging) return; compareDragging = false; compareDivider?.releasePointerCapture?.(event.pointerId); };
    compareDivider?.addEventListener("pointerup", stopCompareDrag); compareDivider?.addEventListener("pointercancel", stopCompareDrag);
    dialog.querySelectorAll("[data-zoom]").forEach((button) => button.addEventListener("click", () => setZoom(dialog, button.dataset.zoom === "fit" ? 1 : Number(button.dataset.zoom))));
    dialog.querySelectorAll(".cs-vfx-image").forEach((image) => image.addEventListener("load", () => { clampPan(dialog); applyViewportTransform(dialog); }));
    dialog.querySelectorAll(".cs-vfx-viewport").forEach((viewport) => {
        viewport.addEventListener("wheel", (event) => { event.preventDefault(); setZoom(dialog, dialog._vfxZoom + (event.deltaY < 0 ? 0.1 : -0.1)); }, { passive: false });
        viewport.addEventListener("pointerdown", (event) => {
            if (dialog._vfxZoom <= 1 || event.button !== 0) return;
            event.preventDefault(); viewport.setPointerCapture?.(event.pointerId); viewport.classList.add("pan-active");
            dialog._vfxPanDrag = { pointerId: event.pointerId, startX: event.clientX, startY: event.clientY, origin: { ...dialog._vfxPan } };
        });
        viewport.addEventListener("pointermove", (event) => {
            const drag = dialog._vfxPanDrag; if (!drag || drag.pointerId !== event.pointerId) return;
            dialog._vfxPan = { x: drag.origin.x + event.clientX - drag.startX, y: drag.origin.y + event.clientY - drag.startY };
            clampPan(dialog); applyViewportTransform(dialog);
        });
        const stopPan = (event) => { if (!dialog._vfxPanDrag || dialog._vfxPanDrag.pointerId !== event.pointerId) return; viewport.releasePointerCapture?.(event.pointerId); dialog._vfxPanDrag = null; viewport.classList.remove("pan-active"); };
        viewport.addEventListener("pointerup", stopPan); viewport.addEventListener("pointercancel", stopPan);
    });
    const resizeObserver = new ResizeObserver(() => { clampPan(dialog); applyViewportTransform(dialog); });
    resizeObserver.observe(dialog.querySelector(".cs-vfx-view-grid"));
    const close = () => { clearTimeout(timer); resizeObserver.disconnect(); dialog.close(); dialog.remove(); }; dialog.querySelector(".cs-vfx-close").addEventListener("click", close); dialog.querySelector(".cs-vfx-cancel").addEventListener("click", close); dialog.addEventListener("cancel", close);
    dialog.querySelector(".cs-vfx-apply").addEventListener("click", () => {
        const setValue = (name, value) => {
            const target = widget(node, name); if (!target) return;
            const index = node.widgets?.indexOf(target) ?? -1;
            target.value = value; target.callback?.(value); target.value = value;
            if (index >= 0 && Array.isArray(node.widgets_values)) node.widgets_values[index] = value;
        };
        if (colourChanged) setValue("colour", colourInput.value.trim());
        setValue("weights", weightsValue());
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
