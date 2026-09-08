import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_ID = "CS_MatAnyone2";
const STYLE_ID = "cinestyle-matanyone2-style";

function widget(node, name) { return node.widgets?.find((item) => item.name === name); }
function valueOf(node, name, fallback) { const value = widget(node, name)?.value; return value == null ? fallback : value; }
function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function parseAnchors(value, fallback = [0]) {
    let parsed = value;
    if (typeof value === "string") {
        try { parsed = JSON.parse(value); }
        catch { parsed = value.replace(/[;\s]+/g, ",").split(",").filter(Boolean); }
    }
    if (!Array.isArray(parsed)) parsed = [parsed];
    const result = [...new Set(parsed.map((item) => Number(item)).filter((item) => Number.isInteger(item) && item >= 0))].sort((a, b) => a - b);
    return result.length ? result : [...fallback];
}
function setWidgetValue(node, name, value) {
    const target = widget(node, name);
    if (!target) return;
    target.value = value;
    target.callback?.(value);
    target.value = value;
    const index = node.widgets?.indexOf(target) ?? -1;
    if (index >= 0 && Array.isArray(node.widgets_values)) node.widgets_values[index] = value;
}
function numberValue(node, name, fallback) { return Number(valueOf(node, name, fallback)); }
function escapeHtml(value) { return String(value).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch])); }

function addStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .cs-ma2-dialog{width:min(1120px,96vw);max-width:none;max-height:95vh;overflow:auto;padding:0;border:1px solid #353a43;border-radius:8px;background:#17191e;color:#e8ebef;box-shadow:0 22px 80px #000b}
      .cs-ma2-dialog::backdrop{background:#050609c4}
      .cs-ma2-shell{display:grid;gap:12px;padding:16px;font:13px/1.35 system-ui,sans-serif}
      .cs-ma2-head,.cs-ma2-row,.cs-ma2-actions{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.cs-ma2-head{justify-content:space-between}.cs-ma2-anchor-row{width:100%}.cs-ma2-anchor-row .cs-ma2-anchor-list{flex:1}.cs-ma2-anchor-actions{display:flex;align-items:center;gap:8px;margin-left:auto}
      .cs-ma2-title{margin:0;font-size:17px}.cs-ma2-muted,.cs-ma2-status{color:#9da5b4}.cs-ma2-error{color:#ff939b}
      .cs-ma2-button{min-height:30px;border:1px solid #454c57;border-radius:5px;padding:5px 9px;background:#22252c;color:#f2f4f7;cursor:pointer}.cs-ma2-button:hover{border-color:#79aee0}.cs-ma2-button.active{border-color:#74b8f0;background:#2879b8}.cs-ma2-close{font-size:18px;padding:2px 9px}
      .cs-ma2-stage{position:relative;display:flex;justify-content:center;min-height:320px}.cs-ma2-image{display:block;width:min(100%,1040px);height:clamp(320px,53vh,620px);object-fit:contain;border:1px solid #343943;border-radius:6px;background:#08090b}.cs-ma2-cache-loading{position:absolute;z-index:3;inset:0;display:flex;align-items:center;justify-content:center;padding:18px;background:#08090be8;color:#dce7f3;text-align:center}.cs-ma2-cache-loading[hidden]{display:none}
      .cs-ma2-timeline{display:grid;grid-template-columns:auto minmax(160px,1fr) 78px auto;align-items:center;gap:8px}.cs-ma2-number,.cs-ma2-text{min-height:29px;box-sizing:border-box;border:1px solid #454c57;border-radius:4px;padding:4px 7px;background:#101216;color:#f2f4f7;font-variant-numeric:tabular-nums}.cs-ma2-frame{width:78px}.cs-ma2-track{position:relative;display:flex;align-items:center}.cs-ma2-track input{width:100%;margin:0}.cs-ma2-markers{position:absolute;left:0;right:0;top:-2px;height:6px;pointer-events:none}.cs-ma2-marker{position:absolute;top:0;width:2px;height:9px;transform:translateX(-1px);background:#f0ad4e}.cs-ma2-marker.anchor{height:14px;top:-3px;background:#62c5ef;box-shadow:0 0 0 1px #101216}.cs-ma2-marker.current{height:18px;top:-5px;background:#f4f5f7;z-index:2}
      .cs-ma2-anchor-list{display:flex;gap:6px;flex-wrap:wrap;min-height:28px;align-items:center}.cs-ma2-anchor-chip{display:inline-flex;align-items:center;gap:4px;padding:3px 7px;border:1px solid #3f8db0;border-radius:12px;background:#173b4b;color:#d9f4ff;cursor:pointer}.cs-ma2-anchor-chip.selected{background:#2879b8;border-color:#8edaff}.cs-ma2-anchor-chip button{border:0;background:transparent;color:inherit;cursor:pointer;padding:0 2px}
      .cs-ma2-system-row,.cs-ma2-controls{display:grid;grid-template-columns:repeat(4,minmax(160px,1fr));gap:10px}.cs-ma2-system-row{align-items:center}.cs-ma2-system-select{display:grid;grid-template-columns:max-content 152px;align-items:center;justify-content:start;gap:8px}.cs-ma2-system-select label{color:#d8dde5;white-space:nowrap}.cs-ma2-system-select select{width:152px}.cs-ma2-system-checks{grid-column:3/5;display:flex;align-items:center;justify-content:flex-end;gap:18px;white-space:nowrap}.cs-ma2-controls{border-top:1px solid #343943;padding-top:12px}.cs-ma2-control{display:grid;grid-template-columns:minmax(0,1fr) 76px;align-items:center;gap:8px}.cs-ma2-control label{color:#d8dde5;text-align:right;white-space:nowrap}.cs-ma2-control input[type=range]{width:100%}.cs-ma2-control input[type=number],.cs-ma2-control select{width:100%}.cs-ma2-control.wide{grid-column:span 2}.cs-ma2-checkbox{display:flex;align-items:center;gap:7px}.cs-ma2-actions{justify-content:flex-end}.cs-ma2-status{flex:1;min-width:180px}
      @media(max-width:850px){.cs-ma2-system-row,.cs-ma2-controls{grid-template-columns:repeat(2,minmax(150px,1fr))}.cs-ma2-system-checks{grid-column:1/-1}.cs-ma2-control.wide{grid-column:span 2}}
      @media(max-width:560px){.cs-ma2-timeline{grid-template-columns:auto 1fr 76px}.cs-ma2-frame-count{grid-column:1/-1}.cs-ma2-system-row,.cs-ma2-controls{grid-template-columns:1fr}.cs-ma2-system-checks{grid-column:auto;justify-content:flex-start;flex-wrap:wrap}.cs-ma2-control.wide{grid-column:auto}}
    `;
    document.head.append(style);
}

async function fetchCache(node) {
    const response = await api.fetchApi(`/cinestyle/matanyone2-cache?${new URLSearchParams({ node_id: String(node.id), t: String(Date.now()) })}`);
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.error || "Run the workflow once to prepare Matte Preview cache.");
    return result.info || {};
}

async function fetchFrame(node, frame) {
    const response = await api.fetchApi(`/cinestyle/matanyone2-frame?${new URLSearchParams({ node_id: String(node.id), frame: String(frame), t: String(Date.now()) })}`);
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.error || "Unable to load the selected frame.");
    return result;
}

async function analyse(node, params) {
    const response = await api.fetchApi("/cinestyle/matanyone2-analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ node_id: String(node.id), ...params }),
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.error || "Unable to analyse coarse masks.");
    return result;
}

function controlMarkup(node) {
    const controls = [
        ["max_megapixels", "Max inference MPixels", 0.1, 64, 0.1, numberValue(node, "max_megapixels", 2.1), "number"],
        ["anchor_min_spacing", "Anchor min spacing", 1, 100000, 1, numberValue(node, "anchor_min_spacing", 48), "number"],
        ["anchor_hysteresis", "Anchor hysteresis", 1, 1000, 1, numberValue(node, "anchor_hysteresis", 3), "number"],
        ["anchor_limit", "Anchor limit", 1, 128, 1, numberValue(node, "anchor_limit", 12), "number"],
        ["overlap", "Overlap frames", 0, 10000, 1, numberValue(node, "overlap", 12), "number"],
        ["analysis_stride", "Analysis stride", 1, 120, 1, numberValue(node, "analysis_stride", 1), "number"],
        ["anchor_sensitivity", "Anchor sensitivity", 0, 1, 0.01, numberValue(node, "anchor_sensitivity", 0.35), "number"],
        ["mask_threshold", "Mask threshold", 0, 1, 0.01, numberValue(node, "mask_threshold", 0.5), "number"],
        ["seed_morphology", "Seed morphology", -64, 64, 1, numberValue(node, "seed_morphology", 0), "number"],
        ["warmup", "Warmup iterations", 1, 50, 1, numberValue(node, "warmup", 10), "number"],
        ["memory_interval", "Memory interval", 1, 100, 1, numberValue(node, "memory_interval", 5), "number"],
        ["memory_frames", "Memory frames", 2, 50, 1, numberValue(node, "memory_frames", 5), "number"],
    ];
    return controls.map(([name, label, min, max, step, value]) => `<div class="cs-ma2-control"><label for="cs-ma2-${name}">${label}</label><input class="cs-ma2-number" id="cs-ma2-${name}" data-param="${name}" type="number" min="${min}" max="${max}" step="${step}" value="${value}"></div>`).join("");
}

function deviceMarkup(node) {
    const options = widget(node, "device")?.options;
    const values = Array.isArray(options?.values) ? options.values : (Array.isArray(options) ? options : ["auto", "cpu"]);
    const selected = String(valueOf(node, "device", "auto"));
    return values.map((value) => `<option value="${escapeHtml(value)}"${String(value) === selected ? " selected" : ""}>${escapeHtml(value)}</option>`).join("");
}

function modelMarkup(node) {
    const options = widget(node, "model_file")?.options;
    const values = Array.isArray(options?.values) ? options.values : (Array.isArray(options) ? options : ["matanyone2.pth"]);
    const selected = String(valueOf(node, "model_file", "matanyone2.pth"));
    return values.map((value) => `<option value="${escapeHtml(value)}"${String(value) === selected ? " selected" : ""}>${escapeHtml(value)}</option>`).join("");
}

async function openPreview(node) {
    addStyles();
    const dialog = document.createElement("dialog");
    dialog.className = "cs-ma2-dialog";
    dialog.innerHTML = `<div class="cs-ma2-shell"><div class="cs-ma2-head"><div><h2 class="cs-ma2-title">Matte Preview</h2><div class="cs-ma2-muted cs-ma2-info">Loading cached image and mask...</div></div><button class="cs-ma2-button cs-ma2-close" type="button">&times;</button></div><div class="cs-ma2-stage"><img class="cs-ma2-image" alt="Image with coarse mask overlay"><div class="cs-ma2-cache-loading" role="status">Preparing preview...</div></div><div class="cs-ma2-timeline"><div class="cs-ma2-row"><button class="cs-ma2-button cs-ma2-prev" type="button">|&lt;</button><button class="cs-ma2-button cs-ma2-next" type="button">&gt;|</button></div><div class="cs-ma2-track"><input class="cs-ma2-slider" type="range" min="0" max="0" step="1" value="0"><div class="cs-ma2-markers"></div></div><input class="cs-ma2-number cs-ma2-frame" type="number" min="0" max="0" step="1" value="0"><span class="cs-ma2-muted cs-ma2-frame-count">0 / 0</span></div><div class="cs-ma2-row cs-ma2-anchor-row"><span class="cs-ma2-muted">Anchors</span><div class="cs-ma2-anchor-list"></div><div class="cs-ma2-anchor-actions"><button class="cs-ma2-button cs-ma2-add" type="button">Add current</button><button class="cs-ma2-button cs-ma2-remove" type="button">Remove current</button></div></div><div class="cs-ma2-system-row"><div class="cs-ma2-system-select"><label for="cs-ma2-model-file">Model File</label><select id="cs-ma2-model-file" data-param="model_file">${modelMarkup(node)}</select></div><div class="cs-ma2-system-select"><label for="cs-ma2-device">Device</label><select id="cs-ma2-device" data-param="device">${deviceMarkup(node)}</select></div><div class="cs-ma2-system-checks"><label class="cs-ma2-checkbox"><input type="checkbox" data-param="use_long_term"${valueOf(node, "use_long_term", false) ? " checked" : ""}>Use long-term memory</label><label class="cs-ma2-checkbox"><input type="checkbox" data-param="auto_unload_model"${valueOf(node, "auto_unload_model", true) ? " checked" : ""}>Auto unload mode</label></div></div><section class="cs-ma2-controls">${controlMarkup(node)}</section><div class="cs-ma2-actions"><span class="cs-ma2-status">Loading...</span><button class="cs-ma2-button cs-ma2-reanalyse" type="button">Re-analyse</button><button class="cs-ma2-button cs-ma2-cancel" type="button">Close</button><button class="cs-ma2-button active cs-ma2-apply" type="button">Apply to Node</button></div></div>`;
    dialog.addEventListener("close", () => dialog.remove(), { once: true });
    document.body.append(dialog);
    const loading = dialog.querySelector(".cs-ma2-cache-loading");
    const image = dialog.querySelector(".cs-ma2-image");
    const slider = dialog.querySelector(".cs-ma2-slider");
    const frameInput = dialog.querySelector(".cs-ma2-frame");
    const infoLabel = dialog.querySelector(".cs-ma2-info");
    const frameCount = dialog.querySelector(".cs-ma2-frame-count");
    const markers = dialog.querySelector(".cs-ma2-markers");
    const anchorList = dialog.querySelector(".cs-ma2-anchor-list");
    const status = dialog.querySelector(".cs-ma2-status");
    let info = null;
    let frame = 0;
    let anchors = parseAnchors(valueOf(node, "anchor_frames", "0"));
    let candidates = [];
    let initialAnalysisDone = Boolean(node.properties?.csMatAnyone2PreviewInitialized);
    let requestSerial = 0;
    const maxFrame = () => Math.max(0, Number(info?.frames || 1) - 1);
    const param = (name, fallback) => {
        const element = dialog.querySelector(`[data-param="${name}"]`);
        if (!element) return fallback;
        if (element.type === "checkbox") return Boolean(element.checked);
        const value = Number(element.value);
        return Number.isFinite(value) ? value : fallback;
    };
    const analysisParams = () => ({
        anchor_min_spacing: Math.max(1, Math.round(param("anchor_min_spacing", 48))),
        anchor_hysteresis: Math.max(1, Math.round(param("anchor_hysteresis", 3))),
        anchor_limit: Math.max(1, Math.round(param("anchor_limit", 12))),
        analysis_stride: Math.max(1, Math.round(param("analysis_stride", 1))),
        anchor_sensitivity: clamp(param("anchor_sensitivity", 0.35), 0, 1),
        mask_threshold: clamp(param("mask_threshold", 0.5), 0, 1),
    });
    function renderMarkers() {
        markers.innerHTML = "";
        const denominator = Math.max(1, maxFrame());
        for (const item of candidates) {
            const marker = document.createElement("span"); marker.className = "cs-ma2-marker"; marker.style.left = `${Number(item.frame) / denominator * 100}%`; marker.title = `Candidate frame ${item.frame}`; markers.append(marker);
        }
        for (const value of anchors) {
            const marker = document.createElement("span"); marker.className = "cs-ma2-marker anchor"; marker.style.left = `${value / denominator * 100}%`; marker.title = `Anchor frame ${value}`; markers.append(marker);
        }
        const current = document.createElement("span"); current.className = "cs-ma2-marker current"; current.style.left = `${frame / denominator * 100}%`; current.title = `Frame ${frame}`; markers.append(current);
        anchorList.innerHTML = anchors.map((value) => `<span class="cs-ma2-anchor-chip${value === frame ? " selected" : ""}" data-anchor="${value}">${value}<button type="button" data-remove-anchor="${value}" title="Remove anchor">&times;</button></span>`).join("");
    }
    function setFrame(value) {
        frame = clamp(Math.round(Number(value) || 0), 0, maxFrame());
        slider.value = String(frame); frameInput.value = String(frame); renderMarkers();
        const serial = ++requestSerial;
        fetchFrame(node, frame).then((result) => { if (serial !== requestSerial) return; image.src = result.image; }).catch((error) => { status.textContent = error.message; status.classList.add("cs-ma2-error"); });
    }
    function addAnchor(value) { const candidate = clamp(Math.round(Number(value) || 0), 0, maxFrame()); if (!anchors.includes(candidate)) anchors = [...anchors, candidate].sort((a, b) => a - b); frame = candidate; renderMarkers(); }
    function validateAnchors() {
        if (!anchors.length) anchors = [0];
        const adaptNumber = (name, minimum, maximum, fallback, integer = false) => {
            const element = dialog.querySelector(`[data-param="${name}"]`);
            if (!element) return fallback;
            let value = Number(element.value);
            if (!Number.isFinite(value)) value = fallback;
            value = clamp(integer ? Math.round(value) : value, minimum, maximum);
            element.value = String(value);
            return value;
        };
        const minimum = adaptNumber("anchor_min_spacing", 1, 100000, 48, true);
        adaptNumber("anchor_hysteresis", 1, Math.max(1, Math.floor(minimum / 2)), 3, true);
        adaptNumber("anchor_limit", 1, 128, 12, true);
        let overlap = adaptNumber("overlap", 0, 10000, 12, true);
        if (anchors.length > 1) {
            const shortest = Math.min(...anchors.slice(1).map((value, index) => value - anchors[index]));
            overlap = Math.min(overlap, Math.max(0, Math.floor((shortest - 1) / 2)));
            dialog.querySelector('[data-param="overlap"]').value = String(overlap);
        }
        adaptNumber("max_megapixels", 0.1, 64, 2.1);
        adaptNumber("analysis_stride", 1, 120, 1, true);
        adaptNumber("anchor_sensitivity", 0, 1, 0.35);
        adaptNumber("mask_threshold", 0, 1, 0.5);
        adaptNumber("seed_morphology", -64, 64, 0, true);
        adaptNumber("warmup", 1, 50, 10, true);
        adaptNumber("memory_interval", 1, 100, 5, true);
        adaptNumber("memory_frames", 2, 50, 5, true);
        return "";
    }
    async function runAnalysis(initial = false) {
        status.classList.remove("cs-ma2-error"); status.textContent = "Analysing coarse masks...";
        const result = await analyse(node, analysisParams());
        candidates = result.candidates || [];
        if (initial) {
            const primary = Number(result.primary_anchor ?? result.anchors?.[0] ?? 0);
            const primaryFrame = clamp(Number.isFinite(primary) ? primary : 0, 0, maxFrame());
            const existing = anchors.length === 1 && anchors[0] === 0 ? [] : anchors;
            anchors = [...new Set([...existing, primaryFrame])].sort((a, b) => a - b);
            initialAnalysisDone = true;
            node.properties = node.properties || {};
            node.properties.csMatAnyone2PreviewInitialized = true;
        }
        anchors = [...new Set(anchors.filter((value) => value >= 0 && value <= maxFrame()))].sort((a, b) => a - b);
        if (!anchors.length) anchors = [0];
        renderMarkers(); status.textContent = `候选 ${candidates.length} 个 · Anchor ${anchors.length} 个`;
    }
    dialog.querySelector(".cs-ma2-close").addEventListener("click", () => { dialog.close(); dialog.remove(); });
    dialog.querySelector(".cs-ma2-cancel").addEventListener("click", () => { dialog.close(); dialog.remove(); });
    slider.addEventListener("input", () => setFrame(slider.value));
    frameInput.addEventListener("change", () => setFrame(frameInput.value));
    dialog.querySelector(".cs-ma2-prev").addEventListener("click", () => setFrame(frame - 1));
    dialog.querySelector(".cs-ma2-next").addEventListener("click", () => setFrame(frame + 1));
    dialog.querySelector(".cs-ma2-add").addEventListener("click", () => { addAnchor(frame); status.textContent = `已添加 Anchor ${frame}`; });
    dialog.querySelector(".cs-ma2-remove").addEventListener("click", () => { anchors = anchors.filter((value) => value !== frame); renderMarkers(); status.textContent = anchors.length ? `已移除 Anchor ${frame}` : "Anchor 已全部删除，Apply 时将恢复第 0 帧。"; });
    anchorList.addEventListener("click", (event) => { const remove = event.target.closest("[data-remove-anchor]"); if (remove) { event.stopPropagation(); anchors = anchors.filter((value) => value !== Number(remove.dataset.removeAnchor)); renderMarkers(); status.textContent = anchors.length ? `已移除 Anchor ${remove.dataset.removeAnchor}` : "Anchor 已全部删除，Apply 时将恢复第 0 帧。"; return; } const chip = event.target.closest("[data-anchor]"); if (chip) setFrame(Number(chip.dataset.anchor)); });
    dialog.querySelector(".cs-ma2-reanalyse").addEventListener("click", () => runAnalysis(false).catch((error) => { status.textContent = error.message; status.classList.add("cs-ma2-error"); }));
    dialog.querySelector(".cs-ma2-apply").addEventListener("click", () => { const error = validateAnchors(); if (error) { status.textContent = error; status.classList.add("cs-ma2-error"); window.alert(error); return; } setWidgetValue(node, "anchor_frames", JSON.stringify(anchors)); ["max_megapixels", "anchor_min_spacing", "anchor_hysteresis", "anchor_limit", "overlap", "analysis_stride", "anchor_sensitivity", "mask_threshold", "seed_morphology", "warmup", "memory_interval", "memory_frames", "device", "model_file"].forEach((name) => { const element = dialog.querySelector(`[data-param="${name}"]`); if (!element) return; setWidgetValue(node, name, element.type === "number" ? Number(element.value) : element.value); }); setWidgetValue(node, "use_long_term", Boolean(dialog.querySelector('[data-param="use_long_term"]')?.checked)); setWidgetValue(node, "auto_unload_model", Boolean(dialog.querySelector('[data-param="auto_unload_model"]')?.checked)); node.properties = node.properties || {}; node.properties.csMatAnyone2PreviewInitialized = true; node.graph?.setDirtyCanvas?.(true, true); dialog.close(); dialog.remove(); });
    dialog.showModal();
    try {
        info = await fetchCache(node);
        anchors = [...new Set(anchors.map((value) => clamp(Math.round(Number(value) || 0), 0, maxFrame())))].sort((a, b) => a - b);
        if (!anchors.length) anchors = [0];
        slider.max = String(maxFrame()); frameInput.max = String(maxFrame()); frameCount.textContent = `${info.frames || 0} frames · ${info.width || 0}×${info.height || 0}`; infoLabel.textContent = `Cached IMAGE + MASK · ${info.fps ? Number(info.fps).toFixed(3) : "24.000"} fps`;
        loading.hidden = true; setFrame(Math.min(anchors[0] || 0, maxFrame())); renderMarkers();
        if (!initialAnalysisDone) await runAnalysis(true);
        else { renderMarkers(); status.textContent = `已读取 ${anchors.length} 个 Anchor。`; }
    } catch (error) {
        loading.textContent = error.message; status.textContent = error.message; status.classList.add("cs-ma2-error");
    }
}

app.registerExtension({
    name: "CineStyle.MatAnyone2",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            original?.apply(this, arguments);
            const button = this.addWidget("button", "Matte Preview", "", () => openPreview(this).catch((error) => app.canvas?.prompt?.(error.message, "")));
            button.name = "Matte Preview"; button.label = "Matte Preview"; button.options = { ...(button.options || {}), serialize: false };
            this.setSize?.([420, this.computeSize?.()[1] || 420]);
        };
        const originalConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info) {
            originalConfigure?.call(this, info);
            this.setSize?.([this.size?.[0] || 420, this.computeSize?.()[1] || this.size?.[1] || 420]);
        };
    },
    loadedGraphNode(node) {
        if (node?.type !== NODE_ID) return;
        node.setSize?.([node.size?.[0] || 420, node.computeSize?.()[1] || node.size?.[1] || 420]);
    },
});
