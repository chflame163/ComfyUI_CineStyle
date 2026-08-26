import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const STYLE_ID = "cinestyle-video-selector-style";

function widget(node, name) { return node.widgets?.find((item) => item.name === name); }
function setWidgetValue(node, name, value) { const target = widget(node, name); if (target) { target.value = value; target.callback?.(value); } }
function removeObsoleteInputs(node, names = []) { for (const name of names) { const index = node.inputs?.findIndex((input) => input.name === name) ?? -1; if (index >= 0) node.removeInput?.(index); } }
function removeObsoleteWidgets(node, names = []) { for (const name of names) { const index = node.widgets?.findIndex((item) => item.name === name) ?? -1; if (index >= 0) node.widgets.splice(index, 1); } }
function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function roundLikePython(value) { const lower = Math.floor(value); const fraction = value - lower; if (fraction < 0.5) return lower; if (fraction > 0.5) return lower + 1; return lower % 2 === 0 ? lower : lower + 1; }
function parseJson(value, fallback) { try { const parsed = JSON.parse(String(value || "")); return parsed ?? fallback; } catch { return fallback; } }
function videoUrl(filename) { const params = new URLSearchParams({ filename, t: String(Date.now()) }); return api.apiURL(`/cinestyle/video-source?${params.toString()}`); }
function splitAnnotatedFilename(value) { const text = String(value || "").trim(); const match = text.match(/^(.*)\s+\[(input|output|temp)\]$/i); return match ? { filename: match[1], type: match[2].toLowerCase() } : { filename: text, type: "input" }; }
function imageUrl(filename) { const source = splitAnnotatedFilename(filename); const params = new URLSearchParams({ filename: source.filename, type: source.type, subfolder: "", t: String(Date.now()) }); return api.apiURL(`/view?${params.toString()}`); }
function isImageFilename(value) { return /\.(png|jpe?g|webp|bmp|tiff?|gif|avif)(?:\s*\[[^\]]+\])?$/i.test(String(value || "").trim()); }
async function fetchImageInfo(filename) {
    const url = imageUrl(filename);
    const image = new Image();
    image.src = url;
    await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = () => reject(new Error(`Image file not found: ${filename}`)); });
    return { width: image.naturalWidth, height: image.naturalHeight, fps: 1, frames: 1, duration: 1, audio_format: null };
}
async function fetchInfo(filename) {
    if (isImageFilename(filename)) return fetchImageInfo(filename);
    const response = await api.fetchApi(`/cinestyle/video-info?${new URLSearchParams({ filename })}`); if (!response.ok) throw new Error(await response.text()); return response.json();
}
async function fetchCachedSource(node) {
    const nodeId = String(node?.id ?? "").trim(); if (!nodeId) return null;
    const response = await api.fetchApi(`/cinestyle/video-selector-cache?${new URLSearchParams({ node_id: nodeId, t: String(Date.now()) })}`);
    if (response.status === 404) return null;
    const result = await response.json(); if (!response.ok) throw new Error(result.error || "Unable to read cached Selector input");
    const info = result.info || {};
    return { filename: "", label: String(result.label || "Cached input from the last workflow run"), url: api.apiURL(result.video_url), token: String(result.token || ""), info, startFrame: 0, endFrame: Math.max(0, Number(info.frames || 1) - 1), targetFps: Number(info.fps || 24) };
}
function graphNode(graph, id) { if (id == null || !graph) return null; const direct = graph.getNodeById?.(id); if (direct) return direct; const nodes = graph._nodes || graph.nodes; return Array.isArray(nodes) ? nodes.find((item) => String(item?.id) === String(id)) || null : graph._nodes_by_id?.[id] || null; }
function graphLink(graph, candidate) { if (candidate == null) return null; if (typeof candidate === "object") { if (candidate.origin_id != null || candidate.originId != null) return candidate; if (candidate.link && typeof candidate.link === "object") return candidate.link; } return graph?.links?.[candidate] || graph?._links?.[candidate] || null; }
function originFromConnection(graph, candidate) { if (!candidate) return null; if (typeof candidate === "object" && (candidate.origin_id != null || candidate.originId != null)) return graphNode(graph, candidate.origin_id ?? candidate.originId); if (candidate?.type || candidate?.comfyClass) return candidate; const link = graphLink(graph, candidate); return link ? graphNode(graph, link.origin_id ?? link.originId ?? link.origin) : null; }
function nodeTypeName(node) { return String(node?.type || node?.comfyClass || node?.constructor?.type || ""); }
function isAnyRerouter(node) { return /layerutility\s*:\s*any\s+rerouter/i.test(nodeTypeName(node)) || /any\s+rerouter/i.test(String(node?.title || "")); }
function isCSLoadVideo(node) { const type = nodeTypeName(node); return type === "CS_Load_Video" || type.endsWith(".CS_Load_Video") || type.endsWith("::CS_Load_Video"); }
function isLoadImage(node) { return /(^|[.:_])load[_-]?image([.:_]|$)/i.test(nodeTypeName(node)) || /image[_-]?loader/i.test(nodeTypeName(node)); }
function sourceFilename(node) {
    const names = isCSLoadVideo(node) ? ["video"] : ["image", "file", "video", "filename", "image_file", "image_path", "input_image", "load_image", "file_path", "filepath", "video_path", "video_file", "video_file_path", "input_path", "path"];
    for (const name of names) { const value = String(widget(node, name)?.value || "").trim(); if (value) return value; }
    return "";
}
function mediaInput(input) { return /video|image|frame|media|source|stream|movie/.test(`${input?.name || ""} ${input?.type || ""}`.toLowerCase()); }
function connectedOrigin(node, inputName) {
    const index = node.inputs?.findIndex((item) => item.name === inputName) ?? -1; if (index < 0) return null;
    const input = node.inputs?.[index]; const graph = node.graph || app.graph; const candidates = [];
    const call = (method, argument) => { try { return typeof method === "function" ? method.call(node, argument) : null; } catch { return null; } };
    candidates.push(call(node.getInputNode, index), call(node.getInputNode, inputName), call(node.getInputLink, index), call(node.getInputLink, inputName), input?.link); if (Array.isArray(input?.links)) candidates.push(...input.links);
    for (const candidate of candidates) { const origin = originFromConnection(graph, candidate); if (origin) return origin; }
    return null;
}
function connectedMediaOrigins(node) { return (node?.inputs || []).filter(mediaInput).map((input) => connectedOrigin(node, input.name)).filter(Boolean); }
function sourceFromOrigin(origin, visited = new Set()) {
    if (!origin) return null; const identity = origin.id != null ? String(origin.id) : `${nodeTypeName(origin)}:${visited.size}`; if (visited.has(identity)) return null; visited.add(identity);
    const filename = sourceFilename(origin);
    if (isCSLoadVideo(origin) || /\.(mp4|mov|mkv|avi|webm|m4v|mpg|mpeg|wmv|flv)(?:\s*\[[^\]]*\])?$/i.test(filename)) {
        if (!filename) return null;
        return {
            filename,
            kind: "video",
            isCSLoad: isCSLoadVideo(origin),
            loaderId: isCSLoadVideo(origin) ? String(origin.id ?? "") : "",
            startFrame: Math.max(0, Number(widget(origin, "start_frame")?.value ?? 0)),
            endFrame: Number(widget(origin, "end_frame")?.value ?? -1),
            targetFps: Number(widget(origin, "fps")?.value ?? 0),
            outputWidth: Number(widget(origin, "width")?.value ?? 0),
            outputHeight: Number(widget(origin, "height")?.value ?? 0),
            multiple: Number(widget(origin, "multiple")?.value ?? 1),
        };
    }
    if (isLoadImage(origin) || isImageFilename(filename)) {
        if (!filename) return null;
        return { filename, kind: "image", startFrame: 0, endFrame: 0, targetFps: 1 };
    }
    if (isAnyRerouter(origin)) {
        const input = origin.inputs?.[0];
        const upstream = input ? connectedOrigin(origin, input.name) : null;
        return sourceFromOrigin(upstream, visited);
    }
    for (const upstream of connectedMediaOrigins(origin)) { const source = sourceFromOrigin(upstream, visited); if (source) return source; }
    return null;
}
function connectedVideoSource(node, inputNames = ["images", "video_input"]) {
    const origins = inputNames.map((name) => connectedOrigin(node, name)).filter(Boolean);
    for (const origin of origins) { const source = sourceFromOrigin(origin); if (source) return source; }
    return null;
}

function loaderPreviewPayload(source) {
    return {
        loader_id: String(source?.loaderId || ""),
        video: String(source?.filename || ""),
        start_frame: Number(source?.startFrame ?? 0),
        end_frame: Number(source?.endFrame ?? -1),
        width: Number(source?.outputWidth ?? 0),
        height: Number(source?.outputHeight ?? 0),
        fps: Number(source?.targetFps ?? 0),
        multiple: Number(source?.multiple ?? 32),
    };
}

function loaderPreviewSource(result, source) {
    const info = result?.info || {};
    return {
        ...source,
        filename: "",
        kind: "video",
        token: String(result?.token || ""),
        url: api.apiURL(String(result?.video_url || "")),
        info,
        label: "Shared preview from CS Load Video",
        sharedLoaderCache: true,
        loaderSignature: String(result?.signature || ""),
        startFrame: 0,
        endFrame: Math.max(0, Number(info.frames || info.loaded_frame_count || 1) - 1),
        targetFps: Number(info.fps || info.loaded_fps || 24),
    };
}

async function ensureLoaderPreviewSource(source, { wait = true, timeoutMs = 300000 } = {}) {
    if (!source?.isCSLoad || !source.loaderId || !source.filename) return source;
    const payload = loaderPreviewPayload(source);
    const response = await api.fetchApi("/cinestyle/loader-preview-cache", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    let result = await response.json().catch(() => ({}));
    if (!response.ok || result.status === "failed") throw new Error(result.error || "Unable to prepare CS Load Video preview cache");
    if (result.status === "ready") return loaderPreviewSource(result, source);
    if (!wait) return source;
    const started = Date.now();
    const signature = String(result.signature || "");
    while (Date.now() - started < timeoutMs) {
        await new Promise((resolve) => window.setTimeout(resolve, 250));
        const params = new URLSearchParams({ loader_id: String(source.loaderId), signature });
        const progressResponse = await api.fetchApi(`/cinestyle/loader-preview-cache-progress?${params}`);
        result = await progressResponse.json().catch(() => ({}));
        if (result.status === "ready") return loaderPreviewSource(result, source);
        if (result.status === "failed") throw new Error(result.error || "CS Load Video preview cache failed");
    }
    throw new Error("Timed out while preparing CS Load Video preview cache.");
}
function prepareInputTimeline(source, sourceInfo) {
    const sourceFrames = Math.max(1, Number(sourceInfo.frames || 1)); const sourceFps = Math.max(0.001, Number(sourceInfo.fps || 24));
    const startFrame = clamp(Math.round(source.startFrame || 0), 0, sourceFrames - 1); const requestedEnd = Number(source.endFrame);
    const endFrame = clamp(Math.round(requestedEnd < 0 ? sourceFrames - 1 : requestedEnd), startFrame, sourceFrames - 1); const targetFps = source.targetFps > 0 ? source.targetFps : sourceFps;
    const selectedFrames = Math.max(1, endFrame - startFrame + 1); const loadedFrames = Math.max(1, roundLikePython(selectedFrames * targetFps / sourceFps));
    return { ...sourceInfo, source_fps: sourceFps, source_frames: sourceFrames, source_start_frame: startFrame, source_end_frame: endFrame, loaded_fps: targetFps, frames: loadedFrames, fps: targetFps };
}
function sourceFrameForLocal(info, localFrame) { const local = clamp(Math.round(Number(localFrame) || 0), 0, Math.max(0, Number(info?.frames || 1) - 1)); const count = Math.max(1, Number(info?.frames || 1)); const start = Number(info?.source_start_frame || 0); const end = Number(info?.source_end_frame ?? start); return count <= 1 || end <= start ? start : start + roundLikePython(local * (end - start) / (count - 1)); }
function localFrameForSource(info, sourceFrame) { const count = Math.max(1, Number(info?.frames || 1)); const start = Number(info?.source_start_frame || 0); const end = Number(info?.source_end_frame ?? start); return count <= 1 || end <= start ? 0 : roundLikePython((Number(sourceFrame) - start) * (count - 1) / (end - start)); }
async function fetchPreview(payload, route) { const response = await api.fetchApi(route, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }); const result = await response.json(); if (!response.ok) throw new Error(result.error || "Video segmentation preview failed"); return result; }

function normalizePoints(value) {
    const list = Array.isArray(value) ? value : value?.points; if (!Array.isArray(list)) return [];
    return list.flatMap((point) => { if (Array.isArray(point) && point.length >= 2) return [{ x: clamp(Number(point[0]) || 0, 0, 1), y: clamp(Number(point[1]) || 0, 0, 1), label: Number(point[2]) === 0 ? 0 : 1 }]; if (!point || typeof point !== "object") return []; return [{ x: clamp(Number(point.x) || 0, 0, 1), y: clamp(Number(point.y) || 0, 0, 1), label: Number(point.label) === 0 || String(point.label).toLowerCase() === "negative" ? 0 : 1 }]; });
}
function normalizeBox(value) { const box = Array.isArray(value) ? value[0] : value; if (Array.isArray(box) && box.length >= 4) return { x: clamp(Number(box[0]) || 0, 0, 1), y: clamp(Number(box[1]) || 0, 0, 1), w: clamp(Number(box[2]) || 0, 0, 1), h: clamp(Number(box[3]) || 0, 0, 1) }; if (!box || typeof box !== "object") return null; const w = Number(box.w ?? box.width); const h = Number(box.h ?? box.height); return Number.isFinite(w) && Number.isFinite(h) ? { x: clamp(Number(box.x) || 0, 0, 1), y: clamp(Number(box.y) || 0, 0, 1), w: clamp(w, 0, 1), h: clamp(h, 0, 1) } : null; }
function normalizePromptObject(value) { const object = value && typeof value === "object" ? value : {}; const mask = object.mask; return { text: String(object.text ?? object.semantic ?? "").trim(), points: normalizePoints(object.points), box: normalizeBox(object.bbox ?? object.box), maskData: typeof mask === "string" ? mask : String(mask?.data || mask?.png || ""), maskCanvas: null }; }
function normalizePrompt(value) { const parsed = parseJson(value, null); const objects = Array.isArray(parsed) ? parsed : parsed?.objects; if (!Array.isArray(objects) || !objects.length) return [normalizePromptObject(null)]; return objects.map(normalizePromptObject); }
function promptDataFromObjects(objects, width, height) { return JSON.stringify({ version: 2, objects: objects.map((object, index) => ({ id: index + 1, text: object.text || "", bbox: object.box ? { x: object.box.x, y: object.box.y, w: object.box.w, h: object.box.h } : null, points: object.points, mask: object.maskData ? { data: object.maskData, width, height } : null })) }); }

function addStyles() {
    if (document.getElementById(STYLE_ID)) return; const style = document.createElement("style"); style.id = STYLE_ID;
    style.textContent = `.cs-vseg-dialog{width:min(980px,94vw);max-width:none;max-height:92vh;overflow:auto;padding:0;border:1px solid #343943;border-radius:10px;background:#17191e;color:#e6e9ef;box-shadow:0 22px 80px #000b}.cs-vseg-dialog::backdrop{background:#050609b8}.cs-vseg-shell{display:grid;gap:12px;padding:16px;font:13px/1.35 system-ui,sans-serif}.cs-vseg-head,.cs-vseg-row,.cs-vseg-actions{display:flex;align-items:center;gap:9px;flex-wrap:wrap}.cs-vseg-head{justify-content:space-between}.cs-vseg-title{margin:0;font-size:17px}.cs-vseg-muted{color:#9da5b4}.cs-vseg-stage{position:relative;width:100%;min-height:180px;background:#08090b;border:1px solid #343943;border-radius:6px;overflow:hidden}.cs-vseg-stage video,.cs-vseg-stage .cs-vseg-image-source{display:block;width:100%;height:auto;max-height:58vh;background:#08090b}.cs-vseg-stage .cs-vseg-image-source{display:none;object-fit:contain}.cs-vseg-stage .cs-vseg-mask-preview{display:none;position:absolute;z-index:1;inset:0;width:100%;height:100%;object-fit:contain;background:#08090b}.cs-vseg-stage canvas{position:absolute;z-index:2;inset:0;width:100%;height:100%;touch-action:none}.cs-vseg-controls{display:grid;grid-template-columns:auto minmax(100px,1fr) 90px auto;align-items:center;gap:8px}.cs-vseg-step-buttons{display:flex;gap:5px}.cs-vseg-step-buttons .cs-vseg-button{width:38px;padding-inline:0}.cs-vseg-controls input[type=range]{width:100%}.cs-vseg-button{min-height:31px;border:1px solid #424956;border-radius:5px;padding:6px 10px;background:#20232a;color:#f2f4f7;cursor:pointer}.cs-vseg-button:hover{border-color:#6aa9df}.cs-vseg-button.active{background:#317ec4;border-color:#6db6ee}.cs-vseg-tabs{display:grid;grid-template-columns:repeat(4,1fr);gap:5px;border-bottom:1px solid #343943}.cs-vseg-tab{border-radius:5px 5px 0 0;border-bottom:2px solid transparent}.cs-vseg-tab.active{background:#263d51;border-bottom-color:#55b7dc}.cs-vseg-card{display:none;min-height:54px;padding:10px;border:1px solid #343943;border-radius:0 0 6px 6px;background:#1c1f25}.cs-vseg-card.active{display:flex;align-items:center;gap:9px;flex-wrap:wrap}.cs-vseg-card label{display:flex;align-items:center;gap:6px;color:#9da5b4}.cs-vseg-brush-size{width:330px!important}.cs-vseg-semantic-input{flex:1 1 360px;min-height:31px;border:1px solid #424956;border-radius:5px;padding:6px 9px;background:#111419;color:#f2f4f7}.cs-vseg-brush-mode[data-brush=paint].active{background:#1b6d4b;border-color:#35c98e}.cs-vseg-brush-mode[data-brush=erase].active{background:#7b2934;border-color:#ff5b68}.cs-vseg-point-menu{position:fixed;z-index:20;display:grid;min-width:190px;padding:5px;gap:3px;border:1px solid #424956;border-radius:6px;background:#20232a;box-shadow:0 10px 32px #000b}.cs-vseg-point-menu button{border:0;border-radius:4px;padding:7px 9px;text-align:left;background:transparent;color:#f2f4f7;cursor:pointer}.cs-vseg-point-menu button:hover{background:#317ec4}.cs-vseg-actions{justify-content:flex-end}.cs-vseg-preview-status{flex:1;min-width:0;color:#9da5b4}.cs-vseg-preview-button{border-color:#348f85}.cs-vseg-apply{background:#317ec4;border-color:#4b9de8}.cs-vseg-prompt-note{min-height:18px;color:#9da5b4}.cs-vseg-tool-group label{display:flex;align-items:center;gap:5px;color:#9da5b4}.cs-vseg-object-select{min-height:31px;border:1px solid #424956;border-radius:5px;padding:5px 8px;background:#20232a;color:#f2f4f7}.cs-vseg-bbox-add.active{background:#8a5f1b;border-color:#f7b955}@media(max-width:640px){.cs-vseg-controls{grid-template-columns:auto 1fr auto}.cs-vseg-controls .cs-vseg-frame-count{grid-column:1/-1}.cs-vseg-brush-size{width:100%!important}}`;
    style.textContent += ".cs-vseg-brush-toggle.brush-active{background:#1b6d4b;border-color:#35c98e}.cs-vseg-brush-toggle.eraser-active{background:#7b2934;border-color:#ff5b68}.cs-vseg-clear-all-prompt{margin-left:auto;border-color:#a65a62;background:#382329;color:#ffd9dc}.cs-vseg-timeline{position:relative;min-width:0;padding-top:16px}.cs-vseg-timeline .cs-vseg-slider{display:block;width:100%;margin:0}.cs-vseg-anchor-pointer{display:none;position:absolute;top:0;left:0;width:18px;height:16px;transform:translateX(-50%);border-radius:2px;background:#55a9f5;clip-path:polygon(0 0,100% 0,50% 100%);pointer-events:none;z-index:3}.cs-vseg-anchor-pointer.visible{display:block}";
    document.head.append(style);
}

async function openSelector(node, config) {
    const names = { frame: "anchor_frame", prompt: "prompt_data", ...(config.widgets || {}) };
    let source = null;
    try { source = await fetchCachedSource(node); } catch { source = null; }
    if (!source) source = connectedVideoSource(node, config.videoInputs || ["images", "video_input"]);
    if (!source) { app.canvas?.prompt?.("Run the workflow once to cache the connected video input before opening the selector", ""); return; }
    const upstreamSource = connectedVideoSource(node, config.videoInputs || ["images", "video_input"]);
    if (upstreamSource?.loaderId) {
        try { source = await ensureLoaderPreviewSource(upstreamSource); } catch (error) { app.canvas?.prompt?.(error.message, ""); return; }
    } else if (source.loaderId && !source.token) {
        try { source = await ensureLoaderPreviewSource(source); } catch (error) { app.canvas?.prompt?.(error.message, ""); return; }
    }
    const filename = source.filename; const sourceLabel = source.label || filename; addStyles();
    const dialog = document.createElement("dialog"); dialog.className = "cs-vseg-dialog";
    dialog.innerHTML = `<div class="cs-vseg-shell"><div class="cs-vseg-head"><div><h2 class="cs-vseg-title">${config.title || "Video Selector"}</h2><div class="cs-vseg-muted cs-vseg-file"></div></div><button class="cs-vseg-button cs-vseg-close" type="button">&times;</button></div><div class="cs-vseg-stage"><video controls muted playsinline preload="metadata"></video><img class="cs-vseg-image-source" alt="Input image"><img class="cs-vseg-mask-preview" alt="Current frame segmentation preview"><canvas></canvas></div><div class="cs-vseg-controls"><div class="cs-vseg-step-buttons"><button class="cs-vseg-button cs-vseg-prev" type="button">|&lt;</button><button class="cs-vseg-button cs-vseg-next" type="button">&gt;|</button></div><div class="cs-vseg-timeline"><span class="cs-vseg-anchor-pointer" title="Anchor frame"></span><input class="cs-vseg-slider" type="range" min="0" max="0" step="1" value="0"></div><input class="cs-vseg-frame-input" type="number" min="0" step="1" value="0"><span class="cs-vseg-frame-count cs-vseg-muted">0 / 0</span></div><div class="cs-vseg-tabs"><button class="cs-vseg-button cs-vseg-tab active" data-tab="mask" type="button">Paint Mask</button><button class="cs-vseg-button cs-vseg-tab" data-tab="bbox" type="button">Edit BBox</button><button class="cs-vseg-button cs-vseg-tab" data-tab="points" type="button">Edit Point</button><button class="cs-vseg-button cs-vseg-tab" data-tab="semantic" type="button">Semantic</button></div><div class="cs-vseg-card cs-vseg-card-mask active"><button class="cs-vseg-button cs-vseg-brush-toggle brush-active" type="button">Brush</button><label>Brush Size <input class="cs-vseg-brush-size" type="range" min="2" max="100" value="32"><span class="cs-vseg-brush-size-value">32</span></label><button class="cs-vseg-button cs-vseg-clear-mask" type="button">Clear Mask</button></div><div class="cs-vseg-card cs-vseg-card-bbox"><button class="cs-vseg-button cs-vseg-bbox-add" type="button">Add BBox</button><button class="cs-vseg-button cs-vseg-clear-all-bbox" type="button">Clear All BBox</button></div><div class="cs-vseg-card cs-vseg-card-points"><button class="cs-vseg-button cs-vseg-clear-all-points" type="button">Clear All Point</button></div><div class="cs-vseg-card cs-vseg-card-semantic"><label for="cs-vseg-semantic-text">Text prompt</label><input id="cs-vseg-semantic-text" class="cs-vseg-semantic-input" type="text" maxlength="240" placeholder="e.g. a yellow car"><button class="cs-vseg-button cs-vseg-clear-semantic" type="button">Clear Semantic</button></div><div class="cs-vseg-row cs-vseg-tool-group"><label>Object <select class="cs-vseg-object-select"></select></label><button class="cs-vseg-button cs-vseg-add-object" type="button">Add Object</button><button class="cs-vseg-button cs-vseg-delete-object" type="button">Delete Object</button><button class="cs-vseg-button cs-vseg-undo" type="button">Undo</button><button class="cs-vseg-button cs-vseg-redo" type="button">Redo</button><button class="cs-vseg-button cs-vseg-clear cs-vseg-clear-all-prompt" type="button">Clear All Prompt</button></div><div class="cs-vseg-fields"><div class="cs-vseg-prompt-note"></div></div><div class="cs-vseg-actions"><span class="cs-vseg-preview-status"></span><button class="cs-vseg-button cs-vseg-preview-button" type="button">Preview Current Frame</button><button class="cs-vseg-button cs-vseg-cancel" type="button">Cancel</button><button class="cs-vseg-button cs-vseg-apply" type="button">Apply to Node</button></div></div>`;
    document.body.append(dialog);
    // Keep the shared selector layout consistent across model variants.
    const shell = dialog.querySelector(".cs-vseg-shell");
    const tabHost = dialog.querySelector(".cs-vseg-tabs");
    ["bbox", "points", "mask", "semantic"].forEach((name) => {
        const tab = tabHost?.querySelector(`[data-tab="${name}"]`);
        if (tab) tabHost.append(tab);
        const card = shell?.querySelector(`.cs-vseg-card-${name}`);
        const toolGroup = shell?.querySelector(".cs-vseg-tool-group");
        if (card && toolGroup) shell.insertBefore(card, toolGroup);
    });
    const maskTab = tabHost?.querySelector('[data-tab="mask"]');
    if (maskTab) maskTab.textContent = "Draw Mask";
    const bboxTab = tabHost?.querySelector('[data-tab="bbox"]');
    const bboxCard = shell?.querySelector('.cs-vseg-card-bbox');
    bboxTab?.classList.add("active");
    maskTab?.classList.remove("active");
    bboxCard?.classList.add("active");
    shell?.querySelector('.cs-vseg-card-mask')?.classList.remove("active");
    if (config.semantic === false) {
        dialog.querySelector('[data-tab="semantic"]')?.remove();
        dialog.querySelector('.cs-vseg-card-semantic')?.remove();
        dialog.querySelector('.cs-vseg-tabs').style.gridTemplateColumns = "repeat(3, 1fr)";
    }
    const video = dialog.querySelector("video"); const imageSource = dialog.querySelector(".cs-vseg-image-source"); const media = source.kind === "image" ? imageSource : video; const previewImage = dialog.querySelector(".cs-vseg-mask-preview"); const canvas = dialog.querySelector("canvas"); const context = canvas.getContext("2d"); const stage = dialog.querySelector(".cs-vseg-stage"); const slider = dialog.querySelector(".cs-vseg-slider"); const anchorPointer = dialog.querySelector(".cs-vseg-anchor-pointer"); const frameInput = dialog.querySelector(".cs-vseg-frame-input"); const frameCount = dialog.querySelector(".cs-vseg-frame-count"); const note = dialog.querySelector(".cs-vseg-prompt-note"); const previewStatus = dialog.querySelector(".cs-vseg-preview-status"); const previewButton = dialog.querySelector(".cs-vseg-preview-button"); const objectSelect = dialog.querySelector(".cs-vseg-object-select"); const semanticInput = dialog.querySelector(".cs-vseg-semantic-input"); const brushSize = dialog.querySelector(".cs-vseg-brush-size"); const brushValue = dialog.querySelector(".cs-vseg-brush-size-value"); const overlayCanvas = document.createElement("canvas"); const overlayContext = overlayCanvas.getContext("2d");
    let info = null; let frame = Math.max(0, Number(widget(node, names.frame)?.value || 0)); let anchorFrame = frame; let anchorActive = false; let sliderCandidate = frame; let displayedSourceFrame = null; let objects = normalizePrompt(widget(node, names.prompt)?.value); if (config.semantic === false) objects.forEach((object) => { object.text = ""; }); let activeObject = 0; let tool = "bbox"; let brushMode = "paint"; let bboxAddMode = false; let drag = null; let pointMenu = null; let boxMenu = null; const undoStack = []; const redoStack = [];
    dialog.querySelector(".cs-vseg-file").textContent = sourceLabel; if (source.kind === "image") { video.style.display = "none"; imageSource.style.display = "block"; imageSource.src = source.url || imageUrl(filename); } else { video.src = source.url || videoUrl(filename); }
    const currentObject = () => objects[activeObject] || objects[0];
    const hidePreview = () => { previewImage.style.display = "none"; previewImage.removeAttribute("src"); previewStatus.textContent = ""; canvas.style.visibility = "visible"; };
    function promptSnapshot() { const width = Number(info?.width || media.videoWidth || media.naturalWidth || 1); const height = Number(info?.height || media.videoHeight || media.naturalHeight || 1); objects.forEach((object) => { if (!object.maskCanvas) return; const pixels = object.maskCanvas.getContext("2d").getImageData(0, 0, object.maskCanvas.width, object.maskCanvas.height).data; let hasMask = false; for (let index = 3; index < pixels.length; index += 4) { if (pixels[index] > 0) { hasMask = true; break; } } object.maskData = hasMask ? object.maskCanvas.toDataURL("image/png") : ""; }); return promptDataFromObjects(objects, width, height); }
    function hasPromptData() { return objects.some((object) => Boolean(object.text?.trim() || object.box || object.points?.length || object.maskData)); }
    function updateAnchorPointer() { const maxFrame = Math.max(0, Number(info?.frames || 1) - 1); const position = maxFrame > 0 ? clamp(anchorFrame, 0, maxFrame) / maxFrame : 0.5; anchorPointer.style.left = `${position * 100}%`; anchorPointer.classList.toggle("visible", anchorActive); anchorPointer.title = anchorActive ? `Anchor frame ${anchorFrame}` : "Anchor frame"; }
    function syncAnchorFromPrompts() { const nextActive = hasPromptData(); if (nextActive && !anchorActive) anchorFrame = frame; anchorActive = nextActive; updateAnchorPointer(); }
    function updateSemanticInput() { if (semanticInput && document.activeElement !== semanticInput) semanticInput.value = currentObject()?.text || ""; }
    function updateObjectSelect() { objectSelect.innerHTML = objects.map((_, index) => `<option value="${index}">Object ${index + 1}</option>`).join(""); objectSelect.value = String(activeObject); dialog.querySelector(".cs-vseg-delete-object").disabled = objects.length <= 1; updateSemanticInput(); }
    async function ensureMaskCanvas(object) { const width = Math.max(1, Number(info?.width || media.videoWidth || media.naturalWidth || 1)); const height = Math.max(1, Number(info?.height || media.videoHeight || media.naturalHeight || 1)); if (object.maskCanvas) return; object.maskCanvas = document.createElement("canvas"); object.maskCanvas.width = width; object.maskCanvas.height = height; if (!object.maskData) return; await new Promise((resolve) => { const image = new Image(); image.onload = () => { object.maskCanvas.getContext("2d").drawImage(image, 0, 0, width, height); resolve(); }; image.onerror = resolve; image.src = object.maskData; }); }
    async function restoreSnapshot(value) { objects = normalizePrompt(value); activeObject = Math.min(activeObject, objects.length - 1); await Promise.all(objects.map(ensureMaskCanvas)); updateObjectSelect(); syncAnchorFromPrompts(); redraw(); }
    function commit(before) { const after = promptSnapshot(); if (before !== after) { undoStack.push(before); redoStack.length = 0; } syncAnchorFromPrompts(); }
    function setTool(next, preserveBBoxAdd = false) {
        tool = next;
        hidePreview();
        dialog.querySelectorAll(".cs-vseg-tab").forEach((button) => button.classList.toggle("active", button.dataset.tab === tool));
        dialog.querySelectorAll(".cs-vseg-card").forEach((card) => card.classList.toggle("active", card.classList.contains(`cs-vseg-card-${tool}`)));
        if (tool !== "bbox" || !preserveBBoxAdd) {
            bboxAddMode = false;
            dialog.querySelector(".cs-vseg-bbox-add").classList.remove("active");
        }
        updateCursor();
        redraw();
    }
    function eventPosition(event) { const rect = canvas.getBoundingClientRect(); return { x: clamp((event.clientX - rect.left) / rect.width, 0, 1), y: clamp((event.clientY - rect.top) / rect.height, 0, 1) }; }
    function drawPoint(point, active) { const x = point.x * canvas.width; const y = point.y * canvas.height; context.globalAlpha = active ? 1 : 0.45; context.beginPath(); context.arc(x, y, 7, 0, Math.PI * 2); context.fillStyle = point.label === 1 ? "#4dd0c2" : "#ff7a87"; context.fill(); context.lineWidth = 2; context.strokeStyle = "#081019"; context.stroke(); context.globalAlpha = 1; }
    function drawBox(box, active) { const left = box.x * canvas.width; const top = box.y * canvas.height; const right = (box.x + box.w) * canvas.width; const bottom = (box.y + box.h) * canvas.height; context.globalAlpha = active ? 1 : 0.45; context.fillStyle = active ? "rgba(247,185,85,.12)" : "rgba(247,185,85,.05)"; context.fillRect(left, top, right - left, bottom - top); context.strokeStyle = "#f7b955"; context.lineWidth = active ? 3 : 1.5; context.strokeRect(left, top, right - left, bottom - top); if (active) { context.fillStyle = "#f7b955"; context.strokeStyle = "#17191e"; context.lineWidth = 1; for (const [x, y] of [[left, top], [right, top], [left, bottom], [right, bottom]]) { context.beginPath(); context.rect(x - 4, y - 4, 8, 8); context.fill(); context.stroke(); } } context.globalAlpha = 1; }
    function drawMask(object, active) { if (!object?.maskCanvas) return; overlayCanvas.width = canvas.width; overlayCanvas.height = canvas.height; overlayContext.clearRect(0, 0, canvas.width, canvas.height); overlayContext.fillStyle = active ? "#35c8b2" : "#6f8ea0"; overlayContext.fillRect(0, 0, canvas.width, canvas.height); overlayContext.globalCompositeOperation = "destination-in"; overlayContext.drawImage(object.maskCanvas, 0, 0, canvas.width, canvas.height); overlayContext.globalCompositeOperation = "source-over"; context.globalAlpha = active ? .32 : .15; context.drawImage(overlayCanvas, 0, 0); context.globalAlpha = 1; }
    function redraw() { context.clearRect(0, 0, canvas.width, canvas.height); objects.forEach((object, index) => drawMask(object, index === activeObject)); objects.forEach((object, index) => object.box && drawBox(object.box, index === activeObject)); objects.forEach((object, index) => object.points.forEach((point) => drawPoint(point, index === activeObject))); }
    function resizeCanvas() { const rect = media.getBoundingClientRect(); const width = Math.max(1, Math.round(rect.width)); const height = Math.max(1, Math.round(rect.height)); if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; } redraw(); }
    function pointIndexAt(position) { const points = currentObject().points; const rect = canvas.getBoundingClientRect(); let hit = -1; let distance = 12; points.forEach((point, index) => { const next = Math.hypot((point.x - position.x) * rect.width, (point.y - position.y) * rect.height); if (next <= distance) { hit = index; distance = next; } }); return hit; }
    function closePointMenu() { pointMenu?.remove(); pointMenu = null; } function closeBoxMenu() { boxMenu?.remove(); boxMenu = null; }
    function showPointMenu(index, event) { closePointMenu(); const point = currentObject().points[index]; pointMenu = document.createElement("div"); pointMenu.className = "cs-vseg-point-menu"; pointMenu.innerHTML = `<button type="button" data-action="delete">Delete point</button><button type="button" data-action="toggle">${point.label === 1 ? "Change point to negative" : "Change point to positive"}</button>`; dialog.append(pointMenu); pointMenu.style.left = `${clamp(event.clientX, 8, window.innerWidth - pointMenu.offsetWidth - 8)}px`; pointMenu.style.top = `${clamp(event.clientY, 8, window.innerHeight - pointMenu.offsetHeight - 8)}px`; pointMenu.addEventListener("click", (menuEvent) => { const before = promptSnapshot(); const action = menuEvent.target.closest("button")?.dataset.action; if (action === "delete") currentObject().points.splice(index, 1); if (action === "toggle" && currentObject().points[index]) currentObject().points[index].label = currentObject().points[index].label === 1 ? 0 : 1; commit(before); closePointMenu(); hidePreview(); redraw(); }); }
    function boxTargetAt(position) { const box = currentObject().box; if (!box) return null; const rect = canvas.getBoundingClientRect(); const px = position.x * rect.width; const py = position.y * rect.height; const left = box.x * rect.width; const top = box.y * rect.height; const right = (box.x + box.w) * rect.width; const bottom = (box.y + box.h) * rect.height; for (const [target, x, y] of [["nw", left, top], ["ne", right, top], ["sw", left, bottom], ["se", right, bottom]]) if (Math.hypot(px - x, py - y) <= 11) return target; if (px >= left - 7 && px <= right + 7 && Math.abs(py - top) <= 7) return "n"; if (px >= left - 7 && px <= right + 7 && Math.abs(py - bottom) <= 7) return "s"; if (py >= top - 7 && py <= bottom + 7 && Math.abs(px - left) <= 7) return "w"; if (py >= top - 7 && py <= bottom + 7 && Math.abs(px - right) <= 7) return "e"; if (px > left && px < right && py > top && py < bottom) return "move"; return null; }
    function boxCursor(target) { if (target === "nw" || target === "se") return "nwse-resize"; if (target === "ne" || target === "sw") return "nesw-resize"; if (target === "n" || target === "s") return "ns-resize"; if (target === "e" || target === "w") return "ew-resize"; if (target === "move") return "grab"; return "crosshair"; }
    function brushCursor() {
        const diameter = clamp(Number(brushSize.value) || 32, 2, 100);
        const color = brushMode === "erase" ? "#ff5b68" : "#35c98e";
        const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${diameter}" height="${diameter}" viewBox="0 0 ${diameter} ${diameter}"><circle cx="${diameter / 2}" cy="${diameter / 2}" r="${Math.max(1, diameter / 2 - 1)}" fill="none" stroke="${color}" stroke-opacity="0.82" stroke-width="2"/></svg>`;
        return `url("data:image/svg+xml,${encodeURIComponent(svg)}") ${diameter / 2} ${diameter / 2}, crosshair`;
    }
    function updateCursor(position = null) { canvas.style.cursor = tool === "bbox" ? (bboxAddMode && !boxTargetAt(position || { x: -1, y: -1 }) ? "crosshair" : boxCursor(position ? boxTargetAt(position) : null)) : tool === "points" ? (position && pointIndexAt(position) >= 0 ? "grab" : "crosshair") : tool === "semantic" ? "text" : brushCursor(); }
    function updateBox(position) { const object = currentObject(); const original = drag.original; if (drag.target === "draw") { object.box = { x: Math.min(drag.start.x, position.x), y: Math.min(drag.start.y, position.y), w: Math.abs(position.x - drag.start.x), h: Math.abs(position.y - drag.start.y) }; return; } if (drag.target === "move") { object.box = { ...original, x: clamp(original.x + position.x - drag.start.x, 0, 1 - original.w), y: clamp(original.y + position.y - drag.start.y, 0, 1 - original.h) }; return; } let left = original.x; let top = original.y; let right = original.x + original.w; let bottom = original.y + original.h; if (drag.target.includes("w")) left = position.x; if (drag.target.includes("e")) right = position.x; if (drag.target.includes("n")) top = position.y; if (drag.target.includes("s")) bottom = position.y; object.box = { x: clamp(Math.min(left, right), 0, 1), y: clamp(Math.min(top, bottom), 0, 1), w: clamp(Math.abs(right - left), 0, 1), h: clamp(Math.abs(bottom - top), 0, 1) }; }
    function paintMask(position) { const object = currentObject(); if (!object.maskCanvas) return; const maskContext = object.maskCanvas.getContext("2d"); const x = position.x * object.maskCanvas.width; const y = position.y * object.maskCanvas.height; const radius = Number(brushSize.value) * object.maskCanvas.width / Math.max(1, canvas.width) / 2; maskContext.save(); maskContext.globalCompositeOperation = brushMode === "erase" ? "destination-out" : "source-over"; maskContext.fillStyle = "#fff"; maskContext.beginPath(); maskContext.arc(x, y, Math.max(1, radius), 0, Math.PI * 2); maskContext.fill(); maskContext.restore(); }
    function showBoxMenu(event) { closePointMenu(); closeBoxMenu(); boxMenu = document.createElement("div"); boxMenu.className = "cs-vseg-point-menu"; boxMenu.innerHTML = `<button type="button" data-action="delete">删除BBox</button>`; dialog.append(boxMenu); boxMenu.style.left = `${clamp(event.clientX, 8, window.innerWidth - boxMenu.offsetWidth - 8)}px`; boxMenu.style.top = `${clamp(event.clientY, 8, window.innerHeight - boxMenu.offsetHeight - 8)}px`; boxMenu.addEventListener("click", (menuEvent) => { if (menuEvent.target.closest("button")?.dataset.action === "delete") { const before = promptSnapshot(); currentObject().box = null; commit(before); closeBoxMenu(); hidePreview(); redraw(); } }); }
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    canvas.addEventListener("pointerdown", (event) => {
        const position = eventPosition(event); closeBoxMenu();
        if (tool === "mask") { if (event.button !== 0) return; drag = { kind: "mask", before: promptSnapshot(), pointerId: event.pointerId }; paintMask(position); canvas.setPointerCapture?.(event.pointerId); hidePreview(); redraw(); return; }
        if (tool === "points") { if (event.button !== 0 && event.button !== 2) return; event.preventDefault(); const index = pointIndexAt(position); if (event.button === 2) { if (index >= 0) showPointMenu(index, event); else { const before = promptSnapshot(); currentObject().points.push({ ...position, label: 0 }); commit(before); hidePreview(); redraw(); } return; } closePointMenu(); if (index >= 0) { drag = { kind: "point", pointIndex: index, start: position, original: { ...currentObject().points[index] }, before: promptSnapshot(), moved: false, pointerId: event.pointerId }; canvas.setPointerCapture?.(event.pointerId); return; } const before = promptSnapshot(); currentObject().points.push({ ...position, label: 1 }); commit(before); hidePreview(); redraw(); return; }
        if (tool === "bbox") { if (event.button === 2) { if (boxTargetAt(position) === "move") showBoxMenu(event); return; } if (event.button !== 0) return; const target = boxTargetAt(position); if (!target && !bboxAddMode) return; drag = { kind: "bbox", target: target || "draw", start: position, original: currentObject().box ? { ...currentObject().box } : null, previous: currentObject().box ? { ...currentObject().box } : null, before: promptSnapshot(), pointerId: event.pointerId }; if (!target) currentObject().box = { x: position.x, y: position.y, w: 0, h: 0 }; canvas.setPointerCapture?.(event.pointerId); hidePreview(); redraw(); }
    });
    canvas.addEventListener("pointermove", (event) => { const position = eventPosition(event); if (!drag) { updateCursor(position); return; } if (drag.kind === "mask") { paintMask(position); hidePreview(); redraw(); return; } if (drag.kind === "point") { const rect = canvas.getBoundingClientRect(); if (!drag.moved && Math.hypot((position.x - drag.start.x) * rect.width, (position.y - drag.start.y) * rect.height) < 2) return; drag.moved = true; const point = currentObject().points[drag.pointIndex]; if (point) currentObject().points[drag.pointIndex] = { ...point, x: position.x, y: position.y }; canvas.style.cursor = "grabbing"; hidePreview(); redraw(); return; } updateBox(position); canvas.style.cursor = drag.target === "move" ? "grabbing" : boxCursor(drag.target); hidePreview(); redraw(); });
    canvas.addEventListener("pointerleave", () => { if (!drag) updateCursor(); });
    canvas.addEventListener("pointerup", (event) => { if (!drag) return; if (drag.kind === "bbox" && (!currentObject().box || currentObject().box.w * canvas.width < 3 || currentObject().box.h * canvas.height < 3)) currentObject().box = drag.previous; if (drag.kind === "bbox" && drag.target === "draw") { bboxAddMode = false; dialog.querySelector(".cs-vseg-bbox-add").classList.remove("active"); } commit(drag.before); canvas.releasePointerCapture?.(drag.pointerId); drag = null; updateCursor(eventPosition(event)); redraw(); });
    canvas.addEventListener("pointercancel", () => { if (drag?.kind === "point" && currentObject().points[drag.pointIndex]) currentObject().points[drag.pointIndex] = drag.original; if (drag?.kind === "bbox") { currentObject().box = drag.previous; if (drag.target === "draw") { bboxAddMode = false; dialog.querySelector(".cs-vseg-bbox-add").classList.remove("active"); } } if (drag?.kind === "mask") restoreSnapshot(drag.before); drag = null; redraw(); });
    dialog.addEventListener("pointerdown", (event) => { if (pointMenu && !pointMenu.contains(event.target)) closePointMenu(); if (boxMenu && !boxMenu.contains(event.target)) closeBoxMenu(); }, true);
    dialog.querySelectorAll(".cs-vseg-tab").forEach((button) => button.addEventListener("click", () => setTool(button.dataset.tab)));
    dialog.querySelector(".cs-vseg-brush-toggle").addEventListener("click", (event) => { brushMode = brushMode === "paint" ? "erase" : "paint"; const button = event.currentTarget; button.textContent = brushMode === "paint" ? "Brush" : "Eraser"; button.classList.toggle("brush-active", brushMode === "paint"); button.classList.toggle("eraser-active", brushMode === "erase"); updateCursor(); });
    brushSize.addEventListener("input", () => { brushValue.textContent = brushSize.value; updateCursor(); }); canvas.addEventListener("wheel", (event) => { if (tool !== "mask") return; event.preventDefault(); const step = event.shiftKey ? 10 : 2; const current = Number(brushSize.value) || 32; const next = clamp(current + (event.deltaY < 0 ? step : -step), 2, 100); brushSize.value = String(next); brushValue.textContent = String(next); updateCursor(); }, { passive: false });
    dialog.querySelector(".cs-vseg-bbox-add").addEventListener("click", async () => { if (currentObject().box) { const before = promptSnapshot(); objects.push(normalizePromptObject(null)); activeObject = objects.length - 1; await ensureMaskCanvas(currentObject()); commit(before); updateObjectSelect(); } bboxAddMode = true; setTool("bbox", true); dialog.querySelector(".cs-vseg-bbox-add").classList.add("active"); updateCursor(); });
    dialog.querySelector(".cs-vseg-clear-all-bbox").addEventListener("click", () => { const before = promptSnapshot(); objects.forEach((object) => { object.box = null; }); bboxAddMode = false; dialog.querySelector(".cs-vseg-bbox-add").classList.remove("active"); commit(before); hidePreview(); redraw(); });
    dialog.querySelector(".cs-vseg-clear-all-points").addEventListener("click", () => { const before = promptSnapshot(); objects.forEach((object) => { object.points = []; }); commit(before); hidePreview(); redraw(); });
    objectSelect.addEventListener("change", async () => { activeObject = clamp(Number(objectSelect.value) || 0, 0, objects.length - 1); await ensureMaskCanvas(currentObject()); updateSemanticInput(); redraw(); });
    if (semanticInput) {
        semanticInput.addEventListener("change", () => { const before = promptSnapshot(); currentObject().text = semanticInput.value.trim(); commit(before); hidePreview(); redraw(); });
        semanticInput.addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); semanticInput.blur(); } });
        dialog.querySelector(".cs-vseg-clear-semantic")?.addEventListener("click", () => { const before = promptSnapshot(); currentObject().text = ""; semanticInput.value = ""; commit(before); hidePreview(); redraw(); });
    }
    dialog.querySelector(".cs-vseg-add-object").addEventListener("click", async () => { const before = promptSnapshot(); objects.push(normalizePromptObject(null)); activeObject = objects.length - 1; await ensureMaskCanvas(currentObject()); commit(before); updateObjectSelect(); redraw(); });
    dialog.querySelector(".cs-vseg-delete-object").addEventListener("click", () => { if (objects.length <= 1) return; const before = promptSnapshot(); objects.splice(activeObject, 1); activeObject = Math.min(activeObject, objects.length - 1); commit(before); updateObjectSelect(); redraw(); });
    dialog.querySelector(".cs-vseg-undo").addEventListener("click", async () => { if (!undoStack.length) return; redoStack.push(promptSnapshot()); await restoreSnapshot(undoStack.pop()); hidePreview(); });
    dialog.querySelector(".cs-vseg-redo").addEventListener("click", async () => { if (!redoStack.length) return; undoStack.push(promptSnapshot()); await restoreSnapshot(redoStack.pop()); hidePreview(); });
    dialog.querySelector(".cs-vseg-clear-mask").addEventListener("click", () => { const before = promptSnapshot(); const object = currentObject(); object.maskCanvas?.getContext("2d").clearRect(0, 0, object.maskCanvas.width, object.maskCanvas.height); object.maskData = ""; commit(before); hidePreview(); redraw(); });
    dialog.querySelector(".cs-vseg-clear").addEventListener("click", async () => { const before = promptSnapshot(); objects = [normalizePromptObject(null)]; activeObject = 0; await ensureMaskCanvas(currentObject()); commit(before); updateObjectSelect(); hidePreview(); redraw(); });
    function clearPromptsForFrameChange() { closePointMenu(); closeBoxMenu(); drag = null; bboxAddMode = false; dialog.querySelector(".cs-vseg-bbox-add").classList.remove("active"); objects = [normalizePromptObject(null)]; activeObject = 0; void ensureMaskCanvas(currentObject()); undoStack.length = 0; redoStack.length = 0; anchorActive = false; updateObjectSelect(); updateAnchorPointer(); hidePreview(); redraw(); }
    function setFrameDirect(nextFrame, seek = true) { const maxFrame = Math.max(0, Number(info?.frames || 1) - 1); frame = clamp(Math.round(Number(nextFrame) || 0), 0, maxFrame); sliderCandidate = frame; slider.value = String(frame); frameInput.value = String(frame); frameCount.textContent = `${frame} / ${maxFrame}`; hidePreview(); updateAnchorPointer(); if (seek && source.kind !== "image" && info?.source_fps && Number.isFinite(video.duration)) { displayedSourceFrame = sourceFrameForLocal(info, frame); video.currentTime = displayedSourceFrame / info.source_fps; } }
    function requestFrameChange(nextFrame, seek = true, fromVideo = false) { const maxFrame = Math.max(0, Number(info?.frames || 1) - 1); const target = clamp(Math.round(Number(nextFrame) || 0), 0, maxFrame); if (target === frame) { sliderCandidate = frame; slider.value = String(frame); frameInput.value = String(frame); frameCount.textContent = `${frame} / ${maxFrame}`; return true; } if (anchorActive) { video.pause(); if (!window.confirm("当前已存在编辑数据，切换锚点帧将自动清除，是否继续？")) { sliderCandidate = frame; slider.value = String(frame); frameInput.value = String(frame); frameCount.textContent = `${frame} / ${maxFrame}`; if (fromVideo && source.kind !== "image" && info?.source_fps && Number.isFinite(video.duration)) { displayedSourceFrame = sourceFrameForLocal(info, frame); video.currentTime = displayedSourceFrame / info.source_fps; } return false; } clearPromptsForFrameChange(); } else { undoStack.length = 0; redoStack.length = 0; } setFrameDirect(target, seek); return true; }
    slider.addEventListener("input", () => { sliderCandidate = clamp(Math.round(Number(slider.value) || 0), 0, Math.max(0, Number(info?.frames || 1) - 1)); if (!anchorActive) requestFrameChange(sliderCandidate); }); slider.addEventListener("change", () => requestFrameChange(sliderCandidate)); frameInput.addEventListener("change", () => requestFrameChange(frameInput.value)); dialog.querySelector(".cs-vseg-prev").addEventListener("click", () => requestFrameChange(frame - 1)); dialog.querySelector(".cs-vseg-next").addEventListener("click", () => requestFrameChange(frame + 1));
    previewButton.addEventListener("click", async () => { semanticInput?.blur(); previewButton.disabled = true; previewStatus.textContent = config.previewLabel || "Running segmentation on this frame..."; video.pause(); try { const result = await config.preview({ node, filename, frame, previewFrame: sourceFrameForLocal(info, frame), info, promptData: promptSnapshot(), fetchPreview: (payload) => fetchPreview({ ...payload, source_kind: source.kind || "video", source_token: source.token || "" }, config.previewRoute) }); previewImage.src = result.image; previewImage.style.display = "block"; canvas.style.visibility = "hidden"; previewStatus.textContent = `Frame ${result.frame} · mask ${(Number(result.mask_area || 0) * 100).toFixed(1)}%`; } catch (error) { canvas.style.visibility = "visible"; previewStatus.textContent = error.message; } finally { previewButton.disabled = false; } });
    function syncFrameFromVideo() { if (source.kind === "image" || !info?.source_fps) return; const sourceFrame = Math.round(video.currentTime * info.source_fps); if (sourceFrame === displayedSourceFrame) return; const first = Number(info.source_start_frame || 0); const last = Number(info.source_end_frame ?? first); const accepted = sourceFrame < first || sourceFrame > last ? requestFrameChange(sourceFrame < first ? 0 : Math.max(0, Number(info.frames || 1) - 1), false, true) : requestFrameChange(localFrameForSource(info, sourceFrame), false, true); if (accepted) displayedSourceFrame = sourceFrame; }
    video.addEventListener("timeupdate", () => { if (!video.seeking) syncFrameFromVideo(); }); video.addEventListener("seeked", syncFrameFromVideo); media.addEventListener(source.kind === "image" ? "load" : "loadedmetadata", () => { resizeCanvas(); if (info) setFrameDirect(frame); }); new ResizeObserver(resizeCanvas).observe(stage);
    const close = () => { video.pause(); closePointMenu(); closeBoxMenu(); dialog.close(); dialog.remove(); }; dialog.querySelector(".cs-vseg-close").addEventListener("click", close); dialog.querySelector(".cs-vseg-cancel").addEventListener("click", close);
    dialog.querySelector(".cs-vseg-apply").addEventListener("click", () => { semanticInput?.blur(); const promptData = promptSnapshot(); if (config.apply) config.apply({ node, frame, promptData, info, setWidgetValue }); else { setWidgetValue(node, names.frame, frame); setWidgetValue(node, names.prompt, promptData); } node.graph?.setDirtyCanvas(true, true); close(); }); dialog.addEventListener("cancel", close);
    (source.info ? Promise.resolve(source.info) : fetchInfo(filename)).then(async (result) => { info = prepareInputTimeline(source, result); dialog.querySelector(".cs-vseg-file").textContent = `${sourceLabel} · ${info.frames} input frames`; const maxFrame = Math.max(0, Number(info.frames || 1) - 1); slider.max = String(maxFrame); frameInput.max = String(maxFrame); await Promise.all(objects.map(ensureMaskCanvas)); updateObjectSelect(); setTool(tool); setFrameDirect(frame); syncAnchorFromPrompts(); resizeCanvas(); note.textContent = "Use Semantic, draw a coarse mask, or add points and boxes to the active object. Add objects for additional prompt groups."; }).catch((error) => { note.textContent = error.message; });
    dialog.showModal();
}

export function registerVideoSelector(config) {
    app.registerExtension({
        name: config.extensionName,
        async beforeRegisterNodeDef(nodeType, nodeData) {
            if (nodeData?.name !== config.nodeId) return;
            const original = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () { original?.apply(this, arguments); removeObsoleteInputs(this, config.removeInputs || []); removeObsoleteWidgets(this, config.removeWidgets || []); const button = this.addWidget("button", "Open Selector", "", () => openSelector(this, config)); button.name = "Open Selector"; button.label = "Open Selector"; button.options = { ...(button.options || {}), serialize: false }; this.setSize?.([390, Math.max(360, this.computeSize?.()[1] || 360)]); };
        },
        loadedGraphNode(node) { if (node?.type !== config.nodeId) return; removeObsoleteInputs(node, config.removeInputs || []); removeObsoleteWidgets(node, config.removeWidgets || []); node.setSize?.([node.size?.[0] || 390, node.computeSize?.()[1] || node.size?.[1] || 360]); },
    });
}

// Shared by lightweight preview dialogs that need the same recursive input
// discovery and Selector cache as the full Video Segment UI.
export {
    connectedVideoSource,
    ensureLoaderPreviewSource,
    fetchCachedSource,
    fetchInfo,
    prepareInputTimeline,
    sourceFrameForLocal,
};
