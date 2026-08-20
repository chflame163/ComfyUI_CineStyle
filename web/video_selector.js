import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const STYLE_ID = "cinestyle-video-selector-style";

function widget(node, name) {
    return node.widgets?.find((item) => item.name === name);
}

function setWidgetValue(node, name, value) {
    const target = widget(node, name);
    if (!target) return;
    target.value = value;
    target.callback?.(value);
}

function removeObsoleteInputs(node, names = []) {
    for (const name of names) {
        const index = node.inputs?.findIndex((input) => input.name === name) ?? -1;
        if (index >= 0) node.removeInput?.(index);
    }
}

function removeObsoleteWidgets(node, names = []) {
    for (const name of names) {
        const index = node.widgets?.findIndex((item) => item.name === name) ?? -1;
        if (index >= 0) node.widgets.splice(index, 1);
    }
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function roundLikePython(value) {
    const lower = Math.floor(value);
    const fraction = value - lower;
    if (fraction < 0.5) return lower;
    if (fraction > 0.5) return lower + 1;
    return lower % 2 === 0 ? lower : lower + 1;
}

function videoUrl(filename) {
    const params = new URLSearchParams({
        filename,
        t: String(Date.now()),
    });
    return api.apiURL(`/cinestyle/video-source?${params.toString()}`);
}

async function fetchInfo(filename) {
    const params = new URLSearchParams({ filename });
    const response = await api.fetchApi(`/cinestyle/video-info?${params.toString()}`);
    if (!response.ok) throw new Error(await response.text());
    return await response.json();
}

async function fetchCachedSource(node) {
    const nodeId = String(node?.id ?? "").trim();
    if (!nodeId) return null;
    const params = new URLSearchParams({ node_id: nodeId, t: String(Date.now()) });
    const response = await api.fetchApi(`/cinestyle/video-selector-cache?${params.toString()}`);
    if (response.status === 404) return null;
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Unable to read cached Selector input");
    const info = result.info || {};
    return {
        filename: "",
        label: String(result.label || "Cached input from the last workflow run"),
        url: api.apiURL(result.video_url),
        token: String(result.token || ""),
        info,
        startFrame: 0,
        endFrame: Math.max(0, Number(info.frames || 1) - 1),
        targetFps: Number(info.fps || 24),
    };
}

function graphNode(graph, id) {
    if (id == null || !graph) return null;
    const direct = graph.getNodeById?.(id);
    if (direct) return direct;
    const nodes = graph._nodes || graph.nodes;
    if (Array.isArray(nodes)) {
        return nodes.find((candidate) => String(candidate?.id) === String(id)) || null;
    }
    return graph._nodes_by_id?.[id] || null;
}

function graphLink(graph, candidate) {
    if (candidate == null) return null;
    if (typeof candidate === "object") {
        // Some LiteGraph builds put the LLink object directly in input.link.
        if (candidate.origin_id != null || candidate.originId != null) return candidate;
        if (candidate.link && typeof candidate.link === "object") return candidate.link;
    }
    return graph?.links?.[candidate] || graph?._links?.[candidate] || null;
}

function originFromConnection(graph, candidate) {
    if (!candidate) return null;
    // Newer ComfyUI builds return the origin node from getInputNode().
    if (candidate.type || candidate.comfyClass || candidate.id == null) {
        if (candidate.origin_id != null || candidate.originId != null) {
            const linkOrigin = candidate.origin_id ?? candidate.originId;
            return graphNode(graph, linkOrigin);
        }
        if (candidate.type || candidate.comfyClass) return candidate;
    }
    const link = graphLink(graph, candidate);
    if (!link) return null;
    const originId = link.origin_id ?? link.originId ?? link.origin;
    return graphNode(graph, originId);
}

function connectedOrigin(node, inputName) {
    const inputIndex = node.inputs?.findIndex((item) => item.name === inputName) ?? -1;
    if (inputIndex < 0) return null;
    const input = node.inputs?.[inputIndex];
    const graph = node.graph || app.graph;
    const candidates = [];

    const call = (method, argument) => {
        try {
            return typeof method === "function" ? method.call(node, argument) : null;
        } catch {
            return null;
        }
    };
    // LiteGraph exposes the connected origin directly on newer ComfyUI
    // frontends. Keep both index and name calls for frontend compatibility.
    candidates.push(call(node.getInputNode, inputIndex), call(node.getInputNode, inputName));
    candidates.push(call(node.getInputLink, inputIndex), call(node.getInputLink, inputName));
    candidates.push(input?.link);
    if (Array.isArray(input?.links)) candidates.push(...input.links);

    for (const candidate of candidates) {
        const origin = originFromConnection(graph, candidate);
        if (origin) return origin;
    }
    return null;
}

function nodeTypeName(node) {
    return String(node?.type || node?.comfyClass || node?.constructor?.type || "");
}

function isCSLoadVideo(node) {
    const type = nodeTypeName(node);
    return type === "CS_Load_Video" || type.endsWith(".CS_Load_Video") || type.endsWith("::CS_Load_Video");
}

function isVideoFileSource(node) {
    const type = nodeTypeName(node);
    const filename = sourceFilename(node);
    if (!filename) return false;
    return isCSLoadVideo(node)
        || /load.*video|video.*load|input.*video|video.*input/i.test(type)
        || /\.(mp4|mov|mkv|avi|webm|m4v|mpg|mpeg|wmv|flv)(?:\s*\[[^\]]*\])?$/i.test(filename);
}

function sourceFilename(node) {
    const names = isCSLoadVideo(node)
        ? ["video"]
        : ["file", "video", "filename", "file_path", "filepath", "video_path", "video_file", "video_file_path", "input_path", "path"];
    for (const name of names) {
        const value = String(widget(node, name)?.value || "").trim();
        if (value) return value;
    }
    return "";
}

function mediaInput(input) {
    const descriptor = `${input?.name || ""} ${input?.type || ""}`.toLowerCase();
    return /video|image|frame|media|source|stream|movie/.test(descriptor);
}

function connectedMediaOrigins(node) {
    const inputs = Array.isArray(node?.inputs) ? node.inputs : [];
    const matches = [];
    for (const input of inputs) {
        if (!mediaInput(input)) continue;
        const origin = connectedOrigin(node, input.name);
        if (origin) matches.push(origin);
    }
    return matches;
}

function sourceFromOrigin(origin, visited = new Set()) {
    if (!origin) return null;
    const identity = origin.id != null ? String(origin.id) : `${nodeTypeName(origin)}:${visited.size}`;
    if (visited.has(identity)) return null;
    visited.add(identity);

    if (isVideoFileSource(origin)) {
        const filename = sourceFilename(origin);
        if (!filename) return null;
        return {
            filename,
            startFrame: Math.max(0, Number(widget(origin, "start_frame")?.value ?? 0)),
            endFrame: Number(widget(origin, "end_frame")?.value ?? -1),
            targetFps: Number(widget(origin, "fps")?.value ?? 0),
        };
    }

    // Walk every media-typed input, not a hard-coded intermediate node. This
    // covers video splitters, component extractors, format converters, and
    // custom video pass-through nodes while avoiding unrelated MODEL inputs.
    for (const upstream of connectedMediaOrigins(origin)) {
        const source = sourceFromOrigin(upstream, visited);
        if (source) return source;
    }
    return null;
}

function connectedVideoSource(node, inputNames = ["images", "video_input"]) {
    // IMAGE is the canonical input. If it is connected, never fall back to
    // VIDEO, even when the IMAGE origin cannot provide selector metadata.
    const imagesOrigin = connectedOrigin(node, "images");
    const origins = imagesOrigin ? [["images", imagesOrigin]] : inputNames
        .filter((name) => name !== "images")
        .map((name) => [name, connectedOrigin(node, name)]);
    for (const [, origin] of origins) {
        const source = sourceFromOrigin(origin);
        if (source) return source;
    }
    return null;
}

function prepareInputTimeline(source, sourceInfo) {
    const sourceFrames = Math.max(1, Number(sourceInfo.frames || 1));
    const sourceFps = Math.max(0.001, Number(sourceInfo.fps || 24));
    const startFrame = clamp(Math.round(source.startFrame || 0), 0, sourceFrames - 1);
    const requestedEnd = Number(source.endFrame);
    const endFrame = clamp(
        Math.round(requestedEnd < 0 ? sourceFrames - 1 : requestedEnd),
        startFrame,
        sourceFrames - 1,
    );
    const targetFps = source.targetFps > 0 ? source.targetFps : sourceFps;
    const selectedFrames = Math.max(1, endFrame - startFrame + 1);
    const loadedFrames = Math.max(1, roundLikePython(selectedFrames * targetFps / sourceFps));
    return {
        ...sourceInfo,
        source_fps: sourceFps,
        source_frames: sourceFrames,
        source_start_frame: startFrame,
        source_end_frame: endFrame,
        loaded_fps: targetFps,
        frames: loadedFrames,
        fps: targetFps,
    };
}

function sourceFrameForLocal(info, localFrame) {
    const local = clamp(Math.round(Number(localFrame) || 0), 0, Math.max(0, Number(info?.frames || 1) - 1));
    const count = Math.max(1, Number(info?.frames || 1));
    const start = Number(info?.source_start_frame || 0);
    const end = Number(info?.source_end_frame ?? start);
    if (count <= 1 || end <= start) return start;
    return start + roundLikePython(local * (end - start) / (count - 1));
}

function localFrameForSource(info, sourceFrame) {
    const count = Math.max(1, Number(info?.frames || 1));
    const start = Number(info?.source_start_frame || 0);
    const end = Number(info?.source_end_frame ?? start);
    if (count <= 1 || end <= start) return 0;
    return roundLikePython((Number(sourceFrame) - start) * (count - 1) / (end - start));
}

async function fetchPreview(payload, route) {
    const response = await api.fetchApi(route, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "SAM3.1 preview failed");
    return result;
}

function parseJson(value, fallback) {
    try {
        const parsed = JSON.parse(String(value || ""));
        return parsed ?? fallback;
    } catch {
        return fallback;
    }
}

function normalizePoints(value) {
    const parsed = parseJson(value, []);
    const list = Array.isArray(parsed) ? parsed : parsed?.points;
    if (!Array.isArray(list)) return [];
    return list.flatMap((point) => {
        if (Array.isArray(point) && point.length >= 2) {
            return [{ x: clamp(Number(point[0]) || 0, 0, 1), y: clamp(Number(point[1]) || 0, 0, 1), label: Number(point[2]) === 0 ? 0 : 1 }];
        }
        if (!point || typeof point !== "object") return [];
        return [{
            x: clamp(Number(point.x) || 0, 0, 1),
            y: clamp(Number(point.y) || 0, 0, 1),
            label: Number(point.label) === 0 || String(point.label).toLowerCase() === "negative" ? 0 : 1,
        }];
    });
}

function normalizeBox(value) {
    const parsed = parseJson(value, null);
    const box = Array.isArray(parsed) ? parsed[0] : parsed;
    if (Array.isArray(box) && box.length >= 4) {
        return { x: clamp(Number(box[0]) || 0, 0, 1), y: clamp(Number(box[1]) || 0, 0, 1), w: clamp(Number(box[2]) || 0, 0, 1), h: clamp(Number(box[3]) || 0, 0, 1) };
    }
    if (!box || typeof box !== "object") return null;
    const w = Number(box.w ?? box.width);
    const h = Number(box.h ?? box.height);
    if (!Number.isFinite(w) || !Number.isFinite(h)) return null;
    return { x: clamp(Number(box.x) || 0, 0, 1), y: clamp(Number(box.y) || 0, 0, 1), w: clamp(w, 0, 1), h: clamp(h, 0, 1) };
}

function addStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .cs-vseg-dialog { width: min(980px, 94vw); max-width: none; max-height: 92vh; overflow: auto; padding: 0; border: 1px solid #343943; border-radius: 10px; background: #17191e; color: #e6e9ef; box-shadow: 0 22px 80px #000b; }
      .cs-vseg-dialog::backdrop { background: #050609b8; }
      .cs-vseg-shell { display: grid; gap: 12px; padding: 16px; font: 13px/1.35 system-ui, sans-serif; }
      .cs-vseg-head, .cs-vseg-row, .cs-vseg-actions { display: flex; align-items: center; gap: 9px; }
      .cs-vseg-head { justify-content: space-between; }
      .cs-vseg-title { margin: 0; font-size: 17px; }
      .cs-vseg-muted { color: #9da5b4; }
      .cs-vseg-stage { position: relative; width: 100%; min-height: 180px; background: #08090b; border: 1px solid #343943; border-radius: 6px; overflow: hidden; }
      .cs-vseg-stage video { display: block; width: 100%; height: auto; max-height: 58vh; background: #08090b; }
      .cs-vseg-stage .cs-vseg-mask-preview { display: none; position: absolute; z-index: 1; inset: 0; width: 100%; height: 100%; object-fit: contain; background: #08090b; }
      .cs-vseg-stage canvas { position: absolute; z-index: 2; inset: 0; width: 100%; height: 100%; touch-action: none; cursor: crosshair; }
      .cs-vseg-controls { display: grid; grid-template-columns: auto minmax(100px, 1fr) 90px auto; align-items: center; gap: 8px; }
      .cs-vseg-step-buttons { display: flex; gap: 5px; }
      .cs-vseg-step-buttons .cs-vseg-button { width: 38px; padding-inline: 0; }
      .cs-vseg-controls input[type=range] { width: 100%; }
      .cs-vseg-fields { display: grid; grid-template-columns: 150px minmax(0, 1fr) auto; gap: 9px; align-items: end; }
      .cs-vseg-field { display: grid; gap: 5px; color: #9da5b4; }
      .cs-vseg-field input, .cs-vseg-field select { box-sizing: border-box; width: 100%; min-height: 31px; border: 1px solid #424956; border-radius: 5px; padding: 6px 8px; background: #20232a; color: #f2f4f7; }
      .cs-vseg-button { min-height: 31px; border: 1px solid #424956; border-radius: 5px; padding: 6px 10px; background: #20232a; color: #f2f4f7; cursor: pointer; }
      .cs-vseg-button:hover { border-color: #6aa9df; }
      .cs-vseg-button.active { background: #317ec4; border-color: #6db6ee; }
      .cs-vseg-point-menu { position: fixed; z-index: 20; display: grid; min-width: 190px; padding: 5px; gap: 3px; border: 1px solid #424956; border-radius: 6px; background: #20232a; box-shadow: 0 10px 32px #000b; }
      .cs-vseg-point-menu button { border: 0; border-radius: 4px; padding: 7px 9px; text-align: left; background: transparent; color: #f2f4f7; cursor: pointer; }
      .cs-vseg-point-menu button:hover { background: #317ec4; }
      .cs-vseg-actions { justify-content: flex-end; }
      .cs-vseg-preview-status { flex: 1; min-width: 0; color: #9da5b4; }
      .cs-vseg-preview-button { border-color: #348f85; }
      .cs-vseg-apply { background: #317ec4; border-color: #4b9de8; }
      .cs-vseg-prompt-note { min-height: 18px; color: #9da5b4; }
      @media (max-width: 640px) { .cs-vseg-fields { grid-template-columns: 1fr; } .cs-vseg-controls { grid-template-columns: auto 1fr auto; } .cs-vseg-controls .cs-vseg-frame-count { grid-column: 1 / -1; } }
    `;
    document.head.append(style);
}

async function openSelector(node, config) {
    const names = { frame: "anchor_frame", mode: "selection_mode", points: "points", bbox: "bbox", semantic: "semantic_prompt", ...(config.widgets || {}) };
    let source = connectedVideoSource(node, config.videoInputs || ["images", "video_input"]);
    if (!source) {
        try {
            source = await fetchCachedSource(node);
        } catch (error) {
            app.canvas?.prompt?.(error.message, "");
            return;
        }
        if (!source) {
            app.canvas?.prompt?.("Run the workflow once to cache the connected video input before opening the selector", "");
            return;
        }
    }
    const filename = source.filename;
    const sourceLabel = source.label || filename;
    addStyles();
    const dialog = document.createElement("dialog");
    dialog.className = "cs-vseg-dialog";
    const modeOptions = config.modes || [{ value: "points", label: "Points" }, { value: "bbox", label: "Bounding box" }];
    const modeMarkup = modeOptions.map((item) => `<option value="${item.value}">${item.label}</option>`).join("");
    const semanticMarkup = config.semantic ? `<label class="cs-vseg-field cs-vseg-semantic-field">Semantic prompt<input class="cs-vseg-semantic" type="text" placeholder="person, hair, dress"></label>` : "";
    dialog.innerHTML = `
      <div class="cs-vseg-shell">
        <div class="cs-vseg-head"><div><h2 class="cs-vseg-title">${config.title || "Video Selector"}</h2><div class="cs-vseg-muted cs-vseg-file"></div></div><button class="cs-vseg-button cs-vseg-close" type="button" aria-label="Close">&times;</button></div>
        <div class="cs-vseg-stage"><video controls muted playsinline preload="metadata"></video><img class="cs-vseg-mask-preview" alt="Current frame segmentation preview"><canvas></canvas></div>
        <div class="cs-vseg-controls"><div class="cs-vseg-step-buttons"><button class="cs-vseg-button cs-vseg-prev" type="button">|&lt;</button><button class="cs-vseg-button cs-vseg-next" type="button">&gt;|</button></div><input class="cs-vseg-slider" type="range" min="0" max="0" step="1" value="0"><input class="cs-vseg-frame-input" type="number" min="0" step="1" value="0"><span class="cs-vseg-frame-count cs-vseg-muted">0 / 0</span></div>
        <div class="cs-vseg-row"><button class="cs-vseg-button cs-vseg-clear" type="button">Clear prompt</button></div>
        <div class="cs-vseg-fields"><label class="cs-vseg-field">Mode<select class="cs-vseg-mode">${modeMarkup}</select></label>${semanticMarkup}<div class="cs-vseg-prompt-note"></div></div>
        <div class="cs-vseg-actions"><span class="cs-vseg-preview-status"></span><button class="cs-vseg-button cs-vseg-preview-button" type="button">Preview current frame</button><button class="cs-vseg-button cs-vseg-cancel" type="button">Cancel</button><button class="cs-vseg-button cs-vseg-apply" type="button">Apply to node</button></div>
      </div>`;
    document.body.append(dialog);

    const video = dialog.querySelector("video");
    const previewImage = dialog.querySelector(".cs-vseg-mask-preview");
    const canvas = dialog.querySelector("canvas");
    const context = canvas.getContext("2d");
    const stage = dialog.querySelector(".cs-vseg-stage");
    const slider = dialog.querySelector(".cs-vseg-slider");
    const frameInput = dialog.querySelector(".cs-vseg-frame-input");
    const frameCount = dialog.querySelector(".cs-vseg-frame-count");
    const modeSelect = dialog.querySelector(".cs-vseg-mode");
    const semanticInput = dialog.querySelector(".cs-vseg-semantic");
    const semanticField = dialog.querySelector(".cs-vseg-semantic-field");
    const note = dialog.querySelector(".cs-vseg-prompt-note");
    const previewStatus = dialog.querySelector(".cs-vseg-preview-status");
    const previewButton = dialog.querySelector(".cs-vseg-preview-button");
    let info = null;
    let frame = Math.max(0, Number(widget(node, names.frame)?.value || 0));
    let points = normalizePoints(widget(node, names.points)?.value);
    let box = normalizeBox(widget(node, names.bbox)?.value);
    let drag = null;
    let pointMenu = null;

    dialog.querySelector(".cs-vseg-file").textContent = sourceLabel;
    if (semanticInput) semanticInput.value = String(widget(node, names.semantic)?.value || "");
    modeSelect.value = String(widget(node, names.mode)?.value || modeOptions[0]?.value || "points");
    video.src = source.url || videoUrl(filename);

    function hidePreview() {
        previewImage.style.display = "none";
        previewImage.removeAttribute("src");
        previewStatus.textContent = "";
    }

    function updateMode() {
        const mode = modeSelect.value;
        if (semanticField) semanticField.style.display = mode === "semantic" ? "grid" : "none";
        note.textContent = config.note?.[mode] || (mode === "bbox" ? "Drag to draw a box; drag its corners or edges to resize it." : "Left-click adds or drags a positive/negative point; right-click adds a negative point or edits an existing point.");
        updateCanvasCursor();
        hidePreview();
        redraw();
    }

    function resizeCanvas() {
        const rect = video.getBoundingClientRect();
        const width = Math.max(1, Math.round(rect.width));
        const height = Math.max(1, Math.round(rect.height));
        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
        }
        redraw();
    }

    function drawPoint(point) {
        const x = point.x * canvas.width;
        const y = point.y * canvas.height;
        context.beginPath();
        context.arc(x, y, 7, 0, Math.PI * 2);
        context.fillStyle = point.label === 1 ? "#4dd0c2" : "#ff7a87";
        context.fill();
        context.lineWidth = 2;
        context.strokeStyle = "#081019";
        context.stroke();
    }

    function drawBoxHandles() {
        const left = box.x * canvas.width;
        const top = box.y * canvas.height;
        const right = (box.x + box.w) * canvas.width;
        const bottom = (box.y + box.h) * canvas.height;
        context.fillStyle = "#f7b955";
        context.strokeStyle = "#17191e";
        context.lineWidth = 1;
        for (const [x, y] of [[left, top], [right, top], [left, bottom], [right, bottom]]) {
            context.beginPath();
            context.rect(x - 4, y - 4, 8, 8);
            context.fill();
            context.stroke();
        }
    }

    function redraw() {
        context.clearRect(0, 0, canvas.width, canvas.height);
        if (modeSelect.value === "points") points.forEach(drawPoint);
        if (modeSelect.value === "bbox" && box) {
            context.fillStyle = "rgba(247, 185, 85, 0.12)";
            context.fillRect(box.x * canvas.width, box.y * canvas.height, box.w * canvas.width, box.h * canvas.height);
            context.strokeStyle = "#f7b955";
            context.lineWidth = 3;
            context.strokeRect(box.x * canvas.width, box.y * canvas.height, box.w * canvas.width, box.h * canvas.height);
            drawBoxHandles();
        }
    }

    function setFrame(nextFrame, seek = true) {
        const maxFrame = Math.max(0, Number(info?.frames || 1) - 1);
        frame = clamp(Math.round(Number(nextFrame) || 0), 0, maxFrame);
        slider.value = String(frame);
        frameInput.value = String(frame);
        frameCount.textContent = `${frame} / ${maxFrame}`;
        hidePreview();
        if (seek && info?.source_fps && Number.isFinite(video.duration)) {
            video.currentTime = sourceFrameForLocal(info, frame) / info.source_fps;
        }
    }

    function eventPosition(event) {
        const rect = canvas.getBoundingClientRect();
        return { x: clamp((event.clientX - rect.left) / rect.width, 0, 1), y: clamp((event.clientY - rect.top) / rect.height, 0, 1) };
    }

    function pointIndexAt(position) {
        const rect = canvas.getBoundingClientRect();
        let hit = -1;
        let distance = 12;
        points.forEach((point, index) => {
            const dx = (point.x - position.x) * rect.width;
            const dy = (point.y - position.y) * rect.height;
            const nextDistance = Math.hypot(dx, dy);
            if (nextDistance <= distance) {
                hit = index;
                distance = nextDistance;
            }
        });
        return hit;
    }

    function closePointMenu() {
        pointMenu?.remove();
        pointMenu = null;
    }

    function showPointMenu(index, event) {
        closePointMenu();
        const point = points[index];
        const changeLabel = point.label === 1 ? "Change point to negative" : "Change point to positive";
        pointMenu = document.createElement("div");
        pointMenu.className = "cs-vseg-point-menu";
        pointMenu.innerHTML = `<button type="button" data-action="delete">Delete point</button><button type="button" data-action="toggle">${changeLabel}</button>`;
        dialog.append(pointMenu);
        const menuWidth = pointMenu.offsetWidth;
        const menuHeight = pointMenu.offsetHeight;
        pointMenu.style.left = `${clamp(event.clientX, 8, window.innerWidth - menuWidth - 8)}px`;
        pointMenu.style.top = `${clamp(event.clientY, 8, window.innerHeight - menuHeight - 8)}px`;
        pointMenu.addEventListener("click", (menuEvent) => {
            const action = menuEvent.target.closest("button")?.dataset.action;
            if (action === "delete") points.splice(index, 1);
            if (action === "toggle" && points[index]) points[index].label = points[index].label === 1 ? 0 : 1;
            hidePreview();
            closePointMenu();
            redraw();
        });
    }

    function boxTargetAt(position) {
        if (!box) return null;
        const rect = canvas.getBoundingClientRect();
        const px = position.x * rect.width;
        const py = position.y * rect.height;
        const left = box.x * rect.width;
        const top = box.y * rect.height;
        const right = (box.x + box.w) * rect.width;
        const bottom = (box.y + box.h) * rect.height;
        const cornerRadius = 11;
        const edgeRadius = 7;
        const corners = [
            ["nw", left, top], ["ne", right, top],
            ["sw", left, bottom], ["se", right, bottom],
        ];
        for (const [target, x, y] of corners) {
            if (Math.hypot(px - x, py - y) <= cornerRadius) return target;
        }
        if (px >= left - edgeRadius && px <= right + edgeRadius) {
            if (Math.abs(py - top) <= edgeRadius) return "n";
            if (Math.abs(py - bottom) <= edgeRadius) return "s";
        }
        if (py >= top - edgeRadius && py <= bottom + edgeRadius) {
            if (Math.abs(px - left) <= edgeRadius) return "w";
            if (Math.abs(px - right) <= edgeRadius) return "e";
        }
        if (px > left && px < right && py > top && py < bottom) return "move";
        return null;
    }

    function boxCursor(target) {
        if (target === "nw" || target === "se") return "nwse-resize";
        if (target === "ne" || target === "sw") return "nesw-resize";
        if (target === "n" || target === "s") return "ns-resize";
        if (target === "e" || target === "w") return "ew-resize";
        if (target === "move") return "move";
        return "crosshair";
    }

    function updateCanvasCursor(position = null) {
        if (modeSelect.value === "bbox") {
            canvas.style.cursor = boxCursor(position ? boxTargetAt(position) : null);
        } else if (modeSelect.value === "points") {
            canvas.style.cursor = position && pointIndexAt(position) >= 0 ? "move" : "crosshair";
        } else {
            canvas.style.cursor = "default";
        }
    }

    function updateBoxDrag(position) {
        const original = drag.original;
        if (drag.target === "draw") {
            box = {
                x: Math.min(drag.start.x, position.x),
                y: Math.min(drag.start.y, position.y),
                w: Math.abs(position.x - drag.start.x),
                h: Math.abs(position.y - drag.start.y),
            };
            return;
        }
        if (drag.target === "move") {
            box = {
                ...original,
                x: clamp(original.x + position.x - drag.start.x, 0, 1 - original.w),
                y: clamp(original.y + position.y - drag.start.y, 0, 1 - original.h),
            };
            return;
        }
        let left = original.x;
        let top = original.y;
        let right = original.x + original.w;
        let bottom = original.y + original.h;
        if (drag.target.includes("w")) left = position.x;
        if (drag.target.includes("e")) right = position.x;
        if (drag.target.includes("n")) top = position.y;
        if (drag.target.includes("s")) bottom = position.y;
        box = {
            x: Math.min(left, right),
            y: Math.min(top, bottom),
            w: Math.abs(right - left),
            h: Math.abs(bottom - top),
        };
    }

    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    canvas.addEventListener("pointerdown", (event) => {
        if (modeSelect.value === "points") {
            if (event.button !== 0 && event.button !== 2) return;
            event.preventDefault();
            const position = eventPosition(event);
            const index = pointIndexAt(position);
            if (event.button === 2) {
                if (index >= 0) {
                    showPointMenu(index, event);
                    return;
                }
                points.push({ ...position, label: 0 });
            } else {
                closePointMenu();
                if (index >= 0) {
                    drag = {
                        kind: "point",
                        pointIndex: index,
                        start: position,
                        original: { ...points[index] },
                        moved: false,
                        pointerId: event.pointerId,
                    };
                    canvas.setPointerCapture?.(event.pointerId);
                    canvas.style.cursor = "move";
                    return;
                }
                points.push({ ...position, label: 1 });
            }
            hidePreview();
            redraw();
            return;
        }
        if (modeSelect.value === "bbox" && event.button === 0) {
            event.preventDefault();
            closePointMenu();
            const position = eventPosition(event);
            const target = boxTargetAt(position);
            drag = {
                kind: "bbox",
                target: target || "draw",
                start: position,
                original: box ? { ...box } : null,
                previous: box ? { ...box } : null,
                pointerId: event.pointerId,
            };
            if (!target) box = { x: position.x, y: position.y, w: 0, h: 0 };
            hidePreview();
            canvas.setPointerCapture?.(event.pointerId);
            updateCanvasCursor(position);
            redraw();
        }
    });
    canvas.addEventListener("pointermove", (event) => {
        const position = eventPosition(event);
        if (!drag) {
            updateCanvasCursor(position);
            return;
        }
        if (drag.kind === "point") {
            const rect = canvas.getBoundingClientRect();
            const distance = Math.hypot(
                (position.x - drag.start.x) * rect.width,
                (position.y - drag.start.y) * rect.height,
            );
            if (!drag.moved && distance < 2) return;
            drag.moved = true;
            const point = points[drag.pointIndex];
            if (point) points[drag.pointIndex] = { ...point, x: position.x, y: position.y };
            canvas.style.cursor = "move";
            hidePreview();
            redraw();
            return;
        }
        updateBoxDrag(position);
        canvas.style.cursor = boxCursor(drag.target);
        hidePreview();
        redraw();
    });
    canvas.addEventListener("pointerleave", () => { if (!drag) updateCanvasCursor(); });
    canvas.addEventListener("pointerup", (event) => {
        if (!drag) return;
        if (drag.kind === "bbox" && (box.w * canvas.width < 3 || box.h * canvas.height < 3)) {
            box = drag.previous;
        }
        canvas.releasePointerCapture?.(drag.pointerId);
        drag = null;
        updateCanvasCursor(eventPosition(event));
        redraw();
    });
    canvas.addEventListener("pointercancel", () => {
        if (drag?.kind === "point" && points[drag.pointIndex]) points[drag.pointIndex] = drag.original;
        if (drag?.kind === "bbox") box = drag.previous;
        drag = null;
        updateCanvasCursor();
        redraw();
    });
    dialog.addEventListener("pointerdown", (event) => {
        if (pointMenu && !pointMenu.contains(event.target)) closePointMenu();
    }, true);
    dialog.querySelector(".cs-vseg-clear").addEventListener("click", () => { points = []; box = null; hidePreview(); redraw(); });
    semanticInput?.addEventListener("input", hidePreview);
    modeSelect.addEventListener("change", updateMode);
    slider.addEventListener("input", () => setFrame(slider.value));
    frameInput.addEventListener("change", () => setFrame(frameInput.value));
    dialog.querySelector(".cs-vseg-prev").addEventListener("click", () => setFrame(frame - 1));
    dialog.querySelector(".cs-vseg-next").addEventListener("click", () => setFrame(frame + 1));
    previewButton.addEventListener("click", async () => {
        previewButton.disabled = true;
        previewStatus.textContent = config.previewLabel || "Running segmentation on this frame...";
        video.pause();
        try {
            const result = await config.preview({
                node,
                filename,
                frame,
                previewFrame: sourceFrameForLocal(info, frame),
                info,
                mode: modeSelect.value,
                semanticPrompt: semanticInput?.value.trim() || "",
                points,
                box,
                fetchPreview: (payload) => fetchPreview({ ...payload, source_token: source.token || "" }, config.previewRoute),
            });
            previewImage.src = result.image;
            previewImage.style.display = "block";
            previewStatus.textContent = `Frame ${result.frame} · mask ${(Number(result.mask_area || 0) * 100).toFixed(1)}%`;
        } catch (error) {
            previewStatus.textContent = error.message;
        } finally {
            previewButton.disabled = false;
        }
    });
    video.addEventListener("timeupdate", () => {
        if (!info?.source_fps || video.seeking) return;
        const sourceFrame = Math.round(video.currentTime * info.source_fps);
        const first = Number(info.source_start_frame || 0);
        const last = Number(info.source_end_frame ?? first);
        if (sourceFrame < first || sourceFrame > last) {
            const local = sourceFrame < first ? 0 : Math.max(0, Number(info.frames || 1) - 1);
            setFrame(local);
            return;
        }
        setFrame(localFrameForSource(info, sourceFrame), false);
    });
    video.addEventListener("loadedmetadata", () => { resizeCanvas(); setFrame(frame); });
    new ResizeObserver(resizeCanvas).observe(stage);

    const close = () => { video.pause(); closePointMenu(); dialog.close(); dialog.remove(); };
    dialog.querySelector(".cs-vseg-close").addEventListener("click", close);
    dialog.querySelector(".cs-vseg-cancel").addEventListener("click", close);
    dialog.querySelector(".cs-vseg-apply").addEventListener("click", () => {
        if (config.apply) {
            config.apply({ node, frame, mode: modeSelect.value, semanticPrompt: semanticInput?.value.trim() || "", points, box, info, setWidgetValue });
        } else {
            setWidgetValue(node, names.mode, modeSelect.value);
            setWidgetValue(node, names.frame, frame);
            setWidgetValue(node, names.semantic, semanticInput?.value.trim() || "");
            setWidgetValue(node, names.points, JSON.stringify(points));
            setWidgetValue(node, names.bbox, JSON.stringify(box || {}));
        }
        node.graph?.setDirtyCanvas(true, true);
        close();
    });
    dialog.addEventListener("cancel", close);
    (source.info ? Promise.resolve(source.info) : fetchInfo(filename)).then((result) => {
        info = prepareInputTimeline(source, result);
        dialog.querySelector(".cs-vseg-file").textContent = `${sourceLabel} · ${info.frames} input frames`;
        const maxFrame = Math.max(0, Number(info.frames || 1) - 1);
        slider.max = String(maxFrame);
        frameInput.max = String(maxFrame);
        setFrame(frame);
        updateMode();
        resizeCanvas();
    }).catch((error) => { note.textContent = error.message; });
    dialog.showModal();
}

export function registerVideoSelector(config) {
    app.registerExtension({
        name: config.extensionName,
        async beforeRegisterNodeDef(nodeType, nodeData) {
            if (nodeData?.name !== config.nodeId) return;
            const original = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                original?.apply(this, arguments);
                removeObsoleteInputs(this, config.removeInputs || []);
                removeObsoleteWidgets(this, config.removeWidgets || []);
                const button = this.addWidget("button", "Open Selector", "", () => openSelector(this, config));
                button.name = "Open Selector";
                button.label = "Open Selector";
                button.options = { ...(button.options || {}), serialize: false };
                this.setSize?.([390, Math.max(360, this.computeSize?.()[1] || 360)]);
            };
        },
        loadedGraphNode(node) {
            if (node?.type !== config.nodeId) return;
            removeObsoleteInputs(node, config.removeInputs || []);
            removeObsoleteWidgets(node, config.removeWidgets || []);
            node.setSize?.([node.size?.[0] || 390, node.computeSize?.()[1] || node.size?.[1] || 360]);
        },
    });
}
