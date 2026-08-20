import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_ID = "CS_Video_Segment_SAM3";
const STYLE_ID = "cinestyle-video-segment-style";

function widget(node, name) {
    return node.widgets?.find((item) => item.name === name);
}

function setWidgetValue(node, name, value) {
    const target = widget(node, name);
    if (!target) return;
    target.value = value;
    target.callback?.(value);
}

function removeObsoleteInputs(node) {
    for (const name of ["clip", "conditioning"]) {
        const index = node.inputs?.findIndex((input) => input.name === name) ?? -1;
        if (index >= 0) node.removeInput?.(index);
    }
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function videoUrl(filename) {
    const params = new URLSearchParams({
        filename,
        type: "input",
        subfolder: "",
        t: String(Date.now()),
    });
    return api.apiURL(`/view?${params.toString()}`);
}

async function fetchInfo(filename) {
    const params = new URLSearchParams({ filename });
    const response = await api.fetchApi(`/cinestyle/video-info?${params.toString()}`);
    if (!response.ok) throw new Error(await response.text());
    return await response.json();
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

async function fetchPreview(payload) {
    const response = await api.fetchApi("/cinestyle/video-segment-preview", {
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

function openSelector(node) {
    const filename = String(widget(node, "video")?.value || "").trim();
    if (!filename) {
        app.canvas?.prompt?.("Choose a video in the node before opening the selector", "");
        return;
    }
    addStyles();
    const dialog = document.createElement("dialog");
    dialog.className = "cs-vseg-dialog";
    dialog.innerHTML = `
      <div class="cs-vseg-shell">
        <div class="cs-vseg-head"><div><h2 class="cs-vseg-title">SAM3.1 Video Selector</h2><div class="cs-vseg-muted cs-vseg-file"></div></div><button class="cs-vseg-button cs-vseg-close" type="button" aria-label="Close">&times;</button></div>
        <div class="cs-vseg-stage"><video controls muted playsinline preload="metadata"></video><img class="cs-vseg-mask-preview" alt="Current frame segmentation preview"><canvas></canvas></div>
        <div class="cs-vseg-controls"><div class="cs-vseg-step-buttons"><button class="cs-vseg-button cs-vseg-prev" type="button">|&lt;</button><button class="cs-vseg-button cs-vseg-next" type="button">&gt;|</button></div><input class="cs-vseg-slider" type="range" min="0" max="0" step="1" value="0"><input class="cs-vseg-frame-input" type="number" min="0" step="1" value="0"><span class="cs-vseg-frame-count cs-vseg-muted">0 / 0</span></div>
        <div class="cs-vseg-row"><button class="cs-vseg-button cs-vseg-clear" type="button">Clear prompt</button></div>
        <div class="cs-vseg-fields"><label class="cs-vseg-field">Mode<select class="cs-vseg-mode"><option value="points">Points</option><option value="bbox">Bounding box</option><option value="semantic">Semantic</option></select></label><label class="cs-vseg-field cs-vseg-semantic-field">Semantic prompt<input class="cs-vseg-semantic" type="text" placeholder="person, hair, dress"></label><div class="cs-vseg-prompt-note"></div></div>
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
    let frame = Math.max(0, Number(widget(node, "anchor_frame")?.value || 0));
    let points = normalizePoints(widget(node, "points")?.value);
    let box = normalizeBox(widget(node, "bbox")?.value);
    let drag = null;
    let pointMenu = null;

    dialog.querySelector(".cs-vseg-file").textContent = filename;
    semanticInput.value = String(widget(node, "semantic_prompt")?.value || "");
    modeSelect.value = String(widget(node, "selection_mode")?.value || "points");
    video.src = videoUrl(filename);

    function hidePreview() {
        previewImage.style.display = "none";
        previewImage.removeAttribute("src");
        previewStatus.textContent = "";
    }

    function updateMode() {
        const mode = modeSelect.value;
        semanticField.style.display = mode === "semantic" ? "grid" : "none";
        note.textContent = mode === "semantic" ? "Enter the object description, then preview the current frame." : mode === "bbox" ? "Drag to draw a box; drag its corners or edges to resize it." : "Left-click adds or drags a positive/negative point; right-click adds a negative point or edits an existing point.";
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
        if (seek && info?.fps && Number.isFinite(video.duration)) video.currentTime = frame / info.fps;
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
    semanticInput.addEventListener("input", hidePreview);
    modeSelect.addEventListener("change", updateMode);
    slider.addEventListener("input", () => setFrame(slider.value));
    frameInput.addEventListener("change", () => setFrame(frameInput.value));
    dialog.querySelector(".cs-vseg-prev").addEventListener("click", () => setFrame(frame - 1));
    dialog.querySelector(".cs-vseg-next").addEventListener("click", () => setFrame(frame + 1));
    previewButton.addEventListener("click", async () => {
        previewButton.disabled = true;
        previewStatus.textContent = "Running SAM3.1 on this frame...";
        video.pause();
        try {
            const result = await fetchPreview({
                video: filename,
                frame,
                mode: modeSelect.value,
                semantic_prompt: semanticInput.value.trim(),
                points: JSON.stringify(points),
                bbox: JSON.stringify(box || {}),
                threshold: Number(widget(node, "threshold")?.value ?? 0.5),
                model_source: connectedModelSource(node),
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
        if (!info?.fps || video.seeking) return;
        setFrame(Math.round(video.currentTime * info.fps), false);
    });
    video.addEventListener("loadedmetadata", () => { resizeCanvas(); setFrame(frame); });
    new ResizeObserver(resizeCanvas).observe(stage);

    const close = () => { video.pause(); closePointMenu(); dialog.close(); dialog.remove(); };
    dialog.querySelector(".cs-vseg-close").addEventListener("click", close);
    dialog.querySelector(".cs-vseg-cancel").addEventListener("click", close);
    dialog.querySelector(".cs-vseg-apply").addEventListener("click", () => {
        setWidgetValue(node, "selection_mode", modeSelect.value);
        setWidgetValue(node, "anchor_frame", frame);
        setWidgetValue(node, "semantic_prompt", semanticInput.value.trim());
        setWidgetValue(node, "points", JSON.stringify(points));
        setWidgetValue(node, "bbox", JSON.stringify(box || {}));
        node.graph?.setDirtyCanvas(true, true);
        close();
    });
    dialog.addEventListener("cancel", close);
    fetchInfo(filename).then((result) => {
        info = result;
        const maxFrame = Math.max(0, Number(result.frames || 1) - 1);
        slider.max = String(maxFrame);
        frameInput.max = String(maxFrame);
        setFrame(frame);
        updateMode();
        resizeCanvas();
    }).catch((error) => { note.textContent = error.message; });
    dialog.showModal();
}

app.registerExtension({
    name: "CineStyle.VideoSegmentSAM3",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_ID) return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            original?.apply(this, arguments);
            removeObsoleteInputs(this);
            const button = this.addWidget("button", "Open Selector", "", () => openSelector(this));
            button.name = "Open Selector";
            button.label = "Open Selector";
            button.options = { ...(button.options || {}), serialize: false };
            this.setSize?.([390, Math.max(360, this.computeSize?.()[1] || 360)]);
        };
    },
    loadedGraphNode(node) {
        if (node?.type !== NODE_ID) return;
        removeObsoleteInputs(node);
        node.setSize?.([node.size?.[0] || 390, node.computeSize?.()[1] || node.size?.[1] || 360]);
    },
});
