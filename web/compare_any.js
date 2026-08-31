import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_ID = "CS_Compare_Any";
const STYLE_ID = "cinestyle-compare-any-style";
const STATE = Symbol("cinestyle-compare-any-state");
const WIDGET = Symbol("cinestyle-compare-any-widget");
const DEFAULT_BOTTOM_HEIGHT = 306;
const MIN_BOTTOM_HEIGHT = 180;
const MAX_VIEWPORT_HEIGHT = 1200;

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function formatFrameCode(frame, fps) {
    const frameRate = Math.max(1, Math.round(Number(fps) || 1));
    const totalFrames = Math.max(0, Math.floor(Number(frame) || 0));
    const minuteFrames = frameRate * 60;
    const minutes = Math.floor(totalFrames / minuteFrames);
    const seconds = Math.floor(totalFrames / frameRate) % 60;
    const frameInSecond = totalFrames % frameRate;
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}:${String(frameInSecond).padStart(2, "0")}`;
}

function latestPayload(output) {
    const values = output?.compare_any;
    if (!Array.isArray(values) || !values.length) return {};
    const payload = values[values.length - 1];
    return payload && typeof payload === "object" ? payload : {};
}

function mediaUrl(source) {
    const value = String(source?.video_url || "");
    return value ? api.apiURL(`${value}${value.includes("?") ? "&" : "?"}t=${Date.now()}`) : "";
}

function addStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .cs-compare-any-shell{position:relative;display:grid;gap:8px;box-sizing:border-box;width:100%;padding:8px;background:var(--comfy-input-bg,#17191e);color:var(--input-text,#e6e9ef);font:12px/1.35 system-ui,sans-serif}
      .cs-compare-any-head,.cs-compare-any-controls{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
      .cs-compare-any-controls[hidden]{display:none}
      .cs-compare-any-head{justify-content:space-between;color:var(--descrip-text,#9da5b4)}
      .cs-compare-any-title{font-weight:600;color:var(--input-text,#e6e9ef)}
      .cs-compare-any-status{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:75%;color:var(--descrip-text,#9da5b4)}
      .cs-compare-any-grid{display:grid;grid-template-columns:1fr;grid-template-rows:var(--cs-compare-bottom-height,306px);gap:6px;min-height:0}
      .cs-compare-any-viewport{position:relative;min-width:0;min-height:0;overflow:hidden;border:1px solid var(--border-color,#3c424d);border-radius:5px;background:#08090b}
      .cs-compare-any-viewport canvas{display:block;width:100%;height:100%;min-height:150px}
      .cs-compare-any-compare{grid-column:1/-1}
      .cs-compare-any-divider{position:absolute;z-index:3;top:0;bottom:0;left:var(--compare-position,50%);width:2px;transform:translateX(-1px);background:#f4f7fb;box-shadow:0 0 0 1px #11141980;cursor:ew-resize;touch-action:none}
      .cs-compare-any-divider::before{content:"";position:absolute;top:50%;left:50%;width:22px;height:22px;transform:translate(-50%,-50%);border:2px solid #f4f7fb;border-radius:50%;background:#20232a;box-shadow:0 2px 8px #000b}
      .cs-compare-any-divider::after{content:"↔";position:absolute;top:50%;left:50%;transform:translate(-50%,-53%);color:#f4f7fb;font-size:13px;line-height:1}
      .cs-compare-any-diff{display:none;height:100%;overflow:auto;padding:5px 0;font:11px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace}
      .cs-compare-any-diff-row{display:grid;grid-template-columns:24px minmax(0,1fr) 24px minmax(0,1fr);min-height:16px}
      .cs-compare-any-diff-prefix{padding:0 5px;color:#707987;text-align:center;user-select:none}
      .cs-compare-any-diff-cell{min-width:0;padding:0 6px;white-space:pre-wrap;overflow-wrap:anywhere}
      .cs-compare-any-diff-row.equal .cs-compare-any-diff-cell{color:#7f8794}
      .cs-compare-any-diff-row.delete .cs-compare-any-diff-cell.a{background:#60272b80;color:#ffc5c8}
      .cs-compare-any-diff-row.insert .cs-compare-any-diff-cell.b{background:#1e563680;color:#c6f4ce}
      .cs-compare-any-diff-row.replace .cs-compare-any-diff-cell.a{background:#60272b55;color:#ffc5c8}
      .cs-compare-any-diff-row.replace .cs-compare-any-diff-cell.b{background:#1e563655;color:#c6f4ce}
      .cs-compare-any-diff-part.equal{color:#7f8794}
      .cs-compare-any-diff-part.changed.a{background:#9c3b4580;color:#fff0f1}
      .cs-compare-any-diff-part.changed.b{background:#2c895080;color:#effff2}
      .cs-compare-any-button{min-height:26px;min-width:30px;border:1px solid var(--border-color,#424956);border-radius:4px;padding:4px 7px;background:#20232a;color:#e9edf3;cursor:pointer}
      .cs-compare-any-button:hover{border-color:#65aeea;background:#282d36}
      .cs-compare-any-button.playing{background:#317ec4;border-color:#6db6ee}
      .cs-compare-any-controls input[type=range]{flex:1;min-width:100px}
      .cs-compare-any-zoom-group{display:flex;align-items:center;gap:4px;margin-left:auto}
      .cs-compare-any-zoom-button{min-width:27px;padding-inline:5px;font-size:14px;line-height:1}
      .cs-compare-any-readout{min-width:72px;color:var(--descrip-text,#9da5b4);font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
      .cs-compare-any-readout-total{min-width:72px;color:var(--descrip-text,#9da5b4);font-variant-numeric:tabular-nums;white-space:nowrap}
      .cs-compare-any-error{position:absolute;z-index:4;inset:0;display:flex;align-items:center;justify-content:center;padding:18px;box-sizing:border-box;text-align:center;color:#ff9ba2;white-space:pre-wrap;pointer-events:none}
      .cs-compare-any-loading{position:absolute;z-index:6;inset:0;display:flex;align-items:center;justify-content:center;padding:18px;box-sizing:border-box;background:#08090bd9;color:#dce7f3;text-align:center;white-space:pre-wrap;pointer-events:none}
      .cs-compare-any-loading[hidden],.cs-compare-any-error[hidden]{display:none}
      .cs-compare-any-audio-button{min-width:48px}
      .cs-compare-any-media-only .cs-compare-any-diff{display:none}
      .cs-compare-any-diff-mode .cs-compare-any-diff{display:block}
      .cs-compare-any-diff-mode .cs-compare-any-viewport canvas,.cs-compare-any-diff-mode .cs-compare-any-divider{display:none}
      .cs-compare-any-diff-mode .cs-compare-any-viewport{background:#111419}
      .cs-compare-any-diff-mode .cs-compare-any-grid{grid-template-rows:var(--cs-compare-bottom-height,306px)}
      .cs-compare-any-diff-mode .cs-compare-any-compare{overflow:hidden}
      .cs-compare-any-resize-handle{position:absolute;z-index:5;right:8px;bottom:1px;left:8px;height:10px;cursor:ns-resize;touch-action:none}
      .cs-compare-any-resize-handle::after{content:"";position:absolute;top:4px;left:50%;width:34px;height:2px;border-radius:2px;transform:translateX(-50%);background:#8c96a5;opacity:.72}
      .cs-compare-any-resize-handle:hover::after{background:#b8d9f5;opacity:1}
      .cs-compare-any-compare.cs-compare-any-pan-ready{cursor:grab}
      .cs-compare-any-compare.cs-compare-any-pan-active{cursor:grabbing}
      @media(max-width:430px){.cs-compare-any-grid{grid-template-columns:1fr;grid-template-rows:var(--cs-compare-bottom-height,306px)}.cs-compare-any-compare{grid-column:1}}
    `;
    document.head.appendChild(style);
}

function containedRect(source, width, height, zoom = 1, panX = 0, panY = 0) {
    if (!source || !source.videoWidth || !source.videoHeight) return;
    const scale = Math.min(width / source.videoWidth, height / source.videoHeight);
    const drawWidth = source.videoWidth * scale * zoom;
    const drawHeight = source.videoHeight * scale * zoom;
    return {
        x: (width - drawWidth) * 0.5 + panX,
        y: (height - drawHeight) * 0.5 + panY,
        width: drawWidth,
        height: drawHeight,
    };
}

function drawContained(context, source, width, height, zoom = 1, panX = 0, panY = 0) {
    const rect = containedRect(source, width, height, zoom, panX, panY);
    if (!rect) return;
    context.drawImage(source, rect.x, rect.y, rect.width, rect.height);
}

function clampComparePan(state, width, height) {
    if (!state || state.compareZoom <= 1) {
        state.comparePanX = 0;
        state.comparePanY = 0;
        return;
    }
    const rectA = containedRect(state.videoA, width, height, state.compareZoom);
    const rectB = containedRect(state.videoB, width, height, state.compareZoom);
    const maxWidth = Math.max(rectA?.width || 0, rectB?.width || 0);
    const maxHeight = Math.max(rectA?.height || 0, rectB?.height || 0);
    const maxX = Math.max(0, (maxWidth - width) * 0.5);
    const maxY = Math.max(0, (maxHeight - height) * 0.5);
    state.comparePanX = clamp(Number(state.comparePanX) || 0, -maxX, maxX);
    state.comparePanY = clamp(Number(state.comparePanY) || 0, -maxY, maxY);
}

function setupCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.max(1, window.devicePixelRatio || 1);
    const width = Math.max(2, Math.round(rect.width * ratio));
    const height = Math.max(2, Math.round(rect.height * ratio));
    if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
    }
    const context = canvas.getContext("2d");
    if (!context) return null;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { context, width: Math.max(1, rect.width), height: Math.max(1, rect.height) };
}

function drawMedia(state) {
    if (!state || state.mode !== "media") return;
    const canvasCompare = setupCanvas(state.canvasCompare);
    if (!canvasCompare) return;
    const { context, width, height } = canvasCompare;
    clampComparePan(state, width, height);
    context.fillStyle = "#08090b";
    context.fillRect(0, 0, width, height);
    drawContained(context, state.videoB, width, height, state.compareZoom, state.comparePanX, state.comparePanY);
    const position = clamp(Number(state.comparePosition) || 50, 0, 100) / 100;
    context.save();
    context.beginPath();
    context.rect(0, 0, width * position, height);
    context.clip();
    drawContained(context, state.videoA, width, height, state.compareZoom, state.comparePanX, state.comparePanY);
    context.restore();
}

function applyViewportHeights(state) {
    state.shell.style.setProperty("--cs-compare-bottom-height", `${Math.round(state.bottomHeight)}px`);
}

function syncNodeHeight(state) {
    const node = state.node;
    if (!node) return;
    const gridHeight = state.bottomHeight;
    const targetHeight = Math.max(430, Math.round(state.baseNodeHeight + gridHeight - state.defaultGridHeight));
    const width = Math.max(560, Number(node.size?.[0]) || 560);
    if (Math.abs(Number(node.size?.[1] || 0) - targetHeight) >= 2) node.setSize?.([width, targetHeight]);
}

function setViewportHeight(state, requested) {
    state.bottomHeight = clamp(Math.round(requested), MIN_BOTTOM_HEIGHT, MAX_VIEWPORT_HEIGHT);
    state.bottomHeightManual = true;
    applyViewportHeights(state);
    syncNodeHeight(state);
    nodeGraphDirty(state.node);
}

function nodeGraphDirty(node) {
    node?.graph?.setDirtyCanvas?.(true, true);
}

function updateAutoBottomHeight(state) {
    if (state.bottomHeightManual || !state.grid) return;
    const width = Number(state.grid.clientWidth) || 0;
    if (width <= 0) return;
    const next = clamp(Math.round(width * 9 / 16), MIN_BOTTOM_HEIGHT, MAX_VIEWPORT_HEIGHT);
    if (Math.abs(next - state.bottomHeight) < 2) return;
    state.bottomHeight = next;
    applyViewportHeights(state);
    syncNodeHeight(state);
}

function attachHeightHandle(state, handle) {
    if (!handle) return;
    let drag = null;
    const finish = (event) => {
        if (!drag || event.pointerId !== drag.pointerId) return;
        drag = null;
        handle.releasePointerCapture?.(event.pointerId);
    };
    handle.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        event.preventDefault();
        event.stopPropagation();
        drag = { pointerId: event.pointerId, startY: event.clientY, startHeight: state.bottomHeight };
        handle.setPointerCapture?.(event.pointerId);
    });
    handle.addEventListener("pointermove", (event) => {
        if (!drag || event.pointerId !== drag.pointerId) return;
        event.preventDefault();
        setViewportHeight(state, drag.startHeight + event.clientY - drag.startY);
    });
    handle.addEventListener("pointerup", finish);
    handle.addEventListener("pointercancel", finish);
}

function setCompareZoom(state, requested) {
    state.compareZoom = clamp(Number(requested) || 1, 1, 4);
    if (state.compareZoom <= 1) {
        state.comparePanX = 0;
        state.comparePanY = 0;
    }
    const rect = state.canvasCompare?.getBoundingClientRect();
    if (rect) clampComparePan(state, rect.width, rect.height);
    state.compareViewport?.classList.toggle("cs-compare-any-pan-ready", state.compareZoom > 1);
    drawMedia(state);
}

function setAudioChoice(state, choice) {
    state.audioChoice = choice === "B" || choice === "MUTE" ? choice : "A";
    state.videoA.muted = state.audioChoice !== "A";
    state.videoB.muted = state.audioChoice !== "B";
    if (state.audioButton) {
        const muted = state.audioChoice === "MUTE";
        state.audioButton.textContent = muted ? "🔇" : `🔊 ${state.audioChoice}`;
        state.audioButton.title = muted ? "Mute audio" : `Play source ${state.audioChoice} audio`;
        state.audioButton.setAttribute("aria-label", muted ? "Mute audio" : `Play source ${state.audioChoice} audio`);
        state.audioButton.setAttribute("aria-pressed", muted ? "true" : "false");
    }
}

function clearMediaCanvases(state) {
    const prepared = setupCanvas(state.canvasCompare);
    if (!prepared) return;
    prepared.context.fillStyle = "#08090b";
    prepared.context.fillRect(0, 0, prepared.width, prepared.height);
}

function stopCacheProgress(state) {
    if (state.progressTimer != null) {
        clearTimeout(state.progressTimer);
        state.progressTimer = null;
    }
    state.progressSerial = (state.progressSerial || 0) + 1;
    if (state.loading) state.loading.hidden = true;
}

function beginCacheProgress(node) {
    const state = node?.[STATE];
    if (!state) return;
    stopCacheProgress(state);
    const serial = state.progressSerial;
    state.mode = "loading";
    state.shell.classList.remove("cs-compare-any-diff-mode");
    state.shell.classList.add("cs-compare-any-media-only");
    state.controls.hidden = true;
    state.error.hidden = true;
    state.canvasCompare.hidden = false;
    state.diff.hidden = true;
    state.divider.hidden = true;
    clearMediaCanvases(state);
    state.loading.hidden = false;
    state.loading.textContent = "Loading comparison cache 0%";

    const poll = async () => {
        if (state.disposed || state.progressSerial !== serial) return;
        try {
            const response = await api.fetchApi(`/cinestyle/compare-any-progress?node_id=${encodeURIComponent(String(node.id))}`);
            const result = await response.json().catch(() => ({}));
            if (state.progressSerial !== serial || state.disposed) return;
            const progress = clamp(Number(result.progress) || 0, 0, 100);
            if (result.status === "failed") {
                state.loading.textContent = String(result.message || "Unable to prepare comparison cache");
                state.progressTimer = setTimeout(() => stopCacheProgress(state), 3000);
                return;
            }
            state.loading.textContent = `${String(result.message || "Loading comparison cache")} ${progress}%`;
            state.loading.hidden = false;
            if (result.status === "ready") {
                state.progressTimer = setTimeout(() => {
                    if (state.progressSerial === serial && state.mode === "loading") stopCacheProgress(state);
                }, 500);
                return;
            }
            state.progressTimer = setTimeout(poll, 180);
        } catch {
            if (state.progressSerial === serial && !state.disposed) state.progressTimer = setTimeout(poll, 300);
        }
    };
    poll();
}

function bindComparePanZoom(state) {
    const viewport = state.compareViewport;
    if (!viewport) return;
    viewport.addEventListener("wheel", (event) => {
        if (state.mode !== "media") return;
        event.preventDefault();
        event.stopPropagation();
        setCompareZoom(state, state.compareZoom + (event.deltaY < 0 ? 0.1 : -0.1));
    }, { passive: false });
    let drag = null;
    viewport.addEventListener("pointerdown", (event) => {
        if (event.button !== 0 || state.compareZoom <= 1 || event.target.closest?.(".cs-compare-any-divider,.cs-compare-any-resize-handle")) return;
        event.preventDefault();
        event.stopPropagation();
        drag = { pointerId: event.pointerId, startX: event.clientX, startY: event.clientY, panX: state.comparePanX, panY: state.comparePanY };
        viewport.setPointerCapture?.(event.pointerId);
        viewport.classList.add("cs-compare-any-pan-active");
    });
    viewport.addEventListener("pointermove", (event) => {
        if (!drag || event.pointerId !== drag.pointerId) return;
        const rect = viewport.getBoundingClientRect();
        state.comparePanX = drag.panX + event.clientX - drag.startX;
        state.comparePanY = drag.panY + event.clientY - drag.startY;
        clampComparePan(state, rect.width, rect.height);
        drawMedia(state);
    });
    const finish = (event) => {
        if (!drag || event.pointerId !== drag.pointerId) return;
        drag = null;
        viewport.releasePointerCapture?.(event.pointerId);
        viewport.classList.remove("cs-compare-any-pan-active");
    };
    viewport.addEventListener("pointerup", finish);
    viewport.addEventListener("pointercancel", finish);
}

function animationTick(state) {
    if (!state || state.disposed) return;
    if (state.mode === "media") {
        const master = state.videoA.readyState >= 2 ? state.videoA : state.videoB;
        if (state.playing && master.readyState >= 2) {
            const maxFrame = Math.max(0, state.frames - 1);
            const frame = clamp(Math.round(master.currentTime * state.fps), 0, maxFrame);
            if (Math.abs(state.videoA.currentTime - master.currentTime) > 1 / Math.max(1, state.fps)) state.videoA.currentTime = master.currentTime;
            if (Math.abs(state.videoB.currentTime - master.currentTime) > 1 / Math.max(1, state.fps)) state.videoB.currentTime = master.currentTime;
            if (frame !== state.frame) {
                state.frame = frame;
                updateFrameControls(state);
            }
        }
        drawMedia(state);
    }
    state.animationFrame = requestAnimationFrame(() => animationTick(state));
}

function updateFrameControls(state) {
    if (!state.timeline) return;
    state.timeline.value = String(state.frame);
    if (state.kind === "VIDEO") {
        state.readout.textContent = formatFrameCode(state.frame, state.fps);
        state.readoutTotal.textContent = formatFrameCode(state.frames, state.fps);
    } else {
        state.readout.textContent = String(state.frame);
        state.readoutTotal.textContent = String(state.frames);
    }
}

function seekFrame(state, frame, pause = true) {
    const next = clamp(Math.round(Number(frame) || 0), 0, Math.max(0, state.frames - 1));
    state.frame = next;
    if (pause) {
        state.playing = false;
        state.videoA.pause();
        state.videoB.pause();
        state.playButton.classList.remove("playing");
        state.playButton.textContent = ">";
    }
    const seconds = next / Math.max(0.001, state.fps);
    state.videoA.currentTime = seconds;
    state.videoB.currentTime = seconds;
    updateFrameControls(state);
    drawMedia(state);
}

function stopPlayback(state) {
    state.playing = false;
    state.videoA.pause();
    state.videoB.pause();
    state.playButton.classList.remove("playing");
    state.playButton.textContent = ">";
}

async function startVideo(video) {
    // Call play() synchronously from the click handler. Waiting for canplay
    // first loses the user-activation token and browsers reject audible play.
    try {
        const result = video.play();
        if (result && typeof result.then === "function") {
            await result;
        }
        return true;
    } catch {
        return false;
    }
}

async function togglePlayback(state) {
    if (state.playing) {
        state.playing = false;
        state.videoA.pause();
        state.videoB.pause();
        state.playButton.classList.remove("playing");
        state.playButton.textContent = ">";
        return;
    }
    const maxSeconds = Math.max(0, (state.frames - 1) / Math.max(0.001, state.fps));
    if (state.frame >= state.frames - 1) seekFrame(state, 0, false);
    const starts = [startVideo(state.videoA), startVideo(state.videoB)];
    // Reflect the user's click immediately while the media elements finish
    // buffering in the background.
    state.playing = true;
    state.playButton.classList.add("playing");
    state.playButton.textContent = "||";
    const started = await Promise.all(starts);
    if (!started.some(Boolean)) {
        stopPlayback(state);
        state.status.textContent = "Video preview is still loading";
        return;
    }
    try {
        if (state.videoA.currentTime > maxSeconds + 0.1) seekFrame(state, 0, false);
    } catch (error) {
        stopPlayback(state);
        state.status.textContent = error?.message || "Unable to play preview";
    }
}

function setMediaSources(state, payload) {
    const sources = payload.sources || {};
    const timeline = payload.timeline || {};
    state.mode = "media";
    state.kind = String(payload.media_kind || "MEDIA");
    state.frames = Math.max(1, Number(timeline.frames || 1));
    state.fps = Math.max(0.001, Number(timeline.fps || 24));
    state.frame = clamp(state.frame, 0, state.frames - 1);
    state.timeline.max = String(Math.max(0, state.frames - 1));
    const urlA = mediaUrl(sources.a);
    const urlB = mediaUrl(sources.b);
    state.videoA.onloadedmetadata = () => seekFrame(state, state.frame, false);
    state.videoB.onloadedmetadata = () => seekFrame(state, state.frame, false);
    state.videoA.onended = () => stopPlayback(state);
    state.videoB.onended = () => stopPlayback(state);
    if (state.videoA.src !== urlA) { state.videoA.src = urlA; state.videoA.load(); }
    if (state.videoB.src !== urlB) { state.videoB.src = urlB; state.videoB.load(); }
    state.status.textContent = `${state.kind} · ${state.frames} frames · ${state.fps.toFixed(3)} fps`;
    state.playButton.disabled = false;
    state.prevButton.disabled = false;
    state.nextButton.disabled = false;
    state.timeline.disabled = false;
    state.controls.hidden = false;
    state.playButton.hidden = state.kind !== "VIDEO";
    state.playButton.disabled = state.kind !== "VIDEO";
    state.audioButton.hidden = state.kind !== "VIDEO";
    state.audioButton.disabled = state.kind !== "VIDEO";
    setAudioChoice(state, state.audioChoice);
    state.zoomButtons.forEach((button) => { button.disabled = false; });
    updateFrameControls(state);
    drawMedia(state);
}

function appendParts(parent, parts, side) {
    for (const part of Array.isArray(parts) ? parts : []) {
        const span = document.createElement("span");
        span.className = `cs-compare-any-diff-part ${part.kind || "equal"} ${side}`;
        span.textContent = String(part.text || "");
        parent.appendChild(span);
    }
}

function renderDiff(state, payload) {
    stopCacheProgress(state);
    state.mode = "diff";
    stopPlayback(state);
    state.kind = String(payload.diff_kind || payload.type_a || "DIFF");
    state.status.textContent = `${state.kind} · text comparison`;
    state.playButton.disabled = true;
    state.prevButton.disabled = true;
    state.nextButton.disabled = true;
    state.timeline.disabled = true;
    state.zoomButtons.forEach((button) => { button.disabled = true; });
    state.controls.hidden = true;
    state.error.hidden = true;
    state.diff.replaceChildren();
    for (const row of Array.isArray(payload.rows) ? payload.rows : []) {
        const element = document.createElement("div");
        const operation = String(row.op || "equal");
        element.className = `cs-compare-any-diff-row ${operation}`;
        const prefixA = document.createElement("span");
        prefixA.className = "cs-compare-any-diff-prefix";
        prefixA.textContent = operation === "insert" ? " " : operation === "equal" ? " " : "−";
        const cellA = document.createElement("span");
        cellA.className = "cs-compare-any-diff-cell a";
        appendParts(cellA, row.a_parts, "a");
        if (!cellA.childNodes.length) cellA.textContent = String(row.a || "");
        const prefixB = document.createElement("span");
        prefixB.className = "cs-compare-any-diff-prefix";
        prefixB.textContent = operation === "delete" ? " " : operation === "equal" ? " " : "+";
        const cellB = document.createElement("span");
        cellB.className = "cs-compare-any-diff-cell b";
        appendParts(cellB, row.b_parts, "b");
        if (!cellB.childNodes.length) cellB.textContent = String(row.b || "");
        element.append(prefixA, cellA, prefixB, cellB);
        state.diff.appendChild(element);
    }
    state.shell.classList.add("cs-compare-any-diff-mode");
    state.shell.classList.remove("cs-compare-any-media-only");
    state.divider.hidden = true;
    state.canvasCompare.hidden = true;
    state.diff.hidden = false;
}

function renderError(state, payload) {
    stopCacheProgress(state);
    state.mode = "error";
    stopPlayback(state);
    state.status.textContent = "Type mismatch or unsupported input";
    state.playButton.disabled = true;
    state.prevButton.disabled = true;
    state.nextButton.disabled = true;
    state.timeline.disabled = true;
    state.zoomButtons.forEach((button) => { button.disabled = true; });
    state.controls.hidden = true;
    state.error.hidden = false;
    const unsupported = payload.type_a === "UNSUPPORTED" && payload.type_b === "UNSUPPORTED";
    state.error.textContent = unsupported ? "Unsupported Input Type" : "source_a and source-b must be the same type";
    state.shell.classList.remove("cs-compare-any-diff-mode");
    state.shell.classList.add("cs-compare-any-media-only");
    state.canvasCompare.hidden = false;
    state.divider.hidden = true;
    state.diff.hidden = true;
    clearMediaCanvases(state);
}

function renderMedia(state, payload) {
    stopCacheProgress(state);
    stopPlayback(state);
    setCompareZoom(state, 1);
    state.error.hidden = true;
    state.shell.classList.remove("cs-compare-any-diff-mode");
    state.shell.classList.add("cs-compare-any-media-only");
    state.canvasCompare.hidden = false;
    state.diff.hidden = true;
    state.divider.hidden = false;
    setMediaSources(state, payload);
}

function updateNode(node, output) {
    const state = node?.[STATE];
    if (!state || !output) return;
    const payload = latestPayload(output);
    if (payload.mode === "media") renderMedia(state, payload);
    else if (payload.mode === "diff") renderDiff(state, payload);
    else renderError(state, payload);
    node.graph?.setDirtyCanvas?.(true, true);
}

function addViewport(node) {
    addStyles();
    const shell = document.createElement("div");
    shell.className = "cs-compare-any-shell cs-compare-any-media-only";
    shell.innerHTML = `
      <div class="cs-compare-any-head"><span class="cs-compare-any-title">CS Compare Any</span><span class="cs-compare-any-status">Waiting for execution...</span></div>
      <div class="cs-compare-any-grid">
        <div class="cs-compare-any-viewport cs-compare-any-compare"><canvas class="cs-compare-any-canvas-compare"></canvas><div class="cs-compare-any-diff"></div><div class="cs-compare-any-divider" title="Drag to compare A and B"></div><div class="cs-compare-any-error" hidden></div><div class="cs-compare-any-loading" hidden>Loading comparison cache 0%</div><div class="cs-compare-any-resize-handle" data-resize="bottom" title="Resize viewport" aria-label="Resize viewport"></div></div>
      </div>
      <div class="cs-compare-any-controls" hidden><button class="cs-compare-any-button cs-compare-any-prev" type="button" title="Previous frame" aria-label="Previous frame" disabled>|&lt;</button><button class="cs-compare-any-button cs-compare-any-play" type="button" title="Play or pause" aria-label="Play or pause" disabled>&gt;</button><button class="cs-compare-any-button cs-compare-any-next" type="button" title="Next frame" aria-label="Next frame" disabled>&gt;|</button><button class="cs-compare-any-button cs-compare-any-audio-button" type="button" title="Play source A audio" aria-label="Play source A audio" disabled>🔊 A</button><input class="cs-compare-any-timeline" type="range" min="0" max="0" step="1" value="0" aria-label="Frame timeline" disabled><span class="cs-compare-any-readout">00:00:00</span><span class="cs-compare-any-readout-separator">/</span><span class="cs-compare-any-readout-total">00:00:00</span><div class="cs-compare-any-zoom-group"><button class="cs-compare-any-button cs-compare-any-zoom-button" data-zoom="in" type="button" title="Zoom in" aria-label="Zoom in" disabled>+</button><button class="cs-compare-any-button cs-compare-any-zoom-button" data-zoom="fit" type="button" title="Fit" aria-label="Fit" disabled>Fit</button><button class="cs-compare-any-button cs-compare-any-zoom-button" data-zoom="out" type="button" title="Zoom out" aria-label="Zoom out" disabled>−</button></div></div>`;
    for (const event of ["pointerdown", "click", "dblclick", "wheel"]) shell.addEventListener(event, (value) => value.stopPropagation(), { passive: false });
    const videoA = document.createElement("video");
    const videoB = document.createElement("video");
    for (const video of [videoA, videoB]) {
        video.preload = "auto";
        video.muted = true;
        video.playsInline = true;
        video.setAttribute("aria-hidden", "true");
        video.style.position = "absolute";
        video.style.width = "1px";
        video.style.height = "1px";
        video.style.opacity = "0";
        video.style.pointerEvents = "none";
    }
    shell.append(videoA, videoB);
    const widget = node.addDOMWidget("compare_any_view", "compare_any_view", shell, { margin: 4 });
    widget.serialize = false;
    widget.options = { ...(widget.options || {}), serialize: false, canvasOnly: false };
    const state = {
        shell,
        widget,
        status: shell.querySelector(".cs-compare-any-status"),
        canvasCompare: shell.querySelector(".cs-compare-any-canvas-compare"),
        compareViewport: shell.querySelector(".cs-compare-any-compare"),
        grid: shell.querySelector(".cs-compare-any-grid"),
        diff: shell.querySelector(".cs-compare-any-diff"),
        divider: shell.querySelector(".cs-compare-any-divider"),
        error: shell.querySelector(".cs-compare-any-error"),
        loading: shell.querySelector(".cs-compare-any-loading"),
        controls: shell.querySelector(".cs-compare-any-controls"),
        timeline: shell.querySelector(".cs-compare-any-timeline"),
        readout: shell.querySelector(".cs-compare-any-readout"),
        readoutTotal: shell.querySelector(".cs-compare-any-readout-total"),
        zoomButtons: [...shell.querySelectorAll(".cs-compare-any-zoom-button")],
        audioButton: shell.querySelector(".cs-compare-any-audio-button"),
        playButton: shell.querySelector(".cs-compare-any-play"),
        prevButton: shell.querySelector(".cs-compare-any-prev"),
        nextButton: shell.querySelector(".cs-compare-any-next"),
        videoA,
        videoB,
        mode: "none",
        kind: "",
        frame: 0,
        frames: 1,
        fps: 24,
        playing: false,
        comparePosition: 50,
        compareZoom: 1,
        comparePanX: 0,
        comparePanY: 0,
        audioChoice: "A",
        bottomHeight: DEFAULT_BOTTOM_HEIGHT,
        bottomHeightManual: false,
        defaultGridHeight: DEFAULT_BOTTOM_HEIGHT,
        baseNodeHeight: Math.max(430, Number(node.size?.[1]) || 430),
        node,
        disposed: false,
        animationFrame: null,
        progressTimer: null,
        progressSerial: 0,
    };
    widget.options.getMinHeight = () => Math.max(390, state.bottomHeight + 72);
    widget.options.getMaxHeight = () => undefined;
    state.timeline.addEventListener("input", () => seekFrame(state, state.timeline.value));
    state.prevButton.addEventListener("click", () => seekFrame(state, state.frame - 1));
    state.nextButton.addEventListener("click", () => seekFrame(state, state.frame + 1));
    state.playButton.addEventListener("click", () => togglePlayback(state));
    state.audioButton.addEventListener("click", () => {
        const next = state.audioChoice === "A" ? "B" : state.audioChoice === "B" ? "MUTE" : "A";
        setAudioChoice(state, next);
    });
    state.zoomButtons.forEach((button) => button.addEventListener("click", () => {
        if (button.dataset.zoom === "fit") setCompareZoom(state, 1);
        else setCompareZoom(state, state.compareZoom + (button.dataset.zoom === "in" ? 0.1 : -0.1));
    }));
    setAudioChoice(state, "A");
    state.playButton.disabled = true;
    state.audioButton.disabled = true;
    state.audioButton.hidden = true;
    state.prevButton.disabled = true;
    state.nextButton.disabled = true;
    state.timeline.disabled = true;
    state.zoomButtons.forEach((button) => { button.disabled = true; });
    state.grid.style.setProperty("--compare-position", `${state.comparePosition}%`);
    applyViewportHeights(state);
    updateAutoBottomHeight(state);
    state.shell.querySelectorAll("[data-resize]").forEach((handle) => attachHeightHandle(state, handle));
    bindComparePanZoom(state);
    let dragging = false;
    const updateDivider = (event) => {
        const rect = state.divider.parentElement.getBoundingClientRect();
        state.comparePosition = clamp(((event.clientX - rect.left) / Math.max(1, rect.width)) * 100, 0, 100);
        state.divider.parentElement.style.setProperty("--compare-position", `${state.comparePosition}%`);
        drawMedia(state);
    };
    state.divider.addEventListener("pointerdown", (event) => { if (event.button !== 0) return; event.preventDefault(); dragging = true; state.divider.setPointerCapture?.(event.pointerId); updateDivider(event); });
    state.divider.addEventListener("pointermove", (event) => { if (dragging) updateDivider(event); });
    const stopDragging = (event) => { if (!dragging) return; dragging = false; state.divider.releasePointerCapture?.(event.pointerId); };
    state.divider.addEventListener("pointerup", stopDragging);
    state.divider.addEventListener("pointercancel", stopDragging);
    const resizeObserver = new ResizeObserver(() => { updateAutoBottomHeight(state); drawMedia(state); });
    resizeObserver.observe(state.grid);
    state.dispose = () => { state.disposed = true; stopCacheProgress(state); cancelAnimationFrame(state.animationFrame); resizeObserver.disconnect(); videoA.pause(); videoB.pause(); videoA.removeAttribute("src"); videoB.removeAttribute("src"); };
    node[STATE] = state;
    node[WIDGET] = widget;
    state.animationFrame = requestAnimationFrame(() => animationTick(state));
    return state;
}

function graphNode(locator) {
    const rawId = String(locator || "").split(":").at(-1);
    const numericId = Number(rawId);
    const ids = Number.isFinite(numericId) ? [numericId, rawId] : [rawId];
    const graphs = [app.canvas?.graph, app.graph, app.rootGraph].filter(Boolean);
    for (const graph of graphs) {
        for (const id of ids) {
            const node = graph.getNodeById?.(id);
            if (node) return node;
        }
    }
    return null;
}

api.addEventListener("executing", ({ detail }) => {
    if (detail == null) return;
    const node = graphNode(detail);
    if (node?.type === NODE_ID || node?.comfyClass === NODE_ID) beginCacheProgress(node);
});

app.registerExtension({
    name: "CineStyle.CompareAny",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_ID) return;
        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            originalCreated?.apply(this, arguments);
            addViewport(this);
            this.setSize?.([560, 430]);
        };
        const originalExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            updateNode(this, output);
            originalExecuted?.apply(this, arguments);
        };
        const originalRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this[STATE]?.dispose?.();
            originalRemoved?.apply(this, arguments);
        };
    },
    loadedGraphNode(node) {
        if (node?.type !== NODE_ID && node?.comfyClass !== NODE_ID) return;
        if (!node[STATE]) addViewport(node);
        const width = Math.max(560, Number(node.size?.[0]) || 560);
        const storedHeight = Number(node.size?.[1]) || 430;
        // 650px was the old two-viewport default. Collapse that legacy size
        // so loaded nodes do not retain an empty upper area.
        const height = storedHeight === 650 ? 430 : Math.max(430, storedHeight);
        if (node[STATE]) node[STATE].baseNodeHeight = height;
        node.setSize?.([width, height]);
    },
    onNodeOutputsUpdated(outputs) {
        for (const [locator, output] of Object.entries(outputs || {})) {
            const node = graphNode(locator);
            if (node && (node.type === NODE_ID || node.comfyClass === NODE_ID)) updateNode(node, output);
        }
    },
});
