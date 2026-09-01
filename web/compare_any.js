import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_ID = "CS_Compare_Any";
const STYLE_ID = "cinestyle-compare-any-style";
const STATE = Symbol("cinestyle-compare-any-state");
const WIDGET = Symbol("cinestyle-compare-any-widget");
const UNKNOWN_VIEWPORT_ASPECT = 1;
// The combined diff viewport preserves each pane's original 2:3 / 3:2 aspect.
const TEXT_HORIZONTAL_ASPECT = 4 / 3;
const TEXT_VERTICAL_ASPECT = 3 / 4;
const DEFAULT_BOTTOM_HEIGHT = 306;
const MIN_BOTTOM_HEIGHT = 180;
const MAX_VIEWPORT_HEIGHT = 1200;
const NODE_MIN_WIDTH = 560;
const NODE_MIN_HEIGHT = 430;
const GRID_GAP = 6;
const SHELL_GAP = 8;
const NODE_BOTTOM_PADDING = 6;
const MEDIA_MIN_VIEWPORT_HEIGHT = 1;
const SPLIT_MIN_SIZE = 120;

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
      .cs-compare-any-shell{position:relative;display:grid;align-content:start;gap:8px;box-sizing:border-box;width:100%;padding:8px;background:var(--comfy-input-bg,#17191e);color:var(--input-text,#e6e9ef);font:12px/1.35 system-ui,sans-serif}
      .cs-compare-any-head,.cs-compare-any-controls{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
      .cs-compare-any-controls[hidden]{display:none}
      .cs-compare-any-head{justify-content:space-between;color:var(--descrip-text,#9da5b4)}
      .cs-compare-any-title{font-weight:600;color:var(--input-text,#e6e9ef)}
      .cs-compare-any-status{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:75%;color:var(--descrip-text,#9da5b4)}
      .cs-compare-any-grid{position:relative;display:grid;align-self:start;grid-template-columns:repeat(2,minmax(0,1fr));grid-template-rows:var(--cs-compare-bottom-height,306px);gap:6px;min-height:0}
      .cs-compare-any-media-layout .cs-compare-any-grid{grid-template-columns:repeat(2,minmax(0,var(--cs-compare-media-width,1fr)));justify-content:center;background:inherit}
      .cs-compare-any-media-layout .cs-compare-any-viewport{width:var(--cs-compare-media-width,100%);height:var(--cs-compare-bottom-height,306px);justify-self:center}
      .cs-compare-any-media-layout.cs-compare-any-layout-vertical .cs-compare-any-grid{grid-template-columns:minmax(0,var(--cs-compare-media-width,1fr));justify-items:center}
      .cs-compare-any-media-layout.cs-compare-any-layout-vertical .cs-compare-any-viewport{grid-column:1}
      .cs-compare-any-viewport{position:relative;box-sizing:border-box;min-width:0;min-height:0;overflow:hidden;border:1px solid var(--border-color,#3c424d);border-radius:5px;background:#08090b}
      .cs-compare-any-viewport canvas{display:block;width:100%;height:100%;min-height:0}
      .cs-compare-any-source{grid-column:1;grid-row:1}
      .cs-compare-any-compare{grid-column:2;grid-row:1}
      .cs-compare-any-layout-vertical .cs-compare-any-grid{grid-template-columns:1fr;grid-template-rows:repeat(2,var(--cs-compare-bottom-height,306px))}
      .cs-compare-any-layout-vertical .cs-compare-any-source{grid-column:1;grid-row:1}
      .cs-compare-any-layout-vertical .cs-compare-any-compare{grid-column:1;grid-row:2}
      .cs-compare-any-divider{position:absolute;z-index:3;top:0;bottom:0;left:var(--compare-position,0%);width:2px;transform:translateX(-1px);background:#f4f7fb;box-shadow:0 0 0 1px #11141980;cursor:ew-resize;touch-action:none}
      .cs-compare-any-divider::before{content:"";position:absolute;top:50%;left:50%;width:22px;height:22px;transform:translate(-50%,-50%);border:2px solid #f4f7fb;border-radius:50%;background:#20232a;box-shadow:0 2px 8px #000b}
      .cs-compare-any-divider::after{content:"↔";position:absolute;top:50%;left:50%;transform:translate(-50%,-53%);color:#f4f7fb;font-size:13px;line-height:1}
      .cs-compare-any-diff{display:none;box-sizing:border-box;width:100%;height:100%;min-width:0;min-height:0;grid-template-columns:minmax(0,var(--cs-compare-split-first,1fr)) minmax(0,var(--cs-compare-split-second,1fr));gap:6px;overflow:hidden;font:11px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace}
      .cs-compare-any-layout-vertical .cs-compare-any-diff{grid-template-columns:1fr;grid-template-rows:minmax(0,var(--cs-compare-split-first,1fr)) minmax(0,var(--cs-compare-split-second,1fr))}
      .cs-compare-any-diff-pane{display:grid;grid-template-rows:auto minmax(0,1fr);min-width:0;min-height:0;overflow:hidden;background:#111419}
      .cs-compare-any-diff-label{padding:4px 8px;border-bottom:1px solid var(--border-color,#3c424d);color:var(--descrip-text,#9da5b4);font:600 11px/1.35 system-ui,sans-serif;user-select:none}
      .cs-compare-any-diff-content{min-width:0;min-height:0;overflow:auto;padding:5px 0}
      .cs-compare-any-diff-line{display:grid;grid-template-columns:24px minmax(0,1fr);box-sizing:border-box;min-height:16px;padding:0 8px;white-space:pre-wrap;overflow-wrap:anywhere}
      .cs-compare-any-diff-prefix{padding-right:5px;color:#707987;text-align:center;user-select:none}
      .cs-compare-any-diff-line-content{min-width:0;white-space:pre-wrap;overflow-wrap:anywhere}
      .cs-compare-any-diff-line.equal{color:#7f8794}
      .cs-compare-any-diff-line.delete.a{background:#60272b80;color:#ffc5c8}
      .cs-compare-any-diff-line.insert.b{background:#1e563680;color:#c6f4ce}
      .cs-compare-any-diff-line.replace.a{background:#60272b55;color:#ffc5c8}
      .cs-compare-any-diff-line.replace.b{background:#1e563655;color:#c6f4ce}
      .cs-compare-any-text{display:none;box-sizing:border-box;width:100%;height:100%;margin:0;padding:8px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;font:11px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace;color:#dce2ea}
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
      .cs-compare-any-media-only .cs-compare-any-diff,.cs-compare-any-media-only .cs-compare-any-text{display:none}
      .cs-compare-any-diff-mode .cs-compare-any-grid{grid-template-columns:1fr;grid-template-rows:var(--cs-compare-bottom-height,306px);gap:0}
      .cs-compare-any-diff-mode .cs-compare-any-source{display:none}
      .cs-compare-any-diff-mode .cs-compare-any-compare{grid-column:1;grid-row:1;width:100%;height:var(--cs-compare-bottom-height,306px)}
      .cs-compare-any-diff-mode .cs-compare-any-diff{display:grid}
      .cs-compare-any-diff-mode .cs-compare-any-viewport canvas,.cs-compare-any-diff-mode .cs-compare-any-divider{display:none}
      .cs-compare-any-diff-mode .cs-compare-any-viewport{background:#111419}
      .cs-compare-any-diff-mode .cs-compare-any-compare{overflow:hidden}
      .cs-compare-any-resize-handle{position:absolute;z-index:5;right:24px;bottom:1px;left:8px;height:10px;cursor:ns-resize;touch-action:none}
      .cs-compare-any-resize-handle::after{content:"";position:absolute;top:4px;left:50%;width:34px;height:2px;border-radius:2px;transform:translateX(-50%);background:#8c96a5;opacity:.72}
      .cs-compare-any-resize-handle:hover::after{background:#b8d9f5;opacity:1}
      .cs-compare-any-split-handle{display:none;position:absolute;z-index:7;background:transparent;touch-action:none}
      .cs-compare-any-diff-mode .cs-compare-any-split-handle{display:block}
      .cs-compare-any-split-handle-horizontal{top:0;bottom:0;width:10px;cursor:col-resize}
      .cs-compare-any-split-handle-vertical{right:0;left:0;height:10px;cursor:row-resize}
      .cs-compare-any-split-handle::after{content:"";position:absolute;background:#8c96a5;opacity:.75;border-radius:2px}
      .cs-compare-any-split-handle-horizontal::after{top:50%;left:4px;width:2px;height:34px;transform:translateY(-50%)}
      .cs-compare-any-split-handle-vertical::after{top:4px;left:50%;width:34px;height:2px;transform:translateX(-50%)}
      .cs-compare-any-split-handle:hover::after{background:#b8d9f5;opacity:1}
      .cs-compare-any-pan-ready{cursor:grab}
      .cs-compare-any-pan-active{cursor:grabbing}
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
    const width = Math.max(1, Number(canvas.clientWidth) || 0);
    const height = Math.max(1, Number(canvas.clientHeight) || 0);
    const ratio = Math.max(1, window.devicePixelRatio || 1);
    const pixelWidth = Math.max(2, Math.round(width * ratio));
    const pixelHeight = Math.max(2, Math.round(height * ratio));
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
        canvas.width = pixelWidth;
        canvas.height = pixelHeight;
    }
    const context = canvas.getContext("2d");
    if (!context) return null;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { context, width, height };
}

function drawMedia(state) {
    if (!state || state.mode !== "media") return;
    const canvasSource = setupCanvas(state.canvasA);
    const canvasCompare = setupCanvas(state.canvasCompare);
    if (!canvasSource || !canvasCompare) return;
    const source = canvasSource;
    source.context.save();
    source.context.beginPath();
    source.context.rect(0, 0, source.width, source.height);
    source.context.clip();
    source.context.fillStyle = "#08090b";
    source.context.fillRect(0, 0, source.width, source.height);
    drawContained(source.context, state.videoA, source.width, source.height, state.compareZoom, state.comparePanX, state.comparePanY);
    source.context.restore();
    const { context, width, height } = canvasCompare;
    clampComparePan(state, width, height);
    context.save();
    context.beginPath();
    context.rect(0, 0, width, height);
    context.clip();
    context.fillStyle = "#08090b";
    context.fillRect(0, 0, width, height);
    drawContained(context, state.videoB, width, height, state.compareZoom, state.comparePanX, state.comparePanY);
    const requestedPosition = Number(state.comparePosition);
    const position = clamp(Number.isFinite(requestedPosition) ? requestedPosition : 0, 0, 100) / 100;
    context.save();
    context.beginPath();
    context.rect(0, 0, width * position, height);
    context.clip();
    context.fillStyle = "#08090b";
    context.fillRect(0, 0, width, height);
    drawContained(context, state.videoA, width, height, state.compareZoom, state.comparePanX, state.comparePanY);
    context.restore();
    context.restore();
}

function forceCompareRefresh(state) {
    if (!state || state.disposed) return;
    drawMedia(state);
    if (state.compareRefreshFrame != null) cancelAnimationFrame(state.compareRefreshFrame);
    state.compareRefreshFrame = requestAnimationFrame(() => {
        state.compareRefreshFrame = requestAnimationFrame(() => {
            state.compareRefreshFrame = null;
            if (!state.disposed) drawMedia(state);
        });
    });
}

function applyViewportHeights(state) {
    state.shell.style.setProperty("--cs-compare-bottom-height", `${Math.round(state.bottomHeight)}px`);
}

function gridHeight(state) {
    const rows = state.mode === "diff" ? 1 : state.layout === "vertical" ? 2 : 1;
    return state.bottomHeight * rows + GRID_GAP * (rows - 1);
}

function minimumGridHeight(state) {
    const minimum = state.mode === "media" ? MEDIA_MIN_VIEWPORT_HEIGHT : MIN_BOTTOM_HEIGHT;
    const rows = state.mode === "diff" ? 1 : state.layout === "vertical" ? 2 : 1;
    return minimum * rows + GRID_GAP * (rows - 1);
}

function setMediaLayout(state, active) {
    state.shell.classList.toggle("cs-compare-any-media-layout", Boolean(active));
}

function setMediaViewportSize(state, width, height) {
    const nextWidth = Math.max(1, Math.round(Number(width) || 1));
    const nextHeight = Math.max(MEDIA_MIN_VIEWPORT_HEIGHT, Math.round(Number(height) || MEDIA_MIN_VIEWPORT_HEIGHT));
    state.mediaViewportWidth = nextWidth;
    state.bottomHeight = nextHeight;
    state.shell.style.setProperty("--cs-compare-media-width", `${Math.round(nextWidth)}px`);
    applyViewportHeights(state);
}

function splitAxis(state) {
    return state.layout === "vertical" ? "vertical" : "horizontal";
}

function splitTrackSize(state) {
    const width = Number(state.grid?.clientWidth) || 0;
    const height = Number(state.grid?.clientHeight) || 0;
    const total = splitAxis(state) === "vertical" ? height : width;
    return Math.max(0, total - GRID_GAP);
}

function applyDiffSplit(state, firstSize, preserveRatio = false) {
    if (!state?.grid || !state.splitHandle) return false;
    const axis = splitAxis(state);
    const total = splitTrackSize(state);
    if (total <= 0) return false;
    let first = Number(firstSize);
    if (preserveRatio || !Number.isFinite(first)) first = total * (Number.isFinite(state.splitRatio) ? state.splitRatio : 0.5);
    first = clamp(first, Math.min(SPLIT_MIN_SIZE, total / 2), Math.max(Math.min(SPLIT_MIN_SIZE, total / 2), total - Math.min(SPLIT_MIN_SIZE, total / 2)));
    const second = Math.max(0, total - first);
    state.splitRatio = total > 0 ? first / total : 0.5;
    state.grid.style.setProperty("--cs-compare-split-first", `${Math.round(first)}px`);
    state.grid.style.setProperty("--cs-compare-split-second", `${Math.round(second)}px`);
    state.splitHandle.classList.toggle("cs-compare-any-split-handle-horizontal", axis === "horizontal");
    state.splitHandle.classList.toggle("cs-compare-any-split-handle-vertical", axis === "vertical");
    if (axis === "horizontal") {
        state.splitHandle.style.left = `${Math.round(first + GRID_GAP / 2 - 5)}px`;
        state.splitHandle.style.top = "0px";
        state.splitHandle.style.bottom = "0px";
        state.splitHandle.style.right = "auto";
        state.splitHandle.style.height = "auto";
    } else {
        state.splitHandle.style.top = `${Math.round(first + GRID_GAP / 2 - 5)}px`;
        state.splitHandle.style.left = "0px";
        state.splitHandle.style.right = "0px";
        state.splitHandle.style.bottom = "auto";
        state.splitHandle.style.width = "auto";
    }
    return true;
}

function syncDiffSplitToGrid(state) {
    if (!state || state.mode !== "diff" || state.splitDragging) return false;
    return applyDiffSplit(state, null, true);
}

function attachDiffSplitHandle(state) {
    const handle = state?.splitHandle;
    if (!handle) return;
    let drag = null;
    const finish = (event) => {
        if (!drag || event.pointerId !== drag.pointerId) return;
        drag = null;
        state.splitDragging = false;
        handle.releasePointerCapture?.(event.pointerId);
        nodeGraphDirty(state.node);
    };
    handle.addEventListener("pointerdown", (event) => {
        if (event.button !== 0 || state.mode !== "diff") return;
        event.preventDefault();
        event.stopPropagation();
        const axis = splitAxis(state);
        const rect = state.grid.getBoundingClientRect();
        const cssScale = axis === "vertical"
            ? rect.height / Math.max(1, state.grid.clientHeight)
            : rect.width / Math.max(1, state.grid.clientWidth);
        const total = splitTrackSize(state);
        const first = total * (Number.isFinite(state.splitRatio) ? state.splitRatio : 0.5);
        drag = {
            pointerId: event.pointerId,
            axis,
            startPointer: axis === "vertical" ? event.clientY : event.clientX,
            startFirst: first,
            cssScale: Number.isFinite(cssScale) && cssScale > 0 ? cssScale : 1,
        };
        state.splitDragging = true;
        handle.setPointerCapture?.(event.pointerId);
    });
    handle.addEventListener("pointermove", (event) => {
        if (!drag || event.pointerId !== drag.pointerId) return;
        event.preventDefault();
        event.stopPropagation();
        const pointer = drag.axis === "vertical" ? event.clientY : event.clientX;
        applyDiffSplit(state, drag.startFirst + (pointer - drag.startPointer) / drag.cssScale);
        nodeGraphDirty(state.node);
    });
    handle.addEventListener("pointerup", finish);
    handle.addEventListener("pointercancel", finish);
}

function mediaAvailableWidth(state) {
    const width = Number(state.grid?.clientWidth) || 0;
    if (width <= 0) return 0;
    return state.layout === "vertical" ? width : Math.max(0, (width - GRID_GAP) / 2);
}

function fitMediaViewport(state, fitOuterHeight = true) {
    if (!state || state.mode !== "media") return false;
    const availableWidth = mediaAvailableWidth(state) - 2;
    // A freshly mounted DOM widget can report zero width for a frame; wait for
    // the observer/retry rather than collapsing the viewport to a tiny size.
    if (availableWidth <= 0) return false;
    const aspect = Number(state.viewportAspect);
    const safeAspect = Number.isFinite(aspect) && aspect > 0 ? aspect : UNKNOWN_VIEWPORT_ASPECT;
    let nextWidth = availableWidth;
    let nextHeight = availableWidth / safeAspect;
    if (fitOuterHeight) {
        const outerHeight = widgetInnerHeight(state);
        if (outerHeight > 0) {
            const rows = state.layout === "vertical" ? 2 : 1;
            const availableGridHeight = outerHeight - shellChromeHeight(state);
            const maxHeight = (availableGridHeight - GRID_GAP * (rows - 1)) / rows;
            if (Number.isFinite(maxHeight) && maxHeight > 0 && nextHeight > maxHeight) {
                nextHeight = maxHeight;
                nextWidth = nextHeight * safeAspect;
            }
        }
    }
    if (nextWidth > availableWidth) {
        nextWidth = availableWidth;
        nextHeight = nextWidth / safeAspect;
    }
    nextHeight = clamp(nextHeight, MEDIA_MIN_VIEWPORT_HEIGHT, MAX_VIEWPORT_HEIGHT);
    nextWidth = nextHeight * safeAspect;
    const changed = Math.abs(nextWidth - (Number(state.mediaViewportWidth) || 0)) >= 2 || Math.abs(nextHeight - state.bottomHeight) >= 2;
    setMediaViewportSize(state, nextWidth, nextHeight);
    if (changed) nodeGraphDirty(state.node);
    return changed;
}

function scheduleMediaFitRetry(state, attempt = 0) {
    if (!state || state.disposed || state.mode !== "media" || state.mediaFitRetryFrame != null || attempt >= 16) return;
    state.mediaFitRetryFrame = requestAnimationFrame(() => {
        state.mediaFitRetryFrame = requestAnimationFrame(() => {
            state.mediaFitRetryFrame = null;
            if (state.disposed || state.mode !== "media") return;
            if (mediaAvailableWidth(state) - 2 > 0) {
                fitMediaViewport(state, false);
                syncNodeHeight(state);
                drawMedia(state);
                return;
            }
            scheduleMediaFitRetry(state, attempt + 1);
        });
    });
}

function textViewportAspect(layout) {
    return String(layout || "").toLowerCase() === "vertical" ? TEXT_VERTICAL_ASPECT : TEXT_HORIZONTAL_ASPECT;
}

function viewportWidth(state) {
    if (state.mode === "diff") return Number(state.grid?.clientWidth) || 0;
    const viewportClientWidth = Number(state.sourceViewport?.clientWidth) || 0;
    if (viewportClientWidth > 0) return viewportClientWidth;
    const width = Number(state.grid?.clientWidth) || 0;
    if (width <= 0) return 0;
    return state.layout === "vertical" ? width : Math.max(0, (width - GRID_GAP) / 2);
}

function shellChromeHeight(state) {
    const shell = state.shell;
    if (!shell) return 0;
    const styles = getComputedStyle(shell);
    const padding = (Number.parseFloat(styles.paddingTop) || 0) + (Number.parseFloat(styles.paddingBottom) || 0);
    const gap = Number.parseFloat(styles.rowGap || styles.gap) || SHELL_GAP;
    const headHeight = Number(state.head?.offsetHeight) || 0;
    const controlsVisible = Boolean(state.controls && !state.controls.hidden);
    const controlsHeight = controlsVisible
        ? Math.max(34, Number(state.controls.offsetHeight) || 0)
        : 0;
    const rows = 2 + (controlsVisible ? 1 : 0);
    return padding + headHeight + controlsHeight + gap * Math.max(0, rows - 1);
}

function widgetInnerHeight(state) {
    const computedHeight = Number(state.widget?.computedHeight);
    const nodeHeight = Number(state.node?.size?.[1]);
    const widgetY = Number(state.widget?.y);
    const nodeCapacity = Number.isFinite(nodeHeight) && Number.isFinite(widgetY)
        ? Math.max(0, nodeHeight - widgetY - NODE_BOTTOM_PADDING)
        : 0;
    if (Number.isFinite(computedHeight) && computedHeight > 0) {
        const margin = Number(state.widget?.options?.margin) || 0;
        const computedInner = Math.max(0, computedHeight - margin * 2);
        return nodeCapacity > 0 ? Math.min(computedInner, nodeCapacity) : computedInner;
    }
    if (Number.isFinite(nodeHeight) && Number.isFinite(widgetY)) {
        return nodeCapacity;
    }
    return 0;
}

function setNodeSize(state, width, height, minHeight = NODE_MIN_HEIGHT) {
    const node = state?.node;
    if (!node || state.internalResize) return false;
    const currentWidth = Number(node.size?.[0]) || NODE_MIN_WIDTH;
    const currentHeight = Number(node.size?.[1]) || NODE_MIN_HEIGHT;
    const nextWidth = Math.max(NODE_MIN_WIDTH, Math.round(Number(width) || currentWidth));
    const nextHeight = Math.max(0, Number(minHeight) || 0, Math.round(Number(height) || currentHeight));
    if (Math.abs(currentWidth - nextWidth) < 2 && Math.abs(currentHeight - nextHeight) < 2) return false;
    state.internalResize = true;
    try {
        node.setSize?.([nextWidth, nextHeight]);
    } finally {
        state.internalResize = false;
    }
    return true;
}

function syncNodeHeight(state) {
    const node = state.node;
    if (!node) return;
    const extraControlsHeight = state.controls && !state.controls.hidden
        ? Math.max(34, Math.ceil(state.controls.offsetHeight || 0) + 8)
        : 0;
    const targetHeight = Math.max(NODE_MIN_HEIGHT, Math.round(state.baseNodeHeight + gridHeight(state) - state.defaultGridHeight + extraControlsHeight));
    const width = Math.max(NODE_MIN_WIDTH, Number(node.size?.[0]) || NODE_MIN_WIDTH);
    setNodeSize(state, width, targetHeight);
}

function setViewportHeight(state, requested) {
    const previousGridHeight = gridHeight(state);
    const minimum = state.mode === "media" ? MEDIA_MIN_VIEWPORT_HEIGHT : MIN_BOTTOM_HEIGHT;
    let maximum = MAX_VIEWPORT_HEIGHT;
    if (state.mode === "media") {
        const aspect = Number(state.viewportAspect);
        const safeAspect = Number.isFinite(aspect) && aspect > 0 ? aspect : UNKNOWN_VIEWPORT_ASPECT;
        const availableWidth = mediaAvailableWidth(state);
        if (availableWidth > 0) maximum = Math.min(maximum, availableWidth / safeAspect);
    }
    state.bottomHeight = clamp(Math.round(requested), minimum, Math.max(minimum, maximum));
    state.bottomHeightManual = true;
    if (state.mode === "media") {
        const aspect = Number(state.viewportAspect);
        const safeAspect = Number.isFinite(aspect) && aspect > 0 ? aspect : UNKNOWN_VIEWPORT_ASPECT;
        setMediaViewportSize(state, state.bottomHeight * safeAspect, state.bottomHeight);
    } else {
        applyViewportHeights(state);
    }
    if (state.mode === "diff") {
        const width = viewportWidth(state);
        if (width > 0 && state.bottomHeight > 0) state.viewportAspect = width / state.bottomHeight;
    }
    const gridDelta = gridHeight(state) - previousGridHeight;
    const currentWidth = Number(state.node?.size?.[0]) || NODE_MIN_WIDTH;
    const currentHeight = Number(state.node?.size?.[1]) || NODE_MIN_HEIGHT;
    setNodeSize(state, currentWidth, currentHeight + gridDelta, 0);
    nodeGraphDirty(state.node);
}

function nodeGraphDirty(node) {
    node?.graph?.setDirtyCanvas?.(true, true);
}

function updateAutoBottomHeight(state, force = false) {
    if ((!force && state.bottomHeightManual) || !state.grid || state.displayReady || state.preserveOuterSize || (state.mode === "diff" && state.outerSizeManual)) return false;
    const width = viewportWidth(state);
    if (width <= 0) return false;
    const aspect = Number(state.viewportAspect);
    const next = clamp(Math.round(width / (Number.isFinite(aspect) && aspect > 0 ? aspect : UNKNOWN_VIEWPORT_ASPECT)), MIN_BOTTOM_HEIGHT, MAX_VIEWPORT_HEIGHT);
    if (Math.abs(next - state.bottomHeight) < 2) {
        if (force) syncNodeHeight(state);
        return false;
    }
    state.bottomHeight = next;
    applyViewportHeights(state);
    syncNodeHeight(state);
    return true;
}

function updateTextViewportFromOuter(state) {
    if (!state || state.mode !== "diff" || !state.displayReady) return false;
    const width = viewportWidth(state);
    const outerHeight = widgetInnerHeight(state);
    if (width <= 0 || outerHeight <= 0) return false;
    const availableGridHeight = outerHeight - shellChromeHeight(state);
    const next = clamp(
        Math.round(availableGridHeight),
        MIN_BOTTOM_HEIGHT,
        MAX_VIEWPORT_HEIGHT,
    );
    if (next <= 0) return false;
    const changed = Math.abs(next - state.bottomHeight) >= 2;
    state.bottomHeight = next;
    state.viewportAspect = width / next;
    state.bottomHeightManual = false;
    applyViewportHeights(state);
    if (changed) nodeGraphDirty(state.node);
    return changed;
}

function updateMediaViewportFromOuter(state) {
    return fitMediaViewport(state, true);
}

function scheduleOuterResize(state) {
    if (!state || state.disposed || state.outerResizeFrame != null) return;
    state.outerResizeFrame = requestAnimationFrame(() => {
        state.outerResizeFrame = requestAnimationFrame(() => {
            state.outerResizeFrame = null;
            if (state.disposed || state.internalResize) return;
            if (state.mode === "diff") updateTextViewportFromOuter(state);
            else if (state.mode === "media") updateMediaViewportFromOuter(state);
            syncDiffSplitToGrid(state);
            drawMedia(state);
        });
    });
}

function scheduleOutputFit(state) {
    if (!state || state.disposed || state.outputFitFrame != null) return;
    state.outputFitFrame = requestAnimationFrame(() => {
        state.outputFitFrame = requestAnimationFrame(() => {
            state.outputFitFrame = null;
            if (state.disposed) return;
            const preserveTextOuterSize = state.mode === "diff" && (state.preserveOuterSize || state.outerSizeManual);
            if (!preserveTextOuterSize) {
                state.bottomHeightManual = false;
                if (state.mode === "media") {
                    const fitted = fitMediaViewport(state, false);
                    if (!fitted && mediaAvailableWidth(state) - 2 <= 0) scheduleMediaFitRetry(state);
                }
                else updateAutoBottomHeight(state, true);
                syncNodeHeight(state);
            }
            state.displayReady = true;
            if (preserveTextOuterSize) updateTextViewportFromOuter(state);
            syncDiffSplitToGrid(state);
            nodeGraphDirty(state.node);
            drawMedia(state);
        });
    });
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

function setViewportLayout(state, layout) {
    const next = String(layout || "").toLowerCase() === "vertical" ? "vertical" : "horizontal";
    if (state.layout !== next) state.splitRatio = 0.5;
    state.layout = next;
    state.shell.classList.toggle("cs-compare-any-layout-vertical", next === "vertical");
    if (state.mode === "diff") state.viewportAspect = textViewportAspect(next);
    applyViewportHeights(state);
    updateAutoBottomHeight(state);
    nodeGraphDirty(state.node);
}

function setViewportAspect(state, aspect) {
    const next = Number(aspect);
    state.viewportAspect = Number.isFinite(next) && next > 0 ? clamp(next, 0.05, 20) : UNKNOWN_VIEWPORT_ASPECT;
    updateAutoBottomHeight(state);
    nodeGraphDirty(state.node);
}

function setCompareZoom(state, requested) {
    state.compareZoom = clamp(Number(requested) || 1, 1, 4);
    if (state.compareZoom <= 1) {
        state.comparePanX = 0;
        state.comparePanY = 0;
    }
    const width = Number(state.canvasCompare?.clientWidth) || 0;
    const height = Number(state.canvasCompare?.clientHeight) || 0;
    if (width > 0 && height > 0) clampComparePan(state, width, height);
    state.sourceViewport?.classList.toggle("cs-compare-any-pan-ready", state.compareZoom > 1);
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
    for (const canvas of [state.canvasA, state.canvasCompare]) {
        const prepared = setupCanvas(canvas);
        if (!prepared) continue;
        prepared.context.fillStyle = "#08090b";
        prepared.context.fillRect(0, 0, prepared.width, prepared.height);
    }
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
    state.preserveOuterSize = state.mode === "diff" && state.outerSizeManual;
    state.mode = "loading";
    state.displayReady = false;
    state.bottomHeightManual = false;
    setMediaLayout(state, false);
    state.shell.classList.remove("cs-compare-any-diff-mode");
    state.shell.classList.add("cs-compare-any-media-only");
    state.controls.hidden = true;
    state.error.hidden = true;
    state.canvasA.hidden = false;
    state.canvasCompare.hidden = false;
    state.sourceText.hidden = true;
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
    const viewports = [state.sourceViewport, state.compareViewport].filter(Boolean);
    for (const viewport of viewports) {
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
            const width = Number(viewport.clientWidth) || 0;
            const height = Number(viewport.clientHeight) || 0;
            state.comparePanX = drag.panX + event.clientX - drag.startX;
            state.comparePanY = drag.panY + event.clientY - drag.startY;
            if (width > 0 && height > 0) clampComparePan(state, width, height);
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
}

function animationTick(state) {
    if (!state || state.disposed) return;
    if (state.mode === "media") {
        const master = !state.videoA.ended && state.videoA.readyState >= 2 ? state.videoA : state.videoB;
        if (state.playing && master.readyState >= 2) {
            const maxFrame = Math.max(0, state.frames - 1);
            const maxSeconds = maxFrame / Math.max(0.001, state.fps);
            const frame = clamp(Math.round(master.currentTime * state.fps), 0, maxFrame);
            if (Math.abs(state.videoA.currentTime - master.currentTime) > 1 / Math.max(1, state.fps)) state.videoA.currentTime = master.currentTime;
            if (Math.abs(state.videoB.currentTime - master.currentTime) > 1 / Math.max(1, state.fps)) state.videoB.currentTime = master.currentTime;
            if (frame !== state.frame) {
                state.frame = frame;
                updateFrameControls(state);
            }
            // Both cached previews use the same padded timeline. Stop from
            // the shared frame index so a tiny container-duration mismatch
            // cannot leave one side on a different last frame.
            if (frame >= maxFrame && (master.ended || master.currentTime >= maxSeconds)) stopPlayback(state);
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
        forceCompareRefresh(state);
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
    setMediaLayout(state, true);
    cancelAnimationFrame(state.mediaFitRetryFrame);
    state.mediaFitRetryFrame = null;
    state.mediaViewportWidth = 0;
    state.shell.style.removeProperty("--cs-compare-media-width");
    state.kind = String(payload.media_kind || "MEDIA");
    state.frames = Math.max(1, Number(timeline.frames || 1));
    state.fps = Math.max(0.001, Number(timeline.fps || 24));
    state.frame = clamp(state.frame, 0, state.frames - 1);
    state.timeline.max = String(Math.max(0, state.frames - 1));
    const urlA = mediaUrl(sources.a);
    const urlB = mediaUrl(sources.b);
    state.videoA.onloadedmetadata = () => seekFrame(state, state.frame, false);
    state.videoB.onloadedmetadata = () => seekFrame(state, state.frame, false);
    const onVideoEnded = () => {
        if (state.playing && state.videoA.ended && state.videoB.ended) stopPlayback(state);
    };
    state.videoA.onended = onVideoEnded;
    state.videoB.onended = onVideoEnded;
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

function createDiffPane(side, rows) {
    const pane = document.createElement("section");
    pane.className = `cs-compare-any-diff-pane cs-compare-any-diff-pane-${side}`;
    const label = document.createElement("div");
    label.className = "cs-compare-any-diff-label";
    label.textContent = side.toUpperCase();
    const body = document.createElement("div");
    body.className = `cs-compare-any-diff-content cs-compare-any-diff-content-${side}`;
    for (const row of rows) {
        const operation = String(row?.op || "equal");
        const line = document.createElement("div");
        line.className = `cs-compare-any-diff-line ${operation} ${side}`;
        const prefix = document.createElement("span");
        prefix.className = "cs-compare-any-diff-prefix";
        prefix.textContent = operation === "equal" || (operation === "insert" && side === "a") || (operation === "delete" && side === "b") ? " " : side === "a" ? "−" : "+";
        const lineContent = document.createElement("span");
        lineContent.className = "cs-compare-any-diff-line-content";
        appendParts(lineContent, row?.[`${side}_parts`], side);
        if (!lineContent.childNodes.length) lineContent.textContent = String(row?.[side] || "");
        line.append(prefix, lineContent);
        body.appendChild(line);
    }
    pane.append(label, body);
    return pane;
}

function renderDiff(state, payload) {
    stopCacheProgress(state);
    state.mode = "diff";
    setMediaLayout(state, false);
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
    const rows = Array.isArray(payload.rows) ? payload.rows : [];
    state.diff.replaceChildren(createDiffPane("a", rows), createDiffPane("b", rows));
    const contentA = state.diff.querySelector(".cs-compare-any-diff-content-a");
    const contentB = state.diff.querySelector(".cs-compare-any-diff-content-b");
    let scrollSyncFrame = null;
    const syncScroll = (source, target) => {
        if (!source || !target || scrollSyncFrame != null) return;
        target.scrollTop = source.scrollTop;
        target.scrollLeft = source.scrollLeft;
        scrollSyncFrame = requestAnimationFrame(() => { scrollSyncFrame = null; });
    };
    contentA?.addEventListener("scroll", () => syncScroll(contentA, contentB), { passive: true });
    contentB?.addEventListener("scroll", () => syncScroll(contentB, contentA), { passive: true });
    state.shell.classList.add("cs-compare-any-diff-mode");
    state.shell.classList.remove("cs-compare-any-media-only");
    state.divider.hidden = true;
    state.canvasA.hidden = true;
    state.canvasCompare.hidden = true;
    state.sourceText.hidden = true;
    state.diff.hidden = false;
}

function renderError(state, payload) {
    stopCacheProgress(state);
    state.mode = "error";
    setMediaLayout(state, false);
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
    state.canvasA.hidden = false;
    state.canvasCompare.hidden = false;
    state.sourceText.hidden = true;
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
    state.canvasA.hidden = false;
    state.canvasCompare.hidden = false;
    state.sourceText.hidden = true;
    state.diff.hidden = true;
    state.divider.hidden = false;
    setMediaSources(state, payload);
}

function updateNode(node, output) {
    const state = node?.[STATE];
    if (!state || !output) return;
    const payload = latestPayload(output);
    const layout = String(payload.view_port_layout || "").toLowerCase() === "vertical" ? "vertical" : "horizontal";
    const mode = String(payload.mode || "error").toLowerCase();
    const preserveOuterSize = mode === "diff" && (
        state.preserveOuterSize ||
        (state.mode === "diff" && state.outerSizeManual) ||
        (state.mode === "loading" && state.outerSizeManual)
    );
    state.layout = layout;
    state.shell.classList.toggle("cs-compare-any-layout-vertical", layout === "vertical");
    state.bottomHeightManual = false;
    state.displayReady = false;
    state.preserveOuterSize = preserveOuterSize;
    if (mode !== "diff") state.outerSizeManual = false;
    if (mode === "media") {
        setViewportAspect(state, payload.viewport_aspect);
        renderMedia(state, payload);
    } else if (mode === "diff") {
        setViewportAspect(state, textViewportAspect(layout));
        renderDiff(state, payload);
    } else {
        setViewportAspect(state, UNKNOWN_VIEWPORT_ASPECT);
        renderError(state, payload);
    }
    scheduleOutputFit(state);
    node.graph?.setDirtyCanvas?.(true, true);
}

function addViewport(node) {
    addStyles();
    const shell = document.createElement("div");
    shell.className = "cs-compare-any-shell cs-compare-any-media-only";
    shell.innerHTML = `
      <div class="cs-compare-any-head"><span class="cs-compare-any-title">CS Compare Any</span><span class="cs-compare-any-status">Waiting for execution...</span></div>
      <div class="cs-compare-any-grid">
        <div class="cs-compare-any-viewport cs-compare-any-source"><canvas class="cs-compare-any-canvas-a"></canvas><pre class="cs-compare-any-text cs-compare-any-text-a"></pre></div>
        <div class="cs-compare-any-viewport cs-compare-any-compare"><canvas class="cs-compare-any-canvas-compare"></canvas><div class="cs-compare-any-diff"></div><div class="cs-compare-any-divider" title="Drag to compare A and B"></div><div class="cs-compare-any-error" hidden></div><div class="cs-compare-any-loading" hidden>Loading comparison cache 0%</div><div class="cs-compare-any-resize-handle" data-resize="bottom" title="Resize viewport" aria-label="Resize viewport"></div></div>
        <div class="cs-compare-any-split-handle" title="Resize comparison panes" aria-label="Resize comparison panes"></div>
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
        head: shell.querySelector(".cs-compare-any-head"),
        status: shell.querySelector(".cs-compare-any-status"),
        canvasA: shell.querySelector(".cs-compare-any-canvas-a"),
        canvasCompare: shell.querySelector(".cs-compare-any-canvas-compare"),
        sourceViewport: shell.querySelector(".cs-compare-any-source"),
        compareViewport: shell.querySelector(".cs-compare-any-compare"),
        grid: shell.querySelector(".cs-compare-any-grid"),
        splitHandle: shell.querySelector(".cs-compare-any-split-handle"),
        sourceText: shell.querySelector(".cs-compare-any-text-a"),
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
        viewportAspect: UNKNOWN_VIEWPORT_ASPECT,
        frame: 0,
        frames: 1,
        fps: 24,
        playing: false,
        comparePosition: 0,
        compareZoom: 1,
        comparePanX: 0,
        comparePanY: 0,
        audioChoice: "A",
        layout: "horizontal",
        bottomHeight: DEFAULT_BOTTOM_HEIGHT,
        mediaViewportWidth: 0,
        splitRatio: 0.5,
        splitDragging: false,
        bottomHeightManual: false,
        defaultGridHeight: DEFAULT_BOTTOM_HEIGHT,
        baseNodeHeight: NODE_MIN_HEIGHT,
        node,
        disposed: false,
        displayReady: false,
        outerSizeManual: false,
        preserveOuterSize: false,
        internalResize: false,
        outerResizeFrame: null,
        outputFitFrame: null,
        mediaFitRetryFrame: null,
        animationFrame: null,
        compareRefreshFrame: null,
        progressTimer: null,
        progressSerial: 0,
    };
    widget.options.getMinHeight = () => {
        const gridMinimum = minimumGridHeight(state);
        const gridRequired = ((state.mode === "diff" && state.displayReady) || state.preserveOuterSize || state.outerSizeManual)
            ? gridMinimum
            : gridHeight(state);
        return Math.max(0, gridRequired + shellChromeHeight(state) + 4);
    };
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
    state.compareViewport.style.setProperty("--compare-position", `${state.comparePosition}%`);
    applyViewportHeights(state);
    updateAutoBottomHeight(state);
    state.shell.querySelectorAll("[data-resize]").forEach((handle) => attachHeightHandle(state, handle));
    attachDiffSplitHandle(state);
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
    const stopDragging = (event) => { if (!dragging) return; dragging = false; state.divider.releasePointerCapture?.(event.pointerId); forceCompareRefresh(state); };
    state.divider.addEventListener("pointerup", stopDragging);
    state.divider.addEventListener("pointercancel", stopDragging);
    const originalResize = node.onResize;
    node.onResize = function () {
        const result = originalResize?.apply(this, arguments);
        if (!state.internalResize) {
            if (state.displayReady && state.mode === "diff") state.outerSizeManual = true;
            scheduleOuterResize(state);
        }
        return result;
    };
    const resizeObserver = new ResizeObserver(() => {
        if (state.mode === "media" && state.mediaViewportWidth <= 1 && mediaAvailableWidth(state) - 2 > 0) {
            fitMediaViewport(state, false);
            syncNodeHeight(state);
        } else {
            syncDiffSplitToGrid(state);
        }
        drawMedia(state);
    });
    resizeObserver.observe(state.grid);
    state.dispose = () => { state.disposed = true; stopCacheProgress(state); cancelAnimationFrame(state.animationFrame); cancelAnimationFrame(state.compareRefreshFrame); cancelAnimationFrame(state.outerResizeFrame); cancelAnimationFrame(state.outputFitFrame); cancelAnimationFrame(state.mediaFitRetryFrame); resizeObserver.disconnect(); videoA.pause(); videoB.pause(); videoA.removeAttribute("src"); videoB.removeAttribute("src"); };
    node[STATE] = state;
    node[WIDGET] = widget;
    scheduleOutputFit(state);
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
            this.setSize?.([NODE_MIN_WIDTH, NODE_MIN_HEIGHT]);
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
        const width = Math.max(NODE_MIN_WIDTH, Number(node.size?.[0]) || NODE_MIN_WIDTH);
        const storedHeight = Number(node.size?.[1]) || NODE_MIN_HEIGHT;
        // 650px was the old two-viewport default. Collapse that legacy size
        // so loaded nodes do not retain an empty upper area.
        const height = storedHeight === 650 ? NODE_MIN_HEIGHT : Math.max(NODE_MIN_HEIGHT, storedHeight);
        if (node[STATE]) {
            node[STATE].baseNodeHeight = NODE_MIN_HEIGHT;
        }
        node.setSize?.([width, height]);
    },
    onNodeOutputsUpdated(outputs) {
        for (const [locator, output] of Object.entries(outputs || {})) {
            const node = graphNode(locator);
            if (node && (node.type === NODE_ID || node.comfyClass === NODE_ID)) updateNode(node, output);
        }
    },
});
