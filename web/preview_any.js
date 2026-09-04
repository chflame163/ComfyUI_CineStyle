import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_ID = "CS_Preview_Any";
const STYLE_ID = "cinestyle-preview-any-style";
const AUDIO_WIDGET = Symbol("cinestyle-preview-any-audio");
const TEXT_WIDGET = Symbol("cinestyle-preview-any-text");
const CURRENT_KIND = Symbol("cinestyle-preview-any-current-kind");
const LAYOUT_HOOK = Symbol("cinestyle-preview-any-layout-hook");
const RESIZE_HOOK = Symbol("cinestyle-preview-any-resize-hook");
const LAYOUT_FRAME = Symbol("cinestyle-preview-any-layout-frame");
const TEXT_OUTER_SYNC = Symbol("cinestyle-preview-any-text-outer-sync");
const OUTPUT_SIGNATURE = Symbol("cinestyle-preview-any-output-signature");
const MEDIA_SIZE = Symbol("cinestyle-preview-any-media-size");
const CANVAS_IMAGE_WIDGET = "$$canvas-image-preview";
const TEXT_DEFAULT_HEIGHT = 150;
const TEXT_MIN_HEIGHT = 90;
const TEXT_MAX_HEIGHT = Number.POSITIVE_INFINITY;
const TEXT_WIDGET_MARGIN = 8;
const NODE_MIN_WIDTH = 420;
const NODE_MIN_HEIGHT = 240;
// 58px waveform + 36px controls + 8px gap + 16px padding + 2px border.
const AUDIO_VIEWPORT_HEIGHT = 120;
const AUDIO_WIDGET_HEIGHT = AUDIO_VIEWPORT_HEIGHT + TEXT_WIDGET_MARGIN;

function positiveNumber(value, fallback = 0) {
    const number = Number(value);
    return Number.isFinite(number) && number > 0 ? number : fallback;
}

function mediaAspect(node) {
    const size = node?.[MEDIA_SIZE];
    const width = positiveNumber(size?.width);
    const height = positiveNumber(size?.height);
    return width && height ? width / height : 0;
}

function mediaContentWidth(node) {
    // ComfyUI's image/video preview widgets use the node content width for
    // their aspect calculation; keep the same reference so the viewport and
    // the encoded media do not accumulate an extra margin error.
    return Math.max(1, Number(node?.size?.[0]) || NODE_MIN_WIDTH);
}

function mediaMinimumHeight(node, kind) {
    if (kind === "audio") return AUDIO_WIDGET_HEIGHT;
    const aspect = mediaAspect(node);
    if (!aspect) return kind === "image" || kind === "video" ? 220 : 0;
    return Math.max(1, Math.round(mediaContentWidth(node) / aspect));
}

function outerResizeZone(event, element) {
    const rect = element?.getBoundingClientRect?.();
    if (!rect) return false;
    return event.clientX >= rect.right - 24 && event.clientY >= rect.bottom - 22;
}

function isTextOnlyKind(kind) {
    return kind !== "image" && kind !== "video" && kind !== "audio";
}

function addStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .cs-preview-any-audio { display:grid; box-sizing:border-box; width:100%; height:${AUDIO_VIEWPORT_HEIGHT}px; min-height:${AUDIO_VIEWPORT_HEIGHT}px; max-height:${AUDIO_VIEWPORT_HEIGHT}px; gap:8px; padding:8px; border:1px solid var(--border-color,#454b55); border-radius:6px; background:var(--comfy-input-bg,#17191e); }
      .cs-preview-any-audio[hidden] { display:none; }
      .cs-preview-any-wave { display:block; width:100%; height:58px; border-radius:4px; background:var(--comfy-menu-bg,#111318); }
      .cs-preview-any-audio audio { display:block; width:100%; height:36px; }
      .cs-preview-any-text { position:relative; box-sizing:border-box; width:100%; height:var(--cs-preview-any-text-height,150px); min-height:var(--cs-preview-any-text-height,150px); max-height:none; overflow:auto; border:1px solid var(--border-color,#454b55); border-radius:6px; padding:9px 10px 18px; background:var(--comfy-input-bg,#17191e); color:var(--input-text,#e6e9ef); }
      .cs-preview-any-text pre { min-width:100%; width:max-content; margin:0; white-space:pre-wrap; overflow-wrap:anywhere; font:12px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace; letter-spacing:0; }
      .cs-preview-any-text-resize { position:absolute; right:28px; bottom:2px; left:6px; z-index:2; height:10px; cursor:ns-resize; touch-action:none; }
      .cs-preview-any-text-resize::after { content:""; position:absolute; top:4px; left:50%; width:34px; height:2px; border-radius:2px; transform:translateX(-50%); background:var(--descrip-text,#7f8794); opacity:.75; }
    `;
    document.head.appendChild(style);
}

function addTextViewport(node) {
    const container = document.createElement("div");
    container.className = "cs-preview-any-text";
    container.style.setProperty("--cs-preview-any-text-height", `${TEXT_DEFAULT_HEIGHT}px`);
    const pre = document.createElement("pre");
    pre.textContent = "Waiting for input...";
    const resizeHandle = document.createElement("div");
    resizeHandle.className = "cs-preview-any-text-resize";
    resizeHandle.setAttribute("role", "separator");
    resizeHandle.setAttribute("aria-label", "Resize text viewport");
    container.append(pre, resizeHandle);
    for (const event of ["pointerdown", "click", "dblclick"]) {
        container.addEventListener(event, (value) => {
            if (event === "pointerdown" && outerResizeZone(value, container)) return;
            value.stopPropagation();
        });
    }
    const state = {
        container,
        pre,
        resizeHandle,
        widget: null,
        height: TEXT_DEFAULT_HEIGHT,
        drag: null,
        locked: false,
    };
    node[CURRENT_KIND] ??= "none";
    const widget = node.addDOMWidget("preview_text", "preview_text", container, {
        margin: TEXT_WIDGET_MARGIN / 2,
    });
    state.widget = widget;
    widget.serialize = false;
    widget.options = {
        ...(widget.options || {}),
        margin: TEXT_WIDGET_MARGIN / 2,
        serialize: false,
        canvasOnly: false,
        getMinHeight: () => TEXT_MIN_HEIGHT + TEXT_WIDGET_MARGIN,
        // Text consumes the remaining space after the media widget. The CSS
        // height is synchronized from computedHeight after every outer resize.
        getMaxHeight: () => undefined,
    };

    const finishResize = (event) => {
        if (!state.drag || event.pointerId !== state.drag.pointerId) return;
        state.drag = null;
        resizeHandle.releasePointerCapture?.(event.pointerId);
        node.graph?.setDirtyCanvas?.(true, true);
    };
    resizeHandle.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        event.stopPropagation();
        state.drag = {
            pointerId: event.pointerId,
            startY: event.clientY,
            startHeight: state.height,
        };
        state.locked = true;
        resizeHandle.setPointerCapture?.(event.pointerId);
    });
    resizeHandle.addEventListener("pointermove", (event) => {
        if (!state.drag || event.pointerId !== state.drag.pointerId) return;
        event.preventDefault();
        const nextHeight = state.drag.startHeight + event.clientY - state.drag.startY;
        setTextViewportHeight(node, state, nextHeight);
    });
    resizeHandle.addEventListener("pointerup", finishResize);
    resizeHandle.addEventListener("pointercancel", finishResize);
    node[TEXT_WIDGET] = state;
    return state;
}

function setTextViewportHeight(node, state, requestedHeight) {
    const nextHeight = Math.max(TEXT_MIN_HEIGHT, Math.min(TEXT_MAX_HEIGHT, Math.round(requestedHeight)));
    if (nextHeight === state.height) return;
    const delta = nextHeight - state.height;
    state.height = nextHeight;
    state.container.style.setProperty("--cs-preview-any-text-height", `${nextHeight}px`);

    const currentWidth = Number(node.size?.[0]) || 420;
    const currentHeight = Number(node.size?.[1]) || 250;
    node.setSize?.([currentWidth, Math.max(NODE_MIN_HEIGHT, currentHeight + delta)]);
    node.arrange?.();
    ensureMinimumNodeHeight(node);
    scheduleLayoutSync(node);
    node.graph?.setDirtyCanvas?.(true, true);
}

function syncTextViewportToOuter(node) {
    const state = node?.[TEXT_WIDGET];
    if (!state) return;
    node.arrange?.();
    const computedHeight = Number(state.widget?.computedHeight);
    if (Number.isFinite(computedHeight)) {
        const nextHeight = Math.max(
            TEXT_MIN_HEIGHT,
            Math.min(TEXT_MAX_HEIGHT, Math.round(computedHeight - TEXT_WIDGET_MARGIN)),
        );
        state.height = nextHeight;
        state.container.style.setProperty("--cs-preview-any-text-height", `${nextHeight}px`);
    }
    node.graph?.setDirtyCanvas?.(true, true);
}

function scheduleTextOuterSync(node) {
    const state = node?.[TEXT_WIDGET];
    if (!state || state[TEXT_OUTER_SYNC] != null) return;
    state[TEXT_OUTER_SYNC] = requestAnimationFrame(() => {
        state[TEXT_OUTER_SYNC] = null;
        if (!state.drag) syncTextViewportToOuter(node);
    });
}

function updateTextViewport(node, output) {
    const state = node[TEXT_WIDGET] || addTextViewport(node);
    const text = output?.text;
    state.pre.textContent = Array.isArray(text) ? text.join("\n\n") : String(text || "");
}

function audioUrl(result) {
    if (!result?.filename) return "";
    const params = new URLSearchParams({
        filename: result.filename,
        subfolder: result.subfolder || "",
        type: result.type || "temp",
    });
    return api.apiURL(`/view?${params}`);
}

function drawWaveform(state) {
    const { canvas, bars, audio } = state;
    const bounds = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(bounds.width));
    const height = Math.max(1, Math.round(bounds.height));
    const ratio = Math.max(1, window.devicePixelRatio || 1);
    const pixelWidth = Math.round(width * ratio);
    const pixelHeight = Math.round(height * ratio);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
        canvas.width = pixelWidth;
        canvas.height = pixelHeight;
    }
    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    const styles = getComputedStyle(canvas);
    const idle = styles.getPropertyValue("--descrip-text").trim() || "#7f8794";
    const active = styles.getPropertyValue("--input-text").trim() || "#e6e9ef";
    const played = Number.isFinite(audio.duration) && audio.duration > 0
        ? Math.max(0, Math.min(1, audio.currentTime / audio.duration))
        : 0;
    context.clearRect(0, 0, width, height);
    if (!bars.length) return;
    const step = width / bars.length;
    const lineWidth = Math.max(1, Math.min(3, step * 0.58));
    context.lineCap = "round";
    context.lineWidth = lineWidth;
    for (let index = 0; index < bars.length; index += 1) {
        const amplitude = Math.max(0.04, Math.min(1, Number(bars[index]) || 0));
        const barHeight = amplitude * (height - 8);
        const x = (index + 0.5) * step;
        context.strokeStyle = index / bars.length <= played ? active : idle;
        context.beginPath();
        context.moveTo(x, (height - barHeight) / 2);
        context.lineTo(x, (height + barHeight) / 2);
        context.stroke();
    }
}

function addAudioViewport(node) {
    // A serialized/stale widget can survive a graph reload. Keep this node's
    // audio layout to one widget so hidden audio never accumulates rows.
    for (const widget of [...(node.widgets || [])].filter((item) => item.name === "preview_audio")) {
        removeLegacyWidget(node, widget.name, true);
    }
    const container = document.createElement("div");
    container.className = "cs-preview-any-audio";
    container.hidden = true;
    const canvas = document.createElement("canvas");
    canvas.className = "cs-preview-any-wave";
    canvas.setAttribute("aria-label", "Audio waveform");
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "metadata";
    audio.classList.add("comfy-audio");
    audio.setAttribute("name", "media");
    container.append(canvas, audio);
    for (const event of ["pointerdown", "click", "dblclick"]) {
        container.addEventListener(event, (value) => {
            if (event === "pointerdown" && outerResizeZone(value, container)) return;
            value.stopPropagation();
        });
    }

    const widget = node.addDOMWidget("preview_audio", "preview_audio", container, {
        margin: TEXT_WIDGET_MARGIN / 2,
    });
    widget.serialize = false;
    widget.hidden = true;
    widget.options = {
        ...(widget.options || {}),
        margin: TEXT_WIDGET_MARGIN / 2,
        serialize: false,
        canvasOnly: false,
        getMinHeight: () => container.hidden ? 0 : AUDIO_WIDGET_HEIGHT,
        getMaxHeight: () => container.hidden ? 0 : AUDIO_WIDGET_HEIGHT,
    };
    const state = { container, canvas, audio, widget, bars: [] };
    const repaint = () => drawWaveform(state);
    audio.addEventListener("timeupdate", repaint);
    audio.addEventListener("durationchange", repaint);
    audio.addEventListener("loadedmetadata", repaint);
    new ResizeObserver(repaint).observe(canvas);
    node[AUDIO_WIDGET] = state;
    return state;
}

function updateAudioViewport(node, output, payload) {
    const show = payload?.kind === "audio" && output?.audio?.length;
    if (!show) {
        const state = node[AUDIO_WIDGET];
        if (!state) return;
        state.container.hidden = true;
        state.widget.hidden = true;
        state.audio.pause();
        state.audio.removeAttribute("src");
        state.audio.load();
        state.bars = [];
        drawWaveform(state);
        return;
    }
    const state = node[AUDIO_WIDGET] || addAudioViewport(node);
    state.container.hidden = false;
    state.widget.hidden = false;
    const url = audioUrl(output.audio[0]);
    if (state.audio.src !== url) {
        state.audio.src = url;
        state.audio.load();
    }
    state.bars = Array.isArray(payload.waveform) ? payload.waveform : [];
    requestAnimationFrame(() => drawWaveform(state));
}

function latestPayload(output) {
    const values = output?.preview_any;
    if (!Array.isArray(values) || !values.length) return {};
    const payload = values[values.length - 1];
    return payload && typeof payload === "object" ? payload : {};
}

function previewEntryKey(value) {
    if (!value || typeof value !== "object") return String(value ?? "");
    return [
        value.filename,
        value.subfolder,
        value.type,
        value.url,
        value.video_url,
        value.audio_url,
    ].map((item) => String(item ?? "")).join("\u001f");
}

function samePreviewEntries(first, second) {
    if (!Array.isArray(first) || !Array.isArray(second) || first.length !== second.length) return false;
    return first.every((item, index) => previewEntryKey(item) === previewEntryKey(second[index]));
}

function outputSignature(output, payload) {
    const text = output?.text;
    const textKey = Array.isArray(text) ? text.map((item) => String(item ?? "")).join("\u001e") : String(text ?? "");
    const images = Array.isArray(output?.images) ? output.images.map(previewEntryKey).join("\u001e") : "";
    const audio = Array.isArray(output?.audio) ? output.audio.map(previewEntryKey).join("\u001e") : "";
    let payloadKey;
    try {
        payloadKey = JSON.stringify(payload || {});
    } catch {
        payloadKey = String(payload?.kind || "none");
    }
    return [payloadKey, textKey, images, audio].join("\u001d");
}

function mediaWidget(node, kind) {
    const name = kind === "image" ? CANVAS_IMAGE_WIDGET : kind === "video" ? "video-preview" : null;
    return name ? node.widgets?.find((widget) => widget.name === name) : null;
}

function syncMediaWidgetLayout(node, kind) {
    const widget = mediaWidget(node, kind);
    if (!widget) return false;
    if (kind === "image" || kind === "video") {
        // Keep the source aspect ratio as the minimum media height used for
        // the initial node expansion. Do not set maxHeight: the user-owned
        // outer node height must be allowed to grow the viewport freely. Read
        // the current node width on every layout pass so a width drag cannot
        // briefly use a stale aspect-derived height.
        widget.computeLayoutSize = () => ({
            minHeight: mediaMinimumHeight(node, kind),
            minWidth: 1,
        });
    }
    return true;
}

function minimumNodeHeight(node, kind) {
    const computed = Number(node.computeSize?.()?.[1]) || NODE_MIN_HEIGHT;
    const mediaWidgetPresent = kind === "image" || kind === "video"
        ? Boolean(mediaWidget(node, kind))
        : kind === "audio"
            ? Boolean(node[AUDIO_WIDGET]?.widget && !node[AUDIO_WIDGET].widget.hidden)
            : false;
    const pendingMedia = !mediaWidgetPresent && (kind === "image" || kind === "video")
        ? mediaMinimumHeight(node, kind) + 4
        : 0;
    return Math.max(NODE_MIN_HEIGHT, computed + pendingMedia);
}

function ensureMinimumNodeHeight(node) {
    const kind = node?.[CURRENT_KIND] || "none";
    const minimum = minimumNodeHeight(node, kind);
    const width = Math.round(Number(node?.size?.[0]) || NODE_MIN_WIDTH);
    const currentHeight = Math.round(Number(node?.size?.[1]) || NODE_MIN_HEIGHT);
    if (currentHeight >= minimum) return false;
    node.setSize?.([width, minimum]);
    node.graph?.setDirtyCanvas?.(true, true);
    return true;
}

function updateAutomaticSize(node, kind) {
    syncMediaWidgetLayout(node, kind);
    ensureMinimumNodeHeight(node);
    node.arrange?.();
    syncTextViewportToOuter(node);
}

function hasLegacyPreview(node) {
    return Boolean(
        (Array.isArray(node.imgs) && node.imgs.length) ||
        node.videoContainer ||
        node.widgets?.some((widget) => widget.name === CANVAS_IMAGE_WIDGET || widget.name === "video-preview"),
    );
}

function clearImagePreview(node) {
    const legacyImages = Array.isArray(node.imgs) ? [...node.imgs] : [];
    for (const value of legacyImages) unloadLegacyMedia(value);
    const removed = removeLegacyWidget(node, CANVAS_IMAGE_WIDGET, true);
    const changed = Boolean(legacyImages.length || removed);
    node.imgs = [];
    node.imageIndex = 0;
    return changed;
}

function clearVideoPreview(node) {
    const hadVideo = Boolean(node.videoContainer || node.widgets?.some((widget) => widget.name === "video-preview"));
    unloadLegacyMedia(node.videoContainer);
    removeLegacyWidget(node, "video-preview", true);
    node.videoContainer = undefined;
    return hadVideo;
}

function clearIncompatibleMedia(node, kind) {
    if (kind === "image") return clearVideoPreview(node);
    if (kind === "video") return clearImagePreview(node);
    if (!hasLegacyPreview(node)) return false;
    clearLegacyMedia(node);
    return true;
}

function syncPreviewMetadata(node, kind) {
    if (kind === "image") node.previewMediaType = "image";
    else if (kind === "video") node.previewMediaType = "video";
    else if (kind === "audio") node.previewMediaType = "audio";
    else {
        node.previewMediaType = undefined;
        node.images = [];
    }
}

function scheduleLayoutSync(node) {
    if (!node || node[LAYOUT_FRAME] != null) return;
    node[LAYOUT_FRAME] = requestAnimationFrame(() => {
        node[LAYOUT_FRAME] = null;
        const kind = node[CURRENT_KIND];
        if (kind === undefined) return;
        const cleared = clearIncompatibleMedia(node, kind);
        syncPreviewMetadata(node, kind);
        updateAutomaticSize(node, kind);
        if (cleared) node.graph?.setDirtyCanvas?.(true, true);
    });
}

function installLayoutHook(node) {
    if (!node) return;
    if (!node[RESIZE_HOOK]) {
        const onResize = node.onResize;
        node.onResize = function () {
            const result = onResize?.apply(this, arguments);
            scheduleTextOuterSync(this);
            scheduleLayoutSync(this);
            return result;
        };
        node[RESIZE_HOOK] = true;
    }
    if (node[LAYOUT_HOOK]) return;
    const onDrawBackground = node.onDrawBackground;
    if (typeof onDrawBackground !== "function") return;
    node.onDrawBackground = function () {
        const result = onDrawBackground.apply(this, arguments);
        // A previous image/video load can finish after the input changed. Do
        // this second guard in the same frame so stale media cannot remain in
        // the DOM until the next output update.
        const kind = this[CURRENT_KIND];
        if (kind !== undefined) {
            const cleared = clearIncompatibleMedia(this, kind);
            syncPreviewMetadata(this, kind);
            syncMediaWidgetLayout(this, kind);
            ensureMinimumNodeHeight(this);
            if (cleared) this.graph?.setDirtyCanvas?.(true, true);
        }
        scheduleLayoutSync(this);
        return result;
    };
    node[LAYOUT_HOOK] = true;
}

function unloadLegacyMedia(value) {
    if (!value) return;
    const elements = [];
    if (typeof Element !== "undefined" && value instanceof Element) {
        if (value.matches("video, audio, img")) elements.push(value);
        elements.push(...value.querySelectorAll("video, audio, img"));
    } else {
        elements.push(value);
    }
    for (const element of elements) {
        if (typeof HTMLMediaElement !== "undefined" && element instanceof HTMLMediaElement) {
            element.pause();
            element.removeAttribute("src");
            element.load();
        } else if (typeof HTMLImageElement !== "undefined" && element instanceof HTMLImageElement) {
            element.removeAttribute("src");
        }
    }
}

function removeLegacyWidget(node, widgetName, all = false) {
    const widgets = (node.widgets || []).filter((item) => item.name === widgetName);
    if (!widgets.length) return false;
    const targets = all ? widgets : widgets.slice(0, 1);
    for (const widget of targets) {
        if (typeof node.removeWidget === "function") {
            node.removeWidget(widget);
            continue;
        }
        const index = node.widgets.indexOf(widget);
        if (index >= 0) node.widgets.splice(index, 1);
        widget.onRemove?.();
    }
    return true;
}

function clearLegacyMedia(node) {
    // ComfyUI's legacy image/video previews reserve space through widgets,
    // so clearing only imgs/videoContainer leaves a blank preview slot.
    clearImagePreview(node);
    clearVideoPreview(node);
    node.images = [];
}

function updateNode(node, output, disposeLegacyMedia = true) {
    if (!node || !output) return;
    const payload = latestPayload(output);
    const kind = payload.kind || "none";
    const kindChanged = node[CURRENT_KIND] !== kind;
    const signature = outputSignature(output, payload);
    const duplicateOutput = !kindChanged && node[OUTPUT_SIGNATURE] === signature;
    if (duplicateOutput) {
        // onExecuted and onNodeOutputsUpdated can deliver the same UI payload.
        // Keep the DOM content current, but do not run another size transition.
        updateTextViewport(node, output);
        updateAudioViewport(node, output, payload);
        scheduleLayoutSync(node);
        return;
    }
    node[OUTPUT_SIGNATURE] = signature;
    node[CURRENT_KIND] = kind;
    const mediaWidth = positiveNumber(payload.media_width);
    const mediaHeight = positiveNumber(payload.media_height);
    node[MEDIA_SIZE] = mediaWidth && mediaHeight ? { width: mediaWidth, height: mediaHeight } : null;
    const textState = node[TEXT_WIDGET];
    if (kindChanged && textState && isTextOnlyKind(kind)) {
        // Let the outer-node resize determine the text allocation after a
        // media widget has been removed.
        textState.locked = false;
    }
    // Removing a compatible preview before every output forces ComfyUI to
    // recreate its media widget and re-run layout. Only clear on a type
    // transition; stale incompatible media is handled by the draw hook.
    if (disposeLegacyMedia && kindChanged) clearLegacyMedia(node);
    const images = kind === "image" || kind === "video"
        ? (Array.isArray(output.images) ? output.images : [])
        : [];
    if (!samePreviewEntries(node.images, images)) node.images = images.slice();
    if (disposeLegacyMedia || (kind !== "image" && kind !== "video")) {
        node.imgs = [];
        node.imageIndex = 0;
    }
    updateTextViewport(node, output);
    updateAudioViewport(node, output, payload);
    if (kind === "video") node.previewMediaType = "video";
    else if (kind === "image") node.previewMediaType = "image";
    else if (kind === "audio") node.previewMediaType = "audio";
    else node.previewMediaType = undefined;
    updateAutomaticSize(node, kind, kindChanged);
    if (isTextOnlyKind(kind)) scheduleTextOuterSync(node);
    scheduleLayoutSync(node);
    node.graph?.setDirtyCanvas?.(true, true);
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

addStyles();

app.registerExtension({
    name: "CineStyle.PreviewAny",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_ID) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            addTextViewport(this);
            this.setSize?.([NODE_MIN_WIDTH, 250]);
            installLayoutHook(this);
            scheduleTextOuterSync(this);
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            updateNode(this, output);
            onExecuted?.apply(this, arguments);
            // The core callback may create the current image/video element in
            // legacy mode. Sync metadata again without clearing that element.
            updateNode(this, output, false);
        };
    },
    loadedGraphNode(node) {
        if (node?.type !== NODE_ID && node?.comfyClass !== NODE_ID) return;
        installLayoutHook(node);
        // Preserve the saved outer size. The first output/layout pass will
        // only raise the height when the current content cannot fit.
        node.setSize?.([
            Math.round(Number(node.size?.[0]) || NODE_MIN_WIDTH),
            Math.max(NODE_MIN_HEIGHT, Number(node.size?.[1]) || NODE_MIN_HEIGHT),
        ]);
        scheduleLayoutSync(node);
    },
    onNodeOutputsUpdated(outputs) {
        for (const [locator, output] of Object.entries(outputs || {})) {
            const node = graphNode(locator);
            if (node?.type === NODE_ID || node?.comfyClass === NODE_ID) updateNode(node, output);
        }
    },
});
