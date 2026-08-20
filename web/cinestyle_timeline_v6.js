import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_ID = "CS_Load_Video";
const STYLE_ID = "cinestyle-timeline-style";
const DEFAULT_TIMELINE_VALUES = Object.freeze({
    start_frame: 0,
    end_frame: -1,
    keep_aspect_ratio: true,
    multiple: 32,
    width: 0,
    height: 0,
    fps: 0,
});

function widget(node, name) {
    return node.widgets?.find((item) => item.name === name);
}

function setWidgetValue(node, name, value) {
    const target = widget(node, name);
    if (!target) return;
    target.value = value;
    target.callback?.(value);
}

function resetTimelineWidgets(node) {
    for (const [name, value] of Object.entries(DEFAULT_TIMELINE_VALUES)) {
        setWidgetValue(node, name, value);
    }
    node.graph?.setDirtyCanvas(true, true);
}

function syncVideoSelection(node, filename) {
    const nextFilename = String(filename ?? "");
    if (node.__csTimelineVideo !== undefined && node.__csTimelineVideo !== nextFilename) {
        resetTimelineWidgets(node);
    }
    node.__csTimelineVideo = nextFilename;
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function roundToMultiple(value, multiple) {
    const safeMultiple = Math.max(1, Math.round(Number(multiple) || 1));
    return Math.max(safeMultiple, Math.floor(Number(value) / safeMultiple + 0.5) * safeMultiple);
}

function formatTime(seconds) {
    const safe = Math.max(0, Number(seconds) || 0);
    const minutes = Math.floor(safe / 60);
    const remainder = safe - minutes * 60;
    return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(2).padStart(5, "0")}`;
}

function formatDuration(seconds) {
    const safe = Math.max(0, Number(seconds) || 0);
    if (safe < 3600) return formatTime(safe);
    const hours = Math.floor(safe / 3600);
    const minutes = Math.floor((safe - hours * 3600) / 60);
    const remainder = safe - hours * 3600 - minutes * 60;
    return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${remainder.toFixed(2).padStart(5, "0")}`;
}

function formatAspectRatio(width, height) {
    const safeWidth = Math.max(1, Math.round(Number(width) || 1));
    const safeHeight = Math.max(1, Math.round(Number(height) || 1));
    let a = safeWidth;
    let b = safeHeight;
    while (b) [a, b] = [b, a % b];
    return `${safeWidth / a}:${safeHeight / a}`;
}

function formatOriginalInfo(info) {
    const audio = info.audio_format ? String(info.audio_format).toUpperCase() : "无音频";
    return `宽 ${info.width}px · 高 ${info.height}px · 画幅 ${formatAspectRatio(info.width, info.height)} · 帧率 ${Number(info.fps || 0).toFixed(2)} fps · 总长度 ${formatDuration(info.duration)} · 音频 ${audio}`;
}

function addStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .cs-timeline-dialog { width: min(980px, 94vw); max-width: none; max-height: 92vh; overflow: auto; padding: 0; border: 1px solid #343943;
        border-radius: 10px; background: #17191e; color: #e6e9ef; box-shadow: 0 22px 80px #000b; }
      .cs-timeline-dialog::backdrop { background: #050609b8; }
      .cs-timeline-shell { display: grid; gap: 14px; padding: 18px; font: 13px/1.35 system-ui, sans-serif; }
      .cs-timeline-head, .cs-timeline-foot, .cs-timeline-row { display: flex; align-items: center; gap: 10px; }
      .cs-timeline-head { justify-content: space-between; }
      .cs-timeline-head > div { min-width: 0; flex: 1; }
      .cs-timeline-title { margin: 0; font-size: 16px; font-weight: 600; }
      .cs-timeline-muted { color: #9299a8; }
      .cs-timeline-close { border: 0; background: transparent; color: #aeb5c2; font-size: 22px; cursor: pointer; padding: 0 5px; }
      .cs-timeline-video { width: 100%; aspect-ratio: 16 / 9; height: auto; display: block; background: #08090b; border-radius: 6px; object-fit: contain; }
      .cs-file-name { min-width: 0; max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .cs-original-info { max-width: 100%; color: #aeb5c2; font-size: 12px; line-height: 1.45; overflow-wrap: anywhere; word-break: break-word; }
      .cs-timeline-readout { display: flex; justify-content: space-between; color: #aeb5c2; font-variant-numeric: tabular-nums; }
      .cs-timeline-pointer-row { position: relative; height: 16px; margin-bottom: -4px; user-select: none; touch-action: none; }
      .cs-timeline-pointer { position: absolute; top: 0; left: 0; width: 18px; height: 16px; transform: translateX(-50%); padding: 0; border: 0; border-radius: 2px; background: #55a9f5; clip-path: polygon(0 0, 100% 0, 50% 100%); cursor: ew-resize; z-index: 3; }
      .cs-timeline-pointer:hover, .cs-timeline-pointer:focus-visible { background: #78bcff; outline: none; }
      .cs-timeline-track { position: relative; height: 48px; border-radius: 6px; background: #292d35; cursor: crosshair; user-select: none; touch-action: none; }
      .cs-timeline-track::before { content: ""; position: absolute; inset: 17px 0 17px; background: repeating-linear-gradient(90deg, #4b5360 0 1px, transparent 1px 10%); opacity: .6; }
      .cs-timeline-selection { position: absolute; top: 13px; bottom: 13px; background: #55a9f5; opacity: .88; border-radius: 3px; }
      .cs-timeline-handle { position: absolute; top: 5px; bottom: 5px; width: 12px; transform: translateX(-50%); border: 0; border-radius: 3px; background: #f5f7fb; box-shadow: 0 0 0 1px #16181c, 0 2px 8px #0008; cursor: ew-resize; z-index: 2; }
      .cs-timeline-handle::after { content: ""; position: absolute; left: 4px; top: 17px; width: 4px; height: 14px; border-left: 1px solid #6b7280; border-right: 1px solid #6b7280; }
      .cs-timeline-controls { display: flex; gap: 6px; }
      .cs-timeline-controls button, .cs-timeline-foot button { border: 1px solid #424956; border-radius: 5px; padding: 7px 12px; background: #242832; color: #e6e9ef; cursor: pointer; }
      .cs-timeline-controls button:hover, .cs-timeline-foot button:hover { background: #303643; }
      .cs-timeline-fields { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; }
      .cs-timeline-field { display: grid; gap: 5px; color: #9da5b4; }
      .cs-timeline-field input { width: 100%; box-sizing: border-box; border: 1px solid #424956; border-radius: 5px; padding: 7px 8px; background: #20232a; color: #f2f4f7; }
      .cs-timeline-check { align-content: start; }
      .cs-timeline-check span { display: flex; align-items: center; gap: 6px; min-height: 32px; color: #e6e9ef; }
      .cs-timeline-check input { width: auto; }
      .cs-timeline-foot { justify-content: flex-end; }
      .cs-timeline-foot .cs-apply { background: #317ec4; border-color: #4b9de8; }
      @media (max-width: 640px) { .cs-timeline-fields { grid-template-columns: 1fr; } .cs-timeline-shell { padding: 12px; } }
    `;
    document.head.append(style);
}

function videoUrl(filename) {
    const params = new URLSearchParams({ filename, type: "input", subfolder: "", t: String(Date.now()) });
    return api.apiURL(`/view?${params.toString()}`);
}

async function fetchInfo(filename) {
    const params = new URLSearchParams({ filename });
    const response = await api.fetchApi(`/cinestyle/video-info?${params.toString()}`);
    if (!response.ok) throw new Error(await response.text());
    return await response.json();
}

function openTimeline(node) {
    const filename = widget(node, "video")?.value;
    if (!filename) {
        app.canvas?.prompt?.("Choose a video before opening the timeline", "");
        return;
    }
    addStyles();
    const dialog = document.createElement("dialog");
    dialog.className = "cs-timeline-dialog";
    dialog.innerHTML = `
      <div class="cs-timeline-shell">
        <div class="cs-timeline-head"><div><h2 class="cs-timeline-title">Edit Timeline</h2><div class="cs-timeline-muted cs-file-name"></div><div class="cs-original-info"></div></div><button class="cs-timeline-close" type="button" aria-label="Close">&times;</button></div>
        <video class="cs-timeline-video" controls muted playsinline preload="metadata"></video>
        <div class="cs-timeline-readout"><span class="cs-current">00:00.00</span><span class="cs-range"></span><span class="cs-duration">00:00.00</span></div>
        <div class="cs-timeline-pointer-row" aria-label="Current frame"><button class="cs-timeline-pointer" type="button" aria-label="Drag current frame" title="Drag current frame"></button></div>
        <div class="cs-timeline-track" aria-label="Video timeline"><div class="cs-timeline-selection"></div><button class="cs-timeline-handle cs-in" type="button" aria-label="In point"></button><button class="cs-timeline-handle cs-out" type="button" aria-label="Out point"></button></div>
        <div class="cs-timeline-controls"><button class="cs-set-in" type="button">Set In</button><button class="cs-back" type="button">|&lt;</button><button class="cs-play" type="button">Play</button><button class="cs-forward" type="button">&gt;|</button><button class="cs-set-out" type="button">Set Out</button></div>
        <div class="cs-timeline-fields"><label class="cs-timeline-field cs-timeline-check"><span><input class="cs-keep-aspect" type="checkbox"> keep aspect ratio</span></label><label class="cs-timeline-field">multiple<input class="cs-multiple" type="number" min="1" step="1"></label><label class="cs-timeline-field">Width<input class="cs-width" type="number" min="1" step="1"></label><label class="cs-timeline-field">Height<input class="cs-height" type="number" min="1" step="1"></label><label class="cs-timeline-field">FPS<input class="cs-fps" type="number" min="0.01" max="240" step="0.01"></label></div>
        <div class="cs-timeline-foot"><button class="cs-cancel" type="button">Cancel</button><button class="cs-apply" type="button">Apply</button></div>
      </div>`;
    document.body.append(dialog);

    const video = dialog.querySelector(".cs-timeline-video");
    const track = dialog.querySelector(".cs-timeline-track");
    const selection = dialog.querySelector(".cs-timeline-selection");
    const inHandle = dialog.querySelector(".cs-in");
    const outHandle = dialog.querySelector(".cs-out");
    const current = dialog.querySelector(".cs-current");
    const range = dialog.querySelector(".cs-range");
    const durationLabel = dialog.querySelector(".cs-duration");
    const fileLabel = dialog.querySelector(".cs-file-name");
    const originalInfo = dialog.querySelector(".cs-original-info");
    const pointerRow = dialog.querySelector(".cs-timeline-pointer-row");
    const pointer = dialog.querySelector(".cs-timeline-pointer");
    const playButton = dialog.querySelector(".cs-play");
    const keepAspectInput = dialog.querySelector(".cs-keep-aspect");
    const multipleInput = dialog.querySelector(".cs-multiple");
    const widthInput = dialog.querySelector(".cs-width");
    const heightInput = dialog.querySelector(".cs-height");
    const fpsInput = dialog.querySelector(".cs-fps");
    const currentValues = {
        start: Number(widget(node, "start_frame")?.value ?? 0),
        end: Number(widget(node, "end_frame")?.value ?? -1),
        width: Number(widget(node, "width")?.value ?? 0),
        height: Number(widget(node, "height")?.value ?? 0),
        fps: Number(widget(node, "fps")?.value ?? 0),
        keepAspect: Boolean(widget(node, "keep_aspect_ratio")?.value ?? true),
        multiple: Number(widget(node, "multiple")?.value ?? 32),
    };
    let info = null;
    let start = Math.max(0, currentValues.start);
    let end = currentValues.end;
    let dragging = null;
    let selectionPlayback = false;
    let customPlayRequest = false;

    fileLabel.textContent = filename;
    video.src = videoUrl(filename);

    function activeMultiple() {
        return Math.max(1, Math.round(Number(multipleInput.value) || 1));
    }

    function syncAspect(changedField, finalize = false) {
        if (!info || !keepAspectInput.checked || !info.width || !info.height) return;
        const multiple = activeMultiple();
        const aspect = info.width / info.height;
        if (changedField === "height") {
            let height = Number(heightInput.value);
            if (!Number.isFinite(height) || height <= 0) {
                if (!finalize) return;
                height = roundToMultiple(info.height, multiple);
                heightInput.value = height;
            }
            if (finalize) {
                height = roundToMultiple(height, multiple);
                heightInput.value = height;
            }
            widthInput.value = roundToMultiple(height * aspect, multiple);
        } else {
            let width = Number(widthInput.value);
            if (!Number.isFinite(width) || width <= 0) {
                if (!finalize) return;
                width = roundToMultiple(info.width, multiple);
                widthInput.value = width;
            }
            if (finalize) {
                width = roundToMultiple(width, multiple);
                widthInput.value = width;
            }
            heightInput.value = roundToMultiple(width / aspect, multiple);
        }
    }

    function normalizeRange(changedField = null) {
        if (!info) return;
        const maxFrame = Math.max(0, info.frames - 1);
        start = clamp(Math.round(start), 0, maxFrame);
        end = clamp(Math.round(end < 0 ? maxFrame : end), 0, maxFrame);
        if (start < end || maxFrame === 0) return;

        if (changedField === "in") {
            if (start >= maxFrame) {
                start = Math.max(0, maxFrame - 1);
                end = maxFrame;
            } else {
                end = start + 1;
            }
        } else {
            if (end <= 0) {
                start = 0;
                end = Math.min(1, maxFrame);
            } else {
                start = end - 1;
            }
        }
    }

    function currentFrame() {
        if (!info) return 0;
        return clamp(Math.round(video.currentTime * info.fps), 0, Math.max(0, info.frames - 1));
    }

    function frameAtPointerEvent(event) {
        const rect = pointerRow.getBoundingClientRect();
        const ratio = clamp((event.clientX - rect.left) / rect.width, 0, 1);
        return Math.round(ratio * Math.max(0, info.frames - 1));
    }

    function beginPointerDrag(event) {
        event.preventDefault();
        video.pause();
        const move = (moveEvent) => {
            if (!info) return;
            const frame = frameAtPointerEvent(moveEvent);
            seek(frame);
            updateTimeline(frame);
        };
        const up = () => {
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", up);
        };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
        move(event);
    }

    function setInPoint() {
        start = currentFrame();
        normalizeRange("in");
        seek(start);
        updateTimeline(start);
    }

    function setOutPoint() {
        end = currentFrame();
        normalizeRange("out");
        seek(end);
        updateTimeline(end);
    }

    widthInput.addEventListener("input", () => syncAspect("width"));
    widthInput.addEventListener("change", () => syncAspect("width", true));
    widthInput.addEventListener("blur", () => syncAspect("width", true));
    heightInput.addEventListener("input", () => syncAspect("height"));
    heightInput.addEventListener("change", () => syncAspect("height", true));
    heightInput.addEventListener("blur", () => syncAspect("height", true));
    multipleInput.addEventListener("input", () => {
        if (Number.isFinite(Number(multipleInput.value)) && Number(multipleInput.value) > 0) syncAspect("width");
    });
    multipleInput.addEventListener("change", () => syncAspect("width", true));
    multipleInput.addEventListener("blur", () => syncAspect("width", true));
    keepAspectInput.addEventListener("change", () => { if (keepAspectInput.checked) syncAspect("width", true); });

    function updateTimeline(frameOverride = null) {
        if (!info) return;
        const maxFrame = Math.max(0, info.frames - 1);
        normalizeRange();
        const frame = frameOverride === null ? currentFrame() : clamp(Math.round(frameOverride), 0, maxFrame);
        const frameRatio = maxFrame ? frame / maxFrame : 0;
        pointer.style.left = `${frameRatio * 100}%`;
        const startRatio = maxFrame ? start / maxFrame : 0;
        const endRatio = maxFrame ? end / maxFrame : 1;
        selection.style.left = `${startRatio * 100}%`;
        selection.style.width = `${Math.max(0, (endRatio - startRatio) * 100)}%`;
        inHandle.style.left = `${startRatio * 100}%`;
        outHandle.style.left = `${endRatio * 100}%`;
        range.textContent = `In ${formatTime(start / info.fps)}  -  Out ${formatTime(end / info.fps)}`;
        durationLabel.textContent = formatTime((end - start + 1) / info.fps);
        current.textContent = formatTime(frame / info.fps);
    }

    function frameAtEvent(event) {
        const rect = track.getBoundingClientRect();
        const ratio = clamp((event.clientX - rect.left) / rect.width, 0, 1);
        return Math.round(ratio * Math.max(0, info.frames - 1));
    }

    function seek(frame) {
        if (info?.fps) video.currentTime = frame / info.fps;
    }

    function stepFrame(delta) {
        if (!info) return;
        video.pause();
        const maxFrame = Math.max(0, info.frames - 1);
        const frame = clamp(currentFrame() + delta, 0, maxFrame);
        seek(frame);
        updateTimeline(frame);
    }

    function beginDrag(which, event) {
        event.preventDefault();
        dragging = which;
        const move = (moveEvent) => {
            if (!dragging || !info) return;
            const frame = frameAtEvent(moveEvent);
            if (dragging === "in") {
                start = frame;
                normalizeRange("in");
            } else {
                end = frame;
                normalizeRange("out");
            }
            seek(dragging === "in" ? start : end);
            updateTimeline(dragging === "in" ? start : end);
        };
        const up = () => { dragging = null; window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
    }

    inHandle.addEventListener("pointerdown", (event) => beginDrag("in", event));
    outHandle.addEventListener("pointerdown", (event) => beginDrag("out", event));
    track.addEventListener("pointerdown", (event) => {
        if (event.target === inHandle || event.target === outHandle) return;
        const frame = frameAtEvent(event);
        video.pause();
        seek(frame);
        if (Math.abs(frame - start) <= Math.abs(frame - end)) {
            start = frame;
            normalizeRange("in");
        } else {
            end = frame;
            normalizeRange("out");
        }
        updateTimeline(frame);
    });
    pointer.addEventListener("pointerdown", beginPointerDrag);
    pointerRow.addEventListener("pointerdown", (event) => {
        if (event.target === pointer) return;
        beginPointerDrag(event);
    });
    video.addEventListener("timeupdate", () => {
        if (info && selectionPlayback && !video.paused) {
            const selectionStart = start / info.fps;
            const selectionEnd = Math.min(video.duration || Infinity, (end + 1) / info.fps);
            if (video.currentTime < selectionStart) {
                seek(start);
            } else if (video.currentTime >= selectionEnd) {
                video.pause();
                seek(end);
                updateTimeline(end);
                return;
            }
        }
        updateTimeline();
    });
    video.addEventListener("play", () => {
        if (!customPlayRequest) selectionPlayback = false;
        customPlayRequest = false;
        playButton.textContent = "Pause";
    });
    video.addEventListener("pause", () => {
        selectionPlayback = false;
        customPlayRequest = false;
        playButton.textContent = "Play";
    });
    dialog.querySelector(".cs-set-in").addEventListener("click", setInPoint);
    dialog.querySelector(".cs-set-out").addEventListener("click", setOutPoint);
    playButton.addEventListener("click", () => {
        if (!info) return;
        if (!video.paused) {
            selectionPlayback = false;
            video.pause();
            return;
        }
        const frame = currentFrame();
        if (frame < start || frame >= end) seek(start);
        selectionPlayback = true;
        customPlayRequest = true;
        video.play().catch(() => {
            selectionPlayback = false;
            customPlayRequest = false;
        });
    });
    dialog.querySelector(".cs-back").addEventListener("click", () => stepFrame(-1));
    dialog.querySelector(".cs-forward").addEventListener("click", () => stepFrame(1));

    const close = () => { video.pause(); dialog.close(); dialog.remove(); };
    dialog.querySelector(".cs-close")?.addEventListener("click", close);
    dialog.querySelector(".cs-timeline-close").addEventListener("click", close);
    dialog.querySelector(".cs-cancel").addEventListener("click", close);
    dialog.querySelector(".cs-apply").addEventListener("click", () => {
        setWidgetValue(node, "start_frame", start);
        setWidgetValue(node, "end_frame", end);
        setWidgetValue(node, "keep_aspect_ratio", keepAspectInput.checked);
        setWidgetValue(node, "multiple", activeMultiple());
        setWidgetValue(node, "width", Math.max(1, Number(widthInput.value)));
        setWidgetValue(node, "height", Math.max(1, Number(heightInput.value)));
        setWidgetValue(node, "fps", Math.max(0.01, Number(fpsInput.value)));
        node.graph?.setDirtyCanvas(true, true);
        close();
    });
    dialog.addEventListener("cancel", close);

    fetchInfo(filename).then((result) => {
        info = result;
        originalInfo.textContent = formatOriginalInfo(result);
        originalInfo.title = originalInfo.textContent;
        end = end < 0 ? result.frames - 1 : end;
        keepAspectInput.checked = currentValues.keepAspect;
        multipleInput.value = currentValues.multiple > 0 ? currentValues.multiple : 32;
        widthInput.value = currentValues.width > 0 ? currentValues.width : roundToMultiple(result.width, activeMultiple());
        heightInput.value = currentValues.height > 0 ? currentValues.height : roundToMultiple(result.height, activeMultiple());
        fpsInput.value = currentValues.fps || result.fps;
        if (keepAspectInput.checked) syncAspect(currentValues.width > 0 ? "width" : "height");
        updateTimeline();
    }).catch((error) => {
        fileLabel.textContent = `${filename} - ${error.message}`;
    });
    dialog.showModal();
}

app.registerExtension({
    name: "CineStyle.VideoTimeline",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_ID) return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            original?.apply(this, arguments);
            const videoWidget = widget(this, "video");
            if (videoWidget && !this.__csVideoResetInstalled) {
                this.__csTimelineVideo = String(videoWidget.value ?? "");
                const originalVideoCallback = videoWidget.callback;
                videoWidget.callback = function (value) {
                    syncVideoSelection(thisNode, value ?? this.value);
                    return originalVideoCallback?.apply(this, arguments);
                };
                const thisNode = this;
                this.__csVideoResetInstalled = true;
            }
            const button = this.addWidget("button", "Edit Timeline", "", () => openTimeline(this));
            button.name = "Edit Timeline";
            button.label = "Edit Timeline";
            button.options = { ...(button.options || {}), serialize: false };
            this.setSize?.([360, Math.max(220, this.computeSize?.()[1] || 220)]);
        };
    },
});
