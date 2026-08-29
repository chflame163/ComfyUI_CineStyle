import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { installTimelineControlStyles, timelineControlsMarkup, createTimelineRangeController } from "./cinestyle_timeline_controls.js";
import { formatFrameCount } from "./cinestyle_timeline_range.js";

const NODE_ID = "CS_Load_Video";
const STYLE_ID = "cinestyle-timeline-style";
const DEFAULT_TIMELINE_VALUES = Object.freeze({
    start_frame: 0,
    end_frame: -1,
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

const VIDEO_UPLOAD_MIME_TYPES = [
    "video/webm",
    "video/mp4",
    "video/x-matroska",
    "image/gif",
];

function isSupportedVideoFile(file) {
    if (!file) return false;
    const mime = String(file.type || "").toLowerCase();
    return VIDEO_UPLOAD_MIME_TYPES.includes(mime)
        || mime.startsWith("video/")
        || /\.(mp4|webm|mov|m4v|mkv|avi|gif)$/i.test(String(file.name || ""));
}

async function uploadVideoFile(file, progressCallback) {
    try {
        const body = new FormData();
        const relativePath = String(file.webkitRelativePath || "");
        const separator = relativePath.lastIndexOf("/");
        const subfolder = separator >= 0 ? relativePath.slice(0, separator + 1) : "";
        const uploadFile = new File([file], file.name, {
            type: file.type,
            lastModified: file.lastModified,
        });
        body.append("image", uploadFile);
        if (subfolder) body.append("subfolder", subfolder);
        const response = await new Promise((resolve, reject) => {
            const request = new XMLHttpRequest();
            request.upload.onprogress = (event) => {
                if (event.lengthComputable) progressCallback?.(event.loaded / event.total);
            };
            request.onload = () => resolve(request);
            request.onerror = () => reject(new Error("Video upload request failed"));
            request.open("POST", api.apiURL("/upload/image"), true);
            request.send(body);
        });
        if (response.status !== 200) {
            if (response.status === 413) {
                const sizeMb = (Number(file.size || 0) / (1024 * 1024)).toFixed(1);
                alert(
                    `Video upload rejected (413): this file is ${sizeMb} MB, which exceeds ComfyUI's server upload limit. `
                    + `Restart ComfyUI with --max-upload-size <MB> (for example, --max-upload-size 256), then try again.`,
                );
            } else {
                alert(`${response.status} - ${response.statusText}`);
            }
        }
        return response;
    } catch (error) {
        alert(error);
        return null;
    }
}

function addVideoUpload(node, pathWidget) {
    if (!node || !pathWidget || node.__csVideoUploadInstalled) return;
    const fileInput = document.createElement("input");
    Object.assign(fileInput, {
        type: "file",
        accept: VIDEO_UPLOAD_MIME_TYPES.join(","),
        style: "display: none",
    });

    async function doUpload(file) {
        const response = await uploadVideoFile(file, (progress) => { node.progress = progress; });
        node.progress = undefined;
        if (!response || response.status !== 200) return false;
        const result = JSON.parse(response.responseText);
        const filename = result.subfolder ? `${result.subfolder}/${result.name}` : result.name;
        if (!pathWidget.options.values.includes(filename)) pathWidget.options.values.push(filename);
        pathWidget.value = filename;
        pathWidget.callback?.(filename);
        node.graph?.setDirtyCanvas(true, true);
        return true;
    }

    fileInput.onchange = async () => {
        if (fileInput.files.length) await doUpload(fileInput.files[0]);
        fileInput.value = "";
    };
    node.onDragOver = (event) => {
        const hasFiles = Boolean(event?.dataTransfer?.types?.includes?.("Files"));
        if (hasFiles) app.dragOverNode = node;
        return hasFiles;
    };
    node.onDragDrop = async (event) => {
        if (!event?.dataTransfer?.types?.includes?.("Files")) return false;
        const file = event.dataTransfer?.files?.[0];
        if (!isSupportedVideoFile(file)) return false;
        return doUpload(file);
    };

    document.body.append(fileInput);
    const uploadWidget = node.addWidget("button", "choose file to upload", "image", () => {
        app.canvas.node_widget = null;
        fileInput.click();
    });
    uploadWidget.options = { ...(uploadWidget.options || {}), serialize: false };

    const originalRemoved = node.onRemoved;
    node.onRemoved = function () {
        fileInput.remove();
        return originalRemoved?.apply(this, arguments);
    };
    node.__csVideoUploadInstalled = true;
}

function addVideoPreview(node, pathWidget) {
    if (!node || !pathWidget || node.__csVideoPreviewInstalled) return;
    const element = document.createElement("div");
    const previewWidget = node.addDOMWidget("videopreview", "preview", element, {
        serialize: false,
        hideOnZoom: false,
        getValue: () => element.value,
        setValue: (value) => { element.value = value; },
    });
    const rangeIndicator = document.createElement("div");
    const rangeSelection = document.createElement("div");
    Object.assign(rangeIndicator.style, {
        position: "relative",
        width: "100%",
        height: "4px",
        margin: "0",
        padding: "0",
        background: "#626a73",
        overflow: "hidden",
        pointerEvents: "none",
    });
    Object.assign(rangeSelection.style, {
        position: "absolute",
        top: "0",
        bottom: "0",
        left: "0",
        width: "0%",
        background: "var(--cs-timeline-accent, #55a9f5)",
    });
    rangeIndicator.append(rangeSelection);
    rangeIndicator.hidden = true;
    element.append(rangeIndicator);
    const video = document.createElement("video");
    video.controls = false;
    video.loop = true;
    video.muted = true;
    video.playsInline = true;
    video.style.width = "100%";
    video.style.display = "block";
    video.style.background = "#08090b";
    element.append(video);
    element.style.width = "100%";
    element.addEventListener("dragover", (event) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = "copy";
        app.dragOverNode = node;
    });
    element.addEventListener("drop", async (event) => {
        event.preventDefault();
        event.stopPropagation();
        const file = event.dataTransfer?.files?.[0];
        if (isSupportedVideoFile(file)) await node.onDragDrop?.(event);
        app.dragOverNode = null;
    });
    element.value = { hidden: true, filename: "" };
    let rangeInfo = null;
    let rangeInfoRequest = 0;
    const updateRangeIndicator = () => {
        const frameCount = Math.max(0, Math.round(Number(rangeInfo?.frames) || 0));
        if (!frameCount) {
            rangeIndicator.hidden = true;
            return;
        }
        const lastFrame = frameCount - 1;
        const startValue = Number(widget(node, "start_frame")?.value ?? 0);
        const endValue = Number(widget(node, "end_frame")?.value ?? -1);
        const start = clamp(Number.isFinite(startValue) ? startValue : 0, 0, lastFrame);
        const end = clamp(Number.isFinite(endValue) && endValue >= 0 ? endValue : lastFrame, start, lastFrame);
        rangeIndicator.hidden = false;
        rangeSelection.style.left = `${(start / frameCount) * 100}%`;
        rangeSelection.style.width = `${((end - start + 1) / frameCount) * 100}%`;
    };
    const updateRangeInfo = (filename) => {
        const value = String(filename || "");
        const requestId = ++rangeInfoRequest;
        rangeInfo = null;
        updateRangeIndicator();
        if (!value) return;
        fetchInfo(value).then((result) => {
            if (requestId !== rangeInfoRequest || String(pathWidget.value || "") !== value) return;
            rangeInfo = result;
            updateRangeIndicator();
        }).catch(() => {
            if (requestId === rangeInfoRequest) updateRangeIndicator();
        });
    };
    for (const name of ["start_frame", "end_frame"]) {
        const rangeWidget = widget(node, name);
        if (!rangeWidget || rangeWidget.__csRangeIndicatorBound) continue;
        const originalCallback = rangeWidget.callback;
        rangeWidget.callback = function (value) {
            const result = originalCallback?.apply(this, arguments);
            updateRangeIndicator();
            return result;
        };
        rangeWidget.__csRangeIndicatorBound = true;
    }
    previewWidget.computeSize = (width) => {
        if (video.videoWidth > 0 && video.videoHeight > 0 && !element.value.hidden) {
            const aspectRatio = video.videoWidth / video.videoHeight;
            // Match VHS's DOM preview sizing and reserve the widget's
            // vertical padding so the final rows of the video stay visible.
            const videoHeight = Math.max(0, (Math.max(160, node.size?.[0] || width) - 20) / aspectRatio + 10);
            const height = 4 + videoHeight;
            return [width, height];
        }
        return [width, rangeIndicator.hidden ? -4 : 4];
    };
    const updateSource = (filename) => {
        const value = String(filename || "");
        element.value = { hidden: !value, filename: value };
        updateRangeInfo(value);
        if (!value) {
            video.removeAttribute("src");
            video.load();
            node.graph?.setDirtyCanvas(true, true);
            return;
        }
        const params = new URLSearchParams({ filename: value, type: "input", subfolder: "", t: String(Date.now()) });
        video.src = api.apiURL(`/view?${params.toString()}`);
        video.load();
        video.play().catch(() => {});
        node.graph?.setDirtyCanvas(true, true);
    };
    const fitPreviewHeight = () => {
        const width = Math.max(160, node.size?.[0] || 360);
        // Compute the complete node height so the preview does not clip the
        // widgets above it (LiteGraph's computeSize includes all widgets).
        const computed = node.computeSize?.([width, node.size?.[1] || 0]);
        const totalHeight = Number(computed?.[1]);
        if (Number.isFinite(totalHeight) && totalHeight > 0) {
            node.setSize?.([width, totalHeight]);
        }
        node.graph?.setDirtyCanvas(true, true);
    };
    video.addEventListener("loadedmetadata", () => {
        previewWidget.aspectRatio = video.videoWidth / video.videoHeight;
        fitPreviewHeight();
        updateRangeIndicator();
    });
    video.addEventListener("error", () => {
        element.value.hidden = true;
        fitPreviewHeight();
    });
    node.__csVideoPreviewUpdate = updateSource;
    node.__csVideoRangeIndicatorUpdate = updateRangeIndicator;
    node.__csVideoPreviewInstalled = true;
    requestAnimationFrame(() => updateSource(pathWidget.value));
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

function syncNodeDimensions(node, changed = null, forceSource = false) {
    const sourceInfo = node?.__csSourceVideoInfo;
    const widthWidget = widget(node, "width");
    const heightWidget = widget(node, "height");
    const multipleWidget = widget(node, "multiple");
    if (!sourceInfo || !widthWidget || !heightWidget) return;
    const sourceWidth = Math.max(1, Math.round(Number(sourceInfo.width) || 1));
    const sourceHeight = Math.max(1, Math.round(Number(sourceInfo.height) || 1));
    const multiple = Math.max(1, Math.round(Number(multipleWidget?.value) || 1));
    let width = Number(widthWidget.value);
    let height = Number(heightWidget.value);
    if (forceSource || (!Number.isFinite(width) || width <= 0) && (!Number.isFinite(height) || height <= 0)) {
        // A newly selected source starts from its own dimensions. Rounding
        // each axis independently preserves the documented default behavior.
        width = roundToMultiple(sourceWidth, multiple);
        height = roundToMultiple(sourceHeight, multiple);
    } else if (changed === "height") {
        height = roundToMultiple(height, multiple);
        width = roundToMultiple(height * sourceWidth / sourceHeight, multiple);
    } else if (changed === "width") {
        width = roundToMultiple(width, multiple);
        height = roundToMultiple(width * sourceHeight / sourceWidth, multiple);
    } else {
        // Workflow values remain intact on load, but are always normalized to
        // the active multiple. A later edit establishes aspect-ratio linkage.
        width = roundToMultiple(Number.isFinite(width) && width > 0 ? width : sourceWidth, multiple);
        height = roundToMultiple(Number.isFinite(height) && height > 0 ? height : sourceHeight, multiple);
    }
    node.__csDimensionSyncing = true;
    widthWidget.value = width;
    heightWidget.value = height;
    node.__csDimensionSyncing = false;
    node.graph?.setDirtyCanvas(true, true);
}

function refreshNodeSourceInfo(node, filename, forceSource = false) {
    const value = String(filename || "");
    const requestId = (node.__csSourceInfoRequest || 0) + 1;
    node.__csSourceInfoRequest = requestId;
    node.__csSourceVideoInfo = null;
    if (!value) return;
    fetchInfo(value).then((result) => {
        if (node.__csSourceInfoRequest !== requestId || String(widget(node, "video")?.value || "") !== value) return;
        node.__csSourceVideoInfo = result;
        syncNodeDimensions(node, null, forceSource);
    }).catch(() => {});
}

function effectiveOutputDimensions(values, sourceInfo) {
    const sourceWidth = Math.max(1, Math.round(Number(sourceInfo?.width) || 1));
    const sourceHeight = Math.max(1, Math.round(Number(sourceInfo?.height) || 1));
    const aspect = sourceWidth / sourceHeight;
    const multiple = Math.max(1, Math.round(Number(values?.multiple) || 1));
    let width = Number(values?.width) || 0;
    let height = Number(values?.height) || 0;
    if (width > 0) {
        width = roundToMultiple(width, multiple);
        height = roundToMultiple(width / aspect, multiple);
    } else if (height > 0) {
        height = roundToMultiple(height, multiple);
        width = roundToMultiple(height * aspect, multiple);
    } else {
        // The Timeline initializes the empty-width case from the source
        // height, so use the same branch when comparing against defaults.
        height = roundToMultiple(sourceHeight, multiple);
        width = roundToMultiple(height * aspect, multiple);
    }
    return { width, height };
}

function effectiveTimelineValues(values, sourceInfo) {
    const lastFrame = Math.max(0, Math.round(Number(sourceInfo?.frames) || 1) - 1);
    const start = clamp(Math.round(Number(values?.start) || 0), 0, lastFrame);
    const rawEnd = Number(values?.end);
    const end = !Number.isFinite(rawEnd) || rawEnd < 0
        ? lastFrame
        : clamp(Math.round(rawEnd), start, lastFrame);
    const sourceFps = Math.max(0.001, Number(sourceInfo?.fps) || 24);
    const fpsValue = Number(values?.fps);
    const fps = Number.isFinite(fpsValue) && fpsValue > 0 ? fpsValue : sourceFps;
    return {
        start,
        end,
        fps,
        dimensions: effectiveOutputDimensions(values, sourceInfo),
        multiple: Math.max(1, Math.round(Number(values?.multiple) || 1)),
    };
}

function timelineParametersChanged(previous, next, sourceInfo) {
    if (!sourceInfo) return true;
    const before = effectiveTimelineValues(previous, sourceInfo);
    const after = effectiveTimelineValues(next, sourceInfo);
    return before.start !== after.start
        || before.end !== after.end
        || before.multiple !== after.multiple
        || before.dimensions.width !== after.dimensions.width
        || before.dimensions.height !== after.dimensions.height
        || Math.abs(before.fps - after.fps) > 1e-6;
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
      .cs-video-stage { position: relative; width: 100%; aspect-ratio: 16 / 9; background: #08090b; border-radius: 6px; overflow: hidden; }
      .cs-timeline-video { width: 100%; height: 100%; display: block; background: #08090b; object-fit: contain; }
      .cs-file-name { min-width: 0; max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .cs-original-info { max-width: 100%; color: #aeb5c2; font-size: 12px; line-height: 1.45; overflow-wrap: anywhere; word-break: break-word; }
      .cs-timeline-fields { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
      .cs-timeline-field { display: grid; gap: 5px; color: #9da5b4; }
      .cs-timeline-field input { width: 100%; box-sizing: border-box; border: 1px solid #424956; border-radius: 5px; padding: 7px 8px; background: #20232a; color: #f2f4f7; }
      .cs-timeline-foot { justify-content: flex-end; }
      .cs-timeline-foot .cs-apply { background: #317ec4; border-color: #4b9de8; }
      @media (max-width: 640px) { .cs-timeline-fields { grid-template-columns: 1fr; } .cs-timeline-shell { padding: 12px; } }
    `;
    document.head.append(style);
    installTimelineControlStyles();
}

function videoUrl(filename) {
    const params = new URLSearchParams({ filename, type: "input", subfolder: "", t: String(Date.now()) });
    return api.apiURL(`/view?${params.toString()}`);
}

function requestLoaderPreviewCache(node, filename, values) {
    const payload = {
        loader_id: String(node?.id ?? ""),
        video: String(filename || ""),
        start_frame: Number(values?.start ?? 0),
        end_frame: Number(values?.end ?? -1),
        width: Number(values?.width ?? 0),
        height: Number(values?.height ?? 0),
        fps: Number(values?.fps ?? 0),
        multiple: Number(values?.multiple ?? 32),
        start_build: values?.startBuild !== false,
    };
    return api.fetchApi("/cinestyle/loader-preview-cache", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    }).then(async (response) => {
        const result = await response.json().catch(() => ({}));
        if (!response.ok || result.status === "failed") throw new Error(result.error || "Unable to prepare loader preview cache");
        return result;
    });
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
        <div class="cs-video-stage"><video class="cs-timeline-video" controls playsinline preload="auto"></video></div>
        <div class="cs-timeline-readout"><span class="cs-current">00:00.00</span><span class="cs-range"></span><span class="cs-duration">00:00.00</span></div>
        ${timelineControlsMarkup()}
        <div class="cs-timeline-fields"><label class="cs-timeline-field">multiple<input class="cs-multiple" type="number" min="1" step="1"></label><label class="cs-timeline-field">Width<input class="cs-width" type="number" min="1" step="1"></label><label class="cs-timeline-field">Height<input class="cs-height" type="number" min="1" step="1"></label><label class="cs-timeline-field">FPS<input class="cs-fps" type="number" min="0.01" max="240" step="0.01"></label></div>
        <div class="cs-timeline-foot"><button class="cs-cancel" type="button">Cancel</button><button class="cs-apply" type="button">Apply</button></div>
      </div>`;
    document.body.append(dialog);

    const video = dialog.querySelector(".cs-timeline-video");
    const current = dialog.querySelector(".cs-current");
    const range = dialog.querySelector(".cs-range");
    const durationLabel = dialog.querySelector(".cs-duration");
    const fileLabel = dialog.querySelector(".cs-file-name");
    const originalInfo = dialog.querySelector(".cs-original-info");
    const playButton = dialog.querySelector(".cs-play");
    const axis = dialog.querySelector(".cs-timeline-axis");
    const track = dialog.querySelector(".cs-timeline-track");
    const rangeBand = dialog.querySelector(".cs-timeline-range-band");
    const inHandle = dialog.querySelector(".cs-range-marker.in");
    const outHandle = dialog.querySelector(".cs-range-marker.out");
    const pointer = dialog.querySelector(".cs-timeline-pointer");
    const inFrameButton = dialog.querySelector(".cs-in-frame");
    const outFrameButton = dialog.querySelector(".cs-out-frame");
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
        multiple: Number(widget(node, "multiple")?.value ?? 32),
    };
    let info = null;
    let start = Math.max(0, currentValues.start);
    let end = currentValues.end;
    let selectionPlayback = false;
    let customPlayRequest = false;
    let timelineControls = null;

    function renderTimeline(frameOverride = null) {
        if (!info) return;
        axis.innerHTML = "";
        const tickCount = Math.max(2, Math.min(12, Math.round(track.clientWidth / 100)));
        const totalSeconds = Math.max(0, Number(info.duration) || (info.frames / Math.max(0.01, info.fps)));
        for (let i = 0; i <= tickCount; i++) {
            const tick = document.createElement("span");
            tick.style.left = `${(i / tickCount) * 100}%`;
            tick.textContent = formatTime(totalSeconds * i / tickCount);
            axis.append(tick);
        }
        const maxFrame = Math.max(0, info.frames - 1);
        const frame = frameOverride === null
            ? clamp(Math.round(video.currentTime * info.fps), 0, maxFrame)
            : clamp(Math.round(frameOverride), 0, maxFrame);
        const startRatio = maxFrame ? start / maxFrame : 0;
        const endRatio = maxFrame ? end / maxFrame : 1;
        const frameRatio = maxFrame ? frame / maxFrame : 0;
        rangeBand.style.display = endRatio > startRatio ? "block" : "none";
        rangeBand.style.left = `${startRatio * 100}%`;
        rangeBand.style.width = `${Math.max(0, (endRatio - startRatio) * 100)}%`;
        pointer.style.left = `${frameRatio * 100}%`;
        inFrameButton.textContent = String(start);
        outFrameButton.textContent = String(end);
        current.textContent = formatTime(frame / info.fps);
        const selectedFrames = end - start + 1;
        range.textContent = `In ${formatTime(start / info.fps)}  -  Out ${formatTime(end / info.fps)}  ·  Duration ${formatFrameCount(selectedFrames)}`;
        durationLabel.textContent = formatFrameCount(selectedFrames);
    }

    fileLabel.textContent = filename;
    video.muted = false;
    video.volume = 1;
    timelineControls = createTimelineRangeController({
        root: dialog,
        video,
        getInfo: () => info,
        getRange: () => ({ start, end }),
        setRange: (range) => { start = range.start; end = range.end; },
        render: renderTimeline,
    });

    function activeMultiple() {
        return Math.max(1, Math.round(Number(multipleInput.value) || 1));
    }

    function syncAspect(changedField, finalize = false) {
        if (!info || !info.width || !info.height) return;
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

    function currentFrame() {
        if (!info) return 0;
        return clamp(Math.round(video.currentTime * info.fps), 0, Math.max(0, info.frames - 1));
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
    function seek(frame) {
        if (info?.fps) video.currentTime = frame / info.fps;
    }

    video.addEventListener("timeupdate", () => {
        if (info && selectionPlayback && !video.paused) {
            const selectionStart = start / info.fps;
            const selectionEnd = Math.min(video.duration || Infinity, (end + 1) / info.fps);
            if (video.currentTime < selectionStart) {
                seek(start);
            } else if (video.currentTime >= selectionEnd) {
                video.pause();
                seek(end);
                timelineControls?.render(end);
                return;
            }
        }
        timelineControls?.render();
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
    playButton.addEventListener("click", () => {
        if (!info) return;
        video.muted = false;
        video.volume = 1;
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

    const close = () => { video.pause(); dialog.close(); dialog.remove(); };
    dialog.querySelector(".cs-close")?.addEventListener("click", close);
    dialog.querySelector(".cs-timeline-close").addEventListener("click", close);
    dialog.querySelector(".cs-cancel").addEventListener("click", close);
    dialog.querySelector(".cs-apply").addEventListener("click", () => {
        const nextValues = {
            start: Math.max(0, Math.round(Number(start) || 0)),
            end: Number.isFinite(Number(end)) ? Math.round(Number(end)) : -1,
            multiple: activeMultiple(),
            width: Math.max(1, Math.round(Number(widthInput.value) || 1)),
            height: Math.max(1, Math.round(Number(heightInput.value) || 1)),
            fps: Math.max(0.01, Number(fpsInput.value) || 0.01),
        };
        const cacheInputChanged = timelineParametersChanged(currentValues, nextValues, info);
        // Preserve default sentinels on a true no-op. This avoids turning
        // end=-1/width=0/fps=0 into explicit values and dirtying the node.
        const valuesToApply = cacheInputChanged ? nextValues : currentValues;
        if (cacheInputChanged) {
            setWidgetValue(node, "start_frame", valuesToApply.start);
            setWidgetValue(node, "end_frame", valuesToApply.end);
            setWidgetValue(node, "multiple", valuesToApply.multiple);
            setWidgetValue(node, "width", valuesToApply.width);
            setWidgetValue(node, "height", valuesToApply.height);
            setWidgetValue(node, "fps", valuesToApply.fps);
            node.graph?.setDirtyCanvas(true, true);
        }
        if (cacheInputChanged) {
            void requestLoaderPreviewCache(node, filename, nextValues).catch(() => {});
        }
        close();
    });
    dialog.addEventListener("cancel", close);

    fetchInfo(filename).then((result) => {
        // This editor must always use the complete source timeline. A shared
        // loader cache represents the currently selected trim and would make
        // frames outside that trim impossible to select on a later edit.
        info = result;
        originalInfo.textContent = formatOriginalInfo(result);
        video.src = videoUrl(filename);
        originalInfo.title = originalInfo.textContent;
        video.load();
        const sourceLastFrame = Math.max(0, Number(info.frames || 1) - 1);
        start = clamp(start, 0, sourceLastFrame);
        end = end < 0 ? sourceLastFrame : clamp(end, start, sourceLastFrame);
        multipleInput.value = currentValues.multiple > 0 ? currentValues.multiple : 32;
        widthInput.value = currentValues.width > 0 ? currentValues.width : roundToMultiple(info.width, activeMultiple());
        heightInput.value = currentValues.height > 0 ? currentValues.height : roundToMultiple(info.height, activeMultiple());
        fpsInput.value = currentValues.fps || info.fps;
        if (currentValues.width > 0 || currentValues.height > 0) {
            syncAspect(currentValues.width > 0 ? "width" : "height", true);
        }
        timelineControls?.render();
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
            for (const name of ["width", "height", "multiple"]) {
                const dimensionWidget = widget(this, name);
                if (!dimensionWidget || dimensionWidget.__csDimensionSyncBound) continue;
                const originalDimensionCallback = dimensionWidget.callback;
                dimensionWidget.callback = function (value) {
                    const result = originalDimensionCallback?.apply(this, arguments);
                    if (!thisNode.__csDimensionSyncing) {
                        syncNodeDimensions(thisNode, name === "multiple" ? null : name, false);
                    }
                    return result;
                };
                dimensionWidget.__csDimensionSyncBound = true;
            }
            if (videoWidget && !this.__csVideoResetInstalled) {
                this.__csTimelineVideo = String(videoWidget.value ?? "");
                const originalVideoCallback = videoWidget.callback;
                videoWidget.callback = function (value) {
                    const filename = value ?? this.value;
                    syncVideoSelection(thisNode, filename);
                    thisNode.__csVideoPreviewUpdate?.(filename);
                    refreshNodeSourceInfo(thisNode, filename, true);
                    return originalVideoCallback?.apply(this, arguments);
                };
                const thisNode = this;
                this.__csVideoResetInstalled = true;
            }
            const thisNode = this;
            refreshNodeSourceInfo(thisNode, videoWidget?.value, false);
            addVideoUpload(this, videoWidget);
            const button = this.addWidget("button", "Edit Timeline", "", () => openTimeline(this));
            button.name = "Edit Timeline";
            button.label = "Edit Timeline";
            button.options = { ...(button.options || {}), serialize: false };
            addVideoPreview(this, videoWidget);
            this.setSize?.([360, Math.max(220, this.computeSize?.()[1] || 220)]);
        };
        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            originalConfigure?.apply(this, arguments);
            const configuredVideo = widget(this, "video")?.value;
            this.__csVideoPreviewUpdate?.(configuredVideo);
            this.__csVideoRangeIndicatorUpdate?.();
            refreshNodeSourceInfo(this, configuredVideo, false);
        };
    },
});
