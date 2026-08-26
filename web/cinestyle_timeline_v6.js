import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { installTimelineControlStyles, timelineControlsMarkup, createTimelineRangeController } from "./cinestyle_timeline_controls.js";
import { formatFrameCount } from "./cinestyle_timeline_range.js";

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

function sanitizeProxyWidgets(node) {
    const threshold = widget(node, "proxy_threshold");
    const size = widget(node, "proxy_size");
    const normalize = (target, fallback) => {
        if (!target) return;
        const value = Number(target.value);
        if (!Number.isFinite(value) || value <= 0) {
            target.value = fallback;
            target.callback?.(fallback);
        }
    };
    normalize(threshold, 2.1);
    normalize(size, 0.8);
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

function proxyParameter(value, fallback) {
    const number = Number(value);
    return Number.isFinite(number) && number > 0 ? clamp(number, 0.1, 1000) : fallback;
}

function roundToMultiple(value, multiple) {
    const safeMultiple = Math.max(1, Math.round(Number(multiple) || 1));
    return Math.max(safeMultiple, Math.floor(Number(value) / safeMultiple + 0.5) * safeMultiple);
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
      .cs-proxy-wait { position: absolute; inset: 0; display: none; align-items: center; justify-content: center; padding: 16px; color: #e6e9ef; background: #08090be6; font-size: 15px; text-align: center; pointer-events: none; }
      .cs-file-name { min-width: 0; max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .cs-original-info { max-width: 100%; color: #aeb5c2; font-size: 12px; line-height: 1.45; overflow-wrap: anywhere; word-break: break-word; }
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
    installTimelineControlStyles();
}

function videoUrl(filename) {
    const params = new URLSearchParams({ filename, type: "input", subfolder: "", t: String(Date.now()) });
    return api.apiURL(`/view?${params.toString()}`);
}

function proxyVideoUrl(filename, proxyThreshold, proxySize) {
    const params = new URLSearchParams({ filename, proxy_threshold: String(proxyThreshold), proxy_size: String(proxySize), t: String(Date.now()) });
    return api.apiURL(`/cinestyle/video-proxy?${params.toString()}`);
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

function proxyProgressUrl(filename, proxyThreshold, proxySize) {
    const params = new URLSearchParams({ filename, proxy_threshold: String(proxyThreshold), proxy_size: String(proxySize) });
    return api.apiURL(`/cinestyle/video-proxy-progress?${params.toString()}`);
}

async function fetchInfo(filename, proxyThreshold, proxySize) {
    const params = new URLSearchParams({ filename, proxy_threshold: String(proxyThreshold), proxy_size: String(proxySize) });
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
        <div class="cs-video-stage"><video class="cs-timeline-video" controls playsinline preload="auto"></video><div class="cs-proxy-wait">Wait for Generate Proxy</div></div>
        <div class="cs-timeline-readout"><span class="cs-current">00:00.00</span><span class="cs-range"></span><span class="cs-duration">00:00.00</span></div>
        ${timelineControlsMarkup()}
        <div class="cs-timeline-fields"><label class="cs-timeline-field cs-timeline-check"><span><input class="cs-keep-aspect" type="checkbox"> keep aspect ratio</span></label><label class="cs-timeline-field">multiple<input class="cs-multiple" type="number" min="1" step="1"></label><label class="cs-timeline-field">Width<input class="cs-width" type="number" min="1" step="1"></label><label class="cs-timeline-field">Height<input class="cs-height" type="number" min="1" step="1"></label><label class="cs-timeline-field">FPS<input class="cs-fps" type="number" min="0.01" max="240" step="0.01"></label></div>
        <div class="cs-timeline-foot"><button class="cs-cancel" type="button">Cancel</button><button class="cs-apply" type="button">Apply</button></div>
      </div>`;
    document.body.append(dialog);

    const video = dialog.querySelector(".cs-timeline-video");
    const proxyWait = dialog.querySelector(".cs-proxy-wait");
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
    const keepAspectInput = dialog.querySelector(".cs-keep-aspect");
    const multipleInput = dialog.querySelector(".cs-multiple");
    const widthInput = dialog.querySelector(".cs-width");
    const heightInput = dialog.querySelector(".cs-height");
    const fpsInput = dialog.querySelector(".cs-fps");
    const proxyThreshold = proxyParameter(widget(node, "proxy_threshold")?.value, 2.1);
    const proxySize = proxyParameter(widget(node, "proxy_size")?.value, 0.8);
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
    keepAspectInput.checked = true;
    keepAspectInput.disabled = true;
    let proxyProgressTimer = null;
    const setProxyWait = (visible, progress = 0) => {
        proxyWait.style.display = visible ? "flex" : "none";
        if (visible) proxyWait.textContent = `Wait for Generate Proxy ${Math.max(0, Math.min(100, Math.round(progress)))}%`;
    };
    const stopProxyProgress = () => {
        if (proxyProgressTimer !== null) window.clearTimeout(proxyProgressTimer);
        proxyProgressTimer = null;
    };
    video.addEventListener("loadeddata", () => { stopProxyProgress(); setProxyWait(false); });
    video.addEventListener("canplay", () => { stopProxyProgress(); setProxyWait(false); });
    video.addEventListener("error", () => { stopProxyProgress(); setProxyWait(false); });
    const watchProxyProgress = async () => {
        try {
            const response = await api.fetchApi(proxyProgressUrl(filename, proxyThreshold, proxySize));
            const state = await response.json();
            if (state.error) {
                setProxyWait(true, 0);
                proxyWait.textContent = `Proxy generation failed: ${state.error}`;
                stopProxyProgress();
                return;
            }
            setProxyWait(true, state.progress);
            if (state.done) {
                setProxyWait(true, 100);
                stopProxyProgress();
                return;
            }
        } catch (error) {
            proxyWait.textContent = `Proxy progress unavailable: ${error.message}`;
            stopProxyProgress();
            return;
        }
        proxyProgressTimer = window.setTimeout(watchProxyProgress, 250);
    };

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
    keepAspectInput.addEventListener("change", () => { if (keepAspectInput.checked) syncAspect("width", true); });

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

    const close = () => { stopProxyProgress(); video.pause(); dialog.close(); dialog.remove(); };
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
        const aspectSettingChanged = currentValues.keepAspect !== true;
        // Preserve default sentinels on a true no-op. This avoids turning
        // end=-1/width=0/fps=0 into explicit values and dirtying the node.
        const valuesToApply = cacheInputChanged ? nextValues : currentValues;
        if (cacheInputChanged || aspectSettingChanged) {
            setWidgetValue(node, "start_frame", valuesToApply.start);
            setWidgetValue(node, "end_frame", valuesToApply.end);
            setWidgetValue(node, "keep_aspect_ratio", true);
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

    fetchInfo(filename, proxyThreshold, proxySize).then((result) => {
        // This editor must always use the complete source timeline. A shared
        // loader cache represents the currently selected trim and would make
        // frames outside that trim impossible to select on a later edit.
        info = result;
        originalInfo.textContent = formatOriginalInfo(result);
        if (result.proxy_required) {
            originalInfo.textContent += ` · 正在生成 ${result.proxy_width}×${result.proxy_height} 预览代理`;
            setProxyWait(true);
            video.src = proxyVideoUrl(filename, proxyThreshold, proxySize);
            watchProxyProgress();
        } else {
            setProxyWait(false);
            video.src = videoUrl(filename);
        }
        originalInfo.title = originalInfo.textContent;
        video.load();
        const sourceLastFrame = Math.max(0, Number(info.frames || 1) - 1);
        start = clamp(start, 0, sourceLastFrame);
        end = end < 0 ? sourceLastFrame : clamp(end, start, sourceLastFrame);
        keepAspectInput.checked = true;
        multipleInput.value = currentValues.multiple > 0 ? currentValues.multiple : 32;
        widthInput.value = currentValues.width > 0 ? currentValues.width : roundToMultiple(info.width, activeMultiple());
        heightInput.value = currentValues.height > 0 ? currentValues.height : roundToMultiple(info.height, activeMultiple());
        fpsInput.value = currentValues.fps || info.fps;
        if (keepAspectInput.checked) syncAspect(currentValues.width > 0 ? "width" : "height");
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
            sanitizeProxyWidgets(this);
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
        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            originalConfigure?.apply(this, arguments);
            sanitizeProxyWidgets(this);
        };
    },
});
