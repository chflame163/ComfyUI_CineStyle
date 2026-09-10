import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import {
    connectedInputChain,
    connectedVideoSource,
    ensureLoaderPreviewSource,
    fetchInfo,
    fetchWaitInputCache,
    prepareInputTimeline,
} from "./video_selector_multi.js";

// The Python implementation accepts the same canonical, frame-based shape.
// Keep this file self-contained: ComfyUI loads every JS file in WEB_DIRECTORY,
// and this editor must also work when the optional subtitle editor is absent.
const NODE_ID = "CS_Video_Timeline_Edit";
const STYLE_ID = "cinestyle-video-time-edit-style";
const SCHEMA_VERSION = 1;
const TRANSFORM_VERSION = 1;
const SOURCE_FINGERPRINT_VERSION = 2;
const DEFAULT_FPS = 24;
const DEFAULT_MULTIPLE = 32;
const DEFAULT_FILL = "#000000";
const PREVIEW_DEBOUNCE_MS = 75;
const STATE_DEBOUNCE_MS = 180;
// Keep zoom controls precise: the previous 1.5× factor used a 50% jump.
// A 1.125× factor is one quarter of that increment, for both buttons and
// mouse-wheel zooming.
const TIMELINE_ZOOM_FACTOR = 1.125;
const ACTIVE_DRAG = new WeakMap();
const STATE_REVISIONS = new WeakMap();

function widget(node, name) {
    return node?.widgets?.find((item) => item?.name === name) || null;
}

function setWidgetValue(node, name, value) {
    const target = widget(node, name);
    if (!target) return;
    target.value = value;
    try { target.callback?.(value); } catch (_) { /* widget callbacks are optional */ }
}

function number(value, fallback = 0) {
    const result = Number(value);
    return Number.isFinite(result) ? result : fallback;
}

function int(value, fallback = 0) {
    return Math.round(number(value, fallback));
}

function clamp(value, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, value));
}

function parseJson(value, fallback = {}) {
    if (value && typeof value === "object") return value;
    try {
        const parsed = JSON.parse(String(value || ""));
        return parsed && typeof parsed === "object" ? parsed : fallback;
    } catch (_) {
        return fallback;
    }
}

function canonicalJson(value) {
    // JSON.stringify preserves insertion order, which is useful while editing;
    // the backend performs its own sorted canonicalisation before rendering.
    return JSON.stringify(value);
}

function hex(value, fallback = DEFAULT_FILL) {
    let text = String(value ?? "").trim().toUpperCase();
    if (/^#[0-9A-F]{3}$/.test(text)) text = `#${text.slice(1).split("").map((c) => c + c).join("")}`;
    return /^#[0-9A-F]{6}$/.test(text) ? text : fallback;
}

function ceilMultiple(value, multiple) {
    const amount = Math.max(1, int(value, 1));
    const step = Math.max(1, int(multiple, 1));
    return Math.ceil(amount / step) * step;
}

function dimensionRequest(value, multiple = DEFAULT_MULTIPLE) {
    const parsed = int(value, -1);
    if (parsed <= 0) return -1;
    return Math.max(Math.max(1, int(multiple, 1)), ceilMultiple(parsed, multiple));
}

function bool(value, fallback = false) {
    if (value == null) return fallback;
    if (typeof value === "string") {
        const text = value.trim().toLowerCase();
        if (["true", "yes", "on", "1"].includes(text)) return true;
        if (["false", "no", "off", "0", ""].includes(text)) return false;
    }
    return Boolean(value);
}

function idFor(index = 0) {
    return `clip-${Date.now().toString(36)}-${index.toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
}

function normalizeTransform(value = {}) {
    const source = value && typeof value === "object" ? value : {};
    const translation = source.translation ?? source.translate ?? source.position;
    let tx = number(source.translate_x ?? source.translation_x ?? source.x ?? source.offset_x, 0);
    let ty = number(source.translate_y ?? source.translation_y ?? source.y ?? source.offset_y, 0);
    if (translation && typeof translation === "object" && !Array.isArray(translation)) {
        tx = number(translation.x ?? translation.translate_x, tx);
        ty = number(translation.y ?? translation.translate_y, ty);
    } else if (Array.isArray(translation)) {
        tx = number(translation[0], tx);
        ty = number(translation[1], ty);
    }
    let unit = String(source.translation_unit ?? source.position_unit ?? "auto").toLowerCase();
    if (!["normalized", "normalised", "pixel", "pixels"].includes(unit)) unit = (Math.abs(tx) <= 1 && Math.abs(ty) <= 1) ? "normalized" : "pixel";
    if (unit === "normalised") unit = "normalized";
    if (unit === "pixels") unit = "pixel";
    let sx = source.scale_x ?? source.scaleX ?? source.scale ?? source.zoom ?? 1;
    let sy = source.scale_y ?? source.scaleY ?? source.scale ?? source.zoom ?? sx;
    if (Array.isArray(sx)) { sy = sx[1] ?? sx[0]; sx = sx[0]; }
    if (sx && typeof sx === "object") { sy = sx.y ?? sx.scale_y ?? sx.x ?? 1; sx = sx.x ?? sx.scale_x ?? 1; }
    sx = number(sx, 1); sy = number(sy, sx);
    // Percentages are accepted for interoperability with older prototypes.
    if (Math.abs(sx) > 20) sx /= 100;
    if (Math.abs(sy) > 20) sy /= 100;
    sx = Math.abs(sx) < 0.001 ? 1 : clamp(Math.abs(sx), 0.1, 4);
    sy = Math.abs(sy) < 0.001 ? 1 : clamp(Math.abs(sy), 0.1, 4);
    const mirror = source.mirror;
    let flipX = bool(source.flip_x ?? source.flipX ?? source.mirror_x ?? source.mirrorX ?? source.mirror_horizontal ?? source.mirrorHorizontal ?? source.horizontal_flip, false);
    let flipY = bool(source.flip_y ?? source.flipY ?? source.mirror_y ?? source.mirrorY ?? source.mirror_vertical ?? source.mirrorVertical ?? source.vertical_flip, false);
    if (typeof mirror === "string") {
        const text = mirror.toLowerCase().trim().replace(/_/g, "-");
        flipX ||= ["x", "h", "horizontal", "left-right", "leftright", "both", "xy"].includes(text);
        flipY ||= ["y", "v", "vertical", "up-down", "updown", "both", "xy"].includes(text);
    } else if (Array.isArray(mirror)) {
        flipX ||= bool(mirror[0]); flipY ||= bool(mirror[1]);
    }
    return {
        version: TRANSFORM_VERSION,
        scale_x: sx,
        scale_y: sy,
        rotation: clamp(number(source.rotation ?? source.rotate ?? source.angle ?? source.rotation_degrees, 0), -90, 90),
        translate_x: tx,
        translate_y: ty,
        translation_unit: unit,
        flip_x: flipX,
        flip_y: flipY,
    };
}

function normalizeClip(raw, index, sourceFrames, fps) {
    if (!raw || typeof raw !== "object") return null;
    const maxSource = Math.max(1, int(sourceFrames, 1));
    let sourceStart = int(raw.source_start ?? raw.sourceStart ?? raw.source_in ?? raw.sourceIn ?? raw.in_frame ?? raw.inFrame ?? raw.start_frame ?? raw.startFrame, 0);
    let sourceEnd = raw.source_end ?? raw.sourceEnd ?? raw.source_out ?? raw.sourceOut ?? raw.out_frame ?? raw.outFrame ?? raw.end_frame ?? raw.endFrame;
    if (sourceEnd == null) sourceEnd = sourceStart + int(raw.duration_frames ?? raw.frame_count ?? raw.frames, maxSource - sourceStart);
    sourceStart = clamp(sourceStart, 0, maxSource);
    if (bool(raw.source_end_inclusive ?? raw.sourceEndInclusive ?? raw.inclusive, false)) sourceEnd = int(sourceEnd, maxSource) + 1;
    sourceEnd = clamp(int(sourceEnd, maxSource), sourceStart, maxSource);
    if (sourceEnd <= sourceStart) return null;
    let timelineStart = Math.max(0, int(raw.timeline_start ?? raw.timelineStart ?? raw.start ?? raw.start_frame_timeline ?? raw.at_frame ?? raw.atFrame, 0));
    let timelineEnd = raw.timeline_end ?? raw.timelineEnd ?? raw.end_frame_timeline ?? raw.end;
    if (timelineEnd == null) timelineEnd = timelineStart + int(raw.timeline_duration_frames ?? raw.length_frames ?? raw.length, sourceEnd - sourceStart);
    timelineEnd = timelineStart + Math.max(0, int(timelineEnd, timelineStart) - timelineStart);
    if (bool(raw.timeline_end_inclusive ?? raw.timelineEndInclusive, false)) timelineEnd += 1;
    // A source frame can only be used once per timeline frame. Do not permit
    // stretching past source_end; this mirrors the Python normaliser.
    timelineEnd = Math.min(timelineEnd, timelineStart + sourceEnd - sourceStart);
    if (timelineEnd <= timelineStart) return null;
    const videoTrackRaw = raw.video_track ?? raw.videoTrack ?? raw.track ?? raw.track_index ?? raw.video_layer ?? 0;
    const audioTrackRaw = raw.audio_track ?? raw.audioTrack ?? raw.audio_layer ?? 0;
    const videoTrackText = String(videoTrackRaw).toLowerCase();
    const videoOnly = videoTrackText === "none" || videoTrackText === "audio" || videoTrackText === "audio-only" || int(videoTrackRaw, 0) < 0;
    const videoTrack = videoOnly ? -1 : (videoTrackText.includes("upper") || videoTrackText.includes("top") ? 1 : (int(videoTrackRaw, 0) > 0 ? 1 : 0));
    let audioTrack = String(audioTrackRaw).toLowerCase() === "none" || int(audioTrackRaw, 0) < 0 ? -1 : (String(audioTrackRaw).toLowerCase().includes("upper") || String(audioTrackRaw).toLowerCase().includes("top") ? 1 : (int(audioTrackRaw, 0) > 0 ? 1 : 0));
    const audioEnabledValue = raw.audio_enabled ?? raw.audioEnabled;
    const audioDisabledValue = raw.mute ?? raw.muted ?? raw.audio_disabled ?? raw.audioDisabled;
    const audioEnabled = audioEnabledValue == null ? !bool(audioDisabledValue, false) : bool(audioEnabledValue, true);
    if (!audioEnabled) audioTrack = -1;
    const linkedValue = raw.linked ?? raw.av_linked ?? raw.linkAudio ?? raw.link_audio;
    const linked = videoOnly ? false : (linkedValue == null ? true : bool(linkedValue, true));
    return {
        id: String(raw.id ?? raw.clip_id ?? raw.clipId ?? idFor(index)),
        source_start: sourceStart,
        source_end: sourceEnd,
        timeline_start: timelineStart,
        timeline_end: timelineEnd,
        video_track: videoTrack,
        audio_track: audioTrack,
        linked,
        transform: normalizeTransform(raw.transform ?? raw.transforms ?? raw),
        enabled: raw.enabled == null ? !bool(raw.disabled, false) : bool(raw.enabled, true),
        order: int(raw.order ?? raw.z ?? raw.placement_order ?? raw.placementOrder ?? index, index),
    };
}

function defaultState(sourceFrames, fps, threshold, minSceneSeconds) {
    const frames = Math.max(1, int(sourceFrames, 1));
    return {
        version: SCHEMA_VERSION,
        transform_version: TRANSFORM_VERSION,
        fps: number(fps, DEFAULT_FPS),
        duration_frames: frames,
        in_frame: 0,
        out_frame: frames,
        threshold: number(threshold, 0.5),
        min_scene_seconds: number(minSceneSeconds, 0),
        output: { width: -1, height: -1, multiple: DEFAULT_MULTIPLE, fit_mode: "letterbox", fill_color: DEFAULT_FILL },
        clips: [{
            id: "clip-1",
            source_start: 0,
            source_end: frames,
            timeline_start: 0,
            timeline_end: frames,
            video_track: 0,
            audio_track: 0,
            linked: true,
            transform: normalizeTransform({}),
            enabled: true,
            order: 0,
        }],
        audio_clips: [],
    };
}

function normalizeState(value, sourceFrames, fps, threshold = 0.5, minSceneSeconds = 0) {
    const raw = parseJson(value, {});
    const state = defaultState(sourceFrames, fps, threshold, minSceneSeconds);
    // Presence of an audio-only collection is explicit too; do not reinsert
    // the default lower-track source when the caller intentionally supplied
    // only audio data.
    const hasExplicitClips = ["clips", "video_clips", "videoClips", "audio_clips", "audioClips"]
        .some((key) => Object.prototype.hasOwnProperty.call(raw, key));
    let clipsRaw = Array.isArray(raw.clips)
        ? raw.clips.slice()
        : (raw.clips && typeof raw.clips === "object"
            ? Object.values(raw.clips)
            : (Array.isArray(raw.video_clips)
                ? raw.video_clips.slice()
                : (Array.isArray(raw.videoClips) ? raw.videoClips.slice() : [])));
    if (!hasExplicitClips) {
        for (const key of ["videoClips", "items", "segments"]) if (Array.isArray(raw[key])) clipsRaw.push(...raw[key]);
        for (const [name, value] of [["lower", raw.lower_track ?? raw.lowerTrack], ["upper", raw.upper_track ?? raw.upperTrack]]) {
            if (!Array.isArray(value)) continue;
            for (const item of value) if (item && typeof item === "object") clipsRaw.push({ ...item, video_track: name === "upper" ? 1 : 0 });
        }
    }
    const clips = clipsRaw.map((clip, index) => normalizeClip(clip, index, sourceFrames, fps)).filter(Boolean);
    if (clips.length || hasExplicitClips) state.clips = clips;
    const audioValue = raw.audio_clips ?? raw.audioClips;
    const audioRaw = Array.isArray(audioValue)
        ? audioValue
        : (audioValue && typeof audioValue === "object" ? Object.values(audioValue) : []);
    state.audio_clips = audioRaw.map((clip, index) => {
        const result = normalizeClip({ ...clip, video_track: -1 }, index, sourceFrames, fps);
        return result ? { ...result, video_track: -1 } : null;
    }).filter(Boolean);
    // The timeline ends at the furthest serialized clip. Keep a serialized
    // duration only as a compatibility value for an intentionally empty
    // timeline; it must not extend a populated timeline.
    const activeEnds = state.clips.concat(state.audio_clips)
        .map((clip) => int(clip.timeline_end, 0));
    const rawDuration = raw.duration_frames ?? raw.durationFrames ?? raw.timeline_duration_frames ?? raw.timelineDurationFrames;
    if (activeEnds.length) {
        state.duration_frames = Math.max(0, ...activeEnds);
    } else if (rawDuration != null) {
        state.duration_frames = Math.max(1, int(rawDuration, 1));
    } else {
        state.duration_frames = hasExplicitClips ? 1 : Math.max(1, int(sourceFrames, 1));
    }
    state.in_frame = clamp(int(raw.in_frame ?? raw.inFrame ?? raw.preview_in, 0), 0, state.duration_frames > 0 ? state.duration_frames - 1 : 0);
    const rawOut = raw.out_frame ?? raw.outFrame ?? raw.preview_out;
    state.out_frame = rawOut == null || int(rawOut, -1) < 0 ? state.duration_frames : clamp(int(rawOut, state.duration_frames), state.in_frame, state.duration_frames);
    if (state.out_frame <= state.in_frame && state.duration_frames > state.in_frame) state.out_frame = Math.min(state.duration_frames, state.in_frame + 1);
    state.threshold = clamp(number(raw.threshold ?? raw.Threshold, threshold), 0, 1);
    state.min_scene_seconds = Math.max(0, number(raw.min_scene_seconds ?? raw.MinSceneSeconds ?? raw.minSceneSeconds, minSceneSeconds));
    const output = raw.output && typeof raw.output === "object" ? raw.output : {};
    const outputWidth = output.width ?? output.output_width ?? output.outputWidth ?? raw.width ?? raw.output_width ?? raw.outputWidth;
    const outputHeight = output.height ?? output.output_height ?? output.outputHeight ?? raw.height ?? raw.output_height ?? raw.outputHeight;
    const outputMultiple = output.multiple ?? output.output_multiple ?? output.outputMultiple ?? raw.multiple ?? raw.output_multiple ?? raw.outputMultiple;
    const outputFit = output.fit_mode ?? output.fitMode ?? raw.fit_mode ?? raw.fitMode;
    const outputFill = output.fill_color ?? output.fillColor ?? raw.fill_color ?? raw.fillColor;
    state.output = {
        width: int(outputWidth, -1) > 0 ? int(outputWidth, -1) : -1,
        height: int(outputHeight, -1) > 0 ? int(outputHeight, -1) : -1,
        multiple: Math.max(1, int(outputMultiple, DEFAULT_MULTIPLE) > 0 ? int(outputMultiple, DEFAULT_MULTIPLE) : int(raw.multiple ?? raw.output_multiple ?? raw.outputMultiple, DEFAULT_MULTIPLE)),
        fit_mode: ["letterbox", "crop", "fill"].includes(String(outputFit ?? "").toLowerCase()) ? String(outputFit).toLowerCase() : "letterbox",
        fill_color: hex(outputFill, DEFAULT_FILL),
    };
    if (raw.source_fingerprint || raw.sourceFingerprint || raw.input_signature) state.source_fingerprint = String(raw.source_fingerprint ?? raw.sourceFingerprint ?? raw.input_signature);
    if (raw.source_fingerprint_version != null || raw.sourceFingerprintVersion != null) state.source_fingerprint_version = Math.max(1, int(raw.source_fingerprint_version ?? raw.sourceFingerprintVersion, 1));
    if (["file", "content", "metadata", "unknown"].includes(String(raw.source_fingerprint_kind ?? raw.sourceFingerprintKind ?? "").toLowerCase())) state.source_fingerprint_kind = String(raw.source_fingerprint_kind ?? raw.sourceFingerprintKind).toLowerCase();
    if (raw.source_identity || raw.sourceIdentity) state.source_identity = String(raw.source_identity ?? raw.sourceIdentity);
    if (raw.source_frame_count || raw.sourceFrameCount) state.source_frame_count = Math.max(0, int(raw.source_frame_count ?? raw.sourceFrameCount, 0));
    if (raw.source_fps || raw.sourceFps) state.source_fps = Math.max(0, number(raw.source_fps ?? raw.sourceFps, 0));
    if (raw.source_start_frame != null || raw.sourceStartFrame != null) state.source_start_frame = Math.max(0, int(raw.source_start_frame ?? raw.sourceStartFrame, 0));
    if (raw.source_end_frame != null || raw.sourceEndFrame != null) state.source_end_frame = Math.max(0, int(raw.source_end_frame ?? raw.sourceEndFrame, 0));
    state._nextOrder = Math.max(0, ...state.clips.concat(state.audio_clips).map((clip) => int(clip.order, 0))) + 1;
    return state;
}

function publicState(state) {
    const result = { ...state };
    delete result._nextOrder;
    return result;
}

function sourceVideoUrl(filename) {
    const params = new URLSearchParams({ filename: String(filename || ""), t: String(Date.now()) });
    return api.apiURL(`/cinestyle/video-source?${params.toString()}`);
}

function infoFrames(info) {
    const direct = int(info?.frames, 0) || int(info?.frame_count, 0) || int(info?.loaded_frame_count, 0);
    return Math.max(1, direct || Math.round(number(info?.duration, 1) * number(info?.fps, DEFAULT_FPS)));
}

function infoFps(info) {
    const parseRate = (value, fallback = 0) => {
        if (typeof value === "string" && value.includes("/")) {
            const [numerator, denominator] = value.split("/", 2).map(Number);
            return Number.isFinite(numerator) && Number.isFinite(denominator) && denominator !== 0 ? numerator / denominator : fallback;
        }
        return number(value, fallback);
    };
    return Math.max(0.001, parseRate(info?.fps, 0) || parseRate(info?.frame_rate, 0) || parseRate(info?.loaded_fps, DEFAULT_FPS));
}

function sourceCanvasDimensions(info = {}) {
    // CS Load Video's preview manifest contains both original-file dimensions
    // (source_width/source_height) and the dimensions actually delivered by
    // the connected VIDEO (loaded_width/loaded_height). Timeline rendering
    // must use the latter; generic VIDEO caches use source_width as their
    // actual tensor size and therefore fall through to it.
    const loadedWidth = int(info?.loaded_width, 0);
    const loadedHeight = int(info?.loaded_height, 0);
    if (loadedWidth > 0 && loadedHeight > 0) return { width: loadedWidth, height: loadedHeight };
    const contentWidth = int(info?.content_width, 0);
    const contentHeight = int(info?.content_height, 0);
    if (contentWidth > 0 && contentHeight > 0) return { width: contentWidth, height: contentHeight };
    // Node-owned source caches deliberately store a low-resolution tensor;
    // their ``source_width`` describes provenance, while ``width/height`` is
    // the actual preview frame size.
    if (info?.source === true || info?.source === "true") {
        const previewWidth = int(info?.width, 0);
        const previewHeight = int(info?.height, 0);
        if (previewWidth > 0 && previewHeight > 0) return { width: previewWidth, height: previewHeight };
    }
    return {
        width: int(info?.source_width ?? info?.width, 0),
        height: int(info?.source_height ?? info?.height, 0),
    };
}

function formatTime(frame, fps) {
    const seconds = Math.max(0, number(frame, 0) / Math.max(0.001, number(fps, DEFAULT_FPS)));
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainder = seconds % 60;
    const body = `${String(minutes).padStart(2, "0")}:${remainder.toFixed(2).padStart(5, "0")}`;
    return hours ? `${String(hours).padStart(2, "0")}:${body}` : body;
}

function addStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .cs-time-edit-dialog { width:min(1280px,97vw); max-width:none; max-height:96vh; overflow:auto; padding:0; border:1px solid #343943; border-radius:10px; background:#17191e; color:#e6e9ef; box-shadow:0 24px 90px #000d; }
      .cs-time-edit-dialog::backdrop { background:#050609c9; }
      .cs-time-edit-shell { display:grid; gap:10px; padding:14px; font:13px/1.35 system-ui,sans-serif; min-width:0; }
      .cs-time-edit-head,.cs-time-edit-actions,.cs-time-edit-row,.cs-time-edit-controls { display:flex; align-items:center; gap:8px; min-width:0; }
      .cs-time-edit-head { justify-content:space-between; }
      .cs-time-edit-head h2 { margin:0; font-size:17px; }
      .cs-time-edit-muted { color:#9299a8; font-size:12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
      .cs-time-edit-close { border:0; background:transparent; color:#aeb5c2; font-size:23px; cursor:pointer; padding:0 4px; }
      .cs-time-edit-stage { position:relative; width:100%; aspect-ratio:16/9; min-height:220px; max-height:48vh; background:#17191e; border:1px solid #363b45; border-radius:6px; overflow:hidden; }
      .cs-time-edit-stage video,.cs-time-edit-stage img { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; background:transparent; }
      .cs-time-edit-stage img { pointer-events:none; display:none; }
      .cs-time-edit-stage .local-transform { transform-origin:center center; transition:transform .06s linear; }
      .cs-time-edit-stage-status { position:absolute; left:50%; top:50%; transform:translate(-50%,-50%); min-width:min(74%,440px); padding:14px 18px; border-radius:6px; color:#dce6f4; background:#17191ee8; font-size:16px; line-height:1.4; text-align:center; pointer-events:none; }
      .cs-time-edit-stage-status[hidden] { display:none; }
      .cs-time-edit-stage-progress { display:block; height:5px; margin-top:10px; overflow:hidden; border-radius:3px; background:#343943; }
      .cs-time-edit-stage-progress-bar { display:block; width:0%; height:100%; background:#55a9f5; transition:width .16s ease; }
      .cs-time-edit-readout { display:flex; justify-content:space-between; color:#aeb5c2; font-variant-numeric:tabular-nums; }
      .cs-time-edit-controls { flex-wrap:wrap; }
      .cs-time-edit-shot-group { display:flex; align-items:center; gap:7px; margin-left:auto; }
      .cs-time-edit-shot-group input[type=number] { width:66px; }
      .cs-time-edit-shot-group label { display:flex; align-items:center; gap:4px; color:#9da5b4; }
      .cs-time-edit-controls button,.cs-time-edit-actions button,.cs-time-edit-editor button { border:1px solid #424956; border-radius:5px; padding:6px 9px; background:#242832; color:#e6e9ef; cursor:pointer; }
      .cs-time-edit-controls button:hover,.cs-time-edit-actions button:hover,.cs-time-edit-editor button:hover { border-color:#6aa9df; }
      .cs-time-edit-controls button:disabled { opacity:.45; cursor:not-allowed; }
      .cs-time-edit-controls .primary,.cs-time-edit-actions .primary { background:#317ec4; border-color:#4b9de8; }
      .cs-time-edit-controls input[type=number] { width:70px; }
      .cs-time-edit-timeline-tools { display:flex; align-items:center; gap:6px; }
      .cs-time-edit-timeline-tools button { border:1px solid #424956; border-radius:5px; padding:5px 10px; background:#242832; color:#e6e9ef; cursor:pointer; }
      .cs-time-edit-timeline-tools button:hover { border-color:#6aa9df; }
      .cs-time-edit-timeline-tools button:disabled { opacity:.45; cursor:not-allowed; }
      .cs-time-edit-timeline-tools .cs-time-edit-zoom-state { margin-left:4px; color:#9299a8; font-size:11px; font-variant-numeric:tabular-nums; }
      .cs-time-edit-timeline { border:1px solid #363b45; border-radius:6px; background:#20232a; overflow:hidden; min-height:190px; }
      .cs-time-edit-inner { position:relative; width:100%; min-width:0; padding-bottom:5px; user-select:none; }
      .cs-time-edit-pointer-row { position:relative; height:15px; user-select:none; }
      .cs-time-edit-pointer { position:absolute; top:0; width:16px; height:15px; transform:translateX(-50%); border:0; padding:0; background:#55a9f5; clip-path:polygon(0 0,100% 0,50% 100%); cursor:ew-resize; z-index:20; }
      .cs-time-edit-axis { position:relative; height:22px; border-bottom:1px solid #343943; color:#9299a8; font-size:11px; font-variant-numeric:tabular-nums; }
      .cs-time-edit-axis span { position:absolute; top:4px; transform:translateX(-50%); white-space:nowrap; }
      .cs-time-edit-range { position:absolute; top:37px; bottom:5px; background:rgba(188,198,210,.16); border-left:1px solid rgba(210,220,230,.6); border-right:1px solid rgba(210,220,230,.6); pointer-events:none; z-index:8; }
      .cs-time-edit-range-marker { position:absolute; top:0; bottom:0; width:2px; background:#c9d4df; box-shadow:0 0 0 1px #15181d; pointer-events:auto; cursor:ew-resize; touch-action:none; }
      .cs-time-edit-range-marker::after { content:""; position:absolute; top:0; bottom:0; left:-9px; width:20px; background:transparent; pointer-events:auto; cursor:ew-resize; }
      .cs-time-edit-range-marker::before { content:""; position:absolute; top:-1px; width:0; height:0; border-left:5px solid transparent; border-right:5px solid transparent; border-top:6px solid #c9d4df; }
      .cs-time-edit-range-marker.in { left:-1px; } .cs-time-edit-range-marker.in::before { left:-4px; }
      .cs-time-edit-range-marker.out { right:-1px; } .cs-time-edit-range-marker.out::before { right:-4px; }
      .cs-time-edit-track { position:relative; height:34px; border-bottom:1px solid #30343c; }
      .cs-time-edit-track-label { position:absolute; left:5px; top:9px; z-index:4; color:#9da5b4; font-size:11px; width:95px; pointer-events:none; }
      .cs-time-edit-track-body { position:absolute; inset:0 0 0 100px; cursor:crosshair; }
      .cs-time-edit-clip { position:absolute; top:5px; height:24px; min-width:5px; border:1px solid #5f9ed1; border-radius:3px; box-sizing:border-box; padding:3px 13px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#f2f4f7; background:#30658aaa; cursor:grab; user-select:none; font-size:11px; }
      .cs-time-edit-clip.audio { background:#6d4c8aaa; border-color:#a88bd2; }
      .cs-time-edit-clip.upper { background:#a86436aa; border-color:#e4aa70; }
      .cs-time-edit-clip.selected { outline:2px solid #f7b955; outline-offset:1px; z-index:15; }
      .cs-time-edit-clip.dragging { cursor:grabbing; opacity:.82; }
      .cs-time-edit-edge { position:absolute; top:0; bottom:0; width:8px; cursor:ew-resize; z-index:2; }
      .cs-time-edit-edge.left { left:0; } .cs-time-edit-edge.right { right:0; }
      .cs-time-edit-editor { display:grid; grid-template-columns:minmax(220px,1fr) minmax(0,2fr); gap:10px; }
      .cs-time-edit-panel { border:1px solid #363b45; border-radius:6px; padding:9px; background:#1d2026; min-width:0; }
      .cs-time-edit-panel h3 { margin:0 0 7px; font-size:12px; color:#bfc7d5; }
      .cs-time-edit-fields { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:7px; }
      .cs-time-edit-output-fields { grid-template-columns:repeat(2,minmax(0,1fr)); gap:5px; }
      .cs-time-edit-output-fields .cs-time-edit-field { grid-template-columns:minmax(0,1fr) 29px; align-items:end; }
      .cs-time-edit-output-fields .cs-time-edit-field > input,.cs-time-edit-output-fields .cs-time-edit-field > select { grid-column:1; }
      .cs-time-edit-output-fields .cs-time-edit-field > .cs-default { grid-column:2; grid-row:2; }
      .cs-time-edit-output-fields .cs-time-edit-field > .cs-fill-color { grid-column:1; grid-row:2; }
      .cs-time-edit-output-fields .cs-time-edit-field > .cs-fill-text { grid-column:1; grid-row:3; padding:4px 6px; }
      .cs-time-edit-output-fields .cs-time-edit-field > .cs-default { grid-row:2 / span 2; align-self:center; }
      .cs-default { width:29px; min-height:27px; padding:3px !important; border:1px solid #424956; border-radius:5px; background:#20232a; color:#f2f4f7; cursor:pointer; font-size:15px; line-height:1; }
      .cs-time-edit-field { display:grid; gap:3px; color:#9da5b4; min-width:0; }
      .cs-time-edit-field input,.cs-time-edit-field select { width:100%; min-width:0; box-sizing:border-box; border:1px solid #424956; border-radius:4px; padding:6px; background:#20232a; color:#f2f4f7; }
      .cs-time-edit-field input[type=color] { height:30px; padding:2px; }
      .cs-time-edit-transform-grid { display:grid; grid-template-columns:1fr; gap:15px; }
      .cs-time-edit-transform-line { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); align-items:center; gap:10px; min-width:0; }
      .cs-time-edit-transform-line.scale-line { grid-template-columns:minmax(0,1fr) 34px minmax(0,1fr); }
      .cs-time-edit-transform-row { display:grid; grid-template-columns:68px minmax(0,1fr) minmax(74px,92px) 29px; align-items:center; gap:7px; min-width:0; }
      .cs-time-edit-transform-row label { color:#cbd2dc; white-space:nowrap; }
      .cs-time-edit-transform-row input[type=range] { width:100%; min-width:0; accent-color:#55a9f5; }
      .cs-time-edit-transform-value { width:100%; min-width:0; box-sizing:border-box; border:1px solid #424956; border-radius:4px; padding:5px 6px; background:#20232a; color:#f2f4f7; font-variant-numeric:tabular-nums; }
      .cs-time-edit-sync { width:34px; min-height:27px; padding:3px !important; font-size:14px; }
      .cs-time-edit-sync[aria-pressed="true"] { color:#9fd3ff; border-color:#55a9f5; background:#263d51; }
      .cs-time-edit-checks { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-top:7px; color:#c6cdd8; }
      .cs-time-edit-checks label { display:flex; align-items:center; gap:4px; }
      .cs-time-edit-context-menu { position:fixed; z-index:60; display:grid; min-width:260px; padding:4px; gap:2px; border:1px solid #424956; border-radius:6px; background:#20232a; box-shadow:0 10px 32px #000b; }
      .cs-time-edit-context-menu[hidden] { display:none; }
      .cs-time-edit-context-menu button { border:0; border-radius:4px; padding:8px 10px; background:transparent; color:#f2f4f7; text-align:left; cursor:pointer; }
      .cs-time-edit-context-menu button:hover { background:#317ec4; }
      .cs-time-edit-context-menu button:disabled { opacity:.42; cursor:not-allowed; }
      .cs-time-edit-actions { justify-content:flex-end; }
      .cs-time-edit-actions .cs-time-edit-history-spacer { width:72px; flex:0 0 72px; }
      .cs-time-edit-status { flex:1; min-width:0; color:#9299a8; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
      .cs-time-edit-proxy { color:#f7b955; font-size:11px; }
      @media(max-width:850px) { .cs-time-edit-editor{grid-template-columns:1fr;} .cs-time-edit-fields{grid-template-columns:repeat(2,minmax(0,1fr));} }
      @media(max-width:500px) { .cs-time-edit-shell{padding:9px;} .cs-time-edit-fields{grid-template-columns:1fr 1fr;} .cs-time-edit-transform-line,.cs-time-edit-transform-line.scale-line{grid-template-columns:1fr;} .cs-time-edit-sync{justify-self:start;} }
    `;
    document.head.append(style);
}

async function fetchTimelineState(node) {
    const nodeId = String(node?.id ?? "").trim();
    if (!nodeId) return null;
    try {
        const response = await api.fetchApi(`/cinestyle/video-time-edit-state?${new URLSearchParams({ node_id: nodeId, t: String(Date.now()) })}`);
        if (!response.ok) return null;
        const result = await response.json();
        return result && typeof result === "object" ? result : null;
    } catch (_) { return null; }
}

async function saveTimelineState(node, state) {
    const nodeId = String(node?.id ?? "").trim();
    if (!nodeId) return;
    try {
        const revision = Math.max(0, int(STATE_REVISIONS.get(node), 0)) + 1;
        STATE_REVISIONS.set(node, revision);
        await api.fetchApi("/cinestyle/video-time-edit-state", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ node_id: nodeId, revision, timeline: publicState(state) }),
        });
    } catch (_) { /* state is also persisted in timeline_json */ }
}

function editableTimelineSnapshot(state) {
    const snapshot = publicState(state || {});
    return JSON.parse(JSON.stringify({
        version: snapshot.version,
        transform_version: snapshot.transform_version,
        fps: snapshot.fps,
        duration_frames: snapshot.duration_frames,
        clips: snapshot.clips || [],
        audio_clips: snapshot.audio_clips || [],
        output: snapshot.output || {},
        source_identity: snapshot.source_identity || "",
        source_fingerprint: snapshot.source_fingerprint || "",
        source_fingerprint_version: snapshot.source_fingerprint_version || SOURCE_FINGERPRINT_VERSION,
        source_fingerprint_kind: snapshot.source_fingerprint_kind || "",
        source_frame_count: snapshot.source_frame_count || 0,
        source_fps: snapshot.source_fps || 0,
        source_start_frame: snapshot.source_start_frame,
        source_end_frame: snapshot.source_end_frame,
    }));
}

async function timelineHistoryRequest(node, action, timeline = null, label = "") {
    const nodeId = String(node?.id ?? "").trim();
    if (!nodeId) return null;
    try {
        const body = { node_id: nodeId, action };
        if (timeline) body.timeline = timeline;
        if (label) body.label = label;
        const response = await api.fetchApi("/cinestyle/video-time-edit-history", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        if (!response.ok) return null;
        const result = await response.json().catch(() => null);
        return result && typeof result === "object" ? result : null;
    } catch (_) { return null; }
}

function buildPreviewInfoUrl(node, cacheKey = "", sourceIdentity = "") {
    const params = new URLSearchParams({ node_id: String(node?.id ?? ""), t: String(Date.now()) });
    if (cacheKey) params.set("cache_key", cacheKey);
    if (sourceIdentity) params.set("source_identity", sourceIdentity);
    return `/cinestyle/video-time-edit-preview-info?${params}`;
}

async function fetchProxy(node, cacheKey = "", expectedIdentity = "", expectedSourceFrames = 0) {
    try {
        const response = await api.fetchApi(buildPreviewInfoUrl(node, cacheKey, expectedIdentity));
        const result = await response.json().catch(() => ({}));
        if (!response.ok) return null;
        const info = result.info || {};
        const actualIdentity = String(info.source_identity || info.source_fingerprint || "").trim();
        if (expectedIdentity && actualIdentity && actualIdentity !== expectedIdentity) return null;
        if (expectedSourceFrames > 0 && Number(info.source_frame_count || 0) > 0 && Number(info.source_frame_count) !== expectedSourceFrames) return null;
        return {
            url: api.apiURL(String(result.video_url || "")),
            token: String(result.token || ""),
            info,
            label: String(result.label || "Timeline proxy preview"),
        };
    } catch (_) { return null; }
}

async function buildTimelineProxy(node, state, controls, source, onProgress = null) {
    const payload = {
        node_id: String(node?.id ?? ""),
        timeline: publicState(state),
        timeline_json: canonicalJson(publicState(state)),
        source_token: String(source?.cached?.token || ""),
        video_filename: String(source?.filename || ""),
        source_cache_key: String(source?.sourceCacheKey || ""),
        source_identity: String(source?.info?.source_identity || source?.info?.source_fingerprint || source?.info?.input_signature || ""),
        source_frame_count: int(source?.info?.frames ?? source?.info?.loaded_frame_count ?? source?.info?.source_frame_count, 0),
        source_fingerprint_kind: String(source?.info?.source_fingerprint_kind || ""),
        source_start_frame: int(source?.startFrame, 0),
        source_end_frame: int(source?.endFrame, -1),
        source_target_fps: number(source?.targetFps, 0),
        source_output_width: int(source?.outputWidth, 0),
        source_output_height: int(source?.outputHeight, 0),
        source_output_multiple: Math.max(1, int(source?.multiple, DEFAULT_MULTIPLE)),
        source_width: sourceCanvasDimensions(source?.info).width,
        source_height: sourceCanvasDimensions(source?.info).height,
        width: dimensionRequest(controls.width.value, int(controls.multiple.value, DEFAULT_MULTIPLE)),
        height: dimensionRequest(controls.height.value, int(controls.multiple.value, DEFAULT_MULTIPLE)),
        multiple: Math.max(1, int(controls.multiple.value, DEFAULT_MULTIPLE)),
        fit_mode: String(controls.fit.value || "letterbox"),
        fill_color: hex(controls.fillText.value, DEFAULT_FILL),
    };
    try {
        const response = await api.fetchApi("/cinestyle/video-time-edit-proxy", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        let result = await response.json().catch(() => ({}));
        if (!response.ok) return null;
        onProgress?.(clamp(number(result.progress, 0), 0, 100), result);
        if (result.status === "ready") return result;
        const jobKey = String(result.job_key || "");
        if (!jobKey) return null;
        const started = Date.now();
        while (Date.now() - started < 300000) {
            await new Promise((resolve) => window.setTimeout(resolve, 250));
            const progressResponse = await api.fetchApi(`/cinestyle/video-time-edit-proxy-progress?${new URLSearchParams({ job_key: jobKey })}`).catch(() => null);
            if (!progressResponse) continue;
            result = await progressResponse.json().catch(() => ({}));
            onProgress?.(clamp(number(result.progress, 0), 0, 100), result);
            if (result.status === "ready" || result.status === "failed") return result;
        }
        return { status: "failed", error: "Timed out while generating the timeline proxy." };
    } catch (_) {
        return null;
    }
}

async function fetchSourceCache(node) {
    const nodeId = String(node?.id ?? "").trim();
    if (!nodeId) return null;
    // Newer backends expose the original low-resolution source cache so a
    // generic VIDEO (which has no filename) can still be scrubbed and cut.
    // Older backends simply return 404; the editor then uses the input-chain
    // cache or the persisted proxy as before.
    try {
        const response = await api.fetchApi(`/cinestyle/video-time-edit-source-info?${new URLSearchParams({ node_id: nodeId, t: String(Date.now()) })}`);
        const result = await response.json().catch(() => ({}));
        if (!response.ok || !result.video_url) return null;
        return { url: api.apiURL(String(result.video_url)), token: String(result.token || ""), info: result.info || {}, label: String(result.label || "Timeline source cache") };
    } catch (_) { return null; }
}

function directVideoOrigin(node) {
    const index = node?.inputs?.findIndex((item) => item?.name === "video");
    if (index == null || index < 0) return null;
    const calls = [
        () => node.getInputNode?.(index),
        () => node.getInputNode?.("video"),
        () => node.getInputLink?.(index),
        () => node.getInputLink?.("video"),
    ];
    for (const call of calls) {
        try {
            const candidate = call();
            if ((candidate?.type || candidate?.comfyClass) && (candidate?.inputs || candidate?.widgets)) return candidate;
            const graph = node.graph || app.graph;
            const link = graph?.links?.[candidate] || graph?._links?.[candidate] || candidate?.link || candidate;
            const originId = link?.origin_id ?? link?.originId ?? link?.origin;
            const origin = graph?.getNodeById?.(originId) || (graph?._nodes || graph?.nodes || []).find((item) => String(item?.id) === String(originId));
            if (origin) return origin;
        } catch (_) { /* try the next ComfyUI graph API shape */ }
    }
    return null;
}

function directOriginMaySeekFile(origin) {
    const type = String(origin?.type || origin?.comfyClass || origin?.constructor?.type || "");
    return /CS[_:.]Load[_:.]?Video|load[_:.]?video/i.test(type) || /video[_:.]?loader/i.test(type);
}

async function findSource(node, onProgress = null) {
    let source = connectedVideoSource(node, ["video"]);
    const chain = connectedInputChain(node, ["video"]);
    const directOrigin = directVideoOrigin(node);
    const directFileSource = directOriginMaySeekFile(directOrigin);
    let cached = null;
    if (chain) {
        onProgress?.(0, { stage: "cache" });
        cached = await fetchWaitInputCache(chain).catch(() => null);
        if (cached) onProgress?.(100, { stage: "cache" });
    }
    if (directFileSource && source?.isCSLoad && source.filename) {
        const shared = await ensureLoaderPreviewSource(source, { onProgress: (progress, result) => onProgress?.(progress, result) }).catch(() => null);
        if (shared?.url) cached = shared;
    }
    let info = cached?.info || null;
    // A recursive filename from an intermediate transform is not the actual
    // VIDEO input.  Use it only for a direct CS Load Video connection; other
    // upstream nodes must have a wait-input/source cache generated from their
    // evaluated VIDEO value.
    const filename = directFileSource ? String(source?.filename || "") : "";
    if (!info && filename) info = await fetchInfo(filename).catch(() => null);
    if (!info && cached?.url) info = cached.info || null;
    if (!cached) {
        const sourceCache = await fetchSourceCache(node);
        if (sourceCache) { cached = sourceCache; info = sourceCache.info || info; }
    }
    if (!info) {
        const proxy = await fetchProxy(node);
        if (proxy?.info) info = proxy.info;
    }
    info = info || {};
    if (info && source && directFileSource && !cached?.sharedLoaderCache && !cached?.waitInputCache && (Number(source.startFrame || 0) > 0 || Number(source.endFrame ?? -1) >= 0 || Number(source.targetFps || 0) > 0 || Number(source.outputWidth || 0) > 0 || Number(source.outputHeight || 0) > 0)) {
        // If the optional shared loader cache could not be built, still model
        // the selected CS Load Video window instead of presenting the entire
        // file as the connected VIDEO.
        info = prepareInputTimeline(source, info);
    }
    if (info && source?.isCSLoad && directFileSource && !cached?.sharedLoaderCache && !cached?.waitInputCache) {
        // Mirror CS Load Video's output canvas for the rare direct-file
        // fallback (the shared cache normally supplies these fields).
        const baseWidth = Math.max(1, int(info.source_width ?? info.width, 1));
        const baseHeight = Math.max(1, int(info.source_height ?? info.height, 1));
        const multiple = Math.max(1, int(source.multiple, DEFAULT_MULTIPLE));
        const requestedWidth = Math.max(0, int(source.outputWidth, 0));
        const requestedHeight = Math.max(0, int(source.outputHeight, 0));
        const targetWidth = requestedWidth || (requestedHeight ? Math.round(requestedHeight * baseWidth / baseHeight) : baseWidth);
        const targetHeight = requestedHeight || (requestedWidth ? Math.round(requestedWidth * baseHeight / baseWidth) : baseHeight);
        info = { ...info, loaded_width: ceilMultiple(targetWidth, multiple), loaded_height: ceilMultiple(targetHeight, multiple) };
    }
    return {
        source,
        chain,
        cached,
        directFileSource,
        sourceCacheKey: String(cached?.info?.cache_fingerprint || cached?.info?.cache_key || ""),
        filename,
        info,
        fps: infoFps(info),
        frames: infoFrames(info),
        url: cached?.url || (filename ? sourceVideoUrl(filename) : ""),
        label: cached?.label || filename || "VIDEO input",
    };
}

function activeClip(state, frame) {
    const clips = (state?.clips || []).filter((clip) => clip.enabled && clip.video_track >= 0 && clip.timeline_start <= frame && frame < clip.timeline_end);
    if (!clips.length) return null;
    return clips.reduce((best, clip) => (!best || clip.video_track > best.video_track || (clip.video_track === best.video_track && (clip.order > best.order || (clip.order === best.order && clip.timeline_start > best.timeline_start)))) ? clip : best, null);
}

function transformCss(transform, stage) {
    const t = normalizeTransform(transform);
    const rect = stage?.getBoundingClientRect?.();
    const tx = t.translation_unit === "pixel" ? t.translate_x : t.translate_x * (rect?.width || 1);
    const ty = t.translation_unit === "pixel" ? t.translate_y : t.translate_y * (rect?.height || 1);
    const sx = t.scale_x * (t.flip_x ? -1 : 1);
    const sy = t.scale_y * (t.flip_y ? -1 : 1);
    // Keep the quick CSS feedback in the same order as the renderer:
    // center -> scale/mirror -> rotation -> translation.
    return `translate3d(${tx}px,${ty}px,0) rotate(${t.rotation}deg) scale(${sx},${sy})`;
}

function openTimeline(node) {
    addStyles();
    const dialog = document.createElement("dialog");
    dialog.className = "cs-time-edit-dialog";
    dialog.innerHTML = `
      <div class="cs-time-edit-shell">
        <div class="cs-time-edit-head"><div><h2>Edit Timeline</h2><div class="cs-time-edit-muted cs-source-label">Loading VIDEO input…</div><div class="cs-time-edit-proxy" aria-live="polite"></div></div><button class="cs-time-edit-close" type="button" aria-label="Close">&times;</button></div>
        <div class="cs-time-edit-stage"><video class="cs-stage-video" playsinline preload="metadata"></video><img class="cs-stage-frame" alt="Timeline frame" draggable="false"><div class="cs-time-edit-stage-status" hidden><span class="cs-time-edit-stage-status-text"></span><span class="cs-time-edit-stage-progress"><span class="cs-time-edit-stage-progress-bar"></span></span></div></div>
        <div class="cs-time-edit-readout"><span class="cs-current">00:00.00</span><span class="cs-range"></span><span class="cs-duration">00:00.00</span></div>
        <div class="cs-time-edit-controls"><button class="cs-set-in" type="button">Set In</button><input class="cs-in" type="number" min="0" step="1" title="In frame"><button class="cs-step-back" type="button">|&lt;</button><button class="cs-play primary" type="button">Play</button><button class="cs-step-forward" type="button">&gt;|</button><input class="cs-out" type="number" min="1" step="1" title="Out frame (exclusive)"><button class="cs-set-out" type="button">Set Out</button><span class="cs-time-edit-shot-group"><button class="cs-shot primary" type="button">Detect Shots</button><label>Threshold <input class="cs-threshold" type="number" min="0" max="1" step="0.01"></label><label>Min scene seconds <input class="cs-min-scene" type="number" min="0" max="60" step="0.01"></label></span></div>
        <div class="cs-time-edit-timeline-tools"><button class="cs-time-edit-zoom-in" type="button" title="Zoom in">+</button><button class="cs-time-edit-zoom-fit" type="button" title="Fit timeline">Fit</button><button class="cs-time-edit-zoom-out" type="button" title="Zoom out">−</button><span class="cs-time-edit-zoom-state">Fit</span></div>
        <div class="cs-time-edit-timeline"><div class="cs-time-edit-inner"><div class="cs-time-edit-pointer-row"><button class="cs-time-edit-pointer" type="button" aria-label="Current frame"></button></div><div class="cs-time-edit-axis"></div><div class="cs-time-edit-range"><span class="cs-time-edit-range-marker in"></span><span class="cs-time-edit-range-marker out"></span></div><div class="cs-time-edit-track" data-kind="video" data-track="1"><span class="cs-time-edit-track-label">Video1</span><div class="cs-time-edit-track-body"></div></div><div class="cs-time-edit-track" data-kind="video" data-track="0"><span class="cs-time-edit-track-label">Video2</span><div class="cs-time-edit-track-body"></div></div><div class="cs-time-edit-track" data-kind="audio" data-track="1"><span class="cs-time-edit-track-label">Audio1</span><div class="cs-time-edit-track-body"></div></div><div class="cs-time-edit-track" data-kind="audio" data-track="0"><span class="cs-time-edit-track-label">Audio2</span><div class="cs-time-edit-track-body"></div></div></div></div>
        <div class="cs-time-edit-editor"><section class="cs-time-edit-panel"><h3>Output canvas</h3><div class="cs-time-edit-fields cs-time-edit-output-fields"><label class="cs-time-edit-field">Width<input class="cs-width" type="number" min="-1" step="1"><button class="cs-default" data-reset="width" type="button" title="Reset Width">&#8634;</button></label><label class="cs-time-edit-field">Height<input class="cs-height" type="number" min="-1" step="1"><button class="cs-default" data-reset="height" type="button" title="Reset Height">&#8634;</button></label><label class="cs-time-edit-field">Multiple<input class="cs-multiple" type="number" min="1" step="1"><button class="cs-default" data-reset="multiple" type="button" title="Reset Multiple">&#8634;</button></label><label class="cs-time-edit-field">Fit<select class="cs-fit"><option value="letterbox">letterbox</option><option value="crop">crop</option><option value="fill">fill (stretch)</option></select></label><label class="cs-time-edit-field">Fill color<input class="cs-fill-color" type="color"><input class="cs-fill-text" type="text" maxlength="7" spellcheck="false"></label></div></section><section class="cs-time-edit-panel"><h3>Selected clip <span class="cs-selected-label"></span></h3><div class="cs-time-edit-transform-grid"><div class="cs-time-edit-transform-line scale-line"><div class="cs-time-edit-transform-row scale-row"><label for="cs-scale-x">Scale X</label><input id="cs-scale-x" class="cs-scale-x" type="range" min="0.1" max="4" step="0.01"><input class="cs-time-edit-transform-value cs-scale-x-value" type="number" min="0.1" max="4" step="0.01"><button class="cs-default" data-reset="scale-x" type="button" title="Reset Scale X">&#8634;</button></div><button class="cs-time-edit-sync" type="button" aria-pressed="true" title="Sync Scale X and Y">&#128279;</button><div class="cs-time-edit-transform-row scale-row"><label for="cs-scale-y">Scale Y</label><input id="cs-scale-y" class="cs-scale-y" type="range" min="0.1" max="4" step="0.01"><input class="cs-time-edit-transform-value cs-scale-y-value" type="number" min="0.1" max="4" step="0.01"><button class="cs-default" data-reset="scale-y" type="button" title="Reset Scale Y">&#8634;</button></div></div><div class="cs-time-edit-transform-line"><div class="cs-time-edit-transform-row translate-row"><label for="cs-translate-x">Translate X</label><input id="cs-translate-x" class="cs-translate-x" type="range" min="-1" max="1" step="0.01"><input class="cs-time-edit-transform-value cs-translate-x-value" type="number" min="-1" max="1" step="0.01"><button class="cs-default" data-reset="translate-x" type="button" title="Reset Translate X">&#8634;</button></div><div class="cs-time-edit-transform-row translate-row"><label for="cs-translate-y">Translate Y</label><input id="cs-translate-y" class="cs-translate-y" type="range" min="-1" max="1" step="0.01"><input class="cs-time-edit-transform-value cs-translate-y-value" type="number" min="-1" max="1" step="0.01"><button class="cs-default" data-reset="translate-y" type="button" title="Reset Translate Y">&#8634;</button></div></div><div class="cs-time-edit-transform-line"><div class="cs-time-edit-transform-row rotation-row"><label for="cs-rotation">Rotation</label><input id="cs-rotation" class="cs-rotation" type="range" min="-90" max="90" step="0.1"><input class="cs-time-edit-transform-value cs-rotation-value" type="number" min="-90" max="90" step="0.1"><button class="cs-default" data-reset="rotation" type="button" title="Reset Rotation">&#8634;</button></div><label class="cs-time-edit-field cs-time-edit-mirror-field">Mirror<select class="cs-mirror"><option value="none">none</option><option value="horizontal">horizontal</option><option value="vertical">vertical</option></select></label></div></div></section></div>
        <div class="cs-time-edit-context-menu" hidden></div>
        <div class="cs-time-edit-actions"><span class="cs-time-edit-status" aria-live="polite"></span><button class="cs-undo" type="button" title="Undo" disabled>&#8630; Undo</button><button class="cs-redo" type="button" title="Redo" disabled>&#8631; Redo</button><span class="cs-time-edit-history-spacer"></span><button class="cs-cancel" type="button">Cancel</button><button class="cs-apply primary" type="button">Apply to node</button></div>
      </div>`;
    document.body.append(dialog);
    dialog.showModal();

    const stage = dialog.querySelector(".cs-time-edit-stage");
    const stageVideo = dialog.querySelector(".cs-stage-video");
    const stageFrame = dialog.querySelector(".cs-stage-frame");
    const stageStatus = dialog.querySelector(".cs-time-edit-stage-status");
    const stageStatusText = dialog.querySelector(".cs-time-edit-stage-status-text");
    const stageProgressBar = dialog.querySelector(".cs-time-edit-stage-progress-bar");
    const sourceLabel = dialog.querySelector(".cs-source-label");
    const proxyLabel = dialog.querySelector(".cs-time-edit-proxy");
    const status = dialog.querySelector(".cs-time-edit-status");
    const inner = dialog.querySelector(".cs-time-edit-inner");
    const timelineViewport = dialog.querySelector(".cs-time-edit-timeline");
    const zoomStateLabel = dialog.querySelector(".cs-time-edit-zoom-state");
    const pointerRow = dialog.querySelector(".cs-time-edit-pointer-row");
    const axis = dialog.querySelector(".cs-time-edit-axis");
    const rangeBand = dialog.querySelector(".cs-time-edit-range");
    const pointer = dialog.querySelector(".cs-time-edit-pointer");
    const inMarker = rangeBand.querySelector(".cs-time-edit-range-marker.in");
    const outMarker = rangeBand.querySelector(".cs-time-edit-range-marker.out");
    const currentLabel = dialog.querySelector(".cs-current");
    const rangeLabel = dialog.querySelector(".cs-range");
    const durationLabel = dialog.querySelector(".cs-duration");
    const setStageStatus = (message, progress = null, visible = true) => {
        if (!stageStatus) return;
        stageStatus.hidden = !visible;
        if (stageStatusText) stageStatusText.textContent = String(message || "");
        if (stageProgressBar) {
            const value = progress == null ? 100 : clamp(number(progress, 0), 0, 100);
            stageProgressBar.style.width = `${value}%`;
        }
    };
    const currentValues = {
        width: dimensionRequest(widget(node, "width")?.value, int(widget(node, "multiple")?.value, DEFAULT_MULTIPLE)),
        height: dimensionRequest(widget(node, "height")?.value, int(widget(node, "multiple")?.value, DEFAULT_MULTIPLE)),
        multiple: Math.max(1, int(widget(node, "multiple")?.value, DEFAULT_MULTIPLE)),
        fit_mode: String(widget(node, "fit_mode")?.value || "letterbox"),
        fill_color: hex(widget(node, "fill_color")?.value, DEFAULT_FILL),
        in_frame: int(widget(node, "in_frame")?.value, 0),
        out_frame: int(widget(node, "out_frame")?.value, -1),
        shot_detect_threshold: number(widget(node, "shot_detect_threshold")?.value, 0.5),
        shot_detect_min_scene_sec: number(widget(node, "shot_detect_min_scene_sec")?.value, 0),
    };
    let source = null;
    let state = null;
    let fps = DEFAULT_FPS;
    let frames = 1;
    let frame = 0;
    let selectedId = null;
    let proxy = null;
    let cacheKey = "";
    let proxyStartFrame = 0;
    let previewTimer = null;
    let previewRequest = 0;
    // ``stageFrame`` is an asynchronous exact-frame overlay.  Keep track of
    // which timeline frame it actually contains and which frame is currently
    // waiting for a response; otherwise the pause event can briefly reveal a
    // stale (usually In-frame) image over the final proxy frame.
    let previewPendingFrame = null;
    let stateTimer = null;
    let closed = false;
    let playingSelection = false;
    let directSourceMode = false;
    let frameExact = false;
    let proxyRequestGeneration = 0;
    let drag = null;
    let initialState = null;
    let applied = false;
    let historyReady = false;
    let historyBusy = false;
    let historyQueue = Promise.resolve(null);
    let timelineZoom = 1;
    let timelineViewStart = 0;
    let timelineViewDuration = 1;
    let timelinePan = null;

    const controls = {
        in: dialog.querySelector(".cs-in"), out: dialog.querySelector(".cs-out"),
        width: dialog.querySelector(".cs-width"), height: dialog.querySelector(".cs-height"), multiple: dialog.querySelector(".cs-multiple"),
        fit: dialog.querySelector(".cs-fit"), fillColor: dialog.querySelector(".cs-fill-color"), fillText: dialog.querySelector(".cs-fill-text"),
        threshold: dialog.querySelector(".cs-threshold"), minScene: dialog.querySelector(".cs-min-scene"),
        scaleX: dialog.querySelector(".cs-scale-x"), scaleY: dialog.querySelector(".cs-scale-y"), scaleXValue: dialog.querySelector(".cs-scale-x-value"), scaleYValue: dialog.querySelector(".cs-scale-y-value"), rotation: dialog.querySelector(".cs-rotation"), rotationValue: dialog.querySelector(".cs-rotation-value"), translateX: dialog.querySelector(".cs-translate-x"), translateXValue: dialog.querySelector(".cs-translate-x-value"), translateY: dialog.querySelector(".cs-translate-y"), translateYValue: dialog.querySelector(".cs-translate-y-value"), mirror: dialog.querySelector(".cs-mirror"), syncScale: dialog.querySelector(".cs-time-edit-sync"),
        undo: dialog.querySelector(".cs-undo"), redo: dialog.querySelector(".cs-redo"),
    };

    function selectedClip() { return state?.clips?.find((clip) => clip.id === selectedId) || state?.audio_clips?.find((clip) => clip.id === selectedId) || null; }
    function collectionForClip(clip) {
        // Audio-only items may arrive in either collection in hand-authored
        // descriptors.  Use object membership instead of video_track so split
        // and merge never splice the wrong array.
        if (state?.audio_clips?.includes(clip)) return state.audio_clips;
        return state?.clips || [];
    }
    function durationFrames() { return Math.max(1, int(state?.duration_frames, frames)); }
    function setStatus(message) { status.textContent = String(message || ""); }
    function updateStageAspect() {
        const sourceDimensions = sourceCanvasDimensions(source?.info);
        if (!sourceDimensions.width || !sourceDimensions.height) return;
        const requestedWidth = Math.max(0, int(controls.width.value, 0));
        const requestedHeight = Math.max(0, int(controls.height.value, 0));
        const multiple = Math.max(1, int(controls.multiple.value, DEFAULT_MULTIPLE));
        const width = requestedWidth || (requestedHeight ? Math.round(requestedHeight * sourceDimensions.width / sourceDimensions.height) : sourceDimensions.width);
        const height = requestedHeight || (requestedWidth ? Math.round(requestedWidth * sourceDimensions.height / sourceDimensions.width) : sourceDimensions.height);
        stage.style.aspectRatio = `${ceilMultiple(width, multiple)}/${ceilMultiple(height, multiple)}`;
    }
    function setFrame(next, seek = true, preview = true) {
        frame = clamp(int(next, 0), 0, Math.max(0, durationFrames() - 1));
        // The playback proxy is rendered only for the current In/Out range.
        // Seeking that short media element while scrubbing outside the range
        // makes the browser clamp currentTime back to the proxy edge and its
        // timeupdate event would pull the playhead back as well. Exact-frame
        // preview is requested independently, so only seek a video element
        // while transport playback is actually active (or when showing the
        // complete direct source).
        if (seek && stageVideo.readyState >= 1 && fps > 0 && (playingSelection || directSourceMode)) {
            try { stageVideo.currentTime = Math.max(0, frame - proxyStartFrame) / fps; } catch (_) { /* media may still be loading */ }
        }
        currentLabel.textContent = formatTime(frame, fps);
        durationLabel.textContent = formatTime(durationFrames(), fps);
        controls.in.value = String(state?.in_frame ?? 0);
        controls.out.value = String(state?.out_frame ?? durationFrames());
        rangeLabel.textContent = `In ${formatTime(state?.in_frame ?? 0, fps)} · Out ${formatTime(state?.out_frame ?? durationFrames(), fps)} · Frame ${frame}`;
        pointer.style.left = `${clamp(frame / Math.max(1, durationFrames() - 1), 0, 1) * 100}%`;
        updateLocalTransform();
        if (preview) scheduleFramePreview();
    }

    function invalidatePlaybackProxy() {
        proxyRequestGeneration += 1;
        if (playingSelection) {
            playingSelection = false;
            stageVideo.pause();
        }
    }

    function stopTransportForScrub() {
        if (playingSelection || !stageVideo.paused) {
            invalidatePlaybackProxy();
            stageVideo.pause();
        }
    }

    function scrubToFrame(next) {
        stopTransportForScrub();
        setFrame(next);
    }

    function scheduleStateSave() {
        if (stateTimer) clearTimeout(stateTimer);
        stateTimer = setTimeout(() => { stateTimer = null; void saveTimelineState(node, state); }, STATE_DEBOUNCE_MS);
    }

    function markChanged(message = "Unsaved timeline changes", options = {}) {
        const preview = options.preview !== false;
        const persist = options.persist !== false;
        // Any edit invalidates a proxy job or currently playing proxy.  Without
        // this generation bump, a job started before a drag could finish and
        // replace the stage with a stale composition.
        invalidatePlaybackProxy();
        const priorDuration = Math.max(1, int(state.duration_frames, 1));
        const followedTimelineEnd = int(state.out_frame, priorDuration) >= priorDuration;
        const activeEnds = state.clips.concat(state.audio_clips).map((clip) => clip.timeline_end);
        state.duration_frames = activeEnds.length ? Math.max(1, ...activeEnds) : priorDuration;
        state.in_frame = clamp(int(state.in_frame, 0), 0, state.duration_frames > 0 ? state.duration_frames - 1 : 0);
        state.out_frame = followedTimelineEnd
            ? state.duration_frames
            : clamp(int(state.out_frame, state.duration_frames), state.in_frame + (state.duration_frames > state.in_frame ? 1 : 0), state.duration_frames);
        if (state.out_frame <= state.in_frame) state.out_frame = state.duration_frames;
        setStatus(message);
        renderTimeline();
        updateSelectedPanel();
        setFrame(frame, false, preview);
        if (persist) scheduleStateSave();
    }

    function updateHistoryButtons(result = {}) {
        if (controls.undo) controls.undo.disabled = historyBusy || !result.can_undo;
        if (controls.redo) controls.redo.disabled = historyBusy || !result.can_redo;
    }

    async function recordEditableHistory(label) {
        if (!state || !historyReady || historyBusy) return;
        const snapshot = editableTimelineSnapshot(state);
        historyQueue = historyQueue.then(() => timelineHistoryRequest(node, "record", snapshot, label));
        const result = await historyQueue;
        if (!closed && result) updateHistoryButtons(result);
    }

    function applyHistorySnapshot(snapshot) {
        if (!state || !snapshot || typeof snapshot !== "object") return false;
        const preserved = { in_frame: state.in_frame, out_frame: state.out_frame, threshold: state.threshold, min_scene_seconds: state.min_scene_seconds };
        state = normalizeState({ ...snapshot, in_frame: 0, out_frame: -1, threshold: preserved.threshold, min_scene_seconds: preserved.min_scene_seconds }, frames, fps, preserved.threshold, preserved.min_scene_seconds);
        state.in_frame = clamp(preserved.in_frame, 0, Math.max(0, state.duration_frames - 1));
        state.out_frame = clamp(preserved.out_frame, state.in_frame + 1, state.duration_frames);
        state.fps = fps;
        setOutputControls(state.output);
        selectedId = state.clips.find((clip) => clip.id === selectedId)?.id || state.audio_clips.find((clip) => clip.id === selectedId)?.id || state.clips[0]?.id || state.audio_clips[0]?.id || null;
        markChanged("History step applied");
        return true;
    }

    async function stepHistory(action) {
        if (!state || !historyReady || historyBusy) return;
        historyBusy = true;
        updateHistoryButtons({});
        try {
            await historyQueue;
            const result = await timelineHistoryRequest(node, action);
            if (result?.timeline) applyHistorySnapshot(result.timeline);
            if (result) { controls.undo.disabled = !result.can_undo; controls.redo.disabled = !result.can_redo; }
        } finally {
            historyBusy = false;
        }
    }

    function timelineXToFrame(clientX) {
        const body = inner.querySelector(".cs-time-edit-track-body");
        const rect = body?.getBoundingClientRect();
        if (!rect || !rect.width) return frame;
        return clamp(Math.round(timelineViewStart + (clientX - rect.left) / rect.width * timelineViewDuration), 0, durationFrames());
    }

    function clampTimelineView() {
        const duration = durationFrames();
        timelineZoom = Math.max(1, number(timelineZoom, 1));
        timelineViewDuration = timelineZoom <= 1 ? duration : clamp(duration / timelineZoom, 1, duration);
        timelineViewStart = clamp(number(timelineViewStart, 0), 0, Math.max(0, duration - timelineViewDuration));
    }

    function updateZoomState() {
        if (!zoomStateLabel) return;
        zoomStateLabel.textContent = timelineZoom <= 1.0001 ? "Fit" : `${timelineZoom.toFixed(2)}×`;
        const zoomOut = dialog.querySelector(".cs-time-edit-zoom-out");
        if (zoomOut) zoomOut.disabled = timelineZoom <= 1.0001;
    }

    function fitTimeline() {
        timelineZoom = 1;
        timelineViewStart = 0;
        clampTimelineView();
        renderTimeline();
    }

    function zoomTimeline(direction, centerFrame = frame) {
        if (!state) return;
        const oldDuration = Math.max(1, timelineViewDuration || durationFrames());
        const oldStart = timelineViewStart;
        const centerRatio = clamp((number(centerFrame, frame) - oldStart) / oldDuration, 0, 1);
        const nextZoom = direction > 0 ? Math.min(32, timelineZoom * TIMELINE_ZOOM_FACTOR) : Math.max(1, timelineZoom / TIMELINE_ZOOM_FACTOR);
        if (direction < 0 && timelineZoom <= 1.0001) return;
        timelineZoom = nextZoom;
        clampTimelineView();
        timelineViewStart = clamp(number(centerFrame, frame) - centerRatio * timelineViewDuration, 0, Math.max(0, durationFrames() - timelineViewDuration));
        renderTimeline();
    }

    function trackAtPoint(clientX, clientY, kind) {
        // Pointer events often bubble from the clip itself (or from the
        // document while the pointer is outside the original row), so using
        // event.target.closest() is not sufficient for cross-track drags.
        return Array.from(inner.querySelectorAll(`.cs-time-edit-track[data-kind="${kind}"]`)).find((row) => {
            const rect = row.getBoundingClientRect();
            return clientY >= rect.top && clientY <= rect.bottom && clientX >= rect.left - 20 && clientX <= rect.right + 20;
        }) || null;
    }

    function renderTimeline() {
        if (!state) return;
        const duration = durationFrames();
        clampTimelineView();
        durationLabel.textContent = formatTime(duration, fps);
        const width = Math.max(1, timelineViewport?.clientWidth || 760);
        inner.style.width = `${width}px`;
        const bodyWidth = Math.max(1, width - 100);
        axis.style.left = "100px";
        axis.style.width = `${bodyWidth}px`;
        pointerRow.style.marginLeft = "100px";
        pointerRow.style.width = `${bodyWidth}px`;
        axis.innerHTML = "";
        const ticks = Math.max(2, Math.min(20, Math.round(width / 90)));
        for (let index = 0; index <= ticks; index += 1) {
            const mark = document.createElement("span");
            mark.style.left = `${(index / ticks) * 100}%`;
            mark.textContent = formatTime(Math.round(timelineViewStart + timelineViewDuration * index / ticks), fps);
            axis.append(mark);
        }
        const visibleIn = clamp(state.in_frame, timelineViewStart, timelineViewStart + timelineViewDuration);
        const visibleOut = clamp(state.out_frame, timelineViewStart, timelineViewStart + timelineViewDuration);
        rangeBand.style.left = `${100 + bodyWidth * ((visibleIn - timelineViewStart) / timelineViewDuration)}px`;
        rangeBand.style.width = `${Math.max(0, bodyWidth * ((visibleOut - visibleIn) / timelineViewDuration))}px`;
        pointer.style.left = `${clamp((frame - timelineViewStart) / timelineViewDuration, 0, 1) * 100}%`;
        pointer.style.opacity = frame < timelineViewStart || frame > timelineViewStart + timelineViewDuration ? "0.45" : "1";
        updateZoomState();
        inner.querySelectorAll(".cs-time-edit-clip").forEach((item) => item.remove());
        const rows = Array.from(inner.querySelectorAll(".cs-time-edit-track"));
        const draw = (clip, kind, track) => {
            const row = rows.find((candidate) => candidate.dataset.kind === kind && Number(candidate.dataset.track) === Number(track));
            if (!row || !clip.enabled) return;
            const body = row.querySelector(".cs-time-edit-track-body");
            const item = document.createElement("div");
            item.className = `cs-time-edit-clip ${kind}${Number(clip.video_track) === 1 ? " upper" : ""}${clip.id === selectedId ? " selected" : ""}`;
            item.dataset.clipId = clip.id;
            item.title = `${clip.id} · source ${clip.source_start}–${clip.source_end} · timeline ${clip.timeline_start}–${clip.timeline_end}`;
            item.style.left = `${((clip.timeline_start - timelineViewStart) / timelineViewDuration) * 100}%`;
            item.style.width = `${Math.max(0.3, (clip.timeline_end - clip.timeline_start) / timelineViewDuration * 100)}%`;
            item.textContent = kind === "audio" ? `♪ ${clip.id}` : clip.id;
            const left = document.createElement("span"); left.className = "cs-time-edit-edge left";
            const right = document.createElement("span"); right.className = "cs-time-edit-edge right";
            item.prepend(left); item.append(right); body.append(item);
            item.addEventListener("pointerdown", (event) => beginClipDrag(event, clip, kind, track, item));
            item.addEventListener("click", (event) => { event.stopPropagation(); selectedId = clip.id; renderTimeline(); updateSelectedPanel(); scheduleFramePreview(); });
        };
        state.clips.forEach((clip) => { draw(clip, "video", clip.video_track); if (clip.audio_track >= 0) draw(clip, "audio", clip.audio_track); });
        state.audio_clips.forEach((clip) => { if (clip.audio_track >= 0) draw(clip, "audio", clip.audio_track); });
        rows.forEach((row) => {
            const body = row.querySelector(".cs-time-edit-track-body");
            if (!body) return;
            body.onpointerdown = (event) => {
                if (event.button !== 0 || event.target.closest(".cs-time-edit-clip")) return;
                const startX = event.clientX;
                const startFrame = timelineXToFrame(startX);
                const startView = timelineViewStart;
                const rect = body.getBoundingClientRect();
                let moved = false;
                const move = (moveEvent) => {
                    if (timelineZoom <= 1.0001) return;
                    const deltaPixels = moveEvent.clientX - startX;
                    if (Math.abs(deltaPixels) > 3) moved = true;
                    if (!moved || !rect.width) return;
                    timelineViewStart = clamp(startView - (deltaPixels / rect.width) * timelineViewDuration, 0, Math.max(0, durationFrames() - timelineViewDuration));
                    renderTimeline();
                };
                const up = () => {
                    window.removeEventListener("pointermove", move);
                    window.removeEventListener("pointerup", up);
                    window.removeEventListener("pointercancel", up);
                    timelinePan = null;
                    if (!moved) scrubToFrame(startFrame);
                };
                timelinePan = { startX, startFrame };
                window.addEventListener("pointermove", move);
                window.addEventListener("pointerup", up);
                window.addEventListener("pointercancel", up);
            };
        });
    }

    function isTimelineStartClip(clip) {
        return Boolean(clip && clip.enabled !== false && int(clip.timeline_start, 0) === 0);
    }

    function updateLocalTransform() {
        // The frame endpoint renders the complete two-track composition with
        // the selected affine transform. Applying CSS on top would transform
        // the whole composite a second time, so only use the local transform
        // while an exact frame is not available.
        if (frameExact) { stageFrame.classList.remove("local-transform"); stageFrame.style.transform = ""; return; }
        const clip = activeClip(state || {}, frame);
        const selected = selectedClip();
        const target = selected && selected.timeline_start <= frame && frame < selected.timeline_end ? selected : clip;
        if (!target) { stageFrame.classList.remove("local-transform"); stageFrame.style.transform = ""; return; }
        stageFrame.classList.add("local-transform");
        stageFrame.style.transform = transformCss(target.transform, stage);
    }

    function updateSelectedPanel() {
        const clip = selectedClip();
        const label = dialog.querySelector(".cs-selected-label");
        if (!clip) {
            label.textContent = "(none)";
            [controls.scaleX, controls.scaleY, controls.scaleXValue, controls.scaleYValue].forEach((input) => { if (input) input.value = "1"; });
            [controls.rotation, controls.translateX, controls.translateY, controls.rotationValue, controls.translateXValue, controls.translateYValue].forEach((input) => { if (input) input.value = "0"; });
            if (controls.mirror) controls.mirror.value = "none";
            return;
        }
        label.textContent = clip.id;
        const transform = normalizeTransform(clip.transform);
        controls.scaleX.value = String(transform.scale_x); controls.scaleY.value = String(transform.scale_y); controls.rotation.value = String(transform.rotation);
        controls.translateX.value = String(transform.translate_x); controls.translateY.value = String(transform.translate_y);
        updateTransformControlDisplay();
        if (controls.mirror) controls.mirror.value = transform.flip_x ? "horizontal" : transform.flip_y ? "vertical" : "none";
    }

    function beginClipDrag(event, clip, kind, track, item) {
        if (event.button !== 0) return;
        event.preventDefault(); event.stopPropagation();
        selectedId = clip.id;
        const edge = event.target.closest(".cs-time-edit-edge")?.classList;
        const mode = edge?.contains("left") ? "left" : edge?.contains("right") ? "right" : "move";
        drag = {
            clip, kind, track, mode,
            startX: event.clientX,
            startY: event.clientY,
            sourceStart: clip.source_start,
            sourceEnd: clip.source_end,
            timelineStart: clip.timeline_start,
            timelineEnd: clip.timeline_end,
            videoTrack: clip.video_track,
            audioTrack: clip.audio_track,
            duration: timelineViewDuration,
            width: Math.max(1, inner.querySelector(".cs-time-edit-track-body")?.clientWidth || inner.clientWidth),
            item,
            frame: int(frame, 0),
            inFrame: int(state.in_frame, 0),
            outFrame: int(state.out_frame, durationFrames()),
            clips: state.clips.slice(),
            audioClips: state.audio_clips.slice(),
            timelineItems: state.clips.concat(state.audio_clips).map((item) => ({
                item,
                timelineStart: int(item.timeline_start, 0),
                timelineEnd: int(item.timeline_end, 0),
                sourceStart: int(item.source_start, 0),
                sourceEnd: int(item.source_end, 0),
            })),
            rebaseTimeline: mode === "left" && isTimelineStartClip(clip),
        };
        item.classList.add("dragging");
        ACTIVE_DRAG.set(dialog, drag);
        const move = (moveEvent) => {
            if (!drag || drag.clip !== clip) return;
            const delta = Math.round((moveEvent.clientX - drag.startX) / drag.width * drag.duration);
            if (mode === "move") {
                const nextStart = int(Math.max(0, drag.timelineStart + delta), 0);
                const length = drag.timelineEnd - drag.timelineStart;
                // Moving a clip may extend the timeline; duration is defined
                // as the maximum clip end, so do not clamp to the old end.
                clip.timeline_start = Math.max(0, nextStart);
                clip.timeline_end = clip.timeline_start + length;
                if (kind === "video") {
                    const row = trackAtPoint(moveEvent.clientX, moveEvent.clientY, "video");
                    const nextTrack = row?.dataset.kind === "video" ? int(row.dataset.track, clip.video_track) : clip.video_track;
                    clip.video_track = clamp(nextTrack, 0, 1);
                    if (clip.linked && clip.audio_track >= 0) clip.audio_track = clip.video_track;
                } else {
                    const row = trackAtPoint(moveEvent.clientX, moveEvent.clientY, "audio");
                    const nextTrack = row?.dataset.kind === "audio" ? int(row.dataset.track, clip.audio_track < 0 ? 0 : clip.audio_track) : (clip.audio_track < 0 ? 0 : clip.audio_track);
                    const normalizedTrack = clamp(nextTrack, 0, 1);
                    // Video-to-audio linking is driven by moving the video
                    // clip.  Moving the audio representation independently
                    // intentionally unlinks it instead of unexpectedly
                    // moving the picture to another video track.
                    if (clip.linked && clip.audio_track !== normalizedTrack) clip.linked = false;
                    clip.audio_track = normalizedTrack;
                }
            } else if (mode === "left") {
                const next = clamp(int(drag.timelineStart + delta, 0), 0, drag.timelineEnd - 1);
                const sourceDelta = next - drag.timelineStart;
                const sourceStart = drag.sourceStart + sourceDelta;
                if (sourceStart >= 0 && sourceStart < drag.sourceEnd) {
                    if (drag.rebaseTimeline) {
                        // Trimming the first clip must remove the discarded
                        // leading range from the whole timeline.  Rebase every
                        // clip, playhead and In/Out point by the same amount so
                        // no empty region is left before the new first frame.
                        const shift = next;
                        const removed = new Set();
                        state.clips = drag.clips.slice();
                        state.audio_clips = drag.audioClips.slice();
                        drag.timelineItems.forEach(({ item, timelineStart, timelineEnd, sourceStart: itemSourceStart, sourceEnd }) => {
                            if (item !== clip && timelineEnd <= shift) {
                                removed.add(item);
                                return;
                            }
                            const retainedStart = Math.max(timelineStart, shift);
                            const trimmedFrames = retainedStart - timelineStart;
                            item.timeline_start = retainedStart - shift;
                            item.timeline_end = timelineEnd - shift;
                            item.source_start = itemSourceStart + trimmedFrames;
                            item.source_end = sourceEnd;
                            if (item.timeline_end <= item.timeline_start || item.source_start >= item.source_end) removed.add(item);
                        });
                        state.clips = state.clips.filter((item) => !removed.has(item));
                        state.audio_clips = state.audio_clips.filter((item) => !removed.has(item));
                        frame = Math.max(0, drag.frame - shift);
                        state.in_frame = Math.max(0, drag.inFrame - shift);
                        state.out_frame = Math.max(state.in_frame + 1, drag.outFrame - shift);
                        clip.timeline_start = 0;
                    } else {
                        clip.timeline_start = next;
                    }
                    clip.source_start = sourceStart;
                }
            } else {
                // The timeline grows with the maximum clip end.  Clamp only
                // against the remaining source frames, not the old timeline
                // duration, so the final clip can be extended into new blank
                // time until its source range is exhausted.
                const maxTimelineEnd = drag.timelineStart + (frames - drag.sourceStart);
                const rawNext = drag.timelineEnd + delta;
                // Do not snap an extension back to the old duration marker;
                // that marker is only a useful target while the pointer is
                // still inside the existing timeline.
                const snappedNext = rawNext;
                const next = clamp(snappedNext, drag.timelineStart + 1, maxTimelineEnd);
                const sourceEnd = drag.sourceEnd + (next - drag.timelineEnd);
                if (sourceEnd > drag.sourceStart && sourceEnd <= frames) { clip.timeline_end = next; clip.source_end = sourceEnd; }
            }
            markChanged("Dragging clip…");
        };
        const up = () => {
            window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); window.removeEventListener("pointercancel", up);
            item.classList.remove("dragging");
            // The backend resolves same-track overlaps by placement order.
            // Bump the order when a clip is moved so equal-start overlaps have
            // the same winner in the proxy and in the final render.
            if (mode === "move" && (kind === "video" || clip.video_track < 0)) clip.order = state._nextOrder++;
            drag = null; ACTIVE_DRAG.delete(dialog); markChanged("Clip edit ready"); void recordEditableHistory(mode === "move" ? "Move or trim clip" : "Trim clip");
        };
        window.addEventListener("pointermove", move); window.addEventListener("pointerup", up); window.addEventListener("pointercancel", up);
    }

    function splitClipAt(clip, splitFrame) {
        if (!clip || splitFrame <= clip.timeline_start || splitFrame >= clip.timeline_end) { setStatus("Place the split frame inside a clip."); return false; }
        const offset = splitFrame - clip.timeline_start;
        const first = { ...clip, id: idFor(1), source_end: clip.source_start + offset, timeline_end: splitFrame, order: state._nextOrder++ };
        const second = { ...clip, id: idFor(2), source_start: clip.source_start + offset, timeline_start: splitFrame, order: state._nextOrder++ };
        const collection = collectionForClip(clip);
        collection.splice(collection.indexOf(clip), 1, first, second);
        selectedId = second.id; markChanged("Clip split"); void recordEditableHistory("Split clip"); return true;
    }

    function splitSelected() { const clip = selectedClip(); splitClipAt(clip, frame); }

    function mergeClipToNext(clip) {
        if (!clip) { setStatus("Select a clip to merge."); return; }
        const collection = collectionForClip(clip);
        const next = collection.filter((item) => item !== clip && item.timeline_start >= clip.timeline_end && (clip.video_track < 0 ? item.audio_track === clip.audio_track : (item.video_track === clip.video_track && item.audio_track === clip.audio_track))).sort((a, b) => a.timeline_start - b.timeline_start)[0];
        const other = next;
        const first = clip;
        const second = next;
        if (!other || clip.timeline_end !== other.timeline_start || clip.source_end !== other.source_start) { setStatus("The next clip must be adjacent with contiguous source frames."); return; }
        if (canonicalJson(normalizeTransform(clip.transform)) !== canonicalJson(normalizeTransform(other.transform))) {
            setStatus("Reset both transforms before merging clips with different transforms.");
            return;
        }
        const merged = {
            ...first,
            id: idFor(3),
            source_start: first.source_start,
            source_end: second.source_end,
            timeline_start: first.timeline_start,
            timeline_end: second.timeline_end,
            order: state._nextOrder++,
        };
        if (collection === state.clips) {
            state.clips = state.clips.filter((item) => item !== clip && item !== other);
            state.clips.push(merged);
        } else {
            state.audio_clips = state.audio_clips.filter((item) => item !== clip && item !== other);
            state.audio_clips.push({ ...merged, video_track: -1 });
        }
        selectedId = merged.id; markChanged("Adjacent clips merged"); void recordEditableHistory("Merge clip to next");
    }

    function mergeSelected() { mergeClipToNext(selectedClip()); }

    function deleteSelected() {
        const clip = selectedClip();
        if (!clip) return;
        state.clips = state.clips.filter((item) => item !== clip); state.audio_clips = state.audio_clips.filter((item) => item !== clip);
        if (!state.clips.length && !state.audio_clips.length) {
            // Keep an explicit disabled sentinel so the backend does not
            // interpret an empty list as “restore the default full-source
            // clip”. The timeline then renders its configured fill color.
            state.clips.push({ id: "empty-sentinel", source_start: 0, source_end: Math.min(1, frames), timeline_start: 0, timeline_end: Math.max(1, state.duration_frames), video_track: 0, audio_track: -1, linked: false, transform: normalizeTransform({}), enabled: false, order: state._nextOrder++ });
        }
        selectedId = null; markChanged("Clip deleted"); void recordEditableHistory("Delete clip");
    }

    function resetTransform() {
        const clip = selectedClip(); if (!clip) return;
        clip.transform = normalizeTransform({}); markChanged("Transform reset"); void recordEditableHistory("Reset clip transform");
    }

    function updateTransformControlDisplay() {
        const pairs = [[controls.scaleX, controls.scaleXValue], [controls.scaleY, controls.scaleYValue], [controls.rotation, controls.rotationValue], [controls.translateX, controls.translateXValue], [controls.translateY, controls.translateYValue]];
        pairs.forEach(([range, value]) => { if (range && value) value.value = String(range.value); });
    }

    function applyTransformControls(record = false, changed = "", preview = true) {
        const clip = selectedClip(); if (!clip) return;
        if (changed === "scaleX") controls.scaleXValue.value = controls.scaleX.value;
        if (changed === "scaleXValue") controls.scaleX.value = controls.scaleXValue.value;
        if (changed === "scaleY") controls.scaleYValue.value = controls.scaleY.value;
        if (changed === "scaleYValue") controls.scaleY.value = controls.scaleYValue.value;
        if (changed === "rotation") controls.rotationValue.value = controls.rotation.value;
        if (changed === "rotationValue") controls.rotation.value = controls.rotationValue.value;
        if (changed === "translateX") controls.translateXValue.value = controls.translateX.value;
        if (changed === "translateXValue") controls.translateX.value = controls.translateXValue.value;
        if (changed === "translateY") controls.translateYValue.value = controls.translateY.value;
        if (changed === "translateYValue") controls.translateY.value = controls.translateYValue.value;
        if ((changed === "scaleX" || changed === "scaleXValue" || changed === "scaleY" || changed === "scaleYValue") && controls.syncScale?.getAttribute("aria-pressed") === "true") {
            const changedInput = controls[changed] || (changed.startsWith("scaleY") ? controls.scaleY : controls.scaleX);
            const next = clamp(number(changedInput?.value, 1), 0.05, 20);
            controls.scaleX.value = String(next); controls.scaleY.value = String(next); controls.scaleXValue.value = String(next); controls.scaleYValue.value = String(next);
        }
        const scaleX = clamp(number(controls.scaleXValue.value, number(controls.scaleX.value, 1)), 0.1, 4);
        const scaleY = clamp(number(controls.scaleYValue.value, number(controls.scaleY.value, 1)), 0.1, 4);
        const rotation = clamp(number(controls.rotationValue.value, number(controls.rotation.value, 0)), -90, 90);
        const translateX = clamp(number(controls.translateXValue.value, number(controls.translateX.value, 0)), -1, 1);
        const translateY = clamp(number(controls.translateYValue.value, number(controls.translateY.value, 0)), -1, 1);
        controls.scaleX.value = String(scaleX); controls.scaleY.value = String(scaleY); controls.scaleXValue.value = String(scaleX); controls.scaleYValue.value = String(scaleY);
        controls.rotation.value = String(rotation); controls.rotationValue.value = String(rotation); controls.translateX.value = String(translateX); controls.translateXValue.value = String(translateX); controls.translateY.value = String(translateY); controls.translateYValue.value = String(translateY);
        const mirror = String(controls.mirror?.value || "none");
        clip.transform = normalizeTransform({ scale_x: scaleX, scale_y: scaleY, rotation, translate_x: translateX, translate_y: translateY, translation_unit: "normalized", flip_x: mirror === "horizontal", flip_y: mirror === "vertical" });
        markChanged("Selected clip updated", { preview, persist: record });
        if (record) void recordEditableHistory("Adjust clip transform");
    }

    function setIn() { if (!state) return; state.in_frame = clamp(frame, 0, Math.max(0, state.out_frame - 1)); markChanged("In point set"); }
    function setOut() { if (!state) return; state.out_frame = clamp(frame + 1, Math.min(state.duration_frames, state.in_frame + 1), state.duration_frames); markChanged("Out point set"); }

    function beginRangeDrag(mode, event) {
        if (!state) return;
        event.preventDefault(); event.stopPropagation();
        const rect = inner.querySelector(".cs-time-edit-track-body")?.getBoundingClientRect() || inner.getBoundingClientRect();
        const move = (moveEvent) => {
            const next = clamp(Math.round(timelineViewStart + clamp((moveEvent.clientX - rect.left) / Math.max(1, rect.width), 0, 1) * timelineViewDuration), 0, durationFrames());
            if (mode === "in") state.in_frame = clamp(next, 0, state.out_frame - 1); else state.out_frame = clamp(next, state.in_frame + 1, durationFrames());
            markChanged(mode === "in" ? "In point moved" : "Out point moved");
        };
        const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); window.removeEventListener("pointercancel", up); };
        window.addEventListener("pointermove", move); window.addEventListener("pointerup", up); window.addEventListener("pointercancel", up); move(event);
    }

    function beginPlayheadDrag(event) {
        if (!state) return;
        event.preventDefault();
        // A scrub must always address the full timeline, even after proxy
        // playback was started. Stop transport first so its short In/Out
        // media element cannot race the pointer and clamp it back in range.
        stopTransportForScrub();
        const rect = inner.querySelector(".cs-time-edit-track-body")?.getBoundingClientRect() || inner.getBoundingClientRect();
        const move = (moveEvent) => setFrame(Math.round(timelineViewStart + clamp((moveEvent.clientX - rect.left) / Math.max(1, rect.width), 0, 1) * Math.max(0, timelineViewDuration - 1)));
        const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); window.removeEventListener("pointercancel", up); };
        window.addEventListener("pointermove", move); window.addEventListener("pointerup", up); window.addEventListener("pointercancel", up); move(event);
    }

    function clipAtTimelinePosition(event, atFrame) {
        const item = event.target.closest(".cs-time-edit-clip");
        if (item?.dataset.clipId) return state.clips.concat(state.audio_clips).find((clip) => clip.id === item.dataset.clipId) || null;
        const row = event.target.closest(".cs-time-edit-track");
        if (!row) return activeClip(state, atFrame);
        const kind = String(row.dataset.kind || "video");
        const track = int(row.dataset.track, 0);
        return state.clips.concat(state.audio_clips).filter((clip) => clip.enabled && clip.timeline_start <= atFrame && atFrame < clip.timeline_end && (kind === "video" ? clip.video_track === track : clip.audio_track === track)).sort((left, right) => right.order - left.order)[0] || null;
    }

    function closeContextMenu() {
        const menu = dialog.querySelector(".cs-time-edit-context-menu");
        if (menu) { menu.hidden = true; menu.innerHTML = ""; }
    }

    function showContextMenu(event) {
        if (!state || !event.target.closest(".cs-time-edit-timeline")) return;
        event.preventDefault();
        const atFrame = timelineXToFrame(event.clientX);
        const clip = clipAtTimelinePosition(event, atFrame);
        if (clip) selectedId = clip.id;
        scrubToFrame(Math.min(atFrame, durationFrames() - 1));
        renderTimeline(); updateSelectedPanel();
        const menu = dialog.querySelector(".cs-time-edit-context-menu");
        const items = [
            { label: "Split at this frame", enabled: Boolean(clip && atFrame > clip.timeline_start && atFrame < clip.timeline_end), action: () => splitClipAt(clip, atFrame) },
            { label: "Merge Clip to Next", enabled: Boolean(clip), action: () => mergeClipToNext(clip) },
            { label: "Delete Clip", enabled: Boolean(clip), action: () => { if (clip) { selectedId = clip.id; deleteSelected(); } } },
            { label: "Reset Clip Transform", enabled: Boolean(clip), action: () => { if (clip) { selectedId = clip.id; resetTransform(); } } },
        ];
        menu.innerHTML = "";
        items.forEach((definition) => {
            const button = document.createElement("button");
            button.type = "button"; button.textContent = definition.label; button.disabled = !definition.enabled;
            button.addEventListener("click", () => { closeContextMenu(); definition.action(); });
            menu.append(button);
        });
        menu.hidden = false;
        menu.style.left = `${Math.min(event.clientX, window.innerWidth - 280)}px`;
        menu.style.top = `${Math.min(event.clientY, window.innerHeight - 170)}px`;
    }

    async function detectShots() {
        const button = dialog.querySelector(".cs-shot");
        invalidatePlaybackProxy();
        button.disabled = true; setStatus("Detecting shots…"); setStageStatus("Shot detection in progress…", 0, true);
        let progress = 8;
        const progressTimer = window.setInterval(() => {
            progress = Math.min(92, progress + 4);
            setStageStatus(`Shot detection in progress… ${progress}%`, progress, true);
        }, 350);
        try {
            const response = await api.fetchApi("/cinestyle/video-time-edit-shot-detect", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    node_id: String(node.id ?? ""),
                    video_filename: source?.filename || "",
                    source_token: String(source?.cached?.token || ""),
                    source_identity: String(source?.info?.source_identity || source?.info?.source_fingerprint || source?.info?.input_signature || ""),
                    source_frames: frames,
                    source_start_frame: int(source?.startFrame, 0),
                    source_end_frame: int(source?.endFrame, -1),
                    source_target_fps: number(source?.targetFps, 0),
                    source_output_width: int(source?.outputWidth, 0),
                    source_output_height: int(source?.outputHeight, 0),
                    source_output_multiple: Math.max(1, int(source?.multiple, DEFAULT_MULTIPLE)),
                    source_cache_key: String(source?.sourceCacheKey || ""),
                    cache_key: cacheKey,
                    shot_detect_threshold: number(controls.threshold.value, state.threshold),
                    shot_detect_min_scene_sec: number(controls.minScene.value, state.min_scene_seconds),
                }),
            });
            const result = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(result.error || "Shot detection failed");
            const shots = Array.isArray(result.shots) ? result.shots : [];
            const normalizedShots = shots
                .map((shot) => ({
                    start: clamp(int(shot?.start ?? shot?.source_start, 0), 0, Math.max(0, frames - 1)),
                    end: clamp(int(shot?.end ?? shot?.source_end, 0), 0, frames),
                }))
                .filter((shot) => shot.end > shot.start)
                .sort((left, right) => left.start - right.start || left.end - right.end);
            if (!normalizedShots.length) throw new Error("No shots detected.");
            const detectedFps = infoFps(result); if (detectedFps > 0) fps = detectedFps;
            state.fps = fps;
            let cursor = 0;
            state.clips = normalizedShots.map((shot, index) => {
                const start = shot.start;
                const end = shot.end;
                const length = end - start;
                const clip = { id: `shot-${index + 1}`, source_start: start, source_end: end, timeline_start: cursor, timeline_end: cursor + length, video_track: 0, audio_track: 0, linked: true, transform: normalizeTransform({}), enabled: true, order: state._nextOrder++ };
                cursor += length;
                return clip;
            });
            state.audio_clips = []; state.duration_frames = cursor; state.in_frame = 0; state.out_frame = cursor; selectedId = state.clips[0]?.id || null;
            markChanged(`Detected ${normalizedShots.length} shots`); void recordEditableHistory("Detect and cut shots"); setStageStatus("Shot detection complete", 100, true); window.setTimeout(() => { if (!closed) setStageStatus("", null, false); }, 700);
        } catch (error) {
            setStatus(error?.message || "Shot detection failed"); setStageStatus("Shot detection failed", 0, true);
        } finally { window.clearInterval(progressTimer); button.disabled = false; }
    }

    async function renderFramePreview() {
        if (closed || !state) return;
        const requestId = ++previewRequest;
        const requestedFrame = frame;
        previewPendingFrame = requestedFrame;
        frameExact = false;
        stageFrame.style.transform = "";
        // Never leave the previous exact frame visible while this request is
        // in flight.  The video remains visible underneath, including when a
        // playback proxy has just been paused at the In/Out boundary.
        stageFrame.style.display = "none";
        const publicTimeline = publicState(state);
        const hasSource = Boolean(source?.cached?.token || source?.filename);
        // If no source token/path is available, let the backend serve the
        // last encoded proxy rather than forcing a re-render that cannot find
        // its source frames. Once a source token is available, send the full
        // descriptor so transforms and edits are rendered exactly.
        const previewPayload = {
            node_id: String(node.id ?? ""),
            frame,
            cache_key: cacheKey,
            source_cache_key: String(source?.sourceCacheKey || ""),
        };
        if (hasSource) Object.assign(previewPayload, {
            timeline: publicTimeline,
            timeline_json: canonicalJson(publicTimeline),
            source_token: String(source?.cached?.token || ""),
            video_filename: String(source?.filename || ""),
            source_identity: String(source?.info?.source_identity || source?.info?.source_fingerprint || source?.info?.input_signature || ""),
            source_fingerprint_kind: String(source?.info?.source_fingerprint_kind || ""),
            source_frames: frames,
            source_frame_count: frames,
            source_start_frame: int(source?.startFrame, 0),
            source_end_frame: int(source?.endFrame, -1),
            source_target_fps: number(source?.targetFps, 0),
            source_output_width: int(source?.outputWidth, 0),
            source_output_height: int(source?.outputHeight, 0),
            source_output_multiple: Math.max(1, int(source?.multiple, DEFAULT_MULTIPLE)),
            // Cache entries expose both the encoded proxy dimensions and the
            // dimensions of the VIDEO that the timeline is actually editing.
            // Prefer the latter; using ``width`` here would make an odd-size
            // source (or a loader cache downscaled to 1 MP) change the final
            // canvas aspect during exact-frame preview.
            source_width: sourceCanvasDimensions(source?.info).width,
            source_height: sourceCanvasDimensions(source?.info).height,
            width: dimensionRequest(controls.width.value, int(controls.multiple.value, DEFAULT_MULTIPLE)),
            height: dimensionRequest(controls.height.value, int(controls.multiple.value, DEFAULT_MULTIPLE)),
            multiple: Math.max(1, int(controls.multiple.value, DEFAULT_MULTIPLE)),
            fit_mode: String(controls.fit.value || "letterbox"),
            fill_color: hex(controls.fillText.value, DEFAULT_FILL),
        });
        const response = await api.fetchApi("/cinestyle/video-time-edit-preview", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(previewPayload),
        }).catch(() => null);
        if (!response || !response.ok || requestId !== previewRequest || requestedFrame !== frame || previewPendingFrame !== requestedFrame) {
            // Never leave an image from an earlier frame visible after a
            // failed/stale request; the transport source (if any) is the only
            // remaining trustworthy preview.
            if (requestId === previewRequest && previewPendingFrame === requestedFrame) {
                previewPendingFrame = null;
                stageFrame.style.display = "none";
            }
            return;
        }
        const contentType = String(response.headers.get("content-type") || "").toLowerCase();
        if (contentType.includes("application/json")) {
            const result = await response.json().catch(() => ({}));
            if (requestId !== previewRequest || requestedFrame !== frame || previewPendingFrame !== requestedFrame) return;
            const image = String(result.image || result.preview || result.url || "");
            if (image) {
                frameExact = true;
                previewPendingFrame = null;
                stageFrame.dataset.frame = String(requestedFrame);
                stageFrame.src = image.startsWith("http") || image.startsWith("data:") ? image : api.apiURL(image);
                stageFrame.style.display = stageVideo.paused && !playingSelection ? "block" : "none";
                setStageStatus("", null, false); updateLocalTransform();
            } else {
                previewPendingFrame = null;
                stageFrame.style.display = "none";
            }
            return;
        }
        const blob = await response.blob().catch(() => null);
        if (!blob || requestId !== previewRequest || requestedFrame !== frame || previewPendingFrame !== requestedFrame) {
            if (!blob && requestId === previewRequest && previewPendingFrame === requestedFrame) {
                previewPendingFrame = null;
                stageFrame.style.display = "none";
            }
            return;
        }
        const url = URL.createObjectURL(blob);
        const old = stageFrame.dataset.objectUrl;
        if (old) URL.revokeObjectURL(old);
        frameExact = true;
        previewPendingFrame = null;
        stageFrame.dataset.frame = String(requestedFrame);
        stageFrame.dataset.objectUrl = url;
        stageFrame.src = url;
        stageFrame.style.display = stageVideo.paused && !playingSelection ? "block" : "none";
        setStageStatus("", null, false); updateLocalTransform();
    }

    function scheduleFramePreview() {
        if (previewTimer) clearTimeout(previewTimer);
        // Invalidate any request already in flight even when the numeric frame
        // stays the same (for example after a transform/output edit).  Frame
        // equality alone is not enough to prove that an old response still
        // represents the current composition.
        ++previewRequest;
        // Mark the target as pending immediately, not only when the debounce
        // timer fires.  This prevents the pause handler from treating the
        // previous image as an exact match during the debounce window.
        previewPendingFrame = frame;
        previewTimer = setTimeout(() => { previewTimer = null; void renderFramePreview(); }, PREVIEW_DEBOUNCE_MS);
    }

    function syncOutputState() {
        if (!state) return;
        const multiple = Math.max(1, int(controls.multiple.value, DEFAULT_MULTIPLE));
        state.output = {
            width: dimensionRequest(controls.width.value, multiple),
            height: dimensionRequest(controls.height.value, multiple),
            multiple,
            fit_mode: ["letterbox", "crop", "fill"].includes(String(controls.fit.value)) ? String(controls.fit.value) : "letterbox",
            fill_color: hex(controls.fillText.value, DEFAULT_FILL),
        };
    }

    function setOutputControls(output = {}) {
        controls.multiple.value = String(Math.max(1, int(output.multiple, DEFAULT_MULTIPLE)));
        controls.width.value = String(dimensionRequest(output.width, int(controls.multiple.value, DEFAULT_MULTIPLE)));
        controls.height.value = String(dimensionRequest(output.height, int(controls.multiple.value, DEFAULT_MULTIPLE)));
        controls.fit.value = ["letterbox", "crop", "fill"].includes(String(output.fit_mode)) ? String(output.fit_mode) : "letterbox";
        controls.fillColor.value = hex(output.fill_color, DEFAULT_FILL);
        controls.fillText.value = controls.fillColor.value;
    }

    function outputChanged(record = false) {
        if (!state) return;
        syncOutputState(); invalidatePlaybackProxy(); updateStageAspect(); scheduleFramePreview(); scheduleStateSave();
        setStatus("Output canvas updated");
        if (record) void recordEditableHistory("Change output canvas");
    }

    function normalizeOutputDimensionInputs() {
        const multiple = Math.max(1, int(controls.multiple.value, DEFAULT_MULTIPLE));
        controls.width.value = String(dimensionRequest(controls.width.value, multiple));
        controls.height.value = String(dimensionRequest(controls.height.value, multiple));
    }

    function configureInputs() {
        controls.width.value = String(currentValues.width); controls.height.value = String(currentValues.height); controls.multiple.value = String(currentValues.multiple); controls.fit.value = ["letterbox", "crop", "fill"].includes(currentValues.fit_mode) ? currentValues.fit_mode : "letterbox"; controls.fillColor.value = currentValues.fill_color; controls.fillText.value = currentValues.fill_color; controls.threshold.value = String(currentValues.shot_detect_threshold); controls.minScene.value = String(currentValues.shot_detect_min_scene_sec);
        controls.fillColor.addEventListener("input", () => { controls.fillText.value = hex(controls.fillColor.value); outputChanged(false); });
        controls.fillColor.addEventListener("change", () => outputChanged(true));
        controls.fillText.addEventListener("change", () => { const value = hex(controls.fillText.value); controls.fillText.value = value; controls.fillColor.value = value; outputChanged(true); });
        [controls.width, controls.height, controls.multiple, controls.fit].forEach((input) => { input.addEventListener("input", () => outputChanged(false)); input.addEventListener("change", () => { normalizeOutputDimensionInputs(); outputChanged(true); }); });
        controls.threshold.addEventListener("change", () => { if (!state) return; state.threshold = clamp(number(controls.threshold.value, 0.5), 0, 1); invalidatePlaybackProxy(); scheduleStateSave(); });
        controls.minScene.addEventListener("change", () => { if (!state) return; state.min_scene_seconds = Math.max(0, number(controls.minScene.value, 0)); invalidatePlaybackProxy(); scheduleStateSave(); });
        controls.in.addEventListener("change", () => { if (!state) return; state.in_frame = clamp(int(controls.in.value, 0), 0, Math.max(0, state.out_frame - 1)); markChanged("In point changed"); });
        controls.out.addEventListener("change", () => { if (!state) return; state.out_frame = clamp(int(controls.out.value, state.duration_frames), state.in_frame + 1, state.duration_frames); markChanged("Out point changed"); });
        [["scaleX", controls.scaleX], ["scaleY", controls.scaleY], ["rotation", controls.rotation], ["translateX", controls.translateX], ["translateY", controls.translateY]].forEach(([name, input]) => {
            input.addEventListener("input", () => applyTransformControls(false, name, false));
            input.addEventListener("change", () => applyTransformControls(true, name));
        });
        [["scaleXValue", controls.scaleXValue], ["scaleYValue", controls.scaleYValue], ["rotationValue", controls.rotationValue], ["translateXValue", controls.translateXValue], ["translateYValue", controls.translateYValue]].forEach(([name, input]) => {
            input.addEventListener("input", () => applyTransformControls(false, name));
            input.addEventListener("change", () => applyTransformControls(true, name));
        });
        controls.syncScale.addEventListener("click", () => {
            const enabled = controls.syncScale.getAttribute("aria-pressed") !== "true";
            controls.syncScale.setAttribute("aria-pressed", String(enabled));
            if (enabled) { controls.scaleY.value = controls.scaleX.value; controls.scaleYValue.value = controls.scaleX.value; applyTransformControls(true, "scaleX"); }
        });
        controls.mirror.addEventListener("change", () => applyTransformControls(true, "mirror"));
        dialog.querySelectorAll(".cs-default").forEach((button) => button.addEventListener("click", (event) => {
            event.preventDefault();
            const name = String(button.dataset.reset || "");
            if (name === "width") controls.width.value = "-1";
            else if (name === "height") controls.height.value = "-1";
            else if (name === "multiple") controls.multiple.value = String(DEFAULT_MULTIPLE);
            else if (name === "fit") controls.fit.value = "letterbox";
            else if (name === "fill") { controls.fillColor.value = DEFAULT_FILL; controls.fillText.value = DEFAULT_FILL; }
            else if (name === "threshold") { controls.threshold.value = "0.5"; if (state) state.threshold = 0.5; scheduleStateSave(); return; }
            else if (name === "min-scene") { controls.minScene.value = "0"; if (state) state.min_scene_seconds = 0; scheduleStateSave(); return; }
            else if (name === "scale-x") { controls.scaleX.value = "1"; controls.scaleXValue.value = "1"; applyTransformControls(true, "scaleX"); return; }
            else if (name === "scale-y") { controls.scaleY.value = "1"; controls.scaleYValue.value = "1"; applyTransformControls(true, "scaleY"); return; }
            else if (name === "rotation") { controls.rotation.value = "0"; controls.rotationValue.value = "0"; applyTransformControls(true, "rotation"); return; }
            else if (name === "translate-x") { controls.translateX.value = "0"; controls.translateXValue.value = "0"; applyTransformControls(true, "translateX"); return; }
            else if (name === "translate-y") { controls.translateY.value = "0"; controls.translateYValue.value = "0"; applyTransformControls(true, "translateY"); return; }
            outputChanged(true);
        }));
    }

    function installEvents() {
        dialog.querySelector(".cs-time-edit-close").addEventListener("click", close);
        dialog.querySelector(".cs-cancel").addEventListener("click", close);
        dialog.querySelector(".cs-apply").addEventListener("click", apply);
        dialog.querySelector(".cs-set-in").addEventListener("click", setIn); dialog.querySelector(".cs-set-out").addEventListener("click", setOut);
        inMarker.addEventListener("pointerdown", (event) => beginRangeDrag("in", event));
        outMarker.addEventListener("pointerdown", (event) => beginRangeDrag("out", event));
        pointerRow.addEventListener("pointerdown", beginPlayheadDrag);
        axis.addEventListener("pointerdown", beginPlayheadDrag);
        dialog.querySelector(".cs-step-back").addEventListener("click", () => scrubToFrame(frame - 1)); dialog.querySelector(".cs-step-forward").addEventListener("click", () => scrubToFrame(frame + 1));
        controls.undo.addEventListener("click", () => void stepHistory("undo"));
        controls.redo.addEventListener("click", () => void stepHistory("redo"));
        dialog.querySelector(".cs-shot").addEventListener("click", () => void detectShots());
        dialog.querySelector(".cs-time-edit-zoom-in").addEventListener("click", () => zoomTimeline(1));
        dialog.querySelector(".cs-time-edit-zoom-fit").addEventListener("click", fitTimeline);
        dialog.querySelector(".cs-time-edit-zoom-out").addEventListener("click", () => zoomTimeline(-1));
        timelineViewport.addEventListener("wheel", (event) => {
            event.preventDefault();
            zoomTimeline(event.deltaY < 0 ? 1 : -1, timelineXToFrame(event.clientX));
        }, { passive: false });
        dialog.querySelector(".cs-time-edit-timeline").addEventListener("contextmenu", showContextMenu);
        dialog.addEventListener("pointerdown", (event) => { if (!event.target.closest(".cs-time-edit-context-menu")) closeContextMenu(); }, true);
        dialog.addEventListener("keydown", (event) => {
            if (!(event.ctrlKey || event.metaKey) || String(event.key).toLowerCase() !== "z") return;
            event.preventDefault(); void stepHistory(event.shiftKey ? "redo" : "undo");
        });
        dialog.querySelector(".cs-play").addEventListener("click", () => {
            if (!state) return;
            if (!stageVideo.paused) { stageVideo.pause(); playingSelection = false; return; }
            const generation = ++proxyRequestGeneration;
            playingSelection = true;
            // The proxy is now the authoritative transport surface.  Hide any
            // old exact-frame overlay immediately so it cannot remain visible
            // while the proxy is being prepared or after it is paused.
            stageFrame.style.display = "none";
            setStatus("Building playback proxy…");
            setStageStatus("Building preview cache…", 0, true);
            void (async () => {
                const result = await buildTimelineProxy(node, state, controls, source, (progress) => setStageStatus(`Building preview cache… ${Math.round(progress)}%`, progress, true));
                if (closed || generation !== proxyRequestGeneration || !playingSelection) return;
                if (result?.status === "ready" && result.video_url) {
                    proxy = {
                        url: api.apiURL(String(result.video_url)),
                        token: String(result.token || ""),
                        info: result.info || {},
                        label: "Current timeline proxy",
                    };
                    cacheKey = String(proxy.info?.cache_fingerprint || cacheKey || "");
                    proxyStartFrame = int(proxy.info?.timeline_in_frame ?? proxy.info?.in_frame, state.in_frame);
                    stageVideo.src = proxy.url;
                    stageVideo.load();
                    proxyLabel.textContent = "Current edits · proxy playback";
                    setStageStatus("Proxy ready", 100, true);
                    window.setTimeout(() => { if (!closed && !playingSelection) setStageStatus("", null, false); }, 500);
                    setStatus("Playing In/Out range");
                    try {
                        await new Promise((resolve) => {
                            if (stageVideo.readyState >= 1) resolve();
                            else stageVideo.addEventListener("loadedmetadata", resolve, { once: true });
                        });
                    } catch (_) { /* media element may be closed */ }
                    if (closed || generation !== proxyRequestGeneration || !playingSelection) return;
                    setFrame(state.in_frame);
                    stageVideo.play().catch(() => { playingSelection = false; });
                } else {
                    // Playing the raw source here would silently ignore moved,
                    // trimmed, or transformed clips.  Keep the exact-frame
                    // editor feedback visible and require a composed proxy for
                    // transport playback so the preview never lies about the
                    // current timeline.
                    setStatus(result?.error || "Playback proxy unavailable; scrub for exact-frame preview");
                    setStageStatus("Proxy unavailable · exact-frame preview only", 0, true);
                    playingSelection = false;
                    stageVideo.pause();
                    setFrame(frame);
                }
            })();
        });
        stageVideo.addEventListener("timeupdate", () => {
            // Only transport playback owns the playhead. Assigning
            // currentTime on a paused, In/Out-bounded proxy while scrubbing
            // must not overwrite the user's arbitrary timeline position.
            if (playingSelection && fps > 0) frame = clamp(proxyStartFrame + Math.round(stageVideo.currentTime * fps), 0, durationFrames() - 1);
            currentLabel.textContent = formatTime(frame, fps); pointer.style.left = `${clamp(frame / Math.max(1, durationFrames() - 1), 0, 1) * 100}%`; updateLocalTransform();
            if (playingSelection && frame >= state.out_frame - 1) {
                // Keep the decoded proxy frame visible while the pause event
                // and the final exact-frame request settle.  In particular,
                // do not let the pause handler reveal the previous image.
                stageFrame.style.display = "none";
                stageVideo.pause();
                playingSelection = false;
                setFrame(state.out_frame - 1, false);
            }
        });
        stageVideo.addEventListener("loadedmetadata", () => { setFrame(frame); });
        stageVideo.addEventListener("play", () => { dialog.querySelector(".cs-play").textContent = "Pause"; });
        stageVideo.addEventListener("play", () => { stageFrame.style.display = "none"; });
        stageVideo.addEventListener("pause", () => {
            dialog.querySelector(".cs-play").textContent = "Play";
            // Only reveal an exact overlay when it is known to represent the
            // current playhead and no newer request is waiting.  Otherwise
            // leave the paused video frame visible until the matching PNG
            // arrives.
            const imageFrame = int(stageFrame.dataset.frame, -1);
            const imageMatchesFrame = imageFrame === frame;
            if (stageVideo.paused && !playingSelection && stageFrame.src && frameExact && previewPendingFrame == null && imageMatchesFrame) {
                stageFrame.style.display = "block";
            } else {
                stageFrame.style.display = "none";
            }
        });
        dialog.addEventListener("cancel", (event) => { event.preventDefault(); close(); });
        new ResizeObserver(() => { updateLocalTransform(); renderTimeline(); }).observe(inner);
    }

    function close() {
        if (closed) return; closed = true; stageVideo.pause();
        proxyRequestGeneration += 1; playingSelection = false;
        if (stateTimer) { clearTimeout(stateTimer); stateTimer = null; }
        if (!applied && initialState) {
            // Edits are provisional until Apply.  Restore the snapshot because
            // debounced state writes may already have reached the server.
            void saveTimelineState(node, initialState);
            historyQueue = historyQueue.then(() => timelineHistoryRequest(node, "reset", editableTimelineSnapshot(initialState), "Cancel edits"));
            void historyQueue;
        }
        const objectUrl = stageFrame.dataset.objectUrl; if (objectUrl) URL.revokeObjectURL(objectUrl);
        if (previewTimer) clearTimeout(previewTimer);
        dialog.close(); dialog.remove();
    }

    function apply() {
        if (!state) return;
        if (stateTimer) { clearTimeout(stateTimer); stateTimer = null; }
        // Keep the requested dimensions in the widgets/descriptor and let the
        // backend derive a missing side before rounding both sides upward.
        // Pre-rounding one supplied side here can change the derived aspect
        // result for very wide or tall sources.
        const multiple = Math.max(1, int(controls.multiple.value, DEFAULT_MULTIPLE));
        normalizeOutputDimensionInputs();
        const width = dimensionRequest(controls.width.value, multiple); const height = dimensionRequest(controls.height.value, multiple);
        state.threshold = clamp(number(controls.threshold.value, state.threshold), 0, 1); state.min_scene_seconds = Math.max(0, number(controls.minScene.value, state.min_scene_seconds));
        state.in_frame = clamp(int(controls.in.value, state.in_frame), 0, state.duration_frames > 0 ? state.duration_frames - 1 : 0); state.out_frame = clamp(int(controls.out.value, state.out_frame), state.in_frame + 1, state.duration_frames);
        state.output = { width, height, multiple, fit_mode: String(controls.fit.value || "letterbox"), fill_color: hex(controls.fillText.value, DEFAULT_FILL) };
        const json = canonicalJson(publicState(state));
        applied = true;
        void recordEditableHistory("Apply timeline edits");
        setWidgetValue(node, "timeline_json", json); setWidgetValue(node, "width", width); setWidgetValue(node, "height", height); setWidgetValue(node, "multiple", multiple); setWidgetValue(node, "fit_mode", controls.fit.value); setWidgetValue(node, "fill_color", hex(controls.fillText.value, DEFAULT_FILL)); setWidgetValue(node, "in_frame", state.in_frame); setWidgetValue(node, "out_frame", state.out_frame >= state.duration_frames ? -1 : state.out_frame); setWidgetValue(node, "shot_detect_threshold", state.threshold); setWidgetValue(node, "shot_detect_min_scene_sec", state.min_scene_seconds);
        void saveTimelineState(node, state); node.graph?.setDirtyCanvas?.(true, true); close();
    }

    installEvents(); configureInputs();
    setStageStatus("Preparing preview… 0%", 0, true);
    void (async () => {
        try {
            source = await findSource(node, (progress, result) => {
                const stage = String(result?.stage || "cache").toLowerCase();
                setStageStatus(`Preparing preview… ${Math.round(progress)}%`, progress, true);
                if (stage) setStatus(`Building cache… ${Math.round(progress)}%`);
            }); fps = source.fps; frames = source.frames;
            if (closed) return;
            const persisted = String(widget(node, "timeline_json")?.value || "");
            const remote = await fetchTimelineState(node);
            if (closed) return;
            STATE_REVISIONS.set(node, Math.max(int(STATE_REVISIONS.get(node), 0), int(remote?.revision, 0)));
            state = normalizeState(persisted || remote || {}, frames, fps, currentValues.shot_detect_threshold, currentValues.shot_detect_min_scene_sec);
            state.fps = fps;
            const currentIdentity = String(source?.info?.source_identity || source?.info?.source_fingerprint || source?.info?.input_signature || "").trim();
            const currentFingerprint = String(source?.info?.source_fingerprint || source?.info?.input_signature || "").trim();
            const currentFingerprintKind = String(source?.info?.source_fingerprint_kind || "").toLowerCase();
            const savedIdentity = String(state.source_identity || "").trim();
            const savedFingerprint = String(state.source_fingerprint || "").trim();
            const savedFingerprintKind = String(state.source_fingerprint_kind || "").toLowerCase();
            const savedSourceFrames = int(state.source_frame_count, 0);
            const savedSourceFps = number(state.source_fps, 0);
            const savedWindowStart = state.source_start_frame == null ? null : int(state.source_start_frame, 0);
            const savedWindowEnd = state.source_end_frame == null ? null : int(state.source_end_frame, -1);
            const currentWindowStart = (source?.info?.source_start_frame != null || source?.info?.start_frame != null)
                ? int(source.info.source_start_frame ?? source.info.start_frame, 0)
                : null;
            const currentWindowEnd = (source?.info?.source_end_frame != null || source?.info?.end_frame != null)
                ? int(source.info.source_end_frame ?? source.info.end_frame, -1)
                : null;
            const comparableFingerprint = savedFingerprint && currentFingerprint && (
                (savedFingerprintKind && currentFingerprintKind && savedFingerprintKind === currentFingerprintKind)
                || (!savedFingerprintKind && !currentFingerprintKind && savedFingerprint.length === currentFingerprint.length)
            );
            const sourceShapeChanged = (savedSourceFrames > 0 && savedSourceFrames !== frames)
                || (savedSourceFps > 0 && Math.abs(savedSourceFps - fps) > 1e-3);
            const sourceWindowChanged = (savedWindowStart != null && currentWindowStart != null && savedWindowStart !== currentWindowStart)
                || (savedWindowEnd != null && currentWindowEnd != null && savedWindowEnd !== currentWindowEnd);
            if (sourceShapeChanged || sourceWindowChanged || (savedIdentity && currentIdentity && savedIdentity !== currentIdentity && (!savedFingerprintKind || !currentFingerprintKind || savedFingerprintKind === currentFingerprintKind)) || (comparableFingerprint && savedFingerprint !== currentFingerprint)) {
                const output = state.output;
                state = normalizeState({}, frames, fps, currentValues.shot_detect_threshold, currentValues.shot_detect_min_scene_sec);
                if (output) state.output = output;
                setStatus("Source changed; timeline was reset to the default clip");
            }
            if (currentIdentity) state.source_identity = currentIdentity;
            if (currentFingerprint) {
                state.source_fingerprint = currentFingerprint;
                state.source_fingerprint_version = SOURCE_FINGERPRINT_VERSION;
            }
            if (source?.info?.source_fingerprint_kind) state.source_fingerprint_kind = String(source.info.source_fingerprint_kind).toLowerCase();
            state.source_frame_count = frames;
            state.source_fps = fps;
            if (source?.info?.source_start_frame != null || source?.info?.start_frame != null) state.source_start_frame = Math.max(0, int(source.info.source_start_frame ?? source.info.start_frame, 0));
            if (source?.info?.source_end_frame != null || source?.info?.end_frame != null) state.source_end_frame = Math.max(0, int(source.info.source_end_frame ?? source.info.end_frame, frames - 1));
            initialState = JSON.parse(JSON.stringify(publicState(state)));
            if (closed) return;
            if (state.output) {
                // A saved timeline may carry output settings even when the
                // corresponding node widgets were reset by an older graph.
                if (currentValues.width <= 0 && state.output.width > 0) controls.width.value = String(state.output.width);
                if (currentValues.height <= 0 && state.output.height > 0) controls.height.value = String(state.output.height);
                if (state.output.multiple) controls.multiple.value = String(state.output.multiple);
                controls.fit.value = ["letterbox", "crop", "fill"].includes(state.output.fit_mode) ? state.output.fit_mode : controls.fit.value;
                controls.fillColor.value = hex(state.output.fill_color, controls.fillColor.value); controls.fillText.value = hex(state.output.fill_color, DEFAULT_FILL);
            }
            if (currentValues.in_frame !== 0 || currentValues.out_frame !== -1) { state.in_frame = clamp(currentValues.in_frame, 0, state.duration_frames > 0 ? state.duration_frames - 1 : 0); state.out_frame = currentValues.out_frame < 0 ? state.duration_frames : clamp(currentValues.out_frame, state.in_frame + 1, state.duration_frames); }
            syncOutputState();
            const history = await timelineHistoryRequest(node, "init", editableTimelineSnapshot(state), "Open editor");
            historyReady = Boolean(history);
            updateHistoryButtons(history || {});
            sourceLabel.textContent = `${source.label} · ${frames} frames · ${fps.toFixed(3)} fps`;
            const stageDimensions = sourceCanvasDimensions(source.info);
            if (stageDimensions.width > 0 && stageDimensions.height > 0) stage.style.aspectRatio = `${stageDimensions.width}/${stageDimensions.height}`;
            selectedId = state.clips[0]?.id || state.audio_clips[0]?.id || null; frame = state.in_frame;
            // A direct filename URL represents the complete source file. If
            // CS Load Video has a non-default trim/FPS/resize, showing that
            // URL would disagree with the connected VIDEO. Keep transport
            // playback disabled until a composed proxy is available; exact
            // frame requests still use the selected window in the backend.
            const directWindowed = Boolean(
                source?.directFileSource
                && (
                    source?.isCSLoad
                    ||
                    int(source?.startFrame, 0) > 0
                    || int(source?.endFrame, -1) >= 0
                    || number(source?.targetFps, 0) > 0
                    || int(source?.outputWidth, 0) > 0
                    || int(source?.outputHeight, 0) > 0
                )
            );
            if (source.url && !directWindowed) {
                stageVideo.src = source.url;
                stageVideo.muted = false;
                stageVideo.load();
                directSourceMode = true;
            } else {
                stageVideo.removeAttribute("src");
                stageVideo.load();
                directSourceMode = false;
            }
            proxy = await fetchProxy(node, "", currentIdentity, frames); cacheKey = String(proxy?.info?.cache_fingerprint || proxy?.info?.cache_key || "");
            proxyStartFrame = proxy ? int(proxy.info?.timeline_in_frame ?? proxy.info?.in_frame, state.in_frame) : 0;
            if (proxy?.url) { stageVideo.src = proxy.url; stageVideo.load(); directSourceMode = false; proxyLabel.textContent = "Proxy preview available · edits regenerate after Apply/run"; setStageStatus("", null, false); } else if (source.url && !directWindowed) { proxyLabel.textContent = "Proxy not available · showing source while editing (run node to build proxy)"; setStageStatus("", null, false); } else { proxyLabel.textContent = "Proxy pending · scrub for exact-frame preview or run the node"; setStageStatus("", null, false); }
            renderTimeline(); updateSelectedPanel(); setFrame(frame); setStatus(proxy ? "Ready · proxy playback" : "Ready · low-resolution/source feedback");
            updateStageAspect();
        } catch (error) {
            state = normalizeState({}, frames, fps, currentValues.shot_detect_threshold, currentValues.shot_detect_min_scene_sec); renderTimeline(); updateSelectedPanel(); setStatus(error?.message || "Unable to load VIDEO input"); setStageStatus("Preview unavailable", 0, true);
        }
    })();
    return dialog;
}

app.registerExtension({
    name: "CineStyle.VideoTimelineEdit",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_ID) return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            original?.apply(this, arguments);
            if (this.__csTimeEditButton) return;
            const button = this.addWidget("button", "Edit Timeline", "", () => { openTimeline(this); });
            button.name = "Edit Timeline"; button.label = "Edit Timeline"; button.options = { ...(button.options || {}), serialize: false };
            this.__csTimeEditButton = button;
            this.setSize?.([430, Math.max(390, this.computeSize?.()[1] || 390)]);
        };
        const originalConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info) {
            originalConfigure?.call(this, info);
            const button = this.widgets?.find((item) => item.name === "Edit Timeline");
            if (button) button.options = { ...(button.options || {}), serialize: false };
        };
    },
    // Some ComfyUI versions restore graph nodes without invoking the patched
    // constructor. Install the button for those nodes as well.
    loadedGraphNode(node) {
        if (node?.type !== NODE_ID || node.__csTimeEditButton) return;
        const button = node.addWidget?.("button", "Edit Timeline", "", () => { openTimeline(node); });
        if (!button) return;
        button.name = "Edit Timeline"; button.label = "Edit Timeline"; button.options = { ...(button.options || {}), serialize: false };
        node.__csTimeEditButton = button;
        node.setSize?.([430, Math.max(390, node.computeSize?.()[1] || 390)]);
    },
});

export { normalizeState, normalizeClip, normalizeTransform, canonicalJson };
