import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { createCineStyleTextEditor } from "./cinestyle_text_editor.js";

const NODE_ID = "CS_Video_Subtitle";
const STYLE_ID = "cinestyle-subtitle-timeline-style";
const SOURCE_VIDEO_REQUIRED_MESSAGE = "找不到可访问的源视频文件，请先执行一次节点并保持源视频路径有效。";
const PERSISTED_WIDGET_NAMES = [
    "edited_srt", "preview_in", "preview_out", "font", "font_size", "primary_color", "secondary_color",
    "gradient", "text_align", "italic", "letter_spacing", "position_x", "position_y",
    "outline_size", "outline_color", "shadow_size", "shadow_color", "Edit Timeline",
];
const NODE_SUBTITLE_CLIPBOARDS = new WeakMap();

function widget(node, name) { return node.widgets?.find((item) => item.name === name); }
const NUMERIC_LIMITS = { font_size: [8, 200], outline_size: [0, 20], shadow_size: [0, 20], letter_spacing: [-10, 50], position_x: [0, 1], position_y: [0, 1], preview_in: [0, 10000000], preview_out: [-1, 10000000] };
const COLOR_DEFAULTS = { primary_color: "#FFFFFF", secondary_color: "#FF0000", outline_color: "#000000", shadow_color: "#000000" };
function finiteNumber(value, fallback) {
    if (value == null || (typeof value === "string" && value.trim() === "")) return fallback;
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
}
function normalizePosition(value, fallback) {
    return clamp(Math.round(finiteNumber(value, fallback) * 100) / 100, 0, 1);
}
function normalizeHex(value, fallback) {
    const text = String(value ?? "").trim().toUpperCase();
    return /^#[0-9A-F]{6}$/.test(text) ? text : fallback;
}
function configureSubtitleWidgetValues(node, info) {
    const incoming = Array.isArray(info?.widgets_values) ? info.widgets_values : null;
    if (!incoming || !node.widgets?.length) return info;
    const hasPersistedSrtInput = Array.isArray(info?.inputs) && info.inputs.some((item) => item?.name === "edited_srt");
    const values = hasPersistedSrtInput ? incoming : ["", ...incoming];
    if (values.length !== PERSISTED_WIDGET_NAMES.length) {
        console.warn(`[CS Video Subtitle] widgets_values length ${incoming.length} does not match the ${PERSISTED_WIDGET_NAMES.length} named values.`);
    }
    const byName = new Map(PERSISTED_WIDGET_NAMES.map((name, index) => [name, values[index]]));
    const mapped = node.widgets.map((item, index) => {
        if (item?.name === "srt") return item.value ?? "";
        if (byName.has(item?.name)) return byName.get(item.name);
        return incoming[index];
    });
    return { ...info, widgets_values: mapped };
}
function canonicalSubtitleValues(info) {
    const incoming = Array.isArray(info?.widgets_values) ? info.widgets_values : null;
    if (!incoming) return null;
    const hasPersistedSrtInput = Array.isArray(info?.inputs) && info.inputs.some((item) => item?.name === "edited_srt");
    return hasPersistedSrtInput ? incoming : ["", ...incoming];
}
function applySubtitleWidgetValuesByName(node, info) {
    const values = canonicalSubtitleValues(info);
    if (!values) return;
    const byName = new Map(PERSISTED_WIDGET_NAMES.map((name, index) => [name, values[index]]));
    for (const name of PERSISTED_WIDGET_NAMES) {
        const target = widget(node, name);
        if (target && byName.has(name)) target.value = byName.get(name);
    }
}
function serializeSubtitleWidgetValues(node) {
    return PERSISTED_WIDGET_NAMES.map((name) => widget(node, name)?.value ?? null);
}
function graphNode(graph, id) { return graph?.getNodeById?.(id) || (graph?._nodes || []).find((item) => String(item?.id) === String(id)) || null; }
function filenameValue(value) {
    if (typeof value === "string") return value.trim();
    if (!value || typeof value !== "object") return "";
    for (const key of ["filename", "video", "video_path", "path", "source"]) {
        if (typeof value[key] === "string" && value[key].trim()) return value[key].trim();
    }
    return "";
}
function connectedVideoFilename(node) {
    const visited = new Set();
    function findFilename(origin) {
        if (!origin) return "";
        const identity = String(origin.id ?? origin.type ?? visited.size);
        if (visited.has(identity)) return "";
        visited.add(identity);
        for (const name of ["video", "file", "filename", "video_file", "path", "filepath", "input", "source"]) {
            const value = filenameValue(widget(origin, name)?.value);
            if (/\.(mp4|mov|mkv|avi|webm|m4v|mpg|mpeg|wmv|flv)(?:\s*\[[^\]]+\])?$/i.test(value)) return value;
        }
        for (const input of origin.inputs || []) {
            const upstream = connectedOrigin(origin, input.name);
            const value = findFilename(upstream);
            if (value) return value;
        }
        return "";
    }
    return findFilename(connectedOrigin(node, "video"));
}
function connectedOrigin(node, inputName) {
    const input = node.inputs?.find((item) => item.name === inputName);
    if (!input) return null;
    const graph = node.graph || app.graph;
    const candidates = [];
    if (input.link != null) candidates.push(input.link);
    if (Array.isArray(input.links)) candidates.push(...input.links);
    for (const candidate of candidates) {
        const link = typeof candidate === "object" ? candidate : (graph?.links?.[candidate] || graph?._links?.[candidate]);
        const originId = link?.origin_id ?? link?.originId ?? link?.origin;
        const origin = originId == null ? null : graphNode(graph, originId);
        if (origin) return origin;
    }
    return null;
}
function graphSrtText(node) {
    const visited = new Set();
    function findText(origin) {
    if (!origin) return "";
    const identity = String(origin.id ?? origin.type ?? visited.size);
    if (visited.has(identity)) return "";
    visited.add(identity);
    const candidateNames = ["srt", "text", "string", "value", "content", "prompt", "file", "filename"];
    for (const name of candidateNames) {
        const value = widget(origin, name)?.value;
        if (typeof value === "string" && parseSrt(value).length) return value;
    }
    for (const item of origin.widgets || []) {
        const value = item?.value;
        if (typeof value === "string" && parseSrt(value).length) return value;
    }
    for (const input of origin.inputs || []) {
        const upstream = connectedOrigin(origin, input.name);
        if (!upstream) continue;
        const nested = findText(upstream);
        if (nested) return nested;
    }
    return "";
    }
    return findText(connectedOrigin(node, "srt"));
}
async function fetchCachedProxy(node, filename = "") {
    const nodeId = String(node?.id ?? "").trim();
    if (!nodeId) return null;
    const response = await api.fetchApi(`/cinestyle/video-subtitle-preview-info?${new URLSearchParams({ node_id: nodeId, video_filename: String(filename || ""), t: String(Date.now()) })}`);
    if (response.status === 404) return null;
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Unable to read subtitle preview cache");
    return { url: api.apiURL(String(result.video_url || "")), info: result.info || {}, label: String(result.label || "Subtitle preview cache") };
}
async function fetchAudioWaveform(node, filename = "") {
    const nodeId = String(node?.id ?? "").trim();
    if (!nodeId && !filename) return null;
    const params = new URLSearchParams({ node_id: nodeId, video_filename: String(filename || ""), t: String(Date.now()) });
    const response = await api.fetchApi(`/cinestyle/video-subtitle-waveform?${params}`);
    if (!response.ok) return null;
    const result = await response.json();
    return {
        peaks: Array.isArray(result.peaks) ? result.peaks.map((value) => Math.max(0, Math.min(1, Number(value) || 0))) : [],
        duration: Number(result.duration) || 0,
    };
}
async function setTimelineOpen(node, open) {
    const nodeId = String(node?.id ?? "").trim();
    if (!nodeId) return;
    await api.fetchApi("/cinestyle/video-subtitle-timeline-state", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ node_id: nodeId, open: Boolean(open) }),
    });
}
async function fetchCachedSrt(node) {
    const nodeId = String(node?.id ?? "").trim();
    if (!nodeId) return null;
    const response = await api.fetchApi(`/cinestyle/video-subtitle-srt-cache?${new URLSearchParams({ node_id: nodeId, t: String(Date.now()) })}`);
    if (response.status === 404) return null;
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Unable to read cached SRT");
    return { srt: String(result.srt || ""), sourceHash: String(result.source_hash || "") };
}
async function srtSourceHash(value) {
    const data = new TextEncoder().encode(String(value || ""));
    const digest = await crypto.subtle.digest("SHA-1", data);
    return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}
async function saveCachedSrt(node, sourceHash, srt) {
    const response = await api.fetchApi("/cinestyle/video-subtitle-srt-cache", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ node_id: String(node?.id ?? ""), source_hash: sourceHash, srt }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Unable to save edited SRT");
}
function setWidgetValue(node, name, value) {
    const target = widget(node, name);
    if (!target) return;
    if (Object.prototype.hasOwnProperty.call(NUMERIC_LIMITS, name)) {
        const numericValue = Number(value);
        if (Number.isFinite(numericValue)) value = numericValue;
    }
    target.value = value;
    target.callback?.(value);
}
function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function boolValue(value) { return typeof value === "string" ? ["1", "true", "yes", "on"].includes(value.trim().toLowerCase()) : Boolean(value); }
function formatTime(seconds) {
    const safe = Math.max(0, Number(seconds) || 0);
    const hours = Math.floor(safe / 3600);
    const minutes = Math.floor((safe - hours * 3600) / 60);
    const remainder = safe - hours * 3600 - minutes * 60;
    return hours ? `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${remainder.toFixed(2).padStart(5, "0")}` : `${String(minutes).padStart(2, "0")}:${remainder.toFixed(2).padStart(5, "0")}`;
}
function parseSrtTime(value) {
    const match = String(value).trim().match(/^(\d+):(\d{2}):(\d{2})[,.](\d{3})$/);
    if (!match) return 0;
    return Number(match[1]) * 3600 + Number(match[2]) * 60 + Number(match[3]) + Number(match[4]) / 1000;
}
function parseSrt(text) {
    const source = String(text || "").replace(/^\ufeff/, "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
    const cues = [];
    for (const block of source.trim().split(/\n\s*\n/)) {
        const lines = block.split("\n");
        if (lines[0] && !lines[0].includes("-->")) lines.shift();
        if (!lines[0] || !lines[0].includes("-->")) continue;
        const [start, end] = lines[0].split("-->").map((part) => part.trim().split(/\s+/, 1)[0]);
        const cue = { id: cues.length + 1, start: parseSrtTime(start), end: parseSrtTime(end), text: lines.slice(1).join("\n").trim() };
        if (cue.text && cue.end > cue.start) cues.push(cue);
    }
    return cues;
}
function readCues(node, sourceSrt = "") {
    return parseSrt(sourceSrt || String(widget(node, "srt")?.value || ""));
}
function formatSrtTime(seconds) {
    const milliseconds = Math.max(0, Math.round((Number(seconds) || 0) * 1000));
    const hours = Math.floor(milliseconds / 3600000);
    const minutes = Math.floor((milliseconds % 3600000) / 60000);
    const secs = Math.floor((milliseconds % 60000) / 1000);
    const millis = milliseconds % 1000;
    return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")},${String(millis).padStart(3, "0")}`;
}
function cuesToSrt(cues) {
    return cues.filter((cue) => cue.text && cue.end > cue.start).map((cue, index) => `${index + 1}\n${formatSrtTime(cue.start)} --> ${formatSrtTime(cue.end)}\n${cue.text}`).join("\n\n") + "\n\n";
}
function videoUrl(filename) {
    const params = new URLSearchParams({ filename, type: "input", subfolder: "", t: String(Date.now()) });
    return api.apiURL(`/view?${params.toString()}`);
}
function proxyVideoUrl(filename, threshold = 1, size = 0.8) {
    const params = new URLSearchParams({ filename, proxy_threshold: String(threshold), proxy_size: String(size), t: String(Date.now()) });
    return api.apiURL(`/cinestyle/video-proxy?${params.toString()}`);
}
async function fetchInfo(filename) {
    const response = await api.fetchApi(`/cinestyle/video-info?filename=${encodeURIComponent(filename)}&proxy_threshold=1&proxy_size=0.8`);
    if (!response.ok) throw new Error(await response.text());
    return response.json();
}
async function fetchFonts() {
    try {
        const response = await api.fetchApi("/cinestyle/fonts");
        if (!response.ok) return [];
        const result = await response.json();
        return Array.isArray(result.fonts) ? result.fonts : [];
    } catch (_) { return []; }
}
function addStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .cs-subtitle-dialog { width:min(1180px,96vw); max-width:none; max-height:94vh; overflow:auto; padding:0; border:1px solid #343943; border-radius:10px; background:#17191e; color:#e6e9ef; box-shadow:0 22px 80px #000b; }
      .cs-subtitle-dialog::backdrop { background:#050609c2; }
      .cs-subtitle-shell { display:grid; gap:12px; padding:16px; font:13px/1.35 system-ui,sans-serif; }
      .cs-subtitle-head,.cs-subtitle-row,.cs-subtitle-foot { display:flex; align-items:center; gap:8px; }
      .cs-subtitle-head { justify-content:space-between; }
      .cs-subtitle-head > div { min-width:0; flex:1; }
      .cs-subtitle-title { margin:0; font-size:16px; font-weight:600; }
      .cs-subtitle-muted { color:#9299a8; }
      .cs-subtitle-close { border:0; background:transparent; color:#aeb5c2; font-size:22px; cursor:pointer; padding:0 5px; }
      .cs-subtitle-preview-wrap { position:relative; width:100%; aspect-ratio:16/9; background:#08090b; border-radius:6px; overflow:hidden; }
      .cs-subtitle-video { width:100%; height:100%; object-fit:contain; display:block; }
      .cs-subtitle-preview-loading { position:absolute; inset:0; z-index:6; display:flex; align-items:center; justify-content:center; padding:16px; background:#08090bd9; color:#dce7f3; text-align:center; pointer-events:none; }
      .cs-subtitle-preview-loading[hidden] { display:none; }
      .cs-subtitle-preview-loading::before { content:""; width:16px; height:16px; margin-right:9px; border:2px solid #5c7185; border-top-color:#7dc6ff; border-radius:50%; animation:cs-subtitle-spin .8s linear infinite; }
      @keyframes cs-subtitle-spin { to { transform:rotate(360deg); } }
      .cs-subtitle-overlay-clip { position:absolute; inset:0; overflow:hidden; pointer-events:none; }
      .cs-subtitle-overlay-image { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; pointer-events:none; display:none; }
      .cs-subtitle-interaction-box { position:absolute; display:none; box-sizing:border-box; border:1px dashed transparent; z-index:4; cursor:grab; touch-action:none; }
      .cs-subtitle-interaction-box:hover { border-color:#8bc7f5; }
      .cs-subtitle-interaction-box:active { cursor:grabbing; }
      .cs-subtitle-resize-handle { position:absolute; width:10px; height:10px; border:1px solid #e8f3ff; border-radius:2px; background:#317ec4; display:none; }
      .cs-subtitle-interaction-box:hover .cs-subtitle-resize-handle { display:block; }
      .cs-subtitle-resize-handle.nw { left:-6px; top:-6px; cursor:nwse-resize; }
      .cs-subtitle-resize-handle.ne { right:-6px; top:-6px; cursor:nesw-resize; }
      .cs-subtitle-resize-handle.sw { left:-6px; bottom:-6px; cursor:nesw-resize; }
      .cs-subtitle-resize-handle.se { right:-6px; bottom:-6px; cursor:nwse-resize; }
      .cs-subtitle-readout { display:flex; justify-content:space-between; color:#aeb5c2; font-variant-numeric:tabular-nums; }
      .cs-subtitle-pointer-row { position:relative; height:15px; user-select:none; }
      .cs-subtitle-pointer { position:absolute; top:0; width:16px; height:15px; transform:translateX(-50%); border:0; background:#55a9f5; clip-path:polygon(0 0,100% 0,50% 100%); cursor:ew-resize; z-index:5; }
      .cs-subtitle-viewport { position:relative; overflow:hidden; border:1px solid #363b45; border-radius:6px; background:#20232a; }
      .cs-subtitle-axis { position:relative; height:22px; color:#9299a8; font-size:11px; font-variant-numeric:tabular-nums; }
      .cs-subtitle-axis span { position:absolute; transform:translateX(-50%); top:4px; }
      .cs-subtitle-track { position:relative; height:34px; border-top:1px solid #343943; }
      .cs-subtitle-track-label { position:absolute; left:8px; top:9px; z-index:1; color:#aeb5c2; font-size:11px; pointer-events:none; }
      .cs-subtitle-track-body { position:absolute; inset:0; margin-left:0; }
      .cs-subtitle-track-video .cs-subtitle-track-body { background:repeating-linear-gradient(90deg,#343941 0 1px,transparent 1px 10%); }
      .cs-subtitle-track-subtitles .cs-subtitle-track-body { background:#292e36; }
      .cs-subtitle-cue { position:absolute; top:5px; bottom:5px; min-width:5px; overflow:visible; z-index:2; border:1px solid #4b9de8; border-radius:3px; background:#317ec4; color:#f5f7fb; cursor:grab; user-select:none; }
      .cs-subtitle-cue:active { cursor:grabbing; }
      .cs-subtitle-cue-label { display:block; overflow:hidden; padding:3px 8px; white-space:nowrap; text-overflow:ellipsis; pointer-events:none; }
      .cs-subtitle-cue-handle { position:absolute; top:-2px; bottom:-2px; width:7px; background:#f5f7fb; border-radius:2px; cursor:ew-resize; z-index:2; }
      .cs-subtitle-cue-handle.in { left:-4px; } .cs-subtitle-cue-handle.out { right:-4px; }
      .cs-subtitle-context-menu { position:fixed; z-index:30; display:grid; min-width:140px; padding:4px; gap:2px; border:1px solid #424956; border-radius:6px; background:#20232a; box-shadow:0 10px 32px #000b; }
      .cs-subtitle-context-menu button { border:0; border-radius:4px; padding:8px 10px; background:transparent; color:#f2f4f7; text-align:left; cursor:pointer; }
      .cs-subtitle-context-menu button:hover { background:#317ec4; }
      .cs-subtitle-range-band { position:absolute; top:22px; bottom:0; z-index:1; pointer-events:none; background:rgba(188,198,210,.16); border-left:1px solid rgba(210,220,230,.6); border-right:1px solid rgba(210,220,230,.6); }
      .cs-subtitle-range-marker { position:absolute; top:0; bottom:0; width:2px; background:#c9d4df; box-shadow:0 0 0 1px #15181d; pointer-events:auto; cursor:ew-resize; touch-action:none; }
      .cs-subtitle-range-marker::after { content:""; position:absolute; top:0; bottom:0; left:-9px; width:20px; background:transparent; pointer-events:auto; cursor:ew-resize; }
      .cs-subtitle-range-marker::before { content:""; position:absolute; top:-1px; width:0; height:0; border-left:5px solid transparent; border-right:5px solid transparent; border-top:6px solid #c9d4df; }
      .cs-subtitle-range-marker.in { left:-1px; } .cs-subtitle-range-marker.in::before { left:-4px; }
      .cs-subtitle-range-marker.out { right:-1px; } .cs-subtitle-range-marker.out::before { right:-4px; }
      .cs-subtitle-controls { display:flex; flex-wrap:wrap; gap:6px; }
      .cs-subtitle-controls button,.cs-subtitle-foot button { border:1px solid #424956; border-radius:5px; padding:7px 10px; background:#242832; color:#e6e9ef; cursor:pointer; }
      .cs-subtitle-controls button:hover,.cs-subtitle-foot button:hover { background:#303643; }
      .cs-subtitle-point-frame { min-width:52px; padding-left:7px !important; padding-right:7px !important; color:#9fc9ec !important; font-variant-numeric:tabular-nums; }
      .cs-subtitle-fields { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:9px; }
      .cs-subtitle-style-section { display:grid; gap:8px; padding:10px; border:1px solid #343943; border-radius:6px; background:#1b1e24; }
      .cs-subtitle-style-section-title { color:#cbd2dc; font-size:12px; font-weight:600; letter-spacing:.02em; }
      .cs-subtitle-style-section .cs-subtitle-fields { grid-template-columns:repeat(4,minmax(0,1fr)); gap:9px; }
      .cs-subtitle-style-editor { gap:9px; }
      .cs-subtitle-style-grid { display:grid; grid-template-columns:minmax(360px,2fr) minmax(200px,1.1fr) minmax(250px,1.35fr); gap:9px; }
      .cs-subtitle-style-group { display:grid; align-content:start; gap:7px; min-width:0; padding:8px; border:1px solid #343943; border-radius:5px; background:#20232a; }
      .cs-subtitle-style-group-title { color:#cbd2dc; font-size:11px; font-weight:600; }
      .cs-subtitle-style-group-fields { display:grid; gap:6px; min-width:0; }
      .cs-subtitle-typography-fields { grid-template-columns:minmax(0,1fr) auto; }
      .cs-subtitle-typography-fields > .cs-subtitle-field:first-child { grid-column:1 / -1; }
      .cs-subtitle-typography-fields > .cs-subtitle-param-compact:last-child { grid-column:1 / -1; }
      .cs-subtitle-typography-fields .cs-subtitle-param-compact { grid-template-columns:minmax(42px,auto) minmax(0,1fr) 42px 29px; min-width:0; }
      .cs-subtitle-typography-fields .cs-subtitle-param-compact input[type=range] { min-width:0; }
      .cs-subtitle-typography-fields .cs-subtitle-check { min-width:0; white-space:nowrap; }
      .cs-subtitle-fill-fields { grid-template-columns:minmax(0,1fr); }
      .cs-subtitle-effects-group { gap:6px; }
      .cs-subtitle-effect-row { display:grid; grid-template-columns:minmax(0,1fr); align-items:center; gap:5px; min-width:0; }
      .cs-subtitle-effect-color { display:flex; align-items:center; gap:6px; min-width:0; color:#9da5b4; font-size:11px; }
      .cs-subtitle-effect-color > span { flex:0 0 auto; }
      .cs-subtitle-effect-color .cs-subtitle-color-row { min-width:0; }
      .cs-subtitle-effect-color .cs-subtitle-hex { overflow:hidden; text-overflow:ellipsis; }
      .cs-subtitle-effect-divider { padding-top:7px; border-top:1px solid #343943; }
      .cs-subtitle-position-section .cs-subtitle-fields { grid-template-columns:minmax(90px,.7fr) minmax(180px,1fr) minmax(180px,1fr) auto; column-gap:14px; }
      .cs-subtitle-field { display:grid; gap:5px; color:#9da5b4; min-width:0; }
      .cs-subtitle-field input,.cs-subtitle-field select { width:100%; box-sizing:border-box; border:1px solid #424956; border-radius:5px; padding:7px 8px; background:#20232a; color:#f2f4f7; }
      .cs-subtitle-param { display:grid; grid-template-columns:minmax(52px,auto) 1fr 48px 29px; align-items:center; gap:7px; color:#d9dee6; min-width:0; }
      .cs-subtitle-param-compact { grid-template-columns:minmax(38px,auto) minmax(90px,1fr) 42px 29px; gap:4px; }
      .cs-subtitle-position-param { grid-template-columns:18px minmax(90px,1fr) 42px 29px; gap:4px; }
      .cs-subtitle-param input[type=range] { width:100%; accent-color:#55a9f5; }
      .cs-subtitle-color-row { display:flex; align-items:center; gap:7px; }
      .cs-subtitle-color-row input[type=color] { width:102px; min-width:102px; height:32px; flex:0 0 102px; box-sizing:border-box; padding:2px; border:1px solid #424956; border-radius:5px; background:#20232a; }
      .cs-subtitle-hex { color:#f7b955; font-variant-numeric:tabular-nums; font-family:ui-monospace,monospace; }
      .cs-subtitle-param output { color:#f7b955; text-align:right; font-variant-numeric:tabular-nums; }
      .cs-subtitle-param-reset { width:29px; min-height:27px; padding:3px; border:1px solid #424956; border-radius:5px; background:#20232a; color:#f2f4f7; cursor:pointer; font-size:15px; line-height:1; }
      .cs-subtitle-param-reset:hover { border-color:#6aa9df; }
      .cs-subtitle-position-reset { min-height:29px; white-space:nowrap; border:1px solid #424956; border-radius:5px; padding:6px 10px; background:#242832; color:#e6e9ef; cursor:pointer; }
      .cs-subtitle-position-reset:hover { background:#303643; }
      .cs-subtitle-field input[type=color] { height:32px; padding:2px; }
      .cs-subtitle-check { display:flex; align-items:center; gap:6px; min-height:32px; }
      .cs-subtitle-check input { width:auto; }
      .cs-subtitle-status { min-width:0; flex:1; color:#9299a8; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
      .cs-subtitle-foot { justify-content:flex-end; }
      .cs-subtitle-foot .apply { background:#317ec4; border-color:#4b9de8; }
      .cs-subtitle-track-audio { cursor:default !important; pointer-events:none !important; }
      .cs-subtitle-track-audio .cs-subtitle-track-body { background:#252a31; pointer-events:none !important; }
      .cs-subtitle-waveform { display:block; width:100%; height:100%; opacity:.82; pointer-events:none; }
      @media(max-width:900px) { .cs-subtitle-style-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
      @media(max-width:760px) { .cs-subtitle-style-section .cs-subtitle-fields,.cs-subtitle-position-section .cs-subtitle-fields{grid-template-columns:repeat(2,minmax(0,1fr));} .cs-subtitle-style-grid { grid-template-columns:1fr 1fr; } }
      @media(max-width:460px) { .cs-subtitle-style-section .cs-subtitle-fields{grid-template-columns:1fr;} .cs-subtitle-shell{padding:10px;} }
    `;
    document.head.append(style);
}

async function openTimeline(node) {
    const filename = String(connectedVideoFilename(node) || "");
    const externalSrt = graphSrtText(node) || String(widget(node, "srt")?.value || "");
    const persistedSrt = String(widget(node, "edited_srt")?.value || "").trim();
    let cachedSrt = null;
    let cachedProxy = null;
    let sourceSrt = persistedSrt || externalSrt;
    addStyles();
    const dialog = document.createElement("dialog");
    dialog.className = "cs-subtitle-dialog";
    dialog.innerHTML = `
      <div class="cs-subtitle-shell">
        <div class="cs-subtitle-head"><div><h2 class="cs-subtitle-title">Subtitle Timeline</h2><div class="cs-subtitle-muted cs-subtitle-file"></div></div><button class="cs-subtitle-close" type="button" aria-label="Close">&times;</button></div>
         <div class="cs-subtitle-preview-wrap"><video class="cs-subtitle-video" controls playsinline preload="metadata"></video><div class="cs-subtitle-overlay-clip"><img class="cs-subtitle-overlay-image" alt="" draggable="false"></div><div class="cs-subtitle-interaction-box"><span class="cs-subtitle-resize-handle nw"></span><span class="cs-subtitle-resize-handle ne"></span><span class="cs-subtitle-resize-handle sw"></span><span class="cs-subtitle-resize-handle se"></span></div><div class="cs-subtitle-preview-loading" role="status" aria-live="polite"><span class="cs-subtitle-preview-loading-text">Preparing subtitle preview...</span></div></div>
        <div class="cs-subtitle-readout"><span class="current">00:00.00</span><span class="range"></span><span class="duration">00:00.00</span></div>
        <div class="cs-subtitle-pointer-row"><button class="cs-subtitle-pointer" type="button" aria-label="Current time"></button></div>
        <div class="cs-subtitle-viewport"><div class="cs-subtitle-axis"></div><div class="cs-subtitle-range-band"><span class="cs-subtitle-range-marker in"></span><span class="cs-subtitle-range-marker out"></span></div><div class="cs-subtitle-track cs-subtitle-track-subtitles"><span class="cs-subtitle-track-label">Subtitles</span><div class="cs-subtitle-track-body"></div></div><div class="cs-subtitle-track cs-subtitle-track-audio" aria-label="Audio"><span class="cs-subtitle-track-label">Audio</span><div class="cs-subtitle-track-body"><canvas class="cs-subtitle-waveform" aria-hidden="true"></canvas></div></div></div>
        <div class="cs-subtitle-controls"><button class="set-in">Set In</button><button class="cs-subtitle-point-frame in-frame" type="button" aria-label="Jump to in point" title="Jump to in point">0</button><button class="back">|&lt;</button><button class="play">Play</button><button class="forward">&gt;|</button><button class="cs-subtitle-point-frame out-frame" type="button" aria-label="Jump to out point" title="Jump to out point">0</button><button class="set-out">Set Out</button></div>
        <div class="cs-subtitle-style-section cs-subtitle-style-editor"><div class="cs-subtitle-style-section-title">Text Style</div><div class="cs-subtitle-style-grid">
          <div class="cs-subtitle-style-group"><div class="cs-subtitle-style-group-title">Typography</div><div class="cs-subtitle-style-group-fields cs-subtitle-typography-fields"><label class="cs-subtitle-field">Font<select class="font"></select></label><div class="cs-subtitle-param cs-subtitle-param-compact"><label for="cs-subtitle-font-size">Size</label><input id="cs-subtitle-font-size" class="font-size" type="range" min="8" max="200" step="1"><output data-subtitle-output="font_size">30</output><button class="cs-subtitle-param-reset" data-reset="font_size" type="button" title="Reset Size">&#8634;</button></div><label class="cs-subtitle-field cs-subtitle-check"><span><input class="italic" type="checkbox"> Italic</span></label><div class="cs-subtitle-param cs-subtitle-param-compact"><label for="cs-subtitle-letter-spacing">Spacing</label><input id="cs-subtitle-letter-spacing" class="letter-spacing" type="range" min="-10" max="50" step="1"><output data-subtitle-output="letter_spacing">0</output><button class="cs-subtitle-param-reset" data-reset="letter_spacing" type="button" title="Reset Letter Spacing">&#8634;</button></div></div></div>
          <div class="cs-subtitle-style-group"><div class="cs-subtitle-style-group-title">Fill</div><div class="cs-subtitle-style-group-fields cs-subtitle-fill-fields"><div class="cs-subtitle-field">Primary Color<div class="cs-subtitle-color-row"><input class="primary-color" type="color"><output class="cs-subtitle-hex primary-color-hex">#FFFFFF</output></div></div><div class="cs-subtitle-field">Secondary Color<div class="cs-subtitle-color-row"><input class="secondary-color" type="color" value="#FF0000"><output class="cs-subtitle-hex secondary-color-hex">#FF0000</output></div></div><label class="cs-subtitle-field cs-subtitle-check"><span><input class="gradient" type="checkbox"> Vertical Gradient</span></label></div></div>
          <div class="cs-subtitle-style-group cs-subtitle-effects-group"><div class="cs-subtitle-style-group-title">Shadow</div><div class="cs-subtitle-effect-row"><div class="cs-subtitle-param cs-subtitle-param-compact"><label for="cs-subtitle-shadow-size">Size</label><input id="cs-subtitle-shadow-size" class="shadow-size" type="range" min="0" max="20" step="1"><output data-subtitle-output="shadow_size">3</output><button class="cs-subtitle-param-reset" data-reset="shadow_size" type="button" title="Reset Shadow">&#8634;</button></div><div class="cs-subtitle-effect-color"><span>Color</span><div class="cs-subtitle-color-row"><input class="shadow-color" type="color"><output class="cs-subtitle-hex shadow-color-hex">#000000</output></div></div></div><div class="cs-subtitle-style-group-title cs-subtitle-effect-divider">Outline</div><div class="cs-subtitle-effect-row"><div class="cs-subtitle-param cs-subtitle-param-compact"><label for="cs-subtitle-outline-size">Size</label><input id="cs-subtitle-outline-size" class="outline-size" type="range" min="0" max="20" step="1"><output data-subtitle-output="outline_size">2</output><button class="cs-subtitle-param-reset" data-reset="outline_size" type="button" title="Reset Outline">&#8634;</button></div><div class="cs-subtitle-effect-color"><span>Color</span><div class="cs-subtitle-color-row"><input class="outline-color" type="color"><output class="cs-subtitle-hex outline-color-hex">#000000</output></div></div></div></div>
        </div></div>
        <div class="cs-subtitle-style-section cs-subtitle-position-section"><div class="cs-subtitle-style-section-title">Position</div><div class="cs-subtitle-fields"><label class="cs-subtitle-field">Align<select class="text-align"><option value="left">Left</option><option value="center">Center</option><option value="right">Right</option></select></label><div class="cs-subtitle-param cs-subtitle-position-param"><label for="cs-subtitle-position-x">X</label><input id="cs-subtitle-position-x" class="position-x" type="range" min="0" max="1" step="0.01"><output data-subtitle-output="position_x">0.50</output><button class="cs-subtitle-param-reset" data-reset="position_x" type="button" title="Reset X">&#8634;</button></div><div class="cs-subtitle-param cs-subtitle-position-param"><label for="cs-subtitle-position-y">Y</label><input id="cs-subtitle-position-y" class="position-y" type="range" min="0" max="1" step="0.01"><output data-subtitle-output="position_y">0.88</output><button class="cs-subtitle-param-reset" data-reset="position_y" type="button" title="Reset Y">&#8634;</button></div><button class="cs-subtitle-position-reset move-reset" type="button">Reset Position</button></div></div>
        <div class="cs-subtitle-row"><span class="cs-subtitle-status"></span></div>
        <div class="cs-subtitle-foot"><button class="cancel">Cancel</button><button class="apply">Apply</button></div>
      </div>`;
    document.body.append(dialog);
    dialog.showModal();
    await setTimelineOpen(node, true).catch(() => {});
    const textEditor = createCineStyleTextEditor(dialog);

    const video = dialog.querySelector(".cs-subtitle-video");
    const previewWrap = dialog.querySelector(".cs-subtitle-preview-wrap");
    const overlayClip = dialog.querySelector(".cs-subtitle-overlay-clip");
    const overlayImage = dialog.querySelector(".cs-subtitle-overlay-image");
    const interactionBox = dialog.querySelector(".cs-subtitle-interaction-box");
    const viewport = dialog.querySelector(".cs-subtitle-viewport");
    const waveformCanvas = dialog.querySelector(".cs-subtitle-waveform");
    const axis = dialog.querySelector(".cs-subtitle-axis");
    const body = dialog.querySelector(".cs-subtitle-track-subtitles .cs-subtitle-track-body");
    const rangeBand = dialog.querySelector(".cs-subtitle-range-band");
    const pointer = dialog.querySelector(".cs-subtitle-pointer");
    const inFrameButton = dialog.querySelector(".in-frame");
    const outFrameButton = dialog.querySelector(".out-frame");
    const current = dialog.querySelector(".current");
    const range = dialog.querySelector(".range");
    const durationLabel = dialog.querySelector(".duration");
    const status = dialog.querySelector(".cs-subtitle-status");
    const loading = dialog.querySelector(".cs-subtitle-preview-loading");
    const loadingText = dialog.querySelector(".cs-subtitle-preview-loading-text");
    const setLoading = (message, visible = true) => { loadingText.textContent = message; loading.hidden = !visible; };
    const cues = readCues(node, sourceSrt).map((cue, index) => ({ ...cue, id: cue.id ?? index + 1 }));
    function normalizeCueLayout() {
        cues.sort((a, b) => Number(a.start) - Number(b.start));
        let cursor = 0;
        for (const cue of cues) {
            const start = Math.max(cursor, Number(cue.start) || 0);
            const end = Math.max(start + 0.05, Number(cue.end) || start + 0.05);
            cue.start = start;
            cue.end = end;
            cursor = end;
        }
    }
    normalizeCueLayout();
    function replaceCues(nextSrt) {
        const next = readCues(node, nextSrt).map((cue, index) => ({ ...cue, id: cue.id ?? index + 1 }));
        cues.splice(0, cues.length, ...next);
        normalizeCueLayout();
        duration = Math.max(1, ...cues.map((cue) => Number(cue.end) || 0));
        viewDuration = duration;
    }
    let sourceHash = "";
    let info = null;
    let duration = Math.max(1, ...cues.map((cue) => Number(cue.end) || 0));
    let fps = 30;
    let inFrame = clamp(finiteNumber(widget(node, "preview_in")?.value, 0), 0, 10000000);
    let outFrame = clamp(finiteNumber(widget(node, "preview_out")?.value, -1), -1, 10000000);
    let viewStart = 0;
    let viewDuration = duration;
    let waveformPeaks = [];
    let waveformDuration = 0;
    let drag = null;
    let playingSelection = false;
    let previewTimer = null;
    let previewRequest = 0;
    let previewInFlight = false;
    let previewPending = false;
    let renderedPreviewKey = null;
    let previewObjectUrl = "";
    let previewBounds = null;
    let previewTransform = { x: 0, y: 0, scale: 1 };
    let localTransformActive = false;
    const style = {
        font: String(widget(node, "font")?.value || ""),
        font_size: clamp(finiteNumber(widget(node, "font_size")?.value, 30), 8, 200),
        primary_color: normalizeHex(widget(node, "primary_color")?.value, "#FFFFFF"),
        secondary_color: normalizeHex(widget(node, "secondary_color")?.value, "#FF0000"),
        gradient: boolValue(widget(node, "gradient")?.value || false),
        text_align: String(widget(node, "text_align")?.value || "center"),
        italic: boolValue(widget(node, "italic")?.value || false),
        letter_spacing: clamp(finiteNumber(widget(node, "letter_spacing")?.value, 0), -10, 50),
        position_x: normalizePosition(widget(node, "position_x")?.value, 0.5),
        position_y: normalizePosition(widget(node, "position_y")?.value, 0.88),
        outline_size: clamp(finiteNumber(widget(node, "outline_size")?.value, 2), 0, 20),
        outline_color: normalizeHex(widget(node, "outline_color")?.value, "#000000"),
        shadow_size: clamp(finiteNumber(widget(node, "shadow_size")?.value, 3), 0, 20),
        shadow_color: normalizeHex(widget(node, "shadow_color")?.value, "#000000"),
    };
    const inputs = {
        font: dialog.querySelector(".font"), font_size: dialog.querySelector(".font-size"),
        primary_color: dialog.querySelector(".primary-color"), secondary_color: dialog.querySelector(".secondary-color"), gradient: dialog.querySelector(".gradient"),
        text_align: dialog.querySelector(".text-align"), position_x: dialog.querySelector(".position-x"), position_y: dialog.querySelector(".position-y"),
        italic: dialog.querySelector(".italic"), letter_spacing: dialog.querySelector(".letter-spacing"),
        outline_size: dialog.querySelector(".outline-size"), outline_color: dialog.querySelector(".outline-color"),
        shadow_size: dialog.querySelector(".shadow-size"), shadow_color: dialog.querySelector(".shadow-color"),
    };
    function imageContentRect() {
        const rect = previewWrap.getBoundingClientRect();
        const imageWidth = Number(info?.width || overlayImage.naturalWidth || 0);
        const imageHeight = Number(info?.height || overlayImage.naturalHeight || 0);
        if (!imageWidth || !imageHeight) return null;
        const scale = Math.min(rect.width / imageWidth, rect.height / imageHeight);
        const width = imageWidth * scale;
        const height = imageHeight * scale;
        return { left: rect.left + (rect.width - width) / 2, top: rect.top + (rect.height - height) / 2, width, height, scale };
    }
    function applyPreviewTransform(x, y, scale = 1) {
        previewTransform = { x, y, scale };
        const transform = `translate3d(${x}px,${y}px,0) scale(${scale})`;
        overlayImage.style.transform = transform;
        interactionBox.style.transform = transform;
    }
    function clearPreviewTransform() {
        previewTransform = { x: 0, y: 0, scale: 1 };
        overlayImage.style.transform = "";
        interactionBox.style.transform = "";
    }
    function updateOverlayClip() {
        if (!overlayClip) return;
        const content = imageContentRect();
        if (!content) {
            overlayClip.style.clipPath = "";
            return;
        }
        const previewRect = previewWrap.getBoundingClientRect();
        const left = Math.max(0, content.left - previewRect.left);
        const top = Math.max(0, content.top - previewRect.top);
        const right = Math.max(0, previewRect.right - (content.left + content.width));
        const bottom = Math.max(0, previewRect.bottom - (content.top + content.height));
        overlayClip.style.clipPath = `inset(${top}px ${right}px ${bottom}px ${left}px)`;
    }
    function updateInteractionBox() {
        updateOverlayClip();
        if (!previewBounds) { interactionBox.style.display = "none"; return; }
        const content = imageContentRect();
        if (!content) return;
        interactionBox.style.display = "block";
        interactionBox.style.left = `${content.left - previewWrap.getBoundingClientRect().left + previewBounds.left * content.scale}px`;
        interactionBox.style.top = `${content.top - previewWrap.getBoundingClientRect().top + previewBounds.top * content.scale}px`;
        interactionBox.style.width = `${Math.max(12, previewBounds.width * content.scale)}px`;
        interactionBox.style.height = `${Math.max(12, previewBounds.height * content.scale)}px`;
        interactionBox.style.transformOrigin = "center center";
        overlayImage.style.transformOrigin = `${content.left - previewWrap.getBoundingClientRect().left + (previewBounds.left + previewBounds.width / 2) * content.scale}px ${content.top - previewWrap.getBoundingClientRect().top + (previewBounds.top + previewBounds.height / 2) * content.scale}px`;
    }
    function activePreviewState() {
        const frame = currentFrame();
        const currentTime = frame / Math.max(0.001, Number(fps) || 30);
        const active = cues.filter((cue) => Number(cue.start) <= currentTime && currentTime < Number(cue.end) && String(cue.text || "").trim());
        const width = Number(info?.proxy_required && info?.proxy_width ? info.proxy_width : info?.width) || 0;
        const height = Number(info?.proxy_required && info?.proxy_height ? info.proxy_height : info?.height) || 0;
        const previewFps = Number(info?.fps) || Number(fps) || 30;
        const key = JSON.stringify({
            active: active.map((cue) => ({ id: cue.id, start: cue.start, end: cue.end, text: cue.text })),
            style,
            width,
            height,
            fps: previewFps,
        });
        return { frame, key, active, width, height, previewFps };
    }
    function clearRenderedOverlay() {
        previewBounds = null;
        if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
        previewObjectUrl = "";
        overlayImage.removeAttribute("src");
        overlayImage.style.display = "none";
        clearPreviewTransform();
        interactionBox.style.display = "none";
    }
    async function renderPreviewFrame() {
        const preview = activePreviewState();
        if (preview.key === renderedPreviewKey) return;
        if (previewInFlight) {
            previewPending = true;
            return;
        }
        previewInFlight = true;
        const requestId = ++previewRequest;
        if (!preview.active.length) {
            clearRenderedOverlay();
            renderedPreviewKey = preview.key;
            previewInFlight = false;
            if (previewPending) {
                previewPending = false;
                schedulePreview();
            }
            return;
        }
        const response = await api.fetchApi("/cinestyle/video-subtitle-preview", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                node_id: String(node.id ?? ""),
                video_filename: filename,
                frame: preview.frame,
                preview_width: preview.width,
                preview_height: preview.height,
                preview_fps: preview.previewFps,
                cues,
                style,
            }),
        }).catch(() => null);
        try {
            if (!response || requestId !== previewRequest) return;
            if (!response.ok) {
                clearRenderedOverlay();
                renderedPreviewKey = preview.key;
                status.textContent = response.status === 404
                    ? SOURCE_VIDEO_REQUIRED_MESSAGE
                    : "Pillow subtitle preview is unavailable.";
                return;
            }
            const blob = await response.blob();
            if (requestId !== previewRequest) return;
            try { previewBounds = JSON.parse(response.headers.get("X-CineStyle-Subtitle-Bounds") || "null"); } catch (_) { previewBounds = null; }
            if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
            previewObjectUrl = URL.createObjectURL(blob);
            overlayImage.onload = () => {
                if (requestId !== previewRequest) return;
                clearPreviewTransform();
                updateInteractionBox();
            };
            overlayImage.src = previewObjectUrl;
            overlayImage.style.display = "block";
            updateInteractionBox();
            renderedPreviewKey = preview.key;
            status.textContent = "Pillow preview";
        } finally {
            previewInFlight = false;
            if (previewPending) {
                previewPending = false;
                if (activePreviewState().key !== renderedPreviewKey) schedulePreview();
            }
        }
    }
    function schedulePreview() {
        if (previewTimer) clearTimeout(previewTimer);
        previewTimer = setTimeout(() => { previewTimer = null; renderPreviewFrame(); }, 60);
    }
    function updateParamOutput(name, value) {
        const output = dialog.querySelector(`[data-subtitle-output="${name}"]`);
        if (output) output.textContent = name === "position_x" || name === "position_y"
            ? normalizePosition(value, name === "position_x" ? 0.5 : 0.88).toFixed(2)
            : String(value ?? "");
    }
    function updateColorOutput(name, value) {
        const output = dialog.querySelector(`.${name.replace("_color", "-color")}-hex`);
        if (output) output.textContent = normalizeHex(value, COLOR_DEFAULTS[name]);
    }
    for (const [key, input] of Object.entries(inputs)) {
        if (input.type === "checkbox") input.checked = boolValue(style[key]); else input.value = style[key];
        updateParamOutput(key, style[key]);
        if (Object.prototype.hasOwnProperty.call(COLOR_DEFAULTS, key)) updateColorOutput(key, style[key]);
        input.addEventListener("input", () => {
            style[key] = input.type === "checkbox" ? input.checked : input.value;
            if (Object.prototype.hasOwnProperty.call(COLOR_DEFAULTS, key)) { style[key] = normalizeHex(style[key], COLOR_DEFAULTS[key]); input.value = style[key]; updateColorOutput(key, style[key]); }
            if (key === "position_x" || key === "position_y") {
                style[key] = normalizePosition(style[key], key === "position_x" ? 0.5 : 0.88);
                input.value = style[key];
            }
            updateParamOutput(key, style[key]);
            updateOverlay();
        });
    }

    function rangeStartSeconds() { return inFrame / fps; }
    function rangeEndSeconds() { return (outFrame < 0 ? duration : (outFrame + 1) / fps); }
    function beginRangeMarkerDrag(mode, event) {
        event.preventDefault();
        event.stopPropagation();
        const move = (moveEvent) => {
            const rect = viewport.getBoundingClientRect();
            const ratio = clamp((moveEvent.clientX - rect.left) / rect.width, 0, 1);
            const frame = Math.round((viewStart + ratio * viewDuration) * fps);
            const maxFrame = Math.max(1, Math.round(duration * fps) - 1);
            if (mode === "in") inFrame = clamp(frame, 0, Math.max(0, outFrame - 1));
            else outFrame = clamp(frame, Math.min(maxFrame, inFrame + 1), maxFrame);
            renderTimeline();
        };
        const up = () => {
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", up);
            window.removeEventListener("pointercancel", up);
        };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
        window.addEventListener("pointercancel", up);
        move(event);
    }
    function jumpToMarkedFrame(frame) {
        video.pause();
        setFrame(frame);
    }
    inFrameButton.addEventListener("click", () => jumpToMarkedFrame(inFrame));
    outFrameButton.addEventListener("click", () => jumpToMarkedFrame(outFrame < 0 ? Math.max(0, Math.round(duration * fps) - 1) : outFrame));
    rangeBand.querySelector(".in").addEventListener("pointerdown", (event) => beginRangeMarkerDrag("in", event));
    rangeBand.querySelector(".out").addEventListener("pointerdown", (event) => beginRangeMarkerDrag("out", event));
    function updateOverlay() {
        if (localTransformActive) return;
        schedulePreview();
    }
    function updateReadout() { const now = video.currentTime || 0; current.textContent = formatTime(now); range.textContent = `In ${formatTime(rangeStartSeconds())}  -  Out ${formatTime(rangeEndSeconds())}`; durationLabel.textContent = formatTime(duration); }
    function drawAudioWaveform() {
        if (!waveformCanvas) return;
        const width = Math.max(1, Math.round(waveformCanvas.clientWidth));
        const height = Math.max(1, Math.round(waveformCanvas.clientHeight));
        const ratio = window.devicePixelRatio || 1;
        if (waveformCanvas.width !== Math.round(width * ratio) || waveformCanvas.height !== Math.round(height * ratio)) {
            waveformCanvas.width = Math.round(width * ratio);
            waveformCanvas.height = Math.round(height * ratio);
        }
        const context = waveformCanvas.getContext("2d");
        if (!context) return;
        context.setTransform(ratio, 0, 0, ratio, 0, 0);
        context.clearRect(0, 0, width, height);
        context.strokeStyle = "rgba(210,216,223,.24)";
        context.lineWidth = 1;
        context.beginPath();
        context.moveTo(0, height / 2 + 0.5);
        context.lineTo(width, height / 2 + 0.5);
        context.stroke();
        if (!waveformPeaks.length || waveformDuration <= 0 || viewDuration <= 0) return;
        const start = Math.max(0, Math.floor((viewStart / waveformDuration) * waveformPeaks.length));
        const end = Math.min(waveformPeaks.length, Math.ceil(((viewStart + viewDuration) / waveformDuration) * waveformPeaks.length));
        if (end <= start) return;
        context.fillStyle = "rgba(210,216,223,.78)";
        for (let x = 0; x < width; x += 1) {
            const index = Math.min(end - 1, start + Math.floor((x / width) * (end - start)));
            const amplitude = Math.max(1, Math.round(Math.min(1, Number(waveformPeaks[index]) || 0) * (height * 0.44)));
            context.fillRect(x, Math.round(height / 2 - amplitude), 1, amplitude * 2);
        }
    }
    function cuePosition(cue) { const start = ((cue.start - viewStart) / viewDuration) * 100; const width = ((cue.end - cue.start) / viewDuration) * 100; return { left: `${start}%`, width: `${Math.max(0.35, width)}%` }; }
    let contextMenu = null;
    let contextMenuOutsidePointer = null;
    function closeContextMenu() {
        contextMenu?.remove();
        contextMenu = null;
        if (contextMenuOutsidePointer) {
            window.removeEventListener("pointerdown", contextMenuOutsidePointer, true);
            contextMenuOutsidePointer = null;
        }
    }
    function showContextMenu(event, items) {
        closeContextMenu();
        contextMenu = document.createElement("div");
        contextMenu.className = "cs-subtitle-context-menu";
        for (const item of items) {
            const button = document.createElement("button");
            button.type = "button";
            button.textContent = item.label;
            button.addEventListener("click", () => { closeContextMenu(); item.action(); });
            contextMenu.append(button);
        }
        dialog.append(contextMenu);
        contextMenu.style.left = `${Math.min(event.clientX, window.innerWidth - 160)}px`;
        contextMenu.style.top = `${Math.min(event.clientY, window.innerHeight - 100)}px`;
        contextMenuOutsidePointer = (pointerEvent) => {
            if (!contextMenu?.contains(pointerEvent.target)) closeContextMenu();
        };
        setTimeout(() => {
            if (contextMenu) window.addEventListener("pointerdown", contextMenuOutsidePointer, true);
        }, 0);
    }
    function cueToSrt(cue) {
        return cuesToSrt([{ ...cue, start: Number(cue.start) || 0, end: Number(cue.end) || 0 }]);
    }
    function nodeClipboard() { return NODE_SUBTITLE_CLIPBOARDS.get(node) || ""; }
    function copyCueToClipboard(cue, removeAfterCopy = false) {
        NODE_SUBTITLE_CLIPBOARDS.set(node, cueToSrt(cue));
        if (removeAfterCopy) {
            const index = cues.indexOf(cue);
            if (index >= 0) cues.splice(index, 1);
            renderTimeline();
            updateOverlay();
        }
        status.textContent = removeAfterCopy ? "字幕已剪切到本节点剪贴板" : "字幕已复制到本节点剪贴板";
    }
    function cueNeighbors(cue) {
        const index = cues.indexOf(cue);
        return { previous: index > 0 ? cues[index - 1] : null, next: index >= 0 && index + 1 < cues.length ? cues[index + 1] : null };
    }
    function placeCueInNearestGap(cue, requestedStart, requestedLength) {
        const others = cues.filter((item) => item !== cue).sort((a, b) => Number(a.start) - Number(b.start));
        const gaps = [];
        let lower = 0;
        for (const other of others) {
            const upper = Math.max(lower, Number(other.start) || lower);
            if (upper - lower >= 0.05) gaps.push({ lower, upper });
            lower = Math.max(lower, Number(other.end) || lower);
        }
        if (duration - lower >= 0.05) gaps.push({ lower, upper: duration });
        if (!gaps.length) return;
        let best = null;
        for (const gap of gaps) {
            const length = Math.min(Math.max(0.05, requestedLength), gap.upper - gap.lower);
            const start = clamp(requestedStart, gap.lower, Math.max(gap.lower, gap.upper - length));
            const distance = Math.abs(start - requestedStart);
            if (!best || distance < best.distance) best = { start, length, distance };
        }
        if (!best) return;
        cue.start = best.start;
        cue.end = best.start + best.length;
        cues.sort((a, b) => Number(a.start) - Number(b.start));
    }
    function insertionGap(seconds) {
        normalizeCueLayout();
        const requested = Math.max(0, Number(seconds) || 0);
        let previous = null;
        let next = null;
        for (const cue of cues) {
            if (cue.end <= requested) previous = cue;
            else if (cue.start >= requested) { next = cue; break; }
        }
        const lower = previous?.end ?? 0;
        const upper = next?.start ?? duration;
        const start = clamp(requested, lower, Math.max(lower, upper));
        return { start, end: Math.max(start, upper), available: Math.max(0, upper - start) };
    }
    function insertPastedSubtitle(text, seconds) {
        const parsed = parseSrt(text);
        const fallbackText = String(text || "").trim();
        if (!parsed.length && !fallbackText) return;
        const gap = insertionGap(seconds);
        if (gap.available < 0.05) return;
        const startAt = gap.start;
        const available = gap.available;
        if (!parsed.length) {
            const end = startAt + Math.min(2, available);
            cues.push({ id: cues.reduce((max, item) => Math.max(max, Number(item.id) || 0), 0) + 1, start: startAt, end, text: fallbackText });
        } else {
            const firstStart = Math.min(...parsed.map((cue) => cue.start));
            const lastEnd = Math.max(...parsed.map((cue) => cue.end));
            const sourceDuration = Math.max(0.05, lastEnd - firstStart);
            const scale = Math.min(1, available / sourceDuration);
            const idStart = cues.reduce((max, item) => Math.max(max, Number(item.id) || 0), 0) + 1;
            parsed.forEach((cue, index) => {
                const start = startAt + (cue.start - firstStart) * scale;
                const end = Math.min(startAt + available, startAt + (cue.end - firstStart) * scale);
                if (end > start && cue.text) cues.push({ id: idStart + index, start, end, text: cue.text });
            });
        }
        normalizeCueLayout();
        renderTimeline();
        updateOverlay();
    }
    function pasteSubtitleFromClipboard(seconds) {
        const clipboard = nodeClipboard();
        if (!clipboard.trim()) {
            status.textContent = "本节点剪贴板中没有可用字幕";
            return;
        }
        insertPastedSubtitle(clipboard, seconds);
    }
    function showTrackContextMenu(event, seconds) {
        const items = [{ label: "新增字幕", action: () => { void addCueAt(seconds); } }];
        if (nodeClipboard().trim()) items.push({ label: "粘贴字幕", action: () => pasteSubtitleFromClipboard(seconds) });
        showContextMenu(event, items);
    }
    async function editCueText(cue, title = "Edit subtitle") {
        const value = await textEditor.open({ title, value: cue.text, allowEmpty: false });
        if (value === null) return false;
        const text = String(value).trim();
        if (!text) return false;
        cue.text = text;
        return true;
    }
    async function addCueAt(seconds) {
        const gap = insertionGap(seconds);
        if (gap.available < 0.05) return;
        const start = gap.start;
        const available = gap.available;
        const end = start + Math.min(2, available);
        const cue = { id: cues.reduce((max, item) => Math.max(max, Number(item.id) || 0), 0) + 1, start, end, text: "" };
        if (!await editCueText(cue, "New subtitle")) return;
        cues.push(cue);
        normalizeCueLayout();
        renderTimeline();
        updateOverlay();
    }
    function renderTimeline() {
        viewDuration = clamp(viewDuration, Math.min(duration, 0.5), duration);
        viewStart = clamp(viewStart, 0, Math.max(0, duration - viewDuration));
        axis.innerHTML = "";
        const tickCount = Math.max(2, Math.min(12, Math.round(viewport.clientWidth / 100)));
        for (let i = 0; i <= tickCount; i++) { const span = document.createElement("span"); span.style.left = `${(i / tickCount) * 100}%`; span.textContent = formatTime(viewStart + (viewDuration * i / tickCount)); axis.append(span); }
        body.innerHTML = "";
        for (const cue of cues) {
            if (cue.end < viewStart || cue.start > viewStart + viewDuration) continue;
            const item = document.createElement("div"); item.className = "cs-subtitle-cue"; item.dataset.id = String(cue.id); Object.assign(item.style, cuePosition(cue));
            item.innerHTML = `<span class="cs-subtitle-cue-handle in"></span><span class="cs-subtitle-cue-label"></span><span class="cs-subtitle-cue-handle out"></span>`;
            item.querySelector(".cs-subtitle-cue-label").textContent = cue.text.replace(/\n/g, " ");
            item.addEventListener("pointerdown", (event) => beginCueDrag(cue, event));
            item.addEventListener("dblclick", async (event) => { event.stopPropagation(); if (await editCueText(cue)) { renderTimeline(); updateOverlay(); } });
            item.querySelector(".in").addEventListener("pointerdown", (event) => beginCueEdge(cue, "in", event));
            item.querySelector(".out").addEventListener("pointerdown", (event) => beginCueEdge(cue, "out", event));
            body.append(item);
        }
        const rangeStart = (rangeStartSeconds() - viewStart) / viewDuration;
        const rangeEnd = (rangeEndSeconds() - viewStart) / viewDuration;
        const visibleStart = clamp(Math.min(rangeStart, rangeEnd), 0, 1);
        const visibleEnd = clamp(Math.max(rangeStart, rangeEnd), 0, 1);
        rangeBand.style.display = visibleEnd > visibleStart ? "block" : "none";
        rangeBand.style.left = `${visibleStart * 100}%`;
        rangeBand.style.width = `${Math.max(0, (visibleEnd - visibleStart) * 100)}%`;
        rangeBand.querySelector(".in").style.display = rangeStart >= 0 && rangeStart <= 1 ? "block" : "none";
        rangeBand.querySelector(".out").style.display = rangeEnd >= 0 && rangeEnd <= 1 ? "block" : "none";
        const ratio = duration ? ((video.currentTime || 0) - viewStart) / viewDuration : 0;
        pointer.style.left = `${clamp(ratio, 0, 1) * 100}%`;
        inFrameButton.textContent = String(inFrame);
        outFrameButton.textContent = String(outFrame < 0 ? Math.max(0, Math.round(duration * fps) - 1) : outFrame);
        drawAudioWaveform(); updateReadout(); updateOverlay();
    }
    function secondsAtEvent(event) { const rect = body.getBoundingClientRect(); return viewStart + clamp((event.clientX - rect.left) / rect.width, 0, 1) * viewDuration; }
    function handleTimelineContextMenu(event) {
        const cueElement = event.target.closest?.(".cs-subtitle-cue");
        const trackElement = event.target.closest?.(".cs-subtitle-track-subtitles");
        if (!cueElement && !trackElement) return;
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation?.();
        if (cueElement) {
            const cue = cues.find((item) => String(item.id) === String(cueElement.dataset.id));
            if (!cue) return;
            showContextMenu(event, [
                { label: "编辑", action: async () => { if (await editCueText(cue)) { renderTimeline(); updateOverlay(); } } },
                { label: "删除", action: () => { const index = cues.indexOf(cue); if (index >= 0) cues.splice(index, 1); renderTimeline(); updateOverlay(); } },
                { label: "复制到剪贴板", action: () => copyCueToClipboard(cue) },
                { label: "剪切到剪贴板", action: () => copyCueToClipboard(cue, true) },
            ]);
            return;
        }
        void showTrackContextMenu(event, secondsAtEvent(event));
    }
    let rightPointerHandled = false;
    const handleTimelinePointerDown = (event) => {
        if (event.button !== 2 || !dialog.contains(event.target)) return;
        const target = event.target.closest?.(".cs-subtitle-cue, .cs-subtitle-track-subtitles");
        if (!target) return;
        rightPointerHandled = true;
        handleTimelineContextMenu(event);
    };
    const handleWindowContextMenu = (event) => {
        if (!dialog.contains(event.target)) return;
        if (rightPointerHandled) {
            rightPointerHandled = false;
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation?.();
            return;
        }
        handleTimelineContextMenu(event);
    };
    window.addEventListener("pointerdown", handleTimelinePointerDown, true);
    window.addEventListener("contextmenu", handleWindowContextMenu, true);
    function beginCueDrag(cue, event) {
        if (event.target.classList.contains("cs-subtitle-cue-handle")) return;
        event.preventDefault(); const origin = secondsAtEvent(event); const start = cue.start; const end = cue.end; const length = Math.max(0.05, end - start); drag = { cue, mode: "move", origin, start, end, length };
        const move = (moveEvent) => {
            const delta = secondsAtEvent(moveEvent) - drag.origin;
            placeCueInNearestGap(cue, drag.start + delta, drag.length);
            renderTimeline();
        };
        const up = () => { drag = null; window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
        window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
    }
    function beginCueEdge(cue, mode, event) {
        event.preventDefault(); event.stopPropagation(); const { previous, next } = cueNeighbors(cue); drag = { cue, mode, previous, next };
        const move = (moveEvent) => {
            const time = secondsAtEvent(moveEvent);
            if (mode === "in") cue.start = clamp(Math.min(time, cue.end - 0.05), previous?.end ?? 0, cue.end - 0.05);
            else cue.end = clamp(Math.max(time, cue.start + 0.05), cue.start + 0.05, next?.start ?? duration);
            renderTimeline();
        };
        const up = () => { drag = null; window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
        window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
    }
    function setFrame(frame) { video.currentTime = clamp(frame / fps, 0, duration); renderTimeline(); }
    function currentFrame() { return Math.round((video.currentTime || 0) * fps); }
    function normalizeRange() { const max = Math.max(0, Math.round(duration * fps) - 1); inFrame = clamp(Math.round(inFrame), 0, max); outFrame = outFrame < 0 ? max : clamp(Math.round(outFrame), 0, max); if (outFrame <= inFrame) outFrame = Math.min(max, inFrame + 1); }
    function close() { void setTimelineOpen(node, false).catch(() => {}); video.pause(); closeContextMenu(); textEditor.destroy(); if (previewTimer) clearTimeout(previewTimer); if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl); window.removeEventListener("resize", updateInteractionBox); window.removeEventListener("resize", drawAudioWaveform); window.removeEventListener("pointerdown", handleTimelinePointerDown, true); window.removeEventListener("contextmenu", handleWindowContextMenu, true); dialog.close(); dialog.remove(); }

    dialog.querySelector(".set-in").addEventListener("click", () => { inFrame = currentFrame(); normalizeRange(); renderTimeline(); });
    dialog.querySelector(".set-out").addEventListener("click", () => { outFrame = currentFrame(); normalizeRange(); renderTimeline(); });
    dialog.querySelector(".back").addEventListener("click", () => setFrame(currentFrame() - 1));
    dialog.querySelector(".forward").addEventListener("click", () => setFrame(currentFrame() + 1));
    dialog.querySelector(".play").addEventListener("click", () => { if (!video.paused) { video.pause(); return; } normalizeRange(); if (video.currentTime < rangeStartSeconds() || video.currentTime >= rangeEndSeconds()) video.currentTime = rangeStartSeconds(); playingSelection = true; video.play().catch(() => { playingSelection = false; }); });
    dialog.querySelector(".move-reset").addEventListener("click", () => { setStyleInput("position_x", 0.5); setStyleInput("position_y", 0.88); renderTimeline(); updateOverlay(); });
    function setStyleInput(name, value) {
        if (name === "position_x" || name === "position_y") value = normalizePosition(value, name === "position_x" ? 0.5 : 0.88);
        if (Object.prototype.hasOwnProperty.call(COLOR_DEFAULTS, name)) value = normalizeHex(value, COLOR_DEFAULTS[name]);
        style[name] = value;
        const input = inputs[name];
        if (input) input.value = value;
        updateParamOutput(name, value);
    }
    const paramDefaults = { font_size: 30, letter_spacing: 0, position_x: 0.5, position_y: 0.88, outline_size: 2, shadow_size: 3 };
    dialog.querySelectorAll(".cs-subtitle-param-reset").forEach((button) => {
        button.addEventListener("click", () => { const name = button.dataset.reset; setStyleInput(name, paramDefaults[name]); updateOverlay(); });
    });
    function beginTextDrag(event) {
        event.preventDefault();
        previewRequest += 1;
        const content = imageContentRect();
        if (!content) return;
        localTransformActive = true;
        const startX = event.clientX;
        const startY = event.clientY;
        const originX = style.position_x;
        const originY = style.position_y;
        let pendingX = originX;
        let pendingY = originY;
        const move = (moveEvent) => {
            const dx = moveEvent.clientX - startX;
            const dy = moveEvent.clientY - startY;
            pendingY = clamp(originY + dy / content.height, 0, 1);
            pendingX = clamp(originX + dx / content.width, 0, 1);
            setStyleInput("position_x", pendingX);
            setStyleInput("position_y", pendingY);
            applyPreviewTransform(dx, dy, 1);
        };
        const up = () => {
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", up);
            window.removeEventListener("pointercancel", up);
            setStyleInput("position_x", pendingX);
            setStyleInput("position_y", pendingY);
            localTransformActive = false;
            updateOverlay();
        };
        window.addEventListener("pointermove", move); window.addEventListener("pointerup", up); window.addEventListener("pointercancel", up);
    }
    function beginTextResize(event, handle) {
        event.preventDefault();
        event.stopPropagation();
        previewRequest += 1;
        localTransformActive = true;
        const box = interactionBox.getBoundingClientRect();
        const isRightHandle = handle.classList.contains("ne") || handle.classList.contains("se");
        const isBottomHandle = handle.classList.contains("sw") || handle.classList.contains("se");
        const anchorX = isRightHandle ? box.left : box.right;
        const anchorY = isBottomHandle ? box.top : box.bottom;
        const startDistance = Math.max(8, Math.max(Math.abs(event.clientX - anchorX), Math.abs(event.clientY - anchorY)));
        const originSize = finiteNumber(style.font_size, 30);
        const move = (moveEvent) => {
            const distance = Math.max(8, Math.max(Math.abs(moveEvent.clientX - anchorX), Math.abs(moveEvent.clientY - anchorY)));
            const nextSize = clamp(Math.round(originSize * distance / startDistance), 8, 200);
            setStyleInput("font_size", nextSize);
            applyPreviewTransform(0, 0, nextSize / originSize);
        };
        const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); window.removeEventListener("pointercancel", up); localTransformActive = false; updateOverlay(); };
        window.addEventListener("pointermove", move); window.addEventListener("pointerup", up); window.addEventListener("pointercancel", up);
    }
    interactionBox.addEventListener("pointerdown", (event) => {
        const handle = event.target.closest?.(".cs-subtitle-resize-handle");
        if (handle) beginTextResize(event, handle);
        else beginTextDrag(event);
    });
    window.addEventListener("resize", updateInteractionBox);
    window.addEventListener("resize", drawAudioWaveform);
    dialog.querySelector(".cs-subtitle-pointer-row").addEventListener("pointerdown", (event) => { const row = event.currentTarget; const move = (moveEvent) => { const rect = row.getBoundingClientRect(); setFrame(Math.round(clamp((moveEvent.clientX - rect.left) / rect.width, 0, 1) * Math.max(0, Math.round(duration * fps) - 1))); }; const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); }; window.addEventListener("pointermove", move); window.addEventListener("pointerup", up); move(event); });
    video.addEventListener("timeupdate", () => { if (playingSelection && video.currentTime >= rangeEndSeconds()) { video.pause(); video.currentTime = rangeEndSeconds(); playingSelection = false; } renderTimeline(); });
    video.addEventListener("pause", () => { playingSelection = false; dialog.querySelector(".play").textContent = "Play"; });
    video.addEventListener("play", () => { dialog.querySelector(".play").textContent = "Pause"; });
    dialog.querySelector(".cs-subtitle-close").addEventListener("click", close); dialog.querySelector(".cancel").addEventListener("click", close); dialog.addEventListener("cancel", close);
    dialog.querySelector(".apply").addEventListener("click", async () => {
        normalizeRange();
        setWidgetValue(node, "preview_in", inFrame); setWidgetValue(node, "preview_out", outFrame);
        setWidgetValue(node, "position_x", style.position_x); setWidgetValue(node, "position_y", style.position_y);
        for (const [key, input] of Object.entries(inputs)) setWidgetValue(node, key, input.type === "checkbox" ? input.checked : input.value);
        const editedSrt = cuesToSrt(cues);
        setWidgetValue(node, "edited_srt", editedSrt);
        try {
            await saveCachedSrt(node, sourceHash, editedSrt);
            node.graph?.setDirtyCanvas(true, true); close();
        } catch (error) {
            status.textContent = error.message;
        }
    });
    async function loadAudioWaveform() {
        const result = await fetchAudioWaveform(node, filename).catch(() => null);
        waveformPeaks = result?.peaks || [];
        waveformDuration = Number(result?.duration) || duration;
        drawAudioWaveform();
    }
    fetchFonts().then((fonts) => { inputs.font.innerHTML = ""; for (const font of fonts) { const option = document.createElement("option"); option.value = font; option.textContent = font; inputs.font.append(option); } if (!style.font && fonts.length) style.font = fonts[0]; inputs.font.value = style.font; updateOverlay(); }).catch(() => {});
    const initialize = async () => {
        setLoading("Reading subtitle cache...");
        cachedSrt = await fetchCachedSrt(node).catch(() => null);
        sourceHash = cachedSrt?.sourceHash || (externalSrt ? await srtSourceHash(externalSrt).catch(() => "") : "");
        if (!persistedSrt && cachedSrt?.srt) {
            sourceSrt = cachedSrt.srt;
            replaceCues(sourceSrt);
        }
        setLoading("Preparing video preview cache...");
        cachedProxy = await fetchCachedProxy(node, filename).catch(() => null);
        dialog.querySelector(".cs-subtitle-file").textContent = cachedProxy?.label || filename || "Subtitle preview cache";
        if (cachedProxy) {
            setLoading("Loading preview video...");
            info = cachedProxy.info || {};
            fps = Number(info.fps) || 30;
            duration = Number(info.duration) || duration;
            outFrame = outFrame < 0 ? Math.max(0, Math.round(duration * fps) - 1) : outFrame;
            viewDuration = duration;
            const useCachedPlayback = () => { video.src = cachedProxy.url; video.load(); normalizeRange(); renderTimeline(); };
            useCachedPlayback();
        } else if (filename) {
            setLoading("Reading video information...");
            const result = await fetchInfo(filename);
            info = result;
            fps = Number(result.fps) || 30;
            duration = Number(result.duration) || duration;
            outFrame = outFrame < 0 ? Math.max(0, Math.round(duration * fps) - 1) : outFrame;
            viewDuration = duration;
            video.src = result.proxy_required ? proxyVideoUrl(filename, result.proxy_threshold, result.proxy_size) : videoUrl(filename);
            video.load();
            normalizeRange();
            renderTimeline();
        } else {
            status.textContent = SOURCE_VIDEO_REQUIRED_MESSAGE;
            renderTimeline();
        }
        setLoading("", false);
        void loadAudioWaveform();
    };
    void initialize().catch((error) => {
        setLoading("Unable to initialize subtitle preview", false);
        status.textContent = filename ? SOURCE_VIDEO_REQUIRED_MESSAGE : (error?.message || SOURCE_VIDEO_REQUIRED_MESSAGE);
        renderTimeline();
    });
}

app.registerExtension({
    name: "CineStyle.VideoSubtitleTimeline",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_ID) return;
        const original = nodeType.prototype.onNodeCreated;
        const originalConfigure = nodeType.prototype.configure;
        const originalSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onNodeCreated = function () {
            original?.apply(this, arguments);
            const button = this.addWidget("button", "Edit Timeline", "", () => { openTimeline(this).catch((error) => app.canvas?.prompt?.(error.message, "")); });
            button.name = "Edit Timeline"; button.label = "Edit Timeline"; button.options = { ...(button.options || {}), serialize: false };
            this.setSize?.([410, Math.max(380, this.computeSize?.()[1] || 380)]);
        };
        nodeType.prototype.configure = function (info) {
            const configured = configureSubtitleWidgetValues(this, info);
            originalConfigure?.call(this, configured);
            applySubtitleWidgetValuesByName(this, info);
        };
        nodeType.prototype.onSerialize = function () {
            originalSerialize?.apply(this, arguments);
            const data = arguments[0];
            if (data && typeof data === "object") data.widgets_values = serializeSubtitleWidgetValues(this);
        };
    },
});
