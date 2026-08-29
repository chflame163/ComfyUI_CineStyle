import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import {
    connectedVideoSource,
    ensureLoaderPreviewSource,
    fetchInfo,
    prepareInputTimeline,
    sourceFrameForLocal,
} from "./video_selector_multi.js";

const NODE_ID = "CS_Color_Grade";
const NO_LUT = "None";
const STYLE_ID = "cinestyle-color-grade-style";
const CURVE_CHANNELS = ["rgb", "r", "g", "b"];
const CURVE_COLORS = { rgb: "#f4f5f7", r: "#ff5b63", g: "#48db76", b: "#4b82ff" };
const MIN_POINT_DISTANCE = 0.012;
const DEFAULT_CURVES = Object.freeze({
    version: 1,
    domain: [0, 1],
    rgb: [[0, 0], [1, 1]],
    r: [[0, 0], [1, 1]],
    g: [[0, 0], [1, 1]],
    b: [[0, 0], [1, 1]],
});
const COLOR_PARAMS = [
    ["lut_strength", "LUT Strength", 0, 1, 1, 0.001],
    ["color_temperature", "Color Temperature", -1, 1, 0, 0.001],
    ["tint", "Tint", -1, 1, 0, 0.001],
    ["brightness", "Brightness", -1, 1, 0, 0.001],
    ["contrast", "Contrast", -1, 1, 0, 0.001],
    ["saturation", "Saturation", 0, 10, 1, 0.001],
];
const GRADE_GROUPS = [
    { name: "offset", label: "Offset", min: -1, max: 1, defaultValue: 0, vector: "rgb_offset", neutral: [0, 0, 0], vectorMin: -1, vectorMax: 1 },
    { name: "multiply", label: "Multiply", min: 0, max: 2, defaultValue: 1, vector: "rgb_multiply", neutral: [1, 1, 1], vectorMin: 0, vectorMax: 2 },
    { name: "gamma", label: "Gamma", min: 0.000001, max: 10, defaultValue: 1, vector: "rgb_gamma", neutral: [1, 1, 1], vectorMin: 0.000001, vectorMax: 10 },
];
const PARAMS = [...COLOR_PARAMS, ...GRADE_GROUPS.map(({ name, label, min, max, defaultValue }) => [name, label, min, max, defaultValue, 0.001])];

function addStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .cs-grade-dialog{width:min(1320px,96vw);max-width:none;max-height:95vh;overflow:auto;padding:0;border:1px solid #353a43;border-radius:8px;background:#17191e;color:#e8ebef;box-shadow:0 22px 80px #000b}
      .cs-grade-dialog::backdrop{background:#050609b8}
      .cs-grade-shell{display:grid;gap:12px;padding:16px;font:13px/1.35 system-ui,sans-serif}
      .cs-grade-head,.cs-grade-row,.cs-grade-actions,.cs-grade-zoom,.cs-grade-tabs{display:flex;align-items:center;gap:7px;flex-wrap:wrap}
      .cs-grade-head{justify-content:space-between}.cs-grade-title{margin:0;font-size:17px;letter-spacing:0}.cs-grade-muted,.cs-grade-status{color:#9da5b4}
      .cs-grade-button{min-height:30px;border:1px solid #454c57;border-radius:5px;padding:5px 9px;background:#22252c;color:#f2f4f7;cursor:pointer}
      .cs-grade-button:hover{border-color:#79aee0}.cs-grade-button.active{border-color:#74b8f0;background:#2879b8}.cs-grade-close{font-size:18px;padding:2px 9px}
      .cs-grade-preview-wrap{position:relative;display:flex;justify-content:center;min-height:320px}
      .cs-grade-cache-loading{position:absolute;z-index:10;inset:0;display:flex;align-items:center;justify-content:center;padding:18px;background:#08090be8;color:#dce7f3;text-align:center}.cs-grade-cache-loading[hidden]{display:none}
      .cs-grade-viewport{position:relative;width:min(980px,100%);height:clamp(320px,44vh,560px);overflow:hidden;border:1px solid #343943;border-radius:6px;background:#08090b;--compare-position:0%;cursor:default;touch-action:none;user-select:none}
      .cs-grade-viewport.pan-ready{cursor:grab}.cs-grade-viewport.pan-active{cursor:grabbing}.cs-grade-viewport-label{position:absolute;z-index:4;top:8px;left:9px;padding:3px 6px;border-radius:4px;background:#111419cf;color:#d8dde5;font-size:12px}
      .cs-grade-image{position:absolute;inset:0;display:block;width:100%;height:100%;object-fit:contain;transform-origin:center center;transition:transform .08s linear;pointer-events:none;user-select:none;-webkit-user-drag:none}
      .cs-grade-original-clip{position:absolute;inset:0;z-index:2;overflow:hidden;pointer-events:none;clip-path:inset(0 calc(100% - var(--compare-position)) 0 0)}
      .cs-grade-divider{position:absolute;z-index:3;top:0;bottom:0;left:var(--compare-position);width:2px;transform:translateX(-1px);background:#f3f5f7;box-shadow:0 0 0 1px #1118;cursor:ew-resize;touch-action:none}
      .cs-grade-divider::before{content:"";position:absolute;top:50%;left:50%;width:24px;height:24px;transform:translate(-50%,-50%);border:2px solid #f2f4f7;border-radius:50%;background:#20232a;box-shadow:0 2px 8px #000b}
      .cs-grade-divider::after{content:"\u2194";position:absolute;top:50%;left:50%;transform:translate(-50%,-53%);font-size:14px;line-height:1;color:#f2f4f7}
      .cs-grade-zoom{justify-content:center}.cs-grade-zoom .cs-grade-button{min-height:27px;padding:3px 8px}
      .cs-grade-timeline{display:grid;grid-template-columns:auto minmax(120px,1fr) 76px auto;align-items:center;gap:8px}.cs-grade-step{display:flex;gap:5px}.cs-grade-step .cs-grade-button{width:38px;padding-inline:0}
      .cs-grade-number,.cs-grade-text,.cs-grade-vector-number{min-height:28px;box-sizing:border-box;border:1px solid #454c57;border-radius:4px;padding:4px 6px;background:#101216;color:#f2f4f7;font-variant-numeric:tabular-nums}.cs-grade-frame{width:76px}
      .cs-grade-lower{display:grid;grid-template-columns:minmax(600px,2fr) minmax(300px,1fr);gap:20px;border-top:1px solid #343943;padding-top:14px;align-items:stretch}
      .cs-grade-left{display:grid;gap:18px;min-width:0}.cs-grade-color-block{display:grid;gap:8px;width:100%}.cs-grade-top-row{display:grid;grid-template-columns:minmax(0,1fr) max-content;align-items:center;column-gap:clamp(48px,8vw,140px);width:100%}.cs-grade-lut-row{display:grid;grid-template-columns:auto minmax(180px,1fr);align-items:center;gap:8px;justify-self:start;min-width:0}.cs-grade-lut-select{width:100%;min-height:30px;box-sizing:border-box;border:1px solid #454c57;border-radius:4px;padding:4px 7px;background:#101216;color:#f2f4f7}.cs-grade-color-row{display:grid;grid-template-columns:145px minmax(140px,1fr) 82px 30px;align-items:center;gap:8px}.cs-grade-color-row input[type=range]{width:100%}.cs-grade-color-row .cs-grade-number{width:82px}.cs-grade-white-row{display:grid;grid-template-columns:auto 84px 36px 84px 30px;align-items:center;justify-self:end;gap:8px;white-space:nowrap}.cs-grade-text{width:84px;text-transform:uppercase}.cs-grade-swatch{width:36px;height:30px;box-sizing:border-box;border:1px solid #454c57;border-radius:4px;padding:2px;background:#22252c;cursor:pointer}.cs-grade-white-value{color:#f0b958;font-variant-numeric:tabular-nums}
      .cs-grade-triplet{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.cs-grade-group{display:grid;gap:9px;min-width:0}.cs-grade-group-head{display:flex;align-items:center;justify-content:space-between}.cs-grade-group-head h3,.cs-grade-curves h3{margin:0;font-size:14px;letter-spacing:0;color:#dce1e8}.cs-grade-scalar-row{display:grid;grid-template-columns:minmax(80px,1fr) 68px 30px;align-items:center;gap:6px}.cs-grade-scalar-row input[type=range]{width:100%}.cs-grade-scalar-row .cs-grade-number{width:68px}.cs-grade-reset{width:30px;padding:2px;font-size:15px}
      .cs-grade-wheel-body{display:grid;grid-template-columns:minmax(110px,1fr) 76px;align-items:center;gap:8px}.cs-grade-wheel{display:block;width:100%;aspect-ratio:1;touch-action:none;cursor:crosshair}.cs-grade-vector-values{display:grid;gap:8px}.cs-grade-vector-field{display:grid;grid-template-columns:14px 1fr;align-items:center;gap:5px}.cs-grade-vector-field[data-channel=r] label{color:#ff727a}.cs-grade-vector-field[data-channel=g] label{color:#54df7d}.cs-grade-vector-field[data-channel=b] label{color:#6094ff}.cs-grade-vector-number{width:100%;min-width:0}
      .cs-grade-curves{display:grid;grid-template-rows:auto auto minmax(0,1fr);gap:10px;min-width:0;height:100%}.cs-grade-tabs{justify-content:space-between;flex-wrap:nowrap}.cs-grade-tabset{display:flex;gap:5px}.cs-grade-tab{min-width:43px;padding-inline:8px}.cs-grade-tab[data-channel=r]{color:#ff7d83}.cs-grade-tab[data-channel=g]{color:#65e28b}.cs-grade-tab[data-channel=b]{color:#70a0ff}.cs-grade-curve-reset{margin-left:auto}
      .cs-grade-curve-wrap{position:relative;width:100%;height:100%;min-height:300px;border:1px solid #424852;border-radius:4px;background:#22252a;overflow:hidden}.cs-grade-curve{display:block;width:100%;height:100%;touch-action:none;cursor:crosshair}
      .cs-grade-actions{justify-content:flex-end}.cs-grade-status{flex:1;min-width:160px}.cs-grade-error{color:#ff939b}
      @media(max-width:980px){.cs-grade-lower{grid-template-columns:1fr}.cs-grade-color-block{max-width:none}.cs-grade-curves{height:auto}.cs-grade-curve-wrap{height:auto;aspect-ratio:1.15;min-height:340px}}
      @media(max-width:720px){.cs-grade-triplet{grid-template-columns:1fr}.cs-grade-wheel-body{grid-template-columns:minmax(150px,230px) 90px;justify-content:center}.cs-grade-color-row{grid-template-columns:120px minmax(90px,1fr) 76px 30px}.cs-grade-top-row{grid-template-columns:1fr;row-gap:10px}.cs-grade-white-row{justify-self:end}}
      @media(max-width:620px){.cs-grade-timeline{grid-template-columns:auto 1fr 76px}.cs-grade-frame-count{grid-column:1/-1}.cs-grade-preview-wrap{min-height:280px}.cs-grade-viewport{height:330px}.cs-grade-white-value{display:none}.cs-grade-white-row{grid-template-columns:auto 84px 36px 30px}.cs-grade-lut-row{grid-template-columns:auto minmax(120px,1fr)}.cs-grade-color-row{grid-template-columns:110px 1fr 70px 30px}}
    `;
    document.head.append(style);
}

function widget(node, name) { return node.widgets?.find((item) => item.name === name); }
function valueOf(node, name, fallback) { const value = widget(node, name)?.value; return value == null ? fallback : value; }
function comboOptions(node, name, fallback = []) {
    const options = widget(node, name)?.options;
    const values = Array.isArray(options?.values) ? options.values : (Array.isArray(options) ? options : fallback);
    return values.map((value) => String(value));
}
function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character]));
}
function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function roundValue(value) { return Number(clamp(Number(value) || 0, 0, 1).toFixed(6)); }
function normalizeHex(value) { const text = String(value || "").trim(); return /^#[0-9a-f]{6}$/i.test(text) ? text.toUpperCase() : null; }

function parseVec3(value, fallback) {
    let parsed = value;
    if (typeof value === "string") {
        try { parsed = JSON.parse(value); }
        catch { parsed = value.replace(/[\[\]()]/g, "").split(","); }
    }
    if (!Array.isArray(parsed) || parsed.length !== 3) return [...fallback];
    return fallback.map((item, index) => Number.isFinite(Number(parsed[index])) ? Number(parsed[index]) : item);
}

function canonicalPoints(points) {
    const sorted = points
        .map(([x, y]) => [roundValue(x), roundValue(y)])
        .sort((left, right) => left[0] - right[0]);
    const unique = [];
    for (const point of sorted) {
        if (!unique.length || point[0] - unique.at(-1)[0] > 0.000001) unique.push(point);
        else unique[unique.length - 1] = point;
    }
    return unique;
}

function parseCurves(value) {
    let parsed = value;
    if (typeof value === "string") {
        try { parsed = JSON.parse(value); } catch { parsed = null; }
    }
    const result = {};
    for (const channel of CURVE_CHANNELS) {
        const points = Array.isArray(parsed?.[channel]) ? parsed[channel] : DEFAULT_CURVES[channel];
        const valid = points
            .filter((point) => Array.isArray(point) && point.length === 2 && point.every((item) => Number.isFinite(Number(item))))
            .map(([x, y]) => [clamp(Number(x), 0, 1), clamp(Number(y), 0, 1)]);
        const canonical = canonicalPoints(valid.length >= 2 ? valid : DEFAULT_CURVES[channel]);
        result[channel] = canonical.length >= 2 ? canonical : DEFAULT_CURVES[channel].map((point) => [...point]);
    }
    return result;
}

function curvesPayload(curves) {
    return {
        version: 1,
        domain: [0, 1],
        ...Object.fromEntries(CURVE_CHANNELS.map((channel) => [channel, canonicalPoints(curves[channel])])),
    };
}

function pchipSlopes(points) {
    const n = points.length;
    const h = Array.from({ length: n - 1 }, (_, index) => points[index + 1][0] - points[index][0]);
    const delta = h.map((width, index) => (points[index + 1][1] - points[index][1]) / width);
    if (n === 2) return [delta[0], delta[0]];
    const endpoint = (h0, h1, d0, d1) => {
        let slope = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
        if (slope * d0 <= 0) slope = 0;
        else if (d0 * d1 < 0 && Math.abs(slope) > Math.abs(3 * d0)) slope = 3 * d0;
        return slope;
    };
    const slopes = Array(n).fill(0);
    slopes[0] = endpoint(h[0], h[1], delta[0], delta[1]);
    slopes[n - 1] = endpoint(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);
    for (let index = 1; index < n - 1; index += 1) {
        const left = delta[index - 1]; const right = delta[index];
        if (left === 0 || right === 0 || left * right < 0) { slopes[index] = 0; continue; }
        const w1 = 2 * h[index] + h[index - 1]; const w2 = h[index] + 2 * h[index - 1];
        slopes[index] = (w1 + w2) / (w1 / left + w2 / right);
    }
    return slopes;
}

function evaluatePchip(points, slopes, x) {
    if (x <= points[0][0]) return points[0][1];
    if (x >= points.at(-1)[0]) return points.at(-1)[1];
    let index = 0;
    while (index < points.length - 2 && x >= points[index + 1][0]) index += 1;
    const [x0, y0] = points[index]; const [x1, y1] = points[index + 1];
    const width = x1 - x0; const t = (x - x0) / width; const t2 = t * t; const t3 = t2 * t;
    return clamp(
        (2 * t3 - 3 * t2 + 1) * y0
        + (t3 - 2 * t2 + t) * width * slopes[index]
        + (-2 * t3 + 3 * t2) * y1
        + (t3 - t2) * width * slopes[index + 1],
        0,
        1,
    );
}

function drawCurveEditor(canvas, curves, activeChannel) {
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * ratio); canvas.height = Math.round(rect.height * ratio);
    const ctx = canvas.getContext("2d"); ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const width = rect.width; const height = rect.height; const pad = 14;
    const plotW = width - 2 * pad; const plotH = height - 2 * pad;
    ctx.fillStyle = "#22252a"; ctx.fillRect(0, 0, width, height);
    ctx.lineWidth = 1;
    for (let index = 0; index <= 4; index += 1) {
        const x = pad + plotW * index / 4; const y = pad + plotH * index / 4;
        ctx.strokeStyle = index === 0 || index === 4 ? "#59606a" : "#393e45";
        ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, pad + plotH); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(pad + plotW, y); ctx.stroke();
    }
    ctx.strokeStyle = "#6a717c"; ctx.strokeRect(pad, pad, plotW, plotH);
    const order = CURVE_CHANNELS.filter((channel) => channel !== activeChannel).concat(activeChannel);
    for (const channel of order) {
        const points = curves[channel]; const slopes = pchipSlopes(points);
        ctx.strokeStyle = CURVE_COLORS[channel]; ctx.globalAlpha = channel === activeChannel ? 1 : 0.25; ctx.lineWidth = channel === activeChannel ? 2.2 : 1.2;
        ctx.beginPath();
        for (let index = 0; index <= 320; index += 1) {
            const x = index / 320; const y = evaluatePchip(points, slopes, x);
            const px = pad + x * plotW; const py = pad + (1 - y) * plotH;
            if (index === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke(); ctx.globalAlpha = 1;
    }
    for (const [x, y] of curves[activeChannel]) {
        const px = pad + x * plotW; const py = pad + (1 - y) * plotH;
        ctx.fillStyle = "#202329"; ctx.strokeStyle = CURVE_COLORS[activeChannel]; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(px, py, 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    }
    canvas._curveGeometry = { pad, plotW, plotH };
}

function curvePosition(canvas, event) {
    const rect = canvas.getBoundingClientRect(); const geometry = canvas._curveGeometry;
    return {
        x: clamp((event.clientX - rect.left - geometry.pad) / geometry.plotW, 0, 1),
        y: clamp(1 - (event.clientY - rect.top - geometry.pad) / geometry.plotH, 0, 1),
    };
}

function nearestCurvePoint(canvas, points, event, radius = 12) {
    const rect = canvas.getBoundingClientRect(); const geometry = canvas._curveGeometry;
    let nearest = -1; let distance = radius;
    points.forEach(([x, y], index) => {
        const px = rect.left + geometry.pad + x * geometry.plotW;
        const py = rect.top + geometry.pad + (1 - y) * geometry.plotH;
        const candidate = Math.hypot(event.clientX - px, event.clientY - py);
        if (candidate <= distance) { distance = candidate; nearest = index; }
    });
    return nearest;
}

async function fetchCachedSource(node) {
    const nodeId = String(node?.id ?? "").trim();
    if (!nodeId) return null;
    const response = await api.fetchApi(`/cinestyle/color-grade-cache?${new URLSearchParams({ node_id: nodeId, t: String(Date.now()) })}`);
    if (response.status === 404) return null;
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Unable to read CS Color Grade preview cache");
    const info = result.info || {};
    return {
        filename: "",
        kind: "video",
        label: String(result.label || "Cached CS Color Grade input"),
        url: api.apiURL(result.video_url),
        token: String(result.token || ""),
        info,
        startFrame: 0,
        endFrame: Math.max(0, Number(info.frames || 1) - 1),
        targetFps: Number(info.fps || 24),
    };
}

function hsvToRgb(hue, saturation = 1, value = 1) {
    const h = ((hue % 1) + 1) % 1 * 6; const sector = Math.floor(h); const fraction = h - sector;
    const p = value * (1 - saturation); const q = value * (1 - fraction * saturation); const t = value * (1 - (1 - fraction) * saturation);
    return [[value, t, p], [q, value, p], [p, value, t], [p, q, value], [t, p, value], [value, p, q]][sector % 6];
}

function rgbHue(values) {
    const min = Math.min(...values); const shifted = values.map((value) => value - min); const max = Math.max(...shifted);
    if (max < 1e-8) return 0;
    const [r, g, b] = shifted.map((value) => value / max); let hue = 0;
    if (r >= g && r >= b) hue = ((g - b) / Math.max(1e-8, r)) / 6;
    else if (g >= r && g >= b) hue = (2 + (b - r) / Math.max(1e-8, g)) / 6;
    else hue = (4 + (r - g) / Math.max(1e-8, b)) / 6;
    return (hue + 1) % 1;
}

function wheelDirection(hue) {
    const rgb = hsvToRgb(hue); const mean = (rgb[0] + rgb[1] + rgb[2]) / 3;
    const centered = rgb.map((value) => value - mean); const scale = Math.max(...centered.map(Math.abs), 1e-8);
    return centered.map((value) => value / scale);
}

function drawColorWheel(canvas, hue, radius) {
    const size = Math.max(110, Math.round(canvas.clientWidth || 180)); const ratio = window.devicePixelRatio || 1;
    const pixelSize = Math.round(size * ratio);
    if (canvas.width !== pixelSize || canvas.height !== pixelSize || canvas._gradeWheelSize !== pixelSize) {
        canvas.width = pixelSize; canvas.height = pixelSize;
        const baseContext = canvas.getContext("2d"); const image = baseContext.createImageData(pixelSize, pixelSize); const center = pixelSize / 2; const maxRadius = center - 3 * ratio;
        for (let y = 0; y < pixelSize; y += 1) for (let x = 0; x < pixelSize; x += 1) {
            const dx = x - center; const dy = center - y; const distance = Math.hypot(dx, dy) / maxRadius; const offset = (y * pixelSize + x) * 4;
            if (distance > 1) { image.data[offset + 3] = 0; continue; }
            const h = (Math.atan2(dy, dx) / (2 * Math.PI) + 1) % 1; const rgb = hsvToRgb(h, distance, 1);
            image.data[offset] = Math.round(rgb[0] * 255); image.data[offset + 1] = Math.round(rgb[1] * 255); image.data[offset + 2] = Math.round(rgb[2] * 255); image.data[offset + 3] = 255;
        }
        canvas._gradeWheelBase = image;
        canvas._gradeWheelSize = pixelSize;
    }
    const ctx = canvas.getContext("2d"); ctx.putImageData(canvas._gradeWheelBase, 0, 0); ctx.save(); ctx.scale(ratio, ratio);
    const indicatorX = size / 2 + Math.cos(hue * 2 * Math.PI) * radius * (size / 2 - 3);
    const indicatorY = size / 2 - Math.sin(hue * 2 * Math.PI) * radius * (size / 2 - 3);
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(indicatorX, indicatorY, 7, 0, Math.PI * 2); ctx.stroke(); ctx.strokeStyle = "#111"; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(indicatorX, indicatorY, 9, 0, Math.PI * 2); ctx.stroke(); ctx.restore();
}

function wheelState(kind, values) {
    const neutral = kind === "rgb_offset" ? 0 : 1;
    const relative = values.map((value) => value - neutral);
    return { hue: rgbHue(relative), radius: clamp(Math.max(...relative.map(Math.abs)), 0, 1) };
}

function installInlineWheel(dialog, group, vectors, schedulePreview) {
    const wheel = dialog.querySelector(`[data-grade-wheel="${group.vector}"]`);
    const state = wheelState(group.vector, vectors[group.vector]);
    let dragging = false;
    const redraw = () => drawColorWheel(wheel, state.hue, state.radius);
    const syncFields = () => vectors[group.vector].forEach((value, index) => {
        const field = dialog.querySelector(`[data-vector="${group.vector}"][data-index="${index}"]`);
        field.value = Number(value.toFixed(6));
    });
    const updateStateFromValues = () => Object.assign(state, wheelState(group.vector, vectors[group.vector]));
    const updateFromPointer = (event) => {
        const rect = wheel.getBoundingClientRect();
        const dx = event.clientX - (rect.left + rect.width / 2);
        const dy = rect.top + rect.height / 2 - event.clientY;
        state.radius = clamp(Math.hypot(dx, dy) / Math.max(1, rect.width / 2 - 3), 0, 1);
        state.hue = (Math.atan2(dy, dx) / (2 * Math.PI) + 1) % 1;
        const direction = wheelDirection(state.hue);
        const neutral = group.vector === "rgb_offset" ? 0 : 1;
        vectors[group.vector] = direction.map((value) => clamp(neutral + value * state.radius, group.vectorMin, group.vectorMax));
        syncFields(); redraw(); schedulePreview();
    };
    wheel.addEventListener("pointerdown", (event) => { if (event.button !== 0) return; dragging = true; wheel.setPointerCapture?.(event.pointerId); updateFromPointer(event); });
    wheel.addEventListener("pointermove", (event) => { if (dragging) updateFromPointer(event); });
    const stop = (event) => { dragging = false; wheel.releasePointerCapture?.(event.pointerId); };
    wheel.addEventListener("pointerup", stop); wheel.addEventListener("pointercancel", stop);
    dialog.querySelectorAll(`[data-vector="${group.vector}"]`).forEach((field) => field.addEventListener("input", () => {
        const value = Number(field.value); if (!Number.isFinite(value)) return;
        vectors[group.vector][Number(field.dataset.index)] = clamp(value, group.vectorMin, group.vectorMax);
        updateStateFromValues(); redraw(); schedulePreview();
    }));
    redraw();
    return { redraw, reset: () => { vectors[group.vector] = [...group.neutral]; updateStateFromValues(); syncFields(); redraw(); } };
}

function clampPan(dialog) {
    const pan = dialog._gradePan || { x: 0, y: 0 }; const viewport = dialog.querySelector(".cs-grade-viewport"); const image = dialog.querySelector(".cs-grade-result"); const zoom = Number(dialog._gradeZoom || 1);
    if (!viewport || !image || zoom <= 1) { dialog._gradePan = { x: 0, y: 0 }; return; }
    const maxX = Math.max(0, (image.clientWidth * zoom - viewport.clientWidth) / 2); const maxY = Math.max(0, (image.clientHeight * zoom - viewport.clientHeight) / 2);
    dialog._gradePan = { x: clamp(pan.x, -maxX, maxX), y: clamp(pan.y, -maxY, maxY) };
}

function applyViewportTransform(dialog) {
    const pan = dialog._gradePan || { x: 0, y: 0 }; const zoom = Number(dialog._gradeZoom || 1);
    dialog.querySelectorAll(".cs-grade-image").forEach((image) => { image.style.transform = `translate3d(${pan.x}px,${pan.y}px,0) scale(${zoom})`; });
    dialog.querySelector(".cs-grade-viewport")?.classList.toggle("pan-ready", zoom > 1);
}

function setZoom(dialog, zoom) {
    dialog._gradeZoom = clamp(Number(zoom) || 1, 0.25, 4); if (dialog._gradeZoom <= 1) dialog._gradePan = { x: 0, y: 0 }; clampPan(dialog); applyViewportTransform(dialog);
    dialog.querySelector(".cs-grade-zoom-value").textContent = `${Math.round(dialog._gradeZoom * 100)}%`;
}

function setCompare(dialog, percent) {
    const value = clamp(Number(percent) || 0, 0, 100); dialog._gradeCompare = value; dialog.querySelector(".cs-grade-viewport")?.style.setProperty("--compare-position", `${value}%`);
}

async function openPreview(node) {
    addStyles();
    let dialog = document.createElement("dialog"); dialog.className = "cs-grade-dialog"; dialog._gradeZoom = 1; dialog._gradePan = { x: 0, y: 0 }; dialog._gradeCompare = 0;
    const vectors = {
        rgb_offset: parseVec3(valueOf(node, "rgb_offset", "[0,0,0]"), [0, 0, 0]),
        rgb_multiply: parseVec3(valueOf(node, "rgb_multiply", "[1,1,1]"), [1, 1, 1]),
        rgb_gamma: parseVec3(valueOf(node, "rgb_gamma", "[1,1,1]"), [1, 1, 1]),
    };
    let whitePoint = normalizeHex(valueOf(node, "white_point", "#FFFFFF")) || "#FFFFFF";
    let lut = String(valueOf(node, "lut", NO_LUT) || NO_LUT);
    const lutOptions = comboOptions(node, "lut", [NO_LUT]);
    if (!lutOptions.includes(lut)) lut = NO_LUT;
    const curves = parseCurves(valueOf(node, "curves", JSON.stringify(DEFAULT_CURVES))); let activeChannel = "rgb";
    const colorMarkup = COLOR_PARAMS.map(([name, label, min, max, defaultValue, step]) => `<div class="cs-grade-color-row"><label for="cs-grade-${name}">${label}</label><input id="cs-grade-${name}" data-grade-param="${name}" type="range" min="${min}" max="${max}" step="${step}" value="${valueOf(node, name, defaultValue)}"><input class="cs-grade-number" data-grade-number="${name}" type="number" min="${min}" max="${max}" step="${step}" value="${valueOf(node, name, defaultValue)}"><button class="cs-grade-button cs-grade-reset" data-grade-reset="${name}" type="button" title="Reset ${label}">&#8634;</button></div>`).join("");
    const gradeMarkup = GRADE_GROUPS.map((group) => `<section class="cs-grade-group" data-grade-group="${group.name}"><div class="cs-grade-group-head"><h3>${group.label}</h3></div><div class="cs-grade-scalar-row"><input data-grade-param="${group.name}" type="range" min="${group.min}" max="${group.max}" step="0.001" value="${valueOf(node, group.name, group.defaultValue)}"><input class="cs-grade-number" data-grade-number="${group.name}" type="number" min="${group.min}" max="${group.max}" step="0.001" value="${valueOf(node, group.name, group.defaultValue)}"><button class="cs-grade-button cs-grade-reset" data-grade-group-reset="${group.name}" type="button" title="Reset ${group.label} and RGB ${group.label}">&#8634;</button></div><div class="cs-grade-wheel-body"><canvas class="cs-grade-wheel" data-grade-wheel="${group.vector}" aria-label="RGB ${group.label} colour wheel"></canvas><div class="cs-grade-vector-values">${["r", "g", "b"].map((channel, index) => `<label class="cs-grade-vector-field" data-channel="${channel}"><span>${channel.toUpperCase()}</span><input class="cs-grade-vector-number" data-vector="${group.vector}" data-index="${index}" type="number" min="${group.vectorMin}" max="${group.vectorMax}" step="0.001" value="${vectors[group.vector][index]}"></label>`).join("")}</div></div></section>`).join("");
    const lutMarkup = `<div class="cs-grade-lut-row"><label for="cs-grade-lut">Load LUT</label><select id="cs-grade-lut" class="cs-grade-lut-select">${lutOptions.map((option) => `<option value="${escapeHtml(option)}"${option === lut ? " selected" : ""}>${escapeHtml(option)}</option>`).join("")}</select></div>`;
    const whiteMarkup = `<div class="cs-grade-white-row"><label for="cs-grade-white-point">White Point</label><input id="cs-grade-white-point" class="cs-grade-text cs-grade-white-point" value="${whitePoint}"><input class="cs-grade-swatch" type="color" value="${whitePoint}" title="Choose White Point"><span class="cs-grade-white-value">${whitePoint}</span><button class="cs-grade-button cs-grade-reset cs-grade-white-reset" type="button" title="Reset White Point">&#8634;</button></div>`;
    dialog.innerHTML = `<div class="cs-grade-shell"><div class="cs-grade-head"><div><h2 class="cs-grade-title">Grade Preview</h2><div class="cs-grade-muted cs-grade-file"></div></div><button class="cs-grade-button cs-grade-close" type="button">&times;</button></div><div class="cs-grade-preview-wrap"><div class="cs-grade-viewport"><span class="cs-grade-viewport-label">Result / Original</span><img class="cs-grade-image cs-grade-result" draggable="false" alt="Graded preview"><div class="cs-grade-original-clip"><img class="cs-grade-image cs-grade-original" draggable="false" alt="Original comparison"></div><div class="cs-grade-divider" title="Drag to compare Original and Result"></div></div><div class="cs-grade-cache-loading" role="status">Preparing cache, please wait 0%</div></div><div class="cs-grade-zoom"><span class="cs-grade-muted">Zoom</span><button class="cs-grade-button" data-grade-zoom="0.5" type="button">50%</button><button class="cs-grade-button" data-grade-zoom="1" type="button">100%</button><button class="cs-grade-button" data-grade-zoom="2" type="button">200%</button><button class="cs-grade-button" data-grade-zoom="fit" type="button">Fit</button><span class="cs-grade-muted cs-grade-zoom-value">100%</span></div><div class="cs-grade-timeline"><div class="cs-grade-step"><button class="cs-grade-button cs-grade-prev" type="button">|&lt;</button><button class="cs-grade-button cs-grade-next" type="button">&gt;|</button></div><input class="cs-grade-timeline-range" type="range" min="0" max="0" step="1" value="0"><input class="cs-grade-number cs-grade-frame" type="number" min="0" max="0" step="1" value="0"><span class="cs-grade-muted cs-grade-frame-count">0 / 0</span></div><div class="cs-grade-lower"><section class="cs-grade-left"><div class="cs-grade-color-block"><div class="cs-grade-top-row">${lutMarkup}${whiteMarkup}</div>${colorMarkup}</div><div class="cs-grade-triplet">${gradeMarkup}</div></section><section class="cs-grade-curves"><h3>Curves</h3><div class="cs-grade-tabs"><div class="cs-grade-tabset">${CURVE_CHANNELS.map((channel) => `<button class="cs-grade-button cs-grade-tab${channel === "rgb" ? " active" : ""}" data-channel="${channel}" type="button">${channel.toUpperCase()}</button>`).join("")}</div><button class="cs-grade-button cs-grade-curve-reset" type="button" title="Reset selected curve">Reset</button></div><div class="cs-grade-curve-wrap"><canvas class="cs-grade-curve" aria-label="RGB curve editor" title="Add or drag points with the primary pointer; remove non-endpoints with the context menu."></canvas></div></section></div><div class="cs-grade-actions"><span class="cs-grade-status">Loading preview...</span><button class="cs-grade-button cs-grade-reset-all" type="button">Reset All</button><button class="cs-grade-button cs-grade-cancel" type="button">Close</button><button class="cs-grade-button active cs-grade-apply" type="button">Apply to Node</button></div></div>`;
    document.body.append(dialog);
    const loading = dialog.querySelector(".cs-grade-cache-loading"); const statusElement = dialog.querySelector(".cs-grade-status"); let closed = false; let timer = null; let resizeObserver = null;
    const status = (message, error = false) => { statusElement.textContent = message; statusElement.classList.toggle("cs-grade-error", error); };
    const setLoading = (message, visible = true) => { loading.textContent = message; loading.hidden = !visible; };
    const earlyClose = () => { closed = true; dialog.close(); dialog.remove(); };
    dialog.querySelector(".cs-grade-close").addEventListener("click", earlyClose); dialog.querySelector(".cs-grade-cancel").addEventListener("click", earlyClose); dialog.addEventListener("cancel", earlyClose); dialog.showModal();
    let source = null; let info = null;
    try {
        const cachedSource = await fetchCachedSource(node); const upstreamSource = connectedVideoSource(node, ["image", "images", "video_input"]);
        source = upstreamSource?.loaderId ? upstreamSource : cachedSource || upstreamSource;
        if (!source) throw new Error("Run CS Color Grade once to cache its connected image/video input.");
        if (source.loaderId && !source.token) source = await ensureLoaderPreviewSource(source, { onProgress: (progress) => setLoading(`Preparing cache, please wait ${progress}%`) });
        if (!source.token && !source.filename) throw new Error("No previewable input source was found.");
        info = source.info || prepareInputTimeline(source, await fetchInfo(source.filename));
    } catch (error) { if (!closed) setLoading(error?.message || "Unable to prepare preview cache"); return; }
    if (closed) return;
    dialog.querySelector(".cs-grade-close").removeEventListener("click", earlyClose); dialog.querySelector(".cs-grade-cancel").removeEventListener("click", earlyClose); dialog.removeEventListener("cancel", earlyClose); setLoading("Preparing cache, please wait 100%", false);
    dialog.querySelector(".cs-grade-file").textContent = `${source.label || source.filename || "Cached input"} · ${info.frames || 1} frames`;
    const timeline = dialog.querySelector(".cs-grade-timeline-range"); const frameInput = dialog.querySelector(".cs-grade-frame"); const curveCanvas = dialog.querySelector(".cs-grade-curve"); const maxFrame = Math.max(0, Number(info.frames || 1) - 1); timeline.max = String(maxFrame); frameInput.max = String(maxFrame);
    let frame = 0; let requestSerial = 0; let curveDrag = null; let compareDragging = false;
    const scalarValues = () => Object.fromEntries(PARAMS.map(([name]) => [name, Number(dialog.querySelector(`[data-grade-param="${name}"]`).value)]));
    const payload = () => ({ node_id: String(node.id), source_kind: source.kind || "video", source_token: source.token || "", video: source.filename || "", frame: source.token ? frame : sourceFrameForLocal(info, frame), local_frame: frame, lut, white_point: whitePoint, ...scalarValues(), rgb_offset: vectors.rgb_offset, rgb_multiply: vectors.rgb_multiply, rgb_gamma: vectors.rgb_gamma, curves: curvesPayload(curves) });
    async function preview() {
        const serial = ++requestSerial; status(`Rendering frame ${frame}...`);
        try { const response = await api.fetchApi("/cinestyle/color-grade-preview", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload()) }); const result = await response.json(); if (!response.ok) throw new Error(result.error || "Preview failed"); if (serial !== requestSerial) return; dialog.querySelector(".cs-grade-original").src = result.original; dialog.querySelector(".cs-grade-result").src = result.preview; status(`Frame ${result.frame}`); }
        catch (error) { if (serial === requestSerial) status(error.message, true); }
    }
    const schedulePreview = () => { clearTimeout(timer); timer = setTimeout(preview, 100); };
    const wheelControllers = Object.fromEntries(GRADE_GROUPS.map((group) => [group.name, installInlineWheel(dialog, group, vectors, schedulePreview)]));
    const setFrame = (value) => { frame = clamp(Math.round(Number(value) || 0), 0, maxFrame); timeline.value = String(frame); frameInput.value = String(frame); dialog.querySelector(".cs-grade-frame-count").textContent = `${frame} / ${maxFrame}`; schedulePreview(); };
    timeline.addEventListener("input", () => setFrame(timeline.value)); frameInput.addEventListener("change", () => setFrame(frameInput.value)); dialog.querySelector(".cs-grade-prev").addEventListener("click", () => setFrame(frame - 1)); dialog.querySelector(".cs-grade-next").addEventListener("click", () => setFrame(frame + 1));
    dialog.querySelectorAll("[data-grade-param]").forEach((range) => range.addEventListener("input", () => { dialog.querySelector(`[data-grade-number="${range.dataset.gradeParam}"]`).value = range.value; schedulePreview(); }));
    dialog.querySelectorAll("[data-grade-number]").forEach((number) => number.addEventListener("input", () => { const value = Number(number.value); if (!Number.isFinite(value)) return; const range = dialog.querySelector(`[data-grade-param="${number.dataset.gradeNumber}"]`); range.value = String(value); schedulePreview(); }));
    dialog.querySelectorAll("[data-grade-reset]").forEach((button) => button.addEventListener("click", () => { const definition = PARAMS.find(([name]) => name === button.dataset.gradeReset); const range = dialog.querySelector(`[data-grade-param="${button.dataset.gradeReset}"]`); const number = dialog.querySelector(`[data-grade-number="${button.dataset.gradeReset}"]`); range.value = String(definition[4]); number.value = String(definition[4]); schedulePreview(); }));
    dialog.querySelectorAll("[data-grade-group-reset]").forEach((button) => button.addEventListener("click", () => { const group = GRADE_GROUPS.find((item) => item.name === button.dataset.gradeGroupReset); const range = dialog.querySelector(`[data-grade-param="${group.name}"]`); const number = dialog.querySelector(`[data-grade-number="${group.name}"]`); range.value = String(group.defaultValue); number.value = String(group.defaultValue); wheelControllers[group.name].reset(); schedulePreview(); }));
    const whiteInput = dialog.querySelector(".cs-grade-white-point"); const whiteSwatch = dialog.querySelector(".cs-grade-swatch"); const whiteValue = dialog.querySelector(".cs-grade-white-value");
    const setWhitePoint = (value, updateText = true) => { const normalized = normalizeHex(value); if (!normalized) return false; whitePoint = normalized; whiteSwatch.value = normalized; whiteValue.textContent = normalized; whiteInput.setCustomValidity(""); if (updateText) whiteInput.value = normalized; schedulePreview(); return true; };
    whiteInput.addEventListener("input", () => { const normalized = normalizeHex(whiteInput.value); whiteInput.setCustomValidity(normalized ? "" : "Use #RRGGBB"); if (normalized) setWhitePoint(normalized, false); });
    whiteInput.addEventListener("change", () => { if (!setWhitePoint(whiteInput.value)) whiteInput.value = whitePoint; });
    whiteSwatch.addEventListener("input", () => setWhitePoint(whiteSwatch.value));
    dialog.querySelector(".cs-grade-white-reset").addEventListener("click", () => setWhitePoint("#FFFFFF"));
    dialog.querySelector(".cs-grade-lut-select").addEventListener("change", (event) => { lut = String(event.target.value || NO_LUT); schedulePreview(); });
    const redrawCurves = () => drawCurveEditor(curveCanvas, curves, activeChannel); dialog.querySelectorAll("[data-channel]").forEach((button) => button.addEventListener("click", () => { activeChannel = button.dataset.channel; dialog.querySelectorAll("[data-channel]").forEach((item) => item.classList.toggle("active", item === button)); redrawCurves(); }));
    dialog.querySelector(".cs-grade-curve-reset").addEventListener("click", () => { curves[activeChannel] = [[0, 0], [1, 1]]; redrawCurves(); schedulePreview(); });
    curveCanvas.addEventListener("contextmenu", (event) => { event.preventDefault(); const points = curves[activeChannel]; const index = nearestCurvePoint(curveCanvas, points, event); if (index <= 0 || index >= points.length - 1 || points.length <= 2) return; points.splice(index, 1); redrawCurves(); schedulePreview(); });
    curveCanvas.addEventListener("pointerdown", (event) => { if (event.button !== 0) return; event.preventDefault(); const points = curves[activeChannel]; let index = nearestCurvePoint(curveCanvas, points, event); if (index < 0) { const position = curvePosition(curveCanvas, event); const firstX = points[0][0]; const lastX = points.at(-1)[0]; if (position.x <= firstX + MIN_POINT_DISTANCE || position.x >= lastX - MIN_POINT_DISTANCE || points.some(([x]) => Math.abs(x - position.x) < MIN_POINT_DISTANCE)) return; points.push([position.x, position.y]); points.sort((left, right) => left[0] - right[0]); index = points.findIndex((point) => point[0] === position.x && point[1] === position.y); } curveDrag = { pointerId: event.pointerId, channel: activeChannel, index }; curveCanvas.setPointerCapture?.(event.pointerId); redrawCurves(); schedulePreview(); });
    curveCanvas.addEventListener("pointermove", (event) => { if (!curveDrag || curveDrag.pointerId !== event.pointerId) return; const points = curves[curveDrag.channel]; const position = curvePosition(curveCanvas, event); const left = curveDrag.index > 0 ? points[curveDrag.index - 1][0] + MIN_POINT_DISTANCE : 0; const right = curveDrag.index < points.length - 1 ? points[curveDrag.index + 1][0] - MIN_POINT_DISTANCE : 1; points[curveDrag.index] = [clamp(position.x, left, right), position.y]; redrawCurves(); schedulePreview(); });
    const stopCurveDrag = (event) => { if (!curveDrag || curveDrag.pointerId !== event.pointerId) return; curves[curveDrag.channel] = canonicalPoints(curves[curveDrag.channel]); curveCanvas.releasePointerCapture?.(event.pointerId); curveDrag = null; redrawCurves(); schedulePreview(); }; curveCanvas.addEventListener("pointerup", stopCurveDrag); curveCanvas.addEventListener("pointercancel", stopCurveDrag);
    const viewport = dialog.querySelector(".cs-grade-viewport"); const divider = dialog.querySelector(".cs-grade-divider"); const updateCompare = (event) => { const rect = viewport.getBoundingClientRect(); setCompare(dialog, (event.clientX - rect.left) / Math.max(1, rect.width) * 100); };
    divider.addEventListener("pointerdown", (event) => { if (event.button !== 0) return; event.preventDefault(); event.stopPropagation(); compareDragging = true; divider.setPointerCapture?.(event.pointerId); updateCompare(event); }); divider.addEventListener("pointermove", (event) => { if (compareDragging) updateCompare(event); }); const stopCompare = (event) => { compareDragging = false; divider.releasePointerCapture?.(event.pointerId); }; divider.addEventListener("pointerup", stopCompare); divider.addEventListener("pointercancel", stopCompare);
    dialog.querySelectorAll("[data-grade-zoom]").forEach((button) => button.addEventListener("click", () => setZoom(dialog, button.dataset.gradeZoom === "fit" ? 1 : Number(button.dataset.gradeZoom)))); dialog.querySelectorAll(".cs-grade-image").forEach((image) => image.addEventListener("load", () => { clampPan(dialog); applyViewportTransform(dialog); }));
    viewport.addEventListener("wheel", (event) => { event.preventDefault(); setZoom(dialog, dialog._gradeZoom + (event.deltaY < 0 ? 0.1 : -0.1)); }, { passive: false });
    viewport.addEventListener("pointerdown", (event) => {
        if (dialog._gradeZoom <= 1 || event.button !== 0 || event.target === divider) return;
        event.preventDefault();
        viewport.setPointerCapture?.(event.pointerId);
        viewport.classList.add("pan-active");
        dialog._gradePanDrag = { pointerId: event.pointerId, startX: event.clientX, startY: event.clientY, origin: { ...dialog._gradePan } };
    });
    viewport.addEventListener("pointermove", (event) => {
        const drag = dialog._gradePanDrag;
        if (!drag || drag.pointerId !== event.pointerId) return;
        event.preventDefault();
        dialog._gradePan = { x: drag.origin.x + event.clientX - drag.startX, y: drag.origin.y + event.clientY - drag.startY };
        clampPan(dialog);
        applyViewportTransform(dialog);
    });
    const stopPan = (event) => {
        if (!dialog._gradePanDrag || dialog._gradePanDrag.pointerId !== event.pointerId) return;
        dialog._gradePanDrag = null;
        viewport.releasePointerCapture?.(event.pointerId);
        viewport.classList.remove("pan-active");
    };
    viewport.addEventListener("pointerup", stopPan);
    viewport.addEventListener("pointercancel", stopPan);
    viewport.addEventListener("lostpointercapture", stopPan);
    resizeObserver = new ResizeObserver(() => { clampPan(dialog); applyViewportTransform(dialog); redrawCurves(); Object.values(wheelControllers).forEach((controller) => controller.redraw()); }); resizeObserver.observe(dialog.querySelector(".cs-grade-preview-wrap")); resizeObserver.observe(dialog.querySelector(".cs-grade-curve-wrap")); resizeObserver.observe(dialog.querySelector(".cs-grade-triplet"));
    const close = () => { closed = true; clearTimeout(timer); resizeObserver.disconnect(); dialog.close(); dialog.remove(); }; dialog.querySelector(".cs-grade-close").addEventListener("click", close); dialog.querySelector(".cs-grade-cancel").addEventListener("click", close); dialog.addEventListener("cancel", close);
    dialog.querySelector(".cs-grade-reset-all").addEventListener("click", () => {
        lut = NO_LUT;
        dialog.querySelector(".cs-grade-lut-select").value = NO_LUT;
        setWhitePoint("#FFFFFF");
        for (const [name, , , , defaultValue] of PARAMS) {
            dialog.querySelector(`[data-grade-param="${name}"]`).value = String(defaultValue);
            dialog.querySelector(`[data-grade-number="${name}"]`).value = String(defaultValue);
        }
        for (const group of GRADE_GROUPS) wheelControllers[group.name].reset();
        for (const channel of CURVE_CHANNELS) curves[channel] = [[0, 0], [1, 1]];
        activeChannel = "rgb";
        dialog.querySelectorAll("[data-channel]").forEach((button) => button.classList.toggle("active", button.dataset.channel === activeChannel));
        redrawCurves();
        schedulePreview();
    });
    dialog.querySelector(".cs-grade-apply").addEventListener("click", () => { const setValue = (name, value) => { const target = widget(node, name); if (!target) return; const index = node.widgets?.indexOf(target) ?? -1; target.value = value; target.callback?.(value); target.value = value; if (index >= 0 && Array.isArray(node.widgets_values)) node.widgets_values[index] = value; }; setValue("lut", lut); setValue("white_point", whitePoint); for (const [name] of PARAMS) setValue(name, Number(dialog.querySelector(`[data-grade-param="${name}"]`).value)); setValue("rgb_offset", JSON.stringify(vectors.rgb_offset.map((value) => Number(value.toFixed(6))))); setValue("rgb_multiply", JSON.stringify(vectors.rgb_multiply.map((value) => Number(value.toFixed(6))))); setValue("rgb_gamma", JSON.stringify(vectors.rgb_gamma.map((value) => Number(value.toFixed(6))))); setValue("curves", JSON.stringify(curvesPayload(curves))); node.graph?.setDirtyCanvas(true, true); close(); });
    setCompare(dialog, 0); setZoom(dialog, 1); redrawCurves(); setFrame(0);
}

app.registerExtension({
    name: "CineStyle.ColorGradePreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_ID) return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            original?.apply(this, arguments);
            const button = this.addWidget("button", "Grade Preview", "", () => openPreview(this));
            button.name = "Grade Preview"; button.label = "Grade Preview"; button.options = { ...(button.options || {}), serialize: false };
            this.setSize?.([430, Math.max(450, this.computeSize?.()[1] || 450)]);
        };
    },
    loadedGraphNode(node) {
        if (node?.type !== NODE_ID) return;
        node.setSize?.([node.size?.[0] || 430, Math.max(450, node.computeSize?.()[1] || node.size?.[1] || 450)]);
    },
});
