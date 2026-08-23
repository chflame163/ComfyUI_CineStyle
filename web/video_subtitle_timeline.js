import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_ID = "CS_Video_Subtitle_Track";
const STYLE_ID = "cinestyle-subtitle-timeline-style";

function widget(node, name) { return node.widgets?.find((item) => item.name === name); }
function graphNode(graph, id) { return graph?.getNodeById?.(id) || (graph?._nodes || []).find((item) => String(item?.id) === String(id)) || null; }
function connectedVideoFilename(node) {
    const input = node.inputs?.find((item) => item.name === "video");
    const link = input?.link == null ? null : (node.graph?.links?.[input.link] || app.graph?.links?.[input.link]);
    const origin = link ? graphNode(node.graph || app.graph, link.origin_id ?? link.originId) : null;
    return String(widget(origin, "video")?.value || "").trim();
}
function setWidgetValue(node, name, value) {
    const target = widget(node, name);
    if (!target) return;
    target.value = value;
    target.callback?.(value);
}
function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
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
function readCues(node) {
    try {
        const data = JSON.parse(String(widget(node, "subtitle_data")?.value || "[]"));
        if (Array.isArray(data) && data.some((cue) => cue?.text)) return data.map((cue, index) => ({ id: cue.id ?? index + 1, start: Number(cue.start) || 0, end: Number(cue.end) || 0, text: String(cue.text || "") })).filter((cue) => cue.end > cue.start && cue.text);
    } catch (_) { /* fall back to the source SRT */ }
    return parseSrt(widget(node, "srt")?.value || "");
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
function fontUrl(relative) { return api.apiURL(`/cinestyle/font/${String(relative || "").split("/").map(encodeURIComponent).join("/")}`); }
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
      .cs-subtitle-overlay { position:absolute; left:50%; top:88%; transform:translate(-50%,-100%); max-width:90%; color:#fff; text-align:center; white-space:pre-wrap; word-break:break-word; font:48px/1.1 sans-serif; text-shadow:2px 3px 3px #000; pointer-events:auto; cursor:move; user-select:none; }
      .cs-subtitle-readout { display:flex; justify-content:space-between; color:#aeb5c2; font-variant-numeric:tabular-nums; }
      .cs-subtitle-pointer-row { position:relative; height:15px; user-select:none; }
      .cs-subtitle-pointer { position:absolute; top:0; width:16px; height:15px; transform:translateX(-50%); border:0; background:#55a9f5; clip-path:polygon(0 0,100% 0,50% 100%); cursor:ew-resize; z-index:5; }
      .cs-subtitle-viewport { position:relative; overflow:hidden; border:1px solid #363b45; border-radius:6px; background:#20232a; }
      .cs-subtitle-axis { position:relative; height:22px; color:#9299a8; font-size:11px; font-variant-numeric:tabular-nums; }
      .cs-subtitle-axis span { position:absolute; transform:translateX(-50%); top:4px; }
      .cs-subtitle-track { position:relative; height:34px; border-top:1px solid #343943; }
      .cs-subtitle-track-label { position:absolute; left:8px; top:9px; z-index:4; color:#aeb5c2; font-size:11px; pointer-events:none; }
      .cs-subtitle-track-body { position:absolute; inset:0; margin-left:72px; }
      .cs-subtitle-track-video .cs-subtitle-track-body { background:repeating-linear-gradient(90deg,#2b3039 0 1px,transparent 1px 10%); }
      .cs-subtitle-track-subtitles .cs-subtitle-track-body { background:#1c2026; }
      .cs-subtitle-cue { position:absolute; top:5px; bottom:5px; min-width:5px; overflow:visible; border:1px solid #4b9de8; border-radius:3px; background:#317ec4; color:#f5f7fb; cursor:grab; user-select:none; }
      .cs-subtitle-cue.selected { background:#3f9f83; border-color:#68d0ad; z-index:3; }
      .cs-subtitle-cue:active { cursor:grabbing; }
      .cs-subtitle-cue-label { display:block; overflow:hidden; padding:3px 8px; white-space:nowrap; text-overflow:ellipsis; pointer-events:none; }
      .cs-subtitle-cue-handle { position:absolute; top:-2px; bottom:-2px; width:7px; background:#f5f7fb; border-radius:2px; cursor:ew-resize; z-index:2; }
      .cs-subtitle-cue-handle.in { left:-4px; } .cs-subtitle-cue-handle.out { right:-4px; }
      .cs-subtitle-controls { display:flex; flex-wrap:wrap; gap:6px; }
      .cs-subtitle-controls button,.cs-subtitle-foot button { border:1px solid #424956; border-radius:5px; padding:7px 10px; background:#242832; color:#e6e9ef; cursor:pointer; }
      .cs-subtitle-controls button:hover,.cs-subtitle-foot button:hover { background:#303643; }
      .cs-subtitle-point-frame { min-width:52px; padding-left:7px !important; padding-right:7px !important; color:#9fc9ec !important; font-variant-numeric:tabular-nums; }
      .cs-subtitle-fields { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:9px; }
      .cs-subtitle-field { display:grid; gap:5px; color:#9da5b4; min-width:0; }
      .cs-subtitle-field input,.cs-subtitle-field select { width:100%; box-sizing:border-box; border:1px solid #424956; border-radius:5px; padding:7px 8px; background:#20232a; color:#f2f4f7; }
      .cs-subtitle-field input[type=color] { height:32px; padding:2px; }
      .cs-subtitle-check { display:flex; align-items:center; gap:6px; min-height:32px; }
      .cs-subtitle-check input { width:auto; }
      .cs-subtitle-status { min-width:0; flex:1; color:#9299a8; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
      .cs-subtitle-foot { justify-content:flex-end; }
      .cs-subtitle-foot .apply { background:#317ec4; border-color:#4b9de8; }
      @media(max-width:760px) { .cs-subtitle-fields{grid-template-columns:repeat(2,minmax(0,1fr));} }
      @media(max-width:460px) { .cs-subtitle-fields{grid-template-columns:1fr;} .cs-subtitle-shell{padding:10px;} }
    `;
    document.head.append(style);
}

function openTimeline(node) {
    const filename = String(widget(node, "video_file")?.value || connectedVideoFilename(node) || "");
    addStyles();
    const dialog = document.createElement("dialog");
    dialog.className = "cs-subtitle-dialog";
    dialog.innerHTML = `
      <div class="cs-subtitle-shell">
        <div class="cs-subtitle-head"><div><h2 class="cs-subtitle-title">Subtitle Timeline</h2><div class="cs-subtitle-muted cs-subtitle-file"></div></div><button class="cs-subtitle-close" type="button" aria-label="Close">&times;</button></div>
        <div class="cs-subtitle-preview-wrap"><video class="cs-subtitle-video" controls playsinline preload="metadata"></video><div class="cs-subtitle-overlay"></div></div>
        <div class="cs-subtitle-readout"><span class="current">00:00.00</span><span class="range"></span><span class="duration">00:00.00</span></div>
        <div class="cs-subtitle-pointer-row"><button class="cs-subtitle-pointer" type="button" aria-label="Current time"></button></div>
        <div class="cs-subtitle-viewport"><div class="cs-subtitle-axis"></div><div class="cs-subtitle-track cs-subtitle-track-subtitles"><span class="cs-subtitle-track-label">Subtitles</span><div class="cs-subtitle-track-body"></div></div><div class="cs-subtitle-track cs-subtitle-track-video"><span class="cs-subtitle-track-label">Video</span><div class="cs-subtitle-track-body"></div></div></div>
        <div class="cs-subtitle-controls"><button class="set-in">Set In</button><button class="cs-subtitle-point-frame in-frame" type="button" aria-label="Jump to in point" title="Jump to in point">0</button><button class="back">|&lt;</button><button class="play">Play</button><button class="forward">&gt;|</button><button class="cs-subtitle-point-frame out-frame" type="button" aria-label="Jump to out point" title="Jump to out point">0</button><button class="set-out">Set Out</button><button class="zoom-out">−</button><button class="zoom-in">+</button><button class="pan-left">◀</button><button class="pan-right">▶</button></div>
        <div class="cs-subtitle-fields"><label class="cs-subtitle-field">Font<select class="font-family"></select></label><label class="cs-subtitle-field">Size<input class="font-size" type="number" min="8" max="256" step="1"></label><label class="cs-subtitle-field">Fill<input class="fill-1" type="color"></label><label class="cs-subtitle-field">Gradient Fill<input class="fill-2" type="color"></label><label class="cs-subtitle-field cs-subtitle-check"><span><input class="gradient" type="checkbox"> vertical gradient</span></label><label class="cs-subtitle-field">Align<select class="text-align"><option value="left">Left</option><option value="center">Center</option><option value="right">Right</option></select></label><label class="cs-subtitle-field">Outline<input class="outline-size" type="number" min="0" max="32" step="1"></label><label class="cs-subtitle-field">Outline Color<input class="outline-color" type="color"></label><label class="cs-subtitle-field">Shadow<input class="shadow-size" type="number" min="0" max="32" step="1"></label><label class="cs-subtitle-field">Shadow Color<input class="shadow-color" type="color"></label></div>
        <div class="cs-subtitle-row"><span class="cs-subtitle-status"></span><button class="move-reset">Reset Position</button></div>
        <div class="cs-subtitle-foot"><button class="cancel">Cancel</button><button class="apply">Apply</button></div>
      </div>`;
    document.body.append(dialog);

    const video = dialog.querySelector(".cs-subtitle-video");
    const overlay = dialog.querySelector(".cs-subtitle-overlay");
    const viewport = dialog.querySelector(".cs-subtitle-viewport");
    const axis = dialog.querySelector(".cs-subtitle-axis");
    const body = dialog.querySelector(".cs-subtitle-track-subtitles .cs-subtitle-track-body");
    const pointer = dialog.querySelector(".cs-subtitle-pointer");
    const inFrameButton = dialog.querySelector(".in-frame");
    const outFrameButton = dialog.querySelector(".out-frame");
    const current = dialog.querySelector(".current");
    const range = dialog.querySelector(".range");
    const durationLabel = dialog.querySelector(".duration");
    const status = dialog.querySelector(".cs-subtitle-status");
    const cues = readCues(node).map((cue, index) => ({ ...cue, id: cue.id ?? index + 1 }));
    let info = null;
    let duration = Math.max(1, ...cues.map((cue) => Number(cue.end) || 0));
    let fps = 30;
    let inFrame = Math.max(0, Number(widget(node, "start_frame")?.value || 0));
    let outFrame = Number(widget(node, "end_frame")?.value ?? -1);
    let viewStart = 0;
    let viewDuration = duration;
    let selected = null;
    let drag = null;
    const loadedFonts = new Set();
    let playingSelection = false;
    const style = {
        font_family: String(widget(node, "font_family")?.value || ""),
        font_size: Number(widget(node, "font_size")?.value || 48),
        fill_color_1: String(widget(node, "fill_color_1")?.value || "#FFFFFF"),
        fill_color_2: String(widget(node, "fill_color_2")?.value || "#FFFFFF"),
        gradient: Boolean(widget(node, "gradient")?.value || false),
        text_align: String(widget(node, "text_align")?.value || "center"),
        position_x: Number(widget(node, "position_x")?.value ?? 0.5),
        position_y: Number(widget(node, "position_y")?.value ?? 0.88),
        outline_size: Number(widget(node, "outline_size")?.value || 2),
        outline_color: String(widget(node, "outline_color")?.value || "#000000"),
        shadow_size: Number(widget(node, "shadow_size")?.value || 3),
        shadow_color: String(widget(node, "shadow_color")?.value || "#000000"),
    };
    const inputs = {
        font_family: dialog.querySelector(".font-family"), font_size: dialog.querySelector(".font-size"),
        fill_color_1: dialog.querySelector(".fill-1"), fill_color_2: dialog.querySelector(".fill-2"), gradient: dialog.querySelector(".gradient"),
        text_align: dialog.querySelector(".text-align"), outline_size: dialog.querySelector(".outline-size"), outline_color: dialog.querySelector(".outline-color"),
        shadow_size: dialog.querySelector(".shadow-size"), shadow_color: dialog.querySelector(".shadow-color"),
    };
    for (const [key, input] of Object.entries(inputs)) { if (input.type === "checkbox") input.checked = Boolean(style[key]); else input.value = style[key]; input.addEventListener("input", () => { style[key] = input.type === "checkbox" ? input.checked : input.value; updateOverlay(); }); }

    function rangeStartSeconds() { return inFrame / fps; }
    function rangeEndSeconds() { return (outFrame < 0 ? duration : (outFrame + 1) / fps); }
    function jumpToMarkedFrame(frame) {
        video.pause();
        setFrame(frame);
    }
    inFrameButton.addEventListener("click", () => jumpToMarkedFrame(inFrame));
    outFrameButton.addEventListener("click", () => jumpToMarkedFrame(outFrame < 0 ? Math.max(0, Math.round(duration * fps) - 1) : outFrame));
    function updateOverlay() {
        const active = cues.filter((cue) => cue.start <= (video.currentTime || 0) && cue.end > (video.currentTime || 0));
        overlay.textContent = active.map((cue) => cue.text).join("\n");
        overlay.style.left = `${style.position_x * 100}%`;
        overlay.style.top = `${style.position_y * 100}%`;
        overlay.style.fontSize = `${Math.max(8, Number(style.font_size) || 48)}px`;
        overlay.style.textAlign = style.text_align;
        overlay.style.textShadow = `${Number(style.shadow_size) || 0}px ${Number(style.shadow_size) || 0}px ${Number(style.shadow_size) || 0}px ${style.shadow_color}`;
        overlay.style.webkitTextStroke = `${Number(style.outline_size) || 0}px ${style.outline_color}`;
        overlay.style.background = style.gradient ? `linear-gradient(${style.fill_color_1},${style.fill_color_2})` : style.fill_color_1;
        overlay.style.webkitBackgroundClip = "text";
        overlay.style.color = "transparent";
        overlay.style.display = active.length ? "block" : "none";
        if (style.font_family) {
            const family = `CineStyleSubtitle_${String(style.font_family).replace(/[^a-zA-Z0-9_]/g, "_")}`;
            if (!loadedFonts.has(family)) {
                loadedFonts.add(family);
                const fontFace = new FontFace(family, `url(${fontUrl(style.font_family)})`);
                fontFace.load().then((loaded) => { document.fonts.add(loaded); overlay.style.fontFamily = family; }).catch(() => {});
            } else overlay.style.fontFamily = family;
        } else overlay.style.fontFamily = "sans-serif";
    }
    function updateReadout() { const now = video.currentTime || 0; current.textContent = formatTime(now); range.textContent = `In ${formatTime(rangeStartSeconds())}  -  Out ${formatTime(rangeEndSeconds())}`; durationLabel.textContent = formatTime(duration); }
    function cuePosition(cue) { const start = ((cue.start - viewStart) / viewDuration) * 100; const width = ((cue.end - cue.start) / viewDuration) * 100; return { left: `${start}%`, width: `${Math.max(0.35, width)}%` }; }
    function renderTimeline() {
        viewDuration = clamp(viewDuration, Math.min(duration, 0.5), duration);
        viewStart = clamp(viewStart, 0, Math.max(0, duration - viewDuration));
        axis.innerHTML = "";
        const tickCount = Math.max(2, Math.min(12, Math.round(viewport.clientWidth / 100)));
        for (let i = 0; i <= tickCount; i++) { const span = document.createElement("span"); span.style.left = `${(i / tickCount) * 100}%`; span.textContent = formatTime(viewStart + (viewDuration * i / tickCount)); axis.append(span); }
        body.innerHTML = "";
        for (const cue of cues) {
            if (cue.end < viewStart || cue.start > viewStart + viewDuration) continue;
            const item = document.createElement("div"); item.className = `cs-subtitle-cue${selected === cue.id ? " selected" : ""}`; item.dataset.id = String(cue.id); Object.assign(item.style, cuePosition(cue));
            item.innerHTML = `<span class="cs-subtitle-cue-handle in"></span><span class="cs-subtitle-cue-label"></span><span class="cs-subtitle-cue-handle out"></span>`;
            item.querySelector(".cs-subtitle-cue-label").textContent = cue.text.replace(/\n/g, " ");
            item.addEventListener("pointerdown", (event) => beginCueDrag(cue, event));
            item.addEventListener("dblclick", (event) => { event.stopPropagation(); const value = window.prompt("Edit subtitle", cue.text); if (value !== null) { cue.text = value; renderTimeline(); updateOverlay(); } });
            item.querySelector(".in").addEventListener("pointerdown", (event) => beginCueEdge(cue, "in", event));
            item.querySelector(".out").addEventListener("pointerdown", (event) => beginCueEdge(cue, "out", event));
            body.append(item);
        }
        const ratio = duration ? ((video.currentTime || 0) - viewStart) / viewDuration : 0;
        pointer.style.left = `${clamp(ratio, 0, 1) * 100}%`;
        inFrameButton.textContent = String(inFrame);
        outFrameButton.textContent = String(outFrame < 0 ? Math.max(0, Math.round(duration * fps) - 1) : outFrame);
        updateReadout(); updateOverlay();
    }
    function secondsAtEvent(event) { const rect = body.getBoundingClientRect(); return viewStart + clamp((event.clientX - rect.left) / rect.width, 0, 1) * viewDuration; }
    function beginCueDrag(cue, event) {
        if (event.target.classList.contains("cs-subtitle-cue-handle")) return;
        event.preventDefault(); selected = cue.id; const origin = secondsAtEvent(event); const start = cue.start; const end = cue.end; drag = { cue, mode: "move", origin, start, end };
        const move = (moveEvent) => { const delta = secondsAtEvent(moveEvent) - drag.origin; const length = drag.end - drag.start; cue.start = clamp(drag.start + delta, 0, Math.max(0, duration - length)); cue.end = cue.start + length; renderTimeline(); };
        const up = () => { drag = null; window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
        window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
    }
    function beginCueEdge(cue, mode, event) {
        event.preventDefault(); event.stopPropagation(); selected = cue.id; drag = { cue, mode };
        const move = (moveEvent) => { const time = secondsAtEvent(moveEvent); if (mode === "in") cue.start = clamp(Math.min(time, cue.end - 0.05), 0, cue.end - 0.05); else cue.end = clamp(Math.max(time, cue.start + 0.05), cue.start + 0.05, duration); renderTimeline(); };
        const up = () => { drag = null; window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
        window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
    }
    function setFrame(frame) { video.currentTime = clamp(frame / fps, 0, duration); renderTimeline(); }
    function currentFrame() { return Math.round((video.currentTime || 0) * fps); }
    function normalizeRange() { const max = Math.max(0, Math.round(duration * fps) - 1); inFrame = clamp(Math.round(inFrame), 0, max); outFrame = outFrame < 0 ? max : clamp(Math.round(outFrame), 0, max); if (outFrame <= inFrame) outFrame = Math.min(max, inFrame + 1); }
    function close() { video.pause(); dialog.close(); dialog.remove(); }

    dialog.querySelector(".set-in").addEventListener("click", () => { inFrame = currentFrame(); normalizeRange(); renderTimeline(); });
    dialog.querySelector(".set-out").addEventListener("click", () => { outFrame = currentFrame(); normalizeRange(); renderTimeline(); });
    dialog.querySelector(".back").addEventListener("click", () => setFrame(currentFrame() - 1));
    dialog.querySelector(".forward").addEventListener("click", () => setFrame(currentFrame() + 1));
    dialog.querySelector(".play").addEventListener("click", () => { if (!video.paused) { video.pause(); return; } normalizeRange(); if (video.currentTime < rangeStartSeconds() || video.currentTime >= rangeEndSeconds()) video.currentTime = rangeStartSeconds(); playingSelection = true; video.play().catch(() => { playingSelection = false; }); });
    dialog.querySelector(".zoom-in").addEventListener("click", () => { const center = viewStart + viewDuration / 2; viewDuration = Math.max(0.5, viewDuration / 1.6); viewStart = center - viewDuration / 2; renderTimeline(); });
    dialog.querySelector(".zoom-out").addEventListener("click", () => { const center = viewStart + viewDuration / 2; viewDuration = Math.min(duration, viewDuration * 1.6); viewStart = center - viewDuration / 2; renderTimeline(); });
    dialog.querySelector(".pan-left").addEventListener("click", () => { viewStart -= viewDuration * 0.25; renderTimeline(); });
    dialog.querySelector(".pan-right").addEventListener("click", () => { viewStart += viewDuration * 0.25; renderTimeline(); });
    dialog.querySelector(".move-reset").addEventListener("click", () => { style.position_x = 0.5; style.position_y = 0.88; renderTimeline(); });
    overlay.addEventListener("pointerdown", (event) => { event.preventDefault(); const startX = event.clientX; const startY = event.clientY; const x = style.position_x; const y = style.position_y; const move = (moveEvent) => { const rect = video.getBoundingClientRect(); style.position_x = clamp(x + (moveEvent.clientX - startX) / rect.width, 0, 1); style.position_y = clamp(y + (moveEvent.clientY - startY) / rect.height, 0, 1); updateOverlay(); }; const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); }; window.addEventListener("pointermove", move); window.addEventListener("pointerup", up); });
    dialog.querySelector(".cs-subtitle-pointer-row").addEventListener("pointerdown", (event) => { const row = event.currentTarget; const move = (moveEvent) => { const rect = row.getBoundingClientRect(); setFrame(Math.round(clamp((moveEvent.clientX - rect.left) / rect.width, 0, 1) * Math.max(0, Math.round(duration * fps) - 1))); }; const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); }; window.addEventListener("pointermove", move); window.addEventListener("pointerup", up); move(event); });
    video.addEventListener("timeupdate", () => { if (playingSelection && video.currentTime >= rangeEndSeconds()) { video.pause(); video.currentTime = rangeEndSeconds(); playingSelection = false; } renderTimeline(); });
    video.addEventListener("pause", () => { playingSelection = false; dialog.querySelector(".play").textContent = "Play"; });
    video.addEventListener("play", () => { dialog.querySelector(".play").textContent = "Pause"; });
    dialog.querySelector(".cs-subtitle-close").addEventListener("click", close); dialog.querySelector(".cancel").addEventListener("click", close); dialog.addEventListener("cancel", close);
    dialog.querySelector(".apply").addEventListener("click", () => {
        normalizeRange();
        setWidgetValue(node, "start_frame", inFrame); setWidgetValue(node, "end_frame", outFrame);
        setWidgetValue(node, "subtitle_data", JSON.stringify(cues));
        for (const [key, input] of Object.entries(inputs)) setWidgetValue(node, key, input.type === "checkbox" ? input.checked : input.value);
        node.graph?.setDirtyCanvas(true, true); close();
    });
    dialog.showModal();
    dialog.querySelector(".cs-subtitle-file").textContent = filename || "Select a source video in video_file for preview";

    fetchFonts().then((fonts) => { inputs.font_family.innerHTML = "<option value=\"\">Default</option>"; for (const font of fonts) { const option = document.createElement("option"); option.value = font; option.textContent = font; inputs.font_family.append(option); } if (style.font_family && !fonts.includes(style.font_family)) { const option = document.createElement("option"); option.value = style.font_family; option.textContent = style.font_family; inputs.font_family.append(option); } inputs.font_family.value = style.font_family; }).catch(() => {});
    if (filename) {
        fetchInfo(filename).then((result) => { info = result; fps = Number(result.fps) || 30; duration = Number(result.duration) || duration; outFrame = outFrame < 0 ? Math.max(0, Math.round(duration * fps) - 1) : outFrame; viewDuration = duration; video.src = result.proxy_required ? proxyVideoUrl(filename, result.proxy_threshold, result.proxy_size) : videoUrl(filename); video.load(); normalizeRange(); renderTimeline(); }).catch((error) => { status.textContent = error.message; renderTimeline(); });
    } else {
        status.textContent = "Choose a source video for proxy preview; the connected VIDEO is rendered on execution.";
        renderTimeline();
    }
}

app.registerExtension({
    name: "CineStyle.VideoSubtitleTimeline",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_ID) return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            original?.apply(this, arguments);
            const button = this.addWidget("button", "Edit Timeline", "", () => openTimeline(this));
            button.name = "Edit Timeline"; button.label = "Edit Timeline"; button.options = { ...(button.options || {}), serialize: false };
            this.setSize?.([410, Math.max(380, this.computeSize?.()[1] || 380)]);
        };
    },
});
