const STYLE_ID = "cinestyle-timeline-range-style";

function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }

export function formatFrameCount(count) {
    return `${Math.max(0, Math.round(Number(count) || 0))} frames`;
}

export function installTimelineRangeStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .cs-timeline-readout,.cs-subtitle-readout { display:flex; justify-content:space-between; color:#aeb5c2; font-variant-numeric:tabular-nums; }
      .cs-timeline-pointer-row,.cs-subtitle-pointer-row { position:relative; height:16px; margin-bottom:-4px; user-select:none; touch-action:none; }
      .cs-timeline-pointer,.cs-subtitle-pointer { position:absolute; top:0; width:18px; height:16px; transform:translateX(-50%); padding:0; border:0; background:#55a9f5; clip-path:polygon(0 0,100% 0,50% 100%); cursor:ew-resize; z-index:5; }
      .cs-timeline-pointer:hover,.cs-subtitle-pointer:hover { background:#78bcff; }
      .cs-timeline-viewport,.cs-subtitle-viewport { position:relative; overflow:hidden; border:1px solid #363b45; border-radius:6px; background:#20232a; }
      .cs-timeline-axis,.cs-subtitle-axis { position:relative; height:22px; color:#9299a8; font-size:11px; font-variant-numeric:tabular-nums; }
      .cs-timeline-axis span,.cs-subtitle-axis span { position:absolute; transform:translateX(-50%); top:4px; }
      .cs-timeline-track,.cs-subtitle-track { position:relative; height:34px; border-top:1px solid #343943; cursor:crosshair; user-select:none; touch-action:none; }
      .cs-timeline-track-label,.cs-subtitle-track-label { position:absolute; left:8px; top:9px; z-index:1; color:#aeb5c2; font-size:11px; pointer-events:none; }
      .cs-timeline-track-body,.cs-subtitle-track-body { position:absolute; inset:0; margin-left:0; }
      .cs-timeline-track-body,.cs-subtitle-track-video .cs-subtitle-track-body { background:repeating-linear-gradient(90deg,#343941 0 1px,transparent 1px 10%); }
      .cs-timeline-range-band,.cs-subtitle-range-band { position:absolute; top:22px; bottom:0; left:0; right:0; z-index:1; pointer-events:none; background:rgba(188,198,210,.16); border-left:1px solid rgba(210,220,230,.6); border-right:1px solid rgba(210,220,230,.6); }
      .cs-range-marker { position:absolute; top:0; bottom:0; display:block; width:2px; background:#c9d4df; box-shadow:0 0 0 1px #15181d; cursor:ew-resize; z-index:2; touch-action:none; pointer-events:auto; }
      .cs-range-marker::after { content:""; position:absolute; top:0; bottom:0; left:-9px; width:20px; background:transparent; cursor:ew-resize; pointer-events:auto; }
      .cs-range-marker::before { content:""; position:absolute; top:-1px; width:0; height:0; border-left:5px solid transparent; border-right:5px solid transparent; border-top:6px solid #c9d4df; }
      .cs-range-marker.in { left:-1px; }
      .cs-range-marker.out { right:-1px; }
      .cs-range-marker.in::before { left:-4px; }
      .cs-range-marker.out::before { right:-4px; }
      .cs-range-marker:hover { background:#e1ebf4; }
      .cs-timeline-controls,.cs-subtitle-controls { display:flex; flex-wrap:wrap; gap:6px; }
      .cs-timeline-controls button,.cs-subtitle-controls button { border:1px solid #424956; border-radius:5px; padding:7px 12px; background:#242832; color:#e6e9ef; cursor:pointer; }
      .cs-timeline-controls button:hover,.cs-subtitle-controls button:hover { background:#303643; }
      .cs-point-frame,.cs-subtitle-point-frame { min-width:52px; padding-left:7px !important; padding-right:7px !important; color:#9fc9ec !important; font-variant-numeric:tabular-nums; }
    `;
    document.head.append(style);
}

export function createRangeController({
    root,
    video,
    getInfo,
    getRange,
    setRange,
    render,
    frameAt,
    seekFrame,
    getCurrentFrame,
    selectors = {},
    bindTrack = true,
}) {
    installTimelineRangeStyles();
    const track = root.querySelector(selectors.track || ".cs-timeline-track");
    const inHandle = root.querySelector(selectors.inHandle || ".cs-range-marker.in");
    const outHandle = root.querySelector(selectors.outHandle || ".cs-range-marker.out");
    const pointer = root.querySelector(selectors.pointer || ".cs-timeline-pointer");
    const pointerRow = root.querySelector(selectors.pointerRow || ".cs-timeline-pointer-row");
    const inFrame = root.querySelector(selectors.inFrame || ".cs-in-frame");
    const outFrame = root.querySelector(selectors.outFrame || ".cs-out-frame");
    const info = () => getInfo();
    const maxFrame = () => Math.max(0, (info()?.frames || 1) - 1);
    const current = () => getCurrentFrame ? getCurrentFrame() : clamp(Math.round((video.currentTime || 0) * (info()?.fps || 1)), 0, maxFrame());
    const seek = (frame) => seekFrame ? seekFrame(frame) : (info()?.fps ? video.currentTime = frame / info().fps : undefined);
    const locate = (event, element) => frameAt ? frameAt(event, element, info()) : Math.round(clamp((event.clientX - element.getBoundingClientRect().left) / element.getBoundingClientRect().width, 0, 1) * maxFrame());
    function normalize(changed = null) {
        const range = getRange();
        const last = maxFrame();
        range.start = clamp(Math.round(range.start), 0, last);
        range.end = clamp(Math.round(range.end < 0 ? last : range.end), 0, last);
        if (range.start < range.end || last === 0) { setRange(range); return range; }
        if (changed === "in") range.end = range.start >= last ? last : range.start + 1;
        else range.start = range.end <= 0 ? 0 : range.end - 1;
        setRange(range);
        return range;
    }
    function repaint(frame = null) { normalize(); render?.(frame); }
    function drag(which, event) {
        event.preventDefault(); event.stopPropagation();
        const move = (moveEvent) => { const range = getRange(); const frame = locate(moveEvent, track); if (which === "in") range.start = frame; else range.end = frame; setRange(range); normalize(which); seek(which === "in" ? getRange().start : getRange().end); repaint(which === "in" ? getRange().start : getRange().end); };
        const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); window.removeEventListener("pointercancel", up); };
        window.addEventListener("pointermove", move); window.addEventListener("pointerup", up); window.addEventListener("pointercancel", up); move(event);
    }
    function jump(frame) { video.pause(); seek(clamp(frame, 0, maxFrame())); repaint(frame); }
    function step(delta) { jump(current() + delta); }
    function setIn() { const range = getRange(); range.start = current(); setRange(range); normalize("in"); jump(getRange().start); }
    function setOut() { const range = getRange(); range.end = current(); setRange(range); normalize("out"); jump(getRange().end); }
    inHandle?.addEventListener("pointerdown", (event) => drag("in", event));
    outHandle?.addEventListener("pointerdown", (event) => drag("out", event));
    if (bindTrack && track) track.addEventListener("pointerdown", (event) => { if (event.target === inHandle || event.target === outHandle) return; const frame = locate(event, track); const range = getRange(); if (Math.abs(frame - range.start) <= Math.abs(frame - range.end)) range.start = frame, normalize("in"); else range.end = frame, normalize("out"); setRange(range); jump(frame); });
    if (pointer && pointerRow) {
        const beginPointerDrag = (event) => {
            event.preventDefault();
            event.stopPropagation();
            video.pause();
            const move = (moveEvent) => { const frame = locate(moveEvent, pointerRow); seek(frame); repaint(frame); };
            const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); window.removeEventListener("pointercancel", up); };
            window.addEventListener("pointermove", move); window.addEventListener("pointerup", up); window.addEventListener("pointercancel", up); move(event);
        };
        pointer.addEventListener("pointerdown", beginPointerDrag);
        pointerRow.addEventListener("pointerdown", (event) => { if (event.target !== pointer) beginPointerDrag(event); });
    }
    inFrame?.addEventListener("click", () => jump(getRange().start));
    outFrame?.addEventListener("click", () => jump(getRange().end));
    root.querySelector(selectors.setIn || ".cs-set-in")?.addEventListener("click", setIn);
    root.querySelector(selectors.setOut || ".cs-set-out")?.addEventListener("click", setOut);
    root.querySelector(selectors.back || ".cs-back")?.addEventListener("click", () => step(-1));
    root.querySelector(selectors.forward || ".cs-forward")?.addEventListener("click", () => step(1));
    return { render: repaint, normalize, jump, step, setIn, setOut, currentFrame: current };
}
