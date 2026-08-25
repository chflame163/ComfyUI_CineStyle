import { installTimelineRangeStyles, createRangeController } from "./cinestyle_timeline_range.js";

export function installTimelineControlStyles() {
    installTimelineRangeStyles();
}

export function timelineControlsMarkup() {
    return `
      <div class="cs-timeline-pointer-row" aria-label="Current frame"><button class="cs-timeline-pointer" type="button" aria-label="Drag current frame" title="Drag current frame"></button></div>
      <div class="cs-timeline-viewport"><div class="cs-timeline-axis"></div><div class="cs-timeline-range-band"><span class="cs-range-marker in" role="slider" aria-label="In point" tabindex="0"></span><span class="cs-range-marker out" role="slider" aria-label="Out point" tabindex="0"></span></div><div class="cs-timeline-track"><span class="cs-timeline-track-label">Video</span><div class="cs-timeline-track-body"></div></div></div>
      <div class="cs-timeline-controls"><button class="cs-set-in" type="button">Set In</button><button class="cs-point-frame cs-in-frame" type="button">0</button><button class="cs-back" type="button">|&lt;</button><button class="cs-play" type="button">Play</button><button class="cs-forward" type="button">&gt;|</button><button class="cs-point-frame cs-out-frame" type="button">0</button><button class="cs-set-out" type="button">Set Out</button></div>`;
}

export function createTimelineRangeController(options) {
    installTimelineControlStyles();
    return createRangeController(options);
}
