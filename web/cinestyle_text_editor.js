const STYLE_ID = "cinestyle-shared-text-editor-style";
const DEFAULT_FONT_SIZE = 16;
const MIN_FONT_SIZE = 12;
const MAX_FONT_SIZE = 32;

function addStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .cs-shared-text-editor { position:fixed; inset:0; z-index:40; display:grid; place-items:center; padding:20px; background:#05060999; }
      .cs-shared-text-editor[hidden] { display:none; }
      .cs-shared-text-editor-panel { display:grid; width:min(520px,calc(100vw - 40px)); gap:12px; padding:16px; border:1px solid #424956; border-radius:8px; background:#17191e; box-shadow:0 18px 60px #000b; }
      .cs-shared-text-editor-title { margin:0; color:#e6e9ef; font-size:15px; font-weight:600; }
      .cs-shared-text-editor-input { width:100%; min-height:110px; box-sizing:border-box; resize:vertical; border:1px solid #424956; border-radius:5px; padding:9px; background:#111419; color:#f2f4f7; font:16px/1.45 system-ui,sans-serif; }
      .cs-shared-text-editor-input:focus { outline:0; border-color:#6aa9df; box-shadow:0 0 0 2px #317ec455; }
      .cs-shared-text-editor-size { display:flex; align-items:center; justify-content:flex-end; gap:6px; color:#9299a8; }
      .cs-shared-text-editor-size output { min-width:42px; color:#f7b955; text-align:center; font-variant-numeric:tabular-nums; }
      .cs-shared-text-editor-size button { width:32px; height:28px; border:1px solid #424956; border-radius:5px; background:#242832; color:#e6e9ef; cursor:pointer; font-weight:600; }
      .cs-shared-text-editor-size button:hover { border-color:#6aa9df; background:#303643; }
      .cs-shared-text-editor-size button:disabled { opacity:.45; cursor:not-allowed; }
      .cs-shared-text-editor-actions { display:flex; justify-content:flex-end; gap:7px; }
      .cs-shared-text-editor-actions button { min-height:31px; border:1px solid #424956; border-radius:5px; padding:6px 12px; background:#242832; color:#e6e9ef; cursor:pointer; }
      .cs-shared-text-editor-actions button:hover { background:#303643; }
      .cs-shared-text-editor-actions .confirm { background:#317ec4; border-color:#4b9de8; }
      .cs-shared-text-editor-actions .confirm:disabled { opacity:.45; cursor:not-allowed; }
    `;
    document.head.append(style);
}

export function createCineStyleTextEditor(host) {
    if (!host || typeof host.append !== "function") throw new TypeError("A host element is required.");
    addStyles();
    const root = document.createElement("div");
    root.className = "cs-shared-text-editor";
    root.hidden = true;
    root.setAttribute("role", "dialog");
    root.setAttribute("aria-modal", "true");
    root.innerHTML = `
      <div class="cs-shared-text-editor-panel">
        <h3 class="cs-shared-text-editor-title"></h3>
        <textarea class="cs-shared-text-editor-input" rows="4"></textarea>
        <div class="cs-shared-text-editor-size" aria-label="Text display size">
          <button type="button" class="decrease-size" aria-label="Decrease text size" title="Decrease text size">A−</button>
          <output class="size-value">16px</output>
          <button type="button" class="increase-size" aria-label="Increase text size" title="Increase text size">A+</button>
        </div>
        <div class="cs-shared-text-editor-actions">
          <button type="button" class="cancel">Cancel</button>
          <button type="button" class="confirm">Save</button>
        </div>
      </div>`;
    host.append(root);
    const title = root.querySelector(".cs-shared-text-editor-title");
    const input = root.querySelector(".cs-shared-text-editor-input");
    const decreaseSize = root.querySelector(".decrease-size");
    const increaseSize = root.querySelector(".increase-size");
    const sizeValue = root.querySelector(".size-value");
    const cancel = root.querySelector(".cancel");
    const confirm = root.querySelector(".confirm");
    let pending = null;
    let fontSize = DEFAULT_FONT_SIZE;

    function syncFontSize() {
        input.style.fontSize = `${fontSize}px`;
        sizeValue.textContent = `${fontSize}px`;
        decreaseSize.disabled = fontSize <= MIN_FONT_SIZE;
        increaseSize.disabled = fontSize >= MAX_FONT_SIZE;
    }

    function syncConfirmState() {
        confirm.disabled = Boolean(pending && !pending.allowEmpty && !input.value.trim());
    }
    function finish(value) {
        if (!pending) return;
        const state = pending;
        pending = null;
        root.hidden = true;
        state.resolve(value);
    }
    function confirmValue() {
        if (!pending) return;
        if (!pending.allowEmpty && !input.value.trim()) {
            input.focus();
            return;
        }
        finish(input.value);
    }
    function open(options = {}) {
        if (pending) finish(null);
        return new Promise((resolve) => {
            pending = {
                resolve,
                allowEmpty: Boolean(options.allowEmpty),
            };
            title.textContent = String(options.title || "Edit text");
            input.value = String(options.value || "");
            syncFontSize();
            cancel.textContent = String(options.cancelLabel || "Cancel");
            confirm.textContent = String(options.confirmLabel || "Save");
            syncConfirmState();
            root.hidden = false;
            requestAnimationFrame(() => { input.focus(); input.select(); });
        });
    }
    input.addEventListener("input", syncConfirmState);
    decreaseSize.addEventListener("click", () => { fontSize = Math.max(MIN_FONT_SIZE, fontSize - 2); syncFontSize(); input.focus(); });
    increaseSize.addEventListener("click", () => { fontSize = Math.min(MAX_FONT_SIZE, fontSize + 2); syncFontSize(); input.focus(); });
    input.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            event.preventDefault();
            event.stopPropagation();
            finish(null);
        } else if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
            event.preventDefault();
            confirmValue();
        }
    });
    cancel.addEventListener("click", () => finish(null));
    confirm.addEventListener("click", confirmValue);
    root.addEventListener("pointerdown", (event) => {
        if (event.target === root) finish(null);
    });

    return {
        open,
        close: () => finish(null),
        destroy: () => { finish(null); root.remove(); },
        isOpen: () => Boolean(pending),
    };
}
