import { api } from "../../../scripts/api.js";

const STYLE_ID = "cinestyle-preview-cache-notice-style";
const EVENT_NAME = "cinestyle_preview_cache_ready";
const NODE_NAMES = {
    CS_Image_Composite: "CS Image Composite",
    CS_Video_Timeline_Edit: "CS Video Timeline Edit",
    CS_MatAnyone2: "CS MatAnyone2",
    CS_Color_Match: "CS Color Match",
    CS_Color_Grade: "CS Color Grade",
    CS_Video_Subtitle: "CS Video Subtitle",
    CS_VFX_Beauty: "CS VFX Beauty",
    CS_Video_Segment_SAM3: "CS Video Segment (SAM3.1)",
    CS_Video_Segment_SeC: "CS Video Segment (SeC-4B)",
};

function addStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .cs-preview-cache-notice{width:min(620px,92vw);max-width:none;max-height:86vh;padding:0;border:1px solid #3b414d;border-radius:9px;background:#17191e;color:#e8ebef;box-shadow:0 22px 80px #000b;font:13px/1.45 system-ui,sans-serif}
      .cs-preview-cache-notice::backdrop{background:#050609b8}
      .cs-preview-cache-notice-shell{display:grid;gap:14px;padding:18px}
      .cs-preview-cache-notice-title{margin:0;color:#f1f4f8;font-size:16px;font-weight:600}
      .cs-preview-cache-notice-list{display:grid;gap:10px;max-height:calc(86vh - 80px);overflow:auto;padding-right:2px}
      .cs-preview-cache-notice-item{display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:center;gap:14px;padding:13px 14px;border:1px solid #343b47;border-radius:7px;background:#20232a}
      .cs-preview-cache-notice-message{margin:0;color:#e8ebef;white-space:normal}
      .cs-preview-cache-notice-actions{display:flex;justify-content:flex-end}
      .cs-preview-cache-notice-ok{min-width:74px;min-height:31px;border:1px solid #5b8db8;border-radius:5px;padding:5px 14px;background:#2879b8;color:#f5f8fb;cursor:pointer}
      .cs-preview-cache-notice-ok:hover{background:#3189ca;border-color:#8cc8f2}
      @media(max-width:560px){.cs-preview-cache-notice-item{grid-template-columns:1fr}.cs-preview-cache-notice-actions{justify-content:flex-start}}
    `;
    document.head.append(style);
}

let noticeDialog = null;
let noticeList = null;

function closeDialog(dialog = noticeDialog) {
    if (!dialog) return;
    if (dialog.open) dialog.close();
    dialog.remove();
    if (noticeDialog === dialog) {
        noticeDialog = null;
        noticeList = null;
    }
}

function ensureDialog() {
    if (noticeDialog?.isConnected && noticeList?.isConnected) return noticeDialog;
    addStyles();

    const dialog = document.createElement("dialog");
    dialog.className = "cs-preview-cache-notice";
    const shell = document.createElement("div");
    shell.className = "cs-preview-cache-notice-shell";
    const title = document.createElement("h2");
    title.className = "cs-preview-cache-notice-title";
    title.textContent = "Preview Cache";
    const list = document.createElement("div");
    list.className = "cs-preview-cache-notice-list";
    shell.append(title, list);
    dialog.append(shell);
    document.body.append(dialog);
    noticeDialog = dialog;
    noticeList = list;
    dialog.addEventListener("close", () => {
        if (noticeDialog === dialog) {
            noticeDialog = null;
            noticeList = null;
        }
        dialog.remove();
    }, { once: true });
    dialog.addEventListener("cancel", (event) => event.preventDefault());
    dialog.showModal();
    return dialog;
}

function showPreviewCacheNotice(detail) {
    const payload = detail && typeof detail === "object" ? detail : {};
    const nodeType = String(payload.node_type || "").trim();
    const nodeName = String(payload.node_name || NODE_NAMES[nodeType] || nodeType || "CineStyle node");
    const dialog = ensureDialog();
    const item = document.createElement("div");
    item.className = "cs-preview-cache-notice-item";
    const message = document.createElement("p");
    message.className = "cs-preview-cache-notice-message";
    message.textContent = `Preview Cache created, you can open the Edit Window of ${nodeName} to preview and edit it now.`;
    const actions = document.createElement("div");
    actions.className = "cs-preview-cache-notice-actions";
    const ok = document.createElement("button");
    ok.className = "cs-preview-cache-notice-ok";
    ok.type = "button";
    ok.textContent = "OK";
    actions.append(ok);
    item.append(message, actions);
    noticeList.append(item);
    ok.addEventListener("click", () => {
        item.remove();
        if (!noticeList?.childElementCount) closeDialog(dialog);
    });
    ok.focus();
}

if (!globalThis.__cinestylePreviewCacheNoticeInstalled) {
    globalThis.__cinestylePreviewCacheNoticeInstalled = true;
    api.addEventListener(EVENT_NAME, ({ detail }) => showPreviewCacheNotice(detail));
}
