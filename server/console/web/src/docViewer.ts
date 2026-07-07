// @summary
// Source-document viewer side panel. Fetches a source document and renders it in
// an in-page <iframe> that splits the chat area 50/50 (the chat slides left),
// instead of opening a new browser tab. Shared by citation-card [view] links and
// the bubble "Sources:" links via citations.openSourceDocument.
// Exports: openDocViewer, closeDocViewer, initDocViewer
// Deps: dom, api, toast; type-only: citations.ViewPayload
// @end-summary

import { byId } from "./dom";
import { apiBase, authHeaders } from "./api";
import { showToast } from "./toast";
import type { ViewPayload } from "./citations";

// The single blob URL currently mounted in the iframe. Revoked when replaced or
// when the panel closes so we never leak object URLs across views.
let _blobUrl: string | null = null;

function _revoke(): void {
    if (_blobUrl) {
        URL.revokeObjectURL(_blobUrl);
        _blobUrl = null;
    }
}

function _panel(): HTMLElement {
    return byId("docViewer");
}

function _chatBody(): HTMLElement | null {
    return document.querySelector(".chat-body");
}

/** A human title for the panel header: the source's basename, else a fallback. */
function _titleFor(p: ViewPayload): string {
    const raw = String(p.source ?? p.source_uri ?? p.source_key ?? "").trim();
    if (!raw) return "Source document";
    return raw.split(/[\\/]/).pop() || raw;
}

/**
 * Open (or replace) the source document in the side panel and slide the chat
 * left so the two panes share the width equally. Fetching the document is the
 * same POST the old new-tab path used; only the rendering target changed.
 */
export async function openDocViewer(payload: ViewPayload): Promise<void> {
    const panel = _panel();
    const frame = byId<HTMLIFrameElement>("docViewerFrame");
    const status = byId("docViewerStatus");

    byId("docViewerTitle").textContent = _titleFor(payload);
    panel.classList.remove("is-loaded", "is-error");
    status.textContent = "Loading…";
    panel.setAttribute("aria-hidden", "false");
    _chatBody()?.classList.add("doc-open");

    try {
        const res = await fetch(apiBase() + "/console/source-document/view", {
            method: "POST",
            headers: authHeaders(),
            body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const contentType = (res.headers.get("Content-Type") ?? "text/html")
            .split(";")[0]
            .trim();
        const buf = await res.arrayBuffer();
        const blob = new Blob([buf], { type: contentType });
        _revoke();
        _blobUrl = URL.createObjectURL(blob);
        frame.src = _blobUrl;
        // The pop-out button mirrors the same content for users who still want a
        // real browser tab.
        byId<HTMLAnchorElement>("docViewerPopout").href = _blobUrl;
        panel.classList.add("is-loaded");
    } catch (err) {
        panel.classList.add("is-error");
        status.textContent = "Could not open source: " + String(err);
        showToast("Could not open source document");
    }
}

/** Close the panel and restore the full-width chat. The blob URL is deliberately
 *  NOT revoked here so the pop-out link keeps working after close; it is released
 *  when the next document opens (see `_revoke()` in openDocViewer), which bounds
 *  live blobs to at most one. */
export function closeDocViewer(): void {
    const panel = _panel();
    _chatBody()?.classList.remove("doc-open");
    panel.setAttribute("aria-hidden", "true");
    byId<HTMLIFrameElement>("docViewerFrame").removeAttribute("src");
    panel.classList.remove("is-loaded", "is-error");
}

export function initDocViewer(): void {
    byId("docViewerClose").addEventListener("click", closeDocViewer);
    document.addEventListener("keydown", (e) => {
        // Respect an Esc already consumed by a modal/menu (they preventDefault
        // without stopPropagation), so Esc-to-dismiss-modal doesn't also close us.
        if (e.defaultPrevented) return;
        if (e.key === "Escape" && _panel().getAttribute("aria-hidden") === "false") {
            closeDocViewer();
        }
    });
}
