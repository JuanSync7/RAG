// @summary
// Toast notifications + clipboard-copy helpers (message bubbles + code blocks).
// Exposes copyMsg/copyBubble/showToast on `window` for the few inline onclick
// handlers still emitted by the markdown renderer and bubble templates.
// @end-summary

import { byId } from "./dom";
import { refs } from "./refs";

export function showToast(msg: string): void {
    const t = byId("toast");
    t.textContent = msg;
    t.classList.add("show");
    setTimeout(() => t.classList.remove("show"), 2200);
}

export function copyMsg(btn: HTMLElement, text: string): void {
    navigator.clipboard.writeText(text).then(() => {
        btn.classList.add("copied");
        btn.textContent = "✓ Copied";
        setTimeout(() => {
            btn.classList.remove("copied");
            btn.innerHTML = "&#128203; Copy";
        }, 2000);
    });
    showToast("Copied to clipboard");
}

export function copyBubble(btn: HTMLElement, id: string): void {
    copyMsg(btn, document.getElementById(id)?.innerText || "");
}

export function initToast(): void {
    (window as unknown as Record<string, unknown>)["copyMsg"] = copyMsg;
    (window as unknown as Record<string, unknown>)["copyBubble"] = copyBubble;
    (window as unknown as Record<string, unknown>)["showToast"] = showToast;

    // Single delegated handler for code-block copy buttons emitted by marked.
    refs.thread.addEventListener("click", (e) => {
        const btn = (e.target as HTMLElement).closest<HTMLElement>(".copy-code-btn");
        if (!btn) return;
        const codeDiv = btn.closest(".code-block-wrap")?.querySelector<HTMLElement>(".code-block");
        if (codeDiv) {
            navigator.clipboard.writeText(codeDiv.textContent ?? "");
            showToast("Code copied");
        }
    });
}
