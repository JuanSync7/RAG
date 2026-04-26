// @summary
// Message-thread rendering helpers: append user/system/error bubbles, build a
// pending assistant bubble (used by streaming + non-stream paths), and toggle
// the empty-state placeholder.
// @end-summary

import { escHtml, fmtTime } from "./dom";
import { parseMarkdown } from "./markdown";
import { refs } from "./refs";
import { copyMsg } from "./toast";
import { scrollToBottom } from "./scrollFab";

export function setEmptyState(visible: boolean): void {
    const el = document.getElementById("threadEmpty");
    if (el) el.style.display = visible ? "" : "none";
}

export function appendUserMsg(text: string): HTMLElement {
    setEmptyState(false);
    const ts = fmtTime(Date.now());
    const group = document.createElement("div");
    group.className = "msg-group";
    group.innerHTML = `
        <div class="msg-row user">
          <div class="avatar user-av">U</div>
          <div class="bubble-wrap">
            <div class="bubble">${escHtml(text)}</div>
            <div class="msg-actions">
              <button class="msg-action-btn" onclick="copyMsg(this,'${escHtml(text)}')" >&#128203; Copy</button>
            </div>
            <div class="msg-meta">${ts}</div>
          </div>
        </div>`;
    refs.thread.appendChild(group);
    scrollToBottom();
    return group;
}

export interface PendingAssistantHandles {
    group: HTMLElement;
    bubbleEl: HTMLElement;
    typingEl: HTMLElement;
    citationsEl: HTMLElement;
    actionsEl: HTMLElement;
    metaEl: HTMLElement;
    fbUpBtn: HTMLButtonElement;
    fbDownBtn: HTMLButtonElement;
}

/** Creates a pending assistant bubble with typing indicator. */
export function appendPendingAssistant(): PendingAssistantHandles {
    setEmptyState(false);
    const group = document.createElement("div");
    group.className = "msg-group";
    group.innerHTML = `
        <div class="msg-row assistant">
          <div class="avatar ai-av">AI</div>
          <div class="bubble-wrap">
            <div class="typing-indicator">
              <div class="typing-dot"></div>
              <div class="typing-dot"></div>
              <div class="typing-dot"></div>
            </div>
            <div class="bubble" style="display:none"></div>
            <div class="citations" style="display:none"></div>
            <div class="msg-actions" style="display:none">
              <button class="msg-action-btn">&#128203; Copy</button>
              <button class="msg-action-btn">&#128257; Regenerate</button>
              <button class="msg-action-btn fb-btn fb-up" title="Helpful" data-rating="up">&#128077;</button>
              <button class="msg-action-btn fb-btn fb-down" title="Not helpful" data-rating="down">&#128078;</button>
            </div>
            <div class="msg-meta" style="display:none"></div>
          </div>
        </div>`;
    refs.thread.appendChild(group);
    scrollToBottom();
    const bw = group.querySelector(".bubble-wrap")!;
    const bubbleEl = bw.querySelector<HTMLElement>(".bubble")!;
    const typingEl = bw.querySelector<HTMLElement>(".typing-indicator")!;
    const citationsEl = bw.querySelector<HTMLElement>(".citations")!;
    const actionsEl = bw.querySelector<HTMLElement>(".msg-actions")!;
    const metaEl = bw.querySelector<HTMLElement>(".msg-meta")!;
    const copyBtn = actionsEl.querySelector<HTMLButtonElement>("button");
    if (copyBtn) {
        copyBtn.addEventListener("click", () => copyMsg(copyBtn, bubbleEl.innerText));
    }
    const fbUpBtn = actionsEl.querySelector<HTMLButtonElement>(".fb-up")!;
    const fbDownBtn = actionsEl.querySelector<HTMLButtonElement>(".fb-down")!;
    return { group, bubbleEl, typingEl, citationsEl, actionsEl, metaEl, fbUpBtn, fbDownBtn };
}

export function appendSystemMsg(text: string): void {
    const div = document.createElement("div");
    div.className = "msg-group";
    div.innerHTML = `<div class="msg-row assistant"><div class="avatar ai-av">&#9432;</div><div class="bubble-wrap"><div class="bubble">${parseMarkdown(text)}</div></div></div>`;
    refs.thread.appendChild(div);
    scrollToBottom();
}

export function appendErrorMsg(text: string): void {
    const div = document.createElement("div");
    div.className = "msg-group";
    div.innerHTML = `<div class="msg-row assistant"><div class="avatar ai-av">!</div><div class="bubble-wrap"><div class="bubble error-bubble">&#9888; ${escHtml(text)}</div></div></div>`;
    refs.thread.appendChild(div);
    scrollToBottom();
}
