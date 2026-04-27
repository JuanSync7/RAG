// @summary
// Conversation list, history loading, context menu, rename modal, delete.
// Also owns updateConvTitle + createNewConversation for cross-feature use by
// streaming/slash submission.
// @end-summary

import { byId, escHtml, fmtRelative, fmtTime } from "./dom";
import { api } from "./api";
import { parseMarkdown } from "./markdown";
import { refs } from "./refs";
import { state, setActiveConversation } from "./state";
import { showToast, copyMsg } from "./toast";
import { appendUserMsg, appendErrorMsg, setEmptyState } from "./thread";
import { buildCitationsHtml } from "./citations";
import { sourceRefToChunkResult } from "./user-types";
import type { ConversationMeta, SourceRef } from "./user-types";

export function renderConversationList(convs: ConversationMeta[]): void {
    const container = byId("convList");
    if (!convs.length) {
        container.innerHTML = `<div class="conv-list-empty">No conversations yet.<br>Start one below!</div>`;
        return;
    }
    const groups: Record<string, ConversationMeta[]> = {};
    convs.forEach((c) => {
        const label = fmtRelative(c.updated_at_ms ?? Date.now());
        if (!groups[label]) groups[label] = [];
        groups[label].push(c);
    });
    let html = "";
    for (const [label, items] of Object.entries(groups)) {
        html += `<div class="conv-section-label">${escHtml(label)}</div>`;
        items.forEach((c) => {
            const isActive = c.conversation_id === state.activeConversationId;
            const title = escHtml(c.title || c.conversation_id);
            html += `
              <div class="conv-item-wrap">
                <div class="conv-item${isActive ? " active" : ""}" data-conv-id="${escHtml(c.conversation_id)}" title="${title}">
                  <span class="dot"></span>${title}
                </div>
                <button class="conv-item-del" data-conv-id="${escHtml(c.conversation_id)}" title="Delete">&#10005;</button>
              </div>`;
        });
    }
    container.innerHTML = html;

    container.querySelectorAll<HTMLElement>(".conv-item").forEach((el) => {
        el.addEventListener("click", () => {
            const id = el.dataset.convId;
            if (id) void selectConversation(id);
        });
        el.addEventListener("contextmenu", (e) => {
            e.preventDefault();
            const id = el.dataset.convId;
            if (!id) return;
            const conv = convs.find((c) => c.conversation_id === id);
            showConvCtxMenu(e.clientX, e.clientY, id, conv?.title || "");
        });
    });
    container.querySelectorAll<HTMLElement>(".conv-item-del").forEach((btn) => {
        btn.addEventListener("click", (e) => {
            e.stopPropagation();
            const id = btn.dataset.convId;
            if (id) void deleteConversation(id);
        });
    });
}

export async function loadConversations(): Promise<void> {
    try {
        const data = await api<{ conversations: ConversationMeta[] }>(
            "GET",
            "/console/conversations?limit=50"
        );
        renderConversationList(data.conversations || []);
    } catch {
        // Non-fatal — sidebar just stays empty
    }
}

export async function selectConversation(id: string): Promise<void> {
    if (state.isStreaming) {
        state.streamAbortCtrl?.abort();
        state.isStreaming = false;
    }
    setActiveConversation(id);
    byId("convList").querySelectorAll<HTMLElement>(".conv-item").forEach((el) => {
        el.classList.toggle("active", el.dataset.convId === id);
    });
    await loadConversationHistory(id);
}

export async function loadConversationHistory(id?: string): Promise<void> {
    const convId = id ?? state.activeConversationId;
    if (!convId) return;
    try {
        const data = await api<{
            conversation_id: string;
            turns: Array<{ role: string; content: string; timestamp_ms?: number; sources?: SourceRef[] }>;
        }>("GET", `/console/conversations/${convId}/history?limit=100`);
        refs.thread.innerHTML = "";
        if (!data.turns || !data.turns.length) {
            refs.thread.innerHTML = `<div class="thread-empty" id="threadEmpty"><div class="thread-empty-icon">&#128172;</div><div class="thread-empty-title">Empty conversation</div><div class="thread-empty-sub">Send a message to start the conversation.</div></div>`;
            return;
        }
        data.turns.forEach((turn) => {
            if (turn.role === "user") {
                appendUserMsg(turn.content);
            } else {
                setEmptyState(false);
                const group = document.createElement("div");
                group.className = "msg-group";
                const ts = fmtTime(turn.timestamp_ms ?? Date.now());
                const sources = turn.sources ?? [];
                const citationsHtml = sources.length
                    ? `<div class="citations">${buildCitationsHtml(sources.map(sourceRefToChunkResult))}</div>`
                    : "";
                group.innerHTML = `
                    <div class="msg-row assistant">
                      <div class="avatar ai-av">AI</div>
                      <div class="bubble-wrap">
                        <div class="bubble">${parseMarkdown(turn.content)}</div>
                        ${citationsHtml}
                        <div class="msg-actions">
                          <button class="msg-action-btn">&#128203; Copy</button>
                        </div>
                        <div class="msg-meta">${ts}</div>
                      </div>
                    </div>`;
                const copyBtn = group.querySelector<HTMLElement>(".msg-action-btn");
                if (copyBtn) {
                    const bubbleEl = group.querySelector<HTMLElement>(".bubble")!;
                    copyBtn.addEventListener("click", () => copyMsg(copyBtn, bubbleEl.innerText));
                }
                refs.thread.appendChild(group);
            }
        });
        const activeConv = document.querySelector<HTMLElement>(`.conv-item[data-conv-id="${convId}"]`);
        if (activeConv) {
            byId("convTitle").textContent =
                activeConv.title ?? activeConv.textContent?.trim() ?? "Conversation";
        }
        setTimeout(() => {
            refs.thread.scrollTop = refs.thread.scrollHeight;
        }, 50);
    } catch (err) {
        appendErrorMsg("Failed to load conversation history: " + String(err));
    }
}

export function createNewConversation(): void {
    setActiveConversation(null);
    refs.thread.innerHTML = `<div class="thread-empty" id="threadEmpty"><div class="thread-empty-icon">&#128172;</div><div class="thread-empty-title">New conversation</div><div class="thread-empty-sub">Send a message to get started.</div></div>`;
    byId("convTitle").textContent = "New conversation";
    byId("convList").querySelectorAll<HTMLElement>(".conv-item").forEach((el) => {
        el.classList.remove("active");
    });
    const input = document.getElementById("msgInput") as HTMLTextAreaElement | null;
    if (input) input.focus();
}

export async function deleteConversation(id: string): Promise<void> {
    try {
        await api("DELETE", `/console/conversations/${id}`);
        if (state.activeConversationId === id) {
            setActiveConversation(null);
            refs.thread.innerHTML = `<div class="thread-empty" id="threadEmpty"><div class="thread-empty-icon">&#9670;</div><div class="thread-empty-title">RagWeave</div><div class="thread-empty-sub">Ask anything — I'll search your knowledge base and generate a response with sources.</div></div>`;
            byId("convTitle").textContent = "RagWeave";
        }
        await loadConversations();
        showToast("Conversation deleted");
    } catch {
        showToast("Failed to delete conversation");
    }
}

export function updateConvTitle(): void {
    if (!state.activeConversationId) return;
    const item = byId("convList").querySelector<HTMLElement>(
        `.conv-item[data-conv-id="${state.activeConversationId}"]`
    );
    if (item) {
        byId("convTitle").textContent =
            item.textContent?.trim().replace(/^●/, "").trim() ?? "Conversation";
    }
}

// ── Context menu + rename modal ──

function hideConvCtxMenu(): void {
    const menu = byId("convCtxMenu");
    menu.classList.remove("open");
    menu.setAttribute("aria-hidden", "true");
    state.convMenuTargetId = null;
}

function showConvCtxMenu(x: number, y: number, id: string, title: string): void {
    const menu = byId("convCtxMenu");
    state.convMenuTargetId = id;
    state.convMenuTargetTitle = title;
    menu.style.left = "-9999px";
    menu.style.top = "-9999px";
    menu.classList.add("open");
    menu.setAttribute("aria-hidden", "false");
    const rect = menu.getBoundingClientRect();
    const maxX = window.innerWidth - rect.width - 8;
    const maxY = window.innerHeight - rect.height - 8;
    menu.style.left = `${Math.max(8, Math.min(x, maxX))}px`;
    menu.style.top = `${Math.max(8, Math.min(y, maxY))}px`;
}

function openRenameModal(id: string, currentTitle: string): void {
    state.renameTargetId = id;
    const overlay = byId("renameModal");
    const input = byId<HTMLInputElement>("renameInput");
    input.value = currentTitle || "";
    overlay.classList.add("open");
    overlay.setAttribute("aria-hidden", "false");
    setTimeout(() => {
        input.focus();
        input.select();
    }, 40);
}

function closeRenameModal(): void {
    const overlay = byId("renameModal");
    overlay.classList.remove("open");
    overlay.setAttribute("aria-hidden", "true");
    state.renameTargetId = null;
}

async function submitRename(): Promise<void> {
    const id = state.renameTargetId;
    if (!id) return;
    const input = byId<HTMLInputElement>("renameInput");
    const trimmed = input.value.trim();
    if (!trimmed) {
        input.focus();
        return;
    }
    closeRenameModal();
    try {
        await api("PATCH", `/console/conversations/${encodeURIComponent(id)}`, { title: trimmed });
        if (state.activeConversationId === id) {
            byId("convTitle").textContent = trimmed;
        }
        await loadConversations();
        showToast("Conversation renamed");
    } catch (err) {
        showToast("Failed to rename: " + String(err));
    }
}

export function initConversations(): void {
    byId("convCtxMenu").querySelectorAll<HTMLButtonElement>("[data-action]").forEach((btn) => {
        btn.addEventListener("click", (e) => {
            e.stopPropagation();
            const id = state.convMenuTargetId;
            const title = state.convMenuTargetTitle;
            const action = btn.dataset.action;
            hideConvCtxMenu();
            if (!id) return;
            if (action === "rename") openRenameModal(id, title);
            else if (action === "delete") void deleteConversation(id);
        });
    });

    document.addEventListener("mousedown", (e) => {
        const menu = byId("convCtxMenu");
        if (!menu.classList.contains("open")) return;
        if (e.target instanceof Node && menu.contains(e.target)) return;
        hideConvCtxMenu();
    });
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            hideConvCtxMenu();
            if (byId("renameModal").classList.contains("open")) closeRenameModal();
        }
    });
    window.addEventListener("resize", hideConvCtxMenu);
    window.addEventListener("scroll", hideConvCtxMenu, true);

    byId("renameCancel").addEventListener("click", closeRenameModal);
    byId("renameSave").addEventListener("click", () => void submitRename());
    byId("renameInput").addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            void submitRename();
        } else if (e.key === "Escape") {
            e.preventDefault();
            closeRenameModal();
        }
    });
    byId("renameModal").addEventListener("click", (e) => {
        if (e.target === e.currentTarget) closeRenameModal();
    });

    byId("newChatBtn").addEventListener("click", createNewConversation);

    document.getElementById("convSearch")?.addEventListener("input", (e: Event) => {
        const q = (e.target as HTMLInputElement).value.toLowerCase();
        byId("convList").querySelectorAll<HTMLElement>(".conv-item-wrap").forEach((wrap) => {
            const text = wrap.querySelector<HTMLElement>(".conv-item")?.textContent?.toLowerCase() || "";
            wrap.style.display = text.includes(q) ? "" : "none";
        });
    });
}
