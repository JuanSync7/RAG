/**
 * @summary
 * Conversation list, history pane, CRUD (create/compact/delete), and right-click
 * context menu for the admin/operator console.
 * Exports: setActiveConversation, renderConversationList, loadConversations,
 *   createConversation, loadConversationHistory, compactConversation,
 *   compactActiveConversation, deleteConversation, deleteConversationWithFeedback,
 *   deleteActiveConversation, hideConvContextMenu, bindConversations
 * Deps: admin-types, admin-state, admin-api, admin-render
 * @end-summary
 */

import type { ConversationMeta, ConversationTurn, JsonObject } from "./admin-types.js";
import {
    getActiveConversationId,
    setActiveConversationIdRaw,
    getConversationCache,
    setConversationCache,
    getConvContextMenuTarget,
    setConvContextMenuTarget,
    getConvMenuCloseHandler,
    setConvMenuCloseHandler,
    getConvMenuEscapeHandler,
    setConvMenuEscapeHandler,
} from "./admin-state.js";
import { api, byId } from "./admin-api.js";
import { escapeHtml, renderMarkdown } from "./admin-render.js";

export function setActiveConversation(id: string | null): void {
    setActiveConversationIdRaw(id);
    byId("activeConversationLabel").textContent = `Active: ${id || "(auto/new on query)"}`;
    renderConversationList();
}

export function renderConversationList(): void {
    const out = byId("conversationList");
    const cache = getConversationCache();
    if (!cache.length) {
        out.innerHTML = '<div class="muted">No conversations yet.</div>';
        return;
    }
    const activeId = getActiveConversationId();
    const rows = cache.map((item) => {
        const cid = item.conversation_id;
        const title = item.title || "New conversation";
        const msgCount = Number(item.message_count ?? 0);
        const klass = cid === activeId ? "conversation-item active" : "conversation-item";
        return `<button class="${klass}" data-conv-id="${escapeHtml(cid)}"><div>${escapeHtml(title)}</div><div class="muted">${escapeHtml(cid)} • ${msgCount} turns</div></button>`;
    });
    out.innerHTML = rows.join("");
    out.querySelectorAll<HTMLButtonElement>("[data-conv-id]").forEach((btn) => {
        const cid = btn.dataset.convId || "";
        btn.addEventListener("click", async () => {
            if (!cid) return;
            setActiveConversation(cid);
            await loadConversationHistory();
        });
        btn.addEventListener("contextmenu", (e) => {
            e.preventDefault();
            if (!cid) return;
            hideConvContextMenu();
            setConvContextMenuTarget(cid);
            const menu = byId("convContextMenu");
            menu.classList.add("visible");
            menu.style.left = `${e.clientX}px`;
            menu.style.top = `${e.clientY}px`;
            const closeHandler = (ev: MouseEvent) => {
                const target = ev.target as Node;
                if (target && menu.contains(target)) return;
                hideConvContextMenu();
            };
            const escapeHandler = (ev: KeyboardEvent) => {
                if (ev.key === "Escape") hideConvContextMenu();
            };
            setConvMenuCloseHandler(closeHandler);
            setConvMenuEscapeHandler(escapeHandler);
            document.addEventListener("mousedown", closeHandler);
            document.addEventListener("keydown", escapeHandler);
        });
    });
}

export async function loadConversations(): Promise<void> {
    const out = await api("GET", "/console/conversations?limit=50");
    const data = (out.data as JsonObject | undefined) || {};
    const items = Array.isArray(data.conversations) ? (data.conversations as ConversationMeta[]) : [];
    setConversationCache(items);
    if (!getActiveConversationId() && items.length) {
        setActiveConversation(items[0].conversation_id);
    }
    renderConversationList();
}

export async function createConversation(title = "New conversation"): Promise<void> {
    const out = await api("POST", "/console/conversations/new", { title });
    const data = (out.data as JsonObject | undefined) || {};
    const conv = (data.conversation as ConversationMeta | undefined) || null;
    if (conv && conv.conversation_id) {
        setActiveConversation(conv.conversation_id);
    }
    await loadConversations();
    await loadConversationHistory();
}

export async function loadConversationHistory(): Promise<void> {
    const pane = byId("chatHistoryPane");
    const activeId = getActiveConversationId();
    if (!activeId) {
        pane.textContent = "No history yet.";
        return;
    }
    const out = await api("GET", `/console/conversations/${encodeURIComponent(activeId)}/history?limit=60`);
    const data = (out.data as JsonObject | undefined) || {};
    const turns = Array.isArray(data.turns) ? (data.turns as ConversationTurn[]) : [];
    if (!turns.length) {
        pane.textContent = "No history yet.";
        return;
    }
    pane.textContent = turns
        .map((turn) => `${turn.role.toUpperCase()}: ${turn.content}`)
        .join("\n\n");
    pane.scrollTop = pane.scrollHeight;
}

export async function compactConversation(conversationId: string): Promise<void> {
    try {
        const payload = await api("POST", `/console/conversations/${encodeURIComponent(conversationId)}/compact`, {
            conversation_id: conversationId,
        });
        const data = (payload.data as JsonObject | undefined) || {};
        const summary = String(data.summary ?? "").trim();
        if (summary) {
            renderMarkdown("queryMarkdown", `### Compacted summary\n\n${summary}`);
        }
        await loadConversations();
        await loadConversationHistory();
    } catch (err) {
        renderMarkdown("queryMarkdown", `**Compact error:** ${escapeHtml(String(err))}`);
    }
}

export async function compactActiveConversation(): Promise<void> {
    const activeId = getActiveConversationId();
    if (!activeId) {
        renderMarkdown("queryMarkdown", "**/compact:** Select or create a conversation first.");
        return;
    }
    await compactConversation(activeId);
}

export async function deleteConversation(conversationId: string): Promise<boolean> {
    const payload = await api("DELETE", `/console/conversations/${encodeURIComponent(conversationId)}`);
    const data = (payload.data as JsonObject | undefined) || {};
    return Boolean(data.deleted);
}

export async function deleteConversationWithFeedback(conversationId: string): Promise<void> {
    try {
        const deleted = await deleteConversation(conversationId);
        if (deleted && getActiveConversationId() === conversationId) {
            setActiveConversation(null);
        }
        await loadConversations();
        await loadConversationHistory();
        if (deleted) {
            renderMarkdown("queryMarkdown", `**Deleted** conversation \`${escapeHtml(conversationId)}\`.`);
        } else {
            renderMarkdown("queryMarkdown", "**Delete:** Conversation not found (may already be deleted).");
        }
    } catch (err) {
        renderMarkdown("queryMarkdown", `**Delete error:** ${escapeHtml(String(err))}`);
    }
}

export async function deleteActiveConversation(): Promise<void> {
    const activeId = getActiveConversationId();
    if (!activeId) {
        renderMarkdown("queryMarkdown", "**/delete:** Select a conversation to delete first.");
        return;
    }
    await deleteConversationWithFeedback(activeId);
}

export function hideConvContextMenu(): void {
    const menu = byId("convContextMenu");
    menu.classList.remove("visible");
    setConvContextMenuTarget(null);
    const closeHandler = getConvMenuCloseHandler();
    if (closeHandler) {
        document.removeEventListener("mousedown", closeHandler);
        setConvMenuCloseHandler(null);
    }
    const escapeHandler = getConvMenuEscapeHandler();
    if (escapeHandler) {
        document.removeEventListener("keydown", escapeHandler);
        setConvMenuEscapeHandler(null);
    }
}

export function bindConversations(): void {
    byId("newConversationBtn").addEventListener("click", async () => {
        await createConversation("New conversation");
    });
    byId("refreshConversationsBtn").addEventListener("click", async () => {
        await loadConversations();
        await loadConversationHistory();
    });
    byId("compactConversationBtn").addEventListener("click", async () => {
        await compactActiveConversation();
    });
    byId("deleteConversationBtn").addEventListener("click", async () => {
        await deleteActiveConversation();
    });

    const menu = byId("convContextMenu");
    menu.querySelectorAll<HTMLButtonElement>("[data-action]").forEach((btn) => {
        btn.addEventListener("click", async (e) => {
            e.stopPropagation();
            const cid = getConvContextMenuTarget();
            hideConvContextMenu();
            if (!cid) return;
            if (btn.dataset.action === "compact") {
                await compactConversation(cid);
            } else if (btn.dataset.action === "delete") {
                await deleteConversationWithFeedback(cid);
            }
        });
    });
}
