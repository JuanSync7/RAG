// @summary
// Retrieval view-tab for the user console. Lets users query the knowledge
// base directly and manage per-conversation document visibility (hide/restore).
// Fresh user-facing design — not a port of the admin retrieval tab.
// Exports: initRetrievalView
// Deps: api, dom, toast, state, user-types
// @end-summary

import { api } from "./api";
import { escHtml } from "./dom";
import { showToast } from "./toast";
import { state } from "./state";
import type { RetrievalResultItem, RetrievalResponse, DocStateResponse } from "./user-types";

// ── In-memory state ──────────────────────────────────────────────────────────

interface RetrievalState {
    results: RetrievalResultItem[];
    ignoredDocIds: Set<string>;
    lastConversationId: string | null;
}

const rs: RetrievalState = {
    results: [],
    ignoredDocIds: new Set(),
    lastConversationId: null,
};

// ── DOM helpers ───────────────────────────────────────────────────────────────

function q<T extends HTMLElement>(id: string): T {
    const el = document.getElementById(id);
    if (!el) throw new Error(`Missing #${id}`);
    return el as T;
}

// ── Conversation context pill ─────────────────────────────────────────────────

function getActiveConvTitle(): string {
    const activeItem = document.querySelector<HTMLElement>(".conv-item.active");
    if (!activeItem) return "";
    return activeItem.textContent?.trim().replace(/^●/, "").trim() ?? "";
}

function renderConvPill(): void {
    const pill = q("retrievalConvPill");
    const noConvNote = q("retrievalNoConvNote");
    const convId = state.activeConversationId;

    if (convId) {
        const title = getActiveConvTitle() || convId;
        pill.style.display = "inline-flex";
        noConvNote.style.display = "none";
        // Safe text node assignment — no escaping needed for textContent
        const labelEl = pill.querySelector<HTMLElement>(".retrieval-conv-label");
        if (labelEl) labelEl.textContent = title;
    } else {
        pill.style.display = "none";
        noConvNote.style.display = "block";
    }
}

// ── Score badge ───────────────────────────────────────────────────────────────

function scoreClass(score: number): string {
    if (score >= 0.75) return "high";
    if (score >= 0.45) return "mid";
    return "low";
}

function scorePct(score: number): string {
    return `${Math.round(score * 100)}%`;
}

// ── Result cards ──────────────────────────────────────────────────────────────

function buildResultCard(item: RetrievalResultItem, idx: number): HTMLElement {
    const docId = String(
        item.metadata.doc_id || item.metadata.document_id || `result-${idx}`
    );
    const sourceName = String(item.metadata.source_name || item.metadata.source || "Unknown source");
    const sourceUri = String(item.metadata.source_uri || "");
    const excerpt = item.text ? item.text.slice(0, 200) : "";
    const cls = scoreClass(item.score);

    const card = document.createElement("div");
    card.className = "retrieval-card";
    card.dataset.docId = docId;

    // Score badge
    const badge = document.createElement("span");
    badge.className = `retrieval-score-badge ${cls}`;
    badge.textContent = scorePct(item.score);

    // Source name — link if URI available
    const nameEl = sourceUri
        ? document.createElement("a")
        : document.createElement("span");
    nameEl.className = "retrieval-source-name";
    nameEl.textContent = sourceName;
    if (nameEl instanceof HTMLAnchorElement && sourceUri) {
        nameEl.href = sourceUri;
        nameEl.target = "_blank";
        nameEl.rel = "noopener noreferrer";
    }

    // Header row
    const header = document.createElement("div");
    header.className = "retrieval-card-header";
    header.appendChild(nameEl);
    header.appendChild(badge);

    // Excerpt
    const excerptEl = document.createElement("p");
    excerptEl.className = "retrieval-card-excerpt";
    // Use textContent for user-supplied content — safe from XSS
    excerptEl.textContent = excerpt;

    // Hide button
    const hideBtn = document.createElement("button");
    hideBtn.className = "retrieval-hide-btn";
    hideBtn.title = "Hide from this conversation";
    hideBtn.textContent = "Hide";
    hideBtn.addEventListener("click", () => void hideDoc(docId, sourceName, card));

    const footer = document.createElement("div");
    footer.className = "retrieval-card-footer";
    footer.appendChild(excerptEl);
    footer.appendChild(hideBtn);

    card.appendChild(header);
    card.appendChild(footer);

    return card;
}

// ── Hidden docs section ───────────────────────────────────────────────────────

function buildHiddenItem(docId: string): HTMLElement {
    const item = document.createElement("div");
    item.className = "retrieval-hidden-item";
    item.dataset.docId = docId;

    const label = document.createElement("span");
    label.className = "retrieval-hidden-label";
    // docId may be arbitrary — escape for display
    label.textContent = docId;

    const restoreBtn = document.createElement("button");
    restoreBtn.className = "retrieval-restore-btn";
    restoreBtn.textContent = "Restore";
    restoreBtn.addEventListener("click", () => void restoreDoc(docId, item));

    item.appendChild(label);
    item.appendChild(restoreBtn);
    return item;
}

function renderHiddenSection(): void {
    const section = q("retrievalHiddenSection");
    const list = q("retrievalHiddenList");
    const countEl = q("retrievalHiddenCount");

    const count = rs.ignoredDocIds.size;
    countEl.textContent = String(count);

    list.innerHTML = "";
    rs.ignoredDocIds.forEach((docId) => {
        list.appendChild(buildHiddenItem(docId));
    });

    if (count > 0) {
        section.style.display = "block";
    } else {
        section.style.display = "none";
        // Collapse when emptied
        const details = section.querySelector("details");
        if (details) details.removeAttribute("open");
    }
}

// ── Result list ───────────────────────────────────────────────────────────────

function renderResults(): void {
    const list = q("retrievalResultsList");
    const emptyState = q("retrievalEmptyState");

    list.innerHTML = "";

    const visible = rs.results.filter((r) => {
        const docId = String(r.metadata.doc_id || r.metadata.document_id || "");
        return !docId || !rs.ignoredDocIds.has(docId);
    });

    if (visible.length === 0) {
        emptyState.style.display = "block";
        return;
    }

    emptyState.style.display = "none";
    visible.forEach((item, idx) => {
        list.appendChild(buildResultCard(item, idx));
    });
}

// ── Status line ───────────────────────────────────────────────────────────────

function setStatus(msg: string): void {
    q("retrievalStatus").textContent = msg;
}

// ── Doc-state API calls ───────────────────────────────────────────────────────

async function fetchDocState(conversationId: string): Promise<void> {
    try {
        const data = await api<DocStateResponse>(
            "GET",
            `/console/conversations/${encodeURIComponent(conversationId)}/doc-state`
        );
        rs.ignoredDocIds = new Set(data.ignored_doc_ids ?? []);
        renderHiddenSection();
        renderResults();
    } catch {
        // Non-fatal — hidden section just stays stale
    }
}

async function hideDoc(docId: string, sourceName: string, card: HTMLElement): Promise<void> {
    if (!state.activeConversationId) {
        showToast("Select a conversation first");
        return;
    }

    // Optimistic UI
    rs.ignoredDocIds.add(docId);
    card.remove();
    renderHiddenSection();

    try {
        await api<DocStateResponse>(
            "POST",
            `/console/conversations/${encodeURIComponent(state.activeConversationId)}/ignore`,
            { doc_id: docId }
        );
        showToast(`Hidden: ${sourceName}`);
    } catch (err) {
        // Revert
        rs.ignoredDocIds.delete(docId);
        renderResults();
        renderHiddenSection();
        showToast("Failed to hide document: " + String(err));
    }
}

async function restoreDoc(docId: string, item: HTMLElement): Promise<void> {
    if (!state.activeConversationId) {
        showToast("Select a conversation first");
        return;
    }

    // Optimistic UI
    rs.ignoredDocIds.delete(docId);
    item.remove();
    renderHiddenSection();
    renderResults();

    try {
        await api<DocStateResponse>(
            "DELETE",
            `/console/conversations/${encodeURIComponent(state.activeConversationId)}/ignore/${encodeURIComponent(docId)}`
        );
        showToast("Document restored");
    } catch (err) {
        // Revert
        rs.ignoredDocIds.add(docId);
        renderHiddenSection();
        renderResults();
        showToast("Failed to restore document: " + String(err));
    }
}

// ── Query submission ──────────────────────────────────────────────────────────

async function submitQuery(): Promise<void> {
    const textarea = q<HTMLTextAreaElement>("retrievalQueryInput");
    const findBtn = q<HTMLButtonElement>("retrievalFindBtn");
    const modeSmartBtn = q<HTMLButtonElement>("retrievalModeSmart");

    const query = textarea.value.trim();
    if (!query) {
        textarea.focus();
        return;
    }

    const mode = modeSmartBtn.classList.contains("active") ? "auto" : "hard";
    const topNInput = q<HTMLInputElement>("retrievalTopN");
    const searchLimit = Math.max(1, Math.min(50, parseInt(topNInput.value, 10) || 10));

    findBtn.disabled = true;
    setStatus("Searching…");

    try {
        const body: Record<string, unknown> = {
            query,
            mode: "retrieval",
            retrieval_sub_mode: mode,
            search_limit: searchLimit,
            rerank_top_k: Math.min(searchLimit, 10),
        };
        if (state.activeConversationId) {
            body.conversation_id = state.activeConversationId;
        }

        const data = await api<RetrievalResponse>("POST", "/query", body);

        rs.results = data.results ?? [];
        if (data.ignored_doc_ids) {
            rs.ignoredDocIds = new Set(data.ignored_doc_ids);
        }
        rs.lastConversationId = state.activeConversationId;

        const latency = data.latency_ms != null ? ` in ${Math.round(data.latency_ms)} ms` : "";
        setStatus(`Found ${rs.results.length} document${rs.results.length !== 1 ? "s" : ""}${latency}.`);

        renderResults();
        renderHiddenSection();
    } catch (err) {
        setStatus("Search failed.");
        showToast("Retrieval error: " + String(err));
    } finally {
        findBtn.disabled = false;
    }
}

// ── Auto-grow textarea ────────────────────────────────────────────────────────

function autoGrow(el: HTMLTextAreaElement): void {
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
}

// ── Init ──────────────────────────────────────────────────────────────────────

export function initRetrievalView(): void {
    const textarea = q<HTMLTextAreaElement>("retrievalQueryInput");
    const findBtn = q<HTMLButtonElement>("retrievalFindBtn");
    const modeSmartBtn = q<HTMLButtonElement>("retrievalModeSmart");
    const modeExactBtn = q<HTMLButtonElement>("retrievalModeExact");

    // Mode toggle
    modeSmartBtn.addEventListener("click", () => {
        modeSmartBtn.classList.add("active");
        modeExactBtn.classList.remove("active");
    });
    modeExactBtn.addEventListener("click", () => {
        modeExactBtn.classList.add("active");
        modeSmartBtn.classList.remove("active");
    });

    // Query textarea
    textarea.addEventListener("input", () => autoGrow(textarea));
    textarea.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            void submitQuery();
        }
    });

    // Find button
    findBtn.addEventListener("click", () => void submitQuery());

    // Hidden section collapsible
    // Rendered by renderHiddenSection() — nothing extra to wire here.

    // Conversation change — re-fetch doc state and clear stale results
    document.addEventListener("conversation-changed", () => {
        renderConvPill();
        rs.results = [];
        rs.ignoredDocIds = new Set();
        rs.lastConversationId = null;
        setStatus("");
        renderResults();
        renderHiddenSection();

        // If the retrieval pane is currently visible, pre-fetch doc state
        const pane = document.getElementById("view-retrieval");
        if (pane?.classList.contains("active") && state.activeConversationId) {
            void fetchDocState(state.activeConversationId);
        }
    });

    // View-tab switch — refresh doc state when switching TO retrieval
    document.querySelectorAll<HTMLElement>(".view-tab").forEach((btn) => {
        btn.addEventListener("click", () => {
            if (btn.dataset.view === "retrieval") {
                renderConvPill();
                if (state.activeConversationId && state.activeConversationId !== rs.lastConversationId) {
                    void fetchDocState(state.activeConversationId);
                }
            }
        });
    });

    // Initial render
    renderConvPill();
    renderResults();
    renderHiddenSection();
}
