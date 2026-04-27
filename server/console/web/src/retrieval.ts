// @summary
// Retrieval view-tab for the user console. Chat-like UX: each query becomes
// a turn in a bubble thread, the assistant reply is "Here are your documents"
// followed by deduplicated document cards. Right-side rail shows the
// conversation's relevant + hidden document lists.
// Exports: initRetrievalView
// Deps: api, toast, state, citations, user-types
// @end-summary

import { api } from "./api";
import { showToast } from "./toast";
import { state } from "./state";
import { openSourceDocument } from "./citations";
import { createNewConversation } from "./conversations";
import type { RetrievalResultItem, RetrievalResponse, DocStateResponse } from "./user-types";

// ── State ─────────────────────────────────────────────────────────────────────

interface DocGroup {
    docId: string;
    sourceName: string;
    sourceUri: string;
    bestScore: number;
    chunks: RetrievalResultItem[];
    isSynthetic: boolean;
}

interface RetrievalTurn {
    id: string;
    query: string;
    status: "pending" | "done" | "error";
    groups: DocGroup[];
    rawCount: number;
    latencyMs: number | null;
    errorMsg?: string;
}

interface RetrievalState {
    turns: RetrievalTurn[];
    ignoredDocIds: Set<string>;
    relevantDocIds: Set<string>;
    docNames: Map<string, string>;
    lastConversationId: string | null;
}

const rs: RetrievalState = {
    turns: [],
    ignoredDocIds: new Set(),
    relevantDocIds: new Set(),
    docNames: new Map(),
    lastConversationId: null,
};

// ── DOM helpers ───────────────────────────────────────────────────────────────

function q<T extends HTMLElement>(id: string): T {
    const el = document.getElementById(id);
    if (!el) throw new Error(`Missing #${id}`);
    return el as T;
}

function qOpt<T extends HTMLElement>(id: string): T | null {
    return document.getElementById(id) as T | null;
}

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
        const labelEl = pill.querySelector<HTMLElement>(".retrieval-conv-label");
        if (labelEl) labelEl.textContent = title;
    } else {
        pill.style.display = "none";
        noConvNote.style.display = "inline-flex";
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

function numOrUndef(v: unknown): number | undefined {
    const n = Number(v);
    return Number.isFinite(n) ? n : undefined;
}

// ── Document grouping ─────────────────────────────────────────────────────────

interface DocIdInfo {
    id: string;
    isSynthetic: boolean;
}

function docIdOf(item: RetrievalResultItem, fallbackIdx: number): DocIdInfo {
    const m = item.metadata;
    const candidates = [m.doc_id, m.document_id, m.source_key, m.source_uri, m.source_name, m.source];
    for (const c of candidates) {
        const s = c == null ? "" : String(c).trim();
        if (s) return { id: s, isSynthetic: false };
    }
    return { id: `result-${fallbackIdx}`, isSynthetic: true };
}

function groupByDoc(items: RetrievalResultItem[]): DocGroup[] {
    const groups = new Map<string, DocGroup>();
    items.forEach((item, idx) => {
        const { id: docId, isSynthetic } = docIdOf(item, idx);
        const existing = groups.get(docId);
        if (existing) {
            existing.chunks.push(item);
            if (item.score > existing.bestScore) existing.bestScore = item.score;
        } else {
            const sourceName = String(item.metadata.source_name || item.metadata.source || "Unknown source");
            groups.set(docId, {
                docId,
                sourceName,
                sourceUri: String(item.metadata.source_uri || ""),
                bestScore: item.score,
                chunks: [item],
                isSynthetic,
            });
        }
    });
    const arr = Array.from(groups.values()).sort((a, b) => b.bestScore - a.bestScore);
    arr.forEach((g) => {
        if (g.sourceName && g.sourceName !== "Unknown source") rs.docNames.set(g.docId, g.sourceName);
    });
    return arr;
}

// ── Card rendering ────────────────────────────────────────────────────────────

function buildChunkExcerpt(item: RetrievalResultItem): HTMLElement {
    const wrap = document.createElement("div");
    wrap.className = "retrieval-chunk";
    const score = document.createElement("span");
    score.className = `retrieval-chunk-score ${scoreClass(item.score)}`;
    score.textContent = scorePct(item.score);
    const text = document.createElement("p");
    text.className = "retrieval-chunk-text";
    text.textContent = item.text ? item.text.slice(0, 400) : "";
    wrap.appendChild(score);
    wrap.appendChild(text);
    return wrap;
}

function buildResultCard(group: DocGroup): HTMLElement {
    const card = document.createElement("div");
    card.className = "retrieval-card";
    card.dataset.docId = group.docId;

    const badge = document.createElement("span");
    badge.className = `retrieval-score-badge ${scoreClass(group.bestScore)}`;
    badge.textContent = scorePct(group.bestScore);

    const chunkBadge = document.createElement("span");
    chunkBadge.className = "retrieval-chunk-count";
    const n = group.chunks.length;
    chunkBadge.textContent = `${n} chunk${n === 1 ? "" : "s"}`;

    const nameEl = document.createElement("span");
    nameEl.className = "retrieval-source-name";
    nameEl.textContent = group.sourceName;

    const viewLink = document.createElement("a");
    viewLink.className = "retrieval-view-link";
    viewLink.href = "#";
    viewLink.textContent = "[view]";
    viewLink.title = "Open document";
    viewLink.addEventListener("click", (e) => {
        e.preventDefault();
        const top = group.chunks[0];
        const m = top?.metadata ?? {};
        void openSourceDocument({
            source: String(m.source ?? group.sourceName ?? "") || undefined,
            source_uri: String(m.source_uri ?? group.sourceUri ?? "") || undefined,
            source_key: m.source_key != null ? String(m.source_key) : undefined,
            chunk_text: top?.text || undefined,
            original_start: numOrUndef(m.original_char_start),
            original_end: numOrUndef(m.original_char_end),
            refactored_start: numOrUndef(m.refactored_char_start),
            refactored_end: numOrUndef(m.refactored_char_end),
            provenance_confidence: numOrUndef(m.provenance_confidence),
        });
    });

    const header = document.createElement("div");
    header.className = "retrieval-card-header";
    header.appendChild(nameEl);
    header.appendChild(viewLink);
    header.appendChild(chunkBadge);
    header.appendChild(badge);

    const topExcerpt = document.createElement("p");
    topExcerpt.className = "retrieval-card-excerpt";
    topExcerpt.textContent = group.chunks[0]?.text ? group.chunks[0].text.slice(0, 240) : "";

    const hideBtn = document.createElement("button");
    hideBtn.className = "retrieval-hide-btn";
    if (group.isSynthetic) {
        hideBtn.disabled = true;
        hideBtn.title = "No document id — cannot hide";
    } else {
        hideBtn.title = "Hide from this conversation";
        hideBtn.addEventListener("click", () => void hideDoc(group.docId, group.sourceName));
    }
    hideBtn.textContent = "Hide";

    const footer = document.createElement("div");
    footer.className = "retrieval-card-footer";
    footer.appendChild(topExcerpt);
    footer.appendChild(hideBtn);

    card.appendChild(header);
    card.appendChild(footer);

    if (group.chunks.length > 1) {
        const details = document.createElement("details");
        details.className = "retrieval-chunks-details";
        const summary = document.createElement("summary");
        summary.textContent = `Show all ${group.chunks.length} chunks`;
        details.appendChild(summary);
        group.chunks.forEach((c) => details.appendChild(buildChunkExcerpt(c)));
        card.appendChild(details);
    }

    return card;
}

// ── Bubble thread ─────────────────────────────────────────────────────────────

function buildUserBubble(text: string): HTMLElement {
    const row = document.createElement("div");
    row.className = "msg-row user";
    const wrap = document.createElement("div");
    wrap.className = "bubble-wrap";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    wrap.appendChild(bubble);
    row.appendChild(wrap);
    return row;
}

function buildAssistantBubble(turn: RetrievalTurn): HTMLElement {
    const row = document.createElement("div");
    row.className = "msg-row assistant";
    const wrap = document.createElement("div");
    wrap.className = "bubble-wrap";
    const bubble = document.createElement("div");
    bubble.className = "bubble";

    if (turn.status === "pending") {
        bubble.classList.add("streaming");
        bubble.textContent = "Searching…";
        wrap.appendChild(bubble);
        row.appendChild(wrap);
        return row;
    }

    if (turn.status === "error") {
        bubble.classList.add("error-bubble");
        bubble.textContent = `Search failed: ${turn.errorMsg ?? "unknown error"}`;
        wrap.appendChild(bubble);
        row.appendChild(wrap);
        return row;
    }

    const visibleGroups = turn.groups.filter((g) => !rs.ignoredDocIds.has(g.docId));
    const docCount = visibleGroups.length;
    const chunkCount = visibleGroups.reduce((acc, g) => acc + g.chunks.length, 0);
    const latency = turn.latencyMs != null ? ` in ${Math.round(turn.latencyMs)} ms` : "";

    const intro = document.createElement("p");
    intro.className = "retrieval-bubble-intro";
    if (docCount === 0) {
        intro.textContent = "No matching documents — try a different query.";
        bubble.appendChild(intro);
    } else {
        intro.textContent = `Here are your documents — ${docCount} doc${docCount !== 1 ? "s" : ""} (${chunkCount} chunk${chunkCount !== 1 ? "s" : ""})${latency}.`;
        bubble.appendChild(intro);
        const cards = document.createElement("div");
        cards.className = "retrieval-bubble-cards";
        visibleGroups.forEach((g) => cards.appendChild(buildResultCard(g)));
        bubble.appendChild(cards);
    }

    wrap.appendChild(bubble);
    row.appendChild(wrap);
    return row;
}

function renderThread(): void {
    const thread = q("retrievalThread");
    const empty = qOpt("retrievalThreadEmpty");
    thread.innerHTML = "";

    if (rs.turns.length === 0) {
        if (empty) thread.appendChild(empty);
        return;
    }

    rs.turns.forEach((turn) => {
        thread.appendChild(buildUserBubble(turn.query));
        thread.appendChild(buildAssistantBubble(turn));
    });

    // Auto-scroll to bottom.
    thread.scrollTop = thread.scrollHeight;
}

// ── Right rail ────────────────────────────────────────────────────────────────

function labelForDocId(docId: string): string {
    return rs.docNames.get(docId) || docId;
}

function buildRailItem(
    docId: string,
    actionLabel: string,
    actionClass: string,
    onAction: () => void,
): HTMLElement {
    const item = document.createElement("div");
    item.className = "retrieval-rail-item";
    item.dataset.docId = docId;

    const label = document.createElement("span");
    label.className = "retrieval-rail-label";
    label.textContent = labelForDocId(docId);
    label.title = docId;

    const btn = document.createElement("button");
    btn.className = actionClass;
    btn.textContent = actionLabel;
    btn.addEventListener("click", onAction);

    item.appendChild(label);
    item.appendChild(btn);
    return item;
}

function renderRail(): void {
    const relevantSection = qOpt("retrievalRelevantSection");
    const hiddenSection = qOpt("retrievalHiddenSection");
    const railEmpty = qOpt("retrievalRailEmpty");

    if (relevantSection) {
        const list = q("retrievalRelevantList");
        const countEl = q("retrievalRelevantCount");
        countEl.textContent = String(rs.relevantDocIds.size);
        list.innerHTML = "";
        rs.relevantDocIds.forEach((docId) => {
            list.appendChild(
                buildRailItem(docId, "Hide", "retrieval-hide-btn", () => void hideDocFromRail(docId)),
            );
        });
        relevantSection.style.display = rs.relevantDocIds.size > 0 ? "block" : "none";
    }

    if (hiddenSection) {
        const list = q("retrievalHiddenList");
        const countEl = q("retrievalHiddenCount");
        countEl.textContent = String(rs.ignoredDocIds.size);
        list.innerHTML = "";
        rs.ignoredDocIds.forEach((docId) => {
            list.appendChild(
                buildRailItem(docId, "Restore", "retrieval-restore-btn", () => void restoreDoc(docId)),
            );
        });
        hiddenSection.style.display = rs.ignoredDocIds.size > 0 ? "block" : "none";
    }

    if (railEmpty) {
        const isEmpty = rs.relevantDocIds.size === 0 && rs.ignoredDocIds.size === 0;
        railEmpty.style.display = isEmpty ? "block" : "none";
    }
}

// ── Doc-state API calls ───────────────────────────────────────────────────────

async function fetchDocState(conversationId: string): Promise<void> {
    try {
        const data = await api<DocStateResponse>(
            "GET",
            `/console/conversations/${encodeURIComponent(conversationId)}/doc-state`,
        );
        rs.ignoredDocIds = new Set(data.ignored_doc_ids ?? []);
        rs.relevantDocIds = new Set(data.relevant_doc_ids ?? []);
        renderRail();
        renderThread();
    } catch {
        // Non-fatal — rail just stays stale
    }
}

async function hideDoc(docId: string, sourceName: string): Promise<void> {
    if (!state.activeConversationId) {
        showToast("Select a conversation first");
        return;
    }
    rs.ignoredDocIds.add(docId);
    rs.relevantDocIds.delete(docId);
    renderRail();
    renderThread();
    try {
        await api<DocStateResponse>(
            "POST",
            `/console/conversations/${encodeURIComponent(state.activeConversationId)}/ignore`,
            { doc_id: docId },
        );
        showToast(`Hidden: ${sourceName}`);
    } catch (err) {
        rs.ignoredDocIds.delete(docId);
        renderRail();
        renderThread();
        showToast("Failed to hide document: " + String(err));
    }
}

async function hideDocFromRail(docId: string): Promise<void> {
    if (!state.activeConversationId) {
        showToast("Select a conversation first");
        return;
    }
    rs.ignoredDocIds.add(docId);
    rs.relevantDocIds.delete(docId);
    renderRail();
    renderThread();
    try {
        await api<DocStateResponse>(
            "POST",
            `/console/conversations/${encodeURIComponent(state.activeConversationId)}/ignore`,
            { doc_id: docId },
        );
        showToast("Document hidden");
    } catch (err) {
        rs.ignoredDocIds.delete(docId);
        rs.relevantDocIds.add(docId);
        renderRail();
        renderThread();
        showToast("Failed to hide document: " + String(err));
    }
}

async function restoreDoc(docId: string): Promise<void> {
    if (!state.activeConversationId) {
        showToast("Select a conversation first");
        return;
    }
    rs.ignoredDocIds.delete(docId);
    rs.relevantDocIds.add(docId);
    renderRail();
    renderThread();
    try {
        await api<DocStateResponse>(
            "DELETE",
            `/console/conversations/${encodeURIComponent(state.activeConversationId)}/ignore/${encodeURIComponent(docId)}`,
        );
        showToast("Document restored");
    } catch (err) {
        rs.ignoredDocIds.add(docId);
        rs.relevantDocIds.delete(docId);
        renderRail();
        renderThread();
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

    const turn: RetrievalTurn = {
        id: `t-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        query,
        status: "pending",
        groups: [],
        rawCount: 0,
        latencyMs: null,
    };
    rs.turns.push(turn);

    findBtn.disabled = true;
    textarea.value = "";
    textarea.style.height = "auto";
    renderThread();

    try {
        const body: Record<string, unknown> = {
            query,
            mode: "retrieval",
            retrieval_sub_mode: mode,
            search_limit: searchLimit,
            rerank_top_k: Math.min(searchLimit, 10),
        };
        if (state.activeConversationId) body.conversation_id = state.activeConversationId;

        const data = await api<RetrievalResponse>("POST", "/console/query", body);

        const items = data.results ?? [];
        turn.groups = groupByDoc(items);
        turn.rawCount = items.length;
        turn.latencyMs = data.latency_ms ?? null;
        turn.status = "done";

        if (data.ignored_doc_ids) rs.ignoredDocIds = new Set(data.ignored_doc_ids);
        if (data.relevant_doc_ids) rs.relevantDocIds = new Set(data.relevant_doc_ids);
        rs.lastConversationId = state.activeConversationId;

        renderThread();
        renderRail();
    } catch (err) {
        turn.status = "error";
        turn.errorMsg = String(err);
        renderThread();
        showToast("Retrieval error: " + String(err));
    } finally {
        findBtn.disabled = false;
    }
}

// ── Auto-grow textarea ────────────────────────────────────────────────────────

function autoGrow(el: HTMLTextAreaElement): void {
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
}

// ── Init ──────────────────────────────────────────────────────────────────────

export function initRetrievalView(): void {
    const textarea = q<HTMLTextAreaElement>("retrievalQueryInput");
    const findBtn = q<HTMLButtonElement>("retrievalFindBtn");
    const modeSmartBtn = q<HTMLButtonElement>("retrievalModeSmart");
    const modeExactBtn = q<HTMLButtonElement>("retrievalModeExact");
    const toggleBtn = qOpt<HTMLButtonElement>("retrievalToggleBtn");

    modeSmartBtn.addEventListener("click", () => {
        modeSmartBtn.classList.add("active");
        modeExactBtn.classList.remove("active");
    });
    modeExactBtn.addEventListener("click", () => {
        modeExactBtn.classList.add("active");
        modeSmartBtn.classList.remove("active");
    });

    textarea.addEventListener("input", () => autoGrow(textarea));
    textarea.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            void submitQuery();
        }
    });

    findBtn.addEventListener("click", () => void submitQuery());

    // Toggle the global sidebar (shared with chat) so users can create a new
    // conversation from the retrieval pane just like from chat.
    if (toggleBtn) {
        toggleBtn.addEventListener("click", () => {
            document.body.classList.toggle("sidebar-collapsed");
        });
    }

    const newConvBtn = qOpt<HTMLButtonElement>("retrievalNewConvBtn");
    if (newConvBtn) {
        newConvBtn.addEventListener("click", () => {
            createNewConversation();
            // conversation-changed fires from setActiveConversation(null);
            // its listener clears the retrieval thread and rail, so nothing
            // extra is needed here. Focus the query box for the next search.
            qOpt<HTMLTextAreaElement>("retrievalQueryInput")?.focus();
        });
    }

    // Conversation change — clear the thread (turns are conversation-scoped),
    // refresh per-conversation doc-state lists.
    document.addEventListener("conversation-changed", () => {
        renderConvPill();
        rs.turns = [];
        if (state.activeConversationId) {
            void fetchDocState(state.activeConversationId);
        } else {
            rs.ignoredDocIds = new Set();
            rs.relevantDocIds = new Set();
            renderRail();
        }
        renderThread();
    });

    // View-tab switch — refresh doc-state when switching TO retrieval if the
    // conversation has changed since the last fetch.
    document.querySelectorAll<HTMLElement>(".view-tab").forEach((btn) => {
        btn.addEventListener("click", () => {
            if (btn.dataset.view === "retrieval") {
                renderConvPill();
                if (state.activeConversationId && state.activeConversationId !== rs.lastConversationId) {
                    void fetchDocState(state.activeConversationId);
                    rs.lastConversationId = state.activeConversationId;
                }
            }
        });
    });

    renderConvPill();
    renderThread();
    renderRail();
}
