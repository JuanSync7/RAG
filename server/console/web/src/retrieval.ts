// @summary
// Retrieval view-tab for the user console. Lets users query the knowledge
// base directly and manage per-conversation document visibility (hide/restore).
// Results are deduped by document — multiple matching chunks for one document
// collapse into a single card with an expandable chunk list.
// Exports: initRetrievalView
// Deps: api, toast, state, user-types
// @end-summary

import { api } from "./api";
import { showToast } from "./toast";
import { state } from "./state";
import { openSourceDocument } from "./citations";
import type { RetrievalResultItem, RetrievalResponse, DocStateResponse } from "./user-types";

// ── In-memory state ──────────────────────────────────────────────────────────

interface RetrievalState {
    results: RetrievalResultItem[];
    ignoredDocIds: Set<string>;
    relevantDocIds: Set<string>;
    docNames: Map<string, string>;  // docId → friendly source name (cached from queries)
    lastConversationId: string | null;
}

const rs: RetrievalState = {
    results: [],
    ignoredDocIds: new Set(),
    relevantDocIds: new Set(),
    docNames: new Map(),
    lastConversationId: null,
};

interface DocGroup {
    docId: string;
    sourceName: string;
    sourceUri: string;
    bestScore: number;
    chunks: RetrievalResultItem[];
}

// ── DOM helpers ───────────────────────────────────────────────────────────────

function q<T extends HTMLElement>(id: string): T {
    const el = document.getElementById(id);
    if (!el) throw new Error(`Missing #${id}`);
    return el as T;
}

function qOpt<T extends HTMLElement>(id: string): T | null {
    return document.getElementById(id) as T | null;
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

function numOrUndef(v: unknown): number | undefined {
    const n = Number(v);
    return Number.isFinite(n) ? n : undefined;
}

// ── Document grouping ─────────────────────────────────────────────────────────

interface DocIdInfo {
    id: string;
    isSynthetic: boolean;
}

/**
 * Resolve a stable per-document key for dedupe + hide/restore state.
 * Walks doc_id → document_id → source_key → source_uri → source name. Two chunks
 * from the same document share at least one of these, so they collapse into a
 * single group even when the explicit document_id is missing.
 */
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
        const { id: docId } = docIdOf(item, idx);
        const existing = groups.get(docId);
        if (existing) {
            existing.chunks.push(item);
            if (item.score > existing.bestScore) existing.bestScore = item.score;
        } else {
            groups.set(docId, {
                docId,
                sourceName: String(item.metadata.source_name || item.metadata.source || "Unknown source"),
                sourceUri: String(item.metadata.source_uri || ""),
                bestScore: item.score,
                chunks: [item],
            });
        }
    });
    return Array.from(groups.values()).sort((a, b) => b.bestScore - a.bestScore);
}

// ── Result cards ──────────────────────────────────────────────────────────────

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

function buildResultCard(group: DocGroup, isSynthetic: boolean): HTMLElement {
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

    // [view] link — opens the document just like chat citations do.
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
    if (isSynthetic) {
        hideBtn.disabled = true;
        hideBtn.title = "No document id — cannot hide";
    } else {
        hideBtn.title = "Hide from this conversation";
        hideBtn.addEventListener("click", () => void hideDoc(group.docId, group.sourceName, card));
    }
    hideBtn.textContent = "Hide";

    const footer = document.createElement("div");
    footer.className = "retrieval-card-footer";
    footer.appendChild(topExcerpt);
    footer.appendChild(hideBtn);

    card.appendChild(header);
    card.appendChild(footer);

    // Expandable chunks list (only when >1)
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

// ── Doc-state list items ──────────────────────────────────────────────────────

function labelForDocId(docId: string): string {
    return rs.docNames.get(docId) || docId;
}

function buildDocStateItem(
    docId: string,
    actionLabel: string,
    actionClass: string,
    onAction: (item: HTMLElement) => void,
): HTMLElement {
    const item = document.createElement("div");
    item.className = "retrieval-hidden-item";
    item.dataset.docId = docId;

    const label = document.createElement("span");
    label.className = "retrieval-hidden-label";
    label.textContent = labelForDocId(docId);
    label.title = docId;

    const btn = document.createElement("button");
    btn.className = actionClass;
    btn.textContent = actionLabel;
    btn.addEventListener("click", () => onAction(item));

    item.appendChild(label);
    item.appendChild(btn);
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
        list.appendChild(
            buildDocStateItem(docId, "Restore", "retrieval-restore-btn", (item) =>
                void restoreDoc(docId, item),
            ),
        );
    });

    if (count > 0) {
        section.style.display = "block";
    } else {
        section.style.display = "none";
        const details = section.querySelector("details");
        if (details) details.removeAttribute("open");
    }
}

function renderRelevantSection(): void {
    const section = qOpt("retrievalRelevantSection");
    if (!section) return; // HTML may not yet have the relevant accordion
    const list = q("retrievalRelevantList");
    const countEl = q("retrievalRelevantCount");

    const count = rs.relevantDocIds.size;
    countEl.textContent = String(count);
    list.innerHTML = "";
    rs.relevantDocIds.forEach((docId) => {
        list.appendChild(
            buildDocStateItem(docId, "Hide", "retrieval-hide-btn", (item) =>
                void hideDocFromList(docId, item),
            ),
        );
    });

    if (count > 0) {
        section.style.display = "block";
    } else {
        section.style.display = "none";
        const details = section.querySelector("details");
        if (details) details.removeAttribute("open");
    }
}

// ── Result list ───────────────────────────────────────────────────────────────

function renderResults(): void {
    const list = q("retrievalResultsList");
    const emptyState = q("retrievalEmptyState");

    list.innerHTML = "";

    const visibleItems = rs.results.filter((r, idx) => {
        const { id } = docIdOf(r, idx);
        return !rs.ignoredDocIds.has(id);
    });

    const groups = groupByDoc(visibleItems);

    // Cache id → friendly name so the Hidden/Relevant lists can show source
    // names instead of raw doc-ids.
    groups.forEach((g) => {
        if (g.sourceName && g.sourceName !== "Unknown source") {
            rs.docNames.set(g.docId, g.sourceName);
        }
    });

    if (groups.length === 0) {
        emptyState.style.display = "block";
        return;
    }

    emptyState.style.display = "none";
    groups.forEach((g) => {
        const isSynthetic = g.docId.startsWith("result-");
        list.appendChild(buildResultCard(g, isSynthetic));
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
            `/console/conversations/${encodeURIComponent(conversationId)}/doc-state`,
        );
        rs.ignoredDocIds = new Set(data.ignored_doc_ids ?? []);
        rs.relevantDocIds = new Set(data.relevant_doc_ids ?? []);
        renderHiddenSection();
        renderRelevantSection();
        renderResults();
    } catch {
        // Non-fatal — sidebar lists just stay stale
    }
}

async function hideDoc(docId: string, sourceName: string, card: HTMLElement): Promise<void> {
    if (!state.activeConversationId) {
        showToast("Select a conversation first");
        return;
    }

    rs.ignoredDocIds.add(docId);
    rs.relevantDocIds.delete(docId);
    card.remove();
    renderHiddenSection();
    renderRelevantSection();

    try {
        await api<DocStateResponse>(
            "POST",
            `/console/conversations/${encodeURIComponent(state.activeConversationId)}/ignore`,
            { doc_id: docId },
        );
        showToast(`Hidden: ${sourceName}`);
    } catch (err) {
        rs.ignoredDocIds.delete(docId);
        renderResults();
        renderHiddenSection();
        renderRelevantSection();
        showToast("Failed to hide document: " + String(err));
    }
}

async function hideDocFromList(docId: string, item: HTMLElement): Promise<void> {
    if (!state.activeConversationId) {
        showToast("Select a conversation first");
        return;
    }
    rs.ignoredDocIds.add(docId);
    rs.relevantDocIds.delete(docId);
    item.remove();
    renderHiddenSection();
    renderRelevantSection();
    renderResults();

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
        renderHiddenSection();
        renderRelevantSection();
        renderResults();
        showToast("Failed to hide document: " + String(err));
    }
}

async function restoreDoc(docId: string, item: HTMLElement): Promise<void> {
    if (!state.activeConversationId) {
        showToast("Select a conversation first");
        return;
    }

    rs.ignoredDocIds.delete(docId);
    rs.relevantDocIds.add(docId);
    item.remove();
    renderHiddenSection();
    renderRelevantSection();
    renderResults();

    try {
        await api<DocStateResponse>(
            "DELETE",
            `/console/conversations/${encodeURIComponent(state.activeConversationId)}/ignore/${encodeURIComponent(docId)}`,
        );
        showToast("Document restored");
    } catch (err) {
        rs.ignoredDocIds.add(docId);
        rs.relevantDocIds.delete(docId);
        renderHiddenSection();
        renderRelevantSection();
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

        const data = await api<RetrievalResponse>("POST", "/console/query", body);

        rs.results = data.results ?? [];
        if (data.ignored_doc_ids) rs.ignoredDocIds = new Set(data.ignored_doc_ids);
        if (data.relevant_doc_ids) rs.relevantDocIds = new Set(data.relevant_doc_ids);
        rs.lastConversationId = state.activeConversationId;

        const groups = groupByDoc(rs.results);
        groups.forEach((g) => {
            if (g.sourceName && g.sourceName !== "Unknown source") {
                rs.docNames.set(g.docId, g.sourceName);
            }
        });
        const docCount = groups.length;
        const latency = data.latency_ms != null ? ` in ${Math.round(data.latency_ms)} ms` : "";
        setStatus(
            `Found ${docCount} document${docCount !== 1 ? "s" : ""} (${rs.results.length} chunk${rs.results.length !== 1 ? "s" : ""})${latency}.`,
        );

        renderResults();
        renderHiddenSection();
        renderRelevantSection();
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

    // Conversation change — refresh per-conversation doc-state lists, but
    // KEEP the last query's results so they survive sidebar clicks and tab
    // switches. The user can re-submit to refresh against the new conversation.
    document.addEventListener("conversation-changed", () => {
        renderConvPill();
        if (state.activeConversationId) {
            void fetchDocState(state.activeConversationId);
        } else {
            rs.ignoredDocIds = new Set();
            rs.relevantDocIds = new Set();
            renderHiddenSection();
            renderRelevantSection();
            renderResults();
        }
    });

    // View-tab switch — refresh doc state when switching TO retrieval if the
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
    renderResults();
    renderHiddenSection();
    renderRelevantSection();
}
