// @summary
// Chat-pane renderer toggle: "answer" (LLM bubble + citation strip) vs
// "sources" (doc-cards only, no generation). Mode is persisted globally in
// localStorage and forwarded to /console/query as `mode: "retrieval"` when
// in sources mode. Backend writes both modes to the same conversation history,
// so a single thread can be replayed under either renderer. Sources mode also
// renders a right rail (Relevant / Hidden) backed by the conversation's
// doc-state, mirroring the standalone Retrieval tab's mark-relevant/hide flow.
// @end-summary

import { byId, escHtml, fmtTime } from "./dom";
import { pct } from "./format";
import { parseMarkdown } from "./markdown";
import { api } from "./api";
import { openSourceDocument } from "./citations";
import { showToast } from "./toast";
import { state } from "./state";
import type { SourceRef, DocStateResponse } from "./user-types";

export type ChatRenderMode = "answer" | "sources";
export type RetrievalSubMode = "hard" | "auto";

const STORAGE_KEY = "rw_chat_mode";
const SUBMODE_STORAGE_KEY = "rw_retrieval_submode";
const DEEP_RESEARCH_STORAGE_KEY = "rw_deep_research";
const RAIL_COLLAPSED_KEY = "rw_chat_rail_collapsed";
const TOPK_STORAGE_KEY = "rw_sources_top_k";
const DEFAULT_TOPK = 5;

let _topK: number = (() => {
    const raw = parseInt(localStorage.getItem(TOPK_STORAGE_KEY) || "", 10);
    return Number.isFinite(raw) && raw > 0 ? Math.min(50, raw) : DEFAULT_TOPK;
})();

export function getSourcesTopK(): number {
    return _topK;
}

function initTopKInput(): void {
    const input = document.getElementById("chatTopkInput") as HTMLInputElement | null;
    if (!input) return;
    input.value = String(_topK);
    input.addEventListener("change", () => {
        const v = parseInt(input.value, 10);
        if (!Number.isFinite(v) || v < 1) {
            input.value = String(_topK);
            return;
        }
        _topK = Math.min(50, Math.max(1, v));
        input.value = String(_topK);
        localStorage.setItem(TOPK_STORAGE_KEY, String(_topK));
    });
}

let _mode: ChatRenderMode =
    (localStorage.getItem(STORAGE_KEY) as ChatRenderMode | null) === "sources" ? "sources" : "answer";
let _subMode: RetrievalSubMode =
    (localStorage.getItem(SUBMODE_STORAGE_KEY) as RetrievalSubMode | null) === "auto" ? "auto" : "hard";

export function getRetrievalSubMode(): RetrievalSubMode {
    return _subMode;
}

export function setRetrievalSubMode(sub: RetrievalSubMode): void {
    _subMode = sub;
    localStorage.setItem(SUBMODE_STORAGE_KEY, sub);
    syncSubmodeUI();
}

let _deepResearch: boolean = localStorage.getItem(DEEP_RESEARCH_STORAGE_KEY) === "1";

export function getDeepResearch(): boolean {
    return _deepResearch;
}

export function setDeepResearch(enabled: boolean): void {
    _deepResearch = enabled;
    localStorage.setItem(DEEP_RESEARCH_STORAGE_KEY, enabled ? "1" : "0");
    syncDeepResearchUI();
}

function syncDeepResearchUI(): void {
    const btn = document.getElementById("chatDeepResearch");
    if (btn) {
        btn.setAttribute("aria-pressed", _deepResearch ? "true" : "false");
        btn.classList.toggle("active", _deepResearch);
    }
}

let _lastSuggestedQuery: string = "";
type ResubmitFn = (text: string) => void | Promise<void>;
let _resubmit: ResubmitFn | null = null;

export function registerDrSuggestionResubmit(fn: ResubmitFn): void {
    _resubmit = fn;
}

export function showDrSuggestionChip(forQuery: string): void {
    if (_deepResearch) return;
    const chip = document.getElementById("drSuggestChip");
    if (!chip) return;
    _lastSuggestedQuery = forQuery;
    chip.removeAttribute("hidden");
}

export function hideDrSuggestionChip(): void {
    const chip = document.getElementById("drSuggestChip");
    if (!chip) return;
    chip.setAttribute("hidden", "");
}

function initDrSuggestionChip(): void {
    const chip = document.getElementById("drSuggestChip");
    if (!chip) return;
    chip.addEventListener("click", () => {
        const q = _lastSuggestedQuery.trim();
        hideDrSuggestionChip();
        setDeepResearch(true);
        if (q && _resubmit) void _resubmit(q);
    });
}

function syncSubmodeUI(): void {
    const hardBtn = document.getElementById("chatSubmodeHard");
    const autoBtn = document.getElementById("chatSubmodeAuto");
    if (hardBtn) {
        hardBtn.classList.toggle("active", _subMode === "hard");
        hardBtn.setAttribute("aria-selected", _subMode === "hard" ? "true" : "false");
    }
    if (autoBtn) {
        autoBtn.classList.toggle("active", _subMode === "auto");
        autoBtn.setAttribute("aria-selected", _subMode === "auto" ? "true" : "false");
    }
}

interface DocCacheEntry {
    name: string;
    sourceUri: string;
    sourceKey: string;
    source: string;
}
const _docCache = new Map<string, DocCacheEntry>();
const _ignoredDocIds = new Set<string>();
const _relevantDocIds = new Set<string>();

export function getChatMode(): ChatRenderMode {
    return _mode;
}

export function setChatMode(mode: ChatRenderMode): void {
    _mode = mode;
    localStorage.setItem(STORAGE_KEY, mode);
    syncToggleUI();
    applyModeToView();
    if (mode === "sources" && state.activeConversationId) {
        void fetchAndRenderDocState(state.activeConversationId);
    }
    document.dispatchEvent(new CustomEvent("chat-mode-changed", { detail: { mode } }));
}

function syncToggleUI(): void {
    const answerBtn = document.getElementById("chatModeAnswer");
    const sourcesBtn = document.getElementById("chatModeSources");
    if (answerBtn) {
        answerBtn.classList.toggle("active", _mode === "answer");
        answerBtn.setAttribute("aria-selected", _mode === "answer" ? "true" : "false");
    }
    if (sourcesBtn) {
        sourcesBtn.classList.toggle("active", _mode === "sources");
        sourcesBtn.setAttribute("aria-selected", _mode === "sources" ? "true" : "false");
    }
}

function applyModeToView(): void {
    const view = document.getElementById("view-chat");
    if (view) view.dataset.chatMode = _mode;
}

export function initChatMode(): void {
    const toggle = document.getElementById("chatModeToggle");
    if (toggle) {
        toggle.querySelectorAll<HTMLButtonElement>("[data-mode]").forEach((btn) => {
            btn.addEventListener("click", () => {
                const mode = btn.dataset.mode as ChatRenderMode | undefined;
                if (mode === "answer" || mode === "sources") setChatMode(mode);
            });
        });
    }
    const subToggle = document.getElementById("chatSubmodeToggle");
    if (subToggle) {
        subToggle.querySelectorAll<HTMLButtonElement>("[data-submode]").forEach((btn) => {
            btn.addEventListener("click", () => {
                const sub = btn.dataset.submode as RetrievalSubMode | undefined;
                if (sub === "hard" || sub === "auto") setRetrievalSubMode(sub);
            });
        });
    }
    const drBtn = document.getElementById("chatDeepResearch");
    if (drBtn) {
        drBtn.addEventListener("click", () => setDeepResearch(!_deepResearch));
    }
    initDrSuggestionChip();
    initRailCollapse();
    initTopKInput();
    syncToggleUI();
    syncSubmodeUI();
    syncDeepResearchUI();
    applyModeToView();
    renderRail();

    document.addEventListener("conversation-changed", () => {
        _ignoredDocIds.clear();
        _relevantDocIds.clear();
        renderRail();
        if (_mode === "sources" && state.activeConversationId) {
            void fetchAndRenderDocState(state.activeConversationId);
        }
    });
}

interface DocGroup {
    docKey: string;
    name: string;
    bestScore: number;
    excerpt: string;
    refs: SourceRef[];
    sourceUri: string;
    sourceKey: string;
    source: string;
}

function docKeyOf(ref: SourceRef): string {
    return (
        ref.document_id ||
        ref.source_key ||
        ref.source_uri ||
        ref.source ||
        ""
    ).trim();
}

/** Compute the docKey used by doc-state from a citation's metadata blob. */
export function docKeyFromMeta(meta: Record<string, unknown> | undefined | null): string {
    if (!meta) return "";
    const pick = (k: string) => {
        const v = meta[k];
        return typeof v === "string" ? v : v == null ? "" : String(v);
    };
    return (
        pick("document_id") ||
        pick("source_key") ||
        pick("source_uri") ||
        pick("source") ||
        ""
    ).trim();
}

function nameOf(ref: SourceRef): string {
    const raw =
        ref.source ||
        ref.source_uri ||
        ref.source_key ||
        ref.document_id ||
        "Unknown source";
    const stripped = String(raw).split("/").pop() || String(raw);
    return stripped;
}

export function groupSources(sources: SourceRef[]): DocGroup[] {
    const groups = new Map<string, DocGroup>();
    let synthCounter = 0;
    for (const ref of sources) {
        let key = docKeyOf(ref);
        if (!key) key = `__synth_${synthCounter++}`;
        const score = ref.score ?? 0;
        const text = ref.text ?? "";
        const existing = groups.get(key);
        if (existing) {
            existing.refs.push(ref);
            if (score > existing.bestScore) {
                existing.bestScore = score;
                existing.excerpt = text;
            }
        } else {
            groups.set(key, {
                docKey: key,
                name: nameOf(ref),
                bestScore: score,
                excerpt: text,
                refs: [ref],
                sourceUri: ref.source_uri ?? "",
                sourceKey: ref.source_key ?? "",
                source: ref.source ?? "",
            });
        }
    }
    return [...groups.values()].sort((a, b) => b.bestScore - a.bestScore);
}

function cacheDocs(groups: DocGroup[]): void {
    for (const g of groups) {
        if (!g.docKey || g.docKey.startsWith("__synth_")) continue;
        _docCache.set(g.docKey, {
            name: g.name,
            sourceUri: g.sourceUri,
            sourceKey: g.sourceKey,
            source: g.source,
        });
    }
}

/** Public helper for non-sources callers (chat-mode streaming) to seed the
 *  doc cache so the rail can render names/links for relevant/ignored ids. */
export function cacheDocsFromSources(sources: SourceRef[]): void {
    cacheDocs(groupSources(sources));
}

/** Append a "sources turn" (doc cards, no LLM bubble) to the message thread. */
export function appendSourcesTurn(thread: HTMLElement, sources: SourceRef[]): HTMLElement {
    const group = document.createElement("div");
    group.className = "msg-group";
    const ts = fmtTime(Date.now());
    const docs = groupSources(sources);
    cacheDocs(docs);

    const cardsHtml = docs.length
        ? docs
              .map((d) => renderCardHtml(d))
              .join("")
        : `<div class="sources-empty">No sources matched.</div>`;

    const lead = docs.length
        ? `Here are the documents I found:`
        : `I couldn't find any relevant documents.`;

    group.innerHTML = `
        <div class="msg-row assistant">
          <div class="avatar ai-av">AI</div>
          <div class="bubble-wrap">
            <div class="bubble assistant sources-lead">${escHtml(lead)}</div>
            <div class="sources-turn"><div class="sources-cards">${cardsHtml}</div></div>
            <div class="msg-meta">Sources · ${ts}</div>
          </div>
        </div>`;

    thread.appendChild(group);
    wireCardActions(group, docs);
    return group;
}

function renderCardHtml(d: DocGroup): string {
    const score = pct(d.bestScore);
    const chunkCount = d.refs.length;
    const synthetic = d.docKey.startsWith("__synth_");
    const viewBtn = !synthetic
        ? `<a href="#" class="sources-card-view" data-doc-key="${escHtml(d.docKey)}">[view]</a>`
        : "";
    const minimizeBtn = `<button class="sources-card-minimize" data-doc-key="${escHtml(d.docKey)}" title="Minimize" aria-expanded="true">&#9650;</button>`;
    const isRelevant = _relevantDocIds.has(d.docKey);
    const isHidden = _ignoredDocIds.has(d.docKey);
    const actionsDisabled = synthetic ? "disabled" : "";
    const relevantLabel = isRelevant ? "Relevant ✓" : "Mark relevant";
    const hideLabel = isHidden ? "Hidden" : "Hide";

    const orderedRefs = [...d.refs].sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
    const chunksHtml = orderedRefs
        .map((ref, idx) => {
            const refScore = pct(ref.score);
            const text = ref.text ?? "";
            const sectionLabel = ref.section ? escHtml(String(ref.section)) : "";
            const sectionHtml = sectionLabel
                ? `<span class="sources-chunk-section">${sectionLabel}</span>`
                : "";
            return `
              <div class="sources-chunk">
                <div class="sources-chunk-meta">
                  <span class="sources-chunk-idx">#${idx + 1}</span>
                  ${sectionHtml}
                  <span class="sources-chunk-score">${refScore}%</span>
                </div>
                <div class="sources-chunk-text markdown-body">${parseMarkdown(text)}</div>
              </div>`;
        })
        .join("");

    const chunkLabel = chunkCount > 1 ? `${chunkCount} chunks matched` : `1 chunk matched`;

    return `
      <div class="sources-card" data-doc-key="${escHtml(d.docKey)}">
        <div class="sources-card-header">
          <span class="sources-card-name" title="${escHtml(d.name)}">${escHtml(d.name)}</span>
          <span class="sources-card-score">${score}%</span>
          ${viewBtn}
          ${minimizeBtn}
        </div>
        <div class="sources-card-meta">${chunkLabel}</div>
        <div class="sources-card-chunks">${chunksHtml}</div>
        <div class="sources-card-actions">
          <button class="sources-card-action" data-action="relevant" data-doc-key="${escHtml(d.docKey)}" ${actionsDisabled || (isRelevant ? "disabled" : "")}>${relevantLabel}</button>
          <button class="sources-card-action" data-action="hide" data-doc-key="${escHtml(d.docKey)}" ${actionsDisabled || (isHidden ? "disabled" : "")}>${hideLabel}</button>
          <button class="sources-card-action" data-action="reset" data-doc-key="${escHtml(d.docKey)}" ${actionsDisabled || (!isRelevant && !isHidden ? "disabled" : "")}>Reset</button>
        </div>
      </div>`;
}

function wireCardActions(scope: HTMLElement, docs: DocGroup[]): void {
    const docMap = new Map(docs.map((d) => [d.docKey, d]));
    scope.querySelectorAll<HTMLElement>(".sources-card-view").forEach((link) => {
        link.addEventListener("click", (e) => {
            e.preventDefault();
            const key = link.dataset.docKey;
            if (!key) return;
            const doc = docMap.get(key);
            if (!doc) return;
            const top = doc.refs[0];
            void openSourceDocument({
                source: doc.source || undefined,
                source_uri: doc.sourceUri || undefined,
                source_key: doc.sourceKey || undefined,
                chunk_text: top?.text || undefined,
                original_start: top?.original_char_start,
                original_end: top?.original_char_end,
            });
        });
    });
    scope.querySelectorAll<HTMLButtonElement>(".sources-card-minimize").forEach((btn) => {
        btn.addEventListener("click", () => {
            const card = btn.closest(".sources-card");
            if (!card) return;
            const collapsed = card.classList.toggle("collapsed");
            btn.setAttribute("aria-expanded", collapsed ? "false" : "true");
            btn.setAttribute("title", collapsed ? "Expand" : "Minimize");
            btn.innerHTML = collapsed ? "&#9660;" : "&#9650;";
        });
    });
    scope.querySelectorAll<HTMLButtonElement>(".sources-card-action").forEach((btn) => {
        btn.addEventListener("click", () => {
            const action = btn.dataset.action;
            const key = btn.dataset.docKey;
            if (!key) return;
            const doc = docMap.get(key);
            if (!doc) return;
            if (action === "relevant") void markRelevant(doc);
            else if (action === "hide") void hideDoc(doc);
            else if (action === "reset") void clearDocState(doc);
        });
    });
}

async function markRelevant(doc: DocGroup): Promise<void> {
    const cid = state.activeConversationId;
    if (!cid) {
        showToast("Select a conversation first");
        return;
    }
    _relevantDocIds.add(doc.docKey);
    _ignoredDocIds.delete(doc.docKey);
    refreshCardActions(doc.docKey);
    renderRail();
    try {
        // DELETE /ignore/{doc_id} also serves as restore-to-relevant.
        await api<DocStateResponse>(
            "DELETE",
            `/console/conversations/${encodeURIComponent(cid)}/ignore/${encodeURIComponent(doc.docKey)}`,
        );
        showToast(`Marked relevant: ${doc.name}`);
    } catch (err) {
        _relevantDocIds.delete(doc.docKey);
        renderRail();
        refreshCardActions(doc.docKey);
        showToast("Failed to mark relevant: " + String(err));
    }
}

async function clearDocState(doc: DocGroup): Promise<void> {
    const cid = state.activeConversationId;
    if (!cid) {
        showToast("Select a conversation first");
        return;
    }
    const wasRelevant = _relevantDocIds.has(doc.docKey);
    const wasIgnored = _ignoredDocIds.has(doc.docKey);
    _relevantDocIds.delete(doc.docKey);
    _ignoredDocIds.delete(doc.docKey);
    refreshCardActions(doc.docKey);
    renderRail();
    try {
        await api<DocStateResponse>(
            "DELETE",
            `/console/conversations/${encodeURIComponent(cid)}/doc-state/${encodeURIComponent(doc.docKey)}`,
        );
        showToast(`Reset: ${doc.name}`);
    } catch (err) {
        if (wasRelevant) _relevantDocIds.add(doc.docKey);
        if (wasIgnored) _ignoredDocIds.add(doc.docKey);
        renderRail();
        refreshCardActions(doc.docKey);
        showToast("Failed to reset: " + String(err));
    }
}

async function hideDoc(doc: DocGroup): Promise<void> {
    const cid = state.activeConversationId;
    if (!cid) {
        showToast("Select a conversation first");
        return;
    }
    _ignoredDocIds.add(doc.docKey);
    _relevantDocIds.delete(doc.docKey);
    refreshCardActions(doc.docKey);
    renderRail();
    try {
        await api<DocStateResponse>(
            "POST",
            `/console/conversations/${encodeURIComponent(cid)}/ignore`,
            { doc_id: doc.docKey },
        );
        showToast(`Hidden: ${doc.name}`);
    } catch (err) {
        _ignoredDocIds.delete(doc.docKey);
        renderRail();
        refreshCardActions(doc.docKey);
        showToast("Failed to hide document: " + String(err));
    }
}

function refreshCardActions(docKey: string): void {
    const selector = `.sources-card[data-doc-key="${CSS.escape(docKey)}"], .citation-card[data-doc-key="${CSS.escape(docKey)}"]`;
    document.querySelectorAll<HTMLElement>(selector).forEach((card) => {
        const isRelevant = _relevantDocIds.has(docKey);
        const isHidden = _ignoredDocIds.has(docKey);
        const relBtn = card.querySelector<HTMLButtonElement>('[data-action="relevant"]');
        const hideBtn = card.querySelector<HTMLButtonElement>('[data-action="hide"]');
        const resetBtn = card.querySelector<HTMLButtonElement>('[data-action="reset"]');
        if (relBtn) {
            relBtn.textContent = isRelevant ? "Relevant ✓" : "Mark relevant";
            relBtn.disabled = isRelevant;
        }
        if (hideBtn) {
            hideBtn.textContent = isHidden ? "Hidden" : "Hide";
            hideBtn.disabled = isHidden;
        }
        if (resetBtn) {
            resetBtn.disabled = !isRelevant && !isHidden;
        }
    });
}

/** Wire Relevant/Hide/Reset buttons on citation cards in answer mode. */
export function wireCitationActions(scope: HTMLElement): void {
    scope.querySelectorAll<HTMLElement>(".citation-card[data-doc-key]").forEach((card) => {
        const docKey = card.getAttribute("data-doc-key") || "";
        if (!docKey) return;
        const docName =
            card.getAttribute("data-doc-name") ||
            card.querySelector<HTMLElement>(".citation-name")?.textContent?.trim() ||
            docKey;
        const cached = _docCache.get(docKey);
        const stub: DocGroup = {
            docKey,
            name: cached?.name || docName,
            bestScore: 0,
            excerpt: "",
            refs: [],
            sourceUri: cached?.sourceUri || card.getAttribute("data-source-uri") || "",
            sourceKey: cached?.sourceKey || card.getAttribute("data-source-key") || "",
            source: cached?.source || card.getAttribute("data-source") || "",
        };
        card.querySelectorAll<HTMLButtonElement>(".citation-card-action").forEach((btn) => {
            btn.addEventListener("click", (ev) => {
                ev.stopPropagation();
                const action = btn.dataset.action;
                if (action === "relevant") void markRelevant(stub);
                else if (action === "hide") void hideDoc(stub);
                else if (action === "reset") void clearDocState(stub);
            });
        });
        refreshCardActions(docKey);
    });
}

// ── Rail collapse ────────────────────────────────────────────────────────────

function initRailCollapse(): void {
    const rail = document.getElementById("chatRail");
    const btn = document.getElementById("chatRailToggle");
    if (!rail || !btn) return;
    const collapsed = localStorage.getItem(RAIL_COLLAPSED_KEY) === "1";
    applyRailCollapsed(rail, btn, collapsed);
    btn.addEventListener("click", () => {
        const next = !rail.classList.contains("collapsed");
        applyRailCollapsed(rail, btn, next);
        localStorage.setItem(RAIL_COLLAPSED_KEY, next ? "1" : "0");
    });
}

function applyRailCollapsed(rail: HTMLElement, btn: HTMLElement, collapsed: boolean): void {
    rail.classList.toggle("collapsed", collapsed);
    btn.setAttribute("aria-expanded", collapsed ? "false" : "true");
    btn.setAttribute("title", collapsed ? "Expand" : "Collapse");
    btn.innerHTML = collapsed ? "&#9664;" : "&#9654;";
}

// ── Right rail rendering ─────────────────────────────────────────────────────

function renderRail(): void {
    renderRailList("chatRailRelevant", "chatRailRelevantCount", _relevantDocIds, "relevant");
    renderRailList("chatRailHidden", "chatRailHiddenCount", _ignoredDocIds, "hidden");
}

function renderRailList(
    listId: string,
    countId: string,
    ids: Set<string>,
    kind: "relevant" | "hidden",
): void {
    const list = document.getElementById(listId);
    const count = document.getElementById(countId);
    if (!list || !count) return;
    count.textContent = String(ids.size);
    if (!ids.size) {
        list.innerHTML = `<div class="chat-rail-list-empty">${kind === "relevant" ? "No relevant docs yet." : "Nothing hidden."}</div>`;
        return;
    }
    let html = "";
    for (const docKey of ids) {
        const cached = _docCache.get(docKey);
        const name = escHtml(cached?.name || docKey);
        const synthetic = !cached;
        const viewBtn = !synthetic
            ? `<a href="#" class="chat-rail-item-view" data-doc-key="${escHtml(docKey)}">view</a>`
            : "";
        const actionLabel = kind === "relevant" ? "hide" : "restore";
        const actionAttr = kind === "relevant" ? "hide" : "relevant";
        html += `
          <div class="chat-rail-item" data-doc-key="${escHtml(docKey)}">
            <span class="chat-rail-item-name" title="${name}">${name}</span>
            ${viewBtn}
            <button class="chat-rail-item-action" data-action="${actionAttr}" data-doc-key="${escHtml(docKey)}">${actionLabel}</button>
            <button class="chat-rail-item-action" data-action="reset" data-doc-key="${escHtml(docKey)}" title="Remove from list">reset</button>
          </div>`;
    }
    list.innerHTML = html;
    wireRailActions(list);
}

function wireRailActions(scope: HTMLElement): void {
    scope.querySelectorAll<HTMLElement>(".chat-rail-item-view").forEach((link) => {
        link.addEventListener("click", (e) => {
            e.preventDefault();
            const key = link.dataset.docKey;
            const cached = key ? _docCache.get(key) : undefined;
            if (!cached) return;
            void openSourceDocument({
                source: cached.source || undefined,
                source_uri: cached.sourceUri || undefined,
                source_key: cached.sourceKey || undefined,
            });
        });
    });
    scope.querySelectorAll<HTMLButtonElement>(".chat-rail-item-action").forEach((btn) => {
        btn.addEventListener("click", () => {
            const key = btn.dataset.docKey;
            const action = btn.dataset.action;
            if (!key) return;
            const cached = _docCache.get(key);
            const docStub: DocGroup = {
                docKey: key,
                name: cached?.name || key,
                bestScore: 0,
                excerpt: "",
                refs: [],
                sourceUri: cached?.sourceUri || "",
                sourceKey: cached?.sourceKey || "",
                source: cached?.source || "",
            };
            if (action === "hide") void hideDoc(docStub);
            else if (action === "relevant") void markRelevant(docStub);
            else if (action === "reset") void clearDocState(docStub);
        });
    });
}

export function applyDocState(relevant: string[], ignored: string[]): void {
    _relevantDocIds.clear();
    _ignoredDocIds.clear();
    relevant.forEach((id) => _relevantDocIds.add(id));
    ignored.forEach((id) => _ignoredDocIds.add(id));
    renderRail();
    // Refresh any visible cards for these ids.
    [..._relevantDocIds, ..._ignoredDocIds].forEach(refreshCardActions);
}

async function fetchAndRenderDocState(conversationId: string): Promise<void> {
    try {
        const data = await api<DocStateResponse>(
            "GET",
            `/console/conversations/${encodeURIComponent(conversationId)}/doc-state`,
        );
        applyDocState(data.relevant_doc_ids ?? [], data.ignored_doc_ids ?? []);
    } catch {
        // Non-fatal; rail just stays empty.
    }
}

/** True iff a stored history turn should be rendered as cards instead of bubble. */
export function isSourcesTurn(turn: { content: string; sources?: SourceRef[] }): boolean {
    return !turn.content.trim() && !!turn.sources && turn.sources.length > 0;
}
