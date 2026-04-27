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
import { parseMarkdown } from "./markdown";
import { api } from "./api";
import { openSourceDocument } from "./citations";
import { showToast } from "./toast";
import { state } from "./state";
import type { SourceRef, DocStateResponse } from "./user-types";

export type ChatRenderMode = "answer" | "sources";

const STORAGE_KEY = "rw_chat_mode";

let _mode: ChatRenderMode =
    (localStorage.getItem(STORAGE_KEY) as ChatRenderMode | null) === "sources" ? "sources" : "answer";

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
    syncToggleUI();
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

    group.innerHTML = `
        <div class="msg-row assistant">
          <div class="avatar ai-av">AI</div>
          <div class="bubble-wrap">
            <div class="sources-turn"><div class="sources-cards">${cardsHtml}</div></div>
            <div class="msg-meta">Sources · ${ts}</div>
          </div>
        </div>`;

    thread.appendChild(group);
    wireCardActions(group, docs);
    return group;
}

function renderCardHtml(d: DocGroup): string {
    const score = Math.round(d.bestScore * 100);
    const excerptHtml = d.excerpt ? parseMarkdown(d.excerpt) : "";
    const chunkCount = d.refs.length;
    const meta = chunkCount > 1 ? `${chunkCount} chunks matched` : `1 chunk matched`;
    const synthetic = d.docKey.startsWith("__synth_");
    const viewBtn = !synthetic
        ? `<a href="#" class="sources-card-view" data-doc-key="${escHtml(d.docKey)}">[view]</a>`
        : "";
    const isRelevant = _relevantDocIds.has(d.docKey);
    const isHidden = _ignoredDocIds.has(d.docKey);
    const actionsDisabled = synthetic ? "disabled" : "";
    const relevantLabel = isRelevant ? "Relevant ✓" : "Mark relevant";
    const hideLabel = isHidden ? "Hidden" : "Hide";
    return `
      <div class="sources-card" data-doc-key="${escHtml(d.docKey)}">
        <div class="sources-card-header">
          <span class="sources-card-name" title="${escHtml(d.name)}">${escHtml(d.name)}</span>
          <span class="sources-card-score">${score}%</span>
          ${viewBtn}
        </div>
        <div class="sources-card-excerpt markdown-body">${excerptHtml}</div>
        <div class="sources-card-meta">${meta}</div>
        <div class="sources-card-actions">
          <button class="sources-card-action" data-action="relevant" data-doc-key="${escHtml(d.docKey)}" ${actionsDisabled || (isRelevant ? "disabled" : "")}>${relevantLabel}</button>
          <button class="sources-card-action" data-action="hide" data-doc-key="${escHtml(d.docKey)}" ${actionsDisabled || (isHidden ? "disabled" : "")}>${hideLabel}</button>
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
    scope.querySelectorAll<HTMLElement>(".sources-card-excerpt").forEach((el) => {
        el.addEventListener("click", () => el.classList.toggle("expanded"));
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
    document.querySelectorAll<HTMLElement>(`.sources-card[data-doc-key="${CSS.escape(docKey)}"]`).forEach((card) => {
        const isRelevant = _relevantDocIds.has(docKey);
        const isHidden = _ignoredDocIds.has(docKey);
        const relBtn = card.querySelector<HTMLButtonElement>('[data-action="relevant"]');
        const hideBtn = card.querySelector<HTMLButtonElement>('[data-action="hide"]');
        if (relBtn) {
            relBtn.textContent = isRelevant ? "Relevant ✓" : "Mark relevant";
            relBtn.disabled = isRelevant;
        }
        if (hideBtn) {
            hideBtn.textContent = isHidden ? "Hidden" : "Hide";
            hideBtn.disabled = isHidden;
        }
    });
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
