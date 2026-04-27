// @summary
// Chat-pane renderer toggle: "answer" (LLM bubble + citation strip) vs
// "sources" (doc-cards only, no generation). Mode is persisted globally in
// localStorage and forwarded to /console/query as `mode: "retrieval"` when
// in sources mode. Backend writes both modes to the same conversation history,
// so a single thread can be replayed under either renderer.
// @end-summary

import { byId, escHtml, fmtTime } from "./dom";
import { parseMarkdown } from "./markdown";
import { openSourceDocument } from "./citations";
import type { SourceRef } from "./user-types";

export type ChatRenderMode = "answer" | "sources";

const STORAGE_KEY = "rw_chat_mode";

let _mode: ChatRenderMode =
    (localStorage.getItem(STORAGE_KEY) as ChatRenderMode | null) === "sources" ? "sources" : "answer";

export function getChatMode(): ChatRenderMode {
    return _mode;
}

export function setChatMode(mode: ChatRenderMode): void {
    _mode = mode;
    localStorage.setItem(STORAGE_KEY, mode);
    syncToggleUI();
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

export function initChatMode(): void {
    const toggle = document.getElementById("chatModeToggle");
    if (!toggle) return;
    toggle.querySelectorAll<HTMLButtonElement>("[data-mode]").forEach((btn) => {
        btn.addEventListener("click", () => {
            const mode = btn.dataset.mode as ChatRenderMode | undefined;
            if (mode === "answer" || mode === "sources") setChatMode(mode);
        });
    });
    syncToggleUI();
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

/** Append a "sources turn" (doc cards, no LLM bubble) to the message thread. */
export function appendSourcesTurn(thread: HTMLElement, sources: SourceRef[]): HTMLElement {
    const group = document.createElement("div");
    group.className = "msg-group";
    const ts = fmtTime(Date.now());
    const docs = groupSources(sources);

    const cardsHtml = docs.length
        ? docs
              .map((d) => {
                  const score = Math.round(d.bestScore * 100);
                  const excerptHtml = d.excerpt ? parseMarkdown(d.excerpt) : "";
                  const chunkCount = d.refs.length;
                  const meta = chunkCount > 1 ? `${chunkCount} chunks matched` : `1 chunk matched`;
                  const canView = !d.docKey.startsWith("__synth_");
                  const viewBtn = canView
                      ? `<a href="#" class="sources-card-view" data-doc-key="${escHtml(d.docKey)}">[view]</a>`
                      : "";
                  return `
                    <div class="sources-card" data-doc-key="${escHtml(d.docKey)}">
                      <div class="sources-card-header">
                        <span class="sources-card-name" title="${escHtml(d.name)}">${escHtml(d.name)}</span>
                        <span class="sources-card-score">${score}%</span>
                        ${viewBtn}
                      </div>
                      <div class="sources-card-excerpt markdown-body">${excerptHtml}</div>
                      <div class="sources-card-meta">${meta}</div>
                    </div>`;
              })
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
}

/** True iff a stored history turn should be rendered as cards instead of bubble. */
export function isSourcesTurn(turn: { content: string; sources?: SourceRef[] }): boolean {
    return !turn.content.trim() && !!turn.sources && turn.sources.length > 0;
}
