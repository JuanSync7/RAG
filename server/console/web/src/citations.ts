// @summary
// Citation rendering + expand/collapse handlers for the user console.
// `buildCitationsHtml` is consumed by both the streaming and non-stream code
// paths plus the conversation-history replay in conversations.ts.
// @end-summary

import { escHtml } from "./dom";
import { pct } from "./format";
import { parseMarkdown } from "./markdown";
import { showToast } from "./toast";
import { docKeyFromMeta } from "./chatMode";
import { openDocViewer } from "./docViewer";
import type { ChunkResult } from "./user-types";

export interface ViewPayload {
    source?: string;
    source_uri?: string;
    source_key?: string;
    chunk_text?: string;
    original_start?: number;
    original_end?: number;
    refactored_start?: number;
    refactored_end?: number;
    provenance_confidence?: number;
}

/**
 * Open a source document. Delegates to the in-page side-panel viewer
 * (``docViewer``), which splits the chat 50/50 and renders the document in an
 * iframe instead of opening a new browser tab. Kept as the shared entry point so
 * citation cards, the bubble "Sources:" links, and the chat-rail all open
 * documents the same way.
 */
export async function openSourceDocument(payload: ViewPayload): Promise<void> {
    await openDocViewer(payload);
}

const _viewPayloads = new Map<string, ViewPayload>();
let _viewCounter = 0;

/** Drop all registered source-view payloads. Called when the thread is reset
 *  (conversation switch / new chat) so the map can't grow unbounded across a
 *  long session — the `viewKey`s in the discarded HTML are unreachable anyway. */
export function clearViewPayloads(): void {
    _viewPayloads.clear();
}

/**
 * Render the references panel.
 *
 * The generator numbers context chunks `[1..N]` in the SAME order they appear in
 * `results`, so citation `[k]` in the answer maps to `results[k-1]`. We parse the
 * `[N]` markers out of `answerText` to show the chunks the answer actually CITED
 * first (each tagged with its `[N]`), and tuck the remaining retrieved-but-uncited
 * chunks under a collapsed "show more" expander — nothing is hidden, the panel is
 * just honest about what the answer used.
 *
 * Grounding is intentionally soft (the model may lean on a chunk without dropping a
 * `[N]`), so if we can't detect ANY citations (no answer text, or nothing parseable)
 * we fall back to showing every retrieved chunk, exactly as before.
 *
 * NOTE (CLI/UI parity): this cited-first presentation is intentionally specific to
 * the web chat references panel. The CLI's `Top N retrieved chunks` view is a
 * retrieval-inspection surface and deliberately lists every retrieved chunk.
 */
export function buildCitationsHtml(results: ChunkResult[], answerText?: string): string {
    if (!results.length) return "";

    const citedNums = parseCitedNums(results, answerText);

    const renderCard = (r: ChunkResult, n: number): string => {
        const meta = r.metadata || {};
        const filenameRaw = String(meta.source ?? meta.filename ?? "Unknown source");
        const filename = escHtml(filenameRaw);
        const section = escHtml(String(meta.section ?? meta.heading ?? ""));
        const score = pct(r.score);
        const scoreClass = score >= 80 ? "high" : score >= 50 ? "mid" : "low";
        const chunkHtml = parseMarkdown(r.text || "");
        const sourceUri = String(meta.source_uri ?? "").trim();
        const source = String(meta.source ?? "").trim();
        const sourceKey = String(meta.source_key ?? "").trim();
        const docKey = docKeyFromMeta(meta as Record<string, unknown>);
        const cardAttrs = docKey
            ? ` data-doc-key="${escHtml(docKey)}" data-doc-name="${escHtml(filenameRaw)}"`
                + (source ? ` data-source="${escHtml(source)}"` : "")
                + (sourceUri ? ` data-source-uri="${escHtml(sourceUri)}"` : "")
                + (sourceKey ? ` data-source-key="${escHtml(sourceKey)}"` : "")
            : "";
        const actionsHtml = docKey
            ? `<div class="citation-card-actions" onclick="event.stopPropagation()">`
                + `<button class="citation-card-action" data-action="relevant">Mark relevant</button>`
                + `<button class="citation-card-action" data-action="hide">Hide</button>`
                + `<button class="citation-card-action" data-action="reset" disabled>Reset</button>`
                + `</div>`
            : "";
        const viewKey = registerViewPayload(r);
        return `
          <div class="citation-card"${cardAttrs} onclick="toggleCitation(this)">
            <div class="citation-header">
              <span class="citation-num">[${n}]</span>
              <span class="citation-icon">&#128196;</span>
              <div class="citation-info">
                <div class="citation-filename"><span class="citation-name">${filename}</span>${viewKey ? `<a href="#" class="citation-view" onclick="event.stopPropagation();openSourceView(event,'${viewKey}')">[view]</a>` : ""}</div>
                ${section ? `<div class="citation-section">${section}</div>` : ""}
              </div>
              <div class="relevance-bar-wrap">
                <span class="relevance-pct ${scoreClass}">${score}%</span>
                <div class="relevance-bar"><div class="relevance-fill ${scoreClass}" style="width:${score}%"></div></div>
              </div>
              <span class="citation-chevron">&#8964;</span>
            </div>
            <div class="citation-body">
              <div class="citation-chunk markdown-body">${chunkHtml}</div>
              ${actionsHtml}
            </div>
          </div>`;
    };

    const cited: string[] = [];
    const uncited: string[] = [];
    results.forEach((r, i) => {
        const n = i + 1;
        (citedNums.has(n) ? cited : uncited).push(renderCard(r, n));
    });

    // The concise "Sources:" line is rendered inside the assistant bubble
    // (see buildSourcesLineHtml), not here — this panel is the detailed evidence.

    // Fallback: no detectable citations — show everything as "retrieved".
    if (!cited.length) {
        const label = `<div class="citation-label">&#128206; ${results.length} source${results.length > 1 ? "s" : ""} retrieved</div>`;
        return label + uncited.join("");
    }

    let html = `<div class="citation-label">&#128206; ${cited.length} source${cited.length > 1 ? "s" : ""} cited</div>`;
    html += cited.join("");
    if (uncited.length) {
        html += `
          <details class="citations-more">
            <summary class="citations-more-summary">Show ${uncited.length} more retrieved source${uncited.length > 1 ? "s" : ""} (not cited)</summary>
            <div class="citations-more-body">${uncited.join("")}</div>
          </details>`;
    }
    return html;
}

function numOrUndef(v: unknown): number | undefined {
    const n = Number(v);
    return Number.isFinite(n) ? n : undefined;
}

/** Register a chunk's source identity as a view payload and return its key (or
 *  "" when the chunk has no openable source). Shared by the citation cards and
 *  the bubble "Sources:" links so both open a document through the one viewer. */
function registerViewPayload(r: ChunkResult): string {
    const meta = r.metadata || {};
    const source = String(meta.source ?? "").trim();
    const sourceUri = String(meta.source_uri ?? "").trim();
    const sourceKey = String(meta.source_key ?? "").trim();
    if (!(source || sourceUri || sourceKey)) return "";
    const viewKey = `view-${++_viewCounter}`;
    _viewPayloads.set(viewKey, {
        source: source || undefined,
        source_uri: sourceUri || undefined,
        source_key: sourceKey || undefined,
        chunk_text: r.text || undefined,
        original_start: numOrUndef(meta.original_char_start),
        original_end: numOrUndef(meta.original_char_end),
        refactored_start: numOrUndef(meta.refactored_char_start),
        refactored_end: numOrUndef(meta.refactored_char_end),
        provenance_confidence: numOrUndef(meta.provenance_confidence),
    });
    return viewKey;
}

/** Parse the `[N]` citation markers out of the answer into the set of result
 *  indices (1-based) the answer actually cited. Shared by the references panel
 *  and the bubble "Sources:" line so both agree on what was used. */
function parseCitedNums(results: ChunkResult[], answerText?: string): Set<number> {
    const citedNums = new Set<number>();
    if (answerText) {
        const re = /\[(\d+)\]/g;
        let m: RegExpExecArray | null;
        while ((m = re.exec(answerText)) !== null) {
            const n = Number(m[1]);
            if (Number.isInteger(n) && n >= 1 && n <= results.length) citedNums.add(n);
        }
    }
    return citedNums;
}

/** The concise "Sources: <docs>" line shown INSIDE the assistant answer bubble.
 *  Same document-set logic as the references panel (cited docs first, else all
 *  retrieved), basename, deduped. Returns "" when there are no results. */
export function buildSourcesLineHtml(results: ChunkResult[], answerText?: string): string {
    if (!results.length) return "";
    return sourcesUsedHtml(sourcesUsed(results, parseCitedNums(results, answerText)));
}

/** One distinct document per basename (first-seen order), each paired with a
 *  representative chunk so the "Sources:" line can link straight to the document.
 *  When the answer dropped `[N]` citations we list only the cited documents (the
 *  ones it actually used); otherwise we list every retrieved document. Shared
 *  shape with the CLI's "Sources:" line (CLI/UI parity). */
interface SourceUsedDoc {
    name: string;
    result: ChunkResult;
}

function sourcesUsed(results: ChunkResult[], citedNums: Set<number>): SourceUsedDoc[] {
    const pick = citedNums.size
        ? results.filter((_, i) => citedNums.has(i + 1))
        : results;
    const seen = new Set<string>();
    const docs: SourceUsedDoc[] = [];
    for (const r of pick) {
        const meta = r.metadata || {};
        const raw = String(meta.source ?? meta.filename ?? "").trim();
        if (!raw) continue;
        const name = raw.split("/").pop() || raw;
        if (!seen.has(name)) {
            seen.add(name);
            docs.push({ name, result: r });
        }
    }
    return docs;
}

/** The "Sources:" line: each document is a link that opens it in the side-panel
 *  viewer (same `openSourceView` path as the citation cards' [view]). Documents
 *  without an openable source fall back to plain text. */
function sourcesUsedHtml(docs: SourceUsedDoc[]): string {
    if (!docs.length) return "";
    const list = docs
        .map((d) => {
            const label = escHtml(d.name);
            const viewKey = registerViewPayload(d.result);
            return viewKey
                ? `<a href="#" class="sources-used-doc" onclick="event.preventDefault();openSourceView(event,'${viewKey}')">${label}</a>`
                : `<span class="sources-used-doc">${label}</span>`;
        })
        .join(", ");
    return `<div class="sources-used"><span class="sources-used-label">Sources:</span> ${list}</div>`;
}

async function openSourceView(e: Event, viewKey: string): Promise<void> {
    e.preventDefault();
    const payload = _viewPayloads.get(viewKey);
    if (!payload) {
        showToast("Citation context lost — try re-running the query.");
        return;
    }
    await openSourceDocument(payload);
}

function toggleCitation(card: HTMLElement): void {
    card.classList.toggle("expanded");
}

export function revealCitations(citationsEl: HTMLElement): void {
    citationsEl.style.display = "block";
    citationsEl.classList.remove("reveal");
    // Force reflow so the animation re-fires on subsequent reveals.
    void citationsEl.offsetWidth;
    citationsEl.classList.add("reveal");
}

export function initCitations(): void {
    (window as unknown as Record<string, unknown>)["toggleCitation"] = toggleCitation;
    (window as unknown as Record<string, unknown>)["openSourceView"] = openSourceView;
}
