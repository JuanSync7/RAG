// @summary
// Citation rendering + expand/collapse handlers for the user console.
// `buildCitationsHtml` is consumed by both the streaming and non-stream code
// paths plus the conversation-history replay in conversations.ts.
// @end-summary

import { escHtml } from "./dom";
import { parseMarkdown } from "./markdown";
import { apiBase, authHeaders } from "./api";
import { showToast } from "./toast";
import { docKeyFromMeta } from "./chatMode";
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

export async function openSourceDocument(payload: ViewPayload): Promise<void> {
    const url = apiBase() + "/console/source-document/view";
    try {
        const res = await fetch(url, {
            method: "POST",
            headers: authHeaders(),
            body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const contentType = (res.headers.get("Content-Type") ?? "text/html").split(";")[0].trim();
        const buf = await res.arrayBuffer();
        const blob = new Blob([buf], { type: contentType });
        const blobUrl = URL.createObjectURL(blob);
        const win = window.open(blobUrl, "_blank");
        if (!win) {
            showToast("Pop-up blocked. Allow pop-ups to view sources.");
            URL.revokeObjectURL(blobUrl);
            return;
        }
        setTimeout(() => URL.revokeObjectURL(blobUrl), 60_000);
    } catch (err) {
        showToast("Could not open source: " + String(err));
    }
}

const _viewPayloads = new Map<string, ViewPayload>();
let _viewCounter = 0;

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

    const citedNums = new Set<number>();
    if (answerText) {
        const re = /\[(\d+)\]/g;
        let m: RegExpExecArray | null;
        while ((m = re.exec(answerText)) !== null) {
            const n = Number(m[1]);
            if (Number.isInteger(n) && n >= 1 && n <= results.length) citedNums.add(n);
        }
    }

    const renderCard = (r: ChunkResult, n: number): string => {
        const meta = r.metadata || {};
        const filenameRaw = String(meta.source ?? meta.filename ?? "Unknown source");
        const filename = escHtml(filenameRaw);
        const section = escHtml(String(meta.section ?? meta.heading ?? ""));
        const score = Math.round(r.score * 100);
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
        let viewKey = "";
        if (sourceKey || sourceUri || source) {
            viewKey = `view-${++_viewCounter}`;
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
        }
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
