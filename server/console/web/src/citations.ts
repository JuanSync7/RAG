// @summary
// Citation rendering + expand/collapse handlers for the user console.
// `buildCitationsHtml` is consumed by both the streaming and non-stream code
// paths plus the conversation-history replay in conversations.ts.
// @end-summary

import { byId, escHtml } from "./dom";
import { parseMarkdown } from "./markdown";
import { apiBase, authHeaders } from "./api";
import { showToast } from "./toast";
import type { ChunkResult } from "./user-types";

interface ViewPayload {
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

const _viewPayloads = new Map<string, ViewPayload>();
let _viewCounter = 0;

export function buildCitationsHtml(results: ChunkResult[]): string {
    if (!results.length) return "";
    let html = `<div class="citation-label">&#128206; ${results.length} source${results.length > 1 ? "s" : ""} cited</div>`;
    results.forEach((r, i) => {
        const meta = r.metadata || {};
        const filename = escHtml(String(meta.source ?? meta.filename ?? "Unknown source"));
        const section = escHtml(String(meta.section ?? meta.heading ?? ""));
        const score = Math.round(r.score * 100);
        const scoreClass = score >= 80 ? "high" : score >= 50 ? "mid" : "low";
        const chunkHtml = parseMarkdown(r.text || "");
        const chunkId = `chunk-${i}-${Date.now()}`;
        const sourceUri = String(meta.source_uri ?? "").trim();
        const source = String(meta.source ?? "").trim();
        const sourceKey = String(meta.source_key ?? "").trim();
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
        html += `
          <div class="citation-card" onclick="toggleCitation(this)">
            <div class="citation-header">
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
              <div class="citation-chunk markdown-body" id="${chunkId}">${chunkHtml}</div>
              <button class="citation-show-more" onclick="event.stopPropagation();toggleChunk(event,'${chunkId}')">Show more</button>
            </div>
          </div>`;
    });
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

function toggleCitation(card: HTMLElement): void {
    card.classList.toggle("expanded");
}

function toggleChunk(e: Event, id: string): void {
    e.stopPropagation();
    const el = byId(id);
    el.classList.toggle("show-all");
    (e.target as HTMLElement).textContent = el.classList.contains("show-all") ? "Show less" : "Show more";
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
    (window as unknown as Record<string, unknown>)["toggleChunk"] = toggleChunk;
    (window as unknown as Record<string, unknown>)["openSourceView"] = openSourceView;
}
