/**
 * @summary
 * DOM rendering helpers for the admin/operator console: text/JSON writers,
 * markdown answer normalization, timing tables, and reranked-source rendering.
 * Also includes shared slash-command summary helpers.
 * Exports: write, asNumber, asOptionalNumber, escapeHtml, normalizeGeneratedAnswer,
 *   renderMarkdown, renderTiming, rerankSourceLink, formatChunkNumber,
 *   compactExcerpt, renderRerankedOriginalDocs, commandSummary, setSuggestions,
 *   initSlashCommandHints
 * Deps: admin-types, admin-api
 * @end-summary
 */

import type {
    ConsoleCommandSpec,
    QueryResult,
    TimingPayload,
} from "./admin-types.js";
import { byId, fetchConsoleCommands } from "./admin-api.js";

export function write(id: string, value: unknown): void {
    const out = typeof value === "string" ? value : JSON.stringify(value, null, 2);
    byId(id).textContent = out;
}

export function asNumber(value: unknown): number | null {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
}

export function asOptionalNumber(value: unknown): number | undefined {
    const n = asNumber(value);
    return n === null ? undefined : n;
}

export function escapeHtml(value: unknown): string {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

export function normalizeGeneratedAnswer(markdown: string): string {
    let text = markdown || "";
    if (!text) {
        return "";
    }

    // Some model outputs carry escaped newlines as literal characters.
    if (!text.includes("\n") && text.includes("\\n")) {
        text = text.replace(/\\n/g, "\n");
    }

    // Keep source previews in the dedicated rerank table instead of mixed into answer body.
    const rerankSectionIdx = text.search(/\n#{1,6}\s*Top reranked original documents\b/i);
    if (rerankSectionIdx >= 0) {
        text = text.slice(0, rerankSectionIdx).trimEnd();
    }

    // Remove wrapper headings that make the answer feel like a template.
    text = text.replace(/^\s*#{0,6}\s*Output(?:s)?\s*:?\s*$/gim, "");
    text = text.replace(/^\s*#{0,6}\s*Comprehensive Overview of the Entire System\s*:?\s*$/gim, "");
    text = text.replace(/^\s*Inputs?\s*:\s*$/gim, "");
    text = text.replace(/^\s*Outputs?\s*:\s*$/gim, "");

    // If a rigid template starts, keep only the answer lead.
    const templateCutCandidates = [
        text.search(/\n\s*Inputs?\s*:/i),
        text.search(/\n\s*Outputs?\s*:/i),
        text.search(/\n\s*Comprehensive Overview of the Entire System/i),
    ].filter((idx) => idx >= 0);
    if (templateCutCandidates.length > 0) {
        const cutIdx = Math.min(...templateCutCandidates);
        if (cutIdx > 0) {
            text = text.slice(0, cutIdx).trimEnd();
        }
    }

    // Remove noisy runtime placeholder lines if they appear in answer text.
    text = text.replace(
        /^<original source file unavailable in API runtime; showing reranked chunk excerpts>\s*$/gim,
        "",
    );
    return text.trim();
}

export function renderMarkdown(id: string, markdown: string): void {
    const raw = normalizeGeneratedAnswer(markdown);
    const parsed = window.marked ? window.marked.parse(raw) : raw;
    const safe = window.DOMPurify ? window.DOMPurify.sanitize(parsed) : parsed;
    byId(id).innerHTML = safe || "<em>No answer generated yet.</em>";
}

export function renderTiming(timing: TimingPayload | null): void {
    const el = byId("timingOut");
    if (!timing) {
        el.innerHTML = '<div class="muted">No timing yet.</div>';
        return;
    }

    const latency = asNumber(timing.latency_ms);
    const retrieval = asNumber(timing.retrieval_ms);
    const generation = asNumber(timing.generation_ms);
    const tokens = asNumber(timing.token_count);
    const totals = (timing.timing_totals || {}) as Record<string, unknown>;
    const stageTimings = Array.isArray(timing.stage_timings) ? timing.stage_timings : [];

    const metric = (key: string, value: string): string =>
        `<div class="timing-item"><div class="timing-key">${escapeHtml(key)}</div><div class="timing-val">${escapeHtml(value)}</div></div>`;

    const topGrid = [
        metric("Total", latency !== null ? `${latency.toFixed(1)} ms` : "n/a"),
        metric("Retrieval", retrieval !== null ? `${retrieval.toFixed(1)} ms` : "n/a"),
        metric("Generation", generation !== null ? `${generation.toFixed(1)} ms` : "n/a"),
        metric("Tokens", tokens !== null ? `${Math.round(tokens)}` : "n/a"),
    ].join("");

    const totalRows = Object.entries(totals)
        .map(([k, v]) => `<tr><td>${escapeHtml(k)}</td><td>${escapeHtml(v)}</td></tr>`)
        .join("");
    const totalsTable = totalRows
        ? `<div style="margin-top:10px;"><div class="muted" style="margin-bottom:6px;">Bucket totals</div><table class="rerank-table"><thead><tr><th>Bucket</th><th>ms</th></tr></thead><tbody>${totalRows}</tbody></table></div>`
        : "";

    const stageRows = stageTimings
        .map((stage) => {
            const name = String(stage.stage ?? "unknown");
            const bucket = String(stage.bucket ?? "other");
            const ms = asNumber(stage.ms);
            return `<tr><td>${escapeHtml(name)}</td><td>${escapeHtml(bucket)}</td><td>${escapeHtml(ms !== null ? ms.toFixed(1) : "n/a")}</td></tr>`;
        })
        .join("");
    const stageTable = stageRows
        ? `<div style="margin-top:10px;"><div class="muted" style="margin-bottom:6px;">Stage timings</div><table class="rerank-table"><thead><tr><th>Stage</th><th>Bucket</th><th>ms</th></tr></thead><tbody>${stageRows}</tbody></table></div>`
        : "";

    // Token budget / context window usage
    let budgetHtml = "";
    const tb = timing.token_budget;
    if (tb && typeof tb === "object") {
        const pct = tb.usage_percent ?? 0;
        const inp = tb.input_tokens ?? 0;
        const ctx = tb.context_length ?? 0;
        const mdl = tb.model_name ?? "";
        const pctColor = pct >= 90 ? "#f87171" : pct >= 70 ? "#facc15" : "#4ade80";
        let budgetRows = `<tr><td>Context usage</td><td style="color:${pctColor};font-weight:700">${pct.toFixed(0)}%</td><td>${inp} / ${ctx} tokens</td><td>${escapeHtml(mdl)}</td></tr>`;
        const bd = tb.breakdown;
        if (bd && typeof bd === "object") {
            budgetRows += `<tr><td colspan="4" style="color:var(--muted);font-size:12px">system:${bd.system_prompt ?? 0}  memory:${bd.memory_context ?? 0}  chunks:${bd.retrieval_chunks ?? 0}  query:${bd.user_query ?? 0}  overhead:${bd.template_overhead ?? 0}</td></tr>`;
        }
        const apt = tb.actual_prompt_tokens ?? 0;
        const act = tb.actual_completion_tokens ?? 0;
        if (apt) {
            budgetRows += `<tr><td>Actual tokens</td><td colspan="3">${apt} in + ${act} out = ${apt + act} total</td></tr>`;
        }
        if (tb.cost_usd && tb.cost_usd > 0) {
            budgetRows += `<tr><td>Cost</td><td colspan="3">$${tb.cost_usd.toFixed(4)}</td></tr>`;
        }
        budgetHtml = `<div style="margin-top:10px;"><div class="muted" style="margin-bottom:6px;">Context window</div><table class="rerank-table"><tbody>${budgetRows}</tbody></table></div>`;
    }

    el.innerHTML = `<div class="timing-grid">${topGrid}</div>${budgetHtml}${totalsTable}${stageTable}`;
}

export function rerankSourceLink(metadata: Record<string, unknown>): { label: string; href: string } {
    const source = String(metadata.source ?? "").trim();
    const sourceUri = String(metadata.source_uri ?? "").trim();
    const chunkIdxRaw = Number(metadata.chunk_index);
    const chunkIdx = Number.isFinite(chunkIdxRaw) ? chunkIdxRaw : null;
    const start = Number(metadata.original_char_start);
    const end = Number(metadata.original_char_end);
    const params = new URLSearchParams();
    if (sourceUri) {
        params.set("source_uri", sourceUri);
    } else if (source) {
        params.set("source", source);
    } else {
        return { label: "unknown", href: "" };
    }
    if (Number.isFinite(start) && Number.isFinite(end) && end > start) {
        params.set("start", String(start));
        params.set("end", String(end));
    }
    if (chunkIdx !== null && chunkIdx >= 0) {
        params.set("chunk", String(chunkIdx + 1));
    }
    const display = sourceUri || source;
    return {
        label: display,
        href: `/console/source-document/view?${params.toString()}`,
    };
}

export function formatChunkNumber(metadata: Record<string, unknown>): string {
    const idx = Number(metadata.chunk_index);
    if (!Number.isFinite(idx) || idx < 0) {
        return "n/a";
    }
    return String(idx + 1);
}

export function compactExcerpt(text: string, maxLen = 220): string {
    const flat = text.replace(/\s+/g, " ").trim();
    if (!flat) {
        return "<no excerpt>";
    }
    if (flat.length <= maxLen) {
        return flat;
    }
    return `${flat.slice(0, maxLen)}...`;
}

export async function renderRerankedOriginalDocs(results: QueryResult[]): Promise<void> {
    const out = byId("rerankDocsOut");
    if (!Array.isArray(results) || results.length === 0) {
        out.innerHTML = '<div class="muted">No reranked results.</div>';
        return;
    }
    const rows = results.map((r, i) => {
        const metadata = (r.metadata || {}) as Record<string, unknown>;
        const link = rerankSourceLink(metadata);
        const score = Number(r.score);
        const scoreText = Number.isFinite(score) ? score.toFixed(4) : "n/a";
        const chunkNo = formatChunkNumber(metadata);
        const excerpt = compactExcerpt(String(r.text || ""));
        const sourceCell = link.href
            ? `<a class="source-link" target="_blank" rel="noopener noreferrer" href="${escapeHtml(link.href)}">${escapeHtml(link.label)}</a>`
            : `<span>${escapeHtml(link.label)}</span>`;
        return (
            "<tr>" +
            `<td>${i + 1}</td>` +
            `<td>${sourceCell}</td>` +
            `<td><code>${escapeHtml(chunkNo)}</code></td>` +
            `<td><code>${escapeHtml(scoreText)}</code></td>` +
            `<td>${escapeHtml(excerpt)}</td>` +
            "</tr>"
        );
    });
    out.innerHTML =
        '<table class="rerank-table">' +
        "<thead><tr><th>#</th><th>Document</th><th>Chunk</th><th>Score</th><th>Excerpt</th></tr></thead>" +
        `<tbody>${rows.join("")}</tbody>` +
        "</table>";
}

export function commandSummary(commands: ConsoleCommandSpec[]): string {
    if (!commands.length) {
        return "No commands available.";
    }
    return commands
        .map((cmd) => {
            const args = cmd.args_hint ? ` ${cmd.args_hint}` : "";
            return `/${cmd.name}${args} - ${cmd.description}`;
        })
        .join("\n");
}

export function setSuggestions(elementId: string, commands: ConsoleCommandSpec[]): void {
    const out = byId(elementId);
    out.textContent = commands.map((cmd) => `/${cmd.name}`).join("  ");
}

export async function initSlashCommandHints(): Promise<void> {
    try {
        const [queryCmds, ingestCmds] = await Promise.all([
            fetchConsoleCommands("query"),
            fetchConsoleCommands("ingest"),
        ]);
        setSuggestions("querySlashSuggestions", queryCmds);
        setSuggestions("ingestSlashSuggestions", ingestCmds);
    } catch {
        setSuggestions("querySlashSuggestions", []);
        setSuggestions("ingestSlashSuggestions", []);
    }
}
