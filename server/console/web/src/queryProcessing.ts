// @summary
// Renders the "Query processing" panel: a collapsible block shown ABOVE the
// reasoning ("thinking") block in an answer turn, exposing what the retriever
// did to the query BEFORE generation — the query rewrite, the agentic HyDE loop
// (per-round hypothetical answers + lexical anchor terms + target aspect, round/
// variant/kept counts, the hyde_failures fallback alarm, stop reason), any
// deep-research decomposition stats, and KG-expanded terms. Data comes from the
// `retrieval` SSE event (stream) or the query response (non-stream): the raw
// RAGResponse.metadata dict + processed_query, already shipped by the backend.
// All model-authored text is written via textContent (never innerHTML) — no
// injection. Fail-quiet: renders nothing when there is no processing to show
// (e.g. the linear non-agentic path), so plain queries stay uncluttered.
// Exports: renderQueryProcessing
// Deps: ./user-types (StreamEventData, telemetry shapes)
// @end-summary
import type {
    AgenticTelemetry,
    DeepResearchTelemetry,
    HydeRound,
    StreamEventData,
} from "./user-types";

function num(v: unknown): number {
    return typeof v === "number" && isFinite(v) ? v : 0;
}

function plural(n: number): string {
    return n === 1 ? "" : "s";
}

/** Small element builder; text (when given) is set via textContent (injection-safe). */
function el(tag: string, cls?: string, text?: string): HTMLElement {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
}

function hydeRoundEl(r: HydeRound, fallbackIndex: number): HTMLElement {
    const box = el("div", "qp-round");
    const head = el("div", "qp-round-head");
    head.textContent = `Round ${r.round ?? fallbackIndex}` + (r.target_aspect ? ` — ${r.target_aspect}` : "");
    if (r.fell_back) {
        head.appendChild(document.createTextNode(" "));
        head.appendChild(el("span", "qp-badge-warn", "literal fallback"));
    }
    box.appendChild(head);
    if (r.hypothetical_answer) {
        const hyp = el("div", "qp-hyp");
        hyp.textContent = r.hypothetical_answer; // model text → textContent
        box.appendChild(hyp);
    }
    if (r.search_terms && r.search_terms.length) {
        box.appendChild(el("div", "qp-terms", "Search terms: " + r.search_terms.join(", ")));
    }
    return box;
}

function hydeSection(a: AgenticTelemetry): HTMLElement {
    const sec = el("div", "qp-section");
    sec.appendChild(el("div", "qp-section-title", "HyDE retrieval"));

    const stats: string[] = [];
    if (num(a.rounds_run)) stats.push(`${a.rounds_run} round${plural(num(a.rounds_run))}`);
    if (num(a.hyde_variants_tried)) stats.push(`${a.hyde_variants_tried} variant${plural(num(a.hyde_variants_tried))}`);
    if (a.kept_count != null) stats.push(`${a.kept_count} chunk${plural(num(a.kept_count))} kept`);
    if (num(a.backfilled)) stats.push(`${a.backfilled} backfilled`);
    if (a.ranker) stats.push(`ranker: ${a.ranker}`);
    if (a.stop_reason) stats.push(`stop: ${a.stop_reason}`);
    if (num(a.elapsed_ms)) stats.push(`${(num(a.elapsed_ms) / 1000).toFixed(1)}s`);
    if (stats.length) sec.appendChild(el("div", "qp-stats", stats.join(" · ")));

    if (num(a.hyde_failures) > 0) {
        sec.appendChild(el(
            "div",
            "qp-warn",
            `⚠ HyDE generation failed on ${a.hyde_failures} round${plural(num(a.hyde_failures))} — ` +
                `fell back to plain literal-query retrieval (the answer-space HyDE advantage was forfeited).`,
        ));
    }

    if (a.hyde_rounds && a.hyde_rounds.length) {
        a.hyde_rounds.forEach((r, i) => sec.appendChild(hydeRoundEl(r, i + 1)));
    } else if (a.tried_hyde && a.tried_hyde.length) {
        // Older deployments carry only the hypothetical texts, not the full variant.
        a.tried_hyde.forEach((t, i) => sec.appendChild(hydeRoundEl({ hypothetical_answer: t }, i + 1)));
    }
    return sec;
}

function drSection(d: DeepResearchTelemetry): HTMLElement {
    const sec = el("div", "qp-section");
    sec.appendChild(el("div", "qp-section-title", "Deep research"));
    const stats: string[] = [];
    if (d.decomposed != null) stats.push(d.decomposed ? "decomposed" : "unified");
    if (num(d.topic_count)) stats.push(`${d.topic_count} topic${plural(num(d.topic_count))}`);
    if (num(d.iteration_count)) stats.push(`${d.iteration_count} iteration${plural(num(d.iteration_count))}`);
    if (num(d.node_count)) stats.push(`${d.node_count} node${plural(num(d.node_count))}`);
    if (num(d.llm_call_count)) stats.push(`${d.llm_call_count} LLM call${plural(num(d.llm_call_count))}`);
    if (d.budget_exhausted) {
        stats.push(`⚠ budget exhausted${d.budget_exhausted_reason ? ` (${d.budget_exhausted_reason})` : ""}`);
    }
    if (num(d.elapsed_ms)) stats.push(`${(num(d.elapsed_ms) / 1000).toFixed(1)}s`);
    if (stats.length) sec.appendChild(el("div", "qp-stats", stats.join(" · ")));
    return sec;
}

/** Concise one-line summary for the collapsed <summary>. */
function summarize(
    agentic: AgenticTelemetry | undefined,
    hasAgentic: boolean,
    dr: DeepResearchTelemetry | undefined,
    hasDr: boolean,
    showRewrite: boolean,
): string {
    const parts: string[] = [];
    if (hasAgentic && agentic) {
        const rounds = num(agentic.rounds_run) || (agentic.hyde_rounds?.length ?? 0);
        if (rounds) parts.push(`HyDE · ${rounds} round${plural(rounds)}`);
        else parts.push("HyDE");
        if (num(agentic.hyde_failures) > 0) parts.push(`⚠ ${agentic.hyde_failures} fallback${plural(num(agentic.hyde_failures))}`);
    }
    if (hasDr && dr) parts.push(`Deep research · ${num(dr.topic_count)} topic${plural(num(dr.topic_count))}`);
    if (showRewrite && !parts.length) parts.push("query rewritten");
    return "🔎 Query processing" + (parts.length ? " · " + parts.join(" · ") : "");
}

/**
 * Build/refresh the query-processing panel inside a turn's `.bubble-wrap`,
 * positioned ABOVE the reasoning block (or before the answer bubble if none yet),
 * so top-to-bottom reads: query-processing → thinking → answer.
 *
 * Renders nothing (and removes any prior panel) when there is no processing worth
 * showing — a plain literal-query retrieval leaves the turn uncluttered.
 */
export function renderQueryProcessing(
    bubbleWrap: HTMLElement | null | undefined,
    data: StreamEventData,
    rawQuery: string,
): void {
    if (!bubbleWrap) return;
    bubbleWrap.querySelector(".query-processing-block")?.remove();

    const agentic = data.metadata?.agentic_retrieval;
    const dr = data.metadata?.deep_research;
    const processed = (data.processed_query || "").trim();
    const raw = (rawQuery || "").trim();
    const showRewrite = !!processed && processed.toLowerCase() !== raw.toLowerCase();
    const kg = Array.isArray(data.kg_expanded_terms)
        ? data.kg_expanded_terms.map((t) => (t || "").trim()).filter(Boolean)
        : [];
    const hasAgentic = !!agentic &&
        (num(agentic.rounds_run) > 0 ||
            (agentic.hyde_rounds?.length ?? 0) > 0 ||
            (agentic.tried_hyde?.length ?? 0) > 0);
    const hasDr = !!dr && (dr.decomposed === true || num(dr.topic_count) > 0);

    if (!showRewrite && !hasAgentic && !hasDr && !kg.length) return;

    const details = document.createElement("details");
    details.className = "query-processing-block";
    details.open = true;
    const summary = el("summary", "qp-summary", summarize(agentic, hasAgentic, dr, hasDr, showRewrite));
    details.appendChild(summary);
    const body = el("div", "qp-body");
    details.appendChild(body);

    if (showRewrite) {
        const sec = el("div", "qp-section");
        sec.appendChild(el("div", "qp-section-title", "Query rewrite"));
        sec.appendChild(el("div", "qp-hyp", processed)); // rewritten retrieval query
        if (num(data.query_confidence)) {
            sec.appendChild(el("div", "qp-terms", `Confidence: ${num(data.query_confidence).toFixed(2)}`));
        }
        body.appendChild(sec);
    }
    if (hasAgentic && agentic) body.appendChild(hydeSection(agentic));
    if (hasDr && dr) body.appendChild(drSection(dr));
    if (kg.length) {
        const sec = el("div", "qp-section");
        sec.appendChild(el("div", "qp-section-title", "KG-expanded terms"));
        sec.appendChild(el("div", "qp-terms", kg.join(", ")));
        body.appendChild(sec);
    }

    // Anchor: before the reasoning block if it exists yet, else before the answer
    // bubble; a null anchor appends (defensive — .bubble always exists in practice).
    const anchor = bubbleWrap.querySelector(".reasoning-block") ?? bubbleWrap.querySelector(".bubble");
    bubbleWrap.insertBefore(details, anchor);
}
