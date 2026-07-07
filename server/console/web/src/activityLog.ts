// @summary
// Shared turn-loop ACTIVITY LOG renderer used by BOTH consoles (user console
// `streaming.ts`, operator console `admin-query.ts`): a lazy collapsible
// <details> block that renders the nine typed turn-loop SSE events
// (TURN_LOOP_DESIGN.md §8) as compact lines — controller decisions, HyDE
// hypotheticals (verbatim), retrieval/judge outcomes, deep-study progress,
// dim LLM-call telemetry, live draft attempts (replaced per attempt), and
// gate verdicts. Clarify chips are built by `renderClarifyChips` and live
// OUTSIDE the collapsible log so they stay clickable after it collapses;
// clicking a chip resubmits the chip text as the next user query.
// XSS discipline: every event-derived string is written via `textContent`
// (never innerHTML) because these strings carry LLM/document content.
// Exports: TURN_LOOP_EVENT_NAMES, isTurnLoopEvent, createActivityLog,
//          renderClarifyChips, ActivityLog, ResubmitFn
// Deps: shared-types (types only)
// @end-summary

import type {
    ClarifyEventData,
    DeepStudyEventData,
    DraftEventData,
    GateEventData,
    HydeQueryEventData,
    JudgeVerdictEventData,
    LlmCallEventData,
    RetrieveResultEventData,
    TurnActionEventData,
    TurnLoopEventName,
} from "./shared-types";

/** Callback used by clarify chips to resubmit the chip text as a new query. */
export type ResubmitFn = (text: string) => void | Promise<void>;

/**
 * The nine turn-loop SSE event names (1:1 with `TurnEventType` in
 * `src/retrieval/pipeline/turn_loop/schemas.py` and the OTel span vocabulary).
 */
export const TURN_LOOP_EVENT_NAMES: ReadonlySet<string> = new Set<TurnLoopEventName>([
    "turn_action",
    "hyde_query",
    "retrieve_result",
    "judge_verdict",
    "deep_study",
    "llm_call",
    "draft",
    "gate",
    "clarify",
]);

/** True iff `eventType` is one of the nine turn-loop activity events. */
export function isTurnLoopEvent(eventType: string): eventType is TurnLoopEventName {
    return TURN_LOOP_EVENT_NAMES.has(eventType);
}

/** A HyDE hypothetical longer than this starts collapsed (still expandable). */
const HYDE_OPEN_MAX_CHARS = 280;

// ── Tolerant wire-value coercion (SSE payloads are untrusted) ────────────────

function asStr(value: unknown): string {
    if (value === null || value === undefined) return "";
    return typeof value === "string" ? value : String(value);
}

function asNum(value: unknown): number | null {
    const n = typeof value === "number" ? value : Number(value);
    return Number.isFinite(n) ? n : null;
}

function asStrList(value: unknown): string[] {
    if (!Array.isArray(value)) return [];
    return value.map(asStr).filter((s) => s.trim().length > 0);
}

/** `0.7234` → `"0.72"`; null-safe (`"?"` when absent/miscoded). */
function fmtScore(value: unknown): string {
    const n = asNum(value);
    return n === null ? "?" : n.toFixed(2);
}

// ── DOM helpers (textContent-only writes) ────────────────────────────────────

function div(className: string, text?: string): HTMLDivElement {
    const el = document.createElement("div");
    el.className = className;
    if (text !== undefined) el.textContent = text;
    return el;
}

/** The rendered activity-log handle returned by `createActivityLog`. */
export interface ActivityLog {
    /** The `<details class="activity-log">` element to insert into the DOM. */
    root: HTMLDetailsElement;
    /** Render one turn-loop SSE event (`type` must be a turn-loop name). */
    handle(type: string, data: Record<string, unknown>): void;
    /** Relabel the summary with step count + elapsed; optionally collapse. */
    finalize(collapse: boolean): void;
    /** Number of events rendered so far. */
    eventCount(): number;
}

/**
 * Build a lazy collapsible activity-log block (the reasoning-block pattern:
 * a `<details>` with a relabelable `<summary>`, open while streaming, meant
 * to be collapsed by the caller when the first real answer token arrives).
 *
 * The caller inserts `root` into the bubble wrap BEFORE the reasoning block
 * and feeds every turn-loop SSE event through `handle`. Clarify chips are a
 * separate concern — see `renderClarifyChips` (they must survive collapse).
 */
export function createActivityLog(): ActivityLog {
    const root = document.createElement("details");
    root.className = "activity-log";
    root.open = true;

    const summary = document.createElement("summary");
    summary.className = "activity-summary";
    summary.textContent = "Activity…";
    root.appendChild(summary);

    const body = div("activity-body");
    root.appendChild(body);

    const startedAt = performance.now();
    let events = 0;

    // Draft attempts stream token-by-token; the draft area is REPLACED when a
    // new attempt begins (the user watches drafts being made and discarded).
    let draftAttempt: number | null = null;
    let draftText = "";
    let draftLabelEl: HTMLElement | null = null;
    let draftTextEl: HTMLElement | null = null;

    const addLine = (className: string, text: string): HTMLDivElement => {
        const el = div(className, text);
        body.appendChild(el);
        return el;
    };

    const renderTurnAction = (data: TurnActionEventData): void => {
        const index = asNum(data.index);
        const action = asStr(data.action).trim() || "?";
        const reason = asStr(data.reason).trim();
        const conf = asNum(data.confidence);
        // "#2 RETRIEVE — controller reason" (index shown verbatim as emitted).
        let text = `${index === null ? "#?" : `#${index}`} ${action}`;
        if (reason) text += ` — ${reason}`;
        if (conf !== null) text += ` · conf ${fmtScore(conf)}`;
        addLine("activity-line activity-action", text);
    };

    const renderHydeQuery = (data: HydeQueryEventData): void => {
        const hypothetical = asStr(data.hypothetical_answer);
        const round = asNum(data.round);
        const aspect = asStr(data.target_aspect).trim();
        // The HyDE hypothetical is shown VERBATIM (this is the visibility the
        // loop promises); long ones start collapsed but stay expandable.
        const block = document.createElement("details");
        block.className = "activity-hyde";
        block.open = hypothetical.length <= HYDE_OPEN_MAX_CHARS;
        const label = document.createElement("summary");
        label.className = "activity-hyde-summary";
        label.textContent =
            `HyDE hypothetical${round === null ? "" : ` — round ${round}`}` +
            (aspect ? ` · aspect: ${aspect}` : "");
        block.appendChild(label);
        block.appendChild(div("activity-hyde-text", hypothetical));
        const terms = asStrList(data.search_terms);
        if (terms.length) {
            block.appendChild(div("activity-line dim", `search terms: ${terms.join(", ")}`));
        }
        body.appendChild(block);
    };

    const renderRetrieveResult = (data: RetrieveResultEventData): void => {
        const added = asNum(data.added) ?? 0;
        const dup = asNum(data.dup);
        const pool = asNum(data.pool_size);
        // "+5 new (pool 12): top doc › heading"
        let text = `+${added} new`;
        if (pool !== null) text += ` (pool ${pool})`;
        if (dup !== null && dup > 0) text += `, ${dup} dup`;
        const top = Array.isArray(data.top) ? data.top[0] : undefined;
        if (top && typeof top === "object") {
            const doc = asStr(top.doc).trim();
            const heading = asStr(top.heading).trim();
            if (doc || heading) {
                text += `: ${doc}${doc && heading ? " › " : ""}${heading}`;
            }
        }
        addLine("activity-line activity-retrieve", text);
    };

    const renderJudgeVerdict = (data: JudgeVerdictEventData): void => {
        const kept = asNum(data.kept);
        const conf = asNum(data.confidence);
        const sufficient = data.sufficient === true;
        let text = `judge: kept ${kept === null ? "?" : kept}`;
        if (conf !== null) text += ` · confidence ${fmtScore(conf)}`;
        text += ` · ${sufficient ? "sufficient" : "insufficient"}`;
        addLine("activity-line activity-judge", text);
        // Free text on the wire (JudgeVerdictEvent); a legacy list is joined.
        const rawMissing: unknown = data.missing_information;
        const missing = Array.isArray(rawMissing)
            ? asStrList(rawMissing).join("; ")
            : asStr(rawMissing).trim();
        if (missing) {
            addLine("activity-line dim", `missing: ${missing}`);
        }
    };

    const renderDeepStudy = (data: DeepStudyEventData): void => {
        const title = asStr(data.title).trim() || asStr(data.document_id).trim() || "document";
        const win = asNum(data.window);
        const of = asNum(data.of_windows);
        // "reading <title> window 2/8" (window shown verbatim as emitted).
        let text = `reading ${title}`;
        if (win !== null) text += ` — window ${win}${of === null ? "" : `/${of}`}`;
        addLine("activity-line activity-study", text);
        const notes = asStr(data.notes_preview).trim();
        if (notes) addLine("activity-line dim", notes);
    };

    const renderLlmCall = (data: LlmCallEventData): void => {
        const alias = asStr(data.alias).trim() || "llm";
        const purpose = asStr(data.purpose).trim();
        const ms = asNum(data.ms);
        const tokens =
            (asNum(data.prompt_tokens) ?? 0) + (asNum(data.completion_tokens) ?? 0);
        // Dim one-liner: "controller 1.2s 850tok" (purpose shown when distinct).
        let text = alias;
        if (purpose && purpose !== alias) text += ` · ${purpose}`;
        if (ms !== null) text += ` ${(ms / 1000).toFixed(1)}s`;
        if (tokens > 0) text += ` ${tokens}tok`;
        addLine("activity-line activity-llm dim", text);
    };

    const renderDraft = (data: DraftEventData): void => {
        const attempt = asNum(data.attempt);
        if (!draftTextEl || (attempt !== null && attempt !== draftAttempt)) {
            // New attempt: (re)create or reset the nested draft area — each
            // attempt REPLACES the previous draft's text.
            draftAttempt = attempt;
            draftText = "";
            if (!draftTextEl) {
                const wrap = div("activity-draft");
                draftLabelEl = div("activity-draft-label");
                draftTextEl = div("activity-draft-text");
                wrap.appendChild(draftLabelEl);
                wrap.appendChild(draftTextEl);
                body.appendChild(wrap);
            }
        }
        if (draftLabelEl) {
            draftLabelEl.textContent =
                draftAttempt === null ? "Draft" : `Draft — attempt ${draftAttempt}`;
        }
        draftText += asStr(data.text_delta);
        draftTextEl.textContent = draftText;
    };

    const renderGate = (data: GateEventData): void => {
        const attempt = asNum(data.attempt);
        const passed = data.passed === true;
        const weakest = asStr(data.weakest).trim();
        let text = `gate${attempt === null ? "" : ` — attempt ${attempt}`}: `;
        text += passed ? "PASS" : "FAIL";
        text += ` (${fmtScore(data.score)} vs threshold ${fmtScore(data.threshold)})`;
        if (weakest) text += ` · weakest: ${weakest}`;
        addLine(`activity-line activity-gate ${passed ? "pass" : "fail"}`, text);
    };

    const renderClarify = (data: ClarifyEventData): void => {
        const question = asStr(data.question).trim();
        addLine("activity-line activity-clarify", `clarify${question ? `: ${question}` : ""}`);
        // Chips are rendered OUTSIDE the log by the caller (renderClarifyChips)
        // so they remain visible/clickable when the log is collapsed.
    };

    const handle = (type: string, data: Record<string, unknown>): void => {
        if (!isTurnLoopEvent(type)) return;
        events += 1;
        if (type === "turn_action") renderTurnAction(data as TurnActionEventData);
        else if (type === "hyde_query") renderHydeQuery(data as HydeQueryEventData);
        else if (type === "retrieve_result") renderRetrieveResult(data as RetrieveResultEventData);
        else if (type === "judge_verdict") renderJudgeVerdict(data as JudgeVerdictEventData);
        else if (type === "deep_study") renderDeepStudy(data as DeepStudyEventData);
        else if (type === "llm_call") renderLlmCall(data as LlmCallEventData);
        else if (type === "draft") renderDraft(data as DraftEventData);
        else if (type === "gate") renderGate(data as GateEventData);
        else renderClarify(data as ClarifyEventData);
    };

    const finalize = (collapse: boolean): void => {
        const secs = (performance.now() - startedAt) / 1000;
        summary.textContent =
            `Activity — ${events} step${events === 1 ? "" : "s"}` +
            (secs > 0.05 ? ` (${secs.toFixed(1)}s)` : "");
        if (collapse) root.open = false;
    };

    return { root, handle, finalize, eventCount: () => events };
}

/**
 * Build the clickable clarify chips for a `clarify` event: the question as a
 * text line, then one chip per hint and per scoping question. Clicking a chip
 * resubmits the chip text as the next user query through `onResubmit` (the
 * same registered-resubmit mechanism as the deep-research suggestion chip).
 *
 * Returns `null` when the payload carries nothing renderable. All strings are
 * written via `textContent` — they are LLM-authored content.
 */
export function renderClarifyChips(
    data: Record<string, unknown>,
    onResubmit: ResubmitFn,
): HTMLElement | null {
    const payload = data as ClarifyEventData;
    const question = asStr(payload.question).trim();
    const hints = asStrList(payload.hints);
    const scoping = asStrList(payload.scoping_questions);
    if (!question && !hints.length && !scoping.length) return null;

    const wrap = div("clarify-chips");
    if (question) wrap.appendChild(div("clarify-question", question));

    const addChip = (text: string, extraClass: string): void => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = `clarify-chip ${extraClass}`.trim();
        chip.textContent = text;
        chip.title = "Send this as your reply";
        chip.addEventListener("click", () => {
            void onResubmit(text);
        });
        wrap.appendChild(chip);
    };

    hints.forEach((hint) => addChip(hint, "clarify-chip-hint"));
    scoping.forEach((q) => addChip(q, "clarify-chip-scope"));
    return wrap;
}
