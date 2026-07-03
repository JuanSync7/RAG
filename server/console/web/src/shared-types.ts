/** @summary
 * Canonical shared TypeScript type definitions used by both the operator
 * console (`types.ts`) and the user console (`user-types.ts`).
 *
 * Anything truly shared between both consoles lives here; console-specific
 * shapes (notably `ConversationTurn`, `ContextBreakdown`, `StreamEventData`,
 * `ChunkResult` / `QueryResult`) remain in their respective files because the
 * two consoles model those domains differently. The turn-loop SSE event
 * payloads (TURN_LOOP_DESIGN.md §8) live here because both consoles render
 * the same activity log from them (see `activityLog.ts`).
 *
 * Exports: ConversationMeta, SlashCommand, CommandResult,
 *   TurnActionEventData, HydeQueryEventData, RetrieveResultTopDoc,
 *   RetrieveResultEventData, JudgeVerdictEventData, DeepStudyEventData,
 *   LlmCallEventData, DraftEventData, GateEventData, ClarifyEventData,
 *   TurnLoopEventDataMap, TurnLoopEventName, TurnLoopStreamEventFields
 * Deps: none
 * @end-summary
 */

/** Conversation metadata as returned by `/console/conversations`. */
export interface ConversationMeta {
    conversation_id: string;
    title?: string;
    updated_at_ms?: number;
    message_count?: number;
}

/**
 * Slash-command descriptor exposed to both consoles.
 *
 * `category` is used by the user console for grouping in the picker;
 * `intent` is used by the operator console for command routing.
 * Both are optional so each surface can populate what it needs.
 */
export interface SlashCommand {
    name: string;
    description: string;
    args_hint?: string;
    category?: string;
    intent?: string;
}

/** Result envelope returned after executing a slash command. */
export interface CommandResult {
    intent?: string;
    action?: string;
    message?: string;
    data?: Record<string, unknown>;
}

// ── Turn-loop stream events (TURN_LOOP_DESIGN.md §8) ─────────────────────────
//
// One interface per typed SSE event emitted by the turn-level agentic
// conversation loop. Every field is optional because SSE payloads are
// untrusted wire data — renderers must tolerate absent/miscoerced fields.
// Names are 1:1 with `TurnEventType` in
// `src/retrieval/pipeline/turn_loop/schemas.py` and the OTel span vocabulary.

/** `turn_action` — one controller decision (which action, and why). */
export interface TurnActionEventData {
    index?: number;
    action?: string;
    reason?: string;
    confidence?: number;
}

/** `hyde_query` — the hypothetical answer used to steer a retrieval round. */
export interface HydeQueryEventData {
    round?: number;
    hypothetical_answer?: string;
    search_terms?: string[];
    target_aspect?: string | null;
}

/** One top-ranked document entry inside a `retrieve_result` payload. */
export interface RetrieveResultTopDoc {
    doc?: string;
    heading?: string;
    score?: number;
}

/** `retrieve_result` — outcome of one retrieve_ranked round (pool growth). */
export interface RetrieveResultEventData {
    round?: number;
    added?: number;
    dup?: number;
    pool_size?: number;
    top?: RetrieveResultTopDoc[];
}

/** `judge_verdict` — evidence-judge outcome for one retrieval round. */
export interface JudgeVerdictEventData {
    round?: number;
    kept?: number;
    sufficient?: boolean;
    confidence?: number;
    /** Free-text gap description from the judge (may be empty). */
    missing_information?: string;
}

/** `deep_study` — progress reading one full source document window-by-window. */
export interface DeepStudyEventData {
    document_id?: string;
    title?: string;
    window?: number;
    of_windows?: number;
    notes_preview?: string;
}

/** `llm_call` — telemetry one-liner for a single loop LLM call. */
export interface LlmCallEventData {
    alias?: string;
    purpose?: string;
    ms?: number;
    prompt_tokens?: number;
    completion_tokens?: number;
}

/** `draft` — a live token delta of an ANSWER draft attempt. */
export interface DraftEventData {
    attempt?: number;
    /** `content` (answer text) or `reasoning` (chain-of-thought delta). */
    kind?: string;
    text_delta?: string;
}

/** `gate` — answer-confidence gate verdict for one draft attempt. */
export interface GateEventData {
    attempt?: number;
    score?: number;
    threshold?: number;
    passed?: boolean;
    weakest?: string;
}

/** `clarify` — terminal clarification with clickable hint/scoping chips. */
export interface ClarifyEventData {
    question?: string;
    hints?: string[];
    scoping_questions?: string[];
}

/** Event-name → payload map for the nine turn-loop SSE events. */
export interface TurnLoopEventDataMap {
    turn_action: TurnActionEventData;
    hyde_query: HydeQueryEventData;
    retrieve_result: RetrieveResultEventData;
    judge_verdict: JudgeVerdictEventData;
    deep_study: DeepStudyEventData;
    llm_call: LlmCallEventData;
    draft: DraftEventData;
    gate: GateEventData;
    clarify: ClarifyEventData;
}

/** Union of the nine turn-loop SSE event names. */
export type TurnLoopEventName = keyof TurnLoopEventDataMap;

/**
 * Intersection of all turn-loop payloads: each console's `StreamEventData`
 * extends this so the raw SSE data object is typed for whichever event it
 * carries (every field is optional, so the intersection is conflict-free).
 */
export type TurnLoopStreamEventFields = TurnActionEventData &
    HydeQueryEventData &
    RetrieveResultEventData &
    JudgeVerdictEventData &
    DeepStudyEventData &
    LlmCallEventData &
    DraftEventData &
    GateEventData &
    ClarifyEventData;
