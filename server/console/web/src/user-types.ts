// @summary
// User-console-specific type definitions.
// Cross-console shapes (`ConversationMeta`, `SlashCommand`, and the nine
// turn-loop SSE event payloads) are re-exported from `./shared-types` so both
// consoles share one contract; `StreamEventData` extends the turn-loop fields
// so event handlers get typed payload access (TURN_LOOP_DESIGN.md §8).
// @end-summary

export { ConversationMeta, SlashCommand } from "./shared-types";
export type {
    TurnActionEventData,
    HydeQueryEventData,
    RetrieveResultTopDoc,
    RetrieveResultEventData,
    JudgeVerdictEventData,
    DeepStudyEventData,
    LlmCallEventData,
    DraftEventData,
    GateEventData,
    ClarifyEventData,
    TurnLoopEventDataMap,
    TurnLoopEventName,
    TurnLoopStreamEventFields,
} from "./shared-types";
import type { TurnLoopStreamEventFields } from "./shared-types";

export type ThemeValue = "dark" | "light" | "system";

export interface PresetConfig {
    searchLimit: number;
    rerankTopK: number;
}

export interface ContextBreakdown {
    system: number;
    memory: number;
    chunks: number;
    query: number;
}

export interface ChunkResult {
    text: string;
    score: number;
    metadata: Record<string, unknown>;
}

export interface SourceRef {
    source?: string;
    source_uri?: string;
    source_key?: string;
    document_id?: string;
    section?: string;
    score?: number;
    text?: string;
    original_char_start?: number;
    original_char_end?: number;
}

export interface TokenBudget {
    usage_percent?: number;
    breakdown?: Record<string, unknown>;
    input_tokens?: number;
    context_length?: number;
    output_reservation?: number;
    model_name?: string;
    actual_prompt_tokens?: number;
    actual_completion_tokens?: number;
    actual_total_tokens?: number;
    cost_usd?: number;
}

export interface LastTurnStats {
    promptTokens: number;
    completionTokens: number;
    tokensPerSecond: number;
    costUsd: number;
}

// -- Query-processing telemetry (surfaced in the "Query processing" panel) --
// Mirrors metadata['agentic_retrieval'] / metadata['deep_research'] from the
// backend. All fields optional: older deployments omit hyde_rounds/tried_hyde,
// and the linear (non-agentic, non-DR) path emits no telemetry at all.

export interface HydeRound {
    round?: number;
    hypothetical_answer?: string;
    search_terms?: string[];
    target_aspect?: string;
    fell_back?: boolean;
}

export interface AgenticTelemetry {
    rounds_run?: number;
    hyde_variants_tried?: number;
    hyde_failures?: number;
    kept_count?: number;
    llm_calls?: number;
    judge_calls?: number;
    ranker?: string;
    ranker_calls?: number;
    backfilled?: number;
    stop_reason?: string;
    elapsed_ms?: number;
    tried_hyde?: string[];
    hyde_rounds?: HydeRound[];
}

export interface DeepResearchTelemetry {
    decomposed?: boolean;
    topic_count?: number;
    iteration_count?: number;
    node_count?: number;
    llm_call_count?: number;
    is_unified?: boolean;
    budget_exhausted?: boolean;
    budget_exhausted_reason?: string;
    elapsed_ms?: number;
}

export interface QueryMetadata {
    agentic_retrieval?: AgenticTelemetry;
    deep_research?: DeepResearchTelemetry;
    [key: string]: unknown;
}

/**
 * Data object of one SSE frame. Extends the turn-loop payload fields
 * (TURN_LOOP_DESIGN.md §8) so handlers for `turn_action` / `hyde_query` /
 * `retrieve_result` / `judge_verdict` / `deep_study` / `llm_call` / `draft` /
 * `gate` / `clarify` events read typed fields instead of index-signature
 * lookups. Which fields are populated depends on the frame's `event:` name.
 */
export interface StreamEventData extends TurnLoopStreamEventFields {
    token?: string;
    text?: string;
    message?: string;
    results?: ChunkResult[];
    conversation_id?: string;
    token_budget?: TokenBudget;
    summary?: string;
    processed_query?: string;
    query_confidence?: number;
    kg_expanded_terms?: string[] | null;
    metadata?: QueryMetadata;
    [key: string]: unknown;
}

export interface CommandResult {
    action?: string;
    message?: string;
    data?: Record<string, unknown>;
}

// -- Retrieval tab --

export interface RetrievalResultItem {
    score: number;
    text: string;
    metadata: {
        source_name?: string;
        source?: string;
        source_uri?: string;
        chunk_id?: string;
        doc_id?: string;
        document_id?: string;
        [key: string]: unknown;
    };
}

export interface RetrievalResponse {
    results: RetrievalResultItem[];
    relevant_doc_ids: string[];
    ignored_doc_ids: string[];
    latency_ms: number;
    conversation_id?: string;
}

export interface DocStateResponse {
    conversation_id: string;
    relevant_doc_ids: string[];
    ignored_doc_ids: string[];
}

export function sourceRefToChunkResult(ref: SourceRef): ChunkResult {
    return {
        text: ref.text ?? "",
        score: ref.score ?? 0,
        metadata: {
            source: ref.source ?? "",
            source_uri: ref.source_uri ?? "",
            source_key: ref.source_key ?? "",
            document_id: ref.document_id ?? "",
            section: ref.section ?? "",
            original_char_start: ref.original_char_start,
            original_char_end: ref.original_char_end,
        },
    };
}
