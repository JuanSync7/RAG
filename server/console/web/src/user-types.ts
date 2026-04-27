// @summary
// User-console-specific type definitions.
// Cross-console shapes (`ConversationMeta`, `SlashCommand`) are re-exported
// from `./shared-types` so both consoles share one contract.
// @end-summary

export { ConversationMeta, SlashCommand } from "./shared-types";

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

export interface StreamEventData {
    token?: string;
    message?: string;
    results?: ChunkResult[];
    conversation_id?: string;
    token_budget?: TokenBudget;
    summary?: string;
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
