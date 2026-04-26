// @summary
// User-console-specific type definitions.
//
// Shapes that are also useful to the operator console live in `./shared-types`
// and are re-exported from here so existing user-console call-sites keep their
// import path unchanged.
// @end-summary

export type { ConversationMeta, SlashCommand, CommandResult } from "./shared-types";

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

export function sourceRefToChunkResult(ref: SourceRef): ChunkResult {
    return {
        text: ref.text ?? "",
        score: ref.score ?? 0,
        metadata: {
            source: ref.source ?? "",
            source_uri: ref.source_uri ?? "",
            section: ref.section ?? "",
            original_char_start: ref.original_char_start,
            original_char_end: ref.original_char_end,
        },
    };
}
