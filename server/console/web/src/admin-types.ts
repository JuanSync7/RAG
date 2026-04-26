/**
 * @summary
 * Shared TypeScript type aliases used across the admin/operator console modules.
 * Exports: JsonObject, QueryResult, StreamEventData, TokenBudgetPayload, TimingPayload,
 *   ConsoleCommandSpec, CommandExecution, ConversationMeta, ConversationTurn,
 *   MarkedLike, DomPurifyLike
 * Deps: none
 * @end-summary
 */

export type JsonObject = Record<string, unknown>;

export type QueryResult = {
    score?: number;
    text?: string;
    metadata?: Record<string, unknown>;
};

export type StreamEventData = {
    token?: string;
    message?: string;
    results?: QueryResult[];
    [key: string]: unknown;
};

export type TokenBudgetPayload = {
    input_tokens?: number;
    context_length?: number;
    usage_percent?: number;
    model_name?: string;
    breakdown?: Record<string, number>;
    actual_prompt_tokens?: number;
    actual_completion_tokens?: number;
    cost_usd?: number;
};

export type TimingPayload = {
    latency_ms?: number;
    retrieval_ms?: number;
    generation_ms?: number;
    token_count?: number;
    stage_timings?: Array<Record<string, unknown>>;
    timing_totals?: Record<string, unknown>;
    token_budget?: TokenBudgetPayload;
};

export type ConsoleCommandSpec = {
    name: string;
    description: string;
    args_hint?: string;
    intent?: string;
};

export type CommandExecution = {
    mode?: string;
    command?: string;
    intent?: string;
    action?: string;
    message?: string;
    data?: JsonObject;
};

export type ConversationMeta = {
    conversation_id: string;
    title?: string;
    updated_at_ms?: number;
    message_count?: number;
};

export type ConversationTurn = {
    role: string;
    content: string;
    timestamp_ms?: number;
};

export type MarkedLike = {
    parse: (markdown: string) => string;
};

export type DomPurifyLike = {
    sanitize: (dirty: string) => string;
};

declare global {
    interface Window {
        marked?: MarkedLike;
        DOMPurify?: DomPurifyLike;
    }
}
