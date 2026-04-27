/** @summary
 * Canonical shared TypeScript type definitions used by both the operator
 * console (`types.ts`) and the user console (`user-types.ts`).
 *
 * Anything truly shared between both consoles lives here; console-specific
 * shapes (notably `ConversationTurn`, `ContextBreakdown`, `StreamEventData`,
 * `ChunkResult` / `QueryResult`) remain in their respective files because the
 * two consoles model those domains differently.
 *
 * Exports: ConversationMeta, SlashCommand, CommandResult
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
