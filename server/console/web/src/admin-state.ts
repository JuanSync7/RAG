/**
 * @summary
 * Module-level mutable state for the admin/operator console (active conversation,
 * cached conversation list, context-menu target, and slash-command cache). Exposed
 * as get/set helpers so other modules avoid sharing raw bindings.
 * Exports: getActiveConversationId, setActiveConversationIdRaw,
 *   getConversationCache, setConversationCache,
 *   getConvContextMenuTarget, setConvContextMenuTarget,
 *   getConvMenuCloseHandler, setConvMenuCloseHandler,
 *   getConvMenuEscapeHandler, setConvMenuEscapeHandler,
 *   getCommandCache
 * Deps: admin-types
 * @end-summary
 */

import type { ConsoleCommandSpec, ConversationMeta } from "./admin-types.js";

let activeConversationId: string | null = null;
let conversationCache: ConversationMeta[] = [];
let convContextMenuTarget: string | null = null;
let convMenuCloseHandler: ((e: MouseEvent) => void) | null = null;
let convMenuEscapeHandler: ((e: KeyboardEvent) => void) | null = null;
const commandCache: Record<string, ConsoleCommandSpec[]> = {};

export function getActiveConversationId(): string | null {
    return activeConversationId;
}

export function setActiveConversationIdRaw(id: string | null): void {
    activeConversationId = id;
}

export function getConversationCache(): ConversationMeta[] {
    return conversationCache;
}

export function setConversationCache(items: ConversationMeta[]): void {
    conversationCache = items;
}

export function getConvContextMenuTarget(): string | null {
    return convContextMenuTarget;
}

export function setConvContextMenuTarget(id: string | null): void {
    convContextMenuTarget = id;
}

export function getConvMenuCloseHandler(): ((e: MouseEvent) => void) | null {
    return convMenuCloseHandler;
}

export function setConvMenuCloseHandler(handler: ((e: MouseEvent) => void) | null): void {
    convMenuCloseHandler = handler;
}

export function getConvMenuEscapeHandler(): ((e: KeyboardEvent) => void) | null {
    return convMenuEscapeHandler;
}

export function setConvMenuEscapeHandler(handler: ((e: KeyboardEvent) => void) | null): void {
    convMenuEscapeHandler = handler;
}

export function getCommandCache(): Record<string, ConsoleCommandSpec[]> {
    return commandCache;
}
