// @summary
// TypeScript type definitions for the operator console.
//
// Cross-console shapes (ConversationMeta, SlashCommand, CommandResult) live
// in `./shared-types` and are re-exported here so existing import sites
// continue to work.
//
// Exports: ConsoleEnvelope, MessageRole, CitationSource, ChatMessage, ConversationMeta,
//          ConversationTurn, QueryParams, QueryResult, StreamEventData, ContextBreakdown,
//          SlashCommand, CommandResult, ContextAttachment, UserSettings, HealthSummary, SidebarNavItem
// Deps: ./shared-types
// @end-summary

export type { ConversationMeta, SlashCommand, CommandResult } from "./shared-types";

/** Standard console API response envelope. */
export type ConsoleEnvelope<T = Record<string, unknown>> = {
  ok: boolean;
  request_id?: string;
  data?: T;
  error?: { code: string; message: string; details?: string };
};

// -- Messages --

export type MessageRole = "user" | "assistant";

export type CitationSource = {
  filename: string;
  section: string;
  score: number;
  chunk_text: string;
  source_uri?: string;
};

export type ChatMessage = {
  id: string;
  role: MessageRole;
  content: string;
  citations?: CitationSource[];
  timestamp_ms: number;
  is_streaming?: boolean;
};

// -- Conversations --

export type ConversationTurn = {
  role: MessageRole;
  content: string;
  timestamp_ms?: number;
};

// -- Query --

export type QueryParams = {
  query: string;
  search_limit?: number;
  rerank_top_k?: number;
  stream?: boolean;
  conversation_id?: string;
  memory_enabled?: boolean;
  context_attachments?: ContextAttachment[];
};

export type QueryResult = {
  score?: number;
  text?: string;
  metadata?: Record<string, unknown>;
};

export type StreamEventData = {
  token?: string;
  message?: string;
  results?: QueryResult[];
  context_usage_pct?: number;
  context_breakdown?: ContextBreakdown;
};

export type ContextBreakdown = {
  memory_tokens?: number;
  chunk_tokens?: number;
  query_tokens?: number;
  system_tokens?: number;
};

// -- Context Attachments --

export type ContextAttachment = {
  type: "file" | "web" | "kb";
  label: string;
  data?: string;
  uri?: string;
};

// -- Settings --

export type UserSettings = {
  theme: "dark" | "light" | "system";
  preset: string;
  search_limit: number;
  rerank_top_k: number;
  streaming: boolean;
  show_citations: boolean;
  api_endpoint?: string;
  auth_token?: string;
};

// -- Health --

export type HealthSummary = {
  status: string;
  temporal_connected?: boolean;
  worker_available?: boolean;
  ollama_reachable?: boolean;
};

// -- Sidebar --

export type SidebarNavItem = "conversations" | "projects" | "search" | "customize";
