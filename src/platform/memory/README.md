<!-- @summary
Tenant-aware conversation memory: Redis canonical backend + no-op fallback. Manages sliding-window turns, rolling summaries, and the turn-loop cross-turn context (chunk refs, docs-studied ledger, pending clarifications) for multi-turn RAG queries.
@end-summary -->

# platform/memory

## Overview

This package provides persistent conversation memory for multi-turn RAG queries. Memory is tenant-scoped, stored in Redis, and supports a sliding-window of recent turns plus a rolling summary for older context.

It also carries the **turn-context memory transfer** for the turn-level agentic conversation loop (`docs/retrieval/TURN_LOOP_DESIGN.md` §7): turns persist the loop's action records, served-chunk references, answer confidence, and clarification payloads; conversation meta carries the deep-study ledger; and `build_context` returns a `structured` dict alongside the prose `context_text`. All turn-loop fields are default-empty and decode-tolerant, so records written before the migration keep loading (same posture as the `_decode_id_list` doc-list migration).

## Files

| File | Purpose | Key Exports |
| --- | --- | --- |
| `provider.py` | Redis-backed and no-op memory implementations with singleton factory; turn-loop persistence (`append_turn` kwargs, `record_doc_studied`) and the write-boundary preview cap | `ConversationMemoryProvider`, `RedisConversationMemory`, `NoopConversationMemory`, `get_conversation_memory` |
| `schemas.py` | Typed dataclasses for memory persistence (incl. turn-loop fields) | `ConversationTurn`, `ConversationSummary`, `ConversationMeta`, `MemoryContext` |
| `utils.py` | Memory context assembly helpers: prose builder + structured sibling + summarizer grounding renderers | `build_context_text`, `build_structured_context`, `render_docs_studied_grounding`, `render_clarification_grounding`, `now_ms` |
| `__init__.py` | Package facade | re-exports from `provider.py` and `schemas.py` |

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `RAG_MEMORY_ENABLED` | `true` | Enable conversation memory |
| `RAG_MEMORY_PROVIDER` | `redis` | Backend: `redis` or `noop` |
| `RAG_MEMORY_REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `RAG_MEMORY_REDIS_PREFIX` | `rag:memory` | Redis key prefix |
| `RAG_MEMORY_MAX_RECENT_TURNS` | `8` | Sliding window size (recent turns kept verbatim) |
| `RAG_MEMORY_SUMMARY_TRIGGER_TURNS` | `12` | Rolling summary trigger threshold |
| `RAG_MEMORY_MAX_CONTEXT_TOKENS_ESTIMATE` | `1400` | Max estimated tokens for injected memory context |
| `RAG_TURN_CONTEXT_MAX_CHUNK_REFS` | `24` | Cap on chunk refs in the structured context and on the docs-studied ledger (newest kept) |
| `RAG_TURN_CONTEXT_PREVIEW_CHARS` | `320` | Per-ref preview cap applied at the provider write boundary and in context assembly |
| `RAG_TURN_CONTEXT_STORE_FULL_TEXT` | `false` | When `true`, stores full source/chunk text on turns (debugging escape hatch; unbounded Redis growth). Context previews stay capped either way |

## Context Management

```
New turn added (optionally with turn-loop fields:
actions, chunk_refs, answer_confidence, clarification;
source/chunk text preview-capped at the write boundary)
  ↓
Recent turns (last N) kept verbatim in MemoryContext
  ↓
Older turns → rolling summary (LLM-generated when trigger_turns threshold
crossed; summarizer input includes docs-studied + clarification grounding)
  ↓
MemoryContext injected into RAG prompt as system/user context
  + MemoryContext.structured consumed by the turn loop (TurnContext)
```

## Turn-loop context transfer (design §7)

- **Per-turn fields** (`ConversationTurn`): `actions` (TurnActionRecord dicts `{action, reason, ms, llm_calls}`), `chunk_refs` (ChunkRef dicts `{chunk_id, document_id, source_key, heading, score, refactored_char_start, refactored_char_end, preview}`), `answer_confidence`, `clarification` (`{question, hints, scoping_questions}` when the turn ended `ask_user`).
- **Meta ledger** (`ConversationMeta.docs_studied`): appended via `record_doc_studied(...)` (`{document_id, windows_read, sections, conclusion, ts}`; `ts` stamped when absent), same JSON-list hash pattern as `relevant_doc_ids`/`ignored_doc_ids`, capped at `RAG_TURN_CONTEXT_MAX_CHUNK_REFS` (newest kept).
- **Structured context** (`MemoryContext.structured`, built by `utils.build_structured_context` over the full fetched turn window): `{rolling_summary, recent_turns: [{question, answer}], chunk_refs (newest turn first, capped, preview-capped), docs_studied, pending_clarification}`. A clarification is *pending* while no later user turn exists; the first user reply consumes it.
- **Growth control**: `append_turn` is the single write boundary — `sources[].text` and `chunk_refs[].preview` are capped at `RAG_TURN_CONTEXT_PREVIEW_CHARS` unless `RAG_TURN_CONTEXT_STORE_FULL_TEXT=true`. Full chunk text stays recoverable by `chunk_id` (Weaviate uuid).
- **Compaction grounding**: `_llm_summarize` receives the docs-studied ledger and renders clarification-asked lines per turn, so the rolling summary preserves what was studied/asked. Pre-migration two-argument `_llm_summarize` overrides (test stubs, subclasses) keep working — the dispatch inspects the callable before passing `docs_studied`.
