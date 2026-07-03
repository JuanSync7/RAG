<!-- @summary
Full test suite for the RAG platform: API contracts, ingestion pipeline, retrieval subsystem, platform services, and end-to-end integration tests.
@end-summary -->

# tests

## Overview

This directory contains the full test suite for the RAG platform, covering API contracts, ingestion pipeline, retrieval subsystem, platform services, and end-to-end integration tests.

## Structure

| Path | Purpose |
| --- | --- |
| `conftest.py` | Shared pytest fixtures used across the full test suite |
| `ingest/` | Ingestion pipeline tests (pipeline schema, helpers, idempotency) |
| `retrieval/` | Retrieval subsystem tests (confidence routing, scoring, document formatting) |

## Test Files

| File | What it Tests |
| --- | --- |
| `test_api_error_envelope.py` | API error envelope normalization — all non-2xx responses produce `{ok, error, request_id}` |
| `test_api_key_store.py` | API key lifecycle: create, list, revoke, lookup with SHA-256 hashing |
| `test_api_schemas.py` | API request/response Pydantic model validation and `extra="forbid"` contracts |
| `test_cache_provider.py` | Cache provider behavior: in-memory TTL expiry, Redis round-trip, no-op fallback |
| `test_command_catalog_memory.py` | Slash-command catalog registration and memory command integration |
| `test_conversation_memory_api.py` | Conversation memory API endpoints: create, history, compact, delete |
| `test_document_processor.py` | Document processing pipeline stage behavior |
| `test_generator.py` | LLM answer generator (non-stream and stream paths) |
| `test_markdown_processor.py` | Markdown semantic chunking and cleaning |
| `test_mcp_adapter.py` | MCP tooling adapter tool registration and dispatch |
| `test_memory_provider.py` | Conversation memory provider: turn storage, sliding window, rolling summary trigger |
| `test_memory_turn_context.py` | Turn-loop memory extensions: append_turn loop kwargs, docs-studied ledger, structured turn context, preview capping, pending-clarification lifecycle |
| `test_podman_migration.py` | Podman compose compatibility and runtime detection |
| `test_project_config.py` | Configuration validation and env var binding |
| `test_query_filters.py` | Retrieval query filter parsing and application |
| `test_query_processor.py` | Query reformulation, evaluation, and confidence-based routing |
| `test_quota_store.py` | Tenant quota storage: set, get, delete, enforcement check |
| `test_rag_chain_budget.py` | Token budget calculation across retrieval stages |
| `test_rag_chain_integration.py` | End-to-end RAG chain: query → retrieval → generation |
| `test_rate_limiter.py` | Rate limiter: fixed-window counting, allow/deny transitions |
| `test_reranker.py` | Reranker candidate ordering |
| `test_security_auth.py` | Authentication (API key, JWT, OIDC) and authorization (role checks) |
| `test_server_config.py` | Server configuration binding and defaults |
| `test_validation.py` | Boundary validation helpers (alpha range, positive int, filter value, path) |
| `test_vector_store_integration.py` | Weaviate vector store: collection management, upsert, hybrid search |

## Running Tests

```bash
source .venv/bin/activate

pytest                                           # full suite
pytest tests/ingest/ -v                          # ingestion only
pytest tests/retrieval/ -v                       # retrieval only
pytest tests/test_rag_chain_integration.py -v    # integration only
pytest tests/test_api_schemas.py -v              # API contracts only
```
