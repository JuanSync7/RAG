<!-- @summary
Phase 2 of the two-phase ingestion pipeline: transforms clean Markdown from the Clean Document Store into vector embeddings. Knowledge-graph ingest is owned by KGWeave and runs out-of-process via the Temporal handoff (KG_PHASE2B_ACTIVITY on KG_TASK_QUEUE).
@end-summary -->

# ingest/embedding

## Overview

This sub-package implements Phase 2 of the ingestion pipeline — the **Embedding Pipeline** (LangGraph nodes). It reads clean Markdown text (produced by Phase 1) and transforms it into vector embeddings stored in Weaviate.

Knowledge-graph ingest is *not* part of this pipeline anymore. It runs in the KGWeave worker fleet and is dispatched by `IngestDocumentWorkflow._run_kg_phase2b` after embedding succeeds (see `src/ingest/temporal/workflows.py`).

**Entry point:** `run_embedding_pipeline(runtime, source_key, clean_text, ...)` in `impl.py`

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Re-exports `run_embedding_pipeline` |
| `state.py` | `EmbeddingPipelineState` TypedDict — Phase 2 state contract |
| `workflow.py` | `build_embedding_graph()` — StateGraph with conditional routing |
| `impl.py` | Runtime: compiles graph, runs it, returns `EmbeddingPipelineState` |
| `nodes/chunking.py` | Semantic, character, or section-aware chunking |
| `nodes/chunk_enrichment.py` | Chunk ID generation, provenance, metadata enrichment |
| `nodes/metadata_generation.py` | LLM keyword/summary extraction with TF-IDF fallback |
| `nodes/cross_reference_extraction.py` | Inter-document reference detection (optional) |
| `nodes/quality_validation.py` | Quality scoring + dedup filtering (optional) |
| `nodes/embedding_storage.py` | Embedding generation + Weaviate upsert |
| `nodes/visual_embedding.py` | Optional ColQwen2 visual page embeddings |
| `nodes/commit_node.py` | Atomic-commit barrier for Weaviate / MinIO writes |

## State Contract

`EmbeddingPipelineState` key inputs (provided by orchestrator):
- `clean_text` / `cleaned_text` — clean Markdown from CleanDocumentStore
- `clean_hash` — SHA-256 of clean text (for change detection in Phase 2)
- `source_key`, `source_name`, `source_uri`, `source_id`, `connector`, `source_version`

Key outputs:
- `stored_count` — number of chunks successfully written to Weaviate
- `chunks` — list of `ProcessedChunk` objects
- `errors` — list of error strings

## Conditional Nodes

- `cross_reference_extraction` — enabled by `enable_cross_reference_extraction`
- `quality_validation` — enabled by `enable_quality_validation`
- `visual_embedding` — enabled by `enable_visual_embedding`
