<!-- @summary
Phase 2 of the two-phase ingestion pipeline: transforms clean Markdown from the Clean Document Store into vector embeddings and optional knowledge graph triples.
@end-summary -->

# ingest/embedding

## Overview

This sub-package implements Phase 2 of the ingestion pipeline — the **Embedding Pipeline** (8 LangGraph nodes). It reads clean Markdown text (produced by Phase 1) and transforms it into:

- vector embeddings stored in Weaviate,
- optional knowledge graph triples.

**Entry point:** `run_embedding_pipeline(runtime, source_key, clean_text, ...)` in `impl.py`

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Re-exports `run_embedding_pipeline` |
| `state.py` | `EmbeddingPipelineState` TypedDict — Phase 2 state contract |
| `workflow.py` | `build_embedding_graph()` — 8-node StateGraph with conditional routing |
| `impl.py` | Runtime: compiles graph, runs it, returns `EmbeddingPipelineState` |
| `nodes/chunking.py` | Node 6: semantic, character, or section-aware chunking |
| `nodes/chunk_enrichment.py` | Node 7: chunk ID generation, provenance, metadata enrichment |
| `nodes/metadata_generation.py` | Node 8: LLM keyword/summary extraction with TF-IDF fallback |
| `nodes/cross_reference_extraction.py` | Node 9: inter-document reference detection (optional) |
| `nodes/knowledge_graph_extraction.py` | Node 10: triple extraction (optional) |
| `nodes/quality_validation.py` | Node 11: quality scoring + dedup filtering (optional) |
| `nodes/embedding_storage.py` | Node 12: embedding generation + Weaviate upsert |
| `nodes/knowledge_graph_storage.py` | Node 13: graph store persistence (optional) |

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

Three nodes are optional (controlled by `IngestionConfig`):
- `cross_reference_extraction` — enabled by `enable_cross_reference_extraction`
- `knowledge_graph_extraction` — enabled by `enable_knowledge_graph_extraction`
- `knowledge_graph_storage` — enabled by `enable_knowledge_graph_storage`
