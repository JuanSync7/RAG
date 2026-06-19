<!-- @summary
One-stage-per-file LangGraph node implementations for the Embedding Pipeline,
covering chunking, enrichment, deduplication, embedding, storage, and the
optional visual-embedding path. Knowledge-graph ingest lives in KGWeave and
runs out-of-process via the Temporal handoff (KG_PHASE2B_ACTIVITY).
@end-summary -->

# embedding/nodes

Each file in this directory implements a single LangGraph pipeline stage. Nodes
are composed into a graph by `workflow.py`; this directory contains only
the stage logic, not the wiring.

## Contents

| Path | Purpose |
| --- | --- |
| `chunking.py` | Chunk generation via parser abstraction, with legacy markdown fallback |
| `chunk_enrichment.py` | Chunk ID assignment and enriched content projection |
| `quality_validation.py` | Optional chunk quality gating and intra-document deduplication |
| `cross_document_dedup.py` | Cross-document deduplication using Tier 1 (SHA-256) and optional Tier 2 (MinHash) matching |
| `tree_node_synthesis.py` | Optional tree retrieval: emits one section node per unique `heading_path` prefix; no-op when `config.enable_tree_retrieval_ingest` is False |
| `document_card.py` | Optional RAPTOR-lite routing: builds one card per document (title + headings), embeds `card_text`, stages `staged_card_records`; strict no-op when `config.build_document_cards` is False (default) |
| `embedding_storage.py` | Embedding generation (batched) and vector store persistence |
| `document_storage_node.py` | Persists the clean markdown document to MinIO before chunking |
| `metadata_generation.py` | Document-level summary and keyword generation with fallback extraction |
| `cross_reference_extraction.py` | Optional cross-reference pattern extraction from document text |
| `visual_embedding.py` | Dual-track visual embedding: page images via Docling, stored in MinIO, indexed in Weaviate |
| `vlm_enrichment.py` | Post-chunking VLM image enrichment — resolves image placeholders in chunks |
| `__init__.py` | Package marker |
