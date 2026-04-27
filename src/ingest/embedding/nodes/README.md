<!-- @summary
One-stage-per-file LangGraph node implementations for the Embedding Pipeline,
covering chunking, enrichment, deduplication, embedding, storage, and optional
knowledge-graph and visual-embedding paths.
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
| `embedding_storage.py` | Embedding generation (batched) and vector store persistence |
| `document_storage_node.py` | Persists the clean markdown document to MinIO before chunking |
| `metadata_generation.py` | Document-level summary and keyword generation with fallback extraction |
| `cross_reference_extraction.py` | Optional cross-reference pattern extraction from document text |
| `knowledge_graph_extraction.py` | Optional relation extraction to intermediate KG triples |
| `knowledge_graph_storage.py` | Optional persistence of chunks into the knowledge graph builder |
| `visual_embedding.py` | Dual-track visual embedding: page images via Docling, stored in MinIO, indexed in Weaviate |
| `vlm_enrichment.py` | Post-chunking VLM image enrichment — resolves image placeholders in chunks |
| `__init__.py` | Package marker |
