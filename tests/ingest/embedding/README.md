<!-- @summary
Unit tests for the embedding pipeline nodes: chunking, chunk enrichment,
metadata generation, quality validation, cross-reference extraction,
document storage, embedding storage, and knowledge-graph extraction/storage.
@end-summary -->

# tests/ingest/embedding

Unit tests for the `src/ingest/embedding` subsystem. Each file covers one
pipeline node, verifying correct behaviour for happy paths, disabled-flag
passthrough, error isolation, and edge cases such as empty chunk lists.

## Contents

| Path | Purpose |
| --- | --- |
| `test_chunk_enrichment.py` | `chunk_enrichment_node` — chunk-id derivation, determinism, ordinal uniqueness, source propagation, enriched-content population |
| `test_chunking.py` | `chunking_node` — text source selection, heading normalisation, semantic vs standard dispatch, `ProcessedChunk` metadata, empty-chunk handling |
| `test_cross_reference.py` | `cross_reference_extraction_node` — disabled passthrough, DOC/Section/RFC pattern extraction, deduplication, text source preference |
| `test_document_storage.py` | `document_storage_node` — document-id derivation (SHA-256), upload gating, MinIO client presence checks, error capture |
| `test_embedding_storage.py` | `embedding_storage_node` — empty-chunk short-circuit, single/multi-chunk storage, `update_mode` delete-before-insert ordering, collection routing, error isolation |
| `test_kg_extraction.py` | `knowledge_graph_extraction_node` — disabled passthrough, triple extraction per chunk, error isolation, partial-success accumulation |
| `test_kg_storage.py` | `knowledge_graph_storage_node` — disabled skip, None-builder skip, `add_chunk` call contract, error isolation, partial-success continuation |
| `test_metadata_generation.py` | `metadata_generation_node` — LLM summary/keyword extraction, disabled/empty-response fallback, `max_keywords` cap, empty-chunks guard |
| `test_quality_validation.py` | `quality_validation_node` — disabled passthrough, short-chunk and quality-score filtering, deduplication, metadata injection |
