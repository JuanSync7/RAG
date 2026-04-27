<!-- @summary
Coverage tests for the doc_processing pipeline nodes: document ingestion,
structure detection, text cleaning, document refactoring, multimodal processing,
and the CleanDocumentStore backing store.
@end-summary -->

# tests/ingest/doc_processing

Unit and coverage tests for the `src/ingest/doc_processing` subsystem. Each
file targets a specific pipeline node or shared store component, exercising
error paths, boundary conditions, and processing-log correctness.

## Contents

| Path | Purpose |
| --- | --- |
| `test_clean_store_coverage.py` | `CleanDocumentStore` — tmp-file cleanup on write failure, `_safe_key` sanitization, delete/list edge cases |
| `test_document_ingestion_coverage.py` | `document_ingestion_node` — error paths, hash correctness, boundary inputs, and `processing_log` entries |
| `test_document_refactoring_coverage.py` | `document_refactoring_node` — LLM fallback paths, disabled/empty/oversized-text cases, prompt-format verification |
| `test_multimodal_processing_coverage.py` | `multimodal_processing_node` — vision failure modes, skip conditions, note composition, and structure telemetry |
| `test_structure_detection_coverage.py` | `structure_detection_node` — regex fallback, Docling integration, boundary cases, and `processing_log` correctness |
| `test_text_cleaning_coverage.py` | `text_cleaning_node` — exception propagation, empty/large/whitespace-only text, figure-note composition, and log recording |
