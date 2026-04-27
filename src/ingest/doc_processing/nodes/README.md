<!-- @summary
LangGraph node implementations for the Document Processing Pipeline. Each file
is a single pipeline stage (ingestion, structure detection, text cleaning,
multimodal processing, and LLM refactoring) that reads from and writes to
DocumentProcessingState.
@end-summary -->

# nodes

One file per pipeline stage. Nodes are pure state-transformer functions consumed
by the workflow graph defined in the parent `doc_processing` package.

## Contents

| Path | Purpose |
| --- | --- |
| `__init__.py` | Re-exports all node functions as the stable package surface |
| `document_ingestion.py` | Phase 1 — reads source file bytes and computes SHA-256 hash |
| `structure_detection.py` | Extracts structural cues via ParserRegistry or Docling fallback; stores `parse_result` and `parser_instance` on state |
| `text_cleaning.py` | Markdown-aware text normalization and figure note injection |
| `multimodal_processing.py` | Optional VLM note synthesis from detected figure mentions |
| `document_refactoring.py` | Optional LLM-driven document rewrite pass |
