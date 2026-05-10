<!-- @summary
Phase 1 of the two-phase ingestion pipeline: transforms source documents into clean Markdown persisted to the Clean Document Store.
@end-summary -->

# ingest/doc_processing

## Overview

This sub-package implements Phase 1 of the ingestion pipeline — the **Document Processing Pipeline** (5 LangGraph nodes). It transforms raw source documents into clean, structured Markdown text, which is written atomically to the `CleanDocumentStore` at `src/ingest/clean_store.py`.

**Entry point:** `run_document_processing(runtime, source_path, ...)` in `impl.py`

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Re-exports `run_document_processing` |
| `state.py` | `DocumentProcessingState` TypedDict — Phase 1 state contract |
| `workflow.py` | `build_document_processing_graph()` — 5-node StateGraph with conditional routing |
| `impl.py` | Runtime: compiles graph, runs it, returns `DocumentProcessingState` |
| `nodes/document_ingestion.py` | Node 1: format detection, text extraction, SHA-256 hashing |
| `nodes/structure_detection.py` | Node 2: section tree, tables, figures (Docling) |
| `nodes/multimodal_processing.py` | Node 3: VLM figure descriptions (optional, qwen2.5vl:3b) |
| `nodes/text_cleaning.py` | Node 4: whitespace, boilerplate, multimodal note integration |
| `nodes/document_refactoring.py` | Node 5: paragraph self-containment via LLM (optional) |

## State Contract

`DocumentProcessingState` key outputs:
- `source_hash` — SHA-256 of the raw source file (for change detection)
- `cleaned_text` — final clean Markdown text written to CleanDocumentStore
- `refactored_text` — LLM-refactored variant (optional, may be None)
- `errors` — list of error strings; non-empty means Phase 2 should be skipped

## Phase Boundary

After `run_document_processing` completes without errors, the orchestrator (`pipeline/impl.py`) writes `cleaned_text` to `CleanDocumentStore` before calling Phase 2.
