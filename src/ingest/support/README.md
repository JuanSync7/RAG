<!-- @summary
Node support libraries for specialized processing: document parsing (Docling), vision/VLM enrichment, LLM helpers, and text processing primitives.
@end-summary -->

# ingest/support

## Overview

This directory contains support libraries consumed by pipeline nodes. These are **not** nodes themselves -- they provide reusable processing capabilities that specific nodes depend on.

## Files

| File | Purpose | Key Exports | Consumed By |
| --- | --- | --- | --- |
| `docling.py` | Docling document parsing and model warmup | `parse_with_docling`, `ensure_docling_ready`, `warmup_docling_models` | `nodes/structure_detection`, `pipeline/impl` |
| `vision.py` | Vision/VLM figure caption, OCR, and tag extraction via LiteLLM Router | `generate_vision_notes`, `ensure_vision_ready` | `nodes/multimodal_processing`, `pipeline/impl` |
| `llm.py` | LLM JSON helper backed by LiteLLM Router | `_llm_json` | `nodes/document_refactoring`, `nodes/metadata_generation` |
| `document.py` | Text cleaning primitives and metadata extraction | `extract_metadata`, `metadata_to_dict`, `strip_boilerplate`, `normalize_unicode`, `clean_whitespace`, `strip_trailing_short_lines` | `support/markdown` |
| `markdown.py` | Markdown-aware cleaning and semantic chunking | `chunk_markdown`, `clean_document`, `normalize_headings_to_markdown` | `nodes/chunking`, `nodes/text_cleaning` |

## Dependency Notes

- `markdown.py` depends on `document.py` for foundational text cleaning helpers.
- `vision.py` and `llm.py` depend on `common/types.py` for `IngestionConfig` and `src.platform.llm` for LiteLLM-backed completions.
- `docling.py` has no ingest-internal dependencies (standalone Docling integration).
