<!-- @summary
Document ingestion package with a two-phase LangGraph pipeline: doc_processing (5 nodes) and embedding (8 nodes), connected by the Clean Document Store.
@end-summary -->

# ingest

## Overview

This package powers embedding ingestion for the RAG system. The pipeline is organized as two phases connected by the Clean Document Store:

- `common/`: cross-cutting contracts, state/config types, node helpers, and deterministic utilities
- `doc_processing/`: Phase 1 pipeline — document ingestion through clean text output (5 LangGraph nodes)
- `embedding/`: Phase 2 pipeline — chunking through vector and KG storage (8 LangGraph nodes)
- `clean_store.py`: `CleanDocumentStore` — the Phase 1→2 boundary (atomic read/write)
- `support/`: node support libraries (document parsing, vision/VLM, LLM helpers, text processing)
- `impl.py`: pipeline orchestrator — source discovery, idempotency, Phase 1 → Phase 2 coordination
- `temporal/`: Temporal execution layer — wraps phases as distributed activities for parallel GPU workers

Stage 2 (`structure_detection`) is the parser front-end: Docling converts source files
into markdown-first text suitable for downstream LLM and chunking stages, and emits
multimodal cues (for example figure presence) consumed by `multimodal_processing`.
When enabled, the multimodal stage can call an Ollama-hosted VLM model to describe
figure images (caption + OCR + tags), and appends those notes into cleaned text.

## Subdirectories

| Directory / File | Purpose |
| --- | --- |
| `common/` | Shared schemas, state/config types, universal node helpers, and deterministic utilities |
| `doc_processing/` | Phase 1 — document ingestion through clean text output (5 nodes) |
| `embedding/` | Phase 2 — chunking through vector and KG storage (8 nodes) |
| `clean_store.py` | `CleanDocumentStore` — atomic read/write boundary between Phase 1 and Phase 2 |
| `support/` | Node support libraries: Docling parsing, vision/VLM, LLM helpers, text processing |
| `impl.py` | Pipeline orchestrator: source discovery, idempotency checks, Phase 1 → Phase 2 coordination, KG export |
| `temporal/` | Temporal execution subsystem: activities, workflows, and worker entry point for distributed parallel ingestion |

## Internal Dependencies

- `impl.py` calls `run_document_processing` (from `doc_processing/`) and `run_embedding_pipeline` (from `embedding/`), and reads `common/types.py` for schema/runtime types.
- `temporal/activities.py` calls the same phase functions (`run_document_processing`, `run_embedding_pipeline`) — Temporal wraps them, it does not replace them.
- `doc_processing/workflow.py` wires nodes 1–5 with conditional graph transitions for multimodal and refactoring stages.
- `embedding/workflow.py` wires nodes 6–13 with conditional graph transitions for optional extraction and storage stages.
- Node modules in both sub-packages consume shared helpers from `common/shared.py`, deterministic helpers from `common/utils.py`, and specialized processing from `support/`.

## Engineering Documentation

- `docs/ingestion/INGESTION_PIPELINE_ENGINEERING_GUIDE.md`: implementation-oriented walkthrough of architecture, stage contracts, config/routing decisions, extension steps, and troubleshooting.
- `docs/ingestion/INGESTION_NEW_ENGINEER_ONBOARDING_CHECKLIST.md`: one-page onboarding checklist for first-day setup, first change workflow, and common gotchas.

## Runtime Controls

- `RAG_INGESTION_VERBOSE_STAGE_LOGS=true` enables per-stage progress logs for every document.
- `ingest.py --verbose-stages` forces per-stage logs on for that run.
- `ingest.py --no-verbose-stages` forces per-stage logs off for that run.
- Omitting both flags keeps the config default (`RAG_INGESTION_VERBOSE_STAGE_LOGS`).
- `RAG_INGESTION_DOCLING_ENABLED=true` enables Docling parser in stage 2 (`structure_detection`).
- `RAG_INGESTION_DOCLING_MODEL` sets the Docling parser model identifier used by stage 2.
- `RAG_INGESTION_DOCLING_ARTIFACTS_PATH` sets an optional local artifacts/cache path for Docling.
- `RAG_INGESTION_DOCLING_STRICT=true` fails fast if Docling parsing fails (recommended production mode).
- `RAG_INGESTION_DOCLING_AUTO_DOWNLOAD=true` pre-downloads Heron layout + TableFormer models during preflight.
- Ingestion performs a startup Docling preflight (`ensure_docling_ready`) before processing files and fails immediately if Docling runtime or artifacts configuration is invalid.
- `ingest.py --docling-model <id>` overrides parser model per run.
- `ingest.py --docling-artifacts-path <path>` sets local artifacts path per run.
- `ingest.py --no-docling` disables Docling parser for a run.
- `ingest.py --no-docling-auto-download` disables Docling model pre-download during preflight.
- `RAG_INGESTION_VISION_ENABLED=true` enables vision analysis in `multimodal_processing`.
- `RAG_INGESTION_VISION_PROVIDER` selects `ollama` or `openai_compatible`.
- `RAG_INGESTION_VISION_MODEL` sets the Ollama VLM model (for example `qwen2.5vl:3b`).
- `RAG_INGESTION_VISION_API_BASE_URL` sets endpoint root for `openai_compatible`.
- `RAG_INGESTION_VISION_API_KEY` sets bearer key for `openai_compatible` (keep in secrets store).
- `RAG_INGESTION_VISION_API_PATH` overrides chat completion path (default `/v1/chat/completions`).
- `RAG_INGESTION_VISION_MAX_FIGURES` limits figure analysis calls per document.
- `RAG_INGESTION_VISION_AUTO_PULL=true` auto-pulls missing vision model during preflight.
- `RAG_INGESTION_VISION_STRICT=true` fails a source if vision analysis errors occur.
- `ingest.py --vision` enables vision analysis for a run.
- `ingest.py --vision-provider <provider>` switches backend provider per run.
- `ingest.py --vision-model <id>` overrides the VLM model for a run.
- `ingest.py --vision-api-base-url <url>` sets endpoint base URL for `openai_compatible`.
- `ingest.py --vision-max-figures <n>` limits figure analysis calls for a run.
- `ingest.py --no-vision-auto-pull` disables VLM auto-pull in preflight.
- `ingest.py --vision-strict` turns vision errors into document-level failures.
- `RAG_INGESTION_PERSIST_REFACTOR_MIRROR=true` persists original/refactored mirror files plus chunk provenance mappings.
- Source identity is tracked with `source_key`, `source_id`, and `source_uri` metadata so files remain unique across directories/connectors and retrieval can reference original location.

## Source Identity Notes

- `source_key` is the stable ingestion identity used for update cleanup and manifest indexing.
- `source_uri` is the canonical retrieval location shown to users/operators.
- Chunk metadata always carries both `source` (human-readable display/filter field) and `source_key` (stable identity field for idempotent chunk IDs).
- Filename equality does not imply identity equality; files with the same name in different directories are treated as distinct sources.
- Manifest persistence is atomic (`.tmp` then replace), and corrupted manifest JSON is moved aside as `manifest.json.corrupt.<timestamp>` before continuing with an empty manifest.
- Refactoring never mutates original source files; mirror artifacts are written under `processed/refactor_mirror/`.
- Chunk metadata stores provenance (`original_char_*`, `refactored_char_*`, `provenance_*`) to map retrieval chunks back to source text.
