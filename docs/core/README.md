<!-- @summary
Cross-cutting core infrastructure: embedding providers, reranker providers, inference backend configuration.
@end-summary -->

# docs/core

Cross-cutting infrastructure used by both the ingestion and retrieval pipelines.

## Documents

| Document | Coverage | Type |
| --- | --- | --- |
| [INFERENCE_BACKEND_ENGINEERING_GUIDE.md](INFERENCE_BACKEND_ENGINEERING_GUIDE.md) | `src/core/embeddings.py`, `src/retrieval/query/nodes/reranker.py`, vLLM Docker services, `config/settings.py` inference vars | Engineering Guide |

## Overview

The inference backend subsystem provides pluggable embedding and reranking providers. A single `RAG_INFERENCE_BACKEND` env var switches between:

- **`local`** (default) — BGE models run in-process inside the dev venv (requires the `local-embed` extra)
- **`tei`** — BGE models served by the always-on `rag-embed` and `rag-rerank` containers (HuggingFace TEI) over HTTP

See the [engineering guide](INFERENCE_BACKEND_ENGINEERING_GUIDE.md) for architecture, configuration reference, operational runbook, and extension instructions.
