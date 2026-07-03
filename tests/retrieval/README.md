<!-- @summary
Retrieval subsystem tests: confidence-based query routing, confidence score computation, and document result formatting.
@end-summary -->

# tests/retrieval

## Overview

This directory contains tests specific to the retrieval subsystem.

## Files

| File | Purpose |
| --- | --- |
| `test_confidence_routing.py` | LangGraph confidence-based query routing decisions (ANSWER / REFINE / REJECT paths) |
| `test_confidence_scoring.py` | Confidence score computation from query evaluation outputs |
| `test_document_formatter.py` | Retrieved document formatting for prompt injection |
| `test_retrieve_primitive.py` | `RAGChain.retrieve_primitive` + the `retrieve_ranked` Temporal activity: hyde-vs-query embedding, LRU keying, rerank anchoring, chunk_id stability, request validation |
| `turn_loop/` | Turn-level agentic conversation loop package tests (controller, retrieve, deep-study, clarify, answer gate, orchestrator, events, context) |

## Running

```bash
source .venv/bin/activate
pytest tests/retrieval/ -v
```
