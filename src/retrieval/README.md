<!-- @summary
Retrieval subsystem for query processing, hybrid search, reranking, and optional LLM generation with timing/observability support. Organised into query/, generation/, pipeline/, and routing/ sub-packages with shared pipeline contracts in common/.
@end-summary -->

# retrieval

## Overview

This directory contains the runtime retrieval path used by query serving, including:

- LangGraph-based query processing and confidence routing,
- hybrid vector + keyword retrieval orchestration,
- reranking and optional answer generation,
- stage timing instrumentation for retrieval vs generation latency splits.

## Package Structure

```
retrieval/
├── __init__.py           — public API surface
├── common/               — pipeline boundary contracts (RAGRequest, RAGResponse, RankedResult)
├── pipeline/             — end-to-end RAG orchestration (RAGChain)
├── query/                — query sub-pipeline: sanitization, reformulation, reranking
│   ├── schemas.py        — QueryAction, QueryResult, QueryState
│   └── nodes/            — query_processor, reranker
├── routing/              — RAPTOR-lite document routing: Stage-1 card-index router, comparison decomposition, protocol glossary (routing-only; gated OFF by default)
│   ├── schemas.py        — DocCard, RoutingResult, DecompositionResult
│   ├── glossary.py       — PROTOCOL_GLOSSARY, detect_comparison_intent, canonicalize_terms
│   ├── decomposition.py  — regex_decompose, llm_decompose, decompose_query
│   └── router.py         — route_documents
└── generation/           — generation sub-pipeline: formatting, LLM synthesis, confidence routing
    ├── schemas.py        — FormattedContext, VersionConflict
    ├── confidence/       — ConfidenceBreakdown, PostGuardrailAction, scoring, routing
    └── nodes/            — generator, document_formatter, output_sanitizer
```

## Schema Ownership

| Schema | Location | Purpose |
| --- | --- | --- |
| `RAGRequest` | `common/schemas.py` | Pipeline input contract |
| `RAGResponse` | `common/schemas.py` | Pipeline output contract |
| `RankedResult` | `common/schemas.py` | Wire type crossing query → generation |
| `QueryAction`, `QueryResult`, `QueryState` | `query/schemas.py` | Query sub-package internals |
| `FormattedContext`, `VersionConflict` | `generation/schemas.py` | Generation sub-package internals |
| `ConfidenceBreakdown`, `PostGuardrailAction` | `generation/confidence/schemas.py` | Post-gen confidence routing |

## Subdirectories

- `common/`: pipeline boundary contracts and shared wire types.
- `pipeline/`: `RAGChain` — composes query, KG expansion, hybrid search, reranking, and generation.
- `query/`: query sanitization, LLM-based reformulation, confidence routing, and reranking.
- `routing/`: RAPTOR-lite document routing — Stage-1 card-index router, comparison-query decomposition (regex + glossary-LLM), and the shared protocol glossary. Routing-only (cards/summaries never reach the LLM) and gated OFF by default; consumed by `RAGChain` at Stage 3.5 / Stage 4.
- `generation/`: document formatting, LLM answer synthesis, output sanitization, and composite confidence scoring.

## Engineering Documentation

- `docs/retrieval/README.md`: architecture overview and onboarding checklist.
- `docs/retrieval/DOCUMENT_ROUTING_DESIGN.md`: RAPTOR-lite document-routing design / rationale.
- `docs/retrieval/DOCUMENT_ROUTING_ENGINEERING_GUIDE.md`: as-built architecture, config keys, how-to-enable, extension, and troubleshooting for document routing.
