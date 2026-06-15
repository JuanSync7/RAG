<!-- @summary
RAPTOR-lite document-routing package: Stage-1 card-index router, comparison-query decomposition (regex + glossary-LLM, gated orchestrator), the shared AMBA protocol glossary + comparison-intent detector, and the typed routing contracts. Routing-only; cards/summaries are NEVER sent to the LLM. All behaviour is gated OFF by default.
@end-summary -->

# retrieval/routing

## Overview

This sub-package implements the query-side of the **RAPTOR-lite document
routing** feature: Stage-1 "where do I look?" routing over a small per-document
**card** index, plus the comparison-query **decomposition** that feeds it. It is
a thin, side-effect-light package — the heavy work (vector search, LLM calls) is
delegated to existing facades (`src.vector_db.search`, `src.platform.llm`).

Routing is **soft** (top-N, never top-1; never a hard filter) and degrades to
pure flat retrieval on low confidence or any failure. Cards/summaries are
**routing-only and are never sent to the LLM**. Everything is gated OFF by
default via `RAG_DOCUMENT_ROUTING_*` / `RAG_DECOMPOSITION_*` config keys.

See the design ([`docs/retrieval/DOCUMENT_ROUTING_DESIGN.md`](../../../docs/retrieval/DOCUMENT_ROUTING_DESIGN.md))
and the engineering guide
([`docs/retrieval/DOCUMENT_ROUTING_ENGINEERING_GUIDE.md`](../../../docs/retrieval/DOCUMENT_ROUTING_ENGINEERING_GUIDE.md)).

## Files

| File | Purpose | Key Exports |
| --- | --- | --- |
| `__init__.py` | Stable public facade (thin; imports nothing heavy) | `route_documents`, `decompose_query`, `DecompositionError`, `PROTOCOL_GLOSSARY`, `detect_comparison_intent`, `canonicalize_terms`, `RoutingResult`, `DecompositionResult`, `DocCard` |
| `schemas.py` | Typed cross-slice contracts (pure dataclasses, no I/O) | `DocCard`, `RoutingResult`, `DecompositionResult` |
| `glossary.py` | Shared AMBA protocol glossary + lexical comparison-intent detector + term canonicalizer | `PROTOCOL_GLOSSARY`, `detect_comparison_intent`, `canonicalize_terms` |
| `decomposition.py` | Tier-1 regex (fallback), Tier-2 glossary-LLM (primary), and the gated orchestrator | `regex_decompose`, `llm_decompose`, `decompose_query`, `DecompositionError` |
| `router.py` | Stage-1 card-index router: query embedding → routed `document_id`s (soft hint) | `route_documents` |

## Schema Ownership

- `DocCard` — a routing-only per-document summary card (title + section headings
  [+ optional LLM summary]); embedded into the card index, never sent to the LLM.
- `RoutingResult` — Stage-1 output: routed `doc_ids`, optional per-doc `scores`,
  and a `used` flag (`used=False` → fall back to pure flat retrieval).
- `DecompositionResult` — decomposition output: `subqueries`, the `method`
  (`"identity"` / `"regex"` / `"llm"`), and a `decomposed` flag.

## Flow

```
query
  → decompose_query        (gated on RAG_DECOMPOSITION_ENABLED + comparison intent;
                            identity → [query] otherwise; LLM-primary, regex fallback)
  → route_documents        (per sub-query: vector-first search over the card index
                            → top-N routed document_ids; used=False → flat fallback)
  → union routed doc_ids   (de-duped, first-seen order across sub-queries)
  → (consumed by RAGChain._collect_candidates as a soft candidate-pool hint)
```

Consumed by `src/retrieval/pipeline/rag_chain.py` at Stage 3.5
(`_route_documents_stage1`) and Stage 4 (`_collect_candidates`, routed `in`-filter
union). The single rerank pass downstream remains the final authority.
