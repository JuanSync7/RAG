<!-- @summary
Post-generation confidence scoring and routing for the RAG pipeline. Combines three independent
signals — retrieval quality, LLM self-reported confidence, and citation coverage — into a single
weighted composite score, then routes the answer to RETURN, RE_RETRIEVE, FLAG, or BLOCK.
@end-summary -->

# retrieval/generation/confidence

This package implements the 3-signal composite confidence model used after answer generation.
The composite score drives post-guardrail routing decisions defined in REQ-706: high-confidence
answers are returned immediately, medium-confidence answers trigger a re-retrieval attempt, and
low-confidence answers are flagged or blocked after retries are exhausted.

## Contents

| Path | Purpose |
| --- | --- |
| `schemas.py` | Typed contracts: `ConfidenceBreakdown` dataclass (three signals + composite), `ConfidenceWeights` frozen dataclass (validated sum-to-1 weights), and `PostGuardrailAction` enum (RETURN, RE_RETRIEVE, FLAG, BLOCK) |
| `scoring.py` | Pure scoring functions: `compute_retrieval_confidence` (top-N reranker average), `parse_llm_confidence` (label-to-float mapping with overconfidence correction), `compute_citation_coverage` (citation markers + n-gram overlap blend), and `compute_composite_confidence` (weighted aggregation) |
| `routing.py` | `route_by_confidence` — maps a composite score and retry count to a `PostGuardrailAction` using the REQ-706 decision table; NaN inputs are safe-failed to BLOCK |
| `__init__.py` | Package facade re-exporting `ConfidenceBreakdown`, `ConfidenceWeights`, `PostGuardrailAction`, `compute_composite_confidence`, and `route_by_confidence` |

## Behaviour changes (robustness fixes)

### Sentence splitter (#1)
`_split_sentences` was replaced with a hand-rolled splitter that handles:
- Common English abbreviations (Mr., Dr., etc., e.g., i.e., vs.) — not split
- Decimal numbers (3.14, v2.0) — not split
- URLs (https://...) — periods inside URL runs not split
- Ellipses (`...`, `…`) — treated as continuation, not a sentence boundary
- Single `\n` without trailing space — treated as a sentence boundary
- Double `\n` — treated as a hard paragraph boundary

### Partial citation credit (#9)
`compute_citation_coverage` now awards fractional credit per sentence.
A sentence with citations `[1, 99]` against 1 chunk now scores `0.5`
(1 valid of 2 cited), not `1.0` (any-valid).  The 70%/30% blend weights
at the aggregation level are unchanged.

### Short-sentence handling (#10)
The word-count filter threshold was lowered from `>= 4` to `>= 3` words.
The splitter also keeps at least one fragment for non-empty input so the list
is never spuriously empty (which previously returned coverage `1.0`).
When the answer has no substantive sentences, `compute_citation_coverage`
returns `0.5` (neutral) instead of `1.0` (vacuously perfect).

### ConfidenceWeights dataclass (#13)
`ConfidenceWeights(retrieval, llm, citation)` is a frozen dataclass in
`schemas.py` that validates `sum == 1.0` once at construction.
`compute_composite_confidence` accepts either the dataclass via the `weights=`
kwarg or the existing float kwargs (back-compat).

### Memory-path confidence gate (#8)
The gate condition in `rag_chain.py` previously excluded the scoring block
entirely when `reranked == []` (memory-only generation path).  The condition
now also allows scoring when `generation_source == "memory"`.  On this path:
- `reranker_scores=[]` → `retrieval_score=0.0` (no retrieval signal, by design)
- `retrieved_texts=[]` → `citation_score` reflects only citation-marker presence
- The composite still provides meaningful signal via the LLM self-report weight
