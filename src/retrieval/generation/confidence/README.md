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
| `schemas.py` | Typed contracts: `ConfidenceBreakdown` dataclass (three signals + composite) and `PostGuardrailAction` enum (RETURN, RE_RETRIEVE, FLAG, BLOCK) |
| `scoring.py` | Pure scoring functions: `compute_retrieval_confidence` (top-N reranker average), `parse_llm_confidence` (label-to-float mapping with overconfidence correction), `compute_citation_coverage` (citation markers + n-gram overlap blend), and `compute_composite_confidence` (weighted aggregation) |
| `routing.py` | `route_by_confidence` — maps a composite score and retry count to a `PostGuardrailAction` using the REQ-706 decision table; NaN inputs are safe-failed to BLOCK |
| `__init__.py` | Package facade re-exporting `ConfidenceBreakdown`, `PostGuardrailAction`, `compute_composite_confidence`, and `route_by_confidence` |
