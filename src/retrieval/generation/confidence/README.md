<!-- @summary
Post-generation confidence scoring and routing for the RAG pipeline. Combines three independent
signals — retrieval quality, LLM self-reported confidence, and citation coverage — into a single
weighted composite score, then routes the answer to RETURN, RE_RETRIEVE, FLAG, or BLOCK.
The RE_RETRIEVE action drives a real bounded internal loop inside rag_chain.run() (option B).
@end-summary -->

# retrieval/generation/confidence

This package implements the 3-signal composite confidence model used after answer generation.
The composite score drives post-guardrail routing decisions defined in REQ-706: high-confidence
answers are returned immediately, medium-confidence answers trigger a bounded internal re-retrieval
loop, and low-confidence answers are flagged or blocked after retries are exhausted.

## Contents

| Path | Purpose |
| --- | --- |
| `schemas.py` | Typed contracts: `ConfidenceBreakdown` dataclass (three signals + composite) and `PostGuardrailAction` enum (RETURN, RE_RETRIEVE, FLAG, BLOCK) |
| `scoring.py` | Pure scoring functions: `compute_retrieval_confidence` (top-N reranker average), `parse_llm_confidence` (label-to-float mapping with overconfidence correction), `compute_citation_coverage` (citation markers + n-gram overlap blend), and `compute_composite_confidence` (weighted aggregation) |
| `routing.py` | `route_by_confidence` — maps a composite score and retry count to a `PostGuardrailAction` using the REQ-706 decision table; NaN inputs are safe-failed to BLOCK |
| `__init__.py` | Package facade re-exporting `ConfidenceBreakdown`, `PostGuardrailAction`, `compute_composite_confidence`, and `route_by_confidence` |

## Re-retrieval semantics (option B — bounded internal loop)

When `route_by_confidence` returns `RE_RETRIEVE`, the loop lives **inside** `rag_chain.run()`,
not in the caller. The caller does not need to re-invoke the pipeline to trigger a second pass.

### Loop contract

| Composite | retry budget | Outcome |
| --- | --- | --- |
| >= `high_threshold` (0.70) | any | `RETURN` — answer returned immediately, no retry |
| < `high_threshold`, retries available | > 0 | internal loop fires: search + rerank + generate + rescore |
| < `high_threshold`, retries exhausted | 0 | `FLAG` or `BLOCK` based on `low_threshold` |
| NaN | any | `BLOCK` (safe-fail, no retry) |

### Internal loop behaviour

When the loop fires on a `RE_RETRIEVE` decision:

1. **Re-search** — hybrid search with broader params (`alpha -= 0.15`, `search_limit += 5`).
   Uses the already-embedded query vector — no re-embed, no re-PII-gate, no re-input-rails.
2. **Re-rank** — cross-encoder reranker on broader candidate set.
3. **Re-generate** — LLM generation on broader context. Memory context is omitted on the retry
   to keep the re-retrieval focused on document evidence.
4. **Rescore** — composite confidence computed on retry answer.
5. **Pick best** — whichever attempt (first or second) has the higher composite score is
   returned. Both scores are surfaced: `first_composite` holds the initial score,
   `composite_confidence` holds the winning score.

### Response fields

| Field | Type | Meaning |
| --- | --- | --- |
| `composite_confidence` | `float \| None` | Winning composite score (best of all attempts) |
| `first_composite` | `float \| None` | Initial composite score before any retry; `None` if no retry was attempted |
| `post_guardrail_action` | `str \| None` | Final action after all retries: `"return"`, `"flag"`, or `"block"` |
| `re_retrieval_suggested` | `bool` | `True` when retries are exhausted and composite is still in the medium band — caller may request a third pass with even-broader params |
| `re_retrieval_params` | `dict \| None` | Suggested params for a caller-driven third pass when `re_retrieval_suggested=True` |

### Memory-path exception

Re-retrieval is not applicable when `generation_source == "memory"` — there are no documents
to broaden. In that case the `RE_RETRIEVE` decision is re-routed to `FLAG` without an internal
loop, and a verification warning is attached to the answer.

### Hard cap

The loop is capped at `RAG_CONFIDENCE_RE_RETRIEVE_MAX_RETRIES` (default `1`, env-configurable).
Set this to `0` to disable the internal loop entirely — the chain will then fall through to
`FLAG`/`BLOCK` directly from the first attempt.
