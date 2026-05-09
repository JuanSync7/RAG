# @summary
# Prometheus metrics for the deep-research orchestrator.
# Exports: DR_RUNS_TOTAL, DR_LLM_CALLS_TOTAL, DR_ITERATIONS, DR_TOPIC_COUNT,
#          DR_ITERATION_DURATION, DR_RUN_DURATION
# Deps: prometheus_client
# @end-summary
"""Prometheus metrics for the deep-research orchestrator.

Importing this module registers DR-specific counters and histograms in the
default Prometheus registry. The metrics are scraped via the existing
``/metrics`` endpoint exposed by the rag-api server.

Outcome label semantics for ``DR_RUNS_TOTAL``:

- ``ok`` — the orchestrator finished without budget exhaustion.
- ``early_stop`` — terminated early because the sufficiency check was satisfied
  on the root retrieval (no decomposition occurred).
- ``budget_exhausted`` — one of the budget caps (nodes / llm calls / wall
  clock) tripped before completion.
- ``error`` — the orchestrator raised before producing a result.

Stage label semantics for ``DR_LLM_CALLS_TOTAL``:

- ``sufficiency`` — ``_sufficiency_check`` invocations.
- ``multi_queries`` — ``_decompose`` invocations (the multi-queries-gen call).
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

DR_RUNS_TOTAL = Counter(
    "rag_dr_runs_total",
    "DR runs",
    ["outcome"],
)

DR_LLM_CALLS_TOTAL = Counter(
    "rag_dr_llm_calls_total",
    "DR LLM calls",
    ["stage"],
)

DR_ITERATIONS = Histogram(
    "rag_dr_iterations",
    "DR loop iterations",
    buckets=[1, 2, 3, 4, 5, 6, 8, 10],
)

DR_TOPIC_COUNT = Histogram(
    "rag_dr_topic_count",
    "DR topics per run",
    buckets=[1, 2, 3, 4, 5, 8],
)

DR_ITERATION_DURATION = Histogram(
    "rag_dr_iteration_duration_seconds",
    "Per-iteration duration",
)

DR_RUN_DURATION = Histogram(
    "rag_dr_run_duration_seconds",
    "End-to-end DR run duration",
)

DR_PER_TOPIC_RERANK_TOTAL = Counter(
    "rag_dr_per_topic_rerank_total",
    "DR per-topic rerank decisions per run",
    ["mode"],
)


__all__ = [
    "DR_RUNS_TOTAL",
    "DR_LLM_CALLS_TOTAL",
    "DR_ITERATIONS",
    "DR_TOPIC_COUNT",
    "DR_ITERATION_DURATION",
    "DR_RUN_DURATION",
    "DR_PER_TOPIC_RERANK_TOTAL",
]
