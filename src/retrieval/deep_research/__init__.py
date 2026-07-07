# @summary
# Deep research instrumentation package: Prometheus metrics for the DR loop.
# Exports: metrics module symbols (re-exported for convenience).
# Deps: prometheus_client
# @end-summary
"""Deep research observability surface.

Hosts Prometheus counters and histograms for the deep-research orchestrator
implemented in :mod:`src.retrieval.pipeline.deep_research`. Importing this
package registers metrics in the default Prometheus registry.
"""

from src.retrieval.deep_research.metrics import (
    DR_ITERATION_DURATION,
    DR_ITERATIONS,
    DR_LLM_CALLS_TOTAL,
    DR_PER_TOPIC_RERANK_TOTAL,
    DR_RUN_DURATION,
    DR_RUNS_TOTAL,
    DR_TOPIC_COUNT,
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
