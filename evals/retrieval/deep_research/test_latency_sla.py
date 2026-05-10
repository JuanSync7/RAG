"""Latency SLA tests for the deep-research A/B harness.

Asserts wall-clock budgets on the existing 4 golden queries:

  * baseline p95 latency_ms   <  5 000 ms
  * deep_research p95 latency <  30 000 ms
  * DR/baseline ratio         <  6x on disjoint, < 3x on single_aspect

Skips when Weaviate / Ollama is not up (RAGChain init fails). Marked
``eval_deep_research`` so it stays out of default CI runs.
"""

from __future__ import annotations

import math
from typing import Iterable

import pytest

pytestmark = pytest.mark.eval_deep_research


def _p95(values: Iterable[float]) -> float:
    vals = sorted(v for v in values if v is not None and not math.isnan(v))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    # Nearest-rank p95 (small-N safe).
    idx = max(0, int(math.ceil(0.95 * len(vals))) - 1)
    return vals[idx]


def test_latency_sla_golden_queries(rag_chain, golden_queries_deep_research_asic):
    """Run the 4 OpenTitan golden queries through baseline + DR and assert SLAs."""
    from evals.retrieval.deep_research.harness import run_suite

    report = run_suite(rag_chain, golden_queries_deep_research_asic, top_k=10)

    runnable = [
        r for r in report.results
        if r.baseline and r.deep_research
        and not r.baseline.error and not r.deep_research.error
    ]
    if not runnable:
        pytest.skip("No queries executed cleanly through both modes.")

    baseline_lat = [r.baseline.latency_ms for r in runnable]
    dr_lat = [r.deep_research.latency_ms for r in runnable]

    p95_baseline = _p95(baseline_lat)
    p95_dr = _p95(dr_lat)

    assert p95_baseline < 5_000, (
        f"baseline p95 latency_ms={p95_baseline:.1f} exceeds 5 000 ms SLA. "
        f"Per-query baseline_lat={baseline_lat}"
    )
    assert p95_dr < 30_000, (
        f"deep_research p95 latency_ms={p95_dr:.1f} exceeds 30 000 ms SLA. "
        f"Per-query dr_lat={dr_lat}"
    )

    # Per-category ratio gates. We only assert when the matching category has
    # at least one valid (baseline > 0, both succeeded) sample — otherwise we
    # silently skip that subset to keep the test resilient to fixture edits.
    ratio_caps = {"disjoint": 6.0, "single_aspect": 3.0}
    failures: list[str] = []
    for r in runnable:
        cap = ratio_caps.get(r.category)
        if cap is None:
            continue
        if r.baseline.latency_ms <= 0:
            continue
        ratio = r.deep_research.latency_ms / r.baseline.latency_ms
        if ratio >= cap:
            failures.append(
                f"{r.id} ({r.category}): dr/baseline={ratio:.2f}x >= cap {cap:.1f}x "
                f"(dr={r.deep_research.latency_ms:.0f}ms, base={r.baseline.latency_ms:.0f}ms)"
            )
    assert not failures, "Latency-ratio cap exceeded:\n  " + "\n  ".join(failures)
