"""Pytest entry for the deep-research A/B eval.

Marked with `eval_deep_research` so it stays out of default CI runs. Skips
gracefully when the golden set is unpopulated or RAGChain cannot initialize
(e.g., no Weaviate, no models).

Run with:
    pytest evals/retrieval/deep_research/ -m eval_deep_research -v
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.eval_deep_research


def test_baseline_vs_deep_research(rag_chain, golden_queries_deep_research_asic):
    """Run the suite and assert deep_research does not regress on multi_aspect queries."""
    from evals.retrieval.deep_research.harness import run_suite

    report = run_suite(rag_chain, golden_queries_deep_research_asic, top_k=10)

    if report.aggregate["evaluated_count"] == 0:
        pytest.skip("No queries had populated expected_chunks/expected_topics")

    # Soft assertion: on populated multi-aspect/disjoint queries, deep research
    # should match or exceed baseline recall on average. We do NOT assert on
    # single_aspect — deep research is allowed to be equal-but-slower there.
    multi_results = [
        r for r in report.results
        if r.category in ("multi_aspect", "disjoint") and not r.skipped_reason
        and r.baseline and r.deep_research
        and not r.baseline.error and not r.deep_research.error
    ]
    if not multi_results:
        pytest.skip("No multi_aspect/disjoint queries with both modes succeeded")

    baseline_recalls = [r.baseline.recall_at_k for r in multi_results if not math.isnan(r.baseline.recall_at_k)]
    dr_recalls = [r.deep_research.recall_at_k for r in multi_results if not math.isnan(r.deep_research.recall_at_k)]

    if baseline_recalls and dr_recalls:
        baseline_avg = sum(baseline_recalls) / len(baseline_recalls)
        dr_avg = sum(dr_recalls) / len(dr_recalls)
        assert dr_avg >= baseline_avg - 0.05, (
            f"Deep research regressed on multi-aspect queries: "
            f"baseline={baseline_avg:.3f} vs deep_research={dr_avg:.3f}"
        )


def test_messy_queries_recall_within_tolerance(rag_chain, messy_queries_deep_research_asic):
    """DR should survive misspelled / terse / ungrammatical input.

    Acceptance: DR avg recall on the messy set is within 0.10 of baseline avg.
    If DR collapses on noisy queries this asserts loudly — which is the point.
    """
    from evals.retrieval.deep_research.harness import run_suite

    report = run_suite(rag_chain, messy_queries_deep_research_asic, top_k=20)
    if report.aggregate["evaluated_count"] == 0:
        pytest.skip("messy fixture produced no evaluable queries")

    baseline_recalls = [
        r.baseline.recall_at_k for r in report.results
        if r.baseline and not r.baseline.error and not math.isnan(r.baseline.recall_at_k)
    ]
    dr_recalls = [
        r.deep_research.recall_at_k for r in report.results
        if r.deep_research and not r.deep_research.error and not math.isnan(r.deep_research.recall_at_k)
    ]
    if not baseline_recalls or not dr_recalls:
        pytest.skip("messy suite produced no comparable recall numbers")

    baseline_avg = sum(baseline_recalls) / len(baseline_recalls)
    dr_avg = sum(dr_recalls) / len(dr_recalls)
    assert dr_avg >= baseline_avg - 0.10, (
        f"DR collapsed on messy queries: baseline={baseline_avg:.3f} vs dr={dr_avg:.3f}"
    )


def test_efficiency_single_aspect_llm_calls_bounded(
    rag_chain, efficiency_queries_deep_research_asic
):
    """DR must not over-decompose single-aspect queries.

    Threshold: <=12 LLM calls per single_aspect query. Set above the
    observed 4–8 baseline so this asserts only on real regression. If the
    orchestrator starts looping on single-fact lookups, this catches it.
    """
    from evals.retrieval.deep_research.harness import run_suite

    report = run_suite(rag_chain, efficiency_queries_deep_research_asic, top_k=20)
    if report.aggregate["evaluated_count"] == 0:
        pytest.skip("efficiency fixture produced no evaluable queries")

    over = []
    for r in report.results:
        if not r.deep_research or r.deep_research.error:
            continue
        if r.deep_research.llm_calls is None:
            continue
        if r.deep_research.llm_calls > 12:
            over.append((r.id, r.deep_research.llm_calls))
    assert not over, f"DR exceeded 12 LLM calls on single_aspect queries: {over}"
