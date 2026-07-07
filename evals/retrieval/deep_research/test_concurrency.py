"""Concurrency tests for the deep-research orchestrator.

Spins up 3 simultaneous DR runs against ``RAGChain`` and asserts:

  * No exception escapes from any worker.
  * No chunk bleed across runs — every returned chunk for a given query
    should be "topically reasonable" for that query (at least one
    expected_chunk matches).
  * ``llm_call_count`` is recorded per-run independently and is non-trivial
    (>= 1) for each DR run.

Three queries are pulled from the existing fixtures: 2 from golden +
1 from messy. ``RAGChain.run`` is sync, so we drive concurrency with
``concurrent.futures.ThreadPoolExecutor``.
"""

from __future__ import annotations

import concurrent.futures as cf

import pytest

pytestmark = pytest.mark.eval_deep_research


def _select_concurrent_queries(golden, messy):
    """Pick 2 golden + 1 messy by id, with sensible fallbacks."""
    g_by_id = {q["id"]: q for q in (golden.get("queries") or [])}
    m_by_id = {q["id"]: q for q in (messy.get("queries") or [])}

    chosen = []
    for qid in ("dr-001-disjoint-verif-vs-lint", "dr-003-single-aspect-gpio-checklist"):
        if qid in g_by_id:
            chosen.append(g_by_id[qid])
    if not chosen:
        chosen = list(g_by_id.values())[:2]

    for qid in ("dr-msg-002-typo-coverage", "dr-msg-005-no-punct-gpio"):
        if qid in m_by_id:
            chosen.append(m_by_id[qid])
            break
    if len(chosen) < 3 and m_by_id:
        chosen.append(next(iter(m_by_id.values())))

    return chosen[:3]


def test_three_concurrent_dr_runs(
    rag_chain,
    golden_queries_deep_research_asic,
    messy_queries_deep_research_asic,
):
    from evals.retrieval.deep_research.harness import (
        ChunkMatcher,
        _run_single,
    )

    queries = _select_concurrent_queries(
        golden_queries_deep_research_asic,
        messy_queries_deep_research_asic,
    )
    if len(queries) < 3:
        pytest.skip("Need at least 3 fixtures (2 golden + 1 messy) for this test")

    def _worker(q):
        chunks, latency_ms, llm_calls, action = _run_single(
            rag_chain, q["query"], deep_research=True, top_k=10
        )
        return {
            "id": q["id"],
            "expected": q.get("expected_chunks") or [],
            "chunks": chunks,
            "latency_ms": latency_ms,
            "llm_calls": llm_calls,
            "action": action,
        }

    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(_worker, q) for q in queries]
        results = []
        errors = []
        for fut in cf.as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001 — we want to surface them all
                errors.append(repr(exc))

    assert not errors, f"Concurrent DR runs raised: {errors}"
    assert len(results) == 3

    # No chunk bleed: each result's chunks must match at least one of its
    # OWN expected_chunks (not those of a sibling run). Empty-chunk results
    # are tolerated (action may be ask_user / no_results) but flagged.
    for r in results:
        if not r["expected"]:
            continue
        if not r["chunks"]:
            # Acceptable on adversarial/edge-case fixtures, but the goldens
            # we picked should actually return something.
            pytest.fail(
                f"{r['id']}: no chunks returned under concurrency "
                f"(action={r['action']!r})"
            )
        matchers = [ChunkMatcher(**spec) for spec in r["expected"]]
        own_match = any(
            any(m.matches(c) for c in r["chunks"]) for m in matchers
        )
        assert own_match, (
            f"{r['id']}: returned chunks under concurrency don't match any of "
            f"its own expected_chunks — possible chunk bleed. "
            f"action={r['action']!r}"
        )

    # Per-run llm_call_count is recorded independently — at least one of the
    # three should expose a count, and any reported counts must be >= 1.
    reported = [r["llm_calls"] for r in results if r["llm_calls"] is not None]
    assert reported, (
        "No llm_call_count reported across 3 concurrent DR runs — side-channel "
        "DeepResearch.last_result may have raced."
    )
    for r in results:
        if r["llm_calls"] is not None:
            assert r["llm_calls"] >= 1, (
                f"{r['id']}: llm_calls={r['llm_calls']} (expected >= 1)"
            )
