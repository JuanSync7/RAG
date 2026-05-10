"""A/B eval harness for per-topic reranking with split-confidence gate.

For each disjoint multi-topic query, runs deep_research twice — once with
``per_topic_rerank_enabled=False`` (baseline) and once with it enabled
(treatment). Aggregates recall@k delta and p50/p95 latency delta.

Skip behaviour: skips when Weaviate is empty, when the disjoint fixture is
unpopulated, or when RAGChain cannot initialize.

Run with:
    pytest evals/retrieval/deep_research/test_per_topic_rerank_ab.py \
        -m eval_deep_research -v -s

Pass-only-with-report mode (no failure even if recall regresses):
    RAGWEAVE_AB_REPORT_ONLY=1 pytest evals/retrieval/deep_research/test_per_topic_rerank_ab.py \
        -m eval_deep_research -v -s
"""

from __future__ import annotations

import os
import time
from typing import Any

import pytest

import src.retrieval.pipeline.deep_research as dr_module
from src.retrieval.pipeline.deep_research import DeepResearchBudget
from config.settings import WEAVIATE_COLLECTION_NAME

pytestmark = pytest.mark.eval_deep_research


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def _recall_at_k(results: list[Any], expected_sources: list[str]) -> float:
    if not expected_sources:
        return float("nan")
    found = 0
    seen_sources: set[str] = set()
    for r in results:
        md = getattr(r, "metadata", {}) or {}
        src = str(md.get("source", ""))
        for needle in expected_sources:
            if needle and needle in src and needle not in seen_sources:
                seen_sources.add(needle)
                found += 1
                break
    return found / len(expected_sources)


def _patch_budget(monkeypatch, *, enabled: bool) -> dict[str, int]:
    """Patch DeepResearch.__init__ to override per_topic_rerank_enabled on
    whatever budget the caller passes. Returns a dict the test can read for
    counters."""
    state = {"applied": 0}
    original_init = dr_module.DeepResearch.__init__

    def patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        budget = kwargs.get("budget")
        if budget is None and len(args) >= 6:
            # signature: provider, kb, kg, original, processed, budget
            budget = args[5] if len(args) > 5 else None
        if budget is None:
            budget = DeepResearchBudget()
            kwargs["budget"] = budget
        budget.per_topic_rerank_enabled = enabled
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(dr_module.DeepResearch, "__init__", patched_init)

    # Spy on _apply_per_topic_rerank to count how many runs actually fired.
    original_apply = dr_module.DeepResearch._apply_per_topic_rerank

    def spied_apply(self, result):  # type: ignore[no-untyped-def]
        out = original_apply(self, result)
        if out.per_topic_rerank_applied:
            state["applied"] += 1
        return out

    monkeypatch.setattr(dr_module.DeepResearch, "_apply_per_topic_rerank", spied_apply)
    return state


def _run_one(rag_chain, query: str, top_k: int) -> tuple[list[Any], float]:
    t0 = time.perf_counter()
    resp = rag_chain.run(query, deep_research=True, rerank_top_k=top_k)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    docs = list(getattr(resp, "documents", []) or [])
    return docs, elapsed_ms


def test_per_topic_rerank_ab(rag_chain, disjoint_queries_deep_research_asic, monkeypatch):
    """Two-pass A/B over disjoint queries; report recall and latency deltas."""
    # Skip cleanly if there is no corpus.
    try:
        total = _corpus_total(rag_chain)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Cannot read Weaviate aggregate: {exc}")
    if not total:
        pytest.skip("Weaviate collection is empty — cannot run A/B")

    queries = disjoint_queries_deep_research_asic.get("queries") or []
    if not queries:
        pytest.skip("disjoint_queries fixture has no queries")

    top_k = 10
    rows: list[dict[str, Any]] = []

    for q in queries:
        qs = q["query"]
        expected = q.get("expected_sources") or []

        # Baseline: per-topic rerank OFF.
        with monkeypatch.context() as mp:
            _patch_budget(mp, enabled=False)
            try:
                base_docs, base_lat = _run_one(rag_chain, qs, top_k)
                base_err = None
            except Exception as exc:  # noqa: BLE001
                base_docs, base_lat, base_err = [], float("nan"), str(exc)

        # Treatment: per-topic rerank ON.
        with monkeypatch.context() as mp:
            state = _patch_budget(mp, enabled=True)
            try:
                t_docs, t_lat = _run_one(rag_chain, qs, top_k)
                t_err = None
            except Exception as exc:  # noqa: BLE001
                t_docs, t_lat, t_err = [], float("nan"), str(exc)
            fired = state["applied"]

        rows.append({
            "id": q.get("id", qs[:30]),
            "baseline_recall": _recall_at_k(base_docs, expected),
            "treatment_recall": _recall_at_k(t_docs, expected),
            "baseline_lat_ms": base_lat,
            "treatment_lat_ms": t_lat,
            "fired": fired,
            "base_err": base_err,
            "t_err": t_err,
        })

    # Aggregate.
    base_recalls = [r["baseline_recall"] for r in rows if not _isnan(r["baseline_recall"])]
    t_recalls = [r["treatment_recall"] for r in rows if not _isnan(r["treatment_recall"])]
    base_lats = [r["baseline_lat_ms"] for r in rows if not _isnan(r["baseline_lat_ms"])]
    t_lats = [r["treatment_lat_ms"] for r in rows if not _isnan(r["treatment_lat_ms"])]
    fired_count = sum(int(r["fired"]) for r in rows)

    avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    recall_delta = avg(t_recalls) - avg(base_recalls) if t_recalls and base_recalls else float("nan")
    p50_delta = _percentile(t_lats, 50) - _percentile(base_lats, 50)
    p95_delta = _percentile(t_lats, 95) - _percentile(base_lats, 95)

    # Report.
    lines = ["", "=== Per-Topic Rerank A/B Report ===",
             f"queries={len(rows)}  fired_per_topic_rerank={fired_count}/{len(rows)}",
             f"avg recall   baseline={avg(base_recalls):.3f}  treatment={avg(t_recalls):.3f}  delta={recall_delta:+.3f}",
             f"p50 latency  baseline={_percentile(base_lats, 50):.0f}ms  treatment={_percentile(t_lats, 50):.0f}ms  delta={p50_delta:+.0f}ms",
             f"p95 latency  baseline={_percentile(base_lats, 95):.0f}ms  treatment={_percentile(t_lats, 95):.0f}ms  delta={p95_delta:+.0f}ms",
             "", "Per-query:"]
    for r in rows:
        lines.append(
            f"  {r['id']:40s}  base_recall={r['baseline_recall']:.2f} t_recall={r['treatment_recall']:.2f} "
            f"base_lat={r['baseline_lat_ms']:.0f} t_lat={r['treatment_lat_ms']:.0f} fired={r['fired']}"
        )
    report = "\n".join(lines)
    print(report)

    if os.environ.get("RAGWEAVE_AB_REPORT_ONLY") == "1":
        return

    # Default policy: fail if recall regressed by > 0.05 OR p95 latency
    # increased by > 50%.
    if not _isnan(recall_delta) and recall_delta < -0.05:
        pytest.fail(f"Per-topic rerank regressed recall by {recall_delta:+.3f}\n{report}")
    base_p95 = _percentile(base_lats, 95)
    t_p95 = _percentile(t_lats, 95)
    if base_p95 > 0 and t_p95 > base_p95 * 1.5:
        pytest.fail(f"Per-topic rerank p95 latency regressed >50% ({base_p95:.0f}->{t_p95:.0f}ms)\n{report}")


def _isnan(x: float) -> bool:
    return x != x


def _corpus_total(rag_chain) -> int:
    """RAGChain holds the Weaviate client at ``_weaviate_client``; the named
    collection is what every search hits."""
    client = getattr(rag_chain, "_weaviate_client", None)
    if client is None:
        raise RuntimeError("RAGChain has no _weaviate_client")
    col = client.collections.get(WEAVIATE_COLLECTION_NAME)
    return col.aggregate.over_all(total_count=True).total_count or 0


def test_per_topic_rerank_ab_ragweave_corpus(
    rag_chain, disjoint_queries_deep_research_ragweave, monkeypatch
):
    """Latency-only A/B against the live RagWeave engineering-doc corpus.

    Recall is reported as a best-effort prefix match (fixture's
    ``expected_sources`` are directory prefixes like ``guardrails/``) but is
    not gated — the corpus is small and the queries are crafted to exercise
    decomposition latency, not recall. Default policy: report-only unless
    p95 latency regresses by >50%.
    """
    try:
        total = _corpus_total(rag_chain)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Cannot read Weaviate aggregate: {exc}")
    if not total:
        pytest.skip("Weaviate collection is empty — cannot run A/B")

    queries = disjoint_queries_deep_research_ragweave.get("queries") or []
    if not queries:
        pytest.skip("ragweave disjoint fixture has no queries")

    top_k = 10
    rows: list[dict[str, Any]] = []

    for q in queries:
        qs = q["query"]
        expected = q.get("expected_sources") or []

        with monkeypatch.context() as mp:
            _patch_budget(mp, enabled=False)
            try:
                base_docs, base_lat = _run_one(rag_chain, qs, top_k)
                base_err = None
            except Exception as exc:  # noqa: BLE001
                base_docs, base_lat, base_err = [], float("nan"), str(exc)

        with monkeypatch.context() as mp:
            state = _patch_budget(mp, enabled=True)
            try:
                t_docs, t_lat = _run_one(rag_chain, qs, top_k)
                t_err = None
            except Exception as exc:  # noqa: BLE001
                t_docs, t_lat, t_err = [], float("nan"), str(exc)
            fired = state["applied"]

        rows.append({
            "id": q.get("id", qs[:30]),
            "baseline_recall": _recall_at_k(base_docs, expected),
            "treatment_recall": _recall_at_k(t_docs, expected),
            "baseline_lat_ms": base_lat,
            "treatment_lat_ms": t_lat,
            "fired": fired,
            "base_err": base_err,
            "t_err": t_err,
        })

    base_lats = [r["baseline_lat_ms"] for r in rows if not _isnan(r["baseline_lat_ms"])]
    t_lats = [r["treatment_lat_ms"] for r in rows if not _isnan(r["treatment_lat_ms"])]
    fired_count = sum(int(r["fired"]) for r in rows)

    base_recalls = [r["baseline_recall"] for r in rows if not _isnan(r["baseline_recall"])]
    t_recalls = [r["treatment_recall"] for r in rows if not _isnan(r["treatment_recall"])]
    avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731

    p50_delta = _percentile(t_lats, 50) - _percentile(base_lats, 50)
    p95_delta = _percentile(t_lats, 95) - _percentile(base_lats, 95)

    lines = ["", "=== Per-Topic Rerank A/B (RagWeave corpus, latency-only) ===",
             f"queries={len(rows)}  fired_per_topic_rerank={fired_count}/{len(rows)}",
             f"avg recall (best-effort prefix)  base={avg(base_recalls):.3f}  treat={avg(t_recalls):.3f}",
             f"p50 latency  base={_percentile(base_lats, 50):.0f}ms  treat={_percentile(t_lats, 50):.0f}ms  delta={p50_delta:+.0f}ms",
             f"p95 latency  base={_percentile(base_lats, 95):.0f}ms  treat={_percentile(t_lats, 95):.0f}ms  delta={p95_delta:+.0f}ms",
             "", "Per-query:"]
    for r in rows:
        err = r.get("t_err") or r.get("base_err") or ""
        lines.append(
            f"  {r['id']:30s}  base_lat={r['baseline_lat_ms']:.0f}ms t_lat={r['treatment_lat_ms']:.0f}ms "
            f"fired={r['fired']} br={r['baseline_recall']:.2f} tr={r['treatment_recall']:.2f} {err}"
        )
    report = "\n".join(lines)
    print(report)

    if os.environ.get("RAGWEAVE_AB_REPORT_ONLY") == "1":
        return

    base_p95 = _percentile(base_lats, 95)
    t_p95 = _percentile(t_lats, 95)
    if base_p95 > 0 and t_p95 > base_p95 * 1.5:
        pytest.fail(f"Per-topic rerank p95 latency regressed >50% ({base_p95:.0f}->{t_p95:.0f}ms)\n{report}")
