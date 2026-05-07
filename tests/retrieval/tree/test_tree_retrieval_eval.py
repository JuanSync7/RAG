"""P3 eval tests for tree-retrieval — metrics, fixture loader, runner, and the
design-doc gate.

Written test-first per TDD. Files imported here may not exist yet on RED.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# P3.1 — metrics
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_full_recall_when_all_gold_in_top_k(self):
        from src.eval.metrics import recall_at_k

        gold = {"a", "b"}
        retrieved = ["a", "b", "c", "d", "e"]
        assert recall_at_k(retrieved, gold, k=5) == pytest.approx(1.0)

    def test_zero_recall_when_no_gold_retrieved(self):
        from src.eval.metrics import recall_at_k

        gold = {"a", "b"}
        retrieved = ["x", "y", "z"]
        assert recall_at_k(retrieved, gold, k=5) == pytest.approx(0.0)

    def test_partial_recall_respects_k_cutoff(self):
        from src.eval.metrics import recall_at_k

        gold = {"a", "b", "c"}
        retrieved = ["a", "b", "x", "y", "c"]
        # k=2 sees only {a,b} → 2/3
        assert recall_at_k(retrieved, gold, k=2) == pytest.approx(2 / 3)

    def test_empty_gold_returns_zero(self):
        from src.eval.metrics import recall_at_k

        assert recall_at_k(["a", "b"], set(), k=5) == 0.0


class TestNDCGAtK:
    def test_perfect_ranking_is_one(self):
        from src.eval.metrics import ndcg_at_k

        gold = {"a", "b", "c"}
        retrieved = ["a", "b", "c", "x", "y"]
        assert ndcg_at_k(retrieved, gold, k=5) == pytest.approx(1.0)

    def test_no_gold_in_top_k_is_zero(self):
        from src.eval.metrics import ndcg_at_k

        assert ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0

    def test_later_rank_lower_score(self):
        from src.eval.metrics import ndcg_at_k

        # gold at rank 1 vs rank 3 — first must be strictly higher
        gold = {"a"}
        score_top = ndcg_at_k(["a", "x", "y", "z"], gold, k=4)
        score_late = ndcg_at_k(["x", "y", "a", "z"], gold, k=4)
        assert score_top > score_late
        # exact: dcg_late = 1/log2(4) ; idcg = 1 → ratio = 1/2
        assert score_late == pytest.approx(1.0 / math.log2(4))


class TestMeanOverQueries:
    def test_mean_recall_across_queries(self):
        from src.eval.metrics import mean_metric

        per_query = [1.0, 0.5, 0.0, 0.5]
        assert mean_metric(per_query) == pytest.approx(0.5)

    def test_empty_returns_zero(self):
        from src.eval.metrics import mean_metric

        assert mean_metric([]) == 0.0


# ---------------------------------------------------------------------------
# P3.3 — fixture loader & eval runner
# ---------------------------------------------------------------------------


class TestFixtureLoader:
    def test_load_synthetic_chip_design_fixture(self):
        from src.eval.fixtures import load_fixture

        path = (
            Path(__file__).resolve().parents[2]
            / "fixtures"
            / "chip_design_spec"
        )
        fixture = load_fixture(path)
        assert fixture.documents, "expected at least one synthetic doc"
        assert fixture.queries, "expected query set"
        assert fixture.gold, "expected gold mapping"
        # every query must have at least one gold chunk id
        for qid, chunks in fixture.gold.items():
            assert chunks, f"empty gold for {qid}"

    def test_fixture_distinguishes_qfs_from_factoid(self):
        from src.eval.fixtures import load_fixture

        path = (
            Path(__file__).resolve().parents[2]
            / "fixtures"
            / "chip_design_spec"
        )
        fixture = load_fixture(path)
        qtypes = {q.qtype for q in fixture.queries}
        assert "qfs" in qtypes, "fixture must include QFS queries"
        assert "factoid" in qtypes, "fixture must include factoid queries"


class TestEvalRunner:
    def test_run_eval_returns_report_with_metrics(self):
        from src.eval.tree_retrieval_eval import run_eval

        # trivial in-memory retriever: returns gold IDs in order for any query.
        queries = [
            {"qid": "q1", "text": "anything", "qtype": "factoid"},
        ]
        gold = {"q1": {"c1", "c2"}}

        def retriever(query_text: str, *, tree_enabled: bool):  # noqa: ARG001
            return ["c1", "c2", "c3"]

        report = run_eval(
            retriever=retriever,
            queries=queries,
            gold=gold,
            tree_enabled=False,
            k_recall=5,
            k_ndcg=10,
        )
        assert report.recall_at_k == pytest.approx(1.0)
        assert report.ndcg_at_k == pytest.approx(1.0)
        assert report.n_queries == 1


# ---------------------------------------------------------------------------
# P3.5 — design-doc gate: tree_on beats tree_off on QFS, ≤ baseline on factoid
# ---------------------------------------------------------------------------


class TestTreeRetrievalGate:
    def _load(self):
        from src.eval.fixtures import load_fixture

        path = (
            Path(__file__).resolve().parents[2]
            / "fixtures"
            / "chip_design_spec"
        )
        return load_fixture(path)

    def test_qfs_recall_at_5_beats_baseline_by_5pp(self):
        from src.eval.tree_retrieval_eval import (
            build_simulated_retriever,
            run_eval,
        )

        fx = self._load()
        retriever = build_simulated_retriever(fx)
        qfs = [q for q in fx.queries if q.qtype == "qfs"]
        gold = {q.qid: fx.gold[q.qid] for q in qfs}

        off = run_eval(
            retriever=retriever,
            queries=[q.__dict__ for q in qfs],
            gold=gold,
            tree_enabled=False,
            k_recall=5,
            k_ndcg=10,
        )
        on = run_eval(
            retriever=retriever,
            queries=[q.__dict__ for q in qfs],
            gold=gold,
            tree_enabled=True,
            k_recall=5,
            k_ndcg=10,
        )
        assert on.recall_at_k >= off.recall_at_k + 0.05, (
            f"QFS recall@5 gate failed: tree_on={on.recall_at_k:.3f} "
            f"tree_off={off.recall_at_k:.3f}"
        )

    def test_qfs_ndcg_at_10_beats_baseline_by_5pp(self):
        from src.eval.tree_retrieval_eval import (
            build_simulated_retriever,
            run_eval,
        )

        fx = self._load()
        retriever = build_simulated_retriever(fx)
        qfs = [q for q in fx.queries if q.qtype == "qfs"]
        gold = {q.qid: fx.gold[q.qid] for q in qfs}

        off = run_eval(
            retriever=retriever,
            queries=[q.__dict__ for q in qfs],
            gold=gold,
            tree_enabled=False,
            k_recall=5,
            k_ndcg=10,
        )
        on = run_eval(
            retriever=retriever,
            queries=[q.__dict__ for q in qfs],
            gold=gold,
            tree_enabled=True,
            k_recall=5,
            k_ndcg=10,
        )
        assert on.ndcg_at_k >= off.ndcg_at_k + 0.05, (
            f"QFS nDCG@10 gate failed: tree_on={on.ndcg_at_k:.3f} "
            f"tree_off={off.ndcg_at_k:.3f}"
        )

    def test_factoid_no_regression(self):
        from src.eval.tree_retrieval_eval import (
            build_simulated_retriever,
            run_eval,
        )

        fx = self._load()
        retriever = build_simulated_retriever(fx)
        factoids = [q for q in fx.queries if q.qtype == "factoid"]
        gold = {q.qid: fx.gold[q.qid] for q in factoids}

        off = run_eval(
            retriever=retriever,
            queries=[q.__dict__ for q in factoids],
            gold=gold,
            tree_enabled=False,
            k_recall=5,
            k_ndcg=10,
        )
        on = run_eval(
            retriever=retriever,
            queries=[q.__dict__ for q in factoids],
            gold=gold,
            tree_enabled=True,
            k_recall=5,
            k_ndcg=10,
        )
        # design doc §8: "≤ baseline on factoid" — allow tiny float wiggle.
        assert on.recall_at_k >= off.recall_at_k - 1e-9, (
            f"Factoid regression: tree_on={on.recall_at_k:.3f} "
            f"tree_off={off.recall_at_k:.3f}"
        )
