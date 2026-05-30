# @summary
# Fast tests for P6a/P7e threshold gating (src.eval.runner.gate).
# Covers: pass-all, recall floor red, faithfulness floor red, qtype-absent
# skip, recall-key-absent skip, total_queries_judged=0 anti-gaming guard,
# multi-failure aggregation, frozen-result partner; P7e mrr floor red,
# mrr at-floor/above boundary pass, mrr key-absent skip, mrr qtype-absent
# skip, and mrr-is-not-judged-guarded discriminator.
# Deps: pytest, dataclasses (FrozenInstanceError), src.eval.runner.gate,
#       src.eval.runner.report.EvalReport, src.eval.pack.schema (EvalPack,
#       Thresholds, PackMeta, JudgeConfig).
# @end-summary
"""Threshold gating (P6a) — pure-logic tests."""
from __future__ import annotations

import dataclasses

import pytest

from src.eval.pack.schema import (
    EvalPack,
    JudgeConfig,
    PackMeta,
    Thresholds,
)
from src.eval.runner.gate import GateFailure, GateResult, validate_eval_report
from src.eval.runner.report import EvalReport


def _make_pack(defaults: dict[str, dict[str, float]]) -> EvalPack:
    """Build a minimal EvalPack carrying only the thresholds we need."""
    return EvalPack(
        meta=PackMeta(
            name="p6a-test",
            version=1,
            profile="generic",
            corpus_pin="0" * 40,
            judge=JudgeConfig(
                tier1_model="stub",
                tier1_prompt_version="v0",
                temperature=0.0,
                samples_per_claim=1,
            ),
            collection_name_template="P6A_{name}_{corpus_pin_short}",
        ),
        manifest=[],
        goldens={},
        thresholds=Thresholds(profile="generic", defaults=defaults),
    )


def _make_report(
    *,
    k: int = 5,
    recall_by_qtype: dict[str, float] | None = None,
    faithfulness_by_qtype: dict[str, float] | None = None,
    mrr_by_qtype: dict[str, float] | None = None,
    total_queries_judged: int = 10,
) -> EvalReport:
    return EvalReport(
        collection_name="P6A_test",
        k=k,
        per_query_recall={},
        recall_by_qtype=recall_by_qtype or {},
        total_queries_scored=10,
        total_queries_skipped=0,
        faithfulness_by_qtype=faithfulness_by_qtype or {},
        total_queries_judged=total_queries_judged,
        mrr_by_qtype=mrr_by_qtype or {},
    )


def test_gate_passes_when_all_metrics_above_thresholds() -> None:
    pack = _make_pack(
        {"factoid": {"recall_at_5": 0.8, "faithfulness": 0.6}},
    )
    report = _make_report(
        recall_by_qtype={"factoid": 0.9},
        faithfulness_by_qtype={"factoid": 0.7},
    )

    result = validate_eval_report(pack, report)

    assert isinstance(result, GateResult)
    assert result.passed is True
    assert result.failures == ()


def test_gate_fails_on_recall_below_threshold() -> None:
    pack = _make_pack({"factoid": {"recall_at_5": 0.8}})
    report = _make_report(
        recall_by_qtype={"factoid": 0.7},
        faithfulness_by_qtype={},
        total_queries_judged=0,
    )

    result = validate_eval_report(pack, report)

    assert result.passed is False
    assert len(result.failures) == 1
    fail = result.failures[0]
    assert fail.qtype == "factoid"
    assert fail.metric == "recall_at_5"
    assert fail.expected == pytest.approx(0.8)
    assert fail.actual == pytest.approx(0.7)


def test_gate_fails_on_faithfulness_below_threshold() -> None:
    pack = _make_pack({"factoid": {"faithfulness": 0.6}})
    report = _make_report(
        recall_by_qtype={"factoid": 0.95},
        faithfulness_by_qtype={"factoid": 0.4},
        total_queries_judged=10,
    )

    result = validate_eval_report(pack, report)

    assert result.passed is False
    assert len(result.failures) == 1
    fail = result.failures[0]
    assert fail.qtype == "factoid"
    assert fail.metric == "faithfulness"
    assert fail.expected == pytest.approx(0.6)
    assert fail.actual == pytest.approx(0.4)


def test_gate_skips_qtype_absent_from_thresholds() -> None:
    pack = _make_pack({"factoid": {"recall_at_5": 0.8}})
    # "messy" not declared in thresholds → not gated, even if below 0.8.
    report = _make_report(
        recall_by_qtype={"factoid": 0.9, "messy": 0.1},
        faithfulness_by_qtype={"messy": 0.0},
        total_queries_judged=10,
    )

    result = validate_eval_report(pack, report)

    assert result.passed is True
    assert result.failures == ()


def test_gate_skips_recall_check_when_threshold_key_absent() -> None:
    # Only faithfulness floor declared; no recall_at_K key.
    pack = _make_pack({"factoid": {"faithfulness": 0.6}})
    # Recall is well below 0.8 but should NOT trigger because there's no
    # recall_at_5 floor. Faithfulness must still be enforced.
    report = _make_report(
        k=5,
        recall_by_qtype={"factoid": 0.1},
        faithfulness_by_qtype={"factoid": 0.4},
        total_queries_judged=10,
    )

    result = validate_eval_report(pack, report)

    assert result.passed is False
    # Exactly one failure, and it's faithfulness — recall was not gated.
    assert len(result.failures) == 1
    assert result.failures[0].metric == "faithfulness"


def test_gate_skips_all_faithfulness_when_total_queries_judged_is_zero() -> None:
    pack = _make_pack(
        {"factoid": {"recall_at_5": 0.8, "faithfulness": 0.6}},
    )
    # faithfulness_by_qtype non-empty stale data BELOW floor, but
    # total_queries_judged=0. Anti-gaming guard: skip ALL faithfulness
    # checks (otherwise this would add a second failure). Recall still
    # gated normally.
    report = _make_report(
        recall_by_qtype={"factoid": 0.7},
        faithfulness_by_qtype={"factoid": 0.2},
        total_queries_judged=0,
    )

    result = validate_eval_report(pack, report)

    assert result.passed is False
    assert len(result.failures) == 1
    # The single failure must be recall, not faithfulness (which was skipped).
    fail = result.failures[0]
    assert fail.metric == "recall_at_5"
    assert fail.qtype == "factoid"


def test_gate_aggregates_multiple_failures_across_qtypes() -> None:
    pack = _make_pack(
        {
            "factoid": {"recall_at_5": 0.8, "faithfulness": 0.6},
            "list_lookup": {"recall_at_5": 0.7},
            "definition": {"recall_at_5": 0.75},
        },
    )
    report = _make_report(
        k=5,
        recall_by_qtype={
            "factoid": 0.5,  # recall fail
            "list_lookup": 0.6,  # recall fail
            "definition": 0.9,  # pass
        },
        faithfulness_by_qtype={
            "factoid": 0.3,  # faithfulness fail
        },
        total_queries_judged=10,
    )

    result = validate_eval_report(pack, report)

    assert result.passed is False
    assert len(result.failures) == 3
    # All failures collected (not short-circuited).
    by_key = {(f.qtype, f.metric) for f in result.failures}
    assert ("factoid", "recall_at_5") in by_key
    assert ("list_lookup", "recall_at_5") in by_key
    assert ("factoid", "faithfulness") in by_key


def test_gate_fails_on_mrr_below_threshold() -> None:
    """An mrr floor declared + report mrr below it → a GateFailure(metric=mrr).

    MUTATION A TARGET: removing the gate mrr block reds this.
    """
    pack = _make_pack({"factoid": {"mrr": 0.6}})
    report = _make_report(
        recall_by_qtype={},
        mrr_by_qtype={"factoid": 0.59},
        total_queries_judged=0,  # MRR is retrieval-based, NOT judged-guarded.
    )

    result = validate_eval_report(pack, report)

    assert result.passed is False
    assert len(result.failures) == 1
    fail = result.failures[0]
    assert fail.qtype == "factoid"
    assert fail.metric == "mrr"
    assert fail.expected == pytest.approx(0.6)
    assert fail.actual == pytest.approx(0.59)


def test_gate_passes_on_mrr_at_floor_boundary() -> None:
    """Floor semantics are >=: actual == floor passes; one notch above passes.

    Discriminating partner to the FAIL test above — gives the floor teeth.
    """
    pack = _make_pack({"factoid": {"mrr": 0.6}})

    at_floor = _make_report(
        mrr_by_qtype={"factoid": 0.6}, total_queries_judged=0
    )
    result_eq = validate_eval_report(pack, at_floor)
    assert result_eq.passed is True
    assert result_eq.failures == ()

    above = _make_report(
        mrr_by_qtype={"factoid": 0.61}, total_queries_judged=0
    )
    result_gt = validate_eval_report(pack, above)
    assert result_gt.passed is True
    assert result_gt.failures == ()


def test_gate_skips_mrr_when_threshold_key_absent() -> None:
    """No 'mrr' floor declared → no mrr check even if mrr_by_qtype is below 0.6."""
    pack = _make_pack({"factoid": {"recall_at_5": 0.8}})
    report = _make_report(
        recall_by_qtype={"factoid": 0.9},  # recall passes
        mrr_by_qtype={"factoid": 0.1},  # would fail IF gated, but it isn't
        total_queries_judged=0,
    )

    result = validate_eval_report(pack, report)

    assert result.passed is True
    assert result.failures == ()


def test_gate_skips_mrr_when_qtype_absent_from_report() -> None:
    """'mrr' floor declared but qtype missing from report.mrr_by_qtype → skip."""
    pack = _make_pack({"factoid": {"mrr": 0.6}})
    report = _make_report(
        recall_by_qtype={},
        mrr_by_qtype={},  # factoid absent → graceful skip, no failure
        total_queries_judged=0,
    )

    result = validate_eval_report(pack, report)

    assert result.passed is True
    assert result.failures == ()


def test_gate_mrr_floor_is_not_judged_guarded() -> None:
    """MRR is retrieval-based: it gates even when total_queries_judged == 0.

    Distinguishes mrr from faithfulness (which IS judged-guarded). With
    judged=0 a faithfulness floor would be skipped, but an mrr floor fires.
    """
    pack = _make_pack({"factoid": {"mrr": 0.6, "faithfulness": 0.6}})
    report = _make_report(
        mrr_by_qtype={"factoid": 0.4},  # below mrr floor
        faithfulness_by_qtype={"factoid": 0.1},  # below floor but judged=0
        total_queries_judged=0,
    )

    result = validate_eval_report(pack, report)

    assert result.passed is False
    # Exactly one failure: mrr fires, faithfulness is judged-guarded away.
    assert len(result.failures) == 1
    assert result.failures[0].metric == "mrr"


def test_gate_result_is_frozen() -> None:
    failure = GateFailure(
        qtype="factoid",
        metric="recall_at_5",
        expected=0.8,
        actual=0.7,
    )
    result = GateResult(passed=False, failures=(failure,))

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.passed = True  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        failure.metric = "faithfulness"  # type: ignore[misc]
