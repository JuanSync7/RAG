# @summary
# P7f — tests for Thresholds.overrides consumption (qtype merge + per-qid floors).
# Covers: ThresholdOverride schema validation (XOR qtype/qid, >=1 numeric
# metric, non-numeric/bool/empty rejection, fail-fast via Thresholds); gate
# qtype-override merge (replace, last-wins, union introduces new qtype/metric),
# boundary semantics; gate qid-override per-query checks (recall/mrr/
# faithfulness metric-key matching, GateFailure.qid set, missing-key skip,
# k-mismatch skip); report per_query_mrr/per_query_faithfulness population.
# Deps: pytest, pydantic ValidationError, src.eval.pack.schema
#       (ThresholdOverride, Thresholds, EvalPack, PackMeta, JudgeConfig,
#       Golden), src.eval.runner.gate, src.eval.runner.report.
# @end-summary
"""P7f — threshold overrides (qtype merge + per-qid floor exceptions)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.eval.pack.schema import (
    EvalPack,
    Golden,
    JudgeConfig,
    PackMeta,
    ThresholdOverride,
    Thresholds,
)
from src.eval.runner.gate import GateFailure, validate_eval_report
from src.eval.runner.report import EvalReport


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_pack(
    defaults: dict[str, dict[str, float]] | None = None,
    *,
    overrides: list[dict] | None = None,
    goldens: dict[str, list[Golden]] | None = None,
) -> EvalPack:
    return EvalPack(
        meta=PackMeta(
            name="p7f-test",
            version=1,
            profile="generic",
            corpus_pin="0" * 40,
            judge=JudgeConfig(
                tier1_model="stub",
                tier1_prompt_version="v0",
                temperature=0.0,
                samples_per_claim=1,
            ),
            collection_name_template="P7F_{name}_{corpus_pin_short}",
        ),
        manifest=[],
        goldens=goldens or {},
        thresholds=Thresholds(
            profile="generic",
            defaults=defaults or {},
            overrides=overrides or [],
        ),
    )


def _make_report(
    *,
    k: int = 5,
    recall_by_qtype: dict[str, float] | None = None,
    faithfulness_by_qtype: dict[str, float] | None = None,
    mrr_by_qtype: dict[str, float] | None = None,
    per_query_recall: dict[str, float] | None = None,
    per_query_mrr: dict[str, float] | None = None,
    per_query_faithfulness: dict[str, float] | None = None,
    total_queries_judged: int = 10,
) -> EvalReport:
    return EvalReport(
        collection_name="P7F_test",
        k=k,
        per_query_recall=per_query_recall or {},
        recall_by_qtype=recall_by_qtype or {},
        total_queries_scored=10,
        total_queries_skipped=0,
        faithfulness_by_qtype=faithfulness_by_qtype or {},
        total_queries_judged=total_queries_judged,
        mrr_by_qtype=mrr_by_qtype or {},
        per_query_mrr=per_query_mrr or {},
        per_query_faithfulness=per_query_faithfulness or {},
    )


def _golden(qid: str, qtype: str) -> Golden:
    return Golden(
        qid=qid,
        qtype=qtype,
        query="q",
        expected_answer_span="span",
        expected_source_docs=["a.md"],
    )


# ---------------------------------------------------------------------------
# A. Schema validation
# ---------------------------------------------------------------------------


def test_override_qtype_form_valid() -> None:
    ov = ThresholdOverride(qtype="factoid", mrr=0.5)
    assert ov.qtype == "factoid"
    assert ov.qid is None
    assert ov.floors() == {"mrr": 0.5}


def test_override_qid_form_valid() -> None:
    ov = ThresholdOverride(qid="factoid_017", recall_at_5=0.4)
    assert ov.qid == "factoid_017"
    assert ov.qtype is None
    assert ov.floors() == {"recall_at_5": 0.4}


def test_override_floors_coerces_int_to_float() -> None:
    ov = ThresholdOverride(qtype="factoid", recall_at_5=1)
    floors = ov.floors()
    assert floors == {"recall_at_5": 1.0}
    assert isinstance(floors["recall_at_5"], float)


def test_override_rejects_both_qtype_and_qid() -> None:
    """MUTATION D TARGET: a validator that allows both-set reds this."""
    with pytest.raises(ValidationError):
        ThresholdOverride(qtype="factoid", qid="factoid_017", mrr=0.5)


def test_override_rejects_neither_qtype_nor_qid() -> None:
    """MUTATION D TARGET: a validator that allows neither-set reds this."""
    with pytest.raises(ValidationError):
        ThresholdOverride(mrr=0.5)


def test_override_rejects_empty_metrics() -> None:
    with pytest.raises(ValidationError):
        ThresholdOverride(qtype="factoid")


def test_override_rejects_non_numeric_metric() -> None:
    with pytest.raises(ValidationError):
        ThresholdOverride(qtype="factoid", mrr="high")


def test_override_rejects_bool_metric() -> None:
    """bool is an int subclass but is NOT a valid floor."""
    with pytest.raises(ValidationError):
        ThresholdOverride(qtype="factoid", mrr=True)


def test_thresholds_fails_fast_on_malformed_override() -> None:
    """A malformed override entry fails fast at Thresholds construction
    (mirrors loader.py / validate.py which both build Thresholds)."""
    with pytest.raises(ValidationError):
        Thresholds(
            profile="generic",
            defaults={},
            overrides=[{"qtype": "factoid", "qid": "x", "mrr": 0.5}],
        )


def test_thresholds_coerces_list_of_dicts() -> None:
    th = Thresholds(
        profile="generic",
        defaults={},
        overrides=[{"qtype": "factoid", "mrr": 0.5}],
    )
    assert len(th.overrides) == 1
    assert isinstance(th.overrides[0], ThresholdOverride)
    assert th.overrides[0].floors() == {"mrr": 0.5}


def test_thresholds_empty_overrides_unaffected() -> None:
    th = Thresholds(profile="generic", defaults={}, overrides=[])
    assert th.overrides == []


# ---------------------------------------------------------------------------
# B. Gate — qtype overrides (merge into effective_floors)
# ---------------------------------------------------------------------------


def test_qtype_override_tightens_floor_causes_fail() -> None:
    """MUTATION A TARGET: gate iterating raw defaults (not merged) reds this.

    default mrr=0.6, override tightens to mrr=0.8, report mrr=0.7 → FAIL.
    """
    pack = _make_pack(
        {"factoid": {"mrr": 0.6}},
        overrides=[{"qtype": "factoid", "mrr": 0.8}],
    )
    report = _make_report(mrr_by_qtype={"factoid": 0.7}, total_queries_judged=0)

    result = validate_eval_report(pack, report)

    assert result.passed is False
    assert len(result.failures) == 1
    fail = result.failures[0]
    assert fail.qtype == "factoid"
    assert fail.metric == "mrr"
    assert fail.expected == pytest.approx(0.8)
    assert fail.actual == pytest.approx(0.7)
    assert fail.qid == ""  # qtype-level failure carries no qid


def test_no_override_same_value_passes() -> None:
    """Discriminating partner: WITHOUT the override the same report PASSES
    (0.7 >= default 0.6). The single value 0.7 discriminates merge-applied."""
    pack = _make_pack({"factoid": {"mrr": 0.6}})
    report = _make_report(mrr_by_qtype={"factoid": 0.7}, total_queries_judged=0)

    result = validate_eval_report(pack, report)

    assert result.passed is True
    assert result.failures == ()


def test_qtype_override_relaxes_floor_causes_pass() -> None:
    """An override can RELAX a floor: default 0.8, override 0.5, report 0.6 PASS."""
    pack = _make_pack(
        {"factoid": {"mrr": 0.8}},
        overrides=[{"qtype": "factoid", "mrr": 0.5}],
    )
    report = _make_report(mrr_by_qtype={"factoid": 0.6}, total_queries_judged=0)

    result = validate_eval_report(pack, report)

    assert result.passed is True
    assert result.failures == ()


def test_qtype_override_boundary_at_floor_passes() -> None:
    """Override floor 0.8, actual exactly 0.8 → PASS (>= semantics)."""
    pack = _make_pack(
        {"factoid": {"mrr": 0.6}},
        overrides=[{"qtype": "factoid", "mrr": 0.8}],
    )
    report = _make_report(mrr_by_qtype={"factoid": 0.8}, total_queries_judged=0)

    result = validate_eval_report(pack, report)

    assert result.passed is True


def test_qtype_override_boundary_below_floor_fails() -> None:
    """Override floor 0.8, actual 0.79 → FAIL."""
    pack = _make_pack(
        {"factoid": {"mrr": 0.6}},
        overrides=[{"qtype": "factoid", "mrr": 0.8}],
    )
    report = _make_report(mrr_by_qtype={"factoid": 0.79}, total_queries_judged=0)

    result = validate_eval_report(pack, report)

    assert result.passed is False
    assert result.failures[0].actual == pytest.approx(0.79)


def test_qtype_override_introduces_new_qtype() -> None:
    """MUTATION C TARGET: iterating defaults keys only (not the union) reds this.

    defaults has no 'adversarial'; an override introduces it; the gate checks it.
    """
    pack = _make_pack(
        {"factoid": {"recall_at_5": 0.8}},
        overrides=[{"qtype": "adversarial", "recall_at_5": 0.9}],
    )
    report = _make_report(
        recall_by_qtype={"factoid": 0.85, "adversarial": 0.5},
        total_queries_judged=0,
    )

    result = validate_eval_report(pack, report)

    assert result.passed is False
    assert len(result.failures) == 1
    fail = result.failures[0]
    assert fail.qtype == "adversarial"
    assert fail.metric == "recall_at_5"
    assert fail.expected == pytest.approx(0.9)
    assert fail.actual == pytest.approx(0.5)


def test_qtype_override_introduces_new_metric() -> None:
    """An override may add a metric absent from defaults for an existing qtype."""
    pack = _make_pack(
        {"factoid": {"recall_at_5": 0.8}},  # no mrr floor in defaults
        overrides=[{"qtype": "factoid", "mrr": 0.7}],
    )
    report = _make_report(
        recall_by_qtype={"factoid": 0.9},  # recall passes
        mrr_by_qtype={"factoid": 0.5},  # below the introduced mrr floor
        total_queries_judged=0,
    )

    result = validate_eval_report(pack, report)

    assert result.passed is False
    assert len(result.failures) == 1
    assert result.failures[0].metric == "mrr"


def test_qtype_override_last_wins_on_duplicate_metric() -> None:
    """MUTATION F TARGET: first-wins reds this.

    Two overrides for the same qtype+metric; the LAST one wins. Last floor
    0.9 > report 0.85 → FAIL; if first-wins (0.5) it would PASS.
    """
    pack = _make_pack(
        {"factoid": {"mrr": 0.6}},
        overrides=[
            {"qtype": "factoid", "mrr": 0.5},  # first
            {"qtype": "factoid", "mrr": 0.9},  # last — wins
        ],
    )
    report = _make_report(mrr_by_qtype={"factoid": 0.85}, total_queries_judged=0)

    result = validate_eval_report(pack, report)

    assert result.passed is False
    assert result.failures[0].expected == pytest.approx(0.9)


def test_qtype_override_does_not_clobber_other_metrics() -> None:
    """A per-metric override replaces only that metric; siblings survive."""
    pack = _make_pack(
        {"factoid": {"recall_at_5": 0.8, "mrr": 0.6}},
        overrides=[{"qtype": "factoid", "mrr": 0.9}],  # only mrr replaced
    )
    report = _make_report(
        recall_by_qtype={"factoid": 0.7},  # still gated at 0.8 → FAIL
        mrr_by_qtype={"factoid": 0.95},  # passes the 0.9 override
        total_queries_judged=0,
    )

    result = validate_eval_report(pack, report)

    assert result.passed is False
    assert len(result.failures) == 1
    assert result.failures[0].metric == "recall_at_5"


# ---------------------------------------------------------------------------
# C. Gate — qid overrides (per-query checks)
# ---------------------------------------------------------------------------


def test_qid_override_recall_below_floor_fails() -> None:
    """MUTATION B TARGET: ignoring qid overrides reds this.

    per_query_recall for factoid_017 is 0.3 < qid floor 0.4 → FAIL with qid set.
    """
    pack = _make_pack(
        {},
        overrides=[{"qid": "factoid_017", "recall_at_5": 0.4}],
        goldens={"factoid": [_golden("factoid_017", "factoid")]},
    )
    report = _make_report(per_query_recall={"factoid_017": 0.3})

    result = validate_eval_report(pack, report)

    assert result.passed is False
    assert len(result.failures) == 1
    fail = result.failures[0]
    assert fail.qid == "factoid_017"
    assert fail.qtype == "factoid"  # resolved from goldens
    assert fail.metric == "recall_at_5"
    assert fail.expected == pytest.approx(0.4)
    assert fail.actual == pytest.approx(0.3)


def test_qid_override_sibling_above_floor_passes() -> None:
    """Discriminating partner: a sibling qid at 0.5 >= floor 0.4 → PASS."""
    pack = _make_pack(
        {},
        overrides=[{"qid": "factoid_018", "recall_at_5": 0.4}],
        goldens={"factoid": [_golden("factoid_018", "factoid")]},
    )
    report = _make_report(per_query_recall={"factoid_018": 0.5})

    result = validate_eval_report(pack, report)

    assert result.passed is True
    assert result.failures == ()


def test_qid_override_mrr_metric() -> None:
    """MUTATION E TARGET: per_query_mrr never populated → this can't fire.

    qid floor on mrr checks report.per_query_mrr.
    """
    pack = _make_pack(
        {},
        overrides=[{"qid": "factoid_020", "mrr": 0.5}],
        goldens={"factoid": [_golden("factoid_020", "factoid")]},
    )
    report = _make_report(per_query_mrr={"factoid_020": 0.25})

    result = validate_eval_report(pack, report)

    assert result.passed is False
    assert result.failures[0].metric == "mrr"
    assert result.failures[0].qid == "factoid_020"
    assert result.failures[0].actual == pytest.approx(0.25)


def test_qid_override_faithfulness_metric() -> None:
    """qid floor on faithfulness checks report.per_query_faithfulness."""
    pack = _make_pack(
        {},
        overrides=[{"qid": "factoid_021", "faithfulness": 0.7}],
        goldens={"factoid": [_golden("factoid_021", "factoid")]},
    )
    report = _make_report(per_query_faithfulness={"factoid_021": 0.4})

    result = validate_eval_report(pack, report)

    assert result.passed is False
    assert result.failures[0].metric == "faithfulness"
    assert result.failures[0].qid == "factoid_021"


def test_qid_override_missing_qid_in_map_is_skipped() -> None:
    """A qid override on a query not in the per-query map (skipped/un-judged)
    is gracefully skipped — no failure."""
    pack = _make_pack(
        {},
        overrides=[{"qid": "factoid_999", "recall_at_5": 0.9}],
        goldens={"factoid": [_golden("factoid_999", "factoid")]},
    )
    report = _make_report(per_query_recall={})  # factoid_999 absent

    result = validate_eval_report(pack, report)

    assert result.passed is True
    assert result.failures == ()


def test_qid_override_k_mismatched_recall_key_is_skipped() -> None:
    """A recall_at_10 qid floor against a k=5 report skips (key-mismatch),
    mirroring the per-qtype skip-on-missing semantics."""
    pack = _make_pack(
        {},
        overrides=[{"qid": "factoid_030", "recall_at_10": 0.9}],
        goldens={"factoid": [_golden("factoid_030", "factoid")]},
    )
    report = _make_report(k=5, per_query_recall={"factoid_030": 0.1})

    result = validate_eval_report(pack, report)

    assert result.passed is True
    assert result.failures == ()


def test_qid_override_unknown_metric_is_skipped() -> None:
    """An unknown metric key on a qid override is skipped, not an error."""
    pack = _make_pack(
        {},
        overrides=[{"qid": "factoid_031", "precision_at_5": 0.9}],
        goldens={"factoid": [_golden("factoid_031", "factoid")]},
    )
    report = _make_report(per_query_recall={"factoid_031": 0.0})

    result = validate_eval_report(pack, report)

    assert result.passed is True
    assert result.failures == ()


def test_qid_override_qtype_falls_back_to_empty_when_not_in_goldens() -> None:
    """If the qid is not in goldens, GateFailure.qtype is '' (still fails)."""
    pack = _make_pack(
        {},
        overrides=[{"qid": "orphan_001", "recall_at_5": 0.5}],
        goldens={},  # no goldens → qid_to_qtype empty
    )
    report = _make_report(per_query_recall={"orphan_001": 0.2})

    result = validate_eval_report(pack, report)

    assert result.passed is False
    assert result.failures[0].qid == "orphan_001"
    assert result.failures[0].qtype == ""


def test_qtype_and_qid_failures_aggregate() -> None:
    """Both qtype-level and qid-level failures are collected together."""
    pack = _make_pack(
        {"factoid": {"recall_at_5": 0.8}},
        overrides=[{"qid": "factoid_040", "mrr": 0.6}],
        goldens={"factoid": [_golden("factoid_040", "factoid")]},
    )
    report = _make_report(
        recall_by_qtype={"factoid": 0.5},  # qtype-level recall fail
        per_query_mrr={"factoid_040": 0.3},  # qid-level mrr fail
        total_queries_judged=0,
    )

    result = validate_eval_report(pack, report)

    assert result.passed is False
    assert len(result.failures) == 2
    keys = {(f.qtype, f.metric, f.qid) for f in result.failures}
    assert ("factoid", "recall_at_5", "") in keys
    assert ("factoid", "mrr", "factoid_040") in keys


# ---------------------------------------------------------------------------
# D. GateFailure additive field
# ---------------------------------------------------------------------------


def test_gatefailure_qid_defaults_to_empty() -> None:
    """GateFailure constructed without qid (existing callers) defaults to ''."""
    f = GateFailure(qtype="factoid", metric="mrr", expected=0.6, actual=0.5)
    assert f.qid == ""
