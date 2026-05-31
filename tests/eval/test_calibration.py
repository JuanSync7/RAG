# @summary
# P9 — tests for the judge calibration loop (src/eval/runner/calibration.py).
# Pins: Cohen's kappa arithmetic against a hand-computed oracle (kappa=0.4 with
# confusion tp=4/tn=3/fp=2/fn=1, agreement=0.7), perfect/total-disagreement and
# the pe==1 all-one-class edge (kappa=1.0, not nan), binarize boundary
# inclusivity, strict drift detection, log-only drift alert (WARNING on drift /
# INFO otherwise), deterministic persistence round-trip with exact values, and
# frozen-dataclass immutability.
# Exports: test_kappa_on_seeded_disagreement and the rest of the P9 matrix.
# Deps: src.eval.runner.calibration (module under test); stdlib (json,
#       dataclasses, logging, datetime, pathlib); pytest.
# @end-summary
"""P9 — judge calibration loop tests (TDD-first)."""
from __future__ import annotations

import dataclasses
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Deterministic injected clock (mirror tests/eval/test_orchestrator_e2e.py:34).
FIXED_NOW = datetime(2026, 5, 30, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Test 1 — the kappa ORACLE: 10 seeded samples, confusion tp=4/tn=3/fp=2/fn=1.
# ---------------------------------------------------------------------------
def test_kappa_on_seeded_disagreement() -> None:
    """Hand-computed oracle. With threshold=0.5 and judge passes=0.9 / fails=0.1:

    Construct exactly confusion (vs reference, "pass" positive):
      tp=4 (judge pass, ref pass), tn=3 (judge fail, ref fail),
      fp=2 (judge pass, ref fail), fn=1 (judge fail, ref pass).
      po (agreement) = (tp+tn)/n = 7/10 = 0.7
      marginals: judge pass = tp+fp = 6, ref pass = tp+fn = 5
      pe = (6/10)(5/10) + (4/10)(5/10) = 0.30 + 0.20 = 0.50
      kappa = (0.7 - 0.5)/(1 - 0.5) = 0.2/0.5 = 0.40
    kappa(0.4) != agreement(0.7), so an agreement-in-place-of-kappa mutation reds.
    """
    from src.eval.runner.calibration import compute_calibration

    # 4 tp, 3 tn, 2 fp, 1 fn  -> all four quadrants distinct.
    judge_scores = [
        0.9, 0.9, 0.9, 0.9,   # judge pass ...
        0.9, 0.9,             # ... (fp: judge pass)
        0.1, 0.1, 0.1,        # judge fail (tn)
        0.1,                  # judge fail (fn)
    ]
    reference_labels = [
        "pass", "pass", "pass", "pass",  # tp x4
        "fail", "fail",                  # fp x2 (judge pass, ref fail)
        "fail", "fail", "fail",          # tn x3 (judge fail, ref fail)
        "pass",                          # fn x1 (judge fail, ref pass)
    ]

    result = compute_calibration(
        judge_scores, reference_labels, threshold=0.5, now=FIXED_NOW
    )

    assert result.kappa == pytest.approx(0.4)
    assert result.agreement_rate == pytest.approx(0.7)
    assert result.confusion_counts == {"tp": 4, "tn": 3, "fp": 2, "fn": 1}
    assert result.n == 10
    assert result.threshold == 0.5
    assert result.timestamp == FIXED_NOW.isoformat()


# ---------------------------------------------------------------------------
# Test 2 — perfect agreement => kappa == 1.0, incl. pe==1 all-one-class edge.
# ---------------------------------------------------------------------------
def test_perfect_agreement_kappa_is_one() -> None:
    from src.eval.runner.calibration import compute_calibration

    judge_scores = [0.9, 0.1, 0.9, 0.1]
    reference_labels = ["pass", "fail", "pass", "fail"]
    result = compute_calibration(
        judge_scores, reference_labels, threshold=0.5
    )
    assert result.kappa == pytest.approx(1.0)
    assert result.agreement_rate == pytest.approx(1.0)


def test_all_one_class_edge_kappa_is_one_not_nan() -> None:
    """When both raters put everything in one class, pe == 1.0; the 0/0 case is
    defined as kappa = 1.0 (not nan / not ZeroDivisionError)."""
    from src.eval.runner.calibration import cohens_kappa

    labels = ["pass"] * 5
    k = cohens_kappa(labels, labels)
    assert k == 1.0  # not nan, not a raise


# ---------------------------------------------------------------------------
# Test 3 — total disagreement => kappa < 0 (negative).
# ---------------------------------------------------------------------------
def test_total_disagreement_kappa_is_negative() -> None:
    from src.eval.runner.calibration import cohens_kappa

    a = ["pass", "fail", "pass", "fail"]
    b = ["fail", "pass", "fail", "pass"]
    assert cohens_kappa(a, b) < 0


# ---------------------------------------------------------------------------
# Test 4 — cohens_kappa length mismatch => ValueError.
# ---------------------------------------------------------------------------
def test_cohens_kappa_length_mismatch_raises() -> None:
    from src.eval.runner.calibration import cohens_kappa

    with pytest.raises(ValueError):
        cohens_kappa(["pass", "fail"], ["pass"])


def test_compute_calibration_length_mismatch_raises() -> None:
    from src.eval.runner.calibration import compute_calibration

    with pytest.raises(ValueError):
        compute_calibration([0.9, 0.1], ["pass"], threshold=0.5)


# ---------------------------------------------------------------------------
# Test 5 — binarize boundary is INCLUSIVE (>=), discriminating partner below.
# ---------------------------------------------------------------------------
def test_binarize_boundary_is_inclusive() -> None:
    from src.eval.runner.calibration import binarize_score

    assert binarize_score(0.5, 0.5) == "pass"          # == threshold -> pass
    assert binarize_score(0.5 - 1e-9, 0.5) == "fail"   # one step below -> fail
    assert binarize_score(0.9, 0.5) == "pass"
    assert binarize_score(0.1, 0.5) == "fail"


# ---------------------------------------------------------------------------
# Test 6 — detect_drift strictness: kappa == min_kappa -> False; below -> True.
# ---------------------------------------------------------------------------
def test_detect_drift_strict_boundary() -> None:
    from src.eval.runner.calibration import detect_drift

    assert detect_drift(0.6, min_kappa=0.6) is False        # equal -> no drift
    assert detect_drift(0.6 - 1e-9, min_kappa=0.6) is True   # below -> drift
    assert detect_drift(0.9, min_kappa=0.6) is False
    assert detect_drift(0.2, min_kappa=0.6) is True


# ---------------------------------------------------------------------------
# Test 7 — drift alert fires (WARNING on drift, INFO + no WARNING otherwise).
# ---------------------------------------------------------------------------
def test_drift_alert_fires_warning(caplog) -> None:
    from src.eval.runner.calibration import (
        CalibrationAlert,
        send_calibration_alert,
    )

    payload = CalibrationAlert(
        pack_name="opentitan",
        qtype="factoid",
        kappa=0.30,
        min_kappa=0.60,
        drift_detected=True,
        timestamp=FIXED_NOW.isoformat(),
        record_path="/tmp/opentitan_cal.json",
    )
    with caplog.at_level(logging.WARNING, logger="rag.eval.calibration"):
        assert send_calibration_alert(payload) is None

    warnings = [
        r for r in caplog.records if r.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert "DRIFT" in warnings[0].getMessage()


def test_no_drift_alert_is_info_not_warning(caplog) -> None:
    from src.eval.runner.calibration import (
        CalibrationAlert,
        send_calibration_alert,
    )

    payload = CalibrationAlert(
        pack_name="opentitan",
        qtype="factoid",
        kappa=0.90,
        min_kappa=0.60,
        drift_detected=False,
        timestamp=FIXED_NOW.isoformat(),
    )
    with caplog.at_level(logging.INFO, logger="rag.eval.calibration"):
        send_calibration_alert(payload)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert warnings == []
    assert len(infos) == 1
    assert "OK" in infos[0].getMessage()


# ---------------------------------------------------------------------------
# Test 8 — persist round-trip: deterministic filename + EXACT values back.
# ---------------------------------------------------------------------------
def test_persist_calibration_record_roundtrip(tmp_path: Path) -> None:
    from src.eval.runner.calibration import (
        CalibrationRecord,
        CalibrationResult,
        persist_calibration_record,
    )

    result = CalibrationResult(
        kappa=0.4,
        n=10,
        agreement_rate=0.7,
        confusion_counts={"tp": 4, "tn": 3, "fp": 2, "fn": 1},
        threshold=0.5,
        timestamp=FIXED_NOW.isoformat(),
    )
    record = CalibrationRecord(
        pack_name="opentitan",
        qtype="factoid",
        min_kappa=0.6,
        drift_detected=True,
        result=result,
    )

    out_dir = tmp_path / "calibration"
    path = persist_calibration_record(record, out_dir, now=FIXED_NOW)

    # Deterministic filename: {pack_name}_{filesafe_ts}.json (no colons).
    filesafe_ts = FIXED_NOW.isoformat().replace(":", "")
    assert path.name == f"opentitan_{filesafe_ts}.json"
    assert ":" not in path.name
    assert path.exists()

    payload = json.loads(path.read_text(encoding="utf-8"))

    # EXACT values (serialization-test-teeth: not mere key-presence).
    assert payload["pack_name"] == "opentitan"
    assert payload["qtype"] == "factoid"
    assert payload["min_kappa"] == 0.6
    assert payload["drift_detected"] is True
    assert payload["result"]["kappa"] == 0.4
    assert payload["result"]["n"] == 10
    assert payload["result"]["agreement_rate"] == 0.7
    assert payload["result"]["confusion_counts"] == {
        "tp": 4, "tn": 3, "fp": 2, "fn": 1
    }
    assert payload["result"]["threshold"] == 0.5
    assert payload["result"]["timestamp"] == FIXED_NOW.isoformat()


# ---------------------------------------------------------------------------
# Test 9 — frozen dataclasses (mirror test_judge_client.py:226-237 style).
# ---------------------------------------------------------------------------
def test_calibration_result_is_frozen() -> None:
    from src.eval.runner.calibration import CalibrationResult

    result = CalibrationResult(
        kappa=0.4,
        n=10,
        agreement_rate=0.7,
        confusion_counts={"tp": 4, "tn": 3, "fp": 2, "fn": 1},
        threshold=0.5,
        timestamp=FIXED_NOW.isoformat(),
    )
    assert dataclasses.is_dataclass(result)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.kappa = 0.99  # type: ignore[misc]
