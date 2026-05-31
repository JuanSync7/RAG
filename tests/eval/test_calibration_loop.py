# @summary
# P9 — tests for the one-shot judge-calibration loop
# (src/eval/runner/calibration_loop.py). Pins the orchestrator's composition:
# a hand-computed kappa oracle through run_calibration (confusion tp=4/tn=3/
# fp=2/fn=1 -> kappa=0.4, agreement=0.7, drift True vs min_kappa 0.6), the
# real-fixture one-shot baseline + persistence wiring (judge scores every
# example once, kappa=1.0, exactly one JSON written), the alert carrying the
# persisted record path, the end-to-end WARNING through the real log sink on
# drift, and that the binarization threshold is load-bearing (different
# thresholds -> different confusion), the facade re-export contract, and that
# `now` is resolved once so the record timestamp / filename / alert timestamp
# agree on the production path.
# Exports: test_run_calibration_seeded_oracle and the rest of the loop matrix.
# Deps: src.eval.runner.calibration_loop (module under test);
#       src.eval.runner.calibration_fixture (CalibrationExample),
#       src.eval.runner.judge (JudgmentScore), src.eval.runner
#       (load_calibration_fixture, send_calibration_alert); stdlib + pytest.
# @end-summary
"""P9 — one-shot judge-calibration loop tests (TDD-first)."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.eval.runner.calibration_fixture import CalibrationExample

# Deterministic injected clock (mirror tests/eval/test_calibration.py:25).
FIXED_NOW = datetime(2026, 5, 30, 12, 0, 0, tzinfo=timezone.utc)


class _StubJudge:
    """Deterministic judge keyed by qid; records every question it scores."""

    def __init__(self, scores_by_qid: dict[str, float]) -> None:
        self.scores_by_qid = scores_by_qid
        self.calls: list = []

    def score(self, question):
        from src.eval.runner.judge import JudgmentScore

        self.calls.append(question)
        return JudgmentScore(
            score=self.scores_by_qid[question.qid], reasoning="stub"
        )


def _example(qid: str, reference_label: str) -> CalibrationExample:
    """Build a minimal calibration example (chunk text is irrelevant — the stub
    judge keys purely off qid)."""
    return CalibrationExample(
        qid=qid,
        qtype="factoid",
        query=f"query for {qid}",
        expected_answer_span="span",
        chunk_texts=("chunk text",),
        reference_label=reference_label,
    )


def _oracle_examples() -> list[CalibrationExample]:
    """10 examples that, scored by a 0.5-threshold judge giving pass-refs 0.9 and
    a couple of fail-refs 0.9 (false positives) and the rest 0.1, yield exactly
    confusion tp=4/tn=3/fp=2/fn=1 (mirror test_calibration::oracle, kappa=0.4)."""
    return [
        _example("cal-tp-1", "pass"),
        _example("cal-tp-2", "pass"),
        _example("cal-tp-3", "pass"),
        _example("cal-tp-4", "pass"),
        _example("cal-fp-1", "fail"),
        _example("cal-fp-2", "fail"),
        _example("cal-tn-1", "fail"),
        _example("cal-tn-2", "fail"),
        _example("cal-tn-3", "fail"),
        _example("cal-fn-1", "pass"),
    ]


def _oracle_scores() -> dict[str, float]:
    return {
        "cal-tp-1": 0.9, "cal-tp-2": 0.9, "cal-tp-3": 0.9, "cal-tp-4": 0.9,
        "cal-fp-1": 0.9, "cal-fp-2": 0.9,        # judge pass, ref fail -> fp
        "cal-tn-1": 0.1, "cal-tn-2": 0.1, "cal-tn-3": 0.1,
        "cal-fn-1": 0.1,                          # judge fail, ref pass -> fn
    }


# ---------------------------------------------------------------------------
# Test 1 — seeded kappa oracle through the full orchestrator (injected examples,
# robust to fixture edits).
# ---------------------------------------------------------------------------
def test_run_calibration_seeded_oracle(tmp_path: Path) -> None:
    from src.eval.runner.calibration_loop import run_calibration

    stub = _StubJudge(_oracle_scores())
    record = run_calibration(
        stub,
        pack_name="oraclepack",
        threshold=0.5,
        min_kappa=0.6,
        output_dir=tmp_path,
        examples=_oracle_examples(),
        now=FIXED_NOW,
    )

    assert record.result.kappa == pytest.approx(0.4)
    assert record.result.agreement_rate == pytest.approx(0.7)
    assert record.result.confusion_counts == {"tp": 4, "tn": 3, "fp": 2, "fn": 1}
    assert record.result.n == 10
    assert record.drift_detected is True  # 0.4 < 0.6
    assert record.pack_name == "oraclepack"
    assert record.qtype == "all"
    assert record.min_kappa == 0.6


# ---------------------------------------------------------------------------
# Test 2 — one-shot baseline on the REAL pinned fixture (proves the production
# load path + persistence).
# ---------------------------------------------------------------------------
def test_run_calibration_one_shot_baseline_on_real_fixture(tmp_path: Path) -> None:
    from src.eval.runner import load_calibration_fixture
    from src.eval.runner.calibration_loop import run_calibration

    fx = load_calibration_fixture()
    # Judge perfectly agrees with the reference.
    scores_by_qid = {
        ex.qid: (0.9 if ex.reference_label == "pass" else 0.1) for ex in fx
    }
    stub = _StubJudge(scores_by_qid)

    record = run_calibration(
        stub,
        pack_name="opentitan",
        threshold=0.5,
        min_kappa=0.6,
        output_dir=tmp_path,
        now=FIXED_NOW,  # examples=None -> loads the real fixture
    )

    assert record.result.n == len(fx)
    assert record.result.kappa == pytest.approx(1.0)
    assert record.drift_detected is False
    assert len(stub.calls) == len(fx)  # every example scored exactly once

    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1


# ---------------------------------------------------------------------------
# Test 3 — the alert carries the persisted record path + mirrors the record.
# ---------------------------------------------------------------------------
def test_run_calibration_alert_carries_record_path(tmp_path: Path) -> None:
    from src.eval.runner.calibration_loop import run_calibration

    captured: list = []
    stub = _StubJudge(_oracle_scores())
    record = run_calibration(
        stub,
        pack_name="oraclepack",
        threshold=0.5,
        min_kappa=0.6,
        output_dir=tmp_path,
        examples=_oracle_examples(),
        alert_fn=captured.append,
        now=FIXED_NOW,
    )

    assert len(captured) == 1
    alert = captured[0]
    assert alert.record_path is not None

    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1
    assert Path(alert.record_path) == written[0]

    assert alert.kappa == record.result.kappa
    assert alert.drift_detected == record.drift_detected
    assert alert.pack_name == record.pack_name


# ---------------------------------------------------------------------------
# Test 4 — drift fires exactly one WARNING through the REAL alert sink.
# ---------------------------------------------------------------------------
def test_run_calibration_drift_fires_warning(tmp_path: Path, caplog) -> None:
    from src.eval.runner.calibration_loop import run_calibration

    stub = _StubJudge(_oracle_scores())  # kappa 0.4 < min_kappa 0.6 -> drift
    with caplog.at_level(logging.WARNING, logger="rag.eval.calibration"):
        run_calibration(
            stub,
            pack_name="oraclepack",
            threshold=0.5,
            min_kappa=0.6,
            output_dir=tmp_path,
            examples=_oracle_examples(),
            now=FIXED_NOW,
            # NOTE: alert_fn intentionally NOT injected -> real send_calibration_alert
        )

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "DRIFT" in warnings[0].getMessage()


# ---------------------------------------------------------------------------
# Test 5 — the binarization threshold is load-bearing (different thresholds ->
# different confusion), proving `threshold` flows into compute_calibration.
# ---------------------------------------------------------------------------
def test_run_calibration_threshold_changes_binarization(tmp_path: Path) -> None:
    from src.eval.runner.calibration_loop import run_calibration

    # One example scores 0.7 — "pass" at threshold 0.5, "fail" at threshold 0.95.
    examples = [
        _example("mid-1", "pass"),  # judge 0.7: pass@0.5 (tp) / fail@0.95 (fn)
        _example("hi-1", "pass"),   # judge 0.99: pass at both thresholds (tp)
        _example("lo-1", "fail"),   # judge 0.1: fail at both thresholds (tn)
    ]
    scores = {"mid-1": 0.7, "hi-1": 0.99, "lo-1": 0.1}

    low = run_calibration(
        _StubJudge(scores),
        pack_name="thr",
        threshold=0.5,
        min_kappa=0.6,
        output_dir=tmp_path / "low",
        examples=examples,
        now=FIXED_NOW,
    )
    high = run_calibration(
        _StubJudge(scores),
        pack_name="thr",
        threshold=0.95,
        min_kappa=0.6,
        output_dir=tmp_path / "high",
        examples=examples,
        now=FIXED_NOW,
    )

    assert low.result.confusion_counts != high.result.confusion_counts
    assert low.result.confusion_counts == {"tp": 2, "tn": 1, "fp": 0, "fn": 0}
    assert high.result.confusion_counts == {"tp": 1, "tn": 1, "fp": 0, "fn": 1}


# ---------------------------------------------------------------------------
# Test 6 — the stable import surface re-exports run_calibration (gives the
# __init__/__all__ facade contract teeth — removing the re-export reds this).
# ---------------------------------------------------------------------------
def test_run_calibration_is_exported_from_facade() -> None:
    from src.eval.runner import run_calibration as facade_run
    from src.eval.runner.calibration_loop import run_calibration as module_run

    assert facade_run is module_run


# ---------------------------------------------------------------------------
# Test 7 — on the production path (now=None) the record timestamp, the persisted
# filename, and the alert timestamp all agree. `now` must be resolved ONCE; if it
# were read independently by compute_calibration and persist they would diverge at
# microsecond resolution and the filename would not match result.timestamp.
# ---------------------------------------------------------------------------
def test_run_calibration_resolves_now_once(tmp_path: Path) -> None:
    from src.eval.runner.calibration_loop import run_calibration

    captured: list = []
    record = run_calibration(
        _StubJudge(_oracle_scores()),
        pack_name="nowpack",
        threshold=0.5,
        min_kappa=0.6,
        output_dir=tmp_path,
        examples=_oracle_examples(),
        alert_fn=captured.append,
        # now=None -> production path: the orchestrator resolves the clock once.
    )

    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1

    alert = captured[0]
    assert alert.timestamp == record.result.timestamp
    # Filename is the SAME instant with colons stripped -> all three agree.
    expected_name = f"nowpack_{record.result.timestamp.replace(':', '')}.json"
    assert written[0].name == expected_name
