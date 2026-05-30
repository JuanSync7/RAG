# @summary
# P8a.0 — tests for the reusable run_eval(...) core extracted from run_cli.
# Asserts run_eval RETURNS an EvalRunResult (report + gate + collection_name +
# faithfulness_results + pack) rather than an exit code, and that it RAISES on
# pack errors instead of mapping them to exit codes.
# Reuses _make_synthetic_pack / _patch_full_chain from tests.eval.test_cli.
# Exports: test_run_eval_returns_report_and_gate,
#          test_run_eval_gate_fails_on_low_metrics,
#          test_run_eval_raises_on_bad_pack
# Deps: src.eval.run (run_eval, EvalRunResult); src.eval.runner; tests.eval.test_cli
# @end-summary
"""P8a.0 — run_eval core tests (returns result objects, raises on bad pack)."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.eval.test_cli import _make_synthetic_pack, _patch_full_chain


def test_run_eval_returns_report_and_gate(tmp_path: Path, monkeypatch) -> None:
    """run_eval returns an EvalRunResult with a populated report + passing gate."""
    from src.eval import EvalRunResult, run_eval
    from src.eval.runner import EvalReport, GateResult

    pack_dir = _make_synthetic_pack(
        tmp_path, recall_floor=0.5, faithfulness_floor=0.5
    )
    _patch_full_chain(monkeypatch, judge_score=0.9)

    result = run_eval(str(pack_dir))

    assert isinstance(result, EvalRunResult)
    assert isinstance(result.report, EvalReport)
    assert isinstance(result.gate, GateResult)
    # recall=1.0 (retrieval hits doc1.md), faithfulness=0.9 → both above floor.
    assert result.report.recall_by_qtype["factoid"] == 1.0
    assert result.gate.passed is True
    assert result.collection_name
    assert result.collection_name == result.report.collection_name
    # faithfulness_results is threaded through (needed for --show-samples).
    assert result.faithfulness_results is not None


def test_run_eval_gate_fails_on_low_metrics(tmp_path: Path, monkeypatch) -> None:
    """A retrieval miss drives recall below floor → gate fails (one-step-below
    partner to the pass test)."""
    from src.eval import run_eval

    pack_dir = _make_synthetic_pack(
        tmp_path, recall_floor=0.5, faithfulness_floor=0.5
    )
    # Retrieval returns the wrong source → recall=0.0 < 0.5 floor.
    _patch_full_chain(monkeypatch, retrieved_source="other.md", judge_score=0.9)

    result = run_eval(str(pack_dir))

    assert result.gate.passed is False
    assert result.gate.failures, "expected at least one GateFailure"
    metrics = {f.metric for f in result.gate.failures}
    assert "recall_at_5" in metrics


def test_run_eval_raises_on_bad_pack(tmp_path: Path) -> None:
    """A nonexistent pack path RAISES (PackValidationError/FileNotFoundError),
    it does NOT return an int exit code."""
    from src.eval import run_eval
    from src.eval.pack import PackValidationError

    bogus = tmp_path / "no_such_pack"
    with pytest.raises((PackValidationError, FileNotFoundError)):
        run_eval(str(bogus))
