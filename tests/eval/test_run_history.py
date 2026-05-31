# @summary
# Tests for the per-pack append-only run-history index + per-qid regression
# deltas vs baseline (src.eval.runner.run_history). Headline red:
# test_persist_and_index_run_history drives a baseline-vs-regressed pair through
# index_run_report and asserts the JSONL index shape, ordering, and hand-computed
# per-qid deltas (sign convention = current - baseline). Unit tests pin the
# epsilon boundary on compute_per_query_deltas, JSONL append order, timestamp
# sorting on load_run_history, and the first-run (no baseline) empty-delta path.
# Exports: test_persist_and_index_run_history, test_compute_per_query_deltas,
#          test_index_jsonl_format_and_append, test_load_run_history_sorts_by_timestamp,
#          test_first_run_no_baseline_yields_empty_deltas
# Deps: src.eval.runner.run_history; src.eval.runner.persistence; tests.eval.test_cli
# @end-summary
"""Tests for the run-history index + per-qid regression deltas."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.eval.test_cli import _make_synthetic_pack, _patch_full_chain

FIXED_NOW_1 = datetime(2026, 5, 30, 12, 0, 0, tzinfo=timezone.utc)
FIXED_NOW_2 = datetime(2026, 5, 31, 12, 0, 0, tzinfo=timezone.utc)


def _run(tmp_path: Path, monkeypatch, *, judge_score: float = 0.9):
    from src.eval import run_eval

    pack_dir = _make_synthetic_pack(
        tmp_path, recall_floor=0.5, faithfulness_floor=0.5
    )
    _patch_full_chain(monkeypatch, retrieved_source="doc1.md", judge_score=judge_score)
    return run_eval(str(pack_dir))


# ---------------------------------------------------------------------------
# Headline RED — e2e index across two runs
# ---------------------------------------------------------------------------


def test_persist_and_index_run_history(tmp_path: Path, monkeypatch) -> None:
    """Two runs (baseline then regressed) indexed → one ordered JSONL per run.

    Run 1 is the baseline payload; run 2 is a regressed variant. We index BOTH
    against the baseline payload and assert:
      (a) <history_dir>/<pack>.history.jsonl exists,
      (b) it has one JSONL line per run, in timestamp order,
      (c) each entry carries timestamp/pack_name/gate_passed/per_query_deltas
          (qid -> {metric -> delta}) and per_qtype_deltas,
      (d) per-qid deltas equal hand-computed values for >=2 queries (exact +
          sign).
    """
    from src.eval.runner.run_history import index_run_report

    history_dir = tmp_path / "reports"

    # Hand-built baseline + current payloads with two qids so we can check >=2
    # deltas with exact sign. recall regresses (-0.4), faithfulness improves
    # (+0.1) for g1; recall steady for g2, mrr regresses for g2.
    baseline_payload = {
        "pack_name": "synpack",
        "timestamp": FIXED_NOW_1.isoformat(),
        "passed": True,
        "k": 5,
        "recall_by_qtype": {"factoid": 1.0},
        "mrr_by_qtype": {"factoid": 1.0},
        "faithfulness_by_qtype": {"factoid": 0.8},
        "per_query_recall": {"g1": 1.0, "g2": 1.0},
        "per_query_mrr": {"g1": 1.0, "g2": 1.0},
        "per_query_faithfulness": {"g1": 0.8, "g2": 0.9},
    }
    current_payload = {
        "pack_name": "synpack",
        "timestamp": FIXED_NOW_2.isoformat(),
        "passed": True,
        "k": 5,
        "recall_by_qtype": {"factoid": 0.8},
        "mrr_by_qtype": {"factoid": 0.75},
        "faithfulness_by_qtype": {"factoid": 0.9},
        "per_query_recall": {"g1": 0.6, "g2": 1.0},
        "per_query_mrr": {"g1": 1.0, "g2": 0.5},
        "per_query_faithfulness": {"g1": 0.9, "g2": 0.9},
    }

    class _FakeResult:
        def __init__(self, payload):
            self._payload = payload

    # Run 1: first run, baseline is None → empty deltas.
    p1 = index_run_report(
        _FakeResult(baseline_payload),
        baseline_payload=None,
        current_payload=baseline_payload,
        history_dir=history_dir,
        now=FIXED_NOW_1,
    )
    # Run 2: regressed-variant indexed vs the baseline payload.
    p2 = index_run_report(
        _FakeResult(current_payload),
        baseline_payload=baseline_payload,
        current_payload=current_payload,
        history_dir=history_dir,
        now=FIXED_NOW_2,
    )

    # (a) path keyed on pack_name only, both calls land on the same file.
    assert p1 == p2
    index_path = history_dir / "synpack.history.jsonl"
    assert index_path.exists()
    assert p2 == index_path

    # (b) one line per run, in timestamp order.
    lines = index_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    entries = [json.loads(line) for line in lines]
    assert [e["timestamp"] for e in entries] == [
        FIXED_NOW_1.isoformat(),
        FIXED_NOW_2.isoformat(),
    ]

    # (c) entry shape.
    for entry in entries:
        for key in (
            "timestamp",
            "pack_name",
            "gate_passed",
            "per_query_deltas",
            "per_qtype_deltas",
        ):
            assert key in entry, f"entry missing {key!r}: {entry}"
        assert entry["pack_name"] == "synpack"

    # First run: no prior baseline → empty per-query deltas.
    assert entries[0]["per_query_deltas"] == {}

    # (d) second run hand-computed per-qid deltas (current - baseline).
    deltas = entries[1]["per_query_deltas"]
    # g1 recall: 0.6 - 1.0 = -0.4 (regression).
    assert deltas["g1"]["recall"] == pytest.approx(-0.4)
    # g1 faithfulness: 0.9 - 0.8 = +0.1 (improvement, positive sign).
    assert deltas["g1"]["faithfulness"] == pytest.approx(0.1)
    # g2 mrr: 0.5 - 1.0 = -0.5 (regression).
    assert deltas["g2"]["mrr"] == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_compute_per_query_deltas() -> None:
    """Per-qid deltas across the three maps; sign + epsilon boundary.

    delta = current - baseline. One qid sits JUST INSIDE epsilon (|delta| ==
    epsilon, not a regression) and one JUST OUTSIDE (regression).
    """
    from src.eval.runner.run_history import compute_per_query_deltas

    baseline = {
        "per_query_recall": {"inside": 1.0, "outside": 1.0},
        "per_query_mrr": {},
        "per_query_faithfulness": {},
    }
    current = {
        # delta == -0.05 exactly → NOT a regression (== -epsilon), but still a
        # recorded delta with the correct sign.
        "per_query_recall": {"inside": 0.95, "outside": 0.94},
        "per_query_mrr": {},
        "per_query_faithfulness": {},
    }

    deltas = compute_per_query_deltas(baseline, current, epsilon=0.05)

    assert deltas["inside"]["recall"] == pytest.approx(-0.05)
    assert deltas["outside"]["recall"] == pytest.approx(-0.06)
    # Sign convention: both negative (current < baseline → worse).
    assert deltas["inside"]["recall"] < 0
    assert deltas["outside"]["recall"] < 0


def test_index_jsonl_format_and_append(tmp_path: Path) -> None:
    """Append two runs, read back, assert JSONL shape + append order."""
    from src.eval.runner.run_history import index_run_report

    history_dir = tmp_path / "reports"

    payload_a = {
        "pack_name": "p",
        "passed": True,
        "k": 5,
        "recall_by_qtype": {"factoid": 1.0},
        "mrr_by_qtype": {"factoid": 1.0},
        "faithfulness_by_qtype": {"factoid": 1.0},
        "per_query_recall": {"g1": 1.0},
        "per_query_mrr": {"g1": 1.0},
        "per_query_faithfulness": {"g1": 1.0},
    }
    payload_b = dict(payload_a, passed=False)

    class _R:
        def __init__(self, p):
            self._p = p

    path = index_run_report(
        _R(payload_a),
        baseline_payload=None,
        current_payload=payload_a,
        history_dir=history_dir,
        now=FIXED_NOW_1,
    )
    index_run_report(
        _R(payload_b),
        baseline_payload=payload_a,
        current_payload=payload_b,
        history_dir=history_dir,
        now=FIXED_NOW_2,
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    a, b = (json.loads(line) for line in lines)
    assert a["gate_passed"] is True
    assert b["gate_passed"] is False
    assert a["k"] == 5
    assert a["timestamp"] == FIXED_NOW_1.isoformat()
    assert b["timestamp"] == FIXED_NOW_2.isoformat()


def test_load_run_history_sorts_by_timestamp(tmp_path: Path) -> None:
    """Out-of-order writes load sorted; missing file → []."""
    from src.eval.runner.run_history import index_run_report, load_run_history

    history_dir = tmp_path / "reports"

    # Missing file → [].
    assert load_run_history("nope", history_dir) == []

    payload = {
        "pack_name": "p",
        "passed": True,
        "k": 5,
        "recall_by_qtype": {},
        "mrr_by_qtype": {},
        "faithfulness_by_qtype": {},
        "per_query_recall": {},
        "per_query_mrr": {},
        "per_query_faithfulness": {},
    }

    class _R:
        def __init__(self, p):
            self._p = p

    # Write the LATER timestamp first, then the EARLIER one.
    index_run_report(
        _R(payload),
        baseline_payload=None,
        current_payload=payload,
        history_dir=history_dir,
        now=FIXED_NOW_2,
    )
    index_run_report(
        _R(payload),
        baseline_payload=None,
        current_payload=payload,
        history_dir=history_dir,
        now=FIXED_NOW_1,
    )

    history = load_run_history("p", history_dir)
    assert [e["timestamp"] for e in history] == [
        FIXED_NOW_1.isoformat(),
        FIXED_NOW_2.isoformat(),
    ]


def test_first_run_no_baseline_yields_empty_deltas() -> None:
    """baseline None/empty → per-query deltas {} and no crash."""
    from src.eval.runner.run_history import compute_per_query_deltas

    current = {
        "per_query_recall": {"g1": 1.0},
        "per_query_mrr": {"g1": 1.0},
        "per_query_faithfulness": {"g1": 1.0},
    }
    assert compute_per_query_deltas(None, current) == {}
    assert compute_per_query_deltas({}, current) == {}
