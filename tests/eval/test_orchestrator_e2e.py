# @summary
# P8a — nightly orchestrator e2e + unit tests.
# Drives src.eval.orchestrator.run_nightly / discover_packs end-to-end over
# synthetic pack(s) with monkeypatched infra (reusing tests.eval.test_cli's
# _make_synthetic_pack / _patch_full_chain), plus a direct send_alert log test.
# Headline red: test_full_loop_with_injected_regression (run→persist→gate-fail
# →alert-shape, rc==1). NOTE: baseline/diff comparison is NOT in this slice —
# deferred to P8b; the e2e only asserts run→persist→gate-fail→alert-shape.
# Mutation targets:
#   M-A run_nightly returns 0 unconditionally → test_full_loop_with_injected_regression
#   M-B remove per-pack try/except          → test_run_nightly_continues_after_pack_error
#   M-C send_alert info-only on failure     → test_send_alert_logs_failure
#   M-D discover_packs returns all subdirs  → test_discover_packs_globs_pack_dirs
#   M-E remove/mis-wire index_run_report     → test_run_nightly_writes_run_history_index
# Exports: test_full_loop_with_injected_regression,
#          test_full_loop_all_pass_returns_zero,
#          test_run_nightly_continues_after_pack_error,
#          test_discover_packs_globs_pack_dirs,
#          test_send_alert_logs_failure, test_send_alert_pass_no_warning,
#          test_run_nightly_writes_run_history_index,
#          test_run_nightly_run_history_appends_across_runs
# Deps: src.eval.orchestrator, src.eval.runner.alert, tests.eval.test_cli
# @end-summary
"""P8a — nightly orchestrator tests."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Reuse the CLI test fixtures — do NOT duplicate the synthetic pack / infra patch.
from tests.eval.test_cli import _make_synthetic_pack, _patch_full_chain

FIXED_NOW = datetime(2026, 5, 30, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# run_nightly e2e
# ---------------------------------------------------------------------------


def test_full_loop_with_injected_regression(tmp_path: Path, monkeypatch) -> None:
    """Full loop over a regressed pack: run→persist→gate-fail→alert-shape, rc==1.

    HEADLINE RED. High recall floor (0.9) + retrieval returning a WRONG source
    (recall=0) → the gate MUST fail. We assert the orchestrator persists exactly
    one report JSON (passed=False, >=1 failure) and shapes an AlertPayload with
    gate_passed=False, non-empty failures, and report_path pointing at the file.

    NOTE: NO baseline comparison here — that is P8b. This slice only proves the
    run→persist→gate-fail→alert wiring.

    MUTATION A TARGET: run_nightly returning 0 unconditionally reds this.
    """
    from src.eval.orchestrator import run_nightly

    pack_dir = _make_synthetic_pack(
        tmp_path, recall_floor=0.9, faithfulness_floor=0.5
    )
    # Retrieval returns the WRONG source → recall=0.0 < 0.9 floor → gate fails.
    _patch_full_chain(monkeypatch, retrieved_source="other.md", judge_score=0.9)

    captured: list = []
    out_dir = tmp_path / "reports"

    rc = run_nightly(
        [pack_dir],
        output_dir=out_dir,
        now=FIXED_NOW,
        alert_fn=captured.append,
    )

    assert rc == 1

    written = list(out_dir.glob("*.json"))
    assert len(written) == 1, f"expected exactly one report, got {written!r}"
    persisted = json.loads(written[0].read_text())
    assert persisted["passed"] is False
    assert len(persisted["failures"]) >= 1

    assert len(captured) == 1
    payload = captured[0]
    assert payload.gate_passed is False
    assert payload.failures, "alert payload should carry the gate failures"
    assert payload.report_path == str(written[0])


def test_full_loop_all_pass_returns_zero(tmp_path: Path, monkeypatch) -> None:
    """Discriminating partner: LOW floors + good retrieval → gate PASSES, rc==0.

    Persisted report has passed=True; captured AlertPayload gate_passed=True.
    """
    from src.eval.orchestrator import run_nightly

    pack_dir = _make_synthetic_pack(
        tmp_path, recall_floor=0.5, faithfulness_floor=0.5
    )
    _patch_full_chain(monkeypatch, retrieved_source="doc1.md", judge_score=0.9)

    captured: list = []
    out_dir = tmp_path / "reports"

    rc = run_nightly(
        [pack_dir],
        output_dir=out_dir,
        now=FIXED_NOW,
        alert_fn=captured.append,
    )

    assert rc == 0
    written = list(out_dir.glob("*.json"))
    assert len(written) == 1
    persisted = json.loads(written[0].read_text())
    assert persisted["passed"] is True

    assert len(captured) == 1
    assert captured[0].gate_passed is True


def test_run_nightly_default_now_syncs_alert_and_persist_timestamp(
    tmp_path: Path, monkeypatch
) -> None:
    """Default path (no ``now``): alert timestamp is tz-aware AND equals persist's.

    Exercises the PRODUCTION default (``now=None``), which earlier resolved the
    persisted-filename timestamp and the alert timestamp INDEPENDENTLY — the
    alert side via a tz-naive ``datetime.now()`` — so the two desynced and the
    alert timestamp lost its timezone. We pin both invariants:

    * the captured AlertPayload.timestamp is non-empty and timezone-AWARE, and
    * it EQUALS the persisted report's ``timestamp`` field.

    REGRESSION GUARD: reintroducing a separate ``datetime.now().isoformat()`` for
    the alert reds this (timestamps differ and/or the alert is tz-naive).
    """
    from src.eval.orchestrator import run_nightly

    pack_dir = _make_synthetic_pack(
        tmp_path, recall_floor=0.5, faithfulness_floor=0.5
    )
    _patch_full_chain(monkeypatch, retrieved_source="doc1.md", judge_score=0.9)

    captured: list = []
    out_dir = tmp_path / "reports"

    # NO `now` argument — exercise the real production default.
    rc = run_nightly(
        [pack_dir],
        output_dir=out_dir,
        alert_fn=captured.append,
    )

    assert rc == 0
    assert len(captured) == 1
    alert_ts = captured[0].timestamp
    assert alert_ts, "alert timestamp must be non-empty on the default path"

    # Timezone-aware: fromisoformat must yield a tzinfo-bearing datetime.
    parsed = datetime.fromisoformat(alert_ts)
    assert parsed.tzinfo is not None, "alert timestamp must be timezone-aware"

    # The persisted report's timestamp must EQUAL the alert's timestamp.
    written = list(out_dir.glob("*.json"))
    assert len(written) == 1, f"expected exactly one report, got {written!r}"
    persisted = json.loads(written[0].read_text())
    assert persisted["timestamp"] == alert_ts, (
        "persisted-filename timestamp and alert timestamp must be the SAME value"
    )


def test_run_nightly_continues_after_pack_error(
    tmp_path: Path, monkeypatch
) -> None:
    """Resilient batch: pack A raises in run_eval, pack B still runs; rc==1.

    Pins the "one bad pack doesn't abort the batch" contract: B is persisted and
    alerted even though A errored, and an errored batch returns non-zero.

    MUTATION B TARGET: removing the per-pack try/except reds this (A's error
    aborts the batch before B runs).
    """
    import src.eval.orchestrator as orch

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    pack_a = _make_synthetic_pack(dir_a, recall_floor=0.5)
    pack_b = _make_synthetic_pack(dir_b, recall_floor=0.5)
    _patch_full_chain(monkeypatch, retrieved_source="doc1.md", judge_score=0.9)

    real_run_eval = orch.run_eval

    def flaky_run_eval(pack_path, **kwargs):
        if str(pack_a) in str(pack_path):
            raise RuntimeError("pack A blew up")
        return real_run_eval(pack_path, **kwargs)

    monkeypatch.setattr(orch, "run_eval", flaky_run_eval)

    captured: list = []
    out_dir = tmp_path / "reports"

    rc = orch.run_nightly(
        [pack_a, pack_b],
        output_dir=out_dir,
        now=FIXED_NOW,
        alert_fn=captured.append,
    )

    assert rc == 1, "an errored batch must return non-zero"
    written = list(out_dir.glob("*.json"))
    assert len(written) == 1, f"only pack B should persist, got {written!r}"
    persisted = json.loads(written[0].read_text())
    assert persisted["pack_name"] == "synpack"
    # Pack B's alert was emitted (pack A errored before producing a payload).
    assert len(captured) == 1
    assert captured[0].gate_passed is True


def test_discover_packs_globs_pack_dirs(tmp_path: Path) -> None:
    """discover_packs finds only dirs containing pack.yaml, sorted.

    MUTATION D TARGET: returning all subdirs (ignoring pack.yaml) reds this.
    """
    from src.eval.orchestrator import discover_packs

    root = tmp_path / "packs"
    root.mkdir()
    (root / "zebra").mkdir()
    (root / "zebra" / "pack.yaml").write_text("name: zebra\n")
    (root / "alpha").mkdir()
    (root / "alpha" / "pack.yaml").write_text("name: alpha\n")
    (root / "not_a_pack").mkdir()  # no pack.yaml → excluded
    (root / "not_a_pack" / "readme.txt").write_text("nope\n")

    found = discover_packs(root)
    assert [p.name for p in found] == ["alpha", "zebra"]


def test_discover_packs_respects_glob(tmp_path: Path) -> None:
    """A specific pack name selects exactly that pack dir."""
    from src.eval.orchestrator import discover_packs

    root = tmp_path / "packs"
    root.mkdir()
    for name in ("alpha", "beta"):
        (root / name).mkdir()
        (root / name / "pack.yaml").write_text(f"name: {name}\n")

    found = discover_packs(root, pack="alpha")
    assert [p.name for p in found] == ["alpha"]


# ---------------------------------------------------------------------------
# send_alert log stub
# ---------------------------------------------------------------------------


def test_send_alert_logs_failure(caplog) -> None:
    """A failing AlertPayload logs a WARNING naming the pack + failing metric.

    MUTATION C TARGET: logging at info even on failure reds this.
    """
    from src.eval.runner.alert import AlertPayload, send_alert

    payload = AlertPayload(
        pack_name="opentitan_riscv",
        collection_name="test_coll",
        gate_passed=False,
        failures=[
            {
                "qtype": "factoid",
                "metric": "recall_at_5",
                "expected": 0.9,
                "actual": 0.0,
                "qid": "g1",
            }
        ],
        report_path="/tmp/report.json",
        timestamp=FIXED_NOW.isoformat(),
    )

    with caplog.at_level(logging.WARNING, logger="rag.eval.alert"):
        send_alert(payload)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "a failing gate must emit a WARNING"
    text = " ".join(r.getMessage() for r in warnings)
    assert "opentitan_riscv" in text
    assert "recall_at_5" in text


def test_send_alert_pass_no_warning(caplog) -> None:
    """Partner: a passing AlertPayload emits no WARNING (info-only)."""
    from src.eval.runner.alert import AlertPayload, send_alert

    payload = AlertPayload(
        pack_name="opentitan_riscv",
        collection_name="test_coll",
        gate_passed=True,
        failures=[],
        report_path="/tmp/report.json",
        timestamp=FIXED_NOW.isoformat(),
    )

    with caplog.at_level(logging.INFO, logger="rag.eval.alert"):
        send_alert(payload)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not warnings, "a passing gate must not emit a WARNING"


# ---------------------------------------------------------------------------
# P8c — baseline-diff wiring
# ---------------------------------------------------------------------------


def _write_baseline(pack_dir: Path, baseline: dict) -> Path:
    """Dump ``baseline`` as the pack's committed ``baseline.json`` sibling."""
    path = Path(pack_dir) / "baseline.json"
    path.write_text(json.dumps(baseline), encoding="utf-8")
    return path


# A baseline where recall is steady (1.0) but faithfulness was higher (1.0) than
# the current run (0.9) → faithfulness is the SOLE regression past epsilon 0.05.
_BASELINE_FAITHFULNESS_REGRESSED = {
    "pack_name": "synpack",
    "timestamp": "2025-01-01T00:00:00+00:00",
    "passed": True,
    "recall_by_qtype": {"factoid": 1.0},
    "mrr_by_qtype": {"factoid": 1.0},
    "faithfulness_by_qtype": {"factoid": 1.0},
}


def test_baseline_present_attaches_diff_with_regressions(
    tmp_path: Path, monkeypatch
) -> None:
    """A committed baseline.json → AlertPayload.baseline_diff carries the diff.

    Gate PASSES (recall 1.0 >= 0.5, faithfulness 0.9 >= 0.5) but faithfulness
    REGRESSED vs the baseline's 1.0 (delta -0.1, past epsilon). Default
    fail_on_regression=False → rc stays 0 (no flip).
    """
    from src.eval.orchestrator import run_nightly

    pack_dir = _make_synthetic_pack(
        tmp_path, recall_floor=0.5, faithfulness_floor=0.5
    )
    _patch_full_chain(monkeypatch, retrieved_source="doc1.md", judge_score=0.9)
    _write_baseline(pack_dir, _BASELINE_FAITHFULNESS_REGRESSED)

    captured: list = []
    out_dir = tmp_path / "reports"

    rc = run_nightly(
        [pack_dir],
        output_dir=out_dir,
        now=FIXED_NOW,
        alert_fn=captured.append,
    )

    assert rc == 0, "gate passed and fail_on_regression is False by default"
    assert len(captured) == 1
    diff = captured[0].baseline_diff
    assert diff is not None, "a committed baseline.json must attach a diff"
    assert diff.regressions, "faithfulness 0.9 vs baseline 1.0 must regress"
    metrics = {md.metric for md in diff.regressions}
    assert metrics == {"faithfulness"}, (
        f"faithfulness is the sole regression; recall (1.0==1.0) is not: {metrics}"
    )
    (faith,) = [md for md in diff.regressions if md.metric == "faithfulness"]
    assert faith.delta == pytest.approx(-0.1)


def test_no_baseline_leaves_diff_none_and_unchanged_exit(
    tmp_path: Path, monkeypatch
) -> None:
    """No baseline.json → baseline_diff is None and rc is unchanged (pre-P8c)."""
    from src.eval.orchestrator import run_nightly

    pack_dir = _make_synthetic_pack(
        tmp_path, recall_floor=0.5, faithfulness_floor=0.5
    )
    _patch_full_chain(monkeypatch, retrieved_source="doc1.md", judge_score=0.9)
    # Intentionally DO NOT write baseline.json.

    captured: list = []
    out_dir = tmp_path / "reports"

    rc = run_nightly(
        [pack_dir],
        output_dir=out_dir,
        now=FIXED_NOW,
        alert_fn=captured.append,
    )

    assert rc == 0
    assert len(captured) == 1
    assert captured[0].baseline_diff is None


def test_fail_on_regression_flips_exit_despite_gate_pass(
    tmp_path: Path, monkeypatch
) -> None:
    """fail_on_regression=True flips rc to 1 on a regression even when gate passes.

    DISCRIMINATING PARTNER to test_baseline_present_attaches_diff_with_regressions:
    identical setup (gate passes, faithfulness regresses), only the flag flips.
    Proves the flip is regression-driven (gate_passed is still True) and kills a
    "default True" mutation of fail_on_regression.
    """
    from src.eval.orchestrator import run_nightly

    pack_dir = _make_synthetic_pack(
        tmp_path, recall_floor=0.5, faithfulness_floor=0.5
    )
    _patch_full_chain(monkeypatch, retrieved_source="doc1.md", judge_score=0.9)
    _write_baseline(pack_dir, _BASELINE_FAITHFULNESS_REGRESSED)

    captured: list = []
    out_dir = tmp_path / "reports"

    rc = run_nightly(
        [pack_dir],
        output_dir=out_dir,
        now=FIXED_NOW,
        alert_fn=captured.append,
        fail_on_regression=True,
    )

    assert rc == 1, "a regression with fail_on_regression=True must flip the exit"
    assert len(captured) == 1
    assert captured[0].gate_passed is True, (
        "the flip is regression-driven, not gate-driven"
    )
    assert captured[0].baseline_diff is not None
    assert captured[0].baseline_diff.regressions


def test_send_alert_logs_regression_even_on_gate_pass(caplog) -> None:
    """send_alert logs a REGRESSION WARNING when the diff has regressions.

    Partner assertion: a passing payload with baseline_diff=None emits NO
    REGRESSION warning (only the INFO pass line).
    """
    from src.eval.runner.alert import AlertPayload, send_alert
    from src.eval.runner.baseline import BaselineDiff, MetricDelta

    md = MetricDelta(
        qtype="factoid",
        metric="faithfulness",
        baseline_value=1.0,
        current_value=0.9,
        delta=-0.1,
        regressed=True,
    )
    diff = BaselineDiff(
        pack_name="synpack",
        baseline_timestamp="2025-01-01T00:00:00+00:00",
        current_timestamp=FIXED_NOW.isoformat(),
        passed_baseline=True,
        passed_current=True,
        regressions=(md,),
        improvements=(),
        unchanged=(),
        new_qtypes=(),
        dropped_qtypes=(),
        gate_flipped_to_fail=False,
    )
    payload = AlertPayload(
        pack_name="synpack",
        collection_name="test_coll",
        gate_passed=True,
        failures=[],
        report_path="/tmp/report.json",
        timestamp=FIXED_NOW.isoformat(),
        baseline_diff=diff,
    )

    with caplog.at_level(logging.INFO, logger="rag.eval.alert"):
        send_alert(payload)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "a regression must emit a WARNING even on gate pass"
    assert any("REGRESSION" in r.getMessage() for r in warnings)

    # Partner: gate-pass + no diff → no REGRESSION warning.
    caplog.clear()
    no_diff_payload = AlertPayload(
        pack_name="synpack",
        collection_name="test_coll",
        gate_passed=True,
        failures=[],
        report_path="/tmp/report.json",
        timestamp=FIXED_NOW.isoformat(),
        baseline_diff=None,
    )
    with caplog.at_level(logging.INFO, logger="rag.eval.alert"):
        send_alert(no_diff_payload)

    regression_warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "REGRESSION" in r.getMessage()
    ]
    assert not regression_warnings, "no diff → no REGRESSION warning"


def test_corrupt_baseline_degrades_gracefully_not_errored(
    tmp_path: Path, monkeypatch
) -> None:
    """A malformed/corrupt baseline.json must NOT mark the pack errored.

    A committed baseline that won't parse as JSON (hand-edit, merge-conflict
    markers, truncation) should degrade like an ABSENT baseline: the pack's eval
    still ran and persisted, so its alert must still fire with baseline_diff=None
    and the exit code must reflect the GATE only — never a spurious error-exit.
    """
    from src.eval.orchestrator import run_nightly

    pack_dir = _make_synthetic_pack(
        tmp_path, recall_floor=0.5, faithfulness_floor=0.5
    )
    _patch_full_chain(monkeypatch, retrieved_source="doc1.md", judge_score=0.9)
    # Garbage that json.loads cannot parse (e.g. a merge-conflict leftover).
    (Path(pack_dir) / "baseline.json").write_text(
        "<<<<<<< HEAD\nnot json\n", encoding="utf-8"
    )

    captured: list = []
    out_dir = tmp_path / "reports"

    rc = run_nightly(
        [pack_dir],
        output_dir=out_dir,
        now=FIXED_NOW,
        # fail_on_regression on to prove a corrupt baseline can't flip either.
        fail_on_regression=True,
        alert_fn=captured.append,
    )

    # Gate passes (recall 1.0, faithfulness 0.9 ≥ floors) → rc 0, NOT errored.
    assert rc == 0
    # The pack's alert still fired (not swallowed by an exception) ...
    assert len(captured) == 1
    # ... and the unusable baseline is treated as absent.
    assert captured[0].baseline_diff is None
    # The report was still persisted.
    assert len(list(out_dir.glob("*.json"))) == 1


# ---------------------------------------------------------------------------
# Run-history index wiring (gives the orchestrator integration seam teeth)
#   M-E remove/mis-wire index_run_report call → test_run_nightly_writes_run_history_index
# ---------------------------------------------------------------------------


# Baseline carrying BOTH per-qtype and per-query maps: per-qid faithfulness for
# g1 was 1.0; the current run scores 0.9 → the history index must record a
# faithfulness delta of -0.1 for qid g1 (sign = current - baseline).
_BASELINE_WITH_PER_QUERY = {
    "pack_name": "synpack",
    "timestamp": "2025-01-01T00:00:00+00:00",
    "passed": True,
    "recall_by_qtype": {"factoid": 1.0},
    "mrr_by_qtype": {"factoid": 1.0},
    "faithfulness_by_qtype": {"factoid": 1.0},
    "per_query_recall": {"g1": 1.0},
    "per_query_mrr": {"g1": 1.0},
    "per_query_faithfulness": {"g1": 1.0},
}


def test_run_nightly_writes_run_history_index(
    tmp_path: Path, monkeypatch
) -> None:
    """run_nightly appends a per-pack run-history entry with per-qid deltas.

    Gives the orchestrator's ``index_run_report`` wiring END-TO-END teeth: with a
    committed baseline carrying per-query maps, a nightly run must write
    ``<output_dir>/synpack.history.jsonl`` whose single entry records the per-qid
    faithfulness delta (current 0.9 - baseline 1.0 = -0.1) and the per-qtype
    delta. Deleting or mis-wiring the ``index_run_report`` call reds this (no
    file, or wrong/swapped delta); the per-qtype-only baseline tests above stay
    green and do NOT catch it.
    """
    from src.eval.orchestrator import run_nightly
    from src.eval.runner.run_history import load_run_history

    pack_dir = _make_synthetic_pack(
        tmp_path, recall_floor=0.5, faithfulness_floor=0.5
    )
    _patch_full_chain(monkeypatch, retrieved_source="doc1.md", judge_score=0.9)
    _write_baseline(pack_dir, _BASELINE_WITH_PER_QUERY)

    out_dir = tmp_path / "reports"

    rc = run_nightly(
        [pack_dir],
        output_dir=out_dir,
        now=FIXED_NOW,
        alert_fn=lambda _p: None,
    )

    assert rc == 0
    index_path = out_dir / "synpack.history.jsonl"
    assert index_path.is_file(), (
        "run_nightly must append the per-pack run-history index"
    )

    history = load_run_history("synpack", out_dir)
    assert len(history) == 1, "exactly one nightly run was indexed"
    entry = history[0]
    assert entry["pack_name"] == "synpack"
    assert entry["timestamp"] == FIXED_NOW.isoformat()
    # Per-qid teeth: g1 faithfulness regressed by 0.9 - 1.0 = -0.1 (sign =
    # current - baseline; a swapped payload arg would flip this to +0.1).
    assert entry["per_query_deltas"]["g1"]["faithfulness"] == pytest.approx(-0.1)
    # Per-qtype layer is present and single-sourced from compute_baseline_diff.
    assert entry["per_qtype_deltas"]["factoid"]["faithfulness"] == pytest.approx(
        -0.1
    )


def test_run_nightly_run_history_appends_across_runs(
    tmp_path: Path, monkeypatch
) -> None:
    """Two nightly runs for one pack append two timestamp-sorted lines.

    Proves the accumulation contract (append, not overwrite) at the ORCHESTRATOR
    level — the whole point of "trusted signal over time". An ``"a"``→``"w"``
    regression in index_run_report reds here via run_nightly, not only via the
    unit test.
    """
    from src.eval.orchestrator import run_nightly
    from src.eval.runner.run_history import load_run_history

    pack_dir = _make_synthetic_pack(
        tmp_path, recall_floor=0.5, faithfulness_floor=0.5
    )
    _patch_full_chain(monkeypatch, retrieved_source="doc1.md", judge_score=0.9)
    _write_baseline(pack_dir, _BASELINE_WITH_PER_QUERY)

    out_dir = tmp_path / "reports"
    later = datetime(2026, 5, 31, 12, 0, 0, tzinfo=timezone.utc)

    run_nightly(
        [pack_dir], output_dir=out_dir, now=FIXED_NOW, alert_fn=lambda _p: None
    )
    run_nightly(
        [pack_dir], output_dir=out_dir, now=later, alert_fn=lambda _p: None
    )

    history = load_run_history("synpack", out_dir)
    assert len(history) == 2, "each run appends one entry (no overwrite)"
    assert [e["timestamp"] for e in history] == [
        FIXED_NOW.isoformat(),
        later.isoformat(),
    ], "entries are timestamp-sorted across runs"


# ---------------------------------------------------------------------------
# Sustained-regression detector wiring (multi-run verdict on AlertPayload)
#   delete the orchestrator detector wiring → test_run_nightly_attaches_
#   sustained_regressions reds (empty tuple, no entry)
# ---------------------------------------------------------------------------


def _history_entry(timestamp: str, per_qtype_deltas: dict) -> dict:
    """A run-history JSONL entry in the exact shape ``load_run_history`` parses."""
    return {
        "timestamp": timestamp,
        "pack_name": "synpack",
        "k": 5,
        "gate_passed": True,
        "per_query_deltas": {},
        "per_qtype_deltas": per_qtype_deltas,
    }


def test_run_nightly_attaches_sustained_regressions(
    tmp_path: Path, monkeypatch
) -> None:
    """run_nightly attaches the multi-run sustained-regression verdict.

    PRE-SEED ``<out_dir>/synpack.history.jsonl`` with 2 prior entries where
    ``(factoid, faithfulness)`` regressed in both (timestamps BEFORE FIXED_NOW so
    the appended current run sorts last). The current run scores faithfulness 0.9
    vs the committed baseline's 1.0 → also regresses (delta -0.1). After
    ``index_run_report`` appends the current run, the detector sees a 3-run window
    with faithfulness regressed in all 3 → the captured AlertPayload must carry
    a ``(factoid, faithfulness)`` sustained regression with ``runs_regressed==3``.

    Deleting the orchestrator detector wiring reds this (empty tuple).
    """
    from src.eval.orchestrator import run_nightly

    pack_dir = _make_synthetic_pack(
        tmp_path, recall_floor=0.5, faithfulness_floor=0.5
    )
    _patch_full_chain(monkeypatch, retrieved_source="doc1.md", judge_score=0.9)
    _write_baseline(pack_dir, _BASELINE_WITH_PER_QUERY)

    out_dir = tmp_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Pre-seed two prior runs; both have (factoid, faithfulness) regressed.
    seed = [
        _history_entry(
            "2026-05-28T12:00:00+00:00", {"factoid": {"faithfulness": -0.2}}
        ),
        _history_entry(
            "2026-05-29T12:00:00+00:00", {"factoid": {"faithfulness": -0.15}}
        ),
    ]
    index_path = out_dir / "synpack.history.jsonl"
    index_path.write_text(
        "\n".join(json.dumps(e) for e in seed) + "\n", encoding="utf-8"
    )

    captured: list = []
    rc = run_nightly(
        [pack_dir],
        output_dir=out_dir,
        now=FIXED_NOW,
        alert_fn=captured.append,
    )

    assert rc == 0
    assert len(captured) == 1
    sustained = captured[0].sustained_regressions
    assert sustained, "the 3-run faithfulness regression must be flagged"
    faith = [
        s
        for s in sustained
        if s.qtype == "factoid" and s.metric == "faithfulness"
    ]
    assert len(faith) == 1, f"factoid/faithfulness must be flagged: {sustained}"
    assert faith[0].runs_regressed == 3, (
        "2 pre-seeded + the current run = all 3 in the window regressed"
    )
    assert faith[0].window == 3
    assert faith[0].latest_delta == pytest.approx(-0.1)
