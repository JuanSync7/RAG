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
# Exports: test_full_loop_with_injected_regression,
#          test_full_loop_all_pass_returns_zero,
#          test_run_nightly_continues_after_pack_error,
#          test_discover_packs_globs_pack_dirs,
#          test_send_alert_logs_failure, test_send_alert_pass_no_warning
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
