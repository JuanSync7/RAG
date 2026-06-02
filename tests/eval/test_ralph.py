# @summary
# Tests for P11a — the Ralph-A proposal-only eval-failure investigator
# (src/eval/runner/ralph.py + persist_ralph_report + the `ralph` CLI
# subcommand). The HEADLINE test proves run_ralph_a ranks sustained
# regressions worst-first, investigates the top-K via an injected fake
# investigate_fn, returns a human-reviewable RalphReport, and — EMPIRICALLY —
# NEVER mutates the repo/cwd/pack tree (byte-identical snapshot + clean
# `git status --porcelain` before/after). Unit tests pin the exact ranking
# key (mutation-aware), the no-sustained no-op, max_iterations truncation,
# persistence filename/round-trip, the default investigator's envelope
# parsing + every JudgeBackendError raise path, the constrained argv contract,
# and the argv→handler CLI wiring. One model-gated live test (never run here).
# Exports: test_run_ralph_a_investigates_worst_first_and_never_mutates,
#          test_run_ralph_a_no_sustained_is_noop,
#          test_rank_regressions_orders_worst_first,
#          test_run_ralph_a_respects_max_iterations,
#          test_persist_ralph_report_shape,
#          test_build_ralph_investigator_parses_proposal,
#          test_ralph_investigator_argv_is_constrained,
#          test_ralph_cli_dispatch_returns_int,
#          test_ralph_investigator_real_claude
# Deps: src.eval.runner.ralph, src.eval.runner.persistence, src.eval.cli
# @end-summary
"""Tests for the Ralph-A proposal-only investigator (P11a).

All fast tests inject a fake ``investigate_fn`` (or a fake ``run_fn`` for the
default investigator), so NO ``claude`` process is ever spawned and NO
Weaviate/embeddings/LLM are touched. The one live test is double-gated on a
``claude`` binary on PATH AND an explicit opt-in env var.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# History-seeding helpers (mirror tests/eval/test_show_trend.py)
# ---------------------------------------------------------------------------


def _write_history(tmp_path: Path, pack: str, entries: list[dict]) -> None:
    """Hand-seed a ``<pack>.history.jsonl`` index (one JSON line per entry)."""
    path = tmp_path / f"{pack}.history.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")


def _entry(ts: str, per_qtype_deltas: dict) -> dict:
    """A minimal history entry carrying only what the detector reads."""
    return {
        "timestamp": ts,
        "pack_name": "p",
        "k": 5,
        "gate_passed": True,
        "per_query_deltas": {},
        "per_qtype_deltas": per_qtype_deltas,
    }


# ---------------------------------------------------------------------------
# Fake investigate_fn helpers
# ---------------------------------------------------------------------------


def _recording_investigate_fn(recorder: list):
    """A fake investigate_fn that records target order and returns canned text."""
    from src.eval.runner.ralph import InvestigationProposal

    def _fn(target, context):
        recorder.append(target)
        return InvestigationProposal(
            diagnosis=f"diag::{target.qtype}/{target.metric}",
            suggested_fix=f"fix::{target.qtype}/{target.metric}",
        )

    return _fn


def _raising_investigate_fn(target, context):
    """A fake investigate_fn that fails the test if it is ever called."""
    raise AssertionError("investigate_fn must NOT be called when no sustained")


# ---------------------------------------------------------------------------
# Empirical no-mutation snapshot helper
# ---------------------------------------------------------------------------


def _tree_snapshot(*roots: Path) -> set[tuple[str, float, int]]:
    """Snapshot (path, mtime, size) for every file under each root.

    This is the empirical teeth behind the never-mutate invariant: a created,
    modified, or deleted file changes the set, so an identical set proves the
    file tree is byte-stable across a run.
    """
    snap: set[tuple[str, float, int]] = set()
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file():
                st = p.stat()
                snap.add((str(p), st.st_mtime, st.st_size))
    return snap


# ===========================================================================
# HEADLINE RED — worst-first investigation + EMPIRICAL never-mutate
# ===========================================================================


def test_run_ralph_a_investigates_worst_first_and_never_mutates(
    tmp_path: Path,
) -> None:
    """Worst regression is investigated first; the run mutates NOTHING.

    Seeds a history producing TWO sustained regressions of distinct severity
    (factoid/recall at -0.40 is worse than lookup/mrr at -0.10). With
    max_iterations=1 only the WORST must be investigated, the canned proposal
    must flow into the single RalphInvestigation, and — empirically — the cwd +
    pack-dir file tree AND `git status --porcelain` must be byte-identical
    before and after.
    """
    from src.eval.runner.ralph import RalphReport, run_ralph_a

    pack = "p"
    # Both pairs regress in 3/3 runs (sustained). factoid/recall is the worse
    # latest_delta, so it must be ranked + investigated first.
    _write_history(
        tmp_path,
        pack,
        [
            _entry("2026-01-01T00:00:00", {"factoid": {"recall": -0.40},
                                           "lookup": {"mrr": -0.10}}),
            _entry("2026-01-02T00:00:00", {"factoid": {"recall": -0.40},
                                           "lookup": {"mrr": -0.10}}),
            _entry("2026-01-03T00:00:00", {"factoid": {"recall": -0.40},
                                           "lookup": {"mrr": -0.10}}),
        ],
    )

    recorder: list = []
    investigate_fn = _recording_investigate_fn(recorder)

    # Snapshot the worktree (cwd) + the tmp history dir BEFORE the run.
    cwd = Path.cwd()
    git_before = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=cwd, capture_output=True, text=True,
    ).stdout
    snap_before = _tree_snapshot(cwd, tmp_path)

    report = run_ralph_a(
        pack,
        history_dir=tmp_path,
        max_iterations=1,
        investigate_fn=investigate_fn,
    )

    snap_after = _tree_snapshot(cwd, tmp_path)
    git_after = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=cwd, capture_output=True, text=True,
    ).stdout

    # (a) the WORST target (most-negative latest_delta) was investigated first.
    assert recorder, "investigate_fn was never called"
    assert (recorder[0].qtype, recorder[0].metric) == ("factoid", "recall")

    # (b) exactly one investigation, carrying the canned proposal.
    assert isinstance(report, RalphReport)
    assert report.pack_name == pack
    assert report.sustained_count == 2
    assert report.max_iterations == 1
    assert len(report.investigations) == 1
    inv = report.investigations[0]
    assert inv.qtype == "factoid" and inv.metric == "recall"
    assert inv.latest_delta == pytest.approx(-0.40)
    assert inv.runs_regressed == 3
    assert inv.diagnosis == "diag::factoid/recall"
    assert inv.suggested_fix == "fix::factoid/recall"

    # (c) EMPIRICAL no-mutation: tree + git status byte-identical.
    assert snap_after == snap_before, (
        "run_ralph_a mutated the file tree:\n"
        f"created/changed: {sorted(snap_after - snap_before)}\n"
        f"removed: {sorted(snap_before - snap_after)}"
    )
    assert git_after == git_before, "run_ralph_a changed `git status --porcelain`"


# ===========================================================================
# No sustained → no-op (investigate_fn never called)
# ===========================================================================


def test_run_ralph_a_no_sustained_is_noop(tmp_path: Path) -> None:
    """A clean history → sustained_count 0, no investigations, fn never called."""
    from src.eval.runner.ralph import run_ralph_a

    pack = "p"
    _write_history(
        tmp_path,
        pack,
        [
            _entry("2026-01-01T00:00:00", {"factoid": {"recall": 0.10}}),
            _entry("2026-01-02T00:00:00", {"factoid": {"recall": 0.02}}),
            _entry("2026-01-03T00:00:00", {"factoid": {"recall": 0.05}}),
        ],
    )

    report = run_ralph_a(
        pack,
        history_dir=tmp_path,
        investigate_fn=_raising_investigate_fn,
    )
    assert report.sustained_count == 0
    assert report.investigations == ()


# ===========================================================================
# rank_regressions — EXACT key, mutation-aware
# ===========================================================================


def test_rank_regressions_orders_worst_first() -> None:
    """The exact sort key (latest_delta, -runs_regressed, qtype, metric) is pinned.

    Cases chosen so each tie-break level is load-bearing:
      * R_worst: most-negative delta (-0.50) but FEWEST runs (2) -> still FIRST
        (delta dominates runs_regressed).
      * R_a / R_b: SAME delta (-0.20); R_a has MORE runs (3 > 2) -> R_a before
        R_b (more runs_regressed wins on a delta tie).
      * R_q: same delta+runs as R_b but qtype "z" sorts after "a" -> last.
    """
    from src.eval.runner.ralph import rank_regressions
    from src.eval.runner.sustained_regression import SustainedMetricRegression

    r_worst = SustainedMetricRegression("factoid", "recall", 2, 3, -0.50)
    r_a = SustainedMetricRegression("alpha", "mrr", 3, 3, -0.20)
    r_b = SustainedMetricRegression("alpha", "recall", 2, 3, -0.20)
    r_q = SustainedMetricRegression("zeta", "recall", 2, 3, -0.20)

    # Feed in a deliberately scrambled order.
    ranked = rank_regressions([r_b, r_q, r_worst, r_a])
    assert ranked == [r_worst, r_a, r_b, r_q]


def test_select_worst_regression_picks_head_or_none() -> None:
    """select_worst_regression returns the ranked head, or None when empty."""
    from src.eval.runner.ralph import select_worst_regression
    from src.eval.runner.sustained_regression import SustainedMetricRegression

    assert select_worst_regression(()) is None
    a = SustainedMetricRegression("a", "recall", 2, 3, -0.10)
    b = SustainedMetricRegression("b", "recall", 2, 3, -0.40)
    assert select_worst_regression((a, b)) == b


# ===========================================================================
# max_iterations truncation
# ===========================================================================


def test_run_ralph_a_respects_max_iterations(tmp_path: Path) -> None:
    """3 sustained, max_iterations=2 → top-2 worst investigated; 3rd detected only."""
    from src.eval.runner.ralph import run_ralph_a

    pack = "p"
    # Three distinct sustained severities: -0.40 (worst), -0.25, -0.10 (mildest).
    deltas = {
        "factoid": {"recall": -0.40},
        "lookup": {"mrr": -0.25},
        "howto": {"faithfulness": -0.10},
    }
    _write_history(
        tmp_path,
        pack,
        [
            _entry("2026-01-01T00:00:00", deltas),
            _entry("2026-01-02T00:00:00", deltas),
            _entry("2026-01-03T00:00:00", deltas),
        ],
    )

    recorder: list = []
    report = run_ralph_a(
        pack,
        history_dir=tmp_path,
        max_iterations=2,
        investigate_fn=_recording_investigate_fn(recorder),
    )

    assert report.sustained_count == 3
    assert len(report.investigations) == 2
    investigated = [(i.qtype, i.metric) for i in report.investigations]
    assert investigated == [("factoid", "recall"), ("lookup", "mrr")]
    # The mildest (howto/faithfulness) was detected but NOT investigated.
    assert ("howto", "faithfulness") not in investigated


# ===========================================================================
# persist_ralph_report — filename + round-trip
# ===========================================================================


def test_persist_ralph_report_shape(tmp_path: Path) -> None:
    """persist_ralph_report writes <pack>_ralph_<ts>.json under ralph/ + round-trips."""
    from datetime import datetime, timezone

    from src.eval.runner.persistence import persist_ralph_report
    from src.eval.runner.ralph import RalphInvestigation, RalphReport

    now = datetime(2026, 6, 2, 12, 30, 45, tzinfo=timezone.utc)
    report = RalphReport(
        pack_name="mypack",
        timestamp=now.isoformat(),
        sustained_count=2,
        max_iterations=1,
        investigations=(
            RalphInvestigation(
                qtype="factoid",
                metric="recall",
                runs_regressed=3,
                window=3,
                latest_delta=-0.40,
                diagnosis="recall dropped after chunker change",
                suggested_fix="revert chunk overlap to 64",
            ),
        ),
    )

    out_dir = tmp_path / "out"
    path = persist_ralph_report(report, out_dir, now=now)

    filesafe_ts = now.isoformat().replace(":", "")
    assert path == out_dir / "ralph" / f"mypack_ralph_{filesafe_ts}.json"
    assert path.is_file()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["pack_name"] == "mypack"
    assert payload["timestamp"] == now.isoformat()
    assert payload["sustained_count"] == 2
    assert payload["max_iterations"] == 1
    assert len(payload["investigations"]) == 1
    pinv = payload["investigations"][0]
    assert pinv["qtype"] == "factoid"
    assert pinv["metric"] == "recall"
    assert pinv["runs_regressed"] == 3
    assert pinv["window"] == 3
    assert pinv["latest_delta"] == pytest.approx(-0.40)
    assert pinv["diagnosis"] == "recall dropped after chunker change"
    assert pinv["suggested_fix"] == "revert chunk overlap to 64"


# ===========================================================================
# build_ralph_investigator — envelope parsing + raise paths
# ===========================================================================


def _success_envelope(result_payload: str) -> str:
    """Serialize a VERIFIED-shape success envelope with a stringified result."""
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": result_payload,
        }
    )


def _make_run_fn(envelope_stdout: str, *, returncode: int = 0, capture: list | None = None):
    """Return a (argv, *, timeout) -> result callable serving a canned stdout."""

    def _run_fn(argv, *, timeout):
        if capture is not None:
            capture.append(argv)
        return SimpleNamespace(returncode=returncode, stdout=envelope_stdout, stderr="")

    return _run_fn


def _target():
    from src.eval.runner.sustained_regression import SustainedMetricRegression

    return SustainedMetricRegression("factoid", "recall", 3, 3, -0.40)


def test_build_ralph_investigator_parses_proposal() -> None:
    """The default investigator parses a success envelope into InvestigationProposal."""
    from src.eval.runner.ralph import InvestigationProposal, build_ralph_investigator

    run_fn = _make_run_fn(
        _success_envelope(json.dumps({"diagnosis": "D", "suggested_fix": "F"}))
    )
    investigate = build_ralph_investigator(run_fn=run_fn)
    out = investigate(_target(), "CONTEXT")
    assert isinstance(out, InvestigationProposal)
    assert out.diagnosis == "D"
    assert out.suggested_fix == "F"


def test_build_ralph_investigator_strips_fence() -> None:
    """A ```json ... ``` fenced result is stripped before parsing."""
    from src.eval.runner.ralph import build_ralph_investigator

    fenced = "```json\n" + json.dumps({"diagnosis": "D", "suggested_fix": "F"}) + "\n```"
    investigate = build_ralph_investigator(run_fn=_make_run_fn(_success_envelope(fenced)))
    out = investigate(_target(), "CONTEXT")
    assert out.diagnosis == "D"
    assert out.suggested_fix == "F"


def test_build_ralph_investigator_is_error_raises() -> None:
    """is_error: true raises JudgeBackendError (valid payload so guard is the cause)."""
    from src.eval.runner.judge_cli import JudgeBackendError
    from src.eval.runner.ralph import build_ralph_investigator

    envelope = json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": True,
            "result": json.dumps({"diagnosis": "D", "suggested_fix": "F"}),
        }
    )
    investigate = build_ralph_investigator(run_fn=_make_run_fn(envelope))
    with pytest.raises(JudgeBackendError):
        investigate(_target(), "CONTEXT")


def test_build_ralph_investigator_unparseable_raises() -> None:
    """A non-JSON result payload raises JudgeBackendError."""
    from src.eval.runner.judge_cli import JudgeBackendError
    from src.eval.runner.ralph import build_ralph_investigator

    investigate = build_ralph_investigator(
        run_fn=_make_run_fn(_success_envelope("not json at all"))
    )
    with pytest.raises(JudgeBackendError):
        investigate(_target(), "CONTEXT")


def test_build_ralph_investigator_nonzero_returncode_raises() -> None:
    """A non-zero CLI returncode raises JudgeBackendError before parsing."""
    from src.eval.runner.judge_cli import JudgeBackendError
    from src.eval.runner.ralph import build_ralph_investigator

    investigate = build_ralph_investigator(
        run_fn=_make_run_fn(
            _success_envelope(json.dumps({"diagnosis": "D", "suggested_fix": "F"})),
            returncode=1,
        )
    )
    with pytest.raises(JudgeBackendError):
        investigate(_target(), "CONTEXT")


# ===========================================================================
# argv constraint contract — single-sourced via build_claude_argv
# ===========================================================================


def test_ralph_investigator_argv_is_constrained() -> None:
    """The investigator's argv carries the cost/safety constraints.

    Pins --model, --system-prompt, --allowed-tools, --setting-sources, and
    --output-format json so a future edit can't silently turn the investigator
    into the full (expensive, tool-enabled) coding agent that could mutate.
    """
    from src.eval.runner.ralph import build_ralph_investigator

    captured: list = []
    run_fn = _make_run_fn(
        _success_envelope(json.dumps({"diagnosis": "D", "suggested_fix": "F"})),
        capture=captured,
    )
    investigate = build_ralph_investigator(run_fn=run_fn, model="claude-haiku-4-5-20251001")
    investigate(_target(), "CONTEXT")

    assert captured, "run_fn was never invoked"
    argv = captured[0]
    assert argv[0] == "claude"

    def _val(flag):
        return argv[argv.index(flag) + 1]

    assert "--model" in argv and _val("--model") == "claude-haiku-4-5-20251001"
    assert "--system-prompt" in argv
    assert "--output-format" in argv and _val("--output-format") == "json"
    # No tools, clean room — the never-mutate guard at the CLI level.
    assert "--allowed-tools" in argv and _val("--allowed-tools") == ""
    assert "--setting-sources" in argv and _val("--setting-sources") == ""


# ===========================================================================
# build_investigation_context — deterministic, pure
# ===========================================================================


def test_build_investigation_context_is_pure_and_summarizes(tmp_path: Path) -> None:
    """Context summarizes the target + delta trail; includes goldens when pack given."""
    from src.eval.runner.ralph import build_investigation_context

    history = [
        _entry("2026-01-01T00:00:00", {"factoid": {"recall": -0.40}}),
        _entry("2026-01-02T00:00:00", {"factoid": {"recall": -0.40}}),
    ]
    target = _target()
    ctx = build_investigation_context(target, history, None)
    assert isinstance(ctx, str)
    assert "factoid" in ctx and "recall" in ctx
    assert "-0.4" in ctx
    # Pure: calling twice with the same inputs yields the same string.
    assert build_investigation_context(target, history, None) == ctx


# ===========================================================================
# CLI dispatch — argv→handler wiring
# ===========================================================================


def test_ralph_cli_dispatch_returns_int(tmp_path: Path, capsys) -> None:
    """main(['ralph', pack, ...]) returns int 0, prints markdown, persists a file."""
    from src.eval.cli import main
    from src.eval.runner.ralph import InvestigationProposal

    pack = "p"
    _write_history(
        tmp_path,
        pack,
        [
            _entry("2026-01-01T00:00:00", {"factoid": {"recall": -0.40}}),
            _entry("2026-01-02T00:00:00", {"factoid": {"recall": -0.40}}),
            _entry("2026-01-03T00:00:00", {"factoid": {"recall": -0.40}}),
        ],
    )

    def fake_investigate(target, context):
        return InvestigationProposal(diagnosis="DIAG-CLI", suggested_fix="FIX-CLI")

    # Drive the public handler directly with an injected fake (no real claude).
    from src.eval.cli import ralph as ralph_handler

    rc = ralph_handler(
        pack,
        reports_dir=str(tmp_path),
        investigate_fn=fake_investigate,
    )
    assert isinstance(rc, int)
    assert rc == 0
    out = capsys.readouterr().out
    assert "DIAG-CLI" in out
    assert "FIX-CLI" in out
    # A ralph report file was persisted under <reports-dir>/ralph/.
    persisted = list((tmp_path / "ralph").glob("p_ralph_*.json"))
    assert persisted, "no ralph report persisted"


def test_ralph_cli_main_argv_wires_to_handler(tmp_path: Path, monkeypatch) -> None:
    """main(['ralph', ...]) parses argv and dispatches to the ralph handler.

    Spies the handler to prove the argv→handler wiring (independent of the real
    investigator), and pins that main() returns the handler's int.
    """
    from src.eval import cli as cli_mod

    calls: list = []

    def spy_ralph(pack, **kwargs):
        calls.append((pack, kwargs))
        return 0

    monkeypatch.setattr(cli_mod, "ralph", spy_ralph)
    rc = cli_mod.main(["ralph", "p", "--reports-dir", str(tmp_path), "--max-iterations", "2"])
    assert rc == 0
    assert calls, "ralph handler was never dispatched"
    pack, kwargs = calls[0]
    assert pack == "p"
    assert kwargs["reports_dir"] == str(tmp_path)
    assert kwargs["max_iterations"] == 2


# ===========================================================================
# MODEL-GATED REAL TEST — live claude CLI (double opt-in)
# ===========================================================================


@pytest.mark.slow
@pytest.mark.integration
def test_ralph_investigator_real_claude() -> None:
    """Investigate one synthetic target with a real `claude` CLI.

    Double-gated: skipped unless a `claude` binary is on PATH AND the explicit
    RAGWEAVE_EVAL_CLI_LIVE=1 opt-in is set, so it never runs by accident.
    """
    if not shutil.which("claude"):
        pytest.skip("no `claude` binary on PATH")
    if os.environ.get("RAGWEAVE_EVAL_CLI_LIVE") != "1":
        pytest.skip("RAGWEAVE_EVAL_CLI_LIVE != 1 (live investigator opt-in not set)")

    from src.eval.runner.ralph import build_investigation_context, build_ralph_investigator

    target = _target()
    history = [
        _entry("2026-01-01T00:00:00", {"factoid": {"recall": -0.40}}),
        _entry("2026-01-02T00:00:00", {"factoid": {"recall": -0.40}}),
    ]
    context = build_investigation_context(target, history, None)
    investigate = build_ralph_investigator()
    out = investigate(target, context)
    assert out.diagnosis.strip()
    assert out.suggested_fix.strip()
