# @summary
# RED tests for P11b — the Ralph-B actuation step (src/eval/runner/ralph_apply.py,
# NOT YET IMPLEMENTED). run_ralph_b takes a P11a InvestigationProposal + a
# SustainedMetricRegression target, opens an ISOLATED sandbox, applies the fix
# inside it, measures the target metric before (current tree) and after
# (sandbox), and ONLY keeps a committed branch when the metric IMPROVED past
# epsilon (strict >). It NEVER opens a PR / imports gh. The HEADLINE test proves
# — empirically — that the PRIMARY worktree file-tree AND `git status --porcelain`
# are byte-identical before/after a run (mutations land only in the injected
# sandbox's tmp_path subdir), modelled on P11a's never-mutate headline. Unit
# tests pin: improved→commit-once+branch-kept, not-improved→discard+branch=None,
# the strict-`>` epsilon boundary (== epsilon NOT improved; epsilon+ε IS — a
# discriminating partner), no-changes→discard+applied=False+measure-called-once,
# diff-too-large→discard+reject_reason, and that `now` is resolved ONCE (fixed-now
# matches report.timestamp; now=None still yields an ISO timestamp).
# Exports: test_run_ralph_b_actuates_and_never_mutates_primary_worktree,
#          test_run_ralph_b_improved_path_commits_and_keeps_branch,
#          test_run_ralph_b_not_improved_discards_and_drops_branch,
#          test_run_ralph_b_epsilon_boundary_is_strict_not_improved,
#          test_run_ralph_b_one_step_above_epsilon_is_improved,
#          test_run_ralph_b_no_changes_discards_and_skips_second_measure,
#          test_run_ralph_b_diff_too_large_discards,
#          test_run_ralph_b_fixed_now_flows_to_timestamp,
#          test_run_ralph_b_default_now_is_iso,
#          test_ralph_apply_module_imports_no_gh
# Deps: src.eval.runner.ralph_apply (run_ralph_b, ActuationReport,
#       ActuationOutcome — DOES NOT EXIST YET → these tests are RED),
#       src.eval.runner.ralph (InvestigationProposal),
#       src.eval.runner.sustained_regression (SustainedMetricRegression).
# @end-summary
"""RED tests for the Ralph-B actuation step (P11b).

All fast tests inject fakes for every external seam — ``sandbox_factory``,
``apply_fn``, and ``measure_fn`` — so NO ``claude`` process, NO git worktree,
NO Weaviate/embeddings/LLM is ever touched. The fake sandbox writes ONLY into a
``tmp_path`` subdir, which lets the headline test assert the real repo worktree
is byte-identical before/after. The default seams (the real git-worktree
sandbox + the constrained headless-claude coding step + the run_eval-backed
measure) are exercised only by future model-gated/live tests, exactly like
P11a's ``investigate_fn``.

Distinct numeric design (mutation-test-data discipline): ``before`` and
``after`` are chosen so ``before``, ``after``, ``delta``, and ``epsilon`` are
all mutually distinct — no first/last/max/mean collision can silently insulate
an arithmetic mutation.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

# Default epsilon mirrored from the baseline/sustained-regression band so the
# boundary tests speak the same language as the rest of the eval surface.
EPSILON = 0.05


# ---------------------------------------------------------------------------
# Target / proposal builders (real frozen P11a + P8c dataclasses)
# ---------------------------------------------------------------------------


def _target():
    """The regression Ralph-B is actuating against (factoid/recall, -0.40)."""
    from src.eval.runner.sustained_regression import SustainedMetricRegression

    return SustainedMetricRegression(
        qtype="factoid",
        metric="recall",
        runs_regressed=3,
        window=3,
        latest_delta=-0.40,
    )


def _proposal():
    """The advisory P11a proposal Ralph-B will try to apply."""
    from src.eval.runner.ralph import InvestigationProposal

    return InvestigationProposal(
        diagnosis="recall dropped after the chunker overlap change",
        suggested_fix="revert chunk overlap to 64",
    )


# ---------------------------------------------------------------------------
# Recording fakes for the three injected seams
# ---------------------------------------------------------------------------


class FakeSandbox:
    """A recording sandbox that writes ONLY inside a tmp_path subdir.

    Records the ordered sequence of lifecycle calls ("commit"/"discard") plus
    every commit message, so tests can assert commit-vs-discard exclusivity,
    call counts, and the no-backticks invariant. ``path`` is a real on-disk
    directory under ``tmp_path`` — the apply_fn fake may scribble into it, and
    the headline test proves those scribbles never escape into the repo.
    """

    def __init__(self, base: Path) -> None:
        self.path = str(base)
        self.branch = "ralph/factoid-recall-fix"
        self.calls: list[str] = []
        self.commit_messages: list[str] = []

    def commit(self, message: str) -> None:
        self.calls.append("commit")
        self.commit_messages.append(message)

    def discard(self) -> None:
        self.calls.append("discard")


def _sandbox_factory(sandbox: FakeSandbox):
    """A factory closure returning a pre-built recording sandbox."""

    def _factory(pack_name: str) -> FakeSandbox:
        # Record the pack name onto the sandbox so callers can assert wiring.
        sandbox.requested_pack = pack_name  # type: ignore[attr-defined]
        return sandbox

    return _factory


def _apply_fn(*, changed: bool = True, summary: str = "applied the fix",
              diff_lines: int = 12, sink: Path | None = None,
              recorder: list | None = None):
    """A fake apply_fn returning a canned ApplyResult-shaped object.

    Uses :class:`types.SimpleNamespace` so the fake is INDEPENDENT of where SA2
    decides the real ``ApplyResult`` dataclass lives — the contract is purely
    structural (``.changed``, ``.summary``, ``.diff_lines``). If ``sink`` is
    given, the fake writes a file INTO the sandbox path to mimic a real edit,
    proving the headline non-mutation invariant has teeth.
    """

    def _fn(proposal, *, sandbox):
        if recorder is not None:
            recorder.append((proposal, sandbox))
        if sink is not None:
            (sink / "applied_change.txt").write_text(
                "scribbled inside the sandbox only", encoding="utf-8"
            )
        return SimpleNamespace(
            changed=changed, summary=summary, diff_lines=diff_lines
        )

    return _fn


def _measure_fn(*, before: float, after: float, recorder: list | None = None):
    """A fake measure_fn returning ``before`` on the current tree, ``after`` in the sandbox.

    The seam SHAPE under test: ``before`` is the measurement on the CURRENT tree
    (no ``working_dir`` / ``working_dir=None``), ``after`` is the measurement in
    the sandbox (``working_dir=<sandbox.path>``). Records every call's kwargs so
    tests can assert call COUNT (no-changes path must not measure twice) and that
    the sandbox path is threaded into the after-measurement.
    """

    def _fn(pack_name, qtype, metric, *, working_dir=None):
        if recorder is not None:
            recorder.append(
                {"pack_name": pack_name, "qtype": qtype,
                 "metric": metric, "working_dir": working_dir}
            )
        return after if working_dir else before

    return _fn


# ---------------------------------------------------------------------------
# Empirical no-mutation snapshot helper (mirrors tests/eval/test_ralph.py)
# ---------------------------------------------------------------------------


def _tree_snapshot(*roots: Path) -> set[tuple[str, float, int]]:
    """Snapshot (path, mtime, size) for every file under each root.

    A created, modified, or deleted file changes the set, so an identical set
    before/after proves the file tree is byte-stable across a run.
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
# HEADLINE RED — actuate-in-sandbox + EMPIRICAL never-mutate-primary-worktree
# ===========================================================================


def test_run_ralph_b_actuates_and_never_mutates_primary_worktree(
    tmp_path: Path,
) -> None:
    """Ralph-B mutates ONLY inside the injected sandbox; the repo is byte-stable.

    Uses an improved-path run (the fix lands a committed branch) and a fake
    apply_fn that writes a real file INTO the sandbox dir. After the run:

      * the PRIMARY worktree (cwd) file tree AND `git status --porcelain` are
        byte-identical (every edit landed in the tmp sandbox, NOT the repo);
      * NO gh/PR call ever happened (the module imports no gh — see the
        import-surface test);
      * the outcome reflects an improvement (commit kept).
    """
    from src.eval.runner.ralph_apply import ActuationReport, run_ralph_b

    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    sandbox = FakeSandbox(sandbox_dir)

    # before=0.40 on the current tree, after=0.62 in the sandbox → delta=+0.22,
    # well past epsilon → improved (commit kept). before/after/delta/epsilon all
    # distinct, so no arithmetic mutation can coincidentally pass.
    measure = _measure_fn(before=0.40, after=0.62)
    apply = _apply_fn(changed=True, diff_lines=12, sink=sandbox_dir)

    cwd = Path.cwd()
    git_before = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=cwd, capture_output=True, text=True,
    ).stdout
    snap_before = _tree_snapshot(cwd)

    report = run_ralph_b(
        "mypack",
        proposal=_proposal(),
        target=_target(),
        packs_root=str(tmp_path / "packs"),
        sandbox_factory=_sandbox_factory(sandbox),
        apply_fn=apply,
        measure_fn=measure,
        epsilon=EPSILON,
    )

    snap_after = _tree_snapshot(cwd)
    git_after = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=cwd, capture_output=True, text=True,
    ).stdout

    # (a) the run produced an actuation report describing an improvement.
    assert isinstance(report, ActuationReport)
    assert report.outcome.improved is True
    assert report.outcome.applied is True

    # (b) the fake apply_fn DID write into the sandbox (so the no-mutation
    # assertion below is meaningful, not vacuous).
    assert (sandbox_dir / "applied_change.txt").is_file()

    # (c) EMPIRICAL no-mutation of the PRIMARY worktree: tree + git status
    # byte-identical (the sandbox lives under tmp_path, excluded from the snap).
    assert snap_after == snap_before, (
        "run_ralph_b mutated the primary worktree file tree:\n"
        f"created/changed: {sorted(snap_after - snap_before)}\n"
        f"removed: {sorted(snap_before - snap_after)}"
    )
    assert git_after == git_before, (
        "run_ralph_b changed `git status --porcelain` of the primary worktree"
    )


# ===========================================================================
# Improved path → commit ONCE, branch kept, delta correct
# ===========================================================================


def test_run_ralph_b_improved_path_commits_and_keeps_branch(
    tmp_path: Path,
) -> None:
    """delta > epsilon → sandbox.commit called ONCE, branch kept, outcome populated."""
    from src.eval.runner.ralph_apply import run_ralph_b

    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    sandbox = FakeSandbox(sandbox_dir)

    measure_calls: list = []
    # before=0.30, after=0.71 → delta=+0.41 (> 0.05) → improved.
    measure = _measure_fn(before=0.30, after=0.71, recorder=measure_calls)

    report = run_ralph_b(
        "mypack",
        proposal=_proposal(),
        target=_target(),
        packs_root=str(tmp_path / "packs"),
        sandbox_factory=_sandbox_factory(sandbox),
        apply_fn=_apply_fn(changed=True, diff_lines=20),
        measure_fn=measure,
        epsilon=EPSILON,
    )

    out = report.outcome
    assert out.improved is True
    assert out.applied is True
    assert out.reject_reason is None
    assert out.before_value == pytest.approx(0.30)
    assert out.after_value == pytest.approx(0.71)
    assert out.delta == pytest.approx(0.41)  # after - before, not before - after
    assert out.target_qtype == "factoid"
    assert out.target_metric == "recall"

    # Committed exactly once; never discarded on the improved path.
    assert sandbox.calls == ["commit"]
    # Branch + path kept for human review.
    assert out.branch_name == sandbox.branch
    assert out.sandbox_path == sandbox.path

    # The commit message must contain NO backticks (command-substitution footgun).
    assert sandbox.commit_messages, "commit message was never recorded"
    assert "`" not in sandbox.commit_messages[0]

    # measure_fn called twice: before (current tree) then after (sandbox path).
    assert len(measure_calls) == 2
    assert measure_calls[0]["working_dir"] in (None, "")
    assert measure_calls[1]["working_dir"] == sandbox.path
    assert measure_calls[1]["qtype"] == "factoid"
    assert measure_calls[1]["metric"] == "recall"

    # Diagnosis + suggested_fix carried from the proposal onto the report.
    assert report.diagnosis == "recall dropped after the chunker overlap change"
    assert report.suggested_fix == "revert chunk overlap to 64"


# ===========================================================================
# Not-improved path → discard, branch=None
# ===========================================================================


def test_run_ralph_b_not_improved_discards_and_drops_branch(
    tmp_path: Path,
) -> None:
    """delta <= epsilon → sandbox.discard called, branch_name None, improved False."""
    from src.eval.runner.ralph_apply import run_ralph_b

    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    sandbox = FakeSandbox(sandbox_dir)

    # before=0.50, after=0.52 → delta=+0.02 (< 0.05) → NOT improved.
    measure = _measure_fn(before=0.50, after=0.52)

    report = run_ralph_b(
        "mypack",
        proposal=_proposal(),
        target=_target(),
        packs_root=str(tmp_path / "packs"),
        sandbox_factory=_sandbox_factory(sandbox),
        apply_fn=_apply_fn(changed=True, diff_lines=8),
        measure_fn=measure,
        epsilon=EPSILON,
    )

    out = report.outcome
    assert out.improved is False
    assert out.applied is True  # the fix WAS applied; it just didn't help
    assert out.delta == pytest.approx(0.02)
    assert out.branch_name is None
    assert sandbox.calls == ["discard"]
    assert "commit" not in sandbox.calls


# ===========================================================================
# Epsilon boundary — strict `>` (delta == epsilon is NOT improved)
# + discriminating one-step-above partner
# ===========================================================================


def test_run_ralph_b_epsilon_boundary_is_strict_not_improved(
    tmp_path: Path,
) -> None:
    """delta EXACTLY == epsilon is NOT an improvement (strict `>`)."""
    from src.eval.runner.ralph_apply import run_ralph_b

    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    sandbox = FakeSandbox(sandbox_dir)

    # before=0.40, after=0.45 → delta=+0.05 == epsilon → NOT improved.
    measure = _measure_fn(before=0.40, after=0.45)

    report = run_ralph_b(
        "mypack",
        proposal=_proposal(),
        target=_target(),
        packs_root=str(tmp_path / "packs"),
        sandbox_factory=_sandbox_factory(sandbox),
        apply_fn=_apply_fn(changed=True, diff_lines=8),
        measure_fn=measure,
        epsilon=EPSILON,
    )

    out = report.outcome
    assert out.delta == pytest.approx(EPSILON)
    assert out.improved is False
    assert out.branch_name is None
    assert sandbox.calls == ["discard"]


def test_run_ralph_b_one_step_above_epsilon_is_improved(
    tmp_path: Path,
) -> None:
    """delta one step ABOVE epsilon IS an improvement — gives the strict gate teeth.

    Pairs with the ==epsilon test: a `>=` mutation would pass the boundary test
    (it would call the ==epsilon case "improved", which that test forbids) but a
    `<` / `<=` mutation that rejects this just-above case is caught HERE.
    """
    from src.eval.runner.ralph_apply import run_ralph_b

    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    sandbox = FakeSandbox(sandbox_dir)

    # before=0.40, after≈0.451 → delta≈0.051 (one small step above epsilon) → improved.
    measure = _measure_fn(before=0.40, after=0.40 + EPSILON + 1e-3)

    report = run_ralph_b(
        "mypack",
        proposal=_proposal(),
        target=_target(),
        packs_root=str(tmp_path / "packs"),
        sandbox_factory=_sandbox_factory(sandbox),
        apply_fn=_apply_fn(changed=True, diff_lines=8),
        measure_fn=measure,
        epsilon=EPSILON,
    )

    out = report.outcome
    assert out.delta == pytest.approx(EPSILON + 1e-3)
    assert out.improved is True
    assert sandbox.calls == ["commit"]
    assert out.branch_name == sandbox.branch


# ===========================================================================
# No-changes path → discard, applied=False, reject_reason set,
# measure_fn NOT called a second time
# ===========================================================================


def test_run_ralph_b_no_changes_discards_and_skips_second_measure(
    tmp_path: Path,
) -> None:
    """apply.changed=False → discard, applied=False, reject_reason set, one measure only."""
    from src.eval.runner.ralph_apply import run_ralph_b

    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    sandbox = FakeSandbox(sandbox_dir)

    measure_calls: list = []
    # after would be a big improvement IF it were measured — but it must NOT be.
    measure = _measure_fn(before=0.30, after=0.99, recorder=measure_calls)

    report = run_ralph_b(
        "mypack",
        proposal=_proposal(),
        target=_target(),
        packs_root=str(tmp_path / "packs"),
        sandbox_factory=_sandbox_factory(sandbox),
        apply_fn=_apply_fn(changed=False, summary="no changes produced", diff_lines=0),
        measure_fn=measure,
        epsilon=EPSILON,
    )

    out = report.outcome
    assert out.applied is False
    assert out.improved is False
    assert out.reject_reason == "no-changes"
    assert out.branch_name is None
    # Sandbox opened then discarded (no commit).
    assert sandbox.calls == ["discard"]
    # measure_fn called AT MOST once (the `before` measure); the after-measure
    # is short-circuited because nothing changed. The after_value must be None.
    assert len(measure_calls) <= 1
    assert out.after_value is None


# ===========================================================================
# Diff-too-large path → discard, improved False, reject_reason
# ===========================================================================


def test_run_ralph_b_diff_too_large_discards(tmp_path: Path) -> None:
    """apply.diff_lines > max_diff_lines → applied True, improved False, rejected, discarded."""
    from src.eval.runner.ralph_apply import run_ralph_b

    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    sandbox = FakeSandbox(sandbox_dir)

    measure_calls: list = []
    measure = _measure_fn(before=0.30, after=0.99, recorder=measure_calls)

    report = run_ralph_b(
        "mypack",
        proposal=_proposal(),
        target=_target(),
        packs_root=str(tmp_path / "packs"),
        sandbox_factory=_sandbox_factory(sandbox),
        apply_fn=_apply_fn(changed=True, diff_lines=201),  # > max_diff_lines
        measure_fn=measure,
        epsilon=EPSILON,
        max_diff_lines=200,
    )

    out = report.outcome
    assert out.applied is True  # a change WAS produced...
    assert out.improved is False  # ...but it's too large to keep
    assert out.reject_reason == "diff-too-large"
    assert out.branch_name is None
    assert sandbox.calls == ["discard"]
    # The over-large diff is rejected BEFORE the after-measure runs.
    assert len(measure_calls) <= 1
    assert out.after_value is None


# ===========================================================================
# `now` resolved ONCE — fixed-now flows to timestamp; now=None still ISO
# ===========================================================================


def test_run_ralph_b_fixed_now_flows_to_timestamp(tmp_path: Path) -> None:
    """An injected fixed `now` is the report.timestamp (resolved once at top)."""
    from datetime import datetime, timezone

    from src.eval.runner.ralph_apply import run_ralph_b

    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    sandbox = FakeSandbox(sandbox_dir)

    fixed = datetime(2026, 6, 2, 12, 30, 45, tzinfo=timezone.utc)
    report = run_ralph_b(
        "mypack",
        proposal=_proposal(),
        target=_target(),
        packs_root=str(tmp_path / "packs"),
        sandbox_factory=_sandbox_factory(sandbox),
        apply_fn=_apply_fn(changed=True, diff_lines=10),
        measure_fn=_measure_fn(before=0.30, after=0.71),
        epsilon=EPSILON,
        now=fixed,
    )
    assert report.timestamp == fixed.isoformat()


def test_run_ralph_b_default_now_is_iso(tmp_path: Path) -> None:
    """now=None still yields a parseable ISO-8601 timestamp (single resolution).

    Guards against forwarding ``None`` into multiple now-defaulting helpers and
    desyncing — the fixed-now test alone would mask that.
    """
    from datetime import datetime

    from src.eval.runner.ralph_apply import run_ralph_b

    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    sandbox = FakeSandbox(sandbox_dir)

    report = run_ralph_b(
        "mypack",
        proposal=_proposal(),
        target=_target(),
        packs_root=str(tmp_path / "packs"),
        sandbox_factory=_sandbox_factory(sandbox),
        apply_fn=_apply_fn(changed=True, diff_lines=10),
        measure_fn=_measure_fn(before=0.30, after=0.71),
        epsilon=EPSILON,
        now=None,
    )
    # Parses as ISO-8601 without raising.
    parsed = datetime.fromisoformat(report.timestamp)
    assert parsed.tzinfo is not None  # timezone-aware (datetime.now(timezone.utc))


# ===========================================================================
# Import-surface guard — the module imports NO gh / PyGithub
# ===========================================================================


def test_ralph_apply_module_imports_no_gh() -> None:
    """The ralph_apply module source must reference NO gh/PyGithub/PR machinery.

    This is the static counterpart to the empirical never-PR invariant: even the
    default (live) actuation path may open a sandbox + commit, but it must NEVER
    open a pull request. Scanning the source keeps the constraint single-sourced
    and un-bypassable by a future edit.
    """
    import src.eval.runner.ralph_apply as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    lowered = source.lower()
    assert "pygithub" not in lowered
    assert "import github" not in lowered
    assert "from github" not in lowered
    # No shelling out to the gh CLI to open a PR.
    assert "gh pr" not in lowered
    assert '"pr"' not in lowered and "'pr'" not in lowered


# ===========================================================================
# SA2-added: exception after sandbox open → discard THEN re-raise
# ===========================================================================


def test_run_ralph_b_discards_sandbox_then_reraises_on_error(
    tmp_path: Path,
) -> None:
    """An exception after the sandbox opens discards it, then RE-RAISES.

    SA1 deliberately left the exception branch to SA2; the chosen contract is
    discard-then-re-raise: the never-mutate invariant must hold even on the
    failure path (the sandbox is discarded), but a real actuation failure must
    NOT be hidden behind a clean report (the original error propagates). A fake
    apply_fn raises; the test asserts the same error type bubbles out AND the
    sandbox was discarded exactly once (never committed).
    """
    from src.eval.runner.ralph_apply import run_ralph_b

    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    sandbox = FakeSandbox(sandbox_dir)

    class _BoomError(RuntimeError):
        pass

    def _boom_apply(proposal, *, sandbox):
        raise _BoomError("apply step exploded")

    with pytest.raises(_BoomError):
        run_ralph_b(
            "mypack",
            proposal=_proposal(),
            target=_target(),
            packs_root=str(tmp_path / "packs"),
            sandbox_factory=_sandbox_factory(sandbox),
            apply_fn=_boom_apply,
            measure_fn=_measure_fn(before=0.30, after=0.71),
            epsilon=EPSILON,
        )

    # Discarded exactly once on the failure path; never committed.
    assert sandbox.calls == ["discard"]


# ===========================================================================
# SA2-added: ralph-apply CLI handler — ranks worst-first, actuates, persists,
# prints markdown (both DI seams injected; no infra)
# ===========================================================================


def test_ralph_apply_handler_actuates_worst_persists_and_prints(
    tmp_path: Path,
) -> None:
    """The CLI handler picks the WORST regression, calls run_b_fn on it, persists,
    and prints a markdown summary — all without touching git/claude/Weaviate.

    Two sustained regressions are seeded into a real history index; the handler
    must rank worst-first (most-negative latest_delta) and actuate ONLY the
    worst. Both DI seams (investigate_fn + run_b_fn) are injected, so no live
    investigator or sandbox is ever touched. The actuation report is then
    written under <reports-dir>/ralph_apply/.
    """
    import io
    import json

    from src.eval.cli import ralph_apply
    from src.eval.runner.ralph import InvestigationProposal
    from src.eval.runner.ralph_apply import (
        ActuationOutcome,
        ActuationReport,
        run_ralph_b,
    )

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    # Seed a real history index with two regressed (qtype, metric) pairs across
    # three runs. factoid/recall is the WORST (more negative latest delta).
    history = reports_dir / "mypack.history.jsonl"
    rows = []
    for ts, fr, mm in [
        ("2026-06-01T00:00:00+00:00", -0.30, -0.10),
        ("2026-06-01T01:00:00+00:00", -0.35, -0.12),
        ("2026-06-01T02:00:00+00:00", -0.40, -0.15),
    ]:
        rows.append(
            json.dumps(
                {
                    "timestamp": ts,
                    "per_qtype_deltas": {
                        "factoid": {"recall": fr},
                        "multi": {"recall": mm},
                    },
                }
            )
        )
    history.write_text("\n".join(rows) + "\n", encoding="utf-8")

    # Fake investigator: a canned proposal regardless of target.
    def _fake_investigate(target, context):
        return InvestigationProposal(
            diagnosis="diag", suggested_fix="fix"
        )

    # Fake run_b_fn: records the target it was called with and returns a report.
    run_b_calls: list = []

    def _fake_run_b(pack_name, *, proposal, target, packs_root,
                    epsilon, max_diff_lines):
        run_b_calls.append(target)
        outcome = ActuationOutcome(
            target_qtype=target.qtype,
            target_metric=target.metric,
            before_value=0.40,
            after_value=0.70,
            delta=0.30,
            improved=True,
            applied=True,
            branch_name="ralph-b/mypack-x",
            sandbox_path="/tmp/sbx",
            reject_reason=None,
        )
        return ActuationReport(
            pack_name=pack_name,
            timestamp="2026-06-02T12:00:00+00:00",
            target_qtype=target.qtype,
            target_metric=target.metric,
            diagnosis=proposal.diagnosis,
            suggested_fix=proposal.suggested_fix,
            outcome=outcome,
            notes=(),
        )

    buf = io.StringIO()
    rc = ralph_apply(
        "mypack",
        reports_dir=str(reports_dir),
        packs_root=str(tmp_path / "packs"),
        window=3,
        min_regressed=2,
        epsilon=EPSILON,
        investigate_fn=_fake_investigate,
        run_b_fn=_fake_run_b,
        stdout=buf,
    )

    assert rc == 0
    # Actuated EXACTLY the worst regression (factoid/recall, -0.40), once.
    assert len(run_b_calls) == 1
    assert run_b_calls[0].qtype == "factoid"
    assert run_b_calls[0].metric == "recall"

    # Markdown summary was printed and reflects the improvement.
    text = buf.getvalue()
    assert "Ralph-B actuation" in text
    assert "factoid / recall" in text
    assert "Improved" in text

    # The actuation report was persisted under <reports-dir>/ralph_apply/.
    persisted = list((reports_dir / "ralph_apply").glob("mypack_actuation_*.json"))
    assert len(persisted) == 1
    payload = json.loads(persisted[0].read_text(encoding="utf-8"))
    assert payload["target_qtype"] == "factoid"
    assert payload["outcome"]["improved"] is True

    # Sanity: the real run_ralph_b symbol is importable (DI default exists).
    assert callable(run_ralph_b)


def test_ralph_apply_handler_no_regressions_is_friendly(tmp_path: Path) -> None:
    """No sustained regressions → friendly line, exit 0, no sandbox, no persist."""
    import io

    from src.eval.cli import ralph_apply

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    # No history file → empty history → no sustained regressions.

    run_b_calls: list = []

    def _fake_run_b(*args, **kwargs):  # pragma: no cover — must NOT be called
        run_b_calls.append(kwargs)
        raise AssertionError("run_b_fn must not be called with no regressions")

    buf = io.StringIO()
    rc = ralph_apply(
        "mypack",
        reports_dir=str(reports_dir),
        packs_root=str(tmp_path / "packs"),
        run_b_fn=_fake_run_b,
        stdout=buf,
    )
    assert rc == 0
    assert run_b_calls == []
    assert "no sustained regressions" in buf.getvalue().lower()
    assert not (reports_dir / "ralph_apply").exists()


# ===========================================================================
# SA2-added: model-gated/live default-seam end-to-end (dual-marker; skips
# cleanly without infra). Gives packs_root + the default seams real teeth.
# ===========================================================================


@pytest.mark.slow
@pytest.mark.integration
def test_run_ralph_b_default_seams_live_end_to_end(tmp_path: Path) -> None:
    """Exercise the DEFAULT seams (real git worktree + claude + run_eval).

    Dual-marked (slow + integration) so the offline suite (``-m "not slow"``)
    never runs it. Skips cleanly unless BOTH a tiny eval pack and the live infra
    are wired via env (RALPH_B_LIVE_PACK). When it does run, it proves the real
    git-worktree sandbox is created and torn down without mutating the primary
    worktree.
    """
    import os
    import subprocess

    pack = os.environ.get("RALPH_B_LIVE_PACK")
    if not pack:
        pytest.skip("RALPH_B_LIVE_PACK not set — live default-seam test skipped")

    from src.eval.runner.ralph_apply import run_ralph_b

    git_before = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True,
    ).stdout

    run_ralph_b(
        pack,
        proposal=_proposal(),
        target=_target(),
        packs_root=os.environ.get("RALPH_B_LIVE_PACKS_ROOT", "evals/packs"),
        epsilon=EPSILON,
    )

    git_after = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True,
    ).stdout
    assert git_after == git_before, "live run_ralph_b mutated the primary worktree"
