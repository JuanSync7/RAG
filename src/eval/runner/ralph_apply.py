# @summary
# P11b — Ralph-B actuation step. run_ralph_b takes a P11a InvestigationProposal
# + a SustainedMetricRegression target, measures the target metric on the
# current tree (before), opens an ISOLATED git-worktree sandbox, applies the
# advisory fix INSIDE that sandbox via a constrained headless-`claude` coding
# step, measures the target metric again in the sandbox (after), and ONLY keeps
# a committed branch when the metric IMPROVED past epsilon (strict `>`). On
# no-changes / diff-too-large / not-improved / ANY exception the sandbox is
# discarded so the primary worktree is NEVER mutated. This module imports no
# GitHub client and never shells out to the gh CLI; it opens NO pull request —
# actuation stops at a committed branch
# for human review. Determinism: `now` resolved ONCE at top via
# datetime.now(timezone.utc); the report timestamp is single-sourced from it.
# All external touchpoints (the git-worktree sandbox, the headless-claude coding
# step, the run_eval-backed measure) are injectable seams; their live defaults
# are exercised only by a model-gated/live test, exactly like P11a.
# Exports: ApplyResult, ActuationOutcome, ActuationReport, run_ralph_b.
# Deps: stdlib (dataclasses, datetime, logging, os, shutil, subprocess,
#       tempfile, pathlib); src.eval.runner.judge_cli (build_claude_argv —
#       constraint single-sourced); src.eval.runner.ralph (InvestigationProposal);
#       src.eval.runner.sustained_regression (SustainedMetricRegression).
# @end-summary
"""Ralph-B — a sandboxed actuation step for sustained eval regressions (P11b).

This is the *actuation* counterpart to P11a's proposal-only Ralph-A. Given a
:class:`~src.eval.runner.ralph.InvestigationProposal` (diagnosis + suggested
fix) and a :class:`~src.eval.runner.sustained_regression.SustainedMetricRegression`
target, :func:`run_ralph_b`:

1. measures the target ``(qtype, metric)`` on the CURRENT tree (``before``),
2. opens an ISOLATED sandbox (a dedicated ``git worktree`` by default),
3. applies the advisory fix INSIDE the sandbox via a constrained headless
   ``claude`` coding step,
4. re-measures the target metric in the sandbox (``after``),
5. keeps a committed branch ONLY when ``after - before > epsilon`` (strict).

Safety invariants (load-bearing):

* The primary worktree is NEVER mutated. Every edit lands in the sandbox; on
  no-changes, diff-too-large, not-improved, or ANY exception the sandbox is
  discarded. This module NEVER runs ``git checkout``/``restore``/``stash``/
  ``reset``/``clean`` against a tracked file in the primary worktree.
* This module imports no GitHub client and never invokes the gh CLI; it opens NO pull request. Actuation stops
  at a committed branch for a human to review and PR.

All three external seams — ``sandbox_factory``, ``apply_fn``, ``measure_fn`` —
are injectable. Their live defaults (a real ``git worktree`` sandbox, a
constrained headless-``claude`` coding step, and a ``run_eval``-backed measure)
perform real I/O and are documented as *live-only*: they are exercised solely
by a model-gated/live test, exactly like P11a's ``investigate_fn``. The fast
unit suite injects fakes for all three, so no ``claude`` process, git worktree,
or Weaviate/embeddings/LLM is ever touched.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.eval.runner.judge_cli import build_claude_argv
from src.eval.runner.ralph import InvestigationProposal
from src.eval.runner.sustained_regression import SustainedMetricRegression

logger = logging.getLogger("rag.eval.ralph_apply")

# Default coding model for the live apply step: Haiku is cheap and adequate for
# a small, targeted fix. (Live-only; the fast suite injects a fake apply_fn.)
_DEFAULT_APPLY_MODEL = "claude-haiku-4-5-20251001"

# System-prompt rubric for the live coding step. Unlike P11a's investigator,
# this step IS allowed to edit files — but ONLY inside its own worktree.
_APPLY_RUBRIC = (
    "You are a focused coding agent. Apply the smallest possible change that "
    "implements the suggested fix for the eval regression. Edit only files in "
    "the current working directory. Do NOT open a pull request, push, or touch "
    "any git remote. When done, briefly summarize what you changed."
)


# ---------------------------------------------------------------------------
# Frozen contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApplyResult:
    """What an ``apply_fn`` returns for one actuation attempt.

    :param changed: whether the apply step produced any file change at all.
    :param summary: a short prose summary of what was changed (advisory).
    :param diff_lines: the size of the change (numstat insertions+deletions),
        used to reject implausibly-large diffs before measuring.
    """

    changed: bool
    summary: str
    diff_lines: int


@dataclass(frozen=True)
class ActuationOutcome:
    """The outcome of a single Ralph-B actuation attempt.

    :param target_qtype: the golden question-type whose metric was actuated.
    :param target_metric: the metric label (``recall`` / ``mrr`` /
        ``faithfulness``).
    :param before_value: the metric on the current tree (``None`` if unknown).
    :param after_value: the metric in the sandbox after the fix; ``None`` when
        the after-measure was short-circuited (no-changes / diff-too-large).
    :param delta: ``after_value - before_value`` (``None`` when no after-measure).
    :param improved: whether ``delta`` strictly exceeds ``epsilon``.
    :param applied: whether a file change was produced at all.
    :param branch_name: the kept sandbox branch (``None`` unless improved).
    :param sandbox_path: the kept sandbox path (``None`` unless improved).
    :param reject_reason: why the attempt was rejected (``"no-changes"`` /
        ``"diff-too-large"``), or ``None`` when not rejected.
    """

    target_qtype: str
    target_metric: str
    before_value: float | None
    after_value: float | None
    delta: float | None
    improved: bool
    applied: bool
    branch_name: str | None
    sandbox_path: str | None
    reject_reason: str | None


@dataclass(frozen=True)
class ActuationReport:
    """The human-reviewable result of a Ralph-B run.

    :param pack_name: the pack actuated against.
    :param timestamp: ISO-8601 string of the run's resolved ``now``.
    :param target_qtype: the actuated question-type.
    :param target_metric: the actuated metric label.
    :param diagnosis: the P11a diagnosis carried onto the report.
    :param suggested_fix: the P11a suggested fix carried onto the report.
    :param outcome: the :class:`ActuationOutcome`.
    :param notes: free-form advisory notes (e.g. the apply summary).
    """

    pack_name: str
    timestamp: str
    target_qtype: str
    target_metric: str
    diagnosis: str
    suggested_fix: str
    outcome: ActuationOutcome
    notes: tuple[str, ...]


# ---------------------------------------------------------------------------
# The actuation loop — run_ralph_b
# ---------------------------------------------------------------------------


def run_ralph_b(
    pack_name: str,
    *,
    proposal: InvestigationProposal,
    target: SustainedMetricRegression,
    packs_root: str,
    sandbox_factory: Callable[[str], Any] | None = None,
    apply_fn: Callable[..., Any] | None = None,
    measure_fn: Callable[..., float | None] | None = None,
    epsilon: float = 0.05,
    max_diff_lines: int = 200,
    now: datetime | None = None,
) -> ActuationReport:
    """Actuate ``proposal`` against ``target`` inside an isolated sandbox.

    Deterministic flow:

    1. measure ``before`` on the CURRENT tree (``working_dir`` omitted);
    2. open a sandbox via ``sandbox_factory``;
    3. apply the fix inside it via ``apply_fn``;
    4. if nothing changed → discard, ``reject_reason="no-changes"``,
       ``applied=False`` (the after-measure is NOT run);
    5. else if the diff is larger than ``max_diff_lines`` → discard,
       ``reject_reason="diff-too-large"`` (the after-measure is NOT run);
    6. else measure ``after`` in the sandbox, compute ``delta = after - before``,
       and keep a committed branch ONLY when ``delta > epsilon`` (strict);
       otherwise discard and drop the branch.

    The sandbox is ALWAYS discarded on any exception (then re-raised — see
    below), so the primary worktree is NEVER mutated. This function imports no
    GitHub client and never invokes the gh CLI; it opens NO pull request.

    **Exception policy (SA1 left this to SA2 — chosen: discard-then-re-raise).**
    If ANY step after the sandbox is opened raises, the sandbox is discarded
    (best-effort) and the original exception is RE-RAISED unchanged. Swallowing
    would hide a real actuation failure behind a misleadingly-clean report; the
    caller (CLI handler / orchestrator) decides how to surface it. The
    discard-before-reraise keeps the never-mutate invariant intact even on the
    failure path.

    :param pack_name: the pack to actuate against.
    :param proposal: the P11a diagnosis + suggested fix to apply.
    :param target: the regression being actuated (qtype + metric).
    :param packs_root: root holding pack source dirs (threaded to the live
        measure seam; unused by the fast fakes).
    :param sandbox_factory: ``(pack_name) -> sandbox`` exposing
        ``.path``/``.branch``/``.commit(msg)``/``.discard()``; ``None`` uses the
        live git-worktree sandbox.
    :param apply_fn: ``(proposal, *, sandbox) -> ApplyResult``; ``None`` uses the
        live constrained headless-claude coding step.
    :param measure_fn: ``(pack_name, qtype, metric, *, working_dir=None) ->
        float | None``; ``None`` uses the live ``run_eval``-backed measure.
    :param epsilon: improvement band; ``delta`` must be STRICTLY ``> epsilon``.
    :param max_diff_lines: reject (discard) any diff larger than this.
    :param now: injected timestamp, resolved ONCE at top; defaults to
        ``datetime.now(timezone.utc)``.
    :returns: an :class:`ActuationReport`.
    """
    # Resolve `now` ONCE at the top (determinism convention).
    if now is None:
        now = datetime.now(timezone.utc)
    timestamp = now.isoformat()

    measure = measure_fn if measure_fn is not None else _default_measure_fn
    make_sandbox = (
        sandbox_factory if sandbox_factory is not None else _default_sandbox_factory
    )
    apply = apply_fn if apply_fn is not None else _default_apply_fn

    # Measure the CURRENT tree before touching anything (no working_dir).
    before = measure(pack_name, target.qtype, target.metric)

    sandbox = make_sandbox(pack_name)
    try:
        apply_result = apply(proposal, sandbox=sandbox)
        notes: tuple[str, ...] = (
            (apply_result.summary,) if getattr(apply_result, "summary", "") else ()
        )

        # (4) Nothing changed → discard; do NOT run the after-measure.
        if not apply_result.changed:
            sandbox.discard()
            outcome = ActuationOutcome(
                target_qtype=target.qtype,
                target_metric=target.metric,
                before_value=before,
                after_value=None,
                delta=None,
                improved=False,
                applied=False,
                branch_name=None,
                sandbox_path=None,
                reject_reason="no-changes",
            )
            return _report(pack_name, timestamp, target, proposal, outcome, notes)

        # (5) Diff too large → discard; do NOT run the after-measure.
        if apply_result.diff_lines > max_diff_lines:
            sandbox.discard()
            outcome = ActuationOutcome(
                target_qtype=target.qtype,
                target_metric=target.metric,
                before_value=before,
                after_value=None,
                delta=None,
                improved=False,
                applied=True,
                branch_name=None,
                sandbox_path=None,
                reject_reason="diff-too-large",
            )
            return _report(pack_name, timestamp, target, proposal, outcome, notes)

        # (6) Measure the sandbox tree and decide on the strict-`>` epsilon gate.
        after = measure(
            pack_name, target.qtype, target.metric, working_dir=sandbox.path
        )
        delta = after - before
        improved = delta > epsilon  # STRICT: delta == epsilon is NOT improved.

        if improved:
            sandbox.commit(
                f"ralph-b: improve {target.qtype}/{target.metric} "
                f"by {delta:+.4f} (epsilon {epsilon})"
            )
            branch_name: str | None = sandbox.branch
            sandbox_path: str | None = sandbox.path
        else:
            sandbox.discard()
            branch_name = None
            sandbox_path = None

        outcome = ActuationOutcome(
            target_qtype=target.qtype,
            target_metric=target.metric,
            before_value=before,
            after_value=after,
            delta=delta,
            improved=improved,
            applied=True,
            branch_name=branch_name,
            sandbox_path=sandbox_path,
            reject_reason=None,
        )
        return _report(pack_name, timestamp, target, proposal, outcome, notes)
    except Exception:
        # Discard-then-re-raise: keep the never-mutate invariant on the failure
        # path, but never hide a real actuation failure behind a clean report.
        try:
            sandbox.discard()
        except Exception:  # pragma: no cover — best-effort cleanup.
            logger.exception("ralph-b: sandbox discard failed during error cleanup")
        raise


def _report(
    pack_name: str,
    timestamp: str,
    target: SustainedMetricRegression,
    proposal: InvestigationProposal,
    outcome: ActuationOutcome,
    notes: tuple[str, ...],
) -> ActuationReport:
    """Assemble an :class:`ActuationReport` from the resolved parts."""
    return ActuationReport(
        pack_name=pack_name,
        timestamp=timestamp,
        target_qtype=target.qtype,
        target_metric=target.metric,
        diagnosis=proposal.diagnosis,
        suggested_fix=proposal.suggested_fix,
        outcome=outcome,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Live-only default seams (exercised ONLY by the model-gated/live test)
# ---------------------------------------------------------------------------


def _default_measure_fn(
    pack_name: str,
    qtype: str,
    metric: str,
    *,
    working_dir: str | None = None,
) -> float | None:
    """Live default measure — wraps ``run_eval`` for the target ``(qtype, metric)``.

    LIVE-ONLY. With ``working_dir=None`` it runs the eval in-process against the
    current tree and reads ``getattr(result.report, f"{metric}_by_qtype")[qtype]``.
    With ``working_dir`` set it must measure the SANDBOX tree instead, so it
    shells out to a fresh ``python -m src.eval.cli run`` with ``cwd=working_dir``
    and parses the per-qtype metric from the JSON report. Kept simple and
    documented; the fast suite injects a fake so this path is never unit-tested.

    :param pack_name: the pack/pack-path to evaluate.
    :param qtype: the golden question-type to read the metric for.
    :param metric: ``recall`` / ``mrr`` / ``faithfulness``.
    :param working_dir: ``None`` = current tree (in-process); otherwise the
        sandbox tree (measured via a subprocess rooted there).
    :returns: the per-qtype metric value, or ``None`` when absent.
    """
    if working_dir is None:
        # In-process measure against the current tree (lazy import: avoid a
        # module-import cycle, since cli imports the runner package).
        from src.eval.cli import run_eval

        result = run_eval(pack_name)
        by_qtype = getattr(result.report, f"{metric}_by_qtype", {}) or {}
        return by_qtype.get(qtype)

    # Sandbox measure: run the eval CLI with cwd rooted in the sandbox tree so
    # the freshly-applied edits are exercised, then parse the per-qtype metric
    # from the emitted JSON report.
    import json

    completed = subprocess.run(
        ["python", "-m", "src.eval.cli", "run", pack_name, "--format", "json"],
        cwd=working_dir,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in (0, 1):  # 0=gate pass, 1=gate fail (both valid)
        logger.warning(
            "ralph-b: sandbox measure exited %s: %s",
            completed.returncode,
            (completed.stderr or "").strip()[:500],
        )
        return None
    try:
        payload = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError):
        logger.warning("ralph-b: sandbox measure produced unparseable JSON")
        return None
    by_qtype = (payload.get("report") or {}).get(f"{metric}_by_qtype") or {}
    return by_qtype.get(qtype)


class _GitWorktreeSandbox:
    """An isolated ``git worktree`` sandbox rooted in a temp directory.

    LIVE-ONLY. Created by :func:`_default_sandbox_factory`. Every mutation lands
    in this worktree's own directory; ``commit`` stages + commits INSIDE the
    worktree, and ``discard`` removes the worktree, deletes the branch, and
    prunes. It touches ONLY its own worktree — it NEVER runs git
    checkout/restore/stash/reset/clean against a tracked file in the primary
    worktree.
    """

    def __init__(self, pack_name: str, *, base_ref: str = "develop") -> None:
        filesafe_ts = datetime.now(timezone.utc).isoformat().replace(":", "")
        self.branch = f"ralph-b/{pack_name}-{filesafe_ts}"
        self._tmp = tempfile.mkdtemp(prefix="ralph-b-")
        self.path = self._tmp
        # Create a fresh worktree on a new branch off the base ref. Touches only
        # the new worktree dir; the primary worktree is untouched.
        subprocess.run(
            ["git", "worktree", "add", "-b", self.branch, self._tmp, base_ref],
            check=True,
            capture_output=True,
            text=True,
        )

    def commit(self, message: str) -> None:
        """Stage + commit everything INSIDE this worktree (no backticks in msg)."""
        subprocess.run(
            ["git", "add", "-A"], cwd=self._tmp, check=True, capture_output=True
        )
        # Use ``-F -`` (message on stdin) so a message that ever needs code
        # spans cannot be command-substituted by a shell; here there are none.
        subprocess.run(
            ["git", "commit", "-F", "-"],
            cwd=self._tmp,
            input=message,
            text=True,
            check=True,
            capture_output=True,
        )

    def discard(self) -> None:
        """Remove the worktree, delete the branch, and prune — best effort."""
        subprocess.run(
            ["git", "worktree", "remove", "--force", self._tmp],
            capture_output=True,
            text=True,
        )
        if os.path.isdir(self._tmp):  # belt-and-braces if remove left the dir
            shutil.rmtree(self._tmp, ignore_errors=True)
        subprocess.run(
            ["git", "branch", "-D", self.branch], capture_output=True, text=True
        )
        subprocess.run(
            ["git", "worktree", "prune"], capture_output=True, text=True
        )


def _default_sandbox_factory(pack_name: str) -> _GitWorktreeSandbox:
    """Live default sandbox factory — a fresh :class:`_GitWorktreeSandbox`."""
    return _GitWorktreeSandbox(pack_name)


def _default_apply_fn(
    proposal: InvestigationProposal, *, sandbox: Any
) -> ApplyResult:
    """Live default apply step — a constrained headless-``claude`` coding call.

    LIVE-ONLY. Reuses the constraint philosophy of
    :func:`~src.eval.runner.judge_cli.build_claude_argv` (pinned model, full
    system-prompt replace) but runs with ``cwd=sandbox.path`` so every edit is
    confined to the sandbox worktree. After the call it computes ``diff_lines``
    from ``git diff --numstat`` inside the sandbox. The fast suite injects a fake
    apply_fn, so this path is never unit-tested.
    """
    prompt = (
        "A sustained eval regression was diagnosed as follows:\n\n"
        f"Diagnosis: {proposal.diagnosis}\n"
        f"Suggested fix: {proposal.suggested_fix}\n\n"
        "Apply the smallest change that implements the suggested fix. Edit only "
        "files under the current working directory."
    )
    # build_claude_argv pins the model + replaces the system prompt + clean-room
    # settings. The apply rubric (unlike the investigator) permits edits, but the
    # cwd confines them to the sandbox worktree.
    argv = build_claude_argv(
        prompt, model=_DEFAULT_APPLY_MODEL, system_prompt=_APPLY_RUBRIC
    )
    completed = subprocess.run(
        argv, cwd=sandbox.path, capture_output=True, text=True
    )
    summary = (completed.stdout or "").strip()[:500]

    numstat = subprocess.run(
        ["git", "diff", "--numstat"],
        cwd=sandbox.path,
        capture_output=True,
        text=True,
    )
    diff_lines = 0
    changed = False
    for line in (numstat.stdout or "").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            changed = True
            for col in parts[:2]:
                if col.isdigit():
                    diff_lines += int(col)
    return ApplyResult(changed=changed, summary=summary, diff_lines=diff_lines)
