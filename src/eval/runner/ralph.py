# @summary
# P11a — Ralph-A proposal-only eval-failure investigator. run_ralph_a reads a
# pack's run-history (#140) + sustained-regression detector (#141), ranks the
# worst regressions worst-first (rank_regressions; exact key = (latest_delta,
# -runs_regressed, qtype, metric)), and for the top-K invokes a CONSTRAINED
# headless-`claude` investigator (reusing P14 judge_cli.build_claude_argv +
# _strip_json_fence so the cost/safety constraint stays single-sourced) to
# produce a diagnosis + suggested-fix, returning a human-reviewable RalphReport.
# It NEVER applies patches, commits, opens PRs, or merges: this module imports
# NO git, runs NO subprocess except the read-only investigator CLI call, and
# writes NOTHING (persistence is a separate explicit function the CLI calls).
# Determinism: `now` resolved ONCE at top via datetime.now(timezone.utc);
# build_investigation_context is pure.
# Exports: InvestigationProposal, RalphInvestigation, RalphReport,
#          rank_regressions, select_worst_regression, build_investigation_context,
#          build_ralph_investigator, run_ralph_a.
# Deps: stdlib (json, dataclasses, datetime, logging); src.eval.runner.judge_cli
#       (build_claude_argv, _strip_json_fence, JudgeBackendError — constraint
#       single-sourced); src.eval.runner.run_history (load_run_history);
#       src.eval.runner.sustained_regression (detect_sustained_regressions);
#       src.eval.pack (load_pack — optional, lazy).
# @end-summary
"""Ralph-A — a proposal-only investigator for sustained eval regressions.

This is the *read-only, human-in-the-loop* counterpart to a Ralph-style
auto-fixer. It builds on the run-history index (#140) and the sustained-
regression detector (#141): rank the worst sustained ``(qtype, metric)``
regressions, and for the top ``max_iterations`` ask a CONSTRAINED headless
``claude`` investigator to diagnose the likely cause and suggest a minimal fix.

The output — a :class:`RalphReport` — is purely advisory. By construction this
module:

* applies NO patches, makes NO commits, opens NO PRs, and merges NOTHING;
* imports NO git and performs NO subprocess EXCEPT the read-only investigator
  CLI call (itself tool-free + clean-room via
  :func:`~src.eval.runner.judge_cli.build_claude_argv`);
* writes NOTHING — persistence is a *separate* explicit function
  (:func:`~src.eval.runner.persistence.persist_ralph_report`) the CLI calls.

The investigator's cost/safety constraint is single-sourced with the headless
judge: :func:`run_ralph_a`'s default investigator reuses
:func:`~src.eval.runner.judge_cli.build_claude_argv` (``--allowed-tools ""``,
``--setting-sources ""``, ``--output-format json``, pinned ``--model``) and
:func:`~src.eval.runner.judge_cli._strip_json_fence`, so a future edit cannot
silently turn the investigator into the full tool-enabled coding agent.
"""
from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.eval.runner.judge_cli import (
    JudgeBackendError,
    _strip_json_fence,
    build_claude_argv,
)
from src.eval.runner.run_history import load_run_history
from src.eval.runner.sustained_regression import (
    SustainedMetricRegression,
    detect_sustained_regressions,
)

logger = logging.getLogger("rag.eval.ralph")

# Default investigator model: Haiku is cheap and adequate for a diagnosis.
_DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# System-prompt rubric: an investigator persona that NEVER applies changes.
_INVESTIGATOR_RUBRIC = (
    "You are an eval-regression investigator. Diagnose the likely cause of the "
    "regression and suggest a minimal fix. You do NOT apply changes."
)

# Appended to the investigation prompt to force a parseable, prose-free answer.
_JSON_ONLY_INSTRUCTION = (
    "\n\nRespond with ONLY a JSON object of the form "
    '{"diagnosis": <string>, "suggested_fix": <string>} '
    "— no prose, no markdown, no code fences."
)

# How many recent history entries to summarize in the investigation context.
_CONTEXT_TRAIL = 6
# How many goldens of the target qtype to include in the context (when a pack
# is provided).
_CONTEXT_GOLDENS = 3


# ---------------------------------------------------------------------------
# Frozen contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InvestigationProposal:
    """What an ``investigate_fn`` returns for one regression target.

    :param diagnosis: the investigator's likely-cause analysis.
    :param suggested_fix: a minimal suggested fix (advisory — never applied).
    """

    diagnosis: str
    suggested_fix: str


@dataclass(frozen=True)
class RalphInvestigation:
    """One recorded investigation: the target fields flattened + the proposal.

    :param qtype: the golden question-type the regressed metric aggregates over.
    :param metric: the regressed metric label (``recall`` / ``mrr`` /
        ``faithfulness``).
    :param runs_regressed: how many considered runs regressed for this pair.
    :param window: the number of recent runs the detector considered.
    :param latest_delta: the most-recent delta for this pair (negative = worse).
    :param diagnosis: the investigator's likely-cause analysis.
    :param suggested_fix: the investigator's advisory minimal fix.
    """

    qtype: str
    metric: str
    runs_regressed: int
    window: int
    latest_delta: float
    diagnosis: str
    suggested_fix: str


@dataclass(frozen=True)
class RalphReport:
    """The human-reviewable result of a Ralph-A run.

    :param pack_name: the pack investigated.
    :param timestamp: ISO-8601 string of the run's resolved ``now``.
    :param sustained_count: how many sustained regressions the detector found
        (may exceed ``len(investigations)`` when ``max_iterations`` truncates).
    :param max_iterations: the top-K cap applied to investigations.
    :param investigations: the recorded investigations, worst-first.
    """

    pack_name: str
    timestamp: str
    sustained_count: int
    max_iterations: int
    investigations: tuple[RalphInvestigation, ...]


# ---------------------------------------------------------------------------
# Ranking — deterministic, worst-first
# ---------------------------------------------------------------------------


def rank_regressions(
    sustained: Any,
) -> list[SustainedMetricRegression]:
    """Sort sustained regressions worst-first, deterministically.

    The EXACT key is ``(latest_delta, -runs_regressed, qtype, metric)``:

    * most-negative ``latest_delta`` first (the worst regression),
    * ties broken by MORE ``runs_regressed`` (``-runs_regressed`` ascending),
    * then ``qtype`` then ``metric`` for total determinism.

    :param sustained: an iterable of :class:`SustainedMetricRegression`.
    :returns: a new worst-first sorted list.
    """
    return sorted(
        sustained,
        key=lambda r: (r.latest_delta, -r.runs_regressed, r.qtype, r.metric),
    )


def select_worst_regression(
    sustained: Any,
) -> SustainedMetricRegression | None:
    """Return the single worst sustained regression, or ``None`` when empty."""
    ranked = rank_regressions(sustained)
    return ranked[0] if ranked else None


# ---------------------------------------------------------------------------
# Investigation context — deterministic, pure
# ---------------------------------------------------------------------------


def build_investigation_context(
    target: SustainedMetricRegression,
    history: list[dict],
    pack: Any | None = None,
) -> str:
    """Build a deterministic, pure investigation-context string for *target*.

    Summarizes (no I/O beyond what is passed in):

    * the target — qtype, metric, latest_delta, runs_regressed, window;
    * the per-qtype delta trail for that ``(qtype, metric)`` across the most
      recent history entries (timestamp + delta, oldest→newest);
    * IF ``pack`` is provided — up to :data:`_CONTEXT_GOLDENS` goldens of the
      target qtype (qid + query + expected_answer_span).

    :param target: the regression to investigate.
    :param history: the run-history entries (timestamp-ascending).
    :param pack: an optional :class:`~src.eval.pack.schema.EvalPack` for golden
        context; ``None`` omits the goldens section.
    :returns: a stable multi-line context string (identical for identical
        inputs).
    """
    lines: list[str] = []
    lines.append("# Eval regression to investigate")
    lines.append(f"qtype: {target.qtype}")
    lines.append(f"metric: {target.metric}")
    lines.append(f"latest_delta: {target.latest_delta}")
    lines.append(
        f"runs_regressed: {target.runs_regressed} of window {target.window}"
    )
    lines.append("(delta = current - baseline; negative = worse)")

    # Per-qtype delta trail for THIS (qtype, metric), oldest→newest.
    trail = history[-_CONTEXT_TRAIL:]
    lines.append("")
    lines.append("## Recent delta trail for this (qtype, metric)")
    any_trail = False
    for entry in trail:
        per_qtype = entry.get("per_qtype_deltas") or {}
        metric_map = per_qtype.get(target.qtype) or {}
        if target.metric not in metric_map:
            continue
        any_trail = True
        ts = entry.get("timestamp", "")
        lines.append(f"- {ts}: delta={metric_map[target.metric]}")
    if not any_trail:
        lines.append("(no recorded deltas for this pair in the recent trail)")

    # Optional goldens of the target qtype.
    if pack is not None:
        goldens = getattr(pack, "goldens", {}) or {}
        qtype_goldens = list(goldens.get(target.qtype, []))[:_CONTEXT_GOLDENS]
        if qtype_goldens:
            lines.append("")
            lines.append(f"## Sample {target.qtype} goldens")
            for g in qtype_goldens:
                span = getattr(g, "expected_answer_span", None)
                lines.append(
                    f"- qid={getattr(g, 'qid', '')} "
                    f"query={getattr(g, 'query', '')!r} "
                    f"expected_answer_span={span!r}"
                )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Default investigator — constrained headless `claude` (reuses judge_cli)
# ---------------------------------------------------------------------------


def _default_run_fn(argv: list[str], *, timeout: float):
    """Default runner: spawn the real ``claude`` CLI via subprocess.run.

    This is the ONLY subprocess this module ever performs — a read-only,
    tool-free, clean-room investigation call (the argv is built by
    :func:`~src.eval.runner.judge_cli.build_claude_argv`).
    """
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _build_investigation_prompt(
    target: SustainedMetricRegression, context: str
) -> str:
    """Assemble the investigation prompt (target + context + JSON instruction)."""
    return (
        "Investigate the following sustained eval regression and propose a "
        "minimal fix. Do NOT apply any changes.\n\n"
        f"{context}"
        f"{_JSON_ONLY_INSTRUCTION}"
    )


def _parse_proposal_envelope(stdout: str) -> InvestigationProposal:
    """Parse the ``--output-format json`` envelope into an InvestigationProposal.

    Reuses :func:`~src.eval.runner.judge_cli._strip_json_fence` so fence
    handling stays single-sourced with the judge backend. Raises
    :class:`~src.eval.runner.judge_cli.JudgeBackendError` on any failure.
    """
    try:
        envelope = json.loads(stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise JudgeBackendError(
            f"ralph investigator returned unparseable envelope JSON: {exc}"
        ) from exc

    if envelope.get("is_error") is not False:
        raise JudgeBackendError(
            f"ralph investigator envelope reported an error: "
            f"is_error={envelope.get('is_error')!r}"
        )
    if envelope.get("subtype") != "success":
        raise JudgeBackendError(
            f"ralph investigator envelope subtype was not 'success': "
            f"{envelope.get('subtype')!r}"
        )

    result = envelope.get("result")
    if not result or not isinstance(result, str):
        raise JudgeBackendError(
            f"ralph investigator envelope missing/empty .result: {result!r}"
        )

    payload_text = _strip_json_fence(result)
    try:
        payload = json.loads(payload_text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise JudgeBackendError(
            f"ralph investigator .result was not parseable JSON: {exc}"
        ) from exc

    try:
        diagnosis = str(payload["diagnosis"])
        suggested_fix = str(payload["suggested_fix"])
    except (KeyError, TypeError) as exc:
        raise JudgeBackendError(
            f"ralph investigator .result missing diagnosis/suggested_fix: {exc}"
        ) from exc

    return InvestigationProposal(diagnosis=diagnosis, suggested_fix=suggested_fix)


def build_ralph_investigator(
    *,
    model: str = _DEFAULT_MODEL,
    run_fn: Callable[..., Any] | None = None,
    timeout: float = 60.0,
) -> Callable[[SustainedMetricRegression, str], InvestigationProposal]:
    """Build the default ``investigate_fn`` — a constrained headless investigator.

    The returned callable, given ``(target, context)``:

    * builds an investigation prompt (context + a strict JSON-only instruction),
    * builds the constrained argv via
      :func:`~src.eval.runner.judge_cli.build_claude_argv` (the cost/safety
      constraint — ``--allowed-tools ""``, ``--setting-sources ""``,
      ``--output-format json``, pinned ``--model`` — is single-sourced there),
    * runs it via ``run_fn`` (default real :func:`subprocess.run`),
    * parses the envelope into an :class:`InvestigationProposal`, raising
      :class:`~src.eval.runner.judge_cli.JudgeBackendError` on any failure.

    :param model: investigator model (Haiku by default — cheap).
    :param run_fn: testability seam ``(argv, *, timeout) -> obj`` exposing
        ``.returncode`` / ``.stdout`` / ``.stderr``; ``None`` uses subprocess.
    :param timeout: per-call subprocess timeout in seconds.
    """
    runner = run_fn if run_fn is not None else _default_run_fn

    def _investigate(
        target: SustainedMetricRegression, context: str
    ) -> InvestigationProposal:
        prompt = _build_investigation_prompt(target, context)
        argv = build_claude_argv(
            prompt, model=model, system_prompt=_INVESTIGATOR_RUBRIC
        )
        try:
            completed = runner(argv, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise JudgeBackendError(
                f"ralph investigator timed out after {timeout}s"
            ) from exc

        if completed.returncode != 0:
            raise JudgeBackendError(
                f"ralph investigator exited with code {completed.returncode}: "
                f"{(completed.stderr or '').strip()[:500]}"
            )

        return _parse_proposal_envelope(completed.stdout)

    return _investigate


# ---------------------------------------------------------------------------
# The loop — run_ralph_a
# ---------------------------------------------------------------------------


def run_ralph_a(
    pack_name: str,
    *,
    history_dir: str | Path,
    pack_dir: str | Path | None = None,
    max_iterations: int = 3,
    window: int = 3,
    min_regressed: int = 2,
    epsilon: float = 0.05,
    investigate_fn: Callable[
        [SustainedMetricRegression, str], InvestigationProposal
    ]
    | None = None,
    now: datetime | None = None,
) -> RalphReport:
    """Investigate a pack's worst sustained regressions — proposal-only.

    Reads the run-history (:func:`~src.eval.runner.run_history.load_run_history`),
    detects sustained regressions
    (:func:`~src.eval.runner.sustained_regression.detect_sustained_regressions`),
    ranks them worst-first (:func:`rank_regressions`), and for the top
    ``max_iterations`` invokes ``investigate_fn`` to produce a diagnosis +
    suggested fix. Returns a :class:`RalphReport`.

    NEVER mutates anything: imports no git, performs no subprocess except the
    read-only investigator CLI call, and writes nothing (persistence is the
    caller's explicit concern via
    :func:`~src.eval.runner.persistence.persist_ralph_report`).

    :param pack_name: the pack whose ``<pack_name>.history.jsonl`` lives under
        ``history_dir``.
    :param history_dir: directory holding the run-history index.
    :param pack_dir: optional pack directory for golden context; a missing/bad
        pack dir degrades to no-goldens rather than crashing the run.
    :param max_iterations: top-K worst regressions to investigate (default 3).
    :param window: detector lookback (default 3).
    :param min_regressed: min regressed runs in the window to flag (default 2).
    :param epsilon: regression tolerance band, strict ``delta < -epsilon``.
    :param investigate_fn: the per-target investigator; ``None`` lazily builds
        the default constrained headless investigator — but ONLY when there is
        something to investigate.
    :param now: injected timestamp, resolved ONCE at top for determinism;
        defaults to ``datetime.now(timezone.utc)``.
    :returns: a :class:`RalphReport`.
    """
    # Resolve `now` ONCE at the top (determinism convention).
    if now is None:
        now = datetime.now(timezone.utc)

    history = load_run_history(pack_name, history_dir)
    sustained = detect_sustained_regressions(
        history,
        window=window,
        min_regressed=min_regressed,
        epsilon=epsilon,
    )

    # No sustained regressions → never invoke the investigator.
    if not sustained:
        return RalphReport(
            pack_name=pack_name,
            timestamp=now.isoformat(),
            sustained_count=0,
            max_iterations=max_iterations,
            investigations=(),
        )

    ranked = rank_regressions(sustained)
    targets = ranked[:max_iterations]

    # Lazily build the default investigator — only now that there is work.
    if investigate_fn is None:
        investigate_fn = build_ralph_investigator()

    # Optional pack load for golden context — a missing/bad pack dir must NOT
    # crash the investigation (deliberate broad-ish degrade, narrowed to the
    # known load failures).
    pack = None
    if pack_dir is not None:
        from src.eval.pack import load_pack
        from src.eval.pack.errors import PackValidationError

        try:
            pack = load_pack(pack_dir)
        except (PackValidationError, FileNotFoundError, OSError) as exc:
            logger.warning(
                "ralph: could not load pack at %s for golden context (%s); "
                "proceeding without goldens",
                pack_dir,
                exc,
            )
            pack = None

    investigations: list[RalphInvestigation] = []
    for target in targets:
        context = build_investigation_context(target, history, pack)
        proposal = investigate_fn(target, context)
        investigations.append(
            RalphInvestigation(
                qtype=target.qtype,
                metric=target.metric,
                runs_regressed=target.runs_regressed,
                window=target.window,
                latest_delta=target.latest_delta,
                diagnosis=proposal.diagnosis,
                suggested_fix=proposal.suggested_fix,
            )
        )

    return RalphReport(
        pack_name=pack_name,
        timestamp=now.isoformat(),
        sustained_count=len(sustained),
        max_iterations=max_iterations,
        investigations=tuple(investigations),
    )
