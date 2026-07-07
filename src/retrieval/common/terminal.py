# @summary
# Terminal-decision contracts for the RAG pipeline: stage outcomes, ask_user
# reasons, and the single TerminalDecision dataclass returned by
# ``_decide_terminal``. Centralises the choice between answer/search/ask_user/
# canned so the orchestrator no longer scatters that decision across stages.
# Exports: AskUserReason, StageStatus, StageOutcome, TerminalAction,
#          TerminalDecision, decide_terminal
# Deps: dataclasses, enum, typing
# @end-summary
"""Typed contracts for the centralised terminal-decision step in RAGChain.run.

Why this exists:
    Previously RAGChain.run had 8 scattered ``action="ask_user"`` early-return
    sites, each with its own copy of the deep-research bypass / budget /
    no-results gating. Adding a new control mode (deep research, future agent
    modes) required patching every site; one miss equalled a silent bug.

    This module replaces that pattern with a single function — ``decide_terminal``
    — that maps the recorded pipeline state (``StageOutcome`` list + a few
    booleans) to a typed ``TerminalDecision``. The decision is the *only*
    place that may produce ``ask_user``.

    The companion ``AskUserReason`` enum surfaces a typed reason on the
    response so UIs can render distinct messages for "vague query" vs
    "budget exhausted" vs "no results".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional


class AskUserReason(str, Enum):
    """Why the pipeline gave up and asked the user for help.

    Inherits from ``str`` so ``dataclasses.asdict`` and orjson serialise
    members as their string values without custom encoders.
    """

    SANITIZER_REJECT = "sanitizer_reject"
    INJECTION_BLOCKED = "injection_blocked"
    VAGUE_QUERY = "vague_query"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_RESULTS = "no_results"


class StageStatus(str, Enum):
    """Outcome status recorded by each pipeline stage onto ``outcomes``."""

    OK = "ok"
    SKIPPED = "skipped"
    BUDGET_EXHAUSTED = "budget_exhausted"
    EMPTY = "empty"
    ERROR = "error"


@dataclass(frozen=True)
class StageOutcome:
    """A single stage's terminal status, appended to ``PipelineState.outcomes``.

    Stages no longer early-return on failure; they record an outcome and
    continue. ``decide_terminal`` reads the list at the end of the run.
    """

    stage: str
    status: StageStatus
    detail: Optional[str] = None


TerminalAction = Literal["answer", "search", "ask_user", "canned"]


@dataclass(frozen=True)
class TerminalDecision:
    """Output of ``decide_terminal``.

    Attributes:
        action: One of answer/search/ask_user/canned.
        ask_user_reason: Populated only when action == "ask_user".
        degraded: True when the pipeline took a fallback path (eg. BM25-only
            re-search after the primary hybrid call returned 0 hits) but still
            produced usable results.
        clarification_message: Optional human-readable message; when None,
            callers should fall back to a default message keyed off the
            reason.
    """

    action: TerminalAction
    ask_user_reason: Optional[AskUserReason] = None
    degraded: bool = False
    clarification_message: Optional[str] = None


def _has_outcome(outcomes: list[StageOutcome], stage: str, status: StageStatus) -> bool:
    return any(o.stage == stage and o.status == status for o in outcomes)


def _any_outcome(outcomes: list[StageOutcome], status: StageStatus) -> bool:
    return any(o.status == status for o in outcomes)


def decide_terminal(
    outcomes: list[StageOutcome],
    *,
    deep_research: bool,
    has_results: bool,
    sanitizer_verdict: Optional[str] = None,
    confidence_ask_user: bool = False,
    degraded: bool = False,
) -> TerminalDecision:
    """Map pipeline state to a single terminal decision.

    This is the only function in the codebase allowed to choose
    ``action="ask_user"``. Callers funnel every "we're stuck" path through
    here so adding a new control mode (eg. deep research) requires editing
    one place, not eight.

    Args:
        outcomes: Per-stage outcomes recorded as the pipeline ran.
        deep_research: True when the caller opted into the DR orchestrator.
            DR has its own budget and decomposer, so most "give up" gates
            do NOT apply when it's on.
        has_results: True when the pipeline produced at least one result
            (post-rerank, or post-degraded fallback).
        sanitizer_verdict: Optional verdict from the merge-gate sanitizer:
            ``"reject"`` (hard block — empty/injection), ``"canned"`` (canned
            response from rails), or None.
        confidence_ask_user: True when the LLM-confidence loop said the
            query is too vague to retrieve. Bypassed under DR.
        degraded: True when the chain already took a fallback path; surfaced
            as-is on the decision.
    """
    # 1) Sanitizer hard rejects always win — they represent a request-level
    #    block (empty/injection), not a retrieval-quality concern. DR cannot
    #    override them.
    if sanitizer_verdict == "reject":
        # Distinguish injection vs empty/other by scanning outcomes.
        reason = AskUserReason.SANITIZER_REJECT
        if _has_outcome(outcomes, "input_rails", StageStatus.ERROR):
            reason = AskUserReason.INJECTION_BLOCKED
        return TerminalDecision(action="ask_user", ask_user_reason=reason, degraded=degraded)
    if sanitizer_verdict == "canned":
        return TerminalDecision(action="canned", degraded=degraded)

    # 2) Vague-query verdict from the LLM-confidence loop. DR's recursive
    #    decomposer is built to recover from messy/terse queries, so we
    #    bypass this gate when DR is active.
    if confidence_ask_user and not deep_research:
        return TerminalDecision(
            action="ask_user",
            ask_user_reason=AskUserReason.VAGUE_QUERY,
            degraded=degraded,
        )

    # 3) Budget exhaustion. If we have any results, return them with
    #    action="search" (don't give up on partial state). With nothing to
    #    show and DR off, ask the user. DR carries its own budget so we
    #    stay quiet.
    budget_blown = _any_outcome(outcomes, StageStatus.BUDGET_EXHAUSTED)
    if budget_blown and not deep_research:
        if has_results:
            return TerminalDecision(action="search", degraded=degraded)
        return TerminalDecision(
            action="ask_user",
            ask_user_reason=AskUserReason.BUDGET_EXHAUSTED,
            degraded=degraded,
        )

    # 4) No results. DR may have its own pools so it skips this gate.
    if not has_results and not deep_research:
        return TerminalDecision(
            action="ask_user",
            ask_user_reason=AskUserReason.NO_RESULTS,
            degraded=degraded,
        )

    # 5) Happy path / DR path with whatever results we have.
    return TerminalDecision(action="answer", degraded=degraded)
