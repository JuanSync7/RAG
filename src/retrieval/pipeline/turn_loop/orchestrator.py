# @summary
# The turn loop itself: run_turn_loop drives controller-decide -> dispatch
# (RETRIEVE / DEEP_STUDY / CLARIFY / ANSWER) iterations while the budgets hold
# (max_actions, max_llm_calls, wall clock), emitting a turn_action event per
# decision. CLARIFY and a gate-passing ANSWER are terminal; a failed gate
# feeds GateFeedback back into the next decision. Budget exhaustion exits
# best-effort: the best failed draft, else one final draft if the LLM ledger
# allows, else an explicit user-facing cannot-answer-confidently message
# (source names + capped previews only; the internal evidence digest is
# logged, never returned) — never an empty response. Pure control flow over
# TurnLoopDeps; per-action errors are contained so one broken seam degrades,
# not crashes, a turn.
# Exports: run_turn_loop
# Deps: src.retrieval.pipeline.turn_loop.{schemas,events,controller,retrieve,
#       deep_study,clarify,answer}
# @end-summary
"""Turn-loop orchestrator: budgets, dispatch, stop conditions (design §5).

Pure, DI-injected control flow: every capability (retrieval activity,
document fetch, LLM provider, event sink) arrives through
:class:`TurnLoopDeps`, so this module imports no Temporal/MinIO/Redis/
Weaviate/server code and tests drive the full loop with fakes.

Input safety (sanitize + PII gate) is NOT run here — the server layer applies
it once, before the loop starts (design §3). The accepted answer is returned
in :class:`TurnLoopResult`; replaying it as standard ``token`` SSE events is
likewise the endpoint's job.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.retrieval.pipeline.turn_loop.answer import run_answer
from src.retrieval.pipeline.turn_loop.clarify import run_clarify
from src.retrieval.pipeline.turn_loop.controller import (
    build_evidence_digest,
    decide,
)
from src.retrieval.pipeline.turn_loop.decompose import run_decompose
from src.retrieval.pipeline.turn_loop.deep_study import run_deep_study
from src.retrieval.pipeline.turn_loop.events import TurnEventEmitter
from src.retrieval.pipeline.turn_loop.retrieve import run_retrieve
from src.retrieval.pipeline.turn_loop.router import RouteHint
from src.retrieval.pipeline.turn_loop.schemas import (
    AnswerArgs,
    DecomposeArgs,
    DeepStudyArgs,
    RetrieveArgs,
    TurnAction,
    TurnBudget,
    TurnContext,
    TurnDecision,
    TurnEventType,
    TurnLoopDeps,
    TurnLoopResult,
    TurnState,
)

logger = logging.getLogger(__name__)

# Stop-reason vocabulary (TurnLoopResult.stop_reason).
STOP_GATE_PASSED = "gate_passed"
STOP_CLARIFY = "clarify"
STOP_MAX_ACTIONS = "max_actions"
STOP_MAX_LLM_CALLS = "max_llm_calls"
STOP_WALL_CLOCK = "wall_clock"
STOP_MAX_ANSWER_ATTEMPTS = "max_answer_attempts"
STOP_ERROR = "error"


def _budget_stop_reason(state: TurnState, budget: TurnBudget) -> Optional[str]:
    """Name the first exhausted loop budget, or ``None`` while all hold.

    The wall clock stops the loop while ``min_call_budget_ms`` still remains
    (not at zero): every action needs at least one meaningful LLM call, so
    once the remainder can only fund near-zero-timeout calls the loop must go
    to its best-effort exit instead of firing doomed requests (observed live:
    1-2s controller/judge timeouts at the budget tail).
    """
    if state.elapsed_ms() >= budget.wall_clock_ms - budget.min_call_budget_ms:
        return STOP_WALL_CLOCK
    if state.iteration >= budget.max_actions:
        return STOP_MAX_ACTIONS
    if state.llm_calls >= budget.max_llm_calls:
        return STOP_MAX_LLM_CALLS
    return None


def _preview_chars() -> int:
    """Preview cap for user-facing evidence lines (the one preview knob)."""
    from config import settings

    return int(getattr(settings, "RAG_TURN_CONTEXT_PREVIEW_CHARS", 320))


def _cannot_answer_text(state: TurnState) -> str:
    """Deterministic best-effort answer when no draft could be produced.

    Explicit low-confidence framing (design §5: never an unexplained empty
    response; low-confidence-warning precedent) rendered USER-FACING: source
    display names, headings and capped text previews only. The controller's
    internal evidence digest (document_id/source_key storage keys, tried
    queries/HyDE texts, judge internals) is prompt-internal — it is logged
    for observability but must never reach the client or be persisted as an
    assistant turn (it would also re-enter future prompts via recent_turns).
    """
    logger.info(
        "turn loop exiting without a draft; internal evidence digest:\n%s",
        build_evidence_digest(state),
    )
    lines = ["I could not reach a confident answer within this turn's budget."]
    preview_cap = _preview_chars()
    seen: set[tuple[str, str]] = set()
    evidence_lines: list[str] = []
    for chunk in state.pool:
        key = (chunk.source, chunk.heading)
        if key in seen:
            continue
        seen.add(key)
        label = chunk.source or "unnamed source"
        if chunk.heading:
            label += f" — {chunk.heading}"
        preview = " ".join((chunk.text or "").split())
        if preview_cap > 0 and len(preview) > preview_cap:
            preview = preview[:preview_cap]
        evidence_lines.append(f"- {label}: {preview}" if preview else f"- {label}")
    if evidence_lines:
        lines.append("")
        lines.append("The evidence gathered so far comes from:")
        lines.extend(evidence_lines)
    else:
        lines.append(
            "No relevant evidence was found for this question — rephrasing "
            "or narrowing it may help."
        )
    return "\n".join(lines)


def _result(
    state: TurnState,
    *,
    answer: str = "",
    action: str = TurnLoopResult.ACTION_ANSWERED,
    clarification=None,
    confidence: float = 0.0,
    stop_reason: str = "",
) -> TurnLoopResult:
    """Assemble the terminal result from the turn state."""
    return TurnLoopResult(
        answer=answer,
        action=action,
        clarification=clarification,
        confidence=confidence,
        pool=state.pool,
        trace=state.events,
        stop_reason=stop_reason,
        llm_calls=state.llm_calls,
        actions_taken=state.iteration,
        elapsed_ms=state.elapsed_ms(),
    )


async def _best_effort_result(
    *,
    stop_reason: str,
    best_draft: Optional[tuple[float, str]],
    query: str,
    context: TurnContext,
    state: TurnState,
    budget: TurnBudget,
    deps: TurnLoopDeps,
    emitter: TurnEventEmitter,
) -> TurnLoopResult:
    """Exit the exhausted loop with the best answer still producible.

    Preference order (design §5 / track spec): the best gate-scored draft
    already made; else ONE final draft attempt when the LLM ledger still
    allows; else the explicit cannot-answer-confidently digest. The budget
    name that ended the loop is preserved as the (low-confidence) stop reason.
    """
    if best_draft is None and state.llm_calls < budget.max_llm_calls:
        try:
            draft, feedback = await run_answer(
                query=query,
                context=context,
                state=state,
                budget=budget,
                deps=deps,
                emitter=emitter,
            )
            if draft:
                best_draft = (feedback.score, draft)
        except Exception as exc:  # noqa: BLE001 — best-effort must not raise
            logger.warning("turn loop final best-effort draft failed: %s", exc)
    if best_draft is not None:
        confidence, answer = best_draft
        return _result(
            state,
            answer=answer,
            confidence=confidence,
            stop_reason=stop_reason,
        )
    return _result(
        state,
        answer=_cannot_answer_text(state),
        confidence=0.0,
        stop_reason=stop_reason,
    )


def _seed_decision(
    route_hint: Optional[RouteHint], state: TurnState, query: str
) -> Optional[TurnDecision]:
    """The fast-lane's deterministic opening decision, or ``None``.

    When the router took the fast lane, the first two iterations are fixed —
    a RETRIEVE (iteration 0) then, once it yielded a pool, an ANSWER (iteration
    1) — with NO controller LLM call between them (the ~2-round-trip latency
    win). Deliberately narrow so the fast lane always degrades to the
    controller (fail-open): a later iteration, an empty pool, or a prior gate
    failure returns ``None`` here, handing the turn to :func:`decide`. A
    non-fast-lane hint also returns ``None`` — that seed is advisory-only and
    reaches the controller through the prompt, never as a forced decision.
    """
    if route_hint is None or not route_hint.fast_lane:
        return None
    if state.iteration == 0:
        return TurnDecision(
            action=TurnAction.RETRIEVE,
            reason=(
                "router fast-lane: high-confidence self-contained factoid — "
                "retrieving directly (controller skipped)"
            ),
            confidence=0.0,
            args=RetrieveArgs(query_text=query),
        )
    if state.iteration == 1 and state.pool and state.last_gate is None:
        return TurnDecision(
            action=TurnAction.ANSWER,
            reason=(
                "router fast-lane: pool covers the factoid — answering directly "
                "(controller skipped)"
            ),
            confidence=0.0,
            args=AnswerArgs(),
        )
    return None


def _no_progress_decision(
    state: TurnState, budget: TurnBudget, no_progress_rounds: int
) -> Optional[TurnDecision]:
    """Force an ANSWER when evidence-gathering has stalled, else ``None``.

    Guards two forms of the same stall: (a) a weak controller re-retrieving
    already-pooled chunks after the judge marked the pool sufficient, and (b) a
    round judge that rejects every fresh batch so the judged pool never grows —
    the empty-pool spiral that otherwise burns every action to a max_actions
    best-effort exit (draft from "(no evidence retrieved)"). Once
    ``max_no_progress_rounds`` consecutive gather rounds add ZERO new judged
    chunks, the loop answers from whatever grounding it has — the judged pool if
    non-empty, else the best-effort ``fallback_chunks`` floor. Only fires while
    answer attempts remain — once they're spent the ANSWER branch's own cap
    takes the turn to best-effort. Disabled when ``max_no_progress_rounds <= 0``.
    """
    if (
        budget.max_no_progress_rounds > 0
        and no_progress_rounds >= budget.max_no_progress_rounds
        and (state.pool or state.fallback_chunks)
        and state.answer_attempts < budget.max_answer_attempts
    ):
        return TurnDecision(
            action=TurnAction.ANSWER,
            reason=(
                f"loop guard: {no_progress_rounds} consecutive retrieval rounds "
                "added no new evidence — answering from the pool gathered so far"
            ),
            confidence=0.0,
            args=AnswerArgs(),
        )
    return None


# Evidence-gathering actions (grow the pool); the no-progress guard tracks
# consecutive rounds of these that add nothing new.
_GATHER_ACTIONS = frozenset(
    {TurnAction.RETRIEVE, TurnAction.DECOMPOSE, TurnAction.DEEP_STUDY}
)


async def run_turn_loop(
    query: str,
    context: TurnContext,
    deps: TurnLoopDeps,
    budget: TurnBudget,
    route_hint: Optional[RouteHint] = None,
) -> TurnLoopResult:
    """Run one full conversation turn through the agentic loop.

    Per iteration: check budgets (wall clock / actions / LLM ledger), ask the
    controller for a :class:`TurnDecision`, emit ``turn_action``, dispatch.
    RETRIEVE and DEEP_STUDY grow the evidence pool and loop; CLARIFY ends the
    turn as ``ask_user``; ANSWER drafts and gates — pass ends the turn as
    ``answered``, fail records :class:`GateFeedback` for the next decision.
    Budget exhaustion (including a controller insisting on ANSWER past
    ``max_answer_attempts``) exits through the best-effort path. Never raises:
    an unexpected error terminates best-effort with ``stop_reason='error'``.

    Args:
        query: The sanitized user query for this turn (input safety runs in
            the server layer before the loop, design §3).
        context: The typed cross-turn context (see ``context.py``).
        deps: Injected capabilities (retrieval activity, document fetch, LLM
            provider, event sink).
        budget: The frozen per-turn budget (``TurnBudget.from_settings()`` at
            the endpoint; hand-built in tests).
        route_hint: The pre-flight router's advisory verdict (``None`` = no
            router / neutral, the pre-router baseline). A non-fast-lane hint
            biases only the first controller prompt; a fast-lane hint drives
            the deterministic RETRIEVE->ANSWER opening (see
            :func:`_seed_decision`). Always fail-open — a wrong hint degrades
            to the controller's own judgment.

    Returns:
        The :class:`TurnLoopResult` for the route layer — answer or
        clarification, final pool, full event trace, and stop accounting.
    """
    state = TurnState()
    emitter = TurnEventEmitter(deps=deps, state=state, budget=budget)
    best_draft: Optional[tuple[float, str]] = None
    # Consecutive gather rounds (RETRIEVE/DECOMPOSE/DEEP_STUDY) that added no
    # new chunks — drives the no-progress ANSWER guard (_no_progress_decision).
    no_progress_rounds = 0

    try:
        while True:
            stop_reason = _budget_stop_reason(state, budget)
            if stop_reason is not None:
                return await _best_effort_result(
                    stop_reason=stop_reason,
                    best_draft=best_draft,
                    query=query,
                    context=context,
                    state=state,
                    budget=budget,
                    deps=deps,
                    emitter=emitter,
                )

            # Decision precedence: (1) the router fast-lane's deterministic
            # opening skips the controller LLM; (2) the no-progress guard forces
            # an ANSWER when gathering has stalled; (3) otherwise the controller
            # decides. Both deterministic paths are fail-open (return None to
            # hand back to the controller).
            decision: Optional[TurnDecision] = _seed_decision(
                route_hint, state, query
            )
            source = "router" if decision is not None else "controller"
            if decision is None:
                decision = _no_progress_decision(state, budget, no_progress_rounds)
                if decision is not None:
                    source = "loop_guard"
            if decision is None:
                decision = await decide(
                    query=query,
                    context=context,
                    state=state,
                    budget=budget,
                    emitter=emitter,
                    route_hint=route_hint,
                )
            await emitter.emit(
                TurnEventType.TURN_ACTION,
                {
                    # 1-based, human-facing (design §8 examples: "#2 RETRIEVE");
                    # consoles/CLI render this value verbatim.
                    "index": state.iteration + 1,
                    "action": decision.action,
                    "reason": decision.reason,
                    "confidence": decision.confidence,
                    # Who chose this action: the fast-lane router or the
                    # no-progress loop guard (both deterministic, no LLM) or the
                    # controller.
                    "source": source,
                },
            )
            state.iteration += 1

            if decision.action == TurnAction.CLARIFY:
                clarification = await run_clarify(
                    query=query,
                    context=context,
                    state=state,
                    budget=budget,
                    emitter=emitter,
                )
                return _result(
                    state,
                    action=TurnLoopResult.ACTION_ASK_USER,
                    clarification=clarification,
                    stop_reason=STOP_CLARIFY,
                )

            if decision.action == TurnAction.ANSWER:
                if state.answer_attempts >= budget.max_answer_attempts:
                    return await _best_effort_result(
                        stop_reason=STOP_MAX_ANSWER_ATTEMPTS,
                        best_draft=best_draft,
                        query=query,
                        context=context,
                        state=state,
                        budget=budget,
                        deps=deps,
                        emitter=emitter,
                    )
                draft, feedback = await run_answer(
                    query=query,
                    context=context,
                    state=state,
                    budget=budget,
                    deps=deps,
                    emitter=emitter,
                )
                if draft and (best_draft is None or feedback.score > best_draft[0]):
                    best_draft = (feedback.score, draft)
                if feedback.passed:
                    return _result(
                        state,
                        answer=draft,
                        confidence=feedback.score,
                        stop_reason=STOP_GATE_PASSED,
                    )
                state.last_gate = feedback
                continue

            # Evidence-gathering actions: contained per-action so one broken
            # injected seam degrades the action, not the turn (budgets still
            # bound the loop).
            pool_before = len(state.pool)
            try:
                if decision.action == TurnAction.RETRIEVE:
                    args = (
                        decision.args
                        if isinstance(decision.args, RetrieveArgs)
                        else RetrieveArgs(query_text=query)
                    )
                    await run_retrieve(
                        args,
                        query=query,
                        state=state,
                        budget=budget,
                        deps=deps,
                        emitter=emitter,
                    )
                elif decision.action == TurnAction.DECOMPOSE:
                    args = (
                        decision.args
                        if isinstance(decision.args, DecomposeArgs)
                        else DecomposeArgs(question=query)
                    )
                    await run_decompose(
                        args,
                        query=query,
                        state=state,
                        budget=budget,
                        deps=deps,
                        emitter=emitter,
                    )
                elif decision.action == TurnAction.DEEP_STUDY:
                    if isinstance(decision.args, DeepStudyArgs):
                        await run_deep_study(
                            decision.args,
                            context=context,
                            state=state,
                            budget=budget,
                            deps=deps,
                            emitter=emitter,
                        )
                    else:
                        logger.warning(
                            "turn loop DEEP_STUDY decision without args — skipping"
                        )
            except Exception as exc:  # noqa: BLE001 — contain per-action failures
                logger.warning(
                    "turn loop %s action failed: %s", decision.action, exc
                )
            # No-progress tracking: a gather round that grew the pool resets the
            # streak; one that added nothing (all dedup, empty, or a failed
            # seam) advances it toward the _no_progress_decision ANSWER guard.
            if len(state.pool) > pool_before:
                no_progress_rounds = 0
            else:
                no_progress_rounds += 1
    except Exception:  # noqa: BLE001 — the loop itself must never raise
        logger.exception("turn loop failed unexpectedly — exiting best-effort")
        if best_draft is not None:
            confidence, answer = best_draft
            return _result(
                state,
                answer=answer,
                confidence=confidence,
                stop_reason=STOP_ERROR,
            )
        return _result(
            state,
            answer=_cannot_answer_text(state),
            stop_reason=STOP_ERROR,
        )


__all__ = ["run_turn_loop"]
