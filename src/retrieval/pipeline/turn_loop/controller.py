# @summary
# Per-iteration action selection for the turn loop: renders the
# turn_controller_decide.md prompt from the TurnContext digest, a deterministic
# evidence-pool digest, the budgets remaining, and the last gate feedback; runs
# the controller LLM through the charged-call wrapper; salvage-parses the JSON
# into a TurnDecision. Fail-open per design §6: an unusable decision becomes
# RETRIEVE-with-the-verbatim-user-query on iteration 1, else an ANSWER attempt.
# Also the canonical home of the model-alias getters and the evidence digest
# (shared with answer.py's self-score).
# Exports: decide, build_evidence_digest, controller_model_alias,
#          judge_model_alias
# Deps: src.common.prompts, src.common.utils, src.retrieval.pipeline.turn_loop
#       .{schemas,events} (config.settings lazily)
# @end-summary
"""Controller decision stage: choose the turn's next action.

One stage per file (CLAUDE.md §2). The controller is the only component that
decides what happens next; its inputs are pure renderings of typed state — the
digest builders here are deterministic string assembly with no content
pattern-matching (CLAUDE.md §0), so identical state always produces an
identical prompt (evalable/cachable).
"""

from __future__ import annotations

import logging

from typing import Optional

from src.common.prompts import load_prompt, render, strip_reasoning
from src.common.utils import parse_json_object
from src.retrieval.pipeline.turn_loop.events import (
    TurnEventEmitter,
    latest_event_payload,
)
from src.retrieval.pipeline.turn_loop.router import RouteHint
from src.retrieval.pipeline.turn_loop.schemas import (
    AnswerArgs,
    RetrieveArgs,
    TurnAction,
    TurnBudget,
    TurnContext,
    TurnDecision,
    TurnEventType,
    TurnState,
)

logger = logging.getLogger(__name__)

_PROMPT_FILE = "turn_controller_decide.md"

# Structural cap on the rendered conversation-context digest fed to the
# controller prompt (older material is already compacted into the rolling
# summary, so the head carries the signal). A layout constant of the prompt,
# not a behavior tunable — behavior tunables live in RAG_TURN_LOOP_* settings.
CONTEXT_DIGEST_MAX_CHARS = 8000


def controller_model_alias() -> str:
    """Router alias for controller-role calls (decision, deep-study read,
    clarification authoring). Reads ``RAG_TURN_LOOP_CONTROLLER_MODEL_ALIAS``
    lazily so importing this module has zero config side effects."""
    from config import settings

    return str(
        getattr(settings, "RAG_TURN_LOOP_CONTROLLER_MODEL_ALIAS", "controller")
    )


def judge_model_alias() -> str:
    """Router alias for judge-role calls (round judging, answer self-score).
    Reads ``RAG_TURN_LOOP_JUDGE_MODEL_ALIAS`` lazily."""
    from config import settings

    return str(getattr(settings, "RAG_TURN_LOOP_JUDGE_MODEL_ALIAS", "judge"))


def _preview_chars() -> int:
    """Per-chunk preview cap for digests (``RAG_TURN_CONTEXT_PREVIEW_CHARS``)."""
    from config import settings

    return int(getattr(settings, "RAG_TURN_CONTEXT_PREVIEW_CHARS", 320))


def _one_line(text: str, max_chars: int) -> str:
    """Flatten ``text`` to one whitespace-normalized line capped at ``max_chars``."""
    flat = " ".join((text or "").split())
    if max_chars > 0 and len(flat) > max_chars:
        flat = flat[:max_chars]
    return flat


def build_evidence_digest(state: TurnState) -> str:
    """Render the deterministic evidence digest of the turn so far.

    Pure function of :class:`TurnState`: per-chunk one-liners over the pool
    (``[n]`` indexes match the citation indexes drafting uses, and each line
    carries ``document_id``/``source_key`` verbatim so the controller can copy
    them into a DEEP_STUDY decision without inventing names), the tried
    queries/HyDE variants (anti-repeat), and the latest judge verdict from the
    event trace (its ``missing_information`` is how the judge feeds the
    controller, design §5).

    Args:
        state: The turn's mutable state (pool, tried lists, event trace).

    Returns:
        The prompt-ready digest string; a placeholder line when no evidence
        has been gathered yet.
    """
    preview_cap = _preview_chars()
    lines: list[str] = []
    if state.pool:
        lines.append(f"Evidence pool ({len(state.pool)} chunks):")
        for index, chunk in enumerate(state.pool, start=1):
            lines.append(
                f"[{index}] document_id={chunk.document_id} "
                f"source_key={chunk.source_key} source={chunk.source} "
                f"heading={chunk.heading} score={chunk.score:.2f} "
                f"({chunk.provenance}) :: {_one_line(chunk.text, preview_cap)}"
            )
    else:
        lines.append("(no evidence gathered yet this turn)")
    if state.tried_queries:
        lines.append("")
        lines.append("Queries already tried (do not repeat verbatim):")
        for query in state.tried_queries:
            lines.append(f"- {_one_line(query, preview_cap)}")
    if state.tried_hyde:
        lines.append("")
        lines.append("HyDE hypothetical answers already tried:")
        for hyde in state.tried_hyde:
            lines.append(f"- {_one_line(hyde, preview_cap)}")
    if state.studied:
        lines.append("")
        lines.append("Documents deep-studied this turn:")
        for doc in state.studied.values():
            lines.append(
                f"- document_id={doc.document_id} "
                f"windows_read={len(doc.windows_read)} :: "
                f"{_one_line(doc.conclusion or doc.notes, preview_cap)}"
            )
    verdict = latest_event_payload(state, TurnEventType.JUDGE_VERDICT)
    if verdict is not None:
        lines.append("")
        lines.append(
            "Latest judge verdict: "
            f"sufficient={verdict.get('sufficient')} "
            f"confidence={verdict.get('confidence')} "
            f"missing_information={_one_line(str(verdict.get('missing_information') or ''), preview_cap)}"
        )
    return "\n".join(lines)


def _render_budgets(state: TurnState, budget: TurnBudget) -> str:
    """Render the budgets-remaining line for the controller prompt."""
    return (
        f"actions_left={state.remaining_actions(budget)} "
        f"llm_calls_left={max(0, budget.max_llm_calls - state.llm_calls)} "
        f"wall_clock_ms_left={max(0, budget.wall_clock_ms - state.elapsed_ms())} "
        f"answer_attempts_left={max(0, budget.max_answer_attempts - state.answer_attempts)} "
        f"deep_study_docs_left={max(0, budget.deep_study_max_docs - len(state.studied))}"
    )


def _render_gate_feedback(state: TurnState) -> str:
    """Render the last failed-gate feedback block, or an empty marker."""
    gate = state.last_gate
    if gate is None:
        return "(none)"
    lines = [
        "Last answer attempt failed the confidence gate: "
        f"score={gate.score:.2f} vs threshold={gate.threshold:.2f}; "
        f"weakest component: {gate.weakest_component}."
    ]
    if gate.unsupported_claims:
        lines.append("Unsupported claims in the failed draft:")
        for claim in gate.unsupported_claims:
            lines.append(f"- {claim}")
    if gate.missing_information:
        lines.append("Missing information named by the judge:")
        for gap in gate.missing_information:
            lines.append(f"- {gap}")
    return "\n".join(lines)


def _render_router_hint(route_hint: Optional[RouteHint], state: TurnState) -> str:
    """Render the pre-flight router's advisory hint for the controller prompt.

    Only the FIRST controller decision (``iteration == 0``) is seeded — the
    hint is a suggestion for how to *open* the turn; once the loop is under way
    the controller reasons from the actual evidence/gate state, so later
    iterations render ``(none)``. The fast-lane hint is handled by the
    orchestrator (it skips this call entirely), so a hint reaching here is
    always advisory: the controller may follow or ignore it (fail-open).
    """
    if route_hint is None or state.iteration != 0 or not route_hint.initial_action:
        return "(none)"
    return (
        f"A cheap pre-flight classifier suggests opening with "
        f"{route_hint.initial_action} because: {route_hint.reason}. Treat this "
        "as advice — choose the action the state above actually warrants."
    )


def _fail_open_decision(query: str, state: TurnState) -> TurnDecision:
    """The design-§6 fail-open default when no usable decision was parsed.

    Iteration 1 (nothing dispatched yet): RETRIEVE with the verbatim user
    query — evidence-gathering is always a safe first move. Later iterations:
    an ANSWER attempt — the gate (not the broken controller) then decides
    whether the pool suffices.
    """
    if state.iteration == 0:
        return TurnDecision(
            action=TurnAction.RETRIEVE,
            reason=(
                "fail-open: controller decision unavailable — retrieving with "
                "the verbatim user query"
            ),
            confidence=0.0,
            args=RetrieveArgs(query_text=query),
        )
    return TurnDecision(
        action=TurnAction.ANSWER,
        reason=(
            "fail-open: controller decision unavailable — attempting an "
            "answer from the evidence gathered so far"
        ),
        confidence=0.0,
        args=AnswerArgs(),
    )


async def decide(
    *,
    query: str,
    context: TurnContext,
    state: TurnState,
    budget: TurnBudget,
    emitter: TurnEventEmitter,
    route_hint: Optional[RouteHint] = None,
) -> TurnDecision:
    """Run one controller iteration: choose the turn's next action.

    Renders ``turn_controller_decide.md`` from the typed state, calls the
    controller LLM through the charged wrapper (ledger + ``llm_call`` event),
    and coerces the salvage-parsed JSON via
    :meth:`TurnDecision.from_llm_json`. Any failure — exhausted ledger,
    provider error, unparseable JSON, unknown action — falls open to
    :func:`_fail_open_decision`; this stage never raises.

    Args:
        query: The user's turn query (verbatim; used by the fail-open path).
        context: The cross-turn context digest source.
        state: The turn's mutable state.
        budget: The frozen per-turn budget.
        emitter: The turn's event emitter / charged-call wrapper.

    Returns:
        A valid :class:`TurnDecision` in all cases. A RETRIEVE decision with
        an empty ``query_text`` is normalized to the verbatim user query.
    """
    prompt = render(
        load_prompt(_PROMPT_FILE),
        user_query=query,
        turn_context=context.render_for_prompt(CONTEXT_DIGEST_MAX_CHARS)
        or "(no prior conversation context)",
        evidence_digest=build_evidence_digest(state),
        budgets=_render_budgets(state, budget),
        gate_feedback=_render_gate_feedback(state),
        router_hint=_render_router_hint(route_hint, state),
    )
    response = await emitter.charged_call(
        alias=controller_model_alias(),
        purpose="controller",
        prompt=prompt,
        temperature=0.0,
    )
    decision = None
    if response is not None:
        payload = parse_json_object(
            strip_reasoning(getattr(response, "content", "") or "")
        )
        decision = TurnDecision.from_llm_json(payload)
        if decision is None:
            logger.warning(
                "turn loop controller returned unusable JSON — failing open"
            )
    if decision is None:
        return _fail_open_decision(query, state)
    # Normalize a RETRIEVE with an empty query to the verbatim user query so a
    # sloppy controller output still produces a valid retrieval round.
    if decision.action == TurnAction.RETRIEVE and isinstance(
        decision.args, RetrieveArgs
    ):
        if not decision.args.query_text:
            decision.args.query_text = query
    return decision


__all__ = [
    "decide",
    "build_evidence_digest",
    "controller_model_alias",
    "judge_model_alias",
]
