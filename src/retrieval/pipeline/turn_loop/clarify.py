# @summary
# CLARIFY action for the turn loop (terminal): renders
# turn_clarify_generate.md from the named information gaps (latest judge
# verdict + last gate feedback), the tried queries, and any surfaced
# conflicts; runs the controller-alias LLM through the charged wrapper;
# salvage-parses into a ClarificationOut with hints capped at
# TurnBudget.clarify_max_hints; emits the clarify event. Fail-open: a dead
# provider degrades to a deterministic clarification built from the gaps —
# a CLARIFY decision always ends the turn with a question, never an error.
# Exports: run_clarify
# Deps: src.common.prompts, src.common.utils,
#       src.retrieval.pipeline.turn_loop.{schemas,events,controller}
# @end-summary
"""CLARIFY stage: LLM-authored clarification question + clickable hints.

Terminal action (design §5): the loop ends by handing the user a question.
The clarification content comes from a dedicated LLM call — the controller
only decides THAT clarification is needed; this stage decides WHAT to ask,
grounded in the gaps the evidence search actually hit.
"""

from __future__ import annotations

import logging

from src.common.prompts import load_prompt, render, strip_reasoning
from src.common.utils import parse_json_object
from src.retrieval.pipeline.turn_loop.controller import (
    CONTEXT_DIGEST_MAX_CHARS,
    controller_model_alias,
)
from src.retrieval.pipeline.turn_loop.events import (
    TurnEventEmitter,
    latest_event_payload,
)
from src.retrieval.pipeline.turn_loop.schemas import (
    ClarificationOut,
    TurnBudget,
    TurnContext,
    TurnEventType,
    TurnState,
)

logger = logging.getLogger(__name__)

_PROMPT_FILE = "turn_clarify_generate.md"

# Structural cap on scoping questions (the prompt asks for 0-3).
_MAX_SCOPING_QUESTIONS = 3

# Structural cap on one hint/scoping chip's text. Chips are one-click user
# replies rendered as buttons and resubmitted verbatim as the next query —
# a bound of the UI/reply surface (like _MAX_SCOPING_QUESTIONS), not a
# behavior tunable. Also defense in depth: LLM-authored chip text is the
# first attacker-steerable string that flows into the consoles' resubmit
# machinery, so it is normalized to one bounded line here at the source.
_MAX_ITEM_CHARS = 300


def _collect_gaps(state: TurnState) -> list[str]:
    """Collect the named information gaps driving this clarification.

    Sources (deduped, order-preserving): the latest judge verdict's
    ``missing_information`` from the event trace and the last gate feedback's
    ``missing_information`` items. Pure typed-state reads — no content
    inspection.
    """
    gaps: list[str] = []
    verdict = latest_event_payload(state, TurnEventType.JUDGE_VERDICT)
    if verdict is not None:
        missing = str(verdict.get("missing_information") or "").strip()
        if missing:
            gaps.append(missing)
    if state.last_gate is not None:
        for gap in state.last_gate.missing_information:
            text = str(gap).strip()
            if text and text not in gaps:
                gaps.append(text)
    return gaps


def _coerce_str_list(value: object, cap: int) -> list[str]:
    """Coerce an LLM JSON field to a list of at most ``cap`` chip strings.

    Each entry is whitespace-flattened to a single line and truncated at
    ``_MAX_ITEM_CHARS`` — chips are bounded one-line reply buttons, and
    normalizing here keeps every downstream renderer (web chips, CLI list,
    memory persistence) inside that contract regardless of what the LLM
    emitted. Pure str ops, no content matching (CLAUDE.md §0).
    """
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for entry in value:
        text = " ".join(str(entry).split())
        if len(text) > _MAX_ITEM_CHARS:
            text = text[:_MAX_ITEM_CHARS]
        if text:
            items.append(text)
        if len(items) >= cap:
            break
    return items


def _fallback_clarification(query: str, gaps: list[str]) -> ClarificationOut:
    """Deterministic clarification when the authoring call fails.

    Template assembly over the typed gaps (fail-open precedent, design §6) —
    the user still gets an actionable question naming what blocked the turn.
    """
    if gaps:
        detail = "; ".join(gaps)
        question = (
            "I could not settle this from the available evidence — "
            f"could you narrow it down? Specifically: {detail}"
        )
    else:
        question = (
            "I could not settle this from the available evidence — could you "
            f'rephrase or narrow down what you mean by: "{query}"?'
        )
    return ClarificationOut(question=question)


async def run_clarify(
    *,
    query: str,
    context: TurnContext,
    state: TurnState,
    budget: TurnBudget,
    emitter: TurnEventEmitter,
) -> ClarificationOut:
    """Author the terminal clarification for this turn.

    Renders ``turn_clarify_generate.md`` from the user query, the turn
    context, the collected gaps, and the tried queries; caps ``hints`` at
    ``budget.clarify_max_hints`` and ``scoping_questions`` at three (the
    prompt's own bound); emits the ``clarify`` event. Never raises — every
    failure path returns the deterministic fallback clarification.

    Args:
        query: The user's turn query.
        context: The cross-turn context digest source.
        state: The turn's mutable state (gap sources, tried queries).
        budget: The frozen per-turn budget (hint cap).
        emitter: The turn's event emitter / charged-call wrapper.

    Returns:
        The clarification payload for the ``ask_user`` result.
    """
    gaps = _collect_gaps(state)
    tried = "\n".join(f"- {q}" for q in state.tried_queries) or "(none)"
    prompt = render(
        load_prompt(_PROMPT_FILE),
        user_query=query,
        turn_context=context.render_for_prompt(CONTEXT_DIGEST_MAX_CHARS)
        or "(no prior conversation context)",
        missing_information="\n".join(f"- {gap}" for gap in gaps) or "(none named)",
        tried_queries=tried,
        conflicts="(none detected)",
        max_hints=str(budget.clarify_max_hints),
    )
    response = await emitter.charged_call(
        alias=controller_model_alias(),
        purpose="clarify",
        prompt=prompt,
        temperature=0.0,
    )
    clarification: ClarificationOut
    if response is None:
        clarification = _fallback_clarification(query, gaps)
    else:
        payload = parse_json_object(
            strip_reasoning(getattr(response, "content", "") or "")
        )
        question = str(payload.get("question") or "").strip()
        if not question:
            logger.warning(
                "turn loop clarify call returned no question — using fallback"
            )
            clarification = _fallback_clarification(query, gaps)
        else:
            clarification = ClarificationOut(
                question=question,
                hints=_coerce_str_list(
                    payload.get("hints"), budget.clarify_max_hints
                ),
                scoping_questions=_coerce_str_list(
                    payload.get("scoping_questions"), _MAX_SCOPING_QUESTIONS
                ),
            )
    await emitter.emit(
        TurnEventType.CLARIFY,
        {
            "question": clarification.question,
            "hints": list(clarification.hints),
            "scoping_questions": list(clarification.scoping_questions),
        },
    )
    return clarification


__all__ = ["run_clarify"]
