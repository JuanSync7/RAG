# @summary
# Tests for the turn-loop CLARIFY stage: hint capping at
# TurnBudget.clarify_max_hints, scoping-question capping, the clarify event
# payload, gap collection from the judge verdict trace + gate feedback into
# the prompt, and the deterministic fallback clarification when the authoring
# call fails or returns no question.
# @end-summary
"""Tests for ``turn_loop/clarify.py``."""

from __future__ import annotations

from src.retrieval.pipeline.turn_loop.clarify import run_clarify
from src.retrieval.pipeline.turn_loop.events import TurnEventEmitter
from src.retrieval.pipeline.turn_loop.schemas import (
    GateFeedback,
    TurnEvent,
    TurnEventType,
    TurnState,
)

from tests.retrieval.turn_loop.conftest import (
    FakeProvider,
    clarify_json,
    events_of,
    make_budget,
    make_deps,
)


def _setup(provider, budget=None):
    deps, emitted = make_deps(provider)
    state = TurnState()
    budget = budget or make_budget()
    emitter = TurnEventEmitter(deps=deps, state=state, budget=budget, stream_events=True)
    return emitted, state, budget, emitter


async def test_hints_capped_at_clarify_max_hints(empty_context):
    provider = FakeProvider(
        responses=[
            clarify_json(
                "Which environment?",
                ["Block-level", "Top-level", "Gate-level", "Cluster", "System"],
                ["How do I set up a block-level env?"] * 5,
            )
        ]
    )
    budget = make_budget(clarify_max_hints=2)
    emitted, state, budget, emitter = _setup(provider, budget)

    clarification = await run_clarify(
        query="q", context=empty_context, state=state, budget=budget, emitter=emitter
    )

    assert clarification.question == "Which environment?"
    assert clarification.hints == ["Block-level", "Top-level"]  # capped at 2
    assert len(clarification.scoping_questions) == 3  # prompt's own 0-3 bound


async def test_clarify_event_matches_the_returned_payload(empty_context):
    provider = FakeProvider(responses=[clarify_json("Q?", ["a"], ["s?"])])
    emitted, state, budget, emitter = _setup(provider)

    clarification = await run_clarify(
        query="q", context=empty_context, state=state, budget=budget, emitter=emitter
    )

    event = events_of(emitted, TurnEventType.CLARIFY)[0]
    assert event == {
        "question": clarification.question,
        "hints": clarification.hints,
        "scoping_questions": clarification.scoping_questions,
    }


async def test_gaps_from_judge_trace_and_gate_feedback_reach_the_prompt(
    empty_context,
):
    provider = FakeProvider(responses=[clarify_json("Q?", [])])
    emitted, state, budget, emitter = _setup(provider)
    state.events.append(
        TurnEvent.now(
            TurnEventType.JUDGE_VERDICT,
            {"missing_information": "which release train applies"},
        )
    )
    state.last_gate = GateFeedback(
        score=0.3,
        threshold=0.6,
        passed=False,
        weakest_component="judge",
        missing_information=["the deployment target"],
    )
    state.tried_queries.append("tried query one")

    await run_clarify(
        query="q", context=empty_context, state=state, budget=budget, emitter=emitter
    )

    _, messages, _ = provider.calls[0]
    prompt = messages[0]["content"]
    assert "which release train applies" in prompt
    assert "the deployment target" in prompt
    assert "tried query one" in prompt
    assert "{{" not in prompt


async def test_provider_failure_falls_back_to_deterministic_clarification(
    empty_context,
):
    provider = FakeProvider(responses=[RuntimeError("down")])
    emitted, state, budget, emitter = _setup(provider)

    clarification = await run_clarify(
        query="the ambiguous question",
        context=empty_context,
        state=state,
        budget=budget,
        emitter=emitter,
    )

    assert "the ambiguous question" in clarification.question
    assert clarification.hints == []
    assert events_of(emitted, TurnEventType.CLARIFY)  # event still emitted


async def test_missing_question_in_json_falls_back(empty_context):
    provider = FakeProvider(responses=['{"hints": ["a", "b"]}'])
    emitted, state, budget, emitter = _setup(provider)

    clarification = await run_clarify(
        query="q", context=empty_context, state=state, budget=budget, emitter=emitter
    )

    assert clarification.question  # never empty


async def test_hint_text_flattened_to_one_bounded_line(empty_context):
    """LLM-authored chip text is normalized at the source: multi-line hints
    collapse to one line and oversized ones are truncated at the structural
    chip cap — chips are one-click reply buttons resubmitted verbatim, so
    every renderer downstream may assume bounded single-line text."""
    from src.retrieval.pipeline.turn_loop.clarify import _MAX_ITEM_CHARS

    provider = FakeProvider(
        responses=[
            clarify_json(
                "Which flow?",
                ["line one\nline two\n\tline three", "y" * (_MAX_ITEM_CHARS + 50)],
                [],
            )
        ]
    )
    emitted, state, budget, emitter = _setup(provider)

    clarification = await run_clarify(
        query="q", context=empty_context, state=state, budget=budget, emitter=emitter
    )

    assert clarification.hints[0] == "line one line two line three"
    assert "\n" not in clarification.hints[0]
    assert clarification.hints[1] == "y" * _MAX_ITEM_CHARS
