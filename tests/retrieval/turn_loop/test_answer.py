# @summary
# Tests for the turn-loop ANSWER stage: deterministic citation-coverage
# scanning (bracketed-integer tokens, nesting, range validation), live draft
# events carrying reasoning|content kinds with only content accumulating,
# draft-precedes-gate event ordering, the weighted gate (judge component from
# the trace with neutral default, self-score, coverage) against the threshold,
# weakest-component + unsupported-claims + judge-gap feedback on failure, the
# single-system-message prompt shape, and empty-draft failure.
# @end-summary
"""Tests for ``turn_loop/answer.py``."""

from __future__ import annotations

import pytest

from src.retrieval.pipeline.turn_loop.answer import citation_coverage, run_answer
from src.retrieval.pipeline.turn_loop.events import TurnEventEmitter
from src.retrieval.pipeline.turn_loop.schemas import (
    TurnContext,
    TurnEvent,
    TurnEventType,
    TurnState,
)

from tests.retrieval.turn_loop.conftest import (
    FakeProvider,
    event_types,
    events_of,
    make_budget,
    make_chunk,
    make_deps,
    selfscore_json,
)


def _setup(provider, *, pool=None, budget=None):
    deps, emitted = make_deps(provider)
    state = TurnState()
    for chunk in pool or []:
        state.pool.append(chunk)
        state.seen_chunk_ids.add(chunk.chunk_id)
    budget = budget or make_budget()
    emitter = TurnEventEmitter(deps=deps, state=state, budget=budget, stream_events=True)
    return deps, emitted, state, budget, emitter


class TestCitationCoverage:
    def test_counts_distinct_valid_citations(self):
        assert citation_coverage("see [1] and [2] and [2] again", 4) == 0.5

    def test_ignores_out_of_range_and_non_integer_tokens(self):
        assert citation_coverage("[9] [abc] [1.5] [ ] [-2]", 3) == 0.0

    def test_finds_citation_nested_inside_brackets(self):
        assert citation_coverage("[see [2]]", 2) == 0.5

    def test_empty_pool_or_draft_is_zero(self):
        assert citation_coverage("cited [1]", 0) == 0.0
        assert citation_coverage("", 3) == 0.0

    def test_full_coverage(self):
        assert citation_coverage("[1][2][3]", 3) == 1.0


async def test_gate_pass_with_full_components():
    """judge 0.9 (trace) / self 0.95 / coverage 1.0 under 0.5/0.3/0.2 weights
    -> 0.935 >= 0.6 threshold."""
    provider = FakeProvider(
        responses=[selfscore_json(0.95)],
        streams=[[("reasoning", "thinking..."), ("content", "Answer [1] and [2].")]],
    )
    pool = [make_chunk("c1"), make_chunk("c2")]
    deps, emitted, state, budget, emitter = _setup(provider, pool=pool)
    state.events.append(
        TurnEvent.now(
            TurnEventType.JUDGE_VERDICT,
            {"confidence": 0.9, "missing_information": ""},
        )
    )

    draft, feedback = await run_answer(
        query="q",
        context=TurnContext(conversation_id="c"),
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    assert draft == "Answer [1] and [2]."  # reasoning deltas excluded
    assert feedback.passed
    assert feedback.score == pytest.approx(0.935)
    assert state.answer_attempts == 1


async def test_draft_events_stream_live_and_precede_gate(empty_context):
    provider = FakeProvider(
        responses=[selfscore_json(0.9)],
        streams=[[("reasoning", "hmm"), ("content", "final [1]")]],
    )
    deps, emitted, state, budget, emitter = _setup(provider, pool=[make_chunk("c1")])

    await run_answer(
        query="q",
        context=empty_context,
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    types = event_types(emitted)
    draft_positions = [i for i, t in enumerate(types) if t == TurnEventType.DRAFT]
    gate_position = types.index(TurnEventType.GATE)
    assert draft_positions and all(p < gate_position for p in draft_positions)
    drafts = events_of(emitted, TurnEventType.DRAFT)
    assert drafts[0] == {"attempt": 1, "kind": "reasoning", "text_delta": "hmm"}
    assert drafts[1] == {"attempt": 1, "kind": "content", "text_delta": "final [1]"}


async def test_gate_failure_names_weakest_component_and_claims(empty_context):
    """No judge verdict (neutral 0.5), self 0.2, coverage 0 -> 0.31 < 0.6;
    citation is the weakest raw component."""
    provider = FakeProvider(
        responses=[selfscore_json(0.2, ["the timeout defaults to 30 seconds"])],
        streams=[[("content", "uncited claim")]],
    )
    deps, emitted, state, budget, emitter = _setup(provider, pool=[make_chunk("c1")])

    draft, feedback = await run_answer(
        query="q",
        context=empty_context,
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    assert not feedback.passed
    assert feedback.score == pytest.approx(0.5 * 0.5 + 0.3 * 0.2)
    assert feedback.weakest_component == "citation"
    assert feedback.unsupported_claims == ["the timeout defaults to 30 seconds"]
    gate = events_of(emitted, TurnEventType.GATE)[0]
    assert gate["passed"] is False
    assert gate["weakest"] == "citation"


async def test_judge_missing_information_flows_into_feedback(empty_context):
    provider = FakeProvider(
        responses=[selfscore_json(0.9)], streams=[[("content", "a [1]")]]
    )
    deps, emitted, state, budget, emitter = _setup(provider, pool=[make_chunk("c1")])
    state.events.append(
        TurnEvent.now(
            TurnEventType.JUDGE_VERDICT,
            {"confidence": 0.2, "missing_information": "the limits table"},
        )
    )

    _, feedback = await run_answer(
        query="q",
        context=empty_context,
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    assert feedback.missing_information == ["the limits table"]


async def test_empty_stream_fails_the_gate_without_selfscore_call(empty_context):
    provider = FakeProvider(streams=[[]])  # dead stream, no content at all
    deps, emitted, state, budget, emitter = _setup(provider, pool=[make_chunk("c1")])

    draft, feedback = await run_answer(
        query="q",
        context=empty_context,
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    assert draft == ""
    assert not feedback.passed
    # Only the draft call happened (charged); no self-score on an empty draft.
    assert state.llm_calls == 1
    assert [c[0] for c in provider.calls] == ["generate_stream"]


async def test_mid_stream_error_keeps_partial_draft(empty_context):
    provider = FakeProvider(
        responses=[selfscore_json(0.9)],
        streams=[[("content", "partial [1] "), RuntimeError("connection reset")]],
    )
    deps, emitted, state, budget, emitter = _setup(provider, pool=[make_chunk("c1")])

    draft, _ = await run_answer(
        query="q",
        context=empty_context,
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    assert draft == "partial [1]"


async def test_messages_use_single_leading_system_message(empty_context):
    provider = FakeProvider(
        responses=[selfscore_json(0.9)], streams=[[("content", "a [1]")]]
    )
    deps, emitted, state, budget, emitter = _setup(provider, pool=[make_chunk("c1")])

    await run_answer(
        query="the question",
        context=empty_context,
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    stream_calls = [c for c in provider.calls if c[0] == "generate_stream"]
    messages = stream_calls[0][1]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "[1]" in messages[1]["content"]  # citation-indexed evidence
    assert "the question" in messages[1]["content"]
