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


class TestGroundingContract:
    """B1/B2 (generation grounding contract): the shared system prompt and the
    turn_loop self-score rubric must carry the GENERIC faithfulness principles
    (not query-specific patches) that the 20Q generation failures surfaced:
    - prefer documented specifics over generic background; flag [background];
    - surface documented gaps/TBDs instead of smoothing them;
    - state when a documented value contradicts the question's premise;
    - include documented breakdowns/splits (completeness).
    Guards against silent prompt drift (behaviour itself is eval-validated)."""

    def test_system_prompt_carries_grounding_calibration(self):
        from src.retrieval.pipeline.turn_loop.answer import _build_messages

        msgs = _build_messages("q", TurnContext(conversation_id="c"), TurnState())
        system = msgs[0]["content"].lower()
        # [background] flag + generic-over-specific
        assert "[background]" in system
        assert "background knowledge" in system
        # documented uncertainty / TBD surfacing
        assert "tbd" in system
        # premise contradicted by a documented value
        assert "contradict" in system
        # completeness of documented breakdowns
        assert "breakdown" in system or "distinction" in system
        # specificity matches the retrieved context (generic->generic,
        # documented-specific->specific) — the derivation guardrail, not forcing
        assert "match the answer's specificity" in system

    def test_selfscore_prompt_credits_mandated_behaviours(self):
        from src.common.prompts import load_prompt

        rubric = load_prompt("turn_answer_selfscore.md").lower()
        # signalled inferences whose premises are in the digest are supported
        assert "inference" in rubric
        # gap/conflict flagging and [background] disclosure are not "unsupported"
        assert "[background]" in rubric
        assert "credit" in rubric


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

    def test_saturates_at_target_for_large_filled_pool(self):
        # A refusal citing 1 source out of a 12-chunk filled pool must NOT score
        # 1/12 (which would fail the gate) — the denominator saturates at target.
        assert citation_coverage("only [1]", 12, target=5) == pytest.approx(0.2)

    def test_over_target_citations_clamp_to_one(self):
        assert citation_coverage("[1][2][3][4][5][6]", 12, target=5) == 1.0

    def test_target_zero_restores_raw_fraction(self):
        assert citation_coverage("[1]", 12, target=0) == pytest.approx(1 / 12)


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


async def test_thin_judged_pool_is_filled_from_fallback_for_generation():
    """A judge that kept only 1 chunk starves generation of disambiguating
    context; run_answer tops the pool up toward fallback_pool_size with the best
    raw chunks, kept chunk FIRST (stable citation indices), no duplicates."""
    provider = FakeProvider(
        responses=[selfscore_json(0.9)],
        streams=[[("content", "grounded [1]")]],
    )
    kept = make_chunk("kept1", score=0.9)
    deps, emitted, state, budget, emitter = _setup(provider, pool=[kept])
    # The judge-independent floor retained more raw candidates than the judge kept.
    state.fallback_chunks = [kept] + [
        make_chunk(f"raw{i}", score=0.8 - i * 0.01) for i in range(2, 12)
    ]

    await run_answer(
        query="q",
        context=TurnContext(conversation_id="c"),
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    assert len(state.pool) == budget.fallback_pool_size  # topped up (default 8)
    assert state.pool[0].chunk_id == "kept1"  # kept chunk stays first
    ids = [chunk.chunk_id for chunk in state.pool]
    assert len(ids) == len(set(ids))  # kept1 not re-appended


async def test_fill_prefers_source_diversity_over_more_of_one_doc():
    """Cross-document fix: a thin judged pool (one doc) filled toward N must pull
    in the OTHER retrieved documents (source diversity), not 6 more chunks of the
    same doc — the 'answered from one document' failure. Diversity-first fill: the
    best chunk of each NEW source before topping up by score."""
    provider = FakeProvider(
        responses=[selfscore_json(0.9)],
        streams=[[("content", "answer [1]")]],
    )
    kept = make_chunk("a1", document_id="A", source="DocA", score=0.95)
    deps, emitted, state, budget, emitter = _setup(provider, pool=[kept])
    # Floor: many high-scored DocA chunks + two lower-scored OTHER docs (which a
    # score-only fill would never reach before the pool is full).
    state.fallback_chunks = [kept] + [
        make_chunk(f"a{i}", document_id="A", source="DocA", score=0.9 - i * 0.01)
        for i in range(2, 8)
    ] + [
        make_chunk("b1", document_id="B", source="DocB", score=0.50),
        make_chunk("c1", document_id="C", source="DocC", score=0.40),
    ]

    await run_answer(
        query="q",
        context=TurnContext(conversation_id="c"),
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    sources = {chunk.source for chunk in state.pool}
    assert {"DocA", "DocB", "DocC"} <= sources  # other docs pulled in for coverage
    assert state.pool[0].chunk_id == "a1"  # judge-kept chunk still first


async def test_fill_disabled_when_fallback_size_zero():
    """fallback_pool_size=0 disables the fill — generation sees judge-kept only."""
    provider = FakeProvider(
        responses=[selfscore_json(0.9)],
        streams=[[("content", "just [1]")]],
    )
    kept = make_chunk("kept1")
    budget = make_budget(fallback_pool_size=0)
    deps, emitted, state, budget, emitter = _setup(provider, pool=[kept], budget=budget)
    state.fallback_chunks = [kept, make_chunk("raw2"), make_chunk("raw3")]

    await run_answer(
        query="q",
        context=TurnContext(conversation_id="c"),
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    assert len(state.pool) == 1  # no fill


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


async def test_gate_judge_component_uses_best_round_not_latest():
    """A dud final judge round must not poison a pool judged strong earlier.

    Class: latest-sample-overwrites-best — the draft is grounded in the whole
    pool, so the gate's judge component reflects the max round confidence.
    """
    from src.retrieval.pipeline.turn_loop.schemas import TurnEvent, TurnEventType

    state = TurnState()
    state.pool.append(make_chunk(0))
    state.events.append(TurnEvent.now(
        TurnEventType.JUDGE_VERDICT,
        {"round": 1, "kept": 1, "confidence": 0.9, "missing_information": ""},
    ))
    state.events.append(TurnEvent.now(
        TurnEventType.JUDGE_VERDICT,
        {"round": 2, "kept": 0, "confidence": 0.0,
         "missing_information": "still missing X"},
    ))
    provider = FakeProvider(
        streams=[[("content", "grounded answer [1]")]],
        responses=['{"self_score": 1.0, "unsupported_claims": []}'],
    )
    deps, _ = make_deps(provider)
    budget = make_budget(answer_confidence_threshold=0.6)
    emitter = TurnEventEmitter(deps=deps, state=state, budget=budget, stream_events=False)

    from src.retrieval.pipeline.turn_loop.answer import run_answer
    from src.retrieval.pipeline.turn_loop.schemas import TurnContext

    draft, feedback = await run_answer(
        query="q",
        context=TurnContext(conversation_id="c"),
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    gate = [e for e in state.events if e.type == TurnEventType.GATE][-1]
    # judge component 0.9 (best round), not 0.0 (latest round)
    assert gate.payload["score"] >= 0.6
    assert gate.payload["passed"] is True
