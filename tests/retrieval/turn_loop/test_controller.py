# @summary
# Tests for the turn-loop controller stage: valid decision parsing (action +
# typed args), the design-§6 fail-open ladder (unparseable JSON / provider
# error / exhausted ledger -> RETRIEVE-verbatim on iteration 1, ANSWER later),
# empty-query normalization, full prompt-variable substitution, and the
# deterministic evidence digest (chunk one-liners carry document_id/source_key
# verbatim; judge missing_information surfaces from the trace).
# @end-summary
"""Tests for ``turn_loop/controller.py``."""

from __future__ import annotations

from src.retrieval.pipeline.turn_loop.controller import (
    build_evidence_digest,
    decide,
)
from src.retrieval.pipeline.turn_loop.events import TurnEventEmitter
from src.retrieval.pipeline.turn_loop.schemas import (
    DeepStudyArgs,
    RetrieveArgs,
    TurnAction,
    TurnEvent,
    TurnEventType,
    TurnState,
)

from tests.retrieval.turn_loop.conftest import (
    FakeProvider,
    decision_json,
    make_budget,
    make_chunk,
    make_deps,
)


def _emitter(provider, state, budget):
    deps, events = make_deps(provider)
    return (
        TurnEventEmitter(deps=deps, state=state, budget=budget, stream_events=True),
        events,
    )


async def test_valid_decision_parses_action_and_args(empty_context):
    provider = FakeProvider(
        responses=[
            decision_json(
                "DEEP_STUDY",
                question="what are the timing values?",
                document_id="doc-9",
                source_key="docs/doc-9.md",
            )
        ]
    )
    state, budget = TurnState(), make_budget()
    emitter, _ = _emitter(provider, state, budget)

    decision = await decide(
        query="q", context=empty_context, state=state, budget=budget, emitter=emitter
    )

    assert decision.action == TurnAction.DEEP_STUDY
    assert isinstance(decision.args, DeepStudyArgs)
    assert decision.args.document_id == "doc-9"
    assert decision.args.source_key == "docs/doc-9.md"


async def test_parse_failure_fails_open_to_retrieve_verbatim_on_iteration_one(
    empty_context,
):
    """Unusable controller JSON on the first iteration -> RETRIEVE the user query."""
    provider = FakeProvider(responses=["this is not json at all"])
    state, budget = TurnState(), make_budget()
    emitter, _ = _emitter(provider, state, budget)

    decision = await decide(
        query="how do I set up X?",
        context=empty_context,
        state=state,
        budget=budget,
        emitter=emitter,
    )

    assert decision.action == TurnAction.RETRIEVE
    assert isinstance(decision.args, RetrieveArgs)
    assert decision.args.query_text == "how do I set up X?"


async def test_parse_failure_fails_open_to_answer_after_iteration_one(empty_context):
    """Unusable controller JSON after actions were dispatched -> ANSWER attempt."""
    provider = FakeProvider(responses=['{"action": "LAUNCH_ROCKETS"}'])
    state, budget = TurnState(), make_budget()
    state.iteration = 2
    emitter, _ = _emitter(provider, state, budget)

    decision = await decide(
        query="q", context=empty_context, state=state, budget=budget, emitter=emitter
    )

    assert decision.action == TurnAction.ANSWER


async def test_provider_error_fails_open(empty_context):
    provider = FakeProvider(responses=[RuntimeError("endpoint down")])
    state, budget = TurnState(), make_budget()
    emitter, _ = _emitter(provider, state, budget)

    decision = await decide(
        query="q", context=empty_context, state=state, budget=budget, emitter=emitter
    )

    assert decision.action == TurnAction.RETRIEVE
    # The attempted call still consumed ledger budget (charge-before-call).
    assert state.llm_calls == 1


async def test_exhausted_ledger_fails_open_without_provider_call(empty_context):
    provider = FakeProvider(responses=[decision_json("ANSWER")])
    state, budget = TurnState(), make_budget(max_llm_calls=1)
    state.llm_calls = 1
    emitter, _ = _emitter(provider, state, budget)

    decision = await decide(
        query="q", context=empty_context, state=state, budget=budget, emitter=emitter
    )

    assert decision.action == TurnAction.RETRIEVE
    assert provider.calls == []  # never reached the provider


async def test_empty_retrieve_query_normalized_to_verbatim_query(empty_context):
    provider = FakeProvider(responses=[decision_json("RETRIEVE", query_text="")])
    state, budget = TurnState(), make_budget()
    emitter, _ = _emitter(provider, state, budget)

    decision = await decide(
        query="the real question",
        context=empty_context,
        state=state,
        budget=budget,
        emitter=emitter,
    )

    assert isinstance(decision.args, RetrieveArgs)
    assert decision.args.query_text == "the real question"


async def test_prompt_substitutes_every_template_variable(empty_context):
    provider = FakeProvider(responses=[decision_json("ANSWER")])
    state, budget = TurnState(), make_budget()
    emitter, _ = _emitter(provider, state, budget)

    await decide(
        query="q", context=empty_context, state=state, budget=budget, emitter=emitter
    )

    _, messages, _ = provider.calls[0]
    prompt = messages[0]["content"]
    assert "{{" not in prompt  # every {{ var }} token was rendered
    assert "q" in prompt


def test_evidence_digest_carries_verbatim_refs_and_judge_gaps():
    """Digest lines expose document_id/source_key (DEEP_STUDY copy source) and
    the latest judge verdict's missing_information from the trace."""
    state = TurnState()
    state.pool.append(make_chunk("c1", document_id="doc-7", source_key="docs/d7.md"))
    state.tried_queries.append("first query")
    state.events.append(
        TurnEvent.now(
            TurnEventType.JUDGE_VERDICT,
            {
                "round": 0,
                "kept": 1,
                "sufficient": False,
                "confidence": 0.4,
                "missing_information": "the frequency limits table",
            },
        )
    )

    digest = build_evidence_digest(state)

    assert "document_id=doc-7" in digest
    assert "source_key=docs/d7.md" in digest
    assert "first query" in digest
    assert "the frequency limits table" in digest


def test_evidence_digest_is_deterministic_and_handles_empty_pool():
    state = TurnState()
    assert build_evidence_digest(state) == build_evidence_digest(state)
    assert "no evidence" in build_evidence_digest(state)
