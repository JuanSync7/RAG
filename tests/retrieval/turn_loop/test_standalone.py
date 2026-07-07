# @summary
# Unit tests for turn_loop/standalone.py: the follow-up standalone-query
# resolver. Covers the fresh-turn no-LLM fast path, a resolved back-reference,
# use_as_is / empty / unchanged no-ops, fail-open on provider error, the config
# toggle, and render_history formatting. Pure fakes (FakeProvider), no infra.
# @end-summary
"""Tests for history-stripped standalone-query resolution."""

from __future__ import annotations

import json

import pytest

from src.retrieval.pipeline.turn_loop.events import TurnEventEmitter
from src.retrieval.pipeline.turn_loop.schemas import (
    TurnContext,
    TurnEventType,
    TurnState,
)
from src.retrieval.pipeline.turn_loop.standalone import (
    render_history,
    resolve_standalone_query,
)

from tests.retrieval.turn_loop.conftest import (
    FakeProvider,
    make_budget,
    make_deps,
)


def _rewrite(decision: str, processed: str, turns: int = 1) -> str:
    """Serialize a retrieval_query_rewriter response."""
    return json.dumps(
        {
            "decision": decision,
            "processed_query": processed,
            "history_turns_used": turns,
        }
    )


def _follow_up_context() -> TurnContext:
    """A conversation with one prior turn to resolve a pronoun against."""
    return TurnContext(
        conversation_id="conv-1",
        recent_turns=[{"query": "What is MBIST?", "answer": "Memory BIST."}],
    )


async def _resolve(provider: FakeProvider, context: TurnContext, query: str):
    """Run the resolver over fakes; return ``(resolved, state, provider)``."""
    state = TurnState()
    budget = make_budget()
    deps, _events = make_deps(provider)
    emitter = TurnEventEmitter(deps=deps, state=state, budget=budget)
    resolved = await resolve_standalone_query(
        query=query,
        context=context,
        state=state,
        budget=budget,
        emitter=emitter,
    )
    return resolved, state, provider


@pytest.mark.asyncio
async def test_fresh_turn_skips_llm_and_returns_verbatim(empty_context):
    """A first turn (no prior turns) must not spend an LLM call."""
    provider = FakeProvider(responses=[])  # would raise if called
    resolved, state, provider = await _resolve(
        provider, empty_context, "What is MBIST?"
    )
    assert resolved == "What is MBIST?"
    assert state.llm_calls == 0
    assert provider.calls == []


@pytest.mark.asyncio
async def test_followup_backreference_is_resolved():
    """A follow-up with history is rewritten to a self-contained query."""
    provider = FakeProvider(
        responses=[_rewrite("partial_history", "steps before inserting MBIST")]
    )
    resolved, state, _ = await _resolve(
        provider, _follow_up_context(), "What are the steps before inserting it?"
    )
    assert resolved == "steps before inserting MBIST"
    assert state.llm_calls == 1
    # The rewriter saw the verbatim query AND the rendered history.
    _method, messages, _kwargs = provider.calls[0]
    prompt = messages[0]["content"]
    assert "USER_QUERY: What are the steps before inserting it?" in prompt
    assert "MBIST" in prompt


@pytest.mark.asyncio
async def test_use_as_is_keeps_literal_query_even_if_processed_differs():
    """A ``use_as_is`` decision never replaces the query (avoids drift)."""
    provider = FakeProvider(
        responses=[_rewrite("use_as_is", "some rewritten thing")]
    )
    resolved, _state, _ = await _resolve(
        provider, _follow_up_context(), "What is CHI?"
    )
    assert resolved == "What is CHI?"


@pytest.mark.asyncio
async def test_empty_or_unchanged_rewrite_is_a_noop():
    """An empty processed_query falls back to the literal query."""
    provider = FakeProvider(responses=[_rewrite("partial_history", "")])
    resolved, _state, _ = await _resolve(
        provider, _follow_up_context(), "compare it to AXI"
    )
    assert resolved == "compare it to AXI"


@pytest.mark.asyncio
async def test_provider_error_fails_open_to_literal_query():
    """A raising provider must not break the turn — return the literal query."""
    provider = FakeProvider(responses=[RuntimeError("llm down")])
    resolved, state, _ = await _resolve(
        provider, _follow_up_context(), "and the next one?"
    )
    assert resolved == "and the next one?"
    # The call was still charged (attempt consumes budget) and surfaced.
    assert state.llm_calls == 1


@pytest.mark.asyncio
async def test_malformed_json_fails_open_to_literal_query():
    """Unparseable rewriter output falls back to the literal query."""
    provider = FakeProvider(responses=["not json at all"])
    resolved, _state, _ = await _resolve(
        provider, _follow_up_context(), "keep going"
    )
    assert resolved == "keep going"


@pytest.mark.asyncio
async def test_disabled_via_setting_skips_resolution(monkeypatch):
    """The config toggle restores verbatim-follow-up behavior (no LLM call)."""
    from config import settings

    monkeypatch.setattr(
        settings, "RAG_TURN_LOOP_STANDALONE_QUERY_ENABLED", False, raising=False
    )
    provider = FakeProvider(responses=[])  # would raise if called
    resolved, state, provider = await _resolve(
        provider, _follow_up_context(), "how does it compare?"
    )
    assert resolved == "how does it compare?"
    assert state.llm_calls == 0
    assert provider.calls == []


@pytest.mark.asyncio
async def test_resolution_emits_charged_llm_call_event():
    """The rewrite is charged and surfaced as an llm_call event (observability)."""
    provider = FakeProvider(responses=[_rewrite("partial_history", "resolved q")])
    state = TurnState()
    budget = make_budget()
    deps, events = make_deps(provider)
    emitter = TurnEventEmitter(deps=deps, state=state, budget=budget)
    await resolve_standalone_query(
        query="what about it?",
        context=_follow_up_context(),
        state=state,
        budget=budget,
        emitter=emitter,
    )
    llm_calls = [e for e in events if e.type == TurnEventType.LLM_CALL]
    assert len(llm_calls) == 1
    assert llm_calls[0].payload["purpose"] == "standalone_query"


def test_render_history_formats_roles_and_skips_empties():
    """recent_turns render as ROLE lines; blank fields and the summary handled."""
    context = TurnContext(
        conversation_id="c",
        rolling_summary="Earlier we discussed AXI.",
        recent_turns=[
            {"query": "What is AXI?", "answer": "A bus protocol."},
            {"query": "", "answer": "orphan answer"},  # blank query skipped
        ],
    )
    rendered = render_history(context, max_chars=0)
    assert "SUMMARY: Earlier we discussed AXI." in rendered
    assert "USER: What is AXI?" in rendered
    assert "ASSISTANT: A bus protocol." in rendered
    assert "ASSISTANT: orphan answer" in rendered
    assert "USER: \n" not in rendered  # the blank query produced no USER line


def test_render_history_empty_for_fresh_conversation(empty_context):
    """A fresh conversation renders no history (drives the no-LLM fast path)."""
    assert render_history(empty_context, max_chars=0) == ""
