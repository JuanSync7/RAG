# @summary
# Tests for the turn-loop event emitter: trace-append always happens, sink
# errors are swallowed (observability never kills the loop), the streaming
# flag gates the sink but not the trace, latest_event_payload returns the
# newest matching payload, and charged_call enforces the single LLM ledger —
# charge-before-call (failures still consume budget), llm_call event with
# timing/tokens on every attempt, None on exhausted ledger without a provider
# call.
# @end-summary
"""Tests for ``turn_loop/events.py``."""

from __future__ import annotations

from src.retrieval.pipeline.turn_loop.events import (
    TurnEventEmitter,
    latest_event_payload,
)
from src.retrieval.pipeline.turn_loop.schemas import (
    TurnEvent,
    TurnEventType,
    TurnState,
)

from tests.retrieval.turn_loop.conftest import (
    FakeProvider,
    make_budget,
    make_deps,
)


async def test_emit_appends_trace_and_forwards_to_sink():
    provider = FakeProvider()
    deps, emitted = make_deps(provider)
    state, budget = TurnState(), make_budget()
    emitter = TurnEventEmitter(deps=deps, state=state, budget=budget, stream_events=True)

    event = await emitter.emit(TurnEventType.GATE, {"attempt": 1})

    assert state.events == [event]
    assert emitted == [event]


async def test_sink_error_is_swallowed_and_trace_still_grows():
    provider = FakeProvider()
    deps, _ = make_deps(provider, emit_error=RuntimeError("SSE bridge died"))
    state, budget = TurnState(), make_budget()
    emitter = TurnEventEmitter(deps=deps, state=state, budget=budget, stream_events=True)

    event = await emitter.emit(TurnEventType.DRAFT, {"attempt": 1})

    assert state.events == [event]  # no exception escaped, trace intact


async def test_stream_events_off_skips_sink_but_keeps_trace():
    provider = FakeProvider()
    deps, emitted = make_deps(provider)
    state, budget = TurnState(), make_budget()
    emitter = TurnEventEmitter(
        deps=deps, state=state, budget=budget, stream_events=False
    )

    await emitter.emit(TurnEventType.CLARIFY, {"question": "?"})

    assert len(state.events) == 1
    assert emitted == []


def test_latest_event_payload_returns_newest_match():
    state = TurnState()
    state.events.append(TurnEvent.now(TurnEventType.JUDGE_VERDICT, {"round": 0}))
    state.events.append(TurnEvent.now(TurnEventType.GATE, {"attempt": 1}))
    state.events.append(TurnEvent.now(TurnEventType.JUDGE_VERDICT, {"round": 1}))

    assert latest_event_payload(state, TurnEventType.JUDGE_VERDICT) == {"round": 1}
    assert latest_event_payload(state, TurnEventType.DRAFT) is None


async def test_charged_call_charges_ledger_and_emits_llm_call_with_tokens():
    provider = FakeProvider(responses=["hello"])
    deps, _ = make_deps(provider)
    state, budget = TurnState(), make_budget()
    emitter = TurnEventEmitter(deps=deps, state=state, budget=budget, stream_events=True)

    response = await emitter.charged_call(
        alias="controller", purpose="controller", prompt="p"
    )

    assert response.content == "hello"
    assert state.llm_calls == 1
    payloads = [e.payload for e in state.events if e.type == TurnEventType.LLM_CALL]
    assert len(payloads) == 1
    assert payloads[0]["alias"] == "controller"
    assert payloads[0]["purpose"] == "controller"
    assert payloads[0]["prompt_tokens"] == 7
    assert payloads[0]["completion_tokens"] == 3
    assert payloads[0]["ms"] >= 0


async def test_charged_call_failure_still_consumes_budget_and_returns_none():
    provider = FakeProvider(responses=[RuntimeError("boom")])
    deps, _ = make_deps(provider)
    state, budget = TurnState(), make_budget()
    emitter = TurnEventEmitter(deps=deps, state=state, budget=budget, stream_events=True)

    response = await emitter.charged_call(alias="judge", purpose="judge", prompt="p")

    assert response is None
    assert state.llm_calls == 1  # a flaky endpoint cannot spin the loop for free
    llm_events = [e for e in state.events if e.type == TurnEventType.LLM_CALL]
    assert len(llm_events) == 1
    assert "prompt_tokens" not in llm_events[0].payload  # timing only


async def test_charged_call_exhausted_ledger_skips_provider():
    provider = FakeProvider(responses=["never used"])
    deps, _ = make_deps(provider)
    state, budget = TurnState(), make_budget(max_llm_calls=2)
    state.llm_calls = 2
    emitter = TurnEventEmitter(deps=deps, state=state, budget=budget, stream_events=True)

    response = await emitter.charged_call(alias="a", purpose="controller", prompt="p")

    assert response is None
    assert state.llm_calls == 2  # unchanged
    assert provider.calls == []
