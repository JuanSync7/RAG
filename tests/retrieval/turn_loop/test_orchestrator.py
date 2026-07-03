# @summary
# End-to-end tests for run_turn_loop on fakes: action dispatch to the right
# stage, event-stream ordering (turn_action precedes its action's events,
# draft precedes gate), gate pass/fail loop-back feeding GateFeedback into the
# next controller prompt, terminal CLARIFY (ask_user), single-ledger LLM-call
# accounting across purposes, budget-exhaustion best-effort paths (max_actions
# with a final draft; max_llm_calls with the cannot-answer digest;
# max_answer_attempts reusing the best failed draft), wall-clock stop via a
# patched monotonic clock, and never-raises error containment.
# @end-summary
"""Tests for ``turn_loop/orchestrator.py`` (full loop over fakes)."""

from __future__ import annotations

import types

import pytest

import src.retrieval.pipeline.turn_loop.schemas as turn_schemas
from src.retrieval.pipeline.turn_loop import RouteHint, TurnAction, run_turn_loop
from src.retrieval.pipeline.turn_loop.orchestrator import (
    STOP_CLARIFY,
    STOP_GATE_PASSED,
    STOP_MAX_ACTIONS,
    STOP_MAX_ANSWER_ATTEMPTS,
    STOP_MAX_LLM_CALLS,
    STOP_WALL_CLOCK,
)
from src.retrieval.pipeline.turn_loop.schemas import (
    TurnEventType,
    TurnLoopResult,
)


def _fast_lane_hint() -> RouteHint:
    return RouteHint(
        initial_action=TurnAction.RETRIEVE,
        effort="fast",
        fast_lane=True,
        reason="high-confidence factoid",
    )

from tests.retrieval.turn_loop.conftest import (
    FakeProvider,
    clarify_json,
    decision_json,
    event_types,
    events_of,
    judge_json,
    make_budget,
    make_chunk,
    make_deps,
    selfscore_json,
)


async def test_full_pass_dispatch_ordering_and_llm_accounting(empty_context):
    """RETRIEVE (with HyDE) then ANSWER passing the gate: verifies dispatch,
    the §8 event ordering, and the single LLM ledger across purposes."""
    provider = FakeProvider(
        responses=[
            decision_json(
                "RETRIEVE",
                query_text="timing values",
                hypothetical_answer="The timing values are ...",
            ),
            judge_json([0, 1], 2, confidence=0.9),
            decision_json("ANSWER"),
            selfscore_json(0.95),
        ],
        streams=[[("reasoning", "let me check"), ("content", "Answer [1] and [2].")]],
    )
    deps, emitted = make_deps(
        provider, retrieve_batches=[[make_chunk("c1"), make_chunk("c2")]]
    )
    budget = make_budget()

    result = await run_turn_loop("q", empty_context, deps, budget)

    assert result.action == TurnLoopResult.ACTION_ANSWERED
    assert result.stop_reason == STOP_GATE_PASSED
    assert result.answer == "Answer [1] and [2]."
    assert result.confidence == pytest.approx(0.935)
    assert result.actions_taken == 2
    # Single ledger: controller x2 + judge + draft + self_score = 5.
    assert result.llm_calls == 5
    purposes = [p["purpose"] for p in events_of(emitted, TurnEventType.LLM_CALL)]
    assert purposes == ["controller", "judge", "controller", "draft", "self_score"]

    types_seq = event_types(emitted)
    actions = [i for i, t in enumerate(types_seq) if t == TurnEventType.TURN_ACTION]
    # turn_action precedes its action's events.
    assert actions[0] < types_seq.index(TurnEventType.HYDE_QUERY)
    assert actions[0] < types_seq.index(TurnEventType.RETRIEVE_RESULT)
    assert actions[0] < types_seq.index(TurnEventType.JUDGE_VERDICT)
    assert actions[1] < types_seq.index(TurnEventType.DRAFT)
    # draft precedes gate.
    assert types_seq.index(TurnEventType.DRAFT) < types_seq.index(TurnEventType.GATE)
    # turn_action payloads carry the decision.
    action_events = events_of(emitted, TurnEventType.TURN_ACTION)
    assert [a["action"] for a in action_events] == ["RETRIEVE", "ANSWER"]
    # index is the 1-based, human-facing action number (design §8 "#2 RETRIEVE").
    assert [a["index"] for a in action_events] == [1, 2]
    # The result trace is the same ordered record.
    assert event_types(result.trace) == types_seq


async def test_gate_failure_feeds_feedback_into_next_controller_prompt(
    empty_context,
):
    """A failed ANSWER attempt loops back; the next controller prompt carries
    the gate feedback; a CLARIFY decision then ends the turn as ask_user."""
    provider = FakeProvider(
        responses=[
            decision_json("ANSWER"),
            selfscore_json(0.1, ["invented detail"]),
            decision_json("CLARIFY"),
            clarify_json("Which one?", ["First", "Second", "Third"]),
        ],
        streams=[[("content", "weak uncited draft")]],
    )
    deps, emitted = make_deps(provider)
    budget = make_budget(clarify_max_hints=2)

    result = await run_turn_loop("q", empty_context, deps, budget)

    assert result.action == TurnLoopResult.ACTION_ASK_USER
    assert result.stop_reason == STOP_CLARIFY
    assert result.answer == ""
    assert result.clarification is not None
    assert result.clarification.question == "Which one?"
    assert result.clarification.hints == ["First", "Second"]  # capped
    # The second controller call saw the gate feedback (call order:
    # controller, draft stream, self-score, controller, clarify).
    second_controller_prompt = provider.calls[3][1][0]["content"]
    assert "failed the confidence gate" in second_controller_prompt
    assert "invented detail" in second_controller_prompt


async def test_max_actions_exhaustion_takes_one_final_best_effort_draft(
    empty_context,
):
    provider = FakeProvider(
        responses=[
            decision_json("RETRIEVE", query_text="q1"),
            judge_json([0], 1, confidence=0.4, missing="more detail"),
            selfscore_json(0.9),  # final best-effort draft self-score
        ],
        streams=[[("content", "best effort answer [1]")]],
    )
    deps, emitted = make_deps(provider, retrieve_batches=[[make_chunk("c1")]])
    budget = make_budget(max_actions=1)

    result = await run_turn_loop("q", empty_context, deps, budget)

    assert result.stop_reason == STOP_MAX_ACTIONS
    assert result.action == TurnLoopResult.ACTION_ANSWERED
    assert result.answer == "best effort answer [1]"
    assert result.confidence > 0.0


async def test_llm_budget_exhaustion_returns_explicit_cannot_answer_digest(
    empty_context,
):
    """Ledger fully consumed: no final draft is possible — the answer is the
    explicit low-confidence USER-FACING message, never empty and never the
    controller's internal digest."""
    provider = FakeProvider(responses=[decision_json("RETRIEVE", query_text="q1")])
    deps, emitted = make_deps(provider, retrieve_batches=[[make_chunk("c1")]])
    # 1 call: the controller decision consumes the whole ledger; the judge is
    # then skipped fail-open and the loop stops at the top of iteration 2.
    budget = make_budget(max_llm_calls=1)

    result = await run_turn_loop("q", empty_context, deps, budget)

    assert result.stop_reason == STOP_MAX_LLM_CALLS
    assert result.llm_calls == 1
    assert "could not reach a confident answer" in result.answer
    assert result.answer.strip() != ""
    # The fallback is user-facing (source display names + capped previews).
    # None of the controller digest's internals may leak into the returned /
    # memory-persisted answer: chunk ids, storage keys, id= prompt markers,
    # tried-query/HyDE sections, judge internals.
    assert "c1" not in result.answer
    assert "document_id=" not in result.answer
    assert "source_key=" not in result.answer
    assert "docs/doc-1.md" not in result.answer  # the MinIO object key
    assert "Queries already tried" not in result.answer
    assert "judge verdict" not in result.answer
    # The human-readable evidence summary IS present.
    assert "Doc One" in result.answer
    assert [c.chunk_id for c in result.pool] == ["c1"]  # kept fail-open, un-judged


async def test_answer_attempts_cap_reuses_best_failed_draft(empty_context):
    """Controller insists on ANSWER past max_answer_attempts: the loop exits
    best-effort with the best gate-scored draft already made."""
    provider = FakeProvider(
        responses=[
            decision_json("ANSWER"),
            selfscore_json(0.2),
            decision_json("ANSWER"),
        ],
        streams=[[("content", "the only draft")]],
    )
    deps, emitted = make_deps(provider)
    budget = make_budget(max_answer_attempts=1)

    result = await run_turn_loop("q", empty_context, deps, budget)

    assert result.stop_reason == STOP_MAX_ANSWER_ATTEMPTS
    assert result.answer == "the only draft"
    # No second draft was attempted: one stream call only.
    assert [c[0] for c in provider.calls].count("generate_stream") == 1


async def test_wall_clock_stop(empty_context, monkeypatch):
    """Patched monotonic clock: the wall-clock budget trips at the first loop
    check; the exit is the best-effort final draft, stop_reason wall_clock."""
    # Base the fake clock on the real one: TurnState.started_monotonic's
    # default_factory captured the real time.monotonic at class definition,
    # so only elapsed_ms's module-global lookup sees the patch.
    clock = {"now": turn_schemas.time.monotonic()}

    def fake_monotonic():
        clock["now"] += 120.0  # every look at the clock costs 120s
        return clock["now"]

    # Patch only the schemas module's clock (TurnState.elapsed_ms), not the
    # global time module — asyncio's own clock must stay sane.
    monkeypatch.setattr(
        turn_schemas,
        "time",
        types.SimpleNamespace(monotonic=fake_monotonic, time=turn_schemas.time.time),
    )
    provider = FakeProvider(
        responses=[selfscore_json(0.9)],
        streams=[[("content", "rushed answer")]],
    )
    deps, emitted = make_deps(provider)
    budget = make_budget(wall_clock_ms=60_000)

    result = await run_turn_loop("q", empty_context, deps, budget)

    assert result.stop_reason == STOP_WALL_CLOCK
    assert result.actions_taken == 0  # never reached a controller decision
    assert result.answer == "rushed answer"
    assert result.elapsed_ms >= 60_000


async def test_controller_fail_open_drives_first_action_to_retrieve(empty_context):
    """Garbage controller output on iteration 1 still produces a retrieval
    round with the verbatim query (design §6 fail-open, end to end)."""
    provider = FakeProvider(
        responses=[
            "garbage",  # controller iteration 1 -> fail-open RETRIEVE
            judge_json([0], 1, confidence=0.9),
            decision_json("ANSWER"),
            selfscore_json(0.95),
        ],
        streams=[[("content", "grounded [1]")]],
    )
    captured: list = []

    async def capture_retrieve(query_text, hyde_text, top_k):
        captured.append((query_text, hyde_text, top_k))
        return [make_chunk("c1")]

    deps, emitted = make_deps(provider)
    deps.retrieve_ranked = capture_retrieve
    budget = make_budget()

    result = await run_turn_loop("verbatim user query", empty_context, deps, budget)

    assert captured == [("verbatim user query", None, budget.retrieve_top_k)]
    assert result.stop_reason == STOP_GATE_PASSED


async def test_unexpected_error_exits_best_effort_not_raise(empty_context):
    """An exploding event sink is swallowed; but even a hard failure inside
    the loop machinery must surface as a result, never an exception."""
    provider = FakeProvider(responses=[decision_json("RETRIEVE", query_text="q")])
    deps, emitted = make_deps(provider)

    async def exploding_retrieve(query_text, hyde_text, top_k):
        raise MemoryError("catastrophic")

    deps.retrieve_ranked = exploding_retrieve
    budget = make_budget(max_actions=1, max_llm_calls=1)

    result = await run_turn_loop("q", empty_context, deps, budget)

    assert isinstance(result, TurnLoopResult)
    assert result.answer  # never empty


async def test_fast_lane_skips_controller_retrieve_then_answer(empty_context):
    """Router fast lane: iteration 0 RETRIEVE and iteration 1 ANSWER are both
    seeded deterministically — NO controller LLM call is made (the ~2-round-trip
    latency win), and both turn_action events are sourced 'router'."""
    provider = FakeProvider(
        responses=[
            judge_json([0], 1, confidence=0.9),  # RETRIEVE round judge
            selfscore_json(0.95),  # ANSWER self-score
        ],
        streams=[[("content", "The answer is [1].")]],
    )
    deps, emitted = make_deps(provider, retrieve_batches=[[make_chunk("c1")]])
    budget = make_budget()

    result = await run_turn_loop(
        "what is the reset value?", empty_context, deps, budget,
        route_hint=_fast_lane_hint(),
    )

    assert result.action == TurnLoopResult.ACTION_ANSWERED
    assert result.stop_reason == STOP_GATE_PASSED
    # No controller call: the two decisions were seeded by the router.
    purposes = [p["purpose"] for p in events_of(emitted, TurnEventType.LLM_CALL)]
    assert purposes == ["judge", "draft", "self_score"]
    action_events = events_of(emitted, TurnEventType.TURN_ACTION)
    assert [a["action"] for a in action_events] == ["RETRIEVE", "ANSWER"]
    assert [a["source"] for a in action_events] == ["router", "router"]


async def test_fast_lane_gate_failure_reengages_controller(empty_context):
    """A fast-lane ANSWER that fails the gate must hand control back to the
    controller (fail-open escalation): iteration 2 is a real controller call."""
    provider = FakeProvider(
        responses=[
            judge_json([0], 1, confidence=0.9),  # RETRIEVE judge
            selfscore_json(0.1, ["invented"]),  # weak ANSWER -> gate fails
            decision_json("CLARIFY"),  # controller re-engages
            clarify_json("Which subsystem?", ["A", "B"]),
        ],
        streams=[[("content", "weak uncited draft")]],
    )
    deps, emitted = make_deps(provider, retrieve_batches=[[make_chunk("c1")]])
    budget = make_budget()

    result = await run_turn_loop(
        "q", empty_context, deps, budget, route_hint=_fast_lane_hint()
    )

    assert result.action == TurnLoopResult.ACTION_ASK_USER
    action_events = events_of(emitted, TurnEventType.TURN_ACTION)
    assert [a["action"] for a in action_events] == ["RETRIEVE", "ANSWER", "CLARIFY"]
    assert [a["source"] for a in action_events] == ["router", "router", "controller"]
    purposes = [p["purpose"] for p in events_of(emitted, TurnEventType.LLM_CALL)]
    assert "controller" in purposes  # the re-engaged controller call


async def test_fast_lane_empty_pool_hands_answer_to_controller(empty_context):
    """When the fast-lane RETRIEVE returns nothing, iteration 1 must NOT force
    an ANSWER on an empty pool — the controller decides instead."""
    provider = FakeProvider(
        responses=[
            decision_json("CLARIFY"),  # controller opens iteration 1
            clarify_json("Can you specify?", ["X"]),
        ],
    )
    deps, emitted = make_deps(provider, retrieve_batches=[[]])  # empty retrieve
    budget = make_budget()

    result = await run_turn_loop(
        "q", empty_context, deps, budget, route_hint=_fast_lane_hint()
    )

    assert result.action == TurnLoopResult.ACTION_ASK_USER
    action_events = events_of(emitted, TurnEventType.TURN_ACTION)
    assert [a["action"] for a in action_events] == ["RETRIEVE", "CLARIFY"]
    # iteration 0 seeded by the router; iteration 1 fell through to the controller.
    assert [a["source"] for a in action_events] == ["router", "controller"]


async def test_non_fast_lane_hint_still_calls_controller(empty_context):
    """A DECOMPOSE seed (fast_lane=False) is advisory only: the controller is
    still called on iteration 0 (source 'controller')."""
    hint = RouteHint(
        initial_action=TurnAction.DECOMPOSE,
        effort="balanced",
        fast_lane=False,
        reason="compound query",
    )
    provider = FakeProvider(responses=[decision_json("CLARIFY"), clarify_json("?", ["a"])])
    deps, emitted = make_deps(provider)
    budget = make_budget()

    result = await run_turn_loop("a and b", empty_context, deps, budget, route_hint=hint)

    assert result.action == TurnLoopResult.ACTION_ASK_USER
    action_events = events_of(emitted, TurnEventType.TURN_ACTION)
    assert action_events[0]["source"] == "controller"


async def test_no_progress_guard_forces_answer_after_stalled_retrieval(empty_context):
    """Once ``max_no_progress_rounds`` gather rounds add no new chunks, the loop
    forces an ANSWER (source=loop_guard) instead of retrieving the same chunks
    forever — the observed 'controller won't commit' pathology."""
    provider = FakeProvider(
        responses=[
            decision_json("RETRIEVE", query_text="q1"),
            judge_json([0], 1, confidence=0.9),
            decision_json("RETRIEVE", query_text="q2"),
            decision_json("RETRIEVE", query_text="q3"),
            selfscore_json(0.95),  # forced-answer self-score
        ],
        streams=[[("content", "grounded answer [1]")]],
    )
    # Same chunk every round → rounds 2 and 3 add nothing (all dup).
    deps, emitted = make_deps(
        provider,
        retrieve_batches=[[make_chunk("c1")], [make_chunk("c1")], [make_chunk("c1")]],
    )
    budget = make_budget(max_actions=10, max_no_progress_rounds=2)

    result = await run_turn_loop("q", empty_context, deps, budget)

    assert result.stop_reason == STOP_GATE_PASSED
    action_events = events_of(emitted, TurnEventType.TURN_ACTION)
    assert [a["action"] for a in action_events] == [
        "RETRIEVE", "RETRIEVE", "RETRIEVE", "ANSWER",
    ]
    # The ANSWER was forced by the loop guard, not chosen by the controller.
    assert [a["source"] for a in action_events] == [
        "controller", "controller", "controller", "loop_guard",
    ]
    # The forced ANSWER made NO controller LLM call.
    purposes = [p["purpose"] for p in events_of(emitted, TurnEventType.LLM_CALL)]
    assert purposes == [
        "controller", "judge", "controller", "controller", "draft", "self_score",
    ]


async def test_no_progress_guard_disabled_with_zero_setting(empty_context):
    """max_no_progress_rounds=0 disables the guard — the controller keeps
    control (and here retrieves duplicates until max_actions)."""
    provider = FakeProvider(
        responses=[
            decision_json("RETRIEVE", query_text="q1"),
            judge_json([0], 1, confidence=0.4, missing="more"),
            decision_json("RETRIEVE", query_text="q2"),
            decision_json("RETRIEVE", query_text="q3"),
            selfscore_json(0.9),  # best-effort final draft self-score
        ],
        streams=[[("content", "best effort [1]")]],
    )
    deps, emitted = make_deps(
        provider,
        retrieve_batches=[[make_chunk("c1")], [make_chunk("c1")], [make_chunk("c1")]],
    )
    budget = make_budget(max_actions=3, max_no_progress_rounds=0)

    result = await run_turn_loop("q", empty_context, deps, budget)

    assert result.stop_reason == STOP_MAX_ACTIONS
    action_events = events_of(emitted, TurnEventType.TURN_ACTION)
    assert [a["action"] for a in action_events] == ["RETRIEVE", "RETRIEVE", "RETRIEVE"]
    assert all(a["source"] == "controller" for a in action_events)


async def test_wall_clock_stops_early_leaving_min_call_headroom(monkeypatch):
    """The loop must exit to best-effort while one meaningful call still fits.

    Class: budget-tail thrash — actions started with < min_call_budget_ms of
    wall clock left can only fire near-zero-timeout LLM calls (observed live
    as cascading 1-2s controller/judge timeouts).
    """
    from src.retrieval.pipeline.turn_loop.orchestrator import (
        STOP_WALL_CLOCK,
        _budget_stop_reason,
    )
    from src.retrieval.pipeline.turn_loop.schemas import TurnState

    state = TurnState()
    budget = make_budget(wall_clock_ms=60_000, min_call_budget_ms=8_000)
    monkeypatch.setattr(state, "elapsed_ms", lambda: 53_000)  # 7s left < 8s floor

    assert _budget_stop_reason(state, budget) == STOP_WALL_CLOCK
