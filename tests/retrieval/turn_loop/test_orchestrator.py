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

import json
import types

import pytest

import src.retrieval.pipeline.turn_loop.schemas as turn_schemas
from src.retrieval.pipeline.turn_loop import RouteHint, TurnAction, run_turn_loop
from src.retrieval.pipeline.turn_loop.orchestrator import (
    STOP_CLARIFY,
    STOP_FACETS_COVERED,
    STOP_GATE_PASSED,
    STOP_MAX_ACTIONS,
    STOP_MAX_ANSWER_ATTEMPTS,
    STOP_MAX_LLM_CALLS,
    STOP_NO_PROGRESS,
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


async def test_empty_judged_pool_grounds_answer_on_fallback_floor(empty_context):
    """Empty-pool spiral fix: when the round judge rejects EVERY fresh batch the
    judged pool never grows, so the no-progress guard (now firing on the
    fallback floor, not only a non-empty pool) commits to ANSWER and the draft is
    grounded on the best-effort raw chunks — not "(no evidence retrieved)".
    Without the floor the loop would spiral to a max_actions empty best-effort."""
    provider = FakeProvider(
        responses=[
            decision_json("RETRIEVE", query_text="q1"),
            judge_json([], 1, sufficient=False, confidence=0.2),  # round 1 rejects
            decision_json("RETRIEVE", query_text="q2"),
            judge_json([], 1, sufficient=False, confidence=0.2),  # round 2 rejects
            selfscore_json(0.9),  # forced-answer self-score
        ],
        streams=[[("content", "Based on the retrieved sources [1][2], ...")]],
    )
    # Each round retrieves a DIFFERENT chunk the judge rejects → pool never grows.
    deps, emitted = make_deps(
        provider,
        retrieve_batches=[
            [make_chunk("c1", score=0.9)],
            [make_chunk("c2", score=0.7)],
        ],
    )
    budget = make_budget(
        max_actions=10, max_no_progress_rounds=2, answer_confidence_threshold=0.3
    )

    result = await run_turn_loop("q", empty_context, deps, budget)

    action_events = events_of(emitted, TurnEventType.TURN_ACTION)
    assert [a["action"] for a in action_events] == ["RETRIEVE", "RETRIEVE", "ANSWER"]
    # The ANSWER was forced by the loop guard firing on the fallback floor.
    assert action_events[-1]["source"] == "loop_guard"
    assert result.stop_reason == STOP_GATE_PASSED
    # The draft was GROUNDED on the promoted fallback floor (not cannot-answer),
    # and the returned pool carries the cited best-effort sources.
    assert "[1]" in result.answer
    assert [c.chunk_id for c in result.pool] == ["c1", "c2"]


async def test_stalled_gate_failing_answer_exits_without_more_retrieval(empty_context):
    """Latency fix: when gathering has stalled (>= max_no_progress_rounds) AND a
    floor-grounded answer fails the gate, commit the best draft instead of
    resuming futile retrieval to max_actions/wall_clock. A low-confidence
    grounded refusal is the best possible answer for an unanswerable query."""
    provider = FakeProvider(
        responses=[
            decision_json("RETRIEVE", query_text="q1"),
            judge_json([], 1, sufficient=False, confidence=0.2),  # round 1 rejects
            decision_json("RETRIEVE", query_text="q2"),
            judge_json([], 1, sufficient=False, confidence=0.2),  # round 2 rejects
            selfscore_json(0.1),  # low self-score → the grounded answer fails the gate
        ],
        streams=[[("content", "The context does not support an answer [1][2].")]],
    )
    deps, emitted = make_deps(
        provider,
        retrieve_batches=[
            [make_chunk("c1", score=0.9)],
            [make_chunk("c2", score=0.7)],
        ],
    )
    # High threshold so the floor-grounded refusal fails the gate; big action
    # budget so ONLY the stall-exit (not max_actions) can end the turn early.
    budget = make_budget(
        max_actions=10, max_no_progress_rounds=2, answer_confidence_threshold=0.62
    )

    result = await run_turn_loop("q", empty_context, deps, budget)

    # Exited on the stall, NOT max_actions — and after just RETRIEVE,RETRIEVE,ANSWER.
    assert result.stop_reason == STOP_NO_PROGRESS
    action_events = events_of(emitted, TurnEventType.TURN_ACTION)
    assert [a["action"] for a in action_events] == ["RETRIEVE", "RETRIEVE", "ANSWER"]
    # Best grounded draft is still returned (not a bare cannot-answer).
    assert "[1]" in result.answer
    assert [c.chunk_id for c in result.pool] == ["c1", "c2"]


async def test_facet_commit_guard_forces_answer_when_all_facets_covered(empty_context):
    """The DECOMPOSE-spiral fix (different instance of the class than the c001
    comparison that surfaced it — a two-part 'throughput and latency' question):
    once a multi-way DECOMPOSE covers every facet, the loop forces an ANSWER
    (source=facet_guard, NO controller call) instead of gathering the comparison
    it can already synthesize forever."""
    provider = FakeProvider(
        responses=[
            decision_json("DECOMPOSE", question="throughput and latency"),
            # split -> two facets; each leg retrieves a chunk the judge keeps.
            json.dumps({"sub_questions": ["throughput", "latency"]}),
            judge_json([0, 1], 2, confidence=0.9),
            selfscore_json(0.95),  # forced-answer self-score
        ],
        streams=[[("content", "Throughput is X [1]; latency is Y [2].")]],
    )
    deps, emitted = make_deps(
        provider, retrieve_batches=[[make_chunk("c1")], [make_chunk("c2")]]
    )
    budget = make_budget(max_actions=10, decompose_anchor_raw=False)

    result = await run_turn_loop("throughput and latency?", empty_context, deps, budget)

    assert result.stop_reason == STOP_GATE_PASSED
    action_events = events_of(emitted, TurnEventType.TURN_ACTION)
    assert [a["action"] for a in action_events] == ["DECOMPOSE", "ANSWER"]
    # The ANSWER was forced by the facet guard, not chosen by the controller.
    assert [a["source"] for a in action_events] == ["controller", "facet_guard"]
    # The forced ANSWER made NO controller LLM call (split=decompose, not controller).
    purposes = [p["purpose"] for p in events_of(emitted, TurnEventType.LLM_CALL)]
    assert purposes == ["controller", "decompose", "judge", "draft", "self_score"]


async def test_shape_compound_opening_retrieve_becomes_decompose_then_commits(
    empty_context,
):
    """The c002 fix, end to end (a DIFFERENT instance of the class than c001's
    facet-guard test above): the controller labels the OPENING query compound
    (query_shape) yet still chooses a single RETRIEVE — the exact c002 pathology
    where it recognised the comparison in its own reason but retrieved anyway,
    burning 8 LLM calls before a guard rescued it. The shape coercion rewrites
    that opening move to DECOMPOSE, so the facets fan out at once and the
    facet-commit guard answers on the very next iteration — collapsing to c001's
    minimal DECOMPOSE→facet_guard→ANSWER path (one controller call, not eight)."""
    provider = FakeProvider(
        responses=[
            # Controller opens with a single RETRIEVE but self-labels the query
            # compound — the coercion (not a second controller round) fixes it.
            decision_json(
                "RETRIEVE", query_text="AXI4 features", query_shape="compound"
            ),
            json.dumps({"sub_questions": ["AXI4 features", "AXI4-Lite features"]}),
            judge_json([0, 1, 2], 3, confidence=0.9),
            selfscore_json(0.95),  # forced-answer self-score
        ],
        streams=[[("content", "AXI4 is X [1]; AXI4-Lite is Y [2].")]],
    )
    # Anchor ON (the default, and the coercion's precondition): the DECOMPOSE
    # fan-out is [raw-query anchor] + [subq1, subq2] = 3 legs, so 3 batches.
    deps, emitted = make_deps(
        provider,
        retrieve_batches=[[make_chunk("c0")], [make_chunk("c1")], [make_chunk("c2")]],
    )
    budget = make_budget(max_actions=10)

    result = await run_turn_loop(
        "compare AXI4 and AXI4-Lite", empty_context, deps, budget
    )

    assert result.stop_reason == STOP_GATE_PASSED
    action_events = events_of(emitted, TurnEventType.TURN_ACTION)
    # The dispatched first action is DECOMPOSE even though the controller SAID
    # RETRIEVE — proof the query_shape coercion fired.
    assert [a["action"] for a in action_events] == ["DECOMPOSE", "ANSWER"]
    assert [a["source"] for a in action_events] == ["controller", "facet_guard"]
    # ONE controller call total — the 8-call c002 spiral is gone.
    purposes = [p["purpose"] for p in events_of(emitted, TurnEventType.LLM_CALL)]
    assert purposes == ["controller", "decompose", "judge", "draft", "self_score"]


async def test_facet_guard_commits_on_sufficient_pool_when_coverage_incomplete(
    empty_context,
):
    """c002 class (a DIFFERENT instance than the fully-covered facet test): a
    DECOMPOSE whose per-facet coverage never completes — because a thin first
    round covers only some facets and later rounds rephrase the sub-queries, so
    the cumulative facet set never fully covers — would otherwise spiral
    (re-DECOMPOSE forever). But the round judge has explicitly marked the pool
    SUFFICIENT, so the decomposed question IS answerable: the facet-commit guard
    commits on that holistic signal even though not every tracked facet has its
    own kept chunk. Here the judge keeps only facet 'a' yet reports sufficient."""
    provider = FakeProvider(
        responses=[
            decision_json("DECOMPOSE", question="a and b"),
            json.dumps({"sub_questions": ["a", "b"]}),
            judge_json([0], 2, sufficient=True, confidence=0.9),  # keeps a, not b
            selfscore_json(0.95),
        ],
        streams=[[("content", "a is X [1]; b is inferred.")]],
    )
    deps, emitted = make_deps(
        provider, retrieve_batches=[[make_chunk("c1")], [make_chunk("c2")]]
    )
    budget = make_budget(max_actions=10, decompose_anchor_raw=False)

    result = await run_turn_loop("a and b?", empty_context, deps, budget)

    assert result.stop_reason == STOP_GATE_PASSED
    action_events = events_of(emitted, TurnEventType.TURN_ACTION)
    # No second gather round — the sufficient verdict committed the turn.
    assert [a["action"] for a in action_events] == ["DECOMPOSE", "ANSWER"]
    assert action_events[-1]["source"] == "facet_guard"
    purposes = [p["purpose"] for p in events_of(emitted, TurnEventType.LLM_CALL)]
    assert purposes == ["controller", "decompose", "judge", "draft", "self_score"]


async def test_facet_guard_holds_when_pool_insufficient_and_coverage_incomplete(
    empty_context,
):
    """The complement: after a DECOMPOSE, if the judge marks the pool NOT
    sufficient AND not every facet is covered, the guard must NOT commit — the
    decomposed question genuinely needs more gathering, so the controller keeps
    control (never a forced facet_guard ANSWER on an insufficient pool)."""
    provider = FakeProvider(
        responses=[
            decision_json("DECOMPOSE", question="a and b"),
            json.dumps({"sub_questions": ["a", "b"]}),
            judge_json([0], 2, sufficient=False, confidence=0.4),  # a only, insufficient
            decision_json("RETRIEVE", query_text="more about b"),  # controller gathers on
        ]
        # (the run may continue past here; we only assert the guard stayed out
        # for the decision immediately after the DECOMPOSE)
        + [judge_json([], 0, sufficient=False, confidence=0.1)] * 6
        + [decision_json("ANSWER")] * 3
        + [selfscore_json(0.2)] * 3,
        streams=[[("content", "partial [1].")]] * 4,
    )
    deps, emitted = make_deps(
        provider,
        retrieve_batches=[[make_chunk("c1")], [make_chunk("c2")]],
    )
    budget = make_budget(max_actions=10, decompose_anchor_raw=False)

    result = await run_turn_loop("a and b?", empty_context, deps, budget)

    action_events = events_of(emitted, TurnEventType.TURN_ACTION)
    assert action_events[0]["action"] == "DECOMPOSE"
    # The decision right after the DECOMPOSE was the controller's — the guard did
    # NOT force an ANSWER on the insufficient, incompletely-covered pool.
    assert action_events[1]["source"] == "controller"
    assert not any(a["source"] == "facet_guard" for a in action_events)


async def test_facet_commit_guard_disabled_hands_iteration_to_controller(empty_context):
    """facet_commit_enabled=False disarms the guard: after DECOMPOSE the
    controller decides the next action as before (here it chooses ANSWER)."""
    provider = FakeProvider(
        responses=[
            decision_json("DECOMPOSE", question="a and b"),
            json.dumps({"sub_questions": ["a", "b"]}),
            judge_json([0, 1], 2, confidence=0.9),
            decision_json("ANSWER"),  # controller keeps control on iteration 1
            selfscore_json(0.95),
        ],
        streams=[[("content", "a is X [1]; b is Y [2].")]],
    )
    deps, emitted = make_deps(
        provider, retrieve_batches=[[make_chunk("c1")], [make_chunk("c2")]]
    )
    budget = make_budget(max_actions=10, facet_commit_enabled=False)

    result = await run_turn_loop("a and b?", empty_context, deps, budget)

    assert result.stop_reason == STOP_GATE_PASSED
    action_events = events_of(emitted, TurnEventType.TURN_ACTION)
    assert [a["action"] for a in action_events] == ["DECOMPOSE", "ANSWER"]
    assert [a["source"] for a in action_events] == ["controller", "controller"]
    # A second controller call WAS made (the guard did not preempt it).
    purposes = [p["purpose"] for p in events_of(emitted, TurnEventType.LLM_CALL)]
    assert purposes.count("controller") == 2


async def test_facet_covered_gate_failure_commits_best_draft(empty_context):
    """A facet-covered pool whose forced ANSWER fails the gate must commit the
    best grounded draft (stop_reason=facets_covered) rather than resume futile
    retrieval — the comparison is as complete as it will get, so more gathering
    would only re-fail. Mirrors the no-progress stall-exit for the facet signal."""
    provider = FakeProvider(
        responses=[
            decision_json("DECOMPOSE", question="a and b"),
            json.dumps({"sub_questions": ["a", "b"]}),
            judge_json([0, 1], 2, confidence=0.9),
            selfscore_json(0.1),  # weak self-score -> the grounded draft fails the gate
        ],
        streams=[[("content", "Partial comparison [1][2].")]],
    )
    deps, emitted = make_deps(
        provider, retrieve_batches=[[make_chunk("c1")], [make_chunk("c2")]]
    )
    # High threshold so the draft fails; large action budget so ONLY the
    # facet-commit exit (not max_actions) can end the turn early.
    budget = make_budget(max_actions=10, answer_confidence_threshold=0.9, decompose_anchor_raw=False)

    result = await run_turn_loop("a and b?", empty_context, deps, budget)

    assert result.stop_reason == STOP_FACETS_COVERED
    action_events = events_of(emitted, TurnEventType.TURN_ACTION)
    # No further retrieval after the covered DECOMPOSE — just the forced ANSWER.
    assert [a["action"] for a in action_events] == ["DECOMPOSE", "ANSWER"]
    assert action_events[-1]["source"] == "facet_guard"
    # The best grounded draft is returned (not a bare cannot-answer).
    assert "[1]" in result.answer


async def test_facet_covered_commits_after_answer_budget_spent(empty_context):
    """Guard-precondition-hole class: the facet/no-progress guards force a FRESH
    ANSWER so they need an attempt left. If the controller spends the answer
    budget BEFORE a DECOMPOSE covers the facets (premature ANSWER, then
    DECOMPOSE), the guards can't fire — the loop must STILL commit best-effort
    (facets_covered), not spiral to max_actions. Different interleaving than the
    happy-path facet tests (which decompose first)."""
    provider = FakeProvider(
        responses=[
            decision_json("ANSWER"),  # iteration 0: premature answer on a thin pool
            selfscore_json(0.1),      # -> fails the gate, spends the 1 attempt
            decision_json("DECOMPOSE", question="a and b"),  # iteration 1
            json.dumps({"sub_questions": ["a", "b"]}),
            judge_json([0, 1], 2, confidence=0.9),  # covers both facets
        ],
        streams=[[("content", "weak premature draft")]],
    )
    deps, emitted = make_deps(
        provider, retrieve_batches=[[make_chunk("c1")], [make_chunk("c2")]]
    )
    # One attempt only + generous actions: only the facet-covered commit (NOT
    # max_actions) can end the turn once the attempt is spent pre-coverage.
    budget = make_budget(
        max_actions=10, max_answer_attempts=1, answer_confidence_threshold=0.9,
        decompose_anchor_raw=False,
    )

    result = await run_turn_loop("a and b?", empty_context, deps, budget)

    assert result.stop_reason == STOP_FACETS_COVERED
    actions = [a["action"] for a in events_of(emitted, TurnEventType.TURN_ACTION)]
    assert actions == ["ANSWER", "DECOMPOSE"]  # no RETRIEVE spiral after coverage
    assert "weak premature draft" in result.answer  # best (only) draft committed


async def test_failopen_judge_during_decompose_does_not_force_facet_answer(empty_context):
    """A DECOMPOSE whose judge fails open (keep-all, no validation) must NOT trip
    the facet guard on the next iteration — coverage requires a real judgment.
    The controller keeps control (here it CLARIFYs)."""
    provider = FakeProvider(
        responses=[
            decision_json("DECOMPOSE", question="a and b"),
            json.dumps({"sub_questions": ["a", "b"]}),
            "unparseable judge output",  # keep-all fail-open, pool_verdict None
            decision_json("CLARIFY"),    # iteration 1: controller, NOT facet_guard
            clarify_json("Which aspect?", ["A", "B"]),
        ],
    )
    deps, emitted = make_deps(
        provider, retrieve_batches=[[make_chunk("c1")], [make_chunk("c2")]]
    )
    budget = make_budget(max_actions=10)

    result = await run_turn_loop("a and b?", empty_context, deps, budget)

    assert result.action == TurnLoopResult.ACTION_ASK_USER
    action_events = events_of(emitted, TurnEventType.TURN_ACTION)
    assert [a["action"] for a in action_events] == ["DECOMPOSE", "CLARIFY"]
    # The post-DECOMPOSE decision was the controller's, NOT a forced facet ANSWER.
    assert action_events[1]["source"] == "controller"


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


import json as _json

from src.retrieval.pipeline.turn_loop.schemas import TurnContext as _TurnContext


def _rewrite_json(decision: str, processed: str) -> str:
    """Serialize a retrieval_query_rewriter response (standalone stage)."""
    return _json.dumps(
        {"decision": decision, "processed_query": processed, "history_turns_used": 1}
    )


async def test_followup_resolves_standalone_query_for_retrieval_not_generation():
    """A follow-up turn seeds RETRIEVAL from the history-resolved standalone
    query, while GENERATION keeps the verbatim user query (memory grounds the
    answer without poisoning the retrieval seed — the multi-turn fix)."""
    context = _TurnContext(
        conversation_id="conv-1",
        recent_turns=[{"query": "What is MBIST?", "answer": "Memory BIST."}],
    )
    provider = FakeProvider(
        responses=[
            # 1) standalone resolver runs BEFORE the loop
            _rewrite_json("partial_history", "steps before inserting MBIST"),
            # 2) controller decides RETRIEVE (no query_text -> falls back to
            #    the resolved query), then 4) ANSWER
            decision_json("RETRIEVE"),
            judge_json([0], 1, confidence=0.9),
            decision_json("ANSWER"),
            selfscore_json(0.95),
        ],
        streams=[[("content", "The steps are [1].")]],
    )
    deps, _emitted = make_deps(provider, retrieve_batches=[[make_chunk("c1")]])

    result = await run_turn_loop(
        "What are the steps before inserting it?", context, deps, make_budget()
    )

    assert result.action == TurnLoopResult.ACTION_ANSWERED
    # First call is the standalone rewrite: it sees the verbatim query + history.
    assert provider.calls[0][0] == "agenerate"
    rewrite_prompt = provider.calls[0][1][0]["content"]
    assert "USER_QUERY: What are the steps before inserting it?" in rewrite_prompt
    assert "MBIST" in rewrite_prompt  # history rendered in

    # The controller (retrieval-facing) is prompted with the RESOLVED query.
    controller_prompts = [
        msgs[0]["content"]
        for method, msgs, _kw in provider.calls
        if method == "agenerate" and "steps before inserting MBIST" in msgs[0]["content"]
    ]
    assert controller_prompts, "controller was not seeded with the resolved query"

    # Generation (draft stream) keeps the VERBATIM follow-up query.
    draft_msgs = [msgs for method, msgs, _kw in provider.calls if method == "generate_stream"]
    assert draft_msgs, "no draft stream call recorded"
    draft_text = draft_msgs[0][-1]["content"]
    assert "Question: What are the steps before inserting it?" in draft_text
