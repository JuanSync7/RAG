# @summary
# Tests for the turn-loop DECOMPOSE stage: one split LLM call → sub-queries,
# PARALLEL retrieve fan-out (one retrieve_ranked call per sub-query), merge +
# cross-sub-query and seen-set dedup, a single judge round over the merged pool,
# two-LLM-call budget charge (split + judge), fail-open to a single sub-query
# when the split yields nothing, and a failed retrieval leg not aborting the
# round. Sub-query coercion (dedup/cap, sub_questions & topics shapes) is unit-
# tested directly.
# @end-summary
"""Tests for ``turn_loop/decompose.py``."""

from __future__ import annotations

import json

from src.retrieval.pipeline.turn_loop.decompose import _coerce_subqueries, run_decompose
from src.retrieval.pipeline.turn_loop.events import TurnEventEmitter
from src.retrieval.pipeline.turn_loop.schemas import (
    DecomposeArgs,
    TurnEventType,
    TurnLoopDeps,
    TurnState,
)

from tests.retrieval.turn_loop.conftest import (
    FakeProvider,
    events_of,
    judge_json,
    make_budget,
    make_chunk,
    make_deps,
)


def _split_json(*subqs: str) -> str:
    return json.dumps({"sub_questions": list(subqs)})


def _setup(provider, batches, *, emitted=None):
    deps, ev = make_deps(provider, retrieve_batches=batches, emitted=emitted)
    state, budget = TurnState(), make_budget()
    emitter = TurnEventEmitter(deps=deps, state=state, budget=budget, stream_events=True)
    return deps, ev, state, budget, emitter


# ── coercion (pure) ──────────────────────────────────────────────────────────

def test_coerce_dedups_and_caps():
    assert _coerce_subqueries({"sub_questions": ["a", "b", "A", "", "c"]}, 2) == ["a", "b"]


def test_coerce_accepts_topics_shape():
    payload = {"topics": [{"questions": ["x"]}, {"questions": ["y", "x"]}]}
    assert _coerce_subqueries(payload, 4) == ["x", "y"]


def test_coerce_non_dict_is_empty():
    assert _coerce_subqueries(["a"], 4) == []


# ── stage behaviour ──────────────────────────────────────────────────────────

async def test_split_fanout_merge_judge_pool():
    # split -> ['a','b']; two parallel retrievals; judge keeps both.
    provider = FakeProvider(responses=[_split_json("a", "b"), judge_json([0, 1], 2)])
    batches = [[make_chunk("c1")], [make_chunk("c2")]]
    deps, ev, state, budget, emitter = _setup(provider, batches)

    await run_decompose(
        DecomposeArgs(question="broad q"),
        query="broad q", state=state, budget=budget, deps=deps, emitter=emitter,
    )

    assert sorted(c.chunk_id for c in state.pool) == ["c1", "c2"]
    result = events_of(ev, TurnEventType.RETRIEVE_RESULT)[0]
    assert result["added"] == 2 and result["sub_queries"] == 2
    # split + judge = exactly two charged LLM calls.
    assert state.llm_calls == 2


async def test_parallel_fanout_calls_retrieve_per_subquery():
    provider = FakeProvider(responses=[_split_json("a", "b", "c"), judge_json([], 0)])
    seen = {"n": 0}

    async def retrieve_ranked(qt, hyde, k):
        seen["n"] += 1
        return []

    async def emit(e):  # noqa: ANN001
        pass

    deps = TurnLoopDeps(retrieve_ranked=retrieve_ranked, fetch_document=None,
                        llm_provider=provider, emit=emit)
    state, budget = TurnState(), make_budget()
    emitter = TurnEventEmitter(deps=deps, state=state, budget=budget, stream_events=True)

    await run_decompose(DecomposeArgs(question="q"), query="q", state=state,
                        budget=budget, deps=deps, emitter=emitter)
    assert seen["n"] == 3  # one retrieve per sub-query, fanned out


async def test_dedup_across_subqueries_and_seen_set():
    provider = FakeProvider(responses=[_split_json("a", "b"), judge_json([0], 1)])
    # both sub-queries surface c1; the second also returns c2 (already seen).
    batches = [[make_chunk("c1")], [make_chunk("c1"), make_chunk("c2")]]
    deps, ev, state, budget, emitter = _setup(provider, batches)
    state.seen_chunk_ids.add("c2")  # already served earlier in the turn

    await run_decompose(DecomposeArgs(question="q"), query="q", state=state,
                        budget=budget, deps=deps, emitter=emitter)
    result = events_of(ev, TurnEventType.RETRIEVE_RESULT)[0]
    assert result["dup"] == 2  # duplicate c1 + already-seen c2


async def test_split_failopen_falls_back_to_single_subquery():
    # split returns unparseable content -> fall back to the question itself.
    provider = FakeProvider(responses=["not json at all", judge_json([0], 1)])
    batches = [[make_chunk("c1")]]
    deps, ev, state, budget, emitter = _setup(provider, batches)

    await run_decompose(DecomposeArgs(question="the whole question"),
                        query="the whole question", state=state, budget=budget,
                        deps=deps, emitter=emitter)

    assert [c.chunk_id for c in state.pool] == ["c1"]
    hyde = events_of(ev, TurnEventType.HYDE_QUERY)[0]
    assert hyde["search_terms"] == ["the whole question"]  # single fallback leg


async def test_one_failed_retrieve_leg_does_not_abort():
    provider = FakeProvider(responses=[_split_json("good", "bad"), judge_json([0], 1)])

    async def retrieve_ranked(qt, hyde, k):
        if qt == "bad":
            raise RuntimeError("backend down")
        return [make_chunk("c1")]

    async def emit(e):  # noqa: ANN001
        pass

    deps = TurnLoopDeps(retrieve_ranked=retrieve_ranked, fetch_document=None,
                        llm_provider=provider, emit=emit)
    state, budget = TurnState(), make_budget()
    emitter = TurnEventEmitter(deps=deps, state=state, budget=budget, stream_events=True)

    await run_decompose(DecomposeArgs(question="q"), query="q", state=state,
                        budget=budget, deps=deps, emitter=emitter)
    assert [c.chunk_id for c in state.pool] == ["c1"]  # good leg survived
