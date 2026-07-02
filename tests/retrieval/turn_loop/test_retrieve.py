# @summary
# Tests for the turn-loop RETRIEVE stage: cross-action dedup against
# seen_chunk_ids (dup counting), judge composition (kept chunks enter the pool
# in listwise-rank order; drops excluded; judge charged to the single ledger),
# fail-open keep-all on unusable judge output (and no judge_verdict event),
# ledger-exhausted skip of the judge call, hyde_query event carrying the
# controller-authored hypothetical answer, and tried-query/HyDE bookkeeping.
# @end-summary
"""Tests for ``turn_loop/retrieve.py``."""

from __future__ import annotations

from src.retrieval.pipeline.turn_loop.events import TurnEventEmitter
from src.retrieval.pipeline.turn_loop.retrieve import run_retrieve
from src.retrieval.pipeline.turn_loop.schemas import (
    RetrieveArgs,
    TurnEventType,
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


def _setup(provider, batches):
    deps, emitted = make_deps(provider, retrieve_batches=batches)
    state, budget = TurnState(), make_budget()
    emitter = TurnEventEmitter(deps=deps, state=state, budget=budget, stream_events=True)
    return deps, emitted, state, budget, emitter


async def test_dedup_against_seen_chunk_ids_counts_dups():
    provider = FakeProvider(responses=[judge_json([0], 1)])
    batch = [make_chunk("c1"), make_chunk("c2")]
    deps, emitted, state, budget, emitter = _setup(provider, [batch])
    state.seen_chunk_ids.add("c1")  # served by an earlier action/round

    await run_retrieve(
        RetrieveArgs(query_text="q"),
        query="q",
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    assert [c.chunk_id for c in state.pool] == ["c2"]
    result = events_of(emitted, TurnEventType.RETRIEVE_RESULT)[0]
    assert result["added"] == 1
    assert result["dup"] == 1
    assert result["pool_size"] == 1


async def test_kept_chunks_enter_pool_in_judge_rank_order():
    """Judge keeps [2, 0] (best->worst): pool order follows the ranking."""
    provider = FakeProvider(responses=[judge_json([2, 0], 3)])
    batch = [make_chunk("c1"), make_chunk("c2"), make_chunk("c3")]
    deps, emitted, state, budget, emitter = _setup(provider, [batch])

    await run_retrieve(
        RetrieveArgs(query_text="q"),
        query="q",
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    assert [c.chunk_id for c in state.pool] == ["c3", "c1"]  # c2 dropped
    assert state.llm_calls == 1  # judge charged to the single ledger
    verdict = events_of(emitted, TurnEventType.JUDGE_VERDICT)[0]
    assert verdict["kept"] == 2
    assert verdict["confidence"] == 0.9
    # Dropped chunk is still marked seen — it will not be re-judged later.
    assert state.seen_chunk_ids == {"c1", "c2", "c3"}


async def test_judge_garbage_fails_open_keep_all_without_verdict_event():
    provider = FakeProvider(responses=["not json"])
    batch = [make_chunk("c1"), make_chunk("c2")]
    deps, emitted, state, budget, emitter = _setup(provider, [batch])

    await run_retrieve(
        RetrieveArgs(query_text="q"),
        query="q",
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    assert [c.chunk_id for c in state.pool] == ["c1", "c2"]
    # No fabricated verdict: the gate later reads a neutral judge component.
    assert events_of(emitted, TurnEventType.JUDGE_VERDICT) == []


async def test_exhausted_ledger_skips_judge_and_keeps_all():
    provider = FakeProvider(responses=["never used"])
    batch = [make_chunk("c1")]
    deps, emitted, state, budget, emitter = _setup(provider, [batch])
    state.llm_calls = budget.max_llm_calls

    await run_retrieve(
        RetrieveArgs(query_text="q"),
        query="q",
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    assert [c.chunk_id for c in state.pool] == ["c1"]
    assert provider.calls == []


async def test_hyde_event_carries_controller_authored_hypothetical():
    provider = FakeProvider(responses=[judge_json([0], 1)])
    deps, emitted, state, budget, emitter = _setup(provider, [[make_chunk("c1")]])

    await run_retrieve(
        RetrieveArgs(
            query_text="timing values",
            hypothetical_answer="The timing values are ...",
            target_aspect="signal timing",
        ),
        query="q",
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    hyde = events_of(emitted, TurnEventType.HYDE_QUERY)[0]
    assert hyde["hypothetical_answer"] == "The timing values are ..."
    assert hyde["search_terms"] == ["timing values"]
    assert hyde["target_aspect"] == "signal timing"
    assert state.tried_queries == ["timing values"]
    assert state.tried_hyde == ["The timing values are ..."]


async def test_no_hyde_event_without_hypothetical_answer():
    provider = FakeProvider(responses=[judge_json([0], 1)])
    deps, emitted, state, budget, emitter = _setup(provider, [[make_chunk("c1")]])

    await run_retrieve(
        RetrieveArgs(query_text="q2"),
        query="q",
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    assert events_of(emitted, TurnEventType.HYDE_QUERY) == []
    assert state.tried_hyde == []


async def test_retrieve_seam_failure_degrades_to_empty_round():
    provider = FakeProvider()

    async def broken_retrieve(query_text, hyde_text, top_k):
        raise RuntimeError("activity timeout")

    deps, emitted = make_deps(provider)
    deps.retrieve_ranked = broken_retrieve
    state, budget = TurnState(), make_budget()
    emitter = TurnEventEmitter(deps=deps, state=state, budget=budget, stream_events=True)

    await run_retrieve(
        RetrieveArgs(query_text="q"),
        query="q",
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    assert state.pool == []
    result = events_of(emitted, TurnEventType.RETRIEVE_RESULT)[0]
    assert result["added"] == 0
