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


async def test_pool_confidence_derived_from_kept_relevance_when_unusable():
    """A zero/omitted set-level confidence must not deaden the answer gate.

    Class: judge emits per-chunk keep/relevance but an unusable pool
    confidence (omitted -> clamped 0.0, observed live) — derive the pool
    confidence from the mean kept relevance instead of pinning the gate's
    judge component at 0.
    """
    import src.retrieval.pipeline.turn_loop.retrieve as retrieve_mod

    chunks = [make_chunk(i) for i in range(2)]
    provider = FakeProvider()
    deps, _ = make_deps(provider, retrieve_batches=[chunks])
    state, budget = TurnState(), make_budget()
    state.iteration = 1
    emitter = TurnEventEmitter(deps=deps, state=state, budget=budget, stream_events=False)

    class _Verdict:
        def __init__(self, index, relevance):
            self.index = index
            self.relevance = relevance
            self.faithfulness = 1.0
            self.keep = True
            self.rank = index

    class _Pool:
        sufficient = False
        confidence = 0.0
        missing_information = ""

    async def fake_judge_chunks(_provider, **_kwargs):
        return [_Verdict(0, 0.9), _Verdict(1, 0.7)], _Pool()

    import src.retrieval.pipeline.agentic.judge as judge_mod
    original = judge_mod.judge_chunks
    judge_mod.judge_chunks = fake_judge_chunks
    try:
        kept, pool_verdict = await retrieve_mod._judge_round(
            query="q", fresh=chunks, state=state, budget=budget, deps=deps,
            emitter=emitter,
        )
    finally:
        judge_mod.judge_chunks = original

    assert len(kept) == 2
    assert pool_verdict is not None
    assert abs(pool_verdict.confidence - 0.8) < 1e-6


async def test_pool_empty_all_dups_rejudges_dropped_chunks():
    """One over-aggressive judge round must not starve the whole turn.

    Class: dropped-chunk dedup poisoning — chunks judged-and-dropped earlier
    return as 'dups' forever while the pool stays empty (observed live: the
    right chunk was dropped in round 1 and blocked for the next 5 rounds).
    With an EMPTY pool and an all-dup round, the candidates are re-judged.
    """
    import src.retrieval.pipeline.agentic.judge as judge_mod
    import src.retrieval.pipeline.turn_loop.retrieve as retrieve_mod
    from src.retrieval.pipeline.turn_loop.schemas import RetrieveArgs

    chunks = [make_chunk(i) for i in range(2)]
    provider = FakeProvider()
    deps, _ = make_deps(provider, retrieve_batches=[chunks])
    state, budget = TurnState(), make_budget()
    state.iteration = 2
    # Simulate an earlier round that judged-and-dropped both chunks.
    for c in chunks:
        state.seen_chunk_ids.add(c.chunk_id)
    assert not state.pool
    emitter = TurnEventEmitter(deps=deps, state=state, budget=budget, stream_events=False)

    class _Verdict:
        def __init__(self, index):
            self.index = index
            self.relevance = 0.9
            self.faithfulness = 1.0
            self.keep = True
            self.rank = index

    async def fake_judge_chunks(_provider, **_kwargs):
        return [_Verdict(0), _Verdict(1)], None

    original = judge_mod.judge_chunks
    judge_mod.judge_chunks = fake_judge_chunks
    try:
        await retrieve_mod.run_retrieve(
            RetrieveArgs(query_text="q", hypothetical_answer=None, target_aspect=None),
            query="q", state=state, budget=budget, deps=deps, emitter=emitter,
        )
    finally:
        judge_mod.judge_chunks = original

    assert len(state.pool) == 2  # dropped chunks got their second judgment


async def test_judge_rejects_all_retains_fallback_floor_without_pooling():
    """A round judge that rejects an ENTIRE fresh batch keeps the judged pool
    empty (filtering preserved) but the best-scored RAW candidates are retained
    as the grounding floor — so a later ANSWER grounds a refusal instead of
    drafting from "(no evidence retrieved)". Keys on judged-pool-empty, never on
    query content (CLAUDE.md §0)."""
    provider = FakeProvider(
        responses=[judge_json([], 2, sufficient=False, confidence=0.2)]
    )
    batch = [make_chunk("c1", score=0.9), make_chunk("c2", score=0.7)]
    deps, emitted, state, budget, emitter = _setup(provider, [batch])

    await run_retrieve(
        RetrieveArgs(query_text="q"),
        query="q",
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    assert state.pool == []  # judge rejected everything — no pooling
    # Best-effort floor retained the raw candidates, highest score first.
    assert [c.chunk_id for c in state.fallback_chunks] == ["c1", "c2"]


async def test_round_judge_uses_concise_mode_by_default():
    """The round judge runs the agentic CONCISE judge by default — matching the
    agentic path (the shared 7B judge is better-calibrated + faster there). A
    regression that dropped the flag would revert to verbose scored-per-chunk."""
    import src.retrieval.pipeline.agentic.judge as judge_mod

    captured: dict = {}

    async def spy_judge_chunks(_provider, **kwargs):
        captured.update(kwargs)
        return [], None

    chunks = [make_chunk("c1")]
    provider = FakeProvider()
    deps, _ = make_deps(provider, retrieve_batches=[chunks])
    state, budget = TurnState(), make_budget()
    state.iteration = 1
    emitter = TurnEventEmitter(
        deps=deps, state=state, budget=budget, stream_events=False
    )
    original = judge_mod.judge_chunks
    judge_mod.judge_chunks = spy_judge_chunks
    try:
        await run_retrieve(
            RetrieveArgs(query_text="q"),
            query="q",
            state=state,
            budget=budget,
            deps=deps,
            emitter=emitter,
        )
    finally:
        judge_mod.judge_chunks = original

    assert captured.get("concise") is True


async def test_fallback_floor_disabled_with_zero_size():
    """``reservoir_size=0`` disables the grounding-floor retention (pre-fix
    behavior). The reservoir cap is now SEPARATE from the generation target
    (``fallback_pool_size``) — retention is governed by ``reservoir_size``."""
    provider = FakeProvider(
        responses=[judge_json([], 1, sufficient=False, confidence=0.2)]
    )
    deps, emitted = make_deps(provider, retrieve_batches=[[make_chunk("c1")]])
    state = TurnState()
    budget = make_budget(reservoir_size=0)
    emitter = TurnEventEmitter(
        deps=deps, state=state, budget=budget, stream_events=True
    )

    await run_retrieve(
        RetrieveArgs(query_text="q"),
        query="q",
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    assert state.pool == []
    assert state.fallback_chunks == []


def test_retain_fallback_uses_wide_reservoir_cap():
    """The reservoir retention cap (``reservoir_size``, wide) is SEPARATE from the
    generation target (``fallback_pool_size``). A wide reservoir is what lets the
    source-diverse ANSWER fill reach the OTHER documents a multi-part / cross-doc
    question needs — the v4 breadth deficit was an 8-chunk cap that starved the
    fill of distinct docs (measured: turn_loop's pool missed the sub-part docs
    single-round covered). Retention keeps ALL distinct docs up to the wide cap."""
    from src.retrieval.pipeline.turn_loop.retrieve import _retain_fallback

    state = TurnState()
    chunks = [
        make_chunk(f"c{i}", document_id=f"D{i}", source=f"Doc{i}", score=1.0 - i * 0.01)
        for i in range(20)
    ]
    _retain_fallback(state, chunks, cap=40)  # reservoir_size default

    # All 20 retained (below the 40 cap); the pre-split 8-cap would have dropped 12.
    assert len(state.fallback_chunks) == 20
    assert len({c.source for c in state.fallback_chunks}) == 20  # breadth preserved
