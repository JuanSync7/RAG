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
    # These tests exercise sub-query fan-out / dedup / facet mechanics in
    # isolation; the additive raw-query anchor leg (a separate concern) is
    # covered by its own tests, so disable it here to keep the FIFO-batch
    # arithmetic 1:1 with the sub-queries.
    state, budget = TurnState(), make_budget(decompose_anchor_raw=False)
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
    # anchor off: assert one retrieve PER SUB-QUERY (the anchor leg is separate).
    state, budget = TurnState(), make_budget(decompose_anchor_raw=False)
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


# ── grounding floor (DECOMPOSE feeds fallback_chunks, symmetric with RETRIEVE) ─

async def test_decompose_retains_raw_candidates_in_grounding_floor():
    """DECOMPOSE must retain its RAW candidates in the judge-independent floor
    (``fallback_chunks``) — symmetric with RETRIEVE — so a thin judged pool or an
    empty-pool ANSWER can still ground on / fill from what DECOMPOSE retrieved.
    Both legs' candidates are retained even though the judge kept only one."""
    provider = FakeProvider(responses=[_split_json("a", "b"), judge_json([0], 2)])
    deps, ev, state, budget, emitter = _setup(
        provider, [[make_chunk("c1", score=0.9)], [make_chunk("c2", score=0.7)]]
    )

    await run_decompose(DecomposeArgs(question="q"), query="q", state=state,
                        budget=budget, deps=deps, emitter=emitter)

    assert [c.chunk_id for c in state.pool] == ["c1"]  # judge kept only c1
    floor_ids = {c.chunk_id for c in state.fallback_chunks}
    assert {"c1", "c2"} <= floor_ids  # BOTH raw candidates retained in the floor


# ── additive raw-query anchor leg (RRF/union — mode-D fix) ───────────────────

async def test_raw_query_anchor_recovers_a_doc_the_subqueries_drift_off():
    """Additive/RRF property (the SMP-3 mode-D regression): DECOMPOSE rewrites the
    query into sub-queries that drift off the doc the RAW query matched. The
    raw-query anchor leg re-includes that doc, so a rewrite can only ADD
    candidates, never DROP a raw-query hit."""
    provider = FakeProvider(
        responses=[_split_json("amba chi channel a", "amba chi channel b"),
                   judge_json([0, 1, 2], 3)],
    )
    raw_hit = make_chunk("pt_eco_flow", source="pt_eco_flow.pdf")

    async def retrieve_ranked(qt, hyde, k):
        # Only the RAW query surfaces the target doc; the drifted sub-queries
        # return unrelated CHI-noise chunks.
        if qt == "back-end ECO flow signoff order":
            return [raw_hit]
        return [make_chunk(f"chi-{qt[-1]}")]

    async def emit(e):  # noqa: ANN001
        pass

    deps = TurnLoopDeps(retrieve_ranked=retrieve_ranked, fetch_document=None,
                        llm_provider=provider, emit=emit)
    state, budget = TurnState(), make_budget()  # anchor ON by default
    emitter = TurnEventEmitter(deps=deps, state=state, budget=budget, stream_events=True)

    await run_decompose(
        DecomposeArgs(question="back-end ECO flow signoff order"),
        query="back-end ECO flow signoff order",
        state=state, budget=budget, deps=deps, emitter=emitter,
    )

    # The raw-query doc was RETRIEVED (anchor leg) and survived into the pool.
    assert "pt_eco_flow" in state.seen_chunk_ids  # anchor leg retrieved it
    assert "pt_eco_flow" in {c.chunk_id for c in state.pool}  # judge kept it


async def test_raw_query_anchor_leg_is_retrieved():
    """The fan-out issues a retrieval on the RAW query itself (the anchor leg),
    in addition to the sub-queries."""
    provider = FakeProvider(responses=[_split_json("sub one", "sub two"),
                                       judge_json([], 0)])
    seen_qts = []

    async def retrieve_ranked(qt, hyde, k):
        seen_qts.append(qt)
        return []

    async def emit(e):  # noqa: ANN001
        pass

    deps = TurnLoopDeps(retrieve_ranked=retrieve_ranked, fetch_document=None,
                        llm_provider=provider, emit=emit)
    state, budget = TurnState(), make_budget()
    emitter = TurnEventEmitter(deps=deps, state=state, budget=budget, stream_events=True)

    await run_decompose(DecomposeArgs(question="the raw question"),
                        query="the raw question", state=state, budget=budget,
                        deps=deps, emitter=emitter)

    assert "the raw question" in seen_qts  # anchor leg
    assert seen_qts.count("the raw question") == 1  # not duplicated
    assert {"sub one", "sub two"} <= set(seen_qts)  # sub-query legs too


async def test_raw_anchor_not_registered_as_a_facet():
    """The raw-query anchor leg is an additive retrieval, NOT a decomposed facet
    — only the sub-queries are facets (the anchor must not distort coverage)."""
    provider = FakeProvider(responses=[_split_json("a", "b"), judge_json([0, 1, 2], 3)])

    async def retrieve_ranked(qt, hyde, k):
        return [make_chunk(f"doc-{qt}")]

    async def emit(e):  # noqa: ANN001
        pass

    deps = TurnLoopDeps(retrieve_ranked=retrieve_ranked, fetch_document=None,
                        llm_provider=provider, emit=emit)
    state, budget = TurnState(), make_budget()
    emitter = TurnEventEmitter(deps=deps, state=state, budget=budget, stream_events=True)

    await run_decompose(DecomposeArgs(question="a and b"), query="a and b",
                        state=state, budget=budget, deps=deps, emitter=emitter)

    # Facets are the two sub-queries only — never the raw "a and b" anchor.
    assert [f.question for f in state.facets] == ["a", "b"]


async def test_raw_anchor_disabled_reverts_to_subqueries_only():
    provider = FakeProvider(responses=[_split_json("sub one", "sub two"),
                                       judge_json([], 0)])
    seen_qts = []

    async def retrieve_ranked(qt, hyde, k):
        seen_qts.append(qt)
        return []

    async def emit(e):  # noqa: ANN001
        pass

    deps = TurnLoopDeps(retrieve_ranked=retrieve_ranked, fetch_document=None,
                        llm_provider=provider, emit=emit)
    state, budget = TurnState(), make_budget(decompose_anchor_raw=False)
    emitter = TurnEventEmitter(deps=deps, state=state, budget=budget, stream_events=True)

    await run_decompose(DecomposeArgs(question="the raw question"),
                        query="the raw question", state=state, budget=budget,
                        deps=deps, emitter=emitter)

    assert "the raw question" not in seen_qts  # no anchor leg
    assert set(seen_qts) == {"sub one", "sub two"}


# ── facet coverage (drives the commit guard) ─────────────────────────────────

async def test_facets_recorded_with_partial_coverage():
    """A genuine 2-way split registers one facet per sub-query; a facet is
    covered only when a judge-KEPT chunk traces back to its leg. Here the judge
    keeps c1 (leg 'a') but not c2 (leg 'b') → 'a' covered, 'b' not."""
    provider = FakeProvider(responses=[_split_json("a", "b"), judge_json([0], 1)])
    batches = [[make_chunk("c1")], [make_chunk("c2")]]
    deps, ev, state, budget, emitter = _setup(provider, batches)

    await run_decompose(DecomposeArgs(question="a and b"), query="a and b",
                        state=state, budget=budget, deps=deps, emitter=emitter)

    assert [(f.question, f.covered) for f in state.facets] == [("a", True), ("b", False)]
    assert state.facets_fully_covered() is False


async def test_facets_all_covered_when_judge_keeps_every_leg():
    provider = FakeProvider(responses=[_split_json("a", "b"), judge_json([0, 1], 2)])
    batches = [[make_chunk("c1")], [make_chunk("c2")]]
    deps, ev, state, budget, emitter = _setup(provider, batches)

    await run_decompose(DecomposeArgs(question="a and b"), query="a and b",
                        state=state, budget=budget, deps=deps, emitter=emitter)

    assert state.facets_fully_covered() is True


async def test_chunk_shared_across_legs_covers_both_facets():
    """A chunk retrieved by two sub-queries is deduped into one pool entry, but
    it must still cover BOTH facets (coverage is per-leg raw-id membership, not
    the deduped pool)."""
    provider = FakeProvider(responses=[_split_json("a", "b"), judge_json([0], 1)])
    # Both legs return c1; dedup keeps one, the judge keeps it → covers a AND b.
    batches = [[make_chunk("c1")], [make_chunk("c1")]]
    deps, ev, state, budget, emitter = _setup(provider, batches)

    await run_decompose(DecomposeArgs(question="a and b"), query="a and b",
                        state=state, budget=budget, deps=deps, emitter=emitter)

    assert [f.covered for f in state.facets] == [True, True]


async def test_single_query_fallback_records_no_facets():
    """The split fail-open (one sub-query = the whole question) is a plain
    retrieval, not a decomposition — it must register NO facets so the commit
    guard does not hijack a degenerate DECOMPOSE."""
    provider = FakeProvider(responses=["not json", judge_json([0], 1)])
    deps, ev, state, budget, emitter = _setup(provider, [[make_chunk("c1")]])

    await run_decompose(DecomposeArgs(question="one question"), query="one question",
                        state=state, budget=budget, deps=deps, emitter=emitter)

    assert state.facets == []


async def test_facet_recording_disabled_by_budget_flag():
    provider = FakeProvider(responses=[_split_json("a", "b"), judge_json([0, 1], 2)])
    deps, ev = make_deps(provider, retrieve_batches=[[make_chunk("c1")], [make_chunk("c2")]])
    state, budget = TurnState(), make_budget(facet_commit_enabled=False)
    emitter = TurnEventEmitter(deps=deps, state=state, budget=budget, stream_events=True)

    await run_decompose(DecomposeArgs(question="a and b"), query="a and b",
                        state=state, budget=budget, deps=deps, emitter=emitter)

    assert state.facets == []  # guard disarmed → no bookkeeping overhead


async def test_facet_covered_by_chunk_pooled_in_an_earlier_round():
    """Cross-round dedup class: a facet whose only hit was judge-kept by an
    EARLIER round is deduped out of this round's `fresh`, so it never re-enters
    `kept`. Coverage must attribute against the accumulated pool, not just this
    round's keeps — else the richest (chunk-reused) pool falsely reads as
    uncovered and the guard never commits."""
    provider = FakeProvider(responses=[_split_json("a", "b"), judge_json([0], 1)])
    shared = make_chunk("cShared")
    # leg 'a' -> a fresh chunk the judge keeps; leg 'b' -> cShared, already pooled.
    deps, ev, state, budget, emitter = _setup(provider, [[make_chunk("cA")], [shared]])
    state.pool.append(shared)             # an earlier round already kept it
    state.seen_chunk_ids.add("cShared")   # so this round dedups it out of `fresh`

    await run_decompose(DecomposeArgs(question="a and b"), query="a and b",
                        state=state, budget=budget, deps=deps, emitter=emitter)

    assert [(f.question, f.covered) for f in state.facets] == [("a", True), ("b", True)]
    assert state.facets_fully_covered() is True


async def test_failopen_judge_covers_no_facet():
    """Judge fail-open (keep-all, pool_verdict None) validated nothing, so no
    facet may be marked covered even though every chunk is 'kept' — otherwise
    the guard would force-answer precisely when the judge is broken. Chunks are
    still pooled (grounding preserved); a genuinely-judged retry can upgrade
    coverage later (record_facet is monotonic)."""
    # Unparseable judge output → judge_chunks keeps all with pool_verdict None.
    provider = FakeProvider(responses=[_split_json("a", "b"), "not valid json"])
    deps, ev, state, budget, emitter = _setup(
        provider, [[make_chunk("c1")], [make_chunk("c2")]]
    )

    await run_decompose(DecomposeArgs(question="a and b"), query="a and b",
                        state=state, budget=budget, deps=deps, emitter=emitter)

    assert {c.chunk_id for c in state.pool} == {"c1", "c2"}  # grounding preserved
    assert [f.covered for f in state.facets] == [False, False]  # nothing validated
    assert state.facets_fully_covered() is False


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
