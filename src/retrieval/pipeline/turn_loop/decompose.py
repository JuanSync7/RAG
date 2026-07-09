# @summary
# DECOMPOSE action for the turn loop: one controller LLM call splits a broad /
# compound question into a few focused sub-queries, which are retrieved in
# PARALLEL via the injected retrieve_ranked seam (asyncio.gather — the latency
# contract: one concurrent wave, never N sequential rounds), merged + deduped
# against TurnState.seen_chunk_ids, judged once as a single round (reusing
# retrieve._judge_round), and added to the flat pool. This is the cheap
# decompose/fan-out demoted from deep_research — NOT its heavy report loop.
# Fail-open everywhere: a failed split falls back to a single sub-query (the
# question itself), a failed retrieval contributes nothing, and the stage never
# raises. Emits hyde_query (the decomposition) + retrieve_result + judge_verdict,
# reusing the existing event vocabulary (no new SSE event / frontend change).
# Exports: run_decompose
# Deps: src.retrieval.pipeline.turn_loop.{schemas,events,controller,retrieve};
#       src.common.{prompts,utils}; config.settings (lazily)
# @end-summary
"""DECOMPOSE stage: split -> PARALLEL fan-out -> merge -> single judge round."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.common.prompts import load_prompt, render, strip_reasoning
from src.common.utils import parse_json_object
from src.retrieval.pipeline.turn_loop.common import TOP_PREVIEW_COUNT
from src.retrieval.pipeline.turn_loop.controller import controller_model_alias
from src.retrieval.pipeline.turn_loop.events import TurnEventEmitter
from src.retrieval.pipeline.turn_loop.retrieve import _judge_round, _retain_fallback
from src.retrieval.pipeline.turn_loop.schemas import (
    DecomposeArgs,
    EvidenceChunk,
    TurnBudget,
    TurnEventType,
    TurnLoopDeps,
    TurnState,
)

logger = logging.getLogger(__name__)

_PROMPT_FILE = "turn_decompose.md"


def _max_subqueries() -> int:
    """Cap on sub-queries per DECOMPOSE (bounds the parallel fan-out width)."""
    from config import settings

    return max(2, int(getattr(settings, "RAG_TURN_LOOP_DECOMPOSE_MAX_SUBQUERIES", 4)))


def _coerce_subqueries(payload: Any, cap: int) -> list[str]:
    """Extract a clean, deduped, capped list of sub-question strings.

    Tolerates the ``{"sub_questions": [...]}`` shape this stage's prompt emits
    AND the deep-research ``{"topics": [{"questions": [...]}]}`` shape, so the
    same planner prompt could be swapped in. Pure dict/str ops (CLAUDE.md §0).
    """
    if not isinstance(payload, dict):
        return []
    raw = payload.get("sub_questions") or payload.get("questions") or []
    if not raw and isinstance(payload.get("topics"), list):
        raw = []
        for topic in payload["topics"]:
            if isinstance(topic, dict):
                raw.extend(topic.get("questions") or [])
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        q = str(item or "").strip()
        key = q.lower()
        if q and key not in seen:
            seen.add(key)
            out.append(q)
        if len(out) >= cap:
            break
    return out


async def _split(
    question: str,
    missing: str | None,
    *,
    state: TurnState,
    budget: TurnBudget,
    emitter: TurnEventEmitter,
) -> list[str]:
    """One controller LLM call → up to ``_max_subqueries()`` sub-queries.

    Fail-open: an exhausted ledger, provider error, or unparseable JSON yields
    ``[]`` (the caller falls back to the whole question as a single sub-query).
    """
    if state.llm_calls >= budget.max_llm_calls:
        return []
    try:
        prompt = render(
            load_prompt(_PROMPT_FILE),
            question=question,
            missing_information=missing or "(no specific gap named)",
            max_subqueries=str(_max_subqueries()),
        )
        response = await emitter.charged_call(
            alias=controller_model_alias(),
            purpose="decompose",
            prompt=prompt,
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001 — fail open to the single-query fallback
        logger.warning("turn loop decompose split failed: %s", exc)
        return []
    if response is None:
        return []
    payload = parse_json_object(strip_reasoning(getattr(response, "content", "") or ""))
    return _coerce_subqueries(payload, _max_subqueries())


async def run_decompose(
    args: DecomposeArgs,
    *,
    query: str,
    state: TurnState,
    budget: TurnBudget,
    deps: TurnLoopDeps,
    emitter: TurnEventEmitter,
) -> None:
    """Run one DECOMPOSE action: split → parallel retrieve → merge → judge → pool.

    Flow: split the (compound) question into a few sub-queries with ONE
    controller call; fire ``deps.retrieve_ranked`` for all of them CONCURRENTLY
    (``asyncio.gather`` — so wall-clock is one retrieval wave, not the sum);
    merge and dedup the returns against everything the turn has seen; judge the
    merged set once (reusing the RETRIEVE round judge); kept chunks enter the
    flat ``state.pool``. Two sequential LLM calls total (split + judge) — the
    cost model behind the latency GO in the unification blueprint.

    Args:
        args: The controller's decompose arguments (question + optional gap).
        query: The user's turn query (the judge's grounding question).
        state: The turn's mutable state (pool, dedup set, tried lists).
        budget: The frozen per-turn budget.
        deps: Injected capabilities (``retrieve_ranked``, provider, emit).
        emitter: The turn's event emitter / charged-call wrapper.
    """
    round_index = max(0, state.iteration - 1)
    round_display = round_index + 1
    question = args.question or query

    # 1. Split (one controller LLM call). Fail-open → the whole question as a
    #    single sub-query, so DECOMPOSE degrades to a plain retrieval, never a
    #    no-op.
    subqueries = await _split(
        question, args.missing_information, state=state, budget=budget, emitter=emitter
    )
    if not subqueries:
        subqueries = [question]

    state.tried_queries.extend(subqueries)
    # Surface the decomposition on the existing hyde_query event (the sub-queries
    # ARE this round's retrieval queries) — no new SSE event / frontend change.
    await emitter.emit(
        TurnEventType.HYDE_QUERY,
        {
            "round": round_display,
            "hypothetical_answer": "",
            "search_terms": list(subqueries),
            "target_aspect": "decomposition",
        },
    )

    # 2. PARALLEL fan-out — the latency contract. One concurrent wave of
    #    retrievals; a single leg failure contributes nothing (fail-open) but
    #    never aborts the others.
    #
    #    Additive anchor (RRF/union): the RAW turn query is retrieved as its own
    #    leg alongside the sub-queries, so the merged pool is the UNION of the
    #    raw-query hits and the rewritten sub-query hits. A DECOMPOSE rewrite can
    #    then only ADD candidates — never DROP a document the raw query matched
    #    (the query-transform variance the sub-queries-only fan-out caused: the
    #    rewrite steered retrieval off a perfectly-matched doc). Mirrors the
    #    agentic loop's HyDE asymmetry (BM25 stays anchored to the raw query).
    #    Skipped when the split degraded to the whole question (the anchor would
    #    duplicate the single leg) or when disabled via decompose_anchor_raw.
    anchor_raw = bool(
        budget.decompose_anchor_raw
        and query.strip()
        and query.strip().lower() not in {s.strip().lower() for s in subqueries}
    )
    fanout = ([query] if anchor_raw else []) + subqueries

    async def _one(leg_query: str) -> list[EvidenceChunk]:
        try:
            return await deps.retrieve_ranked(leg_query, None, budget.retrieve_top_k)
        except Exception as exc:  # noqa: BLE001 — one failed leg, not the round
            logger.warning(
                "turn loop decompose retrieve failed for %r: %s", leg_query, exc
            )
            return []

    all_legs = await asyncio.gather(*[_one(q) for q in fanout])
    # Sub-query legs (for per-facet attribution) are the fan-out minus the
    # anchor leg — the raw-query anchor is additive retrieval, not a facet.
    sub_legs = all_legs[1:] if anchor_raw else all_legs

    # Grounding floor: retain the RAW fan-out candidates (judge-INDEPENDENT),
    # symmetric with RETRIEVE (run_retrieve) — DECOMPOSE previously left the floor
    # empty, so a DECOMPOSE-driven turn whose judge kept a thin pool had nothing
    # to fill generation from (answered from one doc) and an empty-pool ANSWER had
    # no floor to ground on. Reuses the shared helper (CLAUDE.md §2).
    _retain_fallback(
        state,
        [chunk for leg in all_legs for chunk in leg],
        cap=budget.reservoir_size,
    )

    # 3. Merge + dedup — done AFTER the gather (single-threaded) so the shared
    #    seen-set is never mutated concurrently.
    fresh: list[EvidenceChunk] = []
    dup_count = 0
    for chunks in all_legs:
        for chunk in chunks:
            if chunk.chunk_id in state.seen_chunk_ids:
                dup_count += 1
                continue
            state.seen_chunk_ids.add(chunk.chunk_id)
            fresh.append(chunk)

    # 4. One judge round over the merged fresh set (reuse the RETRIEVE judge).
    kept, pool_verdict = await _judge_round(
        query=query,
        fresh=fresh,
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )
    for chunk in kept:
        chunk.round_added = round_index
        state.pool.append(chunk)

    # 5. Per-facet coverage (drives the facet-commit guard). Only a GENUINE
    #    multi-way split registers facets — the single-query fallback is a plain
    #    retrieval, not a decomposition. A facet is covered when its sub-query
    #    leg points at judge-KEPT evidence the turn HAS (pure id-set membership,
    #    CLAUDE.md §0 — no content matching). Two subtleties the naive
    #    "this round's kept only" attribution got wrong:
    #    - A fail-open keep-all round (pool_verdict is None: exhausted ledger,
    #      judge error, unparseable output) VALIDATED nothing — kept-set
    #      membership is then not evidence, so no facet may be marked covered on
    #      it (else the guard force-answers precisely when the judge is broken).
    #      The facets are still registered so a genuinely-judged retry can
    #      upgrade them (record_facet is monotonic).
    #    - Attribution runs against the turn's accumulated kept pool (state.pool,
    #      which already includes this round's keeps), NOT just this round's
    #      kept: cross-round dedup drops an already-pooled chunk from `fresh`, so
    #      a facet whose only hit was kept by an earlier round would otherwise be
    #      under-counted. Pooled ids only (never seen_chunk_ids, which also holds
    #      judge-DROPPED chunks that must not cover a facet).
    if budget.facet_commit_enabled and len(subqueries) >= 2:
        judged = pool_verdict is not None
        evidence_ids = {chunk.chunk_id for chunk in state.pool}
        for subq, leg in zip(subqueries, sub_legs):
            leg_ids = {chunk.chunk_id for chunk in leg}
            state.record_facet(subq, judged and bool(evidence_ids & leg_ids))

    await emitter.emit(
        TurnEventType.RETRIEVE_RESULT,
        {
            "round": round_display,
            "added": len(kept),
            "dup": dup_count,
            "pool_size": len(state.pool),
            "sub_queries": len(subqueries),
            "top": [
                {
                    "doc": chunk.source or chunk.document_id,
                    "heading": chunk.heading,
                    "score": chunk.score,
                }
                for chunk in kept[:TOP_PREVIEW_COUNT]
            ],
        },
    )
    if pool_verdict is not None:
        await emitter.emit(
            TurnEventType.JUDGE_VERDICT,
            {
                "round": round_display,
                "kept": len(kept),
                "sufficient": bool(pool_verdict.sufficient),
                "confidence": float(pool_verdict.confidence),
                "missing_information": str(pool_verdict.missing_information or ""),
            },
        )


__all__ = ["run_decompose"]
