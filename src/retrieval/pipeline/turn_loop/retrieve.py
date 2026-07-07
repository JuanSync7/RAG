# @summary
# RETRIEVE action for the turn loop: one hybrid-search round via the injected
# retrieve_ranked seam using the controller-authored query (and its optional
# HyDE hypothetical answer — no separate generate_hyde call, one LLM call
# fewer), dedup against TurnState.seen_chunk_ids, then a round judge composed
# from the agentic judge_chunks primitive (charged to the turn's single LLM
# ledger, fail-open keep-all). Kept chunks enter the pool in judge-rank order;
# emits hyde_query / retrieve_result / judge_verdict events.
# Exports: run_retrieve
# Deps: src.retrieval.pipeline.turn_loop.{schemas,events,controller}
#       (src.retrieval.pipeline.agentic.judge + src.retrieval.common.schemas
#       lazily inside the judge step; config.settings lazily)
# @end-summary
"""RETRIEVE stage: hybrid-search round + dedup + judge composition.

The controller authors both the lexical query and (optionally) the HyDE
hypothetical answer inside its decision, so this stage makes at most ONE LLM
call — the round judge. The judge is the only place a retrieved chunk can be
dropped, and it judges generic relevance/faithfulness properties of the
``(question, chunk)`` pair, never a vendor/phrase match (CLAUDE.md §0).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from src.retrieval.pipeline.turn_loop.common import TOP_PREVIEW_COUNT
from src.retrieval.pipeline.turn_loop.controller import judge_model_alias
from src.retrieval.pipeline.turn_loop.events import TurnEventEmitter
from src.retrieval.pipeline.turn_loop.schemas import (
    EvidenceChunk,
    RetrieveArgs,
    TurnBudget,
    TurnEventType,
    TurnLoopDeps,
    TurnState,
)

logger = logging.getLogger(__name__)


def _retain_fallback(
    state: TurnState, retrieved: list[EvidenceChunk], *, cap: int
) -> None:
    """Retain the turn's best-scored RAW retrieved chunks as a grounding floor.

    Judge-INDEPENDENT by design: merges this round's raw candidates into
    ``state.fallback_chunks`` (dedup by chunk_id), keeps the top ``cap`` by
    score. Consumed only when the judged ``pool`` is empty — so a round judge
    that rejects an entire fresh batch can no longer strand the turn with
    nothing to cite (class fix, not a query/content match — CLAUDE.md §0).
    ``cap <= 0`` disables the floor (pre-fix behavior).
    """
    if cap <= 0 or not retrieved:
        return
    seen = {chunk.chunk_id for chunk in state.fallback_chunks}
    for chunk in retrieved:
        if chunk.chunk_id not in seen:
            state.fallback_chunks.append(chunk)
            seen.add(chunk.chunk_id)
    state.fallback_chunks.sort(key=lambda chunk: chunk.score, reverse=True)
    del state.fallback_chunks[cap:]


def _judge_json_mode() -> bool:
    """Guided-JSON toggle for the judge call (``RAG_AGENTIC_LLM_JSON_MODE``
    rationale, design §6: off by default because reasoning models emit empty
    output under a JSON constraint; free output is salvage-parsed instead)."""
    from config import settings

    return bool(getattr(settings, "RAG_AGENTIC_LLM_JSON_MODE", False))


def _judge_concise() -> bool:
    """Whether the round judge uses the agentic CONCISE judge (ranked id-list +
    sufficiency) rather than a scored object per chunk. Default true to match the
    agentic path — the shared 7B judge is better-calibrated and faster in concise
    mode (``RAG_TURN_LOOP_JUDGE_CONCISE``)."""
    from config import settings

    return bool(getattr(settings, "RAG_TURN_LOOP_JUDGE_CONCISE", True))


async def _judge_round(
    *,
    query: str,
    fresh: list[EvidenceChunk],
    state: TurnState,
    budget: TurnBudget,
    deps: TurnLoopDeps,
    emitter: TurnEventEmitter,
) -> tuple[list[EvidenceChunk], Optional[Any]]:
    """Judge this round's fresh chunks; return ``(kept_in_rank_order, pool_verdict)``.

    Composes the agentic ``judge_chunks`` primitive (design §4: consumed as a
    library) against the injected provider. The call is charged manually to
    the turn's single LLM ledger and surfaced as an ``llm_call`` event —
    ``judge_chunks`` talks to the provider directly, so the charged-call
    wrapper cannot intercept it. Fail-open on every failure path (exhausted
    ledger, provider error): keep all fresh chunks in retrieval order with no
    pool verdict, so a flaky judge never drops a round.
    """
    if not fresh:
        return [], None
    if state.llm_calls >= budget.max_llm_calls:
        logger.warning(
            "turn loop LLM budget exhausted — keeping round un-judged"
        )
        return list(fresh), None

    # Lazy imports: keep this module import-light (the judge pulls the
    # LiteLLM-backed provider module transitively).
    from src.retrieval.common.schemas import RankedResult
    from src.retrieval.pipeline.agentic.judge import judge_chunks

    candidates = [
        RankedResult(
            text=chunk.text,
            score=chunk.score,
            metadata={
                "source": chunk.source,
                "heading": chunk.heading,
                "document_id": chunk.document_id,
            },
        )
        for chunk in fresh
    ]
    alias = judge_model_alias()
    state.charge_llm_call()
    started = time.perf_counter()
    try:
        verdicts, pool_verdict = await judge_chunks(
            deps.llm_provider,
            model_alias=alias,
            original_question=query,
            candidates=candidates,
            timeout_s=emitter.remaining_timeout_s(),
            json_mode=_judge_json_mode(),
            concise=_judge_concise(),
        )
    except Exception as exc:  # noqa: BLE001 — fail open to keep-all
        logger.warning("turn loop round judge failed: %s — keeping all", exc)
        verdicts, pool_verdict = [], None
    ms = int((time.perf_counter() - started) * 1000)
    await emitter.emit_llm_call(alias=alias, purpose="judge", ms=ms)

    if not verdicts:
        return list(fresh), pool_verdict
    # Kept chunks in the judge's listwise-rank order (rank 0 = best); chunks
    # the judge kept but did not rank follow in retrieval order (fail-open).
    kept_pairs = [
        (verdict.rank if verdict.rank >= 0 else len(fresh) + verdict.index, verdict.index)
        for verdict in verdicts
        if verdict.keep and 0 <= verdict.index < len(fresh)
    ]
    kept_pairs.sort()
    kept = [fresh[index] for _, index in kept_pairs]
    # A judge that keeps chunks but reports no usable pool confidence (field
    # omitted / 0.0 — observed live) would pin the answer gate's judge
    # component at 0 and make the threshold unreachable. Derive the pool
    # confidence from its own per-chunk relevance judgments instead: any
    # unusable set-level score falls back to the mean kept relevance.
    if pool_verdict is not None and pool_verdict.confidence <= 0.0 and kept_pairs:
        kept_relevance = [
            verdict.relevance
            for verdict in verdicts
            if verdict.keep and 0 <= verdict.index < len(fresh)
        ]
        if kept_relevance and max(kept_relevance) > 0.0:
            pool_verdict.confidence = min(
                1.0, sum(kept_relevance) / len(kept_relevance)
            )
            logger.debug(
                "turn loop judge pool confidence derived from kept relevance: %.2f",
                pool_verdict.confidence,
            )
    return kept, pool_verdict


async def run_retrieve(
    args: RetrieveArgs,
    *,
    query: str,
    state: TurnState,
    budget: TurnBudget,
    deps: TurnLoopDeps,
    emitter: TurnEventEmitter,
) -> None:
    """Run one RETRIEVE action: search, dedup, judge, grow the pool.

    Flow (design §5): the controller-authored ``query_text`` (and optional
    ``hypothetical_answer``) go to ``deps.retrieve_ranked``; returned chunks
    are deduped against every chunk id the turn has already seen (kept OR
    previously judged-and-dropped — re-judging a dropped chunk would only burn
    budget); survivors are judged; kept chunks enter ``state.pool`` stamped
    with this round's index. The judge's ``missing_information`` reaches the
    controller through the ``judge_verdict`` event in the trace.

    Args:
        args: The controller's retrieval arguments.
        query: The user's turn query (the judge's grounding question).
        state: The turn's mutable state (pool, dedup set, tried lists).
        budget: The frozen per-turn budget.
        deps: Injected capabilities (``retrieve_ranked``, provider, emit).
        emitter: The turn's event emitter / charged-call wrapper.
    """
    # Pool round stamp stays 0-based (frozen EvidenceChunk.round_added
    # contract); event payloads carry the 1-based, human-facing round number
    # (consoles/CLI render it verbatim, matching turn_action.index).
    round_index = max(0, state.iteration - 1)
    round_display = round_index + 1
    query_text = args.query_text or query
    hyde_text = args.hypothetical_answer

    state.tried_queries.append(query_text)
    if hyde_text:
        state.tried_hyde.append(hyde_text)
        # The HyDE text was authored by the controller inside its decision —
        # surface it so the activity log shows what steered the embedding.
        await emitter.emit(
            TurnEventType.HYDE_QUERY,
            {
                "round": round_display,
                "hypothetical_answer": hyde_text,
                "search_terms": [query_text],
                "target_aspect": args.target_aspect or "",
            },
        )

    try:
        retrieved = await deps.retrieve_ranked(
            query_text, hyde_text, budget.retrieve_top_k
        )
    except Exception as exc:  # noqa: BLE001 — a failed round must not kill the turn
        logger.warning("turn loop retrieve_ranked failed: %s", exc)
        retrieved = []

    # Grounding floor: retain the best RAW candidates (before dedup/judge) so an
    # empty judged pool can still ground a draft/refusal (see _retain_fallback).
    _retain_fallback(state, retrieved, cap=budget.fallback_pool_size)

    fresh: list[EvidenceChunk] = []
    dup_count = 0
    for chunk in retrieved:
        if chunk.chunk_id in state.seen_chunk_ids:
            dup_count += 1
            continue
        state.seen_chunk_ids.add(chunk.chunk_id)
        fresh.append(chunk)

    if not fresh and dup_count and not state.pool:
        # Second chance (class: dropped-chunk dedup poisoning a whole turn).
        # Every retrieved candidate was judged-and-dropped in an earlier round
        # yet the pool is still EMPTY — one over-aggressive judge round must
        # not permanently starve the turn, so re-judge the returning
        # candidates instead of discarding the round. Pooled chunks can never
        # reach here (a non-empty pool skips this path), so this re-admits
        # only dropped ids; the ids stay in ``seen_chunk_ids``.
        fresh = list(retrieved)
        logger.info(
            "turn loop pool empty and all %d candidates previously dropped — "
            "re-judging them",
            dup_count,
        )

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

    await emitter.emit(
        TurnEventType.RETRIEVE_RESULT,
        {
            "round": round_display,
            "added": len(kept),
            "dup": dup_count,
            "pool_size": len(state.pool),
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
    # No judge_verdict event on judge fail-open (pool_verdict None): the gate
    # then reads a neutral judge component instead of a fabricated one.
    if pool_verdict is not None:
        await emitter.emit(
            TurnEventType.JUDGE_VERDICT,
            {
                "round": round_display,
                "kept": len(kept),
                "sufficient": bool(pool_verdict.sufficient),
                "confidence": float(pool_verdict.confidence),
                "missing_information": str(
                    pool_verdict.missing_information or ""
                ),
            },
        )


__all__ = ["run_retrieve"]
