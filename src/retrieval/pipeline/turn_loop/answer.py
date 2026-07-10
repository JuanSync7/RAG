# @summary
# ANSWER action for the turn loop: FILLS the generation pool toward
# fallback_pool_size with the best raw chunks when the judge kept too few (kept
# chunks first — stable citation indices), builds generation messages (canonical
# RAG system prompt + single leading system message carrying the turn context +
# citation-indexed evidence pool), streams the draft through the sync provider
# generate_stream pulled via asyncio.to_thread (draft events carry
# kind='reasoning'|'content' deltas live), self-scores the draft with
# turn_answer_selfscore.md, computes the weighted confidence gate
# (judge pool confidence / LLM self-score / deterministic citation coverage)
# against TurnBudget.answer_confidence_threshold, emits the gate event, and
# returns the draft plus GateFeedback (weakest component + unsupported claims
# + judge missing_information) for the controller loop-back.
# Exports: run_answer, citation_coverage
# Deps: asyncio, src.common.prompts, src.common.utils,
#       src.retrieval.pipeline.turn_loop.{schemas,events,controller}
#       (src.retrieval.generation.nodes.generator / config.settings lazily)
# @end-summary
"""ANSWER stage: draft generation + weighted confidence gate (design §5).

The gate deliberately does NOT reuse ``compute_composite_confidence`` — its
weights assume calibrated reranker scores, while this loop's pool carries raw
server-scale scores (design §5 confidence-semantics rule). Instead it blends
three loop-native components under configurable weights: the judge's latest
pool confidence (trace-sourced, neutral 0.5 when no verdict exists), the
draft's LLM self-score, and deterministic citation coverage (distinct ``[n]``
citations over a target-saturated denominator — pure token scanning, no content
pattern-matching). Because the pool is FILLED with context chunks the answer
need not all cite, coverage saturates at ``citation_target`` rather than dividing
by the full pool size.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from src.common.prompts import load_prompt, render, strip_reasoning
from src.common.utils import parse_json_object
from src.retrieval.pipeline.turn_loop.controller import (
    CONTEXT_DIGEST_MAX_CHARS,
    build_evidence_digest,
    judge_model_alias,
)
from src.retrieval.pipeline.turn_loop.events import (
    TurnEventEmitter,
    latest_event_payload,
)
from src.retrieval.pipeline.turn_loop.schemas import (
    GateFeedback,
    TurnBudget,
    TurnContext,
    TurnEventType,
    TurnLoopDeps,
    TurnState,
)

logger = logging.getLogger(__name__)

_SELFSCORE_PROMPT_FILE = "turn_answer_selfscore.md"

# Router alias for the draft call — the same alias the single-shot generation
# path streams with (server/routes/query.py `_stream_llm`), so loop answers
# come from the same model the rest of the product answers with.
_DRAFT_MODEL_ALIAS = "default"

# Neutral judge component when no judge verdict exists in the trace (e.g. the
# turn went straight to ANSWER, or every judge call failed open).
_NEUTRAL_JUDGE_CONFIDENCE = 0.5

# Sentinel handed to next() when pulling the sync stream from a worker thread.
_STREAM_DONE = object()


def _system_prompt() -> str:
    """The canonical RAG system prompt.

    Prefers the generator's cached loader (single source of truth with the
    non-loop path); falls back to loading ``prompts/rag_system.md`` directly
    when the generation package cannot be imported in this process.
    """
    try:
        from src.retrieval.generation.nodes.generator import get_system_prompt

        return get_system_prompt()
    except Exception:  # noqa: BLE001 — fall back to the shared prompt loader
        logger.warning(
            "generator system prompt unavailable — loading rag_system.md directly"
        )
        return load_prompt("rag_system.md").strip()


def _generation_params() -> tuple[float, int]:
    """Draft sampling parameters (the generation path's own tunables)."""
    from config import settings

    return (
        float(getattr(settings, "GENERATION_TEMPERATURE", 0.2)),
        int(getattr(settings, "GENERATION_MAX_TOKENS", 2048)),
    )


def _build_messages(
    query: str, context: TurnContext, state: TurnState
) -> list[dict]:
    """Assemble the draft chat messages.

    Single-system-message rule (generator.py precedent: the vLLM chat template
    allows exactly ONE system message and it must be first) — the turn-context
    digest is merged into the leading system message, never appended as a
    second one. The user message carries the evidence pool with 1-based
    ``[n]`` citation indexes matching the pool order the gate scores against.
    """
    system_content = _system_prompt()
    context_digest = context.render_for_prompt(CONTEXT_DIGEST_MAX_CHARS)
    if context_digest:
        system_content = (
            system_content
            + "\n\n--- Conversation context (supporting context for follow-up intent only) ---\n"
            + context_digest
        )
    evidence = "\n\n".join(
        f"[{index}] (source: {chunk.source} — {chunk.heading}) {chunk.text}"
        for index, chunk in enumerate(state.pool, start=1)
    ) or "(no evidence retrieved)"
    user_content = f"Context:\n{evidence}\n\nQuestion: {query}"
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


async def _stream_draft(
    messages: list[dict],
    *,
    attempt: int,
    state: TurnState,
    budget: TurnBudget,
    deps: TurnLoopDeps,
    emitter: TurnEventEmitter,
) -> str:
    """Stream one draft, forwarding every delta as a live ``draft`` event.

    ``provider.generate_stream`` is a synchronous generator (the provider
    module is not modified — design constraint), so each pull runs in a
    worker thread via ``asyncio.to_thread(next, ...)`` and the event loop
    stays free to flush SSE frames between tokens. Draft events carry
    ``{attempt, kind: 'reasoning'|'content', text_delta}``; only ``content``
    deltas accumulate into the returned draft. Charged as ONE ledger call.
    Fail-open: a mid-stream provider error returns whatever content was
    collected (a partial grounded draft beats an empty turn).
    """
    if state.llm_calls >= budget.max_llm_calls:
        logger.warning("turn loop LLM budget exhausted — skipping draft")
        return ""
    state.charge_llm_call()
    temperature, max_tokens = _generation_params()
    started = time.perf_counter()
    parts: list[str] = []
    try:
        stream = iter(
            deps.llm_provider.generate_stream(
                messages,
                model_alias=_DRAFT_MODEL_ALIAS,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=emitter.remaining_timeout_s(),
                include_reasoning=True,
            )
        )
        while True:
            item = await asyncio.to_thread(next, stream, _STREAM_DONE)
            if item is _STREAM_DONE:
                break
            kind, text = item
            if kind == "content":
                parts.append(text)
            await emitter.emit(
                TurnEventType.DRAFT,
                {"attempt": attempt, "kind": kind, "text_delta": text},
            )
    except Exception as exc:  # noqa: BLE001 — keep any partial draft
        logger.warning("turn loop draft stream failed: %s", exc)
    ms = int((time.perf_counter() - started) * 1000)
    await emitter.emit_llm_call(alias=_DRAFT_MODEL_ALIAS, purpose="draft", ms=ms)
    return strip_reasoning("".join(parts)).strip()


def citation_coverage(draft: str, pool_size: int, target: int = 5) -> float:
    """Grounding score from the draft's distinct citations — deterministic ``[n]`` scan.

    Structural token parsing over the draft with plain ``str`` methods (finds
    bracketed integers, validates them against the 1-based pool range) — this
    counts citation MARKERS, it never matches answer content (CLAUDE.md §0).

    The denominator SATURATES at ``target`` (``min(pool_size, target)``): a
    grounded answer cites a handful of distinct sources, it need not cite EVERY
    pooled chunk. Without this, filling the pool with context chunks (see
    ``run_answer``) would spuriously tank coverage — a valid refusal citing one
    source out of a 12-chunk filled pool would score 0.08 and fail the gate.
    ``target <= 0`` restores the raw ``/pool_size`` fraction.

    Args:
        draft: The drafted answer text.
        pool_size: Current evidence pool size (citation index upper bound).
        target: Distinct-source count at which coverage saturates to 1.0.

    Returns:
        ``len(distinct valid citations) / min(pool_size, target)`` clamped to
        ``[0, 1]``; ``0.0`` for an empty pool or draft.
    """
    if pool_size <= 0 or not draft:
        return 0.0
    denominator = min(pool_size, target) if target > 0 else pool_size
    cited: set[int] = set()
    position = 0
    while True:
        open_at = draft.find("[", position)
        if open_at == -1:
            break
        close_at = draft.find("]", open_at + 1)
        if close_at == -1:
            break
        token = draft[open_at + 1 : close_at].strip()
        if token.isdigit():
            index = int(token)
            if 1 <= index <= pool_size:
                cited.add(index)
            position = close_at + 1
        else:
            # Not a citation token — resume scanning INSIDE this bracket so a
            # nested "[see [2]]" still finds the inner citation.
            position = open_at + 1
    return min(1.0, len(cited) / denominator)


async def _self_score(
    query: str,
    draft: str,
    *,
    state: TurnState,
    emitter: TurnEventEmitter,
) -> tuple[float, list[str]]:
    """Self-score the draft's grounding via ``turn_answer_selfscore.md``.

    Returns ``(self_score, unsupported_claims)``. Fail-open to a neutral 0.5
    with no flagged claims when the call or parse fails — a dead judge model
    must not veto (or inflate) the gate on its own.
    """
    prompt = render(
        load_prompt(_SELFSCORE_PROMPT_FILE),
        user_query=query,
        draft_answer=draft,
        evidence_digest=build_evidence_digest(state),
    )
    response = await emitter.charged_call(
        alias=judge_model_alias(),
        purpose="self_score",
        prompt=prompt,
        temperature=0.0,
    )
    if response is None:
        return _NEUTRAL_JUDGE_CONFIDENCE, []
    payload = parse_json_object(
        strip_reasoning(getattr(response, "content", "") or "")
    )
    try:
        score = float(payload.get("self_score"))
    except (TypeError, ValueError):
        return _NEUTRAL_JUDGE_CONFIDENCE, []
    score = min(1.0, max(0.0, score))
    raw_claims = payload.get("unsupported_claims")
    claims = (
        [str(claim).strip() for claim in raw_claims if str(claim).strip()]
        if isinstance(raw_claims, list)
        else []
    )
    return score, claims


async def _seed_baseline_floor(
    query: str, state: TurnState, budget: TurnBudget, deps: TurnLoopDeps
) -> None:
    """Seed the PERMANENT raw-query grounding floor ONCE per turn.

    Retrieves the RAW user query (``hyde_text=None`` — no HyDE, no decomposition)
    via the shared ``retrieve_ranked`` seam and stores its top-k in
    ``state.baseline_floor``: the plain single-shot baseline the ANSWER pool
    always includes, so the loop can never end up with LESS than the plain query
    would have found (union, not replace). This is deliberately NOT
    ``fallback_chunks`` — on a RETRIEVE turn that holds only the HyDE-driven
    candidates, so it can (and on SMP-2 did) drift off the doc the plain query
    nailed. Fail-open: any error leaves the floor empty (never worse than the
    pre-fix behavior). Runs at most once per turn (``baseline_seeded`` guard),
    including across answer-attempt retries; ``baseline_floor_k`` 0 disables it.
    """
    if budget.baseline_floor_k <= 0 or state.baseline_seeded:
        return
    state.baseline_seeded = True
    try:
        chunks = await deps.retrieve_ranked(query, None, budget.baseline_floor_k)
        state.baseline_floor = list(chunks or [])
    except Exception as exc:  # noqa: BLE001 — fail open; the floor stays empty
        logger.warning("turn loop baseline-floor seed retrieval failed: %s", exc)
        state.baseline_floor = []


def _fill_generation_pool(state: TurnState, budget: TurnBudget) -> None:
    """Top the ANSWER generation pool up from the judge-independent raw floor
    (``state.fallback_chunks``, best-scored raw retrieved candidates). Kept
    chunks always stay FIRST so citation indices are stable and the judge's
    ranking is preserved; appended context chunks the answer need not all cite
    (citation_coverage saturates). Two mechanisms, both generic (pool-shape
    properties), never a query/content match (CLAUDE.md §0):

    1. PERMANENT baseline floor (``baseline_floor_k``): ensure up to K
       source-diverse chunks from the RAW-QUERY retrieval (``state.baseline_floor``)
       are present EVEN WHEN the judged pool is already FULL. This is the
       union-not-replace guarantee — a HyDE-drifted RETRIEVE round, a DECOMPOSE
       rewrite, or an over-strict judge can never DROP a document the plain query
       found (measured regression SMP-2: single-shot retrieved RTL_Coding.pdf@1
       and answered; the loop's HyDE rounds drifted to AMBA/protocol specs, the
       raw baseline never entered ``fallback_chunks`` — which on a RETRIEVE turn
       holds only the HyDE-driven candidates — so the loop answered "not in the
       context"). The floor is sourced from a dedicated raw-query retrieval
       (``_seed_baseline_floor``), NOT ``fallback_chunks``, precisely because the
       latter is not the plain-query baseline on a RETRIEVE turn. Each floor
       chunk must add a NEW source: a doc-level additivity guarantee.
    2. THIN fill (``fallback_pool_size``): when the judged pool is under target,
       top it up from ``fallback_chunks`` — source-diversity first (cross-document
       coverage), then by score — so a generator handed one strictly-judged chunk
       still has the surrounding context to disambiguate a near-miss or cover both
       sides of a comparison. Agentic-style (cf. RAG_AGENTIC_FINAL_MAX_CHUNKS).

    Mutates ``state.pool`` in place.
    """
    if budget.fallback_pool_size <= 0 and budget.baseline_floor_k <= 0:
        return
    if not state.fallback_chunks and not state.baseline_floor:
        return

    have = {chunk.chunk_id for chunk in state.pool}
    have_sources = {chunk.source for chunk in state.pool if chunk.source}
    before = len(state.pool)

    def _take(chunk) -> None:
        state.pool.append(chunk)
        have.add(chunk.chunk_id)
        if chunk.source:
            have_sources.add(chunk.source)

    # 1. Permanent baseline floor — additivity guarantee; runs even on a FULL
    #    pool. Source-diverse so it adds baseline DOCUMENTS the judged pool lacks.
    #    Sourced from the RAW-QUERY retrieval, not the (HyDE-drifted) fallback.
    floor_added = 0
    if budget.baseline_floor_k > 0:
        for chunk in state.baseline_floor:  # best-first by score
            if floor_added >= budget.baseline_floor_k:
                break
            if (
                chunk.source
                and chunk.source not in have_sources
                and chunk.chunk_id not in have
            ):
                _take(chunk)
                floor_added += 1

    # 2. Thin fill toward the target (only when still under it).
    candidates = [c for c in state.fallback_chunks if c.chunk_id not in have]
    if budget.fallback_pool_size > 0 and len(state.pool) < budget.fallback_pool_size:
        # Pass 1 — source DIVERSITY first: the best-scored chunk of each source
        # not yet pooled (cross-document coverage the "answered from one doc"
        # failure lacked).
        for chunk in candidates:
            if len(state.pool) >= budget.fallback_pool_size:
                break
            if (
                chunk.source
                and chunk.source not in have_sources
                and chunk.chunk_id not in have
            ):
                _take(chunk)
        # Pass 2 — top up any remaining slots by score (sources may repeat).
        for chunk in candidates:
            if len(state.pool) >= budget.fallback_pool_size:
                break
            if chunk.chunk_id not in have:
                _take(chunk)

    if len(state.pool) > before:
        logger.info(
            "turn loop filled generation pool %d -> %d (permanent baseline "
            "floor=%d, then thin-fill toward %d)",
            before, len(state.pool), floor_added, budget.fallback_pool_size,
        )


async def run_answer(
    *,
    query: str,
    context: TurnContext,
    state: TurnState,
    budget: TurnBudget,
    deps: TurnLoopDeps,
    emitter: TurnEventEmitter,
) -> tuple[str, GateFeedback]:
    """Run one gated ANSWER attempt: draft, self-score, gate.

    Increments ``state.answer_attempts``, streams the draft (live ``draft``
    events), then evaluates the weighted gate and emits the ``gate`` event —
    draft events always precede their gate event (design §8 ordering). The
    caller (orchestrator) accepts the draft on ``feedback.passed``, otherwise
    stores the feedback in ``state.last_gate`` and loops.

    Args:
        query: The user's turn query.
        context: The cross-turn context digest source.
        state: The turn's mutable state.
        budget: The frozen per-turn budget (threshold + gate weights).
        deps: Injected capabilities (provider, emit).
        emitter: The turn's event emitter / charged-call wrapper.

    Returns:
        ``(draft_text, gate_feedback)``. ``draft_text`` may be empty when the
        LLM budget was exhausted before drafting (feedback then fails with a
        zero self component).
    """
    attempt = state.answer_attempts + 1
    state.answer_attempts = attempt

    await _seed_baseline_floor(query, state, budget, deps)
    _fill_generation_pool(state, budget)

    messages = _build_messages(query, context, state)
    draft = await _stream_draft(
        messages,
        attempt=attempt,
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    if draft:
        self_score, unsupported_claims = await _self_score(
            query, draft, state=state, emitter=emitter
        )
    else:
        self_score, unsupported_claims = 0.0, []

    # The generation draft is grounded in the WHOLE pool, so the gate's judge
    # component must reflect the pool's best evidence — the maximum verdict
    # confidence across the turn's rounds. Reading only the latest verdict
    # let a dud final round (kept=0, confidence 0) poison a pool that earlier
    # rounds had judged strong (observed live). ``missing_information`` still
    # comes from the latest verdict: it is the freshest statement of the gap.
    latest = latest_event_payload(state, TurnEventType.JUDGE_VERDICT)
    judge_confidence = _NEUTRAL_JUDGE_CONFIDENCE
    missing_information: list[str] = []
    round_confidences: list[float] = []
    for event in state.events:
        if event.type != TurnEventType.JUDGE_VERDICT:
            continue
        try:
            round_confidences.append(
                min(1.0, max(0.0, float(event.payload.get("confidence"))))
            )
        except (TypeError, ValueError):
            continue
    if round_confidences:
        judge_confidence = max(round_confidences)
    if latest is not None:
        missing = str(latest.get("missing_information") or "").strip()
        if missing:
            missing_information.append(missing)

    coverage = citation_coverage(
        draft, len(state.pool), target=budget.citation_target
    )
    components = {
        "judge": judge_confidence,
        "self": self_score,
        "citation": coverage,
    }
    score = (
        budget.gate_weight_judge * judge_confidence
        + budget.gate_weight_self * self_score
        + budget.gate_weight_citation * coverage
    )
    # Weakest raw component (first wins ties — judge/self/citation order).
    weakest = min(components, key=components.get)
    passed = bool(draft) and score >= budget.answer_confidence_threshold

    await emitter.emit(
        TurnEventType.GATE,
        {
            "attempt": attempt,
            "score": round(score, 4),
            "threshold": budget.answer_confidence_threshold,
            "passed": passed,
            "weakest": weakest,
        },
    )
    feedback = GateFeedback(
        score=score,
        threshold=budget.answer_confidence_threshold,
        passed=passed,
        weakest_component=weakest,
        unsupported_claims=unsupported_claims,
        missing_information=missing_information,
    )
    return draft, feedback


__all__ = ["run_answer", "citation_coverage"]
