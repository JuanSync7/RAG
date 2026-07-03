# @summary
# API-process assembly for the turn-level agentic conversation loop: resolves
# the per-request/env enable flag (yielding to explicitly-requested competing
# orchestrators and fail-fast-validating the RAG_TURN_LOOP_* config on
# activation), runs the pre-loop input sanitize + PII gate (the same Stage-1
# modules the pipeline uses), builds TurnLoopDeps (Temporal
# TurnRetrieveWorkflow bridge -> EvidenceChunk with remaining-wall-clock
# timeouts, conversation-cached MinIO clean-document fetch via
# asyncio.to_thread, LLMProvider, asyncio.Queue event sink), drives
# run_turn_loop as a task while draining typed events for SSE, applies the
# configured output rails to the terminal answer, and hosts the pure
# result->payload helpers (source refs, chunk refs, action records,
# studied-doc entries, reasoning extraction, answer token replay) consumed by
# the query routes.
# Exports: RESULT_KIND, STOP_REASON_INPUT_REJECTED, STOP_REASON_ERROR,
#          PreparedTurnInput, TurnStreamOutcome, resolve_turn_loop_enabled,
#          prepare_turn_input, build_turn_context, evidence_from_dicts,
#          build_retrieve_ranked, build_fetch_document, run_turn_stream,
#          validate_event_payload, turn_loop_metadata, trace_records,
#          pool_chunk_results, pool_source_refs, build_chunk_refs,
#          build_action_records, build_studied_entries,
#          extract_draft_reasoning, iter_answer_tokens
# Deps: asyncio, dataclasses, datetime, time, server.schemas,
#       src.retrieval.pipeline.turn_loop (config.settings, fastapi,
#       server.workflows, src.db, src.guardrails, src.platform.llm lazily)
# @end-summary
"""API-process runner for the turn-level conversation loop.

The loop itself (``run_turn_loop``) is pure control flow over the injected
``TurnLoopDeps`` seam; this module is the *endpoint-side* injector
(TURN_LOOP_DESIGN.md §3): it wires the real Temporal retrieval activity, the
MinIO-backed document fetch, the platform LLM provider and an
``asyncio.Queue``-bridged event sink into that seam, and exposes one async
generator — :func:`run_turn_stream` — that the query routes consume for both
the SSE and the non-stream paths.

Input safety runs ONCE here, before the loop starts, reusing the exact
modules pipeline Stage 1 uses: the query sanitizer node (empty/injection
class) plus the guardrails PII gate and input rails when a guardrail backend
is configured. The pre-retrieval LangGraph reformulate loop is deliberately
NOT run — the loop controller owns query design (design §3).

Every helper that maps a ``TurnLoopResult`` into response payloads lives here
(not in the route) so both ``/query`` and ``/query/stream`` share one source
of truth and stay thin.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, AsyncIterator, Callable, Iterator, Optional

from src.retrieval.pipeline.turn_loop import (
    ClarificationOut,
    EvidenceChunk,
    TurnBudget,
    TurnContext,
    TurnEvent,
    TurnEventType,
    TurnLoopDeps,
    TurnLoopResult,
)

logger = logging.getLogger(__name__)

RESULT_KIND = "result"
"""Terminal yield kind of :func:`run_turn_stream` (never a TurnEventType)."""

STOP_REASON_INPUT_REJECTED = "input_rejected"
"""Stop reason when the pre-loop sanitize/PII gate rejects the turn."""

STOP_REASON_ERROR = "error"
"""Stop reason when the loop task raised instead of returning a result."""

_QUEUE_SENTINEL: Any = object()

_ERROR_CLARIFICATION = (
    "The assistant hit an internal error while working on this question. "
    "Please try again."
)


# ---------------------------------------------------------------------------
# Flag resolution + pre-loop input safety
# ---------------------------------------------------------------------------


def _competing_orchestrator(request: Any) -> Optional[str]:
    """Name the competing per-request orchestrator/mode, or ``None``.

    The class this guards (design §5 "config-default bypass"): the request
    schema validator only fires when ``turn_loop is True``, so an env default
    ``RAG_TURN_LOOP_ENABLED=true`` could otherwise silently hijack a request
    that *explicitly* asked for a different orchestrator (deep research,
    agentic, tree) or for retrieval-only mode (``skip_generation``, which the
    loop cannot honor). Explicit request flags outrank the env default —
    mirroring the design's ``_agentic_active = ... and not _turn_active``
    priority rule in the opposite direction.
    """
    if request is None:
        return None
    if bool(getattr(request, "deep_research", False)):
        return "deep_research"
    if getattr(request, "agentic_retrieval", None) is True:
        return "agentic_retrieval"
    if getattr(request, "tree_retrieval", None) is True:
        return "tree_retrieval"
    if getattr(request, "mode", None) == "retrieval":
        return "mode=retrieval"
    return None


def _validate_turn_loop_config_or_500() -> None:
    """Fail fast on contradictory ``RAG_TURN_LOOP_*`` config at activation.

    Mirrors the agentic precedent (``validate_agentic_retrieval_config()`` at
    ``rag_chain.py`` when that loop activates): the validator runs the first
    time a request actually takes the loop path, so importing settings with a
    broken env never raises but no loop turn can run on silently-misbehaving
    config (CLAUDE.md §3 fail-fast). A ``ValueError`` is converted to a clear
    HTTP 500 naming the offending key(s).
    """
    from config import settings

    try:
        settings.validate_turn_loop_config()
    except ValueError as exc:
        logger.error("turn loop configuration invalid: %s", exc)
        from fastapi import HTTPException

        raise HTTPException(
            status_code=500,
            detail=f"Turn-loop configuration invalid: {exc}",
        ) from exc


def resolve_turn_loop_enabled(
    request_override: Optional[bool], *, request: Any = None
) -> bool:
    """Resolve the effective turn-loop flag for one request.

    The per-request override wins when present; otherwise the
    ``RAG_TURN_LOOP_ENABLED`` env default applies (same precedence contract as
    ``agentic_retrieval``) — EXCEPT when the request explicitly enables a
    competing orchestrator (``deep_research`` / ``agentic_retrieval`` /
    ``tree_retrieval``) or retrieval-only mode: an env default must never
    hijack an explicit request (design §5 mutual exclusion, config-default
    bypass included). Whenever the resolution is True the ``RAG_TURN_LOOP_*``
    config is validated fail-fast (:func:`_validate_turn_loop_config_or_500`).
    Reads settings lazily so tests can monkeypatch the env default.

    Args:
        request_override: ``QueryRequest.turn_loop`` (tri-state).
        request: The originating request model (competing-flag inspection);
            ``None`` skips the precedence check (unit callers).

    Returns:
        True when this request should take the turn-loop path.
    """
    if request_override is not None:
        enabled = bool(request_override)
        if enabled:
            _validate_turn_loop_config_or_500()
        return enabled
    from config import settings

    if not bool(getattr(settings, "RAG_TURN_LOOP_ENABLED", False)):
        return False
    competing = _competing_orchestrator(request)
    if competing is not None:
        logger.info(
            "turn loop env default yielded to the explicitly requested "
            "%s for this request (design §5 precedence)",
            competing,
        )
        return False
    _validate_turn_loop_config_or_500()
    return True


@dataclass
class PreparedTurnInput:
    """Outcome of the once-per-turn input sanitize + PII gate."""

    query: str
    """The effective loop query (sanitized; PII-redacted when detected)."""

    rejected: bool = False
    """True when input safety terminated the turn before the loop started."""

    rejection_message: str = ""
    """Clarification text shown to the user on rejection."""

    pii_redactions: int = 0
    """Number of PII findings redacted from the query."""

    rails_degraded: bool = False
    """True when the guardrail backend was configured but unusable in the
    API process (logged fallback: sanitized-only, design §3)."""


def prepare_turn_input(query: str, tenant_id: str) -> PreparedTurnInput:
    """Run the Stage-1 input safety gates once for a loop turn (sync).

    Reuses the exact modules the single-shot pipeline uses so the loop path
    accepts/rejects the same input classes:

    1. The query sanitizer node (empty / prompt-injection class) from
       ``src.retrieval.query.nodes.query_processor``.
    2. When a guardrail backend is configured: the PII gate (``redact_pii``)
       and the input rails merged through the same ``RailMergeGate`` priority
       order ``rag_chain`` applies (injection > toxicity > topic > intent >
       PII-rewrite).

    Fail-open on guardrail *infrastructure* errors (the backend may be
    worker-bound, e.g. a local PII model): the turn proceeds with the
    sanitized query only and the degradation is logged + flagged.

    Args:
        query: Raw user query text.
        tenant_id: Tenant for the input rails.

    Returns:
        The prepared input (or a rejection carrying the clarification text).
    """
    from src.retrieval.query.nodes.query_processor import sanitize_node

    verdict = sanitize_node({"current_query": query})
    if verdict.get("action") == "ask_user":
        return PreparedTurnInput(
            query=query,
            rejected=True,
            rejection_message=str(verdict.get("clarification_message") or ""),
        )
    effective = str(verdict.get("current_query") or query)

    from config import settings

    if not getattr(settings, "GUARDRAIL_BACKEND", ""):
        return PreparedTurnInput(query=effective)

    try:
        from types import SimpleNamespace

        from src.guardrails import RailMergeGate, redact_pii, run_input_rails

        pii_count = 0
        try:
            redacted, detections = redact_pii(effective)
            pii_count = len(detections or [])
            if pii_count:
                effective = redacted
                logger.info(
                    "turn loop PII gate: %d detections redacted before the loop",
                    pii_count,
                )
        except Exception as exc:  # noqa: BLE001 — same fail-open as Stage 1
            logger.warning(
                "turn loop PII gate failed: %s — continuing with original query",
                exc,
            )
        rails = run_input_rails(query, tenant_id or "")
        merge = RailMergeGate().merge(
            SimpleNamespace(processed_query=effective), rails
        )
        if merge.get("action") in ("reject", "canned"):
            return PreparedTurnInput(
                query=effective,
                rejected=True,
                rejection_message=str(merge.get("message") or ""),
                pii_redactions=pii_count,
            )
        return PreparedTurnInput(
            query=str(merge.get("query") or effective),
            pii_redactions=pii_count,
        )
    except Exception as exc:  # noqa: BLE001 — degrade, never kill the turn
        logger.warning(
            "turn loop guardrails unavailable in the API process (%s) — "
            "continuing with the sanitized query only",
            exc,
        )
        return PreparedTurnInput(query=effective, rails_degraded=True)


# ---------------------------------------------------------------------------
# TurnContext + dependency builders
# ---------------------------------------------------------------------------


def build_turn_context(conversation_id: str, structured: Optional[dict]) -> TurnContext:
    """Assemble the typed cross-turn context from memory's structured dict.

    Thin delegation to the loop package's ``build_turn_context`` — the ONE
    tolerant bridge from the ``MemoryContext.structured`` contract (design §7)
    to the typed :class:`TurnContext` (it aliases the memory layer's
    ``question`` recent-turn key onto the frozen ``query`` contract and caps
    ``chunk_refs`` to the most recent ``RAG_TURN_CONTEXT_MAX_CHUNK_REFS``).
    Every field falls back to its empty default so pre-migration
    conversations (or memory-disabled requests, ``structured={}``) produce a
    minimal but valid context.

    Args:
        conversation_id: Owning conversation id.
        structured: ``MemoryContext.structured`` (may be None/empty).

    Returns:
        The :class:`TurnContext` handed to ``run_turn_loop``.
    """
    from src.retrieval.pipeline.turn_loop import build_turn_context as _build

    return _build(conversation_id, structured if isinstance(structured, dict) else None)


def _coerce_int(value: Any, default: int) -> int:
    """Best-effort int coercion preserving legitimate zeros."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float coercion."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def evidence_from_dicts(chunks: Any) -> list[EvidenceChunk]:
    """Map ``TurnRetrieveWorkflow`` result dicts into :class:`EvidenceChunk`.

    The worker returns EvidenceChunk-shaped dicts; provenance/round default
    here (``retrieve``/0) and are re-stamped by the loop's RETRIEVE stage,
    which owns round accounting. Non-dict entries are dropped.

    Args:
        chunks: The ``chunks`` list from the workflow result.

    Returns:
        Typed evidence chunks in result order.
    """
    out: list[EvidenceChunk] = []
    for raw in chunks or []:
        if not isinstance(raw, dict):
            continue
        out.append(
            EvidenceChunk(
                chunk_id=str(raw.get("chunk_id") or ""),
                document_id=str(raw.get("document_id") or ""),
                source_key=str(raw.get("source_key") or ""),
                source=str(raw.get("source") or ""),
                heading=str(raw.get("heading") or ""),
                text=str(raw.get("text") or ""),
                score=_coerce_float(raw.get("score")),
                refactored_char_start=_coerce_int(raw.get("refactored_char_start"), -1),
                refactored_char_end=_coerce_int(raw.get("refactored_char_end"), -1),
                provenance=str(raw.get("provenance") or EvidenceChunk.PROVENANCE_RETRIEVE),
                round_added=_coerce_int(raw.get("round_added"), 0),
            )
        )
    return out


# Floor for the per-retrieve timeout so a retrieve dispatched at the very end
# of the turn budget is still schedulable (matches the workflow's 1s rounding).
_RETRIEVE_MIN_TIMEOUT_MS = 1000


def build_retrieve_ranked(temporal_client: Any, request: Any) -> Callable:
    """Build the ``retrieve_ranked`` dependency over the Temporal client.

    Each call executes one ``TurnRetrieveWorkflow`` (embed -> hybrid search ->
    rerank -> thin/nav filter; no LLM calls inside, idempotent under retries)
    and maps the result to evidence chunks. The workflow/task-queue symbols
    are resolved lazily from ``server.workflows`` so this module imports
    cleanly while the worker track lands.

    Budget hierarchy (design §3): every call carries the turn's REMAINING
    wall-clock budget — never the full ``RAG_TURN_LOOP_WALL_CLOCK_MS`` — as
    ``timeout_ms`` (the activity start/schedule-to-close bound), as the
    workflow ``execution_timeout`` (so a down/backlogged worker cannot park
    the request forever waiting for a workflow task pickup), and as a
    client-side ``asyncio.wait_for`` (the hard guarantee that no single
    retrieve can extend the turn past its wall clock). The injector clock
    starts when the deps are built, i.e. at turn start.

    Args:
        temporal_client: Connected Temporal client.
        request: The originating ``QueryRequest`` (source/heading filters).

    Returns:
        An async ``retrieve_ranked(query_text, hyde_text, top_k)`` callable.
    """
    turn_started = time.monotonic()

    async def retrieve_ranked(
        query_text: str, hyde_text: Optional[str], top_k: int
    ) -> list[EvidenceChunk]:
        import server.workflows as workflows_mod
        from config import settings

        workflow = workflows_mod.TurnRetrieveWorkflow
        task_queue = getattr(
            workflows_mod, "TURN_RETRIEVE_TASK_QUEUE", workflows_mod.RAG_QUERY_TASK_QUEUE
        )
        wall_clock_ms = int(settings.RAG_TURN_LOOP_WALL_CLOCK_MS)
        elapsed_ms = int((time.monotonic() - turn_started) * 1000)
        # Remaining turn budget: strictly below the full wall clock, floored
        # so a last-moment retrieve is still schedulable. The loop's own
        # between-action wall-clock check then ends the turn.
        remaining_ms = max(
            _RETRIEVE_MIN_TIMEOUT_MS,
            min(wall_clock_ms - 1, wall_clock_ms - elapsed_ms),
        )
        payload = {
            "query_text": query_text,
            "hyde_text": hyde_text,
            "top_k": int(top_k),
            "source_filter": request.source_filter,
            "heading_filter": request.heading_filter,
            # Per-retrieve activity bound = remaining turn budget (< wall
            # clock, design §3 budget hierarchy).
            "timeout_ms": remaining_ms,
        }
        result = await asyncio.wait_for(
            temporal_client.execute_workflow(
                workflow.run,
                payload,
                id=f"turn-retrieve-{uuid.uuid4().hex[:12]}",
                task_queue=task_queue,
                execution_timeout=timedelta(milliseconds=remaining_ms),
            ),
            timeout=remaining_ms / 1000,
        )
        return evidence_from_dicts((result or {}).get("chunks"))

    return retrieve_ranked


# Conversation-scoped LRU of resolved clean documents, keyed by
# (conversation_id, document_id, source_key). Implements the design §5 /
# TurnLoopDeps contract that the injector caches DEEP_STUDY fetches per
# conversation: resuming a study of the same document (across actions AND
# across turns of one conversation) costs one MinIO round-trip. Bounded so a
# long-lived API process cannot grow it without limit (structural cap, not a
# behavior tunable); only successful resolutions are cached so misses retry.
_DOC_FETCH_CACHE: "OrderedDict[tuple[str, str, str], Any]" = OrderedDict()
_DOC_FETCH_CACHE_MAX_ENTRIES = 32


def build_fetch_document(db_client: Any, conversation_id: str = "") -> Callable:
    """Build the ``fetch_document`` dependency over the document store client.

    Wraps the sync ``resolve_clean_document`` in ``asyncio.to_thread``,
    caches successful resolutions per conversation (``_DOC_FETCH_CACHE``,
    the "conversation-cached by the injector" ``FetchDocumentFn`` contract)
    and fails open to ``None`` (missing client, unresolvable document,
    transport error) — the loop's DEEP_STUDY stage falls back instead of
    raising.

    Args:
        db_client: MinIO-backed document store client (may be None when the
            API booted without one).
        conversation_id: Owning conversation id (cache scope); empty scopes
            the cache entries to the anonymous conversation.

    Returns:
        An async ``fetch_document(document_id, source_key)`` callable.
    """

    async def fetch_document(
        document_id: Optional[str], source_key: Optional[str]
    ) -> Optional[object]:
        if db_client is None:
            logger.warning(
                "turn loop fetch_document: document store client unavailable — "
                "DEEP_STUDY degraded to retrieval-only evidence"
            )
            return None
        cache_key = (conversation_id, document_id or "", source_key or "")
        cached = _DOC_FETCH_CACHE.get(cache_key)
        if cached is not None:
            _DOC_FETCH_CACHE.move_to_end(cache_key)
            return cached
        try:
            from src.db import resolve_clean_document
        except ImportError as exc:
            logger.warning(
                "turn loop fetch_document: resolve_clean_document unavailable "
                "(%s) — DEEP_STUDY degraded", exc,
            )
            return None
        try:
            document = await asyncio.to_thread(
                resolve_clean_document,
                db_client,
                document_id=document_id,
                source_key=source_key,
            )
        except Exception as exc:  # noqa: BLE001 — loop falls back, never crashes
            logger.warning(
                "turn loop fetch_document failed for document_id=%s source_key=%s: %s",
                document_id,
                source_key,
                exc,
            )
            return None
        if document is not None:
            _DOC_FETCH_CACHE[cache_key] = document
            _DOC_FETCH_CACHE.move_to_end(cache_key)
            while len(_DOC_FETCH_CACHE) > _DOC_FETCH_CACHE_MAX_ENTRIES:
                _DOC_FETCH_CACHE.popitem(last=False)
        return document

    return fetch_document


def _get_run_turn_loop() -> Callable:
    """Resolve the loop entry point lazily (test seam + phased delivery)."""
    from src.retrieval.pipeline.turn_loop import run_turn_loop

    return run_turn_loop


def guardrail_backend_configured() -> bool:
    """True when an output-guardrail backend is configured (lazy read).

    Public so the stream route can gate the reasoning replay: the accepted
    answer is railed, but the captured chain-of-thought is not, so on a railed
    deployment the reasoning frame must be suppressed rather than replayed
    verbatim (parity with the live-draft suppression).
    """
    from config import settings

    return bool(getattr(settings, "GUARDRAIL_BACKEND", ""))


# Backward-compatible private alias (internal callers).
_guardrail_backend_configured = guardrail_backend_configured


async def _apply_output_rails(result: TurnLoopResult) -> TurnLoopResult:
    """Run the configured output rails over the loop's terminal answer.

    REFUSE-parity with the non-loop ``/query`` path (rag_chain Stage 7): when
    ``GUARDRAIL_BACKEND`` is configured, the accepted answer AND the
    budget-exhausted best-effort answer pass through ``run_output_rails``
    before replay/response; a rewritten/blocked ``final_answer`` replaces the
    draft. Fail-open on rail *infrastructure* errors, mirroring rag_chain: the
    un-railed answer is returned and the degradation logged — a broken rail
    backend must not kill the turn.
    """
    if not _guardrail_backend_configured():
        return result
    context_chunks = [chunk.text for chunk in result.pool]
    try:
        from src.guardrails import run_output_rails

        async def _rail(text: str) -> str:
            """Rail one string; return the rewritten text or the original."""
            if not (text or "").strip():
                return text
            rail_result = await asyncio.to_thread(
                run_output_rails, answer=text, context_chunks=context_chunks
            )
            final = getattr(rail_result, "final_answer", "") or ""
            return final if final else text

        answer = (result.answer or "").strip()
        if answer:
            railed = await _rail(result.answer)
            if railed != result.answer:
                logger.info("turn loop output rails replaced the terminal answer")
                result.answer = railed
        # ask_user terminals carry no answer but DO carry LLM-authored,
        # corpus-derived text (question / hints / scoping questions) that must
        # be railed too — otherwise a railed deployment ships un-railed model
        # output (and console resubmit chips) on every CLARIFY terminal.
        clar = result.clarification
        if clar is not None:
            clar.question = await _rail(clar.question)
            clar.hints = [await _rail(h) for h in clar.hints]
            clar.scoping_questions = [
                await _rail(q) for q in clar.scoping_questions
            ]
    except Exception as exc:  # noqa: BLE001 — fail open, Stage-7 parity
        logger.warning(
            "turn loop output rails unavailable (%s) — returning the "
            "un-railed answer (fail-open, rag_chain Stage-7 parity)",
            exc,
        )
    return result


def _get_llm_provider() -> Any:
    """Resolve the platform LLM provider lazily (test seam)."""
    from src.platform.llm import get_llm_provider

    return get_llm_provider()


# ---------------------------------------------------------------------------
# Event payload validation
# ---------------------------------------------------------------------------


def validate_event_payload(event_type: str, payload: dict) -> dict:
    """Normalize one event payload through its typed SSE model.

    Uses the ``TURN_EVENT_PAYLOAD_MODELS`` registry (server/schemas.py); a
    payload the model rejects is passed through verbatim with a debug log —
    observability must never kill the stream. Unknown event types (future
    vocabulary) pass through untouched.

    Args:
        event_type: One of the ``TurnEventType`` constants.
        payload: Raw event payload dict from the loop.

    Returns:
        The validated (default-filled) payload dict, or the raw payload on
        validation failure.
    """
    from server.schemas import TURN_EVENT_PAYLOAD_MODELS

    model = TURN_EVENT_PAYLOAD_MODELS.get(event_type)
    if model is None:
        return payload
    try:
        return model.model_validate(payload or {}).model_dump()
    except Exception as exc:  # noqa: BLE001 — pass through, never drop the frame
        logger.debug(
            "turn loop %s event payload failed validation (%s) — passing raw",
            event_type,
            exc,
        )
        return payload


# ---------------------------------------------------------------------------
# The stream: queue-bridged run_turn_loop
# ---------------------------------------------------------------------------


@dataclass
class TurnStreamOutcome:
    """Terminal payload of :func:`run_turn_stream` (yield kind ``result``)."""

    result: TurnLoopResult
    """The loop's terminal result (answer / clarification / error terminal)."""

    effective_query: str
    """The sanitized (and PII-redacted, when applicable) loop query."""

    pii_redactions: int = 0
    """PII findings redacted pre-loop (surfaced for observability)."""

    rails_degraded: bool = False
    """True when input rails were configured but degraded (logged fallback)."""


async def run_turn_stream(
    *,
    query: str,
    request: Any,
    tenant_id: str,
    conversation_id: str,
    memory_structured: Optional[dict],
    temporal_client: Any,
    db_client: Any,
) -> AsyncIterator[tuple[str, Any]]:
    """Run one loop turn, yielding typed events then the terminal outcome.

    Yields ``(event_type, payload_dict)`` tuples for every loop event (only
    when ``RAG_TURN_LOOP_STREAM_EVENTS`` is on — the full trace is always
    available on the result), followed by exactly one
    ``(RESULT_KIND, TurnStreamOutcome)``. Input sanitize + PII gate run once
    before the loop; a rejection short-circuits to an ``ask_user`` terminal
    with ``stop_reason=STOP_REASON_INPUT_REJECTED``. A loop crash degrades to
    an ``ask_user`` terminal with ``stop_reason=STOP_REASON_ERROR`` — never an
    unexplained empty response.

    Output safety: the terminal answer passes through the configured output
    rails (:func:`_apply_output_rails`) before it is yielded. When a guardrail
    backend is configured, live ``draft`` frames are additionally SUPPRESSED
    from the stream (and their text redacted from ``metadata.turn_loop.trace``
    by :func:`trace_records`): draft deltas are pre-rail model output, so a
    railed deployment must not receive them verbatim. Without a backend the
    drafts stream live — that visibility is the design §8 contract.

    Args:
        query: Raw user query for this turn.
        request: The originating ``QueryRequest`` (filters, overrides).
        tenant_id: Resolved tenant id.
        conversation_id: Owning conversation id.
        memory_structured: ``MemoryContext.structured`` (empty dict/None when
            memory is disabled).
        temporal_client: Connected Temporal client (retrieval activity).
        db_client: Document store client (deep-study fetch), may be None.
    """
    prepared = await asyncio.to_thread(prepare_turn_input, query, tenant_id)
    if prepared.rejected:
        result = TurnLoopResult(
            answer="",
            action=TurnLoopResult.ACTION_ASK_USER,
            clarification=ClarificationOut(question=prepared.rejection_message),
            stop_reason=STOP_REASON_INPUT_REJECTED,
        )
        yield RESULT_KIND, TurnStreamOutcome(
            result=result,
            effective_query=prepared.query,
            pii_redactions=prepared.pii_redactions,
            rails_degraded=prepared.rails_degraded,
        )
        return

    from config import settings

    stream_events = bool(getattr(settings, "RAG_TURN_LOOP_STREAM_EVENTS", True))
    suppress_drafts = _guardrail_backend_configured()

    queue: asyncio.Queue = asyncio.Queue()

    async def _emit(event: TurnEvent) -> None:
        await queue.put(event)

    deps = TurnLoopDeps(
        retrieve_ranked=build_retrieve_ranked(temporal_client, request),
        fetch_document=build_fetch_document(db_client, conversation_id),
        llm_provider=_get_llm_provider(),
        emit=_emit,
    )
    context = build_turn_context(conversation_id, memory_structured)
    budget = TurnBudget.from_settings()
    run_turn_loop = _get_run_turn_loop()

    async def _run() -> TurnLoopResult:
        try:
            return await run_turn_loop(prepared.query, context, deps, budget)
        finally:
            await queue.put(_QUEUE_SENTINEL)

    task = asyncio.create_task(_run())
    try:
        while True:
            item = await queue.get()
            if item is _QUEUE_SENTINEL:
                break
            if not stream_events:
                continue
            if suppress_drafts and item.type == TurnEventType.DRAFT:
                # Pre-rail draft text must not reach a railed client live;
                # the trace keeps a redacted record (see trace_records).
                continue
            yield item.type, validate_event_payload(item.type, item.payload)
        try:
            result = await task
        except Exception as exc:  # noqa: BLE001 — degrade to an explained terminal
            logger.exception("turn loop crashed: %s", exc)
            result = TurnLoopResult(
                answer="",
                action=TurnLoopResult.ACTION_ASK_USER,
                clarification=ClarificationOut(question=_ERROR_CLARIFICATION),
                stop_reason=STOP_REASON_ERROR,
            )
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
    result = await _apply_output_rails(result)
    yield RESULT_KIND, TurnStreamOutcome(
        result=result,
        effective_query=prepared.query,
        pii_redactions=prepared.pii_redactions,
        rails_degraded=prepared.rails_degraded,
    )


# ---------------------------------------------------------------------------
# Result -> payload helpers (shared by /query and /query/stream)
# ---------------------------------------------------------------------------


def turn_loop_metadata(result: TurnLoopResult) -> dict:
    """Build the ``metadata.turn_loop`` block for responses/done frames."""
    meta: dict = {
        "enabled": True,
        "terminal": result.action,
        "confidence": round(_coerce_float(result.confidence), 4),
        "stop_reason": result.stop_reason,
        "actions": int(result.actions_taken),
        "llm_calls": int(result.llm_calls),
        "elapsed_ms": int(result.elapsed_ms),
    }
    if result.clarification is not None:
        meta["hints"] = list(result.clarification.hints)
        meta["scoping_questions"] = list(result.clarification.scoping_questions)
    return meta


def trace_records(trace: list[TurnEvent]) -> list[dict]:
    """Serialize the event trace for ``metadata.turn_loop.trace``.

    When a guardrail backend is configured, ``draft`` payload text is redacted
    (``text_delta`` emptied, ``redacted: true`` marker): drafts are pre-rail
    model output and the trace ships to the same client the railed answer
    does. Event structure/counts are preserved so trace consumers (action
    records, reasoning extraction) keep working.
    """
    redact_drafts = _guardrail_backend_configured()
    records: list[dict] = []
    for event in trace:
        payload = event.payload
        if redact_drafts and event.type == TurnEventType.DRAFT:
            payload = {**(payload or {}), "text_delta": "", "redacted": True}
        records.append({"type": event.type, "payload": payload, "ts_ms": event.ts_ms})
    return records


def pool_chunk_results(pool: list[EvidenceChunk]) -> list[dict]:
    """Map the evidence pool into the ``QueryResponse.results`` chunk shape."""
    results: list[dict] = []
    for chunk in pool:
        results.append(
            {
                "text": chunk.text,
                "score": chunk.score,
                "metadata": {
                    "source": chunk.source,
                    "source_key": chunk.source_key,
                    "document_id": chunk.document_id,
                    "heading": chunk.heading,
                    "section": chunk.heading,
                    "chunk_id": chunk.chunk_id,
                    "provenance": chunk.provenance,
                    "round_added": chunk.round_added,
                },
            }
        )
    return results


def pool_source_refs(pool: list[EvidenceChunk]) -> list[dict]:
    """Build citation source refs (with 1-based citation indexes) from the pool.

    Mirrors the ``_source_refs`` citation-panel shape for the loop path; the
    refactored char offsets are included when anchored (>= 0) so the source
    viewer can land highlights.
    """
    refs: list[dict] = []
    for index, chunk in enumerate(pool, start=1):
        ref: dict = {
            "citation_index": index,
            "source": chunk.source,
            "source_uri": "",
            "source_key": chunk.source_key,
            "document_id": chunk.document_id,
            "section": chunk.heading,
            "score": chunk.score,
            "text": chunk.text,
            "chunk_id": chunk.chunk_id,
        }
        if chunk.refactored_char_start >= 0:
            ref["refactored_char_start"] = chunk.refactored_char_start
        if chunk.refactored_char_end >= 0:
            ref["refactored_char_end"] = chunk.refactored_char_end
        refs.append(ref)
    return refs


def build_chunk_refs(pool: list[EvidenceChunk]) -> list[dict]:
    """Build memory ChunkRef dicts from the pool (design §7).

    ``preview`` is capped at ``RAG_TURN_CONTEXT_PREVIEW_CHARS`` unless
    ``RAG_TURN_CONTEXT_STORE_FULL_TEXT`` restores full-text storage (the
    memory provider re-applies the same cap at its boundary — this keeps the
    route honest even against a legacy provider). Full text stays recoverable
    by ``chunk_id``.
    """
    from config import settings

    store_full = bool(getattr(settings, "RAG_TURN_CONTEXT_STORE_FULL_TEXT", False))
    cap = int(getattr(settings, "RAG_TURN_CONTEXT_PREVIEW_CHARS", 320))
    refs: list[dict] = []
    for chunk in pool:
        preview = chunk.text if store_full else chunk.text[:cap]
        refs.append(
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "source_key": chunk.source_key,
                "heading": chunk.heading,
                "score": chunk.score,
                "refactored_char_start": chunk.refactored_char_start,
                "refactored_char_end": chunk.refactored_char_end,
                "preview": preview,
            }
        )
    return refs


def build_action_records(trace: list[TurnEvent]) -> list[dict]:
    """Derive TurnActionRecord dicts from the trace (design §7).

    Each ``turn_action`` event opens a record; its duration is the gap to the
    next action (or the last trace event) and its ``llm_calls`` counts the
    ``llm_call`` events dispatched under it. Pure trace arithmetic — no loop
    state needed.
    """
    records: list[dict] = []
    open_record: Optional[dict] = None
    open_ts = 0
    last_ts = 0
    for event in trace:
        last_ts = event.ts_ms
        if event.type == TurnEventType.TURN_ACTION:
            if open_record is not None:
                open_record["ms"] = max(0, event.ts_ms - open_ts)
            payload = event.payload or {}
            open_record = {
                "action": str(payload.get("action") or ""),
                "reason": str(payload.get("reason") or ""),
                "confidence": _coerce_float(payload.get("confidence")),
                "ms": 0,
                "llm_calls": 0,
            }
            open_ts = event.ts_ms
            records.append(open_record)
        elif event.type == TurnEventType.LLM_CALL and open_record is not None:
            open_record["llm_calls"] += 1
    if open_record is not None:
        open_record["ms"] = max(0, last_ts - open_ts)
    return records


def build_studied_entries(trace: list[TurnEvent]) -> list[dict]:
    """Aggregate ``deep_study`` events into docs-studied ledger entries.

    One entry per document (design §7 shape: ``{document_id, windows_read,
    sections, conclusion, ts}``); ``conclusion`` carries the last non-empty
    notes preview as a best-effort takeaway.
    """
    by_doc: dict[str, dict] = {}
    for event in trace:
        if event.type != TurnEventType.DEEP_STUDY:
            continue
        payload = event.payload or {}
        document_id = str(payload.get("document_id") or "")
        if not document_id:
            continue
        entry = by_doc.setdefault(
            document_id,
            {
                "document_id": document_id,
                "windows_read": [],
                "sections": [],
                "conclusion": "",
                "ts": event.ts_ms,
            },
        )
        window = payload.get("window")
        if window is not None and window not in entry["windows_read"]:
            entry["windows_read"].append(_coerce_int(window, 0))
        title = str(payload.get("title") or "")
        if title and title not in entry["sections"]:
            entry["sections"].append(title)
        notes = str(payload.get("notes_preview") or "")
        if notes:
            entry["conclusion"] = notes
        entry["ts"] = event.ts_ms
    return list(by_doc.values())


def extract_draft_reasoning(trace: list[TurnEvent]) -> str:
    """Concatenate the accepted draft attempt's captured reasoning deltas.

    The accepted attempt is the one whose ``gate`` event passed (else the
    last drafted attempt — the budget-exhausted best-effort answer). The loop
    emits draft events as ``{attempt, kind: 'reasoning'|'content',
    text_delta}`` — reasoning is every ``text_delta`` whose ``kind`` is
    ``reasoning`` (legacy ``reasoning_delta``/``reasoning`` keys are still
    honored as fallbacks). Drafts without reasoning yield an empty string, in
    which case the route skips the reasoning replay.
    """
    accepted_attempt: Optional[int] = None
    last_draft_attempt: Optional[int] = None
    for event in trace:
        payload = event.payload or {}
        if event.type == TurnEventType.GATE and payload.get("passed"):
            accepted_attempt = _coerce_int(payload.get("attempt"), 0)
        elif event.type == TurnEventType.DRAFT:
            last_draft_attempt = _coerce_int(payload.get("attempt"), 0)
    attempt = accepted_attempt if accepted_attempt is not None else last_draft_attempt
    if attempt is None:
        return ""
    parts: list[str] = []
    for event in trace:
        if event.type != TurnEventType.DRAFT:
            continue
        payload = event.payload or {}
        if _coerce_int(payload.get("attempt"), 0) != attempt:
            continue
        if str(payload.get("kind") or "") == "reasoning":
            delta = payload.get("text_delta") or ""
        else:
            delta = payload.get("reasoning_delta") or payload.get("reasoning") or ""
        if delta:
            parts.append(str(delta))
    return "".join(parts)


def iter_answer_tokens(text: str) -> Iterator[str]:
    """Replay an accepted answer as word-cadence token chunks.

    Splits on single spaces keeping separators attached, so joining the
    yielded tokens reproduces ``text`` exactly (newlines/tabs stay embedded
    in their word chunk, matching provider-delta behavior). Pure string ops.
    """
    if not text:
        return
    pieces = text.split(" ")
    last = len(pieces) - 1
    for index, piece in enumerate(pieces):
        token = piece if index == last else piece + " "
        if token:
            yield token


__all__ = [
    "RESULT_KIND",
    "STOP_REASON_INPUT_REJECTED",
    "STOP_REASON_ERROR",
    "PreparedTurnInput",
    "TurnStreamOutcome",
    "resolve_turn_loop_enabled",
    "prepare_turn_input",
    "build_turn_context",
    "evidence_from_dicts",
    "build_retrieve_ranked",
    "build_fetch_document",
    "run_turn_stream",
    "validate_event_payload",
    "turn_loop_metadata",
    "trace_records",
    "pool_chunk_results",
    "pool_source_refs",
    "build_chunk_refs",
    "build_action_records",
    "build_studied_entries",
    "extract_draft_reasoning",
    "iter_answer_tokens",
]
