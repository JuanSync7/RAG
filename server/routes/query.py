# @summary
# Query routes for standard and streaming retrieval endpoints with Temporal
# orchestration, plus the turn-loop branch: when the turn-level agentic
# conversation loop is active (request override or RAG_TURN_LOOP_ENABLED),
# both /query and /query/stream route through server.turn_loop_runner instead
# of RAGQueryWorkflow — streaming every typed loop event as its own SSE frame,
# replaying the accepted answer as standard token events, and persisting
# loop turns to memory with chunk refs / action records / studied docs.
# Exports: create_query_router, run_query
# Deps: fastapi, temporalio, server.schemas, server.turn_loop_runner,
#       src.platform.security, server.workflows, src.platform.llm
# @end-summary
"""Query API routes."""

from __future__ import annotations

import asyncio
import json as _json
import orjson as json_mod
import logging
import time
import uuid
from typing import Any, Awaitable, Callable

from fastapi import APIRouter, Body, Depends, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse
from temporalio.client import Client  # pyright: ignore[reportMissingImports]
from temporalio.service import RPCError  # pyright: ignore[reportMissingImports]

from config.settings import (
    GENERATION_MAX_TOKENS,
    GENERATION_TEMPERATURE,
    RAG_MEMORY_TITLE_DERIVE_MAX_CHARS,
    RAG_SUGGESTED_QUESTIONS_ENABLED,
)
from src.retrieval.pipeline.follow_up import generate_follow_ups_from_config
from server.schemas import ApiErrorResponse, QueryRequest, QueryResponse
from server.schemas import (
    ConversationCompactRequest,
    ConversationCreateRequest,
    ConversationHistoryResponse,
    ConversationMetaResponse,
    ConversationTitleUpdateRequest,
)
from server import turn_loop_runner
from server.workflows import RAG_QUERY_TASK_QUEUE, RAGQueryWorkflow
from src.retrieval.pipeline.turn_loop import TurnLoopResult
from src.platform import (
    MEMORY_OP_MS,
    MEMORY_SUMMARY_TRIGGERS,
    REQUESTS_TOTAL,
    REQUEST_LATENCY_MS,
    render_metrics,
)
from src.platform.memory import (
    conversation_meta_to_dict,
    conversation_turns_to_dict,
    get_conversation_memory,
)
from src.platform.security import (
    Principal,
    authenticate_request,
)
from src.platform.security import resolve_tenant_id


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json_mod.dumps(data).decode()}\n\n"


def _derive_title_from_query(
    query: str, *, max_len: int = RAG_MEMORY_TITLE_DERIVE_MAX_CHARS
) -> str:
    """Condense the user's first query into a short conversation title."""
    text = " ".join(str(query or "").split())
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rsplit(" ", 1)[0] or text[:max_len]
    return cut.rstrip(".,;:!?-") + "…"


def _autotitle_if_new(memory, *, tenant_id: str, principal, conv, query: str):
    """If the conversation is brand new, set its title from the first query.

    Returns the (possibly updated) ConversationMeta. Safe no-op otherwise.
    """
    if conv.message_count != 0:
        return conv
    existing = (conv.title or "").strip()
    if existing and existing != "New conversation":
        return conv
    derived = _derive_title_from_query(query)
    if not derived:
        return conv
    updated = memory.update_conversation_title(
        tenant_id=tenant_id,
        subject=principal.subject,
        project_id=principal.project_id,
        conversation_id=conv.conversation_id,
        title=derived,
    )
    return updated or conv


def _decode_page_bbox(raw: Any) -> dict | None:
    """Decode a stored ``page_bbox`` value into ``{l,t,r,b}`` or None.

    The Weaviate store persists ``page_bbox`` as a JSON-encoded TEXT
    property (see ``src/vector_db/weaviate/store.py``); the underlying
    Docling shape is a 4-tuple ``(x0, y0, x1, y1)``. This decoder accepts
    either a JSON list ``[x0, y0, x1, y1]`` or a JSON object with
    ``l/t/r/b`` keys and maps both to a ``{"l","t","r","b"}`` dict.

    Returns ``None`` for empty/missing values or malformed JSON — callers
    must never crash retrieval on a bad bbox.
    """
    if raw is None or raw == "":
        return None
    # Already-decoded values from in-process callers (defensive).
    if isinstance(raw, dict):
        try:
            return {k: float(raw[k]) for k in ("l", "t", "r", "b")}
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        try:
            x0, y0, x1, y1 = (float(v) for v in raw)
        except (TypeError, ValueError):
            return None
        return {"l": x0, "t": y0, "r": x1, "b": y1}
    if not isinstance(raw, str):
        return None
    try:
        decoded = _json.loads(raw)
    except (ValueError, TypeError):
        logging.getLogger(__name__).debug(
            "malformed page_bbox JSON; dropping bbox: %r", raw
        )
        return None
    if isinstance(decoded, list) and len(decoded) == 4:
        try:
            x0, y0, x1, y1 = (float(v) for v in decoded)
        except (TypeError, ValueError):
            return None
        return {"l": x0, "t": y0, "r": x1, "b": y1}
    if isinstance(decoded, dict):
        try:
            return {k: float(decoded[k]) for k in ("l", "t", "r", "b")}
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _chunk_headings(chunks: list) -> list[str]:
    """Section headings from stream-path chunk dicts — the in-corpus topics used to
    ground follow-up-question suggestions. Prefers the heading_path breadcrumb."""
    out: list[str] = []
    for c in chunks:
        meta = c.get("metadata", {}) if isinstance(c, dict) else {}
        hp = meta.get("heading_path") or []
        if isinstance(hp, list) and hp:
            out.append(" > ".join(str(h) for h in hp if h))
        elif meta.get("heading"):
            out.append(str(meta.get("heading")))
    return out


def _source_refs(results: list) -> list[dict]:
    """Extract source references from RAG results (full chunk text for the citations panel)."""
    refs = []
    for r in results:
        meta: dict = r.get("metadata", {}) if isinstance(r, dict) else getattr(r, "metadata", {}) or {}
        score: float = r.get("score", 0.0) if isinstance(r, dict) else getattr(r, "score", 0.0)
        text: str = r.get("text", "") if isinstance(r, dict) else getattr(r, "text", "")
        ref: dict = {
            "source": meta.get("source", ""),
            "source_uri": meta.get("source_uri", ""),
            "source_key": meta.get("source_key", ""),
            "document_id": meta.get("document_id", ""),
            "section": meta.get("section") or meta.get("heading", ""),
            "score": score,
            "text": text or "",
        }
        start = meta.get("original_char_start")
        end = meta.get("original_char_end")
        if start is not None:
            ref["original_char_start"] = start
        if end is not None:
            ref["original_char_end"] = end
        # Page provenance. ``page_no`` is stored as int; ``page_bbox`` is
        # JSON-encoded TEXT in Weaviate (empty string when absent). Decode
        # at read-time so the citation UI/API receives structured shapes.
        page_no = meta.get("page_no")
        if page_no is not None and page_no != "":
            try:
                ref["page_no"] = int(page_no)
            except (TypeError, ValueError):
                pass
        ref["page_bbox"] = _decode_page_bbox(meta.get("page_bbox"))
        refs.append(ref)
    return refs


def _aggregate_stage_totals(stage_timings: list[dict]) -> dict:
    bucket_totals: dict[str, float] = {}
    for stage in stage_timings:
        bucket = str(stage.get("bucket", "other"))
        bucket_totals[bucket] = bucket_totals.get(bucket, 0.0) + float(stage.get("ms", 0.0))
    totals = {f"{bucket}_ms": round(ms, 1) for bucket, ms in bucket_totals.items()}
    totals["total_ms"] = round(sum(bucket_totals.values()), 1)
    return totals


def _stream_llm(
    query: str,
    context_chunks: list[str],
    scores: list[float],
    stage_timings: list[dict] | None = None,
    memory_context: str | None = None,
    memory_recent_turns: list[dict] | None = None,
):
    """Stream generation events via LLMProvider (provider-agnostic).

    Delegates prompt assembly to `OllamaGenerator.build_messages` so the
    streaming and non-streaming paths share a single source of truth for
    prompt shape (system prompt, doc-context layout, memory framing).

    Yields ``(kind, text)`` tuples where ``kind`` is ``"reasoning"`` for live
    chain-of-thought deltas (reasoning models) or ``"content"`` for answer
    deltas. The endpoint maps these to distinct SSE event types so the UI can
    show the model "thinking" before the answer streams.
    """
    from src.platform.llm import get_llm_provider
    from src.retrieval.generation.nodes import OllamaGenerator

    def _record_stage(stage: str, bucket: str, started_at: float) -> None:
        if stage_timings is None:
            return
        stage_timings.append(
            {"stage": stage, "bucket": bucket, "ms": round((time.perf_counter() - started_at) * 1000, 1)}
        )

    prep_start = time.perf_counter()
    messages = OllamaGenerator.build_messages(
        query=query,
        context_chunks=context_chunks,
        scores=scores,
        memory_context=memory_context,
        recent_turns=memory_recent_turns,
    )
    _record_stage("prompt_prepare", "generation", prep_start)

    stream_start = time.perf_counter()
    provider = get_llm_provider()
    for kind, text in provider.generate_stream(
        messages,
        model_alias="default",
        temperature=GENERATION_TEMPERATURE,
        max_tokens=GENERATION_MAX_TOKENS,
        include_reasoning=True,
    ):
        yield kind, text
    _record_stage("stream_tokens", "generation", stream_start)


# ---------------------------------------------------------------------------
# Turn-loop branch (TURN_LOOP_DESIGN.md §3, §5, §8)
#
# When the loop is active the request never reaches RAGQueryWorkflow: the
# API-process runner drives retrieval/deep-study/clarify/answer itself. The
# loop path deliberately does NOT inject ignored_doc_ids and does NOT call
# mark_retrieved (design §5 — deepening into a prior document is a
# first-class move); the non-loop path below stays byte-identical.
# ---------------------------------------------------------------------------


def _write_turn_loop_memory(
    *,
    memory,
    request: QueryRequest,
    principal: Principal,
    tenant_id: str,
    conversation_id: str,
    workflow_id: str,
    result: TurnLoopResult,
) -> None:
    """Persist one loop turn to conversation memory (design §7).

    Writes the user turn, then the assistant turn carrying the loop's typed
    extras (action records, preview-capped chunk refs, answer confidence and
    any pending clarification), records each deep-studied document on the
    conversation ledger, and runs the usual compaction pass. No-op when
    memory is disabled for the request.
    """
    if not request.memory_enabled:
        return
    scope = dict(
        tenant_id=tenant_id,
        subject=principal.subject,
        project_id=principal.project_id,
        conversation_id=conversation_id,
    )
    mem_start = time.perf_counter()
    memory.append_turn(**scope, role="user", content=request.query.strip(), query_id=workflow_id)
    clarification = None
    if result.clarification is not None:
        clarification = {
            "question": result.clarification.question,
            "hints": list(result.clarification.hints),
        }
    assistant_text = (result.answer or "").strip()
    if not assistant_text and result.clarification is not None:
        assistant_text = result.clarification.question.strip()
    if assistant_text:
        memory.append_turn(
            **scope,
            role="assistant",
            content=assistant_text,
            query_id=workflow_id,
            sources=turn_loop_runner.pool_source_refs(result.pool),
            actions=turn_loop_runner.build_action_records(result.trace),
            chunk_refs=turn_loop_runner.build_chunk_refs(result.pool),
            answer_confidence=result.confidence,
            clarification=clarification,
        )
    MEMORY_OP_MS.labels(operation="append_turn").observe(
        (time.perf_counter() - mem_start) * 1000
    )
    for entry in turn_loop_runner.build_studied_entries(result.trace):
        memory.record_doc_studied(**scope, entry=entry)
    mem_start = time.perf_counter()
    memory.compact_if_needed(**scope, force=request.compact_now)
    MEMORY_OP_MS.labels(operation="compact").observe(
        (time.perf_counter() - mem_start) * 1000
    )
    MEMORY_SUMMARY_TRIGGERS.labels(
        reason="manual" if request.compact_now else "threshold"
    ).inc()


def _turn_loop_memory_structured(
    memory, *, request: QueryRequest, principal: Principal, tenant_id: str, conversation_id: str
) -> dict:
    """Build the structured TurnContext input from conversation memory."""
    if not request.memory_enabled:
        return {}
    mem_start = time.perf_counter()
    built = memory.build_context(
        tenant_id=tenant_id,
        subject=principal.subject,
        project_id=principal.project_id,
        conversation_id=conversation_id,
        turn_window=request.memory_turn_window,
    )
    MEMORY_OP_MS.labels(operation="build_context").observe(
        (time.perf_counter() - mem_start) * 1000
    )
    return getattr(built, "structured", None) or {}


def _turn_loop_base_response(
    *,
    request: QueryRequest,
    outcome: "turn_loop_runner.TurnStreamOutcome",
    workflow_id: str,
    conversation_id: str,
    latency_ms: float,
) -> dict:
    """Map a loop outcome onto the standard QueryResponse field set.

    Answered terminals use the standard ``action='search'`` vocabulary (the
    loop truth lives in ``metadata.turn_loop.terminal``) so existing clients
    render loop turns unchanged; clarify terminals use the existing
    ``action='ask_user'`` + ``clarification_message`` shape.
    """
    result = outcome.result
    ask_user = result.action == TurnLoopResult.ACTION_ASK_USER
    clarification = result.clarification
    base = {
        "query": request.query,
        "processed_query": outcome.effective_query,
        "query_confidence": result.confidence,
        "action": "ask_user" if ask_user else "search",
        "results": [] if ask_user else turn_loop_runner.pool_chunk_results(result.pool),
        "clarification_message": (
            clarification.question if (ask_user and clarification) else None
        ),
        "generated_answer": (result.answer or None) if not ask_user else None,
        "workflow_id": workflow_id,
        "conversation_id": conversation_id,
        "latency_ms": round(latency_ms, 1),
        "stage_timings": [],
        "timing_totals": {"total_ms": round(latency_ms, 1)},
    }
    if ask_user and result.stop_reason == turn_loop_runner.STOP_REASON_INPUT_REJECTED:
        base["ask_user_reason"] = "sanitizer_reject"
    return base


async def _run_turn_loop_query(
    *,
    request: QueryRequest,
    principal: Principal,
    memory,
    tenant_id: str,
    conversation_id: str,
    endpoint: str,
    temporal_client: Client,
    db_client,
    logger: logging.Logger,
) -> JSONResponse:
    """Non-stream turn-loop execution: run the loop, return the enriched body.

    The full typed event trace is returned in ``metadata.turn_loop.trace``
    (``{type, payload, ts_ms}`` records) alongside the standard answer /
    clarification fields. Returned as a ``JSONResponse`` (validated through
    ``QueryResponse`` first) so the loop-only ``metadata`` key rides along
    without touching the non-loop response schema.
    """
    workflow_id = f"rag-turn-{uuid.uuid4().hex[:12]}"
    # Memory access is synchronous (Redis round-trips + a possible blocking
    # LLM summarization inside compaction); run it off the event loop so a
    # single turn cannot freeze every concurrent SSE stream.
    structured = await asyncio.to_thread(
        _turn_loop_memory_structured,
        memory,
        request=request,
        principal=principal,
        tenant_id=tenant_id,
        conversation_id=conversation_id,
    )
    start = time.perf_counter()
    outcome = None
    async for kind, payload in turn_loop_runner.run_turn_stream(
        query=request.query,
        request=request,
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        memory_structured=structured,
        temporal_client=temporal_client,
        db_client=db_client,
    ):
        if kind == turn_loop_runner.RESULT_KIND:
            outcome = payload
    if outcome is None:  # defensive: the runner always yields a terminal
        raise HTTPException(status_code=500, detail="turn loop produced no result")
    result = outcome.result
    total_ms = (time.perf_counter() - start) * 1000
    base = _turn_loop_base_response(
        request=request,
        outcome=outcome,
        workflow_id=workflow_id,
        conversation_id=conversation_id,
        latency_ms=total_ms,
    )
    # Validate through the response model, then attach the loop metadata.
    body = QueryResponse.model_validate(base).model_dump(mode="json")
    body["metadata"] = {
        "turn_loop": {
            **turn_loop_runner.turn_loop_metadata(result),
            "trace": turn_loop_runner.trace_records(result.trace),
        }
    }
    await asyncio.to_thread(
        _write_turn_loop_memory,
        memory=memory,
        request=request,
        principal=principal,
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        workflow_id=workflow_id,
        result=result,
    )
    REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="200").inc()
    REQUEST_LATENCY_MS.labels(endpoint=endpoint, method="POST").observe(total_ms)
    return JSONResponse(content=body)


async def _turn_loop_event_stream(
    *,
    request: QueryRequest,
    principal: Principal,
    memory,
    tenant_id: str,
    conversation_id: str,
    temporal_client: Client,
    db_client,
    slot_acquired: bool,
    release_request_slot: Callable[[bool], None],
    emit_stream_observability: Callable[..., None],
    logger: logging.Logger,
):
    """SSE generator for a turn-loop stream (design §8 ordering contract).

    Frame order: every typed loop event live (each ``TurnEventType`` as its
    own SSE event; draft deltas pass through as ``draft``), then a standard
    ``retrieval`` frame (sources panel / clarify shape), then — for answered
    terminals — the accepted draft's captured ``reasoning`` followed by the
    answer replayed as standard ``token`` frames, and finally the ``done``
    frame carrying pool-derived sources plus ``metadata.turn_loop``.
    """
    workflow_id = f"rag-turn-{uuid.uuid4().hex[:12]}"
    start = time.perf_counter()
    token_count = 0
    error_message: str | None = None
    try:
        structured = await asyncio.to_thread(
            _turn_loop_memory_structured,
            memory,
            request=request,
            principal=principal,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
        )
        outcome = None
        async for kind, payload in turn_loop_runner.run_turn_stream(
            query=request.query,
            request=request,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            memory_structured=structured,
            temporal_client=temporal_client,
            db_client=db_client,
        ):
            if kind == turn_loop_runner.RESULT_KIND:
                outcome = payload
                break
            yield _sse(kind, payload)
            await asyncio.sleep(0)
        if outcome is None:  # defensive: the runner always yields a terminal
            raise RuntimeError("turn loop produced no result")
        result = outcome.result
        loop_meta = turn_loop_runner.turn_loop_metadata(result)
        retrieval_payload = _turn_loop_base_response(
            request=request,
            outcome=outcome,
            workflow_id=workflow_id,
            conversation_id=conversation_id,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
        retrieval_payload["metadata"] = {"turn_loop": loop_meta}
        yield _sse("retrieval", retrieval_payload)
        if result.action != TurnLoopResult.ACTION_ASK_USER:
            # The accepted answer is railed (run_turn_stream ->
            # _apply_output_rails), but the captured chain-of-thought is not;
            # on a railed deployment it must be suppressed rather than replayed
            # verbatim (parity with live-draft suppression) so pre-rail model
            # output never reaches a railed client.
            if not turn_loop_runner.guardrail_backend_configured():
                reasoning = turn_loop_runner.extract_draft_reasoning(result.trace)
                if reasoning:
                    yield _sse("reasoning", {"text": reasoning})
            for token in turn_loop_runner.iter_answer_tokens(result.answer):
                token_count += 1
                yield _sse("token", {"token": token})
                await asyncio.sleep(0)
        total_ms = (time.perf_counter() - start) * 1000
        timing_totals = {"total_ms": round(total_ms, 1)}
        yield _sse(
            "done",
            {
                "latency_ms": round(total_ms, 1),
                "retrieval_ms": 0.0,
                "generation_ms": round(total_ms, 1),
                "token_count": token_count,
                "stage_timings": [],
                "timing_totals": timing_totals,
                "conversation_id": conversation_id,
                "token_budget": None,
                "sources": turn_loop_runner.pool_source_refs(result.pool),
                "metadata": {"turn_loop": loop_meta},
            },
        )
        await asyncio.to_thread(
            _write_turn_loop_memory,
            memory=memory,
            request=request,
            principal=principal,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            workflow_id=workflow_id,
            result=result,
        )
        REQUESTS_TOTAL.labels(endpoint="/query/stream", method="POST", status="200").inc()
        REQUEST_LATENCY_MS.labels(endpoint="/query/stream", method="POST").observe(total_ms)
        emit_stream_observability(
            workflow_id=workflow_id,
            request=request,
            retrieval_ms=0.0,
            generation_ms=total_ms,
            latency_ms=total_ms,
            token_count=token_count,
            stage_timings=[],
            timing_totals=timing_totals,
            outcome="completed",
        )
    except Exception as exc:  # noqa: BLE001 — stream errors surface as frames
        logger.exception("Turn loop stream error: %s", exc)
        error_message = str(exc)
        yield _sse("error", {"message": error_message})
        emit_stream_observability(
            workflow_id=workflow_id,
            request=request,
            retrieval_ms=0.0,
            generation_ms=0.0,
            latency_ms=(time.perf_counter() - start) * 1000,
            token_count=token_count,
            stage_timings=[],
            timing_totals={"total_ms": 0.0},
            outcome="error",
            error_message=error_message,
        )
    finally:
        release_request_slot(slot_acquired)


async def run_query(
    request: QueryRequest,
    principal: Principal,
    *,
    endpoint: str,
    temporal_client: Client | None,
    require_role: Callable[[Principal, str], None],
    enforce_rate_limit: Callable[[Principal, str], None],
    acquire_request_slot: Callable[[str], Awaitable[bool]],
    release_request_slot: Callable[[bool], None],
    logger: logging.Logger,
    db_client=None,
) -> QueryResponse | JSONResponse:
    """Execute non-stream query workflow and return API response model.

    Turn-loop requests return a ``JSONResponse`` (QueryResponse-validated body
    plus the loop-only ``metadata.turn_loop`` block); all other requests
    return the ``QueryResponse`` model unchanged.
    """
    if temporal_client is None:
        raise HTTPException(status_code=503, detail="Temporal client not connected")
    require_role(principal, "query")
    enforce_rate_limit(principal, endpoint)
    slot_acquired = await acquire_request_slot(endpoint)
    memory = get_conversation_memory()

    try:
        workflow_id = f"rag-query-{uuid.uuid4().hex[:12]}"
        tenant_id = resolve_tenant_id(principal, request.tenant_id)
        mem_start = time.perf_counter()
        conv = memory.ensure_conversation(
            tenant_id=tenant_id,
            subject=principal.subject,
            project_id=principal.project_id,
            conversation_id=request.conversation_id,
        )
        MEMORY_OP_MS.labels(operation="ensure_conversation").observe(
            (time.perf_counter() - mem_start) * 1000
        )
        conv = _autotitle_if_new(
            memory,
            tenant_id=tenant_id,
            principal=principal,
            conv=conv,
            query=request.query,
        )
        # Turn-loop branch: the loop owns retrieval + generation for the
        # turn — the RAGQueryWorkflow dispatch below never runs (design §3).
        # The full request rides along so the env default yields to
        # explicitly-requested competing orchestrators / retrieval mode.
        if turn_loop_runner.resolve_turn_loop_enabled(
            request.turn_loop, request=request
        ):
            return await _run_turn_loop_query(
                request=request,
                principal=principal,
                memory=memory,
                tenant_id=tenant_id,
                conversation_id=conv.conversation_id,
                endpoint=endpoint,
                temporal_client=temporal_client,
                db_client=db_client,
                logger=logger,
            )
        payload = request.model_dump(exclude_none=True)
        payload["tenant_id"] = tenant_id
        payload["conversation_id"] = conv.conversation_id
        if request.mode == "retrieval":
            payload["skip_generation"] = True
        # Hard-suppress previously-served docs (relevant ∪ ignored).
        # Lives outside ``memory_enabled`` because doc state tracking is
        # logically separate from chat-history context building — Noop
        # provider naturally short-circuits this to an empty list.
        seen_doc_ids = memory.get_seen_doc_ids(
            tenant_id=tenant_id,
            subject=principal.subject,
            project_id=principal.project_id,
            conversation_id=conv.conversation_id,
        )
        if seen_doc_ids:
            payload["ignored_doc_ids"] = seen_doc_ids
        if request.memory_enabled:
            mem_start = time.perf_counter()
            ctx = memory.build_context(
                tenant_id=tenant_id,
                subject=principal.subject,
                project_id=principal.project_id,
                conversation_id=conv.conversation_id,
                turn_window=request.memory_turn_window,
            )
            payload["memory_context"] = ctx.context_text
            payload["memory_recent_turns"] = conversation_turns_to_dict(ctx.recent_turns)
            MEMORY_OP_MS.labels(operation="build_context").observe(
                (time.perf_counter() - mem_start) * 1000
            )

        start = time.perf_counter()
        try:
            result = await temporal_client.execute_workflow(
                RAGQueryWorkflow.run,
                payload,
                id=workflow_id,
                task_queue=RAG_QUERY_TASK_QUEUE,
            )
        except RPCError as exc:
            REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="503").inc()
            logger.error("Temporal RPC error: %s", exc)
            raise HTTPException(
                status_code=503,
                detail=f"Temporal unavailable: {exc}. Is the worker running?",
            )
        except Exception as exc:
            REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="500").inc()
            logger.error("Workflow execution failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

        total_ms = (time.perf_counter() - start) * 1000
        result["workflow_id"] = workflow_id
        result["conversation_id"] = conv.conversation_id
        if "latency_ms" not in result:
            result["latency_ms"] = round(total_ms, 1)
        # Update conversation doc-state so subsequent queries don't re-serve
        # what was just returned. Marks new doc_ids as relevant by default;
        # already-ignored ids stay ignored. Echoed back to the client so the
        # Retrieval tab can hydrate panes without a follow-up call.
        new_doc_ids = [
            r["metadata"]["document_id"]
            for r in result.get("results", []) or []
            if isinstance(r, dict)
            and isinstance(r.get("metadata"), dict)
            and r["metadata"].get("document_id")
        ]
        meta_after = memory.mark_retrieved(
            tenant_id=tenant_id,
            subject=principal.subject,
            project_id=principal.project_id,
            conversation_id=conv.conversation_id,
            doc_ids=new_doc_ids,
        )
        result["relevant_doc_ids"] = list(meta_after.relevant_doc_ids)
        result["ignored_doc_ids"] = list(meta_after.ignored_doc_ids)
        result["seen_doc_ids"] = list(
            dict.fromkeys([*meta_after.relevant_doc_ids, *meta_after.ignored_doc_ids])
        )
        # Retrieval and chat both write turns now — retrieval turns carry sources
        # but empty assistant content, so a single conversation can be replayed
        # under either renderer (citation strip vs doc cards) on the frontend.
        if request.memory_enabled:
            user_text = request.query.strip()
            generated = result.get("generated_answer") or ""
            clarification = result.get("clarification_message") or ""
            assistant_text = str(generated).strip() or str(clarification).strip()
            is_retrieval = request.mode == "retrieval"
            source_refs = _source_refs(result.get("results", []))
            mem_start = time.perf_counter()
            memory.append_turn(
                tenant_id=tenant_id,
                subject=principal.subject,
                project_id=principal.project_id,
                conversation_id=conv.conversation_id,
                role="user",
                content=user_text,
                query_id=workflow_id,
            )
            # REQ-1207: Don't store BLOCK/FLAG responses in memory —
            # prevents error echo accumulation across turns.
            post_action = result.get("post_guardrail_action", "")
            should_write_assistant = (
                (assistant_text or (is_retrieval and source_refs))
                and post_action not in ("block", "flag")
            )
            if should_write_assistant:
                memory.append_turn(
                    tenant_id=tenant_id,
                    subject=principal.subject,
                    project_id=principal.project_id,
                    conversation_id=conv.conversation_id,
                    role="assistant",
                    content=assistant_text,
                    query_id=workflow_id,
                    sources=source_refs,
                )
            MEMORY_OP_MS.labels(operation="append_turn").observe(
                (time.perf_counter() - mem_start) * 1000
            )
            mem_start = time.perf_counter()
            memory.compact_if_needed(
                tenant_id=tenant_id,
                subject=principal.subject,
                project_id=principal.project_id,
                conversation_id=conv.conversation_id,
                force=request.compact_now,
            )
            MEMORY_OP_MS.labels(operation="compact").observe((time.perf_counter() - mem_start) * 1000)
            MEMORY_SUMMARY_TRIGGERS.labels(
                reason="manual" if request.compact_now else "threshold"
            ).inc()
        REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="200").inc()
        REQUEST_LATENCY_MS.labels(endpoint=endpoint, method="POST").observe(total_ms)
        return QueryResponse(**result)
    finally:
        release_request_slot(slot_acquired)


def create_query_router(
    *,
    get_temporal_client: Callable[[], Client | None],
    require_role: Callable[[Principal, str], None],
    enforce_rate_limit: Callable[[Principal, str], None],
    acquire_request_slot: Callable[[str], Awaitable[bool]],
    release_request_slot: Callable[[bool], None],
    emit_stream_observability: Callable[..., None],
    logger: logging.Logger,
    db_client=None,
) -> APIRouter:
    """Create router for query, query-stream, and metrics endpoints.

    Args:
        db_client: Optional document store client (same handle the documents
            router receives) — required by the turn loop's DEEP_STUDY fetch;
            the loop degrades to retrieval-only evidence when absent.
    """
    standard_error_responses = {
        401: {"model": ApiErrorResponse},
        403: {"model": ApiErrorResponse},
        404: {"model": ApiErrorResponse},
        422: {"model": ApiErrorResponse},
        429: {"model": ApiErrorResponse},
        500: {"model": ApiErrorResponse},
        503: {"model": ApiErrorResponse},
    }
    router = APIRouter()

    @router.post("/query", response_model=QueryResponse, responses=standard_error_responses)
    async def query(request: QueryRequest, principal: Principal = Depends(authenticate_request)):
        return await run_query(
            request,
            principal,
            endpoint="/query",
            temporal_client=get_temporal_client(),
            require_role=require_role,
            enforce_rate_limit=enforce_rate_limit,
            acquire_request_slot=acquire_request_slot,
            release_request_slot=release_request_slot,
            logger=logger,
            db_client=db_client,
        )

    @router.post("/query/stream", responses=standard_error_responses)
    async def query_stream(request: QueryRequest, principal: Principal = Depends(authenticate_request)):
        temporal_client = get_temporal_client()
        if temporal_client is None:
            raise HTTPException(status_code=503, detail="Temporal client not connected")
        require_role(principal, "query")
        enforce_rate_limit(principal, "/query/stream")
        slot_acquired = await acquire_request_slot("/query/stream")
        # Setup between slot acquisition and ownership transfer to a streaming
        # generator must release the slot if it raises — otherwise a raising
        # call here (e.g. resolve_turn_loop_enabled -> HTTP 500 on invalid
        # RAG_TURN_LOOP_* config) permanently leaks an inflight permit and the
        # server wedges at 503. Each StreamingResponse below hands the slot to
        # a generator whose finally releases it, so ownership transfer happens
        # outside this guard.
        try:
            memory = get_conversation_memory()
            tenant_id = resolve_tenant_id(principal, request.tenant_id)
            mem_start = time.perf_counter()
            conv = memory.ensure_conversation(
                tenant_id=tenant_id,
                subject=principal.subject,
                project_id=principal.project_id,
                conversation_id=request.conversation_id,
            )
            MEMORY_OP_MS.labels(operation="ensure_conversation").observe(
                (time.perf_counter() - mem_start) * 1000
            )
            conv = _autotitle_if_new(
                memory,
                tenant_id=tenant_id,
                principal=principal,
                conv=conv,
                query=request.query,
            )
            turn_loop_on = turn_loop_runner.resolve_turn_loop_enabled(
                request.turn_loop, request=request
            )
        except BaseException:
            release_request_slot(slot_acquired)
            raise
        # Turn-loop branch: stream the loop's typed events instead of the
        # RAGQueryWorkflow retrieval + in-process generation path below.
        # The full request rides along so the env default yields to
        # explicitly-requested competing orchestrators / retrieval mode.
        if turn_loop_on:
            return StreamingResponse(
                _turn_loop_event_stream(
                    request=request,
                    principal=principal,
                    memory=memory,
                    tenant_id=tenant_id,
                    conversation_id=conv.conversation_id,
                    temporal_client=temporal_client,
                    db_client=db_client,
                    slot_acquired=slot_acquired,
                    release_request_slot=release_request_slot,
                    emit_stream_observability=emit_stream_observability,
                    logger=logger,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        mem_ctx_text = ""
        mem_recent_turns: list[dict] = []
        if request.memory_enabled:
            mem_start = time.perf_counter()
            built = memory.build_context(
                tenant_id=tenant_id,
                subject=principal.subject,
                project_id=principal.project_id,
                conversation_id=conv.conversation_id,
                turn_window=request.memory_turn_window,
            )
            mem_ctx_text = built.context_text
            mem_recent_turns = conversation_turns_to_dict(built.recent_turns)
            MEMORY_OP_MS.labels(operation="build_context").observe(
                (time.perf_counter() - mem_start) * 1000
            )

        workflow_id = f"rag-stream-{uuid.uuid4().hex[:12]}"
        payload = request.model_dump(exclude_none=True)
        payload["skip_generation"] = True
        payload["tenant_id"] = tenant_id
        payload["conversation_id"] = conv.conversation_id
        # Hard-suppress previously-served docs (relevant ∪ ignored).
        seen_doc_ids = memory.get_seen_doc_ids(
            tenant_id=tenant_id,
            subject=principal.subject,
            project_id=principal.project_id,
            conversation_id=conv.conversation_id,
        )
        if seen_doc_ids:
            payload["ignored_doc_ids"] = seen_doc_ids
        if request.memory_enabled:
            payload["memory_context"] = mem_ctx_text
            payload["memory_recent_turns"] = mem_recent_turns

        async def event_generator():
            try:
                start = time.perf_counter()
                stream_error_message: str | None = None
                generated_text_parts: list[str] = []
                try:
                    retrieval_result = await temporal_client.execute_workflow(
                        RAGQueryWorkflow.run,
                        payload,
                        id=workflow_id,
                        task_queue=RAG_QUERY_TASK_QUEUE,
                    )
                except RPCError as exc:
                    REQUESTS_TOTAL.labels(endpoint="/query/stream", method="POST", status="503").inc()
                    stream_error_message = f"Retrieval failed: {exc}"
                    yield _sse("error", {"message": stream_error_message})
                    emit_stream_observability(
                        workflow_id=workflow_id,
                        request=request,
                        retrieval_ms=0.0,
                        generation_ms=0.0,
                        latency_ms=(time.perf_counter() - start) * 1000,
                        token_count=0,
                        stage_timings=[],
                        timing_totals={"total_ms": 0.0},
                        outcome="error",
                        error_message=stream_error_message,
                    )
                    return
                except Exception as exc:
                    REQUESTS_TOTAL.labels(endpoint="/query/stream", method="POST", status="500").inc()
                    stream_error_message = str(exc)
                    yield _sse("error", {"message": stream_error_message})
                    emit_stream_observability(
                        workflow_id=workflow_id,
                        request=request,
                        retrieval_ms=0.0,
                        generation_ms=0.0,
                        latency_ms=(time.perf_counter() - start) * 1000,
                        token_count=0,
                        stage_timings=[],
                        timing_totals={"total_ms": 0.0},
                        outcome="error",
                        error_message=stream_error_message,
                    )
                    return

                retrieval_ms = (time.perf_counter() - start) * 1000
                retrieval_result["workflow_id"] = workflow_id
                retrieval_result["latency_ms"] = round(retrieval_ms, 1)
                retrieval_result["conversation_id"] = conv.conversation_id
                # Mark newly-returned docs as relevant on the conversation;
                # echo doc-state for client hydration.
                stream_new_doc_ids = [
                    r["metadata"]["document_id"]
                    for r in retrieval_result.get("results", []) or []
                    if isinstance(r, dict)
                    and isinstance(r.get("metadata"), dict)
                    and r["metadata"].get("document_id")
                ]
                stream_meta_after = memory.mark_retrieved(
                    tenant_id=tenant_id,
                    subject=principal.subject,
                    project_id=principal.project_id,
                    conversation_id=conv.conversation_id,
                    doc_ids=stream_new_doc_ids,
                )
                retrieval_result["relevant_doc_ids"] = list(stream_meta_after.relevant_doc_ids)
                retrieval_result["ignored_doc_ids"] = list(stream_meta_after.ignored_doc_ids)
                retrieval_result["seen_doc_ids"] = list(
                    dict.fromkeys([
                        *stream_meta_after.relevant_doc_ids,
                        *stream_meta_after.ignored_doc_ids,
                    ])
                )
                retrieval_stages = list(retrieval_result.get("stage_timings", []))
                yield _sse("retrieval", retrieval_result)

                chunks = retrieval_result.get("results", [])
                processed_query = retrieval_result.get("processed_query", request.query)
                _stream_token_budget = retrieval_result.get("token_budget")
                if retrieval_result.get("action") != "search" or not chunks:
                    done_payload = {
                        "latency_ms": round(retrieval_ms, 1),
                        "retrieval_ms": round(retrieval_ms, 1),
                        "generation_ms": 0.0,
                        "token_count": 0,
                        "stage_timings": retrieval_stages,
                        "timing_totals": _aggregate_stage_totals(retrieval_stages),
                        "conversation_id": conv.conversation_id,
                        "token_budget": _stream_token_budget,
                    }
                    if request.memory_enabled and request.mode != "retrieval":
                        mem_start = time.perf_counter()
                        memory.append_turn(
                            tenant_id=tenant_id,
                            subject=principal.subject,
                            project_id=principal.project_id,
                            conversation_id=conv.conversation_id,
                            role="user",
                            content=request.query,
                            query_id=workflow_id,
                        )
                        clar = str(retrieval_result.get("clarification_message", "")).strip()
                        if clar:
                            memory.append_turn(
                                tenant_id=tenant_id,
                                subject=principal.subject,
                                project_id=principal.project_id,
                                conversation_id=conv.conversation_id,
                                role="assistant",
                                content=clar,
                                query_id=workflow_id,
                                sources=[],
                            )
                        MEMORY_OP_MS.labels(operation="append_turn").observe(
                            (time.perf_counter() - mem_start) * 1000
                        )
                        mem_start = time.perf_counter()
                        memory.compact_if_needed(
                            tenant_id=tenant_id,
                            subject=principal.subject,
                            project_id=principal.project_id,
                            conversation_id=conv.conversation_id,
                            force=request.compact_now,
                        )
                        MEMORY_OP_MS.labels(operation="compact").observe(
                            (time.perf_counter() - mem_start) * 1000
                        )
                        MEMORY_SUMMARY_TRIGGERS.labels(
                            reason="manual" if request.compact_now else "threshold"
                        ).inc()
                    yield _sse("done", done_payload)
                    emit_stream_observability(
                        workflow_id=workflow_id,
                        request=request,
                        retrieval_ms=retrieval_ms,
                        generation_ms=0.0,
                        latency_ms=retrieval_ms,
                        token_count=0,
                        stage_timings=retrieval_stages,
                        timing_totals=done_payload["timing_totals"],
                        outcome="completed",
                    )
                    return

                context_texts = [c["text"] for c in chunks]
                scores = [c["score"] for c in chunks]
                gen_start = time.perf_counter()
                token_count = 0
                generation_stages: list[dict] = []
                try:
                    for kind, text in _stream_llm(
                        processed_query,
                        context_texts,
                        scores,
                        stage_timings=generation_stages,
                        memory_context=mem_ctx_text,
                        memory_recent_turns=mem_recent_turns,
                    ):
                        if kind == "reasoning":
                            # Live chain-of-thought — surfaced as a distinct event so
                            # the UI shows the model "thinking". NOT counted as an answer
                            # token and NOT stored in memory (it is not the answer).
                            yield _sse("reasoning", {"text": text})
                            await asyncio.sleep(0)
                            continue
                        token_count += 1
                        generated_text_parts.append(text)
                        yield _sse("token", {"token": text})
                        await asyncio.sleep(0)
                except Exception as exc:
                    logger.warning("Generation stream error: %s", exc)
                    stream_error_message = str(exc)
                    yield _sse("error", {"message": f"Generation error: {exc}"})

                gen_ms = (time.perf_counter() - gen_start) * 1000
                # Suggested follow-up questions (advisory, fail-open). The answer
                # has already fully streamed; these chips appear a moment later,
                # grounded in the answer + retrieved section headings.
                _assistant_text = "".join(generated_text_parts).strip()
                suggested_questions: list[str] = []
                if RAG_SUGGESTED_QUESTIONS_ENABLED and _assistant_text and not stream_error_message:
                    try:
                        from src.platform.llm import get_llm_provider
                        suggested_questions = await generate_follow_ups_from_config(
                            get_llm_provider(),
                            question=request.query,
                            answer=_assistant_text,
                            headings=_chunk_headings(chunks),
                        )
                    except Exception as exc:  # noqa: BLE001 — advisory, never break the stream
                        logger.warning("stream follow-up generation failed: %s", exc)
                total_ms = (time.perf_counter() - start) * 1000
                REQUESTS_TOTAL.labels(endpoint="/query/stream", method="POST", status="200").inc()
                REQUEST_LATENCY_MS.labels(endpoint="/query/stream", method="POST").observe(total_ms)
                all_stages = retrieval_stages + generation_stages
                stage_totals = _aggregate_stage_totals(all_stages)
                yield _sse(
                    "done",
                    {
                        "latency_ms": round(total_ms, 1),
                        "retrieval_ms": round(retrieval_ms, 1),
                        "generation_ms": round(gen_ms, 1),
                        "token_count": token_count,
                        "stage_timings": all_stages,
                        "timing_totals": stage_totals,
                        "conversation_id": conv.conversation_id,
                        "token_budget": _stream_token_budget,
                        "suggested_questions": suggested_questions,
                    },
                )
                if request.memory_enabled and request.mode != "retrieval":
                    mem_start = time.perf_counter()
                    memory.append_turn(
                        tenant_id=tenant_id,
                        subject=principal.subject,
                        project_id=principal.project_id,
                        conversation_id=conv.conversation_id,
                        role="user",
                        content=request.query,
                        query_id=workflow_id,
                    )
                    assistant_text = "".join(generated_text_parts).strip()
                    if assistant_text:
                        memory.append_turn(
                            tenant_id=tenant_id,
                            subject=principal.subject,
                            project_id=principal.project_id,
                            conversation_id=conv.conversation_id,
                            role="assistant",
                            content=assistant_text,
                            query_id=workflow_id,
                            sources=_source_refs(chunks),
                        )
                    MEMORY_OP_MS.labels(operation="append_turn").observe(
                        (time.perf_counter() - mem_start) * 1000
                    )
                    mem_start = time.perf_counter()
                    memory.compact_if_needed(
                        tenant_id=tenant_id,
                        subject=principal.subject,
                        project_id=principal.project_id,
                        conversation_id=conv.conversation_id,
                        force=request.compact_now,
                    )
                    MEMORY_OP_MS.labels(operation="compact").observe(
                        (time.perf_counter() - mem_start) * 1000
                    )
                    MEMORY_SUMMARY_TRIGGERS.labels(
                        reason="manual" if request.compact_now else "threshold"
                    ).inc()
                emit_stream_observability(
                    workflow_id=workflow_id,
                    request=request,
                    retrieval_ms=retrieval_ms,
                    generation_ms=gen_ms,
                    latency_ms=total_ms,
                    token_count=token_count,
                    stage_timings=all_stages,
                    timing_totals=stage_totals,
                    outcome="error" if stream_error_message else "completed",
                    error_message=stream_error_message,
                )
            finally:
                release_request_slot(slot_acquired)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.get(
        "/conversations",
        response_model=list[ConversationMetaResponse],
        responses=standard_error_responses,
    )
    async def list_conversations(
        limit: int = 50,
        principal: Principal = Depends(authenticate_request),
    ):
        require_role(principal, "query")
        memory = get_conversation_memory()
        items = memory.list_conversations(
            tenant_id=principal.tenant_id,
            subject=principal.subject,
            project_id=principal.project_id,
            limit=max(1, min(limit, 100)),
        )
        return [ConversationMetaResponse(**conversation_meta_to_dict(item)) for item in items]

    @router.post(
        "/conversations/new",
        response_model=ConversationMetaResponse,
        responses=standard_error_responses,
    )
    async def new_conversation(
        payload: ConversationCreateRequest,
        principal: Principal = Depends(authenticate_request),
    ):
        require_role(principal, "query")
        memory = get_conversation_memory()
        item = memory.ensure_conversation(
            tenant_id=principal.tenant_id,
            subject=principal.subject,
            project_id=principal.project_id,
            conversation_id=payload.conversation_id,
            title=payload.title,
        )
        return ConversationMetaResponse(**conversation_meta_to_dict(item))

    @router.get(
        "/conversations/{conversation_id}/history",
        response_model=ConversationHistoryResponse,
        responses=standard_error_responses,
    )
    async def conversation_history(
        conversation_id: str,
        limit: int = 100,
        principal: Principal = Depends(authenticate_request),
    ):
        require_role(principal, "query")
        memory = get_conversation_memory()
        turns = memory.get_turns(
            tenant_id=principal.tenant_id,
            subject=principal.subject,
            project_id=principal.project_id,
            conversation_id=conversation_id,
            limit=max(1, min(limit, 300)),
        )
        return ConversationHistoryResponse(
            conversation_id=conversation_id,
            turns=conversation_turns_to_dict(turns),
        )

    @router.post(
        "/conversations/{conversation_id}/compact",
        responses=standard_error_responses,
    )
    async def compact_conversation(
        conversation_id: str,
        payload: ConversationCompactRequest | None = Body(None),
        principal: Principal = Depends(authenticate_request),
    ):
        require_role(principal, "query")
        target_id = payload.conversation_id if payload else conversation_id
        memory = get_conversation_memory()
        summary = await asyncio.to_thread(
            memory.compact_if_needed,
            tenant_id=principal.tenant_id,
            subject=principal.subject,
            project_id=principal.project_id,
            conversation_id=target_id,
            force=True,
        )
        return {"conversation_id": target_id, "summary": summary.text, "updated_at_ms": summary.updated_at_ms}

    @router.delete(
        "/conversations/{conversation_id}",
        responses=standard_error_responses,
    )
    async def delete_conversation(
        conversation_id: str,
        principal: Principal = Depends(authenticate_request),
    ):
        require_role(principal, "query")
        memory = get_conversation_memory()
        deleted = memory.delete_conversation(
            tenant_id=principal.tenant_id,
            subject=principal.subject,
            project_id=principal.project_id,
            conversation_id=conversation_id,
        )
        return {"conversation_id": conversation_id, "deleted": deleted}

    @router.patch(
        "/conversations/{conversation_id}",
        response_model=ConversationMetaResponse,
        responses=standard_error_responses,
    )
    async def update_conversation(
        conversation_id: str,
        payload: ConversationTitleUpdateRequest,
        principal: Principal = Depends(authenticate_request),
    ):
        require_role(principal, "query")
        memory = get_conversation_memory()
        meta = memory.update_conversation_title(
            tenant_id=principal.tenant_id,
            subject=principal.subject,
            project_id=principal.project_id,
            conversation_id=conversation_id,
            title=payload.title,
        )
        if meta is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return ConversationMetaResponse(**conversation_meta_to_dict(meta))

    @router.get("/metrics")
    async def metrics():
        payload, content_type = render_metrics()
        return Response(content=payload, media_type=content_type)

    return router


__all__ = ["create_query_router", "run_query"]
