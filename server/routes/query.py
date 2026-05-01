# @summary
# Query routes for standard and streaming retrieval endpoints with Temporal orchestration.
# Exports: create_query_router, run_query
# Deps: fastapi, temporalio, server.schemas, src.platform.security, server.workflows, src.platform.llm
# @end-summary
"""Query API routes."""

from __future__ import annotations

import asyncio
import orjson as json_mod
import logging
import time
import uuid
from typing import Awaitable, Callable

from fastapi import APIRouter, Body, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from temporalio.client import Client  # pyright: ignore[reportMissingImports]
from temporalio.service import RPCError  # pyright: ignore[reportMissingImports]

from config.settings import (
    GENERATION_MAX_TOKENS,
    GENERATION_TEMPERATURE,
)
from server.schemas import ApiErrorResponse, QueryRequest, QueryResponse
from server.schemas import (
    ConversationCompactRequest,
    ConversationCreateRequest,
    ConversationHistoryResponse,
    ConversationMetaResponse,
    ConversationTitleUpdateRequest,
)
from server.workflows import RAG_QUERY_TASK_QUEUE, RAGQueryWorkflow
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


def _derive_title_from_query(query: str, *, max_len: int = 60) -> str:
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


def _source_refs(results: list) -> list[dict]:
    """Extract lightweight source references from RAG results (no full chunk text)."""
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
            "text": text[:400] if text else "",
        }
        start = meta.get("original_char_start")
        end = meta.get("original_char_end")
        if start is not None:
            ref["original_char_start"] = start
        if end is not None:
            ref["original_char_end"] = end
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
    """Stream generation tokens via LLMProvider (provider-agnostic).

    Delegates prompt assembly to `OllamaGenerator.build_messages` so the
    streaming and non-streaming paths share a single source of truth for
    prompt shape (system prompt, doc-context layout, memory framing).
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
    for token in provider.generate_stream(
        messages,
        model_alias="default",
        temperature=GENERATION_TEMPERATURE,
        max_tokens=GENERATION_MAX_TOKENS,
    ):
        yield token
    _record_stage("stream_tokens", "generation", stream_start)


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
) -> QueryResponse:
    """Execute non-stream query workflow and return API response model."""
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
) -> APIRouter:
    """Create router for query, query-stream, and metrics endpoints."""
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
        )

    @router.post("/query/stream", responses=standard_error_responses)
    async def query_stream(request: QueryRequest, principal: Principal = Depends(authenticate_request)):
        temporal_client = get_temporal_client()
        if temporal_client is None:
            raise HTTPException(status_code=503, detail="Temporal client not connected")
        require_role(principal, "query")
        enforce_rate_limit(principal, "/query/stream")
        slot_acquired = await acquire_request_slot("/query/stream")
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
                    for token in _stream_llm(
                        processed_query,
                        context_texts,
                        scores,
                        stage_timings=generation_stages,
                        memory_context=mem_ctx_text,
                        memory_recent_turns=mem_recent_turns,
                    ):
                        token_count += 1
                        generated_text_parts.append(token)
                        yield _sse("token", {"token": token})
                        await asyncio.sleep(0)
                except Exception as exc:
                    logger.warning("Generation stream error: %s", exc)
                    stream_error_message = str(exc)
                    yield _sse("error", {"message": f"Generation error: {exc}"})

                gen_ms = (time.perf_counter() - gen_start) * 1000
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
