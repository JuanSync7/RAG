# @summary
# Console routes for dual web console (User Console at /console, Admin Console at /console/admin).
# Includes UI page serving, query wrapper, ingestion wrapper, conversation management, and source preview.
# Exports: create_console_router
# Deps: fastapi, server.schemas, server.console.services
# @end-summary
"""Console route module — dual-console architecture."""

from __future__ import annotations

import asyncio
import json
import logging
from html import escape
from pathlib import Path
from typing import Awaitable, Callable

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from temporalio.client import Client  # pyright: ignore[reportMissingImports]

from config.settings import RAG_CONSOLE_PROVENANCE_TRUST_THRESHOLD
from server.common import ApiErrorResponse
from server.console.services import (
    CONSOLE_HTML_PATH,
    USER_CONSOLE_HTML_PATH,
    build_source_preview_payload,
    is_ollama_reachable,
    read_clean_document_from_minio,
    render_source_document_html,
    resolve_console_static_asset,
    is_remote_view_uri,
    resolve_console_source_path,
    resolve_user_console_static_asset,
    tail_log_lines,
)
from server.routes import build_health_response, run_query
from server.routes import (
    create_api_key_handler,
    list_api_keys_handler,
    list_quotas_handler,
)
from server.schemas import (
    ConversationCompactRequest,
    ConversationCreateRequest,
    ConversationHistoryResponse,
    ConversationTitleUpdateRequest,
    ConsoleCommandRequest,
    ConsoleEnvelope,
    ConsoleFeedbackRequest,
    ConsoleHealthSummary,
    ConsoleModelInfo,
    ConsoleSourceViewRequest,
    ConsoleIngestionRequest,
    ConsoleQueryRequest,
    CreateApiKeyRequest,
    QueryRequest,
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
from src.platform.security import require_role
from src.platform import (
    MODE_CONSOLE_INGEST,
    MODE_CONSOLE_QUERY,
    get_command_spec,
    list_command_specs,
    to_payload,
)


_PROVENANCE_TRUST_THRESHOLD = RAG_CONSOLE_PROVENANCE_TRUST_THRESHOLD


def _resolve_highlight_range(
    *,
    text: str,
    primary_start: int | None,
    primary_end: int | None,
    confidence: float,
    chunk_text: str | None,
) -> tuple[int | None, int | None]:
    """Pick a highlight range in `text` for the cited chunk.

    Order of preference:
    1. Trust offsets when confidence >= threshold and they look valid.
    2. Substring-locate `chunk_text` in `text` (always exact when it matches).
    3. Give up — return (None, None); viewer will skip the highlight.
    """
    if (
        primary_start is not None
        and primary_end is not None
        and primary_start >= 0
        and primary_end > primary_start
        and confidence >= _PROVENANCE_TRUST_THRESHOLD
    ):
        return primary_start, primary_end

    if chunk_text:
        needle = chunk_text.strip()
        if needle:
            idx = text.find(needle)
            if idx >= 0:
                return idx, idx + len(needle)
            # Loose retry: collapse whitespace on both sides for resilience.
            import re

            normalised_text = re.sub(r"\s+", " ", text)
            normalised_needle = re.sub(r"\s+", " ", needle)
            idx = normalised_text.find(normalised_needle)
            if idx >= 0:
                # Map back to original-text indices by counting whitespace runs.
                # Cheap approximation: walk text and count condensed chars.
                target = idx
                consumed = 0
                start_in_text = 0
                in_ws = False
                for i, ch in enumerate(text):
                    if consumed >= target:
                        start_in_text = i
                        break
                    if ch.isspace():
                        if not in_ws:
                            consumed += 1
                            in_ws = True
                    else:
                        consumed += 1
                        in_ws = False
                end_in_text = min(start_in_text + len(needle), len(text))
                return start_in_text, end_in_text

    return None, None


def create_console_router(
    *,
    get_temporal_client: Callable[[], Client | None],
    logger: logging.Logger,
    enforce_rate_limit: Callable[[Principal, str], None],
    acquire_request_slot: Callable[[str], Awaitable[bool]],
    release_request_slot: Callable[[bool], None],
    console_ok: Callable[[Request, dict], ConsoleEnvelope],
) -> APIRouter:
    """Create web console router."""
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

    # --- User Console (modern chat interface at /console) ---
    # NOTE: User Console static route (/console/static/user/) must be registered
    # before the general static route (/console/static/) to avoid catch-all matching.

    @router.get("/console", response_class=HTMLResponse)
    async def user_console_ui():
        if not USER_CONSOLE_HTML_PATH.exists():
            raise HTTPException(status_code=404, detail="User Console UI file not found")
        return HTMLResponse(
            USER_CONSOLE_HTML_PATH.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @router.get("/console/static/user/{asset_path:path}")
    async def user_console_static_asset(asset_path: str):
        asset = resolve_user_console_static_asset(asset_path)
        return FileResponse(
            path=asset,
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    # --- Admin Console (tabbed debug/ops interface at /console/admin) ---

    @router.get("/console/admin", response_class=HTMLResponse)
    async def admin_console_ui():
        if not CONSOLE_HTML_PATH.exists():
            raise HTTPException(status_code=404, detail="Admin Console UI file not found")
        return HTMLResponse(
            CONSOLE_HTML_PATH.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @router.get("/console/static/{asset_path:path}")
    async def console_static_asset(asset_path: str):
        asset = resolve_console_static_asset(asset_path)
        return FileResponse(
            path=asset,
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @router.get("/console/health", response_model=ConsoleEnvelope, responses=standard_error_responses)
    async def console_health(request: Request, principal: Principal = Depends(authenticate_request)):
        require_role(principal, "query")
        base = await build_health_response(get_temporal_client(), logger)
        summary = ConsoleHealthSummary(
            status=base.status,
            temporal_connected=base.temporal_connected,
            worker_available=base.worker_available,
            ingest_worker_available=base.ingest_worker_available,
            ollama_reachable=is_ollama_reachable(),
        )
        return console_ok(request, summary.model_dump())

    @router.get("/console/model-info", response_model=ConsoleEnvelope, responses=standard_error_responses)
    async def console_model_info(request: Request, principal: Principal = Depends(authenticate_request)):
        require_role(principal, "query")
        from config import settings as _settings

        gen_enabled = bool(getattr(_settings, "GENERATION_ENABLED", True))

        raw_model = ""
        try:
            from src.platform.token_budget.provider import get_capabilities

            caps = get_capabilities()
            raw_model = str(caps.model_name or "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("model-info: capabilities lookup failed: %s", exc)

        if not raw_model:
            raw_model = str(getattr(_settings, "LLM_MODEL", "") or "")

        if "/" in raw_model:
            provider, _, model_name = raw_model.partition("/")
        else:
            provider, model_name = ("", raw_model)
        if not provider and model_name.startswith("claude"):
            provider = "anthropic"
        elif not provider and (model_name.startswith("gpt") or model_name.startswith("o1") or model_name.startswith("o3")):
            provider = "openai"
        display = model_name or raw_model or "unknown"
        info = ConsoleModelInfo(
            generation_enabled=gen_enabled,
            provider=provider,
            model=model_name or raw_model,
            display=display,
        )
        return console_ok(request, info.model_dump())

    @router.post("/console/feedback", response_model=ConsoleEnvelope, responses=standard_error_responses)
    async def console_feedback(
        request: Request,
        payload: ConsoleFeedbackRequest,
        principal: Principal = Depends(authenticate_request),
    ):
        """Accept thumbs up/down feedback for an assistant turn.

        Stub endpoint: validates and logs the rating; persistence is wired in a
        follow-up. The frontend treats a 200 as acknowledgement.
        """
        require_role(principal, "query")
        logger.info(
            "console feedback: rating=%s conv=%s idx=%s principal=%s",
            payload.rating,
            payload.conversation_id or "-",
            payload.message_index if payload.message_index is not None else "-",
            getattr(principal, "name", "?"),
        )
        return console_ok(request, {"accepted": True, "rating": payload.rating})

    @router.get("/console/logs", response_model=ConsoleEnvelope, responses=standard_error_responses)
    async def console_logs(
        request: Request,
        lines: int = 120,
        principal: Principal = Depends(authenticate_request),
    ):
        require_role(principal, "query")
        snapshot = tail_log_lines(lines=max(10, min(lines, 500)))
        return console_ok(request, snapshot.model_dump())

    @router.get("/console/commands", response_model=ConsoleEnvelope, responses=standard_error_responses)
    async def console_commands(
        request: Request,
        mode: str = "query",
        principal: Principal = Depends(authenticate_request),
    ):
        require_role(principal, "query")
        normalized_mode = mode.strip().lower()
        if normalized_mode == "query":
            catalog_mode = MODE_CONSOLE_QUERY
        elif normalized_mode == "ingest":
            catalog_mode = MODE_CONSOLE_INGEST
        else:
            raise HTTPException(status_code=400, detail="mode must be one of: query, ingest")
        specs = list_command_specs(catalog_mode)
        return console_ok(
            request,
            {
                "mode": normalized_mode,
                "commands": to_payload(specs),
            },
        )

    @router.post("/console/command", response_model=ConsoleEnvelope, responses=standard_error_responses)
    async def console_command(
        request: Request,
        payload: ConsoleCommandRequest,
        principal: Principal = Depends(authenticate_request),
    ):
        require_role(principal, "query")
        mode = payload.mode.strip().lower()
        command_name = payload.command.strip().lstrip("/").strip().lower()
        if not command_name:
            raise HTTPException(status_code=400, detail="command is required")
        if mode == "query":
            catalog_mode = MODE_CONSOLE_QUERY
        elif mode == "ingest":
            catalog_mode = MODE_CONSOLE_INGEST
        else:
            raise HTTPException(status_code=400, detail="mode must be one of: query, ingest")

        spec = get_command_spec(catalog_mode, command_name)
        if spec is None:
            raise HTTPException(status_code=400, detail=f"Unknown command: /{command_name}")

        intent = spec.intent or ""
        action = "noop"
        message = ""
        data: dict = {}
        state = payload.state or {}
        if intent == "show_help":
            action = "render_help"
            data = {"commands": to_payload(list_command_specs(catalog_mode))}
        elif intent == "show_status":
            action = "render_status"
            data = {"state": state}
        elif intent == "show_health":
            action = "refresh_health"
            base = await build_health_response(get_temporal_client(), logger)
            from config import settings as _settings

            ollama_ok = is_ollama_reachable()
            llm_model = str(getattr(_settings, "LLM_MODEL", "") or "")
            uses_ollama = llm_model.startswith("ollama/") or not llm_model
            generation_enabled = bool(getattr(_settings, "GENERATION_ENABLED", True))

            if not generation_enabled:
                generation_state = "disabled"
            elif uses_ollama:
                generation_state = "ready" if ollama_ok else "unreachable"
            else:
                # Cloud provider (anthropic/openai/openrouter/...): we don't
                # ping it here, so just report it's configured.
                generation_state = "configured"

            jobs_ok = base.temporal_connected and base.ingest_worker_available
            jobs_state = "ready" if jobs_ok else (
                "no worker" if base.temporal_connected else "offline"
            )

            data = {
                "health": {
                    "status": base.status,
                    "server": "up",
                    "generation": generation_state,
                    "background_jobs": jobs_state,
                }
            }
        elif intent == "clear_view":
            action = "clear_view"
        elif intent == "run":
            action = "run_stream_query" if mode == "query" else "run_ingest"
        elif intent == "run_non_stream":
            action = "run_non_stream_query"
        elif intent == "list_conversations":
            action = "list_conversations"
            memory = get_conversation_memory()
            items = memory.list_conversations(
                tenant_id=principal.tenant_id,
                subject=principal.subject,
                project_id=principal.project_id,
                limit=50,
            )
            data = {"conversations": [conversation_meta_to_dict(item) for item in items]}
        elif intent == "new_conversation":
            action = "new_conversation"
            title = (payload.arg or str(state.get("title", "New conversation"))).strip()
            memory = get_conversation_memory()
            item = memory.ensure_conversation(
                tenant_id=principal.tenant_id,
                subject=principal.subject,
                project_id=principal.project_id,
                title=title,
            )
            data = {"conversation": conversation_meta_to_dict(item)}
        elif intent == "switch_conversation":
            action = "switch_conversation"
            target = (payload.arg or str(state.get("conversation_id", ""))).strip()
            if not target:
                message = "conversation id is required for /switch"
            else:
                data = {"conversation_id": target}
        elif intent == "show_history":
            action = "show_history"
            conversation_id = str(state.get("conversation_id", "")).strip()
            if not conversation_id:
                message = "conversation_id is required for /history"
            else:
                limit = 100
                if payload.arg and payload.arg.strip().isdigit():
                    limit = max(1, min(int(payload.arg.strip()), 300))
                memory = get_conversation_memory()
                turns = memory.get_turns(
                    tenant_id=principal.tenant_id,
                    subject=principal.subject,
                    project_id=principal.project_id,
                    conversation_id=conversation_id,
                    limit=limit,
                )
                data = {"conversation_id": conversation_id, "turns": conversation_turns_to_dict(turns)}
        elif intent == "compact_conversation":
            action = "compact_conversation"
            conversation_id = str(state.get("conversation_id", "")).strip()
            if not conversation_id:
                message = "conversation_id is required for /compact"
            else:
                memory = get_conversation_memory()
                summary = await asyncio.to_thread(
                    memory.compact_if_needed,
                    tenant_id=principal.tenant_id,
                    subject=principal.subject,
                    project_id=principal.project_id,
                    conversation_id=conversation_id,
                    force=True,
                )
                data = {
                    "conversation_id": conversation_id,
                    "summary": summary.text,
                    "updated_at_ms": summary.updated_at_ms,
                }
        elif intent == "delete_conversation":
            action = "delete_conversation"
            target = (payload.arg or str(state.get("conversation_id", ""))).strip()
            if not target:
                message = "conversation_id is required for /delete"
            else:
                memory = get_conversation_memory()
                deleted = memory.delete_conversation(
                    tenant_id=principal.tenant_id,
                    subject=principal.subject,
                    project_id=principal.project_id,
                    conversation_id=target,
                )
                data = {"conversation_id": target, "deleted": deleted}
        else:
            message = "Command is recognized but not supported by web console action handler yet."

        return console_ok(
            request,
            {
                "mode": mode,
                "command": command_name,
                "arg": payload.arg or "",
                "intent": intent,
                "action": action,
                "message": message,
                "data": data,
            },
        )

    @router.get("/console/source-document", response_model=ConsoleEnvelope, responses=standard_error_responses)
    async def console_source_document(
        request: Request,
        source: str | None = None,
        source_uri: str | None = None,
        start: int | None = None,
        end: int | None = None,
        context_chars: int = 700,
        max_chars: int = 5000,
        principal: Principal = Depends(authenticate_request),
    ):
        require_role(principal, "query")
        target = resolve_console_source_path(source, source_uri)
        text = target.read_text(encoding="utf-8", errors="replace")
        payload = build_source_preview_payload(
            target=target,
            source_uri=source_uri,
            text=text,
            start=start,
            end=end,
            context_chars=context_chars,
            max_chars=max_chars,
        )
        return console_ok(request, payload)

    @router.post("/console/source-document/view", response_class=HTMLResponse, responses=standard_error_responses)
    async def console_source_document_view(
        payload: ConsoleSourceViewRequest,
        principal: Principal = Depends(authenticate_request),
    ):
        """Render the source document with the cited chunk highlighted.

        Coordinate strategy:
        - When MinIO clean store serves the doc → use refactored offsets (exact).
        - When the raw file serves the doc → use original offsets if confidence
          is high; otherwise locate via substring search on chunk_text.
        """
        require_role(principal, "query")

        source = payload.source
        source_uri = payload.source_uri
        source_key = payload.source_key
        chunk_text = payload.chunk_text
        chunk = payload.chunk
        confidence = payload.provenance_confidence or 0.0

        # Prefer MinIO clean store: works for any source type (incl. binary docx/pptx)
        # because the worker writes a clean markdown rendering during ingest.
        minio_error: HTTPException | None = None
        if source_key:
            try:
                text, meta = read_clean_document_from_minio(source_key)
                display_name = str(meta.get("source") or source or source_key)
                start, end = _resolve_highlight_range(
                    text=text,
                    primary_start=payload.refactored_start,
                    primary_end=payload.refactored_end,
                    confidence=confidence,
                    chunk_text=chunk_text,
                )
                html = render_source_document_html(
                    target=Path(display_name),
                    text=text,
                    start=start,
                    end=end,
                    chunk=chunk,
                    is_markdown=True,
                    title=display_name,
                    chunk_text=chunk_text,
                )
                return HTMLResponse(content=html)
            except HTTPException as exc:
                minio_error = exc

        # Remote-origin redirect (SharePoint/Drive/etc.): when the source URI is
        # http(s) and the operator has opted in via RAG_CONSOLE_REMOTE_VIEW_ENABLED,
        # 302 to the origin so SSO handles auth/render. We only reach here when
        # MinIO did not have a clean rendering, so origin is the best fallback.
        remote = is_remote_view_uri(source_uri)
        if remote:
            return RedirectResponse(url=remote, status_code=302)

        # Fallback: raw file from an allowed source root.
        try:
            target = resolve_console_source_path(source, source_uri)
        except HTTPException as exc:
            detail = exc.detail
            if minio_error is not None:
                detail = (
                    f"{detail} | MinIO clean store: {minio_error.detail} "
                    f"(source_key={source_key!r})"
                )
            raise HTTPException(status_code=exc.status_code, detail=detail) from exc

        ext = target.suffix.lower()

        if ext == ".pdf":
            return FileResponse(path=str(target), media_type="application/pdf", filename=target.name)

        if ext in {".docx", ".pptx", ".xlsx", ".doc", ".ppt", ".xls"}:
            html = (
                "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                "<title>Source unavailable</title>"
                "<style>body{font-family:system-ui,sans-serif;background:#0f1115;color:#e8e8ec;"
                "padding:32px;line-height:1.6}code{background:#1a1d24;padding:2px 6px;border-radius:4px}</style>"
                "</head><body>"
                f"<h2>{escape(target.name)} is a binary office document</h2>"
                "<p>This format can only be viewed via the MinIO clean-markdown rendering "
                "produced during ingest. The clean store does not currently hold this document "
                f"(<code>source_key={escape(str(source_key or ''))}</code>).</p>"
                "<p>To fix: re-ingest this file with the worker (which writes the clean markdown to MinIO), "
                "or open the original file directly.</p>"
                "</body></html>"
            )
            return HTMLResponse(content=html)

        text = target.read_text(encoding="utf-8", errors="replace")
        is_md = ext in {".md", ".markdown", ".mdx"}
        start, end = _resolve_highlight_range(
            text=text,
            primary_start=payload.original_start,
            primary_end=payload.original_end,
            confidence=confidence,
            chunk_text=chunk_text,
        )
        html = render_source_document_html(
            target=target, text=text, start=start, end=end, chunk=chunk,
            is_markdown=is_md, chunk_text=chunk_text,
        )
        return HTMLResponse(content=html)

    @router.get("/console/source-document/raw", responses=standard_error_responses)
    async def console_source_document_raw(
        source: str | None = None,
        source_uri: str | None = None,
        principal: Principal = Depends(authenticate_request),
    ):
        """Stream raw bytes for a source document. Used by the PDF iframe preview."""
        require_role(principal, "query")
        remote = is_remote_view_uri(source_uri)
        if remote:
            return RedirectResponse(url=remote, status_code=302)
        target = resolve_console_source_path(source, source_uri)
        ext = target.suffix.lower()
        media_type = {
            ".pdf": "application/pdf",
            ".md": "text/markdown; charset=utf-8",
            ".txt": "text/plain; charset=utf-8",
            ".json": "application/json",
            ".html": "text/html; charset=utf-8",
            ".csv": "text/csv; charset=utf-8",
        }.get(ext, "application/octet-stream")
        return FileResponse(path=str(target), media_type=media_type, filename=target.name)

    @router.post("/console/query", response_model=ConsoleEnvelope, responses=standard_error_responses)
    async def console_query(
        request: Request,
        payload: ConsoleQueryRequest,
        principal: Principal = Depends(authenticate_request),
    ):
        require_role(principal, "query")
        query_request = QueryRequest(**payload.model_dump(exclude={"stream"}))
        result = await run_query(
            query_request,
            principal,
            endpoint="/query",
            temporal_client=get_temporal_client(),
            require_role=require_role,
            enforce_rate_limit=enforce_rate_limit,
            acquire_request_slot=acquire_request_slot,
            release_request_slot=release_request_slot,
            logger=logger,
        )
        # Turn-loop requests return a JSONResponse (QueryResponse-shaped body
        # plus metadata.turn_loop) instead of the QueryResponse model —
        # unwrap it into the console envelope so the admin surface works
        # whenever the loop is active (per-request or env default).
        if isinstance(result, JSONResponse):
            return console_ok(request, json.loads(bytes(result.body)))
        return console_ok(request, result.model_dump())

    @router.get(
        "/console/conversations",
        response_model=ConsoleEnvelope,
        responses=standard_error_responses,
    )
    async def console_conversations(
        request: Request,
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
        payload = {"conversations": [conversation_meta_to_dict(item) for item in items]}
        return console_ok(request, payload)

    @router.post(
        "/console/conversations/new",
        response_model=ConsoleEnvelope,
        responses=standard_error_responses,
    )
    async def console_new_conversation(
        request: Request,
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
        return console_ok(request, {"conversation": conversation_meta_to_dict(item)})

    @router.get(
        "/console/conversations/{conversation_id}/history",
        response_model=ConsoleEnvelope,
        responses=standard_error_responses,
    )
    async def console_conversation_history(
        request: Request,
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
        payload = ConversationHistoryResponse(
            conversation_id=conversation_id,
            turns=conversation_turns_to_dict(turns),
        ).model_dump()
        return console_ok(request, payload)

    @router.post(
        "/console/conversations/{conversation_id}/compact",
        response_model=ConsoleEnvelope,
        responses=standard_error_responses,
    )
    async def console_conversation_compact(
        request: Request,
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
        return console_ok(
            request,
            {
                "conversation_id": target_id,
                "summary": summary.text,
                "updated_at_ms": summary.updated_at_ms,
            },
        )

    @router.delete(
        "/console/conversations/{conversation_id}",
        response_model=ConsoleEnvelope,
        responses=standard_error_responses,
    )
    async def console_conversation_delete(
        request: Request,
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
        return console_ok(request, {"conversation_id": conversation_id, "deleted": deleted})

    @router.patch(
        "/console/conversations/{conversation_id}",
        response_model=ConsoleEnvelope,
        responses=standard_error_responses,
    )
    async def console_conversation_update(
        request: Request,
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
        return console_ok(request, {"conversation": conversation_meta_to_dict(meta)})

    @router.get(
        "/console/conversations/{conversation_id}/doc-state",
        response_model=ConsoleEnvelope,
        responses=standard_error_responses,
    )
    async def console_conversation_doc_state(
        request: Request,
        conversation_id: str,
        principal: Principal = Depends(authenticate_request),
    ):
        require_role(principal, "query")
        memory = get_conversation_memory()
        meta = memory.ensure_conversation(
            tenant_id=principal.tenant_id,
            subject=principal.subject,
            project_id=principal.project_id,
            conversation_id=conversation_id,
        )
        return console_ok(
            request,
            {
                "conversation_id": conversation_id,
                "relevant_doc_ids": list(meta.relevant_doc_ids),
                "ignored_doc_ids": list(meta.ignored_doc_ids),
            },
        )

    @router.post(
        "/console/conversations/{conversation_id}/ignore",
        response_model=ConsoleEnvelope,
        responses=standard_error_responses,
    )
    async def console_conversation_ignore(
        request: Request,
        conversation_id: str,
        payload: dict = Body(...),
        principal: Principal = Depends(authenticate_request),
    ):
        require_role(principal, "query")
        raw_ids = payload.get("doc_ids")
        if isinstance(raw_ids, list):
            doc_ids = [str(d).strip() for d in raw_ids if str(d).strip()]
        else:
            single = payload.get("doc_id")
            doc_ids = [str(single).strip()] if single and str(single).strip() else []
        if not doc_ids:
            raise HTTPException(status_code=400, detail="doc_id or doc_ids required")
        memory = get_conversation_memory()
        meta = None
        for doc_id in doc_ids:
            meta = memory.move_to_ignored(
                tenant_id=principal.tenant_id,
                subject=principal.subject,
                project_id=principal.project_id,
                conversation_id=conversation_id,
                doc_id=doc_id,
            )
        relevant = list(meta.relevant_doc_ids) if meta else []
        ignored = list(meta.ignored_doc_ids) if meta else []
        return console_ok(
            request,
            {
                "conversation_id": conversation_id,
                "relevant_doc_ids": relevant,
                "ignored_doc_ids": ignored,
            },
        )

    @router.delete(
        "/console/conversations/{conversation_id}/ignore/{doc_id}",
        response_model=ConsoleEnvelope,
        responses=standard_error_responses,
    )
    async def console_conversation_restore(
        request: Request,
        conversation_id: str,
        doc_id: str,
        principal: Principal = Depends(authenticate_request),
    ):
        require_role(principal, "query")
        memory = get_conversation_memory()
        meta = memory.restore_to_relevant(
            tenant_id=principal.tenant_id,
            subject=principal.subject,
            project_id=principal.project_id,
            conversation_id=conversation_id,
            doc_id=doc_id,
        )
        return console_ok(
            request,
            {
                "conversation_id": conversation_id,
                "relevant_doc_ids": list(meta.relevant_doc_ids),
                "ignored_doc_ids": list(meta.ignored_doc_ids),
            },
        )

    @router.delete(
        "/console/conversations/{conversation_id}/doc-state/{doc_id}",
        response_model=ConsoleEnvelope,
        responses=standard_error_responses,
    )
    async def console_conversation_clear_doc_state(
        request: Request,
        conversation_id: str,
        doc_id: str,
        principal: Principal = Depends(authenticate_request),
    ):
        require_role(principal, "query")
        memory = get_conversation_memory()
        meta = memory.clear_doc_state(
            tenant_id=principal.tenant_id,
            subject=principal.subject,
            project_id=principal.project_id,
            conversation_id=conversation_id,
            doc_id=doc_id,
        )
        return console_ok(
            request,
            {
                "conversation_id": conversation_id,
                "relevant_doc_ids": list(meta.relevant_doc_ids),
                "ignored_doc_ids": list(meta.ignored_doc_ids),
            },
        )

    @router.post("/console/ingest", response_model=ConsoleEnvelope, responses=standard_error_responses)
    async def console_ingest(
        request: Request,
        payload: ConsoleIngestionRequest,
        principal: Principal = Depends(authenticate_request),
    ):
        require_role(principal, "admin")
        from config.settings import DOCUMENTS_DIR, PROJECT_ROOT
        from src.ingest import IngestionConfig, ingest_directory
        from src.platform import validate_documents_dir

        selected_file = None
        documents_dir = DOCUMENTS_DIR
        if payload.mode == "single_file":
            if not payload.target_path:
                raise HTTPException(status_code=400, detail="target_path is required for mode=single_file")
            selected_file = Path(payload.target_path).resolve()
            if not selected_file.exists() or not selected_file.is_file():
                raise HTTPException(status_code=400, detail="target_path is not a valid file")
            documents_dir = selected_file.parent
        elif payload.mode == "directory":
            if not payload.target_path:
                raise HTTPException(status_code=400, detail="target_path is required for mode=directory")
            documents_dir = validate_documents_dir(Path(payload.target_path), PROJECT_ROOT)

        cfg_kwargs = {
            "semantic_chunking": payload.semantic_chunking,
            "export_processed": payload.export_processed,
            "update_mode": payload.update_mode,
        }
        if payload.verbose_stages is not None:
            cfg_kwargs["verbose_stage_logs"] = payload.verbose_stages
        if payload.persist_refactor_mirror is not None:
            cfg_kwargs["persist_refactor_mirror"] = payload.persist_refactor_mirror
        if payload.docling_enabled is not None:
            cfg_kwargs["enable_docling_parser"] = payload.docling_enabled
        if payload.docling_model:
            cfg_kwargs["docling_model"] = payload.docling_model
        if payload.docling_artifacts_path is not None:
            cfg_kwargs["docling_artifacts_path"] = payload.docling_artifacts_path
        if payload.docling_strict is not None:
            cfg_kwargs["docling_strict"] = payload.docling_strict
        if payload.docling_auto_download is not None:
            cfg_kwargs["docling_auto_download"] = payload.docling_auto_download
        if payload.vision_enabled is not None:
            cfg_kwargs["enable_vision_processing"] = payload.vision_enabled
        if payload.vision_provider:
            cfg_kwargs["vision_provider"] = payload.vision_provider
        if payload.vision_model:
            cfg_kwargs["vision_model"] = payload.vision_model
        if payload.vision_api_base_url is not None:
            cfg_kwargs["vision_api_base_url"] = payload.vision_api_base_url
        if payload.vision_timeout_seconds is not None:
            cfg_kwargs["vision_timeout_seconds"] = payload.vision_timeout_seconds
        if payload.vision_max_figures is not None:
            cfg_kwargs["vision_max_figures"] = payload.vision_max_figures
        if payload.vision_auto_pull is not None:
            cfg_kwargs["vision_auto_pull"] = payload.vision_auto_pull
        if payload.vision_strict is not None:
            cfg_kwargs["vision_strict"] = payload.vision_strict

        cfg = IngestionConfig(**cfg_kwargs)
        selected_sources = [selected_file] if selected_file is not None else None
        summary = await asyncio.to_thread(
            ingest_directory,
            documents_dir=documents_dir,
            config=cfg,
            fresh=not payload.update_mode,
            update=payload.update_mode,
            selected_sources=selected_sources,
        )
        return console_ok(
            request,
            {
                "processed": summary.processed,
                "skipped": summary.skipped,
                "failed": summary.failed,
                "stored_chunks": summary.stored_chunks,
                "removed_sources": summary.removed_sources,
                "errors": summary.errors,
                "design_warnings": summary.design_warnings,
            },
        )

    @router.get("/console/admin/api-keys", response_model=ConsoleEnvelope, responses=standard_error_responses)
    async def console_admin_list_api_keys(
        request: Request,
        include_revoked: bool = False,
        principal: Principal = Depends(authenticate_request),
    ):
        records = await list_api_keys_handler(include_revoked, principal)
        return console_ok(request, {"api_keys": [record.model_dump() for record in records]})

    @router.post("/console/admin/api-keys", response_model=ConsoleEnvelope, responses=standard_error_responses)
    async def console_admin_create_api_key(
        request: Request,
        payload: CreateApiKeyRequest,
        principal: Principal = Depends(authenticate_request),
    ):
        created = await create_api_key_handler(payload, principal)
        return console_ok(request, created.model_dump())

    @router.get("/console/admin/quotas", response_model=ConsoleEnvelope, responses=standard_error_responses)
    async def console_admin_quotas(
        request: Request,
        principal: Principal = Depends(authenticate_request),
    ):
        quotas = await list_quotas_handler(principal)
        return console_ok(request, quotas.model_dump())

    return router


__all__ = ["create_console_router"]
