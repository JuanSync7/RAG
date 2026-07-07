# @summary
# FastAPI server that accepts RAG queries over HTTP and dispatches them to
# Temporal workers where models are preloaded. No models loaded in this process.
# Exports: app
# Deps: fastapi, temporalio, server.schemas, server.workflows, config.settings
# @end-summary
"""FastAPI server — the HTTP frontend for the RAG system.

This process is lightweight: no GPU models, no torch imports. It accepts
HTTP requests, starts Temporal workflows, and returns results.

Usage:
    uvicorn server.api:app --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from temporalio.client import Client  # pyright: ignore[reportMissingImports]

from config.settings import (
    TEMPORAL_TARGET_HOST,
    RAG_API_MAX_INFLIGHT_REQUESTS,
    RAG_API_OVERLOAD_QUEUE_TIMEOUT_MS,
    RAG_API_CORS_ORIGINS,
    RAG_WORKFLOW_DEFAULT_TIMEOUT_MS,
    RATE_LIMIT_ENABLED,
    RATE_LIMIT_REQUESTS_PER_MINUTE,
    RATE_LIMIT_WINDOW_SECONDS,
)
from src.platform.limits import InMemoryRateLimiter
from src.platform import (
    INFLIGHT_REQUESTS,
    OVERLOAD_REJECTS,
    RATE_LIMIT_REJECTS,
    REQUESTS_TOTAL,
)
from src.platform.observability import get_tracer
from src.platform.security import Principal
from src.platform.security import get_tenant_quota
from src.platform.security import require_role
from server.console import create_console_router
from server.observability_middleware import install_observability_middleware
from server.routes import (
    create_admin_router,
    create_documents_router,
    create_ingest_router,
    start_job_sweeper,
    create_query_router,
    create_system_router,
)
from server.utils import console_ok as _console_ok, error_payload as _error_payload, validate_optional_dependencies as _validate_optional_dependencies, validate_startup_config as _validate_startup_config
from config.settings import validate_all_config as _validate_all_config
from server.schemas import (
    QueryRequest,
)
import src.db as _db

# Use uvicorn's logger/formatter so API logs match server output
# (INFO prefix + colorized level formatting).
logger = logging.getLogger("uvicorn.error").getChild("rag.server.api")

_temporal_client: Client | None = None

# Document store clients — gracefully degrade if unavailable at startup.
# MinIO is a separate HTTP service (should always connect).
# Weaviate runs in embedded mode inside the worker; the API uses it read-only
# for chunk-count aggregation and degrades to None if unavailable.
try:
    _db_client = _db.create_persistent_client()
except Exception as _exc:
    logger.warning("Document store (MinIO) client init failed: %s — /documents endpoints unavailable", _exc)
    _db_client = None

# The API container does not run embedded Weaviate — chunk counts are unavailable here.
_vector_client = None
_obs_tracer = get_tracer()
_rate_limiter = InMemoryRateLimiter(
    limit=RATE_LIMIT_REQUESTS_PER_MINUTE,
    window_seconds=RATE_LIMIT_WINDOW_SECONDS,
)
_api_inflight_semaphore = (
    asyncio.Semaphore(RAG_API_MAX_INFLIGHT_REQUESTS)
    if RAG_API_MAX_INFLIGHT_REQUESTS > 0
    else None
)
_api_overload_queue_timeout_ms = max(1, RAG_API_OVERLOAD_QUEUE_TIMEOUT_MS)

API_PORT = int(os.environ.get("RAG_API_PORT", "8000"))


def _enforce_rate_limit(principal: Principal, endpoint: str) -> None:
    if not RATE_LIMIT_ENABLED:
        return
    tenant_quota = get_tenant_quota(principal.tenant_id)
    scope_key = principal.project_id or principal.subject
    limit_result = _rate_limiter.check(
        key=f"{principal.tenant_id}:{scope_key}",
        limit=tenant_quota,
        window_seconds=RATE_LIMIT_WINDOW_SECONDS,
    )
    if not limit_result.allowed:
        RATE_LIMIT_REJECTS.labels(endpoint=endpoint).inc()
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry in {limit_result.retry_after_seconds}s",
        )


async def _acquire_request_slot(endpoint: str) -> bool:
    """Acquire API overload slot and raise 503 when capacity is saturated."""
    if _api_inflight_semaphore is None:
        return False
    try:
        await asyncio.wait_for(
            _api_inflight_semaphore.acquire(),
            timeout=_api_overload_queue_timeout_ms / 1000.0,
        )
    except asyncio.TimeoutError:
        OVERLOAD_REJECTS.labels(endpoint=endpoint).inc()
        REQUESTS_TOTAL.labels(endpoint=endpoint, method="POST", status="503").inc()
        raise HTTPException(
            status_code=503,
            detail=(
                "Server overloaded: too many in-flight requests. "
                "Please retry shortly."
            ),
        )
    INFLIGHT_REQUESTS.inc()
    return True


def _release_request_slot(acquired: bool) -> None:
    """Release API overload slot previously acquired."""
    if not acquired:
        return
    INFLIGHT_REQUESTS.dec()
    if _api_inflight_semaphore is not None:
        _api_inflight_semaphore.release()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _validate_startup_config()
    _validate_all_config()
    _validate_optional_dependencies()
    global _temporal_client
    logger.info("Connecting to Temporal at %s", TEMPORAL_TARGET_HOST)
    _temporal_client = await Client.connect(TEMPORAL_TARGET_HOST)
    logger.info("API server ready — queries route through Temporal to preloaded workers")
    sweeper_task = start_job_sweeper()
    try:
        yield
    finally:
        sweeper_task.cancel()
        try:
            await sweeper_task
        except (asyncio.CancelledError, Exception):
            pass
        # temporalio.client.Client has no close(); the underlying gRPC channel
        # is torn down with the process. Dropping the reference is the correct
        # shutdown behavior (calling a nonexistent close() fails the whole
        # FastAPI shutdown sequence).
        _temporal_client = None
        logger.info("API server shutting down")


app = FastAPI(
    title="RAG Query API",
    description="Production RAG endpoint — models preloaded in Temporal workers, "
                "queries dispatched via durable workflows.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=RAG_API_CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Observability middleware: starts a root trace per request and extracts the
# W3C traceparent header so external client traces link into backend traces.
# Installed after CORS so CORS preflights are not traced, and before the
# request_id middleware so the trace's lifetime envelops request_id handling.
install_observability_middleware(app)


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """Attach an id to each request for traceability and consistent error payloads."""
    request_id = request.headers.get("x-request-id") or f"req-{uuid.uuid4().hex[:12]}"
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


@app.exception_handler(HTTPException)
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Normalize HTTPException payloads to the API error envelope."""
    detail = exc.detail
    # For 500, always return a generic message regardless of detail and use
    # the same code as the unhandled exception handler for consistency.
    if exc.status_code == 500:
        code = "INTERNAL_SERVER_ERROR"
        message = "Internal server error"
        details = None
    elif isinstance(detail, str):
        code = f"HTTP_{exc.status_code}"
        message = detail
        details = None
    elif isinstance(detail, dict):
        code = f"HTTP_{exc.status_code}"
        message = str(detail.get("message", "Request failed"))
        details = detail
    else:
        code = f"HTTP_{exc.status_code}"
        message = str(detail)
        details = None
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload(request, code=code, message=message, details=details),
    )


def _jsonsafe_validation_errors(exc: RequestValidationError) -> list[dict]:
    """Make ``exc.errors()`` JSON-serializable.

    Pydantic v2 puts the raw raised exception in ``ctx['error']`` for
    ``model_validator``/field-validator failures (e.g. the orchestrator
    mutual-exclusion check). A raw ``ValueError`` is not JSON-serializable, so
    without stringifying it the 422 handler itself raises ``TypeError`` and the
    catch-all turns a clean client error into a 500. Stringify any non-primitive
    ``ctx`` value so the 422 payload always renders.
    """
    safe: list[dict] = []
    _prim = (str, int, float, bool, type(None))
    for err in exc.errors():
        e = dict(err)
        ctx = e.get("ctx")
        if isinstance(ctx, dict):
            e["ctx"] = {k: (v if isinstance(v, _prim) else str(v)) for k, v in ctx.items()}
        safe.append(e)
    return safe


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    """Normalize request-body/params validation failures."""
    return JSONResponse(
        status_code=422,
        content=_error_payload(
            request,
            code="REQUEST_VALIDATION_ERROR",
            message="Request validation failed",
            details={"errors": _jsonsafe_validation_errors(exc)},
        ),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all handler to keep server errors schema-consistent."""
    logger.exception("Unhandled API exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content=_error_payload(
            request,
            code="INTERNAL_SERVER_ERROR",
            message="Internal server error",
        ),
    )


def _emit_stream_observability_async(
    *,
    workflow_id: str,
    request: QueryRequest,
    retrieval_ms: float,
    generation_ms: float,
    latency_ms: float,
    token_count: int,
    stage_timings: list[dict],
    timing_totals: dict,
    outcome: str,
    error_message: str | None = None,
) -> None:
    """Emit stream observability in background to keep streaming latency stable."""

    async def _emit() -> None:
        def _sync_emit() -> None:
            span = _obs_tracer.start_span(
                "rag.api.stream",
                {
                    "workflow_id": workflow_id,
                    "query_length": len(request.query),
                    "search_limit": request.search_limit,
                    "rerank_top_k": request.rerank_top_k,
                    "alpha": request.alpha,
                    "outcome": outcome,
                },
            )
            try:
                span.set_attribute("latency_ms", round(latency_ms, 1))
                span.set_attribute("retrieval_ms", round(retrieval_ms, 1))
                span.set_attribute("generation_ms", round(generation_ms, 1))
                span.set_attribute("token_count", int(token_count))
                span.set_attribute("stage_timings", stage_timings)
                span.set_attribute("timing_totals", timing_totals)
                if request.source_filter:
                    span.set_attribute("source_filter", request.source_filter)
                if request.heading_filter:
                    span.set_attribute("heading_filter", request.heading_filter)
                if error_message:
                    span.set_attribute("error_message", error_message)
            finally:
                span.end(
                    status="ok" if outcome == "completed" else "error",
                    error=Exception(error_message) if error_message else None,
                )

        try:
            await asyncio.to_thread(_sync_emit)
        except Exception as exc:
            logger.debug("Background stream observability emit failed: %s", exc)

    asyncio.create_task(_emit())


def _get_temporal_client() -> Client | None:
    return _temporal_client


app.include_router(
    create_query_router(
        get_temporal_client=_get_temporal_client,
        require_role=require_role,
        enforce_rate_limit=_enforce_rate_limit,
        acquire_request_slot=_acquire_request_slot,
        release_request_slot=_release_request_slot,
        emit_stream_observability=_emit_stream_observability_async,
        logger=logger,
        # Same MinIO-backed handle the documents router uses — the turn
        # loop's DEEP_STUDY action fetches clean documents through it.
        db_client=_db_client,
    )
)
app.include_router(create_admin_router())
app.include_router(create_documents_router(db_client=_db_client, vector_client=_vector_client))
app.include_router(create_ingest_router(get_temporal_client=_get_temporal_client))
app.include_router(create_system_router(get_temporal_client=_get_temporal_client, logger=logger))
app.include_router(
    create_console_router(
        get_temporal_client=_get_temporal_client,
        logger=logger,
        enforce_rate_limit=_enforce_rate_limit,
        acquire_request_slot=_acquire_request_slot,
        release_request_slot=_release_request_slot,
        console_ok=_console_ok,
    )
)
