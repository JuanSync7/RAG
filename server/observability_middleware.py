# @summary
# FastAPI middleware bridging inbound HTTP requests to the observability backend:
# extracts W3C traceparent, starts a root trace, records http.* attributes.
# Exports: install_observability_middleware
# Deps: starlette, src.platform.observability
# @end-summary
"""HTTP observability middleware for the RAG API.

Installs a Starlette/FastAPI middleware that wraps every request in a root
observability trace named ``api.<resource>.<verb>``. If the inbound request
carries a W3C ``traceparent`` header, the trace inherits its trace_id so
external client traces link with backend traces; otherwise a fresh trace is
started.

The middleware uses only the public observability API
(:func:`src.platform.observability.get_tracer`) and therefore does not
import the OpenTelemetry SDK — keeping all ``opentelemetry`` imports
encapsulated under ``src/platform/observability/otel/``.
"""
from __future__ import annotations

from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp

from src.platform.observability import get_tracer


def _matched_route(request: Request) -> Optional[str]:
    """Return the matched route's templated path, if routing has run."""
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path if path else None


def _route_template(request: Request) -> str:
    """Return the matched route template (``/documents/{id}``) when available.

    Falls back to ``request.url.path`` if the route has not been matched yet
    (rare — happens for requests that 404 before routing).
    """
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if path:
        return path
    return request.url.path or "/"


def _span_name_for(method: str, route: str) -> str:
    """Build a span name like ``api.query.post`` from method + matched route.

    The first path segment (or "root" for "/") is used as the resource
    component. Routes with templated segments (``/documents/{id}``) use the
    static prefix so the span name stays low-cardinality.
    """
    segments = [seg for seg in route.split("/") if seg and not seg.startswith("{")]
    resource = segments[0] if segments else "root"
    return f"api.{resource}.{method.lower()}"


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Root-trace-per-request middleware.

    For every HTTP request:

    1. Extract headers (case-insensitive) into a dict carrier.
    2. Start a root trace via
       :meth:`ObservabilityBackend.start_trace_from_carrier` so an inbound
       ``traceparent`` is honoured.
    3. Record ``http.method``, ``http.route``, ``request_id``, optional
       ``user_id`` / ``project_id`` from headers/request state.
    4. Set ``http.status_code`` on the trace root when the response is
       produced. Errors mark the trace ``status=error``.
    """

    async def dispatch(self, request: Request, call_next):
        tracer = get_tracer()
        method = request.method
        route = _route_template(request)
        span_name = _span_name_for(method, route)

        carrier = {k.lower(): v for k, v in request.headers.items()}
        request_id = (
            getattr(request.state, "request_id", None)
            or carrier.get("x-request-id")
        )
        user_id = carrier.get("x-user-id")
        project_id = carrier.get("x-project-id")

        metadata: dict = {
            "http.method": method,
            "http.route": route,
        }
        if request_id:
            metadata["request_id"] = request_id
        if user_id:
            metadata["user_id"] = user_id
        if project_id:
            metadata["project_id"] = project_id

        trace = tracer.start_trace_from_carrier(span_name, carrier=carrier, metadata=metadata)
        # Stash the trace so route handlers can attach domain-specific
        # attributes (e.g., resolved principal/project) without re-fetching
        # the ambient context.
        request.state.observability_trace = trace

        status_code: Optional[int] = None
        with trace:
            try:
                response = await call_next(request)
                status_code = getattr(response, "status_code", None)
            except BaseException:  # noqa: BLE001 — re-raised below
                # Emit a status marker span so http.status_code is observable
                # even on uncaught exceptions, then re-raise so the trace
                # context manager records status=error.
                try:
                    with trace.span(
                        "api.response",
                        {
                            "http.status_code": 500,
                            "http.route": _matched_route(request) or route,
                        },
                    ):
                        pass
                except Exception:
                    pass
                raise
            # Successful response path — emit status marker child span carrying
            # the matched (templated) route path so cardinality stays bounded.
            try:
                with trace.span(
                    "api.response",
                    {
                        "http.status_code": int(status_code) if status_code is not None else 0,
                        "http.route": _matched_route(request) or route,
                    },
                ):
                    pass
            except Exception:
                pass
            return response


def install_observability_middleware(app: ASGIApp) -> None:
    """Install :class:`ObservabilityMiddleware` on a FastAPI/Starlette app.

    Must be called once at app construction. Idempotent only at the
    application-construction level — calling twice will register two
    middlewares and double-instrument every request.
    """
    app.add_middleware(ObservabilityMiddleware)
