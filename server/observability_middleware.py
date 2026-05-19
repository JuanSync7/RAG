# @summary
# Raw ASGI middleware bridging inbound HTTP requests to the observability backend:
# extracts W3C traceparent, starts a root trace, defers trace close until the
# final response body chunk (with more_body=False) so SSE/StreamingResponse
# generators emit child spans under a still-open root.
# Exports: install_observability_middleware, ObservabilityMiddleware
# Deps: starlette, src.platform.observability
# @end-summary
"""HTTP observability middleware for the RAG API.

A pure ASGI middleware (not :class:`starlette.middleware.base.BaseHTTPMiddleware`)
that wraps every request in a root observability trace named
``api.<resource>.<verb>``. The root trace's lifetime extends until the **final**
``http.response.body`` ASGI message (``more_body=False``) has been forwarded
downstream — so streaming responses (SSE) keep the root span open while their
generators are still emitting child spans.

If the inbound request carries a W3C ``traceparent`` header, the trace inherits
its trace_id so external client traces link with backend traces; otherwise a
fresh trace is started.

The middleware uses only the public observability API
(:func:`src.platform.observability.get_tracer`) and therefore does not
import the OpenTelemetry SDK — keeping all ``opentelemetry`` imports
encapsulated under ``src/platform/observability/otel/``.
"""
from __future__ import annotations

from typing import Optional

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.platform.observability import get_tracer


def _route_template_from_scope(scope: Scope) -> str:
    """Return the matched route template if routing has run, else raw path."""
    route = scope.get("route")
    path = getattr(route, "path", None)
    if path:
        return path
    return scope.get("path") or "/"


def _matched_route_from_scope(scope: Scope) -> Optional[str]:
    """Return only the matched (templated) route path, or None if not matched yet."""
    route = scope.get("route")
    path = getattr(route, "path", None)
    return path if path else None


def _span_name_for(method: str, route: str) -> str:
    """Build a span name like ``api.query.post`` from method + matched route.

    The first path segment (or "root" for "/") is used as the resource
    component. Routes with templated segments (``/documents/{id}``) use the
    static prefix so the span name stays low-cardinality.
    """
    segments = [seg for seg in route.split("/") if seg and not seg.startswith("{")]
    resource = segments[0] if segments else "root"
    return f"api.{resource}.{method.lower()}"


def _headers_to_carrier(raw_headers) -> dict:
    """Decode ASGI raw header pairs into a lower-cased dict carrier."""
    carrier: dict = {}
    for raw_key, raw_val in raw_headers or ():
        try:
            key = raw_key.decode("latin-1").lower()
            val = raw_val.decode("latin-1")
        except Exception:
            continue
        carrier[key] = val
    return carrier


class ObservabilityMiddleware:
    """Root-trace-per-request ASGI middleware.

    For every HTTP request:

    1. Decode raw scope headers into a lower-cased carrier dict.
    2. Start a root trace via
       :meth:`ObservabilityBackend.start_trace_from_carrier` so an inbound
       ``traceparent`` is honoured.
    3. Record ``http.method``, ``http.route``, ``request_id``, optional
       ``user_id`` / ``project_id`` from headers.
    4. Stash the trace in ``scope["state"]["observability_trace"]`` so route
       handlers can attach domain-specific attributes (FastAPI surfaces this
       as ``request.state.observability_trace``).
    5. Wrap ``send`` so the trace is closed only on the final response body
       chunk (``http.response.body`` with ``more_body=False``) — guaranteeing
       streaming generators emit child spans under a still-open root.
    6. On the final chunk, emit a single ``api.response`` child span carrying
       ``http.status_code`` and the templated ``http.route`` (resolved late,
       after routing has populated ``scope["route"]``).
    7. If the downstream app raises, emit an error ``api.response`` and close
       the trace with error status, then re-raise.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        tracer = get_tracer()
        method = scope.get("method", "GET")
        # Route template at request entry — at this point routing has NOT yet
        # run, so this is typically the raw URL path. The templated route is
        # resolved lazily inside send_wrapper after the app has matched.
        early_route = _route_template_from_scope(scope)
        span_name = _span_name_for(method, early_route)

        carrier = _headers_to_carrier(scope.get("headers"))
        request_id = carrier.get("x-request-id")
        user_id = carrier.get("x-user-id")
        project_id = carrier.get("x-project-id")

        metadata: dict = {
            "http.method": method,
            "http.route": early_route,
        }
        if request_id:
            metadata["request_id"] = request_id
        if user_id:
            metadata["user_id"] = user_id
        if project_id:
            metadata["project_id"] = project_id

        trace = tracer.start_trace_from_carrier(
            span_name, carrier=carrier, metadata=metadata
        )
        # Enter the trace's ambient OTel context so spans started later (by
        # route handlers, by the streaming generator) parent under the root.
        try:
            trace.__enter__()
        except Exception:
            pass

        # FastAPI/Starlette populates scope["state"] with a dict-like object
        # that becomes request.state. Wire the trace through so route handlers
        # can attach attributes (e.g. resolved principal/project).
        state = scope.setdefault("state", {})
        try:
            state["observability_trace"] = trace
        except Exception:
            pass

        status_code_holder: dict = {"code": None}
        closed: dict = {"done": False}

        def _close(exc: Optional[BaseException] = None) -> None:
            """Emit the api.response marker and exit the trace exactly once."""
            if closed["done"]:
                return
            closed["done"] = True
            templated = _matched_route_from_scope(scope) or early_route
            status = status_code_holder["code"]
            if exc is not None and status is None:
                status = 500
            try:
                with trace.span(
                    "api.response",
                    {
                        "http.status_code": int(status) if status is not None else 0,
                        "http.route": templated,
                    },
                ):
                    pass
            except Exception:
                pass
            try:
                if exc is not None:
                    trace.__exit__(type(exc), exc, exc.__traceback__)
                else:
                    trace.__exit__(None, None, None)
            except Exception:
                pass

        async def send_wrapper(message: Message) -> None:
            mtype = message.get("type")
            if mtype == "http.response.start":
                status_code_holder["code"] = message.get("status")
            await send(message)
            if mtype == "http.response.body" and not message.get("more_body", False):
                _close(None)

        try:
            await self.app(scope, receive, send_wrapper)
        except BaseException as exc:  # noqa: BLE001 — re-raised below
            _close(exc)
            raise
        else:
            # Guard against apps that complete without ever emitting a final
            # body message (rare — e.g. websocket upgrades that share http
            # scope shouldn't happen here, but be defensive).
            _close(None)


def install_observability_middleware(app: ASGIApp) -> None:
    """Install :class:`ObservabilityMiddleware` on a FastAPI/Starlette app.

    Must be called once at app construction. Idempotent only at the
    application-construction level — calling twice will register two
    middlewares and double-instrument every request.
    """
    app.add_middleware(ObservabilityMiddleware)
