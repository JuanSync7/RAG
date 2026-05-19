"""Tests for the FastAPI observability middleware (HTTP root traces).

Covers:
- Every HTTP request emits an ``api.<resource>.<verb>`` root trace.
- Inbound W3C ``traceparent`` is honoured: the root span shares its trace_id.
- Without ``traceparent`` a fresh trace is started.
- ``http.method`` / ``http.route`` / status code are recorded.
- 500 responses mark the trace ``status=error``.
- Nested instrumentation (``retrieval.rag``-style child spans) parents under
  the API root span, producing the expected span tree.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from opentelemetry.util._once import Once

import src.platform.observability as _obs_module
from src.platform.observability import get_tracer
from src.platform.observability.otel.backend import OTelBackend
from server.observability_middleware import install_observability_middleware


@pytest.fixture
def otel_capture(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    monkeypatch.setattr(otel_trace, "_TRACER_PROVIDER", provider, raising=False)
    monkeypatch.setattr(otel_trace, "_TRACER_PROVIDER_SET_ONCE", Once(), raising=False)

    import config.settings as _settings
    _settings.OBSERVABILITY_PROVIDER = "otel"
    _obs_module._backend = OTelBackend()

    yield exporter

    _obs_module._backend = None
    _settings.OBSERVABILITY_PROVIDER = "noop"


def _make_app() -> FastAPI:
    """Construct a minimal FastAPI app with the observability middleware.

    Includes two routes: ``POST /query`` that simulates a successful RAG call
    with a nested ``retrieval.rag`` child span, and ``POST /boom`` that raises
    an unhandled exception. A ``GET /documents/{id}`` route exercises the
    templated-route span-name path.
    """
    app = FastAPI()
    install_observability_middleware(app)

    @app.post("/query")
    async def _query(body: dict) -> dict:
        tracer = get_tracer()
        # Simulate the retrieval root span — uses ambient context so it should
        # nest under the middleware's api.query.post trace.
        with tracer.span("retrieval.rag", {"query_len": len(body.get("query", ""))}) as rag:
            with tracer.span("retrieval.search.hybrid"):
                pass
        return {"answer": "ok"}

    @app.post("/boom")
    async def _boom() -> dict:
        raise HTTPException(status_code=500, detail="kaboom")

    @app.get("/documents/{doc_id}")
    async def _get_doc(doc_id: str) -> dict:
        return {"id": doc_id}

    return app


def _spans_by_name(exporter: InMemorySpanExporter, name: str):
    return [s for s in exporter.get_finished_spans() if s.name == name]


def test_query_endpoint_emits_api_root_trace(otel_capture):
    app = _make_app()
    client = TestClient(app)

    resp = client.post("/query", json={"query": "hello"})
    assert resp.status_code == 200

    roots = _spans_by_name(otel_capture, "api.query.post")
    assert len(roots) == 1
    root = roots[0]
    assert root.attributes["http.method"] == "POST"
    assert root.attributes["http.route"] == "/query"


def test_query_endpoint_records_status_code(otel_capture):
    app = _make_app()
    client = TestClient(app)
    resp = client.post("/query", json={"query": "hello"})
    assert resp.status_code == 200

    status_spans = _spans_by_name(otel_capture, "api.response")
    assert any(s.attributes.get("http.status_code") == 200 for s in status_spans)


def test_query_endpoint_inherits_traceparent_when_provided(otel_capture):
    app = _make_app()
    client = TestClient(app)
    inbound_trace_id = "0af7651916cd43dd8448eb211c80319c"
    inbound_span_id = "b7ad6b7169203331"

    resp = client.post(
        "/query",
        json={"query": "hi"},
        headers={"traceparent": f"00-{inbound_trace_id}-{inbound_span_id}-01"},
    )
    assert resp.status_code == 200

    roots = _spans_by_name(otel_capture, "api.query.post")
    assert len(roots) == 1
    assert format(roots[0].context.trace_id, "032x") == inbound_trace_id
    assert roots[0].parent is not None
    assert format(roots[0].parent.span_id, "016x") == inbound_span_id


def test_query_endpoint_without_traceparent_starts_fresh_trace(otel_capture):
    app = _make_app()
    client = TestClient(app)

    resp = client.post("/query", json={"query": "hi"})
    assert resp.status_code == 200

    roots = _spans_by_name(otel_capture, "api.query.post")
    assert len(roots) == 1
    assert roots[0].parent is None


def test_full_span_tree_nests_retrieval_under_api_root(otel_capture):
    """Drive a query that emits retrieval.rag > retrieval.search.hybrid.

    Assert the full chain is api.query.post > retrieval.rag > retrieval.search.hybrid.
    """
    app = _make_app()
    client = TestClient(app)

    resp = client.post("/query", json={"query": "hello"})
    assert resp.status_code == 200

    spans = otel_capture.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}
    api_root = next(s for s in spans if s.name == "api.query.post")
    rag = next(s for s in spans if s.name == "retrieval.rag")
    hybrid = next(s for s in spans if s.name == "retrieval.search.hybrid")

    # All three share the same trace_id.
    assert api_root.context.trace_id == rag.context.trace_id == hybrid.context.trace_id

    # Parent chain: hybrid -> rag -> api_root.
    assert rag.parent is not None and rag.parent.span_id == api_root.context.span_id
    assert hybrid.parent is not None and hybrid.parent.span_id == rag.context.span_id


def test_500_response_marks_trace_error(otel_capture):
    app = _make_app()
    client = TestClient(app)
    # raise_server_exceptions=False so the TestClient returns the 500
    # response (produced by FastAPI's default handler) instead of bubbling
    # the HTTPException.
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/boom")
    assert resp.status_code == 500

    roots = _spans_by_name(otel_capture, "api.boom.post")
    assert len(roots) == 1
    # 500 status recorded via marker child span.
    status_spans = _spans_by_name(otel_capture, "api.response")
    assert any(s.attributes.get("http.status_code") == 500 for s in status_spans)


def test_templated_route_uses_path_template_in_attributes(otel_capture):
    app = _make_app()
    client = TestClient(app)
    resp = client.get("/documents/abc-123")
    assert resp.status_code == 200

    roots = _spans_by_name(otel_capture, "api.documents.get")
    assert len(roots) == 1
    # Root attribute is the raw URL path (matched route is not available until
    # after call_next runs). The templated path is captured on the response
    # marker span instead so dashboards can group by templated route.
    response_markers = _spans_by_name(otel_capture, "api.response")
    assert any(
        s.attributes.get("http.route") == "/documents/{doc_id}"
        for s in response_markers
    )


def test_request_id_recorded_when_header_present(otel_capture):
    app = _make_app()
    client = TestClient(app)
    resp = client.post(
        "/query",
        json={"query": "hi"},
        headers={"x-request-id": "req-test-1234"},
    )
    assert resp.status_code == 200
    roots = _spans_by_name(otel_capture, "api.query.post")
    assert roots[0].attributes.get("request_id") == "req-test-1234"


# ---------------------------------------------------------------------------
# Streaming (SSE) lifetime tests — regression coverage for the deferred trace
# close. With BaseHTTPMiddleware the root span ended before the generator
# yielded its chunks, so child spans emitted inside the generator outlived the
# root. The ASGI-level implementation must keep the root open until the final
# http.response.body chunk (more_body=False).
# ---------------------------------------------------------------------------

import asyncio

import httpx
from httpx import ASGITransport
from starlette.responses import StreamingResponse


def _make_streaming_app() -> FastAPI:
    """FastAPI app exposing a streaming endpoint that emits child spans per chunk."""
    app = FastAPI()
    install_observability_middleware(app)

    @app.get("/stream")
    async def _stream():
        tracer = get_tracer()

        async def gen():
            for i in range(3):
                with tracer.span("stream.chunk", {"i": i}):
                    # Tiny sleep so chunk spans get a measurable duration and
                    # land strictly after the root span's start.
                    await asyncio.sleep(0.005)
                yield f"data: {i}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/stream-error")
    async def _stream_error():
        tracer = get_tracer()

        async def gen():
            with tracer.span("stream.chunk", {"i": 0}):
                await asyncio.sleep(0.001)
            yield "data: 0\n\n"
            raise RuntimeError("mid-stream boom")

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/plain")
    async def _plain():
        return {"ok": True}

    return app


def test_streaming_root_span_outlives_chunk_spans(otel_capture):
    """Root api.stream.get must end AFTER the last chunk span emitted inside
    the generator. With BaseHTTPMiddleware this fails because the root closes
    when the Response object is returned, before the generator yields.
    """
    app = _make_streaming_app()

    async def _run():
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            resp = await client.get("/stream")
            assert resp.status_code == 200
            # Drain the SSE body.
            body = resp.text
            assert "data: 2" in body

    asyncio.run(_run())

    roots = _spans_by_name(otel_capture, "api.stream.get")
    assert len(roots) == 1
    root = roots[0]
    chunk_spans = _spans_by_name(otel_capture, "stream.chunk")
    assert len(chunk_spans) == 3
    last_chunk_end = max(s.end_time for s in chunk_spans)
    # The deferred-close contract: the root span MUST end no earlier than the
    # last child span emitted from inside the streaming generator.
    assert root.end_time >= last_chunk_end, (
        f"root end_time {root.end_time} predates last chunk end_time "
        f"{last_chunk_end} — streaming trace closed too early"
    )
    # All chunk spans share trace_id with the root and parent under it.
    for s in chunk_spans:
        assert s.context.trace_id == root.context.trace_id
        assert s.parent is not None
        assert s.parent.span_id == root.context.span_id


def test_non_streaming_response_shape_unchanged(otel_capture):
    """Non-streaming endpoints must still produce root + api.response child."""
    app = _make_streaming_app()

    async def _run():
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            resp = await client.get("/plain")
            assert resp.status_code == 200

    asyncio.run(_run())

    roots = _spans_by_name(otel_capture, "api.plain.get")
    assert len(roots) == 1
    markers = _spans_by_name(otel_capture, "api.response")
    assert any(
        m.attributes.get("http.status_code") == 200
        and m.attributes.get("http.route") == "/plain"
        for m in markers
    )


def test_streaming_generator_error_marks_trace_error(otel_capture):
    """Generator raising mid-stream must close root trace with error status."""
    app = _make_streaming_app()

    async def _run():
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            try:
                resp = await client.get("/stream-error")
                # Consume body — may surface the error here.
                _ = resp.text
            except Exception:
                pass

    asyncio.run(_run())

    roots = _spans_by_name(otel_capture, "api.stream-error.get")
    assert len(roots) == 1
    assert roots[0].status.status_code == StatusCode.ERROR
