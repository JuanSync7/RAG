"""Unit tests for ``submit_with_context``.

Covers:
- OTel ambient span context propagates across the thread boundary so a span
  opened in the worker nests under the caller's span (matching trace_id and
  parent linkage).
- Generic contextvars also propagate.
- Negative control: bare ``ThreadPoolExecutor.submit`` orphans the worker
  span as a fresh trace root (documents the bug we're fixing).
"""
from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor

import pytest

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util._once import Once

import src.platform.observability as _obs_module
from src.platform.observability import submit_with_context
from src.platform.observability.otel.backend import OTelBackend


@pytest.fixture
def otel_capture(monkeypatch):
    """Install a fresh SDK TracerProvider + InMemorySpanExporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    monkeypatch.setattr(otel_trace, "_TRACER_PROVIDER", provider, raising=False)
    monkeypatch.setattr(otel_trace, "_TRACER_PROVIDER_SET_ONCE", Once(), raising=False)

    import config.settings as _settings
    _settings.OBSERVABILITY_PROVIDER = "otel"
    _obs_module._backend = OTelBackend()
    try:
        yield exporter
    finally:
        _obs_module._backend = None
        _settings.OBSERVABILITY_PROVIDER = "noop"


def _worker_open_span(span_name: str):
    """Open a span via the public tracer; return its trace_id + span_id."""
    tracer = _obs_module.get_tracer()
    with tracer.span(span_name) as _s:
        # Resolve via otel current span to get IDs.
        cur = otel_trace.get_current_span().get_span_context()
        return cur.trace_id, cur.span_id


def test_submit_with_context_propagates_otel_span(otel_capture):
    """Worker span nests under caller span via copy_context()."""
    tracer = _obs_module.get_tracer()
    with ThreadPoolExecutor(max_workers=2) as pool:
        with tracer.span("caller.outer"):
            caller_ctx = otel_trace.get_current_span().get_span_context()
            fut = submit_with_context(pool, _worker_open_span, "worker.inner")
            worker_trace_id, worker_span_id = fut.result(timeout=5)

    spans = {s.name: s for s in otel_capture.get_finished_spans()}
    assert "caller.outer" in spans
    assert "worker.inner" in spans

    # Same trace
    assert spans["worker.inner"].context.trace_id == caller_ctx.trace_id
    # Worker span's parent is the caller span
    assert spans["worker.inner"].parent is not None
    assert spans["worker.inner"].parent.span_id == caller_ctx.span_id
    # And same trace_id as reported from inside the worker
    assert worker_trace_id == caller_ctx.trace_id


def test_bare_submit_orphans_worker_span(otel_capture):
    """Negative control: bare ``pool.submit`` loses ambient context."""
    tracer = _obs_module.get_tracer()
    with ThreadPoolExecutor(max_workers=2) as pool:
        with tracer.span("caller.outer"):
            caller_ctx = otel_trace.get_current_span().get_span_context()
            fut = pool.submit(_worker_open_span, "worker.orphan")
            _worker_trace_id, _ = fut.result(timeout=5)

    spans = {s.name: s for s in otel_capture.get_finished_spans()}
    # The worker span is a fresh root: no parent, different trace_id.
    orphan = spans["worker.orphan"]
    assert orphan.parent is None
    assert orphan.context.trace_id != caller_ctx.trace_id


def test_submit_with_context_propagates_generic_contextvars():
    """Arbitrary ContextVar set on the caller is visible in the worker."""
    var: contextvars.ContextVar[str] = contextvars.ContextVar("rag_test_var")
    var.set("hello-from-caller")

    def worker():
        return var.get("MISSING")

    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = submit_with_context(pool, worker)
        assert fut.result(timeout=5) == "hello-from-caller"


def test_submit_with_context_isolation():
    """Mutations inside the worker do not leak back to the caller context."""
    var: contextvars.ContextVar[str] = contextvars.ContextVar(
        "rag_test_var_iso", default="caller"
    )

    def worker():
        var.set("worker")
        return var.get()

    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = submit_with_context(pool, worker)
        assert fut.result(timeout=5) == "worker"
    # Caller side unchanged
    assert var.get() == "caller"
