"""Tests for OTel span emission from the @node_span decorator.

Contract:
- A @node_span("foo")-decorated node emits an OTel span named "node.foo"
  in addition to the existing structured JSON log.
- The span carries stage_name, trace_id, and source_key attributes.
- Spans opened inside the node via get_tracer().span(...) nest under the
  node span via ambient OTel context propagation.
- Failures in the observability layer must not break the node (fail-open).
"""
from __future__ import annotations

import pytest

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture
def otel_capture(monkeypatch):
    """Install fresh SDK TracerProvider + in-memory exporter; wire OTelBackend.

    Bypasses the autouse noop fixture in tests/observability/conftest.py (which
    doesn't apply here in tests/ingest/) by directly instantiating an OTel
    backend and pinning it as the public-API singleton.
    """
    from opentelemetry.util._once import Once
    import src.platform.observability as obs_module
    import config.settings as _settings

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    monkeypatch.setattr(otel_trace, "_TRACER_PROVIDER", provider, raising=False)
    monkeypatch.setattr(otel_trace, "_TRACER_PROVIDER_SET_ONCE", Once(), raising=False)

    # Force the public-API singleton to OTel for this test.
    prev_provider = getattr(_settings, "OBSERVABILITY_PROVIDER", "")
    prev_backend = obs_module._backend
    _settings.OBSERVABILITY_PROVIDER = "otel"
    obs_module._backend = None

    yield exporter

    obs_module._backend = prev_backend
    _settings.OBSERVABILITY_PROVIDER = prev_provider


def test_node_span_emits_otel_span_with_attributes(otel_capture):
    """A decorated node emits an OTel span named node.<stage> with key attrs."""
    from src.ingest.common.observability import node_span

    @node_span("chunking")
    def my_node(state):
        return {"ok": True}

    out = my_node({"trace_id": "trace-99", "source_key": "docs/a.md"})
    assert out == {"ok": True}

    spans = otel_capture.get_finished_spans()
    node_spans = [s for s in spans if s.name == "node.chunking"]
    assert len(node_spans) == 1
    attrs = dict(node_spans[0].attributes)
    assert attrs["stage_name"] == "chunking"
    assert attrs["trace_id"] == "trace-99"
    assert attrs["source_key"] == "docs/a.md"


def test_node_span_nests_inner_tracer_spans(otel_capture):
    """Spans opened inside the node nest under the outer node.<stage> span."""
    from src.ingest.common.observability import node_span
    from src.platform.observability import get_tracer

    @node_span("embedding_storage")
    def my_node(state):
        with get_tracer().span("inner.work"):
            pass
        return {}

    my_node({"trace_id": "t-1", "source_key": "k"})

    spans = otel_capture.get_finished_spans()
    by_name = {s.name: s for s in spans}
    outer = by_name["node.embedding_storage"]
    inner = by_name["inner.work"]
    assert inner.parent is not None
    assert inner.parent.span_id == outer.context.span_id
    assert inner.context.trace_id == outer.context.trace_id


def test_node_span_records_error_on_otel_span(otel_capture):
    """An exception inside the node marks the OTel span as ERROR and re-raises."""
    from opentelemetry.trace import StatusCode
    from src.ingest.common.observability import node_span

    @node_span("boom_node")
    def my_node(state):
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        my_node({"trace_id": "t"})

    spans = otel_capture.get_finished_spans()
    node_spans = [s for s in spans if s.name == "node.boom_node"]
    assert len(node_spans) == 1
    assert node_spans[0].status.status_code == StatusCode.ERROR


def test_node_span_fail_open_when_get_tracer_raises(monkeypatch, caplog):
    """If get_tracer() blows up, the node still runs and JSON log still emits."""
    import json
    import logging

    from src.ingest.common import observability as obs_mod

    def explode():
        raise RuntimeError("backend broken")

    monkeypatch.setattr(obs_mod, "get_tracer", explode)

    @obs_mod.node_span("resilient")
    def my_node(state):
        return {"ran": True}

    caplog.set_level(logging.INFO, logger="rag.ingest.obs")
    out = my_node({"trace_id": "t", "source_key": "k"})
    assert out == {"ran": True}

    payloads = []
    for rec in caplog.records:
        try:
            p = json.loads(rec.getMessage())
        except (ValueError, TypeError):
            continue
        if isinstance(p, dict) and p.get("event") == "stage_span":
            payloads.append(p)
    assert len(payloads) == 1
    assert payloads[0]["stage_name"] == "resilient"
