"""Tests for ObservabilityBackend.start_trace_from_carrier (W3C traceparent).

Covers:
- ABC default falls back to trace() when no propagator support.
- NoopBackend default-method inheritance (no override needed).
- OTelBackend extracts trace_id from a valid W3C traceparent header.
- OTelBackend starts a fresh trace when no traceparent header present.
- OTelBackend tolerates malformed traceparent and falls back to fresh trace.
"""
from __future__ import annotations

import pytest

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._once import Once

import src.platform.observability as _obs_module
from src.platform.observability.noop.backend import NoopBackend, NoopTrace
from src.platform.observability.otel.backend import OTelBackend, OTelTrace


@pytest.fixture
def otel_capture(monkeypatch):
    """Install a fresh OTel SDK provider with in-memory exporter for assertions."""
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


def test_noop_start_trace_from_carrier_returns_noop_trace():
    backend = NoopBackend()
    trace = backend.start_trace_from_carrier(
        "api.query.post",
        carrier={"traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"},
    )
    assert isinstance(trace, NoopTrace)


def test_otel_start_trace_inherits_trace_id_from_carrier(otel_capture):
    backend = _obs_module.get_tracer()
    inbound_trace_id = "0af7651916cd43dd8448eb211c80319c"
    inbound_span_id = "b7ad6b7169203331"
    carrier = {"traceparent": f"00-{inbound_trace_id}-{inbound_span_id}-01"}

    trace = backend.start_trace_from_carrier("api.query.post", carrier=carrier)
    assert isinstance(trace, OTelTrace)
    with trace:
        pass

    spans = otel_capture.get_finished_spans()
    assert len(spans) == 1
    root = spans[0]
    assert root.name == "api.query.post"
    assert format(root.context.trace_id, "032x") == inbound_trace_id
    # The root span is parented under the inbound span.
    assert root.parent is not None
    assert format(root.parent.span_id, "016x") == inbound_span_id


def test_otel_start_trace_without_carrier_creates_fresh_trace(otel_capture):
    backend = _obs_module.get_tracer()
    trace = backend.start_trace_from_carrier("api.query.post", carrier={})
    with trace:
        pass
    spans = otel_capture.get_finished_spans()
    assert len(spans) == 1
    root = spans[0]
    assert root.name == "api.query.post"
    # No parent — fresh root.
    assert root.parent is None


def test_otel_start_trace_with_malformed_carrier_falls_back(otel_capture):
    backend = _obs_module.get_tracer()
    trace = backend.start_trace_from_carrier(
        "api.query.post",
        carrier={"traceparent": "garbage-not-a-real-traceparent"},
    )
    with trace:
        pass
    spans = otel_capture.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].parent is None


def test_otel_start_trace_case_insensitive_header(otel_capture):
    backend = _obs_module.get_tracer()
    inbound_trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    carrier = {"Traceparent": f"00-{inbound_trace_id}-00f067aa0ba902b7-01"}
    trace = backend.start_trace_from_carrier("api.query.post", carrier=carrier)
    with trace:
        pass
    spans = otel_capture.get_finished_spans()
    assert len(spans) == 1
    assert format(spans[0].context.trace_id, "032x") == inbound_trace_id


def test_otel_start_trace_metadata_recorded_as_attributes(otel_capture):
    backend = _obs_module.get_tracer()
    trace = backend.start_trace_from_carrier(
        "api.query.post",
        carrier={},
        metadata={"http.method": "POST", "http.route": "/query"},
    )
    with trace:
        pass
    spans = otel_capture.get_finished_spans()
    assert spans[0].attributes["http.method"] == "POST"
    assert spans[0].attributes["http.route"] == "/query"
