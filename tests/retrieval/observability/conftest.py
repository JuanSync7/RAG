"""Shared OTel capture fixture for retrieval instrumentation tests.

Mirrors tests/llm/conftest.py: installs a fresh OTel SDK TracerProvider with
an in-memory exporter, swaps the observability backend to a real OTelBackend,
and yields the exporter so tests can inspect finished spans.
"""
from __future__ import annotations

import pytest

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._once import Once

import src.platform.observability as _obs_module
from src.platform.observability.otel.backend import OTelBackend


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
