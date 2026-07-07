"""Shared fixtures for LLM instrumentation tests.

Installs a fresh OTel SDK TracerProvider + InMemorySpanExporter and binds the
global ``get_tracer()`` singleton to a real ``OTelBackend`` so instrumented
production code emits spans the test can inspect.
"""
from __future__ import annotations

import sys

# The top-level tests/conftest.py installs a *partial* langchain_core stub
# (only langchain_core.embeddings) that shadows the real package and breaks
# submodule imports such as langchain_core.callbacks / .globals / .runnables.
# Evict the stub here — conftest is imported before sibling test modules —
# so src.common.llm (and the tests below) bind the real langchain_core.
for _mod in list(sys.modules):
    if (
        _mod == "langchain_core"
        or _mod.startswith("langchain_core.")
        or _mod == "langgraph"
        or _mod.startswith("langgraph.")
        or _mod == "langchain_text_splitters"
    ):
        del sys.modules[_mod]

import pytest

# NOTE: the stub/real langchain boundary across test packages is enforced by the
# root tests/conftest.py ``pytest_collectstart`` hook (it evicts the stub for
# tests/llm modules and restores it for every other package, order-independently).
# The eviction above is retained so this conftest's own OTel imports below — and
# the tests/llm modules collected immediately after it — bind the real package.

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._once import Once

import src.platform.observability as _obs_module
from src.platform.observability.otel.backend import OTelBackend


@pytest.fixture
def otel_capture(monkeypatch):
    """Install a fresh SDK TracerProvider with an in-memory exporter,
    swap the global observability backend to a real OTelBackend, and yield
    the exporter so tests can read finished spans."""
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
