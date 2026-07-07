"""Tests for OTelBackend and its companion wrapper classes.

Coverage areas:
    - Init: with and without OTEL_EXPORTER_OTLP_ENDPOINT.
    - Trace/span/generation emission via an in-memory exporter.
    - gen_ai.* semantic-convention attribute mapping.
    - Truncation caps for free-text fields.
    - Error semantics: record_exception + status=ERROR.
    - Fail-open: SDK exceptions in set_attribute/end/etc. swallowed + logged.
    - flush()/shutdown(): propagate exceptions from the underlying SDK.
    - ABC conformance: OTelBackend/OTelSpan/OTelTrace/OTelGeneration satisfy
      their abstract bases.
    - Encapsulation: no opentelemetry imports outside otel/backend.py.
"""
from __future__ import annotations

import logging
import pathlib
import re

import pytest

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.platform.observability.backend import (
    Generation,
    ObservabilityBackend,
    Span,
    Trace,
)
from src.platform.observability.otel.backend import (
    OTelBackend,
    OTelGeneration,
    OTelSpan,
    OTelTrace,
    _TEXT_TRUNCATION_BYTES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def otel_capture(monkeypatch):
    """Install a fresh SDK TracerProvider with an in-memory exporter.

    OTel's TracerProvider is process-global. We override the private
    ``trace._TRACER_PROVIDER`` slot directly so each test gets a clean
    provider/exporter pair without leaking spans across tests. We also
    install a fresh ``Once`` so any later ``set_tracer_provider`` call
    inside ``OTelBackend.__init__`` is a harmless no-op (the backend
    short-circuits anyway because the provider is already an SDK one).
    """
    from opentelemetry.util._once import Once

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    monkeypatch.setattr(otel_trace, "_TRACER_PROVIDER", provider, raising=False)
    monkeypatch.setattr(otel_trace, "_TRACER_PROVIDER_SET_ONCE", Once(), raising=False)

    yield exporter


@pytest.fixture
def clean_otel(monkeypatch):
    """Reset the global TracerProvider slot to the proxy default + fresh Once.

    Mirrors the state of a process that has never called
    ``set_tracer_provider`` — ProxyTracerProvider present, Once unset.
    """
    from opentelemetry.trace import ProxyTracerProvider
    from opentelemetry.util._once import Once

    monkeypatch.setattr(otel_trace, "_TRACER_PROVIDER", ProxyTracerProvider(), raising=False)
    monkeypatch.setattr(otel_trace, "_TRACER_PROVIDER_SET_ONCE", Once(), raising=False)
    yield


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

def test_backend_initializes_without_endpoint(monkeypatch, clean_otel):
    """OTelBackend() must not raise when OTEL_EXPORTER_OTLP_ENDPOINT is unset."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    backend = OTelBackend()
    assert isinstance(backend, ObservabilityBackend)


def test_backend_installs_otlp_exporter_when_endpoint_set(monkeypatch, clean_otel):
    """When the env var is set, a BatchSpanProcessor must be registered on the provider."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
    OTelBackend()
    provider = otel_trace.get_tracer_provider()
    assert isinstance(provider, TracerProvider)
    # The active_span_processor is a MultiSpanProcessor wrapping the registered ones.
    msp = provider._active_span_processor
    inner = getattr(msp, "_span_processors", ())
    # At least one BatchSpanProcessor should be present.
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    assert any(isinstance(p, BatchSpanProcessor) for p in inner)


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------

def test_trace_returns_otel_trace_and_emits_span(otel_capture):
    """trace() returns OTelTrace; on exit, exactly one root span is exported."""
    backend = OTelBackend()
    t = backend.trace("pipeline.run")
    assert isinstance(t, OTelTrace)
    t.__exit__(None, None, None)
    spans = otel_capture.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "pipeline.run"


def test_trace_applies_metadata_as_attributes(otel_capture):
    """trace() metadata keys/values appear as span attributes on the root span."""
    backend = OTelBackend()
    t = backend.trace("pipeline.run", metadata={"run_id": "abc", "user": "alice"})
    t.__exit__(None, None, None)
    spans = otel_capture.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes)
    assert attrs["run_id"] == "abc"
    assert attrs["user"] == "alice"


# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------

def test_span_under_trace_is_parented(otel_capture):
    """A span() created with parent=trace must share the root's span context."""
    backend = OTelBackend()
    t = backend.trace("pipeline.run")
    child = backend.span("child.op", parent=t)
    child.end()
    t.__exit__(None, None, None)
    spans = otel_capture.get_finished_spans()
    by_name = {s.name: s for s in spans}
    root = by_name["pipeline.run"]
    kid = by_name["child.op"]
    assert kid.parent is not None
    assert kid.parent.span_id == root.context.span_id
    assert kid.context.trace_id == root.context.trace_id


def test_span_without_parent_is_unparented(otel_capture):
    """A span() with no parent must be a root span itself."""
    backend = OTelBackend()
    s = backend.span("standalone.op")
    s.end()
    spans = otel_capture.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].parent is None


# ---------------------------------------------------------------------------
# Generation / gen_ai semantic conventions
# ---------------------------------------------------------------------------

def test_generation_emits_all_gen_ai_attributes(otel_capture):
    """generation() must set gen_ai.system, gen_ai.request.model, gen_ai.prompt,
    and (after set_output/set_token_counts) gen_ai.completion and gen_ai.usage.*.
    """
    backend = OTelBackend()
    g = backend.generation(
        "llm.call",
        model="gpt-4o",
        input="hello",
        metadata={"gen_ai.system": "openai"},
    )
    g.set_output("hi there")
    g.set_token_counts(prompt_tokens=12, completion_tokens=34)
    g.end()
    spans = otel_capture.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes)
    assert attrs["gen_ai.system"] == "openai"
    assert attrs["gen_ai.request.model"] == "gpt-4o"
    assert attrs["gen_ai.prompt"] == "hello"
    assert attrs["gen_ai.completion"] == "hi there"
    assert attrs["gen_ai.usage.input_tokens"] == 12
    assert attrs["gen_ai.usage.output_tokens"] == 34


def test_generation_defaults_system_to_unknown(otel_capture):
    """If no system is provided in metadata, gen_ai.system defaults to 'unknown'."""
    backend = OTelBackend()
    g = backend.generation("llm.call", model="m", input="i")
    g.end()
    spans = otel_capture.get_finished_spans()
    assert dict(spans[0].attributes)["gen_ai.system"] == "unknown"


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------

def test_set_output_truncates_to_cap(otel_capture):
    """gen_ai.completion is truncated to _TEXT_TRUNCATION_BYTES."""
    backend = OTelBackend()
    big = "x" * (_TEXT_TRUNCATION_BYTES + 1000)
    g = backend.generation("llm.call", model="m", input="i")
    g.set_output(big)
    g.end()
    spans = otel_capture.get_finished_spans()
    completion = dict(spans[0].attributes)["gen_ai.completion"]
    assert len(completion.encode("utf-8")) <= _TEXT_TRUNCATION_BYTES


def test_prompt_truncates_to_cap(otel_capture):
    """gen_ai.prompt is truncated to _TEXT_TRUNCATION_BYTES."""
    backend = OTelBackend()
    big = "y" * (_TEXT_TRUNCATION_BYTES + 5000)
    g = backend.generation("llm.call", model="m", input=big)
    g.end()
    spans = otel_capture.get_finished_spans()
    prompt = dict(spans[0].attributes)["gen_ai.prompt"]
    assert len(prompt.encode("utf-8")) <= _TEXT_TRUNCATION_BYTES


# ---------------------------------------------------------------------------
# Error semantics
# ---------------------------------------------------------------------------

def test_end_with_error_records_exception_and_sets_status_error(otel_capture):
    """OTelSpan.end(error=...) records the exception and sets ERROR status."""
    from opentelemetry.trace import StatusCode
    backend = OTelBackend()
    s = backend.span("op")
    s.end(status="error", error=ValueError("boom"))
    spans = otel_capture.get_finished_spans()
    assert len(spans) == 1
    sp = spans[0]
    assert sp.status.status_code == StatusCode.ERROR
    assert any(ev.name == "exception" for ev in sp.events)


def test_generation_end_with_error_records_exception(otel_capture):
    """OTelGeneration.end(error=...) records the exception and sets ERROR status."""
    from opentelemetry.trace import StatusCode
    backend = OTelBackend()
    g = backend.generation("llm.call", model="m", input="i")
    g.end(status="error", error=RuntimeError("kaboom"))
    spans = otel_capture.get_finished_spans()
    assert spans[0].status.status_code == StatusCode.ERROR
    assert any(ev.name == "exception" for ev in spans[0].events)


def test_trace_exit_with_exception_records_and_sets_error(otel_capture):
    """OTelTrace.__exit__ with exc_val records exception and sets ERROR status."""
    from opentelemetry.trace import StatusCode
    backend = OTelBackend()
    t = backend.trace("pipeline.run")
    err = ValueError("trace-fail")
    t.__exit__(type(err), err, None)
    spans = otel_capture.get_finished_spans()
    assert spans[0].status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Fail-open
# ---------------------------------------------------------------------------

def test_set_attribute_swallows_sdk_exception_and_warns(otel_capture, caplog):
    """If the inner span's set_attribute raises, OTelSpan must swallow + warn."""
    backend = OTelBackend()
    s = backend.span("op")
    # Force the underlying SDK call to raise.
    def boom(*_args, **_kwargs):
        raise RuntimeError("sdk-down")
    s._span.set_attribute = boom  # type: ignore[assignment]
    caplog.set_level(logging.WARNING, logger="rag.observability.otel")
    s.set_attribute("k", "v")  # must not raise
    assert any("OTelSpan.set_attribute failed" in r.message for r in caplog.records)
    s._span.set_attribute = lambda *_a, **_k: None  # restore harmless callable
    s.end()


# ---------------------------------------------------------------------------
# flush / shutdown — must propagate
# ---------------------------------------------------------------------------

def test_flush_propagates_processor_exception(otel_capture, monkeypatch):
    """OTelBackend.flush() must propagate exceptions from force_flush."""
    backend = OTelBackend()
    provider = otel_trace.get_tracer_provider()
    assert isinstance(provider, TracerProvider)

    def boom(*_a, **_k):
        raise RuntimeError("flush-fail")

    monkeypatch.setattr(provider, "force_flush", boom)
    with pytest.raises(RuntimeError, match="flush-fail"):
        backend.flush()


def test_shutdown_propagates_processor_exception(otel_capture, monkeypatch):
    """OTelBackend.shutdown() must propagate exceptions from provider.shutdown."""
    backend = OTelBackend()
    provider = otel_trace.get_tracer_provider()
    assert isinstance(provider, TracerProvider)

    def boom(*_a, **_k):
        raise RuntimeError("shutdown-fail")

    monkeypatch.setattr(provider, "shutdown", boom)
    with pytest.raises(RuntimeError, match="shutdown-fail"):
        backend.shutdown()


# ---------------------------------------------------------------------------
# ABC conformance
# ---------------------------------------------------------------------------

def test_isinstance_checks(otel_capture):
    """OTelBackend/Trace/Span/Generation satisfy their ABC contracts."""
    backend = OTelBackend()
    assert isinstance(backend, ObservabilityBackend)
    t = backend.trace("p")
    assert isinstance(t, Trace)
    s = backend.span("op", parent=t)
    assert isinstance(s, Span)
    g = backend.generation("llm.call", model="m", input="i")
    assert isinstance(g, Generation)
    s.end()
    g.end()
    t.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Ambient context propagation — implicit nesting via OTel current-context
# ---------------------------------------------------------------------------

def test_with_span_activates_ambient_context_for_children(otel_capture):
    """Entering a span via `with` must make subsequent backend.span() calls children."""
    backend = OTelBackend()
    with backend.span("outer.op"):
        inner = backend.span("inner.op")
        inner.end()
    spans = otel_capture.get_finished_spans()
    by_name = {s.name: s for s in spans}
    assert by_name["inner.op"].parent is not None
    assert by_name["inner.op"].parent.span_id == by_name["outer.op"].context.span_id
    assert by_name["inner.op"].context.trace_id == by_name["outer.op"].context.trace_id


def test_with_trace_activates_ambient_context_for_descendants(otel_capture):
    """A trace entered via `with` parents transitively across nested function calls."""
    backend = OTelBackend()

    def deep_work():
        with backend.span("deep.op"):
            with backend.generation("llm.call", model="m", input="p"):
                pass

    with backend.trace("pipeline.run"):
        deep_work()

    spans = otel_capture.get_finished_spans()
    by_name = {s.name: s for s in spans}
    root_id = by_name["pipeline.run"].context.span_id
    deep_id = by_name["deep.op"].context.span_id
    assert by_name["deep.op"].parent.span_id == root_id
    assert by_name["llm.call"].parent.span_id == deep_id
    # All share the same trace_id.
    trace_ids = {s.context.trace_id for s in spans}
    assert len(trace_ids) == 1


def test_context_detached_after_exit(otel_capture):
    """After exiting a span context, a new backend.span() must be a root again."""
    backend = OTelBackend()
    with backend.span("outer.op"):
        pass
    after = backend.span("after.op")
    after.end()
    spans = otel_capture.get_finished_spans()
    by_name = {s.name: s for s in spans}
    assert by_name["after.op"].parent is None


def test_explicit_parent_still_wins_over_ambient(otel_capture):
    """When parent= is passed explicitly, ambient context must be ignored."""
    backend = OTelBackend()
    t = backend.trace("root.a")
    with backend.span("ambient.outer"):
        # Explicit parent=trace overrides the currently-active ambient span.
        forced = backend.span("forced.under.root", parent=t)
        forced.end()
    t.__exit__(None, None, None)
    spans = otel_capture.get_finished_spans()
    by_name = {s.name: s for s in spans}
    assert by_name["forced.under.root"].parent.span_id == by_name["root.a"].context.span_id


# ---------------------------------------------------------------------------
# Encapsulation — no opentelemetry imports outside otel/backend.py
# ---------------------------------------------------------------------------

def test_no_opentelemetry_imports_outside_otel_backend():
    """Mirror REQ-201 from langfuse: SDK leakage is forbidden outside otel/backend.py."""
    pkg_root = pathlib.Path("src/platform/observability")
    allowed = (pkg_root / "otel" / "backend.py").resolve()
    pattern = re.compile(r"^\s*(from|import)\s+opentelemetry(\.|\s|$)", re.MULTILINE)
    offenders = []
    for path in pkg_root.rglob("*.py"):
        if path.resolve() == allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(str(path))
    assert offenders == [], f"opentelemetry imports leaked outside otel/backend.py: {offenders}"
