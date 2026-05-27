"""Integration tests for the Swappable Observability Subsystem end-to-end flows.

Covers:
    Scenario 3.1 — Happy Path: Full noop pipeline
    Scenario 3.2 — OTel Fallback: OTelBackend init failure → NoopBackend
                   (and legacy 'langfuse' provider value routes to OTel with a
                    DeprecationWarning)
    Scenario 3.3 — @observe Decorator Error Path
"""
import logging
import warnings

import pytest
from unittest.mock import patch

import src.platform.observability as obs_module
from src.platform.observability import get_tracer, observe
from src.platform.observability.noop.backend import NoopBackend, NoopSpan, NoopTrace, NoopGeneration
from src.platform.observability.backend import ObservabilityBackend, Span, Trace, Generation


# ---------------------------------------------------------------------------
# SpyBackend / SpySpan test doubles (module level)
# ---------------------------------------------------------------------------

class SpySpan(Span):
    """Test double for Span — records all calls for assertion."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.attributes: dict = {}
        self.end_calls: list = []

    def set_attribute(self, key: str, value) -> None:
        self.attributes[key] = value

    def end(self, status: str = "ok", error=None) -> None:
        self.end_calls.append({"status": status, "error": error})


class SpyBackend(ObservabilityBackend):
    """Test double for ObservabilityBackend — records span calls."""

    def __init__(self) -> None:
        self.span_calls: list = []
        self._last_span: SpySpan = None

    def span(self, name: str, attributes=None, parent=None) -> SpySpan:
        s = SpySpan(name)
        self.span_calls.append(name)
        self._last_span = s
        return s

    def trace(self, name: str, metadata=None) -> Trace:
        return NoopTrace()

    def generation(self, name: str, model: str, input: str, metadata=None) -> Generation:
        return NoopGeneration()

    def flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the observability singleton before and after every test."""
    obs_module._backend = None
    yield
    obs_module._backend = None


# ---------------------------------------------------------------------------
# --- Integration: Scenario 3.1 Happy Path ---
# ---------------------------------------------------------------------------

class TestScenario31NooopHappyPath:
    """Scenario 3.1 — get_tracer() with OBSERVABILITY_PROVIDER=noop → full noop pipeline.

    config.settings reads RAG_OBSERVABILITY_PROVIDER (not OBSERVABILITY_PROVIDER),
    so we set both to ensure the correct branch is taken regardless of whether
    config.settings is importable.
    """

    def test_get_tracer_returns_noop_backend(self, monkeypatch):
        monkeypatch.setenv("RAG_OBSERVABILITY_PROVIDER", "noop")
        monkeypatch.setenv("OBSERVABILITY_PROVIDER", "noop")
        backend = get_tracer()
        assert isinstance(backend, NoopBackend)

    def test_span_returns_noop_span_instance(self, monkeypatch):
        monkeypatch.setenv("RAG_OBSERVABILITY_PROVIDER", "noop")
        monkeypatch.setenv("OBSERVABILITY_PROVIDER", "noop")
        backend = get_tracer()
        span = backend.span("test.span")
        assert isinstance(span, Span), "span() must return a Span ABC instance"
        assert isinstance(span, NoopSpan)

    def test_span_set_attribute_no_exception(self, monkeypatch):
        monkeypatch.setenv("RAG_OBSERVABILITY_PROVIDER", "noop")
        monkeypatch.setenv("OBSERVABILITY_PROVIDER", "noop")
        backend = get_tracer()
        span = backend.span("test.span")
        # Must not raise
        span.set_attribute("key", "value")

    def test_span_end_no_exception(self, monkeypatch):
        monkeypatch.setenv("RAG_OBSERVABILITY_PROVIDER", "noop")
        monkeypatch.setenv("OBSERVABILITY_PROVIDER", "noop")
        backend = get_tracer()
        span = backend.span("test.span")
        result = span.end(status="ok")
        assert result is None

    def test_trace_returns_trace_instance(self, monkeypatch):
        monkeypatch.setenv("RAG_OBSERVABILITY_PROVIDER", "noop")
        monkeypatch.setenv("OBSERVABILITY_PROVIDER", "noop")
        backend = get_tracer()
        trace = backend.trace("test.trace")
        assert isinstance(trace, Trace)
        assert isinstance(trace, NoopTrace)

    def test_generation_returns_generation_instance(self, monkeypatch):
        monkeypatch.setenv("RAG_OBSERVABILITY_PROVIDER", "noop")
        monkeypatch.setenv("OBSERVABILITY_PROVIDER", "noop")
        backend = get_tracer()
        gen = backend.generation("g", "gpt-4", "hello")
        assert isinstance(gen, Generation)
        assert isinstance(gen, NoopGeneration)

    def test_flush_returns_none_no_exception(self, monkeypatch):
        monkeypatch.setenv("RAG_OBSERVABILITY_PROVIDER", "noop")
        monkeypatch.setenv("OBSERVABILITY_PROVIDER", "noop")
        backend = get_tracer()
        result = backend.flush()
        assert result is None

    def test_shutdown_returns_none_no_exception(self, monkeypatch):
        monkeypatch.setenv("RAG_OBSERVABILITY_PROVIDER", "noop")
        monkeypatch.setenv("OBSERVABILITY_PROVIDER", "noop")
        backend = get_tracer()
        result = backend.shutdown()
        assert result is None

    def test_singleton_identity(self, monkeypatch):
        """Second call to get_tracer() must return the same instance."""
        monkeypatch.setenv("RAG_OBSERVABILITY_PROVIDER", "noop")
        monkeypatch.setenv("OBSERVABILITY_PROVIDER", "noop")
        first = get_tracer()
        second = get_tracer()
        assert first is second, "get_tracer() must return the same singleton instance"

    def test_full_pipeline_no_exceptions(self, monkeypatch):
        """End-to-end: all calls in sequence raise no exceptions."""
        monkeypatch.setenv("RAG_OBSERVABILITY_PROVIDER", "noop")
        monkeypatch.setenv("OBSERVABILITY_PROVIDER", "noop")
        backend = get_tracer()
        span = backend.span("test.span")
        span.set_attribute("key", "value")
        span.end(status="ok")
        trace = backend.trace("test.trace")
        gen = backend.generation("g", "gpt-4", "hello")
        backend.flush()
        backend.shutdown()
        # All isinstance checks
        assert isinstance(span, Span)
        assert isinstance(trace, Trace)
        assert isinstance(gen, Generation)


# ---------------------------------------------------------------------------
# --- Integration: Scenario 3.2 OTel Fallback + Legacy Langfuse Alias ---
# ---------------------------------------------------------------------------

def _patch_provider(value: str):
    """Context manager: patch config.settings.OBSERVABILITY_PROVIDER.

    config.settings is a module-level singleton already imported; setting the
    env var alone does not retroactively change OBSERVABILITY_PROVIDER, so we
    patch the module attribute directly as well.
    """
    import config.settings as _settings
    return patch.object(_settings, "OBSERVABILITY_PROVIDER", value)


class TestScenario32OTelFallback:
    """Scenario 3.2 — OTelBackend init failure → fallback to NoopBackend.

    The OTel adapter has replaced the broken langfuse SDK adapter. When the
    OTel backend constructor raises (e.g. due to a broken SDK install), the
    factory must log a WARNING and return NoopBackend so that the
    application never crashes for observability reasons.
    """

    def test_get_tracer_does_not_raise_on_otel_failure(self):
        with _patch_provider("otel"):
            with patch(
                "src.platform.observability.otel.OTelBackend",
                side_effect=RuntimeError("otel sdk broken"),
            ):
                backend = get_tracer()
        assert backend is not None

    def test_fallback_returns_noop_backend(self):
        with _patch_provider("otel"):
            with patch(
                "src.platform.observability.otel.OTelBackend",
                side_effect=RuntimeError("otel sdk broken"),
            ):
                backend = get_tracer()
        assert isinstance(backend, NoopBackend), (
            "Fallback must produce a NoopBackend, got: %s" % type(backend)
        )

    def test_fallback_span_returns_span_instance(self):
        with _patch_provider("otel"):
            with patch(
                "src.platform.observability.otel.OTelBackend",
                side_effect=RuntimeError("otel sdk broken"),
            ):
                backend = get_tracer()
        span = backend.span("x")
        assert isinstance(span, Span)

    def test_warning_logged_on_fallback(self, caplog):
        with caplog.at_level(logging.WARNING, logger="rag.observability"):
            with _patch_provider("otel"):
                with patch(
                    "src.platform.observability.otel.OTelBackend",
                    side_effect=RuntimeError("otel sdk broken"),
                ):
                    get_tracer()
        assert any(
            record.levelno >= logging.WARNING for record in caplog.records
        ), "A WARNING must be logged when falling back from otel to noop"


class TestScenario32LangfuseAlias:
    """The legacy 'langfuse' provider value still works (deprecation warning)."""

    def test_langfuse_alias_routes_to_otel_backend(self):
        from src.platform.observability.otel import OTelBackend
        with _patch_provider("langfuse"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                backend = get_tracer()
        assert isinstance(backend, OTelBackend)

    def test_langfuse_alias_emits_deprecation_warning(self):
        with _patch_provider("langfuse"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                get_tracer()
            deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
            assert deprecations, "Expected a DeprecationWarning for legacy 'langfuse' value"
            msg = str(deprecations[0].message).lower()
            assert "langfuse" in msg and "otel" in msg


# ---------------------------------------------------------------------------
# --- Integration: Scenario 3.3 @observe Decorator Error Path ---
# ---------------------------------------------------------------------------

class TestScenario33ObserveDecoratorErrorPath:
    """Scenario 3.3 — @observe-decorated function raises → span records error → exception re-raised."""

    def _make_spy_backend(self) -> SpyBackend:
        spy = SpyBackend()
        obs_module._backend = spy
        return spy

    def test_exception_is_reraised(self):
        spy = self._make_spy_backend()

        @observe("test.operation")
        def failing_func():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            failing_func()

    def test_error_attribute_set_on_span(self):
        spy = self._make_spy_backend()

        @observe("test.operation")
        def failing_func():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            failing_func()

        assert spy._last_span is not None, "Expected a span to be created"
        assert spy._last_span.attributes.get("error") == "boom", (
            "Expected span attribute 'error' == 'boom', got: %s" % spy._last_span.attributes
        )

    def test_span_end_called_with_error_status(self):
        spy = self._make_spy_backend()

        @observe("test.operation")
        def failing_func():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            failing_func()

        end_calls = spy._last_span.end_calls
        assert len(end_calls) == 1, "Expected exactly one end() call"
        assert end_calls[0]["status"] == "error", (
            "Expected end(status='error'), got: %s" % end_calls[0]["status"]
        )

    def test_span_end_called_with_exception_object(self):
        spy = self._make_spy_backend()

        @observe("test.operation")
        def failing_func():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            failing_func()

        end_calls = spy._last_span.end_calls
        assert isinstance(end_calls[0]["error"], RuntimeError), (
            "Expected end() to receive the RuntimeError instance"
        )

    def test_span_name_matches_observe_argument(self):
        spy = self._make_spy_backend()

        @observe("test.operation")
        def failing_func():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            failing_func()

        assert spy._last_span.name == "test.operation", (
            "Span name must be 'test.operation', got: %s" % spy._last_span.name
        )

    def test_observe_preserves_dunder_name(self):
        spy = self._make_spy_backend()

        @observe("test.operation")
        def my_special_func():
            pass

        assert my_special_func.__name__ == "my_special_func", (
            "@observe must preserve __name__ via functools.wraps"
        )

    def test_observe_preserves_dunder_qualname(self):
        spy = self._make_spy_backend()

        @observe("test.operation")
        def my_special_func():
            pass

        assert "my_special_func" in my_special_func.__qualname__, (
            "@observe must preserve __qualname__ via functools.wraps"
        )

    def test_spy_span_is_span_abc_instance(self):
        """Sanity check: SpySpan satisfies the Span ABC contract."""
        s = SpySpan("check")
        assert isinstance(s, Span)

    def test_spy_backend_is_backend_abc_instance(self):
        """Sanity check: SpyBackend satisfies the ObservabilityBackend ABC contract."""
        b = SpyBackend()
        assert isinstance(b, ObservabilityBackend)
