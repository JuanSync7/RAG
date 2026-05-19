"""Tests for src.platform.observability.__init__ — public API facade.

Covers:
- get_tracer() singleton factory (happy path, env var variants, unknown provider)
- @observe decorator (span lifecycle, capture_input/output, exception handling)
- functools.wraps preservation
- __all__ contents
- Thread-safety smoke test
- Boundary conditions (empty string, uppercase, no args, None name, langfuse fallback)
"""
from __future__ import annotations

import concurrent.futures
import logging
import threading
from typing import Optional
from unittest.mock import patch

import pytest

import src.platform.observability as obs_module
from src.platform.observability import get_tracer, observe
from src.platform.observability.backend import ObservabilityBackend, Span
from src.platform.observability.noop.backend import NoopBackend


# ---------------------------------------------------------------------------
# Spy classes for @observe tests (no live backend needed)
# ---------------------------------------------------------------------------

class SpySpan(Span):
    """A Span that records every call for assertion."""

    def __init__(self):
        self.attributes: dict = {}
        self.end_calls: list[tuple[str, Optional[Exception]]] = []
        self.name: str = ""

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def end(self, status: str = "ok", error: Optional[Exception] = None) -> None:
        self.end_calls.append((status, error))

    # Context manager protocol (inherited from Span base class, but
    # SpySpan.end is concrete so we override __exit__ here to record correctly)
    def __enter__(self) -> "SpySpan":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_val is not None:
            self.end(status="error", error=exc_val)
        else:
            self.end(status="ok")
        return False


class SpyBackend(ObservabilityBackend):
    """Backend that captures all span() calls into a list."""

    def __init__(self):
        self.spans: list[SpySpan] = []

    def span(self, name: str, attributes=None, parent=None) -> SpySpan:
        s = SpySpan()
        s.name = name
        self.spans.append(s)
        return s

    def trace(self, name: str, metadata=None):  # type: ignore[override]
        ...

    def generation(self, name: str, model: str, input: str, metadata=None):  # type: ignore[override]
        ...

    def flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level _backend singleton before and after each test."""
    obs_module._backend = None
    yield
    obs_module._backend = None


@pytest.fixture()
def spy_backend() -> SpyBackend:
    """Inject a SpyBackend as the active singleton and return it."""
    spy = SpyBackend()
    obs_module._backend = spy
    return spy


# ---------------------------------------------------------------------------
# Helper: patch config.settings.OBSERVABILITY_PROVIDER
# ---------------------------------------------------------------------------

def _patch_provider(monkeypatch, value: str):
    """Set OBSERVABILITY_PROVIDER in config.settings and os.environ."""
    import config.settings as settings_mod
    monkeypatch.setattr(settings_mod, "OBSERVABILITY_PROVIDER", value)
    monkeypatch.setenv("OBSERVABILITY_PROVIDER", value)


def _clear_provider(monkeypatch):
    """Unset OBSERVABILITY_PROVIDER in config.settings and os.environ."""
    import config.settings as settings_mod
    monkeypatch.setattr(settings_mod, "OBSERVABILITY_PROVIDER", "")
    monkeypatch.delenv("OBSERVABILITY_PROVIDER", raising=False)


# ===========================================================================
# 1. Happy-path: get_tracer() provider selection
# ===========================================================================

class TestGetTracerProviderSelection:
    """get_tracer() returns the correct backend type per OBSERVABILITY_PROVIDER."""

    def test_noop_provider_explicit(self, monkeypatch):
        """OBSERVABILITY_PROVIDER=noop returns a NoopBackend."""
        _patch_provider(monkeypatch, "noop")
        result = get_tracer()
        assert isinstance(result, NoopBackend)

    def test_env_var_unset_defaults_to_otel(self, monkeypatch):
        """No env var returns OTelBackend (new default)."""
        from src.platform.observability.otel import OTelBackend
        _clear_provider(monkeypatch)
        result = get_tracer()
        assert isinstance(result, OTelBackend)

    def test_otel_provider_explicit(self, monkeypatch):
        """OBSERVABILITY_PROVIDER=otel returns an OTelBackend."""
        from src.platform.observability.otel import OTelBackend
        _patch_provider(monkeypatch, "otel")
        result = get_tracer()
        assert isinstance(result, OTelBackend)

    def test_singleton_identity_same_object(self, monkeypatch):
        """Two calls to get_tracer() return the exact same instance."""
        _patch_provider(monkeypatch, "noop")
        first = get_tracer()
        second = get_tracer()
        assert first is second

    def test_singleton_identity_three_calls(self, monkeypatch):
        """Three calls all return the same instance."""
        _patch_provider(monkeypatch, "noop")
        refs = [get_tracer() for _ in range(3)]
        assert refs[0] is refs[1] is refs[2]


# ===========================================================================
# 2. Error scenarios: unknown provider
# ===========================================================================

class TestGetTracerUnknownProvider:
    """get_tracer() raises ValueError for unknown provider strings."""

    def test_unknown_provider_raises_value_error(self, monkeypatch):
        """Unrecognized provider raises ValueError."""
        _patch_provider(monkeypatch, "unknown_backend")
        with pytest.raises(ValueError, match="unknown_backend"):
            get_tracer()

    def test_unknown_provider_error_message_content(self, monkeypatch):
        """ValueError message mentions the unknown provider name."""
        _patch_provider(monkeypatch, "my_custom_provider")
        with pytest.raises(ValueError) as exc_info:
            get_tracer()
        assert "my_custom_provider" in str(exc_info.value)


# ===========================================================================
# 3. Boundary conditions: provider string variants
# ===========================================================================

class TestGetTracerBoundaryConditions:
    """Edge cases for provider string interpretation."""

    def test_empty_string_provider_returns_otel(self, monkeypatch):
        """OBSERVABILITY_PROVIDER='' (empty string) is treated as unset → OTelBackend (default)."""
        from src.platform.observability.otel import OTelBackend
        _patch_provider(monkeypatch, "")
        result = get_tracer()
        assert isinstance(result, OTelBackend)

    def test_uppercase_noop_returns_noop(self, monkeypatch):
        """OBSERVABILITY_PROVIDER='NOOP' (uppercase) is case-insensitive → NoopBackend."""
        _patch_provider(monkeypatch, "NOOP")
        result = get_tracer()
        assert isinstance(result, NoopBackend)

    def test_mixed_case_noop_returns_noop(self, monkeypatch):
        """OBSERVABILITY_PROVIDER='NoOp' is case-insensitive → NoopBackend."""
        _patch_provider(monkeypatch, "NoOp")
        result = get_tracer()
        assert isinstance(result, NoopBackend)

    def test_otel_init_failure_falls_back_to_noop(self, monkeypatch, caplog):
        """OTelBackend init failure falls back to NoopBackend and logs a warning."""
        _patch_provider(monkeypatch, "otel")

        with patch(
            "src.platform.observability.otel.OTelBackend",
            side_effect=RuntimeError("otel SDK broken"),
        ):
            with caplog.at_level(logging.WARNING, logger="rag.observability"):
                result = get_tracer()

        assert isinstance(result, NoopBackend)
        assert len(caplog.records) >= 1
        warning_msgs = " ".join(r.message for r in caplog.records)
        assert "otel" in warning_msgs.lower() or "noop" in warning_msgs.lower()

    def test_langfuse_alias_routes_to_otel(self, monkeypatch):
        """OBSERVABILITY_PROVIDER=langfuse is a deprecated alias for 'otel'."""
        import warnings
        from src.platform.observability.otel import OTelBackend
        _patch_provider(monkeypatch, "langfuse")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = get_tracer()
        assert isinstance(result, OTelBackend)
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deprecations, "Expected DeprecationWarning for legacy 'langfuse' value"


# ===========================================================================
# 4. @observe decorator — span lifecycle
# ===========================================================================

class TestObserveSpanLifecycle:
    """@observe opens and closes a span correctly."""

    def test_observe_with_explicit_name_opens_named_span(self, spy_backend):
        """@observe('my.span') passes 'my.span' to backend.span()."""
        @observe("my.span")
        def func():
            return 42

        func()
        assert len(spy_backend.spans) == 1
        assert spy_backend.spans[0].name == "my.span"

    def test_observe_span_ended_after_call(self, spy_backend):
        """Span is ended with status='ok' when function completes normally."""
        @observe("test.op")
        def func():
            return "result"

        func()
        span = spy_backend.spans[0]
        assert len(span.end_calls) == 1
        status, error = span.end_calls[0]
        assert status == "ok"
        assert error is None

    def test_observe_no_name_uses_qualname(self, spy_backend):
        """@observe() without a name uses func.__qualname__ as span name."""
        @observe()
        def my_test_function():
            pass

        my_test_function()
        # __qualname__ inside a method body includes the enclosing class/method path
        assert spy_backend.spans[0].name == my_test_function.__qualname__

    def test_observe_none_name_uses_qualname(self, spy_backend):
        """@observe(name=None) explicitly uses func.__qualname__ as span name."""
        @observe(name=None)
        def another_function():
            pass

        another_function()
        # __qualname__ inside a method body includes the enclosing class/method path
        assert spy_backend.spans[0].name == another_function.__qualname__

    def test_observe_returns_function_result(self, spy_backend):
        """@observe wraps transparently — return value is unchanged."""
        @observe("op")
        def func():
            return {"key": "value"}

        result = func()
        assert result == {"key": "value"}

    def test_observe_called_multiple_times_creates_multiple_spans(self, spy_backend):
        """Each call to a decorated function creates a new span."""
        @observe("repeated")
        def func():
            pass

        func()
        func()
        func()
        assert len(spy_backend.spans) == 3


# ===========================================================================
# 5. @observe decorator — capture_input
# ===========================================================================

class TestObserveCaptureInput:
    """@observe(capture_input=True) records positional args excluding self."""

    def test_capture_input_sets_attribute(self, spy_backend):
        """'input' attribute is set when capture_input=True and args provided."""
        @observe("op", capture_input=True)
        def func(self_arg, a, b):
            pass

        func("self_value", "hello", "world")
        span = spy_backend.spans[0]
        assert "input" in span.attributes

    def test_capture_input_excludes_first_arg(self, spy_backend):
        """'input' attribute uses args[1:], excluding first positional arg (self)."""
        @observe("op", capture_input=True)
        def func(self_arg, a, b):
            pass

        func("self_val", "arg_a", "arg_b")
        span = spy_backend.spans[0]
        # args[1:] = ("arg_a", "arg_b") — self_val must NOT appear
        assert "self_val" not in span.attributes.get("input", "")
        assert "arg_a" in span.attributes.get("input", "")

    def test_capture_input_no_positional_args_skips_attribute(self, spy_backend):
        """'input' attribute is NOT set when only kwargs passed (args is empty/just self)."""
        @observe("op", capture_input=True)
        def func(**kwargs):
            pass

        func(x=1, y=2)
        span = spy_backend.spans[0]
        assert "input" not in span.attributes

    def test_capture_input_only_self_skips_attribute(self, spy_backend):
        """'input' NOT set when no positional args at all (empty args tuple)."""
        @observe("op", capture_input=True)
        def func():
            pass

        func()
        span = spy_backend.spans[0]
        # args = () — falsy, so the `if capture_input and args:` guard prevents setting
        assert "input" not in span.attributes

    def test_capture_input_truncated_to_500_chars(self, spy_backend):
        """'input' attribute is truncated to exactly 500 chars when repr is longer."""
        @observe("op", capture_input=True)
        def func(self_arg, big_arg):
            pass

        long_arg = "x" * 600
        func("self", long_arg)
        span = spy_backend.spans[0]
        assert "input" in span.attributes
        assert len(span.attributes["input"]) == 500

    def test_capture_input_false_no_attribute_set(self, spy_backend):
        """'input' attribute not set when capture_input=False (default)."""
        @observe("op")
        def func(self_arg, a):
            pass

        func("self", "value")
        span = spy_backend.spans[0]
        assert "input" not in span.attributes


# ===========================================================================
# 6. @observe decorator — capture_output
# ===========================================================================

class TestObserveCaptureOutput:
    """@observe(capture_output=True) records the return value."""

    def test_capture_output_sets_attribute(self, spy_backend):
        """'output' attribute set when capture_output=True."""
        @observe("op", capture_output=True)
        def func():
            return "hello"

        func()
        span = spy_backend.spans[0]
        assert "output" in span.attributes

    def test_capture_output_value_matches_repr(self, spy_backend):
        """'output' attribute matches repr(return_value)[:500]."""
        @observe("op", capture_output=True)
        def func():
            return [1, 2, 3]

        func()
        span = spy_backend.spans[0]
        assert span.attributes["output"] == repr([1, 2, 3])[:500]

    def test_capture_output_truncated_to_500_chars(self, spy_backend):
        """'output' attribute is truncated to exactly 500 chars when repr is longer."""
        @observe("op", capture_output=True)
        def func():
            return "z" * 600

        func()
        span = spy_backend.spans[0]
        assert "output" in span.attributes
        assert len(span.attributes["output"]) == 500

    def test_capture_output_false_no_attribute_set(self, spy_backend):
        """'output' attribute not set when capture_output=False (default)."""
        @observe("op")
        def func():
            return "something"

        func()
        span = spy_backend.spans[0]
        assert "output" not in span.attributes


# ===========================================================================
# 7. @observe decorator — exception handling
# ===========================================================================

class TestObserveExceptionHandling:
    """@observe sets error attribute on span and re-raises exceptions."""

    def test_exception_propagates_to_caller(self, spy_backend):
        """RuntimeError raised inside decorated function propagates to caller."""
        @observe("op")
        def func():
            raise RuntimeError("test error")

        with pytest.raises(RuntimeError, match="test error"):
            func()

    def test_exception_sets_error_attribute(self, spy_backend):
        """'error' attribute is set on the span with str(exc) on exception."""
        @observe("op")
        def func():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            func()

        span = spy_backend.spans[0]
        assert span.attributes.get("error") == "boom"

    def test_exception_span_ended_with_error_status(self, spy_backend):
        """Span is ended with status='error' when exception occurs."""
        @observe("op")
        def func():
            raise ValueError("bad input")

        with pytest.raises(ValueError):
            func()

        span = spy_backend.spans[0]
        assert len(span.end_calls) == 1
        status, _ = span.end_calls[0]
        assert status == "error"

    def test_exception_does_not_set_output_attribute(self, spy_backend):
        """'output' attribute is NOT set when function raises (no return value)."""
        @observe("op", capture_output=True)
        def func():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            func()

        span = spy_backend.spans[0]
        assert "output" not in span.attributes

    def test_different_exception_types_propagate(self, spy_backend):
        """Various exception types all propagate correctly."""
        for exc_class, exc_msg in [
            (ValueError, "val error"),
            (TypeError, "type error"),
            (KeyError, "key"),
        ]:
            obs_module._backend = SpyBackend()

            @observe("op")
            def func():
                raise exc_class(exc_msg)

            with pytest.raises(exc_class):
                func()


# ===========================================================================
# 8. functools.wraps preservation
# ===========================================================================

class TestFunctoolsWraps:
    """@observe preserves __name__, __doc__, __qualname__ via functools.wraps."""

    def test_name_preserved(self, spy_backend):
        """Decorated function.__name__ matches original."""
        @observe("op")
        def my_func():
            """My docstring."""
            pass

        assert my_func.__name__ == "my_func"

    def test_qualname_preserved(self, spy_backend):
        """Decorated function.__qualname__ matches original (includes enclosing scope)."""
        @observe("op")
        def my_func():
            pass

        # __qualname__ inside a method includes the enclosing class + method path;
        # functools.wraps preserves it faithfully from the original function
        assert "my_func" in my_func.__qualname__

    def test_doc_preserved(self, spy_backend):
        """Decorated function.__doc__ matches original."""
        @observe("op")
        def my_func():
            """Original docstring."""
            pass

        assert my_func.__doc__ == "Original docstring."

    def test_wraps_inside_class(self, spy_backend):
        """Decorated method inside a class preserves __qualname__ correctly."""
        class MyClass:
            @observe("op")
            def my_method(self):
                """Method doc."""
                pass

        obj = MyClass()
        assert "my_method" in obj.my_method.__qualname__
        assert obj.my_method.__doc__ == "Method doc."


# ===========================================================================
# 9. __all__ contents
# ===========================================================================

class TestAllExports:
    """__all__ contains exactly the documented public symbols."""

    def test_all_contains_get_tracer(self):
        assert "get_tracer" in obs_module.__all__

    def test_all_contains_observe(self):
        assert "observe" in obs_module.__all__

    def test_all_contains_tracer(self):
        assert "Tracer" in obs_module.__all__

    def test_all_contains_span(self):
        assert "Span" in obs_module.__all__

    def test_all_contains_trace(self):
        assert "Trace" in obs_module.__all__

    def test_all_contains_generation(self):
        assert "Generation" in obs_module.__all__

    def test_all_has_exactly_six_symbols(self):
        """__all__ contains exactly the 6 expected symbols (no extras, no missing)."""
        expected = {"get_tracer", "observe", "Tracer", "Span", "Trace", "Generation"}
        assert set(obs_module.__all__) == expected


# ===========================================================================
# 10. Thread safety — smoke test
# ===========================================================================

class TestThreadSafety:
    """Concurrent get_tracer() calls return the same singleton instance."""

    def test_concurrent_get_tracer_same_instance(self, monkeypatch):
        """50 concurrent threads calling get_tracer() all get the same object."""
        _patch_provider(monkeypatch, "noop")

        results: list[ObservabilityBackend] = []
        lock = threading.Lock()

        def call_get_tracer():
            tracer = get_tracer()
            with lock:
                results.append(tracer)

        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(call_get_tracer) for _ in range(50)]
            concurrent.futures.wait(futures)

        assert len(results) == 50
        first = results[0]
        assert all(r is first for r in results), (
            "Not all threads received the same singleton instance"
        )


# ===========================================================================
# 11. Integration: _init_backend called once, subsequent calls skip init
# ===========================================================================

class TestSingletonInitialization:
    """_init_backend() is invoked only on the first get_tracer() call."""

    def test_init_backend_called_once(self, monkeypatch):
        """_init_backend() is called exactly once across multiple get_tracer() calls."""
        _patch_provider(monkeypatch, "noop")
        call_count = 0
        original_init = obs_module._init_backend

        def counting_init():
            nonlocal call_count
            call_count += 1
            return original_init()

        with patch.object(obs_module, "_init_backend", side_effect=counting_init):
            get_tracer()
            get_tracer()
            get_tracer()

        assert call_count == 1

    def test_pre_injected_backend_skips_init(self):
        """If _backend is already set, get_tracer() returns it without calling _init_backend."""
        spy = SpyBackend()
        obs_module._backend = spy

        with patch.object(obs_module, "_init_backend", side_effect=AssertionError("should not be called")):
            result = get_tracer()

        assert result is spy


# ===========================================================================
# 12. Integration: @observe calls get_tracer() each invocation
# ===========================================================================

class TestObserveCallsGetTracer:
    """@observe delegates span creation to get_tracer() on each call."""

    def test_observe_uses_active_backend(self, spy_backend):
        """@observe retrieves the active backend and creates a span through it."""
        @observe("integration.test")
        def func():
            return "ok"

        func()
        # Verify the span went through our SpyBackend
        assert len(spy_backend.spans) == 1
        assert spy_backend.spans[0].name == "integration.test"

    def test_observe_span_is_context_manager(self, spy_backend):
        """Span returned by backend.span() is used as a context manager."""
        @observe("ctx.test")
        def func():
            pass

        func()
        span = spy_backend.spans[0]
        # end() is called by __exit__, so end_calls should be populated
        assert len(span.end_calls) >= 1
