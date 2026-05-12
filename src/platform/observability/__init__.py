# @summary
# Public API facade for the observability subsystem.
# Exports: get_tracer, observe, Tracer, Span, Trace, Generation
# Deps: src.platform.observability.backend, threading, functools, config.settings
# @end-summary
"""Swappable observability subsystem — public API.

This module is the only import surface consumers should use:

    from src.platform.observability import get_tracer, observe

All other modules within this package are implementation details.

Backend selection:
    OBSERVABILITY_PROVIDER=noop     → NoopBackend (default)
    OBSERVABILITY_PROVIDER=langfuse → LangfuseBackend
    Any other value                 → ValueError at first get_tracer() call
"""
from __future__ import annotations

import functools
import threading
from typing import Callable, Optional, TypeVar

from config.settings import RAG_OBSERVABILITY_MAX_CAPTURE_LEN
from src.platform.observability.backend import (  # noqa: F401
    Generation,
    ObservabilityBackend,
    Span,
    Trace,
)

F = TypeVar("F", bound=Callable)

# Internal singleton state — do not access directly outside this module
_backend: Optional[ObservabilityBackend] = None
_backend_lock = threading.Lock()


def get_tracer() -> ObservabilityBackend:
    """Return the process-wide ObservabilityBackend singleton.

    Initializes the backend on first call using the OBSERVABILITY_PROVIDER
    environment variable. Subsequent calls return the same instance.
    Thread-safe via double-checked locking.

    Returns:
        The active ObservabilityBackend. Never raises — falls back to NoopBackend
        on initialization failure, logging a warning.
    """
    global _backend
    # Fast path: singleton already initialized (no lock needed)
    if _backend is not None:
        return _backend
    # Slow path: first call — acquire lock and initialize
    with _backend_lock:
        if _backend is None:  # Double-check under lock
            _backend = _init_backend()
    return _backend


def observe(
    name: Optional[str] = None,
    capture_input: bool = False,
    capture_output: bool = False,
) -> Callable[[F], F]:
    """Decorator factory that wraps a function with an observability span.

    Usage::

        @observe("reranker.rerank")
        def rerank(self, query, documents): ...

        @observe(capture_output=True)
        def generate(self, prompt): ...

    Args:
        name: Span name. Defaults to func.__qualname__ if not provided.
        capture_input: If True, records positional args (excluding self/cls)
            as "input" attribute, truncated to 500 chars. Defaults to False.
        capture_output: If True, records return value as "output" attribute,
            truncated to 500 chars. Defaults to False.

    Returns:
        A decorator that preserves __name__, __qualname__, __doc__ via functools.wraps.
        The wrapped function raises exceptions normally (no suppression).
    """
    def decorator(func: F) -> F:
        span_name = name if name is not None else func.__qualname__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            backend = get_tracer()
            with backend.span(span_name) as span:
                if capture_input and args:
                    span.set_attribute("input", repr(args[1:])[:RAG_OBSERVABILITY_MAX_CAPTURE_LEN])
                try:
                    result = func(*args, **kwargs)
                except Exception as exc:
                    span.set_attribute("error", str(exc))
                    raise
                if capture_output:
                    span.set_attribute("output", repr(result)[:RAG_OBSERVABILITY_MAX_CAPTURE_LEN])
                return result
        return wrapper  # type: ignore[return-value]

    return decorator


def _init_backend() -> ObservabilityBackend:
    """Initialize and return the backend based on OBSERVABILITY_PROVIDER.

    Called exactly once at first get_tracer() invocation. Not intended for
    direct use.

    Returns:
        The initialized ObservabilityBackend.

    Raises:
        ValueError: If OBSERVABILITY_PROVIDER contains an unrecognized value.

    Notes:
        Falls back to NoopBackend on any initialization error for the
        'langfuse' provider, logging a warning with the provider name and
        exception message.
    """
    import logging
    logger = logging.getLogger("rag.observability")

    try:
        from config.settings import OBSERVABILITY_PROVIDER
        provider = (OBSERVABILITY_PROVIDER or "").strip().lower()
    except ImportError:
        import os
        provider = os.environ.get("OBSERVABILITY_PROVIDER", "noop").strip().lower()

    if not provider or provider == "noop":
        from src.platform.observability.noop import NoopBackend
        return NoopBackend()

    if provider == "langfuse":
        try:
            from src.platform.observability.langfuse.backend import LangfuseBackend
            return LangfuseBackend()
        except Exception as exc:
            logger.warning(
                "Failed to initialize langfuse backend (%s); falling back to noop.",
                exc,
            )
            from src.platform.observability.noop import NoopBackend
            return NoopBackend()

    raise ValueError(
        f"Unknown OBSERVABILITY_PROVIDER: {provider!r}. "
        "Valid values: 'noop', 'langfuse'."
    )


# Backward-compatible alias — use ObservabilityBackend instead
Tracer = ObservabilityBackend

__all__ = ["get_tracer", "observe", "Tracer", "Span", "Trace", "Generation"]

# --- Auto-generated re-exports (fix_encapsulation.py) ---
from src.platform.observability.schemas import (
    GenerationRecord,
    TraceRecord,
)
