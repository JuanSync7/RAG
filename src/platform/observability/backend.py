# @summary
# Abstract base class definitions for the observability subsystem.
# Exports: ObservabilityBackend, Span, Trace, Generation, Tracer
# Deps: abc, typing
# @end-summary
"""Abstract base classes for the swappable observability subsystem.

Defines the provider-agnostic contract for Span, Trace, Generation,
and ObservabilityBackend. All backend implementations must subclass
these ABCs. Consumers interact only with these types — never with
concrete provider implementations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class Span(ABC):
    """Abstract tracing span for a single timed operation.

    Supports both direct lifecycle management and use as a context manager.
    All concrete implementations must be fail-open: exceptions raised inside
    set_attribute or end must be caught internally and never propagated.
    """

    @abstractmethod
    def set_attribute(self, key: str, value: object) -> None:
        """Set a key-value attribute on this span.

        Args:
            key: Attribute name. Must be a snake_case string.
            value: Attribute value. Any Python object accepted.

        Returns:
            None. Never raises — fail-open contract.
        """

    @abstractmethod
    def end(self, status: str = "ok", error: Optional[Exception] = None) -> None:
        """Finalize this span.

        Args:
            status: "ok" for successful completion, "error" for failures.
            error: The exception that caused the failure, if any.

        Returns:
            None. Never raises — fail-open contract.
        """

    def __enter__(self) -> "Span":
        """Return self to enable use as a context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """End the span on context manager exit. Returns False (never suppresses exceptions)."""
        if exc_val is not None:
            self.end(status="error", error=exc_val)
        else:
            self.end(status="ok")
        return False


class Generation(ABC):
    """Abstract tracing generation for a single LLM call.

    A Generation is a specialised Span that additionally captures LLM-specific
    fields: prompt input, completion output, model name, and token counts.
    """

    @abstractmethod
    def set_output(self, output: str) -> None:
        """Record the LLM completion output.

        Args:
            output: The model completion text. Overwrites any previous value.

        Returns:
            None. Never raises — fail-open contract.
        """

    @abstractmethod
    def set_token_counts(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record token usage for this generation.

        Args:
            prompt_tokens: Number of tokens in the input prompt.
            completion_tokens: Number of tokens in the model completion.

        Returns:
            None. Never raises — fail-open contract.
        """

    @abstractmethod
    def end(self, status: str = "ok", error: Optional[Exception] = None) -> None:
        """Finalize this generation record.

        Args:
            status: "ok" for successful completion, "error" for failures.
            error: The exception that caused the failure, if any.

        Returns:
            None. Never raises — fail-open contract.
        """

    def __enter__(self) -> "Generation":
        """Return self to enable use as a context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """End the generation on context manager exit. Returns False."""
        if exc_val is not None:
            self.end(status="error", error=exc_val)
        else:
            self.end(status="ok")
        return False


class Trace(ABC):
    """Abstract trace root — a logical grouping of spans and generations.

    A Trace represents one request or pipeline run. All child spans and
    generations created through this object share the same trace_id.
    """

    @abstractmethod
    def span(self, name: str, attributes: Optional[dict] = None) -> Span:
        """Create a child span under this trace.

        Args:
            name: Span name. Convention: "component.operation".
            attributes: Optional initial attributes. Defaults to empty dict.

        Returns:
            A Span instance correlated to this trace. Never raises — fail-open contract.
        """

    @abstractmethod
    def generation(
        self,
        name: str,
        model: str,
        input: str,
        metadata: Optional[dict] = None,
    ) -> Generation:
        """Create a child generation (LLM call) under this trace.

        Args:
            name: Generation name.
            model: Model identifier (e.g., "gpt-4o", "claude-3-5-sonnet").
            input: The prompt text sent to the model.
            metadata: Optional additional metadata.

        Returns:
            A Generation instance correlated to this trace. Never raises — fail-open.
        """

    def __enter__(self) -> "Trace":
        """Return self to enable use as a context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit the trace context. Returns False (never suppresses exceptions)."""
        return False


class ObservabilityBackend(ABC):
    """Abstract base class for all observability backend providers.

    Implementations must be substitutable without changes to consumers.
    The active backend is a process-wide singleton accessed via get_tracer().
    """

    @abstractmethod
    def span(
        self,
        name: str,
        attributes: Optional[dict] = None,
        parent: Optional[Span] = None,
    ) -> Span:
        """Start and return a new span.

        Args:
            name: Span name. Convention: "component.operation".
            attributes: Optional initial attributes dict.
            parent: Optional parent span for nesting.

        Returns:
            A Span instance. Never raises — fail-open contract.
        """

    @abstractmethod
    def trace(self, name: str, metadata: Optional[dict] = None) -> Trace:
        """Start and return a new trace root.

        Args:
            name: Trace name. Convention: "pipeline.operation".
            metadata: Optional metadata dict attached to the trace root.

        Returns:
            A Trace instance. Never raises — fail-open contract.
        """

    @abstractmethod
    def generation(
        self,
        name: str,
        model: str,
        input: str,
        metadata: Optional[dict] = None,
    ) -> Generation:
        """Start and return a new generation (LLM call tracking).

        Args:
            name: Generation name.
            model: Model identifier.
            input: The prompt text sent to the model.
            metadata: Optional additional metadata.

        Returns:
            A Generation instance. Never raises — fail-open contract.
        """

    @abstractmethod
    def flush(self) -> None:
        """Drain all pending buffered observations to the backend.

        Blocks until all buffered data is flushed. Propagates exceptions
        from the underlying SDK (callers must handle timeout/network errors).

        Returns:
            None.

        Raises:
            Any exception raised by the underlying SDK flush operation.
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Gracefully shut down the backend client.

        Called on process exit. Propagates exceptions from the underlying SDK.

        Returns:
            None.

        Raises:
            Any exception raised by the underlying SDK shutdown operation.
        """

    def start_trace_from_carrier(
        self,
        name: str,
        carrier: dict,
        metadata: Optional[dict] = None,
    ) -> Trace:
        """Start a trace whose parent context is extracted from a W3C carrier.

        If the carrier contains a valid ``traceparent`` header (per the W3C
        trace-context specification), the returned Trace shares its trace_id
        with the inbound span and is parented under it. If the header is
        absent, malformed, or extraction is unsupported, this method falls
        back to :meth:`trace` (a fresh root).

        Subclasses with real propagator support (e.g. OTel) override this.
        The default implementation is the fallback path used by NoopBackend
        and any backend that does not implement carrier propagation.

        Args:
            name: Trace name. Convention: "api.<resource>.<verb>".
            carrier: Request header mapping (e.g. ``dict(request.headers)``)
                conforming to W3C trace-context (``traceparent`` /
                ``tracestate``). Case-insensitive lookup is the
                implementation's responsibility.
            metadata: Optional metadata dict attached to the trace root.

        Returns:
            A Trace instance. Never raises — fail-open contract.
        """
        return self.trace(name, metadata)

    def start_span(
        self,
        name: str,
        attributes: Optional[dict] = None,
        parent: Optional[Span] = None,
    ) -> Span:
        """Deprecated alias for span(). Use span() instead.

        This method exists for backward compatibility during migration from
        the old observability API. It will be removed in a future release.
        """
        import warnings
        warnings.warn(
            "start_span() is deprecated; use span() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.span(name, attributes, parent)


# Backward-compatible alias — deprecated, use ObservabilityBackend
Tracer = ObservabilityBackend
