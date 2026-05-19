# @summary
# OpenTelemetry-native backend implementation for the observability subsystem.
# Exports: OTelBackend, OTelSpan, OTelTrace, OTelGeneration
# Deps: opentelemetry-api, opentelemetry-sdk, opentelemetry-exporter-otlp-proto-http,
#       src.platform.observability.backend
# @end-summary
"""OpenTelemetry-native observability backend.

All imports from the ``opentelemetry`` package are confined exclusively to
this file. Consumer code must never import from this module directly — use
the public API at :mod:`src.platform.observability` instead.

This backend speaks OTLP/HTTP and therefore reaches any OTel-compatible
ingestor — Langfuse v3 (``/api/public/otel/v1/traces``), Phoenix, Braintrust,
Arize, Honeycomb, Datadog, and so on — via the standard
``OTEL_EXPORTER_OTLP_ENDPOINT`` environment variable. When the env var is
unset, no exporter is registered and spans are dropped silently (the global
ProxyTracerProvider's tracers are no-ops).

Mapping the project ABC contract onto OTel:

    backend.trace(name)          -> tracer.start_span(name) wrapped as OTelTrace
    backend.span(name, parent)   -> tracer.start_span(name, context=...) wrapped as OTelSpan
    backend.generation(...)      -> tracer.start_span(name) with gen_ai.* attributes
    trace.span(name)             -> child span under the root span context
    trace.generation(...)        -> child generation under the root span context

LLM-specific fields are emitted using the OpenTelemetry gen_ai semantic
conventions, which the Langfuse v3 OTLP ingestor (and most other OTel-aware
LLM observability tools) understand natively. See
https://opentelemetry.io/docs/specs/semconv/gen-ai/ for the schema.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

from src.platform.observability.backend import (
    Generation,
    ObservabilityBackend,
    Span,
    Trace,
)

logger = logging.getLogger("rag.observability.otel")


# Truncation cap for free-text attributes (prompt/completion). 8 KiB is the
# documented cap in the spec for this backend — it keeps individual spans
# well under typical collector size limits (Langfuse and Phoenix accept much
# more, but Honeycomb caps span size at ~100 KB total and Datadog at 1 MB —
# 8 KiB per text field leaves headroom for many such fields).
_TEXT_TRUNCATION_BYTES = 8 * 1024


def _truncate_text(value: str, cap: int = _TEXT_TRUNCATION_BYTES) -> str:
    """Truncate a string to at most ``cap`` bytes (UTF-8), preserving validity.

    Args:
        value: Input string.
        cap: Maximum byte length.

    Returns:
        The original string if it fits, otherwise a UTF-8-safe prefix.
    """
    if value is None:
        return ""
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= cap:
        return value
    # Decode with errors='ignore' to drop any partial multi-byte character
    # at the boundary.
    return encoded[:cap].decode("utf-8", errors="ignore")


def _coerce_attribute_value(value: object) -> object:
    """Coerce an arbitrary Python value into an OTel-acceptable attribute value.

    OTel attribute values must be primitives (str, bool, int, float) or
    homogeneous sequences thereof. Anything else is repr()'d and truncated.

    Args:
        value: Any Python object.

    Returns:
        A value safe to pass to ``span.set_attribute``.
    """
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, (str, bool, int, float)) for v in value):
            return list(value)
    return _truncate_text(repr(value))


class OTelSpan(Span):
    """OpenTelemetry-backed span wrapper.

    Wraps an ``opentelemetry.trace.Span``. All SDK calls are fail-open:
    exceptions are caught and logged as warnings, never propagated.
    """

    def __init__(self, inner_span) -> None:
        """Wrap an opentelemetry Span.

        Args:
            inner_span: The underlying OTel SDK span.
        """
        self._span = inner_span

    def set_attribute(self, key: str, value: object) -> None:
        """Set an attribute on the underlying OTel span. Fail-open."""
        try:
            self._span.set_attribute(key, _coerce_attribute_value(value))
        except Exception as exc:
            logger.warning("OTelSpan.set_attribute failed: %s", exc)

    def end(self, status: str = "ok", error: Optional[Exception] = None) -> None:
        """Finalize this span, recording exception info if present. Fail-open."""
        try:
            if error is not None:
                try:
                    self._span.record_exception(error)
                except Exception as exc:
                    logger.warning("OTelSpan.record_exception failed: %s", exc)
                self._span.set_status(Status(StatusCode.ERROR, str(error)))
            elif status == "error":
                self._span.set_status(Status(StatusCode.ERROR))
            else:
                self._span.set_status(Status(StatusCode.OK))
            self._span.end()
        except Exception as exc:
            logger.warning("OTelSpan.end failed: %s", exc)


class OTelGeneration(Generation):
    """OpenTelemetry-backed generation wrapper.

    Wraps an OTel span carrying gen_ai.* semantic-convention attributes.
    SDK calls are fail-open.
    """

    def __init__(self, inner_span) -> None:
        """Wrap an opentelemetry Span representing an LLM generation."""
        self._span = inner_span

    def set_output(self, output: str) -> None:
        """Record the LLM completion text on ``gen_ai.completion``. Fail-open."""
        try:
            self._span.set_attribute("gen_ai.completion", _truncate_text(output))
        except Exception as exc:
            logger.warning("OTelGeneration.set_output failed: %s", exc)

    def set_token_counts(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record input/output token counts under gen_ai.usage.*. Fail-open."""
        try:
            self._span.set_attribute("gen_ai.usage.input_tokens", int(prompt_tokens))
            self._span.set_attribute("gen_ai.usage.output_tokens", int(completion_tokens))
        except Exception as exc:
            logger.warning("OTelGeneration.set_token_counts failed: %s", exc)

    def end(self, status: str = "ok", error: Optional[Exception] = None) -> None:
        """Finalize this generation span. Fail-open."""
        try:
            if error is not None:
                try:
                    self._span.record_exception(error)
                except Exception as exc:
                    logger.warning("OTelGeneration.record_exception failed: %s", exc)
                self._span.set_status(Status(StatusCode.ERROR, str(error)))
            elif status == "error":
                self._span.set_status(Status(StatusCode.ERROR))
            else:
                self._span.set_status(Status(StatusCode.OK))
            self._span.end()
        except Exception as exc:
            logger.warning("OTelGeneration.end failed: %s", exc)


class OTelTrace(Trace):
    """OpenTelemetry-backed trace root.

    OTel has no first-class "trace" object — a trace is the set of spans
    sharing a trace_id. This wrapper owns the root span; child spans and
    generations are started in the OTel context of that root span so they
    inherit the trace_id automatically.
    """

    def __init__(self, tracer, root_span) -> None:
        """Construct a trace wrapper around a root span.

        Args:
            tracer: The :class:`opentelemetry.trace.Tracer` used to start children.
            root_span: The root :class:`opentelemetry.trace.Span` for this trace.
        """
        self._tracer = tracer
        self._span = root_span

    def span(self, name: str, attributes: Optional[dict] = None) -> Span:
        """Start a child span parented under the trace root. Fail-open."""
        try:
            ctx = otel_trace.set_span_in_context(self._span)
            child = self._tracer.start_span(name, context=ctx)
            if attributes:
                for key, value in attributes.items():
                    try:
                        child.set_attribute(key, _coerce_attribute_value(value))
                    except Exception as exc:
                        logger.warning("OTelTrace.span set_attribute failed: %s", exc)
            return OTelSpan(child)
        except Exception as exc:
            logger.warning("OTelTrace.span failed: %s", exc)
            from src.platform.observability.noop.backend import NoopSpan
            return NoopSpan()

    def generation(
        self,
        name: str,
        model: str,
        input: str,
        metadata: Optional[dict] = None,
    ) -> Generation:
        """Start a child generation under this trace's root span. Fail-open."""
        try:
            ctx = otel_trace.set_span_in_context(self._span)
            child = self._tracer.start_span(name, context=ctx)
            _apply_gen_ai_attributes(child, model=model, input=input, metadata=metadata)
            return OTelGeneration(child)
        except Exception as exc:
            logger.warning("OTelTrace.generation failed: %s", exc)
            from src.platform.observability.noop.backend import NoopGeneration
            return NoopGeneration()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """End the root span on context exit, recording any exception. Fail-open."""
        try:
            if exc_val is not None:
                try:
                    self._span.record_exception(exc_val)
                except Exception as exc:
                    logger.warning("OTelTrace.record_exception failed: %s", exc)
                self._span.set_status(Status(StatusCode.ERROR, str(exc_val)))
            else:
                self._span.set_status(Status(StatusCode.OK))
            self._span.end()
        except Exception as exc:
            logger.warning("OTelTrace.__exit__ failed: %s", exc)
        return False


def _apply_gen_ai_attributes(
    span,
    *,
    model: str,
    input: str,
    metadata: Optional[dict],
) -> None:
    """Apply gen_ai.* semantic-convention attributes to a generation span.

    Sets:
        - ``gen_ai.system``: from ``metadata['gen_ai.system']`` or
          ``metadata['system']``; falls back to ``"unknown"``.
        - ``gen_ai.request.model``: from the ``model`` argument.
        - ``gen_ai.prompt``: from the ``input`` argument, truncated.

    Any other metadata keys are emitted verbatim as span attributes (coerced
    through :func:`_coerce_attribute_value`).
    """
    meta = metadata or {}
    system = meta.get("gen_ai.system") or meta.get("system") or "unknown"
    try:
        span.set_attribute("gen_ai.system", str(system))
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("gen_ai.system set failed: %s", exc)
    try:
        span.set_attribute("gen_ai.request.model", str(model))
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("gen_ai.request.model set failed: %s", exc)
    try:
        span.set_attribute("gen_ai.prompt", _truncate_text(input or ""))
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("gen_ai.prompt set failed: %s", exc)
    for key, value in meta.items():
        if key in ("gen_ai.system", "system"):
            continue
        try:
            span.set_attribute(key, _coerce_attribute_value(value))
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("metadata attribute set failed (%s): %s", key, exc)


class OTelBackend(ObservabilityBackend):
    """OpenTelemetry-native observability backend.

    On construction, configures the global :class:`TracerProvider` once
    (idempotently). If the env var ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set,
    installs an OTLP/HTTP :class:`BatchSpanProcessor` exporting to that
    endpoint; otherwise registers no exporter (spans are dropped). The
    factory then acquires a tracer named ``"ragweave"`` and uses it for all
    span/trace/generation construction.

    The check for "already configured" uses :func:`trace.get_tracer_provider`
    — if it does not return an instance of the SDK's ``TracerProvider``
    (i.e., the default ``ProxyTracerProvider`` is still in place, or a test
    has injected its own provider), we install ours. This makes the backend
    safe to instantiate multiple times in the same process without
    accumulating duplicate exporters.
    """

    def __init__(self) -> None:
        """Configure the global TracerProvider (idempotently) and acquire a tracer."""
        provider = otel_trace.get_tracer_provider()
        if not isinstance(provider, TracerProvider):
            # Default ProxyTracerProvider — replace with a real SDK provider.
            provider = TracerProvider()
            otel_trace.set_tracer_provider(provider)
            self._owns_provider = True
        else:
            self._owns_provider = False

        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if endpoint and self._owns_provider:
            try:
                # Derive Basic-auth header from LANGFUSE_PUBLIC_KEY/SECRET_KEY
                # if the operator hasn't supplied OTEL_EXPORTER_OTLP_HEADERS
                # explicitly. The exporter reads env at construction time, so
                # this must happen before OTLPSpanExporter() is built below.
                from src.platform.observability.otel.auth import (
                    ensure_otlp_headers_from_env,
                )
                ensure_otlp_headers_from_env()
                # Local import keeps the optional exporter dep isolated to
                # the code path that actually needs it.
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )
                # IMPORTANT: do NOT pass endpoint= as a kwarg. When the
                # constructor receives an explicit endpoint, the exporter
                # uses it verbatim. The env-var path (which we want) calls
                # _append_trace_path() so OTEL_EXPORTER_OTLP_ENDPOINT
                # (the non-signal-specific base URL, e.g.
                # http://localhost:3000/api/public/otel) gets the standard
                # /v1/traces suffix appended automatically. Forwarding the
                # kwarg here would silently short-circuit that and result
                # in 404s against Langfuse's OTLP ingest.
                exporter = OTLPSpanExporter()
                provider.add_span_processor(BatchSpanProcessor(exporter))
            except Exception as exc:
                logger.warning("OTLP exporter setup failed: %s", exc)

        self._tracer = otel_trace.get_tracer("ragweave")

    def span(
        self,
        name: str,
        attributes: Optional[dict] = None,
        parent: Optional[Span] = None,
    ) -> Span:
        """Start a span, optionally parented under an OTelTrace or OTelSpan. Fail-open."""
        try:
            context = None
            parent_span = None
            if isinstance(parent, OTelTrace):
                parent_span = parent._span
            elif isinstance(parent, OTelSpan):
                parent_span = parent._span
            if parent_span is not None:
                context = otel_trace.set_span_in_context(parent_span)
            inner = self._tracer.start_span(name, context=context)
            if attributes:
                for key, value in attributes.items():
                    try:
                        inner.set_attribute(key, _coerce_attribute_value(value))
                    except Exception as exc:
                        logger.warning("OTelBackend.span set_attribute failed: %s", exc)
            return OTelSpan(inner)
        except Exception as exc:
            logger.warning("OTelBackend.span failed: %s", exc)
            from src.platform.observability.noop.backend import NoopSpan
            return NoopSpan()

    def trace(self, name: str, metadata: Optional[dict] = None) -> Trace:
        """Start a new root span and wrap it as an OTelTrace. Fail-open."""
        try:
            root = self._tracer.start_span(name)
            if metadata:
                for key, value in metadata.items():
                    try:
                        root.set_attribute(key, _coerce_attribute_value(value))
                    except Exception as exc:
                        logger.warning("OTelBackend.trace set_attribute failed: %s", exc)
            return OTelTrace(self._tracer, root)
        except Exception as exc:
            logger.warning("OTelBackend.trace failed: %s", exc)
            from src.platform.observability.noop.backend import NoopTrace
            return NoopTrace()

    def generation(
        self,
        name: str,
        model: str,
        input: str,
        metadata: Optional[dict] = None,
    ) -> Generation:
        """Start a standalone generation span with gen_ai.* attributes. Fail-open."""
        try:
            inner = self._tracer.start_span(name)
            _apply_gen_ai_attributes(inner, model=model, input=input, metadata=metadata)
            return OTelGeneration(inner)
        except Exception as exc:
            logger.warning("OTelBackend.generation failed: %s", exc)
            from src.platform.observability.noop.backend import NoopGeneration
            return NoopGeneration()

    def flush(self) -> None:
        """Force-flush all span processors on the current SDK TracerProvider.

        No-op when the active provider is the default ProxyTracerProvider
        (which has no processors). Propagates exceptions from the SDK.
        """
        provider = otel_trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            provider.force_flush()

    def shutdown(self) -> None:
        """Shut down the SDK TracerProvider (drains and closes processors).

        No-op when the active provider is not an SDK TracerProvider.
        Propagates exceptions from the SDK.
        """
        provider = otel_trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            provider.shutdown()
