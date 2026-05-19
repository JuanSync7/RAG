<!-- @summary
Observability subsystem: one OpenTelemetry-native adapter that speaks OTLP/HTTP to any OTel-compatible collector (Langfuse, Phoenix, Honeycomb, etc.), with a NoopBackend fallback. Configured via RAG_OBSERVABILITY_PROVIDER and the standard OTEL_EXPORTER_OTLP_* env vars.
@end-summary -->

# platform/observability

## Overview

This package provides pluggable LLM-pipeline tracing. Call sites use the public API:

```python
from src.platform.observability import get_tracer, observe

tracer = get_tracer()
with tracer.span("retrieval") as span:
    span.set_attribute("query", query_text)
    result = run_retrieval(...)
```

`get_tracer()` returns a process-wide singleton selected once from
`RAG_OBSERVABILITY_PROVIDER`. The actual backend is one of:

- **`OTelBackend`** — OpenTelemetry-native; exports OTLP/HTTP to whatever
  collector `OTEL_EXPORTER_OTLP_ENDPOINT` points at.
- **`NoopBackend`** — every operation is a safe no-op.

## Provider matrix

| `RAG_OBSERVABILITY_PROVIDER` | Backend | Notes |
| --- | --- | --- |
| (unset or empty) | `OTelBackend` | New default. If no collector is listening, spans drop silently — fail-open. |
| `otel` | `OTelBackend` | Canonical value. |
| `langfuse` | `OTelBackend` | Deprecated alias. Routed to `OTelBackend`; emits `DeprecationWarning`. |
| `noop` | `NoopBackend` | Explicit opt-out. |
| anything else | — | `ValueError` at first `get_tracer()` call. |

The `langfuse` Python SDK is no longer a dependency. We send OTLP straight to
Langfuse's `/api/public/otel/v1/traces` endpoint.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `RAG_OBSERVABILITY_PROVIDER` | Backend selector (see matrix above). |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP base URL. The OTel HTTP exporter appends `/v1/traces` itself. |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | Typically `http/protobuf`. |
| `OTEL_EXPORTER_OTLP_HEADERS` | Optional `key=value,key=value` header pairs (e.g. auth). |
| `OTEL_SERVICE_NAME` | Resource attribute attached to all spans. |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Used to auto-derive `Authorization=Basic` when `OTEL_EXPORTER_OTLP_HEADERS` is unset and the target is a Langfuse OTLP ingest. |

When `OTEL_EXPORTER_OTLP_HEADERS` is unset but both Langfuse keys are present,
the backend constructs the Basic-auth header automatically (see
[`otel/auth.py`](otel/auth.py)).

## Example endpoints

| Collector | `OTEL_EXPORTER_OTLP_ENDPOINT` |
| --- | --- |
| Langfuse (self-hosted) | `http://localhost:3000/api/public/otel` |
| Phoenix (self-hosted) | `http://localhost:6006` |
| Braintrust | `https://api.braintrust.dev/otel` |
| Honeycomb | `https://api.honeycomb.io` (set `x-honeycomb-team=<key>` in `OTEL_EXPORTER_OTLP_HEADERS`) |

## Files

| File | Purpose |
| --- | --- |
| `__init__.py` | Public API — `get_tracer`, `observe`, the backend ABCs. |
| `backend.py` | Abstract `ObservabilityBackend` / `Span` / `Trace` / `Generation` contracts. |
| `noop/` | `NoopBackend` and friends — safe no-ops. |
| `otel/backend.py` | `OTelBackend` — OpenTelemetry-native implementation. All `opentelemetry` imports are confined here. |
| `otel/auth.py` | `ensure_otlp_headers_from_env` — derives Basic auth from Langfuse keys. |
| `providers.py` | Deprecated shim — re-exports `get_tracer` from the public API. |
| `schemas.py` | TypedDict span/generation records (export helpers). |
