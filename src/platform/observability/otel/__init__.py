# @summary
# Public exports for the OpenTelemetry observability backend package.
# Exports: OTelBackend
# Deps: otel.backend
# @end-summary
"""OpenTelemetry observability backend package.

Exports only OTelBackend. All opentelemetry SDK imports are confined to
otel/backend.py — no SDK symbols are re-exported from this package. The
backend speaks OTLP/HTTP, so any OTel-compatible collector (Langfuse v3,
Phoenix, Braintrust, Honeycomb, Datadog, etc.) can be targeted via the
standard OTEL_EXPORTER_OTLP_ENDPOINT environment variable.
"""
from src.platform.observability.otel.backend import OTelBackend

__all__ = ["OTelBackend"]
