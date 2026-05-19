# @summary
# OTLP exporter auth-header derivation helper.
# Exports: ensure_otlp_headers_from_env
# Deps: base64, os
# @end-summary
"""Derive ``OTEL_EXPORTER_OTLP_HEADERS`` from vendor-specific env vars.

The OpenTelemetry OTLP/HTTP exporter reads ``OTEL_EXPORTER_OTLP_HEADERS``
at exporter construction time (formatted as ``key1=value1,key2=value2``).
Some collectors — notably self-hosted Langfuse — require HTTP Basic auth
with the project public/secret key pair. Rather than asking operators to
manually base64-encode the credentials, we derive the header here.

Policy:
    - If ``OTEL_EXPORTER_OTLP_HEADERS`` is already set, respect it (no-op).
    - Else if both ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are
      set, construct ``Authorization=Basic <base64(public:secret)>`` and
      write it into ``os.environ`` before any exporter is built.
    - Else, leave ``OTEL_EXPORTER_OTLP_HEADERS`` unset. The exporter will
      receive whatever response the collector returns (typically 401 from
      Langfuse) but the backend itself stays healthy — fail-open.
"""
from __future__ import annotations

import base64
import os


def ensure_otlp_headers_from_env(env: dict | None = None) -> None:
    """Populate ``OTEL_EXPORTER_OTLP_HEADERS`` in ``env`` if derivable.

    Args:
        env: Mutable mapping to read/write. Defaults to ``os.environ``.

    Returns:
        None. Side effect: ``env['OTEL_EXPORTER_OTLP_HEADERS']`` may be set.
    """
    target = env if env is not None else os.environ

    existing = target.get("OTEL_EXPORTER_OTLP_HEADERS")
    if existing:
        return

    public_key = target.get("LANGFUSE_PUBLIC_KEY")
    secret_key = target.get("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return

    raw = f"{public_key}:{secret_key}".encode("utf-8")
    encoded = base64.b64encode(raw).decode("ascii")
    target["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {encoded}"


__all__ = ["ensure_otlp_headers_from_env"]
