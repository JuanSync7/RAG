# @summary
# Shared server helper functions for request id extraction and error/console envelopes.
# Exports: request_id_from_request, error_payload, console_ok, console_err
# Deps: fastapi, server.common.schemas
# @end-summary
"""Server-common utility helpers."""

from __future__ import annotations

import importlib
import logging

from fastapi import Request

from config.settings import RAG_WORKFLOW_DEFAULT_TIMEOUT_MS

_startup_logger = logging.getLogger("rag.startup")
from server.common.schemas import ApiErrorDetail, ApiErrorResponse, ConsoleEnvelope


def request_id_from_request(request: Request) -> str | None:
    """Extract request id from request state, if present."""

    return getattr(request.state, "request_id", None)


def error_payload(
    request: Request,
    *,
    code: str,
    message: str,
    details: dict | None = None,
) -> dict:
    """Build normalized API error payload."""

    return ApiErrorResponse(
        ok=False,
        error=ApiErrorDetail(code=code, message=message, details=details),
        request_id=request_id_from_request(request),
    ).model_dump()


def console_ok(request: Request, data: dict) -> ConsoleEnvelope:
    """Build success envelope payload for console endpoints."""

    return ConsoleEnvelope(ok=True, request_id=request_id_from_request(request), data=data)


def validate_startup_config(
    workflow_timeout_ms: int = RAG_WORKFLOW_DEFAULT_TIMEOUT_MS,
) -> None:
    """Raise ValueError for config values that would cause silent misbehaviour."""
    if workflow_timeout_ms <= 0:
        raise ValueError(
            f"RAG_WORKFLOW_DEFAULT_TIMEOUT_MS must be a positive integer, "
            f"got {workflow_timeout_ms!r}. "
            "Set RAG_WORKFLOW_DEFAULT_TIMEOUT_MS to a value > 0 (milliseconds)."
        )


def _is_importable(module_name: str) -> bool:
    """Check whether *module_name* can be imported without side-effects."""
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def validate_optional_dependencies() -> list[str]:
    """Cross-check enabled feature flags against installed optional deps.

    Logs a WARNING for every mismatch and returns the list of warning
    strings (useful for testing).
    """
    from config.settings import (
        GLINER_ENABLED,
        KG_ENABLED,
        RAG_INGESTION_ENABLE_VISUAL_EMBEDDING,
        RAG_NEMO_PII_GLINER_ENABLED,
    )

    checks: list[tuple[bool, str, str, str]] = [
        (
            GLINER_ENABLED,
            "gliner",
            "GLINER_ENABLED=true",
            "Entity extraction will fall back to regex. Install with: pip install ragweave[gliner]",
        ),
        (
            RAG_NEMO_PII_GLINER_ENABLED,
            "gliner",
            "RAG_NEMO_PII_GLINER_ENABLED=true",
            "PII GLiNER detection will be unavailable. Install with: pip install ragweave[gliner]",
        ),
        (
            RAG_INGESTION_ENABLE_VISUAL_EMBEDDING,
            "colpali_engine",
            "RAG_INGESTION_ENABLE_VISUAL_EMBEDDING=true",
            "Visual embedding will fail at runtime. Install with: pip install ragweave[visual]",
        ),
        (
            KG_ENABLED,
            "igraph",
            "KG_ENABLED=true",
            "Community detection will be disabled. Install with: pip install ragweave[kg]",
        ),
        (
            KG_ENABLED,
            "leidenalg",
            "KG_ENABLED=true",
            "Leiden community detection will be disabled. Install with: pip install ragweave[kg]",
        ),
    ]

    warnings: list[str] = []
    for enabled, package, flag, message in checks:
        if enabled and not _is_importable(package):
            msg = f"{flag} but '{package}' is not installed. {message}"
            _startup_logger.warning(msg)
            warnings.append(msg)
    return warnings


def console_err(
    request: Request,
    *,
    code: str,
    message: str,
    details: dict | None = None,
) -> ConsoleEnvelope:
    """Build error envelope payload for console endpoints."""

    return ConsoleEnvelope(
        ok=False,
        request_id=request_id_from_request(request),
        error=ApiErrorDetail(code=code, message=message, details=details),
    )
