"""Tests for OTLP exporter auth-header derivation.

Covers:
- Explicit OTEL_EXPORTER_OTLP_HEADERS is respected (no overwrite).
- Both LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY → Basic header constructed.
- Missing either key → no header written (fail-open).
- Base64 encoding matches public:secret with the Basic prefix.
"""
from __future__ import annotations

import base64

import pytest

from src.platform.observability.otel.auth import ensure_otlp_headers_from_env


def test_explicit_headers_are_respected():
    """If OTEL_EXPORTER_OTLP_HEADERS is already set, helper is a no-op."""
    env = {
        "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Bearer custom",
        "LANGFUSE_PUBLIC_KEY": "pk",
        "LANGFUSE_SECRET_KEY": "sk",
    }
    ensure_otlp_headers_from_env(env)
    assert env["OTEL_EXPORTER_OTLP_HEADERS"] == "Authorization=Bearer custom"


def test_langfuse_keys_construct_basic_header():
    """Public+secret keys produce a correct base64 Basic header."""
    env = {
        "LANGFUSE_PUBLIC_KEY": "pk-lf-abc",
        "LANGFUSE_SECRET_KEY": "sk-lf-xyz",
    }
    ensure_otlp_headers_from_env(env)
    expected_b64 = base64.b64encode(b"pk-lf-abc:sk-lf-xyz").decode("ascii")
    assert env["OTEL_EXPORTER_OTLP_HEADERS"] == f"Authorization=Basic {expected_b64}"


def test_no_keys_no_headers_written():
    """Missing keys leave OTEL_EXPORTER_OTLP_HEADERS unset."""
    env: dict = {}
    ensure_otlp_headers_from_env(env)
    assert "OTEL_EXPORTER_OTLP_HEADERS" not in env


def test_only_public_key_no_headers_written():
    """Public-key-only is insufficient; nothing written."""
    env = {"LANGFUSE_PUBLIC_KEY": "pk"}
    ensure_otlp_headers_from_env(env)
    assert "OTEL_EXPORTER_OTLP_HEADERS" not in env


def test_only_secret_key_no_headers_written():
    """Secret-key-only is insufficient; nothing written."""
    env = {"LANGFUSE_SECRET_KEY": "sk"}
    ensure_otlp_headers_from_env(env)
    assert "OTEL_EXPORTER_OTLP_HEADERS" not in env


def test_empty_string_keys_treated_as_missing():
    """Empty-string credentials should not produce a header."""
    env = {"LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": ""}
    ensure_otlp_headers_from_env(env)
    assert "OTEL_EXPORTER_OTLP_HEADERS" not in env


def test_defaults_to_os_environ(monkeypatch):
    """When env arg is None, helper reads/writes os.environ."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pkA")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "skB")
    ensure_otlp_headers_from_env()
    import os
    expected_b64 = base64.b64encode(b"pkA:skB").decode("ascii")
    assert os.environ["OTEL_EXPORTER_OTLP_HEADERS"] == f"Authorization=Basic {expected_b64}"
