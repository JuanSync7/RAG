# @summary
# Tests for the RAG_INGESTION_CHUNKER selection config and its validation —
# guards the native/markdown/legacy chunker switch that controls whether the
# embedding pipeline uses the native HybridChunker (table-aware + contextualize)
# or the legacy markdown chunker.
# Exports: (pytest test functions)
# Deps: pytest, src.ingest.impl, src.ingest.common.types
# @end-summary
"""Unit tests for chunker-backend selection and validation."""

import pytest

from src.ingest.common.types import IngestionConfig
from src.ingest.impl import _check_parser_abstraction_config


@pytest.mark.parametrize("value", ["native", "markdown", "legacy"])
def test_valid_chunker_values_accepted(value):
    cfg = IngestionConfig(chunker=value)
    errors, _warnings = _check_parser_abstraction_config(cfg)
    assert not any("chunker" in e for e in errors), errors


def test_invalid_chunker_rejected():
    cfg = IngestionConfig(chunker="bogus")
    errors, _warnings = _check_parser_abstraction_config(cfg)
    assert any("chunker" in e for e in errors)


def test_default_chunker_is_native():
    # Default flows from RAG_INGESTION_CHUNKER (default "native").
    assert IngestionConfig().chunker == "native"


def test_markdown_emits_override_warning():
    cfg = IngestionConfig(chunker="markdown")
    _errors, warnings_list = _check_parser_abstraction_config(cfg)
    assert any("markdown" in w.lower() for w in warnings_list)


def test_legacy_does_not_emit_markdown_override_warning():
    cfg = IngestionConfig(chunker="legacy")
    _errors, warnings_list = _check_parser_abstraction_config(cfg)
    # legacy is a distinct path; it should not trip the markdown-override notice
    assert not any("markdown-based chunking" in w.lower() for w in warnings_list)
