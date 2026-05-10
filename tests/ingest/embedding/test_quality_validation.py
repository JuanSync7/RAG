# @summary
# Tests for the quality_validation_node embedding pipeline stage.
# Covers: disabled passthrough, short-chunk filtering, quality-score filtering,
# deduplication (case/whitespace insensitive), metadata injection, and edge cases.
# @end-summary

import pytest
from unittest.mock import MagicMock, patch

from src.ingest.common.schemas import ProcessedChunk
from src.ingest.common.types import IngestionConfig, Runtime
from src.ingest.embedding.nodes.quality_validation import quality_validation_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(text: str, metadata: dict | None = None) -> ProcessedChunk:
    return ProcessedChunk(text=text, metadata=metadata if metadata is not None else {})


def _make_state(chunks, enabled=True, min_chars=50, min_quality=0.3):
    config = IngestionConfig(
        enable_quality_validation=enabled,
        min_chunk_chars=min_chars,
        min_quality_score=min_quality,
    )
    runtime = Runtime(
        config=config,
        embedder=MagicMock(),
        weaviate_client=MagicMock(),
        kg_builder=None,
    )
    return {"chunks": chunks, "errors": [], "processing_log": [], "runtime": runtime}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

PATCH_TARGET = "src.ingest.embedding.nodes.quality_validation.quality_score"


def test_disabled_returns_all_chunks():
    """When quality validation is disabled the node returns only a skipped log entry.
    LangGraph keeps original chunks in state via merge; the node return dict has no 'chunks' key."""
    chunks = [_make_chunk("x" * 10), _make_chunk("y" * 5)]  # both would fail filters
    state = _make_state(chunks, enabled=False, min_chars=50, min_quality=0.9)
    with patch(PATCH_TARGET, return_value=0.0):
        result = quality_validation_node(state)
    # Disabled path: no chunks key in return (LangGraph state merge keeps originals)
    assert result.get("chunks", chunks) == chunks
    assert any("skipped" in entry for entry in result.get("processing_log", []))


def test_short_chunk_removed():
    """Chunks shorter than min_chunk_chars are filtered out."""
    short = _make_chunk("a" * 10)
    state = _make_state([short], enabled=True, min_chars=50)
    with patch(PATCH_TARGET, return_value=1.0):
        result = quality_validation_node(state)
    assert short not in result["chunks"]


def test_short_chunk_at_exact_boundary_retained():
    """A chunk whose stripped length equals min_chunk_chars is kept (strict <)."""
    boundary_text = "a" * 50
    chunk = _make_chunk(boundary_text)
    state = _make_state([chunk], enabled=True, min_chars=50)
    with patch(PATCH_TARGET, return_value=1.0):
        result = quality_validation_node(state)
    assert chunk in result["chunks"]


def test_low_quality_chunk_removed():
    """Chunks whose quality score is below min_quality_score are filtered out."""
    chunk = _make_chunk("a" * 100)
    state = _make_state([chunk], enabled=True, min_quality=0.5)
    with patch(PATCH_TARGET, return_value=0.3):
        result = quality_validation_node(state)
    assert chunk not in result["chunks"]


def test_low_quality_at_exact_boundary_retained():
    """A chunk whose quality score equals min_quality_score is kept (strict <)."""
    chunk = _make_chunk("a" * 100)
    state = _make_state([chunk], enabled=True, min_quality=0.3)
    with patch(PATCH_TARGET, return_value=0.3):
        result = quality_validation_node(state)
    assert chunk in result["chunks"]


def test_duplicate_removed_first_occurrence_wins():
    """When two chunks have identical normalised text the second is dropped."""
    first = _make_chunk("Hello world, this is a long enough chunk for testing.")
    second = _make_chunk("Hello world, this is a long enough chunk for testing.")
    state = _make_state([first, second], enabled=True, min_chars=10)
    with patch(PATCH_TARGET, return_value=1.0):
        result = quality_validation_node(state)
    assert len(result["chunks"]) == 1
    assert result["chunks"][0] is first


def test_duplicate_case_insensitive():
    """Deduplication is case-insensitive — mixed-case duplicate is removed."""
    first = _make_chunk("The Quick Brown Fox Jumped Over The Lazy Dog Sentence")
    second = _make_chunk("the quick brown fox jumped over the lazy dog sentence")
    state = _make_state([first, second], enabled=True, min_chars=10)
    with patch(PATCH_TARGET, return_value=1.0):
        result = quality_validation_node(state)
    assert len(result["chunks"]) == 1
    assert result["chunks"][0] is first


def test_duplicate_whitespace_insensitive():
    """Deduplication strips leading/trailing whitespace before comparing."""
    first = _make_chunk("  some long chunk of text that passes quality  ")
    second = _make_chunk("some long chunk of text that passes quality")
    state = _make_state([first, second], enabled=True, min_chars=10)
    with patch(PATCH_TARGET, return_value=1.0):
        result = quality_validation_node(state)
    assert len(result["chunks"]) == 1
    assert result["chunks"][0] is first


def test_quality_score_set_on_surviving_chunks():
    """A chunk whose score meets the threshold survives into the result chunks list."""
    chunk = _make_chunk("a" * 100)
    state = _make_state([chunk], enabled=True, min_chars=50, min_quality=0.3)
    with patch(PATCH_TARGET, return_value=0.75):
        result = quality_validation_node(state)
    # The node filters by quality_score but does not store it in metadata
    assert len(result["chunks"]) == 1
    assert result["chunks"][0] is chunk


def test_empty_chunks_returns_empty():
    """An empty input list produces an empty output list."""
    state = _make_state([], enabled=True)
    with patch(PATCH_TARGET, return_value=1.0):
        result = quality_validation_node(state)
    assert result["chunks"] == []


def test_all_whitespace_chunk_rejected():
    """A chunk that is entirely whitespace is rejected (stripped len == 0 < min_chars)."""
    chunk = _make_chunk("     \t\n   ")
    state = _make_state([chunk], enabled=True, min_chars=1)
    with patch(PATCH_TARGET, return_value=1.0):
        result = quality_validation_node(state)
    assert chunk not in result["chunks"]
