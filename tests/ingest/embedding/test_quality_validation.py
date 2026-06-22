# @summary
# Tests for the quality_validation_node embedding pipeline stage (collapsed to
# dedup + empty-skip). Covers: disabled passthrough, empty/whitespace-body
# removal, exact-duplicate removal (case/whitespace insensitive), same-body-
# different-section retention, and — the key behaviour change — small/short and
# small-table chunks are now KEPT (no size or quality-score gate).
# @end-summary

from unittest.mock import MagicMock

from src.ingest.common.schemas import ProcessedChunk
from src.ingest.common.types import IngestionConfig, Runtime
from src.ingest.embedding.nodes.quality_validation import quality_validation_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(text: str, metadata: dict | None = None) -> ProcessedChunk:
    return ProcessedChunk(text=text, metadata=metadata if metadata is not None else {})


def _make_state(chunks, enabled=True):
    config = IngestionConfig(enable_quality_validation=enabled)
    runtime = Runtime(config=config, embedder=MagicMock(), weaviate_client=MagicMock())
    return {"chunks": chunks, "errors": [], "processing_log": [], "runtime": runtime,
            "source_name": "doc"}


# ---------------------------------------------------------------------------
# Disabled passthrough
# ---------------------------------------------------------------------------

def test_disabled_returns_all_chunks():
    """Disabled → only a skipped log entry; LangGraph keeps originals (no chunks key)."""
    chunks = [_make_chunk("x" * 10), _make_chunk("y" * 5)]
    state = _make_state(chunks, enabled=False)
    result = quality_validation_node(state)
    assert result.get("chunks", chunks) == chunks
    assert any("skipped" in entry for entry in result.get("processing_log", []))


# ---------------------------------------------------------------------------
# THE behaviour change: no size / quality-score gate — small content is KEPT
# ---------------------------------------------------------------------------

def test_tiny_nonempty_chunk_is_kept():
    """A short, non-empty chunk that the OLD 40-char floor would have dropped is
    now retained (lossless: chunking sizes up; this node no longer gates size)."""
    chunk = _make_chunk("a" * 10)
    result = quality_validation_node(_make_state([chunk]))
    assert chunk in result["chunks"]


def test_small_table_block_is_kept():
    """A small table_row block (short body, the kind the old floor/score gate
    silently dropped) survives — 'every row captured' must hold end-to-end."""
    chunk = _make_chunk("| G60 | 1053 | done |", {"chunk_type": "table_row", "heading_path": []})
    result = quality_validation_node(_make_state([chunk]))
    assert chunk in result["chunks"]


def test_low_information_short_chunk_kept():
    """A short low-'quality' chunk (would fail the old 0.45 score gate) is kept."""
    chunk = _make_chunk("ok.")
    result = quality_validation_node(_make_state([chunk]))
    assert chunk in result["chunks"]


# ---------------------------------------------------------------------------
# Empty / whitespace-only body removal (the only size-ish drop that remains)
# ---------------------------------------------------------------------------

def test_all_whitespace_chunk_rejected():
    """A chunk that is entirely whitespace has no content to embed → dropped."""
    chunk = _make_chunk("     \t\n   ")
    result = quality_validation_node(_make_state([chunk]))
    assert chunk not in result["chunks"]


def test_breadcrumb_only_empty_body_rejected():
    """A heading-only chunk (breadcrumb prefix, empty body) is dropped — the body
    after stripping the breadcrumb is empty, so there is nothing to embed."""
    hp = ["Chapter 5", "5.2 Address Channel"]
    chunk = _make_chunk("\n".join(hp) + "\n", {"heading_path": hp})
    result = quality_validation_node(_make_state([chunk]))
    assert chunk not in result["chunks"]


def test_breadcrumb_with_tiny_body_kept():
    """Inverse of the above: a breadcrumb + a TINY but non-empty body is now KEPT
    (the old body-aware floor dropped this 'title-only' case; we no longer do)."""
    hp = ["Chapter 5", "5.2 Address Channel"]
    chunk = _make_chunk("\n".join(hp) + "\nSee 5.3.", {"heading_path": hp})
    result = quality_validation_node(_make_state([chunk]))
    assert chunk in result["chunks"]


# ---------------------------------------------------------------------------
# Exact-duplicate removal (lossless: the twin remains)
# ---------------------------------------------------------------------------

def test_duplicate_removed_first_occurrence_wins():
    first = _make_chunk("Hello world, this is a chunk for testing.")
    second = _make_chunk("Hello world, this is a chunk for testing.")
    result = quality_validation_node(_make_state([first, second]))
    assert len(result["chunks"]) == 1 and result["chunks"][0] is first


def test_duplicate_case_insensitive():
    first = _make_chunk("The Quick Brown Fox Jumped Over The Lazy Dog")
    second = _make_chunk("the quick brown fox jumped over the lazy dog")
    result = quality_validation_node(_make_state([first, second]))
    assert len(result["chunks"]) == 1 and result["chunks"][0] is first


def test_duplicate_whitespace_insensitive():
    first = _make_chunk("  some chunk of text  ")
    second = _make_chunk("some chunk of text")
    result = quality_validation_node(_make_state([first, second]))
    assert len(result["chunks"]) == 1 and result["chunks"][0] is first


def test_same_body_different_section_both_kept():
    """Identical bodies under DIFFERENT headings are kept (dedup is on full text)."""
    body = "Reset value is 0x00 for this register field."
    a = _make_chunk("Ch1\n1.1 A\n" + body, {"heading_path": ["Ch1", "1.1 A"]})
    b = _make_chunk("Ch2\n2.1 B\n" + body, {"heading_path": ["Ch2", "2.1 B"]})
    result = quality_validation_node(_make_state([a, b]))
    assert len(result["chunks"]) == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_chunks_returns_empty():
    result = quality_validation_node(_make_state([]))
    assert result["chunks"] == []


def test_no_heading_path_uses_full_text_as_body():
    """Chunks without heading_path (legacy/table/figure) measure the full text;
    non-empty → kept."""
    chunk = _make_chunk("a" * 60, {})
    result = quality_validation_node(_make_state([chunk]))
    assert chunk in result["chunks"]
