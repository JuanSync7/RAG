# @summary
# Tests for the knowledge_graph_storage_node embedding pipeline stage.
# Covers: disabled skip, None-builder skip, staged_kg_chunks population,
# and no-direct-add_chunk behavior (Issue #42 staging refactor).
# @end-summary

import pytest
from unittest.mock import MagicMock, call

from src.ingest.common.schemas import ProcessedChunk
from src.ingest.common.types import IngestionConfig, Runtime
from src.ingest.embedding.nodes.knowledge_graph_storage import knowledge_graph_storage_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(text: str = "sample text") -> ProcessedChunk:
    return ProcessedChunk(text=text, metadata={})


def _make_state(
    chunks=None,
    kg_triples=None,
    kg_builder=None,
    enabled=True,
    source_key="doc-1",
    source_name="doc.txt",
):
    config = IngestionConfig(enable_knowledge_graph_storage=enabled)
    runtime = Runtime(
        config=config,
        embedder=MagicMock(),
        weaviate_client=MagicMock(),
        kg_builder=kg_builder,
    )
    return {
        "chunks": chunks or [],
        "kg_triples": kg_triples or [],
        "source_key": source_key,
        "source_name": source_name,
        "errors": [],
        "processing_log": [],
        "runtime": runtime,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_disabled_skips():
    """When KG storage is disabled the node returns without error."""
    mock_builder = MagicMock()
    state = _make_state(
        chunks=[_make_chunk()],
        kg_builder=mock_builder,
        enabled=False,
    )
    result = knowledge_graph_storage_node(state)
    assert result is not None  # node completed cleanly


def test_none_builder_skips():
    """When kg_builder is None the node returns without error."""
    state = _make_state(chunks=[_make_chunk()], kg_builder=None, enabled=True)
    result = knowledge_graph_storage_node(state)
    assert result is not None  # node completed cleanly


def test_disabled_add_chunk_not_called():
    """add_chunk must not be called when the feature is disabled (deferred to commit_node)."""
    mock_builder = MagicMock()
    state = _make_state(
        chunks=[_make_chunk()],
        kg_builder=mock_builder,
        enabled=False,
    )
    knowledge_graph_storage_node(state)
    mock_builder.add_chunk.assert_not_called()


def test_none_builder_add_chunk_not_called():
    """add_chunk cannot be called when kg_builder is None (no AttributeError either)."""
    state = _make_state(chunks=[_make_chunk()], kg_builder=None, enabled=True)
    # If the node tries to call None.add_chunk it would raise AttributeError —
    # completing without error is sufficient.
    knowledge_graph_storage_node(state)  # must not raise


def test_single_chunk_staged_not_written():
    """Single chunk is staged in staged_kg_chunks; add_chunk is NOT called (Issue #42)."""
    mock_builder = MagicMock()
    chunk = _make_chunk("Alice knows Bob.")
    state = _make_state(chunks=[chunk], kg_builder=mock_builder)
    result = knowledge_graph_storage_node(state)
    # No direct write — deferred to commit_node.
    mock_builder.add_chunk.assert_not_called()
    staged = result.get("staged_kg_chunks", [])
    assert len(staged) == 1
    assert staged[0][0] == "Alice knows Bob."


def test_multiple_chunks_all_staged():
    """All chunks are staged; add_chunk is NOT called; staged list length equals chunk count."""
    mock_builder = MagicMock()
    chunks = [_make_chunk(f"text {i}") for i in range(4)]
    state = _make_state(chunks=chunks, kg_builder=mock_builder)
    result = knowledge_graph_storage_node(state)
    mock_builder.add_chunk.assert_not_called()
    assert len(result.get("staged_kg_chunks", [])) == 4


def test_empty_chunks_produces_empty_staged():
    """When the chunk list is empty, staged_kg_chunks is an empty list."""
    mock_builder = MagicMock()
    state = _make_state(chunks=[], kg_builder=mock_builder)
    result = knowledge_graph_storage_node(state)
    mock_builder.add_chunk.assert_not_called()
    assert result.get("staged_kg_chunks", None) == []


def test_staged_kg_chunks_carry_correct_source_name():
    """Each staged tuple is (chunk.text, source_name) for commit_node to replay."""
    mock_builder = MagicMock()
    chunk = _make_chunk("Entity mention text.")
    source_name = "my-doc.txt"
    state = _make_state(
        chunks=[chunk],
        kg_builder=mock_builder,
        source_name=source_name,
    )
    result = knowledge_graph_storage_node(state)
    staged = result.get("staged_kg_chunks", [])
    assert staged == [("Entity mention text.", "my-doc.txt")]


def test_disabled_returns_empty_staged():
    """When disabled, staged_kg_chunks is an empty list."""
    mock_builder = MagicMock()
    state = _make_state(
        chunks=[_make_chunk()],
        kg_builder=mock_builder,
        enabled=False,
    )
    result = knowledge_graph_storage_node(state)
    assert result.get("staged_kg_chunks", None) == []


def test_none_builder_returns_empty_staged():
    """When kg_builder is None, staged_kg_chunks is an empty list."""
    state = _make_state(chunks=[_make_chunk()], kg_builder=None, enabled=True)
    result = knowledge_graph_storage_node(state)
    assert result.get("staged_kg_chunks", None) == []


def test_existing_errors_preserved():
    """Errors that existed before the node runs are not wiped.

    The success path returns only {"staged_kg_chunks": [...], "processing_log": [...]}.
    LangGraph preserves unmodified state keys, so the caller must check
    result.get("errors", state["errors"]).
    """
    mock_builder = MagicMock()
    chunk = _make_chunk()
    state = _make_state(chunks=[chunk], kg_builder=mock_builder)
    state["errors"] = ["pre-existing error"]
    result = knowledge_graph_storage_node(state)
    # Success path: result has no "errors" key → LangGraph keeps original state errors
    effective_errors = result.get("errors", state["errors"])
    assert "pre-existing error" in effective_errors
