# @summary
# Tests for the embedding_storage_node embedding pipeline stage.
# Covers: empty-chunk short-circuit, single/multi-chunk staging, update_mode
# flag propagation, collection routing via staged_weaviate_delete_old,
# error isolation, and staged_weaviate_records accuracy.
# Since Issue #42 the node stages records; commit_node performs the actual write.
# @end-summary

import pytest
from unittest.mock import MagicMock, call, patch

from src.ingest.common.schemas import ProcessedChunk
from src.ingest.common.types import IngestionConfig, Runtime
from src.ingest.embedding.nodes.embedding_storage import embedding_storage_node

# Patch targets — functions imported at module level in embedding_storage
_ADD_DOCS = "src.ingest.embedding.nodes.embedding_storage.add_documents"
_DELETE = "src.ingest.embedding.nodes.embedding_storage.delete_by_source_key"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(text: str = "sample chunk text") -> ProcessedChunk:
    return ProcessedChunk(text=text, metadata={})


def _make_state(
    chunks=None,
    update_mode=False,
    target_collection="Docs",
    source_key="doc-001",
    source_name="doc.md",
):
    config = IngestionConfig(
        update_mode=update_mode,
        target_collection=target_collection,
    )
    mock_weaviate = MagicMock()
    mock_embedder = MagicMock()
    # embed_documents returns a list of vectors (one per chunk)
    mock_embedder.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    runtime = Runtime(
        config=config,
        embedder=mock_embedder,
        weaviate_client=mock_weaviate,
    )
    return {
        "chunks": chunks or [],
        "source_key": source_key,
        "source_name": source_name,
        "stored_count": 0,
        "errors": [],
        "processing_log": [],
        "runtime": runtime,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_empty_chunks_returns_zero_stored():
    """With no chunks, stored_count must be 0."""
    state = _make_state(chunks=[])
    result = embedding_storage_node(state)
    assert result["stored_count"] == 0


def test_empty_chunks_no_weaviate_call():
    """With no chunks, add_documents must not be called (nothing to stage)."""
    with patch(_ADD_DOCS) as mock_add:
        state = _make_state(chunks=[])
        embedding_storage_node(state)
    mock_add.assert_not_called()


def test_empty_chunks_returns_empty_staged_records():
    """With no chunks, staged_weaviate_records must be an empty list."""
    state = _make_state(chunks=[])
    result = embedding_storage_node(state)
    assert result.get("staged_weaviate_records", []) == []


def test_single_chunk_staged_not_written():
    """A single chunk is embedded and staged; add_documents is NOT called (Issue #42).

    stored_count stays 0 here; commit_node sets the real count at flush time.
    """
    state = _make_state(chunks=[_make_chunk()])
    state["runtime"].embedder.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    with patch(_ADD_DOCS) as mock_add:
        result = embedding_storage_node(state)
    mock_add.assert_not_called()
    assert result["stored_count"] == 0
    assert len(result.get("staged_weaviate_records", [])) == 1


def test_multiple_chunks_all_staged():
    """All three chunks produce records in staged_weaviate_records; add_documents not called."""
    chunks = [_make_chunk(f"chunk {i}") for i in range(3)]
    state = _make_state(chunks=chunks)
    state["runtime"].embedder.embed_documents.return_value = [
        [0.1] * 3, [0.2] * 3, [0.3] * 3
    ]
    with patch(_ADD_DOCS) as mock_add:
        result = embedding_storage_node(state)
    mock_add.assert_not_called()
    assert len(result.get("staged_weaviate_records", [])) == 3


def test_update_mode_sets_staged_weaviate_delete_old():
    """In update_mode, staged_weaviate_delete_old=True is set so commit_node can delete-before-insert."""
    chunks = [_make_chunk()]
    state = _make_state(chunks=chunks, update_mode=True)
    state["runtime"].embedder.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    with patch(_DELETE) as mock_del, patch(_ADD_DOCS) as mock_add:
        result = embedding_storage_node(state)
    # Staging node must NOT call delete or add directly (Issue #42).
    mock_del.assert_not_called()
    mock_add.assert_not_called()
    assert result.get("staged_weaviate_delete_old") is True


def test_non_update_mode_sets_staged_weaviate_delete_old_false():
    """When update_mode is False, staged_weaviate_delete_old is False."""
    chunks = [_make_chunk()]
    state = _make_state(chunks=chunks, update_mode=False)
    state["runtime"].embedder.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    result = embedding_storage_node(state)
    assert result.get("staged_weaviate_delete_old") is False


def test_records_carry_correct_text_from_embedder():
    """DocumentRecord.text equals the chunk text (or enriched_content when present)."""
    chunk = _make_chunk("hello world")
    state = _make_state(chunks=[chunk])
    state["runtime"].embedder.embed_documents.return_value = [[0.5, 0.6]]
    result = embedding_storage_node(state)
    records = result.get("staged_weaviate_records", [])
    assert records
    assert records[0].text == "hello world"


def test_embed_error_appended_not_raised():
    """An embed_documents error is caught and an error message is appended to state errors."""
    chunk = _make_chunk()
    state = _make_state(chunks=[chunk])
    state["runtime"].embedder.embed_documents.side_effect = RuntimeError("embed boom")
    result = embedding_storage_node(state)
    assert len(result["errors"]) >= 1
    error_str = " ".join(str(e) for e in result["errors"])
    assert "embed" in error_str.lower() or "embedding" in error_str.lower()


def test_all_chunks_fail_staged_records_empty():
    """When embed_documents fails, staged_weaviate_records is empty and error is recorded."""
    chunks = [_make_chunk(f"chunk {i}") for i in range(3)]
    state = _make_state(chunks=chunks)
    state["runtime"].embedder.embed_documents.side_effect = RuntimeError("always fails")
    result = embedding_storage_node(state)
    assert result.get("stored_count", 0) == 0
    assert len(result.get("errors", [])) >= 1
    assert result.get("staged_weaviate_records", []) == []


def test_should_skip_flag_skips_staging():
    """When should_skip=True the node returns stored_count=0 and empty staged records."""
    chunks = [_make_chunk("some text")]
    state = _make_state(chunks=chunks)
    state["should_skip"] = True
    with patch(_ADD_DOCS) as mock_add:
        result = embedding_storage_node(state)
    assert result["stored_count"] == 0
    mock_add.assert_not_called()
    assert result.get("staged_weaviate_records", []) == []


def test_vector_store_raises_propagated_as_error():
    """When add_documents raises from commit_node, staging node is unaffected.

    This test verifies the staging node itself doesn't raise when embedding
    succeeds — the add_documents call lives in commit_node, not here.
    """
    chunks = [_make_chunk()]
    state = _make_state(chunks=chunks)
    state["runtime"].embedder.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    # Patch add_documents to raise — should NOT be called here anyway.
    with patch(_ADD_DOCS, side_effect=RuntimeError("weaviate down")) as mock_add:
        result = embedding_storage_node(state)
    # Since Issue #42, add_documents is never called in this node.
    mock_add.assert_not_called()
    # No errors — staging succeeded cleanly.
    assert result.get("errors", []) == []
    assert len(result.get("staged_weaviate_records", [])) == 1


# ---------------------------------------------------------------------------
# Tests: partial state resilience (optional upstream fields absent/None)
# ---------------------------------------------------------------------------

def test_enriched_content_fallback_to_chunk_text():
    """When enriched_content metadata is absent, node falls back to chunk.text."""
    chunk = _make_chunk("raw chunk text")
    # No enriched_content metadata key set (simulates prior node being skipped).
    assert "enriched_content" not in chunk.metadata
    state = _make_state(chunks=[chunk])
    state["runtime"].embedder.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    result = embedding_storage_node(state)
    # Node must not raise; must stage records.
    records = result.get("staged_weaviate_records", [])
    assert len(records) == 1
    # Verify the text passed to embedder is the chunk's raw text.
    embed_call_args = state["runtime"].embedder.embed_documents.call_args
    texts_passed = embed_call_args[0][0]
    assert texts_passed == ["raw chunk text"]


# ---------------------------------------------------------------------------
# Regression: batch failures must enter state["errors"] as STRINGS, not dicts,
# so EmbeddingResult.errors (list[str]) decodes across the Temporal boundary.
# (Found by the live ingest->serve e2e; see
# tests/ingest/temporal/test_payload_contract.py.)
# ---------------------------------------------------------------------------

def test_batch_failure_errors_are_strings_and_temporal_decodable(monkeypatch):
    """A failing embedder yields STRING errors that round-trip as EmbeddingResult.

    Before the fix, embedding_storage merged dict batch-failure entries into the
    list[str] ``errors`` field; Temporal then failed to decode EmbeddingResult in
    the workflow and the ingest workflow task retried forever. This pins that the
    node now stringifies them (preserving the batch/range/cause info) AND that the
    resulting EmbeddingResult decodes cleanly through Temporal's payload converter.
    """
    # Make retries instant so the test stays fast.
    monkeypatch.setattr(
        "src.ingest.embedding.nodes.embedding_storage.RAG_INGEST_EMBEDDING_BATCH_RETRY_DELAY_S",
        0.0,
    )
    state = _make_state(chunks=[_make_chunk("chunk a"), _make_chunk("chunk b")])
    state["runtime"].embedder.embed_documents.side_effect = RuntimeError("embed backend down")

    result = embedding_storage_node(state)

    errors = result["errors"]
    assert errors, "expected at least one batch-failure error"
    # Load-bearing: every error is a STRING (the contract), not a dict.
    assert all(isinstance(e, str) for e in errors), f"non-str errors: {errors!r}"
    # Structured info is preserved in the string form.
    assert any("batch_embedding_failure" in e for e in errors)
    assert any("embed backend down" in e for e in errors)

    # End-to-end proof: the EmbeddingResult built from these errors decodes
    # across the Temporal activity->workflow boundary (the original failure mode).
    from temporalio.converter import default as _default_converter
    from src.ingest.temporal.activities import EmbeddingResult

    conv = _default_converter().payload_converter
    res = EmbeddingResult(
        errors=errors,
        stored_count=0,
        metadata_summary="",
        metadata_keywords=[],
        processing_log=["embedding_storage:staged"],
    )
    decoded = conv.from_payloads(conv.to_payloads([res]), [EmbeddingResult])[0]
    assert decoded.errors == errors
