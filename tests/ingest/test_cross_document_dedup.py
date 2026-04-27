"""Mock-based tests for src/ingest/embedding/nodes/cross_document_dedup.py.

Covers previously-uncovered paths:
- update_mode path → calls remove_source_document_refs (line 97)
- Tier 2 fuzzy dedup path (lines 147-157)
- Exception / degraded path (lines 163-169)
- _try_fuzzy_dedup() full body (lines 216-254)
- _replace_canonical() (lines 273-275)

All test functions that rely on mocks are named test_mock_*.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from src.ingest.common.schemas import ProcessedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    enable_cross_document_dedup=True,
    update_mode=False,
    dedup_override_sources=None,
    enable_fuzzy_dedup=False,
    fuzzy_similarity_threshold=0.95,
    fuzzy_shingle_size=3,
    fuzzy_num_hashes=128,
):
    cfg = MagicMock()
    cfg.enable_cross_document_dedup = enable_cross_document_dedup
    cfg.update_mode = update_mode
    cfg.dedup_override_sources = dedup_override_sources or []
    cfg.enable_fuzzy_dedup = enable_fuzzy_dedup
    cfg.fuzzy_similarity_threshold = fuzzy_similarity_threshold
    cfg.fuzzy_shingle_size = fuzzy_shingle_size
    cfg.fuzzy_num_hashes = fuzzy_num_hashes
    return cfg


def _make_state(chunks=None, config=None, source_key="test.md"):
    runtime = MagicMock()
    runtime.config = config or _make_config()
    runtime.weaviate_client = MagicMock()
    return {
        "runtime": runtime,
        "source_key": source_key,
        "source_name": source_key,
        "chunks": chunks
        or [ProcessedChunk(text="hello world foo bar", metadata={})],
        "processing_log": [],
    }


# ---------------------------------------------------------------------------
# test_mock_dedup_update_mode_removes_refs
# ---------------------------------------------------------------------------


def test_mock_dedup_update_mode_removes_refs():
    """config with update_mode=True → remove_source_document_refs is called
    with the weaviate client and source_key before per-chunk processing."""
    from src.ingest.embedding.nodes.cross_document_dedup import (
        cross_document_dedup_node,
    )

    cfg = _make_config(update_mode=True)
    state = _make_state(config=cfg)
    client = state["runtime"].weaviate_client

    with patch(
        "src.ingest.embedding.nodes.cross_document_dedup.find_chunk_by_content_hash"
    ) as mock_find, patch(
        "src.ingest.embedding.nodes.cross_document_dedup.remove_source_document_refs"
    ) as mock_remove, patch(
        "src.ingest.embedding.nodes.cross_document_dedup.compute_content_hash"
    ) as mock_hash:
        mock_hash.return_value = "aabbcc"
        mock_find.return_value = None  # chunk is novel

        result = cross_document_dedup_node(state)

    mock_remove.assert_called_once_with(client, "test.md")
    assert result["dedup_stats"]["novel_chunks"] == 1


# ---------------------------------------------------------------------------
# test_mock_dedup_fuzzy_match
# ---------------------------------------------------------------------------


def test_mock_dedup_fuzzy_match():
    """enable_fuzzy_dedup=True with a fuzzy match → chunk is merged and
    fuzzy_matches counter incremented."""
    from src.ingest.embedding.nodes.cross_document_dedup import (
        cross_document_dedup_node,
    )

    cfg = _make_config(enable_fuzzy_dedup=True)
    state = _make_state(config=cfg)

    with patch(
        "src.ingest.embedding.nodes.cross_document_dedup.find_chunk_by_content_hash"
    ) as mock_find_exact, patch(
        "src.ingest.embedding.nodes.cross_document_dedup.compute_content_hash"
    ) as mock_hash, patch(
        "src.ingest.embedding.nodes.cross_document_dedup._try_fuzzy_dedup"
    ) as mock_fuzzy:
        mock_hash.return_value = "aabb"
        mock_find_exact.return_value = None  # no exact match
        mock_fuzzy.return_value = True  # fuzzy match found

        result = cross_document_dedup_node(state)

    assert result["dedup_stats"]["fuzzy_matches"] == 1
    assert result["dedup_stats"]["novel_chunks"] == 0
    # chunk was merged → output list should be empty
    assert len(result["chunks"]) == 0


# ---------------------------------------------------------------------------
# test_mock_dedup_fuzzy_no_match
# ---------------------------------------------------------------------------


def test_mock_dedup_fuzzy_no_match():
    """enable_fuzzy_dedup=True but _try_fuzzy_dedup returns False → chunk is novel."""
    from src.ingest.embedding.nodes.cross_document_dedup import (
        cross_document_dedup_node,
    )

    cfg = _make_config(enable_fuzzy_dedup=True)
    state = _make_state(config=cfg)

    with patch(
        "src.ingest.embedding.nodes.cross_document_dedup.find_chunk_by_content_hash"
    ) as mock_find_exact, patch(
        "src.ingest.embedding.nodes.cross_document_dedup.compute_content_hash"
    ) as mock_hash, patch(
        "src.ingest.embedding.nodes.cross_document_dedup._try_fuzzy_dedup"
    ) as mock_fuzzy:
        mock_hash.return_value = "aabb"
        mock_find_exact.return_value = None
        mock_fuzzy.return_value = False  # no fuzzy match

        result = cross_document_dedup_node(state)

    assert result["dedup_stats"]["fuzzy_matches"] == 0
    assert result["dedup_stats"]["novel_chunks"] == 1
    assert len(result["chunks"]) == 1


# ---------------------------------------------------------------------------
# test_mock_dedup_degraded_on_exception
# ---------------------------------------------------------------------------


def test_mock_dedup_degraded_on_exception():
    """When compute_content_hash raises an unhandled exception, the node
    degrades gracefully: degraded=True and all original chunks pass through."""
    from src.ingest.embedding.nodes.cross_document_dedup import (
        cross_document_dedup_node,
    )

    cfg = _make_config()
    chunks = [
        ProcessedChunk(text="alpha", metadata={}),
        ProcessedChunk(text="beta", metadata={}),
    ]
    state = _make_state(config=cfg, chunks=chunks)

    with patch(
        "src.ingest.embedding.nodes.cross_document_dedup.compute_content_hash"
    ) as mock_hash:
        mock_hash.side_effect = RuntimeError("simulated failure")

        result = cross_document_dedup_node(state)

    assert result["dedup_stats"]["degraded"] is True
    # All original chunks should pass through
    assert result["chunks"] == chunks
    assert len(result["chunks"]) == 2
    assert "cross_document_dedup:degraded" in result["processing_log"]


# ---------------------------------------------------------------------------
# test_mock_dedup_fuzzy_canonical_replaced
# ---------------------------------------------------------------------------


def test_mock_dedup_fuzzy_canonical_replaced():
    """_try_fuzzy_dedup when incoming chunk is longer than canonical → calls
    _replace_canonical and sets canonical_replaced=True in merge event."""
    import sys
    import types

    client = MagicMock()
    cfg = _make_config(enable_fuzzy_dedup=True)

    # Incoming chunk is longer than canonical (text_length=5)
    chunk = ProcessedChunk(text="a much longer text here", metadata={})
    merge_report: list = []

    fuzzy_match = {
        "uuid": "canonical-uuid-123",
        "text_length": 5,  # shorter than chunk.text
        "similarity": 0.97,
    }

    mock_compute_fp = MagicMock(return_value="fingerprint-hex")
    mock_find_fp = MagicMock(return_value=fuzzy_match)

    minhash_mod = types.ModuleType("src.ingest.embedding.support.minhash_engine")
    minhash_mod.compute_fuzzy_fingerprint = mock_compute_fp
    minhash_mod.find_chunk_by_fuzzy_fingerprint = mock_find_fp
    sys.modules["src.ingest.embedding.support.minhash_engine"] = minhash_mod

    try:
        from src.ingest.embedding.nodes.cross_document_dedup import _try_fuzzy_dedup

        with patch(
            "src.ingest.embedding.nodes.cross_document_dedup._replace_canonical"
        ) as mock_replace, patch(
            "src.ingest.embedding.nodes.cross_document_dedup.append_source_document"
        ) as mock_append:
            result = _try_fuzzy_dedup(client, chunk, "hash123", "test.md", cfg, merge_report)
    finally:
        sys.modules.pop("src.ingest.embedding.support.minhash_engine", None)

    assert result is True
    mock_replace.assert_called_once()
    mock_append.assert_called_once_with(client, "canonical-uuid-123", "test.md")
    assert len(merge_report) == 1
    assert merge_report[0]["canonical_replaced"] is True
    assert merge_report[0]["action"] == "replaced"


# ---------------------------------------------------------------------------
# test_mock_replace_canonical
# ---------------------------------------------------------------------------


def test_mock_replace_canonical():
    """_replace_canonical calls update_chunk_content with the correct arguments."""
    from src.ingest.embedding.nodes.cross_document_dedup import _replace_canonical

    client = MagicMock()
    chunk = ProcessedChunk(text="new longer content", metadata={})

    with patch("src.ingest.embedding.nodes.cross_document_dedup.update_chunk_content", create=True) as mock_update, \
         patch.dict("sys.modules", {"src.vector_db": MagicMock(update_chunk_content=mock_update)}):
        # Re-import the function inside the patched context to get the mock injected.
        # Instead, directly patch the import inside the function.
        pass

    # Patch the lazy import inside _replace_canonical via sys.modules
    import sys
    import types

    fake_vector_db = types.ModuleType("src.vector_db")
    mock_update_fn = MagicMock()
    fake_vector_db.update_chunk_content = mock_update_fn
    sys.modules["src.vector_db"] = fake_vector_db

    try:
        _replace_canonical(
            client=client,
            chunk_uuid="uuid-abc",
            chunk=chunk,
            content_hash="hash-xyz",
            fingerprint="fp-hex",
        )
    finally:
        sys.modules.pop("src.vector_db", None)

    mock_update_fn.assert_called_once_with(
        client,
        "uuid-abc",
        text="new longer content",
        content_hash="hash-xyz",
        fuzzy_fingerprint="fp-hex",
    )


# ---------------------------------------------------------------------------
# test_mock_dedup_bypass_when_disabled
# ---------------------------------------------------------------------------


def test_mock_dedup_bypass_when_disabled():
    """When enable_cross_document_dedup=False the node returns all chunks unchanged."""
    from src.ingest.embedding.nodes.cross_document_dedup import (
        cross_document_dedup_node,
    )

    cfg = _make_config(enable_cross_document_dedup=False)
    chunks = [ProcessedChunk(text="abc", metadata={})]
    state = _make_state(config=cfg, chunks=chunks)

    result = cross_document_dedup_node(state)

    assert result["chunks"] is chunks
    assert result["dedup_stats"] == {}
    assert result["dedup_merge_report"] == []
    assert "cross_document_dedup:skipped" in result["processing_log"]


# ---------------------------------------------------------------------------
# test_mock_dedup_override_source
# ---------------------------------------------------------------------------


def test_mock_dedup_override_source():
    """When source_key is in dedup_override_sources, chunks are stored
    independently with an override_skipped merge event and no lookup performed."""
    from src.ingest.embedding.nodes.cross_document_dedup import (
        cross_document_dedup_node,
    )

    cfg = _make_config(dedup_override_sources=["override.md"])
    chunks = [ProcessedChunk(text="test chunk content", metadata={})]
    state = _make_state(config=cfg, chunks=chunks, source_key="override.md")

    with patch(
        "src.ingest.embedding.nodes.cross_document_dedup.compute_content_hash"
    ) as mock_hash, patch(
        "src.ingest.embedding.nodes.cross_document_dedup.find_chunk_by_content_hash"
    ) as mock_find:
        mock_hash.return_value = "override-hash"

        result = cross_document_dedup_node(state)

    # No exact hash lookup should happen for override sources
    mock_find.assert_not_called()

    # Chunk should still be in novel_chunks
    assert len(result["chunks"]) == 1
    assert len(result["dedup_merge_report"]) == 1
    assert result["dedup_merge_report"][0]["action"] == "override_skipped"


# ---------------------------------------------------------------------------
# test_mock_try_fuzzy_dedup_no_match
# ---------------------------------------------------------------------------


def test_mock_try_fuzzy_dedup_no_match():
    """_try_fuzzy_dedup when find_chunk_by_fuzzy_fingerprint returns None →
    returns False and sets source_documents on chunk metadata."""
    from src.ingest.embedding.nodes.cross_document_dedup import _try_fuzzy_dedup

    client = MagicMock()
    cfg = _make_config(enable_fuzzy_dedup=True)
    chunk = ProcessedChunk(text="some text", metadata={})
    merge_report: list = []

    with patch(
        "src.ingest.embedding.support.minhash_engine.compute_fuzzy_fingerprint",
        return_value="fp",
        create=True,
    ), patch(
        "src.ingest.embedding.support.minhash_engine.find_chunk_by_fuzzy_fingerprint",
        return_value=None,
        create=True,
    ):
        import sys
        import types

        minhash_mod = types.ModuleType("src.ingest.embedding.support.minhash_engine")
        minhash_mod.compute_fuzzy_fingerprint = MagicMock(return_value="fp-hash")
        minhash_mod.find_chunk_by_fuzzy_fingerprint = MagicMock(return_value=None)
        sys.modules["src.ingest.embedding.support.minhash_engine"] = minhash_mod

        try:
            result = _try_fuzzy_dedup(
                client, chunk, "hash", "source.md", cfg, merge_report
            )
        finally:
            sys.modules.pop("src.ingest.embedding.support.minhash_engine", None)

    assert result is False
    assert chunk.metadata.get("source_documents") == ["source.md"]
    assert len(merge_report) == 0
