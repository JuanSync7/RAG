"""Tests for src.ingest.embedding.common.dedup_utils helper functions.

Covers exception paths, edge cases, and the build_fuzzy_fingerprint facade.
All mock tests are prefixed with test_mock_.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.ingest.embedding.common.dedup_utils import (
    normalise_chunk_text,
    compute_content_hash,
    find_chunk_by_content_hash,
    append_source_document,
    remove_source_document_refs,
    build_fuzzy_fingerprint,
)


# ---------------------------------------------------------------------------
# Tests: normalise_chunk_text and compute_content_hash (deterministic)
# ---------------------------------------------------------------------------


class TestNormaliseAndHash:
    def test_normalise_collapses_whitespace(self):
        assert normalise_chunk_text("  Hello   World\n") == "Hello World"

    def test_normalise_idempotent(self):
        text = "Hello World"
        assert normalise_chunk_text(text) == text

    def test_compute_hash_deterministic(self):
        h1 = compute_content_hash("  Hello   World\n")
        h2 = compute_content_hash("Hello World")
        assert h1 == h2

    def test_compute_hash_case_sensitive(self):
        assert compute_content_hash("Hello") != compute_content_hash("hello")

    def test_compute_hash_is_64_chars(self):
        h = compute_content_hash("test content")
        assert len(h) == 64


# ---------------------------------------------------------------------------
# Tests: find_chunk_by_content_hash — exception path
# ---------------------------------------------------------------------------


class TestFindChunkByContentHash:
    def test_mock_find_chunk_exception(self, monkeypatch):
        """find_chunk_by_content_hash should return None on exception."""
        client = MagicMock()
        # Simulate an exception when accessing collections
        client.collections.get.side_effect = RuntimeError("connection error")

        result = find_chunk_by_content_hash(client, "abc123")
        assert result is None

    def test_mock_find_chunk_no_match_returns_none(self, monkeypatch):
        """find_chunk_by_content_hash should return None when no objects found."""
        # Patch weaviate Filter import
        with patch.dict("sys.modules", {"weaviate.classes.query": MagicMock()}):
            client = MagicMock()
            collection = MagicMock()
            client.collections.get.return_value = collection
            collection.query.fetch_objects.return_value = MagicMock(objects=[])

            result = find_chunk_by_content_hash(client, "deadbeef")
            assert result is None

    def test_mock_find_chunk_success(self, monkeypatch):
        """find_chunk_by_content_hash should return dict on match."""
        with patch.dict("sys.modules", {"weaviate.classes.query": MagicMock()}):
            fake_obj = MagicMock()
            fake_obj.uuid = "uuid-1234"
            fake_obj.properties = {
                "source_documents": ["doc_a"],
                "text": "hello world",
                "content_hash": "abc",
            }
            client = MagicMock()
            collection = MagicMock()
            client.collections.get.return_value = collection
            collection.query.fetch_objects.return_value = MagicMock(objects=[fake_obj])

            result = find_chunk_by_content_hash(client, "abc")
            assert result is not None
            assert result["uuid"] == "uuid-1234"
            assert result["source_documents"] == ["doc_a"]


# ---------------------------------------------------------------------------
# Tests: append_source_document — edge cases
# ---------------------------------------------------------------------------


class TestAppendSourceDocument:
    def test_mock_append_source_already_present(self):
        """append_source_document returns True without mutation when source already present."""
        client = MagicMock()
        collection = MagicMock()
        client.collections.get.return_value = collection
        obj = MagicMock()
        obj.properties = {"source_documents": ["existing_key"]}
        collection.query.fetch_object_by_id.return_value = obj

        result = append_source_document(client, "uuid-1", "existing_key")
        assert result is True
        # data.update should NOT have been called
        collection.data.update.assert_not_called()

    def test_mock_append_source_new_key(self):
        """append_source_document should add new source_key and call update."""
        client = MagicMock()
        collection = MagicMock()
        client.collections.get.return_value = collection
        obj = MagicMock()
        obj.properties = {"source_documents": ["existing_key"]}
        collection.query.fetch_object_by_id.return_value = obj

        result = append_source_document(client, "uuid-1", "new_key")
        assert result is True
        collection.data.update.assert_called_once()

    def test_mock_append_source_exception(self):
        """append_source_document returns False on exception."""
        client = MagicMock()
        client.collections.get.side_effect = RuntimeError("db error")

        result = append_source_document(client, "uuid-1", "new_key")
        assert result is False


# ---------------------------------------------------------------------------
# Tests: remove_source_document_refs — exception and deletion paths
# ---------------------------------------------------------------------------


class TestRemoveSourceDocumentRefs:
    def test_mock_remove_source_refs_exception(self, caplog):
        """remove_source_document_refs should log error on exception without raising."""
        client = MagicMock()
        client.collections.get.side_effect = RuntimeError("unavailable")

        # Should not raise
        with caplog.at_level(logging.ERROR):
            remove_source_document_refs(client, "doc_a")

        assert any("doc_a" in r.message or "Failed" in r.message for r in caplog.records)

    def test_mock_remove_source_refs_deletes_empty_sources(self):
        """remove_source_document_refs should delete chunks when source_docs becomes empty."""
        with patch.dict("sys.modules", {"weaviate.classes.query": MagicMock()}):
            client = MagicMock()
            collection = MagicMock()
            client.collections.get.return_value = collection

            obj = MagicMock()
            obj.uuid = "uuid-del"
            obj.properties = {"source_documents": ["doc_a"]}  # only source_a, will become empty

            collection.query.fetch_objects.return_value = MagicMock(objects=[obj])

            remove_source_document_refs(client, "doc_a")

            # Should call delete_by_id since list becomes empty
            collection.data.delete_by_id.assert_called_once_with("uuid-del")

    def test_mock_remove_source_refs_updates_multi_source(self):
        """remove_source_document_refs should update when other sources remain."""
        with patch.dict("sys.modules", {"weaviate.classes.query": MagicMock()}):
            client = MagicMock()
            collection = MagicMock()
            client.collections.get.return_value = collection

            obj = MagicMock()
            obj.uuid = "uuid-update"
            obj.properties = {"source_documents": ["doc_a", "doc_b"]}

            collection.query.fetch_objects.return_value = MagicMock(objects=[obj])

            remove_source_document_refs(client, "doc_a")

            # Should update with ["doc_b"] remaining
            collection.data.update.assert_called_once()
            call_kwargs = collection.data.update.call_args[1]
            assert call_kwargs["properties"]["source_documents"] == ["doc_b"]


# ---------------------------------------------------------------------------
# Tests: build_fuzzy_fingerprint — minhash facade
# ---------------------------------------------------------------------------


class TestBuildFuzzyFingerprint:
    def test_mock_build_fuzzy_fingerprint_returns_hex(self):
        """build_fuzzy_fingerprint should return a non-empty hex string."""
        result = build_fuzzy_fingerprint("Hello world this is a test for shingles")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_mock_build_fuzzy_fingerprint_with_config(self):
        """build_fuzzy_fingerprint should use config.fuzzy_shingle_size and num_hashes."""
        config = MagicMock()
        config.fuzzy_shingle_size = 2
        config.fuzzy_num_hashes = 64

        # compute_fuzzy_fingerprint is imported lazily; patch at source module
        import src.ingest.embedding.support.minhash_engine as _me_mod
        with patch.object(_me_mod, "compute_fuzzy_fingerprint", return_value="abcdef") as mock_fp:
            result = build_fuzzy_fingerprint("some text", config=config)

        assert result == "abcdef"
        mock_fp.assert_called_once_with("some text", shingle_size=2, num_hashes=64)

    def test_mock_build_fuzzy_fingerprint_no_config_uses_defaults(self):
        """build_fuzzy_fingerprint without config should use shingle_size=3, num_hashes=128."""
        import src.ingest.embedding.support.minhash_engine as _me_mod
        with patch.object(_me_mod, "compute_fuzzy_fingerprint", return_value="deadbeef") as mock_fp:
            result = build_fuzzy_fingerprint("some text", config=None)

        assert result == "deadbeef"
        mock_fp.assert_called_once_with("some text", shingle_size=3, num_hashes=128)
