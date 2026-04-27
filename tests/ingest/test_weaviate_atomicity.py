"""Tests for atomicity + idempotency primitives in the Weaviate store.

Covers:
- ``staging_batch_id`` schema property and persistence in ``add_documents``
- ``delete_documents_by_staging_batch`` helper
- ``source_hash`` schema property and persistence
- ``get_source_hash`` lookup helper for idempotent re-ingest

Uses MagicMock against the ``weaviate.WeaviateClient`` surface; no live Weaviate.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_collection_with_props(prop_names: list[str]) -> MagicMock:
    col = MagicMock()
    config = MagicMock()
    config.properties = [MagicMock(name=n) for n in prop_names]
    # MagicMock(name=...) sets the *mock* name, not the attribute we want.
    # Use a real list of objects with .name set explicitly.
    props = []
    for n in prop_names:
        p = MagicMock()
        p.name = n
        props.append(p)
    config.properties = props
    col.config.get.return_value = config
    return col


def _make_client(col: MagicMock) -> MagicMock:
    client = MagicMock()
    client.collections.get.return_value = col
    return client


# ---------------------------------------------------------------------------
# delete_documents_by_staging_batch
# ---------------------------------------------------------------------------

class TestDeleteDocumentsByStagingBatch:
    def test_invokes_delete_many_with_staging_batch_filter(self):
        from src.vector_db.weaviate.store import delete_documents_by_staging_batch

        col = _make_collection_with_props(["staging_batch_id"])
        result = MagicMock()
        result.matches = 7
        col.data.delete_many.return_value = result
        client = _make_client(col)

        deleted = delete_documents_by_staging_batch(client, "abc-123")

        assert deleted == 7
        col.data.delete_many.assert_called_once()
        # The filter should reference staging_batch_id; we don't introspect
        # weaviate's Filter internals, just confirm a where= was passed.
        _, kwargs = col.data.delete_many.call_args
        assert "where" in kwargs

    def test_returns_zero_when_no_matches(self):
        from src.vector_db.weaviate.store import delete_documents_by_staging_batch

        col = _make_collection_with_props(["staging_batch_id"])
        result = MagicMock()
        result.matches = 0
        col.data.delete_many.return_value = result
        client = _make_client(col)

        assert delete_documents_by_staging_batch(client, "missing") == 0

    def test_handles_missing_matches_attr(self):
        from src.vector_db.weaviate.store import delete_documents_by_staging_batch

        col = _make_collection_with_props(["staging_batch_id"])
        result = MagicMock(spec=[])  # no .matches attribute
        col.data.delete_many.return_value = result
        client = _make_client(col)

        assert delete_documents_by_staging_batch(client, "x") == 0


# ---------------------------------------------------------------------------
# add_documents persists staging_batch_id and source_hash
# ---------------------------------------------------------------------------

class TestAddDocumentsPersistsAtomicityFields:
    def _capture_adds(self, prop_names: list[str]):
        col = _make_collection_with_props(prop_names)
        captured: list[dict] = []

        class _Batch:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

            def add_object(self_inner, properties, vector, uuid):  # noqa: A002
                captured.append({"properties": properties, "uuid": uuid})

        col.batch.dynamic.return_value = _Batch()
        return col, captured

    def test_staging_batch_id_written_when_property_in_schema(self):
        from src.vector_db.weaviate.store import add_documents

        col, captured = self._capture_adds(
            ["text", "source", "source_key", "staging_batch_id", "source_hash"]
        )
        client = _make_client(col)

        add_documents(
            client,
            texts=["hello"],
            embeddings=[[0.1, 0.2]],
            metadatas=[{
                "source": "test.md",
                "source_key": "k1",
                "chunk_index": 0,
                "staging_batch_id": "batch-xyz",
                "source_hash": "deadbeef",
            }],
        )

        assert len(captured) == 1
        props = captured[0]["properties"]
        assert props.get("staging_batch_id") == "batch-xyz"
        assert props.get("source_hash") == "deadbeef"

    def test_fields_omitted_when_not_in_schema(self):
        """Legacy collections without the new properties should not error."""
        from src.vector_db.weaviate.store import add_documents

        col, captured = self._capture_adds(["text", "source", "source_key"])
        client = _make_client(col)

        add_documents(
            client,
            texts=["hi"],
            embeddings=[[0.0]],
            metadatas=[{
                "source": "t.md",
                "source_key": "k",
                "chunk_index": 0,
                "staging_batch_id": "should-not-write",
                "source_hash": "abc",
            }],
        )

        props = captured[0]["properties"]
        assert "staging_batch_id" not in props
        assert "source_hash" not in props

    def test_missing_metadata_defaults_to_empty_string(self):
        from src.vector_db.weaviate.store import add_documents

        col, captured = self._capture_adds(
            ["text", "source", "source_key", "staging_batch_id", "source_hash"]
        )
        client = _make_client(col)

        add_documents(
            client,
            texts=["hi"],
            embeddings=[[0.0]],
            metadatas=[{"source": "t.md", "source_key": "k", "chunk_index": 0}],
        )

        props = captured[0]["properties"]
        assert props["staging_batch_id"] == ""
        assert props["source_hash"] == ""


# ---------------------------------------------------------------------------
# get_source_hash
# ---------------------------------------------------------------------------

class TestGetSourceHash:
    def _client_with_query_result(self, objects: list):
        col = MagicMock()
        response = MagicMock()
        response.objects = objects
        col.query.fetch_objects.return_value = response
        client = MagicMock()
        client.collections.get.return_value = col
        client.collections.exists.return_value = True
        return client, col

    def test_returns_none_when_collection_missing(self):
        from src.vector_db.weaviate.store import get_source_hash

        client = MagicMock()
        client.collections.exists.return_value = False
        assert get_source_hash(client, "any-key") is None

    def test_returns_none_when_no_chunks(self):
        from src.vector_db.weaviate.store import get_source_hash

        client, _ = self._client_with_query_result([])
        assert get_source_hash(client, "k1") is None

    def test_returns_stored_hash_when_chunk_exists(self):
        from src.vector_db.weaviate.store import get_source_hash

        obj = MagicMock()
        obj.properties = {"source_hash": "abc123"}
        client, col = self._client_with_query_result([obj])
        assert get_source_hash(client, "k1") == "abc123"
        # Confirms a filtered fetch was made on source_key
        col.query.fetch_objects.assert_called_once()

    def test_returns_none_when_property_missing(self):
        from src.vector_db.weaviate.store import get_source_hash

        obj = MagicMock()
        obj.properties = {}
        client, _ = self._client_with_query_result([obj])
        assert get_source_hash(client, "k1") is None

    def test_returns_none_when_property_empty_string(self):
        from src.vector_db.weaviate.store import get_source_hash

        obj = MagicMock()
        obj.properties = {"source_hash": ""}
        client, _ = self._client_with_query_result([obj])
        assert get_source_hash(client, "k1") is None

    def test_returns_none_on_query_error(self):
        from src.vector_db.weaviate.store import get_source_hash

        col = MagicMock()
        col.query.fetch_objects.side_effect = RuntimeError("weaviate down")
        client = MagicMock()
        client.collections.get.return_value = col
        client.collections.exists.return_value = True

        assert get_source_hash(client, "k1") is None
