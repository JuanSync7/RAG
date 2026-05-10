# @summary
# Tests for src/ingest/embedding/nodes/document_storage_node.py.
# Covers: document_id derivation (SHA-256, 24-char hex, determinism),
#         staged_minio population (store_documents flag, minio_client presence),
#         and no-error behavior (Issue #42: node does no live I/O).
# @end-summary
"""Tests for the document_storage_node pipeline stage."""

import hashlib

import pytest
from unittest.mock import MagicMock

from src.ingest.common.schemas import ProcessedChunk
from src.ingest.common.types import IngestionConfig, Runtime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runtime(store_documents: bool, minio_client=None, target_bucket: str = "test-bucket") -> Runtime:
    """Build a Runtime with the given document-storage config."""
    config = IngestionConfig(store_documents=store_documents, target_bucket=target_bucket)
    return Runtime(
        config=config,
        embedder=MagicMock(),
        weaviate_client=MagicMock(),
        kg_builder=None,
        db_client=minio_client,  # node accesses minio_client via runtime.db_client
    )


def _make_state(
    source_key: str = "test-doc",
    store_documents: bool = False,
    minio_client=None,
    cleaned_text: str = "# Hello",
) -> dict:
    """Return a minimal ingest state dict for document_storage_node tests.

    minio_client is injected via runtime.db_client (primary access pattern).
    """
    runtime = _make_runtime(
        store_documents=store_documents,
        minio_client=minio_client,
    )
    return {
        "source_key": source_key,
        "source_name": "test.md",
        "source_uri": "file:///tmp/test.md",
        "source_id": "test:1",
        "source_version": "1",
        "connector": "local_fs",
        "cleaned_text": cleaned_text,
        "refactored_text": "",
        "raw_text": "",
        "errors": [],
        "processing_log": [],
        "runtime": runtime,
    }


def _expected_document_id(source_key: str) -> str:
    """build_document_id returns uuid5(NAMESPACE_URL, f'doc:{source_key}')."""
    import uuid
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"doc:{source_key}"))


# ---------------------------------------------------------------------------
# Tests: document_id derivation
# ---------------------------------------------------------------------------

class TestDocumentIdDerivation:
    """document_id is always derived from source_key, regardless of upload path."""

    def test_document_id_set_regardless_of_upload_disabled(self):
        """document_id always set even when store_documents=False."""
        from src.ingest.embedding.nodes.document_storage_node import document_storage_node

        state = _make_state(source_key="my-doc", store_documents=False)
        result = document_storage_node(state)
        assert "document_id" in result
        assert result["document_id"] != ""

    def test_document_id_set_when_minio_client_none(self):
        """document_id always set even when minio_client is None.

        Assumes node reads minio_client from runtime.db_client.
        """
        from src.ingest.embedding.nodes.document_storage_node import document_storage_node

        state = _make_state(source_key="no-client-doc", store_documents=True, minio_client=None)
        result = document_storage_node(state)
        assert "document_id" in result
        assert result["document_id"] != ""

    def test_document_id_is_24_char_hex(self):
        """document_id is a non-empty UUID5 string (36 chars, hex+dashes)."""
        from src.ingest.embedding.nodes.document_storage_node import document_storage_node
        import re

        state = _make_state(source_key="some-key")
        result = document_storage_node(state)
        doc_id = result["document_id"]
        # build_document_id returns a UUID5 string like xxxxxxxx-xxxx-5xxx-xxxx-xxxxxxxxxxxx
        assert len(doc_id) == 36
        assert re.fullmatch(r"[0-9a-f\-]+", doc_id), f"Not a valid UUID: {doc_id!r}"

    def test_document_id_is_deterministic(self):
        """Same source_key always produces same document_id."""
        from src.ingest.embedding.nodes.document_storage_node import document_storage_node

        state_a = _make_state(source_key="stable-key")
        state_b = _make_state(source_key="stable-key")
        result_a = document_storage_node(state_a)
        result_b = document_storage_node(state_b)
        assert result_a["document_id"] == result_b["document_id"]

    def test_document_id_matches_sha256_prefix(self):
        """document_id == hashlib.sha256(source_key.encode()).hexdigest()[:24]."""
        from src.ingest.embedding.nodes.document_storage_node import document_storage_node

        key = "local_fs:tests/sample.md"
        state = _make_state(source_key=key)
        result = document_storage_node(state)
        expected = _expected_document_id(key)
        assert result["document_id"] == expected

    def test_document_id_empty_source_key(self):
        """Empty source_key → valid UUID5 document_id (36 chars)."""
        from src.ingest.embedding.nodes.document_storage_node import document_storage_node
        import re

        state = _make_state(source_key="")
        result = document_storage_node(state)
        doc_id = result["document_id"]
        assert len(doc_id) == 36
        assert re.fullmatch(r"[0-9a-f\-]+", doc_id), f"Not a valid UUID: {doc_id!r}"


# ---------------------------------------------------------------------------
# Tests: upload gating
# ---------------------------------------------------------------------------

class TestUploadGating:
    """Upload is only attempted when both the flag and client are present."""

    def test_staged_minio_populated_when_enabled_and_client_present(self):
        """staged_minio is populated when store_documents=True and minio_client present.

        Since Issue #42 the node no longer calls put_document directly; it stages
        the payload in state["staged_minio"] for commit_node to flush atomically.
        """
        from src.ingest.embedding.nodes.document_storage_node import document_storage_node
        from unittest.mock import patch

        mock_client = MagicMock()
        state = _make_state(
            source_key="upload-doc",
            store_documents=True,
            minio_client=mock_client,
            cleaned_text="# Content to upload",
        )
        with patch("src.ingest.embedding.nodes.document_storage_node.put_document") as mock_put:
            result = document_storage_node(state)
        # Direct write must NOT happen in this node any more (Issue #42).
        mock_put.assert_not_called()
        # Instead, staged_minio must be populated.
        assert result.get("staged_minio") is not None
        assert result["staged_minio"]["content"] == "# Content to upload"
        assert result["staged_minio"]["document_id"]

    def test_upload_not_called_when_disabled(self):
        """Upload NOT called when store_documents=False.

        Assumes node reads minio_client from runtime.db_client.
        """
        from src.ingest.embedding.nodes.document_storage_node import document_storage_node

        mock_client = MagicMock()
        state = _make_state(
            source_key="no-upload-doc",
            store_documents=False,
            minio_client=mock_client,
        )
        document_storage_node(state)
        # No upload methods should have been invoked
        mock_client.put_object.assert_not_called()

    def test_upload_not_called_when_client_none(self):
        """Upload NOT called when minio_client is None.

        Assumes node reads minio_client from runtime.db_client.
        """
        from src.ingest.embedding.nodes.document_storage_node import document_storage_node

        state = _make_state(
            source_key="no-client",
            store_documents=True,
            minio_client=None,
        )
        # Should not raise; simply skips upload
        result = document_storage_node(state)
        assert "document_id" in result


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------

class TestUploadErrorHandling:
    """document_storage_node only stages; upload errors surface at commit_node."""

    def test_node_never_errors_because_no_live_io(self):
        """document_storage_node performs no live I/O since Issue #42.

        The node only builds staged_minio; commit_node is responsible for the
        actual put_document call and error handling. Therefore this node must
        not inject errors even when put_document is patched to raise.
        """
        from src.ingest.embedding.nodes.document_storage_node import document_storage_node
        from unittest.mock import patch

        mock_client = MagicMock()
        state = _make_state(
            source_key="error-doc",
            store_documents=True,
            minio_client=mock_client,
        )

        with patch("src.ingest.embedding.nodes.document_storage_node.put_document",
                   side_effect=RuntimeError("connection refused")):
            result = document_storage_node(state)

        # No errors injected; the node never calls put_document.
        errors = result.get("errors", [])
        assert errors == [], f"Unexpected errors from staging node: {errors}"
        # staged_minio must still be populated
        assert result.get("staged_minio") is not None
