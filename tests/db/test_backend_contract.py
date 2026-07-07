"""Contract + delegation tests for the document store backend layer.

This suite isolates the *backend abstraction* (``src.db.backend.DocumentBackend``
and ``src.db.minio.backend.MinioBackend``) from the underlying MinIO store
functions, which are exercised directly in ``test_minio_store.py``. The store
functions are imported into ``backend.py`` under ``_mn_*`` aliases; every test
that probes delegation monkeypatches those aliases on the backend module with
``MagicMock`` spies. This proves the backend's own logic — default bucket
resolution, argument forwarding, ``StoredDocument`` mapping, and the
``get_ephemeral_client`` context manager — without touching real MinIO,
network, or store internals.

The ``client`` handle is an opaque sentinel (a bare ``object()``); the backend
never inspects it, it only threads it through to the store layer.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import src.db.minio.backend as mb
from src.db.backend import DocumentBackend
from src.db.common import StoredDocument
from config.settings import MINIO_BUCKET


# --------------------------------------------------------------------------- #
# ABC contract
# --------------------------------------------------------------------------- #
class TestAbcContract:
    def test_document_backend_cannot_be_instantiated(self):
        """DocumentBackend is abstract — direct instantiation raises TypeError."""
        with pytest.raises(TypeError):
            DocumentBackend()

    def test_minio_backend_instantiates(self):
        """MinioBackend is concrete and constructs without arguments."""
        assert isinstance(mb.MinioBackend(), DocumentBackend)

    def test_minio_backend_overrides_all_abstractmethods(self):
        """MinioBackend implements every abstractmethod (no abstracts remain)."""
        assert mb.MinioBackend.__abstractmethods__ == frozenset()

    def test_document_backend_declares_expected_abstractmethods(self):
        """The ABC pins the full set of required backend operations."""
        assert DocumentBackend.__abstractmethods__ == frozenset({
            "create_persistent_client",
            "get_ephemeral_client",
            "ensure_bucket",
            "put_document",
            "get_document",
            "delete_document",
            "document_exists",
            "get_document_url",
            "list_documents",
        })


# --------------------------------------------------------------------------- #
# _bucket resolution (load-bearing)
# --------------------------------------------------------------------------- #
class TestBucketResolution:
    def test_none_resolves_to_default(self):
        """_bucket(None) falls back to the configured MINIO_BUCKET."""
        assert mb.MinioBackend()._bucket(None) == MINIO_BUCKET

    def test_explicit_bucket_preserved(self):
        """_bucket('custom') returns the caller-supplied bucket unchanged."""
        assert mb.MinioBackend()._bucket("custom") == "custom"

    def test_threaded_none_resolves_to_default(self, monkeypatch):
        """ensure_bucket(bucket=None) forwards bucket=MINIO_BUCKET to the store fn."""
        spy = MagicMock()
        monkeypatch.setattr(mb, "_mn_ensure_bucket", spy)
        client = object()

        mb.MinioBackend().ensure_bucket(client, bucket=None)

        spy.assert_called_once_with(client, bucket=MINIO_BUCKET)

    def test_threaded_explicit_bucket_forwarded(self, monkeypatch):
        """ensure_bucket(bucket='custom') forwards bucket='custom' verbatim."""
        spy = MagicMock()
        monkeypatch.setattr(mb, "_mn_ensure_bucket", spy)
        client = object()

        mb.MinioBackend().ensure_bucket(client, bucket="custom")

        spy.assert_called_once_with(client, bucket="custom")


# --------------------------------------------------------------------------- #
# put_document delegation
# --------------------------------------------------------------------------- #
class TestPutDocument:
    def test_forwards_all_args_with_resolved_bucket(self, monkeypatch):
        """put_document forwards (client, id, content, metadata) + resolved bucket."""
        spy = MagicMock()
        monkeypatch.setattr(mb, "_mn_put_document", spy)
        client = object()

        mb.MinioBackend().put_document(
            client, "doc-id", "the-content", {"k": "v"}, bucket=None
        )

        spy.assert_called_once_with(
            client, "doc-id", "the-content", {"k": "v"}, bucket=MINIO_BUCKET
        )

    def test_explicit_bucket_forwarded(self, monkeypatch):
        """An explicit bucket is threaded into the store call unchanged."""
        spy = MagicMock()
        monkeypatch.setattr(mb, "_mn_put_document", spy)
        client = object()

        mb.MinioBackend().put_document(
            client, "doc-id", "the-content", {"k": "v"}, bucket="custom"
        )

        spy.assert_called_once_with(
            client, "doc-id", "the-content", {"k": "v"}, bucket="custom"
        )


# --------------------------------------------------------------------------- #
# get_document mapping
# --------------------------------------------------------------------------- #
class TestGetDocument:
    def test_maps_raw_dict_to_stored_document(self, monkeypatch):
        """A raw dict is mapped field-for-field onto StoredDocument."""
        spy = MagicMock(return_value={
            "document_id": "d",
            "content": "c",
            "metadata": {"k": "v"},
        })
        monkeypatch.setattr(mb, "_mn_get_document", spy)
        client = object()

        result = mb.MinioBackend().get_document(client, "d", bucket=None)

        assert isinstance(result, StoredDocument)
        assert result.document_id == "d"
        assert result.content == "c"
        assert result.metadata == {"k": "v"}

    def test_resolves_default_bucket(self, monkeypatch):
        """get_document threads the resolved default bucket to the store fn."""
        spy = MagicMock(return_value=None)
        monkeypatch.setattr(mb, "_mn_get_document", spy)
        client = object()

        mb.MinioBackend().get_document(client, "d", bucket=None)

        spy.assert_called_once_with(client, "d", bucket=MINIO_BUCKET)

    def test_none_passthrough(self, monkeypatch):
        """A None from the store fn (not found) is passed straight through."""
        spy = MagicMock(return_value=None)
        monkeypatch.setattr(mb, "_mn_get_document", spy)
        client = object()

        assert mb.MinioBackend().get_document(client, "missing", bucket=None) is None


# --------------------------------------------------------------------------- #
# delete_document
# --------------------------------------------------------------------------- #
class TestDeleteDocument:
    def test_returns_store_bool_and_resolves_bucket(self, monkeypatch):
        """delete_document returns the store fn's bool and resolves the bucket."""
        spy = MagicMock(return_value=True)
        monkeypatch.setattr(mb, "_mn_delete_document", spy)
        client = object()

        result = mb.MinioBackend().delete_document(client, "d", bucket=None)

        assert result is True
        spy.assert_called_once_with(client, "d", bucket=MINIO_BUCKET)

    def test_false_passthrough(self, monkeypatch):
        """A False return (document did not exist) is propagated."""
        spy = MagicMock(return_value=False)
        monkeypatch.setattr(mb, "_mn_delete_document", spy)

        assert mb.MinioBackend().delete_document(object(), "d", bucket="custom") is False


# --------------------------------------------------------------------------- #
# document_exists
# --------------------------------------------------------------------------- #
class TestDocumentExists:
    def test_returns_store_bool_and_resolves_bucket(self, monkeypatch):
        """document_exists returns the store fn's bool and resolves the bucket."""
        spy = MagicMock(return_value=True)
        monkeypatch.setattr(mb, "_mn_document_exists", spy)
        client = object()

        result = mb.MinioBackend().document_exists(client, "d", bucket=None)

        assert result is True
        spy.assert_called_once_with(client, "d", bucket=MINIO_BUCKET)

    def test_false_passthrough(self, monkeypatch):
        """A False return is propagated unchanged."""
        spy = MagicMock(return_value=False)
        monkeypatch.setattr(mb, "_mn_document_exists", spy)

        assert mb.MinioBackend().document_exists(object(), "d", bucket=None) is False


# --------------------------------------------------------------------------- #
# get_document_url
# --------------------------------------------------------------------------- #
class TestGetDocumentUrl:
    def test_threads_expiry_and_resolved_bucket(self, monkeypatch):
        """get_document_url forwards expires_in_seconds + resolved bucket."""
        spy = MagicMock(return_value="https://signed")
        monkeypatch.setattr(mb, "_mn_get_document_url", spy)
        client = object()

        url = mb.MinioBackend().get_document_url(
            client, "d", bucket=None, expires_in_seconds=120
        )

        assert url == "https://signed"
        spy.assert_called_once_with(
            client, "d", bucket=MINIO_BUCKET, expires_in_seconds=120
        )

    def test_default_expiry_threaded(self, monkeypatch):
        """The default expires_in_seconds (3600) is threaded when unspecified."""
        spy = MagicMock(return_value="u")
        monkeypatch.setattr(mb, "_mn_get_document_url", spy)
        client = object()

        mb.MinioBackend().get_document_url(client, "d", bucket="custom")

        spy.assert_called_once_with(
            client, "d", bucket="custom", expires_in_seconds=3600
        )


# --------------------------------------------------------------------------- #
# list_documents
# --------------------------------------------------------------------------- #
class TestListDocuments:
    def test_forwards_paging_args_and_resolved_bucket(self, monkeypatch):
        """list_documents forwards prefix/limit/offset + resolved bucket."""
        sentinel = [{"document_id": "d"}]
        spy = MagicMock(return_value=sentinel)
        monkeypatch.setattr(mb, "_mn_list_documents", spy)
        client = object()

        result = mb.MinioBackend().list_documents(
            client, bucket=None, prefix="pre/", limit=7, offset=3
        )

        assert result is sentinel
        spy.assert_called_once_with(
            client, bucket=MINIO_BUCKET, prefix="pre/", limit=7, offset=3
        )

    def test_explicit_bucket_forwarded(self, monkeypatch):
        """An explicit bucket is threaded into the store call unchanged."""
        spy = MagicMock(return_value=[])
        monkeypatch.setattr(mb, "_mn_list_documents", spy)
        client = object()

        mb.MinioBackend().list_documents(
            client, bucket="custom", prefix="", limit=5, offset=0
        )

        spy.assert_called_once_with(
            client, bucket="custom", prefix="", limit=5, offset=0
        )


# --------------------------------------------------------------------------- #
# Client lifecycle
# --------------------------------------------------------------------------- #
class TestClientLifecycle:
    def test_create_persistent_client_delegates(self, monkeypatch):
        """create_persistent_client returns whatever the store factory returns."""
        spy = MagicMock(return_value="CLIENT")
        monkeypatch.setattr(mb, "_mn_create_client", spy)

        assert mb.MinioBackend().create_persistent_client() == "CLIENT"
        spy.assert_called_once_with()

    def test_get_ephemeral_client_yields_factory_result(self, monkeypatch):
        """The context manager yields the store factory's client sentinel."""
        sentinel = object()
        spy = MagicMock(return_value=sentinel)
        monkeypatch.setattr(mb, "_mn_create_client", spy)

        with mb.MinioBackend().get_ephemeral_client() as client:
            assert client is sentinel
        spy.assert_called_once_with()

    def test_close_client_is_noop(self):
        """close_client (inherited default) returns None and does not raise."""
        assert mb.MinioBackend().close_client(object()) is None
