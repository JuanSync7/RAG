"""Mocked-integration tests for src.db.minio.store.

Every operation in the module takes a ``Minio`` client as its first argument,
so these tests inject a ``MagicMock`` (or a small hand-written fake) in its
place — no real MinIO server or network is ever contacted. The module-level
``tracer`` is replaced with a ``MagicMock`` per-test so that ``span.end(...)``
calls are inert.

S3Error construction note: the installed ``minio`` exposes
``S3Error.__init__(self, response, code, message, resource, request_id,
host_id, bucket_name=None, object_name=None)``. ``exc.code`` is read by the
module, so error instances are built with ``code=`` passed as a keyword to
guarantee the branch under test fires (positional construction would land the
code string in ``response`` instead).
"""

from __future__ import annotations

import io
import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from minio.error import S3Error

from src.db.minio import store


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _s3error(code: str) -> S3Error:
    """Build a real S3Error whose ``.code`` is exactly ``code``."""
    return S3Error(
        response=None,
        code=code,
        message="test-message",
        resource="test-resource",
        request_id="test-request-id",
        host_id="test-host-id",
    )


def _make_response(payload: bytes) -> MagicMock:
    """A fake get_object response: ``.read()`` -> bytes; close/release present."""
    resp = MagicMock()
    resp.read.return_value = payload
    return resp


def _make_list_object(name: str, size: int = 100, last_modified=None) -> MagicMock:
    obj = MagicMock()
    obj.object_name = name
    obj.size = size
    obj.last_modified = last_modified
    return obj


@pytest.fixture(autouse=True)
def _silence_tracer(monkeypatch):
    """Replace the module tracer so span/end calls are no-ops."""
    monkeypatch.setattr(store, "tracer", MagicMock())


# --------------------------------------------------------------------------- #
# build_document_id
# --------------------------------------------------------------------------- #
class TestBuildDocumentId:
    def test_deterministic_same_key_same_id(self):
        """Same source_key yields the identical id on repeated calls."""
        assert store.build_document_id("alpha") == store.build_document_id("alpha")

    def test_different_keys_different_ids(self):
        """Distinct source_keys yield distinct ids."""
        assert store.build_document_id("alpha") != store.build_document_id("beta")

    def test_pinned_expected_value(self):
        """id equals uuid5(NAMESPACE_URL, 'doc:<key>') exactly."""
        expected = str(uuid.uuid5(uuid.NAMESPACE_URL, "doc:foo"))
        assert store.build_document_id("foo") == expected


# --------------------------------------------------------------------------- #
# create_client
# --------------------------------------------------------------------------- #
class TestCreateClient:
    def test_constructs_minio_from_settings(self, monkeypatch):
        """Minio is constructed with endpoint + keys + secure from settings."""
        fake_minio = MagicMock(return_value="CLIENT")
        monkeypatch.setattr(store, "Minio", fake_minio)

        result = store.create_client()

        assert result == "CLIENT"
        args, kwargs = fake_minio.call_args
        assert args[0] == store.MINIO_ENDPOINT
        assert kwargs["access_key"] == store.MINIO_ACCESS_KEY
        assert kwargs["secret_key"] == store.MINIO_SECRET_KEY
        assert kwargs["secure"] == store.MINIO_SECURE


# --------------------------------------------------------------------------- #
# ensure_bucket
# --------------------------------------------------------------------------- #
class TestEnsureBucket:
    def test_creates_bucket_when_absent(self):
        """bucket_exists False -> make_bucket called once with the bucket name."""
        client = MagicMock()
        client.bucket_exists.return_value = False

        store.ensure_bucket(client, "my-bucket")

        client.make_bucket.assert_called_once_with("my-bucket")

    def test_skips_creation_when_present(self):
        """bucket_exists True -> make_bucket NOT called (idempotency)."""
        client = MagicMock()
        client.bucket_exists.return_value = True

        store.ensure_bucket(client, "my-bucket")

        client.make_bucket.assert_not_called()


# --------------------------------------------------------------------------- #
# put_document
# --------------------------------------------------------------------------- #
class TestPutDocument:
    def test_two_put_object_calls(self):
        """Exactly two put_object calls: content + metadata sidecar."""
        client = MagicMock()
        store.put_document(client, "doc1", "hello", {"k": "v"}, bucket="b")
        assert client.put_object.call_count == 2

    def test_content_object_routing_and_payload(self):
        """First call writes <id>.md as utf-8 markdown with matching length."""
        client = MagicMock()
        store.put_document(client, "doc1", "héllo", {"k": "v"}, bucket="b")

        first_args, first_kwargs = client.put_object.call_args_list[0]
        assert first_args[0] == "b"
        assert first_args[1] == "doc1.md"
        body = first_args[2]
        assert isinstance(body, io.BytesIO)
        assert body.getvalue() == "héllo".encode("utf-8")
        assert first_kwargs["length"] == len("héllo".encode("utf-8"))
        assert first_kwargs["content_type"] == "text/markdown; charset=utf-8"

    def test_metadata_object_routing_and_payload(self):
        """Second call writes <id>.meta.json as JSON of the metadata dict."""
        client = MagicMock()
        store.put_document(client, "doc1", "hello", {"k": "v"}, bucket="b")

        second_args, second_kwargs = client.put_object.call_args_list[1]
        assert second_args[1] == "doc1.meta.json"
        assert json.loads(second_args[2].getvalue().decode("utf-8")) == {"k": "v"}
        assert second_kwargs["content_type"] == "application/json; charset=utf-8"

    def test_none_metadata_stores_empty_dict(self):
        """metadata=None serializes to an empty JSON object."""
        client = MagicMock()
        store.put_document(client, "doc1", "hello", None, bucket="b")

        second_args, _ = client.put_object.call_args_list[1]
        assert json.loads(second_args[2].getvalue().decode("utf-8")) == {}


# --------------------------------------------------------------------------- #
# get_document
# --------------------------------------------------------------------------- #
class TestGetDocument:
    def test_happy_path_returns_content_and_metadata(self):
        """Returns dict with decoded content and parsed metadata."""
        client = MagicMock()
        client.get_object.side_effect = [
            _make_response("body text".encode("utf-8")),
            _make_response(json.dumps({"source_key": "s"}).encode("utf-8")),
        ]

        result = store.get_document(client, "doc1", bucket="b")

        assert result == {
            "document_id": "doc1",
            "content": "body text",
            "metadata": {"source_key": "s"},
        }

    def test_content_not_found_returns_none(self):
        """NoSuchKey on the content object -> None."""
        client = MagicMock()
        client.get_object.side_effect = _s3error("NoSuchKey")

        assert store.get_document(client, "doc1", bucket="b") is None

    def test_content_other_s3error_reraises(self):
        """A non-NoSuchKey/NoSuchBucket S3Error propagates."""
        client = MagicMock()
        client.get_object.side_effect = _s3error("AccessDenied")

        with pytest.raises(S3Error):
            store.get_document(client, "doc1", bucket="b")

    def test_metadata_sidecar_missing_tolerated(self):
        """Missing metadata sidecar -> metadata == {} (content still returned)."""
        client = MagicMock()
        client.get_object.side_effect = [
            _make_response("body".encode("utf-8")),
            _s3error("NoSuchKey"),
        ]

        result = store.get_document(client, "doc1", bucket="b")

        assert result["content"] == "body"
        assert result["metadata"] == {}


# --------------------------------------------------------------------------- #
# delete_document
# --------------------------------------------------------------------------- #
class TestDeleteDocument:
    def test_returns_true_when_content_exists(self):
        """stat_object succeeds -> returns True."""
        client = MagicMock()
        client.stat_object.return_value = MagicMock()

        assert store.delete_document(client, "doc1", bucket="b") is True

    def test_returns_false_when_content_absent(self):
        """stat_object NoSuchKey -> returns False."""
        client = MagicMock()
        client.stat_object.side_effect = _s3error("NoSuchKey")

        assert store.delete_document(client, "doc1", bucket="b") is False

    def test_removes_both_objects(self):
        """remove_object called for both <id>.md and <id>.meta.json."""
        client = MagicMock()
        client.stat_object.return_value = MagicMock()

        store.delete_document(client, "doc1", bucket="b")

        removed = [c.args[1] for c in client.remove_object.call_args_list]
        assert removed == ["doc1.md", "doc1.meta.json"]

    def test_remove_nosuchkey_tolerated(self):
        """A NoSuchKey raised by remove_object is swallowed (no raise)."""
        client = MagicMock()
        client.stat_object.return_value = MagicMock()
        client.remove_object.side_effect = _s3error("NoSuchKey")

        # Should not raise.
        assert store.delete_document(client, "doc1", bucket="b") is True


# --------------------------------------------------------------------------- #
# document_exists
# --------------------------------------------------------------------------- #
class TestDocumentExists:
    def test_stat_ok_returns_true(self):
        """stat_object succeeds -> True."""
        client = MagicMock()
        client.stat_object.return_value = MagicMock()
        assert store.document_exists(client, "doc1", bucket="b") is True

    def test_nosuchkey_returns_false(self):
        """stat_object NoSuchKey -> False."""
        client = MagicMock()
        client.stat_object.side_effect = _s3error("NoSuchKey")
        assert store.document_exists(client, "doc1", bucket="b") is False

    def test_other_code_reraises(self):
        """A non-NoSuchKey/NoSuchBucket S3Error propagates."""
        client = MagicMock()
        client.stat_object.side_effect = _s3error("AccessDenied")
        with pytest.raises(S3Error):
            store.document_exists(client, "doc1", bucket="b")


# --------------------------------------------------------------------------- #
# get_document_url
# --------------------------------------------------------------------------- #
class TestGetDocumentUrl:
    def test_presigned_with_md_key_and_exact_expiry(self):
        """presigned_get_object called with <id>.md and timedelta(seconds=120)."""
        client = MagicMock()
        client.presigned_get_object.return_value = "https://signed"

        url = store.get_document_url(client, "doc1", bucket="b", expires_in_seconds=120)

        assert url == "https://signed"
        args, kwargs = client.presigned_get_object.call_args
        assert args[0] == "b"
        assert args[1] == "doc1.md"
        assert kwargs["expires"] == timedelta(seconds=120)


# --------------------------------------------------------------------------- #
# list_documents
# --------------------------------------------------------------------------- #
class TestListDocuments:
    def test_meta_json_excluded(self):
        """Only .md objects produce results; .meta.json sidecars are skipped.

        Sidecar fetch raises NoSuchKey so source_key falls back to the stem;
        if the filter were inverted, the surviving entry's stem would end in
        ``.meta.json`` — the exact-value assertion below would then fail.
        """
        client = MagicMock()
        client.list_objects.return_value = iter([
            _make_list_object("alpha.md"),
            _make_list_object("alpha.meta.json"),
        ])
        client.get_object.side_effect = _s3error("NoSuchKey")

        results = store.list_documents(client, bucket="b", limit=100, offset=0)

        assert len(results) == 1
        assert results[0]["source_key"] == "alpha"

    def test_source_key_from_sidecar(self):
        """source_key is read from the sidecar metadata when present."""
        client = MagicMock()
        client.list_objects.return_value = iter([_make_list_object("alpha.md", size=42)])
        client.get_object.return_value = _make_response(
            json.dumps({"source_key": "real/source"}).encode("utf-8")
        )

        results = store.list_documents(client, bucket="b", limit=100, offset=0)

        assert results[0]["source_key"] == "real/source"
        assert results[0]["document_id"] == store.build_document_id("real/source")
        assert results[0]["size_bytes"] == 42

    def test_missing_sidecar_falls_back_to_stem_and_warns(self, caplog):
        """Missing sidecar -> source_key is the object stem; a WARNING is logged."""
        client = MagicMock()
        client.list_objects.return_value = iter([_make_list_object("stemonly.md")])
        client.get_object.side_effect = _s3error("NoSuchKey")

        with caplog.at_level("WARNING", logger="rag.db.minio.store"):
            results = store.list_documents(client, bucket="b", limit=100, offset=0)

        assert results[0]["source_key"] == "stemonly"
        assert any("sidecar missing" in r.message for r in caplog.records)

    def test_pagination_offset_limit_middle(self):
        """offset=1, limit=1 returns exactly the middle of three matches."""
        client = MagicMock()
        client.list_objects.return_value = iter([
            _make_list_object("a.md"),
            _make_list_object("b.md"),
            _make_list_object("c.md"),
        ])
        # No sidecar -> stem fallback gives distinct source_keys a/b/c.
        client.get_object.side_effect = _s3error("NoSuchKey")

        results = store.list_documents(client, bucket="b", limit=1, offset=1)

        assert len(results) == 1
        assert results[0]["source_key"] == "b"

    def test_last_modified_isoformat_when_present(self):
        """last_modified is ISO-8601 when the object carries a timestamp."""
        ts = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        client = MagicMock()
        client.list_objects.return_value = iter([
            _make_list_object("a.md", last_modified=ts),
        ])
        client.get_object.side_effect = _s3error("NoSuchKey")

        results = store.list_documents(client, bucket="b", limit=100, offset=0)

        assert results[0]["last_modified"] == ts.isoformat()

    def test_last_modified_none_when_absent(self):
        """last_modified is None when the object has no timestamp."""
        client = MagicMock()
        client.list_objects.return_value = iter([
            _make_list_object("a.md", last_modified=None),
        ])
        client.get_object.side_effect = _s3error("NoSuchKey")

        results = store.list_documents(client, bucket="b", limit=100, offset=0)

        assert results[0]["last_modified"] is None


# --------------------------------------------------------------------------- #
# get_page_image_url
# --------------------------------------------------------------------------- #
class TestGetPageImageUrl:
    def test_raw_key_no_suffix_and_exact_expiry(self):
        """presigned_get_object called with the raw key (no suffix) + exact expiry."""
        client = MagicMock()
        client.presigned_get_object.return_value = "https://img"

        url = store.get_page_image_url(
            client, "pages/doc/0001.jpg", bucket="b", expires_in_seconds=120
        )

        assert url == "https://img"
        args, kwargs = client.presigned_get_object.call_args
        assert args[0] == "b"
        assert args[1] == "pages/doc/0001.jpg"
        assert kwargs["expires"] == timedelta(seconds=120)

    def test_default_bucket_resolution_when_empty(self):
        """Empty bucket resolves to MINIO_BUCKET (FR-403)."""
        client = MagicMock()
        client.presigned_get_object.return_value = "u"

        store.get_page_image_url(client, "k", bucket="", expires_in_seconds=120)

        args, _ = client.presigned_get_object.call_args
        assert args[0] == store.MINIO_BUCKET


# --------------------------------------------------------------------------- #
# store_figure_image
# --------------------------------------------------------------------------- #
class TestStoreFigureImage:
    def test_happy_path_key_and_put_object(self):
        """Builds sanitized key, jpeg->jpg ext, and returns the key."""
        client = MagicMock()

        key = store.store_figure_image(
            client, "src/key", "Figure 1", b"\x00\x01", "image/jpeg", bucket="b"
        )

        assert key == "figures/src_key/figure_1.jpg"
        args, kwargs = client.put_object.call_args
        assert args[0] == "b"
        assert args[1] == "figures/src_key/figure_1.jpg"
        assert kwargs["length"] == 2
        assert kwargs["content_type"] == "image/jpeg"

    def test_upload_failure_returns_none(self):
        """An exception during put_object is swallowed and returns None."""
        client = MagicMock()
        client.put_object.side_effect = RuntimeError("boom")

        result = store.store_figure_image(
            client, "src", "Figure 1", b"x", "image/png", bucket="b"
        )

        assert result is None


# --------------------------------------------------------------------------- #
# store_page_images
# --------------------------------------------------------------------------- #
class TestStorePageImages:
    def test_stores_each_page_and_returns_keys(self):
        """Each page is saved as JPEG and put under pages/<id>/<page:04d>.jpg."""
        client = MagicMock()
        img1, img2 = MagicMock(), MagicMock()

        keys = store.store_page_images(
            client, "doc1", [(1, img1), (7, img2)], quality=85, bucket="b"
        )

        assert keys == ["pages/doc1/0001.jpg", "pages/doc1/0007.jpg"]
        assert client.put_object.call_count == 2
        first_args, first_kwargs = client.put_object.call_args_list[0]
        assert first_args[1] == "pages/doc1/0001.jpg"
        assert first_kwargs["content_type"] == "image/jpeg"
        img1.save.assert_called_once()
        assert img1.save.call_args.kwargs["quality"] == 85

    def test_per_page_error_isolated(self):
        """A failing page is skipped; remaining pages still stored."""
        client = MagicMock()
        good, bad = MagicMock(), MagicMock()
        bad.save.side_effect = RuntimeError("encode fail")

        keys = store.store_page_images(
            client, "doc1", [(1, bad), (2, good)], bucket="b"
        )

        assert keys == ["pages/doc1/0002.jpg"]


# --------------------------------------------------------------------------- #
# delete_page_images
# --------------------------------------------------------------------------- #
class TestDeletePageImages:
    def test_deletes_all_under_prefix(self):
        """All objects under pages/<id>/ are removed; returns the count."""
        client = MagicMock()
        client.list_objects.return_value = iter([
            _make_list_object("pages/doc1/0001.jpg"),
            _make_list_object("pages/doc1/0002.jpg"),
        ])

        deleted = store.delete_page_images(client, "doc1", bucket="b")

        assert deleted == 2
        client.list_objects.assert_called_once_with("b", prefix="pages/doc1/", recursive=True)
        removed = [c.args[1] for c in client.remove_object.call_args_list]
        assert removed == ["pages/doc1/0001.jpg", "pages/doc1/0002.jpg"]

    def test_list_failure_returns_zero(self):
        """A failure listing objects is swallowed and returns 0."""
        client = MagicMock()
        client.list_objects.side_effect = RuntimeError("list boom")

        assert store.delete_page_images(client, "doc1", bucket="b") == 0
