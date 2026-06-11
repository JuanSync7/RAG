# @summary
# Live-stack integration tests for the object-storage facade (src.db) against
# the real MinIO container. Exercises the public document-store API end-to-end:
# ensure_bucket -> put_document -> document_exists/get_document (byte-identical
# round-trip) -> list_documents -> delete_document -> exists False. Every test
# creates a dedicated ``itest-<uuid>`` bucket and deletes it (and its objects)
# in a finalizer, so nothing leaks into the shared instance.
# Deps: pytest, src.db (real facade), minio (real package), config.settings
# @end-summary
"""Live-MinIO integration tests for the document-store round-trip.

All tests carry ``pytestmark = [slow, integration]`` so the offline CI gate
(``-m "not slow and not integration"``) deselects them. They run against the
shared ``rag-minio`` instance only when the live markers are selected.

Shared-instance safety:
  - Every test creates a *dedicated* bucket named ``itest-<uuid>`` and a
    finalizer always removes every object plus the bucket, even on failure.
  - No pre-existing bucket (e.g. the default ``rag-documents``) is touched.

Why a dedicated bucket per test: the ``src.db`` facade is document-oriented
(content is text written as ``<id>.md`` + a ``.meta.json`` sidecar) and has no
key-prefix isolation primitive, so an isolated bucket is the clean unit of
teardown on a shared server.

The stub problem: ``tests/conftest.py`` installs an in-memory ``minio`` stub at
collection time. ``_real_minio_or_skip()`` evicts that stub, imports the real
``minio`` package, and reloads ``src.db`` so the facade binds the real client —
mirroring the weaviate eviction in test_weaviate_store_integration.py.
"""

from __future__ import annotations

# Force the REAL ``minio`` package into sys.modules BEFORE tests/conftest.py
# installs its in-memory stub. Without this the live tests below would hold a
# stubbed client and silently lose their round-trip guarantees.
try:  # pragma: no cover - import-side-effect only
    import minio  # noqa: F401
    import src.db  # noqa: F401
except Exception:  # pragma: no cover - infra-dependent fallback
    pass

import uuid
from typing import Any, Iterator

import pytest


# Dual-marker per project memory (feedback_dual_marker_gating): both slow AND
# integration so ``-m "not slow and not integration"`` reliably excludes them.
pytestmark = [pytest.mark.slow, pytest.mark.integration]


# ---------------------------------------------------------------------------
# Stub eviction + live client fixtures
# ---------------------------------------------------------------------------

def _real_minio_or_skip() -> None:
    """Swap the conftest stub for the real minio package and reload src.db.

    Skips (rather than fails) when the real package or the live instance is
    unavailable so the suite does not pretend-pass with infra down.
    """
    import importlib  # noqa: PLC0415
    import sys  # noqa: PLC0415

    for mod_name in list(sys.modules):
        if mod_name == "minio" or mod_name.startswith("minio."):
            del sys.modules[mod_name]
    try:
        import minio  # noqa: F401, PLC0415
        import minio.error  # noqa: F401, PLC0415
    except Exception as exc:  # pragma: no cover - infra-dependent
        pytest.skip(f"Real minio package not importable: {exc}")

    # The real minio package has a __file__; the conftest stub does not. If the
    # stub is still bound, eviction failed and we must not proceed.
    if getattr(sys.modules.get("minio"), "__file__", None) is None:  # pragma: no cover
        pytest.skip("Real minio package could not be loaded over the stub")

    # Reload the facade so MinioBackend/store bind the real Minio + S3Error.
    for mod_name in [
        "src.db.minio.store",
        "src.db.minio.backend",
        "src.db.minio",
        "src.db.backend",
        "src.db",
    ]:
        if mod_name in sys.modules:
            try:
                importlib.reload(sys.modules[mod_name])
            except Exception:
                pass


@pytest.fixture
def live_db_client() -> Iterator[Any]:
    """Yield a real persistent MinIO client; skip if the instance is down."""
    _real_minio_or_skip()

    import src.db as db  # noqa: PLC0415

    # Reset the facade singleton so it rebuilds the backend against the real
    # minio package (it may have been constructed against the stub earlier).
    db._db_backend = None  # type: ignore[attr-defined]

    try:
        client = db.create_persistent_client()
    except Exception as exc:  # pragma: no cover - infra-dependent
        pytest.skip(f"Live MinIO client could not be created: {exc}")

    # Liveness probe: a trivial call that requires a real server round-trip.
    try:
        client.bucket_exists(f"itest-probe-{uuid.uuid4().hex}")
    except Exception as exc:  # pragma: no cover - infra-dependent
        pytest.skip(f"Live MinIO not reachable: {exc}")

    yield client


@pytest.fixture
def itest_bucket(live_db_client: Any) -> Iterator[str]:
    """Create a unique ``itest-<uuid>`` bucket; always purge + delete it."""
    import src.db as db  # noqa: PLC0415

    bucket = f"itest-{uuid.uuid4().hex}"
    db.ensure_bucket(live_db_client, bucket)
    try:
        yield bucket
    finally:
        # Remove every object then drop the bucket. MinIO refuses to delete a
        # non-empty bucket, so the object sweep must precede remove_bucket.
        try:
            for obj in live_db_client.list_objects(bucket, recursive=True):
                try:
                    live_db_client.remove_object(bucket, obj.object_name)
                except Exception:
                    pass
            live_db_client.remove_bucket(bucket)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Full round-trip: put -> exists -> get (byte-identical) -> list -> delete
# ---------------------------------------------------------------------------

def test_document_round_trip_byte_identical(
    live_db_client: Any, itest_bucket: str
) -> None:
    """Full lifecycle against live MinIO through the public src.db facade.

    Asserts, in order:
      * the doc is absent before any write;
      * after put_document the content object exists;
      * get_document returns a StoredDocument whose content is BYTE-IDENTICAL
        to what was written (utf-8 round-trip, including non-ASCII) and whose
        metadata round-trips field-for-field;
      * list_documents surfaces exactly this document (by source_key);
      * delete_document reports it existed and exists flips back to False;
      * get_document returns None after delete.
    """
    import src.db as db  # noqa: PLC0415
    from src.db import StoredDocument  # noqa: PLC0415

    client = live_db_client
    bucket = itest_bucket

    source_key = f"itest/{uuid.uuid4().hex}/datasheet.pdf"
    document_id = db.build_document_id(source_key)
    # Non-ASCII + newlines so a lossy/garbled round-trip would be caught.
    content = "Operating voltage is 3.3 V — ±5%.\nChar�actérs: αβγ 数据手册\n"
    metadata = {
        "source_key": source_key,
        "source_uri": "file:///itest/datasheet.pdf",
        "connector": "itest",
    }

    # --- absent before write ---
    assert db.document_exists(client, document_id, bucket=bucket) is False
    assert db.get_document(client, document_id, bucket=bucket) is None

    # --- put ---
    db.put_document(client, document_id, content, metadata, bucket=bucket)

    # --- exists ---
    assert db.document_exists(client, document_id, bucket=bucket) is True

    # --- get: byte-identical content + metadata round-trip ---
    fetched = db.get_document(client, document_id, bucket=bucket)
    assert isinstance(fetched, StoredDocument)
    assert fetched.document_id == document_id
    # Load-bearing: content survives the round-trip byte-for-byte.
    assert fetched.content == content
    assert fetched.content.encode("utf-8") == content.encode("utf-8")
    assert fetched.metadata == metadata

    # --- list: this doc is present, keyed by its source_key ---
    listed = db.list_documents(client, bucket=bucket, limit=100)
    keys = {entry["source_key"] for entry in listed}
    assert source_key in keys, f"{source_key!r} not in listed source_keys {keys}"
    entry = next(e for e in listed if e["source_key"] == source_key)
    # The list view derives document_id from the sidecar source_key — it must
    # agree with the deterministic id we wrote under.
    assert entry["document_id"] == document_id
    assert entry["size_bytes"] == len(content.encode("utf-8"))

    # --- delete: reports prior existence, then absent ---
    existed = db.delete_document(client, document_id, bucket=bucket)
    assert existed is True
    assert db.document_exists(client, document_id, bucket=bucket) is False
    assert db.get_document(client, document_id, bucket=bucket) is None


# ---------------------------------------------------------------------------
# Overwrite semantics: a second put replaces the content in place
# ---------------------------------------------------------------------------

def test_put_document_overwrites_in_place(
    live_db_client: Any, itest_bucket: str
) -> None:
    """A second put_document under the same id replaces content + metadata.

    Gives the round-trip teeth: if put were append-only or id-namespaced wrong,
    the second fetch would still see the first body. The exact-equality checks
    on the *new* values would then fail.
    """
    import src.db as db  # noqa: PLC0415

    client = live_db_client
    bucket = itest_bucket
    document_id = db.build_document_id(f"itest/{uuid.uuid4().hex}/v")

    db.put_document(client, document_id, "first body", {"v": "1"}, bucket=bucket)
    first = db.get_document(client, document_id, bucket=bucket)
    assert first is not None and first.content == "first body"
    assert first.metadata == {"v": "1"}

    db.put_document(client, document_id, "second body", {"v": "2"}, bucket=bucket)
    second = db.get_document(client, document_id, bucket=bucket)
    assert second is not None
    assert second.content == "second body"  # load-bearing: overwrote, not appended
    assert second.metadata == {"v": "2"}


# ---------------------------------------------------------------------------
# delete_document on a missing id reports False (idempotent absence)
# ---------------------------------------------------------------------------

def test_delete_missing_document_returns_false(
    live_db_client: Any, itest_bucket: str
) -> None:
    """delete_document on an id that was never written returns False."""
    import src.db as db  # noqa: PLC0415

    client = live_db_client
    bucket = itest_bucket
    missing_id = db.build_document_id(f"itest/{uuid.uuid4().hex}/never-written")

    assert db.delete_document(client, missing_id, bucket=bucket) is False


# ---------------------------------------------------------------------------
# Bucket isolation: a doc written to bucket A is invisible in bucket B
# ---------------------------------------------------------------------------

def test_bucket_isolation(live_db_client: Any, itest_bucket: str) -> None:
    """A document written to one bucket never surfaces via another bucket.

    The defining assertion: get_document against a *second* isolated bucket
    returns None for the same document_id, proving writes are bucket-scoped.
    """
    import src.db as db  # noqa: PLC0415

    client = live_db_client
    bucket_a = itest_bucket
    bucket_b = f"itest-{uuid.uuid4().hex}"
    db.ensure_bucket(client, bucket_b)

    document_id = db.build_document_id(f"itest/{uuid.uuid4().hex}/iso")
    try:
        db.put_document(client, document_id, "only in A", {"k": "a"}, bucket=bucket_a)

        # Present in A ...
        in_a = db.get_document(client, document_id, bucket=bucket_a)
        assert in_a is not None and in_a.content == "only in A"
        # ... and absent in B.
        assert db.document_exists(client, document_id, bucket=bucket_b) is False
        assert db.get_document(client, document_id, bucket=bucket_b) is None
        assert db.list_documents(client, bucket=bucket_b, limit=100) == []
    finally:
        try:
            for obj in client.list_objects(bucket_b, recursive=True):
                try:
                    client.remove_object(bucket_b, obj.object_name)
                except Exception:
                    pass
            client.remove_bucket(bucket_b)
        except Exception:
            pass
