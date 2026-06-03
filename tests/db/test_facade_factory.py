"""Factory dispatch + facade-forwarding tests for the ``src.db`` package API.

This suite isolates the *public facade* in ``src/db/__init__.py`` from any real
backend. It proves four things:

1. **Backend dispatch** — ``_get_db_backend()`` selects ``MinioBackend`` when
   ``DATABASE_BACKEND == "minio"`` and raises a descriptive ``ValueError`` for
   any other value.
2. **Singleton caching** — the backend is instantiated exactly once and reused
   across calls (the module-global ``_db_backend`` cache).
3. **Exact-args forwarding** — every public facade function threads its
   arguments to the singleton backend with the precise positional/keyword shape
   (including default values), and returns the backend's return value verbatim.
4. **Context-manager plumbing** — ``get_client()`` enters and exits the
   backend's ``get_ephemeral_client`` CM and yields its client.

Hazards handled here:

* ``src.db._db_backend`` is a process-wide module global; an autouse fixture
  resets it to ``None`` before and after each test.
* ``_get_db_backend`` does ``from config.settings import DATABASE_BACKEND`` and
  ``from src.db.minio import MinioBackend`` at *call time*, so both are patched
  at their import sites (``config.settings.DATABASE_BACKEND`` and
  ``src.db.minio.MinioBackend``).
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

import src.db as db


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class FakeBackend:
    """Recording double standing in for ``MinioBackend``.

    Every method records ``(args, kwargs)`` into ``self.calls[name]`` and
    returns a per-method sentinel so callers can assert the facade returns the
    backend's value verbatim. ``get_ephemeral_client`` returns a contextmanager
    yielding ``self.ephemeral_client`` and records enter/exit.
    """

    #: class-level instantiation counter so the "instantiated once" invariant
    #: can be asserted independently of any single instance.
    instances = 0

    def __init__(self) -> None:
        FakeBackend.instances += 1
        self.calls: dict = {}
        self.ephemeral_client = object()
        self.cm_entered = False
        self.cm_exited = False
        # Per-method return sentinels.
        self.ret = {
            "create_persistent_client": object(),
            "close_client": None,
            "ensure_bucket": None,
            "put_document": None,
            "get_document": object(),
            "delete_document": object(),
            "document_exists": object(),
            "get_document_url": object(),
            "list_documents": object(),
        }

    def _record(self, name, args, kwargs):
        self.calls[name] = (args, kwargs)
        return self.ret[name]

    def create_persistent_client(self, *args, **kwargs):
        return self._record("create_persistent_client", args, kwargs)

    def close_client(self, *args, **kwargs):
        return self._record("close_client", args, kwargs)

    def ensure_bucket(self, *args, **kwargs):
        return self._record("ensure_bucket", args, kwargs)

    def put_document(self, *args, **kwargs):
        return self._record("put_document", args, kwargs)

    def get_document(self, *args, **kwargs):
        return self._record("get_document", args, kwargs)

    def delete_document(self, *args, **kwargs):
        return self._record("delete_document", args, kwargs)

    def document_exists(self, *args, **kwargs):
        return self._record("document_exists", args, kwargs)

    def get_document_url(self, *args, **kwargs):
        return self._record("get_document_url", args, kwargs)

    def list_documents(self, *args, **kwargs):
        return self._record("list_documents", args, kwargs)

    @contextmanager
    def get_ephemeral_client(self, *args, **kwargs):
        self.calls["get_ephemeral_client"] = (args, kwargs)
        self.cm_entered = True
        try:
            yield self.ephemeral_client
        finally:
            self.cm_exited = True


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the process-wide backend cache before and after each test."""
    db._db_backend = None
    FakeBackend.instances = 0
    yield
    db._db_backend = None
    FakeBackend.instances = 0


@pytest.fixture
def minio_env(monkeypatch):
    """Patch config + backend class so dispatch resolves to ``FakeBackend``."""
    monkeypatch.setattr("config.settings.DATABASE_BACKEND", "minio")
    monkeypatch.setattr("src.db.minio.MinioBackend", FakeBackend)


@pytest.fixture
def backend(minio_env):
    """Return the cached FakeBackend singleton after first dispatch."""
    return db._get_db_backend()


# --------------------------------------------------------------------------- #
# Dispatch + caching
# --------------------------------------------------------------------------- #
class TestDispatch:
    def test_happy_path_returns_minio_backend(self, minio_env):
        """DATABASE_BACKEND='minio' yields a MinioBackend (fake) instance."""
        result = db._get_db_backend()
        assert isinstance(result, FakeBackend)

    def test_singleton_caching_same_object_instantiated_once(self, minio_env):
        """Repeat calls return the SAME object; class constructed exactly once."""
        first = db._get_db_backend()
        second = db._get_db_backend()
        assert first is second
        assert FakeBackend.instances == 1

    def test_unknown_backend_raises_valueerror(self, monkeypatch):
        """A non-'minio' backend raises ValueError naming the bad value + minio."""
        monkeypatch.setattr("config.settings.DATABASE_BACKEND", "postgres")
        with pytest.raises(ValueError) as exc:
            db._get_db_backend()
        msg = str(exc.value)
        assert "postgres" in msg
        assert "minio" in msg


# --------------------------------------------------------------------------- #
# Facade forwarding — exact args + return value
# --------------------------------------------------------------------------- #
class TestFacadeForwarding:
    def test_create_persistent_client(self, backend):
        ret = db.create_persistent_client()
        assert backend.calls["create_persistent_client"] == ((), {})
        assert ret is backend.ret["create_persistent_client"]

    def test_close_client(self, backend):
        client = object()
        db.close_client(client)
        assert backend.calls["close_client"] == ((client,), {})

    def test_ensure_bucket_default(self, backend):
        client = object()
        db.ensure_bucket(client)
        assert backend.calls["ensure_bucket"] == ((client, None), {})

    def test_ensure_bucket_explicit(self, backend):
        client = object()
        db.ensure_bucket(client, "mybucket")
        assert backend.calls["ensure_bucket"] == ((client, "mybucket"), {})

    def test_put_document_all_args(self, backend):
        client = object()
        meta = {"source_key": "k", "connector": "c"}
        db.put_document(client, "doc-id", "the content", meta, "buck")
        assert backend.calls["put_document"] == (
            (client, "doc-id", "the content", meta, "buck"),
            {},
        )

    def test_put_document_default_bucket(self, backend):
        client = object()
        meta = {"a": 1}
        db.put_document(client, "doc-id", "body", meta)
        assert backend.calls["put_document"] == (
            (client, "doc-id", "body", meta, None),
            {},
        )

    def test_get_document(self, backend):
        client = object()
        ret = db.get_document(client, "doc-id", "buck")
        assert backend.calls["get_document"] == ((client, "doc-id", "buck"), {})
        assert ret is backend.ret["get_document"]

    def test_get_document_default_bucket(self, backend):
        client = object()
        db.get_document(client, "doc-id")
        assert backend.calls["get_document"] == ((client, "doc-id", None), {})

    def test_delete_document(self, backend):
        client = object()
        ret = db.delete_document(client, "doc-id", "buck")
        assert backend.calls["delete_document"] == ((client, "doc-id", "buck"), {})
        assert ret is backend.ret["delete_document"]

    def test_document_exists(self, backend):
        client = object()
        ret = db.document_exists(client, "doc-id", "buck")
        assert backend.calls["document_exists"] == ((client, "doc-id", "buck"), {})
        assert ret is backend.ret["document_exists"]

    def test_get_document_url_default_expiry(self, backend):
        """Omitting expires_in_seconds must forward the 3600 default."""
        client = object()
        ret = db.get_document_url(client, "doc-id")
        assert backend.calls["get_document_url"] == (
            (client, "doc-id", None, 3600),
            {},
        )
        assert ret is backend.ret["get_document_url"]

    def test_get_document_url_explicit_expiry(self, backend):
        client = object()
        db.get_document_url(client, "doc-id", "buck", 120)
        assert backend.calls["get_document_url"] == (
            (client, "doc-id", "buck", 120),
            {},
        )

    def test_list_documents_defaults(self, backend):
        """Omitting optional args forwards prefix=''/limit=1000/offset=0."""
        client = object()
        ret = db.list_documents(client)
        assert backend.calls["list_documents"] == (
            (client, None, "", 1000, 0),
            {},
        )
        assert ret is backend.ret["list_documents"]

    def test_list_documents_explicit(self, backend):
        client = object()
        db.list_documents(client, "buck", "pre/", 25, 50)
        assert backend.calls["list_documents"] == (
            (client, "buck", "pre/", 25, 50),
            {},
        )


# --------------------------------------------------------------------------- #
# get_client context manager
# --------------------------------------------------------------------------- #
class TestGetClientContextManager:
    def test_yields_backend_ephemeral_client_and_enters_exits(self, backend):
        assert backend.cm_entered is False
        with db.get_client() as client:
            assert backend.cm_entered is True
            assert backend.cm_exited is False
            assert client is backend.ephemeral_client
        assert backend.cm_exited is True
        assert "get_ephemeral_client" in backend.calls


# --------------------------------------------------------------------------- #
# Re-exports
# --------------------------------------------------------------------------- #
class TestReExports:
    def test_stored_document_and_build_document_id_importable(self):
        from src.db import StoredDocument, build_document_id

        assert StoredDocument is not None
        assert callable(build_document_id)
