# @summary
# Unit tests for src.db.resolve_clean_document: the shared clean-document
# resolver's identity fallback chain (document_id -> build_document_id(source_key)
# -> build_document_id(source)), the MinioCleanStore layout fallback, library
# never-raise semantics (miss/storage-error -> None), and candidate dedup.
# Deps: pytest, src.db, src.db.common.schemas
# @end-summary
"""Tests for ``src.db.resolve_clean_document``.

All storage collaborators are monkeypatched at their lookup sites (the same
seams ``tests/server/test_console_services.py`` uses for the console wrapper):
``src.db.get_document``, ``src.db.build_document_id`` and
``src.ingest.common.minio_clean_store.MinioCleanStore``. No MinIO required.
"""
from __future__ import annotations

import pytest

import src.db as db
from src.db.common.schemas import StoredDocument


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class _FakeCleanStore:
    """MinioCleanStore double keyed by source_key."""

    def __init__(self, docs: dict | None = None, *, read_raises: bool = False):
        self._docs = docs or {}
        self._read_raises = read_raises
        self.reads: list[str] = []

    def exists(self, key: str) -> bool:
        return key in self._docs

    def read(self, key: str):
        self.reads.append(key)
        if self._read_raises:
            raise RuntimeError("boom")
        return self._docs[key]


def _install(monkeypatch, *, documents: dict | None = None, store=None):
    """Install fakes for both layouts.

    ``documents`` maps document_id -> StoredDocument for the document-store
    layout; ``store`` is the MinioCleanStore double for the clean/ layout.
    ``build_document_id`` is faked to the visible ``docid:<key>`` form so
    tests can assert exactly which fallback produced the hit.
    """
    documents = documents or {}
    calls: list[str] = []

    def _fake_get_document(client, document_id, bucket=None):
        calls.append(document_id)
        return documents.get(document_id)

    monkeypatch.setattr(db, "get_document", _fake_get_document)
    monkeypatch.setattr(db, "build_document_id", lambda key: f"docid:{key}")
    monkeypatch.setattr(
        "src.ingest.common.minio_clean_store.MinioCleanStore",
        lambda client, bucket: store if store is not None else _FakeCleanStore(),
    )
    return calls


_CLIENT = object()


# --------------------------------------------------------------------------- #
# Identity fallback chain
# --------------------------------------------------------------------------- #


def test_resolves_by_document_id_first(monkeypatch):
    doc = StoredDocument(document_id="uuid-1", content="# by id", metadata={"a": 1})
    calls = _install(monkeypatch, documents={"uuid-1": doc})
    result = db.resolve_clean_document(
        _CLIENT, document_id="uuid-1", source_key="k", source="s"
    )
    assert result is doc
    # First candidate hit — no further lookups.
    assert calls == ["uuid-1"]


def test_falls_back_to_source_key_document_id(monkeypatch):
    doc = StoredDocument(document_id="docid:key1", content="# by key", metadata={})
    calls = _install(monkeypatch, documents={"docid:key1": doc})
    result = db.resolve_clean_document(
        _CLIENT, document_id="uuid-miss", source_key="key1", source="s"
    )
    assert result is doc
    assert calls == ["uuid-miss", "docid:key1"]


def test_falls_back_to_source_document_id(monkeypatch):
    doc = StoredDocument(document_id="docid:name.pdf", content="# by source", metadata={})
    calls = _install(monkeypatch, documents={"docid:name.pdf": doc})
    result = db.resolve_clean_document(
        _CLIENT, document_id="uuid-miss", source_key="key-miss", source="name.pdf"
    )
    assert result is doc
    assert calls == ["uuid-miss", "docid:key-miss", "docid:name.pdf"]


def test_duplicate_candidates_deduped(monkeypatch):
    # document_id identical to build_document_id(source_key): looked up once.
    calls = _install(monkeypatch)
    db.resolve_clean_document(_CLIENT, document_id="docid:key1", source_key="key1")
    assert calls == ["docid:key1"]


def test_empty_content_treated_as_miss(monkeypatch):
    hollow = StoredDocument(document_id="uuid-1", content="", metadata={})
    full = StoredDocument(document_id="docid:key1", content="# full", metadata={})
    _install(monkeypatch, documents={"uuid-1": hollow, "docid:key1": full})
    result = db.resolve_clean_document(_CLIENT, document_id="uuid-1", source_key="key1")
    assert result is full


# --------------------------------------------------------------------------- #
# Clean-store layout fallback
# --------------------------------------------------------------------------- #


def test_clean_store_fallback_by_source_key(monkeypatch):
    store = _FakeCleanStore({"key1": ("# clean md", {"source": "x"})})
    _install(monkeypatch, store=store)
    result = db.resolve_clean_document(_CLIENT, source_key="key1")
    assert result is not None
    assert result.content == "# clean md"
    assert result.metadata == {"source": "x"}
    assert result.document_id == "docid:key1"


def test_clean_store_fallback_by_source(monkeypatch):
    store = _FakeCleanStore({"name.pdf": ("# via source", {})})
    _install(monkeypatch, store=store)
    result = db.resolve_clean_document(_CLIENT, source_key="key-miss", source="name.pdf")
    assert result is not None
    assert result.content == "# via source"


def test_clean_store_not_tried_without_source_identity(monkeypatch):
    # document_id only: the clean/ layout is keyed by source identity, so a
    # document-store miss with no source_key/source resolves to None without
    # touching the clean store.
    store = _FakeCleanStore({"anything": ("# md", {})})
    _install(monkeypatch, store=store)
    assert db.resolve_clean_document(_CLIENT, document_id="uuid-miss") is None
    assert store.reads == []


# --------------------------------------------------------------------------- #
# Miss and never-raise semantics
# --------------------------------------------------------------------------- #


def test_all_layouts_miss_returns_none(monkeypatch):
    _install(monkeypatch)
    assert (
        db.resolve_clean_document(
            _CLIENT, document_id="u", source_key="k", source="s"
        )
        is None
    )


def test_no_identifiers_returns_none(monkeypatch):
    calls = _install(monkeypatch)
    assert db.resolve_clean_document(_CLIENT) is None
    assert calls == []


def test_document_store_error_is_swallowed_and_fallback_continues(monkeypatch):
    doc = StoredDocument(document_id="docid:key1", content="# ok", metadata={})

    def _boom_then_hit(client, document_id, bucket=None):
        if document_id == "uuid-1":
            raise RuntimeError("minio down")
        return {"docid:key1": doc}.get(document_id)

    monkeypatch.setattr(db, "get_document", _boom_then_hit)
    monkeypatch.setattr(db, "build_document_id", lambda key: f"docid:{key}")
    result = db.resolve_clean_document(_CLIENT, document_id="uuid-1", source_key="key1")
    assert result is doc


def test_clean_store_read_error_returns_none(monkeypatch):
    store = _FakeCleanStore({"key1": ("md", {})}, read_raises=True)
    _install(monkeypatch, store=store)
    assert db.resolve_clean_document(_CLIENT, source_key="key1") is None
