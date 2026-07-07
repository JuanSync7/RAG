"""Unit tests for ``server/routes/documents.py`` (read-only doc/collection API).

These tests drive the router built by ``create_documents_router`` through a
FastAPI ``TestClient`` with no infrastructure: the ``db`` / ``vector_db``
backend functions are monkeypatched on the route module namespace, and the
``authenticate_request`` dependency is overridden with a static principal.

Coverage spans happy paths, error mapping (404 / 503), graceful Weaviate
degradation, filtering, sorting, pagination, source grouping, and query-param
validation (422). The data is deliberately shaped so that no-sort, no-slice,
total-from-page, filter-inversion, degrade-to-zero, and 404/503 mutations all
red a specific named test (see the slice's mutation ledger).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.routes.documents import create_documents_router
from src.platform.security import Principal, authenticate_request


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _principal() -> Principal:
    return Principal(
        subject="user-1",
        tenant_id="tenant-a",
        roles=["query"],
        auth_type="none",
        project_id="proj-x",
    )


_DB_SENTINEL = object()
_VECTOR_SENTINEL = object()


@pytest.fixture
def client() -> TestClient:
    """A TestClient wrapping an app with only the documents router mounted."""
    app = FastAPI()
    app.include_router(
        create_documents_router(
            db_client=_DB_SENTINEL, vector_client=_VECTOR_SENTINEL
        )
    )
    app.dependency_overrides[authenticate_request] = _principal
    return TestClient(app)


def _set(monkeypatch, name, fn):
    monkeypatch.setattr(f"server.routes.documents.{name}", fn)


def _raise(*_a, **_k):
    raise RuntimeError("backend boom")


# ===========================================================================
# GET /api/v1/documents
# ===========================================================================


def test_list_documents_happy(client, monkeypatch):
    """Multiple docs returned; chunk_count + connector resolved from the
    Weaviate aggregate lookup keyed by source_key."""
    minio_docs = [
        {"document_id": "d-alpha", "source_key": "alpha.pdf", "last_modified": "2026-01-01"},
        {"document_id": "d-beta", "source_key": "beta.pdf", "last_modified": "2026-01-02"},
    ]
    agg_rows = [
        {"source_key": "alpha.pdf", "source": "alpha.pdf", "connector": "minio", "chunk_count": 7},
        {"source_key": "beta.pdf", "source": "beta.pdf", "connector": "s3", "chunk_count": 3},
    ]
    _set(monkeypatch, "db.list_documents", lambda *_a, **_k: list(minio_docs))
    _set(monkeypatch, "vector_db.aggregate_by_source", lambda *_a, **_k: list(agg_rows))

    resp = client.get("/api/v1/documents")
    assert resp.status_code == 200
    body = resp.json()

    assert body["total"] == 2
    assert body["limit"] == 20
    assert body["offset"] == 0
    by_id = {d["document_id"]: d for d in body["documents"]}
    assert by_id["d-alpha"]["chunk_count"] == 7
    assert by_id["d-alpha"]["connector"] == "minio"
    assert by_id["d-beta"]["chunk_count"] == 3
    assert by_id["d-beta"]["connector"] == "s3"
    assert by_id["d-alpha"]["ingested_at"] == "2026-01-01"


def test_list_documents_sorted_by_source(client, monkeypatch):
    """Results are sorted by ``source`` ascending even when input is
    deliberately out of order (kills a no-sort mutation)."""
    minio_docs = [
        {"document_id": "d3", "source_key": "ccc.pdf"},
        {"document_id": "d1", "source_key": "aaa.pdf"},
        {"document_id": "d2", "source_key": "bbb.pdf"},
    ]
    _set(monkeypatch, "db.list_documents", lambda *_a, **_k: list(minio_docs))
    _set(monkeypatch, "vector_db.aggregate_by_source", lambda *_a, **_k: [])

    resp = client.get("/api/v1/documents")
    assert resp.status_code == 200
    sources = [d["source"] for d in resp.json()["documents"]]
    assert sources == ["aaa.pdf", "bbb.pdf", "ccc.pdf"]


def test_list_documents_pagination(client, monkeypatch):
    """offset+limit slices the page; total reflects the FULL filtered count,
    not the page length (kills both no-slice and total=len(page) mutations)."""
    minio_docs = [
        {"document_id": f"d{i}", "source_key": f"{i:02d}.pdf"} for i in range(5)
    ]
    _set(monkeypatch, "db.list_documents", lambda *_a, **_k: list(minio_docs))
    _set(monkeypatch, "vector_db.aggregate_by_source", lambda *_a, **_k: [])

    resp = client.get("/api/v1/documents", params={"limit": 2, "offset": 1})
    assert resp.status_code == 200
    body = resp.json()

    # Sorted sources: 00,01,02,03,04 -> offset 1, limit 2 -> 01,02
    assert [d["source"] for d in body["documents"]] == ["01.pdf", "02.pdf"]
    assert len(body["documents"]) == 2
    assert body["total"] == 5  # full count, not page length
    assert body["limit"] == 2
    assert body["offset"] == 1


def test_list_documents_source_filter(client, monkeypatch):
    """source_filter keeps only source_keys containing the substring; a row
    that does NOT match is excluded (kills not_in->in inversion)."""
    minio_docs = [
        {"document_id": "keep", "source_key": "reports/q1.pdf"},
        {"document_id": "drop", "source_key": "images/logo.png"},
    ]
    _set(monkeypatch, "db.list_documents", lambda *_a, **_k: list(minio_docs))
    _set(monkeypatch, "vector_db.aggregate_by_source", lambda *_a, **_k: [])

    resp = client.get("/api/v1/documents", params={"source_filter": "reports"})
    assert resp.status_code == 200
    ids = [d["document_id"] for d in resp.json()["documents"]]
    assert ids == ["keep"]


def test_list_documents_connector_filter(client, monkeypatch):
    """connector_filter keeps only exact connector matches."""
    minio_docs = [
        {"document_id": "d-minio", "source_key": "a.pdf"},
        {"document_id": "d-s3", "source_key": "b.pdf"},
    ]
    agg_rows = [
        {"source_key": "a.pdf", "source": "a.pdf", "connector": "minio", "chunk_count": 1},
        {"source_key": "b.pdf", "source": "b.pdf", "connector": "s3", "chunk_count": 1},
    ]
    _set(monkeypatch, "db.list_documents", lambda *_a, **_k: list(minio_docs))
    _set(monkeypatch, "vector_db.aggregate_by_source", lambda *_a, **_k: list(agg_rows))

    resp = client.get("/api/v1/documents", params={"connector_filter": "s3"})
    assert resp.status_code == 200
    ids = [d["document_id"] for d in resp.json()["documents"]]
    assert ids == ["d-s3"]


def test_list_documents_weaviate_degraded(client, monkeypatch):
    """When aggregate_by_source raises, the endpoint still 200s and
    chunk_count is null (None), not 0 (kills degrade-to-zero mutation)."""
    minio_docs = [{"document_id": "d1", "source_key": "a.pdf"}]
    _set(monkeypatch, "db.list_documents", lambda *_a, **_k: list(minio_docs))
    _set(monkeypatch, "vector_db.aggregate_by_source", _raise)

    resp = client.get("/api/v1/documents")
    assert resp.status_code == 200
    doc = resp.json()["documents"][0]
    assert doc["chunk_count"] is None
    assert doc["connector"] == "unknown"


def test_list_documents_minio_unavailable_503(client, monkeypatch):
    """db.list_documents raising maps to a 503 service_unavailable envelope."""
    _set(monkeypatch, "db.list_documents", _raise)
    _set(monkeypatch, "vector_db.aggregate_by_source", lambda *_a, **_k: [])

    resp = client.get("/api/v1/documents")
    assert resp.status_code == 503
    assert resp.json()["detail"]["error"]["code"] == "service_unavailable"


def test_list_documents_limit_out_of_bounds_422(client, monkeypatch):
    """limit above the le=100 bound triggers FastAPI validation (422)."""
    _set(monkeypatch, "db.list_documents", lambda *_a, **_k: [])
    _set(monkeypatch, "vector_db.aggregate_by_source", lambda *_a, **_k: [])

    resp = client.get("/api/v1/documents", params={"limit": 9999})
    assert resp.status_code == 422


# ===========================================================================
# GET /api/v1/documents/{document_id}
# ===========================================================================


def test_get_document_happy(client, monkeypatch):
    """Content, metadata, and chunk_count (from aggregate) are returned."""
    doc = SimpleNamespace(
        content="hello world", metadata={"source_key": "a.pdf", "k": "v"}
    )
    agg_rows = [{"source_key": "a.pdf", "source": "a.pdf", "connector": "minio", "chunk_count": 9}]
    _set(monkeypatch, "db.get_document", lambda *_a, **_k: doc)
    _set(monkeypatch, "vector_db.aggregate_by_source", lambda *_a, **_k: list(agg_rows))

    resp = client.get("/api/v1/documents/doc-1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["document_id"] == "doc-1"
    assert body["content"] == "hello world"
    assert body["metadata"] == {"source_key": "a.pdf", "k": "v"}
    assert body["chunk_count"] == 9


def test_get_document_404(client, monkeypatch):
    """db.get_document returning None maps to a 404 not_found envelope."""
    _set(monkeypatch, "db.get_document", lambda *_a, **_k: None)
    _set(monkeypatch, "vector_db.aggregate_by_source", lambda *_a, **_k: [])

    resp = client.get("/api/v1/documents/missing")
    assert resp.status_code == 404
    assert resp.json()["detail"]["error"]["code"] == "not_found"


def test_get_document_store_error_503(client, monkeypatch):
    """db.get_document raising maps to a 503 service_unavailable envelope."""
    _set(monkeypatch, "db.get_document", _raise)

    resp = client.get("/api/v1/documents/doc-1")
    assert resp.status_code == 503
    assert resp.json()["detail"]["error"]["code"] == "service_unavailable"


# ===========================================================================
# GET /api/v1/documents/{document_id}/url
# ===========================================================================


def test_document_url_happy(client, monkeypatch):
    """Existing doc -> presigned url returned with expires_in echoed."""
    _set(monkeypatch, "db.document_exists", lambda *_a, **_k: True)
    _set(monkeypatch, "db.get_document_url", lambda *_a, **_k: "https://minio/presigned")

    resp = client.get("/api/v1/documents/doc-1/url", params={"expires_in": 120})
    assert resp.status_code == 200
    body = resp.json()
    assert body["document_id"] == "doc-1"
    assert body["url"] == "https://minio/presigned"
    assert body["expires_in"] == 120


def test_document_url_404(client, monkeypatch):
    """document_exists False -> 404 not_found."""
    _set(monkeypatch, "db.document_exists", lambda *_a, **_k: False)
    _set(monkeypatch, "db.get_document_url", lambda *_a, **_k: "unused")

    resp = client.get("/api/v1/documents/missing/url")
    assert resp.status_code == 404
    assert resp.json()["detail"]["error"]["code"] == "not_found"


def test_document_url_exists_check_error_503(client, monkeypatch):
    """document_exists raising maps to 503."""
    _set(monkeypatch, "db.document_exists", _raise)

    resp = client.get("/api/v1/documents/doc-1/url")
    assert resp.status_code == 503
    assert resp.json()["detail"]["error"]["code"] == "service_unavailable"


def test_document_url_gen_error_503(client, monkeypatch):
    """get_document_url raising (after exists=True) maps to 503."""
    _set(monkeypatch, "db.document_exists", lambda *_a, **_k: True)
    _set(monkeypatch, "db.get_document_url", _raise)

    resp = client.get("/api/v1/documents/doc-1/url")
    assert resp.status_code == 503
    assert resp.json()["detail"]["error"]["code"] == "service_unavailable"


def test_document_url_expires_out_of_bounds_422(client, monkeypatch):
    """expires_in below the ge=60 bound triggers validation (422)."""
    _set(monkeypatch, "db.document_exists", lambda *_a, **_k: True)
    _set(monkeypatch, "db.get_document_url", lambda *_a, **_k: "x")

    resp = client.get("/api/v1/documents/doc-1/url", params={"expires_in": 1})
    assert resp.status_code == 422


# ===========================================================================
# GET /api/v1/sources
# ===========================================================================


def test_list_sources_grouping(client, monkeypatch):
    """Distinct source:connector keys aggregate document_count and SUM
    chunk_count. Two rows share a key (exercise increment), and a second
    connector on the same source forms a distinct group."""
    agg_rows = [
        {"source": "alpha", "connector": "minio", "source_key": "alpha/1", "chunk_count": 4},
        {"source": "alpha", "connector": "minio", "source_key": "alpha/2", "chunk_count": 6},
        {"source": "alpha", "connector": "s3", "source_key": "alpha/3", "chunk_count": 5},
    ]
    _set(monkeypatch, "vector_db.aggregate_by_source", lambda *_a, **_k: list(agg_rows))

    resp = client.get("/api/v1/sources")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2  # (alpha:minio), (alpha:s3)
    by = {(s["source"], s["connector"]): s for s in body["sources"]}

    minio = by[("alpha", "minio")]
    assert minio["document_count"] == 2
    assert minio["chunk_count"] == 10  # 4 + 6, kills +=0 mutation

    s3 = by[("alpha", "s3")]
    assert s3["document_count"] == 1
    assert s3["chunk_count"] == 5


def test_list_sources_ordering_and_pagination(client, monkeypatch):
    """Sources are sorted by source ascending then sliced by offset/limit;
    total is the full grouped count."""
    agg_rows = [
        {"source": "ccc", "connector": "x", "source_key": "k1", "chunk_count": 1},
        {"source": "aaa", "connector": "x", "source_key": "k2", "chunk_count": 1},
        {"source": "bbb", "connector": "x", "source_key": "k3", "chunk_count": 1},
    ]
    _set(monkeypatch, "vector_db.aggregate_by_source", lambda *_a, **_k: list(agg_rows))

    resp = client.get("/api/v1/sources", params={"limit": 1, "offset": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert [s["source"] for s in body["sources"]] == ["bbb"]
    assert body["total"] == 3


def test_list_sources_503(client, monkeypatch):
    """aggregate_by_source raising maps to 503."""
    _set(monkeypatch, "vector_db.aggregate_by_source", _raise)

    resp = client.get("/api/v1/sources")
    assert resp.status_code == 503
    assert resp.json()["detail"]["error"]["code"] == "service_unavailable"


# ===========================================================================
# GET /api/v1/collections
# ===========================================================================


def test_list_collections_happy(client, monkeypatch):
    """Rows map to CollectionItem entries."""
    rows = [
        {"collection_name": "DocsA", "chunk_count": 12},
        {"collection_name": "DocsB", "chunk_count": 3},
    ]
    _set(monkeypatch, "vector_db.list_collections", lambda *_a, **_k: list(rows))

    resp = client.get("/api/v1/collections")
    assert resp.status_code == 200
    cols = resp.json()["collections"]
    assert {c["collection_name"]: c["chunk_count"] for c in cols} == {
        "DocsA": 12,
        "DocsB": 3,
    }


def test_list_collections_503(client, monkeypatch):
    """list_collections raising maps to 503."""
    _set(monkeypatch, "vector_db.list_collections", _raise)

    resp = client.get("/api/v1/collections")
    assert resp.status_code == 503
    assert resp.json()["detail"]["error"]["code"] == "service_unavailable"


# ===========================================================================
# GET /api/v1/collections/{collection_name}/stats
# ===========================================================================


def test_collection_stats_happy(client, monkeypatch):
    """Stats fields are mapped onto the response with the path collection name."""
    stats = {
        "document_count": 4,
        "chunk_count": 40,
        "connector_breakdown": {"minio": 30, "s3": 10},
    }
    _set(monkeypatch, "vector_db.get_collection_stats", lambda *_a, **_k: dict(stats))

    resp = client.get("/api/v1/collections/DocsA/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["collection_name"] == "DocsA"
    assert body["document_count"] == 4
    assert body["chunk_count"] == 40
    assert body["connector_breakdown"] == {"minio": 30, "s3": 10}


def test_collection_stats_404(client, monkeypatch):
    """get_collection_stats returning None maps to 404 not_found."""
    _set(monkeypatch, "vector_db.get_collection_stats", lambda *_a, **_k: None)

    resp = client.get("/api/v1/collections/Missing/stats")
    assert resp.status_code == 404
    assert resp.json()["detail"]["error"]["code"] == "not_found"


def test_collection_stats_503(client, monkeypatch):
    """get_collection_stats raising maps to 503."""
    _set(monkeypatch, "vector_db.get_collection_stats", _raise)

    resp = client.get("/api/v1/collections/DocsA/stats")
    assert resp.status_code == 503
    assert resp.json()["detail"]["error"]["code"] == "service_unavailable"
