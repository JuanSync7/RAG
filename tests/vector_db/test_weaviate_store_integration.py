# @summary
# Slice 14 — real-DB integration tests for the Weaviate vector store.
# Exercises the public src.vector_db facade end-to-end against the LIVE
# rag-weaviate container: ensure_collection -> add_documents (synthetic 1024-d
# vectors) -> hybrid search (+ source_key filter) -> stats -> delete, plus
# write isolation between two collections and delete_by_source_key scoping.
# Deps: pytest, src.vector_db, weaviate (real package), config.settings
# @end-summary
"""Live-Weaviate integration tests for the vector store round-trip.

All tests carry ``pytestmark = [slow, integration]`` so the offline CI gate
(``-m "not slow and not integration"``) deselects them. They are meant to RUN
against the shared ``rag-weaviate`` instance only when the live markers are
selected.

Shared-instance safety:
  - Every collection is named ``ITEST_<uuid>`` and is created by the test.
  - A fixture guarantees teardown (delete_collection) even on failure.
  - No pre-existing collection (e.g. the default ``RAGDocuments``) is touched.

Why synthetic vectors: ``add_documents`` takes PRE-COMPUTED embeddings, so we
hand-build deterministic 1024-dim (BGE-M3 dim) vectors. The "match" doc shares
the query vector; the "non-match" doc is orthogonal — making hybrid ranking
assertable without any embedding service.
"""

from __future__ import annotations

# Force the REAL ``weaviate`` package into sys.modules BEFORE tests/conftest.py
# installs its in-memory stub. Without this the live tests below would hold a
# stubbed client and silently lose their round-trip / isolation guarantees.
try:  # pragma: no cover - import-side-effect only
    import src.vector_db.weaviate.store  # noqa: F401
    import weaviate  # noqa: F401
except Exception:  # pragma: no cover - infra-dependent fallback
    pass

import uuid
from typing import Any, Iterator

import pytest


# Dual-marker per project memory (feedback_dual_marker_gating): both slow AND
# integration so ``-m "not slow and not integration"`` reliably excludes them.
pytestmark = [pytest.mark.slow, pytest.mark.integration]

# BGE-M3 embedding dimensionality used throughout the project.
_DIM = 1024


# ---------------------------------------------------------------------------
# Synthetic-vector helpers
# ---------------------------------------------------------------------------

def _basis_vector(index: int, scale: float = 1.0) -> list[float]:
    """Return a 1024-d one-hot-ish vector with ``scale`` at ``index``.

    Distinct ``index`` values are mutually orthogonal, so cosine similarity
    between two different basis vectors is 0 — this makes the "should match"
    vs "should not match" ranking deterministic.
    """
    vec = [0.0] * _DIM
    vec[index % _DIM] = scale
    return vec


# ---------------------------------------------------------------------------
# Live client + collection fixtures
# ---------------------------------------------------------------------------

def _real_weaviate_or_skip() -> Any:
    """Swap the conftest stub for the real weaviate package and reload modules.

    Mirrors the approach in test_collection_selection.py. Skips (rather than
    fails) when the real package or the live instance is unavailable so the
    suite does not pretend-pass with infra down.
    """
    import importlib  # noqa: PLC0415
    import sys  # noqa: PLC0415

    for mod_name in list(sys.modules):
        if mod_name == "weaviate" or mod_name.startswith("weaviate."):
            del sys.modules[mod_name]
    try:
        import weaviate  # noqa: F401, PLC0415
    except Exception as exc:  # pragma: no cover - infra-dependent
        pytest.skip(f"Real weaviate package not importable: {exc}")

    for mod_name in [
        "src.vector_db.weaviate.store",
        "src.vector_db.weaviate.backend",
        "src.vector_db.weaviate.visual_store",
        "src.vector_db.weaviate",
        "src.vector_db",
    ]:
        if mod_name in sys.modules:
            try:
                importlib.reload(sys.modules[mod_name])
            except Exception:
                pass


@pytest.fixture
def live_client() -> Iterator[Any]:
    """Yield a real persistent Weaviate client; skip if the instance is down.

    Requires ``RAG_WEAVIATE_MODE=networked`` in the environment so the store
    opens a networked client against the shared container rather than spawning
    an embedded one.
    """
    _real_weaviate_or_skip()

    from src.vector_db import close_client, create_persistent_client  # noqa: PLC0415

    try:
        client = create_persistent_client()
    except Exception as exc:  # pragma: no cover - infra-dependent
        pytest.skip(f"Live Weaviate not reachable: {exc}")

    if not getattr(client, "is_ready", lambda: True)():  # pragma: no cover
        try:
            close_client(client)
        finally:
            pytest.skip("Live Weaviate not ready")

    try:
        yield client
    finally:
        try:
            close_client(client)
        except Exception:
            pass


@pytest.fixture
def itest_collection(live_client: Any) -> Iterator[str]:
    """Provide a unique ITEST_* collection name with guaranteed teardown.

    The collection is NOT created here — the round-trip test asserts on
    creation itself — but it is ALWAYS deleted in the finalizer so nothing
    leaks into the shared instance even if the test body raises.
    """
    from src.vector_db import delete_collection  # noqa: PLC0415

    name = f"ITEST_{uuid.uuid4().hex}"
    try:
        yield name
    finally:
        try:
            delete_collection(live_client, name)
        except Exception:
            pass


def _count(client: Any, collection: str) -> int:
    """Direct object count via the low-level aggregate primitive."""
    return int(
        client.collections.get(collection)
        .aggregate.over_all(total_count=True)
        .total_count
        or 0
    )


def _wait_indexed(client: Any, collection: str, expected: int, tries: int = 20) -> int:
    """Poll until the collection reports ``expected`` objects (eventual consistency)."""
    import time as _time  # noqa: PLC0415

    count = 0
    for _ in range(tries):
        count = _count(client, collection)
        if count >= expected:
            return count
        _time.sleep(0.5)
    return count


# ---------------------------------------------------------------------------
# Round-trip: create -> exists -> add -> stats -> search -> delete -> absent
# ---------------------------------------------------------------------------

def test_round_trip_create_add_search_delete(
    live_client: Any, itest_collection: str
) -> None:
    """Full lifecycle against live Weaviate through the public facade.

    Asserts, in order:
      * ensure_collection makes collection_exists flip False -> True;
      * add_documents reports the exact count inserted;
      * get_collection_stats.chunk_count matches once indexed;
      * hybrid search with the matching query vector ranks doc A first and
        returns real-shaped SearchResult fields (text/score/metadata);
      * delete_collection flips collection_exists back to False.
    """
    from src.vector_db import (  # noqa: PLC0415
        DocumentRecord,
        add_documents,
        collection_exists,
        delete_collection,
        ensure_collection,
        get_collection_stats,
        search,
    )

    client = live_client
    coll = itest_collection

    # --- create ---
    assert collection_exists(client, coll) is False
    ensure_collection(client, coll)
    assert collection_exists(client, coll) is True

    # --- add (3 docs; doc A shares the query vector, B/C are orthogonal) ---
    query_vec = _basis_vector(0, scale=1.0)
    docs = [
        DocumentRecord(
            text="alpha bravo charlie target document about widgets",
            embedding=_basis_vector(0, scale=1.0),  # == query_vec direction
            metadata={
                "source": "itest://doc-a",
                "source_key": "itest-key-a",
                "chunk_index": 0,
            },
        ),
        DocumentRecord(
            text="delta echo foxtrot unrelated content about gizmos",
            embedding=_basis_vector(1, scale=1.0),  # orthogonal to query_vec
            metadata={
                "source": "itest://doc-b",
                "source_key": "itest-key-b",
                "chunk_index": 1,
            },
        ),
        DocumentRecord(
            text="golf hotel india more unrelated content about gadgets",
            embedding=_basis_vector(2, scale=1.0),  # orthogonal to query_vec
            metadata={
                "source": "itest://doc-c",
                "source_key": "itest-key-c",
                "chunk_index": 2,
            },
        ),
    ]
    n_added = add_documents(client, docs, collection=coll)
    assert n_added == 3, f"add_documents returned {n_added}, expected 3"

    # --- stats ---
    indexed = _wait_indexed(client, coll, expected=3)
    assert indexed == 3, f"expected 3 indexed objects, saw {indexed}"
    stats = get_collection_stats(client, coll)
    assert stats is not None
    assert stats["chunk_count"] == 3, f"stats chunk_count={stats['chunk_count']}"
    # 3 distinct source_key values -> 3 documents.
    assert stats["document_count"] == 3

    # --- search: pure-vector (alpha=1.0) so the shared-vector doc A wins ---
    results = search(
        client,
        query="alpha bravo charlie target",
        query_embedding=query_vec,
        alpha=1.0,
        limit=3,
        filters=None,
        collection=coll,
    )
    assert len(results) == 3, f"expected 3 results, got {len(results)}"
    top = results[0]
    # Real-shaped SearchResult fields.
    assert top.text == "alpha bravo charlie target document about widgets"
    assert top.metadata["source_key"] == "itest-key-a"
    assert top.metadata["source"] == "itest://doc-a"
    assert isinstance(top.score, float)
    assert top.collection == coll
    # Doc A (vector-identical to the query) must outrank the orthogonal docs.
    assert top.score >= results[1].score
    assert top.score >= results[2].score

    # --- delete ---
    delete_collection(client, coll)
    assert collection_exists(client, coll) is False


# ---------------------------------------------------------------------------
# Filter: source_key SearchFilter restricts the result set
# ---------------------------------------------------------------------------

def test_search_filter_by_source_key(
    live_client: Any, itest_collection: str
) -> None:
    """A source_key ``eq`` SearchFilter returns only that source's docs.

    Two docs share source_key ``keep`` and one has ``drop``; an identical
    vector is given to all three so vector similarity cannot do the filtering
    on its own — the filter is the only thing that can isolate the ``keep``
    set, giving the assertion teeth.
    """
    from src.vector_db import (  # noqa: PLC0415
        DocumentRecord,
        SearchFilter,
        add_documents,
        ensure_collection,
        search,
    )

    client = live_client
    coll = itest_collection
    ensure_collection(client, coll)

    shared_vec = _basis_vector(5, scale=1.0)
    docs = [
        DocumentRecord(
            text="keep one", embedding=shared_vec,
            metadata={"source": "itest://k1", "source_key": "keep", "chunk_index": 0},
        ),
        DocumentRecord(
            text="keep two", embedding=shared_vec,
            metadata={"source": "itest://k2", "source_key": "keep", "chunk_index": 1},
        ),
        DocumentRecord(
            text="drop one", embedding=shared_vec,
            metadata={"source": "itest://d1", "source_key": "drop", "chunk_index": 2},
        ),
    ]
    assert add_documents(client, docs, collection=coll) == 3
    assert _wait_indexed(client, coll, expected=3) == 3

    results = search(
        client,
        query="keep",
        query_embedding=shared_vec,
        alpha=1.0,
        limit=10,
        filters=[SearchFilter(property="source_key", operator="eq", value="keep")],
        collection=coll,
    )
    assert len(results) == 2, f"filter should keep exactly 2, got {len(results)}"
    assert {r.metadata["source_key"] for r in results} == {"keep"}
    assert {r.text for r in results} == {"keep one", "keep two"}


# ---------------------------------------------------------------------------
# Isolation: a search in collection A never returns collection B's docs
# ---------------------------------------------------------------------------

def test_write_isolation_between_collections(live_client: Any) -> None:
    """Writes to collection A do not leak into collection B (and vice versa).

    Both collections get a doc with the SAME vector and the SAME query text,
    so the only thing keeping them apart is collection scoping. A search of A
    must return A's doc and never B's, proving real write isolation.
    """
    from src.vector_db import (  # noqa: PLC0415
        DocumentRecord,
        add_documents,
        delete_collection,
        ensure_collection,
        search,
    )

    client = live_client
    suffix = uuid.uuid4().hex
    coll_a = f"ITEST_{suffix}_a"
    coll_b = f"ITEST_{suffix}_b"
    shared_vec = _basis_vector(7, scale=1.0)

    try:
        ensure_collection(client, coll_a)
        ensure_collection(client, coll_b)

        add_documents(
            client,
            [DocumentRecord(
                text="only in A",
                embedding=shared_vec,
                metadata={"source": "itest://a", "source_key": "in-a", "chunk_index": 0},
            )],
            collection=coll_a,
        )
        add_documents(
            client,
            [DocumentRecord(
                text="only in B",
                embedding=shared_vec,
                metadata={"source": "itest://b", "source_key": "in-b", "chunk_index": 0},
            )],
            collection=coll_b,
        )
        assert _wait_indexed(client, coll_a, expected=1) == 1
        assert _wait_indexed(client, coll_b, expected=1) == 1

        res_a = search(
            client, query="only", query_embedding=shared_vec,
            alpha=1.0, limit=10, filters=None, collection=coll_a,
        )
        assert len(res_a) == 1
        assert res_a[0].text == "only in A"
        assert res_a[0].metadata["source_key"] == "in-a"
        # The defining assertion: B's content never surfaces in an A search.
        assert all(r.metadata["source_key"] != "in-b" for r in res_a)
        assert all(r.text != "only in B" for r in res_a)
    finally:
        for name in (coll_a, coll_b):
            try:
                delete_collection(client, name)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# delete_by_source_key removes only the targeted source's docs
# ---------------------------------------------------------------------------

def test_delete_by_source_key_is_scoped(
    live_client: Any, itest_collection: str
) -> None:
    """delete_by_source_key removes only the matching source_key's chunks."""
    from src.vector_db import (  # noqa: PLC0415
        DocumentRecord,
        add_documents,
        delete_by_source_key,
        ensure_collection,
        get_collection_stats,
    )

    client = live_client
    coll = itest_collection
    ensure_collection(client, coll)

    docs = [
        DocumentRecord(
            text="del one", embedding=_basis_vector(3),
            metadata={"source": "itest://del", "source_key": "to-delete", "chunk_index": 0},
        ),
        DocumentRecord(
            text="del two", embedding=_basis_vector(4),
            metadata={"source": "itest://del", "source_key": "to-delete", "chunk_index": 1},
        ),
        DocumentRecord(
            text="keep me", embedding=_basis_vector(6),
            metadata={"source": "itest://survivor", "source_key": "survivor", "chunk_index": 2},
        ),
    ]
    assert add_documents(client, docs, collection=coll) == 3
    assert _wait_indexed(client, coll, expected=3) == 3

    deleted = delete_by_source_key(client, "to-delete", collection=coll)
    assert deleted == 2, f"expected 2 deleted, got {deleted}"

    # Survivor remains.
    remaining = _wait_indexed(client, coll, expected=1)
    assert remaining == 1
    stats = get_collection_stats(client, coll)
    assert stats is not None and stats["chunk_count"] == 1
