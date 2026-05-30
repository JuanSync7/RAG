# @summary
# P0 slice — collection selection plumbing tests.
# Validates: collection_exists helper, RAGChain.__init__ collection_name kwarg,
# ingest CLI --collection flag, and live-Weaviate write isolation between two
# collections selected programmatically.
# Deps: pytest, src.vector_db, src.retrieval.pipeline.rag_chain, src.ingest.cli
# @end-summary
"""Tests for the P0 collection-selection slice.

Four tests, two flavours:
  - test_isolation_writes_do_not_leak and test_collection_exists_helper carry
    pytestmark=[slow, integration] because they exercise a live Weaviate.
  - test_rag_chain_accepts_collection_name and test_ingest_cli_accepts_collection_flag
    are unit-level and run in the default sweep.
"""

from __future__ import annotations

# Force the real ``weaviate`` package into sys.modules BEFORE tests/conftest.py
# installs its stub module. Without this the live-Weaviate tests below would
# end up holding a stubbed client and silently lose isolation guarantees.
try:  # pragma: no cover - import-side-effect only
    import src.vector_db.weaviate.store  # noqa: F401
    import weaviate  # noqa: F401
except Exception:  # pragma: no cover - infra-dependent fallback
    pass

import os
import uuid
from typing import Any
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Unit-level tests (default sweep)
# ---------------------------------------------------------------------------

def test_rag_chain_accepts_collection_name() -> None:
    """RAGChain(collection_name=...) resolves the requested collection.

    Observes the chosen value through ``self._collection_name`` — the same
    attribute the chain internals will read when threading the value through
    ensure_collection / search / expansion call sites.
    """
    custom = f"custom_foo_{uuid.uuid4().hex[:8]}"

    with patch("src.retrieval.pipeline.rag_chain.create_persistent_client"), \
         patch("src.retrieval.pipeline.rag_chain.ensure_collection"), \
         patch("src.retrieval.pipeline.rag_chain.get_embedding_provider"), \
         patch("src.retrieval.pipeline.rag_chain.get_reranker_provider"), \
         patch("src.retrieval.pipeline.rag_chain.OllamaGenerator"), \
         patch("src.retrieval.pipeline.rag_chain._get_kg_client"), \
         patch("src.retrieval.pipeline.rag_chain.KG_ENABLED", False):
        from src.retrieval.pipeline.rag_chain import RAGChain  # noqa: PLC0415

        chain = RAGChain(persistent_weaviate=False, collection_name=custom)

    assert chain._collection_name == custom


def test_rag_chain_default_collection_name_unchanged() -> None:
    """Default behaviour: no collection_name kwarg falls back to VECTOR_COLLECTION_DEFAULT.

    Regression guard — the slice must not change default-path behaviour.
    """
    from config.settings import VECTOR_COLLECTION_DEFAULT  # noqa: PLC0415

    with patch("src.retrieval.pipeline.rag_chain.create_persistent_client"), \
         patch("src.retrieval.pipeline.rag_chain.ensure_collection"), \
         patch("src.retrieval.pipeline.rag_chain.get_embedding_provider"), \
         patch("src.retrieval.pipeline.rag_chain.get_reranker_provider"), \
         patch("src.retrieval.pipeline.rag_chain.OllamaGenerator"), \
         patch("src.retrieval.pipeline.rag_chain._get_kg_client"), \
         patch("src.retrieval.pipeline.rag_chain.KG_ENABLED", False):
        from src.retrieval.pipeline.rag_chain import RAGChain  # noqa: PLC0415

        chain = RAGChain(persistent_weaviate=False)

    assert chain._collection_name == VECTOR_COLLECTION_DEFAULT


def test_ingest_cli_accepts_collection_flag() -> None:
    """CLI parser accepts --collection custom_foo and surfaces it as args.collection."""
    from src.ingest.cli import _build_parser  # noqa: PLC0415

    parser = _build_parser()
    args = parser.parse_args(
        ["--file", "/tmp/dummy.pdf", "--collection", "custom_foo"]
    )
    assert args.collection == "custom_foo"


def test_ingest_cli_collection_flag_short_form() -> None:
    """-c is an alias for --collection."""
    from src.ingest.cli import _build_parser  # noqa: PLC0415

    parser = _build_parser()
    args = parser.parse_args(["--file", "/tmp/dummy.pdf", "-c", "bar_x"])
    assert args.collection == "bar_x"


# ---------------------------------------------------------------------------
# Live-Weaviate tests (slow + integration)
# ---------------------------------------------------------------------------

# Dual-marker per project memory feedback_dual_marker_gating: both slow AND
# integration so `-m "not slow and not integration"` reliably excludes them.
_live_pytestmark = [pytest.mark.slow, pytest.mark.integration]


@pytest.fixture
def _live_weaviate_client() -> Any:
    """Yield a real persistent Weaviate client.

    The root tests/conftest.py installs an in-memory ``weaviate`` stub when
    weaviate is not already in sys.modules. For live-Weaviate tests we must
    swap the stub out for the real package and reload the vector_db modules
    that captured the stub at import time. Skips if the live instance is
    unreachable so the test doesn't pretend-pass when infra is down.
    """
    import sys  # noqa: PLC0415
    import importlib  # noqa: PLC0415

    # Drop the conftest stub (and any sub-stubs) and re-import the real one.
    for mod_name in list(sys.modules):
        if mod_name == "weaviate" or mod_name.startswith("weaviate."):
            del sys.modules[mod_name]
    try:
        import weaviate  # noqa: F401, PLC0415
    except Exception as exc:  # pragma: no cover - infra-dependent
        pytest.skip(f"Real weaviate package not importable: {exc}")

    # Reload the store and backend modules so they bind to the real
    # weaviate.classes.{config,query} symbols rather than the stubs.
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
                # Reload failures here are non-fatal: if the module isn't
                # safely reloadable we fall through to the original binding.
                pass

    from src.vector_db import create_persistent_client, close_client  # noqa: PLC0415

    try:
        client = create_persistent_client()
    except Exception as exc:  # pragma: no cover - infra-dependent
        pytest.skip(f"Live Weaviate not reachable: {exc}")

    try:
        yield client
    finally:
        try:
            close_client(client)
        except Exception:
            pass


@pytest.mark.slow
@pytest.mark.integration
def test_collection_exists_helper(_live_weaviate_client) -> None:
    """collection_exists(client, name) returns True for live coll, False otherwise."""
    from src.vector_db import (  # noqa: PLC0415
        collection_exists,
        delete_collection,
        ensure_collection,
    )

    client = _live_weaviate_client
    name = f"ragweave_test_exists_{uuid.uuid4().hex[:10]}"
    absent = f"ragweave_test_absent_{uuid.uuid4().hex[:10]}"

    try:
        ensure_collection(client, name)
        assert collection_exists(client, name) is True
        assert collection_exists(client, absent) is False
    finally:
        try:
            delete_collection(client, name)
        except Exception:
            pass


@pytest.mark.slow
@pytest.mark.integration
def test_isolation_writes_do_not_leak(_live_weaviate_client) -> None:
    """Writes to collection A do not appear in collection B (full isolation).

    Uses the real Weaviate add_documents write path through the public
    vector_db API — no mocking of ensure_collection or add_documents.
    """
    import time as _time  # noqa: PLC0415

    from src.vector_db import (  # noqa: PLC0415
        DocumentRecord,
        add_documents,
        delete_collection,
        ensure_collection,
    )

    def _count(client_: Any, coll: str) -> int:
        """Count objects in a collection directly via the low-level client.

        Uses aggregate.over_all(total_count=True) — the same primitive
        ``get_collection_stats`` is built on, but skipped because we don't
        need the per-source/per-connector group-bys for this assertion.
        """
        return int(
            client_.collections.get(coll).aggregate.over_all(
                total_count=True
            ).total_count or 0
        )

    suffix = uuid.uuid4().hex[:10]
    coll_a = f"ragweave_test_a_{suffix}"
    coll_b = f"ragweave_test_b_{suffix}"
    client = _live_weaviate_client

    try:
        ensure_collection(client, coll_a)
        ensure_collection(client, coll_b)

        # Real write path. 1024-dim zero vector matches the schema's vector dim
        # config (vectorizer=none expects caller-provided vectors of arbitrary
        # length; the schema does not enforce dim at insert time).
        record = DocumentRecord(
            text="hello collection A",
            embedding=[0.0] * 1024,
            metadata={
                "source": "test://isolation",
                "source_key": f"isolation-{suffix}",
                "chunk_index": 0,
            },
        )

        n_added = add_documents(client, [record], collection=coll_a)
        assert n_added == 1, f"add_documents returned {n_added}, expected 1"

        # Weaviate aggregate is eventually consistent — give the indexer up to
        # ~5s to catch up before asserting (real production loads do not hit
        # this; the loop is conservative for CI timing variance only).
        count_a = 0
        for _ in range(10):
            count_a = _count(client, coll_a)
            if count_a > 0:
                break
            _time.sleep(0.5)
        count_b = _count(client, coll_b)

        assert count_a > 0, f"Collection A should have chunks; got {count_a}"
        assert count_b == 0, f"Collection B should be empty; got {count_b}"
    finally:
        for name in (coll_a, coll_b):
            try:
                delete_collection(client, name)
            except Exception:
                pass
