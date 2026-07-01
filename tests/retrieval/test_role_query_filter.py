# @summary
# Slice-C tests: the gated query-time chunk_role exclusion filter injected at
# the single _do_search choke point (rag_chain.py). A fake search() captures the
# filters it receives; asserts the ne-navigation / ne-boilerplate clauses are
# appended only when RAG_RETRIEVAL_ROLE_FILTER and RAG_RETRIEVAL_ROLE_SCHEMA_PRESENT
# are BOTH true, never otherwise (un-migrated safety), and that caller-supplied
# filters are preserved alongside (additive AND).
# Deps: pytest, src.retrieval.pipeline.rag_chain.RAGChain, SearchFilter
# @end-summary
"""Tests for the Slice-C gated role-exclusion query filter.

``_do_search`` is the single seam every retrieval mode flows through (standard
retrieval, the deep-research ``kb_retrieve`` closure, and the agentic
``retrieve`` closure all delegate to ``self._do_search``). So asserting the
filter injection here proves it for all three paths. These tests construct a
bare ``RAGChain`` and monkeypatch the ``search`` symbol in the rag_chain module
namespace to capture the ``filters`` kwarg without touching Weaviate.
"""
from __future__ import annotations

from collections import OrderedDict

import pytest

import src.retrieval.pipeline.rag_chain as rc
from src.retrieval.pipeline.rag_chain import RAGChain
from src.vector_db.common.schemas import SearchFilter


class _DummySpan:
    def set_attribute(self, *_a, **_kw):
        return None

    def end(self, *_a, **_kw):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _DummyTracer:
    def span(self, *_a, **_kw):
        return _DummySpan()

    def start_span(self, *_a, **_kw):
        return _DummySpan()


def _make_chain() -> RAGChain:
    chain = object.__new__(RAGChain)
    chain.tracer = _DummyTracer()
    chain._persistent_weaviate = True
    # A non-None sentinel routes _do_search through the persistent-client branch
    # (so it calls the patched module-level ``search`` directly, not get_client).
    chain._weaviate_client = object()
    chain._embedding_cache = OrderedDict()
    chain._embedding_cache_max = 128
    return chain


def _resolve_collection(self):  # bound helper for the bare instance
    return "TestCollection"


@pytest.fixture
def captured(monkeypatch):
    """Patch the module-level ``search`` to capture the filters it receives."""
    box: dict = {}

    def _fake_search(*, client, query, query_embedding, alpha, limit, filters, collection):
        box["filters"] = filters
        return []

    monkeypatch.setattr(rc, "search", _fake_search)
    # Give the bare instance a deterministic collection resolver.
    monkeypatch.setattr(RAGChain, "_resolve_collection", _resolve_collection, raising=False)
    return box


def _role_clauses(filters):
    """Extract the chunk_role ``ne`` clauses from a filter list."""
    if not filters:
        return []
    return [
        f for f in filters
        if getattr(f, "property", None) == "chunk_role"
        and getattr(f, "operator", None) == "ne"
    ]


def _excluded_role_values(filters):
    return sorted(f.value for f in _role_clauses(filters))


# ─── (a) flag + schema ON ⇒ ne-navigation AND ne-boilerplate appended ────────


def test_role_filter_injected_when_flag_and_schema_on(monkeypatch, captured):
    monkeypatch.setattr(rc, "RAG_RETRIEVAL_ROLE_FILTER", True)
    monkeypatch.setattr(rc, "RAG_RETRIEVAL_ROLE_SCHEMA_PRESENT", True)
    monkeypatch.setattr(
        rc, "RAG_RETRIEVAL_EXCLUDED_ROLES", ["navigation", "boilerplate"]
    )

    chain = _make_chain()
    chain._do_search("q", [0.1, 0.2], 0.5, 10, None)

    clauses = _role_clauses(captured["filters"])
    assert len(clauses) == 2
    assert _excluded_role_values(captured["filters"]) == ["boilerplate", "navigation"]
    # The contract: ``ne`` (fail-open) on the ``chunk_role`` property.
    for c in clauses:
        assert isinstance(c, SearchFilter)
        assert c.property == "chunk_role"
        assert c.operator == "ne"


def test_role_filter_honours_excluded_roles_config(monkeypatch, captured):
    """The injected clauses come from RAG_RETRIEVAL_EXCLUDED_ROLES, not a literal."""
    monkeypatch.setattr(rc, "RAG_RETRIEVAL_ROLE_FILTER", True)
    monkeypatch.setattr(rc, "RAG_RETRIEVAL_ROLE_SCHEMA_PRESENT", True)
    monkeypatch.setattr(rc, "RAG_RETRIEVAL_EXCLUDED_ROLES", ["navigation"])

    chain = _make_chain()
    chain._do_search("q", [0.1], 0.5, 10, None)

    assert _excluded_role_values(captured["filters"]) == ["navigation"]


# ─── (b) un-migrated safety: schema OFF or filter OFF ⇒ NO role clause ───────


def test_no_role_filter_when_schema_absent(monkeypatch, captured):
    monkeypatch.setattr(rc, "RAG_RETRIEVAL_ROLE_FILTER", True)
    monkeypatch.setattr(rc, "RAG_RETRIEVAL_ROLE_SCHEMA_PRESENT", False)
    monkeypatch.setattr(
        rc, "RAG_RETRIEVAL_EXCLUDED_ROLES", ["navigation", "boilerplate"]
    )

    chain = _make_chain()
    chain._do_search("q", [0.1], 0.5, 10, None)

    assert _role_clauses(captured["filters"]) == []


def test_no_role_filter_when_filter_flag_off(monkeypatch, captured):
    monkeypatch.setattr(rc, "RAG_RETRIEVAL_ROLE_FILTER", False)
    monkeypatch.setattr(rc, "RAG_RETRIEVAL_ROLE_SCHEMA_PRESENT", True)
    monkeypatch.setattr(
        rc, "RAG_RETRIEVAL_EXCLUDED_ROLES", ["navigation", "boilerplate"]
    )

    chain = _make_chain()
    chain._do_search("q", [0.1], 0.5, 10, None)

    assert _role_clauses(captured["filters"]) == []


def test_no_role_filter_when_both_off(monkeypatch, captured):
    monkeypatch.setattr(rc, "RAG_RETRIEVAL_ROLE_FILTER", False)
    monkeypatch.setattr(rc, "RAG_RETRIEVAL_ROLE_SCHEMA_PRESENT", False)

    chain = _make_chain()
    chain._do_search("q", [0.1], 0.5, 10, None)

    # filters passed in were None and nothing was added → still falsy.
    assert not captured["filters"]


def test_empty_excluded_roles_adds_nothing(monkeypatch, captured):
    """Even with the gate fully open, an empty exclusion list adds no clause."""
    monkeypatch.setattr(rc, "RAG_RETRIEVAL_ROLE_FILTER", True)
    monkeypatch.setattr(rc, "RAG_RETRIEVAL_ROLE_SCHEMA_PRESENT", True)
    monkeypatch.setattr(rc, "RAG_RETRIEVAL_EXCLUDED_ROLES", [])

    chain = _make_chain()
    chain._do_search("q", [0.1], 0.5, 10, None)

    assert not captured["filters"]


# ─── (c) caller-supplied filters preserved alongside (additive AND) ──────────


def test_caller_filters_preserved_with_role_filter(monkeypatch, captured):
    monkeypatch.setattr(rc, "RAG_RETRIEVAL_ROLE_FILTER", True)
    monkeypatch.setattr(rc, "RAG_RETRIEVAL_ROLE_SCHEMA_PRESENT", True)
    monkeypatch.setattr(
        rc, "RAG_RETRIEVAL_EXCLUDED_ROLES", ["navigation", "boilerplate"]
    )

    caller_filter = SearchFilter(property="source", operator="eq", value="doc.pdf")
    incoming = [caller_filter]
    chain = _make_chain()
    chain._do_search("q", [0.1], 0.5, 10, incoming)

    out = captured["filters"]
    # Caller's clause survives untouched...
    assert caller_filter in out
    # ...alongside the two appended role clauses (additive AND).
    assert _excluded_role_values(out) == ["boilerplate", "navigation"]
    assert len(out) == 3


def test_caller_filter_list_not_mutated(monkeypatch, captured):
    """The role injection must not mutate the caller's own filter list in place."""
    monkeypatch.setattr(rc, "RAG_RETRIEVAL_ROLE_FILTER", True)
    monkeypatch.setattr(rc, "RAG_RETRIEVAL_ROLE_SCHEMA_PRESENT", True)
    monkeypatch.setattr(
        rc, "RAG_RETRIEVAL_EXCLUDED_ROLES", ["navigation", "boilerplate"]
    )

    caller_filter = SearchFilter(property="source", operator="eq", value="doc.pdf")
    incoming = [caller_filter]
    chain = _make_chain()
    chain._do_search("q", [0.1], 0.5, 10, incoming)

    # The caller's list is unchanged; injection happened on a copy.
    assert incoming == [caller_filter]
    assert len(incoming) == 1


def test_caller_filters_preserved_when_role_filter_off(monkeypatch, captured):
    """With the gate closed, caller filters pass through verbatim (no role clause)."""
    monkeypatch.setattr(rc, "RAG_RETRIEVAL_ROLE_FILTER", True)
    monkeypatch.setattr(rc, "RAG_RETRIEVAL_ROLE_SCHEMA_PRESENT", False)

    caller_filter = SearchFilter(property="tenant_id", operator="eq", value="t1")
    chain = _make_chain()
    chain._do_search("q", [0.1], 0.5, 10, [caller_filter])

    out = captured["filters"]
    assert out == [caller_filter]
    assert _role_clauses(out) == []
