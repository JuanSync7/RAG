"""Unit tests for SearchFilter -> Weaviate Filter translation.

Focuses on operator dispatch in WeaviateBackend._single_filter, especially the
``not_in`` operator added for the conversation-scoped ignore-list. The tests
inject a recording stand-in for ``weaviate.classes.query.Filter`` so the
generated filter structure can be inspected without a live Weaviate client.
"""

from __future__ import annotations

import sys
import types

import pytest

from src.vector_db.common.schemas import SearchFilter
from src.vector_db.weaviate.backend import WeaviateBackend


class _Clause:
    """Captures one Filter.by_property(...).<op>(value) call."""

    def __init__(self, prop, op, value):
        self.prop = prop
        self.op = op
        self.value = value
        self.children: list = []  # populated when AND-combined

    def __and__(self, other):
        combined = _Clause(prop=None, op="and", value=None)
        combined.children = [self, other]
        return combined


class _PropertyFilter:
    def __init__(self, prop):
        self._prop = prop

    def equal(self, value):
        return _Clause(self._prop, "equal", value)

    def not_equal(self, value):
        return _Clause(self._prop, "not_equal", value)

    def like(self, value):
        return _Clause(self._prop, "like", value)

    def greater_than(self, value):
        return _Clause(self._prop, "greater_than", value)

    def less_than(self, value):
        return _Clause(self._prop, "less_than", value)

    def greater_or_equal(self, value):
        return _Clause(self._prop, "greater_or_equal", value)

    def less_or_equal(self, value):
        return _Clause(self._prop, "less_or_equal", value)


class _FilterFactory:
    @staticmethod
    def by_property(prop):
        return _PropertyFilter(prop)


@pytest.fixture(autouse=True)
def _install_recording_filter_stub(monkeypatch):
    """Replace conftest's minimal Filter stub with a richer recording one."""
    query_mod = types.ModuleType("weaviate.classes.query")
    query_mod.Filter = _FilterFactory
    monkeypatch.setitem(sys.modules, "weaviate.classes.query", query_mod)
    yield


def test_eq_operator_dispatches_to_equal():
    clause = WeaviateBackend._single_filter(SearchFilter("source", "eq", "doc.md"))
    assert clause.op == "equal"
    assert clause.prop == "source"
    assert clause.value == "doc.md"


def test_ne_operator_dispatches_to_not_equal():
    clause = WeaviateBackend._single_filter(SearchFilter("source", "ne", "doc.md"))
    assert clause.op == "not_equal"


def test_unsupported_operator_raises_with_helpful_message():
    with pytest.raises(ValueError) as excinfo:
        WeaviateBackend._single_filter(SearchFilter("x", "bogus", 1))
    assert "not_in" in str(excinfo.value)


def test_not_in_with_empty_list_returns_none():
    result = WeaviateBackend._single_filter(
        SearchFilter("document_id", "not_in", [])
    )
    assert result is None


def test_not_in_with_single_value_emits_one_not_equal():
    clause = WeaviateBackend._single_filter(
        SearchFilter("document_id", "not_in", ["a"])
    )
    assert clause.op == "not_equal"
    assert clause.prop == "document_id"
    assert clause.value == "a"


def test_not_in_with_multiple_values_chains_with_and():
    clause = WeaviateBackend._single_filter(
        SearchFilter("document_id", "not_in", ["a", "b", "c"])
    )
    # Top-level should be an AND combiner.
    assert clause.op == "and"
    # Walk the AND tree, collect leaf not_equal clauses.
    collected: list[_Clause] = []

    def _walk(node):
        if node.op == "and":
            for child in node.children:
                _walk(child)
        else:
            collected.append(node)

    _walk(clause)
    assert [c.op for c in collected] == ["not_equal", "not_equal", "not_equal"]
    assert [c.value for c in collected] == ["a", "b", "c"]
    assert all(c.prop == "document_id" for c in collected)


def test_translate_filters_drops_none_clauses():
    """A SearchFilter that translates to None (e.g. empty not_in) should not
    poison the AND chain — _translate_filters must drop it before combining."""
    backend = object.__new__(WeaviateBackend)  # bypass __init__
    result = WeaviateBackend._translate_filters(
        backend,
        [
            SearchFilter("source", "eq", "doc.md"),
            SearchFilter("document_id", "not_in", []),
        ],
    )
    # Only the eq clause survives; result should be a single _Clause.
    assert isinstance(result, _Clause)
    assert result.op == "equal"
    assert result.value == "doc.md"


def test_translate_filters_returns_none_when_all_clauses_drop():
    backend = object.__new__(WeaviateBackend)
    result = WeaviateBackend._translate_filters(
        backend,
        [SearchFilter("document_id", "not_in", [])],
    )
    assert result is None
