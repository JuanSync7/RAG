"""Unit tests for the ``in`` (``contains_any``) SearchFilter operator.

Slice S0a of the RAPTOR-lite document-routing feature. The retrieval path needs
to restrict a single hybrid search to a *set* of ``document_id`` values (the
docs chosen by the card-index router), so the vector-store filter abstraction
gains an ``in`` operator that translates to Weaviate's ``contains_any``.

``WeaviateBackend._single_filter`` is a ``@staticmethod``, so these tests call
it directly without instantiating the backend or touching a live client. The
method does a *function-local* ``from weaviate.classes.query import Filter``.
In the shared test env that module may be a conftest-installed stub (its content
depends on collection/import order), so — mirroring ``test_backend_contract.py``
— these tests inject a *recording* fake ``Filter`` into ``weaviate.classes.query``
via the ``fake_weaviate_filter`` fixture. The fake records each
``by_property(...).<op>(value)`` call and each ``&`` combination as a small
immutable node tree, letting us assert the exact target/operator/value mapping
the backend builds, independent of whichever real/stub weaviate is loaded.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import pytest

import src.vector_db.weaviate.backend as wb
from src.vector_db.common import SearchFilter


# --------------------------------------------------------------------------
# Recording fake Filter (mirrors the structure used in test_backend_contract)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Leaf:
    """A single ``by_property(target).<op>(value)`` clause."""

    target: str
    op: str
    value: Any

    def __and__(self, other: Any) -> "_And":
        return _And((self, other))


@dataclass(frozen=True)
class _And:
    """An AND-combination of clauses produced by ``&``."""

    clauses: tuple = field(default=())

    def __and__(self, other: Any) -> "_And":
        return _And((*self.clauses, other))


class _PropBuilder:
    """Result of ``Filter.by_property(target)`` — records the op + value."""

    def __init__(self, target: str) -> None:
        self._target = target

    def equal(self, value: Any) -> _Leaf:
        return _Leaf(self._target, "equal", value)

    def not_equal(self, value: Any) -> _Leaf:
        return _Leaf(self._target, "not_equal", value)

    def like(self, value: Any) -> _Leaf:
        return _Leaf(self._target, "like", value)

    def greater_than(self, value: Any) -> _Leaf:
        return _Leaf(self._target, "greater_than", value)

    def less_than(self, value: Any) -> _Leaf:
        return _Leaf(self._target, "less_than", value)

    def greater_or_equal(self, value: Any) -> _Leaf:
        return _Leaf(self._target, "greater_or_equal", value)

    def less_or_equal(self, value: Any) -> _Leaf:
        return _Leaf(self._target, "less_or_equal", value)

    def contains_any(self, value: Any) -> _Leaf:
        return _Leaf(self._target, "contains_any", value)


class _FakeFilter:
    """Stand-in for ``weaviate.classes.query.Filter``."""

    @staticmethod
    def by_property(target: str) -> _PropBuilder:
        return _PropBuilder(target)


@pytest.fixture
def fake_weaviate_filter(monkeypatch: pytest.MonkeyPatch) -> type[_FakeFilter]:
    """Inject the recording fake ``Filter`` into ``weaviate.classes.query``.

    ``_single_filter`` imports ``Filter`` lazily from that module, so patching
    the attribute there is enough; no need to reload the backend module. We
    resolve the module the same way ``test_backend_contract.py`` does — via
    ``sys.modules`` — which is populated either by the real weaviate package or
    by the conftest stub, regardless of import/collection order. If neither has
    registered it yet (e.g. this file collected in isolation with the real
    weaviate loaded but its ``classes`` subpackage untouched), import it.
    """

    mod = sys.modules.get("weaviate.classes.query")
    if mod is None:
        import weaviate.classes.query as mod  # noqa: F811  (resolve + register)

    monkeypatch.setattr(mod, "Filter", _FakeFilter, raising=False)
    return _FakeFilter


# --------------------------------------------------------------------------
# `in` operator translation
# --------------------------------------------------------------------------


def test_in_operator_translates_to_contains_any(fake_weaviate_filter) -> None:
    """``in`` with a non-empty list → a single ``contains_any`` clause."""

    f = SearchFilter("document_id", "in", ["a", "b"])

    result = wb.WeaviateBackend._single_filter(f)

    assert result is not None
    assert isinstance(result, _Leaf)
    assert result.target == "document_id"
    assert result.op == "contains_any"
    assert result.value == ["a", "b"]


def test_in_operator_is_case_insensitive(fake_weaviate_filter) -> None:
    """Operator matching lower-cases the operator (mirrors ``not_in``/eq)."""

    f = SearchFilter("document_id", "IN", ["x"])

    result = wb.WeaviateBackend._single_filter(f)

    assert isinstance(result, _Leaf)
    assert result.op == "contains_any"
    assert result.value == ["x"]


def test_in_operator_coerces_value_to_list(fake_weaviate_filter) -> None:
    """The clause value is always a fresh ``list`` (mirrors ``not_in``)."""

    f = SearchFilter("document_id", "in", ("a", "b"))

    result = wb.WeaviateBackend._single_filter(f)

    assert isinstance(result, _Leaf)
    assert result.value == ["a", "b"]
    assert isinstance(result.value, list)


def test_in_operator_empty_list_returns_none(fake_weaviate_filter) -> None:
    """Empty ``in`` set → None (mirrors empty ``not_in``: no-op clause)."""

    f = SearchFilter("document_id", "in", [])

    assert wb.WeaviateBackend._single_filter(f) is None


def test_in_operator_none_value_returns_none(fake_weaviate_filter) -> None:
    """A ``None`` value behaves like an empty set → None."""

    f = SearchFilter("document_id", "in", None)

    assert wb.WeaviateBackend._single_filter(f) is None


# --------------------------------------------------------------------------
# Real weaviate-client reference equivalence (weaviate-client 4.20.4)
# --------------------------------------------------------------------------


def test_in_operator_matches_real_contains_any_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Against the *real* weaviate-client (4.20.4), the produced filter equals a
    reference ``Filter.by_property("document_id").contains_any(["a","b"])``.

    This is the load-bearing assertion of S0a: that ``in`` actually maps onto a
    real ``contains_any`` clause on the installed client. ``_single_filter``
    imports ``Filter`` from ``weaviate.classes.query``, which the root conftest
    may have replaced with a stub (whose ``Filter`` has no ``contains_any``)
    depending on import order. The *real* implementation, however, lives in
    ``weaviate.collections.classes.filters`` and is never stubbed, so we import
    it from there and pin it onto ``weaviate.classes.query.Filter`` for the
    duration of this test — guaranteeing the real translation path runs
    regardless of collection order. If the real client is genuinely absent,
    skip (S0a targets the installed 4.20.4 client).
    """

    try:
        from weaviate.collections.classes.filters import Filter as RealFilter
    except Exception:  # pragma: no cover - real client is installed in this env
        pytest.skip("real weaviate-client Filter not importable")

    if not hasattr(RealFilter.by_property("document_id"), "contains_any"):
        pytest.skip("installed weaviate Filter lacks contains_any")

    # Force `_single_filter` to use the REAL Filter, not a conftest stub.
    mod = sys.modules.get("weaviate.classes.query")
    if mod is None:
        import weaviate.classes.query as mod  # noqa: F811
    monkeypatch.setattr(mod, "Filter", RealFilter, raising=False)

    reference = RealFilter.by_property("document_id").contains_any(["a", "b"])

    f = SearchFilter("document_id", "in", ["a", "b"])
    result = wb.WeaviateBackend._single_filter(f)

    assert result is not None
    # _FilterValue is a pydantic model with stable public attrs + repr.
    assert repr(result) == repr(reference)
    assert getattr(result, "target") == "document_id"
    assert getattr(result, "value") == ["a", "b"]
    assert getattr(result.operator, "value") == "ContainsAny"


# --------------------------------------------------------------------------
# Unknown operator error message must now list `in`
# --------------------------------------------------------------------------


def test_unknown_operator_error_lists_in(fake_weaviate_filter) -> None:
    """An unsupported operator still raises ValueError, and the valid-ops list
    in the message now includes ``in``."""

    f = SearchFilter("document_id", "bogus", "x")

    with pytest.raises(ValueError) as excinfo:
        wb.WeaviateBackend._single_filter(f)

    msg = str(excinfo.value)
    assert "Unsupported filter operator" in msg
    assert "'in'" in msg
    # not_in should still be listed too (no regression).
    assert "'not_in'" in msg


# --------------------------------------------------------------------------
# `not_in` regression — unchanged behaviour
# --------------------------------------------------------------------------


def test_not_in_still_builds_and_chain_of_not_equal(fake_weaviate_filter) -> None:
    """``not_in`` is unchanged: an AND-chain of ``not_equal`` clauses."""

    f = SearchFilter("document_id", "not_in", ["a", "b", "c"])

    result = wb.WeaviateBackend._single_filter(f)

    assert isinstance(result, _And)
    assert all(isinstance(c, _Leaf) for c in result.clauses)
    assert [c.op for c in result.clauses] == ["not_equal", "not_equal", "not_equal"]
    assert [c.value for c in result.clauses] == ["a", "b", "c"]
    assert {c.target for c in result.clauses} == {"document_id"}


def test_not_in_empty_returns_none(fake_weaviate_filter) -> None:
    """``not_in`` with an empty list still returns None (no regression)."""

    f = SearchFilter("document_id", "not_in", [])

    assert wb.WeaviateBackend._single_filter(f) is None
