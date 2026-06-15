# @summary
# Unit tests for the C3 gated decomposition orchestrator ``decompose_query``.
# Pins the DOCUMENT_ROUTING_DESIGN.md §6.5 contract: gate on
# RAG_DECOMPOSITION_ENABLED + comparison-intent, then LLM-primary (Tier-2) with
# regex (Tier-1) fallback (order flips when RAG_DECOMPOSITION_LLM_PRIMARY is
# False); a tier "succeeds" only with >=2 non-empty sub-queries; neither tier
# succeeding → identity safe baseline. ``decompose_query`` NEVER raises.
# Every collaborator (settings flags, llm_decompose, regex_decompose,
# detect_comparison_intent) is monkeypatched AS LOOKED UP in the decomposition
# module — no live LLM, no real config.
# Deps: pytest, src.retrieval.routing.decomposition, config.settings
# @end-summary
"""Unit tests for ``src.retrieval.routing.decomposition.decompose_query``.

The orchestrator is the *safe entry point* for query decomposition. It ties the
comparison-intent gate to the two decomposition tiers and degrades to the
identity baseline on any miss/failure. Per §6.5:

* disabled (``RAG_DECOMPOSITION_ENABLED`` False) → identity, no tier called;
* enabled but non-comparison → identity, no tier called;
* enabled + comparison + LLM-primary → ``llm_decompose`` first, ``regex_decompose``
  on ``DecompositionError``;
* enabled + comparison + regex-primary → ``regex_decompose`` first, ``llm_decompose``
  when regex returns ``[]``;
* a tier succeeds only when it yields **>= 2** non-empty sub-queries
  (exactly-one is treated as failure);
* neither tier succeeds → identity ``[query]`` (today's safe baseline);
* the orchestrator NEVER raises.

Monkeypatch strategy: ``decompose_query`` reads the enable/primary flags from
``config.settings`` at CALL time and looks up ``llm_decompose`` /
``regex_decompose`` / ``detect_comparison_intent`` in its own module namespace,
so tests patch those names there (and the flags on ``config.settings``).
"""
from __future__ import annotations

import pytest

import config.settings as settings
from src.retrieval.routing import decomposition
from src.retrieval.routing.decomposition import (
    DecompositionError,
    decompose_query,
)
from src.retrieval.routing.schemas import DecompositionResult


# ─── Spies ──────────────────────────────────────────────────────────────────


class _Spy:
    """Records calls (and order) and returns/raises a canned outcome."""

    def __init__(self, *, returns=None, raises: Exception | None = None, order=None):
        self._returns = returns
        self._raises = raises
        self._order = order  # shared list to record relative call order
        self.calls: list[str] = []

    def __call__(self, query, *args, **kwargs):
        self.calls.append(query)
        if self._order is not None:
            self._order.append(self._name())
        if self._raises is not None:
            raise self._raises
        return self._returns

    def _name(self) -> str:  # overridden per-instance below
        return getattr(self, "label", "spy")

    @property
    def called(self) -> bool:
        return len(self.calls) > 0


def _patch_tiers(
    monkeypatch,
    *,
    llm: _Spy | None = None,
    regex: _Spy | None = None,
    comparison: bool = True,
):
    """Patch the gate + both tiers AS LOOKED UP inside the decomposition module."""
    monkeypatch.setattr(
        decomposition, "detect_comparison_intent", lambda q: comparison
    )
    if llm is not None:
        llm.label = "llm"
        monkeypatch.setattr(decomposition, "llm_decompose", llm)
    if regex is not None:
        regex.label = "regex"
        monkeypatch.setattr(decomposition, "regex_decompose", regex)


def _enable(monkeypatch, *, enabled: bool = True, llm_primary: bool = True):
    monkeypatch.setattr(settings, "RAG_DECOMPOSITION_ENABLED", enabled)
    monkeypatch.setattr(settings, "RAG_DECOMPOSITION_LLM_PRIMARY", llm_primary)


# ─── Gate: disabled → identity, no tier called ──────────────────────────────


def test_disabled_returns_identity_and_calls_no_tier(monkeypatch):
    _enable(monkeypatch, enabled=False)
    llm = _Spy(returns=["a", "b"])
    regex = _Spy(returns=["a", "b"])
    # Even though the query LOOKS like a comparison, disabled short-circuits.
    _patch_tiers(monkeypatch, llm=llm, regex=regex, comparison=True)

    result = decompose_query("AXI4 vs CHI")

    assert isinstance(result, DecompositionResult)
    assert result.subqueries == ["AXI4 vs CHI"]
    assert result.method == "identity"
    assert result.decomposed is False
    assert not llm.called
    assert not regex.called


# ─── Gate: enabled + non-comparison → identity, no tier called ──────────────


def test_non_comparison_returns_identity_and_calls_no_tier(monkeypatch):
    _enable(monkeypatch, enabled=True, llm_primary=True)
    llm = _Spy(returns=["a", "b"])
    regex = _Spy(returns=["a", "b"])
    _patch_tiers(monkeypatch, llm=llm, regex=regex, comparison=False)

    result = decompose_query("explain the MBIST flow")

    assert result.subqueries == ["explain the MBIST flow"]
    assert result.method == "identity"
    assert result.decomposed is False
    assert not llm.called
    assert not regex.called


# ─── LLM-primary happy path ─────────────────────────────────────────────────


def test_llm_primary_success(monkeypatch):
    _enable(monkeypatch, enabled=True, llm_primary=True)
    llm = _Spy(returns=["AXI4 coherency", "CHI coherency"])
    regex = _Spy(returns=["should", "not", "be", "used"])
    _patch_tiers(monkeypatch, llm=llm, regex=regex, comparison=True)

    result = decompose_query("AXI4 vs CHI")

    assert result.subqueries == ["AXI4 coherency", "CHI coherency"]
    assert result.method == "llm"
    assert result.decomposed is True
    assert llm.called
    assert not regex.called  # primary succeeded → fallback untouched


# ─── LLM fails → regex fallback succeeds ────────────────────────────────────


def test_llm_failure_falls_back_to_regex(monkeypatch):
    _enable(monkeypatch, enabled=True, llm_primary=True)
    llm = _Spy(raises=DecompositionError("boom"))
    regex = _Spy(returns=["AXI4", "CHI"])
    _patch_tiers(monkeypatch, llm=llm, regex=regex, comparison=True)

    result = decompose_query("AXI4 vs CHI")

    assert result.subqueries == ["AXI4", "CHI"]
    assert result.method == "regex"
    assert result.decomposed is True
    assert llm.called and regex.called


# ─── LLM fails, regex empty → identity baseline ─────────────────────────────


def test_llm_failure_and_regex_empty_returns_identity(monkeypatch):
    _enable(monkeypatch, enabled=True, llm_primary=True)
    llm = _Spy(raises=DecompositionError("boom"))
    regex = _Spy(returns=[])
    _patch_tiers(monkeypatch, llm=llm, regex=regex, comparison=True)

    result = decompose_query("AXI4 vs CHI")

    assert result.subqueries == ["AXI4 vs CHI"]
    assert result.method == "identity"
    assert result.decomposed is False
    assert llm.called and regex.called


# ─── regex-primary: regex tried FIRST, then llm on empty ────────────────────


def test_regex_primary_order_and_llm_fallback(monkeypatch):
    _enable(monkeypatch, enabled=True, llm_primary=False)
    order: list[str] = []
    regex = _Spy(returns=[], order=order)  # regex misses → try llm
    llm = _Spy(returns=["AXI4 x", "CHI y"], order=order)
    _patch_tiers(monkeypatch, llm=llm, regex=regex, comparison=True)

    result = decompose_query("AXI4 vs CHI")

    # Regex was attempted strictly before the LLM.
    assert order == ["regex", "llm"]
    assert result.subqueries == ["AXI4 x", "CHI y"]
    assert result.method == "llm"
    assert result.decomposed is True


def test_regex_primary_success_skips_llm(monkeypatch):
    _enable(monkeypatch, enabled=True, llm_primary=False)
    order: list[str] = []
    regex = _Spy(returns=["AXI4", "CHI"], order=order)
    llm = _Spy(returns=["nope", "nope"], order=order)
    _patch_tiers(monkeypatch, llm=llm, regex=regex, comparison=True)

    result = decompose_query("AXI4 vs CHI")

    assert order == ["regex"]  # regex succeeded → llm never called
    assert result.subqueries == ["AXI4", "CHI"]
    assert result.method == "regex"
    assert result.decomposed is True
    assert not llm.called


# ─── A tier yielding exactly 1 item is a FAILURE (>= 2 required) ────────────


def test_llm_single_item_is_failure_falls_back_to_regex(monkeypatch):
    _enable(monkeypatch, enabled=True, llm_primary=True)
    llm = _Spy(returns=["only one"])  # 1 item → not a success
    regex = _Spy(returns=["AXI4", "CHI"])
    _patch_tiers(monkeypatch, llm=llm, regex=regex, comparison=True)

    result = decompose_query("AXI4 vs CHI")

    assert result.subqueries == ["AXI4", "CHI"]
    assert result.method == "regex"
    assert result.decomposed is True


def test_regex_single_item_is_failure_returns_identity(monkeypatch):
    # regex-primary: regex yields exactly 1 → treated as miss → try llm; llm
    # also yields 1 → both fail → identity.
    _enable(monkeypatch, enabled=True, llm_primary=False)
    regex = _Spy(returns=["just-one"])
    llm = _Spy(returns=["also-one"])
    _patch_tiers(monkeypatch, llm=llm, regex=regex, comparison=True)

    result = decompose_query("AXI4 vs CHI")

    assert result.subqueries == ["AXI4 vs CHI"]
    assert result.method == "identity"
    assert result.decomposed is False


def test_both_tiers_single_item_llm_primary_returns_identity(monkeypatch):
    _enable(monkeypatch, enabled=True, llm_primary=True)
    llm = _Spy(returns=["one"])
    regex = _Spy(returns=["one"])
    _patch_tiers(monkeypatch, llm=llm, regex=regex, comparison=True)

    result = decompose_query("AXI4 vs CHI")

    assert result.subqueries == ["AXI4 vs CHI"]
    assert result.method == "identity"
    assert result.decomposed is False


# ─── Robustness: a tier with empty-string items is not a success ────────────


def test_tier_with_blank_items_not_counted_as_success(monkeypatch):
    # LLM returns 2 items but they are empty/whitespace → not >=2 NON-EMPTY →
    # fall back to regex.
    _enable(monkeypatch, enabled=True, llm_primary=True)
    llm = _Spy(returns=["", "   "])
    regex = _Spy(returns=["AXI4", "CHI"])
    _patch_tiers(monkeypatch, llm=llm, regex=regex, comparison=True)

    result = decompose_query("AXI4 vs CHI")

    assert result.method == "regex"
    assert result.subqueries == ["AXI4", "CHI"]


# ─── decompose_query NEVER raises ───────────────────────────────────────────


def test_never_raises_when_regex_blows_up(monkeypatch):
    # Even an UNEXPECTED (non-DecompositionError) explosion from a tier must not
    # propagate out of the safe entry point.
    _enable(monkeypatch, enabled=True, llm_primary=True)
    llm = _Spy(raises=DecompositionError("llm down"))
    regex = _Spy(raises=ValueError("regex exploded"))
    _patch_tiers(monkeypatch, llm=llm, regex=regex, comparison=True)

    result = decompose_query("AXI4 vs CHI")

    assert result.subqueries == ["AXI4 vs CHI"]
    assert result.method == "identity"
    assert result.decomposed is False


def test_never_raises_when_llm_raises_unexpected(monkeypatch):
    _enable(monkeypatch, enabled=True, llm_primary=True)
    llm = _Spy(raises=RuntimeError("unexpected"))
    regex = _Spy(returns=["AXI4", "CHI"])
    _patch_tiers(monkeypatch, llm=llm, regex=regex, comparison=True)

    result = decompose_query("AXI4 vs CHI")

    # Unexpected LLM error still degrades to the regex tier.
    assert result.method == "regex"
    assert result.subqueries == ["AXI4", "CHI"]


# ─── Facade export ──────────────────────────────────────────────────────────


def test_facade_exports_decompose_query_and_error():
    from src.retrieval import routing

    assert hasattr(routing, "decompose_query")
    assert hasattr(routing, "DecompositionError")
    assert "decompose_query" in routing.__all__
    assert "DecompositionError" in routing.__all__


def test_default_disabled_identity_via_facade(monkeypatch):
    # Sanity: with routing disabled (the default), the facade entry point returns
    # the identity baseline regardless of comparison phrasing.
    monkeypatch.setattr(settings, "RAG_DECOMPOSITION_ENABLED", False)
    from src.retrieval.routing import decompose_query as facade_decompose

    result = facade_decompose("AXI4 vs CHI")
    assert result.subqueries == ["AXI4 vs CHI"]
    assert result.method == "identity"
    assert result.decomposed is False
