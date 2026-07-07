# @summary
# Unit tests for TurnState's facet-coverage bookkeeping (record_facet +
# facets_fully_covered): case-insensitive dedup of decomposed sub-questions,
# monotonic coverage (once covered stays covered), blank-question rejection, and
# the "all facets covered" commit signal (False with no facets so the guard
# stays out of the way until a DECOMPOSE runs).
# @end-summary
"""Tests for ``TurnState`` facet-coverage state (``turn_loop/schemas.py``)."""

from __future__ import annotations

from src.retrieval.pipeline.turn_loop.schemas import TurnState


def test_record_facet_appends_new_facets():
    state = TurnState()
    state.record_facet("throughput of the bus", covered=True)
    state.record_facet("latency of the bus", covered=False)
    assert [(f.question, f.covered) for f in state.facets] == [
        ("throughput of the bus", True),
        ("latency of the bus", False),
    ]


def test_record_facet_dedups_case_insensitively():
    state = TurnState()
    state.record_facet("Reset value", covered=False)
    state.record_facet("  reset value  ", covered=False)
    assert len(state.facets) == 1  # same facet, whitespace/case-folded


def test_record_facet_coverage_is_monotonic():
    """A facet covered by an earlier round stays covered when a later round names
    it again with no fresh evidence (class: latest-sample-overwrites-best)."""
    state = TurnState()
    state.record_facet("the AXI channels", covered=True)
    state.record_facet("the axi channels", covered=False)  # re-named, no new keep
    assert state.facets[0].covered is True


def test_record_facet_upgrades_uncovered_to_covered():
    state = TurnState()
    state.record_facet("the AXI channels", covered=False)
    state.record_facet("the AXI channels", covered=True)  # later round found evidence
    assert len(state.facets) == 1
    assert state.facets[0].covered is True


def test_record_facet_ignores_blank_question():
    state = TurnState()
    state.record_facet("   ", covered=True)
    assert state.facets == []


def test_facets_fully_covered_false_without_facets():
    # No DECOMPOSE has run: the guard must stay inert (not vacuously fire).
    assert TurnState().facets_fully_covered() is False


def test_facets_fully_covered_requires_every_facet_covered():
    state = TurnState()
    state.record_facet("a", covered=True)
    state.record_facet("b", covered=False)
    assert state.facets_fully_covered() is False
    state.record_facet("b", covered=True)  # b now has evidence
    assert state.facets_fully_covered() is True
