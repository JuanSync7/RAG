# @summary
# Unit tests for the Tier-1 regex comparison-decomposition fallback
# (``regex_decompose``). Pins the safe-split behaviour AND the mis-split
# guards from DOCUMENT_ROUTING_DESIGN.md §6.5: a MISS (return ``[]``) is safe;
# a MIS-SPLIT is actively harmful, so every observed mis-split case must
# return ``[]`` rather than shred a query into garbage sub-queries.
# Deps: pytest, src.retrieval.routing.decomposition
# @end-summary
"""Unit tests for ``src.retrieval.routing.decomposition.regex_decompose``.

The Tier-1 regex decomposer is a *fallback only* (the design's primary path is
the glossary-grounded LLM, added in a later slice). Its contract:

* Split a **clear** comparison into one standalone single-entity sub-query per
  compared entity, returning that ordered list of cleaned entity strings.
* Otherwise return ``[]`` — meaning the caller keeps the original query and
  runs normal flat retrieval. Safety (avoiding mis-splits) beats recall.

Normalization pinned by these tests (also documented in the function docstring):

* Each produced sub-query is whitespace-trimmed and has any leading filler
  (a lead-in verb like ``compare``, articles like ``the``) and trailing aspect
  clause (``for ...`` / ``with ...`` / ``in a ...`` etc.) stripped.
* Casing of the entity token is preserved as written by the user (the function
  does not upper/lower-case entities; ``"AXI4"`` stays ``"AXI4"``).
* When a split yields fewer than two real entities, or any candidate "entity"
  looks sentence-like (too many words / contains an aspect preposition), the
  whole split is rejected as a probable mis-split and ``[]`` is returned.
"""
from __future__ import annotations

import pytest

from src.retrieval.routing.decomposition import regex_decompose


# ─── POSITIVE: clear comparisons split into standalone entities ─────────────


def test_vs_simple():
    assert regex_decompose("AXI4 vs CHI") == ["AXI4", "CHI"]


def test_vs_with_period():
    assert regex_decompose("AXI4 vs. CHI") == ["AXI4", "CHI"]


def test_versus_word():
    assert regex_decompose("AXI4 versus CHI") == ["AXI4", "CHI"]


def test_compared_to():
    assert regex_decompose("AHB compared to APB") == ["AHB", "APB"]


def test_compared_with():
    assert regex_decompose("AHB compared with APB") == ["AHB", "APB"]


def test_difference_between_and():
    """The ONLY frame where 'and' is allowed to split."""
    assert regex_decompose("difference between AHB and APB") == ["AHB", "APB"]


def test_differences_between_and():
    assert regex_decompose("differences between AHB and APB") == ["AHB", "APB"]


def test_slash_list_with_leadin_and_aspect():
    """Drop the 'Compare AMBA' lead-in and the 'for coherency' aspect; keep the
    slash items as standalone entities."""
    assert regex_decompose("Compare AMBA AXI4 / AXI5 / CHI for coherency") == [
        "AXI4",
        "AXI5",
        "CHI",
    ]


def test_slash_list_bare():
    assert regex_decompose("AXI4 / AXI5 / CHI") == ["AXI4", "AXI5", "CHI"]


def test_vs_casing_preserved():
    """Entity casing is preserved verbatim (no upper/lower-casing)."""
    assert regex_decompose("axi4 vs chi") == ["axi4", "chi"]


def test_vs_whitespace_trimmed():
    assert regex_decompose("  AXI4   vs   CHI  ") == ["AXI4", "CHI"]


# ─── MIS-SPLIT GUARDS: every one of these MUST return [] ────────────────────


def test_guard_aspect_phrase_not_an_entity():
    """'for a high-performance SoC' is an aspect, not a second entity → no split."""
    assert regex_decompose("AHB for a high-performance SoC") == []


def test_guard_bare_and_not_a_comparison():
    """A bare 'and' outside the 'difference between' frame must not split."""
    assert regex_decompose("design and verification of a block") == []


def test_guard_informal_or_phrasing_left_for_llm():
    """Informal comparison regex cannot ground → leave it for the LLM tier."""
    assert (
        regex_decompose("the hub-based approach or the extension-based one?") == []
    )


def test_guard_needle_query_not_split():
    """A plain needle/fact query is not a comparison."""
    assert (
        regex_decompose("reset value of Filters_N_Status at 0x0064") == []
    )


def test_guard_sequencing_not_comparison():
    """'steps before inserting MBIST' is sequencing, not comparison."""
    assert regex_decompose("steps before inserting MBIST") == []


def test_guard_empty_query():
    assert regex_decompose("") == []


def test_guard_single_entity_no_split():
    """A single topic with a comparison-ish word but no second entity → []."""
    assert regex_decompose("describe the AXI4 write channel") == []


# ─── Additional mis-split robustness ────────────────────────────────────────


def test_guard_difference_between_with_aspect_tail_still_clean():
    """Even within the allowed frame, a long aspect tail must not become an
    entity — the second 'entity' here is sentence-like, so reject the split."""
    out = regex_decompose(
        "difference between AHB and the way you verify a whole subsystem"
    )
    assert out == []


def test_guard_slash_list_of_non_entities_not_split():
    """A slash path / non-entity slash list must not be treated as a comparison."""
    assert regex_decompose("read the file at src/foo/bar baseline") == []


def test_returns_list_type():
    """Always returns a list (never None)."""
    assert isinstance(regex_decompose("AXI4 vs CHI"), list)
    assert isinstance(regex_decompose("not a comparison"), list)
