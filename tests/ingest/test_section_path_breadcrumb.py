# @summary
# Unit tests for `_resolve_table_section_paths` covering nested-heading
# breadcrumbs, sibling-section replacement, and same-level replacement at depth.
# Constructs synthetic DoclingDocument-shaped fixtures (no real Docling).
# Exports: (test cases only)
# Deps: pytest
# @end-summary
"""Tests for full-breadcrumb section_path resolution.

The synthetic fixture mimics the shape `iterate_items()` yields:
each entry is a (node, depth) tuple where ``node`` exposes ``.label``,
``.level``, ``.text``, ``.self_ref`` like a real ``DoclingDocument``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from src.ingest.support.docling import _resolve_table_section_paths


@dataclass
class _Item:
    label: str
    level: int = 0
    text: str = ""
    self_ref: str = ""


class _Doc:
    def __init__(self, items: list[_Item]) -> None:
        self._items = items

    def iterate_items(self):
        for it in self._items:
            yield it, 0


def _h(text: str, level: int) -> _Item:
    return _Item(label="section_header", level=level, text=text)


def _table(self_ref: str) -> _Item:
    return _Item(label="table", self_ref=self_ref)


def test_h1_h2_table_emits_full_breadcrumb() -> None:
    """H1 > H2 > table resolves to the two-level breadcrumb."""
    doc = _Doc([
        _h("Bit Field Details", 1),
        _h("CTRL Register Bits", 2),
        _table("#/tables/0"),
    ])
    paths = _resolve_table_section_paths(doc)
    assert paths["#/tables/0"] == "Bit Field Details > CTRL Register Bits"


def test_h1_h2_h3_table_emits_three_level_chain() -> None:
    """Deep nesting produces a 3-segment breadcrumb."""
    doc = _Doc([
        _h("Top", 1),
        _h("Middle", 2),
        _h("Inner", 3),
        _table("#/tables/0"),
    ])
    paths = _resolve_table_section_paths(doc)
    assert paths["#/tables/0"] == "Top > Middle > Inner"


def test_sibling_h2_replaces_prior_h2_under_same_h1() -> None:
    """H1 > H2a [text]; H2b > table -> 'H1 > H2b' (not H1 > H2a > H2b)."""
    doc = _Doc([
        _h("Outer", 1),
        _h("H2a", 2),
        # H2a had body content (a paragraph); H2b is a true sibling and must
        # pop H2a from the stack.
        _Item(label="paragraph", text="some body text under H2a"),
        _h("H2b", 2),
        _table("#/tables/0"),
    ])
    paths = _resolve_table_section_paths(doc)
    assert paths["#/tables/0"] == "Outer > H2b"


def test_consecutive_h1_replaces_previous_h1() -> None:
    """Two H1s in a row with intervening content -> later H1 replaces earlier."""
    doc = _Doc([
        _h("H1a", 1),
        _Item(label="paragraph", text="body for H1a"),
        _h("H1b", 1),
        _table("#/tables/0"),
    ])
    paths = _resolve_table_section_paths(doc)
    assert paths["#/tables/0"] == "H1b"


def test_same_level_back_to_back_headings_nest_when_no_intervening_content() -> None:
    """Docling sometimes emits visually-same-level headings for logical
    parent/child. Two same-level headings with NO content between them are
    treated as parent/child rather than siblings — this is the bug from the
    real-datasheet smoke (Bit Field Details > CTRL Register Bits, both at
    level 2 in Docling's output)."""
    doc = _Doc([
        _h("Bit Field Details", 2),
        _h("CTRL Register Bits", 2),
        _table("#/tables/0"),
    ])
    paths = _resolve_table_section_paths(doc)
    assert paths["#/tables/0"] == "Bit Field Details > CTRL Register Bits"


def test_flat_outline_many_same_level_headings_do_not_stack() -> None:
    """Real datasheets emit dozens of consecutive same-level headings with no
    body items between them (page breaks, captions, etc. don't register as
    content). The W heuristic must not let these stack as ancestors — five
    L1 chapter headings followed by a table should produce a breadcrumb of
    the most recent heading only, not all five stacked.

    Regression for soak finding: ESP32-S3 datasheet produced breadcrumbs up
    to depth 15 ('Chapter1 > Chapter2 > ... > current').
    """
    items = [_h(f"Chapter {i}", 1) for i in range(1, 6)]
    items.append(_table("#/tables/0"))
    doc = _Doc(items)
    paths = _resolve_table_section_paths(doc)
    # Last heading should win; depth must be 1 (≤4 sanity cap also holds).
    assert paths["#/tables/0"] == "Chapter 5"


def test_soak_shaped_71_flat_headings_breadcrumb_depth_capped() -> None:
    """Soak-shaped regression: 71 consecutive same-level headings with no
    body items between them, then a table. Breadcrumb depth must be small
    (≤4) — pre-fix this produced depth 12-15.
    """
    items = [_h(f"Heading {i}", 1) for i in range(71)]
    items.append(_table("#/tables/0"))
    doc = _Doc(items)
    paths = _resolve_table_section_paths(doc)
    breadcrumb = paths["#/tables/0"]
    depth = len([s for s in breadcrumb.split(" > ") if s])
    assert depth <= 4, f"breadcrumb depth {depth} exceeds sanity cap: {breadcrumb!r}"


def test_multiple_tables_each_capture_current_breadcrumb() -> None:
    """Each table snapshots the stack at its position."""
    doc = _Doc([
        _h("A", 1),
        _h("A1", 2),
        _table("#/tables/0"),
        _h("A2", 2),
        _table("#/tables/1"),
        _h("B", 1),
        _table("#/tables/2"),
    ])
    paths = _resolve_table_section_paths(doc)
    assert paths["#/tables/0"] == "A > A1"
    assert paths["#/tables/1"] == "A > A2"
    assert paths["#/tables/2"] == "B"
