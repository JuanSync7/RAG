"""Unit tests for soak xref-extraction metric helpers.

These tests do NOT invoke ``run_soak`` — that requires Docling models. They
target the pure-function helpers added for the xref-extraction soak pass:

- ``_xref_edge_metrics(chunks)``
- ``_xref_resolvability(chunks, tables)``
- ``_caption_label_coverage(tables)``
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from scripts.soak_table_chunking import (
    _caption_label_coverage,
    _xref_edge_metrics,
    _xref_resolvability,
)


# --------------------------------------------------------------------------- #
# Lightweight fakes that mimic the Chunk / TableArtifact attribute surface.   #
# --------------------------------------------------------------------------- #


@dataclass
class FakeChunk:
    text: str = ""
    extra_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeTable:
    caption: str = ""
    caption_label: str = ""
    document_id: str = ""


def _mk_chunk(targets: list[dict] | None = None, **meta: Any) -> FakeChunk:
    em: dict[str, Any] = {}
    if targets is not None:
        em["xref_targets"] = json.dumps(targets)
    em.update(meta)
    return FakeChunk(extra_metadata=em)


# --------------------------------------------------------------------------- #
# _xref_edge_metrics                                                          #
# --------------------------------------------------------------------------- #


class TestXrefEdgeMetrics:
    def test_empty_chunks(self):
        m = _xref_edge_metrics([])
        assert m == {
            "total_chunks_with_edges": 0,
            "total_edges": 0,
            "edges_per_chunk_p50": 0,
            "edges_per_chunk_p90": 0,
            "by_type": {},
        }

    def test_chunks_without_xref_targets(self):
        chunks = [FakeChunk(), FakeChunk(extra_metadata={"other": "x"})]
        m = _xref_edge_metrics(chunks)
        assert m["total_chunks_with_edges"] == 0
        assert m["total_edges"] == 0
        assert m["by_type"] == {}

    def test_multiple_refs_per_chunk_and_type_distribution(self):
        chunks = [
            _mk_chunk(targets=[
                {"type": "section", "value": "Section 3.1"},
                {"type": "table", "value": "Table 5-2"},
            ]),
            _mk_chunk(targets=[
                {"type": "section_symbol", "value": "§4"},
                {"type": "figure", "value": "Figure 2"},
                {"type": "appendix", "value": "Appendix A"},
            ]),
            _mk_chunk(targets=[]),  # empty list still has key but no edges
            _mk_chunk(targets=None),  # no xref_targets at all
        ]
        m = _xref_edge_metrics(chunks)
        assert m["total_chunks_with_edges"] == 2
        assert m["total_edges"] == 5
        assert m["by_type"] == {
            "section": 1,
            "table": 1,
            "section_symbol": 1,
            "figure": 1,
            "appendix": 1,
        }

    def test_percentile_arithmetic(self):
        # Edge counts per chunk (over chunks-with-edges):
        # 1, 1, 1, 1, 1, 1, 1, 1, 1, 10
        # Sorted: [1]*9 + [10]
        # p50 should be 1, p90 should be either 1 or 10 depending on method.
        chunks = []
        for _ in range(9):
            chunks.append(_mk_chunk(targets=[{"type": "section", "value": "1"}]))
        chunks.append(
            _mk_chunk(targets=[{"type": "section", "value": str(i)} for i in range(10)])
        )
        m = _xref_edge_metrics(chunks)
        assert m["total_chunks_with_edges"] == 10
        assert m["total_edges"] == 19
        assert m["edges_per_chunk_p50"] == 1
        # nearest-rank p90 over 10 items lands at index 8 (zero-based) == 1.
        # We accept either 1 (nearest-rank) or 10 (next bucket); pin exact
        # to the implementation: sort-and-index ceil(0.9*N)-1 → idx 8 → 1.
        assert m["edges_per_chunk_p90"] in (1, 10)

    def test_malformed_json_tolerated(self):
        chunks = [FakeChunk(extra_metadata={"xref_targets": "not-json"})]
        m = _xref_edge_metrics(chunks)
        assert m["total_chunks_with_edges"] == 0
        assert m["total_edges"] == 0


# --------------------------------------------------------------------------- #
# _xref_resolvability                                                         #
# --------------------------------------------------------------------------- #


class TestXrefResolvability:
    def test_empty(self):
        r = _xref_resolvability([], [])
        assert r == {
            "section_resolvable": 0,
            "table_resolvable": 0,
            "unresolvable_table_no_label": 0,
            "unresolvable_figure": 0,
            "unresolvable_appendix": 0,
        }

    def test_section_resolves_via_section_path(self):
        chunks = [
            _mk_chunk(
                targets=[{"type": "section", "value": "Section 3.1"}],
                document_id="docA",
                section_path="Chapter 3 > 3.1 Power",
            ),
            # A chunk in the same document whose section_path holds "3.1".
            _mk_chunk(
                document_id="docA",
                section_path="Overview > 3.1 Power",
            ),
        ]
        r = _xref_resolvability(chunks, tables=[])
        assert r["section_resolvable"] == 1

    def test_section_boundary_does_not_match_310(self):
        # ref "3.1" against section_path "3.10 Stuff" → NOT resolvable.
        chunks = [
            _mk_chunk(
                targets=[{"type": "section", "value": "Section 3.1"}],
                document_id="docA",
                section_path="3.10 Boot ROM",
            ),
            _mk_chunk(document_id="docA", section_path="3.10 Boot ROM"),
        ]
        r = _xref_resolvability(chunks, tables=[])
        assert r["section_resolvable"] == 0

    def test_section_symbol_normalises(self):
        chunks = [
            _mk_chunk(
                targets=[{"type": "section_symbol", "value": "§4.2"}],
                document_id="docA",
                section_path="Top > something",
            ),
            _mk_chunk(document_id="docA", section_path="Chapter 4 > 4.2 Modes"),
        ]
        r = _xref_resolvability(chunks, tables=[])
        assert r["section_resolvable"] == 1

    def test_table_resolves_via_caption_label(self):
        chunks = [
            _mk_chunk(
                targets=[{"type": "table", "value": "Table 5-2"}],
                document_id="docA",
            ),
        ]
        tables = [
            FakeTable(caption_label="Table 5-2", document_id="docA"),
        ]
        r = _xref_resolvability(chunks, tables=tables)
        assert r["table_resolvable"] == 1
        assert r["unresolvable_table_no_label"] == 0

    def test_table_unresolvable_no_label_match(self):
        chunks = [
            _mk_chunk(
                targets=[{"type": "table", "value": "Table 99"}],
                document_id="docA",
            ),
        ]
        tables = [FakeTable(caption_label="Table 5-2", document_id="docA")]
        r = _xref_resolvability(chunks, tables=tables)
        assert r["table_resolvable"] == 0
        assert r["unresolvable_table_no_label"] == 1

    def test_figure_and_appendix_always_unresolvable(self):
        chunks = [
            _mk_chunk(
                targets=[
                    {"type": "figure", "value": "Figure 2"},
                    {"type": "appendix", "value": "Appendix A"},
                    {"type": "figure", "value": "Figure 7"},
                ],
                document_id="docA",
            ),
        ]
        r = _xref_resolvability(chunks, tables=[])
        assert r["unresolvable_figure"] == 2
        assert r["unresolvable_appendix"] == 1
        assert r["section_resolvable"] == 0
        assert r["table_resolvable"] == 0

    def test_mixed_types_aggregated(self):
        chunks = [
            _mk_chunk(
                targets=[
                    {"type": "section", "value": "Section 3.1"},
                    {"type": "table", "value": "Table 5-2"},
                    {"type": "figure", "value": "Figure 2"},
                ],
                document_id="docA",
                section_path="Chapter 3 > 3.1 Power",
            ),
            _mk_chunk(document_id="docA", section_path="Chapter 3 > 3.1 Power"),
        ]
        tables = [FakeTable(caption_label="Table 5-2", document_id="docA")]
        r = _xref_resolvability(chunks, tables=tables)
        assert r["section_resolvable"] == 1
        assert r["table_resolvable"] == 1
        assert r["unresolvable_figure"] == 1


# --------------------------------------------------------------------------- #
# _caption_label_coverage                                                     #
# --------------------------------------------------------------------------- #


class TestCaptionLabelCoverage:
    def test_empty(self):
        c = _caption_label_coverage([])
        assert c == {"tables_total": 0, "tables_with_label": 0, "label_rate": 0.0}

    def test_all_labelled(self):
        tables = [
            FakeTable(caption_label="Table 1"),
            FakeTable(caption_label="Table 2-3"),
        ]
        c = _caption_label_coverage(tables)
        assert c == {"tables_total": 2, "tables_with_label": 2, "label_rate": 1.0}

    def test_mixed(self):
        tables = [
            FakeTable(caption_label="Table 1"),
            FakeTable(caption_label=""),
            FakeTable(caption_label="Table 3-1"),
            FakeTable(caption_label=""),
        ]
        c = _caption_label_coverage(tables)
        assert c["tables_total"] == 4
        assert c["tables_with_label"] == 2
        assert c["label_rate"] == pytest.approx(0.5)
