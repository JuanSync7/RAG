# @summary
# Regression: chunking_node MUST NOT clobber a pre-set ``chunk_type`` (notably
# ``"table_summary"`` / ``"table_row"`` stamped by the adaptive table chunker
# upstream in ``src.ingest.support.docling._apply_adaptive_table_chunking``).
#
# Two value-spaces flow into the same Weaviate ``chunk_type`` column:
#   - Legacy (this file, L191-203):       "table" | "text"
#   - Adaptive (docling.py, L964/L996):   "table_summary" | "table_row"
#
# Pass-through is *safe in practice* because the adaptive chunker emits chunks
# whose text never matches a pipe-table markdown signature (summary text is
# ``"Table: ...\nColumns: a | b\nRows: N"`` and row text is
# ``"caption\nh1: v1 | h2: v2"`` — neither contains the leading-pipe header
# row ``"| a | b |"`` that ``_match_table_artifact`` fingerprints on).
#
# This file locks that contract so a future change to either side (signature
# heuristic OR adaptive text format) that re-introduces collision fails CI.
# Exports: tests in TestChunkTypePreservedThroughChunkingNode
# Deps: src.ingest.embedding.nodes.chunking, src.ingest.support.parser_base,
#       src.ingest.common.types
# @end-summary

"""Pin chunk_type precedence between adaptive chunker and legacy chunking_node.

Without these tests, a regression that flips the chunking_node L192 assignment
from ``meta["chunk_type"] = "table"`` (unconditional) to a setdefault-style
guard would be free to silently overwrite ``"table_summary"`` / ``"table_row"``
stamps coming from the adaptive chunker — which would corrupt every downstream
consumer that branches on ``chunk_type`` (snippet selection, store row IDs,
table-summary join keys).
"""
from __future__ import annotations

from unittest.mock import MagicMock

from src.ingest.common.types import IngestionConfig
from src.ingest.embedding.nodes.chunking import chunking_node
from src.ingest.support.parser_base import (
    Chunk,
    PageRef,
    ParseResult,
    TableArtifact,
)


def _make_table_artifact() -> TableArtifact:
    """Build a TableArtifact whose markdown signature is ``"| Region | Revenue |"``."""
    cells = [["Region", "Revenue"], ["NA", "100"], ["EU", "80"]]
    md = (
        "| Region | Revenue |\n"
        "| --- | --- |\n"
        "| NA | 100 |\n"
        "| EU | 80 |"
    )
    tbl = TableArtifact(
        table_id="table-1",
        markdown=md,
        cells=cells,
        num_rows=3,
        num_cols=2,
        has_header=True,
        section_path="Results > FY24",
        caption="Quarterly Sales",
        page_ref=PageRef(page_no=7),
    )
    tbl.self_ref = "#/tables/0"
    return tbl


def _make_state(parser_chunks: list[Chunk], tables: list[TableArtifact]) -> dict:
    parser_instance = MagicMock()
    parser_instance.chunk = MagicMock(return_value=parser_chunks)
    runtime = MagicMock()
    runtime.config = IngestionConfig()
    runtime.embedder = MagicMock()
    return {
        "runtime": runtime,
        "source_key": "doc.pdf",
        "source_name": "doc.pdf",
        "source_uri": "file:///tmp/doc.pdf",
        "source_id": "sid-1",
        "source_version": "v1",
        "connector": "local_fs",
        "raw_text": "stub",
        "cleaned_text": "stub",
        "parse_result": ParseResult(
            markdown="",
            headings=[],
            has_figures=False,
            page_count=0,
            tables=tables,
        ),
        "parser_instance": parser_instance,
        "errors": [],
        "processing_log": [],
    }


class TestChunkTypePreservedThroughChunkingNode:
    """Lock the chunking_node L191-203 precedence rule for adaptive chunk_type."""

    def test_table_summary_chunk_type_survives_chunking_node(self):
        """A pre-stamped ``"table_summary"`` chunk reaches Weaviate metadata intact."""
        tbl = _make_table_artifact()
        # Realistic adaptive summary text: ``_build_table_summary_text`` output.
        summary_text = (
            "Table: Quarterly Sales\n"
            "Columns: Region | Revenue\n"
            "Rows: 2"
        )
        summary_chunk = Chunk(
            text=summary_text,
            section_path="Results > FY24",
            heading="FY24",
            heading_level=2,
            chunk_index=0,
            extra_metadata={
                "chunk_type": "table_summary",
                "table_id": tbl.table_id,
                "table_group_id": "#/tables/0",
                "table_num_rows": 3,
                "table_num_cols": 2,
                "table_has_header": True,
                "table_markdown": tbl.markdown,
                "table_caption": "Quarterly Sales",
            },
            heading_path=["Results", "FY24"],
            page_ref=PageRef(page_no=7),
        )

        state = _make_state([summary_chunk], [tbl])
        update = chunking_node(state)  # type: ignore[arg-type]

        chunks = update["chunks"]
        assert len(chunks) == 1
        meta = chunks[0].metadata
        assert meta["chunk_type"] == "table_summary", (
            f"chunking_node clobbered adaptive chunk_type; got {meta['chunk_type']!r}"
        )
        # Adaptive-only fields survive the spread-then-overwrite mapping.
        assert meta["table_group_id"] == "#/tables/0"
        assert meta["table_markdown"] == tbl.markdown

    def test_table_row_chunk_type_survives_chunking_node(self):
        """A pre-stamped ``"table_row"`` chunk reaches Weaviate metadata intact."""
        tbl = _make_table_artifact()
        # Realistic adaptive row text: ``"caption\nh1: v1 | h2: v2"``.
        row_text = "Quarterly Sales\nRegion: NA | Revenue: 100"
        row_chunk = Chunk(
            text=row_text,
            section_path="Results > FY24",
            heading="FY24",
            heading_level=2,
            chunk_index=0,
            extra_metadata={
                "chunk_type": "table_row",
                "table_id": tbl.table_id,
                "table_group_id": "#/tables/0",
                "table_row_index": 0,
                "table_num_rows": 3,
                "table_num_cols": 2,
            },
            heading_path=["Results", "FY24"],
            page_ref=PageRef(page_no=7),
        )

        state = _make_state([row_chunk], [tbl])
        update = chunking_node(state)  # type: ignore[arg-type]

        chunks = update["chunks"]
        assert len(chunks) == 1
        meta = chunks[0].metadata
        assert meta["chunk_type"] == "table_row", (
            f"chunking_node clobbered adaptive chunk_type; got {meta['chunk_type']!r}"
        )
        assert meta["table_group_id"] == "#/tables/0"
        assert meta["table_row_index"] == 0

    def test_legacy_table_stamp_still_applied_to_unstamped_chunks(self):
        """Chunks without a pre-set chunk_type still get the legacy ``"table"`` stamp.

        Locks the converse: removing/weakening the legacy heuristic would silently
        regress the non-adaptive code path (e.g., parsers without adaptive
        chunking, or HybridChunker output that still surfaces table markdown
        when adaptive chunking is disabled).
        """
        tbl = _make_table_artifact()
        # Chunk text contains the table's pipe-row signature verbatim AND
        # carries no chunk_type stamp upstream.
        unstamped = Chunk(
            text="Intro prose.\n" + tbl.markdown,
            section_path="Results > FY24",
            heading="FY24",
            heading_level=2,
            chunk_index=0,
            extra_metadata={},
            heading_path=["Results", "FY24"],
            page_ref=PageRef(page_no=7),
        )

        state = _make_state([unstamped], [tbl])
        update = chunking_node(state)  # type: ignore[arg-type]

        meta = update["chunks"][0].metadata
        assert meta["chunk_type"] == "table"
        assert meta["table_id"] == tbl.table_id
