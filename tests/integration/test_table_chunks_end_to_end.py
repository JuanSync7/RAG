# @summary
# End-to-end integration test: prove ``table_group_id`` (and friends) stamped by
# the adaptive table chunker survive the full ingestion mapping and land in the
# ``properties`` dict passed to ``weaviate.Collection.batch.add_object()``.
#
# Closes the regression gap left by the existing unit tests:
#   - tests/ingest/test_docling_adaptive_table_chunking.py  -- proves the chunker
#     stamps ``table_group_id`` on emitted Chunk objects.
#   - tests/vector_db/weaviate/test_table_field_schema.py   -- proves the store
#     forwards ``table_group_id`` to Weaviate when given a metadata dict that
#     already contains it.
# Neither verifies that ``chunking_node`` (the metadata-dict builder) actually
# carries ``table_group_id`` through from Chunk.extra_metadata to the store.
# A regression in the spread (``**c.extra_metadata``) or a later
# ``_match_table_artifact`` overwrite would not be caught by either suite.
# Exports: TestTableChunksReachWeaviateProperties
# Deps: src.ingest.support.docling, src.ingest.embedding.nodes.chunking,
#       src.ingest.support.parser_base, src.vector_db.weaviate.store
# @end-summary

"""Integration test: chunker -> chunking_node -> store -> ``add_object`` properties.

The HybridChunker, tokenizer, and Weaviate client are stubbed. The
``chunking_node`` mapping and ``add_documents`` write path are exercised for
real so a regression in either layer that strips ``table_group_id`` (or any
sibling table-* field) would fail this test.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.ingest.common.types import IngestionConfig
from src.ingest.embedding.nodes.chunking import chunking_node
from src.ingest.support import docling as _docling_mod
from src.ingest.support.docling import DoclingParser
from src.ingest.support.parser_base import ParseResult, PageRef, TableArtifact


# ---------------------------------------------------------------------------
# Builders mirroring the existing adaptive-chunker test fixtures.
# ---------------------------------------------------------------------------

def _make_raw_chunk(text: str, headings: list[str] | None = None) -> MagicMock:
    chunk = MagicMock()
    chunk.text = text
    chunk.meta = MagicMock()
    chunk.meta.headings = list(headings or [])
    chunk.meta.doc_items = []
    return chunk


def _markdown_for(cells: list[list[str]]) -> str:
    width = max(len(r) for r in cells)
    norm = [r + [""] * (width - len(r)) for r in cells]
    header = "| " + " | ".join(norm[0]) + " |"
    sep = "| " + " | ".join(["---"] * width) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in norm[1:])
    return "\n".join([header, sep, body]).strip()


def _make_table_artifact(
    *,
    table_id: str = "table-1",
    cells: list[list[str]] | None = None,
    has_header: bool = True,
    caption: str = "Quarterly Sales",
    section_path: str = "Results > FY24",
    page_no: int = 7,
    self_ref: str = "#/tables/0",
) -> TableArtifact:
    if cells is None:
        cells = [["Region", "Revenue"], ["NA", "100"], ["EU", "80"], ["APAC", "60"]]
    md = _markdown_for(cells)
    tbl = TableArtifact(
        table_id=table_id,
        markdown=md,
        cells=cells,
        num_rows=len(cells),
        num_cols=max(len(r) for r in cells),
        has_header=has_header,
        section_path=section_path,
        caption=caption,
        page_ref=PageRef(page_no=page_no, page_label=str(page_no)),
    )
    tbl.self_ref = self_ref
    return tbl


def _real_adaptive_chunks(
    tables: list[TableArtifact], raw_chunks: list[Any]
) -> list[Any]:
    """Drive the *real* ``DoclingParser.chunk()`` with stubbed HybridChunker.

    Returns the post-adaptive ``Chunk`` list — same shape ``chunking_node`` would
    see from ``parser_instance.chunk()`` at runtime. This is the entry point
    whose metadata mapping we want to verify, not mock.
    """
    config = IngestionConfig()
    mock_chunker = MagicMock()
    mock_chunker.chunk = MagicMock(return_value=raw_chunks)
    mock_hc_cls = MagicMock(return_value=mock_chunker)
    with patch.dict(
        "sys.modules",
        {"docling_core.transforms.chunker": MagicMock(HybridChunker=mock_hc_cls)},
    ), patch.object(
        _docling_mod, "_get_or_build_tokenizer", return_value=MagicMock()
    ):
        parser = DoclingParser()
        parser._docling_document = MagicMock()
        parser._max_tokens = 512
        parser._config = config
        return parser.chunk(
            ParseResult(
                markdown="",
                headings=[],
                has_figures=False,
                page_count=0,
                tables=tables,
            )
        )


# ---------------------------------------------------------------------------
# Weaviate ``add_object`` capture (same shape used by test_table_field_schema).
# ---------------------------------------------------------------------------

# Mirrors store.ensure_collection — declares the full set of table+page props
# so add_documents writes them through (rather than silently dropping).
_ALL_TABLE_PROPS: list[str] = [
    "text",
    "source",
    "source_key",
    "section_path",
    "heading",
    "heading_level",
    "heading_path",
    "chunk_index",
    "total_chunks",
    "chunk_type",
    "table_id",
    "table_group_id",
    "table_row_index",
    "table_num_rows",
    "table_num_cols",
    "table_has_header",
    "table_caption",
    "table_markdown",
    "page_no",
    "page_label",
    "page_bbox",
]


def _make_collection_with_props(prop_names: list[str]) -> MagicMock:
    col = MagicMock()
    config = MagicMock()
    props = []
    for n in prop_names:
        p = MagicMock()
        p.name = n
        props.append(p)
    config.properties = props
    col.config.get.return_value = config
    return col


def _capture_add_documents(prop_names: list[str]):
    col = _make_collection_with_props(prop_names)
    captured: list[dict] = []

    class _Batch:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, *exc):
            return False

        def add_object(self_inner, properties, vector, uuid):  # noqa: A002
            captured.append({"properties": properties, "uuid": uuid})

    col.batch.dynamic.return_value = _Batch()
    return col, captured


def _make_client(col: MagicMock) -> MagicMock:
    client = MagicMock()
    client.collections.get.return_value = col
    return client


# ---------------------------------------------------------------------------
# Chunking-node driver.
# ---------------------------------------------------------------------------

def _run_chunking_node(adaptive_chunks: list[Any]) -> list[Any]:
    """Drive the real ``chunking_node`` with a stub parser_instance.

    parser_instance.chunk() simply returns the already-adaptive Chunk list — this
    keeps the test focused on the *mapping* (Chunk -> ProcessedChunk metadata
    dict), which is the layer that must preserve ``table_group_id``.
    """
    parser_instance = MagicMock()
    parser_instance.chunk = MagicMock(return_value=adaptive_chunks)

    parse_result = ParseResult(
        markdown="",
        headings=[],
        has_figures=False,
        page_count=0,
        # Empty tables list: keeps the legacy ``_match_table_artifact`` from
        # rewriting ``chunk_type`` for our already-stamped adaptive chunks.
        # (Adaptive summary/row chunk text does not contain a markdown row
        # signature, so this would be a no-op even with tables present — but
        # we keep the input minimal so the test fails only on mapping bugs.)
        tables=[],
    )

    runtime = MagicMock()
    runtime.config = IngestionConfig()
    runtime.embedder = MagicMock()

    state = {
        "runtime": runtime,
        "source_key": "doc.pdf",
        "source_name": "doc.pdf",
        "source_uri": "file:///tmp/doc.pdf",
        "source_id": "sid",
        "source_version": "v1",
        "connector": "local_fs",
        "raw_text": "irrelevant for chunking when parse_result is set",
        "cleaned_text": "irrelevant",
        "parse_result": parse_result,
        "parser_instance": parser_instance,
        "errors": [],
        "processing_log": [],
    }

    update = chunking_node(state)  # type: ignore[arg-type]
    assert "chunks" in update, f"chunking_node returned no chunks; update={update!r}"
    return update["chunks"]


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------

class TestTableChunksReachWeaviateProperties:
    """End-to-end: table_group_id stamped by adaptive chunker reaches add_object."""

    def test_table_group_id_threads_summary_and_rows_to_properties(self):
        """Summary + row chunks reach Weaviate properties with shared table_group_id."""
        from src.vector_db.weaviate.store import add_documents

        # 1) Build a small uniform table that triggers summary + row emission.
        cells = [
            ["Region", "Revenue"],
            ["NA", "100"],
            ["EU", "80"],
            ["APAC", "60"],
        ]
        tbl = _make_table_artifact(cells=cells, self_ref="#/tables/0")
        table_chunk = _make_raw_chunk(tbl.markdown, headings=["Results", "FY24"])
        prose = _make_raw_chunk("Headline: revenue is up.", headings=["Results"])

        # 2) Real adaptive chunker → Chunk objects with table_group_id stamped.
        adaptive = _real_adaptive_chunks([tbl], [prose, table_chunk])
        types = [c.extra_metadata.get("chunk_type") for c in adaptive]
        assert types.count("table_summary") == 1
        assert types.count("table_row") == 3, types
        # Sanity: the chunker did stamp table_group_id on every adaptive chunk.
        for c in adaptive:
            if c.extra_metadata.get("chunk_type") in ("table_summary", "table_row"):
                assert c.extra_metadata.get("table_group_id") == "#/tables/0", (
                    "adaptive chunker dropped table_group_id; "
                    "fix tests/ingest/test_docling_adaptive_table_chunking.py first"
                )

        # 3) Real chunking_node mapping → list[ProcessedChunk] with metadata dicts.
        processed = _run_chunking_node(adaptive)
        texts = [pc.text for pc in processed]
        metadatas = [pc.metadata for pc in processed]
        embeddings = [[0.0] for _ in processed]

        # Spot-check the mapping itself (cheap, points at the right layer if
        # the downstream add_object assertion fails).
        meta_by_type: dict[str, list[dict]] = {}
        for m in metadatas:
            meta_by_type.setdefault(str(m.get("chunk_type") or ""), []).append(m)
        assert len(meta_by_type.get("table_summary", [])) == 1, (
            f"chunking_node lost the summary chunk_type stamp: "
            f"{[m.get('chunk_type') for m in metadatas]}"
        )
        assert len(meta_by_type.get("table_row", [])) == 3, (
            f"chunking_node lost row chunk_type stamps: "
            f"{[m.get('chunk_type') for m in metadatas]}"
        )
        for m in meta_by_type["table_summary"] + meta_by_type["table_row"]:
            assert m.get("table_group_id") == "#/tables/0", (
                "chunking_node dropped table_group_id from chunk.extra_metadata "
                "during ProcessedChunk mapping — this is the regression the "
                "test is built to catch."
            )

        # 4) Real add_documents → captured properties dict.
        col, captured = _capture_add_documents(_ALL_TABLE_PROPS)
        client = _make_client(col)
        n = add_documents(client, texts=texts, embeddings=embeddings, metadatas=metadatas)
        assert n == len(processed)
        assert len(captured) == len(processed)

        # 5) Final E2E assertions — what landed in ``properties=`` on add_object.
        props_by_type: dict[str, list[dict]] = {}
        for c in captured:
            ct = str(c["properties"].get("chunk_type") or "")
            props_by_type.setdefault(ct, []).append(c["properties"])

        # Summary chunk: full table metadata present, including non-empty markdown.
        summaries = props_by_type.get("table_summary", [])
        assert len(summaries) == 1, props_by_type
        s = summaries[0]
        assert s["chunk_type"] == "table_summary"
        assert s["table_group_id"] == "#/tables/0"
        assert s["table_id"] == "table-1"
        assert s["table_num_rows"] == len(cells)
        assert s["table_num_cols"] == 2
        assert s["table_has_header"] is True
        assert s["table_caption"] == "Quarterly Sales"
        assert s["table_markdown"], "summary chunk lost table_markdown"
        assert "Region" in s["table_markdown"] and "Revenue" in s["table_markdown"]
        # Row chunks: same group id, indexed 0..N-1, no markdown payload.
        rows = sorted(
            props_by_type.get("table_row", []),
            key=lambda p: p["table_row_index"],
        )
        assert len(rows) == 3, props_by_type
        assert [r["table_row_index"] for r in rows] == [0, 1, 2]
        for r in rows:
            assert r["chunk_type"] == "table_row"
            assert r["table_group_id"] == "#/tables/0", (
                "row chunk reached Weaviate without the summary's table_group_id"
            )
            assert r["table_id"] == "table-1"
            # table_markdown is summary-only; row defaults to empty string.
            assert r["table_markdown"] == "", (
                "row chunk leaked table_markdown into properties; should be summary-only"
            )

    def test_distinct_tables_get_distinct_group_ids_in_properties(self):
        """Two tables in one doc → two distinct table_group_id values at the store."""
        from src.vector_db.weaviate.store import add_documents

        tbl_a = _make_table_artifact(
            table_id="table-1",
            cells=[["H1", "H2"], ["a", "b"], ["c", "d"]],
            self_ref="#/tables/0",
            caption="Alpha",
            section_path="A",
        )
        tbl_b = _make_table_artifact(
            table_id="table-2",
            cells=[["X", "Y"], ["1", "2"]],
            self_ref="#/tables/1",
            caption="Beta",
            section_path="B",
        )
        chunk_a = _make_raw_chunk(tbl_a.markdown)
        chunk_b = _make_raw_chunk(tbl_b.markdown)

        adaptive = _real_adaptive_chunks([tbl_a, tbl_b], [chunk_a, chunk_b])
        processed = _run_chunking_node(adaptive)

        col, captured = _capture_add_documents(_ALL_TABLE_PROPS)
        client = _make_client(col)
        add_documents(
            client,
            texts=[pc.text for pc in processed],
            embeddings=[[0.0] for _ in processed],
            metadatas=[pc.metadata for pc in processed],
        )

        groups_by_table: dict[str, set[str]] = {}
        for c in captured:
            p = c["properties"]
            tid = str(p.get("table_id") or "")
            if not tid:
                continue
            groups_by_table.setdefault(tid, set()).add(str(p.get("table_group_id") or ""))

        assert groups_by_table.get("table-1") == {"#/tables/0"}, groups_by_table
        assert groups_by_table.get("table-2") == {"#/tables/1"}, groups_by_table
