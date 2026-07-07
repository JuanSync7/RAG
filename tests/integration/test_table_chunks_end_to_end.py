# @summary
# End-to-end integration test: prove the table-row *block* fields stamped by the
# adaptive table chunker survive the full ingestion mapping. ``table_group_id``
# (and the other store-declared table-* fields) must land in the ``properties``
# dict passed to ``weaviate.Collection.batch.add_object()``; the block-position
# fields (``table_row_block_index`` / ``table_block_total``) must thread through
# the ``chunking_node`` ProcessedChunk metadata dict (they are intentionally NOT
# in the store's Weaviate property set — see the test docstrings).
#
# New-world contract (post token-budget table-chunking rewrite): a table emits
# ONLY ``table_row`` block chunks (NO ``table_summary``). Every block restates
# the header (never header-blind), packs as many whole body rows as fit the
# token budget, and a large table therefore splits into multiple lossless blocks
# that cover every body row exactly once. Block 0 carries the full
# ``table_markdown``; later blocks carry "".
#
# Closes the regression gap left by the existing unit tests:
#   - tests/ingest/test_docling_adaptive_table_chunking.py  -- proves the chunker
#     stamps ``table_group_id`` / block fields on emitted Chunk objects.
#   - tests/vector_db/weaviate/test_table_field_schema.py   -- proves the store
#     forwards ``table_group_id`` to Weaviate when given a metadata dict that
#     already contains it.
# Neither verifies that ``chunking_node`` (the metadata-dict builder) actually
# carries ``table_group_id`` (and the block fields) through from
# Chunk.extra_metadata to the store. A regression in the spread
# (``**c.extra_metadata``) or a later ``_match_table_artifact`` overwrite would
# not be caught by either suite.
# Exports: TestTableChunksReachWeaviateProperties
# Deps: src.ingest.support.docling, src.ingest.embedding.nodes.chunking,
#       src.ingest.support.parser_base, src.vector_db.weaviate.store
# @end-summary

"""Integration test: chunker -> chunking_node -> store -> ``add_object`` properties.

The HybridChunker, tokenizer, and Weaviate client are stubbed. The
``chunking_node`` mapping and ``add_documents`` write path are exercised for
real so a regression in either layer that strips ``table_group_id`` (or any
sibling table-* field) would fail this test.

Token budget in these tests: the stubbed tokenizer makes the chunker's token
counter return ~1 token per row, so passing a small ``max_tokens`` to
``_real_adaptive_chunks`` forces a large table to split into several
token-bounded ``table_row`` blocks — the mechanism the multi-block assertions
exercise without depending on a real HF tokenizer.
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
    tables: list[TableArtifact],
    raw_chunks: list[Any],
    *,
    max_tokens: int | None = None,
) -> list[Any]:
    """Drive the *real* ``DoclingParser.chunk()`` with stubbed HybridChunker.

    Returns the post-adaptive ``Chunk`` list — same shape ``chunking_node`` would
    see from ``parser_instance.chunk()`` at runtime. This is the entry point
    whose metadata mapping we want to verify, not mock.

    ``max_tokens`` overrides ``IngestionConfig.hybrid_chunker_max_tokens`` — the
    single knob the adaptive table chunker uses to greedy-pack whole body rows
    into token-bounded blocks. With the stubbed tokenizer the counter yields
    ~1 token/row, so a small value (e.g. 3) deterministically forces a large
    table to split into multiple lossless ``table_row`` blocks.
    """
    config = IngestionConfig()
    if max_tokens is not None:
        config.hybrid_chunker_max_tokens = max_tokens
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

    def test_table_group_id_and_block_fields_thread_to_properties(self):
        """table_row blocks thread table_group_id + block fields to the store.

        Rewrite of the former ``..._threads_summary_and_rows_to_properties``
        test, which asserted a single ``table_summary`` chunk. The token-budget
        rewrite of the adaptive table chunker REMOVED the summary concept: a
        table now emits ONLY ``table_row`` block chunks. This test proves the
        new contract end-to-end:

          * the chunker emits NO ``table_summary`` chunk_type at all;
          * the single uniform table fits one block → one ``table_row`` chunk;
          * ``table_group_id`` (and the store-declared table-* fields) thread
            through ``chunking_node`` → ``add_documents`` → ``add_object``
            properties;
          * block 0 carries the full ``table_markdown``;
          * the block-position fields ``table_row_block_index`` /
            ``table_block_total`` thread through the ``chunking_node`` metadata
            dict (they are intentionally absent from the store's Weaviate
            property set — ``add_documents`` only forwards its fixed
            ``optional`` table-* keys — so they are asserted at the metadata
            layer, not at ``add_object`` properties).
        """
        from src.vector_db.weaviate.store import add_documents

        # 1) Build a small uniform table that fits a single token-bounded block.
        cells = [
            ["Region", "Revenue"],
            ["NA", "100"],
            ["EU", "80"],
            ["APAC", "60"],
        ]
        tbl = _make_table_artifact(cells=cells, self_ref="#/tables/0")
        table_chunk = _make_raw_chunk(tbl.markdown, headings=["Results", "FY24"])
        prose = _make_raw_chunk("Headline: revenue is up.", headings=["Results"])

        # 2) Real adaptive chunker → Chunk objects. New world: ONLY table_row.
        adaptive = _real_adaptive_chunks([tbl], [prose, table_chunk])
        types = [c.extra_metadata.get("chunk_type") for c in adaptive]
        assert types.count("table_summary") == 0, (
            f"summary concept was removed; chunker must emit no table_summary: {types}"
        )
        assert types.count("table_row") == 1, types  # 3 body rows fit one block
        # Sanity: the chunker stamped the group id + block fields on the block.
        row_chunks = [
            c for c in adaptive if c.extra_metadata.get("chunk_type") == "table_row"
        ]
        only = row_chunks[0]
        assert only.extra_metadata.get("table_group_id") == "#/tables/0", (
            "adaptive chunker dropped table_group_id; "
            "fix tests/ingest/test_docling_adaptive_table_chunking.py first"
        )
        assert only.extra_metadata.get("table_row_block_index") == 0
        assert only.extra_metadata.get("table_block_total") == 1
        assert only.extra_metadata.get("table_markdown"), "block 0 lost table_markdown"

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
        assert not meta_by_type.get("table_summary"), (
            "chunking_node fabricated a table_summary chunk_type: "
            f"{[m.get('chunk_type') for m in metadatas]}"
        )
        assert len(meta_by_type.get("table_row", [])) == 1, (
            f"chunking_node lost the row chunk_type stamp: "
            f"{[m.get('chunk_type') for m in metadatas]}"
        )
        row_meta = meta_by_type["table_row"][0]
        assert row_meta.get("table_group_id") == "#/tables/0", (
            "chunking_node dropped table_group_id from chunk.extra_metadata "
            "during ProcessedChunk mapping — this is the regression the test "
            "is built to catch."
        )
        # Block-position fields survive the ``**c.extra_metadata`` spread. These
        # are NOT written to Weaviate by add_documents (the store's optional set
        # has no slot for them), so the metadata dict is the layer that must
        # carry them through to any block-aware consumer.
        assert row_meta.get("table_row_block_index") == 0
        assert row_meta.get("table_block_total") == 1

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

        assert not props_by_type.get("table_summary"), (
            "no table_summary chunk should reach the store after the rewrite"
        )
        rows = props_by_type.get("table_row", [])
        assert len(rows) == 1, props_by_type
        r = rows[0]
        # The (sole, block-0) row chunk: full table metadata + non-empty markdown.
        assert r["chunk_type"] == "table_row"
        assert r["table_group_id"] == "#/tables/0", (
            "row chunk reached Weaviate without table_group_id"
        )
        assert r["table_id"] == "table-1"
        assert r["table_row_index"] == 0
        assert r["table_num_rows"] == len(cells)
        assert r["table_num_cols"] == 2
        assert r["table_has_header"] is True
        assert r["table_caption"] == "Quarterly Sales"
        assert r["table_markdown"], "block-0 row chunk lost table_markdown"
        assert "Region" in r["table_markdown"] and "Revenue" in r["table_markdown"]

    def test_large_table_splits_into_multiple_lossless_blocks(self):
        """Large table → multiple token-bounded table_row blocks; lossless cover.

        Replaces the removed summary-truncation / row-gate / group_size tests:
        instead of capping or summarising a big table, the chunker splits it into
        several ``table_row`` blocks, each restating the header and packing whole
        body rows under the token budget. This asserts the load-bearing new-world
        invariants:

          * NO ``table_summary`` chunk_type emitted;
          * a 12-body-row table at a tiny token budget yields >1 block;
          * ``table_block_total`` is consistent across blocks and equals the
            block count; block indices are a contiguous 0..N-1 sequence;
          * the header row is restated in EVERY block (never header-blind);
          * the blocks cover every body row exactly once (lossless, no dup, no
            gap) — verified from ``table_row_block_start`` / ``..._count``;
          * only block 0 carries ``table_markdown``;
          * the shared ``table_group_id`` reaches Weaviate properties on every
            block (so query-time expansion can rejoin them).
        """
        from src.vector_db.weaviate.store import add_documents

        n_body = 12
        cells = [["ID", "Name"]] + [[str(i), f"row{i}"] for i in range(n_body)]
        tbl = _make_table_artifact(
            cells=cells,
            table_id="table-1",
            self_ref="#/tables/0",
            caption="Quarterly Sales",
            section_path="Results > FY24",
        )
        table_chunk = _make_raw_chunk(tbl.markdown, headings=["Results", "FY24"])

        # Tiny budget so the stubbed-tokenizer (~1 token/row) counter forces a
        # multi-block split.
        adaptive = _real_adaptive_chunks([tbl], [table_chunk], max_tokens=3)
        types = [c.extra_metadata.get("chunk_type") for c in adaptive]
        assert types.count("table_summary") == 0, types
        blocks = [
            c for c in adaptive if c.extra_metadata.get("chunk_type") == "table_row"
        ]
        assert len(blocks) > 1, (
            f"large table should split into multiple blocks, got {len(blocks)}"
        )

        # Order by block index and assert the block-index invariants.
        blocks.sort(key=lambda c: c.extra_metadata["table_row_block_index"])
        indices = [c.extra_metadata["table_row_block_index"] for c in blocks]
        assert indices == list(range(len(blocks))), indices
        totals = {c.extra_metadata["table_block_total"] for c in blocks}
        assert totals == {len(blocks)}, totals

        # Header restated in every block; markdown only on block 0.
        header_md = "| ID | Name |"
        for c in blocks:
            assert header_md in c.text, (
                "block is header-blind; header must be restated in every block"
            )
        md_blocks = [
            c.extra_metadata["table_row_block_index"]
            for c in blocks
            if c.extra_metadata.get("table_markdown")
        ]
        assert md_blocks == [0], (
            f"table_markdown must be set on block 0 only, got blocks {md_blocks}"
        )

        # Lossless coverage: every body row appears in exactly one block.
        covered: list[int] = []
        for c in blocks:
            start = c.extra_metadata["table_row_block_start"]
            count = c.extra_metadata["table_row_block_count"]
            covered.extend(range(start, start + count))
        assert sorted(covered) == list(range(n_body)), (
            f"rows not covered exactly once: {sorted(covered)}"
        )
        assert len(covered) == len(set(covered)), "a body row was duplicated across blocks"

        # Shared table_group_id reaches Weaviate properties on every block.
        processed = _run_chunking_node(adaptive)
        col, captured = _capture_add_documents(_ALL_TABLE_PROPS)
        client = _make_client(col)
        add_documents(
            client,
            texts=[pc.text for pc in processed],
            embeddings=[[0.0] for _ in processed],
            metadatas=[pc.metadata for pc in processed],
        )
        store_rows = [
            c["properties"]
            for c in captured
            if c["properties"].get("chunk_type") == "table_row"
        ]
        assert len(store_rows) == len(blocks), (
            f"every block must reach the store: {len(store_rows)} vs {len(blocks)}"
        )
        assert all(p["table_group_id"] == "#/tables/0" for p in store_rows), (
            "a block reached Weaviate without the shared table_group_id"
        )
        # Exactly one stored block carries the full markdown (block 0).
        with_md = [p for p in store_rows if p["table_markdown"]]
        assert len(with_md) == 1, "exactly one block (block 0) should carry table_markdown"
        assert "ID" in with_md[0]["table_markdown"]

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
