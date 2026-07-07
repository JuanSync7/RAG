"""Tests for page/section provenance on chunks (FR-3212).

Contracts:
- PageRef dataclass exposes page_no, page_label, optional bbox.
- Chunk has optional page_ref and heading_path fields with safe defaults.
- chunk_with_markdown populates heading_path from section_path.
- Docling chunk meta with provenance produces a populated PageRef.
- Docling chunk meta without provenance produces page_ref=None.
- TableArtifact carries optional page_ref derived from TableItem.prov.
- chunking_node legacy fallback path emits heading_path and degrades to no
  page_no when the parser does not provide page provenance.
- chunking_node parser-abstraction path forwards page_no into ProcessedChunk
  metadata when the parser supplies a page_ref.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.ingest.support.parser_base import (
    Chunk,
    PageRef,
    ParseResult,
    TableArtifact,
    chunk_with_markdown,
)


def test_page_ref_defaults():
    p = PageRef(page_no=3)
    assert p.page_no == 3
    assert p.page_label == ""
    assert p.bbox is None


def test_page_ref_with_bbox():
    p = PageRef(page_no=1, page_label="i", bbox=(0.0, 0.0, 100.0, 200.0))
    assert p.bbox == (0.0, 0.0, 100.0, 200.0)
    assert p.page_label == "i"


def test_chunk_default_provenance_fields():
    c = Chunk(
        text="x", section_path="", heading="", heading_level=0, chunk_index=0,
    )
    assert c.heading_path == []
    assert c.page_ref is None


def test_table_artifact_default_page_ref_none():
    t = TableArtifact(
        table_id="t1", markdown="| a |", cells=[["a"]], num_rows=1, num_cols=1,
    )
    assert t.page_ref is None


def test_table_artifact_with_page_ref():
    t = TableArtifact(
        table_id="t1", markdown="| a |", cells=[["a"]], num_rows=1, num_cols=1,
        page_ref=PageRef(page_no=5),
    )
    assert t.page_ref.page_no == 5


def test_chunk_with_markdown_emits_heading_path_field():
    """chunk_with_markdown chunks always carry a heading_path list (possibly empty)
    and never embed the ' > ' separator inside list items."""
    pr = ParseResult(
        markdown="# Top\n\nbody text.\n",
        headings=["Top"],
        has_figures=False,
        page_count=0,
    )
    cfg = SimpleNamespace(chunk_size=512, chunk_overlap=0)
    chunks = chunk_with_markdown(pr, cfg)
    assert chunks
    for c in chunks:
        assert isinstance(c.heading_path, list)
        assert all(isinstance(h, str) and " > " not in h for h in c.heading_path)


def test_chunk_with_markdown_page_ref_is_none_for_text_input():
    pr = ParseResult(markdown="# H\n\nbody", headings=["H"], has_figures=False, page_count=0)
    cfg = SimpleNamespace(chunk_size=512, chunk_overlap=0)
    chunks = chunk_with_markdown(pr, cfg)
    for c in chunks:
        assert c.page_ref is None


def test_page_ref_from_chunk_meta_with_provenance():
    from src.ingest.support.docling import _page_ref_from_chunk_meta

    bbox = SimpleNamespace(l=10.0, t=20.0, r=110.0, b=220.0)
    prov = SimpleNamespace(page_no=4, bbox=bbox)
    item = SimpleNamespace(prov=[prov])
    meta = SimpleNamespace(doc_items=[item])

    page_ref = _page_ref_from_chunk_meta(meta)
    assert page_ref is not None
    assert page_ref.page_no == 4
    assert page_ref.bbox == (10.0, 20.0, 110.0, 220.0)


def test_page_ref_from_chunk_meta_without_provenance():
    from src.ingest.support.docling import _page_ref_from_chunk_meta

    assert _page_ref_from_chunk_meta(None) is None
    assert _page_ref_from_chunk_meta(SimpleNamespace(doc_items=[])) is None
    item = SimpleNamespace(prov=[])
    assert _page_ref_from_chunk_meta(SimpleNamespace(doc_items=[item])) is None


def test_page_ref_from_table_item():
    from src.ingest.support.docling import _page_ref_from_table_item

    tbl = SimpleNamespace(prov=[SimpleNamespace(page_no=7)])
    p = _page_ref_from_table_item(tbl)
    assert p is not None and p.page_no == 7

    assert _page_ref_from_table_item(SimpleNamespace(prov=[])) is None


def test_page_ref_from_table_item_decodes_bbox():
    """Regression for DEFECT-2 from real-datasheet soak: TableItem provenance
    carries a bbox just like ChunkMeta does — must be decoded into PageRef.bbox
    rather than hardcoded to None (which defeated the citation-bbox plumbing).
    """
    from src.ingest.support.docling import _page_ref_from_table_item

    bbox = SimpleNamespace(l=12.5, t=30.0, r=200.5, b=400.0)
    tbl = SimpleNamespace(prov=[SimpleNamespace(page_no=7, bbox=bbox)])
    p = _page_ref_from_table_item(tbl)
    assert p is not None
    assert p.page_no == 7
    assert p.bbox == (12.5, 30.0, 200.5, 400.0)


def test_page_ref_from_table_item_x0y0_bbox_variant():
    """Some Docling versions expose x0/y0/x1/y1 instead of l/t/r/b."""
    from src.ingest.support.docling import _page_ref_from_table_item

    bbox = SimpleNamespace(x0=1.0, y0=2.0, x1=3.0, y1=4.0)
    tbl = SimpleNamespace(prov=[SimpleNamespace(page_no=2, bbox=bbox)])
    p = _page_ref_from_table_item(tbl)
    assert p is not None and p.bbox == (1.0, 2.0, 3.0, 4.0)


def test_chunking_node_legacy_fallback_emits_heading_path():
    """Regex-fallback path: no parse_result; chunks still carry heading_path
    derived from markdown headings, page_no absent (graceful degradation)."""
    from src.ingest.embedding.nodes.chunking import chunking_node

    runtime = SimpleNamespace(
        config=SimpleNamespace(
            chunker="markdown",
            chunk_size=512,
            chunk_overlap=0,
            semantic_chunking=False,
        ),
        embedder=None,
    )
    state = {
        "raw_text": "# Top\n\n## Sub\n\nbody text here.\n",
        "cleaned_text": "# Top\n\n## Sub\n\nbody text here.\n",
        "source_name": "doc.md",
        "source_uri": "doc.md",
        "source_key": "doc.md",
        "source_id": "x",
        "connector": "local",
        "source_version": "1",
        "runtime": runtime,
        "processing_log": [],
        "trace_id": "t",
    }
    out = chunking_node(state)
    assert out.get("chunks"), out
    # Every legacy-path chunk must carry chunk_type="text" and a heading_path field
    for c in out["chunks"]:
        assert c.metadata.get("chunk_type") == "text"
        assert "heading_path" in c.metadata
        assert "page_no" not in c.metadata  # graceful degradation


def test_chunking_node_parser_path_propagates_page_ref():
    """When parse_result+parser_instance are present and chunks carry page_ref,
    page_no flows through to ProcessedChunk metadata."""
    from src.ingest.embedding.nodes.chunking import chunking_node

    fake_chunks = [
        Chunk(
            text="paragraph one",
            section_path="Top > Sub",
            heading="Sub",
            heading_level=2,
            chunk_index=0,
            extra_metadata={},
            heading_path=["Top", "Sub"],
            page_ref=PageRef(page_no=2, page_label="2"),
        ),
    ]
    pr = ParseResult(markdown="x", headings=["Top", "Sub"], has_figures=False, page_count=3)

    class _FakeParser:
        def chunk(self, _pr):
            return fake_chunks

    runtime = SimpleNamespace(
        config=SimpleNamespace(
            chunker="native",
            chunk_size=512,
            chunk_overlap=0,
            semantic_chunking=False,
        ),
        embedder=None,
        db_client=None,
    )
    state = {
        "raw_text": "x",
        "cleaned_text": "x",
        "source_name": "doc.pdf",
        "source_uri": "doc.pdf",
        "source_key": "doc.pdf",
        "source_id": "x",
        "connector": "local",
        "source_version": "1",
        "runtime": runtime,
        "parse_result": pr,
        "parser_instance": _FakeParser(),
        "processing_log": [],
        "trace_id": "t",
    }
    out = chunking_node(state)
    assert out.get("chunks"), out
    c0 = out["chunks"][0]
    assert c0.metadata["page_no"] == 2
    assert c0.metadata["page_label"] == "2"
    assert c0.metadata["heading_path"] == ["Top", "Sub"]
    assert c0.metadata["chunk_type"] == "text"
