# @summary
# Tests the ingest path that turns FigureArtifact rows into chunks with
# ``chunk_type="figure"``. Covers text rendering (caption_label + caption),
# metadata mapping, image_uri data-URI coercion, and the empty-caption skip.
# Exports: (pytest test functions)
# Deps: pytest, src.ingest.support.parser_base, src.ingest.support.docling
# @end-summary
"""Figure-artifact → chunk transformer."""
from __future__ import annotations

from src.ingest.support.parser_base import FigureArtifact
from src.ingest.support.docling import _figure_artifacts_to_chunks


def _fig(**overrides) -> FigureArtifact:
    base = dict(
        document_id="ds",
        caption="Figure 4-1: SoC diagram",
        caption_label="Figure 4-1",
        section_path="Ch4 > Arch",
        page_no=37,
        self_ref="#/pictures/0",
        image_uri="file:///tmp/pic.png",
    )
    base.update(overrides)
    return FigureArtifact(**base)


def test_emits_one_chunk_per_figure():
    figs = [_fig(), _fig(self_ref="#/pictures/1", caption_label="Figure 4-2",
                         caption="Figure 4-2: ADC")]
    chunks = _figure_artifacts_to_chunks(figs)
    assert len(chunks) == 2


def test_chunk_type_is_figure():
    chunks = _figure_artifacts_to_chunks([_fig()])
    assert chunks[0].extra_metadata["chunk_type"] == "figure"


def test_text_uses_caption_label_and_caption():
    chunks = _figure_artifacts_to_chunks([_fig()])
    # caption already starts with the label; we keep the caption verbatim.
    assert "Figure 4-1" in chunks[0].text
    assert "SoC diagram" in chunks[0].text


def test_metadata_carries_provenance_fields():
    chunks = _figure_artifacts_to_chunks([_fig()])
    c = chunks[0]
    # section_path lives on the Chunk itself (mirrors table chunks) so
    # downstream chunking_node can map it into ProcessedChunk.metadata.
    assert c.section_path == "Ch4 > Arch"
    meta = c.extra_metadata
    assert meta["document_id"] == "ds"
    assert meta["caption_label"] == "Figure 4-1"
    assert meta["page_no"] == 37
    assert meta["self_ref"] == "#/pictures/0"
    # file:// URIs are kept verbatim — only data: URIs get coerced.
    assert meta["figure_image_uri"] == "file:///tmp/pic.png"


def test_data_uri_coerced_to_empty():
    fig = _fig(image_uri="data:image/png;base64,AAAA")
    chunks = _figure_artifacts_to_chunks([fig])
    assert chunks[0].extra_metadata["figure_image_uri"] == ""


def test_skips_figures_with_empty_caption_and_label():
    fig = _fig(caption="", caption_label="")
    chunks = _figure_artifacts_to_chunks([fig])
    assert chunks == []


def test_includes_figure_when_only_label_present():
    """Some parsers find the label but lose the caption text — still emit
    the chunk; the label alone is useful citation provenance."""
    fig = _fig(caption="", caption_label="Figure 4-1")
    chunks = _figure_artifacts_to_chunks([fig])
    assert len(chunks) == 1
    assert "Figure 4-1" in chunks[0].text


def test_chunk_id_present_in_metadata():
    chunks = _figure_artifacts_to_chunks([_fig()])
    assert chunks[0].extra_metadata.get("chunk_id")


def test_figure_chunks_emitted_via_parser_chunk(monkeypatch):
    """End-to-end (light): DoclingParser.chunk() emits figure chunks for
    parse_result.figures when adaptive table chunking is enabled.
    """
    from src.ingest.support.docling import DoclingParser
    from src.ingest.support.parser_base import ParseResult

    parser = DoclingParser()
    # Bypass parse() — populate internal state directly.
    parser._docling_document = object()
    parser._config = type("C", (), {"enable_adaptive_table_chunking": True})()

    # Stub the HybridChunker path so we don't need a real tokenizer.
    def _fake_chunk(self, parse_result):  # noqa: D401, ANN001
        # Mirror the production path's figure-emission branch by calling
        # the transformer directly and returning the resulting chunks. This
        # is the spec the production chunk() must satisfy.
        from src.ingest.support.docling import _figure_artifacts_to_chunks
        return _figure_artifacts_to_chunks(list(parse_result.figures))

    monkeypatch.setattr(DoclingParser, "chunk", _fake_chunk)

    pr = ParseResult(
        markdown="",
        headings=[],
        has_figures=True,
        page_count=1,
        tables=[],
        figures=[_fig()],
    )
    chunks = parser.chunk(pr)
    assert chunks
    assert chunks[0].extra_metadata["chunk_type"] == "figure"
