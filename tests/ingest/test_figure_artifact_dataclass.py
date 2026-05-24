"""Tests for the FigureArtifact dataclass contract.

Mirrors ``test_table_artifact.py`` style: pin field defaults so downstream
consumers (Weaviate properties, xref retrieval, telemetry) can rely on the
schema; round-trip through a construction so attribute access stays stable.
"""

from __future__ import annotations

from dataclasses import fields

from src.ingest.support.parser_base import FigureArtifact, ParseResult


def test_figure_artifact_field_defaults_empty():
    fig = FigureArtifact()
    assert fig.document_id == ""
    assert fig.caption == ""
    assert fig.caption_label == ""
    assert fig.section_path == ""
    assert fig.page_no == 0
    assert fig.self_ref == ""
    assert fig.image_uri == ""


def test_figure_artifact_field_names_pinned():
    # Pin the schema so storage/retrieval teams have a stable contract.
    expected = {
        "document_id",
        "caption",
        "caption_label",
        "section_path",
        "page_no",
        "self_ref",
        "image_uri",
    }
    assert {f.name for f in fields(FigureArtifact)} == expected


def test_figure_artifact_roundtrip_construction():
    fig = FigureArtifact(
        document_id="esp32-s3_datasheet_en",
        caption="Figure 4-1: SoC block diagram",
        caption_label="Figure 4-1",
        section_path="Chapter 4 > Functional Description",
        page_no=42,
        self_ref="#/pictures/3",
        image_uri="data:image/png;base64,iVBOR...",
    )
    assert fig.document_id == "esp32-s3_datasheet_en"
    assert fig.caption_label == "Figure 4-1"
    assert fig.section_path == "Chapter 4 > Functional Description"
    assert fig.page_no == 42
    assert fig.self_ref == "#/pictures/3"
    assert fig.image_uri.startswith("data:image/png")


def test_parse_result_figures_default_empty():
    pr = ParseResult(markdown="x", headings=[], has_figures=False, page_count=0)
    assert pr.figures == []


def test_parse_result_figures_populated():
    fig = FigureArtifact(document_id="d", caption="Figure 1: x", caption_label="Figure 1")
    pr = ParseResult(
        markdown="x",
        headings=[],
        has_figures=True,
        page_count=1,
        figures=[fig],
    )
    assert len(pr.figures) == 1
    assert pr.figures[0].caption_label == "Figure 1"
