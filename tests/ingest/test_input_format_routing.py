# @summary
# Tests that different input file formats route correctly through the ingestion pipeline.
# Covers: Docling-supported formats → hybrid_chunker, unsupported formats → markdown fallback,
#   format-based fallback regardless of docling_strict setting.
# Deps: pytest, unittest.mock, src.ingest.doc_processing.nodes.structure_detection,
#       src.ingest.support.docling, src.ingest.common.types
# @end-summary
"""Input format routing tests for the ingestion pipeline.

Validates that:
  - Docling-supported formats (.md, .pdf, .docx, .html, .csv, .pptx, .xlsx,
    .asciidoc, .latex) go through the Docling native path (hybrid_chunker).
  - Unsupported formats (.txt, .log, .json, .yaml) fall back to the regex/markdown
    pipeline — even when docling_strict=True.
  - The "format not allowed" error is distinguished from real parser failures.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from src.ingest.common.types import IngestionConfig, Runtime
from src.ingest.doc_processing.nodes.structure_detection import structure_detection_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    source_path: str,
    source_name: str,
    raw_text: str = "Some raw text content for testing.",
    enable_docling: bool = True,
    docling_strict: bool = True,
    vlm_mode: str = "disabled",
) -> dict:
    """Build a minimal pipeline state for structure_detection_node."""
    config = IngestionConfig(
        enable_docling_parser=enable_docling,
        docling_strict=docling_strict,
        vlm_mode=vlm_mode,
    )
    runtime = Runtime(
        config=config,
        embedder=MagicMock(),
        weaviate_client=MagicMock(),
        kg_builder=None,
    )
    return {
        "raw_text": raw_text,
        "source_path": source_path,
        "source_name": source_name,
        "runtime": runtime,
        "errors": [],
        "processing_log": [],
    }


def _make_successful_parse_result(markdown="# Parsed\n\nContent."):
    """Build a mock DoclingParseResult for a successful parse."""
    from src.ingest.support.docling import DoclingParseResult

    mock_doc = MagicMock()
    mock_doc.export_to_markdown.return_value = markdown
    mock_doc.pictures = []
    return DoclingParseResult(
        text_markdown=markdown,
        has_figures=False,
        figures=[],
        headings=["Parsed"],
        parser_model="docling-parse-v2",
        docling_document=mock_doc,
    )


# ---------------------------------------------------------------------------
# Docling-supported formats → native path
# ---------------------------------------------------------------------------


DOCLING_SUPPORTED_EXTENSIONS = [
    ".md", ".pdf", ".docx", ".pptx", ".html",
    ".csv", ".xlsx", ".asciidoc", ".latex",
]


class TestDoclingNativePath:
    """Docling-supported formats should go through the native path."""

    @pytest.mark.parametrize("ext", DOCLING_SUPPORTED_EXTENSIONS)
    def test_supported_format_uses_docling_path(self, ext):
        """Files with Docling-supported extensions should produce docling_document."""
        state = _make_state(
            source_path=f"/tmp/test_doc{ext}",
            source_name=f"test_doc{ext}",
        )
        mock_result = _make_successful_parse_result()

        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            return_value=mock_result,
        ):
            update = structure_detection_node(state)

        assert update["structure"]["docling_document_available"] is True
        assert "docling_document" in update
        assert "should_skip" not in update

    @pytest.mark.parametrize("ext", DOCLING_SUPPORTED_EXTENSIONS)
    def test_supported_format_calls_parse_with_docling(self, ext):
        """parse_with_docling should be called for supported formats."""
        state = _make_state(
            source_path=f"/tmp/test_doc{ext}",
            source_name=f"test_doc{ext}",
        )
        mock_result = _make_successful_parse_result()

        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            return_value=mock_result,
        ) as mock_parse:
            structure_detection_node(state)

        mock_parse.assert_called_once()


# ---------------------------------------------------------------------------
# Unsupported formats → fallback path
# ---------------------------------------------------------------------------


UNSUPPORTED_EXTENSIONS = [".txt", ".log", ".json", ".yaml", ".yml", ".ini", ".cfg", ".toml"]


class TestFallbackPath:
    """Unsupported formats should fall back to regex heuristics, not fail."""

    @pytest.mark.parametrize("ext", UNSUPPORTED_EXTENSIONS)
    def test_unsupported_format_falls_back_when_strict(self, ext):
        """Unsupported format with docling_strict=True should NOT fail.

        Format errors are not parser bugs — strict mode should only fail on
        real parser errors, not on unsupported file types.
        """
        state = _make_state(
            source_path=f"/tmp/data{ext}",
            source_name=f"data{ext}",
            docling_strict=True,
            raw_text="# Heading\n\nSome content with Figure 1 reference.",
        )

        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            side_effect=RuntimeError(
                f"Docling conversion failed for /tmp/data{ext}: "
                f"File format not allowed: data{ext}"
            ),
        ):
            update = structure_detection_node(state)

        # Should NOT skip — should fall back gracefully.
        assert "should_skip" not in update or update.get("should_skip") is not True
        assert update["structure"]["docling_document_available"] is False
        assert "docling_document" not in update
        # Processing log should indicate success (fallback worked).
        assert any("structure_detection:ok" in entry for entry in update["processing_log"])

    @pytest.mark.parametrize("ext", UNSUPPORTED_EXTENSIONS)
    def test_unsupported_format_falls_back_when_not_strict(self, ext):
        """Unsupported format with docling_strict=False should also fall back."""
        state = _make_state(
            source_path=f"/tmp/data{ext}",
            source_name=f"data{ext}",
            docling_strict=False,
            raw_text="Some plain text content.",
        )

        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            side_effect=RuntimeError(
                f"Docling conversion failed for /tmp/data{ext}: "
                f"File format not allowed: data{ext}"
            ),
        ):
            update = structure_detection_node(state)

        assert update["structure"]["docling_document_available"] is False
        assert "should_skip" not in update or update.get("should_skip") is not True

    @pytest.mark.parametrize("ext", UNSUPPORTED_EXTENSIONS)
    def test_unsupported_format_preserves_raw_text(self, ext):
        """Fallback path should keep the original raw_text, not replace it."""
        raw = "Original raw text content."
        state = _make_state(
            source_path=f"/tmp/data{ext}",
            source_name=f"data{ext}",
            raw_text=raw,
        )

        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            side_effect=RuntimeError(
                f"Docling conversion failed for /tmp/data{ext}: "
                f"File format not allowed: data{ext}"
            ),
        ):
            update = structure_detection_node(state)

        assert update["raw_text"] == raw

    def test_unsupported_format_extracts_headings_via_regex(self):
        """Fallback path should still extract headings via regex."""
        state = _make_state(
            source_path="/tmp/notes.txt",
            source_name="notes.txt",
            raw_text="# Introduction\n\nSome text.\n\n## Methods\n\nMore text.",
        )

        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            side_effect=RuntimeError(
                "Docling conversion failed for /tmp/notes.txt: "
                "File format not allowed: notes.txt"
            ),
        ):
            update = structure_detection_node(state)

        assert update["structure"]["heading_count"] >= 2

    def test_unsupported_format_extracts_figures_via_regex(self):
        """Fallback path should still detect figure references via regex."""
        state = _make_state(
            source_path="/tmp/notes.txt",
            source_name="notes.txt",
            raw_text="As shown in Figure 1, the results are clear. See also Fig. 2.",
        )

        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            side_effect=RuntimeError(
                "Docling conversion failed for /tmp/notes.txt: "
                "File format not allowed: notes.txt"
            ),
        ):
            update = structure_detection_node(state)

        assert update["structure"]["has_figures"] is True
        assert len(update["structure"]["figures"]) >= 2


# ---------------------------------------------------------------------------
# Real parser errors should still respect docling_strict
# ---------------------------------------------------------------------------


class TestRealParserErrors:
    """Real parser failures (not format errors) should respect docling_strict."""

    def test_real_error_with_strict_skips(self):
        """A real parser error in strict mode should set should_skip=True."""
        state = _make_state(
            source_path="/tmp/corrupt.pdf",
            source_name="corrupt.pdf",
            docling_strict=True,
        )

        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            side_effect=RuntimeError("Segfault in PDF parser"),
        ):
            update = structure_detection_node(state)

        assert update.get("should_skip") is True
        assert any("docling_parse_failed" in e for e in update["errors"])

    def test_real_error_without_strict_falls_back(self):
        """A real parser error in non-strict mode should fall back, not skip."""
        state = _make_state(
            source_path="/tmp/corrupt.pdf",
            source_name="corrupt.pdf",
            docling_strict=False,
            raw_text="# Fallback heading\n\nContent.",
        )

        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            side_effect=RuntimeError("Segfault in PDF parser"),
        ):
            update = structure_detection_node(state)

        assert "should_skip" not in update or update.get("should_skip") is not True
        assert update["structure"]["docling_document_available"] is False

    def test_format_error_distinguished_from_parser_error(self):
        """Format errors and parser errors should produce different outcomes in strict mode."""
        # Format error → fallback (no skip)
        format_state = _make_state(
            source_path="/tmp/data.txt",
            source_name="data.txt",
            docling_strict=True,
        )
        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            side_effect=RuntimeError("File format not allowed: data.txt"),
        ):
            format_update = structure_detection_node(format_state)

        # Parser error → skip
        parser_state = _make_state(
            source_path="/tmp/data.pdf",
            source_name="data.pdf",
            docling_strict=True,
        )
        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            side_effect=RuntimeError("Internal parser crash"),
        ):
            parser_update = structure_detection_node(parser_state)

        # Format error: no skip, parser error: skip
        assert format_update.get("should_skip") is not True
        assert parser_update.get("should_skip") is True


# ---------------------------------------------------------------------------
# Docling disabled → always regex
# ---------------------------------------------------------------------------


class TestDoclingDisabled:
    """When docling is disabled, all formats use regex heuristics."""

    @pytest.mark.parametrize("ext", [".md", ".pdf", ".txt", ".docx"])
    def test_disabled_docling_uses_regex_for_all_formats(self, ext):
        """With enable_docling_parser=False, no format hits Docling."""
        state = _make_state(
            source_path=f"/tmp/doc{ext}",
            source_name=f"doc{ext}",
            enable_docling=False,
            raw_text="# Title\n\nSee Figure 1 for details.",
        )

        # parse_with_docling should never be called.
        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
        ) as mock_parse:
            update = structure_detection_node(state)

        mock_parse.assert_not_called()
        assert update["structure"]["docling_document_available"] is False
        assert update["structure"]["heading_count"] >= 1
        assert update["structure"]["has_figures"] is True
