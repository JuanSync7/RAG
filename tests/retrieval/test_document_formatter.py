"""Unit tests for document formatter module."""

import pytest
from dataclasses import dataclass

from src.retrieval.generation.nodes.document_formatter import (
    format_context,
    _detect_version_conflicts,
    _extract_metadata_header,
    FormattedContext,
    VersionConflict,
)


@dataclass
class MockRankedResult:
    """Lightweight mock for RankedResult."""
    text: str
    score: float
    metadata: dict


class TestFormatContext:
    """Tests for format_context."""

    def test_empty_results(self):
        result = format_context([])
        assert result.context_string == ""
        assert result.chunk_count == 0
        assert result.version_conflicts == []

    def test_single_chunk(self):
        results = [MockRankedResult(
            text="Some document text here.",
            score=0.85,
            metadata={"source": "spec.pdf", "heading": "Section 1"},
        )]
        result = format_context(results)
        assert result.chunk_count == 1
        assert "[1] (relevance: 85%)" in result.context_string
        assert "Source: spec.pdf" in result.context_string
        assert "Some document text here." in result.context_string

    def test_stamps_1based_citation_index_on_metadata(self):
        """Each result's metadata gets citation_index = its 1-based position, so
        the [N] the model cites maps to the Nth returned source."""
        results = [
            MockRankedResult(text="a", score=0.9, metadata={"source": "x.pdf"}),
            MockRankedResult(text="b", score=0.8, metadata={"source": "y.pdf"}),
            MockRankedResult(text="c", score=0.7, metadata={"source": "z.pdf"}),
        ]
        format_context(results)
        assert [r.metadata["citation_index"] for r in results] == [1, 2, 3]
        # the stamped index matches the [N] header in the formatted context
        assert "[2]" in format_context(results).context_string

    def test_multiple_chunks(self):
        results = [
            MockRankedResult(text="Chunk one.", score=0.9, metadata={}),
            MockRankedResult(text="Chunk two.", score=0.7, metadata={}),
        ]
        result = format_context(results)
        assert result.chunk_count == 2
        assert "[1]" in result.context_string
        assert "[2]" in result.context_string

    def test_no_scores(self):
        results = [MockRankedResult(text="Text.", score=0.5, metadata={})]
        result = format_context(results, include_scores=False)
        assert "relevance" not in result.context_string

    def test_version_conflict_warning(self):
        results = [
            MockRankedResult(
                text="Version 1 text.",
                score=0.9,
                metadata={"source": "power_spec.pdf", "source_version": "v1"},
            ),
            MockRankedResult(
                text="Version 2 text.",
                score=0.8,
                metadata={"source": "power_spec.pdf", "source_version": "v2"},
            ),
        ]
        result = format_context(results)
        assert len(result.version_conflicts) == 1
        assert "VERSION CONFLICT" in result.context_string
        assert "power_spec" in result.context_string


class TestDetectVersionConflicts:
    """Tests for _detect_version_conflicts."""

    def test_no_conflicts(self):
        results = [
            MockRankedResult(text="", score=0.0, metadata={
                "source": "a.pdf", "source_version": "v1"
            }),
            MockRankedResult(text="", score=0.0, metadata={
                "source": "a.pdf", "source_version": "v1"
            }),
        ]
        assert _detect_version_conflicts(results) == []

    def test_conflict_detected(self):
        results = [
            MockRankedResult(text="", score=0.0, metadata={
                "source": "spec.pdf", "source_version": "v1"
            }),
            MockRankedResult(text="", score=0.0, metadata={
                "source": "spec.pdf", "source_version": "v2"
            }),
        ]
        conflicts = _detect_version_conflicts(results)
        assert len(conflicts) == 1
        assert conflicts[0].spec_stem == "spec"
        assert set(conflicts[0].versions) == {"v1", "v2"}

    def test_different_documents_no_conflict(self):
        results = [
            MockRankedResult(text="", score=0.0, metadata={
                "source": "a.pdf", "source_version": "v1"
            }),
            MockRankedResult(text="", score=0.0, metadata={
                "source": "b.pdf", "source_version": "v2"
            }),
        ]
        assert _detect_version_conflicts(results) == []

    def test_no_version_metadata(self):
        results = [
            MockRankedResult(text="", score=0.0, metadata={"source": "a.pdf"}),
            MockRankedResult(text="", score=0.0, metadata={"source": "a.pdf"}),
        ]
        assert _detect_version_conflicts(results) == []

    def test_spec_id_used_as_stem(self):
        results = [
            MockRankedResult(text="", score=0.0, metadata={
                "source_id": "PWR-001", "source_version": "v1"
            }),
            MockRankedResult(text="", score=0.0, metadata={
                "source_id": "PWR-001", "source_version": "v3"
            }),
        ]
        conflicts = _detect_version_conflicts(results)
        assert len(conflicts) == 1
        assert conflicts[0].spec_stem == "PWR-001"


class TestExtractMetadataHeader:
    """Tests for _extract_metadata_header."""

    def test_empty_metadata(self):
        assert _extract_metadata_header({}) == ""

    def test_full_metadata(self):
        header = _extract_metadata_header({
            "source": "power_spec.pdf",
            "source_version": "v3",
            "date": "2025-01-15",
            "heading": "3.2 Voltage Rails",
            "domain": "physical_design",
        })
        assert "Source: power_spec.pdf" in header
        assert "Version: v3" in header
        assert "Date: 2025-01-15" in header
        assert "Section: 3.2 Voltage Rails" in header
        assert "Domain: physical_design" in header

    def test_partial_metadata(self):
        header = _extract_metadata_header({"source": "test.pdf"})
        assert "Source: test.pdf" in header
        assert "Version" not in header

    def test_list_domain(self):
        header = _extract_metadata_header({
            "domain": ["digital", "timing"],
        })
        assert "Domain: digital, timing" in header


# ---------------------------------------------------------------------------
# Additional metadata edge-case tests
# ---------------------------------------------------------------------------


class TestFormatContextMetadataCases:
    """Edge-case tests for format_context metadata handling."""

    def test_chunks_with_metadata_include_source_attribution(self):
        """Chunks that have a 'source' field must include it in the formatted output."""
        results = [
            MockRankedResult(
                text="Engine specifications text.",
                score=0.8,
                metadata={"source": "engine_spec.pdf", "heading": "Section 4"},
            )
        ]
        result = format_context(results)
        assert "Source: engine_spec.pdf" in result.context_string
        assert "Engine specifications text." in result.context_string

    def test_chunks_without_metadata_do_not_raise_key_error(self):
        """Chunks with completely empty metadata must not raise KeyError."""
        results = [
            MockRankedResult(text="Plain text chunk.", score=0.6, metadata={})
        ]
        result = format_context(results)
        assert "Plain text chunk." in result.context_string
        assert result.chunk_count == 1

    def test_empty_chunk_list_returns_empty_string(self):
        """format_context([]) must return an empty context_string."""
        result = format_context([])
        assert result.context_string == ""
        assert result.chunk_count == 0
