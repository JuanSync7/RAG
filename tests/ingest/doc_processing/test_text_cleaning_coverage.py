# @summary
# Coverage tests for the text_cleaning_node pipeline node.
# Exports: TestTextCleaningError, TestTextCleaningBoundary, TestTextCleaningErrorScenarios
# Deps: pytest, unittest.mock, src.ingest.doc_processing.nodes.text_cleaning,
#       src.ingest.common.types
# @end-summary

"""Coverage tests for text_cleaning_node.

Tests are grouped into three classes:
- TestTextCleaningError: exception propagation paths (no internal guard).
- TestTextCleaningBoundary: edge-case inputs (empty text, large text, whitespace-only).
- TestTextCleaningErrorScenarios: figure-note composition, formatting, and log recording.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.ingest.doc_processing.nodes.text_cleaning import text_cleaning_node
from src.ingest.common.types import IngestionConfig, Runtime


# ---------------------------------------------------------------------------
# State builder
# ---------------------------------------------------------------------------

def _make_state(raw_text="Hello world.", multimodal_notes=None, **overrides) -> dict:
    """Build a minimal pipeline state dict for text_cleaning_node tests.

    Args:
        raw_text: Value for ``state["raw_text"]``.
        multimodal_notes: Value for ``state["multimodal_notes"]``; defaults to ``[]``.
        **overrides: Additional keys merged into the returned state dict.

    Returns:
        A state dict compatible with ``DocumentProcessingState``.
    """
    config = IngestionConfig()
    runtime = Runtime(
        config=config,
        embedder=MagicMock(),
        weaviate_client=MagicMock(),
        kg_builder=None,
    )
    state = {
        "raw_text": raw_text,
        "multimodal_notes": multimodal_notes if multimodal_notes is not None else [],
        "runtime": runtime,
        "errors": [],
        "processing_log": [],
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Helper: patch target
# ---------------------------------------------------------------------------

_PATCH_TARGET = "src.ingest.doc_processing.nodes.text_cleaning.clean_document"


# ---------------------------------------------------------------------------
# TestTextCleaningError
# ---------------------------------------------------------------------------

class TestTextCleaningError:
    """Tests covering exception propagation — the node has no internal guard."""

    def test_text_cleaning_node_handles_clean_document_exception(self):
        """An exception from clean_document must propagate uncaught to the caller."""
        state = _make_state()
        with patch(_PATCH_TARGET, side_effect=TypeError("NoneType")):
            with pytest.raises(TypeError):
                text_cleaning_node(state)

    def test_text_cleaning_node_handles_none_raw_text(self):
        """Passing raw_text=None must raise some exception — documents the missing guard."""
        state = _make_state(raw_text=None)
        with pytest.raises(Exception):
            text_cleaning_node(state)


# ---------------------------------------------------------------------------
# TestTextCleaningBoundary
# ---------------------------------------------------------------------------

class TestTextCleaningBoundary:
    """Tests covering edge-case inputs at the boundary of valid values."""

    def test_text_cleaning_node_handles_empty_raw_text(self):
        """An empty raw_text must yield an empty cleaned_text when clean_document returns ''."""
        state = _make_state(raw_text="")
        with patch(_PATCH_TARGET, return_value=""):
            result = text_cleaning_node(state)

        assert result["cleaned_text"] == ""

    def test_text_cleaning_node_handles_empty_multimodal_notes(self):
        """An empty multimodal_notes list must not append a Figure Notes section."""
        state = _make_state(multimodal_notes=[])
        with patch(_PATCH_TARGET, return_value="cleaned"):
            result = text_cleaning_node(state)

        assert result["cleaned_text"] == "cleaned"

    def test_text_cleaning_node_handles_whitespace_only_raw_text(self):
        """A whitespace-only raw_text must yield '' when clean_document returns ''."""
        state = _make_state(raw_text="   \n  ")
        with patch(_PATCH_TARGET, return_value=""):
            result = text_cleaning_node(state)

        assert result["cleaned_text"] == ""

    def test_text_cleaning_node_handles_very_large_text(self):
        """A 1 MB text must be passed through without truncation."""
        large_text = "x" * 1_000_000
        state = _make_state(raw_text=large_text)
        with patch(_PATCH_TARGET, return_value=large_text):
            result = text_cleaning_node(state)

        assert len(result["cleaned_text"]) == 1_000_000


# ---------------------------------------------------------------------------
# TestTextCleaningErrorScenarios
# ---------------------------------------------------------------------------

class TestTextCleaningErrorScenarios:
    """Tests covering figure-note composition, formatting, and log recording."""

    def test_text_cleaning_node_appends_figure_notes_section(self):
        """Figure notes must be appended under a '## Figure Notes' heading."""
        state = _make_state(
            multimodal_notes=["Figure 1: clock distribution", "Figure 2: power grid"],
        )
        with patch(_PATCH_TARGET, return_value="Main text"):
            result = text_cleaning_node(state)

        assert "## Figure Notes" in result["cleaned_text"]
        assert "- Figure 1: clock distribution" in result["cleaned_text"]
        assert "- Figure 2: power grid" in result["cleaned_text"]

    def test_text_cleaning_node_figure_notes_format(self):
        """The figure notes section must match the exact '\n\n## Figure Notes\n- ...' format."""
        state = _make_state(
            multimodal_notes=["Figure 1: clock distribution", "Figure 2: power grid"],
        )
        with patch(_PATCH_TARGET, return_value="Main text"):
            result = text_cleaning_node(state)

        expected_suffix = (
            "\n\n## Figure Notes\n"
            "- Figure 1: clock distribution\n"
            "- Figure 2: power grid"
        )
        assert result["cleaned_text"].endswith(expected_suffix)

    def test_text_cleaning_node_preserves_heading_hierarchy(self):
        """clean_document must receive the raw text and its return value used as-is."""
        raw = "# H1\n## H2\n### H3"
        state = _make_state(raw_text=raw)
        with patch(_PATCH_TARGET, side_effect=lambda text: text):
            result = text_cleaning_node(state)

        assert "# H1" in result["cleaned_text"]
        assert "## H2" in result["cleaned_text"]
        assert "### H3" in result["cleaned_text"]

    def test_text_cleaning_node_removes_boilerplate(self):
        """The node must use the return value of clean_document, not the raw input."""
        state = _make_state(raw_text="CONFIDENTIAL " * 10)
        with patch(_PATCH_TARGET, return_value="Cleaned text without boilerplate."):
            result = text_cleaning_node(state)

        assert result["cleaned_text"] == "Cleaned text without boilerplate."

    def test_text_cleaning_node_normalizes_whitespace(self):
        """The node must propagate the normalized string returned by clean_document."""
        state = _make_state(raw_text="5   consecutive   spaces")
        with patch(_PATCH_TARGET, return_value="5 consecutive spaces"):
            result = text_cleaning_node(state)

        assert result["cleaned_text"] == "5 consecutive spaces"

    def test_text_cleaning_node_processing_log_records_ok(self):
        """A successful run must append 'text_cleaning:ok' as the last log entry."""
        state = _make_state()
        with patch(_PATCH_TARGET, return_value="ok"):
            result = text_cleaning_node(state)

        assert result["processing_log"][-1].endswith("text_cleaning:ok")
