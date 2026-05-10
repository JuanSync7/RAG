# @summary
# Coverage tests for the multimodal_processing_node pipeline node.
# Exports: TestMultimodalError, TestMultimodalBoundary, TestMultimodalErrorScenarios
# Deps: pytest, unittest.mock, src.ingest.doc_processing.nodes.multimodal_processing,
#       src.ingest.common.types
# @end-summary

"""Coverage tests for multimodal_processing_node.

Tests are grouped into three classes:
- TestMultimodalError: vision failure/strict/non-strict error paths.
- TestMultimodalBoundary: skip conditions and edge-case inputs.
- TestMultimodalErrorScenarios: note composition and structure telemetry.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.ingest.doc_processing.nodes.multimodal_processing import multimodal_processing_node
from src.ingest.common.types import IngestionConfig, Runtime


# ---------------------------------------------------------------------------
# State builder
# ---------------------------------------------------------------------------

def _make_state(
    enable_multimodal: bool = True,
    enable_vision: bool = False,
    vision_strict: bool = False,
    has_figures: bool = True,
    figures: list[str] | None = None,
    **config_overrides,
) -> dict:
    """Build a minimal pipeline state dict for multimodal_processing_node tests.

    Args:
        enable_multimodal: Toggle for ``enable_multimodal_processing`` config flag.
        enable_vision: Toggle for ``enable_vision_processing`` config flag.
        vision_strict: Toggle for ``vision_strict`` config flag.
        has_figures: Value for ``structure["has_figures"]``.
        figures: Figure list for ``structure["figures"]``; defaults to two entries.
        **config_overrides: Additional keyword args forwarded to ``IngestionConfig``.

    Returns:
        A state dict compatible with ``DocumentProcessingState``.
    """
    config = IngestionConfig(
        enable_multimodal_processing=enable_multimodal,
        enable_vision_processing=enable_vision,
        vision_strict=vision_strict,
        **config_overrides,
    )
    runtime = Runtime(
        config=config,
        embedder=MagicMock(),
        weaviate_client=MagicMock(),
        kg_builder=None,
    )
    effective_figures = figures if figures is not None else ["Figure 1", "Figure 2"]
    structure = {"has_figures": has_figures, "figures": effective_figures}
    return {
        "runtime": runtime,
        "structure": structure,
        "raw_text": "Some text with Figure 1 and Figure 2.",
        "source_path": "/tmp/test.pdf",
        "source_name": "test.pdf",
        "errors": [],
        "processing_log": [],
    }


# ---------------------------------------------------------------------------
# Helper: patch target
# ---------------------------------------------------------------------------

_PATCH_TARGET = "src.ingest.doc_processing.nodes.multimodal_processing.generate_vision_notes"


# ---------------------------------------------------------------------------
# TestMultimodalError
# ---------------------------------------------------------------------------

class TestMultimodalError:
    """Tests covering vision failure paths (strict and non-strict)."""

    def test_multimodal_node_returns_error_when_vision_strict_fails(self):
        """Strict mode: a vision exception must surface errors and set should_skip."""
        state = _make_state(
            enable_multimodal=True,
            enable_vision=True,
            vision_strict=True,
        )
        with patch(_PATCH_TARGET, side_effect=RuntimeError("VLM failed")):
            result = multimodal_processing_node(state)

        assert "errors" in result
        assert any("vision_processing_failed:" in e for e in result["errors"])
        assert result.get("should_skip") is True
        assert result["processing_log"][-1].endswith("multimodal_processing:failed")

    def test_multimodal_node_swallows_exception_when_vision_nonstrict(self):
        """Non-strict mode: a vision exception must NOT surface errors to the caller."""
        state = _make_state(
            enable_multimodal=True,
            enable_vision=True,
            vision_strict=False,
        )
        with patch(_PATCH_TARGET, side_effect=RuntimeError("VLM failed")):
            result = multimodal_processing_node(state)

        # Either no "errors" key at all, or the list is empty.
        assert not result.get("errors")
        assert "multimodal_notes" in result

    def test_multimodal_node_strict_failure_processing_log(self):
        """Strict mode: the last processing_log entry must record the failure token."""
        state = _make_state(
            enable_multimodal=True,
            enable_vision=True,
            vision_strict=True,
        )
        with patch(_PATCH_TARGET, side_effect=RuntimeError("VLM failed")):
            result = multimodal_processing_node(state)

        assert result["processing_log"][-1].endswith("multimodal_processing:failed")


# ---------------------------------------------------------------------------
# TestMultimodalBoundary
# ---------------------------------------------------------------------------

class TestMultimodalBoundary:
    """Tests covering skip conditions and edge-case inputs."""

    def test_multimodal_node_skipped_when_multimodal_disabled(self):
        """When multimodal processing is disabled the node returns only processing_log."""
        state = _make_state(enable_multimodal=False, has_figures=True)
        result = multimodal_processing_node(state)

        assert set(result.keys()) == {"processing_log"}
        assert result["processing_log"][-1].endswith("multimodal_processing:skipped")

    def test_multimodal_node_skipped_when_no_figures_detected(self):
        """When has_figures is False the node skips and returns only processing_log."""
        state = _make_state(enable_multimodal=True, has_figures=False)
        result = multimodal_processing_node(state)

        assert set(result.keys()) == {"processing_log"}
        assert result["processing_log"][-1].endswith("multimodal_processing:skipped")

    def test_multimodal_node_handles_empty_figures_list(self):
        """An empty figures list should produce an empty multimodal_notes without error."""
        state = _make_state(has_figures=True, figures=[])
        result = multimodal_processing_node(state)

        assert result["multimodal_notes"] == []

    def test_multimodal_node_handles_missing_has_figures_key(self):
        """A structure dict without has_figures defaults to False and skips processing."""
        state = _make_state(enable_multimodal=True)
        # Remove the has_figures key to exercise the .get("has_figures", False) default.
        state["structure"] = {"figures": ["Figure 1"]}
        result = multimodal_processing_node(state)

        assert set(result.keys()) == {"processing_log"}
        assert result["processing_log"][-1].endswith("multimodal_processing:skipped")

    def test_multimodal_node_handles_missing_figures_key(self):
        """A structure with has_figures=True but no figures key should yield empty notes."""
        state = _make_state(enable_multimodal=True)
        # Replace structure so that "figures" key is absent.
        state["structure"] = {"has_figures": True}
        result = multimodal_processing_node(state)

        assert result["multimodal_notes"] == []


# ---------------------------------------------------------------------------
# TestMultimodalErrorScenarios
# ---------------------------------------------------------------------------

class TestMultimodalErrorScenarios:
    """Tests covering note composition, partial replacement, and structure telemetry."""

    def test_multimodal_node_generates_text_only_notes_without_vision(self):
        """Without vision, notes are plain '<figure>: referenced in text' strings."""
        state = _make_state(
            enable_vision=False,
            figures=["Figure 1", "Figure 2"],
        )
        result = multimodal_processing_node(state)

        assert result["multimodal_notes"] == [
            "Figure 1: referenced in text",
            "Figure 2: referenced in text",
        ]

    def test_multimodal_node_vision_notes_replace_text_notes(self):
        """VLM notes fully replace text notes up to the number returned by vision."""
        state = _make_state(
            enable_vision=True,
            figures=["Figure 1", "Figure 2", "Figure 3"],
        )
        with patch(_PATCH_TARGET, return_value=(["VLM note 1", "VLM note 2"], 2)):
            result = multimodal_processing_node(state)

        notes = result["multimodal_notes"]
        assert notes[0] == "VLM note 1"
        assert notes[1] == "VLM note 2"
        assert notes[2] == "Figure 3: referenced in text"

    def test_multimodal_node_vision_notes_partial_replacement(self):
        """A single VLM note replaces only the first text note; the rest are preserved."""
        state = _make_state(
            enable_vision=True,
            figures=["Figure 1", "Figure 2"],
        )
        with patch(_PATCH_TARGET, return_value=(["VLM note 1"], 1)):
            result = multimodal_processing_node(state)

        notes = result["multimodal_notes"]
        assert notes[0] == "VLM note 1"
        assert notes[1] == "Figure 2: referenced in text"

    def test_multimodal_node_structure_updated_with_vision_telemetry(self):
        """When vision is enabled, structure must carry vision telemetry keys."""
        state = _make_state(enable_vision=True)
        with patch(_PATCH_TARGET, return_value=([], 0)):
            result = multimodal_processing_node(state)

        assert "vision_provider" in result["structure"]
        assert "vision_model" in result["structure"]
        assert "vision_described_count" in result["structure"]

    def test_multimodal_node_structure_not_updated_without_vision(self):
        """When vision is disabled, structure must NOT contain vision telemetry keys."""
        state = _make_state(
            enable_vision=False,
            enable_multimodal=True,
            has_figures=True,
        )
        result = multimodal_processing_node(state)

        assert "vision_provider" not in result["structure"]

    def test_multimodal_node_processing_log_records_ok(self):
        """A successful run must record the ok token as the last processing_log entry."""
        state = _make_state(
            enable_multimodal=True,
            enable_vision=False,
            has_figures=True,
        )
        result = multimodal_processing_node(state)

        assert result["processing_log"][-1].endswith("multimodal_processing:ok")
