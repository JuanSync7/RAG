# @summary
# Coverage tests for structure_detection_node: error paths, regex fallback,
# Docling integration, boundary cases, and processing_log correctness.
# Exports: TestStructureDetectionError, TestStructureDetectionBoundary,
#          TestStructureDetectionErrorScenarios
# Deps: src.ingest.doc_processing.nodes.structure_detection,
#       src.ingest.common.types, unittest.mock, pytest
# @end-summary

"""Coverage tests for the structure_detection_node (doc_processing pipeline)."""

from unittest.mock import MagicMock, patch

import pytest

from src.ingest.doc_processing.nodes.structure_detection import structure_detection_node
from src.ingest.common.types import IngestionConfig, Runtime


# ---------------------------------------------------------------------------
# State builder
# ---------------------------------------------------------------------------

def _make_state(
    raw_text="",
    source_path="/tmp/test.txt",
    source_name="test.txt",
    enable_docling=False,
    docling_strict=False,
    **config_overrides,
):
    config = IngestionConfig(
        enable_docling_parser=enable_docling,
        docling_strict=docling_strict,
        **config_overrides,
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


# ---------------------------------------------------------------------------
# TestStructureDetectionError
# ---------------------------------------------------------------------------

class TestStructureDetectionError:
    """Tests for error-path behavior of structure_detection_node."""

    def test_structure_detection_node_returns_error_when_docling_strict_fails(self):
        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            side_effect=RuntimeError("parse failed"),
        ):
            state = _make_state(enable_docling=True, docling_strict=True)
            result = structure_detection_node(state)

        assert result["errors"], "errors list should be non-empty"
        assert any("docling_parse_failed:" in e for e in result["errors"])
        assert result["should_skip"] is True
        assert result["processing_log"][-1].endswith("structure_detection:failed")

    def test_structure_detection_node_falls_back_to_regex_when_docling_nonstrict_fails(self):
        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            side_effect=RuntimeError("parse failed"),
        ):
            state = _make_state(
                enable_docling=True,
                docling_strict=False,
                raw_text="Figure 1 is shown.",
            )
            result = structure_detection_node(state)

        # Non-strict failure must not produce errors or should_skip
        assert not result.get("errors"), "errors should be absent or empty on non-strict failure"
        assert result["structure"]["has_figures"] is True
        assert result["structure"]["heading_count"] == 0

    def test_structure_detection_node_strict_failure_processing_log(self):
        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            side_effect=RuntimeError("parse failed"),
        ):
            state = _make_state(enable_docling=True, docling_strict=True)
            result = structure_detection_node(state)

        assert result["processing_log"][-1].endswith("structure_detection:failed")


# ---------------------------------------------------------------------------
# TestStructureDetectionBoundary
# ---------------------------------------------------------------------------

class TestStructureDetectionBoundary:
    """Tests for boundary and edge-case inputs to structure_detection_node."""

    def test_structure_detection_node_handles_empty_raw_text(self):
        state = _make_state(raw_text="", enable_docling=False)
        result = structure_detection_node(state)

        assert result["structure"]["has_figures"] is False
        assert result["structure"]["figures"] == []
        assert result["structure"]["heading_count"] == 0

    def test_structure_detection_node_handles_text_with_no_headings_or_figures(self):
        state = _make_state(raw_text="This is plain text.", enable_docling=False)
        result = structure_detection_node(state)

        assert result["structure"]["has_figures"] is False
        assert result["structure"]["figures"] == []
        assert result["structure"]["heading_count"] == 0

    def test_structure_detection_node_truncates_figures_to_32(self):
        # Build raw_text containing 35 figure references
        raw_text = " ".join(f"Figure {i}." for i in range(1, 36))
        state = _make_state(raw_text=raw_text, enable_docling=False)
        result = structure_detection_node(state)

        assert len(result["structure"]["figures"]) == 32

    def test_structure_detection_node_handles_none_raw_text(self):
        state = _make_state(raw_text=None, enable_docling=False)
        with pytest.raises(Exception):
            structure_detection_node(state)


# ---------------------------------------------------------------------------
# TestStructureDetectionErrorScenarios
# ---------------------------------------------------------------------------

class TestStructureDetectionErrorScenarios:
    """Tests for regex detection accuracy, Docling integration, schema, and log entries."""

    def test_structure_detection_node_regex_detects_figure_variations(self):
        raw_text = "Figure 1 is here. See Fig. 2 also. FIGURE 3a and fig. 4B."
        state = _make_state(raw_text=raw_text, enable_docling=False)
        result = structure_detection_node(state)

        assert result["structure"]["has_figures"] is True
        assert len(result["structure"]["figures"]) == 4

    def test_structure_detection_node_regex_detects_markdown_headings(self):
        raw_text = "# H1\n## H2\n### H3\nsome text"
        state = _make_state(raw_text=raw_text, enable_docling=False)
        result = structure_detection_node(state)

        assert result["structure"]["heading_count"] == 3

    def test_structure_detection_node_regex_detects_numbered_headings(self):
        raw_text = "1. INTRODUCTION\nsome text\n2.1 Supervised Learning"
        state = _make_state(raw_text=raw_text, enable_docling=False)
        result = structure_detection_node(state)

        assert result["structure"]["heading_count"] == 2

    def test_structure_detection_node_docling_disabled_uses_regex(self):
        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling"
        ) as mock_docling:
            state = _make_state(enable_docling=False, raw_text="Figure 1 ref.")
            result = structure_detection_node(state)

        mock_docling.assert_not_called()
        assert result["structure"]["has_figures"] is True

    def test_structure_detection_node_docling_enabled_replaces_raw_text(self):
        parsed = MagicMock()
        parsed.text_markdown = "# Docling Output"
        parsed.figures = ["Figure 1", "Fig. 2"]
        parsed.headings = ["# Introduction"]

        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            return_value=parsed,
        ):
            state = _make_state(enable_docling=True, raw_text="original text")
            result = structure_detection_node(state)

        assert result["raw_text"] == "# Docling Output"

    def test_structure_detection_node_structure_dict_schema(self):
        state = _make_state(enable_docling=False)
        result = structure_detection_node(state)

        assert set(result["structure"].keys()) == {
            "has_figures",
            "figures",
            "heading_count",
            "docling_enabled",
            "docling_model",
            "docling_document_available",
            "parser_strategy",
        }

    def test_structure_detection_node_processing_log_records_ok(self):
        state = _make_state(enable_docling=False)
        result = structure_detection_node(state)

        assert result["processing_log"][-1].endswith("structure_detection:ok")


# ---------------------------------------------------------------------------
# Iter-003: registry path error handler (lines 100-118)
# ---------------------------------------------------------------------------


class TestStructureDetectionRegistryPath:
    """Lines 100-118: parser_registry exception paths (strict and non-strict)."""

    def _make_state_with_registry(
        self,
        raw_text="",
        source_path="/tmp/test.txt",
        source_name="test.txt",
        docling_strict=False,
        registry=None,
        **config_overrides,
    ):
        """Build state with a parser_registry set on the Runtime."""
        config = IngestionConfig(
            enable_docling_parser=True,
            docling_strict=docling_strict,
            **config_overrides,
        )
        runtime = Runtime(
            config=config,
            embedder=MagicMock(),
            weaviate_client=MagicMock(),
            kg_builder=None,
        )
        if registry is not None:
            runtime.parser_registry = registry
        return {
            "raw_text": raw_text,
            "source_path": source_path,
            "source_name": source_name,
            "runtime": runtime,
            "errors": [],
            "processing_log": [],
        }

    def test_mock_registry_parser_failure_strict_returns_error(self):
        """Registry path: parser raises + strict=True → should_skip=True (lines 100-108)."""
        failing_registry = MagicMock()
        failing_parser = MagicMock()
        failing_parser.parse.side_effect = RuntimeError("parser blew up")
        failing_registry.get_parser.return_value = failing_parser

        state = self._make_state_with_registry(
            raw_text="some text",
            docling_strict=True,
            registry=failing_registry,
        )
        result = structure_detection_node(state)

        assert result.get("should_skip") is True
        assert result.get("errors"), "errors must be non-empty on strict failure"
        assert any("parser_failed" in e for e in result["errors"])

    def test_mock_registry_parser_failure_non_strict_falls_back_to_regex(self):
        """Registry path: parser raises + strict=False → regex fallback (lines 109-118)."""
        failing_registry = MagicMock()
        failing_parser = MagicMock()
        failing_parser.parse.side_effect = RuntimeError("parser blew up")
        failing_registry.get_parser.return_value = failing_parser

        state = self._make_state_with_registry(
            raw_text="Figure 1 is shown. See also Fig. 2.",
            docling_strict=False,
            registry=failing_registry,
        )
        result = structure_detection_node(state)

        # Non-strict: no errors, no skip, regex detects figures
        assert not result.get("should_skip")
        assert not result.get("errors")
        assert result["structure"]["has_figures"] is True
        assert result["structure"]["parser_strategy"] == "regex_fallback"

    def test_mock_registry_parser_failure_non_strict_clears_parse_result(self):
        """After registry parser failure, parse_result/parser_instance cleared (lines 114-116)."""
        failing_registry = MagicMock()
        failing_parser = MagicMock()
        failing_parser.parse.side_effect = ValueError("bad input")
        failing_registry.get_parser.return_value = failing_parser

        state = self._make_state_with_registry(
            raw_text="# Heading one\nsome text",
            docling_strict=False,
            registry=failing_registry,
        )
        result = structure_detection_node(state)

        # Should still have processed output (regex path)
        assert "structure" in result
        assert result["structure"]["heading_count"] >= 1

    def test_mock_registry_get_parser_failure_non_strict(self):
        """Registry path: get_parser itself raises (before parse) — non-strict fallback."""
        failing_registry = MagicMock()
        failing_registry.get_parser.side_effect = RuntimeError("no parser for type")

        state = self._make_state_with_registry(
            raw_text="Figure 3 and more text.",
            docling_strict=False,
            registry=failing_registry,
        )
        result = structure_detection_node(state)

        assert not result.get("should_skip")
        assert result["structure"]["has_figures"] is True

    def test_mock_registry_get_parser_failure_strict(self):
        """Registry path: get_parser itself raises — strict mode → should_skip=True."""
        failing_registry = MagicMock()
        failing_registry.get_parser.side_effect = RuntimeError("no parser for type")

        state = self._make_state_with_registry(
            raw_text="some text",
            docling_strict=True,
            registry=failing_registry,
        )
        result = structure_detection_node(state)

        assert result.get("should_skip") is True
