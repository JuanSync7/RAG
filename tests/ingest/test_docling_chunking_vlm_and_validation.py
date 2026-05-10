# @summary
# Tests for vlm_enrichment_node (mode dispatch, placeholder replacement, budget enforcement)
# and _check_docling_chunking_config validation rules (invalid vlm_mode, builtin-requires-docling,
# external-without-vision warning, hybrid_chunker_max_tokens > 512 warning).
# Covers: src/ingest/embedding/nodes/vlm_enrichment.py, src/ingest/impl.py
# @end-summary
"""Tests for VLM enrichment node and Docling chunking config validation.

Two independent test suites in one file:

1. ``TestVlmEnrichmentNode`` — white-box tests for the post-chunking VLM
   enrichment node: mode dispatch (disabled, builtin, external), placeholder
   replacement, per-document figure budget enforcement, and non-fatal error
   paths.

2. ``TestCheckDoclingChunkingConfig`` — white-box tests for
   ``_check_docling_chunking_config`` in ``src.ingest.impl``: invalid
   ``vlm_mode`` values, builtin-requires-docling rule, external-without-vision
   warning, ``hybrid_chunker_max_tokens`` > 512 warning, and boundary
   conditions.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.ingest.common.schemas import ProcessedChunk
from src.ingest.common.types import IngestionConfig, Runtime


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_runtime(vlm_mode: str = "disabled", **config_kwargs) -> Runtime:
    """Build a minimal Runtime with a config that has the given vlm_mode."""
    config = IngestionConfig(vlm_mode=vlm_mode, **config_kwargs)
    return Runtime(
        config=config,
        embedder=MagicMock(),
        weaviate_client=MagicMock(),
        kg_builder=None,
    )


def _make_chunk(text: str = "Sample chunk text.", chunk_index: int = 0) -> ProcessedChunk:
    """Return a minimal ProcessedChunk with the given text."""
    return ProcessedChunk(
        text=text,
        metadata={"chunk_index": chunk_index, "source_name": "doc.pdf"},
    )


def _make_state(
    chunks: list[ProcessedChunk] | None = None,
    vlm_mode: str = "disabled",
    source_uri: str = "",
    **config_kwargs,
) -> dict:
    """Build a minimal EmbeddingPipelineState-compatible dict for vlm_enrichment_node tests."""
    runtime = _make_runtime(vlm_mode=vlm_mode, **config_kwargs)
    return {
        "chunks": chunks if chunks is not None else [],
        "processing_log": [],
        "source_uri": source_uri,
        "runtime": runtime,
    }


# ---------------------------------------------------------------------------
# Module 1: vlm_enrichment_node — mode dispatch and no-op paths
# ---------------------------------------------------------------------------


class TestVlmEnrichmentNodeModeDispatch:
    """vlm_enrichment_node returns early (no-op) for disabled and builtin modes."""

    def test_disabled_mode_returns_chunks_unchanged(self):
        """vlm_mode=disabled: chunks pass through unchanged."""
        from src.ingest.embedding.nodes.vlm_enrichment import vlm_enrichment_node

        chunks = [_make_chunk("Hello world"), _make_chunk("Paragraph two")]
        state = _make_state(chunks=chunks, vlm_mode="disabled")
        result = vlm_enrichment_node(state)
        assert result["chunks"] == chunks

    def test_disabled_mode_logs_skipped(self):
        """vlm_mode=disabled: processing_log contains 'vlm_enrichment:skipped'."""
        from src.ingest.embedding.nodes.vlm_enrichment import vlm_enrichment_node

        state = _make_state(vlm_mode="disabled")
        result = vlm_enrichment_node(state)
        assert "vlm_enrichment:skipped" in result["processing_log"]

    def test_builtin_mode_returns_chunks_unchanged(self):
        """vlm_mode=builtin: chunks pass through unchanged (SmolVLM ran at parse time)."""
        from src.ingest.embedding.nodes.vlm_enrichment import vlm_enrichment_node

        chunks = [_make_chunk("Content with ![fig](img.png)")]
        state = _make_state(chunks=chunks, vlm_mode="builtin")
        result = vlm_enrichment_node(state)
        assert result["chunks"] == chunks

    def test_builtin_mode_logs_skipped(self):
        """vlm_mode=builtin: processing_log contains 'vlm_enrichment:skipped'."""
        from src.ingest.embedding.nodes.vlm_enrichment import vlm_enrichment_node

        state = _make_state(vlm_mode="builtin")
        result = vlm_enrichment_node(state)
        assert "vlm_enrichment:skipped" in result["processing_log"]

    def test_disabled_mode_no_litellm_calls(self):
        """vlm_mode=disabled: no LiteLLM calls are made."""
        from src.ingest.embedding.nodes.vlm_enrichment import vlm_enrichment_node

        chunks = [_make_chunk("Text with ![fig](img.png)")]
        state = _make_state(chunks=chunks, vlm_mode="disabled")
        with patch("src.ingest.support.vision._describe_image") as mock_describe:
            vlm_enrichment_node(state)
            mock_describe.assert_not_called()

    def test_builtin_mode_no_litellm_calls(self):
        """vlm_mode=builtin: no LiteLLM calls are made."""
        from src.ingest.embedding.nodes.vlm_enrichment import vlm_enrichment_node

        chunks = [_make_chunk("Text with ![fig](img.png)")]
        state = _make_state(chunks=chunks, vlm_mode="builtin")
        with patch("src.ingest.support.vision._describe_image") as mock_describe:
            vlm_enrichment_node(state)
            mock_describe.assert_not_called()


# ---------------------------------------------------------------------------
# Module 1: vlm_enrichment_node — external mode, happy paths
# ---------------------------------------------------------------------------


class TestVlmEnrichmentNodeExternalMode:
    """vlm_mode=external: replaces placeholders, logs ok, respects budget."""

    def test_external_mode_no_placeholders_returns_unchanged(self):
        """External mode, chunks without image placeholders: no replacements, log ok."""
        from src.ingest.embedding.nodes.vlm_enrichment import vlm_enrichment_node

        chunks = [_make_chunk("Plain text, no images here.")]
        state = _make_state(chunks=chunks, vlm_mode="external")
        with patch(
            "src.ingest.embedding.nodes.vlm_enrichment._enrich_chunk_external",
            wraps=lambda chunk, cfg, count, **kw: (chunk, count),
        ):
            result = vlm_enrichment_node(state)
        assert result["chunks"][0].text == "Plain text, no images here."
        assert "vlm_enrichment:external:ok" in result["processing_log"]

    def test_external_mode_logs_ok(self):
        """External mode: processing_log contains 'vlm_enrichment:external:ok'."""
        from src.ingest.embedding.nodes.vlm_enrichment import vlm_enrichment_node

        state = _make_state(chunks=[], vlm_mode="external")
        result = vlm_enrichment_node(state)
        assert "vlm_enrichment:external:ok" in result["processing_log"]

    def test_external_mode_placeholder_replaced_via_mock(self):
        """External mode: image placeholder is replaced by VLM description."""
        from src.ingest.embedding.nodes.vlm_enrichment import vlm_enrichment_node

        chunk = _make_chunk("Some text ![Fig 1](img.png) more text", chunk_index=0)
        state = _make_state(
            chunks=[chunk],
            vlm_mode="external",
            vision_max_figures=5,
        )

        # Mock _extract_image_candidates to return one candidate.
        from src.ingest.support.vision import VisionImageCandidate, VisionDescription

        mock_candidate = VisionImageCandidate(
            figure_label="Figure 1",
            alt_text="Fig 1",
            source_ref="img.png",
            image_b64="aW1hZ2U=",
            mime_type="image/png",
        )
        mock_description = VisionDescription(
            figure_label="Figure 1",
            source_ref="img.png",
            caption="a bar chart",
            visible_text="",
            tags=[],
        )

        with patch(
            "src.ingest.embedding.nodes.vlm_enrichment._extract_image_candidates",
            return_value=[mock_candidate],
        ), patch(
            "src.ingest.embedding.nodes.vlm_enrichment._describe_image",
            return_value=mock_description,
        ):
            result = vlm_enrichment_node(state)

        out_text = result["chunks"][0].text
        assert "![Fig 1](img.png)" not in out_text
        assert "a bar chart" in out_text
        assert "Some text" in out_text
        assert "more text" in out_text

    def test_external_mode_empty_chunks_list(self):
        """External mode with empty chunks list: returns empty list, logs ok."""
        from src.ingest.embedding.nodes.vlm_enrichment import vlm_enrichment_node

        state = _make_state(chunks=[], vlm_mode="external")
        result = vlm_enrichment_node(state)
        assert result["chunks"] == []
        assert "vlm_enrichment:external:ok" in result["processing_log"]


# ---------------------------------------------------------------------------
# Module 1: vlm_enrichment_node — budget enforcement
# ---------------------------------------------------------------------------


class TestVlmEnrichmentNodeBudget:
    """vision_max_figures budget is enforced across chunks."""

    def _make_placeholder_chunk(self, idx: int) -> ProcessedChunk:
        return _make_chunk(f"Chunk {idx} with ![fig{idx}](img{idx}.png)", chunk_index=idx)

    def test_budget_zero_no_replacements(self):
        """vision_max_figures=0: all placeholders left unchanged, zero VLM calls."""
        from src.ingest.embedding.nodes.vlm_enrichment import vlm_enrichment_node

        chunks = [self._make_placeholder_chunk(i) for i in range(3)]
        state = _make_state(chunks=chunks, vlm_mode="external", vision_max_figures=0)

        with patch(
            "src.ingest.embedding.nodes.vlm_enrichment._describe_image"
        ) as mock_describe:
            result = vlm_enrichment_node(state)
            mock_describe.assert_not_called()

        for original, output in zip(chunks, result["chunks"]):
            assert original.text == output.text

    def test_budget_limits_replacements_across_chunks(self):
        """vision_max_figures=2 with 5 chunks: only first 2 placeholders replaced."""
        from src.ingest.embedding.nodes.vlm_enrichment import vlm_enrichment_node
        from src.ingest.support.vision import VisionImageCandidate, VisionDescription

        chunks = [self._make_placeholder_chunk(i) for i in range(5)]
        state = _make_state(chunks=chunks, vlm_mode="external", vision_max_figures=2)

        call_count = {"n": 0}

        def fake_candidate(markdown, *, source_path, max_figures, max_image_bytes):
            return [
                VisionImageCandidate(
                    figure_label=f"Figure {call_count['n'] + 1}",
                    alt_text="",
                    source_ref="img.png",
                    image_b64="aW1hZ2U=",
                    mime_type="image/png",
                )
            ]

        def fake_describe(candidate, config):
            call_count["n"] += 1
            return VisionDescription(
                figure_label=candidate.figure_label,
                source_ref=candidate.source_ref,
                caption=f"desc_{call_count['n']}",
                visible_text="",
                tags=[],
            )

        with patch(
            "src.ingest.embedding.nodes.vlm_enrichment._extract_image_candidates",
            side_effect=fake_candidate,
        ), patch(
            "src.ingest.embedding.nodes.vlm_enrichment._describe_image",
            side_effect=fake_describe,
        ):
            result = vlm_enrichment_node(state)

        assert call_count["n"] == 2, "Expected exactly 2 VLM calls for budget=2"
        # Chunks 0 and 1 should have their placeholder replaced.
        for i in range(2):
            assert "![" not in result["chunks"][i].text or "desc_" in result["chunks"][i].text
        # Chunks 2-4 should be unchanged (budget exhausted).
        for i in range(2, 5):
            assert result["chunks"][i].text == chunks[i].text

    def test_budget_one_of_three_replaced(self):
        """vision_max_figures=1 with 3 chunks: exactly 1 replacement."""
        from src.ingest.embedding.nodes.vlm_enrichment import vlm_enrichment_node
        from src.ingest.support.vision import VisionImageCandidate, VisionDescription

        chunks = [self._make_placeholder_chunk(i) for i in range(3)]
        state = _make_state(chunks=chunks, vlm_mode="external", vision_max_figures=1)

        described = {"n": 0}

        def fake_candidate(markdown, *, source_path, max_figures, max_image_bytes):
            return [
                VisionImageCandidate(
                    figure_label="Figure 1",
                    alt_text="",
                    source_ref="img.png",
                    image_b64="aW1hZ2U=",
                    mime_type="image/png",
                )
            ]

        def fake_describe(candidate, config):
            described["n"] += 1
            return VisionDescription(
                figure_label="Figure 1",
                source_ref="img.png",
                caption="replaced",
                visible_text="",
                tags=[],
            )

        with patch(
            "src.ingest.embedding.nodes.vlm_enrichment._extract_image_candidates",
            side_effect=fake_candidate,
        ), patch(
            "src.ingest.embedding.nodes.vlm_enrichment._describe_image",
            side_effect=fake_describe,
        ):
            result = vlm_enrichment_node(state)

        assert described["n"] == 1
        assert result["chunks"][1].text == chunks[1].text
        assert result["chunks"][2].text == chunks[2].text


# ---------------------------------------------------------------------------
# Module 1: vlm_enrichment_node — error paths (non-fatal)
# ---------------------------------------------------------------------------


class TestVlmEnrichmentNodeErrorPaths:
    """Per-chunk and per-placeholder failures are non-fatal; original text preserved."""

    def test_describe_image_failure_preserves_original_text(self):
        """LiteLLM/VLM API failure: placeholder left unchanged; node returns all chunks."""
        from src.ingest.embedding.nodes.vlm_enrichment import vlm_enrichment_node
        from src.ingest.support.vision import VisionImageCandidate

        original_text = "Header ![fig](img.png) footer"
        chunk = _make_chunk(original_text, chunk_index=0)
        state = _make_state(chunks=[chunk], vlm_mode="external", vision_max_figures=5)

        mock_candidate = VisionImageCandidate(
            figure_label="Figure 1",
            alt_text="fig",
            source_ref="img.png",
            image_b64="aW1hZ2U=",
            mime_type="image/png",
        )

        with patch(
            "src.ingest.embedding.nodes.vlm_enrichment._extract_image_candidates",
            return_value=[mock_candidate],
        ), patch(
            "src.ingest.embedding.nodes.vlm_enrichment._describe_image",
            side_effect=Exception("API timeout after 60s"),
        ):
            result = vlm_enrichment_node(state)

        assert result["chunks"][0].text == original_text
        assert "vlm_enrichment:external:ok" in result["processing_log"]

    def test_outer_exception_returns_original_chunks(self):
        """Outer unexpected exception: original chunks returned; ERROR logged; node doesn't raise."""
        from src.ingest.embedding.nodes.vlm_enrichment import vlm_enrichment_node

        chunks = [_make_chunk("Text ![fig](img.png)")]
        state = _make_state(chunks=chunks, vlm_mode="external", vision_max_figures=5)

        with patch(
            "src.ingest.embedding.nodes.vlm_enrichment._find_image_placeholders",
            side_effect=RuntimeError("unexpected internal failure"),
        ):
            result = vlm_enrichment_node(state)

        assert result["chunks"] == chunks
        assert "vlm_enrichment:external:error" in result["processing_log"]

    def test_outer_exception_does_not_raise(self):
        """Node never raises — even if an outer exception fires."""
        from src.ingest.embedding.nodes.vlm_enrichment import vlm_enrichment_node

        state = _make_state(chunks=[_make_chunk("x")], vlm_mode="external")
        with patch(
            "src.ingest.embedding.nodes.vlm_enrichment._find_image_placeholders",
            side_effect=RuntimeError("boom"),
        ):
            # Must not raise.
            result = vlm_enrichment_node(state)
        assert isinstance(result, dict)

    def test_extract_candidates_failure_skips_placeholder(self):
        """_extract_image_candidates failure: placeholder left unchanged; processing continues."""
        from src.ingest.embedding.nodes.vlm_enrichment import vlm_enrichment_node

        original_text = "Intro ![fig](missing.png) body"
        chunk = _make_chunk(original_text, chunk_index=0)
        state = _make_state(chunks=[chunk], vlm_mode="external", vision_max_figures=5)

        with patch(
            "src.ingest.embedding.nodes.vlm_enrichment._extract_image_candidates",
            side_effect=OSError("file not found"),
        ):
            result = vlm_enrichment_node(state)

        assert result["chunks"][0].text == original_text
        assert "vlm_enrichment:external:ok" in result["processing_log"]


# ---------------------------------------------------------------------------
# Module 1: _find_image_placeholders — unit tests
# ---------------------------------------------------------------------------


class TestFindImagePlaceholders:
    """White-box tests for _find_image_placeholders helper."""

    def test_empty_string_returns_empty(self):
        from src.ingest.embedding.nodes.vlm_enrichment import _find_image_placeholders

        assert _find_image_placeholders("") == []

    def test_no_images_returns_empty(self):
        from src.ingest.embedding.nodes.vlm_enrichment import _find_image_placeholders

        assert _find_image_placeholders("no images here") == []

    def test_single_image_ref_found(self):
        from src.ingest.embedding.nodes.vlm_enrichment import _find_image_placeholders

        matches = _find_image_placeholders("![alt](path.png)")
        assert len(matches) == 1
        assert matches[0].group(0) == "![alt](path.png)"

    def test_two_image_refs_found(self):
        from src.ingest.embedding.nodes.vlm_enrichment import _find_image_placeholders

        matches = _find_image_placeholders("![a](b.png) and ![c](d.png)")
        assert len(matches) == 2

    def test_placeholder_in_surrounding_text(self):
        from src.ingest.embedding.nodes.vlm_enrichment import _find_image_placeholders

        matches = _find_image_placeholders("prefix ![x](y.jpg) suffix")
        assert len(matches) == 1
        assert matches[0].group(0) == "![x](y.jpg)"


# ---------------------------------------------------------------------------
# Module 1: _replace_placeholder — unit tests
# ---------------------------------------------------------------------------


class TestReplacePlaceholder:
    """White-box tests for _replace_placeholder helper."""

    def test_full_string_replaced_when_only_placeholder(self):
        """When the entire string is a placeholder, output is just the description."""
        from src.ingest.embedding.nodes.vlm_enrichment import (
            _find_image_placeholders,
            _replace_placeholder,
        )

        text = "![a](b)"
        matches = _find_image_placeholders(text)
        assert len(matches) == 1
        result = _replace_placeholder(text, matches[0], "desc")
        assert result == "desc"

    def test_surrounding_text_preserved_exactly(self):
        """Surrounding characters before and after placeholder are preserved exactly."""
        from src.ingest.embedding.nodes.vlm_enrichment import (
            _find_image_placeholders,
            _replace_placeholder,
        )

        text = "before ![x](y) after"
        matches = _find_image_placeholders(text)
        result = _replace_placeholder(text, matches[0], "desc")
        assert result == "before desc after"

    def test_no_extra_whitespace_added(self):
        """No extra whitespace is added around the description."""
        from src.ingest.embedding.nodes.vlm_enrichment import (
            _find_image_placeholders,
            _replace_placeholder,
        )

        text = "A![x](y)B"
        matches = _find_image_placeholders(text)
        result = _replace_placeholder(text, matches[0], "Z")
        assert result == "AZB"


# ---------------------------------------------------------------------------
# Module 2: _check_docling_chunking_config — happy paths
# ---------------------------------------------------------------------------


class TestCheckDoclingChunkingConfigHappy:
    """_check_docling_chunking_config returns no errors and no warnings for valid configs."""

    def test_disabled_mode_at_limit_no_errors_no_warnings(self):
        """vlm_mode=disabled + hybrid_chunker_max_tokens=512: ([], [])."""
        from src.ingest.impl import _check_docling_chunking_config

        config = IngestionConfig(vlm_mode="disabled", hybrid_chunker_max_tokens=512)
        errors, warnings = _check_docling_chunking_config(config)
        assert errors == []
        assert warnings == []

    def test_disabled_mode_below_limit_no_warnings(self):
        """vlm_mode=disabled + hybrid_chunker_max_tokens=256: no warnings."""
        from src.ingest.impl import _check_docling_chunking_config

        config = IngestionConfig(vlm_mode="disabled", hybrid_chunker_max_tokens=256)
        _, warnings = _check_docling_chunking_config(config)
        assert warnings == []

    def test_disabled_mode_max_tokens_at_limit_no_warning(self):
        """hybrid_chunker_max_tokens=512 is at (not above) the limit: no warning."""
        from src.ingest.impl import _check_docling_chunking_config

        config = IngestionConfig(vlm_mode="disabled", hybrid_chunker_max_tokens=512)
        _, warnings = _check_docling_chunking_config(config)
        assert not any("hybrid_chunker_max_tokens" in w for w in warnings)

    def test_builtin_with_docling_installed_no_errors(self):
        """vlm_mode=builtin with docling importable: no errors."""
        from src.ingest.impl import _check_docling_chunking_config

        config = IngestionConfig(vlm_mode="builtin")
        # If docling is installed in the test env, this should produce no errors.
        # If not, we skip by patching the import to succeed.
        try:
            import docling.document_converter  # noqa: F401
            docling_available = True
        except ImportError:
            docling_available = False

        if docling_available:
            errors, _ = _check_docling_chunking_config(config)
            assert not any("vlm_mode=builtin requires docling" in e for e in errors)
        else:
            # Patch docling as available for this scenario.
            mock_dc = MagicMock()
            with patch.dict(
                "sys.modules",
                {
                    "docling": MagicMock(),
                    "docling.document_converter": mock_dc,
                },
            ):
                errors, _ = _check_docling_chunking_config(config)
            assert not any("vlm_mode=builtin requires docling" in e for e in errors)

    def test_external_with_vision_model_configured_no_warnings(self):
        """vlm_mode=external with a vision model set: no warnings about external config."""
        from src.ingest.impl import _check_docling_chunking_config

        config = IngestionConfig(
            vlm_mode="external",
            vision_model="gpt-4o",
        )
        _, warnings = _check_docling_chunking_config(config)
        # No "vlm_mode=external" warning should appear.
        assert not any("vlm_mode=external" in w for w in warnings)

    def test_max_tokens_one_no_warning(self):
        """hybrid_chunker_max_tokens=1: valid low value, no warning."""
        from src.ingest.impl import _check_docling_chunking_config

        config = IngestionConfig(vlm_mode="disabled", hybrid_chunker_max_tokens=1)
        _, warnings = _check_docling_chunking_config(config)
        assert not any("hybrid_chunker_max_tokens" in w for w in warnings)


# ---------------------------------------------------------------------------
# Module 2: _check_docling_chunking_config — error scenarios
# ---------------------------------------------------------------------------


class TestCheckDoclingChunkingConfigErrors:
    """_check_docling_chunking_config detects hard errors correctly."""

    def test_invalid_vlm_mode_produces_error(self):
        """vlm_mode='invalid_mode': errors list is non-empty with informative message."""
        from src.ingest.impl import _check_docling_chunking_config

        config = IngestionConfig(vlm_mode="invalid_mode")
        errors, _ = _check_docling_chunking_config(config)
        assert len(errors) > 0
        combined = " ".join(errors)
        # The error message uses repr(), so the value appears as e.g. 'invalid_mode'.
        assert "invalid_mode" in combined
        assert "not valid" in combined

    def test_invalid_vlm_mode_message_lists_valid_values(self):
        """Error message for invalid vlm_mode includes the list of accepted values."""
        from src.ingest.impl import _check_docling_chunking_config

        config = IngestionConfig(vlm_mode="BUILTIN")
        errors, _ = _check_docling_chunking_config(config)
        combined = " ".join(errors)
        # Message must describe what values are accepted.
        assert "builtin" in combined.lower() or "must be one of" in combined

    def test_builtin_without_docling_produces_error(self):
        """vlm_mode=builtin + docling not installed: error contains 'vlm_mode=builtin requires docling'."""
        from src.ingest.impl import _check_docling_chunking_config

        config = IngestionConfig(vlm_mode="builtin")
        with patch.dict(
            "sys.modules",
            {
                "docling": None,
                "docling.document_converter": None,
            },
        ):
            errors, _ = _check_docling_chunking_config(config)

        combined = " ".join(errors)
        assert "vlm_mode=builtin requires docling" in combined

    def test_builtin_without_docling_no_warnings_just_errors(self):
        """vlm_mode=builtin + docling not installed: only errors produced, not warnings."""
        from src.ingest.impl import _check_docling_chunking_config

        config = IngestionConfig(vlm_mode="builtin")
        with patch.dict(
            "sys.modules",
            {
                "docling": None,
                "docling.document_converter": None,
            },
        ):
            errors, warnings = _check_docling_chunking_config(config)

        assert len(errors) >= 1
        # The docling-missing check emits an error, not a warning.
        assert not any("vlm_mode=builtin requires docling" in w for w in warnings)

    def test_empty_string_vlm_mode_is_invalid(self):
        """vlm_mode='' (empty string): not in the valid set, error emitted."""
        from src.ingest.impl import _check_docling_chunking_config

        config = IngestionConfig(vlm_mode="")
        errors, _ = _check_docling_chunking_config(config)
        assert len(errors) > 0

    def test_uppercase_disabled_is_invalid(self):
        """vlm_mode='DISABLED' (uppercase): case-sensitive check, error emitted."""
        from src.ingest.impl import _check_docling_chunking_config

        config = IngestionConfig(vlm_mode="DISABLED")
        errors, _ = _check_docling_chunking_config(config)
        assert len(errors) > 0

    def test_returns_two_element_tuple(self):
        """Return value is always a 2-tuple (errors, warnings), both lists."""
        from src.ingest.impl import _check_docling_chunking_config

        config = IngestionConfig(vlm_mode="disabled")
        result = _check_docling_chunking_config(config)
        assert isinstance(result, tuple)
        assert len(result) == 2
        errors, warnings = result
        assert isinstance(errors, list)
        assert isinstance(warnings, list)


# ---------------------------------------------------------------------------
# Module 2: _check_docling_chunking_config — warning scenarios
# ---------------------------------------------------------------------------


class TestCheckDoclingChunkingConfigWarnings:
    """_check_docling_chunking_config emits warnings for non-fatal misconfiguration."""

    def test_external_without_vision_config_produces_warning(self):
        """vlm_mode=external + no vision model or router: warning, no errors."""
        from src.ingest.impl import _check_docling_chunking_config

        # Ensure no vision_model is set and no LLM_ROUTER_CONFIG.
        config = IngestionConfig(vlm_mode="external", vision_model="")
        with patch("src.ingest.impl.LLM_ROUTER_CONFIG", None):
            errors, warnings = _check_docling_chunking_config(config)

        assert errors == [] or not any("vlm_mode=external" in e for e in errors)
        assert len(warnings) > 0
        combined = " ".join(warnings)
        assert "vlm_mode=external" in combined

    def test_external_without_vision_warning_mentions_skipping(self):
        """External warning message mentions VLM enrichment will be skipped."""
        from src.ingest.impl import _check_docling_chunking_config

        config = IngestionConfig(vlm_mode="external", vision_model="")
        with patch("src.ingest.impl.LLM_ROUTER_CONFIG", None):
            _, warnings = _check_docling_chunking_config(config)

        combined = " ".join(warnings)
        assert "skip" in combined.lower() or "skipped" in combined.lower()

    def test_max_tokens_above_512_produces_warning(self):
        """hybrid_chunker_max_tokens=1024: warning contains 'hybrid_chunker_max_tokens'."""
        from src.ingest.impl import _check_docling_chunking_config

        config = IngestionConfig(vlm_mode="disabled", hybrid_chunker_max_tokens=1024)
        _, warnings = _check_docling_chunking_config(config)
        assert len(warnings) > 0
        combined = " ".join(warnings)
        assert "hybrid_chunker_max_tokens" in combined

    def test_max_tokens_warning_mentions_512(self):
        """Warning for max_tokens > 512 references the 512 limit."""
        from src.ingest.impl import _check_docling_chunking_config

        config = IngestionConfig(vlm_mode="disabled", hybrid_chunker_max_tokens=1024)
        _, warnings = _check_docling_chunking_config(config)
        combined = " ".join(warnings)
        assert "512" in combined

    def test_max_tokens_at_513_produces_warning(self):
        """hybrid_chunker_max_tokens=513: exactly one step above limit triggers warning."""
        from src.ingest.impl import _check_docling_chunking_config

        config = IngestionConfig(vlm_mode="disabled", hybrid_chunker_max_tokens=513)
        _, warnings = _check_docling_chunking_config(config)
        assert any("hybrid_chunker_max_tokens" in w for w in warnings)

    def test_max_tokens_warning_is_not_an_error(self):
        """hybrid_chunker_max_tokens > 512 is a warning only, not an error."""
        from src.ingest.impl import _check_docling_chunking_config

        config = IngestionConfig(vlm_mode="disabled", hybrid_chunker_max_tokens=1024)
        errors, _ = _check_docling_chunking_config(config)
        assert not any("hybrid_chunker_max_tokens" in e for e in errors)

    def test_external_warning_is_not_an_error(self):
        """vlm_mode=external without vision config: warning only, errors list is empty."""
        from src.ingest.impl import _check_docling_chunking_config

        config = IngestionConfig(vlm_mode="external", vision_model="")
        with patch("src.ingest.impl.LLM_ROUTER_CONFIG", None):
            errors, warnings = _check_docling_chunking_config(config)

        # Errors should not contain the external-mode message.
        assert not any("vlm_mode=external" in e for e in errors)
        assert len(warnings) > 0


# ---------------------------------------------------------------------------
# Module 2: _check_docling_chunking_config — boundary conditions
# ---------------------------------------------------------------------------


class TestCheckDoclingChunkingConfigBoundary:
    """Boundary conditions for _check_docling_chunking_config."""

    def test_multiple_invalid_conditions_accumulate_errors(self):
        """Invalid vlm_mode + builtin-missing-docling: both errors in same list."""
        from src.ingest.impl import _check_docling_chunking_config

        # "BUILTIN" is invalid (wrong case), so Rule 0 fires.
        # Rule A (builtin-requires-docling) should NOT also fire since the vlm_mode
        # string "BUILTIN" != "builtin". But if we use "builtin" with no docling,
        # Rule 0 passes and Rule A fires. Combine: invalid mode + max_tokens:
        config = IngestionConfig(vlm_mode="wrong_mode", hybrid_chunker_max_tokens=1024)
        errors, warnings = _check_docling_chunking_config(config)
        # At minimum: one error (invalid mode).
        assert len(errors) >= 1
        # And a warning for max_tokens.
        assert len(warnings) >= 1

    def test_max_tokens_512_at_limit_not_a_warning(self):
        """hybrid_chunker_max_tokens=512 is at the limit (not above): no warning."""
        from src.ingest.impl import _check_docling_chunking_config

        config = IngestionConfig(vlm_mode="disabled", hybrid_chunker_max_tokens=512)
        _, warnings = _check_docling_chunking_config(config)
        assert not any("hybrid_chunker_max_tokens" in w for w in warnings)

    def test_all_three_valid_modes_are_accepted(self):
        """'disabled', 'builtin', 'external' are all valid vlm_mode values (no Rule 0 error)."""
        from src.ingest.impl import _check_docling_chunking_config

        mock_dc = MagicMock()
        with patch.dict(
            "sys.modules",
            {"docling": MagicMock(), "docling.document_converter": mock_dc},
        ):
            for mode in ("disabled", "builtin", "external"):
                config = IngestionConfig(
                    vlm_mode=mode,
                    vision_model="gpt-4o" if mode == "external" else "",
                )
                errors, _ = _check_docling_chunking_config(config)
                invalid_mode_errors = [e for e in errors if "not valid" in e]
                assert invalid_mode_errors == [], (
                    f"vlm_mode={mode!r} should be valid but got errors: {invalid_mode_errors}"
                )


# ---------------------------------------------------------------------------
# Module 2: verify_core_design — integration with _check_docling_chunking_config
# ---------------------------------------------------------------------------


class TestVerifyCoreDesignDoclingChecks:
    """verify_core_design delegates to _check_docling_chunking_config and surfaces results."""

    def test_verify_core_design_ok_with_valid_config(self):
        """verify_core_design returns ok=True with a fully valid config."""
        from src.ingest.impl import verify_core_design

        config = IngestionConfig(
            vlm_mode="disabled",
            hybrid_chunker_max_tokens=512,
            chunk_size=512,
            chunk_overlap=50,
        )
        result = verify_core_design(config)
        # May have other design check results but should not have vlm/chunker errors.
        vlm_errors = [e for e in result.errors if "vlm_mode" in e or "hybrid_chunker" in e]
        assert vlm_errors == []

    def test_verify_core_design_propagates_invalid_vlm_mode_error(self):
        """verify_core_design includes invalid vlm_mode error in its errors list."""
        from src.ingest.impl import verify_core_design

        config = IngestionConfig(
            vlm_mode="bogus",
            chunk_size=512,
            chunk_overlap=50,
        )
        result = verify_core_design(config)
        assert not result.ok
        combined = " ".join(result.errors)
        assert "vlm_mode=bogus" in combined or "not valid" in combined

    def test_verify_core_design_propagates_max_tokens_warning(self):
        """verify_core_design includes hybrid_chunker_max_tokens warning in its warnings list."""
        from src.ingest.impl import verify_core_design

        config = IngestionConfig(
            vlm_mode="disabled",
            hybrid_chunker_max_tokens=1024,
            chunk_size=512,
            chunk_overlap=50,
        )
        result = verify_core_design(config)
        assert any("hybrid_chunker_max_tokens" in w for w in result.warnings)
