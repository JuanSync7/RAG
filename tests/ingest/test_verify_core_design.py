# @summary
# Tests for verify_core_design() and the _check_parser_abstraction_config() sub-validator.
# Covers all six rules added in T9 (FR-3301, FR-3320, FR-3340–3342) plus the
# pre-existing VLM, docling, and visual-embedding validation rules.
# Exports: TestParserStrategyValidation, TestChunkerValidation, TestVlmMutualExclusion,
#          TestVlmEnrichmentCodeConflict, TestVerifyCoreDesignIntegration
# Deps: pytest, src.ingest.impl.verify_core_design, src.ingest.common.types.IngestionConfig
# @end-summary

"""Tests for verify_core_design() — parser abstraction validators (T9).

Covers:
  - parser_strategy: accepted values ("auto", "document", "code", "text") and
    rejected unknown values (FR-3301 AC 3).
  - chunker: accepted values ("native", "markdown") and rejected unknown values
    (FR-3322); "markdown" emits a warning (FR-3323).
  - VLM mutual exclusion: vlm_mode="builtin" + enable_multimodal_processing=True
    is a fatal error (FR-3340, FR-3341).
  - VLM coexistence: vlm_mode="external" + enable_multimodal_processing=True
    emits a warning (FR-3342).
  - VLM enrichment + code parser: enable_vlm_enrichment=True +
    parser_strategy="code" is a fatal error (FR-3341).
  - Integration: valid configs pass; errors accumulate correctly.
"""

from __future__ import annotations

import pytest

from src.ingest.common.types import IngestionConfig
from src.ingest.impl import verify_core_design


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(**overrides) -> IngestionConfig:
    """Build an IngestionConfig with sensible safe defaults plus overrides.

    Safe defaults ensure no pre-existing validators fire accidentally:
    - chunk_size > chunk_overlap
    - build_kg=False (avoids KG storage cross-checks without storage enabled)
    - enable_docling_parser=False (avoids docling_model empty-string check)
    - vlm_mode="disabled"
    - enable_multimodal_processing=False
    - enable_vision_processing=False
    - enable_visual_embedding=False
    - enable_knowledge_graph_storage=False
    - parser_strategy="auto", chunker="native"
    """
    defaults = dict(
        chunk_size=512,
        chunk_overlap=64,
        build_kg=False,
        enable_docling_parser=False,
        vlm_mode="disabled",
        enable_multimodal_processing=False,
        enable_vision_processing=False,
        enable_visual_embedding=False,
        enable_knowledge_graph_storage=False,
        enable_knowledge_graph_extraction=False,
        parser_strategy="auto",
        chunker="native",
    )
    defaults.update(overrides)
    return IngestionConfig(**defaults)


# ---------------------------------------------------------------------------
# parser_strategy validation (Rule 1, FR-3301 AC 3)
# ---------------------------------------------------------------------------


class TestParserStrategyValidation:
    """Tests for parser_strategy field validation."""

    @pytest.mark.parametrize("strategy", ["auto", "document", "code", "text"])
    def test_valid_parser_strategy_passes(self, strategy: str):
        """All four accepted parser_strategy values should not produce errors."""
        cfg = _cfg(parser_strategy=strategy)
        report = verify_core_design(cfg)

        strategy_errors = [e for e in report.errors if "parser_strategy" in e]
        assert not strategy_errors, (
            f"parser_strategy={strategy!r} should be valid but got errors: {strategy_errors}"
        )

    def test_invalid_parser_strategy_produces_error(self):
        """An unknown parser_strategy should produce a fatal error."""
        cfg = _cfg(parser_strategy="unknown_strategy")
        report = verify_core_design(cfg)

        assert report.ok is False
        assert any("parser_strategy" in e for e in report.errors)

    def test_invalid_parser_strategy_error_message(self):
        """Error message should include the bad value and the accepted values."""
        cfg = _cfg(parser_strategy="neural")
        report = verify_core_design(cfg)

        matching = [e for e in report.errors if "parser_strategy" in e]
        assert matching
        assert "neural" in matching[0]

    @pytest.mark.parametrize("bad", ["AUTO", "Doc", " auto", "auto ", ""])
    def test_case_sensitive_and_whitespace_rejected(self, bad: str):
        """parser_strategy is case-sensitive; whitespace variants must be rejected."""
        cfg = _cfg(parser_strategy=bad)
        report = verify_core_design(cfg)

        assert any("parser_strategy" in e for e in report.errors), (
            f"parser_strategy={bad!r} should be invalid but was accepted"
        )


# ---------------------------------------------------------------------------
# chunker validation (Rule 2 + Rule 3, FR-3322, FR-3323)
# ---------------------------------------------------------------------------


class TestChunkerValidation:
    """Tests for chunker field validation."""

    @pytest.mark.parametrize("chunker", ["native", "markdown"])
    def test_valid_chunker_passes(self, chunker: str):
        """Both accepted chunker values should not produce errors."""
        cfg = _cfg(chunker=chunker)
        report = verify_core_design(cfg)

        chunker_errors = [e for e in report.errors if "chunker" in e]
        assert not chunker_errors, (
            f"chunker={chunker!r} should be valid but got errors: {chunker_errors}"
        )

    def test_invalid_chunker_produces_error(self):
        """An unknown chunker value should produce a fatal error."""
        cfg = _cfg(chunker="deepdoc")
        report = verify_core_design(cfg)

        assert report.ok is False
        assert any("chunker" in e for e in report.errors)

    def test_invalid_chunker_error_mentions_valid_values(self):
        """Error message for bad chunker should mention the valid options."""
        cfg = _cfg(chunker="sentence")
        report = verify_core_design(cfg)

        matching = [e for e in report.errors if "chunker" in e]
        assert matching
        msg = matching[0]
        assert "native" in msg or "markdown" in msg

    def test_markdown_chunker_emits_warning(self):
        """chunker='markdown' override should emit a non-fatal warning."""
        cfg = _cfg(chunker="markdown")
        report = verify_core_design(cfg)

        chunker_warnings = [w for w in report.warnings if "chunker" in w.lower() or "markdown" in w.lower()]
        assert chunker_warnings, (
            "chunker='markdown' should emit a warning about native chunking being disabled"
        )

    def test_native_chunker_no_warning(self):
        """chunker='native' (default) should not emit any chunker-related warning."""
        cfg = _cfg(chunker="native")
        report = verify_core_design(cfg)

        chunker_warnings = [
            w for w in report.warnings
            if "chunker" in w.lower() and "markdown" in w.lower()
        ]
        assert not chunker_warnings


# ---------------------------------------------------------------------------
# VLM mutual exclusion (Rule 4, FR-3340, FR-3341)
# ---------------------------------------------------------------------------


class TestVlmMutualExclusion:
    """Tests for VLM mutual exclusion: vlm_mode='builtin' + multimodal is fatal."""

    def test_builtin_vlm_with_multimodal_is_error(self):
        """vlm_mode='builtin' + enable_multimodal_processing=True must be a fatal error."""
        cfg = _cfg(vlm_mode="builtin", enable_multimodal_processing=True)
        report = verify_core_design(cfg)

        assert report.ok is False
        vlm_errors = [e for e in report.errors if "builtin" in e or "mutually exclusive" in e]
        assert vlm_errors, (
            "Expected error for builtin VLM + multimodal combination, "
            f"but got: {report.errors}"
        )

    def test_builtin_vlm_without_multimodal_passes(self):
        """vlm_mode='builtin' alone (no multimodal) must not produce mutual-exclusion error."""
        # Note: vlm_mode='builtin' also requires docling to be installed; if it's not
        # installed in this test environment, we just check there is no *mutual-exclusion* error.
        cfg = _cfg(vlm_mode="builtin", enable_multimodal_processing=False)
        report = verify_core_design(cfg)

        mutual_excl_errors = [
            e for e in report.errors
            if "mutually exclusive" in e or "double VLM" in e
        ]
        assert not mutual_excl_errors

    def test_disabled_vlm_with_multimodal_passes(self):
        """vlm_mode='disabled' + enable_multimodal_processing=True is valid."""
        cfg = _cfg(vlm_mode="disabled", enable_multimodal_processing=True)
        report = verify_core_design(cfg)

        vlm_errors = [e for e in report.errors if "mutually exclusive" in e]
        assert not vlm_errors

    def test_external_vlm_with_multimodal_is_warning_not_error(self):
        """vlm_mode='external' + enable_multimodal_processing=True is a warning, not an error."""
        cfg = _cfg(vlm_mode="external", enable_multimodal_processing=True)
        report = verify_core_design(cfg)

        # No fatal error for external+multimodal
        mutual_excl_errors = [e for e in report.errors if "mutually exclusive" in e]
        assert not mutual_excl_errors

        # But a warning should be present (FR-3342)
        coexist_warnings = [w for w in report.warnings if "external" in w and "multimodal" in w.lower()]
        assert coexist_warnings, (
            "Expected a coexistence warning for vlm_mode='external' + multimodal. "
            f"Got warnings: {report.warnings}"
        )


# ---------------------------------------------------------------------------
# VLM enrichment + code parser (Rule 6, FR-3341)
# ---------------------------------------------------------------------------


class TestVlmEnrichmentCodeConflict:
    """Tests for VLM enrichment incompatibility with code parser strategy."""

    def test_vlm_enrichment_with_code_strategy_is_error(self):
        """enable_vlm_enrichment=True + parser_strategy='code' must be a fatal error."""
        # enable_vlm_enrichment is not yet a declared IngestionConfig field;
        # the validator reads it via getattr(..., False), so we inject it post-construction.
        cfg = _cfg(parser_strategy="code")
        cfg.enable_vlm_enrichment = True  # type: ignore[attr-defined]  # not yet a declared field
        report = verify_core_design(cfg)

        assert report.ok is False
        vlm_code_errors = [
            e for e in report.errors
            if "vlm" in e.lower() and "code" in e.lower()
        ]
        assert vlm_code_errors, (
            "Expected error for VLM enrichment + code strategy combination, "
            f"but got: {report.errors}"
        )

    def test_vlm_enrichment_with_text_strategy_passes(self):
        """enable_vlm_enrichment=True + parser_strategy='text' should not error."""
        cfg = _cfg(parser_strategy="text")
        cfg.enable_vlm_enrichment = True  # type: ignore[attr-defined]  # not yet a declared field
        report = verify_core_design(cfg)

        vlm_code_errors = [
            e for e in report.errors
            if "vlm" in e.lower() and "code" in e.lower()
        ]
        assert not vlm_code_errors

    def test_vlm_enrichment_disabled_with_code_strategy_passes(self):
        """enable_vlm_enrichment=False (default) + parser_strategy='code' should not error."""
        cfg = _cfg(parser_strategy="code")
        # enable_vlm_enrichment not set → defaults to False
        report = verify_core_design(cfg)

        vlm_code_errors = [
            e for e in report.errors
            if "vlm" in e.lower() and "code" in e.lower()
        ]
        assert not vlm_code_errors

    @pytest.mark.parametrize("strategy", ["auto", "document", "text"])
    def test_vlm_enrichment_compatible_strategies(self, strategy: str):
        """VLM enrichment should not produce errors for non-code strategies."""
        cfg = _cfg(parser_strategy=strategy)
        cfg.enable_vlm_enrichment = True  # type: ignore[attr-defined]  # not yet a declared field
        report = verify_core_design(cfg)

        vlm_code_errors = [
            e for e in report.errors
            if "vlm" in e.lower() and "code" in e.lower()
        ]
        assert not vlm_code_errors, (
            f"parser_strategy={strategy!r} + vlm enrichment should be valid, "
            f"but got errors: {vlm_code_errors}"
        )


# ---------------------------------------------------------------------------
# Integration: multiple rule violations accumulate
# ---------------------------------------------------------------------------


class TestVerifyCoreDesignIntegration:
    """Integration tests: multiple violations accumulate; valid configs pass cleanly."""

    def test_clean_default_config_passes(self):
        """Default IngestionConfig (with parser_strategy and chunker defaults) passes."""
        cfg = _cfg()
        report = verify_core_design(cfg)

        # The safe _cfg() defaults should produce no errors.
        assert report.ok is True
        assert not report.errors

    def test_multiple_parser_abstraction_violations_accumulate(self):
        """Multiple bad parser-abstraction fields should all appear in errors."""
        cfg = _cfg(
            parser_strategy="bogus_strategy",
            chunker="bogus_chunker",
            vlm_mode="builtin",
            enable_multimodal_processing=True,
        )
        report = verify_core_design(cfg)

        assert report.ok is False
        # Expect at least: bad parser_strategy, bad chunker, builtin+multimodal conflict
        assert len(report.errors) >= 3

    def test_pre_existing_kg_error_still_reported(self):
        """Pre-existing KG validation still fires alongside new parser checks."""
        cfg = _cfg(
            build_kg=False,
            enable_knowledge_graph_extraction=False,
            enable_knowledge_graph_storage=True,
            parser_strategy="bogus",
        )
        report = verify_core_design(cfg)

        assert report.ok is False
        # Both KG error and parser_strategy error should be present.
        kg_errors = [e for e in report.errors if "knowledge_graph_storage" in e]
        ps_errors = [e for e in report.errors if "parser_strategy" in e]
        assert kg_errors, "KG storage error should still be reported"
        assert ps_errors, "parser_strategy error should also be reported"

    def test_valid_auto_strategy_native_chunker_no_errors(self):
        """The recommended defaults (parser_strategy='auto', chunker='native') produce no errors."""
        cfg = _cfg(parser_strategy="auto", chunker="native")
        report = verify_core_design(cfg)

        parser_errors = [
            e for e in report.errors
            if "parser_strategy" in e or "chunker" in e
        ]
        assert not parser_errors

    def test_all_four_strategies_with_native_chunker_pass(self):
        """All valid strategy + native chunker combinations should not produce errors."""
        for strategy in ("auto", "document", "code", "text"):
            cfg = _cfg(parser_strategy=strategy, chunker="native")
            report = verify_core_design(cfg)

            parser_errors = [e for e in report.errors if "parser_strategy" in e]
            chunker_errors = [e for e in report.errors if "chunker" in e]
            assert not parser_errors, f"strategy={strategy!r} should be valid"
            assert not chunker_errors, f"native chunker should be valid for strategy={strategy!r}"
