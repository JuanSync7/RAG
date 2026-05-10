# @summary
# Tests for visual embedding config, state, and pipeline wiring.
# Covers: config/settings.py, src/ingest/common/types.py, src/ingest/embedding/state.py,
#          src/ingest/embedding/workflow.py
# Exports: (pytest test functions)
# Deps: pytest, importlib, config.settings, src.ingest.common.types, src.ingest.embedding.state,
#        src.ingest.embedding.workflow, src.ingest.impl
# @end-summary
"""Tests for visual embedding configuration constants, IngestionConfig fields,
_check_visual_embedding_config validation, EmbeddingPipelineState extensions,
PIPELINE_NODE_NAMES ordering, IngestFileResult defaults, and graph topology."""

import importlib
import os
import pytest

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _reload_settings():
    """Force config.settings to re-evaluate all os.environ.get() calls."""
    import config.settings as s
    importlib.reload(s)
    return s


# ===========================================================================
# TestSettingsVisualDefaults
# ===========================================================================

class TestSettingsVisualDefaults:
    """FR-S01: Default values for all visual embedding settings constants."""

    def test_enable_visual_embedding_default_false(self, monkeypatch):
        """FR-S01a: RAG_INGESTION_ENABLE_VISUAL_EMBEDDING defaults to False."""
        monkeypatch.delenv("RAG_INGESTION_ENABLE_VISUAL_EMBEDDING", raising=False)
        s = _reload_settings()
        assert s.RAG_INGESTION_ENABLE_VISUAL_EMBEDDING is False

    def test_visual_target_collection_default(self, monkeypatch):
        """FR-S01b: RAG_INGESTION_VISUAL_TARGET_COLLECTION defaults to 'RAGVisualPages'."""
        monkeypatch.delenv("RAG_INGESTION_VISUAL_TARGET_COLLECTION", raising=False)
        s = _reload_settings()
        assert s.RAG_INGESTION_VISUAL_TARGET_COLLECTION == "RAGVisualPages"

    def test_colqwen_model_default(self, monkeypatch):
        """FR-S01c: RAG_INGESTION_COLQWEN_MODEL defaults to 'vidore/colqwen2-v1.0'."""
        monkeypatch.delenv("RAG_INGESTION_COLQWEN_MODEL", raising=False)
        s = _reload_settings()
        assert s.RAG_INGESTION_COLQWEN_MODEL == "vidore/colqwen2-v1.0"

    def test_colqwen_batch_size_default(self, monkeypatch):
        """FR-S01d: RAG_INGESTION_COLQWEN_BATCH_SIZE defaults to integer 4."""
        monkeypatch.delenv("RAG_INGESTION_COLQWEN_BATCH_SIZE", raising=False)
        s = _reload_settings()
        assert s.RAG_INGESTION_COLQWEN_BATCH_SIZE == 4
        assert isinstance(s.RAG_INGESTION_COLQWEN_BATCH_SIZE, int)

    def test_page_image_quality_default(self, monkeypatch):
        """FR-S01e: RAG_INGESTION_PAGE_IMAGE_QUALITY defaults to integer 85."""
        monkeypatch.delenv("RAG_INGESTION_PAGE_IMAGE_QUALITY", raising=False)
        s = _reload_settings()
        assert s.RAG_INGESTION_PAGE_IMAGE_QUALITY == 85
        assert isinstance(s.RAG_INGESTION_PAGE_IMAGE_QUALITY, int)

    def test_page_image_max_dimension_default(self, monkeypatch):
        """FR-S01f: RAG_INGESTION_PAGE_IMAGE_MAX_DIMENSION defaults to integer 1024."""
        monkeypatch.delenv("RAG_INGESTION_PAGE_IMAGE_MAX_DIMENSION", raising=False)
        s = _reload_settings()
        assert s.RAG_INGESTION_PAGE_IMAGE_MAX_DIMENSION == 1024
        assert isinstance(s.RAG_INGESTION_PAGE_IMAGE_MAX_DIMENSION, int)


# ===========================================================================
# TestSettingsBooleanParsing
# ===========================================================================

class TestSettingsBooleanParsing:
    """FR-S02: Boolean env var parsing for ENABLE_VISUAL_EMBEDDING."""

    def test_enable_true_via_lowercase_true(self, monkeypatch):
        """FR-S02a: 'true' evaluates to True."""
        monkeypatch.setenv("RAG_INGESTION_ENABLE_VISUAL_EMBEDDING", "true")
        s = _reload_settings()
        assert s.RAG_INGESTION_ENABLE_VISUAL_EMBEDDING is True

    def test_enable_true_via_one(self, monkeypatch):
        """FR-S02b: '1' evaluates to True."""
        monkeypatch.setenv("RAG_INGESTION_ENABLE_VISUAL_EMBEDDING", "1")
        s = _reload_settings()
        assert s.RAG_INGESTION_ENABLE_VISUAL_EMBEDDING is True

    def test_enable_true_via_yes(self, monkeypatch):
        """FR-S02c: 'yes' evaluates to True."""
        monkeypatch.setenv("RAG_INGESTION_ENABLE_VISUAL_EMBEDDING", "yes")
        s = _reload_settings()
        assert s.RAG_INGESTION_ENABLE_VISUAL_EMBEDDING is True

    def test_enable_false_via_explicit_false(self, monkeypatch):
        """FR-S02d: 'false' evaluates to False."""
        monkeypatch.setenv("RAG_INGESTION_ENABLE_VISUAL_EMBEDDING", "false")
        s = _reload_settings()
        assert s.RAG_INGESTION_ENABLE_VISUAL_EMBEDDING is False

    def test_enable_false_via_uppercase_true(self, monkeypatch):
        """FR-S02e: 'TRUE' (uppercase) evaluates to False — case-sensitive parsing.
        NOTE: observed behavior; spec says only 'true', '1', 'yes' are truthy.
        """
        monkeypatch.setenv("RAG_INGESTION_ENABLE_VISUAL_EMBEDDING", "TRUE")
        s = _reload_settings()
        assert s.RAG_INGESTION_ENABLE_VISUAL_EMBEDDING is False

    def test_enable_false_via_uppercase_yes(self, monkeypatch):
        """FR-S02f: 'YES' (uppercase) evaluates to False — case-sensitive parsing.
        NOTE: observed behavior; spec says only lowercase 'yes' is truthy.
        """
        monkeypatch.setenv("RAG_INGESTION_ENABLE_VISUAL_EMBEDDING", "YES")
        s = _reload_settings()
        assert s.RAG_INGESTION_ENABLE_VISUAL_EMBEDDING is False

    def test_enable_false_via_yes_please(self, monkeypatch):
        """FR-S02g: 'yes_please' evaluates to False."""
        monkeypatch.setenv("RAG_INGESTION_ENABLE_VISUAL_EMBEDDING", "yes_please")
        s = _reload_settings()
        assert s.RAG_INGESTION_ENABLE_VISUAL_EMBEDDING is False

    def test_enable_false_via_on(self, monkeypatch):
        """FR-S02h: 'on' evaluates to False — 'on' is not in the truthy set."""
        monkeypatch.setenv("RAG_INGESTION_ENABLE_VISUAL_EMBEDDING", "on")
        s = _reload_settings()
        assert s.RAG_INGESTION_ENABLE_VISUAL_EMBEDDING is False

    def test_enable_false_via_empty_string(self, monkeypatch):
        """FR-S02i: '' (empty string) evaluates to False."""
        monkeypatch.setenv("RAG_INGESTION_ENABLE_VISUAL_EMBEDDING", "")
        s = _reload_settings()
        assert s.RAG_INGESTION_ENABLE_VISUAL_EMBEDDING is False


# ===========================================================================
# TestSettingsIntegerParsing
# ===========================================================================

class TestSettingsIntegerParsing:
    """FR-S03: Integer env var parsing for batch size, quality, and max dimension."""

    def test_custom_collection_name(self, monkeypatch):
        """FR-S03a: Custom visual target collection string is used."""
        monkeypatch.setenv("RAG_INGESTION_VISUAL_TARGET_COLLECTION", "TestVisual")
        s = _reload_settings()
        assert s.RAG_INGESTION_VISUAL_TARGET_COLLECTION == "TestVisual"

    def test_custom_model_name(self, monkeypatch):
        """FR-S03b: Custom colqwen model string is used."""
        monkeypatch.setenv("RAG_INGESTION_COLQWEN_MODEL", "vidore/other-model")
        s = _reload_settings()
        assert s.RAG_INGESTION_COLQWEN_MODEL == "vidore/other-model"

    def test_custom_batch_size(self, monkeypatch):
        """FR-S03c: RAG_INGESTION_COLQWEN_BATCH_SIZE='8' -> int 8."""
        monkeypatch.setenv("RAG_INGESTION_COLQWEN_BATCH_SIZE", "8")
        s = _reload_settings()
        assert s.RAG_INGESTION_COLQWEN_BATCH_SIZE == 8
        assert isinstance(s.RAG_INGESTION_COLQWEN_BATCH_SIZE, int)

    def test_custom_image_quality(self, monkeypatch):
        """FR-S03d: RAG_INGESTION_PAGE_IMAGE_QUALITY='90' -> int 90."""
        monkeypatch.setenv("RAG_INGESTION_PAGE_IMAGE_QUALITY", "90")
        s = _reload_settings()
        assert s.RAG_INGESTION_PAGE_IMAGE_QUALITY == 90
        assert isinstance(s.RAG_INGESTION_PAGE_IMAGE_QUALITY, int)

    def test_custom_max_dimension(self, monkeypatch):
        """FR-S03e: RAG_INGESTION_PAGE_IMAGE_MAX_DIMENSION='2048' -> int 2048."""
        monkeypatch.setenv("RAG_INGESTION_PAGE_IMAGE_MAX_DIMENSION", "2048")
        s = _reload_settings()
        assert s.RAG_INGESTION_PAGE_IMAGE_MAX_DIMENSION == 2048
        assert isinstance(s.RAG_INGESTION_PAGE_IMAGE_MAX_DIMENSION, int)


# ===========================================================================
# TestSettingsErrors
# ===========================================================================

class TestSettingsErrors:
    """FR-S04: settings.py raises ValueError for non-integer integer env vars."""

    def test_non_integer_batch_size_raises(self, monkeypatch):
        """FR-S04a: RAG_INGESTION_COLQWEN_BATCH_SIZE='abc' raises ValueError at import time."""
        monkeypatch.setenv("RAG_INGESTION_COLQWEN_BATCH_SIZE", "abc")
        with pytest.raises((ValueError, SystemExit)):
            _reload_settings()

    def test_non_integer_quality_raises(self, monkeypatch):
        """FR-S04b: RAG_INGESTION_PAGE_IMAGE_QUALITY='abc' raises ValueError at import time."""
        monkeypatch.setenv("RAG_INGESTION_PAGE_IMAGE_QUALITY", "abc")
        with pytest.raises((ValueError, SystemExit)):
            _reload_settings()

    def test_non_integer_max_dimension_raises(self, monkeypatch):
        """FR-S04c: RAG_INGESTION_PAGE_IMAGE_MAX_DIMENSION='abc' raises ValueError at import time."""
        monkeypatch.setenv("RAG_INGESTION_PAGE_IMAGE_MAX_DIMENSION", "abc")
        with pytest.raises((ValueError, SystemExit)):
            _reload_settings()


# ===========================================================================
# TestIngestionConfigVisual
# ===========================================================================

class TestIngestionConfigVisual:
    """FR-IC01: IngestionConfig visual embedding fields and generate_page_images property."""

    def test_defaults_match_settings_constants(self, monkeypatch):
        """FR-IC01a: IngestionConfig() defaults match the six settings.py constants."""
        # Ensure clean env for settings
        for var in [
            "RAG_INGESTION_ENABLE_VISUAL_EMBEDDING",
            "RAG_INGESTION_VISUAL_TARGET_COLLECTION",
            "RAG_INGESTION_COLQWEN_MODEL",
            "RAG_INGESTION_COLQWEN_BATCH_SIZE",
            "RAG_INGESTION_PAGE_IMAGE_QUALITY",
            "RAG_INGESTION_PAGE_IMAGE_MAX_DIMENSION",
        ]:
            monkeypatch.delenv(var, raising=False)
        s = _reload_settings()

        from src.ingest.common.types import IngestionConfig
        cfg = IngestionConfig()

        assert cfg.enable_visual_embedding == s.RAG_INGESTION_ENABLE_VISUAL_EMBEDDING
        assert cfg.visual_target_collection == s.RAG_INGESTION_VISUAL_TARGET_COLLECTION
        assert cfg.colqwen_model_name == s.RAG_INGESTION_COLQWEN_MODEL
        assert cfg.colqwen_batch_size == s.RAG_INGESTION_COLQWEN_BATCH_SIZE
        assert cfg.page_image_quality == s.RAG_INGESTION_PAGE_IMAGE_QUALITY
        assert cfg.page_image_max_dimension == s.RAG_INGESTION_PAGE_IMAGE_MAX_DIMENSION

    def test_generate_page_images_when_enabled(self):
        """FR-IC01b: generate_page_images returns True when enable_visual_embedding=True."""
        from src.ingest.common.types import IngestionConfig
        cfg = IngestionConfig(enable_visual_embedding=True)
        assert cfg.generate_page_images is True

    def test_generate_page_images_when_disabled(self):
        """FR-IC01c: generate_page_images returns False when enable_visual_embedding=False."""
        from src.ingest.common.types import IngestionConfig
        cfg = IngestionConfig(enable_visual_embedding=False)
        assert cfg.generate_page_images is False

    def test_generate_page_images_is_property_not_stored_field(self):
        """FR-IC01d: generate_page_images must not appear as a stored key in __dict__."""
        from src.ingest.common.types import IngestionConfig
        cfg = IngestionConfig(enable_visual_embedding=True)
        # Dataclasses store fields in __dict__; a property is not stored there
        assert "generate_page_images" not in cfg.__dict__


# ===========================================================================
# TestCheckVisualEmbeddingConfig
# ===========================================================================

class TestCheckVisualEmbeddingConfig:
    """FR-CV01: _check_visual_embedding_config validation logic."""

    def _make_valid_config(self, **overrides):
        """Build a minimal valid IngestionConfig for visual embedding tests."""
        from src.ingest.common.types import IngestionConfig
        defaults = dict(
            enable_visual_embedding=True,
            enable_docling_parser=True,
            colqwen_batch_size=4,
            page_image_quality=85,
            page_image_max_dimension=1024,
        )
        defaults.update(overrides)
        return IngestionConfig(**defaults)

    def test_disabled_fast_path_returns_empty_tuple(self):
        """FR-CV01a: When enable_visual_embedding=False, returns ([], []) immediately."""
        from src.ingest.common.types import IngestionConfig
        from src.ingest.impl import _check_visual_embedding_config
        cfg = IngestionConfig(enable_visual_embedding=False)
        result = _check_visual_embedding_config(cfg)
        errors, warnings = result
        assert errors == []
        assert warnings == []

    def test_disabled_fast_path_skips_range_checks_even_for_invalid_values(self):
        """FR-CV01b: Fast-path skips range validation even when batch_size=0."""
        from src.ingest.common.types import IngestionConfig
        from src.ingest.impl import _check_visual_embedding_config
        # colqwen_batch_size=0 would normally be invalid, but fast-path prevents checks
        cfg = IngestionConfig(
            enable_visual_embedding=False,
            colqwen_batch_size=0,
            page_image_quality=0,
            page_image_max_dimension=0,
        )
        result = _check_visual_embedding_config(cfg)
        errors, warnings = result
        assert errors == []

    def test_valid_config_returns_empty_errors(self):
        """FR-CV01c: Valid config with all in-range values returns ([], [])."""
        from src.ingest.impl import _check_visual_embedding_config
        cfg = self._make_valid_config(
            colqwen_batch_size=16,
            page_image_quality=75,
            page_image_max_dimension=1024,
        )
        errors, warnings = _check_visual_embedding_config(cfg)
        assert errors == []

    def test_return_value_is_always_two_tuple(self):
        """FR-CV01d: Return value is always a 2-tuple (list, list), never raises."""
        from src.ingest.impl import _check_visual_embedding_config
        cfg = self._make_valid_config()
        result = _check_visual_embedding_config(cfg)
        assert isinstance(result, tuple)
        assert len(result) == 2
        errors, warnings = result
        assert isinstance(errors, list)
        assert isinstance(warnings, list)

    # --- Boundary: batch size ---

    def test_boundary_batch_size_minimum_valid(self):
        """FR-CV01e: colqwen_batch_size=1 (minimum valid) returns no batch-size error."""
        from src.ingest.impl import _check_visual_embedding_config
        cfg = self._make_valid_config(colqwen_batch_size=1)
        errors, _ = _check_visual_embedding_config(cfg)
        batch_errors = [e for e in errors if "colqwen_batch_size" in e]
        assert batch_errors == []

    def test_boundary_batch_size_maximum_valid(self):
        """FR-CV01f: colqwen_batch_size=32 (maximum valid) returns no batch-size error."""
        from src.ingest.impl import _check_visual_embedding_config
        cfg = self._make_valid_config(colqwen_batch_size=32)
        errors, _ = _check_visual_embedding_config(cfg)
        batch_errors = [e for e in errors if "colqwen_batch_size" in e]
        assert batch_errors == []

    def test_boundary_batch_size_below_range(self):
        """FR-CV01g: colqwen_batch_size=0 returns error naming field and range 1-32."""
        from src.ingest.impl import _check_visual_embedding_config
        cfg = self._make_valid_config(colqwen_batch_size=0)
        errors, _ = _check_visual_embedding_config(cfg)
        assert len(errors) > 0
        combined = " ".join(errors)
        assert "colqwen_batch_size" in combined
        # Range 1-32 should be mentioned
        assert "1" in combined and "32" in combined

    def test_boundary_batch_size_above_range(self):
        """FR-CV01h: colqwen_batch_size=64 returns error naming field and range 1-32."""
        from src.ingest.impl import _check_visual_embedding_config
        cfg = self._make_valid_config(colqwen_batch_size=64)
        errors, _ = _check_visual_embedding_config(cfg)
        assert len(errors) > 0
        combined = " ".join(errors)
        assert "colqwen_batch_size" in combined
        assert "1" in combined and "32" in combined

    # --- Boundary: quality ---

    def test_boundary_quality_minimum_valid(self):
        """FR-CV01i: page_image_quality=1 (minimum valid) returns no quality error."""
        from src.ingest.impl import _check_visual_embedding_config
        cfg = self._make_valid_config(page_image_quality=1)
        errors, _ = _check_visual_embedding_config(cfg)
        quality_errors = [e for e in errors if "page_image_quality" in e]
        assert quality_errors == []

    def test_boundary_quality_maximum_valid(self):
        """FR-CV01j: page_image_quality=100 (maximum valid) returns no quality error."""
        from src.ingest.impl import _check_visual_embedding_config
        cfg = self._make_valid_config(page_image_quality=100)
        errors, _ = _check_visual_embedding_config(cfg)
        quality_errors = [e for e in errors if "page_image_quality" in e]
        assert quality_errors == []

    def test_boundary_quality_below_range(self):
        """FR-CV01k: page_image_quality=0 returns error naming field and range 1-100."""
        from src.ingest.impl import _check_visual_embedding_config
        cfg = self._make_valid_config(page_image_quality=0)
        errors, _ = _check_visual_embedding_config(cfg)
        assert len(errors) > 0
        combined = " ".join(errors)
        assert "page_image_quality" in combined
        assert "1" in combined and "100" in combined

    def test_boundary_quality_above_range(self):
        """FR-CV01l: page_image_quality=101 returns error naming field and range 1-100."""
        from src.ingest.impl import _check_visual_embedding_config
        cfg = self._make_valid_config(page_image_quality=101)
        errors, _ = _check_visual_embedding_config(cfg)
        assert len(errors) > 0
        combined = " ".join(errors)
        assert "page_image_quality" in combined
        assert "1" in combined and "100" in combined

    # --- Boundary: dimension ---

    def test_boundary_dimension_minimum_valid(self):
        """FR-CV01m: page_image_max_dimension=256 (minimum valid) returns no dimension error."""
        from src.ingest.impl import _check_visual_embedding_config
        cfg = self._make_valid_config(page_image_max_dimension=256)
        errors, _ = _check_visual_embedding_config(cfg)
        dim_errors = [e for e in errors if "page_image_max_dimension" in e]
        assert dim_errors == []

    def test_boundary_dimension_maximum_valid(self):
        """FR-CV01n: page_image_max_dimension=4096 (maximum valid) returns no dimension error."""
        from src.ingest.impl import _check_visual_embedding_config
        cfg = self._make_valid_config(page_image_max_dimension=4096)
        errors, _ = _check_visual_embedding_config(cfg)
        dim_errors = [e for e in errors if "page_image_max_dimension" in e]
        assert dim_errors == []

    def test_boundary_dimension_below_range(self):
        """FR-CV01o: page_image_max_dimension=128 returns error naming field and range 256-4096."""
        from src.ingest.impl import _check_visual_embedding_config
        cfg = self._make_valid_config(page_image_max_dimension=128)
        errors, _ = _check_visual_embedding_config(cfg)
        assert len(errors) > 0
        combined = " ".join(errors)
        assert "page_image_max_dimension" in combined
        assert "256" in combined and "4096" in combined

    def test_boundary_dimension_above_range(self):
        """FR-CV01p: page_image_max_dimension=8192 returns error naming field and range 256-4096."""
        from src.ingest.impl import _check_visual_embedding_config
        cfg = self._make_valid_config(page_image_max_dimension=8192)
        errors, _ = _check_visual_embedding_config(cfg)
        assert len(errors) > 0
        combined = " ".join(errors)
        assert "page_image_max_dimension" in combined
        assert "256" in combined and "4096" in combined

    # --- Docling dependency ---

    def test_docling_not_enabled_raises_error(self):
        """FR-CV01q: enable_docling_parser=False with visual embedding enabled returns error."""
        from src.ingest.impl import _check_visual_embedding_config
        cfg = self._make_valid_config(enable_docling_parser=False)
        errors, _ = _check_visual_embedding_config(cfg)
        assert len(errors) > 0
        combined = " ".join(errors).lower()
        # Error message should name Docling or docling_parser
        assert "docling" in combined

    # --- Multiple simultaneous violations ---

    def test_multiple_violations_returns_multiple_errors(self):
        """FR-CV01r: Multiple out-of-range values produce one error per violation (>=3)."""
        from src.ingest.impl import _check_visual_embedding_config
        cfg = self._make_valid_config(
            colqwen_batch_size=0,
            page_image_quality=0,
            page_image_max_dimension=128,
        )
        errors, _ = _check_visual_embedding_config(cfg)
        assert len(errors) >= 3


# ===========================================================================
# TestPipelineNodeNames
# ===========================================================================

class TestPipelineNodeNames:
    """FR-PN01: PIPELINE_NODE_NAMES list membership and ordering."""

    def test_visual_embedding_present(self):
        """FR-PN01a: 'visual_embedding' is present in PIPELINE_NODE_NAMES."""
        from src.ingest.common.types import PIPELINE_NODE_NAMES
        assert "visual_embedding" in PIPELINE_NODE_NAMES

    def test_visual_embedding_after_embedding_storage(self):
        """FR-PN01b: 'visual_embedding' appears immediately after 'embedding_storage'."""
        from src.ingest.common.types import PIPELINE_NODE_NAMES
        idx_storage = PIPELINE_NODE_NAMES.index("embedding_storage")
        idx_visual = PIPELINE_NODE_NAMES.index("visual_embedding")
        assert idx_visual == idx_storage + 1

    def test_visual_embedding_before_knowledge_graph_storage(self):
        """FR-PN01c: 'visual_embedding' appears immediately before 'knowledge_graph_storage'."""
        from src.ingest.common.types import PIPELINE_NODE_NAMES
        idx_visual = PIPELINE_NODE_NAMES.index("visual_embedding")
        idx_kg = PIPELINE_NODE_NAMES.index("knowledge_graph_storage")
        assert idx_kg == idx_visual + 1

    def test_total_count_is_fifteen(self):
        """FR-PN01d: PIPELINE_NODE_NAMES contains exactly 15 entries."""
        from src.ingest.common.types import PIPELINE_NODE_NAMES
        assert len(PIPELINE_NODE_NAMES) == 15


# ===========================================================================
# TestIngestFileResult
# ===========================================================================

class TestIngestFileResult:
    """FR-IF01: IngestFileResult default visual_stored_count field."""

    def _minimal_result(self, **overrides):
        """Build an IngestFileResult with all required fields supplied."""
        from src.ingest.common.types import IngestFileResult
        defaults = dict(
            errors=[],
            stored_count=0,
            metadata_summary="",
            metadata_keywords=[],
            processing_log=[],
            source_hash="",
            clean_hash="",
        )
        defaults.update(overrides)
        return IngestFileResult(**defaults)

    def test_visual_stored_count_defaults_to_zero(self):
        """FR-IF01a: IngestFileResult() visual_stored_count field defaults to 0."""
        result = self._minimal_result()
        assert result.visual_stored_count == 0

    def test_visual_stored_count_can_be_set(self):
        """FR-IF01b: IngestFileResult.visual_stored_count accepts a non-zero value."""
        result = self._minimal_result(visual_stored_count=42)
        assert result.visual_stored_count == 42

    def test_visual_stored_count_is_int(self):
        """FR-IF01c: IngestFileResult.visual_stored_count is of type int."""
        result = self._minimal_result()
        assert isinstance(result.visual_stored_count, int)


# ===========================================================================
# TestEmbeddingPipelineState
# ===========================================================================

class TestEmbeddingPipelineState:
    """FR-ES01: EmbeddingPipelineState extensions for visual embedding."""

    def test_visual_stored_count_accessible(self):
        """FR-ES01a: state['visual_stored_count'] is accessible when set."""
        from src.ingest.embedding.state import EmbeddingPipelineState
        state: EmbeddingPipelineState = {"visual_stored_count": 5}  # type: ignore[typeddict-item]
        assert state["visual_stored_count"] == 5

    def test_page_images_accessible(self):
        """FR-ES01b: state['page_images'] is accessible when set."""
        from src.ingest.embedding.state import EmbeddingPipelineState
        mock_images = [object(), object()]
        state: EmbeddingPipelineState = {"page_images": mock_images}  # type: ignore[typeddict-item]
        assert state["page_images"] is mock_images

    def test_visual_stored_count_safe_default_via_get(self):
        """FR-ES01c: state.get('visual_stored_count', 0) returns 0 without KeyError."""
        from src.ingest.embedding.state import EmbeddingPipelineState
        state: EmbeddingPipelineState = {}  # type: ignore[typeddict-item]
        result = state.get("visual_stored_count", 0)  # type: ignore[call-overload]
        assert result == 0

    def test_page_images_accepts_none(self):
        """FR-ES01d: page_images=None does not raise; field holds None."""
        from src.ingest.embedding.state import EmbeddingPipelineState
        state: EmbeddingPipelineState = {"page_images": None}  # type: ignore[typeddict-item]
        assert state["page_images"] is None

    def test_page_images_accepts_list(self):
        """FR-ES01e: page_images accepts a list of arbitrary objects."""
        from src.ingest.embedding.state import EmbeddingPipelineState

        class _MockImage:
            pass

        images = [_MockImage(), _MockImage()]
        state: EmbeddingPipelineState = {"page_images": images}  # type: ignore[typeddict-item]
        assert state["page_images"] == images
        assert len(state["page_images"]) == 2  # type: ignore[arg-type]

    def test_both_new_fields_together(self):
        """FR-ES01f: Both visual_stored_count and page_images coexist without conflict."""
        from src.ingest.embedding.state import EmbeddingPipelineState
        state: EmbeddingPipelineState = {  # type: ignore[typeddict-item]
            "visual_stored_count": 3,
            "page_images": ["img_a", "img_b", "img_c"],
        }
        assert state["visual_stored_count"] == 3
        assert len(state["page_images"]) == 3  # type: ignore[arg-type]


# ===========================================================================
# TestBuildEmbeddingGraph
# ===========================================================================

class TestBuildEmbeddingGraph:
    """FR-BG01: build_embedding_graph() includes visual_embedding node.

    NOTE: LangGraph introspection API stability is unknown. Tests that rely on
    internal graph-node inspection are wrapped to tolerate API changes.
    The compiled graph's node count (15) and full edge topology are not tested
    here due to LangGraph API instability concerns.
    """

    def test_build_embedding_graph_returns_something(self):
        """FR-BG01a: build_embedding_graph() returns a non-None compiled graph."""
        from src.ingest.common.types import IngestionConfig
        from src.ingest.embedding.workflow import build_embedding_graph
        cfg = IngestionConfig(enable_visual_embedding=True)
        graph = build_embedding_graph(cfg)
        assert graph is not None

    def test_visual_embedding_node_in_compiled_graph(self):
        """FR-BG01b: compiled graph contains 'visual_embedding' as a node.

        NOTE: Uses LangGraph's get_graph().nodes introspection API.
        If the API changes this test will be skipped rather than fail.
        """
        from src.ingest.common.types import IngestionConfig
        from src.ingest.embedding.workflow import build_embedding_graph
        cfg = IngestionConfig(enable_visual_embedding=True)
        graph = build_embedding_graph(cfg)
        try:
            nodes = graph.get_graph().nodes
            node_ids = list(nodes.keys()) if hasattr(nodes, "keys") else list(nodes)
            assert "visual_embedding" in node_ids
        except (AttributeError, TypeError) as exc:
            pytest.skip(
                f"LangGraph introspection API unavailable or changed: {exc}"
            )

    def test_build_embedding_graph_disabled_visual_embedding(self):
        """FR-BG01c: build_embedding_graph() succeeds even when visual embedding is disabled.

        NOTE: Whether 'visual_embedding' node is present when disabled is unspecified.
        This test only asserts the call does not raise.
        """
        from src.ingest.common.types import IngestionConfig
        from src.ingest.embedding.workflow import build_embedding_graph
        cfg = IngestionConfig(enable_visual_embedding=False)
        graph = build_embedding_graph(cfg)
        assert graph is not None
