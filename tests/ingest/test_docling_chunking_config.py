# @summary
# Tests for Docling-native chunking pipeline config fields.
# Covers: config/settings.py env var reads for the three new Docling chunking vars,
#         and src/ingest/common/types.py IngestionConfig new fields + PIPELINE_NODE_NAMES.
# Exports: (pytest test functions)
# Deps: pytest, importlib, config.settings, src.ingest.common.types
# @end-summary
"""Tests for Docling-native chunking pipeline configuration.

Covers two modules:
- ``config/settings.py`` — env var reads for RAG_INGESTION_VLM_MODE,
  RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS, RAG_INGESTION_PERSIST_DOCLING_DOCUMENT
- ``src/ingest/common/types.py`` — IngestionConfig new fields (vlm_mode,
  hybrid_chunker_max_tokens, persist_docling_document) and PIPELINE_NODE_NAMES ordering

Note on env var testing: Python caches module-level constants at import time.
Tests that verify env var effects must use ``monkeypatch.setenv`` combined with
``importlib.reload(config.settings)`` to force the module to re-read the environment.
"""

import importlib

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reload_settings():
    """Force config.settings to re-evaluate all os.environ.get() calls."""
    import config.settings as s
    importlib.reload(s)
    return s


# ===========================================================================
# Module 1: config/settings.py — env var definitions
# ===========================================================================


class TestSettingsDefaults:
    """RAG_INGESTION_VLM_MODE, _HYBRID_CHUNKER_MAX_TOKENS, _PERSIST_DOCLING_DOCUMENT
    must expose their documented defaults when no env vars are set."""

    def test_vlm_mode_default_is_disabled(self, monkeypatch):
        monkeypatch.delenv("RAG_INGESTION_VLM_MODE", raising=False)
        s = _reload_settings()
        assert s.RAG_INGESTION_VLM_MODE == "disabled"

    def test_hybrid_chunker_max_tokens_default_is_512(self, monkeypatch):
        monkeypatch.delenv("RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS", raising=False)
        s = _reload_settings()
        assert s.RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS == 512

    def test_hybrid_chunker_max_tokens_default_is_int(self, monkeypatch):
        monkeypatch.delenv("RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS", raising=False)
        s = _reload_settings()
        assert isinstance(s.RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS, int)

    def test_persist_docling_document_default_is_true(self, monkeypatch):
        monkeypatch.delenv("RAG_INGESTION_PERSIST_DOCLING_DOCUMENT", raising=False)
        s = _reload_settings()
        assert s.RAG_INGESTION_PERSIST_DOCLING_DOCUMENT is True

    def test_persist_docling_document_default_is_bool(self, monkeypatch):
        monkeypatch.delenv("RAG_INGESTION_PERSIST_DOCLING_DOCUMENT", raising=False)
        s = _reload_settings()
        assert isinstance(s.RAG_INGESTION_PERSIST_DOCLING_DOCUMENT, bool)

    def test_all_three_absent_uses_all_defaults(self, monkeypatch):
        """All three vars absent at once → all three return their documented defaults."""
        monkeypatch.delenv("RAG_INGESTION_VLM_MODE", raising=False)
        monkeypatch.delenv("RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS", raising=False)
        monkeypatch.delenv("RAG_INGESTION_PERSIST_DOCLING_DOCUMENT", raising=False)
        s = _reload_settings()
        assert s.RAG_INGESTION_VLM_MODE == "disabled"
        assert s.RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS == 512
        assert s.RAG_INGESTION_PERSIST_DOCLING_DOCUMENT is True


class TestSettingsVlmModeOverride:
    """RAG_INGESTION_VLM_MODE accepts any string value and passes it through."""

    def test_vlm_mode_external(self, monkeypatch):
        monkeypatch.setenv("RAG_INGESTION_VLM_MODE", "external")
        s = _reload_settings()
        assert s.RAG_INGESTION_VLM_MODE == "external"

    def test_vlm_mode_builtin(self, monkeypatch):
        monkeypatch.setenv("RAG_INGESTION_VLM_MODE", "builtin")
        s = _reload_settings()
        assert s.RAG_INGESTION_VLM_MODE == "builtin"

    def test_vlm_mode_disabled_explicit(self, monkeypatch):
        monkeypatch.setenv("RAG_INGESTION_VLM_MODE", "disabled")
        s = _reload_settings()
        assert s.RAG_INGESTION_VLM_MODE == "disabled"

    def test_vlm_mode_empty_string_passes_through(self, monkeypatch):
        """Empty string is not validated by settings.py — passes through unchanged."""
        monkeypatch.setenv("RAG_INGESTION_VLM_MODE", "")
        s = _reload_settings()
        assert s.RAG_INGESTION_VLM_MODE == ""

    def test_vlm_mode_result_is_str(self, monkeypatch):
        monkeypatch.setenv("RAG_INGESTION_VLM_MODE", "external")
        s = _reload_settings()
        assert isinstance(s.RAG_INGESTION_VLM_MODE, str)


class TestSettingsHybridChunkerMaxTokens:
    """RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS is coerced to int."""

    def test_max_tokens_override_256(self, monkeypatch):
        monkeypatch.setenv("RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS", "256")
        s = _reload_settings()
        assert s.RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS == 256

    def test_max_tokens_override_is_int_not_str(self, monkeypatch):
        monkeypatch.setenv("RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS", "256")
        s = _reload_settings()
        assert isinstance(s.RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS, int)

    def test_max_tokens_at_bge_m3_limit_512(self, monkeypatch):
        """512 is the bge-m3 token limit; must be stored as int 512."""
        monkeypatch.setenv("RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS", "512")
        s = _reload_settings()
        assert s.RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS == 512

    def test_max_tokens_zero_stored_as_zero(self, monkeypatch):
        """Zero passes through as int 0 — validation is impl.py's responsibility."""
        monkeypatch.setenv("RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS", "0")
        s = _reload_settings()
        assert s.RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS == 0

    def test_max_tokens_non_integer_raises_value_error(self, monkeypatch):
        """A non-numeric string causes ValueError at module import time (int() cast)."""
        monkeypatch.setenv("RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS", "not_a_number")
        with pytest.raises(ValueError):
            _reload_settings()


class TestSettingsPersistDoclingDocument:
    """RAG_INGESTION_PERSIST_DOCLING_DOCUMENT uses .lower() in ("true", "1", "yes")."""

    def test_persist_false_via_false_string(self, monkeypatch):
        monkeypatch.setenv("RAG_INGESTION_PERSIST_DOCLING_DOCUMENT", "false")
        s = _reload_settings()
        assert s.RAG_INGESTION_PERSIST_DOCLING_DOCUMENT is False

    def test_persist_false_via_zero_string(self, monkeypatch):
        monkeypatch.setenv("RAG_INGESTION_PERSIST_DOCLING_DOCUMENT", "0")
        s = _reload_settings()
        assert s.RAG_INGESTION_PERSIST_DOCLING_DOCUMENT is False

    def test_persist_true_via_yes_string(self, monkeypatch):
        monkeypatch.setenv("RAG_INGESTION_PERSIST_DOCLING_DOCUMENT", "yes")
        s = _reload_settings()
        assert s.RAG_INGESTION_PERSIST_DOCLING_DOCUMENT is True

    def test_persist_true_via_one_string(self, monkeypatch):
        monkeypatch.setenv("RAG_INGESTION_PERSIST_DOCLING_DOCUMENT", "1")
        s = _reload_settings()
        assert s.RAG_INGESTION_PERSIST_DOCLING_DOCUMENT is True

    def test_persist_true_via_true_string(self, monkeypatch):
        monkeypatch.setenv("RAG_INGESTION_PERSIST_DOCLING_DOCUMENT", "true")
        s = _reload_settings()
        assert s.RAG_INGESTION_PERSIST_DOCLING_DOCUMENT is True

    def test_persist_true_uppercase_TRUE(self, monkeypatch):
        """.lower() normalisation means "TRUE" → True."""
        monkeypatch.setenv("RAG_INGESTION_PERSIST_DOCLING_DOCUMENT", "TRUE")
        s = _reload_settings()
        assert s.RAG_INGESTION_PERSIST_DOCLING_DOCUMENT is True

    def test_persist_true_uppercase_YES(self, monkeypatch):
        """.lower() normalisation means "YES" → True."""
        monkeypatch.setenv("RAG_INGESTION_PERSIST_DOCLING_DOCUMENT", "YES")
        s = _reload_settings()
        assert s.RAG_INGESTION_PERSIST_DOCLING_DOCUMENT is True

    def test_persist_result_is_bool(self, monkeypatch):
        monkeypatch.setenv("RAG_INGESTION_PERSIST_DOCLING_DOCUMENT", "true")
        s = _reload_settings()
        assert isinstance(s.RAG_INGESTION_PERSIST_DOCLING_DOCUMENT, bool)


# ===========================================================================
# Module 2: src/ingest/common/types.py — IngestionConfig new fields
# ===========================================================================

from src.ingest.common.types import IngestionConfig, PIPELINE_NODE_NAMES  # noqa: E402


class TestIngestionConfigNewFieldDefaults:
    """IngestionConfig() with no args exposes all three new fields at documented defaults."""

    def test_default_vlm_mode_is_disabled(self):
        cfg = IngestionConfig()
        assert cfg.vlm_mode == "disabled"

    def test_default_hybrid_chunker_max_tokens_is_512(self):
        cfg = IngestionConfig()
        assert cfg.hybrid_chunker_max_tokens == 512

    def test_default_persist_docling_document_is_true(self):
        cfg = IngestionConfig()
        assert cfg.persist_docling_document is True

    def test_default_vlm_mode_is_str(self):
        cfg = IngestionConfig()
        assert isinstance(cfg.vlm_mode, str)

    def test_default_hybrid_chunker_max_tokens_is_int(self):
        cfg = IngestionConfig()
        assert isinstance(cfg.hybrid_chunker_max_tokens, int)

    def test_default_persist_docling_document_is_bool(self):
        cfg = IngestionConfig()
        assert isinstance(cfg.persist_docling_document, bool)


class TestIngestionConfigNewFieldOverride:
    """IngestionConfig constructor accepts override values for all three new fields."""

    def test_vlm_mode_override_external(self):
        cfg = IngestionConfig(vlm_mode="external")
        assert cfg.vlm_mode == "external"

    def test_vlm_mode_override_builtin(self):
        cfg = IngestionConfig(vlm_mode="builtin")
        assert cfg.vlm_mode == "builtin"

    def test_hybrid_chunker_max_tokens_override_256(self):
        cfg = IngestionConfig(hybrid_chunker_max_tokens=256)
        assert cfg.hybrid_chunker_max_tokens == 256

    def test_persist_docling_document_override_false(self):
        cfg = IngestionConfig(persist_docling_document=False)
        assert cfg.persist_docling_document is False


class TestIngestionConfigNewFieldBoundary:
    """Boundary values for new IngestionConfig fields — no validation in the dataclass."""

    def test_vlm_mode_empty_string_stored(self):
        """Empty string is accepted without error (validation delegated to impl.py)."""
        cfg = IngestionConfig(vlm_mode="")
        assert cfg.vlm_mode == ""

    def test_hybrid_chunker_max_tokens_zero_stored(self):
        """Zero is stored as-is; validation is impl.py's responsibility."""
        cfg = IngestionConfig(hybrid_chunker_max_tokens=0)
        assert cfg.hybrid_chunker_max_tokens == 0

    def test_hybrid_chunker_max_tokens_above_bge_m3_limit_stored(self):
        """513 exceeds bge-m3 limit but is stored without error in the dataclass."""
        cfg = IngestionConfig(hybrid_chunker_max_tokens=513)
        assert cfg.hybrid_chunker_max_tokens == 513

    def test_no_post_init_validation_any_string_accepted(self):
        """IngestionConfig has no __post_init__ — any string vlm_mode is accepted."""
        cfg = IngestionConfig(vlm_mode="invalid_value")
        assert cfg.vlm_mode == "invalid_value"


class TestIngestionConfigBackwardCompatibility:
    """New fields must not disturb any pre-existing IngestionConfig fields."""

    def test_existing_field_enable_docling_parser_present(self):
        cfg = IngestionConfig()
        assert hasattr(cfg, "enable_docling_parser")

    def test_existing_field_docling_strict_present(self):
        cfg = IngestionConfig()
        assert hasattr(cfg, "docling_strict")

    def test_existing_field_chunk_size_present(self):
        cfg = IngestionConfig()
        assert hasattr(cfg, "chunk_size")

    def test_existing_field_chunk_overlap_present(self):
        cfg = IngestionConfig()
        assert hasattr(cfg, "chunk_overlap")

    def test_existing_field_semantic_chunking_present(self):
        cfg = IngestionConfig()
        assert hasattr(cfg, "semantic_chunking")

    def test_existing_field_enable_vision_processing_present(self):
        cfg = IngestionConfig()
        assert hasattr(cfg, "enable_vision_processing")

    def test_existing_fields_unaffected_by_new_field_override(self):
        """Overriding a new field must not change any pre-existing field default."""
        baseline = IngestionConfig()
        overridden = IngestionConfig(vlm_mode="external", hybrid_chunker_max_tokens=256)
        assert overridden.enable_docling_parser == baseline.enable_docling_parser
        assert overridden.chunk_size == baseline.chunk_size
        assert overridden.docling_strict == baseline.docling_strict


class TestPipelineNodeNamesOrdering:
    """PIPELINE_NODE_NAMES must include 'vlm_enrichment' between 'chunking' and
    'chunk_enrichment'."""

    def test_vlm_enrichment_present(self):
        assert "vlm_enrichment" in PIPELINE_NODE_NAMES

    def test_chunking_present(self):
        assert "chunking" in PIPELINE_NODE_NAMES

    def test_chunk_enrichment_present(self):
        assert "chunk_enrichment" in PIPELINE_NODE_NAMES

    def test_vlm_enrichment_immediately_after_chunking(self):
        chunking_idx = PIPELINE_NODE_NAMES.index("chunking")
        vlm_idx = PIPELINE_NODE_NAMES.index("vlm_enrichment")
        assert vlm_idx == chunking_idx + 1, (
            f"'vlm_enrichment' (index {vlm_idx}) must be immediately after "
            f"'chunking' (index {chunking_idx})"
        )

    def test_vlm_enrichment_immediately_before_chunk_enrichment(self):
        vlm_idx = PIPELINE_NODE_NAMES.index("vlm_enrichment")
        enrichment_idx = PIPELINE_NODE_NAMES.index("chunk_enrichment")
        assert enrichment_idx == vlm_idx + 1, (
            f"'chunk_enrichment' (index {enrichment_idx}) must be immediately after "
            f"'vlm_enrichment' (index {vlm_idx})"
        )

    def test_pipeline_starts_with_document_ingestion(self):
        assert PIPELINE_NODE_NAMES[0] == "document_ingestion"

    def test_pipeline_ends_with_knowledge_graph_storage(self):
        assert PIPELINE_NODE_NAMES[-1] == "knowledge_graph_storage"

    def test_pipeline_node_names_has_15_entries(self):
        """Bumped from 14 to 15 after 'visual_embedding' was inserted between
        'embedding_storage' and 'knowledge_graph_storage'."""
        assert len(PIPELINE_NODE_NAMES) == 15

    def test_pipeline_node_names_are_all_strings(self):
        assert all(isinstance(name, str) for name in PIPELINE_NODE_NAMES)

    def test_pipeline_node_names_no_duplicates(self):
        assert len(PIPELINE_NODE_NAMES) == len(set(PIPELINE_NODE_NAMES))
