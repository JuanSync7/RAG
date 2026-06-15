# @summary
# Tests for the IngestionConfig document-card fields added in slice S0b
# (src/ingest/common/types.py). Verifies the four new fields exist with the
# stated defaults sourced from config.settings, and that they remain overridable
# via the dataclass constructor.
# Exports: TestIngestionConfigCardFields
# Deps: pytest, src.ingest.common.types, config.settings
# @end-summary
"""IngestionConfig document-card field-default tests (S0b).

Mirrors the IngestionConfig field-default idiom in
``tests/ingest/test_visual_embedding_config_state.py`` (construct
``IngestionConfig()`` and assert defaults match the settings constants).
"""

from __future__ import annotations

import config.settings as _settings
from src.ingest.common.types import IngestionConfig


class TestIngestionConfigCardFields:
    """IngestionConfig exposes the four S0b card fields with correct defaults."""

    def test_build_document_cards_default_false(self):
        cfg = IngestionConfig()
        assert cfg.build_document_cards is False
        assert cfg.build_document_cards == _settings.RAG_INGESTION_BUILD_DOCUMENT_CARDS

    def test_card_llm_summary_default_false(self):
        cfg = IngestionConfig()
        assert cfg.card_llm_summary is False
        assert cfg.card_llm_summary == _settings.RAG_INGESTION_CARD_LLM_SUMMARY

    def test_card_max_headings_default(self):
        cfg = IngestionConfig()
        assert cfg.card_max_headings == 60
        assert cfg.card_max_headings == _settings.RAG_INGESTION_CARD_MAX_HEADINGS
        assert isinstance(cfg.card_max_headings, int)

    def test_card_collection_default(self):
        cfg = IngestionConfig()
        assert cfg.card_collection == "RAGDocumentCards"
        assert cfg.card_collection == _settings.RAG_DOCUMENT_CARD_COLLECTION
        assert isinstance(cfg.card_collection, str)

    def test_fields_are_overridable(self):
        cfg = IngestionConfig(
            build_document_cards=True,
            card_llm_summary=True,
            card_max_headings=10,
            card_collection="CustomCards",
        )
        assert cfg.build_document_cards is True
        assert cfg.card_llm_summary is True
        assert cfg.card_max_headings == 10
        assert cfg.card_collection == "CustomCards"
