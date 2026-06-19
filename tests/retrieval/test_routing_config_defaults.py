# @summary
# Config-default + fail-fast-validation tests for the RAPTOR-lite document
# routing knobs added in slice S0b (config/settings.py).
# Verifies every new RAG_DOCUMENT_ROUTING_* / RAG_DECOMPOSITION_* /
# RAG_INGESTION_CARD_* / RAG_DOCUMENT_CARD_COLLECTION attribute imports with the
# stated default and type, that routing + decomposition are OFF by default, and
# that validate_document_routing_config() fails fast on contradictory values.
# Exports: TestRoutingConfigDefaults, TestRoutingConfigValidation
# Deps: pytest, config.settings
# @end-summary
"""Settings-default and validation tests for document-routing config (S0b).

These tests assert declared defaults only — no behaviour is exercised. They
mirror the existing settings-test idiom in ``tests/test_project_config.py``
(monkeypatch module attrs, call the validate helper directly) and the reload
idiom in ``tests/ingest/test_visual_embedding_config_state.py``.
"""

from __future__ import annotations

import pytest

import config.settings as _settings


# ---------------------------------------------------------------------------
# Defaults + types (with default environment)
# ---------------------------------------------------------------------------


class TestRoutingConfigDefaults:
    """Every new S0b settings attribute imports with the stated default/type."""

    def test_routing_disabled_by_default(self):
        assert _settings.RAG_DOCUMENT_ROUTING_ENABLED is False

    def test_routing_top_n_default(self):
        assert _settings.RAG_DOCUMENT_ROUTING_TOP_N == 6
        assert isinstance(_settings.RAG_DOCUMENT_ROUTING_TOP_N, int)

    def test_routing_min_score_default(self):
        assert _settings.RAG_DOCUMENT_ROUTING_MIN_SCORE == 0.0
        assert isinstance(_settings.RAG_DOCUMENT_ROUTING_MIN_SCORE, float)

    def test_routing_per_doc_leaves_default(self):
        assert _settings.RAG_DOCUMENT_ROUTING_PER_DOC_LEAVES == 5
        assert isinstance(_settings.RAG_DOCUMENT_ROUTING_PER_DOC_LEAVES, int)

    def test_routing_max_candidates_default(self):
        assert _settings.RAG_DOCUMENT_ROUTING_MAX_CANDIDATES == 60
        assert isinstance(_settings.RAG_DOCUMENT_ROUTING_MAX_CANDIDATES, int)

    def test_routing_boost_default(self):
        assert _settings.RAG_DOCUMENT_ROUTING_BOOST == 0.0
        assert isinstance(_settings.RAG_DOCUMENT_ROUTING_BOOST, float)

    def test_card_collection_default(self):
        assert _settings.RAG_DOCUMENT_CARD_COLLECTION == "RAGDocumentCards"
        assert isinstance(_settings.RAG_DOCUMENT_CARD_COLLECTION, str)

    def test_decomposition_disabled_by_default(self):
        assert _settings.RAG_DECOMPOSITION_ENABLED is False

    def test_decomposition_llm_primary_default(self):
        assert _settings.RAG_DECOMPOSITION_LLM_PRIMARY is True

    def test_decomposition_llm_timeout_default(self):
        assert _settings.RAG_DECOMPOSITION_LLM_TIMEOUT_SECONDS == 8
        assert isinstance(_settings.RAG_DECOMPOSITION_LLM_TIMEOUT_SECONDS, int)

    def test_decomposition_min_subqueries_default(self):
        assert _settings.RAG_DECOMPOSITION_MIN_SUBQUERIES == 2
        assert isinstance(_settings.RAG_DECOMPOSITION_MIN_SUBQUERIES, int)

    def test_decomposition_max_subqueries_default(self):
        assert _settings.RAG_DECOMPOSITION_MAX_SUBQUERIES == 5
        assert isinstance(_settings.RAG_DECOMPOSITION_MAX_SUBQUERIES, int)

    def test_ingestion_build_document_cards_default(self):
        assert _settings.RAG_INGESTION_BUILD_DOCUMENT_CARDS is False

    def test_ingestion_card_llm_summary_default(self):
        assert _settings.RAG_INGESTION_CARD_LLM_SUMMARY is False

    def test_ingestion_card_max_headings_default(self):
        assert _settings.RAG_INGESTION_CARD_MAX_HEADINGS == 60
        assert isinstance(_settings.RAG_INGESTION_CARD_MAX_HEADINGS, int)

    def test_default_config_does_not_raise(self):
        """With default env, the validate helper accepts the declared defaults."""
        _settings.validate_document_routing_config()  # should not raise


# ---------------------------------------------------------------------------
# Fail-fast validation (monkeypatch module attrs, call helper directly —
# mirrors tests/test_project_config.py TestValidate* pattern)
# ---------------------------------------------------------------------------


def _call(monkeypatch, overrides: dict) -> None:
    for attr, val in overrides.items():
        monkeypatch.setattr(_settings, attr, val)
    _settings.validate_document_routing_config()


class TestRoutingConfigValidation:
    """validate_document_routing_config() enforces the design's safety rules."""

    def test_defaults_valid(self, monkeypatch):
        """The declared defaults are a valid configuration."""
        _call(monkeypatch, {})  # no overrides — defaults

    def test_top_n_two_is_valid(self, monkeypatch):
        """TOP_N == 2 is the minimum allowed (never top-1)."""
        _call(monkeypatch, {"RAG_DOCUMENT_ROUTING_TOP_N": 2})

    @pytest.mark.parametrize("bad_top_n", [1, 0, -3])
    def test_top_n_below_two_raises(self, monkeypatch, bad_top_n):
        """TOP_N < 2 violates 'never top-1' and must raise ValueError."""
        with pytest.raises(ValueError, match="RAG_DOCUMENT_ROUTING_TOP_N"):
            _call(monkeypatch, {"RAG_DOCUMENT_ROUTING_TOP_N": bad_top_n})

    def test_min_subqueries_greater_than_max_raises(self, monkeypatch):
        """MIN_SUBQUERIES > MAX_SUBQUERIES must raise ValueError."""
        with pytest.raises(ValueError, match="SUBQUERIES"):
            _call(monkeypatch, {
                "RAG_DECOMPOSITION_MIN_SUBQUERIES": 5,
                "RAG_DECOMPOSITION_MAX_SUBQUERIES": 2,
            })

    def test_min_equals_max_subqueries_valid(self, monkeypatch):
        """MIN == MAX subqueries is a valid edge case."""
        _call(monkeypatch, {
            "RAG_DECOMPOSITION_MIN_SUBQUERIES": 3,
            "RAG_DECOMPOSITION_MAX_SUBQUERIES": 3,
        })

    @pytest.mark.parametrize("bad_min", [1, 0])
    def test_min_subqueries_below_two_raises(self, monkeypatch, bad_min):
        """MIN_SUBQUERIES < 2 must raise ValueError."""
        with pytest.raises(ValueError, match="RAG_DECOMPOSITION_MIN_SUBQUERIES"):
            _call(monkeypatch, {
                "RAG_DECOMPOSITION_MIN_SUBQUERIES": bad_min,
                "RAG_DECOMPOSITION_MAX_SUBQUERIES": 5,
            })

    def test_max_candidates_below_per_doc_leaves_raises(self, monkeypatch):
        """MAX_CANDIDATES < PER_DOC_LEAVES is contradictory and must raise."""
        with pytest.raises(ValueError, match="RAG_DOCUMENT_ROUTING_MAX_CANDIDATES"):
            _call(monkeypatch, {
                "RAG_DOCUMENT_ROUTING_MAX_CANDIDATES": 3,
                "RAG_DOCUMENT_ROUTING_PER_DOC_LEAVES": 5,
            })

    def test_max_candidates_equals_per_doc_leaves_valid(self, monkeypatch):
        """MAX_CANDIDATES == PER_DOC_LEAVES is the valid boundary."""
        _call(monkeypatch, {
            "RAG_DOCUMENT_ROUTING_MAX_CANDIDATES": 5,
            "RAG_DOCUMENT_ROUTING_PER_DOC_LEAVES": 5,
        })
