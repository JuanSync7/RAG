# @summary
# Contract tests ensuring API Pydantic schemas stay aligned with internal pipeline contracts.
# Covers: ConsoleIngestionRequest ↔ IngestionConfig, QueryRequest ↔ RAGRequest, QueryResponse ↔ RAGResponse
# Deps: pytest, dataclasses, server.schemas, src.ingest.common.types, src.retrieval.common.schemas
# @end-summary
"""Contract tests for API ↔ pipeline schema alignment.

These tests catch drift when internal pipeline contracts (IngestionConfig,
RAGRequest, RAGResponse) add, remove, or rename fields that the API layer
(ConsoleIngestionRequest, QueryRequest, QueryResponse) must track.

Design:
  - Every IngestionConfig field is classified as either EXPOSED (mapped via
    ConsoleIngestionRequest._FIELD_MAP) or INFRA_ONLY (explicitly listed here).
    If a new field is added to IngestionConfig without updating either set,
    the test fails — forcing a conscious decision about API exposure.
  - QueryRequest/QueryResponse fields are validated against RAGRequest/RAGResponse
    to ensure the API layer doesn't reference fields that no longer exist
    internally, and that new internal fields are consciously excluded or added.
"""

from __future__ import annotations

import dataclasses

import pytest


# ═══════════════════════════════════════════════════════════════════════
# Ingestion: ConsoleIngestionRequest ↔ IngestionConfig
# ═══════════════════════════════════════════════════════════════════════

# Fields in IngestionConfig that are intentionally NOT exposed via the API.
# When a new field is added to IngestionConfig, add it here or to
# ConsoleIngestionRequest._FIELD_MAP — the test will fail otherwise.
INGESTION_INFRA_ONLY_FIELDS = frozenset({
    # LLM behavioral (server-side tuning, not user-facing)
    "llm_temperature",
    "llm_timeout_seconds",
    "max_keywords",
    "enable_llm_metadata",
    "llm_model",
    # Chunking internals
    "chunk_size",
    "chunk_overlap",
    # Multimodal processing toggle
    "enable_multimodal_processing",
    # Vision internals (API keys, paths, byte limits)
    "vision_max_image_bytes",
    "vision_temperature",
    "vision_max_tokens",
    "vision_api_key",
    "vision_api_path",
    # Pipeline feature toggles (server-side defaults)
    "enable_document_refactoring",
    "enable_cross_reference_extraction",
    "enable_knowledge_graph_extraction",
    "enable_quality_validation",
    "enable_knowledge_graph_storage",
    # Quality thresholds
    "min_chunk_chars",
    "min_quality_score",
    # Storage / infra paths
    "clean_store_dir",
    "store_documents",
    "target_bucket",
    "mirror_output_dir",
    # Runtime / debug
    "ollama_url",
    "persist_docling_document",
    # Visual embedding (infrastructure/model config, not user-facing API fields)
    "enable_visual_embedding",
    "colqwen_model_name",
    "colqwen_batch_size",
    "page_image_max_dimension",
    "page_image_quality",
    "visual_target_collection",
    # Ingestion hardening (server-side infra controls, not user-facing API fields)
    "chunker",
    "clean_store_bucket",
    "dedup_override_sources",
    "embedding_batch_size",
    "enable_cross_document_dedup",
    "enable_fuzzy_dedup",
    "fuzzy_num_hashes",
    "fuzzy_shingle_size",
    "fuzzy_similarity_threshold",
    "gc_mode",
    "gc_retention_days",
    "gc_schedule",
    "parser_strategy",
})

# Fields on ConsoleIngestionRequest that don't map to IngestionConfig
# (they control execution mode, not pipeline config).
INGESTION_REQUEST_ONLY_FIELDS = frozenset({
    "mode",
    "target_path",
    "export_obsidian",
})


class TestIngestionSchemaContract:
    """Every IngestionConfig field must be either EXPOSED or INFRA_ONLY."""

    def _config_fields(self) -> set[str]:
        from src.ingest.common.types import IngestionConfig
        return {f.name for f in dataclasses.fields(IngestionConfig)}

    def _field_map(self) -> dict[str, str]:
        from server.schemas import INGESTION_REQUEST_FIELD_MAP
        return dict(INGESTION_REQUEST_FIELD_MAP)

    def test_all_config_fields_are_classified(self):
        """Every IngestionConfig field must appear in _FIELD_MAP values or INFRA_ONLY."""
        config_fields = self._config_fields()
        mapped_config_fields = set(self._field_map().values())
        classified = mapped_config_fields | INGESTION_INFRA_ONLY_FIELDS
        unclassified = config_fields - classified
        assert not unclassified, (
            f"IngestionConfig field(s) {sorted(unclassified)} are not classified. "
            "Add them to ConsoleIngestionRequest._FIELD_MAP (to expose via API) "
            "or to INGESTION_INFRA_ONLY_FIELDS in this test (to mark as infra-only)."
        )

    def test_no_phantom_exposed_fields(self):
        """Every _FIELD_MAP value must reference a real IngestionConfig field."""
        config_fields = self._config_fields()
        mapped_config_fields = set(self._field_map().values())
        phantoms = mapped_config_fields - config_fields
        assert not phantoms, (
            f"ConsoleIngestionRequest._FIELD_MAP references IngestionConfig field(s) "
            f"{sorted(phantoms)} that no longer exist. Update the mapping."
        )

    def test_no_phantom_infra_fields(self):
        """Every INFRA_ONLY entry must reference a real IngestionConfig field."""
        config_fields = self._config_fields()
        phantoms = INGESTION_INFRA_ONLY_FIELDS - config_fields
        assert not phantoms, (
            f"INGESTION_INFRA_ONLY_FIELDS references IngestionConfig field(s) "
            f"{sorted(phantoms)} that no longer exist. Remove stale entries."
        )

    def test_no_overlap_between_exposed_and_infra(self):
        """A field cannot be both exposed and infra-only."""
        mapped_config_fields = set(self._field_map().values())
        overlap = mapped_config_fields & INGESTION_INFRA_ONLY_FIELDS
        assert not overlap, (
            f"Field(s) {sorted(overlap)} appear in both _FIELD_MAP and "
            "INGESTION_INFRA_ONLY_FIELDS. Classify each field exactly once."
        )

    def test_to_config_produces_valid_instance(self):
        """to_config() with defaults produces a valid IngestionConfig."""
        from server.schemas import ConsoleIngestionRequest
        from src.ingest.common.types import IngestionConfig

        req = ConsoleIngestionRequest()
        config = req.to_config()
        assert isinstance(config, IngestionConfig)

    def test_to_config_overlays_non_none_fields(self):
        """Non-None request fields override IngestionConfig defaults."""
        from server.schemas import ConsoleIngestionRequest

        req = ConsoleIngestionRequest(
            vlm_mode="external",
            hybrid_chunker_max_tokens=256,
            docling_enabled=True,
            vision_max_figures=8,
            target_collection="test_collection",
        )
        config = req.to_config()
        assert config.vlm_mode == "external"
        assert config.hybrid_chunker_max_tokens == 256
        assert config.enable_docling_parser is True
        assert config.vision_max_figures == 8
        assert config.target_collection == "test_collection"

    def test_to_config_preserves_defaults_for_none_fields(self):
        """None request fields don't override IngestionConfig defaults."""
        from server.schemas import ConsoleIngestionRequest
        from src.ingest.common.types import IngestionConfig

        req = ConsoleIngestionRequest()
        config = req.to_config()
        default_config = IngestionConfig()
        assert config.vlm_mode == default_config.vlm_mode
        assert config.hybrid_chunker_max_tokens == default_config.hybrid_chunker_max_tokens

    def test_request_only_fields_not_in_field_map(self):
        """Fields like mode/target_path/export_obsidian should not be in _FIELD_MAP."""
        mapped_req_fields = set(self._field_map().keys())
        leaked = INGESTION_REQUEST_ONLY_FIELDS & mapped_req_fields
        assert not leaked, (
            f"Request-only field(s) {sorted(leaked)} should not be in _FIELD_MAP "
            "because they don't correspond to IngestionConfig fields."
        )


# ═══════════════════════════════════════════════════════════════════════
# Retrieval: QueryRequest ↔ RAGRequest, QueryResponse ↔ RAGResponse
# ═══════════════════════════════════════════════════════════════════════

# QueryRequest fields that don't map 1:1 to RAGRequest
# (filled by route handler, not by client).
QUERY_REQUEST_ONLY_FIELDS = frozenset({
    "memory_enabled",       # controls whether route handler injects memory
    "memory_turn_window",   # controls how many turns route handler fetches
    "compact_now",          # triggers compaction after response
})

# RAGRequest fields that are NOT exposed via QueryRequest
# (filled by route handler or server-side logic).
RAG_REQUEST_INTERNAL_FIELDS = frozenset({
    "skip_generation",      # server controls this for stream vs non-stream
    "memory_context",       # built by route handler from conversation store
    "memory_recent_turns",  # built by route handler from conversation store
})

# RAGResponse fields that are NOT exposed via QueryResponse
# (internal pipeline signals not surfaced to API clients).
RAG_RESPONSE_INTERNAL_FIELDS = frozenset({
    "guardrails",
    "composite_confidence",
    "confidence_breakdown",
    "post_guardrail_action",
    "version_conflicts",
    "retry_count",
    "verification_warning",
    "retrieval_quality",
    "retrieval_quality_note",
    "re_retrieval_suggested",
    "re_retrieval_params",
})

# QueryResponse fields that don't come from RAGResponse
# (added by the route handler).
QUERY_RESPONSE_ONLY_FIELDS = frozenset({
    "workflow_id",  # Temporal workflow ID
    "latency_ms",   # measured by route handler
})


class TestRetrievalSchemaContract:
    """QueryRequest/QueryResponse must stay aligned with RAGRequest/RAGResponse."""

    def _rag_request_fields(self) -> set[str]:
        from src.retrieval.common.schemas import RAGRequest
        return {f.name for f in dataclasses.fields(RAGRequest)}

    def _rag_response_fields(self) -> set[str]:
        from src.retrieval.common.schemas import RAGResponse
        return {f.name for f in dataclasses.fields(RAGResponse)}

    def _query_request_fields(self) -> set[str]:
        from server.schemas import QueryRequest
        return set(QueryRequest.model_fields.keys())

    def _query_response_fields(self) -> set[str]:
        from server.schemas import QueryResponse
        return set(QueryResponse.model_fields.keys())

    # ── RAGRequest ↔ QueryRequest ───────────────────────────────────

    def test_query_request_fields_exist_in_rag_request(self):
        """Every QueryRequest field must map to a RAGRequest field or be request-only."""
        qr_fields = self._query_request_fields()
        rag_fields = self._rag_request_fields()
        api_only = qr_fields - rag_fields
        unclassified = api_only - QUERY_REQUEST_ONLY_FIELDS
        assert not unclassified, (
            f"QueryRequest field(s) {sorted(unclassified)} don't exist in RAGRequest "
            "and aren't in QUERY_REQUEST_ONLY_FIELDS. Either add them to RAGRequest "
            "or classify them as request-only."
        )

    def test_all_rag_request_fields_are_classified(self):
        """Every RAGRequest field must be exposed via QueryRequest or marked internal."""
        rag_fields = self._rag_request_fields()
        qr_fields = self._query_request_fields()
        classified = qr_fields | RAG_REQUEST_INTERNAL_FIELDS
        unclassified = rag_fields - classified
        assert not unclassified, (
            f"RAGRequest field(s) {sorted(unclassified)} are not classified. "
            "Add them to QueryRequest (to expose via API) or to "
            "RAG_REQUEST_INTERNAL_FIELDS in this test."
        )

    def test_no_phantom_rag_request_internal_fields(self):
        """Every RAG_REQUEST_INTERNAL_FIELDS entry must reference a real RAGRequest field."""
        rag_fields = self._rag_request_fields()
        phantoms = RAG_REQUEST_INTERNAL_FIELDS - rag_fields
        assert not phantoms, (
            f"RAG_REQUEST_INTERNAL_FIELDS references RAGRequest field(s) "
            f"{sorted(phantoms)} that no longer exist."
        )

    # ── RAGResponse ↔ QueryResponse ─────────────────────────────────

    def test_query_response_fields_exist_in_rag_response(self):
        """Every QueryResponse field must map to a RAGResponse field or be response-only."""
        qresp_fields = self._query_response_fields()
        rag_fields = self._rag_response_fields()
        api_only = qresp_fields - rag_fields
        unclassified = api_only - QUERY_RESPONSE_ONLY_FIELDS
        assert not unclassified, (
            f"QueryResponse field(s) {sorted(unclassified)} don't exist in RAGResponse "
            "and aren't in QUERY_RESPONSE_ONLY_FIELDS. Either add them to RAGResponse "
            "or classify them as response-only."
        )

    def test_all_rag_response_fields_are_classified(self):
        """Every RAGResponse field must be exposed via QueryResponse or marked internal."""
        rag_fields = self._rag_response_fields()
        qresp_fields = self._query_response_fields()
        classified = qresp_fields | RAG_RESPONSE_INTERNAL_FIELDS
        unclassified = rag_fields - classified
        assert not unclassified, (
            f"RAGResponse field(s) {sorted(unclassified)} are not classified. "
            "Add them to QueryResponse (to expose via API) or to "
            "RAG_RESPONSE_INTERNAL_FIELDS in this test."
        )

    def test_no_phantom_rag_response_internal_fields(self):
        """Every RAG_RESPONSE_INTERNAL_FIELDS entry must reference a real RAGResponse field."""
        rag_fields = self._rag_response_fields()
        phantoms = RAG_RESPONSE_INTERNAL_FIELDS - rag_fields
        assert not phantoms, (
            f"RAG_RESPONSE_INTERNAL_FIELDS references RAGResponse field(s) "
            f"{sorted(phantoms)} that no longer exist."
        )

    def test_no_phantom_query_response_only_fields(self):
        """Every QUERY_RESPONSE_ONLY_FIELDS entry must be a real QueryResponse field."""
        qresp_fields = self._query_response_fields()
        phantoms = QUERY_RESPONSE_ONLY_FIELDS - qresp_fields
        assert not phantoms, (
            f"QUERY_RESPONSE_ONLY_FIELDS references QueryResponse field(s) "
            f"{sorted(phantoms)} that no longer exist."
        )

    # ── Wire type alignment ─────────────────────────────────────────

    def test_chunk_result_matches_ranked_result(self):
        """ChunkResult (API) must have the same fields as RankedResult (internal)."""
        from server.schemas import ChunkResult
        from src.retrieval.common.schemas import RankedResult

        api_fields = set(ChunkResult.model_fields.keys())
        internal_fields = {f.name for f in dataclasses.fields(RankedResult)}
        assert api_fields == internal_fields, (
            f"ChunkResult and RankedResult field mismatch. "
            f"API-only: {api_fields - internal_fields}, "
            f"Internal-only: {internal_fields - api_fields}"
        )
