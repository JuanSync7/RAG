# @summary
# Pydantic request/response models for the RAG API server.
# Exports: QueryRequest, QueryResponse, ChunkResult, HealthResponse,
#          DocumentSummary, DocumentListResponse, DocumentDetailResponse,
#          DocumentUrlResponse, SourceSummary, SourceListResponse,
#          CollectionItem, CollectionStatsResponse, CollectionListResponse,
#          VisualPageResultResponse
# Deps: pydantic
# @end-summary
"""Pydantic models for RAG API request/response serialization."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Optional

from server.common import (
    ApiErrorDetail,
    ApiErrorResponse,
    ConsoleEnvelope,
)


class QueryRequest(BaseModel):
    """Incoming query from a user."""
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=32000, description="The search query")
    source_filter: Optional[str] = Field(None, description="Filter by source document filename")
    heading_filter: Optional[str] = Field(None, description="Filter by section heading")
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Hybrid search balance (0=BM25, 1=vector)")
    search_limit: int = Field(10, ge=1, le=100, description="Max results from hybrid search")
    rerank_top_k: int = Field(5, ge=1, le=50, description="Top-K results after reranking")
    tenant_id: Optional[str] = Field(None, description="Optional tenant override for admins")
    max_query_iterations: Optional[int] = Field(
        None,
        ge=1,
        le=8,
        description="Cap for reformulation iterations",
    )
    fast_path: Optional[bool] = Field(
        None,
        description="Enable fast-path mode to bypass iterative reformulation",
    )
    overall_timeout_ms: Optional[int] = Field(
        None,
        ge=1000,
        le=180000,
        description="Overall retrieval timeout budget",
    )
    stage_budget_overrides: dict[str, int] = Field(
        default_factory=dict,
        description="Optional per-stage budget overrides in milliseconds",
    )
    conversation_id: Optional[str] = Field(
        None,
        min_length=3,
        max_length=128,
        description="Conversation identifier for multi-turn memory",
    )
    memory_enabled: bool = Field(
        default=True,
        description="Whether conversation memory should be applied",
    )
    memory_turn_window: Optional[int] = Field(
        default=None,
        ge=1,
        le=40,
        description="Optional override for number of recent turns injected as context",
    )
    compact_now: bool = Field(
        default=False,
        description="Force a summary compaction pass for the conversation after this turn",
    )
    mode: Literal["query", "retrieval"] = Field(
        default="query",
        description=(
            "Pipeline mode. ``query`` (default) generates an answer from "
            "retrieved chunks. ``retrieval`` returns ranked docs only and "
            "skips answer generation."
        ),
    )
    retrieval_sub_mode: Literal["auto", "hard"] = Field(
        default="auto",
        description=(
            "Only consulted when ``mode='retrieval'``. ``auto`` lets the "
            "rewriter decide whether to pull conversation history. "
            "``hard`` runs the literal user query with no rewriting and "
            "no history pull."
        ),
    )
    extra_processing: bool = Field(
        default=False,
        description=(
            "Reserved: enables an optional retrieval-only processing loop "
            "(e.g. query expansion / multi-query fan-out). The behaviour "
            "behind this flag is gated separately and may be a no-op in "
            "the current build."
        ),
    )
    tree_retrieval: Optional[bool] = Field(
        default=None,
        description=(
            "Per-request override for hierarchical (tree) retrieval. "
            "``None`` (default) uses the ``RAG_TREE_RETRIEVAL_ENABLED`` "
            "config setting. ``true``/``false`` force on/off for this "
            "request only. See TREE_RETRIEVAL_DESIGN.md §6."
        ),
    )
    deep_research: bool = Field(
        default=False,
        description=(
            "Opt-in deep-research retrieval. Replaces the linear "
            "kg_expand → embed → hybrid_search → rerank stages with a "
            "recursive topic-grouped loop: the LLM checks whether the "
            "evidence answers the original question, decomposes into "
            "sub-questions when not, and gathers chunks across multiple "
            "rounds. Uses substantially more LLM/retrieval calls; intended "
            "for analytical / multi-aspect questions."
        ),
    )

    @model_validator(mode="after")
    def _validate_stage_budget_overrides(self) -> "QueryRequest":
        allowed = {
            "query_processing",
            "kg_expansion",
            "embedding",
            "hybrid_search",
            "reranking",
            "generation",
            "visual_retrieval",
        }
        for stage, budget_ms in self.stage_budget_overrides.items():
            if stage not in allowed:
                raise ValueError(
                    f"Invalid stage budget key '{stage}'. Allowed: {sorted(allowed)}"
                )
            if int(budget_ms) < 100 or int(budget_ms) > 300000:
                raise ValueError(
                    f"Stage budget for '{stage}' must be between 100 and 300000 ms"
                )
        return self


class ChunkResult(BaseModel):
    """A single retrieved chunk with its reranking score."""
    text: str
    score: float
    metadata: dict


class VisualPageResultResponse(BaseModel):
    """A single visual page retrieval result for API serialization.

    Maps 1:1 from the ``VisualPageResult`` dataclass in the retrieval
    schemas module. Included in the OpenAPI schema.
    """

    document_id: str            # FR-701
    page_number: int            # FR-701
    source_key: str             # FR-701
    source_name: str            # FR-701
    score: float                # FR-701 -- cosine similarity 0.0-1.0
    page_image_url: str         # FR-701 -- presigned MinIO URL
    total_pages: int            # FR-701
    page_width_px: int          # FR-701
    page_height_px: int         # FR-701


class TokenBudgetResponse(BaseModel):
    """Token budget snapshot included in query responses."""

    input_tokens: int = 0
    context_length: int = 0
    output_reservation: int = 0
    usage_percent: float = 0.0
    model_name: str = ""
    breakdown: Optional[dict] = None
    actual_prompt_tokens: int = 0
    actual_completion_tokens: int = 0
    actual_total_tokens: int = 0
    cost_usd: float = 0.0


class QueryResponse(BaseModel):
    """Full response from the RAG pipeline."""
    query: str
    processed_query: str
    query_confidence: float
    action: str
    results: list[ChunkResult] = []
    clarification_message: Optional[str] = None
    kg_expanded_terms: Optional[list[str]] = None
    generated_answer: Optional[str] = None
    generation_error: Optional[dict] = Field(
        default=None,
        description=(
            "Typed generation failure. Shape: "
            "{kind, user_message, internal_detail}. "
            "Front-ends should render `user_message` when set."
        ),
    )
    llm_confidence: Optional[str] = None  # "high" | "medium" | "low"
    generation_source: Optional[str] = None  # "retrieval" | "memory" | "retrieval+memory"
    workflow_id: Optional[str] = None
    latency_ms: Optional[float] = None
    stage_timings: list[dict] = Field(default_factory=list)
    timing_totals: dict = Field(default_factory=dict)
    budget_exhausted: bool = False
    budget_exhausted_stage: Optional[str] = None
    visual_results: Optional[list[VisualPageResultResponse]] = None  # FR-703
    conversation_id: Optional[str] = None
    token_budget: Optional[TokenBudgetResponse] = None
    # Conversation-scoped doc-id state for the Retrieval tab. The Query tab
    # also benefits: previously-served docs are excluded from this turn's
    # retrieval. ``seen_doc_ids`` is the union of relevant + ignored.
    seen_doc_ids: list[str] = Field(default_factory=list)
    relevant_doc_ids: list[str] = Field(default_factory=list)
    ignored_doc_ids: list[str] = Field(default_factory=list)
    history_decision: Optional[str] = Field(
        default=None,
        description=(
            "For retrieval-mode auto runs: which history strategy the "
            "rewriter chose (use_as_is / partial_history / full_history). "
            "For retrieval-mode hard runs: 'hard_query'. None for "
            "query-mode requests."
        ),
    )
    history_turns_used: int = Field(
        default=0,
        description=(
            "Number of conversation turns the retrieval-mode rewriter "
            "actually consumed when shaping the processed query."
        ),
    )
    dr_suggestion: Optional[dict] = Field(
        default=None,
        description=(
            "Advisory hint for the UI when the user ran in baseline mode "
            "but cheap heuristics indicate Deep Research would have done "
            "better. Shape: {suggest: bool, reason: str|None}. None on the "
            "DR path (suggesting DR while DR is already on is meaningless)."
        ),
    )
    ask_user_reason: Optional[str] = Field(
        default=None,
        description=(
            "Typed reason carried whenever action == 'ask_user'. One of: "
            "'sanitizer_reject', 'injection_blocked', 'vague_query', "
            "'budget_exhausted', 'no_results'. None for any other action. "
            "Front-ends should switch on this to render a reason-specific "
            "clarification message."
        ),
    )
    degraded: bool = Field(
        default=False,
        description=(
            "True when the chain took a degraded fallback path (eg. BM25-only "
            "re-search after the primary hybrid call returned 0 hits) but "
            "still produced usable results."
        ),
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    temporal_connected: bool
    worker_available: bool
    ingest_worker_available: bool = False


class CreateApiKeyRequest(BaseModel):
    subject: str = Field(..., min_length=1, max_length=200)
    tenant_id: Optional[str] = Field(None, description="Tenant bound to this key")
    roles: list[str] = Field(default_factory=lambda: ["query"])
    description: str = Field(default="", max_length=300)


class ApiKeyRecord(BaseModel):
    key_id: str
    subject: str
    tenant_id: str
    roles: list[str]
    description: str = ""
    created_at: int
    revoked_at: Optional[int] = None


class CreateApiKeyResponse(ApiKeyRecord):
    api_key: str


class QuotaUpdateRequest(BaseModel):
    requests_per_minute: int = Field(..., ge=1, le=100000)


class QuotasResponse(BaseModel):
    defaults: dict
    tenants: dict
    projects: dict


class QuotaSetResponse(BaseModel):
    """Payload for tenant quota upsert operations."""
    tenant_id: str
    requests_per_minute: int


class StatusResponse(BaseModel):
    """Standard status payload for simple mutating admin endpoints."""
    status: str
    key_id: Optional[str] = None
    tenant_id: Optional[str] = None


class RootResponse(BaseModel):
    """Root endpoint service metadata."""
    service: str
    docs: str
    health: str
    query_endpoint: str


class ConsoleQueryRequest(BaseModel):
    """Console query request supporting both stream and non-stream modes."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=32000)
    stream: bool = Field(default=True)
    source_filter: Optional[str] = None
    heading_filter: Optional[str] = None
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    search_limit: int = Field(default=10, ge=1, le=100)
    rerank_top_k: int = Field(default=5, ge=1, le=50)
    tenant_id: Optional[str] = None
    max_query_iterations: Optional[int] = Field(default=None, ge=1, le=8)
    fast_path: Optional[bool] = None
    overall_timeout_ms: Optional[int] = Field(default=None, ge=1000, le=180000)
    stage_budget_overrides: dict[str, int] = Field(default_factory=dict)
    conversation_id: Optional[str] = Field(default=None, min_length=3, max_length=128)
    memory_enabled: bool = Field(default=True)
    memory_turn_window: Optional[int] = Field(default=None, ge=1, le=40)
    compact_now: bool = Field(default=False)
    mode: Literal["query", "retrieval"] = Field(default="query")
    retrieval_sub_mode: Literal["auto", "hard"] = Field(default="auto")
    extra_processing: bool = Field(default=False)
    tree_retrieval: Optional[bool] = Field(default=None)
    deep_research: bool = Field(default=False)


# Mapping from ConsoleIngestionRequest field name → IngestionConfig field name.
# Used by to_config() and contract tests.  Only non-None Optional fields
# are overlaid — fields with required defaults (update_mode, …) are always
# forwarded.
INGESTION_REQUEST_FIELD_MAP: dict[str, str] = {
    "update_mode": "update_mode",
    "semantic_chunking": "semantic_chunking",
    "export_processed": "export_processed",
    "verbose_stages": "verbose_stage_logs",
    "persist_refactor_mirror": "persist_refactor_mirror",
    "docling_enabled": "enable_docling_parser",
    "docling_model": "docling_model",
    "docling_artifacts_path": "docling_artifacts_path",
    "docling_strict": "docling_strict",
    "docling_auto_download": "docling_auto_download",
    "vlm_mode": "vlm_mode",
    "hybrid_chunker_max_tokens": "hybrid_chunker_max_tokens",
    "vision_enabled": "enable_vision_processing",
    "vision_provider": "vision_provider",
    "vision_model": "vision_model",
    "vision_api_base_url": "vision_api_base_url",
    "vision_timeout_seconds": "vision_timeout_seconds",
    "vision_max_figures": "vision_max_figures",
    "vision_auto_pull": "vision_auto_pull",
    "vision_strict": "vision_strict",
    "target_collection": "target_collection",
}


class ConsoleIngestionRequest(BaseModel):
    """Console ingestion request for file, directory, or full documents ingestion.

    Fields marked Optional default to None — ``to_config()`` overlays only
    non-None values onto an ``IngestionConfig`` with its own env-var defaults.
    This keeps the API schema decoupled from config defaults.
    """

    model_config = ConfigDict(extra="forbid")

    # ── Execution mode (not part of IngestionConfig) ────────────────────
    mode: Literal["single_file", "directory", "all_documents"] = Field(
        default="all_documents"
    )
    target_path: Optional[str] = Field(
        default=None,
        description="Required for single_file and directory modes",
    )
    # ── Pipeline behavioral knobs ───────────────────────────────────────
    update_mode: bool = Field(default=True)
    semantic_chunking: bool = Field(default=True)
    export_processed: bool = Field(default=False)
    verbose_stages: Optional[bool] = None
    persist_refactor_mirror: Optional[bool] = None
    # ── Docling configuration ───────────────────────────────────────────
    docling_enabled: Optional[bool] = None
    docling_model: Optional[str] = None
    docling_artifacts_path: Optional[str] = None
    docling_strict: Optional[bool] = None
    docling_auto_download: Optional[bool] = None
    # ── VLM / vision configuration ──────────────────────────────────────
    vlm_mode: Optional[Literal["disabled", "builtin", "external"]] = None
    hybrid_chunker_max_tokens: Optional[int] = Field(default=None, ge=64, le=2048)
    vision_enabled: Optional[bool] = None
    vision_provider: Optional[Literal["ollama", "openai_compatible"]] = None
    vision_model: Optional[str] = None
    vision_api_base_url: Optional[str] = None
    vision_timeout_seconds: Optional[int] = Field(default=None, ge=5, le=600)
    vision_max_figures: Optional[int] = Field(default=None, ge=1, le=32)
    vision_auto_pull: Optional[bool] = None
    vision_strict: Optional[bool] = None
    # ── Target collection ───────────────────────────────────────────────
    target_collection: Optional[str] = Field(
        default=None, min_length=1, max_length=128,
        description="Vector store collection name",
    )

    def to_config(self) -> "IngestionConfig":
        """Build an IngestionConfig by overlaying non-None request fields.

        Fields not present in the request (None) keep their IngestionConfig
        env-var defaults.  Fields with explicit values override the defaults.
        """
        from src.ingest.common import IngestionConfig

        overrides: dict[str, Any] = {}
        for req_field, cfg_field in INGESTION_REQUEST_FIELD_MAP.items():
            value = getattr(self, req_field, None)
            if value is not None:
                overrides[cfg_field] = value
        return IngestionConfig(**overrides)


class ConsoleCommandRequest(BaseModel):
    """Unified slash-command request for console query/ingest surfaces."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["query", "ingest"] = Field(default="query")
    command: str = Field(..., min_length=1, max_length=100)
    arg: Optional[str] = Field(default=None, max_length=2000)
    state: dict[str, Any] = Field(default_factory=dict)


class ConsoleLogsResponse(BaseModel):
    """Log snapshot payload for the console."""

    files: list[str] = Field(default_factory=list)
    lines: list[str] = Field(default_factory=list)


class ConsoleHealthSummary(BaseModel):
    """Extended health payload used in console panel."""

    status: str
    temporal_connected: bool
    worker_available: bool
    ingest_worker_available: bool = False
    ollama_reachable: bool


class ConsoleModelInfo(BaseModel):
    """Active generation model + provider exposed to the console UI."""

    generation_enabled: bool
    provider: str
    model: str
    display: str


class ConsoleSourceViewRequest(BaseModel):
    """Source-document view payload. Carries both coordinate systems plus the
    exact chunk text so the viewer can land the highlight precisely regardless
    of which storage path (MinIO clean store vs. raw file) serves the document.
    """

    source: Optional[str] = Field(default=None, max_length=2000)
    source_uri: Optional[str] = Field(default=None, max_length=2000)
    source_key: Optional[str] = Field(default=None, max_length=512)
    chunk_text: Optional[str] = Field(default=None, max_length=32000)
    chunk: Optional[int] = Field(default=None, ge=0)
    original_start: Optional[int] = None
    original_end: Optional[int] = None
    refactored_start: Optional[int] = None
    refactored_end: Optional[int] = None
    provenance_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ConsoleFeedbackTurn(BaseModel):
    role: Literal["user", "assistant", "system"]
    text: str = Field(default="", max_length=32000)


class ConsoleFeedbackRequest(BaseModel):
    """User-supplied thumbs-up/down feedback for an assistant turn."""

    rating: Literal["up", "down"]
    conversation_id: Optional[str] = Field(default=None, max_length=128)
    message_index: Optional[int] = Field(default=None, ge=0)
    query: Optional[str] = Field(default=None, max_length=8000)
    answer: Optional[str] = Field(default=None, max_length=32000)
    comment: Optional[str] = Field(default=None, max_length=4000)
    transcript: Optional[list[ConsoleFeedbackTurn]] = Field(default=None, max_length=200)


class ConversationCreateRequest(BaseModel):
    title: str = Field(default="New conversation", max_length=200)
    conversation_id: Optional[str] = Field(default=None, min_length=3, max_length=128)


class ConversationMetaResponse(BaseModel):
    conversation_id: str
    tenant_id: str
    subject: str
    project_id: str = ""
    title: str = ""
    created_at_ms: int
    updated_at_ms: int
    message_count: int
    summary: dict = Field(default_factory=dict)


class ConversationTurnResponse(BaseModel):
    role: str
    content: str
    timestamp_ms: int
    query_id: str = ""
    sources: list[dict] = Field(default_factory=list)


class ConversationHistoryResponse(BaseModel):
    conversation_id: str
    turns: list[ConversationTurnResponse] = Field(default_factory=list)


class ConversationCompactRequest(BaseModel):
    conversation_id: str = Field(..., min_length=3, max_length=128)


class ConversationTitleUpdateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


# ---------------------------------------------------------------------------
# Document & Collection Management schemas (FR-3060 through FR-3067)
# ---------------------------------------------------------------------------


class DocumentSummary(BaseModel):
    """Summary of a single ingested document (FR-3060)."""

    document_id: str
    source: str
    source_key: str
    connector: str
    chunk_count: Optional[int] = None
    ingested_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Paginated list of ingested documents (FR-3061)."""

    documents: list[DocumentSummary] = []
    total: int
    limit: int
    offset: int


class DocumentDetailResponse(BaseModel):
    """Full content and metadata of a single document (FR-3062)."""

    document_id: str
    content: str
    metadata: dict
    chunk_count: Optional[int] = None


class DocumentUrlResponse(BaseModel):
    """Presigned download URL for a document (FR-3063)."""

    document_id: str
    url: str
    expires_in: int


class SourceSummary(BaseModel):
    """Summary statistics for a single document source (FR-3064)."""

    source: str
    connector: str
    document_count: int
    chunk_count: int


class SourceListResponse(BaseModel):
    """Paginated list of document sources (FR-3065)."""

    sources: list[SourceSummary] = []
    total: int
    limit: int
    offset: int


class CollectionItem(BaseModel):
    """A single vector collection with its chunk count (FR-3067)."""

    collection_name: str
    chunk_count: int


class CollectionStatsResponse(BaseModel):
    """Aggregate statistics for a vector collection (FR-3066)."""

    collection_name: str
    document_count: int
    chunk_count: int
    connector_breakdown: dict[str, int]


class CollectionListResponse(BaseModel):
    """List of all vector collections (FR-3067)."""

    collections: list[CollectionItem]
