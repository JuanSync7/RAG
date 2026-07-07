# @summary
# Pydantic request/response models for the RAG API server.
# Exports: RetrievalStrategy, QueryRequest, QueryResponse, ChunkResult, HealthResponse,
#          DocumentSummary, DocumentListResponse, DocumentDetailResponse,
#          DocumentUrlResponse, SourceSummary, SourceListResponse,
#          CollectionItem, CollectionStatsResponse, CollectionListResponse,
#          VisualPageResultResponse, TurnActionEvent, HydeQueryEvent,
#          RetrieveResultEvent, JudgeVerdictEvent, DeepStudyEvent,
#          LlmCallEvent, DraftEvent, GateEvent, ClarifyEvent,
#          TURN_EVENT_PAYLOAD_MODELS
# Deps: pydantic, src.retrieval.pipeline.turn_loop (event-name constants only)
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

# Single source of truth for retrieval sizing (env-driven). Request schema
# defaults derive from these so there is ONE place to change them.
from config.settings import HYBRID_SEARCH_ALPHA, RERANK_TOP_K, SEARCH_LIMIT


# --- Retrieval-strategy selection (single source of truth) --------------------
# The orchestrator strategies. ``auto`` resolves from the legacy booleans / config;
# ``linear`` forces the plain pipeline; the rest each name ONE orchestrator that
# replaces the linear stages. The enum makes the choice STRUCTURAL — you cannot
# select two — so the former pairwise "X and Y cannot both be forced" validators
# collapse into the single resolver below (Phase 0b waypoint toward an effort dial).
RetrievalStrategy = Literal["auto", "linear", "agentic", "deep_research", "tree", "turn_loop"]

# strategy -> the legacy request boolean it drives (the dispatch still reads these,
# so mapping enum<->boolean keeps behaviour identical whichever the caller sends).
_STRATEGY_TO_BOOL = {
    "deep_research": "deep_research",
    "agentic": "agentic_retrieval",
    "tree": "tree_retrieval",
    "turn_loop": "turn_loop",
}


def _forced_orchestrators(model: BaseModel) -> list[str]:
    """Strategies explicitly forced via a legacy boolean (``True`` — not None/False)."""
    forced: list[str] = []
    if getattr(model, "deep_research", False):
        forced.append("deep_research")
    if getattr(model, "agentic_retrieval", None) is True:
        forced.append("agentic")
    if getattr(model, "tree_retrieval", None) is True:
        forced.append("tree")
    if getattr(model, "turn_loop", None) is True:
        forced.append("turn_loop")
    return forced


def _resolve_retrieval_strategy(model: BaseModel) -> BaseModel:
    """Reconcile ``retrieval_strategy`` <-> the legacy booleans and enforce the
    single-orchestrator contract. Shared by QueryRequest and ConsoleQueryRequest so
    the selection rules live in ONE place (CLI/UI parity)."""
    forced = _forced_orchestrators(model)
    strat = getattr(model, "retrieval_strategy", "auto")

    if strat == "auto":
        if len(forced) > 1:
            raise ValueError(
                "only one retrieval orchestrator may be forced at once — got "
                f"{forced}. deep_research / agentic_retrieval / tree_retrieval / "
                "turn_loop each replace the linear stages and are mutually exclusive."
            )
        if forced:
            model.retrieval_strategy = forced[0]
    else:
        want = None if strat == "linear" else strat
        conflicting = [f for f in forced if f != want]
        if conflicting:
            raise ValueError(
                f"retrieval_strategy={strat!r} conflicts with forced {conflicting} "
                "— use one selection mechanism (the enum or the legacy booleans)."
            )
        # An explicit strategy overrides the RAG_*_ENABLED config defaults: set the
        # matching boolean on and the competitors off so the dispatch sees one mode.
        for s, attr in _STRATEGY_TO_BOOL.items():
            if hasattr(model, attr):
                setattr(model, attr, s == strat)

    eff = getattr(model, "retrieval_strategy", "auto")
    if eff == "turn_loop" and getattr(model, "mode", "query") == "retrieval":
        raise ValueError(
            "turn_loop cannot be combined with mode='retrieval' — retrieval mode "
            "skips generation while the turn loop's terminal action IS generation."
        )
    if (
        eff == "agentic"
        and getattr(model, "fast_path", None) is True
        and getattr(model, "max_agentic_rounds", None) is not None
        and model.max_agentic_rounds > 1
    ):
        raise ValueError(
            "fast_path forces a single agentic round; max_agentic_rounds>1 contradicts it."
        )
    return model


class QueryRequest(BaseModel):
    """Incoming query from a user."""
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=32000, description="The search query")
    source_filter: Optional[str] = Field(None, description="Filter by source document filename")
    heading_filter: Optional[str] = Field(None, description="Filter by section heading")
    alpha: float = Field(HYBRID_SEARCH_ALPHA, ge=0.0, le=1.0, description="Hybrid search balance (0=BM25, 1=vector)")
    search_limit: int = Field(SEARCH_LIMIT, ge=1, le=100, description="Max results from hybrid search")
    rerank_top_k: int = Field(RERANK_TOP_K, ge=1, le=50, description="Top-K results after reranking")
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
    agentic_retrieval: Optional[bool] = Field(
        default=None,
        description=(
            "Per-request override for the agentic HyDE/controller/judge "
            "retrieval loop. ``None`` (default) uses the "
            "``RAG_AGENTIC_RETRIEVAL_ENABLED`` config setting. ``true``/"
            "``false`` force on/off for this request only. Replaces the "
            "linear kg_expand → embed → hybrid_search → rerank stages with a "
            "controller LLM that drives retrieval as a tool, generating HyDE "
            "and judging each chunk for relevance/faithfulness until the kept "
            "set is sufficient. See AGENTIC_RETRIEVAL_DESIGN.md."
        ),
    )
    max_agentic_rounds: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description=(
            "Per-request override of the agentic round cap (number of distinct "
            "HyDE variants tried). Clamped to ``RAG_AGENTIC_MAX_ROUNDS`` at "
            "runtime. Only consulted when the agentic loop is active."
        ),
    )
    turn_loop: Optional[bool] = Field(
        default=None,
        description=(
            "Per-request override of ``RAG_TURN_LOOP_ENABLED`` (the "
            "turn-level agentic conversation loop). ``None`` (default) uses "
            "the config setting; ``true``/``false`` force on/off for this "
            "request only. The loop iterates RETRIEVE / DEEP_STUDY / CLARIFY "
            "/ ANSWER actions until a final-answer confidence gate passes. "
            "Mutually exclusive with ``deep_research``, ``agentic_retrieval``, "
            "``tree_retrieval`` and ``mode='retrieval'`` — the loop owns query "
            "design, composes the retrieval primitives itself, and always "
            "generates. When this override is None, explicitly enabling any "
            "of those on the request also wins over the "
            "``RAG_TURN_LOOP_ENABLED`` env default. See TURN_LOOP_DESIGN.md."
        ),
    )
    retrieval_strategy: RetrievalStrategy = Field(
        default="auto",
        description=(
            "Unified orchestrator selector (Phase 0b waypoint). ``auto`` (default) "
            "resolves from the legacy booleans / config; ``linear`` forces the plain "
            "pipeline; ``agentic`` / ``deep_research`` / ``tree`` / ``turn_loop`` each "
            "select one orchestrator. Mutually exclusive by construction — replaces "
            "the pairwise 'cannot both' validation. The legacy booleans remain "
            "accepted and are mapped to/from this field; a future release collapses "
            "this into an ``effort`` dial."
        ),
    )

    @model_validator(mode="after")
    def _resolve_strategy(self) -> "QueryRequest":
        return _resolve_retrieval_strategy(self)

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
    suggested_questions: Optional[list[str]] = Field(
        default=None,
        description=(
            "Clickable 'you might also ask...' follow-up questions proposed after "
            "the answer. Specific, in-corpus, grounded in the answer + retrieved "
            "headings. None/empty when disabled or on any generation failure."
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
    metadata: dict = Field(
        default_factory=dict,
        description=(
            "Free-form retrieval telemetry passed through from RAGResponse.metadata. "
            "Carries the query-processing detail the UI surfaces: "
            "metadata['agentic_retrieval'] (HyDE hypotheticals tried, per-round "
            "hyde_rounds, rounds_run, hyde_failures, stop_reason, kept_count, LLM/"
            "judge call counts) and metadata['deep_research'] (decomposition stats). "
            "Empty on the linear (non-agentic, non-DR) path. Already present on the "
            "stream path's 'retrieval' SSE event; this field exposes it on /query too."
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
    alpha: float = Field(default=HYBRID_SEARCH_ALPHA, ge=0.0, le=1.0)
    search_limit: int = Field(default=SEARCH_LIMIT, ge=1, le=100)
    rerank_top_k: int = Field(default=RERANK_TOP_K, ge=1, le=50)
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
    agentic_retrieval: Optional[bool] = Field(default=None)
    max_agentic_rounds: Optional[int] = Field(default=None, ge=1, le=20)
    turn_loop: Optional[bool] = Field(
        default=None,
        description=(
            "Per-request override of RAG_TURN_LOOP_ENABLED (turn-level "
            "agentic conversation loop). Mirrors QueryRequest.turn_loop for "
            "CLI/UI parity: the admin console can opt in/out per request; "
            "None uses the config default. Mutually exclusive with "
            "deep_research/agentic_retrieval/tree_retrieval and "
            "mode='retrieval'."
        ),
    )

    retrieval_strategy: RetrievalStrategy = Field(
        default="auto",
        description=(
            "Unified orchestrator selector (Phase 0b). Mirrors "
            "QueryRequest.retrieval_strategy for CLI/UI parity. ``auto`` resolves "
            "from the legacy booleans / config; ``linear`` forces the plain pipeline; "
            "the rest select one orchestrator. Mutually exclusive by construction."
        ),
    )

    @model_validator(mode="after")
    def _resolve_strategy(self) -> "ConsoleQueryRequest":
        return _resolve_retrieval_strategy(self)


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


# ---------------------------------------------------------------------------
# Turn-loop SSE payload models (TURN_LOOP_DESIGN.md §8)
#
# One pydantic model per typed turn-loop stream event. The API-process runner
# (``server/turn_loop_runner.py``) validates every loop event payload against
# these before it is serialized into an SSE frame, so the stream vocabulary is
# a typed contract rather than a convention. Event names are single-sourced
# from ``TurnEventType`` (the frozen turn-loop contract module) via
# ``TURN_EVENT_PAYLOAD_MODELS``. All models tolerate extra keys
# (``extra="allow"``) so an upstream loop stage can enrich a payload without a
# lockstep schema change — unknown fields pass through to clients verbatim.
# ---------------------------------------------------------------------------

from src.retrieval.pipeline.turn_loop import TurnEventType


class TurnActionEvent(BaseModel):
    """Payload of the ``turn_action`` SSE event: one action decision.

    ``source`` names who chose the action: ``controller`` (the controller LLM)
    or ``router`` (the pre-flight fast lane, chosen deterministically with no
    LLM call). Defaults ``controller`` for pre-router traces.
    """

    model_config = ConfigDict(extra="allow")

    index: int = 0
    action: str = ""
    reason: str = ""
    confidence: float = 0.0
    source: str = "controller"


class HydeQueryEvent(BaseModel):
    """Payload of the ``hyde_query`` SSE event: one HyDE generation round."""

    model_config = ConfigDict(extra="allow")

    round: int = 0
    hypothetical_answer: str = ""
    search_terms: list[str] = Field(default_factory=list)
    target_aspect: Optional[str] = None


class RetrieveResultEvent(BaseModel):
    """Payload of the ``retrieve_result`` SSE event: one retrieval round's
    pool delta. ``top`` carries abridged ``{doc, heading, score}`` dicts."""

    model_config = ConfigDict(extra="allow")

    round: int = 0
    added: int = 0
    dup: int = 0
    pool_size: int = 0
    top: list[dict] = Field(default_factory=list)


class JudgeVerdictEvent(BaseModel):
    """Payload of the ``judge_verdict`` SSE event: the evidence judge's
    keep/sufficiency verdict for one retrieval round.

    ``missing_information`` is free text (possibly empty) — the agentic judge
    names its gaps as one prose string and the loop forwards it verbatim.
    """

    model_config = ConfigDict(extra="allow")

    round: int = 0
    kept: int = 0
    sufficient: bool = False
    confidence: float = 0.0
    missing_information: str = ""


class DeepStudyEvent(BaseModel):
    """Payload of the ``deep_study`` SSE event: one windowed document read."""

    model_config = ConfigDict(extra="allow")

    document_id: str = ""
    title: str = ""
    window: int = 0
    of_windows: int = 0
    notes_preview: str = ""


class LlmCallEvent(BaseModel):
    """Payload of the ``llm_call`` SSE event: one charged provider call."""

    model_config = ConfigDict(extra="allow")

    alias: str = ""
    purpose: str = ""
    ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class DraftEvent(BaseModel):
    """Payload of the ``draft`` SSE event: a live answer-draft delta.

    ``kind`` distinguishes ``content`` deltas (answer text) from
    ``reasoning`` deltas (chain-of-thought from reasoning models); the route
    replays the accepted attempt's reasoning deltas as a ``reasoning`` frame
    before the answer's token replay.
    """

    model_config = ConfigDict(extra="allow")

    attempt: int = 0
    kind: str = "content"
    text_delta: str = ""


class GateEvent(BaseModel):
    """Payload of the ``gate`` SSE event: one answer-confidence gate verdict."""

    model_config = ConfigDict(extra="allow")

    attempt: int = 0
    score: float = 0.0
    threshold: float = 0.0
    passed: bool = False
    weakest: str = ""


class ClarifyEvent(BaseModel):
    """Payload of the ``clarify`` SSE event: the LLM-authored clarification."""

    model_config = ConfigDict(extra="allow")

    question: str = ""
    hints: list[str] = Field(default_factory=list)
    scoping_questions: list[str] = Field(default_factory=list)


TURN_EVENT_PAYLOAD_MODELS: dict[str, type[BaseModel]] = {
    TurnEventType.TURN_ACTION: TurnActionEvent,
    TurnEventType.HYDE_QUERY: HydeQueryEvent,
    TurnEventType.RETRIEVE_RESULT: RetrieveResultEvent,
    TurnEventType.JUDGE_VERDICT: JudgeVerdictEvent,
    TurnEventType.DEEP_STUDY: DeepStudyEvent,
    TurnEventType.LLM_CALL: LlmCallEvent,
    TurnEventType.DRAFT: DraftEvent,
    TurnEventType.GATE: GateEvent,
    TurnEventType.CLARIFY: ClarifyEvent,
}
"""Event-name → payload-model registry (names 1:1 with ``TurnEventType``)."""
