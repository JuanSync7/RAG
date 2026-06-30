# @summary
# Pipeline boundary contracts: input/output schemas and the cross-cutting wire type.
# Exports: RAGRequest, RAGResponse, RankedResult, VisualPageResult
# RAGResponse includes first_composite (initial confidence before internal re-retrieval loop).
# Deps: dataclasses, typing
# @end-summary
"""Pipeline-level schema contracts — what enters and exits the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.platform.token_budget import TokenBudgetSnapshot


@dataclass
class RankedResult:
    """A search result with its reranking score.

    Wire type that flows from query (reranker output) through to generation.
    """

    text: str
    score: float
    metadata: dict


@dataclass
class VisualPageResult:
    """A single visual page retrieval result with its similarity score.

    Represents a document page matched by ColQwen2 visual embedding search,
    with a presigned MinIO URL for direct image access.
    """

    document_id: str          # Document identifier (FR-501)
    page_number: int          # 1-indexed page number (FR-501)
    source_key: str           # Stable source key for traceability (FR-501)
    source_name: str          # Human-readable source name (FR-501)
    score: float              # Cosine similarity 0.0-1.0 (FR-501)
    page_image_url: str       # Presigned MinIO URL (FR-501, FR-607)
    total_pages: int          # Total pages in source document (FR-501)
    page_width_px: int        # Page image width in pixels (FR-501)
    page_height_px: int       # Page image height in pixels (FR-501)


@dataclass
class RAGRequest:
    """Input contract for the RAG pipeline.

    Optional numeric fields default to None — the pipeline fills in
    config defaults when None, keeping this schema config-free.
    """

    query: str
    alpha: Optional[float] = None
    search_limit: Optional[int] = None
    rerank_top_k: Optional[int] = None
    source_filter: Optional[str] = None
    heading_filter: Optional[str] = None
    tenant_id: Optional[str] = None
    skip_generation: bool = False
    fast_path: Optional[bool] = None
    memory_context: Optional[str] = None
    memory_recent_turns: Optional[list[dict[str, str]]] = None
    conversation_id: Optional[str] = None
    overall_timeout_ms: Optional[int] = None
    stage_budget_overrides: Optional[dict[str, int]] = None
    max_query_iterations: Optional[int] = None
    # Tree retrieval per-request override (TREE_RETRIEVAL_DESIGN.md §6).
    # None = use the global ``RAG_TREE_RETRIEVAL_ENABLED`` config default.
    # True/False = force on/off for this request only.
    tree_retrieval: Optional[bool] = None
    # Per-request retrieval-mode flags that flow straight to ``RAGChain.run``.
    mode: str = "query"
    retrieval_sub_mode: str = "auto"
    extra_processing: bool = False
    deep_research: bool = False
    # Agentic-retrieval per-request override (AGENTIC_RETRIEVAL_DESIGN.md).
    # None = use the global ``RAG_AGENTIC_RETRIEVAL_ENABLED`` config default.
    agentic_retrieval: Optional[bool] = None
    max_agentic_rounds: Optional[int] = None


@dataclass
class RAGResponse:
    """Complete response from the retrieval pipeline."""

    query: str
    processed_query: str
    query_confidence: float
    action: str
    results: list[RankedResult] = field(default_factory=list)
    clarification_message: Optional[str] = None
    kg_expanded_terms: Optional[list[str]] = None
    generated_answer: Optional[str] = None
    stage_timings: list[dict[str, Any]] = field(default_factory=list)
    timing_totals: dict[str, float] = field(default_factory=dict)
    budget_exhausted: bool = False
    budget_exhausted_stage: Optional[str] = None
    conversation_id: Optional[str] = None
    guardrails: Optional[dict[str, Any]] = None
    token_budget: Optional["TokenBudgetSnapshot"] = None
    composite_confidence: Optional[float] = None
    confidence_breakdown: Optional[dict[str, Any]] = None
    post_guardrail_action: Optional[str] = None
    version_conflicts: Optional[list[dict[str, Any]]] = None
    retry_count: int = 0
    verification_warning: Optional[str] = None
    retrieval_quality: Optional[str] = None  # "strong" | "moderate" | "weak" | "insufficient"
    retrieval_quality_note: Optional[str] = None
    re_retrieval_suggested: bool = False
    re_retrieval_params: Optional[dict[str, Any]] = None
    first_composite: Optional[float] = None
    visual_results: Optional[list["VisualPageResult"]] = None  # FR-503
    generation_source: Optional[str] = None       # "retrieval" | "memory" | "retrieval+memory" | None (REQ-1209)
    llm_confidence: Optional[str] = None            # "high" | "medium" | "low" | None — LLM self-reported confidence (REQ-604)
    # Typed generation failure (issue #7). Carries kind/user_message/internal_detail
    # so front-ends can render a meaningful message instead of a blank answer.
    generation_error: Optional[dict[str, Any]] = None
    # Retrieval-tab explainability — populated by the auto-mode rewriter in
    # S4. ``history_decision`` reports which strategy the LLM picked
    # ("use_as_is" | "partial_history" | "full_history" | "hard_query").
    history_decision: Optional[str] = None
    history_turns_used: int = 0
    # Advisory hint for the UI: "this baseline-mode query looks multi-topic —
    # try Deep Research?". Populated only on the baseline path; absent (None)
    # when DR was already active. Shape: {"suggest": bool, "reason": str|None}.
    dr_suggestion: Optional[dict[str, Any]] = None
    # Typed reason carried whenever ``action == "ask_user"``. One of the
    # ``AskUserReason`` enum string values (sanitizer_reject, injection_blocked,
    # vague_query, budget_exhausted, no_results). Additive — the existing
    # ``action == "ask_user"`` string check still works; this enum gives UIs
    # enough signal to pick a reason-specific clarification message.
    ask_user_reason: Optional[str] = None
    # True when the chain took a degraded fallback path (eg. BM25-only re-search
    # after the primary hybrid returned 0 hits) but still produced results.
    degraded: bool = False
    # Free-form structured metadata for observability surfaces (Temporal search
    # attributes, evals, debug payloads). Keys are namespaced by feature, e.g.
    # ``metadata["deep_research"] = {"iteration_count": 3, "llm_call_count": 7,
    # "decomposed": True, "topic_count": 2, "is_unified": False, ...}``.
    # Always populated for DR runs; empty dict otherwise.
    metadata: dict = field(default_factory=dict)
