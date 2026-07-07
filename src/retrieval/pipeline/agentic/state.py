# @summary
# Typed contracts for the agentic retrieval loop: the frozen per-request budget,
# the HyDE variant, per-chunk + pool judge verdicts, the mutable per-request
# state accumulator, and the orchestrator result.
# Exports: AgenticBudget, HydeVariant, ChunkVerdict, PoolVerdict, AgenticState, AgenticResult
# Deps: dataclasses, typing, src.retrieval.common.schemas
# @end-summary
"""Typed state/contract dataclasses for the agentic HyDE/controller/judge loop.

These are deliberately free of config imports — the orchestrator is constructed
with an :class:`AgenticBudget` filled from ``RAG_AGENTIC_*`` by the caller, so
this module stays a pure, side-effect-free contract surface (CLAUDE.md §2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.retrieval.common.schemas import RankedResult


@dataclass(frozen=True)
class AgenticBudget:
    """Immutable per-request caps, sourced from ``RAG_AGENTIC_*`` by the caller.

    Frozen so a round cannot mutate the budget mid-loop (sibling of
    ``DeepResearchBudget``).
    """

    max_rounds: int
    max_llm_calls: int
    wall_clock_ms: int
    keep_top_k_per_round: int
    final_max_chunks: int
    min_kept_chunks: int
    min_sources: int
    relevance_threshold: float
    faithfulness_threshold: float
    sufficiency_target: float
    hyde_diversity_max_cosine: float
    # INC-2: which component ORDERS the pool — "judge" (LLM listwise rank over the
    # raw hybrid pool, the measured-best path) or "cross_encoder" (INC-1). Default
    # keeps a frozen budget constructible without the new fields (tests/back-compat).
    ranker: str = "judge"
    # INC-2: max raw-hybrid candidates judged per round in ranker="judge" mode.
    judge_pool_max: int = 40


@dataclass
class HydeVariant:
    """A single hypothetical-answer (HyDE) retrieval query produced by the
    controller. ``hypothetical_answer`` is embedded (answer-space); the literal
    ``search_terms`` anchor the lexical/BM25 side to the user's terminology."""

    hypothetical_answer: str
    search_terms: list[str] = field(default_factory=list)
    target_aspect: str = ""


@dataclass
class ChunkVerdict:
    """The judge's per-chunk scoring of one retrieved candidate.

    ``relevance`` / ``faithfulness`` are generic [0,1] properties (does it bear
    on the question; is it substantive, self-contained source evidence) — never
    a vendor/phrase match (CLAUDE.md §0).
    """

    index: int
    relevance: float
    faithfulness: float
    keep: bool
    reason: str = ""
    # INC-2: this chunk's position in the judge's LISTWISE ranking for the round
    # (0 = best). -1 means the judge did not rank it (omitted, or fail-open). The
    # ORDER fed to generation comes from this rank, not from the pointwise
    # ``relevance`` (which only gates keep/drop) — pointwise scores tie heavily
    # and an ordering built from them collapses back to raw-hybrid order.
    rank: int = -1


@dataclass
class PoolVerdict:
    """The judge's set-level verdict over the kept pool so far."""

    sufficient: bool
    confidence: float
    missing_information: str = ""
    covered_aspects: list[str] = field(default_factory=list)


@dataclass
class AgenticState:
    """Mutable per-request loop state. Instantiated fresh per request inside the
    orchestrator and discarded — NEVER a module/class global (avoids the
    cross-request bleed that a shared ``last_result`` attribute would cause)."""

    original_question: str
    processed_query: str
    round_index: int = 0
    # Judge-approved chunks accumulated across all rounds. Cross-round
    # uniqueness is guaranteed by ``seen_chunk_ids`` (a chunk retrieved in an
    # earlier round is filtered out before it can be judged/kept again), so this
    # list never holds duplicates.
    kept: list[RankedResult] = field(default_factory=list)
    # Stable ids (SearchResult.object_id-based) of every chunk ever retrieved —
    # the master dedup that prevents re-judging the same chunk across rounds.
    seen_chunk_ids: set[str] = field(default_factory=set)
    # HyDE-variant history (for INC-2 diversification + executed-search dedup).
    tried_hyde: list[str] = field(default_factory=list)
    # Structured per-round record of the FULL HyDE variant the controller produced
    # (hypothetical answer + lexical anchor terms + target aspect + whether the
    # round fell back to literal-query retrieval). tried_hyde keeps only the answer
    # text for diversification; this is the observability record surfaced to the UI
    # so the whole query-processing step is inspectable, not just a count.
    hyde_rounds: list[dict] = field(default_factory=list)
    # The previous round's judge-named gap, steering the next HyDE (INC-2).
    coverage_aspects: list[str] = field(default_factory=list)
    # Cross-round anti-refusal reservoir: EVERY judged candidate from EVERY round,
    # deduped by stable chunk-id. The finalize floor backfills from this — NEVER
    # from just the last round, because the terminating round is frequently the
    # empty ``no_progress`` round (which would otherwise yield empty context on a
    # non-empty retrieval — the refusal regression the floor exists to prevent).
    best_seen: list[RankedResult] = field(default_factory=list)
    best_seen_ids: set[str] = field(default_factory=set)
    # Per-round signals read by the stop gates / finalize.
    last_pool: "Optional[PoolVerdict]" = None
    last_fresh_count: int = 0
    last_newly_kept: int = 0
    last_ranked: bool = False  # did THIS round's judge return a usable ranking?
    llm_calls: int = 0
    judge_calls: int = 0
    ranker_calls: int = 0  # finalize cross-round re-rank calls
    backfilled: int = 0    # chunks added by the anti-refusal floor
    # Rounds where HyDE failed and fell back to literal-query retrieval — a
    # silent-degradation alarm (full rationale at _next_hyde, where it's counted).
    hyde_failures: int = 0
    stop_reason: Optional[str] = None


@dataclass
class AgenticResult:
    """What :meth:`AgenticRetrieval.run` returns to the chain: the final curated
    ranked chunks plus loop telemetry. Mirrors how the deep-research branch hands
    back ``reranked`` + ``graph_context`` + a metadata dict."""

    reranked: list[RankedResult]
    graph_context: str = ""
    rounds_run: int = 0
    hyde_variants_tried: int = 0
    # The actual per-round HyDE hypothetical texts (INC-2 diversification history).
    # Carried into telemetry so the generated hypotheticals are observable and can
    # be judged for quality — a count alone can't reveal an on-topic-but-weak HyDE.
    tried_hyde: list[str] = field(default_factory=list)
    # Structured per-round HyDE variant records (hypothetical_answer + search_terms
    # + target_aspect + fell_back), surfaced to the UI query-processing panel so the
    # whole HyDE step — not just a count — is visible before generation.
    hyde_rounds: list[dict] = field(default_factory=list)
    kept_count: int = 0
    llm_calls: int = 0
    judge_calls: int = 0
    ranker: str = "judge"
    ranker_calls: int = 0
    backfilled: int = 0
    # Per-request count of HyDE failures (silent literal-query fallback); telemetry.
    hyde_failures: int = 0
    stop_reason: str = ""
    elapsed_ms: float = 0.0

    def telemetry(self) -> dict:
        """Free-form telemetry dict for ``response.metadata['agentic_retrieval']``
        (no wire-schema change — the metadata field already exists)."""
        return {
            "rounds_run": int(self.rounds_run),
            "hyde_variants_tried": int(self.hyde_variants_tried),
            "kept_count": int(self.kept_count),
            "llm_calls": int(self.llm_calls),
            "judge_calls": int(self.judge_calls),
            "ranker": self.ranker,
            "ranker_calls": int(self.ranker_calls),
            "backfilled": int(self.backfilled),
            "hyde_failures": int(self.hyde_failures),
            "tried_hyde": list(self.tried_hyde),
            "hyde_rounds": list(self.hyde_rounds),
            "stop_reason": self.stop_reason,
            "elapsed_ms": float(self.elapsed_ms),
        }
