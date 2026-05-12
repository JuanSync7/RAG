# @summary
# Typed contracts for composite confidence scoring and post-generation routing.
# Exports: ConfidenceBreakdown, ConfidenceWeights, PostGuardrailAction
# Deps: dataclasses, enum
# @end-summary
"""Typed contracts for confidence scoring."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from config.settings import (
    RAG_CONFIDENCE_CITATION_WEIGHT,
    RAG_CONFIDENCE_LLM_WEIGHT,
    RAG_CONFIDENCE_RETRIEVAL_WEIGHT,
)


@dataclass(frozen=True)
class ConfidenceWeights:
    """Frozen, validated weight triple for composite confidence scoring.

    Validation happens once at construction time (via __post_init__), so
    compute_composite_confidence pays zero per-call overhead.

    All weights must be in [0.0, 1.0] and sum to exactly 1.0.
    """

    retrieval: float
    llm: float
    citation: float

    def __post_init__(self) -> None:
        total = self.retrieval + self.llm + self.citation
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Confidence weights must sum to 1.0, got {total:.6f} "
                f"(retrieval={self.retrieval}, llm={self.llm}, citation={self.citation})"
            )


@dataclass
class ConfidenceBreakdown:
    """Three-signal composite confidence breakdown.

    Combines three independent signals into a weighted composite score:
    - retrieval_score: objective signal from cross-encoder reranker scores
    - llm_score: subjective signal from LLM self-reported confidence
    - citation_score: structural signal from citation coverage analysis

    All scores are in [0.0, 1.0]. Weights must sum to 1.0.
    """

    retrieval_score: float
    llm_score: float
    citation_score: float
    composite: float
    retrieval_weight: float = RAG_CONFIDENCE_RETRIEVAL_WEIGHT
    llm_weight: float = RAG_CONFIDENCE_LLM_WEIGHT
    citation_weight: float = RAG_CONFIDENCE_CITATION_WEIGHT


class PostGuardrailAction(Enum):
    """Routing action after post-generation confidence evaluation.

    RETURN: Answer is confident enough to return to the user.
    RE_RETRIEVE: Retry with broader search parameters (max 1 retry).
    FLAG: Return answer with a verification warning attached.
    BLOCK: Replace answer with a fallback "insufficient documentation" message.
    """

    RETURN = "return"
    RE_RETRIEVE = "re_retrieve"
    FLAG = "flag"
    BLOCK = "block"
