# @summary
# Typed contracts for composite confidence scoring and post-generation routing.
# Exports: ConfidenceBreakdown, PostGuardrailAction
# Deps: dataclasses, enum
# @end-summary
"""Typed contracts for confidence scoring."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
    retrieval_weight: float = 0.50
    llm_weight: float = 0.25
    citation_weight: float = 0.25


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
