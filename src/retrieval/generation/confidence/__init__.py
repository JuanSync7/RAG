# @summary
# Composite confidence scoring for RAG pipeline post-generation evaluation.
# Exports: ConfidenceBreakdown, PostGuardrailAction, compute_composite_confidence, route_by_confidence
# Deps: src.retrieval.generation.confidence.schemas, src.retrieval.generation.confidence.scoring, src.retrieval.generation.confidence.routing
# @end-summary
"""Composite confidence scoring package.

Provides a 3-signal confidence model that combines retrieval quality,
LLM self-reported confidence, and citation coverage into a single
composite score used for post-generation routing decisions.
"""

from src.retrieval.generation.confidence.schemas import (
    ConfidenceBreakdown,
    PostGuardrailAction,
)
from src.retrieval.generation.confidence.scoring import compute_composite_confidence
from src.retrieval.generation.confidence.routing import route_by_confidence

__all__ = [
    "ConfidenceBreakdown",
    "PostGuardrailAction",
    "compute_composite_confidence",
    "route_by_confidence",
]
