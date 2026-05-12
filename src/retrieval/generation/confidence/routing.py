# @summary
# Post-generation confidence routing logic implementing REQ-706.
# Exports: route_by_confidence
# Deps: src.retrieval.generation.confidence.schemas, logging, math
# @end-summary
"""Post-generation confidence routing.

Implements the routing decision table from REQ-706:

  composite >= high               -> RETURN
  composite < high, retries left  -> RE_RETRIEVE
  retries exhausted, c >= low     -> FLAG
  retries exhausted, c < low      -> BLOCK
"""

from __future__ import annotations

import logging
import math

from config.settings import (
    RAG_CONFIDENCE_HIGH_THRESHOLD,
    RAG_CONFIDENCE_LOW_THRESHOLD,
    RAG_CONFIDENCE_RE_RETRIEVE_MAX_RETRIES,
)
from src.retrieval.generation.confidence.schemas import PostGuardrailAction

logger = logging.getLogger("rag.confidence_routing")


def route_by_confidence(
    composite: float,
    retry_count: int = 0,
    high_threshold: float = RAG_CONFIDENCE_HIGH_THRESHOLD,
    low_threshold: float = RAG_CONFIDENCE_LOW_THRESHOLD,
    max_retries: int = RAG_CONFIDENCE_RE_RETRIEVE_MAX_RETRIES,
) -> PostGuardrailAction:
    """Determine routing action based on composite confidence score.

    The routing table balances answer quality against latency:
    - High confidence answers are returned immediately.
    - Medium or low confidence triggers one re-retrieval attempt with
      broader search parameters before falling back to FLAG or BLOCK.

    Args:
        composite: Composite confidence score in [0.0, 1.0]. Must be a
            finite float; NaN is treated as a safe-fail BLOCK.
        retry_count: Number of re-retrieval attempts already made.
        high_threshold: Score at or above which answers are returned.
            Default RAG_CONFIDENCE_HIGH_THRESHOLD.
        low_threshold: Score below which answers are blocked after
            exhausting retries. Default RAG_CONFIDENCE_LOW_THRESHOLD.
        max_retries: Maximum number of re-retrieval attempts. Default RAG_CONFIDENCE_RE_RETRIEVE_MAX_RETRIES.

    Returns:
        PostGuardrailAction indicating what to do with the answer.
    """
    # Safe-fail guard: NaN from upstream scoring errors must never
    # silently produce a permissive action.
    if math.isnan(composite):
        logger.error(
            "route_by_confidence received NaN composite score — "
            "blocking as safe-fail. Check upstream scoring pipeline."
        )
        return PostGuardrailAction.BLOCK

    # Single-pass routing table (REQ-706).
    # No duplicate logic: high confidence always returns, retries available
    # always re-retrieve, and only when retries are exhausted do we fall
    # through to the FLAG/BLOCK tier decision.
    if composite >= high_threshold:
        return PostGuardrailAction.RETURN

    if retry_count < max_retries:
        return PostGuardrailAction.RE_RETRIEVE

    # Retries exhausted — route by confidence tier.
    if composite >= low_threshold:
        return PostGuardrailAction.FLAG
    return PostGuardrailAction.BLOCK
