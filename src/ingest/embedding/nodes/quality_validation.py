# @summary
# LangGraph node for optional chunk quality gating and de-duplication.
# Exports: quality_validation_node
# Deps: embedding.state
# @end-summary

"""Quality-validation node implementation."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from src.ingest.common import (
    append_processing_log,
    quality_score,
    chunk_body_text,
)
from src.ingest.embedding.state import EmbeddingPipelineState
from src.ingest.common.observability import node_span

logger = logging.getLogger("rag.ingest.embedding.quality_validation")

_WHITESPACE_RE = re.compile(r"\s+")


@node_span("quality_validation")
def quality_validation_node(state: EmbeddingPipelineState) -> dict[str, Any]:
    """Filter chunks by heuristic quality thresholds and deduplicate text.

    Args:
        state: Ingestion pipeline state.

    Returns:
        Partial state update containing filtered ``chunks`` and an updated
        ``processing_log``. When disabled, returns only a skipped log entry.
    """
    t0 = time.monotonic()
    config = state["runtime"].config
    if not config.enable_quality_validation:
        return {
            "processing_log": append_processing_log(state, "quality_validation:skipped")
        }

    filtered_chunks = []
    seen_normalized = set()
    for chunk in state["chunks"]:
        # Gate on the BODY, not the embedded text. HybridChunker output is
        # contextualized (heading breadcrumb prepended via contextualize()), so
        # measuring chunk.text would let a 9-char body under a 3-level heading
        # read as ~60 chars and survive the floor — the "title-only chunk"
        # pathology. chunk_body_text() strips the breadcrumb deterministically
        # and is a no-op on non-contextualized chunks (tables/figures/legacy).
        body = chunk_body_text(chunk.text, chunk.metadata.get("heading_path"))
        body_text = body.strip()
        if len(body_text) < config.min_chunk_chars:
            continue
        # De-duplicate on the FULL embedded text so identical bodies under
        # DIFFERENT section headings are correctly retained as distinct chunks.
        normalized = _WHITESPACE_RE.sub(" ", chunk.text.strip()).lower()
        if normalized in seen_normalized:
            continue
        try:
            score = quality_score(body_text)
        except Exception:
            logger.warning("Quality score computation failed; defaulting to 0.0", exc_info=True)
            score = 0.0  # fail safe: low quality
        if score < config.min_quality_score:
            continue
        seen_normalized.add(normalized)
        filtered_chunks.append(chunk)

    logger.info("quality_validation complete: source=%s input=%d output=%d filtered=%d", state.get("source_name", ""), len(state["chunks"]), len(filtered_chunks), len(state["chunks"]) - len(filtered_chunks))
    logger.debug("quality_validation_node completed in %.3fs", time.monotonic() - t0)
    return {
        "chunks": filtered_chunks,
        "processing_log": append_processing_log(state, "quality_validation:ok"),
    }
