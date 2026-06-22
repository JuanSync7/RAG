# @summary
# LangGraph node for chunk de-duplication and empty-chunk removal. No size or
# heuristic-quality gating (chunking already reaches for target size; gating was
# vestigial + lossy).
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
    chunk_body_text,
)
from src.ingest.embedding.state import EmbeddingPipelineState
from src.ingest.common.observability import node_span

logger = logging.getLogger("rag.ingest.embedding.quality_validation")

_WHITESPACE_RE = re.compile(r"\s+")


@node_span("quality_validation")
def quality_validation_node(state: EmbeddingPipelineState) -> dict[str, Any]:
    """Drop content-free chunks and exact duplicates.

    No size or heuristic-quality gating: chunking already reaches for the target
    size (prose coalesces to ``native_min_chunk_chars``; tables pack to the token
    budget), so the old small-chunk floor and length/digit "quality" score gate
    were vestigial AND lossy (they silently dropped small tables, big-table
    tails, and short content the lossless verifier now expects to be covered).
    This node now does only two lossless things:

      * skips a chunk whose BODY is empty/whitespace — never embed a
        content-free string (there is nothing to lose), and
      * removes EXACT duplicates — a dropped duplicate's content remains in the
        index via its twin.

    Args:
        state: Ingestion pipeline state.

    Returns:
        Partial state update containing the kept ``chunks`` and an updated
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
    dropped_empty = 0
    dropped_dup = 0
    for chunk in state["chunks"]:
        # Measure the BODY (breadcrumb stripped — a no-op on non-contextualized
        # table/figure/legacy chunks). Skip only when there is no content at all.
        body = chunk_body_text(chunk.text, chunk.metadata.get("heading_path"))
        if not body.strip():
            dropped_empty += 1
            continue
        # Exact-duplicate removal on the FULL embedded text, so identical bodies
        # under DIFFERENT section headings are kept as distinct chunks.
        normalized = _WHITESPACE_RE.sub(" ", chunk.text.strip()).lower()
        if normalized in seen_normalized:
            dropped_dup += 1
            continue
        seen_normalized.add(normalized)
        filtered_chunks.append(chunk)

    logger.info(
        "quality_validation complete: source=%s input=%d output=%d dropped_empty=%d dropped_dup=%d",
        state.get("source_name", ""), len(state["chunks"]), len(filtered_chunks),
        dropped_empty, dropped_dup,
    )
    logger.debug("quality_validation_node completed in %.3fs", time.monotonic() - t0)
    return {
        "chunks": filtered_chunks,
        "processing_log": append_processing_log(state, "quality_validation:ok"),
    }
