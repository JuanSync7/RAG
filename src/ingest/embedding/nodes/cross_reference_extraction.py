# @summary
# LangGraph node for optional cross-reference pattern extraction from document text.
# Exports: cross_reference_extraction_node
# Deps: embedding.state
# @end-summary

"""Cross-reference extraction node implementation."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("rag.ingest.embedding.cross_reference_extraction")

from src.ingest.common import (
    append_processing_log,
    cross_refs,
)
from src.ingest.embedding.state import EmbeddingPipelineState


def cross_reference_extraction_node(state: EmbeddingPipelineState) -> dict[str, Any]:
    """Extract document cross-reference patterns when this stage is enabled.

    Args:
        state: Ingestion pipeline state.

    Returns:
        Partial state update containing ``cross_references`` (when enabled) and
        an updated ``processing_log``. When disabled, returns only a skipped log
        entry.
    """
    t0 = time.monotonic()
    if not state["runtime"].config.enable_cross_reference_extraction:
        return {
            "processing_log": append_processing_log(
                state, "cross_reference_extraction:skipped"
            )
        }
    text = state.get("refactored_text") or state.get("cleaned_text", "")
    refs = cross_refs(text)
    logger.info("cross_reference_extraction complete: source=%s refs=%d", state.get("source_name", ""), len(refs))
    logger.debug("cross_reference_extraction_node completed in %.3fs", time.monotonic() - t0)
    return {
        "cross_references": refs,
        "processing_log": append_processing_log(
            state, "cross_reference_extraction:ok"
        ),
    }
