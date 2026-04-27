# @summary
# LangGraph node for markdown-aware text normalization and figure note injection.
# Exports: text_cleaning_node
# Deps: src.ingest.support.markdown, src.ingest.common.shared, src.ingest.doc_processing.state
# @end-summary

"""Text-cleaning node implementation."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("rag.ingest.docproc.text_cleaning")

from src.ingest.support import clean_document
from src.ingest.common import append_processing_log
from src.ingest.doc_processing.state import DocumentProcessingState

_FIGURE_NOTES_HEADER = "\n\n## Figure Notes\n"


def text_cleaning_node(state: DocumentProcessingState) -> dict[str, Any]:
    """Normalize source text and append generated multimodal notes.

    Args:
        state: Document processing pipeline state.

    Returns:
        Partial state update containing ``cleaned_text`` and an updated
        ``processing_log``.
    """
    t0 = time.monotonic()
    cleaned = clean_document(state["raw_text"])
    if state.get("multimodal_notes"):
        cleaned += _FIGURE_NOTES_HEADER + "\n".join(
            f"- {note}" for note in state["multimodal_notes"]
        )
    logger.info("text_cleaning complete: source=%s cleaned_len=%d", state.get("source_name", ""), len(cleaned))
    logger.debug("text_cleaning_node completed in %.3fs", time.monotonic() - t0)
    return {
        "cleaned_text": cleaned,
        "processing_log": append_processing_log(state, "text_cleaning:ok"),
    }
