# @summary
# LangGraph node for optional LLM-driven document refactoring.
# Exports: document_refactoring_node
# Deps: src.ingest.support.llm, src.ingest.common.shared, src.ingest.doc_processing.state
# @end-summary

"""Document-refactoring node implementation."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("rag.ingest.docproc.document_refactoring")

from src.ingest.support import _llm_json
from src.ingest.common import append_processing_log
from src.ingest.doc_processing.state import DocumentProcessingState

_MAX_REFACTOR_INPUT = 10000
_REFACTOR_MAX_TOKENS = 900
_REFACTOR_PROMPT = 'Return {"refactored_text":"..."} for:\n'


def document_refactoring_node(state: DocumentProcessingState) -> dict[str, Any]:
    """Optionally rewrite cleaned text through an LLM-based refactoring pass.

    Args:
        state: Document processing pipeline state.

    Returns:
        Partial state update containing ``refactored_text`` and an updated
        ``processing_log``. When refactoring is disabled or the LLM response is
        empty, this node passes through the cleaned text unchanged.
    """
    t0 = time.monotonic()
    config = state["runtime"].config
    if not config.enable_document_refactoring:
        return {
            "refactored_text": state["cleaned_text"],
            "processing_log": append_processing_log(
                state, "document_refactoring:skipped"
            ),
        }
    prompt = _REFACTOR_PROMPT + state["cleaned_text"][:_MAX_REFACTOR_INPUT]
    response = _llm_json(prompt, config, _REFACTOR_MAX_TOKENS)
    refactored_text = str(response.get("refactored_text", "")).strip()
    logger.info("document_refactoring complete: source=%s", state.get("source_name", ""))
    logger.debug("document_refactoring_node completed in %.3fs", time.monotonic() - t0)
    return {
        "refactored_text": refactored_text or state["cleaned_text"],
        "processing_log": append_processing_log(state, "document_refactoring:ok"),
    }
