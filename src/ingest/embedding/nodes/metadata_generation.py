# @summary
# LangGraph node for document-level summary and keyword generation with fallback extraction.
# Exports: metadata_generation_node
# Deps: embedding.state
# @end-summary

"""Metadata-generation node implementation."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("rag.ingest.embedding.metadata_generation")

from src.ingest.support import _llm_json
from src.ingest.common import (
    append_processing_log,
    extract_keywords_fallback,
)
from src.ingest.embedding.state import EmbeddingPipelineState

_MAX_TEXT_FOR_METADATA = 10000
_MAX_SUMMARY_LEN = 240


def metadata_generation_node(state: EmbeddingPipelineState) -> dict[str, Any]:
    """Generate document summary/keywords and project them into chunk metadata.

    This node uses an LLM (when enabled) to produce a short summary and a list
    of keywords for the document, falling back to deterministic extraction when
    LLM metadata is disabled or unavailable.

    Args:
        state: Ingestion pipeline state.

    Returns:
        Partial state update containing ``metadata_summary``, ``metadata_keywords``,
        and an updated ``processing_log``.
    """
    t0 = time.monotonic()
    config = state["runtime"].config
    if not config.enable_llm_metadata:
        return {
            "processing_log": append_processing_log(state, "metadata_generation:skipped"),
        }
    text = state.get("refactored_text") or state.get("cleaned_text", "")
    prompt = (
        'Return {"summary":"...","keywords":[]} for:\n'
        + text[:_MAX_TEXT_FOR_METADATA]
    )
    response = _llm_json(prompt, config, 250)
    llm_summary = str(response.get("summary", "")).strip()
    if not llm_summary:
        summary_raw = text[:_MAX_SUMMARY_LEN]
        # truncate to last word boundary to avoid cutting mid-word
        summary = summary_raw[:summary_raw.rfind(" ")].strip() if " " in summary_raw else summary_raw.strip()
    else:
        summary = llm_summary

    keywords = response.get("keywords")
    if isinstance(keywords, list):
        parsed_keywords = [str(keyword).strip() for keyword in keywords]
    else:
        parsed_keywords = extract_keywords_fallback(
            text, config.max_keywords
        )
    parsed_keywords = parsed_keywords[: config.max_keywords]
    joined_keywords = ", ".join(parsed_keywords)

    for chunk in state["chunks"]:
        chunk.metadata["document_summary"] = summary
        chunk.metadata["document_keywords"] = joined_keywords

    logger.info("metadata_generation complete: source=%s summary_len=%d keywords=%d", state.get("source_name", ""), len(summary), len(parsed_keywords))
    logger.debug("metadata_generation_node completed in %.3fs", time.monotonic() - t0)
    return {
        "chunks": state["chunks"],
        "metadata_summary": summary,
        "metadata_keywords": parsed_keywords,
        "processing_log": append_processing_log(state, "metadata_generation:ok"),
    }
