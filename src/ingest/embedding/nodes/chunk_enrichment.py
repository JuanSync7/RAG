# @summary
# LangGraph node for chunk ID assignment and enriched content projection.
# Exports: chunk_enrichment_node
# Deps: src.vector_db, src.ingest.embedding.state, src.ingest.common.shared
# @end-summary

"""Chunk-enrichment node implementation."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("rag.ingest.embedding.chunk_enrichment")

from src.vector_db import build_chunk_id
from src.ingest.common import (
    EditLog,
    append_processing_log,
    map_chunk_provenance,
    chunk_body_text,
)
from src.ingest.embedding.state import EmbeddingPipelineState
from src.ingest.common.observability import node_span


@node_span("chunk_enrichment")
def chunk_enrichment_node(state: EmbeddingPipelineState) -> dict[str, Any]:
    """Attach stable chunk IDs and enriched content fields to chunk metadata.

    This node assigns a deterministic chunk ID, adds retrieval/citation fields,
    and attempts to map each chunk back to source offsets for provenance.

    Args:
        state: Ingestion pipeline state.

    Returns:
        Partial state update containing the enriched ``chunks`` list and an
        updated ``processing_log``.
    """
    t0 = time.monotonic()
    original_text = state.get("raw_text", "")
    refactored_text = state.get("cleaned_text") or state.get("raw_text", "")
    origin_label = "original"
    # Build the edit log once per document so each chunk's offsets can be
    # projected back to the raw_text coordinate system. After refactoring was
    # removed (PR1), this maps cleaned_text positions → raw_text positions.
    edit_log = EditLog.from_diff(original_text, refactored_text)
    original_cursor = 0
    refactored_cursor = 0
    for index, chunk in enumerate(state["chunks"]):
        # Map provenance against the BODY, not the contextualized embedded text.
        # HybridChunker output has the heading breadcrumb prepended (contextualize()),
        # and that breadcrumb is NOT contiguous in the source — so passing chunk.text
        # makes every _locate_span().find() miss and falls through to the O(doc)
        # _best_paragraph_span difflib fuzzy match for EVERY chunk (observed ~22 min
        # for a 1,690-chunk spec; ~75 min projected for a 5,582-chunk one). The raw
        # body IS contiguous in source, so exact find hits and the fuzzy path is
        # avoided. No-op for non-contextualized chunks (tables/figures/legacy path).
        body_for_provenance = chunk_body_text(
            chunk.text, chunk.metadata.get("heading_path")
        )
        provenance, original_cursor, refactored_cursor = map_chunk_provenance(
            body_for_provenance,
            original_text=original_text,
            refactored_text=refactored_text,
            original_cursor=original_cursor,
            refactored_cursor=refactored_cursor,
            edit_log=edit_log,
        )
        # Keep source display/filter field explicit even if upstream metadata changes.
        chunk.metadata["source"] = state["source_name"]
        chunk.metadata["source_key"] = state["source_key"]
        chunk.metadata["chunk_id"] = build_chunk_id(state["source_key"], index, chunk.text)
        chunk.metadata["enriched_content"] = chunk.text
        chunk.metadata["retrieval_text_origin"] = origin_label
        chunk.metadata["citation_source_uri"] = state["source_uri"]
        chunk.metadata["document_id"] = state.get("document_id", "")
        chunk.metadata.update(provenance)
    logger.info("chunk_enrichment complete: source=%s chunks=%d", state["source_name"], len(state["chunks"]))
    logger.debug("chunk_enrichment_node completed in %.3fs", time.monotonic() - t0)
    return {
        "chunks": state["chunks"],
        "processing_log": append_processing_log(state, "chunk_enrichment:ok"),
    }
