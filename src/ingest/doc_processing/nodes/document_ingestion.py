# @summary
# LangGraph node for source file read and SHA-256 hash computation (Phase 1).
# Exports: document_ingestion_node
# Deps: src.ingest.common.utils.sha256_bytes, src.ingest.common.utils.decode_with_fallbacks, src.ingest.common.shared, src.ingest.doc_processing.state
# @end-summary

"""Document ingestion node — Phase 1."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("rag.ingest.docproc.document_ingestion")

from src.ingest.common import append_processing_log
from src.ingest.common.utils import decode_with_fallbacks, sha256_bytes
from src.ingest.doc_processing.state import DocumentProcessingState


def document_ingestion_node(state: DocumentProcessingState) -> dict[str, Any]:
    """Read source content and compute SHA-256 hash.

    The idempotency skip check is handled by the orchestrator before this
    node is invoked — this node does not read or write ``should_skip``.

    Reads the file bytes once, computing the hash and decoding text in a
    single pass to avoid redundant I/O.

    Args:
        state: Document processing pipeline state.

    Returns:
        Partial state update with ``raw_text``, ``source_hash``, and
        ``processing_log``. On read failure, returns an ``errors`` payload
        to short-circuit the workflow.
    """
    t0 = time.monotonic()
    source_path = Path(state["source_path"])

    # Use pre-read bytes when available (supplied by ingest_directory to avoid
    # a double disk read). Fall back to reading from disk for direct callers.
    raw_bytes = state.get("raw_bytes")
    if raw_bytes is None:
        try:
            raw_bytes = source_path.read_bytes()
        except Exception as exc:
            return {
                "errors": [f"read_failed:{source_path.name}:{exc}"],
                "processing_log": append_processing_log(state, "document_ingestion:failed"),
            }

    source_hash = sha256_bytes(raw_bytes)

    # Decode with fallbacks (delegates to decode_with_fallbacks for consistency)
    raw_text = decode_with_fallbacks(raw_bytes)

    logger.info("document_ingestion complete: source=%s hash=%s len=%d", source_path.name, source_hash, len(raw_text))
    logger.debug("document_ingestion completed in %.3fs", time.monotonic() - t0)
    return {
        "raw_text": raw_text,
        "source_hash": source_hash,
        "processing_log": append_processing_log(state, "document_ingestion:ok"),
    }
