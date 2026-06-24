# @summary
# LangGraph node that computes document_id and stages MinIO write into pipeline state.
# Write is deferred to commit_node for atomic commit (Issue #42).
# Exports: document_storage_node
# Deps: src.db, src.ingest.embedding.state, src.ingest.common.shared
# @end-summary
"""Document-storage node — stages the MinIO write; defers execution to commit_node."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("rag.ingest.embedding.document_storage")

from src.db import build_document_id
from src.ingest.common import append_processing_log
from src.ingest.embedding.state import EmbeddingPipelineState
from src.ingest.common.observability import node_span


@node_span("document_storage")
def document_storage_node(state: EmbeddingPipelineState) -> dict[str, Any]:
    """Compute a stable document_id and STAGE the MinIO write into pipeline state.

    The actual write is deferred to ``commit_node`` so that MinIO, Weaviate, and
    the KG are all flushed atomically in one terminal phase (Issue #42).

    Skips staging (sets ``staged_minio: None``) when
    ``runtime.config.store_documents`` is False or no db_client is available.

    Args:
        state: Embedding pipeline state.

    Returns:
        Partial state update containing ``document_id``, ``staged_minio``, and
        an updated ``processing_log``.
    """
    t0 = time.monotonic()
    document_id = build_document_id(state["source_key"])
    runtime = state["runtime"]

    if not runtime.config.store_documents or runtime.db_client is None:
        reason = "skipped" if not runtime.config.store_documents else "no_client"
        logger.debug("document_storage_node %s in %.3fs", reason, time.monotonic() - t0)
        return {
            "document_id": document_id,
            "staged_minio": None,
            "processing_log": append_processing_log(state, f"document_storage:{reason}"),
        }

    # Persist the GOLDEN source the chunks are derived from: the parsed Docling
    # markdown. On the native path the chunks come straight from this parse
    # (uncleaned), so storing cleaned_text would diverge the stored doc from the
    # retrieved chunks — and from what lossless_verification checks against.
    # Storing the parse makes object storage the true golden. Falls back to
    # cleaned_text / raw_text when no parse is present (legacy path).
    parse_result = state.get("parse_result")
    parsed_markdown = (
        getattr(parse_result, "text_markdown", "") if parse_result is not None else ""
    )
    content = (
        parsed_markdown
        if parsed_markdown and parsed_markdown.strip()
        else (state.get("cleaned_text") or state.get("raw_text", ""))
    )
    metadata = {
        "source_key": state["source_key"],
        "source_name": state["source_name"],
        "source_uri": state["source_uri"],
        "source_id": state["source_id"],
        "source_version": state["source_version"],
        "connector": state["connector"],
    }

    logger.debug("document_storage_node staged doc_id=%s in %.3fs", document_id, time.monotonic() - t0)
    return {
        "document_id": document_id,
        "staged_minio": {
            "document_id": document_id,
            "content": content,
            "metadata": metadata,
            "bucket": runtime.config.target_bucket or None,
        },
        "processing_log": append_processing_log(state, "document_storage:staged"),
    }
