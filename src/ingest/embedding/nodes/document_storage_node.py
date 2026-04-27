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

from src.db import build_document_id, put_document  # noqa: F401 — put_document kept for legacy refs
from src.ingest.common import append_processing_log
from src.ingest.embedding.state import EmbeddingPipelineState


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

    content = state.get("refactored_text") or state.get("cleaned_text") or state.get("raw_text", "")
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
