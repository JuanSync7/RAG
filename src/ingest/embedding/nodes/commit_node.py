# @summary
# Terminal LangGraph node that atomically commits staged writes (MinIO, Weaviate).
# On any failure, rolls back Weaviate via delete_documents_by_staging_batch and
# MinIO via delete_document. KG ingest is owned by KGWeave and dispatched via
# Temporal Phase 2b (KG_TASK_QUEUE) — no in-process KG writes happen here.
# Exports: commit_node, _rollback
# Deps: src.db, src.vector_db, src.ingest.embedding.state, src.ingest.common
# @end-summary
"""Terminal commit node — flushes staged MinIO + Weaviate writes atomically."""

from __future__ import annotations

import logging
import time
from typing import Any

from src.db import put_document, delete_document
from src.vector_db import add_documents, delete_by_source_key
from src.vector_db.weaviate.store import delete_documents_by_staging_batch
from src.ingest.common import append_processing_log
from src.ingest.embedding.state import EmbeddingPipelineState
from src.ingest.common.observability import node_span

logger = logging.getLogger("rag.ingest.embedding.commit")


def _rollback(
    runtime,
    source_key: str,
    staging_batch_id: str,
    minio_committed: bool,
    weaviate_attempted: bool,
    document_id: str,
    bucket,
) -> None:
    """Best-effort rollback of partially-committed writes.

    Each store is cleaned independently; failures are logged but do not raise so
    the original commit error surfaces cleanly to the caller.

    Args:
        runtime: Shared runtime dependencies.
        source_key: Source identity key (for logging).
        staging_batch_id: Batch ID used to target Weaviate chunks for deletion.
        minio_committed: Whether MinIO write completed before the failure.
        weaviate_attempted: Whether the Weaviate write phase was entered (even if it raised).
        document_id: Document UUID to remove from MinIO (only when minio_committed).
        bucket: MinIO bucket name; None uses the default.
    """
    if weaviate_attempted and runtime.weaviate_client is not None and staging_batch_id:
        try:
            removed = delete_documents_by_staging_batch(
                runtime.weaviate_client,
                staging_batch_id,
                collection=runtime.config.target_collection or None,
            )
            logger.warning(
                "rollback: removed %d Weaviate chunks for batch=%s",
                removed,
                staging_batch_id,
            )
        except Exception:
            logger.exception(
                "rollback: Weaviate cleanup failed batch=%s", staging_batch_id
            )

    if minio_committed and runtime.db_client is not None and document_id:
        try:
            delete_document(runtime.db_client, document_id, bucket)
            logger.warning("rollback: removed MinIO document_id=%s", document_id)
        except Exception:
            logger.exception(
                "rollback: MinIO cleanup failed document_id=%s", document_id
            )


@node_span("commit")
def commit_node(state: EmbeddingPipelineState) -> dict[str, Any]:
    """Flush staged MinIO + Weaviate + KG writes; rollback all on any failure.

    Skips immediately when ``state["errors"]`` is non-empty (upstream failure).

    Args:
        state: Embedding pipeline state with populated ``staged_*`` fields.

    Returns:
        Partial state update with ``stored_count`` and an updated
        ``processing_log``. On failure, includes ``errors`` and sets
        ``stored_count=0``.
    """
    t0 = time.monotonic()

    if state.get("errors"):
        return {
            "processing_log": append_processing_log(
                state, "commit:skipped:upstream_errors"
            ),
        }

    if state.get("should_skip", False):
        return {
            "processing_log": append_processing_log(state, "commit:skipped"),
        }

    runtime = state["runtime"]
    source_key = state.get("source_key", "")
    staging_batch_id = state.get("staging_batch_id", "")
    document_id = state.get("document_id", "")

    staged_minio = state.get("staged_minio")
    staged_records = state.get("staged_weaviate_records", []) or []
    delete_old = bool(state.get("staged_weaviate_delete_old", False))

    bucket = runtime.config.target_bucket or None
    collection = runtime.config.target_collection or None
    minio_committed = False
    weaviate_attempted = False  # True once we begin the Weaviate write phase
    stored_count = 0

    try:
        # 1) MinIO — must succeed before Weaviate so MinIO is the rollback anchor.
        if staged_minio and runtime.db_client is not None:
            put_document(
                runtime.db_client,
                staged_minio["document_id"],
                staged_minio["content"],
                staged_minio["metadata"],
                staged_minio.get("bucket") or bucket,
            )
            minio_committed = True

        # 2) Weaviate
        if staged_records and runtime.weaviate_client is not None:
            weaviate_attempted = True  # mark before write so rollback fires on failure
            if delete_old:
                delete_by_source_key(
                    runtime.weaviate_client,
                    source_key,
                    legacy_source=state.get("source_name"),
                )
            stored_count = add_documents(
                runtime.weaviate_client,
                staged_records,
                collection=collection,
            )

        # KG ingest happens out-of-process: IngestDocumentWorkflow dispatches
        # KG_PHASE2B_ACTIVITY on KG_TASK_QUEUE after embedding succeeds.

    except Exception as exc:
        logger.error(
            "commit failed source=%s batch=%s: %s",
            source_key,
            staging_batch_id,
            exc,
            exc_info=True,
        )
        _rollback(
            runtime,
            source_key,
            staging_batch_id,
            minio_committed,
            weaviate_attempted,
            document_id,
            bucket,
        )
        return {
            "stored_count": 0,
            "errors": (state.get("errors") or []) + [f"commit:{exc}"],
            "processing_log": append_processing_log(state, "commit:rolled_back"),
        }

    logger.debug(
        "commit_node completed in %.3fs stored=%d", time.monotonic() - t0, stored_count
    )
    return {
        "stored_count": stored_count,
        "processing_log": append_processing_log(state, "commit:ok"),
    }
