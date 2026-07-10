# @summary
# LangGraph node for embedding generation; stages records for atomic commit (Issue #42).
# Exports: embedding_storage_node, _form_batches, _embed_batches,
#   _log_batch_metrics, _log_batch_summary
# Deps: src.vector_db, src.ingest.embedding.state, src.ingest.common.shared,
#   src.ingest.common.schemas (PIPELINE_SCHEMA_VERSION)
# trace_id, schema_version, batch_id attached to every chunk payload (FR-3052, FR-3053, FR-3100).
# Chunks are embedded in configurable batches (FR-1210–FR-1214) with per-batch
# retry isolation; failed batches are excluded from output via success_mask.
# Weaviate write is deferred to commit_node for atomicity.
# @end-summary

"""Embedding-storage node implementation."""

from __future__ import annotations

import logging
import time
from typing import Any

from src.vector_db import (
    add_documents,
    delete_by_source_key,
    DocumentRecord,
)
from src.ingest.common import append_processing_log
from src.ingest.common.schemas import PIPELINE_SCHEMA_VERSION
from src.ingest.embedding.state import EmbeddingPipelineState
from src.ingest.common.observability import node_span
from config.settings import (
    RAG_INGEST_EMBEDDING_BATCH_MAX_RETRIES,
    RAG_INGEST_EMBEDDING_BATCH_RETRY_DELAY_S,
)

logger = logging.getLogger("rag.ingest.embedding.storage")


def _form_batches(items: list, batch_size: int) -> list[list]:
    """Split items into sequential batches of at most batch_size.

    Handles partial final batches without error (FR-1212).
    Returns an empty list if items is empty (FR-1212 AC-3).

    >>> _form_batches([1,2,3,4,5], 2)
    [[1, 2], [3, 4], [5]]
    >>> _form_batches([], 64)
    []
    """
    if not items:
        return []
    return [
        items[i : i + batch_size]
        for i in range(0, len(items), batch_size)
    ]


def _log_batch_metrics(
    batch_idx: int,
    total_batches: int,
    chunk_count: int,
    latency_ms: float,
) -> None:
    """Log per-batch embedding metrics (FR-1214 AC-1)."""
    logger.info(
        "embedding_batch_complete",
        extra={
            "batch_index": batch_idx,
            "total_batches": total_batches,
            "chunk_count": chunk_count,
            "latency_ms": round(latency_ms, 1),
        },
    )


def _log_batch_summary(
    total_chunks: int,
    total_batches: int,
    total_ms: float,
) -> None:
    """Log aggregate embedding throughput (FR-1214 AC-2)."""
    throughput = (total_chunks / (total_ms / 1000.0)) if total_ms > 0 else 0.0
    logger.info(
        "embedding_batch_summary",
        extra={
            "total_chunks": total_chunks,
            "total_batches": total_batches,
            "total_ms": round(total_ms, 1),
            "throughput_chunks_per_sec": round(throughput, 2),
        },
    )


def _embed_batches(
    embedder,
    text_batches: list[list[str]],
    max_retries: int = RAG_INGEST_EMBEDDING_BATCH_MAX_RETRIES,
) -> tuple[list[list[float]], list[dict[str, Any]], list[bool]]:
    """Embed text batches with per-batch retry isolation.

    Returns:
        (all_vectors, errors, success_mask) where all_vectors is the flat list of
        embedding vectors for successfully embedded batches, errors is a list of
        error dicts for failed batches, and success_mask[i] is True when batch i
        succeeded.

    Successfully embedded batches are NOT re-embedded during retry (FR-1213 AC-1).
    Failed batch chunks are excluded from output (FR-1213 AC-4).
    """
    all_vectors: list[list[float]] = []
    errors: list[dict[str, Any]] = []
    success_mask: list[bool] = [False] * len(text_batches)
    total_embedded = 0
    total_elapsed_ms = 0.0

    for batch_idx, batch_texts in enumerate(text_batches):
        batch_vectors = None
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                start_time = time.monotonic()
                batch_vectors = embedder.embed_documents(batch_texts)
                elapsed_ms = (time.monotonic() - start_time) * 1000
                _log_batch_metrics(batch_idx + 1, len(text_batches), len(batch_texts), elapsed_ms)
                total_elapsed_ms += elapsed_ms
                break
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "batch %d/%d failed attempt %d/%d: %s",
                    batch_idx + 1, len(text_batches), attempt, max_retries, exc,
                )
                if attempt < max_retries:
                    time.sleep(RAG_INGEST_EMBEDDING_BATCH_RETRY_DELAY_S * attempt)  # seconds; kept short to limit event-loop blocking in async paths

        if batch_vectors is not None:
            all_vectors.extend(batch_vectors)
            success_mask[batch_idx] = True
            total_embedded += len(batch_texts)
        else:
            # Batch exhausted retries (FR-1213 AC-3)
            chunk_start = sum(len(b) for b in text_batches[:batch_idx])
            chunk_end = chunk_start + len(batch_texts)
            errors.append({
                "type": "batch_embedding_failure",
                "batch_index": batch_idx + 1,
                "chunk_range": f"{chunk_start}-{chunk_end - 1}",
                "error": str(last_error),
            })
            logger.error(
                "batch %d/%d exhausted retries batch_index=%d chunk_range=%d-%d",
                batch_idx + 1, len(text_batches),
                batch_idx + 1, chunk_start, chunk_end - 1,
            )

    _log_batch_summary(total_embedded, len(text_batches), total_elapsed_ms)
    return all_vectors, errors, success_mask


def _format_batch_error(err: dict[str, Any]) -> str:
    """Render a batch-failure dict as a single string.

    The pipeline state ``errors`` field is ``list[str]`` and is forwarded
    verbatim into ``EmbeddingResult.errors`` (also ``list[str]``), which crosses
    the Temporal activity->workflow boundary. Temporal decodes that result
    against the dataclass type hint, so a ``dict`` element raises
    ``TypeError: Failed converting field errors`` and fails the workflow task
    (which then retries forever). Stringifying here keeps the structured
    information (batch index, chunk range, cause) while honouring the contract.
    """
    return (
        f"batch_embedding_failure:batch={err.get('batch_index')}"
        f" chunk_range={err.get('chunk_range')}: {err.get('error')}"
    )


@node_span("embedding_storage")
def embedding_storage_node(state: EmbeddingPipelineState) -> dict[str, Any]:  # noqa: D401
    from src.platform.observability import get_tracer

    tracer = get_tracer()
    with tracer.span(
        "ingest.embedding.store",
        {
            "chunk_count": len(state.get("chunks") or []),
            "source_key": state.get("source_key", ""),
            "batch_id": state.get("batch_id", ""),
        },
    ):
        return _embedding_storage_node_impl(state)


def _embedding_storage_node_impl(state: EmbeddingPipelineState) -> dict[str, Any]:
    """Generate chunk embeddings and STAGE records for atomic commit (Issue #42).

    The actual Weaviate write (``add_documents`` / ``delete_by_source_key``) is
    deferred to ``commit_node`` so that MinIO, Weaviate, and the KG are all
    flushed in one terminal phase.

    Args:
        state: Ingestion pipeline state.

    Returns:
        Partial state update containing ``staged_weaviate_records``,
        ``staged_weaviate_delete_old``, ``stored_count=0`` (real count set by
        commit_node), and an updated ``processing_log``.
        When the workflow is skipped or there are no chunks, returns
        empty staging fields.
    """
    t0 = time.monotonic()
    if state.get("should_skip", False) or not state["chunks"]:
        return {
            "stored_count": 0,
            "staged_weaviate_records": [],
            "staged_weaviate_delete_old": False,
            "processing_log": append_processing_log(state, "embedding_storage:skipped"),
        }

    runtime = state["runtime"]
    try:
        # Attach trace/schema/batch metadata to every chunk (FR-3052, FR-3053, FR-3100).
        # These three keys are merged into a copy of each chunk's metadata so that
        # existing keys are preserved and the original chunk objects are not mutated.
        lifecycle_meta = {
            "trace_id": state.get("trace_id", ""),
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "batch_id": state.get("batch_id", ""),
            "staging_batch_id": state.get("staging_batch_id", ""),
            "source_hash": state.get("source_hash", ""),
        }

        chunks = state["chunks"]
        # Decouple STORED text from EMBEDDED text. The stored text (returned to
        # generation + citation) is the heading-enriched chunk body. The embedded
        # text is the contextual-chunking ``embed_text`` (a per-chunk situating
        # prefix + the body) WHEN a contextual-enrichment stage attached it, so
        # contextual retrieval improves the vector without polluting the text the
        # model reads (Anthropic-style contextual retrieval). Backward-compatible:
        # no ``embed_text`` -> embedded == stored (today's behavior).
        store_texts = [chunk.metadata.get("enriched_content", chunk.text) for chunk in chunks]
        embed_texts = [
            chunk.metadata.get("embed_text")
            or chunk.metadata.get("enriched_content", chunk.text)
            for chunk in chunks
        ]
        batch_size = runtime.config.embedding_batch_size
        text_batches = _form_batches(embed_texts, batch_size)

        all_vectors, batch_errors, success_mask = _embed_batches(runtime.embedder, text_batches)

        # Build success index: determine which chunk indices succeeded.
        # success_mask[i] corresponds to text_batches[i]; map back to flat chunk indices.
        successful_chunk_indices: list[int] = []
        offset = 0
        for i, batch in enumerate(text_batches):
            if success_mask[i]:
                successful_chunk_indices.extend(range(offset, offset + len(batch)))
            offset += len(batch)

        records = [
            DocumentRecord(
                text=store_texts[idx],
                embedding=all_vectors[pos],
                metadata={**chunks[idx].metadata, **lifecycle_meta},
            )
            for pos, idx in enumerate(successful_chunk_indices)
        ]
    except Exception as exc:
        logger.error("embedding_storage failed source=%s: %s", state.get("source_key", ""), exc, exc_info=True)
        return {
            "errors": state.get("errors", []) + [f"embedding_storage:{exc}"],
            "staged_weaviate_records": [],
            "staged_weaviate_delete_old": False,
            "processing_log": append_processing_log(state, "embedding_storage:error"),
        }

    existing_errors = state.get("errors", [])
    logger.debug("embedding_storage_node staged %d records in %.3fs", len(records), time.monotonic() - t0)
    return {
        "stored_count": 0,  # not stored yet — commit_node sets the real count
        # Stringify batch failures: state["errors"] is list[str] and is forwarded
        # into EmbeddingResult.errors (list[str]) which must decode across the
        # Temporal activity->workflow boundary (dict elements would not).
        "errors": existing_errors + [_format_batch_error(e) for e in batch_errors],
        "staged_weaviate_records": records,
        "staged_weaviate_delete_old": bool(runtime.config.update_mode),
        "processing_log": append_processing_log(state, "embedding_storage:staged"),
    }
