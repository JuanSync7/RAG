# @summary
# Temporal activity definitions for the two-phase ingestion pipeline.
# Exports: document_processing_activity, embedding_pipeline_activity,
#          prewarm_worker_resources,
#          delete_source_activity, DeleteSourceArgs, DeleteSourceResult
# Deps: temporalio, src.ingest.doc_processing.impl, src.ingest.embedding.impl,
#       src.ingest.common.types, src.vector_db, src.db, src.core.embeddings
# @end-summary
"""Temporal activities wrapping Phase 1 (document processing) and Phase 2 (embedding).

Each activity creates its own Runtime with fresh per-document clients.
The embedder and MinIO client are module-level singletons initialised once
per worker process via ``prewarm_worker_resources()``.

The CleanDocumentStore is the durable checkpoint between the two phases —
Phase 1 writes to it, Phase 2 reads from it. No large data is passed
between activities through Temporal.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from temporalio import activity

from config.settings import (
    RAG_INGESTION_DOCLING_ENABLED,
    RAG_INGESTION_DOCLING_MODEL,
    RAG_INGESTION_DOCLING_ARTIFACTS_PATH,
    RAG_INGESTION_DOCLING_AUTO_DOWNLOAD,
)
from langchain_core.embeddings import Embeddings
from src.core.embeddings import get_embedding_provider
from src.ingest.common import (
    CleanDocumentStore,
    IngestionConfig,
    Runtime,
)
from src.ingest.doc_processing import run_document_processing
from src.ingest.embedding import run_embedding_pipeline
from src.ingest.support import ensure_docling_ready
from src.vector_db.weaviate.store import get_source_hash
import src.db as db
import src.vector_db as vector_db

logger = logging.getLogger("rag.ingest.temporal.activities")

# ---------------------------------------------------------------------------
# Worker-level singletons (initialised once per worker process)
# ---------------------------------------------------------------------------

_embedder: Optional[Embeddings] = None
_db_client: Optional[Any] = None


def prewarm_worker_resources() -> None:
    """Load the embedding model, MinIO client, and optionally Docling models before the worker starts.

    Call this once at worker startup so the first activity execution does not
    pay the model-load penalty.
    """
    global _embedder, _db_client
    # tier="ingest" routes ingest embedding to the CPU pool via rag-nginx.
    _embedder = get_embedding_provider(tier="ingest")
    _db_client = db.create_persistent_client()
    try:
        db.ensure_bucket(_db_client)
    except Exception:
        logger.exception("ensure_bucket failed at prewarm — document_storage_node will fail until resolved")
    logger.info("worker resources prewarmed: embedder + db client ready")

    if RAG_INGESTION_DOCLING_ENABLED:
        logger.info("prewarming Docling models (RAG_INGESTION_DOCLING_ENABLED=true)...")
        try:
            ensure_docling_ready(
                parser_model=RAG_INGESTION_DOCLING_MODEL,
                artifacts_path=RAG_INGESTION_DOCLING_ARTIFACTS_PATH,
                auto_download=RAG_INGESTION_DOCLING_AUTO_DOWNLOAD,
            )
            logger.info("Docling models ready")
        except RuntimeError as exc:
            logger.warning(
                "Docling is enabled but models could not be prepared — "
                "ingestion jobs using Docling will fail: %s",
                exc,
            )
    else:
        logger.info("Docling disabled (RAG_INGESTION_DOCLING_ENABLED=false) — skipping model warmup")


def _get_embedder() -> Embeddings:
    global _embedder
    if _embedder is None:
        _embedder = get_embedding_provider(tier="ingest")
    return _embedder


def _get_db_client() -> Any:
    global _db_client
    if _db_client is None:
        _db_client = db.create_persistent_client()
    return _db_client


# ---------------------------------------------------------------------------
# Activity input / output contracts
# ---------------------------------------------------------------------------

@dataclass
class SourceArgs:
    """Serializable source identity passed to both activities."""
    source_path: str
    source_name: str
    source_uri: str
    source_key: str
    source_id: str
    connector: str
    source_version: str
    existing_hash: str = ""
    existing_source_uri: str = ""


@dataclass
class ActivityArgs:
    """Full input for either ingestion activity."""
    source: SourceArgs
    config: dict[str, Any]  # IngestionConfig serialised via dataclasses.asdict()
    # Per-document UUID minted at workflow entry. Persisted on every chunk so
    # commit_node can roll back partial writes by filtering on this id.
    staging_batch_id: str = ""


@dataclass
class DocProcessingResult:
    """Output of document_processing_activity."""
    errors: list[str]
    source_hash: str
    clean_hash: str
    processing_log: list[str]
    # True when Phase 1 short-circuited because the prior ingest of this
    # source_key has the same source_hash. Workflow uses this to skip Phase 2.
    skipped: bool = False


@dataclass
class EmbeddingResult:
    """Output of embedding_pipeline_activity."""
    errors: list[str]
    stored_count: int
    metadata_summary: str
    metadata_keywords: list[str]
    processing_log: list[str]


@dataclass
class DeleteSourceArgs:
    """Input for delete_source_activity.

    Args:
        source_key: Stable source identity. Always required.
        staging_batch_id: When set, delete only chunks tagged with this batch
            (used by post-cancel cleanup of a single in-flight document).
            When empty, delete all chunks for ``source_key`` (manual purge).
    """
    source_key: str
    staging_batch_id: str = ""


@dataclass
class RecordPhaseStatusArgs:
    """Input for record_phase_status_activity (Step 10).

    Wraps :meth:`CleanDocumentStore.record_attempt` so the workflow can
    write per-phase outcomes to the durable status ledger without holding
    a Python handle to the store.
    """

    clean_store_dir: str
    source_key: str
    phase: str  # "phase2a" or "phase2b"
    success: bool
    error: Optional[str] = None
    error_class: Optional[str] = None  # "transient" | "document" | "system"
    max_attempts: int = 10


@dataclass
class ListPendingPhase2bArgs:
    """Input for list_pending_phase2b_activity (Step 11).

    Used by the backfill workflow to discover sources that need a bounded
    KG retry. ``status`` defaults to ``failed_pending_retry`` — permanent
    failures stay out so they can be triaged.
    """

    clean_store_dir: str
    status: str = "failed_pending_retry"


@dataclass
class DeleteSourceResult:
    """Output of delete_source_activity."""
    weaviate_deleted: int
    minio_deleted: bool
    errors: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deserialise_config(config_dict: dict) -> IngestionConfig:
    """Reconstruct IngestionConfig from a plain dict (Temporal payload)."""
    known = {f.name for f in dataclasses.fields(IngestionConfig)}
    return IngestionConfig(**{k: v for k, v in config_dict.items() if k in known})


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------

@activity.defn
async def document_processing_activity(args: ActivityArgs) -> DocProcessingResult:
    """Phase 1: parse, clean, and optionally refactor a document.

    Saves the clean text to CleanDocumentStore as the durable Phase 1
    checkpoint. The embedding activity reads from there — no large data
    flows through Temporal.
    """
    config = _deserialise_config(args.config)
    s = args.source

    # --- Idempotent re-ingest check ---------------------------------------
    # Cheap pre-check before the expensive parse step. Reading the raw bytes
    # and one Weaviate query is far cheaper than running Docling + embedding.
    # On any error, fall through to the full pipeline — never block re-ingest.
    try:
        raw_bytes = Path(s.source_path).read_bytes()
        precomputed_source_hash = hashlib.sha256(raw_bytes).hexdigest()
    except (OSError, ValueError):
        precomputed_source_hash = ""

    if precomputed_source_hash:
        try:
            with vector_db.get_client() as wv_client:
                prior_hash = get_source_hash(wv_client, s.source_key)
        except Exception:
            logger.debug("idempotency check failed for %s", s.source_key, exc_info=True)
            prior_hash = None

        if prior_hash and prior_hash == precomputed_source_hash:
            logger.info(
                "phase1 short-circuit: source unchanged source_key=%s hash=%s",
                s.source_key, precomputed_source_hash[:12],
            )
            return DocProcessingResult(
                errors=[],
                source_hash=precomputed_source_hash,
                clean_hash="",
                processing_log=["doc_processing:skipped:unchanged"],
                skipped=True,
            )

    # Phase 1 does not need embedder or vector DB — pass minimal Runtime.
    runtime = Runtime(
        config=config,
        embedder=_get_embedder(),   # not used in Phase 1 but required by Runtime
        weaviate_client=None,       # not needed for doc processing
        db_client=None,
    )

    result = run_document_processing(
        runtime=runtime,
        source_path=s.source_path,
        source_name=s.source_name,
        source_uri=s.source_uri,
        source_key=s.source_key,
        source_id=s.source_id,
        connector=s.connector,
        source_version=s.source_version,
        trace_id=args.staging_batch_id,
    )

    errors = list(result.get("errors", []))
    source_hash = result.get("source_hash", "")
    processing_log = list(result.get("processing_log", []))
    clean_hash = ""

    # Write clean text to CleanDocumentStore so Phase 2 can read it.
    # This is the durable boundary between activities — the workflow
    # contract requires Phase 2 to read from here, not from Temporal payload.
    if not errors and config.clean_store_dir:
        clean_text = result.get("cleaned_text", "")
        clean_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
        meta = {
            "source_key": s.source_key,
            "source_name": s.source_name,
            "source_uri": s.source_uri,
            "source_id": s.source_id,
            "connector": s.connector,
            "source_version": s.source_version,
            "source_hash": source_hash,
            "clean_hash": clean_hash,
        }
        try:
            CleanDocumentStore(Path(config.clean_store_dir)).write(s.source_key, clean_text, meta)
        except Exception as exc:
            logger.exception("CleanDocumentStore write failed for %s", s.source_key)
            errors.append(f"clean_store_write_failed: {exc}")

    return DocProcessingResult(
        errors=errors,
        source_hash=source_hash,
        clean_hash=clean_hash,
        processing_log=processing_log,
    )


@activity.defn
async def embedding_pipeline_activity(args: ActivityArgs) -> EmbeddingResult:
    """Phase 2: chunk, embed, store in Weaviate and MinIO.

    Reads clean text from CleanDocumentStore (written by Phase 1).
    Creates a fresh Weaviate client per activity execution.
    """
    config = _deserialise_config(args.config)
    s = args.source

    if not config.clean_store_dir:
        return EmbeddingResult(
            errors=["clean_store_dir is not configured — Phase 2 cannot read Phase 1 output"],
            stored_count=0, metadata_summary="", metadata_keywords=[], processing_log=[],
        )

    store = CleanDocumentStore(Path(config.clean_store_dir))
    try:
        clean_text, meta = store.read(s.source_key)
    except FileNotFoundError as exc:
        return EmbeddingResult(
            errors=[f"clean_store_read_failed: {exc}"],
            stored_count=0, metadata_summary="", metadata_keywords=[], processing_log=[],
        )
    clean_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
    source_hash = meta.get("source_hash", "") if isinstance(meta, dict) else ""

    with vector_db.get_client() as wv_client:
        vector_db.ensure_collection(wv_client, config.target_collection or None)

        runtime = Runtime(
            config=config,
            embedder=_get_embedder(),
            weaviate_client=wv_client,
            db_client=_get_db_client() if config.store_documents else None,
        )

        result = run_embedding_pipeline(
            runtime=runtime,
            source_key=s.source_key,
            source_name=s.source_name,
            source_uri=s.source_uri,
            source_id=s.source_id,
            connector=s.connector,
            source_version=s.source_version,
            clean_text=clean_text,
            clean_hash=clean_hash,
            staging_batch_id=args.staging_batch_id,
            trace_id=args.staging_batch_id,
            source_hash=source_hash,
        )

    return EmbeddingResult(
        errors=result.get("errors", []),
        stored_count=result.get("stored_count", 0),
        metadata_summary=result.get("metadata_summary", ""),
        metadata_keywords=result.get("metadata_keywords", []),
        processing_log=result.get("processing_log", []),
    )


@activity.defn
async def delete_source_activity(args: DeleteSourceArgs) -> DeleteSourceResult:
    """Best-effort delete of a source's chunks (Weaviate) and document (MinIO).

    Idempotent: zero-match deletes are a success, not a failure.
    Each store is cleaned independently — a failure in one does not block
    the other so the operator gets maximum cleanup per attempt.
    Temporal retries the workflow wrapper if the activity itself raises.
    """
    from src.vector_db.weaviate.store import delete_documents_by_staging_batch

    errors: list[str] = []
    weaviate_deleted = 0
    minio_deleted = False

    # --- Weaviate cleanup ---------------------------------------------------
    try:
        with vector_db.get_client() as wv_client:
            if args.staging_batch_id:
                weaviate_deleted = delete_documents_by_staging_batch(
                    wv_client, args.staging_batch_id,
                )
            else:
                weaviate_deleted = vector_db.delete_by_source_key(
                    wv_client, args.source_key,
                )
    except Exception as exc:
        logger.exception(
            "delete_source: Weaviate cleanup failed source_key=%s", args.source_key
        )
        errors.append(f"weaviate_delete_failed:{exc}")

    # --- MinIO cleanup -------------------------------------------------------
    try:
        from src.db import build_document_id
        document_id = build_document_id(args.source_key)
        client = _get_db_client()
        minio_deleted = bool(db.delete_document(client, document_id))
    except Exception as exc:
        logger.exception(
            "delete_source: MinIO cleanup failed source_key=%s", args.source_key
        )
        errors.append(f"minio_delete_failed:{exc}")

    return DeleteSourceResult(
        weaviate_deleted=weaviate_deleted,
        minio_deleted=minio_deleted,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Phase 2b status / backfill activities (Step 10/11)
# ---------------------------------------------------------------------------


@activity.defn
async def record_phase_status_activity(
    args: RecordPhaseStatusArgs,
) -> dict[str, Any]:
    """Persist a per-phase ingest outcome to the CleanDocumentStore ledger.

    Called by ``IngestDocumentWorkflow`` after each attempt at the KG
    activity. Returns the updated phase entry so the workflow can include
    it in its result. Failures here are local FS errors; let them bubble
    up so Temporal retries the activity.
    """
    store = CleanDocumentStore(Path(args.clean_store_dir))
    return store.record_attempt(
        args.source_key,
        phase=args.phase,
        success=args.success,
        error=args.error,
        error_class=args.error_class,
        max_attempts=args.max_attempts,
    )


@activity.defn
async def list_pending_phase2b_activity(
    args: ListPendingPhase2bArgs,
) -> list[str]:
    """Return source_keys whose ``phase2b`` status equals ``args.status``.

    Backfill workflow drains ``failed_pending_retry`` on each tick.
    Returns an empty list when the store directory does not exist (so the
    workflow is safe to schedule before the first ingest).
    """
    store_path = Path(args.clean_store_dir)
    if not store_path.exists():
        return []
    return CleanDocumentStore(store_path).list_pending(
        phase="phase2b", status=args.status,
    )
