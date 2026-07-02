# @summary
# Public API for the db subsystem: config-driven backend dispatcher, document
# store functions, and the shared clean-document resolver
# (resolve_clean_document: document_id -> build_document_id(source_key) ->
# build_document_id(source) fallback chain across both MinIO layouts;
# library semantics — returns None, never raises).
# Exports: create_persistent_client, get_client, close_client, ensure_bucket,
#          put_document, get_document, delete_document, document_exists, get_document_url,
#          list_documents, resolve_clean_document, build_document_id, StoredDocument
# Deps: config.settings, src.db.backend, src.db.common.schemas, src.db.minio.store
# @end-summary
"""Public API for the document store subsystem (MinIO and future backends).

All pipeline code imports only from this module. Backend selection is
controlled by the ``DATABASE_BACKEND`` config key.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator, Optional

from config.settings import (
    RAG_MINIO_LIST_DEFAULT_LIMIT,
    RAG_MINIO_LIST_DEFAULT_OFFSET,
    RAG_MINIO_PRESIGNED_URL_EXPIRY_SECONDS,
)
from src.db.backend import DocumentBackend
from src.db.common import StoredDocument
from src.db.minio import build_document_id

logger = logging.getLogger("rag.db")

_db_backend: DocumentBackend | None = None


def _get_db_backend() -> DocumentBackend:
    """Return the process-wide document store backend singleton."""
    global _db_backend
    if _db_backend is None:
        from config.settings import DATABASE_BACKEND
        if DATABASE_BACKEND == "minio":
            from src.db.minio import MinioBackend
            _db_backend = MinioBackend()
        else:
            raise ValueError(
                f"Unknown DATABASE_BACKEND: {DATABASE_BACKEND!r}. "
                "Valid values: 'minio'."
            )
    return _db_backend


# ---------------------------------------------------------------------------
# Client lifecycle
# ---------------------------------------------------------------------------

def create_persistent_client() -> Any:
    """Create a long-lived document store client."""
    return _get_db_backend().create_persistent_client()


@contextmanager
def get_client() -> Generator[Any, None, None]:
    """Context manager for a short-lived document store client."""
    with _get_db_backend().get_ephemeral_client() as client:
        yield client


def close_client(client: Any) -> None:
    """Close a persistent client (no-op for stateless backends)."""
    _get_db_backend().close_client(client)


# ---------------------------------------------------------------------------
# Bucket management
# ---------------------------------------------------------------------------

def ensure_bucket(client: Any, bucket: Optional[str] = None) -> None:
    """Create the storage bucket if it does not exist (idempotent)."""
    _get_db_backend().ensure_bucket(client, bucket)


# ---------------------------------------------------------------------------
# Document operations
# ---------------------------------------------------------------------------

def put_document(
    client: Any,
    document_id: str,
    content: str,
    metadata: dict,
    bucket: Optional[str] = None,
) -> None:
    """Store or overwrite a document.

    Args:
        client: Document store client handle.
        document_id: Stable UUID for this document (use ``build_document_id()``).
        content: Full text content (clean markdown).
        metadata: Document metadata dict (source_key, source_uri, connector, etc.).
        bucket: Target bucket name. ``None`` uses the default.
    """
    _get_db_backend().put_document(client, document_id, content, metadata, bucket)


def get_document(
    client: Any,
    document_id: str,
    bucket: Optional[str] = None,
) -> Optional[StoredDocument]:
    """Fetch a document by ID.

    Returns:
        ``StoredDocument`` if found, ``None`` otherwise.
    """
    return _get_db_backend().get_document(client, document_id, bucket)


def delete_document(
    client: Any,
    document_id: str,
    bucket: Optional[str] = None,
) -> bool:
    """Remove a document. Returns True if it existed."""
    return _get_db_backend().delete_document(client, document_id, bucket)


def document_exists(
    client: Any,
    document_id: str,
    bucket: Optional[str] = None,
) -> bool:
    """Check whether a document exists without fetching its content."""
    return _get_db_backend().document_exists(client, document_id, bucket)


def get_document_url(
    client: Any,
    document_id: str,
    bucket: Optional[str] = None,
    expires_in_seconds: int = RAG_MINIO_PRESIGNED_URL_EXPIRY_SECONDS,
) -> str:
    """Return a presigned URL for direct download of the document content."""
    return _get_db_backend().get_document_url(client, document_id, bucket, expires_in_seconds)


def list_documents(
    client: Any,
    bucket: Optional[str] = None,
    prefix: str = "",
    limit: int = RAG_MINIO_LIST_DEFAULT_LIMIT,
    offset: int = RAG_MINIO_LIST_DEFAULT_OFFSET,
) -> list[dict]:
    """List documents from the document store.

    Returns:
        List of dicts with document_id, source_key, size_bytes, last_modified.

    Raises:
        S3Error: if the store is unreachable.
    """
    return _get_db_backend().list_documents(client, bucket, prefix, limit, offset)


# ---------------------------------------------------------------------------
# Clean-document resolution (shared by console viewing + turn-loop deep study)
# ---------------------------------------------------------------------------

def resolve_clean_document(
    client: Any,
    document_id: Optional[str] = None,
    source_key: Optional[str] = None,
    source: Optional[str] = None,
    bucket: Optional[str] = None,
) -> Optional[StoredDocument]:
    """Resolve a document's clean markdown across ids and MinIO layouts.

    Library-semantics resolver shared by the console document viewer
    (``server/console/services.py``) and the turn loop's DEEP_STUDY fetch
    (TURN_LOOP_DESIGN.md §5): it returns ``None`` on any miss or storage
    error and **never raises** — HTTP semantics belong to the callers.

    Identity fallback chain (first hit wins, duplicates deduped):

    1. ``document_id`` — used verbatim.
    2. ``build_document_id(source_key)`` — the id normal ingest
       (``commit_node``) writes for every document.
    3. ``build_document_id(source)`` — last resort when only a
       human-readable source name is known.

    Layout fallback (per the console viewer's dual-layout logic):

    1. **Document store** (``<document_id>.md``) — written by normal ingest
       for all formats; tried for every candidate id above.
    2. **MinioCleanStore** (``clean/{safe_key}.md``) — only populated by
       lifecycle tooling (migration/sync); tried with ``source_key`` then
       ``source`` as the store key.

    Args:
        client: Document store client handle (e.g. ``create_persistent_client()``).
        document_id: Optional stable document UUID.
        source_key: Optional stable source identity key.
        source: Optional human-readable source name.
        bucket: Target bucket; ``None`` uses the configured default.

    Returns:
        ``StoredDocument`` with non-empty content, or ``None`` when the
        document cannot be resolved from any identifier/layout.
    """
    # Ordered candidate ids, deduped with order preserved.
    candidates: list[str] = []
    for candidate in (
        document_id,
        build_document_id(source_key) if source_key else None,
        build_document_id(source) if source else None,
    ):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        try:
            doc = get_document(client, candidate, bucket)
        except Exception:  # noqa: BLE001 — library semantics: miss, not raise
            logger.warning(
                "resolve_clean_document: document-store read failed "
                "(document_id=%s)", candidate, exc_info=True,
            )
            continue
        if doc is not None and getattr(doc, "content", None):
            return doc

    # Fallback: lifecycle-populated MinioCleanStore (clean/ prefix), keyed by
    # source identity rather than document UUID.
    keys = [k for k in (source_key, source) if k]
    if not keys:
        return None
    try:
        from config.settings import MINIO_BUCKET
        from src.ingest.common.minio_clean_store import MinioCleanStore
    except Exception:  # noqa: BLE001 — optional layout; absence is a miss
        logger.warning("resolve_clean_document: clean-store import failed", exc_info=True)
        return None
    store = MinioCleanStore(client, bucket or MINIO_BUCKET)
    for key in keys:
        try:
            if store.exists(key):
                text, meta = store.read(key)
                return StoredDocument(
                    document_id=build_document_id(key),
                    content=text,
                    metadata=dict(meta or {}),
                )
        except Exception:  # noqa: BLE001 — library semantics: miss, not raise
            logger.warning(
                "resolve_clean_document: clean-store read failed (source_key=%s)",
                key, exc_info=True,
            )
    return None


__all__ = [
    # Client lifecycle
    "create_persistent_client",
    "get_client",
    "close_client",
    # Bucket management
    "ensure_bucket",
    # Document operations
    "put_document",
    "get_document",
    "delete_document",
    "document_exists",
    "get_document_url",
    "list_documents",
    # Clean-document resolution
    "resolve_clean_document",
    # Re-exported schemas and utilities
    "StoredDocument",
    "build_document_id",
]
