# @summary
# Public API for the db subsystem: config-driven backend dispatcher and document store functions.
# Exports: create_persistent_client, get_client, close_client, ensure_bucket,
#          put_document, get_document, delete_document, document_exists, get_document_url,
#          list_documents, build_document_id, StoredDocument
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
    # Re-exported schemas and utilities
    "StoredDocument",
    "build_document_id",
]
