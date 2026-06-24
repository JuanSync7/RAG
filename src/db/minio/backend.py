# @summary
# MinioBackend: DocumentBackend implementation delegating to src.db.minio.store.
# Exports: MinioBackend
# Deps: src.db.backend, src.db.common.schemas, src.db.minio.store, minio, config.settings
# @end-summary
"""Minio implementation of the DocumentBackend contract."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator, Optional

from src.db.backend import DocumentBackend
from src.db.common import StoredDocument
from src.db.minio.store import (
    create_client as _mn_create_client,
    ensure_bucket as _mn_ensure_bucket,
    put_document as _mn_put_document,
    get_document as _mn_get_document,
    delete_document as _mn_delete_document,
    document_exists as _mn_document_exists,
    get_document_url as _mn_get_document_url,
    list_documents as _mn_list_documents,
)
from config.settings import (
    MINIO_BUCKET,
    RAG_MINIO_LIST_DEFAULT_LIMIT,
    RAG_MINIO_LIST_DEFAULT_OFFSET,
    RAG_MINIO_PRESIGNED_URL_EXPIRY_SECONDS,
)

logger = logging.getLogger("rag.db.minio")


class MinioBackend(DocumentBackend):
    """MinIO document store backend."""

    def _bucket(self, bucket: Optional[str]) -> str:
        return bucket or MINIO_BUCKET

    def create_persistent_client(self) -> Any:
        return _mn_create_client()

    @contextmanager
    def get_ephemeral_client(self) -> Generator[Any, None, None]:
        # MinIO client is stateless — no connection to close.
        yield _mn_create_client()

    def ensure_bucket(self, client: Any, bucket: Optional[str] = None) -> None:
        _mn_ensure_bucket(client, bucket=self._bucket(bucket))

    def put_document(
        self,
        client: Any,
        document_id: str,
        content: str,
        metadata: dict,
        bucket: Optional[str] = None,
    ) -> None:
        _mn_put_document(client, document_id, content, metadata, bucket=self._bucket(bucket))

    def get_document(
        self,
        client: Any,
        document_id: str,
        bucket: Optional[str] = None,
    ) -> Optional[StoredDocument]:
        raw = _mn_get_document(client, document_id, bucket=self._bucket(bucket))
        if raw is None:
            return None
        return StoredDocument(
            document_id=raw["document_id"],
            content=raw["content"],
            metadata=raw["metadata"],
        )

    def delete_document(
        self,
        client: Any,
        document_id: str,
        bucket: Optional[str] = None,
    ) -> bool:
        return _mn_delete_document(client, document_id, bucket=self._bucket(bucket))

    def document_exists(
        self,
        client: Any,
        document_id: str,
        bucket: Optional[str] = None,
    ) -> bool:
        return _mn_document_exists(client, document_id, bucket=self._bucket(bucket))

    def get_document_url(
        self,
        client: Any,
        document_id: str,
        bucket: Optional[str] = None,
        expires_in_seconds: int = RAG_MINIO_PRESIGNED_URL_EXPIRY_SECONDS,
    ) -> str:
        return _mn_get_document_url(
            client, document_id, bucket=self._bucket(bucket),
            expires_in_seconds=expires_in_seconds,
        )

    def list_documents(
        self,
        client: Any,
        bucket: Optional[str] = None,
        prefix: str = "",
        limit: int = RAG_MINIO_LIST_DEFAULT_LIMIT,
        offset: int = RAG_MINIO_LIST_DEFAULT_OFFSET,
    ) -> list[dict]:
        return _mn_list_documents(
            client, bucket=self._bucket(bucket), prefix=prefix, limit=limit, offset=offset
        )
