# @summary
# VectorBackend ABC: formal swappable backend contract for all vector store implementations.
# Exports: VectorBackend
# Deps: abc, contextlib, typing, src.vector_db.common.schemas
# Visual collection methods: ensure_visual_collection, add_visual_documents,
#   delete_visual_by_source_key, search_visual (FR-502, FR-506, FR-507, FR-313, NFR-909)
# @end-summary
"""VectorBackend — abstract base class for all vector store backends.

Defines the formal contract between the ingestion/retrieval pipelines and
any vector store implementation. All collection-scoped operations accept an
optional ``collection`` parameter; passing ``None`` uses the backend's
configured default. New backends implement the abstract methods;
``close_client`` is overridden only when the backend requires explicit teardown.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Generator, Optional

from src.vector_db.common import (
    DocumentRecord,
    SearchFilter,
    SearchResult,
)


class VectorBackend(ABC):
    """Abstract contract for a vector store backend.

    Callers interact only through the public API in ``vector_db/__init__.py``,
    never with backend instances directly. Swapping backends requires only a
    change to the ``VECTOR_DB_BACKEND`` config key.
    """

    @abstractmethod
    def create_persistent_client(self) -> Any:
        """Create a long-lived client.

        Caller is responsible for calling ``close_client()`` on shutdown.
        Preferred for server/worker processes that serve many queries.
        """
        ...

    @abstractmethod
    @contextmanager
    def get_ephemeral_client(self) -> Generator[Any, None, None]:
        """Context manager for a short-lived client.

        Opens and closes the client per use — suitable for CLI and batch
        scripts. For server use, prefer ``create_persistent_client()``.
        """
        ...

    @abstractmethod
    def ensure_collection(self, client: Any, collection: Optional[str] = None) -> None:
        """Create the collection / index if it does not exist (idempotent)."""
        ...

    @abstractmethod
    def collection_exists(
        self, client: Any, collection: Optional[str] = None
    ) -> bool:
        """Return True iff the named collection exists on the server."""
        ...

    @abstractmethod
    def add_documents(
        self,
        client: Any,
        documents: list[DocumentRecord],
        collection: Optional[str] = None,
    ) -> int:
        """Insert documents with pre-computed embeddings.

        Returns:
            Number of documents added.
        """
        ...

    @abstractmethod
    def update_chunk_content(
        self,
        client: Any,
        chunk_uuid: str,
        *,
        text: str,
        content_hash: str,
        fuzzy_fingerprint: Optional[str] = None,
        collection: Optional[str] = None,
    ) -> bool:
        """Replace a canonical chunk's text + dedup metadata (FR-3432).

        Used by the cross-document dedup node when an incoming chunk is richer
        than the existing canonical chunk it fuzzy-matches.

        Returns:
            True on success, False on error (caller treats as non-fatal).
        """
        ...

    @abstractmethod
    def search(
        self,
        client: Any,
        query: str,
        query_embedding: list[float],
        alpha: float,
        limit: int,
        filters: Optional[list[SearchFilter]] = None,
        collection: Optional[str] = None,
    ) -> list[SearchResult]:
        """Perform a hybrid (keyword + vector) search against one collection.

        Args:
            client: Backend client handle.
            query: Text query for the keyword search component.
            query_embedding: Dense vector for the vector search component.
            alpha: Balance between keyword (0.0) and vector (1.0).
            limit: Maximum number of results to return.
            filters: Optional AND-combined metadata filter clauses.
            collection: Target collection name. ``None`` uses the default.

        Returns:
            Ranked list of ``SearchResult`` objects.
        """
        ...

    @abstractmethod
    def delete_collection(self, client: Any, collection: Optional[str] = None) -> None:
        """Drop an entire collection."""
        ...

    @abstractmethod
    def delete_by_source(
        self, client: Any, source: str, collection: Optional[str] = None
    ) -> int:
        """Delete all documents matching the given source path.

        Returns:
            Number of documents deleted.
        """
        ...

    @abstractmethod
    def delete_by_source_key(
        self,
        client: Any,
        source_key: str,
        legacy_source: Optional[str] = None,
        collection: Optional[str] = None,
    ) -> int:
        """Delete documents by stable source_key, with legacy_source fallback.

        Returns:
            Number of documents deleted.
        """
        ...

    @abstractmethod
    def aggregate_by_source(
        self,
        client: Any,
        collection: Optional[str] = None,
        source_filter: Optional[str] = None,
        connector_filter: Optional[str] = None,
    ) -> list[dict]:
        """Group chunk counts by source_key with optional filters."""
        ...

    @abstractmethod
    def get_collection_stats(
        self,
        client: Any,
        collection: Optional[str] = None,
    ) -> Optional[dict]:
        """Return aggregate stats for a collection; None if it does not exist."""
        ...

    @abstractmethod
    def list_collections(self, client: Any) -> list[dict]:
        """Return all collections with their chunk counts."""
        ...

    # -- Visual collection operations (FR-501, FR-502, FR-506, FR-507, NFR-909) --

    @abstractmethod
    def ensure_visual_collection(
        self,
        client: Any,
        collection: Optional[str] = None,
    ) -> None:
        """Create the visual collection if it does not exist (idempotent). FR-502"""
        ...

    @abstractmethod
    def add_visual_documents(
        self,
        client: Any,
        documents: list[dict[str, Any]],
        collection: Optional[str] = None,
    ) -> int:
        """Batch-insert visual page objects. FR-507

        Returns:
            Number of objects inserted.
        """
        ...

    @abstractmethod
    def delete_visual_by_source_key(
        self,
        client: Any,
        source_key: str,
        collection: Optional[str] = None,
    ) -> int:
        """Delete all visual page objects matching source_key. FR-506

        Returns:
            Number of objects deleted.
        """
        ...

    @abstractmethod
    def search_visual(
        self,
        client: Any,
        query_vector: list[float],
        limit: int,
        score_threshold: float,
        tenant_id: Optional[str] = None,
        collection: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Search the visual page collection by near-vector similarity. FR-313

        Args:
            client: Backend client handle.
            query_vector: 128-dim float query vector.
            limit: Maximum number of results.
            score_threshold: Minimum cosine similarity threshold.
            tenant_id: Optional tenant filter.
            collection: Visual collection name. None uses default.

        Returns:
            List of visual page result dicts ordered by descending score.
        """
        ...

    def close_client(self, client: Any) -> None:
        """Close a persistent client. Default is a no-op."""
