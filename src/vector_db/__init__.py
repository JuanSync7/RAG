# @summary
# Public API for the vector_db subsystem: config-driven backend dispatcher,
# single-collection search, multi-collection fan-out, aggregation, re-exported schemas,
# and visual collection operations for the visual embedding pipeline.
# Exports: create_persistent_client, get_client, close_client, ensure_collection,
#          collection_exists, delete_collection, add_documents, update_chunk_content,
#          delete_by_source, delete_by_source_key,
#          search, multi_search, aggregate_by_source, get_collection_stats, list_collections,
#          ensure_visual_collection, add_visual_documents, delete_visual_by_source_key,
#          search_visual, ensure_card_collection, add_document_cards,
#          DocumentRecord, SearchResult, SearchFilter, build_chunk_id
# Deps: config.settings, src.vector_db.backend, src.vector_db.common.schemas,
#       src.vector_db.weaviate.store, src.vector_db.weaviate.visual_store,
#       src.vector_db.weaviate.card_store
# @end-summary
"""Public API for the vector_db subsystem used by the ingestion and retrieval pipelines.

All pipeline code imports only from this module. Backend selection is controlled by
the ``VECTOR_DB_BACKEND`` config key — changing that key is all that is needed to
swap the underlying vector store.

Multi-collection support is available via ``multi_search()``, which fans out across
a list of named collections in parallel, deduplicates by object identity, and returns
a single ranked list. Single-collection callers pass ``collection=None`` (or omit it)
and the configured default is used transparently.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any, Generator, Optional

from src.vector_db.backend import VectorBackend
from src.vector_db.common import (
    DocumentRecord,
    SearchFilter,
    SearchResult,
)
from src.vector_db.weaviate import build_chunk_id
from src.vector_db.weaviate import (
    add_visual_documents,
    delete_visual_by_source_key,
    ensure_visual_collection,
)
from src.vector_db.weaviate import (
    add_document_cards,
    ensure_card_collection,
)

logger = logging.getLogger("rag.vector_db")

_vector_backend: VectorBackend | None = None


def _get_vector_backend() -> VectorBackend:
    """Return the process-wide vector store backend singleton.

    Constructs the backend on first call based on ``VECTOR_DB_BACKEND``.

    Returns:
        The active ``VectorBackend`` instance.

    Raises:
        ValueError: If ``VECTOR_DB_BACKEND`` is set to an unknown value.
    """
    global _vector_backend
    if _vector_backend is None:
        from config.settings import VECTOR_DB_BACKEND
        if VECTOR_DB_BACKEND == "weaviate":
            from src.vector_db.weaviate import WeaviateBackend
            _vector_backend = WeaviateBackend()
        else:
            raise ValueError(
                f"Unknown VECTOR_DB_BACKEND: {VECTOR_DB_BACKEND!r}. "
                "Valid values: 'weaviate'."
            )
    return _vector_backend


def _resolve_collection(collection: Optional[str]) -> str:
    """Resolve a collection name, falling back to the configured default."""
    if collection:
        return collection
    from config.settings import VECTOR_COLLECTION_DEFAULT
    return VECTOR_COLLECTION_DEFAULT


# ---------------------------------------------------------------------------
# Client lifecycle
# ---------------------------------------------------------------------------

def create_persistent_client() -> Any:
    """Create a long-lived vector store client.

    Caller is responsible for calling ``close_client()`` on shutdown.
    Preferred for server/worker processes that serve many queries.
    """
    return _get_vector_backend().create_persistent_client()


@contextmanager
def get_client() -> Generator[Any, None, None]:
    """Context manager for a short-lived vector store client.

    Suitable for CLI and batch scripts. For server use, prefer
    ``create_persistent_client()``.
    """
    with _get_vector_backend().get_ephemeral_client() as client:
        yield client


def close_client(client: Any) -> None:
    """Close a persistent client returned by ``create_persistent_client()``."""
    _get_vector_backend().close_client(client)


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def ensure_collection(client: Any, collection: Optional[str] = None) -> None:
    """Create the named collection if it does not exist (idempotent).

    Args:
        client: Vector store client handle.
        collection: Target collection name. ``None`` uses the default.
    """
    _get_vector_backend().ensure_collection(client, collection)


def collection_exists(client: Any, collection: Optional[str] = None) -> bool:
    """Return True iff the named collection exists on the backend.

    Args:
        client: Vector store client handle.
        collection: Target collection name. ``None`` uses the default.

    Returns:
        True if the collection exists, False otherwise.
    """
    return _get_vector_backend().collection_exists(client, collection)


def delete_collection(client: Any, collection: Optional[str] = None) -> None:
    """Drop an entire collection.

    Args:
        client: Vector store client handle.
        collection: Target collection name. ``None`` uses the default.
    """
    _get_vector_backend().delete_collection(client, collection)


# ---------------------------------------------------------------------------
# Document operations
# ---------------------------------------------------------------------------

def add_documents(
    client: Any,
    documents: list[DocumentRecord],
    collection: Optional[str] = None,
) -> int:
    """Insert documents with pre-computed embeddings.

    Args:
        client: Vector store client handle.
        documents: List of ``DocumentRecord`` objects to insert.
        collection: Target collection name. ``None`` uses the default.

    Returns:
        Number of documents added.
    """
    return _get_vector_backend().add_documents(client, documents, collection)


def update_chunk_content(
    client: Any,
    chunk_uuid: str,
    *,
    text: str,
    content_hash: str,
    fuzzy_fingerprint: Optional[str] = None,
    collection: Optional[str] = None,
) -> bool:
    """Replace a canonical chunk's text + dedup metadata in place (FR-3432).

    Used by the cross-document dedup node when a richer incoming chunk
    replaces an existing canonical chunk via fuzzy match.

    Args:
        client: Vector store client handle.
        chunk_uuid: UUID of the canonical chunk to update.
        text: New canonical text.
        content_hash: SHA-256 of the new normalised text.
        fuzzy_fingerprint: Optional serialised MinHash signature.
        collection: Target collection name. ``None`` uses the default.

    Returns:
        True on success, False on error (non-fatal to the pipeline).
    """
    return _get_vector_backend().update_chunk_content(
        client,
        chunk_uuid,
        text=text,
        content_hash=content_hash,
        fuzzy_fingerprint=fuzzy_fingerprint,
        collection=collection,
    )


def delete_by_source(
    client: Any,
    source: str,
    collection: Optional[str] = None,
) -> int:
    """Delete all documents matching the given source path.

    Args:
        client: Vector store client handle.
        source: Source identifier to match.
        collection: Target collection name. ``None`` uses the default.

    Returns:
        Number of documents deleted.
    """
    return _get_vector_backend().delete_by_source(client, source, collection)


def delete_by_source_key(
    client: Any,
    source_key: str,
    legacy_source: Optional[str] = None,
    collection: Optional[str] = None,
) -> int:
    """Delete documents by stable source_key, with legacy_source fallback.

    Args:
        client: Vector store client handle.
        source_key: Stable source key to match.
        legacy_source: Optional fallback source name for older records.
        collection: Target collection name. ``None`` uses the default.

    Returns:
        Number of documents deleted.
    """
    return _get_vector_backend().delete_by_source_key(
        client, source_key, legacy_source, collection
    )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(
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
        client: Vector store client handle.
        query: Text query for the keyword search component.
        query_embedding: Dense vector for the vector search component.
        alpha: Balance between keyword (0.0) and vector (1.0).
        limit: Maximum number of results to return.
        filters: Optional AND-combined metadata filter clauses.
        collection: Target collection name. ``None`` uses the default.

    Returns:
        Ranked list of ``SearchResult`` objects.
    """
    return _get_vector_backend().search(
        client, query, query_embedding, alpha, limit, filters, collection
    )


def multi_search(
    client: Any,
    query: str,
    query_embedding: list[float],
    alpha: float,
    limit: int,
    collections: Optional[list[str]] = None,
    filters: Optional[list[SearchFilter]] = None,
) -> list[SearchResult]:
    """Fan-out hybrid search across multiple collections.

    Issues one search per collection in parallel, deduplicates results by
    object identity (``object_id`` when set, otherwise by text content),
    keeps the highest-scoring instance of each duplicate, and returns a
    single list sorted by descending score truncated to ``limit``.

    Args:
        client: Vector store client handle.
        query: Text query for the keyword search component.
        query_embedding: Dense vector for the vector search component.
        alpha: Balance between keyword (0.0) and vector (1.0).
        limit: Maximum number of results to return across all collections.
        collections: List of collection names to search. When ``None`` or
            empty, falls back to a single search on the default collection.
        filters: Optional AND-combined metadata filter clauses applied to
            every collection in the fan-out.

    Returns:
        Deduplicated, ranked list of ``SearchResult`` objects.
    """
    if not collections:
        return search(client, query, query_embedding, alpha, limit, filters)

    backend = _get_vector_backend()

    def _search_one(col: str) -> list[SearchResult]:
        return backend.search(
            client, query, query_embedding, alpha, limit, filters, col
        )

    all_results: list[SearchResult] = []
    with ThreadPoolExecutor(max_workers=len(collections)) as pool:
        futures = {pool.submit(_search_one, col): col for col in collections}
        for future in as_completed(futures):
            col = futures[future]
            try:
                all_results.extend(future.result())
            except Exception:
                logger.warning(
                    "multi_search: collection %r search failed", col, exc_info=True
                )

    # Deduplicate — prefer the higher-scoring occurrence of each item.
    seen: dict[str, SearchResult] = {}
    for r in all_results:
        key = r.object_id if r.object_id else r.text
        existing = seen.get(key)
        if existing is None or r.score > existing.score:
            seen[key] = r

    return sorted(seen.values(), key=lambda r: r.score, reverse=True)[:limit]


# ---------------------------------------------------------------------------
# Aggregation and listing (FR-3050 through FR-3053)
# ---------------------------------------------------------------------------


def aggregate_by_source(
    client: Any,
    collection: Optional[str] = None,
    source_filter: Optional[str] = None,
    connector_filter: Optional[str] = None,
) -> list[dict]:
    """Group chunk counts by source_key with optional filters."""
    return _get_vector_backend().aggregate_by_source(
        client, _resolve_collection(collection), source_filter, connector_filter
    )


def get_collection_stats(
    client: Any,
    collection: Optional[str] = None,
) -> Optional[dict]:
    """Return aggregate stats for a collection; None if it does not exist."""
    return _get_vector_backend().get_collection_stats(
        client, _resolve_collection(collection)
    )


def list_collections(client: Any) -> list[dict]:
    """Return all collections with their chunk counts."""
    return _get_vector_backend().list_collections(client)


# ---------------------------------------------------------------------------
# Visual collection operations (FR-501, FR-502, FR-506, FR-507)
# ---------------------------------------------------------------------------

def ensure_visual_collection(
    client: Any,
    collection: Optional[str] = None,
) -> None:
    """Create the visual page collection if it does not exist (idempotent).

    Delegates to the backend implementation. When ``collection`` is ``None``,
    uses the backend default (``RAGVisualPages``).

    Args:
        client: Vector store client handle.
        collection: Visual collection name. ``None`` uses the default.
    """
    _get_vector_backend().ensure_visual_collection(client, collection)


def add_visual_documents(
    client: Any,
    documents: list[dict[str, Any]],
    collection: Optional[str] = None,
) -> int:
    """Batch-insert visual page objects into the visual collection.

    Args:
        client: Vector store client handle.
        documents: List of visual page object dicts, each containing all
            FR-503 properties plus a ``mean_vector`` (128-dim list[float]).
        collection: Visual collection name. ``None`` uses the default.

    Returns:
        Number of objects inserted.
    """
    return _get_vector_backend().add_visual_documents(client, documents, collection)


def delete_visual_by_source_key(
    client: Any,
    source_key: str,
    collection: Optional[str] = None,
) -> int:
    """Delete all visual page objects matching the given source_key.

    Args:
        client: Vector store client handle.
        source_key: Stable source key to match.
        collection: Visual collection name. ``None`` uses the default.

    Returns:
        Number of objects deleted.
    """
    return _get_vector_backend().delete_visual_by_source_key(
        client, source_key, collection
    )


def search_visual(
    client: Any,
    query_vector: list[float],
    limit: int,
    score_threshold: float,
    tenant_id: Optional[str] = None,
    collection: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Search the visual page collection by near-vector similarity. FR-313

    Delegates to the backend implementation. When ``collection`` is ``None``,
    uses the backend default (``RAGVisualPages``).

    Args:
        client: Vector store client handle.
        query_vector: 128-dim float query vector.
        limit: Maximum number of results.
        score_threshold: Minimum cosine similarity threshold.
        tenant_id: Optional tenant filter.
        collection: Visual collection name. ``None`` uses the default.

    Returns:
        List of visual page result dicts ordered by descending score.
    """
    return _get_vector_backend().search_visual(
        client, query_vector, limit, score_threshold, tenant_id, collection
    )


__all__ = [
    # Client lifecycle
    "create_persistent_client",
    "get_client",
    "close_client",
    # Collection management
    "ensure_collection",
    "collection_exists",
    "delete_collection",
    # Document operations
    "add_documents",
    "update_chunk_content",
    "delete_by_source",
    "delete_by_source_key",
    # Search
    "search",
    "multi_search",
    # Aggregation and listing
    "aggregate_by_source",
    "get_collection_stats",
    "list_collections",
    # Visual collection operations (FR-502, FR-506, FR-507, FR-313)
    "ensure_visual_collection",
    "add_visual_documents",
    "delete_visual_by_source_key",
    "search_visual",
    # Document-card collection operations (RAPTOR-lite routing)
    "ensure_card_collection",
    "add_document_cards",
    # Re-exported schemas
    "DocumentRecord",
    "SearchResult",
    "SearchFilter",
    # Utilities
    "build_chunk_id",
]
