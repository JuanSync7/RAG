# @summary
# WeaviateBackend: VectorBackend implementation delegating to src.vector_db.weaviate.store
# and src.vector_db.weaviate.visual_store for visual collection operations.
# Exports: WeaviateBackend
# Deps: src.vector_db.backend, src.vector_db.common.schemas, src.vector_db.weaviate.store,
#       src.vector_db.weaviate.visual_store, weaviate
# Visual methods: ensure_visual_collection, add_visual_documents,
#   delete_visual_by_source_key, search_visual (FR-502, FR-506, FR-507, FR-313)
# @end-summary
"""Weaviate implementation of the VectorBackend contract.

Thin delegation layer — all Weaviate-specific logic lives in
``src.vector_db.weaviate.store``. The only logic added here is
``SearchFilter`` → ``weaviate.classes.query.Filter`` translation.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator, Optional

from src.vector_db.backend import VectorBackend
from src.vector_db.common import (
    DocumentRecord,
    SearchFilter,
    SearchResult,
)
from src.vector_db.weaviate.store import (
    create_persistent_client as _wv_create_persistent,
    get_weaviate_client as _wv_get_ephemeral,
    ensure_collection as _wv_ensure_collection,
    collection_exists as _wv_collection_exists,
    add_documents as _wv_add_documents,
    hybrid_search as _wv_hybrid_search,
    delete_collection as _wv_delete_collection,
    delete_documents_by_source as _wv_delete_by_source,
    delete_documents_by_source_key as _wv_delete_by_source_key,
    aggregate_by_source as _wv_aggregate_by_source,
    get_collection_stats as _wv_get_collection_stats,
    list_collections as _wv_list_collections,
    update_chunk_content as _wv_update_chunk_content,
)
from src.vector_db.weaviate.visual_store import (
    ensure_visual_collection as _wv_ensure_visual_collection,
    add_visual_documents as _wv_add_visual_documents,
    delete_visual_by_source_key as _wv_delete_visual_by_source_key,
    visual_search as _wv_visual_search,
)
from config.settings import WEAVIATE_COLLECTION_NAME

logger = logging.getLogger("rag.vector_db.weaviate")


class WeaviateBackend(VectorBackend):
    """Weaviate vector store backend."""

    def _col(self, collection: Optional[str]) -> str:
        return collection or WEAVIATE_COLLECTION_NAME

    def create_persistent_client(self) -> Any:
        return _wv_create_persistent()

    @contextmanager
    def get_ephemeral_client(self) -> Generator[Any, None, None]:
        with _wv_get_ephemeral() as client:
            yield client

    def ensure_collection(self, client: Any, collection: Optional[str] = None) -> None:
        _wv_ensure_collection(client, collection=self._col(collection))

    def collection_exists(
        self, client: Any, collection: Optional[str] = None
    ) -> bool:
        return _wv_collection_exists(client, collection=self._col(collection))

    def add_documents(
        self,
        client: Any,
        documents: list[DocumentRecord],
        collection: Optional[str] = None,
    ) -> int:
        return _wv_add_documents(
            client,
            texts=[d.text for d in documents],
            embeddings=[d.embedding for d in documents],
            metadatas=[d.metadata for d in documents],
            collection=self._col(collection),
        )

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
        return _wv_update_chunk_content(
            client,
            chunk_uuid,
            text=text,
            content_hash=content_hash,
            fuzzy_fingerprint=fuzzy_fingerprint,
            collection=self._col(collection),
        )

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
        resolved = self._col(collection)
        wv_filter = self._translate_filters(filters)
        raw = _wv_hybrid_search(
            client, query, query_embedding, alpha, limit, wv_filter,
            collection=resolved,
        )
        return [
            SearchResult(
                text=r["text"],
                score=r["score"],
                metadata=r["metadata"],
                collection=resolved,
            )
            for r in raw
        ]

    def delete_collection(self, client: Any, collection: Optional[str] = None) -> None:
        _wv_delete_collection(client, collection=self._col(collection))

    def delete_by_source(
        self, client: Any, source: str, collection: Optional[str] = None
    ) -> int:
        return _wv_delete_by_source(client, source, collection=self._col(collection))

    def delete_by_source_key(
        self,
        client: Any,
        source_key: str,
        legacy_source: Optional[str] = None,
        collection: Optional[str] = None,
    ) -> int:
        return _wv_delete_by_source_key(
            client, source_key, legacy_source, collection=self._col(collection)
        )

    def aggregate_by_source(
        self,
        client: Any,
        collection: Optional[str] = None,
        source_filter: Optional[str] = None,
        connector_filter: Optional[str] = None,
    ) -> list[dict]:
        return _wv_aggregate_by_source(
            client, collection=self._col(collection),
            source_filter=source_filter, connector_filter=connector_filter,
        )

    def get_collection_stats(
        self,
        client: Any,
        collection: Optional[str] = None,
    ) -> Optional[dict]:
        return _wv_get_collection_stats(client, collection=self._col(collection))

    def list_collections(self, client: Any) -> list[dict]:
        return _wv_list_collections(client)

    # ------------------------------------------------------------------
    # Visual collection operations (FR-501, FR-502, FR-506, FR-507)
    # ------------------------------------------------------------------

    _VISUAL_COLLECTION_DEFAULT = "RAGVisualPages"

    def ensure_visual_collection(
        self,
        client: Any,
        collection: Optional[str] = None,
    ) -> None:
        _wv_ensure_visual_collection(
            client, collection or self._VISUAL_COLLECTION_DEFAULT
        )

    def add_visual_documents(
        self,
        client: Any,
        documents: list[dict[str, Any]],
        collection: Optional[str] = None,
    ) -> int:
        return _wv_add_visual_documents(
            client, documents, collection or self._VISUAL_COLLECTION_DEFAULT
        )

    def delete_visual_by_source_key(
        self,
        client: Any,
        source_key: str,
        collection: Optional[str] = None,
    ) -> int:
        return _wv_delete_visual_by_source_key(
            client, source_key, collection or self._VISUAL_COLLECTION_DEFAULT
        )

    def search_visual(
        self,
        client: Any,
        query_vector: list[float],
        limit: int,
        score_threshold: float,
        tenant_id: Optional[str] = None,
        collection: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        return _wv_visual_search(
            client,
            query_vector,
            limit,
            score_threshold,
            tenant_id,
            collection or self._VISUAL_COLLECTION_DEFAULT,
        )

    def close_client(self, client: Any) -> None:
        if client is not None:
            try:
                client.close()
            except Exception:
                logger.debug("Weaviate client close error (ignored)", exc_info=True)

    # ------------------------------------------------------------------
    # Filter translation
    # ------------------------------------------------------------------

    def _translate_filters(self, filters: Optional[list[SearchFilter]]) -> Any:
        if not filters:
            return None
        clauses = [self._single_filter(f) for f in filters]
        clauses = [c for c in clauses if c is not None]
        if not clauses:
            return None
        result = clauses[0]
        for clause in clauses[1:]:
            result = result & clause
        return result

    @staticmethod
    def _single_filter(f: SearchFilter) -> Any:
        from weaviate.classes.query import Filter as WeaviateFilter

        op = f.operator.lower()
        if op == "not_in":
            values = list(f.value or [])
            if not values:
                return None
            # AND-chain of not_equal clauses. Reliable across weaviate-client
            # versions and explicit in query plans. Filter is `prop != v1 AND
            # prop != v2 AND ...`, i.e. "prop is not in [...]".
            chain = WeaviateFilter.by_property(f.property).not_equal(values[0])
            for v in values[1:]:
                chain = chain & WeaviateFilter.by_property(f.property).not_equal(v)
            return chain
        if op == "in":
            values = list(f.value or [])
            if not values:
                return None
            # `prop is any of [...]` — a single contains_any clause. Mirrors the
            # empty-set handling of `not_in` (empty → no-op None). Used to
            # restrict one hybrid search to a SET of document_ids in one query.
            return WeaviateFilter.by_property(f.property).contains_any(values)

        prop = WeaviateFilter.by_property(f.property)
        ops = {
            "eq": prop.equal,
            "ne": prop.not_equal,
            "like": prop.like,
            "gt": prop.greater_than,
            "lt": prop.less_than,
            "gte": prop.greater_or_equal,
            "lte": prop.less_or_equal,
        }
        fn = ops.get(op)
        if fn is None:
            raise ValueError(
                f"Unsupported filter operator: {f.operator!r}. "
                f"Valid operators: {sorted([*ops, 'not_in', 'in'])}"
            )
        return fn(f.value)
