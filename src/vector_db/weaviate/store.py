# @summary
# Weaviate client helpers: connection (embedded or networked), collection management,
# CRUD, hybrid search, and aggregation. Connection mode is controlled by
# ``config.settings.RAG_WEAVIATE_MODE`` ("embedded" or "networked").
# All collection-scoped operations accept a collection parameter for multi-collection support.
# Exports: create_persistent_client, get_weaviate_client, ensure_collection, build_chunk_id,
#          add_documents, hybrid_search, delete_collection,
#          delete_documents_by_source, delete_documents_by_source_key,
#          aggregate_by_source, get_collection_stats, list_collections,
#          update_chunk_content
# Deps: weaviate, config.settings, src.platform.observability
# @end-summary
"""Low-level Weaviate operations: connection, schema, CRUD, and search.

All collection-scoped functions accept an optional ``collection`` parameter
that defaults to ``WEAVIATE_COLLECTION_NAME``. This module is imported only
by ``WeaviateBackend`` — pipeline code accesses these capabilities through
``src.vector_db`` instead.
"""
from __future__ import annotations


import hashlib
import logging
import uuid
from contextlib import contextmanager
from typing import Optional

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter, HybridFusion, MetadataQuery

from config.settings import (
    WEAVIATE_COLLECTION_NAME,
    WEAVIATE_DATA_DIR,
    HYBRID_SEARCH_ALPHA,
    SEARCH_LIMIT,
    RAG_WEAVIATE_MODE,
    RAG_WEAVIATE_HOST,
    RAG_WEAVIATE_HTTP_PORT,
    RAG_WEAVIATE_GRPC_PORT,
)
from src.platform.observability import get_tracer

logger = logging.getLogger("rag.vector_db.weaviate.store")
tracer = get_tracer()


def _connect() -> weaviate.WeaviateClient:
    """Open a Weaviate client based on ``RAG_WEAVIATE_MODE``.

    Wrapped in its own span so the connect cost (previously invisible — especially
    for embedded mode, which spawns a Java subprocess) shows up in traces.
    """
    attrs = {"mode": RAG_WEAVIATE_MODE}
    if RAG_WEAVIATE_MODE == "networked":
        attrs["host"] = RAG_WEAVIATE_HOST
        attrs["http_port"] = RAG_WEAVIATE_HTTP_PORT
        attrs["grpc_port"] = RAG_WEAVIATE_GRPC_PORT
    with tracer.span("vector_store.connect", attrs):
        if RAG_WEAVIATE_MODE == "networked":
            return weaviate.connect_to_local(
                host=RAG_WEAVIATE_HOST,
                port=RAG_WEAVIATE_HTTP_PORT,
                grpc_port=RAG_WEAVIATE_GRPC_PORT,
            )
        return weaviate.connect_to_embedded(persistence_data_path=WEAVIATE_DATA_DIR)


def create_persistent_client() -> weaviate.WeaviateClient:
    """Create a long-lived Weaviate client (embedded or networked)."""
    span = tracer.start_span("vector_store.create_persistent_client")
    client: Optional[weaviate.WeaviateClient] = None
    try:
        client = _connect()
        span.end(status="ok")
        return client
    except Exception:
        span.end(status="error")
        if client is not None:
            client.close()
        raise


@contextmanager
def get_weaviate_client():
    """Context manager for a short-lived Weaviate client (embedded or networked)."""
    span = tracer.start_span("vector_store.get_weaviate_client")
    client = _connect()
    try:
        yield client
    finally:
        client.close()
        span.end(status="ok")


def ensure_collection(
    client: weaviate.WeaviateClient,
    collection: str = WEAVIATE_COLLECTION_NAME,
) -> None:
    """Create the named collection if it does not exist (idempotent)."""
    span = tracer.start_span("vector_store.ensure_collection", {"collection": collection})
    if client.collections.exists(collection):
        span.end(status="ok")
        return

    client.collections.create(
        name=collection,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="source_uri", data_type=DataType.TEXT),
            Property(name="source_key", data_type=DataType.TEXT),
            Property(name="source_id", data_type=DataType.TEXT),
            Property(name="connector", data_type=DataType.TEXT),
            Property(name="source_version", data_type=DataType.TEXT),
            Property(name="retrieval_text_origin", data_type=DataType.TEXT),
            Property(name="citation_source_uri", data_type=DataType.TEXT),
            Property(name="provenance_method", data_type=DataType.TEXT),
            Property(name="provenance_confidence", data_type=DataType.NUMBER),
            Property(name="original_char_start", data_type=DataType.INT),
            Property(name="original_char_end", data_type=DataType.INT),
            Property(name="refactored_char_start", data_type=DataType.INT),
            Property(name="refactored_char_end", data_type=DataType.INT),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="author", data_type=DataType.TEXT),
            Property(name="date", data_type=DataType.TEXT),
            Property(name="tags", data_type=DataType.TEXT),
            Property(name="chunk_index", data_type=DataType.INT),
            Property(name="total_chunks", data_type=DataType.INT),
            Property(name="section_path", data_type=DataType.TEXT),
            Property(name="heading", data_type=DataType.TEXT),
            Property(name="heading_level", data_type=DataType.INT),
            Property(name="tenant_id", data_type=DataType.TEXT),
            Property(name="document_id", data_type=DataType.TEXT),
            # -- Cross-document dedup (FR-3430, FR-3431, FR-3432, FR-3433) --
            Property(
                name="content_hash",
                data_type=DataType.TEXT,
                description="SHA-256 of normalised chunk text for exact-match dedup",
                index_filterable=True,
                index_searchable=False,
            ),
            Property(
                name="source_documents",
                data_type=DataType.TEXT_ARRAY,
                description="Array of source_key values for multi-document provenance",
            ),
            Property(
                name="fuzzy_fingerprint",
                data_type=DataType.TEXT,
                description="Serialised MinHash signature (Tier 2 fuzzy dedup)",
                index_filterable=False,
                index_searchable=False,
            ),
            Property(
                name="canonical",
                data_type=DataType.BOOL,
                description="Always true; reserved for future soft-delete flows",
            ),
        ],
    )
    span.end(status="ok")


def build_chunk_id(source: str, chunk_index: int, text: str) -> str:
    """Deterministic UUID chunk ID for idempotent upserts."""
    payload = f"{source}:{chunk_index}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, payload))


def _normalize_chunk_uuid(candidate: object, source: str, chunk_index: int, text: str) -> str:
    if candidate is not None:
        raw = str(candidate).strip()
        if raw:
            try:
                return str(uuid.UUID(raw))
            except ValueError:
                return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))
    return build_chunk_id(source, chunk_index, text)


def add_documents(
    client: weaviate.WeaviateClient,
    texts: list[str],
    embeddings: list[list[float]],
    metadatas: Optional[list[dict]] = None,
    collection: str = WEAVIATE_COLLECTION_NAME,
) -> int:
    """Add documents with pre-computed embeddings to the named collection."""
    span = tracer.start_span(
        "vector_store.add_documents",
        {"count": len(texts), "collection": collection},
    )
    col = client.collections.get(collection)
    try:
        config = col.config.get()
        raw_props = getattr(config, "properties", []) or []
        collection_props = {
            getattr(p, "name", "") for p in raw_props if getattr(p, "name", "")
        }
    except Exception:
        collection_props = set()

    if metadatas is None:
        metadatas = [{}] * len(texts)

    with col.batch.dynamic() as batch:
        for text, embedding, metadata in zip(texts, embeddings, metadatas):
            source = str(metadata.get("source") or "").strip() or "unknown"
            source_identity = str(metadata.get("source_key") or "").strip() or source
            chunk_index = metadata.get("chunk_index", 0)
            chunk_id = _normalize_chunk_uuid(
                metadata.get("chunk_id"), source_identity, chunk_index, text
            )
            properties = {
                "text": text,
                "source": source,
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "date": metadata.get("date", ""),
                "tags": metadata.get("tags", ""),
                "chunk_index": metadata.get("chunk_index", 0),
                "total_chunks": metadata.get("total_chunks", 0),
                "section_path": metadata.get("section_path", ""),
                "heading": metadata.get("heading", ""),
                "heading_level": metadata.get("heading_level", 0),
                "tenant_id": metadata.get("tenant_id", "default"),
            }
            optional = {
                "source_uri": metadata.get("source_uri", ""),
                "source_key": metadata.get("source_key", ""),
                "source_id": metadata.get("source_id", ""),
                "connector": metadata.get("connector", "local_fs"),
                "source_version": metadata.get("source_version", ""),
                "retrieval_text_origin": metadata.get("retrieval_text_origin", "original"),
                "citation_source_uri": metadata.get("citation_source_uri", ""),
                "provenance_method": metadata.get("provenance_method", ""),
                "provenance_confidence": float(metadata.get("provenance_confidence", 0.0)),
                "original_char_start": int(metadata.get("original_char_start", -1)),
                "original_char_end": int(metadata.get("original_char_end", -1)),
                "refactored_char_start": int(metadata.get("refactored_char_start", -1)),
                "refactored_char_end": int(metadata.get("refactored_char_end", -1)),
                "document_id": metadata.get("document_id", ""),
            }
            for key, val in optional.items():
                if key in collection_props:
                    properties[key] = val
            batch.add_object(properties=properties, vector=embedding, uuid=chunk_id)

    span.end(status="ok")
    return len(texts)


def hybrid_search(
    client: weaviate.WeaviateClient,
    query: str,
    query_embedding: list[float],
    alpha: float = HYBRID_SEARCH_ALPHA,
    limit: int = SEARCH_LIMIT,
    filters: Optional[Filter] = None,
    collection: str = WEAVIATE_COLLECTION_NAME,
) -> list[dict]:
    """Perform hybrid search (BM25 + vector) on the named collection.

    Returns:
        List of dicts with ``text``, ``metadata``, and ``score`` keys.
    """
    span = tracer.start_span(
        "vector_store.hybrid_search",
        {"alpha": alpha, "limit": limit, "collection": collection, "has_filters": filters is not None},
    )
    col = client.collections.get(collection)
    results = col.query.hybrid(
        query=query,
        vector=query_embedding,
        alpha=alpha,
        limit=limit,
        filters=filters,
        fusion_type=HybridFusion.RELATIVE_SCORE,
        return_metadata=MetadataQuery(score=True),
    )

    documents = []
    for obj in results.objects:
        documents.append({
            "text": obj.properties.get("text", ""),
            "metadata": {
                "source": obj.properties.get("source", "unknown"),
                "source_uri": obj.properties.get("source_uri", ""),
                "source_key": obj.properties.get("source_key", ""),
                "source_id": obj.properties.get("source_id", ""),
                "connector": obj.properties.get("connector", "local_fs"),
                "source_version": obj.properties.get("source_version", ""),
                "retrieval_text_origin": obj.properties.get("retrieval_text_origin", "original"),
                "citation_source_uri": obj.properties.get("citation_source_uri", ""),
                "provenance_method": obj.properties.get("provenance_method", ""),
                "provenance_confidence": obj.properties.get("provenance_confidence", 0.0),
                "original_char_start": obj.properties.get("original_char_start", -1),
                "original_char_end": obj.properties.get("original_char_end", -1),
                "refactored_char_start": obj.properties.get("refactored_char_start", -1),
                "refactored_char_end": obj.properties.get("refactored_char_end", -1),
                "title": obj.properties.get("title", ""),
                "author": obj.properties.get("author", ""),
                "date": obj.properties.get("date", ""),
                "tags": obj.properties.get("tags", ""),
                "chunk_index": obj.properties.get("chunk_index", 0),
                "total_chunks": obj.properties.get("total_chunks", 0),
                "section_path": obj.properties.get("section_path", ""),
                "heading": obj.properties.get("heading", ""),
                "heading_level": obj.properties.get("heading_level", 0),
                "tenant_id": obj.properties.get("tenant_id", "default"),
            },
            "score": obj.metadata.score if obj.metadata else 0.0,
        })
    span.set_attribute("result_count", len(documents))
    span.end(status="ok")
    return documents


def delete_collection(
    client: weaviate.WeaviateClient,
    collection: str = WEAVIATE_COLLECTION_NAME,
) -> None:
    """Drop the named collection."""
    span = tracer.start_span("vector_store.delete_collection", {"collection": collection})
    if client.collections.exists(collection):
        client.collections.delete(collection)
    span.end(status="ok")


def delete_documents_by_source(
    client: weaviate.WeaviateClient,
    source: str,
    collection: str = WEAVIATE_COLLECTION_NAME,
) -> int:
    """Delete chunks by source metadata value from the named collection."""
    span = tracer.start_span(
        "vector_store.delete_documents_by_source",
        {"source": source, "collection": collection},
    )
    col = client.collections.get(collection)
    where = Filter.by_property("source").equal(source)
    result = col.data.delete_many(where=where)
    deleted = getattr(result, "matches", 0) or 0
    span.set_attribute("deleted_count", deleted)
    span.end(status="ok")
    return deleted


def delete_documents_by_source_key(
    client: weaviate.WeaviateClient,
    source_key: str,
    legacy_source: Optional[str] = None,
    collection: str = WEAVIATE_COLLECTION_NAME,
) -> int:
    """Delete chunks by stable source_key from the named collection."""
    span = tracer.start_span(
        "vector_store.delete_documents_by_source_key",
        {"source_key": source_key, "collection": collection},
    )
    col = client.collections.get(collection)
    try:
        where = Filter.by_property("source_key").equal(source_key)
        result = col.data.delete_many(where=where)
    except Exception:
        if not legacy_source:
            span.end(status="ok")
            return 0
        where = Filter.by_property("source").equal(legacy_source)
        result = col.data.delete_many(where=where)
    deleted = getattr(result, "matches", 0) or 0
    span.set_attribute("deleted_count", deleted)
    span.end(status="ok")
    return deleted


# ---------------------------------------------------------------------------
# Aggregation and listing (FR-3050 through FR-3052)
# ---------------------------------------------------------------------------


def aggregate_by_source(
    client: weaviate.WeaviateClient,
    collection: str = WEAVIATE_COLLECTION_NAME,
    source_filter: Optional[str] = None,
    connector_filter: Optional[str] = None,
) -> list[dict]:
    """Return chunk counts grouped by source_key.

    Each dict: source_key (str), source (str), connector (str), chunk_count (int).
    Uses Weaviate group_by aggregate -- no full object iteration.

    Raises:
        KeyError: if the collection does not exist.
        weaviate.exceptions.WeaviateQueryError: on query failure.
    """
    span = tracer.start_span(
        "vector_store.aggregate_by_source",
        {"collection": collection},
    )
    col = client.collections.get(collection)
    filters = []
    if source_filter:
        filters.append(Filter.by_property("source").like(f"*{source_filter}*"))
    if connector_filter:
        filters.append(Filter.by_property("connector").equal(connector_filter))
    combined = (
        filters[0]
        if len(filters) == 1
        else (Filter.all_of(filters) if filters else None)
    )
    response = col.aggregate.over_all(
        group_by=weaviate.classes.aggregate.GroupByAggregate(prop="source_key"),
        filters=combined,
        total_count=True,
    )
    results: list[dict] = []
    for group in response.groups:
        results.append({
            "source_key": group.grouped_by.value,
            "source": (
                group.properties.get("source", {})
                .get("top_occurrences", [{}])[0]
                .get("value", "")
            ),
            "connector": (
                group.properties.get("connector", {})
                .get("top_occurrences", [{}])[0]
                .get("value", "")
            ),
            "chunk_count": group.total_count,
        })
    span.end(status="ok")
    return results


def get_collection_stats(
    client: weaviate.WeaviateClient,
    collection: str = WEAVIATE_COLLECTION_NAME,
) -> Optional[dict]:
    """Return aggregate statistics for a collection.

    Returns dict with chunk_count, document_count, connector_breakdown.
    Returns None if the collection does not exist.

    Raises:
        weaviate.exceptions.WeaviateQueryError: on unexpected query failure.
    """
    span = tracer.start_span("vector_store.get_collection_stats", {"collection": collection})
    if not client.collections.exists(collection):
        span.end(status="not_found")
        return None
    col = client.collections.get(collection)
    total = col.aggregate.over_all(total_count=True)
    chunk_count = total.total_count or 0
    by_source = col.aggregate.over_all(
        group_by=weaviate.classes.aggregate.GroupByAggregate(prop="source_key"),
        total_count=True,
    )
    document_count = len(by_source.groups)
    by_connector = col.aggregate.over_all(
        group_by=weaviate.classes.aggregate.GroupByAggregate(prop="connector"),
        total_count=True,
    )
    connector_breakdown = {
        g.grouped_by.value: g.total_count for g in by_connector.groups
    }
    span.end(status="ok")
    return {
        "chunk_count": chunk_count,
        "document_count": document_count,
        "connector_breakdown": connector_breakdown,
    }


def list_collections(
    client: weaviate.WeaviateClient,
) -> list[dict]:
    """Return all collections visible to this client.

    Each dict: collection_name (str), chunk_count (int).
    Uses client.collections.list_all() for enumeration.

    Raises:
        weaviate.exceptions.WeaviateConnectionError: if client is not connected.
    """
    span = tracer.start_span("vector_store.list_collections")
    all_cols = client.collections.list_all(simple=True)
    results: list[dict] = []
    for name in all_cols:
        col = client.collections.get(name)
        agg = col.aggregate.over_all(total_count=True)
        results.append({
            "collection_name": name,
            "chunk_count": agg.total_count or 0,
        })
    span.end(status="ok")
    return results


# ---------------------------------------------------------------------------
# Cross-document dedup: canonical chunk updates (FR-3432)
# ---------------------------------------------------------------------------


def update_chunk_content(
    client: weaviate.WeaviateClient,
    chunk_uuid: str,
    *,
    text: str,
    content_hash: str,
    fuzzy_fingerprint: Optional[str] = None,
    collection: str = WEAVIATE_COLLECTION_NAME,
) -> bool:
    """Replace a canonical chunk's text + dedup metadata (Tier 2 replacement).

    Used by the dedup node when an incoming chunk's text is longer/richer than
    an existing canonical chunk that it fuzzy-matches against. The canonical
    record is updated in place — source_documents provenance is NOT reset.

    Args:
        client: Weaviate client handle.
        chunk_uuid: UUID of the canonical chunk to update.
        text: New canonical text (replaces existing).
        content_hash: SHA-256 of the new normalised text.
        fuzzy_fingerprint: Optional new MinHash signature (hex string).
        collection: Target collection name.

    Returns:
        True on success, False on error.
    """
    span = tracer.start_span(
        "vector_store.update_chunk_content",
        {"collection": collection, "chunk_uuid": chunk_uuid},
    )
    try:
        col = client.collections.get(collection)
        properties: dict = {
            "text": text,
            "content_hash": content_hash,
        }
        if fuzzy_fingerprint is not None:
            properties["fuzzy_fingerprint"] = fuzzy_fingerprint
        col.data.update(uuid=chunk_uuid, properties=properties)
        span.end(status="ok")
        return True
    except Exception:
        logger.error(
            "Failed to update canonical chunk %s in %s",
            chunk_uuid,
            collection,
            exc_info=True,
        )
        span.end(status="error")
        return False
