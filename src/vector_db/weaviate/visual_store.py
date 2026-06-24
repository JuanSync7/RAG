# @summary
# Weaviate visual page collection store for the visual embedding pipeline.
# Exports: ensure_visual_collection, add_visual_documents, delete_visual_by_source_key, visual_search
# Deps: weaviate-client, config.settings, src.platform.observability
# @end-summary
"""Weaviate visual page collection store.

Manages the ``RAGVisualPages`` collection used by the visual embedding pipeline.
The collection stores per-page visual embeddings with a named vector
``mean_vector`` (128-dim, HNSW, cosine) alongside JSON-serialised patch
vectors stored as TEXT. All operations are collection-name-parameterised and
collection creation is idempotent.

Functional requirements covered: FR-501 – FR-507.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter, MetadataQuery

from config.settings import RAG_INGESTION_VISUAL_TARGET_COLLECTION
from src.platform.observability import get_tracer

logger = logging.getLogger("rag.vector_db.weaviate.visual_store")
tracer = get_tracer()


def ensure_visual_collection(
    client: weaviate.WeaviateClient,
    collection: str = RAG_INGESTION_VISUAL_TARGET_COLLECTION,
) -> None:
    """Create the visual page collection if it does not exist (idempotent).

    Collection schema:
    - Named vector "mean_vector": 128-dim float32, HNSW index, cosine distance. FR-504
    - Properties: document_id(str), page_number(int), source_key(str),
      source_uri(str), source_name(str), tenant_id(str), total_pages(int),
      page_width_px(int), page_height_px(int), minio_key(str),
      patch_vectors(text/JSON). FR-503, FR-505
    - Vectorizer: none (pre-computed embeddings). FR-504

    Args:
        client: Weaviate client handle.
        collection: Collection name. Default: "RAGVisualPages". FR-501, FR-502
    """
    if client.collections.exists(collection):
        logger.debug("Visual collection %r already exists — skipping creation.", collection)
        return

    named_vector_config = Configure.NamedVectors.none(
        name="mean_vector",
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric="cosine",
        ),
        dimensions=128,
    )

    properties = [
        Property(name="document_id", data_type=DataType.TEXT),
        Property(name="page_number", data_type=DataType.INT),
        Property(name="source_key", data_type=DataType.TEXT),
        Property(name="source_uri", data_type=DataType.TEXT),
        Property(name="source_name", data_type=DataType.TEXT),
        Property(name="tenant_id", data_type=DataType.TEXT),
        Property(name="total_pages", data_type=DataType.INT),
        Property(name="page_width_px", data_type=DataType.INT),
        Property(name="page_height_px", data_type=DataType.INT),
        Property(name="minio_key", data_type=DataType.TEXT),
        # JSON-serialised list[list[float]]; stored as opaque text, never vectorised.
        Property(name="patch_vectors", data_type=DataType.TEXT, skip_vectorization=True),
    ]

    client.collections.create(
        name=collection,
        properties=properties,
        vectorizer_config=[named_vector_config],
    )
    logger.info("Created visual collection %r with named vector 'mean_vector'.", collection)


def add_visual_documents(
    client: weaviate.WeaviateClient,
    documents: list[dict[str, Any]],
    collection: str = RAG_INGESTION_VISUAL_TARGET_COLLECTION,
) -> int:
    """Batch-insert visual page objects into the named collection.

    Each document dict must contain:
    - All properties from FR-503
    - "mean_vector": list[float] (128-dim) for the named vector
    - "patch_vectors": JSON-serialized list[list[float]] as str

    Args:
        client: Weaviate client handle.
        documents: List of visual page object dicts.
        collection: Target collection name. FR-507

    Returns:
        Number of objects successfully inserted.
    """
    if not documents:
        return 0

    col = client.collections.get(collection)

    with col.batch.dynamic() as batch:
        for doc in documents:
            mean_vector = doc["mean_vector"]
            # All keys except the named vector go into properties.
            properties = {k: v for k, v in doc.items() if k != "mean_vector"}
            batch.add_object(
                properties=properties,
                vector={"mean_vector": mean_vector},
            )

    failed = len(col.batch.failed_objects) if hasattr(col.batch, "failed_objects") else 0
    inserted = len(documents) - failed
    logger.info(
        "Batch insert into %r: %d inserted, %d failed.",
        collection, inserted, failed,
    )
    return inserted


def visual_search(
    client: weaviate.WeaviateClient,
    query_vector: list[float],
    limit: int,
    score_threshold: float,
    tenant_id: Optional[str] = None,
    collection: str = RAG_INGESTION_VISUAL_TARGET_COLLECTION,
) -> list[dict[str, Any]]:
    """Search the visual page collection by near-vector similarity.

    Performs nearest-neighbor search on the ``mean_vector`` named vector
    of the specified collection using cosine distance. Results below
    ``score_threshold`` are excluded. The ``patch_vectors`` property is
    never included in results.

    Args:
        client: Weaviate client handle.
        query_vector: 128-dim float query vector (from ``embed_text_query``).
        limit: Maximum number of results to return (FR-303).
        score_threshold: Minimum cosine similarity; results below this
            are excluded (FR-303).
        tenant_id: When provided, only pages with matching ``tenant_id``
            are returned (FR-305). When None, no tenant filter applied.
        collection: Target collection name. Defaults to
            ``RAG_INGESTION_VISUAL_TARGET_COLLECTION`` value (FR-307).

    Returns:
        List of dicts ordered by descending cosine similarity, each
        containing: ``document_id`` (str), ``page_number`` (int),
        ``source_key`` (str), ``source_name`` (str), ``minio_key`` (str),
        ``tenant_id`` (str), ``total_pages`` (int), ``page_width_px`` (int),
        ``page_height_px`` (int), ``score`` (float). The ``patch_vectors``
        property is excluded (FR-311).

    Raises:
        weaviate.exceptions.WeaviateQueryError: On query failure.
    """
    span = tracer.span(
        "vector_store.visual_search",
        {"collection": collection, "limit": limit, "score_threshold": score_threshold},
    )

    # FR-309: get collection handle
    col = client.collections.get(collection)

    # FR-305: optional tenant filter
    filters = None
    if tenant_id is not None:
        filters = Filter.by_property("tenant_id").equal(tenant_id)

    # FR-311: explicit return properties — excludes patch_vectors
    return_properties = [
        "document_id",
        "page_number",
        "source_key",
        "source_name",
        "minio_key",
        "tenant_id",
        "total_pages",
        "page_width_px",
        "page_height_px",
    ]

    # FR-309: near-vector query on the mean_vector named vector
    response = col.query.near_vector(
        near_vector=query_vector,
        target_vector="mean_vector",
        limit=limit,
        filters=filters,
        return_properties=return_properties,
        return_metadata=MetadataQuery(distance=True),
    )

    # FR-303: convert cosine distance to similarity and filter by threshold
    results: list[dict[str, Any]] = []
    for obj in response.objects:
        distance = obj.metadata.distance if obj.metadata and obj.metadata.distance is not None else 1.0
        score = 1.0 - distance
        if score < score_threshold:
            continue
        results.append({
            "document_id": obj.properties.get("document_id", ""),
            "page_number": obj.properties.get("page_number", 0),
            "source_key": obj.properties.get("source_key", ""),
            "source_name": obj.properties.get("source_name", ""),
            "minio_key": obj.properties.get("minio_key", ""),
            "tenant_id": obj.properties.get("tenant_id", ""),
            "total_pages": obj.properties.get("total_pages", 0),
            "page_width_px": obj.properties.get("page_width_px", 0),
            "page_height_px": obj.properties.get("page_height_px", 0),
            "score": score,
        })

    span.set_attribute("result_count", len(results))
    span.end(status="ok")
    return results


def delete_visual_by_source_key(
    client: weaviate.WeaviateClient,
    source_key: str,
    collection: str = RAG_INGESTION_VISUAL_TARGET_COLLECTION,
) -> int:
    """Delete all visual page objects matching source_key.

    Args:
        client: Weaviate client handle.
        source_key: Stable source key to match. FR-506
        collection: Target collection name.

    Returns:
        Number of objects deleted.
    """
    col = client.collections.get(collection)
    where = Filter.by_property("source_key").equal(source_key)
    result = col.data.delete_many(where=where)
    deleted = getattr(result, "matches", 0) or 0
    logger.info(
        "Deleted %d visual page objects with source_key=%r from %r.",
        deleted, source_key, collection,
    )
    return deleted
