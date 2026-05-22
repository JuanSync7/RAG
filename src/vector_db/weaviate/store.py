# @summary
# Weaviate client helpers: connection (embedded or networked), collection management,
# CRUD, hybrid search, and aggregation. Connection mode is controlled by
# ``config.settings.RAG_WEAVIATE_MODE`` ("embedded" or "networked").
# All collection-scoped operations accept a collection parameter for multi-collection support.
# Exports: create_persistent_client, get_weaviate_client, ensure_collection, build_chunk_id,
#          add_documents, hybrid_search, delete_collection,
#          delete_documents_by_source, delete_documents_by_source_key,
#          delete_documents_by_staging_batch, get_source_hash,
#          aggregate_by_source, get_collection_stats, list_collections,
#          update_chunk_content, TABLE_AWARE_PROPERTIES
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
import json
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


# ---------------------------------------------------------------------------
# Table-aware + page provenance schema (introduced by PR #100).
#
# Extracted as a module-level constant so the schema-migration script
# (``scripts/migrate_weaviate_table_schema.py``) can backfill these properties
# on collections that pre-date the PR. The list is the single source of truth;
# ``ensure_collection`` consumes it below so the two code paths cannot drift.
# ---------------------------------------------------------------------------
TABLE_AWARE_PROPERTIES: list[Property] = [
    # ``chunk_type`` is intentionally free-form TEXT: the adaptive chunker
    # stamps "table_summary"/"table_row", while the legacy chunking node
    # stamps "table"/"text". Both must round-trip.
    Property(
        name="chunk_type",
        data_type=DataType.TEXT,
        description='Free-form chunk kind ("table_summary"|"table_row"|"table"|"text"|...)',
        index_filterable=True,
        index_searchable=False,
    ),
    Property(
        name="table_id",
        data_type=DataType.TEXT,
        description="Parser-stable id for the originating table",
        index_filterable=True,
        index_searchable=False,
    ),
    Property(
        name="table_group_id",
        data_type=DataType.TEXT,
        description="Within-document join key linking summary + row chunks",
        index_filterable=True,
        index_searchable=False,
    ),
    Property(
        name="table_row_index",
        data_type=DataType.INT,
        description="Zero-based body-row index for table_row chunks",
        index_filterable=True,
        index_searchable=False,
    ),
    Property(
        name="table_num_rows",
        data_type=DataType.INT,
        description="Total rows in the originating table (incl. header)",
    ),
    Property(
        name="table_num_cols",
        data_type=DataType.INT,
        description="Total columns in the originating table",
    ),
    Property(
        name="table_has_header",
        data_type=DataType.BOOL,
        description="Whether the originating table has a header row",
    ),
    Property(
        name="table_caption",
        data_type=DataType.TEXT,
        description="Caption text of the originating table (query-relevant)",
        index_filterable=False,
        index_searchable=True,
    ),
    Property(
        name="table_markdown",
        data_type=DataType.TEXT,
        description="Full markdown reconstruction; only set on table_summary chunks",
        index_filterable=False,
        index_searchable=True,
    ),
    Property(
        name="document_id",
        data_type=DataType.TEXT,
        description="Stable pointer to the parent document (e.g., file stem)",
        index_filterable=True,
        index_searchable=False,
    ),
    Property(
        name="page_no",
        data_type=DataType.INT,
        description="1-based page number of the chunk origin",
        index_filterable=True,
        index_searchable=False,
    ),
    Property(
        name="page_label",
        data_type=DataType.TEXT,
        description="Human-facing page label (e.g. 'vii', 'A-3')",
        index_filterable=True,
        index_searchable=False,
    ),
    Property(
        name="page_bbox",
        data_type=DataType.TEXT,
        description='JSON-encoded [x0,y0,x1,y1] bbox on the page; "" when absent',
        index_filterable=False,
        index_searchable=False,
    ),
    # xref_targets stores a JSON-encoded ``list[{type, value}]`` produced by
    # ``src.ingest.common.shared.cross_refs``. Weaviate has no native list-of-
    # struct type, so we serialise as TEXT and decode on the retrieval side
    # (``src.retrieval.xref_expansion``). Encoding is JSON (not csv/yaml) so
    # values like "§3.1" round-trip without escape hassles.
    Property(
        name="xref_targets",
        data_type=DataType.TEXT,
        description=(
            'JSON-encoded list[{type,value}] of cross-references detected in '
            'the chunk text; empty list "[]" when no refs.'
        ),
        index_filterable=False,
        index_searchable=False,
    ),
]


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
            # -- Tree retrieval (TREE_RETRIEVAL_DESIGN.md §3.1) --
            Property(
                name="node_kind",
                data_type=DataType.TEXT,
                description='"chunk" (leaf) or "section" (internal tree node)',
                index_filterable=True,
                index_searchable=False,
            ),
            Property(
                name="tree_level",
                data_type=DataType.INT,
                description="Depth in the heading hierarchy; len(heading_path) for leaves",
                index_filterable=True,
                index_searchable=False,
            ),
            Property(
                name="heading_path",
                data_type=DataType.TEXT_ARRAY,
                description="Ordered ancestor headings from root to this node",
                index_filterable=True,
                index_searchable=False,
            ),
            Property(
                name="parent_section_id",
                data_type=DataType.TEXT,
                description="Stable hash(document_id, heading_path[:-1]); enables sibling lookup",
                index_filterable=True,
                index_searchable=False,
            ),
            Property(
                name="child_count",
                data_type=DataType.INT,
                description="(sections only) Number of direct children; 0 for leaf chunks",
                index_filterable=True,
                index_searchable=False,
            ),
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
            # -- Atomicity & idempotency (Issue #42) --
            Property(
                name="staging_batch_id",
                data_type=DataType.TEXT,
                description="Per-document UUID set at workflow entry; used to roll back partial writes",
                index_filterable=True,
                index_searchable=False,
            ),
            Property(
                name="source_hash",
                data_type=DataType.TEXT,
                description="SHA-256 of source bytes; used to short-circuit unchanged re-ingest",
                index_filterable=True,
                index_searchable=False,
            ),
            # -- Table-aware chunking + page provenance --
            # Defined as a module-level constant (TABLE_AWARE_PROPERTIES) so
            # the schema-migration script can backfill collections created
            # before PR #100 without duplicating these declarations.
            *TABLE_AWARE_PROPERTIES,
        ],
    )
    span.end(status="ok")


def build_chunk_id(source: str, chunk_index: int, text: str) -> str:
    """Deterministic UUID chunk ID for idempotent upserts."""
    payload = f"{source}:{chunk_index}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, payload))


def build_table_chunk_id(
    source: str,
    table_group_id: str,
    chunk_type: str,
    table_row_index: object = None,
) -> str:
    """Deterministic UUID for table-derived chunks (summary + rows).

    Derives from ``(source, table_group_id, chunk_type, table_row_index)`` so
    re-ingesting a document with edited table content updates the same vectors
    rather than orphaning the prior ones. ``table_row_index`` may be ``None``
    for ``table_summary`` chunks; it is substituted with ``-1`` in the payload
    so the summary id never collides with row 0.
    """
    try:
        row = int(table_row_index) if table_row_index is not None else -1
    except (TypeError, ValueError):
        row = -1
    payload = f"{source}|tbl|{table_group_id}|{chunk_type}|{row}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, payload))


def build_parent_section_id(document_id: str, heading_path: list[str]) -> str:
    """Stable hash for a section's parent — `(document_id, heading_path[:-1])`.

    Used by tree-retrieval Stage 4c (lift) for sibling lookup. Returns "" for
    root-level chunks (no parent section). See TREE_RETRIEVAL_DESIGN.md §3.1.
    """
    if not heading_path:
        return ""
    parent_path = heading_path[:-1]
    if not parent_path:
        return ""
    payload = f"{document_id}:{'>'.join(parent_path)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def build_section_node_id(document_id: str, heading_path: list[str]) -> str:
    """Stable id for the section node owning ``heading_path``.

    A leaf chunk under section S has ``parent_section_id ==
    build_section_node_id(doc, S.heading_path)`` (= S's own ``chunk_id``).
    See TREE_RETRIEVAL_DESIGN.md §3.2/§4.1.
    """
    if not heading_path:
        return ""
    payload = f"{document_id}:{'>'.join(heading_path)}>__SECTION__"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


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
            # Table-aware chunk_id: for ``table_summary``/``table_row`` chunks
            # with a non-empty ``table_group_id`` we derive the UUID from the
            # stable (group_id, chunk_type, row_index) tuple so re-ingests of
            # the same document update the same vectors even when chunk_index
            # or text drifts. Explicit ``chunk_id`` in metadata still wins via
            # ``_normalize_chunk_uuid``.
            explicit_chunk_id = metadata.get("chunk_id")
            chunk_type_meta = str(metadata.get("chunk_type") or "")
            table_group_meta = str(metadata.get("table_group_id") or "").strip()
            candidate_id: object = explicit_chunk_id
            if (
                candidate_id in (None, "")
                and chunk_type_meta.startswith("table_")
                and table_group_meta
            ):
                candidate_id = build_table_chunk_id(
                    source_identity,
                    table_group_meta,
                    chunk_type_meta,
                    metadata.get("table_row_index"),
                )
            chunk_id = _normalize_chunk_uuid(
                candidate_id, source_identity, chunk_index, text
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
                "staging_batch_id": metadata.get("staging_batch_id", ""),
                "source_hash": metadata.get("source_hash", ""),
            }
            # Tree-retrieval fields. Backfilled here from existing chunk metadata
            # so legacy ingests gain `node_kind`/`tree_level`/`heading_path`/
            # `parent_section_id` without code changes upstream. Tree-aware ingest
            # nodes can override `node_kind` and `child_count` explicitly.
            heading_path_meta = metadata.get("heading_path") or []
            if not isinstance(heading_path_meta, list):
                heading_path_meta = []
            # Table-aware chunking + page provenance fields. ``page_bbox`` is
            # JSON-encoded so a tuple/list survives as TEXT; "" means absent.
            raw_bbox = metadata.get("page_bbox")
            if raw_bbox is None:
                bbox_str = ""
            elif isinstance(raw_bbox, str):
                bbox_str = raw_bbox
            else:
                try:
                    bbox_str = json.dumps(list(raw_bbox))
                except (TypeError, ValueError):
                    bbox_str = ""
            optional.update(
                {
                    "chunk_type": metadata.get("chunk_type", ""),
                    "table_id": metadata.get("table_id", ""),
                    "table_group_id": metadata.get("table_group_id", ""),
                    "table_row_index": int(metadata.get("table_row_index", -1)),
                    "table_num_rows": int(metadata.get("table_num_rows", 0)),
                    "table_num_cols": int(metadata.get("table_num_cols", 0)),
                    "table_has_header": bool(metadata.get("table_has_header", False)),
                    "table_caption": metadata.get("table_caption", ""),
                    "table_markdown": metadata.get("table_markdown", ""),
                    "page_no": int(metadata.get("page_no", 0)),
                    "page_label": metadata.get("page_label", ""),
                    "page_bbox": bbox_str,
                }
            )
            optional.update(
                {
                    "node_kind": metadata.get("node_kind", "chunk"),
                    "tree_level": int(
                        metadata.get("tree_level", len(heading_path_meta))
                    ),
                    "heading_path": [str(h) for h in heading_path_meta],
                    # Legacy-fallback: a leaf's parent_section_id is the
                    # canonical section-node id for its heading_path
                    # (TREE_RETRIEVAL_DESIGN §4.1). Tree-aware ingest nodes
                    # set this explicitly for sections (their *own* parent).
                    "parent_section_id": metadata.get(
                        "parent_section_id",
                        build_section_node_id(
                            str(metadata.get("document_id", "")),
                            [str(h) for h in heading_path_meta],
                        ),
                    ),
                    "child_count": int(metadata.get("child_count", 0)),
                }
            )
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
                # Tree retrieval fields (TREE_RETRIEVAL_DESIGN.md §3.1).
                # Default to "chunk" when absent so legacy collections behave like leaves.
                "node_kind": obj.properties.get("node_kind", "chunk"),
                "tree_level": obj.properties.get("tree_level", 0),
                "heading_path": obj.properties.get("heading_path", []) or [],
                "parent_section_id": obj.properties.get("parent_section_id", ""),
                "child_count": obj.properties.get("child_count", 0),
                "document_id": obj.properties.get("document_id", ""),
            },
            "score": obj.metadata.score if obj.metadata else 0.0,
            "uuid": str(obj.uuid) if getattr(obj, "uuid", None) else "",
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


def delete_documents_by_staging_batch(
    client: weaviate.WeaviateClient,
    staging_batch_id: str,
    collection: str = WEAVIATE_COLLECTION_NAME,
) -> int:
    """Delete chunks tagged with the given staging_batch_id.

    Used by ``commit_node`` rollback when a per-document commit fails partway
    through. Matches only the active batch; chunks from prior successful
    ingests have a different (or empty) staging_batch_id and are left alone.
    """
    span = tracer.start_span(
        "vector_store.delete_documents_by_staging_batch",
        {"staging_batch_id": staging_batch_id, "collection": collection},
    )
    col = client.collections.get(collection)
    where = Filter.by_property("staging_batch_id").equal(staging_batch_id)
    result = col.data.delete_many(where=where)
    deleted = getattr(result, "matches", 0) or 0
    span.set_attribute("deleted_count", deleted)
    span.end(status="ok")
    return deleted


def get_source_hash(
    client: weaviate.WeaviateClient,
    source_key: str,
    collection: str = WEAVIATE_COLLECTION_NAME,
) -> Optional[str]:
    """Return the source_hash recorded on existing chunks for ``source_key``.

    Used by Phase 1 to short-circuit re-ingest of unchanged content. Returns
    ``None`` if no chunks exist, the property is missing/empty, or the query
    fails (caller treats None as "no prior record" and proceeds with full
    ingest).
    """
    try:
        if not client.collections.exists(collection):
            return None
        col = client.collections.get(collection)
        response = col.query.fetch_objects(
            filters=Filter.by_property("source_key").equal(source_key),
            limit=1,
            return_properties=["source_hash"],
        )
        if not response.objects:
            return None
        value = response.objects[0].properties.get("source_hash") or ""
        return value or None
    except Exception:
        logger.debug("get_source_hash query failed for %s", source_key, exc_info=True)
        return None


def fetch_chunks_by_parent_section(
    client: weaviate.WeaviateClient,
    parent_section_ids: list[str],
    limit_per_section: int = 4,
    extra_filters: Optional[Filter] = None,
    collection: str = WEAVIATE_COLLECTION_NAME,
) -> list[dict]:
    """Fetch leaf chunks under each given parent_section_id (Stage 4c lift).

    One filtered fetch per parent_section_id, capped at ``limit_per_section``.
    Always restricts to ``node_kind="chunk"`` so section nodes never leak into
    lift results. Returns the same dict shape as ``hybrid_search``, but with
    score=0.0 (lift results are unscored — the reranker computes the score).

    Args:
        client: Live Weaviate client.
        parent_section_ids: Parent section ids to fetch siblings for. Empty
            strings are skipped.
        limit_per_section: Max siblings per parent section.
        extra_filters: Optional Filter combined via AND with the parent and
            node_kind clauses (e.g. tenant_id, source_filter).
        collection: Target collection.
    """
    span = tracer.start_span(
        "vector_store.fetch_chunks_by_parent_section",
        {"parent_count": len(parent_section_ids), "limit_per": limit_per_section},
    )
    try:
        if not client.collections.exists(collection):
            span.end(status="ok")
            return []
        col = client.collections.get(collection)
        documents: list[dict] = []
        seen_uuids: set[str] = set()
        for psid in parent_section_ids:
            if not psid:
                continue
            base = (
                Filter.by_property("parent_section_id").equal(psid)
                & Filter.by_property("node_kind").equal("chunk")
            )
            flt = base & extra_filters if extra_filters is not None else base
            response = col.query.fetch_objects(filters=flt, limit=limit_per_section)
            for obj in response.objects:
                obj_uuid = str(obj.uuid) if getattr(obj, "uuid", None) else ""
                if obj_uuid and obj_uuid in seen_uuids:
                    continue
                if obj_uuid:
                    seen_uuids.add(obj_uuid)
                documents.append({
                    "text": obj.properties.get("text", ""),
                    "metadata": {
                        "source": obj.properties.get("source", "unknown"),
                        "source_key": obj.properties.get("source_key", ""),
                        "source_uri": obj.properties.get("source_uri", ""),
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
                        "node_kind": "chunk",
                        "tree_level": obj.properties.get("tree_level", 0),
                        "heading_path": obj.properties.get("heading_path", []) or [],
                        "parent_section_id": obj.properties.get("parent_section_id", ""),
                        "child_count": obj.properties.get("child_count", 0),
                        "document_id": obj.properties.get("document_id", ""),
                    },
                    "score": 0.0,
                    "uuid": obj_uuid,
                })
        span.set_attribute("result_count", len(documents))
        span.end(status="ok")
        return documents
    except Exception:
        span.end(status="error")
        raise


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
