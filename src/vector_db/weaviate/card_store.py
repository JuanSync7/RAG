# @summary
# Weaviate document-card collection store for RAPTOR-lite document routing.
# Exports: ensure_card_collection, add_document_cards, delete_document_cards
# Deps: weaviate-client, src.platform.observability
# @end-summary
"""Weaviate document-card collection store.

Manages the ``RAGDocumentCards`` collection — a small, routing-only index of
one compact "card" per ingested document (title + section headings + an
optional summary). It is deliberately DECOUPLED from the heavy chunk
``DocumentRecord`` path (``store.py``), exactly the way ``visual_store.py``
decouples the visual page collection: a second collection with its own
``ensure``/``add`` helpers and pre-computed (caller-supplied) vectors.

Cards are *never* sent to the LLM; they exist only to let the Stage-1 router
narrow a query to a small set of candidate documents before flat retrieval.

Idempotency: each card's object UUID is derived deterministically from its
``document_id`` via :func:`generate_uuid5` (a stdlib uuid5 helper that is
byte-for-byte identical to ``weaviate.util.generate_uuid5``). Inserting through
the batch path with an explicit UUID performs an insert-or-replace, so
re-ingesting a document REPLACES its card rather than duplicating it. This
mirrors the deterministic-UUID upsert approach used by ``store.add_documents``
(which passes ``uuid=`` to ``batch.add_object``).
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

import weaviate
from weaviate.classes.config import Configure, DataType, Property

from src.platform.observability import get_tracer

logger = logging.getLogger("rag.vector_db.weaviate.card_store")
tracer = get_tracer()


def generate_uuid5(identifier: Any, namespace: str = "") -> str:
    """Deterministic UUIDv5 for a card object, keyed by ``document_id``.

    Byte-for-byte identical to ``weaviate.util.generate_uuid5`` — it applies the
    same ``uuid5(NAMESPACE_DNS, str(namespace) + str(identifier))`` formula — but
    is implemented against the stdlib so it does not import ``weaviate.util``.
    (This module lives in the ``src.vector_db.weaviate`` subpackage, which can
    shadow the installed top-level ``weaviate`` name during submodule-import
    resolution; the test harness also stubs ``weaviate`` without ``weaviate.util``.)

    Re-using the same UUID across re-ingests makes the batch insert an
    insert-or-replace, so a document's card is REPLACED rather than duplicated.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(namespace) + str(identifier)))


# The card-record contract: 7 properties (the embedding vector is supplied
# separately and is NOT stored as a property). Kept as a module-level constant
# so the schema is the single source of truth shared by ``ensure``/``add``.
CARD_PROPERTY_NAMES: tuple[str, ...] = (
    "document_id",
    "title",
    "source",
    "section_headings",
    "card_text",
    "summary",
    "num_chunks",
)


def ensure_card_collection(
    client: weaviate.WeaviateClient,
    collection: str = "RAGDocumentCards",
) -> None:
    """Create the document-card collection if it does not exist (idempotent).

    Collection schema (vectorizer none — embeddings are pre-computed):
    - ``document_id`` (TEXT): stable pointer to the parent document.
    - ``title`` (TEXT): document title (first heading / metadata / source stem).
    - ``source`` (TEXT): originating source identifier.
    - ``section_headings`` (TEXT_ARRAY): ordered, de-duplicated section headings.
    - ``card_text`` (TEXT): the routing text that was embedded.
    - ``summary`` (TEXT): optional LLM summary; empty string when not generated.
    - ``num_chunks`` (INT): number of leaf chunks the document produced.

    Args:
        client: Weaviate client handle.
        collection: Collection name. Default: ``"RAGDocumentCards"``.
    """
    if client.collections.exists(collection):
        logger.debug("Card collection %r already exists — skipping creation.", collection)
        return

    properties = [
        Property(name="document_id", data_type=DataType.TEXT),
        Property(name="title", data_type=DataType.TEXT),
        Property(name="source", data_type=DataType.TEXT),
        Property(name="section_headings", data_type=DataType.TEXT_ARRAY),
        Property(name="card_text", data_type=DataType.TEXT),
        Property(name="summary", data_type=DataType.TEXT),
        Property(name="num_chunks", data_type=DataType.INT),
    ]

    client.collections.create(
        name=collection,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=properties,
    )
    logger.info("Created card collection %r with %d properties.", collection, len(properties))


def add_document_cards(
    client: weaviate.WeaviateClient,
    cards: list[dict[str, Any]],
    collection: str = "RAGDocumentCards",
) -> int:
    """Insert/replace document cards, each with its pre-computed vector.

    Each ``card`` dict must contain the 7 contract keys
    (:data:`CARD_PROPERTY_NAMES`) plus a ``"vector"`` key holding the embedding
    of ``card_text`` (a ``list[float]`` supplied by the caller). The vector is
    passed to Weaviate explicitly and is NOT written as a property.

    Idempotency: the object UUID is derived from ``document_id`` via
    :func:`generate_uuid5`, so re-ingesting a document replaces its existing
    card instead of creating a duplicate.

    Args:
        client: Weaviate client handle.
        cards: List of card dicts (contract keys + ``"vector"``).
        collection: Target collection name. Default: ``"RAGDocumentCards"``.

    Returns:
        Number of cards successfully inserted (total minus failed objects).
    """
    if not cards:
        return 0

    span = tracer.span(
        "vector_store.add_document_cards",
        {"count": len(cards), "collection": collection},
    )
    col = client.collections.get(collection)

    with col.batch.dynamic() as batch:
        for card in cards:
            vector = card["vector"]
            # Only the contract keys go into properties; the vector is passed
            # to Weaviate explicitly via the ``vector`` kwarg.
            properties = {
                key: card[key] for key in CARD_PROPERTY_NAMES if key in card
            }
            # Deterministic UUID → insert-or-replace by document_id.
            card_uuid = generate_uuid5(str(card["document_id"]))
            batch.add_object(properties=properties, vector=vector, uuid=card_uuid)

    failed = len(col.batch.failed_objects) if hasattr(col.batch, "failed_objects") else 0
    inserted = len(cards) - failed
    logger.info(
        "Card insert into %r: %d inserted, %d failed.",
        collection, inserted, failed,
    )
    span.set_attribute("inserted_count", inserted)
    span.end(status="ok")
    return inserted


def delete_document_cards(
    client: weaviate.WeaviateClient,
    document_ids: list[str],
    collection: str = "RAGDocumentCards",
) -> int:
    """Delete document cards for the given ``document_id`` values (rollback).

    Used by ``commit_node`` rollback when a per-document commit fails after the
    card write was attempted: it removes the just-written card(s) so a rolled
    back commit does not leave an orphan card behind.

    Deletion targets each card by its deterministic object UUID
    (:func:`generate_uuid5` of the ``document_id``) — the same UUID the card was
    inserted under by :func:`add_document_cards`. This mirrors the card store's
    own insert-or-replace-by-document_id model exactly (no heavier
    filter/batch-delete machinery than the node already relies on), and is
    best-effort per id: a missing card (already gone) is not an error.

    Args:
        client: Weaviate client handle.
        document_ids: Document UUIDs whose cards should be removed.
        collection: Target collection name. Default: ``"RAGDocumentCards"``.

    Returns:
        Number of cards successfully deleted.
    """
    # De-duplicate while preserving order; ignore empty ids.
    seen: set[str] = set()
    ids: list[str] = []
    for raw in document_ids or []:
        doc_id = str(raw or "")
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        ids.append(doc_id)

    if not ids:
        return 0

    span = tracer.span(
        "vector_store.delete_document_cards",
        {"count": len(ids), "collection": collection},
    )
    col = client.collections.get(collection)

    deleted = 0
    for doc_id in ids:
        card_uuid = generate_uuid5(doc_id)
        try:
            col.data.delete_by_id(card_uuid)
            deleted += 1
        except Exception:
            logger.warning(
                "Card delete failed for document_id=%r (uuid=%s) in %r.",
                doc_id, card_uuid, collection, exc_info=True,
            )

    logger.info("Deleted %d/%d document cards from %r.", deleted, len(ids), collection)
    span.set_attribute("deleted_count", deleted)
    span.end(status="ok")
    return deleted
