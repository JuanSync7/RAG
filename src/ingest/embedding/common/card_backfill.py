# @summary
# Card-backfill tool for the RAPTOR-lite document-routing feature
# (DOCUMENT_ROUTING_DESIGN.md §9 rollout step 1): populate the RAGDocumentCards
# index from the chunks ALREADY in RAGDocuments — no re-ingest, no source files.
# Reuses build_document_card (baseline, no LLM) for the card record, the
# store.py read idioms (iter_document_ids + fetch_chunks_by_document_id, both
# server-side group-by / cursor-paginated for a 475k-object collection), and the
# card_store helpers (ensure_card_collection + add_document_cards) for the write.
# Exports: backfill_cards_from_corpus
# Deps: src.ingest.embedding.common.card_builder, src.vector_db.weaviate.store,
#   src.vector_db.weaviate.card_store
# @end-summary
"""Backfill document cards from an existing chunk corpus (no re-ingest).

DOCUMENT_ROUTING_DESIGN.md §9 rollout step 1 is "build doc cards from EXISTING
title+heading nodes — no re-ingest." The live dev corpus already holds every
chunk in ``RAGDocuments`` (carrying ``document_id`` / ``title`` / ``source`` /
``heading_path`` / ``node_kind``), but the source PDFs are not on the box, so
the card index must be populated from those chunks rather than by re-running the
ingest pipeline.

:func:`backfill_cards_from_corpus` is the pure-ish core:

1. Determine the documents to process — either the explicit ``document_ids`` or,
   when none are given, every distinct ``document_id`` discovered via a
   server-side group-by aggregation (:func:`iter_document_ids`).
2. For each document, fetch ALL its chunks with a cursor-paginated read
   (:func:`fetch_chunks_by_document_id`) and build the §6.4 baseline card via
   :func:`build_document_card` (no LLM). Cards with empty ``card_text`` are
   skipped.
3. Batch-embed every ``card_text`` with the SAME embedder the corpus was
   embedded with (so card vectors live in the same space as the query embedder),
   attach each vector, then ``ensure_card_collection`` once and
   ``add_document_cards`` (idempotent upsert by ``uuid5(document_id)`` — re-running
   refreshes a card in place rather than duplicating it).

The function never raises on a single bad document: per-document failures are
collected into ``errors`` and processing continues, so the tool is resumable and
idempotent. It performs no card logic of its own (delegated to
``build_document_card``) and no card-write logic of its own (delegated to
``card_store``).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from src.ingest.embedding.common.card_builder import build_document_card
from src.vector_db.weaviate.card_store import (
    add_document_cards,
    ensure_card_collection,
)
from src.vector_db.weaviate.store import (
    fetch_chunks_by_document_id,
    iter_document_ids,
)

logger = logging.getLogger("rag.ingest.embedding.card_backfill")


def _report(progress: Optional[Callable[[str], None]], message: str) -> None:
    """Emit a progress message to the optional callable and the debug log."""
    if progress is not None:
        try:
            progress(message)
        except Exception:  # pragma: no cover - progress must never break backfill
            logger.debug("progress callable raised", exc_info=True)
    logger.debug("%s", message)


def backfill_cards_from_corpus(
    client: Any,
    embedder: Any,
    *,
    source_collection: str = "RAGDocuments",
    card_collection: str = "RAGDocumentCards",
    document_ids: Optional[list[str]] = None,
    max_headings: int = 60,
    embed_batch_size: int = 64,
    fetch_page_size: int = 500,
    dry_run: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """Build + write one routing card per document from the existing corpus.

    Args:
        client: Live Weaviate client (or a fake exposing the same read surface).
        embedder: Embedding provider exposing ``embed_documents(list[str]) ->
            list[list[float]]`` — MUST be the same model used to embed the
            corpus so card vectors share the query embedder's space.
        source_collection: Collection holding the existing chunks
            (default ``"RAGDocuments"``).
        card_collection: Target card collection (default ``"RAGDocumentCards"``).
        document_ids: When given, process exactly these documents; otherwise
            discover all distinct ``document_id`` values in ``source_collection``.
        max_headings: §11 heading cap forwarded to ``build_document_card``.
        embed_batch_size: Card texts are embedded in batches of this size.
        fetch_page_size: Cursor page size for the per-document chunk fetch.
        dry_run: When True, build + embed + count cards but DO NOT write them
            (``ensure_card_collection`` / ``add_document_cards`` are not called).
        progress: Optional ``callable(str)`` for progress/log lines.

    Returns:
        A stats dict::

            {
                "documents_seen": int,   # documents enumerated/processed
                "cards_built": int,      # non-empty cards constructed
                "cards_written": int,    # cards reported written by the store
                "skipped_empty": int,    # documents whose card_text was empty
                "errors": list[str],     # per-document failure descriptions
            }

    The function never raises for a single bad document; it records the failure
    in ``errors`` and continues, so a partial/interrupted run can simply be
    re-executed (the deterministic ``uuid5(document_id)`` upsert makes re-writes
    idempotent).
    """
    stats: dict[str, Any] = {
        "documents_seen": 0,
        "cards_built": 0,
        "cards_written": 0,
        "skipped_empty": 0,
        "errors": [],
    }

    # 1) Resolve the document set: explicit ids win; else enumerate distinct ids.
    if document_ids is not None:
        targets: list[str] = [str(d) for d in document_ids if str(d)]
        _report(progress, f"backfill: processing {len(targets)} explicit document_id(s)")
    else:
        _report(progress, f"backfill: enumerating documents in {source_collection!r}")
        try:
            targets = list(iter_document_ids(client, collection=source_collection))
        except Exception as exc:  # enumeration failure is fatal — nothing to do.
            msg = f"document enumeration failed for {source_collection!r}: {exc!r}"
            logger.error(msg, exc_info=True)
            stats["errors"].append(msg)
            return stats
        _report(progress, f"backfill: discovered {len(targets)} document(s)")

    # 2) Per-document: fetch chunks, build the baseline card. Errors are isolated.
    cards: list[dict] = []
    for doc_id in targets:
        stats["documents_seen"] += 1
        try:
            chunks = fetch_chunks_by_document_id(
                client,
                doc_id,
                page_size=fetch_page_size,
                collection=source_collection,
            )
            card = build_document_card(
                chunks,
                max_headings=max_headings,
                with_summary=False,
                summary="",
            )
            if not card.get("card_text"):
                stats["skipped_empty"] += 1
                _report(progress, f"backfill: skip {doc_id!r} (empty card_text)")
                continue
            cards.append(card)
            stats["cards_built"] += 1
        except Exception as exc:
            msg = f"document {doc_id!r} failed: {exc!r}"
            logger.warning(msg, exc_info=True)
            stats["errors"].append(msg)
            continue

    if not cards:
        _report(progress, "backfill: no cards to write")
        return stats

    # 3) Batch-embed every card_text with the corpus embedder, attach vectors.
    _report(progress, f"backfill: embedding {len(cards)} card text(s)")
    for start in range(0, len(cards), max(1, int(embed_batch_size))):
        batch = cards[start: start + max(1, int(embed_batch_size))]
        texts = [c["card_text"] for c in batch]
        try:
            vectors = embedder.embed_documents(texts)
        except Exception as exc:
            msg = f"embedding batch at offset {start} failed: {exc!r}"
            logger.error(msg, exc_info=True)
            stats["errors"].append(msg)
            # Drop this batch's cards from the write set (no vector to attach).
            for c in batch:
                c["_embed_failed"] = True
            continue
        for card, vector in zip(batch, vectors):
            card["vector"] = vector

    writable = [c for c in cards if "vector" in c and not c.get("_embed_failed")]
    if not writable:
        _report(progress, "backfill: no embedded cards to write")
        return stats

    # 4) Write (unless dry-run): ensure the collection once, then upsert cards.
    if dry_run:
        _report(progress, f"backfill: dry-run — would write {len(writable)} card(s)")
        return stats

    try:
        ensure_card_collection(client, card_collection)
        written = add_document_cards(client, writable, collection=card_collection)
        stats["cards_written"] = int(written or 0)
        _report(progress, f"backfill: wrote {stats['cards_written']} card(s)")
    except Exception as exc:
        msg = f"card write to {card_collection!r} failed: {exc!r}"
        logger.error(msg, exc_info=True)
        stats["errors"].append(msg)

    return stats
