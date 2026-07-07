#!/usr/bin/env python3
# @summary
# Runnable wrapper for the RAPTOR-lite document-card backfill (DESIGN §9 step 1):
# populate RAGDocumentCards from the chunks already in RAGDocuments with NO
# re-ingest. Obtains a Weaviate client via the project's persistent client helper
# (src.vector_db.create_persistent_client / close_client) and the corpus embedder
# via get_embedding_provider(tier="ingest") — so inside rag-ingest-worker-dev it
# picks up RAG_TEI_EMBED_URL / RAG_DOCUMENT_CARD_COLLECTION from the container env
# and embeds with the same model as the corpus. Delegates all work to
# backfill_cards_from_corpus and prints the returned stats.
# Exports: main, _parse_args, _split_ids
# Deps: config.settings, src.vector_db, src.core.embeddings,
#   src.ingest.embedding.common.card_backfill
# @end-summary
"""Backfill the document-card index from the existing chunk corpus (no re-ingest).

The card index (``RAGDocumentCards``) is built purely from the title + heading
structure already present on chunks in ``RAGDocuments``; the source files are not
needed. Card vectors are produced with :func:`get_embedding_provider` (tier
``ingest``) so they match the corpus/query embedder, and written via the
idempotent ``uuid5(document_id)`` upsert in the card store — re-running refreshes
cards in place.

Usage::

    # Backfill every document in the default collection.
    uv run python scripts/backfill_document_cards.py

    # Dry-run (build + embed + count, but do NOT write).
    uv run python scripts/backfill_document_cards.py --dry-run

    # Restrict to specific documents.
    uv run python scripts/backfill_document_cards.py --document-ids docA,docB

    # Override collections / caps.
    uv run python scripts/backfill_document_cards.py \\
        --source-collection RAGDocuments \\
        --card-collection RAGDocumentCards \\
        --max-headings 60 --embed-batch-size 64

Connection + endpoints come from the environment (``RAG_WEAVIATE_*``,
``RAG_TEI_EMBED_URL``, ``RAG_DOCUMENT_CARD_COLLECTION``) exactly as the live
ingest worker reads them — there is no separate config here.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Allow imports from project root when run as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    RAG_DOCUMENT_CARD_COLLECTION,
    RAG_INGESTION_CARD_MAX_HEADINGS,
    VECTOR_COLLECTION_DEFAULT,
)
from src.core.embeddings import get_embedding_provider
from src.ingest.embedding.common.card_backfill import backfill_cards_from_corpus
from src.vector_db import close_client, create_persistent_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("rag.scripts.backfill_document_cards")


def _split_ids(raw: Optional[str]) -> Optional[list[str]]:
    """Parse a comma-separated ``--document-ids`` value into a clean list.

    Returns ``None`` when no ids were supplied (so the core function falls back
    to enumerating the whole collection); otherwise a de-blanked, stripped list.
    """
    if not raw:
        return None
    ids = [part.strip() for part in raw.split(",")]
    ids = [i for i in ids if i]
    return ids or None


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI flags for the card-backfill wrapper."""
    parser = argparse.ArgumentParser(
        description=(
            "Backfill RAGDocumentCards from existing RAGDocuments chunks "
            "(RAPTOR-lite document-routing rollout step 1 — no re-ingest)."
        ),
    )
    parser.add_argument(
        "--card-collection",
        default=RAG_DOCUMENT_CARD_COLLECTION,
        help="Target card collection (default: %(default)s).",
    )
    parser.add_argument(
        "--source-collection",
        default=VECTOR_COLLECTION_DEFAULT,
        help="Source chunk collection to read from (default: %(default)s).",
    )
    parser.add_argument(
        "--document-ids",
        default=None,
        help="Optional comma-separated document_ids to restrict the backfill to.",
    )
    parser.add_argument(
        "--max-headings",
        type=int,
        default=RAG_INGESTION_CARD_MAX_HEADINGS,
        help="Max section headings per card (§11 cap; default: %(default)s).",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=64,
        help="Card texts are embedded in batches of this size (default: %(default)s).",
    )
    parser.add_argument(
        "--fetch-page-size",
        type=int,
        default=500,
        help="Cursor page size for the per-document chunk fetch (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build + embed + count cards but do NOT write them.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Wire up the client + embedder, run the backfill, and print stats."""
    args = _parse_args(argv)
    document_ids = _split_ids(args.document_ids)

    # Same embedder as the ingest worker: tier="ingest" pins to the CPU pool so
    # the GPU stays free for query embedding. This keeps card vectors in the same
    # space as the corpus/query embedder.
    embedder = get_embedding_provider(tier="ingest")

    client = create_persistent_client()
    try:
        stats = backfill_cards_from_corpus(
            client,
            embedder,
            source_collection=args.source_collection,
            card_collection=args.card_collection,
            document_ids=document_ids,
            max_headings=args.max_headings,
            embed_batch_size=args.embed_batch_size,
            fetch_page_size=args.fetch_page_size,
            dry_run=args.dry_run,
            progress=logger.info,
        )
    finally:
        close_client(client)

    print("=== card backfill stats ===")
    for key in ("documents_seen", "cards_built", "cards_written", "skipped_empty"):
        print(f"{key}: {stats.get(key)}")
    errors = stats.get("errors") or []
    print(f"errors: {len(errors)}")
    for err in errors:
        print(f"  - {err}")
    if args.dry_run:
        print("(dry-run — no cards were written)")

    # Non-zero exit only when nothing was processed AND an error occurred, so a
    # resumable re-run after transient failures is still possible.
    return 0 if (stats.get("documents_seen") or not errors) else 1


if __name__ == "__main__":
    sys.exit(main())
