#!/usr/bin/env python3
# @summary
# Runnable wrapper for the chunk-ROLE backfill (nav-classify rollout, Slice D):
# tag chunk_role on the chunks ALREADY in the corpus with NO re-ingest, using the
# SAME shared LLM classifier ingest tagging uses (classify_roles_from_config — one
# prompt, one parser). Obtains a Weaviate client via the project's persistent
# client helper (src.vector_db.create_persistent_client / close_client) and the
# LLM provider via get_llm_provider(), then delegates to backfill_roles_from_corpus
# and prints the returned stats. --dry-run classifies + counts but writes nothing.
# Exports: main, _parse_args, _split_ids
# Deps: config.settings, src.vector_db, src.platform.llm,
#   src.ingest.embedding.common.role_backfill
# @end-summary
"""Backfill ``chunk_role`` onto the existing chunk corpus (no re-ingest).

The nav-classify rollout (Slices A-C) replaced the ingest-time regex *drop* of
navigational chunks with an LLM *tag* (``chunk_role``), and the query side now
*filters* by role. Chunks ingested before that change have no ``chunk_role`` (it
reads as legacy/``None``, which the ``ne`` query filter keeps). This script tags
those existing chunks in place so the role filter takes effect — WITHOUT
re-running ingest and WITHOUT the source files.

It uses the SAME shared classifier as ingest tagging
(:func:`classify_roles_from_config`), so the prompt/parser are never duplicated
(CLI/UI parity). It is FAIL-OPEN: any classifier error tags the affected chunks
``content`` (never navigation/boilerplate), so a misclassification can only keep
a chunk in retrieval, never silently drop it. The per-chunk ``chunk_role`` update
is a single-property partial write, so re-running is idempotent.

Prerequisite — schema migration (one-time, per collection):

    The ``chunk_role`` property must exist on the collection before it can be
    written. It is in ``TABLE_AWARE_PROPERTIES`` (Slice B), so the generic
    schema-migration script adds it to any legacy collection. Dry-run then apply::

        uv run python scripts/migrate_weaviate_table_schema.py \\
            --collection RAGDocuments --dry-run
        uv run python scripts/migrate_weaviate_table_schema.py \\
            --collection RAGDocuments

Usage::

    # Tag every chunk of every document in the default collection.
    uv run python scripts/backfill_chunk_roles.py

    # Dry-run (classify + count per role, but do NOT write).
    uv run python scripts/backfill_chunk_roles.py --dry-run

    # Restrict to specific documents, or cap how many documents to process.
    uv run python scripts/backfill_chunk_roles.py --document-ids docA,docB
    uv run python scripts/backfill_chunk_roles.py --limit 50

    # Override the source collection / per-fetch page size.
    uv run python scripts/backfill_chunk_roles.py \\
        --collection RAGDocuments --batch-size 200

Connection + classifier endpoints come from the environment (``RAG_WEAVIATE_*``,
``RAG_NAV_CLASSIFY_*``) exactly as the live ingest worker reads them — there is
no separate config here.
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
    RAG_NAV_CLASSIFY_BATCH_SIZE,
    RAG_NAV_ROLE_DEFAULT,
    VECTOR_COLLECTION_DEFAULT,
)
from src.ingest.embedding.common.role_backfill import backfill_roles_from_corpus
from src.platform.llm import get_llm_provider
from src.vector_db import close_client, create_persistent_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("rag.scripts.backfill_chunk_roles")


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
    """Parse CLI flags for the chunk-role backfill wrapper."""
    parser = argparse.ArgumentParser(
        description=(
            "Backfill chunk_role on existing corpus chunks via the shared LLM "
            "classifier (nav-classify rollout — no re-ingest)."
        ),
    )
    parser.add_argument(
        "--collection",
        default=VECTOR_COLLECTION_DEFAULT,
        help="Source chunk collection to tag in place (default: %(default)s).",
    )
    parser.add_argument(
        "--document-ids",
        default=None,
        help="Optional comma-separated document_ids to restrict the backfill to.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=RAG_NAV_CLASSIFY_BATCH_SIZE,
        help=(
            "Per-document chunk fetch page size — the max chunks handed to the "
            "classifier per page (the classifier sub-batches internally by "
            "RAG_NAV_CLASSIFY_BATCH_SIZE). Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of documents processed (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify + count per role, but do NOT write any chunk_role.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Wire up the client + LLM provider, run the backfill, and print stats."""
    args = _parse_args(argv)
    document_ids = _split_ids(args.document_ids)

    # Same router the ingest tagger uses; the shared classifier reads the
    # RAG_NAV_CLASSIFY_* settings for its alias/timeout/json-mode.
    provider = get_llm_provider()

    client = create_persistent_client()
    try:
        stats = backfill_roles_from_corpus(
            client,
            provider=provider,
            source_collection=args.collection,
            document_ids=document_ids,
            limit=args.limit,
            fetch_page_size=args.batch_size,
            default_role=RAG_NAV_ROLE_DEFAULT,
            dry_run=args.dry_run,
            progress=logger.info,
        )
    finally:
        close_client(client)

    print("=== chunk-role backfill stats ===")
    for key in ("documents_seen", "chunks_seen", "chunks_updated"):
        print(f"{key}: {stats.get(key)}")
    role_counts = stats.get("role_counts") or {}
    print("role_counts:")
    for role, count in role_counts.items():
        print(f"  {role}: {count}")
    errors = stats.get("errors") or []
    print(f"errors: {len(errors)}")
    for err in errors:
        print(f"  - {err}")
    if args.dry_run:
        print("(dry-run — no chunk_role was written)")

    # Non-zero exit only when nothing was processed AND an error occurred, so a
    # resumable re-run after transient failures is still possible.
    return 0 if (stats.get("documents_seen") or not errors) else 1


if __name__ == "__main__":
    sys.exit(main())
