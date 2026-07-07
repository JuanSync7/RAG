# @summary
# Chunk-ROLE backfill tool for the nav-classify rollout (Slice D): tag chunk_role
# on the chunks ALREADY in the corpus — no re-ingest, no source files. Reuses the
# SAME shared LLM classifier (classify_roles_from_config) that ingest tagging uses
# (one prompt, one parser — CLI/UI parity), the store.py read idioms
# (iter_document_ids + the text-bearing fetch_chunks_by_document_id), and the
# in-place update-by-uuid write (update_chunk_role). NEVER drops a chunk and is
# FAIL-OPEN to default_role on any classifier error.
# Exports: backfill_roles_from_corpus
# Deps: src.ingest.common.role_classify, src.vector_db.weaviate.store
# @end-summary
"""Backfill ``chunk_role`` onto an existing chunk corpus (no re-ingest).

The nav-classify rollout replaced the ingest-time regex *drop* of navigational
chunks with an LLM *tag* (``chunk_role`` in {content, navigation, boilerplate});
the query side then *filters* by role. Chunks ingested before that change carry
no ``chunk_role`` (it reads as ``None`` / legacy, which the ``ne`` query filter
deliberately keeps). This tool labels those existing chunks in place so the
query-time role filter can take effect, WITHOUT re-running ingest or needing the
source files.

:func:`backfill_roles_from_corpus` is the pure-ish core:

1. Determine the documents to process — either the explicit ``document_ids`` or,
   when none are given, every distinct ``document_id`` via a server-side group-by
   aggregation (:func:`iter_document_ids`), optionally capped by ``limit``.
2. For each document, fetch ALL its chunks WITH text via the cursor-paginated
   :func:`fetch_chunks_by_document_id` (``include_text=True``), classify the page
   with the SHARED classifier (:func:`classify_roles_from_config` by default —
   the exact prompt/parser ingest uses), and update each chunk in place via
   :func:`update_chunk_role` (a single-property partial update, so re-runs are
   idempotent).

FAIL-OPEN is absolute: the shared classifier already resolves any error/parse
failure/unknown role to ``default_role`` ("content"); on top of that this tool
treats a classifier exception or a roles/chunks length mismatch as "tag every
chunk in the page ``default_role``" — a misclassification can only *keep* a chunk
in retrieval (it is never demoted to navigation/boilerplate on error), and the
function never raises for a single bad document or a single failed update.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from src.ingest.common.role_classify import classify_roles_sync
from src.vector_db.weaviate.store import (
    WEAVIATE_COLLECTION_NAME,
    fetch_chunks_by_document_id,
    iter_document_ids,
    update_chunk_role,
)

logger = logging.getLogger("rag.ingest.embedding.role_backfill")

# Role vocabulary mirrors the classifier prompt / settings. Kept as a module
# constant only to seed the per-run ``role_counts`` stat; the authoritative roles
# come from the classifier itself (which is the only authority for a role).
_ROLE_KEYS = ("content", "navigation", "boilerplate")


def _report(progress: Optional[Callable[[str], None]], message: str) -> None:
    """Emit a progress message to the optional callable and the debug log."""
    if progress is not None:
        try:
            progress(message)
        except Exception:  # pragma: no cover - progress must never break backfill
            logger.debug("progress callable raised", exc_info=True)
    logger.debug("%s", message)


def backfill_roles_from_corpus(
    client: Any,
    *,
    provider: Any = None,
    source_collection: str = WEAVIATE_COLLECTION_NAME,
    document_ids: Optional[list[str]] = None,
    limit: Optional[int] = None,
    fetch_page_size: int = 500,
    default_role: str = "content",
    dry_run: bool = False,
    classify_fn: Optional[Callable[[Any, list], list[str]]] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """Tag ``chunk_role`` on every chunk of the selected documents, in place.

    Args:
        client: Live Weaviate client (or a fake exposing the same read/update
            surface — ``collections.get(name).query.fetch_objects`` /
            ``.aggregate.over_all`` / ``.data.update``).
        provider: LLM provider passed to the classifier; defaults to the live
            router via :func:`classify_roles_sync`. Injectable for tests.
        source_collection: Collection holding the existing chunks.
        document_ids: When given, process exactly these documents; otherwise
            discover all distinct ``document_id`` values in ``source_collection``.
        limit: Optional cap on the number of documents processed (after
            enumeration). ``None`` means no cap.
        fetch_page_size: Cursor page size for the per-document chunk fetch.
        default_role: Fail-open role assigned to a whole page when the classifier
            raises or returns a mis-sized result. Must be the answer-bearing role
            ("content") so a failure can only KEEP a chunk, never drop it.
        dry_run: When True, classify + count but DO NOT write any role
            (``update_chunk_role`` is never called).
        classify_fn: Optional ``(provider, chunks) -> list[str]`` override;
            defaults to the config-driven synchronous SHARED classifier (same
            prompt/parser as ingest tagging). Injectable for tests.
        progress: Optional ``callable(str)`` for progress/log lines.

    Returns:
        A stats dict::

            {
                "documents_seen": int,    # documents enumerated/processed
                "chunks_seen": int,       # chunks fetched + classified
                "chunks_updated": int,    # chunks written (0 on dry-run)
                "role_counts": {role: count, ...},  # classified role tally
                "errors": list[str],      # per-document / per-chunk failures
            }

    The function never raises for a single bad document or a single failed
    update: failures are recorded in ``errors`` and processing continues, so a
    partial/interrupted run can simply be re-executed (the single-property
    ``chunk_role`` update is idempotent).
    """
    stats: dict[str, Any] = {
        "documents_seen": 0,
        "chunks_seen": 0,
        "chunks_updated": 0,
        "role_counts": {key: 0 for key in _ROLE_KEYS},
        "errors": [],
    }

    classify = classify_fn if classify_fn is not None else classify_roles_sync

    # 1) Resolve the document set: explicit ids win; else enumerate distinct ids.
    if document_ids is not None:
        targets: list[str] = [str(d) for d in document_ids if str(d)]
        _report(progress, f"role-backfill: processing {len(targets)} explicit document_id(s)")
    else:
        _report(progress, f"role-backfill: enumerating documents in {source_collection!r}")
        try:
            targets = list(iter_document_ids(client, collection=source_collection))
        except Exception as exc:  # enumeration failure is fatal — nothing to do.
            msg = f"document enumeration failed for {source_collection!r}: {exc!r}"
            logger.error(msg, exc_info=True)
            stats["errors"].append(msg)
            return stats
        _report(progress, f"role-backfill: discovered {len(targets)} document(s)")

    if limit is not None and limit >= 0:
        targets = targets[:limit]

    # 2) Per-document: fetch text-bearing chunks, classify the page, update roles.
    for doc_id in targets:
        stats["documents_seen"] += 1
        try:
            chunks = fetch_chunks_by_document_id(
                client,
                doc_id,
                page_size=fetch_page_size,
                collection=source_collection,
                include_text=True,
            )
        except Exception as exc:
            msg = f"document {doc_id!r} fetch failed: {exc!r}"
            logger.warning(msg, exc_info=True)
            stats["errors"].append(msg)
            continue

        if not chunks:
            continue
        stats["chunks_seen"] += len(chunks)

        # Classify the whole page with the SHARED classifier. The classifier is
        # itself fail-open; we additionally fail open here on an exception or a
        # length mismatch (the shared classifier guarantees 1:1 length, so a
        # mismatch only happens via an injected fake — handle it anyway).
        roles = _classify_page(classify, provider, chunks, default_role, doc_id)

        for chunk, role in zip(chunks, roles):
            role = role if isinstance(role, str) and role else default_role
            stats["role_counts"][role] = stats["role_counts"].get(role, 0) + 1

            uuid = chunk.get("uuid") if isinstance(chunk, dict) else getattr(chunk, "uuid", "")
            if not uuid:
                # Cannot address the chunk for an in-place update — skip (never
                # drop; the chunk simply keeps its existing/legacy role).
                continue

            if dry_run:
                continue

            try:
                update_chunk_role(
                    client, uuid, role=role, collection=source_collection
                )
                stats["chunks_updated"] += 1
            except Exception as exc:
                msg = f"chunk {uuid!r} (doc {doc_id!r}) role-update failed: {exc!r}"
                logger.warning(msg, exc_info=True)
                stats["errors"].append(msg)
                continue

    if dry_run:
        _report(
            progress,
            f"role-backfill: dry-run — classified {stats['chunks_seen']} chunk(s) "
            f"across {stats['documents_seen']} document(s); wrote nothing "
            f"(roles={stats['role_counts']})",
        )
    else:
        _report(
            progress,
            f"role-backfill: updated {stats['chunks_updated']}/{stats['chunks_seen']} "
            f"chunk(s) across {stats['documents_seen']} document(s) "
            f"(roles={stats['role_counts']})",
        )
    return stats


def _classify_page(
    classify: Callable[[Any, list], list[str]],
    provider: Any,
    chunks: list,
    default_role: str,
    doc_id: str,
) -> list[str]:
    """Classify one document's chunks, fail-open to ``default_role``.

    Any classifier exception, or a roles/chunks length mismatch, resolves the
    whole page to ``default_role`` — never raises, never drops. (The shared
    classifier already fails open internally; this is the tool-level guard.)
    """
    try:
        roles = classify(provider, chunks)
    except Exception as exc:  # noqa: BLE001 — fail open, never drop a chunk
        logger.warning(
            "role-backfill: classifier failed for document %r (%d chunk(s)): %s — "
            "defaulting all to %r",
            doc_id, len(chunks), exc, default_role,
        )
        return [default_role] * len(chunks)

    if not isinstance(roles, list) or len(roles) != len(chunks):
        logger.warning(
            "role-backfill: classifier length mismatch for document %r — got %s "
            "for %d chunk(s); defaulting all to %r",
            doc_id,
            len(roles) if isinstance(roles, list) else "non-list",
            len(chunks), default_role,
        )
        return [default_role] * len(chunks)
    return roles
