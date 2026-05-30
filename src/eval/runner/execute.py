# @summary
# Live pack ingest executor — bridges a pure IngestPlan into a real
# ingest_directory run, collects post-ingest collection stats, and
# returns a frozen IngestReport. First runner module with a Weaviate
# dependency; plan.py stays pure.
# Exports: execute_plan
# Deps: src.ingest.impl (ingest_directory), src.ingest.common.types
#       (IngestionConfig), src.vector_db (get_collection_stats,
#       create_persistent_client, close_client).
# @end-summary
"""Executor that runs a planned pack ingest against live Weaviate."""
from __future__ import annotations

from src.ingest.common.types import IngestionConfig
from src.ingest.impl import ingest_directory
from src.vector_db import close_client, create_persistent_client, get_collection_stats

from .plan import IngestPlan
from .report import IngestReport


def execute_plan(plan: IngestPlan, *, fresh: bool = True) -> IngestReport:
    """Execute ``plan`` against the real ingest pipeline and return a report.

    The default ``fresh=True`` recreates the target collection so each
    pack run is deterministic and isolated from prior state. ``update``
    is the explicit dual toggle — ``fresh=True`` implies ``update=False``
    (no incremental manifest merge), ``fresh=False`` implies ``update=True``.

    Post-ingest, the function queries ``get_collection_stats`` so the
    caller has the authoritative distinct-source count from Weaviate
    rather than trusting the run summary alone. The client is opened
    and closed locally; the caller need not manage the connection.
    """

    config = IngestionConfig(target_collection=plan.collection_name)
    update = not fresh

    summary = ingest_directory(
        documents_dir=plan.documents_dir,
        config=config,
        fresh=fresh,
        update=update,
        selected_sources=list(plan.doc_paths),
    )

    client = create_persistent_client()
    try:
        stats = get_collection_stats(client, collection=plan.collection_name)
    finally:
        try:
            close_client(client)
        except Exception:
            # Best-effort close; do not mask the stats outcome.
            pass

    if stats is None:
        raise RuntimeError(
            f"Post-ingest stats unavailable for collection {plan.collection_name}"
        )

    return IngestReport(
        collection_name=plan.collection_name,
        processed=summary.processed,
        skipped=summary.skipped,
        failed=summary.failed,
        stored_chunks=summary.stored_chunks,
        document_count=int(stats.get("document_count", 0)),
        plan=plan,
        errors=tuple(summary.errors),
    )
