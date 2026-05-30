# @summary
# Frozen IngestReport dataclass — return value of execute_plan, capturing
# post-ingest counts, collection routing, and back-reference to the plan.
# Exports: IngestReport
# Deps: stdlib (dataclasses); src.eval.runner.plan (IngestPlan).
# @end-summary
"""Frozen report describing the outcome of a pack ingest execution."""
from __future__ import annotations

from dataclasses import dataclass

from .plan import IngestPlan


@dataclass(frozen=True)
class IngestReport:
    """Immutable summary returned by ``execute_plan``.

    Mirrors the run-level counts from ``IngestionRunSummary`` and pairs
    them with the post-ingest ``document_count`` from Weaviate, the
    target collection name, and a back-reference to the originating
    ``IngestPlan`` so downstream consumers (P4 metrics, P5 judge) can
    read the just-ingested corpus without re-deriving routing.
    """

    collection_name: str
    processed: int
    skipped: int
    failed: int
    stored_chunks: int
    document_count: int
    plan: IngestPlan
    errors: tuple[str, ...]
