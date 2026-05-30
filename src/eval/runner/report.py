# @summary
# Frozen report dataclasses: IngestReport (P3, post-ingest counts) and
# EvalReport (P4, aggregated retrieval recall@k outcome).
# Exports: IngestReport, EvalReport
# Deps: stdlib (dataclasses, typing); src.eval.runner.plan (IngestPlan).
# @end-summary
"""Frozen reports for ingest execution (P3) and retrieval eval (P4)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

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


@dataclass(frozen=True)
class EvalReport:
    """Immutable summary of a retrieval recall@k evaluation pass.

    Holds the aggregated outcome of ``retrieve_for_goldens`` +
    ``aggregate_recall_by_qtype`` over a given collection at top-k.
    """

    collection_name: str
    k: int
    per_query_recall: Mapping[str, float]
    recall_by_qtype: Mapping[str, float]
    total_queries_scored: int
    total_queries_skipped: int
