# @summary
# Frozen report dataclasses + builder: IngestReport (P3, post-ingest counts),
# EvalReport (P4 + P5, aggregated retrieval recall@k + faithfulness),
# build_eval_report (P5 helper that wires retrieval + faithfulness aggregates).
# Exports: IngestReport, EvalReport, build_eval_report
# Deps: stdlib (dataclasses, typing); src.eval.runner.plan (IngestPlan);
#       src.eval.runner.faithfulness (aggregate_faithfulness_by_qtype) — lazy.
# @end-summary
"""Frozen reports for ingest execution (P3) and retrieval/faithfulness eval
(P4 + P5)."""
from __future__ import annotations

from dataclasses import dataclass, field
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
    """Immutable summary of a retrieval recall@k + faithfulness evaluation.

    Holds the aggregated outcome of ``retrieve_for_goldens`` +
    ``aggregate_recall_by_qtype`` (P4) over a given collection at top-k,
    extended in P5 with ``faithfulness_by_qtype`` aggregated from
    ``score_goldens``. The faithfulness fields default to empty so P4
    constructions remain valid without modification.
    """

    collection_name: str
    k: int
    per_query_recall: Mapping[str, float]
    recall_by_qtype: Mapping[str, float]
    total_queries_scored: int
    total_queries_skipped: int
    faithfulness_by_qtype: Mapping[str, float] = field(default_factory=dict)
    total_queries_judged: int = 0


def build_eval_report(
    retrieval_results,
    faithfulness_results,
    recall_by_qtype: Mapping[str, float],
    per_query_recall: Mapping[str, float],
) -> EvalReport:
    """Wire P4 retrieval aggregates + P5 faithfulness aggregates into one report.

    Aggregates the faithfulness scores per qtype using the goldens *implied
    by* ``faithfulness_results.per_query`` so the helper can run without a
    direct goldens mapping. The faithfulness qtype rollup mirrors
    ``aggregate_faithfulness_by_qtype``'s contract (omit empty qtypes).
    """
    # Build a goldens-like view: group judged qids by qtype.
    grouped: dict[str, list] = {}
    for qid, qres in faithfulness_results.per_query.items():
        grouped.setdefault(qres.qtype, []).append(qres)

    import statistics as _stats

    faith_by_qtype: dict[str, float] = {}
    for qtype, items in grouped.items():
        if items:
            faith_by_qtype[qtype] = _stats.mean(i.score for i in items)

    return EvalReport(
        collection_name=retrieval_results.collection_name,
        k=retrieval_results.k,
        per_query_recall=dict(per_query_recall),
        recall_by_qtype=dict(recall_by_qtype),
        total_queries_scored=len(retrieval_results.per_query),
        total_queries_skipped=faithfulness_results.total_queries_skipped,
        faithfulness_by_qtype=faith_by_qtype,
        total_queries_judged=faithfulness_results.total_queries_scored,
    )
