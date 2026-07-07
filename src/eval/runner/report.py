# @summary
# Frozen report dataclasses + builder: IngestReport (P3, post-ingest counts),
# EvalReport (P4 + P5 + P7e + P7f, aggregated retrieval recall@k + faithfulness
# + MRR, plus per-query recall/mrr/faithfulness maps for per-qid gating),
# build_eval_report (helper wiring retrieval + faithfulness + MRR aggregates
# and the per-query maps).
# Exports: IngestReport, EvalReport, build_eval_report
# Deps: stdlib (dataclasses, typing); src.eval.runner.plan (IngestPlan);
#       src.eval.runner.faithfulness (aggregate_faithfulness_by_qtype) — lazy.
# @end-summary
"""Frozen reports for ingest execution (P3) and retrieval/faithfulness eval
(P4 + P5 + P7e MRR)."""
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
    ``score_goldens`` and in P7e with ``mrr_by_qtype`` (mean reciprocal
    rank, same source-path predicate as recall). P7f adds per-query maps
    (``per_query_mrr``, ``per_query_faithfulness``) that mirror the existing
    ``per_query_recall`` so the gate can enforce per-qid threshold overrides.
    The faithfulness, MRR, and per-query-map fields default to empty so
    earlier constructions remain valid without modification.
    """

    collection_name: str
    k: int
    per_query_recall: Mapping[str, float]
    recall_by_qtype: Mapping[str, float]
    total_queries_scored: int
    total_queries_skipped: int
    faithfulness_by_qtype: Mapping[str, float] = field(default_factory=dict)
    total_queries_judged: int = 0
    mrr_by_qtype: Mapping[str, float] = field(default_factory=dict)
    per_query_mrr: Mapping[str, float] = field(default_factory=dict)
    per_query_faithfulness: Mapping[str, float] = field(default_factory=dict)


def build_eval_report(
    retrieval_results,
    faithfulness_results,
    recall_by_qtype: Mapping[str, float],
    per_query_recall: Mapping[str, float],
    mrr_by_qtype: Mapping[str, float] | None = None,
    per_query_mrr: Mapping[str, float] | None = None,
) -> EvalReport:
    """Wire P4 retrieval + P5 faithfulness + P7e MRR aggregates into one report.

    Aggregates the faithfulness scores per qtype using the goldens *implied
    by* ``faithfulness_results.per_query`` so the helper can run without a
    direct goldens mapping. The faithfulness qtype rollup mirrors
    ``aggregate_faithfulness_by_qtype``'s contract (omit empty qtypes).

    ``mrr_by_qtype`` is computed by the caller (via
    ``aggregate_mrr_by_qtype``) and threaded through verbatim; ``None``
    defaults to an empty mapping so existing callers stay valid.

    P7f per-query maps (for per-qid gating):

    - ``per_query_mrr`` is computed by the caller (mirroring
      ``per_query_recall``) and threaded through; ``None`` → empty.
    - ``per_query_faithfulness`` is derived INTERNALLY from
      ``faithfulness_results.per_query`` (``{qid: float(score)}``).
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

    # Per-query faithfulness (for per-qid gating): one score per judged query.
    per_query_faithfulness: dict[str, float] = {
        qid: float(qres.score)
        for qid, qres in faithfulness_results.per_query.items()
    }

    return EvalReport(
        collection_name=retrieval_results.collection_name,
        k=retrieval_results.k,
        per_query_recall=dict(per_query_recall),
        recall_by_qtype=dict(recall_by_qtype),
        total_queries_scored=len(retrieval_results.per_query),
        total_queries_skipped=faithfulness_results.total_queries_skipped,
        faithfulness_by_qtype=faith_by_qtype,
        total_queries_judged=faithfulness_results.total_queries_scored,
        mrr_by_qtype=dict(mrr_by_qtype) if mrr_by_qtype is not None else {},
        per_query_mrr=dict(per_query_mrr) if per_query_mrr is not None else {},
        per_query_faithfulness=per_query_faithfulness,
    )
