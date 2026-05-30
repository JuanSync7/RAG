# @summary
# Public surface of the eval runner subpackage.
# Exports: IngestPlan, IngestReport, EvalReport, RetrievalResults,
#          QueryRetrievalResult, execute_plan, plan_pack_ingest,
#          retrieve_for_goldens, recall_at_k, aggregate_recall_by_qtype
# Deps: .plan (pure), .report (frozen dataclasses), .execute (ingest +
#       vector_db), .retrieve (vector_db search + embeddings), .metrics
#       (pure recall@k).
# @end-summary
"""Eval runner — pack-to-ingest orchestration + retrieval metrics.

P3.0 contributes the pure-logic ``plan_pack_ingest`` / ``IngestPlan``.
P3 layers on ``execute_plan`` + ``IngestReport`` for the first live
Weaviate-backed run. P4 adds ``retrieve_for_goldens`` +
``recall_at_k`` + ``aggregate_recall_by_qtype`` + ``EvalReport``.
"""
from __future__ import annotations

from .execute import execute_plan
from .metrics import aggregate_recall_by_qtype, recall_at_k
from .plan import IngestPlan, plan_pack_ingest
from .report import EvalReport, IngestReport
from .retrieve import QueryRetrievalResult, RetrievalResults, retrieve_for_goldens

__all__ = [
    "EvalReport",
    "IngestPlan",
    "IngestReport",
    "QueryRetrievalResult",
    "RetrievalResults",
    "aggregate_recall_by_qtype",
    "execute_plan",
    "plan_pack_ingest",
    "recall_at_k",
    "retrieve_for_goldens",
]
