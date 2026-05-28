# @summary
# Public surface of the eval runner subpackage.
# Exports: IngestPlan, IngestReport, EvalReport, RetrievalResults,
#          QueryRetrievalResult, JudgeClient, JudgeQuestion, JudgmentScore,
#          execute_plan, plan_pack_ingest, retrieve_for_goldens, recall_at_k,
#          aggregate_recall_by_qtype, build_judge_client, load_judge_prompt,
#          render_judge_prompt.
# Deps: .plan (pure), .report (frozen dataclasses), .execute (ingest +
#       vector_db), .retrieve (vector_db search + embeddings), .metrics
#       (pure recall@k), .judge (pluggable judge contract; P5.0).
# @end-summary
"""Eval runner — pack-to-ingest orchestration + retrieval metrics + judge.

P3.0 contributes the pure-logic ``plan_pack_ingest`` / ``IngestPlan``.
P3 layers on ``execute_plan`` + ``IngestReport`` for the first live
Weaviate-backed run. P4 adds ``retrieve_for_goldens`` +
``recall_at_k`` + ``aggregate_recall_by_qtype`` + ``EvalReport``. P5.0
adds the pluggable ``JudgeClient`` + prompt loader contract used by
P5's faithfulness executor.
"""
from __future__ import annotations

from .execute import execute_plan
from .judge import (
    JudgeClient,
    JudgeQuestion,
    JudgmentScore,
    build_judge_client,
    load_judge_prompt,
    render_judge_prompt,
)
from .metrics import aggregate_recall_by_qtype, recall_at_k
from .plan import IngestPlan, plan_pack_ingest
from .report import EvalReport, IngestReport
from .retrieve import QueryRetrievalResult, RetrievalResults, retrieve_for_goldens

__all__ = [
    "EvalReport",
    "IngestPlan",
    "IngestReport",
    "JudgeClient",
    "JudgeQuestion",
    "JudgmentScore",
    "QueryRetrievalResult",
    "RetrievalResults",
    "aggregate_recall_by_qtype",
    "build_judge_client",
    "execute_plan",
    "load_judge_prompt",
    "plan_pack_ingest",
    "recall_at_k",
    "render_judge_prompt",
    "retrieve_for_goldens",
]
