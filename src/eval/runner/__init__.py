# @summary
# Public surface of the eval runner subpackage.
# Exports: IngestPlan, IngestReport, EvalReport, RetrievalResults,
#          QueryRetrievalResult, QueryFaithfulnessResult, FaithfulnessResults,
#          JudgeClient, JudgeQuestion, JudgmentScore, execute_plan,
#          plan_pack_ingest, retrieve_for_goldens, recall_at_k,
#          aggregate_recall_by_qtype, aggregate_faithfulness_by_qtype,
#          score_goldens, build_eval_report, build_judge_client,
#          load_judge_prompt, render_judge_prompt, read_samples_per_claim,
#          read_max_parallel_judges.
# Deps: .plan (pure), .report (frozen dataclasses + build_eval_report),
#       .execute (ingest + vector_db), .retrieve (vector_db search +
#       embeddings), .metrics (pure recall@k), .judge (pluggable judge
#       contract; P5.0), .faithfulness (P5 executor).
# @end-summary
"""Eval runner — pack-to-ingest orchestration + retrieval metrics + judge
+ faithfulness scoring.

P3.0 contributes the pure-logic ``plan_pack_ingest`` / ``IngestPlan``.
P3 layers on ``execute_plan`` + ``IngestReport`` for the first live
Weaviate-backed run. P4 adds ``retrieve_for_goldens`` +
``recall_at_k`` + ``aggregate_recall_by_qtype`` + ``EvalReport``. P5.0
adds the pluggable ``JudgeClient`` + prompt loader contract. P5 wires
the executor (``score_goldens`` + ``aggregate_faithfulness_by_qtype`` +
``build_eval_report``).
"""
from __future__ import annotations

from .execute import execute_plan
from .gate import GateFailure, GateResult, validate_eval_report
from .faithfulness import (
    FaithfulnessResults,
    QueryFaithfulnessResult,
    aggregate_faithfulness_by_qtype,
    score_goldens,
)
from .judge import (
    JudgeClient,
    JudgeQuestion,
    JudgmentScore,
    build_judge_client,
    load_judge_prompt,
    read_max_parallel_judges,
    read_samples_per_claim,
    render_judge_prompt,
)
from .metrics import aggregate_recall_by_qtype, recall_at_k
from .plan import IngestPlan, plan_pack_ingest
from .report import EvalReport, IngestReport, build_eval_report
from .retrieve import QueryRetrievalResult, RetrievalResults, retrieve_for_goldens

__all__ = [
    "EvalReport",
    "FaithfulnessResults",
    "GateFailure",
    "GateResult",
    "IngestPlan",
    "IngestReport",
    "JudgeClient",
    "JudgeQuestion",
    "JudgmentScore",
    "QueryFaithfulnessResult",
    "QueryRetrievalResult",
    "RetrievalResults",
    "aggregate_faithfulness_by_qtype",
    "aggregate_recall_by_qtype",
    "build_eval_report",
    "build_judge_client",
    "execute_plan",
    "load_judge_prompt",
    "plan_pack_ingest",
    "read_max_parallel_judges",
    "read_samples_per_claim",
    "recall_at_k",
    "render_judge_prompt",
    "retrieve_for_goldens",
    "score_goldens",
    "validate_eval_report",
]
