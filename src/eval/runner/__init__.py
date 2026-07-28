# @summary
# Public surface of the eval runner subpackage.
# Exports: IngestPlan, IngestReport, EvalReport, RetrievalResults,
#          QueryRetrievalResult, QueryFaithfulnessResult, FaithfulnessResults,
#          JudgeClient, JudgeQuestion, JudgmentScore, execute_plan,
#          plan_pack_ingest, retrieve_for_goldens, recall_at_k,
#          aggregate_recall_by_qtype, reciprocal_rank, aggregate_mrr_by_qtype,
#          aggregate_faithfulness_by_qtype,
#          score_goldens, build_eval_report, build_judge_client,
#          load_judge_prompt, render_judge_prompt, read_samples_per_claim,
#          read_max_parallel_judges, render_ci_summary, persist_eval_report,
#          AlertPayload, send_alert, CalibrationResult, CalibrationRecord,
#          CalibrationAlert, cohens_kappa, binarize_score, compute_calibration,
#          detect_drift, persist_calibration_record, send_calibration_alert,
#          CalibrationExample, load_calibration_fixture,
#          DEFAULT_CALIBRATION_FIXTURE_PATH, CALIBRATION_FIXTURE_SHA,
#          run_calibration.
# Deps: .plan (pure), .report (frozen dataclasses + build_eval_report),
#       .execute (ingest + vector_db), .retrieve (vector_db search +
#       embeddings), .metrics (pure recall@k), .judge (pluggable judge
#       contract; P5.0), .faithfulness (P5 executor), .calibration (P9
#       judge-calibration: Cohen's kappa + drift alert), .calibration_fixture
#       (P5.0 tamper-evident ground-truth calibration fixture + loader),
#       .calibration_loop (P9 one-shot run_calibration orchestrator).
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

from .alert import AlertPayload, send_alert
from .baseline import BaselineDiff, MetricDelta, compute_baseline_diff
from .calibration import (
    CalibrationAlert,
    CalibrationRecord,
    CalibrationResult,
    binarize_score,
    cohens_kappa,
    compute_calibration,
    detect_drift,
    persist_calibration_record,
    send_calibration_alert,
)
from .calibration_fixture import (
    CALIBRATION_FIXTURE_SHA,
    DEFAULT_CALIBRATION_FIXTURE_PATH,
    CalibrationExample,
    load_calibration_fixture,
)
from .calibration_loop import run_calibration
from .ci import render_ci_summary
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
from .judge_cli import (
    ClaudeCliJudgeClient,
    JudgeBackendError,
    build_claude_cli_judge_client,
)
from .metrics import (
    aggregate_mrr_by_qtype,
    aggregate_recall_by_qtype,
    recall_at_k,
    reciprocal_rank,
)
from .persistence import (
    persist_actuation_report,
    persist_eval_report,
    persist_ralph_report,
)
from .plan import IngestPlan, plan_pack_ingest
from .ralph import (
    InvestigationProposal,
    RalphInvestigation,
    RalphReport,
    build_investigation_context,
    build_ralph_investigator,
    rank_regressions,
    run_ralph_a,
    select_worst_regression,
)
from .ralph_apply import (
    ActuationOutcome,
    ActuationReport,
    ApplyResult,
    run_ralph_b,
)
from .report import EvalReport, IngestReport, build_eval_report
from .reporting import (
    render_baseline_diff_markdown,
    render_regression_section,
)
from .retrieve import QueryRetrievalResult, RetrievalResults, retrieve_for_goldens
from .sustained_regression import (
    SustainedMetricRegression,
    detect_sustained_regressions,
)

__all__ = [
    "ActuationOutcome",
    "ActuationReport",
    "AlertPayload",
    "ApplyResult",
    "BaselineDiff",
    "CALIBRATION_FIXTURE_SHA",
    "CalibrationAlert",
    "CalibrationExample",
    "CalibrationRecord",
    "CalibrationResult",
    "ClaudeCliJudgeClient",
    "DEFAULT_CALIBRATION_FIXTURE_PATH",
    "EvalReport",
    "FaithfulnessResults",
    "GateFailure",
    "GateResult",
    "IngestPlan",
    "IngestReport",
    "InvestigationProposal",
    "JudgeBackendError",
    "JudgeClient",
    "JudgeQuestion",
    "JudgmentScore",
    "MetricDelta",
    "QueryFaithfulnessResult",
    "QueryRetrievalResult",
    "RalphInvestigation",
    "RalphReport",
    "RetrievalResults",
    "SustainedMetricRegression",
    "aggregate_faithfulness_by_qtype",
    "aggregate_mrr_by_qtype",
    "aggregate_recall_by_qtype",
    "binarize_score",
    "build_claude_cli_judge_client",
    "build_eval_report",
    "build_investigation_context",
    "build_judge_client",
    "build_ralph_investigator",
    "cohens_kappa",
    "compute_baseline_diff",
    "compute_calibration",
    "detect_drift",
    "detect_sustained_regressions",
    "execute_plan",
    "load_calibration_fixture",
    "load_judge_prompt",
    "persist_actuation_report",
    "persist_calibration_record",
    "persist_eval_report",
    "persist_ralph_report",
    "plan_pack_ingest",
    "rank_regressions",
    "read_max_parallel_judges",
    "read_samples_per_claim",
    "recall_at_k",
    "reciprocal_rank",
    "render_baseline_diff_markdown",
    "render_ci_summary",
    "render_judge_prompt",
    "render_regression_section",
    "retrieve_for_goldens",
    "run_calibration",
    "run_ralph_a",
    "run_ralph_b",
    "score_goldens",
    "select_worst_regression",
    "send_alert",
    "send_calibration_alert",
    "validate_eval_report",
]
