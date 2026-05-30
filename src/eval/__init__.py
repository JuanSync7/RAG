# @summary
# Eval package — retrieval quality measurement, fixtures, and harnesses.
# Exports: metrics (recall_at_k, ndcg_at_k, mean_metric), fixtures (load_fixture),
# tree_retrieval_eval (run_eval, build_simulated_retriever, EvalReport),
# pack format (EvalPack, load_pack, validate_pack, PackValidationError),
# runner (IngestPlan, IngestReport, EvalReport, RetrievalResults,
# QueryRetrievalResult, JudgeClient, JudgeQuestion, JudgmentScore,
# execute_plan, plan_pack_ingest, retrieve_for_goldens, recall_at_k,
# aggregate_recall_by_qtype, build_judge_client, load_judge_prompt,
# render_judge_prompt).
# Deps: stdlib only (pack subpackage adds pydantic v2 + pyyaml; runner.execute
#       adds src.ingest + src.vector_db at first use of execute_plan;
#       runner.retrieve adds src.core.embeddings + src.vector_db;
#       runner.judge adds src.common.llm.provider at first use).
# @end-summary
from .pack import EvalPack, PackValidationError, load_pack, validate_pack
from .runner import (
    EvalReport,
    IngestPlan,
    IngestReport,
    JudgeClient,
    JudgeQuestion,
    JudgmentScore,
    QueryRetrievalResult,
    RetrievalResults,
    aggregate_recall_by_qtype,
    build_judge_client,
    execute_plan,
    load_judge_prompt,
    plan_pack_ingest,
    recall_at_k,
    render_judge_prompt,
    retrieve_for_goldens,
)

__all__ = [
    "EvalPack",
    "EvalReport",
    "IngestPlan",
    "IngestReport",
    "JudgeClient",
    "JudgeQuestion",
    "JudgmentScore",
    "PackValidationError",
    "QueryRetrievalResult",
    "RetrievalResults",
    "aggregate_recall_by_qtype",
    "build_judge_client",
    "execute_plan",
    "load_judge_prompt",
    "load_pack",
    "plan_pack_ingest",
    "recall_at_k",
    "render_judge_prompt",
    "retrieve_for_goldens",
    "validate_pack",
]
