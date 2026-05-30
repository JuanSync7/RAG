# @summary
# Eval package — retrieval quality measurement, fixtures, and harnesses.
# Exports: metrics (recall_at_k, ndcg_at_k, mean_metric), fixtures (load_fixture),
# tree_retrieval_eval (run_eval, build_simulated_retriever, EvalReport),
# pack format (EvalPack, load_pack, validate_pack, PackValidationError),
# runner (IngestPlan, IngestReport, EvalReport, RetrievalResults,
# QueryRetrievalResult, execute_plan, plan_pack_ingest, retrieve_for_goldens,
# recall_at_k, aggregate_recall_by_qtype).
# Deps: stdlib only (pack subpackage adds pydantic v2 + pyyaml; runner.execute
#       adds src.ingest + src.vector_db at first use of execute_plan;
#       runner.retrieve adds src.core.embeddings + src.vector_db).
# @end-summary
from .pack import EvalPack, PackValidationError, load_pack, validate_pack
from .runner import (
    EvalReport,
    IngestPlan,
    IngestReport,
    QueryRetrievalResult,
    RetrievalResults,
    aggregate_recall_by_qtype,
    execute_plan,
    plan_pack_ingest,
    recall_at_k,
    retrieve_for_goldens,
)

__all__ = [
    "EvalPack",
    "EvalReport",
    "IngestPlan",
    "IngestReport",
    "PackValidationError",
    "QueryRetrievalResult",
    "RetrievalResults",
    "aggregate_recall_by_qtype",
    "execute_plan",
    "load_pack",
    "plan_pack_ingest",
    "recall_at_k",
    "retrieve_for_goldens",
    "validate_pack",
]
