# @summary
# Eval package — retrieval quality measurement, fixtures, and harnesses.
# Exports: metrics (recall_at_k, ndcg_at_k, mean_metric), fixtures (load_fixture),
# tree_retrieval_eval (run_eval, build_simulated_retriever, EvalReport),
# pack format (EvalPack, load_pack, validate_pack, PackValidationError),
# runner (IngestPlan, IngestReport, execute_plan, plan_pack_ingest).
# Deps: stdlib only (pack subpackage adds pydantic v2 + pyyaml; runner.execute
#       adds src.ingest + src.vector_db at first use of execute_plan).
# @end-summary
from .pack import EvalPack, PackValidationError, load_pack, validate_pack
from .runner import IngestPlan, IngestReport, execute_plan, plan_pack_ingest

__all__ = [
    "EvalPack",
    "IngestPlan",
    "IngestReport",
    "PackValidationError",
    "execute_plan",
    "load_pack",
    "plan_pack_ingest",
    "validate_pack",
]
