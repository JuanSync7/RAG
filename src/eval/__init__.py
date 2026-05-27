# @summary
# Eval package — retrieval quality measurement, fixtures, and harnesses.
# Exports: metrics (recall_at_k, ndcg_at_k, mean_metric), fixtures (load_fixture),
# tree_retrieval_eval (run_eval, build_simulated_retriever, EvalReport),
# pack format (EvalPack, load_pack, validate_pack, PackValidationError).
# Deps: stdlib only (pack subpackage adds pydantic v2 + pyyaml).
# @end-summary
from .pack import EvalPack, PackValidationError, load_pack, validate_pack

__all__ = [
    "EvalPack",
    "PackValidationError",
    "load_pack",
    "validate_pack",
]
