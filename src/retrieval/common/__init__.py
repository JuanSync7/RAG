# @summary
# Retrieval common package: pipeline boundary contracts, wire types, and shared exceptions.
# Exports: RAGRequest, RAGResponse, RankedResult, VisualPageResult, RetrievalError,
#          ModelLoadError, parse_json_object
# Deps: src.retrieval.common.schemas, src.retrieval.common.exceptions, src.retrieval.common.utils
# @end-summary
"""Pipeline boundary contracts, shared wire types, and exception hierarchy."""

from src.retrieval.common.schemas import (
    RAGRequest,
    RAGResponse,
    RankedResult,
    VisualPageResult,
)
from src.retrieval.common.exceptions import RetrievalError, ModelLoadError
from src.retrieval.common.utils import parse_json_object
from src.retrieval.common.terminal import (
    AskUserReason,
    StageOutcome,
    StageStatus,
    TerminalAction,
    TerminalDecision,
    decide_terminal,
)

__all__ = [
    "RAGRequest",
    "RAGResponse",
    "RankedResult",
    "VisualPageResult",
    "RetrievalError",
    "ModelLoadError",
    "parse_json_object",
    "AskUserReason",
    "StageOutcome",
    "StageStatus",
    "TerminalAction",
    "TerminalDecision",
    "decide_terminal",
]
