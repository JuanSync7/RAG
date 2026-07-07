# @summary
# Public facade for the agentic retrieval loop (stable import surface). Re-exports
# the orchestrator, its typed contracts, and the HyDE/judge entry points so callers
# import from ``src.retrieval.pipeline.agentic`` and never reach into submodules.
# Exports: AgenticRetrieval, AgenticBudget, AgenticResult, AgenticState, HydeVariant,
#          ChunkVerdict, PoolVerdict, generate_hyde, judge_chunks
# Deps: .orchestrator, .state, .hyde, .judge
# @end-summary
"""Agentic HyDE/controller/judge retrieval loop — public API.

A controller LLM drives retrieval as a tool (generating HyDE queries), an LLM
judge scores each chunk for relevance/faithfulness/sufficiency, judge-approved
chunks accumulate, and the loop retries until the kept set is sufficient or a
budget caps it. Like deep_research it replaces the linear stages 2-5. See
``docs/retrieval/AGENTIC_RETRIEVAL_DESIGN.md``.
"""

from src.retrieval.pipeline.agentic.hyde import generate_hyde
from src.retrieval.pipeline.agentic.judge import judge_chunks
from src.retrieval.pipeline.agentic.orchestrator import AgenticRetrieval
from src.retrieval.pipeline.agentic.state import (
    AgenticBudget,
    AgenticResult,
    AgenticState,
    ChunkVerdict,
    HydeVariant,
    PoolVerdict,
)

__all__ = [
    "AgenticRetrieval",
    "AgenticBudget",
    "AgenticResult",
    "AgenticState",
    "HydeVariant",
    "ChunkVerdict",
    "PoolVerdict",
    "generate_hyde",
    "judge_chunks",
]
