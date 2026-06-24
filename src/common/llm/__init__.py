# @summary
# Framework-agnostic LLM composition layer.
# Wraps LangChain/LangGraph primitives behind generic interfaces so callers
# never import from third-party frameworks directly.
# Exports: get_llm, structured_output, parallel, aparallel, enable_cache,
#          disable_cache, clear_cache, batch, abatch, fallback_chain,
#          afallback_chain, stream, astream, conversation,
#          and all public schemas.
# Deps: src.common.llm.* submodules
# @end-summary
"""LLM composition layer — provider-agnostic, framework-agnostic.

This package provides generic wrappers around LangChain/LangGraph primitives
so that the rest of the codebase interacts with stable, project-owned
interfaces.  If LangChain is replaced, only the internals of this package
change — callers are unaffected.

Quick start::

    from src.common.llm import get_llm, structured_output, parallel

    llm = get_llm("default")
    result = structured_output(llm, MySchema, "Extract the metadata")

    ctx = parallel(
        vectors=lambda: search_weaviate(q),
        keywords=lambda: search_bm25(q),
    )
"""

# ── Provider ─────────────────────────────────────────────────────────────
from src.common.llm.provider import ChatLLMAdapter, get_llm

# ── Structured output ────────────────────────────────────────────────────
from src.common.llm.output import structured_output

# ── Parallel execution ───────────────────────────────────────────────────
from src.common.llm.parallel import aparallel, parallel

# ── Caching ──────────────────────────────────────────────────────────────
from src.common.llm.cache import clear_cache, disable_cache, enable_cache, get_platform_cache

# ── Batch processing ─────────────────────────────────────────────────────
from src.common.llm.batch import abatch, batch

# ── Fallback chains ──────────────────────────────────────────────────────
from src.common.llm.fallback import afallback_chain, fallback_chain

# ── Streaming ────────────────────────────────────────────────────────────
from src.common.llm.stream import astream, stream

# ── Conversation memory ──────────────────────────────────────────────────
from src.common.llm.memory import ConversationSession, conversation

# ── Public schemas ───────────────────────────────────────────────────────
from src.common.llm.schemas import (
    BatchResult,
    ConversationMessage,
    FallbackResult,
    GateDecision,
    MessageRole,
    ModelTier,
    OutputResult,
    ParallelResult,
    StepResult,
    StepStatus,
    StreamEvent,
    StreamFrame,
)

__all__ = [
    # Provider
    "get_llm",
    "ChatLLMAdapter",
    # Structured output
    "structured_output",
    # Parallel
    "parallel",
    "aparallel",
    # Cache
    "enable_cache",
    "disable_cache",
    "clear_cache",
    "get_platform_cache",
    # Batch
    "batch",
    "abatch",
    # Fallback
    "fallback_chain",
    "afallback_chain",
    # Stream
    "stream",
    "astream",
    # Memory
    "conversation",
    "ConversationSession",
    # Schemas
    "ModelTier",
    "OutputResult",
    "ParallelResult",
    "BatchResult",
    "FallbackResult",
    "StreamEvent",
    "StreamFrame",
    "StepStatus",
    "StepResult",
    "GateDecision",
    "MessageRole",
    "ConversationMessage",
]
