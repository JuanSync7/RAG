# @summary
# Public facade for the turn-level agentic conversation loop (stable import
# surface). Re-exports every schema contract from .schemas, the loop entry
# point run_turn_loop from .orchestrator, and the TurnContext assembler
# build_turn_context from .context. Callers import from this package only —
# never from submodules.
# Exports: run_turn_loop, build_turn_context, route, RouteConfig, RouteEffort,
#          RouteHint, RouteSignals, TurnAction, QueryShape, RetrieveArgs,
#          DecomposeArgs, DeepStudyArgs, ClarifyArgs, AnswerArgs, TurnActionArgs,
#          TurnDecision, TurnBudget, EvidenceChunk, GateFeedback,
#          TurnEventType, TurnEvent, StudiedDoc, FacetCoverage, TurnState,
#          ClarificationOut, TurnLoopResult, TurnLoopDeps, TurnContext
# Deps: .schemas, .orchestrator, .context, .router
# @end-summary
"""Turn-level agentic conversation loop — public API.

One controller LLM per turn iterates RETRIEVE / DEEP_STUDY / CLARIFY / ANSWER
actions until a final-answer confidence gate passes, streaming every decision
as typed events. Callers import from this package only — never from
submodules. See ``docs/retrieval/TURN_LOOP_DESIGN.md``.
"""

from __future__ import annotations

from src.retrieval.pipeline.turn_loop.context import build_turn_context
from src.retrieval.pipeline.turn_loop.orchestrator import run_turn_loop
from src.retrieval.pipeline.turn_loop.router import (
    RouteConfig,
    RouteEffort,
    RouteHint,
    RouteSignals,
    route,
)
from src.retrieval.pipeline.turn_loop.schemas import (
    AnswerArgs,
    ClarificationOut,
    ClarifyArgs,
    DecomposeArgs,
    DeepStudyArgs,
    EvidenceChunk,
    FacetCoverage,
    GateFeedback,
    QueryShape,
    RetrieveArgs,
    StudiedDoc,
    TurnAction,
    TurnActionArgs,
    TurnBudget,
    TurnContext,
    TurnDecision,
    TurnEvent,
    TurnEventType,
    TurnLoopDeps,
    TurnLoopResult,
    TurnState,
)

__all__ = [
    "run_turn_loop",
    "build_turn_context",
    "route",
    "RouteConfig",
    "RouteEffort",
    "RouteHint",
    "RouteSignals",
    "TurnAction",
    "QueryShape",
    "RetrieveArgs",
    "DecomposeArgs",
    "DeepStudyArgs",
    "ClarifyArgs",
    "AnswerArgs",
    "TurnActionArgs",
    "TurnDecision",
    "TurnBudget",
    "EvidenceChunk",
    "GateFeedback",
    "TurnEventType",
    "TurnEvent",
    "StudiedDoc",
    "FacetCoverage",
    "TurnState",
    "ClarificationOut",
    "TurnLoopResult",
    "TurnLoopDeps",
    "TurnContext",
]
