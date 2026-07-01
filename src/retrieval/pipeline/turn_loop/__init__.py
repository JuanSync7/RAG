# @summary
# Public facade for the turn-level agentic conversation loop (stable import
# surface). Eagerly re-exports every schema contract from .schemas; lazily
# resolves run_turn_loop from .orchestrator via module __getattr__ (PEP 562) so
# the package imports cleanly in the CONTRACTS phase, before the orchestrator
# module lands.
# Exports: run_turn_loop, TurnAction, RetrieveArgs, DeepStudyArgs, ClarifyArgs,
#          AnswerArgs, TurnActionArgs, TurnDecision, TurnBudget, EvidenceChunk,
#          GateFeedback, TurnEventType, TurnEvent, StudiedDoc, TurnState,
#          ClarificationOut, TurnLoopResult, TurnLoopDeps, TurnContext
# Deps: .schemas (.orchestrator lazily)
# @end-summary
"""Turn-level agentic conversation loop — public API.

One controller LLM per turn iterates RETRIEVE / DEEP_STUDY / CLARIFY / ANSWER
actions until a final-answer confidence gate passes, streaming every decision
as typed events. Callers import from this package only — never from
submodules. See ``docs/retrieval/TURN_LOOP_DESIGN.md``.
"""

from __future__ import annotations

from typing import Any

from src.retrieval.pipeline.turn_loop.schemas import (
    AnswerArgs,
    ClarificationOut,
    ClarifyArgs,
    DeepStudyArgs,
    EvidenceChunk,
    GateFeedback,
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
    "TurnAction",
    "RetrieveArgs",
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
    "TurnState",
    "ClarificationOut",
    "TurnLoopResult",
    "TurnLoopDeps",
    "TurnContext",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve the orchestrator entry point (PEP 562).

    ``run_turn_loop`` lives in ``.orchestrator``, which is delivered by a later
    phase — deferring the import keeps this package importable now and avoids
    paying the orchestrator's import cost for schema-only consumers later.
    """
    if name == "run_turn_loop":
        from src.retrieval.pipeline.turn_loop.orchestrator import run_turn_loop

        return run_turn_loop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
