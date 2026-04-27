# @summary
# Query sub-package schema contracts: query processing state, results, and routing actions.
# Exports: QueryAction, QueryResult, QueryState
# Deps: dataclasses, enum, typing
# @end-summary
"""Query processing schema contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, TypedDict


class QueryAction(Enum):
    """Action to take after query processing."""

    SEARCH = "search"
    ASK_USER = "ask_user"


@dataclass
class QueryResult:
    """Result of the query processing pipeline."""

    processed_query: str
    confidence: float
    action: QueryAction
    standalone_query: str = ""
    suppress_memory: bool = False
    has_backward_reference: bool = False
    clarification_message: Optional[str] = None
    iterations: int = 0
    # Retrieval-mode explainability — only populated when called with
    # ``mode="retrieval"``. ``history_decision`` ∈ {"use_as_is",
    # "partial_history", "full_history", "hard_query"}.
    history_decision: Optional[str] = None
    history_turns_used: int = 0


class QueryState(TypedDict):
    """LangGraph state schema for query processing."""

    original_query: str
    current_query: str
    confidence: float
    reasoning: str
    iteration: int
    max_iterations: int
    confidence_threshold: float
    action: str
    clarification_message: str
    ollama_available: bool
    fast_path: bool
    standalone_query: str
    suppress_memory: bool
    has_backward_reference: bool
    has_memory_context: bool
