"""Schema contracts for conversation memory persistence and context building.

These dataclasses define the stored conversation turn format and the assembled
memory context injected into prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ConversationRole = Literal["user", "assistant", "system"]


@dataclass
class ConversationTurn:
    """Single conversation turn."""

    role: ConversationRole
    content: str
    timestamp_ms: int
    query_id: str = ""
    sources: list = field(default_factory=list)  # lightweight source refs (no full chunk text)


@dataclass
class ConversationSummary:
    """Rolling compact summary for older turns."""

    text: str = ""
    updated_at_ms: int = 0
    turns_compacted: int = 0


@dataclass
class ConversationMeta:
    """Top-level conversation metadata."""

    conversation_id: str
    tenant_id: str
    subject: str
    project_id: str = ""
    title: str = ""
    created_at_ms: int = 0
    updated_at_ms: int = 0
    message_count: int = 0
    summary: ConversationSummary = field(default_factory=ConversationSummary)
    relevant_doc_ids: list[str] = field(default_factory=list)
    ignored_doc_ids: list[str] = field(default_factory=list)


@dataclass
class MemoryContext:
    """Bounded context assembled for the next query."""

    conversation_id: str
    summary_text: str
    recent_turns: list[ConversationTurn]
    context_text: str


__all__ = [
    "ConversationMeta",
    "ConversationRole",
    "ConversationSummary",
    "ConversationTurn",
    "MemoryContext",
]
