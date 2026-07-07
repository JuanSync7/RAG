# @summary
# Schema contracts for conversation memory: stored turn/meta/summary dataclasses
# and the assembled MemoryContext (prose context_text + structured turn-context
# dict for the turn loop). All turn-loop fields (actions, chunk_refs,
# answer_confidence, clarification, docs_studied, structured) default empty so
# records written before they existed load unchanged.
# Exports: ConversationRole, ConversationTurn, ConversationSummary,
#          ConversationMeta, MemoryContext
# Deps: dataclasses, typing
# @end-summary
"""Schema contracts for conversation memory persistence and context building.

These dataclasses define the stored conversation turn format and the assembled
memory context injected into prompts. Every field added for the turn-level
conversation loop (design: ``docs/retrieval/TURN_LOOP_DESIGN.md`` §7) is
default-empty and decode-tolerant: Redis records written before the fields
existed keep loading (same migration posture as the provider's
``_decode_id_list``).
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

    sources: list = field(default_factory=list)
    """Source references served with this turn (``_source_refs`` shape). Their
    ``text`` field is preview-capped at the provider boundary
    (``RAG_TURN_CONTEXT_PREVIEW_CHARS``) unless
    ``RAG_TURN_CONTEXT_STORE_FULL_TEXT`` is true, in which case the full chunk
    text is stored (debugging escape hatch; unbounded Redis growth)."""

    actions: list[dict] = field(default_factory=list)
    """Turn-loop action records for the turn (TurnActionRecord dicts:
    ``{action, reason, ms, llm_calls}``). Empty for non-loop turns and for
    records written before the turn loop existed."""

    chunk_refs: list[dict] = field(default_factory=list)
    """Served-chunk references (ChunkRef dicts: ``{chunk_id, document_id,
    source_key, heading, score, refactored_char_start, refactored_char_end,
    preview}``). ``preview`` is capped at the provider boundary
    (``RAG_TURN_CONTEXT_PREVIEW_CHARS``) unless
    ``RAG_TURN_CONTEXT_STORE_FULL_TEXT`` is true; full text stays recoverable
    by ``chunk_id`` (Weaviate uuid)."""

    answer_confidence: float | None = None
    """Turn-loop answer-gate composite score for an assistant turn (None for
    non-loop turns / pre-migration records)."""

    clarification: dict | None = None
    """Clarification payload when an assistant turn ended ``ask_user``
    (``{question, hints, scoping_questions}``); None otherwise. The next
    turn's context surfaces it as ``pending_clarification`` until a later
    user turn consumes it."""


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

    docs_studied: list[dict] = field(default_factory=list)
    """Deep-study ledger for the conversation (dicts: ``{document_id,
    windows_read, sections, conclusion, ts}``), newest last, capped at
    ``RAG_TURN_CONTEXT_MAX_CHUNK_REFS``. Stored on the meta hash with the same
    tolerant JSON-list pattern as ``relevant_doc_ids``/``ignored_doc_ids`` so
    pre-migration conversations load with an empty list."""


@dataclass
class MemoryContext:
    """Bounded context assembled for the next query."""

    conversation_id: str
    summary_text: str
    recent_turns: list[ConversationTurn]
    context_text: str

    structured: dict = field(default_factory=dict)
    """Typed cross-turn context for the turn loop (design §7), alongside the
    prose ``context_text``: ``{rolling_summary, recent_turns: [{question,
    answer}], chunk_refs: [ChunkRef dicts, newest first, capped], docs_studied,
    pending_clarification}``. Empty dict from providers/records that predate
    the turn loop."""


__all__ = [
    "ConversationMeta",
    "ConversationRole",
    "ConversationSummary",
    "ConversationTurn",
    "MemoryContext",
]
