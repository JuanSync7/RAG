# @summary
# TurnContext assembly for the turn loop: build_turn_context coerces the plain
# structured dict produced by the memory layer ({rolling_summary, recent_turns,
# chunk_refs, docs_studied, pending_clarification}) into the typed TurnContext
# contract — tolerant of missing/None/malformed keys (older conversations
# predate some fields), recent_turns 'question' keys aliased onto the frozen
# 'query' contract, chunk_refs capped at RAG_TURN_CONTEXT_MAX_CHUNK_REFS
# keeping the most recent refs.
# Exports: build_turn_context
# Deps: src.retrieval.pipeline.turn_loop.schemas (config.settings lazily)
# @end-summary
"""TurnContext assembly from the memory layer's structured turn-context dict.

The memory track (``src/platform/memory``) produces a plain dict per
conversation; this module is the one bridge from that dict to the typed
:class:`TurnContext` the loop consumes. Every field is optional and every
malformed entry is dropped rather than raised on — conversations created
before a field existed must keep working (design §7 tolerant-migration rule).
"""

from __future__ import annotations

from typing import Optional

from src.retrieval.pipeline.turn_loop.schemas import TurnContext


def _max_chunk_refs() -> int:
    """Chunk-ref cap for the assembled context (``RAG_TURN_CONTEXT_MAX_CHUNK_REFS``)."""
    from config import settings

    return int(getattr(settings, "RAG_TURN_CONTEXT_MAX_CHUNK_REFS", 24))


def _dict_list(value: object) -> list[dict]:
    """Coerce a maybe-list into a list of its dict entries (others dropped)."""
    if not isinstance(value, list):
        return []
    return [entry for entry in value if isinstance(entry, dict)]


def _normalize_recent_turns(value: object) -> list[dict]:
    """Coerce recent-turn dicts onto the ``query``/``answer`` key contract.

    The memory layer's structured context names the user side of a Q&A pair
    ``question`` (``build_structured_context``), while the frozen
    :class:`TurnContext` prompt renderer reads ``query`` — this is the single
    bridge between the two vocabularies. Entries already carrying ``query``
    pass through untouched; ``question`` is aliased onto ``query`` without
    dropping any other key.
    """
    turns: list[dict] = []
    for entry in _dict_list(value):
        if "query" not in entry and "question" in entry:
            entry = {**entry, "query": entry.get("question")}
        turns.append(entry)
    return turns


def build_turn_context(
    conversation_id: str, memory: Optional[dict] = None
) -> TurnContext:
    """Build the typed cross-turn context from the memory layer's dict.

    Args:
        conversation_id: Owning conversation identity (always required — a
            brand-new conversation still has an id).
        memory: The structured turn-context dict from conversation memory
            (``rolling_summary`` / ``recent_turns`` / ``chunk_refs`` /
            ``docs_studied`` / ``pending_clarification``), or ``None`` for a
            fresh conversation.

    Returns:
        A :class:`TurnContext` with every absent/malformed field degraded to
        its empty default. ``chunk_refs`` keeps only the MOST RECENT
        ``RAG_TURN_CONTEXT_MAX_CHUNK_REFS`` entries (the list is stored in
        chronological order; recent refs are the live anchors for deepening).
    """
    if not isinstance(memory, dict):
        memory = {}
    chunk_refs = _dict_list(memory.get("chunk_refs"))
    cap = _max_chunk_refs()
    if len(chunk_refs) > cap:
        chunk_refs = chunk_refs[-cap:]
    pending = memory.get("pending_clarification")
    return TurnContext(
        conversation_id=conversation_id,
        rolling_summary=str(memory.get("rolling_summary") or ""),
        recent_turns=_normalize_recent_turns(memory.get("recent_turns")),
        chunk_refs=chunk_refs,
        docs_studied=_dict_list(memory.get("docs_studied")),
        pending_clarification=pending if isinstance(pending, dict) else None,
    )


__all__ = ["build_turn_context"]
