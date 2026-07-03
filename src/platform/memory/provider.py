# @summary
# Conversation memory providers (Redis canonical backend + no-op fallback) and
# singleton factory. Persists turn-loop context on turns (actions, chunk_refs,
# answer_confidence, clarification) and meta (docs_studied ledger via
# record_doc_studied), applies the RAG_TURN_CONTEXT_* preview growth control at
# the write boundary, and assembles MemoryContext.structured for the turn loop.
# Exports: ConversationMemoryProvider, NoopConversationMemory, RedisConversationMemory,
#          get_conversation_memory, conversation_meta_to_dict, conversation_turns_to_dict
# Deps: config.settings, orjson, inspect, logging, uuid, dataclasses, src.platform.memory.utils, src.platform.llm
# @end-summary
"""Conversation memory providers and factory.

This module provides a Redis-backed conversation memory implementation plus a
no-op fallback when memory is disabled or unavailable. It also exposes a
process-wide singleton resolver for use by API and CLI surfaces.

Turn-loop context transfer (design ``docs/retrieval/TURN_LOOP_DESIGN.md`` §7)
is layered on the same records: ``append_turn`` accepts the typed turn-loop
fields, ``record_doc_studied`` maintains the per-conversation deep-study
ledger, and ``build_context`` returns a ``structured`` dict alongside the
prose ``context_text``. All new fields are decode-tolerant so records written
before the migration keep loading. Growth control is applied here — at the
provider boundary — so every writer inherits the preview cap
(``RAG_TURN_CONTEXT_STORE_FULL_TEXT`` / ``RAG_TURN_CONTEXT_PREVIEW_CHARS``).
"""

from __future__ import annotations

import inspect
import orjson
import logging
import time
import uuid
from dataclasses import asdict
from typing import Any

from config.settings import (
    MEMORY_ENABLED,
    MEMORY_MAX_CONTEXT_TOKENS_ESTIMATE,
    MEMORY_MAX_RECENT_TURNS,
    MEMORY_PROVIDER,
    MEMORY_REDIS_CONNECT_TIMEOUT_S,
    MEMORY_REDIS_PREFIX,
    MEMORY_REDIS_URL,
    MEMORY_SUMMARY_MAX_SOURCE_TURNS,
    MEMORY_SUMMARY_TRIGGER_TURNS,
    RAG_MEMORY_GET_TURNS_DEFAULT_LIMIT,
    RAG_MEMORY_LIST_CONVERSATIONS_DEFAULT_LIMIT,
    RAG_MEMORY_CONTEXT_MAX_CHARS,
    RAG_MEMORY_LLM_SUMMARIZER_MAX_TOKENS,
    RAG_MEMORY_LLM_SUMMARY_SANITIZED_MAX_CHARS,
    RAG_MEMORY_CONVERSATION_TITLE_MAX_CHARS,
    RAG_TURN_CONTEXT_MAX_CHUNK_REFS,
    RAG_TURN_CONTEXT_PREVIEW_CHARS,
    RAG_TURN_CONTEXT_STORE_FULL_TEXT,
)
from src.platform.memory.schemas import (
    ConversationMeta,
    ConversationSummary,
    ConversationTurn,
    MemoryContext,
)
from src.platform.memory.utils import (
    build_context_text,
    build_structured_context,
    now_ms,
    render_clarification_grounding,
    render_docs_studied_grounding,
    sanitize_memory_text,
    summarize_heuristic,
    trim_turns_to_budget,
)
from src.platform.llm import get_llm_provider

logger = logging.getLogger("rag.memory")


def _decode_id_list(raw: Any) -> list[str]:
    """Parse a JSON-encoded id list stored on the conversation meta hash.

    Tolerates missing/empty values so older conversations created before the
    relevant/ignored fields existed continue to load cleanly.
    """
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    try:
        parsed = orjson.loads(raw)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(x) for x in parsed]


def _encode_id_list(values: list[str]) -> str:
    return orjson.dumps(list(values)).decode("utf-8")


def _decode_dict_list(raw: Any) -> list[dict]:
    """Parse a JSON-encoded list of dicts stored on the conversation meta hash.

    Same tolerance posture as :func:`_decode_id_list`: missing / empty /
    unparsable values (conversations created before the field existed) decode
    to an empty list, and non-dict entries are dropped rather than raising.
    """
    if not raw:
        return []
    parsed: Any = raw
    if not isinstance(raw, list):
        try:
            parsed = orjson.loads(raw)
        except Exception:
            return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _encode_json_list(values: list) -> str:
    """Encode a list of JSON-serializable items for storage on the meta hash."""
    return orjson.dumps(list(values)).decode("utf-8")


def _dict_list(value: Any) -> list[dict]:
    """Coerce an already-parsed JSON value to a list of dicts (tolerant).

    Turn rows are stored as one JSON document, so per-field values arrive
    parsed; records written before a field existed simply lack the key and
    decode to an empty list here.
    """
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _opt_float(value: Any) -> float | None:
    """Coerce a parsed JSON value to float, or None when absent/invalid."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _opt_dict(value: Any) -> dict | None:
    """Coerce a parsed JSON value to a dict, or None when absent/invalid."""
    return value if isinstance(value, dict) else None


def _apply_preview_cap(refs: list | None, text_field: str) -> list[dict]:
    """Apply the turn-context growth control to a list of reference dicts.

    This is the single write-boundary choke point (design §7): every writer —
    the query route's ``_source_refs``, the turn loop's chunk refs — inherits
    the cap because it runs inside ``append_turn``. When
    ``RAG_TURN_CONTEXT_STORE_FULL_TEXT`` is false (default) the named text
    field of each ref is truncated to ``RAG_TURN_CONTEXT_PREVIEW_CHARS``
    (full text stays recoverable by chunk id); when true the refs are stored
    unchanged (debugging escape hatch; unbounded Redis growth).

    Args:
        refs: Reference dicts to persist (non-dict entries are dropped).
        text_field: Which field carries the cappable text (``"text"`` for
            source refs, ``"preview"`` for chunk refs).

    Returns:
        The list to persist, preview-capped unless full-text storage is on.
    """
    capped: list[dict] = []
    for ref in refs or []:
        if not isinstance(ref, dict):
            continue
        if not RAG_TURN_CONTEXT_STORE_FULL_TEXT:
            text = ref.get(text_field)
            if isinstance(text, str) and len(text) > RAG_TURN_CONTEXT_PREVIEW_CHARS:
                ref = dict(ref)
                ref[text_field] = text[:RAG_TURN_CONTEXT_PREVIEW_CHARS]
        capped.append(ref)
    return capped


class ConversationMemoryProvider:
    """Abstract memory operations."""

    def ensure_conversation(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str | None = None,
        title: str = "",
    ) -> ConversationMeta:
        """Create or load a conversation.

        Args:
            tenant_id: Tenant identifier.
            subject: Subject identifier (user/service).
            project_id: Optional project identifier.
            conversation_id: Optional conversation id to reuse.
            title: Optional title for new conversations.

        Returns:
            Conversation metadata.
        """
        raise NotImplementedError

    def list_conversations(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        limit: int = RAG_MEMORY_LIST_CONVERSATIONS_DEFAULT_LIMIT,
    ) -> list[ConversationMeta]:
        """List recent conversations for a subject scope.

        Args:
            tenant_id: Tenant identifier.
            subject: Subject identifier (user/service).
            project_id: Optional project identifier.
            limit: Max number of conversations to return.

        Returns:
            List of conversation metadata entries, newest first.
        """
        raise NotImplementedError

    def get_turns(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        limit: int = RAG_MEMORY_GET_TURNS_DEFAULT_LIMIT,
    ) -> list[ConversationTurn]:
        """Fetch conversation turns.

        Args:
            tenant_id: Tenant identifier.
            subject: Subject identifier (user/service).
            project_id: Optional project identifier.
            conversation_id: Conversation identifier.
            limit: Max number of turns to return.

        Returns:
            List of conversation turns, oldest-to-newest.
        """
        raise NotImplementedError

    def append_turn(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        role: str,
        content: str,
        query_id: str = "",
        sources: list | None = None,
        actions: list | None = None,
        chunk_refs: list | None = None,
        answer_confidence: float | None = None,
        clarification: dict | None = None,
    ) -> None:
        """Append a single turn to a conversation.

        Args:
            tenant_id: Tenant identifier.
            subject: Subject identifier (user/service).
            project_id: Optional project identifier.
            conversation_id: Conversation identifier.
            role: Turn role ("user", "assistant", "system").
            content: Turn text content.
            query_id: Optional query/request id for traceability.
            sources: Optional source references (``_source_refs`` shape);
                their ``text`` field is preview-capped at this boundary.
            actions: Optional turn-loop action records
                (``{action, reason, ms, llm_calls}`` dicts).
            chunk_refs: Optional served-chunk references (ChunkRef dicts);
                their ``preview`` field is preview-capped at this boundary.
            answer_confidence: Optional turn-loop answer-gate composite score.
            clarification: Optional ``{question, hints, scoping_questions}``
                payload when an assistant turn ended ``ask_user``.
        """
        raise NotImplementedError

    def record_doc_studied(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        entry: dict,
    ) -> ConversationMeta:
        """Append one deep-study ledger entry to the conversation meta.

        The ledger (``docs_studied``) records which documents the turn loop
        deep-studied and what it concluded, so later turns can deepen instead
        of re-reading. Stored with the same JSON-list meta pattern as
        ``relevant_doc_ids``/``ignored_doc_ids`` and capped (newest kept) at
        ``RAG_TURN_CONTEXT_MAX_CHUNK_REFS``.

        Args:
            tenant_id: Tenant identifier.
            subject: Subject identifier (user/service).
            project_id: Optional project identifier.
            conversation_id: Conversation identifier.
            entry: Ledger entry dict (``{document_id, windows_read, sections,
                conclusion, ts}``); ``ts`` is stamped here when absent.

        Returns:
            The updated conversation metadata.
        """
        raise NotImplementedError

    def build_context(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        turn_window: int | None = None,
    ) -> MemoryContext:
        """Build a bounded memory context for the next request.

        Args:
            tenant_id: Tenant identifier.
            subject: Subject identifier (user/service).
            project_id: Optional project identifier.
            conversation_id: Conversation identifier.
            turn_window: Optional max recent turns override.

        Returns:
            A `MemoryContext` containing summary + recent turns + rendered context text.
        """
        raise NotImplementedError

    def compact_if_needed(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        force: bool = False,
    ) -> ConversationSummary:
        """Compact older turns into a rolling summary, if needed.

        Args:
            tenant_id: Tenant identifier.
            subject: Subject identifier (user/service).
            project_id: Optional project identifier.
            conversation_id: Conversation identifier.
            force: If True, compact regardless of thresholds.

        Returns:
            The current (possibly updated) conversation summary.
        """
        raise NotImplementedError

    def delete_conversation(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
    ) -> bool:
        """Delete a conversation and its turns. Returns True if deleted, False if not found."""
        raise NotImplementedError

    def update_conversation_title(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        title: str,
    ) -> ConversationMeta | None:
        """Update a conversation's title. Returns updated meta, or None if not found."""
        raise NotImplementedError

    def mark_retrieved(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        doc_ids: list[str],
    ) -> ConversationMeta:
        """Add doc_ids to the conversation's relevant list (set-union, idempotent).

        New doc ids land in `relevant_doc_ids`. Doc ids already on `ignored_doc_ids`
        are left there unchanged — `mark_retrieved` only touches docs that have
        not been seen at all yet.
        """
        raise NotImplementedError

    def move_to_ignored(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        doc_id: str,
    ) -> ConversationMeta:
        """Move a single doc_id from relevant → ignored. Idempotent."""
        raise NotImplementedError

    def restore_to_relevant(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        doc_id: str,
    ) -> ConversationMeta:
        """Move a single doc_id from ignored → relevant. Idempotent."""
        raise NotImplementedError

    def clear_doc_state(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        doc_id: str,
    ) -> ConversationMeta:
        """Remove a doc_id from both relevant and ignored lists (neutral). Idempotent."""
        raise NotImplementedError

    def get_seen_doc_ids(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
    ) -> list[str]:
        """Return the union of relevant + ignored doc ids for the conversation.

        This is the hard-suppression list passed to the retrieval filter so
        previously-served documents do not surface again.
        """
        raise NotImplementedError


class NoopConversationMemory(ConversationMemoryProvider):
    """No-op provider when memory is disabled."""

    def ensure_conversation(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str | None = None,
        title: str = "",
    ) -> ConversationMeta:
        cid = conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
        ts = now_ms()
        return ConversationMeta(
            conversation_id=cid,
            tenant_id=tenant_id,
            subject=subject,
            project_id=project_id or "",
            title=title,
            created_at_ms=ts,
            updated_at_ms=ts,
            message_count=0,
        )

    def list_conversations(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        limit: int = RAG_MEMORY_LIST_CONVERSATIONS_DEFAULT_LIMIT,
    ) -> list[ConversationMeta]:
        return []

    def get_turns(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        limit: int = RAG_MEMORY_GET_TURNS_DEFAULT_LIMIT,
    ) -> list[ConversationTurn]:
        return []

    def append_turn(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        role: str,
        content: str,
        query_id: str = "",
        sources: list | None = None,
        actions: list | None = None,
        chunk_refs: list | None = None,
        answer_confidence: float | None = None,
        clarification: dict | None = None,
    ) -> None:
        return

    def record_doc_studied(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        entry: dict,
    ) -> ConversationMeta:
        return ConversationMeta(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            subject=subject,
            project_id=project_id or "",
        )

    def build_context(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        turn_window: int | None = None,
    ) -> MemoryContext:
        return MemoryContext(
            conversation_id=conversation_id,
            summary_text="",
            recent_turns=[],
            context_text="",
        )

    def compact_if_needed(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        force: bool = False,
    ) -> ConversationSummary:
        return ConversationSummary()

    def delete_conversation(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
    ) -> bool:
        return False

    def update_conversation_title(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        title: str,
    ) -> ConversationMeta | None:
        return None

    def mark_retrieved(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        doc_ids: list[str],
    ) -> ConversationMeta:
        return ConversationMeta(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            subject=subject,
            project_id=project_id or "",
        )

    def move_to_ignored(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        doc_id: str,
    ) -> ConversationMeta:
        return ConversationMeta(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            subject=subject,
            project_id=project_id or "",
        )

    def restore_to_relevant(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        doc_id: str,
    ) -> ConversationMeta:
        return ConversationMeta(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            subject=subject,
            project_id=project_id or "",
        )

    def clear_doc_state(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        doc_id: str,
    ) -> ConversationMeta:
        return ConversationMeta(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            subject=subject,
            project_id=project_id or "",
        )

    def get_seen_doc_ids(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
    ) -> list[str]:
        return []


class RedisConversationMemory(ConversationMemoryProvider):
    """Redis-backed canonical conversation memory."""

    def __init__(
        self,
        redis_url: str,
        key_prefix: str,
        *,
        connect_timeout_s: float = MEMORY_REDIS_CONNECT_TIMEOUT_S,
    ) -> None:
        """Create a Redis-backed memory provider.

        Args:
            redis_url: Redis connection URL.
            key_prefix: Key prefix namespace for stored data.
            connect_timeout_s: Socket connect/read timeout used for both the
                eager connectivity ping in the factory and subsequent
                operations. Configurable via ``RAG_MEMORY_REDIS_CONNECT_TIMEOUT_S``.
        """
        import redis  # type: ignore

        self._client = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=connect_timeout_s,
        )
        self._prefix = key_prefix.strip() or "rag:memory"
        self._llm_provider = get_llm_provider()

    def _scope(self, tenant_id: str, subject: str, project_id: str | None) -> str:
        """Build the scope key for a tenant/subject/project tuple.

        Each component is percent-encoded so that a colon inside a component
        (e.g. ``subject="a:b"``) cannot collide with the structural colon
        separators between components.
        """

        def _enc(s: str) -> str:
            return s.replace("%", "%25").replace(":", "%3A")

        parts = [_enc(tenant_id), _enc(subject)]
        if project_id:
            parts.append(_enc(project_id))
        return ":".join(parts)

    def _meta_key(self, scope: str, conversation_id: str) -> str:
        """Return the Redis key for conversation metadata."""
        return f"{self._prefix}:conv:{scope}:{conversation_id}:meta"

    def _turns_key(self, scope: str, conversation_id: str) -> str:
        """Return the Redis key for conversation turns list."""
        return f"{self._prefix}:conv:{scope}:{conversation_id}:turns"

    def _index_key(self, scope: str) -> str:
        """Return the Redis key for the conversation index sorted set."""
        return f"{self._prefix}:conv:{scope}:index"

    def _now(self) -> int:
        """Return current timestamp in milliseconds."""
        return now_ms()

    def _meta_from_hash(self, raw: dict[str, Any], conversation_id: str) -> ConversationMeta:
        """Convert a Redis hash payload into `ConversationMeta`."""
        summary_text = str(raw.get("summary_text", "") or "")
        summary = ConversationSummary(
            text=summary_text,
            updated_at_ms=int(raw.get("summary_updated_at_ms", "0") or 0),
            turns_compacted=int(raw.get("summary_turns_compacted", "0") or 0),
        )
        return ConversationMeta(
            conversation_id=conversation_id,
            tenant_id=str(raw.get("tenant_id", "")),
            subject=str(raw.get("subject", "")),
            project_id=str(raw.get("project_id", "")),
            title=str(raw.get("title", "")),
            created_at_ms=int(raw.get("created_at_ms", "0") or 0),
            updated_at_ms=int(raw.get("updated_at_ms", "0") or 0),
            message_count=int(raw.get("message_count", "0") or 0),
            summary=summary,
            relevant_doc_ids=_decode_id_list(raw.get("relevant_doc_ids")),
            ignored_doc_ids=_decode_id_list(raw.get("ignored_doc_ids")),
            docs_studied=_decode_dict_list(raw.get("docs_studied")),
        )

    def ensure_conversation(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str | None = None,
        title: str = "",
    ) -> ConversationMeta:
        scope = self._scope(tenant_id, subject, project_id)
        cid = (conversation_id or "").strip() or f"conv_{uuid.uuid4().hex[:12]}"
        meta_key = self._meta_key(scope, cid)
        if self._client.exists(meta_key):
            raw = self._client.hgetall(meta_key)
            return self._meta_from_hash(raw, cid)
        now = self._now()
        payload = {
            "tenant_id": tenant_id,
            "subject": subject,
            "project_id": project_id or "",
            "title": title.strip() or "New conversation",
            "created_at_ms": now,
            "updated_at_ms": now,
            "message_count": 0,
            "summary_text": "",
            "summary_updated_at_ms": 0,
            "summary_turns_compacted": 0,
            "relevant_doc_ids": _encode_id_list([]),
            "ignored_doc_ids": _encode_id_list([]),
            "docs_studied": _encode_json_list([]),
        }
        self._client.hset(meta_key, mapping=payload)
        self._client.zadd(self._index_key(scope), {cid: now})
        return self._meta_from_hash(payload, cid)

    def list_conversations(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        limit: int = RAG_MEMORY_LIST_CONVERSATIONS_DEFAULT_LIMIT,
    ) -> list[ConversationMeta]:
        scope = self._scope(tenant_id, subject, project_id)
        conv_ids = self._client.zrevrange(self._index_key(scope), 0, max(0, limit - 1))
        items: list[ConversationMeta] = []
        for cid in conv_ids:
            raw = self._client.hgetall(self._meta_key(scope, cid))
            if not raw:
                continue
            items.append(self._meta_from_hash(raw, cid))
        return items

    def get_turns(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        limit: int = RAG_MEMORY_GET_TURNS_DEFAULT_LIMIT,
    ) -> list[ConversationTurn]:
        scope = self._scope(tenant_id, subject, project_id)
        key = self._turns_key(scope, conversation_id)
        raw_rows = self._client.lrange(key, max(0, -int(limit)), -1)
        turns: list[ConversationTurn] = []
        for row in raw_rows:
            try:
                payload = orjson.loads(row)
                turns.append(
                    ConversationTurn(
                        role=str(payload.get("role", "user")),
                        content=str(payload.get("content", "")),
                        timestamp_ms=int(payload.get("timestamp_ms", 0)),
                        query_id=str(payload.get("query_id", "")),
                        sources=payload.get("sources") or [],
                        # Turn-loop fields: tolerant defaults so rows written
                        # before the migration load unchanged (design §7).
                        actions=_dict_list(payload.get("actions")),
                        chunk_refs=_dict_list(payload.get("chunk_refs")),
                        answer_confidence=_opt_float(
                            payload.get("answer_confidence")
                        ),
                        clarification=_opt_dict(payload.get("clarification")),
                    )
                )
            except Exception:
                logger.warning(
                    "memory.get_turns: dropping unparsable turn row in conversation %s (scope=%s)",
                    conversation_id, scope, exc_info=True,
                )
                continue
        return turns

    def append_turn(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        role: str,
        content: str,
        query_id: str = "",
        sources: list | None = None,
        actions: list | None = None,
        chunk_refs: list | None = None,
        answer_confidence: float | None = None,
        clarification: dict | None = None,
    ) -> None:
        if (
            not content.strip()
            and not (sources or [])
            and not (chunk_refs or [])
            and clarification is None
        ):
            return
        meta = self.ensure_conversation(
            tenant_id=tenant_id,
            subject=subject,
            project_id=project_id,
            conversation_id=conversation_id,
        )
        scope = self._scope(tenant_id, subject, project_id)
        key = self._turns_key(scope, meta.conversation_id)
        now = self._now()
        row = {
            "role": role,
            # preserve_newlines: keep the answer's markdown structure so a
            # reloaded conversation renders lists/paragraphs, not a wall of text.
            "content": sanitize_memory_text(
                content, max_chars=RAG_MEMORY_CONTEXT_MAX_CHARS, preserve_newlines=True
            ),
            "timestamp_ms": now,
            "query_id": query_id,
            # Growth control at the ONE write boundary so every writer
            # inherits the preview cap (design §7; 8dfb367 regression fix).
            "sources": _apply_preview_cap(sources, "text"),
            "actions": _dict_list(actions),
            "chunk_refs": _apply_preview_cap(chunk_refs, "preview"),
            "answer_confidence": _opt_float(answer_confidence),
            "clarification": _opt_dict(clarification),
        }
        self._client.rpush(key, orjson.dumps(row))
        self._client.hset(
            self._meta_key(scope, meta.conversation_id),
            mapping={
                "updated_at_ms": now,
                "message_count": meta.message_count + 1,
            },
        )
        self._client.zadd(self._index_key(scope), {meta.conversation_id: now})

    def record_doc_studied(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        entry: dict,
    ) -> ConversationMeta:
        meta = self.ensure_conversation(
            tenant_id=tenant_id,
            subject=subject,
            project_id=project_id,
            conversation_id=conversation_id,
        )
        if not isinstance(entry, dict) or not str(entry.get("document_id") or "").strip():
            return meta
        scope = self._scope(tenant_id, subject, project_id)
        meta_key = self._meta_key(scope, meta.conversation_id)
        ledger = _decode_dict_list(self._client.hgetall(meta_key).get("docs_studied"))
        stamped = dict(entry)
        stamped.setdefault("ts", self._now())
        ledger.append(stamped)
        # Same cap convention as the chunk-ref context budget: newest kept.
        ledger = ledger[-RAG_TURN_CONTEXT_MAX_CHUNK_REFS:]
        self._client.hset(
            meta_key,
            mapping={
                "docs_studied": _encode_json_list(ledger),
                "updated_at_ms": self._now(),
            },
        )
        meta.docs_studied = ledger
        return meta

    def build_context(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        turn_window: int | None = None,
    ) -> MemoryContext:
        scope = self._scope(tenant_id, subject, project_id)
        meta_raw = self._client.hgetall(self._meta_key(scope, conversation_id))
        summary_text = str(meta_raw.get("summary_text", "")) if meta_raw else ""
        turns = self.get_turns(
            tenant_id=tenant_id,
            subject=subject,
            project_id=project_id,
            conversation_id=conversation_id,
            limit=max(MEMORY_MAX_RECENT_TURNS * 3, 30),
        )
        recent = trim_turns_to_budget(
            turns,
            max_turns=turn_window or MEMORY_MAX_RECENT_TURNS,
            max_tokens_estimate=MEMORY_MAX_CONTEXT_TOKENS_ESTIMATE,
        )
        context_text = build_context_text(summary_text, recent)
        # Structured sibling of the prose context (design §7). Built over the
        # full fetched window — not the token-trimmed prose window — so chunk
        # refs and a pending clarification are never lost to prose trimming.
        structured = build_structured_context(
            summary_text,
            turns,
            _decode_dict_list(meta_raw.get("docs_studied")) if meta_raw else [],
            max_recent_pairs=turn_window or MEMORY_MAX_RECENT_TURNS,
            max_chunk_refs=RAG_TURN_CONTEXT_MAX_CHUNK_REFS,
            preview_chars=RAG_TURN_CONTEXT_PREVIEW_CHARS,
        )
        return MemoryContext(
            conversation_id=conversation_id,
            summary_text=summary_text,
            recent_turns=recent,
            context_text=context_text,
            structured=structured,
        )

    def _llm_summarize(
        self,
        turns: list[ConversationTurn],
        existing_summary: str,
        docs_studied: list[dict] | None = None,
    ) -> str:
        """Summarize turns using the configured LLM, with heuristic fallback.

        The input assembly includes the structured turn-loop grounding
        (design §7): the docs-studied ledger and any clarifications asked on
        the turns being compacted, so the rolling summary keeps the anchors
        later turns deepen into and the meaning of short user replies.

        Args:
            turns: Turns to compact, oldest-to-newest.
            existing_summary: The current rolling summary (may be empty).
            docs_studied: Deep-study ledger entries from the conversation
                meta (optional so pre-migration callers keep working).

        Returns:
            The new rolling summary text.
        """
        if not turns:
            return existing_summary
        context_parts: list[str] = []
        if existing_summary:
            context_parts.append("Existing summary:\n" + existing_summary)
        docs_grounding = render_docs_studied_grounding(docs_studied or [])
        if docs_grounding:
            context_parts.append(docs_grounding)
        for turn in turns[-MEMORY_SUMMARY_MAX_SOURCE_TURNS :]:
            context_parts.append(f"{turn.role.upper()}: {turn.content}")
            clarify_grounding = render_clarification_grounding(
                getattr(turn, "clarification", None)
            )
            if clarify_grounding:
                context_parts.append(clarify_grounding)
        messages = [
            {
                "role": "system",
                "content": (
                    "Summarize this conversation history for future follow-up Q&A. "
                    "Keep factual constraints, user goals, and unresolved tasks. "
                    "Return concise bullet points."
                ),
            },
            {"role": "user", "content": "\n\n".join(context_parts)},
        ]
        try:
            response = self._llm_provider.generate(
                messages, model_alias="default", max_tokens=RAG_MEMORY_LLM_SUMMARIZER_MAX_TOKENS
            )
            if response.content:
                return sanitize_memory_text(response.content, max_chars=RAG_MEMORY_LLM_SUMMARY_SANITIZED_MAX_CHARS)
        except Exception:
            logger.debug("LLM summarization failed, using heuristic", exc_info=True)
        return summarize_heuristic(turns)

    def _summarize_with_grounding(
        self,
        turns: list[ConversationTurn],
        existing_summary: str,
        docs_studied: list[dict],
    ) -> str:
        """Dispatch to ``_llm_summarize``, tolerating pre-migration overrides.

        ``_llm_summarize`` is a documented override/stub seam (tests and
        subclasses replace it with two-argument callables written before the
        ``docs_studied`` parameter existed). Inspect the bound callable and
        only pass the structured grounding when it is accepted — the same
        tolerance posture as :func:`_decode_id_list` applies to data.
        """
        summarize = self._llm_summarize
        try:
            params = inspect.signature(summarize).parameters
        except (TypeError, ValueError):
            params = {}
        accepts_grounding = "docs_studied" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if accepts_grounding:
            return summarize(turns, existing_summary, docs_studied=docs_studied)
        return summarize(turns, existing_summary)

    def compact_if_needed(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        force: bool = False,
    ) -> ConversationSummary:
        scope = self._scope(tenant_id, subject, project_id)
        meta_key = self._meta_key(scope, conversation_id)
        raw = self._client.hgetall(meta_key)
        if not raw:
            return ConversationSummary()
        count = int(raw.get("message_count", "0") or 0)
        if not force and count < MEMORY_SUMMARY_TRIGGER_TURNS:
            return ConversationSummary(
                text=str(raw.get("summary_text", "")),
                updated_at_ms=int(raw.get("summary_updated_at_ms", "0") or 0),
                turns_compacted=int(raw.get("summary_turns_compacted", "0") or 0),
            )
        turns = self.get_turns(
            tenant_id=tenant_id,
            subject=subject,
            project_id=project_id,
            conversation_id=conversation_id,
            limit=max(MEMORY_SUMMARY_MAX_SOURCE_TURNS, count),
        )
        summary_text = self._summarize_with_grounding(
            turns,
            str(raw.get("summary_text", "")),
            _decode_dict_list(raw.get("docs_studied")),
        )
        now = self._now()
        turns_compacted = len(turns)
        self._client.hset(
            meta_key,
            mapping={
                "summary_text": summary_text,
                "summary_updated_at_ms": now,
                "summary_turns_compacted": turns_compacted,
                "updated_at_ms": now,
            },
        )
        return ConversationSummary(
            text=summary_text,
            updated_at_ms=now,
            turns_compacted=turns_compacted,
        )

    def delete_conversation(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
    ) -> bool:
        scope = self._scope(tenant_id, subject, project_id)
        meta_key = self._meta_key(scope, conversation_id)
        if not self._client.exists(meta_key):
            return False
        turns_key = self._turns_key(scope, conversation_id)
        index_key = self._index_key(scope)
        self._client.delete(meta_key)
        self._client.delete(turns_key)
        self._client.zrem(index_key, conversation_id)
        return True

    def _load_doc_lists(self, meta_key: str) -> tuple[list[str], list[str]]:
        raw = self._client.hgetall(meta_key)
        return (
            _decode_id_list(raw.get("relevant_doc_ids")),
            _decode_id_list(raw.get("ignored_doc_ids")),
        )

    def _persist_doc_lists(
        self,
        meta_key: str,
        relevant: list[str],
        ignored: list[str],
    ) -> None:
        self._client.hset(
            meta_key,
            mapping={
                "relevant_doc_ids": _encode_id_list(relevant),
                "ignored_doc_ids": _encode_id_list(ignored),
                "updated_at_ms": self._now(),
            },
        )

    def update_conversation_title(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        title: str,
    ) -> ConversationMeta | None:
        scope = self._scope(tenant_id, subject, project_id)
        meta_key = self._meta_key(scope, conversation_id)
        if not self._client.exists(meta_key):
            return None
        cleaned = (title or "").strip()[:RAG_MEMORY_CONVERSATION_TITLE_MAX_CHARS] or "New conversation"
        now = self._now()
        self._client.hset(
            meta_key,
            mapping={
                "title": cleaned,
                "updated_at_ms": now,
            },
        )
        raw = self._client.hgetall(meta_key)
        return self._meta_from_hash(raw, conversation_id)

    def mark_retrieved(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        doc_ids: list[str],
    ) -> ConversationMeta:
        meta = self.ensure_conversation(
            tenant_id=tenant_id,
            subject=subject,
            project_id=project_id,
            conversation_id=conversation_id,
        )
        if not doc_ids:
            return meta
        scope = self._scope(tenant_id, subject, project_id)
        meta_key = self._meta_key(scope, meta.conversation_id)
        relevant, ignored = self._load_doc_lists(meta_key)
        seen = set(relevant) | set(ignored)
        new_relevant = list(relevant)
        for did in doc_ids:
            if did and did not in seen:
                new_relevant.append(did)
                seen.add(did)
        if new_relevant != relevant:
            self._persist_doc_lists(meta_key, new_relevant, ignored)
        meta.relevant_doc_ids = new_relevant
        meta.ignored_doc_ids = ignored
        return meta

    def move_to_ignored(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        doc_id: str,
    ) -> ConversationMeta:
        meta = self.ensure_conversation(
            tenant_id=tenant_id,
            subject=subject,
            project_id=project_id,
            conversation_id=conversation_id,
        )
        scope = self._scope(tenant_id, subject, project_id)
        meta_key = self._meta_key(scope, meta.conversation_id)
        relevant, ignored = self._load_doc_lists(meta_key)
        new_relevant = [d for d in relevant if d != doc_id]
        if doc_id and doc_id not in ignored:
            new_ignored = ignored + [doc_id]
        else:
            new_ignored = ignored
        if new_relevant != relevant or new_ignored != ignored:
            self._persist_doc_lists(meta_key, new_relevant, new_ignored)
        meta.relevant_doc_ids = new_relevant
        meta.ignored_doc_ids = new_ignored
        return meta

    def restore_to_relevant(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        doc_id: str,
    ) -> ConversationMeta:
        meta = self.ensure_conversation(
            tenant_id=tenant_id,
            subject=subject,
            project_id=project_id,
            conversation_id=conversation_id,
        )
        scope = self._scope(tenant_id, subject, project_id)
        meta_key = self._meta_key(scope, meta.conversation_id)
        relevant, ignored = self._load_doc_lists(meta_key)
        new_ignored = [d for d in ignored if d != doc_id]
        if doc_id and doc_id not in relevant:
            new_relevant = relevant + [doc_id]
        else:
            new_relevant = relevant
        if new_relevant != relevant or new_ignored != ignored:
            self._persist_doc_lists(meta_key, new_relevant, new_ignored)
        meta.relevant_doc_ids = new_relevant
        meta.ignored_doc_ids = new_ignored
        return meta

    def clear_doc_state(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
        doc_id: str,
    ) -> ConversationMeta:
        meta = self.ensure_conversation(
            tenant_id=tenant_id,
            subject=subject,
            project_id=project_id,
            conversation_id=conversation_id,
        )
        scope = self._scope(tenant_id, subject, project_id)
        meta_key = self._meta_key(scope, meta.conversation_id)
        relevant, ignored = self._load_doc_lists(meta_key)
        new_relevant = [d for d in relevant if d != doc_id]
        new_ignored = [d for d in ignored if d != doc_id]
        if new_relevant != relevant or new_ignored != ignored:
            self._persist_doc_lists(meta_key, new_relevant, new_ignored)
        meta.relevant_doc_ids = new_relevant
        meta.ignored_doc_ids = new_ignored
        return meta

    def get_seen_doc_ids(
        self,
        *,
        tenant_id: str,
        subject: str,
        project_id: str | None,
        conversation_id: str,
    ) -> list[str]:
        scope = self._scope(tenant_id, subject, project_id)
        meta_key = self._meta_key(scope, conversation_id)
        if not self._client.exists(meta_key):
            return []
        relevant, ignored = self._load_doc_lists(meta_key)
        # Preserve relative ordering: relevant first, then ignored, dedup.
        seen: list[str] = []
        seen_set: set[str] = set()
        for d in (*relevant, *ignored):
            if d and d not in seen_set:
                seen.append(d)
                seen_set.add(d)
        return seen


_MEMORY: ConversationMemoryProvider | None = None

# Redis-reconnect backstop. When Redis is unreachable we serve an *uncached*
# no-op and retry Redis at most once per this interval, so history recovers
# automatically once Redis returns — without a process restart, and without
# adding the ~1s connect-timeout to every request during an outage.
_REDIS_RETRY_INTERVAL_S = 5.0
_last_redis_attempt_at: float | None = None


def get_conversation_memory() -> ConversationMemoryProvider:
    """Resolve the configured conversation memory provider singleton.

    Returns:
        The configured `ConversationMemoryProvider`.
    """

    global _MEMORY, _last_redis_attempt_at
    if _MEMORY is not None:
        return _MEMORY
    if not MEMORY_ENABLED:
        _MEMORY = NoopConversationMemory()
        return _MEMORY
    provider = MEMORY_PROVIDER.strip().lower()
    if provider == "redis":
        # Rate-limit reconnect attempts so a prolonged outage doesn't add the
        # connect-timeout to every request; between attempts serve a transient
        # no-op immediately.
        now = time.monotonic()
        if (
            _last_redis_attempt_at is not None
            and (now - _last_redis_attempt_at) < _REDIS_RETRY_INTERVAL_S
        ):
            return NoopConversationMemory()
        _last_redis_attempt_at = now
        try:
            candidate = RedisConversationMemory(MEMORY_REDIS_URL, MEMORY_REDIS_PREFIX)
            # `redis.from_url()` is lazy — connection errors only surface on
            # first command. Eagerly ping to detect a dead Redis here.
            candidate._client.ping()
            _MEMORY = candidate
            logger.info("Conversation memory connected to Redis.")
            return _MEMORY
        except Exception as exc:
            # Do NOT cache the no-op. get_conversation_memory() is called per
            # request, so returning an *uncached* no-op means the next attempt
            # retries Redis; history recovers automatically once Redis is back,
            # with no restart. (The previous behaviour latched to no-op for the
            # whole process lifetime, silently hiding ALL history whenever the
            # API started during a Redis outage.)
            logger.warning(
                "Conversation memory Redis unavailable; serving no-op and "
                "retrying every %.0fs (auto-recovers, no restart needed): %s",
                _REDIS_RETRY_INTERVAL_S,
                exc,
            )
            return NoopConversationMemory()
    logger.warning("Unsupported memory provider '%s'; using no-op.", provider)
    _MEMORY = NoopConversationMemory()
    return _MEMORY


def conversation_meta_to_dict(meta: ConversationMeta) -> dict[str, Any]:
    """Convert conversation metadata to a JSON-serializable dict."""
    payload = asdict(meta)
    return payload


# Turn-loop extension fields (design §7) omitted from serialized turns when
# empty/None so pre-loop consumers (workflow payloads, history endpoints) keep
# receiving the legacy turn shape byte-identically; loop turns carry them.
_OPTIONAL_TURN_FIELDS = ("actions", "chunk_refs", "answer_confidence", "clarification")


def conversation_turns_to_dict(turns: list[ConversationTurn]) -> list[dict[str, Any]]:
    """Convert conversation turns to JSON-serializable dicts.

    The turn-loop extension fields (``actions`` / ``chunk_refs`` /
    ``answer_confidence`` / ``clarification``) are additive and default-empty;
    they are dropped from the dict when unpopulated so non-loop turns
    serialize exactly as before the extension (tolerant decoders on the read
    side treat absent and empty identically).
    """
    payloads: list[dict[str, Any]] = []
    for turn in turns:
        payload = asdict(turn)
        for key in _OPTIONAL_TURN_FIELDS:
            if payload.get(key) is None or payload.get(key) == []:
                payload.pop(key, None)
        payloads.append(payload)
    return payloads


__all__ = [
    "ConversationMemoryProvider",
    "conversation_meta_to_dict",
    "conversation_turns_to_dict",
    "get_conversation_memory",
]
