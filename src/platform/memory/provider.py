# @summary
# Conversation memory providers (Redis canonical backend + no-op fallback) and singleton factory.
# Exports: ConversationMemoryProvider, NoopConversationMemory, RedisConversationMemory,
#          get_conversation_memory, conversation_meta_to_dict, conversation_turns_to_dict
# Deps: config.settings, json, logging, uuid, dataclasses, src.platform.memory.utils, src.platform.llm
# @end-summary
"""Conversation memory providers and factory.

This module provides a Redis-backed conversation memory implementation plus a
no-op fallback when memory is disabled or unavailable. It also exposes a
process-wide singleton resolver for use by API and CLI surfaces.
"""

from __future__ import annotations

import orjson
import logging
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
)
from src.platform.memory.schemas import (
    ConversationMeta,
    ConversationSummary,
    ConversationTurn,
    MemoryContext,
)
from src.platform.memory.utils import (
    build_context_text,
    now_ms,
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
    ) -> None:
        return

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
    ) -> None:
        if not content.strip() and not (sources or []):
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
            "sources": sources or [],
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
        return MemoryContext(
            conversation_id=conversation_id,
            summary_text=summary_text,
            recent_turns=recent,
            context_text=context_text,
        )

    def _llm_summarize(self, turns: list[ConversationTurn], existing_summary: str) -> str:
        """Summarize turns using the configured LLM, with heuristic fallback."""
        if not turns:
            return existing_summary
        context_parts: list[str] = []
        if existing_summary:
            context_parts.append("Existing summary:\n" + existing_summary)
        for turn in turns[-MEMORY_SUMMARY_MAX_SOURCE_TURNS :]:
            context_parts.append(f"{turn.role.upper()}: {turn.content}")
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
        summary_text = self._llm_summarize(turns, str(raw.get("summary_text", "")))
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


def get_conversation_memory() -> ConversationMemoryProvider:
    """Resolve the configured conversation memory provider singleton.

    Returns:
        The configured `ConversationMemoryProvider`.
    """

    global _MEMORY
    if _MEMORY is not None:
        return _MEMORY
    if not MEMORY_ENABLED:
        _MEMORY = NoopConversationMemory()
        return _MEMORY
    provider = MEMORY_PROVIDER.strip().lower()
    if provider == "redis":
        try:
            candidate = RedisConversationMemory(MEMORY_REDIS_URL, MEMORY_REDIS_PREFIX)
            # `redis.from_url()` is lazy — connection errors only surface on
            # first command. Eagerly ping so we can fall back to the no-op
            # provider here rather than 500ing later inside a request handler.
            candidate._client.ping()
            _MEMORY = candidate
            return _MEMORY
        except Exception as exc:
            logger.warning("Memory Redis unavailable, falling back to no-op memory: %s", exc)
            _MEMORY = NoopConversationMemory()
            return _MEMORY
    logger.warning("Unsupported memory provider '%s'; using no-op.", provider)
    _MEMORY = NoopConversationMemory()
    return _MEMORY


def conversation_meta_to_dict(meta: ConversationMeta) -> dict[str, Any]:
    """Convert conversation metadata to a JSON-serializable dict."""
    payload = asdict(meta)
    return payload


def conversation_turns_to_dict(turns: list[ConversationTurn]) -> list[dict[str, Any]]:
    """Convert conversation turns to JSON-serializable dicts."""
    return [asdict(turn) for turn in turns]


__all__ = [
    "ConversationMemoryProvider",
    "conversation_meta_to_dict",
    "conversation_turns_to_dict",
    "get_conversation_memory",
]
