# @summary
# Helpers for memory trimming, context formatting, token estimation, and safe text normalization.
# Exports: now_ms, sanitize_memory_text, estimate_token_count, trim_turns_to_budget,
#          build_context_text, summarize_heuristic
# Deps: re, time, src.platform.memory.schemas
# @end-summary
"""Utility helpers for memory trimming and context formatting."""

from __future__ import annotations

import re
import time

from config.settings import (
    RAG_MEMORY_SNIPPET_MAX_CHARS,
    RAG_MEMORY_SUMMARY_MAX_CHARS,
    RAG_MEMORY_TURN_MAX_CHARS,
    RAG_MEMORY_SANITIZE_DEFAULT_MAX_CHARS,
    RAG_MEMORY_HEURISTIC_SUMMARY_MAX_CHARS,
)
from src.platform.memory.schemas import ConversationTurn


def now_ms() -> int:
    """Return current time in milliseconds since epoch."""
    return int(time.time() * 1000)


def sanitize_memory_text(
    text: str,
    max_chars: int = RAG_MEMORY_SANITIZE_DEFAULT_MAX_CHARS,
    *,
    preserve_newlines: bool = False,
) -> str:
    """Normalize whitespace and cap text length for prompt safety.

    Args:
        text: Input text.
        max_chars: Max characters to retain (ellipsis may be added).
        preserve_newlines: When True, keep line structure — collapse only
            *horizontal* whitespace, trim spaces around newlines, and cap
            blank-line runs at one. This is essential for STORED turn content:
            collapsing every ``\\n`` to a space (the default) destroys the
            assistant answer's markdown (lists/paragraphs) so it renders as a
            wall of text when a conversation is reloaded from history. The
            default (False) keeps the legacy single-line behavior for
            prompt-context/summary text where structure is not needed.

    Returns:
        Cleaned, bounded text.
    """

    raw = text or ""
    if preserve_newlines:
        clean = re.sub(r"[^\S\n]+", " ", raw)      # collapse spaces/tabs, keep newlines
        clean = re.sub(r" *\n *", "\n", clean)      # trim whitespace around newlines
        clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    else:
        clean = re.sub(r"\s+", " ", raw).strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3] + "..."


def estimate_token_count(text: str) -> int:
    """Estimate token count using a cheap character heuristic.

    Args:
        text: Input text.

    Returns:
        Estimated token count (>= 0).
    """

    if not text:
        return 0
    # Rough rule: ~4 chars per token for English-like text.
    return max(1, len(text) // 4)


def trim_turns_to_budget(
    turns: list[ConversationTurn],
    *,
    max_turns: int,
    max_tokens_estimate: int,
) -> list[ConversationTurn]:
    """Keep newest turns bounded by count and estimated token budget.

    Args:
        turns: Conversation turns in chronological order.
        max_turns: Maximum number of recent turns to consider.
        max_tokens_estimate: Estimated token budget for selected turns.

    Returns:
        Selected recent turns, oldest-to-newest.
    """

    if not turns:
        return []
    limited = turns[-max(1, int(max_turns)) :]
    selected: list[ConversationTurn] = []
    total_tokens = 0
    for turn in reversed(limited):
        t = estimate_token_count(turn.content)
        if selected and (total_tokens + t) > max_tokens_estimate:
            break
        selected.append(turn)
        total_tokens += t
    selected.reverse()
    return selected


def build_context_text(summary_text: str, recent_turns: list[ConversationTurn]) -> str:
    """Compose canonical memory context block for prompt injection.

    Args:
        summary_text: Rolling summary text.
        recent_turns: Recent conversation turns.

    Returns:
        Rendered context text suitable for inclusion in a prompt.
    """

    parts: list[str] = []
    if summary_text:
        parts.append("Conversation summary:\n" + sanitize_memory_text(summary_text, max_chars=RAG_MEMORY_SUMMARY_MAX_CHARS))
    if recent_turns:
        rendered = []
        for turn in recent_turns:
            role = turn.role.upper()
            rendered.append(f"{role}: {sanitize_memory_text(turn.content, max_chars=RAG_MEMORY_TURN_MAX_CHARS)}")
        parts.append("Recent turns:\n" + "\n".join(rendered))
    return "\n\n".join(parts).strip()


def summarize_heuristic(turns: list[ConversationTurn], max_chars: int = RAG_MEMORY_HEURISTIC_SUMMARY_MAX_CHARS) -> str:
    """Create a heuristic compact summary when LLM summarization is unavailable.

    Args:
        turns: Conversation turns in chronological order.
        max_chars: Max output characters.

    Returns:
        A compact summary string.
    """

    if not turns:
        return ""
    lines: list[str] = ["Key conversation points:"]
    for turn in turns[-12:]:
        prefix = "User" if turn.role == "user" else "Assistant"
        lines.append(f"- {prefix}: {sanitize_memory_text(turn.content, max_chars=RAG_MEMORY_SNIPPET_MAX_CHARS)}")
    return sanitize_memory_text("\n".join(lines), max_chars=max_chars)


__all__ = [
    "build_context_text",
    "estimate_token_count",
    "now_ms",
    "sanitize_memory_text",
    "summarize_heuristic",
    "trim_turns_to_budget",
]
