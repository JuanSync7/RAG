# @summary
# Helpers for memory trimming, context formatting, token estimation, safe text
# normalization, and deterministic assembly of the structured turn-context dict
# (recent Q/A pairs, capped chunk refs, docs-studied ledger, pending
# clarification) plus grounding renderers consumed by the rolling summarizer.
# Exports: now_ms, sanitize_memory_text, estimate_token_count, trim_turns_to_budget,
#          build_context_text, build_structured_context, render_docs_studied_grounding,
#          render_clarification_grounding, summarize_heuristic
# Deps: re, time, src.platform.memory.schemas
# @end-summary
"""Utility helpers for memory trimming and context formatting.

The prose context builder (:func:`build_context_text`) is the legacy prompt
surface and is kept byte-identical for existing callers; its structured
sibling (:func:`build_structured_context`) assembles the typed turn-context
dict for the turn-level conversation loop (design
``docs/retrieval/TURN_LOOP_DESIGN.md`` §7). All helpers here are deterministic
and side-effect-light — no Redis, no LLM.
"""

from __future__ import annotations

import re
import time

from config.settings import (
    RAG_MEMORY_SNIPPET_MAX_CHARS,
    RAG_MEMORY_SUMMARY_MAX_CHARS,
    RAG_MEMORY_TURN_MAX_CHARS,
    RAG_MEMORY_SANITIZE_DEFAULT_MAX_CHARS,
    RAG_MEMORY_HEURISTIC_SUMMARY_MAX_CHARS,
    RAG_MEMORY_HEURISTIC_SUMMARY_TURNS,
    TOKEN_BUDGET_CHARS_PER_TOKEN,
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
        clean = raw.replace("\r\n", "\n").replace("\r", "\n")  # normalize line endings
        clean = re.sub(r"(?<=\S)[ \t]{2,}", " ", clean)         # collapse MID-line runs only
        clean = re.sub(r"[ \t]+\n", "\n", clean)                # trim trailing line whitespace
        clean = re.sub(r"\n{3,}", "\n\n", clean).strip()        # cap blank-line runs
        # Leading indentation is preserved on purpose: nested lists and
        # indented / fenced code blocks rely on it to render correctly.
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
    # Rough rule: ~N chars per token (operator-tunable heuristic, default 4).
    cpt = max(1, TOKEN_BUDGET_CHARS_PER_TOKEN)
    return max(1, len(text) // cpt)


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


def _pair_recent_qa(turns: list[ConversationTurn], max_pairs: int) -> list[dict]:
    """Fold chronological turns into ``{question, answer}`` pairs, newest kept.

    Each user turn opens a pair; the next assistant turn closes it. System
    turns are skipped. A trailing user turn without an assistant reply is kept
    with an empty ``answer`` (it is the still-open question). Only the newest
    ``max_pairs`` pairs are returned.

    Args:
        turns: Conversation turns, oldest-to-newest.
        max_pairs: Cap on returned pairs (<= 0 returns an empty list).

    Returns:
        List of ``{"question": str, "answer": str}`` dicts, oldest-to-newest.
    """

    if max_pairs <= 0:
        return []
    pairs: list[dict] = []
    open_pair: dict | None = None
    for turn in turns:
        if turn.role == "user":
            if open_pair is not None:
                pairs.append(open_pair)
            open_pair = {"question": turn.content, "answer": ""}
        elif turn.role == "assistant":
            if open_pair is not None:
                open_pair["answer"] = turn.content
                pairs.append(open_pair)
                open_pair = None
            else:
                # Assistant turn with no preceding user turn in the window —
                # keep it as an answer-only pair so context is not lost.
                pairs.append({"question": "", "answer": turn.content})
    if open_pair is not None:
        pairs.append(open_pair)
    return pairs[-max_pairs:]


def _collect_chunk_refs(
    turns: list[ConversationTurn],
    max_refs: int,
    preview_chars: int,
) -> list[dict]:
    """Gather chunk refs from the newest turns, capped and preview-bounded.

    Turns are walked newest-first so the most recent evidence wins the cap;
    within a turn the stored (ranked) order is preserved. Each ref is shallow-
    copied with its ``preview`` truncated to ``preview_chars`` so the context
    stays bounded even for records stored with
    ``RAG_TURN_CONTEXT_STORE_FULL_TEXT=true``.

    Args:
        turns: Conversation turns, oldest-to-newest.
        max_refs: Total ref cap across all turns (<= 0 returns an empty list).
        preview_chars: Per-ref preview character cap (<= 0 disables capping).

    Returns:
        List of ChunkRef dicts, newest turn first.
    """

    if max_refs <= 0:
        return []
    refs: list[dict] = []
    for turn in reversed(turns):
        for ref in getattr(turn, "chunk_refs", None) or []:
            if not isinstance(ref, dict):
                continue
            if len(refs) >= max_refs:
                return refs
            capped = dict(ref)
            preview = capped.get("preview")
            if isinstance(preview, str) and 0 < preview_chars < len(preview):
                capped["preview"] = preview[:preview_chars]
            refs.append(capped)
    return refs


def _pending_clarification(turns: list[ConversationTurn]) -> dict | None:
    """Return the last assistant turn's clarification if still unanswered.

    A clarification is *pending* when the conversation's last non-system turn
    is an assistant turn that ended ``ask_user`` (carries a ``clarification``
    payload). Any later user turn consumes it — the user has replied, so the
    next turn's controller resolves that reply against the hints instead.

    Args:
        turns: Conversation turns, oldest-to-newest.

    Returns:
        The ``{question, hints, scoping_questions}`` dict, or None.
    """

    for turn in reversed(turns):
        if turn.role == "user":
            return None  # a user reply after the clarification consumed it
        if turn.role == "assistant":
            clarification = getattr(turn, "clarification", None)
            return clarification if isinstance(clarification, dict) else None
    return None


def build_structured_context(
    summary_text: str,
    turns: list[ConversationTurn],
    docs_studied: list[dict],
    *,
    max_recent_pairs: int,
    max_chunk_refs: int,
    preview_chars: int,
) -> dict:
    """Assemble the structured turn-context dict (design §7).

    Structured sibling of :func:`build_context_text`: instead of one prose
    block it returns the typed cross-turn state the turn loop feeds into
    ``TurnContext`` — deterministic dict/str assembly only, so the same inputs
    always produce the same context (evalable/cachable).

    Args:
        summary_text: Rolling summary text from the conversation meta.
        turns: Conversation turns, oldest-to-newest (the provider's fetched
            window — wider than the trimmed prose window so pending
            clarifications and refs are not lost to token trimming).
        docs_studied: Deep-study ledger entries from the conversation meta.
        max_recent_pairs: Cap on ``recent_turns`` Q/A pairs.
        max_chunk_refs: Total cap on ``chunk_refs`` across turns
            (``RAG_TURN_CONTEXT_MAX_CHUNK_REFS``).
        preview_chars: Per-ref preview cap (``RAG_TURN_CONTEXT_PREVIEW_CHARS``).

    Returns:
        Dict with keys ``rolling_summary`` (str), ``recent_turns``
        (list[{question, answer}]), ``chunk_refs`` (list[dict], newest turn
        first), ``docs_studied`` (list[dict]) and ``pending_clarification``
        (dict | None).
    """

    return {
        "rolling_summary": summary_text or "",
        "recent_turns": _pair_recent_qa(turns, max_recent_pairs),
        "chunk_refs": _collect_chunk_refs(turns, max_chunk_refs, preview_chars),
        "docs_studied": [d for d in (docs_studied or []) if isinstance(d, dict)],
        "pending_clarification": _pending_clarification(turns),
    }


def render_docs_studied_grounding(docs_studied: list[dict]) -> str:
    """Render the docs-studied ledger as a grounding block for the summarizer.

    Compaction must preserve which documents were deep-studied and what was
    concluded (design §7) — otherwise the rolling summary loses the anchors
    later turns deepen into. Deterministic string assembly; empty input
    renders empty.

    Args:
        docs_studied: Deep-study ledger entries (``{document_id, windows_read,
            sections, conclusion, ts}`` dicts).

    Returns:
        A multi-line grounding block, or ``""`` when there is nothing to say.
    """

    lines: list[str] = []
    for doc in docs_studied or []:
        if not isinstance(doc, dict):
            continue
        document_id = str(doc.get("document_id") or "").strip()
        if not document_id:
            continue
        line = f"- document_id={document_id}"
        sections = doc.get("sections") or []
        if sections:
            line += " sections=" + ", ".join(str(s) for s in sections)
        conclusion = str(doc.get("conclusion") or "").strip()
        if conclusion:
            line += f": {conclusion}"
        lines.append(line)
    if not lines:
        return ""
    return "Documents deep-studied this conversation:\n" + "\n".join(lines)


def render_clarification_grounding(clarification: dict) -> str:
    """Render one asked clarification as a grounding line for the summarizer.

    Keeps the clarification question (and its offered hints) visible to
    compaction so a summary of older turns still explains *why* the user's
    short reply (e.g. "the second one") meant what it meant.

    Args:
        clarification: The ``{question, hints, scoping_questions}`` payload
            stored on an assistant turn.

    Returns:
        A single grounding line, or ``""`` for an empty/invalid payload.
    """

    if not isinstance(clarification, dict):
        return ""
    question = str(clarification.get("question") or "").strip()
    if not question:
        return ""
    line = f"ASSISTANT ASKED CLARIFICATION: {question}"
    hints = clarification.get("hints") or []
    if hints:
        line += " (hints: " + "; ".join(str(h) for h in hints) + ")"
    return line


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
    for turn in turns[-RAG_MEMORY_HEURISTIC_SUMMARY_TURNS:]:
        prefix = "User" if turn.role == "user" else "Assistant"
        lines.append(f"- {prefix}: {sanitize_memory_text(turn.content, max_chars=RAG_MEMORY_SNIPPET_MAX_CHARS)}")
    return sanitize_memory_text("\n".join(lines), max_chars=max_chars)


__all__ = [
    "build_context_text",
    "build_structured_context",
    "estimate_token_count",
    "now_ms",
    "render_clarification_grounding",
    "render_docs_studied_grounding",
    "sanitize_memory_text",
    "summarize_heuristic",
    "trim_turns_to_budget",
]
