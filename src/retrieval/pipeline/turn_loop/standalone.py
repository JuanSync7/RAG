# @summary
# Standalone-query resolution for the turn loop: before retrieval, rewrite a
# follow-up whose meaning depends on prior turns (unresolved pronouns /
# back-references) into a self-contained retrieval query, reusing the shared
# ``retrieval_query_rewriter`` prompt through the turn's charged LLM wrapper.
# A fresh conversation (no prior turns) skips the LLM entirely. Keys on the
# conversation HAVING history, never on query content (CLAUDE.md §0), and fails
# open to the literal query on any error. Retrieval/HyDE is seeded from this
# history-stripped query while generation still receives full memory — the fix
# for conversation memory poisoning the retrieval seed.
# Exports: resolve_standalone_query, render_history
# Deps: src.common.prompts, src.common.utils, src.retrieval.pipeline.turn_loop
#       .{schemas,events,controller} (config.settings lazily)
# @end-summary
"""Standalone-query resolution: history-stripped retrieval seed for follow-ups.

The turn loop bypasses ``rag_chain`` (it runs in the API process), so it does
not inherit the query-processor's coreference rewrite that the legacy/agentic
paths get for free. Without it, a follow-up like *"how does it compare to X?"*
is embedded verbatim — the pronoun never resolves, retrieval drifts, and the
answer hedges ("the question is ambiguous"). This stage reuses the SAME shared
rewriter prompt (the domain artifact) to resolve the reference against prior
turns and produce the query that seeds retrieval/HyDE. Generation still sees the
full :class:`TurnContext`, so memory grounds the answer without poisoning the
retrieval seed (see docs / memory: multi-turn-memory-poisons-retrieval).
"""

from __future__ import annotations

import logging

from src.common.prompts import load_prompt, strip_reasoning
from src.common.utils import parse_json_object
from src.retrieval.pipeline.turn_loop.controller import controller_model_alias
from src.retrieval.pipeline.turn_loop.events import TurnEventEmitter
from src.retrieval.pipeline.turn_loop.schemas import (
    TurnBudget,
    TurnContext,
    TurnState,
)

logger = logging.getLogger(__name__)

# Shared rewriter prompt (also driven by the query-processor Auto sub-mode) —
# the single source of the coreference-resolution behavior (CLAUDE.md §2 reuse).
_PROMPT_FILE = "retrieval_query_rewriter.md"

# The rewriter's three history strategies; anything else (or a missing field)
# is treated as ``use_as_is`` — leave the literal query untouched.
_VALID_DECISIONS = {"use_as_is", "partial_history", "full_history"}

# Layout cap for the rendered history digest fed to the rewriter prompt (a
# prompt-layout constant, not a behavior tunable — those live in RAG_TURN_LOOP_*
# settings). Older material is already compacted into ``rolling_summary``, so the
# head carries the resolvable references.
HISTORY_DIGEST_MAX_CHARS = 4000


def _enabled() -> bool:
    """Whether follow-up standalone-query resolution runs
    (``RAG_TURN_LOOP_STANDALONE_QUERY_ENABLED``, default on). Off restores the
    verbatim-query behavior (retrieval seeded from the literal follow-up)."""
    from config import settings

    return bool(
        getattr(settings, "RAG_TURN_LOOP_STANDALONE_QUERY_ENABLED", True)
    )


def _max_tokens() -> int:
    """Completion cap for the rewriter call — reuses the shared rewriter budget
    (``RAG_RETRIEVAL_REWRITER_MAX_TOKENS``); the output is a small JSON object."""
    from config import settings

    return int(getattr(settings, "RAG_RETRIEVAL_REWRITER_MAX_TOKENS", 256))


def render_history(context: TurnContext, *, max_chars: int) -> str:
    """Render prior turns as ``ROLE: content`` lines for the rewriter prompt.

    Pure string assembly over the typed context (no LLM, no content matching):
    the compacted ``rolling_summary`` then each recent Q/A turn. Returns an
    empty string when the conversation has no prior turns (a fresh first turn),
    which the resolver uses to skip the LLM call entirely.

    Args:
        context: The cross-turn context for this conversation.
        max_chars: Head-truncation cap for the digest (<= 0 means no cap).

    Returns:
        The rewriter-ready ``CONVERSATION_HISTORY`` block, or ``""`` when there
        is no prior conversation to resolve against.
    """
    lines: list[str] = []
    if context.rolling_summary:
        lines.append(f"SUMMARY: {context.rolling_summary}")
    for turn in context.recent_turns:
        if not isinstance(turn, dict):
            continue
        user = str(turn.get("query") or "").strip()
        assistant = str(turn.get("answer") or "").strip()
        if user:
            lines.append(f"USER: {user}")
        if assistant:
            lines.append(f"ASSISTANT: {assistant}")
    digest = "\n".join(lines).strip()
    if max_chars > 0 and len(digest) > max_chars:
        digest = digest[:max_chars]
    return digest


async def resolve_standalone_query(
    *,
    query: str,
    context: TurnContext,
    state: TurnState,
    budget: TurnBudget,
    emitter: TurnEventEmitter,
) -> str:
    """Resolve ``query`` into a self-contained retrieval query using history.

    Returns the literal ``query`` unchanged when the feature is disabled, the
    conversation has no prior turns (fresh first turn — no LLM call), the LLM
    ledger/wall-clock is exhausted, or the rewriter decides ``use_as_is`` /
    fails / returns an empty rewrite. Only a ``partial_history`` /
    ``full_history`` decision carrying a non-empty, changed rewrite replaces the
    query — so a self-contained question is never altered and a flaky rewriter
    never blocks the turn (fail-open, design §6). Charged to the turn's single
    LLM ledger via :meth:`TurnEventEmitter.charged_call`.

    Args:
        query: The user's verbatim turn query.
        context: The cross-turn context (history to resolve against).
        state: The turn's mutable state (only read for logging here; the
            ledger charge happens inside ``charged_call``).
        budget: The frozen per-turn budget (unused directly; ``charged_call``
            reads it for the ledger/timeout).
        emitter: The turn's event emitter / charged-call wrapper.

    Returns:
        The resolved standalone query, or the literal ``query`` on any
        no-op / failure path.
    """
    if not _enabled():
        return query
    history = render_history(context, max_chars=HISTORY_DIGEST_MAX_CHARS)
    if not history:
        # Fresh first turn: nothing to resolve, and — critically — no LLM call,
        # so single-turn latency is untouched.
        return query

    prompt = (
        f"{load_prompt(_PROMPT_FILE)}\n\n"
        f"USER_QUERY: {query}\n\n"
        f"CONVERSATION_HISTORY:\n{history}"
    )
    response = await emitter.charged_call(
        alias=controller_model_alias(),
        purpose="standalone_query",
        prompt=prompt,
        temperature=0.0,
        max_tokens=_max_tokens(),
    )
    if response is None:
        return query

    parsed = parse_json_object(
        strip_reasoning(getattr(response, "content", "") or "")
    ) or {}
    decision = str(parsed.get("decision", "")).strip().lower()
    rewritten = str(parsed.get("processed_query", "") or "").strip()
    if (
        decision not in _VALID_DECISIONS
        or decision == "use_as_is"
        or not rewritten
        or rewritten == query
    ):
        return query

    logger.info(
        "turn loop standalone-query rewrite (%s): %r -> %r",
        decision,
        query[:80],
        rewritten[:80],
    )
    return rewritten


__all__ = ["resolve_standalone_query", "render_history"]
