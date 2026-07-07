# @summary
# Shared deterministic helpers for the turn_loop stage modules: the one-line
# whitespace-normalizing text flattener, the per-chunk preview char cap reader,
# and the event-payload top-preview count. Centralized here (CLAUDE.md §2) so the
# stage files do not each carry their own copy — previously duplicated across
# controller/orchestrator/deep_study (preview cap), controller/orchestrator
# (one_line), and retrieve/decompose (top-preview count).
# Exports: one_line, preview_chars, TOP_PREVIEW_COUNT
# Deps: config.settings (lazily)
# @end-summary
"""Shared side-effect-light helpers for the turn_loop stages (CLAUDE.md §2)."""

from __future__ import annotations

# How many kept chunks a retrieve/decompose result event previews (display cap
# only — never affects the pool). A layout constant of the event payload, not a
# behavior tunable.
TOP_PREVIEW_COUNT = 3


def one_line(text: str, max_chars: int) -> str:
    """Flatten ``text`` to one whitespace-normalized line capped at ``max_chars``.

    ``max_chars <= 0`` means no cap. Pure function (no I/O, no config).
    """
    flat = " ".join((text or "").split())
    if max_chars > 0 and len(flat) > max_chars:
        flat = flat[:max_chars]
    return flat


def preview_chars() -> int:
    """Per-chunk preview cap for digests/events (``RAG_TURN_CONTEXT_PREVIEW_CHARS``).

    Read lazily so importing a stage module has zero config side effects.
    """
    from config import settings

    return int(getattr(settings, "RAG_TURN_CONTEXT_PREVIEW_CHARS", 320))


__all__ = ["one_line", "preview_chars", "TOP_PREVIEW_COUNT"]
