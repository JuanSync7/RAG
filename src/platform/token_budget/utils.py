# @summary
# Token counting utility backed by litellm.token_counter() with character
# heuristic fallback for unknown models.
# Exports: count_tokens
# Deps: litellm, logging, config.settings
# @end-summary
"""Token counting helper for budget calculations.

Uses litellm.token_counter() for accurate per-model tokenization.
Falls back to a character-based heuristic when litellm cannot resolve the model.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import litellm

from config.settings import TOKEN_BUDGET_CHARS_PER_TOKEN

logger = logging.getLogger("rag.token_budget")


def count_tokens(
    text: Optional[str] = None,
    messages: Optional[list[dict[str, Any]]] = None,
    model: Optional[str] = None,
) -> int:
    """Count tokens using litellm.token_counter(), falling back to heuristic.

    Provide either ``text`` (single string) or ``messages`` (OpenAI-format
    message list).  If both are given, ``messages`` takes precedence.

    Args:
        text: Plain text to tokenize.
        messages: OpenAI-format message list (takes precedence over *text*).
        model: LiteLLM model string (e.g. ``"ollama/qwen2.5:3b"``).
               Used for accurate tokenizer selection.

    Returns:
        Token count (>= 0).
    """
    if messages is not None:
        if not messages:
            return 0
        try:
            return litellm.token_counter(model=model or "gpt-3.5-turbo", messages=messages)
        except Exception:
            # Fallback: concatenate message contents and use heuristic
            combined = " ".join(
                m.get("content", "") for m in messages if isinstance(m.get("content"), str)
            )
            return _heuristic_count(combined)

    if not text:
        return 0

    try:
        return litellm.token_counter(
            model=model or "gpt-3.5-turbo",
            text=text,
        )
    except Exception:
        return _heuristic_count(text)


def _heuristic_count(text: str, chars_per_token: int | None = None) -> int:
    """Count tokens using a character-length heuristic.

    Args:
        text: Plain text to estimate.
        chars_per_token: Optional heuristic override.

    Returns:
        Estimated token count (>= 0).
    """
    if not text:
        return 0
    cpt = max(1, chars_per_token if chars_per_token is not None else TOKEN_BUDGET_CHARS_PER_TOKEN)
    return max(1, len(text) // cpt)


__all__ = ["count_tokens"]
