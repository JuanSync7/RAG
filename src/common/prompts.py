# @summary
# Shared prompt-template helpers for prompt files under config.settings
# PROMPTS_DIR: lazy-cached loading, {{ var }} token rendering (plain
# str.replace — no regex), and leading <think>-block stripping for
# reasoning-model output. The canonical home for the loader/render/strip trio
# previously duplicated per package — NEW code (turn_loop and later) imports
# from here; the existing agentic/deep-research private copies are left
# untouched by policy (compat-alias rule, CLAUDE.md §2/§6).
# Exports: load_prompt, render, strip_reasoning
# Deps: pathlib (config.settings lazily inside load_prompt)
# @end-summary
"""Shared prompt loader/renderer/reasoning-stripper for ``prompts/*.md``.

Semantics mirror the agentic loop's private helpers
(``src/retrieval/pipeline/agentic/hyde.py``) exactly, so migrating a caller
here is behavior-preserving: templates use the ``{{ var }}`` convention
(``prompts/README.md``), rendering is simple ``str.replace`` over the provided
variables only (an unknown token is left verbatim — visible in output rather
than silently dropped), and ``strip_reasoning`` removes a leading
``<think>...</think>`` block before salvage-parsing model output.
"""

from __future__ import annotations

from pathlib import Path

# Lazy module-level cache: templates are small, immutable at runtime, and read
# repeatedly per request — one disk read per filename per process.
_PROMPT_CACHE: dict[str, str] = {}


def load_prompt(filename: str) -> str:
    """Load (and cache) a prompt template from ``config.settings.PROMPTS_DIR``.

    ``config.settings`` is imported lazily so importing this module has zero
    side effects (pure helper surface; testable without env).

    Args:
        filename: Template filename relative to ``PROMPTS_DIR``
            (e.g. ``"turn_controller_decide.md"``).

    Returns:
        The template text.

    Raises:
        RuntimeError: If the file is missing/unreadable — the message names
            the full path (a packaging/deployment error, not a user error).
    """
    cached = _PROMPT_CACHE.get(filename)
    if cached is not None:
        return cached
    from config.settings import PROMPTS_DIR

    path = Path(PROMPTS_DIR) / filename
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Prompt template not found at {path}: {exc}") from exc
    _PROMPT_CACHE[filename] = text
    return text


def render(template: str, **vars_: str) -> str:
    """Substitute ``{{ var }}`` tokens in ``template`` with ``vars_`` values.

    Plain ``str.replace`` over the provided variables only — no regex, no
    templating engine (CLAUDE.md §0; same semantics as the agentic ``_render``).
    A token not present in ``vars_`` is left verbatim in the output.

    Args:
        template: Template text using the ``{{ var }}`` convention.
        **vars_: Variable name → replacement text.

    Returns:
        The rendered prompt.
    """
    rendered = template
    for key, value in vars_.items():
        rendered = rendered.replace("{{ " + key + " }}", value)
    return rendered


def strip_reasoning(text: str) -> str:
    """Drop a leading ``<think>...</think>`` reasoning block from LLM output.

    Reasoning models prepend a hidden-reasoning block; stripping it lets the
    remainder salvage-parse cleanly. No-op for plain output. Splits on the
    LAST closing tag (mirroring the agentic ``_strip_reasoning``) so nested or
    repeated think blocks are all removed.

    Args:
        text: Raw model output.

    Returns:
        The text after the final ``</think>`` tag, or ``text`` unchanged when
        no tag is present.
    """
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    return text


__all__ = ["load_prompt", "render", "strip_reasoning"]
