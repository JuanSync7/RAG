# @summary
# HyDE generation for the agentic loop: ask the controller LLM to write a short
# hypothetical answer (embedded, answer-space) plus literal search_terms (BM25
# anchor) for the current question, conditioned on prior variants + the named
# gap. Fail-open: returns None on any error so the caller degrades to a plain
# hybrid search on the processed query (never worse than baseline).
# Exports: generate_hyde, HydeFailure
# Deps: src.retrieval.pipeline.agentic.state, src.retrieval.pipeline.deep_research (_parse_json_object)
# @end-summary
"""Controller-driven HyDE (Hypothetical Document Embedding) generation."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from src.retrieval.pipeline.agentic.state import HydeVariant
# Reuse the deep-research JSON salvage parser (tolerates code-fenced / noisy
# output). One-way import: deep_research never imports this package.
from src.retrieval.pipeline.deep_research import _parse_json_object

logger = logging.getLogger(__name__)

# This module is at src/retrieval/pipeline/agentic/ — four parents up is the
# repo root (one deeper than deep_research.py, which uses parents[3]).
_PROMPT_DIR = Path(__file__).resolve().parents[4] / "prompts"
_HYDE_PROMPT_FILE = _PROMPT_DIR / "agentic_hyde_generate.md"


class HydeFailure:
    """Typed reasons a HyDE generation fell open to literal-query retrieval.

    A bare ``hyde_failures`` counter says the loop is silently running plain
    literal-query retrieval but not WHY — and the cause changes the fix:
    ``EMPTY_CONTENT`` (a reasoning model spending its whole budget in ``<think>``)
    means re-point ``RAG_AGENTIC_CONTROLLER_MODEL_ALIAS`` at an instruct model;
    ``TIMEOUT`` means raise the budget; ``PROVIDER_ERROR`` means the endpoint is
    flaky; ``PARSE_INVALID`` / ``NO_HYPOTHESIS`` mean the prompt/JSON contract
    drifted. Plain ``str`` constants so they serialize into telemetry directly.
    """

    PROVIDER_ERROR: str = "provider_error"
    TIMEOUT: str = "timeout"
    EMPTY_CONTENT: str = "empty_content"
    PARSE_INVALID: str = "parse_invalid"
    NO_HYPOTHESIS: str = "no_hypothesis"


def _load_prompt() -> str:
    try:
        return _HYDE_PROMPT_FILE.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - packaging error
        raise RuntimeError(
            f"Agentic HyDE prompt not found at {_HYDE_PROMPT_FILE}: {exc}"
        ) from exc


def _render(template: str, **vars_: str) -> str:
    rendered = template
    for k, v in vars_.items():
        rendered = rendered.replace("{{ " + k + " }}", v)
    return rendered


def _strip_reasoning(text: str) -> str:
    """Drop a leading ``<think>…</think>`` reasoning block so a reasoning model's
    free-form output salvage-parses cleanly. No-op for plain output."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    return text


def _coerce_variant(obj: dict[str, Any], max_tokens: int) -> Optional[HydeVariant]:
    answer = str(obj.get("hypothetical_answer") or "").strip()
    if not answer:
        return None
    # Bound the embedded text defensively (the prompt also caps it). ~4 chars
    # per token is the project's token-budget assumption.
    char_cap = max(1, max_tokens) * 4
    if len(answer) > char_cap:
        answer = answer[:char_cap]
    raw_terms = obj.get("search_terms") or []
    if isinstance(raw_terms, str):
        raw_terms = [raw_terms]
    terms = [str(t).strip() for t in raw_terms if str(t).strip()]
    return HydeVariant(
        hypothetical_answer=answer,
        search_terms=terms,
        target_aspect=str(obj.get("target_aspect") or "").strip(),
    )


async def generate_hyde(
    provider,
    *,
    model_alias: str,
    original_question: str,
    tried_hyde: Optional[list[str]] = None,
    coverage_aspects: Optional[list[str]] = None,
    max_tokens: int,
    temperature: float,
    timeout_s: int,
    json_mode: bool = True,
    domain: str = "",
    on_failure: Optional[Callable[[str], None]] = None,
) -> Optional[HydeVariant]:
    """Generate one HyDE variant for ``original_question``.

    Args:
        provider: An LLM provider exposing ``agenerate(messages, model_alias,
            response_format, timeout, temperature)`` (the platform LLMProvider).
        model_alias: Router alias for the controller model.
        original_question: The pinned user question (never the prior HyDE text).
        tried_hyde: Hypothetical answers already tried this request — shown to
            the controller so it diversifies (INC-2 multi-round driver).
        coverage_aspects: The previous round's named gaps to target next.
        max_tokens: Cap for the hypothetical answer (it is embedded, not shown).
        temperature: Controller sampling temperature.
        timeout_s: Per-call timeout in seconds.

    Returns:
        A :class:`HydeVariant`, or ``None`` on any failure / empty output. The
        caller treats ``None`` as "fall back to a plain search on the processed
        query" — so a controller failure is never worse than baseline retrieval.
        On failure, ``on_failure`` (when given) is invoked once with the typed
        :class:`HydeFailure` reason for observability (the return contract is
        unchanged — callers that ignore the reason still just see ``None``).
    """
    def _fail(reason: str) -> None:
        """Report the typed failure reason (fail-open: return value stays None)."""
        if on_failure is not None:
            try:
                on_failure(reason)
            except Exception:  # noqa: BLE001 — telemetry must never break retrieval
                logger.debug("hyde on_failure callback raised", exc_info=True)

    tried = tried_hyde or []
    aspects = coverage_aspects or []
    prior_block = (
        "\n".join(f"- {t}" for t in tried) if tried else "(none yet)"
    )
    gap_block = (
        "\n".join(f"- {a}" for a in aspects if a) if any(aspects) else "(none named)"
    )
    prompt = _render(
        _load_prompt(),
        original_question=original_question,
        prior_hyde_answers=prior_block,
        uncovered_aspects=gap_block,
        max_tokens=str(max_tokens),
        domain=(domain or "(no specific domain configured)"),
    )
    kwargs: dict[str, Any] = dict(
        model_alias=model_alias,
        timeout=timeout_s,
        temperature=temperature,
        # Cap output to the HyDE budget so the controller can't ramble up to the
        # default gen max_tokens (it is embedded, never shown).
        max_tokens=max(1, max_tokens),
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = await provider.agenerate(
            [{"role": "user", "content": prompt}],
            **kwargs,
        )
    except asyncio.TimeoutError as exc:
        logger.warning("agentic HyDE generation timed out: %s", exc)
        _fail(HydeFailure.TIMEOUT)
        return None
    except Exception as exc:  # noqa: BLE001 — fail open to baseline retrieval
        logger.warning("agentic HyDE generation failed: %s", exc)
        _fail(HydeFailure.PROVIDER_ERROR)
        return None

    content = _strip_reasoning(getattr(resp, "content", "") or "")
    if not content.strip():
        # Empty final content — the classic reasoning-model-<think>-burn: the
        # model spent its whole budget reasoning and emitted no answer.
        logger.warning("agentic HyDE returned empty content (reasoning burn?); falling back")
        _fail(HydeFailure.EMPTY_CONTENT)
        return None
    obj = _parse_json_object(content)
    if not obj:
        logger.warning("agentic HyDE returned unparseable JSON; falling back")
        _fail(HydeFailure.PARSE_INVALID)
        return None
    variant = _coerce_variant(obj, max_tokens)
    if variant is None:
        logger.warning("agentic HyDE JSON carried no hypothetical_answer; falling back")
        _fail(HydeFailure.NO_HYPOTHESIS)
        return None
    return variant
