# @summary
# Suggested follow-up questions generator. After an answer is produced, an
# instruct-model call proposes a few specific, IN-CORPUS questions the user could
# ask next (rendered as clickable chips). Grounded in the answer + retrieved
# section headings so it never suggests a question the corpus can't answer.
# FAIL-OPEN everywhere: any error / disabled / empty input yields [] (no chips),
# never raises — a suggestion is advisory and must never break the response.
# Exports: generate_follow_ups (async, config-driven), generate_follow_ups_sync
# Deps: config.settings, src.platform.llm, src.retrieval.pipeline.deep_research (_parse_json_object)
# @end-summary
"""Generate suggested follow-up questions from a query + its answer.

Mirrors the agentic-HyDE generator shape (param-based core → config facade → a
loop-safe sync bridge for the synchronous ``RAGChain.run`` path). The streaming
route, being ``async``, awaits :func:`generate_follow_ups` directly.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, Sequence

# Reuse the same tolerant JSON-object parser HyDE/deep-research use (strips code
# fences / surrounding prose). One-way import: deep_research never imports this.
from src.retrieval.pipeline.deep_research import _parse_json_object

logger = logging.getLogger("rag.retrieval.follow_up")

_PROMPT_FILE = Path(__file__).resolve().parents[3] / "prompts" / "follow_up_questions.md"
_HEADING_CAP = 40  # max distinct headings fed to the prompt (context-budget guard)


def _load_prompt() -> str:
    return _PROMPT_FILE.read_text(encoding="utf-8")


def _render(template: str, **vars_: str) -> str:
    rendered = template
    for key, val in vars_.items():
        rendered = rendered.replace("{{ " + key + " }}", val)
    return rendered


def _headings_block(headings: Sequence[str]) -> str:
    """Dedupe + cap the retrieved section headings into a bulleted block."""
    seen: list[str] = []
    for h in headings:
        h = (h or "").strip()
        if h and h not in seen:
            seen.append(h)
        if len(seen) >= _HEADING_CAP:
            break
    return "\n".join(f"- {h}" for h in seen) if seen else "(no section headings available)"


def _coerce_questions(obj: Any, count: int, original_question: str) -> list[str]:
    """Extract a clean, deduped, capped list of question strings from parsed JSON."""
    if not isinstance(obj, dict):
        return []
    raw = obj.get("questions") or []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    orig = original_question.strip().lower()
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        q = str(item or "").strip()
        if not q:
            continue
        key = q.lower()
        if key == orig or key in seen:  # never echo the original or duplicates
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= max(1, count):
            break
    return out


async def generate_follow_ups(
    provider,
    *,
    question: str,
    answer: str,
    headings: Sequence[str],
    domain: str = "",
    count: int = 3,
    model_alias: str = "judge",
    max_tokens: int = 256,
    timeout_s: int = 20,
) -> list[str]:
    """Return up to ``count`` in-corpus follow-up questions (``[]`` on any failure).

    Never raises: an empty answer, a provider error, unparseable JSON, or a
    disabled feature all resolve to ``[]`` so the caller simply shows no chips.
    """
    if not (answer or "").strip():
        return []
    try:
        prompt = _render(
            _load_prompt(),
            count=str(max(1, count)),
            domain=(domain or "(no specific domain configured)"),
            question=question,
            answer=answer,
            headings=_headings_block(headings),
        )
        resp = await provider.agenerate(
            [{"role": "user", "content": prompt}],
            model_alias=model_alias,
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=max(1, max_tokens),
            timeout=timeout_s,
        )
        content = getattr(resp, "content", "") or ""
        return _coerce_questions(_parse_json_object(content), count, question)
    except Exception as exc:  # noqa: BLE001 — advisory hint, must never break the response
        logger.warning("follow-up question generation failed: %s", exc)
        return []


async def generate_follow_ups_from_config(
    provider, *, question: str, answer: str, headings: Sequence[str],
) -> list[str]:
    """Config-driven async facade — reads ``RAG_SUGGESTED_QUESTIONS_*``.

    Returns ``[]`` immediately when the feature is disabled (no LLM call).
    """
    from config import settings

    if not settings.RAG_SUGGESTED_QUESTIONS_ENABLED:
        return []
    return await generate_follow_ups(
        provider,
        question=question,
        answer=answer,
        headings=headings,
        domain=settings.DOMAIN_DESCRIPTION,
        count=settings.RAG_SUGGESTED_QUESTIONS_COUNT,
        model_alias=settings.RAG_SUGGESTED_QUESTIONS_MODEL_ALIAS,
        max_tokens=settings.RAG_SUGGESTED_QUESTIONS_MAX_TOKENS,
        timeout_s=settings.RAG_SUGGESTED_QUESTIONS_TIMEOUT_SECONDS,
    )


def generate_follow_ups_sync(
    question: str, answer: str, headings: Sequence[str], *, provider: Optional[Any] = None,
) -> list[str]:
    """Loop-safe synchronous bridge for the synchronous ``RAGChain.run`` path.

    Like the ingest role-classify bridge, the coroutine is driven on a dedicated
    worker thread with its own event loop (``RAGChain.run`` executes inside the
    Temporal worker's already-running loop, so ``asyncio.run`` is unusable). Never
    raises: any failure resolves to ``[]``.
    """
    from config import settings

    if not settings.RAG_SUGGESTED_QUESTIONS_ENABLED or not (answer or "").strip():
        return []

    if provider is None:
        from src.platform.llm import get_llm_provider
        provider = get_llm_provider()

    def _run() -> list[str]:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                generate_follow_ups_from_config(
                    provider, question=question, answer=answer, headings=headings,
                )
            )
        finally:
            try:
                asyncio.set_event_loop(None)
            finally:
                loop.close()

    # Bound the blocking wait: the provider timeout only covers post-connect I/O,
    # so a pre-HTTP stall (DNS/TCP/TLS) could otherwise hang this user-facing path
    # indefinitely. A hard result() deadline (a little beyond the provider timeout)
    # guarantees the fail-open [] path is reached. TimeoutError is caught below.
    deadline = settings.RAG_SUGGESTED_QUESTIONS_TIMEOUT_SECONDS + 2
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(_run).result(timeout=deadline)
    except Exception as exc:  # noqa: BLE001 — advisory, fail open (incl. TimeoutError)
        logger.warning("follow-up question generation (sync bridge) failed: %s", exc)
        return []
