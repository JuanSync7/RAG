# @summary
# Shared, model-driven chunk-ROLE classifier used by BOTH ingest tagging and the
# backfill script (one prompt, one parser — CLI/UI parity). Labels each chunk
# content|navigation|boilerplate via the LLM router; NEVER drops a chunk and
# NEVER uses regex/keyword matching to decide role (CLAUDE.md §0/§2). Batched,
# prefix-capped, json-salvage parsed, and FAIL-OPEN to default_role ("content")
# on any error/parse-failure/missing-index/unknown-role so real content is never
# silently demoted. Output is 1:1 with the input, same length and order.
# Exports: classify_roles, classify_roles_from_config, classify_roles_sync
# Deps: asyncio, config.settings, src.retrieval.pipeline.deep_research (_parse_json_object)
# @end-summary
"""Generic LLM chunk-role tagger (content / navigation / boilerplate).

The classifier is the *only* authority for a chunk's role. It judges the chunk's
FUNCTION (does it bear answer content, is it a table-of-contents/index/pointer,
or is it title/legal front-matter) — never a vendor, document family, heading,
or phrasing match. The legacy regex ``is_navigational`` heuristic is retained
elsewhere only as an off-by-default emergency fallback and is not used here.

FAIL-OPEN is the central invariant: any classifier error, parse failure, missing
JSON index, or out-of-vocabulary role resolves to ``default_role`` ("content").
Defaulting a chunk to navigation/boilerplate on error would silently drop real
content from retrieval, which is exactly the failure this subsystem exists to
prevent.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, Sequence

from src.retrieval.pipeline.deep_research import _parse_json_object

logger = logging.getLogger(__name__)

# Repo root: this file is at src/ingest/common/role_classify.py -> parents[3].
_PROMPT_FILE = Path(__file__).resolve().parents[3] / "prompts" / "chunk_role_classify.md"

# Default role vocabulary. Mirrors the prompt's three roles. Kept as a module
# constant so callers that do not pass an explicit set still agree with the
# prompt; the param-based fn always takes the authoritative set from its caller.
_DEFAULT_VALID_ROLES = ("content", "navigation", "boilerplate")


def _load_prompt() -> str:
    """Load the shared classifier prompt template (never inline it)."""
    try:
        return _PROMPT_FILE.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - packaging error
        raise RuntimeError(
            f"Chunk-role classify prompt not found at {_PROMPT_FILE}: {exc}"
        ) from exc


def _render(template: str, **vars_: str) -> str:
    """Substitute ``{{ name }}`` placeholders (same convention as the judge)."""
    rendered = template
    for k, v in vars_.items():
        rendered = rendered.replace("{{ " + k + " }}", v)
    return rendered


def _chunk_text(chunk: Any) -> str:
    """Extract text from a chunk that is either an object with ``.text`` or a
    dict with a ``"text"`` key. Anything else -> empty string (fail-open: an
    unreadable chunk still gets a default role, never an exception)."""
    if isinstance(chunk, dict):
        return str(chunk.get("text") or "")
    return str(getattr(chunk, "text", "") or "")


def _build_chunk_block(batch: Sequence[Any], prefix_chars: int) -> str:
    """Render the numbered, prefix-capped chunk list for one classification call.

    Only the first ``prefix_chars`` of each chunk's text is sent — the role of a
    chunk (ToC vs data table vs body) is determined by its head, and capping
    bounds token cost on large chunks.
    """
    lines: list[str] = []
    for i, chunk in enumerate(batch):
        text = _chunk_text(chunk)[: max(1, prefix_chars)]
        # Collapse newlines so each chunk stays on its own labelled line and the
        # model sees a clean per-index list.
        text = " ".join(text.split())
        lines.append(f"[{i}] {text}")
    return "\n".join(lines)


def _parse_batch_roles(
    content: str,
    n: int,
    *,
    valid_roles: tuple[str, ...],
    default_role: str,
) -> list[str]:
    """Map a model JSON response onto ``n`` roles, fail-open to ``default_role``.

    Returns exactly ``n`` roles. Any of: empty/garbage output, missing ``roles``
    list, an omitted index, a non-int index, or an out-of-vocabulary role string
    resolves that position to ``default_role``. Order is by the JSON entry's
    ``"i"`` (the batch-local index), NOT by entry position.
    """
    fallback = [default_role] * n
    obj = _parse_json_object(content or "")
    if not obj or not isinstance(obj.get("roles"), list):
        return fallback

    by_index: dict[int, str] = {}
    for entry in obj["roles"]:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("i", entry.get("index")))
        except (TypeError, ValueError):
            continue
        role = entry.get("role")
        if not isinstance(role, str):
            continue
        role = role.strip().lower()
        if role not in valid_roles:
            # Out-of-vocabulary / ambiguous -> fail open (do not record it; the
            # position keeps default_role).
            continue
        if 0 <= idx < n:
            by_index[idx] = role

    return [by_index.get(i, default_role) for i in range(n)]


async def classify_roles(
    provider,
    chunks: Sequence[Any],
    *,
    model_alias: str,
    batch_size: int,
    prefix_chars: int,
    timeout_s: int,
    max_output_tokens: int,
    json_mode: bool = True,
    valid_roles: tuple[str, ...] = _DEFAULT_VALID_ROLES,
    default_role: str = "content",
) -> list[str]:
    """Classify each chunk's role via the LLM router; return one role per chunk.

    Args:
        provider: An LLM provider exposing ``async agenerate(messages, **kwargs)``
            whose result has a ``.content`` string (same contract as the agentic
            judge). Calls are awaited sequentially, one per batch.
        chunks: Chunks to classify. Each may be an object with a ``.text``
            attribute or a dict with a ``"text"`` key.
        model_alias: Router alias to classify with (e.g. ``"judge"``).
        batch_size: Max chunks per classification call (>= 1).
        prefix_chars: Only the first ``prefix_chars`` of each chunk's text are
            sent to the model (>= 1).
        timeout_s: Per-call timeout forwarded to the provider.
        max_output_tokens: Max output tokens per call.
        json_mode: When True request guided ``json_object`` output; when the
            alias points at a JSON-unreliable reasoning model, pass False and
            rely on the salvage parser.
        valid_roles: Accepted role vocabulary; anything else is fail-open.
        default_role: Role assigned on any failure/ambiguity (must be in
            ``valid_roles``); the project default is ``"content"``.

    Returns:
        A list of role strings, SAME length and order as ``chunks``. Every value
        is in ``valid_roles``. On any per-batch failure those positions are
        ``default_role`` — a failing batch never poisons a healthy one, and a
        provider/parse error never raises out of this function.
    """
    chunks = list(chunks)
    if not chunks:
        return []

    bs = max(1, int(batch_size))
    template = _load_prompt()
    roles: list[str] = []

    for start in range(0, len(chunks), bs):
        batch = chunks[start : start + bs]
        n = len(batch)
        prompt = _render(
            template, chunks=_build_chunk_block(batch, prefix_chars)
        )
        kwargs: dict[str, Any] = dict(
            model_alias=model_alias,
            timeout=timeout_s,
            max_tokens=max(1, int(max_output_tokens)),
            temperature=0.0,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = await provider.agenerate(
                [{"role": "user", "content": prompt}],
                **kwargs,
            )
            content = getattr(resp, "content", "") or ""
        except Exception as exc:  # noqa: BLE001 — fail open, never drop a batch
            logger.warning(
                "chunk-role classify call failed (batch @%d, n=%d): %s — "
                "defaulting to %r",
                start, n, exc, default_role,
            )
            roles.extend([default_role] * n)
            continue

        roles.extend(
            _parse_batch_roles(
                content, n, valid_roles=valid_roles, default_role=default_role
            )
        )

    # Defensive length guarantee: the loop already yields exactly len(chunks),
    # but never return a mis-sized list to the caller (fail-open pad/truncate).
    if len(roles) != len(chunks):  # pragma: no cover - invariant guard
        logger.error(
            "chunk-role classify length mismatch (%d != %d) — padding with %r",
            len(roles), len(chunks), default_role,
        )
        roles = (roles + [default_role] * len(chunks))[: len(chunks)]
    return roles


async def classify_roles_from_config(provider, chunks: Sequence[Any]) -> list[str]:
    """Config-driven facade over :func:`classify_roles`.

    Reads the ``RAG_NAV_CLASSIFY_*`` / ``RAG_NAV_ROLE_DEFAULT`` settings and
    calls the param-based function so both the ingest tagger and the backfill
    script share one configuration surface. Importing settings lazily keeps this
    module import-light for tests that drive the param-based fn directly.
    """
    from config import settings

    return await classify_roles(
        provider,
        chunks,
        model_alias=settings.RAG_NAV_CLASSIFY_MODEL_ALIAS,
        batch_size=settings.RAG_NAV_CLASSIFY_BATCH_SIZE,
        prefix_chars=settings.RAG_NAV_CLASSIFY_PREFIX_CHARS,
        timeout_s=settings.RAG_NAV_CLASSIFY_TIMEOUT_SECONDS,
        max_output_tokens=settings.RAG_NAV_CLASSIFY_MAX_OUTPUT_TOKENS,
        json_mode=settings.RAG_NAV_CLASSIFY_JSON_MODE,
        valid_roles=_DEFAULT_VALID_ROLES,
        default_role=settings.RAG_NAV_ROLE_DEFAULT,
    )


def classify_roles_sync(provider, chunks: Sequence[Any]) -> list[str]:
    """Synchronous, loop-safe bridge to :func:`classify_roles_from_config`.

    The ingest chunking node is a *synchronous* function, yet it executes inside
    the Temporal worker's running asyncio loop (the embedding activity is
    ``async`` and calls ``run_embedding_pipeline`` → ``_GRAPH.invoke`` inline).
    ``asyncio.run`` therefore cannot be used (it refuses to run inside an already
    running loop). To be safe whether or not the calling thread has a running
    loop, the coroutine is always driven on a *dedicated worker thread* with its
    own fresh event loop.

    Like the async functions, this NEVER raises: any failure resolves to the
    config ``default_role`` for every chunk (fail-open). One role string is
    returned per input chunk, in order.
    """
    chunks = list(chunks)
    if not chunks:
        return []

    from config import settings

    default_role = settings.RAG_NAV_ROLE_DEFAULT

    def _run() -> list[str]:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                classify_roles_from_config(provider, chunks)
            )
        finally:
            try:
                asyncio.set_event_loop(None)
            finally:
                loop.close()

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(_run).result()
    except Exception as exc:  # noqa: BLE001 — fail open, never drop a chunk
        logger.warning(
            "chunk-role classify (sync bridge) failed: %s — defaulting all %d "
            "chunks to %r",
            exc, len(chunks), default_role,
        )
        return [default_role] * len(chunks)
