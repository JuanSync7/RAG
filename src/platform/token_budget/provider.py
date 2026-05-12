# @summary
# Model capability discovery via litellm.get_model_info() (with Ollama /api/show
# fallback) and token budget calculation using litellm.token_counter().
# Exports: get_capabilities, refresh_capabilities, calculate_budget
# Deps: litellm, json, urllib, logging, config.settings, .schemas, .utils
# @end-summary
"""Model capability discovery and token budget calculation.

Uses litellm.get_model_info() as primary source for context window size.
Falls back to Ollama /api/show for locally-served models that litellm
doesn't have metadata for.
"""

from __future__ import annotations

import orjson
import logging
from typing import Any, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from config.settings import (
    GENERATION_MAX_TOKENS,
    LLM_API_BASE,
    LLM_MODEL,
    TOKEN_BUDGET_CHARS_PER_TOKEN,
    TOKEN_BUDGET_DEFAULT_CONTEXT_LENGTH,
    RAG_TOKEN_BUDGET_URLOPEN_TIMEOUT_S,
)
from src.platform.token_budget.schemas import (
    ModelCapabilities,
    TokenBreakdown,
    TokenBudgetSnapshot,
)
from src.platform.token_budget.utils import count_tokens

logger = logging.getLogger("rag.token_budget")

_cached_capabilities: ModelCapabilities | None = None


# ── Model capability discovery ────────────────────────────────────────


def _fetch_via_litellm(model: str) -> Optional[dict[str, Any]]:
    """Fetch model metadata via LiteLLM if available.

    Args:
        model: LiteLLM model string.

    Returns:
        Model info dict if available; otherwise None.
    """
    try:
        import litellm
        info = litellm.get_model_info(model=model)
        if info and info.get("max_input_tokens"):
            return info
    except Exception:
        pass
    return None


def _fetch_via_ollama(
    model_name: str, base_url: str
) -> Optional[dict[str, Any]]:
    """Fetch model info from Ollama `/api/show`.

    Args:
        model_name: LiteLLM model string (may include `ollama/` prefix).
        base_url: Ollama base URL.

    Returns:
        Parsed JSON payload if successful; otherwise None.
    """
    # Strip the "ollama/" prefix if present for the Ollama API
    ollama_model = model_name.removeprefix("ollama/")
    try:
        payload = orjson.dumps({"name": ollama_model})
        req = Request(
            f"{base_url.rstrip('/')}/api/show",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=RAG_TOKEN_BUDGET_URLOPEN_TIMEOUT_S) as resp:
            return orjson.loads(resp.read())
    except (URLError, OSError, orjson.JSONDecodeError, ValueError):
        return None


def fetch_model_capabilities(
    model_name: str | None = None,
    base_url: str | None = None,
    default_context_length: int | None = None,
) -> ModelCapabilities:
    """Discover model context window and metadata.

    Resolution order:
    1. ``litellm.get_model_info()`` — works for known cloud & open-weight models.
    2. Ollama ``/api/show`` — works for locally-pulled Ollama models.
    3. Default fallback (``TOKEN_BUDGET_DEFAULT_CONTEXT_LENGTH``).
    Args:
        model_name: Optional model override.
        base_url: Optional base URL override (used for Ollama fallback).
        default_context_length: Optional default fallback context length.

    Returns:
        Discovered `ModelCapabilities`. `stale=True` indicates a fallback path.
    """
    model = model_name or LLM_MODEL
    url = base_url or LLM_API_BASE
    default_ctx = default_context_length or TOKEN_BUDGET_DEFAULT_CONTEXT_LENGTH

    # ── 1. Try litellm.get_model_info() ─────────────────────────────
    litellm_info = _fetch_via_litellm(model)
    if litellm_info:
        ctx = litellm_info.get("max_input_tokens") or litellm_info.get("max_tokens") or default_ctx
        caps = ModelCapabilities(
            model_name=model,
            context_length=int(ctx),
            family=litellm_info.get("litellm_provider", ""),
            stale=False,
        )
        logger.info(
            "Token budget: %s context_length=%d (litellm)", model, caps.context_length
        )
        return caps

    # ── 2. Try Ollama /api/show ─────────────────────────────────────
    ollama_data = _fetch_via_ollama(model, url)
    if ollama_data:
        details = ollama_data.get("details", {})
        family = details.get("family", "")
        model_info = ollama_data.get("model_info", {})

        context_length = None
        if family:
            context_length = model_info.get(f"{family}.context_length")
        if context_length is None:
            for key, val in model_info.items():
                if key.endswith(".context_length") and isinstance(val, (int, float)):
                    context_length = int(val)
                    break

        if context_length and context_length > 0:
            caps = ModelCapabilities(
                model_name=model,
                context_length=int(context_length),
                family=family,
                parameter_size=details.get("parameter_size", ""),
                quantization_level=details.get("quantization_level", ""),
                stale=False,
            )
            logger.info(
                "Token budget: %s context_length=%d (ollama /api/show)",
                model,
                caps.context_length,
            )
            return caps

    # ── 3. Default fallback ─────────────────────────────────────────
    logger.warning(
        "Token budget: could not determine context_length for %s; using default %d",
        model,
        default_ctx,
    )
    return ModelCapabilities(
        model_name=model,
        context_length=default_ctx,
        stale=True,
    )


def get_capabilities() -> ModelCapabilities:
    """Return cached model capabilities, fetching on first call.

    Returns:
        Cached `ModelCapabilities`.
    """
    global _cached_capabilities
    if _cached_capabilities is None:
        _cached_capabilities = fetch_model_capabilities()
    return _cached_capabilities


def refresh_capabilities() -> ModelCapabilities:
    """Re-fetch model capabilities and update the cache.

    Returns:
        Fresh `ModelCapabilities`.
    """
    global _cached_capabilities
    _cached_capabilities = fetch_model_capabilities()
    return _cached_capabilities


# ── Budget calculation ────────────────────────────────────────────────


def calculate_budget(
    capabilities: ModelCapabilities | None = None,
    *,
    system_prompt: str | None = None,
    memory_context: str | None = None,
    chunks: list[str] | None = None,
    query: str | None = None,
    template_overhead_chars: int = 200,
    output_reservation: int | None = None,
    model: str | None = None,
) -> TokenBudgetSnapshot:
    """Compute token budget snapshot from prompt components.

    Uses ``litellm.token_counter()`` for accurate per-model tokenization
    (falls back to character heuristic for unknown models).

    Args:
        capabilities: Model capabilities (uses cached if None).
        system_prompt: System prompt text.
        memory_context: Memory context text (summary + recent turns).
        chunks: Retrieved context chunk texts.
        query: User query text.
        template_overhead_chars: Estimated chars for prompt template formatting.
        output_reservation: Override GENERATION_MAX_TOKENS.
        model: LiteLLM model string for accurate tokenization.

    Returns:
        TokenBudgetSnapshot with estimated usage.
    """
    caps = capabilities or get_capabilities()
    out_res = output_reservation if output_reservation is not None else GENERATION_MAX_TOKENS
    mdl = model or caps.model_name

    sp_tokens = count_tokens(text=system_prompt, model=mdl)
    mem_tokens = count_tokens(text=memory_context, model=mdl)
    chunk_tokens = sum(count_tokens(text=c, model=mdl) for c in (chunks or []))
    query_tokens = count_tokens(text=query, model=mdl)
    overhead_tokens = max(
        0, template_overhead_chars // max(1, TOKEN_BUDGET_CHARS_PER_TOKEN)
    )

    input_tokens = sp_tokens + mem_tokens + chunk_tokens + query_tokens + overhead_tokens

    effective_budget = max(1, caps.context_length - out_res)
    usage_percent = min(100.0, max(0.0, (input_tokens / effective_budget) * 100))

    breakdown = TokenBreakdown(
        system_prompt=sp_tokens,
        memory_context=mem_tokens,
        retrieval_chunks=chunk_tokens,
        user_query=query_tokens,
        template_overhead=overhead_tokens,
    )

    return TokenBudgetSnapshot(
        input_tokens=input_tokens,
        context_length=caps.context_length,
        output_reservation=out_res,
        usage_percent=round(usage_percent, 1),
        model_name=caps.model_name,
        breakdown=breakdown,
    )


__all__ = [
    "calculate_budget",
    "fetch_model_capabilities",
    "get_capabilities",
    "refresh_capabilities",
]
