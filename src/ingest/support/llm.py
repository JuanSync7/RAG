# @summary
# LLM helper for JSON-only chat calls used by ingestion nodes, backed by LiteLLM.
# Exports: _llm_json
# Deps: json, src.platform.llm, src.ingest.common.utils, src.ingest.common.types
# @end-summary

"""LLM helper utilities for ingestion pipeline."""

from __future__ import annotations

import logging

from src.ingest.common import IngestionConfig
from src.ingest.common import parse_json_object
from src.platform.llm import get_llm_provider

logger = logging.getLogger("rag.ingest.support.llm")


def _llm_json(
    prompt: str, config: IngestionConfig, max_tokens: int = 300
) -> dict[str, object]:
    """Execute a JSON-only LLM chat call via LiteLLM and parse the response.

    Args:
        prompt: User prompt to send to the LLM.
        config: Ingestion configuration controlling whether LLM metadata is enabled.
        max_tokens: Maximum tokens to request for the completion.

    Returns:
        Parsed JSON object as a dictionary, or an empty dictionary when LLM
        metadata is disabled or the call fails.
    """
    if not config.enable_llm_metadata:
        return {}
    messages = [
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": prompt},
    ]
    try:
        provider = get_llm_provider()
        response = provider.json_completion(
            messages,
            temperature=config.llm_temperature,
            max_tokens=max_tokens,
            timeout=config.llm_timeout_seconds,
        )
        return parse_json_object(response.content)
    except Exception:
        logger.warning("LLM JSON call failed, returning empty dict", exc_info=True)
        return {}
