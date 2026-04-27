# @summary
# Unified LLM provider package backed by LiteLLM Router.
# Exports: LLMProvider, get_llm_provider, call_oneshot, LLMConfig, LLMResponse
# Deps: src.platform.llm.provider, src.platform.llm.schemas
# @end-summary
"""Unified LLM provider — single entry point for all LLM calls in the project."""

from src.platform.llm.provider import LLMProvider, get_llm_provider, call_oneshot
from src.platform.llm.schemas import LLMConfig, LLMResponse

__all__ = ["LLMProvider", "get_llm_provider", "call_oneshot", "LLMConfig", "LLMResponse"]
