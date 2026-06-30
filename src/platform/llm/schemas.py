# @summary
# Typed contracts for the unified LLM provider layer.
# Exports: LLMConfig, LLMResponse
# Deps: dataclasses
# @end-summary
"""Schemas for the unified LLM provider backed by LiteLLM.

These dataclasses define the minimal configuration and response envelope used
by the platform LLM provider facade.
"""
from __future__ import annotations


import os
from dataclasses import dataclass, field
from typing import Optional

from config.settings import (
    RAG_LLM_DEFAULT_MAX_TOKENS,
    RAG_LLM_DEFAULT_NUM_RETRIES,
    RAG_LLM_DEFAULT_TEMPERATURE,
)

_OLLAMA_PORT = os.environ.get("RAG_OLLAMA_PORT", "11434")
_DEFAULT_API_BASE = os.environ.get(
    "RAG_LLM_API_BASE",
    os.environ.get("RAG_OLLAMA_URL", f"http://localhost:{_OLLAMA_PORT}"),
)


@dataclass(frozen=True)
class LLMConfig:
    """Resolved LLM configuration built from env vars or YAML Router config."""

    model: str = "ollama/qwen2.5:3b"
    api_base: Optional[str] = _DEFAULT_API_BASE
    api_key: Optional[str] = None
    max_tokens: int = RAG_LLM_DEFAULT_MAX_TOKENS
    temperature: float = RAG_LLM_DEFAULT_TEMPERATURE
    num_retries: int = RAG_LLM_DEFAULT_NUM_RETRIES
    fallback_models: list[str] = field(default_factory=list)
    vision_model: Optional[str] = None
    query_model: Optional[str] = None
    controller_model: Optional[str] = None
    judge_model: Optional[str] = None


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
