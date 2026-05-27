# @summary
# LangChain ChatModel adapter wrapping the existing LiteLLM-backed LLMProvider.
# Exports: get_llm, ChatLLMAdapter
# Deps: langchain_core, src.platform.llm.provider, src.common.llm.schemas
# @end-summary
"""LangChain-compatible ChatModel backed by the platform LLMProvider.

``get_llm()`` is the primary public API.  It returns a LangChain
``BaseChatModel`` that delegates to the project's existing
``src.platform.llm.provider.LLMProvider`` (LiteLLM Router).  This means
all existing config (model aliases, fallbacks, retries, API keys) is
reused — callers get a ChatModel that plugs into LangChain composition
primitives (``|``, ``RunnableParallel``, ``with_structured_output``)
without adding new provider dependencies.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from src.common.llm.schemas import ModelTier
from src.platform.observability import get_tracer

logger = logging.getLogger(__name__)


def _messages_to_dicts(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Convert LangChain messages to OpenAI-style dicts."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})
        else:
            result.append({"role": "user", "content": msg.content})
    return result


class ChatLLMAdapter(BaseChatModel):
    """LangChain ChatModel that delegates to the platform LLMProvider.

    This adapter bridges the existing LiteLLM Router infrastructure with
    LangChain's composition primitives.  It supports:

    - ``invoke()`` / ``ainvoke()`` for single completions
    - ``stream()`` for token-level streaming
    - ``with_structured_output()`` for typed Pydantic responses
    - ``|`` pipe operator for chain composition
    - ``RunnableParallel`` for fan-out execution
    """

    model_alias: str = "default"
    """LiteLLM Router alias (default, vision, query, etc.)."""

    temperature: Optional[float] = None
    """Override temperature (None = use provider default)."""

    max_tokens: Optional[int] = None
    """Override max tokens (None = use provider default)."""

    timeout: Optional[int] = None
    """Per-call timeout in seconds."""

    user_id: Optional[str] = None
    """End-user identifier for cost attribution."""

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "chat-llm-adapter"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Run a synchronous completion via the platform LLMProvider."""
        from src.platform.llm import get_llm_provider

        provider = get_llm_provider()
        msg_dicts = _messages_to_dicts(messages)

        overrides: dict[str, Any] = {}
        if self.temperature is not None:
            overrides["temperature"] = self.temperature
        if self.max_tokens is not None:
            overrides["max_tokens"] = self.max_tokens
        if self.timeout is not None:
            overrides["timeout"] = self.timeout

        # Merge kwargs (allows per-call overrides like response_format)
        response_format = kwargs.pop("response_format", None)
        if response_format:
            overrides["response_format"] = response_format

        tracer = get_tracer()
        with tracer.generation(
            name="llm.adapter.sync",
            model=self.model_alias,
            input=repr(msg_dicts),
            metadata={"gen_ai.system": "langchain"},
        ) as gen:
            llm_response = provider.generate(
                msg_dicts,
                model_alias=self.model_alias,
                user_id=self.user_id,
                **overrides,
            )
            gen.set_output(llm_response.content)
            if llm_response.prompt_tokens or llm_response.completion_tokens:
                gen.set_token_counts(
                    prompt_tokens=llm_response.prompt_tokens,
                    completion_tokens=llm_response.completion_tokens,
                )

        message = AIMessage(
            content=llm_response.content,
            additional_kwargs={
                "model": llm_response.model,
                "prompt_tokens": llm_response.prompt_tokens,
                "completion_tokens": llm_response.completion_tokens,
                "total_tokens": llm_response.total_tokens,
                "cost_usd": llm_response.cost_usd,
            },
        )

        generation = ChatGeneration(message=message)
        return ChatResult(
            generations=[generation],
            llm_output={
                "model": llm_response.model,
                "token_usage": {
                    "prompt_tokens": llm_response.prompt_tokens,
                    "completion_tokens": llm_response.completion_tokens,
                    "total_tokens": llm_response.total_tokens,
                },
                "cost_usd": llm_response.cost_usd,
            },
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream tokens from the platform LLMProvider."""
        from src.platform.llm import get_llm_provider

        provider = get_llm_provider()
        msg_dicts = _messages_to_dicts(messages)

        overrides: dict[str, Any] = {}
        if self.temperature is not None:
            overrides["temperature"] = self.temperature
        if self.max_tokens is not None:
            overrides["max_tokens"] = self.max_tokens
        if self.timeout is not None:
            overrides["timeout"] = self.timeout

        tracer = get_tracer()
        with tracer.generation(
            name="llm.adapter.stream",
            model=self.model_alias,
            input=repr(msg_dicts),
            metadata={"gen_ai.system": "langchain"},
        ) as gen:
            collected: list[str] = []
            for token in provider.generate_stream(
                msg_dicts,
                model_alias=self.model_alias,
                user_id=self.user_id,
                **overrides,
            ):
                collected.append(token)
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content=token)
                )
                if run_manager:
                    run_manager.on_llm_new_token(token, chunk=chunk)
                yield chunk
            gen.set_output("".join(collected))


# ── Model tier → alias mapping ───────────────────────────────────────────

_TIER_ALIASES: dict[ModelTier, str] = {
    ModelTier.HIGH: "default",
    ModelTier.MEDIUM: "default",
    ModelTier.LOW: "default",
    ModelTier.LOCAL: "default",
}


def get_llm(
    model_alias: str = "default",
    *,
    tier: ModelTier | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
    task: str | None = None,
    user_id: str | None = None,
) -> ChatLLMAdapter:
    """Create a LangChain ChatModel backed by the platform LLMProvider.

    This is the primary entry point for the LLM composition layer.  The
    returned model works with all LangChain primitives: ``|``, batch,
    ``with_structured_output()``, ``RunnableParallel``, etc.

    Args:
        model_alias: LiteLLM Router alias — ``"default"``, ``"vision"``,
            ``"query"``, or any alias defined in the router config.
        tier: Optional ``ModelTier`` — resolves to an alias if
            *model_alias* is not provided explicitly.
        temperature: Sampling temperature override.
        max_tokens: Max completion tokens override.
        timeout: Per-call timeout in seconds.
        task: Human-readable task label (for observability / logging).
        user_id: End-user identifier for per-user cost attribution.

    Returns:
        A ``ChatLLMAdapter`` instance usable with LangChain composition.
    """
    alias = model_alias
    if tier is not None and model_alias == "default":
        alias = _TIER_ALIASES.get(tier, "default")

    if task:
        logger.debug("get_llm(alias=%s, task=%s)", alias, task)

    return ChatLLMAdapter(
        model_alias=alias,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        user_id=user_id,
    )
