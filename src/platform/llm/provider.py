# @summary
# Unified LLM provider wrapping litellm.Router for multi-provider completion.
# Exports: LLMProvider, get_llm_provider
# Deps: litellm, yaml, config.settings, src.platform.llm.schemas
# @end-summary
"""Unified LLM provider backed by LiteLLM Router.

The Router provides in-process model aliasing, fallback chains, retries,
and load balancing — without requiring a separate LiteLLM Proxy server.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import litellm
from litellm import Router

from config.settings import (
    LLM_API_BASE,
    LLM_API_KEY,
    LLM_FALLBACK_MODELS,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_NUM_RETRIES,
    LLM_QUERY_MODEL,
    LLM_ROUTER_CONFIG,
    LLM_TEMPERATURE,
    LLM_VISION_MODEL,
    RAG_LLM_ONESHOT_MAX_TOKENS,
    RAG_LLM_RATE_LIMIT_RETRY_DELAY_S,
    RAG_LLM_TOKEN_COUNTING_PROBE_TOKENS,
)
from src.platform.llm.schemas import LLMConfig, LLMResponse
from src.platform.observability import get_tracer

logger = logging.getLogger(__name__)

# Suppress litellm verbose logging unless DEBUG
litellm.suppress_debug_info = True


def _build_router_from_yaml(config_path: str) -> Router:
    """Build a LiteLLM Router from a YAML config file.

    Args:
        config_path: Path to a LiteLLM Router YAML config file.

    Returns:
        A configured `litellm.Router` instance.
    """
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    router_settings = cfg.get("router_settings", {})
    return Router(
        model_list=cfg["model_list"],
        **router_settings,
    )


def _build_router_from_env(config: LLMConfig) -> Router:
    """Build a LiteLLM Router programmatically from environment configuration.

    Args:
        config: Normalized LLM configuration.

    Returns:
        A configured `litellm.Router` instance.
    """
    model_list: list[dict[str, Any]] = [
        {
            "model_name": "default",
            "litellm_params": {
                "model": config.model,
                **({"api_base": config.api_base} if config.api_base else {}),
                **({"api_key": config.api_key} if config.api_key else {}),
            },
        },
    ]

    # Add fallback models under the same "default" alias for automatic failover
    for fallback_model in config.fallback_models:
        model_list.append({
            "model_name": "default",
            "litellm_params": {
                "model": fallback_model,
                **({"api_key": config.api_key} if config.api_key else {}),
            },
        })

    # Add vision model as a separate alias
    vision = config.vision_model
    if vision:
        model_list.append({
            "model_name": "vision",
            "litellm_params": {
                "model": vision,
                **({"api_base": config.api_base} if config.api_base else {}),
                **({"api_key": config.api_key} if config.api_key else {}),
            },
        })

    # Add query model as a separate alias (for query reformulation/evaluation)
    query = config.query_model
    if query and query != config.model:
        model_list.append({
            "model_name": "query",
            "litellm_params": {
                "model": query,
                **({"api_base": config.api_base} if config.api_base else {}),
                **({"api_key": config.api_key} if config.api_key else {}),
            },
        })
    else:
        # Alias "query" to "default" so callers can use either name
        model_list.append({
            "model_name": "query",
            "litellm_params": {
                "model": config.model,
                **({"api_base": config.api_base} if config.api_base else {}),
                **({"api_key": config.api_key} if config.api_key else {}),
            },
        })

    return Router(
        model_list=model_list,
        num_retries=config.num_retries,
        retry_after=RAG_LLM_RATE_LIMIT_RETRY_DELAY_S,
    )


class LLMProvider:
    """Unified interface for all LLM completions, backed by litellm.Router.

    The Router provides:
    - Named model aliases ("default", "smart", "vision", "fast")
    - Automatic fallback when multiple entries share the same alias
    - Load balancing across deployments
    - Built-in retries with backoff
    - Routing strategies (simple-shuffle, latency-based, least-busy)

    Config modes:
    - YAML: Set RAG_LLM_ROUTER_CONFIG=/path/to/llm_router.yaml
    - Env vars: Set RAG_LLM_MODEL, RAG_LLM_API_BASE, etc. (auto-builds Router)
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        router: Optional[Router] = None,
    ) -> None:
        """Create an LLM provider.

        Args:
            config: Optional configuration override. If not provided, reads from
                environment-backed settings.
            router: Optional pre-built `litellm.Router` instance.
        """
        self.config = config or LLMConfig(
            model=LLM_MODEL,
            api_base=LLM_API_BASE,
            api_key=LLM_API_KEY or None,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            num_retries=LLM_NUM_RETRIES,
            fallback_models=LLM_FALLBACK_MODELS,
            vision_model=LLM_VISION_MODEL or None,
            query_model=LLM_QUERY_MODEL or None,
        )

        if router:
            self._router = router
        elif LLM_ROUTER_CONFIG and Path(LLM_ROUTER_CONFIG).exists():
            self._router = _build_router_from_yaml(LLM_ROUTER_CONFIG)
            logger.info("LLM Router loaded from YAML: %s", LLM_ROUTER_CONFIG)
        else:
            self._router = _build_router_from_env(self.config)
            logger.info(
                "LLM Router built from env vars (model=%s)", self.config.model
            )

    def _base_kwargs(
        self,
        model_alias: str = "default",
        user_id: Optional[str] = None,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Build base keyword arguments for `Router.completion()` calls.

        Args:
            model_alias: Router model alias to use.
            user_id: Optional end-user identifier forwarded to LiteLLM as the
                ``user`` field so per-user cost attribution is recorded.
            **overrides: Additional keyword arguments to merge.

        Returns:
            Keyword arguments for a Router completion call.
        """
        kwargs: dict[str, Any] = {
            "model": model_alias,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        if user_id:
            kwargs["user"] = user_id
        kwargs.update(overrides)
        return kwargs

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        model_alias: str = "default",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> LLMResponse:
        """Run a synchronous completion via the Router.

        Args:
            messages: Chat messages in OpenAI-style format.
            model_alias: Router model alias to use.
            temperature: Optional sampling temperature override.
            max_tokens: Optional max completion tokens override.
            response_format: Optional response format payload (e.g. JSON mode).
            timeout: Optional per-call timeout in seconds.
            user_id: Optional end-user identifier for per-user cost attribution.

        Returns:
            Normalized `LLMResponse`.
        """
        kwargs = self._base_kwargs(model_alias=model_alias, user_id=user_id)
        kwargs["messages"] = messages
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format
        if timeout is not None:
            kwargs["timeout"] = timeout

        tracer = get_tracer()
        with tracer.generation(
            name="llm.generate",
            model=self.config.model,
            input=repr(messages),
            metadata={
                "gen_ai.system": "litellm",
                "model_alias": model_alias,
                "temperature": kwargs.get("temperature"),
            },
        ) as gen:
            response = self._router.completion(**kwargs)

            cost = 0.0
            try:
                cost = litellm.completion_cost(completion_response=response)
            except Exception:
                pass  # Local models have no pricing data

            usage = response.usage
            result = LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model or self.config.model,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
                cost_usd=cost,
            )
            gen.set_output(result.content)
            if result.prompt_tokens or result.completion_tokens:
                gen.set_token_counts(
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                )
            return result

    def generate_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        model_alias: str = "default",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> Any:
        """Run a synchronous streaming completion.

        Args:
            messages: Chat messages in OpenAI-style format.
            model_alias: Router model alias to use.
            temperature: Optional sampling temperature override.
            max_tokens: Optional max completion tokens override.
            timeout: Optional per-call timeout in seconds.
            user_id: Optional end-user identifier for per-user cost attribution.

        Yields:
            Content chunks (strings).
        """
        kwargs = self._base_kwargs(model_alias=model_alias, user_id=user_id, stream=True)
        kwargs["messages"] = messages
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if timeout is not None:
            kwargs["timeout"] = timeout

        response = self._router.completion(**kwargs)
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    async def agenerate(
        self,
        messages: list[dict[str, Any]],
        *,
        model_alias: str = "default",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> LLMResponse:
        """Run an async completion via the Router.

        Args:
            messages: Chat messages in OpenAI-style format.
            model_alias: Router model alias to use.
            temperature: Optional sampling temperature override.
            max_tokens: Optional max completion tokens override.
            response_format: Optional response format payload (e.g. JSON mode).
            timeout: Optional per-call timeout in seconds.
            user_id: Optional end-user identifier for per-user cost attribution.

        Returns:
            Normalized `LLMResponse`.
        """
        kwargs = self._base_kwargs(model_alias=model_alias, user_id=user_id)
        kwargs["messages"] = messages
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format
        if timeout is not None:
            kwargs["timeout"] = timeout

        tracer = get_tracer()
        with tracer.generation(
            name="llm.agenerate",
            model=self.config.model,
            input=repr(messages),
            metadata={
                "gen_ai.system": "litellm",
                "model_alias": model_alias,
                "temperature": kwargs.get("temperature"),
            },
        ) as gen:
            response = await self._router.acompletion(**kwargs)

            cost = 0.0
            try:
                cost = litellm.completion_cost(completion_response=response)
            except Exception:
                pass

            usage = response.usage
            result = LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model or self.config.model,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
                cost_usd=cost,
            )
            gen.set_output(result.content)
            if result.prompt_tokens or result.completion_tokens:
                gen.set_token_counts(
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                )
            return result

    async def agenerate_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        model_alias: str = "default",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Run an async streaming completion.

        Args:
            messages: Chat messages in OpenAI-style format.
            model_alias: Router model alias to use.
            temperature: Optional sampling temperature override.
            max_tokens: Optional max completion tokens override.
            timeout: Optional per-call timeout in seconds.
            user_id: Optional end-user identifier for per-user cost attribution.

        Yields:
            Content chunks (strings).
        """
        kwargs = self._base_kwargs(model_alias=model_alias, user_id=user_id, stream=True)
        kwargs["messages"] = messages
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if timeout is not None:
            kwargs["timeout"] = timeout

        response = await self._router.acompletion(**kwargs)
        async for chunk in response:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    def json_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        model_alias: str = "default",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> LLMResponse:
        """Run a completion expecting a JSON object response.

        Args:
            messages: Chat messages in OpenAI-style format.
            model_alias: Router model alias to use.
            temperature: Optional sampling temperature override.
            max_tokens: Optional max completion tokens override.
            timeout: Optional per-call timeout in seconds.
            user_id: Optional end-user identifier for per-user cost attribution.

        Returns:
            Normalized `LLMResponse`.
        """
        return self.generate(
            messages,
            model_alias=model_alias,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            timeout=timeout,
            user_id=user_id,
        )

    def vision_completion(
        self,
        prompt: str,
        image_b64: str,
        mime_type: str = "image/png",
        *,
        model_alias: str = "vision",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> LLMResponse:
        """Run a vision completion with a base64-encoded image.

        Args:
            prompt: User prompt text.
            image_b64: Base64-encoded image bytes (no data URL prefix).
            mime_type: MIME type for the image (used in data URL).
            model_alias: Router model alias to use (defaults to "vision").
            temperature: Optional sampling temperature override.
            max_tokens: Optional max completion tokens override.
            timeout: Optional per-call timeout in seconds.
            user_id: Optional end-user identifier for per-user cost attribution.

        Returns:
            Normalized `LLMResponse`.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}"
                        },
                    },
                ],
            }
        ]
        return self.generate(
            messages,
            model_alias=model_alias,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            timeout=timeout,
            user_id=user_id,
        )

    def is_available(self, model_alias: str = "default") -> bool:
        """Check whether the given model alias is reachable.

        Args:
            model_alias: Router model alias to probe.

        Returns:
            True if a small probe completion succeeds; otherwise False.
        """
        try:
            self.generate(
                [{"role": "user", "content": "ping"}],
                model_alias=model_alias,
                max_tokens=RAG_LLM_TOKEN_COUNTING_PROBE_TOKENS,
            )
            return True
        except Exception:
            return False


# ── Singleton ──────────────────────────────────────────────────────────
_provider: Optional[LLMProvider] = None


def get_llm_provider() -> LLMProvider:
    """Get or create the process-wide LLM provider singleton.

    Returns:
        The singleton `LLMProvider` instance.
    """
    global _provider
    if _provider is None:
        _provider = LLMProvider()
    return _provider


def call_oneshot(
    prompt: str,
    *,
    system: str = "",
    model_alias: str = "query",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = RAG_LLM_ONESHOT_MAX_TOKENS,
) -> Optional[str]:
    """One-shot text completion: build messages from (system, prompt), call the
    provider, return content or None on failure.

    Used by query reformulation and guardrails — callers that need a single
    short answer from the LLM and don't care about token usage or streaming.
    Exceptions are swallowed and logged so guardrail/loop callers degrade
    gracefully when the provider is unreachable.
    """
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        response = get_llm_provider().generate(
            messages,
            model_alias=model_alias,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content or None
    except Exception as e:
        logger.warning("call_oneshot failed (alias=%s): %s", model_alias, e)
        return None
