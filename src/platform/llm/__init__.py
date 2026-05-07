# @summary
# Unified LLM provider package backed by LiteLLM Router.
# Exports: LLMProvider, get_llm_provider, call_oneshot, LLMConfig, LLMResponse
# Deps: src.platform.llm.provider, src.platform.llm.schemas
# @end-summary
"""Unified LLM provider — single entry point for all LLM calls in the project."""

from src.platform.llm.provider import LLMProvider, get_llm_provider, call_oneshot
from src.platform.llm.schemas import LLMConfig, LLMResponse


def _register_kgweave_llm_factory() -> None:
    """Wire RagWeave's LLM provider into KGWeave's default-factory hook.

    KGWeave is transport-agnostic about its LLM client — it asks the host
    application to register a factory at startup. This is the host side
    of that contract; KGWeave never imports from ``src.*``.
    """
    try:
        from kgweave import set_default_llm_factory  # noqa: PLC0415
    except Exception:
        return
    set_default_llm_factory(get_llm_provider)


_register_kgweave_llm_factory()

__all__ = ["LLMProvider", "get_llm_provider", "call_oneshot", "LLMConfig", "LLMResponse"]
