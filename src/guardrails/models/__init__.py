# @summary
# Public API for guardian classifier models. Exposes the GuardianModel
# contract, concrete implementations (Granite Guardian, self-check), and
# a config-driven factory used by guardrail backends.
# Exports: GuardianModel, GuardianRisk, GuardianVerdict, GuardianUnavailable,
#          GraniteGuardian, SelfCheckGuardian, build_guardian
# Deps: src.guardrails.models.{base,granite_guardian,self_check}, config.settings
# @end-summary
"""Guardian models package.

A single guardian instance is shared across rails — one model serves
toxicity, injection, topic safety, and faithfulness. Use
:func:`build_guardian` from a backend's constructor; rails should accept
``guardian: GuardianModel | None`` and call ``guardian.supports(risk)``
before classifying.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.guardrails.models.base import (
    Direction,
    GuardianModel,
    GuardianRisk,
    GuardianUnavailable,
    GuardianVerdict,
)
from src.guardrails.models.granite_guardian import (
    GRANITE_RISK_MAP,
    GraniteGuardian,
    GraniteMode,
)
from src.guardrails.models.self_check import SelfCheckGuardian

logger = logging.getLogger("rag.guardrails.models")


def build_guardian() -> Optional[GuardianModel]:
    """Construct the guardian configured via ``RAG_GUARDIAN_*`` settings.

    Returns ``None`` when guardians are disabled, in which case rails fall
    back to deterministic checks (regex). A guardian construction error
    (e.g. missing endpoint) is logged and downgrades to ``None`` rather
    than crashing the backend — guardrails should never become a hard
    dependency for serving traffic.
    """
    from config.settings import (
        RAG_GUARDIAN_ENABLED,
        RAG_GUARDIAN_PROVIDER,
        RAG_GUARDIAN_MODE,
        RAG_GUARDIAN_MODEL_ID,
        RAG_GUARDIAN_ENDPOINT,
        RAG_GUARDIAN_API_KEY,
        RAG_GUARDIAN_TIMEOUT_S,
        RAG_GUARDIAN_THRESHOLD,
    )

    if not RAG_GUARDIAN_ENABLED:
        logger.info("Guardian disabled (RAG_GUARDIAN_ENABLED=false)")
        return None

    provider = (RAG_GUARDIAN_PROVIDER or "").lower()
    try:
        if provider == "granite":
            logger.info(
                "Building Granite Guardian (mode=%s, model=%s)",
                RAG_GUARDIAN_MODE,
                RAG_GUARDIAN_MODEL_ID,
            )
            return GraniteGuardian(
                mode=RAG_GUARDIAN_MODE,  # type: ignore[arg-type]
                model_id=RAG_GUARDIAN_MODEL_ID,
                endpoint=RAG_GUARDIAN_ENDPOINT or None,
                api_key=RAG_GUARDIAN_API_KEY or None,
                timeout_s=RAG_GUARDIAN_TIMEOUT_S,
                threshold=RAG_GUARDIAN_THRESHOLD,
            )
        if provider == "self_check":
            logger.info("Building self-check guardian")
            return SelfCheckGuardian(threshold=RAG_GUARDIAN_THRESHOLD)
        logger.warning(
            "Unknown RAG_GUARDIAN_PROVIDER=%r — guardian disabled", provider
        )
        return None
    except Exception as e:
        logger.warning(
            "Failed to construct guardian (provider=%s): %s — disabled",
            provider, e,
        )
        return None


__all__ = [
    "Direction",
    "GuardianModel",
    "GuardianRisk",
    "GuardianVerdict",
    "GuardianUnavailable",
    "GraniteGuardian",
    "GraniteMode",
    "GRANITE_RISK_MAP",
    "SelfCheckGuardian",
    "build_guardian",
]
