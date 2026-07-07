# @summary
# Canonical intent classification rail using NeMo runtime with keyword fallback.
# Runtime injected at construction — no direct GuardrailsRuntime.get() call.
# Exports: IntentClassifier, IntentResult
# Deps: src.guardrails.common.schemas, config.settings, logging, re
# @end-summary
"""Canonical intent classification rail.

This module classifies user queries into a small set of canonical intents used
to route requests (e.g., distinguish "rag_search" from greetings/off-topic).
It prefers NeMo Guardrails when available and falls back to deterministic
keyword matching when the runtime is disabled or unavailable.

Requirements references (from internal docs): REQ-101 through REQ-105.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from config.settings import (
    RAG_GUARDRAILS_INTENT_CONFIDENCE_THRESHOLD,
    RAG_GUARDRAILS_INTENT_ASYNC_TIMEOUT_S,
)

logger = logging.getLogger("rag.guardrails.intent")

# Keyword-based fallback patterns for when LLM is unavailable
_GREETING_PATTERNS = re.compile(
    r"^\s*(?:hi|hello|hey|greetings|good\s+(?:morning|afternoon|evening))\b",
    re.IGNORECASE,
)
_FAREWELL_PATTERNS = re.compile(
    r"^\s*(?:bye|goodbye|see\s+you|farewell|thanks?,?\s*bye)\b",
    re.IGNORECASE,
)
_OFF_TOPIC_PATTERNS = re.compile(
    r"\b(?:weather|joke|game|score|music|stock\s+price|play\s+(?:a|some))\b",
    re.IGNORECASE,
)
_ADMIN_PATTERNS = re.compile(
    r"^\s*(?:help|what\s+can\s+you\s+do|how\s+(?:do\s+I|does\s+this))\b",
    re.IGNORECASE,
)

# Canned responses for non-search intents (REQ-103)
INTENT_RESPONSES = {
    "greeting": "Hello! I'm here to help you search the knowledge base. What would you like to know?",
    "farewell": "Goodbye! Feel free to return if you have more questions.",
    "off_topic": "I'm designed to help you find information in the knowledge base. I can't help with that topic, but feel free to ask me a question about the documents I have access to.",
    "administrative": "I can search the knowledge base to answer your questions. Just type your question in natural language and I'll find relevant information from the available documents.",
}


@dataclass
class IntentResult:
    """Result of intent classification.

    Attributes:
        intent: Intent label (e.g., "rag_search", "greeting").
        confidence: Confidence score in the chosen intent.
        canned_response: Optional canned response text for non-search intents.
    """

    intent: str
    confidence: float
    canned_response: Optional[str] = None


class IntentClassifier:
    """Classify user queries into canonical intents.

    Uses NeMo runtime when available, falls back to keyword matching.

    The NeMo runtime path is gated on the injected runtime: when
    ``runtime=None``, only keyword matching is used.
    """

    def __init__(self, confidence_threshold: float = RAG_GUARDRAILS_INTENT_CONFIDENCE_THRESHOLD, runtime=None) -> None:
        """Initialize an intent classifier.

        Args:
            confidence_threshold: Minimum confidence for model-based intent
                selection. This is primarily reserved for future expansion;
                the current implementation returns fixed confidences.
            runtime: Optional ``GuardrailsRuntime`` instance. When provided,
                enables NeMo-based classification. When ``None``, only keyword
                matching is used.
        """
        self._confidence_threshold = confidence_threshold
        self._runtime = runtime

    def classify(self, query: str) -> IntentResult:
        """Classify a query into a canonical intent.

        Args:
            query: User input query text.

        Returns:
            `IntentResult` with intent label, confidence, and optional canned
            response for non-search intents.
        """
        # Try NeMo-based classification if runtime is available
        runtime = self._runtime
        if runtime is not None and runtime.initialized and runtime.rails is not None:
            try:
                return self._classify_with_nemo(query, runtime)
            except Exception as e:
                logger.warning(
                    "NeMo intent classification failed: %s — using fallback", e
                )

        # Deterministic keyword fallback
        return self._classify_with_keywords(query)

    def _classify_with_nemo(self, query: str, runtime) -> IntentResult:
        """Classify intent using the NeMo runtime.

        Args:
            query: User input query text.
            runtime: Initialized ``GuardrailsRuntime`` instance.

        Returns:
            `IntentResult` derived from NeMo's response behavior.
        """
        import asyncio

        messages = [{"role": "user", "content": query}]

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, runtime.generate_async(messages))
                    response = future.result(timeout=RAG_GUARDRAILS_INTENT_ASYNC_TIMEOUT_S)
            else:
                response = asyncio.run(runtime.generate_async(messages))
        except RuntimeError:
            response = asyncio.run(runtime.generate_async(messages))

        content = response.get("content", "")

        # Check if the response matches a canned intent response
        for intent_name, canned in INTENT_RESPONSES.items():
            if content.strip() == canned.strip():
                return IntentResult(
                    intent=intent_name,
                    confidence=0.9,
                    canned_response=canned,
                )

        # If NeMo passed through to LLM, it's a rag_search intent
        return IntentResult(intent="rag_search", confidence=0.85)

    def _classify_with_keywords(self, query: str) -> IntentResult:
        """Classify intent using deterministic keyword patterns.

        Args:
            query: User input query text.

        Returns:
            `IntentResult` from keyword matching, defaulting to "rag_search".
        """
        if _GREETING_PATTERNS.search(query):
            return IntentResult(
                intent="greeting",
                confidence=0.8,
                canned_response=INTENT_RESPONSES["greeting"],
            )
        if _FAREWELL_PATTERNS.search(query):
            return IntentResult(
                intent="farewell",
                confidence=0.8,
                canned_response=INTENT_RESPONSES["farewell"],
            )
        if _ADMIN_PATTERNS.search(query):
            return IntentResult(
                intent="administrative",
                confidence=0.7,
                canned_response=INTENT_RESPONSES["administrative"],
            )
        if _OFF_TOPIC_PATTERNS.search(query):
            return IntentResult(
                intent="off_topic",
                confidence=0.6,
                canned_response=INTENT_RESPONSES["off_topic"],
            )

        return IntentResult(intent="rag_search", confidence=0.85)
