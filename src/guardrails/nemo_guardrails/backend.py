# @summary
# NemoBackend: GuardrailBackend implementation wiring NeMo runtime to shared rails.
# Reads all RAG_NEMO_* config at construction and passes runtime to each rail.
# Exports: NemoBackend
# Deps: src.guardrails.backend, src.guardrails.nemo_guardrails.runtime,
#       src.guardrails.nemo_guardrails.executor, src.guardrails.shared.*,
#       config.settings, logging
# @end-summary
"""NeMo Guardrails backend implementation.

Wires the NeMo runtime to the shared ML rails via constructor injection.
All RAG_NEMO_* config is read here; ``rag_chain.py`` only sees the
``GuardrailBackend`` interface.
"""

from __future__ import annotations

import logging

from src.guardrails.backend import GuardrailBackend
from src.guardrails.common import (
    InputRailResult,
    OutputRailResult,
)

logger = logging.getLogger("rag.guardrails.nemo_backend")


class NemoBackend(GuardrailBackend):
    """GuardrailBackend backed by NeMo Guardrails.

    Reads all ``RAG_NEMO_*`` config settings at construction, initialises
    the ``GuardrailsRuntime`` singleton, and injects it into each rail
    that needs it. ``rag_chain.py`` never imports NeMo-specific modules.
    """

    def __init__(self) -> None:
        from config.settings import (
            RAG_NEMO_CONFIG_DIR,
            RAG_NEMO_FAITHFULNESS_ACTION,
            RAG_NEMO_FAITHFULNESS_ENABLED,
            RAG_NEMO_FAITHFULNESS_SELF_CHECK,
            RAG_NEMO_FAITHFULNESS_THRESHOLD,
            RAG_NEMO_INJECTION_ENABLED,
            RAG_NEMO_INJECTION_LP_THRESHOLD,
            RAG_NEMO_INJECTION_MODEL_ENABLED,
            RAG_NEMO_INJECTION_PERPLEXITY_ENABLED,
            RAG_NEMO_INJECTION_PS_PPL_THRESHOLD,
            RAG_NEMO_INJECTION_SENSITIVITY,
            RAG_NEMO_INTENT_CONFIDENCE_THRESHOLD,
            RAG_NEMO_OUTPUT_PII_ENABLED,
            RAG_NEMO_OUTPUT_TOXICITY_ENABLED,
            RAG_NEMO_PII_ENABLED,
            RAG_NEMO_PII_EXTENDED,
            RAG_NEMO_PII_GLINER_ENABLED,
            RAG_NEMO_PII_SCORE_THRESHOLD,
            RAG_NEMO_RAIL_TIMEOUT_SECONDS,
            RAG_NEMO_TOPIC_SAFETY_ENABLED,
            RAG_NEMO_TOPIC_SAFETY_INSTRUCTIONS,
            RAG_NEMO_TOXICITY_ENABLED,
            RAG_NEMO_TOXICITY_THRESHOLD,
        )
        from src.guardrails.nemo_guardrails.runtime import GuardrailsRuntime
        from src.guardrails.nemo_guardrails.executor import InputRailExecutor, OutputRailExecutor
        from src.guardrails.shared import FaithfulnessChecker
        from src.guardrails.shared import InjectionDetector
        from src.guardrails.shared import IntentClassifier
        from src.guardrails.shared import PIIDetector
        from src.guardrails.shared import TopicSafetyChecker
        from src.guardrails.shared import ToxicityFilter

        logger.info("Initializing NeMo Guardrails backend...")
        runtime = GuardrailsRuntime.get()
        runtime.initialize(RAG_NEMO_CONFIG_DIR)

        intent_classifier = IntentClassifier(
            confidence_threshold=RAG_NEMO_INTENT_CONFIDENCE_THRESHOLD,
            runtime=runtime,
        )

        injection_detector = (
            InjectionDetector(
                sensitivity=RAG_NEMO_INJECTION_SENSITIVITY,
                enable_perplexity=RAG_NEMO_INJECTION_PERPLEXITY_ENABLED,
                enable_model_classifier=RAG_NEMO_INJECTION_MODEL_ENABLED,
                lp_threshold=RAG_NEMO_INJECTION_LP_THRESHOLD,
                ps_ppl_threshold=RAG_NEMO_INJECTION_PS_PPL_THRESHOLD,
                runtime=runtime,
            )
            if RAG_NEMO_INJECTION_ENABLED
            else None
        )

        pii_detector = (
            PIIDetector(
                extended=RAG_NEMO_PII_EXTENDED,
                score_threshold=RAG_NEMO_PII_SCORE_THRESHOLD,
                use_gliner=RAG_NEMO_PII_GLINER_ENABLED,
            )
            if RAG_NEMO_PII_ENABLED
            else None
        )
        toxicity_filter = (
            ToxicityFilter(
                threshold=RAG_NEMO_TOXICITY_THRESHOLD,
                runtime=runtime,
            )
            if RAG_NEMO_TOXICITY_ENABLED
            else None
        )
        topic_safety_checker = (
            TopicSafetyChecker(
                custom_instructions=RAG_NEMO_TOPIC_SAFETY_INSTRUCTIONS,
                runtime=runtime,
            )
            if RAG_NEMO_TOPIC_SAFETY_ENABLED
            else None
        )

        self._input_executor = InputRailExecutor(
            intent_classifier=intent_classifier,
            injection_detector=injection_detector,
            pii_detector=pii_detector,
            toxicity_filter=toxicity_filter,
            topic_safety_checker=topic_safety_checker,
            timeout_seconds=RAG_NEMO_RAIL_TIMEOUT_SECONDS,
        )

        faithfulness_checker = (
            FaithfulnessChecker(
                threshold=RAG_NEMO_FAITHFULNESS_THRESHOLD,
                action=RAG_NEMO_FAITHFULNESS_ACTION,
                use_self_check=RAG_NEMO_FAITHFULNESS_SELF_CHECK,
            )
            if RAG_NEMO_FAITHFULNESS_ENABLED
            else None
        )
        # Reuse the same pii/toxicity instances to avoid loading models twice
        output_pii = pii_detector if RAG_NEMO_OUTPUT_PII_ENABLED else None
        output_toxicity = toxicity_filter if RAG_NEMO_OUTPUT_TOXICITY_ENABLED else None

        self._output_executor = OutputRailExecutor(
            faithfulness_checker=faithfulness_checker,
            pii_detector=output_pii,
            toxicity_filter=output_toxicity,
            timeout_seconds=RAG_NEMO_RAIL_TIMEOUT_SECONDS,
        )

        logger.info("NeMo Guardrails backend initialized.")

    def run_input_rails(self, query: str, tenant_id: str = "") -> InputRailResult:
        """Run all input rails via NeMo executor.

        Args:
            query: Raw user query text.
            tenant_id: Optional tenant identifier.

        Returns:
            Aggregated ``InputRailResult``.
        """
        return self._input_executor.execute(query, tenant_id)

    def run_output_rails(self, answer: str, context_chunks: list[str]) -> OutputRailResult:
        """Run all output rails via NeMo executor.

        Args:
            answer: Generated assistant answer text.
            context_chunks: Source context snippets used to generate the answer.

        Returns:
            Aggregated ``OutputRailResult``.
        """
        return self._output_executor.execute(answer, context_chunks)

    def redact_pii(self, text: str) -> tuple[str, list]:
        """Redact PII using the input executor's PII detector.

        Returns ``(text, [])`` when PII detection is disabled (no PII detector
        configured).

        Args:
            text: Raw input text.

        Returns:
            ``(redacted_text, detections)`` tuple.
        """
        pii = self._input_executor.pii_detector
        if pii is not None:
            return pii.redact(text)
        return text, []

    def register_rag_chain(self, rag_chain: object) -> None:
        """Register the RAG chain reference with Colang action handlers.

        Args:
            rag_chain: The ``RAGChain`` instance.
        """
        try:
            from config.guardrails.actions import set_rag_chain
            set_rag_chain(rag_chain)
            logger.info("RAG chain reference set for Colang rag_retrieve_and_generate action")
        except ImportError:
            logger.warning(
                "Could not register RAG chain reference — config.guardrails.actions not found"
            )
