# @summary
# RailMergeGate: merge query processing result with input rail result for routing.
# Generic schema logic — no backend dependency.
# Exports: RailMergeGate
# Deps: src.guardrails.common.schemas, logging
# @end-summary
"""Rail merge gate for routing decisions after input rail execution.

Applies priority-ordered merge logic to produce a single routing action
from the combined query result and input rail result. Generic — no
backend dependency.

Requirements references (from internal docs): REQ-707.
"""

from __future__ import annotations

import logging
from typing import Any

from src.guardrails.common.schemas import InputRailResult, RailVerdict

logger = logging.getLogger("rag.guardrails.merge_gate")


class RailMergeGate:
    """Merge query processing result with input rail result (REQ-707).

    Priority order:
    1. Injection reject overrides all
    2. Toxicity reject overrides intent routing
    3. Topic safety (off-topic) rejects with canned response
    4. Intent classification determines flow
    5. PII redaction modifies query but does not change flow
    """

    def merge(
        self,
        query_result: Any,
        rail_result: InputRailResult,
    ) -> dict[str, Any]:
        """Return merged routing decision and payload.

        Returns a dict with:
        - action: "reject" | "canned" | "search"
        - message: rejection/canned response text (if not search)
        - query: effective query to use for search (may be PII-redacted)

        Args:
            query_result: Query processing result object (must include
                `processed_query`).
            rail_result: Aggregated input rail result.

        Returns:
            A dict describing routing action and optional message/query payload.
        """
        from src.guardrails.shared import INTENT_RESPONSES

        # Priority 1: injection reject
        if rail_result.injection_verdict == RailVerdict.REJECT:
            from src.guardrails.shared.injection import REJECTION_MESSAGE as INJECTION_MSG

            logger.info("Merge gate: REJECT (injection)")
            return {
                "action": "reject",
                "message": INJECTION_MSG,
            }

        # Priority 2: toxicity reject
        if rail_result.toxicity_verdict == RailVerdict.REJECT:
            from src.guardrails.shared.toxicity import REJECTION_MESSAGE as TOXICITY_MSG

            logger.info("Merge gate: REJECT (toxicity)")
            return {
                "action": "reject",
                "message": TOXICITY_MSG,
            }

        # Priority 3: topic safety (LLM-based off-topic detection)
        if rail_result.topic_off_topic:
            from src.guardrails.shared.topic_safety import REJECTION_MESSAGE as TOPIC_MSG

            logger.info("Merge gate: CANNED (topic_safety: off-topic)")
            return {
                "action": "canned",
                "message": TOPIC_MSG,
            }

        # Priority 4: intent routing
        intent = rail_result.intent
        if intent != "rag_search" and intent in INTENT_RESPONSES:
            logger.info("Merge gate: CANNED (%s)", intent)
            return {
                "action": "canned",
                "message": INTENT_RESPONSES[intent],
            }

        # Priority 5: PII redaction (non-blocking)
        effective_query = (
            rail_result.redacted_query or query_result.processed_query
        )

        logger.info("Merge gate: SEARCH (intent=%s)", intent)
        return {
            "action": "search",
            "query": effective_query,
        }
