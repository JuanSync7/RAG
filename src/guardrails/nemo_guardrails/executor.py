# @summary
# Input and output rail executors with parallel execution and consensus gate.
# Runs intent, injection, PII, toxicity, and topic safety rails in parallel
# for input; faithfulness, PII, toxicity in parallel for output.
# Exports: InputRailExecutor, OutputRailExecutor
# Deps: src.guardrails.shared.*, src.guardrails.common.schemas, concurrent.futures, logging, time
# @end-summary
"""Rail execution orchestration for the NeMo Guardrails backend.

This module runs "rails" (guardrail checks) in parallel with per-rail timeouts,
records structured execution results, and applies merge/consensus logic to
produce a single combined decision for routing and response shaping.

Requirements references (from internal docs): REQ-702, REQ-703.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from typing import Any, Optional

from src.common import make_query_hash
from src.guardrails.common import (
    GuardrailsMetadata,
    InputRailResult,
    OutputRailResult,
    RailExecution,
    RailVerdict,
)
from src.guardrails.shared import FaithfulnessChecker
from src.guardrails.shared import InjectionDetector
from src.guardrails.shared import IntentClassifier
from src.guardrails.shared import PIIDetector
from src.guardrails.shared import TopicSafetyChecker
from src.guardrails.shared import ToxicityFilter
from src.platform.observability import get_tracer
from src.platform import PIPELINE_STAGE_MS
from src.platform import measure_ms

logger = logging.getLogger("rag.guardrails.executor")

# Prometheus metrics for guardrails (REQ-905)
# prometheus_client is a main dependency and already loaded via src.platform above.
from prometheus_client import Counter, Histogram

GUARDRAIL_EXECUTIONS = Counter(
    "rag_guardrail_executions_total",
    "Total guardrail executions",
    ["rail_name", "verdict"],
)
GUARDRAIL_EXECUTION_MS = Histogram(
    "rag_guardrail_execution_ms",
    "Guardrail execution time in milliseconds",
    ["rail_name"],
    buckets=[10, 50, 100, 250, 500, 1000, 2000, 5000, 10000],
)
GUARDRAIL_REJECTIONS = Counter(
    "rag_guardrail_rejections_total",
    "Total guardrail rejections",
    ["rail_name", "reason"],
)


def _record_metric(rail_name: str, verdict: RailVerdict, ms: float) -> None:
    """Record Prometheus metrics for a rail execution.

    Args:
        rail_name: Logical rail name label.
        verdict: Verdict returned by the rail.
        ms: Execution time in milliseconds.
    """
    if GUARDRAIL_EXECUTIONS is not None:
        GUARDRAIL_EXECUTIONS.labels(rail_name=rail_name, verdict=verdict.value).inc()
    if GUARDRAIL_EXECUTION_MS is not None:
        GUARDRAIL_EXECUTION_MS.labels(rail_name=rail_name).observe(ms)
    if verdict == RailVerdict.REJECT and GUARDRAIL_REJECTIONS is not None:
        GUARDRAIL_REJECTIONS.labels(rail_name=rail_name, reason=verdict.value).inc()
    # Also record on the unified pipeline stage histogram
    if PIPELINE_STAGE_MS is not None:
        PIPELINE_STAGE_MS.labels(
            stage=f"guardrail_{rail_name}", bucket="guardrails"
        ).observe(ms)


class InputRailExecutor:
    """Run all enabled input rails in parallel (REQ-702).

    Each rail has a per-rail timeout. By default, timed-out or failed rails
    return a PASS verdict (fail-open). Rails listed in ``fail_closed_rails``
    return REJECT on timeout or error instead (fail-closed).
    """

    def __init__(
        self,
        intent_classifier: Optional[IntentClassifier] = None,
        injection_detector: Optional[InjectionDetector] = None,
        pii_detector: Optional[PIIDetector] = None,
        toxicity_filter: Optional[ToxicityFilter] = None,
        topic_safety_checker: Optional[TopicSafetyChecker] = None,
        timeout_seconds: int = 10,
        fail_closed_rails: frozenset = frozenset(),
    ) -> None:
        """Initialize the input-rail executor.

        Args:
            intent_classifier: Optional intent classifier rail.
            injection_detector: Optional prompt-injection detection rail.
            pii_detector: Optional PII redaction rail.
            toxicity_filter: Optional toxicity detection rail.
            topic_safety_checker: Optional on-topic/off-topic detection rail.
            timeout_seconds: Per-rail timeout in seconds.
            fail_closed_rails: Rail names that return REJECT (not PASS) on
                timeout or error. Use ``frozenset({"injection", "toxicity"})``
                to harden security-critical rails.
        """
        self._intent = intent_classifier
        self._injection = injection_detector
        self._pii = pii_detector
        self._toxicity = toxicity_filter
        self._topic_safety = topic_safety_checker
        self._timeout = timeout_seconds
        self._fail_closed_rails = fail_closed_rails
        self._tracer = get_tracer()

    @property
    def pii_detector(self):
        """Public accessor for the PII detector used in the pre-LLM redaction step."""
        return self._pii

    def execute(
        self,
        query: str,
        tenant_id: str = "",
        parent_span: Any = None,
    ) -> InputRailResult:
        """Run enabled input rails and return a combined result.

        This method is fail-open: timeouts and unexpected exceptions are logged
        and treated as PASS for the affected rail.

        Args:
            query: User input query text.
            tenant_id: Optional tenant identifier used by some rails/policies.
            parent_span: Optional parent tracing span for correlation.

        Returns:
            Aggregated `InputRailResult` including per-rail executions.
        """
        result = InputRailResult()
        executions: list[RailExecution] = []
        query_hash = make_query_hash(query)

        with ThreadPoolExecutor(
            max_workers=5, thread_name_prefix="input_rail"
        ) as pool:
            futures: dict[str, Future] = {}

            if self._intent:
                futures["intent"] = pool.submit(self._intent.classify, query)
            if self._injection:
                futures["injection"] = pool.submit(
                    self._injection.check, query, tenant_id
                )
            if self._pii:
                futures["pii"] = pool.submit(self._pii.redact, query)
            if self._toxicity:
                futures["toxicity"] = pool.submit(self._toxicity.check, query)
            if self._topic_safety:
                futures["topic_safety"] = pool.submit(self._topic_safety.check, query)

            for name, fut in futures.items():
                t0 = time.perf_counter()
                span = self._tracer.start_span(
                    f"guardrails.input.{name}",
                    {"query_hash": query_hash},
                    parent=parent_span,
                )
                try:
                    rail_result = fut.result(timeout=self._timeout)
                    ms = measure_ms(t0)

                    if name == "intent":
                        result.intent = rail_result.intent
                        result.intent_confidence = rail_result.confidence
                        verdict = RailVerdict.PASS
                        executions.append(
                            RailExecution(
                                "intent",
                                verdict,
                                ms,
                                {
                                    "intent": rail_result.intent,
                                    "confidence": rail_result.confidence,
                                },
                            )
                        )
                    elif name == "injection":
                        result.injection_verdict = rail_result.verdict
                        executions.append(
                            RailExecution(
                                "injection",
                                rail_result.verdict,
                                ms,
                                {"source": rail_result.detection_source or "none"},
                            )
                        )
                    elif name == "pii":
                        redacted_text, detections = rail_result
                        if detections:
                            result.redacted_query = redacted_text
                            result.pii_redactions = [
                                {"type": d.pii_type} for d in detections
                            ]
                        verdict = (
                            RailVerdict.MODIFY if detections else RailVerdict.PASS
                        )
                        executions.append(RailExecution("pii", verdict, ms))
                    elif name == "toxicity":
                        result.toxicity_verdict = rail_result.verdict
                        executions.append(
                            RailExecution(
                                "toxicity",
                                rail_result.verdict,
                                ms,
                                {"score": rail_result.score},
                            )
                        )
                    elif name == "topic_safety":
                        verdict = rail_result.verdict
                        if not rail_result.on_topic:
                            result.topic_off_topic = True
                        executions.append(
                            RailExecution(
                                "topic_safety",
                                verdict,
                                ms,
                                {"on_topic": rail_result.on_topic},
                            )
                        )

                    _record_metric(name, executions[-1].verdict, ms)
                    span.set_attribute("verdict", executions[-1].verdict.value)
                    span.end(status="ok")

                    logger.info(
                        "Rail %s | verdict=%s | ms=%.0f | hash=%s | tenant=%s",
                        name,
                        executions[-1].verdict.value,
                        ms,
                        query_hash,
                        tenant_id,
                    )

                except TimeoutError:
                    ms = measure_ms(t0)
                    if name in self._fail_closed_rails:
                        logger.warning(
                            "Rail '%s' timed out after %.0fms — fail-closed: REJECT | hash=%s",
                            name, ms, query_hash,
                        )
                        default_verdict = RailVerdict.REJECT
                        if name == "injection":
                            result.injection_verdict = RailVerdict.REJECT
                        elif name == "toxicity":
                            result.toxicity_verdict = RailVerdict.REJECT
                    else:
                        logger.warning(
                            "Rail '%s' timed out after %.0fms — defaulting to pass | hash=%s",
                            name, ms, query_hash,
                        )
                        default_verdict = RailVerdict.PASS
                    executions.append(RailExecution(name, default_verdict, ms))
                    _record_metric(name, default_verdict, ms)
                    span.end(status="error")

                except Exception as e:
                    ms = measure_ms(t0)
                    if name in self._fail_closed_rails:
                        logger.warning(
                            "Rail '%s' failed: %s — fail-closed: REJECT | hash=%s",
                            name, e, query_hash,
                        )
                        default_verdict = RailVerdict.REJECT
                        if name == "injection":
                            result.injection_verdict = RailVerdict.REJECT
                        elif name == "toxicity":
                            result.toxicity_verdict = RailVerdict.REJECT
                    else:
                        logger.warning(
                            "Rail '%s' failed: %s — defaulting to pass | hash=%s",
                            name, e, query_hash,
                        )
                        default_verdict = RailVerdict.PASS
                    executions.append(RailExecution(name, default_verdict, ms))
                    _record_metric(name, default_verdict, ms)
                    span.end(status="error", error=e)

        result.rail_executions = executions
        return result


class OutputRailExecutor:
    """Run output rails in parallel with consensus gate (REQ-703).

    All enabled output rails (faithfulness, PII, toxicity) execute
    concurrently. A consensus gate then evaluates results:
    - Faithfulness reject → discard other results, return fallback
    - Otherwise → apply PII redaction, then toxicity filter
    """

    def __init__(
        self,
        faithfulness_checker: Optional[FaithfulnessChecker] = None,
        pii_detector: Optional[PIIDetector] = None,
        toxicity_filter: Optional[ToxicityFilter] = None,
        timeout_seconds: int = 10,
        fail_closed_rails: frozenset = frozenset(),
    ) -> None:
        """Initialize the output-rail executor.

        Args:
            faithfulness_checker: Optional faithfulness evaluation rail.
            pii_detector: Optional PII redaction rail.
            toxicity_filter: Optional toxicity filter rail for generated outputs.
            timeout_seconds: Per-rail timeout in seconds.
            fail_closed_rails: Rail names that return REJECT on timeout or error
                instead of PASS. Use ``frozenset({"faithfulness"})`` to reject
                answers when the faithfulness rail cannot complete.
        """
        self._faithfulness = faithfulness_checker
        self._pii = pii_detector
        self._toxicity = toxicity_filter
        self._timeout = timeout_seconds
        self._fail_closed_rails = fail_closed_rails
        self._tracer = get_tracer()

    def execute(
        self,
        answer: str,
        context_chunks: list[str],
        parent_span: Any = None,
    ) -> OutputRailResult:
        """Run enabled output rails and apply consensus/merge logic.

        Args:
            answer: Proposed assistant answer text.
            context_chunks: Source context snippets used to generate the answer.
            parent_span: Optional parent tracing span for correlation.

        Returns:
            Aggregated `OutputRailResult` including per-rail executions and any
            modified `final_answer`.
        """
        result = OutputRailResult(final_answer=answer)
        executions: list[RailExecution] = []

        with ThreadPoolExecutor(
            max_workers=3, thread_name_prefix="output_rail"
        ) as pool:
            futures: dict[str, Future] = {}

            if self._faithfulness:
                futures["faithfulness"] = pool.submit(
                    self._faithfulness.check, answer, context_chunks
                )
            if self._pii:
                futures["pii"] = pool.submit(self._pii.redact, answer)
            if self._toxicity:
                futures["toxicity"] = pool.submit(
                    self._toxicity.filter_output, answer
                )

            # Collect results from all rails
            rail_results: dict[str, Any] = {}
            for name, fut in futures.items():
                t0 = time.perf_counter()
                span = self._tracer.start_span(
                    f"guardrails.output.{name}", parent=parent_span
                )
                try:
                    rail_results[name] = fut.result(timeout=self._timeout)
                    ms = measure_ms(t0)

                    if name == "faithfulness":
                        faith_result = rail_results[name]
                        result.faithfulness_score = faith_result.overall_score
                        result.faithfulness_verdict = faith_result.verdict
                        result.faithfulness_warning = faith_result.warning
                        result.claim_scores = [
                            {
                                "claim": c.claim,
                                "score": c.score,
                                "supported": c.supported,
                            }
                            for c in faith_result.claim_scores
                        ]
                        executions.append(
                            RailExecution(
                                "faithfulness",
                                faith_result.verdict,
                                ms,
                                {"score": faith_result.overall_score},
                            )
                        )
                        _record_metric("faithfulness", faith_result.verdict, ms)
                        span.set_attribute("score", faith_result.overall_score)

                    elif name == "pii":
                        redacted_text, detections = rail_results[name]
                        verdict = RailVerdict.MODIFY if detections else RailVerdict.PASS
                        executions.append(RailExecution("output_pii", verdict, ms))
                        _record_metric("output_pii", verdict, ms)

                    elif name == "toxicity":
                        filtered_text = rail_results[name]
                        verdict = (
                            RailVerdict.MODIFY
                            if filtered_text != answer
                            else RailVerdict.PASS
                        )
                        executions.append(RailExecution("output_toxicity", verdict, ms))
                        _record_metric("output_toxicity", verdict, ms)

                    span.set_attribute("verdict", executions[-1].verdict.value)
                    span.end(status="ok")

                    logger.info(
                        "Rail output.%s | verdict=%s | ms=%.0f",
                        name,
                        executions[-1].verdict.value,
                        ms,
                    )

                except TimeoutError:
                    ms = measure_ms(t0)
                    if name in self._fail_closed_rails:
                        logger.warning(
                            "Output rail '%s' timed out after %.0fms — fail-closed: REJECT",
                            name, ms,
                        )
                        default_verdict = RailVerdict.REJECT
                        if name == "faithfulness":
                            from src.guardrails.shared import (
                                FaithfulnessResult,
                                _FALLBACK_MESSAGE,
                            )
                            rail_results["faithfulness"] = FaithfulnessResult(
                                overall_score=0.0,
                                verdict=RailVerdict.REJECT,
                                fallback_message=_FALLBACK_MESSAGE,
                            )
                    else:
                        logger.warning(
                            "Output rail '%s' timed out after %.0fms — defaulting to pass",
                            name, ms,
                        )
                        default_verdict = RailVerdict.PASS
                    executions.append(RailExecution(f"output_{name}", default_verdict, ms))
                    _record_metric(f"output_{name}", default_verdict, ms)
                    span.end(status="error")

                except Exception as e:
                    ms = measure_ms(t0)
                    if name in self._fail_closed_rails:
                        logger.warning(
                            "Output rail '%s' failed: %s — fail-closed: REJECT",
                            name, e,
                        )
                        default_verdict = RailVerdict.REJECT
                        if name == "faithfulness":
                            from src.guardrails.shared import (
                                FaithfulnessResult,
                                _FALLBACK_MESSAGE,
                            )
                            rail_results["faithfulness"] = FaithfulnessResult(
                                overall_score=0.0,
                                verdict=RailVerdict.REJECT,
                                fallback_message=_FALLBACK_MESSAGE,
                            )
                    else:
                        logger.warning(
                            "Output rail '%s' failed: %s — defaulting to pass",
                            name, e,
                        )
                        default_verdict = RailVerdict.PASS
                    executions.append(RailExecution(f"output_{name}", default_verdict, ms))
                    _record_metric(f"output_{name}", default_verdict, ms)
                    span.end(status="error", error=e)

        # ── Consensus Gate ──
        # Priority 1: faithfulness reject → discard everything, return fallback
        faith = rail_results.get("faithfulness")
        if faith is not None and faith.verdict == RailVerdict.REJECT:
            logger.info(
                "Output consensus: REJECT (faithfulness score=%.2f)",
                faith.overall_score,
            )
            result.final_answer = faith.fallback_message
            result.rail_executions = executions
            return result

        # Priority 2: apply PII redaction to the answer
        pii = rail_results.get("pii")
        if pii is not None:
            redacted_text, detections = pii
            if detections:
                result.final_answer = redacted_text
                result.pii_redactions = [
                    {"type": d.pii_type} for d in detections
                ]

        # Priority 3: apply toxicity filter to the (possibly PII-redacted) answer
        toxicity_text = rail_results.get("toxicity")
        if toxicity_text is not None and toxicity_text != answer:
            # Re-run toxicity on the potentially PII-redacted text
            # since the parallel run used the original answer
            if result.final_answer != answer:
                filtered = self._toxicity.filter_output(result.final_answer or "")
            else:
                filtered = toxicity_text
            if filtered != result.final_answer:
                result.final_answer = filtered
                result.toxicity_verdict = RailVerdict.MODIFY

        result.rail_executions = executions
        return result
