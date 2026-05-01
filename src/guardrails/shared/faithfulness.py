# @summary
# Faithfulness and hallucination detection rail using NeMo self-check-facts
# approach with claim scoring, entity hallucination detection, and LLM fallback.
# Exports: FaithfulnessChecker, FaithfulnessResult, ClaimScore
# Deps: config.settings, logging, re, json
# @end-summary
"""Faithfulness and hallucination detection rail.

This module evaluates whether a generated answer is supported by the retrieved
context. It uses NeMo-style self-check prompts and optional claim-level scoring,
plus a lightweight deterministic check for obvious hallucinations (e.g., years
or large numbers not present in evidence).

Requirements references (from internal docs): REQ-501 through REQ-505.
"""

from __future__ import annotations

import orjson
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from src.guardrails.common import RailVerdict

logger = logging.getLogger("rag.guardrails.faithfulness")


def _format_numbered_chunks(chunks: list[str]) -> str:
    """Format context chunks as numbered evidence text.

    Args:
        chunks: Context chunks to format.

    Returns:
        Evidence string with stable numeric prefixes (e.g., ``[1] ...``).
    """
    return "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(chunks))

# NeMo-style self-check-facts prompt (adapted from NeMo's built-in task)
_SELF_CHECK_FACTS_PROMPT = """\
You are given a task to identify if the hypothesis is grounded by / supported by the evidence.
You will only use the contents of the evidence and not rely on external knowledge.

Evidence:
{evidence}

Hypothesis:
{hypothesis}

Based on the evidence, is the hypothesis true? Respond with a score from 0.0 to 1.0.
1.0 means fully supported by the evidence, 0.0 means completely unsupported.
Output only a number."""

# Full claim-scoring prompt for detailed breakdown
_CLAIM_SCORING_PROMPT = """\
You are a faithfulness evaluator. Given an answer and context chunks, score how well each claim in the answer is supported by the context.

Context:
{context}

Answer:
{answer}

For each sentence in the answer, output a JSON array:
[{{"claim": "sentence text", "score": 0.0, "supported": true}}]

Score 1.0 = fully supported by the context, 0.0 = completely unsupported.
Output ONLY the JSON array, no other text."""

_FALLBACK_MESSAGE = (
    "I could not generate a reliable answer from the available documents. "
    "Please try a more specific query."
)


@dataclass
class ClaimScore:
    """Faithfulness score for a single claim.

    Attributes:
        claim: Claim text (typically a sentence from the answer).
        score: Score in [0.0, 1.0] measuring support by context.
        supported: Boolean support flag (typically derived from a threshold).
    """

    claim: str
    score: float
    supported: bool


@dataclass
class FaithfulnessResult:
    """Result of faithfulness evaluation.

    Attributes:
        overall_score: Overall support score in [0.0, 1.0].
        verdict: PASS/REJECT/MODIFY verdict derived from policy.
        warning: Whether to warn while still allowing the answer.
        claim_scores: Optional per-claim breakdown.
        hallucinated_entities: Lightweight hallucination signals (e.g., years/numbers).
        fallback_message: Optional canned fallback message for rejection flows.
    """

    overall_score: float
    verdict: RailVerdict
    warning: bool = False
    claim_scores: list[ClaimScore] = field(default_factory=list)
    hallucinated_entities: list[str] = field(default_factory=list)
    fallback_message: Optional[str] = None


class FaithfulnessChecker:
    """Check generated answers for faithfulness to retrieved context.

    Uses a two-phase approach:
    1. NeMo self-check-facts — quick overall score (adapted from NeMo's
       built-in SelfCheckFactsAction). Returns a single 0.0-1.0 score.
    2. Detailed claim scoring — per-sentence breakdown for observability.
    3. Lightweight entity hallucination detection (no LLM needed).

    Configurable to reject or flag unfaithful answers.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        action: str = "flag",
        use_self_check: bool = True,
    ) -> None:
        """Initialize a faithfulness checker.

        Args:
            threshold: Minimum overall score required to consider the answer
                faithful.
            action: Behavior when below threshold ("reject" or "flag").
            use_self_check: Whether to run a quick self-check prompt in
                addition to claim scoring.
        """
        self._threshold = threshold
        self._action = action  # "reject" or "flag"
        self._use_self_check = use_self_check

    def check(
        self,
        answer: str,
        context_chunks: list[str],
    ) -> FaithfulnessResult:
        """Evaluate faithfulness of answer against context.

        Uses the same context chunks that were passed to the generator (REQ-504).

        Args:
            answer: Generated answer text to evaluate.
            context_chunks: Evidence context chunks used for the generation.

        Returns:
            `FaithfulnessResult` containing overall score, verdict, and optional
            breakdown.
        """
        if not answer or not context_chunks:
            return FaithfulnessResult(
                overall_score=1.0,
                verdict=RailVerdict.PASS,
            )

        # Phases 1-3: Run LLM calls in parallel, entity detection is CPU-only
        self_check_score = None
        claim_scores: list[ClaimScore] = []
        hallucinated = self._detect_hallucinated_entities(answer, context_chunks)

        # Format once, reuse in both parallel tasks
        formatted_context = _format_numbered_chunks(context_chunks)

        if self._use_self_check:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="faith") as pool:
                fut_self = pool.submit(self._self_check_facts, answer, formatted_context)
                fut_claims = pool.submit(self._score_claims, answer, formatted_context)
                self_check_score = fut_self.result()
                claim_scores = fut_claims.result()
        else:
            claim_scores = self._score_claims(answer, formatted_context)

        # Compute overall score — prefer self-check if available
        if self_check_score is not None:
            overall = self_check_score
        elif claim_scores:
            overall = sum(c.score for c in claim_scores) / len(claim_scores)
        else:
            overall = 1.0

        # Penalize for hallucinated entities
        if hallucinated:
            penalty = min(0.3, len(hallucinated) * 0.1)
            overall = max(0.0, overall - penalty)

        # Determine verdict based on threshold and action
        if overall < self._threshold:
            if self._action == "reject":
                return FaithfulnessResult(
                    overall_score=overall,
                    verdict=RailVerdict.REJECT,
                    claim_scores=claim_scores,
                    hallucinated_entities=hallucinated,
                    fallback_message=_FALLBACK_MESSAGE,
                )
            else:
                return FaithfulnessResult(
                    overall_score=overall,
                    verdict=RailVerdict.PASS,
                    warning=True,
                    claim_scores=claim_scores,
                    hallucinated_entities=hallucinated,
                )

        return FaithfulnessResult(
            overall_score=overall,
            verdict=RailVerdict.PASS,
            claim_scores=claim_scores,
            hallucinated_entities=hallucinated,
        )

    def _self_check_facts(
        self, answer: str, formatted_context: str
    ) -> Optional[float]:
        """Compute a quick overall faithfulness score via self-check-facts.

        Adapted from NeMo's SelfCheckFactsAction — uses the evidence/hypothesis
        prompt format for a single 0.0-1.0 score.

        Args:
            answer: Answer text to evaluate as the hypothesis.
            formatted_context: Evidence text formatted for the prompt.

        Returns:
            Score in [0.0, 1.0] if available, otherwise None.
        """
        prompt = (
            "You are given a task to identify if the hypothesis is grounded by / "
            "supported by the evidence.\n"
            "You will only use the contents of the evidence and not rely on external knowledge.\n\n"
            "Based on the evidence, is the hypothesis true? Respond with a score from 0.0 to 1.0.\n"
            "1.0 means fully supported by the evidence, 0.0 means completely unsupported.\n"
            "Output only a number.\n\n"
            f"<evidence>{formatted_context}</evidence>\n\n"
            f"<hypothesis>{answer}</hypothesis>"
        )

        try:
            from src.platform.llm import call_oneshot

            response = call_oneshot(
                prompt,
                system=(
                    "You are a faithfulness evaluator. Output only a number "
                    "between 0.0 and 1.0."
                ),
            )
            if response:
                # Try to extract a float from the response
                cleaned = response.strip()
                # Handle responses like "0.8" or "Score: 0.8" or "The score is 0.8"
                numbers = re.findall(r"\b([01](?:\.\d+)?)\b", cleaned)
                if numbers:
                    score = float(numbers[0])
                    score = max(0.0, min(1.0, score))
                    logger.info("Self-check-facts score: %.2f", score)
                    return score
        except Exception as e:
            logger.warning("Self-check-facts failed: %s", e)

        return None

    def _score_claims(
        self, answer: str, formatted_context: str
    ) -> list[ClaimScore]:
        """Use an LLM to score each claim against context.

        Args:
            answer: Answer text to score.
            formatted_context: Evidence text formatted for the prompt.

        Returns:
            List of `ClaimScore` objects. Returns an empty list on failure.
        """
        prompt = (
            "You are a faithfulness evaluator. Given an answer and context chunks, "
            "score how well each claim in the answer is supported by the context.\n\n"
            "For each sentence in the answer, output a JSON array:\n"
            '[{"claim": "sentence text", "score": 0.0, "supported": true}]\n\n'
            "Score 1.0 = fully supported by the context, 0.0 = completely unsupported.\n"
            "Output ONLY the JSON array, no other text.\n\n"
            f"<context>{formatted_context}</context>\n\n"
            f"<answer_to_evaluate>{answer}</answer_to_evaluate>"
        )

        try:
            from src.platform.llm import call_oneshot
            from src.common import parse_json_object

            response = call_oneshot(
                prompt,
                system="You are a faithfulness evaluator. Output only JSON.",
            )
            if not response:
                logger.warning("Faithfulness LLM returned empty response")
                return []

            # Try to parse as JSON array
            response = response.strip()
            if response.startswith("["):
                parsed = orjson.loads(response)
            else:
                parsed_obj = parse_json_object(response)
                parsed = parsed_obj if isinstance(parsed_obj, list) else [parsed_obj]

            return [
                ClaimScore(
                    claim=str(item.get("claim", "")),
                    score=max(0.0, min(1.0, float(item.get("score", 0.0)))),
                    supported=bool(item.get("supported", False)),
                )
                for item in parsed
                if isinstance(item, dict)
            ]
        except (orjson.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning("Failed to parse faithfulness response: %s", e)
            return []
        except Exception as e:
            logger.warning("Faithfulness scoring failed: %s", e)
            return []

    def _detect_hallucinated_entities(
        self,
        answer: str,
        context_chunks: list[str],
    ) -> list[str]:
        """Check for entities/dates/numbers in answer not in any context chunk.

        Lightweight deterministic check — no LLM needed (REQ-505).

        Args:
            answer: Answer text to scan.
            context_chunks: Evidence chunks to compare against.

        Returns:
            List of lightweight hallucination signals (e.g., ``year:2024``).
        """
        context_text = " ".join(context_chunks).lower()
        hallucinated: list[str] = []

        # Check for years (4-digit numbers starting with 19 or 20)
        date_pattern = re.compile(r"\b(?:19|20)\d{2}\b")
        for match in date_pattern.finditer(answer):
            value = match.group()
            if value not in context_text:
                hallucinated.append(f"year:{value}")

        # Check for large numbers that might be statistics
        stat_pattern = re.compile(r"\b\d{3,}(?:\.\d+)?%?\b")
        for match in stat_pattern.finditer(answer):
            value = match.group()
            if value not in context_text:
                hallucinated.append(f"number:{value}")

        return hallucinated
