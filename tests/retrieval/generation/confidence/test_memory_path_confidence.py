"""Tests for #8: memory-path confidence gate in rag_chain.

Verifies that composite confidence scoring runs even when generation_source=="memory"
(i.e. reranked == []), and that retrieval_score is 0 by design on that path.
"""

import pytest

from src.retrieval.generation.confidence.scoring import compute_composite_confidence
from src.retrieval.generation.confidence.schemas import ConfidenceBreakdown


class TestMemoryPathConfidenceScoring:
    """Confidence scoring with empty reranker scores (memory-only path)."""

    def test_empty_reranker_scores_gives_zero_retrieval(self):
        """Memory path: no retrieval signal → retrieval_score = 0.0."""
        result = compute_composite_confidence(
            reranker_scores=[],
            llm_confidence_text="medium",
            answer="The conversation context indicates the user wants a summary here.",
            retrieved_texts=[],
        )
        assert isinstance(result, ConfidenceBreakdown)
        assert result.retrieval_score == 0.0

    def test_memory_path_composite_is_nonzero_with_llm_signal(self):
        """Even with no retrieval, LLM self-report contributes to composite."""
        result = compute_composite_confidence(
            reranker_scores=[],
            llm_confidence_text="high",
            answer="The conversation shows the user prefers JSON output format here.",
            retrieved_texts=[],
        )
        # LLM weight=0.25, score=0.85 → at least 0.25*0.85 = 0.2125 from LLM alone
        assert result.composite > 0.0
        assert result.llm_score == pytest.approx(0.85)

    def test_memory_path_composite_in_valid_range(self):
        """Composite must be in [0.0, 1.0] even without retrieval docs."""
        result = compute_composite_confidence(
            reranker_scores=[],
            llm_confidence_text="low",
            answer="Not sure about this at all, just going from memory here.",
            retrieved_texts=[],
        )
        assert 0.0 <= result.composite <= 1.0

    def test_memory_path_no_citation_ngram_overlap(self):
        """With retrieved_texts=[], n-gram overlap is 0 by design."""
        result = compute_composite_confidence(
            reranker_scores=[],
            llm_confidence_text="medium",
            answer="This answer comes from conversation memory only, no documents cited.",
            retrieved_texts=[],
        )
        # citation_score: no retrieved text → n-gram overlap = 0
        # citation markers also have no valid indices (num_chunks=0)
        # So citation_score should be near 0 or neutral (not 1.0)
        assert result.citation_score < 1.0

    def test_memory_path_citation_score_not_vacuously_one(self):
        """Empty retrieved_texts must not yield citation_score=1.0.

        Previously _split_sentences returning [] would give coverage=1.0 (vacuous).
        With fix #10, empty-answer coverage is 0.5 (neutral) not 1.0.
        With non-empty answer but no retrieved_texts, coverage should also not be 1.0.
        """
        result = compute_composite_confidence(
            reranker_scores=[],
            llm_confidence_text="medium",
            answer="The user asked about the previous conversation context earlier today.",
            retrieved_texts=[],
        )
        assert result.citation_score != 1.0, (
            f"Memory path should not yield vacuously-perfect citation_score=1.0, "
            f"got {result.citation_score}"
        )

    def test_memory_path_lower_composite_than_strong_retrieval(self):
        """Memory path (retrieval=0) composites lower than strong retrieval path."""
        text = "The system uses vector search for retrieval of important information here."
        memory_result = compute_composite_confidence(
            reranker_scores=[],
            llm_confidence_text="high",
            answer=text,
            retrieved_texts=[],
        )
        retrieval_result = compute_composite_confidence(
            reranker_scores=[0.9, 0.9, 0.9],
            llm_confidence_text="high",
            answer=text,
            retrieved_texts=[text],
        )
        assert memory_result.composite < retrieval_result.composite, (
            f"Memory path ({memory_result.composite:.3f}) should be lower than "
            f"strong retrieval ({retrieval_result.composite:.3f})"
        )
