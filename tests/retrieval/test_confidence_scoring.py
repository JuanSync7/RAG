"""Unit tests for composite confidence scoring module."""

import pytest

from src.retrieval.generation.confidence.scoring import (
    compute_citation_coverage,
    compute_composite_confidence,
    compute_retrieval_confidence,
    parse_llm_confidence,
    _split_sentences,
    _has_substantial_overlap,
)
from src.retrieval.generation.confidence.schemas import ConfidenceBreakdown


class TestComputeRetrievalConfidence:
    """Tests for compute_retrieval_confidence."""

    def test_empty_scores(self):
        assert compute_retrieval_confidence([]) == 0.0

    def test_single_score(self):
        assert compute_retrieval_confidence([0.8]) == pytest.approx(0.8)

    def test_fewer_than_top_n(self):
        result = compute_retrieval_confidence([0.9, 0.7], top_n=3)
        assert result == pytest.approx(0.8)

    def test_top_n_averaging(self):
        scores = [0.9, 0.8, 0.7, 0.3, 0.1]
        result = compute_retrieval_confidence(scores, top_n=3)
        assert result == pytest.approx(0.8)

    def test_unsorted_input(self):
        scores = [0.3, 0.9, 0.1, 0.8, 0.7]
        result = compute_retrieval_confidence(scores, top_n=3)
        assert result == pytest.approx(0.8)

    def test_clamp_to_unit(self):
        assert compute_retrieval_confidence([1.5]) == 1.0
        assert compute_retrieval_confidence([-0.5]) == 0.0

    def test_all_zeros(self):
        assert compute_retrieval_confidence([0.0, 0.0, 0.0]) == 0.0


class TestParseLlmConfidence:
    """Tests for parse_llm_confidence."""

    def test_high(self):
        assert parse_llm_confidence("high") == 0.85

    def test_medium(self):
        assert parse_llm_confidence("medium") == 0.55

    def test_low(self):
        assert parse_llm_confidence("low") == 0.25

    def test_case_insensitive(self):
        assert parse_llm_confidence("HIGH") == 0.85
        assert parse_llm_confidence("Medium") == 0.55

    def test_with_extra_text(self):
        assert parse_llm_confidence("high confidence") == 0.85
        assert parse_llm_confidence("I have medium confidence") == 0.55

    def test_unknown_defaults_to_neutral(self):
        assert parse_llm_confidence("unsure") == 0.5
        assert parse_llm_confidence("") == 0.5
        assert parse_llm_confidence("42") == 0.5

    def test_whitespace_handling(self):
        assert parse_llm_confidence("  high  ") == 0.85


class TestComputeCitationCoverage:
    """Tests for compute_citation_coverage."""

    def test_empty_answer(self):
        assert compute_citation_coverage("", ["some text"]) == 1.0

    def test_empty_chunks(self):
        assert compute_citation_coverage(
            "This is a test sentence with several words.", []
        ) == 0.0

    def test_full_coverage_with_citations(self):
        text = "The system uses vector search for retrieval [1]."
        result = compute_citation_coverage(text, [text])
        assert result == 1.0

    def test_overlap_only_no_citations(self):
        text = "The system uses vector search for retrieval."
        result = compute_citation_coverage(text, [text])
        # No citation markers → citation score=0, n-gram overlap=1.0
        # Blend: 0.70*0 + 0.30*1.0 = 0.30
        assert result == pytest.approx(0.30)

    def test_no_coverage(self):
        result = compute_citation_coverage(
            "Completely unrelated content that does not match anything.",
            ["Vector databases store high-dimensional embeddings efficiently."],
        )
        assert result == 0.0

    def test_partial_coverage(self):
        answer = (
            "The system uses vector search for retrieval of important documents. "
            "It also does something completely new and unrelated to everything we know."
        )
        chunks = ["The system uses vector search for retrieval of important documents."]
        result = compute_citation_coverage(answer, chunks)
        assert 0.0 < result < 1.0

    def test_short_chunks_return_zero(self):
        result = compute_citation_coverage(
            "This is a longer answer sentence.", ["ab cd"]
        )
        assert result == 0.0


class TestSplitSentences:
    """Tests for _split_sentences."""

    def test_empty_string(self):
        assert _split_sentences("") == []

    def test_single_sentence(self):
        result = _split_sentences("This is a test sentence.")
        assert len(result) == 1

    def test_multiple_sentences(self):
        result = _split_sentences(
            "First sentence has enough words. "
            "Second sentence also has enough words. "
            "Third sentence has enough words too."
        )
        assert len(result) == 3

    def test_filters_short_fragments(self):
        result = _split_sentences("OK. This is a real sentence with enough words.")
        assert len(result) == 1
        assert "real sentence" in result[0]


class TestComputeCompositeConfidence:
    """Tests for compute_composite_confidence."""

    def test_default_weights_sum_to_one(self):
        result = compute_composite_confidence(
            reranker_scores=[0.8, 0.7, 0.6],
            llm_confidence_text="high",
            answer="The system uses vector search for efficient retrieval.",
            retrieved_texts=["The system uses vector search for efficient retrieval."],
        )
        assert isinstance(result, ConfidenceBreakdown)
        weights_sum = result.retrieval_weight + result.llm_weight + result.citation_weight
        assert weights_sum == pytest.approx(1.0)

    def test_composite_in_range(self):
        result = compute_composite_confidence(
            reranker_scores=[0.5],
            llm_confidence_text="medium",
            answer="Some answer text.",
            retrieved_texts=["Different text entirely."],
        )
        assert 0.0 <= result.composite <= 1.0

    def test_all_high_signals(self):
        text = "The voltage regulator maintains a steady output of three point three volts."
        result = compute_composite_confidence(
            reranker_scores=[0.95, 0.90, 0.85],
            llm_confidence_text="high",
            answer=text,
            retrieved_texts=[text],
        )
        assert result.composite >= 0.7

    def test_all_low_signals(self):
        result = compute_composite_confidence(
            reranker_scores=[0.1, 0.05],
            llm_confidence_text="low",
            answer="Unrelated answer with no matching content whatsoever.",
            retrieved_texts=["Completely different source material about something else."],
        )
        assert result.composite < 0.5

    def test_custom_weights(self):
        result = compute_composite_confidence(
            reranker_scores=[0.8],
            llm_confidence_text="high",
            answer="test answer.",
            retrieved_texts=["other text."],
            retrieval_weight=1.0,
            llm_weight=0.0,
            citation_weight=0.0,
        )
        assert result.composite == pytest.approx(0.8)
        assert result.retrieval_weight == 1.0

    def test_invalid_weights_raises(self):
        """Weights that do not sum to 1.0 must raise ValueError."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            compute_composite_confidence(
                reranker_scores=[0.8],
                llm_confidence_text="high",
                answer="test.",
                retrieved_texts=["test."],
                retrieval_weight=0.5,
                llm_weight=0.5,
                citation_weight=0.5,
            )

    def test_valid_weights_do_not_raise(self):
        """Weights that sum exactly to 1.0 must not raise."""
        result = compute_composite_confidence(
            reranker_scores=[0.6],
            llm_confidence_text="medium",
            answer="test.",
            retrieved_texts=["test."],
            retrieval_weight=0.4,
            llm_weight=0.4,
            citation_weight=0.2,
        )
        assert 0.0 <= result.composite <= 1.0

    @pytest.mark.parametrize("bad_score", [-0.5, -1.0, 1.5, 2.0])
    def test_retrieval_score_clamped(self, bad_score):
        """Individual reranker scores outside [0, 1] are clamped, not crashed."""
        result = compute_composite_confidence(
            reranker_scores=[bad_score],
            llm_confidence_text="medium",
            answer="test answer.",
            retrieved_texts=["test answer."],
            retrieval_weight=1.0,
            llm_weight=0.0,
            citation_weight=0.0,
        )
        assert 0.0 <= result.composite <= 1.0

    def test_composite_formula_arithmetic(self):
        """Verify weighted composite arithmetic with known, hand-computable inputs.

        retrieval = top-1 of [0.8] = 0.8
        llm       = parse_llm_confidence("high") = 0.85
        citation  = compute_citation_coverage("", [...]) = 1.0  (empty answer → vacuously 1.0)
        composite = 0.50*0.8 + 0.30*0.85 + 0.20*1.0
                  = 0.40 + 0.255 + 0.20 = 0.855
        """
        result = compute_composite_confidence(
            reranker_scores=[0.8],
            llm_confidence_text="high",
            answer="",           # empty answer → citation_score = 1.0
            retrieved_texts=["any text"],
            retrieval_weight=0.50,
            llm_weight=0.30,
            citation_weight=0.20,
        )
        expected = 0.50 * 0.8 + 0.30 * 0.85 + 0.20 * 1.0
        assert result.composite == pytest.approx(expected, abs=1e-6)
