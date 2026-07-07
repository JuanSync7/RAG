"""Regression tests for confidence scoring robustness fixes.

Covers:
  #1  - Sentence splitter fragility (abbreviations, decimals, URLs, ellipses, newlines)
  #9  - Partial citation credit (fractional score per sentence)
  #10 - Short-sentence handling (threshold + empty-list guard)
  #13 - ConfidenceWeights dataclass
"""

import pytest

from src.retrieval.generation.confidence.scoring import (
    _split_sentences,
    compute_citation_coverage,
    compute_composite_confidence,
)
from src.retrieval.generation.confidence.schemas import (
    ConfidenceBreakdown,
    ConfidenceWeights,
)


# ---------------------------------------------------------------------------
# #1 - Sentence splitter robustness
# ---------------------------------------------------------------------------


class TestSplitSentencesRobust:
    """_split_sentences must handle edge cases the old regex got wrong."""

    def test_abbreviation_mr_not_split(self):
        text = "Mr. Smith went to the store. He bought milk."
        sentences = _split_sentences(text)
        # "Mr. Smith went to the store." should not be split at "Mr."
        assert any("Mr." in s for s in sentences), f"Got: {sentences}"
        # Should produce 2 sentences, not 3+
        assert len(sentences) <= 2, f"Over-split on 'Mr.': {sentences}"

    def test_abbreviation_etc_not_split(self):
        text = "We need supplies, etc. Please bring them tomorrow."
        sentences = _split_sentences(text)
        assert len(sentences) <= 2, f"Over-split on 'etc.': {sentences}"

    def test_abbreviation_ie_not_split(self):
        text = "Use common formats, i.e. JSON or XML. This is required."
        sentences = _split_sentences(text)
        assert len(sentences) <= 2, f"Over-split on 'i.e.': {sentences}"

    def test_abbreviation_eg_not_split(self):
        text = "Use common formats, e.g. JSON or XML. This is required."
        sentences = _split_sentences(text)
        assert len(sentences) <= 2, f"Over-split on 'e.g.': {sentences}"

    def test_decimal_not_split(self):
        text = "The voltage is 3.14 volts under load. Measure carefully."
        sentences = _split_sentences(text)
        assert len(sentences) <= 2, f"Over-split on decimal '3.14': {sentences}"

    def test_version_number_not_split(self):
        text = "Use version v2.0 of the library. It has better performance."
        sentences = _split_sentences(text)
        assert len(sentences) <= 2, f"Over-split on 'v2.0': {sentences}"

    def test_url_not_split(self):
        text = "Visit https://example.com/path for details. Documentation is there."
        sentences = _split_sentences(text)
        assert len(sentences) <= 2, f"Over-split inside URL: {sentences}"

    def test_ellipsis_not_split_into_many(self):
        text = "The answer is... complicated to explain. Please read the docs."
        sentences = _split_sentences(text)
        assert len(sentences) <= 2, f"Over-split on ellipsis: {sentences}"

    def test_newline_boundary_treated_as_sentence_break(self):
        text = "First sentence has enough words here.\n\nSecond sentence also has words."
        sentences = _split_sentences(text)
        assert len(sentences) == 2, f"Should split on double newline: {sentences}"

    def test_single_newline_no_space(self):
        text = "First long sentence ends here.\nSecond sentence starts here."
        sentences = _split_sentences(text)
        # Should split at \n even without trailing space
        assert len(sentences) == 2, f"Should split on newline without space: {sentences}"

    def test_exclamation_and_question_still_split(self):
        text = "Is this working correctly? Yes it certainly is working well!"
        sentences = _split_sentences(text)
        assert len(sentences) == 2, f"Should split on ? and !: {sentences}"

    def test_dr_abbreviation_not_split(self):
        text = "Dr. Jones confirmed the results are valid. The test passed."
        sentences = _split_sentences(text)
        assert len(sentences) <= 2, f"Over-split on 'Dr.': {sentences}"

    def test_vs_abbreviation_not_split(self):
        text = "Performance vs. accuracy is a key tradeoff. Both matter greatly."
        sentences = _split_sentences(text)
        assert len(sentences) <= 2, f"Over-split on 'vs.': {sentences}"


# ---------------------------------------------------------------------------
# #9 - Partial citation credit
# ---------------------------------------------------------------------------


class TestPartialCitationCredit:
    """Citation score should use fractional per-sentence credit."""

    def test_fully_valid_citations_score_one(self):
        """All citations valid → credit = 1.0."""
        answer = "The result is confirmed by the source [1] for all cases."
        # 1 chunk → index 1 is valid
        result = compute_citation_coverage(answer, ["source text about the result"])
        # citation_score contribution should be 1.0 (70% weight)
        # n-gram may be 0 → total >= 0.70
        assert result >= 0.70, f"Expected >= 0.70, got {result}"

    def test_partial_citation_half_valid(self):
        """[1, 99] with 1 chunk → citation credit = 0.5 (not 1.0 or 0.0)."""
        answer = "This fact is supported by references [1] and [99] together here."
        result = compute_citation_coverage(answer, ["fact text for one chunk"])
        # citation credit per sentence = 0.5 (1 valid out of 2)
        # citation_score contribution = 0.70 * 0.5 = 0.35
        # n-gram = 0 → total ≈ 0.35
        assert 0.30 <= result <= 0.45, f"Expected ~0.35 for half-valid citations, got {result}"

    def test_all_invalid_citations_score_zero_citation(self):
        """[99, 100] with 1 chunk → citation credit = 0."""
        answer = "The answer is based on reference [99] and reference [100] always."
        result = compute_citation_coverage(answer, ["totally different text here"])
        # citation_score = 0, n-gram ≈ 0 → total ≈ 0
        assert result < 0.20, f"Expected near 0 for all-invalid citations, got {result}"

    def test_two_sentences_one_fully_cited_one_half(self):
        """Mixed per-sentence credits average correctly."""
        # Sentence 1: [1] valid → credit 1.0
        # Sentence 2: [1, 99] → credit 0.5 (1 valid of 2)
        # Average citation credit = 0.75 → weighted 0.70 * 0.75 = 0.525
        answer = (
            "First claim is backed by reference [1] as clearly shown. "
            "Second claim uses sources [1] and [99] for its argument here."
        )
        result = compute_citation_coverage(answer, ["source text about the claim"])
        # citation contribution = 0.70 * 0.75 = 0.525; n-gram may vary
        # Allow a range; the key is it should be between pure 0.5 and pure 0.7
        assert 0.40 <= result <= 0.80, f"Unexpected result: {result}"


# ---------------------------------------------------------------------------
# #10 - Short-sentence threshold and empty-list guard
# ---------------------------------------------------------------------------


class TestShortSentenceHandling:
    """Short sentences: threshold >= 3 words, never return empty list spuriously."""

    def test_three_word_sentence_kept(self):
        """A 3-word sentence should be kept (not filtered)."""
        sentences = _split_sentences("Limit is 100.")
        # Should NOT be filtered out — it's >= 3 words
        assert len(sentences) == 1, f"3-word sentence should be kept, got: {sentences}"
        assert "Limit is 100" in sentences[0]

    def test_two_word_sentence_may_be_filtered(self):
        """A 2-word sentence can still be filtered."""
        sentences = _split_sentences("Done. This is a longer sentence with many words.")
        # "Done." is 1 word — filtered; keep the longer one
        longer = [s for s in sentences if "longer sentence" in s]
        assert len(longer) == 1

    def test_all_short_returns_at_least_one(self):
        """When all fragments are short, keep at least one (no empty list)."""
        # All 1-2 word fragments — old code would return []
        text = "Yes. No. Done. OK."
        sentences = _split_sentences(text)
        assert len(sentences) >= 1, "Should return at least one sentence even if all are short"

    def test_empty_sentence_list_yields_neutral_coverage(self):
        """When splitter returns no substantive sentences, coverage should be 0.5 (neutral).

        Previously returned 1.0 (vacuously perfect), which could inflate scores.
        The 0.5 neutral value avoids penalising short answers while not rewarding them.
        """
        # A 2-word answer — even with the keep-1 guard, this tests the neutral fallback
        # by feeding an empty retrieved_texts (forces 0 n-gram) and no citation markers
        # The coverage should NOT be 1.0
        result = compute_citation_coverage("", [])
        # Empty answer with no chunks — substantive? No. Should be 0.5 neutral.
        assert result == 0.5, f"Empty answer+chunks should yield neutral 0.5, got {result}"

    def test_all_short_fragments_neutral_not_perfect(self):
        """An answer made entirely of < 3-word fragments should yield 0.5, not 1.0."""
        # "Yes. No." — after splitting, all fragments are 1 word each
        # keep-1 guard may keep one, but it's still a short non-substantive answer
        # When _split_sentences returns only very short kept fragments,
        # coverage shouldn't be vacuously 1.0
        result = compute_citation_coverage("OK.", ["some retrieved text that is longer"])
        # Not 1.0 — either 0.5 (neutral when no substantive) or low (no citation/overlap)
        assert result != 1.0, f"Short-only answer should not be 1.0, got {result}"


# ---------------------------------------------------------------------------
# #13 - ConfidenceWeights dataclass
# ---------------------------------------------------------------------------


class TestConfidenceWeights:
    """ConfidenceWeights frozen dataclass with sum-to-1 validation."""

    def test_valid_weights_construct(self):
        w = ConfidenceWeights(retrieval=0.50, llm=0.25, citation=0.25)
        assert w.retrieval == 0.50
        assert w.llm == 0.25
        assert w.citation == 0.25

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            ConfidenceWeights(retrieval=0.5, llm=0.5, citation=0.5)

    def test_frozen(self):
        w = ConfidenceWeights(retrieval=0.50, llm=0.25, citation=0.25)
        with pytest.raises((AttributeError, TypeError)):
            w.retrieval = 0.8  # type: ignore[misc]

    def test_composite_accepts_weights_dataclass(self):
        """compute_composite_confidence should accept ConfidenceWeights instance."""
        w = ConfidenceWeights(retrieval=0.50, llm=0.25, citation=0.25)
        result = compute_composite_confidence(
            reranker_scores=[0.8],
            llm_confidence_text="high",
            answer="The system uses vector search for efficient retrieval here.",
            retrieved_texts=["The system uses vector search for efficient retrieval here."],
            weights=w,
        )
        assert isinstance(result, ConfidenceBreakdown)
        assert 0.0 <= result.composite <= 1.0

    def test_composite_kwargs_still_work(self):
        """Existing float-kwarg call must still work (back-compat)."""
        result = compute_composite_confidence(
            reranker_scores=[0.8],
            llm_confidence_text="high",
            answer="The system uses vector search for efficient retrieval here.",
            retrieved_texts=["The system uses vector search for efficient retrieval here."],
            retrieval_weight=0.50,
            llm_weight=0.25,
            citation_weight=0.25,
        )
        assert isinstance(result, ConfidenceBreakdown)

    def test_weights_stored_in_breakdown(self):
        """Breakdown should record actual weights used."""
        w = ConfidenceWeights(retrieval=0.40, llm=0.40, citation=0.20)
        result = compute_composite_confidence(
            reranker_scores=[0.7],
            llm_confidence_text="medium",
            answer="Some answer text with enough words for processing.",
            retrieved_texts=["Some retrieved text with more content here."],
            weights=w,
        )
        assert result.retrieval_weight == pytest.approx(0.40)
        assert result.llm_weight == pytest.approx(0.40)
        assert result.citation_weight == pytest.approx(0.20)
