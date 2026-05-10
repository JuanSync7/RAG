# @summary
# Pure utility functions for 3-signal composite confidence scoring.
# Exports: compute_retrieval_confidence, parse_llm_confidence, compute_citation_coverage, compute_composite_confidence
# Deps: re, src.retrieval.generation.confidence.schemas
# @end-summary
"""Pure utility functions for composite confidence scoring.

All functions are deterministic, side-effect-free, and require no I/O.
They implement the 3-signal confidence model from the retrieval spec:

  composite = (W_r * retrieval) + (W_l * llm) + (W_c * citation)

Default weights: retrieval=0.50, llm=0.25, citation=0.25.
"""

from __future__ import annotations

import re

from src.retrieval.generation.confidence.schemas import ConfidenceBreakdown

# Downward correction map for LLM overconfidence bias.
# LLMs tend to report "high" even when evidence is weak,
# so we map conservatively.
LLM_CONFIDENCE_MAP = {
    "high": 0.85,
    "medium": 0.55,
    "low": 0.25,
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def compute_retrieval_confidence(
    reranker_scores: list[float],
    top_n: int = 3,
) -> float:
    """Compute retrieval confidence from reranker scores.

    Takes the average of the top-N reranker scores. Using top-N instead of
    the single best score smooths outliers while still reflecting overall
    retrieval quality.

    Args:
        reranker_scores: Sigmoid-normalized scores from the cross-encoder
            reranker, each in [0.0, 1.0].
        top_n: Number of top scores to average. Defaults to 3.

    Returns:
        Average of top-N scores, clamped to [0.0, 1.0].
        Returns 0.0 if the score list is empty.
    """
    if not reranker_scores:
        return 0.0
    sorted_scores = sorted(reranker_scores, reverse=True)
    top_scores = sorted_scores[:top_n]
    avg = sum(top_scores) / len(top_scores)
    return max(0.0, min(1.0, avg))


def parse_llm_confidence(llm_confidence_text: str) -> float:
    """Map LLM self-reported confidence to a numerical score.

    The LLM is prompted to report confidence as "high", "medium", or "low".
    This function maps those labels to numerical values with a downward
    correction for overconfidence bias.

    Args:
        llm_confidence_text: Raw confidence text from the LLM response
            (e.g., "high", "MEDIUM", "Low confidence").

    Returns:
        Mapped score in [0.0, 1.0]. Defaults to 0.5 (neutral) if the
        text cannot be parsed.
    """
    if not llm_confidence_text:
        return 0.5
    normalized = llm_confidence_text.strip().lower()
    # Try direct match first
    if normalized in LLM_CONFIDENCE_MAP:
        return LLM_CONFIDENCE_MAP[normalized]
    # Try partial match (e.g., "high confidence" or "medium level")
    for key, value in LLM_CONFIDENCE_MAP.items():
        if key in normalized:
            return value
    return 0.5


_CITATION_RE = re.compile(r"\[(\d+)\]")


def compute_citation_coverage(
    answer: str,
    retrieved_texts: list[str],
    min_overlap_words: int = 5,
) -> float:
    """Compute citation coverage using a dual approach.

    Primary signal (70% weight): Citation marker checking — counts what
    fraction of substantive sentences contain valid citation markers
    like [1], [2] that reference actual chunk indices. This is reliable
    because the LLM is explicitly prompted to cite, and invalid citations
    (e.g., [7] when only 5 chunks exist) are a hallucination signal.

    Secondary signal (30% weight): N-gram overlap — checks for 5+ word
    consecutive overlap between answer sentences and retrieved text.
    This catches cases where the LLM uses source material without citing.

    Args:
        answer: Generated answer text.
        retrieved_texts: List of retrieved document chunk texts.
        min_overlap_words: Minimum consecutive word overlap for the
            secondary n-gram check. Defaults to 5.

    Returns:
        Blended citation coverage score in [0.0, 1.0].
        Returns 1.0 if the answer has no sentences (vacuously true).
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return 1.0

    num_chunks = len(retrieved_texts)
    valid_indices = set(range(1, num_chunks + 1))  # 1-based

    # Primary: citation marker checking
    cited_count = 0
    for sentence in sentences:
        citations = _CITATION_RE.findall(sentence)
        if citations:
            # At least one valid citation index
            cited_indices = {int(c) for c in citations}
            if cited_indices & valid_indices:
                cited_count += 1

    citation_score = cited_count / len(sentences) if sentences else 1.0

    # Secondary: n-gram overlap
    overlap_score = _compute_ngram_overlap(sentences, retrieved_texts, min_overlap_words)

    # Blend: 70% citation markers, 30% n-gram overlap
    return 0.70 * citation_score + 0.30 * overlap_score


def _compute_ngram_overlap(
    sentences: list[str],
    retrieved_texts: list[str],
    min_overlap_words: int = 5,
) -> float:
    """Compute n-gram overlap score between sentences and retrieved text."""
    if not sentences or not retrieved_texts:
        return 0.0

    all_retrieved = " ".join(retrieved_texts).lower()
    retrieved_words = all_retrieved.split()
    if len(retrieved_words) < min_overlap_words:
        return 0.0

    retrieved_ngrams = set()
    for i in range(len(retrieved_words) - min_overlap_words + 1):
        ngram = " ".join(retrieved_words[i : i + min_overlap_words])
        retrieved_ngrams.add(ngram)

    grounded_count = 0
    for sentence in sentences:
        if _has_substantial_overlap(sentence, retrieved_ngrams, min_overlap_words):
            grounded_count += 1

    return grounded_count / len(sentences)


def compute_composite_confidence(
    reranker_scores: list[float],
    llm_confidence_text: str,
    answer: str,
    retrieved_texts: list[str],
    retrieval_weight: float = 0.50,
    llm_weight: float = 0.25,
    citation_weight: float = 0.25,
) -> ConfidenceBreakdown:
    """Compute the 3-signal composite confidence score.

    Combines retrieval quality (objective), LLM self-report (subjective),
    and citation coverage (structural) into a single weighted composite.

    Args:
        reranker_scores: Reranker scores for retrieved documents.
        llm_confidence_text: LLM self-reported confidence level text.
        answer: Generated answer text.
        retrieved_texts: Retrieved document chunk texts.
        retrieval_weight: Weight for retrieval signal. Default 0.50.
        llm_weight: Weight for LLM signal. Default 0.25.
        citation_weight: Weight for citation signal. Default 0.25.

    Returns:
        ConfidenceBreakdown with all three signals and the composite score.
    """
    total_weight = retrieval_weight + llm_weight + citation_weight
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(
            f"Confidence weights must sum to 1.0, got {total_weight:.6f} "
            f"(retrieval={retrieval_weight}, llm={llm_weight}, citation={citation_weight})"
        )

    retrieval_score = compute_retrieval_confidence(reranker_scores)
    llm_score = parse_llm_confidence(llm_confidence_text)
    citation_score = compute_citation_coverage(answer, retrieved_texts)

    composite = (
        retrieval_weight * retrieval_score
        + llm_weight * llm_score
        + citation_weight * citation_score
    )
    composite = max(0.0, min(1.0, composite))

    return ConfidenceBreakdown(
        retrieval_score=retrieval_score,
        llm_score=llm_score,
        citation_score=citation_score,
        composite=composite,
        retrieval_weight=retrieval_weight,
        llm_weight=llm_weight,
        citation_weight=citation_weight,
    )


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation boundaries.

    Filters out very short fragments (< 4 words) that are likely
    headings, citations, or formatting artifacts rather than claims.

    Args:
        text: Input text to split.

    Returns:
        List of sentence strings with at least 4 words each.
    """
    if not text or not text.strip():
        return []
    raw = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in raw if len(s.strip().split()) >= 4]


def _has_substantial_overlap(
    sentence: str,
    retrieved_ngrams: set,
    min_overlap_words: int = 5,
) -> bool:
    """Check if a sentence has substantial n-gram overlap with retrieved text.

    Args:
        sentence: A single sentence from the answer.
        retrieved_ngrams: Pre-computed set of n-grams from retrieved text.
        min_overlap_words: N-gram size used for overlap checking.

    Returns:
        True if the sentence contains at least one n-gram match.
    """
    words = sentence.lower().split()
    if len(words) < min_overlap_words:
        return False
    for i in range(len(words) - min_overlap_words + 1):
        ngram = " ".join(words[i : i + min_overlap_words])
        if ngram in retrieved_ngrams:
            return True
    return False
