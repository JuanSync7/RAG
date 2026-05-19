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
from typing import Optional

from src.retrieval.generation.confidence.schemas import ConfidenceBreakdown, ConfidenceWeights
from src.platform.observability import get_tracer

# Downward correction map for LLM overconfidence bias.
# LLMs tend to report "high" even when evidence is weak,
# so we map conservatively.
LLM_CONFIDENCE_MAP = {
    "high": 0.85,
    "medium": 0.55,
    "low": 0.25,
}

# ---------------------------------------------------------------------------
# Sentence splitting — hand-rolled, no heavy deps
# ---------------------------------------------------------------------------

# Abbreviations that end with a period but are NOT sentence boundaries.
# Ordered longest-first to avoid prefix shadowing (e.g. "jr" before "j").
_ABBREVS: frozenset[str] = frozenset(
    [
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st",
        "vs", "etc", "approx", "dept", "est", "govt", "inc",
        "corp", "ltd", "co", "fig", "vol", "no",
        "e.g", "i.e", "viz", "cf",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug",
        "sep", "oct", "nov", "dec",
    ]
)

# Matches a run of digits.digits (decimal) or word.digits (version like v2.0).
_DECIMAL_RE = re.compile(r"\w\.\d")

# Matches URLs — skip any period inside an http(s):// run.
_URL_RE = re.compile(r"https?://\S+")

# Terminal punctuation followed by whitespace or newline (the split candidate).
# We exclude pure ellipsis sequences ("..." or "…") — they indicate continuation,
# not a sentence boundary, unless followed by a capital letter (handled in loop).
_TERM_PUNCT_RE = re.compile(r"([.!?…]+)(\s+|\n)")

# Pure ellipsis pattern — "...", "....", "…" — not a sentence boundary.
_ELLIPSIS_RE = re.compile(r"^\.{2,}$|^…+$")


def _is_abbreviation_boundary(text: str, match_start: int) -> bool:
    """Return True if the period at match_start is an abbreviation, not a sentence end."""
    # Find the word preceding the period.
    word_end = match_start
    word_start = word_end - 1
    while word_start >= 0 and (text[word_start].isalpha() or text[word_start] == "."):
        word_start -= 1
    token = text[word_start + 1 : word_end].lower().rstrip(".")
    return token in _ABBREVS


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences with a robust hand-rolled splitter.

    Handles:
      - Common abbreviations (Mr., Dr., etc., e.g., i.e., vs.)
      - Decimal numbers (3.14, v2.0) — not split
      - URLs (https://...) — periods inside not split
      - Ellipses (... or …) — treated as a single boundary
      - Single newline without trailing space
      - Double newline as a hard paragraph boundary

    Filters out fragments shorter than 3 words.  If ALL fragments are
    shorter than 3 words, keeps the longest one to avoid returning an
    empty list spuriously (a vacuously empty list inflated coverage to
    1.0 — see #10).

    Args:
        text: Input text to split.

    Returns:
        List of sentence strings, each with at least 3 words.
        Never returns an empty list if the input is non-empty.
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    # Replace URLs with a placeholder to protect periods inside them.
    url_map: dict[str, str] = {}
    def _store_url(m: re.Match) -> str:
        key = f"__URL{len(url_map)}__"
        url_map[key] = m.group(0)
        return key

    protected = _URL_RE.sub(_store_url, text)

    # Split on hard paragraph breaks first.
    paragraphs = re.split(r"\n{2,}", protected)

    raw_sentences: list[str] = []
    for para in paragraphs:
        # Within a paragraph, split on terminal punctuation + whitespace/newline.
        parts: list[str] = []
        last = 0
        for m in _TERM_PUNCT_RE.finditer(para):
            punct = m.group(1)
            end = m.end()
            boundary_pos = m.start()

            # Ellipsis check: "..." or "…" indicates continuation, not a sentence end.
            if _ELLIPSIS_RE.match(punct):
                continue

            # Decimal / version check: digit.digit — not a boundary.
            if "." in punct and _DECIMAL_RE.search(para, max(0, boundary_pos - 2)):
                before_window = para[max(0, boundary_pos - 2): boundary_pos + 2]
                if _DECIMAL_RE.search(before_window):
                    continue

            # Abbreviation check (only for single ".").
            if punct == "." and _is_abbreviation_boundary(para, boundary_pos):
                continue

            sentence = para[last:end].strip()
            if sentence:
                parts.append(sentence)
            last = end

        tail = para[last:].strip()
        if tail:
            parts.append(tail)

        # Also split on single newline within a paragraph (no trailing space needed).
        expanded: list[str] = []
        for part in parts:
            sub = [s.strip() for s in part.split("\n") if s.strip()]
            expanded.extend(sub)

        raw_sentences.extend(expanded)

    # Restore URLs.
    restored = []
    for s in raw_sentences:
        for key, url in url_map.items():
            s = s.replace(key, url)
        restored.append(s)

    # Filter by word count (>= 3).
    substantive = [s for s in restored if len(s.split()) >= 3]
    if substantive:
        return substantive

    # Keep-at-least-one guard: return the longest fragment so the list is
    # never empty for non-empty input (prevents vacuous coverage=1.0, #10).
    if restored:
        return [max(restored, key=lambda s: len(s.split()))]
    return []


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
        # #10: Return 0.5 (neutral) rather than 1.0 (vacuously perfect).
        # An answer with no substantive sentences should neither be rewarded
        # nor penalised — 0.5 keeps the composite in a neutral range while
        # avoiding the false-confidence of a perfect coverage score.
        return 0.5

    num_chunks = len(retrieved_texts)
    valid_indices = set(range(1, num_chunks + 1))  # 1-based

    # Primary: citation marker checking — fractional credit per sentence (#9).
    # Previously a sentence got full credit if *any* citation was valid.
    # Now credit = valid_count / total_count (e.g. [1, 99] with 1 chunk → 0.5).
    citation_credit_sum = 0.0
    for sentence in sentences:
        citations = _CITATION_RE.findall(sentence)
        if citations:
            cited_indices = {int(c) for c in citations}
            valid_count = len(cited_indices & valid_indices)
            total_count = len(cited_indices)
            citation_credit_sum += valid_count / total_count

    citation_score = citation_credit_sum / len(sentences)

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


_DEFAULT_WEIGHTS = ConfidenceWeights(retrieval=0.50, llm=0.25, citation=0.25)


def compute_composite_confidence(
    reranker_scores: list[float],
    llm_confidence_text: str,
    answer: str,
    retrieved_texts: list[str],
    retrieval_weight: float = 0.50,
    llm_weight: float = 0.25,
    citation_weight: float = 0.25,
    weights: Optional[ConfidenceWeights] = None,
) -> ConfidenceBreakdown:
    """Compute the 3-signal composite confidence score.

    Combines retrieval quality (objective), LLM self-report (subjective),
    and citation coverage (structural) into a single weighted composite.

    Accepts weights as either a ConfidenceWeights dataclass (preferred, validated
    once at construction) or as three float kwargs (back-compat).  If both are
    provided, the dataclass takes precedence.

    Args:
        reranker_scores: Reranker scores for retrieved documents.
        llm_confidence_text: LLM self-reported confidence level text.
        answer: Generated answer text.
        retrieved_texts: Retrieved document chunk texts.
        retrieval_weight: Weight for retrieval signal. Default 0.50.
        llm_weight: Weight for LLM signal. Default 0.25.
        citation_weight: Weight for citation signal. Default 0.25.
        weights: Optional ConfidenceWeights dataclass (takes precedence over
            individual float kwargs).

    Returns:
        ConfidenceBreakdown with all three signals and the composite score.
    """
    with get_tracer().span(
        "retrieval.confidence.score",
        {
            "reranker_score_count": len(reranker_scores),
            "retrieved_doc_count": len(retrieved_texts),
            "answer_len": len(answer or ""),
        },
    ) as _span:
        if weights is not None:
            w = weights
        else:
            # Build from kwargs — ConfidenceWeights.__post_init__ validates sum.
            w = ConfidenceWeights(
                retrieval=retrieval_weight,
                llm=llm_weight,
                citation=citation_weight,
            )

        retrieval_score = compute_retrieval_confidence(reranker_scores)
        llm_score = parse_llm_confidence(llm_confidence_text)
        citation_score = compute_citation_coverage(answer, retrieved_texts)

        composite = (
            w.retrieval * retrieval_score
            + w.llm * llm_score
            + w.citation * citation_score
        )
        composite = max(0.0, min(1.0, composite))
        _span.set_attribute("composite", float(composite))

        return ConfidenceBreakdown(
            retrieval_score=retrieval_score,
            llm_score=llm_score,
            citation_score=citation_score,
            composite=composite,
            retrieval_weight=w.retrieval,
            llm_weight=w.llm,
            citation_weight=w.citation,
        )


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
