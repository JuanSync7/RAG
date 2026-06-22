# @summary
# Deterministic token-coverage check: do a set of chunks losslessly cover the
# document they were derived from? Pure + side-effect free so it is cheap to
# unit-test; the ingest node lossless_verification wraps it.
# Exports: CoverageResult, MissingSpan, coverage_report, tokenize, shingles
# Deps: (stdlib only)
# @end-summary
"""Lossless coverage verification.

The verifier answers one question per document: **do the emitted chunks, taken
together, contain every piece of content in the golden source text?** The
golden is the parsed Docling markdown the chunks were derived from (see the
``lossless_verification`` node); a coverage shortfall means chunking dropped
content somewhere (a skipped section, lost table rows, a truncated body).

Why content tokens, not byte-equality:

The chunker is not byte-preserving — it prepends heading breadcrumbs, restates
table headers per block, and the markdown export normalizes whitespace/quotes.
Requiring byte-identity would flag every such formatting difference as "loss".
Instead we compare on *content tokens*: ``\\w+`` words, lowercased (punctuation,
markdown pipes and separators dropped), so two renderings of the same content
tokenize identically.

Why anchor on k-grams instead of a shingle-set ratio:

A naive "fraction of golden k-grams that appear in some chunk" punishes every
chunk boundary — a k-gram straddling two non-overlapping chunks appears in
neither, so a cleanly-split lossless document would score (chunks-1)·(k-1)
"missing" k-grams and fail. Instead we use each golden k-gram a chunk matches
as an ANCHOR and mark the golden WORD window [p, p+k) it covers. Every golden
token inside any chunk's matched run is therefore covered — including the token
at a chunk boundary, which is covered by the within-chunk k-gram that ends on
it. A token stays uncovered only when its surrounding k-gram appears in NO
chunk, i.e. the content was genuinely dropped or reordered. Coverage is then the
fraction of golden TOKENS covered, and the uncovered tokens collapse into
contiguous missing spans (a dropped tail or skipped section shows as one big
span, not a scatter).

Everything here is O(n) in the token count and fully deterministic (same input
⇒ same output), so it is safe to run on every document and inside tests.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["CoverageResult", "MissingSpan", "coverage_report", "tokenize", "shingles"]

# Default k for the anchor k-gram. 5 words is long enough that incidental
# cross-chunk word collisions are rare, short enough that a single dropped
# sentence still leaves uncovered tokens.
DEFAULT_SHINGLE_SIZE = 5

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercased ``\\w+`` word tokens. Drops punctuation, markdown pipes/dashes,
    and whitespace so two renderings of the same content tokenize identically."""
    return _WORD_RE.findall((text or "").lower())


def shingles(tokens: list[str], k: int) -> list[tuple]:
    """Overlapping k-grams of ``tokens``. A token list shorter than ``k`` yields
    a single shingle of the whole list; an empty list yields none. (Retained as
    a small reusable primitive; coverage_report indexes k-grams directly.)"""
    if not tokens:
        return []
    if len(tokens) <= k:
        return [tuple(tokens)]
    return [tuple(tokens[i : i + k]) for i in range(len(tokens) - k + 1)]


def _contains_sublist(haystack: list[str], needle: tuple) -> bool:
    """True if ``needle`` occurs as a CONTIGUOUS run in ``haystack``."""
    if not needle:
        return True
    m = len(needle)
    if m > len(haystack):
        return False
    first = needle[0]
    for i in range(len(haystack) - m + 1):
        if haystack[i] == first and tuple(haystack[i : i + m]) == needle:
            return True
    return False


@dataclass
class MissingSpan:
    """A contiguous run of golden content not covered by any chunk.

    ``start_word``/``length`` index the golden WORD-token stream;
    ``snippet`` is the leading words of the span for human triage.
    """

    start_word: int
    length: int
    snippet: str


@dataclass
class CoverageResult:
    coverage: float
    golden_words: int
    covered_words: int
    missing_spans: list[MissingSpan] = field(default_factory=list)

    def is_lossless(self, threshold: float) -> bool:
        return self.coverage >= threshold


def coverage_report(
    golden_text: str,
    chunk_texts: list[str],
    *,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
    max_missing_spans: int = 10,
    snippet_words: int = 12,
) -> CoverageResult:
    """Fraction of ``golden_text`` content tokens covered by ``chunk_texts``.

    Args:
        golden_text: The reference (source-of-truth) document text.
        chunk_texts: The chunk BODIES to test (breadcrumbs already stripped).
        shingle_size: k for the anchor k-gram.
        max_missing_spans: cap on returned missing spans (largest first).
        snippet_words: words of context to include in each span snippet.

    Returns:
        A :class:`CoverageResult`. An empty golden ⇒ coverage 1.0 (nothing to
        cover); empty chunks against a non-empty golden ⇒ coverage 0.0.
    """
    k = max(1, int(shingle_size))
    g = tokenize(golden_text)
    n = len(g)
    if n == 0:
        return CoverageResult(1.0, 0, 0, [])

    # delta-array so each anchor marks its window [p, p+k) in O(1).
    delta = [0] * (n + 1)

    if n < k:
        # Golden too short for a k-gram: covered iff it appears verbatim
        # (contiguous) in some chunk.
        gt = tuple(g)
        if any(_contains_sublist(tokenize(ct), gt) for ct in chunk_texts):
            delta[0] += 1
            delta[n] -= 1
    else:
        index: dict[tuple, list[int]] = {}
        for p in range(n - k + 1):
            index.setdefault(tuple(g[p : p + k]), []).append(p)
        for ct in chunk_texts:
            c = tokenize(ct)
            lc = len(c)
            if lc >= k:
                for i in range(lc - k + 1):
                    positions = index.get(tuple(c[i : i + k]))
                    if positions:
                        for p in positions:
                            delta[p] += 1
                            delta[p + k] -= 1
            elif lc > 0:
                # Chunk shorter than k (e.g. a tiny caption): mark wherever its
                # tokens occur as a contiguous run in golden.
                needle = tuple(c)
                for p in range(n - lc + 1):
                    if g[p] == needle[0] and tuple(g[p : p + lc]) == needle:
                        delta[p] += 1
                        delta[p + lc] -= 1

    covered = bytearray(n)
    run = 0
    for i in range(n):
        run += delta[i]
        if run > 0:
            covered[i] = 1
    covered_count = sum(covered)
    coverage = covered_count / n

    # Collapse runs of uncovered tokens into spans (largest first).
    spans: list[MissingSpan] = []
    i = 0
    while i < n:
        if covered[i]:
            i += 1
            continue
        j = i
        while j < n and not covered[j]:
            j += 1
        snippet = " ".join(g[i : i + snippet_words])
        spans.append(MissingSpan(i, j - i, snippet))
        i = j
    spans.sort(key=lambda s: s.length, reverse=True)

    return CoverageResult(
        coverage=coverage,
        golden_words=n,
        covered_words=covered_count,
        missing_spans=spans[:max_missing_spans],
    )
