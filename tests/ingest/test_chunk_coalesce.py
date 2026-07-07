# @summary
# Regression tests for the markdown chunker's minimum-size coalescing pass and
# the de-backtracked numbered-heading regex. Guards against the semantic
# chunker fragmenting heterogeneous/table-dense content into a flood of tiny
# single-sentence chunks (one ARM CHI spec section exploded 1,488 -> 13,758
# chunks before this fix).
# Exports: (pytest test functions)
# Deps: pytest, src.ingest.support.markdown
# @end-summary
"""Unit tests for `_coalesce_segments`, `_semantic_split` floor, and the
numbered-heading regex backtracking fix."""

import time

import numpy as np
import pytest

from src.ingest.support.markdown import (
    _coalesce_segments,
    _semantic_split,
    _NUMBERED_HEADING_RE,
    normalize_headings_to_markdown,
)

MIN = 512
MAX = 3500


def test_coalesce_merges_tiny_segments():
    segs = ["a" * 40] * 100  # 100 tiny fragments
    out = _coalesce_segments(segs, min_chars=MIN, max_chars=MAX)
    # every emitted segment (except possibly the last) must be >= MIN
    assert all(len(s) >= MIN for s in out[:-1])
    assert len(out) < len(segs)  # actually merged
    # no segment exceeds MAX
    assert all(len(s) <= MAX for s in out)


def test_coalesce_respects_max():
    # large segments that are already big must not be merged past MAX
    segs = ["x" * 2000, "y" * 2000, "z" * 2000]
    out = _coalesce_segments(segs, min_chars=MIN, max_chars=MAX)
    assert out == segs  # each already >= MIN, merging would exceed MAX
    assert all(len(s) <= MAX for s in out)


def test_coalesce_trailing_tiny_folds_back():
    segs = ["x" * 600, "y" * 30]  # last is tiny
    out = _coalesce_segments(segs, min_chars=MIN, max_chars=MAX)
    assert len(out) == 1
    assert out[0].endswith("y" * 30)


def test_coalesce_single_tiny_segment_kept():
    # a lone tiny segment (only content) is still emitted
    out = _coalesce_segments(["tiny"], min_chars=MIN, max_chars=MAX)
    assert out == ["tiny"]


def test_coalesce_disabled_when_min_zero():
    segs = ["a", "b", "c"]
    assert _coalesce_segments(segs, min_chars=0, max_chars=MAX) == segs


class _OrthogonalEmbedder:
    """Worst case: every sentence embeds to a distinct orthogonal unit vector,
    so consecutive cosine == 0 < threshold -> the splitter would emit one chunk
    per sentence without the coalescing floor."""

    def encode_sentences(self, sentences):
        n = len(sentences)
        return np.eye(n, dtype=float)


def test_semantic_split_does_not_fragment_under_worst_case():
    # 200 short sentences -> without coalescing this returns ~200 tiny chunks
    sentences = [f"Field F{i} carries control information for the transaction. " for i in range(200)]
    text = "".join(sentences)
    out = _semantic_split(text, _OrthogonalEmbedder(), threshold=0.75, min_chars=MIN, max_chars=MAX)
    assert len(out) < 50, f"expected coalesced output, got {len(out)} fragments"
    assert all(len(s) >= MIN for s in out[:-1])
    assert all(len(s) <= MAX for s in out)


def test_semantic_split_fallback_path_coalesces():
    """When the embedder raises, the sentence fallback must also coalesce."""
    class _Boom:
        def encode_sentences(self, sentences):
            raise RuntimeError("embedder down")

    sentences = [f"Sentence number {i} about coherency. " for i in range(120)]
    out = _semantic_split("".join(sentences), _Boom(), min_chars=MIN, max_chars=MAX)
    assert len(out) < 40
    assert all(len(s) >= MIN for s in out[:-1])


def test_numbered_heading_regex_no_backtracking():
    """A long non-matching line must not cause catastrophic backtracking."""
    # a long line that 'almost' matches: number + many capitalized words + a
    # trailing char that breaks the anchor. The old (space-in-class) pattern
    # backtracked exponentially here.
    line = "1.2 " + "Word " * 400 + "!"
    t0 = time.perf_counter()
    m = _NUMBERED_HEADING_RE.search(line)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, f"regex took {elapsed:.2f}s (backtracking)"
    assert m is None  # trailing '!' breaks the $ anchor


def test_numbered_heading_still_matches_real_headings():
    assert normalize_headings_to_markdown("1. INTRODUCTION").startswith("## ")
    assert normalize_headings_to_markdown("2.1 Supervised Learning").startswith("### ")
    # a table row "3 ReadNoSnp" still converts (acceptable; the coalescer and
    # nav-filter handle downstream effects) — just assert it doesn't hang/crash
    assert isinstance(normalize_headings_to_markdown("3 ReadNoSnp"), str)
