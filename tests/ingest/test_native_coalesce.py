# @summary
# Tests for the native (Docling HybridChunker) min-body coalesce floor and the
# breadcrumb-recovery helper that underpins it.
# Covers: chunk_body_text breadcrumb stripping + no-op fallbacks; _shared_heading_prefix;
# _coalesce_native_chunks merge/skip/cap/re-attribution behaviour; idempotency.
# @end-summary

"""Tests for native-path chunk coalescing (Option A min-body floor).

The Docling HybridChunker's ``merge_peers`` pass only merges chunks that share
the SAME heading path and only up to the token budget — it has no minimum-size
floor, so leaf sections with tiny bodies survive as heading-dominated stubs.
``_coalesce_native_chunks`` folds those sub-floor bodies into a same-ancestor
neighbour. ``chunk_body_text`` recovers the raw body from the contextualized
(breadcrumb-prepended) embedded text so the floor measures the body, not the
heading padding.
"""

from types import SimpleNamespace

import pytest

from src.ingest.common.shared import chunk_body_text
from src.ingest.support.docling import (
    _shared_heading_prefix,
    _coalesce_native_chunks,
)


# ---------------------------------------------------------------------------
# chunk_body_text — recover body from HybridChunker-contextualized text
# ---------------------------------------------------------------------------

def test_chunk_body_text_strips_breadcrumb():
    hp = ["Overview", "3 Transaction Structure", "3.1 Address Phase"]
    ctx = "\n".join(hp) + "\n" + "See Section 3.2."
    assert chunk_body_text(ctx, hp) == "See Section 3.2."


def test_chunk_body_text_empty_heading_path_is_noop():
    text = "Overview\n3 Foo\nbody text here"
    assert chunk_body_text(text, []) == text
    assert chunk_body_text(text, None) == text


def test_chunk_body_text_prefix_mismatch_is_noop():
    # Table/figure chunk text does not begin with the breadcrumb -> unchanged.
    assert chunk_body_text("Columns: A | B\nRows: 5", ["Ch1", "1.1"]) == "Columns: A | B\nRows: 5"


def test_chunk_body_text_multiline_body_preserved():
    hp = ["A", "B"]
    body = "line one\nline two\nline three"
    ctx = "\n".join(hp) + "\n" + body
    assert chunk_body_text(ctx, hp) == body


# ---------------------------------------------------------------------------
# _shared_heading_prefix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "a,b,expected",
    [
        (["Ch3", "3.1"], ["Ch3", "3.2"], ["Ch3"]),       # siblings share chapter
        (["Ch3"], ["Ch4"], []),                            # different chapters
        (["Ch3", "3.1"], ["Ch3", "3.1"], ["Ch3", "3.1"]),  # identical -> full
        (["Ch3", "3.1"], ["Ch3"], ["Ch3"]),               # child vs parent
        ([], ["Ch3"], []),                                 # empty
    ],
)
def test_shared_heading_prefix(a, b, expected):
    assert _shared_heading_prefix(a, b) == expected


# ---------------------------------------------------------------------------
# _coalesce_native_chunks
# ---------------------------------------------------------------------------

def _mk(hp, body, page=None, extra=None):
    """Build a Chunk-like object as DoclingParser.chunk produces (contextualized)."""
    prefix = ("\n".join(hp) + "\n") if hp else ""
    return SimpleNamespace(
        text=prefix + body,
        heading_path=list(hp),
        section_path=" > ".join(hp),
        heading=hp[-1] if hp else "",
        heading_level=len(hp),
        extra_metadata=dict(extra or {}),
        chunk_index=0,
        page_ref=page,
    )


def test_coalesce_merges_sibling_stubs_under_shared_chapter():
    chunks = [_mk(["Ch3", "3.1"], "See 3.2."), _mk(["Ch3", "3.2"], "Tiny.")]
    out = _coalesce_native_chunks(chunks, min_chars=512, max_chars=3500)
    assert len(out) == 1
    # Re-attributed to the shared ancestor; both bodies present.
    assert out[0].heading_path == ["Ch3"]
    assert chunk_body_text(out[0].text, out[0].heading_path) == "See 3.2.\nTiny."


def test_coalesce_keeps_full_path_for_same_section_merge():
    chunks = [_mk(["Ch3", "3.1"], "a."), _mk(["Ch3", "3.1"], "b.")]
    out = _coalesce_native_chunks(chunks, min_chars=512, max_chars=3500)
    assert len(out) == 1
    assert out[0].heading_path == ["Ch3", "3.1"]  # no specificity lost
    assert chunk_body_text(out[0].text, out[0].heading_path) == "a.\nb."


def test_coalesce_does_not_merge_across_different_chapters():
    big = "X" * 600  # >= floor so it would not itself need merging
    chunks = [_mk(["Ch3"], "stub."), _mk(["Ch4"], big)]
    out = _coalesce_native_chunks(chunks, min_chars=512, max_chars=3500)
    assert len(out) == 2  # no shared ancestor -> left for the DROP backstop


def test_coalesce_respects_max_chars_cap():
    # stub + near-cap host: merged body would exceed cap -> not merged.
    chunks = [_mk(["Ch3", "3.1"], "stub."), _mk(["Ch3", "3.2"], "Y" * 3500)]
    out = _coalesce_native_chunks(chunks, min_chars=512, max_chars=3500)
    assert len(out) == 2


def test_coalesce_leaves_full_size_chunks_untouched():
    a = _mk(["Ch3", "3.1"], "A" * 600)
    b = _mk(["Ch3", "3.2"], "B" * 600)
    out = _coalesce_native_chunks([a, b], min_chars=512, max_chars=3500)
    assert len(out) == 2
    assert out[0] is a and out[1] is b


def test_coalesce_disabled_when_min_chars_zero():
    chunks = [_mk(["Ch3", "3.1"], "tiny"), _mk(["Ch3", "3.2"], "also tiny")]
    out = _coalesce_native_chunks(chunks, min_chars=0, max_chars=3500)
    assert out is chunks  # disabled -> returned unchanged


def test_coalesce_chains_multiple_tiny_siblings():
    chunks = [
        _mk(["Ch3", "3.1"], "one."),
        _mk(["Ch3", "3.2"], "two."),
        _mk(["Ch3", "3.3"], "three."),
    ]
    out = _coalesce_native_chunks(chunks, min_chars=512, max_chars=3500)
    assert len(out) == 1
    assert out[0].heading_path == ["Ch3"]
    assert chunk_body_text(out[0].text, out[0].heading_path) == "one.\ntwo.\nthree."


def test_coalesce_re_stamps_xref_from_merged_body():
    # The merged body mentions a section ref; xref edges should be re-stamped.
    chunks = [_mk(["Ch3", "3.1"], "stub."), _mk(["Ch3", "3.2"], "See Section 3.5 for details.")]
    out = _coalesce_native_chunks(chunks, min_chars=512, max_chars=3500)
    assert len(out) == 1
    # xref_targets is JSON-encoded list[{type,value}]; a section ref must appear.
    assert "xref_targets" in out[0].extra_metadata
    assert "3.5" in out[0].extra_metadata["xref_targets"]


def test_coalesce_is_idempotent():
    chunks = [_mk(["Ch3", "3.1"], "x."), _mk(["Ch3", "3.2"], "y.")]
    once = _coalesce_native_chunks(chunks, min_chars=512, max_chars=3500)
    twice = _coalesce_native_chunks(list(once), min_chars=512, max_chars=3500)
    assert len(once) == len(twice) == 1
    assert chunk_body_text(twice[0].text, twice[0].heading_path) == "x.\ny."


def test_coalesce_preserves_figures_metadata():
    a = _mk(["Ch3", "3.1"], "stub.", extra={"figures": [{"label": "Figure 1"}]})
    b = _mk(["Ch3", "3.2"], "more.", extra={"figures": [{"label": "Figure 2"}]})
    out = _coalesce_native_chunks([a, b], min_chars=512, max_chars=3500)
    assert len(out) == 1
    labels = [f["label"] for f in out[0].extra_metadata.get("figures", [])]
    assert labels == ["Figure 1", "Figure 2"]
