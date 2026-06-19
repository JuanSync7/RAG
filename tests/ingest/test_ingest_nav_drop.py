# @summary
# Tests for ingest-time navigational-chunk dropping and the shared is_navigational
# predicate (incl. the widened front-matter phrase regex). Covers the predicate,
# the per-document over-prune guard, and symmetry with the query-side filter.
# @end-summary

"""Tests for the ingest-time ToC/front-matter drop (commit-3 fix F1-toc)."""

from types import SimpleNamespace

import pytest

from src.ingest.common.shared import is_navigational, toc_leader_ratio
from src.ingest.embedding.nodes.chunking import _drop_navigational_chunks


# ---------------------------------------------------------------------------
# Shared predicate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "A1.3 AXI Architecture .................... A1-22",          # ToC dotted leader
        "This specification describes the AMBA AXI protocol.",        # widened phrase
        "This document describes the register map.",                  # widened phrase
        "This appendix contains the signal list.",                    # widened phrase
        "Read this chapter for a description of the basic transactions.",
        "For a description of the coherency model, see Chapter 4.",
    ],
)
def test_is_navigational_true_cases(text):
    assert is_navigational(text, 320) is True


@pytest.mark.parametrize(
    "text",
    [
        "The AWVALID signal is asserted when the master drives a valid write "
        "address onto the bus and remains high until AWREADY is asserted.",
        "The reset value of the STATUS register is 0x00.",
        "AWVALID",  # thin but not navigational (handled by the length floor elsewhere)
    ],
)
def test_is_navigational_false_cases(text):
    assert is_navigational(text, 320) is False


def test_long_cross_reference_body_not_navigational():
    # A real body paragraph can contain a pointer-phrase ("for a description of")
    # yet still be substantive; the length cap (320) is what distinguishes a stub
    # pointer from a real paragraph, so anything well over the cap is NOT dropped.
    text = (
        "For a description of the burst types, the encoding table in this section "
        "lists AWBURST values: FIXED (0b00) for repeated accesses to the same "
        "address such as a peripheral FIFO, INCR (0b01) for incrementing bursts that "
        "advance by the transfer size, and WRAP (0b10) for cache-line accesses that "
        "wrap at the container boundary; the AWLEN field gives the burst length minus "
        "one, and the AWSIZE field encodes the number of bytes per transfer."
    )
    assert len(text) > 320  # well over the pointer-phrase length cap
    assert is_navigational(text, 320) is False


def test_toc_leader_ratio_dense_vs_prose():
    assert toc_leader_ratio("Foo ........ 12\nBar ........ 18") >= 0.08
    assert toc_leader_ratio("A normal sentence with no dotted leaders at all.") == 0.0


# ---------------------------------------------------------------------------
# _drop_navigational_chunks
# ---------------------------------------------------------------------------

def _c(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text, metadata={})


def test_drops_nav_keeps_body():
    body = _c("The reset value of REG37 is 0x25 and it is sticky across warm reset.")
    toc = _c("A1.3 AXI Architecture .................... A1-22")
    kept, n = _drop_navigational_chunks([body, toc], 320, source_name="doc")
    assert n == 1
    assert kept == [body]


def test_over_prune_guard_keeps_originals_when_all_navigational():
    # A pure-ToC document: every chunk looks navigational -> keep all (never nuke).
    chunks = [
        _c("A1.1 Intro .................... 1"),
        _c("A1.2 Scope .................... 2"),
    ]
    kept, n = _drop_navigational_chunks(chunks, 320, source_name="toc-doc")
    assert n == 0
    assert kept is chunks


def test_no_drop_when_no_navigational():
    chunks = [_c("Substantive body paragraph one with real content about signals."),
              _c("Substantive body paragraph two describing the handshake in detail.")]
    kept, n = _drop_navigational_chunks(chunks, 320, source_name="doc")
    assert n == 0
    assert kept is chunks
