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
        "A1.3 AXI Architecture . . . . . . . . . . . . A1-22",        # space-separated leader
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
    # Space-separated leaders (PDF extraction artifact) must score the same way —
    # the contiguous-only regex used to score these at 0.0 and let the page through.
    assert toc_leader_ratio("Foo . . . . . . . 12\nBar . . . . . . . 18") >= 0.08
    assert toc_leader_ratio("A normal sentence with no dotted leaders at all.") == 0.0
    # A lone prose ellipsis (3 dots) is not a leader and must not register.
    assert toc_leader_ratio("Well... it depends on the configuration.") == 0.0


def test_space_separated_toc_is_navigational():
    """The exact class that slipped through: a spec ToC linearised by the PDF
    parser into "<label> <title> . . . . <page>" with space-separated leaders.
    A *different* instance of the navigational class than the contiguous-dot
    fixtures above (per the project's generic-fix rule)."""
    linearised_toc = (
        "AMBA CHI Architecture Specification , 1 = B2.9.1 . , 2 = Data size "
        ". . . . . . . . . . . . . . . 162 , 1 = B2.10.3 . , 2 = Transaction "
        "Retry flow . . . . . . . . . . . . 181 , 1 = B11.3 Data . , 2 = Target "
        ". . . . . . . . . . . 415 , 1 = B12.3 . , 2 = Tag coherency "
        ". . . . . . . . . . . . . . . . 432 , 1 = B13.4 . , 2 = Channel mapping "
        ". . . . . . . . . . . 458 , 1 = B15.2 . , 2 = Request Node rules "
        ". . . . . . . . . . . . . . 503 , 1 = B16.1 . , 2 = Enhanced Features "
        ". . . . . . . . . . . 564"
    )
    assert len(linearised_toc) > 320  # long chunk; only the leader-density gate can catch it
    assert is_navigational(linearised_toc, 320) is True


def test_numeric_data_table_not_navigational():
    """False-positive guard: a real numeric data table is dense with integers but
    has NO dotted leaders, so it must be kept. (An earlier page-pointer-density
    heuristic wrongly flagged exactly this kind of table — keep the gate keyed on
    leaders, which real tables do not contain.)"""
    data_table = (
        "B2.8.4 Transaction attribute combinations Table B2.11: Legal combinations "
        "of MemAttr, SnpAttr, and Order. | MemAttr | SnpAttr | Order | Allowed | "
        "| Device | 0 | 00 | Yes | | Normal | 1 | 01 | Yes | | Device | 1 | 11 | No | "
        "| Normal | 0 | 10 | Yes |"
    )
    assert is_navigational(data_table, 320) is False


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
