# @summary
# Regression tests for the navigational/thin-chunk candidate filter in
# rag_chain. Guards the heuristics that drop table-of-contents leaders and
# chapter/part front-matter stubs (which beat real body chunks on both dense
# similarity and the cross-encoder reranker) while keeping real answer content.
# Cases are drawn from the live ARM AMBA-spec sanity corpus.
# Exports: (pytest test functions)
# Deps: pytest, src.retrieval.pipeline.rag_chain
# @end-summary
"""Unit tests for `_is_navigational` / `_is_low_value_chunk` / `_filter_thin_candidates`."""

from types import SimpleNamespace

import pytest

from src.retrieval.pipeline.rag_chain import (
    _filter_thin_candidates,
    _is_low_value_chunk,
    _is_navigational,
    _is_thin_or_heading,
)

NAV_MAX = 320
MIN_CHARS = 40

# Real navigational chunks from the corpus that MUST be dropped.
NAV_TOC = [
    "| C6-234 | | | C6.2 | Sequencing transactions ......................................................... C6-235",
    "xvii | | | AXI3 and AXI4 Protocol Specification Introduction | | Chapter A1 | A1.1 About the AXI protocol ........... A1-21",
    "A2-30 | | | A2.2 Write address channel signals ............................................. A2-31",
    "| C6-248 |   | Chapter C7 | Cache Maintenance | Cache Maintenance | | |----|----| C7.1 | About cache maintenance ......... C7-249",
]
# Stored multi-line format: heading on its own line, body on following lines
# (this is how Docling/contextual-chunking actually persists chunks — see the
# live corpus). The pointer phrase lives in the body line, so these are caught
# by the navigational phrase rule, NOT the thin/heading rule.
NAV_FRONTMATTER = [
    "## Chapter A3 Single Interface Requirements  \nRead this for a description of the basic AXI protocol transaction requirements between a master and slave.",
    "## Chapter A2 Signal Descriptions  \nRead this for a description of the signals that are used by the AXI3 and AXI4 protocols.",
    "## Chapter C4 Coherency Transactions on the Read Address and Write Address Channels  \nThis chapter describes the transactions issued on the read address and write address channels.",
    "## Preface  \nThis preface introduces the AMBA AXI and ACE Protocol Specification . It contains the following sections:  \n- About this specification",
    "## Chapter 4 Signal Descriptions  \nRead this chapter for descriptions of the APB signals.",
]

# Real *answer* chunks from the corpus that MUST be kept (verbatim multi-line).
REAL_CONTENT = [
    "## A1.3 AXI Architecture  \nThe AXI protocol is burst-based and defines the following independent transaction channels:  \n- read address\n- read data\n- write address\n- write data\n- write response.  \nAn address channel carries control information that describes the nature of the data to be transferred.",
    "# AMBA AXI  \n- Interface divided into 5 channels:\n- Write Address\n- Write Data\n- Write Response\n- Read Address\n- Read Data/Response\n- Each channel is independent and uses two-way flow control",
    "## A10.2.1 Read/write interface  \nA read write interface includes the following AXI channels:  \nAR Read address channel.  \nR Read data channel.  \nAW Write address channel.  \nW Write data channel.  \nB Write response channel.",
    "## A2.2 Write address channel signals  \nTable A2-2 shows the AXI write address channel signals. AWID identifies the write address group of signals. AWADDR is the write address.",
    "## Channel handshake signals  \nEach channel has its own VALID / READY handshake signal pair. Table A3-1 shows the signals for each channel and the source of each signal in the handshake.",
]


@pytest.mark.parametrize("text", NAV_TOC)
def test_toc_leaders_are_navigational(text):
    assert _is_navigational(text, NAV_MAX) is True
    assert _is_low_value_chunk(text, MIN_CHARS, True, NAV_MAX) is True


@pytest.mark.parametrize("text", NAV_FRONTMATTER)
def test_frontmatter_stubs_are_navigational(text):
    assert _is_navigational(text, NAV_MAX) is True


@pytest.mark.parametrize("text", REAL_CONTENT)
def test_real_content_is_kept(text):
    assert _is_navigational(text, NAV_MAX) is False
    assert _is_low_value_chunk(text, MIN_CHARS, True, NAV_MAX) is False


def test_long_chunk_mentioning_a_crossref_is_kept():
    """A real body chunk that merely cross-references a section must NOT be
    dropped — only short pointer stubs are navigational."""
    text = (
        "## A1.3 AXI Architecture   The AXI protocol is burst-based. For a description of "
        "the additional signaling see Chapter A8. The five channels each carry an "
        "independent stream of transactions and use a VALID/READY handshake. The write "
        "address channel carries AWADDR/AWID; the write data channel carries WDATA/WSTRB; "
        "the write response channel carries BRESP; the read address channel carries "
        "ARADDR; the read data channel carries RDATA and RRESP. " * 2
    )
    assert len(text) > NAV_MAX
    assert _is_navigational(text, NAV_MAX) is False


def test_phrase_rule_respects_length_cap():
    """The 'read this for' phrase only flags SHORT chunks; the same phrase inside
    a long content chunk does not trigger a drop."""
    short = "## Chapter A8   Read this for a description of additional signaling."
    assert _is_navigational(short, NAV_MAX) is True
    long_body = short + " " + ("The AWQOS signal carries the QoS identifier. " * 12)
    assert len(long_body) > NAV_MAX
    assert _is_navigational(long_body, NAV_MAX) is False


def test_drop_nav_toggle_off_keeps_navigational():
    """With drop_nav=False the navigational pass is skipped (back-compat)."""
    text = NAV_FRONTMATTER[0]
    assert _is_low_value_chunk(text, MIN_CHARS, False, NAV_MAX) is False


def test_filter_preserves_floor():
    """Even if every candidate is navigational, the floor is honored."""
    items = [SimpleNamespace(text=t) for t in NAV_TOC + NAV_FRONTMATTER]
    floor = 5
    kept = _filter_thin_candidates(
        items, min_chars=MIN_CHARS, floor=floor, drop_nav=True, nav_max_chars=NAV_MAX
    )
    assert len(kept) == floor  # topped up from dropped, never below floor


def test_filter_drops_nav_keeps_real():
    items = [SimpleNamespace(text=t) for t in NAV_TOC + REAL_CONTENT]
    kept = _filter_thin_candidates(
        items, min_chars=MIN_CHARS, floor=2, drop_nav=True, nav_max_chars=NAV_MAX
    )
    kept_texts = {it.text for it in kept}
    for t in REAL_CONTENT:
        assert t in kept_texts
    for t in NAV_TOC:
        assert t not in kept_texts


def test_filter_noop_when_disabled():
    items = [SimpleNamespace(text=t) for t in NAV_TOC]
    kept = _filter_thin_candidates(
        items, min_chars=0, floor=1, drop_nav=False, nav_max_chars=NAV_MAX
    )
    assert kept is items  # fully disabled -> returns input unchanged
