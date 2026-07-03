# @summary
# Regression tests for the query-time THIN/heading-only candidate floor in
# rag_chain (`_is_thin_or_heading` / `_filter_thin_candidates`). Guards that tiny
# title-only / bare-heading / ToC-noise stubs — which win dense similarity on
# topical queries but carry no answer — do not consume rerank slots ahead of real
# body chunks, while keeping at least `floor` items.
#
# NOTE: navigational/boilerplate suppression (ToC dotted-leaders, front-matter
# pointers) moved to the ingest-time `chunk_role` metadata + query-time role
# filter (RAG_RETRIEVAL_ROLE_SCHEMA_PRESENT). The former query-side
# `is_navigational` regex heuristic was retired from rag_chain; its behaviour is
# now covered by tests/ingest/test_ingest_nav_drop.py (shared predicate) and by
# the live role-filter validation. This file covers ONLY the retained thin floor.
# Exports: (pytest test functions)
# Deps: pytest, src.retrieval.pipeline.rag_chain
# @end-summary
"""Unit tests for the retained thin/heading-only floor (`_is_thin_or_heading` /
`_filter_thin_candidates`)."""

from types import SimpleNamespace

import pytest

from src.retrieval.pipeline.rag_chain import (
    _filter_thin_candidates,
    _is_thin_or_heading,
)

MIN_CHARS = 40

# Thin / heading-only / ToC-noise stubs that MUST be dropped by the floor.
THIN = [
    "## AXI",                                   # below min_chars
    "## AMBA CHI Architecture Specification",   # heading-only single line
    "## Chapter A3 Single Interface Requirements And Signalling Overview",  # long, still heading-only
    "..................................... | 234 | .....................",  # pure ToC noise
    "   ",                                       # whitespace-only
]

# Real answer chunks (heading + body) that MUST be kept.
REAL_CONTENT = [
    "## A1.3 AXI Architecture  \nThe AXI protocol is burst-based and defines the "
    "following independent transaction channels: read address, read data, write "
    "address, write data, write response.",
    "## A10.2.1 Read/write interface  \nA read write interface includes the AXI "
    "channels: AR read address, R read data, AW write address, W write data, B "
    "write response channel.",
    "The AWQOS signal carries the QoS identifier for the write address channel. "
    "The default value of 0b0000 indicates the transaction is not part of any QoS "
    "scheme, and higher values indicate higher priority.",
]


@pytest.mark.parametrize("text", THIN)
def test_thin_or_heading_only_is_dropped(text):
    assert _is_thin_or_heading(text, MIN_CHARS) is True


@pytest.mark.parametrize("text", REAL_CONTENT)
def test_real_content_is_kept(text):
    assert _is_thin_or_heading(text, MIN_CHARS) is False


def test_filter_preserves_floor():
    """Even if every candidate is thin, the floor is honored (topped up in order)."""
    items = [SimpleNamespace(text=t) for t in THIN]
    floor = 3
    kept = _filter_thin_candidates(items, min_chars=MIN_CHARS, floor=floor)
    assert len(kept) == floor  # never below floor, topped up from dropped


def test_filter_drops_thin_keeps_real():
    items = [SimpleNamespace(text=t) for t in THIN + REAL_CONTENT]
    kept = _filter_thin_candidates(items, min_chars=MIN_CHARS, floor=2)
    kept_texts = {it.text for it in kept}
    for t in REAL_CONTENT:
        assert t in kept_texts


def test_filter_noop_when_min_chars_zero():
    items = [SimpleNamespace(text=t) for t in THIN]
    kept = _filter_thin_candidates(items, min_chars=0, floor=1)
    assert kept is items  # disabled -> returns input unchanged
