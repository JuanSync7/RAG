# @summary
# Tests for lossless coverage verification: the pure coverage_report core
# (src.ingest.support.lossless) and the lossless_verification_node ingest stage.
# @end-summary
"""Lossless-verification tests.

Two layers:
  * TestCoverageReport — the deterministic word-coverage core. The headline
    property is BOUNDARY ROBUSTNESS: a document split into many non-overlapping
    contiguous chunks must score 1.0 (no chunk-boundary penalty), while a
    genuinely dropped section/tail must surface as a localized missing span.
  * TestLosslessVerificationNode — config-gated wiring: skip when disabled,
    warn below threshold by default, RAISE when strict, golden source
    selection, and breadcrumb-stripping of contextualized chunks.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.ingest.common.schemas import ProcessedChunk
from src.ingest.common.types import IngestionConfig, Runtime
from src.ingest.embedding.nodes.lossless_verification import lossless_verification_node
from src.ingest.support.lossless import coverage_report


# ---------------------------------------------------------------------------
# Pure core: coverage_report
# ---------------------------------------------------------------------------


class TestCoverageReport:
    def _golden(self, n: int) -> str:
        return " ".join(f"word{i}" for i in range(n))

    def test_identical_text_is_full_coverage(self):
        g = self._golden(40)
        r = coverage_report(g, [g])
        assert r.coverage == 1.0
        assert r.covered_words == r.golden_words == 40
        assert r.missing_spans == []

    def test_boundary_only_split_is_lossless(self):
        """THE key property: a doc split into many NON-overlapping contiguous
        chunks covers 100% — chunk boundaries must not be penalized."""
        toks = self._golden(60).split()
        chunks = [" ".join(toks[a : a + 15]) for a in range(0, 60, 15)]
        assert len(chunks) == 4
        r = coverage_report(g := " ".join(toks), chunks)
        assert r.coverage == 1.0, f"boundary penalty leaked: {r.coverage}"
        assert r.missing_spans == []

    def test_overlapping_chunks_still_full_coverage(self):
        toks = self._golden(50).split()
        # 5-token overlap between adjacent chunks
        chunks = [
            " ".join(toks[0:20]),
            " ".join(toks[15:35]),
            " ".join(toks[30:50]),
        ]
        r = coverage_report(" ".join(toks), chunks)
        assert r.coverage == 1.0
        assert r.missing_spans == []

    def test_dropped_section_is_localized_span(self):
        toks = self._golden(60).split()
        # keep [0:30) and [45:60); drop the middle [30:45)
        chunks = [" ".join(toks[0:30]), " ".join(toks[45:60])]
        r = coverage_report(" ".join(toks), chunks)
        assert r.coverage == pytest.approx(45 / 60)
        assert len(r.missing_spans) == 1
        span = r.missing_spans[0]
        assert span.start_word == 30
        assert span.length == 15
        assert span.snippet.startswith("word30")

    def test_dropped_tail_span_at_end(self):
        toks = self._golden(40).split()
        chunks = [" ".join(toks[0:30])]  # last 10 words dropped
        r = coverage_report(" ".join(toks), chunks)
        assert r.coverage == pytest.approx(30 / 40)
        assert r.missing_spans[0].start_word == 30
        assert r.missing_spans[0].length == 10

    def test_table_markdown_reconstruction_is_covered(self):
        """A markdown table in the golden, restated as a table_row block chunk
        (pipes/dashes/breadcrumb), is fully covered — formatting is ignored."""
        golden = (
            "Section Foo\n\n| Name | Val |\n| --- | --- |\n"
            "| alpha | 1 |\n| beta | 2 |\n| gamma | 3 |"
        )
        chunk = (
            "| Name | Val |\n| --- | --- |\n"
            "| alpha | 1 |\n| beta | 2 |\n| gamma | 3 |"
        )
        r = coverage_report(golden, [chunk, "Section Foo"])
        assert r.coverage == 1.0
        assert r.missing_spans == []

    def test_formatting_and_case_insensitive(self):
        golden = "The Quick, Brown Fox — jumps over the LAZY dog."
        chunk = "the quick brown fox jumps over the lazy dog"
        r = coverage_report(golden, [chunk])
        assert r.coverage == 1.0

    def test_empty_golden_is_full_coverage(self):
        r = coverage_report("", ["anything"])
        assert r.coverage == 1.0
        assert r.golden_words == 0

    def test_empty_chunks_zero_coverage(self):
        r = coverage_report(self._golden(20), [])
        assert r.coverage == 0.0
        assert r.missing_spans and r.missing_spans[0].length == 20

    def test_short_golden_covered_iff_present(self):
        # golden shorter than k(=5): covered only if it appears contiguously.
        assert coverage_report("alpha beta gamma", ["x alpha beta gamma y"]).coverage == 1.0
        assert coverage_report("alpha beta gamma", ["alpha zzz gamma"]).coverage == 0.0


# ---------------------------------------------------------------------------
# Node: lossless_verification_node
# ---------------------------------------------------------------------------


def _chunk(text: str, heading_path: list | None = None) -> ProcessedChunk:
    meta: dict = {}
    if heading_path is not None:
        meta["heading_path"] = heading_path
    return ProcessedChunk(text=text, metadata=meta)


def _state(
    *,
    chunks,
    parse_markdown: str | None = None,
    cleaned_text: str = "",
    raw_text: str = "",
    verify=True,
    min_cov=0.98,
    strict=False,
):
    config = IngestionConfig(
        verify_lossless=verify,
        lossless_min_coverage=min_cov,
        lossless_strict=strict,
    )
    runtime = Runtime(config=config, embedder=MagicMock(), weaviate_client=MagicMock())
    parse_result = (
        SimpleNamespace(text_markdown=parse_markdown) if parse_markdown is not None else None
    )
    return {
        "runtime": runtime,
        "chunks": chunks,
        "parse_result": parse_result,
        "cleaned_text": cleaned_text,
        "raw_text": raw_text,
        "source_name": "doc.pdf",
        "source_key": "doc.pdf",
        "processing_log": [],
        "errors": [],
    }


def _golden_text(n: int) -> str:
    return " ".join(f"word{i}" for i in range(n))


class TestLosslessVerificationNode:
    def test_disabled_skips_without_raising(self):
        state = _state(chunks=[_chunk("anything")], parse_markdown=_golden_text(40), verify=False)
        out = lossless_verification_node(state)
        assert any("skipped" in e for e in out["processing_log"])

    def test_no_golden_or_no_chunks_skips(self):
        out = lossless_verification_node(_state(chunks=[], parse_markdown=_golden_text(20)))
        assert any("skipped" in e for e in out["processing_log"])
        out2 = lossless_verification_node(_state(chunks=[_chunk("x")], parse_markdown=""))
        assert any("skipped" in e for e in out2["processing_log"])

    def test_full_coverage_logs_ok_no_raise(self):
        g = _golden_text(40)
        toks = g.split()
        chunks = [_chunk(" ".join(toks[0:20])), _chunk(" ".join(toks[20:40]))]
        out = lossless_verification_node(_state(chunks=chunks, parse_markdown=g))
        assert any("lossless_verification:ok" in e for e in out["processing_log"])

    def test_below_threshold_warns_not_strict(self):
        g = _golden_text(40)
        chunks = [_chunk(" ".join(g.split()[0:10]))]  # ~25% coverage
        out = lossless_verification_node(_state(chunks=chunks, parse_markdown=g, strict=False))
        assert any("below" in e for e in out["processing_log"])  # warned, did not raise

    def test_below_threshold_strict_raises(self):
        g = _golden_text(40)
        chunks = [_chunk(" ".join(g.split()[0:10]))]
        with pytest.raises(ValueError, match="Lossless verification failed"):
            lossless_verification_node(_state(chunks=chunks, parse_markdown=g, strict=True))

    def test_prefers_parse_markdown_over_cleaned_text(self):
        # parse golden is fully covered; cleaned_text is DIFFERENT content that
        # the chunks would NOT cover — if the node wrongly used cleaned_text it
        # would warn. It must use the parse and log ok.
        g = _golden_text(40)
        toks = g.split()
        chunks = [_chunk(" ".join(toks[0:20])), _chunk(" ".join(toks[20:40]))]
        out = lossless_verification_node(
            _state(chunks=chunks, parse_markdown=g, cleaned_text="totally different uncovered text here " * 5)
        )
        assert any("lossless_verification:ok" in e for e in out["processing_log"])

    def test_falls_back_to_cleaned_text_when_no_parse(self):
        g = _golden_text(40)
        toks = g.split()
        chunks = [_chunk(" ".join(toks[0:20])), _chunk(" ".join(toks[20:40]))]
        out = lossless_verification_node(
            _state(chunks=chunks, parse_markdown=None, cleaned_text=g)
        )
        assert any("lossless_verification:ok" in e for e in out["processing_log"])

    def test_breadcrumb_is_stripped_before_coverage(self):
        # The chunk text is contextualized (heading breadcrumb prepended). The
        # breadcrumb words ("Chapter One") are NOT in the golden body; stripping
        # them keeps coverage about the BODY and avoids treating the breadcrumb
        # as covered-or-not noise. Body fully covers the golden.
        body = _golden_text(30)
        toks = body.split()
        c1 = _chunk("Chapter One\n" + " ".join(toks[0:15]), heading_path=["Chapter One"])
        c2 = _chunk("Chapter One\n" + " ".join(toks[15:30]), heading_path=["Chapter One"])
        out = lossless_verification_node(_state(chunks=[c1, c2], parse_markdown=body))
        assert any("lossless_verification:ok" in e for e in out["processing_log"])
