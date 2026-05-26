# @summary
# Unit tests for the e2e sanity runbook helpers. Validates the assertion
# helpers used by ``scripts/e2e_sanity_table_ingest.py`` (expansion-fired
# detection, citation-bbox presence) against in-memory fake retrieval-hit
# and source-ref shapes. NO live Weaviate / TEI / LLM clients are used —
# the script's live-stack path is exercised separately by an operator
# following ``docs/ingest/E2E_SANITY_RUNBOOK.md``.
# Exports: TestAssertExpansionFired, TestAssertCitationHasBbox,
#          TestSanityReportRendering
# Deps: pytest, scripts.e2e_sanity_table_ingest
# @end-summary
"""Unit-test the runbook helpers in isolation from live infra.

The script under test (``scripts/e2e_sanity_table_ingest.py``) connects
to a live stack when invoked from the CLI, but its assertion helpers are
pure functions that operate on the standard retrieval-hit dict shape and
the ``_source_refs`` output. This test pins those helpers' contracts so
the runbook can be trusted even before a human runs it against compose.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the script as a module (it lives under scripts/ which is not a
# package). We load via importlib so the test does not depend on PYTHONPATH
# tweaks at collection time.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "e2e_sanity_table_ingest.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "e2e_sanity_table_ingest", _SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["e2e_sanity_table_ingest"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def script():
    return _load_script_module()


# ---------------------------------------------------------------------------
# _assert_expansion_fired
# ---------------------------------------------------------------------------


class TestAssertExpansionFired:
    """The helper must FAIL loudly when expansion did not fire and PASS
    quietly when at least one hit carries ``metadata.expanded_from``.
    """

    def test_passes_when_at_least_one_hit_has_expanded_from(self, script):
        hits = [
            {
                "text": "Vdd row",
                "metadata": {
                    "chunk_type": "table_row",
                    "table_group_id": "gid-1",
                    "chunk_id": "row-1",
                },
            },
            {
                "text": "Operating Conditions summary",
                "metadata": {
                    "chunk_type": "table_summary",
                    "table_group_id": "gid-1",
                    "chunk_id": "sum-1",
                    "expanded_from": "gid-1",  # <- the marker
                },
            },
        ]
        # Must not raise.
        script._assert_expansion_fired(hits)

    def test_fails_when_no_hit_carries_expanded_from(self, script):
        hits = [
            {
                "text": "Vdd row",
                "metadata": {
                    "chunk_type": "table_row",
                    "table_group_id": "gid-1",
                    "chunk_id": "row-1",
                },
            },
        ]
        with pytest.raises(AssertionError) as excinfo:
            script._assert_expansion_fired(hits)
        # Diagnostic: message should mention the marker name and the
        # number of hits inspected, so the failure is debuggable.
        msg = str(excinfo.value)
        assert "expanded_from" in msg
        assert "1" in msg  # hit count surfaced

    def test_fails_with_empty_hits(self, script):
        with pytest.raises(AssertionError):
            script._assert_expansion_fired([])

    def test_ignores_unrelated_metadata_fields(self, script):
        """The presence of arbitrary metadata keys must not pollute the
        decision — only ``expanded_from`` counts."""
        hits = [
            {"text": "a", "metadata": {"chunk_type": "text"}},
            {"text": "b", "metadata": {"chunk_type": "table_row",
                                        "expanded_from": "g"}},
        ]
        script._assert_expansion_fired(hits)


# ---------------------------------------------------------------------------
# _assert_citation_has_bbox
# ---------------------------------------------------------------------------


class TestAssertCitationHasBbox:
    """The helper must FAIL loudly when no source_ref carries both
    ``page_no`` and a decoded ``page_bbox`` dict. This is the proof
    that the DEFECT-2 JSON-decode fix is in effect post-merge.
    """

    def test_passes_with_well_formed_bbox(self, script):
        refs = [
            {
                "source": "esp32-s3.pdf",
                "page_no": 12,
                "page_bbox": {"l": 0.1, "t": 0.2, "r": 0.9, "b": 0.8},
            },
        ]
        script._assert_citation_has_bbox(refs)

    def test_fails_when_no_ref_has_bbox(self, script):
        refs = [
            {"source": "esp32-s3.pdf", "page_no": 12, "page_bbox": None},
            {"source": "esp32-s3.pdf", "page_no": 13},  # no bbox key at all
        ]
        with pytest.raises(AssertionError) as excinfo:
            script._assert_citation_has_bbox(refs)
        msg = str(excinfo.value)
        assert "page_bbox" in msg

    def test_fails_when_bbox_is_string_not_decoded(self, script):
        """A raw JSON string in ``page_bbox`` would mean the
        ``_decode_page_bbox`` path was bypassed — that's the regression
        we're guarding against."""
        refs = [
            {
                "source": "esp32-s3.pdf",
                "page_no": 12,
                "page_bbox": '[0.1, 0.2, 0.9, 0.8]',  # raw, undecoded
            },
        ]
        with pytest.raises(AssertionError) as excinfo:
            script._assert_citation_has_bbox(refs)
        assert "decoded" in str(excinfo.value).lower() or "dict" in str(excinfo.value).lower()

    def test_fails_when_bbox_missing_required_keys(self, script):
        refs = [
            {
                "source": "esp32-s3.pdf",
                "page_no": 12,
                "page_bbox": {"l": 0.1, "t": 0.2},  # missing r,b
            },
        ]
        with pytest.raises(AssertionError):
            script._assert_citation_has_bbox(refs)

    def test_fails_when_page_no_missing(self, script):
        refs = [
            {
                "source": "esp32-s3.pdf",
                "page_bbox": {"l": 0.1, "t": 0.2, "r": 0.9, "b": 0.8},
            },
        ]
        with pytest.raises(AssertionError) as excinfo:
            script._assert_citation_has_bbox(refs)
        assert "page_no" in str(excinfo.value)

    def test_empty_refs_fail(self, script):
        with pytest.raises(AssertionError):
            script._assert_citation_has_bbox([])


# ---------------------------------------------------------------------------
# Report rendering — proves the script can emit a structured PASS/FAIL
# block without needing the live stack.
# ---------------------------------------------------------------------------


class TestSanityReportRendering:
    """``_render_report`` must produce a deterministic markdown block
    summarising the run, including PASS/FAIL lines for each gate."""

    def test_pass_report_lists_all_gates_as_pass(self, script):
        report = script._render_report(
            pdf_path=Path("data/datasheets/esp32-s3_datasheet.pdf"),
            collection="e2e_sanity",
            query="What is the operating voltage range of the ESP32-S3?",
            results={
                "ingest_ok": True,
                "ingest_stored_chunks": 1234,
                "retrieval_ok": True,
                "retrieval_hits": 7,
                "expansion_ok": True,
                "expanded_hit_count": 2,
                "citation_ok": True,
                "citation_with_bbox_count": 3,
                "error": None,
            },
        )
        assert "PASS" in report
        assert "FAIL" not in report
        assert "expansion" in report.lower()
        assert "citation" in report.lower()

    def test_fail_report_surfaces_error_message(self, script):
        report = script._render_report(
            pdf_path=Path("data/datasheets/esp32-s3_datasheet.pdf"),
            collection="e2e_sanity",
            query="q",
            results={
                "ingest_ok": False,
                "ingest_stored_chunks": 0,
                "retrieval_ok": False,
                "retrieval_hits": 0,
                "expansion_ok": False,
                "expanded_hit_count": 0,
                "citation_ok": False,
                "citation_with_bbox_count": 0,
                "error": "weaviate connection refused at localhost:8080",
            },
        )
        assert "FAIL" in report
        assert "weaviate connection refused" in report


# ---------------------------------------------------------------------------
# Argument surface — pin the CLI flags so the runbook doc stays in sync.
# ---------------------------------------------------------------------------


class TestArgSurface:
    """Pin the CLI flag names the runbook documents."""

    def test_parser_accepts_documented_flags(self, script):
        parser = script._build_parser()
        ns = parser.parse_args([
            "--pdf", "data/datasheets/esp32-s3_datasheet.pdf",
            "--collection", "e2e_sanity",
            "--query", "What is the operating voltage range of the ESP32-S3?",
        ])
        assert str(ns.pdf).endswith("esp32-s3_datasheet.pdf")
        assert ns.collection == "e2e_sanity"
        assert "operating voltage" in ns.query

    def test_parser_has_sensible_defaults(self, script):
        parser = script._build_parser()
        ns = parser.parse_args([])
        # Defaults must be set so a bare invocation is meaningful.
        assert ns.pdf is not None
        assert ns.collection
        assert ns.query
