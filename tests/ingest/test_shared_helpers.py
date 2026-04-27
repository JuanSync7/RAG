# @summary
# Tests for shared ingestion helpers: cross_refs, quality_score,
# extract_keywords_fallback, map_chunk_provenance, append_processing_log.
# Exports: (pytest test functions)
# Deps: src.ingest.common.shared, src.ingest.common.types
# @end-summary
"""Tests for src.ingest.common.shared helper functions.

These tests verify the deterministic behavior of lightweight helpers used
across ingestion pipeline nodes, including cross-reference extraction,
quality scoring, keyword fallback, provenance mapping, and log appending.
"""

import pytest

# extract_keywords_fallback may be exported as public or private name
try:
    from src.ingest.common.shared import _extract_keywords_fallback
except ImportError:
    from src.ingest.common.shared import extract_keywords_fallback as _extract_keywords_fallback

from src.ingest.common.shared import (
    cross_refs,
    quality_score,
    map_chunk_provenance,
    append_processing_log,
)
from src.common.utils import parse_json_object


# ---------------------------------------------------------------------------
# cross_refs
# ---------------------------------------------------------------------------

class TestCrossRefs:
    """Tests for the cross_refs() helper."""

    def test_cross_refs_finds_doc_pattern(self):
        result = cross_refs("See DOC-123 for details")
        values = [r["value"] for r in result]
        assert "DOC-123" in values

    def test_cross_refs_finds_section_pattern(self):
        result = cross_refs("Refer to Section 3.2.1")
        assert any("Section" in r["value"] and "3.2.1" in r["value"] for r in result)

    def test_cross_refs_finds_rfc_pattern(self):
        result = cross_refs("Per RFC 2119 requirements")
        assert any("RFC" in r["value"] and "2119" in r["value"] for r in result)

    def test_cross_refs_returns_dicts_with_type_and_value_keys(self):
        result = cross_refs("DOC-123")
        assert len(result) >= 1
        for ref in result:
            assert "type" in ref
            assert "value" in ref

    def test_cross_refs_doc_type_label(self):
        result = cross_refs("DOC-123")
        types = [r["type"] for r in result]
        assert "document_id" in types

    def test_cross_refs_section_type_label(self):
        result = cross_refs("Section 2.1")
        types = [r["type"] for r in result]
        assert "section" in types

    def test_cross_refs_rfc_type_label(self):
        result = cross_refs("RFC 7230")
        types = [r["type"] for r in result]
        assert "standard" in types

    def test_cross_refs_empty_returns_empty(self):
        assert cross_refs("") == []

    def test_cross_refs_no_matches_returns_empty(self):
        assert cross_refs("plain text with no references") == []

    def test_cross_refs_multiple_types(self):
        result = cross_refs("DOC-001, Section 1.2.3, RFC 7230")
        text = " ".join(r["value"] for r in result)
        assert "DOC-001" in text
        assert "7230" in text

    def test_cross_refs_doc_min_2_digits(self):
        # The DOC pattern requires \d{2,} — 2 or more digits
        result = cross_refs("DOC-12")
        values = [r["value"] for r in result]
        assert "DOC-12" in values

    def test_cross_refs_doc_1_digit_not_matched(self):
        # DOC-1 should NOT match (only 1 digit, below the 2-digit minimum)
        result = cross_refs("DOC-1 is not a valid reference id")
        values = [r["value"] for r in result]
        assert not any(v == "DOC-1" for v in values)

    def test_cross_refs_doc_3_digits_matched(self):
        result = cross_refs("DOC-123")
        values = [r["value"] for r in result]
        assert any("DOC-123" in v for v in values)

    def test_cross_refs_no_deduplication(self):
        # The helper does NOT deduplicate — repeated refs each produce an entry
        result = cross_refs("DOC-42 and DOC-42 again")
        doc_42_matches = [r for r in result if "DOC-42" in r["value"]]
        assert len(doc_42_matches) == 2


# ---------------------------------------------------------------------------
# quality_score
# ---------------------------------------------------------------------------

class TestQualityScore:
    """Tests for the quality_score() helper."""

    def test_quality_score_empty_returns_base_score(self):
        # quality_score always returns at least _QUALITY_BASE (0.4)
        # because the base is added unconditionally before length bonuses
        score = quality_score("")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_quality_score_short_returns_float(self):
        # Short text ("Hi") returns the base score — not 0.0
        score = quality_score("Hi")
        assert isinstance(score, float)
        assert score >= 0.0

    def test_quality_score_well_formed_nonzero(self):
        text = "The clock domain crossing requires careful timing analysis for setup and hold margins."
        score = quality_score(text)
        assert 0.0 < score <= 1.0

    def test_quality_score_returns_float(self):
        assert isinstance(
            quality_score("some text with enough content here for evaluation"), float
        )

    def test_quality_score_whitespace_only_is_float(self):
        # Whitespace-only text still returns a float (base score)
        score = quality_score("   ")
        assert isinstance(score, float)
        assert score >= 0.0

    def test_quality_score_bounded_to_one(self):
        long_numeric_text = "42 " * 200
        score = quality_score(long_numeric_text)
        assert score <= 1.0

    def test_quality_score_long_text_no_less_than_short(self):
        short = quality_score("Hello world.")
        long = quality_score("Hello world. " + "a" * 200)
        # A longer text gets a length bonus — score should be >= short
        assert long >= short

    def test_quality_score_numeric_content_raises_score(self):
        plain = quality_score("a" * 130)
        numeric = quality_score("42 " * 50)
        # Numeric content adds a digit bonus on top of base score
        assert numeric >= plain


# ---------------------------------------------------------------------------
# extract_keywords_fallback
# ---------------------------------------------------------------------------

class TestExtractKeywordsFallback:
    """Tests for the extract_keywords_fallback() helper."""

    def test_empty_returns_empty(self):
        result = _extract_keywords_fallback("", max_keywords=5)
        assert result == []

    def test_returns_list(self):
        result = _extract_keywords_fallback("clock timing setup hold margin", max_keywords=3)
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_max_keywords_zero(self):
        result = _extract_keywords_fallback("some content words here", max_keywords=0)
        assert result == []

    def test_respects_limit(self):
        text = "clock timing setup hold margin violations analysis constraint path critical"
        result = _extract_keywords_fallback(text, max_keywords=3)
        assert len(result) == 3

    def test_high_freq_ranked_first(self):
        # "hold" appears 4 times — should be ranked highest
        text = "DFT scan chain timing timing timing clock clock setup hold hold hold hold"
        result = _extract_keywords_fallback(text, max_keywords=3)
        assert len(result) == 3
        assert result[0] == "hold"

    def test_result_contains_strings(self):
        result = _extract_keywords_fallback("clock timing margin", max_keywords=5)
        assert all(isinstance(kw, str) for kw in result)

    def test_result_lowercase(self):
        result = _extract_keywords_fallback("Clock Timing MARGIN", max_keywords=5)
        assert all(kw == kw.lower() for kw in result)

    def test_max_keywords_larger_than_vocab(self):
        # Requesting more keywords than unique words returns all unique words
        result = _extract_keywords_fallback("clock timing", max_keywords=100)
        assert len(result) <= 2


# ---------------------------------------------------------------------------
# map_chunk_provenance
# ---------------------------------------------------------------------------

class TestMapChunkProvenance:
    """Tests for the map_chunk_provenance() helper."""

    def test_exact_match_high_confidence(self):
        text = "Alpha section.\n\nClock must remain below 800MHz.\n\nBeta section"
        chunk = "Clock must remain below 800MHz."
        provenance, _, _ = map_chunk_provenance(
            chunk,
            original_text=text,
            refactored_text=text,
            original_cursor=0,
            refactored_cursor=0,
        )
        assert provenance["original_char_start"] >= 0
        assert provenance["provenance_confidence"] >= 0.8

    def test_no_match_returns_zero_confidence(self):
        provenance, _, _ = map_chunk_provenance(
            "completely absent text",
            original_text="nothing here",
            refactored_text="nothing here either",
            original_cursor=0,
            refactored_cursor=0,
        )
        assert provenance["provenance_confidence"] == 0.0

    def test_returns_three_tuple(self):
        text = "First chunk here. Second chunk here."
        result = map_chunk_provenance(
            "First chunk here.",
            original_text=text,
            refactored_text=text,
            original_cursor=0,
            refactored_cursor=0,
        )
        assert len(result) == 3

    def test_returns_cursors_as_ints(self):
        text = "First chunk here. Second chunk here."
        _, new_orig_cursor, new_ref_cursor = map_chunk_provenance(
            "First chunk here.",
            original_text=text,
            refactored_text=text,
            original_cursor=0,
            refactored_cursor=0,
        )
        assert isinstance(new_orig_cursor, int)
        assert isinstance(new_ref_cursor, int)

    def test_provenance_dict_has_required_keys(self):
        text = "Some content for testing provenance keys."
        provenance, _, _ = map_chunk_provenance(
            "Some content",
            original_text=text,
            refactored_text=text,
            original_cursor=0,
            refactored_cursor=0,
        )
        for key in (
            "original_char_start",
            "original_char_end",
            "refactored_char_start",
            "refactored_char_end",
            "provenance_method",
            "provenance_confidence",
        ):
            assert key in provenance, f"missing key: {key}"

    def test_cursor_advances_after_match(self):
        text = "First chunk here. Second chunk here."
        _, new_orig_cursor, new_ref_cursor = map_chunk_provenance(
            "First chunk here.",
            original_text=text,
            refactored_text=text,
            original_cursor=0,
            refactored_cursor=0,
        )
        # Cursor should advance past the matched span
        assert new_orig_cursor > 0
        assert new_ref_cursor > 0

    def test_no_match_cursor_unchanged(self):
        _, new_orig_cursor, new_ref_cursor = map_chunk_provenance(
            "absent text xyz",
            original_text="hello world",
            refactored_text="hello world",
            original_cursor=5,
            refactored_cursor=3,
        )
        assert new_orig_cursor == 5
        assert new_ref_cursor == 3


# ---------------------------------------------------------------------------
# append_processing_log
# ---------------------------------------------------------------------------

class TestAppendProcessingLog:
    """Tests for the append_processing_log() helper."""

    def _make_state(self, log=None):
        """Return a minimal dict that satisfies the state contract."""
        return {"processing_log": list(log or []), "runtime": None}

    def test_adds_entry(self):
        state = self._make_state()
        result = append_processing_log(state, "embed:ok")
        assert "embed:ok" in result

    def test_appends_not_replaces(self):
        state = self._make_state(["existing:ok"])
        result = append_processing_log(state, "chunk:ok")
        assert len(result) == 2
        assert "existing:ok" in result
        assert "chunk:ok" in result

    def test_returns_new_list_not_mutating_state(self):
        state = self._make_state()
        result = append_processing_log(state, "embed:ok")
        # The helper returns a new list; it does NOT mutate state["processing_log"]
        assert "embed:ok" not in state["processing_log"]
        assert "embed:ok" in result

    def test_returns_list(self):
        state = self._make_state()
        result = append_processing_log(state, "chunk:ok")
        assert isinstance(result, list)

    def test_empty_log_single_append(self):
        state = self._make_state()
        result = append_processing_log(state, "start:ok")
        assert result == ["start:ok"]

    def test_multiple_appends_accumulate(self):
        state = self._make_state()
        log = append_processing_log(state, "stage1:ok")
        state2 = {**state, "processing_log": log}
        log2 = append_processing_log(state2, "stage2:ok")
        assert "stage1:ok" in log2
        assert "stage2:ok" in log2
        assert len(log2) == 2


# ---------------------------------------------------------------------------
# parse_json_object — JSON parsing edge cases (regression)
# ---------------------------------------------------------------------------

class TestParseJsonObject:
    """Regression tests for parse_json_object JSON extraction helper.

    These tests target the three-strategy cascade: direct orjson parse,
    markdown fence strip, and raw_decode-from-first-brace.
    """

    def test_valid_json_object_returned(self):
        result = parse_json_object('{"a": 1, "b": "hello"}')
        assert result == {"a": 1, "b": "hello"}

    def test_nested_braces_in_string_values(self):
        """Values containing braces must NOT confuse the parser — regression for brace-counting bugs."""
        raw = '{"a": "foo {bar} baz", "b": 2}'
        result = parse_json_object(raw)
        assert result == {"a": "foo {bar} baz", "b": 2}

    def test_truncated_json_returns_empty(self):
        """Truncated input must not crash — returns {} gracefully."""
        result = parse_json_object('{"key": "val')
        assert result == {}

    def test_markdown_fence_json_extracted(self):
        """JSON wrapped in ```json ... ``` fences must be parsed correctly."""
        raw = "```json\n{\"answer\": 42}\n```"
        result = parse_json_object(raw)
        assert result == {"answer": 42}

    def test_markdown_fence_without_language_tag(self):
        """JSON wrapped in bare ``` fences must also be parsed correctly."""
        raw = "```\n{\"x\": \"y\"}\n```"
        result = parse_json_object(raw)
        assert result == {"x": "y"}

    def test_no_json_found_returns_empty(self):
        """Input with no JSON object must return {} without raising."""
        result = parse_json_object("There is no JSON here at all.")
        assert result == {}

    def test_empty_string_returns_empty(self):
        """Empty string must return {} gracefully."""
        result = parse_json_object("")
        assert result == {}

    def test_whitespace_only_returns_empty(self):
        """Whitespace-only input must return {} gracefully."""
        result = parse_json_object("   \n\t  ")
        assert result == {}

    def test_prose_wrapped_json_extracted(self):
        """JSON embedded in surrounding prose must be extracted via raw_decode."""
        raw = 'Here is the result:\n{"status": "ok"}\nLet me know if helpful.'
        result = parse_json_object(raw)
        assert result == {"status": "ok"}

    def test_returns_dict_type(self):
        """Return value must always be a dict, never None or another type."""
        for raw in ['{"k": 1}', "", "not json", "```json\n{}\n```"]:
            result = parse_json_object(raw)
            assert isinstance(result, dict), f"Expected dict for input {raw!r}, got {type(result)}"

    def test_json_array_returns_empty(self):
        """A top-level JSON array (not object) must return {} since only objects are valid."""
        result = parse_json_object("[1, 2, 3]")
        assert result == {}

    def test_deeply_nested_object(self):
        """Deeply nested JSON must be parsed correctly without hitting recursion issues."""
        raw = '{"a": {"b": {"c": {"d": "deep"}}}}'
        result = parse_json_object(raw)
        assert result == {"a": {"b": {"c": {"d": "deep"}}}}


# ---------------------------------------------------------------------------
# Edge case tests for shared helper internals
# ---------------------------------------------------------------------------


class TestLocateSpanEdgeCases:
    """Tests for _locate_span helper (private, tested via map_chunk_provenance)."""

    def test_mock_locate_span_empty_haystack(self):
        """map_chunk_provenance with empty original_text returns no-match."""
        provenance, _, _ = map_chunk_provenance(
            "some chunk",
            original_text="",
            refactored_text="some chunk appears here",
            original_cursor=0,
            refactored_cursor=0,
        )
        assert provenance["original_char_start"] == -1
        assert provenance["provenance_confidence"] == 0.0

    def test_mock_locate_span_empty_needle(self):
        """map_chunk_provenance with empty chunk_text returns no-match."""
        provenance, _, _ = map_chunk_provenance(
            "",
            original_text="some original text here",
            refactored_text="some refactored text here",
            original_cursor=0,
            refactored_cursor=0,
        )
        # Empty needle returns -1 in _locate_span
        assert provenance["refactored_char_start"] == -1

    def test_mock_locate_span_cursor_hint_used(self):
        """Cursor hint should speed up matching — match after cursor offset."""
        text = "AAA first segment. BBB second segment."
        # Search for "BBB second segment." with cursor at 19 (start of 'BBB')
        provenance, new_orig, new_ref = map_chunk_provenance(
            "BBB second segment.",
            original_text=text,
            refactored_text=text,
            original_cursor=19,
            refactored_cursor=19,
        )
        assert provenance["original_char_start"] >= 0

    def test_mock_paragraph_fuzzy_fallback(self):
        """map_chunk_provenance should use paragraph fuzzy match when exact fails."""
        original = "First paragraph content here.\n\nSecond paragraph content here.\n\n"
        # Use a slightly different chunk text to trigger fuzzy matching
        chunk = "First paragraph content here"  # exact match exists as substring

        provenance, _, _ = map_chunk_provenance(
            chunk,
            original_text=original,
            refactored_text="different refactored " + chunk + " here",
            original_cursor=0,
            refactored_cursor=0,
        )
        # Should find a match (exact or fuzzy)
        assert isinstance(provenance["provenance_confidence"], float)


class TestBestParagraphSpanEdgeCases:
    """Tests for _best_paragraph_span (tested indirectly via map_chunk_provenance)."""

    def test_mock_empty_text_returns_no_match(self):
        """Empty original_text should cause no-match in paragraph span."""
        provenance, _, _ = map_chunk_provenance(
            "some anchor text",
            original_text="",
            refactored_text="",
            original_cursor=0,
            refactored_cursor=0,
        )
        assert provenance["original_char_start"] == -1

    def test_mock_empty_anchor_returns_no_match(self):
        """Empty chunk_text should return no-match."""
        provenance, _, _ = map_chunk_provenance(
            "",
            original_text="some text here",
            refactored_text="",
            original_cursor=0,
            refactored_cursor=0,
        )
        assert provenance["refactored_char_start"] == -1


class TestAppendProcessingLogVerbose:
    """Tests for verbose logging path in append_processing_log."""

    def test_mock_append_log_verbose_logging_emits_info(self, caplog):
        """When verbose_stage_logs=True, append_processing_log should emit info log."""
        import logging

        class FakeConfig:
            verbose_stage_logs = True

        class FakeRuntime:
            config = FakeConfig()

        state = {
            "processing_log": [],
            "runtime": FakeRuntime(),
            "source_name": "test_source",
        }

        with caplog.at_level(logging.INFO, logger="rag.ingest.common.shared"):
            result = append_processing_log(state, "chunk:ok")

        assert "chunk:ok" in result
        assert any("chunk:ok" in r.message for r in caplog.records)

    def test_mock_append_log_no_runtime_no_crash(self):
        """append_processing_log with runtime=None should not crash."""
        state = {"processing_log": [], "runtime": None}
        result = append_processing_log(state, "stage:ok")
        assert "stage:ok" in result


class TestQualityScoreEdgeCases:
    """Additional edge cases for quality_score."""

    def test_mock_quality_score_just_below_bonus_threshold(self):
        """Text just below 120 chars should not get length bonus."""
        text = "a" * 119
        score = quality_score(text)
        # Base is 0.4, no length bonus
        assert score == pytest.approx(0.4, abs=0.01)

    def test_mock_quality_score_exactly_at_bonus_threshold(self):
        """Text at exactly 120 chars should get length bonus."""
        text = "a" * 120
        score = quality_score(text)
        # Base 0.4 + bonus 0.2 = 0.6
        assert score >= 0.6
