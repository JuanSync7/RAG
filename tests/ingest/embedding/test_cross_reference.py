# @summary
# Tests for the cross_reference_extraction_node embedding pipeline stage.
# Covers: disabled passthrough, DOC-\d{3,6} patterns, Section patterns,
# RFC patterns, text source preference, deduplication, and edge cases.
# @end-summary

import pytest
from unittest.mock import MagicMock

from src.ingest.common.types import IngestionConfig, Runtime
from src.ingest.embedding.nodes.cross_reference_extraction import cross_reference_extraction_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(cleaned="", refactored="", enabled=True):
    config = IngestionConfig(enable_cross_reference_extraction=enabled)
    runtime = Runtime(
        config=config,
        embedder=MagicMock(),
        weaviate_client=MagicMock(),
        kg_builder=None,
    )
    return {
        "cleaned_text": cleaned,
        "refactored_text": refactored,
        "cross_references": [],
        "errors": [],
        "processing_log": [],
        "runtime": runtime,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _ref_values(refs: list) -> list[str]:
    """Extract the 'value' string from cross_refs result (list of dicts)."""
    return [r["value"] if isinstance(r, dict) else str(r) for r in refs]


def test_disabled_returns_empty():
    """When extraction is disabled the result list is empty.
    Disabled path returns only processing_log (no cross_references key)."""
    state = _make_state(cleaned="DOC-1234 RFC 2119", enabled=False)
    result = cross_reference_extraction_node(state)
    assert result.get("cross_references", []) == []


def test_doc_pattern_matched():
    """Standard DOC-NNNN reference is found."""
    state = _make_state(cleaned="See DOC-1234 for details.")
    result = cross_reference_extraction_node(state)
    values = _ref_values(result["cross_references"])
    assert any("DOC-1234" in v for v in values)


def test_doc_min_digits_3():
    """DOC references with exactly 3 digits are matched."""
    state = _make_state(cleaned="Refer to DOC-123 here.")
    result = cross_reference_extraction_node(state)
    values = _ref_values(result["cross_references"])
    assert any("DOC-123" in v for v in values)


def test_doc_max_digits_6():
    """DOC references with exactly 6 digits are matched."""
    state = _make_state(cleaned="Refer to DOC-123456 here.")
    result = cross_reference_extraction_node(state)
    values = _ref_values(result["cross_references"])
    assert any("DOC-123456" in v for v in values)


def test_doc_1_digit_not_matched():
    """DOC references with only 1 digit are NOT matched (minimum is 2 digits: \\d{2,})."""
    state = _make_state(cleaned="See DOC-1 for details.")
    result = cross_reference_extraction_node(state)
    values = _ref_values(result.get("cross_references", []))
    assert not any("DOC-1" == v for v in values)


def test_section_simple():
    """Simple 'Section N' reference is found."""
    state = _make_state(cleaned="As described in Section 3.")
    result = cross_reference_extraction_node(state)
    values = _ref_values(result["cross_references"])
    assert any("Section 3" in v or "3" in v for v in values)


def test_section_multilevel():
    """Multi-level section reference like 'Section 3.1.2' is found."""
    state = _make_state(cleaned="Refer to Section 3.1.2 for context.")
    result = cross_reference_extraction_node(state)
    values = _ref_values(result["cross_references"])
    assert any("3.1.2" in v for v in values)


def test_section_lowercase():
    """Lowercase 'section' keyword is also matched."""
    state = _make_state(cleaned="As noted in section 4.2.")
    result = cross_reference_extraction_node(state)
    values = _ref_values(result["cross_references"])
    assert any("4.2" in v for v in values)


def test_rfc_with_space():
    """RFC reference with a space before digits is matched."""
    state = _make_state(cleaned="Complies with RFC 2119.")
    result = cross_reference_extraction_node(state)
    values = _ref_values(result["cross_references"])
    assert any("2119" in v for v in values)


def test_rfc_without_space_not_matched():
    """RFC reference without a space before digits is NOT matched.

    The regex pattern is r'\\bRFC\\s+\\d{3,5}\\b' which requires whitespace.
    """
    state = _make_state(cleaned="Complies with RFC2119.")
    result = cross_reference_extraction_node(state)
    values = _ref_values(result.get("cross_references", []))
    assert not any("2119" in v for v in values)


def test_rfc_2_digits_not_matched():
    """RFC references with only 2 digits are NOT matched (minimum is 3)."""
    state = _make_state(cleaned="See RFC 12 for info.")
    result = cross_reference_extraction_node(state)
    values = _ref_values(result.get("cross_references", []))
    assert not any(v.strip() in ("RFC 12", "RFC12") for v in values)


def test_multiple_types_all_found():
    """DOC, Section, and RFC references in the same text are all extracted."""
    text = "See DOC-4567, Section 2.1, and RFC 4918 for full details."
    state = _make_state(cleaned=text)
    result = cross_reference_extraction_node(state)
    values = _ref_values(result["cross_references"])
    assert any("DOC-4567" in v for v in values)
    assert any("2.1" in v for v in values)
    assert any("4918" in v for v in values)


def test_repeated_reference_both_returned():
    """The same reference appearing twice is returned twice (no deduplication).

    cross_refs() appends every regex match without deduplication.
    """
    text = "See DOC-9999 and also DOC-9999 again."
    state = _make_state(cleaned=text)
    result = cross_reference_extraction_node(state)
    values = _ref_values(result["cross_references"])
    matching = [v for v in values if "DOC-9999" in v]
    assert len(matching) == 2


def test_prefers_refactored_text():
    """When refactored_text is non-empty it is used for extraction."""
    state = _make_state(
        cleaned="DOC-0001",
        refactored="See DOC-9876 for details.",
    )
    result = cross_reference_extraction_node(state)
    values = _ref_values(result["cross_references"])
    # Should find the reference from refactored_text
    assert any("DOC-9876" in v for v in values)
    # Should NOT find the reference that only exists in cleaned_text
    assert not any("DOC-0001" in v for v in values)


def test_fallback_to_cleaned_text():
    """When refactored_text is empty, cleaned_text is used instead."""
    state = _make_state(cleaned="See DOC-5555 for details.", refactored="")
    result = cross_reference_extraction_node(state)
    values = _ref_values(result["cross_references"])
    assert any("DOC-5555" in v for v in values)


def test_no_matches_returns_empty():
    """Plain prose with no recognisable patterns returns an empty list."""
    state = _make_state(cleaned="The weather today is quite pleasant.")
    result = cross_reference_extraction_node(state)
    assert result["cross_references"] == []


def test_both_texts_empty_returns_empty():
    """When both cleaned and refactored texts are empty the result is empty."""
    state = _make_state(cleaned="", refactored="")
    result = cross_reference_extraction_node(state)
    assert result["cross_references"] == []


def test_partial_state_refactored_text_none():
    """When refactored_text is None (absent optional field) cleaned_text is used."""
    state = _make_state(cleaned="See DOC-7777 here.", refactored="")
    state["refactored_text"] = None  # simulate absent optional upstream field
    result = cross_reference_extraction_node(state)
    values = _ref_values(result.get("cross_references", []))
    assert any("DOC-7777" in v for v in values)


def test_partial_state_cleaned_text_absent():
    """When cleaned_text is absent from state, node returns cross_references (possibly empty)."""
    state = _make_state(cleaned="DOC-1234", refactored="")
    state.pop("cleaned_text", None)  # simulate absent optional upstream field
    # node uses .get() so should gracefully fall back to empty string
    result = cross_reference_extraction_node(state)
    assert "cross_references" in result or "processing_log" in result
