"""Regression + new-pattern tests for cross_refs().

Covers the four datasheet-dominant citation patterns added on top of the
original three (DOC-NNN, Section, RFC):
    - section_symbol  (e.g. "§3.1", "§ 3.1.4", "§3")
    - table           (e.g. "Table 5-2", "Table 5.2", "table 3-1")
    - figure          (e.g. "Figure 7-1", "Fig 7-1", "Fig. 7-1")
    - appendix        (e.g. "Appendix A", "Appendix B.2", "appendix C")

Negative cases guard against false positives like "tablespoon", "figurine",
"appendices", and "§abc".
"""

from __future__ import annotations

from src.ingest.common.shared import cross_refs


def _types(refs: list[dict[str, str]]) -> list[str]:
    return [r["type"] for r in refs]


def _values(refs: list[dict[str, str]]) -> list[str]:
    return [r["value"] for r in refs]


# --- section_symbol -----------------------------------------------------------

class TestSectionSymbolPattern:
    def test_section_symbol_basic(self):
        refs = cross_refs("see §3.1 for details")
        assert any(r["type"] == "section_symbol" for r in refs)

    def test_section_symbol_with_space(self):
        refs = cross_refs("see § 3.1 for details")
        assert any(r["type"] == "section_symbol" for r in refs)

    def test_section_symbol_three_levels(self):
        refs = cross_refs("see §3.1.4 for details")
        assert any(
            r["type"] == "section_symbol" and "3.1.4" in r["value"] for r in refs
        )

    def test_section_symbol_single_digit(self):
        refs = cross_refs("see §3 for details")
        assert any(r["type"] == "section_symbol" for r in refs)

    def test_section_symbol_followed_by_letters_does_not_match(self):
        refs = cross_refs("§abc is not a section")
        assert not any(r["type"] == "section_symbol" for r in refs)


# --- table --------------------------------------------------------------------

class TestTablePattern:
    def test_table_dash_form(self):
        refs = cross_refs("see Table 5-2")
        assert any(r["type"] == "table" for r in refs)

    def test_table_dot_form(self):
        refs = cross_refs("see Table 5.2")
        assert any(r["type"] == "table" for r in refs)

    def test_table_plain_number(self):
        refs = cross_refs("see Table 12")
        assert any(r["type"] == "table" for r in refs)

    def test_table_lowercase(self):
        refs = cross_refs("see table 3-1")
        assert any(r["type"] == "table" for r in refs)

    def test_tablespoon_does_not_match(self):
        refs = cross_refs("add a tablespoon 5 of sugar")
        assert not any(r["type"] == "table" for r in refs)


# --- figure -------------------------------------------------------------------

class TestFigurePattern:
    def test_figure_full(self):
        refs = cross_refs("see Figure 7-1")
        assert any(r["type"] == "figure" for r in refs)

    def test_figure_abbrev_no_dot(self):
        refs = cross_refs("see Fig 7-1")
        assert any(r["type"] == "figure" for r in refs)

    def test_figure_abbrev_with_dot(self):
        refs = cross_refs("see Fig. 7-1")
        assert any(r["type"] == "figure" for r in refs)

    def test_figure_lowercase_plain(self):
        refs = cross_refs("see figure 12")
        assert any(r["type"] == "figure" for r in refs)

    def test_figurine_does_not_match(self):
        refs = cross_refs("a figurine 5 stood on the shelf")
        assert not any(r["type"] == "figure" for r in refs)

    def test_figure_three_segment_dot_section_heading_does_not_match(self):
        # Real ESP32-S3 false positive: "Figure 4.1.2 Memory Organization"
        # is a section heading, not a figure citation. The third dot-segment
        # is the disambiguator — figure numbers in datasheets are at most
        # two segments (chapter.figure or chapter-figure).
        refs = cross_refs("see Figure 4.1.2 Memory Organization for details")
        assert not any(r["type"] == "figure" for r in refs)

    def test_figure_three_segment_dot_form_does_not_match(self):
        refs = cross_refs("see Figure 10.3.5 for details")
        assert not any(r["type"] == "figure" for r in refs)

    def test_figure_dash_then_dot_does_not_match(self):
        # "Figure 4-1.2" — dash form then a trailing dot-segment is rejected;
        # genuine figures use either dash or dot, not both.
        refs = cross_refs("see Figure 4-1.2 here")
        assert not any(r["type"] == "figure" for r in refs)

    def test_figure_two_segment_dot_form_still_matches(self):
        refs = cross_refs("see Figure 10.3 for details")
        assert any(r["type"] == "figure" and r["value"].endswith("10.3") for r in refs)

    def test_figure_two_segment_dash_form_still_matches(self):
        refs = cross_refs("see Figure 4-1 for details")
        assert any(r["type"] == "figure" and r["value"].endswith("4-1") for r in refs)

    def test_figure_fig_dot_plain_still_matches(self):
        refs = cross_refs("see Fig. 2 for details")
        assert any(r["type"] == "figure" for r in refs)


# --- appendix -----------------------------------------------------------------

class TestAppendixPattern:
    def test_appendix_letter(self):
        refs = cross_refs("see Appendix A")
        assert any(r["type"] == "appendix" for r in refs)

    def test_appendix_letter_dot_number(self):
        refs = cross_refs("see Appendix B.2")
        assert any(r["type"] == "appendix" for r in refs)

    def test_appendix_lowercase(self):
        refs = cross_refs("see appendix C")
        assert any(r["type"] == "appendix" for r in refs)

    def test_appendices_alone_does_not_match(self):
        refs = cross_refs("multiple appendices follow")
        assert not any(r["type"] == "appendix" for r in refs)


# --- mixed-input + regression -------------------------------------------------

class TestMixedAndRegression:
    def test_mixed_all_four_new_types(self):
        text = "see §3.1 and Table 5-2 and Figure 7-1 in Appendix A"
        refs = cross_refs(text)
        types = _types(refs)
        assert "section_symbol" in types
        assert "table" in types
        assert "figure" in types
        assert "appendix" in types
        # Exactly one of each new type for this input
        assert types.count("section_symbol") == 1
        assert types.count("table") == 1
        assert types.count("figure") == 1
        assert types.count("appendix") == 1

    def test_existing_doc_pattern_still_works(self):
        refs = cross_refs("see DOC-001 for info")
        assert any(r["type"] == "document_id" and r["value"] == "DOC-001" for r in refs)

    def test_existing_section_pattern_still_works(self):
        refs = cross_refs("see Section 3.1 for info")
        assert any(r["type"] == "section" for r in refs)

    def test_existing_rfc_pattern_still_works(self):
        refs = cross_refs("see RFC 2616 for info")
        assert any(r["type"] == "standard" for r in refs)

    def test_return_shape_is_list_of_dicts_with_type_value(self):
        refs = cross_refs("§3.1 Table 5 Figure 6 Appendix A DOC-12 Section 1.1 RFC 2616")
        assert isinstance(refs, list)
        for r in refs:
            assert isinstance(r, dict)
            assert set(r.keys()) == {"type", "value"}
            assert isinstance(r["type"], str)
            assert isinstance(r["value"], str)
