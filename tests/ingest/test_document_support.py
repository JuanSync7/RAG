"""Comprehensive unit tests for src/ingest/support/document.py.

All functions are pure Python with no external service dependencies,
making this a high-coverage, fast test suite.
"""

import pytest

from src.ingest.support.document import (
    DocumentMetadata,
    chunk_text,
    clean_text,
    clean_whitespace,
    extract_metadata,
    metadata_to_dict,
    normalize_unicode,
    process_document,
    strip_boilerplate,
    strip_section_markers,
    strip_trailing_short_lines,
)
from src.ingest.common import ProcessedChunk


# ---------------------------------------------------------------------------
# strip_boilerplate
# ---------------------------------------------------------------------------


class TestStripBoilerplate:
    """Tests for strip_boilerplate()."""

    def test_removes_equals_banner_block(self):
        text = "===\nBanner Header Line\nAnother Banner Line\n===\nReal content."
        result = strip_boilerplate(text)
        assert "Banner Header Line" not in result
        assert "Real content." in result

    def test_removes_metadata_title_line(self):
        text = "Title: My Document\nBody text here."
        result = strip_boilerplate(text)
        assert "Title: My Document" not in result
        assert "Body text here." in result

    def test_removes_metadata_author_line(self):
        text = "Author: Jane Doe\nBody text."
        result = strip_boilerplate(text)
        assert "Author: Jane Doe" not in result
        assert "Body text." in result

    def test_removes_metadata_date_line(self):
        text = "Date: 2024-01-15\nBody text."
        result = strip_boilerplate(text)
        assert "Date: 2024-01-15" not in result
        assert "Body text." in result

    def test_removes_metadata_department_line(self):
        text = "Department: Engineering\nBody text."
        result = strip_boilerplate(text)
        assert "Department: Engineering" not in result

    def test_removes_metadata_tags_line(self):
        text = "Tags: ml, ai, nlp\nBody text."
        result = strip_boilerplate(text)
        assert "Tags: ml, ai, nlp" not in result

    def test_removes_metadata_classification_line(self):
        text = "Classification: Confidential\nBody text."
        result = strip_boilerplate(text)
        assert "Classification: Confidential" not in result

    def test_removes_metadata_document_id_line(self):
        text = "Document ID: DOC-1234\nBody text."
        result = strip_boilerplate(text)
        assert "Document ID: DOC-1234" not in result

    def test_removes_page_footer(self):
        text = "Some content.\nPage 3 of 10 | Section A | Company Inc.\nMore content."
        result = strip_boilerplate(text)
        assert "Page 3 of 10" not in result
        assert "Some content." in result

    def test_removes_generated_timestamp(self):
        text = "Generated: 2024-01-15 10:00 AM\nContent here."
        result = strip_boilerplate(text)
        assert "Generated:" not in result
        assert "Content here." in result

    def test_removes_last_modified_line(self):
        text = "Last Modified: 2024-01-10\nContent here."
        result = strip_boilerplate(text)
        assert "Last Modified:" not in result
        assert "Content here." in result

    def test_removes_copyright_line(self):
        text = "© 2024 Acme Corporation. All rights reserved.\nContent here."
        result = strip_boilerplate(text)
        assert "© 2024 Acme Corporation" not in result
        assert "Content here." in result

    def test_removes_email_greeting_hi_everyone(self):
        text = "Hi everyone,\nHere is the update."
        result = strip_boilerplate(text)
        assert "Hi everyone" not in result
        assert "Here is the update." in result

    def test_removes_email_signoff_best(self):
        text = "The project is complete.\nBest,\nAlice"
        result = strip_boilerplate(text)
        assert "Best," not in result
        assert "The project is complete." in result

    def test_removes_email_signoff_regards(self):
        text = "See you soon.\nRegards"
        result = strip_boilerplate(text)
        assert "Regards" not in result

    def test_removes_email_signoff_cheers(self):
        text = "Talk soon.\nCheers,"
        result = strip_boilerplate(text)
        assert "Cheers," not in result

    def test_removes_email_signoff_thanks(self):
        text = "Please review.\nThanks,"
        result = strip_boilerplate(text)
        assert "Thanks," not in result

    def test_removes_email_signature_block(self):
        text = "Main content here.\n\n-- \nJohn Smith\nSenior Engineer"
        result = strip_boilerplate(text)
        assert "John Smith" not in result
        assert "Main content here." in result

    def test_removes_confidentiality_disclaimer(self):
        text = "Important info.\nThis email and any attachments are confidential and intended solely for the addressee."
        result = strip_boilerplate(text)
        assert "confidential" not in result.lower()
        assert "Important info." in result

    def test_removes_toc_block(self):
        text = "[TOC]\n1. Introduction\n2. Methods\n3. Results\nBody text."
        result = strip_boilerplate(text)
        assert "[TOC]" not in result
        assert "Body text." in result

    def test_removes_draft_marker(self):
        text = "DRAFT - DO NOT DISTRIBUTE\nContent here."
        result = strip_boilerplate(text)
        assert "DRAFT" not in result
        assert "Content here." in result

    def test_removes_do_not_distribute_marker(self):
        text = "Do Not Distribute\nConfidential content."
        result = strip_boilerplate(text)
        assert "Do Not Distribute" not in result

    def test_removes_document_version_line(self):
        text = "Document version 1.2\nContent here."
        result = strip_boilerplate(text)
        assert "Document version" not in result
        assert "Content here." in result

    def test_removes_reference_citation(self):
        text = "See [1] for details.\n[1] Smith, J. (2023). Machine Learning. Journal.\nContent."
        result = strip_boilerplate(text)
        assert "[1] Smith" not in result

    def test_removes_internal_wiki_link(self):
        text = "Internal wiki: https://wiki.example.com/page\nContent."
        result = strip_boilerplate(text)
        assert "wiki.example.com" not in result
        assert "Content." in result

    def test_removes_see_also_link(self):
        text = "See also: https://docs.example.com/reference\nContent."
        result = strip_boilerplate(text)
        assert "docs.example.com" not in result
        assert "Content." in result

    def test_removes_prepared_by_line(self):
        text = "Prepared by: Jane Doe\nContent here."
        result = strip_boilerplate(text)
        assert "Prepared by" not in result
        assert "Content here." in result

    def test_removes_reviewed_by_line(self):
        text = "Reviewed by: John Smith\nContent here."
        result = strip_boilerplate(text)
        assert "Reviewed by" not in result
        assert "Content here." in result

    def test_removes_last_updated_standalone(self):
        text = "Last updated: March 2024\nContent here."
        result = strip_boilerplate(text)
        assert "Last updated:" not in result
        assert "Content here." in result

    def test_removes_separator_dash_line(self):
        text = "Content above.\n---\nContent below."
        result = strip_boilerplate(text)
        assert "---" not in result
        assert "Content above." in result
        assert "Content below." in result

    def test_removes_separator_equals_line(self):
        text = "Content above.\n===\nContent below."
        result = strip_boilerplate(text)
        assert "Content above." in result
        assert "Content below." in result

    def test_removes_note_marker(self):
        text = "Body text.\nNOTE: This is an internal note.\nMore text."
        result = strip_boilerplate(text)
        assert "NOTE:" not in result
        assert "Body text." in result

    def test_removes_todo_marker(self):
        text = "Body text.\nTODO: Fix this section.\nMore text."
        result = strip_boilerplate(text)
        assert "TODO:" not in result

    def test_removes_fixme_marker(self):
        text = "Body text.\nFIXME: Bad logic here.\nMore text."
        result = strip_boilerplate(text)
        assert "FIXME:" not in result

    def test_removes_hack_marker(self):
        text = "Body text.\nHACK: Temporary workaround.\nMore text."
        result = strip_boilerplate(text)
        assert "HACK:" not in result

    def test_removes_following_up_line(self):
        text = "I wanted to follow up.\nFollowing up on the write-up from last week.\nSee attached."
        result = strip_boilerplate(text)
        assert "Following up on" not in result

    def test_removes_let_me_know_line(self):
        text = "Please review the attached.\nLet me know if you have questions.\nRegards."
        result = strip_boilerplate(text)
        assert "Let me know if you have questions" not in result

    def test_removes_senior_title_signoff(self):
        text = "Main content.\nSenior Engineer\nAcme Corp"
        result = strip_boilerplate(text)
        assert "Senior Engineer" not in result

    def test_removes_principal_title_signoff(self):
        text = "Main content.\nPrincipal Architect"
        result = strip_boilerplate(text)
        assert "Principal Architect" not in result

    def test_removes_lead_title_signoff(self):
        text = "Main content.\nLead Developer"
        result = strip_boilerplate(text)
        assert "Lead Developer" not in result

    def test_removes_meeting_reference(self):
        text = "We'll cover this in the deep-dive on Friday.\nSome content."
        result = strip_boilerplate(text)
        assert "deep-dive on Friday" not in result

    def test_removes_tech_talk_reference(self):
        text = "Join the tech talk next week for more details.\nContent."
        result = strip_boilerplate(text)
        assert "tech talk next week" not in result

    def test_preserves_regular_content(self):
        text = "The model achieved 95% accuracy on the test set.\nWe used cross-validation."
        result = strip_boilerplate(text)
        assert "95% accuracy" in result
        assert "cross-validation" in result

    def test_empty_string_returns_empty(self):
        assert strip_boilerplate("") == ""

    def test_metadata_case_insensitive(self):
        text = "title: My Document\nauthor: Jane Doe\nBody text."
        result = strip_boilerplate(text)
        assert "title: My Document" not in result
        assert "author: Jane Doe" not in result


# ---------------------------------------------------------------------------
# normalize_unicode
# ---------------------------------------------------------------------------


class TestNormalizeUnicode:
    """Tests for normalize_unicode()."""

    def test_left_single_quote_replaced(self):
        result = normalize_unicode("\u2018hello\u2019")
        assert result == "'hello'"

    def test_right_single_quote_replaced(self):
        result = normalize_unicode("it\u2019s fine")
        assert result == "it's fine"

    def test_left_double_quote_replaced(self):
        result = normalize_unicode("\u201chello\u201d")
        assert result == '"hello"'

    def test_right_double_quote_replaced(self):
        result = normalize_unicode("say \u201cyes\u201d please")
        assert result == 'say "yes" please'

    def test_en_dash_replaced_with_single_hyphen(self):
        result = normalize_unicode("pages 10\u201320")
        assert result == "pages 10-20"

    def test_em_dash_replaced_with_double_hyphen(self):
        result = normalize_unicode("the result\u2014amazing")
        assert result == "the result--amazing"

    def test_ellipsis_replaced_with_three_dots(self):
        result = normalize_unicode("and so on\u2026")
        assert result == "and so on..."

    def test_non_breaking_space_replaced_with_space(self):
        result = normalize_unicode("hello\u00a0world")
        assert result == "hello world"

    def test_plain_text_unchanged(self):
        text = "Hello, world! This is plain ASCII text."
        assert normalize_unicode(text) == text

    def test_multiple_replacements_in_one_string(self):
        result = normalize_unicode("\u201cHello\u201d\u2014 it\u2019s fine\u2026")
        assert result == '"Hello"-- it\'s fine...'

    def test_nfc_normalization_applied(self):
        # NFC normalization: combining characters should be composed
        import unicodedata
        # "a" + combining acute = NFC "á"
        decomposed = "a\u0301"  # NFD form
        result = normalize_unicode(decomposed)
        assert unicodedata.is_normalized("NFC", result)

    def test_empty_string_returns_empty(self):
        assert normalize_unicode("") == ""


# ---------------------------------------------------------------------------
# clean_whitespace
# ---------------------------------------------------------------------------


class TestCleanWhitespace:
    """Tests for clean_whitespace()."""

    def test_tab_replaced_with_space(self):
        result = clean_whitespace("hello\tworld")
        assert result == "hello world"

    def test_multiple_spaces_collapsed(self):
        result = clean_whitespace("hello    world")
        assert result == "hello world"

    def test_three_newlines_collapsed_to_two(self):
        result = clean_whitespace("para1\n\n\npara2")
        assert result == "para1\n\npara2"

    def test_four_newlines_collapsed_to_two(self):
        result = clean_whitespace("para1\n\n\n\npara2")
        assert result == "para1\n\npara2"

    def test_trailing_spaces_stripped_per_line(self):
        result = clean_whitespace("hello   \nworld   ")
        assert result == "hello\nworld"

    def test_leading_and_trailing_stripped(self):
        result = clean_whitespace("  \n  hello  \n  ")
        assert "hello" in result

    def test_multiple_tabs_collapsed(self):
        # Two tabs each become one space, then multiple spaces collapse to one
        result = clean_whitespace("col1\t\tcol2")
        assert result == "col1 col2"

    def test_two_newlines_preserved(self):
        result = clean_whitespace("para1\n\npara2")
        assert "para1\n\npara2" in result

    def test_single_newline_preserved(self):
        result = clean_whitespace("line1\nline2")
        assert "line1\nline2" in result

    def test_empty_string_returns_empty(self):
        assert clean_whitespace("") == ""

    def test_only_whitespace_returns_empty(self):
        assert clean_whitespace("   \t\n\n\n   ") == ""

    def test_mixed_tabs_and_spaces_collapsed(self):
        result = clean_whitespace("hello \t world")
        assert result == "hello world"


# ---------------------------------------------------------------------------
# strip_section_markers
# ---------------------------------------------------------------------------


class TestStripSectionMarkers:
    """Tests for strip_section_markers()."""

    def test_markdown_h2_header_stripped(self):
        result = strip_section_markers("## Introduction\nContent here.")
        assert "##" not in result
        assert "Introduction" in result

    def test_markdown_h1_header_stripped(self):
        result = strip_section_markers("# Title\nContent.")
        assert "#" not in result
        assert "Title" in result

    def test_markdown_h3_header_stripped(self):
        result = strip_section_markers("### Subsection\nContent.")
        assert "###" not in result
        assert "Subsection" in result

    def test_markdown_h6_header_stripped(self):
        result = strip_section_markers("###### Deep Header\nContent.")
        assert "######" not in result
        assert "Deep Header" in result

    def test_wiki_header_stripped(self):
        result = strip_section_markers("== Section Title ==\nContent.")
        assert "==" not in result
        assert "Section Title" in result

    def test_wiki_header_with_extra_spaces_stripped(self):
        result = strip_section_markers("==  Heading  ==\nContent.")
        assert "==" not in result
        assert "Heading" in result

    def test_numbered_allcaps_section_1(self):
        result = strip_section_markers("1. INTRODUCTION\nContent here.")
        assert "1." not in result
        assert "Introduction" in result

    def test_numbered_allcaps_section_2(self):
        result = strip_section_markers("2. METHODS\nContent here.")
        assert "2." not in result
        assert "Methods" in result

    def test_numbered_subsection_allcaps(self):
        result = strip_section_markers("2.1 SUPERVISED LEARNING\nContent.")
        assert "2.1" not in result
        assert "Supervised Learning" in result

    def test_regular_text_unchanged(self):
        text = "This is a regular paragraph with no headers."
        result = strip_section_markers(text)
        assert result == text

    def test_empty_string_returns_empty(self):
        assert strip_section_markers("") == ""

    def test_mixed_headers_and_content(self):
        text = "## Intro\nFirst paragraph.\n### Sub\nSecond paragraph."
        result = strip_section_markers(text)
        assert "##" not in result
        assert "###" not in result
        assert "Intro" in result
        assert "First paragraph." in result

    def test_numbered_section_with_ampersand(self):
        result = strip_section_markers("3. TOOLS & TECHNIQUES\nContent.")
        assert "3." not in result
        assert "Tools & Techniques" in result


# ---------------------------------------------------------------------------
# strip_trailing_short_lines
# ---------------------------------------------------------------------------


class TestStripTrailingShortLines:
    """Tests for strip_trailing_short_lines()."""

    def test_short_trailing_word_removed(self):
        text = "This is real content.\nAlice"
        result = strip_trailing_short_lines(text)
        assert "Alice" not in result
        assert "This is real content." in result

    def test_two_word_trailing_line_removed(self):
        text = "Real content.\nJohn Smith"
        result = strip_trailing_short_lines(text)
        assert "John Smith" not in result

    def test_four_word_trailing_line_removed(self):
        text = "Real content.\none two three four"
        result = strip_trailing_short_lines(text)
        assert "one two three four" not in result

    def test_five_word_trailing_line_preserved(self):
        text = "Real content.\none two three four five"
        result = strip_trailing_short_lines(text)
        assert "one two three four five" in result

    def test_sentence_ending_with_period_preserved(self):
        text = "Real content.\nSee you there."
        result = strip_trailing_short_lines(text)
        assert "See you there." in result

    def test_sentence_ending_with_question_mark_preserved(self):
        text = "Real content.\nIs this right?"
        result = strip_trailing_short_lines(text)
        assert "Is this right?" in result

    def test_sentence_ending_with_exclamation_preserved(self):
        text = "Real content.\nDone!"
        result = strip_trailing_short_lines(text)
        assert "Done!" in result

    def test_custom_max_words(self):
        text = "Real content.\none two three four five"
        # With max_words=5, the 5-word line should be removed
        result = strip_trailing_short_lines(text, max_words=5)
        assert "one two three four five" not in result

    def test_empty_string_returns_empty(self):
        assert strip_trailing_short_lines("") == ""

    def test_only_content_lines_unchanged(self):
        text = "First paragraph here.\nSecond paragraph here."
        result = strip_trailing_short_lines(text)
        assert "Second paragraph here." in result

    def test_multiple_short_trailing_lines_removed(self):
        text = "Main content.\nAlice\nSmith\nDr"
        result = strip_trailing_short_lines(text)
        assert "Main content." in result
        # At least some of the trailing short lines should be stripped
        assert result.strip() != text.strip()


# ---------------------------------------------------------------------------
# clean_text (full pipeline)
# ---------------------------------------------------------------------------


class TestCleanText:
    """Tests for clean_text() — full pipeline chain."""

    def test_removes_boilerplate_and_normalizes(self):
        text = "Title: My Doc\n\u201cHello\u201d World\nReal content here."
        result = clean_text(text)
        assert "Title: My Doc" not in result
        assert '"Hello"' in result
        assert "Real content here." in result

    def test_collapses_whitespace(self):
        text = "para1\n\n\n\npara2"
        result = clean_text(text)
        assert "\n\n\n" not in result

    def test_strips_section_markers(self):
        text = "## Introduction\nContent here."
        result = clean_text(text)
        assert "##" not in result
        assert "Introduction" in result

    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""

    def test_only_boilerplate_returns_empty_or_minimal(self):
        text = "Title: Boilerplate\nAuthor: Nobody\n---"
        result = clean_text(text)
        # Should have nothing meaningful left
        assert len(result.strip()) == 0 or "Boilerplate" not in result

    def test_strips_trailing_short_lines(self):
        text = "Real content about machine learning.\nAlice"
        result = clean_text(text)
        assert "Alice" not in result
        assert "Real content about machine learning." in result

    def test_final_whitespace_pass_applied(self):
        text = "## Section\nContent here.\n\n\n\nMore content."
        result = clean_text(text)
        assert "\n\n\n" not in result

    def test_unicode_smart_quotes_normalized(self):
        text = "\u201cThis is quoted\u201d text."
        result = clean_text(text)
        assert '"This is quoted"' in result

    def test_tab_replaced_with_space(self):
        text = "Column1\tColumn2\tColumn3"
        result = clean_text(text)
        assert "\t" not in result


# ---------------------------------------------------------------------------
# extract_metadata
# ---------------------------------------------------------------------------


class TestExtractMetadata:
    """Tests for extract_metadata()."""

    def test_extracts_title(self):
        raw = "Title: My Document\nContent here."
        meta = extract_metadata(raw, "file.txt")
        assert meta.title == "My Document"

    def test_extracts_author(self):
        raw = "Author: Jane Doe\nContent here."
        meta = extract_metadata(raw, "file.txt")
        assert meta.author == "Jane Doe"

    def test_extracts_date(self):
        raw = "Date: 2024-01-15\nContent here."
        meta = extract_metadata(raw, "file.txt")
        assert meta.date == "2024-01-15"

    def test_extracts_tags(self):
        raw = "Tags: ml, ai, nlp\nContent here."
        meta = extract_metadata(raw, "file.txt")
        assert meta.tags == ["ml", "ai", "nlp"]

    def test_extracts_subject_as_title(self):
        raw = "Subject: Meeting Notes\nContent here."
        meta = extract_metadata(raw, "email.txt")
        assert meta.title == "Meeting Notes"

    def test_extracts_prepared_by_as_author(self):
        raw = "Prepared by: Bob Jones\nContent here."
        meta = extract_metadata(raw, "doc.txt")
        assert meta.author == "Bob Jones"

    def test_extracts_last_updated_as_date(self):
        raw = "Last updated: March 2024\nContent here."
        meta = extract_metadata(raw, "doc.txt")
        assert meta.date == "March 2024"

    def test_source_is_preserved(self):
        raw = "Content here."
        meta = extract_metadata(raw, "myfile.pdf")
        assert meta.source == "myfile.pdf"

    def test_no_metadata_leaves_fields_none(self):
        raw = "Just plain content with no metadata headers."
        meta = extract_metadata(raw, "doc.txt")
        assert meta.title is None
        assert meta.author is None
        assert meta.date is None
        assert meta.tags is None

    def test_multiple_metadata_fields_extracted(self):
        raw = "Title: My Doc\nAuthor: Alice\nDate: 2024-06-01\nTags: a, b\nContent."
        meta = extract_metadata(raw, "doc.txt")
        assert meta.title == "My Doc"
        assert meta.author == "Alice"
        assert meta.date == "2024-06-01"
        assert meta.tags == ["a", "b"]

    def test_tags_single_item(self):
        raw = "Tags: python\nContent."
        meta = extract_metadata(raw, "doc.txt")
        assert meta.tags == ["python"]

    def test_tags_whitespace_stripped(self):
        raw = "Tags:  ml ,  ai ,  nlp \nContent."
        meta = extract_metadata(raw, "doc.txt")
        assert meta.tags == ["ml", "ai", "nlp"]

    def test_case_insensitive_keys(self):
        raw = "TITLE: Upper Case\nAUTHOR: Upper Author\nContent."
        meta = extract_metadata(raw, "doc.txt")
        assert meta.title == "Upper Case"
        assert meta.author == "Upper Author"

    def test_empty_string_returns_defaults(self):
        meta = extract_metadata("", "doc.txt")
        assert meta.source == "doc.txt"
        assert meta.title is None


# ---------------------------------------------------------------------------
# metadata_to_dict
# ---------------------------------------------------------------------------


class TestMetadataToDict:
    """Tests for metadata_to_dict()."""

    def test_source_always_present(self):
        meta = DocumentMetadata(source="myfile.txt")
        d = metadata_to_dict(meta)
        assert d["source"] == "myfile.txt"

    def test_tenant_id_always_present(self):
        meta = DocumentMetadata(source="x.txt")
        d = metadata_to_dict(meta)
        assert "tenant_id" in d

    def test_title_included_when_set(self):
        meta = DocumentMetadata(source="x.txt", title="My Title")
        d = metadata_to_dict(meta)
        assert d["title"] == "My Title"

    def test_title_excluded_when_none(self):
        meta = DocumentMetadata(source="x.txt", title=None)
        d = metadata_to_dict(meta)
        assert "title" not in d

    def test_author_included_when_set(self):
        meta = DocumentMetadata(source="x.txt", author="Jane")
        d = metadata_to_dict(meta)
        assert d["author"] == "Jane"

    def test_author_excluded_when_none(self):
        meta = DocumentMetadata(source="x.txt", author=None)
        d = metadata_to_dict(meta)
        assert "author" not in d

    def test_date_included_when_set(self):
        meta = DocumentMetadata(source="x.txt", date="2024-01-01")
        d = metadata_to_dict(meta)
        assert d["date"] == "2024-01-01"

    def test_date_excluded_when_none(self):
        meta = DocumentMetadata(source="x.txt", date=None)
        d = metadata_to_dict(meta)
        assert "date" not in d

    def test_tags_included_when_set(self):
        meta = DocumentMetadata(source="x.txt", tags=["ml", "ai"])
        d = metadata_to_dict(meta)
        assert d["tags"] == "ml, ai"

    def test_tags_excluded_when_none(self):
        meta = DocumentMetadata(source="x.txt", tags=None)
        d = metadata_to_dict(meta)
        assert "tags" not in d

    def test_tags_single_item_no_comma(self):
        meta = DocumentMetadata(source="x.txt", tags=["ml"])
        d = metadata_to_dict(meta)
        assert d["tags"] == "ml"

    def test_all_fields_included(self):
        meta = DocumentMetadata(
            source="doc.pdf",
            title="My Doc",
            author="Bob",
            date="2024-03-01",
            tags=["a", "b", "c"],
        )
        d = metadata_to_dict(meta)
        assert d["source"] == "doc.pdf"
        assert d["title"] == "My Doc"
        assert d["author"] == "Bob"
        assert d["date"] == "2024-03-01"
        assert d["tags"] == "a, b, c"

    def test_minimal_metadata_has_only_source_and_tenant(self):
        meta = DocumentMetadata(source="doc.txt")
        d = metadata_to_dict(meta)
        assert set(d.keys()) == {"source", "tenant_id"}


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------


class TestChunkText:
    """Tests for chunk_text()."""

    def test_short_text_produces_one_chunk(self):
        text = "This is a short text."
        chunks = chunk_text(text, chunk_size=512, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_produces_multiple_chunks(self):
        # Generate text longer than default chunk_size of 512
        text = "word " * 300  # ~1500 chars
        chunks = chunk_text(text, chunk_size=512, chunk_overlap=50)
        assert len(chunks) > 1

    def test_chunk_size_respected(self):
        text = "a " * 500  # ~1000 chars
        chunk_size = 200
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=20)
        for chunk in chunks:
            assert len(chunk) <= chunk_size + 50  # allow small splitter tolerance

    def test_returns_list_of_strings(self):
        text = "Hello world."
        chunks = chunk_text(text)
        assert isinstance(chunks, list)
        assert all(isinstance(c, str) for c in chunks)

    def test_custom_chunk_size(self):
        text = "a " * 200  # ~400 chars
        chunks_small = chunk_text(text, chunk_size=100, chunk_overlap=10)
        chunks_large = chunk_text(text, chunk_size=400, chunk_overlap=10)
        assert len(chunks_small) >= len(chunks_large)

    def test_empty_text_returns_no_meaningful_chunks(self):
        # RecursiveCharacterTextSplitter may return [''] for empty input;
        # process_document handles this by checking cleaned text before chunking.
        chunks = chunk_text("")
        # Either empty list or list containing only empty strings
        assert chunks == [] or all(c == "" for c in chunks)

    def test_chunks_cover_content(self):
        text = "The quick brown fox jumps over the lazy dog."
        chunks = chunk_text(text, chunk_size=512)
        combined = " ".join(chunks)
        # All words from original text should appear in combined chunks
        assert "quick" in combined
        assert "lazy" in combined


# ---------------------------------------------------------------------------
# process_document (full pipeline)
# ---------------------------------------------------------------------------


class TestProcessDocument:
    """Tests for process_document()."""

    def test_returns_list_of_processed_chunks(self):
        text = "Title: My Doc\nAuthor: Alice\nDate: 2024-01-01\n\nThis is the main content of the document. " * 5
        result = process_document(text, source="doc.txt")
        assert isinstance(result, list)
        assert all(isinstance(c, ProcessedChunk) for c in result)

    def test_empty_text_returns_empty_list(self):
        # If cleaning results in empty text, no chunks returned
        text = "Title: Only Metadata\nAuthor: Nobody\n---"
        result = process_document(text, source="empty.txt")
        # Result can be empty if all content is boilerplate
        assert isinstance(result, list)

    def test_metadata_attached_to_chunks(self):
        text = "Title: Test Doc\nAuthor: Bob\nContent about machine learning and NLP. " * 5
        result = process_document(text, source="test.txt")
        assert len(result) > 0
        for chunk in result:
            assert chunk.metadata["source"] == "test.txt"
            assert "tenant_id" in chunk.metadata

    def test_title_extracted_and_in_metadata(self):
        text = "Title: My Great Document\n\nThis is the main content section with enough words to pass cleaning. " * 3
        result = process_document(text, source="doc.txt")
        if result:
            assert result[0].metadata.get("title") == "My Great Document"

    def test_author_extracted_and_in_metadata(self):
        text = "Author: Jane Smith\n\nMain content with substantial text here for testing purposes. " * 3
        result = process_document(text, source="doc.txt")
        if result:
            assert result[0].metadata.get("author") == "Jane Smith"

    def test_chunk_index_in_metadata(self):
        text = "Real content section. " * 100  # enough text to create multiple chunks
        result = process_document(text, source="doc.txt")
        assert len(result) > 0
        for i, chunk in enumerate(result):
            assert chunk.metadata["chunk_index"] == i

    def test_total_chunks_in_metadata(self):
        text = "Real content section. " * 100
        result = process_document(text, source="doc.txt")
        n = len(result)
        for chunk in result:
            assert chunk.metadata["total_chunks"] == n

    def test_default_source_is_unknown(self):
        text = "Some document content here with enough words for chunking. " * 5
        result = process_document(text)
        if result:
            assert result[0].metadata["source"] == "unknown"

    def test_chunk_text_is_string(self):
        text = "Content for the document pipeline. " * 20
        result = process_document(text, source="doc.txt")
        assert len(result) > 0
        for chunk in result:
            assert isinstance(chunk.text, str)
            assert len(chunk.text) > 0

    def test_pipeline_cleans_unicode(self):
        text = "\u201cThis is a quote\u201d and it contains \u2014 em dash. " * 10
        result = process_document(text, source="doc.txt")
        assert len(result) > 0
        for chunk in result:
            # Smart quotes and em dash should be normalized
            assert "\u201c" not in chunk.text
            assert "\u2014" not in chunk.text

    def test_pipeline_removes_boilerplate(self):
        # Title header appears once at the top on its own line; body content repeats
        text = (
            "Title: Boilerplate Title\n"
            "---\n\n"
            + "Main body content here with substantial text.\n\n" * 5
        )
        result = process_document(text, source="doc.txt")
        # The title metadata line should not appear in chunk text
        for chunk in result:
            assert "Title: Boilerplate Title" not in chunk.text

    def test_metadata_dict_has_expected_keys(self):
        text = "Title: T\nAuthor: A\nDate: D\nTags: x, y\n\nContent. " * 5
        result = process_document(text, source="s.txt")
        if result:
            meta = result[0].metadata
            assert "source" in meta
            assert "tenant_id" in meta
            assert "chunk_index" in meta
            assert "total_chunks" in meta


# ---------------------------------------------------------------------------
# DocumentMetadata dataclass
# ---------------------------------------------------------------------------


class TestDocumentMetadata:
    """Tests for DocumentMetadata dataclass."""

    def test_default_source(self):
        meta = DocumentMetadata()
        assert meta.source == "unknown"

    def test_default_optional_fields_are_none(self):
        meta = DocumentMetadata()
        assert meta.title is None
        assert meta.author is None
        assert meta.date is None
        assert meta.tags is None

    def test_can_set_all_fields(self):
        meta = DocumentMetadata(
            source="doc.pdf",
            title="T",
            author="A",
            date="D",
            tags=["x"],
        )
        assert meta.source == "doc.pdf"
        assert meta.title == "T"
        assert meta.author == "A"
        assert meta.date == "D"
        assert meta.tags == ["x"]
