"""Comprehensive unit tests for src/ingest/support/document.py.

All functions are pure Python with no external service dependencies,
making this a high-coverage, fast test suite.
"""

import pytest

from src.ingest.support.document import (
    DocumentMetadata,
    clean_whitespace,
    extract_metadata,
    metadata_to_dict,
    normalize_unicode,
    strip_boilerplate,
    strip_trailing_short_lines,
)


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
