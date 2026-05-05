"""Unit tests for output_sanitizer.

Tests are organised by issue:
- Issue #4  — punctuation-insensitive prompt-fragment matching (_is_prompt_fragment)
- Issue #15 — word-boundary checks for boundary markers and template artifacts
"""

import pytest

from src.retrieval.generation.nodes.output_sanitizer import (
    sanitize_answer,
    _is_boundary_marker,
    _is_prompt_fragment,
    _is_template_artifact,
)


# ---------------------------------------------------------------------------
# Issue #4 — punctuation-insensitive _is_prompt_fragment
# ---------------------------------------------------------------------------


class TestIsPromptFragmentPunctuation:
    """_is_prompt_fragment must strip punctuation before word comparison."""

    PROMPT = (
        "You are a helpful assistant. Answer the question based on the provided context. "
        "Always cite your sources and provide well-formed responses."
    )

    def test_line_with_trailing_period_matches(self):
        """'answer the question.' (line) should match 'answer the question' in prompt."""
        line = "Answer the question based on the provided context."
        assert _is_prompt_fragment(line, self.PROMPT) is True

    def test_line_without_punctuation_still_matches(self):
        """Baseline: clean words already matched before the fix."""
        line = "Answer the question based on the provided context"
        assert _is_prompt_fragment(line, self.PROMPT) is True

    def test_hyphenated_word_preserved_as_single_token(self):
        """'well-formed' must NOT be split into 'well' and 'formed'."""
        prompt = (
            "You are a helpful assistant that produces well-formed structured answers "
            "and always includes detailed citations for every claim it makes."
        )
        line = "that produces well-formed structured answers and always includes detailed"
        assert _is_prompt_fragment(line, prompt) is True

    def test_hyphenated_word_split_does_not_produce_false_positive(self):
        """A line with 'well formed' (space) must NOT match a prompt with 'well-formed'."""
        prompt = (
            "You are a helpful assistant that produces well-formed structured answers "
            "and always includes detailed citations for every claim it makes."
        )
        line = "that produces well formed structured answers and always includes detailed"
        assert _is_prompt_fragment(line, prompt) is False

    def test_unicode_smart_quotes_stripped(self):
        """Smart quotes around a phrase should not prevent matching."""
        prompt = (
            "You are a helpful assistant. Always answer the question accurately "
            "and concisely given the retrieved context documents."
        )
        line = "“Always answer the question accurately and concisely given”"
        assert _is_prompt_fragment(line, prompt) is True

    def test_short_line_below_min_length_never_flagged(self):
        """Lines shorter than min_fragment_length are skipped regardless."""
        line = "Answer."
        assert _is_prompt_fragment(line, self.PROMPT) is False

    def test_legitimate_answer_not_flagged(self):
        """An answer that shares a few incidental words is NOT flagged."""
        line = "The answer to this question is forty-two and it is well-known."
        assert _is_prompt_fragment(line, self.PROMPT) is False

    def test_sanitize_answer_removes_punctuated_prompt_fragment(self):
        """sanitize_answer must strip a line that is a punctuated prompt fragment."""
        prompt = (
            "You are a helpful assistant. Answer the question based on the provided "
            "context documents. Always cite your sources when making claims."
        )
        leaked_line = "Answer the question based on the provided context documents."
        answer = f"The capital of France is Paris.\n{leaked_line}\nSee the sources above."
        result = sanitize_answer(answer, system_prompt=prompt)
        assert leaked_line not in result
        assert "The capital of France is Paris." in result


# ---------------------------------------------------------------------------
# Issue #15 — word-boundary checks for _is_boundary_marker
# ---------------------------------------------------------------------------


class TestIsBoundaryMarker:
    """_is_boundary_marker must require column-0 start + structural shape."""

    # True positives — real pipeline artifacts
    def test_document_boundary_exact(self):
        assert _is_boundary_marker("--- Document 1 ---") is True

    def test_document_boundary_double_digit(self):
        assert _is_boundary_marker("--- Document 12 ---") is True

    def test_version_conflict_warning_exact(self):
        assert _is_boundary_marker("--- VERSION CONFLICT WARNING ---") is True

    def test_content_filtered_exact(self):
        assert _is_boundary_marker("[CONTENT_FILTERED]") is True

    # False positives — legitimate prose that embeds these strings
    def test_prose_mentioning_document_boundary_not_flagged(self):
        """A bullet that happens to contain '--- Document ' mid-sentence is NOT a marker."""
        line = "The pipeline inserts --- Document 1 --- separators between chunks."
        assert _is_boundary_marker(line) is False

    def test_indented_document_boundary_not_flagged(self):
        """A line starting with whitespace (column > 0) is not a boundary marker."""
        line = "  --- Document 3 ---"
        assert _is_boundary_marker(line) is False

    def test_document_marker_without_trailing_dashes_not_flagged(self):
        """'--- Document 5' without closing ' ---' is not the real pipeline format."""
        line = "--- Document 5 is the most relevant one."
        assert _is_boundary_marker(line) is False

    def test_document_marker_non_digit_number_not_flagged(self):
        """'--- Document X ---' (non-digit) is not the pipeline format."""
        line = "--- Document X ---"
        assert _is_boundary_marker(line) is False

    def test_sanitize_answer_strips_real_boundary(self):
        answer = "Good answer here.\n--- Document 1 ---\nMore text."
        result = sanitize_answer(answer)
        assert "--- Document 1 ---" not in result
        assert "Good answer here." in result

    def test_sanitize_answer_preserves_prose_with_embedded_marker_text(self):
        """Prose mentioning '--- Document N ---' format must survive sanitization."""
        answer = (
            "The pipeline inserts --- Document 1 --- separators between chunks.\n"
            "This is normal behaviour."
        )
        result = sanitize_answer(answer)
        assert "The pipeline inserts" in result


# ---------------------------------------------------------------------------
# Issue #15 — word-boundary checks for _is_template_artifact
# ---------------------------------------------------------------------------


class TestIsTemplateArtifact:
    """_is_template_artifact must only flag lines that ARE the placeholder."""

    # True positives — lines that are just the unreplaced placeholder
    def test_bare_context_placeholder(self):
        assert _is_template_artifact("{context}") is True

    def test_bare_question_placeholder(self):
        assert _is_template_artifact("{question}") is True

    def test_bare_documents_placeholder(self):
        assert _is_template_artifact("{documents}") is True

    def test_bare_query_placeholder(self):
        assert _is_template_artifact("{query}") is True

    # False positives — legitimate prose that contains the token embedded in text
    def test_prose_mentioning_context_placeholder_not_flagged(self):
        """A bullet explaining that {context} is a variable must NOT be stripped."""
        line = "The system fills in {context} with the retrieved document chunks."
        assert _is_template_artifact(line) is False

    def test_prose_mentioning_question_placeholder_not_flagged(self):
        line = "Replace {question} with the user's actual query before sending."
        assert _is_template_artifact(line) is False

    def test_placeholder_with_surrounding_punctuation_flagged(self):
        """A line that IS just the placeholder plus minimal punctuation is still an artifact."""
        assert _is_template_artifact("{context}.") is True

    def test_sanitize_answer_strips_bare_placeholder(self):
        answer = "The answer is here.\n{context}\nEnd of answer."
        result = sanitize_answer(answer)
        assert "{context}" not in result
        assert "The answer is here." in result

    def test_sanitize_answer_preserves_prose_with_embedded_placeholder(self):
        answer = (
            "Note: the template uses {context} to inject retrieved passages.\n"
            "This is documented behaviour."
        )
        result = sanitize_answer(answer)
        assert "the template uses {context}" in result
