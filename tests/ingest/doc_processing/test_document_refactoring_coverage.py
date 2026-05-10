# @summary
# Coverage tests for the document_refactoring_node pipeline node.
# Exports: TestDocumentRefactoringError, TestDocumentRefactoringBoundary, TestDocumentRefactoringErrorScenarios
# Deps: pytest, unittest.mock, src.ingest.doc_processing.nodes.document_refactoring,
#       src.ingest.common.types
# @end-summary

"""Coverage tests for document_refactoring_node.

Tests are grouped into three classes:
- TestDocumentRefactoringError: fallback paths when LLM returns empty/missing keys.
- TestDocumentRefactoringBoundary: edge-case inputs (disabled, empty text, oversized text, None).
- TestDocumentRefactoringErrorScenarios: happy-path, whitespace, coercion, log, and prompt-format checks.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.ingest.doc_processing.nodes.document_refactoring import document_refactoring_node
from src.ingest.common.types import IngestionConfig, Runtime


# ---------------------------------------------------------------------------
# Patch target
# ---------------------------------------------------------------------------

_PATCH_TARGET = "src.ingest.doc_processing.nodes.document_refactoring._llm_json"

# Mirror the constants from the module under test so assertions stay DRY.
_REFACTOR_PROMPT = 'Return {"refactored_text":"..."} for:\n'
_MAX_REFACTOR_INPUT = 10000
_REFACTOR_MAX_TOKENS = 900


# ---------------------------------------------------------------------------
# State builder
# ---------------------------------------------------------------------------

def _make_state(enable_refactoring: bool = True, cleaned_text: str = "This is cleaned text.", **overrides) -> dict:
    """Build a minimal pipeline state dict for document_refactoring_node tests.

    Args:
        enable_refactoring: Value for ``IngestionConfig.enable_document_refactoring``.
        cleaned_text: Value for ``state["cleaned_text"]``.
        **overrides: Additional keys merged into the returned state dict.

    Returns:
        A state dict compatible with ``DocumentProcessingState``.
    """
    config = IngestionConfig(enable_document_refactoring=enable_refactoring)
    runtime = Runtime(
        config=config,
        embedder=MagicMock(),
        weaviate_client=MagicMock(),
        kg_builder=None,
    )
    state = {
        "runtime": runtime,
        "cleaned_text": cleaned_text,
        "errors": [],
        "processing_log": [],
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# TestDocumentRefactoringError
# ---------------------------------------------------------------------------

class TestDocumentRefactoringError:
    """Fallback paths when the LLM response lacks a usable refactored_text value."""

    def test_refactoring_node_falls_back_to_cleaned_text_when_llm_returns_empty(self):
        """An empty string from the LLM must cause the node to fall back to cleaned_text."""
        state = _make_state(enable_refactoring=True, cleaned_text="original")
        with patch(_PATCH_TARGET, return_value={"refactored_text": ""}):
            result = document_refactoring_node(state)

        assert result["refactored_text"] == "original"

    def test_refactoring_node_falls_back_to_cleaned_text_when_llm_returns_no_key(self):
        """A response dict without 'refactored_text' must fall back to cleaned_text."""
        state = _make_state(enable_refactoring=True, cleaned_text="fallback text")
        with patch(_PATCH_TARGET, return_value={}):
            result = document_refactoring_node(state)

        assert result["refactored_text"] == "fallback text"

    def test_refactoring_node_falls_back_to_cleaned_text_when_llm_json_fails(self):
        """An empty dict response (simulating LLM JSON failure) must fall back to cleaned_text."""
        state = _make_state(enable_refactoring=True, cleaned_text="safe fallback")
        with patch(_PATCH_TARGET, return_value={}):
            result = document_refactoring_node(state)

        assert result["refactored_text"] == "safe fallback"


# ---------------------------------------------------------------------------
# TestDocumentRefactoringBoundary
# ---------------------------------------------------------------------------

class TestDocumentRefactoringBoundary:
    """Edge-case inputs at the boundary of valid values."""

    def test_refactoring_node_skipped_when_disabled(self):
        """When disabled, the node must return cleaned_text unchanged."""
        state = _make_state(enable_refactoring=False, cleaned_text="original text")
        result = document_refactoring_node(state)

        assert result["refactored_text"] == "original text"
        assert result["processing_log"][-1].endswith("document_refactoring:skipped")

    def test_refactoring_node_no_llm_call_when_disabled(self):
        """When disabled, the node must not call _llm_json at all."""
        state = _make_state(enable_refactoring=False)
        with patch(_PATCH_TARGET) as mock_llm:
            document_refactoring_node(state)

        mock_llm.assert_not_called()

    def test_refactoring_node_handles_empty_cleaned_text(self):
        """An empty cleaned_text must still yield the LLM result when it is non-empty."""
        state = _make_state(enable_refactoring=True, cleaned_text="")
        with patch(_PATCH_TARGET, return_value={"refactored_text": "LLM result"}):
            result = document_refactoring_node(state)

        assert result["refactored_text"] == "LLM result"

    def test_refactoring_node_truncates_prompt_to_10000_chars(self):
        """The prompt passed to _llm_json must not contain more than 10000 chars of cleaned_text."""
        long_text = "A" * 15000
        state = _make_state(enable_refactoring=True, cleaned_text=long_text)
        with patch(_PATCH_TARGET, return_value={"refactored_text": "ok"}) as mock_llm:
            document_refactoring_node(state)

        call_args = mock_llm.call_args
        prompt = call_args[0][0]
        max_expected_len = len(_REFACTOR_PROMPT) + _MAX_REFACTOR_INPUT
        assert len(prompt) <= max_expected_len

    def test_refactoring_node_handles_none_cleaned_text(self):
        """Passing cleaned_text=None when enabled must raise an exception — documents missing None guard."""
        state = _make_state(enable_refactoring=True, cleaned_text=None)
        with pytest.raises(Exception):
            document_refactoring_node(state)


# ---------------------------------------------------------------------------
# TestDocumentRefactoringErrorScenarios
# ---------------------------------------------------------------------------

class TestDocumentRefactoringErrorScenarios:
    """Happy-path, whitespace handling, type coercion, log recording, and prompt-format checks."""

    def test_refactoring_node_returns_llm_refactored_text_on_success(self):
        """A valid LLM response must be returned as refactored_text."""
        state = _make_state(enable_refactoring=True)
        with patch(_PATCH_TARGET, return_value={"refactored_text": "Refactored content here"}):
            result = document_refactoring_node(state)

        assert result["refactored_text"] == "Refactored content here"

    def test_refactoring_node_strips_whitespace_from_llm_response(self):
        """Leading/trailing whitespace in the LLM response must be stripped."""
        state = _make_state(enable_refactoring=True)
        with patch(_PATCH_TARGET, return_value={"refactored_text": "  text with spaces  "}):
            result = document_refactoring_node(state)

        assert result["refactored_text"] == "text with spaces"

    def test_refactoring_node_coerces_non_string_response_to_string(self):
        """A non-string refactored_text (e.g. int) must be coerced via str() before stripping."""
        state = _make_state(enable_refactoring=True)
        with patch(_PATCH_TARGET, return_value={"refactored_text": 42}):
            result = document_refactoring_node(state)

        assert result["refactored_text"] == "42"

    def test_refactoring_node_processing_log_records_ok_on_success(self):
        """A successful refactoring run must append 'document_refactoring:ok' as the last log entry."""
        state = _make_state(enable_refactoring=True)
        with patch(_PATCH_TARGET, return_value={"refactored_text": "ok"}):
            result = document_refactoring_node(state)

        assert result["processing_log"][-1].endswith("document_refactoring:ok")

    def test_refactoring_node_processing_log_records_skipped_when_disabled(self):
        """A skipped refactoring run must append 'document_refactoring:skipped' as the last log entry."""
        state = _make_state(enable_refactoring=False)
        result = document_refactoring_node(state)

        assert result["processing_log"][-1].endswith("document_refactoring:skipped")

    def test_refactoring_node_prompt_format(self):
        """The prompt passed to _llm_json must start with the expected prefix followed by cleaned_text[:10000]."""
        cleaned = "Sample document text."
        state = _make_state(enable_refactoring=True, cleaned_text=cleaned)
        with patch(_PATCH_TARGET, return_value={"refactored_text": "done"}) as mock_llm:
            document_refactoring_node(state)

        call_args = mock_llm.call_args
        prompt = call_args[0][0]
        assert prompt.startswith(_REFACTOR_PROMPT)
        assert prompt == _REFACTOR_PROMPT + cleaned[:_MAX_REFACTOR_INPUT]

    def test_refactoring_node_max_tokens_is_900(self):
        """The third positional argument passed to _llm_json must be 900."""
        state = _make_state(enable_refactoring=True)
        with patch(_PATCH_TARGET, return_value={"refactored_text": "result"}) as mock_llm:
            document_refactoring_node(state)

        call_args = mock_llm.call_args
        assert call_args[0][2] == _REFACTOR_MAX_TOKENS
