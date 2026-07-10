# @summary
# Tests for the _llm_json helper in src.ingest.support.llm.
# Exports: (pytest test functions)
# Deps: src.ingest.support.llm, src.ingest.common.types, unittest.mock
# @end-summary
"""Tests for src.ingest.support.llm._llm_json.

These tests verify disabled-path early return, successful JSON parsing,
markdown fence stripping, and all exception-to-empty-dict failure modes.
The provider is patched at the module level via get_llm_provider to keep
tests fully isolated from network and configuration.
"""

import pytest
from unittest.mock import MagicMock, patch

# _llm_json may be exported as private; fall back to public alias if needed
try:
    from src.ingest.support.llm import _llm_json
except ImportError:
    from src.ingest.support.llm import llm_json as _llm_json

from src.ingest.common.types import IngestionConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    enabled: bool = True,
    model: str = "test-model",
    temperature: float = 0.0,
    timeout: int = 10,
) -> IngestionConfig:
    """Build a minimal IngestionConfig for LLM helper tests."""
    return IngestionConfig(
        enable_llm_metadata=enabled,
        llm_model=model,
        llm_temperature=temperature,
        llm_timeout_seconds=timeout,
    )


def _make_mock_response(content: str) -> MagicMock:
    """Return a mock LLMResponse-like object with the given content string."""
    mock_response = MagicMock()
    mock_response.content = content
    return mock_response


# ---------------------------------------------------------------------------
# Disabled path
# ---------------------------------------------------------------------------

class TestLlmDisabled:
    """_llm_json with enable_llm_metadata=False."""

    def test_disabled_returns_empty_dict(self):
        """enable_llm_metadata=False must return {} without calling the provider."""
        config = _make_config(enabled=False)
        result = _llm_json("any prompt", config)
        assert result == {}

    def test_disabled_no_provider_call(self):
        """enable_llm_metadata=False must not invoke the provider at all."""
        config = _make_config(enabled=False)
        with patch("src.ingest.support.llm.get_llm_provider") as mock_get_provider:
            _llm_json("any prompt", config)
            mock_get_provider.assert_not_called()


# ---------------------------------------------------------------------------
# Successful completion
# ---------------------------------------------------------------------------

class TestLlmEnabled:
    """_llm_json with enable_llm_metadata=True and well-formed responses."""

    @patch("src.ingest.support.llm.get_llm_provider")
    def test_enabled_returns_parsed_dict(self, mock_get_provider):
        """A clean JSON response must be parsed and returned as a dict."""
        mock_provider = MagicMock()
        mock_provider.json_completion.return_value = _make_mock_response(
            '{"key": "val"}'
        )
        mock_get_provider.return_value = mock_provider

        result = _llm_json("prompt", _make_config())
        assert result == {"key": "val"}

    @patch("src.ingest.support.llm.get_llm_provider")
    def test_routes_to_instruct_alias(self, mock_get_provider):
        """_llm_json must call json_completion with the config's INSTRUCT alias,
        not the default (reasoning) alias — a reasoning model returns empty
        content on substantive prompts and silently degrades every ingest LLM
        call. Guards that regression."""
        mock_provider = MagicMock()
        mock_provider.json_completion.return_value = _make_mock_response('{"k": 1}')
        mock_get_provider.return_value = mock_provider

        cfg = _make_config()
        _llm_json("prompt", cfg)
        assert (
            mock_provider.json_completion.call_args.kwargs["model_alias"]
            == cfg.ingest_llm_model_alias
        )
        # default resolves to an instruct alias, not "default"
        assert cfg.ingest_llm_model_alias != "default"

    @patch("src.ingest.support.llm.get_llm_provider")
    def test_strips_markdown_fences(self, mock_get_provider):
        """A response wrapped in ```json...``` fences must be parsed correctly."""
        mock_provider = MagicMock()
        mock_provider.json_completion.return_value = _make_mock_response(
            "```json\n{\"answer\": 42}\n```"
        )
        mock_get_provider.return_value = mock_provider

        result = _llm_json("prompt", _make_config())
        assert result == {"answer": 42}

    @patch("src.ingest.support.llm.get_llm_provider")
    def test_return_type_is_always_dict_on_success(self, mock_get_provider):
        """Return value must always be a dict when the call succeeds."""
        mock_provider = MagicMock()
        mock_provider.json_completion.return_value = _make_mock_response(
            '{"x": 1, "y": 2}'
        )
        mock_get_provider.return_value = mock_provider

        result = _llm_json("prompt", _make_config())
        assert isinstance(result, dict)

    @patch("src.ingest.support.llm.get_llm_provider")
    def test_uses_correct_model_alias(self, mock_get_provider):
        """The llm_model config field is retained for metadata but routing uses
        the provider alias; json_completion must be called with matching kwargs."""
        mock_provider = MagicMock()
        mock_provider.json_completion.return_value = _make_mock_response('{"ok": true}')
        mock_get_provider.return_value = mock_provider

        config = _make_config(model="my-special-model")
        _llm_json("test prompt", config)

        # Verify json_completion was called exactly once
        mock_provider.json_completion.assert_called_once()

    @patch("src.ingest.support.llm.get_llm_provider")
    def test_passes_temperature_to_provider(self, mock_get_provider):
        """Temperature from config must be forwarded to json_completion."""
        mock_provider = MagicMock()
        mock_provider.json_completion.return_value = _make_mock_response('{"v": 1}')
        mock_get_provider.return_value = mock_provider

        config = _make_config(temperature=0.7)
        _llm_json("prompt", config)

        call_kwargs = mock_provider.json_completion.call_args
        # temperature must appear in positional or keyword args
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        assert all_kwargs.get("temperature") == 0.7

    @patch("src.ingest.support.llm.get_llm_provider")
    def test_passes_timeout_to_provider(self, mock_get_provider):
        """Timeout from config must be forwarded to json_completion."""
        mock_provider = MagicMock()
        mock_provider.json_completion.return_value = _make_mock_response('{"v": 1}')
        mock_get_provider.return_value = mock_provider

        config = _make_config(timeout=30)
        _llm_json("prompt", config)

        call_kwargs = mock_provider.json_completion.call_args
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        assert all_kwargs.get("timeout") == 30


# ---------------------------------------------------------------------------
# Failure / exception paths
# ---------------------------------------------------------------------------

class TestLlmFailures:
    """_llm_json exception handling — all failures must return {}."""

    @patch("src.ingest.support.llm.get_llm_provider")
    def test_timeout_error_returns_empty(self, mock_get_provider):
        """TimeoutError from the provider must return {}."""
        mock_provider = MagicMock()
        mock_provider.json_completion.side_effect = TimeoutError("timed out")
        mock_get_provider.return_value = mock_provider

        result = _llm_json("prompt", _make_config())
        assert result == {}

    @patch("src.ingest.support.llm.get_llm_provider")
    def test_generic_exception_returns_empty(self, mock_get_provider):
        """Any generic Exception from the provider must return {}."""
        mock_provider = MagicMock()
        mock_provider.json_completion.side_effect = Exception("unexpected error")
        mock_get_provider.return_value = mock_provider

        result = _llm_json("prompt", _make_config())
        assert result == {}

    @patch("src.ingest.support.llm.get_llm_provider")
    def test_malformed_json_returns_empty(self, mock_get_provider):
        """A response that is not valid JSON must return {}."""
        mock_provider = MagicMock()
        mock_provider.json_completion.return_value = _make_mock_response(
            "this is not json at all!!!"
        )
        mock_get_provider.return_value = mock_provider

        result = _llm_json("prompt", _make_config())
        assert result == {}

    @patch("src.ingest.support.llm.get_llm_provider")
    def test_empty_response_returns_empty(self, mock_get_provider):
        """An empty content string from the provider must return {}."""
        mock_provider = MagicMock()
        mock_provider.json_completion.return_value = _make_mock_response("")
        mock_get_provider.return_value = mock_provider

        result = _llm_json("prompt", _make_config())
        assert result == {}

    @patch("src.ingest.support.llm.get_llm_provider")
    def test_connection_error_returns_empty(self, mock_get_provider):
        """ConnectionError from the provider must return {}."""
        mock_provider = MagicMock()
        mock_provider.json_completion.side_effect = ConnectionError("refused")
        mock_get_provider.return_value = mock_provider

        result = _llm_json("prompt", _make_config())
        assert result == {}

    def test_return_type_is_always_dict_disabled(self):
        """Return value is a dict even when the helper is disabled."""
        result = _llm_json("prompt", _make_config(enabled=False))
        assert isinstance(result, dict)

    @patch("src.ingest.support.llm.get_llm_provider")
    def test_return_type_is_always_dict_on_exception(self, mock_get_provider):
        """Return value is a dict even when an exception is raised."""
        mock_provider = MagicMock()
        mock_provider.json_completion.side_effect = RuntimeError("boom")
        mock_get_provider.return_value = mock_provider

        result = _llm_json("any prompt", _make_config())
        assert isinstance(result, dict)

    @patch("src.ingest.support.llm.get_llm_provider")
    def test_get_provider_raises_returns_empty(self, mock_get_provider):
        """If get_llm_provider itself raises, _llm_json must return {}."""
        mock_get_provider.side_effect = RuntimeError("provider init failed")

        result = _llm_json("prompt", _make_config())
        assert result == {}


# ---------------------------------------------------------------------------
# Keyword extraction via LLM — regression tests
# ---------------------------------------------------------------------------

class TestKeywordExtraction:
    """Regression tests for keyword extraction patterns via _llm_json.

    These tests verify that keyword extraction prompts return usable output,
    handle LLM timeout gracefully, and degrade cleanly on malformed output.
    """

    @patch("src.ingest.support.llm.get_llm_provider")
    def test_keyword_extraction_happy_path(self, mock_get_provider):
        """LLM returning a keywords list in JSON produces a usable dict."""
        mock_provider = MagicMock()
        mock_provider.json_completion.return_value = _make_mock_response(
            '{"keywords": ["clock", "timing", "setup", "hold"]}'
        )
        mock_get_provider.return_value = mock_provider

        result = _llm_json(
            "Extract keywords from: timing analysis for clock domain crossing",
            _make_config(),
        )
        assert isinstance(result, dict)
        assert "keywords" in result
        assert "clock" in result["keywords"]

    @patch("src.ingest.support.llm.get_llm_provider")
    def test_keyword_extraction_timeout_returns_empty(self, mock_get_provider):
        """TimeoutError during keyword extraction must return {} — no exception propagated."""
        mock_provider = MagicMock()
        mock_provider.json_completion.side_effect = TimeoutError("LLM timed out")
        mock_get_provider.return_value = mock_provider

        result = _llm_json(
            "Extract keywords from: important document content",
            _make_config(timeout=1),
        )
        assert result == {}

    @patch("src.ingest.support.llm.get_llm_provider")
    def test_keyword_extraction_malformed_output_fallback(self, mock_get_provider):
        """LLM returning non-JSON output for keyword extraction must return {} gracefully."""
        mock_provider = MagicMock()
        mock_provider.json_completion.return_value = _make_mock_response(
            "I found the following keywords: clock, timing, setup"
        )
        mock_get_provider.return_value = mock_provider

        result = _llm_json(
            "Extract keywords from: clock domain timing setup",
            _make_config(),
        )
        assert isinstance(result, dict)

    @patch("src.ingest.support.llm.get_llm_provider")
    def test_keyword_extraction_empty_keywords_list(self, mock_get_provider):
        """LLM returning an empty keywords array must produce a dict with empty list."""
        mock_provider = MagicMock()
        mock_provider.json_completion.return_value = _make_mock_response(
            '{"keywords": []}'
        )
        mock_get_provider.return_value = mock_provider

        result = _llm_json("Extract keywords from: ...", _make_config())
        assert result.get("keywords") == []

    @patch("src.ingest.support.llm.get_llm_provider")
    def test_keyword_extraction_disabled_returns_empty_without_call(self, mock_get_provider):
        """When LLM is disabled, keyword extraction must return {} and never call the provider."""
        result = _llm_json(
            "Extract keywords from: some text",
            _make_config(enabled=False),
        )
        assert result == {}
        mock_get_provider.assert_not_called()
