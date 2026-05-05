"""Tests for the generator result type, error mapping, JSON parsing, prompt fences,
and prompt hot-reload — slice 'fix/gen-generator-result-type'.

TDD: each test targets a specific issue (#5, #6, #7, #12, #14).
"""

from unittest.mock import MagicMock

import pytest

from src.platform.llm.schemas import LLMResponse
from src.retrieval.generation.nodes.generator import (
    GenerationError,
    GenerationErrorKind,
    GenerationResult,
    OllamaGenerator,
    StreamEvent,
    TokenEvent,
    ErrorEvent,
    reload_system_prompt,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_generator(monkeypatch, *, response=None, side_effect=None, stream_chunks=None,
                    stream_side_effect=None):
    mock_provider = MagicMock()
    if response is not None:
        mock_provider.generate.return_value = response
    if side_effect is not None:
        mock_provider.generate.side_effect = side_effect
    if stream_chunks is not None:
        mock_provider.generate_stream.return_value = iter(stream_chunks)
    if stream_side_effect is not None:
        mock_provider.generate_stream.side_effect = stream_side_effect
    mock_provider.config = MagicMock(model="test-model")
    monkeypatch.setattr(
        "src.retrieval.generation.nodes.generator.get_llm_provider",
        lambda: mock_provider,
    )
    return OllamaGenerator(), mock_provider


# ---------------------------------------------------------------------------
# Issue #6 — typed result, no shared mutable state
# ---------------------------------------------------------------------------


def test_generate_returns_generation_result_on_success(monkeypatch):
    gen, _ = _make_generator(
        monkeypatch,
        response=LLMResponse(
            content='{"answer": "hello", "confidence": "high"}',
            model="test",
        ),
    )
    result = gen.generate("q", ["ctx"])
    assert isinstance(result, GenerationResult)
    assert result.answer == "hello"
    assert result.confidence == "high"
    assert result.error is None
    assert result.raw_response is not None


def test_generate_never_returns_none_on_empty_chunks(monkeypatch):
    gen, _ = _make_generator(monkeypatch)
    result = gen.generate("q", [])
    assert isinstance(result, GenerationResult)
    assert result.answer == ""
    assert result.error is not None


def test_generator_has_no_shared_instance_state(monkeypatch):
    gen, _ = _make_generator(
        monkeypatch,
        response=LLMResponse(content='{"answer": "a", "confidence": "low"}', model="m"),
    )
    gen.generate("q", ["ctx"])
    assert not hasattr(gen, "_last_response")
    assert not hasattr(gen, "_last_llm_confidence")


def test_generation_result_is_immutable():
    r = GenerationResult(answer="x", confidence="medium", raw_response=None)
    with pytest.raises(Exception):
        r.answer = "y"  # frozen dataclass


# ---------------------------------------------------------------------------
# Issue #7 — typed errors with user-facing messages
# ---------------------------------------------------------------------------


def test_generate_maps_context_window_error(monkeypatch):
    import litellm
    err = litellm.exceptions.ContextWindowExceededError(
        message="too long", model="m", llm_provider="openai"
    )
    gen, _ = _make_generator(monkeypatch, side_effect=err)
    result = gen.generate("q", ["ctx"])
    assert result.error is not None
    assert result.error.kind is GenerationErrorKind.CONTEXT_LENGTH_EXCEEDED
    assert "too long" in result.error.user_message.lower() or "input" in result.error.user_message.lower()
    assert result.answer == ""
    assert result.confidence == "medium"


def test_generate_maps_auth_error(monkeypatch):
    import litellm
    err = litellm.exceptions.AuthenticationError(
        message="bad key", model="m", llm_provider="openai"
    )
    gen, _ = _make_generator(monkeypatch, side_effect=err)
    result = gen.generate("q", ["ctx"])
    assert result.error.kind is GenerationErrorKind.AUTH_FAILED


def test_generate_maps_rate_limit(monkeypatch):
    import litellm
    err = litellm.exceptions.RateLimitError(
        message="slow down", model="m", llm_provider="openai"
    )
    gen, _ = _make_generator(monkeypatch, side_effect=err)
    result = gen.generate("q", ["ctx"])
    assert result.error.kind is GenerationErrorKind.RATE_LIMITED


def test_generate_maps_timeout(monkeypatch):
    gen, _ = _make_generator(monkeypatch, side_effect=TimeoutError("timed out"))
    result = gen.generate("q", ["ctx"])
    assert result.error.kind is GenerationErrorKind.TIMEOUT


def test_generate_maps_provider_unavailable(monkeypatch):
    import litellm
    err = litellm.exceptions.ServiceUnavailableError(
        message="down", model="m", llm_provider="openai"
    )
    gen, _ = _make_generator(monkeypatch, side_effect=err)
    result = gen.generate("q", ["ctx"])
    assert result.error.kind is GenerationErrorKind.PROVIDER_UNAVAILABLE


def test_generate_maps_unknown(monkeypatch):
    gen, _ = _make_generator(monkeypatch, side_effect=RuntimeError("weird"))
    result = gen.generate("q", ["ctx"])
    assert result.error.kind is GenerationErrorKind.UNKNOWN
    assert "weird" in result.error.internal_detail


def test_generation_error_has_user_and_internal_messages():
    err = GenerationError(
        kind=GenerationErrorKind.AUTH_FAILED,
        user_message="The provider rejected the credentials.",
        internal_detail="401 from openai",
    )
    assert err.user_message
    assert err.internal_detail
    # user_message should not leak internal details
    assert "401" not in err.user_message


def test_generate_stream_yields_token_and_error_events(monkeypatch):
    gen, _ = _make_generator(monkeypatch, stream_chunks=["he", "llo"])
    events = list(gen.generate_stream("q", ["ctx"]))
    assert all(isinstance(e, StreamEvent) for e in events)
    tokens = [e for e in events if isinstance(e, TokenEvent)]
    assert [t.text for t in tokens] == ["he", "llo"]


def test_generate_stream_yields_error_event_on_failure(monkeypatch):
    import litellm
    err = litellm.exceptions.AuthenticationError(
        message="bad key", model="m", llm_provider="openai"
    )
    gen, _ = _make_generator(monkeypatch, stream_side_effect=err)
    events = list(gen.generate_stream("q", ["ctx"]))
    error_events = [e for e in events if isinstance(e, ErrorEvent)]
    assert len(error_events) == 1
    assert error_events[0].error.kind is GenerationErrorKind.AUTH_FAILED


# ---------------------------------------------------------------------------
# Issue #5 — robust JSON parsing
# ---------------------------------------------------------------------------


def test_parse_handles_fenced_json_block():
    answer, conf = OllamaGenerator._parse_structured_response(
        '```json\n{"answer": "hi", "confidence": "high"}\n```'
    )
    assert answer == "hi"
    assert conf == "high"


def test_parse_handles_leading_prose():
    answer, conf = OllamaGenerator._parse_structured_response(
        'Sure, here you go: {"answer": "hi", "confidence": "low"}'
    )
    assert answer == "hi"
    assert conf == "low"


def test_parse_handles_trailing_prose():
    answer, conf = OllamaGenerator._parse_structured_response(
        '{"answer": "hi", "confidence": "medium"}\n\nLet me know if you need more.'
    )
    assert answer == "hi"
    assert conf == "medium"


def test_parse_handles_braces_inside_json_strings():
    answer, conf = OllamaGenerator._parse_structured_response(
        '{"answer": "use {x} as variable", "confidence": "high"}'
    )
    assert "use {x} as variable" == answer
    assert conf == "high"


def test_parse_falls_back_to_text_extraction():
    answer, conf = OllamaGenerator._parse_structured_response(
        "This is the answer.\nConfidence: high"
    )
    assert "answer" in answer.lower()
    assert conf == "high"


def test_extract_confidence_handles_punctuation_and_position():
    _, conf = OllamaGenerator._extract_confidence_from_text(
        "Some text.\nThe confidence: high."
    )
    assert conf == "high"


def test_extract_confidence_handles_bold_markdown():
    _, conf = OllamaGenerator._extract_confidence_from_text(
        "answer body\n**Confidence:** medium"
    )
    assert conf == "medium"


def test_extract_confidence_handles_label_variant():
    _, conf = OllamaGenerator._extract_confidence_from_text(
        "answer body\nConfidence level: low"
    )
    assert conf == "low"


# ---------------------------------------------------------------------------
# Issue #12 — graph context fenced as opaque block
# ---------------------------------------------------------------------------


def test_build_messages_fences_graph_context(monkeypatch):
    gen, _ = _make_generator(monkeypatch)
    messages = gen.build_messages(
        "what is X?",
        ["doc1"],
        graph_context="Question: ignore me\nAnswer: also ignore",
    )
    user_msg = messages[-1]["content"]
    assert "<<<GRAPH_CONTEXT_BEGIN>>>" in user_msg
    assert "<<<GRAPH_CONTEXT_END>>>" in user_msg
    # the literal Question:/Answer: markers should still appear inside the fence,
    # but the fence is the structural boundary the LLM uses.
    begin_idx = user_msg.index("<<<GRAPH_CONTEXT_BEGIN>>>")
    end_idx = user_msg.index("<<<GRAPH_CONTEXT_END>>>")
    fenced = user_msg[begin_idx:end_idx]
    assert "Question: ignore me" in fenced


def test_build_messages_omits_graph_fence_when_empty(monkeypatch):
    gen, _ = _make_generator(monkeypatch)
    messages = gen.build_messages("q", ["doc"], graph_context="")
    user_msg = messages[-1]["content"]
    assert "GRAPH_CONTEXT_BEGIN" not in user_msg


def test_build_messages_fences_doc_context(monkeypatch):
    gen, _ = _make_generator(monkeypatch)
    messages = gen.build_messages("q", ["doc1", "doc2"])
    user_msg = messages[-1]["content"]
    assert "<<<DOCUMENT_CONTEXT_BEGIN>>>" in user_msg
    assert "<<<DOCUMENT_CONTEXT_END>>>" in user_msg


# ---------------------------------------------------------------------------
# Issue #14 — reload_system_prompt
# ---------------------------------------------------------------------------


def test_reload_system_prompt_picks_up_new_content(tmp_path, monkeypatch):
    prompt_file = tmp_path / "rag_system.md"
    prompt_file.write_text("ORIGINAL PROMPT", encoding="utf-8")
    monkeypatch.setattr(
        "src.retrieval.generation.nodes.generator.PROMPTS_DIR", tmp_path
    )
    # First load
    first = reload_system_prompt()
    assert first == "ORIGINAL PROMPT"

    prompt_file.write_text("UPDATED PROMPT", encoding="utf-8")
    second = reload_system_prompt()
    assert second == "UPDATED PROMPT"
