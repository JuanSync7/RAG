from unittest.mock import MagicMock

from src.platform.llm.schemas import LLMResponse
from src.retrieval.generation.nodes.generator import (
    GenerationErrorKind,
    GenerationResult,
    OllamaGenerator,
)


def test_generate_returns_text(monkeypatch):
    mock_provider = MagicMock()
    mock_provider.generate.return_value = LLMResponse(content="ok", model="test")
    mock_provider.config = MagicMock(model="test-model")

    monkeypatch.setattr(
        "src.retrieval.generation.nodes.generator.get_llm_provider", lambda: mock_provider
    )
    generator = OllamaGenerator()
    out = generator.generate("q", ["ctx"])
    assert isinstance(out, GenerationResult)
    assert out.answer == "ok"
    assert out.error is None
    mock_provider.generate.assert_called_once()


def test_generate_recovers_from_empty_after_think(monkeypatch):
    """Reasoning model burns its budget in <think> and returns empty content;
    the empty-after-think fallback retries once with a direct-answer nudge."""
    mock_provider = MagicMock()
    mock_provider.generate.side_effect = [
        LLMResponse(content="", model="test"),                  # over-reasoned: empty final
        LLMResponse(content="The answer is 42.", model="test"),  # retry nudge succeeds
    ]
    mock_provider.config = MagicMock(model="test-model")
    monkeypatch.setattr(
        "src.retrieval.generation.nodes.generator.get_llm_provider", lambda: mock_provider
    )
    out = OllamaGenerator().generate("q", ["ctx"])
    assert mock_provider.generate.call_count == 2  # original + one retry
    assert out.error is None
    assert "42" in out.answer


def test_empty_after_think_retry_nudge_restates_the_question(monkeypatch):
    """B3: the empty-after-think recovery nudge must restate the ACTUAL user
    question so the reasoning model re-answers THIS query on the retry, not a
    generic 'provide your final answer now' that lost the intent."""
    mock_provider = MagicMock()
    mock_provider.generate.side_effect = [
        LLMResponse(content="", model="test"),               # over-reasoned: empty
        LLMResponse(content="grounded answer [1]", model="test"),
    ]
    mock_provider.config = MagicMock(model="test-model")
    monkeypatch.setattr(
        "src.retrieval.generation.nodes.generator.get_llm_provider", lambda: mock_provider
    )
    q = "What is the ARLEN reset value for the DWC AXI bridge on cRoCodile?"
    out = OllamaGenerator().generate(q, ["ctx"])

    assert out.error is None
    assert mock_provider.generate.call_count == 2
    retry_messages = mock_provider.generate.call_args_list[1].args[0]
    # The NUDGE itself (the appended last message, the most salient one) must
    # restate the question — not rely on it being buried in the prior context.
    nudge = retry_messages[-1].get("content", "")
    assert q in nudge


def test_generate_errors_when_empty_after_think_retry_also_empty(monkeypatch):
    """If the retry is ALSO empty, fail gracefully (typed error, empty answer) —
    no infinite retry, no silent blank masquerading as success."""
    mock_provider = MagicMock()
    mock_provider.generate.side_effect = [
        LLMResponse(content="", model="test"),
        LLMResponse(content="", model="test"),
    ]
    mock_provider.config = MagicMock(model="test-model")
    monkeypatch.setattr(
        "src.retrieval.generation.nodes.generator.get_llm_provider", lambda: mock_provider
    )
    out = OllamaGenerator().generate("q", ["ctx"])
    assert mock_provider.generate.call_count == 2
    assert out.error is not None
    assert out.error.kind == GenerationErrorKind.MALFORMED_RESPONSE
    assert out.answer == ""


def test_generate_returns_error_result_on_empty_chunks(monkeypatch):
    mock_provider = MagicMock()
    mock_provider.config = MagicMock(model="test-model")
    monkeypatch.setattr(
        "src.retrieval.generation.nodes.generator.get_llm_provider", lambda: mock_provider
    )
    generator = OllamaGenerator()
    out = generator.generate("q", [])
    assert isinstance(out, GenerationResult)
    assert out.answer == ""
    assert out.error is not None
    mock_provider.generate.assert_not_called()


def test_generate_returns_error_on_failure(monkeypatch):
    mock_provider = MagicMock()
    mock_provider.generate.side_effect = RuntimeError("connection refused")
    mock_provider.config = MagicMock(model="test-model")
    monkeypatch.setattr(
        "src.retrieval.generation.nodes.generator.get_llm_provider", lambda: mock_provider
    )
    generator = OllamaGenerator()
    out = generator.generate("q", ["ctx"])
    assert isinstance(out, GenerationResult)
    assert out.answer == ""
    assert out.error is not None
    assert out.error.kind is GenerationErrorKind.UNKNOWN
