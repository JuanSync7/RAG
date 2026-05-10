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
