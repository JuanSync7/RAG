from unittest.mock import MagicMock

from src.platform.llm.schemas import LLMResponse
from src.retrieval.generation.nodes.generator import OllamaGenerator


def test_generate_returns_text(monkeypatch):
    mock_provider = MagicMock()
    mock_provider.generate.return_value = LLMResponse(content="ok", model="test")
    mock_provider.config = MagicMock(model="test-model")

    monkeypatch.setattr(
        "src.retrieval.generation.nodes.generator.get_llm_provider", lambda: mock_provider
    )
    generator = OllamaGenerator()
    out = generator.generate("q", ["ctx"])
    assert out == "ok"
    mock_provider.generate.assert_called_once()


def test_generate_returns_none_on_empty_chunks(monkeypatch):
    mock_provider = MagicMock()
    mock_provider.config = MagicMock(model="test-model")
    monkeypatch.setattr(
        "src.retrieval.generation.nodes.generator.get_llm_provider", lambda: mock_provider
    )
    generator = OllamaGenerator()
    out = generator.generate("q", [])
    assert out is None
    mock_provider.generate.assert_not_called()


def test_generate_returns_none_on_failure(monkeypatch):
    mock_provider = MagicMock()
    mock_provider.generate.side_effect = RuntimeError("connection refused")
    mock_provider.config = MagicMock(model="test-model")
    monkeypatch.setattr(
        "src.retrieval.generation.nodes.generator.get_llm_provider", lambda: mock_provider
    )
    generator = OllamaGenerator()
    out = generator.generate("q", ["ctx"])
    assert out is None
