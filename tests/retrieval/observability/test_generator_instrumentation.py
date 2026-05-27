"""OTel instrumentation contract for OllamaGenerator."""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch, MagicMock

from src.retrieval.generation.nodes.generator import (
    OllamaGenerator,
    TokenEvent,
    ErrorEvent,
)


def _spans_named(exporter, name):
    return [s for s in exporter.get_finished_spans() if s.name == name]


@dataclass
class _Resp:
    content: str = '{"answer": "the answer", "confidence": "high"}'
    prompt_tokens: int = 5
    completion_tokens: int = 7


class _FakeProvider:
    def __init__(self):
        from types import SimpleNamespace
        self.config = SimpleNamespace(model="fake-model")
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return _Resp()

    def generate_stream(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        for tok in ["he", "llo", " world"]:
            yield tok

    def is_available(self, **kwargs):
        return True


def _mk_gen(monkeypatch):
    fake = _FakeProvider()
    monkeypatch.setattr(
        "src.retrieval.generation.nodes.generator.get_llm_provider",
        lambda: fake,
    )
    return OllamaGenerator(), fake


def test_generate_emits_retrieval_generation_answer(otel_capture, monkeypatch):
    gen, _ = _mk_gen(monkeypatch)
    res = gen.generate(
        query="q?",
        context_chunks=["chunk a", "chunk b"],
        scores=[0.9, 0.5],
    )
    assert res.answer == "the answer"

    spans = _spans_named(otel_capture, "retrieval.generation.answer")
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs["gen_ai.system"] == "ollama"
    assert attrs["gen_ai.request.model"] == "fake-model"
    assert attrs["context_chunk_count"] == 2
    assert attrs["doc_count"] == 2
    assert attrs["context_len"] == len("chunk a") + len("chunk b")
    assert attrs["llm_confidence"] == "high"


def test_generate_empty_context_no_span(otel_capture, monkeypatch):
    gen, _ = _mk_gen(monkeypatch)
    res = gen.generate(query="q?", context_chunks=[])
    assert res.error is not None
    assert _spans_named(otel_capture, "retrieval.generation.answer") == []


def test_stream_emits_retrieval_generation_stream(otel_capture, monkeypatch):
    gen, _ = _mk_gen(monkeypatch)
    events = list(gen.generate_stream(query="q?", context_chunks=["x"]))
    tokens = [e for e in events if isinstance(e, TokenEvent)]
    assert len(tokens) == 3

    spans = _spans_named(otel_capture, "retrieval.generation.stream")
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs["gen_ai.system"] == "ollama"
    assert attrs["gen_ai.request.model"] == "fake-model"
    assert attrs["stream_chunk_count"] == 3
    assert attrs["doc_count"] == 1


def test_is_available_emits_span(otel_capture, monkeypatch):
    gen, _ = _mk_gen(monkeypatch)
    assert gen.is_available() is True
    spans = _spans_named(otel_capture, "retrieval.generation.is_available")
    assert len(spans) == 1
    assert spans[0].attributes["gen_ai.system"] == "ollama"
