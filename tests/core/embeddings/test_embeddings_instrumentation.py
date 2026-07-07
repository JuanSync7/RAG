"""OTel instrumentation tests for src.core.embeddings."""
from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

from opentelemetry.trace import StatusCode

# tests/conftest.py installs a stubbed langchain_core that only exports
# Embeddings. Evict the stub so the real package is loaded for these tests.
for _mod in list(sys.modules):
    if _mod == "langchain_core" or _mod.startswith("langchain_core."):
        del sys.modules[_mod]

from src.core.embeddings import LocalBGEEmbeddings, TEIEmbeddings


# ---------------------------------------------------------------------------
# LocalBGEEmbeddings
# ---------------------------------------------------------------------------

class _FakeSentenceTransformer:
    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.array([0.1, 0.2, 0.3], dtype=float)
        return np.array([[0.1, 0.2, 0.3] for _ in texts], dtype=float)


def _local(monkeypatch) -> LocalBGEEmbeddings:
    monkeypatch.setattr(
        "src.core.embeddings._load_sentence_transformer",
        lambda _path: _FakeSentenceTransformer(),
    )
    return LocalBGEEmbeddings()


def test_local_embed_documents_emits_batch_generation(otel_capture, monkeypatch):
    """LocalBGEEmbeddings.embed_documents emits 'embeddings.local.batch' generation."""
    emb = _local(monkeypatch)
    vectors = emb.embed_documents(["alpha", "beta longer"])
    assert len(vectors) == 2

    spans = otel_capture.get_finished_spans()
    matching = [s for s in spans if s.name == "embeddings.local.batch"]
    assert len(matching) == 1
    attrs = dict(matching[0].attributes)
    assert attrs["gen_ai.system"] == "sentence-transformers"
    assert attrs["gen_ai.request.model"]  # model path is non-empty
    # gen_ai.usage.input_tokens documented as the sum of text lengths (chars)
    assert attrs["gen_ai.usage.input_tokens"] == len("alpha") + len("beta longer")
    assert attrs["vector_count"] == 2
    assert attrs["vector_dim"] == 3


def test_local_embed_query_emits_query_generation(otel_capture, monkeypatch):
    """LocalBGEEmbeddings.embed_query emits 'embeddings.local.query' generation."""
    emb = _local(monkeypatch)
    vec = emb.embed_query("hello")
    assert len(vec) == 3

    spans = otel_capture.get_finished_spans()
    matching = [s for s in spans if s.name == "embeddings.local.query"]
    assert len(matching) == 1
    attrs = dict(matching[0].attributes)
    assert attrs["gen_ai.system"] == "sentence-transformers"
    assert attrs["gen_ai.usage.input_tokens"] == len("hello")
    assert attrs["vector_count"] == 1
    assert attrs["vector_dim"] == 3


def test_local_embed_documents_records_error(otel_capture, monkeypatch):
    """If the underlying encode raises, the batch span must record ERROR and re-raise."""
    emb = _local(monkeypatch)

    def boom(*_a, **_k):
        raise RuntimeError("encode-fail")
    monkeypatch.setattr(emb.model, "encode", boom)

    with pytest.raises(RuntimeError, match="encode-fail"):
        emb.embed_documents(["x"])
    spans = otel_capture.get_finished_spans()
    matching = [s for s in spans if s.name == "embeddings.local.batch"]
    assert len(matching) == 1
    assert matching[0].status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# TEIEmbeddings
# ---------------------------------------------------------------------------

class _FakeHttpxResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeHttpxClient:
    def __init__(self, *_, **__) -> None:
        self.last_json: dict[str, Any] | None = None
        self.raises: Exception | None = None

    def post(self, _url, json):
        self.last_json = json
        if self.raises is not None:
            raise self.raises
        inputs = json["input"]
        return _FakeHttpxResponse({
            "data": [{"embedding": [0.1, 0.2, 0.3, 0.4]} for _ in inputs]
        })


def _tei(monkeypatch) -> TEIEmbeddings:
    fake = _FakeHttpxClient()
    monkeypatch.setattr("src.core.embeddings.httpx.Client", lambda *a, **k: fake)
    emb = TEIEmbeddings(base_url="http://embed", model="bge-m3", timeout=5)
    emb._fake_client = fake  # type: ignore[attr-defined]
    return emb


def test_tei_embed_documents_emits_batch_generation(otel_capture, monkeypatch):
    """TEIEmbeddings.embed_documents emits 'embeddings.tei.batch'."""
    emb = _tei(monkeypatch)
    vectors = emb.embed_documents(["one", "two"])
    assert len(vectors) == 2 and len(vectors[0]) == 4

    spans = otel_capture.get_finished_spans()
    matching = [s for s in spans if s.name == "embeddings.tei.batch"]
    assert len(matching) == 1
    attrs = dict(matching[0].attributes)
    assert attrs["gen_ai.system"] == "tei"
    assert attrs["gen_ai.request.model"] == "bge-m3"
    assert attrs["gen_ai.usage.input_tokens"] == len("one") + len("two")
    assert attrs["vector_count"] == 2
    assert attrs["vector_dim"] == 4


def test_tei_embed_query_emits_query_generation(otel_capture, monkeypatch):
    """TEIEmbeddings.embed_query emits 'embeddings.tei.query'."""
    emb = _tei(monkeypatch)
    vec = emb.embed_query("hi there")
    assert len(vec) == 4

    spans = otel_capture.get_finished_spans()
    matching = [s for s in spans if s.name == "embeddings.tei.query"]
    assert len(matching) == 1
    attrs = dict(matching[0].attributes)
    assert attrs["gen_ai.system"] == "tei"
    assert attrs["gen_ai.usage.input_tokens"] == len("hi there")
    assert attrs["vector_count"] == 1
    assert attrs["vector_dim"] == 4


def test_tei_embed_documents_records_error(otel_capture, monkeypatch):
    """If the HTTP call raises, the batch span must record ERROR."""
    emb = _tei(monkeypatch)
    emb._fake_client.raises = RuntimeError("http-fail")  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="http-fail"):
        emb.embed_documents(["x"])
    spans = otel_capture.get_finished_spans()
    matching = [s for s in spans if s.name == "embeddings.tei.batch"]
    assert len(matching) == 1
    assert matching[0].status.status_code == StatusCode.ERROR
