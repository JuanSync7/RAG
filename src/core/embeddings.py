# @summary
# Embedding providers: local BAAI/bge-m3 (in-process) and TEI over HTTP.
# Exports: LocalBGEEmbeddings, TEIEmbeddings, get_embedding_provider
# Deps: sentence-transformers (local path only), httpx, numpy, langchain_core, config.settings
# @end-summary
"""Embedding provider implementations and factory.

Two backends:
  local — BAAI/bge-m3 loaded in-process via sentence-transformers (dev venv only;
          requires the `local-embed` pyproject extra).
  tei   — BAAI/bge-m3 served by a separate TEI container (rag-embed) over HTTP.
"""
from __future__ import annotations


import httpx
import numpy as np
from langchain_core.embeddings import Embeddings

from config.settings import (
    EMBEDDING_MODEL_PATH,
    INFERENCE_BACKEND,
    TEI_EMBED_URL,
    TEI_EMBEDDING_MODEL,
    TEI_TIMEOUT_SECONDS,
)
from src.platform.observability import get_tracer


def _load_sentence_transformer(model_path: str):
    """Lazy import so the worker image can run without sentence-transformers when using TEI."""
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    return SentenceTransformer(model_path)


class LocalBGEEmbeddings(Embeddings):
    """LangChain-compatible embeddings using a local BAAI/bge-m3 model."""

    def __init__(self, model_path: str = EMBEDDING_MODEL_PATH):
        self.model = _load_sentence_transformer(model_path)
        self.tracer = get_tracer()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts."""
        span = self.tracer.start_span("embeddings.embed_documents", {"batch_size": len(texts)})
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )
        span.end(status="ok")
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        span = self.tracer.start_span("embeddings.embed_query", {"text_len": len(text)})
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )
        span.end(status="ok")
        return embedding.tolist()

    def encode_sentences(self, sentences: list[str]) -> np.ndarray:
        """Encode sentences returning numpy array for internal use.

        Used by semantic chunking for cosine similarity computation.
        Returns L2-normalized embeddings so cosine sim = dot product.
        """
        return self.model.encode(
            sentences,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        )


class TEIEmbeddings(Embeddings):
    """LangChain-compatible embeddings backed by a TEI container over HTTP.

    Calls TEI's OpenAI-compatible ``/v1/embeddings`` endpoint. TEI normalizes
    outputs for sentence-transformer-family models (including BGE-M3), so the
    vectors returned are already L2-unit and match the contract of
    :class:`LocalBGEEmbeddings`.
    """

    def __init__(
        self,
        base_url: str = TEI_EMBED_URL,
        model: str = TEI_EMBEDDING_MODEL,
        timeout: int = TEI_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=timeout)
        self.tracer = get_tracer()

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        resp = self._client.post(
            f"{self.base_url}/v1/embeddings",
            json={"model": self.model, "input": inputs},
        )
        resp.raise_for_status()
        return [d["embedding"] for d in resp.json()["data"]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents via TEI."""
        span = self.tracer.start_span("embeddings.embed_documents", {"batch_size": len(texts)})
        try:
            vectors = self._embed(texts)
        finally:
            span.end(status="ok")
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text via TEI."""
        span = self.tracer.start_span("embeddings.embed_query", {"text_len": len(text)})
        try:
            vectors = self._embed([text])
        finally:
            span.end(status="ok")
        return vectors[0]

    def encode_sentences(self, sentences: list[str]) -> np.ndarray:
        """Return L2-normalized embeddings as a numpy array for semantic chunking.

        TEI normalizes BGE-family embeddings by default, so no post-hoc norm needed.
        """
        return np.array(self._embed(sentences))


def get_embedding_provider() -> Embeddings:
    """Return the configured embedding provider.

    Reads ``INFERENCE_BACKEND`` from settings:
      - ``"tei"``   → :class:`TEIEmbeddings` (direct HTTP to rag-embed container)
      - anything else → :class:`LocalBGEEmbeddings` (in-process sentence-transformers;
                         dev venv path — requires the `local-embed` pyproject extra)
    """
    if INFERENCE_BACKEND == "tei":
        return TEIEmbeddings()
    return LocalBGEEmbeddings()
