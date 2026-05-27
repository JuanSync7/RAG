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
    RAG_EMBEDDING_BATCH_SIZE_DOCUMENTS,
    RAG_EMBEDDING_BATCH_SIZE_SEMANTIC_CHUNKING,
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
        self.model_path = model_path
        self.model = _load_sentence_transformer(model_path)
        self.tracer = get_tracer()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts.

        Notes:
            gen_ai.usage.input_tokens is reported as the sum of input text
            character lengths — sentence-transformers does not expose a
            tokenizer hook on this code path.
        """
        attrs = {
            "gen_ai.system": "sentence-transformers",
            "gen_ai.request.model": self.model_path,
            "gen_ai.usage.input_tokens": sum(len(t) for t in texts),
            "batch_size": len(texts),
        }
        with self.tracer.span("embeddings.local.batch", attrs) as span:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=True,
                batch_size=RAG_EMBEDDING_BATCH_SIZE_DOCUMENTS,
            )
            result = embeddings.tolist()
            span.set_attribute("vector_count", len(result))
            span.set_attribute("vector_dim", len(result[0]) if result else 0)
            return result

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        attrs = {
            "gen_ai.system": "sentence-transformers",
            "gen_ai.request.model": self.model_path,
            "gen_ai.usage.input_tokens": len(text),
        }
        with self.tracer.span("embeddings.local.query", attrs) as span:
            embedding = self.model.encode(
                text,
                normalize_embeddings=True,
            )
            vec = embedding.tolist()
            span.set_attribute("vector_count", 1)
            span.set_attribute("vector_dim", len(vec))
            return vec

    def encode_sentences(self, sentences: list[str]) -> np.ndarray:
        """Encode sentences returning numpy array for internal use.

        Used by semantic chunking for cosine similarity computation.
        Returns L2-normalized embeddings so cosine sim = dot product.
        """
        return self.model.encode(
            sentences,
            normalize_embeddings=True,
            batch_size=RAG_EMBEDDING_BATCH_SIZE_SEMANTIC_CHUNKING,
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
        tier: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        # tier="ingest" sets X-RagWeave-Tier so rag-nginx routes to the CPU
        # pool (rag-embed-cpu), keeping the GPU free for latency-sensitive
        # query embedding. Default (None) hits the GPU pool with CPU fallback.
        headers = {"X-RagWeave-Tier": tier} if tier else None
        self._client = httpx.Client(timeout=timeout, headers=headers)
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
        attrs = {
            "gen_ai.system": "tei",
            "gen_ai.request.model": self.model,
            "gen_ai.usage.input_tokens": sum(len(t) for t in texts),
            "batch_size": len(texts),
        }
        with self.tracer.span("embeddings.tei.batch", attrs) as span:
            vectors = self._embed(texts)
            span.set_attribute("vector_count", len(vectors))
            span.set_attribute("vector_dim", len(vectors[0]) if vectors else 0)
            return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text via TEI."""
        attrs = {
            "gen_ai.system": "tei",
            "gen_ai.request.model": self.model,
            "gen_ai.usage.input_tokens": len(text),
        }
        with self.tracer.span("embeddings.tei.query", attrs) as span:
            vectors = self._embed([text])
            span.set_attribute("vector_count", 1)
            span.set_attribute("vector_dim", len(vectors[0]) if vectors else 0)
            return vectors[0]

    def encode_sentences(self, sentences: list[str]) -> np.ndarray:
        """Return L2-normalized embeddings as a numpy array for semantic chunking.

        TEI normalizes BGE-family embeddings by default, so no post-hoc norm needed.
        """
        return np.array(self._embed(sentences))


def get_embedding_provider(tier: str | None = None) -> Embeddings:
    """Return the configured embedding provider.

    Reads ``INFERENCE_BACKEND`` from settings:
      - ``"tei"``   → :class:`TEIEmbeddings` (HTTP to rag-nginx → rag-embed pool)
      - anything else → :class:`LocalBGEEmbeddings` (in-process sentence-transformers;
                         dev venv path — requires the `local-embed` pyproject extra)

    Args:
        tier: Optional rag-nginx routing hint. Pass ``"ingest"`` from ingestion
            paths to pin embedding traffic to the CPU pool (rag-embed-cpu),
            leaving the GPU free for latency-sensitive query embedding.
            Ignored by ``LocalBGEEmbeddings`` (no LB layer in dev).
    """
    if INFERENCE_BACKEND == "tei":
        return TEIEmbeddings(tier=tier)
    return LocalBGEEmbeddings()
