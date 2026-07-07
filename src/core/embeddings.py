# @summary
# Embedding providers: local BGE (in-process) and a remote OpenAI-compatible
# HTTP client (vLLM / TEI / any compatible API), selected by INFERENCE_BACKEND.
# Exports: LocalBGEEmbeddings, TEIEmbeddings, get_embedding_provider
# Deps: sentence-transformers (local path only), httpx, numpy, langchain_core, config.settings
# @end-summary
"""Embedding provider implementations and factory.

Two backends (selected by ``RAG_INFERENCE_BACKEND``):
  local — BGE loaded in-process via sentence-transformers (requires the
          `local-embed` pyproject extra).
  http  — embeddings served over an OpenAI-compatible ``/v1/embeddings`` endpoint,
          i.e. whatever ``RAG_EMBED_URL`` points at: a remote vLLM (the dev path),
          a self-hosted TEI pool, or any compatible API. (``TEIEmbeddings`` is the
          client class — the name predates the move off TEI-specific servers.)
"""
from __future__ import annotations


import httpx
import numpy as np
from langchain_core.embeddings import Embeddings

from config.settings import (
    EMBEDDING_MODEL_PATH,
    INFERENCE_BACKEND,
    RAG_EMBED_MODEL,
    RAG_EMBED_URL,
    RAG_EMBEDDING_BATCH_SIZE_DOCUMENTS,
    RAG_EMBEDDING_BATCH_SIZE_SEMANTIC_CHUNKING,
    RAG_INFERENCE_TIMEOUT_SECONDS,
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
        base_url: str = RAG_EMBED_URL,
        model: str = RAG_EMBED_MODEL,
        timeout: int = RAG_INFERENCE_TIMEOUT_SECONDS,
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
      - ``"http"`` → :class:`TEIEmbeddings` (OpenAI-compatible HTTP to whatever
                      ``RAG_EMBED_URL`` points at — remote vLLM, a self-hosted TEI
                      pool, or any compatible API)
      - anything else → :class:`LocalBGEEmbeddings` (in-process sentence-transformers;
                         requires the `local-embed` pyproject extra)

    Args:
        tier: Optional load-balancer routing hint (X-RagWeave-Tier header). Pass
            ``"ingest"`` from ingestion paths to pin embedding traffic to a CPU
            pool, leaving the GPU free for latency-sensitive query embedding.
            Only meaningful for an nginx-fronted self-hosted pool; ignored
            otherwise (no-op for direct vLLM and ``LocalBGEEmbeddings``).
    """
    if INFERENCE_BACKEND == "http":
        return TEIEmbeddings(tier=tier)
    return LocalBGEEmbeddings()
