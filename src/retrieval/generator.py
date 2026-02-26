# @summary
# Ollama-based LLM generator for RAG answer synthesis. Main exports: OllamaGenerator. Deps: json, urllib.request, typing, config.settings
# @end-summary
"""Ollama-based LLM generator for RAG answer synthesis."""

import json
from typing import List, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

from config.settings import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    GENERATION_MAX_TOKENS,
    GENERATION_TEMPERATURE,
)


_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided context. "
    "Use ONLY the information from the context below to answer the question. "
    "If the context doesn't contain enough information to answer, say so clearly. "
    "Be concise and direct. "
    "IMPORTANT: Cite your sources using bracketed numbers like [1], [2], etc. "
    "that correspond to the context chunk numbers provided. "
    "Every claim should have at least one citation. "
    "Each context chunk has a relevance score (0-100%). "
    "Prioritize information from higher-scoring chunks. "
    "Treat chunks below 10% relevance with caution — they may not be directly relevant."
)

_USER_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""


class OllamaGenerator:
    """Generate answers using Ollama's HTTP API."""

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        max_tokens: int = GENERATION_MAX_TOKENS,
        temperature: float = GENERATION_TEMPERATURE,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(
        self,
        query: str,
        context_chunks: List[str],
        scores: Optional[List[float]] = None,
    ) -> Optional[str]:
        """Generate an answer using retrieved context chunks.

        Args:
            query: The user's question.
            context_chunks: List of relevant text chunks from retrieval.
            scores: Optional reranker scores (0.0-1.0) for each chunk.

        Returns:
            Generated answer string, or None if generation fails.
        """
        if not context_chunks:
            return None

        if scores:
            context = "\n\n".join(
                f"[{i+1}] (relevance: {score:.0%}) {chunk}"
                for i, (chunk, score) in enumerate(zip(context_chunks, scores))
            )
        else:
            context = "\n\n".join(
                f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)
            )
        user_message = _USER_TEMPLATE.format(context=context, question=query)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
            },
        }

        try:
            req = Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("message", {}).get("content")
        except URLError as e:
            print(f"Warning: Ollama generation failed: {e}")
            return None
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse Ollama response: {e}")
            return None

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            req = Request(f"{self.base_url}/api/tags", method="GET")
            with urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                return any(self.model in m for m in models)
        except Exception:
            return False
