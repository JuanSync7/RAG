# @summary
# Local BAAI bge-reranker-v2-m3 wrapper for reranking search results.
# Key exports: LocalBGEReranker, RankedResult
# Deps: transformers, dataclasses, torch, AutoModelForSequenceClassification, AutoTokenizer, config.settings
# @end-summary
"""Local BAAI bge-reranker-v2-m3 wrapper for reranking search results.

Uses transformers directly instead of FlagEmbedding to avoid compatibility
issues with transformers >= 5.x.
"""

from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.settings import RERANKER_MODEL_PATH, RERANK_TOP_K


@dataclass
class RankedResult:
    """A search result with its reranking score."""
    text: str
    score: float
    metadata: dict


class LocalBGEReranker:
    """Reranker using a local BAAI/bge-reranker-v2-m3 model."""

    def __init__(self, model_path: str = RERANKER_MODEL_PATH):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def rerank(
        self,
        query: str,
        documents: List[dict],
        top_k: int = RERANK_TOP_K,
    ) -> List[RankedResult]:
        """Rerank documents against a query.

        Args:
            query: The search query.
            documents: List of dicts with 'text' and 'metadata' keys.
            top_k: Number of top results to return.

        Returns:
            Sorted list of RankedResult (highest score first).
        """
        if not documents:
            return []

        pairs = [[query, doc["text"]] for doc in documents]

        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        scores = self.model(**inputs).logits.squeeze(-1).float().cpu().tolist()

        # If single document, scores is a scalar
        if isinstance(scores, float):
            scores = [scores]

        # Normalize scores to 0-1 via sigmoid
        scores = [1 / (1 + torch.exp(torch.tensor(-s)).item()) for s in scores]

        results = [
            RankedResult(
                text=doc["text"],
                score=score,
                metadata=doc.get("metadata", {}),
            )
            for doc, score in zip(documents, scores)
        ]

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
