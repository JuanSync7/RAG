# @summary
# Embedding-based entity resolution using cosine similarity.
# Exports: EmbeddingResolver
# Deps: numpy (optional), src.knowledge_graph.backend, src.knowledge_graph.resolution.schemas
# @end-summary
"""Embedding-based entity resolution using cosine similarity.

Type-constrained: only entities of the same type are compared.
Uses the configured embedding model (EMBEDDING_MODEL_PATH).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Dict, List, Optional

from src.knowledge_graph.backend import GraphStorageBackend
from src.knowledge_graph.resolution.schemas import MergeCandidate

__all__ = ["EmbeddingResolver"]

logger = logging.getLogger("rag.knowledge_graph.resolution.embedding")


class EmbeddingResolver:
    """Find merge candidates using embedding cosine similarity.

    Type-constrained: only entities of the same type are compared.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ) -> None:
        """Initialize with similarity threshold and optional embedding function.

        Args:
            threshold: Minimum cosine similarity for a merge candidate.
            embed_fn: Function that maps a list of strings to embeddings.
                If None, attempts to load from EMBEDDING_MODEL_PATH.
        """
        self._threshold = threshold
        self._embed_fn = embed_fn
        self._model = None

    def _get_embed_fn(self) -> Optional[Callable[[List[str]], List[List[float]]]]:
        """Lazy-load the embedding function."""
        if self._embed_fn is not None:
            return self._embed_fn

        model_path = os.environ.get("EMBEDDING_MODEL_PATH", "") or os.environ.get(
            "RAG_EMBEDDING_MODEL", ""
        )
        if not model_path:
            logger.warning(
                "No embedding model configured (EMBEDDING_MODEL_PATH / RAG_EMBEDDING_MODEL) "
                "— embedding-based entity resolution disabled"
            )
            return None

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

            if self._model is None:
                self._model = SentenceTransformer(model_path)
            return lambda texts: self._model.encode(texts, normalize_embeddings=True).tolist()
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — embedding-based resolution disabled"
            )
            return None

    def find_candidates(
        self, backend: GraphStorageBackend
    ) -> List[MergeCandidate]:
        """Load entities, compute embeddings, find merge candidates.

        Returns:
            List of MergeCandidate objects.
        """
        embed_fn = self._get_embed_fn()
        if embed_fn is None:
            return []

        try:
            import numpy as np
        except ImportError:
            logger.warning("numpy not available — embedding resolution disabled")
            return []

        all_entities = backend.get_all_entities()
        if len(all_entities) < 2:
            return []

        _t0 = time.monotonic()

        # Group entities by type
        type_buckets: Dict[str, List] = {}
        for entity in all_entities:
            type_buckets.setdefault(entity.type, []).append(entity)

        candidates: List[MergeCandidate] = []
        merged_names: set = set()

        for entity_type, entities in type_buckets.items():
            if len(entities) < 2:
                continue

            names = [e.name for e in entities]
            try:
                embeddings = np.array(embed_fn(names))
            except Exception as exc:
                logger.warning("Embedding failed for type %s: %s", entity_type, exc)
                continue

            # Compute pairwise cosine similarity
            # embeddings are already normalized if using SentenceTransformer
            sim_matrix = embeddings @ embeddings.T

            for i in range(len(names)):
                if names[i] in merged_names:
                    continue
                for j in range(i + 1, len(names)):
                    if names[j] in merged_names:
                        continue
                    similarity = float(sim_matrix[i, j])
                    if similarity >= self._threshold:
                        # Canonical = entity with higher mention count
                        if entities[i].mention_count >= entities[j].mention_count:
                            canonical, duplicate = names[i], names[j]
                        else:
                            canonical, duplicate = names[j], names[i]
                        candidates.append(
                            MergeCandidate(
                                canonical=canonical,
                                duplicate=duplicate,
                                similarity=similarity,
                                reason="embedding_similarity",
                            )
                        )
                        merged_names.add(duplicate)

        elapsed = time.monotonic() - _t0
        logger.info(
            "EmbeddingResolver.find_candidates: %d merge candidates, elapsed=%.1fs",
            len(candidates), elapsed,
        )
        logger.debug(
            "EmbeddingResolver.find_candidates: candidates=%d elapsed=%.3fs",
            len(candidates), elapsed,
        )
        return candidates
