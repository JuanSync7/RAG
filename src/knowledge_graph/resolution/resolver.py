# @summary
# Entity resolution orchestrator: alias merges first, then embedding-based.
# Exports: EntityResolver
# Deps: src.knowledge_graph.backend, src.knowledge_graph.common.types,
#        src.knowledge_graph.resolution.alias_resolver,
#        src.knowledge_graph.resolution.embedding_resolver,
#        src.knowledge_graph.resolution.schemas
# @end-summary
"""Entity resolution orchestrator.

Runs alias-table merges first (deterministic), then embedding-based merges
(fuzzy). Both are type-constrained and configurable via ``KGConfig``.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

from src.knowledge_graph.backend import GraphStorageBackend
from src.knowledge_graph.common import KGConfig
from src.knowledge_graph.resolution.alias_resolver import AliasResolver
from src.knowledge_graph.resolution.embedding_resolver import EmbeddingResolver
from src.knowledge_graph.resolution.schemas import MergeCandidate, ResolutionReport

__all__ = ["EntityResolver"]

logger = logging.getLogger("rag.knowledge_graph.resolution")


class EntityResolver:
    """Orchestrates entity resolution: alias merges first, then embedding-based.

    Controlled by ``KGConfig.enable_entity_resolution`` (default False).
    When disabled, ``resolve()`` returns an empty ``ResolutionReport``.
    """

    def __init__(
        self,
        backend: GraphStorageBackend,
        config: KGConfig,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ) -> None:
        self._backend = backend
        self._config = config
        self._embed_fn = embed_fn

    def resolve(self) -> ResolutionReport:
        """Run the full entity resolution pipeline.

        Algorithm:
            1. Run AliasResolver to produce deterministic merge candidates.
            2. Execute alias merges via ``backend.merge_entities()``.
            3. Run EmbeddingResolver to produce fuzzy merge candidates.
            4. Execute embedding merges via ``backend.merge_entities()``.
            5. Return consolidated ResolutionReport.

        Returns:
            ResolutionReport with all merge operations performed.
        """
        if not self._config.enable_entity_resolution:
            return ResolutionReport()

        report = ResolutionReport()

        # Phase 1: Alias-table merges (deterministic)
        alias_resolver = AliasResolver(
            alias_path=self._config.entity_resolution_alias_path
        )
        alias_candidates = alias_resolver.find_candidates(self._backend)
        for candidate in alias_candidates:
            self._backend.merge_entities(candidate.canonical, candidate.duplicate)
            report.merges.append(candidate)

        # Phase 2: Embedding-based merges (fuzzy)
        embedding_resolver = EmbeddingResolver(
            threshold=self._config.entity_resolution_threshold,
            embed_fn=self._embed_fn,
        )
        embedding_candidates = embedding_resolver.find_candidates(self._backend)
        for candidate in embedding_candidates:
            self._backend.merge_entities(candidate.canonical, candidate.duplicate)
            report.merges.append(candidate)

        report.total_merged = len(report.merges)

        logger.info(
            "Entity resolution complete: %d alias merges, %d embedding merges",
            len(alias_candidates),
            len(embedding_candidates),
        )
        return report
