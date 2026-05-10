# @summary
# Multi-hop path pattern engine for KG retrieval.
# Evaluates ordered sequences of edge types against the graph.
# Exports: PathMatcher
# Deps: src.knowledge_graph.backend, src.knowledge_graph.query.schemas,
#   src.knowledge_graph.common.validation (optional, lazy-imported when schema_path set)
# @end-summary
"""Multi-hop path pattern engine for KG retrieval.

Evaluates ordered sequences of edge types (path patterns) against the
knowledge graph by performing step-by-step typed traversal with per-path
cycle guards.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Set, Tuple

from src.knowledge_graph.backend import GraphStorageBackend
from src.knowledge_graph.query.schemas import PathHop, PathResult

__all__ = ["PathMatcher"]

logger = logging.getLogger("rag.knowledge_graph.query.path_matcher")

# Safety valve for exponential frontier growth on dense graphs.
_MAX_HOP_FANOUT = 50


class PathMatcher:
    """Evaluates ordered path patterns against the knowledge graph.

    REQ-KG-770: Path pattern = ordered sequence of edge type labels.
    REQ-KG-772: Step-by-step traversal with per-path cycle guard.
    REQ-KG-774: Multiple patterns per query, merged results.
    REQ-KG-776: Full hop chains retained in PathResult.
    """

    def __init__(
        self,
        backend: GraphStorageBackend,
        schema_path: Optional[str] = None,
        max_hop_fanout: int = _MAX_HOP_FANOUT,
    ) -> None:
        self._backend = backend
        self._schema_path = schema_path
        self._max_hop_fanout = max_hop_fanout

    def evaluate(
        self, seed_entity: str, patterns: List[List[str]]
    ) -> List[PathResult]:
        """Evaluate all patterns against a seed entity.

        Args:
            seed_entity: Starting entity name.
            patterns: List of path patterns, each an ordered list of edge type strings.

        Returns:
            Merged, deduplicated list of PathResult objects.

        Raises:
            ValueError: If patterns is empty or any pattern is empty/null.
        """
        if not patterns:
            raise ValueError("patterns must be non-empty")

        if self._schema_path and patterns:
            from src.knowledge_graph.common.validation import (
                validate_path_patterns,
            )
            _warnings = validate_path_patterns(patterns, self._schema_path)
            for w in _warnings:
                logger.warning(
                    "Path pattern [%d] hop %d: %s",
                    w.pattern_index,
                    w.hop_index,
                    w.message,
                )

        results: List[PathResult] = []
        seen: Set[Tuple[str, str, str]] = set()  # (seed, terminal, label) for dedup

        for pattern in patterns:
            if not pattern:
                raise ValueError("Each pattern must be a non-empty list of edge types")
            matched = self._match_pattern(seed_entity, pattern)
            for pr in matched:
                key = (pr.seed_entity, pr.terminal_entity, pr.pattern_label)
                if key not in seen:
                    seen.add(key)
                    results.append(pr)

        return results

    def _match_pattern(
        self, seed_entity: str, pattern: List[str]
    ) -> List[PathResult]:
        """Evaluate a single pattern via step-by-step frontier BFS.

        Each frontier entry is (current_entity, hops_so_far, visited_set).
        At each step, follow only edges of the pattern's edge type.
        Per-path visited set prevents cycles.
        """
        label = "->".join(pattern)
        # frontier: list of (current_entity_name, hops_list, visited_set)
        frontier: List[Tuple[str, List[PathHop], frozenset]] = [
            (seed_entity, [], frozenset({seed_entity}))
        ]

        for edge_type in pattern:
            next_frontier: List[Tuple[str, List[PathHop], frozenset]] = []
            for current, hops, visited in frontier:
                try:
                    neighbors = self._backend.query_neighbors_typed(
                        entity=current, edge_types=[edge_type], depth=1
                    )
                except Exception:
                    logger.warning(
                        "typed traversal failed for %r edge=%r",
                        current,
                        edge_type,
                        exc_info=True,
                    )
                    continue

                for neighbor in neighbors:
                    if neighbor.name in visited:
                        continue  # cycle guard
                    new_hops = hops + [PathHop(current, edge_type, neighbor.name)]
                    next_frontier.append(
                        (
                            neighbor.name,
                            new_hops,
                            visited | frozenset({neighbor.name}),
                        )
                    )

            # Fan-out guard (REQ-KG-776 bounded complexity)
            if len(next_frontier) > self._max_hop_fanout:
                logger.debug(
                    "Fan-out guard: truncating frontier from %d to %d at edge_type=%s",
                    len(next_frontier),
                    self._max_hop_fanout,
                    edge_type,
                )
                next_frontier = next_frontier[:self._max_hop_fanout]

            frontier = next_frontier
            if not frontier:
                return []

        return [
            PathResult(
                pattern_label=label,
                seed_entity=seed_entity,
                hops=hops,
                terminal_entity=terminal,
            )
            for terminal, hops, _ in frontier
        ]
