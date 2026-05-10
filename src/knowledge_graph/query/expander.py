# @summary
# Graph-based query expansion for KG-augmented retrieval.
# Finds entities in queries via EntityMatcher, fans out through graph
# neighbours, and optionally includes community-derived terms (global retrieval).
# Dispatches typed traversal (query_neighbors_typed) when KGConfig has
# enable_graph_context_injection=True and retrieval_edge_types non-empty
# (REQ-KG-762); otherwise falls back to untyped query_neighbors.
# Fan-out limits (max_terms, max_depth) are applied identically on both paths
# (REQ-KG-768).
# When enable_graph_context_injection=True, evaluates path patterns via
# PathMatcher and formats graph context via GraphContextFormatter.
# expand() returns ExpansionResult(terms, graph_context) for backward-compat.
# Graceful degradation (REQ-KG-1214): typed traversal, path-pattern evaluation,
# and graph-context formatting are each independently wrapped in try/except so
# a backend or formatter error never fails the request — each path degrades
# silently to its safe fallback (untyped neighbours / empty paths / empty context).
# Exports: GraphQueryExpander
# Deps: src.knowledge_graph.backend, src.knowledge_graph.query.entity_matcher,
#       src.knowledge_graph.query.sanitizer, src.knowledge_graph.query.path_matcher,
#       src.knowledge_graph.query.context_formatter, src.knowledge_graph.query.schemas,
#       src.knowledge_graph.community.detector,
#       src.knowledge_graph.common.types (KGConfig)
# @end-summary
"""Graph-based query expansion for KG-augmented retrieval.

Finds entities in queries using :class:`EntityMatcher`, expands via graph
neighbours through :class:`GraphStorageBackend`, and returns related terms
to augment BM25 search.  Replaces the legacy ``GraphQueryExpander`` from
``src/core/knowledge_graph.py``.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from src.knowledge_graph.community import CommunityDetector
    from src.knowledge_graph.common.types import KGConfig

from src.knowledge_graph.backend import GraphStorageBackend
from src.knowledge_graph.query.context_formatter import (
    GraphContextFormatter,
    format_community_section,
)
from src.knowledge_graph.query.entity_matcher import EntityMatcher
from src.knowledge_graph.query.path_matcher import PathMatcher
from src.knowledge_graph.query.sanitizer import QuerySanitizer
from src.knowledge_graph.query.schemas import ExpansionResult

__all__ = ["GraphQueryExpander", "collect_community_ids"]

logger = logging.getLogger("rag.knowledge_graph.query")


class GraphQueryExpander:
    """Find entities in a query and expand via graph neighbours.

    Uses :class:`EntityMatcher` for token-boundary-aware entity detection
    (replacing legacy substring matching) and :class:`QuerySanitizer` for
    normalisation and alias management.

    When a :class:`CommunityDetector` is provided and global retrieval is
    enabled, expansion includes community-derived terms after local terms.
    """

    def __init__(
        self,
        backend: GraphStorageBackend,
        max_depth: int = 1,
        max_terms: int = 3,
        community_detector: Optional["CommunityDetector"] = None,
        enable_global_retrieval: bool = False,
        config: Optional["KGConfig"] = None,
    ) -> None:
        """Initialise with a graph backend and optional community detector.

        Args:
            backend: Graph storage backend for neighbour queries.
            max_depth: Maximum hop depth for expansion.
            max_terms: Maximum number of expansion terms to return.
            community_detector: Optional detector for community-aware expansion.
                Must have is_ready==True for community terms to be included.
            enable_global_retrieval: When True and detector is ready, include
                community terms in expansion. Default False.
            config: Optional full KGConfig object.  When provided and
                ``config.enable_graph_context_injection`` is True and
                ``config.retrieval_edge_types`` is non-empty, neighbour
                traversal uses the typed backend method
                ``query_neighbors_typed`` (REQ-KG-762).  If ``None``,
                untyped traversal is always used.
        """
        self._backend = backend
        self._max_depth = max_depth
        self._max_terms = max_terms
        self._community_detector = community_detector
        self._enable_global_retrieval = enable_global_retrieval
        self._global_retrieval_warned = False
        self._config = config

        # Build matcher + sanitiser from backend index
        self._matcher: EntityMatcher
        self._sanitizer: QuerySanitizer
        self._build_from_backend()

        # Path-pattern and context-formatting helpers — only created when injection is on.
        self._path_matcher: Optional[PathMatcher] = None
        self._formatter: Optional[GraphContextFormatter] = None
        if config is not None and getattr(config, "enable_graph_context_injection", False):
            self._path_matcher = PathMatcher(
                backend,
                max_hop_fanout=getattr(config, "max_hop_fanout", 50),
            )
            token_budget: int = getattr(config, "graph_context_token_budget", 500)
            marker_style: str = getattr(config, "graph_context_marker_style", "markdown")
            schema_path: Optional[str] = getattr(config, "schema_path", None)
            self._formatter = GraphContextFormatter(
                token_budget=token_budget,
                marker_style=marker_style,
                schema_path=schema_path,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(self, query: str, depth: Optional[int] = None) -> ExpansionResult:
        """Return related entity names and optional graph context to augment the search query.

        Traverses the graph outward (and inward) from matched entities
        up to *depth* hops.  Returns only terms not already in the query,
        capped at ``max_terms``.

        When ``enable_graph_context_injection`` is True on the config, also
        evaluates path patterns and formats a structured graph context block.

        Args:
            query: User query text.
            depth: Override max depth (default: use init value).

        Returns:
            ExpansionResult with ``terms`` (expansion strings) and
            ``graph_context`` (formatted text for prompt injection, or ``""``).
            Iterating the result yields the same strings as the former List[str].
        """
        try:
            depth = depth if depth is not None else self._max_depth
            _t0 = time.monotonic()

            # Normalise query for matching
            normalized = self._sanitizer.normalize(query)

            # Tier 1: spaCy / substring match
            seed_entities = self._matcher.match(normalized)

            # Tier 2: LLM fallback (Phase 1b)
            if not seed_entities:
                seed_entities = self._matcher.match_with_llm_fallback(normalized)

            if not seed_entities:
                return ExpansionResult(terms=[], graph_context="")

            # Phase 3: Multi-hop for connects_to edges
            effective_depth = depth
            if effective_depth < 2:
                for entity in seed_entities:
                    out_edges = self._backend.get_outgoing_edges(entity)
                    if any(e.predicate == "connects_to" for e in out_edges):
                        effective_depth = max(effective_depth, 2)
                        logger.debug(
                            "Increased expansion depth to %d for connects_to edges on %s",
                            effective_depth, entity,
                        )
                        break

            # Expand via graph neighbours
            expanded: set[str] = set(seed_entities)

            for entity in seed_entities:
                # Typed dispatch (REQ-KG-762): use edge-type-filtered traversal when
                # enable_graph_context_injection is True and retrieval_edge_types is
                # non-empty; otherwise fall back to the existing untyped traversal.
                # REQ-KG-1214: typed path is wrapped in try/except so a backend error
                # never propagates — it degrades gracefully to untyped traversal.
                if (
                    self._config is not None
                    and getattr(self._config, "enable_graph_context_injection", False)
                    and getattr(self._config, "retrieval_edge_types", [])
                ):
                    try:
                        neighbours = self._backend.query_neighbors_typed(
                            entity,
                            self._config.retrieval_edge_types,
                            depth=effective_depth,
                        )
                        logger.debug(
                            "typed dispatch: edge_types=%s",
                            self._config.retrieval_edge_types,
                        )
                    except Exception:
                        logger.warning(
                            "Typed traversal failed, falling back to untyped expansion",
                            exc_info=True,
                        )
                        neighbours = self._backend.query_neighbors(entity, depth=effective_depth)
                else:
                    logger.debug("untyped dispatch")
                    neighbours = self._backend.query_neighbors(entity, depth=effective_depth)

                for neighbour in neighbours:
                    expanded.add(neighbour.name)

                # Explicit predecessors (entities that point at this one)
                predecessors = self._backend.get_predecessors(entity)
                for pred in predecessors:
                    expanded.add(pred.name)

            # Filter out terms already in the query
            query_lower = query.lower()
            local_terms = [e for e in expanded if e.lower() not in query_lower]

            # Phase 2: Community-aware global retrieval
            community_terms: List[str] = []
            if self._enable_global_retrieval and self._community_detector is not None:
                if self._community_detector.is_ready:
                    community_terms = self._expand_with_communities(
                        seed_entities, set(local_terms) | expanded
                    )
                    # Filter community terms not already in query
                    community_terms = [
                        t for t in community_terms if t.lower() not in query_lower
                    ]
                elif not self._global_retrieval_warned:
                    logger.warning(
                        "Global retrieval enabled but CommunityDetector is not ready "
                        "(detection or summarization incomplete). "
                        "Falling back to local-only expansion."
                    )
                    self._global_retrieval_warned = True

            # Merge: local terms first, community terms fill remaining slots
            result = local_terms[: self._max_terms]
            remaining = self._max_terms - len(result)
            if remaining > 0 and community_terms:
                result.extend(community_terms[:remaining])

            # Graph context injection: path patterns + context formatting
            graph_context = ""
            if (
                self._config is not None
                and getattr(self._config, "enable_graph_context_injection", False)
                and self._path_matcher is not None
                and self._formatter is not None
            ):
                # Evaluate path patterns across all seed entities
                path_patterns = getattr(self._config, "retrieval_path_patterns", [])
                all_paths = []
                if path_patterns:
                    for entity in seed_entities:
                        try:
                            entity_paths = self._path_matcher.evaluate(entity, path_patterns)
                            all_paths.extend(entity_paths)
                        except Exception:
                            logger.warning(
                                "Path pattern evaluation failed for entity %r",
                                entity,
                                exc_info=True,
                            )

                # Collect entity objects and triples for seed + expanded entities
                from src.knowledge_graph.common.schemas import Entity, Triple

                all_entity_names = list(seed_entities) + result
                entities_for_context: List[Entity] = []
                triples_for_context: List[Triple] = []
                for name in all_entity_names:
                    entity_obj = self._backend.get_entity(name)
                    if entity_obj is not None:
                        entities_for_context.append(entity_obj)
                    triples_for_context.extend(self._backend.get_outgoing_edges(name))

                # Format the context block (REQ-KG-1214: wrap in try/except)
                try:
                    graph_context = self._formatter.format(
                        entities=entities_for_context,
                        triples=triples_for_context,
                        paths=all_paths,
                        seed_entity_names=list(seed_entities),
                    )
                except Exception:
                    logger.warning(
                        "Graph context formatting failed; returning empty context",
                        exc_info=True,
                    )
                    graph_context = ""

                # Phase 2: Community context injection (REQ-KG-1300..1308)
                # Appends pre-built community summaries for all communities
                # touched by traversal. Independent token budget. Zero LLM cost.
                community_budget = getattr(self._config, "community_context_token_budget", 0)
                if (
                    community_budget > 0
                    and self._community_detector is not None
                    and self._community_detector.is_ready
                ):
                    try:
                        cids, entity_counts = collect_community_ids(
                            entities=entities_for_context,
                            paths=all_paths,
                            backend=self._backend,
                        )
                        summaries = {}
                        for cid in cids:
                            summary = self._community_detector.get_summary(cid)
                            if summary is not None:
                                summaries[cid] = summary
                        if summaries and self._formatter is not None:
                            community_section = format_community_section(
                                summaries=summaries,
                                entity_counts=entity_counts,
                                token_budget=community_budget,
                                section_markers=self._formatter._section_markers,
                            )
                            if community_section:
                                graph_context = (
                                    graph_context + "\n" + community_section
                                    if graph_context
                                    else community_section
                                )
                    except Exception:
                        logger.warning(
                            "Community context injection failed; proceeding without",
                            exc_info=True,
                        )

            logger.debug(
                "GraphQueryExpander.expand: query=%r depth=%d terms=%d "
                "graph_context_len=%d elapsed=%.3fs",
                query, effective_depth, len(result), len(graph_context), time.monotonic() - _t0,
            )
            return ExpansionResult(terms=result, graph_context=graph_context)
        except Exception:
            logger.exception(
                "GraphQueryExpander.expand() failed for query %r; returning empty expansion",
                query,
            )
            return ExpansionResult(terms=[], graph_context="")

    def get_context_summary(
        self, entities: List[str], max_lines: int = 5
    ) -> str:
        """Build a short text summary of entity relationships.

        When community summaries are available, appends community context
        for matched entities.

        Args:
            entities: Entity names to summarise.
            max_lines: Maximum relationship lines to include.

        Returns:
            Semicolon-separated relationship summary.
        """
        lines: list[str] = []
        for entity in entities:
            edges = self._backend.get_outgoing_edges(entity)
            for edge in edges:
                lines.append(f"{edge.subject} {edge.predicate} {edge.object}")
                if len(lines) >= max_lines:
                    break
            if len(lines) >= max_lines:
                break

        # Append community context if available
        if (
            self._enable_global_retrieval
            and self._community_detector is not None
            and self._community_detector.is_ready
        ):
            seen_communities: set[int] = set()
            for entity in entities:
                cid = self._community_detector.get_community_for_entity(entity)
                if cid is not None and cid != -1 and cid not in seen_communities:
                    seen_communities.add(cid)
                    summary = self._community_detector.get_summary(cid)
                    if summary:
                        lines.append(f"[Community {cid}] {summary.summary_text}")

        return "; ".join(lines)

    def _expand_with_communities(
        self, seed_entities: List[str], existing_terms: set[str]
    ) -> List[str]:
        """Extract community-derived expansion terms.

        For each seed entity, looks up its community and retrieves other
        member names not already in existing_terms.

        Args:
            seed_entities: Entities matched in the query.
            existing_terms: Local expansion terms already collected.

        Returns:
            Community-derived terms, deduplicated.
        """
        assert self._community_detector is not None
        community_terms: list[str] = []
        seen_communities: set[int] = set()

        for entity in seed_entities:
            cid = self._community_detector.get_community_for_entity(entity)
            if cid is None or cid == -1 or cid in seen_communities:
                continue
            seen_communities.add(cid)

            summary = self._community_detector.get_summary(cid)
            if summary is None:
                continue

            # Add community members not already in expansion
            for member in summary.member_names:
                if member not in existing_terms and member not in community_terms:
                    community_terms.append(member)

        return community_terms

    def rebuild_matcher(self) -> None:
        """Rebuild entity matcher and sanitiser from current graph state.

        Call after the graph has been mutated (new documents ingested)
        so that the matcher picks up newly added entities and aliases.
        """
        self._build_from_backend()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_from_backend(self) -> None:
        """(Re)build matcher and sanitiser from the backend's entity index."""
        # get_all_node_names_and_aliases returns {lowercase_name_or_alias: canonical}
        name_index: Dict[str, str] = self._backend.get_all_node_names_and_aliases()

        # Entity names = unique canonical names
        entity_names = list(set(name_index.values()))

        # Alias index = non-canonical entries  (alias → canonical)
        alias_index: Dict[str, str] = {}
        canonical_lower = {n.lower() for n in entity_names}
        for key, canonical in name_index.items():
            if key not in canonical_lower:
                alias_index[key] = canonical

        self._matcher = EntityMatcher(entity_names, alias_index)
        self._sanitizer = QuerySanitizer(alias_index)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def collect_community_ids(
    entities: List["Entity"],
    paths: List["PathResult"],
    backend: "GraphStorageBackend",
) -> Tuple[Set[int], Dict[int, int]]:
    """Collect community IDs from all entities encountered during traversal.

    Scans seed entities and every hop in every path.  Entities not carrying a
    ``community_id`` attribute (or whose ``community_id`` is ``None`` / ``-1``)
    are silently skipped.

    Args:
        entities: Seed and expanded :class:`Entity` objects already in hand.
        paths: :class:`PathResult` objects from path-pattern evaluation.
        backend: Graph storage backend used to resolve hop entity names that
            are not already present in *entities*.

    Returns:
        A 2-tuple of:
        - ``community_ids``: deduplicated :class:`set` of valid community IDs.
        - ``entity_counts``: mapping of community ID → number of unique entity
          names that belong to it across the full traversal.
    """
    from src.knowledge_graph.common.schemas import Entity  # local import to avoid circularity

    # Build a name → Entity lookup from the already-resolved entities list so
    # we avoid redundant backend round-trips for hops that appear in seeds.
    name_to_entity: Dict[str, "Entity"] = {}
    for ent in entities:
        try:
            name_to_entity[ent.name] = ent
        except Exception:
            pass

    community_ids: Set[int] = set()
    # Maps community_id → set of unique entity names counted so far.
    _community_members: Dict[int, set] = {}

    def _register(ent_obj: "Entity") -> None:
        """Record the community membership of a single entity object."""
        try:
            cid = getattr(ent_obj, "community_id", None)
        except Exception:
            return
        if cid is None or cid == -1:
            return
        try:
            cid = int(cid)
        except (TypeError, ValueError):
            return
        community_ids.add(cid)
        _community_members.setdefault(cid, set()).add(ent_obj.name)

    def _resolve_and_register(name: str) -> None:
        """Look up *name* in the pre-built map or fall back to the backend."""
        ent_obj = name_to_entity.get(name)
        if ent_obj is not None:
            _register(ent_obj)
            return
        # Attempt backend lookup; skip silently on any error.
        try:
            fetched = backend.get_entity(name)
            if fetched is not None:
                name_to_entity[name] = fetched  # cache for repeated hops
                _register(fetched)
        except Exception:
            pass

    # --- seed entities -------------------------------------------------------
    for ent in entities:
        try:
            _register(ent)
        except Exception:
            pass

    # --- path hops -----------------------------------------------------------
    for path in paths:
        try:
            hops = path.hops
        except Exception:
            continue
        for hop in hops:
            try:
                from_name = hop.from_entity
                to_name = hop.to_entity
            except Exception:
                continue
            if from_name:
                _resolve_and_register(from_name)
            if to_name:
                _resolve_and_register(to_name)

    entity_counts: Dict[int, int] = {cid: len(members) for cid, members in _community_members.items()}
    return community_ids, entity_counts
