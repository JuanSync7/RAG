# @summary
# GraphStorageBackend ABC: formal swappable backend contract for all KG storage implementations.
# Exports: GraphStorageBackend, RemovalStats, MergeReport
# Deps: abc, dataclasses, pathlib, typing, src.knowledge_graph.common.schemas
# Notable: query_neighbors_typed added (REQ-KG-760) for edge-type-filtered traversal.
# @end-summary
"""GraphStorageBackend — abstract base class for all KG storage backends.

Defines the formal contract between the knowledge-graph pipeline and any
storage implementation.  New backends implement all abstract methods; the
four concrete helper methods (``get_all_entities``,
``get_all_node_names_and_aliases``, ``get_outgoing_edges``,
``get_incoming_edges``) may be overridden for efficiency but have sensible
default implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.knowledge_graph.common import (
    Entity,
    EntityDescription,
    Triple,
)


__all__ = ["GraphStorageBackend", "RemovalStats", "MergeReport"]


@dataclass
class RemovalStats:
    """Statistics from a remove_by_source operation."""

    entities_removed: int = 0
    entities_pruned: int = 0  # source removed from list but entity kept
    triples_removed: int = 0
    source_key: str = ""


@dataclass
class MergeReport:
    """Report from entity resolution merge operations."""

    merges: List[Tuple[str, str]] = field(default_factory=list)  # [(canonical, duplicate), ...]
    triples_redirected: int = 0
    aliases_transferred: int = 0


class GraphStorageBackend(ABC):
    """Abstract contract for a graph storage backend.

    Callers (pipeline nodes, query layer, export tools) interact only through
    these methods.  Swapping backends requires only a config-key change — no
    changes to the pipeline code.
    """

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    @abstractmethod
    def add_node(
        self,
        name: str,
        type: str,
        source: str,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Add or deduplicate a single entity node.

        Case-insensitive deduplication is the backend's responsibility.  If a
        node with the same normalised name already exists, the backend merges
        the new ``source`` and ``aliases`` into the existing record.

        Args:
            name: Canonical (first-seen) surface form of the entity.
            type: Node type as defined in the KG schema (e.g. ``RTL_Module``).
            source: Document path or URI from which this entity was extracted.
            aliases: Optional alternative names that resolve to this entity.
        """
        ...

    @abstractmethod
    def add_edge(
        self,
        subject: str,
        object: str,
        relation: str,
        source: str,
        weight: float = 1.0,
    ) -> None:
        """Add a directed relationship edge between two entities.

        Self-edges (``subject == object``) must be silently dropped.

        Args:
            subject: Canonical name of the source entity.
            object: Canonical name of the target entity.
            relation: Edge type label (e.g. ``instantiates``, ``specified_by``).
            source: Document path or URI from which this triple was extracted.
            weight: Confidence or frequency weight for the relationship.
        """
        ...

    @abstractmethod
    def upsert_entities(self, entities: List[Entity]) -> None:
        """Batch upsert a list of entities.

        Semantically equivalent to calling ``add_node`` for each entity, but
        backends may implement bulk-write optimisations.

        Args:
            entities: Entities to add or merge into the graph.
        """
        ...

    @abstractmethod
    def upsert_triples(self, triples: List[Triple]) -> None:
        """Batch upsert a list of relationship triples.

        Semantically equivalent to calling ``add_edge`` for each triple, but
        backends may implement bulk-write optimisations.

        Args:
            triples: Triples to add or merge into the graph.
        """
        ...

    @abstractmethod
    def upsert_descriptions(
        self, descriptions: Dict[str, List[EntityDescription]]
    ) -> None:
        """Append textual mentions to entity records.

        New descriptions are appended to the entity's ``raw_mentions`` list.
        Token-budget checks and LLM summarisation may be triggered here by
        concrete backends.

        Args:
            descriptions: Mapping of canonical entity name to list of new
                ``EntityDescription`` mentions to append.
        """
        ...

    @abstractmethod
    def remove_by_source(self, source_key: str) -> RemovalStats:
        """Remove or prune all graph data associated with *source_key*.

        Entities whose ``sources`` list contains *source_key* as the **only**
        source are deleted entirely.  Entities that also appear in other sources
        have *source_key* pruned from their ``sources`` list and their triple
        count decremented — the node itself survives.  All triples whose
        ``source`` field matches *source_key* are removed unconditionally.

        Args:
            source_key: Document path or URI whose data should be removed.

        Returns:
            ``RemovalStats`` describing how many entities were removed, pruned,
            and how many triples were deleted.
        """
        ...

    @abstractmethod
    def merge_entities(self, canonical: str, duplicate: str) -> None:
        """Merge *duplicate* into *canonical*, then delete the duplicate node.

        All triples that reference *duplicate* as subject or object are
        redirected to *canonical*.  Aliases, mention counts, ``raw_mentions``,
        and ``sources`` from *duplicate* are merged into *canonical*.  The
        *duplicate* entity node is deleted after the transfer.

        Args:
            canonical: Canonical name of the surviving entity.
            duplicate: Canonical name of the entity to absorb and delete.
        """
        ...

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    @abstractmethod
    def query_neighbors(self, entity: str, depth: int = 1) -> List[Entity]:
        """Return entities reachable from *entity* within *depth* hops.

        Both forward (outgoing) and backward (incoming) edges are traversed.

        Args:
            entity: Canonical name of the seed entity.
            depth: Maximum number of hops to traverse (default 1).

        Returns:
            Deduplicated list of neighbour ``Entity`` objects (seed excluded).
        """
        ...

    @abstractmethod
    def query_neighbors_typed(
        self,
        entity: str,
        edge_types: List[str],
        depth: int = 1,
    ) -> List["Entity"]:
        """Return neighbors reachable via edges whose type is in edge_types.

        REQ-KG-760: Backends filter by edge type natively when possible.

        Args:
            entity: Name of the seed entity.
            edge_types: Non-empty whitelist of edge type labels.
            depth: Maximum hop depth (>= 1).

        Returns:
            Deduplicated Entity list within depth hops via matching edges.

        Raises:
            ValueError: If edge_types is empty or depth < 1.
        """
        ...

    @abstractmethod
    def get_entity(self, name: str) -> Optional[Entity]:
        """Return the entity record for *name*, or ``None`` if not found.

        Lookup is case-insensitive.

        Args:
            name: Canonical or alias name for the entity.

        Returns:
            Matching ``Entity`` or ``None``.
        """
        ...

    @abstractmethod
    def get_predecessors(self, entity: str) -> List[Entity]:
        """Return entities that have a directed edge *into* *entity*.

        Args:
            entity: Canonical name of the target entity.

        Returns:
            List of ``Entity`` objects with an outgoing edge to *entity*.
        """
        ...

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialize the graph to disk at *path*.

        Args:
            path: Destination file or directory path (format is
                backend-specific).
        """
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Deserialize the graph from *path* and rebuild all indices.

        Args:
            path: Source file or directory path previously written by
                ``save()``.
        """
        ...

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @abstractmethod
    def stats(self) -> Dict[str, object]:
        """Return diagnostic statistics for the current graph state.

        Returns:
            Dictionary containing at minimum the keys ``nodes`` (int),
            ``edges`` (int), and ``top_entities`` (list of the most-mentioned
            canonical entity names).
        """
        ...

    # ------------------------------------------------------------------
    # Concrete helper methods (override for efficiency)
    # ------------------------------------------------------------------

    def get_all_entities(self) -> List[Entity]:
        """Return every entity in the graph.

        Default implementation builds the name/alias index then resolves each
        canonical name via ``get_entity()``.  Backends that maintain an
        internal node registry may override this for a more efficient bulk
        read.

        Returns:
            List of all ``Entity`` objects currently stored.
        """
        names_and_aliases = self.get_all_node_names_and_aliases()
        seen: set = set()
        entities: List[Entity] = []
        for canonical in names_and_aliases.values():
            if canonical not in seen:
                entity = self.get_entity(canonical)
                if entity is not None:
                    entities.append(entity)
                    seen.add(canonical)
        return entities

    def get_all_node_names_and_aliases(self) -> Dict[str, str]:
        """Return a mapping of every lowercase name/alias to its canonical name.

        Default implementation calls ``get_all_entities()`` and builds the
        index from the returned records.  Backends that maintain a
        ``_case_index`` or ``_alias_index`` internally may override this for
        a direct read.

        Returns:
            Dict mapping ``{lowercase_name_or_alias: canonical_name}``.
        """
        index: Dict[str, str] = {}
        for entity in self.get_all_entities():
            index[entity.name.lower()] = entity.name
            for alias in entity.aliases:
                index[alias.lower()] = entity.name
        return index

    def get_outgoing_edges(self, node_id: str) -> List[Triple]:
        """Return outgoing ``Triple`` edges for *node_id*.

        Default returns an empty list.  Override in backends that store edges
        in a queryable structure (required by the Obsidian export in T13).

        Args:
            node_id: Canonical entity name.

        Returns:
            List of ``Triple`` objects where ``subject == node_id``.
        """
        return []

    def get_incoming_edges(self, node_id: str) -> List[Triple]:
        """Return incoming ``Triple`` edges for *node_id*.

        Default returns an empty list.  Override in backends that store edges
        in a queryable structure (required by the Obsidian export in T13).

        Args:
            node_id: Canonical entity name.

        Returns:
            List of ``Triple`` objects where ``object == node_id``.
        """
        return []
