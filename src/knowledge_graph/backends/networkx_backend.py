# @summary
# NetworkX-based concrete implementation of GraphStorageBackend.
# Exports: NetworkXBackend
# Deps: networkx, orjson, src.knowledge_graph.backend, src.knowledge_graph.common.schemas
# Notable: query_neighbors_typed added (REQ-KG-760) — BFS filtered by edge predicate whitelist.
# @end-summary
"""NetworkX-based graph storage backend.

Implements :class:`GraphStorageBackend` using an in-memory ``nx.DiGraph``.
Provides entity resolution (alias + case-insensitive dedup), edge upserting
with weight accumulation, and ``orjson``/``node_link_data`` persistence that
is backward-compatible with the legacy ``KnowledgeGraphBuilder`` format.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import orjson

from src.knowledge_graph.backend import GraphStorageBackend, MergeReport, RemovalStats
from src.knowledge_graph.common import (
    Entity,
    EntityDescription,
    Triple,
)

__all__ = ["NetworkXBackend"]

logger = logging.getLogger(__name__)

# Default token budget for accumulated entity descriptions.
_DEFAULT_DESCRIPTION_TOKEN_BUDGET = 600


class NetworkXBackend(GraphStorageBackend):
    """NetworkX ``DiGraph``-backed knowledge-graph storage.

    Maintains two auxiliary indices for entity resolution:

    * ``_aliases``     — maps surface-form aliases to canonical node names.
    * ``_case_index``  — maps lowercased names to canonical (first-seen) form.

    Persistence uses ``orjson`` + ``nx.node_link_data`` so that saved files
    remain readable by the legacy ``KnowledgeGraphBuilder.load()`` path.
    """

    def __init__(
        self,
        description_token_budget: int = _DEFAULT_DESCRIPTION_TOKEN_BUDGET,
    ) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self._aliases: Dict[str, str] = {}
        self._case_index: Dict[str, str] = {}
        self._description_token_budget = description_token_budget

    # ------------------------------------------------------------------
    # Entity resolution (migrated from KnowledgeGraphBuilder._resolve)
    # ------------------------------------------------------------------

    def _resolve(self, term: str) -> str:
        """Resolve an alias/acronym then deduplicate by case.

        Priority: acronym alias -> case-insensitive existing node -> original.
        First-seen form becomes canonical (preserves original casing).
        """
        # Acronym / alias expansion first
        term = self._aliases.get(term, term)
        # Case-insensitive dedup: reuse existing canonical form
        lower = term.lower()
        if lower in self._case_index:
            return self._case_index[lower]
        # First time seeing this (case-insensitive) — register it
        self._case_index[lower] = term
        return term

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_node(
        self,
        name: str,
        type: str,
        source: str,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Upsert a single entity node with alias/case dedup."""
        canonical = self._resolve(name)

        if self.graph.has_node(canonical):
            data = self.graph.nodes[canonical]
            data["mention_count"] += 1
            if source and source not in data["sources"]:
                data["sources"].append(source)
            if aliases:
                for a in aliases:
                    if a not in data["aliases"]:
                        data["aliases"].append(a)
                    # Register alias in the alias index
                    self._aliases[a] = canonical
                    self._case_index[a.lower()] = canonical
        else:
            self.graph.add_node(
                canonical,
                type=type,
                sources=[source] if source else [],
                mention_count=1,
                aliases=list(aliases) if aliases else [],
            )
            # Register aliases
            if aliases:
                for a in aliases:
                    self._aliases[a] = canonical
                    self._case_index[a.lower()] = canonical

    def add_edge(
        self,
        subject: str,
        object: str,
        relation: str,
        source: str,
        weight: float = 1.0,
    ) -> None:
        """Upsert a directed edge, accumulating weight on duplicates."""
        subj_c = self._resolve(subject)
        obj_c = self._resolve(object)

        # Silently drop self-edges
        if subj_c == obj_c:
            return

        if self.graph.has_edge(subj_c, obj_c):
            edge_data = self.graph[subj_c][obj_c]
            edge_data["weight"] += weight
            if source and source not in edge_data.get("sources", []):
                edge_data.setdefault("sources", []).append(source)
        else:
            self.graph.add_edge(
                subj_c,
                obj_c,
                relation=relation,
                weight=weight,
                sources=[source] if source else [],
            )

    def upsert_entities(self, entities: List[Entity]) -> None:
        """Batch upsert entities via ``add_node``."""
        for ent in entities:
            self.add_node(
                name=ent.name,
                type=ent.type,
                source=ent.sources[0] if ent.sources else "",
                aliases=ent.aliases or None,
            )
            # Merge remaining sources beyond the first
            canonical = self._resolve(ent.name)
            if self.graph.has_node(canonical):
                node_data = self.graph.nodes[canonical]
                for src in ent.sources[1:]:
                    if src and src not in node_data["sources"]:
                        node_data["sources"].append(src)

    def upsert_triples(self, triples: List[Triple]) -> None:
        """Batch upsert triples via ``add_edge``."""
        for t in triples:
            self.add_edge(
                subject=t.subject,
                object=t.object,
                relation=t.predicate,
                source=t.source,
                weight=t.weight,
            )

    def upsert_descriptions(
        self, descriptions: Dict[str, List[EntityDescription]]
    ) -> None:
        """Append textual mentions to entity records, respecting token budget.

        Per REQ-KG-400: new mentions are appended with source attribution.
        When the accumulated token count exceeds ``description_token_budget``,
        the oldest mentions are dropped so the most recent ones fit within
        the budget.
        """
        for name, desc_list in descriptions.items():
            canonical = self._resolve(name)
            if not self.graph.has_node(canonical):
                # Entity not yet in graph — skip silently
                logger.debug(
                    "Skipping descriptions for unknown entity %r", canonical
                )
                continue

            node_data = self.graph.nodes[canonical]
            raw: List[Dict[str, str]] = node_data.setdefault("raw_mentions", [])

            for desc in desc_list:
                raw.append(
                    {
                        "text": desc.text,
                        "source": desc.source,
                        "chunk_id": desc.chunk_id,
                    }
                )

            # Token-budget enforcement: approximate tokens = word count
            self._enforce_token_budget(raw)

    def _enforce_token_budget(self, mentions: List[Dict[str, str]]) -> None:
        """Trim oldest mentions so total tokens stay within budget."""
        budget = self._description_token_budget
        # Walk from newest to oldest, accumulating tokens
        kept: List[Dict[str, str]] = []
        total_tokens = 0
        for mention in reversed(mentions):
            tokens = len(mention["text"].split())
            if total_tokens + tokens > budget and kept:
                # Adding this mention would exceed the budget
                break
            kept.append(mention)
            total_tokens += tokens

        # Restore chronological order (oldest first)
        kept.reverse()
        mentions[:] = kept

    def remove_by_source(self, source_key: str) -> RemovalStats:
        """Remove or prune all graph data associated with *source_key*.

        Entities whose ``sources`` list contains *source_key* as the only
        source are deleted entirely.  Entities that also appear in other
        sources have *source_key* pruned from their ``sources`` list — the
        node itself survives.  All edges whose ``sources`` list contains
        *source_key* are removed unconditionally.

        Args:
            source_key: Document path or URI whose data should be removed.

        Returns:
            ``RemovalStats`` describing how many entities were removed, pruned,
            and how many triples were deleted.
        """
        stats = RemovalStats(source_key=source_key)
        nodes_to_delete: List[str] = []

        # --- Pass 1: classify nodes ---
        for node, data in list(self.graph.nodes(data=True)):
            sources: List[str] = data.get("sources", [])
            if source_key not in sources:
                continue
            if len(sources) == 1:
                # source_key is the sole source — mark for deletion
                nodes_to_delete.append(node)
                stats.entities_removed += 1
            else:
                # Entity lives in other sources too — prune only this source
                data["sources"] = [s for s in sources if s != source_key]
                stats.entities_pruned += 1

        # --- Pass 2: remove edges that belong to source_key ---
        edges_to_remove = [
            (u, v)
            for u, v, data in self.graph.edges(data=True)
            if source_key in data.get("sources", [])
        ]
        for u, v in edges_to_remove:
            self.graph.remove_edge(u, v)
            stats.triples_removed += 1

        # --- Pass 3: delete marked nodes and clean up indices ---
        for node in nodes_to_delete:
            # Clean _case_index entries that point to this node
            stale_case_keys = [k for k, v in self._case_index.items() if v == node]
            for k in stale_case_keys:
                del self._case_index[k]
            # Clean _aliases entries that point to this node
            stale_alias_keys = [k for k, v in self._aliases.items() if v == node]
            for k in stale_alias_keys:
                del self._aliases[k]
            self.graph.remove_node(node)

        logger.debug(
            "remove_by_source(%r): removed=%d pruned=%d triples=%d",
            source_key,
            stats.entities_removed,
            stats.entities_pruned,
            stats.triples_removed,
        )
        return stats

    def merge_entities(self, canonical: str, duplicate: str) -> None:
        """Merge *duplicate* into *canonical*, then delete the duplicate node.

        All edges referencing *duplicate* as subject or object are redirected
        to *canonical*.  Aliases, mention counts, ``raw_mentions``, and
        ``sources`` from *duplicate* are merged into *canonical*.

        Args:
            canonical: Canonical name of the surviving entity.
            duplicate: Canonical name of the entity to absorb and delete.
        """
        if not self.graph.has_node(canonical):
            logger.warning(
                "merge_entities: canonical node %r does not exist — aborting", canonical
            )
            return
        if not self.graph.has_node(duplicate):
            # Duplicate already gone — nothing to do
            logger.debug(
                "merge_entities: duplicate node %r not found — skipping", duplicate
            )
            return

        can_data = self.graph.nodes[canonical]
        dup_data = self.graph.nodes[duplicate]

        # --- Transfer outgoing edges: duplicate → X  becomes  canonical → X ---
        for _, target, edge_data in list(self.graph.out_edges(duplicate, data=True)):
            if target == canonical:
                # Would create a self-loop — skip
                continue
            if self.graph.has_edge(canonical, target):
                self.graph[canonical][target]["weight"] += edge_data.get("weight", 1.0)
            else:
                self.graph.add_edge(
                    canonical,
                    target,
                    relation=edge_data.get("relation", ""),
                    weight=edge_data.get("weight", 1.0),
                    sources=list(edge_data.get("sources", [])),
                )

        # --- Transfer incoming edges: X → duplicate  becomes  X → canonical ---
        for source_node, _, edge_data in list(self.graph.in_edges(duplicate, data=True)):
            if source_node == canonical:
                # Would create a self-loop — skip
                continue
            if self.graph.has_edge(source_node, canonical):
                self.graph[source_node][canonical]["weight"] += edge_data.get(
                    "weight", 1.0
                )
            else:
                self.graph.add_edge(
                    source_node,
                    canonical,
                    relation=edge_data.get("relation", ""),
                    weight=edge_data.get("weight", 1.0),
                    sources=list(edge_data.get("sources", [])),
                )

        # --- Merge aliases: duplicate's name + its aliases → canonical ---
        existing_aliases: List[str] = can_data.setdefault("aliases", [])
        new_aliases: List[str] = [duplicate] + list(dup_data.get("aliases", []))
        for alias in new_aliases:
            if alias not in existing_aliases:
                existing_aliases.append(alias)

        # --- Merge sources ---
        existing_sources: List[str] = can_data.setdefault("sources", [])
        for src in dup_data.get("sources", []):
            if src not in existing_sources:
                existing_sources.append(src)

        # --- Merge mention counts ---
        can_data["mention_count"] = can_data.get("mention_count", 1) + dup_data.get(
            "mention_count", 1
        )

        # --- Merge raw_mentions ---
        can_raw: List[Dict] = can_data.setdefault("raw_mentions", [])
        can_raw.extend(dup_data.get("raw_mentions", []))

        # --- Update _case_index: duplicate's lowercase name → canonical ---
        self._case_index[duplicate.lower()] = canonical

        # --- Update _aliases: duplicate's aliases → canonical ---
        for alias in dup_data.get("aliases", []):
            self._aliases[alias] = canonical
            self._case_index[alias.lower()] = canonical
        # The duplicate name itself acts as an alias now
        self._aliases[duplicate] = canonical

        # --- Remove the duplicate node (its edges were already redirected) ---
        self.graph.remove_node(duplicate)

        logger.debug(
            "merge_entities: merged %r into %r", duplicate, canonical
        )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_entity(self, name: str) -> Optional[Entity]:
        """Return the entity record for *name*, case-insensitive."""
        lower = name.lower()
        # Try case index first, then alias index
        canonical = self._case_index.get(lower)
        if canonical is None:
            canonical = self._aliases.get(name)
        if canonical is None or not self.graph.has_node(canonical):
            return None

        data = self.graph.nodes[canonical]
        raw_mentions = [
            EntityDescription(
                text=m["text"], source=m["source"], chunk_id=m.get("chunk_id", "")
            )
            for m in data.get("raw_mentions", [])
        ]
        return Entity(
            name=canonical,
            type=data.get("type", "concept"),
            sources=list(data.get("sources", [])),
            mention_count=data.get("mention_count", 1),
            aliases=list(data.get("aliases", [])),
            raw_mentions=raw_mentions,
            current_summary=data.get("current_summary", ""),
        )

    def query_neighbors(self, entity: str, depth: int = 1) -> List[Entity]:
        """Return entities reachable within *depth* hops (forward + backward)."""
        canonical = self._resolve(entity)
        if not self.graph.has_node(canonical):
            return []

        neighbor_names: set[str] = set()

        # Forward neighbors within depth hops
        for neighbor in nx.single_source_shortest_path_length(
            self.graph, canonical, cutoff=depth
        ):
            neighbor_names.add(neighbor)

        # Reverse neighbors (predecessors) — one hop only to match legacy
        for predecessor in self.graph.predecessors(canonical):
            neighbor_names.add(predecessor)

        # Exclude the seed entity itself
        neighbor_names.discard(canonical)

        entities: List[Entity] = []
        for n in neighbor_names:
            ent = self.get_entity(n)
            if ent is not None:
                entities.append(ent)
        return entities

    def query_neighbors_typed(
        self,
        entity: str,
        edge_types: List[str],
        depth: int = 1,
    ) -> List[Entity]:
        """Return neighbors reachable via edges whose type is in edge_types.

        Performs BFS up to *depth* hops, traversing both outgoing and incoming
        edges.  Only edges whose ``relation`` predicate appears in *edge_types*
        are followed.

        REQ-KG-760: edge-type filtering applied natively on the NetworkX graph.

        Args:
            entity: Name of the seed entity.
            edge_types: Non-empty whitelist of edge type labels.
            depth: Maximum hop depth (>= 1).

        Returns:
            Deduplicated Entity list within depth hops via matching edges.

        Raises:
            ValueError: If edge_types is empty or depth < 1.
        """
        if not edge_types:
            raise ValueError("edge_types must be a non-empty list")
        if depth < 1:
            raise ValueError("depth must be >= 1")

        edge_types_set: set[str] = set(edge_types)
        canonical = self._resolve(entity)
        if not self.graph.has_node(canonical):
            return []

        visited: set[str] = {canonical}
        # BFS queue: (node_name, current_depth)
        queue: list[tuple[str, int]] = [(canonical, 0)]
        neighbor_names: set[str] = set()

        while queue:
            current, current_depth = queue.pop(0)
            if current_depth >= depth:
                continue

            # Outgoing edges
            for triple in self.get_outgoing_edges(current):
                if triple.predicate in edge_types_set:
                    neighbor = triple.object
                    if neighbor not in visited:
                        visited.add(neighbor)
                        neighbor_names.add(neighbor)
                        queue.append((neighbor, current_depth + 1))

            # Incoming edges
            for triple in self.get_incoming_edges(current):
                if triple.predicate in edge_types_set:
                    neighbor = triple.subject
                    if neighbor not in visited:
                        visited.add(neighbor)
                        neighbor_names.add(neighbor)
                        queue.append((neighbor, current_depth + 1))

        entities: List[Entity] = []
        for n in neighbor_names:
            ent = self.get_entity(n)
            if ent is not None:
                entities.append(ent)
        return entities

    def get_predecessors(self, entity: str) -> List[Entity]:
        """Return entities with a directed edge into *entity*."""
        canonical = self._resolve(entity)
        if not self.graph.has_node(canonical):
            return []

        entities: List[Entity] = []
        for pred in self.graph.predecessors(canonical):
            ent = self.get_entity(pred)
            if ent is not None:
                entities.append(ent)
        return entities

    # ------------------------------------------------------------------
    # Concrete overrides for edge/entity listing
    # ------------------------------------------------------------------

    def get_outgoing_edges(self, node_id: str) -> List[Triple]:
        """Return outgoing ``Triple`` edges for *node_id*."""
        canonical = self._resolve(node_id)
        if not self.graph.has_node(canonical):
            return []
        triples: List[Triple] = []
        for _, target, data in self.graph.out_edges(canonical, data=True):
            triples.append(
                Triple(
                    subject=canonical,
                    predicate=data.get("relation", ""),
                    object=target,
                    source=data.get("sources", [""])[0] if data.get("sources") else "",
                    weight=data.get("weight", 1.0),
                )
            )
        return triples

    def get_incoming_edges(self, node_id: str) -> List[Triple]:
        """Return incoming ``Triple`` edges for *node_id*."""
        canonical = self._resolve(node_id)
        if not self.graph.has_node(canonical):
            return []
        triples: List[Triple] = []
        for source_node, _, data in self.graph.in_edges(canonical, data=True):
            triples.append(
                Triple(
                    subject=source_node,
                    predicate=data.get("relation", ""),
                    object=canonical,
                    source=data.get("sources", [""])[0] if data.get("sources") else "",
                    weight=data.get("weight", 1.0),
                )
            )
        return triples

    def get_all_entities(self) -> List[Entity]:
        """Return all nodes as Entity objects."""
        entities: List[Entity] = []
        for node in self.graph.nodes:
            ent = self.get_entity(node)
            if ent is not None:
                entities.append(ent)
        return entities

    def get_all_node_names_and_aliases(self) -> Dict[str, str]:
        """Return dict mapping every lowercase name/alias to canonical name."""
        # Direct read from internal index — no iteration needed
        return dict(self._case_index)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialize with ``orjson`` + ``nx.node_link_data``.

        Output format is backward-compatible with
        ``KnowledgeGraphBuilder.load()``.
        """
        try:
            data = nx.node_link_data(self.graph, edges="edges")
            path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
        except OSError as exc:
            logger.error("save: I/O error writing graph to %s: %s", path, exc)
            raise
        except ValueError as exc:
            logger.error("save: serialization error for graph at %s: %s", path, exc)
            raise
        except Exception as exc:
            logger.error("save: unexpected error saving graph to %s: %s", path, exc)
            raise

    def load(self, path: Path) -> None:
        """Deserialize from JSON and rebuild alias/case indices."""
        try:
            raw = orjson.loads(path.read_bytes())
            self.graph = nx.node_link_graph(raw, directed=True, edges="edges")
        except OSError as exc:
            logger.error("load: I/O error reading graph from %s: %s", path, exc)
            raise
        except (orjson.JSONDecodeError, ValueError) as exc:
            logger.error("load: JSON decode error for graph at %s: %s", path, exc)
            raise
        except Exception as exc:
            logger.error("load: unexpected error loading graph from %s: %s", path, exc)
            raise

        # Rebuild indices from loaded node data
        self._aliases.clear()
        self._case_index.clear()
        for node, node_data in self.graph.nodes(data=True):
            self._case_index[node.lower()] = node
            for alias in node_data.get("aliases", []):
                self._aliases[alias] = node
                self._case_index[alias.lower()] = node

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, object]:
        """Return node count, edge count, and top-10 most-mentioned entities."""
        top = sorted(
            self.graph.nodes(data=True),
            key=lambda x: x[1].get("mention_count", 0),
            reverse=True,
        )[:10]
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "top_entities": [name for name, _ in top],
        }
