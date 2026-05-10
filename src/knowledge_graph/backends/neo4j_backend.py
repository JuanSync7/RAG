# @summary
# Neo4j-based concrete implementation of GraphStorageBackend.
# Uses the official neo4j Python driver with MERGE-based entity resolution,
# UNWIND bulk operations, and JSON export/import for save/load interop.
# Exports: Neo4jBackend
# Deps: neo4j, orjson, logging, src.knowledge_graph.backend,
#        src.knowledge_graph.common.schemas, src.knowledge_graph.common.types
# Notable: query_neighbors_typed added (REQ-KG-760) — Cypher filters by r.relation whitelist.
# @end-summary
"""Neo4j-based graph storage backend.

Implements :class:`GraphStorageBackend` using a Neo4j graph database.
Provides server-side entity resolution via case-insensitive ``MERGE`` on
``name_lower``, bulk ``UNWIND`` writes for upsert operations, and JSON
export/import for interoperability with the NetworkX format.

Connection parameters are read from :class:`KGConfig` (``neo4j_uri``,
``neo4j_auth_user``, ``neo4j_auth_password``, ``neo4j_database``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.knowledge_graph.backend import GraphStorageBackend, RemovalStats, MergeReport
from src.knowledge_graph.common import (
    Entity,
    EntityDescription,
    Triple,
)

__all__ = ["Neo4jBackend"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency sentinel
# ---------------------------------------------------------------------------
try:
    import neo4j as _neo4j  # type: ignore[import-untyped]

    _NEO4J_AVAILABLE = True
except ImportError:
    _neo4j = None  # type: ignore[assignment]
    _NEO4J_AVAILABLE = False

# Default token budget for accumulated entity descriptions.
_DEFAULT_DESCRIPTION_TOKEN_BUDGET = 600


class Neo4jBackend(GraphStorageBackend):
    """Neo4j graph database storage backend.

    Entity resolution is performed server-side using a ``name_lower``
    property and ``MERGE`` clauses.  Bulk writes use ``UNWIND $batch``
    for efficiency.

    Parameters
    ----------
    config : KGConfig
        Runtime configuration supplying connection details.
    description_token_budget : int
        Token budget for entity description accumulation.
    """

    def __init__(
        self,
        config: Any = None,
        *,
        description_token_budget: int = _DEFAULT_DESCRIPTION_TOKEN_BUDGET,
    ) -> None:
        if not _NEO4J_AVAILABLE:
            raise ImportError(
                "The 'neo4j' package is required for Neo4jBackend. "
                "Install it with: pip install neo4j"
            )

        if config is None:
            from src.knowledge_graph.common import KGConfig

            config = KGConfig(backend="neo4j")

        self._uri = config.neo4j_uri
        self._database = config.neo4j_database
        self._description_token_budget = description_token_budget

        self._driver = _neo4j.GraphDatabase.driver(
            self._uri,
            auth=(config.neo4j_auth_user, config.neo4j_auth_password),
        )
        self._ensure_indexes()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the Neo4j driver and release resources."""
        if hasattr(self, "_driver") and self._driver is not None:
            self._driver.close()
            self._driver = None  # type: ignore[assignment]

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Index bootstrap
    # ------------------------------------------------------------------

    def _ensure_indexes(self) -> None:
        """Create indexes on initial connection for query performance."""
        index_statements = [
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name_lower)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.community_id)",
        ]
        with self._driver.session(database=self._database) as session:
            for stmt in index_statements:
                session.run(stmt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_read(self, query: str, **params: Any) -> list:
        """Execute a read transaction and return all records."""
        with self._driver.session(database=self._database) as session:

            def _tx(tx: Any) -> list:
                result = tx.run(query, **params)
                return list(result)

            return session.execute_read(_tx)

    def _run_write(self, query: str, **params: Any) -> list:
        """Execute a write transaction and return all records."""
        with self._driver.session(database=self._database) as session:

            def _tx(tx: Any) -> list:
                result = tx.run(query, **params)
                return list(result)

            return session.execute_write(_tx)

    def _to_entity(self, record: Any) -> Entity:
        """Convert a Neo4j record (with an ``e`` node column) to Entity."""
        node = record["e"]
        raw_json = node.get("raw_mentions", "[]")
        if isinstance(raw_json, str):
            try:
                raw_list = json.loads(raw_json)
            except (json.JSONDecodeError, TypeError):
                raw_list = []
        else:
            raw_list = raw_json if raw_json else []

        raw_mentions = [
            EntityDescription(
                text=m.get("text", ""),
                source=m.get("source", ""),
                chunk_id=m.get("chunk_id", ""),
            )
            for m in raw_list
        ]

        aliases_raw = node.get("aliases", [])
        if isinstance(aliases_raw, str):
            try:
                aliases_raw = json.loads(aliases_raw)
            except (json.JSONDecodeError, TypeError):
                aliases_raw = []

        sources_raw = node.get("sources", [])
        if isinstance(sources_raw, str):
            try:
                sources_raw = json.loads(sources_raw)
            except (json.JSONDecodeError, TypeError):
                sources_raw = []

        extractor_raw = node.get("extractor_source", [])
        if isinstance(extractor_raw, str):
            try:
                extractor_raw = json.loads(extractor_raw)
            except (json.JSONDecodeError, TypeError):
                extractor_raw = []

        return Entity(
            name=node.get("name", ""),
            type=node.get("type", "concept"),
            sources=list(sources_raw),
            mention_count=node.get("mention_count", 1),
            aliases=list(aliases_raw),
            raw_mentions=raw_mentions,
            current_summary=node.get("current_summary", ""),
            extractor_source=list(extractor_raw),
        )

    @staticmethod
    def _enforce_token_budget(
        mentions: list, budget: int
    ) -> list:
        """Return the newest mentions that fit within *budget* tokens."""
        kept: list = []
        total_tokens = 0
        for mention in reversed(mentions):
            text = mention.get("text", "") if isinstance(mention, dict) else mention
            tokens = len(text.split())
            if total_tokens + tokens > budget and kept:
                break
            kept.append(mention)
            total_tokens += tokens
        kept.reverse()
        return kept

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
        """Upsert a single entity node with case-insensitive MERGE."""
        alias_list = list(aliases) if aliases else []
        source_list = [source] if source else []

        query = """
        MERGE (e:Entity {name_lower: toLower($name)})
        ON CREATE SET
            e.name = $name,
            e.type = $type,
            e.sources = $sources,
            e.mention_count = 1,
            e.aliases = $aliases,
            e.raw_mentions = '[]',
            e.current_summary = '',
            e.extractor_source = '[]'
        ON MATCH SET
            e.mention_count = e.mention_count + 1,
            e.sources = CASE
                WHEN $source <> '' AND NOT $source IN e.sources
                THEN e.sources + $source
                ELSE e.sources
            END,
            e.aliases = [a IN (e.aliases + $aliases) WHERE NOT a IN [] | a]
        """
        # For aliases deduplication on MATCH we use a more explicit approach
        query = """
        MERGE (e:Entity {name_lower: toLower($name)})
        ON CREATE SET
            e.name = $name,
            e.type = $type,
            e.sources = $sources,
            e.mention_count = 1,
            e.aliases = $aliases,
            e.raw_mentions = '[]',
            e.current_summary = '',
            e.extractor_source = '[]'
        ON MATCH SET
            e.mention_count = e.mention_count + 1
        WITH e
        // Merge source
        WITH e, CASE
            WHEN $source <> '' AND NOT $source IN e.sources
            THEN e.sources + $source
            ELSE e.sources
        END AS new_sources
        SET e.sources = new_sources
        WITH e
        // Merge aliases: add only those not already present
        UNWIND CASE WHEN size($aliases) > 0 THEN $aliases ELSE [null] END AS alias
        WITH e, alias
        WHERE alias IS NOT NULL AND NOT alias IN e.aliases
        SET e.aliases = e.aliases + alias
        """
        self._run_write(
            query,
            name=name,
            type=type,
            source=source,
            sources=source_list,
            aliases=alias_list,
        )

    def add_edge(
        self,
        subject: str,
        object: str,
        relation: str,
        source: str,
        weight: float = 1.0,
    ) -> None:
        """Upsert a directed edge, accumulating weight on duplicates."""
        # Silently drop self-edges (case-insensitive)
        if subject.lower() == object.lower():
            return

        query = """
        MATCH (s:Entity {name_lower: toLower($subject)})
        MATCH (o:Entity {name_lower: toLower($object)})
        MERGE (s)-[r:RELATES_TO {relation: $relation}]->(o)
        ON CREATE SET
            r.weight = $weight,
            r.sources = CASE WHEN $source <> '' THEN [$source] ELSE [] END
        ON MATCH SET
            r.weight = r.weight + $weight,
            r.sources = CASE
                WHEN $source <> '' AND NOT $source IN r.sources
                THEN r.sources + $source
                ELSE r.sources
            END
        """
        self._run_write(
            query,
            subject=subject,
            object=object,
            relation=relation,
            source=source,
            weight=weight,
        )

    def upsert_entities(self, entities: List[Entity]) -> None:
        """Batch upsert entities using UNWIND."""
        if not entities:
            return

        batch = []
        for ent in entities:
            batch.append(
                {
                    "name": ent.name,
                    "type": ent.type,
                    "sources": ent.sources or [],
                    "aliases": ent.aliases or [],
                    "mention_count": ent.mention_count,
                    "extractor_source": ent.extractor_source or [],
                }
            )

        query = """
        UNWIND $batch AS item
        MERGE (e:Entity {name_lower: toLower(item.name)})
        ON CREATE SET
            e.name = item.name,
            e.type = item.type,
            e.sources = item.sources,
            e.mention_count = 1,
            e.aliases = item.aliases,
            e.raw_mentions = '[]',
            e.current_summary = '',
            e.extractor_source = item.extractor_source
        ON MATCH SET
            e.mention_count = e.mention_count + 1
        """
        self._run_write(query, batch=batch)

        # Merge sources and aliases in a second pass for simplicity
        merge_query = """
        UNWIND $batch AS item
        MATCH (e:Entity {name_lower: toLower(item.name)})
        WITH e, item
        // Merge sources
        UNWIND CASE WHEN size(item.sources) > 0 THEN item.sources ELSE [null] END AS src
        WITH e, item, src
        WHERE src IS NOT NULL AND NOT src IN e.sources
        SET e.sources = e.sources + src
        """
        self._run_write(merge_query, batch=batch)

    def upsert_triples(self, triples: List[Triple]) -> None:
        """Batch upsert triples using UNWIND."""
        if not triples:
            return

        # Filter self-edges
        batch = []
        for t in triples:
            if t.subject.lower() == t.object.lower():
                continue
            batch.append(
                {
                    "subject": t.subject,
                    "object": t.object,
                    "relation": t.predicate,
                    "source": t.source,
                    "weight": t.weight,
                }
            )

        if not batch:
            return

        query = """
        UNWIND $batch AS item
        MATCH (s:Entity {name_lower: toLower(item.subject)})
        MATCH (o:Entity {name_lower: toLower(item.object)})
        MERGE (s)-[r:RELATES_TO {relation: item.relation}]->(o)
        ON CREATE SET
            r.weight = item.weight,
            r.sources = CASE WHEN item.source <> '' THEN [item.source] ELSE [] END
        ON MATCH SET
            r.weight = r.weight + item.weight,
            r.sources = CASE
                WHEN item.source <> '' AND NOT item.source IN r.sources
                THEN r.sources + item.source
                ELSE r.sources
            END
        """
        self._run_write(query, batch=batch)

    def upsert_descriptions(
        self, descriptions: Dict[str, List[EntityDescription]]
    ) -> None:
        """Append textual mentions to entity records, respecting token budget."""
        for name, desc_list in descriptions.items():
            if not desc_list:
                continue

            new_mentions = [
                {"text": d.text, "source": d.source, "chunk_id": d.chunk_id}
                for d in desc_list
            ]

            # Read current mentions, append, enforce budget, write back
            read_query = """
            MATCH (e:Entity {name_lower: toLower($name)})
            RETURN e.raw_mentions AS raw_mentions
            """
            records = self._run_read(read_query, name=name)
            if not records:
                logger.debug(
                    "Skipping descriptions for unknown entity %r", name
                )
                continue

            raw_json = records[0]["raw_mentions"] or "[]"
            if isinstance(raw_json, str):
                try:
                    existing = json.loads(raw_json)
                except (json.JSONDecodeError, TypeError):
                    existing = []
            else:
                existing = list(raw_json)

            combined = existing + new_mentions
            trimmed = self._enforce_token_budget(
                combined, self._description_token_budget
            )

            write_query = """
            MATCH (e:Entity {name_lower: toLower($name)})
            SET e.raw_mentions = $mentions_json
            """
            self._run_write(
                write_query,
                name=name,
                mentions_json=json.dumps(trimmed),
            )

    # ------------------------------------------------------------------
    # Incremental operations (Phase 3)
    # ------------------------------------------------------------------

    def remove_by_source(self, source_key: str) -> RemovalStats:
        """Remove or prune all graph data associated with *source_key*.

        Deletes entities whose sole source is *source_key*, prunes
        *source_key* from multi-source entities, and removes all edges
        with matching source.

        Args:
            source_key: Document path or URI whose data should be removed.

        Returns:
            ``RemovalStats`` with removal counts.
        """
        stats = RemovalStats(source_key=source_key)

        # Remove edges with matching source
        records = self._run_write(
            """
            MATCH ()-[r]->()
            WHERE $source_key IN r.sources
            DELETE r
            RETURN count(r) AS removed
            """,
            source_key=source_key,
        )
        if records:
            stats.triples_removed = records[0]["removed"]

        # Prune source from multi-source entities
        records = self._run_write(
            """
            MATCH (e:Entity)
            WHERE $source_key IN e.sources AND size(e.sources) > 1
            SET e.sources = [s IN e.sources WHERE s <> $source_key]
            RETURN count(e) AS pruned
            """,
            source_key=source_key,
        )
        if records:
            stats.entities_pruned = records[0]["pruned"]

        # Delete entities where source_key was the only source
        records = self._run_write(
            """
            MATCH (e:Entity)
            WHERE e.sources = [$source_key]
            DETACH DELETE e
            RETURN count(e) AS removed
            """,
            source_key=source_key,
        )
        if records:
            stats.entities_removed = records[0]["removed"]

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

        Redirects all edges from *duplicate* to *canonical*, transfers
        aliases and sources, then deletes the duplicate.

        Args:
            canonical: Canonical name of the surviving entity.
            duplicate: Canonical name of the entity to absorb and delete.
        """
        # Verify both nodes exist
        records = self._run_read(
            """
            OPTIONAL MATCH (c:Entity {name_lower: toLower($canonical)})
            OPTIONAL MATCH (d:Entity {name_lower: toLower($duplicate)})
            RETURN c IS NOT NULL AS canon_exists, d IS NOT NULL AS dup_exists
            """,
            canonical=canonical,
            duplicate=duplicate,
        )
        if not records or not records[0]["canon_exists"]:
            logger.warning(
                "merge_entities: canonical %r does not exist — aborting", canonical
            )
            return
        if not records[0]["dup_exists"]:
            logger.debug(
                "merge_entities: duplicate %r not found — skipping", duplicate
            )
            return

        # Redirect outgoing edges: duplicate → X  becomes  canonical → X
        self._run_write(
            """
            MATCH (dup:Entity {name_lower: toLower($duplicate)})-[r]->(target)
            WHERE target.name_lower <> toLower($canonical)
            MATCH (canon:Entity {name_lower: toLower($canonical)})
            MERGE (canon)-[r2:RELATES_TO {relation: r.relation}]->(target)
            ON CREATE SET r2 = properties(r)
            ON MATCH SET r2.weight = r2.weight + coalesce(r.weight, 1.0),
                         r2.sources = r2.sources + coalesce(r.sources, [])
            DELETE r
            """,
            canonical=canonical,
            duplicate=duplicate,
        )

        # Redirect incoming edges: X → duplicate  becomes  X → canonical
        self._run_write(
            """
            MATCH (source)-[r]->(dup:Entity {name_lower: toLower($duplicate)})
            WHERE source.name_lower <> toLower($canonical)
            MATCH (canon:Entity {name_lower: toLower($canonical)})
            MERGE (source)-[r2:RELATES_TO {relation: r.relation}]->(canon)
            ON CREATE SET r2 = properties(r)
            ON MATCH SET r2.weight = r2.weight + coalesce(r.weight, 1.0),
                         r2.sources = r2.sources + coalesce(r.sources, [])
            DELETE r
            """,
            canonical=canonical,
            duplicate=duplicate,
        )

        # Transfer aliases, sources, mention count from duplicate to canonical
        self._run_write(
            """
            MATCH (canon:Entity {name_lower: toLower($canonical)})
            MATCH (dup:Entity {name_lower: toLower($duplicate)})
            SET canon.aliases = canon.aliases + [$duplicate] + coalesce(dup.aliases, []),
                canon.sources = canon.sources + [s IN coalesce(dup.sources, [])
                                                 WHERE NOT s IN canon.sources],
                canon.mention_count = coalesce(canon.mention_count, 1)
                                    + coalesce(dup.mention_count, 1)
            """,
            canonical=canonical,
            duplicate=duplicate,
        )

        # Delete the duplicate node
        self._run_write(
            """
            MATCH (dup:Entity {name_lower: toLower($duplicate)})
            DETACH DELETE dup
            """,
            duplicate=duplicate,
        )

        logger.debug("merge_entities: merged %r into %r", duplicate, canonical)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_entity(self, name: str) -> Optional[Entity]:
        """Return the entity record for *name*, case-insensitive."""
        query = """
        MATCH (e:Entity {name_lower: toLower($name)})
        RETURN e
        """
        records = self._run_read(query, name=name)
        if not records:
            # Try alias lookup
            alias_query = """
            MATCH (e:Entity)
            WHERE $name IN e.aliases
            RETURN e
            LIMIT 1
            """
            records = self._run_read(alias_query, name=name)
            if not records:
                return None

        return self._to_entity(records[0])

    def query_neighbors(self, entity: str, depth: int = 1) -> List[Entity]:
        """Return entities reachable within *depth* hops (forward + backward)."""
        # Variable-length path traversal both directions
        query = """
        MATCH (seed:Entity {name_lower: toLower($entity)})
        CALL {
            WITH seed
            MATCH (seed)-[*1..$depth]-(neighbor:Entity)
            RETURN DISTINCT neighbor
        }
        WHERE neighbor <> seed
        RETURN neighbor AS e
        """
        # Neo4j does not allow parameter in variable-length range, so we
        # construct the query string with the depth literal.
        safe_depth = int(depth)
        query = f"""
        MATCH (seed:Entity {{name_lower: toLower($entity)}})
        CALL {{
            WITH seed
            MATCH (seed)-[*1..{safe_depth}]-(neighbor:Entity)
            RETURN DISTINCT neighbor
        }}
        WHERE neighbor <> seed
        RETURN neighbor AS e
        """
        records = self._run_read(query, entity=entity)
        entities: List[Entity] = []
        seen: set = set()
        for rec in records:
            ent = self._to_entity(rec)
            if ent.name not in seen:
                entities.append(ent)
                seen.add(ent.name)
        return entities

    def query_neighbors_typed(
        self,
        entity: str,
        edge_types: List[str],
        depth: int = 1,
    ) -> List[Entity]:
        """Return neighbors reachable via edges whose type is in edge_types.

        Issues two Cypher queries (outgoing and incoming) with a variable-length
        path that constrains every relationship's ``relation`` property to the
        provided whitelist.  The depth literal is embedded directly in the query
        string because Neo4j does not accept parameters in ``[*m..n]`` ranges.

        REQ-KG-760: edge-type filtering applied natively via Cypher WHERE clause.

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

        safe_depth = int(depth)

        # Outgoing: seed --> ... --> neighbor, all hops must match edge_types
        outgoing_query = f"""
        MATCH (seed:Entity {{name_lower: toLower($entity)}})-[r*1..{safe_depth}]->(neighbor:Entity)
        WHERE ALL(rel IN r WHERE rel.relation IN $edge_types)
          AND neighbor <> seed
        RETURN DISTINCT neighbor AS e
        """

        # Incoming: neighbor --> ... --> seed, all hops must match edge_types
        incoming_query = f"""
        MATCH (neighbor:Entity)-[r*1..{safe_depth}]->(seed:Entity {{name_lower: toLower($entity)}})
        WHERE ALL(rel IN r WHERE rel.relation IN $edge_types)
          AND neighbor <> seed
        RETURN DISTINCT neighbor AS e
        """

        seen: set = set()
        entities: List[Entity] = []

        for query in (outgoing_query, incoming_query):
            records = self._run_read(query, entity=entity, edge_types=list(edge_types))
            for rec in records:
                ent = self._to_entity(rec)
                if ent.name not in seen:
                    entities.append(ent)
                    seen.add(ent.name)

        return entities

    def get_predecessors(self, entity: str) -> List[Entity]:
        """Return entities with a directed edge into *entity*."""
        query = """
        MATCH (pred:Entity)-[:RELATES_TO]->(target:Entity {name_lower: toLower($entity)})
        RETURN pred AS e
        """
        records = self._run_read(query, entity=entity)
        entities: List[Entity] = []
        seen: set = set()
        for rec in records:
            ent = self._to_entity(rec)
            if ent.name not in seen:
                entities.append(ent)
                seen.add(ent.name)
        return entities

    # ------------------------------------------------------------------
    # Concrete overrides for edge/entity listing
    # ------------------------------------------------------------------

    def get_all_entities(self) -> List[Entity]:
        """Return all Entity nodes in the database."""
        query = "MATCH (e:Entity) RETURN e"
        records = self._run_read(query)
        return [self._to_entity(rec) for rec in records]

    def get_all_node_names_and_aliases(self) -> Dict[str, str]:
        """Return mapping of every lowercase name/alias to canonical name."""
        query = """
        MATCH (e:Entity)
        RETURN e.name AS name, e.aliases AS aliases
        """
        records = self._run_read(query)
        index: Dict[str, str] = {}
        for rec in records:
            name = rec["name"]
            index[name.lower()] = name
            aliases_raw = rec["aliases"] or []
            if isinstance(aliases_raw, str):
                try:
                    aliases_raw = json.loads(aliases_raw)
                except (json.JSONDecodeError, TypeError):
                    aliases_raw = []
            for alias in aliases_raw:
                index[alias.lower()] = name
        return index

    def get_outgoing_edges(self, node_id: str) -> List[Triple]:
        """Return outgoing Triple edges for *node_id*."""
        query = """
        MATCH (s:Entity {name_lower: toLower($node_id)})-[r:RELATES_TO]->(o:Entity)
        RETURN s.name AS subject, r.relation AS relation, o.name AS object,
               r.sources AS sources, r.weight AS weight
        """
        records = self._run_read(query, node_id=node_id)
        triples: List[Triple] = []
        for rec in records:
            sources = rec["sources"] or []
            triples.append(
                Triple(
                    subject=rec["subject"],
                    predicate=rec["relation"] or "",
                    object=rec["object"],
                    source=sources[0] if sources else "",
                    weight=rec["weight"] or 1.0,
                )
            )
        return triples

    def get_incoming_edges(self, node_id: str) -> List[Triple]:
        """Return incoming Triple edges for *node_id*."""
        query = """
        MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity {name_lower: toLower($node_id)})
        RETURN s.name AS subject, r.relation AS relation, o.name AS object,
               r.sources AS sources, r.weight AS weight
        """
        records = self._run_read(query, node_id=node_id)
        triples: List[Triple] = []
        for rec in records:
            sources = rec["sources"] or []
            triples.append(
                Triple(
                    subject=rec["subject"],
                    predicate=rec["relation"] or "",
                    object=rec["object"],
                    source=sources[0] if sources else "",
                    weight=rec["weight"] or 1.0,
                )
            )
        return triples

    # ------------------------------------------------------------------
    # Persistence (export / import)
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Export the graph as JSON in node_link_data compatible format.

        Since Neo4j persists data server-side, ``save()`` exports a snapshot
        for interop with the NetworkX backend format.
        """
        nodes = []
        edges = []

        # Export nodes
        node_query = """
        MATCH (e:Entity)
        RETURN e
        """
        for rec in self._run_read(node_query):
            node = rec["e"]
            props = dict(node.items())
            # Ensure id field for node_link_data compatibility
            props["id"] = props.get("name", "")
            nodes.append(props)

        # Export edges
        edge_query = """
        MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity)
        RETURN s.name AS source, o.name AS target,
               r.relation AS relation, r.weight AS weight,
               r.sources AS sources
        """
        for rec in self._run_read(edge_query):
            edges.append(
                {
                    "source": rec["source"],
                    "target": rec["target"],
                    "relation": rec["relation"] or "",
                    "weight": rec["weight"] or 1.0,
                    "sources": rec["sources"] or [],
                }
            )

        data = {
            "directed": True,
            "multigraph": False,
            "graph": {},
            "nodes": nodes,
            "edges": edges,
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        """Import graph from a JSON file (node_link_data format).

        Clears existing Entity nodes and RELATES_TO edges before import.
        """
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))

        # Clear current graph data
        self._run_write("MATCH (e:Entity) DETACH DELETE e")

        # Import nodes
        nodes_batch = []
        for node_data in raw.get("nodes", []):
            name = node_data.get("id", node_data.get("name", ""))
            if not name:
                continue
            nodes_batch.append(
                {
                    "name": name,
                    "type": node_data.get("type", "concept"),
                    "sources": node_data.get("sources", []),
                    "mention_count": node_data.get("mention_count", 1),
                    "aliases": node_data.get("aliases", []),
                    "raw_mentions": (
                        node_data.get("raw_mentions", "[]")
                        if isinstance(node_data.get("raw_mentions"), str)
                        else json.dumps(node_data.get("raw_mentions", []))
                    ),
                    "current_summary": node_data.get("current_summary", ""),
                    "extractor_source": (
                        node_data.get("extractor_source", [])
                        if isinstance(node_data.get("extractor_source"), list)
                        else []
                    ),
                    "community_id": node_data.get("community_id"),
                }
            )

        if nodes_batch:
            node_query = """
            UNWIND $batch AS item
            CREATE (e:Entity {
                name: item.name,
                name_lower: toLower(item.name),
                type: item.type,
                sources: item.sources,
                mention_count: item.mention_count,
                aliases: item.aliases,
                raw_mentions: item.raw_mentions,
                current_summary: item.current_summary,
                extractor_source: item.extractor_source,
                community_id: item.community_id
            })
            """
            self._run_write(node_query, batch=nodes_batch)

        # Import edges
        edges_batch = []
        for edge_data in raw.get("edges", []):
            source = edge_data.get("source", "")
            target = edge_data.get("target", "")
            if not source or not target:
                continue
            edges_batch.append(
                {
                    "source": source,
                    "target": target,
                    "relation": edge_data.get("relation", ""),
                    "weight": edge_data.get("weight", 1.0),
                    "sources": edge_data.get("sources", []),
                }
            )

        if edges_batch:
            edge_query = """
            UNWIND $batch AS item
            MATCH (s:Entity {name_lower: toLower(item.source)})
            MATCH (o:Entity {name_lower: toLower(item.target)})
            CREATE (s)-[:RELATES_TO {
                relation: item.relation,
                weight: item.weight,
                sources: item.sources
            }]->(o)
            """
            self._run_write(edge_query, batch=edges_batch)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, object]:
        """Return node count, edge count, and top-10 most-mentioned entities."""
        count_query = """
        MATCH (e:Entity)
        WITH count(e) AS node_count
        OPTIONAL MATCH ()-[r:RELATES_TO]->()
        RETURN node_count, count(r) AS edge_count
        """
        counts = self._run_read(count_query)
        node_count = counts[0]["node_count"] if counts else 0
        edge_count = counts[0]["edge_count"] if counts else 0

        top_query = """
        MATCH (e:Entity)
        RETURN e.name AS name
        ORDER BY e.mention_count DESC
        LIMIT 10
        """
        top_records = self._run_read(top_query)
        top_entities = [rec["name"] for rec in top_records]

        return {
            "nodes": node_count,
            "edges": edge_count,
            "top_entities": top_entities,
        }

    # ------------------------------------------------------------------
    # Community support (bonus)
    # ------------------------------------------------------------------

    def upsert_community(
        self,
        community_id: int,
        summary_text: str,
        member_count: int,
    ) -> None:
        """Create or update a Community node with summary metadata.

        This is a convenience method for Phase 2 community detection results.
        """
        query = """
        MERGE (c:Community {community_id: $community_id})
        ON CREATE SET
            c.summary = $summary_text,
            c.member_count = $member_count
        ON MATCH SET
            c.summary = $summary_text,
            c.member_count = $member_count
        """
        self._run_write(
            query,
            community_id=community_id,
            summary_text=summary_text,
            member_count=member_count,
        )
