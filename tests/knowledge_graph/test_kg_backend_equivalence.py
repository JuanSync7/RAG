# @summary
# Cross-backend equivalence tests, path pattern matching tests, and entity
# deduplication tests for the KG subsystem.
# Covers: insert/query round-trip, relation query, delete, dedup, case-insensitive
# normalization, clear-all, path pattern matching, max-hop fanout enforcement, and
# empty/invalid pattern handling.
# Exports: (test module — no public exports)
# Deps: pytest, src.knowledge_graph.backends.networkx_backend,
#       src.knowledge_graph.backend, src.knowledge_graph.common.schemas,
#       src.knowledge_graph.query.path_matcher
# @end-summary
"""Cross-backend equivalence and path-pattern tests for the KG subsystem.

Part 1 — Backend equivalence: same test suite run against NetworkXBackend and
         an in-memory Neo4j stub (Neo4jStubBackend).
Part 2 — Path pattern matching: PathMatcher behavior for valid patterns,
         max-hop fanout, empty pattern list, and invalid edge types.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pytest

from src.knowledge_graph.backend import GraphStorageBackend, RemovalStats
from src.knowledge_graph.common.schemas import Entity, EntityDescription, Triple


# ---------------------------------------------------------------------------
# Neo4j in-memory stub — satisfies the ABC without a live server
# ---------------------------------------------------------------------------


class Neo4jStubBackend(GraphStorageBackend):
    """Pure in-memory Neo4j stub for testing.

    Mirrors the NetworkXBackend semantics exactly (case-insensitive dedup,
    self-edge rejection, source pruning) but uses only plain Python dicts,
    so it requires no ``networkx`` or ``neo4j`` packages.
    """

    def __init__(self) -> None:
        # {canonical_name: {type, sources, mention_count, aliases, raw_mentions}}
        self._nodes: Dict[str, dict] = {}
        # {(subject, object): {relation, weight, sources}}
        self._edges: Dict[tuple, dict] = {}
        # lowercase name/alias → canonical name
        self._case_index: Dict[str, str] = {}
        # alias → canonical name
        self._aliases: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, name: str) -> str:
        name = self._aliases.get(name, name)
        lower = name.lower()
        if lower in self._case_index:
            return self._case_index[lower]
        self._case_index[lower] = name
        return name

    def _register_aliases(self, canonical: str, aliases: List[str]) -> None:
        for a in aliases:
            self._aliases[a] = canonical
            self._case_index[a.lower()] = canonical

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
        canonical = self._resolve(name)
        if canonical in self._nodes:
            data = self._nodes[canonical]
            data["mention_count"] += 1
            if source and source not in data["sources"]:
                data["sources"].append(source)
            if aliases:
                for a in aliases:
                    if a not in data["aliases"]:
                        data["aliases"].append(a)
                self._register_aliases(canonical, aliases)
        else:
            self._nodes[canonical] = {
                "type": type,
                "sources": [source] if source else [],
                "mention_count": 1,
                "aliases": list(aliases) if aliases else [],
                "raw_mentions": [],
            }
            if aliases:
                self._register_aliases(canonical, aliases)

    def add_edge(
        self,
        subject: str,
        object: str,
        relation: str,
        source: str,
        weight: float = 1.0,
    ) -> None:
        subj_c = self._resolve(subject)
        obj_c = self._resolve(object)
        if subj_c == obj_c:
            return  # drop self-edges silently
        key = (subj_c, obj_c)
        if key in self._edges:
            self._edges[key]["weight"] += weight
            if source and source not in self._edges[key]["sources"]:
                self._edges[key]["sources"].append(source)
        else:
            self._edges[key] = {
                "relation": relation,
                "weight": weight,
                "sources": [source] if source else [],
            }

    def upsert_entities(self, entities: List[Entity]) -> None:
        for ent in entities:
            self.add_node(
                name=ent.name,
                type=ent.type,
                source=ent.sources[0] if ent.sources else "",
                aliases=ent.aliases or None,
            )
            canonical = self._resolve(ent.name)
            if canonical in self._nodes:
                for src in ent.sources[1:]:
                    if src and src not in self._nodes[canonical]["sources"]:
                        self._nodes[canonical]["sources"].append(src)

    def upsert_triples(self, triples: List[Triple]) -> None:
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
        for name, desc_list in descriptions.items():
            canonical = self._resolve(name)
            if canonical not in self._nodes:
                continue
            raw = self._nodes[canonical]["raw_mentions"]
            for desc in desc_list:
                raw.append({"text": desc.text, "source": desc.source, "chunk_id": desc.chunk_id})

    def remove_by_source(self, source_key: str) -> RemovalStats:
        stats = RemovalStats(source_key=source_key)
        to_delete: List[str] = []
        for node, data in list(self._nodes.items()):
            sources = data.get("sources", [])
            if source_key not in sources:
                continue
            if len(sources) == 1:
                to_delete.append(node)
                stats.entities_removed += 1
            else:
                data["sources"].remove(source_key)
                stats.entities_pruned += 1

        edges_to_delete = [
            k for k, v in self._edges.items()
            if source_key in v.get("sources", [])
            or k[0] in to_delete
            or k[1] in to_delete
        ]
        for k in edges_to_delete:
            del self._edges[k]
            stats.triples_removed += 1

        for node in to_delete:
            del self._nodes[node]
            lower = node.lower()
            self._case_index.pop(lower, None)

        return stats

    def merge_entities(self, canonical: str, duplicate: str) -> None:
        if canonical not in self._nodes or duplicate not in self._nodes:
            return
        can_data = self._nodes[canonical]
        dup_data = self._nodes[duplicate]

        for (subj, obj), edge_data in list(self._edges.items()):
            new_subj = canonical if subj == duplicate else subj
            new_obj = canonical if obj == duplicate else obj
            if new_subj == new_obj:
                continue
            new_key = (new_subj, new_obj)
            if (subj, obj) != new_key:
                del self._edges[(subj, obj)]
                if new_key in self._edges:
                    self._edges[new_key]["weight"] += edge_data["weight"]
                else:
                    self._edges[new_key] = edge_data

        aliases = can_data.setdefault("aliases", [])
        for alias in [duplicate] + list(dup_data.get("aliases", [])):
            if alias not in aliases:
                aliases.append(alias)

        for src in dup_data.get("sources", []):
            if src not in can_data["sources"]:
                can_data["sources"].append(src)

        can_data["mention_count"] = can_data.get("mention_count", 1) + dup_data.get("mention_count", 1)
        can_data["raw_mentions"].extend(dup_data.get("raw_mentions", []))

        self._case_index[duplicate.lower()] = canonical
        self._aliases[duplicate] = canonical
        for alias in dup_data.get("aliases", []):
            self._aliases[alias] = canonical
            self._case_index[alias.lower()] = canonical

        del self._nodes[duplicate]

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_entity(self, name: str) -> Optional[Entity]:
        lower = name.lower()
        canonical = self._case_index.get(lower) or self._aliases.get(name)
        if canonical is None or canonical not in self._nodes:
            return None
        data = self._nodes[canonical]
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
        canonical = self._resolve(entity)
        if canonical not in self._nodes:
            return []
        neighbor_names: set = set()
        for (subj, obj) in self._edges:
            if subj == canonical:
                neighbor_names.add(obj)
            if obj == canonical:
                neighbor_names.add(subj)
        neighbor_names.discard(canonical)
        return [e for e in (self.get_entity(n) for n in neighbor_names) if e is not None]

    def query_neighbors_typed(
        self,
        entity: str,
        edge_types: List[str],
        depth: int = 1,
    ) -> List[Entity]:
        if not edge_types:
            raise ValueError("edge_types must be non-empty")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        canonical = self._resolve(entity)
        if canonical not in self._nodes:
            return []
        edge_types_set = set(edge_types)
        visited = {canonical}
        queue = [(canonical, 0)]
        neighbor_names: set = set()
        while queue:
            current, current_depth = queue.pop(0)
            if current_depth >= depth:
                continue
            for (subj, obj), ed in self._edges.items():
                if subj == current and ed["relation"] in edge_types_set and obj not in visited:
                    visited.add(obj)
                    neighbor_names.add(obj)
                    queue.append((obj, current_depth + 1))
                if obj == current and ed["relation"] in edge_types_set and subj not in visited:
                    visited.add(subj)
                    neighbor_names.add(subj)
                    queue.append((subj, current_depth + 1))
        return [e for e in (self.get_entity(n) for n in neighbor_names) if e is not None]

    def get_predecessors(self, entity: str) -> List[Entity]:
        canonical = self._resolve(entity)
        preds = [subj for (subj, obj) in self._edges if obj == canonical]
        return [e for e in (self.get_entity(p) for p in preds) if e is not None]

    def get_outgoing_edges(self, node_id: str) -> List[Triple]:
        canonical = self._resolve(node_id)
        triples = []
        for (subj, obj), ed in self._edges.items():
            if subj == canonical:
                triples.append(
                    Triple(
                        subject=subj,
                        predicate=ed["relation"],
                        object=obj,
                        source=ed["sources"][0] if ed["sources"] else "",
                        weight=ed["weight"],
                    )
                )
        return triples

    def get_incoming_edges(self, node_id: str) -> List[Triple]:
        canonical = self._resolve(node_id)
        triples = []
        for (subj, obj), ed in self._edges.items():
            if obj == canonical:
                triples.append(
                    Triple(
                        subject=subj,
                        predicate=ed["relation"],
                        object=obj,
                        source=ed["sources"][0] if ed["sources"] else "",
                        weight=ed["weight"],
                    )
                )
        return triples

    def get_all_entities(self) -> List[Entity]:
        return [e for e in (self.get_entity(n) for n in self._nodes) if e is not None]

    def get_all_node_names_and_aliases(self) -> Dict[str, str]:
        return dict(self._case_index)

    # ------------------------------------------------------------------
    # Persistence (no-ops for test stub)
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, object]:
        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "top_entities": list(self._nodes.keys())[:5],
        }


# ---------------------------------------------------------------------------
# Parametrised backend fixture
# ---------------------------------------------------------------------------


def _make_networkx() -> GraphStorageBackend:
    from src.knowledge_graph.backends.networkx_backend import NetworkXBackend
    return NetworkXBackend()


def _make_neo4j_stub() -> GraphStorageBackend:
    return Neo4jStubBackend()


BACKEND_FACTORIES = [
    pytest.param(_make_networkx, id="networkx"),
    pytest.param(_make_neo4j_stub, id="neo4j_stub"),
]


@pytest.fixture(params=BACKEND_FACTORIES)
def backend(request) -> GraphStorageBackend:
    """Provide a fresh backend instance for each parametrized test."""
    factory = request.param
    return factory()


# ---------------------------------------------------------------------------
# Part 1: Backend equivalence tests
# ---------------------------------------------------------------------------


class TestInsertAndQueryEntity:
    """Insert an entity node and query it back — same attributes returned."""

    def test_insert_and_get_entity(self, backend):
        backend.add_node(name="Alpha", type="Concept", source="doc1.md")
        entity = backend.get_entity("Alpha")

        assert entity is not None
        assert entity.name == "Alpha"
        assert entity.type == "Concept"
        assert "doc1.md" in entity.sources

    def test_get_entity_case_insensitive(self, backend):
        """get_entity lookup is case-insensitive."""
        backend.add_node(name="Alpha", type="Concept", source="doc1.md")
        entity = backend.get_entity("alpha")

        assert entity is not None
        assert entity.name == "Alpha"

    def test_get_nonexistent_entity_returns_none(self, backend):
        result = backend.get_entity("DoesNotExist")
        assert result is None


class TestInsertAndQueryRelation:
    """Insert a relation and query neighbors of subject entity."""

    def test_relation_appears_in_neighbors(self, backend):
        backend.add_node(name="A", type="Concept", source="src")
        backend.add_node(name="B", type="Concept", source="src")
        backend.add_edge(subject="A", object="B", relation="related_to", source="src")

        neighbors = backend.query_neighbors("A")
        neighbor_names = [e.name for e in neighbors]

        assert "B" in neighbor_names

    def test_relation_query_via_typed_neighbors(self, backend):
        """query_neighbors_typed with correct edge type returns the related entity."""
        backend.add_node(name="X", type="Module", source="src")
        backend.add_node(name="Y", type="Module", source="src")
        backend.add_edge(subject="X", object="Y", relation="depends_on", source="src")

        typed_neighbors = backend.query_neighbors_typed("X", edge_types=["depends_on"])
        neighbor_names = [e.name for e in typed_neighbors]

        assert "Y" in neighbor_names

    def test_typed_neighbors_wrong_edge_type_returns_empty(self, backend):
        """query_neighbors_typed with non-matching edge type returns empty list."""
        backend.add_node(name="X", type="Module", source="src")
        backend.add_node(name="Y", type="Module", source="src")
        backend.add_edge(subject="X", object="Y", relation="depends_on", source="src")

        typed_neighbors = backend.query_neighbors_typed("X", edge_types=["specified_by"])

        assert typed_neighbors == []


class TestDeleteEntity:
    """Delete entity → entity no longer findable."""

    def test_remove_by_source_deletes_sole_source_entity(self, backend):
        backend.add_node(name="DeleteMe", type="Concept", source="transient.md")
        assert backend.get_entity("DeleteMe") is not None

        stats = backend.remove_by_source("transient.md")

        assert backend.get_entity("DeleteMe") is None
        assert stats.entities_removed >= 1

    def test_remove_by_source_prunes_multi_source_entity(self, backend):
        """Entity with multiple sources loses the source but the node survives."""
        backend.add_node(name="Survivor", type="Concept", source="s1.md")
        backend.add_node(name="Survivor", type="Concept", source="s2.md")

        stats = backend.remove_by_source("s1.md")

        entity = backend.get_entity("Survivor")
        assert entity is not None, "Node with remaining sources should survive"
        assert "s1.md" not in entity.sources
        assert stats.entities_pruned >= 1


class TestDeduplication:
    """Duplicate entity (same name/type) → only 1 node after insert."""

    def test_duplicate_add_node_deduped(self, backend):
        backend.add_node(name="Widget", type="Concept", source="a.md")
        backend.add_node(name="Widget", type="Concept", source="b.md")

        all_entities = backend.get_all_entities()
        widget_nodes = [e for e in all_entities if e.name == "Widget"]

        assert len(widget_nodes) == 1, "Duplicate add_node must result in a single node"

    def test_duplicate_upsert_entities_deduped(self, backend):
        entities = [
            Entity(name="DupNode", type="Concept", sources=["a.md"]),
            Entity(name="DupNode", type="Concept", sources=["b.md"]),
        ]
        backend.upsert_entities(entities)

        all_entities = backend.get_all_entities()
        dup_nodes = [e for e in all_entities if e.name == "DupNode"]

        assert len(dup_nodes) == 1


class TestEntityNameNormalization:
    """'AI' and 'ai' treated as same entity — single node exists."""

    def test_case_variants_resolve_to_same_node(self, backend):
        backend.add_node(name="AI", type="Concept", source="doc1.md")
        backend.add_node(name="ai", type="Concept", source="doc2.md")

        all_entities = backend.get_all_entities()
        ai_nodes = [e for e in all_entities if e.name.lower() == "ai"]

        assert len(ai_nodes) == 1, "Both 'AI' and 'ai' should map to a single canonical node"

    def test_first_seen_form_is_canonical(self, backend):
        """The first-inserted casing is preserved as the canonical name."""
        backend.add_node(name="RTL_Module", type="Module", source="src")
        backend.add_node(name="rtl_module", type="Module", source="src")

        entity = backend.get_entity("rtl_module")

        assert entity is not None
        assert entity.name == "RTL_Module"

    def test_lookup_by_lowercase_finds_uppercase_canonical(self, backend):
        backend.add_node(name="UART", type="Module", source="src")

        entity = backend.get_entity("uart")

        assert entity is not None
        assert entity.name == "UART"


class TestClearAll:
    """clear all (remove_by_source on all sources) → graph is empty."""

    def test_clear_all_via_remove_by_source(self, backend):
        backend.add_node(name="N1", type="Concept", source="src")
        backend.add_node(name="N2", type="Concept", source="src")
        backend.add_edge(subject="N1", object="N2", relation="related_to", source="src")

        backend.remove_by_source("src")

        all_entities = backend.get_all_entities()
        assert all_entities == [], f"Expected empty graph, got {all_entities}"

    def test_stats_after_clear(self, backend):
        backend.add_node(name="N1", type="Concept", source="src")
        backend.remove_by_source("src")

        s = backend.stats()
        assert s["nodes"] == 0


# ---------------------------------------------------------------------------
# Part 2: Path pattern matching tests
# ---------------------------------------------------------------------------


@pytest.fixture
def path_backend() -> GraphStorageBackend:
    """NetworkXBackend pre-loaded with a three-node chain for path tests."""
    from src.knowledge_graph.backends.networkx_backend import NetworkXBackend

    b = NetworkXBackend()
    b.upsert_entities([
        Entity(name="Start", type="Concept"),
        Entity(name="Middle", type="Concept"),
        Entity(name="End", type="Concept"),
    ])
    b.upsert_triples([
        Triple(subject="Start", predicate="hop_a", object="Middle", source="test"),
        Triple(subject="Middle", predicate="hop_b", object="End", source="test"),
    ])
    return b


class TestPathPatternMatching:
    """PathMatcher evaluates ordered edge-type sequences against the graph."""

    def test_valid_single_hop_pattern_returns_result(self, path_backend):
        from src.knowledge_graph.query.path_matcher import PathMatcher

        matcher = PathMatcher(backend=path_backend)
        results = matcher.evaluate("Start", patterns=[["hop_a"]])

        terminal_names = [r.terminal_entity for r in results]
        assert "Middle" in terminal_names

    def test_valid_two_hop_pattern_returns_terminal(self, path_backend):
        """Two-hop chain Start→Middle→End is matched correctly."""
        from src.knowledge_graph.query.path_matcher import PathMatcher

        matcher = PathMatcher(backend=path_backend)
        results = matcher.evaluate("Start", patterns=[["hop_a", "hop_b"]])

        terminal_names = [r.terminal_entity for r in results]
        assert "End" in terminal_names

    def test_pattern_hops_are_recorded(self, path_backend):
        """PathResult.hops preserves the full chain."""
        from src.knowledge_graph.query.path_matcher import PathMatcher

        matcher = PathMatcher(backend=path_backend)
        results = matcher.evaluate("Start", patterns=[["hop_a", "hop_b"]])

        assert len(results) > 0
        hops = results[0].hops
        assert len(hops) == 2
        assert hops[0].edge_type == "hop_a"
        assert hops[1].edge_type == "hop_b"

    def test_invalid_edge_type_in_pattern_returns_empty(self, path_backend):
        """Pattern containing a non-existent edge type produces no results (no crash)."""
        from src.knowledge_graph.query.path_matcher import PathMatcher

        matcher = PathMatcher(backend=path_backend)
        results = matcher.evaluate("Start", patterns=[["nonexistent_edge"]])

        assert results == []

    def test_empty_pattern_list_raises_value_error(self, path_backend):
        """Empty patterns list raises ValueError — documented contract."""
        from src.knowledge_graph.query.path_matcher import PathMatcher

        matcher = PathMatcher(backend=path_backend)
        with pytest.raises(ValueError):
            matcher.evaluate("Start", patterns=[])


class TestMaxHopFanout:
    """Path traversal is capped at max_hop_fanout entities per step."""

    def _build_star_backend(self, center: str, n_leaves: int) -> GraphStorageBackend:
        """Return a backend with a center node connected to n_leaves via 'fan_edge'."""
        from src.knowledge_graph.backends.networkx_backend import NetworkXBackend

        b = NetworkXBackend()
        b.add_node(name=center, type="Concept", source="src")
        for i in range(n_leaves):
            leaf = f"Leaf{i}"
            b.add_node(name=leaf, type="Concept", source="src")
            b.add_edge(subject=center, object=leaf, relation="fan_edge", source="src")
        return b

    def test_fanout_guard_truncates_results(self):
        """Frontier with more than max_hop_fanout entries is truncated."""
        from src.knowledge_graph.query.path_matcher import PathMatcher

        fanout_limit = 5
        n_leaves = 20  # intentionally more than fanout_limit
        b = self._build_star_backend("Center", n_leaves)

        matcher = PathMatcher(backend=b, max_hop_fanout=fanout_limit)
        results = matcher.evaluate("Center", patterns=[["fan_edge"]])

        # Must not crash and results must be at most fanout_limit
        assert len(results) <= fanout_limit

    def test_fanout_default_is_50(self):
        """Default max_hop_fanout does not affect small graphs."""
        from src.knowledge_graph.query.path_matcher import PathMatcher

        b = self._build_star_backend("Hub", 10)
        matcher = PathMatcher(backend=b)
        results = matcher.evaluate("Hub", patterns=[["fan_edge"]])

        # All 10 leaves should be returned when n_leaves < 50
        assert len(results) == 10

    def test_fanout_config_field_default(self):
        """KGConfig.max_hop_fanout defaults to 50."""
        from src.knowledge_graph.common.types import KGConfig

        cfg = KGConfig()
        assert cfg.max_hop_fanout == 50

    def test_fanout_config_passed_to_path_matcher(self):
        """PathMatcher respects the max_hop_fanout value it was constructed with."""
        from src.knowledge_graph.query.path_matcher import PathMatcher

        b = self._build_star_backend("Hub", 3)
        matcher = PathMatcher(backend=b, max_hop_fanout=2)
        results = matcher.evaluate("Hub", patterns=[["fan_edge"]])

        assert len(results) <= 2
