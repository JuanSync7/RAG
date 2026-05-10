# @summary
# Community detection using the Leiden algorithm over igraph.
# Converts backend graph to undirected igraph, runs Leiden partitioning,
# applies min-size filtering, and persists results via sidecar JSON.
# Exports: CommunityDetector
# Deps: igraph, leidenalg (optional), json, tempfile, os, logging,
#       src.knowledge_graph.backend, src.knowledge_graph.common.types,
#       src.knowledge_graph.community.schemas
# @end-summary
"""Community detection for knowledge graph clustering.

Uses the Leiden algorithm (python-igraph + leidenalg) to detect communities
in the knowledge graph. Supports sidecar persistence, diff tracking between
runs, and integration with LLM-generated community summaries.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from typing import Dict, List, Optional, Set, Tuple

from src.knowledge_graph.backend import GraphStorageBackend
from src.knowledge_graph.common import KGConfig
from src.knowledge_graph.community.schemas import CommunityDiff, CommunitySummary

__all__ = ["CommunityDetector"]

logger = logging.getLogger(__name__)

# Try-import for optional Leiden dependencies
_LEIDEN_AVAILABLE = False
try:
    import igraph  # noqa: F401
    import leidenalg  # noqa: F401

    _LEIDEN_AVAILABLE = True
except ImportError:
    igraph = None  # type: ignore[assignment]
    leidenalg = None  # type: ignore[assignment]


class CommunityDetector:
    """Detects communities in the knowledge graph using the Leiden algorithm.

    When igraph/leidenalg are unavailable, operates in degraded mode:
    ``detect()`` returns an empty dict and ``is_ready`` stays False.
    """

    def __init__(
        self,
        backend: GraphStorageBackend,
        config: KGConfig,
        graph_path: Optional[str] = None,
    ) -> None:
        self._backend = backend
        self._config = config
        self._graph_path = graph_path or config.graph_path

        self._communities: Dict[int, List[str]] = {}
        self._entity_to_community: Dict[str, int] = {}
        self._previous_assignments: Dict[str, int] = {}
        self._summaries: Dict[int, CommunitySummary] = {}
        self._diff: Optional[CommunityDiff] = None
        self._detection_complete: bool = False

        # Phase 3: Hierarchical Leiden
        self._hierarchy: Dict[Tuple[int, int], List[str]] = {}
        self._parent_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self._hierarchy_summaries: Dict[Tuple[int, int], CommunitySummary] = {}

        # Load existing sidecar on init
        self._load_sidecar()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True when detection is complete and at least one summary exists."""
        return self._detection_complete and len(self._summaries) > 0

    def detect(self) -> Dict[int, List[str]]:
        """Run Leiden community detection on the backend graph.

        Returns:
            Mapping of community_id to list of entity names.
            Empty dict when Leiden dependencies are unavailable.
        """
        if not _LEIDEN_AVAILABLE:
            logger.warning(
                "igraph/leidenalg not installed — community detection skipped"
            )
            return {}

        _t0 = time.monotonic()
        ig = self._to_igraph()
        if ig.vcount() == 0:
            logger.info("Empty graph — no communities to detect")
            self._detection_complete = True
            return {}

        raw_communities = self._run_leiden(ig)
        communities = self._apply_min_size(raw_communities)

        self._diff = self._compute_diff(communities)
        self._previous_assignments = dict(self._entity_to_community)

        self._communities = communities
        self._entity_to_community = {
            name: cid for cid, members in communities.items() for name in members
        }
        self._assign_communities(communities)
        self._detection_complete = True

        n_communities = len([c for c in communities if c != -1])
        n_entities = sum(len(m) for m in communities.values())
        elapsed = time.monotonic() - _t0
        logger.info(
            "CommunityDetector.detect: %d communities, %d entities, elapsed=%.1fs",
            n_communities, n_entities, elapsed,
        )
        logger.debug(
            "CommunityDetector.detect: communities=%d entities=%d elapsed=%.3fs",
            n_communities, n_entities, elapsed,
        )
        return communities

    def detect_hierarchical(
        self,
    ) -> Dict[Tuple[int, int], List[str]]:
        """Run hierarchical Leiden: recursive partitioning at multiple levels.

        Returns:
            Flat mapping ``{(level, community_id): [entity_names]}``.
            Level 0 is the coarsest (fewest, largest communities).
            Also populates ``parent_map`` and ``hierarchy`` properties.
        """
        # First run flat detection for level 0
        level0 = self.detect()
        if not level0:
            return {}

        max_levels = getattr(self._config, "community_max_levels", 3)
        self._hierarchy = {}
        self._parent_map = {}

        # Level 0: the initial flat partition
        for cid, members in level0.items():
            self._hierarchy[(0, cid)] = members

        if max_levels <= 1 or not _LEIDEN_AVAILABLE:
            return self._hierarchy

        # Recursive sub-partitioning for levels 1+
        for level in range(1, max_levels):
            parent_communities = {
                k: v for k, v in self._hierarchy.items() if k[0] == level - 1
            }
            found_any = False

            for (plevel, pcid), members in parent_communities.items():
                if len(members) < self._config.community_min_size * 2:
                    # Too small to split further
                    continue

                # Build subgraph for this community
                sub_ig = self._to_igraph_subset(members)
                if sub_ig.vcount() < self._config.community_min_size * 2:
                    continue

                sub_communities = self._run_leiden(sub_ig)
                sub_communities = self._apply_min_size(sub_communities)

                # Only keep if we got more than 1 real community
                real = {k: v for k, v in sub_communities.items() if k != -1}
                if len(real) < 2:
                    continue

                found_any = True
                for sub_cid, sub_members in sub_communities.items():
                    key = (level, sub_cid + pcid * 1000)  # unique ID
                    self._hierarchy[key] = sub_members
                    self._parent_map[key] = (plevel, pcid)

            if not found_any:
                break

        logger.info(
            "Hierarchical Leiden: %d levels, %d total partitions",
            max(k[0] for k in self._hierarchy) + 1 if self._hierarchy else 0,
            len(self._hierarchy),
        )
        return self._hierarchy

    @property
    def hierarchy(self) -> Dict[Tuple[int, int], List[str]]:
        """The hierarchical partition, populated by ``detect_hierarchical()``."""
        return self._hierarchy

    @property
    def parent_map(self) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """Maps each (level, cid) to its parent (level-1, parent_cid)."""
        return self._parent_map

    @property
    def hierarchy_summaries(self) -> Dict[Tuple[int, int], CommunitySummary]:
        """Per-level community summaries."""
        return self._hierarchy_summaries

    @hierarchy_summaries.setter
    def hierarchy_summaries(
        self, value: Dict[Tuple[int, int], CommunitySummary]
    ) -> None:
        self._hierarchy_summaries = value

    @property
    def diff(self) -> Optional[CommunityDiff]:
        """Diff from the most recent ``detect()`` call, or None."""
        return self._diff

    def get_community_for_entity(self, name: str) -> Optional[int]:
        """Return the community ID for *name*, or None if unassigned."""
        return self._entity_to_community.get(name)

    def get_community_members(self, community_id: int) -> List[str]:
        """Return entity names belonging to *community_id*."""
        return list(self._communities.get(community_id, []))

    def get_summary(self, community_id: int) -> Optional[CommunitySummary]:
        """Return the summary for *community_id*, or None."""
        return self._summaries.get(community_id)

    @property
    def summaries(self) -> Dict[int, CommunitySummary]:
        """All community summaries indexed by community ID."""
        return self._summaries

    @summaries.setter
    def summaries(self, value: Dict[int, CommunitySummary]) -> None:
        self._summaries = value

    # ------------------------------------------------------------------
    # Sidecar persistence
    # ------------------------------------------------------------------

    def save_sidecar(self) -> None:
        """Persist communities and summaries to ``<graph_path>.communities.json``.

        Uses atomic write (tempfile + os.replace) to avoid corruption.
        """
        if not self._graph_path:
            logger.debug("No graph_path configured — skipping sidecar save")
            return

        sidecar_path = self._graph_path + ".communities.json"
        payload = {
            "version": 1,
            "summaries": {
                str(cid): {
                    "community_id": s.community_id,
                    "summary_text": s.summary_text,
                    "member_count": s.member_count,
                    "member_names": s.member_names,
                    "generated_at": s.generated_at,
                }
                for cid, s in self._summaries.items()
            },
            "previous_assignments": {
                name: cid for name, cid in self._entity_to_community.items()
            },
        }

        dir_name = os.path.dirname(sidecar_path) or "."
        try:
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2)
                os.replace(tmp_path, sidecar_path)
                logger.debug("Saved community sidecar to %s", sidecar_path)
            except BaseException:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError:
            logger.exception("Failed to save community sidecar to %s", sidecar_path)

    def _load_sidecar(self) -> None:
        """Load community data from sidecar JSON if it exists."""
        if not self._graph_path:
            return

        sidecar_path = self._graph_path + ".communities.json"
        if not os.path.isfile(sidecar_path):
            return

        try:
            with open(sidecar_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            # Restore summaries
            for cid_str, s_data in data.get("summaries", {}).items():
                cid = int(cid_str)
                self._summaries[cid] = CommunitySummary(
                    community_id=s_data["community_id"],
                    summary_text=s_data["summary_text"],
                    member_count=s_data["member_count"],
                    member_names=s_data.get("member_names", []),
                    generated_at=s_data.get("generated_at", ""),
                )

            # Restore previous assignments for diff computation
            for name, cid in data.get("previous_assignments", {}).items():
                self._previous_assignments[name] = int(cid)

            logger.debug("Loaded community sidecar from %s", sidecar_path)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            logger.warning(
                "Corrupt community sidecar at %s — treating as first run",
                sidecar_path,
            )
            self._summaries.clear()
            self._previous_assignments.clear()

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _to_igraph(self) -> "igraph.Graph":
        """Convert the backend graph to an undirected igraph.Graph.

        Directed edges A->B and B->A are collapsed into a single undirected
        edge with weight = max(weight_AB, weight_BA).
        """
        try:
            entities = self._backend.get_all_entities()
            name_to_idx: Dict[str, int] = {}
            names: List[str] = []

            for entity in entities:
                if entity.name not in name_to_idx:
                    name_to_idx[entity.name] = len(names)
                    names.append(entity.name)

            # Collect edges: for each directed edge, track max weight per
            # undirected pair (canonical ordering by sorted tuple).
            edge_weights: Dict[tuple, float] = {}
            for entity in entities:
                outgoing = self._backend.get_outgoing_edges(entity.name)
                for triple in outgoing:
                    src = triple.subject
                    tgt = triple.object
                    if src not in name_to_idx or tgt not in name_to_idx:
                        continue
                    key = tuple(sorted((src, tgt)))
                    edge_weights[key] = max(edge_weights.get(key, 0.0), triple.weight)

            ig = igraph.Graph(n=len(names), directed=False)
            ig.vs["name"] = names

            edges = []
            weights = []
            for (a, b), w in edge_weights.items():
                edges.append((name_to_idx[a], name_to_idx[b]))
                weights.append(w)

            if edges:
                ig.add_edges(edges)
                ig.es["weight"] = weights

            return ig
        except MemoryError as exc:
            logger.critical(
                "_to_igraph failed: out of memory building igraph for %d entities "
                "and %d edges — community detection aborted",
                len(names) if "names" in dir() else -1,
                len(edge_weights) if "edge_weights" in dir() else -1,
            )
            raise
        except Exception as exc:
            logger.error(
                "_to_igraph failed with %s: %s — entities=%s edges=%s",
                type(exc).__name__,
                exc,
                len(names) if "names" in dir() else -1,
                len(edge_weights) if "edge_weights" in dir() else -1,
            )
            raise

    def _to_igraph_subset(self, member_names: List[str]) -> "igraph.Graph":
        """Build an igraph subgraph for a specific set of entity names."""
        member_set = set(member_names)
        name_to_idx: Dict[str, int] = {}
        names: List[str] = []

        for name in member_names:
            name_to_idx[name] = len(names)
            names.append(name)

        ig = igraph.Graph(n=len(names), directed=False)
        ig.vs["name"] = names

        edge_weights: Dict[tuple, float] = {}
        for name in member_names:
            neighbors = self._backend.query_neighbors(name, depth=1)
            for neighbor in neighbors:
                if neighbor.name not in member_set:
                    continue
                a, b = sorted([name, neighbor.name])
                key = (a, b)
                if key not in edge_weights:
                    edge_weights[key] = 1.0

        edges = []
        weights = []
        for (a, b), w in edge_weights.items():
            if a in name_to_idx and b in name_to_idx:
                edges.append((name_to_idx[a], name_to_idx[b]))
                weights.append(w)

        if edges:
            ig.add_edges(edges)
            ig.es["weight"] = weights

        return ig

    def _run_leiden(self, ig: "igraph.Graph") -> Dict[int, List[str]]:
        """Execute Leiden algorithm and return community partition."""
        partition = leidenalg.find_partition(
            ig,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=self._config.community_resolution,
            seed=42,
            weights="weight" if ig.ecount() > 0 else None,
        )

        communities: Dict[int, List[str]] = {}
        for cid, members in enumerate(partition):
            entity_names = [ig.vs[idx]["name"] for idx in members]
            if entity_names:
                communities[cid] = entity_names

        return communities

    def _apply_min_size(
        self, communities: Dict[int, List[str]]
    ) -> Dict[int, List[str]]:
        """Merge communities smaller than min_size into bucket -1."""
        min_size = self._config.community_min_size
        result: Dict[int, List[str]] = {}
        bucket: List[str] = []

        for cid, members in communities.items():
            if len(members) < min_size:
                bucket.extend(members)
            else:
                result[cid] = members

        if bucket:
            result[-1] = bucket

        return result

    def _assign_communities(self, communities: Dict[int, List[str]]) -> None:
        """Store community_id on each backend node."""
        from src.knowledge_graph.backends import NetworkXBackend

        if isinstance(self._backend, NetworkXBackend):
            # Direct access for efficiency
            for cid, members in communities.items():
                for name in members:
                    if self._backend.graph.has_node(name):
                        self._backend.graph.nodes[name]["community_id"] = cid
        else:
            # Generic fallback: re-upsert with community_id via add_node
            for cid, members in communities.items():
                for name in members:
                    entity = self._backend.get_entity(name)
                    if entity is not None:
                        self._backend.add_node(
                            name=entity.name,
                            type=entity.type,
                            source="",
                            aliases=entity.aliases or None,
                        )

    def _compute_diff(
        self, new_communities: Dict[int, List[str]]
    ) -> CommunityDiff:
        """Compare previous assignments with new partition."""
        # Build old partition: {community_id: {entity_names}}
        old_partition: Dict[int, Set[str]] = {}
        for name, cid in self._previous_assignments.items():
            old_partition.setdefault(cid, set()).add(name)

        # Build new partition: {community_id: {entity_names}}
        new_partition: Dict[int, Set[str]] = {
            cid: set(members) for cid, members in new_communities.items()
        }

        old_ids = set(old_partition.keys())
        new_ids = set(new_partition.keys())

        new_cids = new_ids - old_ids
        removed_cids = old_ids - new_ids
        common_cids = old_ids & new_ids

        changed: Set[int] = set()
        unchanged: Set[int] = set()
        for cid in common_cids:
            if old_partition[cid] == new_partition[cid]:
                unchanged.add(cid)
            else:
                changed.add(cid)

        return CommunityDiff(
            new_communities=new_cids,
            removed_communities=removed_cids,
            changed_communities=changed,
            unchanged_communities=unchanged,
        )
