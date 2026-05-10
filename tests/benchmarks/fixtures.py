# @summary
# Synthetic graph fixtures for KG retrieval benchmarks.
# Exports: build_synthetic_graph
# Deps: src.knowledge_graph.backends.networkx_backend, src.knowledge_graph.common
# @end-summary
"""Synthetic graph fixtures for KG retrieval benchmarks."""

from __future__ import annotations

import random
from typing import List

from src.knowledge_graph.backends.networkx_backend import NetworkXBackend
from src.knowledge_graph.common import Entity, Triple


# Edge type distribution matching realistic ASIC design graph
_EDGE_TYPES = [
    "depends_on", "contains", "instantiates", "connects_to",
    "specified_by", "fixed_by", "constrained_by", "authored_by",
    "drives", "reads",
]


def build_synthetic_graph(
    num_nodes: int = 49_000,
    avg_edges_per_node: float = 3.0,
    seed: int = 42,
) -> NetworkXBackend:
    """Build a synthetic graph for benchmarking.

    Generates entities with realistic type distribution and typed edges.

    Args:
        num_nodes: Number of entities to create.
        avg_edges_per_node: Average outgoing edges per entity.
        seed: Random seed for reproducibility.

    Returns:
        Populated NetworkXBackend instance.
    """
    rng = random.Random(seed)
    backend = NetworkXBackend()

    # Entity types with realistic distribution
    entity_types = [
        ("RTL_Module", 0.15),
        ("Port", 0.20),
        ("Signal", 0.15),
        ("KnownIssue", 0.10),
        ("DesignDecision", 0.10),
        ("Specification", 0.08),
        ("ClockDomain", 0.05),
        ("Parameter", 0.07),
        ("Person", 0.05),
        ("Concept", 0.05),
    ]

    # Generate entities
    entities: List[Entity] = []
    for i in range(num_nodes):
        # Weighted random type selection
        r = rng.random()
        cumulative = 0.0
        chosen_type = entity_types[-1][0]
        for etype, weight in entity_types:
            cumulative += weight
            if r < cumulative:
                chosen_type = etype
                break

        entity = Entity(
            name=f"entity_{i:06d}",
            type=chosen_type,
        )
        entities.append(entity)

    # Bulk upsert entities
    backend.upsert_entities(entities)

    # Generate typed edges
    num_edges = int(num_nodes * avg_edges_per_node)
    triples: List[Triple] = []
    for _ in range(num_edges):
        src_idx = rng.randint(0, num_nodes - 1)
        dst_idx = rng.randint(0, num_nodes - 1)
        if src_idx == dst_idx:
            continue
        edge_type = rng.choice(_EDGE_TYPES)
        triples.append(Triple(
            subject=entities[src_idx].name,
            predicate=edge_type,
            object=entities[dst_idx].name,
            source="benchmark",
        ))

    backend.upsert_triples(triples)

    return backend
