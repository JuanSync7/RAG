# @summary
# Performance benchmarks for KG retrieval: typed traversal and context formatting.
# Exports: test_typed_traversal_latency, test_formatting_latency, test_benchmark_imports
# Deps: pytest, time, tests.benchmarks.fixtures
# @end-summary
"""Performance benchmarks for KG retrieval typed traversal and context formatting.

REQ-KG-1210: Typed traversal <= 50ms P95 delta vs untyped (< 50K nodes).
REQ-KG-1212: Context formatting <= 100ms P95 (< 500 tokens).
"""

from __future__ import annotations

import time

import pytest


benchmark = pytest.mark.benchmark


def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile of a sorted list."""
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_vals):
        return sorted_vals[f]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


@benchmark
def test_typed_traversal_latency():
    """REQ-KG-1210: P95 delta (typed - untyped) <= 50ms on 49K-node graph."""
    from tests.benchmarks.fixtures import build_synthetic_graph

    backend = build_synthetic_graph(num_nodes=49_000)
    stats = backend.stats()
    assert stats.get("nodes", 0) >= 40_000, f"Graph too small: {stats}"

    # Pick seed entities that exist
    all_entities = backend.get_all_entities()
    seed_names = [e.name for e in all_entities[:200]]

    # Untyped baseline
    untyped_times = []
    for name in seed_names[:100]:
        t0 = time.perf_counter()
        backend.query_neighbors(name, depth=1)
        untyped_times.append(time.perf_counter() - t0)

    # Typed traversal
    typed_times = []
    edge_types = ["depends_on", "fixed_by", "specified_by"]
    for name in seed_names[100:200]:
        t0 = time.perf_counter()
        backend.query_neighbors_typed(name, edge_types, depth=1)
        typed_times.append(time.perf_counter() - t0)

    p95_untyped = _percentile(untyped_times, 95) * 1000  # ms
    p95_typed = _percentile(typed_times, 95) * 1000  # ms
    delta_ms = p95_typed - p95_untyped

    print(f"P95 untyped: {p95_untyped:.1f}ms, typed: {p95_typed:.1f}ms, delta: {delta_ms:.1f}ms")
    assert delta_ms <= 50.0, f"P95 delta {delta_ms:.1f}ms exceeds 50ms threshold"


@benchmark
@pytest.mark.parametrize("token_budget", [100, 300, 500])
def test_formatting_latency(token_budget: int):
    """REQ-KG-1212: Context formatting P95 <= 100ms for <= 500 tokens."""
    from tests.benchmarks.fixtures import build_synthetic_graph
    from src.knowledge_graph.query.context_formatter import GraphContextFormatter

    backend = build_synthetic_graph(num_nodes=1000)  # smaller for formatting

    formatter = GraphContextFormatter(token_budget=token_budget)

    all_entities = backend.get_all_entities()[:20]
    triples = []
    for e in all_entities[:5]:
        triples.extend(backend.get_outgoing_edges(e.name))

    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        formatter.format(all_entities, triples, [])
        times.append(time.perf_counter() - t0)

    p95_ms = _percentile(times, 95) * 1000
    print(f"Formatting P95 at {token_budget} tokens: {p95_ms:.1f}ms")
    assert p95_ms <= 100.0, f"P95 {p95_ms:.1f}ms exceeds 100ms threshold"


def test_benchmark_imports():
    """Smoke test: verify benchmark modules can be imported (runs in CI)."""
    from tests.benchmarks.fixtures import build_synthetic_graph
    from src.knowledge_graph.query.context_formatter import GraphContextFormatter

    # Just verify imports work
    assert callable(build_synthetic_graph)
    assert callable(GraphContextFormatter)
