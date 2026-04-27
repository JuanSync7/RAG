<!-- @summary
Performance benchmark suite for the knowledge graph retrieval subsystem.
Measures typed traversal latency and context formatting latency against
requirements REQ-KG-1210 and REQ-KG-1212.
@end-summary -->

# tests/benchmarks

Performance benchmarks for the KG retrieval subsystem. Tests are decorated
with `@pytest.mark.benchmark` and measure P95 latency against hard thresholds
(≤ 50ms delta for typed traversal, ≤ 100ms for context formatting). A synthetic
graph fixture builds reproducible large graphs (up to 49K nodes) using realistic
entity-type and edge-type distributions.

## Contents

| Path | Purpose |
| --- | --- |
| `fixtures.py` | `build_synthetic_graph()` — generates a reproducible synthetic `NetworkXBackend` with configurable node count and edge density |
| `kg_retrieval_bench.py` | Benchmarks for typed traversal latency (REQ-KG-1210) and context formatting latency (REQ-KG-1212); includes a CI-safe import smoke test |
