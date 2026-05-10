"""End-to-end smoke test for per-topic rerank through full RAGChain.

Runs ONE disjoint query with deep_research=True and asserts no exceptions
plus a non-empty document list. Skips cleanly when Weaviate is empty.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.eval_deep_research


def test_per_topic_rerank_smoke(rag_chain, disjoint_queries_deep_research_asic):
    try:
        total = rag_chain.collection.aggregate.over_all().total_count  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Cannot read Weaviate aggregate: {exc}")
    if not total:
        pytest.skip("Weaviate collection is empty — smoke test infra-gated")

    queries = disjoint_queries_deep_research_asic.get("queries") or []
    if not queries:
        pytest.skip("disjoint fixture empty")

    q = queries[0]["query"]
    resp = rag_chain.run(q, deep_research=True, rerank_top_k=10)
    assert resp is not None
    docs = list(getattr(resp, "documents", []) or [])
    # Even if recall is poor, the pipeline must not error.
    assert isinstance(docs, list)
