# @summary
# Unit tests for the B2 slice — routed-doc candidate union in
# RAGChain._collect_candidates (DOCUMENT_ROUTING_DESIGN.md §6.2, §7). Drive the
# implementation that adds a 4th candidate source (a hybrid search restricted to
# the routed document_ids via the `in` filter), unions+dedups it with the
# existing (flat/descent/lift) sources, ensures per-routed-doc representation,
# and bounds the merged pool. CRITICAL: routed_doc_ids=None must be byte-identical
# to today.
# @end-summary
"""TDD scaffold for B2 (routed-doc candidate union).

These tests construct a RAGChain via ``object.__new__`` (no live backend) and
patch ``_do_search`` / ``_run_tree_descent`` / ``_run_tree_lift`` on the
instance, mirroring tests/retrieval/tree/test_tree_retrieval_unit.py.
"""

from __future__ import annotations

import types
from collections import OrderedDict
from typing import Any

from src.retrieval.pipeline.rag_chain import RAGChain
from src.vector_db import SearchFilter


class _DummySpan:
    def set_attribute(self, *_a, **_kw):
        return None

    def end(self, *_a, **_kw):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _DummyTracer:
    def span(self, *_a, **_kw):
        return _DummySpan()

    def start_span(self, *_a, **_kw):
        return _DummySpan()


def _make_chain() -> RAGChain:
    chain = object.__new__(RAGChain)
    chain.tracer = _DummyTracer()
    chain.retry_policy = object()
    chain._persistent_weaviate = False
    chain._weaviate_client = None
    chain._embedding_cache = OrderedDict()
    chain._embedding_cache_max = 128
    return chain


def _hit(text: str, *, score: float = 0.5, **meta) -> Any:
    """Build a SearchResult-shaped object for stubbed _do_search."""
    base = {
        "source": "doc",
        "node_kind": "chunk",
        "parent_section_id": "",
        "document_id": "doc-A",
        "heading_path": [],
    }
    base.update(meta)
    return types.SimpleNamespace(
        text=text, score=score, metadata=base, collection="default"
    )


class _Wiring:
    """Helper to patch the four search hooks and record their calls.

    ``_do_search`` returns ``leaf_hits`` on the FIRST (flat) call and
    ``routed_hits`` on every subsequent (routed) call — this lets a single test
    distinguish the flat source from the routed source while still exercising
    the real merge/dedup/bound logic.
    """

    def __init__(self, chain, *, leaf_hits, routed_hits=None,
                 descent_hits=None, lift_hits=None):
        self.do_search_calls: list[dict] = []
        self.descent_calls = 0
        self.lift_calls = 0
        self._leaf_hits = list(leaf_hits)
        self._routed_hits = list(routed_hits or [])
        self._descent_hits = list(descent_hits or [])
        self._lift_hits = list(lift_hits or [])

        def fake_do_search(bm25, qv, alpha, limit, filters):
            call_index = len(self.do_search_calls)
            self.do_search_calls.append(
                {"filters": filters, "limit": limit, "bm25": bm25}
            )
            if call_index == 0:
                return list(self._leaf_hits)
            return list(self._routed_hits)

        def fake_descent(*args, **kwargs):
            self.descent_calls += 1
            return list(self._descent_hits)

        def fake_lift(*args, **kwargs):
            self.lift_calls += 1
            return list(self._lift_hits)

        chain._do_search = fake_do_search  # type: ignore[assignment]
        chain._run_tree_descent = fake_descent  # type: ignore[assignment]
        chain._run_tree_lift = fake_lift  # type: ignore[assignment]


# ─── No-regression: routed_doc_ids=None ────────────────────────────────────


class TestNoRoutingByteIdentical:
    """routed_doc_ids=None (the default) must behave exactly as today."""

    def test_none_does_not_issue_routed_search_tree_off(self):
        chain = _make_chain()
        leaf_hits = [_hit("leaf", document_id="d", parent_section_id="p")]
        w = _Wiring(chain, leaf_hits=leaf_hits, routed_hits=[_hit("routed")])

        candidates = chain._collect_candidates(
            bm25_query="q", query_embedding=[0.1], alpha=0.5,
            search_limit=10, base_filters=[],
            tree_enabled=False, schema_present=True,
            descent_top_k=5, leaves_per_section=3,
            lift_seed_k=5, siblings_per_group=2,
            doc_diversity_top_per_doc=2,
        )
        # Only the flat Stage-4 search fired — no routed search.
        assert len(w.do_search_calls) == 1
        assert candidates == leaf_hits

    def test_none_does_not_issue_routed_search_tree_on(self):
        chain = _make_chain()
        leaf = _hit("L1", document_id="d1", parent_section_id="p1")
        descent_leaf = _hit("D1", document_id="d2", parent_section_id="p2")
        lift_leaf = _hit("S1", document_id="d3", parent_section_id="p3")
        w = _Wiring(
            chain, leaf_hits=[leaf], routed_hits=[_hit("routed")],
            descent_hits=[descent_leaf], lift_hits=[lift_leaf],
        )

        candidates = chain._collect_candidates(
            bm25_query="q", query_embedding=[0.1], alpha=0.5,
            search_limit=10, base_filters=[],
            tree_enabled=True, schema_present=True,
            descent_top_k=5, leaves_per_section=3,
            lift_seed_k=5, siblings_per_group=2,
            doc_diversity_top_per_doc=2,
        )
        # Exactly one _do_search call (the flat path) — descent/lift use their
        # own stubs and do not route through _do_search here.
        assert len(w.do_search_calls) == 1
        assert w.descent_calls == 1
        assert w.lift_calls == 1
        texts = [c.text for c in candidates]
        assert texts == ["L1", "D1", "S1"]
        # No "routed" chunk leaked in.
        assert "routed" not in texts

    def test_empty_list_treated_as_no_routing(self):
        chain = _make_chain()
        leaf_hits = [_hit("leaf", document_id="d")]
        w = _Wiring(chain, leaf_hits=leaf_hits, routed_hits=[_hit("routed")])

        candidates = chain._collect_candidates(
            bm25_query="q", query_embedding=[0.1], alpha=0.5,
            search_limit=10, base_filters=[],
            tree_enabled=False, schema_present=True,
            descent_top_k=5, leaves_per_section=3,
            lift_seed_k=5, siblings_per_group=2,
            doc_diversity_top_per_doc=2,
            routed_doc_ids=[],
        )
        assert len(w.do_search_calls) == 1
        assert candidates == leaf_hits


# ─── Routed-doc candidate union ─────────────────────────────────────────────


class TestRoutedUnion:
    def test_routed_search_uses_in_filter_with_doc_ids(self):
        chain = _make_chain()
        leaf_hits = [_hit("leaf", document_id="d0")]
        routed_hits = [
            _hit("r1", document_id="d1"),
            _hit("r2", document_id="d2"),
        ]
        w = _Wiring(chain, leaf_hits=leaf_hits, routed_hits=routed_hits)

        chain._collect_candidates(
            bm25_query="q", query_embedding=[0.1], alpha=0.5,
            search_limit=10, base_filters=[],
            tree_enabled=False, schema_present=True,
            descent_top_k=5, leaves_per_section=3,
            lift_seed_k=5, siblings_per_group=2,
            doc_diversity_top_per_doc=2,
            routed_doc_ids=["d1", "d2"],
        )
        # Two _do_search calls: flat (first) + routed (second).
        assert len(w.do_search_calls) == 2
        routed_filters = w.do_search_calls[1]["filters"] or []
        in_clauses = [
            f for f in routed_filters
            if getattr(f, "property", None) == "document_id"
            and getattr(f, "operator", None) == "in"
        ]
        assert in_clauses, (
            "routed search must carry a SearchFilter(document_id, in, ...)"
        )
        clause = in_clauses[0]
        assert isinstance(clause, SearchFilter)
        assert "d1" in clause.value and "d2" in clause.value

    def test_routed_filters_extend_base_leaf_filters(self):
        chain = _make_chain()
        base = [SearchFilter(property="tenant_id", operator="eq", value="t1")]
        w = _Wiring(
            chain,
            leaf_hits=[_hit("leaf", document_id="d0")],
            routed_hits=[_hit("r1", document_id="d1")],
        )
        chain._collect_candidates(
            bm25_query="q", query_embedding=[0.1], alpha=0.5,
            search_limit=10, base_filters=base,
            tree_enabled=False, schema_present=True,
            descent_top_k=5, leaves_per_section=3,
            lift_seed_k=5, siblings_per_group=2,
            doc_diversity_top_per_doc=2,
            routed_doc_ids=["d1"],
        )
        routed_filters = w.do_search_calls[1]["filters"] or []
        # Base tenant filter preserved.
        assert any(
            getattr(f, "property", None) == "tenant_id" for f in routed_filters
        )
        # Leaf-only chunk filter preserved (schema_present=True).
        assert any(
            getattr(f, "property", None) == "node_kind"
            and getattr(f, "value", None) == "chunk"
            for f in routed_filters
        )
        # The new document_id `in` filter is present.
        assert any(
            getattr(f, "property", None) == "document_id"
            and getattr(f, "operator", None) == "in"
            for f in routed_filters
        )

    def test_routed_chunks_appear_in_merged_output(self):
        chain = _make_chain()
        leaf_hits = [_hit("flat-1", document_id="d0")]
        routed_hits = [
            _hit("routed-1", document_id="d1"),
            _hit("routed-2", document_id="d2"),
        ]
        _Wiring(chain, leaf_hits=leaf_hits, routed_hits=routed_hits)

        candidates = chain._collect_candidates(
            bm25_query="q", query_embedding=[0.1], alpha=0.5,
            search_limit=10, base_filters=[],
            tree_enabled=False, schema_present=True,
            descent_top_k=5, leaves_per_section=3,
            lift_seed_k=5, siblings_per_group=2,
            doc_diversity_top_per_doc=2,
            routed_doc_ids=["d1", "d2"],
        )
        texts = [c.text for c in candidates]
        assert "flat-1" in texts
        assert "routed-1" in texts
        assert "routed-2" in texts
        # Flat keeps priority (appears before routed in the merge).
        assert texts.index("flat-1") < texts.index("routed-1")

    def test_chunk_in_both_flat_and_routed_dedupes_once(self):
        chain = _make_chain()
        # Same chunk_id surfaced by both the flat and routed searches.
        shared_flat = _hit("shared", document_id="d1", chunk_id="C-SHARED")
        shared_routed = _hit("shared", document_id="d1", chunk_id="C-SHARED")
        leaf_hits = [shared_flat, _hit("flat-only", document_id="d0",
                                       chunk_id="C-FLAT")]
        routed_hits = [shared_routed, _hit("routed-only", document_id="d1",
                                           chunk_id="C-ROUTED")]
        _Wiring(chain, leaf_hits=leaf_hits, routed_hits=routed_hits)

        candidates = chain._collect_candidates(
            bm25_query="q", query_embedding=[0.1], alpha=0.5,
            search_limit=10, base_filters=[],
            tree_enabled=False, schema_present=True,
            descent_top_k=5, leaves_per_section=3,
            lift_seed_k=5, siblings_per_group=2,
            doc_diversity_top_per_doc=2,
            routed_doc_ids=["d1"],
        )
        keys = [c.metadata.get("chunk_id") for c in candidates]
        assert keys.count("C-SHARED") == 1, (
            "a chunk found by both flat and routed must dedupe to one entry"
        )
        # And the first occurrence (flat) is the one kept.
        shared_entries = [c for c in candidates
                          if c.metadata.get("chunk_id") == "C-SHARED"]
        assert shared_entries[0] is shared_flat

    def test_per_doc_cap_respected_on_routed_source(self):
        chain = _make_chain()
        leaf_hits = [_hit("flat", document_id="d0", chunk_id="F0")]
        # Three routed chunks all from d1; per_doc_leaves=2 must cap to 2.
        routed_hits = [
            _hit("r1", document_id="d1", score=0.9, chunk_id="R1"),
            _hit("r2", document_id="d1", score=0.8, chunk_id="R2"),
            _hit("r3", document_id="d1", score=0.7, chunk_id="R3"),
        ]
        _Wiring(chain, leaf_hits=leaf_hits, routed_hits=routed_hits)

        candidates = chain._collect_candidates(
            bm25_query="q", query_embedding=[0.1], alpha=0.5,
            search_limit=10, base_filters=[],
            tree_enabled=False, schema_present=True,
            descent_top_k=5, leaves_per_section=3,
            lift_seed_k=5, siblings_per_group=2,
            doc_diversity_top_per_doc=2,
            routed_doc_ids=["d1"],
            per_doc_leaves=2,
        )
        routed_texts = [
            c.text for c in candidates if c.text in {"r1", "r2", "r3"}
        ]
        assert len(routed_texts) == 2, (
            "routed source must be doc-diversity capped to per_doc_leaves"
        )
        # Highest-scoring two survive.
        assert "r1" in routed_texts and "r2" in routed_texts
        assert "r3" not in routed_texts


# ─── Bound on the merged pool ───────────────────────────────────────────────


class TestUnionBound:
    def test_merged_pool_truncated_to_max_candidates(self):
        chain = _make_chain()
        # 5 distinct flat chunks, 5 distinct routed chunks → union 10.
        leaf_hits = [
            _hit(f"flat-{i}", document_id=f"f{i}", chunk_id=f"F{i}")
            for i in range(5)
        ]
        routed_hits = [
            _hit(f"routed-{i}", document_id="d1", chunk_id=f"R{i}", score=0.5)
            for i in range(5)
        ]
        _Wiring(chain, leaf_hits=leaf_hits, routed_hits=routed_hits)

        candidates = chain._collect_candidates(
            bm25_query="q", query_embedding=[0.1], alpha=0.5,
            search_limit=10, base_filters=[],
            tree_enabled=False, schema_present=True,
            descent_top_k=5, leaves_per_section=3,
            lift_seed_k=5, siblings_per_group=2,
            doc_diversity_top_per_doc=2,
            routed_doc_ids=["d1"],
            per_doc_leaves=5,
            max_candidates=4,
        )
        assert len(candidates) == 4, (
            "merged pool must be truncated to max_candidates when routing active"
        )
        # Order preserved: the first four flat chunks survive the truncation.
        assert [c.text for c in candidates] == [
            "flat-0", "flat-1", "flat-2", "flat-3"
        ]

    def test_no_bound_applied_when_routing_inactive(self):
        chain = _make_chain()
        # 5 flat chunks; no routing → bound must NOT apply even if config small.
        leaf_hits = [
            _hit(f"flat-{i}", document_id=f"f{i}", chunk_id=f"F{i}")
            for i in range(5)
        ]
        _Wiring(chain, leaf_hits=leaf_hits)
        candidates = chain._collect_candidates(
            bm25_query="q", query_embedding=[0.1], alpha=0.5,
            search_limit=10, base_filters=[],
            tree_enabled=False, schema_present=True,
            descent_top_k=5, leaves_per_section=3,
            lift_seed_k=5, siblings_per_group=2,
            doc_diversity_top_per_doc=2,
            routed_doc_ids=None,
            max_candidates=2,
        )
        # All 5 flat chunks survive — no bound on the None path.
        assert len(candidates) == 5


# ─── Tree disabled vs enabled both add the routed source ───────────────────


class TestTreeModesWithRouting:
    def test_routed_source_added_when_tree_disabled(self):
        chain = _make_chain()
        leaf_hits = [_hit("flat", document_id="d0", chunk_id="F0")]
        routed_hits = [_hit("routed", document_id="d1", chunk_id="R0")]
        w = _Wiring(chain, leaf_hits=leaf_hits, routed_hits=routed_hits)

        candidates = chain._collect_candidates(
            bm25_query="q", query_embedding=[0.1], alpha=0.5,
            search_limit=10, base_filters=[],
            tree_enabled=False, schema_present=True,
            descent_top_k=5, leaves_per_section=3,
            lift_seed_k=5, siblings_per_group=2,
            doc_diversity_top_per_doc=2,
            routed_doc_ids=["d1"],
        )
        assert w.descent_calls == 0
        assert w.lift_calls == 0
        texts = [c.text for c in candidates]
        assert "flat" in texts and "routed" in texts

    def test_routed_source_added_when_tree_enabled(self):
        chain = _make_chain()
        leaf = _hit("L1", document_id="d0", chunk_id="L1", parent_section_id="p1")
        descent_leaf = _hit("D1", document_id="d2", chunk_id="D1",
                             parent_section_id="p2")
        lift_leaf = _hit("S1", document_id="d3", chunk_id="S1",
                         parent_section_id="p3")
        routed = _hit("R1", document_id="d1", chunk_id="R1")
        w = _Wiring(
            chain, leaf_hits=[leaf], routed_hits=[routed],
            descent_hits=[descent_leaf], lift_hits=[lift_leaf],
        )

        candidates = chain._collect_candidates(
            bm25_query="q", query_embedding=[0.1], alpha=0.5,
            search_limit=10, base_filters=[],
            tree_enabled=True, schema_present=True,
            descent_top_k=5, leaves_per_section=3,
            lift_seed_k=5, siblings_per_group=2,
            doc_diversity_top_per_doc=2,
            routed_doc_ids=["d1"],
        )
        assert w.descent_calls == 1
        assert w.lift_calls == 1
        texts = [c.text for c in candidates]
        # All four sources contribute; routed is last in merge order.
        assert texts == ["L1", "D1", "S1", "R1"]


# ─── Routed search limit ────────────────────────────────────────────────────


class TestRoutedSearchLimit:
    def test_routed_limit_scales_with_doc_count_and_per_doc_leaves(self):
        chain = _make_chain()
        w = _Wiring(
            chain,
            leaf_hits=[_hit("flat", document_id="d0")],
            routed_hits=[_hit("r", document_id="d1")],
        )
        chain._collect_candidates(
            bm25_query="q", query_embedding=[0.1], alpha=0.5,
            search_limit=10, base_filters=[],
            tree_enabled=False, schema_present=True,
            descent_top_k=5, leaves_per_section=3,
            lift_seed_k=5, siblings_per_group=2,
            doc_diversity_top_per_doc=2,
            routed_doc_ids=["d1", "d2", "d3"],
            per_doc_leaves=4,
            max_candidates=60,
        )
        routed_limit = w.do_search_calls[1]["limit"]
        # >= per_doc_leaves * num_routed_docs so every routed doc can fill up.
        assert routed_limit >= 4 * 3
        # And bounded (never exceeds the union cap).
        assert routed_limit <= 60
