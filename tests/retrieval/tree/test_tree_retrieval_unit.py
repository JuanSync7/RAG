# @summary
# Unit tests for tree retrieval helpers on RAGChain (TREE_RETRIEVAL_DESIGN.md
# §4). Failing-first scaffold for P2: covers descent, lift, doc-diversity cap,
# and the leaf-only Stage 4 filter that must apply whenever the schema has the
# tree fields (regardless of RAG_TREE_RETRIEVAL_ENABLED).
# @end-summary
"""TDD scaffold for P2 — these tests drive the implementation.

The helpers under test (`_run_tree_descent`, `_run_tree_lift`,
`_apply_doc_diversity_cap`, and the leaf-only filter wired into Stage 4)
do not yet exist on RAGChain. These tests are intentionally failing on
the first run; the implementation in src/retrieval/pipeline/rag_chain.py
must make them green.
"""

from __future__ import annotations

import types
from collections import OrderedDict
from typing import Any

import pytest

from src.retrieval.pipeline.rag_chain import RAGChain
from src.vector_db import SearchFilter


class _DummySpan:
    def set_attribute(self, *_a, **_kw): return None
    def end(self, *_a, **_kw): return None
    def __enter__(self): return self
    def __exit__(self, *_): return False


class _DummyTracer:
    def span(self, *_a, **_kw): return _DummySpan()
    def start_span(self, *_a, **_kw): return _DummySpan()


class _DummyRetryProvider:
    def execute(self, operation_name, fn, policy, idempotency_key):
        return fn()


def _make_chain() -> RAGChain:
    chain = object.__new__(RAGChain)
    chain.tracer = _DummyTracer()
    chain.retry_provider = _DummyRetryProvider()
    chain.retry_policy = object()
    chain._persistent_weaviate = False
    chain._weaviate_client = None
    chain._embedding_cache = OrderedDict()
    chain._embedding_cache_max = 128
    return chain


def _hit(text: str, *, score: float = 0.5, **meta) -> Any:
    """Build a SearchResult-shaped object for stubbed _do_search."""
    base = {"source": "doc", "node_kind": "chunk", "parent_section_id": "",
            "document_id": "doc-A", "heading_path": []}
    base.update(meta)
    return types.SimpleNamespace(text=text, score=score, metadata=base,
                                 collection="default")


# ─── Stage 4 leaf-only filter ─────────────────────────────────────────────

class TestLeafOnlyFilter:
    """When RAG_TREE_SCHEMA_PRESENT, Stage 4 must filter to node_kind=chunk
    so section text never leaks into chunk-only retrieval (regardless of
    RAG_TREE_RETRIEVAL_ENABLED)."""

    def test_chain_exposes_build_chunk_filter_helper(self):
        chain = _make_chain()
        # Helper that builds the default Stage-4 filter list. Must include a
        # node_kind=chunk SearchFilter when schema_present=True.
        clauses = chain._build_leaf_only_filter_clauses(schema_present=True)
        assert any(
            isinstance(c, SearchFilter)
            and c.property == "node_kind"
            and c.operator == "eq"
            and c.value == "chunk"
            for c in clauses
        )

    def test_skipped_on_legacy_schema(self):
        chain = _make_chain()
        clauses = chain._build_leaf_only_filter_clauses(schema_present=False)
        assert not any(
            isinstance(c, SearchFilter) and c.property == "node_kind"
            for c in clauses
        )


# ─── Stage 4b descent ──────────────────────────────────────────────────────

class TestTreeDescent:
    def test_descent_uses_section_filter_and_expands_to_leaves(self, monkeypatch):
        chain = _make_chain()

        section_hits = [
            _hit("Section A text", score=0.9, node_kind="section",
                 chunk_id="sec-a", document_id="doc-A",
                 parent_section_id="psid-A", heading_path=["A"]),
            _hit("Section B text", score=0.8, node_kind="section",
                 chunk_id="sec-b", document_id="doc-B",
                 parent_section_id="psid-B", heading_path=["B"]),
        ]

        captured: dict[str, list] = {"calls": []}

        def fake_do_search(bm25, qv, alpha, limit, filters):
            captured["calls"].append({"bm25": bm25, "filters": filters,
                                      "limit": limit})
            return section_hits

        chain._do_search = fake_do_search  # type: ignore[assignment]

        # Sibling expansion is a separate hook — descent calls
        # _expand_sections_to_leaves with the section parent_section_ids of the
        # hits and the per-section limit.
        expand_calls: list[dict] = []

        def fake_expand(parent_ids, per_section_limit, base_filters):
            expand_calls.append({"parent_ids": list(parent_ids),
                                 "per_section_limit": per_section_limit})
            return [
                _hit("leaf 1 of A", node_kind="chunk",
                     parent_section_id="psid-A", document_id="doc-A"),
                _hit("leaf 1 of B", node_kind="chunk",
                     parent_section_id="psid-B", document_id="doc-B"),
            ]

        chain._expand_sections_to_leaves = fake_expand  # type: ignore[attr-defined]

        result = chain._run_tree_descent(
            bm25_query="what is X",
            query_embedding=[0.1, 0.2],
            alpha=0.5,
            descent_top_k=5,
            leaves_per_section=3,
            base_filters=None,
        )

        # Descent must issue exactly one search filtered to node_kind=section.
        assert len(captured["calls"]) == 1
        flt = captured["calls"][0]["filters"] or []
        assert any(
            getattr(f, "property", None) == "node_kind"
            and getattr(f, "value", None) == "section"
            for f in flt
        )
        # Expansion must be called with the descent hits' chunk_ids — the
        # section's own id (per TREE_RETRIEVAL_DESIGN.md §4.1: leaves under
        # a section have parent_section_id == section.chunk_id).
        assert expand_calls
        assert expand_calls[0]["per_section_limit"] == 3
        assert expand_calls[0]["parent_ids"] == ["sec-a", "sec-b"], (
            f"descent should use section.chunk_id for expansion, got "
            f"{expand_calls[0]['parent_ids']}"
        )
        # Result is the expansion output (leaves), not the section hits.
        assert all(r.metadata.get("node_kind") == "chunk" for r in result)
        assert len(result) == 2

    def test_descent_uses_chunk_id_not_parent_section_id(self):
        """Regression for the P2.5 fix: descent expansion key must be
        ``section.chunk_id``, not ``section.parent_section_id``. The two are
        only the same value for a leaf attached at the section's own depth;
        otherwise descent silently fetches sibling sections rather than the
        section's own children."""
        chain = _make_chain()

        section_hit = _hit(
            "Section text", score=0.9, node_kind="section",
            chunk_id="SECTION-CHUNK-ID",
            parent_section_id="GRANDPARENT-PSID",
            heading_path=["A", "B"],
            document_id="doc-A",
        )

        chain._do_search = lambda *a, **kw: [section_hit]  # type: ignore[assignment]
        captured: dict = {}

        def fake_expand(parent_ids, per_section_limit, base_filters):
            captured["parent_ids"] = list(parent_ids)
            return []

        chain._expand_sections_to_leaves = fake_expand  # type: ignore[attr-defined]

        chain._run_tree_descent(
            bm25_query="q", query_embedding=None, alpha=0.5,
            descent_top_k=5, leaves_per_section=3, base_filters=None,
        )
        assert captured["parent_ids"] == ["SECTION-CHUNK-ID"], (
            "descent must expand using section.chunk_id (the section's own id)"
            f" — got {captured['parent_ids']}"
        )


# ─── Stage 4c lift ──────────────────────────────────────────────────────

class TestTreeLift:
    def test_lift_groups_seeds_by_parent_and_caps_siblings(self):
        chain = _make_chain()

        seeds = [
            _hit("seed1", parent_section_id="psid-1", document_id="doc-A"),
            _hit("seed2", parent_section_id="psid-1", document_id="doc-A"),
            _hit("seed3", parent_section_id="psid-2", document_id="doc-A"),
            _hit("seed4", parent_section_id="", document_id="doc-A"),  # rootless
        ]

        sibling_calls: list[dict] = []

        def fake_fetch_siblings(parent_ids, per_group_limit, base_filters):
            sibling_calls.append({"parent_ids": list(parent_ids),
                                  "per_group_limit": per_group_limit})
            return [
                _hit("sib-1a", parent_section_id="psid-1"),
                _hit("sib-1b", parent_section_id="psid-1"),
                _hit("sib-2a", parent_section_id="psid-2"),
            ]

        chain._fetch_siblings = fake_fetch_siblings  # type: ignore[attr-defined]

        result = chain._run_tree_lift(
            seed_results=seeds,
            seed_top_k=5,
            siblings_per_group=2,
            base_filters=None,
        )

        # Only psid-1 and psid-2 are grouped (rootless seed dropped); each
        # parent appears once in the dispatched call.
        assert sibling_calls
        dispatched = set(sibling_calls[0]["parent_ids"])
        assert dispatched == {"psid-1", "psid-2"}
        assert sibling_calls[0]["per_group_limit"] == 2
        # Lift returns whatever the sibling fetcher gave back.
        assert len(result) == 3


# ─── Doc-diversity cap ──────────────────────────────────────────────────────

class TestDocDiversityCap:
    def test_cap_keeps_top_n_per_document(self):
        chain = _make_chain()
        hits = [
            _hit("a1", score=0.9, document_id="doc-A"),
            _hit("a2", score=0.8, document_id="doc-A"),
            _hit("a3", score=0.7, document_id="doc-A"),
            _hit("b1", score=0.85, document_id="doc-B"),
            _hit("b2", score=0.6, document_id="doc-B"),
        ]
        capped = chain._apply_doc_diversity_cap(hits, top_per_doc=2)
        kept_per_doc: dict[str, int] = {}
        for h in capped:
            d = h.metadata["document_id"]
            kept_per_doc[d] = kept_per_doc.get(d, 0) + 1
        assert kept_per_doc == {"doc-A": 2, "doc-B": 2}
        # Within each doc, the highest-scoring entries are kept.
        kept_a = [h.text for h in capped if h.metadata["document_id"] == "doc-A"]
        assert "a1" in kept_a and "a2" in kept_a
        assert "a3" not in kept_a

    def test_cap_preserves_overall_score_order(self):
        chain = _make_chain()
        hits = [
            _hit("b1", score=0.95, document_id="doc-B"),
            _hit("a1", score=0.9, document_id="doc-A"),
            _hit("a2", score=0.4, document_id="doc-A"),
        ]
        capped = chain._apply_doc_diversity_cap(hits, top_per_doc=2)
        # Ordering of survivors follows input score order.
        assert [h.text for h in capped] == ["b1", "a1", "a2"]


# ─── Integration helper: _collect_candidates ───────────────────────────────

class TestCollectCandidates:
    """Composite helper that owns Stage 4 + 4b + 4c. Ensures:
      - default leaf filter is added when schema is present;
      - descent/lift fire only when the retrieval flag is on;
      - merged result is dedup'd by uuid (with text fallback)."""

    def _wire(self, chain, *, leaf_hits, descent_hits=None, lift_hits=None):
        captured: dict[str, Any] = {"do_search_calls": [],
                                    "descent_calls": 0, "lift_calls": 0}

        def fake_do_search(bm25, qv, alpha, limit, filters):
            captured["do_search_calls"].append({"filters": filters,
                                                "limit": limit})
            return list(leaf_hits)

        def fake_descent(*args, **kwargs):
            captured["descent_calls"] += 1
            return list(descent_hits or [])

        def fake_lift(*args, **kwargs):
            captured["lift_calls"] += 1
            return list(lift_hits or [])

        chain._do_search = fake_do_search  # type: ignore
        chain._run_tree_descent = fake_descent  # type: ignore
        chain._run_tree_lift = fake_lift  # type: ignore
        return captured

    def test_disabled_path_skips_tree_stages(self):
        chain = _make_chain()
        leaf_hits = [_hit("leaf", document_id="d", parent_section_id="p")]
        captured = self._wire(chain, leaf_hits=leaf_hits)

        candidates = chain._collect_candidates(
            bm25_query="q", query_embedding=[0.1], alpha=0.5,
            search_limit=10, base_filters=[],
            tree_enabled=False, schema_present=True,
            descent_top_k=5, leaves_per_section=3,
            lift_seed_k=5, siblings_per_group=2,
            doc_diversity_top_per_doc=2,
        )
        assert captured["descent_calls"] == 0
        assert captured["lift_calls"] == 0
        # Only one Stage-4 call.
        assert len(captured["do_search_calls"]) == 1
        # And it carries the leaf-only filter clause.
        flt = captured["do_search_calls"][0]["filters"] or []
        assert any(getattr(f, "property", None) == "node_kind"
                   and getattr(f, "value", None) == "chunk" for f in flt)
        assert candidates == leaf_hits

    def test_enabled_path_invokes_descent_and_lift_and_merges(self):
        chain = _make_chain()
        leaf = _hit("L1", document_id="d1", parent_section_id="p1")
        descent_leaf = _hit("D1", document_id="d2", parent_section_id="p2")
        lift_leaf = _hit("S1", document_id="d1", parent_section_id="p1")
        captured = self._wire(
            chain, leaf_hits=[leaf],
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
        assert captured["descent_calls"] == 1
        assert captured["lift_calls"] == 1
        texts = [c.text for c in candidates]
        # All three sources contribute.
        assert "L1" in texts and "D1" in texts and "S1" in texts

    def test_run_with_tree_off_routes_through_collect_candidates_without_tree(
        self, monkeypatch
    ):
        """Regression gate: with RAG_TREE_RETRIEVAL_ENABLED=false (default),
        run() must invoke _collect_candidates with tree_enabled=False so that
        no descent/lift fires. Behavioral assertion per the design doc's
        regression-gate decision."""
        from src.retrieval.query.schemas import QueryAction, QueryResult

        chain = _make_chain()
        chain._kg_expander = None
        chain._generator = None
        chain.reranker = None
        chain._guardrails_input_executor = None
        chain._guardrails_output_executor = None
        chain._guardrails_merge_gate = None
        chain._visual_retrieval_enabled = False
        chain._visual_model = None
        chain._visual_processor = None

        class _Embeddings:
            def embed_query(self, _q): return [0.1, 0.2, 0.3]
        chain.embeddings = _Embeddings()

        monkeypatch.setattr(
            "src.retrieval.pipeline.rag_chain.process_query",
            lambda *a, **kw: QueryResult(
                processed_query="processed",
                confidence=0.9,
                action=QueryAction.SEARCH,
                clarification_message=None,
                iterations=1,
            ),
        )
        monkeypatch.setattr(
            "src.retrieval.pipeline.rag_chain.RAG_TREE_RETRIEVAL_ENABLED",
            False,
        )

        captured: dict[str, Any] = {"tree_enabled_seen": None,
                                    "calls": 0}

        def fake_collect(*, tree_enabled, **kwargs):
            captured["tree_enabled_seen"] = tree_enabled
            captured["calls"] += 1
            return []  # empty results → run returns "no results" RAGResponse

        chain._collect_candidates = fake_collect  # type: ignore

        chain.run(
            "what is X",
            skip_generation=True,
            stage_budget_overrides={"query_processing": 100_000},
        )
        assert captured["calls"] >= 1
        assert captured["tree_enabled_seen"] is False, (
            "regression gate: run() must propagate the disabled flag"
        )

    def test_run_with_tree_on_propagates_flag(self, monkeypatch):
        from src.retrieval.query.schemas import QueryAction, QueryResult

        chain = _make_chain()
        chain._kg_expander = None
        chain._generator = None
        chain.reranker = None
        chain._guardrails_input_executor = None
        chain._guardrails_output_executor = None
        chain._guardrails_merge_gate = None
        chain._visual_retrieval_enabled = False
        chain._visual_model = None
        chain._visual_processor = None

        class _Embeddings:
            def embed_query(self, _q): return [0.1, 0.2, 0.3]
        chain.embeddings = _Embeddings()

        monkeypatch.setattr(
            "src.retrieval.pipeline.rag_chain.process_query",
            lambda *a, **kw: QueryResult(
                processed_query="processed",
                confidence=0.9,
                action=QueryAction.SEARCH,
                clarification_message=None,
                iterations=1,
            ),
        )
        monkeypatch.setattr(
            "src.retrieval.pipeline.rag_chain.RAG_TREE_RETRIEVAL_ENABLED",
            True,
        )

        captured: dict[str, Any] = {"tree_enabled_seen": None}
        chain._collect_candidates = (  # type: ignore
            lambda *, tree_enabled, **kwargs: (
                captured.update(tree_enabled_seen=tree_enabled) or []
            )
        )

        chain.run(
            "what is X", skip_generation=True,
            stage_budget_overrides={"query_processing": 100_000},
        )
        assert captured["tree_enabled_seen"] is True

    def test_legacy_schema_omits_node_kind_filter(self):
        chain = _make_chain()
        captured = self._wire(chain, leaf_hits=[])
        chain._collect_candidates(
            bm25_query="q", query_embedding=[0.1], alpha=0.5,
            search_limit=10, base_filters=[],
            tree_enabled=False, schema_present=False,
            descent_top_k=5, leaves_per_section=3,
            lift_seed_k=5, siblings_per_group=2,
            doc_diversity_top_per_doc=2,
        )
        flt = captured["do_search_calls"][0]["filters"] or []
        assert not any(
            getattr(f, "property", None) == "node_kind" for f in flt
        )
