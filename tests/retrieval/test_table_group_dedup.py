# @summary
# Unit tests for RAGChain._collapse_table_groups — collapses different renderings
# of the SAME table (shared table_group_id) to one best-scoring rep so two facets
# of one table (e.g. a markdown grid + a row-prose linearization) cannot both
# occupy the reranked top-K and read as a duplicate.
# Deps: pytest, src.retrieval.pipeline.rag_chain, src.retrieval.common.schemas
# @end-summary
"""Unit tests for the table-group representation collapse."""
from __future__ import annotations

import src.retrieval.pipeline.rag_chain as rc
from src.retrieval.common.schemas import RankedResult
from src.retrieval.pipeline.rag_chain import RAGChain


def _r(score, gid="", md="", doc="d", text=None):
    meta = {"document_id": doc}
    if gid:
        meta["table_group_id"] = gid
    if md:
        meta["table_markdown"] = md
    return RankedResult(text=text or f"{gid or 'x'}:{score}", score=score, metadata=meta)


def test_collapses_two_reps_of_same_table(monkeypatch):
    """Two renderings of one table (same group) collapse to the best-scoring rep."""
    monkeypatch.setattr(rc, "RAG_TABLE_GROUP_DEDUP_ENABLED", True)
    ranked = [_r(0.9, gid="g1", md="| A | B |"), _r(0.7, gid="g1", text="A: 1 | B: 2")]
    out = RAGChain._collapse_table_groups(ranked)
    assert len(out) == 1
    assert out[0].score == 0.9  # highest-scoring rep survives


def test_survivor_inherits_dropped_markdown(monkeypatch):
    """If the best rep lacks the grid, it inherits the dropped rep's table_markdown."""
    monkeypatch.setattr(rc, "RAG_TABLE_GROUP_DEDUP_ENABLED", True)
    ranked = [_r(0.9, gid="g1", text="A: 1 | B: 2"), _r(0.7, gid="g1", md="| A | B |")]
    out = RAGChain._collapse_table_groups(ranked)
    assert len(out) == 1
    assert out[0].metadata.get("table_markdown") == "| A | B |"  # display preserved


def test_survivor_keeps_its_own_markdown(monkeypatch):
    """A survivor that already has its own grid is not overwritten by a dropped rep."""
    monkeypatch.setattr(rc, "RAG_TABLE_GROUP_DEDUP_ENABLED", True)
    ranked = [_r(0.9, gid="g1", md="GRID-A"), _r(0.7, gid="g1", md="GRID-B")]
    out = RAGChain._collapse_table_groups(ranked)
    assert out[0].metadata["table_markdown"] == "GRID-A"


def test_distinct_groups_all_kept(monkeypatch):
    monkeypatch.setattr(rc, "RAG_TABLE_GROUP_DEDUP_ENABLED", True)
    ranked = [_r(0.9, gid="g1"), _r(0.8, gid="g2"), _r(0.7, gid="g3")]
    assert len(RAGChain._collapse_table_groups(ranked)) == 3


def test_non_table_chunks_pass_through(monkeypatch):
    """Chunks without a table_group_id are never collapsed."""
    monkeypatch.setattr(rc, "RAG_TABLE_GROUP_DEDUP_ENABLED", True)
    ranked = [_r(0.9), _r(0.8), _r(0.7)]
    assert len(RAGChain._collapse_table_groups(ranked)) == 3


def test_mixed_preserves_order_and_collapses_only_groups(monkeypatch):
    monkeypatch.setattr(rc, "RAG_TABLE_GROUP_DEDUP_ENABLED", True)
    ranked = [
        _r(0.95, text="prose1"),          # non-table
        _r(0.90, gid="g1", md="| A |"),   # table g1 (kept)
        _r(0.85, gid="g1", text="rows"),  # table g1 (dropped)
        _r(0.80, text="prose2"),          # non-table
        _r(0.75, gid="g2"),               # table g2 (kept)
    ]
    out = RAGChain._collapse_table_groups(ranked)
    assert [round(r.score, 2) for r in out] == [0.95, 0.90, 0.80, 0.75]


def test_disabled_is_passthrough(monkeypatch):
    monkeypatch.setattr(rc, "RAG_TABLE_GROUP_DEDUP_ENABLED", False)
    ranked = [_r(0.9, gid="g1"), _r(0.7, gid="g1")]
    assert len(RAGChain._collapse_table_groups(ranked)) == 2  # both kept when disabled
