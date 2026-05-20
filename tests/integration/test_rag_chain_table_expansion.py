# @summary
# Integration test: wiring of ``expand_table_group_hits`` into ``RAGChain``.
# Covers the post-rerank expansion stage (U1-U4):
#   U1: flag off (default) -> expansion never runs, ordering preserved.
#   U2: flag on, row hit lacks summary -> summary attached.
#   U3: flag on, both row+summary already present -> no duplication.
#   U4: flag on, only non-table hits -> no-op (helper not invoked).
# Drives the bound ``_apply_table_expansion`` method on a RAGChain instance
# constructed via ``__new__`` to avoid loading GPU models in the test path.
# The rerank step is stubbed by handing the method a pre-built ``reranked``
# list — the call site (``_apply_table_expansion``) is what the wiring
# guarantees runs between rerank and context assembly in production.
# Exports: TestRAGChainTableExpansion
# Deps: pytest, unittest.mock, src.retrieval.pipeline.rag_chain, RankedResult
# @end-summary
"""Integration test for the post-rerank table-group expansion wiring."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.retrieval.common import RankedResult
from src.retrieval.pipeline.rag_chain import RAGChain


# ---------------------------------------------------------------------------
# Lightweight RAGChain factory — skips the real __init__ (which loads
# embeddings + reranker + KG + generator) and instead stamps just the
# attributes the expansion stage reads.
# ---------------------------------------------------------------------------


def _make_chain(
    *,
    enabled: bool,
    client: Any,
    max_rows: int = 0,
    fetch_summary: bool = True,
    max_groups: int = 8,
) -> RAGChain:
    chain = RAGChain.__new__(RAGChain)
    chain.enable_table_group_expansion = enabled
    chain.table_expansion_max_rows = max_rows
    chain.table_expansion_fetch_summary = fetch_summary
    chain.table_expansion_max_groups = max_groups
    chain._weaviate_client = client
    return chain


def _wv_obj(props: dict, uuid: str) -> Any:
    obj = MagicMock()
    obj.properties = props
    obj.uuid = uuid
    return obj


def _make_client_returning(gid_to_objs: dict[str, list[Any]]):
    """Fake Weaviate client whose fetch_objects keys on the patched
    ``_filter_for_group`` (the unit tests use the same trick)."""
    client = MagicMock()
    col = MagicMock()
    client.collections.get.return_value = col
    call_log: list[dict] = []

    def _fetch(*, filters=None, limit=None, **_kw):
        gid = getattr(filters, "_table_group_id", None)
        call_log.append({"gid": gid, "limit": limit})
        response = MagicMock()
        response.objects = list(gid_to_objs.get(gid, []))[: (limit or 100)]
        return response

    col.query.fetch_objects.side_effect = _fetch
    client._call_log = call_log  # type: ignore[attr-defined]
    return client


def _row_result(
    *, gid: str, idx: int, chunk_id: str | None = None,
    score: float = 0.9,
) -> RankedResult:
    cid = chunk_id or f"row-{gid}-{idx}"
    return RankedResult(
        text=f"row body {idx}",
        score=score,
        metadata={
            "chunk_id": cid,
            "chunk_type": "table_row",
            "table_group_id": gid,
            "table_row_index": idx,
            "source": "doc.pdf",
        },
    )


def _summary_result(*, gid: str, chunk_id: str | None = None, score: float = 0.7) -> RankedResult:
    cid = chunk_id or f"sum-{gid}"
    return RankedResult(
        text="summary body",
        score=score,
        metadata={
            "chunk_id": cid,
            "chunk_type": "table_summary",
            "table_group_id": gid,
            "table_row_index": -1,
            "source": "doc.pdf",
        },
    )


def _text_result(*, chunk_id: str = "txt-1", score: float = 0.5) -> RankedResult:
    return RankedResult(
        text="plain prose",
        score=score,
        metadata={
            "chunk_id": chunk_id,
            "chunk_type": "text",
            "source": "doc.pdf",
        },
    )


# ---------------------------------------------------------------------------
# U1: flag off (default) -> expansion never runs.
# ---------------------------------------------------------------------------


def test_u1_flag_off_preserves_results(monkeypatch):
    """When the flag is off the helper is not called and ordering is intact."""
    client = MagicMock()
    chain = _make_chain(enabled=False, client=client)

    # Sentry: if the helper is invoked the test fails loudly.
    from src.retrieval.pipeline import rag_chain as mod

    def _boom(*_a, **_kw):
        raise AssertionError("expand_table_group_hits must not be called when flag is off")

    monkeypatch.setattr(mod, "expand_table_group_hits", _boom)

    reranked = [
        _row_result(gid="g1", idx=0),
        _text_result(chunk_id="t1"),
    ]
    out = chain._apply_table_expansion(list(reranked))

    assert out == reranked
    assert [r.metadata["chunk_id"] for r in out] == ["row-g1-0", "t1"]
    client.collections.get.assert_not_called()


# ---------------------------------------------------------------------------
# U2: flag on + row hit missing summary -> summary attached adjacent.
# ---------------------------------------------------------------------------


def test_u2_flag_on_attaches_summary_to_row(monkeypatch):
    """Row hit lacking a summary sibling triggers fetch + adjacent splice."""
    summary_obj = _wv_obj(
        {
            "text": "header + caption",
            "chunk_type": "table_summary",
            "table_group_id": "g1",
            "table_row_index": -1,
            "chunk_id": "sum-g1",
            "source": "doc.pdf",
        },
        uuid="sum-g1",
    )
    client = _make_client_returning({"g1": [summary_obj]})
    chain = _make_chain(enabled=True, client=client)

    # Patch the helper module's filter builder so the fake client can
    # recover the table_group_id off the filter object.
    from src.retrieval import table_group_expansion as helper_mod

    def _fake_filter(gid: str):
        f = MagicMock()
        f._table_group_id = gid
        return f

    monkeypatch.setattr(helper_mod, "_filter_for_group", _fake_filter)

    reranked = [_row_result(gid="g1", idx=0, chunk_id="row-0")]
    out = chain._apply_table_expansion(reranked)

    assert len(out) == 2
    # Summary inserted BEFORE its source row (per helper contract).
    assert out[0].metadata["chunk_type"] == "table_summary"
    assert out[0].metadata["expanded_from"] == "g1"
    assert out[1].metadata["chunk_id"] == "row-0"
    # Exactly one Weaviate read for the one group.
    assert len(client._call_log) == 1


# ---------------------------------------------------------------------------
# U3: flag on + both summary and row already present -> no dedup violation.
# ---------------------------------------------------------------------------


def test_u3_flag_on_no_duplication_when_both_present(monkeypatch):
    """If both summary and row already appear in ``reranked`` nothing is added."""
    # The helper inspects ``summary_present_groups`` first; if summary is
    # in the list it won't fetch. Even so, configure the client to return
    # the same summary in case the helper would query.
    summary_obj = _wv_obj(
        {
            "text": "already here",
            "chunk_type": "table_summary",
            "table_group_id": "g1",
            "chunk_id": "sum-g1",
            "source": "doc.pdf",
        },
        uuid="sum-g1",
    )
    client = _make_client_returning({"g1": [summary_obj]})
    chain = _make_chain(enabled=True, client=client)

    from src.retrieval import table_group_expansion as helper_mod

    def _fake_filter(gid: str):
        f = MagicMock()
        f._table_group_id = gid
        return f

    monkeypatch.setattr(helper_mod, "_filter_for_group", _fake_filter)

    reranked = [
        _summary_result(gid="g1", chunk_id="sum-g1"),
        _row_result(gid="g1", idx=0, chunk_id="row-0"),
    ]
    out = chain._apply_table_expansion(reranked)

    assert len(out) == 2
    assert [r.metadata["chunk_id"] for r in out] == ["sum-g1", "row-0"]
    # No fetch should have fired (summary already present + max_rows=0).
    assert client._call_log == []


# ---------------------------------------------------------------------------
# U4: flag on + non-table hits only -> helper short-circuited.
# ---------------------------------------------------------------------------


def test_u4_flag_on_non_table_hits_noop(monkeypatch):
    """When the result set has zero table chunks the helper is not called."""
    client = MagicMock()
    chain = _make_chain(enabled=True, client=client)

    from src.retrieval.pipeline import rag_chain as mod

    def _boom(*_a, **_kw):
        raise AssertionError(
            "expand_table_group_hits must not be called when no table chunk is present"
        )

    monkeypatch.setattr(mod, "expand_table_group_hits", _boom)

    reranked = [
        _text_result(chunk_id="t1", score=0.9),
        _text_result(chunk_id="t2", score=0.8),
    ]
    out = chain._apply_table_expansion(list(reranked))

    assert [r.metadata["chunk_id"] for r in out] == ["t1", "t2"]
    client.collections.get.assert_not_called()
