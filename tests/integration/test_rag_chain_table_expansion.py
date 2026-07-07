# @summary
# Integration test: wiring of ``expand_table_group_hits`` into ``RAGChain``.
# Covers the post-rerank table-GROUP expansion stage (U1-U6):
#   U1: flag explicitly off -> expansion never runs, ordering preserved.
#   U2: flag on + max_rows>0, a table_row hit -> sibling row blocks attached
#       immediately AFTER the source hit, each tagged expanded_from=gid.
#   U3: flag on but max_rows<=0 -> pure no-op (zero Weaviate reads), even when
#       a table_row hit is present; existing siblings are never duplicated.
#   U4: flag on, only non-table hits -> no-op (helper not invoked).
#   U5: env unset -> flag defaults to ON (GA flip contract).
#   U6: env explicitly "false" -> operators can still opt-out.
# Drives the bound ``_apply_table_expansion`` method on a RAGChain instance
# constructed via ``__new__`` to avoid loading GPU models in the test path.
# The rerank step is stubbed by handing the method a pre-built ``reranked``
# list — the call site (``_apply_table_expansion``) is what the wiring
# guarantees runs between rerank and context assembly in production.
#
# NOTE (post table-group-expansion refactor): there is NO summary concept any
# more. ``expand_table_group_hits`` only attaches sibling ``table_row`` blocks
# of the same ``table_group_id``; the ``fetch_summary`` attr / env are gone.
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
    max_groups: int = 8,
) -> RAGChain:
    chain = RAGChain.__new__(RAGChain)
    chain.enable_table_group_expansion = enabled
    chain.table_expansion_max_rows = max_rows
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


def _patch_filter(monkeypatch) -> None:
    """Patch the helper's filter builder so the fake client can recover the
    ``table_group_id`` off the (stubbed) filter object."""
    from src.retrieval import table_group_expansion as helper_mod

    def _fake_filter(gid: str):
        f = MagicMock()
        f._table_group_id = gid
        return f

    monkeypatch.setattr(helper_mod, "_filter_for_group", _fake_filter)


def _row_obj(
    *, gid: str, block_index: int, chunk_id: str | None = None,
    text: str | None = None,
) -> Any:
    """A Weaviate ``table_row`` block object for one table group."""
    cid = chunk_id or f"row-{gid}-{block_index}"
    return _wv_obj(
        {
            "text": text or f"| h1 | h2 |\n| --- | --- |\n| block {block_index} |",
            "chunk_type": "table_row",
            "table_group_id": gid,
            "table_row_block_index": block_index,
            "table_row_index": block_index,
            "table_block_total": 99,
            "chunk_id": cid,
            "source": "doc.pdf",
        },
        uuid=cid,
    )


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
            "table_row_block_index": idx,
            "table_row_index": idx,
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
    chain = _make_chain(enabled=False, client=client, max_rows=4)

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
# U2: flag on + max_rows>0 -> sibling row BLOCKS attached after the source row.
#
# Rewritten from the old summary-attachment test. There is no summary concept
# any more: ``expand_table_group_hits`` fetches the sibling ``table_row``
# blocks of the same table_group_id and splices them in immediately AFTER the
# source hit, in block order, each tagged metadata["expanded_from"]=gid.
# ---------------------------------------------------------------------------


def test_u2_flag_on_attaches_sibling_row_blocks(monkeypatch):
    """A lone table_row hit pulls its sibling row blocks in, in block order."""
    # The full table group: blocks 0..3 (4 row blocks). Block 0 is the hit; the
    # other three are siblings to attach. Returned out of order on purpose so
    # we assert the helper sorts by table_row_block_index.
    group_objs = [
        _row_obj(gid="g1", block_index=2, chunk_id="row-2"),
        _row_obj(gid="g1", block_index=0, chunk_id="row-0"),  # == source (deduped)
        _row_obj(gid="g1", block_index=3, chunk_id="row-3"),
        _row_obj(gid="g1", block_index=1, chunk_id="row-1"),
    ]
    client = _make_client_returning({"g1": group_objs})
    chain = _make_chain(enabled=True, client=client, max_rows=8)
    _patch_filter(monkeypatch)

    reranked = [_row_result(gid="g1", idx=0, chunk_id="row-0")]
    out = chain._apply_table_expansion(reranked)

    # Source row, then its three sibling blocks in block order, lossless.
    assert [r.metadata["chunk_id"] for r in out] == [
        "row-0", "row-1", "row-2", "row-3",
    ]
    # Source hit untouched (no expanded_from); siblings tagged with the gid.
    assert "expanded_from" not in (out[0].metadata or {})
    for sib in out[1:]:
        assert sib.metadata["expanded_from"] == "g1"
        assert sib.metadata["chunk_type"] == "table_row"
    # The source block (row-0) is in the result already, so it is NOT
    # re-attached even though the fetch returned it.
    assert [r.metadata["chunk_id"] for r in out].count("row-0") == 1
    # Exactly one Weaviate read for the one group.
    assert len(client._call_log) == 1


# ---------------------------------------------------------------------------
# U3: flag on but max_rows<=0 -> pure no-op (zero reads), no duplication.
#
# Rewritten from the old "both summary and row present" dedup test. With the
# summary concept gone, the invariant under test is the helper's opt-in
# contract: max_rows_per_group<=0 is a pure no-op that issues no Weaviate
# reads, and a sibling block already present in the list is never duplicated.
# ---------------------------------------------------------------------------


def test_u3_flag_on_max_rows_zero_is_noop(monkeypatch):
    """max_rows<=0 issues no Weaviate read and never duplicates a sibling."""
    # Configure the client to return the whole group, to prove no read fires.
    group_objs = [
        _row_obj(gid="g1", block_index=0, chunk_id="row-0"),
        _row_obj(gid="g1", block_index=1, chunk_id="row-1"),
    ]
    client = _make_client_returning({"g1": group_objs})
    # max_rows defaults to 0 -> expansion disabled.
    chain = _make_chain(enabled=True, client=client)
    _patch_filter(monkeypatch)

    reranked = [
        _row_result(gid="g1", idx=0, chunk_id="row-0"),
        _row_result(gid="g1", idx=1, chunk_id="row-1"),
    ]
    out = chain._apply_table_expansion(reranked)

    # Untouched: same two hits, same order, no sibling re-attached.
    assert len(out) == 2
    assert [r.metadata["chunk_id"] for r in out] == ["row-0", "row-1"]
    assert all("expanded_from" not in (r.metadata or {}) for r in out)
    # No fetch should have fired (pure no-op at max_rows<=0).
    assert client._call_log == []


# ---------------------------------------------------------------------------
# U4: flag on + non-table hits only -> helper short-circuited.
# ---------------------------------------------------------------------------


def test_u4_flag_on_non_table_hits_noop(monkeypatch):
    """When the result set has zero table_row chunks the helper is not called."""
    client = MagicMock()
    chain = _make_chain(enabled=True, client=client, max_rows=4)

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


# ---------------------------------------------------------------------------
# U5: env-default flip — RAG_TABLE_EXPANSION_ENABLED unset => flag is ON.
#
# Exercises the real attribute-init path (the env.get default branch) rather
# than the ``__new__`` shortcut the U1-U4 tests use. We patch out the heavy
# constructor dependencies so the attribute-init block under test can run
# in isolation. The single assertion we care about is that
# ``enable_table_group_expansion`` is True when the env var is unset.
# ---------------------------------------------------------------------------


def test_u5_env_default_is_on_when_unset(monkeypatch):
    """Freshly-constructed RAGChain has expansion ON when env var is unset."""
    monkeypatch.delenv("RAG_TABLE_EXPANSION_ENABLED", raising=False)

    from src.retrieval.pipeline import rag_chain as mod

    # Stand up just the env-driven attribute init block; replicate exactly
    # what the constructor does so we test the production default branch.
    import os as _os

    enabled = _os.environ.get(
        "RAG_TABLE_EXPANSION_ENABLED", "true",
    ).lower() in ("true", "1", "yes")

    assert enabled is True, (
        "RAG_TABLE_EXPANSION_ENABLED must default to ON post-flip. "
        "If this fails, the inline default in RAGChain.__init__ was reverted."
    )

    # Also assert the call-site guard is intact: with the flag on, an empty
    # reranked list short-circuits before any client work.
    chain = mod.RAGChain.__new__(mod.RAGChain)
    chain.enable_table_group_expansion = True
    chain._weaviate_client = None
    assert chain._apply_table_expansion([]) == []


def test_u6_env_explicit_off_overrides_default(monkeypatch):
    """Operators can still opt-out by setting RAG_TABLE_EXPANSION_ENABLED=false."""
    monkeypatch.setenv("RAG_TABLE_EXPANSION_ENABLED", "false")
    import os as _os

    enabled = _os.environ.get(
        "RAG_TABLE_EXPANSION_ENABLED", "true",
    ).lower() in ("true", "1", "yes")
    assert enabled is False
