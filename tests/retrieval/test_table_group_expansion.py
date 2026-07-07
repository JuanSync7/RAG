# @summary
# Tests for table-group adjacency expansion at query time.
# Exercises the siblings-only contract: a table_row hit attaches its sibling
# table_row blocks (same table_group_id), inserted in block-index order
# immediately after the source hit, deduped, with per-group and distinct-group
# budget caps. max_rows_per_group<=0 is a pure no-op (zero Weaviate reads).
# There is no summary concept anymore.
# Exports: (pytest test functions)
# Deps: pytest, unittest.mock, src.retrieval.table_group_expansion
# @end-summary
"""Tests for ``expand_table_group_hits`` (table-aware retrieval).

These tests exercise the helper end-to-end with a mocked Weaviate client so
they run without any live infra. Each test maps to a numbered requirement
(R1-R8) from the (refactored) feature contract.

Contract under test (see ``src/retrieval/table_group_expansion.py``):
  * No summary concept. For the FIRST ``table_row`` hit of each
    ``table_group_id``, when ``max_rows_per_group > 0`` the helper fetches the
    sibling ``table_row`` blocks of that group, sorts them by
    ``table_row_block_index`` (falling back to ``table_row_index``), and inserts
    up to ``max_rows_per_group`` of them IMMEDIATELY AFTER the source hit, each
    tagged ``metadata["expanded_from"] = <gid>``.
  * ``max_rows_per_group <= 0`` is a pure no-op and issues zero Weaviate reads.
  * Deduped by ``chunk_id`` (then ``uuid``); distinct groups bounded by
    ``max_groups_to_expand``.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.retrieval.table_group_expansion import expand_table_group_hits


# ---------------------------------------------------------------------------
# Helpers (preserved fake-artifact builders / mock-client / stub-Filter style)
# ---------------------------------------------------------------------------


def _hit(
    *,
    text: str = "",
    chunk_type: str = "text",
    table_group_id: str = "",
    table_row_index: int = -1,
    table_row_block_index: int = -1,
    chunk_id: str = "",
    score: float = 1.0,
) -> dict:
    """Build a retrieval-hit dict in the shape ``hybrid_search`` returns."""
    return {
        "text": text,
        "score": score,
        "uuid": chunk_id or f"uuid-{text}",
        "metadata": {
            "chunk_id": chunk_id or f"uuid-{text}",
            "chunk_type": chunk_type,
            "table_group_id": table_group_id,
            "table_row_index": table_row_index,
            "table_row_block_index": table_row_block_index,
            "source": "doc.pdf",
        },
    }


def _row_obj(
    *,
    chunk_id: str,
    table_group_id: str,
    block_index: int,
    row_index: int | None = None,
    text: str | None = None,
) -> Any:
    """Build a fake Weaviate object for a sibling ``table_row`` block.

    Every block restates the caption + headers, so each carries a stable
    ``table_row_block_index`` used for sibling ordering.
    """
    props = {
        "text": text if text is not None else f"block {block_index}",
        "chunk_type": "table_row",
        "table_group_id": table_group_id,
        "table_row_block_index": block_index,
        "table_row_index": block_index if row_index is None else row_index,
        "chunk_id": chunk_id,
        "source": "doc.pdf",
    }
    return _wv_obj(props, uuid=chunk_id)


def _wv_obj(props: dict, uuid: str) -> Any:
    """Minimal fake Weaviate object."""
    obj = MagicMock()
    obj.properties = props
    obj.uuid = uuid
    return obj


def _make_client(group_to_objects: dict[str, list[Any]]):
    """Build a Weaviate client mock where ``fetch_objects`` is keyed by
    ``table_group_id`` (recovered from the stubbed filter object).

    The helper builds filters via ``_filter_for_group(gid)``; tests patch that
    with a stub that side-loads ``_table_group_id`` onto the returned object so
    this client can route the right fixture. Every call is recorded in
    ``client._call_log`` so tests can assert read-count / no-op behavior.
    """
    client = MagicMock()
    col = MagicMock()
    client.collections.get.return_value = col
    client.collections.exists.return_value = True

    call_log: list[dict[str, Any]] = []

    def _fetch(*, filters=None, limit=None, **_kw):
        gid = getattr(filters, "_table_group_id", None)
        call_log.append({"gid": gid, "limit": limit})
        response = MagicMock()
        response.objects = list(group_to_objects.get(gid, []))[: (limit or 100)]
        return response

    col.query.fetch_objects.side_effect = _fetch
    client._call_log = call_log  # type: ignore[attr-defined]
    return client


def _stub_filter(monkeypatch) -> None:
    """Patch the helper's filter builder so the mock client can recover gid."""
    from src.retrieval import table_group_expansion as mod

    monkeypatch.setattr(
        mod,
        "_filter_for_group",
        lambda gid: MagicMock(_table_group_id=gid),
    )


# ---------------------------------------------------------------------------
# R1: a row hit attaches its SIBLING row blocks, inserted AFTER the source hit
# ---------------------------------------------------------------------------


def test_r1_row_hit_attaches_summary(monkeypatch):
    """A ``table_row`` hit attaches its sibling row blocks (in block order),
    inserted immediately after the source hit and tagged ``expanded_from``.

    (Renamed-in-spirit: there is no summary anymore — R1 is now the
    row-hit -> sibling-rows attachment, which is the core new-world behavior.)
    """
    _stub_filter(monkeypatch)
    siblings = [
        _row_obj(chunk_id="row-0", table_group_id="g1", block_index=0),
        _row_obj(chunk_id="row-1", table_group_id="g1", block_index=1),
        _row_obj(chunk_id="row-2", table_group_id="g1", block_index=2),
    ]
    client = _make_client({"g1": siblings})

    # The source hit is block 0; its siblings (blocks 1 and 2) should attach.
    hits = [
        _hit(
            text="row content",
            chunk_type="table_row",
            table_group_id="g1",
            table_row_block_index=0,
            chunk_id="row-0",
        )
    ]
    out = expand_table_group_hits(
        hits, client=client, collection="C", max_rows_per_group=5
    )

    ids = [h["metadata"]["chunk_id"] for h in out]
    # Source first, then siblings in block order; block 0 is the source itself
    # and is deduped out of the fetched set.
    assert ids == ["row-0", "row-1", "row-2"]
    assert out[0]["metadata"]["chunk_id"] == "row-0"
    assert "expanded_from" not in (out[0]["metadata"])
    for h in out[1:]:
        assert h["metadata"]["chunk_type"] == "table_row"
        assert h["metadata"]["expanded_from"] == "g1"


# ---------------------------------------------------------------------------
# R2: siblings are inserted in BLOCK-INDEX order after the source hit, capped
# ---------------------------------------------------------------------------
#
# REPLACES the old "summary hit -> rows" test. There is no summary hit anymore.
# The equivalent new-world property is ordering + capping of the fetched
# sibling blocks: regardless of fetch order, siblings attach sorted by
# ``table_row_block_index`` and are limited to ``max_rows_per_group``.


def test_r2_summary_hit_attaches_rows(monkeypatch):
    """Sibling blocks attach sorted by block index and capped at
    ``max_rows_per_group`` (fetch order is intentionally shuffled)."""
    _stub_filter(monkeypatch)
    # Returned out of order; helper must sort by table_row_block_index.
    siblings = [
        _row_obj(chunk_id="row-3", table_group_id="g1", block_index=3),
        _row_obj(chunk_id="row-0", table_group_id="g1", block_index=0),
        _row_obj(chunk_id="row-2", table_group_id="g1", block_index=2),
        _row_obj(chunk_id="row-1", table_group_id="g1", block_index=1),
        _row_obj(chunk_id="row-4", table_group_id="g1", block_index=4),
    ]
    client = _make_client({"g1": siblings})

    # Source is block 0; the remaining siblings (blocks 1..4) are candidates,
    # capped to 3 in block order -> 1, 2, 3.
    hits = [
        _hit(
            text="block 0",
            chunk_type="table_row",
            table_group_id="g1",
            table_row_block_index=0,
            chunk_id="row-0",
        )
    ]
    out = expand_table_group_hits(
        hits, client=client, collection="C", max_rows_per_group=3
    )

    assert len(out) == 4
    assert out[0]["metadata"]["chunk_id"] == "row-0"
    inserted = out[1:]
    assert [h["metadata"]["table_row_block_index"] for h in inserted] == [1, 2, 3]
    assert all(h["metadata"]["expanded_from"] == "g1" for h in inserted)
    assert all(h["metadata"]["chunk_type"] == "table_row" for h in inserted)


# ---------------------------------------------------------------------------
# R3: a sibling already present in the hits is not re-attached (dedup)
# ---------------------------------------------------------------------------
#
# REWRITTEN: the old R3 deduped an already-present *summary*. The new-world
# equivalent is that a sibling row block already in the result list is not
# re-attached. (The old test only passed because the default no-op disabled
# expansion; this version actually exercises the dedup path.)


def test_r3_no_duplication_when_summary_already_present(monkeypatch):
    """A sibling row block already present in ``hits`` must not be duplicated
    by expansion."""
    _stub_filter(monkeypatch)
    siblings = [
        _row_obj(chunk_id="row-0", table_group_id="g1", block_index=0),
        _row_obj(chunk_id="row-1", table_group_id="g1", block_index=1),
        _row_obj(chunk_id="row-2", table_group_id="g1", block_index=2),
    ]
    client = _make_client({"g1": siblings})

    hits = [
        _hit(
            text="block 0",
            chunk_type="table_row",
            table_group_id="g1",
            table_row_block_index=0,
            chunk_id="row-0",
        ),
        # row-2 is already a retrieval hit further down the list.
        _hit(
            text="block 2",
            chunk_type="table_row",
            table_group_id="g1",
            table_row_block_index=2,
            chunk_id="row-2",
        ),
    ]
    out = expand_table_group_hits(
        hits, client=client, collection="C", max_rows_per_group=5
    )

    ids = [h["metadata"]["chunk_id"] for h in out]
    # row-1 attaches after the source (row-0); row-2 already present, not dupes.
    assert ids == ["row-0", "row-1", "row-2"]
    assert ids.count("row-2") == 1
    # The pre-existing row-2 hit keeps its original (non-expanded) identity.
    row2 = [h for h in out if h["metadata"]["chunk_id"] == "row-2"][0]
    assert "expanded_from" not in row2["metadata"]


# ---------------------------------------------------------------------------
# R4: max_groups_to_expand cap (still applies)
# ---------------------------------------------------------------------------


def test_r4_max_groups_cap(monkeypatch):
    """Only the first ``max_groups_to_expand`` distinct groups trigger a read /
    get expanded; later groups pass through unexpanded."""
    _stub_filter(monkeypatch)

    def _group_siblings(gid: str):
        return [
            _row_obj(chunk_id=f"{gid}-b0", table_group_id=gid, block_index=0),
            _row_obj(chunk_id=f"{gid}-b1", table_group_id=gid, block_index=1),
        ]

    client = _make_client({f"g{i}": _group_siblings(f"g{i}") for i in range(5)})

    # 5 distinct groups, each represented by its block-0 hit.
    hits = [
        _hit(
            text=f"g{i} block 0",
            chunk_type="table_row",
            table_group_id=f"g{i}",
            table_row_block_index=0,
            chunk_id=f"g{i}-b0",
        )
        for i in range(5)
    ]
    out = expand_table_group_hits(
        hits,
        client=client,
        collection="C",
        max_rows_per_group=5,
        max_groups_to_expand=2,
    )

    # Only 2 groups read from Weaviate.
    assert len(client._call_log) == 2
    # Exactly 2 expanded siblings inserted (one b1 per expanded group).
    expanded = [h for h in out if h["metadata"].get("expanded_from")]
    assert len(expanded) == 2
    assert {h["metadata"]["expanded_from"] for h in expanded} == {"g0", "g1"}
    # All 5 original block-0 hits preserved.
    originals = [
        h
        for h in out
        if not h["metadata"].get("expanded_from")
        and h["metadata"]["chunk_type"] == "table_row"
    ]
    assert len(originals) == 5


# ---------------------------------------------------------------------------
# R5: non-table hits pass through untouched (and trigger no fetch)
# ---------------------------------------------------------------------------


def test_r5_non_table_hits_pass_through(monkeypatch):
    """Hits with chunk_type='text' or absent metadata must not trigger any
    fetch, even when expansion is enabled."""
    _stub_filter(monkeypatch)
    client = _make_client({})

    hits = [
        _hit(text="plain", chunk_type="text", chunk_id="t1"),
        {"text": "no metadata key", "score": 0.5, "metadata": {"chunk_id": "t2"}},
    ]
    out = expand_table_group_hits(
        hits, client=client, collection="C", max_rows_per_group=5
    )
    assert out == hits
    assert client._call_log == []  # zero Weaviate reads


# ---------------------------------------------------------------------------
# R6: max_rows_per_group=0 is a pure no-op and issues ZERO Weaviate reads
# ---------------------------------------------------------------------------
#
# REPLACES the old "fetch_summary_for_row_hits=False" disable test. That kwarg
# (and the summary concept) is removed and now TypeErrors. The new disable
# switch is ``max_rows_per_group=0`` (the default), which must be a pure no-op:
# the input list is returned unchanged and the client is never touched.


def test_r6_disable_row_to_summary(monkeypatch):
    """``max_rows_per_group=0`` (the default) is a pure no-op: hits are
    returned unchanged and no Weaviate read is issued; and the removed
    ``fetch_summary_for_row_hits`` kwarg now TypeErrors."""
    _stub_filter(monkeypatch)
    siblings = [
        _row_obj(chunk_id="row-0", table_group_id="g1", block_index=0),
        _row_obj(chunk_id="row-1", table_group_id="g1", block_index=1),
    ]
    client = _make_client({"g1": siblings})

    hits = [
        _hit(
            text="block 0",
            chunk_type="table_row",
            table_group_id="g1",
            table_row_block_index=0,
            chunk_id="row-0",
        ),
    ]

    # Explicit zero -> no-op.
    out_zero = expand_table_group_hits(
        hits, client=client, collection="C", max_rows_per_group=0
    )
    assert out_zero == hits
    assert client._call_log == []

    # Default (omitted) -> also no-op.
    out_default = expand_table_group_hits(hits, client=client, collection="C")
    assert out_default == hits
    assert client._call_log == []

    # The removed summary toggle is no longer a valid kwarg.
    with pytest.raises(TypeError):
        expand_table_group_hits(
            hits,
            client=client,
            collection="C",
            fetch_summary_for_row_hits=False,  # type: ignore[call-arg]
        )


# ---------------------------------------------------------------------------
# R7: sibling fetch empty -> graceful passthrough
# ---------------------------------------------------------------------------


def test_r7_no_siblings_passthrough(monkeypatch):
    """When the group has no other ``table_row`` chunks, the helper returns the
    input unchanged for that row hit, without raising."""
    _stub_filter(monkeypatch)
    # The group fetch returns only the source block itself (no other siblings).
    client = _make_client(
        {"g1": [_row_obj(chunk_id="row-0", table_group_id="g1", block_index=0)]}
    )

    hits = [
        _hit(
            text="block 0",
            chunk_type="table_row",
            table_group_id="g1",
            table_row_block_index=0,
            chunk_id="row-0",
        )
    ]
    out = expand_table_group_hits(
        hits, client=client, collection="C", max_rows_per_group=5
    )
    # Source block is the only hit returned; it is deduped from the fetch set,
    # leaving nothing to attach.
    assert out == hits


# ---------------------------------------------------------------------------
# R8: original ranking preserved; siblings inserted AFTER the source hit
# ---------------------------------------------------------------------------


def test_r8_original_ranking_preserved(monkeypatch):
    """Non-expanded hits keep their relative order; sibling blocks are inserted
    immediately AFTER the source row hit, in block order."""
    _stub_filter(monkeypatch)
    siblings = [
        _row_obj(chunk_id="row-0", table_group_id="g1", block_index=0),
        _row_obj(chunk_id="row-1", table_group_id="g1", block_index=1),
    ]
    client = _make_client({"g1": siblings})

    hits = [
        _hit(text="A", chunk_type="text", chunk_id="a"),
        _hit(
            text="block 0",
            chunk_type="table_row",
            table_group_id="g1",
            table_row_block_index=0,
            chunk_id="row-0",
        ),
        _hit(text="B", chunk_type="text", chunk_id="b"),
    ]
    out = expand_table_group_hits(
        hits, client=client, collection="C", max_rows_per_group=5
    )
    ids = [h["metadata"]["chunk_id"] for h in out]
    # Sibling row-1 spliced directly AFTER the source row-0; A and B keep their
    # order around the table block.
    assert ids == ["a", "row-0", "row-1", "b"]
    # And the inserted sibling is tagged as an expansion, not a primary hit.
    spliced = out[2]
    assert spliced["metadata"]["chunk_id"] == "row-1"
    assert spliced["metadata"]["expanded_from"] == "g1"
