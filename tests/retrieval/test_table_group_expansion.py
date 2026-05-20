# @summary
# Tests for table-group adjacency expansion at query time.
# Exercises row -> summary fetch, summary -> rows fetch, dedup, budget caps,
# and ranking preservation against a mocked Weaviate client.
# Exports: (pytest test functions)
# Deps: pytest, unittest.mock, src.retrieval.table_group_expansion
# @end-summary
"""Tests for ``expand_table_group_hits`` (table-aware retrieval).

These tests exercise the helper end-to-end with a mocked Weaviate client so
they run without any live infra. Each test maps to a numbered requirement
(R1-R8) from the feature spec.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.retrieval.table_group_expansion import expand_table_group_hits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hit(
    *,
    text: str = "",
    chunk_type: str = "text",
    table_group_id: str = "",
    table_row_index: int = -1,
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
            "source": "doc.pdf",
        },
    }


def _wv_obj(props: dict, uuid: str) -> Any:
    """Minimal fake Weaviate object."""
    obj = MagicMock()
    obj.properties = props
    obj.uuid = uuid
    return obj


def _make_client(group_to_objects: dict[str, list[Any]]):
    """Build a Weaviate client mock where fetch_objects is keyed by
    ``table_group_id`` (extracted from the filter call).

    The helper builds filters via ``Filter.by_property("table_group_id").equal(...)``;
    we don't introspect the filter object — instead we use a per-call counter
    of ``side_effect`` to return per-group fixtures in invocation order.
    """
    client = MagicMock()
    col = MagicMock()
    client.collections.get.return_value = col
    client.collections.exists.return_value = True

    # Order responses by group id as they're requested. We capture calls and
    # return the prepared list each time.
    call_log: list[dict[str, Any]] = []

    def _fetch(*, filters=None, limit=None, **_kw):
        # The helper passes table_group_id via filter; we encode it on the
        # filter mock by stashing it as ``_table_group_id`` so the test
        # client can resolve it. Production filters obviously don't have
        # this attr — the helper code below sets it when building the call.
        gid = getattr(filters, "_table_group_id", None)
        call_log.append({"gid": gid, "limit": limit})
        response = MagicMock()
        response.objects = list(group_to_objects.get(gid, []))[: (limit or 100)]
        return response

    col.query.fetch_objects.side_effect = _fetch
    client._call_log = call_log  # type: ignore[attr-defined]
    return client


# ---------------------------------------------------------------------------
# R1: row hit -> summary sibling attached, inserted before the row
# ---------------------------------------------------------------------------


def test_r1_row_hit_attaches_summary(monkeypatch):
    """Row hit triggers fetch of its table_summary sibling."""
    summary_obj = _wv_obj(
        {
            "text": "Summary text",
            "chunk_type": "table_summary",
            "table_group_id": "g1",
            "table_row_index": -1,
            "source": "doc.pdf",
            "chunk_id": "sum-1",
        },
        uuid="sum-1",
    )
    client = _make_client({"g1": [summary_obj]})

    # Patch the helper's filter builder so the mock can recover gid.
    from src.retrieval import table_group_expansion as mod

    def _fake_filter(gid: str):
        f = MagicMock()
        f._table_group_id = gid
        return f

    monkeypatch.setattr(mod, "_filter_for_group", _fake_filter)

    hits = [_hit(text="row content", chunk_type="table_row", table_group_id="g1",
                 table_row_index=0, chunk_id="row-0")]
    out = expand_table_group_hits(hits, client=client, collection="C")

    assert len(out) == 2
    assert out[0]["metadata"]["chunk_type"] == "table_summary"
    assert out[0]["metadata"]["expanded_from"] == "g1"
    assert out[1]["metadata"]["chunk_id"] == "row-0"


# ---------------------------------------------------------------------------
# R2: summary hit -> up to N row chunks sorted by table_row_index
# ---------------------------------------------------------------------------


def test_r2_summary_hit_attaches_rows(monkeypatch):
    """Summary hit triggers fetch of row siblings, sorted, capped."""
    rows = [
        _wv_obj({"text": "r2", "chunk_type": "table_row", "table_group_id": "g1",
                 "table_row_index": 2, "chunk_id": "row-2"}, uuid="row-2"),
        _wv_obj({"text": "r0", "chunk_type": "table_row", "table_group_id": "g1",
                 "table_row_index": 0, "chunk_id": "row-0"}, uuid="row-0"),
        _wv_obj({"text": "r1", "chunk_type": "table_row", "table_group_id": "g1",
                 "table_row_index": 1, "chunk_id": "row-1"}, uuid="row-1"),
        _wv_obj({"text": "r3", "chunk_type": "table_row", "table_group_id": "g1",
                 "table_row_index": 3, "chunk_id": "row-3"}, uuid="row-3"),
        # include the summary itself in the returned list - helper must skip it
        _wv_obj({"text": "summary", "chunk_type": "table_summary",
                 "table_group_id": "g1", "table_row_index": -1,
                 "chunk_id": "sum-1"}, uuid="sum-1"),
    ]
    client = _make_client({"g1": rows})

    from src.retrieval import table_group_expansion as mod

    monkeypatch.setattr(
        mod, "_filter_for_group",
        lambda gid: MagicMock(_table_group_id=gid),
    )

    hits = [_hit(text="summary", chunk_type="table_summary", table_group_id="g1",
                 chunk_id="sum-1")]
    out = expand_table_group_hits(hits, client=client, collection="C",
                                   max_rows_per_group=3)

    # Summary at original position + 3 rows after, sorted by row_index 0,1,2
    assert len(out) == 4
    assert out[0]["metadata"]["chunk_id"] == "sum-1"
    inserted = out[1:]
    assert [h["metadata"]["table_row_index"] for h in inserted] == [0, 1, 2]
    assert all(h["metadata"]["expanded_from"] == "g1" for h in inserted)


# ---------------------------------------------------------------------------
# R3: result already contains both row and summary - no duplication
# ---------------------------------------------------------------------------


def test_r3_no_duplication_when_summary_already_present(monkeypatch):
    """If the summary is already a hit, row expansion must not duplicate."""
    summary_obj = _wv_obj(
        {"text": "Summary", "chunk_type": "table_summary", "table_group_id": "g1",
         "table_row_index": -1, "chunk_id": "sum-1"},
        uuid="sum-1",
    )
    client = _make_client({"g1": [summary_obj]})

    from src.retrieval import table_group_expansion as mod
    monkeypatch.setattr(
        mod, "_filter_for_group",
        lambda gid: MagicMock(_table_group_id=gid),
    )

    hits = [
        _hit(text="row", chunk_type="table_row", table_group_id="g1",
             table_row_index=0, chunk_id="row-0"),
        _hit(text="summary", chunk_type="table_summary", table_group_id="g1",
             chunk_id="sum-1"),
    ]
    out = expand_table_group_hits(hits, client=client, collection="C")

    # No expansion added: dedup short-circuits because summary already present.
    assert len(out) == 2
    ids = [h["metadata"]["chunk_id"] for h in out]
    assert ids == ["row-0", "sum-1"]


# ---------------------------------------------------------------------------
# R4: max_groups_to_expand cap
# ---------------------------------------------------------------------------


def test_r4_max_groups_cap(monkeypatch):
    """Only the first ``max_groups_to_expand`` distinct groups get expanded."""

    def _summary_for(gid: str):
        return _wv_obj(
            {"text": f"S-{gid}", "chunk_type": "table_summary",
             "table_group_id": gid, "table_row_index": -1,
             "chunk_id": f"sum-{gid}"},
            uuid=f"sum-{gid}",
        )

    client = _make_client({f"g{i}": [_summary_for(f"g{i}")] for i in range(5)})

    from src.retrieval import table_group_expansion as mod
    monkeypatch.setattr(
        mod, "_filter_for_group",
        lambda gid: MagicMock(_table_group_id=gid),
    )

    hits = [
        _hit(text=f"row{i}", chunk_type="table_row", table_group_id=f"g{i}",
             table_row_index=0, chunk_id=f"row-{i}")
        for i in range(5)
    ]
    out = expand_table_group_hits(hits, client=client, collection="C",
                                   max_groups_to_expand=2)

    # Only 2 summaries should be inserted.
    summaries = [h for h in out
                 if h["metadata"].get("chunk_type") == "table_summary"]
    assert len(summaries) == 2
    # Original 5 row hits preserved.
    rows = [h for h in out if h["metadata"].get("chunk_type") == "table_row"]
    assert len(rows) == 5


# ---------------------------------------------------------------------------
# R5: non-table hits pass through untouched
# ---------------------------------------------------------------------------


def test_r5_non_table_hits_pass_through():
    """Hits with chunk_type='text' or absent must not trigger any fetch."""
    client = MagicMock()
    client.collections.exists.return_value = True
    col = MagicMock()
    client.collections.get.return_value = col

    hits = [
        _hit(text="plain", chunk_type="text", chunk_id="t1"),
        {"text": "no metadata key", "score": 0.5, "metadata": {"chunk_id": "t2"}},
    ]
    out = expand_table_group_hits(hits, client=client, collection="C")
    assert out == hits
    col.query.fetch_objects.assert_not_called()


# ---------------------------------------------------------------------------
# R6: fetch_summary_for_row_hits=False disables row->summary
# ---------------------------------------------------------------------------


def test_r6_disable_row_to_summary(monkeypatch):
    """Row hits do NOT trigger summary fetch when the toggle is off, but
    summary hits still fetch rows."""
    rows = [
        _wv_obj({"text": "r0", "chunk_type": "table_row", "table_group_id": "g1",
                 "table_row_index": 0, "chunk_id": "row-0"}, uuid="row-0"),
    ]
    client = _make_client({"g1": rows, "g2": rows})

    from src.retrieval import table_group_expansion as mod
    monkeypatch.setattr(
        mod, "_filter_for_group",
        lambda gid: MagicMock(_table_group_id=gid),
    )

    hits = [
        _hit(text="row", chunk_type="table_row", table_group_id="g2",
             chunk_id="row-other"),
        _hit(text="sum", chunk_type="table_summary", table_group_id="g1",
             chunk_id="sum-1"),
    ]
    out = expand_table_group_hits(
        hits, client=client, collection="C",
        fetch_summary_for_row_hits=False,
        max_rows_per_group=2,
    )
    types = [h["metadata"].get("chunk_type") for h in out]
    # Original two hits preserved + at most 1 row expanded under summary.
    assert types.count("table_summary") == 1
    assert "table_row" in types
    # No summary was inserted (row->summary disabled).
    summaries = [h for h in out
                 if h["metadata"].get("chunk_type") == "table_summary"]
    assert len(summaries) == 1
    assert summaries[0]["metadata"]["chunk_id"] == "sum-1"


# ---------------------------------------------------------------------------
# R7: sibling fetch empty -> graceful passthrough
# ---------------------------------------------------------------------------


def test_r7_no_siblings_passthrough(monkeypatch):
    """When the group has no other chunks, the helper returns the input
    unchanged for that row hit, without raising."""
    client = _make_client({"g1": []})

    from src.retrieval import table_group_expansion as mod
    monkeypatch.setattr(
        mod, "_filter_for_group",
        lambda gid: MagicMock(_table_group_id=gid),
    )

    hits = [_hit(text="r", chunk_type="table_row", table_group_id="g1",
                 chunk_id="row-0")]
    out = expand_table_group_hits(hits, client=client, collection="C")
    assert out == hits


# ---------------------------------------------------------------------------
# R8: original ranking preserved
# ---------------------------------------------------------------------------


def test_r8_original_ranking_preserved(monkeypatch):
    """Non-expanded hits keep their relative order; expansions are inserted
    adjacent (immediately before the source hit for row->summary; immediately
    after for summary->rows)."""
    summary = _wv_obj(
        {"text": "S", "chunk_type": "table_summary", "table_group_id": "g1",
         "table_row_index": -1, "chunk_id": "sum-1"},
        uuid="sum-1",
    )
    client = _make_client({"g1": [summary]})

    from src.retrieval import table_group_expansion as mod
    monkeypatch.setattr(
        mod, "_filter_for_group",
        lambda gid: MagicMock(_table_group_id=gid),
    )

    hits = [
        _hit(text="A", chunk_type="text", chunk_id="a"),
        _hit(text="row", chunk_type="table_row", table_group_id="g1",
             chunk_id="row-0"),
        _hit(text="B", chunk_type="text", chunk_id="b"),
    ]
    out = expand_table_group_hits(hits, client=client, collection="C")
    ids = [h["metadata"]["chunk_id"] for h in out]
    # Summary inserted directly before row-0; A and B keep their order
    # around the row.
    assert ids == ["a", "sum-1", "row-0", "b"]
