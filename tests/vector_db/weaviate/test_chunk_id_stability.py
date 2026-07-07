"""Stability tests for table-aware chunk_id derivation.

Validates that table chunk UUIDs derive from ``(source, table_group_id,
chunk_type, table_row_index)`` so re-ingests update vectors in place rather
than orphaning them. Non-table chunks must retain the legacy formula.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.vector_db.weaviate import store as store_mod
from src.vector_db.weaviate.store import (
    add_documents,
    build_chunk_id,
    build_table_chunk_id,
)


class _BatchCtx:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def add_object(self, *, properties, vector, uuid):  # noqa: A002 - mirror weaviate api
        self.calls.append({"properties": properties, "vector": vector, "uuid": uuid})


class _Collection:
    def __init__(self) -> None:
        self.batch_ctx = _BatchCtx()

        class _Batch:
            def __init__(inner, ctx):
                inner._ctx = ctx

            def dynamic(inner):
                return inner._ctx

        self.batch = _Batch(self.batch_ctx)

    def config(self):  # pragma: no cover - unused
        return None

    @property
    def config(self):  # type: ignore[no-redef]
        class _Cfg:
            def get(self_inner):
                obj = MagicMock()
                obj.properties = []
                return obj

        return _Cfg()


class _Client:
    def __init__(self) -> None:
        self._col = _Collection()

    @property
    def collections(self):
        client_self = self

        class _Collections:
            def get(self_inner, name):
                return client_self._col

        return _Collections()


def _call(client, text, metadata):
    return add_documents(client, [text], [[0.0, 0.1, 0.2]], [metadata])


def _ids_for(client):
    return [c["uuid"] for c in client._col.batch_ctx.calls]


def test_table_row_chunk_id_stable_across_reingest_with_text_change():
    """T1: same group_id + row_index but different chunk_index/text → same id."""
    c1 = _Client()
    _call(
        c1,
        "row A original",
        {
            "source": "doc.pdf",
            "source_key": "doc.pdf",
            "chunk_index": 5,
            "chunk_type": "table_row",
            "table_group_id": "doc.pdf::tbl0",
            "table_row_index": 0,
        },
    )
    c2 = _Client()
    _call(
        c2,
        "row A EDITED content",
        {
            "source": "doc.pdf",
            "source_key": "doc.pdf",
            "chunk_index": 12,
            "chunk_type": "table_row",
            "table_group_id": "doc.pdf::tbl0",
            "table_row_index": 0,
        },
    )
    assert _ids_for(c1)[0] == _ids_for(c2)[0]


def test_different_table_groups_same_index_do_not_collide():
    """T2: same source+chunk_index but different table_group_id → different ids."""
    a = build_table_chunk_id("doc.pdf", "doc.pdf::tbl0", "table_row", 0)
    b = build_table_chunk_id("doc.pdf", "doc.pdf::tbl1", "table_row", 0)
    assert a != b


def test_summary_and_row_zero_do_not_collide():
    """T3: table_summary vs table_row at row 0 within same group → different."""
    summary = build_table_chunk_id("doc.pdf", "doc.pdf::tbl0", "table_summary", None)
    row0 = build_table_chunk_id("doc.pdf", "doc.pdf::tbl0", "table_row", 0)
    assert summary != row0


def test_non_table_chunk_uses_legacy_formula():
    """T4: chunk_type missing/'text' → identical text+chunk_index → identical id."""
    c1 = _Client()
    _call(
        c1,
        "paragraph body",
        {"source": "doc.pdf", "source_key": "doc.pdf", "chunk_index": 3},
    )
    c2 = _Client()
    _call(
        c2,
        "paragraph body",
        {
            "source": "doc.pdf",
            "source_key": "doc.pdf",
            "chunk_index": 3,
            "chunk_type": "text",
        },
    )
    legacy = build_chunk_id("doc.pdf", 3, "paragraph body")
    assert _ids_for(c1)[0] == legacy
    assert _ids_for(c2)[0] == legacy


def test_explicit_chunk_id_metadata_still_wins():
    """T5: explicit chunk_id in metadata bypasses derivation (table or not)."""
    explicit = "11111111-2222-3333-4444-555555555555"
    c = _Client()
    _call(
        c,
        "row body",
        {
            "source": "doc.pdf",
            "source_key": "doc.pdf",
            "chunk_index": 0,
            "chunk_type": "table_row",
            "table_group_id": "doc.pdf::tbl0",
            "table_row_index": 0,
            "chunk_id": explicit,
        },
    )
    assert _ids_for(c)[0] == explicit
