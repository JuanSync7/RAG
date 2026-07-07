"""Tests for table + page-provenance schema fields in the Weaviate store.

Covers:
- ``ensure_collection`` declares the table/page provenance properties with the
  expected ``data_type`` and ``index_*`` flags.
- ``add_documents`` threads the new fields into ``batch.add_object(properties=)``
  when present in the collection schema, with JSON encoding for ``page_bbox``.
- Round-trips both adaptive-chunker (``table_summary``/``table_row``) and legacy
  chunking-node (``table``/``text``) ``chunk_type`` values.

Uses MagicMock against the ``weaviate.WeaviateClient`` surface; no live Weaviate.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from weaviate.classes.config import DataType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_collection_with_props(prop_names: list[str]) -> MagicMock:
    col = MagicMock()
    config = MagicMock()
    props = []
    for n in prop_names:
        p = MagicMock()
        p.name = n
        props.append(p)
    config.properties = props
    col.config.get.return_value = config
    return col


def _make_client(col: MagicMock) -> MagicMock:
    client = MagicMock()
    client.collections.get.return_value = col
    return client


def _capture_add_documents(prop_names: list[str]):
    col = _make_collection_with_props(prop_names)
    captured: list[dict] = []

    class _Batch:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, *exc):
            return False

        def add_object(self_inner, properties, vector, uuid):  # noqa: A002
            captured.append({"properties": properties, "uuid": uuid})

    col.batch.dynamic.return_value = _Batch()
    return col, captured


# ---------------------------------------------------------------------------
# ensure_collection schema declarations
# ---------------------------------------------------------------------------

# Expected (name, data_type, filterable, searchable). ``None`` means "do not
# assert that flag" (default Weaviate behavior).
EXPECTED_TABLE_FIELDS: list[tuple[str, DataType, bool | None, bool | None]] = [
    ("chunk_type", DataType.TEXT, True, False),
    ("table_id", DataType.TEXT, True, False),
    ("table_group_id", DataType.TEXT, True, False),
    ("table_row_index", DataType.INT, True, None),
    ("table_num_rows", DataType.INT, None, None),
    ("table_num_cols", DataType.INT, None, None),
    ("table_has_header", DataType.BOOL, None, None),
    ("table_caption", DataType.TEXT, None, True),
    ("table_markdown", DataType.TEXT, None, True),
    ("page_no", DataType.INT, True, None),
    ("page_label", DataType.TEXT, True, None),
    ("page_bbox", DataType.TEXT, None, None),
]


def _props_from_ensure_collection_call() -> list:
    """Invoke ``ensure_collection`` against a fresh mock and return ``properties``."""
    from src.vector_db.weaviate import store

    client = MagicMock()
    client.collections.exists.return_value = False
    store.ensure_collection(client, collection="C")
    args, kwargs = client.collections.create.call_args
    return kwargs["properties"]


class TestEnsureCollectionDeclaresTableFields:
    def test_all_expected_fields_present(self):
        props = _props_from_ensure_collection_call()
        names = {p.name for p in props}
        for field_name, _, _, _ in EXPECTED_TABLE_FIELDS:
            assert field_name in names, f"missing schema property: {field_name}"

    def test_field_data_types_match(self):
        props = _props_from_ensure_collection_call()
        by_name = {p.name: p for p in props}
        for field_name, dtype, _, _ in EXPECTED_TABLE_FIELDS:
            assert by_name[field_name].data_type == dtype, (
                f"{field_name}: expected {dtype}, got {by_name[field_name].data_type}"
            )

    def test_field_index_flags_match(self):
        props = _props_from_ensure_collection_call()
        by_name = {p.name: p for p in props}
        for field_name, _, filterable, searchable in EXPECTED_TABLE_FIELDS:
            p = by_name[field_name]
            if filterable is not None:
                assert getattr(p, "index_filterable", None) == filterable, (
                    f"{field_name}: index_filterable expected {filterable}"
                )
            if searchable is not None:
                assert getattr(p, "index_searchable", None) == searchable, (
                    f"{field_name}: index_searchable expected {searchable}"
                )


# ---------------------------------------------------------------------------
# add_documents write-through
# ---------------------------------------------------------------------------

ALL_TABLE_PROPS = [
    "text",
    "source",
    "source_key",
    "chunk_type",
    "table_id",
    "table_group_id",
    "table_row_index",
    "table_num_rows",
    "table_num_cols",
    "table_has_header",
    "table_caption",
    "table_markdown",
    "page_no",
    "page_label",
    "page_bbox",
]


class TestAddDocumentsPersistsTableFields:
    def test_summary_chunk_fields_round_trip(self):
        from src.vector_db.weaviate.store import add_documents

        col, captured = _capture_add_documents(ALL_TABLE_PROPS)
        client = _make_client(col)

        add_documents(
            client,
            texts=["table summary"],
            embeddings=[[0.1]],
            metadatas=[{
                "source": "doc.pdf",
                "source_key": "k",
                "chunk_index": 0,
                "chunk_type": "table_summary",
                "table_id": "tbl-1",
                "table_group_id": "#/tables/0",
                "table_num_rows": 5,
                "table_num_cols": 3,
                "table_has_header": True,
                "table_caption": "Quarterly sales",
                "table_markdown": "| a | b |\n|---|---|\n| 1 | 2 |",
                "page_no": 7,
                "page_label": "vii",
                "page_bbox": [10.0, 20.0, 100.0, 200.0],
            }],
        )

        props = captured[0]["properties"]
        assert props["chunk_type"] == "table_summary"
        assert props["table_id"] == "tbl-1"
        assert props["table_group_id"] == "#/tables/0"
        assert props["table_num_rows"] == 5
        assert props["table_num_cols"] == 3
        assert props["table_has_header"] is True
        assert props["table_caption"] == "Quarterly sales"
        assert props["table_markdown"].startswith("| a | b |")
        assert props["page_no"] == 7
        assert props["page_label"] == "vii"
        # page_bbox should be JSON-encoded
        assert json.loads(props["page_bbox"]) == [10.0, 20.0, 100.0, 200.0]

    def test_row_chunk_fields_round_trip(self):
        from src.vector_db.weaviate.store import add_documents

        col, captured = _capture_add_documents(ALL_TABLE_PROPS)
        client = _make_client(col)

        add_documents(
            client,
            texts=["row text"],
            embeddings=[[0.0]],
            metadatas=[{
                "source": "doc.pdf",
                "source_key": "k",
                "chunk_index": 1,
                "chunk_type": "table_row",
                "table_id": "tbl-1",
                "table_group_id": "#/tables/0",
                "table_row_index": 4,
                "table_num_rows": 5,
                "table_num_cols": 3,
            }],
        )

        props = captured[0]["properties"]
        assert props["chunk_type"] == "table_row"
        assert props["table_group_id"] == "#/tables/0"
        assert props["table_row_index"] == 4

    def test_legacy_chunk_type_values_round_trip(self):
        """Legacy chunking node stamps chunk_type='table'/'text' (no schema constraint)."""
        from src.vector_db.weaviate.store import add_documents

        col, captured = _capture_add_documents(ALL_TABLE_PROPS)
        client = _make_client(col)

        add_documents(
            client,
            texts=["legacy table", "legacy text"],
            embeddings=[[0.0], [0.0]],
            metadatas=[
                {"source": "d", "source_key": "k", "chunk_index": 0,
                 "chunk_type": "table", "table_id": "lt-1"},
                {"source": "d", "source_key": "k", "chunk_index": 1,
                 "chunk_type": "text"},
            ],
        )

        assert captured[0]["properties"]["chunk_type"] == "table"
        assert captured[0]["properties"]["table_id"] == "lt-1"
        assert captured[1]["properties"]["chunk_type"] == "text"

    def test_page_bbox_accepts_tuple(self):
        from src.vector_db.weaviate.store import add_documents

        col, captured = _capture_add_documents(ALL_TABLE_PROPS)
        client = _make_client(col)

        add_documents(
            client,
            texts=["x"],
            embeddings=[[0.0]],
            metadatas=[{
                "source": "d", "source_key": "k", "chunk_index": 0,
                "page_bbox": (1.0, 2.0, 3.0, 4.0),
            }],
        )

        assert json.loads(captured[0]["properties"]["page_bbox"]) == [1.0, 2.0, 3.0, 4.0]

    def test_page_bbox_none_serialises_to_empty_string(self):
        from src.vector_db.weaviate.store import add_documents

        col, captured = _capture_add_documents(ALL_TABLE_PROPS)
        client = _make_client(col)

        add_documents(
            client,
            texts=["x"],
            embeddings=[[0.0]],
            metadatas=[{"source": "d", "source_key": "k", "chunk_index": 0}],
        )

        # Default for page_bbox should be empty string (consistent with other
        # optional TEXT fields), never the literal "null" JSON.
        assert captured[0]["properties"]["page_bbox"] == ""

    def test_fields_omitted_when_not_in_schema(self):
        """Legacy collections without the new properties should silently drop them."""
        from src.vector_db.weaviate.store import add_documents

        col, captured = _capture_add_documents(["text", "source", "source_key"])
        client = _make_client(col)

        add_documents(
            client,
            texts=["hi"],
            embeddings=[[0.0]],
            metadatas=[{
                "source": "d", "source_key": "k", "chunk_index": 0,
                "chunk_type": "table_summary",
                "table_id": "should-not-write",
                "table_group_id": "g",
                "table_markdown": "| a |",
                "page_bbox": [1, 2, 3, 4],
            }],
        )

        props = captured[0]["properties"]
        for absent in (
            "chunk_type", "table_id", "table_group_id", "table_markdown",
            "page_bbox", "page_no", "page_label",
        ):
            assert absent not in props
