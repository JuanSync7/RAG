# @summary
# Tests for the Weaviate document-card collection store (RAPTOR-lite routing A2).
# Covers: src/vector_db/weaviate/card_store.py (ensure_card_collection,
#         add_document_cards) and the backend-agnostic facade re-exports in
#         src/vector_db/__init__.py.
# Exports: (pytest test functions)
# Deps: pytest, unittest.mock, weaviate.classes.config (DataType)
# @end-summary
"""Tests for the Weaviate document-card collection store.

All Weaviate client interactions are mocked via MagicMock so these tests run
without a live Weaviate instance. The store mirrors ``visual_store`` — a
SECOND collection decoupled from the heavy chunk ``DocumentRecord`` path, with
explicit precomputed vectors supplied by the caller.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from weaviate.classes.config import DataType


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

# (name, data_type) for the 7 card-record properties (vector excluded).
EXPECTED_CARD_FIELDS: list[tuple[str, DataType]] = [
    ("document_id", DataType.TEXT),
    ("title", DataType.TEXT),
    ("source", DataType.TEXT),
    ("section_headings", DataType.TEXT_ARRAY),
    ("card_text", DataType.TEXT),
    ("summary", DataType.TEXT),
    ("num_chunks", DataType.INT),
]

_CARD_VECTOR = [float(i) / 64.0 for i in range(64)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_card(
    document_id: str = "doc-001",
    title: str = "AXI4 Protocol Spec",
    source: str = "specs/axi4.pdf",
    section_headings=None,
    card_text: str = "AXI4 Protocol Spec\nOverview\nChannels",
    summary: str = "",
    num_chunks: int = 12,
    vector=None,
) -> dict:
    """Build a complete card dict with the 7 contract keys plus a vector."""
    return {
        "document_id": document_id,
        "title": title,
        "source": source,
        "section_headings": section_headings or ["Overview", "Channels"],
        "card_text": card_text,
        "summary": summary,
        "num_chunks": num_chunks,
        "vector": vector or list(_CARD_VECTOR),
    }


def _make_client_and_col(failed_objects=None):
    """Return (mock_client, mock_col, mock_batch) for batch-insert tests."""
    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_batch = MagicMock()
    mock_batch.failed_objects = failed_objects if failed_objects is not None else []
    mock_col.batch.failed_objects = failed_objects if failed_objects is not None else []
    mock_col.batch.dynamic.return_value.__enter__.return_value = mock_batch
    mock_col.batch.dynamic.return_value.__exit__.return_value = False
    mock_client.collections.get.return_value = mock_col
    return mock_client, mock_col, mock_batch


# ---------------------------------------------------------------------------
# TestEnsureCardCollection
# ---------------------------------------------------------------------------

class TestEnsureCardCollection:
    """Idempotent collection creation with the exact 7-property schema."""

    def test_idempotent_when_collection_exists(self):
        """create is NOT called when the collection already exists."""
        from src.vector_db.weaviate.card_store import ensure_card_collection

        mock_client = MagicMock()
        mock_client.collections.exists.return_value = True

        result = ensure_card_collection(mock_client, collection="RAGDocumentCards")

        mock_client.collections.create.assert_not_called()
        assert result is None

    def test_exists_checked_with_collection_name(self):
        """exists() is queried with the provided collection name."""
        from src.vector_db.weaviate.card_store import ensure_card_collection

        mock_client = MagicMock()
        mock_client.collections.exists.return_value = True

        ensure_card_collection(mock_client, collection="RAGDocumentCards")

        mock_client.collections.exists.assert_called_once_with("RAGDocumentCards")

    def test_creates_collection_when_absent(self):
        """create is called exactly once when the collection is absent."""
        from src.vector_db.weaviate.card_store import ensure_card_collection

        mock_client = MagicMock()
        mock_client.collections.exists.return_value = False

        ensure_card_collection(mock_client, collection="RAGDocumentCards")

        mock_client.collections.create.assert_called_once()

    def test_create_called_with_collection_name(self):
        """The collection name passed to create matches the argument."""
        from src.vector_db.weaviate.card_store import ensure_card_collection

        mock_client = MagicMock()
        mock_client.collections.exists.return_value = False

        ensure_card_collection(mock_client, collection="RAGDocumentCards")

        create_args = mock_client.collections.create.call_args
        name_arg = (
            create_args.args[0]
            if create_args.args
            else create_args.kwargs.get("name")
        )
        assert name_arg == "RAGDocumentCards"

    def test_create_declares_all_seven_properties(self):
        """All 7 contract properties are declared with the right names."""
        from src.vector_db.weaviate.card_store import ensure_card_collection

        mock_client = MagicMock()
        mock_client.collections.exists.return_value = False

        ensure_card_collection(mock_client, collection="RAGDocumentCards")

        props = mock_client.collections.create.call_args.kwargs["properties"]
        names = {p.name for p in props}
        for field_name, _ in EXPECTED_CARD_FIELDS:
            assert field_name in names, f"missing card property: {field_name}"
        # No extra (e.g. the vector must NOT be a property).
        assert names == {f for f, _ in EXPECTED_CARD_FIELDS}
        assert "vector" not in names

    def test_create_property_data_types_match(self):
        """Each property is declared with the contract DataType (incl. TEXT_ARRAY/INT)."""
        from src.vector_db.weaviate.card_store import ensure_card_collection

        mock_client = MagicMock()
        mock_client.collections.exists.return_value = False

        ensure_card_collection(mock_client, collection="RAGDocumentCards")

        props = mock_client.collections.create.call_args.kwargs["properties"]
        by_name = {p.name: p for p in props}
        for field_name, dtype in EXPECTED_CARD_FIELDS:
            assert by_name[field_name].data_type == dtype, (
                f"{field_name}: expected {dtype}, got {by_name[field_name].data_type}"
            )

    def test_create_uses_vectorizer_none(self):
        """Vectorizer is none() — embeddings are precomputed by the caller."""
        from src.vector_db.weaviate.card_store import ensure_card_collection

        mock_client = MagicMock()
        mock_client.collections.exists.return_value = False

        # Patch Configure so we can assert Vectorizer.none() was invoked.
        with patch("src.vector_db.weaviate.card_store.Configure") as mock_cfg:
            ensure_card_collection(mock_client, collection="RAGDocumentCards")

        mock_cfg.Vectorizer.none.assert_called_once()

    def test_default_collection_name(self):
        """Default collection is RAGDocumentCards."""
        from src.vector_db.weaviate.card_store import ensure_card_collection

        mock_client = MagicMock()
        mock_client.collections.exists.return_value = True

        ensure_card_collection(mock_client)

        mock_client.collections.exists.assert_called_once_with("RAGDocumentCards")


# ---------------------------------------------------------------------------
# TestAddDocumentCards
# ---------------------------------------------------------------------------

class TestAddDocumentCards:
    """Insert with explicit precomputed vectors; deterministic upsert by doc id."""

    def test_inserts_two_cards_returns_count(self):
        """Two cards → two objects inserted; returns 2."""
        from src.vector_db.weaviate.card_store import add_document_cards

        cards = [_make_card(document_id="doc-a"), _make_card(document_id="doc-b")]
        mock_client, mock_col, mock_batch = _make_client_and_col(failed_objects=[])

        result = add_document_cards(mock_client, cards, collection="RAGDocumentCards")

        assert result == 2
        assert mock_batch.add_object.call_count == 2

    def test_correct_collection_retrieved(self):
        """collections.get is called with the provided collection name."""
        from src.vector_db.weaviate.card_store import add_document_cards

        cards = [_make_card()]
        mock_client, mock_col, mock_batch = _make_client_and_col(failed_objects=[])

        add_document_cards(mock_client, cards, collection="RAGDocumentCards")

        mock_client.collections.get.assert_called_once_with("RAGDocumentCards")

    def test_vector_passed_explicitly(self):
        """Each object is inserted with its precomputed vector via the vector kwarg."""
        from src.vector_db.weaviate.card_store import add_document_cards

        vec = [float(i) / 64.0 for i in range(64)]
        card = _make_card(vector=vec)
        mock_client, mock_col, mock_batch = _make_client_and_col(failed_objects=[])

        add_document_cards(mock_client, [card], collection="RAGDocumentCards")

        first_call = mock_batch.add_object.call_args_list[0]
        assert first_call.kwargs.get("vector") == vec

    def test_vector_excluded_from_properties(self):
        """The 'vector' key must NOT appear in the written properties."""
        from src.vector_db.weaviate.card_store import add_document_cards

        card = _make_card()
        mock_client, mock_col, mock_batch = _make_client_and_col(failed_objects=[])

        add_document_cards(mock_client, [card], collection="RAGDocumentCards")

        first_call = mock_batch.add_object.call_args_list[0]
        props = first_call.kwargs.get("properties") or first_call.args[0]
        assert "vector" not in props

    def test_all_contract_properties_written(self):
        """All 7 contract keys are present in the written properties."""
        from src.vector_db.weaviate.card_store import add_document_cards

        card = _make_card()
        mock_client, mock_col, mock_batch = _make_client_and_col(failed_objects=[])

        add_document_cards(mock_client, [card], collection="RAGDocumentCards")

        first_call = mock_batch.add_object.call_args_list[0]
        props = first_call.kwargs.get("properties") or first_call.args[0]
        for field_name, _ in EXPECTED_CARD_FIELDS:
            assert field_name in props, f"expected contract key {field_name!r} in properties"
        assert props["document_id"] == "doc-001"
        assert props["section_headings"] == ["Overview", "Channels"]
        assert props["num_chunks"] == 12

    def test_empty_input_returns_zero_no_get(self):
        """Empty list → returns 0 without touching collections.get."""
        from src.vector_db.weaviate.card_store import add_document_cards

        mock_client = MagicMock()

        result = add_document_cards(mock_client, [], collection="RAGDocumentCards")

        assert result == 0
        mock_client.collections.get.assert_not_called()

    def test_partial_failures_subtracted(self):
        """3 cards, 1 failure → returns 2."""
        from src.vector_db.weaviate.card_store import add_document_cards

        cards = [_make_card(document_id=f"doc-{i}") for i in range(3)]
        failed = [MagicMock()]
        mock_client, mock_col, mock_batch = _make_client_and_col(failed_objects=failed)

        result = add_document_cards(mock_client, cards, collection="RAGDocumentCards")

        assert result == 2

    def test_deterministic_uuid_stable_across_calls(self):
        """Same document_id → same object UUID across separate calls (replace, not duplicate)."""
        from src.vector_db.weaviate.card_store import add_document_cards

        card = _make_card(document_id="doc-stable")

        # First ingest.
        c1, _, batch1 = _make_client_and_col(failed_objects=[])
        add_document_cards(c1, [card], collection="RAGDocumentCards")
        uuid1 = batch1.add_object.call_args_list[0].kwargs.get("uuid")

        # Re-ingest the same document.
        c2, _, batch2 = _make_client_and_col(failed_objects=[])
        add_document_cards(c2, [card], collection="RAGDocumentCards")
        uuid2 = batch2.add_object.call_args_list[0].kwargs.get("uuid")

        assert uuid1 is not None
        assert uuid1 == uuid2

    def test_uuid_derived_from_document_id(self):
        """The object UUID matches uuid5(NAMESPACE_DNS, document_id).

        This is byte-for-byte identical to ``weaviate.util.generate_uuid5`` (the
        store's helper replicates that exact formula against the stdlib so it
        does not depend on ``weaviate.util`` being importable).
        """
        import uuid as _uuid
        from src.vector_db.weaviate.card_store import add_document_cards

        card = _make_card(document_id="doc-xyz")
        mock_client, mock_col, mock_batch = _make_client_and_col(failed_objects=[])

        add_document_cards(mock_client, [card], collection="RAGDocumentCards")

        used_uuid = mock_batch.add_object.call_args_list[0].kwargs.get("uuid")
        expected = str(_uuid.uuid5(_uuid.NAMESPACE_DNS, "doc-xyz"))
        assert str(used_uuid) == expected

    def test_distinct_documents_get_distinct_uuids(self):
        """Different document_ids → different object UUIDs (no accidental collision)."""
        from src.vector_db.weaviate.card_store import add_document_cards

        cards = [_make_card(document_id="doc-a"), _make_card(document_id="doc-b")]
        mock_client, mock_col, mock_batch = _make_client_and_col(failed_objects=[])

        add_document_cards(mock_client, cards, collection="RAGDocumentCards")

        uuids = [c.kwargs.get("uuid") for c in mock_batch.add_object.call_args_list]
        assert uuids[0] != uuids[1]


# ---------------------------------------------------------------------------
# TestFacadeReExports
# ---------------------------------------------------------------------------

class TestFacadeReExports:
    """The backend-agnostic facade must re-export the card store helpers."""

    def test_facade_exports_ensure_and_add(self):
        import src.vector_db as vdb

        assert hasattr(vdb, "ensure_card_collection")
        assert hasattr(vdb, "add_document_cards")

    def test_weaviate_package_reexports(self):
        from src.vector_db.weaviate import (  # noqa: F401
            ensure_card_collection,
            add_document_cards,
        )
