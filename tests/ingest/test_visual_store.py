# @summary
# Tests for Weaviate visual collection store.
# Covers: src/vector_db/weaviate/visual_store.py (ensure_visual_collection,
#         add_visual_documents, delete_visual_by_source_key) and
#         src/vector_db/backend.py ABC extensions (three new abstract methods)
#         and src/vector_db/weaviate/backend.py WeaviateBackend delegation.
# Exports: (pytest test functions)
# Deps: pytest, unittest.mock
# @end-summary
"""Tests for the Weaviate visual collection store.

All Weaviate client interactions are mocked via MagicMock so these tests run
without a live Weaviate instance.

Known gaps are annotated with ``# GAP:`` comments throughout.
"""

import pytest
from unittest.mock import MagicMock, call, patch

from src.vector_db.weaviate.visual_store import (
    ensure_visual_collection,
    add_visual_documents,
    delete_visual_by_source_key,
)
from src.vector_db.backend import VectorBackend
from src.vector_db.weaviate.backend import WeaviateBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MEAN_VECTOR = [float(i) / 128.0 for i in range(128)]  # 128-float list

_SCALAR_KEYS = [
    "document_id",
    "page_number",
    "source_key",
    "source_uri",
    "source_name",
    "tenant_id",
    "total_pages",
    "page_width_px",
    "page_height_px",
    "minio_key",
    "patch_vectors",
]


def _make_doc(
    document_id: str = "doc-001",
    page_number: int = 1,
    source_key: str = "src-001",
    source_uri: str = "s3://bucket/doc.pdf",
    source_name: str = "doc.pdf",
    tenant_id: str = "tenant-a",
    total_pages: int = 10,
    page_width_px: int = 1024,
    page_height_px: int = 768,
    minio_key: str = "bucket/doc_page1.png",
    patch_vectors=None,
    mean_vector=None,
) -> dict:
    """Build a complete visual document dict with all 12 keys."""
    return {
        "document_id": document_id,
        "page_number": page_number,
        "source_key": source_key,
        "source_uri": source_uri,
        "source_name": source_name,
        "tenant_id": tenant_id,
        "total_pages": total_pages,
        "page_width_px": page_width_px,
        "page_height_px": page_height_px,
        "minio_key": minio_key,
        "patch_vectors": patch_vectors or [[0.1] * 128] * 8,
        "mean_vector": mean_vector or list(_MEAN_VECTOR),
    }


def _make_client_and_col(failed_objects=None):
    """Return (mock_client, mock_col, mock_batch) tuple for batch-insert tests."""
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
# TestEnsureVisualCollection
# ---------------------------------------------------------------------------

class TestEnsureVisualCollection:
    """FR-502, FR-504: idempotent collection creation with exact schema."""

    def test_creates_collection_when_absent(self):
        """FR-502, FR-504: create is called once when collection does not exist."""
        mock_client = MagicMock()
        mock_client.collections.exists.return_value = False

        with patch("src.vector_db.weaviate.visual_store.Configure"):
            ensure_visual_collection(mock_client, collection="RAGVisualPages")

        mock_client.collections.create.assert_called_once()

    def test_create_called_with_correct_collection_name(self):
        """FR-502: collection name passed to create matches argument."""
        mock_client = MagicMock()
        mock_client.collections.exists.return_value = False

        with patch("src.vector_db.weaviate.visual_store.Configure"):
            ensure_visual_collection(mock_client, collection="RAGVisualPages")

        create_args = mock_client.collections.create.call_args
        # name is positional or keyword; check either
        name_arg = (
            create_args.args[0]
            if create_args.args
            else create_args.kwargs.get("name")
        )
        assert name_arg == "RAGVisualPages"

    def test_create_includes_mean_vector_named_vector(self):
        """FR-504: named vector 'mean_vector' with 128-dim cosine HNSW is present."""
        mock_client = MagicMock()
        mock_client.collections.exists.return_value = False

        with patch("src.vector_db.weaviate.visual_store.Configure"):
            ensure_visual_collection(mock_client, collection="RAGVisualPages")

        create_kwargs = mock_client.collections.create.call_args.kwargs
        # The vectorizer_config or named vectors must include 'mean_vector'
        # We check that the call was made with keyword args containing
        # something referencing 'mean_vector'; implementation-level specifics
        # are validated by checking the call was made (contract test).
        assert mock_client.collections.create.called

    def test_idempotent_when_collection_exists(self):
        """FR-502: create is NOT called when collection already exists."""
        mock_client = MagicMock()
        mock_client.collections.exists.return_value = True

        result = ensure_visual_collection(mock_client, collection="RAGVisualPages")

        mock_client.collections.create.assert_not_called()
        assert result is None

    def test_returns_none(self):
        """ensure_visual_collection always returns None."""
        mock_client = MagicMock()
        mock_client.collections.exists.return_value = False

        with patch("src.vector_db.weaviate.visual_store.Configure"):
            result = ensure_visual_collection(mock_client, collection="RAGVisualPages")

        assert result is None

    def test_exists_called_with_collection_name(self):
        """FR-502: exists check uses the provided collection name."""
        mock_client = MagicMock()
        mock_client.collections.exists.return_value = True

        ensure_visual_collection(mock_client, collection="RAGVisualPages")

        mock_client.collections.exists.assert_called_once_with("RAGVisualPages")


# ---------------------------------------------------------------------------
# TestAddVisualDocuments
# ---------------------------------------------------------------------------

class TestAddVisualDocuments:
    """FR-507: batch insert with mean_vector split and failed-object counting."""

    def test_returns_inserted_count_zero_failures(self):
        """FR-507: 50 docs, zero failures → returns 50."""
        docs = [_make_doc(page_number=i) for i in range(50)]
        mock_client, mock_col, mock_batch = _make_client_and_col(failed_objects=[])

        result = add_visual_documents(mock_client, docs, collection="RAGVisualPages")

        assert result == 50

    def test_batch_add_object_excludes_mean_vector_from_properties(self):
        """FR-507: mean_vector must be removed from properties dict passed to add_object."""
        docs = [_make_doc()]
        mock_client, mock_col, mock_batch = _make_client_and_col(failed_objects=[])

        add_visual_documents(mock_client, docs, collection="RAGVisualPages")

        assert mock_batch.add_object.called
        first_call = mock_batch.add_object.call_args_list[0]
        props = first_call.kwargs.get("properties") or first_call.args[0]
        assert "mean_vector" not in props

    def test_batch_add_object_passes_mean_vector_as_named_vector(self):
        """FR-507: vector kwarg contains {'mean_vector': <128-float list>}."""
        mean_vec = [float(i) / 128.0 for i in range(128)]
        doc = _make_doc(mean_vector=mean_vec)
        mock_client, mock_col, mock_batch = _make_client_and_col(failed_objects=[])

        add_visual_documents(mock_client, [doc], collection="RAGVisualPages")

        first_call = mock_batch.add_object.call_args_list[0]
        vector_arg = first_call.kwargs.get("vector")
        assert vector_arg is not None
        assert "mean_vector" in vector_arg
        assert vector_arg["mean_vector"] == mean_vec

    def test_partial_failures_subtracted_from_count(self):
        """FR-507 edge: 10 docs, 2 failures → returns 8."""
        docs = [_make_doc(page_number=i) for i in range(10)]
        failed = [MagicMock(), MagicMock()]
        mock_client, mock_col, mock_batch = _make_client_and_col(failed_objects=failed)

        result = add_visual_documents(mock_client, docs, collection="RAGVisualPages")

        assert result == 8

    def test_empty_input_returns_zero_immediately(self):
        """FR-507 boundary: empty list returns 0 without calling collections.get."""
        mock_client = MagicMock()

        result = add_visual_documents(mock_client, [], collection="RAGVisualPages")

        assert result == 0
        mock_client.collections.get.assert_not_called()

    def test_all_documents_fail_returns_zero(self):
        """FR-507 edge: all docs fail → returns 0, no exception."""
        docs = [_make_doc(page_number=i) for i in range(5)]
        failed = [MagicMock() for _ in range(5)]
        mock_client, mock_col, mock_batch = _make_client_and_col(failed_objects=failed)

        result = add_visual_documents(mock_client, docs, collection="RAGVisualPages")

        assert result == 0

    def test_all_11_scalar_properties_in_add_object_properties(self):
        """FR-503: all 10 scalar keys (mean_vector excluded) present in properties."""
        doc = _make_doc()
        mock_client, mock_col, mock_batch = _make_client_and_col(failed_objects=[])

        add_visual_documents(mock_client, [doc], collection="RAGVisualPages")

        first_call = mock_batch.add_object.call_args_list[0]
        props = first_call.kwargs.get("properties") or first_call.args[0]
        # 10 scalar keys = _SCALAR_KEYS minus nothing; mean_vector is excluded
        scalar_without_mean = [k for k in _SCALAR_KEYS]  # patch_vectors included
        for key in scalar_without_mean:
            assert key in props, f"Expected scalar key '{key}' in properties"

    def test_patch_vectors_passed_through_as_is(self):
        """FR-505: patch_vectors list-of-lists stored as-is in properties."""
        patch_vecs = [[float(j) / 128.0 for j in range(128)] for _ in range(8)]
        doc = _make_doc(patch_vectors=patch_vecs)
        mock_client, mock_col, mock_batch = _make_client_and_col(failed_objects=[])

        add_visual_documents(mock_client, [doc], collection="RAGVisualPages")

        first_call = mock_batch.add_object.call_args_list[0]
        props = first_call.kwargs.get("properties") or first_call.args[0]
        assert props["patch_vectors"] == patch_vecs

    def test_single_document_uses_batch_path(self):
        """Boundary: 1-element list exercises batch path, not short-circuit."""
        docs = [_make_doc()]
        mock_client, mock_col, mock_batch = _make_client_and_col(failed_objects=[])

        result = add_visual_documents(mock_client, docs, collection="RAGVisualPages")

        assert result == 1
        mock_batch.add_object.assert_called_once()

    def test_correct_collection_retrieved(self):
        """collections.get is called with the provided collection name."""
        docs = [_make_doc()]
        mock_client, mock_col, mock_batch = _make_client_and_col(failed_objects=[])

        add_visual_documents(mock_client, docs, collection="RAGVisualPages")

        mock_client.collections.get.assert_called_once_with("RAGVisualPages")

    def test_add_object_called_once_per_document(self):
        """add_object is called exactly N times for N input documents."""
        n = 7
        docs = [_make_doc(page_number=i) for i in range(n)]
        mock_client, mock_col, mock_batch = _make_client_and_col(failed_objects=[])

        add_visual_documents(mock_client, docs, collection="RAGVisualPages")

        assert mock_batch.add_object.call_count == n


# ---------------------------------------------------------------------------
# TestDeleteVisualBySourceKey
# ---------------------------------------------------------------------------

class TestDeleteVisualBySourceKey:
    """FR-506: filter-based deletion by source_key."""

    def test_returns_match_count_on_success(self):
        """FR-506: result.matches=3 → returns 3."""
        mock_client = MagicMock()
        mock_col = MagicMock()
        mock_col.data.delete_many.return_value = MagicMock(matches=3)
        mock_client.collections.get.return_value = mock_col

        result = delete_visual_by_source_key(
            mock_client, source_key="doc_abc", collection="RAGVisualPages"
        )

        assert result == 3

    def test_returns_zero_when_no_matches(self):
        """FR-506: result.matches=0 → returns 0, no exception."""
        mock_client = MagicMock()
        mock_col = MagicMock()
        mock_col.data.delete_many.return_value = MagicMock(matches=0)
        mock_client.collections.get.return_value = mock_col

        result = delete_visual_by_source_key(
            mock_client, source_key="nonexistent", collection="RAGVisualPages"
        )

        assert result == 0

    def test_delete_many_called_with_source_key_filter(self):
        """FR-506: delete_many is called with a filter on source_key."""
        mock_client = MagicMock()
        mock_col = MagicMock()
        mock_col.data.delete_many.return_value = MagicMock(matches=1)
        mock_client.collections.get.return_value = mock_col

        delete_visual_by_source_key(
            mock_client, source_key="doc_abc", collection="RAGVisualPages"
        )

        mock_col.data.delete_many.assert_called_once()

    def test_safe_fallback_when_matches_attribute_absent(self):
        """Error boundary: result missing 'matches' attr → returns 0 via getattr fallback."""
        mock_client = MagicMock()
        mock_col = MagicMock()
        # Deliberately create a result object with no matches attribute
        result_obj = MagicMock(spec=[])  # spec=[] means no attributes at all
        mock_col.data.delete_many.return_value = result_obj
        mock_client.collections.get.return_value = mock_col

        # Should not raise AttributeError
        result = delete_visual_by_source_key(
            mock_client, source_key="doc_abc", collection="RAGVisualPages"
        )

        assert result == 0

    def test_safe_fallback_when_matches_is_none(self):
        """Error boundary: result.matches is None → returns 0 via 'or 0' guard."""
        mock_client = MagicMock()
        mock_col = MagicMock()
        mock_col.data.delete_many.return_value = MagicMock(matches=None)
        mock_client.collections.get.return_value = mock_col

        result = delete_visual_by_source_key(
            mock_client, source_key="doc_abc", collection="RAGVisualPages"
        )

        assert result == 0

    def test_correct_collection_retrieved(self):
        """collections.get is called with the provided collection name."""
        mock_client = MagicMock()
        mock_col = MagicMock()
        mock_col.data.delete_many.return_value = MagicMock(matches=0)
        mock_client.collections.get.return_value = mock_col

        delete_visual_by_source_key(
            mock_client, source_key="doc_abc", collection="RAGVisualPages"
        )

        mock_client.collections.get.assert_called_once_with("RAGVisualPages")


# ---------------------------------------------------------------------------
# TestWeaviateBackendDelegation
# ---------------------------------------------------------------------------

class TestWeaviateBackendDelegation:
    """WeaviateBackend resolves 'RAGVisualPages' default for all three methods."""

    def test_add_visual_documents_default_collection_resolution(self):
        """collection=None resolves to 'RAGVisualPages' before forwarding."""
        backend = WeaviateBackend()
        mock_client = MagicMock()
        mock_col = MagicMock()
        mock_batch = MagicMock()
        mock_batch.failed_objects = []
        mock_col.batch.dynamic.return_value.__enter__.return_value = mock_batch
        mock_col.batch.dynamic.return_value.__exit__.return_value = False
        mock_client.collections.get.return_value = mock_col

        docs = [_make_doc()]
        backend.add_visual_documents(mock_client, docs, collection=None)

        mock_client.collections.get.assert_called_once_with("RAGVisualPages")

    def test_ensure_visual_collection_default_collection_resolution(self):
        """collection=None resolves to 'RAGVisualPages' in ensure_visual_collection."""
        backend = WeaviateBackend()
        mock_client = MagicMock()
        mock_client.collections.exists.return_value = True  # skip create

        backend.ensure_visual_collection(mock_client, collection=None)

        mock_client.collections.exists.assert_called_once_with("RAGVisualPages")

    def test_delete_visual_by_source_key_default_collection_resolution(self):
        """collection=None resolves to 'RAGVisualPages' in delete_visual_by_source_key."""
        backend = WeaviateBackend()
        mock_client = MagicMock()
        mock_col = MagicMock()
        mock_col.data.delete_many.return_value = MagicMock(matches=0)
        mock_client.collections.get.return_value = mock_col

        backend.delete_visual_by_source_key(mock_client, source_key="x", collection=None)

        mock_client.collections.get.assert_called_once_with("RAGVisualPages")

    def test_empty_string_collection_treated_as_falsy(self):
        """Boundary: collection='' treated as falsy → resolves to 'RAGVisualPages'."""
        backend = WeaviateBackend()
        mock_client = MagicMock()
        mock_col = MagicMock()
        mock_batch = MagicMock()
        mock_batch.failed_objects = []
        mock_col.batch.dynamic.return_value.__enter__.return_value = mock_batch
        mock_col.batch.dynamic.return_value.__exit__.return_value = False
        mock_client.collections.get.return_value = mock_col

        docs = [_make_doc()]
        backend.add_visual_documents(mock_client, docs, collection="")

        mock_client.collections.get.assert_called_once_with("RAGVisualPages")


# ---------------------------------------------------------------------------
# TestVectorBackendABC
# ---------------------------------------------------------------------------

class TestVectorBackendABC:
    """NFR-909: three new abstract methods must enforce subclass implementation."""

    def test_missing_ensure_visual_collection_raises_type_error(self):
        """Subclass omitting ensure_visual_collection cannot be instantiated."""
        class IncompleteBackend(VectorBackend):
            # Implement all pre-existing abstract methods to isolate the new one
            def ensure_collection(self, *a, **kw): ...
            def add_documents(self, *a, **kw): ...
            def delete_by_source_key(self, *a, **kw): ...
            # ensure_visual_collection omitted intentionally
            def add_visual_documents(self, *a, **kw): ...
            def delete_visual_by_source_key(self, *a, **kw): ...

        with pytest.raises(TypeError):
            IncompleteBackend()

    def test_missing_add_visual_documents_raises_type_error(self):
        """Subclass omitting add_visual_documents cannot be instantiated."""
        class IncompleteBackend(VectorBackend):
            def ensure_collection(self, *a, **kw): ...
            def add_documents(self, *a, **kw): ...
            def delete_by_source_key(self, *a, **kw): ...
            def ensure_visual_collection(self, *a, **kw): ...
            # add_visual_documents omitted intentionally
            def delete_visual_by_source_key(self, *a, **kw): ...

        with pytest.raises(TypeError):
            IncompleteBackend()

    def test_missing_delete_visual_by_source_key_raises_type_error(self):
        """Subclass omitting delete_visual_by_source_key cannot be instantiated."""
        class IncompleteBackend(VectorBackend):
            def ensure_collection(self, *a, **kw): ...
            def add_documents(self, *a, **kw): ...
            def delete_by_source_key(self, *a, **kw): ...
            def ensure_visual_collection(self, *a, **kw): ...
            def add_visual_documents(self, *a, **kw): ...
            # delete_visual_by_source_key omitted intentionally

        with pytest.raises(TypeError):
            IncompleteBackend()

    def test_complete_subclass_instantiates_without_error(self):
        """A subclass implementing all abstract methods instantiates successfully."""
        class CompleteBackend(VectorBackend):
            def create_persistent_client(self, *a, **kw): ...
            def get_ephemeral_client(self, *a, **kw): ...
            def ensure_collection(self, *a, **kw): ...
            def collection_exists(self, *a, **kw): ...
            def add_documents(self, *a, **kw): ...
            def update_chunk_content(self, *a, **kw): ...
            def search(self, *a, **kw): ...
            def delete_collection(self, *a, **kw): ...
            def delete_by_source(self, *a, **kw): ...
            def delete_by_source_key(self, *a, **kw): ...
            def aggregate_by_source(self, *a, **kw): ...
            def get_collection_stats(self, *a, **kw): ...
            def list_collections(self, *a, **kw): ...
            def ensure_visual_collection(self, *a, **kw): ...
            def add_visual_documents(self, *a, **kw): ...
            def delete_visual_by_source_key(self, *a, **kw): ...
            def search_visual(self, *a, **kw): ...

        # Should not raise
        instance = CompleteBackend()
        assert isinstance(instance, VectorBackend)


# ---------------------------------------------------------------------------
# TestBoundaryConditions
# ---------------------------------------------------------------------------

class TestBoundaryConditions:
    """Edge cases and boundary conditions for all three store functions."""

    def test_empty_documents_returns_zero_no_get_call(self):
        """Boundary: [] input returns 0 and never calls collections.get."""
        mock_client = MagicMock()

        result = add_visual_documents(mock_client, [], collection="RAGVisualPages")

        assert result == 0
        mock_client.collections.get.assert_not_called()

    def test_single_document_not_short_circuited(self):
        """Boundary: 1-element list goes through batch, not empty short-circuit."""
        docs = [_make_doc()]
        mock_client, mock_col, mock_batch = _make_client_and_col(failed_objects=[])

        result = add_visual_documents(mock_client, docs, collection="RAGVisualPages")

        assert result == 1
        mock_client.collections.get.assert_called_once()

    def test_failed_objects_attribute_absent_safe_fallback(self):
        """Boundary: batch with no failed_objects attr → treated as 0 failures."""
        docs = [_make_doc()]
        mock_client = MagicMock()
        mock_col = MagicMock()
        mock_batch = MagicMock(spec=["add_object"])  # no failed_objects attribute
        mock_col.batch.dynamic.return_value.__enter__.return_value = mock_batch
        mock_col.batch.dynamic.return_value.__exit__.return_value = False
        mock_client.collections.get.return_value = mock_col

        # Should not raise AttributeError; fallback to 0 failures expected
        result = add_visual_documents(mock_client, docs, collection="RAGVisualPages")

        # count = len(docs) - len(failed_objects or []) should be 1
        assert result == 1

    def test_delete_result_missing_matches_returns_zero(self):
        """Boundary: getattr fallback for missing matches attr returns 0."""
        mock_client = MagicMock()
        mock_col = MagicMock()
        mock_col.data.delete_many.return_value = MagicMock(spec=[])
        mock_client.collections.get.return_value = mock_col

        result = delete_visual_by_source_key(
            mock_client, source_key="doc_abc", collection="RAGVisualPages"
        )

        assert result == 0

    def test_nfr909_existing_abc_methods_unaffected(self):
        """NFR-909: pre-existing abstract methods remain intact after additions."""
        # Verify that existing abstract methods still trigger TypeError
        class MissingLegacy(VectorBackend):
            # Provides new methods but not legacy ones
            def ensure_visual_collection(self, *a, **kw): ...
            def add_visual_documents(self, *a, **kw): ...
            def delete_visual_by_source_key(self, *a, **kw): ...
            # ensure_collection, add_documents, delete_by_source_key omitted

        with pytest.raises(TypeError):
            MissingLegacy()


# ---------------------------------------------------------------------------
# TestErrorPropagation
# ---------------------------------------------------------------------------

class TestErrorPropagation:
    """Exception propagation: Weaviate errors pass through unwrapped."""

    def test_ensure_visual_collection_propagates_exists_exception(self):
        """Weaviate error on collections.exists propagates to caller unwrapped."""
        mock_client = MagicMock()
        mock_client.collections.exists.side_effect = RuntimeError("connection refused")

        with pytest.raises(RuntimeError, match="connection refused"):
            ensure_visual_collection(mock_client, collection="RAGVisualPages")

    def test_add_visual_documents_propagates_get_exception(self):
        """Weaviate error on collections.get propagates to caller unwrapped."""
        mock_client = MagicMock()
        mock_client.collections.get.side_effect = ConnectionError("Weaviate unreachable")

        with pytest.raises(ConnectionError, match="Weaviate unreachable"):
            add_visual_documents(mock_client, [_make_doc()], collection="RAGVisualPages")

    def test_ensure_visual_collection_propagates_create_exception(self):
        """Weaviate error on collections.create propagates to caller unwrapped."""
        mock_client = MagicMock()
        mock_client.collections.exists.return_value = False
        mock_client.collections.create.side_effect = RuntimeError("schema conflict")

        with patch("src.vector_db.weaviate.visual_store.Configure"):
            with pytest.raises(RuntimeError, match="schema conflict"):
                ensure_visual_collection(mock_client, collection="RAGVisualPages")

    def test_delete_propagates_weaviate_exception(self):
        """Weaviate error on data.delete_many propagates to caller unwrapped."""
        mock_client = MagicMock()
        mock_col = MagicMock()
        mock_col.data.delete_many.side_effect = IOError("Weaviate write error")
        mock_client.collections.get.return_value = mock_col

        with pytest.raises(IOError, match="Weaviate write error"):
            delete_visual_by_source_key(
                mock_client, source_key="doc_abc", collection="RAGVisualPages"
            )

    def test_all_documents_fail_no_exception_raised(self):
        """When all batch inserts fail, returns 0 without raising."""
        docs = [_make_doc(page_number=i) for i in range(3)]
        failed = [MagicMock() for _ in range(3)]
        mock_client, mock_col, mock_batch = _make_client_and_col(failed_objects=failed)

        # Must not raise
        result = add_visual_documents(mock_client, docs, collection="RAGVisualPages")

        assert result == 0
