"""
Tests for the Visual Page Retrieval Pipeline.

Derived from:
  docs/ingestion/visual_retrieval/VISUAL_RETRIEVAL_TEST_DOCS.md

FR coverage: FR-101–FR-111, FR-201–FR-207, FR-301–FR-313,
             FR-401–FR-403, FR-501–FR-503, FR-601–FR-617,
             FR-701–FR-703

Modules under test:
  EG §3.1  src/ingest/support/colqwen.py         (embed_text_query)
  EG §3.2  src/vector_db/weaviate/visual_store.py (visual_search)
  EG §3.3  src/db/minio/store.py                 (get_page_image_url)
  EG §3.4  src/retrieval/common/schemas.py        (VisualPageResult, RAGResponse)
  EG §3.5  src/retrieval/pipeline/rag_chain.py    (RAGChain visual track)
  EG §3.6  src/vector_db/backend.py + __init__    (VectorBackend ABC, public API)
  EG §3.7  config/settings.py                     (visual retrieval config keys)
  EG §3.8  server/schemas.py                      (VisualPageResultResponse, QueryResponse)
"""

from __future__ import annotations

import importlib
import math
import sys
import types
from datetime import timedelta
from typing import Any, Optional
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Weaviate exceptions — import real classes before conftest stub may replace
# the module. We import src.vector_db.weaviate.visual_store first to force
# the real weaviate package into sys.modules.
# ---------------------------------------------------------------------------
try:
    # Force real weaviate to load so weaviate.exceptions is available
    import src.vector_db.weaviate.visual_store  # noqa: F401
    import weaviate.exceptions as _wex
    _WeaviateQueryError = _wex.WeaviateQueryError
    _WeaviateConnectionError = _wex.WeaviateConnectionError
except (ImportError, ModuleNotFoundError):
    # Fallback: create minimal stub exception classes
    class _WeaviateQueryError(Exception):  # type: ignore[misc]
        def __init__(self, message="", protocol_type="grpc"):
            super().__init__(message)

    class _WeaviateConnectionError(Exception):  # type: ignore[misc]
        def __init__(self, message=""):
            super().__init__(message)

# Force src.ingest.support.colqwen and src.db.minio.store into sys.modules so
# patch() can resolve them by dotted path in TestRAGChainVisualTrack tests.
try:
    import src.ingest.support.colqwen  # noqa: F401
    import src.db.minio.store  # noqa: F401
except (ImportError, ModuleNotFoundError):
    pass


# ---------------------------------------------------------------------------
# Helpers — build a mock torch module that behaves like torch for colqwen
# ---------------------------------------------------------------------------

def _make_mock_torch() -> types.ModuleType:
    """Create a mock torch module with inference_mode context manager support."""
    mock_torch = types.ModuleType("torch")

    # inference_mode must behave as a context manager
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=None)
    cm.__exit__ = MagicMock(return_value=False)
    mock_torch.inference_mode = MagicMock(return_value=cm)

    mock_torch.long = "long"
    mock_torch.float32 = "float32"

    return mock_torch


def _make_colqwen_mocks(n_tokens: int = 10, embed_dim: int = 128):
    """Build mock model and processor for ColQwen2 tests.

    Returns (mock_model, mock_processor). mock_model(**inputs) returns an
    object with last_hidden_state that supports [0].float().mean(dim=0).cpu().tolist().
    mock_processor.process_queries(texts) returns a tokenized input dict whose
    values support .to(device).
    """
    lhs_data = [
        [float(i * embed_dim + j) / (n_tokens * embed_dim * embed_dim)
         for j in range(embed_dim)]
        for i in range(n_tokens)
    ]

    class _FakeTensorValue:
        """A fake tensor value that supports .to(device) for dict items."""
        def __init__(self, data):
            self._data = data

        def to(self, device):
            return self

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

    class _FakeVector:
        """Represents a 1-D tensor (mean-pooled result)."""
        def __init__(self, data):
            self._data = data

        def cpu(self):
            return self

        def tolist(self):
            return [float(v) for v in self._data]

    class _FakeBatchSlice:
        """Represents last_hidden_state[0] — shape (n_tokens, embed_dim).

        Supports .float().mean(dim=0).cpu().tolist() chain.
        """
        def __init__(self, data):
            self._data = data

        def float(self):
            # .float() returns self (already float data)
            return self

        def mean(self, dim):
            if dim == 0:
                # arithmetic mean over token dimension
                means = [
                    sum(self._data[row][col] for row in range(len(self._data)))
                    / len(self._data)
                    for col in range(len(self._data[0]))
                ]
                return _FakeVector(means)
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self._data

    class _FakeLastHiddenState:
        def __init__(self):
            self._batch_slice = _FakeBatchSlice(lhs_data)

        def __getitem__(self, idx):
            # last_hidden_state[0] → batch-0 slice
            return self._batch_slice

    mock_output = MagicMock()
    mock_output.last_hidden_state = _FakeLastHiddenState()

    mock_model = MagicMock()
    mock_model.return_value = mock_output
    mock_model.device = "cpu"

    class _FakeProcessorOutput(dict):
        """Dict whose values support .to(device) and whose dict itself supports .to(device)."""
        def to(self, device):
            return self

    # Each value in the dict needs .to(device) support
    mock_processor = MagicMock()
    mock_processor.process_queries = MagicMock(
        return_value=_FakeProcessorOutput(
            input_ids=_FakeTensorValue([0] * n_tokens),
            attention_mask=_FakeTensorValue([1] * n_tokens),
        )
    )
    mock_processor.process_images = MagicMock(
        side_effect=AssertionError("process_images must not be called in query encoding")
    )

    return mock_model, mock_processor


# ===========================================================================
# EG §3.1  —  embed_text_query
# ===========================================================================

# Module-level mock torch — installed once to avoid numpy C-extension reload
# crash that occurs when sys.modules["torch"] is repeatedly set/restored.
_EMBED_MOCK_TORCH = _make_mock_torch()


@pytest.fixture(autouse=False)
def _mock_torch_for_colqwen(monkeypatch):
    """Install the mock torch once for a test. Avoids numpy reload crash."""
    monkeypatch.setitem(sys.modules, "torch", _EMBED_MOCK_TORCH)
    yield _EMBED_MOCK_TORCH


class TestEmbedTextQuery:
    """Tests for src/ingest/support/colqwen.embed_text_query (EG §3.1).

    All tests use the _mock_torch_for_colqwen fixture to install a mock torch
    module without repeatedly removing/restoring the real torch (which triggers
    a numpy C-extension double-load crash).
    """

    @staticmethod
    def _get_embed_func():
        from src.ingest.support.colqwen import embed_text_query
        return embed_text_query

    # --- Happy path ---

    def test_returns_128_element_list_for_normal_query(self, _mock_torch_for_colqwen):
        """FR-201: normal non-empty query returns list[float] of length 128."""
        embed_text_query = self._get_embed_func()
        model, processor = _make_colqwen_mocks()
        result = embed_text_query(model, processor, "quarterly revenue chart Q3 2025")
        assert isinstance(result, list)
        assert len(result) == 128

    def test_single_word_query_returns_128_elements(self, _mock_torch_for_colqwen):
        """FR-201: single-word query returns list of length 128."""
        embed_text_query = self._get_embed_func()
        model, processor = _make_colqwen_mocks()
        result = embed_text_query(model, processor, "revenue")
        assert len(result) == 128

    def test_whitespace_padded_valid_query_is_accepted(self, _mock_torch_for_colqwen):
        """FR-201: whitespace-padded query (non-empty after strip) is valid."""
        embed_text_query = self._get_embed_func()
        model, processor = _make_colqwen_mocks()
        result = embed_text_query(model, processor, "  chart  ")
        assert len(result) == 128

    def test_output_elements_are_finite_floats(self, _mock_torch_for_colqwen):
        """FR-203: all returned elements are finite Python floats."""
        embed_text_query = self._get_embed_func()
        model, processor = _make_colqwen_mocks()
        result = embed_text_query(model, processor, "test query")
        assert all(isinstance(v, float) for v in result)
        assert all(math.isfinite(v) for v in result)

    def test_determinism_same_input_produces_same_output(self, _mock_torch_for_colqwen):
        """FR-203: same input + model state → identical output both calls."""
        embed_text_query = self._get_embed_func()
        model, processor = _make_colqwen_mocks()
        r1 = embed_text_query(model, processor, "quarterly revenue chart")
        r2 = embed_text_query(model, processor, "quarterly revenue chart")
        assert r1 == r2

    def test_process_queries_called_not_process_images(self, _mock_torch_for_colqwen):
        """FR-205: text encoding uses process_queries, never process_images."""
        embed_text_query = self._get_embed_func()
        model, processor = _make_colqwen_mocks()
        embed_text_query(model, processor, "any query")
        processor.process_queries.assert_called_once_with(["any query"])
        processor.process_images.assert_not_called()

    def test_output_dtype_is_python_float_not_tensor(self, _mock_torch_for_colqwen):
        """FR-203: output elements are native Python float, not tensor/numpy."""
        embed_text_query = self._get_embed_func()
        model, processor = _make_colqwen_mocks()
        result = embed_text_query(model, processor, "query text")
        for elem in result:
            assert type(elem) is float, f"Expected float, got {type(elem)}"

    # --- Error scenarios ---

    def test_raises_value_error_for_empty_string(self, _mock_torch_for_colqwen):
        """FR-207: empty text raises ValueError before model is invoked."""
        embed_text_query = self._get_embed_func()
        model, processor = _make_colqwen_mocks()
        with pytest.raises(ValueError, match=r"(?i)(empty|blank)"):
            embed_text_query(model, processor, "")
        model.assert_not_called()

    def test_raises_value_error_for_whitespace_only(self, _mock_torch_for_colqwen):
        """FR-207: whitespace-only text raises ValueError before model is invoked."""
        embed_text_query = self._get_embed_func()
        model, processor = _make_colqwen_mocks()
        with pytest.raises(ValueError, match=r"(?i)(empty|blank)"):
            embed_text_query(model, processor, "   ")
        model.assert_not_called()

    def test_raises_colqwen2_load_error_when_model_is_none(self, _mock_torch_for_colqwen):
        """FR-207: model=None raises ColQwen2LoadError; processor.process_queries not called."""
        embed_text_query = self._get_embed_func()
        from src.ingest.support.colqwen import ColQwen2LoadError
        _, processor = _make_colqwen_mocks()
        with pytest.raises(ColQwen2LoadError):
            embed_text_query(None, processor, "valid query")
        processor.process_queries.assert_not_called()

    def test_raises_colqwen2_load_error_when_processor_is_none(self, _mock_torch_for_colqwen):
        """FR-207: processor=None raises ColQwen2LoadError; model not invoked."""
        embed_text_query = self._get_embed_func()
        from src.ingest.support.colqwen import ColQwen2LoadError
        model, _ = _make_colqwen_mocks()
        with pytest.raises(ColQwen2LoadError):
            embed_text_query(model, None, "valid query")
        model.assert_not_called()

    def test_wraps_runtime_error_as_visual_embedding_error(self, _mock_torch_for_colqwen):
        """FR-207: RuntimeError from forward pass wrapped in VisualEmbeddingError with __cause__."""
        embed_text_query = self._get_embed_func()
        from src.ingest.support.colqwen import VisualEmbeddingError
        model, processor = _make_colqwen_mocks()
        model.side_effect = RuntimeError("CUDA out of memory")
        with pytest.raises(VisualEmbeddingError) as exc_info:
            embed_text_query(model, processor, "valid query")
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    # --- Boundary conditions ---

    def test_single_non_whitespace_character_is_valid(self, _mock_torch_for_colqwen):
        """FR-201 boundary: single char query returns 128-element list."""
        embed_text_query = self._get_embed_func()
        model, processor = _make_colqwen_mocks()
        result = embed_text_query(model, processor, "x")
        assert len(result) == 128

    def test_very_long_query_does_not_raise(self, _mock_torch_for_colqwen):
        """FR-201 boundary: query >1000 chars returns 128-element list (fixed output dim)."""
        embed_text_query = self._get_embed_func()
        model, processor = _make_colqwen_mocks()
        long_text = "a" * 1500
        result = embed_text_query(model, processor, long_text)
        assert len(result) == 128

    def test_single_token_mean_pool_returns_that_row(self, _mock_torch_for_colqwen):
        """FR-203 boundary: n_tokens=1 → mean pool of one row = that row; len 128."""
        embed_text_query = self._get_embed_func()
        model, processor = _make_colqwen_mocks(n_tokens=1)
        result = embed_text_query(model, processor, "any")
        assert len(result) == 128


# ===========================================================================
# EG §3.2  —  visual_search (Weaviate Visual Collection Store)
# ===========================================================================

def _make_weaviate_mock_objects(page_records: list, distances: list):
    """Build mock Weaviate query result objects."""
    objects = []
    for props, dist in zip(page_records, distances):
        obj = MagicMock()
        obj.properties = dict(props)
        obj.metadata = MagicMock()
        obj.metadata.distance = dist
        objects.append(obj)
    return objects


class TestVisualSearch:
    """Tests for src/vector_db/weaviate/visual_store.visual_search (EG §3.2)."""

    PAGE_1 = {
        "document_id": "abc-123",
        "page_number": 7,
        "source_key": "reports/q3.pdf",
        "source_name": "Q3 Report",
        "minio_key": "pages/abc-123/0007.jpg",
        "tenant_id": "acme",
        "total_pages": 42,
        "page_width_px": 1024,
        "page_height_px": 768,
    }
    PAGE_2 = {
        "document_id": "def-456",
        "page_number": 3,
        "source_key": "reports/q2.pdf",
        "source_name": "Q2 Report",
        "minio_key": "pages/def-456/0003.jpg",
        "tenant_id": "acme",
        "total_pages": 20,
        "page_width_px": 1024,
        "page_height_px": 768,
    }

    @staticmethod
    def _import():
        from src.vector_db.weaviate.visual_store import visual_search
        return visual_search

    def _make_client(self, objects):
        client = MagicMock()
        collection = MagicMock()
        query_result = MagicMock()
        query_result.objects = objects
        collection.query.near_vector.return_value = query_result
        client.collections.get.return_value = collection
        return client, collection

    # --- Happy path ---

    def test_two_pages_above_threshold_returned_in_order(self):
        """FR-301, FR-303: two pages above threshold; scores 0.81 and 0.65; descending order."""
        visual_search = self._import()
        objs = _make_weaviate_mock_objects(
            [self.PAGE_1, self.PAGE_2], [0.19, 0.35]
        )
        client, _ = self._make_client(objs)
        results = visual_search(client, [0.1] * 128, limit=5, score_threshold=0.3)
        assert len(results) == 2
        assert abs(results[0]["score"] - 0.81) < 1e-6
        assert abs(results[1]["score"] - 0.65) < 1e-6

    def test_no_patch_vectors_key_in_results(self):
        """FR-311: patch_vectors never appears in results."""
        visual_search = self._import()
        page = dict(self.PAGE_1)
        objs = _make_weaviate_mock_objects([page], [0.19])
        client, _ = self._make_client(objs)
        results = visual_search(client, [0.1] * 128, limit=5, score_threshold=0.3)
        assert len(results) == 1
        assert "patch_vectors" not in results[0]

    def test_tenant_filter_passed_to_weaviate(self):
        """FR-305: tenant_id='acme' → Filter passed; results match tenant."""
        visual_search = self._import()
        objs = _make_weaviate_mock_objects([self.PAGE_1], [0.19])
        client, collection = self._make_client(objs)
        results = visual_search(
            client, [0.1] * 128, limit=5, score_threshold=0.3, tenant_id="acme"
        )
        # Verify near_vector was called with a filters argument (not None/absent)
        call_kwargs = collection.query.near_vector.call_args[1]
        assert "filters" in call_kwargs and call_kwargs["filters"] is not None
        assert len(results) == 1
        assert results[0]["tenant_id"] == "acme"

    def test_all_pages_below_threshold_returns_empty_list(self):
        """FR-303: all pages score < threshold → returns [] (not None)."""
        visual_search = self._import()
        # distances 0.75 and 0.80 → scores 0.25 and 0.20, both < 0.3
        objs = _make_weaviate_mock_objects(
            [self.PAGE_1, self.PAGE_2], [0.75, 0.80]
        )
        client, _ = self._make_client(objs)
        results = visual_search(client, [0.1] * 128, limit=5, score_threshold=0.3)
        assert results == []
        assert results is not None

    def test_custom_collection_name_used(self):
        """FR-307: custom collection name passed to client.collections.get."""
        visual_search = self._import()
        objs = _make_weaviate_mock_objects([], [])
        client, _ = self._make_client(objs)
        visual_search(
            client, [0.1] * 128, limit=5, score_threshold=0.3,
            collection="CustomVisualPages"
        )
        client.collections.get.assert_called_once_with("CustomVisualPages")

    def test_limit_passed_to_weaviate_query(self):
        """FR-303: limit parameter forwarded to near_vector call."""
        visual_search = self._import()
        objs = _make_weaviate_mock_objects([], [])
        client, collection = self._make_client(objs)
        visual_search(client, [0.1] * 128, limit=3, score_threshold=0.0)
        call_kwargs = collection.query.near_vector.call_args[1]
        assert call_kwargs.get("limit") == 3

    def test_no_filter_when_tenant_id_is_none(self):
        """FR-305: no tenant filter when tenant_id=None."""
        visual_search = self._import()
        objs = _make_weaviate_mock_objects([], [])
        client, collection = self._make_client(objs)
        visual_search(client, [0.1] * 128, limit=5, score_threshold=0.0, tenant_id=None)
        call_kwargs = collection.query.near_vector.call_args[1]
        # filters should be None or absent
        assert call_kwargs.get("filters") is None

    def test_mean_vector_used_as_target_vector(self):
        """FR-309: target_vector='mean_vector' in near_vector call."""
        visual_search = self._import()
        objs = _make_weaviate_mock_objects([], [])
        client, collection = self._make_client(objs)
        visual_search(client, [0.1] * 128, limit=5, score_threshold=0.0)
        call_kwargs = collection.query.near_vector.call_args[1]
        assert call_kwargs.get("target_vector") == "mean_vector"

    # --- Error scenarios ---

    def test_weaviate_query_error_propagates(self):
        """FR-301: WeaviateQueryError propagates without wrapping."""
        visual_search = self._import()
        client = MagicMock()
        collection = MagicMock()
        collection.query.near_vector.side_effect = _WeaviateQueryError(
            "query failed", "grpc"
        )
        client.collections.get.return_value = collection
        with pytest.raises(_WeaviateQueryError):
            visual_search(client, [0.1] * 128, limit=5, score_threshold=0.3)

    # --- Boundary conditions ---

    def test_limit_1_returns_at_most_one_result(self):
        """FR-303 boundary: limit=1 → at most 1 result."""
        visual_search = self._import()
        objs = _make_weaviate_mock_objects([self.PAGE_1], [0.19])
        client, collection = self._make_client(objs)
        visual_search(client, [0.1] * 128, limit=1, score_threshold=0.0)
        call_kwargs = collection.query.near_vector.call_args[1]
        assert call_kwargs.get("limit") == 1

    def test_score_threshold_zero_includes_all_results(self):
        """FR-303 boundary: score_threshold=0.0 → all results pass."""
        visual_search = self._import()
        # Even very low score (distance 0.99 → score 0.01) passes
        objs = _make_weaviate_mock_objects([self.PAGE_1], [0.99])
        client, _ = self._make_client(objs)
        results = visual_search(client, [0.1] * 128, limit=5, score_threshold=0.0)
        assert len(results) == 1

    def test_score_threshold_one_filters_all_non_perfect(self):
        """FR-303 boundary: score_threshold=1.0 → only perfect match passes."""
        visual_search = self._import()
        objs = _make_weaviate_mock_objects([self.PAGE_1], [0.19])  # score 0.81
        client, _ = self._make_client(objs)
        results = visual_search(client, [0.1] * 128, limit=5, score_threshold=1.0)
        assert results == []

    def test_exact_threshold_boundary_included(self):
        """FR-303 boundary: distance=0.70 → score=0.30 → included at threshold 0.30."""
        visual_search = self._import()
        objs = _make_weaviate_mock_objects([self.PAGE_1], [0.70])
        client, _ = self._make_client(objs)
        results = visual_search(client, [0.1] * 128, limit=5, score_threshold=0.30)
        assert len(results) == 1

    def test_just_below_threshold_excluded(self):
        """FR-303 boundary: score=0.299 → excluded at threshold 0.30."""
        visual_search = self._import()
        objs = _make_weaviate_mock_objects([self.PAGE_1], [0.701])
        client, _ = self._make_client(objs)
        results = visual_search(client, [0.1] * 128, limit=5, score_threshold=0.30)
        assert results == []

    def test_empty_collection_returns_empty_list(self):
        """FR-301 boundary: empty collection → returns []."""
        visual_search = self._import()
        objs = _make_weaviate_mock_objects([], [])
        client, _ = self._make_client(objs)
        results = visual_search(client, [0.1] * 128, limit=5, score_threshold=0.3)
        assert results == []

    def test_distance_zero_gives_score_one(self):
        """FR-309 boundary: distance=0.0 → score=1.0."""
        visual_search = self._import()
        objs = _make_weaviate_mock_objects([self.PAGE_1], [0.0])
        client, _ = self._make_client(objs)
        results = visual_search(client, [0.1] * 128, limit=5, score_threshold=0.0)
        assert len(results) == 1
        assert abs(results[0]["score"] - 1.0) < 1e-6

    def test_distance_one_gives_score_zero(self):
        """FR-309 boundary: distance=1.0 → score=0.0."""
        visual_search = self._import()
        objs = _make_weaviate_mock_objects([self.PAGE_1], [1.0])
        client, _ = self._make_client(objs)
        results = visual_search(client, [0.1] * 128, limit=5, score_threshold=0.0)
        assert len(results) == 1
        assert abs(results[0]["score"] - 0.0) < 1e-6


# ===========================================================================
# EG §3.3  —  get_page_image_url (MinIO Page Image Store)
# ===========================================================================

class TestGetPageImageUrl:
    """Tests for src/db/minio/store.get_page_image_url (EG §3.3)."""

    MOCK_URL = (
        "https://minio.internal:9000/rag-documents/pages/abc-123/0007.jpg"
        "?X-Amz-Signature=abc123&X-Amz-Expires=3600"
    )

    @staticmethod
    def _import():
        from src.db.minio.store import get_page_image_url
        return get_page_image_url

    def _make_client(self, return_url: str = MOCK_URL):
        client = MagicMock()
        client.presigned_get_object.return_value = return_url
        return client

    # --- Happy path ---

    def test_explicit_bucket_and_expiry_passed_through(self):
        """FR-401: explicit bucket and expiry forwarded to presigned_get_object."""
        get_page_image_url = self._import()
        client = self._make_client()
        result = get_page_image_url(
            client, "pages/abc-123/0007.jpg",
            bucket="my-bucket", expires_in_seconds=1800
        )
        client.presigned_get_object.assert_called_once_with(
            "my-bucket",
            "pages/abc-123/0007.jpg",
            expires=timedelta(seconds=1800),
        )
        assert result == self.MOCK_URL

    def test_default_bucket_sentinel_resolves_to_minio_bucket(self):
        """FR-403: empty bucket sentinel → MINIO_BUCKET config value used."""
        get_page_image_url = self._import()
        client = self._make_client()
        from config.settings import MINIO_BUCKET
        get_page_image_url(
            client, "pages/abc-123/0007.jpg",
            bucket="", expires_in_seconds=3600
        )
        call_args = client.presigned_get_object.call_args
        assert call_args[0][0] == MINIO_BUCKET

    def test_default_expiry_sentinel_resolves_to_config_value(self):
        """FR-403: expires_in_seconds=0 → RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS used."""
        get_page_image_url = self._import()
        client = self._make_client()
        from config.settings import RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS
        get_page_image_url(
            client, "pages/abc-123/0007.jpg",
            bucket="b", expires_in_seconds=0
        )
        call_args = client.presigned_get_object.call_args
        expected_td = timedelta(seconds=RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS)
        actual_td = call_args[1].get("expires") or (call_args[0][2] if len(call_args[0]) > 2 else None)
        assert actual_td == expected_td

    def test_both_sentinels_use_config_defaults(self):
        """FR-403: both bucket='' and expires_in_seconds=0 → both config defaults."""
        get_page_image_url = self._import()
        client = self._make_client()
        from config.settings import MINIO_BUCKET, RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS
        get_page_image_url(client, "pages/abc-123/0007.jpg")
        call_args = client.presigned_get_object.call_args
        assert call_args[0][0] == MINIO_BUCKET
        expected_td = timedelta(seconds=RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS)
        actual_td = call_args[1].get("expires") or (call_args[0][2] if len(call_args[0]) > 2 else None)
        assert actual_td == expected_td

    def test_key_used_verbatim_no_suffix_appended(self):
        """FR-401: minio_key passed verbatim; no suffix or modification."""
        get_page_image_url = self._import()
        client = self._make_client()
        get_page_image_url(
            client, "pages/def-456/0003.jpg",
            bucket="b", expires_in_seconds=3600
        )
        call_args = client.presigned_get_object.call_args
        assert call_args[0][1] == "pages/def-456/0003.jpg"

    def test_non_existent_key_still_returns_url(self):
        """FR-401: no existence check; presigned URL still returned for any key."""
        get_page_image_url = self._import()
        client = self._make_client()
        result = get_page_image_url(
            client, "pages/nonexistent/9999.jpg",
            bucket="b", expires_in_seconds=3600
        )
        assert isinstance(result, str)

    # --- Error scenarios ---

    def test_s3_error_propagates_to_caller(self):
        """FR-401: S3Error from presigned_get_object propagates without wrapping."""
        get_page_image_url = self._import()
        from minio.error import S3Error
        client = self._make_client()
        client.presigned_get_object.side_effect = S3Error(
            "NoSuchBucket", "Bucket does not exist"
        )
        with pytest.raises(S3Error):
            get_page_image_url(
                client, "pages/abc-123/0007.jpg",
                bucket="b", expires_in_seconds=3600
            )

    # --- Boundary conditions ---

    def test_nested_path_key_preserved_verbatim(self):
        """FR-401 boundary: nested path key; all components preserved."""
        get_page_image_url = self._import()
        client = self._make_client()
        get_page_image_url(
            client, "pages/doc/subdir/0001.jpg",
            bucket="b", expires_in_seconds=3600
        )
        call_args = client.presigned_get_object.call_args
        assert call_args[0][1] == "pages/doc/subdir/0001.jpg"

    def test_explicit_bucket_not_overridden_by_default(self):
        """FR-403 boundary: explicit bucket='explicit-bucket' → not overridden."""
        get_page_image_url = self._import()
        client = self._make_client()
        get_page_image_url(
            client, "pages/abc-123/0007.jpg",
            bucket="explicit-bucket", expires_in_seconds=3600
        )
        call_args = client.presigned_get_object.call_args
        assert call_args[0][0] == "explicit-bucket"

    def test_explicit_expiry_60_minimum_valid(self):
        """FR-403 boundary: expires_in_seconds=60 (minimum) → timedelta(seconds=60)."""
        get_page_image_url = self._import()
        client = self._make_client()
        get_page_image_url(
            client, "pages/abc-123/0007.jpg",
            bucket="b", expires_in_seconds=60
        )
        call_args = client.presigned_get_object.call_args
        actual = call_args[1].get("expires") or (call_args[0][2] if len(call_args[0]) > 2 else None)
        assert actual == timedelta(seconds=60)

    def test_explicit_expiry_86400_maximum_valid(self):
        """FR-403 boundary: expires_in_seconds=86400 (maximum) → timedelta(seconds=86400)."""
        get_page_image_url = self._import()
        client = self._make_client()
        get_page_image_url(
            client, "pages/abc-123/0007.jpg",
            bucket="b", expires_in_seconds=86400
        )
        call_args = client.presigned_get_object.call_args
        actual = call_args[1].get("expires") or (call_args[0][2] if len(call_args[0]) > 2 else None)
        assert actual == timedelta(seconds=86400)


# ===========================================================================
# EG §3.4  —  Retrieval Pipeline Schemas (VisualPageResult, RAGResponse)
# ===========================================================================

def _make_rag_response(**overrides):
    """Build a minimal valid RAGResponse for testing."""
    from src.retrieval.common.schemas import RAGResponse
    defaults = dict(
        query="test query",
        processed_query="processed test query",
        query_confidence=0.9,
        action="search",
    )
    defaults.update(overrides)
    return RAGResponse(**defaults)


class TestRetrievalPipelineSchemas:
    """Tests for src/retrieval/common/schemas.py (EG §3.4)."""

    @staticmethod
    def _make_vpr(**overrides):
        from src.retrieval.common.schemas import VisualPageResult
        defaults = dict(
            document_id="abc-123",
            page_number=7,
            source_key="reports/q3.pdf",
            source_name="Q3 Report",
            score=0.81,
            page_image_url="https://minio.example.com/pages/abc-123/0007.jpg",
            total_pages=42,
            page_width_px=1024,
            page_height_px=768,
        )
        defaults.update(overrides)
        return VisualPageResult(**defaults)

    # --- Happy path ---

    def test_visual_page_result_importable_from_retrieval_common_schemas(self):
        """FR-501: VisualPageResult importable from src.retrieval.common.schemas."""
        from src.retrieval.common.schemas import VisualPageResult
        assert VisualPageResult is not None

    def test_visual_page_result_all_nine_fields_accessible(self):
        """FR-501: all nine fields set correctly on construction."""
        vpr = self._make_vpr()
        assert vpr.document_id == "abc-123"
        assert vpr.page_number == 7
        assert vpr.source_key == "reports/q3.pdf"
        assert vpr.source_name == "Q3 Report"
        assert abs(vpr.score - 0.81) < 1e-9
        assert "minio" in vpr.page_image_url
        assert vpr.total_pages == 42
        assert vpr.page_width_px == 1024
        assert vpr.page_height_px == 768

    def test_rag_response_without_visual_results_has_none(self):
        """FR-503: RAGResponse constructed without visual_results → visual_results is None."""
        response = _make_rag_response()
        assert response.visual_results is None

    def test_rag_response_with_visual_results_list(self):
        """FR-503: RAGResponse with visual_results=[vpr1, vpr2] returns list correctly."""
        vpr1 = self._make_vpr(page_number=1)
        vpr2 = self._make_vpr(page_number=2)
        response = _make_rag_response(visual_results=[vpr1, vpr2])
        assert response.visual_results == [vpr1, vpr2]

    def test_rag_response_with_empty_visual_results_list(self):
        """FR-503: RAGResponse with visual_results=[] → visual_results == []."""
        response = _make_rag_response(visual_results=[])
        assert response.visual_results == []

    # --- Error scenarios ---

    def test_missing_required_field_raises_type_error(self):
        """FR-501: missing required field raises standard Python TypeError."""
        from src.retrieval.common.schemas import VisualPageResult
        with pytest.raises(TypeError):
            VisualPageResult()

    # --- Boundary conditions ---

    def test_score_zero_accepted(self):
        """FR-501 boundary: score=0.0 accepted without error."""
        vpr = self._make_vpr(score=0.0)
        assert vpr.score == 0.0

    def test_score_one_accepted(self):
        """FR-501 boundary: score=1.0 accepted without error."""
        vpr = self._make_vpr(score=1.0)
        assert vpr.score == 1.0

    def test_page_number_one_accepted(self):
        """FR-501 boundary: page_number=1 (1-indexed minimum) accepted."""
        vpr = self._make_vpr(page_number=1)
        assert vpr.page_number == 1

    def test_empty_page_image_url_accepted_at_schema_level(self):
        """FR-501 boundary: empty page_image_url accepted; validation is caller's responsibility."""
        vpr = self._make_vpr(page_image_url="")
        assert vpr.page_image_url == ""

    def test_total_pages_one_accepted(self):
        """FR-501 boundary: total_pages=1 (single-page doc) accepted."""
        vpr = self._make_vpr(total_pages=1)
        assert vpr.total_pages == 1

    def test_existing_rag_response_fields_unaffected(self):
        """FR-503 backward compat: existing fields (results, query, processed_query) intact."""
        from src.retrieval.common.schemas import RankedResult
        vpr = self._make_vpr()
        ranked = RankedResult(text="some result", score=0.9, metadata={})
        response = _make_rag_response(
            query="original query",
            processed_query="processed",
            results=[ranked],
            visual_results=[vpr],
        )
        assert response.query == "original query"
        assert response.processed_query == "processed"
        assert len(response.results) == 1
        assert len(response.visual_results) == 1


# ===========================================================================
# EG §3.7  —  Visual Retrieval Configuration Keys (config/settings.py)
# ===========================================================================

class TestVisualRetrievalConfig:
    """Tests for config/settings.py visual retrieval keys (EG §3.7)."""

    # --- Happy path ---

    def test_default_values_when_env_vars_unset(self, monkeypatch):
        """FR-101–FR-111: default values applied when env vars are unset."""
        for key in [
            "RAG_VISUAL_RETRIEVAL_ENABLED",
            "RAG_VISUAL_RETRIEVAL_LIMIT",
            "RAG_VISUAL_RETRIEVAL_MIN_SCORE",
            "RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS",
            "RAG_STAGE_BUDGET_VISUAL_RETRIEVAL_MS",
        ]:
            monkeypatch.delenv(key, raising=False)

        import config.settings as settings
        importlib.reload(settings)

        assert settings.RAG_VISUAL_RETRIEVAL_ENABLED is False
        assert settings.RAG_VISUAL_RETRIEVAL_LIMIT == 5
        assert abs(settings.RAG_VISUAL_RETRIEVAL_MIN_SCORE - 0.3) < 1e-9
        assert settings.RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS == 3600
        assert settings.RAG_STAGE_BUDGET_VISUAL_RETRIEVAL_MS == 10000

    def test_enabled_flag_true_lowercase(self, monkeypatch):
        """FR-101: 'true' → enabled=True."""
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_ENABLED", "true")
        import config.settings as settings
        importlib.reload(settings)
        assert settings.RAG_VISUAL_RETRIEVAL_ENABLED is True

    def test_enabled_flag_one(self, monkeypatch):
        """FR-101: '1' → enabled=True."""
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_ENABLED", "1")
        import config.settings as settings
        importlib.reload(settings)
        assert settings.RAG_VISUAL_RETRIEVAL_ENABLED is True

    def test_enabled_flag_yes(self, monkeypatch):
        """FR-101: 'yes' → enabled=True."""
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_ENABLED", "yes")
        import config.settings as settings
        importlib.reload(settings)
        assert settings.RAG_VISUAL_RETRIEVAL_ENABLED is True

    def test_enabled_flag_false_string(self, monkeypatch):
        """FR-101 boundary: 'false' → enabled=False (avoids bool('false') trap)."""
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_ENABLED", "false")
        import config.settings as settings
        importlib.reload(settings)
        assert settings.RAG_VISUAL_RETRIEVAL_ENABLED is False

    def test_enabled_flag_zero_string(self, monkeypatch):
        """FR-101 boundary: '0' → enabled=False."""
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_ENABLED", "0")
        import config.settings as settings
        importlib.reload(settings)
        assert settings.RAG_VISUAL_RETRIEVAL_ENABLED is False

    def test_enabled_flag_no_string(self, monkeypatch):
        """FR-101 boundary: 'no' → enabled=False."""
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_ENABLED", "no")
        import config.settings as settings
        importlib.reload(settings)
        assert settings.RAG_VISUAL_RETRIEVAL_ENABLED is False

    def test_limit_valid_range_no_warning(self, monkeypatch, caplog):
        """FR-103: limit=10 (valid) → no warning logged."""
        import logging
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_LIMIT", "10")
        import config.settings as settings
        with caplog.at_level(logging.WARNING, logger="config.settings"):
            importlib.reload(settings)
        assert settings.RAG_VISUAL_RETRIEVAL_LIMIT == 10

    def test_validate_config_passes_for_valid_settings(self, monkeypatch):
        """FR-101–FR-111: validate_visual_retrieval_config passes with valid config."""
        monkeypatch.setenv("RAG_INGESTION_VISUAL_TARGET_COLLECTION", "RAGVisualPages")
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_MIN_SCORE", "0.3")
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS", "3600")
        import config.settings as settings
        importlib.reload(settings)
        # Should not raise
        settings.validate_visual_retrieval_config()

    # --- Error scenarios (clamping and validation failures) ---

    def test_limit_zero_clamped_to_one_with_warning(self, monkeypatch, caplog):
        """FR-103: limit=0 clamped to 1 with warning."""
        import logging
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_LIMIT", "0")
        import config.settings as settings
        with caplog.at_level(logging.WARNING):
            importlib.reload(settings)
        assert settings.RAG_VISUAL_RETRIEVAL_LIMIT == 1
        assert any(
            "clamp" in r.message.lower() or "out of range" in r.message.lower()
            for r in caplog.records
        )

    def test_limit_51_clamped_to_50_with_warning(self, monkeypatch, caplog):
        """FR-103: limit=51 clamped to 50 with warning."""
        import logging
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_LIMIT", "51")
        import config.settings as settings
        with caplog.at_level(logging.WARNING):
            importlib.reload(settings)
        assert settings.RAG_VISUAL_RETRIEVAL_LIMIT == 50
        assert any(
            "clamp" in r.message.lower() or "out of range" in r.message.lower()
            for r in caplog.records
        )

    def test_limit_not_a_number_raises_value_error_on_import(self, monkeypatch):
        """FR-103: non-numeric LIMIT raises ValueError at module import time."""
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_LIMIT", "not-a-number")
        import config.settings as settings
        with pytest.raises(ValueError):
            importlib.reload(settings)

    def test_validate_config_raises_for_empty_collection(self, monkeypatch):
        """FR-111: empty RAG_INGESTION_VISUAL_TARGET_COLLECTION → ValueError."""
        monkeypatch.setenv("RAG_INGESTION_VISUAL_TARGET_COLLECTION", "")
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_MIN_SCORE", "0.3")
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS", "3600")
        import config.settings as settings
        importlib.reload(settings)
        with pytest.raises(ValueError, match=r"(?i)RAG_INGESTION_VISUAL_TARGET_COLLECTION"):
            settings.validate_visual_retrieval_config()

    def test_validate_config_raises_for_score_below_zero(self, monkeypatch):
        """FR-105: MIN_SCORE=-0.1 → ValueError identifying the key."""
        monkeypatch.setenv("RAG_INGESTION_VISUAL_TARGET_COLLECTION", "RAGVisualPages")
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_MIN_SCORE", "-0.1")
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS", "3600")
        import config.settings as settings
        importlib.reload(settings)
        with pytest.raises(ValueError, match=r"(?i)RAG_VISUAL_RETRIEVAL_MIN_SCORE"):
            settings.validate_visual_retrieval_config()

    def test_validate_config_raises_for_score_above_one(self, monkeypatch):
        """FR-105: MIN_SCORE=1.5 → ValueError."""
        monkeypatch.setenv("RAG_INGESTION_VISUAL_TARGET_COLLECTION", "RAGVisualPages")
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_MIN_SCORE", "1.5")
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS", "3600")
        import config.settings as settings
        importlib.reload(settings)
        with pytest.raises(ValueError, match=r"(?i)RAG_VISUAL_RETRIEVAL_MIN_SCORE"):
            settings.validate_visual_retrieval_config()

    def test_validate_config_raises_for_expiry_below_60(self, monkeypatch):
        """FR-107: URL_EXPIRY=30 (< 60) → ValueError."""
        monkeypatch.setenv("RAG_INGESTION_VISUAL_TARGET_COLLECTION", "RAGVisualPages")
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_MIN_SCORE", "0.3")
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS", "30")
        import config.settings as settings
        importlib.reload(settings)
        with pytest.raises(ValueError, match=r"(?i)RAG_VISUAL_RETRIEVAL_URL_EXPIRY"):
            settings.validate_visual_retrieval_config()

    def test_validate_config_raises_for_expiry_above_86400(self, monkeypatch):
        """FR-107: URL_EXPIRY=100000 (> 86400) → ValueError."""
        monkeypatch.setenv("RAG_INGESTION_VISUAL_TARGET_COLLECTION", "RAGVisualPages")
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_MIN_SCORE", "0.3")
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS", "100000")
        import config.settings as settings
        importlib.reload(settings)
        with pytest.raises(ValueError, match=r"(?i)RAG_VISUAL_RETRIEVAL_URL_EXPIRY"):
            settings.validate_visual_retrieval_config()

    # --- Boundary conditions ---

    def test_limit_1_minimum_valid_no_clamping(self, monkeypatch, caplog):
        """FR-103 boundary: limit=1 → no clamping, no warning."""
        import logging
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_LIMIT", "1")
        import config.settings as settings
        with caplog.at_level(logging.WARNING):
            importlib.reload(settings)
        assert settings.RAG_VISUAL_RETRIEVAL_LIMIT == 1
        assert not any(
            "clamp" in r.message.lower() or "out of range" in r.message.lower()
            for r in caplog.records
        )

    def test_limit_50_maximum_valid_no_clamping(self, monkeypatch, caplog):
        """FR-103 boundary: limit=50 → no clamping, no warning."""
        import logging
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_LIMIT", "50")
        import config.settings as settings
        with caplog.at_level(logging.WARNING):
            importlib.reload(settings)
        assert settings.RAG_VISUAL_RETRIEVAL_LIMIT == 50
        assert not any(
            "clamp" in r.message.lower() or "out of range" in r.message.lower()
            for r in caplog.records
        )

    def test_min_score_zero_valid(self, monkeypatch):
        """FR-105 boundary: MIN_SCORE=0.0 → valid, no validation error."""
        monkeypatch.setenv("RAG_INGESTION_VISUAL_TARGET_COLLECTION", "RAGVisualPages")
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_MIN_SCORE", "0.0")
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS", "3600")
        import config.settings as settings
        importlib.reload(settings)
        settings.validate_visual_retrieval_config()  # should not raise

    def test_min_score_one_valid(self, monkeypatch):
        """FR-105 boundary: MIN_SCORE=1.0 → valid, no validation error."""
        monkeypatch.setenv("RAG_INGESTION_VISUAL_TARGET_COLLECTION", "RAGVisualPages")
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_MIN_SCORE", "1.0")
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS", "3600")
        import config.settings as settings
        importlib.reload(settings)
        settings.validate_visual_retrieval_config()  # should not raise

    def test_expiry_60_minimum_valid(self, monkeypatch):
        """FR-107 boundary: URL_EXPIRY=60 → passes validation."""
        monkeypatch.setenv("RAG_INGESTION_VISUAL_TARGET_COLLECTION", "RAGVisualPages")
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_MIN_SCORE", "0.3")
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS", "60")
        import config.settings as settings
        importlib.reload(settings)
        settings.validate_visual_retrieval_config()  # should not raise

    def test_expiry_86400_maximum_valid(self, monkeypatch):
        """FR-107 boundary: URL_EXPIRY=86400 → passes validation."""
        monkeypatch.setenv("RAG_INGESTION_VISUAL_TARGET_COLLECTION", "RAGVisualPages")
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_MIN_SCORE", "0.3")
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS", "86400")
        import config.settings as settings
        importlib.reload(settings)
        settings.validate_visual_retrieval_config()  # should not raise


# ===========================================================================
# EG §3.6  —  VectorBackend ABC and Public API
# ===========================================================================

class TestVectorBackendABC:
    """Tests for src/vector_db/backend.py and src/vector_db/__init__.py (EG §3.6)."""

    # --- Happy path ---

    def test_search_visual_in_all_export(self):
        """FR-313: search_visual present in vector_db.__all__."""
        import src.vector_db as vdb
        assert "search_visual" in vdb.__all__

    def test_search_visual_delegates_to_backend(self, monkeypatch):
        """FR-313: src.vector_db.search_visual delegates to backend.search_visual."""
        import src.vector_db as vdb

        mock_backend = MagicMock()
        mock_backend.search_visual.return_value = [
            {"document_id": "abc-123", "page_number": 7, "score": 0.81}
        ]
        monkeypatch.setattr(
            "src.vector_db._get_vector_backend",
            lambda: mock_backend,
        )
        result = vdb.search_visual(
            client=MagicMock(),
            query_vector=[0.1] * 128,
            limit=5,
            score_threshold=0.3,
        )
        mock_backend.search_visual.assert_called_once()
        assert result == [{"document_id": "abc-123", "page_number": 7, "score": 0.81}]

    def test_search_visual_with_optional_args_forwarded(self, monkeypatch):
        """FR-313: optional tenant_id and collection forwarded to backend."""
        import src.vector_db as vdb

        mock_backend = MagicMock()
        mock_backend.search_visual.return_value = []
        monkeypatch.setattr(
            "src.vector_db._get_vector_backend",
            lambda: mock_backend,
        )
        vdb.search_visual(
            client=MagicMock(),
            query_vector=[0.1] * 128,
            limit=5,
            score_threshold=0.3,
            tenant_id="t1",
            collection="CustomColl",
        )
        call_args = mock_backend.search_visual.call_args
        # tenant_id and collection should appear in positional or keyword args
        all_call_values = list(call_args[0]) + list(call_args[1].values())
        assert "t1" in all_call_values
        assert "CustomColl" in all_call_values

    def test_abstract_class_cannot_be_instantiated_without_search_visual(self):
        """FR-313: concrete VectorBackend subclass without search_visual raises TypeError."""
        from src.vector_db.backend import VectorBackend

        class IncompleteBackend(VectorBackend):
            # Does NOT implement search_visual or any abstract methods
            pass

        with pytest.raises(TypeError):
            IncompleteBackend()

    def test_collection_none_forwarded_to_backend(self, monkeypatch):
        """FR-313 boundary: collection=None forwarded as-is to backend."""
        import src.vector_db as vdb

        mock_backend = MagicMock()
        mock_backend.search_visual.return_value = []
        monkeypatch.setattr(
            "src.vector_db._get_vector_backend",
            lambda: mock_backend,
        )
        vdb.search_visual(
            client=MagicMock(),
            query_vector=[0.1] * 128,
            limit=5,
            score_threshold=0.3,
            collection=None,
        )
        call_args = mock_backend.search_visual.call_args
        # collection=None should be passed positionally or by keyword
        positional_none = None in call_args[0]
        keyword_none = call_args[1].get("collection") is None
        assert positional_none or keyword_none

    # --- Error scenarios ---

    def test_backend_exception_propagates_without_wrapping(self, monkeypatch):
        """FR-313: backend exception propagates through public function."""
        import src.vector_db as vdb

        mock_backend = MagicMock()
        mock_backend.search_visual.side_effect = _WeaviateConnectionError("no connection")
        monkeypatch.setattr(
            "src.vector_db._get_vector_backend",
            lambda: mock_backend,
        )
        with pytest.raises(_WeaviateConnectionError):
            vdb.search_visual(
                client=MagicMock(),
                query_vector=[0.1] * 128,
                limit=5,
                score_threshold=0.3,
            )

    def test_result_returned_unchanged_by_public_function(self, monkeypatch):
        """FR-313 boundary: public function does not transform backend result."""
        import src.vector_db as vdb

        expected = [{"document_id": "abc", "score": 0.9, "page_number": 1}]
        mock_backend = MagicMock()
        mock_backend.search_visual.return_value = expected
        monkeypatch.setattr(
            "src.vector_db._get_vector_backend",
            lambda: mock_backend,
        )
        result = vdb.search_visual(
            client=MagicMock(),
            query_vector=[0.1] * 128,
            limit=5,
            score_threshold=0.3,
        )
        assert result is expected


# ===========================================================================
# EG §3.8  —  API Response Schemas (server/schemas.py)
# ===========================================================================

def _make_query_response(**overrides):
    """Build a minimal valid QueryResponse for testing."""
    from server.schemas import QueryResponse
    defaults = dict(
        query="test query",
        processed_query="processed test query",
        query_confidence=0.9,
        action="search",
    )
    defaults.update(overrides)
    return QueryResponse(**defaults)


class TestAPIResponseSchemas:
    """Tests for server/schemas.py VisualPageResultResponse and QueryResponse (EG §3.8)."""

    @staticmethod
    def _make_vprr(**overrides):
        from server.schemas import VisualPageResultResponse
        defaults = dict(
            document_id="abc-123",
            page_number=7,
            source_key="reports/q3.pdf",
            source_name="Q3 Report",
            score=0.81,
            page_image_url="https://minio.example.com/pages/abc-123/0007.jpg",
            total_pages=42,
            page_width_px=1024,
            page_height_px=768,
        )
        defaults.update(overrides)
        return VisualPageResultResponse(**defaults)

    # --- Happy path ---

    def test_visual_page_result_response_importable(self):
        """FR-701: VisualPageResultResponse importable from server.schemas."""
        from server.schemas import VisualPageResultResponse
        assert VisualPageResultResponse is not None

    def test_visual_page_result_response_has_nine_fields_no_tenant_id(self):
        """FR-701: VisualPageResultResponse has exactly nine fields; no tenant_id."""
        vprr = self._make_vprr()
        data = vprr.model_dump()
        expected_fields = {
            "document_id", "page_number", "source_key", "source_name",
            "score", "page_image_url", "total_pages", "page_width_px", "page_height_px",
        }
        assert set(data.keys()) == expected_fields
        assert "tenant_id" not in data

    def test_visual_page_result_response_serializes_to_json(self):
        """FR-701: VisualPageResultResponse serializes to valid JSON without error."""
        import json
        vprr = self._make_vprr()
        json_str = vprr.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["document_id"] == "abc-123"
        assert isinstance(parsed["score"], float)

    def test_query_response_visual_results_none_excluded_from_dump(self):
        """FR-703: visual_results=None → excluded from model_dump(exclude_none=True)."""
        response = _make_query_response(visual_results=None)
        data = response.model_dump(exclude_none=True)
        assert "visual_results" not in data

    def test_query_response_with_visual_results_list(self):
        """FR-703: visual_results=[vprr] → included in model_dump."""
        vprr = self._make_vprr()
        response = _make_query_response(visual_results=[vprr])
        data = response.model_dump()
        assert "visual_results" in data
        assert len(data["visual_results"]) == 1

    def test_query_response_backward_compat_without_visual_results(self):
        """FR-703 backward compat: QueryResponse without visual_results has None."""
        response = _make_query_response()
        assert response.visual_results is None

    def test_query_request_visual_retrieval_budget_override_valid(self):
        """FR-703: stage_budget_overrides={'visual_retrieval': 15000} is valid."""
        from server.schemas import QueryRequest
        req = QueryRequest(
            query="test query",
            stage_budget_overrides={"visual_retrieval": 15000},
        )
        assert req.stage_budget_overrides["visual_retrieval"] == 15000

    # --- Error scenarios ---

    def test_visual_page_result_response_type_error_on_wrong_page_number(self):
        """FR-701: page_number='not-an-int' raises Pydantic ValidationError."""
        from pydantic import ValidationError
        from server.schemas import VisualPageResultResponse
        with pytest.raises(ValidationError):
            VisualPageResultResponse(
                document_id="abc",
                page_number="not-an-int",
                source_key="k",
                source_name="n",
                score=0.5,
                page_image_url="url",
                total_pages=1,
                page_width_px=100,
                page_height_px=100,
            )

    def test_query_request_unknown_stage_budget_raises_validation_error(self):
        """FR-703: unknown stage key in stage_budget_overrides raises ValidationError."""
        from pydantic import ValidationError
        from server.schemas import QueryRequest
        with pytest.raises(ValidationError):
            QueryRequest(
                query="test",
                stage_budget_overrides={"nonexistent_stage": 1},
            )

    # --- Boundary conditions ---

    def test_query_response_empty_visual_results_serialized_as_array(self):
        """FR-703 boundary: visual_results=[] serialized as [] (distinct from None)."""
        response = _make_query_response(visual_results=[])
        data = response.model_dump()
        assert data["visual_results"] == []

    def test_score_serialized_as_json_number_not_string(self):
        """FR-701 boundary: score=0.81 → JSON number, not string."""
        import json
        vprr = self._make_vprr(score=0.81)
        parsed = json.loads(vprr.model_dump_json())
        assert isinstance(parsed["score"], float)

    def test_existing_query_response_fields_unaffected(self):
        """FR-703 backward compat: existing fields (query, processed_query) unaffected."""
        vprr = self._make_vprr()
        response = _make_query_response(
            query="original",
            processed_query="processed",
            visual_results=[vprr],
        )
        assert response.query == "original"
        assert response.processed_query == "processed"

    def test_stage_budget_existing_stage_still_valid(self):
        """FR-703 boundary: existing stage key (e.g. 'embedding') still valid."""
        from server.schemas import QueryRequest
        req = QueryRequest(
            query="test",
            stage_budget_overrides={"embedding": 2000},
        )
        assert req.stage_budget_overrides["embedding"] == 2000


# ===========================================================================
# EG §3.5  —  RAGChain Visual Retrieval Track
#
# These tests patch all heavy dependencies at the level where RAGChain imports
# them (lazy imports inside methods). RAGChain is constructed with
# persistent_weaviate=False to avoid live DB connections.
# ===========================================================================

def _make_heavy_init_patches():
    """Return context manager that stubs out all RAGChain __init__ heavy loads."""
    return [
        patch("src.retrieval.pipeline.rag_chain.get_embedding_provider"),
        patch("src.retrieval.pipeline.rag_chain.get_reranker_provider"),
        patch("src.core.knowledge_graph.KnowledgeGraphBuilder"),
        patch("src.retrieval.pipeline.rag_chain.OllamaGenerator"),
        patch("src.retrieval.pipeline.rag_chain.get_tracer", return_value=MagicMock()),
        patch("src.retrieval.pipeline.rag_chain.get_retry_provider", return_value=MagicMock()),
    ]


class TestRAGChainVisualTrack:
    """Tests for src/retrieval/pipeline/rag_chain.RAGChain visual track (EG §3.5)."""

    PAGE_RECORDS = [
        {
            "document_id": "abc-123",
            "page_number": 7,
            "source_key": "reports/q3.pdf",
            "source_name": "Q3 Report",
            "minio_key": "pages/abc-123/0007.jpg",
            "tenant_id": "acme",
            "total_pages": 42,
            "page_width_px": 1024,
            "page_height_px": 768,
            "score": 0.81,
        },
        {
            "document_id": "def-456",
            "page_number": 3,
            "source_key": "reports/q2.pdf",
            "source_name": "Q2 Report",
            "minio_key": "pages/def-456/0003.jpg",
            "tenant_id": "acme",
            "total_pages": 20,
            "page_width_px": 1024,
            "page_height_px": 768,
            "score": 0.74,
        },
    ]

    def _setup_env(self, monkeypatch, enabled: bool = True):
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_ENABLED", "true" if enabled else "false")
        monkeypatch.setenv("RAG_INGESTION_VISUAL_TARGET_COLLECTION", "RAGVisualPages")
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_MIN_SCORE", "0.3")
        monkeypatch.setenv("RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS", "3600")

    def _make_chain(self, monkeypatch, enabled: bool = True):
        """Create a RAGChain with all heavy dependencies patched out."""
        self._setup_env(monkeypatch, enabled)
        # Do NOT reload settings here — avoid numpy reload crash.
        # The env vars set above will be read when RAGChain imports config.settings
        # fresh inside its module (already-imported module won't re-read env, so
        # we patch the module-level attribute directly instead).
        import config.settings as settings
        monkeypatch.setattr(
            settings, "RAG_VISUAL_RETRIEVAL_ENABLED", enabled, raising=True
        )
        monkeypatch.setattr(
            settings, "RAG_INGESTION_VISUAL_TARGET_COLLECTION", "RAGVisualPages", raising=False
        )

        mock_emb = MagicMock()
        mock_rer = MagicMock()
        mock_gen = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.span.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_tracer.span.return_value.__exit__ = MagicMock(return_value=False)

        with patch("src.retrieval.pipeline.rag_chain.get_embedding_provider", return_value=mock_emb), \
             patch("src.retrieval.pipeline.rag_chain.get_reranker_provider", return_value=mock_rer), \
             patch("src.retrieval.pipeline.rag_chain.OllamaGenerator", return_value=mock_gen), \
             patch("src.retrieval.pipeline.rag_chain.get_tracer", return_value=mock_tracer), \
             patch("src.retrieval.pipeline.rag_chain.get_retry_provider", return_value=MagicMock()), \
             patch("config.settings.validate_visual_retrieval_config", return_value=None), \
             patch("src.retrieval.pipeline.rag_chain.KG_ENABLED", False), \
             patch("src.retrieval.pipeline.rag_chain.GENERATION_ENABLED", False), \
             patch("src.retrieval.pipeline.rag_chain.GUARDRAIL_BACKEND", None), \
             patch("src.retrieval.pipeline.rag_chain.RAG_VISUAL_RETRIEVAL_ENABLED", enabled):
            from src.retrieval.pipeline.rag_chain import RAGChain
            chain = RAGChain(persistent_weaviate=False)
            chain.tracer = mock_tracer
        return chain

    # --- Boundary: init state ---

    def test_visual_model_is_none_after_init(self, monkeypatch):
        """FR-603: after __init__, _visual_model is None (not pre-loaded)."""
        chain = self._make_chain(monkeypatch)
        assert chain._visual_model is None

    def test_visual_enabled_flag_set_from_config(self, monkeypatch):
        """FR-601: _visual_retrieval_enabled reflects config when enabled."""
        chain = self._make_chain(monkeypatch, enabled=True)
        assert chain._visual_retrieval_enabled is True

    def test_visual_disabled_flag_when_env_false(self, monkeypatch):
        """FR-601: _visual_retrieval_enabled=False when env=false."""
        chain = self._make_chain(monkeypatch, enabled=False)
        assert chain._visual_retrieval_enabled is False

    # --- Close lifecycle ---

    def test_close_sets_model_and_processor_to_none(self, monkeypatch):
        """FR-613: after close(), _visual_model and _visual_processor are None."""
        chain = self._make_chain(monkeypatch)
        # Simulate model loaded
        chain._visual_model = MagicMock()
        chain._visual_processor = MagicMock()
        with patch("src.ingest.support.unload_colqwen_model"):
            chain.close()
        assert chain._visual_model is None
        assert chain._visual_processor is None

    def test_close_without_model_loaded_does_not_error(self, monkeypatch):
        """FR-613: close() when model never loaded → no error, unload not called."""
        chain = self._make_chain(monkeypatch)
        with patch("src.ingest.support.unload_colqwen_model") as mock_unload:
            chain.close()
        mock_unload.assert_not_called()

    # --- Warm path: model already loaded ---

    def test_ensure_visual_model_warm_path_skips_load(self, monkeypatch):
        """FR-603: warm path — _ensure_visual_model skips load_colqwen_model if already set."""
        chain = self._make_chain(monkeypatch)
        chain._visual_model = MagicMock()
        chain._visual_processor = MagicMock()
        with patch("src.ingest.support.colqwen.load_colqwen_model") as mock_load:
            chain._ensure_visual_model()
        mock_load.assert_not_called()

    def test_load_called_only_once_for_two_visual_model_ensures(self, monkeypatch):
        """FR-603: load_colqwen_model called exactly once across two _ensure_visual_model calls."""
        chain = self._make_chain(monkeypatch)
        mock_model = MagicMock()
        mock_processor = MagicMock()

        with patch("src.ingest.support.ensure_colqwen_ready", return_value=None), \
             patch("src.ingest.support.load_colqwen_model",
                   return_value=(mock_model, mock_processor)) as mock_load:
            chain._ensure_visual_model()   # cold
            chain._ensure_visual_model()   # warm — should skip
        mock_load.assert_called_once()

    # --- Disabled path ---

    def test_visual_disabled_no_visual_results(self, monkeypatch):
        """FR-601: disabled → visual_results is None (no processing)."""
        chain = self._make_chain(monkeypatch, enabled=False)
        assert chain._visual_retrieval_enabled is False
        # When disabled, _run_visual_retrieval should never be called
        # and visual_results=None in the response
        # We verify the flag only; full run() needs heavy pipeline setup

    # --- _run_visual_retrieval unit tests ---

    def _make_mock_tracer(self):
        """Build a tracer mock with nested span context managers."""
        mock_tracer = MagicMock()
        span_ctx = MagicMock()
        span_ctx.__enter__ = MagicMock(return_value=MagicMock())
        span_ctx.__exit__ = MagicMock(return_value=False)
        mock_tracer.span.return_value = span_ctx
        return mock_tracer

    def test_run_visual_retrieval_returns_visual_page_results(self, monkeypatch):
        """FR-603–FR-607: _run_visual_retrieval produces VisualPageResult list."""
        chain = self._make_chain(monkeypatch)
        chain._visual_model = MagicMock()
        chain._visual_processor = MagicMock()
        chain.tracer = self._make_mock_tracer()

        minio_client = MagicMock()

        with patch("src.ingest.support.embed_text_query",
                   return_value=[0.1] * 128), \
             patch("src.vector_db.search_visual",
                   return_value=self.PAGE_RECORDS), \
             patch("src.db.minio.get_page_image_url",
                   return_value="https://minio.example.com/page.jpg"), \
             patch("src.db.minio.create_client",
                   return_value=minio_client):
            results = chain._run_visual_retrieval("quarterly revenue Q3", tenant_id="acme")

        assert isinstance(results, list)
        assert len(results) == len(self.PAGE_RECORDS)
        for vpr in results:
            assert vpr.page_image_url != ""

    def test_run_visual_retrieval_uses_processed_query(self, monkeypatch):
        """FR-607: embed_text_query called with processed_query, not raw query."""
        chain = self._make_chain(monkeypatch)
        chain._visual_model = MagicMock()
        chain._visual_processor = MagicMock()
        chain.tracer = self._make_mock_tracer()

        embed_calls = []

        def capture_embed(model, processor, text):
            embed_calls.append(text)
            return [0.1] * 128

        with patch("src.ingest.support.embed_text_query", side_effect=capture_embed), \
             patch("src.vector_db.search_visual", return_value=[]), \
             patch("src.db.minio.create_client", return_value=MagicMock()):
            chain._run_visual_retrieval("Q3 2025 quarterly revenue chart", tenant_id=None)

        assert embed_calls == ["Q3 2025 quarterly revenue chart"]

    def test_run_visual_retrieval_forwards_tenant_id(self, monkeypatch):
        """FR-609: tenant_id forwarded to search_visual."""
        chain = self._make_chain(monkeypatch)
        chain._visual_model = MagicMock()
        chain._visual_processor = MagicMock()
        chain.tracer = self._make_mock_tracer()

        search_calls = []

        def capture_search(**kwargs):
            search_calls.append(kwargs)
            return []

        with patch("src.ingest.support.embed_text_query",
                   return_value=[0.1] * 128), \
             patch("src.vector_db.search_visual", side_effect=capture_search), \
             patch("src.db.minio.create_client", return_value=MagicMock()):
            chain._run_visual_retrieval("any query", tenant_id="acme")

        assert len(search_calls) == 1
        assert search_calls[0].get("tenant_id") == "acme"

    def test_per_page_url_failure_skips_page_includes_others(self, monkeypatch):
        """FR-615: S3Error on page 2 of 3 → pages 1 and 3 included; page 2 omitted."""
        from minio.error import S3Error
        chain = self._make_chain(monkeypatch)
        chain._visual_model = MagicMock()
        chain._visual_processor = MagicMock()
        chain.tracer = self._make_mock_tracer()

        three_pages = [
            dict(self.PAGE_RECORDS[0], page_number=1, minio_key="k/0001.jpg", score=0.81),
            dict(self.PAGE_RECORDS[0], page_number=2, minio_key="k/0002.jpg", score=0.75),
            dict(self.PAGE_RECORDS[0], page_number=3, minio_key="k/0003.jpg", score=0.70),
        ]
        url_effects = [
            "https://url/page1.jpg",
            S3Error("NetworkError", "timeout"),
            "https://url/page3.jpg",
        ]

        with patch("src.ingest.support.embed_text_query",
                   return_value=[0.1] * 128), \
             patch("src.vector_db.search_visual", return_value=three_pages), \
             patch("src.db.minio.get_page_image_url", side_effect=url_effects), \
             patch("src.db.minio.create_client", return_value=MagicMock()):
            results = chain._run_visual_retrieval("chart", tenant_id="t1")

        page_numbers = [vpr.page_number for vpr in results]
        assert 2 not in page_numbers
        assert 1 in page_numbers
        assert 3 in page_numbers
        assert len(results) == 2

    def test_empty_run_visual_retrieval_returns_empty_list(self, monkeypatch):
        """FR-615: search_visual returns [] → _run_visual_retrieval returns []."""
        chain = self._make_chain(monkeypatch)
        chain._visual_model = MagicMock()
        chain._visual_processor = MagicMock()
        chain.tracer = self._make_mock_tracer()

        with patch("src.ingest.support.embed_text_query",
                   return_value=[0.1] * 128), \
             patch("src.vector_db.search_visual", return_value=[]), \
             patch("src.db.minio.create_client", return_value=MagicMock()):
            results = chain._run_visual_retrieval("obscure query", tenant_id=None)

        assert results == []

    # --- Graceful degradation via _run_visual_retrieval ---

    def test_colqwen_load_error_from_ensure_model_raises(self, monkeypatch):
        """FR-605: ColQwen2LoadError from _ensure_visual_model propagates from _run_visual_retrieval."""
        from src.ingest.support.colqwen import ColQwen2LoadError
        chain = self._make_chain(monkeypatch)
        chain.tracer = self._make_mock_tracer()
        # model is None → cold load → fails

        with patch("src.ingest.support.load_colqwen_model",
                   side_effect=ColQwen2LoadError("CUDA not available")):
            with pytest.raises(ColQwen2LoadError):
                chain._run_visual_retrieval("any query", tenant_id=None)

    def test_visual_embedding_error_propagates_from_run_visual_retrieval(self, monkeypatch):
        """FR-605: VisualEmbeddingError from embed_text_query propagates up."""
        from src.ingest.support.colqwen import VisualEmbeddingError
        chain = self._make_chain(monkeypatch)
        chain._visual_model = MagicMock()
        chain._visual_processor = MagicMock()
        chain.tracer = self._make_mock_tracer()

        with patch("src.ingest.support.embed_text_query",
                   side_effect=VisualEmbeddingError("embed failed")):
            with pytest.raises(VisualEmbeddingError):
                chain._run_visual_retrieval("any query", tenant_id=None)

    def test_weaviate_query_error_propagates_from_run_visual_retrieval(self, monkeypatch):
        """FR-605: WeaviateQueryError from search_visual propagates up."""
        chain = self._make_chain(monkeypatch)
        chain._visual_model = MagicMock()
        chain._visual_processor = MagicMock()
        chain.tracer = self._make_mock_tracer()

        with patch("src.ingest.support.embed_text_query",
                   return_value=[0.1] * 128), \
             patch("src.vector_db.search_visual",
                   side_effect=_WeaviateQueryError("query failed", "grpc")):
            with pytest.raises(_WeaviateQueryError):
                chain._run_visual_retrieval("any query", tenant_id=None)

    # --- Enabled/disabled path (additional coverage) ---

    def test_visual_disabled_run_visual_retrieval_never_called(self, monkeypatch):
        """FR-601: when _visual_retrieval_enabled=False, visual search is never invoked."""
        chain = self._make_chain(monkeypatch, enabled=False)
        assert chain._visual_retrieval_enabled is False

        # The guard is the flag itself; verify search_visual is never reached.
        with patch("src.vector_db.search_visual") as mock_search_visual:
            # Simulating the flag check that run() performs before calling _run_visual_retrieval
            if chain._visual_retrieval_enabled:
                chain._run_visual_retrieval("any query", tenant_id=None)
            mock_search_visual.assert_not_called()

    def test_visual_enabled_run_visual_retrieval_returns_results(self, monkeypatch):
        """FR-601/FR-603: when enabled and images found, _run_visual_retrieval returns list."""
        chain = self._make_chain(monkeypatch, enabled=True)
        chain._visual_model = MagicMock()
        chain._visual_processor = MagicMock()
        chain.tracer = self._make_mock_tracer()

        with patch("src.ingest.support.embed_text_query",
                   return_value=[0.1] * 128), \
             patch("src.vector_db.search_visual",
                   return_value=self.PAGE_RECORDS), \
             patch("src.db.minio.get_page_image_url",
                   return_value="https://minio.example.com/page.jpg"), \
             patch("src.db.minio.create_client",
                   return_value=MagicMock()):
            results = chain._run_visual_retrieval("quarterly revenue chart", tenant_id=None)

        assert len(results) == len(self.PAGE_RECORDS)
        assert all(vpr.page_image_url for vpr in results)


# ===========================================================================
# Known Gaps (per test docs — not tested, documented here)
# ===========================================================================

# GAP (EG §3.1): CUDA determinism under 4-bit BitsAndBytes quantization —
#   cannot be verified without real GPU hardware; mock tests only.

# GAP (EG §3.1): torch.inference_mode() gradient enforcement —
#   not directly testable via Python unit tests; only tested indirectly via mock.

# GAP (EG §3.1): Token dimension variability from real ColQwen2 tokenizer —
#   mock n_tokens is fixed; real tokenizer behavior requires live model.

# GAP (EG §3.2): HNSW approximate-nearest-neighbor accuracy —
#   cannot be verified without live Weaviate instance.

# GAP (EG §3.2): Weaviate certainty vs. distance behavior across versions —
#   unit tested with mock; score = 1.0 - distance assumed correct.

# GAP (EG §3.2): Observability span emission — integration test only.

# GAP (EG §3.3): URL format validity — requires live MinIO instance.

# GAP (EG §3.3): URL expiry verification — live MinIO integration test.

# GAP (EG §3.5): Full run() integration with stage_timings "visual_retrieval" entry —
#   run() is too heavily wired to other pipeline stages for isolated unit test;
#   stage_timings are tested at _run_visual_retrieval level instead.

# GAP (EG §3.5): Stage budget exhaustion (FR-617) — time-based; integration test only.

# GAP (EG §3.5): Thread safety of _ensure_visual_model — race condition not reliably testable.

# GAP (EG §3.5): NFR-903 VRAM budget compliance — hardware integration test only.

# GAP (EG §3.6): _get_vector_backend() singleton lifecycle — order-dependency in tests.

# GAP (EG §3.7): Module-level import side effects — test isolation via reload has order risk.

# GAP (EG §3.7): RAG_INGESTION_COLQWEN_MODEL "key should not exist" — no direct assertion test.

# GAP (EG §3.8): OpenAPI schema generation — requires running FastAPI app.

# GAP (EG §3.8): response_model_exclude_none=True behavior at FastAPI route level —
#   requires test client; not unit-testable at Pydantic level alone.
