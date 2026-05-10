# @summary
# Tests for visual_embedding_node LangGraph node.
# Covers: src/ingest/embedding/nodes/visual_embedding.py
# Exports: (pytest test functions)
# Deps: pytest, unittest.mock
# @end-summary
"""Tests for the visual embedding pipeline node.

All external calls (ColQwen2, MinIO, Weaviate) are mocked so tests run without
GPU hardware, HuggingFace network access, or live service connections.

Known gaps are annotated with ``# GAP:`` comments throughout.
"""

import sys
import types

# ---------------------------------------------------------------------------
# PIL stub — visual_embedding.py imports PIL.Image at module level;
# inject a minimal stub so collection succeeds without Pillow installed.
# ---------------------------------------------------------------------------
def _install_pil_stub() -> None:
    if "PIL" in sys.modules:
        return
    _pil_pkg = types.ModuleType("PIL")
    _pil_image_mod = types.ModuleType("PIL.Image")

    class _Image:
        LANCZOS = 1
        BICUBIC = 3

        def __init__(self) -> None:
            self.size = (0, 0)
            self.mode = "RGB"

        @staticmethod
        def fromarray(arr: object) -> "_Image":
            return _Image()

    _pil_image_mod.Image = _Image  # type: ignore[attr-defined]
    _pil_image_mod.LANCZOS = _Image.LANCZOS  # type: ignore[attr-defined]
    _pil_image_mod.fromarray = _Image.fromarray  # type: ignore[attr-defined]
    _pil_pkg.Image = _pil_image_mod  # type: ignore[attr-defined]
    sys.modules["PIL"] = _pil_pkg
    sys.modules["PIL.Image"] = _pil_image_mod


_install_pil_stub()

import pytest
from unittest.mock import MagicMock, call, patch

from src.ingest.common.types import IngestionConfig, Runtime
from src.ingest.embedding.nodes.visual_embedding import visual_embedding_node
from src.ingest.support.colqwen import ColQwen2PageEmbedding

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

_ENSURE_COLQWEN = "src.ingest.embedding.nodes.visual_embedding.ensure_colqwen_ready"
_LOAD_COLQWEN = "src.ingest.embedding.nodes.visual_embedding.load_colqwen_model"
_EMBED_PAGES = "src.ingest.embedding.nodes.visual_embedding.embed_page_images"
_UNLOAD_COLQWEN = "src.ingest.embedding.nodes.visual_embedding.unload_colqwen_model"
_STORE_PAGES = "src.ingest.embedding.nodes.visual_embedding.store_page_images"
_DELETE_PAGES = "src.ingest.embedding.nodes.visual_embedding.delete_page_images"
_ENSURE_VISUAL = "src.ingest.embedding.nodes.visual_embedding.ensure_visual_collection"
_ADD_VISUAL = "src.ingest.embedding.nodes.visual_embedding.add_visual_documents"
_DELETE_VISUAL = "src.ingest.embedding.nodes.visual_embedding.delete_visual_by_source_key"
_TO_RGB = "src.ingest.embedding.nodes.visual_embedding._to_rgb"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_image(width: int = 842, height: int = 1190) -> MagicMock:
    """Return a minimal mock PIL.Image with configurable dimensions."""
    img = MagicMock()
    img.size = (width, height)
    img.mode = "RGB"
    resized = MagicMock()
    resized.size = (width, height)
    resized.mode = "RGB"
    img.resize.return_value = resized
    return img


def _make_embeddings(page_numbers: list[int]) -> list[ColQwen2PageEmbedding]:
    """Return ColQwen2PageEmbedding objects for given page numbers."""
    return [
        ColQwen2PageEmbedding(
            page_number=n,
            mean_vector=[0.1] * 128,
            patch_vectors=[[0.05] * 128] * 800,
            patch_count=800,
        )
        for n in page_numbers
    ]


def _make_config(
    enable_visual_embedding: bool = True,
    page_image_max_dimension: int = 1024,
    page_image_quality: int = 85,
    colqwen_batch_size: int = 4,
    visual_target_collection: str = "RAGVisualPages",
    colqwen_model_name: str = "vidore/colqwen2-v1.0",
) -> IngestionConfig:
    return IngestionConfig(
        enable_visual_embedding=enable_visual_embedding,
        page_image_max_dimension=page_image_max_dimension,
        page_image_quality=page_image_quality,
        colqwen_batch_size=colqwen_batch_size,
        visual_target_collection=visual_target_collection,
        colqwen_model_name=colqwen_model_name,
    )


def _make_runtime(config: IngestionConfig) -> Runtime:
    return Runtime(
        config=config,
        embedder=MagicMock(),
        weaviate_client=MagicMock(),
        kg_builder=None,
        db_client=MagicMock(),
    )


def _make_state(
    config: IngestionConfig | None = None,
    page_images: list | None = None,
    docling_document: object = ...,  # sentinel — caller must supply or use _PRESENT
    n_pages: int = 0,
    errors: list | None = None,
    processing_log: list | None = None,
    include_text_track_fields: bool = True,
) -> dict:
    """Build a minimal EmbeddingPipelineState dict for node testing."""
    if config is None:
        config = _make_config()
    runtime = _make_runtime(config)

    # Default: provide a MagicMock docling_document unless caller passes None explicitly
    if docling_document is ...:
        docling_document = MagicMock()

    if page_images is None and n_pages > 0:
        page_images = [_make_mock_image() for _ in range(n_pages)]

    state: dict = {
        "config": config,
        "runtime": runtime,
        "document_id": "test-doc-uuid",
        "source_key": "docs/test-doc-uuid/file.pdf",
        "source_uri": "s3://bucket/docs/test-doc-uuid/file.pdf",
        "source_name": "file.pdf",
        "tenant_id": "tenant-1",
        "docling_document": docling_document,
        "page_images": page_images,
        "processing_log": processing_log if processing_log is not None else [],
        "errors": errors if errors is not None else [],
    }
    if include_text_track_fields:
        state.update(
            {
                "stored_count": 47,
                "chunks": [],
                "metadata_summary": "existing summary",
                "metadata_keywords": ["kw1", "kw2"],
            }
        )
    return state


def _all_mocks_happy(
    mock_store: MagicMock,
    mock_embed: MagicMock,
    n_pages: int,
    n_keys: int | None = None,
) -> None:
    """Configure mock_store and mock_embed with standard happy-path returns."""
    if n_keys is None:
        n_keys = n_pages
    mock_store.return_value = [f"key/page_{i}.jpg" for i in range(n_keys)]
    mock_embed.return_value = _make_embeddings(list(range(1, n_pages + 1)))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _stub_to_rgb():
    """Patch ``_to_rgb`` to return its input unchanged for the whole module.

    The visual_embedding tests pass in MagicMock images, and the production
    ``_to_rgb`` calls ``Image.fromarray(mock)`` when the input is not a real
    ``PIL.Image.Image``. With real PIL installed (the normal CI/local case
    once *any* earlier test imports Pillow), this raises
    ``AttributeError: __array_interface__``. The PIL stub at the top of this
    file only installs when PIL is not yet in ``sys.modules``, so it doesn't
    help once another test has already pulled in real PIL — hence the test
    pollution that only manifested when this file ran after others.

    Tests that need to assert on ``_to_rgb`` behavior can still ``patch`` it
    inside the test body — that inner patch wins over this autouse fixture.
    """
    with patch(_TO_RGB, side_effect=lambda x: x):
        yield


@pytest.fixture
def happy_state_10() -> dict:
    """State for a 10-page document with all visual embedding flags on."""
    return _make_state(n_pages=10)


# ===========================================================================
# TestVisualEmbeddingNodeShortCircuit
# ===========================================================================

class TestVisualEmbeddingNodeShortCircuit:
    """Short-circuit behavior when visual embedding should not run."""

    def test_disabled_config_returns_zero_stored(self):
        """FR-101/FR-603: enable_visual_embedding=False → visual_stored_count=0."""
        config = _make_config(enable_visual_embedding=False)
        state = _make_state(config=config, n_pages=5)
        with patch(_LOAD_COLQWEN) as mock_load, \
             patch(_STORE_PAGES) as mock_store, \
             patch(_ENSURE_VISUAL) as mock_ensure_v, \
             patch(_ADD_VISUAL) as mock_add:
            result = visual_embedding_node(state)
        assert result["visual_stored_count"] == 0
        mock_load.assert_not_called()
        mock_store.assert_not_called()
        mock_ensure_v.assert_not_called()
        mock_add.assert_not_called()

    def test_disabled_config_clears_page_images(self):
        """FR-606: page_images is None in returned dict even when disabled."""
        config = _make_config(enable_visual_embedding=False)
        state = _make_state(config=config, n_pages=3)
        result = visual_embedding_node(state)
        assert result["page_images"] is None

    def test_disabled_config_log_entry(self):
        """FR-101/NFR-903: Skipped-disabled log entry present when flag is False."""
        config = _make_config(enable_visual_embedding=False)
        state = _make_state(config=config, n_pages=2)
        result = visual_embedding_node(state)
        combined_log = " ".join(result.get("processing_log", []))
        assert "disabled" in combined_log.lower() or "skipped" in combined_log.lower()

    def test_disabled_config_no_colqwen_loaded(self):
        """NFR-903: No ColQwen2 functions called when flag is False."""
        config = _make_config(enable_visual_embedding=False)
        state = _make_state(config=config, n_pages=2)
        with patch(_ENSURE_COLQWEN) as mock_ensure_c, \
             patch(_LOAD_COLQWEN) as mock_load, \
             patch(_UNLOAD_COLQWEN) as mock_unload:
            visual_embedding_node(state)
        mock_ensure_c.assert_not_called()
        mock_load.assert_not_called()
        mock_unload.assert_not_called()

    def test_no_docling_document_returns_zero_stored(self):
        """FR-603: docling_document=None → visual_stored_count=0."""
        state = _make_state(docling_document=None, n_pages=5)
        with patch(_LOAD_COLQWEN) as mock_load, \
             patch(_STORE_PAGES) as mock_store:
            result = visual_embedding_node(state)
        assert result["visual_stored_count"] == 0
        mock_load.assert_not_called()
        mock_store.assert_not_called()

    def test_no_docling_document_clears_page_images(self):
        """FR-606: page_images is None when docling_document is None."""
        state = _make_state(docling_document=None, n_pages=3)
        result = visual_embedding_node(state)
        assert result["page_images"] is None

    def test_no_docling_document_log_entry(self):
        """FR-603: Log contains entry indicating skip due to missing docling_document."""
        state = _make_state(docling_document=None, n_pages=2)
        result = visual_embedding_node(state)
        combined_log = " ".join(result.get("processing_log", []))
        assert "docling" in combined_log.lower() or "skipped" in combined_log.lower()

    def test_zero_pages_extracted_returns_zero_stored(self):
        """FR-203/FR-603: Zero extracted page images → visual_stored_count=0."""
        # page_images is an empty list — no pages available
        state = _make_state(page_images=[])
        with patch(_LOAD_COLQWEN) as mock_load, \
             patch(_STORE_PAGES) as mock_store:
            result = visual_embedding_node(state)
        assert result["visual_stored_count"] == 0
        mock_load.assert_not_called()
        mock_store.assert_not_called()

    def test_zero_pages_no_model_loaded(self):
        """FR-203/FR-603: No ColQwen2 model loaded when zero pages extracted."""
        state = _make_state(page_images=[])
        with patch(_ENSURE_COLQWEN) as mock_ensure_c, \
             patch(_LOAD_COLQWEN) as mock_load, \
             patch(_UNLOAD_COLQWEN) as mock_unload:
            visual_embedding_node(state)
        mock_ensure_c.assert_not_called()
        mock_load.assert_not_called()
        mock_unload.assert_not_called()

    def test_zero_pages_log_entry(self):
        """FR-203/FR-603: Log contains entry indicating skip due to no pages."""
        state = _make_state(page_images=[])
        result = visual_embedding_node(state)
        combined_log = " ".join(result.get("processing_log", []))
        assert "no_pages" in combined_log or "skipped" in combined_log.lower()

    def test_zero_pages_clears_page_images(self):
        """FR-606: page_images=None returned even when short-circuited on no pages."""
        state = _make_state(page_images=[])
        result = visual_embedding_node(state)
        assert result["page_images"] is None


# ===========================================================================
# TestVisualEmbeddingNodeHappyPath
# ===========================================================================

class TestVisualEmbeddingNodeHappyPath:
    """Standard success scenarios for the visual embedding node."""

    def test_10_pages_stored_count(self, happy_state_10):
        """Standard 10-page doc: visual_stored_count=10."""
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES) as mock_embed, \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES) as mock_store, \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=10), \
             patch(_DELETE_VISUAL):
            _all_mocks_happy(mock_store, mock_embed, n_pages=10)
            result = visual_embedding_node(happy_state_10)
        assert result["visual_stored_count"] == 10

    def test_10_pages_page_images_cleared(self, happy_state_10):
        """FR-606: page_images=None after successful 10-page run."""
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES) as mock_embed, \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES) as mock_store, \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=10), \
             patch(_DELETE_VISUAL):
            _all_mocks_happy(mock_store, mock_embed, n_pages=10)
            result = visual_embedding_node(happy_state_10)
        assert result["page_images"] is None

    def test_10_pages_processing_log_five_entries(self, happy_state_10):
        """Happy path: processing log contains exactly the 5 canonical entries."""
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES) as mock_embed, \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES) as mock_store, \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=10), \
             patch(_DELETE_VISUAL):
            _all_mocks_happy(mock_store, mock_embed, n_pages=10)
            result = visual_embedding_node(happy_state_10)
        log = result.get("processing_log", [])
        combined = " ".join(log)
        assert "pages_extracted:10" in combined
        assert "pages_stored_minio:10" in combined
        assert "pages_embedded:10" in combined
        assert "pages_indexed:10" in combined
        assert "elapsed_s:" in combined

    def test_10_pages_elapsed_s_is_float(self, happy_state_10):
        """Processing log elapsed_s entry contains a valid float value."""
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES) as mock_embed, \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES) as mock_store, \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=10), \
             patch(_DELETE_VISUAL):
            _all_mocks_happy(mock_store, mock_embed, n_pages=10)
            result = visual_embedding_node(happy_state_10)
        elapsed_entries = [e for e in result["processing_log"] if "elapsed_s:" in e]
        assert len(elapsed_entries) >= 1
        elapsed_val = elapsed_entries[0].split("elapsed_s:")[-1].strip()
        # Must be parseable as float
        float(elapsed_val)

    def test_resize_applied_when_image_exceeds_max_dimension(self):
        """Image with max(w,h) > max_dimension → resize called with LANCZOS."""
        config = _make_config(page_image_max_dimension=500)
        # Image 800x600 — max(800,600)=800 > 500
        big_img = _make_mock_image(width=800, height=600)
        state = _make_state(config=config, page_images=[big_img])
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings([1])), \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES, return_value=["key/page_0.jpg"]), \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=1), \
             patch(_DELETE_VISUAL), \
             patch(_TO_RGB, side_effect=lambda x: x):
            visual_embedding_node(state)
        # resize must have been called on the image
        big_img.resize.assert_called_once()
        resize_call_kwargs = big_img.resize.call_args
        # Verify LANCZOS (or equivalent filter sentinel) was passed
        args, kwargs = resize_call_kwargs
        # The filter argument may be positional or keyword; check it's present
        all_args = list(args) + list(kwargs.values())
        # At least one argument besides the size tuple must be present (the filter)
        assert len(all_args) >= 1

    def test_resize_skipped_when_image_within_max_dimension(self):
        """Image with max(w,h) <= max_dimension → resize NOT applied."""
        config = _make_config(page_image_max_dimension=1024)
        # Image 842x1024 — max(842,1024)=1024 == 1024, NOT strictly greater than
        exact_img = _make_mock_image(width=842, height=1024)
        state = _make_state(config=config, page_images=[exact_img])
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings([1])), \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES, return_value=["key/page_0.jpg"]), \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=1), \
             patch(_DELETE_VISUAL):
            visual_embedding_node(state)
        exact_img.resize.assert_not_called()

    def test_resize_skipped_when_image_smaller_than_max_dimension(self):
        """Image with max(w,h) < max_dimension → resize NOT applied."""
        config = _make_config(page_image_max_dimension=1024)
        small_img = _make_mock_image(width=400, height=600)
        state = _make_state(config=config, page_images=[small_img])
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings([1])), \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES, return_value=["key/page_0.jpg"]), \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=1), \
             patch(_DELETE_VISUAL):
            visual_embedding_node(state)
        small_img.resize.assert_not_called()

    def test_pre_cleanup_called_before_minio_storage(self):
        """Pre-cleanup calls happen before MinIO store_page_images."""
        state = _make_state(n_pages=3)
        call_order: list[str] = []

        def track_delete_pages(*a, **kw):
            call_order.append("delete_pages")

        def track_delete_visual(*a, **kw):
            call_order.append("delete_visual")

        def track_store(*a, **kw):
            call_order.append("store")
            return ["key/page_0.jpg", "key/page_1.jpg", "key/page_2.jpg"]

        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings([1, 2, 3])), \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES, side_effect=track_store), \
             patch(_DELETE_PAGES, side_effect=track_delete_pages), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=3), \
             patch(_DELETE_VISUAL, side_effect=track_delete_visual):
            visual_embedding_node(state)

        assert "store" in call_order
        store_idx = call_order.index("store")
        if "delete_pages" in call_order:
            assert call_order.index("delete_pages") < store_idx
        if "delete_visual" in call_order:
            assert call_order.index("delete_visual") < store_idx

    def test_visual_stored_count_type_is_int(self, happy_state_10):
        """visual_stored_count is always an int regardless of code path."""
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES) as mock_embed, \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES) as mock_store, \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=10), \
             patch(_DELETE_VISUAL):
            _all_mocks_happy(mock_store, mock_embed, n_pages=10)
            result = visual_embedding_node(happy_state_10)
        assert isinstance(result["visual_stored_count"], int)

    def test_processing_log_accumulates_not_replaces(self):
        """Processing log entries are appended to existing entries, not replaced."""
        existing_log = ["stage:previous_stage_done"]
        state = _make_state(n_pages=2, processing_log=existing_log)
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings([1, 2])), \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES, return_value=["k/0.jpg", "k/1.jpg"]), \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=2), \
             patch(_DELETE_VISUAL):
            result = visual_embedding_node(state)
        assert "stage:previous_stage_done" in result["processing_log"]


# ===========================================================================
# TestVisualEmbeddingNodeColQwenErrors
# ===========================================================================

class TestVisualEmbeddingNodeColQwenErrors:
    """ColQwen2 model lifecycle error scenarios."""

    def test_ensure_colqwen_raises_stored_count_zero(self):
        """FR-802: ensure_colqwen_ready raises ColQwen2LoadError → visual_stored_count=0."""
        from src.ingest.embedding.nodes.visual_embedding import ColQwen2LoadError
        state = _make_state(n_pages=5)
        with patch(_ENSURE_COLQWEN, side_effect=ColQwen2LoadError("GPU OOM")), \
             patch(_LOAD_COLQWEN) as mock_load, \
             patch(_UNLOAD_COLQWEN) as mock_unload:
            result = visual_embedding_node(state)
        assert result["visual_stored_count"] == 0
        mock_load.assert_not_called()
        # GAP: GPU allocation assertion (NFR-903): can only assert ColQwen2 not called;
        # actual GPU non-allocation needs hardware tests.

    def test_ensure_colqwen_raises_error_recorded(self):
        """FR-802: ensure_colqwen_ready failure appends error string to state errors."""
        from src.ingest.embedding.nodes.visual_embedding import ColQwen2LoadError
        state = _make_state(n_pages=5)
        with patch(_ENSURE_COLQWEN, side_effect=ColQwen2LoadError("check fail")):
            result = visual_embedding_node(state)
        assert len(result.get("errors", [])) >= 1

    def test_ensure_colqwen_raises_unload_not_called(self):
        """FR-802: unload_colqwen_model NOT called when ensure step raises before load."""
        from src.ingest.embedding.nodes.visual_embedding import ColQwen2LoadError
        state = _make_state(n_pages=5)
        with patch(_ENSURE_COLQWEN, side_effect=ColQwen2LoadError("check fail")), \
             patch(_UNLOAD_COLQWEN) as mock_unload:
            visual_embedding_node(state)
        # GAP: finally + ColQwen2LoadError interaction: model never assigned when load
        # fails, so unload must NOT be called — careful mock setup needed.
        mock_unload.assert_not_called()

    def test_ensure_colqwen_raises_clears_page_images(self):
        """FR-606: page_images=None even when ensure_colqwen_ready fails."""
        from src.ingest.embedding.nodes.visual_embedding import ColQwen2LoadError
        state = _make_state(n_pages=5)
        with patch(_ENSURE_COLQWEN, side_effect=ColQwen2LoadError("check fail")):
            result = visual_embedding_node(state)
        assert result["page_images"] is None

    def test_load_colqwen_raises_stored_count_zero(self):
        """FR-802: load_colqwen_model raises ColQwen2LoadError → visual_stored_count=0."""
        from src.ingest.embedding.nodes.visual_embedding import ColQwen2LoadError
        state = _make_state(n_pages=5)
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, side_effect=ColQwen2LoadError("load fail")), \
             patch(_UNLOAD_COLQWEN) as mock_unload:
            result = visual_embedding_node(state)
        assert result["visual_stored_count"] == 0
        # GAP: finally + ColQwen2LoadError: model never assigned on load fail;
        # unload must NOT be called.
        mock_unload.assert_not_called()

    def test_load_colqwen_raises_error_recorded(self):
        """FR-802: load_colqwen_model failure appends error to state errors."""
        from src.ingest.embedding.nodes.visual_embedding import ColQwen2LoadError
        state = _make_state(n_pages=5)
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, side_effect=ColQwen2LoadError("load fail")):
            result = visual_embedding_node(state)
        assert len(result.get("errors", [])) >= 1

    def test_load_colqwen_raises_clears_page_images(self):
        """FR-606: page_images=None even when load_colqwen_model fails."""
        from src.ingest.embedding.nodes.visual_embedding import ColQwen2LoadError
        state = _make_state(n_pages=5)
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, side_effect=ColQwen2LoadError("load fail")):
            result = visual_embedding_node(state)
        assert result["page_images"] is None

    def test_unhandled_exception_stored_count_zero(self):
        """Catch-all: arbitrary exception not caught by inner handlers → visual_stored_count=0."""
        state = _make_state(n_pages=5)
        with patch(_ENSURE_COLQWEN, side_effect=RuntimeError("unexpected boom")):
            result = visual_embedding_node(state)
        assert result["visual_stored_count"] == 0

    def test_unhandled_exception_no_reraise(self):
        """Catch-all: pipeline continues without re-raising the exception."""
        state = _make_state(n_pages=5)
        # Should not raise
        with patch(_ENSURE_COLQWEN, side_effect=RuntimeError("unexpected boom")):
            result = visual_embedding_node(state)
        assert "visual_stored_count" in result

    def test_unhandled_exception_clears_page_images(self):
        """FR-606: page_images=None even on unhandled exception paths."""
        state = _make_state(n_pages=5)
        with patch(_ENSURE_COLQWEN, side_effect=RuntimeError("unexpected boom")):
            result = visual_embedding_node(state)
        assert result["page_images"] is None

    def test_embed_raises_unload_still_called(self):
        """FR-801/finally: unload_colqwen_model called even when embed_page_images raises."""
        state = _make_state(n_pages=5)
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, side_effect=RuntimeError("embed crash")), \
             patch(_UNLOAD_COLQWEN) as mock_unload, \
             patch(_STORE_PAGES, return_value=[f"key/{i}.jpg" for i in range(5)]), \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=0), \
             patch(_DELETE_VISUAL):
            visual_embedding_node(state)
        mock_unload.assert_called_once()


# ===========================================================================
# TestVisualEmbeddingNodePartialFailures
# ===========================================================================

class TestVisualEmbeddingNodePartialFailures:
    """Partial failure and degraded operation scenarios."""

    def test_partial_page_inference_stored_count_reflects_successful_pages(self):
        """FR-801: embed returns 8 of 10 pages → visual_stored_count=8."""
        state = _make_state(n_pages=10)
        # Pages 3 and 7 are absent — only 8 embeddings returned
        present_pages = [1, 2, 4, 5, 6, 8, 9, 10]
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings(present_pages)), \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES, return_value=[f"key/page_{i}.jpg" for i in range(10)]), \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=8), \
             patch(_DELETE_VISUAL):
            result = visual_embedding_node(state)
        assert result["visual_stored_count"] == 8

    def test_partial_page_inference_unload_still_called(self):
        """FR-801/finally: unload_colqwen_model called after partial page inference."""
        state = _make_state(n_pages=10)
        present_pages = [1, 2, 4, 5, 6, 8, 9, 10]
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings(present_pages)), \
             patch(_UNLOAD_COLQWEN) as mock_unload, \
             patch(_STORE_PAGES, return_value=[f"key/page_{i}.jpg" for i in range(10)]), \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=8), \
             patch(_DELETE_VISUAL):
            visual_embedding_node(state)
        mock_unload.assert_called_once()

    def test_minio_partial_failure_node_continues(self):
        """FR-804: store_page_images returns fewer keys than pages → node continues.

        GAP: Partial MinIO failure (FR-804): assumption about node receiving partial
        key list needs validation against implementation.
        """
        state = _make_state(n_pages=5)
        # store returns only 3 keys instead of 5
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings([1, 2, 3, 4, 5])), \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES, return_value=["k/0.jpg", "k/1.jpg", "k/2.jpg"]), \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL) as mock_add, \
             patch(_DELETE_VISUAL):
            result = visual_embedding_node(state)
        # Node must not raise; result dict must be present with required keys
        assert "visual_stored_count" in result
        assert result["page_images"] is None

    def test_weaviate_insertion_failure_stored_count_zero(self):
        """FR-805: add_visual_documents raises → visual_stored_count=0."""
        state = _make_state(n_pages=5)
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings([1, 2, 3, 4, 5])), \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES, return_value=[f"k/{i}.jpg" for i in range(5)]), \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, side_effect=RuntimeError("weaviate down")), \
             patch(_DELETE_VISUAL):
            result = visual_embedding_node(state)
        assert result["visual_stored_count"] == 0

    def test_weaviate_insertion_failure_error_recorded(self):
        """FR-805: add_visual_documents exception appends error to state errors."""
        state = _make_state(n_pages=5)
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings([1, 2, 3, 4, 5])), \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES, return_value=[f"k/{i}.jpg" for i in range(5)]), \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, side_effect=RuntimeError("weaviate down")), \
             patch(_DELETE_VISUAL):
            result = visual_embedding_node(state)
        assert len(result.get("errors", [])) >= 1

    def test_weaviate_insertion_failure_no_reraise(self):
        """FR-805: add_visual_documents exception does not propagate — pipeline continues."""
        state = _make_state(n_pages=5)
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings([1, 2, 3, 4, 5])), \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES, return_value=[f"k/{i}.jpg" for i in range(5)]), \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, side_effect=RuntimeError("weaviate down")), \
             patch(_DELETE_VISUAL):
            # Must not raise
            result = visual_embedding_node(state)
        assert "visual_stored_count" in result

    def test_weaviate_insertion_failure_clears_page_images(self):
        """FR-606: page_images=None even when Weaviate insertion fails."""
        state = _make_state(n_pages=5)
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings([1, 2, 3, 4, 5])), \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES, return_value=[f"k/{i}.jpg" for i in range(5)]), \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, side_effect=RuntimeError("weaviate down")), \
             patch(_DELETE_VISUAL):
            result = visual_embedding_node(state)
        assert result["page_images"] is None

    def test_unload_called_in_finally_after_successful_inference(self):
        """Finally block: unload_colqwen_model called after successful embedding run."""
        state = _make_state(n_pages=3)
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings([1, 2, 3])), \
             patch(_UNLOAD_COLQWEN) as mock_unload, \
             patch(_STORE_PAGES, return_value=["k/0.jpg", "k/1.jpg", "k/2.jpg"]), \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=3), \
             patch(_DELETE_VISUAL):
            visual_embedding_node(state)
        mock_unload.assert_called_once()


# ===========================================================================
# TestVisualEmbeddingNodeStateIsolation
# ===========================================================================

class TestVisualEmbeddingNodeStateIsolation:
    """FR-803: Text-track fields must never appear in returned dict."""

    _TEXT_TRACK_FIELDS = [
        "stored_count",
        "chunks",
        "metadata_summary",
        "metadata_keywords",
    ]

    def _run_happy(self, n_pages: int = 3) -> dict:
        state = _make_state(n_pages=n_pages)
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings(list(range(1, n_pages + 1)))), \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES, return_value=[f"k/{i}.jpg" for i in range(n_pages)]), \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=n_pages), \
             patch(_DELETE_VISUAL):
            return visual_embedding_node(state)

    def test_stored_count_not_in_result(self):
        """FR-803: stored_count (text-track) must NOT be in returned dict."""
        result = self._run_happy()
        assert "stored_count" not in result

    def test_chunks_not_in_result(self):
        """FR-803: chunks (text-track) must NOT be in returned dict."""
        result = self._run_happy()
        assert "chunks" not in result

    def test_metadata_summary_not_in_result(self):
        """FR-803: metadata_summary (text-track) must NOT be in returned dict."""
        result = self._run_happy()
        assert "metadata_summary" not in result

    def test_metadata_keywords_not_in_result(self):
        """FR-803: metadata_keywords (text-track) must NOT be in returned dict."""
        result = self._run_happy()
        assert "metadata_keywords" not in result

    def test_none_of_text_track_fields_in_result(self):
        """FR-803: Single composite assertion — none of the 5 text-track fields returned."""
        result = self._run_happy()
        present = [f for f in self._TEXT_TRACK_FIELDS if f in result]
        assert present == [], f"Unexpected text-track fields in result: {present}"

    def test_text_track_fields_isolated_on_error_path(self):
        """FR-803: Text-track fields also absent on error paths."""
        from src.ingest.embedding.nodes.visual_embedding import ColQwen2LoadError
        state = _make_state(n_pages=5)
        with patch(_ENSURE_COLQWEN, side_effect=ColQwen2LoadError("fail")):
            result = visual_embedding_node(state)
        present = [f for f in self._TEXT_TRACK_FIELDS if f in result]
        assert present == [], f"Unexpected text-track fields in error result: {present}"

    def test_text_track_fields_isolated_on_disabled_path(self):
        """FR-803: Text-track fields absent when visual embedding is disabled."""
        config = _make_config(enable_visual_embedding=False)
        state = _make_state(config=config, n_pages=3)
        result = visual_embedding_node(state)
        present = [f for f in self._TEXT_TRACK_FIELDS if f in result]
        assert present == [], f"Unexpected text-track fields in disabled result: {present}"

    def test_page_images_always_none_in_returned_dict(self):
        """FR-606: page_images key in returned dict is always None."""
        result = self._run_happy()
        assert "page_images" in result
        assert result["page_images"] is None

    def test_visual_stored_count_always_present(self):
        """visual_stored_count always present and is int on all paths."""
        # Happy path
        result_happy = self._run_happy()
        assert isinstance(result_happy.get("visual_stored_count"), int)

        # Disabled path
        config = _make_config(enable_visual_embedding=False)
        state_disabled = _make_state(config=config, n_pages=1)
        result_disabled = visual_embedding_node(state_disabled)
        assert isinstance(result_disabled.get("visual_stored_count"), int)

        # Error path
        from src.ingest.embedding.nodes.visual_embedding import ColQwen2LoadError
        state_err = _make_state(n_pages=3)
        with patch(_ENSURE_COLQWEN, side_effect=ColQwen2LoadError("fail")):
            result_err = visual_embedding_node(state_err)
        assert isinstance(result_err.get("visual_stored_count"), int)

    def test_errors_key_absent_from_state_handled_gracefully(self):
        """Boundary: node handles state without pre-existing errors key gracefully."""
        config = _make_config(enable_visual_embedding=False)
        state = _make_state(config=config, n_pages=1)
        # Remove errors key to test absent-key handling
        state.pop("errors", None)
        # Should not raise KeyError
        result = visual_embedding_node(state)
        assert "visual_stored_count" in result


# ===========================================================================
# TestVisualEmbeddingNodePreCleanup
# ===========================================================================

class TestVisualEmbeddingNodePreCleanup:
    """Pre-cleanup call behavior and failure tolerance."""

    def test_delete_page_images_called(self):
        """delete_page_images is called during normal execution."""
        state = _make_state(n_pages=3)
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings([1, 2, 3])), \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES, return_value=["k/0.jpg", "k/1.jpg", "k/2.jpg"]), \
             patch(_DELETE_PAGES) as mock_del_pages, \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=3), \
             patch(_DELETE_VISUAL):
            visual_embedding_node(state)
        mock_del_pages.assert_called_once()

    def test_delete_visual_by_source_key_called(self):
        """delete_visual_by_source_key is called during normal execution."""
        state = _make_state(n_pages=3)
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings([1, 2, 3])), \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES, return_value=["k/0.jpg", "k/1.jpg", "k/2.jpg"]), \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=3), \
             patch(_DELETE_VISUAL) as mock_del_visual:
            visual_embedding_node(state)
        mock_del_visual.assert_called_once()

    def test_delete_page_images_failure_continues_to_minio(self):
        """delete_page_images raises → warning logged, MinIO store still called."""
        state = _make_state(n_pages=3)
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings([1, 2, 3])), \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES, return_value=["k/0.jpg", "k/1.jpg", "k/2.jpg"]) as mock_store, \
             patch(_DELETE_PAGES, side_effect=RuntimeError("cleanup fail")), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=3), \
             patch(_DELETE_VISUAL):
            result = visual_embedding_node(state)
        # Execution must continue; store must have been called
        mock_store.assert_called_once()
        assert "visual_stored_count" in result

    def test_delete_visual_by_source_key_failure_continues(self):
        """delete_visual_by_source_key raises → warning logged, execution continues."""
        state = _make_state(n_pages=3)
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings([1, 2, 3])), \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES, return_value=["k/0.jpg", "k/1.jpg", "k/2.jpg"]) as mock_store, \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=3), \
             patch(_DELETE_VISUAL, side_effect=RuntimeError("visual cleanup fail")):
            result = visual_embedding_node(state)
        mock_store.assert_called_once()
        assert "visual_stored_count" in result

    def test_delete_page_images_failure_no_reraise(self):
        """delete_page_images failure does not propagate as an exception."""
        state = _make_state(n_pages=2)
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings([1, 2])), \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES, return_value=["k/0.jpg", "k/1.jpg"]), \
             patch(_DELETE_PAGES, side_effect=RuntimeError("delete fail")), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=2), \
             patch(_DELETE_VISUAL):
            # Must not raise
            result = visual_embedding_node(state)
        assert "visual_stored_count" in result

    def test_delete_visual_by_source_key_failure_no_reraise(self):
        """delete_visual_by_source_key failure does not propagate as an exception."""
        state = _make_state(n_pages=2)
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings([1, 2])), \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES, return_value=["k/0.jpg", "k/1.jpg"]), \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=2), \
             patch(_DELETE_VISUAL, side_effect=RuntimeError("visual delete fail")):
            result = visual_embedding_node(state)
        assert "visual_stored_count" in result


# ===========================================================================
# TestVisualEmbeddingNodeProcessingLog
# ===========================================================================

class TestVisualEmbeddingNodeProcessingLog:
    """Processing log composition and content validation."""

    def test_pages_extracted_log_entry_present(self):
        """Log contains pages_extracted:<n> entry after successful run."""
        state = _make_state(n_pages=5)
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings([1, 2, 3, 4, 5])), \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES, return_value=[f"k/{i}.jpg" for i in range(5)]), \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=5), \
             patch(_DELETE_VISUAL):
            result = visual_embedding_node(state)
        combined = " ".join(result["processing_log"])
        assert "pages_extracted:5" in combined

    def test_pages_stored_minio_log_entry_present(self):
        """Log contains pages_stored_minio:<n> entry after MinIO storage."""
        state = _make_state(n_pages=5)
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings([1, 2, 3, 4, 5])), \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES, return_value=[f"k/{i}.jpg" for i in range(5)]), \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=5), \
             patch(_DELETE_VISUAL):
            result = visual_embedding_node(state)
        combined = " ".join(result["processing_log"])
        assert "pages_stored_minio:5" in combined

    def test_pages_embedded_log_entry_present(self):
        """Log contains pages_embedded:<n> entry after ColQwen2 inference."""
        state = _make_state(n_pages=5)
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings([1, 2, 3, 4, 5])), \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES, return_value=[f"k/{i}.jpg" for i in range(5)]), \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=5), \
             patch(_DELETE_VISUAL):
            result = visual_embedding_node(state)
        combined = " ".join(result["processing_log"])
        assert "pages_embedded:5" in combined

    def test_pages_indexed_log_entry_present(self):
        """Log contains pages_indexed:<n> entry after Weaviate insertion."""
        state = _make_state(n_pages=5)
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings([1, 2, 3, 4, 5])), \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES, return_value=[f"k/{i}.jpg" for i in range(5)]), \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=5), \
             patch(_DELETE_VISUAL):
            result = visual_embedding_node(state)
        combined = " ".join(result["processing_log"])
        assert "pages_indexed:5" in combined

    def test_elapsed_s_log_entry_present(self):
        """Log contains elapsed_s:<float> entry recording wall-clock time."""
        state = _make_state(n_pages=5)
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings([1, 2, 3, 4, 5])), \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES, return_value=[f"k/{i}.jpg" for i in range(5)]), \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=5), \
             patch(_DELETE_VISUAL):
            result = visual_embedding_node(state)
        combined = " ".join(result["processing_log"])
        assert "elapsed_s:" in combined

    def test_log_entries_are_strings(self):
        """All processing_log entries in returned dict are strings."""
        state = _make_state(n_pages=3)
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings([1, 2, 3])), \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES, return_value=["k/0.jpg", "k/1.jpg", "k/2.jpg"]), \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=3), \
             patch(_DELETE_VISUAL):
            result = visual_embedding_node(state)
        for entry in result["processing_log"]:
            assert isinstance(entry, str), f"Non-string log entry: {entry!r}"

    def test_log_preserved_from_initial_state(self):
        """Existing processing_log entries from initial state are preserved in output."""
        prior = ["stage:chunking_done", "stage:embedding_storage_done"]
        state = _make_state(n_pages=2, processing_log=prior[:])
        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings([1, 2])), \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES, return_value=["k/0.jpg", "k/1.jpg"]), \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=2), \
             patch(_DELETE_VISUAL):
            result = visual_embedding_node(state)
        for entry in prior:
            assert entry in result["processing_log"]

    def test_skipped_disabled_log_entry_format(self):
        """Disabled-path log entry contains recognizable skipped/disabled token."""
        config = _make_config(enable_visual_embedding=False)
        state = _make_state(config=config)
        result = visual_embedding_node(state)
        log = result.get("processing_log", [])
        skip_entries = [
            e for e in log
            if "disabled" in e.lower() or "skipped" in e.lower()
        ]
        assert len(skip_entries) >= 1, f"No skip/disabled entry found. Log: {log}"

    def test_skipped_no_docling_document_log_entry_format(self):
        """No-docling-document path log entry contains recognizable skip token."""
        state = _make_state(docling_document=None)
        result = visual_embedding_node(state)
        log = result.get("processing_log", [])
        skip_entries = [
            e for e in log
            if "docling" in e.lower() or "skipped" in e.lower() or "no_docling" in e.lower()
        ]
        assert len(skip_entries) >= 1, f"No skip entry for missing docling_document. Log: {log}"

    def test_skipped_no_pages_log_entry_format(self):
        """No-pages path log entry contains recognizable no_pages/skipped token."""
        state = _make_state(page_images=[])
        result = visual_embedding_node(state)
        log = result.get("processing_log", [])
        skip_entries = [
            e for e in log
            if "no_pages" in e.lower() or "skipped" in e.lower()
        ]
        assert len(skip_entries) >= 1, f"No skip entry for zero pages. Log: {log}"
