"""Targeted coverage tests for src/ingest/embedding/nodes/visual_embedding.py.

Covers the specific uncovered lines:
- Lines 193-196: missing document_id → skip MinIO operations
- Line 257: no MinIO client → warning log
- Lines 320-321: VisualEmbeddingError during embed_page_images
- Lines 340-345: ensure_visual_collection exception branch
- Lines 383-384: no Weaviate client → skip visual indexing
- Lines 480-481: _extract_from_docling pages attr is None/falsy
- Lines 485-487: _extract_from_docling pages iteration exception
- Lines 491-509: _extract_from_docling per-page extraction
- Lines 523-528: _to_rgb conversion of non-PIL image
- Lines 557-558: _resize_page_images resize exception
"""

from __future__ import annotations

import sys
import types

# ---------------------------------------------------------------------------
# PIL stub — same approach as existing test_visual_embedding_node.py
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
            self.size = (100, 200)
            self.mode = "RGB"

        @staticmethod
        def fromarray(arr: object) -> "_Image":
            obj = _Image()
            return obj

        def convert(self, mode: str) -> "_Image":
            copy = _Image()
            copy.mode = mode
            return copy

        def resize(self, size: tuple, resample: object = None) -> "_Image":
            obj = _Image()
            obj.size = size
            return obj

    _pil_image_mod.Image = _Image  # type: ignore[attr-defined]
    _pil_image_mod.LANCZOS = _Image.LANCZOS  # type: ignore[attr-defined]
    _pil_image_mod.fromarray = _Image.fromarray  # type: ignore[attr-defined]
    _pil_pkg.Image = _pil_image_mod  # type: ignore[attr-defined]
    sys.modules["PIL"] = _pil_pkg
    sys.modules["PIL.Image"] = _pil_image_mod


_install_pil_stub()

import pytest
from unittest.mock import MagicMock, patch

from src.ingest.common.types import IngestionConfig, Runtime
from src.ingest.embedding.nodes.visual_embedding import (
    visual_embedding_node,
    _extract_from_docling,
    _to_rgb,
    _resize_page_images,
)
from src.ingest.support.colqwen import ColQwen2PageEmbedding, VisualEmbeddingError

# ---------------------------------------------------------------------------
# Patch targets (from existing test_visual_embedding_node.py)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> IngestionConfig:
    defaults = dict(
        enable_visual_embedding=True,
        page_image_max_dimension=1024,
        page_image_quality=85,
        colqwen_batch_size=4,
        visual_target_collection="RAGVisualPages",
        colqwen_model_name="vidore/colqwen2-v1.0",
    )
    defaults.update(kwargs)
    return IngestionConfig(**defaults)


def _make_runtime(config: IngestionConfig, db_client=None, weaviate_client=None) -> Runtime:
    return Runtime(
        config=config,
        embedder=MagicMock(),
        weaviate_client=weaviate_client or MagicMock(),
        kg_builder=None,
        db_client=db_client,
    )


def _make_state(
    config: IngestionConfig | None = None,
    document_id: str = "doc-uuid-123",
    page_images: list | None = None,
    docling_document: object = None,
    errors: list | None = None,
    processing_log: list | None = None,
    db_client: object = None,
    weaviate_client: object = None,
) -> dict:
    if config is None:
        config = _make_config()
    runtime = _make_runtime(config, db_client=db_client, weaviate_client=weaviate_client)
    if docling_document is None:
        docling_document = MagicMock()
    return {
        "config": config,
        "runtime": runtime,
        "document_id": document_id,
        "source_key": "local:test.pdf",
        "source_uri": "file:///test.pdf",
        "source_name": "test.pdf",
        "docling_document": docling_document,
        "page_images": page_images,
        "processing_log": processing_log or [],
        "errors": errors or [],
        "stored_count": 5,
        "chunks": [],
        "metadata_summary": "",
        "metadata_keywords": [],
    }


def _fake_page_data():
    """Return one page of (page_num, img, w, h) with a mock PIL image."""
    img = MagicMock()
    img.size = (800, 1000)
    img.mode = "RGB"
    resized = MagicMock()
    resized.size = (800, 1000)
    resized.mode = "RGB"
    img.resize.return_value = resized
    return [(1, img, 800, 1000)]


def _make_embeddings(page_numbers: list[int]) -> list[ColQwen2PageEmbedding]:
    return [
        ColQwen2PageEmbedding(
            page_number=n,
            mean_vector=[0.1] * 128,
            patch_vectors=[[0.05] * 128] * 800,
            patch_count=800,
        )
        for n in page_numbers
    ]


# ---------------------------------------------------------------------------
# Line 193-196: missing document_id → error and early return
# ---------------------------------------------------------------------------


class TestMissingDocumentId:
    def test_mock_missing_document_id_returns_error(self):
        """When document_id is empty, node returns error and skips MinIO ops."""
        state = _make_state(document_id="")

        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=[]), \
             patch(_UNLOAD_COLQWEN), \
             patch(_DELETE_PAGES), \
             patch(_DELETE_VISUAL), \
             patch("src.ingest.embedding.nodes.visual_embedding._extract_page_images",
                   return_value=_fake_page_data()):
            result = visual_embedding_node(state)

        assert result["visual_stored_count"] == 0
        errors = result.get("errors") or []
        assert any("missing_document_id" in str(e) or "document_id" in str(e).lower() for e in errors)


# ---------------------------------------------------------------------------
# Line 257: no MinIO client → warning logged, skips minio storage
# ---------------------------------------------------------------------------


class TestNoMinioClient:
    def test_mock_no_minio_client_skips_storage(self):
        """When db_client is None, page image storage is skipped (line 257 warning)."""
        state = _make_state(db_client=None)

        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings([1])), \
             patch(_UNLOAD_COLQWEN), \
             patch(_STORE_PAGES, return_value=["pages/doc-uuid-123/0001.jpg"]) as mock_store, \
             patch(_DELETE_VISUAL), \
             patch(_DELETE_PAGES), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=1), \
             patch("src.ingest.embedding.nodes.visual_embedding._extract_page_images",
                   return_value=_fake_page_data()):
            result = visual_embedding_node(state)

        # store_page_images should NOT be called since db_client is None
        mock_store.assert_not_called()


# ---------------------------------------------------------------------------
# Lines 320-321: VisualEmbeddingError during embed_page_images
# ---------------------------------------------------------------------------


class TestVisualEmbeddingErrorDuringInference:
    def test_mock_visual_embedding_error_adds_error(self):
        """VisualEmbeddingError during embed_page_images is caught and added to errors."""
        state = _make_state()

        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, side_effect=VisualEmbeddingError("inference failed")), \
             patch(_UNLOAD_COLQWEN), \
             patch(_DELETE_PAGES), \
             patch(_DELETE_VISUAL), \
             patch(_STORE_PAGES, return_value=[]), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=0), \
             patch("src.ingest.embedding.nodes.visual_embedding._extract_page_images",
                   return_value=_fake_page_data()):
            result = visual_embedding_node(state)

        # Should return 0 stored and an error about inference
        assert result["visual_stored_count"] == 0
        errors = result.get("errors") or []
        assert any("inference" in str(e) for e in errors)

    def test_mock_model_unloaded_even_on_inference_error(self):
        """ColQwen2 model must be unloaded in finally block even on error."""
        fake_model = MagicMock()
        state = _make_state()

        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(fake_model, MagicMock())), \
             patch(_EMBED_PAGES, side_effect=VisualEmbeddingError("oops")), \
             patch(_UNLOAD_COLQWEN) as mock_unload, \
             patch(_DELETE_PAGES), \
             patch(_DELETE_VISUAL), \
             patch(_STORE_PAGES, return_value=[]), \
             patch(_ENSURE_VISUAL), \
             patch(_ADD_VISUAL, return_value=0), \
             patch("src.ingest.embedding.nodes.visual_embedding._extract_page_images",
                   return_value=_fake_page_data()):
            visual_embedding_node(state)

        mock_unload.assert_called_once_with(fake_model)


# ---------------------------------------------------------------------------
# Lines 340-345: ensure_visual_collection exception
# ---------------------------------------------------------------------------


class TestEnsureVisualCollectionError:
    def test_mock_ensure_visual_collection_error_adds_error(self):
        """Exception from ensure_visual_collection adds error and skips indexing."""
        state = _make_state()

        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings([1])), \
             patch(_UNLOAD_COLQWEN), \
             patch(_DELETE_PAGES), \
             patch(_DELETE_VISUAL), \
             patch(_STORE_PAGES, return_value=["pages/doc-uuid-123/0001.jpg"]), \
             patch(_ENSURE_VISUAL, side_effect=Exception("collection error")), \
             patch(_ADD_VISUAL) as mock_add, \
             patch("src.ingest.embedding.nodes.visual_embedding._extract_page_images",
                   return_value=_fake_page_data()):
            result = visual_embedding_node(state)

        # add_visual_documents should not be called when ensure fails
        mock_add.assert_not_called()
        errors = result.get("errors") or []
        assert any("weaviate_ensure" in str(e) for e in errors)


# ---------------------------------------------------------------------------
# Lines 383-384: no Weaviate client → skip visual indexing
# ---------------------------------------------------------------------------


class TestNoWeaviateClient:
    def test_mock_no_weaviate_client_skips_indexing(self):
        """When weaviate_client is None, visual indexing is skipped (line 383-384)."""
        state = _make_state(weaviate_client=None)
        # Runtime must also have weaviate_client=None
        state["runtime"] = Runtime(
            config=state["runtime"].config,
            embedder=MagicMock(),
            weaviate_client=None,
            kg_builder=None,
            db_client=MagicMock(),
        )

        with patch(_ENSURE_COLQWEN), \
             patch(_LOAD_COLQWEN, return_value=(MagicMock(), MagicMock())), \
             patch(_EMBED_PAGES, return_value=_make_embeddings([1])), \
             patch(_UNLOAD_COLQWEN), \
             patch(_DELETE_PAGES), \
             patch(_STORE_PAGES, return_value=["pages/x/0001.jpg"]), \
             patch(_ADD_VISUAL) as mock_add, \
             patch("src.ingest.embedding.nodes.visual_embedding._extract_page_images",
                   return_value=_fake_page_data()):
            result = visual_embedding_node(state)

        mock_add.assert_not_called()
        assert result["visual_stored_count"] == 0


# ---------------------------------------------------------------------------
# Lines 480-481: _extract_from_docling with no pages attribute
# ---------------------------------------------------------------------------


class TestExtractFromDoclingNoPages:
    def test_mock_extract_from_docling_no_pages_attr(self):
        """_extract_from_docling returns [] when document has no 'pages' attr."""
        doc = object()  # has no 'pages' attribute
        result = _extract_from_docling(doc)
        assert result == []

    def test_mock_extract_from_docling_empty_pages_attr(self):
        """_extract_from_docling returns [] when pages is falsy (empty dict)."""
        doc = MagicMock()
        doc.pages = {}  # falsy — empty dict
        result = _extract_from_docling(doc)
        assert result == []

    def test_mock_extract_from_docling_none_pages_attr(self):
        """_extract_from_docling returns [] when pages is None."""
        doc = MagicMock()
        doc.pages = None
        result = _extract_from_docling(doc)
        assert result == []


# ---------------------------------------------------------------------------
# Lines 485-487: pages iteration exception
# ---------------------------------------------------------------------------


class TestExtractFromDoclingIterationError:
    def test_mock_extract_from_docling_iteration_error(self):
        """_extract_from_docling returns [] when iterating pages raises."""

        class BadPages:
            def __bool__(self):
                return True

            def values(self):
                raise RuntimeError("cannot iterate")

        doc = MagicMock()
        doc.pages = BadPages()
        result = _extract_from_docling(doc)
        assert result == []


# ---------------------------------------------------------------------------
# Lines 491-509: per-page extraction in _extract_from_docling
# ---------------------------------------------------------------------------


class TestExtractFromDoclingPerPage:
    def _make_page_item(self, page_no: int, image=None, mode: str = "RGB"):
        """Build a fake page item."""
        item = MagicMock()
        item.page_no = page_no
        if image is None:
            img = MagicMock()
            img.size = (800, 1000)
            img.mode = mode
            img.convert.return_value = img
            item.image = img
        else:
            item.image = image
        return item

    def test_mock_extract_from_docling_skips_page_with_no_image(self):
        """Pages with image=None are skipped."""
        item = MagicMock()
        item.page_no = 0
        item.image = None

        doc = MagicMock()
        doc.pages = {0: item}
        result = _extract_from_docling(doc)
        assert result == []

    def test_mock_extract_from_docling_valid_page_extracted(self):
        """Valid page items produce (page_num, pil_img, w, h) tuples."""
        item = self._make_page_item(0)  # page_no=0 → page_num=1

        doc = MagicMock()
        doc.pages = {0: item}

        with patch("src.ingest.embedding.nodes.visual_embedding._to_rgb") as mock_rgb:
            mock_img = MagicMock()
            mock_img.size = (800, 1000)
            mock_rgb.return_value = mock_img
            result = _extract_from_docling(doc)

        assert len(result) == 1
        page_num, pil_img, w, h = result[0]
        assert page_num == 1  # 0 + 1
        assert w == 800
        assert h == 1000

    def test_mock_extract_from_docling_page_conversion_error_skipped(self):
        """Pages where _to_rgb raises are skipped gracefully."""
        item = self._make_page_item(0)

        doc = MagicMock()
        doc.pages = {0: item}

        with patch("src.ingest.embedding.nodes.visual_embedding._to_rgb",
                   side_effect=Exception("conversion error")):
            result = _extract_from_docling(doc)

        assert result == []

    def test_mock_extract_from_docling_non_int_page_no(self):
        """When page_no is not an int, page_num is computed from result length."""
        item = MagicMock()
        item.page_no = "first"  # not an int
        img = MagicMock()
        img.size = (200, 300)
        item.image = img

        doc = MagicMock()
        doc.pages = {"first": item}

        with patch("src.ingest.embedding.nodes.visual_embedding._to_rgb") as mock_rgb:
            mock_img = MagicMock()
            mock_img.size = (200, 300)
            mock_rgb.return_value = mock_img
            result = _extract_from_docling(doc)

        assert len(result) == 1
        # page_num = len(result_so_far) + 1 = 0 + 1 = 1
        assert result[0][0] == 1


# ---------------------------------------------------------------------------
# Lines 523-528: _to_rgb conversion
# ---------------------------------------------------------------------------


class TestToRgb:
    def test_mock_to_rgb_non_pil_image_wrapped(self):
        """Non-PIL images are wrapped with Image.fromarray."""
        import src.ingest.embedding.nodes.visual_embedding as ve_mod

        fake_arr = [[1, 2, 3]]
        fake_pil = MagicMock()
        fake_pil.mode = "RGB"

        # Patch Image.fromarray on the PIL.Image module used by visual_embedding
        orig_fromarray = getattr(ve_mod.Image, "fromarray", None)
        try:
            ve_mod.Image.fromarray = lambda arr: fake_pil  # type: ignore[attr-defined]
            result = _to_rgb(fake_arr)
        finally:
            if orig_fromarray is not None:
                ve_mod.Image.fromarray = orig_fromarray  # type: ignore[attr-defined]

        assert result is fake_pil

    def test_mock_to_rgb_rgba_image_converted_to_rgb(self):
        """Images with non-RGB mode are converted via .convert()."""
        import src.ingest.embedding.nodes.visual_embedding as ve_mod

        # Create an instance that isinstance checks as Image.Image
        fake_img = MagicMock(spec=ve_mod.Image.Image)
        fake_img.mode = "RGBA"
        rgb_img = MagicMock()
        fake_img.convert.return_value = rgb_img

        result = _to_rgb(fake_img)
        fake_img.convert.assert_called_once_with("RGB")
        assert result is rgb_img

    def test_mock_to_rgb_already_rgb_unchanged(self):
        """Images already in RGB mode are returned as-is without conversion."""
        import src.ingest.embedding.nodes.visual_embedding as ve_mod

        fake_img = MagicMock(spec=ve_mod.Image.Image)
        fake_img.mode = "RGB"

        result = _to_rgb(fake_img)
        fake_img.convert.assert_not_called()
        assert result is fake_img


# ---------------------------------------------------------------------------
# Lines 557-558: _resize_page_images resize exception
# ---------------------------------------------------------------------------


class TestResizePageImagesException:
    def test_mock_resize_exception_uses_original_image(self):
        """When img.resize raises, the original image is used (fallback path)."""
        img = MagicMock()
        img.size = (800, 1000)
        img.mode = "RGB"
        img.resize.side_effect = OSError("resize failed")

        page_data = [(1, img, 800, 1000)]
        result = _resize_page_images(page_data, max_dimension=400)

        # Should still produce one entry (with original img on resize failure)
        assert len(result) == 1
        page_num, out_img, orig_w, orig_h = result[0]
        assert page_num == 1
        assert orig_w == 800
        assert orig_h == 1000

    def test_mock_resize_not_needed_when_within_bounds(self):
        """Images smaller than max_dimension are not resized."""
        img = MagicMock()
        img.size = (200, 300)
        img.mode = "RGB"

        page_data = [(1, img, 200, 300)]
        result = _resize_page_images(page_data, max_dimension=1024)

        img.resize.assert_not_called()
        assert result[0][1] is img
