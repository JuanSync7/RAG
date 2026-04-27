# @summary
# Visual embedding LangGraph node for the dual-track embedding pipeline.
# Extracts page images via Docling, stores in MinIO, embeds via ColQwen2,
# indexes in Weaviate visual collection.
# Exports: visual_embedding_node
# Deps: src.ingest.support.colqwen, src.db.minio.store,
#       src.vector_db.weaviate.visual_store, src.ingest.common.shared
# @end-summary

"""Visual embedding node for the Embedding Pipeline.

This node runs after ``embedding_storage_node`` (text track) and implements
the visual track of the dual-track embedding pipeline (FR-601, FR-604).

Responsibilities
----------------
1. Short-circuit if visual embedding is disabled or prerequisites are missing.
2. Extract per-page PIL images from the ``DoclingDocument`` (or from pre-populated
   ``page_images`` in state if the Docling parsing step already produced them).
3. Resize each image so its longer edge does not exceed ``page_image_max_dimension``.
4. Clean up pre-existing MinIO page images and Weaviate visual objects (update mode).
5. Store resized images in MinIO under ``pages/{document_id}/{page_num:04d}.jpg``.
6. Load ColQwen2 with 4-bit quantisation, run batch inference, always unload in finally.
7. Insert visual page objects (11 properties + mean_vector named vector) into Weaviate.
8. Return partial state update — text-track fields are NEVER modified (FR-803).
9. Clear ``page_images`` from state to free memory (FR-606).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from PIL import Image

# Pillow 10+ moved LANCZOS to Image.Resampling; keep compatibility with both.
_LANCZOS = getattr(getattr(Image, "Resampling", None), "LANCZOS", None) or getattr(Image, "LANCZOS", 1)

from src.db.minio import (
    delete_page_images,
    store_page_images,
)
from src.ingest.common import append_processing_log
from src.ingest.embedding.state import EmbeddingPipelineState
from src.ingest.support import (
    ColQwen2LoadError,
    ColQwen2PageEmbedding,
    VisualEmbeddingError,
    embed_page_images,
    ensure_colqwen_ready,
    load_colqwen_model,
    unload_colqwen_model,
)
from src.vector_db.weaviate import (
    add_visual_documents,
    delete_visual_by_source_key,
    ensure_visual_collection,
)

logger = logging.getLogger("rag.ingest.embedding.visual_embedding")


# ---------------------------------------------------------------------------
# Node entry point
# ---------------------------------------------------------------------------


def visual_embedding_node(state: EmbeddingPipelineState) -> dict[str, Any]:
    """Visual embedding pipeline node: extract, store, embed, and index page images.

    Positioned in the LangGraph DAG after embedding_storage, before
    knowledge_graph_storage (FR-601, FR-604).

    Short-circuit conditions (FR-603, NFR-903):
    - enable_visual_embedding=False
    - docling_document is None
    - No extractable pages

    On short-circuit: returns visual_stored_count=0, logs descriptive entry.
    On completion: clears page_images from state (FR-606).

    MUST NOT modify: stored_count, chunks, or any
    text-track state fields (FR-803).

    Args:
        state: EmbeddingPipelineState with runtime, document_id, source_key,
            and docling_document populated by preceding nodes.

    Returns:
        Dict with keys: visual_stored_count, page_images (None), processing_log,
        and optionally errors.
    """
    node_start = time.monotonic()

    # ── Short-circuit: config flag ─────────────────────────────────────────
    runtime = state["runtime"]
    config = runtime.config

    if not config.enable_visual_embedding:
        logger.debug("visual_embedding_node: disabled by config.")
        return {
            "visual_stored_count": 0,
            "page_images": None,
            "processing_log": append_processing_log(
                state, "visual_embedding:skipped:disabled"
            ),
        }

    # ── Short-circuit: no docling document ────────────────────────────────
    docling_document = state.get("docling_document")
    if docling_document is None:
        logger.debug("visual_embedding_node: no docling document in state.")
        return {
            "visual_stored_count": 0,
            "page_images": None,
            "processing_log": append_processing_log(
                state,
                "visual_embedding:skipped:no_docling_document",
            ),
        }

    # ── Top-level try/except: isolate all unhandled failures ──────────────
    try:
        return _run_visual_embedding(state, config, runtime, docling_document, node_start)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error(
            "visual_embedding_node: unhandled error: %s",
            exc,
            exc_info=True,
        )
        return {
            "visual_stored_count": 0,
            "page_images": None,
            "errors": state.get("errors", []) + [f"visual_embedding:{exc}"],
            "processing_log": append_processing_log(
                state, "visual_embedding:error:unhandled"
            ),
        }


# ---------------------------------------------------------------------------
# Core implementation (called from node, isolated from unhandled exceptions)
# ---------------------------------------------------------------------------


def _run_visual_embedding(
    state: EmbeddingPipelineState,
    config: Any,
    runtime: Any,
    docling_document: Any,
    node_start: float,
) -> dict[str, Any]:
    """Core implementation of the visual embedding node.

    Separated from the public node so that the top-level try/except in
    ``visual_embedding_node`` cleanly catches any unhandled error here.
    """
    errors: list[str] = list(state.get("errors") or [])

    # ── Step 1: Extract page images ─────────────────────────────────────────
    # page_data: list of (1-indexed page_num, pil_image, orig_w, orig_h)
    page_data = _extract_page_images(state, docling_document)

    # ── Short-circuit: no extractable pages ─────────────────────────────────
    if not page_data:
        logger.debug("visual_embedding_node: no pages found in document.")
        return {
            "visual_stored_count": 0,
            "page_images": None,
            "processing_log": append_processing_log(
                state,
                "visual_embedding:skipped:no_pages",
            ),
        }

    pages_extracted = len(page_data)
    logger.info(
        "visual_embedding_node: extracted %d page image(s) for source=%r.",
        pages_extracted,
        state.get("source_name", "<unknown>"),
    )

    # ── Step 2: Resize page images ──────────────────────────────────────────
    page_data = _resize_page_images(page_data, config.page_image_max_dimension)

    # ── Step 3: Gather clients ──────────────────────────────────────────────
    minio_client = runtime.db_client  # Optional[Any]; None if not configured
    weaviate_client = runtime.weaviate_client
    document_id: str = state.get("document_id", "")
    if not document_id:
        logger.error(
            "visual_embedding_node: document_id is empty — skipping MinIO operations"
        )
        return {
            "visual_stored_count": 0,
            "page_images": None,
            "errors": [*(state.get("errors") or []), "visual_embedding:missing_document_id"],
            "processing_log": append_processing_log(
                state, "visual_embedding:error:no_document_id"
            ),
        }
    source_key: str = state.get("source_key", "")
    total_pages = len(page_data)

    # ── Step 4: Pre-storage cleanup (FR-404, FR-405) ───────────────────────
    if minio_client is not None:
        try:
            deleted_images = delete_page_images(minio_client, document_id)
            logger.debug(
                "visual_embedding_node: deleted %d pre-existing MinIO page images.",
                deleted_images,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "visual_embedding_node: failed to delete pre-existing MinIO images: %s",
                exc,
            )

    if weaviate_client is not None:
        try:
            delete_visual_by_source_key(
                weaviate_client, source_key, config.visual_target_collection
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "visual_embedding_node: failed to delete pre-existing Weaviate visual objects: %s",
                exc,
            )

    # ── Step 5: Store page images in MinIO (FR-401, FR-402, FR-403) ─────────
    minio_key_map: dict[int, str] = {}
    pages_stored_minio = 0

    if minio_client is not None:
        pages_for_minio = [(pn, img) for pn, img, _w, _h in page_data]
        stored_keys = store_page_images(
            minio_client,
            document_id,
            pages_for_minio,
            config.page_image_quality,
        )
        pages_stored_minio = len(stored_keys)
        # Build {page_number: minio_key} from the returned key list.
        # Key pattern: pages/{document_id}/{page_num:04d}.jpg
        for key in stored_keys:
            try:
                stem = key.rsplit("/", 1)[-1]          # e.g. "0001.jpg"
                page_num = int(stem.replace(".jpg", ""))
                minio_key_map[page_num] = key
            except (ValueError, IndexError):
                logger.warning(
                    "visual_embedding_node: could not parse page number from key %r", key
                )
    else:
        logger.warning(
            "visual_embedding_node: no MinIO client available (db_client is None); "
            "skipping page image storage."
        )

    logger.info(
        "visual_embedding_node: stored %d/%d page images in MinIO.",
        pages_stored_minio,
        total_pages,
    )

    # ── Step 6: Load ColQwen2 and embed (FR-801, FR-806) ───────────────────
    resized_images = [img for _pn, img, _w, _h in page_data]
    page_numbers = [pn for pn, _img, _w, _h in page_data]

    # Pre-check dependencies before attempting load.
    model = None
    processor = None
    embeddings = []

    try:
        ensure_colqwen_ready()
    except ColQwen2LoadError as exc:
        logger.error("visual_embedding_node: ColQwen2 dependency check failed: %s", exc)
        errors.append(f"visual_embedding:colqwen_load:{exc}")
        return {
            "visual_stored_count": 0,
            "page_images": None,
            "errors": errors,
            "processing_log": append_processing_log(
                state, "visual_embedding:error:colqwen_load"
            ),
        }

    try:
        model, processor = load_colqwen_model(config.colqwen_model_name)
    except ColQwen2LoadError as exc:
        logger.error("visual_embedding_node: ColQwen2 model load failed: %s", exc)
        errors.append(f"visual_embedding:colqwen_load:{exc}")
        return {
            "visual_stored_count": 0,
            "page_images": None,
            "errors": errors,
            "processing_log": append_processing_log(
                state, "visual_embedding:error:colqwen_load"
            ),
        }

    # Run inference; always unload model in finally block (FR-301, FR-302, FR-303).
    try:
        embeddings = embed_page_images(
            model,
            processor,
            resized_images,
            config.colqwen_batch_size,
            page_numbers=page_numbers,
        )
        logger.info(
            "visual_embedding_node: embedded %d/%d pages via ColQwen2.",
            len(embeddings),
            total_pages,
        )
    except VisualEmbeddingError as exc:
        logger.error("visual_embedding_node: ColQwen2 inference error: %s", exc)
        errors.append(f"visual_embedding:inference:{exc}")
        # Return partial result — embeddings list stays empty.
    finally:
        if model is not None:
            unload_colqwen_model(model)

    pages_embedded = len(embeddings)

    # ── Step 7: Insert into Weaviate (FR-501..FR-507) ───────────────────────
    visual_stored_count = 0

    if embeddings and weaviate_client is not None:
        # Build lookup: page_number -> (orig_w, orig_h) from page_data.
        dims_map: dict[int, tuple[int, int]] = {
            pn: (w, h) for pn, _img, w, h in page_data
        }

        try:
            ensure_visual_collection(weaviate_client, config.visual_target_collection)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "visual_embedding_node: failed to ensure Weaviate visual collection: %s",
                exc,
            )
            errors.append(f"visual_embedding:weaviate_ensure:{exc}")
        else:
            documents = []
            for emb in embeddings:
                page_num = emb.page_number
                orig_w, orig_h = dims_map.get(page_num, (0, 0))
                minio_key = minio_key_map.get(page_num, "")
                doc = {
                    "document_id": document_id,
                    "page_number": page_num,
                    "source_key": source_key,
                    "source_uri": state.get("source_uri", ""),
                    "source_name": state.get("source_name", ""),
                    "tenant_id": "",
                    "total_pages": total_pages,
                    "page_width_px": orig_w,
                    "page_height_px": orig_h,
                    "minio_key": minio_key,
                    "patch_vectors": json.dumps(emb.patch_vectors),
                    "mean_vector": emb.mean_vector,
                }
                documents.append(doc)

            try:
                visual_stored_count = add_visual_documents(
                    weaviate_client, documents, config.visual_target_collection
                )
                logger.info(
                    "visual_embedding_node: indexed %d visual page objects in Weaviate.",
                    visual_stored_count,
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(
                    "visual_embedding_node: failed to insert visual documents into Weaviate: %s",
                    exc,
                )
                errors.append(f"visual_embedding:weaviate_insert:{exc}")

    elif weaviate_client is None:
        logger.warning(
            "visual_embedding_node: no Weaviate client available; "
            "skipping visual document indexing."
        )

    # ── Step 8: Timing and processing log ──────────────────────────────────
    elapsed_s = time.monotonic() - node_start
    log_entries = [
        f"visual_embedding:pages_extracted:{pages_extracted}",
        f"visual_embedding:pages_stored_minio:{pages_stored_minio}",
        f"visual_embedding:pages_embedded:{pages_embedded}",
        f"visual_embedding:pages_indexed:{visual_stored_count}",
        f"visual_embedding:elapsed_s:{elapsed_s:.2f}",
    ]

    # Build final processing_log by appending all entries sequentially.
    # append_processing_log creates a new list each call; chain them.
    processing_log = state.get("processing_log", [])
    # We need a temporary state-like dict for append_processing_log since it
    # reads state["processing_log"] directly.
    _tmp_state = dict(state)
    for entry in log_entries:
        _tmp_state["processing_log"] = processing_log
        processing_log = append_processing_log(_tmp_state, entry)  # type: ignore[arg-type]

    # ── Step 9: Return partial state update (FR-803, FR-606) ───────────────
    result: dict[str, Any] = {
        "visual_stored_count": visual_stored_count,
        "page_images": None,  # FR-606: clear from state to free memory
        "processing_log": processing_log,
    }
    if errors and errors != list(state.get("errors") or []):
        # Only include errors if we added new ones.
        result["errors"] = errors

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_page_images(
    state: EmbeddingPipelineState,
    docling_document: Any,
) -> list[tuple[int, Any, int, int]]:
    """Extract page images as (1-indexed page_num, PIL.Image, orig_w, orig_h) tuples.

    First tries to use pre-populated ``state["page_images"]`` if present
    (set by the Docling parsing step when ``generate_page_images=True``).
    Falls back to iterating the DoclingDocument's pages if page_images is None
    or empty.

    Args:
        state: Current pipeline state.
        docling_document: DoclingDocument object from state.

    Returns:
        List of (page_num, pil_image, original_width, original_height) tuples.
        Empty list if no images can be extracted.
    """
    # Check for pre-populated page images in state.
    state_images = state.get("page_images")
    if state_images:
        result: list[tuple[int, Any, int, int]] = []
        for idx, img in enumerate(state_images, start=1):
            pil_img = _to_rgb(img)
            w, h = pil_img.size
            result.append((idx, pil_img, w, h))
        return result

    # Fall back to extracting from the DoclingDocument object.
    return _extract_from_docling(docling_document)


def _extract_from_docling(docling_document: Any) -> list[tuple[int, Any, int, int]]:
    """Extract page images directly from a DoclingDocument.

    DoclingDocument stores page images in its ``pages`` dict as
    ``PageItem`` objects.  Each ``PageItem`` may carry a rendered image in
    its ``image`` attribute.  If the image is not pre-rendered we fall back
    to using the ``page_no`` metadata only and skip that page.

    Args:
        docling_document: Native DoclingDocument object.

    Returns:
        List of (page_num, pil_image, orig_w, orig_h) tuples for pages
        that successfully yielded an image.  Empty list on any access error.
    """
    result: list[tuple[int, Any, int, int]] = []

    # DoclingDocument.pages is a dict[int, PageItem] keyed by 0-indexed page_no.
    pages_attr = getattr(docling_document, "pages", None)
    if not pages_attr:
        logger.debug("_extract_from_docling: document has no 'pages' attribute.")
        return result

    try:
        pages_items = pages_attr.values() if hasattr(pages_attr, "values") else pages_attr
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("_extract_from_docling: failed to iterate pages: %s", exc)
        return result

    for page_item in pages_items:
        # Attempt to retrieve the rendered image from the page item.
        page_image = getattr(page_item, "image", None)
        if page_image is None:
            continue

        # page_no is typically 0-indexed; convert to 1-indexed.
        page_no_raw = getattr(page_item, "page_no", None)
        page_num = (page_no_raw + 1) if isinstance(page_no_raw, int) else len(result) + 1

        try:
            pil_img = _to_rgb(page_image)
            w, h = pil_img.size
            result.append((page_num, pil_img, w, h))
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "_extract_from_docling: failed to convert page %d image: %s",
                page_num,
                exc,
            )
            continue

    return result


def _to_rgb(image: Any) -> Any:
    """Convert any PIL-compatible image to RGB mode.

    Args:
        image: A PIL.Image.Image or PIL-compatible object.

    Returns:
        A PIL.Image.Image in RGB mode.
    """
    if not isinstance(image, Image.Image):
        # Attempt to wrap; may raise if not compatible.
        image = Image.fromarray(image)  # type: ignore[arg-type]
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _resize_page_images(
    page_data: list[tuple[int, Any, int, int]],
    max_dimension: int,
) -> list[tuple[int, Any, int, int]]:
    """Resize page images so the longer edge does not exceed max_dimension.

    Aspect ratio is preserved.  Images that already fit within max_dimension
    are returned unchanged.

    Args:
        page_data: List of (page_num, pil_image, orig_w, orig_h) tuples.
        max_dimension: Maximum pixel dimension for the longer edge (FR-202).

    Returns:
        New list of (page_num, resized_pil_image, orig_w, orig_h) tuples.
        orig_w and orig_h reflect the pre-resize dimensions for metadata.
    """
    resized: list[tuple[int, Any, int, int]] = []
    for page_num, img, orig_w, orig_h in page_data:
        longer_edge = max(orig_w, orig_h)
        if longer_edge > max_dimension and longer_edge > 0:
            scale = max_dimension / longer_edge
            new_w = max(1, int(orig_w * scale))
            new_h = max(1, int(orig_h * scale))
            try:
                img = img.resize((new_w, new_h), _LANCZOS)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "_resize_page_images: failed to resize page %d (%dx%d -> %dx%d): %s",
                    page_num,
                    orig_w,
                    orig_h,
                    new_w,
                    new_h,
                    exc,
                )
        resized.append((page_num, img, orig_w, orig_h))
    return resized


__all__ = ["visual_embedding_node"]
