# @summary
# ColQwen2 model adapter for visual embedding pipeline.
# Exports: ColQwen2PageEmbedding, ColQwen2LoadError, VisualEmbeddingError,
#          ensure_colqwen_ready, load_colqwen_model, embed_page_images, unload_colqwen_model,
#          embed_text_query
# Deps: colpali_engine (optional), bitsandbytes (optional), torch, transformers
# @end-summary
"""ColQwen2 model adapter for visual embedding pipeline.

This module provides a minimal adapter around the ColQwen2 model to produce
128-dim patch-level visual embeddings from document page images. It handles
model lifecycle (load with 4-bit quantization, batch inference, GPU memory
release) for downstream visual embedding ingestion nodes.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any

from src.platform.observability import get_tracer

logger = logging.getLogger(__name__)


@dataclass
class ColQwen2PageEmbedding:
    """Embedding result for a single document page from ColQwen2.

    Attributes:
        page_number: 1-indexed page number within the document.
        mean_vector: 128-dim float32 mean-pooled vector (arithmetic mean across patches). FR-303
        patch_vectors: Raw patch vectors as list of list of float, serializable as JSON. FR-304
        patch_count: Number of patches produced for this page.
    """

    page_number: int  # FR-302: 1-indexed page number
    mean_vector: list[float]  # FR-303: 128-dim float32 mean-pooled vector
    patch_vectors: list[list[float]]  # FR-304: raw patch vectors, JSON-serializable
    patch_count: int  # FR-302: number of patches for this page


class VisualEmbeddingError(Exception):
    """Base exception for visual embedding pipeline errors.

    Non-fatal: per-page failures that should be caught and logged,
    allowing the pipeline to continue with remaining pages. FR-307
    """

    pass


class ColQwen2LoadError(VisualEmbeddingError):
    """Fatal: ColQwen2 model failed to load.

    When raised: visual_stored_count=0, error added to state['errors'],
    no retry attempted. FR-802
    """

    pass


def ensure_colqwen_ready() -> None:
    """Validate that colpali-engine and bitsandbytes are installed.

    Raises:
        ColQwen2LoadError: If required packages are not importable,
            with a clear install command message. FR-806, NFR-906
    """
    missing: list[str] = []
    try:
        import colpali_engine  # noqa: F401
    except ImportError:
        missing.append("colpali-engine")

    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        missing.append("bitsandbytes")

    if missing:
        packages_str = ", ".join(missing)
        raise ColQwen2LoadError(
            f"Required package(s) not installed: {packages_str}. "
            'Install with: pip install "rag[visual]" '
            "or: pip install colpali-engine bitsandbytes"
        )


def load_colqwen_model(model_name: str) -> tuple[Any, Any]:
    """Load ColQwen2 model and processor with 4-bit quantization.

    Uses BitsAndBytesConfig with load_in_4bit=True and float16 compute dtype.
    Peak VRAM must be <= 4GB (NFR-901).

    Args:
        model_name: HuggingFace model identifier. FR-103

    Returns:
        Tuple of (model, processor). Both are loaded to CUDA.

    Raises:
        ColQwen2LoadError: On any model load failure (FATAL). FR-802
    """
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except Exception as exc:
        raise ColQwen2LoadError(
            f"Failed to import torch or transformers: {exc}. "
            "Ensure torch and transformers are installed."
        ) from exc

    try:
        from colpali_engine.models import ColQwen2
        from colpali_engine.processors import ColQwen2Processor
    except Exception as exc:
        raise ColQwen2LoadError(
            f"Failed to import ColQwen2 from colpali_engine: {exc}. "
            'Install with: pip install "rag[visual]" or: pip install colpali-engine'
        ) from exc

    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        logger.info("Loading ColQwen2 model '%s' with 4-bit quantization ...", model_name)
        model = ColQwen2.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model.eval()

        logger.info("Loading ColQwen2Processor for '%s' ...", model_name)
        processor = ColQwen2Processor.from_pretrained(model_name)

        logger.info("ColQwen2 model and processor loaded successfully.")
        return model, processor
    except Exception as exc:
        raise ColQwen2LoadError(
            f"ColQwen2 model load failed for '{model_name}': {exc}"
        ) from exc


def embed_page_images(
    model: Any,
    processor: Any,
    images: list[Any],
    batch_size: int,
    *,
    page_numbers: list[int] | None = None,
) -> list[ColQwen2PageEmbedding]:
    """Batch-embed page images through ColQwen2.

    Processes images in batches of batch_size. Each page produces:
    - 128-dim float32 patch vectors (n_patches x 128)
    - 128-dim float32 mean-pooled vector (arithmetic mean across patches)

    Per-page inference failure logs a warning and skips the page (FR-307).
    Logs progress at 10% intervals for documents with >10 pages (FR-306).

    Args:
        model: Loaded ColQwen2 model.
        processor: Loaded ColQwen2Processor.
        images: List of PIL.Image objects (already resized/RGB).
        batch_size: Number of images per forward pass. FR-104
        page_numbers: Optional 1-indexed page numbers. If None,
            defaults to 1..len(images).

    Returns:
        List of ColQwen2PageEmbedding for successfully processed pages.
        Failed pages are omitted (not None entries).
    """
    import torch

    if page_numbers is None:
        page_numbers = list(range(1, len(images) + 1))

    n_pages = len(images)
    results: list[ColQwen2PageEmbedding] = []
    log_progress = n_pages > 10
    progress_interval = max(1, n_pages // 10)

    tracer = get_tracer()
    model_id = str(getattr(model, "name_or_path", None) or getattr(model, "config", None) or "colqwen2")
    span_ctx = tracer.span(
        "ingest.embedding.colqwen",
        {
            "gen_ai.system": "colqwen",
            "gen_ai.request.model": model_id,
            "gen_ai.prompt": f"{n_pages} pages",
            "batch_size": batch_size,
            "page_count": n_pages,
        },
    )
    with span_ctx as span:
      for batch_start in range(0, n_pages, batch_size):
          batch_end = min(batch_start + batch_size, n_pages)
          batch_images = images[batch_start:batch_end]
          batch_page_numbers = page_numbers[batch_start:batch_end]

          # Get model inputs from processor; move to model device.
          try:
              batch_inputs = processor.process_images(batch_images)
              batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}
          except Exception as exc:
              logger.warning(
                  "Failed to process image batch (pages %s-%s): %s — skipping batch.",
                  batch_page_numbers[0],
                  batch_page_numbers[-1],
                  exc,
              )
              continue

          # Run forward pass for the entire batch.
          try:
              with torch.inference_mode():
                  batch_output = model(**batch_inputs)
          except Exception as exc:
              logger.warning(
                  "Inference failed for batch (pages %s-%s): %s — skipping batch.",
                  batch_page_numbers[0],
                  batch_page_numbers[-1],
                  exc,
              )
              continue

          # batch_output may be a tensor or an object with a .last_hidden_state attribute.
          # ColQwen2 returns a tensor of shape (batch_size, n_patches, 128).
          if hasattr(batch_output, "last_hidden_state"):
              batch_tensor = batch_output.last_hidden_state
          elif isinstance(batch_output, torch.Tensor):
              batch_tensor = batch_output
          else:
              # Some colpali_engine versions return an object directly indexable.
              batch_tensor = batch_output

          # Iterate over individual pages within the batch.
          for idx_in_batch, page_num in enumerate(batch_page_numbers):
              try:
                  # Shape: (n_patches, 128)
                  page_tensor = batch_tensor[idx_in_batch]  # type: ignore[index]
                  # Compute arithmetic mean across patches -> (128,)
                  mean_tensor = page_tensor.float().mean(dim=0)

                  patch_vectors: list[list[float]] = page_tensor.float().cpu().tolist()
                  mean_vector: list[float] = mean_tensor.cpu().tolist()
                  patch_count = page_tensor.shape[0]

                  results.append(
                      ColQwen2PageEmbedding(
                          page_number=page_num,
                          mean_vector=mean_vector,
                          patch_vectors=patch_vectors,
                          patch_count=patch_count,
                      )
                  )
              except Exception as exc:
                  logger.warning(
                      "Failed to extract embedding for page %d: %s — skipping page.",
                      page_num,
                      exc,
                  )
                  continue

          # Log progress at 10% intervals for large documents (FR-306).
          if log_progress:
              pages_done = batch_end
              if pages_done % progress_interval == 0 or pages_done == n_pages:
                  pct = int(pages_done / n_pages * 100)
                  logger.info(
                      "Visual embedding progress: %d/%d pages processed (%d%%).",
                      pages_done,
                      n_pages,
                      pct,
                  )

      span.set_attribute("vector_count", len(results))
      if results:
          span.set_attribute("vector_dim", len(results[0].mean_vector))
      return results


def embed_text_query(model: Any, processor: Any, text: str) -> list[float]:
    """Encode a text query into a 128-dimensional float vector using ColQwen2.

    Uses ``processor.process_queries([text])`` to tokenize the input, passes
    the tokenized input through the model under ``torch.inference_mode()``,
    and produces a single 128-dim vector by mean-pooling across token-level
    output vectors. The result dtype is float32 (Python ``float``).

    Args:
        model: Loaded ColQwen2 model (from ``load_colqwen_model()``).
        processor: Loaded ColQwen2Processor (from ``load_colqwen_model()``).
        text: Non-empty query text to encode.

    Returns:
        A list of exactly 128 float values representing the mean-pooled
        query embedding in the ColQwen2 vector space.

    Raises:
        ValueError: If ``text`` is empty or contains only whitespace.
            The model is not invoked. Message contains "empty" or "blank".
        ColQwen2LoadError: If ``model`` or ``processor`` is None or invalid.
        VisualEmbeddingError: If the model's forward pass raises an unexpected
            error (e.g., CUDA OOM). The original exception is chained as
            ``__cause__``.
    """
    # FR-205: empty/whitespace guard
    if not text or not text.strip():
        raise ValueError("Query text is empty or blank")

    # FR-207: None model/processor guard
    if model is None or processor is None:
        raise ColQwen2LoadError(
            "ColQwen2 model or processor is None — call load_colqwen_model() first"
        )

    import torch

    try:
        # FR-203: text encoding via process_queries
        query_inputs = processor.process_queries([text])
        query_inputs = {k: v.to(model.device) for k, v in query_inputs.items()}

        with torch.inference_mode():
            query_output = model(**query_inputs)

        # Extract output tensor — ColQwen2 query mode produces (1, n_tokens, 128)
        if hasattr(query_output, "last_hidden_state"):
            q_tensor = query_output.last_hidden_state[0]
        elif isinstance(query_output, torch.Tensor):
            q_tensor = query_output[0]
        else:
            q_tensor = query_output[0]

        # FR-201: mean-pool across token-level vectors to produce 128-dim vector
        mean_vector: list[float] = q_tensor.float().mean(dim=0).cpu().tolist()
        return mean_vector
    except (ValueError, ColQwen2LoadError):
        raise
    except Exception as exc:
        raise VisualEmbeddingError(
            f"ColQwen2 text encoding failed: {exc}"
        ) from exc


def unload_colqwen_model(model: Any) -> None:
    """Release ColQwen2 model and free GPU memory.

    Deletes the model reference, calls torch.cuda.empty_cache() and
    gc.collect(). VRAM should return to pre-load levels (+/- 200MB). FR-305

    Args:
        model: The ColQwen2 model to unload.
    """
    import torch

    del model
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("ColQwen2 model unloaded and GPU memory released.")
