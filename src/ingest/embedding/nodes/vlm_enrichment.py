# @summary
# Post-chunking VLM image enrichment node for the Embedding Pipeline.
# Exports: vlm_enrichment_node, _find_image_placeholders, _replace_placeholder,
#          _enrich_chunk_external
# Deps: src.ingest.common.types (IngestionConfig), src.ingest.support.vision,
#       src.platform.llm
# @end-summary

"""Post-chunking VLM image enrichment node for the Embedding Pipeline.

This node runs after ``chunking_node`` and replaces ``![alt](src)`` image
placeholder patterns in chunk text with VLM-generated descriptions.

Architectural note
------------------
- ``vlm_mode="builtin"``: Docling's SmolVLM generates figure descriptions at
  parse time, already embedded in the ``DoclingDocument`` before chunking.
  This node is a **no-op** for that mode.
- ``vlm_mode="external"``: This node performs the VLM API calls via LiteLLM
  router for each image placeholder found in chunk text.
- ``vlm_mode="disabled"``: This node is a **no-op**.

All per-chunk failures are non-fatal; the original chunk text is preserved
on any error. The node itself never raises.
"""

from __future__ import annotations

import copy
import logging
import re
import time
from pathlib import Path
from typing import Any

from src.ingest.common import ProcessedChunk
from src.ingest.common import append_processing_log
from src.ingest.common import IngestionConfig
from src.ingest.embedding.state import EmbeddingPipelineState
from src.ingest.support import (
    _IMAGE_REF_PATTERN,
    _describe_image,
    _extract_image_candidates,
)

logger = logging.getLogger("rag.ingest.embedding.vlm_enrichment")


# ── Public API ──────────────────────────────────────────────────────────────


def vlm_enrichment_node(state: EmbeddingPipelineState) -> dict[str, Any]:
    """Replace image placeholders in chunks with VLM-generated descriptions.

    Mode dispatch:

    - ``vlm_mode="disabled"``: immediate no-op, log ``vlm_enrichment:skipped``
    - ``vlm_mode="builtin"``: immediate no-op, log ``vlm_enrichment:skipped``
      (descriptions already embedded in DoclingDocument at parse time by SmolVLM)
    - ``vlm_mode="external"``: iterate chunks, call ``_enrich_chunk_external``
      for chunks with image placeholders, respect ``vision_max_figures`` limit

    Per-chunk failures are non-fatal: original chunk text is preserved and a
    warning is logged. This node never raises — all exceptions are caught
    internally.

    Args:
        state: Must contain ``"chunks"`` (list[ProcessedChunk]),
            ``"runtime"`` (Runtime with a ``config`` attribute), and
            ``"processing_log"`` (list[str]).

    Returns:
        dict with keys ``"chunks"`` and ``"processing_log"``.
    """
    t0 = time.monotonic()
    config: IngestionConfig = state["runtime"].config

    # No-op for modes that do not require post-chunking VLM enrichment.
    if config.vlm_mode != "external":
        return {
            "chunks": state.get("chunks", []),
            "processing_log": append_processing_log(state, "vlm_enrichment:skipped"),
        }

    # External mode: iterate chunks and replace image placeholders.
    chunks: list[ProcessedChunk] = state.get("chunks", [])
    source_uri: str = state.get("source_uri", "")

    try:
        result_chunks: list[ProcessedChunk] = []
        figures_processed_count = 0

        for chunk in chunks:
            enriched_chunk, figures_processed_count = _enrich_chunk_external(
                chunk,
                config,
                figures_processed_count,
                source_uri=source_uri,
            )
            result_chunks.append(enriched_chunk)

        logger.info("vlm_enrichment complete: source=%s", state.get("source_name", ""))
        logger.debug("vlm_enrichment_node completed in %.3fs", time.monotonic() - t0)
        return {
            "chunks": result_chunks,
            "processing_log": append_processing_log(
                state, "vlm_enrichment:external:ok"
            ),
        }

    except Exception as exc:  # pylint: disable=broad-except
        logger.error(
            "vlm_enrichment_node: unexpected error, returning original chunks: %s",
            exc,
            exc_info=True,
        )
        return {
            "chunks": chunks,
            "processing_log": append_processing_log(
                state, "vlm_enrichment:external:error"
            ),
        }


# ── Helpers ─────────────────────────────────────────────────────────────────


def _find_image_placeholders(chunk_text: str) -> list[re.Match]:
    """Find all image reference placeholders in chunk text.

    Uses ``_IMAGE_REF_PATTERN`` from ``src.ingest.support.vision`` to detect
    ``![alt](src)`` patterns.

    Args:
        chunk_text: Text of a single chunk.

    Returns:
        List of ``re.Match`` objects for each placeholder found (may be empty).
    """
    return list(_IMAGE_REF_PATTERN.finditer(chunk_text))


def _replace_placeholder(
    chunk_text: str,
    match: re.Match,
    description: str,
) -> str:
    """Replace a single matched image placeholder with the VLM description.

    Only the matched span is replaced. All surrounding text is preserved exactly.

    Args:
        chunk_text: Full chunk text containing the placeholder.
        match: ``re.Match`` from ``_find_image_placeholders`` identifying the span.
        description: VLM-generated description text.

    Returns:
        Chunk text with the matched placeholder replaced by ``description``.
    """
    return chunk_text[: match.start()] + description + chunk_text[match.end() :]


def _enrich_chunk_external(
    chunk: ProcessedChunk,
    config: IngestionConfig,
    figures_processed_count: int,
    *,
    source_uri: str = "",
) -> tuple[ProcessedChunk, int]:
    """Enrich a single chunk by replacing image placeholders via LiteLLM vision model.

    Respects ``config.vision_max_figures`` limit across the whole document.
    On VLM API failure the original chunk text is preserved and a warning is
    logged.

    The function re-scans the chunk text after each successful replacement so
    that match offsets remain valid (earlier replacements shift character
    positions for later matches).

    Args:
        chunk: The ``ProcessedChunk`` to enrich.
        config: Ingestion configuration (``vision_max_figures``,
            ``vision_timeout_seconds``, ``vision_max_tokens``,
            ``vision_temperature``).
        figures_processed_count: Figures already processed in preceding chunks.
        source_uri: URI (usually a file path) used to resolve relative image
            references inside the chunk text.

    Returns:
        ``(enriched_chunk_or_original, new_figures_processed_count)``
        Returns the original chunk unchanged if no placeholders are found,
        the per-document limit has already been reached, or VLM call fails.
    """
    # Early exit when the per-document figure budget is exhausted.
    if figures_processed_count >= config.vision_max_figures:
        return chunk, figures_processed_count

    placeholders = _find_image_placeholders(chunk.text)
    if not placeholders:
        return chunk, figures_processed_count

    # Derive a Path for resolving relative image references.
    source_path = Path(source_uri) if source_uri else Path(".")

    # Work on a mutable copy of the chunk text so we can replace in place.
    current_text = chunk.text
    new_count = figures_processed_count

    for match in placeholders:
        if new_count >= config.vision_max_figures:
            break

        # Re-compute the current match offset in the (possibly already modified)
        # text by re-scanning for the *original* placeholder string.
        original_placeholder = match.group(0)
        current_match_pos = current_text.find(original_placeholder)
        if current_match_pos == -1:
            # Placeholder was already replaced or is no longer present.
            continue

        # Rebuild a re.Match-compatible span by creating a fresh match on the
        # current text at the known position.
        recheck = _IMAGE_REF_PATTERN.search(current_text, current_match_pos)
        if recheck is None or recheck.group(0) != original_placeholder:
            continue

        # Extract image candidates for this single placeholder's markdown.
        remaining_budget = config.vision_max_figures - new_count
        try:
            candidates = _extract_image_candidates(
                original_placeholder,
                source_path=source_path,
                max_figures=remaining_budget,
                max_image_bytes=max(16_384, config.vision_max_image_bytes),
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "vlm_enrichment: failed to extract image candidate from placeholder "
                "%r (chunk index %s): %s",
                original_placeholder,
                chunk.metadata.get("chunk_index", "?"),
                exc,
            )
            continue

        if not candidates:
            # Placeholder could not be resolved to a readable image (e.g. missing
            # file, oversized, remote URL).  Leave it unchanged.
            continue

        candidate = candidates[0]
        try:
            description_obj = _describe_image(candidate, config)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "vlm_enrichment: VLM call failed for placeholder %r "
                "(chunk index %s): %s",
                original_placeholder,
                chunk.metadata.get("chunk_index", "?"),
                exc,
            )
            continue

        if description_obj is None:
            logger.warning(
                "vlm_enrichment: VLM returned no description for placeholder %r "
                "(chunk index %s)",
                original_placeholder,
                chunk.metadata.get("chunk_index", "?"),
            )
            continue

        description_text = description_obj.as_note()
        current_text = _replace_placeholder(current_text, recheck, description_text)
        new_count += 1

    if current_text == chunk.text:
        # Nothing changed — return the original object unchanged.
        return chunk, new_count

    # Return a shallow copy with updated text so the original is not mutated.
    enriched = copy.copy(chunk)
    enriched.text = current_text
    return enriched, new_count
