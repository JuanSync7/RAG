# @summary
# LangGraph node for optional multimodal note synthesis from detected figure mentions.
# Exports: multimodal_processing_node
# Deps: src.ingest.common.shared, src.ingest.doc_processing.state, src.ingest.support.vision
# @end-summary

"""Multimodal-processing node implementation."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("rag.ingest.docproc.multimodal_processing")

from src.ingest.common import append_processing_log
from src.ingest.doc_processing.state import DocumentProcessingState
from src.ingest.support import generate_vision_notes


def multimodal_processing_node(state: DocumentProcessingState) -> dict[str, Any]:
    """Generate multimodal notes when figure references are detected and enabled.

    This node adds lightweight figure notes when a document references figures,
    and optionally enriches those notes with VLM-generated captions/OCR output.

    Args:
        state: Document processing pipeline state.

    Returns:
        Partial state update containing ``multimodal_notes``, optional
        ``structure`` telemetry fields (vision model info), and an updated
        ``processing_log``. In strict vision mode, failures return an error
        payload with ``should_skip=True``.
    """
    t0 = time.monotonic()
    config = state["runtime"].config
    has_figures = state["structure"].get("has_figures", False)
    if not config.enable_multimodal_processing or not has_figures:
        return {
            "processing_log": append_processing_log(
                state, "multimodal_processing:skipped"
            )
        }
    figures = list(state["structure"].get("figures", []))
    notes = [f"{figure}: referenced in text" for figure in figures]
    described_count = 0
    if config.enable_vision_processing:
        try:
            vision_notes, described_count = generate_vision_notes(
                state["raw_text"],
                source_path=Path(state["source_path"]),
                config=config,
            )
            if vision_notes:
                # Prepend vision notes before the plain-text figure notes,
                # keeping any overflow plain notes that vision did not cover.
                notes = vision_notes + notes[len(vision_notes) :]
        except Exception as exc:
            if config.vision_strict:
                return {
                    "errors": [f"vision_processing_failed:{state['source_name']}:{exc}"],
                    "should_skip": True,
                    "processing_log": append_processing_log(
                        state,
                        "multimodal_processing:failed",
                    ),
                }
            else:
                logger.warning(
                    "vision_processing failed (non-strict) source=%s: %s",
                    state.get("source_name", ""),
                    exc,
                    exc_info=True,
                )

    structure = dict(state.get("structure", {}))
    if config.enable_vision_processing:
        structure["vision_provider"] = config.vision_provider
        structure["vision_model"] = config.vision_model
        structure["vision_described_count"] = described_count

    logger.info("multimodal_processing complete: source=%s notes=%d", state["source_name"], len(notes))
    logger.debug("multimodal_processing_node completed in %.3fs", time.monotonic() - t0)
    return {
        "multimodal_notes": notes,
        "structure": structure,
        "processing_log": append_processing_log(state, "multimodal_processing:ok"),
    }
