# @summary
# LangGraph node for parser-abstraction chunk generation with legacy markdown fallback.
# Primary path: parse_result + parser_instance from state → parser.chunk() or
#   chunk_with_markdown() depending on config.chunker override.
# Legacy fallback: when no parse_result is in state, falls back to markdown chunking
#   on refactored_text/cleaned_text (pre-Phase 3.2 behaviour).
# Exports: chunking_node, _normalize_chunk_text
# Deps: unicodedata, re, src.ingest.embedding.state, src.ingest.common.schemas,
#       src.ingest.common.shared, src.ingest.support.parser_base,
#       src.ingest.support.document, src.ingest.support.markdown
# @end-summary

"""Chunking node implementation.

Selects between the parser-abstraction path (parse_result + parser_instance in
state) and the legacy markdown fallback path based on what structure_detection_node
placed in state.
"""

from __future__ import annotations

import logging
import re as _re
import time
import unicodedata
from typing import Any

from src.ingest.support import (
    extract_metadata,
    metadata_to_dict,
)
from src.ingest.common import ProcessedChunk
from src.ingest.support import (
    _build_section_metadata,
    chunk_markdown,
    normalize_headings_to_markdown,
)
from src.ingest.common import append_processing_log
from src.ingest.embedding.state import EmbeddingPipelineState

logger = logging.getLogger("rag.ingest.pipeline.chunking")

# Pre-compiled regex: strips C0/C1 control characters except \n (0x0a), \t (0x09),
# and \r (0x0d).  Range breakdown:
#   \x00-\x08  — NUL through BS
#   \x0b       — VT (vertical tab)
#   \x0c       — FF (form feed)
#   \x0e-\x1f  — SO through US
#   \x7f       — DEL
_CONTROL_CHAR_RE = _re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _normalize_chunk_text(text: str) -> str:
    """Apply NFC unicode normalization and remove control characters.

    Args:
        text: Raw chunk text.

    Returns:
        NFC-normalized text with C0/C1 control characters removed.
        Newlines (0x0a) and carriage returns (0x0d) are preserved.
    """
    normalized = unicodedata.normalize("NFC", text)
    return _CONTROL_CHAR_RE.sub("", normalized)


def chunking_node(state: EmbeddingPipelineState) -> dict[str, Any]:
    """Split document into chunks using parser abstraction or legacy markdown fallback.

    Path selection:
    - If ``state["parse_result"]`` and ``state["parser_instance"]`` are both present
      (set by structure_detection_node via the ParserRegistry), the parser-abstraction
      path is used:
      - ``config.chunker == "markdown"`` → ``chunk_with_markdown(parse_result, config)``
      - ``config.chunker == "native"`` (default) → ``parser_instance.chunk(parse_result)``
        with auto-fallback to markdown on native chunking failure.
    - Otherwise the legacy path is used: markdown chunking on ``refactored_text`` or
      ``cleaned_text`` (identical to pre-Phase 3.2 behaviour).

    Args:
        state: Embedding pipeline state.

    Returns:
        Partial state update: {"chunks": list[ProcessedChunk], "processing_log": updated_log}
    """
    t0 = time.monotonic()
    config = state["runtime"].config
    try:
        base_metadata = metadata_to_dict(
            extract_metadata(state["raw_text"], state["source_name"])
        )
        base_metadata.update(
            {
                "source": state["source_name"],
                "source_uri": state["source_uri"],
                "source_key": state["source_key"],
                "source_id": state["source_id"],
                "connector": state["connector"],
                "source_version": state["source_version"],
            }
        )

        parse_result = state.get("parse_result")
        parser_instance = state.get("parser_instance")

        if parse_result is not None and parser_instance is not None:
            # ── Parser-abstraction path (Phase 3.2) ───────────────────────
            if config.chunker == "markdown":
                from src.ingest.support.parser_base import chunk_with_markdown
                raw_parser_chunks = chunk_with_markdown(parse_result, config)
                processing_log = append_processing_log(state, "chunking:markdown_override")
            else:
                # config.chunker == "native" (default)
                # Optionally hand the parser a doc-store client + source identity
                # so figure-image uploads (when config.store_figures_in_db=True)
                # land at a stable backend key. No-op when the parser doesn't
                # implement configure_storage or when either dep is missing.
                configure_storage = getattr(parser_instance, "configure_storage", None)
                if callable(configure_storage):
                    configure_storage(
                        source_key=state.get("source_key", ""),
                        doc_store_client=getattr(state["runtime"], "db_client", None),
                    )
                try:
                    raw_parser_chunks = parser_instance.chunk(parse_result)
                    processing_log = append_processing_log(state, "chunking:native_ok")
                except Exception as exc:
                    logger.error(
                        "Native chunking failed for source=%s: %s — "
                        "falling back to markdown",
                        state.get("source_name", "<unknown>"), exc,
                    )
                    from src.ingest.support.parser_base import chunk_with_markdown
                    raw_parser_chunks = chunk_with_markdown(parse_result, config)
                    processing_log = append_processing_log(
                        state, "chunking:fallback_to_markdown"
                    )

            # Map Chunk → ProcessedChunk preserving Weaviate payload shape.
            total = len(raw_parser_chunks)
            chunks: list[ProcessedChunk] = [
                ProcessedChunk(
                    text=_normalize_chunk_text(c.text),
                    metadata={
                        **base_metadata,
                        "section_path": c.section_path,
                        "heading": c.heading,
                        "heading_level": c.heading_level,
                        "chunk_index": c.chunk_index,
                        "total_chunks": total,
                        **c.extra_metadata,
                    },
                )
                for c in raw_parser_chunks
            ]

        else:
            # ── Legacy fallback (no parse_result in state) ─────────────────
            # Backward-compat: markdown chunking on refactored_text/cleaned_text.
            # Preserves pre-Phase 3.2 behaviour exactly.
            chunks = _chunk_with_markdown_legacy(state, config, base_metadata)
            processing_log = append_processing_log(state, "chunking:legacy_markdown")

    except Exception as exc:
        logger.error("chunking failed source=%s: %s", state.get("source_name", ""), exc, exc_info=True)
        return {
            "errors": state.get("errors", []) + [f"chunking:{exc}"],
            "processing_log": append_processing_log(state, "chunking:error"),
        }

    logger.info("chunking complete: source=%s chunks=%d", state["source_name"], len(chunks))
    logger.debug("chunking_node completed in %.3fs", time.monotonic() - t0)
    return {
        "chunks": chunks,
        "processing_log": processing_log,
    }


def _chunk_with_markdown_legacy(
    state: EmbeddingPipelineState,
    config: Any,
    base_metadata: dict[str, Any],
) -> list[ProcessedChunk]:
    """Chunk markdown text using MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter.

    Behaviorally identical to the pre-Phase-3.2 chunking_node body. Preserves
    semantic_chunking flag, heading normalization, and ProcessedChunk metadata shape.

    Args:
        state: Pipeline state; uses refactored_text or cleaned_text for chunking.
        config: IngestionConfig; controls semantic_chunking, chunk_size, chunk_overlap.
        base_metadata: Pre-built source metadata dict.

    Returns:
        List of ProcessedChunk objects with full metadata.

    Raises:
        Any exception from the markdown splitters (propagates; markdown path
        failure is fatal for this document).
    """
    text_for_chunking = normalize_headings_to_markdown(
        state.get("refactored_text") or state.get("cleaned_text", "")
    )

    if config.semantic_chunking:
        raw_chunks = chunk_markdown(
            text_for_chunking,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            embedder=state["runtime"].embedder,
        )
    else:
        raw_chunks = chunk_markdown(
            text_for_chunking,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            embedder=None,
        )

    total_chunks = len(raw_chunks)
    chunks = [
        ProcessedChunk(
            text=_normalize_chunk_text(chunk["text"]),
            metadata={
                **base_metadata,
                **_build_section_metadata(chunk.get("header_metadata", {})),
                "chunk_index": idx,
                "total_chunks": total_chunks,
            },
        )
        for idx, chunk in enumerate(raw_chunks)
    ]
    return chunks
