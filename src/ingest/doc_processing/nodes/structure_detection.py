# @summary
# LangGraph node for structure cues extraction via the parser abstraction.
# Dispatches to ParserRegistry when available; falls back to legacy Docling path
# when registry is absent (backward-compat) and to regex heuristics on non-strict failure.
# Stores parse_result and parser_instance on state for chunking_node consumption.
# Exports: structure_detection_node
# Deps: src.ingest.support.docling, src.ingest.support.parser_base,
#       src.ingest.common.shared, src.ingest.doc_processing.state
# @end-summary

"""Structure-detection node implementation."""

from __future__ import annotations

import logging
import re
import time
import warnings
from pathlib import Path
from typing import Any

from src.ingest.common import append_processing_log
from src.ingest.doc_processing.state import DocumentProcessingState
from src.ingest.support import parse_with_docling

logger = logging.getLogger("rag.ingest.docproc.structure_detection")

_FIGURE_PATTERN = re.compile(r"\b(?:Figure|Fig\.)\s*\d+[A-Za-z]?\b", re.IGNORECASE)
_HEADING_PATTERN = re.compile(
    r"^\s*(?:#{1,6}\s+.+|\d+(?:\.\d+)*\.?\s+[A-Z].+)$", re.MULTILINE
)
_MAX_FIGURES = 32


def structure_detection_node(state: DocumentProcessingState) -> dict[str, Any]:
    """Extract structural signals via parser abstraction or legacy fallback.

    When a ``ParserRegistry`` is attached to ``runtime.parser_registry``, this
    node dispatches to the appropriate parser via the registry, stores the resulting
    ``ParseResult`` and ``parser_instance`` on state for downstream ``chunking_node``
    consumption, and derives figure/heading signals from ``ParseResult`` fields.

    When no registry is present (backward-compat path), the node falls back to the
    legacy ``parse_with_docling()`` call and propagates ``docling_document`` on state.
    A deprecation warning is emitted on this path.

    On non-strict failure the node falls back to regex heuristics; on strict failure
    it returns an error payload with ``should_skip=True``.

    Args:
        state: Document processing pipeline state.

    Returns:
        Partial state update containing:
        - Updated ``raw_text`` (may be parser-generated markdown)
        - A ``structure`` dictionary with figure/heading signals and
          ``parser_strategy`` routing signal
        - Updated ``processing_log``
        - ``parse_result`` and ``parser_instance`` (parser-abstraction path only)
        - ``docling_document`` (legacy Docling path only, for backward compat)

        In strict Docling mode, failures return an error payload with
        ``should_skip=True`` to short-circuit the workflow.
    """
    t0 = time.monotonic()
    config = state["runtime"].config
    registry = getattr(state["runtime"], "parser_registry", None)
    raw_text = state["raw_text"]

    figures: list[str] = []
    headings: list[str] = []
    parsed_text = raw_text
    parse_result = None
    parser_instance = None
    parser_strategy = "unknown"

    if registry is not None:
        # ── New parser-abstraction path ─────────────────────────────────────
        try:
            parser_instance = registry.get_parser(
                Path(state["source_path"]), config
            )
            parse_result = parser_instance.parse(
                Path(state["source_path"]), config
            )
            parsed_text = parse_result.markdown
            headings = list(parse_result.headings)
            figures = (
                [f"Figure {i + 1}" for i in range(10)]
                if parse_result.has_figures
                else []
            )
            # Derive a short strategy label from the parser class name.
            parser_strategy = (
                type(parser_instance).__name__
                .lower()
                .replace("parser", "")
                .strip()
            )
        except Exception as exc:
            if config.docling_strict:
                return {
                    "errors": [f"parser_failed:{state['source_name']}:{exc}"],
                    "should_skip": True,
                    "processing_log": append_processing_log(
                        state, "structure_detection:failed"
                    ),
                }
            # Non-strict fallback: regex heuristics; clear partial results.
            logger.warning(
                "Parser failed for source=%s: %s — falling back to regex heuristics",
                state.get("source_name", "<unknown>"), exc,
            )
            parse_result = None
            parser_instance = None
            parser_strategy = "regex_fallback"
            figures = _FIGURE_PATTERN.findall(raw_text)
            headings = _HEADING_PATTERN.findall(raw_text)

    elif config.enable_docling_parser:
        # ── Legacy Docling path (registry not yet initialized) ─────────────
        warnings.warn(
            "structure_detection_node: parser_registry is not set on Runtime. "
            "Falling back to legacy parse_with_docling() call. "
            "This path is deprecated; initialize ParserRegistry in impl.py.",
            DeprecationWarning,
            stacklevel=2,
        )

        docling_doc = None
        docling_document_available = False

        try:
            parsed = parse_with_docling(
                Path(state["source_path"]),
                parser_model=config.docling_model,
                artifacts_path=config.docling_artifacts_path,
                vlm_mode=config.vlm_mode,
                generate_page_images=config.generate_page_images,
            )
            parsed_text = parsed.text_markdown
            figures = list(parsed.figures)
            headings = list(parsed.headings)
            docling_doc = parsed.docling_document
            docling_document_available = True
            parser_strategy = "docling_legacy"
        except Exception as exc:
            _is_format_error = (
                "format not allowed" in str(exc).lower()
                or "File format not" in str(exc)
            )
            if _is_format_error:
                figures = _FIGURE_PATTERN.findall(raw_text)
                headings = _HEADING_PATTERN.findall(raw_text)
            elif config.docling_strict:
                return {
                    "errors": [f"docling_parse_failed:{state['source_name']}:{exc}"],
                    "should_skip": True,
                    "processing_log": append_processing_log(
                        state, "structure_detection:failed"
                    ),
                }
            else:
                figures = _FIGURE_PATTERN.findall(raw_text)
                headings = _HEADING_PATTERN.findall(raw_text)

        structure = {
            "has_figures": bool(figures),
            "figures": figures[:_MAX_FIGURES],
            "heading_count": len(headings),
            "docling_enabled": bool(config.enable_docling_parser),
            "docling_model": str(config.docling_model),
            "docling_document_available": docling_document_available,
            "parser_strategy": parser_strategy,
        }
        update: dict[str, Any] = {
            "raw_text": parsed_text,
            "structure": structure,
            "processing_log": append_processing_log(state, "structure_detection:ok"),
        }
        if docling_document_available:
            update["docling_document"] = docling_doc
        return update

    else:
        # ── No registry, no Docling: pure regex fallback ────────────────────
        figures = _FIGURE_PATTERN.findall(raw_text)
        headings = _HEADING_PATTERN.findall(raw_text)
        parser_strategy = "regex"

    structure = {
        "has_figures": bool(figures),
        "figures": figures[:_MAX_FIGURES],
        "heading_count": len(headings),
        "docling_enabled": bool(config.enable_docling_parser),
        "docling_model": str(config.docling_model),
        "docling_document_available": False,
        "parser_strategy": parser_strategy,
    }

    update = {
        "raw_text": parsed_text,
        "structure": structure,
        "processing_log": append_processing_log(state, "structure_detection:ok"),
    }

    if parse_result is not None:
        update["parse_result"] = parse_result
    if parser_instance is not None:
        update["parser_instance"] = parser_instance

    logger.info("structure_detection complete: source=%s strategy=%s", state["source_name"], structure.get("strategy", "unknown"))
    logger.debug("structure_detection_node completed in %.3fs", time.monotonic() - t0)
    return update
