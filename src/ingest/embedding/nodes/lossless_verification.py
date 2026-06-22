# @summary
# LangGraph node that verifies the emitted chunks losslessly cover the parsed
# document (the golden). Warn-by-default; RAG_INGESTION_LOSSLESS_STRICT escalates
# a coverage shortfall to a hard failure of the document.
# Exports: lossless_verification_node
# Deps: embedding.state, support.lossless, common.chunk_body_text
# @end-summary

"""Lossless-verification node implementation.

Runs after ``quality_validation`` (chunks finalized) and before storage. It
compares the chunk set against the GOLDEN source the chunks were derived from —
the parsed Docling markdown (``parse_result.text_markdown``) on the native path,
falling back to ``cleaned_text``/``raw_text`` when no parse is present (legacy
path). Coverage is measured with content word-shingles (see
:mod:`src.ingest.support.lossless`), so heading breadcrumbs, restated table
headers, and markdown formatting differences do not register as loss — only
genuinely dropped content does.

Behaviour is config-gated:

* ``verify_lossless`` (default True) — run at all.
* ``lossless_min_coverage`` (default 0.98) — coverage floor.
* ``lossless_strict`` (default False) — below the floor, WARN (record + log)
  by default, or RAISE (fail the document) when strict.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.ingest.common import append_processing_log, chunk_body_text
from src.ingest.common.observability import node_span
from src.ingest.embedding.state import EmbeddingPipelineState
from src.ingest.support.lossless import coverage_report

logger = logging.getLogger("rag.ingest.embedding.lossless_verification")


def _select_golden(state: EmbeddingPipelineState) -> tuple[str, str]:
    """Return ``(golden_text, source_label)``.

    Prefer the parsed markdown the native chunks are derived from; fall back to
    ``cleaned_text`` (legacy path's chunk source) then ``raw_text`` so the
    reference is always the text the chunks actually came from — measuring
    against a differently-derived text would manufacture false misses.
    """
    parse_result = state.get("parse_result")
    text_markdown = getattr(parse_result, "text_markdown", "") if parse_result is not None else ""
    if text_markdown and text_markdown.strip():
        return text_markdown, "parse_result.text_markdown"
    cleaned = state.get("cleaned_text") or ""
    if cleaned.strip():
        return cleaned, "cleaned_text"
    return (state.get("raw_text", "") or ""), "raw_text"


@node_span("lossless_verification")
def lossless_verification_node(state: EmbeddingPipelineState) -> dict[str, Any]:
    """Verify chunk coverage of the golden source; warn or (strict) fail.

    Args:
        state: Ingestion pipeline state (chunks finalized by quality_validation).

    Returns:
        Partial state update with an updated ``processing_log``. Raises
        ``ValueError`` only when strict mode is on AND coverage is below the
        configured floor.
    """
    t0 = time.monotonic()
    config = state["runtime"].config
    if not getattr(config, "verify_lossless", False):
        return {
            "processing_log": append_processing_log(state, "lossless_verification:skipped")
        }

    golden, golden_label = _select_golden(state)
    chunks = state.get("chunks") or []
    if not golden.strip() or not chunks:
        return {
            "processing_log": append_processing_log(
                state, "lossless_verification:skipped(no-golden-or-chunks)"
            )
        }

    # Measure the chunk BODY (breadcrumb stripped — a no-op on non-contextualized
    # chunks), so the restated heading prefix on every chunk does not inflate
    # coverage and table_row blocks contribute their caption+header+rows content.
    bodies = [chunk_body_text(c.text, c.metadata.get("heading_path")) for c in chunks]
    report = coverage_report(golden, bodies)

    min_cov = float(getattr(config, "lossless_min_coverage", 0.98))
    strict = bool(getattr(config, "lossless_strict", False))
    source = state.get("source_name", "") or state.get("source_key", "")

    base = (
        f"coverage={report.coverage:.4f} "
        f"covered_words={report.covered_words}/{report.golden_words} "
        f"chunks={len(chunks)} golden={golden_label}"
    )
    logger.debug("lossless_verification_node completed in %.3fs", time.monotonic() - t0)

    if report.coverage < min_cov:
        top = report.missing_spans[0] if report.missing_spans else None
        detail = (
            f" largest_missing={top.length}w@word{top.start_word} :: {top.snippet!r}"
            if top
            else ""
        )
        if strict:
            logger.error(
                "lossless_verification FAILED (strict) source=%s %s threshold=%.4f%s",
                source, base, min_cov, detail,
            )
            raise ValueError(
                f"Lossless verification failed for {source!r}: "
                f"{base} < threshold {min_cov:.4f}{detail}"
            )
        logger.warning(
            "lossless_verification BELOW threshold source=%s %s threshold=%.4f%s",
            source, base, min_cov, detail,
        )
        status = f"below({report.coverage:.4f})"
    else:
        logger.info("lossless_verification ok source=%s %s", source, base)
        status = f"ok({report.coverage:.4f})"

    return {
        "processing_log": append_processing_log(
            state, f"lossless_verification:{status}"
        )
    }
