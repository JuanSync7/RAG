# @summary
# Ingestion pipeline shared helpers for keyword fallback, stage logging, and provenance mapping.
# Exports: extract_keywords_fallback, cross_refs, quality_score, append_processing_log, map_chunk_provenance
# Deps: difflib, logging, re, src.ingest.common.types, src.ingest.common.edit_log
# @end-summary

"""Shared helpers for ingestion pipeline nodes.

These helpers provide lightweight fallbacks for metadata extraction and support
consistent provenance and stage logging across ingestion nodes.
"""

from __future__ import annotations

import logging
import re
import difflib

from typing import Any, Optional

from src.ingest.common.edit_log import EditLog

logger = logging.getLogger("rag.ingest.common.shared")

# -- Pre-compiled regex patterns -----------------------------------------------

_KEYWORD_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b", re.UNICODE)

_CROSS_REF_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bDOC-\d{2,}\b", re.IGNORECASE), "document_id"),
    (re.compile(r"\bSection\s+\d+(?:\.\d+){0,3}\b", re.IGNORECASE), "section"),
    (re.compile(r"\bRFC\s+\d{3,5}\b", re.IGNORECASE), "standard"),
]

# -- Named constants for magic numbers -----------------------------------------

_QUALITY_BASE = 0.4
_QUALITY_BONUS = 0.2
_MIN_BONUS_LEN = 120
_QUALITY_PER_CHAR = 0.01
_QUALITY_MAX = 1.0

_ANCHOR_PREVIEW = 220
_PARA_PREVIEW = 600
_MIN_PARA_SIM = 0.45


def extract_keywords_fallback(text: str, max_keywords: int) -> list[str]:
    """Extract frequent keyword candidates when LLM metadata is unavailable.

    Args:
        text: Input text to analyze.
        max_keywords: Maximum number of keywords to return.

    Returns:
        A list of lowercase keyword candidates sorted by descending frequency.
    """
    words = _KEYWORD_PATTERN.findall(text.lower())
    freq: dict[str, int] = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    return [word for word, _ in ranked[:max_keywords]]


def cross_refs(text: str) -> list[dict[str, str]]:
    """Extract simple cross-reference patterns from source text.

    Args:
        text: Source text.

    Returns:
        A list of reference dictionaries with ``type`` and ``value`` keys.
    """
    refs: list[dict[str, str]] = []
    for pattern, ref_type in _CROSS_REF_PATTERNS:
        for match in pattern.findall(text):
            refs.append({"type": ref_type, "value": match})
    return refs


def quality_score(text: str) -> float:
    """Compute heuristic quality score for a chunk.

    Args:
        text: Chunk text.

    Returns:
        A score in ``(0.0, 1.0]`` where higher implies "more complete" content.
    """
    score = _QUALITY_BASE + (_QUALITY_BONUS if len(text) >= _MIN_BONUS_LEN else 0)
    score += min(_QUALITY_BONUS, len(re.findall(r"\d", text)) * _QUALITY_PER_CHAR)
    return min(_QUALITY_MAX, score)


def append_processing_log(state: dict[str, Any], message: str) -> list[str]:
    """Append a stage status message to the processing log.

    When verbose stage logging is enabled, the message is also emitted to the
    stage logger.

    Args:
        state: Ingestion pipeline state.
        message: Message to append.

    Returns:
        A new processing log list with the message appended.
    """
    runtime = state.get("runtime")
    config = getattr(runtime, "config", None)
    if bool(getattr(config, "verbose_stage_logs", False)):
        logger.info("source=%s stage=%s", state.get("source_name", "<unknown>"), message)
    return [*state.get("processing_log", []), message]


def _locate_span(haystack: str, needle: str, cursor: int) -> tuple[int, int, str]:
    """Locate a text span with cursor-first exact search and fallback.

    Args:
        haystack: Text to search within.
        needle: Text span to locate.
        cursor: Preferred search start offset.

    Returns:
        Tuple of ``(start, end, method)`` where ``start`` and ``end`` are
        character offsets and ``method`` indicates the match strategy used.
    """
    if not haystack or not needle:
        return -1, -1, "missing"
    start = haystack.find(needle, max(cursor, 0))
    if start >= 0:
        return start, start + len(needle), "exact_cursor"
    start = haystack.find(needle)
    if start >= 0:
        return start, start + len(needle), "exact_global"
    return -1, -1, "not_found"


def _best_paragraph_span(text: str, anchor: str) -> tuple[int, int, float]:
    """Return best matching paragraph span and similarity score.

    Args:
        text: Full text to search.
        anchor: Anchor snippet to match against paragraphs.

    Returns:
        Tuple of ``(start, end, ratio)`` where ``ratio`` is a similarity score
        in ``[0.0, 1.0]``.
    """
    if not text.strip() or not anchor.strip():
        return -1, -1, 0.0
    best_start = best_end = -1
    best_ratio = 0.0
    offset = 0
    for paragraph in text.split("\n\n"):
        para = paragraph.strip()
        if not para:
            offset += len(paragraph) + 2
            continue
        ratio = difflib.SequenceMatcher(
            None,
            anchor[:_ANCHOR_PREVIEW].lower(),
            para[:_PARA_PREVIEW].lower(),
        ).ratio()
        if ratio > best_ratio:
            idx = text.find(paragraph, offset)
            if idx >= 0:
                best_ratio = ratio
                best_start = idx
                best_end = idx + len(paragraph)
            if best_ratio > 0.85:
                break  # high-confidence match; skip remaining paragraphs
        offset += len(paragraph) + 2
    return best_start, best_end, best_ratio


def map_chunk_provenance(
    chunk_text: str,
    original_text: str,
    refactored_text: str,
    original_cursor: int,
    refactored_cursor: int,
    edit_log: Optional[EditLog] = None,
) -> tuple[dict[str, object], int, int]:
    """Map a chunk to refactored and original text spans with confidence.

    This function attempts exact matching first and falls back to weaker
    heuristics (e.g., paragraph similarity) when exact mapping fails.

    When ``edit_log`` is supplied and the chunk is exactly located in the
    refactored text, the original-side offsets are projected via the edit log
    instead of substring-searched in the original — this is deterministic and
    avoids the fuzzy fallback for chunks the LLM rewrote.

    Args:
        chunk_text: The chunk text to map.
        original_text: The original extracted text prior to refactoring.
        refactored_text: The refactored text used for chunking.
        original_cursor: Cursor offset hint for original text to speed up search.
        refactored_cursor: Cursor offset hint for refactored text to speed up
            search.
        edit_log: Optional pre-built diff between original and refactored. When
            provided, takes precedence over substring/fuzzy original-side
            mapping for any chunk found in refactored text.

    Returns:
        Tuple of ``(provenance, next_original_cursor, next_refactored_cursor)``
        where ``provenance`` contains offsets and confidence metadata.
    """
    ref_start, ref_end, ref_method = _locate_span(
        refactored_text,
        chunk_text,
        refactored_cursor,
    )
    next_ref_cursor = ref_end if ref_start >= 0 else refactored_cursor

    orig_start = orig_end = -1
    orig_method = "unmapped"
    confidence = 0.0

    # Edit-log path: exact projection from refactored offsets to original offsets.
    if edit_log is not None and ref_start >= 0:
        proj_start, proj_end, proj_conf = edit_log.map_ref_to_orig(ref_start, ref_end)
        if proj_start >= 0:
            orig_start, orig_end = proj_start, proj_end
            orig_method = "edit_log"
            confidence = proj_conf

    # Substring fallback when no edit log or projection failed.
    if orig_start < 0:
        orig_start, orig_end, orig_method = _locate_span(original_text, chunk_text, original_cursor)
        if orig_start >= 0:
            confidence = 1.0

    if orig_start < 0 and ref_start >= 0:
        ref_slice = refactored_text[ref_start:ref_end]
        orig_start, orig_end, orig_method = _locate_span(original_text, ref_slice, original_cursor)
        if orig_start >= 0:
            confidence = 0.85

    if orig_start < 0:
        para_start, para_end, ratio = _best_paragraph_span(original_text, chunk_text)
        if para_start >= 0 and ratio >= _MIN_PARA_SIM:
            orig_start, orig_end = para_start, para_end
            orig_method = "paragraph_fuzzy"
            confidence = round(min(0.79, ratio), 3)
        else:
            orig_method = "unmapped"

    next_orig_cursor = orig_end if orig_start >= 0 else original_cursor
    provenance = {
        "refactored_char_start": ref_start,
        "refactored_char_end": ref_end,
        "original_char_start": orig_start,
        "original_char_end": orig_end,
        "provenance_method": f"{ref_method}->{orig_method}",
        "provenance_confidence": confidence,
    }
    return provenance, next_orig_cursor, next_ref_cursor


__all__ = [
    "extract_keywords_fallback",
    "cross_refs",
    "quality_score",
    "append_processing_log",
    "map_chunk_provenance",
]
