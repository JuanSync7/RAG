# @summary
# Ingestion pipeline shared helpers for keyword fallback, stage logging, and provenance mapping.
# Exports: extract_keywords_fallback, cross_refs, quality_score, append_processing_log, map_chunk_provenance, chunk_body_text
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
from config.settings import (
    RAG_INGEST_QUALITY_BASE,
    RAG_INGEST_QUALITY_BONUS,
    RAG_INGEST_QUALITY_MIN_BONUS_LEN,
    RAG_INGEST_QUALITY_PER_CHAR,
    RAG_INGEST_QUALITY_MAX,
    RAG_INGEST_ANCHOR_PREVIEW_LEN,
    RAG_INGEST_PARA_PREVIEW_LEN,
    RAG_INGEST_MIN_PARA_SIMILARITY,
)

logger = logging.getLogger("rag.ingest.common.shared")

# -- Pre-compiled regex patterns -----------------------------------------------

_KEYWORD_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b", re.UNICODE)

_CROSS_REF_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bDOC-\d{2,}\b", re.IGNORECASE), "document_id"),
    (re.compile(r"\bSection\s+\d+(?:\.\d+){0,3}\b", re.IGNORECASE), "section"),
    (re.compile(r"\bRFC\s+\d{3,5}\b", re.IGNORECASE), "standard"),
    # Datasheet-style §-section refs: "§3", "§ 3.1", "§3.1.4".
    # Require a digit immediately after the symbol (with optional single space)
    # so prose like "§abc" does not match.
    (re.compile(r"§\s?\d+(?:\.\d+){0,3}\b"), "section_symbol"),
    # Table refs: "Table 5", "Table 5-2", "Table 5.2", "Table 10.3". Word
    # boundary on "Table" prevents matches inside "tablespoon". The trailing
    # negative lookahead `(?![.-]\d)` rejects 3-segment forms like
    # "Table 4.1.2" or "Table 4-1.2" which in datasheets are section
    # headings (e.g. "Table 4.1.2 Memory Organization"), not table
    # citations — table numbers are at most two segments. Mirrors the
    # figure-pattern fix in PR #104 as a preventative parity tightening.
    (re.compile(r"\bTable\s+\d+(?:[.-]\d+)?\b(?![.-]\d)", re.IGNORECASE), "table"),
    # Figure refs: "Figure 7-1", "Fig 7-1", "Fig. 7-1", "Figure 10.3". Word
    # boundary on the head + required digit blocks "figurine". The trailing
    # negative lookahead `(?![.-]\d)` rejects 3-segment forms like
    # "Figure 4.1.2" or "Figure 4-1.2" which in datasheets are section
    # headings (e.g. "Figure 4.1.2 Memory Organization"), not figure
    # citations — figure numbers are at most two segments.
    (
        re.compile(
            r"\b(?:Figure|Fig\.?)\s+\d+(?:[.-]\d+)?\b(?![.-]\d)",
            re.IGNORECASE,
        ),
        "figure",
    ),
    # Appendix refs: "Appendix A", "Appendix B.2", "appendix C". Requires a
    # capital-letter label so plural "appendices" alone does not match.
    (re.compile(r"\bAppendix\s+[A-Z](?:\.\d+)?\b", re.IGNORECASE), "appendix"),
]


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
    score = RAG_INGEST_QUALITY_BASE + (RAG_INGEST_QUALITY_BONUS if len(text) >= RAG_INGEST_QUALITY_MIN_BONUS_LEN else 0)
    score += min(RAG_INGEST_QUALITY_BONUS, len(re.findall(r"\d", text)) * RAG_INGEST_QUALITY_PER_CHAR)
    return min(RAG_INGEST_QUALITY_MAX, score)


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
            anchor[:RAG_INGEST_ANCHOR_PREVIEW_LEN].lower(),
            para[:RAG_INGEST_PARA_PREVIEW_LEN].lower(),
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
    fuzzy_fallback: bool = True,
) -> tuple[dict[str, object], int, int]:
    """Map a chunk to refactored and original text spans with confidence.

    This function attempts exact matching first and falls back to weaker
    heuristics (e.g., paragraph similarity) when exact mapping fails.

    ``fuzzy_fallback`` controls the last-resort ``_best_paragraph_span`` step,
    which runs ``difflib.SequenceMatcher`` against EVERY paragraph of
    ``original_text``. That is O(doc) per chunk — pathological when called once
    per chunk over thousands of chunks (a 1,600-chunk spec stalled
    ``chunk_enrichment`` for ~22 min). It is also low-value for HybridChunker
    output, whose chunk text is re-serialized (tables→triplets, whitespace
    reflowed) and so rarely matches the source verbatim anyway. Callers that map
    many chunks should pass ``fuzzy_fallback=False``: exact-find still maps prose
    chunks (high confidence); non-matching chunks are left ``unmapped`` rather
    than fuzzily approximated. Page-level citation is unaffected (it comes from
    ``page_ref``, not these char offsets).

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

    if orig_start < 0 and fuzzy_fallback:
        para_start, para_end, ratio = _best_paragraph_span(original_text, chunk_text)
        if para_start >= 0 and ratio >= RAG_INGEST_MIN_PARA_SIMILARITY:
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


def chunk_body_text(text: str, heading_path: Optional[list] = None) -> str:
    """Recover a chunk's raw BODY from HybridChunker-contextualized text.

    docling_core's ``BaseChunker.contextualize()`` prepends the section heading
    breadcrumb to the embedded text as ``"\\n".join(headings) + "\\n" + body``
    (see ``docling_core/transforms/chunker/base.py``; only ``headings`` is
    embedded — ``doc_items``/``origin``/``schema_name``/``version`` are excluded).
    Length/quality gates and source-provenance mapping must measure the BODY,
    not the heading-padded embedded text — otherwise:

    * a 9-char body under a 3-level heading reads as ~60 chars and a near-empty
      stub survives the min-length floor (the "title-only chunk" pathology), and
    * provenance ``str.find()`` never matches (the breadcrumb is not contiguous
      with the body in the source), forcing the O(doc) ``_best_paragraph_span``
      fuzzy fallback for *every* chunk — a quadratic ingest-time cost.

    This reverses the contextualization deterministically. It returns ``text``
    unchanged when ``heading_path`` is empty or the breadcrumb prefix is absent
    (table/figure chunks built without contextualize, the legacy markdown path,
    or text that was never contextualized), so callers may always substitute the
    result for ``text`` with no behavioural change on non-contextualized chunks.
    """
    if not heading_path:
        return text
    prefix = "\n".join(heading_path) + "\n"
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


# ---------------------------------------------------------------------------
# Navigational / front-matter chunk detection (shared by ingest-time drop and
# query-time rerank filtering, so emission and filtering stay symmetric). Hoisted
# here from retrieval.rag_chain — both sides import from this neutral module so
# ingest never depends on retrieval (the dependency only ever runs the other way).
# ---------------------------------------------------------------------------

# Dotted-leader runs (>=4 dots) — the fingerprint of a table-of-contents / index
# entry ("A1.3 AXI Architecture ............ A1-22"). Scored by density, not length.
_TOC_LEADER_RE = re.compile(r"\.{4,}")
# Front-matter / navigation pointer idioms used by spec chapter & part intros
# ("Read this chapter for a description of ...", "This specification describes ...").
# High precision: these phrases point AT content elsewhere, never state content.
# Only applied to short chunks (see ``is_navigational`` length cap). Widened beyond
# the original chapter|section|part|preface|book set to also catch
# specification|document|appendix|manual ("This specification describes the AMBA
# AXI protocol" was a known false-negative).
_NAV_PHRASE_RE = re.compile(
    r"(?:read\s+this\s+(?:chapter|section|part|book|specification|manual)?\s*for\b"
    r"|for\s+(?:a\s+)?(?:full\s+)?descriptions?\s+of\b"
    r"|(?:about\s+)?this\s+(?:chapter|section|part|preface|book|specification|document|appendix|manual)\s+(?:describes|introduces|contains|provides)\b"
    r"|see\s+(?:chapter|section|page)\s+[\w.\-]+\s+on\s+page\b)",
    re.IGNORECASE,
)


def toc_leader_ratio(text: str) -> float:
    """Fraction of characters that belong to dotted-leader runs (>=4 dots)."""
    if not text:
        return 0.0
    leader_chars = sum(len(m.group(0)) for m in _TOC_LEADER_RE.finditer(text))
    return leader_chars / max(1, len(text))


def is_navigational(text: str, max_chars: int = 320) -> bool:
    """True if a chunk is *navigational* — a table-of-contents / index entry or a
    chapter/part front-matter pointer stub — rather than answer content.

    Navigational chunks have real-word bodies (so they pass a thin/heading test)
    but only *point at* content ("Read this chapter for a description of the basic
    AXI transactions") or list it ("A1.3 AXI Architecture ......... A1-22"). They
    reliably beat real body chunks on both dense similarity and the cross-encoder
    reranker, so they are dropped at ingest (index hygiene for ALL consumers) and
    again before rerank (defense-in-depth). Measure on the same (contextualized)
    text both sides see, so the leader-ratio denominator stays calibrated.
    """
    t = (text or "").strip()
    if not t:
        return False
    # (A) ToC / index page: dense dotted leaders, or a majority of leader lines.
    if toc_leader_ratio(t) >= 0.08:
        return True
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines and sum(1 for ln in lines if _TOC_LEADER_RE.search(ln)) / len(lines) >= 0.4:
        return True
    # (B) Short front-matter pointer stub. Length cap keeps real body chunks that
    #     merely cross-reference another section (those are much longer).
    if len(t) <= max_chars and _NAV_PHRASE_RE.search(t):
        return True
    return False


__all__ = [
    "extract_keywords_fallback",
    "cross_refs",
    "quality_score",
    "append_processing_log",
    "map_chunk_provenance",
    "chunk_body_text",
    "toc_leader_ratio",
    "is_navigational",
]
