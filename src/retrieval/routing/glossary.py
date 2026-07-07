# @summary
# Shared hardware-protocol glossary plus deterministic, side-effect-light
# helpers for document routing: a lexical comparison-intent detector and a
# term canonicalizer that resolves informal aliases to canonical AMBA entities.
# Exports: PROTOCOL_GLOSSARY, detect_comparison_intent, canonicalize_terms
# Deps: re (stdlib), nothing else (no I/O, no LLM)
# @end-summary
"""Protocol glossary and lexical routing helpers.

This module is the single source of truth for:

* :data:`PROTOCOL_GLOSSARY` — canonical AMBA-family protocol entities, their
  informal/long-form aliases, known spec IDs, and short descriptions. Later
  slices (decomposition, routing) ground their behaviour in this mapping so the
  system splits/routes into *real corpus entities* rather than invented ones.
* :func:`detect_comparison_intent` — a conservative, deterministic lexical
  detector for comparison/juxtaposition phrasing. A false positive is mildly
  wasteful (an extra decomposition attempt) while a false negative is safe
  (normal retrieval), so the detector errs toward precision on plain queries.
* :func:`canonicalize_terms` — scans a query for glossary aliases/entities and
  returns the de-duplicated, order-stable list of canonical entities present.

Canonical-form convention (the rule the whole feature follows):
**the most specific named entity wins**. A versioned token such as ``"AXI4"``
canonicalizes to ``"AXI4"`` and is *not* collapsed to the ``"AXI"`` umbrella;
the bare word ``"AXI"`` canonicalizes to ``"AXI"``. Informal phrases map to
their canonical entity (``"hub-based"`` -> ``"CHI"``,
``"extension-based"`` -> ``"ACE"``). Matching is case-insensitive and
word/phrase-boundary aware so substrings (``"axis"``, ``"taxi"``) never match.
"""
from __future__ import annotations

import re
from typing import Dict, List


# ─── Glossary ────────────────────────────────────────────────────────────────

# Canonical entity -> metadata. Structured so it is easy to extend: add a new
# canonical key, list its informal/long-form aliases, any known spec IDs, and a
# one-line description. The canonical key itself is always also matchable (it is
# treated as an implicit alias of itself in :func:`canonicalize_terms`).
#
# Spec IDs: only the two IDs explicitly named in the design are included
# (AXI IHI0022E, CHI IHI0098). Do not invent IDs for the others.
PROTOCOL_GLOSSARY: Dict[str, Dict[str, object]] = {
    "AMBA": {
        "aliases": [
            "advanced microcontroller bus architecture",
        ],
        "spec_ids": [],
        "description": (
            "ARM's umbrella on-chip interconnect specification family "
            "(AXI, AHB, APB, ACE, CHI)."
        ),
    },
    "AXI": {
        "aliases": [
            "advanced extensible interface",
            "amba axi",
        ],
        "spec_ids": ["IHI0022E"],
        "description": (
            "Advanced eXtensible Interface — high-performance, burst-based "
            "AMBA interconnect protocol (umbrella term across AXI3/4/5)."
        ),
    },
    "AXI3": {
        "aliases": [
            "axi v3",
            "axi3.0",
        ],
        "spec_ids": [],
        "description": "Third-generation AXI (supports write-data interleaving).",
    },
    "AXI4": {
        "aliases": [
            "axi v4",
            "axi 4",
            "axi-4",
        ],
        "spec_ids": [],
        "description": (
            "Fourth-generation AXI (AXI4, AXI4-Lite, AXI4-Stream variants)."
        ),
    },
    "AXI5": {
        "aliases": [
            "axi v5",
            "axi 5",
            "axi-5",
        ],
        "spec_ids": [],
        "description": (
            "Fifth-generation AXI (cache-coherency and atomic-transaction "
            "extensions aligned with ACE/CHI)."
        ),
    },
    "ACE": {
        "aliases": [
            "axi coherency extensions",
            "extension-based",
            "extension based",
            "coherency extensions",
        ],
        "spec_ids": [],
        "description": (
            "AXI Coherency Extensions — adds full hardware cache coherency on "
            "top of AXI (the 'extension-based' coherency approach)."
        ),
    },
    "ACE-Lite": {
        "aliases": [
            "ace lite",
            "acelite",
        ],
        "spec_ids": [],
        "description": (
            "Lightweight ACE variant for I/O-coherent (one-way snoop) masters."
        ),
    },
    "AHB": {
        "aliases": [
            "advanced high-performance bus",
            "advanced high performance bus",
        ],
        "spec_ids": [],
        "description": (
            "Advanced High-performance Bus — pipelined AMBA bus for "
            "higher-bandwidth peripherals and memory."
        ),
    },
    "APB": {
        "aliases": [
            "advanced peripheral bus",
        ],
        "spec_ids": [],
        "description": (
            "Advanced Peripheral Bus — low-bandwidth, low-power AMBA bus for "
            "simple control/status registers."
        ),
    },
    "CHI": {
        "aliases": [
            "coherent hub interface",
            "hub-based",
            "hub based",
        ],
        "spec_ids": ["IHI0098"],
        "description": (
            "Coherent Hub Interface — packet/credit-based coherent "
            "interconnect protocol (the 'hub-based' coherency approach)."
        ),
    },
}


# ─── Comparison-intent detection ───────────────────────────────────────────────

# Explicit comparison cue phrases. These are matched as whole words/phrases,
# case-insensitively. Each is unambiguously comparative.
_COMPARISON_PHRASES: tuple[str, ...] = (
    "versus",
    "compare",
    "comparison",
    "compared to",
    "compared with",
    "difference between",
    "differences between",
    "differ from",
    "differs from",
    "vs",  # handled as a standalone token via word boundaries below
)

# Pre-compiled patterns for the explicit cues. ``vs`` / ``vs.`` need their own
# pattern so we match the token (with optional trailing period) but not the
# letters "vs" inside another word.
_VS_PATTERN = re.compile(r"\bvs\b\.?", re.IGNORECASE)
_PHRASE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
    for phrase in _COMPARISON_PHRASES
    if phrase != "vs"
)

# "A or the B one/approach/version/style" — implicit comparison phrasing where
# the alternatives are presented as an either/or choice of approaches.
_OR_THE_ONE_PATTERN = re.compile(
    r"\bor\s+the\b.*\b(one|approach|version|variant|style|option|way)\b",
    re.IGNORECASE,
)

# A slash-separated list (e.g. "AXI4 / AXI5 / CHI") is treated as a comparison
# only when the slashed tokens resolve to known glossary entities — this avoids
# flagging file paths or "and/or" style slashes.
_SLASH_SPLIT = re.compile(r"\s*/\s*")


def detect_comparison_intent(query: str) -> bool:
    """Return ``True`` when *query* uses comparison/juxtaposition phrasing.

    Conservative by design: plain single-topic queries (e.g.
    ``"reset value of Filters_N_Status"``, ``"explain the MBIST flow"``,
    ``"steps before inserting MBIST"``) must return ``False``. ``"before"`` is
    sequencing, and a bare ``"and"`` joining aspects of one topic is not a
    comparison.

    Detected cues (case-insensitive):

    * explicit: ``vs`` / ``vs.`` / ``versus`` / ``compare`` / ``comparison`` /
      ``compared to`` / ``compared with`` / ``difference(s) between`` /
      ``differ(s) from``;
    * a slash-list of known glossary entities (``"AXI4 / AXI5 / CHI"``);
    * an either/or-of-approaches phrasing (``"... or the ... one/approach"``).

    Args:
        query: The raw user query.

    Returns:
        ``True`` if comparison intent is detected, else ``False``.
    """
    if not query:
        return False

    # Explicit "vs"/"vs." token.
    if _VS_PATTERN.search(query):
        return True

    # Other explicit phrase cues.
    for pattern in _PHRASE_PATTERNS:
        if pattern.search(query):
            return True

    # Implicit "... or the ... one/approach" either/or phrasing.
    if _OR_THE_ONE_PATTERN.search(query):
        return True

    # Slash-list of known entities: require at least two slash-separated parts,
    # each of which contains a recognized glossary entity.
    parts = _SLASH_SPLIT.split(query)
    if len(parts) >= 2:
        entity_parts = [p for p in parts if canonicalize_terms(p)]
        if len(entity_parts) >= 2:
            return True

    return False


# ─── Term canonicalization ─────────────────────────────────────────────────────


def _boundary_pattern(term: str) -> re.Pattern[str]:
    r"""Compile a case-insensitive, boundary-aware pattern for *term*.

    Uses ``\b`` anchors so ``"AXI"`` matches the standalone token but not
    ``"axis"`` or ``"taxi"``. Hyphens inside a term (``"hub-based"``,
    ``"ACE-Lite"``) are matched literally; internal whitespace is matched
    flexibly so ``"coherent  hub interface"`` still resolves.
    """
    # Escape then relax runs of whitespace to ``\s+`` so multi-word aliases are
    # robust to spacing. ``\b`` does not anchor cleanly next to non-word edge
    # characters, so for terms that start/end with a word character we anchor on
    # ``\b``; this is sufficient for every glossary alias (all start/end with a
    # letter or digit).
    escaped = re.escape(term)
    escaped = re.sub(r"\\\s+|\s+", r"\\s+", escaped)
    return re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)


def _build_match_table() -> List[tuple[re.Pattern[str], str]]:
    """Build an ordered (pattern, canonical) table for canonicalization.

    Ordering rule (so the *most specific* match wins): longer surface terms are
    tried first. The canonical key of each entry is treated as an implicit
    alias of itself, so ``"AXI4"`` (a canonical key) matches and resolves to
    ``"AXI4"`` without being collapsed to the ``"AXI"`` umbrella.
    """
    surface: list[tuple[str, str]] = []
    for canonical, entry in PROTOCOL_GLOSSARY.items():
        surface.append((canonical, canonical))
        for alias in entry["aliases"]:  # type: ignore[index]
            surface.append((str(alias), canonical))
    # Longest surface term first → most specific match wins.
    surface.sort(key=lambda pair: len(pair[0]), reverse=True)
    return [(_boundary_pattern(term), canonical) for term, canonical in surface]


# Built once at import (the glossary is static).
_MATCH_TABLE: List[tuple[re.Pattern[str], str]] = _build_match_table()


def canonicalize_terms(query: str) -> List[str]:
    """Return the de-duplicated canonical entities present in *query*.

    Scans the query for every glossary alias/entity (case-insensitive,
    boundary-aware) and returns the matching canonical entities. The result is
    de-duplicated and order-stable (first canonical entity to appear, scanning
    left to right, comes first). Returns ``[]`` when nothing matches.

    Examples:
        >>> canonicalize_terms("the hub-based approach vs the extension-based one")
        ['CHI', 'ACE']
        >>> canonicalize_terms("AXI4 vs CHI")
        ['AXI4', 'CHI']
        >>> canonicalize_terms("reset value of Filters_N_Status")
        []

    Args:
        query: The raw user query.

    Returns:
        Ordered, de-duplicated list of canonical entity names.
    """
    if not query:
        return []

    # Find the earliest match position for each canonical entity. We must not
    # let a more-general alias overwrite a more-specific one at overlapping
    # spans, so we mask out characters already consumed by a (longer, earlier)
    # match — the match table is longest-first, so specific terms claim their
    # span before umbrella terms get a chance.
    masked = list(query)
    first_pos: Dict[str, int] = {}
    for pattern, canonical in _MATCH_TABLE:
        for m in pattern.finditer("".join(masked)):
            start, end = m.start(), m.end()
            # Skip if the span was already consumed by a more-specific match.
            if any(masked[i] == "\x00" for i in range(start, end)):
                continue
            if canonical not in first_pos:
                first_pos[canonical] = start
            # Consume the span so umbrella terms (e.g. "AXI") cannot also claim
            # the same characters that "AXI4" already matched.
            for i in range(start, end):
                masked[i] = "\x00"

    return [c for c, _ in sorted(first_pos.items(), key=lambda kv: kv[1])]
