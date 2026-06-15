# @summary
# Query decomposition for document routing. Splits a CLEAR comparison query
# into standalone single-entity sub-queries so each can be routed independently
# (DESIGN §6.3/§6.5). This slice (C1) implements ONLY the Tier-1 regex
# *fallback* — a deterministic, pure-function heuristic. Later slices (C2/C3)
# append ``llm_decompose`` (Tier-2 primary) and ``decompose_query`` (gated
# orchestrator) to this same module.
# Exports: regex_decompose
# Deps: re (stdlib), src.retrieval.routing.glossary (canonicalize_terms — grounding)
# @end-summary
"""Tier-1 regex comparison-decomposition (fallback only).

Per :doc:`DOCUMENT_ROUTING_DESIGN` §6.5, decomposition has three tiers:

==== ====================== ====================================
Tier Implementation         Role
==== ====================== ====================================
1    regex heuristic        **fallback only** (this module, C1)
2    glossary-grounded LLM   **primary** (added in C2)
3    deep research           **not used for comparisons**
==== ====================== ====================================

Regex on natural language is brittle. The governing safety rule:

* a **MISS is safe** — returning ``[]`` makes the caller keep the original
  query and run normal flat retrieval;
* a **MIS-SPLIT is actively harmful** — it shreds the query into nonsense
  sub-queries that retrieve garbage.

So :func:`regex_decompose` is deliberately *conservative*: it only splits
unambiguous comparison structures, validates that the resulting pieces look
like real entities (short, noun-ish, ideally glossary-grounded), and on any
doubt returns ``[]``. The Tier-2 LLM handles the cases regex cannot ground
(e.g. informal ``"hub-based"`` → CHI, ``"extension-based"`` → ACE).

Split/cleaning contract (also pinned by the unit tests):

* Recognised comparison structures (first match wins):

  1. ``difference(s) between A and B`` — the **only** frame where a bare
     ``and`` is allowed to act as a separator;
  2. ``A vs B`` / ``A vs. B`` / ``A versus B``;
  3. ``A compared to B`` / ``A compared with B``;
  4. a slash list of >= 2 items, optionally with a lead-in
     (``"Compare AMBA AXI4 / AXI5 / CHI for coherency"`` → the slash items).

* Each produced sub-query is the *cleaned* entity string:

  - whitespace-trimmed;
  - leading filler stripped (lead-in verbs like ``compare``; articles
    ``the``/``a``/``an``; an umbrella lead-in word such as ``AMBA`` that
    precedes the real items);
  - trailing aspect clause stripped (``for ...`` / ``with ...`` / ``in a ...``
    / ``of ...`` etc.).
  - Casing is preserved verbatim — the function never upper/lower-cases the
    entity token, so ``"AXI4"`` stays ``"AXI4"``.

* Mis-split guards (any failing → ``[]``): fewer than two non-empty entities;
  any "entity" that is sentence-like (too many words, or still contains an
  aspect preposition after cleaning). A slash list additionally requires the
  cleaned items to look like entities and at least one to ground via the
  glossary (so file paths like ``src/foo/bar`` are not mistaken for a
  comparison).

Determinism: pure function, no I/O, no randomness, no global state.
"""
from __future__ import annotations

import re
from typing import List

from src.retrieval.routing.glossary import canonicalize_terms


# ─── Tuning heuristics ──────────────────────────────────────────────────────

# A cleaned entity should be a short noun-ish phrase. Anything longer than this
# many whitespace tokens is treated as sentence-like → probable mis-split.
_MAX_ENTITY_WORDS = 4

# Aspect prepositions: if a cleaned candidate still *contains* one of these as a
# whole word, it is carrying an aspect clause (not a bare entity) → mis-split.
# These also delimit the trailing-aspect strip below.
_ASPECT_PREPOSITIONS: tuple[str, ...] = (
    "for",
    "with",
    "in",
    "on",
    "of",
    "as",
    "that",
    "which",
    "when",
    "where",
    "by",
    "to",
)

# Leading filler tokens stripped from the front of a candidate entity. Lead-in
# verbs (compare/compared/comparison), articles, and the AMBA umbrella word
# (which commonly prefixes a slash list, e.g. "Compare AMBA AXI4 / AXI5 / CHI").
_LEADING_FILLER: tuple[str, ...] = (
    "compare",
    "compared",
    "comparing",
    "comparison",
    "the",
    "a",
    "an",
    "amba",
)


# ─── Comparison-structure patterns ──────────────────────────────────────────

# 1. "difference(s) between A and B" — capture the two operands. The 'and' here
#    is the ONLY context where a bare 'and' may split (DESIGN §6.5).
_DIFF_BETWEEN = re.compile(
    r"\bdifferences?\s+between\s+(?P<a>.+?)\s+\band\b\s+(?P<b>.+)$",
    re.IGNORECASE,
)

# 2. "A vs B" / "A vs. B" / "A versus B".
_VS = re.compile(r"\s+\bvs\b\.?\s+|\s+\bversus\b\s+", re.IGNORECASE)

# 3. "A compared to/with B".
_COMPARED = re.compile(r"\s+\bcompared\s+(?:to|with)\b\s+", re.IGNORECASE)

# 4. Slash list separator.
_SLASH = re.compile(r"\s*/\s*")


def _strip_leading_filler(text: str) -> str:
    """Strip leading filler tokens (lead-in verbs, articles, AMBA) from *text*.

    Removes filler tokens one at a time from the front so a stacked lead-in such
    as ``"Compare AMBA AXI4"`` is reduced to ``"AXI4"``. Casing of the surviving
    entity is preserved.
    """
    tokens = text.split()
    while tokens and tokens[0].lower().strip(".,:;") in _LEADING_FILLER:
        tokens.pop(0)
    return " ".join(tokens)


def _strip_trailing_aspect(text: str) -> str:
    """Strip a trailing aspect clause introduced by an aspect preposition.

    ``"CHI for coherency"`` → ``"CHI"``; ``"AHB in a high-performance SoC"`` →
    ``"AHB"``. Only the *first* aspect preposition onward is removed, so the
    leading entity token is kept. If the candidate *starts* with an aspect
    preposition (no entity precedes it) nothing is stripped — the guard layer
    will reject such a candidate as sentence-like.
    """
    tokens = text.split()
    for idx, tok in enumerate(tokens):
        if idx == 0:
            # Never treat the very first token as an aspect boundary; a real
            # entity must precede the aspect.
            continue
        if tok.lower().strip(".,:;?") in _ASPECT_PREPOSITIONS:
            return " ".join(tokens[:idx])
    return text


def _clean_entity(raw: str) -> str:
    """Trim, then strip leading filler and any trailing aspect clause."""
    cleaned = raw.strip().strip(".,:;?!")
    cleaned = _strip_leading_filler(cleaned)
    cleaned = _strip_trailing_aspect(cleaned)
    return cleaned.strip()


def _looks_like_entity(text: str) -> bool:
    """Return ``True`` if *text* looks like a bare entity, not a sentence.

    Rejects empty strings, over-long phrases (``> _MAX_ENTITY_WORDS`` tokens),
    and phrases that still contain an aspect preposition as a whole word (a
    leftover aspect clause the cleaner could not safely remove).
    """
    if not text:
        return False
    tokens = text.split()
    if len(tokens) > _MAX_ENTITY_WORDS:
        return False
    lowered = {t.lower().strip(".,:;?!") for t in tokens}
    if lowered & set(_ASPECT_PREPOSITIONS):
        return False
    return True


def _finalize(candidates: List[str], *, require_grounding: bool) -> List[str]:
    """Clean, validate and finalize a list of candidate entity strings.

    Returns the cleaned entities only if the split is *trustworthy*:

    * at least two non-empty cleaned entities survive;
    * every cleaned entity ``_looks_like_entity`` (short, no aspect leftovers);
    * if ``require_grounding`` (slash lists, which are the most ambiguous), at
      least one cleaned entity must resolve via :func:`canonicalize_terms`.

    Any failure → ``[]`` (a safe miss).
    """
    cleaned = [_clean_entity(c) for c in candidates]
    cleaned = [c for c in cleaned if c]

    if len(cleaned) < 2:
        return []
    if not all(_looks_like_entity(c) for c in cleaned):
        return []
    if require_grounding and not any(canonicalize_terms(c) for c in cleaned):
        return []
    return cleaned


def regex_decompose(query: str) -> List[str]:
    """Tier-1 regex fallback: split a clear comparison into sub-queries.

    Recognises ``difference(s) between A and B``, ``A vs/versus B``,
    ``A compared to/with B``, and slash lists of >= 2 entities (optionally with
    a lead-in), and returns one cleaned standalone sub-query per compared
    entity. Returns ``[]`` for everything else (the caller then keeps the
    original query and runs normal flat retrieval).

    This is a **fallback only** and is intentionally conservative: it would
    rather miss a valid comparison (safe) than mis-split a query (harmful). See
    the module docstring for the full split/cleaning/guard contract.

    Args:
        query: The raw user query.

    Returns:
        Ordered list of cleaned single-entity sub-queries, or ``[]`` when the
        query is not an unambiguous comparison (or a split fails its guards).

    Examples:
        >>> regex_decompose("AXI4 vs CHI")
        ['AXI4', 'CHI']
        >>> regex_decompose("difference between AHB and APB")
        ['AHB', 'APB']
        >>> regex_decompose("Compare AMBA AXI4 / AXI5 / CHI for coherency")
        ['AXI4', 'AXI5', 'CHI']
        >>> regex_decompose("AHB for a high-performance SoC")
        []
        >>> regex_decompose("design and verification of a block")
        []
    """
    if not query or not query.strip():
        return []

    text = query.strip()

    # 1. "difference(s) between A and B" — the only 'and'-splitting frame.
    diff_match = _DIFF_BETWEEN.search(text)
    if diff_match:
        return _finalize(
            [diff_match.group("a"), diff_match.group("b")],
            require_grounding=False,
        )

    # 2. "A vs/versus B".
    if _VS.search(text):
        parts = _VS.split(text)
        return _finalize(parts, require_grounding=False)

    # 3. "A compared to/with B".
    if _COMPARED.search(text):
        parts = _COMPARED.split(text)
        return _finalize(parts, require_grounding=False)

    # 4. Slash list of >= 2 items (most ambiguous → require glossary grounding).
    if "/" in text:
        parts = _SLASH.split(text)
        if len(parts) >= 2:
            return _finalize(parts, require_grounding=True)

    # No recognised comparison structure → safe miss.
    return []
