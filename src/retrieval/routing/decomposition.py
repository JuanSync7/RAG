# @summary
# Query decomposition for document routing. Splits a CLEAR comparison query
# into standalone single-entity sub-queries so each can be routed independently
# (DESIGN §6.3/§6.5). Tier-1 (C1) ``regex_decompose`` is the deterministic
# *fallback*; Tier-2 (C2) ``llm_decompose`` is the glossary-grounded LLM
# *primary* path (raises ``DecompositionError`` on any failure so the orchestrator
# can fall back). The C3 gated orchestrator ``decompose_query`` lives here too:
# it gates on RAG_DECOMPOSITION_ENABLED + comparison-intent, runs LLM-primary
# (regex fallback) per RAG_DECOMPOSITION_LLM_PRIMARY, and NEVER raises —
# degrading to the identity single-query baseline on any miss/failure.
# Exports: regex_decompose, llm_decompose, decompose_query, DecompositionError
# Deps: re (stdlib), src.retrieval.routing.glossary (PROTOCOL_GLOSSARY,
#   canonicalize_terms, detect_comparison_intent — gate/grounding),
#   src.retrieval.routing.schemas (DecompositionResult), src.platform.llm
#   (get_llm_provider), src.common (parse_json_object), config.settings
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

import logging
import re
from typing import List

from config import settings
from src.common import parse_json_object
from src.platform.llm import get_llm_provider
from src.retrieval.routing.glossary import (
    PROTOCOL_GLOSSARY,
    canonicalize_terms,
    detect_comparison_intent,
)
from src.retrieval.routing.schemas import DecompositionResult

logger = logging.getLogger("rag.retrieval.routing.decomposition")


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


# ─── Tier-2 — glossary-grounded LLM decomposition (primary) ─────────────────


class DecompositionError(Exception):
    """Signals that LLM (Tier-2) decomposition failed and the caller must fall back.

    Raised by :func:`llm_decompose` on ANY failure mode — provider error,
    timeout, unparseable response, wrong JSON shape, an out-of-range sub-query
    count, or a malformed individual sub-query. C2 itself never falls back; the
    C3 orchestrator catches this and degrades to the regex tier / original query.
    """


# Max characters / words for a single sub-query. The LLM is asked for *short*
# standalone retrieval queries (one per compared entity), so anything longer is
# treated as a malformed item (probable prose leakage) → failure.
_MAX_SUBQUERY_CHARS = 120
_MAX_SUBQUERY_WORDS = 12

# Small, capped output budget — the response is just a short JSON list. Keep
# this tight: the decomposer runs on a reasoning model and latency matters.
_LLM_MAX_TOKENS = 256

# Low temperature for a deterministic, structured extraction task.
_LLM_TEMPERATURE = 0.0


def _render_glossary() -> str:
    """Render a COMPACT glossary block (canonical name + a couple of aliases).

    Keeps the prompt short (latency-sensitive reasoning model): one line per
    canonical entity with at most two informal aliases, no descriptions.
    """
    lines: list[str] = []
    for canonical, entry in PROTOCOL_GLOSSARY.items():
        aliases = entry.get("aliases", [])  # type: ignore[union-attr]
        alias_hint = ", ".join(str(a) for a in list(aliases)[:2])
        if alias_hint:
            lines.append(f"- {canonical} (aka {alias_hint})")
        else:
            lines.append(f"- {canonical}")
    return "\n".join(lines)


_SYSTEM_PROMPT = (
    "You decompose a comparison query into one standalone retrieval sub-query "
    "per compared entity. Use the provided glossary to map informal terms to "
    "canonical entities. Return ONLY JSON: {\"subqueries\": [\"...\"]}. "
    f"{settings.RAG_DECOMPOSITION_MIN_SUBQUERIES}-"
    f"{settings.RAG_DECOMPOSITION_MAX_SUBQUERIES} items, each a short standalone "
    "query.\n\nGlossary (canonical entity, informal aliases):\n" + _render_glossary()
)


def _extract_subqueries(payload: dict[str, object]) -> list[object]:
    """Pull the sub-query list out of the parsed JSON object.

    Accepts ``{"subqueries": [...]}``. Raises :class:`DecompositionError` when
    the key is missing or its value is not a list. (Bare top-level JSON lists
    never reach here: ``parse_json_object`` yields ``{}`` for a JSON array, so
    they are rejected upstream as an empty/missing object.)
    """
    if "subqueries" not in payload:
        raise DecompositionError(
            "LLM decomposition response missing 'subqueries' key"
        )
    raw = payload["subqueries"]
    if not isinstance(raw, list):
        raise DecompositionError(
            f"LLM decomposition 'subqueries' is not a list: {type(raw).__name__}"
        )
    return raw


def _validate_subqueries(raw: list[object]) -> list[str]:
    """Validate + clean the sub-query list, or raise :class:`DecompositionError`.

    Enforces: count within ``[RAG_DECOMPOSITION_MIN_SUBQUERIES,
    RAG_DECOMPOSITION_MAX_SUBQUERIES]``; every item a non-empty string after
    strip that is "short" (``<= _MAX_SUBQUERY_WORDS`` words and
    ``<= _MAX_SUBQUERY_CHARS`` chars).
    """
    lo = settings.RAG_DECOMPOSITION_MIN_SUBQUERIES
    hi = settings.RAG_DECOMPOSITION_MAX_SUBQUERIES
    if not (lo <= len(raw) <= hi):
        raise DecompositionError(
            f"LLM decomposition returned {len(raw)} sub-queries; "
            f"expected {lo}-{hi}"
        )

    cleaned: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise DecompositionError(
                f"LLM decomposition sub-query is not a string: {item!r}"
            )
        text = item.strip()
        if not text:
            raise DecompositionError(
                "LLM decomposition produced an empty/whitespace sub-query"
            )
        if len(text) > _MAX_SUBQUERY_CHARS or len(text.split()) > _MAX_SUBQUERY_WORDS:
            raise DecompositionError(
                f"LLM decomposition sub-query too long (not a short query): {text!r}"
            )
        cleaned.append(text)
    return cleaned


def llm_decompose(query: str, *, timeout: int | None = None) -> List[str]:
    """Tier-2 primary: glossary-grounded LLM comparison decomposition.

    Asks the LLM (grounded in :data:`PROTOCOL_GLOSSARY`) to split a comparison
    query into one *standalone* retrieval sub-query per compared entity, mapping
    informal terms (e.g. ``"hub-based"`` → CHI) to canonical entities. The
    prompt is tight and the output capped (the decomposer runs on a reasoning
    model, so latency is controlled via a short prompt + small ``max_tokens``).

    The response must be a JSON object ``{"subqueries": [...]}`` with
    ``RAG_DECOMPOSITION_MIN_SUBQUERIES``-``RAG_DECOMPOSITION_MAX_SUBQUERIES``
    short, non-empty string items. A **bare top-level JSON list is NOT
    tolerated** (the canonical parser yields ``{}`` for an array, so it is
    rejected as a missing object).

    On ANY failure — provider error/timeout, unparseable content, wrong shape,
    out-of-range count, or a malformed item — this raises
    :class:`DecompositionError` (chaining the provider cause when relevant) so
    the C3 orchestrator can fall back to the regex tier / original query. C2
    does NOT implement that fallback itself.

    Args:
        query: The raw user comparison query.
        timeout: Optional per-call timeout (seconds). Defaults to
            ``settings.RAG_DECOMPOSITION_LLM_TIMEOUT_SECONDS``.

    Returns:
        Ordered list of 2-5 cleaned (whitespace-trimmed) standalone sub-queries.

    Raises:
        DecompositionError: On any provider, parse, shape, or validation failure.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    try:
        response = get_llm_provider().json_completion(
            messages,
            temperature=_LLM_TEMPERATURE,
            max_tokens=_LLM_MAX_TOKENS,
            timeout=timeout or settings.RAG_DECOMPOSITION_LLM_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # provider error / timeout
        logger.debug("LLM decomposition provider call failed: %s", exc)
        raise DecompositionError("LLM decomposition provider call failed") from exc

    payload = parse_json_object(response.content)
    if not payload:
        raise DecompositionError(
            "LLM decomposition response was empty or unparseable JSON"
        )

    raw = _extract_subqueries(payload)
    return _validate_subqueries(raw)


# ─── C3 — gated decomposition orchestrator (safe entry point) ───────────────


def _tier_succeeded(subqueries: List[str]) -> List[str]:
    """Return the non-empty sub-queries iff a tier *succeeded* (>= 2 of them).

    A decomposition tier "succeeds" only when it yields at least two non-empty
    sub-queries. Empty / whitespace-only items are dropped first, so a tier that
    returned two blank strings is correctly treated as a failure. On failure the
    list is empty, signalling the orchestrator to fall through.
    """
    cleaned = [s for s in subqueries if isinstance(s, str) and s.strip()]
    return cleaned if len(cleaned) >= 2 else []


def decompose_query(query: str) -> DecompositionResult:
    """Gated decomposition orchestrator — the safe entry point (DESIGN §6.5).

    Ties together the comparison-intent gate and the two decomposition tiers,
    degrading to the identity baseline on any miss or failure. Precedence:

    1. **Gate (read at call time so config can be monkeypatched):**

       * if ``settings.RAG_DECOMPOSITION_ENABLED`` is False → identity
         ``DecompositionResult([query])`` (no tier is invoked);
       * else if :func:`detect_comparison_intent` is False → identity (no tier).

    2. **Tiers (enabled AND comparison-intent only):**

       * if ``settings.RAG_DECOMPOSITION_LLM_PRIMARY`` (default): try
         :func:`llm_decompose` first; on :class:`DecompositionError` (or any
         unexpected error) try :func:`regex_decompose`;
       * else (regex-primary): try :func:`regex_decompose` first; if it does not
         succeed, try :func:`llm_decompose`.

    3. **Success rule:** a tier succeeds only when it yields **>= 2 non-empty**
       sub-queries. LLM success → ``method="llm"``; regex success →
       ``method="regex"``; both with ``decomposed=True``.

    4. **Safe baseline:** if neither tier succeeds → identity
       ``DecompositionResult([query], method="identity", decomposed=False)``
       (today's single-query retrieval behaviour).

    This function is the safe entry point and **never raises** — every failure
    mode (provider error/timeout, unexpected tier exception, bad shape) degrades
    to a valid :class:`DecompositionResult`.

    Args:
        query: The raw user query.

    Returns:
        A :class:`DecompositionResult`. ``method`` is one of ``"identity"``,
        ``"llm"``, or ``"regex"``; ``decomposed`` is ``True`` only when the query
        was actually split into >= 2 sub-queries.
    """
    identity = DecompositionResult([query], method="identity", decomposed=False)

    # Gate (call-time reads so tests can monkeypatch config.settings).
    if not settings.RAG_DECOMPOSITION_ENABLED:
        return identity
    if not detect_comparison_intent(query):
        return identity

    def _try_llm() -> List[str]:
        try:
            return _tier_succeeded(llm_decompose(query))
        except DecompositionError as exc:
            logger.debug("LLM decomposition failed; falling back: %s", exc)
        except Exception as exc:  # defensive: never let a tier escape
            logger.debug("LLM decomposition raised unexpectedly: %s", exc)
        return []

    def _try_regex() -> List[str]:
        try:
            return _tier_succeeded(regex_decompose(query))
        except Exception as exc:  # defensive: regex is pure but never trust it
            logger.debug("Regex decomposition raised unexpectedly: %s", exc)
        return []

    if settings.RAG_DECOMPOSITION_LLM_PRIMARY:
        subs = _try_llm()
        if subs:
            return DecompositionResult(subs, method="llm", decomposed=True)
        subs = _try_regex()
        if subs:
            return DecompositionResult(subs, method="regex", decomposed=True)
    else:
        subs = _try_regex()
        if subs:
            return DecompositionResult(subs, method="regex", decomposed=True)
        subs = _try_llm()
        if subs:
            return DecompositionResult(subs, method="llm", decomposed=True)

    # Neither tier yielded >= 2 sub-queries → safe single-query baseline.
    return identity
