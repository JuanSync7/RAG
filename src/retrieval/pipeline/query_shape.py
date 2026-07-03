# @summary
# Cheap lexical/structural classifier for English queries — used both to
# (a) flag a baseline query as "would benefit from Deep Research" for the
# advisory chip and (b) hint adaptive DR depth from raw text. Pure function,
# no LLM calls, no extra retrieval.
# Exports: should_suggest_deep_research, has_compound_marker, COMPOUND_MARKERS,
#          MAX_UNIQUE_SOURCES_FOR_SUGGESTION
# Deps: src.retrieval.common.schemas.RankedResult
# @end-summary
"""Query-shape classifier (lexical compound/multi-topic detection).

Combines two cheap signals over a completed baseline retrieval:

1. **Compound-marker check** — the query string contains a multi-topic cue
   (``" and "``, ``" vs "``, ``" versus "``, ``" compare "``) or two or more
   ``"?"`` characters. LLM-free, gates the suggestion.
2. **Topic-spread check** — the top-k baseline results come from a narrow
   set of source files (≤ ``MAX_UNIQUE_SOURCES_FOR_SUGGESTION`` distinct
   sources). Low diversity on a compound query is the signature of a query
   that touched only one of its aspects.

Both signals must fire to suggest. The cost is one regex-style scan plus a
``set()`` over result metadata — well under a millisecond. False-positive
rate matters more than recall: the chip is advisory, not auto-flipping.

Sufficiency-check (signal 3) is intentionally skipped — wiring the DR
orchestrator's sufficiency LLM call here would add hundreds of milliseconds
to every baseline response and defeat the cheapness contract.
"""

from __future__ import annotations

from typing import Iterable

from config.settings import RAG_QUERY_SUGGESTION_MIN_UNIQUE_SOURCES
from src.retrieval.common.schemas import RankedResult


COMPOUND_MARKERS: tuple[str, ...] = (
    " and ",
    " vs ",
    " versus ",
    " compare ",
    " compared ",
)

MAX_UNIQUE_SOURCES_FOR_SUGGESTION: int = RAG_QUERY_SUGGESTION_MIN_UNIQUE_SOURCES

_SOURCE_KEYS: tuple[str, ...] = ("source", "source_key", "source_uri", "document_id")


def has_compound_marker(query: str) -> bool:
    """True when the query names several facets at once.

    Fires on a multi-topic lexical cue (an ``" and "`` / ``" vs "`` /
    ``" compare "`` marker) or two or more ``"?"`` characters. LLM-free, pure,
    sub-millisecond. Shared signal: the Deep-Research suggestion chip
    (:func:`should_suggest_deep_research`) and the turn-loop pre-flight router
    (``turn_loop.router`` via ``turn_loop_runner.build_route_signals``) both
    read it.
    """
    if not query:
        return False
    padded = f" {query.lower()} "
    if any(marker in padded for marker in COMPOUND_MARKERS):
        return True
    if query.count("?") >= 2:
        return True
    return False


# Back-compat private alias (the in-module caller below + existing imports).
_has_compound_marker = has_compound_marker


def _unique_source_count(results: Iterable[RankedResult]) -> int:
    sources: set[str] = set()
    untagged = 0
    for r in results:
        meta = r.metadata or {}
        key = ""
        for candidate in _SOURCE_KEYS:
            value = meta.get(candidate)
            if isinstance(value, str) and value.strip():
                key = value.strip()
                break
        if key:
            sources.add(key)
        else:
            # Each untagged result counts as its own "source" — refuses
            # to claim low diversity when the data can't prove it.
            untagged += 1
    return len(sources) + untagged


def should_suggest_deep_research(
    query: str,
    baseline_results: list[RankedResult],
) -> tuple[bool, str | None]:
    """Decide whether to suggest Deep Research for a baseline-mode query.

    Returns ``(suggest, reason)``. ``reason`` is a short human-readable string
    when ``suggest`` is True, suitable for logging or tooltips; ``None``
    otherwise.
    """

    if not baseline_results:
        return False, None
    if not _has_compound_marker(query):
        return False, None

    unique_sources = _unique_source_count(baseline_results)
    if unique_sources > MAX_UNIQUE_SOURCES_FOR_SUGGESTION:
        return False, None

    reason = (
        f"compound query with low source diversity "
        f"({unique_sources} unique source(s) in top {len(baseline_results)})"
    )
    return True, reason
