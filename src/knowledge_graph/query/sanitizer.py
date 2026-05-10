# @summary
# Query sanitizer: normalization, alias expansion, and alias index management.
# Exports: QuerySanitizer
# Deps: re, typing, logging
# @end-summary
"""Query sanitization: normalization, alias expansion, and fan-out control.

Normalizes queries before graph lookup, expands aliases for matched entity
names, and supports hot-rebuilding the alias index when the graph changes.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

__all__ = ["QuerySanitizer"]


class QuerySanitizer:
    """Normalizes and sanitizes queries for graph lookup.

    Responsibilities:
    - Lowercase and normalize whitespace for entity matching
    - Replace hyphens and underscores with spaces so hyphenated terms
      match their graph counterparts
    - Expand matched entity names to include their known aliases
    - Support alias index rebuilds when the graph is updated

    Phase 2 note: when a Neo4j backend is active, all user-provided values
    must be passed through parameterized queries rather than string
    interpolation.  ``sanitize_cypher`` exists as a contract placeholder
    for that transition.

    Args:
        alias_index: Mapping of alias string → canonical entity name.
            Example: ``{"RAG": "Retrieval-Augmented Generation"}``.
    """

    def __init__(self, alias_index: Dict[str, str]) -> None:
        self._alias_index: Dict[str, str] = alias_index

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def normalize(self, query: str) -> str:
        """Normalize a raw query string for entity matching.

        Transformations applied (in order):

        1. Strip leading/trailing whitespace.
        2. Lowercase the entire string so downstream matching is
           case-insensitive without requiring callers to remember.
        3. Replace hyphens and underscores with spaces so that
           ``"graph-rag"`` and ``"graph_rag"`` both reach the same
           candidate entities.
        4. Collapse runs of whitespace to a single space.

        Args:
            query: Raw query string from the caller.

        Returns:
            Normalized query string.
        """
        q = query.lower().strip()
        q = re.sub(r"[-_]", " ", q)
        q = re.sub(r"\s+", " ", q)
        logger.debug("QuerySanitizer.normalize: %r → %r", query, q)
        return q

    def expand_aliases(self, terms: List[str]) -> List[str]:
        """Expand a list of matched entity names with their known aliases.

        For each canonical entity name in *terms*, searches the alias index
        for any alias that resolves to that canonical name and appends it to
        the result if not already present.

        Example::

            sanitizer = QuerySanitizer({"RAG": "Retrieval-Augmented Generation"})
            sanitizer.expand_aliases(["Retrieval-Augmented Generation"])
            # → ["Retrieval-Augmented Generation", "RAG"]

        Args:
            terms: Canonical entity names returned by the entity matcher.

        Returns:
            Original terms plus any discovered aliases, order-preserving.
        """
        result: List[str] = list(terms)
        for term in terms:
            for alias, canonical in self._alias_index.items():
                if canonical == term and alias not in result:
                    result.append(alias)
        return result

    def sanitize_cypher(self, value: str) -> str:
        """Sanitize a value for use in a Neo4j Cypher query string.

        Phase 2 placeholder.  When the Neo4j backend is active, all
        user-provided values **must** be passed through parameterized
        Cypher queries rather than string interpolation.  This method
        exists as a forward-contract reminder.

        For Phase 1 (NetworkX backend), there is no injection risk, so
        the value is returned unchanged.

        Args:
            value: Untrusted user-supplied string.

        Returns:
            Sanitized string (identity for Phase 1).
        """
        # Phase 2: prefer parameterized queries over manual sanitization.
        return value

    def rebuild(self, alias_index: Dict[str, str]) -> None:
        """Replace the alias index with a fresh mapping.

        Called by the query expander after the graph is mutated so that
        alias expansion stays consistent with the current graph state.

        Args:
            alias_index: New alias → canonical mapping.
        """
        self._alias_index = alias_index
