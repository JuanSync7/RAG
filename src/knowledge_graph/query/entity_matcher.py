# @summary
# Two-tier entity matcher for query-time entity extraction.
# Tier 1: spaCy PhraseMatcher — fast, deterministic, token-boundary aware.
# Tier 2: LLM fallback — active when entity_types and KGConfig.enable_llm_query_fallback are set.
# Exports: EntityMatcher
# Deps: logging, typing, spacy (optional), src.platform.llm (optional, lazy import)
# @end-summary
"""Two-tier entity matcher for query-time entity extraction.

Uses spaCy's ``PhraseMatcher`` as the primary matching strategy to correctly
respect token boundaries (avoiding false positives from pure substring search).
Falls back to a sorted substring scan when spaCy is not installed.  When the
primary tier returns no matches, an optional LLM fallback queries the language
model with the entity list grouped by type to identify relevant entities.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Set

__all__ = ["EntityMatcher"]

logger = logging.getLogger("rag.knowledge_graph.query")


class EntityMatcher:
    """Two-tier entity matcher for query-time entity extraction.

    Tier 1: spaCy PhraseMatcher — fast, deterministic, loaded from graph
    nodes + aliases.  Matching is token-boundary aware, which avoids the
    false positives inherent in plain substring search (e.g. "add" matching
    inside "adder").

    Tier 2: LLM fallback — invoked when primary matching yields no results,
    ``entity_types`` is provided, and ``KGConfig.enable_llm_query_fallback``
    is True.  The LLM receives the query and all known entities grouped by
    type, and returns a JSON array of matching entity names.
    """

    def __init__(
        self,
        entity_names: List[str],
        aliases: Dict[str, str],
        entity_types: Optional[Dict[str, str]] = None,
        config=None,
    ) -> None:
        """Initialise with known entity names, alias mappings, and optional LLM config.

        Args:
            entity_names: All canonical entity names from the graph.
            aliases: Mapping of alias → canonical name.
            entity_types: Optional mapping of entity_name → type string.
                When provided (and config enables it), the LLM fallback is
                active.  Defaults to an empty dict (fallback disabled).
            config: Optional ``KGConfig`` instance.  Used to read
                ``enable_llm_query_fallback`` and ``llm_fallback_timeout_ms``.
                When None, the LLM fallback is disabled.
        """
        self._entity_names: Set[str] = set(entity_names)
        self._aliases: Dict[str, str] = aliases
        self._entity_types: Dict[str, str] = entity_types or {}

        # Resolve fallback settings from config (if provided).
        if config is not None and getattr(config, "enable_llm_query_fallback", False):
            self._enable_llm_fallback: bool = True
            self._llm_fallback_timeout_ms: int = getattr(
                config, "llm_fallback_timeout_ms", 1000
            )
        else:
            self._enable_llm_fallback = False
            self._llm_fallback_timeout_ms = 1000

        self._nlp = None
        self._matcher = None
        self._initialize_spacy()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize_spacy(self) -> None:
        """Load a blank spaCy model and build the PhraseMatcher index.

        A *blank* (tokenization-only) model is used intentionally — we only
        need tokenization, not the full NLP pipeline.  This keeps start-up
        time and memory usage low.

        If spaCy is not installed the method silently degrades to the
        substring fallback strategy.
        """
        try:
            import spacy  # noqa: PLC0415
            from spacy.matcher import PhraseMatcher  # noqa: PLC0415

            # Blank model: tokenization only, no NER/tagger/parser overhead.
            self._nlp = spacy.blank("en")
            self._matcher = PhraseMatcher(self._nlp.vocab, attr="LOWER")

            patterns = []
            for name in self._entity_names:
                patterns.append(self._nlp.make_doc(name))
            for alias in self._aliases:
                patterns.append(self._nlp.make_doc(alias))

            if patterns:
                self._matcher.add("KG_ENTITIES", patterns)

            logger.info(
                "spaCy EntityMatcher initialised with %d patterns",
                len(patterns),
            )
        except ImportError:
            logger.warning(
                "spaCy not available — EntityMatcher will use substring matching"
            )
            self._nlp = None
            self._matcher = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(self, query: str) -> List[str]:
        """Extract canonical entity names present in *query*.

        Uses the spaCy PhraseMatcher when available; degrades to a
        case-insensitive substring scan otherwise.

        Args:
            query: Raw query string.

        Returns:
            Deduplicated list of canonical entity names found in *query*.
        """
        if self._matcher is not None and self._nlp is not None:
            return self._match_spacy(query)
        return self._match_substring(query)

    def match_with_llm_fallback(self, query: str) -> List[str]:
        """Match with LLM fallback when spaCy/substring finds nothing.

        When primary matching returns empty and LLM fallback is enabled,
        sends the query + entity names (grouped by type) to the LLM to
        identify which entities the query is about.

        Args:
            query: Raw query string.

        Returns:
            Deduplicated list of canonical entity names found in *query*.
        """
        results = self.match(query)
        if results:
            return results

        # Check if LLM fallback is enabled.
        if not self._enable_llm_fallback:
            return results

        # Skip very short queries — too little signal for meaningful LLM matching.
        if len(query.split()) < 3:
            return results

        try:
            return self._llm_match(query)
        except Exception as exc:
            logger.warning("LLM query fallback failed: %s", exc)
            return results

    # ------------------------------------------------------------------
    # Private matching strategies
    # ------------------------------------------------------------------

    def _llm_match(self, query: str) -> List[str]:
        """Ask the LLM which entities the query is about.

        Builds an entity list grouped by type and sends it alongside the
        query to the LLM via ``json_completion``.  The LLM is expected to
        return a JSON array of entity name strings drawn exclusively from
        the provided list.

        Args:
            query: Raw query string (caller guarantees >= 3 tokens).

        Returns:
            Filtered list of canonical entity names returned by the LLM
            that are present in the known entity set.
        """
        from src.platform.llm import get_llm_provider  # noqa: PLC0415

        provider = get_llm_provider()

        # Build entity list grouped by type for structured prompt context.
        type_groups: Dict[str, List[str]] = {}
        for name, entity_type in self._entity_types.items():
            type_groups.setdefault(entity_type, []).append(name)

        entity_list = ""
        for etype, names in sorted(type_groups.items()):
            entity_list += f"\n{etype}: {', '.join(sorted(names))}"

        messages = [
            {
                "role": "system",
                "content": (
                    "You extract entity names from queries. "
                    "Return a JSON array of entity names that the query is asking about. "
                    "Only return names from the provided list."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    f"Known entities:{entity_list}\n\n"
                    "Return JSON array of matching entity names:"
                ),
            },
        ]

        # Convert ms timeout to seconds for the provider call.
        timeout_s = self._llm_fallback_timeout_ms / 1000.0
        try:
            response = provider.json_completion(
                messages,
                max_tokens=256,
                timeout=timeout_s,
            )

            matched = json.loads(response.content)
            if not isinstance(matched, list):
                return []

            # Guard: keep only names that are actually in the known entity set.
            return [name for name in matched if name in self._entity_names]
        except json.JSONDecodeError as exc:
            logger.warning(
                "LLM returned malformed JSON for query %r; falling back to empty match. Error: %s",
                query,
                exc,
            )
            return []
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LLM entity match failed for query %r; falling back to empty match. Error: %s",
                query,
                exc,
            )
            return []

    def _match_spacy(self, query: str) -> List[str]:
        """Match using spaCy PhraseMatcher (token-boundary aware).

        For each span matched by the PhraseMatcher the method resolves the
        surface form to a canonical entity name via the alias dictionary,
        falling back to a case-insensitive lookup against the entity name
        set, and finally using the raw span text if neither lookup yields a
        result.

        Args:
            query: Raw query string.

        Returns:
            Deduplicated list of resolved canonical names.
        """
        try:
            doc = self._nlp(query)
            matches = self._matcher(doc)

            matched_entities: Set[str] = set()
            for _match_id, start, end in matches:
                span_text: str = doc[start:end].text

                # 1. Direct alias lookup (exact case as stored).
                canonical: Optional[str] = self._aliases.get(span_text)

                # 2. Case-insensitive alias lookup.
                if canonical is None:
                    span_lower = span_text.lower()
                    for alias, target in self._aliases.items():
                        if alias.lower() == span_lower:
                            canonical = target
                            break

                # 3. Case-insensitive lookup against canonical names.
                if canonical is None:
                    for name in self._entity_names:
                        if name.lower() == span_text.lower():
                            canonical = name
                            break

                # 4. Last resort: use the matched span text as-is.
                matched_entities.add(canonical if canonical is not None else span_text)

            return list(matched_entities)
        except Exception:
            logger.exception(
                "EntityMatcher._match_spacy() failed for query %r; returning empty match list",
                query,
            )
            return []

    def _match_substring(self, query: str) -> List[str]:
        """Fallback: case-insensitive substring matching.

        Matches are checked longest-first to prefer the most specific entity
        name when shorter names are substrings of longer ones.  This
        preserves the behaviour of the legacy ``GraphQueryExpander``.

        Args:
            query: Raw query string.

        Returns:
            Deduplicated list of canonical entity names found in *query*.
        """
        query_lower = query.lower()
        matched: Set[str] = set()

        # Longest entity names first for specificity.
        for name in sorted(self._entity_names, key=len, reverse=True):
            if name.lower() in query_lower:
                matched.add(name)

        # Longest aliases first for specificity.
        for alias, canonical in sorted(
            self._aliases.items(), key=lambda kv: len(kv[0]), reverse=True
        ):
            if alias.lower() in query_lower:
                matched.add(canonical)

        return list(matched)
