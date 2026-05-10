# @summary
# Rule-based entity and relationship extractor migrated from src/core/knowledge_graph.py.
# Exports: RegexEntityExtractor, STOPWORDS
# Deps: logging, re, typing, src.knowledge_graph.common.schemas, src.knowledge_graph.common.types
# @end-summary
"""Rule-based entity and relationship extractor using regex patterns.

Migrated from ``src/core/knowledge_graph.py`` ``EntityExtractor`` class.

Key changes from the monolith:
- Class renamed ``EntityExtractor`` → ``RegexEntityExtractor``.
- ``extract_relations()`` returns ``List[Triple]`` instead of
  ``List[Tuple[str, str, str]]``.
- ``classify_type()`` added (extracted from
  ``KnowledgeGraphBuilder._classify_type()``); accepts optional
  ``SchemaDefinition`` for schema-based typing.
- ``extract()`` added as a convenience entry point that returns
  ``ExtractionResult``.
- All existing regex patterns and filtering logic are preserved exactly.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set

from src.knowledge_graph.common import (
    Entity,
    ExtractionResult,
    Triple,
)
from src.knowledge_graph.common import SchemaDefinition


logger = logging.getLogger(__name__)

__all__ = ["RegexEntityExtractor", "STOPWORDS"]

# ---------------------------------------------------------------------------
# Module-level compiled patterns (shared with gliner_extractor)
# ---------------------------------------------------------------------------

# Common words that look like entities but aren't
STOPWORDS: frozenset[str] = frozenset({
    "The", "This", "That", "These", "Those", "There", "Here",
    "It", "Its", "In", "Is", "Are", "Was", "Were", "Be", "Been",
    "For", "And", "But", "Or", "Nor", "Not", "No", "So",
    "A", "AN", "THE", "AND", "OR", "FOR", "IS", "TO", "OF",
    "BY", "AT", "ON", "AS", "IF", "DO", "UP", "WE", "MY",
    "HE", "ME", "US", "AM", "AN", "IF", "GO", "VS",
    "ALL", "HAS", "HAD", "GET", "GOT", "DID", "MAY", "CAN",
    "LET", "USE", "SET", "HOW", "WHO", "WHY", "NEW", "OLD",
    "ONE", "TWO", "KEY", "SEE", "MAX", "MIN", "TOP", "END",
    "ALSO", "MANY", "EACH", "BOTH", "SUCH", "SOME", "MORE",
    "MOST", "VERY", "WELL", "MUCH", "THAN", "THEN", "WHEN",
    "WITH", "FROM", "HAVE", "WILL", "BEEN", "INTO", "ONLY",
    "OVER", "JUST", "ALSO", "LIKE", "WHAT", "MAKE", "TAKE",
    "USED", "HELP", "MAKE", "DOES", "WIDE", "TYPE", "BEST",
    "HIGH", "LOOK", "ARGS", "NOTE", "TODO", "NONE", "TRUE",
    "LAST", "MUST", "SAME", "LONG", "NEXT", "NEED",
})

# CamelCase: TensorFlow, PyTorch, NumPy
_CAMEL_PAT = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-zA-Z]+)+\b")

# ALL-CAPS acronyms (2-10 chars): RAG, BM25, CNN, NLP
_ACRONYM_PAT = re.compile(r"\b[A-Z][A-Z0-9]{1,9}\b")

# Multi-word capitalized phrases (2-3 words): Machine Learning, Deep Learning
# Each word must start uppercase, be 2+ lowercase chars, max 3 words
# Uses [ ]+ (not \s+) to avoid matching across newlines
_MULTI_WORD_PAT = re.compile(
    r"\b[A-Z][a-z]{2,}(?: [A-Z][a-z]{2,}){1,2}\b"
)

# Acronym expansion: "Retrieval-Augmented Generation (RAG)" or "RAG (Retrieval-Augmented Generation)"
_EXPAND_PAT_1 = re.compile(
    r"([A-Z][a-z]+(?:[\s\-]+[A-Za-z]+){1,5})\s+\(([A-Z][A-Z0-9]{1,9})\)"
)
_EXPAND_PAT_2 = re.compile(
    r"([A-Z][A-Z0-9]{1,9})\s+\(([A-Z][a-z]+(?:[\s\-]+[a-z]+){1,5})\)"
)

# Trailing prepositions/conjunctions to strip from relation objects
_TRAILING_JUNK = re.compile(
    r"\s+(?:with|of|for|in|to|from|by|at|on|and|or|that|which|where|through)$",
    re.IGNORECASE,
)

# Words that indicate a phrase is a verb fragment, not an entity
_VERB_STARTS: frozenset[str] = frozenset({
    "are", "is", "was", "were", "has", "have", "had", "can", "could",
    "will", "would", "should", "may", "might", "do", "does", "did",
    "being", "been",
})

# Adverbs that shouldn't trail entity subjects
_TRAILING_ADVERBS: frozenset[str] = frozenset({
    "natively", "typically", "commonly", "generally", "usually",
    "effectively", "essentially", "primarily", "mainly", "also",
})


class RegexEntityExtractor:
    """Rule-based entity and relationship extractor using regex patterns.

    Migrated from ``src/core/knowledge_graph.py`` ``EntityExtractor``.

    Parameters
    ----------
    schema:
        Optional ``SchemaDefinition`` used to validate and refine entity type
        classification.  When ``None``, legacy heuristic typing is used.
    fallback_type:
        Node type assigned to entities that don't match any heuristic.
        Defaults to ``"concept"``.
    """

    extractor_name: str = "regex"

    # Words that commonly start sentences but aren't entity-leading words
    _SENTENCE_STARTERS: frozenset[str] = frozenset({
        "These", "Those", "This", "That", "There", "Their", "They",
        "Some", "Many", "Most", "Each", "Every", "Several", "Both",
        "One", "Two", "Three", "Four", "Five", "Other", "Another",
        "Common", "Various", "Popular", "Important",
    })

    def __init__(
        self,
        schema: Optional[SchemaDefinition] = None,
        fallback_type: str = "concept",
    ) -> None:
        self._schema = schema
        self._fallback_type = fallback_type

    @property
    def name(self) -> str:
        """Extractor identifier used by the registry."""
        return self.extractor_name

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def extract(self, text: str, source: str = "") -> ExtractionResult:
        """Extract entities and relations from *text* and return typed results.

        Parameters
        ----------
        text:
            Raw document text (may contain markdown).
        source:
            Document path or URI — stored on each produced entity / triple.

        Returns
        -------
        ExtractionResult
            Typed aggregation of extracted ``Entity`` and ``Triple`` objects.
        """
        raw_entities = self.extract_entities(text)
        raw_relations = self.extract_relations(text, raw_entities)

        entity_list: List[Entity] = [
            Entity(
                name=e,
                type=self.classify_type(e),
                sources=[source] if source else [],
                extractor_source=[self.extractor_name],
            )
            for e in raw_entities
        ]

        # Construct new Triple objects to avoid mutating shared references.
        triples: List[Triple] = [
            Triple(
                subject=t.subject,
                predicate=t.predicate,
                object=t.object,
                source=source,
                weight=t.weight,
                extractor_source=self.extractor_name,
            )
            for t in raw_relations
        ]

        logger.debug(
            "RegexEntityExtractor.extract: source=%r entities=%d triples=%d",
            source,
            len(entity_list),
            len(triples),
        )
        return ExtractionResult(entities=entity_list, triples=triples)

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------

    def extract_entities(self, text: str) -> Set[str]:
        """Extract named entities from text.

        Applies CamelCase, ALL-CAPS acronym, and multi-word capitalized phrase
        patterns after stripping markdown header lines.
        """
        # Strip entire markdown header lines before extraction
        clean = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.MULTILINE)
        entities: Set[str] = set()

        # CamelCase terms
        for m in _CAMEL_PAT.finditer(clean):
            entities.add(m.group())

        # ALL-CAPS acronyms
        for m in _ACRONYM_PAT.finditer(clean):
            term = m.group()
            if term not in STOPWORDS:
                entities.add(term)

        # Multi-word capitalized phrases
        for m in _MULTI_WORD_PAT.finditer(clean):
            term = m.group()
            words = term.split()
            # Filter: skip if first word is a common sentence starter
            if words[0] in self._SENTENCE_STARTERS:
                continue
            # Filter: skip if any word is a stopword
            if any(w in STOPWORDS for w in words):
                continue
            # Filter: must have at least 2 meaningful words
            if len(words) >= 2:
                entities.add(term)

        return entities

    # ------------------------------------------------------------------
    # Acronym alias extraction
    # ------------------------------------------------------------------

    def extract_acronym_aliases(self, text: str) -> Dict[str, str]:
        """Find acronym expansions like ``'Long Form (ACRO)'`` → ``{ACRO: Long Form}``.

        Returns a mapping of acronym → long form, which callers can use to
        resolve terms during graph construction.
        """
        aliases: Dict[str, str] = {}

        # Pattern 1: "Long Form (ACRO)"
        for m in _EXPAND_PAT_1.finditer(text):
            long_form = m.group(1).strip()
            acronym = m.group(2).strip()
            aliases[acronym] = long_form

        # Pattern 2: "ACRO (long form)"
        for m in _EXPAND_PAT_2.finditer(text):
            acronym = m.group(1).strip()
            long_form = m.group(2).strip()
            aliases[acronym] = long_form

        return aliases

    # ------------------------------------------------------------------
    # Relation extraction
    # ------------------------------------------------------------------

    def extract_relations(
        self, text: str, known_entities: Set[str]
    ) -> List[Triple]:
        """Extract typed ``Triple`` objects from *text*.

        Uses sentence-level regex patterns for common relationship forms:
        ``subset_of``, ``is_a``, ``used_for``, ``uses``, and ``is_a`` via
        "such as" enumeration patterns.

        Parameters
        ----------
        text:
            Raw document text.
        known_entities:
            Set of entity names already extracted; used to gate some patterns
            so only confirmed subjects produce relations.
        """
        relations: List[Triple] = []
        # Strip markdown headers and split into sentences
        clean_text = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.MULTILINE)
        sentences = re.split(r"[.!?\n]\s*", clean_text)

        for sentence in sentences:
            if len(sentence.strip()) < 10:
                continue

            # "X is a subset of Y"
            m = re.search(
                r"(\b[A-Z][A-Za-z\s\-]{2,30}?)\s+is\s+a\s+subset\s+of\s+"
                r"(\b[a-z][a-z\s\-]{2,40})\b",
                sentence,
            )
            if m:
                subj = m.group(1).strip()
                obj = m.group(2).strip()
                relations.append(Triple(subject=subj, predicate="subset_of", object=obj))
                continue

            # "X is a/an Y" — subject must be a known entity, object capped at ~5 words
            m = re.search(
                r"(\b[A-Z][A-Za-z\s\-]{2,30}?)\s+is\s+an?\s+"
                r"(\b[a-z][a-z\s\-]{2,50})\b",
                sentence,
            )
            if m:
                subj = m.group(1).strip()
                obj = m.group(2).strip()
                # Cap object to first ~5 words
                obj_words = obj.split()[:4]
                obj = " ".join(obj_words)
                if any(subj in e or e in subj for e in known_entities):
                    relations.append(Triple(subject=subj, predicate="is_a", object=obj))

            # "X is/are used in/for Y"
            m = re.search(
                r"(\b[A-Z][A-Za-z\s\-]{2,30}?)\s+(?:is|are)\s+(?:widely\s+)?used\s+"
                r"(?:in|for)\s+(\b[a-z][a-z\s\-,]{2,50})\b",
                sentence,
            )
            if m:
                subj = m.group(1).strip()
                obj = m.group(2).strip()
                obj_words = obj.split()[:4]
                obj = " ".join(obj_words)
                relations.append(Triple(subject=subj, predicate="used_for", object=obj))

            # "X includes/uses/supports/combines Y"
            m = re.search(
                r"(\b[A-Z][A-Za-z\s\-]{2,30}?)\s+"
                r"(?:includes?|uses?|supports?|combines?|provides?|enables?)\s+"
                r"(\b[a-z][a-z\s\-]{2,50})\b",
                sentence,
            )
            if m:
                subj = m.group(1).strip()
                obj = m.group(2).strip()
                obj_words = obj.split()[:4]
                obj = " ".join(obj_words)
                if any(subj in e or e in subj for e in known_entities):
                    relations.append(Triple(subject=subj, predicate="uses", object=obj))

            # "X such as Y, Z, ..." — expands to multiple is_a relations
            m = re.search(
                r"(\b[a-z][a-z\s\-]{2,30}?)\s+such\s+as\s+([A-Z][\w\s,\-]{2,80})",
                sentence,
            )
            if m:
                category = m.group(1).strip()
                examples_str = m.group(2).strip().rstrip("., ")
                examples = [e.strip() for e in re.split(r",\s*(?:and\s+)?", examples_str)]
                for example in examples:
                    if example and example[0].isupper() and len(example.split()) <= 4:
                        relations.append(
                            Triple(subject=example, predicate="is_a", object=category)
                        )

        return relations

    # ------------------------------------------------------------------
    # Type classification
    # ------------------------------------------------------------------

    def classify_type(self, name: str) -> str:
        """Classify *name* into a KG node type string.

        When a ``SchemaDefinition`` is provided, tries schema-based
        classification first; falls back to legacy heuristics otherwise.

        Parameters
        ----------
        name:
            Canonical entity name to classify.

        Returns
        -------
        str
            A node type string (e.g. ``"technology"``, ``"acronym"``,
            ``"concept"``).
        """
        if self._schema:
            # Heuristic guess at candidate type
            if re.match(r"^[A-Z][a-z]+[A-Z]", name):
                candidate = "technology"  # fallback for CamelCase
            elif re.match(r"^[A-Z][A-Z0-9]+$", name):
                candidate = "acronym"
            else:
                candidate = self._fallback_type
            # Return candidate if valid in schema, otherwise fall through to legacy
            if self._schema.is_valid_node_type(candidate, "phase_1"):
                return candidate

        # Legacy heuristics (no schema or candidate not in schema)
        if re.match(r"^[A-Z][a-z]+[A-Z]", name):
            return "technology"
        if re.match(r"^[A-Z][A-Z0-9]+$", name):
            return "acronym"
        return self._fallback_type
