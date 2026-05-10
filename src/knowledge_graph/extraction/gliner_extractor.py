# @summary
# GLiNER-based zero-shot NER entity extractor for the KG subsystem.
# Exports: GLiNEREntityExtractor
# Deps: re, logging, typing, src.knowledge_graph.common.schemas, src.knowledge_graph.common.types,
#       src.knowledge_graph.common.utils, src.knowledge_graph.extraction.regex_extractor,
#       config.settings
# @end-summary
"""GLiNER entity extractor for the KG subsystem.

Uses a GLiNER zero-shot NER model to extract entities from text, driven by
YAML-schema-derived labels.  Acronym alias detection and relation extraction
are delegated to ``RegexEntityExtractor``, which handles those pattern-based
tasks independently of the NER model.

If GLiNER is not installed in the environment, the extractor logs a warning
and falls back transparently to ``RegexEntityExtractor`` for all extraction.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Set, TYPE_CHECKING

from src.knowledge_graph.common import (
    Entity,
    ExtractionResult,
    Triple,
)

if TYPE_CHECKING:
    from src.knowledge_graph.common import (
        KGConfig,
        SchemaDefinition,
    )

__all__ = ["GLiNEREntityExtractor"]

logger = logging.getLogger("rag.knowledge_graph.extraction.gliner")

# Common words that look like entities but are noise — kept in sync with the
# regex extractor's stopword list so both extractors filter consistently.
_STOPWORDS = frozenset({
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

#: Name reported in ``Entity.extractor_source`` and ``Triple.extractor_source``.
_EXTRACTOR_NAME = "gliner"


class GLiNEREntityExtractor:
    """Zero-shot NER entity extractor using GLiNER.

    Uses YAML-schema-derived labels for entity extraction.
    Delegates acronym alias detection and relation extraction to
    ``RegexEntityExtractor``, which handles those pattern-based tasks.

    If GLiNER is unavailable (``ImportError``), the extractor logs a warning
    and falls back to ``RegexEntityExtractor`` for all extraction calls.

    Parameters
    ----------
    model_path:
        Path or HuggingFace model ID for the GLiNER model.  When ``None`` the
        value is read from ``config.settings.GLINER_MODEL_PATH``.
    labels:
        Explicit list of NER label strings.  When ``None`` they are derived
        from the YAML schema via ``derive_gliner_labels()``.  Providing labels
        directly is useful for tests and one-off runs that skip schema loading.
    schema:
        ``SchemaDefinition`` used to derive labels when *labels* is ``None``.
        Either *labels* or *schema* must be provided if no fallback is desired.
    runtime_phase:
        Active schema phase used together with *schema* to filter active node
        types.  Defaults to ``"phase_1"``.
    """

    @property
    def name(self) -> str:
        """Extractor identifier reported in Entity.extractor_source."""
        return _EXTRACTOR_NAME

    def __init__(
        self,
        model_path: Optional[str] = None,
        labels: Optional[List[str]] = None,
        schema: Optional["SchemaDefinition"] = None,
        runtime_phase: str = "phase_1",
    ) -> None:
        self._model = None
        self._gliner_available = False
        self._regex_extractor = None  # lazy — avoids circular import at class-def time

        # Resolve NER labels: explicit list > schema-derived > empty (fallback mode).
        if labels is not None:
            self._labels: List[str] = list(labels)
        elif schema is not None:
            from src.knowledge_graph.common import derive_gliner_labels
            self._labels = derive_gliner_labels(schema, runtime_phase)
        else:
            self._labels = []

        # Attempt to load GLiNER; fall back gracefully if unavailable.
        try:
            from gliner import GLiNER  # type: ignore[import]

            if model_path is None:
                try:
                    from config.settings import GLINER_MODEL_PATH  # type: ignore[import]
                    model_path = GLINER_MODEL_PATH
                except ImportError:
                    logger.warning(
                        "config.settings.GLINER_MODEL_PATH not found; "
                        "GLiNER model path must be supplied explicitly."
                    )

            if model_path:
                self._model = GLiNER.from_pretrained(model_path, local_files_only=True)
                self._gliner_available = True
                logger.debug("GLiNER model loaded from '%s'.", model_path)
            else:
                logger.warning(
                    "No GLiNER model path provided and config lookup failed. "
                    "GLiNEREntityExtractor will fall back to RegexEntityExtractor."
                )

        except ImportError:
            logger.warning(
                "GLiNER is not installed in this environment. "
                "GLiNEREntityExtractor will fall back to RegexEntityExtractor."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_regex_extractor(self):
        """Return a lazily-initialised ``RegexEntityExtractor`` instance."""
        if self._regex_extractor is None:
            from src.knowledge_graph.extraction.regex_extractor import RegexEntityExtractor
            self._regex_extractor = RegexEntityExtractor()
        return self._regex_extractor

    def _extract_entities_gliner(self, text: str) -> List[str]:
        """Run the GLiNER model and return a deduplicated list of entity strings.

        Strips markdown headers before prediction and filters short tokens and
        stopword matches in the same way as the original implementation.

        Parameters
        ----------
        text:
            Raw chunk text to process.
        """
        clean = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.MULTILINE)

        entities: List[str] = []
        seen: set[str] = set()

        from config.settings import RAG_KG_GLINER_THRESHOLD
        predictions = self._model.predict_entities(clean, self._labels, threshold=RAG_KG_GLINER_THRESHOLD)
        for pred in predictions:
            entity_text = pred["text"].strip()
            if len(entity_text) <= 2:
                continue
            if entity_text in _STOPWORDS or entity_text.upper() in _STOPWORDS:
                continue
            if entity_text not in seen:
                seen.add(entity_text)
                entities.append(entity_text)

        return entities

    # ------------------------------------------------------------------
    # EntityExtractor protocol methods
    # ------------------------------------------------------------------

    def extract_entities(self, text: str) -> Set[str]:
        """Return entity name strings from *text* (EntityExtractor protocol)."""
        if self._gliner_available:
            return set(self._extract_entities_gliner(text))
        return self._get_regex_extractor().extract_entities(text)

    def extract_relations(self, text: str, known_entities: Set[str]) -> List[Triple]:
        """Return relation triples from *text* (EntityExtractor protocol)."""
        return self._get_regex_extractor().extract_relations(text, known_entities)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str, source: str = "") -> ExtractionResult:
        """Extract entities, aliases, and relations from *text*.

        When GLiNER is available, entity names are extracted via the NER model
        and wrapped into ``Entity`` objects with ``extractor_source="gliner"``.
        Acronym aliases and relation triples are always sourced from
        ``RegexEntityExtractor``.

        When GLiNER is unavailable, the full extraction is delegated to
        ``RegexEntityExtractor`` and its result is returned unchanged.

        Parameters
        ----------
        text:
            The raw text of a document chunk.
        source:
            Document path or URI used to populate ``Entity.sources`` and
            ``Triple.source``.

        Returns
        -------
        ExtractionResult
            Aggregated entities, triples, and descriptions for this chunk.
        """
        if not self._gliner_available:
            return self._get_regex_extractor().extract(text, source=source)

        # --- Entity extraction via GLiNER -----------------------------------
        raw_entity_names = self._extract_entities_gliner(text)

        entities: List[Entity] = [
            Entity(
                name=name,
                type="unknown",  # type resolution is a post-extraction concern
                sources=[source] if source else [],
                extractor_source=[_EXTRACTOR_NAME],
            )
            for name in raw_entity_names
        ]

        # --- Acronym aliases via regex extractor ----------------------------
        regex = self._get_regex_extractor()
        aliases: dict[str, str] = regex.extract_acronym_aliases(text)

        # Attach aliases to matching entities (best-effort name match).
        entity_index: dict[str, Entity] = {e.name: e for e in entities}
        for acronym, long_form in aliases.items():
            # Prefer attaching to the long-form entity if it was extracted.
            target = entity_index.get(long_form) or entity_index.get(acronym)
            if target is not None and acronym not in target.aliases:
                target.aliases.append(acronym)

        # --- Relation extraction via regex extractor ------------------------
        known_names = {e.name for e in entities}
        raw_triples = regex.extract_relations(text, known_names)

        triples: List[Triple] = [
            Triple(
                subject=t.subject,
                predicate=t.predicate,
                object=t.object,
                source=source,
                extractor_source=_EXTRACTOR_NAME,
            )
            for t in raw_triples
        ]

        return ExtractionResult(entities=entities, triples=triples)
