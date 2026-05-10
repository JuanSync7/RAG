# @summary
# Schema-guided LLM entity/relation extractor using structured JSON output.
# Uses LLMProvider.json_completion() with YAML schema context injected into prompts.
# Validates results against SchemaDefinition, handles retries and rate limits.
# Exports: LLMEntityExtractor
# Deps: json, logging, time, src.knowledge_graph.common.schemas,
#        src.knowledge_graph.common.types, src.platform.llm.provider,
#        config.settings
# @end-summary
"""LLM-based entity and relationship extraction.

Uses structured JSON output (JSON mode) to extract entities, relationships,
and descriptions from document chunks. The YAML schema is injected into the
extraction prompt as context so the LLM produces typed, schema-conformant output.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Set

from src.knowledge_graph.common import (
    Entity,
    EntityDescription,
    ExtractionResult,
    Triple,
)
from src.knowledge_graph.common import (
    KGConfig,
    SchemaDefinition,
)

logger = logging.getLogger(__name__)

__all__ = ["LLMEntityExtractor"]

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

REQUIRED_TEMPLATE_VARS: frozenset[str] = frozenset({
    "schema_types", "schema_edges", "extraction_hints", "chunk_text",
})

SYSTEM_MESSAGE: str = (
    "You are a precise entity and relationship extraction assistant. "
    "Your task is to identify named entities, their types, short descriptions, "
    "and relationships from the provided text. "
    "You MUST respond with a single valid JSON object and nothing else. "
    "The JSON must have exactly two top-level keys: \"entities\" and \"relationships\".\n\n"
    "Output format:\n"
    "{\n"
    '  "entities": [\n'
    '    {"name": "EntityName", "type": "NodeType", "description": "One-sentence description."}\n'
    "  ],\n"
    '  "relationships": [\n'
    '    {"source": "EntityA", "relation": "edge_type", "target": "EntityB", '
    '"description": "Brief explanation of the relationship."}\n'
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    "- Use ONLY the entity types and relationship types listed below.\n"
    "- Entity names should be canonical (e.g. 'TensorFlow' not 'tensorflow').\n"
    "- Descriptions should be concise — one sentence each.\n"
    "- If you cannot find any entities or relationships, return empty lists.\n"
    "- Do NOT invent entities or relationships that are not supported by the text."
)

DEFAULT_PROMPT_TEMPLATE: str = (
    "## Allowed Entity Types\n"
    "{schema_types}\n\n"
    "## Allowed Relationship Types\n"
    "{schema_edges}\n\n"
    "## Extraction Hints\n"
    "{extraction_hints}\n\n"
    "## Text to Analyse\n"
    "```\n{chunk_text}\n```\n\n"
    "Extract all entities and relationships from the text above. "
    "Return a JSON object with \"entities\" and \"relationships\" arrays."
)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class LLMEntityExtractor:
    """Schema-guided LLM entity/relation extractor.

    Uses a single LLM call per chunk with the YAML schema injected as
    prompt context. Returns ExtractionResult compatible with the merge node.

    Parameters
    ----------
    schema:
        Parsed YAML schema for type validation and prompt rendering.
    config:
        KG runtime configuration.
    llm_provider:
        Optional LLMProvider instance. Falls back to
        ``get_llm_provider()`` singleton when ``None``.
    """

    extractor_name: str = "llm"

    def __init__(
        self,
        schema: SchemaDefinition,
        config: KGConfig,
        llm_provider: Optional[Any] = None,
    ) -> None:
        self._schema = schema
        self._config = config

        if llm_provider is not None:
            self._llm = llm_provider
        else:
            from src.platform.llm import get_llm_provider
            self._llm = get_llm_provider()

        self._template = self._load_template()

        # Pre-render schema sections (stable for the extractor's lifetime)
        self._schema_types_block = self._render_schema_types()
        self._schema_edges_block = self._render_schema_edges()
        self._hints_block = self._render_extraction_hints()

        # Build lookup sets for validation
        active_nodes = self._schema.active_node_types(self._config.runtime_phase)
        active_edges = self._schema.active_edge_types(self._config.runtime_phase)
        self._valid_node_types: set[str] = {n.name for n in active_nodes}
        self._valid_edge_types: set[str] = {e.name for e in active_edges}

        logger.info(
            "LLMEntityExtractor initialised: model=%s, %d node types, %d edge types",
            self._config.llm_extraction_model,
            len(self._valid_node_types),
            len(self._valid_edge_types),
        )

    @property
    def name(self) -> str:
        """Extractor identifier used by the registry."""
        return self.extractor_name

    # ------------------------------------------------------------------
    # EntityExtractor protocol methods
    # ------------------------------------------------------------------

    def extract_entities(self, text: str) -> Set[str]:
        """Return entity name strings from *text* (EntityExtractor protocol)."""
        return {e.name for e in self.extract(text).entities}

    def extract_relations(self, text: str, known_entities: Set[str]) -> List[Triple]:
        """Return relation triples from *text* (EntityExtractor protocol)."""
        return self.extract(text).triples

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def extract(self, text: str, source: str = "") -> ExtractionResult:
        """Extract entities, triples, and descriptions from a text chunk.

        Args:
            text: Document chunk text.
            source: Document path or URI for provenance.

        Returns:
            ExtractionResult with entities, triples, and descriptions.
            Returns empty result on persistent LLM failure (no exception).
        """
        if not text or not text.strip():
            return ExtractionResult()

        t0 = time.monotonic()
        max_attempts = 1 + self._config.llm_extraction_max_retries

        for attempt in range(max_attempts):
            try:
                messages = self._build_prompt(text)
                raw_json = self._call_llm(messages)
                parsed = self._parse_response(raw_json)
                result = self._validate_and_build(parsed, source)

                elapsed = time.monotonic() - t0
                logger.info(
                    "LLM extraction: %d entities, %d triples in %.2fs "
                    "(source=%s, attempt=%d)",
                    len(result.entities),
                    len(result.triples),
                    elapsed,
                    source or "<unknown>",
                    attempt + 1,
                )
                return result

            except json.JSONDecodeError as exc:
                logger.warning(
                    "LLM JSON parse failure (attempt %d/%d): %s",
                    attempt + 1, max_attempts, exc,
                )
            except Exception as exc:
                logger.warning(
                    "LLM extraction failure (attempt %d/%d): %s",
                    attempt + 1, max_attempts, exc,
                )

        elapsed = time.monotonic() - t0
        logger.error(
            "LLM extraction failed after %d attempts in %.2fs (source=%s)",
            max_attempts, elapsed, source or "<unknown>",
        )
        return ExtractionResult()

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, text: str) -> List[Dict[str, str]]:
        """Render the prompt template with schema context and chunk text.

        Args:
            text: The chunk text to embed in the prompt.

        Returns:
            OpenAI-style messages list (system + user).
        """
        user_content = self._template.format(
            schema_types=self._schema_types_block,
            schema_edges=self._schema_edges_block,
            extraction_hints=self._hints_block,
            chunk_text=text,
        )
        return [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_content},
        ]

    def _render_schema_types(self) -> str:
        """Render active node types as a compact string for prompt injection."""
        active = self._schema.active_node_types(self._config.runtime_phase)
        if not active:
            return "No entity types defined."

        lines: list[str] = []
        for nt in active:
            line = f"- **{nt.name}**: {nt.description}"
            if nt.extraction_hints:
                line += f" (hints: {nt.extraction_hints})"
            lines.append(line)
        return "\n".join(lines)

    def _render_schema_edges(self) -> str:
        """Render active edge types with descriptions and constraints."""
        active = self._schema.active_edge_types(self._config.runtime_phase)
        if not active:
            return "No relationship types defined."

        lines: list[str] = []
        for et in active:
            constraint = ""
            if et.source_types or et.target_types:
                src = ", ".join(et.source_types) if et.source_types else "any"
                tgt = ", ".join(et.target_types) if et.target_types else "any"
                constraint = f" [source: {src} -> target: {tgt}]"
            lines.append(f"- **{et.name}**: {et.description}{constraint}")
        return "\n".join(lines)

    def _render_extraction_hints(self) -> str:
        """Render concatenated extraction hints from active node types."""
        active = self._schema.active_node_types(self._config.runtime_phase)
        hints: list[str] = []
        for nt in active:
            if nt.extraction_hints:
                hints.append(f"- {nt.name}: {nt.extraction_hints}")
        return "\n".join(hints) if hints else "No additional hints."

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call LLMProvider.json_completion() with rate-limit retry.

        Implements exponential backoff (1s, 2s, 4s) for HTTP 429 errors,
        up to 3 rate-limit retries.

        Args:
            messages: Chat messages for the LLM call.

        Returns:
            Raw JSON string from the LLM response.

        Raises:
            Exception: Re-raises non-rate-limit errors after logging.
        """
        from config.settings import RAG_KG_LLM_RATE_LIMIT_RETRIES, RAG_KG_LLM_RATE_LIMIT_BACKOFF_S
        max_rate_limit_retries = RAG_KG_LLM_RATE_LIMIT_RETRIES
        backoff_seconds = RAG_KG_LLM_RATE_LIMIT_BACKOFF_S

        for attempt in range(max_rate_limit_retries + 1):
            try:
                model_alias = self._config.llm_extraction_model
                response = self._llm.json_completion(
                    messages,
                    model_alias=model_alias,
                    temperature=self._config.llm_extraction_temperature,
                )
                return response.content
            except Exception as exc:
                # Check for rate-limit (HTTP 429) errors
                exc_str = str(exc).lower()
                is_rate_limit = (
                    "429" in exc_str
                    or "rate" in exc_str
                    or "rate_limit" in exc_str
                )
                if is_rate_limit and attempt < max_rate_limit_retries:
                    logger.warning(
                        "Rate limited (attempt %d/%d), backing off %.1fs",
                        attempt + 1, max_rate_limit_retries, backoff_seconds,
                    )
                    time.sleep(backoff_seconds)
                    backoff_seconds *= 2.0
                    continue
                raise

        # Should not reach here, but satisfy type checker
        raise RuntimeError("Exhausted rate-limit retries")  # pragma: no cover

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, json_str: str) -> Dict[str, Any]:
        """Parse LLM JSON response into a dict.

        Args:
            json_str: Raw JSON string from LLM.

        Returns:
            Parsed dict with 'entities' and 'relationships' keys.

        Raises:
            json.JSONDecodeError: On malformed JSON.
        """
        # Strip markdown code fences if present
        cleaned = json_str.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (possibly ```json)
            first_newline = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
            cleaned = cleaned[first_newline + 1:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        parsed = json.loads(cleaned)

        # Normalise: ensure both keys exist
        if not isinstance(parsed, dict):
            raise json.JSONDecodeError(
                "Expected JSON object at top level", json_str, 0
            )
        parsed.setdefault("entities", [])
        parsed.setdefault("relationships", [])
        return parsed

    # ------------------------------------------------------------------
    # Validation and conversion
    # ------------------------------------------------------------------

    def _validate_and_build(
        self, raw: Dict[str, Any], source: str
    ) -> ExtractionResult:
        """Validate extracted data against schema and build ExtractionResult.

        - Entities with invalid types: reclassify to fallback type, log warning.
        - Triples with invalid predicates: drop, log warning.
        - Sets extractor_source="llm" on all Entity and Triple objects.
        - Builds EntityDescription entries from entity description fields.

        Args:
            raw: Parsed LLM output dict.
            source: Document source for provenance.

        Returns:
            Validated ExtractionResult.
        """
        fallback_type = self._config.regex_fallback_type
        entities: List[Entity] = []
        descriptions: Dict[str, List[EntityDescription]] = {}
        seen_entity_names: set[str] = set()

        for raw_entity in raw.get("entities", []):
            if not isinstance(raw_entity, dict):
                continue

            name = str(raw_entity.get("name", "")).strip()
            if not name:
                continue

            entity_type = str(raw_entity.get("type", fallback_type)).strip()
            desc_text = str(raw_entity.get("description", "")).strip()

            # Validate type against schema
            if entity_type not in self._valid_node_types:
                logger.debug(
                    "Entity '%s' has unknown type '%s', reclassifying to '%s'",
                    name, entity_type, fallback_type,
                )
                entity_type = fallback_type

            # Deduplicate within this extraction
            if name in seen_entity_names:
                continue
            seen_entity_names.add(name)

            entity = Entity(
                name=name,
                type=entity_type,
                sources=[source] if source else [],
                extractor_source=[self.extractor_name],
            )
            entities.append(entity)

            # Build EntityDescription if the LLM provided one
            if desc_text:
                ed = EntityDescription(
                    text=desc_text,
                    source=source,
                    chunk_id=source,  # chunk_id = source as best available
                )
                descriptions.setdefault(name, []).append(ed)

        # --- Triples ---
        triples: List[Triple] = []
        for raw_rel in raw.get("relationships", []):
            if not isinstance(raw_rel, dict):
                continue

            subj = str(raw_rel.get("source", "")).strip()
            pred = str(raw_rel.get("relation", "")).strip()
            obj = str(raw_rel.get("target", "")).strip()

            if not subj or not pred or not obj:
                continue

            # Validate predicate against schema
            if pred not in self._valid_edge_types:
                logger.debug(
                    "Dropping triple (%s)-[%s]->(%s): unknown edge type",
                    subj, pred, obj,
                )
                continue

            triples.append(Triple(
                subject=subj,
                predicate=pred,
                object=obj,
                source=source,
                weight=1.0,
                extractor_source=self.extractor_name,
            ))

        return ExtractionResult(
            entities=entities,
            triples=triples,
            descriptions=descriptions,
        )

    # ------------------------------------------------------------------
    # Template loading
    # ------------------------------------------------------------------

    def _load_template(self) -> str:
        """Load prompt template from config path or use default.

        Returns:
            Template string with substitution variables.

        Raises:
            ValueError: If template is missing required variables.
        """
        template_path = self._config.llm_extraction_prompt_template
        if template_path:
            try:
                with open(template_path, "r", encoding="utf-8") as fh:
                    template = fh.read()
            except FileNotFoundError:
                logger.warning(
                    "Prompt template not found at '%s', using default",
                    template_path,
                )
                template = DEFAULT_PROMPT_TEMPLATE
        else:
            template = DEFAULT_PROMPT_TEMPLATE

        # Validate required variables
        missing = REQUIRED_TEMPLATE_VARS - {
            v for v in REQUIRED_TEMPLATE_VARS
            if f"{{{v}}}" in template
        }
        if missing:
            raise ValueError(
                f"Prompt template is missing required variables: {missing}"
            )

        return template
