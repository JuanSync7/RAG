# @summary
# Typed data contracts for the KG subsystem.
# Exports: EntityDescription, Entity, Triple, ExtractionResult
# Deps: dataclasses, typing
# @end-summary
"""Typed data contracts for the KG subsystem.

All types are pure dataclasses with no business logic — they serve as the
shared interchange format between extractors, backends, and query layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


__all__ = [
    "EntityDescription",
    "Entity",
    "Triple",
    "ExtractionResult",
]


@dataclass
class EntityDescription:
    """A single textual mention of an entity in a source document.

    Attributes:
        text: The sentence or passage containing the mention.
        source: Document path or URI where the mention was found.
        chunk_id: Originating chunk identifier within the document.
    """

    text: str
    source: str
    chunk_id: str


@dataclass
class Entity:
    """A node in the knowledge graph.

    Attributes:
        name: Canonical (first-seen) name for the entity.
        type: Node type as defined in the KG schema (e.g. ``RTL_Module``).
        sources: List of document paths that mention this entity.
        mention_count: Number of times this entity has been observed.
        aliases: Alternative names or acronyms that resolve to this entity.
        raw_mentions: Ordered list of raw textual mentions collected so far.
        current_summary: LLM-generated condensed description of the entity.
        extractor_source: Names of the extractors that produced this entity.
    """

    name: str
    type: str
    sources: List[str] = field(default_factory=list)
    mention_count: int = 1
    aliases: List[str] = field(default_factory=list)
    raw_mentions: List[EntityDescription] = field(default_factory=list)
    current_summary: str = ""
    extractor_source: List[str] = field(default_factory=list)


@dataclass
class Triple:
    """A directed relationship triple: subject → predicate → object.

    Attributes:
        subject: Canonical name of the source entity.
        predicate: Edge type label (e.g. ``instantiates``, ``specified_by``).
        object: Canonical name of the target entity.
        source: Document path or URI from which this triple was extracted.
        weight: Confidence or frequency weight for the relationship.
        extractor_source: Name of the extractor that produced this triple.
    """

    subject: str
    predicate: str
    object: str
    source: str = ""
    weight: float = 1.0
    extractor_source: str = ""


@dataclass
class ExtractionResult:
    """Aggregated output from a single extractor run.

    Attributes:
        entities: List of extracted entities.
        triples: List of extracted relationship triples.
        descriptions: Per-entity mention lists keyed by canonical entity name.
    """

    entities: List[Entity] = field(default_factory=list)
    triples: List[Triple] = field(default_factory=list)
    descriptions: Dict[str, List[EntityDescription]] = field(default_factory=dict)
