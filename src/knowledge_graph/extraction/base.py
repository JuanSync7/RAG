# @summary
# EntityExtractor protocol: formal contract for all KG extractor implementations.
# Exports: EntityExtractor
# Deps: typing, src.knowledge_graph.common.schemas
# @end-summary
"""EntityExtractor protocol for all KG extractors."""

from __future__ import annotations

from typing import List, Protocol, Set, runtime_checkable

from src.knowledge_graph.common import Triple


__all__ = ["EntityExtractor"]


@runtime_checkable
class EntityExtractor(Protocol):
    """Structural protocol that all KG extractor implementations must satisfy."""

    @property
    def name(self) -> str: ...

    def extract_entities(self, text: str) -> Set[str]: ...

    def extract_relations(self, text: str, known_entities: Set[str]) -> List[Triple]: ...
