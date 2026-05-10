# @summary
# YAML alias-table-based entity resolution.
# Exports: AliasResolver
# Deps: yaml, pathlib, src.knowledge_graph.backend, src.knowledge_graph.resolution.schemas
# @end-summary
"""YAML alias-table-based entity resolution.

Matches entity names against a curated alias table for deterministic merges.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import yaml

from src.knowledge_graph.backend import GraphStorageBackend
from src.knowledge_graph.resolution.schemas import MergeCandidate

__all__ = ["AliasResolver"]

logger = logging.getLogger("rag.knowledge_graph.resolution.alias")


class AliasResolver:
    """Find merge candidates using a YAML alias table.

    Alias table format (``config/kg_aliases.yaml``)::

        aliases:
          - canonical: "AXI4_Arbiter"
            variants:
              - "axi4_arb"
              - "AXI_ARB"

    Matching is case-insensitive.
    """

    def __init__(self, alias_path: str = "config/kg_aliases.yaml") -> None:
        self._alias_map: Dict[str, str] = {}  # lowercase_variant → canonical
        path = Path(alias_path)
        if not path.is_file():
            logger.debug("Alias table not found at %s — operating with empty table", alias_path)
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            for group in data.get("aliases", []):
                canonical = group.get("canonical", "")
                for variant in group.get("variants", []):
                    self._alias_map[variant.lower()] = canonical
        except Exception as exc:
            logger.warning("Failed to load alias table %s: %s", alias_path, exc)

    def find_candidates(
        self, backend: GraphStorageBackend
    ) -> List[MergeCandidate]:
        """Match entity names against alias table entries.

        Returns:
            List of MergeCandidate with reason="alias_table", similarity=1.0.
        """
        if not self._alias_map:
            return []

        candidates: List[MergeCandidate] = []
        all_entities = backend.get_all_entities()
        entity_names = {e.name for e in all_entities}
        entity_names_lower = {e.name.lower(): e.name for e in all_entities}

        for variant_lower, canonical in self._alias_map.items():
            # Check if the variant exists as an entity in the graph
            if variant_lower not in entity_names_lower:
                continue
            actual_variant = entity_names_lower[variant_lower]

            # Check if the canonical name exists (case-insensitive)
            canonical_lower = canonical.lower()
            if canonical_lower not in entity_names_lower:
                continue
            actual_canonical = entity_names_lower[canonical_lower]

            # Don't merge an entity with itself
            if actual_canonical == actual_variant:
                continue

            candidates.append(
                MergeCandidate(
                    canonical=actual_canonical,
                    duplicate=actual_variant,
                    similarity=1.0,
                    reason="alias_table",
                )
            )

        logger.info("AliasResolver: found %d merge candidates", len(candidates))
        return candidates
