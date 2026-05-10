# @summary
# Schema validation for KG retrieval config fields.
# Validates edge types and path patterns against kg_schema.yaml.
# Exports: KGConfigValidationError, PatternWarning, validate_edge_types, validate_path_patterns
# Deps: yaml, src.knowledge_graph.common (load_schema if available)
# validate_path_patterns accepts optional strict=True to promote warnings to
#   KGConfigValidationError (REQ-KG-778, driven by KGConfig.strict_path_validation).
# @end-summary
"""Schema validation for KG retrieval configuration.

Validates retrieval_edge_types and retrieval_path_patterns against
the canonical edge type vocabulary in kg_schema.yaml.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

logger = logging.getLogger("rag.knowledge_graph.validation")


class KGConfigValidationError(Exception):
    """Raised when retrieval config fails schema validation.

    REQ-KG-1208: Accumulates all errors into a single raise.
    """

    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        formatted = "\n".join(f"  - {e}" for e in errors)
        super().__init__(f"KGConfig validation failed:\n{formatted}")


@dataclass
class PatternWarning:
    """Non-fatal warning for type-incompatible consecutive hops.

    REQ-KG-778: Returned (not raised) — callers log at WARNING level.
    """

    pattern_index: int
    hop_index: int
    edge_type_a: str
    edge_type_b: str
    message: str


def _extract_valid_edge_types(schema_path: str) -> Set[str]:
    """Load the YAML schema and return the set of all valid edge type names.

    Attempts to reuse the ``load_schema`` function from
    ``src.knowledge_graph.common`` when available; falls back to a direct
    ``yaml.safe_load`` parse so the function works even if the common package
    is not yet importable (e.g. during early bootstrap).

    Args:
        schema_path: File-system path to ``kg_schema.yaml``.

    Returns:
        Set of valid edge type name strings (both structural and semantic).

    Raises:
        FileNotFoundError: If *schema_path* does not exist.
    """
    try:
        from src.knowledge_graph.common.types import load_schema

        schema = load_schema(schema_path)
        return {e.name for e in schema.edge_types}
    except ImportError:
        pass

    # Fallback: parse YAML directly
    import yaml

    try:
        with open(schema_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except FileNotFoundError:
        raise FileNotFoundError(f"KG schema file not found: {schema_path}")

    edge_section = raw.get("edge_types", {})
    valid: Set[str] = set()
    if isinstance(edge_section, dict):
        for _category, entries in edge_section.items():
            if isinstance(entries, dict):
                valid.update(entries.keys())
    elif isinstance(edge_section, list):
        for entry in edge_section:
            if isinstance(entry, dict) and "name" in entry:
                valid.add(entry["name"])
    return valid


def _load_edge_constraints(schema_path: str) -> Dict[str, Dict[str, List[str]]]:
    """Return a mapping of edge type name to its source/target type lists.

    Used by :func:`validate_path_patterns` for hop compatibility checks.

    Returns:
        Dict mapping edge type name -> ``{"source_types": [...], "target_types": [...]}``.
        Edge types with empty/unconstrained lists are represented with empty lists.

    Raises:
        FileNotFoundError: If *schema_path* does not exist.
    """
    try:
        from src.knowledge_graph.common.types import load_schema

        schema = load_schema(schema_path)
        return {
            e.name: {"source_types": list(e.source_types), "target_types": list(e.target_types)}
            for e in schema.edge_types
        }
    except ImportError:
        pass

    import yaml

    try:
        with open(schema_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except FileNotFoundError:
        raise FileNotFoundError(f"KG schema file not found: {schema_path}")

    constraints: Dict[str, Dict[str, List[str]]] = {}
    edge_section = raw.get("edge_types", {})
    if isinstance(edge_section, dict):
        for _category, entries in edge_section.items():
            if isinstance(entries, dict):
                for edge_name, edge_def in entries.items():
                    if isinstance(edge_def, dict):
                        constraints[edge_name] = {
                            "source_types": list(edge_def.get("source_types") or []),
                            "target_types": list(edge_def.get("target_types") or []),
                        }
    elif isinstance(edge_section, list):
        for entry in edge_section:
            if isinstance(entry, dict) and "name" in entry:
                constraints[entry["name"]] = {
                    "source_types": list(entry.get("source_types") or []),
                    "target_types": list(entry.get("target_types") or []),
                }
    return constraints


def validate_edge_types(edge_types: List[str], schema_path: str) -> None:
    """Validate that every entry in *edge_types* is a known schema edge type.

    REQ-KG-1208: All unknown types are accumulated and raised together so
    operators see the full list of problems in a single error message.

    Args:
        edge_types: List of edge type name strings from
            ``KGConfig.retrieval_edge_types``.
        schema_path: File-system path to ``kg_schema.yaml``.

    Raises:
        KGConfigValidationError: If any names are not present in the schema.
        FileNotFoundError: If *schema_path* does not exist.
    """
    valid = _extract_valid_edge_types(schema_path)
    errors: List[str] = []
    for et in edge_types:
        if et not in valid:
            errors.append(
                f"Unknown edge type {et!r} in retrieval_edge_types. "
                f"Valid types: {sorted(valid)}"
            )
    if errors:
        raise KGConfigValidationError(errors)


def validate_path_patterns(
    patterns: List[List[str]],
    schema_path: str,
    strict: bool = False,
) -> List[PatternWarning]:
    """Validate path pattern edge type sequences against the schema.

    Two-pass validation:

    1. **Error pass** – every edge type label in every pattern must exist in
       the schema vocabulary. Collects all unknown types and raises
       :class:`KGConfigValidationError` if any are found.

    2. **Warning pass** – for each consecutive hop pair ``(pattern[i],
       pattern[i+1])``, check whether the *target_types* of ``pattern[i]``
       and the *source_types* of ``pattern[i+1]`` share at least one common
       entity type. If the intersection is empty (and both sides have non-empty
       constraints defined in the schema), a :class:`PatternWarning` is
       appended. Edge types with empty source/target lists (unconstrained) skip
       the compatibility check for that side.

    Args:
        patterns: List of ordered hop sequences from
            ``KGConfig.retrieval_path_patterns``.
        schema_path: File-system path to ``kg_schema.yaml``.
        strict: When True and warnings are non-empty, raises
            :class:`KGConfigValidationError` instead of returning the list.
            REQ-KG-778: Controlled by ``KGConfig.strict_path_validation``.

    Returns:
        List of :class:`PatternWarning` objects (empty when all patterns are
        fully compatible). Warnings are non-fatal unless *strict* is True.

    Raises:
        KGConfigValidationError: If any edge type labels are unknown, or if
            *strict* is True and hop-compatibility warnings are found.
        FileNotFoundError: If *schema_path* does not exist.
    """
    valid = _extract_valid_edge_types(schema_path)

    # Pass 1: accumulate all unknown edge type labels across all patterns.
    errors: List[str] = []
    for p_idx, pattern in enumerate(patterns):
        for hop_idx, edge_type in enumerate(pattern):
            if edge_type not in valid:
                errors.append(
                    f"Pattern[{p_idx}][{hop_idx}]: unknown edge type {edge_type!r}. "
                    f"Valid types: {sorted(valid)}"
                )
    if errors:
        raise KGConfigValidationError(errors)

    # Pass 2: hop-compatibility warnings.
    constraints = _load_edge_constraints(schema_path)
    warnings: List[PatternWarning] = []

    for p_idx, pattern in enumerate(patterns):
        for hop_idx in range(len(pattern) - 1):
            edge_a = pattern[hop_idx]
            edge_b = pattern[hop_idx + 1]

            target_types_a: Optional[List[str]] = (
                constraints.get(edge_a, {}).get("target_types")
            )
            source_types_b: Optional[List[str]] = (
                constraints.get(edge_b, {}).get("source_types")
            )

            # Skip check if either side is unconstrained (empty list).
            if not target_types_a or not source_types_b:
                continue

            if not set(target_types_a) & set(source_types_b):
                warnings.append(
                    PatternWarning(
                        pattern_index=p_idx,
                        hop_index=hop_idx,
                        edge_type_a=edge_a,
                        edge_type_b=edge_b,
                        message=(
                            f"Pattern[{p_idx}] hop {hop_idx}->{hop_idx + 1}: "
                            f"edge {edge_a!r} targets {sorted(target_types_a)} "
                            f"but edge {edge_b!r} expects sources {sorted(source_types_b)} "
                            f"— no compatible entity type at the boundary."
                        ),
                    )
                )

    if strict and warnings:
        errors = [w.message for w in warnings]
        raise KGConfigValidationError(errors)

    return warnings
