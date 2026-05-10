# @summary
# Shared helpers for the KG subsystem: normalization, type validation, GLiNER label derivation.
# Exports: normalize_alias, validate_type, derive_gliner_labels, is_phase_active, PHASE_ORDER
# Deps: src.knowledge_graph.common.types
# @end-summary
"""Shared helpers for the KG subsystem.

All functions are deterministic and side-effect-free (no I/O, no mutations of
external state).  They are intended to be imported freely across the package.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.knowledge_graph.common.types import SchemaDefinition

__all__ = [
    "PHASE_ORDER",
    "is_phase_active",
    "normalize_alias",
    "validate_type",
    "derive_gliner_labels",
]

# Phase ordering: higher integer = later / broader phase.
PHASE_ORDER: dict[str, int] = {
    "phase_1": 1,
    "phase_1b": 2,
    "phase_2": 3,
}


def is_phase_active(type_phase: str, runtime_phase: str) -> bool:
    """Return True if *type_phase* is active when running at *runtime_phase*.

    A type is active when its phase level is less than or equal to the
    current runtime phase level.  Unknown phase strings are treated as
    beyond the highest known phase (never active on the type side, always
    on the runtime side — effectively using 99/0 sentinels).

    Parameters
    ----------
    type_phase:
        Phase tag from the schema definition (e.g. ``"phase_1"``).
    runtime_phase:
        Currently active runtime phase (e.g. ``"phase_1b"``).
    """
    return PHASE_ORDER.get(type_phase, 99) <= PHASE_ORDER.get(runtime_phase, 0)


def normalize_alias(name: str) -> str:
    """Return a normalised form of *name* for deduplication comparisons.

    Applies lower-casing, whitespace stripping, and replacement of hyphens
    and underscores with spaces so that e.g. ``"RTL_Module"``,
    ``"rtl-module"``, and ``"rtl module"`` all map to the same key.

    Parameters
    ----------
    name:
        Raw entity name or alias string.
    """
    return name.lower().strip().replace("-", " ").replace("_", " ")


def validate_type(type_name: str, schema: "SchemaDefinition", runtime_phase: str) -> bool:
    """Return True if *type_name* is a valid and active node or edge type.

    Delegates to ``SchemaDefinition.is_valid_node_type`` and
    ``SchemaDefinition.is_valid_edge_type`` so that schema method calls
    remain the single source of truth for validity logic.

    Parameters
    ----------
    type_name:
        The node or edge type name to check.
    schema:
        Parsed schema definition.
    runtime_phase:
        Currently active runtime phase.
    """
    return (
        schema.is_valid_node_type(type_name, runtime_phase)
        or schema.is_valid_edge_type(type_name, runtime_phase)
    )


def derive_gliner_labels(schema: "SchemaDefinition", runtime_phase: str) -> List[str]:
    """Return the ordered list of GLiNER label strings for active node types.

    Uses ``gliner_label`` when set; falls back to the type ``name`` otherwise.
    Only types active in *runtime_phase* are included.

    Parameters
    ----------
    schema:
        Parsed schema definition.
    runtime_phase:
        Currently active runtime phase.
    """
    labels: List[str] = []
    for node in schema.active_node_types(runtime_phase):
        labels.append(node.gliner_label if node.gliner_label else node.name)
    return labels
