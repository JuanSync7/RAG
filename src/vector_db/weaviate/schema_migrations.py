# @summary
# Idempotent schema-migration helpers for Weaviate collections.
# Adds the table-aware + page-provenance properties (PR #100) to collections
# that pre-date the change. Reads the canonical property list from
# ``store.TABLE_AWARE_PROPERTIES`` so the migration cannot drift from the
# create path.
# Exports: diff_missing_table_properties, apply_table_property_migration,
#          format_missing_diff, MigrationResult
# Deps: weaviate, src.vector_db.weaviate.store
# @end-summary
"""Schema-migration helpers for the Weaviate vector store.

Background
----------
PR #100 added 12 new properties to ``ensure_collection``: ``chunk_type``,
``table_id``, ``table_group_id``, ``table_row_index``, ``table_num_rows``,
``table_num_cols``, ``table_has_header``, ``table_caption``,
``table_markdown``, ``page_no``, ``page_label``, ``page_bbox``.

Collections created before PR #100 lack these properties. Weaviate's class
schema is strict on writes — ``add_documents`` (correctly) drops any field
that isn't declared on the target collection, so table metadata silently
disappears on legacy collections.

This module exposes a non-destructive migration: it diffs the live schema
against ``store.TABLE_AWARE_PROPERTIES`` and adds whatever is missing using
the v4 ``collections.config.add_property`` API. Existing data is untouched.
Re-running the migration is a no-op — every operation is keyed by property
name.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from weaviate.classes.config import Property

from src.vector_db.weaviate.store import TABLE_AWARE_PROPERTIES


@dataclass
class MigrationResult:
    """Outcome of a migration run against a single collection."""

    collection: str
    missing_before: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    already_present: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _collection_property_names(client, collection: str) -> set[str]:
    """Return the set of property names currently declared on ``collection``."""
    col = client.collections.get(collection)
    config = col.config.get()
    raw_props = getattr(config, "properties", []) or []
    return {getattr(p, "name", "") for p in raw_props if getattr(p, "name", "")}


def diff_missing_table_properties(
    client,
    collection: str,
    *,
    expected: Optional[Iterable[Property]] = None,
) -> list[Property]:
    """Return the subset of ``expected`` whose ``name`` is missing from ``collection``.

    Defaults to the canonical ``TABLE_AWARE_PROPERTIES`` list. The order of
    the returned list matches the declaration order, so dry-run output is
    deterministic and matches ``ensure_collection``.

    Raises:
        KeyError: if the collection does not exist on the server.
    """
    if not client.collections.exists(collection):
        raise KeyError(f"collection does not exist: {collection!r}")
    declared = _collection_property_names(client, collection)
    props = list(expected) if expected is not None else list(TABLE_AWARE_PROPERTIES)
    return [p for p in props if getattr(p, "name", "") not in declared]


def _format_flags(prop: Property) -> str:
    flags: list[str] = []
    if getattr(prop, "index_filterable", None) is True:
        flags.append("filterable")
    if getattr(prop, "index_searchable", None) is True:
        flags.append("searchable")
    return ", ".join(flags) if flags else "default-index"


def format_missing_diff(missing: list[Property]) -> list[str]:
    """Render a one-line-per-property diff: ``[MISSING] name (TYPE, flags)``."""
    lines: list[str] = []
    for prop in missing:
        dtype = getattr(prop.data_type, "name", str(prop.data_type)).upper()
        lines.append(f"[MISSING] {prop.name} ({dtype}, {_format_flags(prop)})")
    return lines


def apply_table_property_migration(
    client,
    collection: str,
    *,
    expected: Optional[Iterable[Property]] = None,
) -> MigrationResult:
    """Add any missing table/page-provenance properties to ``collection``.

    Idempotent: properties already present are left alone (recorded under
    ``already_present``). Failures on individual ``add_property`` calls are
    captured in ``result.errors`` and do not abort the loop — the caller
    decides whether a partial migration is acceptable.

    Raises:
        KeyError: if the collection does not exist on the server.
    """
    props = list(expected) if expected is not None else list(TABLE_AWARE_PROPERTIES)
    missing = diff_missing_table_properties(client, collection, expected=props)
    missing_names = {p.name for p in missing}
    result = MigrationResult(
        collection=collection,
        missing_before=[p.name for p in missing],
        already_present=[p.name for p in props if p.name not in missing_names],
    )
    if not missing:
        return result

    col = client.collections.get(collection)
    for prop in missing:
        try:
            col.config.add_property(prop)
            result.added.append(prop.name)
        except Exception as exc:  # noqa: BLE001 - surface as a structured error
            result.errors.append((prop.name, f"{type(exc).__name__}: {exc}"))
    return result
