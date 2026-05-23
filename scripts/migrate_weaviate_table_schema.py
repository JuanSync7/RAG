#!/usr/bin/env python3
"""Add table-aware + page-provenance properties to legacy Weaviate collections.

PR #100 added 12 properties to the Weaviate collection schema. Collections
created before that change still work for reads (missing properties read as
``None``), but new writes silently drop the table/page-provenance metadata
because ``add_documents`` filters against the live schema.

This script is the non-destructive backfill. It diffs the live schema against
``src.vector_db.weaviate.store.TABLE_AWARE_PROPERTIES`` and adds whatever is
missing. Existing data and other properties are untouched.

Usage::

    # Dry-run a single collection (prints the diff, no writes)
    uv run python scripts/migrate_weaviate_table_schema.py \\
        --collection RagWeave --dry-run

    # Apply against a single collection
    uv run python scripts/migrate_weaviate_table_schema.py --collection RagWeave

    # Apply against every collection visible to the client
    uv run python scripts/migrate_weaviate_table_schema.py --all-collections

The script honours ``RAG_WEAVIATE_MODE``/``RAG_WEAVIATE_HOST``/etc. via the
existing client builder in ``src.vector_db.weaviate.store`` — there is no
separate connection config.
"""
from __future__ import annotations

import argparse
import sys

from src.vector_db.weaviate.schema_migrations import (
    MigrationResult,
    apply_table_property_migration,
    diff_missing_table_properties,
    format_missing_diff,
)
from src.vector_db.weaviate.store import (
    TABLE_AWARE_PROPERTIES,
    get_weaviate_client,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill table-aware schema properties on Weaviate collections.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--collection",
        help="Name of a single Weaviate collection to migrate.",
    )
    target.add_argument(
        "--all-collections",
        action="store_true",
        help="Migrate every collection visible to the client.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the diff but do not modify the schema.",
    )
    return parser.parse_args(argv)


def _dry_run_one(client, collection: str) -> int:
    """Print missing properties for ``collection``. Returns exit-code contribution."""
    try:
        missing = diff_missing_table_properties(client, collection)
    except KeyError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    print(f"=== {collection} ===")
    if not missing:
        print(f"[OK] all {len(TABLE_AWARE_PROPERTIES)} properties present")
        return 0
    for line in format_missing_diff(missing):
        print(line)
    print(f"[SUMMARY] {len(missing)} missing / {len(TABLE_AWARE_PROPERTIES)} total")
    return 0


def _apply_one(client, collection: str) -> tuple[int, MigrationResult | None]:
    print(f"=== {collection} ===")
    try:
        result = apply_table_property_migration(client, collection)
    except KeyError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2, None
    if not result.missing_before:
        print(f"[OK] all {len(TABLE_AWARE_PROPERTIES)} properties present")
        return 0, result
    for name in result.added:
        print(f"[ADDED] {name}")
    for name, err in result.errors:
        print(f"[FAILED] {name}: {err}", file=sys.stderr)
    print(
        f"[SUMMARY] added={len(result.added)} "
        f"already_present={len(result.already_present)} "
        f"errors={len(result.errors)}"
    )
    return (0 if result.ok else 1), result


def _enumerate_collections(client) -> list[str]:
    all_cols = client.collections.list_all(simple=True)
    # ``list_all`` returns dict-or-list-of-names depending on flavour.
    if hasattr(all_cols, "keys"):
        return list(all_cols.keys())
    return list(all_cols)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    rc = 0
    with get_weaviate_client() as client:
        if args.all_collections:
            targets = _enumerate_collections(client)
            if not targets:
                print("[INFO] no collections found on the server")
                return 0
        else:
            targets = [args.collection]

        for name in targets:
            if args.dry_run:
                rc = max(rc, _dry_run_one(client, name))
            else:
                step_rc, _ = _apply_one(client, name)
                rc = max(rc, step_rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
