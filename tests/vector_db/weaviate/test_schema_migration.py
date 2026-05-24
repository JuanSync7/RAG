"""Tests for ``schema_migrations.apply_table_property_migration``.

Covers the migration tool that backfills the 15 table-aware + page-provenance
properties (PR #100 + document_id + xref_targets follow-ups) onto Weaviate
collections created before the change.

Uses MagicMock against the v4 ``weaviate.WeaviateClient`` surface; no live
Weaviate. The mock collection tracks ``add_property`` calls so we can verify
idempotency by re-running the migration against the same client.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.vector_db.weaviate.schema_migrations import (
    MigrationResult,
    apply_table_property_migration,
    diff_missing_table_properties,
    format_missing_diff,
)
from src.vector_db.weaviate.store import TABLE_AWARE_PROPERTIES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stub_client(
    initial_prop_names: list[str], *, exists: bool = True
) -> MagicMock:
    """Build a MagicMock client whose collection tracks add_property calls.

    The stub keeps a mutable list of Property-like objects with a ``name``
    attribute. ``config.get().properties`` re-reads it each call, so
    re-running the migration sees the post-add state — that's what makes the
    idempotency assertion meaningful.
    """
    state: list[MagicMock] = []
    for name in initial_prop_names:
        p = MagicMock()
        p.name = name
        state.append(p)

    col = MagicMock()

    def _get_config():
        cfg = MagicMock()
        cfg.properties = list(state)
        return cfg

    col.config.get.side_effect = _get_config

    def _add_property(prop):
        # Real v4 client mutates the schema; reflect that in our state list.
        # We don't deep-copy here — the migration only reads ``name``.
        state.append(prop)

    col.config.add_property.side_effect = _add_property

    client = MagicMock()
    client.collections.exists.return_value = exists
    client.collections.get.return_value = col
    # Expose state for tests that want to introspect post-migration state.
    client._table_state = state  # type: ignore[attr-defined]
    return client


# ---------------------------------------------------------------------------
# Dry-run / diff behaviour
# ---------------------------------------------------------------------------

class TestDryRunDiff:
    def test_diff_reports_all_fourteen_missing_on_legacy_collection(self):
        # Legacy name retained for git-history readability — the migration
        # now ships 16 table-aware properties (FIG-2 added figure_image_uri).
        client = _make_stub_client(["text", "source", "source_key"])

        missing = diff_missing_table_properties(client, "Legacy")

        assert [p.name for p in missing] == [p.name for p in TABLE_AWARE_PROPERTIES]
        assert len(missing) == len(TABLE_AWARE_PROPERTIES)

    def test_diff_reports_only_subset_when_some_already_present(self):
        # Three of the properties are already present — diff should report
        # ``len(TABLE_AWARE_PROPERTIES) - 3``.
        already = ["text", "chunk_type", "table_id", "page_no"]
        client = _make_stub_client(already)

        missing = diff_missing_table_properties(client, "Partial")

        names = [p.name for p in missing]
        assert len(names) == len(TABLE_AWARE_PROPERTIES) - 3
        for present in ("chunk_type", "table_id", "page_no"):
            assert present not in names

    def test_format_missing_diff_renders_one_line_per_property(self):
        client = _make_stub_client(["text"])
        missing = diff_missing_table_properties(client, "C")

        lines = format_missing_diff(missing)

        assert len(lines) == len(TABLE_AWARE_PROPERTIES)
        assert all(line.startswith("[MISSING] ") for line in lines)
        # Spot-check: chunk_type is TEXT + filterable
        chunk_type_line = next(line for line in lines if "chunk_type" in line)
        assert "TEXT" in chunk_type_line
        assert "filterable" in chunk_type_line


# ---------------------------------------------------------------------------
# Apply mode + idempotency
# ---------------------------------------------------------------------------

class TestApplyMigration:
    def test_apply_adds_every_missing_property(self):
        client = _make_stub_client(["text", "source"])

        result = apply_table_property_migration(client, "Legacy")

        assert isinstance(result, MigrationResult)
        assert result.ok is True
        # 15 missing → 15 added
        added_names = result.added
        assert added_names == [p.name for p in TABLE_AWARE_PROPERTIES]
        # And the client now reflects the new properties.
        post_names = {p.name for p in client._table_state}
        for prop in TABLE_AWARE_PROPERTIES:
            assert prop.name in post_names

    def test_second_apply_is_a_noop(self):
        client = _make_stub_client(["text", "source"])

        first = apply_table_property_migration(client, "Legacy")
        second = apply_table_property_migration(client, "Legacy")

        assert len(first.added) == len(TABLE_AWARE_PROPERTIES)
        assert second.added == []
        assert second.missing_before == []
        assert len(second.already_present) == len(TABLE_AWARE_PROPERTIES)
        # add_property must not be called again on the no-op run.
        col = client.collections.get.return_value
        assert col.config.add_property.call_count == len(TABLE_AWARE_PROPERTIES)

    def test_partial_collection_only_adds_missing(self):
        # 4 of the 15 properties already present.
        already = ["text", "chunk_type", "page_no", "table_id", "page_bbox"]
        client = _make_stub_client(already)

        result = apply_table_property_migration(client, "Partial")

        # The 4 pre-existing table props show up in already_present, not added.
        assert set(result.already_present) >= {
            "chunk_type",
            "page_no",
            "table_id",
            "page_bbox",
        }
        assert "chunk_type" not in result.added
        assert "page_no" not in result.added
        # Everything else is added (total minus four already-present).
        assert len(result.added) == len(TABLE_AWARE_PROPERTIES) - 4


# ---------------------------------------------------------------------------
# Error surfaces
# ---------------------------------------------------------------------------

class TestErrorSurfaces:
    def test_unknown_collection_raises_keyerror(self):
        client = _make_stub_client([], exists=False)

        with pytest.raises(KeyError, match="does not exist"):
            diff_missing_table_properties(client, "Nope")

        with pytest.raises(KeyError, match="does not exist"):
            apply_table_property_migration(client, "Nope")

    def test_add_property_failure_is_captured_not_raised(self):
        client = _make_stub_client(["text"])
        col = client.collections.get.return_value

        # Make add_property fail for one specific property.
        original = col.config.add_property.side_effect

        def _fail_on_table_caption(prop):
            if prop.name == "table_caption":
                raise RuntimeError("boom")
            original(prop)

        col.config.add_property.side_effect = _fail_on_table_caption

        result = apply_table_property_migration(client, "Flaky")

        assert result.ok is False
        assert any(name == "table_caption" for name, _ in result.errors)
        # The other props still got added — all except ``table_caption``.
        assert len(result.added) == len(TABLE_AWARE_PROPERTIES) - 1
