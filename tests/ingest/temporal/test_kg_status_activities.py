# @summary
# Unit tests for the RagWeave-side ledger activities used by the KG phase
# 2b integration: record_phase_status_activity (writes outcome to the
# CleanDocumentStore status ledger) and list_pending_phase2b_activity
# (drives the backfill workflow).
# Exports: (test module)
# Deps: pytest, pathlib, src.ingest.common.clean_store,
#       src.ingest.temporal.activities
# @end-summary
"""Step 10/11: RagWeave-side ledger activity unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingest.common.clean_store import CleanDocumentStore
from src.ingest.temporal.activities import (
    ListPendingPhase2bArgs,
    RecordPhaseStatusArgs,
    list_pending_phase2b_activity,
    record_phase_status_activity,
)


@pytest.fixture
def store_dir(tmp_path: Path) -> str:
    s = CleanDocumentStore(tmp_path / "clean")
    s.write("doc1.md", "text-1", {"source_name": "doc1"})
    s.write("doc2.md", "text-2", {"source_name": "doc2"})
    return str(tmp_path / "clean")


# ---------------------------------------------------------------------------
# record_phase_status_activity
# ---------------------------------------------------------------------------


async def test_record_success_marks_ok(store_dir: str) -> None:
    res = await record_phase_status_activity(
        RecordPhaseStatusArgs(
            clean_store_dir=store_dir,
            source_key="doc1.md",
            phase="phase2b",
            success=True,
        )
    )
    assert res["status"] == "ok"
    assert res["attempts"] == 1


async def test_record_transient_failure_marks_pending(store_dir: str) -> None:
    res = await record_phase_status_activity(
        RecordPhaseStatusArgs(
            clean_store_dir=store_dir,
            source_key="doc1.md",
            phase="phase2b",
            success=False,
            error="rate limit",
            error_class="transient",
            max_attempts=5,
        )
    )
    assert res["status"] == "failed_pending_retry"
    assert res["attempts"] == 1
    assert res["error_class"] == "transient"


async def test_record_document_failure_marks_permanent(store_dir: str) -> None:
    res = await record_phase_status_activity(
        RecordPhaseStatusArgs(
            clean_store_dir=store_dir,
            source_key="doc1.md",
            phase="phase2b",
            success=False,
            error="bad SV",
            error_class="document",
        )
    )
    assert res["status"] == "failed_permanent"


# ---------------------------------------------------------------------------
# list_pending_phase2b_activity
# ---------------------------------------------------------------------------


async def test_list_pending_finds_failed_pending_retry(store_dir: str) -> None:
    # Mark doc1 as pending retry, doc2 as ok.
    await record_phase_status_activity(
        RecordPhaseStatusArgs(
            clean_store_dir=store_dir,
            source_key="doc1.md",
            phase="phase2b",
            success=False,
            error="x",
            error_class="transient",
        )
    )
    await record_phase_status_activity(
        RecordPhaseStatusArgs(
            clean_store_dir=store_dir,
            source_key="doc2.md",
            phase="phase2b",
            success=True,
        )
    )
    keys = await list_pending_phase2b_activity(
        ListPendingPhase2bArgs(clean_store_dir=store_dir)
    )
    assert keys == ["doc1.md"]


async def test_list_pending_empty_returns_empty(tmp_path: Path) -> None:
    keys = await list_pending_phase2b_activity(
        ListPendingPhase2bArgs(clean_store_dir=str(tmp_path / "missing"))
    )
    assert keys == []


async def test_list_pending_excludes_permanent_failures(store_dir: str) -> None:
    await record_phase_status_activity(
        RecordPhaseStatusArgs(
            clean_store_dir=store_dir,
            source_key="doc1.md",
            phase="phase2b",
            success=False,
            error="bad doc",
            error_class="document",
        )
    )
    keys = await list_pending_phase2b_activity(
        ListPendingPhase2bArgs(clean_store_dir=store_dir)
    )
    assert keys == []
