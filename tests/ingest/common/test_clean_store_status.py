# @summary
# Tests the per-phase status ledger on CleanDocumentStore. The ledger lets
# Phase 2a/2b record success/failure durably so a backfill workflow can
# rediscover documents that need retry without depending on workflow lifetime.
# Exports: (test module)
# Deps: pytest, pathlib, src.ingest.common.clean_store
# @end-summary
"""Step 7: CleanDocumentStore status ledger tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingest.common.clean_store import CleanDocumentStore


@pytest.fixture
def store(tmp_path: Path) -> CleanDocumentStore:
    s = CleanDocumentStore(tmp_path / "clean")
    s.write("doc1.md", "text-1", {"source_name": "doc1"})
    s.write("doc2.md", "text-2", {"source_name": "doc2"})
    return s


# ---------------------------------------------------------------------------
# get_status: default + after writes
# ---------------------------------------------------------------------------


def test_get_status_returns_pending_when_never_set(
    store: CleanDocumentStore,
) -> None:
    status = store.get_status("doc1.md")
    assert status["phase2a"]["status"] == "pending"
    assert status["phase2b"]["status"] == "pending"
    assert status["phase2a"]["attempts"] == 0
    assert status["phase2b"]["attempts"] == 0


def test_get_status_missing_source_raises(store: CleanDocumentStore) -> None:
    with pytest.raises(FileNotFoundError):
        store.get_status("missing.md")


# ---------------------------------------------------------------------------
# record_attempt: increments + outcome
# ---------------------------------------------------------------------------


def test_record_attempt_success_marks_ok(store: CleanDocumentStore) -> None:
    store.record_attempt("doc1.md", phase="phase2a", success=True)
    s = store.get_status("doc1.md")
    assert s["phase2a"]["status"] == "ok"
    assert s["phase2a"]["attempts"] == 1
    assert s["phase2a"]["last_error"] is None


def test_record_attempt_transient_failure_marks_pending_retry(
    store: CleanDocumentStore,
) -> None:
    store.record_attempt(
        "doc1.md",
        phase="phase2b",
        success=False,
        error="RateLimitError: ...",
        error_class="transient",
    )
    s = store.get_status("doc1.md")
    assert s["phase2b"]["status"] == "failed_pending_retry"
    assert s["phase2b"]["attempts"] == 1
    assert s["phase2b"]["error_class"] == "transient"
    assert "RateLimitError" in s["phase2b"]["last_error"]


def test_record_attempt_document_failure_marks_permanent(
    store: CleanDocumentStore,
) -> None:
    store.record_attempt(
        "doc1.md",
        phase="phase2b",
        success=False,
        error="ExtractorError: malformed SV",
        error_class="document",
    )
    s = store.get_status("doc1.md")
    assert s["phase2b"]["status"] == "failed_permanent"
    assert s["phase2b"]["error_class"] == "document"


def test_record_attempt_system_failure_marks_permanent(
    store: CleanDocumentStore,
) -> None:
    store.record_attempt(
        "doc1.md",
        phase="phase2b",
        success=False,
        error="ImportError: kgweave model missing",
        error_class="system",
    )
    s = store.get_status("doc1.md")
    assert s["phase2b"]["status"] == "failed_permanent"


def test_record_attempt_exhausts_retries_at_max_attempts(
    store: CleanDocumentStore,
) -> None:
    for _ in range(3):
        store.record_attempt(
            "doc1.md",
            phase="phase2b",
            success=False,
            error="boom",
            error_class="transient",
            max_attempts=3,
        )
    s = store.get_status("doc1.md")
    assert s["phase2b"]["attempts"] == 3
    # Still transient, but exhausted → permanent so backfill stops retrying.
    assert s["phase2b"]["status"] == "failed_permanent"


def test_record_attempt_resets_attempts_on_success(
    store: CleanDocumentStore,
) -> None:
    store.record_attempt(
        "doc1.md", phase="phase2b", success=False,
        error="x", error_class="transient",
    )
    store.record_attempt("doc1.md", phase="phase2b", success=True)
    s = store.get_status("doc1.md")
    assert s["phase2b"]["status"] == "ok"
    assert s["phase2b"]["last_error"] is None


# ---------------------------------------------------------------------------
# list_pending: backfill discovery
# ---------------------------------------------------------------------------


def test_list_pending_finds_failed_pending_retry(
    store: CleanDocumentStore,
) -> None:
    store.record_attempt(
        "doc1.md", phase="phase2b", success=False,
        error="x", error_class="transient",
    )
    store.record_attempt("doc2.md", phase="phase2b", success=True)
    pending = store.list_pending(phase="phase2b", status="failed_pending_retry")
    assert pending == ["doc1.md"]


def test_list_pending_excludes_permanent_failures(
    store: CleanDocumentStore,
) -> None:
    store.record_attempt(
        "doc1.md", phase="phase2b", success=False,
        error="bad doc", error_class="document",
    )
    pending = store.list_pending(phase="phase2b", status="failed_pending_retry")
    assert pending == []


def test_list_pending_empty_store_returns_empty(tmp_path: Path) -> None:
    s = CleanDocumentStore(tmp_path / "empty")
    assert s.list_pending(phase="phase2b", status="failed_pending_retry") == []


# ---------------------------------------------------------------------------
# Persistence: status survives reopening the store
# ---------------------------------------------------------------------------


def test_status_persists_across_store_instances(tmp_path: Path) -> None:
    s1 = CleanDocumentStore(tmp_path / "p")
    s1.write("doc.md", "t", {})
    s1.record_attempt(
        "doc.md", phase="phase2b", success=False,
        error="x", error_class="transient",
    )
    s2 = CleanDocumentStore(tmp_path / "p")
    status = s2.get_status("doc.md")
    assert status["phase2b"]["status"] == "failed_pending_retry"
    assert status["phase2b"]["attempts"] == 1


# ---------------------------------------------------------------------------
# delete: must remove status sidecar
# ---------------------------------------------------------------------------


def test_delete_removes_status_sidecar(store: CleanDocumentStore) -> None:
    store.record_attempt("doc1.md", phase="phase2a", success=True)
    store.delete("doc1.md")
    # After delete, the entry should not exist.
    assert not store.exists("doc1.md")
    with pytest.raises(FileNotFoundError):
        store.get_status("doc1.md")
