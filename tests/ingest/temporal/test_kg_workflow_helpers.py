# @summary
# Tests for the workflow helpers added in Steps 10/11: error-class
# extraction from a wrapped ApplicationError chain, and shape of the
# IngestDocumentResult / BackfillKG{Args,Result} dataclasses.
# Exports: (test module)
# @end-summary
"""Step 10/11: workflow helper unit tests."""

from __future__ import annotations

from temporalio.exceptions import ActivityError, ApplicationError

from src.ingest.temporal.workflows import (
    BackfillKGArgs,
    BackfillKGResult,
    IngestDocumentResult,
    _extract_error_class,
)


def _wrap(inner: BaseException) -> ActivityError:
    """Build an ActivityError with the supplied cause (mirrors Temporal runtime)."""
    return ActivityError(
        "outer",
        scheduled_event_id=1,
        started_event_id=1,
        identity="test",
        activity_type="kg_phase2b",
        activity_id="1",
        retry_state=0,
        cause=inner,
    ) if "cause" in ActivityError.__init__.__code__.co_varnames else _wrap_via_attr(inner)


def _wrap_via_attr(inner: BaseException) -> ActivityError:
    err = ActivityError(
        "outer",
        scheduled_event_id=1,
        started_event_id=1,
        identity="test",
        activity_type="kg_phase2b",
        activity_id="1",
        retry_state=0,
    )
    err.__cause__ = inner
    return err


def test_extract_error_class_finds_application_error_type() -> None:
    inner = ApplicationError("bad SV", type="document", non_retryable=True)
    wrapped = _wrap_via_attr(inner)
    assert _extract_error_class(wrapped) == "document"


def test_extract_error_class_handles_transient() -> None:
    inner = ApplicationError("rate limit", type="transient", non_retryable=False)
    wrapped = _wrap_via_attr(inner)
    assert _extract_error_class(wrapped) == "transient"


def test_extract_error_class_handles_system() -> None:
    inner = ApplicationError("missing model", type="system", non_retryable=True)
    wrapped = _wrap_via_attr(inner)
    assert _extract_error_class(wrapped) == "system"


def test_extract_error_class_defaults_to_transient_when_unknown() -> None:
    err = RuntimeError("unknown")
    assert _extract_error_class(err) == "transient"


def test_ingest_document_result_default_kg_fields() -> None:
    r = IngestDocumentResult(
        source_key="d", errors=[], stored_count=0, processing_log=[],
    )
    assert r.kg_status == "skipped"
    assert r.kg_error_class == ""
    assert r.kg_entities_added == 0
    assert r.kg_triples_added == 0


def test_backfill_args_defaults() -> None:
    a = BackfillKGArgs(clean_store_dir="/x")
    assert a.source_meta == {}
    assert a.max_per_run == 100


def test_backfill_result_construction() -> None:
    r = BackfillKGResult(discovered=5, succeeded=3, failed_pending_retry=1, failed_permanent=1)
    assert r.discovered == 5
    assert r.succeeded == 3
    assert r.skipped == 0
