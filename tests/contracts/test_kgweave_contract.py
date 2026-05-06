# @summary
# Tests the KGWeave contract package — Pydantic v2 schemas and activity-name
# / task-queue constants that form the wire between RagWeave (caller) and
# KGWeave (Temporal worker). Both repos must pin the same contract version.
# Exports: (test module)
# Deps: pytest, pydantic, src.contracts.kgweave
# @end-summary
"""Step 8: KGWeave contract surface tests."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from kgweave.contracts import (
    CONTRACT_VERSION,
    KG_DELETE_BY_SOURCE_ACTIVITY,
    KG_HEALTH_ACTIVITY,
    KG_PHASE2B_ACTIVITY,
    KG_TASK_QUEUE,
    KGAdminRequest,
    KGAdminResult,
    KGIngestRequest,
    KGIngestResult,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_contract_version_is_v1() -> None:
    assert CONTRACT_VERSION == "v1"


def test_task_queue_constant() -> None:
    assert KG_TASK_QUEUE == "kgweave-default"


def test_activity_name_constants_are_distinct() -> None:
    names = {KG_PHASE2B_ACTIVITY, KG_DELETE_BY_SOURCE_ACTIVITY, KG_HEALTH_ACTIVITY}
    assert len(names) == 3
    for name in names:
        assert isinstance(name, str) and name


# ---------------------------------------------------------------------------
# KGIngestRequest
# ---------------------------------------------------------------------------


def test_ingest_request_minimum_valid() -> None:
    req = KGIngestRequest(source_key="doc1.md", clean_text="text")
    assert req.source_key == "doc1.md"
    assert req.clean_text == "text"
    assert req.meta == {}
    assert req.trace_id is None


def test_ingest_request_rejects_empty_source_key() -> None:
    with pytest.raises(ValidationError):
        KGIngestRequest(source_key="", clean_text="text")


def test_ingest_request_serializes_round_trip() -> None:
    req = KGIngestRequest(
        source_key="doc1.md",
        clean_text="hello",
        meta={"source_name": "doc1"},
        trace_id="t-1",
    )
    payload = req.model_dump_json()
    restored = KGIngestRequest.model_validate_json(payload)
    assert restored == req


# ---------------------------------------------------------------------------
# KGIngestResult
# ---------------------------------------------------------------------------


def test_ingest_result_defaults() -> None:
    res = KGIngestResult(source_key="doc1.md")
    assert res.source_key == "doc1.md"
    assert res.entities_added == 0
    assert res.triples_added == 0
    assert res.chunks_processed == 0
    assert res.elapsed_ms == 0.0


def test_ingest_result_round_trip() -> None:
    res = KGIngestResult(
        source_key="doc1.md",
        entities_added=10,
        triples_added=5,
        chunks_processed=1,
        elapsed_ms=42.5,
    )
    restored = KGIngestResult.model_validate(json.loads(res.model_dump_json()))
    assert restored == res


def test_ingest_result_rejects_negative_counts() -> None:
    with pytest.raises(ValidationError):
        KGIngestResult(source_key="doc1.md", entities_added=-1)


# ---------------------------------------------------------------------------
# KGAdminRequest / KGAdminResult
# ---------------------------------------------------------------------------


def test_admin_request_delete_requires_source_key() -> None:
    req = KGAdminRequest(op="delete_by_source", source_key="doc1.md")
    assert req.op == "delete_by_source"
    assert req.source_key == "doc1.md"


def test_admin_request_health_no_source_key() -> None:
    req = KGAdminRequest(op="health")
    assert req.op == "health"
    assert req.source_key is None


def test_admin_request_unknown_op_rejected() -> None:
    with pytest.raises(ValidationError):
        KGAdminRequest(op="drop_database")


def test_admin_request_delete_without_source_key_rejected() -> None:
    with pytest.raises(ValidationError):
        KGAdminRequest(op="delete_by_source")


def test_admin_result_round_trip() -> None:
    r = KGAdminResult(
        ok=True,
        stats={"nodes": 10, "edges": 5, "backend": "networkx"},
    )
    restored = KGAdminResult.model_validate_json(r.model_dump_json())
    assert restored == r
