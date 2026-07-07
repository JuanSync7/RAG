# @summary
# Tests for P3 pack ingest runner (execute_plan + IngestReport).
# Four FAST tests assert call wiring + report shape via monkeypatch;
# one SLOW+INTEGRATION test exercises real Weaviate when reachable.
# Exports: test_execute_plan_wires_target_collection_via_config,
#          test_execute_plan_returns_ingest_report,
#          test_execute_plan_propagates_fresh_flag,
#          test_ingest_report_is_frozen,
#          test_execute_plan_opentitan_live
# Deps: src.eval.runner, src.eval.pack, src.ingest.common.types
# @end-summary
"""P3 — pack ingest runner tests."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACK_DIR = REPO_ROOT / "evals" / "packs" / "opentitan_riscv"


def _load_plan():
    from src.eval.pack import load_pack
    from src.eval.runner import plan_pack_ingest

    pack = load_pack(PACK_DIR)
    return pack, plan_pack_ingest(pack, PACK_DIR)


def _patch_execute(monkeypatch, *, summary, stats):
    """Patch ingest_directory + collection stats helpers in execute namespace.

    Returns the captured-kwargs dict for assertion.
    """
    from src.eval.runner import execute as execute_mod

    captured: dict = {}

    def fake_ingest(**kwargs):
        captured.update(kwargs)
        return summary

    def fake_stats(_client, *, collection):
        captured["_stats_collection"] = collection
        return stats

    def fake_open():
        return object()

    def fake_close(_client):
        return None

    monkeypatch.setattr(execute_mod, "ingest_directory", fake_ingest)
    monkeypatch.setattr(execute_mod, "get_collection_stats", fake_stats)
    monkeypatch.setattr(execute_mod, "create_persistent_client", fake_open)
    monkeypatch.setattr(execute_mod, "close_client", fake_close)
    return captured


def _synthetic_summary(**overrides):
    from src.ingest.common.types import IngestionRunSummary

    defaults = dict(
        processed=5,
        skipped=0,
        failed=0,
        stored_chunks=42,
        removed_sources=0,
        errors=[],
    )
    defaults.update(overrides)
    return IngestionRunSummary(**defaults)


def test_execute_plan_wires_target_collection_via_config(monkeypatch) -> None:
    """execute_plan must call ingest_directory with exact kwarg shape.

    Asserts documents_dir, IngestionConfig.target_collection, selected_sources
    (order-preserved list), and the deterministic fresh=True/update=False default.
    """
    from src.eval.runner import execute_plan

    _pack, plan = _load_plan()
    summary = _synthetic_summary()
    stats = {"chunk_count": 42, "document_count": 5}
    captured = _patch_execute(monkeypatch, summary=summary, stats=stats)

    report = execute_plan(plan)

    assert captured["documents_dir"] == plan.documents_dir
    cfg = captured["config"]
    assert cfg.target_collection == plan.collection_name
    assert captured["selected_sources"] == list(plan.doc_paths)
    assert captured["fresh"] is True
    assert captured["update"] is False
    # Defensive: ensure stats lookup used the same collection name.
    assert captured["_stats_collection"] == plan.collection_name
    assert report.collection_name == plan.collection_name


def test_execute_plan_returns_ingest_report(monkeypatch) -> None:
    """execute_plan returns an IngestReport whose fields mirror the run summary
    and post-ingest collection stats, with the plan preserved as a back-ref."""
    from src.eval.runner import execute_plan

    _pack, plan = _load_plan()
    summary = _synthetic_summary(processed=5, stored_chunks=42)
    stats = {"chunk_count": 42, "document_count": 5}
    _patch_execute(monkeypatch, summary=summary, stats=stats)

    report = execute_plan(plan)

    assert report.collection_name == plan.collection_name
    assert report.processed == 5
    assert report.stored_chunks == 42
    assert report.document_count == 5
    assert report.failed == 0
    assert report.plan is plan


def test_execute_plan_propagates_fresh_flag(monkeypatch) -> None:
    """fresh=False inverts the toggle pair: ingest_directory is called with
    fresh=False and update=True."""
    from src.eval.runner import execute_plan

    _pack, plan = _load_plan()
    summary = _synthetic_summary()
    stats = {"chunk_count": 42, "document_count": 5}
    captured = _patch_execute(monkeypatch, summary=summary, stats=stats)

    execute_plan(plan, fresh=False)

    assert captured["fresh"] is False
    assert captured["update"] is True


def test_ingest_report_is_frozen() -> None:
    """IngestReport is an immutable (frozen) dataclass."""
    from src.eval.runner import IngestReport

    _pack, plan = _load_plan()
    report = IngestReport(
        collection_name=plan.collection_name,
        processed=0,
        skipped=0,
        failed=0,
        stored_chunks=0,
        document_count=0,
        plan=plan,
        errors=(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.processed = 99  # type: ignore[misc]


@pytest.mark.slow
@pytest.mark.integration
def test_execute_plan_opentitan_live(tmp_path: Path) -> None:
    """Live: execute the OpenTitan plan against real Weaviate.

    Skips on infra absence. On success: 5 distinct docs ingested with
    non-zero stored chunks and zero failures. Best-effort cleanup.
    """
    pytest.importorskip("weaviate")
    import socket

    from config.settings import MINIO_ENDPOINT
    from src.eval.runner import execute_plan
    from src.vector_db import close_client, create_persistent_client

    try:
        client = create_persistent_client()
    except Exception as exc:  # pragma: no cover - infra-dependent
        pytest.skip(f"Live Weaviate not reachable: {exc}")

    try:
        host, _, port = MINIO_ENDPOINT.partition(":")
        with socket.create_connection(
            (host or "localhost", int(port or "9000")), timeout=2
        ):
            pass
    except Exception as exc:  # pragma: no cover - infra-dependent
        try:
            close_client(client)
        except Exception:
            pass
        pytest.skip(f"Live MinIO ({MINIO_ENDPOINT}) not reachable: {exc}")

    _pack, plan = _load_plan()

    report = None
    try:
        report = execute_plan(plan)
        assert report.failed == 0, f"unexpected failures: {report.errors}"
        assert report.document_count == 5, (
            f"expected 5 docs, got {report.document_count}"
        )
        assert report.stored_chunks > 0, "expected non-zero stored chunks"
    finally:
        try:
            target = report.collection_name if report else plan.collection_name
            if client.collections.exists(target):
                client.collections.delete(target)
        except Exception:
            pass
        try:
            close_client(client)
        except Exception:
            pass
