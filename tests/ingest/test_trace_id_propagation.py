"""Trace ID propagation tests.

Contract:
- Temporal activities unify ``trace_id`` with ``staging_batch_id`` so a single
  per-document ID flows through Phase 1 + Phase 2 and is attached to every
  Weaviate chunk and MinIO object metadata.
- ``run_document_processing`` and ``run_embedding_pipeline`` both accept and
  expose ``trace_id`` in their state.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_run_document_processing_accepts_trace_id():
    from src.ingest.doc_processing.impl import run_document_processing

    sig = inspect.signature(run_document_processing)
    assert "trace_id" in sig.parameters


def test_run_embedding_pipeline_accepts_trace_id():
    from src.ingest.embedding.impl import run_embedding_pipeline

    sig = inspect.signature(run_embedding_pipeline)
    assert "trace_id" in sig.parameters


def test_temporal_phase1_activity_passes_staging_batch_id_as_trace_id(tmp_path):
    """document_processing_activity must forward staging_batch_id as trace_id."""
    from src.ingest.temporal.activities import (
        ActivityArgs,
        SourceArgs,
        document_processing_activity,
    )

    doc = tmp_path / "doc.txt"
    doc.write_text("Hello.")
    store_dir = tmp_path / "clean"

    args = ActivityArgs(
        config={"clean_store_dir": str(store_dir)},
        source=SourceArgs(
            source_path=str(doc),
            source_name="doc.txt",
            source_uri=doc.as_uri(),
            source_key="local_fs:test:42",
            source_id="test:42",
            connector="local_fs",
            source_version="1",
        ),
        staging_batch_id="batch-uuid-xyz",
    )

    captured: dict = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        return {
            "errors": [],
            "source_hash": hashlib.sha256(b"Hello.").hexdigest(),
            "cleaned_text": "Hello.",
            "processing_log": [],
            "trace_id": kwargs.get("trace_id", ""),
        }

    with patch("src.ingest.temporal.activities.run_document_processing", side_effect=_fake_run), \
         patch("src.ingest.temporal.activities.vector_db") as mock_vdb:
        mock_vdb.get_client.return_value.__enter__.return_value = MagicMock()
        mock_vdb.get_client.return_value.__exit__.return_value = False
        # Avoid idempotency short-circuit
        with patch("src.ingest.temporal.activities.get_source_hash", return_value=None):
            asyncio.get_event_loop().run_until_complete(
                document_processing_activity(args)
            ) if False else None
            # Pytest event loop policy: run via asyncio.run
            import asyncio as _a
            _a.run(document_processing_activity(args))

    assert captured.get("trace_id") == "batch-uuid-xyz"


def test_temporal_phase2_activity_passes_staging_batch_id_as_trace_id(tmp_path):
    """embedding_pipeline_activity must forward staging_batch_id as trace_id."""
    import asyncio as _a

    from src.ingest.common.clean_store import CleanDocumentStore
    from src.ingest.temporal.activities import (
        ActivityArgs,
        SourceArgs,
        embedding_pipeline_activity,
    )

    store_dir = tmp_path / "clean"
    store = CleanDocumentStore(store_dir)
    store.write(
        "local_fs:test:99",
        "clean text",
        {
            "source_key": "local_fs:test:99",
            "source_name": "doc.txt",
            "source_uri": "file:///tmp/doc.txt",
            "source_id": "test:99",
            "connector": "local_fs",
            "source_version": "1",
            "source_hash": "abc",
            "clean_hash": hashlib.sha256(b"clean text").hexdigest(),
        },
    )

    args = ActivityArgs(
        config={"clean_store_dir": str(store_dir)},
        source=SourceArgs(
            source_path="/tmp/doc.txt",
            source_name="doc.txt",
            source_uri="file:///tmp/doc.txt",
            source_key="local_fs:test:99",
            source_id="test:99",
            connector="local_fs",
            source_version="1",
        ),
        staging_batch_id="batch-uuid-zzz",
    )

    captured: dict = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        return {
            "errors": [],
            "stored_count": 1,
            "metadata_summary": "",
            "metadata_keywords": [],
            "processing_log": [],
        }

    with patch("src.ingest.temporal.activities.run_embedding_pipeline", side_effect=_fake_run), \
         patch("src.ingest.temporal.activities.vector_db") as mock_vdb, \
         patch("src.ingest.temporal.activities._get_embedder", return_value=MagicMock()):
        mock_vdb.get_client.return_value.__enter__.return_value = MagicMock()
        mock_vdb.get_client.return_value.__exit__.return_value = False
        _a.run(embedding_pipeline_activity(args))

    assert captured.get("trace_id") == "batch-uuid-zzz"
    assert captured.get("staging_batch_id") == "batch-uuid-zzz"
