"""Tests for src.ingest.temporal.activities helper functions.

Covers prewarm_worker_resources, _deserialise_config, _get_embedder, _get_db_client
without requiring a live Temporal runtime.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Lazy import helpers — avoid importing temporalio at collection time
# ---------------------------------------------------------------------------


def _import_activities():
    """Import the activities module lazily to avoid top-level Temporal load."""
    import src.ingest.temporal.activities as acts
    return acts


# ---------------------------------------------------------------------------
# Tests: prewarm_worker_resources
# ---------------------------------------------------------------------------


class TestPrewarmWorkerResources:
    def test_mock_prewarm_success(self, monkeypatch):
        """prewarm_worker_resources should set _embedder and _db_client singletons."""
        acts = _import_activities()

        fake_embedder = MagicMock()
        fake_db_client = MagicMock()

        monkeypatch.setattr("src.ingest.temporal.activities._embedder", None)
        monkeypatch.setattr("src.ingest.temporal.activities._db_client", None)
        monkeypatch.setattr("src.ingest.temporal.activities.get_embedding_provider", lambda **kw: fake_embedder)
        monkeypatch.setattr("src.ingest.temporal.activities.db.create_persistent_client", lambda: fake_db_client)
        monkeypatch.setattr("src.ingest.temporal.activities.RAG_INGESTION_DOCLING_ENABLED", False)

        acts.prewarm_worker_resources()

        assert acts._embedder is fake_embedder
        assert acts._db_client is fake_db_client

    def test_mock_prewarm_docling_disabled(self, monkeypatch, caplog):
        """prewarm should skip docling warmup when RAG_INGESTION_DOCLING_ENABLED=False."""
        acts = _import_activities()

        monkeypatch.setattr("src.ingest.temporal.activities.get_embedding_provider", lambda **kw: MagicMock())
        monkeypatch.setattr("src.ingest.temporal.activities.db.create_persistent_client", lambda **kw: MagicMock())
        monkeypatch.setattr("src.ingest.temporal.activities.RAG_INGESTION_DOCLING_ENABLED", False)

        ensure_called = []
        monkeypatch.setattr(
            "src.ingest.temporal.activities.ensure_docling_ready",
            lambda **kw: ensure_called.append(True),
        )

        with caplog.at_level(logging.INFO):
            acts.prewarm_worker_resources()

        # ensure_docling_ready should NOT have been called
        assert len(ensure_called) == 0

    def test_mock_prewarm_runtime_error(self, monkeypatch, caplog):
        """prewarm should log warning when docling setup raises RuntimeError."""
        acts = _import_activities()

        monkeypatch.setattr("src.ingest.temporal.activities.get_embedding_provider", lambda **kw: MagicMock())
        monkeypatch.setattr("src.ingest.temporal.activities.db.create_persistent_client", lambda **kw: MagicMock())
        monkeypatch.setattr("src.ingest.temporal.activities.RAG_INGESTION_DOCLING_ENABLED", True)
        monkeypatch.setattr(
            "src.ingest.temporal.activities.ensure_docling_ready",
            lambda **kw: (_ for _ in ()).throw(RuntimeError("model not found")),
        )

        with caplog.at_level(logging.WARNING):
            acts.prewarm_worker_resources()  # should NOT raise

        assert any("Docling" in r.message or "model" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Tests: _deserialise_config
# ---------------------------------------------------------------------------


class TestDeserialiseConfig:
    def test_mock_deserialise_config(self):
        """_deserialise_config should reconstruct IngestionConfig from a dict."""
        acts = _import_activities()
        from src.ingest.common import IngestionConfig
        import dataclasses

        original = IngestionConfig()
        config_dict = dataclasses.asdict(original)

        result = acts._deserialise_config(config_dict)

        assert isinstance(result, IngestionConfig)
        assert result.verbose_stage_logs == original.verbose_stage_logs

    def test_mock_deserialise_config_extra_keys_ignored(self):
        """_deserialise_config should ignore extra keys not in IngestionConfig."""
        acts = _import_activities()
        from src.ingest.common import IngestionConfig
        import dataclasses

        config_dict = dataclasses.asdict(IngestionConfig())
        config_dict["unknown_future_field"] = "value"

        result = acts._deserialise_config(config_dict)
        assert isinstance(result, IngestionConfig)


# ---------------------------------------------------------------------------
# Tests: _get_embedder and _get_db_client lazy init
# ---------------------------------------------------------------------------


class TestLazyInit:
    def test_mock_get_embedder_lazy_init(self, monkeypatch):
        """_get_embedder should initialise singleton on first call if None."""
        acts = _import_activities()

        fake_embedder = MagicMock()
        monkeypatch.setattr("src.ingest.temporal.activities._embedder", None)
        monkeypatch.setattr("src.ingest.temporal.activities.get_embedding_provider", lambda **kw: fake_embedder)

        result = acts._get_embedder()
        assert result is fake_embedder
        assert acts._embedder is fake_embedder

    def test_mock_get_embedder_returns_existing(self, monkeypatch):
        """_get_embedder should return existing singleton without re-initialising."""
        acts = _import_activities()

        existing = MagicMock()
        monkeypatch.setattr("src.ingest.temporal.activities._embedder", existing)

        called = []
        monkeypatch.setattr(
            "src.ingest.temporal.activities.get_embedding_provider",
            lambda: called.append(True) or MagicMock(),
        )

        result = acts._get_embedder()
        assert result is existing
        assert len(called) == 0  # not called again

    def test_mock_get_db_client_lazy_init(self, monkeypatch):
        """_get_db_client should initialise singleton on first call if None."""
        acts = _import_activities()

        fake_client = MagicMock()
        monkeypatch.setattr("src.ingest.temporal.activities._db_client", None)
        monkeypatch.setattr("src.ingest.temporal.activities.db.create_persistent_client", lambda: fake_client)

        result = acts._get_db_client()
        assert result is fake_client
        assert acts._db_client is fake_client

    def test_mock_get_db_client_returns_existing(self, monkeypatch):
        """_get_db_client should return existing singleton without re-creating."""
        acts = _import_activities()

        existing = MagicMock()
        monkeypatch.setattr("src.ingest.temporal.activities._db_client", existing)

        called = []
        monkeypatch.setattr(
            "src.ingest.temporal.activities.db.create_persistent_client",
            lambda: called.append(True) or MagicMock(),
        )

        result = acts._get_db_client()
        assert result is existing
        assert len(called) == 0


# ---------------------------------------------------------------------------
# Tests: prewarm_worker_resources docling-enabled path (line 77)
# ---------------------------------------------------------------------------


class TestPrewarmDoclingEnabled:
    def test_mock_prewarm_docling_enabled_calls_ensure_docling_ready(self, monkeypatch):
        """prewarm should call ensure_docling_ready when RAG_INGESTION_DOCLING_ENABLED=True."""
        acts = _import_activities()

        monkeypatch.setattr("src.ingest.temporal.activities.get_embedding_provider", lambda **kw: MagicMock())
        monkeypatch.setattr("src.ingest.temporal.activities.db.create_persistent_client", lambda **kw: MagicMock())
        monkeypatch.setattr("src.ingest.temporal.activities.RAG_INGESTION_DOCLING_ENABLED", True)
        monkeypatch.setattr("src.ingest.temporal.activities.RAG_INGESTION_DOCLING_MODEL", "some-model")
        monkeypatch.setattr("src.ingest.temporal.activities.RAG_INGESTION_DOCLING_ARTIFACTS_PATH", "/tmp")
        monkeypatch.setattr("src.ingest.temporal.activities.RAG_INGESTION_DOCLING_AUTO_DOWNLOAD", False)

        ensure_called = []
        monkeypatch.setattr(
            "src.ingest.temporal.activities.ensure_docling_ready",
            lambda **kw: ensure_called.append(kw),
        )

        acts.prewarm_worker_resources()

        assert len(ensure_called) == 1
        assert ensure_called[0]["parser_model"] == "some-model"


# ---------------------------------------------------------------------------
# Tests: document_processing_activity (lines 168-190)
# ---------------------------------------------------------------------------


class TestDocumentProcessingActivity:
    def test_mock_document_processing_activity_returns_result(self, monkeypatch, tmp_path):
        """document_processing_activity should call run_document_processing and return DocProcessingResult.

        clean_hash is now computed by document_processing_activity as sha256 of the
        clean text written to CleanDocumentStore — it is NOT taken from the
        run_document_processing result dict.
        """
        import asyncio
        import dataclasses
        import hashlib
        acts = _import_activities()
        from src.ingest.common import IngestionConfig

        config = IngestionConfig()
        config.clean_store_dir = str(tmp_path / "clean_store")
        config_dict = dataclasses.asdict(config)
        source_args = acts.SourceArgs(
            source_path="/tmp/doc.pdf",
            source_name="doc.pdf",
            source_uri="file:///tmp/doc.pdf",
            source_key="key123",
            source_id="id456",
            connector="local",
            source_version="v1",
        )
        args = acts.ActivityArgs(source=source_args, config=config_dict)

        fake_clean_text = "cleaned document content"
        fake_result = {
            "errors": [],
            "source_hash": "abc123",
            "cleaned_text": fake_clean_text,
            "processing_log": ["step1", "step2"],
        }

        monkeypatch.setattr("src.ingest.temporal.activities._embedder", MagicMock())
        monkeypatch.setattr("src.ingest.temporal.activities._db_client", MagicMock())
        monkeypatch.setattr(
            "src.ingest.temporal.activities.run_document_processing",
            lambda **kw: fake_result,
        )

        result = asyncio.run(acts.document_processing_activity(args))

        expected_clean_hash = hashlib.sha256(fake_clean_text.encode("utf-8")).hexdigest()
        assert isinstance(result, acts.DocProcessingResult)
        assert result.source_hash == "abc123"
        assert result.clean_hash == expected_clean_hash
        assert result.errors == []
        assert result.processing_log == ["step1", "step2"]

    def test_mock_document_processing_activity_errors_propagated(self, monkeypatch):
        """document_processing_activity should propagate errors from run_document_processing."""
        import asyncio
        import dataclasses
        acts = _import_activities()
        from src.ingest.common import IngestionConfig

        config_dict = dataclasses.asdict(IngestionConfig())
        source_args = acts.SourceArgs(
            source_path="/tmp/bad.pdf",
            source_name="bad.pdf",
            source_uri="file:///tmp/bad.pdf",
            source_key="k",
            source_id="i",
            connector="local",
            source_version="v1",
        )
        args = acts.ActivityArgs(source=source_args, config=config_dict)

        fake_result = {
            "errors": ["parse failed"],
            "source_hash": "",
            "clean_hash": "",
            "processing_log": [],
        }

        monkeypatch.setattr("src.ingest.temporal.activities._embedder", MagicMock())
        monkeypatch.setattr("src.ingest.temporal.activities._db_client", MagicMock())
        monkeypatch.setattr(
            "src.ingest.temporal.activities.run_document_processing",
            lambda **kw: fake_result,
        )

        result = asyncio.run(acts.document_processing_activity(args))
        assert result.errors == ["parse failed"]

    def test_mock_document_processing_activity_passes_source_fields(self, monkeypatch):
        """document_processing_activity passes all source fields to run_document_processing."""
        import asyncio
        import dataclasses
        acts = _import_activities()
        from src.ingest.common import IngestionConfig

        config_dict = dataclasses.asdict(IngestionConfig())
        source_args = acts.SourceArgs(
            source_path="/data/myfile.txt",
            source_name="myfile.txt",
            source_uri="file:///data/myfile.txt",
            source_key="mykey",
            source_id="myid",
            connector="s3",
            source_version="v2",
        )
        args = acts.ActivityArgs(source=source_args, config=config_dict)

        captured = {}
        monkeypatch.setattr("src.ingest.temporal.activities._embedder", MagicMock())
        monkeypatch.setattr("src.ingest.temporal.activities._db_client", MagicMock())

        def fake_run_doc_processing(**kw):
            captured.update(kw)
            return {"errors": [], "source_hash": "", "clean_hash": "", "processing_log": []}

        monkeypatch.setattr(
            "src.ingest.temporal.activities.run_document_processing",
            fake_run_doc_processing,
        )

        asyncio.run(acts.document_processing_activity(args))

        assert captured["source_path"] == "/data/myfile.txt"
        assert captured["source_name"] == "myfile.txt"
        assert captured["source_key"] == "mykey"
        assert captured["connector"] == "s3"


# ---------------------------------------------------------------------------
# Tests: embedding_pipeline_activity (lines 205-230)
# ---------------------------------------------------------------------------


class TestEmbeddingPipelineActivity:
    def test_mock_embedding_pipeline_activity_returns_result(self, monkeypatch, tmp_path):
        """embedding_pipeline_activity should call run_embedding_pipeline and return EmbeddingResult.

        Phase 2 reads clean text from CleanDocumentStore — the test must pre-populate
        the store and configure clean_store_dir in IngestionConfig.
        """
        import asyncio
        import dataclasses
        from contextlib import contextmanager
        acts = _import_activities()
        from src.ingest.common import IngestionConfig, CleanDocumentStore

        clean_store_path = tmp_path / "clean_store"
        source_key = "key"
        clean_text = "clean document text for embedding"
        meta = {"source_key": source_key}
        CleanDocumentStore(clean_store_path).write(source_key, clean_text, meta)

        config = IngestionConfig()
        config.clean_store_dir = str(clean_store_path)
        config_dict = dataclasses.asdict(config)
        source_args = acts.SourceArgs(
            source_path="/tmp/doc.pdf",
            source_name="doc.pdf",
            source_uri="file:///tmp/doc.pdf",
            source_key=source_key,
            source_id="id",
            connector="local",
            source_version="v1",
        )
        args = acts.ActivityArgs(source=source_args, config=config_dict)

        fake_wv_client = MagicMock()

        @contextmanager
        def fake_get_client():
            yield fake_wv_client

        fake_embedding_result = {
            "errors": [],
            "stored_count": 5,
            "metadata_summary": "A document about widgets.",
            "metadata_keywords": ["widget", "foo"],
            "processing_log": ["embedded 5 chunks"],
        }

        monkeypatch.setattr("src.ingest.temporal.activities._embedder", MagicMock())
        monkeypatch.setattr("src.ingest.temporal.activities._db_client", MagicMock())
        monkeypatch.setattr("src.ingest.temporal.activities.vector_db.get_client", fake_get_client)
        monkeypatch.setattr("src.ingest.temporal.activities.vector_db.ensure_collection", lambda *a, **kw: None)
        monkeypatch.setattr("src.ingest.temporal.activities.GLINER_ENABLED", False)
        monkeypatch.setattr(
            "src.ingest.temporal.activities.run_embedding_pipeline",
            lambda **kw: fake_embedding_result,
        )

        result = asyncio.run(acts.embedding_pipeline_activity(args))

        assert isinstance(result, acts.EmbeddingResult)
        assert result.stored_count == 5
        assert result.metadata_summary == "A document about widgets."
        assert result.metadata_keywords == ["widget", "foo"]
        assert result.errors == []

    def test_mock_embedding_pipeline_activity_errors_propagated(self, monkeypatch, tmp_path):
        """embedding_pipeline_activity propagates errors from run_embedding_pipeline."""
        import asyncio
        import dataclasses
        from contextlib import contextmanager
        acts = _import_activities()
        from src.ingest.common import IngestionConfig, CleanDocumentStore

        clean_store_path = tmp_path / "clean_store"
        source_key = "k"
        CleanDocumentStore(clean_store_path).write(source_key, "some text", {"source_key": source_key})

        config = IngestionConfig()
        config.clean_store_dir = str(clean_store_path)
        config_dict = dataclasses.asdict(config)
        source_args = acts.SourceArgs(
            source_path="/tmp/bad.pdf",
            source_name="bad.pdf",
            source_uri="file:///tmp/bad.pdf",
            source_key=source_key,
            source_id="i",
            connector="local",
            source_version="v1",
        )
        args = acts.ActivityArgs(source=source_args, config=config_dict)

        @contextmanager
        def fake_get_client():
            yield MagicMock()

        monkeypatch.setattr("src.ingest.temporal.activities._embedder", MagicMock())
        monkeypatch.setattr("src.ingest.temporal.activities._db_client", MagicMock())
        monkeypatch.setattr("src.ingest.temporal.activities.vector_db.get_client", fake_get_client)
        monkeypatch.setattr("src.ingest.temporal.activities.vector_db.ensure_collection", lambda *a, **kw: None)
        monkeypatch.setattr("src.ingest.temporal.activities.GLINER_ENABLED", False)
        monkeypatch.setattr(
            "src.ingest.temporal.activities.run_embedding_pipeline",
            lambda **kw: {
                "errors": ["weaviate timeout"],
                "stored_count": 0,
                "metadata_summary": "",
                "metadata_keywords": [],
                "processing_log": [],
            },
        )

        result = asyncio.run(acts.embedding_pipeline_activity(args))
        assert result.errors == ["weaviate timeout"]
        assert result.stored_count == 0

    def test_mock_embedding_pipeline_activity_with_kg_builder(self, monkeypatch, tmp_path):
        """embedding_pipeline_activity creates KnowledgeGraphBuilder when build_kg=True."""
        import asyncio
        import dataclasses
        from contextlib import contextmanager
        acts = _import_activities()
        from src.ingest.common import IngestionConfig, CleanDocumentStore

        clean_store_path = tmp_path / "clean_store"
        source_key = "k"
        CleanDocumentStore(clean_store_path).write(source_key, "doc text", {"source_key": source_key})

        config = IngestionConfig()
        config.build_kg = True
        config.store_documents = False
        config.clean_store_dir = str(clean_store_path)
        config_dict = dataclasses.asdict(config)

        source_args = acts.SourceArgs(
            source_path="/tmp/doc.pdf",
            source_name="doc.pdf",
            source_uri="file:///tmp/doc.pdf",
            source_key=source_key,
            source_id="i",
            connector="local",
            source_version="v1",
        )
        args = acts.ActivityArgs(source=source_args, config=config_dict)

        @contextmanager
        def fake_get_client():
            yield MagicMock()

        kg_builder_calls = []

        class FakeKGBuilder:
            def __init__(self, **kw):
                kg_builder_calls.append(kw)

        monkeypatch.setattr("src.ingest.temporal.activities._embedder", MagicMock())
        monkeypatch.setattr("src.ingest.temporal.activities._db_client", MagicMock())
        monkeypatch.setattr("src.ingest.temporal.activities.vector_db.get_client", fake_get_client)
        monkeypatch.setattr("src.ingest.temporal.activities.vector_db.ensure_collection", lambda *a, **kw: None)
        monkeypatch.setattr("src.ingest.temporal.activities.GLINER_ENABLED", False)
        monkeypatch.setattr("src.ingest.temporal.activities.KnowledgeGraphBuilder", FakeKGBuilder)
        monkeypatch.setattr(
            "src.ingest.temporal.activities.run_embedding_pipeline",
            lambda **kw: {
                "errors": [], "stored_count": 0, "metadata_summary": "",
                "metadata_keywords": [], "processing_log": [],
            },
        )

        asyncio.run(acts.embedding_pipeline_activity(args))
        assert len(kg_builder_calls) == 1
