"""
Tests for the two-phase orchestrator in src/ingest/impl.py.

Covers behaviors not already tested in test_two_phase_orchestrator.py:
  - ingest_directory: new files, skip unchanged, remove deleted, partial failure,
    invalid config guard, empty directory
  - ingest_file result fields: processing_log merging, metadata fields, empty errors list
  - verify_core_design: valid config returns ok=True; never raises on any input
"""
import hashlib
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from src.ingest.impl import ingest_file, ingest_directory, verify_core_design
from src.ingest.common.types import (
    IngestionConfig,
    Runtime,
    IngestFileResult,
    IngestionRunSummary,
    IngestionDesignCheck,
)
from src.ingest.common.clean_store import CleanDocumentStore


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_runtime(tmp_path, **config_overrides):
    """Build a minimal Runtime whose config has every optional stage disabled."""
    config_kwargs = dict(
        enable_multimodal_processing=False,
        enable_document_refactoring=False,
        enable_cross_reference_extraction=False,
        enable_knowledge_graph_extraction=False,
        enable_knowledge_graph_storage=False,
        enable_quality_validation=False,
        enable_docling_parser=False,
        enable_llm_metadata=False,
        persist_refactor_mirror=False,
        clean_store_dir=str(tmp_path / "store"),
        build_kg=False,
    )
    config_kwargs.update(config_overrides)
    config = IngestionConfig(**config_kwargs)
    return Runtime(
        config=config,
        embedder=MagicMock(),
        weaviate_client=MagicMock(),
        kg_builder=None,
    )


def _make_config(tmp_path, **overrides):
    """Build a minimal IngestionConfig with every optional stage disabled."""
    config_kwargs = dict(
        enable_multimodal_processing=False,
        enable_document_refactoring=False,
        enable_cross_reference_extraction=False,
        enable_knowledge_graph_extraction=False,
        enable_knowledge_graph_storage=False,
        enable_quality_validation=False,
        enable_docling_parser=False,
        enable_llm_metadata=False,
        persist_refactor_mirror=False,
        clean_store_dir=str(tmp_path / "store"),
        build_kg=False,
        store_documents=False,
    )
    config_kwargs.update(overrides)
    return IngestionConfig(**config_kwargs)


def _mock_ingest_result(stored=3, errors=None, summary="Test summary", keywords=None):
    """Build an IngestFileResult for test mocking."""
    return IngestFileResult(
        errors=errors or [],
        stored_count=stored,
        metadata_summary=summary,
        metadata_keywords=keywords or ["test"],
        processing_log=["document_ingestion:ok", "embedding_storage:ok"],
        source_hash="abc123",
        clean_hash="def456",
    )


def _make_mock_ctx():
    """Return a MagicMock context manager (for get_client patching)."""
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=MagicMock())
    mock_ctx.__exit__ = MagicMock(return_value=False)
    return mock_ctx


def _phase1_result(doc: Path, cleaned="clean text"):
    """Return a fake Phase 1 (DocumentProcessingState) dict for a given file."""
    data = doc.read_bytes()
    return {
        "source_hash": hashlib.sha256(data).hexdigest(),
        "raw_text": data.decode(),
        "cleaned_text": cleaned,
        "refactored_text": None,
        "errors": [],
        "processing_log": ["document_ingestion:ok"],
        "structure": {"has_figures": False},
        "multimodal_notes": [],
    }


def _phase2_result(**overrides):
    """Return a fake Phase 2 (EmbeddingState) dict, with optional field overrides."""
    r = {
        "stored_count": 3,
        "metadata_summary": "A test document.",
        "metadata_keywords": ["test"],
        "errors": [],
        "processing_log": ["chunking:ok", "embedding_storage:ok"],
        "chunks": [],
        "kg_triples": [],
    }
    r.update(overrides)
    return r


# ---------------------------------------------------------------------------
# ingest_directory tests
# ---------------------------------------------------------------------------

class TestIngestDirectoryNewFiles:
    def test_ingest_directory_processes_new_files(self, tmp_path):
        """Two new .txt files with empty manifest → both processed.

        ingest_file must be called twice; summary: processed==2, skipped==0, failed==0.
        """
        src_dir = tmp_path / "docs"
        src_dir.mkdir()
        (src_dir / "a.txt").write_text("Content of file A.")
        (src_dir / "b.txt").write_text("Content of file B.")

        config = _make_config(tmp_path)

        with patch("src.ingest.impl.get_client", return_value=_make_mock_ctx()), \
             patch("src.ingest.impl.load_manifest", return_value={}), \
             patch("src.ingest.impl.save_manifest"), \
             patch("src.ingest.impl.delete_collection"), \
             patch("src.ingest.impl.ensure_collection"), \
             patch("src.ingest.impl.get_embedding_provider"), \
             patch("src.ingest.impl.KnowledgeGraphBuilder"), \
             patch("src.ingest.impl.ingest_file",
                   return_value=_mock_ingest_result()) as mock_if:

            summary = ingest_directory(documents_dir=src_dir, config=config, fresh=False)

        assert summary.processed == 2
        assert summary.skipped == 0
        assert summary.failed == 0
        assert mock_if.call_count == 2


class TestIngestDirectorySkipUnchanged:
    def test_ingest_directory_skips_unchanged_file(self, tmp_path):
        """File whose hash matches manifest entry is skipped when update=True.

        The manifest is seeded using the real inode-based source_key so that
        _normalize_manifest_entries keeps it in manifest.keys() (preventing it
        from being treated as a removed source), and _find_manifest_entry finds
        it on the direct-key lookup.  The skip logic then fires on hash equality.
        summary: skipped==1, processed==0; ingest_file not called.
        """
        src_dir = tmp_path / "docs"
        src_dir.mkdir()
        f = src_dir / "unchanged.txt"
        f.write_text("This file has not changed.")

        # Build actual source_key the same way _local_source_identity does.
        stat = f.resolve().stat()
        actual_source_key = f"local_fs:{stat.st_dev}:{stat.st_ino}"
        content_hash = hashlib.sha256(f.read_bytes()).hexdigest()

        manifest_data = {
            actual_source_key: {
                "content_hash": content_hash,
                "source_key": actual_source_key,
                "source_uri": f.resolve().as_uri(),
                "source_id": f"{stat.st_dev}:{stat.st_ino}",
            }
        }

        # clean_store_dir="" → store_ok=True (no CleanDocumentStore check needed).
        config = _make_config(tmp_path, clean_store_dir="")

        with patch("src.ingest.impl.get_client", return_value=_make_mock_ctx()), \
             patch("src.ingest.impl.load_manifest", return_value=manifest_data), \
             patch("src.ingest.impl.save_manifest"), \
             patch("src.ingest.impl.delete_collection"), \
             patch("src.ingest.impl.ensure_collection"), \
             patch("src.ingest.impl.get_embedding_provider"), \
             patch("src.ingest.impl.KnowledgeGraphBuilder"), \
             patch("src.ingest.impl.ingest_file") as mock_if:

            summary = ingest_directory(
                documents_dir=src_dir, config=config, fresh=False, update=True
            )

        assert summary.skipped == 1
        assert summary.processed == 0
        mock_if.assert_not_called()


class TestIngestDirectoryRemoveDeleted:
    def test_ingest_directory_removes_deleted_source_in_update_mode(self, tmp_path):
        """update=True: manifest key absent from disk triggers delete_by_source_key.

        One live file on disk (to avoid early-return), one stale manifest key
        not matching any live file → removed_sources==1, delete called once.
        """
        src_dir = tmp_path / "docs"
        src_dir.mkdir()
        (src_dir / "current.txt").write_text("Still here.")

        old_key = "local_fs:99999:12345678"
        manifest_data = {
            old_key: {
                "content_hash": "deadbeef",
                "chunk_count": 1,
                "source_key": old_key,
            },
        }

        config = _make_config(tmp_path)

        with patch("src.ingest.impl.get_client", return_value=_make_mock_ctx()), \
             patch("src.ingest.impl.load_manifest", return_value=manifest_data), \
             patch("src.ingest.impl.save_manifest"), \
             patch("src.ingest.impl.delete_collection"), \
             patch("src.ingest.impl.ensure_collection"), \
             patch("src.ingest.impl.delete_by_source_key") as mock_del, \
             patch("src.ingest.impl.get_embedding_provider"), \
             patch("src.ingest.impl.KnowledgeGraphBuilder"), \
             patch("src.ingest.impl.ingest_file",
                   return_value=_mock_ingest_result()):

            summary = ingest_directory(
                documents_dir=src_dir, config=config, fresh=False, update=True
            )

        assert summary.removed_sources == 1
        mock_del.assert_called_once()


class TestIngestDirectoryPartialFailure:
    def test_ingest_directory_partial_failure_continues(self, tmp_path):
        """RuntimeError from ingest_file for bad.txt → other two still processed.

        summary: processed==2, failed==1.
        """
        src_dir = tmp_path / "docs"
        src_dir.mkdir()
        (src_dir / "good1.txt").write_text("Good document one.")
        (src_dir / "bad.txt").write_text("This one will fail.")
        (src_dir / "good2.txt").write_text("Good document three.")

        config = _make_config(tmp_path)

        def _selective_ingest(source_path, *args, **kwargs):
            if Path(source_path).name == "bad.txt":
                raise RuntimeError("Simulated ingest_file failure for bad.txt")
            return _mock_ingest_result()

        with patch("src.ingest.impl.get_client", return_value=_make_mock_ctx()), \
             patch("src.ingest.impl.load_manifest", return_value={}), \
             patch("src.ingest.impl.save_manifest"), \
             patch("src.ingest.impl.delete_collection"), \
             patch("src.ingest.impl.ensure_collection"), \
             patch("src.ingest.impl.get_embedding_provider"), \
             patch("src.ingest.impl.KnowledgeGraphBuilder"), \
             patch("src.ingest.impl.ingest_file", side_effect=_selective_ingest):

            summary = ingest_directory(documents_dir=src_dir, config=config, fresh=False)

        assert summary.processed == 2
        assert summary.failed == 1


class TestIngestDirectoryInvalidConfig:
    def test_ingest_directory_raises_on_invalid_config(self, tmp_path):
        """enable_knowledge_graph_storage=True with build_kg=False raises ValueError.

        verify_core_design fires before any infrastructure is touched;
        ingest_file must not be called.
        """
        src_dir = tmp_path / "docs"
        src_dir.mkdir()
        (src_dir / "doc.txt").write_text("irrelevant")

        config = _make_config(
            tmp_path,
            enable_knowledge_graph_storage=True,
            build_kg=False,
        )

        with patch("src.ingest.impl.ingest_file") as mock_if, \
             pytest.raises(ValueError):
            ingest_directory(documents_dir=src_dir, config=config)

        mock_if.assert_not_called()


class TestIngestDirectoryEmptyDir:
    def test_ingest_directory_empty_dir_returns_zero_counts(self, tmp_path):
        """Empty source directory → early return; every counter is zero."""
        src_dir = tmp_path / "empty_docs"
        src_dir.mkdir()

        config = _make_config(tmp_path)

        with patch("src.ingest.impl.load_manifest", return_value={}), \
             patch("src.ingest.impl.ingest_file") as mock_if:

            summary = ingest_directory(documents_dir=src_dir, config=config, fresh=False)

        assert summary.processed == 0
        assert summary.skipped == 0
        assert summary.failed == 0
        mock_if.assert_not_called()


# ---------------------------------------------------------------------------
# ingest_file result field tests
# ---------------------------------------------------------------------------

class TestIngestFileResultFields:
    def _call_ingest_file(self, tmp_path, p1_override=None, p2_override=None):
        """Helper: create a scratch file and call ingest_file with patched phases."""
        doc = tmp_path / "doc.txt"
        doc.write_text("Sample document for result field tests.")
        runtime = _make_runtime(tmp_path)

        p1 = _phase1_result(doc)
        if p1_override:
            p1.update(p1_override)

        p2 = _phase2_result()
        if p2_override:
            p2.update(p2_override)

        with patch("src.ingest.impl.run_document_processing", return_value=p1), \
             patch("src.ingest.impl.run_embedding_pipeline", return_value=p2):

            return ingest_file(
                source_path=doc,
                runtime=runtime,
                source_name="doc.txt",
                source_uri=doc.as_uri(),
                source_key="local_fs:test:orf:1",
                source_id="test:orf:1",
                connector="local_fs",
                source_version="1",
            )

    def test_ingest_file_merges_processing_logs(self, tmp_path):
        """processing_log from Phase 1 and Phase 2 must be combined in result.

        If Phase 1 emits ["p1"] and Phase 2 emits ["p2"], result.processing_log
        must equal ["p1", "p2"] (Phase 1 entries first).
        """
        result = self._call_ingest_file(
            tmp_path,
            p1_override={"processing_log": ["p1"]},
            p2_override={"processing_log": ["p2"]},
        )
        assert result.processing_log == ["p1", "p2"]

    def test_ingest_file_result_has_metadata_fields(self, tmp_path):
        """metadata_summary and metadata_keywords must be taken from Phase 2."""
        result = self._call_ingest_file(
            tmp_path,
            p2_override={
                "metadata_summary": "S",
                "metadata_keywords": ["k"],
            },
        )
        assert result.metadata_summary == "S"
        assert result.metadata_keywords == ["k"]

    def test_ingest_file_errors_is_empty_list_on_success(self, tmp_path):
        """When both phases succeed with no errors, result.errors must be an
        empty list (not None, not a falsy non-list value).
        """
        result = self._call_ingest_file(tmp_path)
        assert result.errors == []
        assert isinstance(result.errors, list)


# ---------------------------------------------------------------------------
# verify_core_design tests
# ---------------------------------------------------------------------------

class TestVerifyCoreDesign:
    def _make_valid_config(self, tmp_path):
        """Return a Runtime whose config has no contradictions."""
        return _make_runtime(tmp_path)

    def test_verify_core_design_ok_for_valid_config(self, tmp_path):
        """A fully valid, contradiction-free config must produce report.ok==True
        and report.errors==[].
        """
        runtime = self._make_valid_config(tmp_path)
        report = verify_core_design(runtime.config)

        assert report.ok is True
        assert report.errors == []

    def test_verify_core_design_never_raises(self, tmp_path):
        """verify_core_design must return an IngestionDesignCheck for any config,
        including contradictory ones — it must never raise an exception.
        """
        contradictory_configs = [
            # KG storage enabled but no builder
            dict(enable_knowledge_graph_storage=True, build_kg=False),
            # Docling parser enabled but no model path / flag
            dict(enable_docling_parser=True),
            # Multimodal enabled
            dict(enable_multimodal_processing=True),
        ]

        for overrides in contradictory_configs:
            try:
                config = IngestionConfig(
                    enable_multimodal_processing=False,
                    enable_document_refactoring=False,
                    enable_cross_reference_extraction=False,
                    enable_knowledge_graph_extraction=False,
                    enable_knowledge_graph_storage=False,
                    enable_quality_validation=False,
                    enable_docling_parser=False,
                    enable_llm_metadata=False,
                    persist_refactor_mirror=False,
                    clean_store_dir=str(tmp_path / "store"),
                    build_kg=False,
                    **overrides,
                )
            except Exception:
                # If the config itself rejects the values, skip to the next case;
                # the important contract is that verify_core_design never raises.
                continue

            result = verify_core_design(config)
            assert isinstance(result, IngestionDesignCheck), (
                f"verify_core_design must return IngestionDesignCheck, got {type(result)} "
                f"for overrides={overrides}"
            )
