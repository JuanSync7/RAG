"""Integration tests for the two-phase orchestrator in pipeline/impl.py."""
import hashlib
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.ingest.impl import ingest_file


def _make_runtime(tmp_path, store_subdir="store"):
    from src.ingest.common.types import IngestionConfig, Runtime
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
        clean_store_dir=str(tmp_path / store_subdir),
    )
    return Runtime(
        config=config,
        embedder=MagicMock(),
        weaviate_client=MagicMock(),
        kg_builder=None,
    )


def _phase1_result(doc: Path, cleaned="clean text"):
    """Return a fake DocumentProcessingState for a doc file."""
    data = doc.read_bytes()
    return {
        "source_hash": hashlib.sha256(data).hexdigest(),
        "raw_text": data.decode(),
        "cleaned_text": cleaned,
        "refactored_text": None,
        "errors": [],
        "processing_log": ["document_ingestion:ok", "structure_detection:ok"],
        "structure": {"has_figures": False},
        "multimodal_notes": [],
    }


def _phase2_result():
    return {
        "stored_count": 3,
        "metadata_summary": "A test document.",
        "metadata_keywords": ["test"],
        "errors": [],
        "processing_log": ["chunking:ok", "embedding_storage:ok"],
        "chunks": [],
        "kg_triples": [],
    }


def test_ingest_file_returns_source_hash(tmp_path):
    """ingest_file must return source_hash in result dict."""
    doc = tmp_path / "test.txt"
    doc.write_text("Hello world document content.")
    runtime = _make_runtime(tmp_path)

    with patch("src.ingest.impl.run_document_processing",
               return_value=_phase1_result(doc)), \
         patch("src.ingest.impl.run_embedding_pipeline",
               return_value=_phase2_result()):
        result = ingest_file(
            source_path=doc, runtime=runtime,
            source_name="test.txt", source_uri=doc.as_uri(),
            source_key="local_fs:test:1", source_id="test:1",
            connector="local_fs", source_version="12345",
        )

    assert result.source_hash == hashlib.sha256(doc.read_bytes()).hexdigest()
    assert result.stored_count == 3
    assert result.errors == []


def test_ingest_file_writes_clean_store(tmp_path):
    """ingest_file must write Phase 1 clean text to CleanDocumentStore.

    Note: as of the export_processed gating change, the clean store write
    only happens when both `export_processed=True` and `clean_store_dir` is
    set. This test exercises the opt-in debug-export path.
    """
    doc = tmp_path / "doc.txt"
    doc.write_text("Clean document text.")
    store_dir = tmp_path / "store"
    runtime = _make_runtime(tmp_path, store_subdir="store")
    runtime.config.export_processed = True

    with patch("src.ingest.impl.run_document_processing",
               return_value=_phase1_result(doc, cleaned="Clean document text.")), \
         patch("src.ingest.impl.run_embedding_pipeline",
               return_value=_phase2_result()):
        ingest_file(
            source_path=doc, runtime=runtime,
            source_name="doc.txt", source_uri=doc.as_uri(),
            source_key="local_fs:test:2", source_id="test:2",
            connector="local_fs", source_version="99999",
        )

    from src.ingest.common.clean_store import CleanDocumentStore
    store = CleanDocumentStore(store_dir)
    assert store.exists("local_fs:test:2")
    text, meta = store.read("local_fs:test:2")
    assert text == "Clean document text."
    assert meta["source_key"] == "local_fs:test:2"


def test_phase1_errors_skip_phase2(tmp_path):
    """If Phase 1 returns errors, ingest_file must not call Phase 2."""
    doc = tmp_path / "doc.txt"
    doc.write_text("x")
    runtime = _make_runtime(tmp_path)

    phase1_with_error = {
        "source_hash": "", "raw_text": "", "cleaned_text": "",
        "refactored_text": None, "structure": {}, "multimodal_notes": [],
        "errors": ["read_failed:doc.txt:some error"],
        "processing_log": ["document_ingestion:failed"],
    }

    with patch("src.ingest.impl.run_document_processing",
               return_value=phase1_with_error), \
         patch("src.ingest.impl.run_embedding_pipeline") as mock_p2:
        result = ingest_file(
            source_path=doc, runtime=runtime,
            source_name="doc.txt", source_uri=doc.as_uri(),
            source_key="local_fs:test:3", source_id="test:3",
            connector="local_fs", source_version="0",
        )

    assert result.errors == ["read_failed:doc.txt:some error"]
    assert result.stored_count == 0
    mock_p2.assert_not_called()


def test_phase2_errors_propagate(tmp_path):
    """Errors from Phase 2 must appear in ingest_file result."""
    doc = tmp_path / "doc.txt"
    doc.write_text("some content")
    runtime = _make_runtime(tmp_path)

    phase2_with_error = {**_phase2_result(), "errors": ["embed_failed:weaviate_down"]}

    with patch("src.ingest.impl.run_document_processing",
               return_value=_phase1_result(doc)), \
         patch("src.ingest.impl.run_embedding_pipeline",
               return_value=phase2_with_error):
        result = ingest_file(
            source_path=doc, runtime=runtime,
            source_name="doc.txt", source_uri=doc.as_uri(),
            source_key="local_fs:test:4", source_id="test:4",
            connector="local_fs", source_version="1",
        )

    assert "embed_failed:weaviate_down" in result.errors
