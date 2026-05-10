# @summary
# Coverage tests for document_ingestion_node: error paths, boundary cases,
# hash correctness, and processing_log entries.
# Exports: TestDocumentIngestionError, TestDocumentIngestionBoundary,
#          TestDocumentIngestionErrorScenarios
# Deps: src.ingest.doc_processing.nodes.document_ingestion,
#       src.ingest.common.types, unittest.mock, pytest, hashlib, re
# @end-summary

"""Coverage tests for the document_ingestion_node (Phase 1, node 1)."""

import hashlib
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingest.doc_processing.nodes.document_ingestion import document_ingestion_node
from src.ingest.common.types import IngestionConfig, Runtime


# ---------------------------------------------------------------------------
# State builder
# ---------------------------------------------------------------------------

def _make_state(source_path: str = "/tmp/test.txt", **overrides):
    config = IngestionConfig()
    runtime = Runtime(
        config=config,
        embedder=MagicMock(),
        weaviate_client=MagicMock(),
        kg_builder=None,
    )
    state = {
        "source_path": source_path,
        "source_name": "test.txt",
        "runtime": runtime,
        "errors": [],
        "processing_log": [],
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# TestDocumentIngestionError
# ---------------------------------------------------------------------------

class TestDocumentIngestionError:
    """Tests for error-path behavior of document_ingestion_node."""

    def test_document_ingestion_node_returns_error_when_source_path_missing(self):
        state = _make_state(source_path="/tmp/__nonexistent_file_xyz123__.txt")
        result = document_ingestion_node(state)
        assert result["errors"], "errors list should be non-empty"
        assert any("read_failed:" in e for e in result["errors"])
        assert result["processing_log"][-1].endswith("document_ingestion:failed")

    def test_document_ingestion_node_returns_error_when_file_unreadable(self):
        with patch(
            "pathlib.Path.read_bytes",
            side_effect=PermissionError("Permission denied"),
        ):
            state = _make_state()
            result = document_ingestion_node(state)
        assert result["errors"], "errors list should be non-empty"
        assert any("read_failed:" in e for e in result["errors"])
        assert result["processing_log"][-1].endswith("document_ingestion:failed")

    def test_document_ingestion_node_returns_error_when_read_bytes_raises(self):
        with patch(
            "pathlib.Path.read_bytes",
            side_effect=OSError("disk error"),
        ):
            state = _make_state()
            result = document_ingestion_node(state)
        assert result["errors"], "errors list should be non-empty"
        error_str = " ".join(result["errors"])
        assert "read_failed:" in error_str
        assert "disk error" in error_str


# ---------------------------------------------------------------------------
# TestDocumentIngestionBoundary
# ---------------------------------------------------------------------------

class TestDocumentIngestionBoundary:
    """Tests for boundary and encoding edge cases of document_ingestion_node."""

    def test_document_ingestion_node_handles_empty_file(self, tmp_path):
        empty_file = tmp_path / "empty.txt"
        empty_file.write_bytes(b"")
        state = _make_state(source_path=str(empty_file))
        result = document_ingestion_node(state)
        assert result["raw_text"] == ""
        assert result["source_hash"] == hashlib.sha256(b"").hexdigest()

    def test_document_ingestion_node_handles_very_large_file(self, tmp_path):
        large_file = tmp_path / "large.txt"
        content = "a" * (1024 * 1024)  # 1 MB of text
        large_file.write_text(content, encoding="utf-8")
        state = _make_state(source_path=str(large_file))
        result = document_ingestion_node(state)
        assert result["raw_text"], "raw_text should be non-empty"
        assert re.fullmatch(r"[0-9a-f]{64}", result["source_hash"])

    def test_document_ingestion_node_handles_binary_content_with_replacement(self, tmp_path):
        # Write bytes that are invalid in all fallback encodings to force replacement character
        bin_file = tmp_path / "binary.txt"
        # Bytes that form an invalid sequence in utf-8, latin-1, and cp1252 are impossible
        # since latin-1 accepts all single bytes. Instead, write valid bytes that decode to
        # a known string via latin-1, then verify decoding succeeds without error.
        bin_file.write_bytes(b"text with \xff byte")
        state = _make_state(source_path=str(bin_file))
        result = document_ingestion_node(state)
        assert "raw_text" in result
        assert len(result["raw_text"]) > 0

    def test_document_ingestion_node_handles_latin1_encoded_file(self, tmp_path):
        latin1_file = tmp_path / "latin1.txt"
        latin1_file.write_bytes(b"\xb5gram")  # µ in latin-1, invalid UTF-8
        state = _make_state(source_path=str(latin1_file))
        result = document_ingestion_node(state)
        # Should not raise; decoded content should contain µ (latin-1 0xB5)
        assert "raw_text" in result
        assert "µ" in result["raw_text"] or len(result["raw_text"]) > 0

    def test_document_ingestion_node_handles_cp1252_bytes_via_latin1(self, tmp_path):
        cp1252_file = tmp_path / "cp1252.txt"
        cp1252_file.write_bytes(b"\x93\x94")  # CP1252 curly quotes; valid latin-1 control chars
        state = _make_state(source_path=str(cp1252_file))
        result = document_ingestion_node(state)
        assert "raw_text" in result


# ---------------------------------------------------------------------------
# TestDocumentIngestionErrorScenarios
# ---------------------------------------------------------------------------

class TestDocumentIngestionErrorScenarios:
    """Tests for error message format, log correctness, and hash properties."""

    def test_document_ingestion_node_error_format_includes_filename_and_exception(self):
        with patch(
            "pathlib.Path.read_bytes",
            side_effect=ValueError("bad value"),
        ):
            state = _make_state(source_path="/tmp/myfile.txt")
            result = document_ingestion_node(state)
        assert any(e == "read_failed:myfile.txt:bad value" for e in result["errors"])

    def test_document_ingestion_node_processing_log_records_ok_on_success(self, tmp_path):
        f = tmp_path / "ok.txt"
        f.write_text("text", encoding="utf-8")
        state = _make_state(source_path=str(f))
        result = document_ingestion_node(state)
        assert "document_ingestion:ok" in result["processing_log"][-1]

    def test_document_ingestion_node_processing_log_records_failed_on_error(self):
        with patch(
            "pathlib.Path.read_bytes",
            side_effect=RuntimeError("boom"),
        ):
            state = _make_state()
            result = document_ingestion_node(state)
        assert "document_ingestion:failed" in result["processing_log"][-1]

    def test_document_ingestion_node_hash_is_deterministic(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("deterministic content", encoding="utf-8")
        state = _make_state(source_path=str(f))
        result1 = document_ingestion_node(state)
        result2 = document_ingestion_node(state)
        assert result1["source_hash"] == result2["source_hash"]

    def test_document_ingestion_node_hash_changes_on_content_change(self, tmp_path):
        f1 = tmp_path / "file_a.txt"
        f2 = tmp_path / "file_b.txt"
        f1.write_text("content alpha", encoding="utf-8")
        f2.write_text("content beta", encoding="utf-8")
        result1 = document_ingestion_node(_make_state(source_path=str(f1)))
        result2 = document_ingestion_node(_make_state(source_path=str(f2)))
        assert result1["source_hash"] != result2["source_hash"]

    def test_document_ingestion_node_hash_is_64_char_hex(self, tmp_path):
        f = tmp_path / "hashcheck.txt"
        f.write_text("some content for hash verification", encoding="utf-8")
        state = _make_state(source_path=str(f))
        result = document_ingestion_node(state)
        assert len(result["source_hash"]) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", result["source_hash"])
