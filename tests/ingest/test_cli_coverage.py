"""Mock-based tests for src/ingest/cli.py to increase coverage.

Covers the ingest() function (lines 73-138) and main() function (lines 282-295).
All test functions that rely on mocks are named test_mock_*.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_summary(
    processed=1,
    skipped=0,
    failed=0,
    stored_chunks=5,
    removed_sources=0,
    errors=None,
    design_warnings=None,
):
    """Build a minimal IngestionRunSummary-like object."""
    summary = MagicMock()
    summary.processed = processed
    summary.skipped = skipped
    summary.failed = failed
    summary.stored_chunks = stored_chunks
    summary.removed_sources = removed_sources
    summary.errors = errors or []
    summary.design_warnings = design_warnings or []
    return summary


# ---------------------------------------------------------------------------
# test_mock_ingest_builds_config
# ---------------------------------------------------------------------------


def test_mock_ingest_builds_config():
    """Verify ingest() constructs IngestionConfig with the right fields and
    calls ingest_directory with them."""
    from src.ingest.cli import ingest

    mock_summary = _make_summary()

    with patch("src.ingest.cli.validate_documents_dir") as mock_validate, patch(
        "src.ingest.cli.ingest_directory"
    ) as mock_ingest_dir, patch("src.ingest.cli.IngestionConfig") as mock_cfg_cls:
        mock_validate.return_value = Path("/docs")
        mock_ingest_dir.return_value = mock_summary
        mock_cfg_instance = MagicMock()
        mock_cfg_cls.return_value = mock_cfg_instance

        ingest(
            documents_dir=Path("/docs"),
            fresh=True,
            update=False,
            build_kg=False,
            semantic_chunking=False,
            export_processed=True,
        )

        # IngestionConfig was constructed with expected kwargs
        call_kwargs = mock_cfg_cls.call_args[1]
        assert call_kwargs["semantic_chunking"] is False
        assert call_kwargs["build_kg"] is False
        assert call_kwargs["export_processed"] is True
        assert call_kwargs["enable_knowledge_graph_extraction"] is False
        assert call_kwargs["enable_knowledge_graph_storage"] is False

        # ingest_directory called with the right arguments
        mock_ingest_dir.assert_called_once()
        id_kwargs = mock_ingest_dir.call_args[1]
        assert id_kwargs["fresh"] is True
        assert id_kwargs["update"] is False
        assert id_kwargs["selected_sources"] is None


# ---------------------------------------------------------------------------
# test_mock_ingest_with_selected_file
# ---------------------------------------------------------------------------


def test_mock_ingest_with_selected_file(tmp_path):
    """Test the selected_file path (lines 116-117) — resolved path is passed
    as a single-element selected_sources list to ingest_directory."""
    from src.ingest.cli import ingest

    target = tmp_path / "doc.md"
    target.write_text("hello")

    mock_summary = _make_summary()

    with patch("src.ingest.cli.validate_documents_dir") as mock_validate, patch(
        "src.ingest.cli.ingest_directory"
    ) as mock_ingest_dir, patch("src.ingest.cli.IngestionConfig"):
        mock_validate.return_value = tmp_path
        mock_ingest_dir.return_value = mock_summary

        ingest(
            documents_dir=tmp_path,
            selected_file=target,
        )

        id_kwargs = mock_ingest_dir.call_args[1]
        assert id_kwargs["selected_sources"] == [target.resolve()]


# ---------------------------------------------------------------------------
# test_mock_ingest_logs_summary
# ---------------------------------------------------------------------------


def test_mock_ingest_logs_summary():
    """Verify logging of summary counts, errors, and design_warnings."""
    from src.ingest.cli import ingest

    mock_summary = _make_summary(
        processed=3,
        failed=1,
        errors=["err1", "err2"],
        design_warnings=["warn1"],
    )

    with patch("src.ingest.cli.validate_documents_dir") as mock_validate, patch(
        "src.ingest.cli.ingest_directory"
    ) as mock_ingest_dir, patch("src.ingest.cli.IngestionConfig"), patch(
        "src.ingest.cli.logger"
    ) as mock_logger:
        mock_validate.return_value = Path("/docs")
        mock_ingest_dir.return_value = mock_summary

        ingest(documents_dir=Path("/docs"))

        # logger.info called once for the summary line
        mock_logger.info.assert_called_once()
        info_call = mock_logger.info.call_args[0]
        assert "Ingestion complete" in info_call[0]

        # logger.warning called for each error + each design_warning
        warning_calls = mock_logger.warning.call_args_list
        assert len(warning_calls) == 3  # 2 errors + 1 design_warning
        warning_msgs = [str(c) for c in warning_calls]
        assert any("err1" in m for m in warning_msgs)
        assert any("warn1" in m for m in warning_msgs)


# ---------------------------------------------------------------------------
# test_mock_ingest_optional_flags
# ---------------------------------------------------------------------------


def test_mock_ingest_optional_flags():
    """Verify optional config kwargs (verbose_stages, persist_refactor_mirror,
    docling_*, vision_*) are forwarded only when not None."""
    from src.ingest.cli import ingest

    mock_summary = _make_summary()

    with patch("src.ingest.cli.validate_documents_dir") as mock_validate, patch(
        "src.ingest.cli.ingest_directory"
    ) as mock_ingest_dir, patch("src.ingest.cli.IngestionConfig") as mock_cfg_cls:
        mock_validate.return_value = Path("/docs")
        mock_ingest_dir.return_value = mock_summary
        mock_cfg_cls.return_value = MagicMock()

        ingest(
            documents_dir=Path("/docs"),
            verbose_stages=True,
            persist_refactor_mirror=False,
            docling_enabled=True,
            docling_model="my-model",
            docling_artifacts_path="/artifacts",
            docling_strict=False,
            docling_auto_download=True,
            vision_enabled=True,
            vision_provider="ollama",
            vision_model="llava",
            vision_api_base_url="http://localhost:8080",
            vision_timeout_seconds=30,
            vision_max_figures=5,
            vision_auto_pull=False,
            vision_strict=True,
        )

        call_kwargs = mock_cfg_cls.call_args[1]
        assert call_kwargs["verbose_stage_logs"] is True
        assert call_kwargs["persist_refactor_mirror"] is False
        assert call_kwargs["enable_docling_parser"] is True
        assert call_kwargs["docling_model"] == "my-model"
        assert call_kwargs["docling_artifacts_path"] == "/artifacts"
        assert call_kwargs["docling_strict"] is False
        assert call_kwargs["docling_auto_download"] is True
        assert call_kwargs["enable_vision_processing"] is True
        assert call_kwargs["vision_provider"] == "ollama"
        assert call_kwargs["vision_model"] == "llava"
        assert call_kwargs["vision_api_base_url"] == "http://localhost:8080"
        assert call_kwargs["vision_timeout_seconds"] == 30
        assert call_kwargs["vision_max_figures"] == 5
        assert call_kwargs["vision_auto_pull"] is False
        assert call_kwargs["vision_strict"] is True


# ---------------------------------------------------------------------------
# test_mock_main_with_file
# ---------------------------------------------------------------------------


def test_mock_main_with_file(tmp_path, monkeypatch):
    """Test main() with --file pointing to an existing tmp file."""
    from src.ingest.cli import main

    target = tmp_path / "test.md"
    target.write_text("content")

    monkeypatch.setattr(sys, "argv", ["cli.py", "--file", str(target)])

    with patch("src.ingest.cli.ingest") as mock_ingest, patch(
        "src.ingest.cli.validate_documents_dir"
    ) as mock_validate:
        mock_validate.return_value = tmp_path

        main()

        mock_ingest.assert_called_once()
        call_kwargs = mock_ingest.call_args[1]
        assert call_kwargs["selected_file"] == target.resolve()
        assert call_kwargs["documents_dir"] == target.resolve().parent


# ---------------------------------------------------------------------------
# test_mock_main_with_dir
# ---------------------------------------------------------------------------


def test_mock_main_with_dir(tmp_path, monkeypatch):
    """Test main() with --dir pointing to a directory."""
    from src.ingest.cli import main

    monkeypatch.setattr(sys, "argv", ["cli.py", "--dir", str(tmp_path)])

    with patch("src.ingest.cli.ingest") as mock_ingest, patch(
        "src.ingest.cli.validate_documents_dir"
    ) as mock_validate:
        mock_validate.return_value = tmp_path

        main()

        mock_ingest.assert_called_once()
        call_kwargs = mock_ingest.call_args[1]
        assert call_kwargs["documents_dir"] == tmp_path
        assert call_kwargs["selected_file"] is None


# ---------------------------------------------------------------------------
# test_mock_main_file_not_found
# ---------------------------------------------------------------------------


def test_mock_main_file_not_found(tmp_path, monkeypatch):
    """Test main() with --file pointing to a nonexistent file triggers parser.error."""
    from src.ingest.cli import main

    nonexistent = tmp_path / "ghost.md"
    monkeypatch.setattr(sys, "argv", ["cli.py", "--file", str(nonexistent)])

    # argparse.error() raises SystemExit
    with pytest.raises(SystemExit):
        main()


# ---------------------------------------------------------------------------
# test_mock_main_no_kg_flag
# ---------------------------------------------------------------------------


def test_mock_main_no_kg_flag(monkeypatch):
    """Test main() with --no-kg sets build_kg=False."""
    from src.ingest.cli import main

    monkeypatch.setattr(sys, "argv", ["cli.py", "--no-kg"])

    with patch("src.ingest.cli.ingest") as mock_ingest, patch(
        "src.ingest.cli.validate_documents_dir"
    ) as mock_validate:
        mock_validate.return_value = Path("/docs")

        main()

        call_kwargs = mock_ingest.call_args[1]
        assert call_kwargs["build_kg"] is False


# ---------------------------------------------------------------------------
# test_mock_main_update_flag
# ---------------------------------------------------------------------------


def test_mock_main_update_flag(monkeypatch):
    """Test main() with --update sets update=True and fresh=False."""
    from src.ingest.cli import main

    monkeypatch.setattr(sys, "argv", ["cli.py", "--update"])

    with patch("src.ingest.cli.ingest") as mock_ingest, patch(
        "src.ingest.cli.validate_documents_dir"
    ) as mock_validate:
        mock_validate.return_value = Path("/docs")

        main()

        call_kwargs = mock_ingest.call_args[1]
        assert call_kwargs["update"] is True
        assert call_kwargs["fresh"] is False
