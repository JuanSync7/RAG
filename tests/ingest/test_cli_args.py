# @summary
# Tests for _build_parser() CLI argument parsing in src.ingest.cli.
# Exports: (pytest test functions)
# Deps: src.ingest.cli, argparse, pytest
# @end-summary
"""Tests for src.ingest.cli._build_parser argument flags.

Note: verbose_stages tri-state tests already exist in test_ingest_cli.py and
are not duplicated here. This module covers all other parser flags.
"""

import pytest

from src.ingest.cli import _build_parser


# ---------------------------------------------------------------------------
# --update / update_mode
# ---------------------------------------------------------------------------

def test_update_flag_sets_update_mode_true():
    parser = _build_parser()
    args = parser.parse_args(["--dir", "/tmp", "--update"])
    assert args.update is True


def test_update_flag_absent_defaults_false():
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.update is False


# ---------------------------------------------------------------------------
# --file / --dir (source selection)
# ---------------------------------------------------------------------------

def test_file_arg_sets_file():
    parser = _build_parser()
    args = parser.parse_args(["--file", "/tmp/doc.pdf"])
    assert str(args.file) == "/tmp/doc.pdf"


def test_file_arg_absent_defaults_none():
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.file is None


def test_dir_arg_sets_source_dir():
    parser = _build_parser()
    args = parser.parse_args(["--dir", "/tmp/docs"])
    assert str(args.dir) == "/tmp/docs"


def test_dir_arg_absent_defaults_none():
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.dir is None


def test_file_and_dir_are_mutually_exclusive():
    """--file and --dir cannot be used together."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--file", "/tmp/a.pdf", "--dir", "/tmp"])


# ---------------------------------------------------------------------------
# --no-kg
# ---------------------------------------------------------------------------

def test_no_kg_flag_sets_no_kg_true():
    parser = _build_parser()
    args = parser.parse_args(["--no-kg"])
    assert args.no_kg is True


def test_no_kg_flag_absent_defaults_false():
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.no_kg is False


# ---------------------------------------------------------------------------
# --no-semantic
# ---------------------------------------------------------------------------

def test_no_semantic_flag_sets_no_semantic_true():
    parser = _build_parser()
    args = parser.parse_args(["--no-semantic"])
    assert args.no_semantic is True


def test_no_semantic_flag_absent_defaults_false():
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.no_semantic is False


# ---------------------------------------------------------------------------
# --export-processed
# ---------------------------------------------------------------------------

def test_export_processed_flag_true():
    parser = _build_parser()
    args = parser.parse_args(["--export-processed"])
    assert args.export_processed is True


def test_export_processed_absent_defaults_false():
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.export_processed is False


# ---------------------------------------------------------------------------
# --no-docling / --docling-model
# ---------------------------------------------------------------------------

def test_no_docling_flag_sets_no_docling_true():
    """--no-docling disables the Docling parser."""
    parser = _build_parser()
    args = parser.parse_args(["--no-docling"])
    assert args.no_docling is True


def test_no_docling_flag_absent_defaults_false():
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.no_docling is False


def test_docling_model_arg():
    parser = _build_parser()
    args = parser.parse_args(["--docling-model", "docling-v2"])
    assert args.docling_model == "docling-v2"


def test_docling_model_absent_defaults_none():
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.docling_model is None


def test_docling_artifacts_path_arg():
    parser = _build_parser()
    args = parser.parse_args(["--docling-artifacts-path", "/models/docling"])
    assert args.docling_artifacts_path == "/models/docling"


def test_docling_non_strict_flag():
    parser = _build_parser()
    args = parser.parse_args(["--docling-non-strict"])
    assert args.docling_non_strict is True


# ---------------------------------------------------------------------------
# --vision / --no-vision / --vision-model / --vision-provider
# ---------------------------------------------------------------------------

def test_enable_vision_flag():
    """--vision sets vision_enabled to True."""
    parser = _build_parser()
    args = parser.parse_args(["--vision"])
    assert args.vision_enabled is True


def test_no_vision_flag_sets_vision_false():
    """--no-vision sets vision_enabled to False."""
    parser = _build_parser()
    args = parser.parse_args(["--no-vision"])
    assert args.vision_enabled is False


def test_vision_absent_defaults_none():
    """With no vision flag, vision_enabled defaults to None (tri-state)."""
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.vision_enabled is None


def test_vision_model_arg():
    parser = _build_parser()
    args = parser.parse_args(["--vision-model", "llava:7b"])
    assert args.vision_model == "llava:7b"


def test_vision_model_absent_defaults_none():
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.vision_model is None


def test_vision_provider_ollama():
    parser = _build_parser()
    args = parser.parse_args(["--vision-provider", "ollama"])
    assert args.vision_provider == "ollama"


def test_vision_provider_openai_compatible():
    parser = _build_parser()
    args = parser.parse_args(["--vision-provider", "openai_compatible"])
    assert args.vision_provider == "openai_compatible"


def test_vision_provider_invalid_raises_system_exit():
    """An unrecognised vision provider must cause SystemExit (choices guard)."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--vision-provider", "invalid_provider"])


def test_vision_strict_flag():
    parser = _build_parser()
    args = parser.parse_args(["--vision-strict"])
    assert args.vision_strict is True


def test_no_vision_auto_pull_flag():
    parser = _build_parser()
    args = parser.parse_args(["--no-vision-auto-pull"])
    assert args.no_vision_auto_pull is True


# ---------------------------------------------------------------------------
# --export-obsidian
# ---------------------------------------------------------------------------

def test_export_obsidian_flag():
    parser = _build_parser()
    args = parser.parse_args(["--export-obsidian"])
    assert args.export_obsidian is True


def test_export_obsidian_absent_defaults_false():
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.export_obsidian is False


# ---------------------------------------------------------------------------
# --no-refactor-mirror
# ---------------------------------------------------------------------------

def test_no_refactor_mirror_flag():
    parser = _build_parser()
    args = parser.parse_args(["--no-refactor-mirror"])
    assert args.no_refactor_mirror is True


def test_no_refactor_mirror_absent_defaults_false():
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.no_refactor_mirror is False


# ---------------------------------------------------------------------------
# Combined flag interactions
# ---------------------------------------------------------------------------

def test_multiple_flags_together():
    """Multiple non-conflicting flags can be combined in one invocation."""
    parser = _build_parser()
    args = parser.parse_args([
        "--dir", "/tmp/docs",
        "--update",
        "--no-kg",
        "--no-semantic",
        "--export-processed",
    ])
    assert str(args.dir) == "/tmp/docs"
    assert args.update is True
    assert args.no_kg is True
    assert args.no_semantic is True
    assert args.export_processed is True


def test_unknown_arg_raises_system_exit():
    """An unrecognised argument must cause SystemExit."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--nonexistent-flag"])
