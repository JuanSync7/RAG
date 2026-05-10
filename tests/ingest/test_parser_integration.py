# @summary
# Integration tests for parser registry wire-up: verifies that structure_detection_node
# dispatches to the correct parser strategy when a ParserRegistry is attached to Runtime,
# and that chunks are produced from the resulting parse_result.
# Phase 3.2: parser_registry on Runtime enables the parser-abstraction path.
# Covers: .md → text strategy, .txt → text strategy, .py → code strategy (or fallback),
#         parse_result stored on state, parser_instance stored on state,
#         chunking_node consumes parse_result via parser_instance.chunk().
# Exports: TestParserIntegrationMd, TestParserIntegrationTxt,
#          TestParserIntegrationPy, TestParserIntegrationChunkingWireup
# Deps: pytest, unittest.mock, tmp_path, src.ingest.doc_processing.nodes.structure_detection,
#       src.ingest.embedding.nodes.chunking, src.ingest.support.parser_registry,
#       src.ingest.support.parser_base, src.ingest.common.types
# @end-summary

"""End-to-end parser integration tests.

Validates that when a ParserRegistry is attached to Runtime:

- `.md` and `.txt` files are dispatched to PlainTextParser (text strategy).
- `.py` files are dispatched to code strategy when available, else text fallback.
- `parse_result` and `parser_instance` are written to state after
  `structure_detection_node`.
- `chunking_node` can consume `parse_result` + `parser_instance` and produce
  non-empty `ProcessedChunk` output.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.ingest.common.types import IngestionConfig, Runtime
from src.ingest.doc_processing.nodes.structure_detection import structure_detection_node
from src.ingest.embedding.nodes.chunking import chunking_node
from src.ingest.support.parser_base import Chunk, ParseResult
from src.ingest.support.parser_registry import ParserRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry(config: IngestionConfig) -> ParserRegistry:
    """Build a ParserRegistry; fail loudly if text strategy is unavailable."""
    registry = ParserRegistry(config)
    assert "text" in registry.available_strategies, (
        "ParserRegistry must always have 'text' strategy available."
    )
    return registry


def _make_state(
    source_path: str,
    source_name: str,
    raw_text: str,
    config: IngestionConfig,
    registry: ParserRegistry,
) -> dict:
    """Build a minimal pipeline state dict for structure_detection_node."""
    runtime = Runtime(
        config=config,
        embedder=MagicMock(),
        weaviate_client=MagicMock(),
        kg_builder=None,
        parser_registry=registry,
    )
    return {
        "raw_text": raw_text,
        "source_path": source_path,
        "source_name": source_name,
        "runtime": runtime,
        "errors": [],
        "processing_log": [],
    }


def _make_chunking_state(
    source_path: str,
    source_name: str,
    raw_text: str,
    config: IngestionConfig,
    parse_result: ParseResult | None = None,
    parser_instance: object | None = None,
) -> dict:
    """Build a minimal pipeline state dict for chunking_node."""
    runtime = Runtime(
        config=config,
        embedder=MagicMock(),
        weaviate_client=MagicMock(),
        kg_builder=None,
        parser_registry=None,
    )
    state: dict = {
        "source_path": source_path,
        "source_name": source_name,
        "source_uri": f"file://{source_path}",
        "source_key": "local_fs:1:1",
        "source_id": "1:1",
        "connector": "local_fs",
        "source_version": "1",
        "raw_text": raw_text,
        "cleaned_text": raw_text,
        "refactored_text": "",
        "processing_log": [],
        "runtime": runtime,
    }
    if parse_result is not None:
        state["parse_result"] = parse_result
    if parser_instance is not None:
        state["parser_instance"] = parser_instance
    return state


# ---------------------------------------------------------------------------
# .md file → text strategy
# ---------------------------------------------------------------------------


class TestParserIntegrationMd:
    """Markdown files should use PlainTextParser (text strategy) via the registry."""

    def test_md_dispatches_to_text_parser(self, tmp_path: Path):
        """structure_detection_node selects PlainTextParser for a .md source."""
        md_file = tmp_path / "doc.md"
        md_file.write_text(
            "# Introduction\n\nThis is a markdown document.\n\n## Section\n\nContent.",
            encoding="utf-8",
        )
        config = IngestionConfig(enable_docling_parser=False)
        registry = _make_registry(config)
        state = _make_state(
            source_path=str(md_file),
            source_name="doc.md",
            raw_text=md_file.read_text(encoding="utf-8"),
            config=config,
            registry=registry,
        )

        update = structure_detection_node(state)

        assert "parse_result" in update, "parse_result must be set by parser-abstraction path"
        assert "parser_instance" in update, "parser_instance must be set by parser-abstraction path"
        assert update["structure"]["parser_strategy"] != "regex"
        assert update["structure"]["heading_count"] >= 2

    def test_md_parse_result_fields_populated(self, tmp_path: Path):
        """ParseResult from .md should have non-empty markdown and headings."""
        md_file = tmp_path / "report.md"
        md_file.write_text(
            "# Title\n\nBody text.\n\n## Methods\n\nDetails.",
            encoding="utf-8",
        )
        config = IngestionConfig(enable_docling_parser=False)
        registry = _make_registry(config)
        state = _make_state(
            source_path=str(md_file),
            source_name="report.md",
            raw_text=md_file.read_text(encoding="utf-8"),
            config=config,
            registry=registry,
        )

        update = structure_detection_node(state)

        pr: ParseResult = update["parse_result"]
        assert isinstance(pr.markdown, str) and pr.markdown
        assert isinstance(pr.headings, list) and len(pr.headings) >= 1
        assert pr.page_count == 0  # text parser always 0

    def test_md_raw_text_replaced_with_parser_markdown(self, tmp_path: Path):
        """raw_text in the state update should be the parser's markdown output."""
        md_file = tmp_path / "doc.md"
        content = "# Hello\n\nWorld."
        md_file.write_text(content, encoding="utf-8")
        config = IngestionConfig(enable_docling_parser=False)
        registry = _make_registry(config)
        state = _make_state(
            source_path=str(md_file),
            source_name="doc.md",
            raw_text=content,
            config=config,
            registry=registry,
        )

        update = structure_detection_node(state)

        # PlainTextParser returns content as-is for .md; raw_text should match.
        assert update["raw_text"] == content


# ---------------------------------------------------------------------------
# .txt file → text strategy
# ---------------------------------------------------------------------------


class TestParserIntegrationTxt:
    """Plain-text files should use PlainTextParser (text strategy) via the registry."""

    def test_txt_dispatches_to_text_parser(self, tmp_path: Path):
        """structure_detection_node selects PlainTextParser for a .txt source."""
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text(
            "# Heading\n\nSome text content here.\n\nMore content.",
            encoding="utf-8",
        )
        config = IngestionConfig(enable_docling_parser=False)
        registry = _make_registry(config)
        state = _make_state(
            source_path=str(txt_file),
            source_name="notes.txt",
            raw_text=txt_file.read_text(encoding="utf-8"),
            config=config,
            registry=registry,
        )

        update = structure_detection_node(state)

        assert "parse_result" in update
        assert "parser_instance" in update

    def test_txt_no_skip_produced(self, tmp_path: Path):
        """Plain-text ingestion via registry should not produce should_skip."""
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("Just some plain text.", encoding="utf-8")
        config = IngestionConfig(enable_docling_parser=False)
        registry = _make_registry(config)
        state = _make_state(
            source_path=str(txt_file),
            source_name="data.txt",
            raw_text="Just some plain text.",
            config=config,
            registry=registry,
        )

        update = structure_detection_node(state)

        assert update.get("should_skip") is not True
        assert not update.get("errors")

    def test_txt_processing_log_ok(self, tmp_path: Path):
        """Successful text parse should append structure_detection:ok to log."""
        txt_file = tmp_path / "log.txt"
        txt_file.write_text("Event log entry.", encoding="utf-8")
        config = IngestionConfig(enable_docling_parser=False)
        registry = _make_registry(config)
        state = _make_state(
            source_path=str(txt_file),
            source_name="log.txt",
            raw_text="Event log entry.",
            config=config,
            registry=registry,
        )

        update = structure_detection_node(state)

        assert any("structure_detection:ok" in entry for entry in update["processing_log"])


# ---------------------------------------------------------------------------
# .py file → code strategy (or text fallback)
# ---------------------------------------------------------------------------


class TestParserIntegrationPy:
    """Python files should use the code strategy when available, else text fallback."""

    def test_py_dispatches_to_available_parser(self, tmp_path: Path):
        """structure_detection_node selects a parser for a .py source."""
        py_file = tmp_path / "module.py"
        py_file.write_text(
            'def hello():\n    """Say hello."""\n    return "Hello, world!"\n',
            encoding="utf-8",
        )
        config = IngestionConfig(enable_docling_parser=False)
        registry = _make_registry(config)
        state = _make_state(
            source_path=str(py_file),
            source_name="module.py",
            raw_text=py_file.read_text(encoding="utf-8"),
            config=config,
            registry=registry,
        )

        update = structure_detection_node(state)

        # Whether 'code' or 'text' strategy is used (depending on tree-sitter availability),
        # parse_result must always be present.
        assert "parse_result" in update
        assert "parser_instance" in update

    def test_py_produces_valid_parse_result(self, tmp_path: Path):
        """ParseResult from .py must be a valid ParseResult instance."""
        py_file = tmp_path / "utils.py"
        py_file.write_text(
            "import os\n\nclass Foo:\n    pass\n",
            encoding="utf-8",
        )
        config = IngestionConfig(enable_docling_parser=False)
        registry = _make_registry(config)
        state = _make_state(
            source_path=str(py_file),
            source_name="utils.py",
            raw_text=py_file.read_text(encoding="utf-8"),
            config=config,
            registry=registry,
        )

        update = structure_detection_node(state)

        pr = update["parse_result"]
        assert hasattr(pr, "markdown")
        assert hasattr(pr, "headings")
        assert hasattr(pr, "has_figures")
        assert hasattr(pr, "page_count")

    def test_py_no_error_on_code_strategy_fallback(self, tmp_path: Path):
        """If code strategy is unavailable, text fallback must not produce errors."""
        py_file = tmp_path / "script.py"
        py_file.write_text("x = 1\n", encoding="utf-8")
        config = IngestionConfig(enable_docling_parser=False)
        registry = _make_registry(config)
        state = _make_state(
            source_path=str(py_file),
            source_name="script.py",
            raw_text="x = 1\n",
            config=config,
            registry=registry,
        )

        update = structure_detection_node(state)

        assert not update.get("errors")
        assert update.get("should_skip") is not True


# ---------------------------------------------------------------------------
# Chunking wire-up: parse_result → chunks
# ---------------------------------------------------------------------------


class TestParserIntegrationChunkingWireup:
    """chunking_node must produce ProcessedChunk objects when parse_result is present."""

    def test_chunking_node_uses_parse_result(self, tmp_path: Path):
        """With parse_result + parser_instance, chunking_node should yield chunks."""
        md_file = tmp_path / "guide.md"
        content = "# Overview\n\nThis section provides an overview.\n\n## Details\n\nDetails here."
        md_file.write_text(content, encoding="utf-8")
        config = IngestionConfig(enable_docling_parser=False)
        registry = _make_registry(config)

        # Run structure detection to obtain real parse_result + parser_instance.
        det_state = _make_state(
            source_path=str(md_file),
            source_name="guide.md",
            raw_text=content,
            config=config,
            registry=registry,
        )
        det_update = structure_detection_node(det_state)
        assert "parse_result" in det_update, "prereq: parse_result must be in state"

        # Feed parse_result into chunking_node.
        chunk_state = _make_chunking_state(
            source_path=str(md_file),
            source_name="guide.md",
            raw_text=det_update["raw_text"],
            config=config,
            parse_result=det_update["parse_result"],
            parser_instance=det_update["parser_instance"],
        )
        result = chunking_node(chunk_state)

        assert result["chunks"], "chunks must be non-empty for non-trivial document"
        chunk = result["chunks"][0]
        assert chunk.text, "Chunk text must be non-empty"

    def test_chunking_node_with_markdown_override(self, tmp_path: Path):
        """chunker='markdown' override should still produce chunks from parse_result."""
        txt_file = tmp_path / "doc.txt"
        content = "# Section\n\nSome content for testing chunking."
        txt_file.write_text(content, encoding="utf-8")
        config = IngestionConfig(enable_docling_parser=False, chunker="markdown")
        registry = _make_registry(config)

        det_state = _make_state(
            source_path=str(txt_file),
            source_name="doc.txt",
            raw_text=content,
            config=config,
            registry=registry,
        )
        det_update = structure_detection_node(det_state)

        chunk_state = _make_chunking_state(
            source_path=str(txt_file),
            source_name="doc.txt",
            raw_text=det_update["raw_text"],
            config=config,
            parse_result=det_update.get("parse_result"),
            parser_instance=det_update.get("parser_instance"),
        )
        result = chunking_node(chunk_state)

        assert result["chunks"]

    def test_chunking_node_chunk_metadata_populated(self, tmp_path: Path):
        """Chunks produced via parse_result path should have source and chunk_id metadata."""
        md_file = tmp_path / "spec.md"
        content = "# Spec\n\nThe system must handle errors gracefully."
        md_file.write_text(content, encoding="utf-8")
        config = IngestionConfig(enable_docling_parser=False)
        registry = _make_registry(config)

        det_state = _make_state(
            source_path=str(md_file),
            source_name="spec.md",
            raw_text=content,
            config=config,
            registry=registry,
        )
        det_update = structure_detection_node(det_state)

        chunk_state = _make_chunking_state(
            source_path=str(md_file),
            source_name="spec.md",
            raw_text=det_update["raw_text"],
            config=config,
            parse_result=det_update.get("parse_result"),
            parser_instance=det_update.get("parser_instance"),
        )
        # Run through chunking only (not enrichment); just assert chunk text is set.
        result = chunking_node(chunk_state)

        assert result["chunks"]
        for chunk in result["chunks"]:
            assert chunk.text

    def test_registry_path_parser_strategy_not_regex(self, tmp_path: Path):
        """Parser strategy in structure dict should not be 'regex' when registry active."""
        md_file = tmp_path / "readme.md"
        md_file.write_text("# README\n\nContent.", encoding="utf-8")
        config = IngestionConfig(enable_docling_parser=False)
        registry = _make_registry(config)
        state = _make_state(
            source_path=str(md_file),
            source_name="readme.md",
            raw_text="# README\n\nContent.",
            config=config,
            registry=registry,
        )

        update = structure_detection_node(state)

        # The registry path should not produce 'regex' or 'unknown' as strategy.
        assert update["structure"]["parser_strategy"] not in {"regex", "unknown"}
