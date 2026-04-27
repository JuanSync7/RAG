"""Tests for ParserRegistry in src/ingest/support/parser_registry.py.

Covers:
- __init__ with Docling ImportError -> 'document' strategy absent
- __init__ with tree-sitter ImportError -> 'code' strategy absent
- get_parser() with forced strategy not registered -> RuntimeError
- get_parser() with unknown extension -> warning + fallback to 'text'
- get_parser() with resolved strategy not registered -> fallback to 'text'
- warmup_all() with a parser that raises in warmup -> warns but doesn't re-raise
- get_parser_for() module-level convenience function
- ensure_all_ready() module-level convenience function
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from src.ingest.support.parser_registry import (
    ParserRegistry,
    get_parser_for,
    ensure_all_ready,
)


# ---------------------------------------------------------------------------
# Config stubs
# ---------------------------------------------------------------------------

def _auto_config(enable_docling: bool = True) -> object:
    return type(
        "Config",
        (),
        {"parser_strategy": "auto", "enable_docling_parser": enable_docling},
    )()


def _forced_config(strategy: str) -> object:
    return type(
        "Config",
        (),
        {"parser_strategy": strategy, "enable_docling_parser": True},
    )()


# ---------------------------------------------------------------------------
# __init__: conditional strategy registration
# ---------------------------------------------------------------------------


class TestRegistryInit:
    def test_docling_import_error_excludes_document_strategy(self) -> None:
        """When Docling import fails, 'document' strategy must NOT be registered."""
        # Patch the Docling import inside parser_registry to fail
        with patch.dict(
            sys.modules,
            {"src.ingest.support.docling": None},
        ):
            registry = ParserRegistry(_auto_config(enable_docling=True))

        assert "document" not in registry.available_strategies
        assert "text" in registry.available_strategies

    def test_docling_disabled_in_config_excludes_document_strategy(self) -> None:
        """enable_docling_parser=False suppresses 'document' strategy without import error."""
        registry = ParserRegistry(_auto_config(enable_docling=False))
        assert "document" not in registry.available_strategies

    def test_tree_sitter_import_error_excludes_code_strategy(self) -> None:
        """When tree-sitter (parser_code) import fails, 'code' strategy must NOT be registered."""
        with patch.dict(
            sys.modules,
            {"src.ingest.support.parser_code": None},
        ):
            registry = ParserRegistry(_auto_config())

        assert "code" not in registry.available_strategies
        assert "text" in registry.available_strategies


# ---------------------------------------------------------------------------
# get_parser(): forced strategy
# ---------------------------------------------------------------------------


class TestGetParserForcedStrategy:
    def test_forced_strategy_not_registered_raises_runtime_error(self) -> None:
        """Forcing an unregistered strategy must raise RuntimeError."""
        # Force 'document' to be unavailable via config flag
        registry = ParserRegistry(_auto_config(enable_docling=False))
        # Verify 'document' is absent
        assert "document" not in registry.available_strategies

        forced_cfg = _forced_config("document")
        with pytest.raises(RuntimeError, match="parser_strategy='document'"):
            registry.get_parser(Path("file.pdf"), forced_cfg)

    def test_forced_text_strategy_returns_plain_text_parser(self) -> None:
        """Forcing 'text' strategy returns a PlainTextParser instance."""
        from src.ingest.support.parser_text import PlainTextParser

        registry = ParserRegistry(_auto_config())
        parser = registry.get_parser(Path("anything.pdf"), _forced_config("text"))
        assert isinstance(parser, PlainTextParser)


# ---------------------------------------------------------------------------
# get_parser(): auto routing — unknown extension
# ---------------------------------------------------------------------------


class TestGetParserAutoRouting:
    def test_unknown_extension_falls_back_to_text(self, caplog) -> None:
        """Unrecognised extension should warn and fall back to 'text' strategy."""
        import logging

        registry = ParserRegistry(_auto_config())
        with caplog.at_level(logging.WARNING, logger="src.ingest.support.parser_registry"):
            parser = registry.get_parser(Path("file.xyz123"), _auto_config())

        from src.ingest.support.parser_text import PlainTextParser
        assert isinstance(parser, PlainTextParser)
        assert any("xyz123" in msg or "falling back" in msg.lower() for msg in caplog.messages)

    def test_known_extension_returns_correct_parser(self) -> None:
        """A .md file should yield a PlainTextParser under auto routing."""
        from src.ingest.support.parser_text import PlainTextParser

        registry = ParserRegistry(_auto_config())
        parser = registry.get_parser(Path("notes.md"), _auto_config())
        assert isinstance(parser, PlainTextParser)

    def test_resolved_strategy_not_registered_falls_back_to_text(self, caplog) -> None:
        """If extension maps to 'document' but it's absent, fall back to 'text'."""
        import logging

        # Build registry without document strategy
        with patch.dict(sys.modules, {"src.ingest.support.docling": None}):
            registry = ParserRegistry(_auto_config(enable_docling=True))

        assert "document" not in registry.available_strategies

        with caplog.at_level(logging.WARNING, logger="src.ingest.support.parser_registry"):
            parser = registry.get_parser(Path("report.pdf"), _auto_config())

        from src.ingest.support.parser_text import PlainTextParser
        assert isinstance(parser, PlainTextParser)

    def test_filename_map_dockerfile_routes_to_code_or_text(self) -> None:
        """'Dockerfile' maps to 'code' in _FILENAME_MAP (or falls back to text)."""
        registry = ParserRegistry(_auto_config())
        # Regardless of whether CodeParser is available, result must not raise.
        parser = registry.get_parser(Path("Dockerfile"), _auto_config())
        assert parser is not None


# ---------------------------------------------------------------------------
# warmup_all(): non-fatal on exception
# ---------------------------------------------------------------------------


class TestWarmupAll:
    def test_mock_warmup_failure_is_non_fatal(self, caplog) -> None:
        """A parser whose warmup() raises must not propagate the exception."""
        import logging

        registry = ParserRegistry(_auto_config(enable_docling=False))

        # Patch the 'text' class to have a warmup that raises
        bad_parser_cls = MagicMock()
        bad_parser_cls.warmup.side_effect = RuntimeError("boom")
        registry._strategy_map["text"] = bad_parser_cls

        with caplog.at_level(logging.WARNING, logger="src.ingest.support.parser_registry"):
            registry.warmup_all(_auto_config())  # must not raise

        assert any("Warmup failed" in msg or "boom" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


class TestModuleLevelFunctions:
    def test_get_parser_for_returns_a_parser(self) -> None:
        """get_parser_for() convenience wrapper returns a valid parser."""
        parser = get_parser_for(Path("notes.md"), _auto_config())
        assert parser is not None
        assert hasattr(parser, "parse")

    def test_ensure_all_ready_does_not_raise(self) -> None:
        """ensure_all_ready() must run without error on a standard config."""
        # Disable docling so we don't need full docling config attrs
        ensure_all_ready(_auto_config(enable_docling=False))  # should not raise
