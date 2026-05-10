# @summary
# Bash script parser-based structural entity extraction.
# Uses regex-based parsing (no tree-sitter dependency) for functions,
# sourced files, and key variable assignments in shell scripts.
# Exports: BashParserExtractor
# Deps: re, src.knowledge_graph.common.schemas
# @end-summary
"""Bash script parser-based structural entity extraction.

Uses lightweight regex-based parsing to extract structural entities
(functions, sourced files, exported variables) from Bash/Shell scripts.
All results are tagged with ``extractor_source="bash_parser"``.

Note: This is a best-effort parser — shell syntax is context-sensitive
and not fully parseable with regexes. The extractor focuses on common
ASIC flow script patterns (Makefile wrappers, tool invocations, source chains).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Set

from src.knowledge_graph.common import (
    Entity,
    ExtractionResult,
    Triple,
)

__all__ = ["BashParserExtractor"]

logger = logging.getLogger("rag.knowledge_graph.bash_parser")

# Regex patterns for shell constructs
_FUNC_PATTERN = re.compile(
    r"^\s*(?:function\s+)?(\w+)\s*\(\s*\)\s*\{", re.MULTILINE
)
_SOURCE_PATTERN = re.compile(
    r"^\s*(?:source|\.)\s+([\"']?)(\S+)\1", re.MULTILINE
)
_EXPORT_PATTERN = re.compile(
    r"^\s*export\s+([A-Z_][A-Z0-9_]*)=", re.MULTILINE
)
_READONLY_PATTERN = re.compile(
    r"^\s*(?:readonly|declare\s+-r)\s+([A-Z_][A-Z0-9_]*)=", re.MULTILINE
)


class BashParserExtractor:
    """Deterministic structural extractor for Bash/Shell scripts.

    No external dependencies — uses regex-based parsing.
    """

    def __init__(self) -> None:
        self._parser_available = True

    @property
    def name(self) -> str:
        return "bash_parser"

    def extract(self, text: str, source: str = "") -> ExtractionResult:
        """Extract structural entities and relationships from shell source.

        Args:
            text: Shell script source code.
            source: File path or identifier for provenance tracking.

        Returns:
            ExtractionResult with entities and triples.
        """
        if not text or not text.strip():
            return ExtractionResult(entities=[], triples=[], descriptions=[])

        entities: List[Entity] = []
        triples: List[Triple] = []
        seen_names: Set[str] = set()

        script_name = Path(source).stem if source else "<script>"

        # Extract functions
        for match in _FUNC_PATTERN.finditer(text):
            fn_name = match.group(1)
            if fn_name not in seen_names:
                seen_names.add(fn_name)
                entities.append(
                    Entity(
                        name=fn_name,
                        type="BashFunction",
                        sources=[source] if source else [],
                        mention_count=1,
                        extractor_source="bash_parser",
                    )
                )
                triples.append(
                    Triple(
                        subject=script_name,
                        predicate="contains",
                        object=fn_name,
                        source=source,
                        extractor_source="bash_parser",
                    )
                )

        # Extract sourced files
        for match in _SOURCE_PATTERN.finditer(text):
            sourced_file = match.group(2)
            # Skip variable expansions
            if "$" in sourced_file:
                continue
            sourced_name = Path(sourced_file).stem
            if sourced_name not in seen_names:
                seen_names.add(sourced_name)
                entities.append(
                    Entity(
                        name=sourced_name,
                        type="BashScript",
                        sources=[source] if source else [],
                        mention_count=1,
                        extractor_source="bash_parser",
                    )
                )
                triples.append(
                    Triple(
                        subject=script_name,
                        predicate="depends_on",
                        object=sourced_name,
                        source=source,
                        extractor_source="bash_parser",
                    )
                )

        # Extract exported/readonly constants
        for pattern in (_EXPORT_PATTERN, _READONLY_PATTERN):
            for match in pattern.finditer(text):
                var_name = match.group(1)
                if var_name not in seen_names:
                    seen_names.add(var_name)
                    entities.append(
                        Entity(
                            name=var_name,
                            type="BashVariable",
                            sources=[source] if source else [],
                            mention_count=1,
                            extractor_source="bash_parser",
                        )
                    )
                    triples.append(
                        Triple(
                            subject=script_name,
                            predicate="contains",
                            object=var_name,
                            source=source,
                            extractor_source="bash_parser",
                        )
                    )

        return ExtractionResult(entities=entities, triples=triples, descriptions=[])

    def extract_file(self, file_path: str) -> ExtractionResult:
        """Extract from a shell script on disk.

        Args:
            file_path: Path to a .sh/.bash file.

        Returns:
            ExtractionResult or empty result on read error.
        """
        path = Path(file_path)
        if path.suffix not in (".sh", ".bash", ""):
            logger.debug("Skipping non-shell file: %s", file_path)
            return ExtractionResult(entities=[], triples=[], descriptions=[])
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Cannot read %s: %s", file_path, exc)
            return ExtractionResult(entities=[], triples=[], descriptions=[])
        return self.extract(text, source=str(path))

    def extract_entities(self, text: str) -> Set[str]:
        """Protocol method: return entity names from text."""
        result = self.extract(text)
        return {e.name for e in result.entities}

    def extract_relations(
        self, text: str, known_entities: Set[str]
    ) -> List[Triple]:
        """Protocol method: return triples from text."""
        result = self.extract(text)
        return result.triples
