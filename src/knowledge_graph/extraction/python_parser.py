# @summary
# Python parser-based structural entity extraction using the ast module.
# Extracts classes, functions, imports, and global variables from Python source.
# Exports: PythonParserExtractor
# Deps: ast, src.knowledge_graph.common.schemas
# @end-summary
"""Python parser-based structural entity extraction.

Uses Python's built-in ``ast`` module for deterministic extraction of
structural entities (classes, functions, imports, globals) from Python
source files.  All results are tagged with ``extractor_source="python_parser"``.
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.knowledge_graph.common import (
    Entity,
    ExtractionResult,
    Triple,
)

__all__ = ["PythonParserExtractor"]

logger = logging.getLogger("rag.knowledge_graph.python_parser")

# Map ast node types to YAML schema node types
AST_TO_SCHEMA_MAP: Dict[str, str] = {
    "ClassDef": "PythonClass",
    "FunctionDef": "PythonFunction",
    "AsyncFunctionDef": "PythonFunction",
    "Import": "PythonImport",
    "ImportFrom": "PythonImport",
}


class PythonParserExtractor:
    """Deterministic structural extractor for Python source files.

    Uses the stdlib ``ast`` module — no external dependencies.
    """

    def __init__(self) -> None:
        self._parser_available = True  # ast is always available

    @property
    def name(self) -> str:
        return "python_parser"

    def extract(self, text: str, source: str = "") -> ExtractionResult:
        """Extract structural entities and relationships from Python source.

        Args:
            text: Python source code.
            source: File path or identifier for provenance tracking.

        Returns:
            ExtractionResult with entities and triples.
        """
        if not text or not text.strip():
            return ExtractionResult(entities=[], triples=[], descriptions=[])

        try:
            tree = ast.parse(text, filename=source or "<string>")
        except SyntaxError as exc:
            logger.warning("Failed to parse %s: %s", source, exc)
            return ExtractionResult(entities=[], triples=[], descriptions=[])

        entities: List[Entity] = []
        triples: List[Triple] = []
        seen_names: Set[str] = set()

        # Module-level entity (the file itself)
        module_name = Path(source).stem if source else "<module>"

        # Walk top-level definitions
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                self._extract_class(
                    node, module_name, source, entities, triples, seen_names
                )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._extract_function(
                    node, module_name, source, entities, triples, seen_names
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imp_name = alias.name
                    if imp_name not in seen_names:
                        seen_names.add(imp_name)
                        entities.append(
                            Entity(
                                name=imp_name,
                                type="PythonImport",
                                sources=[source] if source else [],
                                mention_count=1,
                                extractor_source="python_parser",
                            )
                        )
                        triples.append(
                            Triple(
                                subject=module_name,
                                predicate="depends_on",
                                object=imp_name,
                                source=source,
                                extractor_source="python_parser",
                            )
                        )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod and mod not in seen_names:
                    seen_names.add(mod)
                    entities.append(
                        Entity(
                            name=mod,
                            type="PythonImport",
                            sources=[source] if source else [],
                            mention_count=1,
                            extractor_source="python_parser",
                        )
                    )
                    triples.append(
                        Triple(
                            subject=module_name,
                            predicate="depends_on",
                            object=mod,
                            source=source,
                            extractor_source="python_parser",
                        )
                    )
            elif isinstance(node, ast.Assign):
                self._extract_global_assign(
                    node, module_name, source, entities, triples, seen_names
                )

        return ExtractionResult(entities=entities, triples=triples, descriptions=[])

    def extract_file(self, file_path: str) -> ExtractionResult:
        """Extract from a Python file on disk.

        Args:
            file_path: Path to a .py file.

        Returns:
            ExtractionResult or empty result on read error.
        """
        path = Path(file_path)
        if path.suffix != ".py":
            logger.debug("Skipping non-Python file: %s", file_path)
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

    # ------------------------------------------------------------------
    # Internal extraction helpers
    # ------------------------------------------------------------------

    def _extract_class(
        self,
        node: ast.ClassDef,
        module_name: str,
        source: str,
        entities: List[Entity],
        triples: List[Triple],
        seen_names: Set[str],
    ) -> None:
        """Extract a class definition and its methods."""
        cls_name = node.name
        if cls_name in seen_names:
            return
        seen_names.add(cls_name)

        entities.append(
            Entity(
                name=cls_name,
                type="PythonClass",
                sources=[source] if source else [],
                mention_count=1,
                extractor_source="python_parser",
            )
        )
        triples.append(
            Triple(
                subject=module_name,
                predicate="contains",
                object=cls_name,
                source=source,
                extractor_source="python_parser",
            )
        )

        # Base classes → depends_on
        for base in node.bases:
            base_name = self._name_from_node(base)
            if base_name:
                triples.append(
                    Triple(
                        subject=cls_name,
                        predicate="depends_on",
                        object=base_name,
                        source=source,
                        extractor_source="python_parser",
                    )
                )

        # Methods inside the class
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_name = f"{cls_name}.{child.name}"
                if method_name not in seen_names:
                    seen_names.add(method_name)
                    entities.append(
                        Entity(
                            name=method_name,
                            type="PythonFunction",
                            sources=[source] if source else [],
                            mention_count=1,
                            extractor_source="python_parser",
                        )
                    )
                    triples.append(
                        Triple(
                            subject=cls_name,
                            predicate="contains",
                            object=method_name,
                            source=source,
                            extractor_source="python_parser",
                        )
                    )

    def _extract_function(
        self,
        node: ast.FunctionDef,
        module_name: str,
        source: str,
        entities: List[Entity],
        triples: List[Triple],
        seen_names: Set[str],
    ) -> None:
        """Extract a top-level function definition."""
        fn_name = node.name
        if fn_name in seen_names:
            return
        seen_names.add(fn_name)

        entities.append(
            Entity(
                name=fn_name,
                type="PythonFunction",
                sources=[source] if source else [],
                mention_count=1,
                extractor_source="python_parser",
            )
        )
        triples.append(
            Triple(
                subject=module_name,
                predicate="contains",
                object=fn_name,
                source=source,
                extractor_source="python_parser",
            )
        )

    def _extract_global_assign(
        self,
        node: ast.Assign,
        module_name: str,
        source: str,
        entities: List[Entity],
        triples: List[Triple],
        seen_names: Set[str],
    ) -> None:
        """Extract module-level constant/variable assignments."""
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                # Only extract ALL_CAPS globals (likely constants)
                var_name = target.id
                if var_name not in seen_names:
                    seen_names.add(var_name)
                    entities.append(
                        Entity(
                            name=var_name,
                            type="PythonVariable",
                            sources=[source] if source else [],
                            mention_count=1,
                            extractor_source="python_parser",
                        )
                    )
                    triples.append(
                        Triple(
                            subject=module_name,
                            predicate="contains",
                            object=var_name,
                            source=source,
                            extractor_source="python_parser",
                        )
                    )

    @staticmethod
    def _name_from_node(node: ast.expr) -> Optional[str]:
        """Extract a dotted name from an AST expression node."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = PythonParserExtractor._name_from_node(node.value)
            if prefix:
                return f"{prefix}.{node.attr}"
        return None
