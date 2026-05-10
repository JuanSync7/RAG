#!/usr/bin/env python3
# @summary
# Observability coverage scorer for src/knowledge_graph.
# Checks 7 targets: module-level loggers in sanitizer + regex_extractor,
# logger calls in those files, and time.monotonic() timing in expand(),
# detect(), and find_candidates().
# Run from the RagWeave project root.
# @end-summary
"""
Observability coverage scorer for src/knowledge_graph.

Checks 7 targets:
  T1: query/sanitizer.py             — module-level logger present
  T2: query/sanitizer.py             — at least one logger.* call in a method
  T3: extraction/regex_extractor.py  — module-level logger present
  T4: extraction/regex_extractor.py  — at least one logger.* call in a method
  T5: query/expander.py              — time.monotonic() inside GraphQueryExpander.expand()
  T6: community/detector.py          — time.monotonic() inside CommunityDetector.detect()
  T7: resolution/embedding_resolver.py — time.monotonic() inside EmbeddingResolver.find_candidates()

Usage (from RagWeave project root):
    python3 src/knowledge_graph/observability/score_observability.py
"""

import ast
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

KG_ROOT = Path("src/knowledge_graph")


def _load(rel_path: str) -> ast.Module:
    path = KG_ROOT / rel_path
    return ast.parse(path.read_text())


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _has_module_logger(tree: ast.Module) -> bool:
    """True if the module has a top-level ``logger = logging.getLogger(...)``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Call):
                func = node.value.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "getLogger"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "logging"
                ):
                    return True
    return False


def _has_logger_call_in_any_method(tree: ast.Module) -> bool:
    """True if any method body contains a ``logger.*`` call."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func = child.func
                    if (
                        isinstance(func, ast.Attribute)
                        and isinstance(func.value, ast.Name)
                        and func.value.id == "logger"
                    ):
                        return True
    return False


def _get_method(
    tree: ast.Module, class_name: str, method_name: str
) -> ast.FunctionDef | None:
    """Return the AST node for a specific class method, or None."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == method_name
                ):
                    return item  # type: ignore[return-value]
    return None


def _has_time_monotonic(
    tree: ast.Module, class_name: str, method_name: str
) -> bool:
    """True if the method body contains a ``time.monotonic()`` call."""
    method = _get_method(tree, class_name, method_name)
    if method is None:
        return False
    for node in ast.walk(method):
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "monotonic"
                and isinstance(func.value, ast.Name)
                and func.value.id == "time"
            ):
                return True
    return False


# ---------------------------------------------------------------------------
# Correctness guard
# ---------------------------------------------------------------------------


def _correctness_guard() -> bool:
    """Verify key modules are still importable (detects syntax breaks)."""
    import importlib

    # Ensure project root is on sys.path so `src.*` imports resolve.
    project_root = str(Path(__file__).parent.parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    for mod in [
        "src.knowledge_graph",
        "src.knowledge_graph.backend",
        "src.knowledge_graph.common",
    ]:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            print(
                f"GUARD FAILED: {mod} not importable: {exc}", file=sys.stderr
            )
            return False
    return True


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------


def score() -> int:
    targets = [
        (
            "T1: query/sanitizer.py — module-level logger",
            lambda: _has_module_logger(_load("query/sanitizer.py")),
        ),
        (
            "T2: query/sanitizer.py — logger.* call in method",
            lambda: _has_logger_call_in_any_method(_load("query/sanitizer.py")),
        ),
        (
            "T3: extraction/regex_extractor.py — module-level logger",
            lambda: _has_module_logger(_load("extraction/regex_extractor.py")),
        ),
        (
            "T4: extraction/regex_extractor.py — logger.* call in method",
            lambda: _has_logger_call_in_any_method(
                _load("extraction/regex_extractor.py")
            ),
        ),
        (
            "T5: query/expander.py — time.monotonic() in GraphQueryExpander.expand()",
            lambda: _has_time_monotonic(
                _load("query/expander.py"), "GraphQueryExpander", "expand"
            ),
        ),
        (
            "T6: community/detector.py — time.monotonic() in CommunityDetector.detect()",
            lambda: _has_time_monotonic(
                _load("community/detector.py"), "CommunityDetector", "detect"
            ),
        ),
        (
            "T7: resolution/embedding_resolver.py — time.monotonic() in EmbeddingResolver.find_candidates()",
            lambda: _has_time_monotonic(
                _load("resolution/embedding_resolver.py"),
                "EmbeddingResolver",
                "find_candidates",
            ),
        ),
    ]

    passed = 0
    for desc, check in targets:
        result = check()
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {desc}")
        if result:
            passed += 1

    total = len(targets)
    pct = int(passed / total * 100)
    print(f"\nSCORE: {passed}/{total} ({pct}%)")
    return passed


if __name__ == "__main__":
    if not _correctness_guard():
        sys.exit(1)
    score()
