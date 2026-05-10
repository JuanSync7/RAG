#!/usr/bin/env python3
"""
Error-handling coverage scorer for src/knowledge_graph.

Checks that each identified high-risk hot-path function contains at least
one try/except block.

Score = guarded_count / total_hot_paths (printed as "SCORE: N/8 (X%)")

Correctness guard: verifies the package is importable before scoring.
If the guard fails the script exits with code 1 and prints "GUARD FAILED:".

Usage (from the RagWeave project root):
    python src/knowledge_graph/score_error_handling.py
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Hot-path registry — (relative_file, ClassName, method_name)
# ---------------------------------------------------------------------------
KG_ROOT = Path(__file__).parent

HOT_PATHS: list[tuple[str, str, str]] = [
    # A — Query path (called on every user request)
    ("query/expander.py",                "GraphQueryExpander",    "expand"),
    ("query/entity_matcher.py",          "EntityMatcher",         "_match_spacy"),
    ("query/entity_matcher.py",          "EntityMatcher",         "_llm_match"),
    # B — Backend I/O (disk / connection failure risk)
    ("backends/networkx_backend.py",     "NetworkXBackend",       "save"),
    ("backends/networkx_backend.py",     "NetworkXBackend",       "load"),
    # C — Inference / model operations (LLM + numpy)
    ("community/summarizer.py",          "CommunitySummarizer",   "_call_llm"),
    ("resolution/embedding_resolver.py", "EmbeddingResolver",     "find_candidates"),
    # D — Community graph construction
    ("community/detector.py",            "CommunityDetector",     "_to_igraph"),
]


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _has_try_except(func_node: ast.FunctionDef) -> bool:
    """Return True if the function body contains at least one Try node."""
    for node in ast.walk(func_node):
        if isinstance(node, ast.Try):
            return True
    return False


def _find_method(
    tree: ast.Module, class_name: str, method_name: str
) -> ast.FunctionDef | None:
    """Return the FunctionDef for ClassName.method_name, or None."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if (
                    isinstance(item, ast.FunctionDef)
                    and item.name == method_name
                ):
                    return item
    return None


# ---------------------------------------------------------------------------
# Correctness guard
# ---------------------------------------------------------------------------

def _run_guard() -> None:
    """Verify the package facade is importable. Exits 1 on failure."""
    project_root = KG_ROOT.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    spec = importlib.util.find_spec("src.knowledge_graph")
    if spec is None:
        print("GUARD FAILED: src.knowledge_graph is not importable from project root")
        sys.exit(1)

    # Smoke-import: just load the module, don't call any side-effecting code
    try:
        import importlib as _il
        _il.import_module("src.knowledge_graph.backend")
        _il.import_module("src.knowledge_graph.common")
    except Exception as exc:
        print(f"GUARD FAILED: import error — {exc}")
        sys.exit(1)

    print("GUARD OK")


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

def main() -> int:
    _run_guard()

    guarded = 0
    total = len(HOT_PATHS)
    rows: list[tuple[str, str, str, bool, str]] = []

    for rel_file, cls, method in HOT_PATHS:
        filepath = KG_ROOT / rel_file

        if not filepath.exists():
            rows.append((rel_file, cls, method, False, "MISSING_FILE"))
            continue

        source = filepath.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            rows.append((rel_file, cls, method, False, f"SYNTAX_ERROR:{exc}"))
            continue

        func_node = _find_method(tree, cls, method)
        if func_node is None:
            rows.append((rel_file, cls, method, False, "METHOD_NOT_FOUND"))
            continue

        ok = _has_try_except(func_node)
        if ok:
            guarded += 1
        rows.append((rel_file, cls, method, ok, "OK"))

    # Print per-path results
    print()
    print("Hot-path error-handling coverage:")
    for rel_file, cls, method, ok, status in rows:
        mark = "+" if ok else "-"
        note = f" [{status}]" if status != "OK" else ""
        print(f"  [{mark}] {cls}.{method}  ({rel_file}){note}")

    pct = 100 * guarded // total if total else 0
    print(f"\nSCORE: {guarded}/{total} ({pct}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
