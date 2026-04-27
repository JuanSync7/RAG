#!/usr/bin/env python3
"""Static-analysis scorer for ingestion pipeline cleanup — remaining gaps.

Checks 8 criteria across two dimensions:
  - Speed (3): eliminate double file read in ingest_directory
  - Code quality (5): remove dead enriched_chunks field, remove legacy IngestState

Score = passed_criteria / total_criteria.  Higher is better.

Correctness guard: runs ``pytest tests/ingest/ -x -q --tb=short`` after
the static checks.  If any test fails, the iteration is a crash regardless
of the static score.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent  # src/ingest/
PROJECT_ROOT = ROOT.parent.parent  # RagWeave/

# ── Target files ──────────────────────────────────────────────────────
IMPL_FILE = ROOT / "impl.py"
DOC_INGESTION = ROOT / "doc_processing/nodes/document_ingestion.py"
DOC_PROC_STATE = ROOT / "doc_processing/state.py"
DOC_PROC_IMPL = ROOT / "doc_processing/impl.py"
EMBEDDING_STATE = ROOT / "embedding/state.py"
EMBEDDING_IMPL = ROOT / "embedding/impl.py"
TYPES_FILE = ROOT / "common/types.py"
SHARED_FILE = ROOT / "common/shared.py"
INIT_FILE = ROOT / "common/__init__.py"
VISUAL_EMBEDDING = ROOT / "embedding/nodes/visual_embedding.py"


class CheckResult(NamedTuple):
    name: str
    passed: bool
    detail: str


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════
# Speed: eliminate double file read
# ══════════════════════════════════════════════════════════════════════

def check_no_sha256_path_in_ingest_directory() -> CheckResult:
    """ingest_directory must NOT call sha256_path — should use sha256_bytes on pre-read bytes."""
    src = _read(IMPL_FILE)
    # Find the ingest_directory function body
    match = re.search(r'^def ingest_directory\b', src, re.MULTILINE)
    if not match:
        return CheckResult("double_read:no_sha256_path", False, "ingest_directory not found")
    body_start = match.start()
    # Find end of function (next top-level def or class, or EOF)
    next_def = re.search(r'^(?:def |class )\w', src[body_start + 1:], re.MULTILINE)
    body = src[body_start:body_start + 1 + next_def.start()] if next_def else src[body_start:]
    has_sha256_path = bool(re.search(r'\bsha256_path\s*\(', body))
    if has_sha256_path:
        return CheckResult("double_read:no_sha256_path", False,
                           "ingest_directory still calls sha256_path — double read")
    has_sha256_bytes = bool(re.search(r'\bsha256_bytes\s*\(', body))
    if not has_sha256_bytes:
        return CheckResult("double_read:no_sha256_path", False,
                           "ingest_directory has no sha256_bytes call either")
    return CheckResult("double_read:no_sha256_path", True,
                       "ingest_directory uses sha256_bytes — single read path")


def check_doc_ingestion_uses_preread_bytes() -> CheckResult:
    """document_ingestion_node checks for pre-read raw_bytes in state before disk read."""
    src = _read(DOC_INGESTION)
    # Must check state for raw_bytes (e.g. state.get("raw_bytes") or state["raw_bytes"])
    checks_raw_bytes = bool(re.search(r'state\s*[\.\[]\s*(?:get\s*\(\s*)?["\']raw_bytes["\']', src))
    if not checks_raw_bytes:
        return CheckResult("double_read:preread_bytes", False,
                           "document_ingestion_node does not check for pre-read raw_bytes in state")
    return CheckResult("double_read:preread_bytes", True,
                       "document_ingestion_node uses pre-read raw_bytes when available")


def check_raw_bytes_in_state() -> CheckResult:
    """DocumentProcessingState declares raw_bytes field."""
    src = _read(DOC_PROC_STATE)
    has_raw_bytes = bool(re.search(r'raw_bytes\s*:', src))
    if not has_raw_bytes:
        return CheckResult("double_read:state_field", False,
                           "DocumentProcessingState missing raw_bytes field")
    return CheckResult("double_read:state_field", True,
                       "DocumentProcessingState has raw_bytes field")


# ══════════════════════════════════════════════════════════════════════
# Code quality: remove dead enriched_chunks field
# ══════════════════════════════════════════════════════════════════════

def check_no_enriched_chunks_in_state() -> CheckResult:
    """enriched_chunks must not be declared in EmbeddingPipelineState."""
    src = _read(EMBEDDING_STATE)
    has_field = bool(re.search(r'enriched_chunks\s*:', src))
    if has_field:
        return CheckResult("dead_field:state_decl", False,
                           "enriched_chunks still declared in EmbeddingPipelineState")
    return CheckResult("dead_field:state_decl", True,
                       "enriched_chunks removed from EmbeddingPipelineState")


def check_no_enriched_chunks_in_init() -> CheckResult:
    """enriched_chunks must not be initialized in embedding/impl.py."""
    src = _read(EMBEDDING_IMPL)
    has_init = bool(re.search(r'["\']enriched_chunks["\']', src))
    if has_init:
        return CheckResult("dead_field:init_removed", False,
                           "enriched_chunks still initialized in embedding/impl.py")
    return CheckResult("dead_field:init_removed", True,
                       "enriched_chunks initialization removed from embedding/impl.py")


# ══════════════════════════════════════════════════════════════════════
# Code quality: remove legacy IngestState TypedDict
# ══════════════════════════════════════════════════════════════════════

def check_no_ingest_state_class() -> CheckResult:
    """IngestState class must not exist in types.py."""
    src = _read(TYPES_FILE)
    has_class = bool(re.search(r'^class IngestState\b', src, re.MULTILINE))
    if has_class:
        return CheckResult("legacy_type:class_removed", False,
                           "IngestState class still defined in types.py")
    return CheckResult("legacy_type:class_removed", True,
                       "IngestState class removed from types.py")


def check_no_ingest_state_export() -> CheckResult:
    """IngestState must not be exported from common/__init__.py."""
    src = _read(INIT_FILE)
    has_export = bool(re.search(r'\bIngestState\b', src))
    if has_export:
        return CheckResult("legacy_type:export_removed", False,
                           "IngestState still exported in common/__init__.py")
    return CheckResult("legacy_type:export_removed", True,
                       "IngestState removed from common/__init__.py exports")


def check_append_log_type_hint() -> CheckResult:
    """append_processing_log must not use IngestState type hint."""
    src = _read(SHARED_FILE)
    fn_match = re.search(r'def append_processing_log\([^)]*\)', src)
    if not fn_match:
        return CheckResult("legacy_type:type_hint", False,
                           "append_processing_log function not found")
    signature = fn_match.group(0)
    if "IngestState" in signature:
        return CheckResult("legacy_type:type_hint", False,
                           "append_processing_log still uses IngestState type hint")
    return CheckResult("legacy_type:type_hint", True,
                       "append_processing_log uses updated type hint")


# ══════════════════════════════════════════════════════════════════════
# Main scorer
# ══════════════════════════════════════════════════════════════════════

def run_all_checks() -> list[CheckResult]:
    results: list[CheckResult] = []

    # Speed: double file read elimination (3 criteria)
    results.append(check_no_sha256_path_in_ingest_directory())
    results.append(check_doc_ingestion_uses_preread_bytes())
    results.append(check_raw_bytes_in_state())

    # Code quality: dead enriched_chunks (2 criteria)
    results.append(check_no_enriched_chunks_in_state())
    results.append(check_no_enriched_chunks_in_init())

    # Code quality: legacy IngestState (3 criteria)
    results.append(check_no_ingest_state_class())
    results.append(check_no_ingest_state_export())
    results.append(check_append_log_type_hint())

    return results


def run_correctness_guard() -> tuple[bool, str]:
    """Run pytest to verify no regressions."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/ingest/", "-x", "-q", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(PROJECT_ROOT),
        )
        passed = result.returncode == 0
        summary = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else result.stderr.strip()
        return passed, summary
    except subprocess.TimeoutExpired:
        return False, "pytest timed out (300s)"
    except Exception as exc:
        return False, f"pytest error: {exc}"


def main() -> None:
    results = run_all_checks()

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print(f"\n{'='*60}")
    print(f"  Ingestion Pipeline Cleanup Scorer v2")
    print(f"{'='*60}\n")

    # Group by category
    categories = {
        "Speed (double read)": [r for r in results if r.name.startswith("double_read:")],
        "Code Quality (dead field)": [r for r in results if r.name.startswith("dead_field:")],
        "Code Quality (legacy type)": [r for r in results if r.name.startswith("legacy_type:")],
    }

    for cat_name, cat_results in categories.items():
        cat_passed = sum(1 for r in cat_results if r.passed)
        print(f"  {cat_name}: {cat_passed}/{len(cat_results)}")
        for r in cat_results:
            icon = "PASS" if r.passed else "FAIL"
            print(f"    [{icon}] {r.name}: {r.detail}")
        print()

    print(f"  Static score: {passed}/{total}")

    # Correctness guard
    print(f"\n  Running correctness guard (pytest)...")
    guard_ok, guard_summary = run_correctness_guard()
    guard_status = "pass" if guard_ok else "FAIL"
    print(f"  Guard: {guard_status} — {guard_summary}")

    print(f"\n{'='*60}")
    if guard_ok:
        print(f"  FINAL: {passed}/{total}, guard={guard_status}")
    else:
        print(f"  FINAL: 0/{total} (guard failed — iteration is a crash)")
    print(f"{'='*60}\n")

    sys.exit(0 if (guard_ok and passed == total) else 1)


if __name__ == "__main__":
    main()
