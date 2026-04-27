#!/usr/bin/env python3
"""Test coverage scorer for ingestion pipeline — auto-research v3.

Measures statement coverage of ``src/ingest/`` via ``pytest-cov``.
Score = coverage percentage (integer, 0–100).  Higher is better.

Correctness guard: all existing tests must still pass.  If any test
fails, the iteration is recorded as a crash regardless of coverage.

Naming convention guard: any test function whose name contains "mock"
must also start with ``test_mock_``.  This catches tests that use mocks
but don't follow the naming convention.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # RagWeave/


def run_coverage() -> tuple[bool, int, str]:
    """Run pytest with coverage and return (tests_passed, coverage_pct, summary).

    Returns:
        Tuple of (all_tests_passed, coverage_percentage, summary_line).
        coverage_percentage is 0 if tests fail.
    """
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                "tests/ingest/",
                f"--cov=src/ingest",
                "--cov-report=term-missing",
                "-q", "--tb=short",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        return False, 0, "pytest timed out (300s)"
    except Exception as exc:
        return False, 0, f"pytest error: {exc}"

    stdout = result.stdout
    stderr = result.stderr

    tests_passed = result.returncode == 0

    # Extract summary line (e.g. "1121 passed, 5 skipped")
    summary_lines = stdout.strip().splitlines()
    summary = ""
    for line in reversed(summary_lines):
        if "passed" in line or "failed" in line or "error" in line:
            summary = line.strip()
            break
    if not summary:
        summary = summary_lines[-1].strip() if summary_lines else stderr.strip()

    # Extract TOTAL coverage percentage from the coverage report
    # Format: TOTAL   4785   1855    61%
    coverage_pct = 0
    for line in summary_lines:
        match = re.match(r'^TOTAL\s+\d+\s+\d+\s+(\d+)%', line)
        if match:
            coverage_pct = int(match.group(1))
            break

    return tests_passed, coverage_pct, summary


# Pre-existing naming violations (grandfathered — do not fail on these)
_BASELINE_VIOLATIONS = frozenset({
    "tests/ingest/test_docling_chunking_vlm_and_validation.py::test_external_mode_placeholder_replaced_via_mock",
    "tests/ingest/test_docling_chunking_store_and_chunker.py::test_read_docling_with_mocked_docling_core",
})


def check_mock_naming() -> tuple[bool, list[str]]:
    """Check that mocked tests follow test_mock_* naming convention.

    Scans test files for functions that contain 'mock' in the name but
    don't start with 'test_mock_'.  Pre-existing violations are excluded.

    Returns:
        Tuple of (all_ok, list_of_new_violations).
    """
    test_dir = PROJECT_ROOT / "tests" / "ingest"
    violations: list[str] = []

    for test_file in test_dir.rglob("*.py"):
        if not test_file.name.startswith("test_"):
            continue
        try:
            content = test_file.read_text(encoding="utf-8")
        except Exception:
            continue

        for match in re.finditer(r'^\s*def (test_\w*mock\w*)\s*\(', content, re.MULTILINE | re.IGNORECASE):
            func_name = match.group(1)
            if not func_name.startswith("test_mock_"):
                rel_path = test_file.relative_to(PROJECT_ROOT)
                key = f"{rel_path}::{func_name}"
                if key not in _BASELINE_VIOLATIONS:
                    violations.append(key)

    return len(violations) == 0, violations


def main() -> None:
    print(f"\n{'='*60}")
    print(f"  Ingestion Test Coverage Scorer v3")
    print(f"{'='*60}\n")

    # 1. Run coverage
    print("  Running pytest with coverage...")
    tests_passed, coverage_pct, summary = run_coverage()

    print(f"  Tests: {'PASS' if tests_passed else 'FAIL'} — {summary}")
    print(f"  Coverage: {coverage_pct}%")

    # 2. Check mock naming convention
    print(f"\n  Checking mock test naming convention...")
    naming_ok, violations = check_mock_naming()
    if naming_ok:
        print(f"  Naming: PASS — all mock tests follow test_mock_* convention")
    else:
        print(f"  Naming: FAIL — {len(violations)} violation(s):")
        for v in violations:
            print(f"    - {v}")

    # 3. Final verdict
    print(f"\n{'='*60}")
    if not tests_passed:
        print(f"  FINAL: 0% (tests failed — iteration is a crash)")
        print(f"{'='*60}\n")
        sys.exit(1)
    elif not naming_ok:
        print(f"  FINAL: 0% (naming convention violated — iteration is a crash)")
        print(f"{'='*60}\n")
        sys.exit(1)
    else:
        print(f"  FINAL: {coverage_pct}%")
        print(f"{'='*60}\n")
        sys.exit(0 if coverage_pct >= 75 else 1)


if __name__ == "__main__":
    main()
