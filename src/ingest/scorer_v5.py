#!/usr/bin/env python3
"""Test coverage scorer for ingestion pipeline — auto-research v5.

Measures statement coverage of ``src/ingest/`` via ``pytest-cov``,
excluding scorer/tooling files that are not production code.

Score = coverage percentage (integer, 0–100).  Higher is better.

Correctness guard: all existing tests must still pass.
Naming convention guard: test functions containing "mock" must start
with ``test_mock_``.

Usage:
    uv run python src/ingest/scorer_v5.py [--target N]

    --target N   Minimum coverage % to pass (default: 92)
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # RagWeave/

# Files to exclude from coverage measurement (tooling, not production code)
_COVERAGE_OMIT = [
    "src/ingest/scorer.py",
    "src/ingest/scorer_v2.py",
    "src/ingest/scorer_v3.py",
    "src/ingest/scorer_v5.py",
]


def run_coverage() -> tuple[bool, int, str]:
    """Run pytest with coverage and return (tests_passed, coverage_pct, summary).

    Returns:
        Tuple of (all_tests_passed, coverage_percentage, summary_line).
        coverage_percentage is 0 if tests fail.
    """
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/ingest/",
        "--cov=src/ingest",
        "--cov-report=term-missing",
        "-q", "--tb=short",
    ]
    for omit in _COVERAGE_OMIT:
        cmd.append(f"--cov-config=/dev/null")
    # Use --cov-config with omit via command-line flags
    # pytest-cov supports --cov-report and omit via ini or pyproject;
    # simplest: pass omit patterns directly
    cmd_with_omit = [
        sys.executable, "-m", "pytest",
        "tests/ingest/",
        "--cov=src/ingest",
        "--cov-report=term-missing",
        "-q", "--tb=short",
    ]
    for omit_path in _COVERAGE_OMIT:
        cmd_with_omit.append(f"--cov-config=/dev/null")

    # Build the command with --omit flag for coverage
    final_cmd = [
        sys.executable, "-m", "pytest",
        "tests/ingest/",
        "--cov=src/ingest",
        "--cov-report=term-missing",
        "-q", "--tb=short",
    ]

    try:
        result = subprocess.run(
            final_cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(PROJECT_ROOT),
            env=_build_env(),
        )
    except subprocess.TimeoutExpired:
        return False, 0, "pytest timed out (300s)"
    except Exception as exc:
        return False, 0, f"pytest error: {exc}"

    stdout = result.stdout
    stderr = result.stderr

    tests_passed = result.returncode == 0

    # Extract summary line
    summary_lines = stdout.strip().splitlines()
    summary = ""
    for line in reversed(summary_lines):
        if "passed" in line or "failed" in line or "error" in line:
            summary = line.strip()
            break
    if not summary:
        summary = summary_lines[-1].strip() if summary_lines else stderr.strip()

    # Parse per-file coverage and recompute total excluding omitted files
    omit_basenames = {Path(p).name for p in _COVERAGE_OMIT}
    total_stmts = 0
    total_missed = 0

    for line in summary_lines:
        match = re.match(
            r'^(src/ingest/\S+)\s+(\d+)\s+(\d+)\s+(\d+)%', line
        )
        if match:
            filepath = match.group(1)
            filename = Path(filepath).name
            if filename in omit_basenames:
                continue
            stmts = int(match.group(2))
            missed = int(match.group(3))
            total_stmts += stmts
            total_missed += missed

    coverage_pct = (
        round((total_stmts - total_missed) / total_stmts * 100)
        if total_stmts > 0 else 0
    )

    return tests_passed, coverage_pct, summary


def _build_env():
    """Build environment for subprocess, inheriting current env."""
    import os
    return {**os.environ}


# Pre-existing naming violations (grandfathered — do not fail on these)
_BASELINE_VIOLATIONS = frozenset({
    "tests/ingest/test_docling_chunking_vlm_and_validation.py::test_external_mode_placeholder_replaced_via_mock",
    "tests/ingest/test_docling_chunking_store_and_chunker.py::test_read_docling_with_mocked_docling_core",
})


def check_mock_naming() -> tuple[bool, list[str]]:
    """Check that mocked tests follow test_mock_* naming convention."""
    test_dir = PROJECT_ROOT / "tests" / "ingest"
    violations: list[str] = []

    for test_file in test_dir.rglob("*.py"):
        if not test_file.name.startswith("test_"):
            continue
        try:
            content = test_file.read_text(encoding="utf-8")
        except Exception:
            continue

        for match in re.finditer(
            r'^\s*def (test_\w*mock\w*)\s*\(',
            content, re.MULTILINE | re.IGNORECASE,
        ):
            func_name = match.group(1)
            if not func_name.startswith("test_mock_"):
                rel_path = test_file.relative_to(PROJECT_ROOT)
                key = f"{rel_path}::{func_name}"
                if key not in _BASELINE_VIOLATIONS:
                    violations.append(key)

    return len(violations) == 0, violations


def main() -> None:
    # Parse --target flag
    target = 92
    args = sys.argv[1:]
    if "--target" in args:
        idx = args.index("--target")
        if idx + 1 < len(args):
            target = int(args[idx + 1])

    print(f"\n{'='*60}")
    print(f"  Ingestion Test Coverage Scorer v5")
    print(f"  Target: {target}%")
    print(f"  (excludes scorer_*.py tooling files)")
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
        sys.exit(0 if coverage_pct >= target else 1)


if __name__ == "__main__":
    main()
