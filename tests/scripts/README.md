<!-- @summary
Unit tests for the scripts/ safety utilities. Covers the check_pytest.py SafetyValidator across all detection categories (FR-102–FR-114): dangerous calls/imports, filesystem writes, process/signal manipulation, database access, network libraries, alias tracking, conftest exemptions, group overrides, strict mode, parse error handling, file discovery, and exit code behavior.
@end-summary -->

# tests/scripts/

Tests for the project's developer-safety CLI scripts. Currently focused on `scripts/check_pytest.py`, the AST-based validator that blocks unsafe patterns from appearing in test files before they are executed.

## Contents

| Path | Purpose |
| --- | --- |
| `test_check_pytest.py` | Unit tests for `SafetyValidator` and `main()` — covers all FR-102–FR-114 acceptance criteria |
