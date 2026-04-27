# Brainstorm Notes — Test Coverage Strategy

## Status

- **Phase:** DONE
- **Type:** planning
- **Anticipated shape:** Plan (coverage spec) + Skill (autonomous test engine)
- **Turn count:** 5

## Threads (indexed)

### T1: Testing pyramid layers and ordering
**Status:** resolved
**Depends on:** —
**Lenses applied:** Stakeholder ✓ | Alternative ✓

Established 6-layer pyramid: unit → config validation → contract/schema → idempotency/incremental → mocked integration → real integration. Order is bottom-up, cheap-to-expensive. User confirmed this structure.

### T2: Mock coverage boundaries (~92-95%)
**Status:** resolved
**Depends on:** —
**Lenses applied:** Stakeholder ✓ | Alternative ✓

Mocks cover logic exhaustively. Gaps: contract fidelity (serialization unknowns), DB state/constraints, real latency/resource behavior, connection wiring. User agreed mocks get ~92-95% and the gap is narrow but real.

### T3: Dead code detection as pre-coverage step
**Status:** resolved
**Depends on:** T2
**Lenses applied:** Stakeholder ✓ | Alternative ✓

Use `vulture` to identify unreachable code before measuring coverage. Cross-reference: vulture-flagged + cov-uncovered = delete; not-flagged + cov-uncovered = real gap. Added to plan.

### T4: Config validation — cross-field rules
**Status:** resolved
**Depends on:** T1
**Lenses applied:** Stakeholder ✓ | Alternative ✓

Per-field range checks insufficient. Cross-field validation catches contradictory combos (overlap >= chunk_size, enabled feature with empty required config). Use pydantic model_validator. Added to plan.

### T5: Test data strategy (fixture quality)
**Status:** resolved
**Depends on:** T1
**Lenses applied:** Stakeholder ✓ | Alternative ✓

Three tiers: inline (pure logic), factory fixtures (most tests), fixture files (parser/ingest). Hypothesis for fuzz. vcrpy for API record/replay. Added to plan.

### T6: Idempotency/incremental scoping rule
**Status:** resolved
**Depends on:** T1
**Lenses applied:** Stakeholder ✓ | Alternative ✓

Generic rule: any module that processes external input AND persists state needs idempotency tests. Added to plan.

### T7: Mutation testing for weak test detection
**Status:** resolved
**Depends on:** T2
**Lenses applied:** Stakeholder ✓ | Alternative ✓

mutmut on changed files per PR. ~15-30 min for 1k-line PR. Complements (not replaces) LLM review. Added to plan.

### T8: Linters as pre-test layer
**Status:** resolved
**Depends on:** —
**Lenses applied:** Stakeholder ✓ | Alternative ✓

ruff + mypy pre-commit, bandit in CI. mypy strict partially replaces contract tests. Run before tests. Added to plan.

### T9: Outside-pytest coverage umbrella
**Status:** resolved
**Depends on:** T1
**Lenses applied:** Stakeholder ✓ | Alternative ✓

Load/stress (locust/k6), CI/CD pipeline validation, ops monitoring config checks. Each has its own runner, not pytest. Added to plan.

### T10: Coverage enforcement and tooling
**Status:** resolved
**Depends on:** T1, T2
**Lenses applied:** Stakeholder ✓ | Alternative ✓

Per-directory cov-fail-under thresholds. term-missing for line gaps. JSON + AST script for function-level gaps. Pytest markers for mock-vs-integration diff. Added to plan.

### T11: Defensive code testing policy
**Status:** resolved
**Depends on:** T3
**Lenses applied:** Stakeholder ✓ | Alternative ✓

Defensive code ≠ dead code. Vulture catches dead code; defensive code is syntactically reachable but rare-path. Policy: test defensive code at system boundaries (user input, API input, deserialized data). Internal defensive branches covered by mypy --strict instead. Use `# pragma: no cover` on deliberately untested internal defensive branches to keep coverage honest.

### T12: Cross-field config dependency mapping
**Status:** resolved
**Depends on:** T4
**Lenses applied:** Stakeholder ✓ | Alternative ✓

Maintain a constraint table alongside config schema. Each row = one cross-field rule = one test case. When adding a config field, ask "does this interact with existing fields?" If yes, add row. Can extract dependency graph from model_validator source automatically, but manual table is simpler and doubles as documentation.

### T13: Fixture coverage — input type inventory
**Status:** resolved
**Depends on:** T5
**Lenses applied:** Stakeholder ✓ | Alternative ✓

No tool computes fixture completeness automatically. Use a manual inventory table: rows = input types, columns = edge cases (happy, empty, malformed, large, unicode). Gaps in table = gaps in fixture coverage. Inventory IS the spec. Incomplete spec is inherent (unknown unknowns) — discovered in production, added as regression fixtures.

### T14: AST parsing — stdlib vs tree-sitter
**Status:** resolved
**Depends on:** T10
**Lenses applied:** Stakeholder ✓ | Alternative ✓

Python stdlib `ast` module is sufficient for mapping coverage gaps to function names. Tree-sitter only needed for cross-language or incremental parsing. No extra dependency.

### T15: Dependencies/libs per layer
**Status:** resolved
**Depends on:** T1
**Lenses applied:** Stakeholder ✓ | Alternative ✓

Full dependency list catalogued per layer. Core: pytest, pytest-cov, pytest-mock, pytest-timeout, pydantic, ruff, mypy, bandit, vulture, mutmut. Per use case: pytest-asyncio, respx/responses, testcontainers, vcrpy, hypothesis, deepdiff, locust, httpx.

### T16: Marker registry (full set)
**Status:** resolved
**Depends on:** T10
**Lenses applied:** Stakeholder ✓ | Alternative ✓

9 markers defined: unit, mock, integration, slow, idempotency, contract, config, regression, smoke. Enables selective CI runs and mock-vs-real diff.

### T17: Regression cycle automation
**Status:** resolved
**Depends on:** T5, T13
**Lenses applied:** Stakeholder ✓ | Alternative ✓

Cycle: bug found → add fixture + regression test (@pytest.mark.regression) → update INVENTORY.csv. Enforcement: cov-fail-under prevents test removal, CI checks inventory sync, regression marker ensures tests always run. No manual step beyond writing the test.

### T18: Contract types beyond DB + cross-module
**Status:** resolved
**Depends on:** T1
**Lenses applied:** Stakeholder ✓ | Alternative ✓

6 contract types: DB constraints, cross-module schemas, API request/response, file format contracts, message queue contracts, CLI contracts. Generic rule: anywhere two components exchange data through a defined shape = contract to test.

### T19: Fixture inventory consistency enforcement
**Status:** resolved
**Depends on:** T13
**Lenses applied:** Stakeholder ✓ | Alternative ✓

CI test cross-references fixture files on disk against INVENTORY.csv. Unlisted files → CI fails. Empty cells without N/A → CI warns. Pre-commit hook keeps inventory in sync.

### T20: Plan completeness — first circle-back
**Status:** resolved
**Depends on:** T1-T19
**Lenses applied:** Stakeholder ✓ | Alternative ✓

First sweep completed. No structural gaps in coverage spec. Led to new discussion about skill implementation.

### T21: Skill architecture — autonomous test coverage engine
**Status:** resolved
**Depends on:** T1-T20
**Lenses applied:** Stakeholder ✓ | Alternative ✓

Coverage plan should be implemented as a skill (not just a document) — an autonomous engine that iterates toward full coverage. Architecture: orchestrator + one agent per layer (kept separate for composability — can call any agent alone for a small task). 7 agents: Phase 0 (linters+vulture), Layer 1-2 (unit+config), Layer 3-4 (contract+idempotency), Layer 5 (mocked integration), Layer 6 (real integration), Quality (mutation+coverage diff), Orchestrator (dispatch+aggregate+track). Modes: audit (report gaps), generate (write tests), full (audit+generate+validate loop).

### T22: Mutation testing cost — per-gap vs per-PR vs nightly
**Status:** resolved
**Depends on:** T7, T21
**Lenses applied:** Stakeholder ✓ | Alternative ✓

Per-gap mutation (in auto-regression loop): seconds, only mutates lines just covered (~5-20 lines). Per-PR: skip — per-gap already validated new tests. Nightly/weekly: full-project mutation for pre-existing weak test detection. No redundancy between tiers.

### T23: Test generation unit — per-function, not per-line
**Status:** resolved
**Depends on:** T21, T22
**Lenses applied:** Stakeholder ✓ | Alternative ✓

Per-line test generation produces fragmented, low-quality tests. Correct unit: per-function/per-branch. Agent workflow: (1) find uncovered functions via AST mapping, (2) identify uncovered branches within function, (3) generate test exercising function through that branch, (4) line coverage follows naturally. Mutation testing validates at function level. For functions with existing tests but uncovered branches: agent adds test cases to the existing test file, not a new file. Reads existing tests to understand what's covered, fills gaps.

### T24: Tracking state between runs
**Status:** resolved
**Depends on:** T21
**Lenses applied:** Stakeholder ✓ | Alternative ✓

Skill needs persistent state: what gaps exist, what was auto-generated vs manual, what's been validated. Tracking file (e.g., `tests/COVERAGE_STATE.yaml`) set up from first run and updated each iteration. Enables incremental runs — don't re-analyze what's already closed.

### T25: Auto-regression stopping condition
**Status:** resolved
**Depends on:** T21, T22, T23
**Lenses applied:** Stakeholder ✓ | Alternative ✓

Loop: find uncovered function → generate test → run cov → run per-function mutation → both pass = gap closed. Stopping: all functions covered + mutation kill rate meets threshold. Circuit breaker: if mutation kill rate doesn't improve after N iterations on same function, flag for human review (some code is genuinely hard to test meaningfully). Prevents infinite loops and meaningless test generation.

## Connections

- T3 (dead code) feeds into T2 (mock boundaries) — dead code inflates coverage gap perception
- T4 (cross-field config) is a refinement of T1 pyramid layer 2
- T5 (test data) affects quality of ALL pyramid layers — cross-cutting concern
- T7 (mutation) addresses the gap T2 identified (execution ≠ verification)
- T8 (linters) + T3 (vulture) form the "pre-test" phase before the pyramid

## Resolution Log

- T1 resolved (prior conversation) — user confirmed pyramid structure
- T2 resolved (prior conversation) — agreed on ~92-95% mock coverage
- T3 resolved (current session) — vulture added to plan
- T4 resolved (current session) — cross-field validation added
- T5 resolved (current session) — fixture tiers added
- T6 resolved (current session) — scoping rule added
- T7 resolved (prior conversation) — mutmut on PR diffs
- T8 resolved (prior conversation) — ruff + mypy + bandit
- T9 resolved (prior conversation) — outside-pytest umbrella defined
- T10 resolved (prior conversation) — tooling approach agreed
- T11 resolved (turn 2) — defensive code policy: test at boundaries, mypy for internal
- T12 resolved (turn 2) — constraint table workflow for cross-field config
- T13 resolved (turn 2) — fixture inventory table as spec
- T14 resolved (turn 3) — stdlib ast, not tree-sitter
- T15 resolved (turn 3) — full dependency list per layer
- T16 resolved (turn 3) — 9 pytest markers defined
- T17 resolved (turn 3) — regression cycle automated via cov threshold + inventory
- T18 resolved (turn 3) — 6 contract types beyond DB
- T19 resolved (turn 3) — fixture inventory CI enforcement
- T20 resolved (turn 4) — first circle-back on coverage spec
- T21 resolved (turn 5) — skill architecture: orchestrator + 7 agents, 3 modes
- T22 resolved (turn 5) — mutation tiers: per-gap (seconds), skip per-PR, nightly (full)
- T23 resolved (turn 5) — test generation at function/branch level, add to existing test files
- T24 resolved (turn 5) — persistent COVERAGE_STATE.yaml for tracking between runs
- T25 resolved (turn 5) — stopping condition: cov + mutation kill rate + circuit breaker

## Key Insights

- Coverage measures execution, not verification — mutation testing bridges that gap
- Mocks test your assumptions about the world, real tests check reality
- The "unknown unknowns" problem (serialization shapes you didn't imagine) has no clean test solution — record/replay is the best mitigation
- "Full coverage" is actually three different metrics: code path, behavior, integration
- Defensive code ≠ dead code — different detection tools, different testing policies
- Fixture coverage is an inventory problem, not a percentage problem — the table IS the spec
- Cross-field config rules are discoverable from the schema itself — constraint table as documentation
- Test generation unit is function/branch, not line — lines are measurement, functions are design
- Per-gap mutation testing is cheap (seconds) — full-project is expensive but only needed nightly
- Auto-generated tests should augment existing test files, not create parallel ones
- Coverage plan is the spec; skill is the implementation that executes the spec autonomously

## Tensions

- 100% line coverage vs. meaningful coverage — dead code and defensive branches inflate the denominator
- Test speed vs. test confidence — real integration tests are slow but irreplaceable for boundary concerns
- Generic strategy vs. project-specific scoping — idempotency rule needs per-project application

## Discarded Candidates (Moves, Framings, Options)

- Testing ops monitoring behavior at runtime — out of scope for app tests, belongs to ops runbooks
- ESLint — not applicable to Python-only stack
- Full permutation of flow tests — too expensive, unit tests handle branch permutations instead
- Per-line test generation — produces fragmented, meaningless tests. Per-function/branch is correct unit
- Per-PR mutation testing — redundant when per-gap mutation already validates in the loop
- Tree-sitter for AST parsing — overkill, stdlib `ast` sufficient for Python-only

## Phase A Misses (if any)

(none identified — concerns surfaced organically through conversation)
