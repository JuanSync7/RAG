# Brainstorm Notes — test-coverage-engine skill

## Status
Phase: A
Outcome: pending

## Resolved
- Prior brainstorm established the full coverage spec (25 threads resolved)
- Architecture decided: orchestrator + 7 composable agents (1 per layer)
- Modes decided: audit, generate, full (auto-regression loop)
- Test generation unit: per-function/branch, not per-line
- Mutation tiers: per-gap (seconds, in-loop), nightly (full-project)
- Persistent state: COVERAGE_STATE.yaml
- Stopping condition: cov + mutation kill rate + circuit breaker

## Open Threads
- Is this skill-worthy? (Phase A gate — baseline failure, reusability, injection shape)
- What does Claude currently produce without this skill when asked to "write tests for module X"?
- How much is domain knowledge vs workflow orchestration?
- Where are the companion file boundaries (always-on vs on-demand)?
- Scope boundaries with other skills (e.g., does this overlap with /brainstorm, /autonomous-orchestrator?)

## Key Insights
(from prior brainstorm)
- Coverage plan is the spec; skill is the implementation
- The plan has 6 layers, each with different tools/deps/success criteria — too much for one agent
- Composability matters: user wants to call individual layer agents for small tasks

## Tensions
- Skill complexity vs token budget — this is a large skill with many agents
- Autonomy vs quality — auto-generating tests risks meaningless coverage padding
- Generic (any Python project) vs project-specific (RagWeave conventions)

## Discarded
(none yet)

## Lens Notes
### Usability
### Robustness
### Maintenance
### Preciseness
### Boundary

## Companion Files Anticipated
### references/ — domain knowledge to load on-demand
### templates/ — output format skeletons
### rules/ — hard constraints
### examples/ — worked examples of good output
