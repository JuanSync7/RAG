# Project Navigation Protocol

This project uses **context-agent** for hierarchical context management.
Every source file has an `@summary` block at the top, and every directory has a `README.md`.

## How to Navigate

1. **Start here**: Read the root `README.md` for the project overview
2. **Find your target**: Use the directory map to identify which directory is relevant
3. **Read the directory README**: Each directory's `README.md` explains its contents and file relationships
4. **Check file summaries**: Each source file has an `@summary` comment block at the top — read it before diving into the full source
5. **Load source**: Only then read the full source files you need — and consider loading only the relevant sections

## Context Locations

| Context | Location | Purpose |
|---------|----------|---------|
| Project overview | `README.md` (root) | What this project does, architecture, directory map |
| Directory overview | `<dir>/README.md` | What a directory contains, file relationships |
| File summary | `@summary` block at top of each file | What a file does, key exports, dependencies |
| State ledger | `.context/state.yaml` | File hashes and timestamps for change tracking |
| Config | `.context/config.yaml` | Project-level context-agent settings |

## Summary Formats

### Source files — `@summary` block
Each source file has a comment block at the top:
```python
# @summary
# Brief description of what this file does.
# Exports: ClassName, function_name
# Deps: external_lib, internal.module
# @end-summary
```

### README.md — `<!-- @summary -->` block
Each directory's README.md starts with a promotable summary:
```markdown
<!-- @summary
Brief description of this directory's purpose.
@end-summary -->

# Directory Name
...
```
The `@summary` from each README.md is "promoted" into its parent directory's README.md,
creating a bottom-up chain of context from leaf directories to the project root.

## After Making Changes

When you modify source files:
1. The `@summary` blocks may become stale — this is expected
2. Run `context-agent update` to refresh affected summaries and README.md files
3. If you add a new directory, `context-agent update` will auto-generate its README.md
