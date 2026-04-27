<!-- @summary
Auto-research run artifacts for ingest pipeline cleanup (v2).
Ran 4 iterations against a focused 8-criteria scorer, reaching 8/8 by removing dead code and eliminating a double file read.
@end-summary -->

# research-v2/ — Ingest Cleanup Auto-Research (v2)

Artifacts from the second auto-research run targeting deferred cleanup items from the v1 quality run.
The run scored against 8 criteria focused on dead-code removal and I/O efficiency.
All 3 hypotheses were confirmed; the run reached a perfect 8/8.

## Contents

| Path | Purpose |
| --- | --- |
| `changelog.md` | Narrative log of all 4 iterations, changes made, and assessed-but-rejected items |
| `iterations.tsv` | Per-iteration record: commit, hypothesis, score, status, summary |
