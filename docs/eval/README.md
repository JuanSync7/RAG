<!-- @summary
Engineering documentation for the offline eval loop: pack format, runner stages, judge, gate, and CLI.
@end-summary -->

# eval

## Overview

This directory contains engineering documentation for the **offline eval loop** — a pack-driven pipeline that takes an `eval_pack` directory (corpus + goldens + thresholds + judge prompt) and runs `validate → plan → ingest → retrieve → judge → aggregate → gate`, exiting with a deterministic code.

End-to-end answer-faithfulness in production lives in `src/guardrails/` and is documented separately; the docs here cover the offline eval surface only.

## Documents

| Document | Purpose |
| --- | --- |
| `EVAL_LOOP_ENGINEERING_GUIDE.md` | Implementation-oriented walkthrough: architecture, stage contracts, pack schema, CLI flags, exit codes, gating, multi-sample judge semantics, extension steps, troubleshooting. |

## Key Starting Points

| Goal | Document |
| --- | --- |
| Understand the eval loop end-to-end | [`EVAL_LOOP_ENGINEERING_GUIDE.md`](EVAL_LOOP_ENGINEERING_GUIDE.md) |
| Author or modify a pack | [`EVAL_LOOP_ENGINEERING_GUIDE.md`](EVAL_LOOP_ENGINEERING_GUIDE.md#configuration-model) |
| Add a new metric or qtype | [`EVAL_LOOP_ENGINEERING_GUIDE.md`](EVAL_LOOP_ENGINEERING_GUIDE.md#extending-the-eval-loop) |

## Related Source

- Public pack API: [`src/eval/pack/`](../../src/eval/pack/)
- Runner: [`src/eval/runner/`](../../src/eval/runner/)
- CLI: [`src/eval/cli.py`](../../src/eval/cli.py)
- Reference pack: [`evals/packs/opentitan_riscv/`](../../evals/packs/opentitan_riscv/)
