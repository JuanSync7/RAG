# Eval Loop Engineering Guide

## Audience and Goal

This guide is for engineers who need to:

- understand how the offline eval loop works end-to-end,
- author or modify an eval pack (corpus + goldens + thresholds + judge prompt),
- safely change a runner stage (retrieve, judge, gate),
- troubleshoot pack-validation, ingest, retrieval, or gating failures,
- extend the loop with a new qtype, a new metric, or a new judge model.

It focuses on the implemented code, not the product spec, and reflects what was shipped through slices P0.5 → P6c. End-to-end answer-faithfulness (production guardrails) lives in `src/guardrails/` and is **not** the same subsystem — this guide covers the offline pack-driven eval loop only.

## System at a Glance

The eval loop takes a self-contained **eval pack** (a directory with `pack.yaml`, `corpus/`, `goldens/`, `thresholds.yaml`, and optional `prompts/`) and runs the full retrieval-quality pipeline against it: validate → plan → ingest → retrieve → judge → aggregate → gate. The output is a structured `EvalReport`, a `GateResult` (pass/fail with itemized floor breaches), and a deterministic process exit code.

Top-level entrypoints:

- CLI module entry: `python -m src.eval` — see [`src/eval/__main__.py`](../../src/eval/__main__.py).
- CLI orchestrator: [`src/eval/cli.py`](../../src/eval/cli.py) (`run_cli`).
- Pack format public API: [`src/eval/pack/__init__.py`](../../src/eval/pack/__init__.py) (`load_pack`, `validate_pack`, `EvalPack`).
- Runner public API: [`src/eval/runner/__init__.py`](../../src/eval/runner/__init__.py) (`plan_pack_ingest`, `execute_plan`, `retrieve_for_goldens`, `score_goldens`, `build_eval_report`, `validate_eval_report`).
- Reference pack: [`evals/packs/opentitan_riscv/`](../../evals/packs/opentitan_riscv/).

## Why the Code Was Split This Way

The eval loop was built as an additive DAG of slices, each landed independently so that downstream behaviour could be reasoned about without re-litigating upstream contracts. Every slice's public types live in a single subpackage and the next slice consumes them by name.

1. **P0.5 — Pack format keystone.** `src/eval/pack/` defines the on-disk pack contract (`pack.yaml`, `corpus/manifest.json`, `goldens/*.jsonl`, `thresholds.yaml`) and the typed `EvalPack` tree. Validation is structural-first (`validate_pack`) and schema-second (Pydantic). Without a stable pack format every later slice would re-derive paths.
2. **P3.0 — Pure ingest planner.** `runner/plan.py` produces a frozen `IngestPlan` from a loaded `EvalPack`. It joins paths and reads pack properties; it has no infra dependency. This is the seam tests use to assert plan shape without spinning up Weaviate.
3. **P3 — Live ingest executor.** `runner/execute.py` bridges the plan into `ingest_directory(...)` and reads post-ingest stats from Weaviate. The first runner module with a real infra dependency. `plan.py` stays pure.
4. **P4 — Retrieval + recall@k.** `runner/retrieve.py` runs hybrid search against the ingested collection; `runner/metrics.py` computes set-based recall@k and aggregates per qtype. Pure-logic metrics live in `metrics.py` so they can be unit-tested without a vector store.
5. **P5.0 — Pluggable judge contract.** `runner/judge.py` defines `JudgeClient`, `JudgeQuestion`, `JudgmentScore`, the prompt loader (`load_judge_prompt`), and the builder (`build_judge_client`). The `chat_model_factory` seam exists so fast tests inject a fake without importing langchain.
6. **P5 — Faithfulness executor.** `runner/faithfulness.py` ties P4 retrieval output to the P5.0 judge surface. The judge scores **retrieved chunks directly** — there is no answer synthesis step. End-to-end answer-faithfulness is a different subsystem (`src/guardrails/`).
7. **P6a — Threshold gating.** `runner/gate.py` compares per-qtype recall and faithfulness against `pack.thresholds.defaults`. Pure logic, no I/O. Includes the `total_queries_judged > 0` anti-gaming guard.
8. **P6b — CLI runner.** `src/eval/cli.py` wires P0.5 → P6a into a single subcommand with deterministic exit codes. Heavy modules are lazy-imported inside `run_cli` so argparse stays cheap.
9. **P6c — Multi-sample judge.** `score_goldens` gained `samples_per_claim`; the CLI gained `--samples-per-claim`. Per-claim score is the mean across samples; reasoning comes from the highest-scoring sample.

This split improves:

- **schema clarity** — pack contracts and runner result dataclasses are frozen and centrally defined;
- **test targeting** — pure stages (`plan`, `metrics`, `gate`) are unit-testable; infra stages (`execute`, `retrieve`, `faithfulness`) take injectable seams;
- **slice-by-slice review** — each slice landed independently with its own contract surface.

## Target Architecture

The eval loop is a one-way DAG. Each stage consumes the previous stage's frozen output and produces the next stage's frozen input. There is no fan-out and no feedback edge.

```
eval_pack/ on disk
      │
      ▼
┌──────────────────────────┐
│ pack.load_pack           │  validate_pack → EvalPack tree
│ src/eval/pack/loader.py  │  raises PackValidationError
└──────────┬───────────────┘
           │ EvalPack
           ▼
┌──────────────────────────┐
│ runner.plan_pack_ingest  │  pure: joins paths, picks collection name
│ runner/plan.py           │  IngestPlan
└──────────┬───────────────┘
           │ IngestPlan
           ▼
┌──────────────────────────┐
│ runner.execute_plan      │  live: ingest_directory + Weaviate stats
│ runner/execute.py        │  IngestReport
└──────────┬───────────────┘
           │ collection_name
           ▼
┌──────────────────────────┐
│ retrieve_for_goldens     │  embed query → hybrid search → top-k
│ runner/retrieve.py       │  RetrievalResults (sources + chunk texts)
└──────────┬───────────────┘
           │ RetrievalResults
           ▼
┌──────────────────────────┐
│ score_goldens            │  per golden: judge.score(JudgeQuestion)
│ runner/faithfulness.py   │  mean across N samples, best reasoning
└──────────┬───────────────┘
           │ FaithfulnessResults
           ▼
┌──────────────────────────┐
│ build_eval_report        │  aggregate recall + faithfulness per qtype
│ runner/report.py         │  EvalReport (frozen)
└──────────┬───────────────┘
           │ EvalReport
           ▼
┌──────────────────────────┐
│ validate_eval_report     │  compare to pack.thresholds.defaults
│ runner/gate.py           │  GateResult → exit code 0 or 1
└──────────────────────────┘
```

**Key architectural properties:**

- Each stage's output is a frozen dataclass (`@dataclass(frozen=True)`) or a Pydantic model — no in-place mutation, no implicit state sharing.
- The pure stages (`plan`, `metrics`, `gate`) have **no infra imports** and are unit-testable in isolation.
- The live stages (`execute`, `retrieve`, `faithfulness`) take dependency seams: `execute_plan` is module-top-imported so tests monkeypatch local names; `retrieve_for_goldens` monkeypatches `search`/`create_persistent_client`/`get_embedding_provider` at module-top; `score_goldens` takes a `JudgeClient` instance; `build_judge_client` takes a `chat_model_factory`.
- The CLI is a thin orchestrator. Business logic lives in the stage modules.

## Package Layout

```text
src/eval/
  __init__.py
  __main__.py                       # python -m src.eval entry
  cli.py                            # P6b — argparse + run_cli orchestrator
  pack/
    __init__.py                     # Public API: load_pack, validate_pack, EvalPack
    schema.py                       # Pydantic models: PackMeta, JudgeConfig, Golden, ...
    loader.py                       # load_pack — validate-then-materialise
    validate.py                     # validate_pack — structural checks, raises PackValidationError
    errors.py                       # PackValidationError
  runner/
    __init__.py                     # Public runner surface
    plan.py                         # P3.0 — pure: IngestPlan
    execute.py                      # P3 — live ingest + IngestReport
    retrieve.py                     # P4 — hybrid search + RetrievalResults
    metrics.py                      # P4 — recall_at_k + aggregate_recall_by_qtype
    judge.py                        # P5.0 — JudgeClient, prompt loader, samples_per_claim reader
    faithfulness.py                 # P5 — score_goldens + aggregate_faithfulness_by_qtype
    report.py                       # IngestReport, EvalReport, build_eval_report
    gate.py                         # P6a — validate_eval_report → GateResult
    prompts/
      judge_v1.md                   # Packaged default judge prompt (v1)

evals/packs/opentitan_riscv/        # Reference pack
  pack.yaml                         # PackMeta + JudgeConfig
  thresholds.yaml                   # Per-qtype floors + min_goldens_per_qtype
  corpus/
    manifest.json                   # {entries: [{path, sha256}, ...]}
    <docs...>
  goldens/
    factoid.jsonl
    qfs.jsonl
    multi_topic.jsonl
    multi_aspect.jsonl
    adversarial.jsonl
    messy.jsonl
    out_of_corpus.jsonl
  prompts/
    judge_v1.md                     # Pack-local override of the judge prompt
```

## End-to-End Execution Flow

1. **CLI dispatch.** `python -m src.eval run <pack_path> [...]` calls `main()` in [`cli.py`](../../src/eval/cli.py), which parses argv and delegates to `run_cli(...)`.
2. **Pack load (exit 2 on error).** `run_cli` calls `load_pack(pack_path)`. `load_pack` runs `validate_pack` first, then materialises the typed `EvalPack` tree. `PackValidationError` or `FileNotFoundError` exits with code 2.
3. **Plan.** `plan_pack_ingest(pack, pack_dir)` builds a frozen `IngestPlan` (collection name, documents dir, doc paths, expected count, corpus pin).
4. **Ingest (live).** `execute_plan(plan, fresh=...)` calls `ingest_directory(...)` with `fresh=True` (default) recreating the target Weaviate collection, or `fresh=False` (incremental update). It queries `get_collection_stats` post-ingest and returns an `IngestReport`.
5. **Retrieve.** `retrieve_for_goldens(collection_name, pack.goldens, k=...)` embeds each golden's query, runs hybrid search (alpha=0.5) against the collection, and returns a `RetrievalResults` whose `per_query` map carries both `retrieved_sources` and `retrieved_chunks` (rank-ordered).
6. **Build judge.** `build_judge_client(pack, chat_model_factory=..., pack_dir=...)` reads `pack.meta.judge` and constructs a `JudgeClient` wrapping a structured-output ChatModel. Prompt resolution prefers `<pack_dir>/prompts/judge_<version>.md` and falls back to the packaged default only when the pack has no `prompts/` directory at all (a `prompts/` dir without the expected file is an error).
7. **Score.** `score_goldens(retrieval_results, pack.goldens, judge_client, samples_per_claim=N)` calls the judge `N` times per golden whose `retrieved_chunks` is non-empty; goldens with empty retrieval are skipped (counted in `total_queries_skipped`). The stored score is the mean across samples; the stored reasoning is from the highest-scoring sample (Python `max` is stable — first-occurrence wins on ties).
8. **Aggregate.** `aggregate_recall_by_qtype` computes mean recall@k per qtype (omitting qtypes with no scoreable goldens); `build_eval_report` rolls up the per-query recall map, the per-qtype recall, and the per-qtype faithfulness into a single frozen `EvalReport`.
9. **Gate.** `validate_eval_report(pack, report)` compares per-qtype metrics against `pack.thresholds.defaults`. Returns `GateResult(passed=bool, failures=tuple)`.
10. **Emit and exit.** `--format text` prints a per-qtype table + `PASS`/`FAIL` summary; `--format json` prints a single-line JSON payload. Exit code is `0` on pass, `1` on fail, `3` on any runtime/infra exception, `2` on pack load/validation error.

## Stage-by-Stage Contracts

| Stage | File | Main Input | Main Output |
| --- | --- | --- | --- |
| `validate_pack` | [`src/eval/pack/validate.py`](../../src/eval/pack/validate.py) | pack directory path | `None` (raises `PackValidationError`) |
| `load_pack` | [`src/eval/pack/loader.py`](../../src/eval/pack/loader.py) | pack directory path | `EvalPack` |
| `plan_pack_ingest` | [`src/eval/runner/plan.py`](../../src/eval/runner/plan.py) | `EvalPack`, `pack_dir` | `IngestPlan` |
| `execute_plan` | [`src/eval/runner/execute.py`](../../src/eval/runner/execute.py) | `IngestPlan`, `fresh: bool` | `IngestReport` |
| `retrieve_for_goldens` | [`src/eval/runner/retrieve.py`](../../src/eval/runner/retrieve.py) | `collection_name`, `goldens`, `k` | `RetrievalResults` |
| `build_judge_client` | [`src/eval/runner/judge.py`](../../src/eval/runner/judge.py) | `EvalPack`, factory, `pack_dir` | `JudgeClient` |
| `score_goldens` | [`src/eval/runner/faithfulness.py`](../../src/eval/runner/faithfulness.py) | `RetrievalResults`, `goldens`, `JudgeClient`, `samples_per_claim` | `FaithfulnessResults` |
| `aggregate_recall_by_qtype` | [`src/eval/runner/metrics.py`](../../src/eval/runner/metrics.py) | `RetrievalResults`, `goldens` | `dict[qtype, float]` |
| `build_eval_report` | [`src/eval/runner/report.py`](../../src/eval/runner/report.py) | retrieval + faithfulness + recall map | `EvalReport` |
| `validate_eval_report` | [`src/eval/runner/gate.py`](../../src/eval/runner/gate.py) | `EvalPack`, `EvalReport` | `GateResult` |

## CLI Interface

The CLI is a single subcommand. See [`cli.py`](../../src/eval/cli.py) (`_build_parser`).

```bash
python -m src.eval run <pack_path> \
    [--k INT] \
    [--format {text,json}] \
    [--samples-per-claim INT] \
    [--fresh | --no-fresh] \
    [--verbose]
```

| Flag | Default | Effect |
| --- | --- | --- |
| `pack_path` (positional) | — | Path to the eval_pack directory containing `pack.yaml`. |
| `--k` | `5` | Top-k retrieval cutoff per query (also the recall denominator suffix). |
| `--format` | `text` | `text` prints a table + summary; `json` prints a single-line payload to stdout. |
| `--samples-per-claim` | `None` | Override `pack.meta.judge.samples_per_claim`. `None` uses the pack value (or `1` if absent — see `read_samples_per_claim`). |
| `--fresh` / `--no-fresh` | `--fresh` | `--fresh` drops + recreates the target Weaviate collection (deterministic isolation). `--no-fresh` runs incremental update. Mutually exclusive. |
| `--verbose` | off | Sets root logging to `DEBUG` and prints tracebacks for exit-3 failures. |

**Exit codes** (defined in `run_cli`):

| Code | Meaning |
| --- | --- |
| `0` | Gate passed (`GateResult.passed is True`). |
| `1` | Gate failed (at least one threshold breach). |
| `2` | Pack load / validation error (`PackValidationError` or `FileNotFoundError`). |
| `3` | Any other runtime / infra error (ingest, retrieve, judge, etc.). |

## Configuration Model

The pack is the configuration surface. Two files carry typed config; both are validated by `validate_pack` before any runner stage executes.

### `pack.yaml`

Schema: `PackMeta` in [`src/eval/pack/schema.py`](../../src/eval/pack/schema.py).

```yaml
name: opentitan_riscv                           # str — pack identifier; used in collection_name
version: 1                                      # int — bump on breaking pack edits
profile: asic_riscv_soc                         # str — must be in KNOWN_PROFILES
                                                #       (asic_riscv_soc | eda_command_reference | generic)
corpus_pin: a8e3ff902cb2d2578501a83620634792483b301b0d56fe71add49578bba2daaf
                                                # SHA-256 over sorted {path}:{sha256} lines from manifest.
                                                # Recomputed by validate_pack; mismatch is a hard error.
description: Development-grade ASIC/RISC-V reference pack ...
collection_name_template: "ragweave_test_{name}_{corpus_pin_short}"
                                                # {corpus_pin_short} = corpus_pin[:8].
                                                # Resolved at EvalPack.collection_name.
judge:                                          # JudgeConfig
  tier1_model: claude-haiku-4-5-20251001        # LiteLLM alias passed to get_llm()
  tier1_prompt_version: v1                      # judge_<version>.md must exist in prompts/
  temperature: 0.0                              # 0.0 for deterministic judging
  samples_per_claim: 3                          # judge invocations per golden; mean score, best reasoning
```

### `thresholds.yaml`

Schema: `Thresholds` in [`src/eval/pack/schema.py`](../../src/eval/pack/schema.py).

```yaml
profile: asic_riscv_soc                         # must match pack.yaml profile
defaults:                                       # per-qtype floors checked by validate_eval_report
  factoid:
    recall_at_5: 0.8                            # recall_key = f"recall_at_{report.k}"
    mrr: 0.6                                    # mean reciprocal rank floor (checked by validate_eval_report)
    # faithfulness: 0.75                        # add to floor faithfulness on this qtype
overrides: []                                   # list of qtype/qid floor overrides (see "Threshold Overrides")
min_goldens_per_qtype:                          # enforced by validate_pack on pack load
  factoid: 20                                   # raises PackValidationError if file has fewer
  qfs: 10
  multi_topic: 8
  multi_aspect: 8
  adversarial: 5
  out_of_corpus: 5
  messy: 5
```

#### Threshold Overrides

`overrides` is a list of `ThresholdOverride` entries (`schema.py`), each consumed by `validate_eval_report` (`src/eval/runner/gate.py`). Each entry targets **exactly one** of a `qtype` or a `qid`, and must declare **at least one numeric metric floor** (the floors arrive as extra fields, e.g. `recall_at_5`, `mrr`, `faithfulness`). Both invariants are enforced fail-fast at pack load — a malformed entry raises at `Thresholds` construction (which both `loader.py` and `validate.py` perform), so the gate never sees an invalid override.

```yaml
overrides:
  - qtype: factoid          # qtype-level: replace the factoid mrr floor
    mrr: 0.8
  - qtype: adversarial      # qtype-level: introduces a qtype absent from defaults
    recall_at_5: 0.9
  - qid: factoid_017        # qid-level: floor a single query's individual score
    recall_at_5: 0.4
```

- **qtype overrides** merge into the effective floors used by the gate: each override **replaces** the matching per-metric floor in `defaults` (per metric, **last-wins** on duplicates), and may **introduce** qtypes or metrics absent from `defaults`. The aggregate check then runs over the union of `defaults` and qtype-override qtypes; the recall/mrr/faithfulness check bodies are unchanged and simply read the merged floor.
- **qid overrides** check that specific query's individual score against the floor. The metric key selects the per-query map on the `EvalReport`: `recall_at_{k}` → `per_query_recall`, `mrr` → `per_query_mrr`, `faithfulness` → `per_query_faithfulness`. A metric key that matches none of these (an unknown metric, or a `recall_at_<k>` whose `k` differs from `report.k`) is **skipped**. A qid override on a query that was skipped or not judged (absent from the selected map) is **skipped** — it never manufactures a failure. A qid failure carries `GateFailure.qid` (and the qtype resolved from `goldens`, or `""` if the qid is in no goldens file) and surfaces in the CI summary's failures table (`qtype | qid | metric | expected | actual`).



```json
{
  "entries": [
    {"path": "riscv-privileged.md", "sha256": "..."},
    {"path": "opentitan_uart.md",  "sha256": "..."}
  ]
}
```

Every entry's `path` must resolve under `corpus/`; the recomputed corpus pin must equal `pack.yaml`'s `corpus_pin`.

### `goldens/<qtype>.jsonl`

One JSON object per line. Required fields: `qid`, `qtype`, `query`. `factoid` additionally requires `expected_answer_span` (enforced by `Golden._check_factoid_required` in [`schema.py`](../../src/eval/pack/schema.py)). Each row's `qtype` field must equal the file's filename stem. Extra fields are preserved on `Golden.raw`.

> **Note.** The validator requires at least one `goldens/*.jsonl` file. Corpus-only packs must ship an empty placeholder for each declared qtype.

### `prompts/judge_<version>.md` (optional)

Resolution order in `load_judge_prompt`:

1. If `<pack_dir>/prompts/` exists, the file `judge_<version>.md` must exist there. **Missing it is a hard error — there is no silent fallback when the directory is present.**
2. Otherwise the packaged default at [`src/eval/runner/prompts/judge_<version>.md`](../../src/eval/runner/prompts/) is used.

The template supports four substitutions: `{query}`, `{expected_answer_span}`, `{qtype}`, `{chunk_block}`.

## Metrics and Gating

### Recall@k

`recall_at_k(retrieved_sources, expected_sources)` in [`metrics.py`](../../src/eval/runner/metrics.py) is set-based:

```
recall@k = |expected ∩ retrieved_topk| / |expected|
```

Asymmetries:

- Empty `expected_sources` raises `ValueError` — empty expectation is a categorization mismatch, not a metric value. The aggregator filters these goldens before calling `recall_at_k`.
- Empty `retrieved_sources` returns `0.0` — empty retrieval is a real outcome.

`aggregate_recall_by_qtype` returns `statistics.mean` across the qtype's scoreable goldens; **qtypes whose goldens all lack `expected_source_docs` are omitted entirely** (no NaN, no zero).

### Faithfulness (chunk-level)

`score_goldens` in [`faithfulness.py`](../../src/eval/runner/faithfulness.py) passes the **retrieved chunks themselves** as `JudgeQuestion.chunk_texts` to the judge LLM. The judge returns a `JudgmentScore(score, reasoning)` with `score ∈ [0.0, 1.0]`.

> **Important.** This is chunk-level faithfulness — the eval judge scores retrieved chunks directly. There is no answer-synthesis step. End-to-end answer-faithfulness (response-vs-context) lives in `src/guardrails/` and is a separate production concern.

`aggregate_faithfulness_by_qtype` mirrors the recall aggregator: mean over scored goldens, qtypes with zero scored goldens are omitted.

### Gate

`validate_eval_report(pack, report)` in [`gate.py`](../../src/eval/runner/gate.py) iterates `pack.thresholds.defaults` and checks two floors per qtype:

- **Recall floor.** The recall metric key is `f"recall_at_{report.k}"` — derived from the report's `k`, not hardcoded. If the key is not declared for the qtype, the recall check is skipped.
- **Faithfulness floor.** Key `"faithfulness"`. Only checked if `report.total_queries_judged > 0` — this is the **anti-gaming guard**: a stale eval with no judged goldens cannot pass trivially on faithfulness because every faithfulness check is skipped when zero queries were judged.

Failures are aggregated, not short-circuited. `GateResult.passed` is `True` iff `failures == ()`.

## Multi-Sample Judge

`score_goldens` accepts `samples_per_claim: int = 1`. For each golden whose retrieval is non-empty, the judge is invoked `samples_per_claim` times against the same `JudgeQuestion`. The stored result:

- `score = statistics.mean(float(s.score) for s in samples)` — averaged across samples.
- `reasoning = max(samples, key=lambda s: s.score).reasoning` — taken from the highest-scoring sample. Python `max` is stable, so ties resolve to the **first-occurring** sample.

`samples_per_claim < 1` raises `ValueError`.

**Precedence chain for the effective count** (in [`cli.py`](../../src/eval/cli.py) `run_cli`):

1. CLI `--samples-per-claim N` if supplied.
2. Otherwise `read_samples_per_claim(pack)` (in [`judge.py`](../../src/eval/runner/judge.py)) reads `pack.meta.judge.samples_per_claim`.
3. If neither is set (legacy pack with the field absent), the default is `1`. `0` or negative raises `ValueError`.

Sampling is currently sequential (one call per iteration). Parallel sampling is a future-work item.

## Extending the Eval Loop

### Add a new qtype

1. Add `goldens/<qtype>.jsonl` with one JSON-per-line row. Each row's `qtype` field must equal the file stem.
2. If the new qtype has per-field requirements beyond `qid`/`qtype`/`query`, add a check to `Golden` in [`schema.py`](../../src/eval/pack/schema.py) — mirror the `_check_factoid_required` pattern.
3. Optionally add a per-qtype floor in `thresholds.yaml` under `defaults.<qtype>` and a `min_goldens_per_qtype.<qtype>` count.
4. No runner change required — `retrieve_for_goldens`, `score_goldens`, and the aggregators iterate qtypes generically.

### Add a new metric

1. Implement the pure function in `runner/metrics.py` (same shape as `recall_at_k`).
2. Add a `<metric>_by_qtype` field to `EvalReport` in [`report.py`](../../src/eval/runner/report.py) with a default to keep backward compatibility.
3. Aggregate it in `build_eval_report`.
4. Add a floor check in `validate_eval_report` ([`gate.py`](../../src/eval/runner/gate.py)) — mirror the existing `recall_key` / `"faithfulness"` blocks.
5. Surface it in `_emit_text` / `_emit_json` in [`cli.py`](../../src/eval/cli.py).

### Swap the judge model

Set `pack.meta.judge.tier1_model` to a different LiteLLM alias. `build_judge_client` wires it through `get_llm(model_alias=..., temperature=...)`. No runner change needed.

### Author a pack-local judge prompt

Create `<pack_dir>/prompts/judge_<version>.md` and set `pack.meta.judge.tier1_prompt_version: <version>`. The loader will refuse to fall back to the packaged default when a `prompts/` directory is present, so name the file exactly as the resolver expects.

## Deterministic vs LLM-Dependent Behavior

| Stage | Deterministic? | Notes |
| --- | --- | --- |
| `validate_pack`, `load_pack` | Yes | Pure I/O + schema validation. |
| `plan_pack_ingest` | Yes | Pure path joining. |
| `execute_plan` | Infra-dependent | Calls real ingest pipeline (Docling, embedding provider, Weaviate). Not LLM-dependent. |
| `retrieve_for_goldens` | Infra-dependent | Embedding model + hybrid search; deterministic for fixed inputs and fixed collection state. |
| `recall_at_k`, `aggregate_recall_by_qtype` | Yes | Pure set math. |
| `score_goldens` | **LLM-bound** | One judge call per `samples_per_claim` per golden. With `temperature=0.0` and a stable judge model results are near-deterministic but not byte-equal across runs. |
| `validate_eval_report` | Yes | Pure floor comparison. |

Practical effect: the recall path is reproducible from a fixed ingested collection. The faithfulness path depends on the configured judge LLM and any provider-side non-determinism.

## Troubleshooting

| Symptom | First check |
| --- | --- |
| `pack load error: pack.yaml: corpus_pin mismatch` (exit 2) | Recompute `corpus_pin` over sorted `path:sha256` lines from `corpus/manifest.json`; update `pack.yaml`. The pin is order-independent but order-sorted before hashing. |
| `pack load error: thresholds.yaml: min_goldens_per_qtype not met` (exit 2) | Add more goldens to `goldens/<qtype>.jsonl`, or lower the floor in `thresholds.yaml`. For corpus-only packs, ship an empty placeholder `goldens/<qtype>.jsonl` to satisfy the "no `*.jsonl` files" guard. |
| `pack load error: profile '<x>' is not in the known profile set` (exit 2) | Add the profile to `KNOWN_PROFILES` in [`schema.py`](../../src/eval/pack/schema.py), or use one of the existing profiles. |
| `pack load error: golden qid=... qtype=factoid is missing required field 'expected_answer_span'` (exit 2) | `factoid` rows must carry `expected_answer_span`. Add it or move the row to a non-factoid qtype file. |
| `FileNotFoundError: pack ... authored a prompts/ directory but is missing judge_<v>.md` (exit 2) | Either add `prompts/judge_<v>.md` to the pack, or remove the `prompts/` directory entirely to fall back to the packaged default. There is no halfway. |
| Gate fails on `faithfulness` but `total_queries_judged=0` would have skipped it | Look at the table: `judged=0` means retrieval returned zero chunks for every golden — fix the ingest/retrieve path, not the gate. The anti-gaming guard prevents trivial passes when nothing was scored. |
| `eval runtime error: ... RuntimeError: Post-ingest stats unavailable` (exit 3) | Weaviate collection didn't materialise. Inspect `ingest_directory` logs; verify the collection name template and that Weaviate is reachable. |
| Recall stays at `0.0` for a qtype with non-empty expected docs | Check `metadata['source']` shape — `recall_at_k` compares raw source strings. If ingestion stores absolute paths but goldens reference filenames, the set intersection is always empty. |

## Known Limitations and Future Work

- **Per-sample visibility.** Multi-sample runs collapse `samples_per_claim` calls into one mean score; per-sample scores and reasonings are not surfaced in `EvalReport`. P7-series teaser: report-level per-sample arrays.
- **Sequential sampling.** Samples are issued one at a time. With `samples_per_claim=3` and large golden counts, judge latency dominates wall time. Future P7b candidate: parallel sampling with a configurable concurrency cap.
- **No CI gate integration.** Exit codes are defined and stable, but no CI workflow runs the eval loop yet. P7c teaser: nightly eval + PR-gated regression check.
- **`expected_source_docs` matching is exact-string.** No path normalisation, no case folding. A future-work candidate is to normalise both sides through the ingest source-key contract.
- **No end-to-end answer-faithfulness in this loop.** Production answer-vs-context faithfulness lives in `src/guardrails/`. Bridging the two surfaces is a separate scope.
