# RagWeave Eval Loop — Plan (Vertical-Slice / TDD / Ralph-Aligned)

## 0. Shipped-state ledger (keep this current)

The plan below describes the *intended* architecture. This ledger records what is **actually shipped on `develop`** so the document stops drifting from the code. Update it whenever a slice lands.

| Area | Status | Where |
|---|---|---|
| Collection-selection plumbing (P0) | ✅ shipped | `src/vector_db/weaviate/`, `RAG_COLLECTION_NAME` |
| eval_pack format + validator + loader (P0.5) | ✅ shipped | `src/eval/pack/{schema,validate,loader,errors}.py` |
| Pack min-count gates (P2.0) | ✅ shipped | `src/eval/pack/validate.py`, `tests/eval/test_pack_min_counts.py` |
| OpenTitan goldens (P2) | ✅ shipped | `evals/packs/opentitan_riscv/`, `tests/eval/test_opentitan_goldens.py` (73 goldens) |
| Metric library (P3) | ✅ shipped | `src/eval/runner/metrics.py` (recall@k, MRR), `src/eval/metrics.py` |
| Factoid/retrieval suite runner (P4) | ✅ shipped | `src/eval/runner/{retrieve,execute,report}.py`, `src/eval/cli.py run` |
| LLM-as-judge Tier-1 (P5) + calibration (P9) | ✅ shipped | `src/eval/runner/{judge,faithfulness,calibration}.py` |
| Orchestrator + baseline diff + alert (P7/P8) | ✅ shipped | `src/eval/orchestrator.py`, `src/eval/runner/{baseline,alert,persistence}.py` |
| Multi-sample + parallel judging (P7b/P7c) | ✅ shipped | `src/eval/runner/faithfulness.py` (`max_parallel_judges`) |
| Offline PR smoke gate (P10) | ✅ shipped | `python -m src.eval.smoke` → `ci.yml`; live gate is `eval-gate.yml` |
| Run-history index (#140) | ✅ shipped | `src/eval/runner/run_history.py` → `<pack>.history.jsonl` |
| Sustained-regression detector (#141) | ✅ shipped | `src/eval/runner/sustained_regression.py`, wired into `run_nightly` |
| show-trend inspection CLI (#142) | ✅ shipped | `src/eval/runner/show_trend.py`, `src/eval/cli.py show-trend` |
| **Headless Claude CLI judge backend (P14)** | 🚧 this initiative | `src/eval/runner/judge_cli.py` (new), `--judge-backend` flag |
| **Judge-call robustness: isolation + timeout/retry (P15)** | 🚧 this initiative | `src/eval/runner/faithfulness.py`, `judge.py` |
| Ralph-A eval-failure investigator (P11) | ⏸️ HELD by decision | §10.3; un-defer once headless driver + signal trio are trusted |
| Real Slack/webhook alert delivery | ⏸️ deferred | `src/eval/runner/alert.py` is a log-only sink today |
| Cross-pack dashboard (Grafana) | ⏸️ deferred | premature until multiple customer packs exist |

**Signal-trajectory note.** #140 (data) → #141 (verdict) → #142 (view) built the *trusted signal* substrate so a future Ralph-A has something honest to act on. P14 (headless driver) is the *actuation* substrate: it lets an automated agent **run the eval with no API key**, reusing local Claude auth. P15 hardens the judge so that automation doesn't die on the first transient error. Together they are the precondition for un-holding P11.

## 1. Goals & success criteria

**Primary goal:** A content-agnostic eval loop that grades RagWeave on factoid, QFS, multi-topic, multi-aspect, and command-reference queries against pluggable customer-shaped corpora, with LLM-as-judge for generation quality, trend tracking for regression detection, and a **headless driver** so the loop can be run and scored by an unattended agent.

**Two layers, deliberately separated:**

1. **The loop** — fixed runtime (ingest → suites → metrics → judge → report → alert → trend). Customer-agnostic.
2. **The content** — a declarable, versioned `eval_pack` (corpus + goldens + thresholds + prompt overrides). Swappable per customer/profile.

The loop ships once and stays stable. New customers, new domains, new corpora land as new eval_packs without touching the runtime.

**Three pluggable backends (DI seams, resolved at call-time in `run_eval`):** `execute_fn` (ingest), `retrieve_fn` (retrieval), `judge_client_factory` (LLM-as-judge). Each can be a live implementation, an offline fake (smoke), or — new in P14 — a **headless Claude CLI** judge.

**Success criteria (binary — no wall-clock targets):**
- `python -m src.eval run <pack>` runs the full loop end-to-end against any conforming eval_pack and emits a scored report.
- `python -m src.eval run <pack> --judge-backend claude-cli` scores the pack using the **local headless `claude` CLI** — no `RAG_LLM_API_KEY` required, reusing local Claude auth.
- Test collection is fully isolated from prod by collection-name plumbing — provably no cross-write.
- Every metric has unit tests with synthetic-input expected outputs.
- LLM-as-judge ships with a hand-graded calibration fixture; agreement ≥ 0.8 vs. human before the judge is trusted in production reports.
- Injected regressions in CI synthetic packs are caught and reported (`python -m src.eval.smoke` exit 0).
- **A single transient judge/LLM failure does not abort a run** — the affected golden is skipped and counted; the run completes and reports (P15).
- Every vertical slice is independently shippable, has a failing→passing E2E test as its definition of done, and can be picked up by a fresh agent with the slice prompt alone.

**Non-goals:**
- Real-time eval on production traffic.
- Multi-tenant collection management beyond one collection per pack per env.
- Cross-model A/B for embedding-model migration (separate harness).
- Auto-merging eval-driven fixes (Ralph-on-regression stays human-reviewed).
- Replacing the litellm judge backend — the headless CLI judge is an **alternate**, opt-in backend; litellm remains the default.

## 2. Domain & first customer

RagWeave's first commercial surface is **ASIC design** — RISC-V cores, memory chips, complex SoCs, and EDA tooling (Synopsys, Cadence). The eval content reflects this:

- **OpenTitan + Ibex** — open-source RISC-V SoC IP. Markdown specs, register tables, hierarchical sections, cross-doc entity references. Representative of customer IP documentation. **(Shipped as the reference pack.)**
- **RISC-V Privileged ISA spec (PDF)** — standards-doc pathology: deep cross-refs, large doc, formal language.
- **EDA tool reference manuals (Synopsys / Cadence)** — command-reference pathology: thousands of commands, each with argument semantics. Queries here look like *"what does `set_max_delay -from X -to Y` do?"* — a structured-lookup query type. **First-class query type.**

OpenTitan/Ibex/RISC-V form the **initial reference eval_pack**. The EDA pack lands when a customer corpus is available. Generic / non-ASIC packs land later — same loop, different content.

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ Orchestrator (cron-script `scripts/run_nightly_eval.py` today;   │
│ Temporal workflow once volume justifies it). Reads pack paths:   │
│   1. Resolve pack → 2. Reset/ingest collection → 3. Run suites   │
│   → 4. Judge → 5. Aggregate + baseline-diff → 6. Persist report  │
│   → 7. Index run-history → 8. Detect sustained regressions       │
│   → 9. Alert                                                     │
└──────────────────────────────────────────────────────────────────┘
              │
              ├─► eval_pack loader (corpus manifest + goldens + thresholds + prompts)
              │
              ├─► Test Collection Selector ──► Weaviate `ragweave_test_<pack>_<hash>`
              │
              ├─► Query Suites (one per qtype, pack-driven)
              │
              ├─► Metric Library (recall@k, MRR; faithfulness via judge)
              │
              ├─► LLM-as-Judge Service  ◄── pluggable backend (judge_client_factory)
              │     ├─ litellm backend (default; src/common/llm/provider.py)
              │     └─ headless Claude CLI backend (P14; src/eval/runner/judge_cli.py)
              │
              ├─► Reporter (JSON artifact + Markdown summary + baseline-diff)
              │
              └─► Trend layer
                    ├─ run-history index   (#140; <pack>.history.jsonl)
                    ├─ sustained-regression detector (#141)
                    └─ show-trend CLI      (#142; read-only inspection)
```

## 3.5 Headless Claude CLI judge backend (P14 — the actuation keystone)

**Why.** Today the LLM-as-judge calls models through litellm, which requires `RAG_LLM_API_KEY` and a router config. That is a hard dependency for any unattended runner (CI agent, Ralph-A) on a box that already has an authenticated `claude` CLI. A headless-CLI backend lets the eval **run and score with zero extra credentials**, reusing local Claude auth. It is also the natural backend for a future agent that drives the eval via `claude -p`.

**The seam.** The judge is a single-method client:

```python
class JudgeClient(Protocol):
    def score(self, question: JudgeQuestion) -> JudgmentScore: ...
```

`run_eval` resolves `judge_client_factory` at call-time (`src/eval/cli.py`). A new factory `build_claude_cli_judge_client(pack, *, ...)` returns a `ClaudeCliJudgeClient` satisfying the same contract. **One wiring point.**

**Constrained invocation (load-bearing — do NOT call bare `claude -p`).** A naked `claude -p` loads the *entire Claude Code agent*: full system prompt, tools, MCP, Opus by default. Measured cost of one such call: **~$0.20 and ~29k cache-creation tokens**, and the agent may treat the rubric as a prompt-injection and refuse to echo values. The CLI judge therefore invokes a **clean-room, tool-less, cheap-model** judge:

```
claude -p "<rendered judge prompt + strict JSON-only instruction>"
  --output-format json
  --model <pack.judge.tier1_model>          # default claude-haiku-4-5-20251001
  --system-prompt "<judge rubric>"          # FULL REPLACE of the coding-agent persona
  --allowed-tools ""                        # no tools
  --setting-sources ""                      # clean room: ignore project/user settings
  --permission-mode <non-prompting>
```

**Output contract.** `--output-format json` yields an envelope:
`{"type":"result","subtype":"success","is_error":false,"result":"<model text>", ...}`.
The model's answer is the **stringified `.result`** field. The client must:
1. parse stdout as JSON (the envelope);
2. assert `is_error is False` and `subtype == "success"`;
3. extract `.result`, strip optional ```json fences;
4. `json.loads` → `JudgmentScore(score, reasoning)`, clamping `score` to `[0.0, 1.0]`.

**Testability seam.** `ClaudeCliJudgeClient` takes an injectable `run_fn` (defaults to a `subprocess.run` wrapper). Unit tests inject a fake `run_fn` returning canned envelopes — **no real subprocess in unit tests**. Exactly one model-gated test (env opt-in + `claude` on PATH, mirroring the real-Docling test pattern) shells out for real and asserts a `JudgmentScore` round-trips.

**Robustness (folds into P15 contract).** `score()` wraps the call in a timeout and a bounded retry; a non-zero exit, `is_error`, malformed envelope, or unparseable `.result` raises a typed `JudgeBackendError`, which the faithfulness layer isolates per-golden (skip + count, never crash the run).

## 4. The eval_pack format (keystone — Phase 0.5, shipped)

The format is the architectural keystone. Get this right and every later phase becomes content-agnostic.

### 4.1 Directory layout

```
evals/packs/opentitan_riscv/
├── pack.yaml              # Metadata: name, version, profile, corpus_pin, judge
├── corpus/
│   ├── manifest.json      # Sorted list of doc paths + per-doc SHA-256
│   └── docs/              # Or a git-submodule pin pointing externally
├── goldens/
│   ├── factoid.jsonl
│   ├── qfs.jsonl
│   ├── multi_topic.jsonl
│   ├── multi_aspect.jsonl
│   ├── command_reference.jsonl     # Optional per qtype
│   ├── adversarial.jsonl
│   └── out_of_corpus.jsonl
├── thresholds.yaml        # Per-qtype, per-category pass/fail gates + min counts
└── prompts/               # Optional per-pack judge-prompt overrides
    └── faithfulness.md
```

### 4.2 `pack.yaml`

```yaml
name: opentitan_riscv
version: 1
profile: asic_riscv_soc
corpus_pin: <sha256-of-(sorted-doc-paths + per-doc-content-hashes)>
description: OpenTitan IP + RISC-V Privileged ISA + Ibex
judge:
  tier1_model: claude-haiku-4-5-20251001     # also the default --judge-backend claude-cli model
  tier1_prompt_version: v1
  temperature: 0
  samples_per_claim: 3
collection_name_template: "ragweave_test_{name}_{corpus_pin_short}"
```

### 4.3 Golden query schema (uniform across qtypes)

```json
{
  "qid": "factoid-uart-baud-001",
  "qtype": "factoid",
  "query": "What is the default baud rate of the OpenTitan UART after reset?",
  "expected_answer_span": "the default UART baud rate is configured via the NCO register...",
  "expected_source_docs": ["hw/ip/uart/doc/index.md"],
  "expected_chunk_ids": null,
  "category": "register",
  "difficulty": "easy",
  "min_targets": { "recall_at_5": 0.8, "mrr": 0.5 }
}
```

For QFS: `reference_answer`, `expected_cited_chunks`, `required_aspects` replace `expected_answer_span`.
For command-reference: `command`, `expected_arg_semantics` (per-flag meaning dict) replace the span.

### 4.4 `thresholds.yaml`

Per-profile defaults, overridable per-qtype, per-category, per-difficulty; carries `min_goldens_per_qtype`. ASIC datasheets, RISC-V specs, and EDA manuals have different structural pathologies — one global threshold either passes everything or fails everything. Bake per-profile thresholds in.

### 4.5 Pack validator

`src/eval/pack/validate.py` — schema-validates a pack before any suite runs. Refuses to load malformed packs. **Third-state hardening (see §11.5):** the corrupt-but-present case (truncated manifest, merge-conflict-marked YAML, symlink loop) must degrade with a typed `PackValidationError`, never an uncaught `OSError`.

## 5. Collection-selection plumbing (Phase 0, shipped)

`WeaviateStore`/`RagChain`/ingest accept `collection_name`; `RAG_COLLECTION_NAME` env var; `--collection` CLI flags; `src/vector_db/weaviate/admin.py` lifecycle helpers. Isolation is the hard boundary that keeps eval corpora out of prod collections.

## 6. Query taxonomy

| Type | Definition | Metrics | Ground truth |
|---|---|---|---|
| **Factoid** | Single fact, one correct passage | recall@k, MRR, nDCG@k | Answer span + source doc |
| **QFS — single-doc** | Summarize within one doc | Faithfulness, completeness, citation accuracy | Reference summary + cited chunks |
| **QFS — cross-doc** | Synthesize across docs | Same + cross-doc coverage | Reference summary + cited chunks across docs |
| **Multi-topic disjoint** | One query, N unrelated topics | Per-topic recall@k, topic coverage | Per-topic expected sources |
| **Multi-aspect (1 topic)** | One topic, N sub-questions | Sub-question coverage, recall@k | Per-sub-question expected chunks |
| **Command-reference (EDA)** | Tool command with arg semantics | Arg-semantic-match + citation accuracy | Per-flag meaning dict + source manual section |
| **Adversarial / prompt-injection** | Malicious query | Sanitization behavior | Expected sanitizer action |
| **Out-of-corpus** | Unanswerable | Correct refusal | Expected refuse + low confidence |
| **Messy / typo** | Garbled clean query | Recall preservation vs. clean | Same as clean variant |

## 7. Metric library

```
src/eval/runner/metrics.py    # recall@k, MRR (shipped; pure, unit-tested)
src/eval/metrics.py           # facade / additional metric helpers
```

Each metric ships with: unit tests on synthetic inputs with known outputs, boundary tests (empty / all-relevant / none-relevant), and a golden-fixture integration test against a tiny mini-corpus. **Mutation-aware test data:** design samples so first/last/max/mean differ, or coincident values silently insulate aggregator mutations.

## 8. LLM-as-judge

### Tiering
- **Tier 1** — runs on every query. Default `claude-haiku-4-5-20251001`; pack can override. Per-golden faithfulness scoring against the expected answer span / retrieved chunks (chunk-level; not answer synthesis).
- **Tier 2** — sampled calibration over a subset; compares Tier-1 ↔ Tier-2 agreement.
- **Tier 3** — human-graded fixture (once-off baseline, expanded over time). Validates Tier 2.

### Backends (P14)
- **litellm** (default) — `src/common/llm/provider.py` → litellm Router. Needs `RAG_LLM_API_KEY`.
- **claude-cli** — `src/eval/runner/judge_cli.py`, constrained `claude -p` (§3.5). Needs only local Claude auth.

Backend is selected by `--judge-backend {litellm,claude-cli}` (default `litellm`) and is independent of the pack — the same pack scores identically in shape under either backend; only the model call site differs.

### Calibration gate
Before a judge is trusted in production reports, it must clear ≥ 0.8 agreement (Cohen's kappa or equivalent) on the Tier-3 human-graded fixture. **A backend swap is a recalibration trigger:** the claude-cli backend must clear the same calibration gate against the frozen fixture before its scores are reported in production (its scores may differ from litellm's even at temperature 0 because the harness/system-prompt differs).

### Prompts
Version-tagged under `src/eval/runner/prompts/`, pack-overridable. The judge rubric used as the CLI `--system-prompt` is sourced from the same template (single source of truth).

## 9. Vertical slices

Each slice is **self-contained** — a fresh agent picks it up with the slice prompt alone and produces a red E2E test, minimum implementation, refactor pass, and updated docs.

### Slice dependency DAG (current)

```
[shipped] P0 → P0.5 → P1 → {P2.0→P2, P3→P4, P5.0→P5→P6, P6.5.0→P6.5} → P7 → P8 → P9 → P10
[shipped] #140 run-history ──► #141 sustained-regression ──► #142 show-trend
                                                                   │
                              ┌────────────────────────────────────┘
                              ▼
        P14 headless Claude CLI judge backend ──► P15 judge-call robustness
                              │                          │
                              └──────────┬───────────────┘
                                         ▼
                          [HELD] P11(A) Ralph-on-regression investigator
```

### P14 — Headless Claude CLI judge backend

> **Goal:** Add `src/eval/runner/judge_cli.py` with `ClaudeCliJudgeClient` (satisfies `score(JudgeQuestion) -> JudgmentScore`) and a factory `build_claude_cli_judge_client`, wired behind a new `--judge-backend {litellm,claude-cli}` flag on `python -m src.eval run` (default `litellm`). Invocation is the constrained, tool-less, haiku-pinned `claude -p` of §3.5. `ClaudeCliJudgeClient` takes an injectable `run_fn` for testability.
>
> **Red E2E:** `tests/eval/test_judge_cli.py::test_run_eval_scores_via_claude_cli_backend` — wires the claude-cli factory through `run_eval` with a **fake `run_fn`** returning a canned `--output-format json` envelope; asserts a scored `EvalReport` with the parsed faithfulness value (the integration seam, where teeth go missing).
> **Red-reason proof:** before implementation the test fails with `ImportError`/`AttributeError` on the missing `judge_cli` module / `--judge-backend` flag — not a fixture or parse error.
> **Unit teeth:** envelope parse success; ```json fence stripping; `is_error: true` → `JudgeBackendError`; malformed envelope → error; unparseable `.result` → error; `score` clamped to `[0,1]`; the constructed argv contains `--model`, `--system-prompt`, `--allowed-tools`, `--output-format json` (pins the constraint contract so a future edit can't silently un-constrain it).
> **Model-gated real test:** `test_claude_cli_judge_smoke_real` — `@pytest.mark.slow + integration`, skipped unless `claude` is on PATH **and** `RAGWEAVE_EVAL_CLI_LIVE=1`; shells out once, asserts a `JudgmentScore` with `0.0 ≤ score ≤ 1.0`.
> **DoD:** All teeth green; `tests/eval` full suite green; `python -m src.eval.smoke` exit 0 (unchanged — default backend untouched); CLI `--help` shows `--judge-backend`.

### P15 — Judge-call robustness (isolation + timeout/retry)

> **Goal:** Make a single judge failure non-fatal. In `src/eval/runner/faithfulness.py` (sequential `:179` and parallel `:219/:227` paths), wrap each `judge_client.score()` so a raised exception is caught, logged, the golden counted as **skipped** (incrementing `total_queries_skipped`, excluded from faithfulness aggregation), and the run continues. Add a configurable timeout + bounded retry (default: 1 retry, conservative timeout) around the call; expose `JudgeBackendError` as the typed failure both backends raise.
>
> **Red E2E:** `tests/eval/test_judge_robustness.py::test_one_golden_failure_does_not_abort_run` — a judge client that raises on golden *N* and scores the rest; assert the run completes, golden *N* is skipped (counted), the other goldens are scored, and the gate evaluates on the survivors. **Before the fix this test red-fails with the raised exception propagating out of `run_eval`.**
> **Discriminating teeth:** retry-then-succeed (transient error on attempt 1, success on attempt 2 → golden scored, not skipped); retry-exhausted → skipped, not crash; timeout path → `JudgeBackendError` → skipped. Parallel-path isolation: failure of one submitted future does not poison the pool (other futures still resolve).
> **Coverage backfill (same slice):** add the three currently-zero-coverage error tests flagged by audit — judge raises (litellm path), corrupted `baseline.json` degrades to "absent" (orchestrator narrow-catch), and ingest error surfaces without crashing aggregation.
> **DoD:** All teeth green; full `tests/eval` green; smoke exit 0; the orchestrator still isolates per-pack (broad catch) AND now isolates per-golden (narrow catch around `score`).

### Existing slices P0–P10 (shipped — see §0 ledger)

P0 collection plumbing · P0.5 pack format+validator · P1 OpenTitan pack · P2.0/P2 goldens · P3.0/P3 metrics · P4 factoid/multi-topic suite · P5.0/P5 judge Tier-1 · P6 QFS · P6.5.0/P6.5 command-reference · P7 orchestrator · P8 reporting/baseline-diff · P9 calibration loop · P10 offline smoke gate. Their slice contracts are preserved in git history and the engineering guide; this revision does not re-specify them.

## 10. Delivery patterns (orthogonal to phases)

### 10.0 Slice precondition contract — every slice must declare
1. **Named red-test path** — exact `file::test_name`, runnable verbatim before any code.
2. **Concrete numeric targets** where applicable — exact counts/thresholds, no "tight"/"enough".
3. **Frozen fixture SHAs** for any hand-graded inputs the slice consumes.
4. **Red-reason proof** — capture the failure output; confirm "feature absent" (`ImportError`/`NotImplementedError`/missing flag), not a test-harness error.
5. **Agent green-gate vs. human-review checkpoint, separated** — machine-checkable assertions end the slice; human-judgment gates unblock the *next* slice.
6. **Fixture immutability** — a slice consuming a frozen fixture declares "fixture not modified by this slice."

### 10.1 TDD per slice — non-negotiable
red → green → refactor; the failing E2E test must fail for the *right* reason. For metric/judge work, "eval of eval" applies: synthetic inputs with known outputs before any metric code; hand-graded fixtures before any judge prompt is trusted.

### 10.2 Vertical-slice independence
Each slice prompt is self-contained: file paths, contracts, DoD, E2E test all specified. Commit before any backgrounded soak; never end a turn with an unresolved stash.

### 10.3 Ralph loops (where they fit)

**Ralph-A — Eval-failure investigator (non-merging) — HELD until P14+P15 land:**
```
while sustained_regressions_exist and iterations < cap:
    pick the worst SUSTAINED-regressing (qtype, metric)        # from #141 detector
    dispatch subagent: "diagnose + propose minimal patch"
    apply patch in isolated branch
    re-run smoke eval (and, where credentialled, --judge-backend claude-cli)  # P14 driver
    if improvement: commit + open PR for human review
    else: revert + try next hypothesis
```
Never auto-merges. Bounded iterations. Each PR human-reviewed. It reads `load_run_history` + `detect_sustained_regressions` (the signal trio) and actuates through the P14 headless driver (no API key needed). P15 ensures a flaky judge call doesn't abort its smoke re-runs.

**Ralph-B — Golden expander (human-gated):** drafts to `goldens/_drafts/`; human promotes. Unchanged.

### 10.4 End-to-end testing — the meta-recursion
The eval *is* the E2E test for RagWeave's retrieval+generation. "E2E testing of the eval loop" means: **does the loop catch injected regressions in a known-bad pack?** That is the canonical meta-test (`src/eval/smoke.py`). Every later change must preserve injected-regression detection.

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Judge model drift breaks comparisons over time | Pin judge model + prompt version in every report; calibration loop (P9); recalibrate on bump **or backend swap** |
| Test collection storage accumulates | Auto-delete `ragweave_test_*` older than retention; documented GC job |
| Pack format churn invalidates goldens | `pack.yaml` version field; loader rejects mismatched versions |
| Goldens authoring is the long pole | Start tight; expand via Ralph-B with human review |
| Flaky judge scores cause spurious regressions | Temperature 0 + multi-sample mean; cache by content hash; P15 isolation |
| **Headless `claude -p` is expensive / persona-contaminated** | Never call bare `claude -p`: pin haiku, replace system prompt, disable tools, clean-room settings (§3.5) |
| **Headless CLI auth/PATH absent in CI** | claude-cli backend is opt-in; default stays litellm; real CLI test is env-gated + skipped when `claude` absent |
| **A transient judge error aborts the nightly run** | P15 per-golden isolation + timeout/bounded-retry; orchestrator already isolates per-pack |
| Per-customer corpora leak into shared infra | Collection-name plumbing (P0) is the hard boundary; CI isolation regression test |
| Ralph-A proposes bad patches | Never auto-merges; PR-only; bounded iterations |

## 11.5 Robustness dimension (audit-derived — explicit acceptance bar)

The eval loop is itself production software; its robustness is graded on the same bar as the system it grades. Current audit verdict: **MEDIUM** — strong pack validation, resilient per-pack orchestration, clean resource cleanup; brittle at the judge/ingest call sites.

**Robustness invariants (must hold; tested):**
1. **Per-golden judge isolation** — one `score()` failure → skipped+counted, never a run abort. *(P15)*
2. **Bounded external calls** — every LLM/CLI judge call has a timeout and a bounded retry; no unbounded hang. *(P15)*
3. **Narrow best-effort wiring** — trend/baseline wiring in the orchestrator catches *specific* exception types (`OSError`, `json.JSONDecodeError`, `ValueError`, `KeyError`), never broad `except Exception`, except the deliberate per-pack isolation boundary in `run_nightly`.
4. **Optional-file third state** — present / absent / **corrupt** are three states; corrupt must degrade like absent via a *narrow* catch, not crash. Applies to `baseline.json`, history JSONL, manifest, goldens.
5. **Determinism** — temperature 0 + submission-order sample re-sort + timestamp-sorted history + injected `now`; no `random`, no `utcnow()`. Backend swap may change *values* (recalibrate) but not *ordering*.
6. **No silent coverage loss** — if a run skips goldens (judge failure, empty query), the skip count is reported, never silently dropped from the denominator.

**Coverage targets (raise to here as part of P14/P15):**
- Judge error paths: from **0 tests** → cover raise / timeout / malformed-output / retry-then-succeed / retry-exhausted, for **both** backends.
- Ingest error paths: from **0 tests** → at least one doc-level-failure-tolerated test.
- Corrupted `baseline.json`: from **0 tests** → one degrade-to-absent test.
- New modules (`judge_cli.py`) ship with `@summary` + docstrings and direct unit tests at parity with `judge.py`.

## 12. Open decisions

- **claude-cli judge default model** — haiku 4.5 (cheap, fast) is the default; a pack may pin a stronger model for high-stakes calibration runs. Keep haiku as the global default to bound cost.
- **claude-cli concurrency** — subprocess-per-call is heavier than an in-process litellm call; bound it with the existing `max_parallel_judges` and a per-call timeout. Revisit a batched/stream-json mode only if nightly wall-clock becomes a problem.
- **Where retry lives** — in the client (`score`) vs. the faithfulness layer. Decision: a *thin* timeout+retry in the client (transport concern), and *isolation* (skip+count) in the faithfulness layer (policy concern). Don't double-retry.
- **EDA pack timing** — synthetic for P6.5 unit; real pack lands when corpus arrives.
- **Multi-pack nightly orchestration** — one-per-pack workflow once >1 pack exists; cron-script handles single-pack until then.

## 13. Done definition for the whole loop

The loop is "shipped" when **all** hold:
- Every slice P0 → P10 has its red E2E test green in main. *(met)*
- The OpenTitan reference pack runs end-to-end via `python -m src.eval run <pack>` and produces a valid report. *(met)*
- An injected regression in a copy of the OpenTitan pack is detected and alerted (`smoke` exit 0). *(met)*
- Judge Tier-1 has cleared its calibration gate (agreement ≥ 0.8). *(met for litellm; claude-cli must clear the same gate before production use — P14)*
- A second eval_pack loads and runs through the same runtime with no code changes. *(met)*
- **The loop can be run and scored with no API key via `--judge-backend claude-cli`.** *(P14)*
- **A single transient judge failure does not abort a run.** *(P15)*
- README + engineering guide for the eval loop exist and reflect the shipped code, including the headless backend and the robustness invariants.

When those hold, the loop is content-agnostic, regression-catching, headless-runnable, and robust enough to put under an unattended Ralph-A.
