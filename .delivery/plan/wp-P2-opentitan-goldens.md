---
slice_id: P2-opentitan-goldens
validable_outcome: |
  `uv run pytest tests/eval/test_opentitan_goldens.py -q` reports all tests green, including:
    1. `test_pack_validates_under_count_gate`: `validate_pack(evals/packs/opentitan_riscv)` returns None (no exception) once thresholds declares `min_goldens_per_qtype` AND all 7 goldens files contain ≥ declared minimum rows.
    2. `test_per_qtype_min_counts_pinned`: thresholds.yaml declares `min_goldens_per_qtype: {factoid: 20, qfs: 10, multi_topic: 8, multi_aspect: 8, adversarial: 5, out_of_corpus: 5, messy: 5}` (exact values).
    3. `test_actual_goldens_meet_or_exceed_minimums`: each `goldens/<qtype>.jsonl` has ≥ declared count. Counts read by counting non-blank lines per file.
    4. `test_factoid_spans_are_verbatim_in_source_docs`: for EVERY golden in `goldens/factoid.jsonl`, `expected_answer_span` appears as a verbatim substring of the file named by `expected_source_docs[0]`. Zero exceptions. Failure message must name the offending qid AND the would-be source path.
    5. `test_all_expected_source_docs_resolve_to_manifest`: for every golden across all 7 files, every entry in `expected_source_docs` is a path listed in `corpus/manifest.json`'s entries (i.e., resolves to a real corpus doc). Failure names qid + missing path.
    6. `test_qids_unique_within_each_qtype`: no duplicate `qid` values within any single `goldens/<qtype>.jsonl`. Cross-qtype duplication is allowed.
  PLUS: `uv run pytest tests/eval/ tests/vector_db/ tests/retrieval/ -q -m "not slow and not integration"` stays fully green (no regressions on P0/P0.5/P1/P2.0).
touches:
  - evals/packs/opentitan_riscv/thresholds.yaml
  - evals/packs/opentitan_riscv/goldens/factoid.jsonl
  - evals/packs/opentitan_riscv/goldens/qfs.jsonl
  - evals/packs/opentitan_riscv/goldens/multi_topic.jsonl
  - evals/packs/opentitan_riscv/goldens/multi_aspect.jsonl
  - evals/packs/opentitan_riscv/goldens/adversarial.jsonl
  - evals/packs/opentitan_riscv/goldens/out_of_corpus.jsonl
  - evals/packs/opentitan_riscv/goldens/messy.jsonl
  - tests/eval/test_opentitan_goldens.py
depends_on:
  - P1-opentitan-pack
  - P2.0-golden-quantity-gates
---

# P2 — OpenTitan goldens authoring (corpus-grounded)

## End-state (validable, machine-checkable)

The OpenTitan reference pack ships 61+ hand-authored goldens across seven
qtypes. Every factoid is grounded: its `expected_answer_span` is a
verbatim substring of a real corpus doc, and `expected_source_docs`
points at that doc. The pack passes `validate_pack` with the P2.0 count
gate engaged. No runner or metric implementation in this slice.

## Current state (surveyed inline by SA1 — see corpus map in conversation log)

- Worktree branched off `feat/p2.0-golden-quantity-gates` and merged `feat/p1-opentitan-pack`. The OpenTitan pack is present; the P2.0 count gate is active.
- `goldens/factoid.jsonl` exists as an empty placeholder (0 bytes). Will be overwritten with 20+ grounded factoid rows.
- `thresholds.yaml` declares `profile: asic_riscv_soc`, `defaults`, `overrides: []`. NO `min_goldens_per_qtype` yet — must be added.
- Corpus docs:
  - `docs/riscv_priv_isa_intro.md` — privilege modes (M/S/U), mode transitions, hart state
  - `docs/riscv_priv_csrs.md` — mstatus / mtvec / mepc field-level CSR spec
  - `docs/opentitan_uart.md` — UART feature summary, register map, bring-up
  - `docs/opentitan_hmac.md` — SHA-2/HMAC accelerator, CFG/CMD, software flow
  - `docs/ibex_overview.md` — Ibex CPU core ISA, pipeline, privilege, security
- Validator (`src/eval/pack/validate.py`):
  - `qtype` must match filename stem (e.g. `qfs.jsonl` → `qtype: "qfs"`).
  - `expected_answer_span` is REQUIRED for factoid only (per `Golden._check_factoid_required`).
  - For non-factoid qtypes, span is optional.
  - `expected_source_docs` defaults to `[]` (optional).
  - `min_goldens_per_qtype` (P2.0) raises if counts unmet.

## Plan-spec alignment

This slice pins the concrete numeric targets called out in EVAL_LOOP_PLAN.md §10.0 / P2.0:
`factoid ≥ 20, qfs ≥ 10, multi_topic ≥ 8, multi_aspect ≥ 8, adversarial ≥ 5, out_of_corpus ≥ 5, messy ≥ 5`. Command-reference defers to P6.5.0 (not in this slice).

## Required deliverables

### thresholds.yaml (modify)

Add the `min_goldens_per_qtype` block AT THE END (keep existing keys). Result:

```yaml
profile: asic_riscv_soc
defaults:
  factoid:
    recall_at_5: 0.8
    mrr: 0.6
overrides: []
min_goldens_per_qtype:
  factoid: 20
  qfs: 10
  multi_topic: 8
  multi_aspect: 8
  adversarial: 5
  out_of_corpus: 5
  messy: 5
```

### Goldens — 7 JSONL files

For each file, the **filename stem MUST equal the `qtype`** field. Author at least the declared count (NOT count+1; the boundary is the gate, but a slight buffer ≥1 per qtype is preferred to absorb later edits without redipping below the gate).

#### `goldens/factoid.jsonl` — ≥20 rows

Every row:
- `qid`: zero-padded with a `f` prefix, e.g. `f001` .. `f020`.
- `qtype`: `"factoid"`.
- `query`: a question whose answer is unambiguously findable in ONE corpus doc.
- `expected_answer_span`: VERBATIM substring of the doc named by `expected_source_docs[0]`. No paraphrasing. No re-casing. If the doc says `"32-byte TX and RX FIFOs"`, the span is that exact string.
- `expected_source_docs`: `["docs/<filename>.md"]` — exact manifest path.

Distribution: at least 3 factoids per corpus doc (covers all 5 docs evenly). Pull spans from SA1's corpus-map candidates; SA1 listed ≥10 quotable spans per doc, so coverage is achievable.

#### `goldens/qfs.jsonl` — ≥10 rows (query-focused summary)

Per row: `qid` (`qfs001`..), `qtype: "qfs"`, `query` (asks for a brief summary of a topic that spans 1+ sections within a doc), `expected_source_docs` (1-2 docs). Omit `expected_answer_span` (qfs is summarization, not extractive). Add an optional `summary_anchor_terms: list[str]` field (extra, allowed by `extra="allow"`) listing 3-5 terms a good summary would contain — for future P2's judge prompts to use; not validated structurally in this slice.

#### `goldens/multi_topic.jsonl` — ≥8 rows

Per row: question that spans multiple sections WITHIN a single doc (e.g. "Walk through the UART bring-up: NCO config, FIFO watermark, interrupts"). `expected_source_docs` lists the one doc; `expected_answer_span` may be omitted; `summary_anchor_terms` (extra) lists the section-spanning anchors.

#### `goldens/multi_aspect.jsonl` — ≥8 rows

Per row: question requiring synthesis across 2+ docs (e.g. RISC-V CSR concept applied to Ibex's specific config). `expected_source_docs` lists 2+ docs. Omit `expected_answer_span`. Include `summary_anchor_terms` (extra).

#### `goldens/adversarial.jsonl` — ≥5 rows

Plausible-but-unanswerable: questions the corpus does NOT answer but might appear to (e.g. "Does the OpenTitan UART support I2C mode?", "What is Ibex's fmax?"). Per row: `qid`, `qtype: "adversarial"`, `query`. Add an extra field `expected_behavior: "refuse_or_state_unknown"`. `expected_source_docs: []`. Omit `expected_answer_span`.

#### `goldens/out_of_corpus.jsonl` — ≥5 rows

Clearly out of scope (ARM Cortex specifics, x86 instructions, PowerPC, etc.). Same schema as adversarial. `expected_behavior: "refuse_or_state_unknown"`.

#### `goldens/messy.jsonl` — ≥5 rows

Typoed/terse/jargon-heavy variants of factoid queries. Per row: `qid`, `qtype: "messy"`, `query` (the messy form), `expected_answer_span` (a verbatim doc span — same grounding rule as factoid), `expected_source_docs` (the doc with the span). Add an extra `normalized_query: str` field giving the cleaned-up form.

### Test (`tests/eval/test_opentitan_goldens.py`)

The 6 tests enumerated in `validable_outcome`. Each test reads the goldens directly from the repo (not via `load_pack`, except test 1 which uses `validate_pack`) so failure messages can pinpoint qid + path. Loading verbatim spans verbatim: do an in-Python substring check against the corpus doc file content. NO mocking, NO test-helper indirection.

## Constraints

- Boundary discipline: ONLY the 9 files in `touches`. Do NOT modify pack.yaml (corpus_pin unchanged — no doc content changes), validator code, schema code, or any other tests. The P1 ingest test `tests/eval/test_opentitan_pack_ingest.py` MUST stay green unmodified.
- corpus_pin is NOT affected by goldens edits (it hashes manifest entries, not goldens). Sanity check: rerunning `validate_pack` after this slice must NOT emit a pin mismatch.
- No new runtime deps.
- No tests skipped, xfailed, or marker-softened.
- No drive-by edits to `_example_` pack.
- Author goldens in deterministic order: factoids by source doc then by appearance order in the doc; qfs/multi_topic/multi_aspect by doc; adversarial/out_of_corpus/messy in any sensible order.

## Anti-gaming guards

- Test 4 (`test_factoid_spans_are_verbatim_in_source_docs`) is the load-bearing grounding check. It MUST be a literal `if span not in doc_content: fail(qid, path)` — NOT a regex, NOT a substring lowercased, NOT a "starts-with" or "contains some words" weakening. Verbatim means verbatim.
- Test 5 reads `corpus/manifest.json` to build the source-doc allowlist; do NOT hardcode the doc list in test code (that would let the test pass on a renamed doc).
- The factoid file MUST cover ≥4 of the 5 corpus docs (no concentration in a single easy doc). Add `test_factoid_doc_coverage` if SA2 wants extra strength; otherwise leave as a soft authoring rule called out here.
- Red-reason proof: BEFORE editing thresholds.yaml or any goldens file, run `uv run pytest tests/eval/test_opentitan_goldens.py -q` (after creating the test file). Expected failure: either `FileNotFoundError` on a goldens file not yet created, OR `AssertionError` because count below minimum. NOT an ImportError. Capture verbatim into `.delivery/plan/wp-P2-opentitan-goldens-red-proof.txt`.

## Out of scope (defer)

- Authoring `command` qtype goldens — P6.5 / P6.5.0.
- Judge prompts that consume `summary_anchor_terms` — P5.
- Runner that ingests the pack and computes metrics — P3+.
- Bumping `_example_` pack to also declare `min_goldens_per_qtype` — keep example as the minimal-shape demo.
- Any retrieval/ingest code changes.

## Agent green-gate vs. human-review checkpoint

- **Agent green-gate (this slice ends here):** 6 P2 tests green, regression sweep green, commit on `feat/p2-opentitan-goldens`.
- **Human-review checkpoint (separate, unblocks P3):** subject-matter reviewer skims a sample of goldens, confirms questions read sensibly and spans are non-trivial. Out of scope for slice DoD.
