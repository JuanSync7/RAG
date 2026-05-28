---
slice_id: P2.0-golden-quantity-gates
validable_outcome: |
  `uv run pytest tests/eval/test_pack_min_counts.py -q` reports all tests green, including:
    1. `test_validator_rejects_undersized_pack`: a synthetic pack (built by copytree from `evals/packs/_example_`, with `thresholds.yaml` extended to declare
       `min_goldens_per_qtype: {factoid: 20, qfs: 10, multi_topic: 8, multi_aspect: 8, adversarial: 5, out_of_corpus: 5, messy: 5}`
       and goldens containing exactly ONE valid row per qtype) raises `PackValidationError` from `validate_pack`. The error message MUST:
       - reference the key `min_goldens_per_qtype`,
       - list every deficit qtype with its declared minimum AND actual count (so a reader can see "factoid: declared 20, found 1"),
       - cover ALL seven qtypes above (none silently skipped).
       The error must NOT be a bare `pydantic.ValidationError`; it must be `PackValidationError`.
    2. `test_validator_accepts_when_min_counts_unset`: the existing `evals/packs/_example_` pack (whose `thresholds.yaml` does NOT declare `min_goldens_per_qtype`) continues to load cleanly via `load_pack`. Backward-compatible default = no enforcement.
    3. `test_validator_accepts_when_counts_met`: a pack whose goldens contain ≥ declared minimum per qtype loads cleanly. Build by replicating the single example golden N+1 times per qtype with unique qids.
    4. `test_thresholds_schema_exposes_min_goldens_field`: `Thresholds(profile="generic", min_goldens_per_qtype={"factoid": 20}).min_goldens_per_qtype == {"factoid": 20}` and defaults to `{}` when unset.
  PLUS: `uv run pytest tests/eval/ tests/vector_db/ tests/retrieval/ -q -m "not slow and not integration"` stays fully green (no regressions on P0/P0.5).
touches:
  - src/eval/pack/schema.py
  - src/eval/pack/validate.py
  - tests/eval/test_pack_min_counts.py
depends_on:
  - P0.5-eval-pack-format
---

# P2.0 — Golden quantity gates (precondition micro-slice)

## End-state (validable, machine-checkable)

The P0.5 validator gains a per-qtype minimum-count gate driven by a NEW
`thresholds.yaml` key `min_goldens_per_qtype: dict[str, int]`. When set,
the validator counts goldens loaded under each qtype and raises
`PackValidationError` listing every deficit. When unset (the existing
`_example_` pack), the validator behaves as before — fully backward
compatible. No goldens are authored in this slice; the OpenTitan
`thresholds.yaml` pinning happens in a sibling pack-update, not here.

## Current state (surveyed before slice opens)

- `Thresholds` in `src/eval/pack/schema.py:79-83` has fields `profile`, `defaults: dict[str, dict[str, float]]`, `overrides: list[dict]`. **No `min_goldens_per_qtype` field exists.** `ConfigDict(extra="allow")` means undeclared YAML keys would parse into `__pydantic_extra__` rather than raise, but they would also be invisible to validator logic. We must add an explicit field.
- `validate_pack` in `src/eval/pack/validate.py:68-180` runs the goldens-file scan, then loads `thresholds.yaml`. The new gate plugs in AFTER goldens are parsed (so we know counts per qtype) and AFTER `Thresholds` is constructed (so we have the declared minimums).
- `evals/packs/_example_/thresholds.yaml` does NOT declare `min_goldens_per_qtype`. Backward-compat default of `{}` keeps it green.
- `validate_pack` returns `None`; the count-aware information needed for the new test is already present in-validator at the point of the gate. No re-walk of `goldens/` directory needed — reuse the in-flight count.

## Plan-spec section 10.0 alignment

Concrete numeric targets pinned here for the slice's red test (synthetic pack), NOT for the OpenTitan pack (that pinning is the start of P2 and is *not* a P2.0 deliverable):
`factoid ≥ 20, qfs ≥ 10, multi_topic ≥ 8, multi_aspect ≥ 8, adversarial ≥ 5, out_of_corpus ≥ 5, messy ≥ 5`.

The slice does NOT touch `evals/packs/_example_/thresholds.yaml` (that would force the example pack to author 56+ goldens — out of scope, and not the keystone's job). The targets live in the synthetic in-test thresholds copy.

## Required deliverables

### Code

1. **`src/eval/pack/schema.py`** — extend `Thresholds`:
   - Add `min_goldens_per_qtype: dict[str, int] = Field(default_factory=dict)`.
   - Field is optional with default `{}` (backward-compat).
   - No other changes to schema.py.

2. **`src/eval/pack/validate.py`** — extend `validate_pack`:
   - While scanning goldens (existing loop), track counts per qtype: `counts_by_qtype: dict[str, int]`.
   - After `Thresholds` construction, if `thresholds.min_goldens_per_qtype` is non-empty, compute deficits:
     - For each `(qtype, declared_min)` in declared minimums, `actual = counts_by_qtype.get(qtype, 0)`.
     - If `actual < declared_min`, record the deficit.
   - If any deficits, raise `PackValidationError` with a message that:
     - Names the key `min_goldens_per_qtype`,
     - Lists every deficit in the form `<qtype>: declared <N>, found <M>` (deterministic order: sorted by qtype),
     - Joins deficits with `; `.

### Tests (`tests/eval/test_pack_min_counts.py`)

Four tests enumerated in `validable_outcome`. Build malformations via `tmp_path` + `shutil.copytree` from `evals/packs/_example_/`, then surgically rewrite `thresholds.yaml` and/or add JSONL rows. Each test uses ONLY the public API (`load_pack`, `validate_pack`, `PackValidationError`, `Thresholds`). For the in-tmp pack the corpus_pin recompute is unaffected (the pack body is unchanged); thresholds.yaml is the only mutated file.

## Constraints

- Boundary discipline: ONLY the three files in `touches` may be modified or created.
- No new runtime deps.
- `_example_` pack stays unmodified — backward-compat is part of the proof.
- No tests skipped, xfailed, or marker-softened.
- No changes to `errors.py` or `loader.py`.

## Anti-gaming guards

- Tests must observe behavior through `load_pack` / `validate_pack`, not by importing internal helpers.
- The undersized-pack test MUST construct the malformation in-test from a copy of `_example_` — NOT by hand-authoring a fixture YAML in the repo.
- The error-message assertions must check for BOTH the key name (`min_goldens_per_qtype`) AND ≥1 deficit qtype name AND the actual/declared count tuple, so a stub `raise PackValidationError("min_goldens_per_qtype: bad")` would not pass.
- Red-reason proof: before any implementation, run `uv run pytest tests/eval/test_pack_min_counts.py -q`. Expected failure modes (in priority order): `AttributeError: 'Thresholds' object has no attribute 'min_goldens_per_qtype'` OR `AssertionError` because no `PackValidationError` was raised (the validator currently doesn't enforce counts). NOT an `ImportError` (the public API already exists). Capture verbatim in closeout.

## Out of scope (defer)

- Pinning `min_goldens_per_qtype` into `evals/packs/opentitan_riscv/thresholds.yaml` — P2.
- Authoring real goldens — P2.
- Per-qtype maximums / parity checks — not planned.
- `command` qtype targets — P6.5.0.

## Agent green-gate vs. human-review checkpoint

- **Agent green-gate (this slice ends here):** four tests green, regression sweep green, commit on `feat/p2.0-golden-quantity-gates`.
- **Human-review checkpoint (separate, unblocks P2):** reviewer skims the deficit-listing error message and confirms it reads sensibly. Out of scope for slice DoD.
