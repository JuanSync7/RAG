---
slice_id: P3.0-pack-ingest-plan
validable_outcome: |
  `uv run pytest tests/eval/test_pack_ingest_plan.py -q` reports all tests green, including:
    1. `test_plan_resolves_opentitan_pack`: `plan_pack_ingest(pack, pack_dir)` for the loaded `evals/packs/opentitan_riscv` pack returns an `IngestPlan` with:
       - `pack_name == "opentitan_riscv"`,
       - `collection_name == pack.collection_name` (which equals `"ragweave_test_opentitan_riscv_<first-8-chars-of-corpus_pin>"`),
       - `documents_dir == pack_dir / "corpus"`,
       - `len(doc_paths) == len(pack.manifest) == 5`,
       - `expected_doc_count == 5`,
       - `corpus_pin == pack.meta.corpus_pin`.
    2. `test_plan_doc_paths_preserve_manifest_order`: `plan.doc_paths[i]` equals `pack_dir / "corpus" / pack.manifest[i].path` for every i. Order is preserved exactly.
    3. `test_plan_works_for_example_pack`: planning the `evals/packs/_example_` pack also succeeds and returns a 1-doc plan with `pack_name == "_example_"`.
    4. `test_plan_is_frozen`: `IngestPlan` is immutable — assigning to `plan.collection_name` raises `dataclasses.FrozenInstanceError` (or pydantic `ValidationError` if a pydantic frozen model is used instead).
    5. `test_plan_collection_name_template_resolved`: `plan.collection_name` matches the `pack.meta.collection_name_template` rendered with `name` + `corpus_pin_short` (rebuilt in-test, not hardcoded).
    6. `test_plan_doc_paths_all_exist`: every `Path` in `plan.doc_paths` exists on disk (the OpenTitan pack ships all 5 docs).
  PLUS: `uv run pytest tests/eval/ tests/vector_db/ tests/retrieval/ -q -m "not slow and not integration"` stays fully green.
touches:
  - src/eval/__init__.py
  - src/eval/runner/__init__.py
  - src/eval/runner/plan.py
  - tests/eval/test_pack_ingest_plan.py
depends_on:
  - P2-opentitan-goldens
  - P0.5-eval-pack-format
---

# P3.0 — Pack ingest plan (pure-logic precondition for P3)

## End-state (validable, machine-checkable)

A new pure-logic helper `plan_pack_ingest(pack: EvalPack, pack_dir: Path) -> IngestPlan` lives at `src/eval/runner/plan.py`. It maps an `EvalPack` into a frozen dataclass whose field names match the parameter names of the real `ingest_directory` entrypoint, so P3 can call ingest cleanly. Zero Weaviate dependency. Zero CLI invocation.

## Current state (surveyed by SA1)

- `src/eval/runner/` does NOT exist. New subpackage.
- `tests/eval/` exists from P0.5; will add `test_pack_ingest_plan.py`.
- `src/ingest/impl.py:680` exposes `ingest_directory(documents_dir, config, fresh=False, update=True, selected_sources=None, batch_id="")`. Collection routing is via `config.target_collection`. The runner (P3) will construct an `IngestionConfig` with `target_collection=plan.collection_name`. P3.0 produces the plan only; it does NOT import from `src/ingest/`.
- `EvalPack.collection_name` (P0.5, `src/eval/pack/schema.py:99-105`) resolves `pack.meta.collection_name_template` with `{name}` + `{corpus_pin_short}` (first 8 hex chars of `corpus_pin`).
- `EvalPack.manifest: list[ManifestEntry]` where each entry has `.path` (relative under `corpus/`).
- No existing manifest→doc-paths helper anywhere. This slice introduces it as a side-effect of `plan_pack_ingest`.

## Plan-spec alignment

P3.0 is the precondition micro-slice for P3. P3 will be the first slice with a real Weaviate dependency in the runner path. P3.0 lays the deterministic mapping between `EvalPack` and `ingest_directory` parameters so P3's live-ingest test can be small and focused (just "call planner → call ingest → count rows == expected_doc_count").

## Required deliverables

### Code

1. **`src/eval/runner/__init__.py`** — minimal public surface. Export `IngestPlan`, `plan_pack_ingest`. Add `__all__`.

2. **`src/eval/runner/plan.py`** — implementation:
   - `@dataclass(frozen=True)` `IngestPlan` with fields:
     - `pack_name: str`
     - `collection_name: str`
     - `documents_dir: Path`
     - `doc_paths: tuple[Path, ...]` (tuple, not list — required for frozen dataclass to hash and to discourage mutation)
     - `expected_doc_count: int`
     - `corpus_pin: str`
   - `plan_pack_ingest(pack: EvalPack, pack_dir: Path) -> IngestPlan`:
     - `pack_dir` is the directory containing `pack.yaml`. The function uses `pack_dir / "corpus"` for `documents_dir` and joins each `pack.manifest[i].path` under `documents_dir` to build `doc_paths`.
     - `collection_name = pack.collection_name` (reuse the property, do not duplicate the template logic).
     - `expected_doc_count = len(pack.manifest)`.
     - `corpus_pin = pack.meta.corpus_pin`.
     - Returns the frozen `IngestPlan`. No I/O beyond what the caller already did to construct `pack`.
   - **No** imports from `src/ingest`, `src/vector_db`, or `src/retrieval`. Pure logic only.
   - **No** validation of on-disk existence inside the planner — that's `validate_pack`'s job. The planner trusts a loaded pack. The test asserts existence externally (test 6), not the planner.

3. **`src/eval/__init__.py`** — add `IngestPlan` and `plan_pack_ingest` to the public surface alongside the P0.5 exports. Keep `__all__` sorted.

### Tests (`tests/eval/test_pack_ingest_plan.py`)

Six tests enumerated in `validable_outcome`. Use `load_pack` to construct the input pack. Pack dir is `Path("evals/packs/opentitan_riscv")` resolved against the project root. For test 5, rebuild `collection_name` in-test by reading `pack.meta.collection_name_template` and calling `.format(name=..., corpus_pin_short=...)` — DO NOT hardcode `"ragweave_test_opentitan_riscv_a8e3ff90"`.

For test 4 (frozen check), `import dataclasses` and `with pytest.raises(dataclasses.FrozenInstanceError): plan.collection_name = "x"`.

## Constraints

- Boundary discipline: ONLY the 4 files in `touches`.
- No new runtime deps. Stdlib `dataclasses`, `pathlib` only.
- No Weaviate / vector_db / ingest imports — the planner is pure logic, decoupled from infrastructure. This is the WHOLE point of P3.0.
- No mutation of `_example_` or `opentitan_riscv` pack contents.
- No tests skipped, xfailed, or marker-softened.
- Use `tuple[Path, ...]` (not `list[Path]`) for `doc_paths` so the frozen dataclass is fully immutable.

## Anti-gaming guards

- The planner must NOT import from `src/ingest` even by mistake. Add an explicit `tests/eval/test_pack_ingest_plan.py` test (folded into the file, but written as a small extra check) that does `import sys; src.eval.runner.plan in sys.modules and "src.ingest" not in <set of imported submodules referenced by plan.py>` — OR simpler: SA2 may inspect `src/eval/runner/plan.py` source as a string and `assert "src.ingest" not in source and "weaviate" not in source.lower()`. Pick whichever you implement; report which.
- Test 5 must reconstruct the template — hardcoding the expected string would be a tautology.
- Red-reason proof: before any implementation, run `uv run pytest tests/eval/test_pack_ingest_plan.py -q`. Expected: `ModuleNotFoundError: No module named 'src.eval.runner'` (or equivalent ImportError on `IngestPlan` / `plan_pack_ingest`). NOT a FileNotFoundError on the example pack. Capture verbatim into `.delivery/plan/wp-P3.0-pack-ingest-plan-red-proof.txt`.

## Out of scope (defer)

- Any actual ingest invocation, live or mocked — P3.
- A `runner.execute(plan)` function — P3.
- Pack-level concurrency / multi-pack queuing — not planned.
- Replacing `scripts/eval_ingest.py`'s subprocess fan-out with a direct call — P3+.
- Adding `--collection` to `src/ingest/cli` if it's actually missing — out of scope (SA1 flagged the CLI flag may not exist; verify and document but don't fix here).

## Agent green-gate vs. human-review checkpoint

- **Agent green-gate (this slice ends here):** 6 P3.0 tests green, regression sweep green, commit on `feat/p3.0-pack-ingest-plan`.
- **Human-review checkpoint:** the IngestPlan field names match `ingest_directory`'s parameter names so P3 can call cleanly. Verified by SA1; reaffirmed by SA3.
