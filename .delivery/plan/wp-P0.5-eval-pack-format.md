---
slice_id: P0.5-eval-pack-format
validable_outcome: |
  `uv run pytest tests/eval/test_pack_loader.py -q` reports all tests green, including:
    1. `test_load_valid_pack`: `load_pack("evals/packs/_example_")` returns a typed `EvalPack` dataclass with:
       - `name == "_example_"`, `version == 1`, `profile == "asic_riscv_soc"`.
       - `corpus_pin` equals the SHA-256 hex digest of the sorted (path, content-sha256) lines in `corpus/manifest.json`.
       - `goldens["factoid"]` is a non-empty list of typed `Golden` records (≥1 example), each parsed from the qtype's JSONL.
       - `thresholds.defaults["factoid"]["recall_at_5"]` is a float between 0 and 1.
       - `collection_name` resolves the `pack.yaml` template `ragweave_test_{name}_{corpus_pin_short}` with the first 8 chars of `corpus_pin`.
    2. `test_rejects_missing_manifest`: loading a pack copy with `corpus/manifest.json` removed raises `PackValidationError` whose message contains `"corpus/manifest.json"`. Error type is NOT `FileNotFoundError`, NOT a bare `KeyError`.
    3. `test_rejects_schema_invalid_golden`: a pack copy where one `factoid.jsonl` line has `qtype: "qfs"` (mismatch) and is missing `expected_answer_span` raises `PackValidationError` referencing the offending qid and the failing field. The error must NOT be a bare `pydantic.ValidationError` bubbled through — it must be caught and re-raised as `PackValidationError`.
    4. `test_rejects_unknown_profile`: a pack copy whose `pack.yaml` declares `profile: undeclared_profile` (not in the validator's known-profile set) raises `PackValidationError` whose message contains both `"profile"` and `"undeclared_profile"`.
  PLUS: `uv run pytest tests/eval/ tests/vector_db/ tests/retrieval/ -q -m "not slow and not integration"` stays fully green (no regressions; P0's existing tests untouched).
touches:
  - src/eval/__init__.py
  - src/eval/pack/__init__.py
  - src/eval/pack/schema.py
  - src/eval/pack/validate.py
  - src/eval/pack/loader.py
  - src/eval/pack/errors.py
  - evals/packs/_example_/pack.yaml
  - evals/packs/_example_/corpus/manifest.json
  - evals/packs/_example_/corpus/docs/intro.md
  - evals/packs/_example_/goldens/factoid.jsonl
  - evals/packs/_example_/thresholds.yaml
  - tests/eval/__init__.py
  - tests/eval/test_pack_loader.py
depends_on:
  - P0-collection-selection
---

# P0.5 — eval_pack format spec + validator (keystone)

## End-state (validable, machine-checkable)

The eval_pack format is implemented as code. A minimal example pack lives at `evals/packs/_example_/` and round-trips through `load_pack(path) -> EvalPack` (typed). The validator rejects three named malformations with `PackValidationError` whose messages name the offending location. No suite, no runner, no metric implementation in this slice — only the format keystone.

## Current state (surveyed before slice opens)

- `src/eval/` does NOT exist on `feat/p0-collection-selection` HEAD. Confirmed via `ls src/eval` (no such directory). The slice is starting from a clean keystone — no pre-existing pack code to integrate with.
- `tests/eval/` does NOT exist. New test directory.
- `evals/` does NOT exist. New top-level data directory (sibling of `src/`).
- P0's `collection_name` plumbing is in scope to be *consumed* (via `EvalPack.collection_name`) but the runner that uses it is out of scope for this slice.

## Plan-spec section 4 alignment

This slice implements §4.1 (directory layout), §4.2 (`pack.yaml` shape), §4.3 (golden query schema), §4.4 (`thresholds.yaml` shape — minimum keys only, no enforcement), §4.5 (validator). No `min_goldens_per_qtype` enforcement — that lands in P2.0. No prompts directory — optional per §4.1; the example pack omits it. No `expected_cited_chunks` / `command` qtype schemas — only `factoid` qtype is required for keystone proof; richer qtypes land in P2.

## Required deliverables

### Code (`src/eval/pack/`)

1. **`schema.py`** — typed dataclasses (or pydantic v2 models — pick one and stay consistent; pydantic v2 is already a project dependency):
   - `PackMeta`: `name`, `version: int`, `profile: str`, `corpus_pin: str`, `description: str | None`, `judge: JudgeConfig`, `collection_name_template: str`.
   - `JudgeConfig`: `tier1_model: str`, `tier1_prompt_version: str`, `temperature: float`, `samples_per_claim: int`.
   - `ManifestEntry`: `path: str`, `sha256: str`.
   - `Golden`: discriminated by `qtype` literal (start with `factoid`; allow other qtype strings to parse through as `Golden` with extra fields preserved on a `raw: dict` field so P2 can extend without re-authoring schema.py).
   - `Thresholds`: `profile: str`, `defaults: dict[str, dict[str, float]]`, `overrides: list[dict]` (loose-typed list is acceptable for keystone).
   - `EvalPack`: aggregate with `meta: PackMeta`, `manifest: list[ManifestEntry]`, `goldens: dict[str, list[Golden]]`, `thresholds: Thresholds`, and a computed `collection_name: str` property that resolves the template.
2. **`errors.py`** — `class PackValidationError(Exception)`. Single exception type for all validation failures; message must name (a) the file/field path inside the pack and (b) the reason.
3. **`validate.py`** — `validate_pack(path: Path | str) -> None` performs structural checks:
   - `pack.yaml` exists and parses; required keys present; `profile` is in a small known set `{asic_riscv_soc, eda_command_reference, generic}`.
   - `corpus/manifest.json` exists and parses; each entry has `path` + `sha256`.
   - `corpus_pin` in `pack.yaml` matches `sha256_hex(sorted "{path}:{sha256}" joined by "\n")` of the manifest.
   - Each goldens file in `goldens/` is JSONL; each line parses; each line's `qtype` matches the filename stem; each line passes per-qtype required-field checks (factoid: `qid`, `qtype`, `query`, `expected_answer_span`).
   - `thresholds.yaml` exists and parses; `profile` matches `pack.yaml.profile`.
   Any failure raises `PackValidationError` with a message that name-checks the offending path.
4. **`loader.py`** — `load_pack(path: Path | str) -> EvalPack`:
   - Calls `validate_pack(path)` first; lets `PackValidationError` propagate.
   - Loads + populates the dataclass tree.
   - Sets `EvalPack.collection_name` via template substitution (`{name}`, `{corpus_pin_short}` where short = first 8 chars).
5. **`__init__.py`** for `src/eval/` and `src/eval/pack/` — public exports: `EvalPack`, `load_pack`, `validate_pack`, `PackValidationError`. Add `__all__`.

### Example pack (`evals/packs/_example_/`)

Smallest pack that proves the loader works. Used by the validator test only — NOT a real corpus, NOT used by any runner.
- `pack.yaml` per §4.2 with `name: _example_`, `version: 1`, `profile: asic_riscv_soc`, judge block stub, and `collection_name_template: "ragweave_test_{name}_{corpus_pin_short}"`.
- `corpus/docs/intro.md` — single tiny doc (≤ 200 bytes).
- `corpus/manifest.json` — single entry pointing at `intro.md` with its real SHA-256.
- `corpus_pin` in `pack.yaml` — computed once and pinned. **The test must recompute it and assert equality.**
- `goldens/factoid.jsonl` — one valid line per §4.3 (qid, qtype="factoid", query, expected_answer_span, expected_source_docs).
- `thresholds.yaml` — minimum: `profile: asic_riscv_soc`, `defaults.factoid.recall_at_5: 0.8`.

### Tests (`tests/eval/test_pack_loader.py`)

The four tests enumerated in `validable_outcome`. Use `tmp_path` + `shutil.copytree` from the real example pack to build each malformation variant — DO NOT hand-write three full pack trees. This keeps the test about validator behaviour, not pack-authoring fidelity.

Also add `tests/eval/__init__.py` (empty) so pytest treats it as a package.

## Constraints

- **Default behavior unchanged for everything outside this slice.** P0's regression sweep stays green.
- **No new runtime dependencies.** Use `pyyaml` (already pinned), `pydantic` v2 (already pinned), stdlib `json`, `hashlib`, `pathlib`. If you reach for a new dep, stop and re-plan.
- **No coupling to ingest/retrieval code.** `src/eval/pack/` imports nothing from `src/ingest`, `src/retrieval`, `src/vector_db`. The pack is a content artifact + a parser; the runner that bridges to vector_db is a later slice.
- **Profile set is hardcoded for keystone.** `{asic_riscv_soc, eda_command_reference, generic}`. Extending later is a one-line change; do not over-engineer with a profiles plugin system.
- **No tests skipped or xfailed.** No new markers softened.
- **Boundary discipline.** Modifications confined to `touches`. No drive-by refactors to `src/eval/` siblings (there are none yet — keep it that way).

## Anti-gaming guards

- The four tests must observe behaviour through the **public** loader API (`load_pack`, `validate_pack`, `PackValidationError`) — not by importing and mock-patching internal validator helpers.
- The "valid pack" test must recompute `corpus_pin` from the manifest in-test and compare against `pack.yaml`'s declared value. Hardcoding the expected pin in the test body is a tautology and is forbidden.
- The "rejects schema-invalid golden" test must construct the malformation by editing a copy of the example pack, NOT by hand-authoring a JSONL that conveniently triggers the codepath the implementer just wrote.
- **Red-reason proof:** before any implementation in `src/eval/`, run `uv run pytest tests/eval/test_pack_loader.py -q`. Failure output MUST include `ModuleNotFoundError: No module named 'src.eval.pack'` (or equivalent on the public symbols `load_pack` / `PackValidationError`). It must NOT be a `FileNotFoundError` on the example pack (that would mean the example pack wasn't authored first), and it must NOT be an `ImportError` from the test file itself referencing a missing test helper. Capture the failure output verbatim in the slice closeout.

## Out of scope (defer)

- `min_goldens_per_qtype` enforcement — P2.0.
- Report-JSON schema (the `SuiteReport`) — P4.
- Pack-level git-submodule corpus pinning — P1 (this slice's example pack uses inline docs).
- QFS / multi-topic / command-reference golden schemas beyond keystone `factoid` — P2 / P6.5.
- Prompts directory + judge prompt overrides — P5.
- Migration tooling for `version` bumps — risk-list item, not this slice.

## Agent green-gate vs. human-review checkpoint

- **Agent green-gate (this slice ends here):** the four tests above are green; regression sweep stays green; commit lands on `feat/p0.5-eval-pack-format`.
- **Human-review checkpoint (separate, unblocks P1, not P0.5):** human reviewer skims `_example_/pack.yaml` + the validator error messages and confirms they read sensibly. This is judgment, not a machine gate, and is explicitly excluded from this slice's DoD per §10.0 of the plan.
