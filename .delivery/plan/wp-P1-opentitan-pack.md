---
slice_id: P1-opentitan-pack
validable_outcome: |
  Three named tests pass against the new pack + ingest script:

    1. `tests/eval/test_opentitan_pack_ingest.py::test_pack_validates_under_p0_5_loader` (fast):
       - `load_pack("evals/packs/opentitan_riscv")` returns an `EvalPack` (P0.5 type).
       - `meta.name == "opentitan_riscv"`, `meta.profile == "asic_riscv_soc"`, `meta.version == 1`.
       - `len(manifest) >= 3` (multi-doc corpus, not a single-file toy).
       - Recomputed `corpus_pin` from manifest equals `meta.corpus_pin` (no tautological pin assertion).
       - `goldens == {}` (this slice declares no goldens; P2 owns them).
       - `collection_name` resolves to `ragweave_test_opentitan_riscv_<first-8-of-corpus_pin>`.

    2. `tests/eval/test_opentitan_pack_ingest.py::test_eval_ingest_script_dry_run` (fast):
       - `python scripts/eval_ingest.py --pack opentitan_riscv --dry-run` exits 0.
       - stdout is JSON containing keys: `pack`, `collection`, `manifest_pin`, `files_to_ingest`.
       - `collection` equals the resolved `collection_name` from test #1.
       - `files_to_ingest` is a list whose length equals `len(manifest)`.
       - No Weaviate writes occur (verified by absence of network calls: dry-run path must not import or call the real ingest engine).

    3. `tests/eval/test_opentitan_pack_ingest.py::test_ingest_into_isolated_collection`
       (markers: `[pytest.mark.slow, pytest.mark.integration]`):
       - Requires live Weaviate. Skips cleanly if `WEAVIATE_URL` is unreachable.
       - Captures `default_count_before` = chunk count in `VECTOR_COLLECTION_DEFAULT`.
       - Runs `python scripts/eval_ingest.py --pack opentitan_riscv` (no `--dry-run`) — must complete with exit code 0.
       - Asserts the resolved target collection exists (`collection_exists`) and has `chunk_count > 0`.
       - Asserts `default_count_after == default_count_before` (zero leakage into the prod collection).
       - try/finally cleanup deletes the target collection on both pass and fail paths.

  PLUS: `uv run pytest tests/eval/ tests/vector_db/ tests/retrieval/ -q -m "not slow and not integration"` stays fully green (no regressions in P0/P0.5 tests).
touches:
  - evals/packs/opentitan_riscv/pack.yaml
  - evals/packs/opentitan_riscv/thresholds.yaml
  - evals/packs/opentitan_riscv/README.md
  - evals/packs/opentitan_riscv/corpus/manifest.json
  - evals/packs/opentitan_riscv/corpus/docs/riscv_priv_isa_intro.md
  - evals/packs/opentitan_riscv/corpus/docs/riscv_priv_csrs.md
  - evals/packs/opentitan_riscv/corpus/docs/opentitan_uart.md
  - evals/packs/opentitan_riscv/corpus/docs/opentitan_hmac.md
  - evals/packs/opentitan_riscv/corpus/docs/ibex_overview.md
  - scripts/eval_ingest.py
  - tests/eval/test_opentitan_pack_ingest.py
depends_on:
  - P0-collection-selection
  - P0.5-eval-pack-format
---

# P1 — OpenTitan/Ibex/RISC-V reference pack (corpus + manifest only)

## End-state (validable, machine-checkable)

A real, conforming eval_pack named `opentitan_riscv` lives at `evals/packs/opentitan_riscv/`, drawn from public ASIC/RISC-V documentation. `scripts/eval_ingest.py` loads it via P0.5's loader, resolves the per-pack collection name via the template, and ingests the corpus into an **isolated** Weaviate collection using P0's `collection_name` plumbing. The prod collection (`RAGDocuments`) is provably unaffected.

No goldens are authored in this slice (P2's job). The keystone proof for P1 is *content/loop separation* — does a real corpus flow through the same runtime by declaring a pack, without any code change inside `src/ingest/` or `src/retrieval/`?

## Current state (surveyed before slice opens; SA1 will confirm)

- `evals/packs/opentitan_riscv/` does NOT exist. Only `_example_/` from P0.5 lives under `evals/packs/`.
- `scripts/eval_ingest.py` does NOT exist. The `scripts/` directory has soak/ops scripts but no eval-runner entry point.
- `tests/eval/test_opentitan_pack_ingest.py` does NOT exist.
- P0 plumbing (`collection_name` through `RAGChain`, ingest CLI `--collection`, `collection_exists` helper) is committed on the parent branch.
- P0.5 loader (`load_pack`, `EvalPack`, `PackValidationError`) is committed on the parent branch and importable from `src.eval.pack`.
- The existing ingest entry point is `python -m src.ingest.cli` with a `--collection` flag (added in P0). `scripts/eval_ingest.py` SHOULD shell out to that or invoke its programmatic equivalent — DO NOT duplicate the ingest pipeline.

## Corpus authoring rules (read carefully)

The corpus is **real, public ASIC/RISC-V documentation excerpted into 5 small Markdown files** (≤ 4 KB each). It is a *development-grade* representative subset, not the full canonical corpus — the full corpus arrives in a later slice that wires submodule pinning. This is explicit and is documented in `evals/packs/opentitan_riscv/README.md`.

Required document set (exact filenames, exact provenance):

| File | Topic | Source (public) |
|---|---|---|
| `corpus/docs/riscv_priv_isa_intro.md` | RISC-V Privileged ISA — privilege levels overview | RISC-V Privileged ISA spec, "Privilege Levels" section |
| `corpus/docs/riscv_priv_csrs.md` | RISC-V Privileged ISA — selected CSRs (mstatus, mtvec, mepc) | Same spec, CSR section |
| `corpus/docs/opentitan_uart.md` | OpenTitan UART IP — register map summary | OpenTitan UART HWIP documentation |
| `corpus/docs/opentitan_hmac.md` | OpenTitan HMAC IP — block diagram and key registers | OpenTitan HMAC HWIP documentation |
| `corpus/docs/ibex_overview.md` | Ibex core — pipeline + ISA support summary | lowRISC Ibex documentation |

Content rules:
- Each file is hand-authored prose paraphrasing the public spec (you may use `WebFetch` to read the public source if needed). Verbatim copying of long passages is not necessary — terse, accurate technical summaries are sufficient for ingest-pipeline exercise.
- Each file starts with a level-1 heading equal to the topic in the table.
- Each file is ≤ 4 KB. If you exceed this, trim — the slice is about plumbing, not corpus depth.
- Provenance is captured in `README.md` (per-doc source link), NOT inside each Markdown file (we want clean ingest content).
- Do NOT add HTML, frontmatter, or markdown extensions Docling cannot parse. Plain Markdown with headings, lists, and code blocks only.

## Required deliverables

### Pack (`evals/packs/opentitan_riscv/`)

1. `pack.yaml` per §4.2 of EVAL_LOOP_PLAN.md:
   - `name: opentitan_riscv`, `version: 1`, `profile: asic_riscv_soc`.
   - `corpus_pin: <sha256-of-sorted-manifest-lines>` — computed by the slice script, not hand-guessed.
   - `judge:` stub block matching the keystone's `JudgeConfig` shape (tier1_model: `claude-haiku-4-5-20251001`, tier1_prompt_version: v1, temperature: 0, samples_per_claim: 3).
   - `collection_name_template: "ragweave_test_{name}_{corpus_pin_short}"`.
   - `description:` 1-line summary citing the three source projects.
2. `corpus/manifest.json` — JSON array of `{ path, sha256 }` entries, one per Markdown doc, sorted by `path`.
3. `corpus/docs/*.md` — five files per the table above.
4. `thresholds.yaml` — `profile: asic_riscv_soc`, `defaults.factoid.recall_at_5: 0.8`, `defaults.factoid.mrr: 0.6`. No per-qtype overrides yet; P2.0 owns count-gate additions.
5. `README.md` — provenance table + a clear "development-grade subset, not the canonical corpus" disclaimer.

### `scripts/eval_ingest.py`

A new script — invokable as `python scripts/eval_ingest.py --pack <name> [--dry-run]`.

Required behavior:
- `argparse`: `--pack` (required), `--dry-run` (flag), `--pack-root` (optional, defaults to `evals/packs/`).
- Loads the pack via `src.eval.pack.load_pack`. If `PackValidationError` is raised, prints the error to stderr and exits with code 2.
- Resolves `target_collection = pack.collection_name` (uses the P0.5-computed template substitution).
- In `--dry-run` mode: emits a single JSON object on stdout with `{pack, collection, manifest_pin, files_to_ingest: [<paths>]}` and exits 0. **Does not import the ingest engine.** This is the anti-coupling guard for fast tests.
- In live mode: invokes the existing ingest pipeline with `collection_name=target_collection` for every file in `corpus/manifest.json`. The simplest correct path is `subprocess.run([sys.executable, "-m", "src.ingest.cli", "--collection", target_collection, <doc_path>], check=True)` per file, OR (preferred if straightforward) the programmatic equivalent from `src.ingest.impl`. Either is acceptable; the test asserts behavior, not invocation style.
- Exits 0 on full success, 1 on partial ingest failure (with stderr summary).

### Tests (`tests/eval/test_opentitan_pack_ingest.py`)

The three tests enumerated in `validable_outcome`. Live test gating:

```python
import pytest
pytestmark = []  # per-test markers below to keep fast tests fast

# On the live test only:
@pytest.mark.slow
@pytest.mark.integration
def test_ingest_into_isolated_collection(...):
    ...
```

Live test must:
- Use a uniqueified collection-name override (UUID suffix appended to the template-resolved name) so concurrent runs do not collide. Achieve this by passing `--pack-root tmp_path/packs` after copying the pack into `tmp_path` and rewriting `pack.yaml`'s `name` field to `opentitan_riscv_test_<uuid_hex_8>`. The corpus_pin will recompute identically (same content).
- Read counts via `client.collections.get(c).aggregate.over_all(total_count=True).total_count`, with a 5s settle loop for indexer consistency.
- `try/finally` clean up the test collection on both pass and fail paths.
- Skip cleanly if `WEAVIATE_URL` is set but unreachable — do NOT fail the suite on missing infra.

## Constraints

- **No edits inside `src/ingest/`, `src/retrieval/`, `src/vector_db/`.** Content-on-loop separation is the keystone proof of P1. If you find yourself needing a fix there, STOP and report it — that is a P1.1 follow-up slice, not in-scope.
- **Reuse P0.5 loader; do NOT re-parse the pack independently.** `scripts/eval_ingest.py` imports `load_pack`.
- **No new env var names.** The collection name comes from the pack template — not from `RAG_EVAL_COLLECTION_NAME` or any new env var.
- **Live test gating: dual markers.** Both `pytest.mark.slow` AND `pytest.mark.integration` per memory `feedback_dual_marker_gating`. Single-marker gating slips through `-m "not slow"`.
- **No test softening.** No skips, no xfails, no new markers added to `pyproject.toml`.
- **Boundary discipline.** All edits confined to `touches`.

## Anti-gaming guards

- The `test_pack_validates_under_p0_5_loader` test must recompute `corpus_pin` from the manifest in-test and compare against `pack.yaml`'s declared value — NOT hardcode a literal hex string.
- The `test_eval_ingest_script_dry_run` test must assert that the dry-run path produces NO Weaviate writes. The simplest proof: count chunks in `VECTOR_COLLECTION_DEFAULT` before and after the dry-run; delta must be 0. (This implicitly requires live Weaviate; if you want the dry-run test fast, instead assert that `scripts/eval_ingest.py` in dry-run mode does not import `weaviate` or `src.vector_db` — verify via `subprocess` capture of `sys.modules` keys, or via a structural check that the dry-run code path is reachable without those imports. **Pick one and document the choice.**)
- The `test_ingest_into_isolated_collection` test must read chunk counts via the real Weaviate client aggregation API — not via any helper the slice just authored. Use `client.collections.get(c).aggregate.over_all(total_count=True).total_count` directly.
- **Red-reason proof:** before any implementation, run `uv run pytest tests/eval/test_opentitan_pack_ingest.py -q -m "not slow and not integration"`. Failure output MUST include one of:
  - `FileNotFoundError` on `evals/packs/opentitan_riscv/pack.yaml` (pack absent),
  - `ModuleNotFoundError` / `FileNotFoundError` on `scripts/eval_ingest.py` (script absent), OR
  - `PackValidationError` (pack present but malformed mid-authoring).
  It must NOT be an `ImportError` on `src.eval.pack` (that would mean P0.5 broke; halt and escalate), and it must NOT be an `AttributeError` on `EvalPack` fields (means the loader API changed; out of scope here). Capture the failure output verbatim in the slice closeout.

## Out of scope (defer)

- Goldens authoring — P2.
- Submodule-pinned full OpenTitan corpus — separate follow-up slice once development-grade pack proves the loop.
- `RAG_EVAL_COLLECTION_NAME` env override — explicitly listed as P0-out-of-scope and stays out of scope.
- Eval runner that scores queries — P4 (factoid runner) and onward.
- Auto-GC of `ragweave_test_*` collections — operational slice, separate.

## Agent green-gate vs. human-review checkpoint

- **Agent green-gate (this slice ends here):** the three named tests pass (live test against live Weaviate; fast tests in any env); regression sweep stays green; commit lands on `feat/p1-opentitan-pack`.
- **Human-review checkpoint (separate, unblocks P2; not P1):** human reviewer skims the five corpus documents for accuracy and confirms attribution in `README.md` is correct. This is judgment, not a machine gate, and is explicitly excluded from this slice's DoD per §10.0 of the plan.
