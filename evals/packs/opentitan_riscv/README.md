<!-- @summary
Development-grade ASIC/RISC-V reference pack: provenance, content, and disclaimer.
@end-summary -->

# opentitan_riscv eval_pack

A small, hand-authored reference pack covering three public ASIC/RISC-V
projects:

- **RISC-V Privileged ISA** (Privilege Levels overview + selected CSRs)
- **OpenTitan** UART and HMAC IPs (register-map summaries)
- **lowRISC Ibex** core (pipeline + ISA support summary)

This pack is the P1 keystone artifact: it proves that the content/loop
separation holds — a real corpus flows through the existing ingest
pipeline by declaring a pack, with **no code change in `src/ingest/`,
`src/retrieval/`, or `src/vector_db/`**.

## Disclaimer — development-grade subset

> **The corpus in this pack is NOT the full canonical OpenTitan / Ibex
> / RISC-V documentation.** It is a small (~13 KB total, 5 files),
> hand-authored development-grade subset assembled from publicly
> available specifications. A later slice will wire submodule-pinned
> upstream sources for full coverage. Do not rely on this pack for
> production grading or coverage claims.

## Document provenance

| File | Topic | Source (public) |
|---|---|---|
| `corpus/docs/riscv_priv_isa_intro.md` | RISC-V Privileged ISA — privilege levels overview | [RISC-V Privileged Architectures Spec](https://github.com/riscv/riscv-isa-manual), "Privilege Levels" section |
| `corpus/docs/riscv_priv_csrs.md` | RISC-V Privileged ISA — selected CSRs (mstatus, mtvec, mepc) | [RISC-V Privileged Architectures Spec](https://github.com/riscv/riscv-isa-manual), CSR sections |
| `corpus/docs/opentitan_uart.md` | OpenTitan UART IP — register map summary | [OpenTitan UART HWIP docs](https://opentitan.org/book/hw/ip/uart/) |
| `corpus/docs/opentitan_hmac.md` | OpenTitan HMAC IP — block diagram and key registers | [OpenTitan HMAC HWIP docs](https://opentitan.org/book/hw/ip/hmac/) |
| `corpus/docs/ibex_overview.md` | Ibex core — pipeline + ISA support summary | [lowRISC Ibex docs](https://ibex-core.readthedocs.io/) |

Each document is a terse paraphrase of the cited public source. None
contain verbatim long-form copies of the upstream text.

## Pack contents

- `pack.yaml` — metadata, judge config, collection-name template.
- `thresholds.yaml` — profile-default factoid thresholds (`recall_at_5=0.8`, `mrr=0.6`).
- `corpus/manifest.json` — sorted `{path, sha256}` entries pinned by
  `pack.yaml:corpus_pin`.
- `corpus/docs/*.md` — five Markdown files (≤ 4 KB each).
- `goldens/factoid.jsonl` — placeholder (intentionally empty). P2 owns
  golden-query authoring; this file exists only to satisfy the P0.5
  loader's structural contract.

## Profile

`asic_riscv_soc` — ASIC RISC-V SoC documentation profile.

## Collection naming

The collection name is computed by the P0.5 loader as:

```
ragweave_test_{name}_{corpus_pin_short}
```

i.e. `ragweave_test_opentitan_riscv_a8e3ff90` for the current pin. The
short suffix is the first 8 hex chars of `pack.yaml:corpus_pin`.

## Re-computing the corpus pin

```bash
uv run python -c "
import hashlib, json, pathlib
root = pathlib.Path('evals/packs/opentitan_riscv/corpus')
docs = sorted((root/'docs').glob('*.md'))
entries = [{'path': f'docs/{p.name}',
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
           for p in docs]
pin = hashlib.sha256('\n'.join(sorted(f\"{e['path']}:{e['sha256']}\" for e in entries)).encode()).hexdigest()
print(pin)
"
```
