# Table-chunking soak — real datasheet (2026-05-21)

## Dataset

- PDF: `/home/kok-shew-juan/RagWeave/.worktrees/table-aware-chunking/data/datasheets/ti_msp430f5529.pdf`
- Source: Texas Instruments MSP430F5529 mixed-signal microcontroller datasheet
  (public, redistributable; SLAS590, downloaded from ti.com/lit/ds/symlink/msp430f5529.pdf)
- Size: 4,498,682 bytes
- Pages: 145

## Environment

- Docling models ready: **True**

## Counts

| Metric | Value |
|---|---|
| total chunks emitted | 2037 |
| TableArtifact entries | 144 |
| distinct `table_group_id`s | 144 |
| `table_summary` chunks | 144 |
| `table_row` chunks | 1539 |
| truncation events (V's guard) | 0 |

## section_path breadcrumb depth distribution (over TableArtifacts)

| heading levels | tables |
|---|---|
| 0 | 3 |
| 1 | 141 |

## Spot-checks (3 random `table_summary` chunks)

### Sample 1 — `table_group_id='#/tables/130'`

- section_path: `10.2 Device Nomenclature`
- caption: `'Figure 10-1. Device Nomenclature'`
- page_no: `114`
- page_bbox: `(110.58732604980469, 605.9351348876953, 501.1798400878906, 304.53497314453125)`
- rows × cols: `10 × 3`
- table_markdown length: `3575` chars

text excerpt:

```
Table: Figure 10-1. Device Nomenclature
Columns: 
Rows: 10
```

markdown excerpt:

```
Figure 10-1. Device Nomenclature

| Processor Family              | CC = Embedded RF Radio MSP = Mixed-Signal Processor XMS = Experimental Silicon PMS = Prototype Device                    |                                                  
```

### Sample 2 — `table_group_id='#/tables/62'`

- section_path: `9.4 Memory Organization`
- caption: `'Table 9-2. Memory Organization  (1)'`
- page_no: `56`
- page_bbox: `(55.758331298828125, 647.0240936279297, 556.0317993164062, 154.3128662109375)`
- rows × cols: `20 × 6`
- table_markdown length: `4803` chars

text excerpt:

```
Table: Table 9-2. Memory Organization  (1)
Columns:  |  | MSP430F5522 MSP430F5521 MSP430F5513 | MSP430F5525 MSP430F5524 MSP430F5515 MSP430F5514 | MSP430F5527 MSP430F5526 MSP430F5517 | MSP430F5529 MSP430F5528 MSP430F5519
Rows: 19
```

markdown excerpt:

```
Table 9-2. Memory Organization  (1)

|                                       |            | MSP430F5522 MSP430F5521 MSP430F5513   | MSP430F5525 MSP430F5524 MSP430F5515 MSP430F5514   | MSP430F5527 MSP430F5526 MSP430F5517   | MSP430F5529 MSP4
```

### Sample 3 — `table_group_id='#/tables/45'`

- section_path: `8.35 12-Bit ADC, Power Supply and Input Range Conditions`
- caption: `''`
- page_no: `44`
- page_bbox: `(55.56804275512695, 688.1776275634766, 556.0166625976562, 560.2726898193359)`
- rows × cols: `7 × 8`
- table_markdown length: `1695` chars

text excerpt:

```
Columns: PARAMETER | PARAMETER | TEST CONDITIONS | V CC | MIN | TYP | MAX | UNIT
Rows: 6
```

markdown excerpt:

```
| PARAMETER   | PARAMETER                                       | TEST CONDITIONS                                                                                       | V CC   |   MIN |   TYP | MAX   | UNIT   |
|-------------|-------------
```

## Determinism (cross-reparse)

- Identical `table_group_id` set across two parses: **True**
- first parse: 144 group_ids
- second parse: 144 group_ids

## Ground-truth outline (read from PDF bookmark tree before parsing)

Read with `pypdf.PdfReader.outline` against
`data/datasheets/ti_msp430f5529.pdf`:

| outline depth | entries |
|---|---|
| 0 | 12 |
| 1 | 73 |
| 2 | 40 |

Total 125 outline entries. The PDF's own bookmark tree is **genuinely
multi-level** (e.g. `8 Specifications` at depth 0, `8.1 Absolute Maximum
Ratings` at depth 1, `8.7.x …` at depth 2). This is what made it a good
candidate to stress the flat-outline gate.

## Docling-side heading levels (what the pipeline actually sees)

Iterated every `section_header` item via `doc.iterate_items()`:

- distinct `.level` values: **1**
- level histogram: `{1: 225}`
- total section_headers: 225

**Docling assigns `level=1` to every heading in this PDF**, despite the PDF
bookmark tree being multi-level. The chunker only ever sees a flat outline.
This means Z's flat-outline gate (≥4 headings AND ≤1 distinct `.level`) fires
correctly: 225 ≥ 4 and exactly 1 distinct level. The cap at depth 1 is the
right behaviour given the upstream signal.

## PASS/FAIL: flat-outline gate non-misfiring

**Decision matrix from task brief:**

- PDF outline genuinely multi-level? **Yes** (depths 0–2 in bookmark tree).
- Docling outline (what the chunker sees) multi-level? **No** — uniformly `level=1`.
- Breadcrumb depth distribution observed: 3 at depth 0, 141 at depth 1 (≤ 1 everywhere).

This falls into the third row of the brief's decision matrix: *"If new doc is
also flat-outline (single level) → expected depth ≤ 1; PASS but note we still
don't have hierarchical evidence."* The gate fired as designed and capped
breadcrumb depth at 1 — no ESP32-S3-style bloat (no depth-10+ samples). No
hierarchical Docling outline was available against which to confirm the gate
**doesn't** fire on a multi-level doc, but it also did not regress: there is
no misfire signal here.

**Verdict: PASS (gate did not misfire; non-bloat confirmed).** Confidence on
the misfire question is **partial** — see Lesson 1 below.

## Spot-check sanity (3 random samples)

All three spot-checks pass:

1. **Sample 1 (#/tables/130, page 114)** — `section_path='10.2 Device
   Nomenclature'` exactly matches Figure 10-1's enclosing sub-section.
   `page_bbox` is a real 4-tuple (Z's DEFECT-2 fix verified live on a second
   PDF). Caption `Figure 10-1. Device Nomenclature` is correctly captured.
2. **Sample 2 (#/tables/62, page 56)** — `section_path='9.4 Memory
   Organization'` matches the source PDF. Caption `Table 9-2. Memory
   Organization (1)` correct. Real bbox tuple.
3. **Sample 3 (#/tables/45, page 44)** — `section_path='8.35 12-Bit ADC,
   Power Supply and Input Range Conditions'` matches sub-section 8.35.
   Caption empty (Docling did not detect a caption for this electrical-spec
   table; the markdown grid itself is intact). bbox populated.

`page_no` and `page_bbox` populated on every sample. No truncation events
(V's guard never tripped; same as ESP32-S3).

## Determinism on a 145-page / 144-table doc

The cross-reparse `table_group_id` set is identical across two parses on a
larger document than ESP32-S3 (144 tables vs 71). The positional `self_ref`
invariant (`project_table_group_id_determinism`) continues to hold at scale.

## DEFECT-3 finding

**None.** No new defects surfaced on this run. The two known defects from the
ESP32-S3 soak (DEFECT-1 section_path bloat, DEFECT-2 missing page_bbox) are
both demonstrably closed:

- DEFECT-1: no depth-12+ breadcrumbs (max observed = 1).
- DEFECT-2: every spot-checked `table_summary` has a populated bbox tuple.

## Lessons learnt

1. **Docling 2.82.0's PDF reader appears to flatten heading levels to `level=1`
   regardless of the underlying PDF bookmark tree.** This is at least the second
   real datasheet (ESP32-S3 was first; MSP430F5529 is second) where the
   PDF-side outline is hierarchical but Docling's `iterate_items()` yields
   uniformly `level=1` headings. Implication: the flat-outline gate Z added
   will fire on essentially every real PDF in our target domain. **We
   currently have no real-PDF evidence that the gate *would* misfire on a
   genuinely multi-level Docling outline** — confirming the negative side
   requires either a synthetic Docling-aware fixture (HTML→Docling pipeline,
   or DOCX→Docling with explicit Heading styles) or a Docling backend that
   preserves heading depth from PDF TOC entries. Worth opening as a follow-up
   investigation before declaring the gate fully validated.
2. **Determinism + bbox + caption + page_no all hold at 2× scale.** Going
   from 71 to 144 tables surfaced zero new ordering or provenance issues —
   the positional-`self_ref` design is robust.
3. **TI's electrical-spec tables (Table 9-2 Memory Organization, multi-device
   variant columns) reach ~4.8 KB of markdown without tripping V's truncation
   guard.** Confirms the current `table_summary_max_chars` budget has
   headroom; we still haven't found a real datasheet that needs truncation.

## Artifacts

- Soak script: `scripts/soak_table_chunking.py`
- Report (this file): `docs/soak/table_chunking_real_datasheet_ti_msp430_2026-05-21.md`
- PDF: `data/datasheets/ti_msp430f5529.pdf` (4.3 MB, TI public redistribution;
  gitignored)
- Raw soak JSON: `/tmp/ti_soak.json` (ephemeral)
