# Table-chunking soak — real datasheet (2026-05-26)

## Dataset

- PDF: `/home/kok-shew-juan/RagWeave/.worktrees/table-aware-chunking/data/datasheets/ti_msp430f5529.pdf`
- Source: Espressif ESP32-S3 datasheet (public, redistributable)
- Size: 4,498,682 bytes
- Pages: 145

## Environment

- Docling models ready: **True**

## Counts

| Metric | Value |
|---|---|
| total chunks emitted | 2098 |
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

## Xref edges

- chunks with edges: 699
- total edges: 891
- edges per chunk p50: 1
- edges per chunk p90: 1
- by type: figure=169, section=83, table=639

## Xref resolvability

- section resolvable: 79
- table resolvable: 593
- unresolvable table (no matching caption_label): 46
- unresolvable figure: 169
- unresolvable appendix: 0

## Caption label coverage

- tables total: 144
- tables with caption_label: 40
- label rate: 27.78%

## Table soak verdict

| criterion | threshold | actual | result |
|---|---|---|---|
| table_count_gt_0 | > 0 | 144 | **PASS** |
| table_caption_label_rate_of_referenced_ge_0_90 | >= 0.90 | 0.928 | **PASS** |

**Table verdict: PASS** — see criteria table above.

## Figure-artifact soak (FIG-3)

| Metric | Value |
|---|---|
| figure_count | 210 |
| figure_chunks_emitted | 61 |
| figures_with_caption_label | 61 |
| figure_caption_label_rate | 29.05% |
| figure_caption_label_rate_of_captioned | 100.00% (61/61) |
| figure_caption_via_fallback_count | 0 (of 61 captioned; 0.00%) |
| figure_caption_via_native_count | 61 |
| figure_image_uri_sanitized | True |
| figure_chunk_idempotent | True (210 ids, 0 dupes) |
| mode_a figure_refs_emitted (flag off) | 0 |
| mode_b figure_refs_emitted (flag on) | 106 |
| mode_b unique_targets | 40 |
| mode_b resolved | 36 |
| mode_b resolvable_rate | 90.00% |

### Top section_paths by figure count

| section_path | figures |
|---|---|
| `7.1 Pin Diagrams` | 13 |
| `PACKAGE MATERIALS INFORMATION` | 10 |
| `4 Functional Block Diagrams` | 6 |
| `VQFN - 1 mm max height` | 6 |
| `8.43 Ports PU.0 and PU.1` | 5 |

### Verdict criteria

| criterion | threshold | actual | result |
|---|---|---|---|
| figure_count_gt_0 | > 0 | 210 | **PASS** |
| figure_caption_label_rate_of_captioned_ge_0_95 | >= 0.95 | 1.0 | **PASS** |
| figure_image_uri_sanitized | true | True | **PASS** |
| figure_chunk_idempotent | true | True | **PASS** |
| mode_a_emits_zero_figure_refs | == 0 | 0 | **PASS** |
| mode_b_resolvable_rate_ge_0_30 | >= 0.30 | 0.9 | **PASS** |

**FIG-3 verdict: PASS** — see criteria table above.
