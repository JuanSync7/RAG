# Table-chunking soak — real datasheet (2026-05-23)

## Dataset

- PDF: `/home/kok-shew-juan/RagWeave/.worktrees/table-aware-chunking/data/datasheets/esp32-s3_datasheet.pdf`
- Source: Espressif ESP32-S3 datasheet (public, redistributable)
- Size: 1,098,115 bytes
- Pages: 87

## Environment

- Docling models ready: **True**

## Counts

| Metric | Value |
|---|---|
| total chunks emitted | 831 |
| TableArtifact entries | 71 |
| distinct `table_group_id`s | 71 |
| `table_summary` chunks | 71 |
| `table_row` chunks | 536 |
| truncation events (V's guard) | 0 |

## section_path breadcrumb depth distribution (over TableArtifacts)

| heading levels | tables |
|---|---|
| 0 | 6 |
| 1 | 65 |

## Spot-checks (3 random `table_summary` chunks)

### Sample 1 — `table_group_id='#/tables/28'`

- section_path: `3.4 JTAG Signal Source Control`
- caption: `'Table 3-5. JTAG Signal Source Control'`
- page_no: `35`
- page_bbox: `(55.898406982421875, 760.8178482055664, 559.1690063476562, 648.7037353515625)`
- rows × cols: `7 × 5`
- table_markdown length: `942` chars

text excerpt:

```
Table: Table 3-5. JTAG Signal Source Control
Columns: JTAG Signal Source | EFUSE_DIS_PAD_JTAG | EFUSE_DIS_USB_JTAG | EFUSE_STRAP_JTAG_SEL | GPIO3
Rows: 6
```

markdown excerpt:

```
Table 3-5. JTAG Signal Source Control

| JTAG Signal Source          |   EFUSE_DIS_PAD_JTAG |   EFUSE_DIS_USB_JTAG | EFUSE_STRAP_JTAG_SEL   | GPIO3   |
|-----------------------------|----------------------|----------------------|-----------
```

### Sample 2 — `table_group_id='#/tables/11'`

- section_path: `2.3.1 IO MUX Functions`
- caption: `'Table 2-3. Peripheral Signals Routed via IO MUX'`
- page_no: `20`
- page_bbox: `(55.7794075012207, 508.8037414550781, 546.2653198242188, 118.4287109375)`
- rows × cols: `8 × 3`
- table_markdown length: `3216` chars

text excerpt:

```
Table: Table 2-3. Peripheral Signals Routed via IO MUX
Columns: Pin Function | Signal | Description
Rows: 7
```

markdown excerpt:

```
Table 2-3. Peripheral Signals Routed via IO MUX

| Pin Function                                          | Signal                                                                         | Description                                         
```

### Sample 3 — `table_group_id='#/tables/5'`

- section_path: ``
- caption: `''`
- page_no: `12`
- page_bbox: `(69.79170989990234, 746.8679275512695, 538.8440551757812, 611.82080078125)`
- rows × cols: `9 × 3`
- table_markdown length: `769` chars

text excerpt:

```
Columns: 
Rows: 9
```

markdown excerpt:

```
| 1-1   | ESP32-S3 Series Nomenclature                              |   13 |
|-------|-----------------------------------------------------------|------|
| 2-1   | ESP32-S3 Pin Layout (Top View)                            |   15 |
| 2-2   |
```

## Determinism (cross-reparse)

- Identical `table_group_id` set across two parses: **True**
- first parse: 71 group_ids
- second parse: 71 group_ids

## Xref edges

- chunks with edges: 561
- total edges: 701
- edges per chunk p50: 1
- edges per chunk p90: 1
- by type: figure=20, section=114, standard=1, table=566

## Xref resolvability

- section resolvable: 0
- table resolvable: 559
- unresolvable table (no matching caption_label): 7
- unresolvable figure: 20
- unresolvable appendix: 0

## Caption label coverage

- tables total: 71
- tables with caption_label: 54
- label rate: 76.06%
