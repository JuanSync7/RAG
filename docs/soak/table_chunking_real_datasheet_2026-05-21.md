# Table-chunking soak — real datasheet (2026-05-21)

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
| 4 | 1 |
| 6 | 4 |
| 7 | 9 |
| 8 | 9 |
| 10 | 1 |
| 12 | 6 |
| 13 | 31 |
| 15 | 4 |

## Spot-checks (3 random `table_summary` chunks)

### Sample 1 — `table_group_id='#/tables/28'`

- section_path: `ESP32-S3 Series > Features > 1 ESP32-S3 Series Comparison > 1.2 Comparison > 2 Pins > 2.3 IO Pins > 2.5 Power Supply > 3.4 JTAG Signal Source Control`
- caption: `'Table 3-5. JTAG Signal Source Control'`
- page_no: `35`
- page_bbox: `None`
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

- section_path: `ESP32-S3 Series > Features > 1 ESP32-S3 Series Comparison > 1.2 Comparison > 2 Pins > 2.3 IO Pins > 2.3.1 IO MUX Functions`
- caption: `'Table 2-3. Peripheral Signals Routed via IO MUX'`
- page_no: `20`
- page_bbox: `None`
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
- page_bbox: `None`
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

Determinism invariant (project memory `project_table_group_id_determinism`) holds
on a real 87-page datasheet — positional Docling `self_ref` values (`#/tables/0…#/tables/70`)
are reproduced identically. Re-ingest will update existing chunks in-place.

## Observations

- **Adaptive chunker fires aggressively**: 71 tables yielded 71 summary chunks
  (1:1) and 536 row chunks. 56 / 71 tables (79%) qualified as "small + uniform"
  and got per-row chunks.
- **Token-budget guard never tripped** on this datasheet — all summary texts
  fit within the configured `table_summary_max_chars`. Confirms V's guard is
  not over-firing; need a wider-header datasheet to exercise it.
- **Determinism invariant holds** across the full document.
- **All 71 tables have populated `table_markdown`** with a real markdown grid.
- **`page_no` is populated** on every spot-checked table_summary chunk.
- **section_path depth distribution is pathological**: median depth 12,
  mean 9.97, max 15, with 35 / 71 (49%) tables at depth >= 13. The expected
  depth for ESP32-S3 sub-sections is 2–4 (e.g. "3 Boot Configurations >
  3.4 JTAG Signal Source Control"). See DEFECT-1 below.

## DEFECT-1: section_path breadcrumb bloat on real-world flat-outline PDFs

**Severity**: medium. Affects retrieval relevance + UI breadcrumb display.

**Evidence**: Sample 1's table_3-5 (page 35, sub-section "3.4 JTAG Signal
Source Control") emits

```
ESP32-S3 Series > Features > 1 ESP32-S3 Series Comparison > 1.2 Comparison
  > 2 Pins > 2.3 IO Pins > 2.5 Power Supply > 3.4 JTAG Signal Source Control
```

The correct breadcrumb is `3 Boot Configurations > 3.4 JTAG Signal Source Control`
(depth 2). The emitted breadcrumb carries 6 stale sibling-section headings from
earlier in the document, including unrelated chapter-2 leaves.

**Root cause**: Docling assigns `level=1` to **every** section header in this
datasheet (verified by walking `iterate_items()` on the parsed document — all
55 section headers have `getattr(item, 'level', None) == 1`). The "same-level
no-content-yet ⇒ logical child" heuristic in
`src/ingest/support/docling.py::_resolve_table_section_paths` (lines ~744–757,
introduced to fix the inverse `project_docling_section_path_collapse` defect)
then stacks consecutive same-level headings as parent/child whenever no body
item intervenes (e.g. table-of-contents headings, sub-sub-sub-sections,
"Note:" markers, "Cont'd from previous page" markers — all categorised as
`section_header`). The stack grows monotonically across the document.

**Reproduction**:

```bash
uv run python scripts/soak_table_chunking.py \
  data/datasheets/esp32-s3_datasheet.pdf /tmp/soak.md
# Open /tmp/soak.md and inspect any depth>=12 sample's section_path.
```

**Suggested fix** (gate the W heuristic):
1. Compute `outline_is_flat = len({h.level for h in headings}) <= 1` over a
   short prelude of the document (or on demand). When True, disable the
   "same-level no-content-yet ⇒ child" branch and treat every same-level
   header as a sibling that pops the previous one regardless of `had_content`.
2. Alternatively, **cap the stack depth** at a small constant (e.g. 6) — when
   a flat outline pushes beyond that, drop the oldest frames.
3. Add a regression test that synthesises a flat-outline PDF (all H1, no H2/H3)
   with three sibling sections and asserts breadcrumb depth ≤ 2 on a table
   in the last section.

**Pre-existing limitation vs new defect**: this is a *new* defect introduced
by commit 1654cab ("fix(ingest): section_path emits full ancestor breadcrumb")
when combined with W's same-level-collapse-into-child rule. Before that
commit, only the nearest enclosing header was attached, so the bloat was
invisible. The fix solved the synthetic-test case (single-document, well-formed
H1/H2/H3) and broke the real-world case (flat outline).

## DEFECT-2: `table_summary.page_bbox` is always None on TableArtifact PageRef

**Severity**: low. UI cannot draw a per-table highlight box even though
X's PR (commit 15507e7) wired bbox decoding into the citation payload.

**Evidence**: All three spot-checked `table_summary` chunks have
`page_bbox: None` despite valid `page_no` values.

**Root cause**: `_page_ref_from_table_item` in
`src/ingest/support/docling.py` (lines 662–672) hard-codes `bbox=None`
when constructing the PageRef:

```python
def _page_ref_from_table_item(tbl: Any) -> Any:
    ...
    for p in prov:
        page_no = getattr(p, "page_no", None)
        ...
        return PageRef(page_no=int(page_no), page_label="", bbox=None)  # <-- bbox dropped
```

Compare to `_page_ref_from_chunk_meta` (lines 627–659), which **does** decode
the bbox tuple from the provenance entry. The two helpers diverged.

**Suggested fix**: copy the bbox-decoding block from
`_page_ref_from_chunk_meta` into `_page_ref_from_table_item`. Trivial,
~15 lines, and adds a free signal to every table-summary citation.

**Pre-existing limitation vs new defect**: technically pre-existing
(the helper has always returned `bbox=None`), but it materially regresses
the value of X's citation-bbox patch for the most retrieval-relevant chunk
type (table summaries). Treating as a defect-of-omission.

## OBSERVATION-3 (not a defect): table-5 on page 12 has empty header

Sample 3 (`#/tables/5`, page 12, "List of Tables") is a 9 × 3 grid whose
first row is blank — Docling does not detect a header row, so the
table_summary emits `Columns: \nRows: 9`. Inspecting the markdown shows
the table itself (a list-of-tables index page) has no column titles in
the source PDF — Docling is faithfully reproducing the source. The
`section_path` for this artifact is also empty (the "List of Tables"
heading was popped before the table emission). Not a pipeline bug;
arguably the table-of-contents pages should be filtered upstream
(out-of-scope for this task).

## Artifacts

- Soak script: `scripts/soak_table_chunking.py`
- Report (this file): `docs/soak/table_chunking_real_datasheet_2026-05-21.md`
- PDF: `data/datasheets/esp32-s3_datasheet.pdf` (1.05 MB, Espressif public
  redistribution; **NOT** auto-committed — main agent to decide on inclusion
  vs gitignore).

## Lessons learnt

- **Real-world PDFs frequently flatten their outline to level=1.** Any
  heuristic that distinguishes "logical child vs sibling" by `(level, had_content)`
  must gate on whether the document actually exposes >1 level. Synthetic
  fixtures using `reportlab` Heading1/Heading2 styles will not reproduce
  this — they faithfully assign different levels.
- **Two recent PRs (W's section_path full breadcrumb, X's citation bbox)
  shipped passing all synthetic tests yet regress on a real 87-page
  datasheet.** Add at least one real-PDF soak to the table-chunking
  feature's CI gate (offline, gated on local model cache) before merging
  further table-aware changes.
- **Determinism invariant scales.** 71-table real PDF reparses identically —
  positional `self_ref` continues to be the right load-bearing primitive.
