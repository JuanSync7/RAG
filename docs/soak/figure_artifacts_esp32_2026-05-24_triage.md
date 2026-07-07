# FIG-4 figure xref triage — ESP32-S3 (2026-05-24)

## Setup

- PDF: `data/datasheets/esp32-s3_datasheet.pdf` (87 pages, 1.05 MB)
- Baseline soak: `docs/soak/figure_artifacts_esp32_2026-05-24.{md,json}` —
  mode_b (`RAG_XREF_EXTRACT_FIGURE_REFS=true`) resolved **2 of 7** unique
  prose-figure refs (28.57%).
- Triage method: replayed the parse, enumerated `xref_targets` figure refs
  from prose chunks, cross-checked each unresolved value against
  `doc.pictures` (PictureItems and their `pic.captions`) and against
  every TextItem matching `^\s*(Figure|Fig\.?)\s+\d+(-\d+|\.\d+)?` in the
  same document.

## Per-ref triage table

| Ref value | Prose page | Resolved (baseline) | Classification | Evidence |
|---|---:|---|---|---|
| `Figure 2-1` | 15 | NO | **(a) caption-binding failure** | Caption TextItem `"Figure 2-1. ESP32-S3 Pin Layout (Top View)"` present on page 15; no PictureItem on page 15 has `pic.captions` bound — only `#/pictures/12` on page 13 (Figure 1-1) was bound. |
| `Figure 3-1` | 32 | YES | resolved | `#/pictures/31` on page 33 carries bound caption `"Figure 3-1. Visualization of Timing Parameters for the Strapping Pins"`. |
| `Figure 4-1` | 38 | YES | resolved | `#/pictures/35` on page 38 carries bound caption `"Figure 4-1. Address Mapping Structure"`. |
| `Figure 4-2` | 42 | NO | **(a) caption-binding failure** | Caption TextItem `"Figure 4-2. Components and Power Domains"` exists on page 43; no PictureItem on page 42–43 has `pic.captions` bound. |
| `Figure 4.1` | 83 | NO | **(d) cross_refs false positive** | Prose excerpt: `"Updated Figure 4.1.2 Memory Organization in Section 4-1 Address Mapping Stru…"`. The `cross_refs` regex `\b(?:Figure\|Fig\.?)\s+\d+(?:[.-]\d+)?\b` matches `Figure 4.1`, then leaves `.2 Memory Organization` as residue. The actual referent is a section heading, not a figure. No figure entity ever existed to resolve against. |
| `Figure 7-1` | 77 | NO | **(a) caption-binding failure** | Caption TextItem `"Figure 7-1. QFN56 (7×7 mm) Package"` present on page 77; matching PictureItem on the same page has `captions=[]`. |
| `Figure 7-2` | 77 | NO | **(a) caption-binding failure** | Caption TextItem `"Figure 7-2. QFN56 (7×7 mm) Package (Only for ESP32-S3FH4R2)"` present on page 78; matching PictureItem unbound. |

## Root-cause taxonomy

- **(a) caption-binding failure: 4 of 5 misses (80%)** — Docling 2.82.0
  emits the figure caption as an independent body TextItem and fails to
  populate `pic.captions` on the matching PictureItem. The extractor's
  `pic.caption_text(doc=doc)` therefore returns `""` and the artifact
  ships with `caption_label=""`. This is the same parent-binding quirk
  documented in memory `project_docling_heading_provenance` for tables.
- **(d) cross_refs false positive: 1 of 5 misses (20%)** — `Figure 4.1`
  is not a figure ref at all; it's a fragment of the section heading
  `Figure 4.1.2 Memory Organization` (changelog entry on the appendix
  page). `cross_refs`' greedy regex cannot tell the difference without
  surrounding context.
- **(b) Docling-missed picture: 0 of 5** — every failed ref had its
  caption text emitted somewhere; nothing was wholly invisible.
- **(c) unicode normalization: 0 of 5** — all caption hyphens were
  ASCII; existing normalization is sufficient on this corpus.

## Fix applied

**`src/ingest/support/docling.py`** — new `_build_unbound_caption_fallback`
helper, wired into `_extract_figure_artifacts`. When `pic.captions` is
empty, the extractor walks forward through `iterate_items()` and adopts
the first TextItem that

1. lives on the same page as the PictureItem, and
2. matches the canonical figure-caption regex
   (`^\s*(Figure|Fig\.?)\s+N(-N|.N)?`).

Scan terminates at the next PictureItem or at a page boundary, so we
never invent cross-page provenance.

Tests (TDD, failing-first):

- `tests/ingest/test_figure_caption_binding_fallback.py` — 4 cases:
  forward-textitem adoption, cross-page rejection, non-figure-textitem
  rejection, bound-caption short-circuit. All pass post-fix.

## Re-soak result

| Metric | Baseline (2026-05-24) | After fix |
|---|---:|---:|
| mode_b unique_targets | 7 | 7 |
| mode_b resolved | 2 | **6** |
| mode_b resolvable_rate | 28.57% | **85.71%** |
| figures_with_caption_label | 5 | **9** |
| figure_caption_label_rate | 5.88% | 10.59% |
| figure_image_uri_sanitized | True | True |
| figure_chunk_idempotent | True | True |

Full re-soak transcript: `docs/soak/figure_artifacts_esp32_2026-05-24_after_triage.{md,json}`.

The single remaining unresolved ref (`Figure 4.1`) is a `cross_refs`
false positive, not a binding failure. Resolving it would require
either:

1. tightening `cross_refs` to reject `Figure N.N` when followed by
   `.M` in the source text (context-aware lookbehind/ahead), or
2. accepting that some prose figure-ref emissions will go unresolved
   when the prose is actually citing a section number that happens to
   start with the word "Figure".

Option (1) is non-trivial and risks dropping legitimate `Figure 10.3`
references in other corpora. Deferred as a FIG-5 follow-up.

## Flag-flip decision

**Flip `RAG_XREF_EXTRACT_FIGURE_REFS` default → `true`.**

Justification: post-fix resolvable_rate (85.71%) is comfortably above
the 30% bar that originally held FIG-3 back, and the residual
unresolved case is a cross-reference-extractor concern, not a figure
resolver concern. Idempotency, sanitization, and prose-ref emission
counts are unchanged.

Operators can still set `RAG_XREF_EXTRACT_FIGURE_REFS=false` to opt
out (e.g. on corpora that exhibit pathological cross_refs false
positives).

## Follow-ups (out of FIG-4 scope)

- **FIG-5 — `cross_refs` context awareness.** Reject `Figure N.N` when
  the next character in the source is `.M`. Validate on a different
  datasheet (TI MSP430 already in fixtures).
- **caption-binding rate at the source.** `figure_caption_label_rate`
  is still only 10.59% because most of the 85 raw pictures are
  *unlabelled* graphics (logos, decorative panels, etc.) — not all
  pictures *should* have a label. The metric is misleading as an
  acceptance criterion; consider gating on
  `figures_with_label > 0 AND mode_b_resolvable_rate >= 0.30` instead
  of label rate over all pictures.

## Lessons honoured

- `feedback_real_pdf_soak_before_merge` — re-soaked on real ESP32-S3
  after the fix; the JSON delta is the proof, not the synthetic tests.
- `project_docling_heading_provenance` — same parent-binding quirk
  shows up on figures; same fix shape (replay `iterate_items()` and
  reconstruct what Docling didn't bind).
- `feedback_heuristic_gates_independent` — page-scope **AND** regex
  match are independent guards in the fallback; either alone would
  over-bind.
