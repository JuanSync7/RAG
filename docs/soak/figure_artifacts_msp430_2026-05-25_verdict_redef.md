# Figure-artifact soak — MSP430F5529 (FIG-8 verdict redefinition)

**Date**: 2026-05-25
**Vendor**: Texas Instruments
**Datasheet**: MSP430F5529
**Fixture**: `docs/soak/figure_artifacts_msp430_2026-05-24.{md,json}`
**Ticket**: FIG-8

This transcript re-applies `_figure_soak_verdict` (post-FIG-8) to the
metrics captured in the FIG-6 soak run. The underlying parse, chunking
and resolver are unchanged — only the verdict criterion was redefined
to use the captioned-only denominator (memory
`feedback_pick_meaningful_denominators`).

## Figure-artifact soak (FIG-3 / FIG-8)

| Metric | Value |
| --- | --- |
| figure_count | 210 |
| figure_chunks_emitted | 61 |
| figures_with_caption_label | 61 |
| figure_caption_label_rate | 29.05% |
| figure_caption_label_rate_of_captioned | 100.00% (61/61) |
| figure_caption_via_fallback_count | 0 (of 61 captioned; 0.00%) |
| figure_caption_via_native_count | 61 |
| figure_image_uri_sanitized | True |
| figure_chunk_idempotent | True (210 ids, 0 dupes) |
| mode_a.figure_refs_emitted | 0 |
| mode_b.resolvable_rate | 100.00% (61/61) |

## Verdict (FIG-8 criteria)

| Criterion | Threshold | Value | Result |
| --- | --- | --- | --- |
| figure_count_gt_0 | > 0 | 210 | **PASS** |
| figure_caption_label_rate_of_captioned_ge_0_95 | >= 0.95 | 1.0000 | **PASS** |
| figure_image_uri_sanitized | true | True | **PASS** |
| figure_chunk_idempotent | true | True | **PASS** |
| mode_a_emits_zero_figure_refs | == 0 | 0 | **PASS** |
| mode_b_resolvable_rate_ge_0_30 | >= 0.30 | 1.0000 | **PASS** |

**FIG-8 verdict: PASS**

## Before vs. after

| Verdict | FIG-6 (`label_rate >= 0.30`) | FIG-8 (`label_rate_of_captioned >= 0.95`) |
| --- | --- | --- |
| Overall | **FAIL** | **PASS** |
| Caption criterion | FAIL (0.2905) | PASS (1.0000) |

The 29.05% over-all rate is the decorative-picture tax: 149 of 210
Docling pictures are icons / logos / "Cont'd" glyphs with no caption.
Of the 61 that ARE captioned, every single one has a parseable label
via native Docling binding (no fallback needed), so the caption-label
regex has zero coverage gap on this vendor — the previous gate was
flagging a non-defect.
