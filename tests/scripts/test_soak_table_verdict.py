"""Unit tests for ``_table_soak_verdict``.

Mirrors the FIG-8 figure-verdict tests: gate on the referenced-only
denominator (``table_resolvable / (table_resolvable +
unresolvable_table_no_label)``), not on the over-all
``caption_label_coverage.label_rate`` (which is polluted by layout /
header / continuation rows Docling reports as "tables" but prose never
references).

See memory ``feedback_pick_meaningful_denominators``.
"""
from __future__ import annotations

from scripts.soak_table_chunking import _table_soak_verdict


def _good_soak(*, resolved: int = 559, unresolved: int = 8, tables_total: int = 71) -> dict:
    """A soak dict that should PASS every verdict criterion.

    Default numbers mirror ESP32-S3 (98.76% rate, 559/567 referenced).
    """
    return {
        "counts": {
            "tables": tables_total,
            "total_chunks": 841,
            "summary_chunks": tables_total,
            "row_chunks": 200,
            "group_ids": tables_total,
        },
        "xref_resolvability": {
            "section_resolvable": 97,
            "table_resolvable": resolved,
            "unresolvable_table_no_label": unresolved,
            "unresolvable_figure": 27,
            "unresolvable_appendix": 0,
        },
        "caption_label_coverage": {
            "tables_total": tables_total,
            "tables_with_label": 54,
            "label_rate": 0.7605633802816901,  # decorative-tax noise — observability only
        },
    }


class TestTableSoakVerdict:
    def test_all_pass_esp32_shape(self):
        """Happy path: ESP32-S3 real numbers (559/567 ≈ 98.59%) PASS."""
        v = _table_soak_verdict(_good_soak())
        assert v["all_pass"] is True
        for c in v["criteria"].values():
            assert c["ok"] is True

    def test_all_pass_msp430_shape(self):
        """Happy path: MSP430F5529 real numbers (589/628 ≈ 93.79%) PASS."""
        v = _table_soak_verdict(_good_soak(resolved=589, unresolved=39, tables_total=144))
        assert v["all_pass"] is True
        crit = v["criteria"]["table_caption_label_rate_of_referenced_ge_0_90"]
        assert crit["ok"] is True
        assert crit["value"] >= 0.93

    def test_low_overall_label_rate_does_not_fail_verdict(self):
        """Decorative-tax: a low ``caption_label_coverage.label_rate``
        must NOT gate the verdict — the whole point of this verdict.

        MSP430 hits ~28% over-all but is 93.8% on referenced-only.
        """
        soak = _good_soak(resolved=589, unresolved=39, tables_total=144)
        soak["caption_label_coverage"]["label_rate"] = 0.2778  # very low
        v = _table_soak_verdict(soak)
        assert v["all_pass"] is True

    def test_rate_just_below_threshold_fails(self):
        """89.99% → FAIL the >= 0.90 gate."""
        # 89 / 100 = 0.89
        soak = _good_soak(resolved=89, unresolved=11)
        v = _table_soak_verdict(soak)
        crit = v["criteria"]["table_caption_label_rate_of_referenced_ge_0_90"]
        assert crit["ok"] is False
        assert v["all_pass"] is False

    def test_rate_exactly_at_threshold_passes(self):
        """0.90 boundary: exact threshold should PASS (>= 0.90)."""
        # 90 / 100 = 0.90
        soak = _good_soak(resolved=90, unresolved=10)
        v = _table_soak_verdict(soak)
        crit = v["criteria"]["table_caption_label_rate_of_referenced_ge_0_90"]
        assert crit["ok"] is True
        assert crit["value"] == 0.9
        assert v["all_pass"] is True

    def test_zero_table_count_fails(self):
        """No tables in the doc → ``table_count_gt_0`` FAILs (sanity gate)."""
        soak = _good_soak(tables_total=0)
        v = _table_soak_verdict(soak)
        assert v["criteria"]["table_count_gt_0"]["ok"] is False
        assert v["all_pass"] is False

    def test_zero_referenced_denominator_skips_rate_gate(self):
        """No table xrefs at all (denom == 0) → rate criterion PASSes with 'n/a'.

        Mirrors figure verdict: nothing to gate on is not a regression in
        caption-label parsing.
        """
        soak = _good_soak(resolved=0, unresolved=0)
        v = _table_soak_verdict(soak)
        crit = v["criteria"]["table_caption_label_rate_of_referenced_ge_0_90"]
        assert crit["ok"] is True
        assert crit["value"] == "n/a"
        # table_count_gt_0 still PASSes from the default tables_total=71
        assert v["all_pass"] is True

    def test_missing_xref_resolvability_block(self):
        """Tolerate a soak run that never populated xref_resolvability."""
        soak = _good_soak()
        soak.pop("xref_resolvability")
        v = _table_soak_verdict(soak)
        # denom 0 → rate criterion PASSes as "n/a"
        crit = v["criteria"]["table_caption_label_rate_of_referenced_ge_0_90"]
        assert crit["value"] == "n/a"
        assert crit["ok"] is True

    def test_missing_counts_block(self):
        """Tolerate a soak run that never populated counts."""
        soak = _good_soak()
        soak.pop("counts")
        v = _table_soak_verdict(soak)
        assert v["criteria"]["table_count_gt_0"]["ok"] is False
        assert v["criteria"]["table_count_gt_0"]["value"] == 0

    def test_none_fields_tolerated(self):
        """Explicit Nones in nested fields must not crash."""
        soak = {
            "counts": {"tables": None},
            "xref_resolvability": {
                "table_resolvable": None,
                "unresolvable_table_no_label": None,
            },
        }
        v = _table_soak_verdict(soak)
        # Doesn't crash; rate is n/a (denom 0).
        assert v["criteria"]["table_count_gt_0"]["ok"] is False
        assert (
            v["criteria"]["table_caption_label_rate_of_referenced_ge_0_90"]["value"]
            == "n/a"
        )

    def test_empty_soak_dict(self):
        """Completely empty dict must not crash."""
        v = _table_soak_verdict({})
        assert v["all_pass"] is False
        assert v["criteria"]["table_count_gt_0"]["ok"] is False
        assert (
            v["criteria"]["table_caption_label_rate_of_referenced_ge_0_90"]["ok"]
            is True
        )

    def test_returned_shape_matches_figure_verdict(self):
        """Top-level shape must match ``_figure_soak_verdict`` for parity."""
        v = _table_soak_verdict(_good_soak())
        assert set(v.keys()) == {"criteria", "all_pass"}
        for c in v["criteria"].values():
            assert set(c.keys()) == {"ok", "value", "threshold"}
