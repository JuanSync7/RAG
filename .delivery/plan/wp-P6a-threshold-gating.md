---
slice_id: P6a
validable_outcome: |
  `uv run pytest tests/eval/test_gate.py -q` reports all 8 FAST tests green, and
  `uv run pytest -m "not slow and not integration" -q` shows no regressions.
  Specifically `src/eval/runner/gate.py` exposes:
    - `GateFailure` (frozen dataclass): qtype, metric, expected, actual.
    - `GateResult` (frozen dataclass): passed, failures (tuple).
    - `validate_eval_report(pack, report) -> GateResult` comparing per-qtype
      recall and faithfulness in an `EvalReport` against
      `pack.thresholds.defaults`.
  Gate semantics:
    - Floors: actual >= expected passes.
    - Recall metric key per qtype = f"recall_at_{report.k}". Absent → skip.
    - Faithfulness metric key = "faithfulness". Absent → skip.
    - qtype absent from thresholds.defaults → not gated (no failure).
    - qtype absent from report's recall/faithfulness map → skip (graceful).
    - If report.total_queries_judged == 0 → skip ALL faithfulness checks
      (anti-gaming guard against stale evals).
    - passed iff failures == ().
  Exports added (sorted, `__all__`):
    - `src/eval/runner/__init__.py`: GateFailure, GateResult, validate_eval_report.
    - `src/eval/__init__.py`: same three.
  Mutation probes:
    A. Force `passed = True` regardless of failures → reds
       `test_gate_fails_on_recall_below_threshold` (or faithfulness twin).
    B. Drop the empty-qtype-skip guard so missing report keys raise KeyError →
       reds `test_gate_omits_missing_qtype_from_thresholds` (or zero-judged twin).
touches:
  - src/eval/runner/gate.py (new)
  - src/eval/runner/__init__.py (additive export)
  - src/eval/__init__.py (additive export)
  - tests/eval/test_gate.py (new)
depends_on:
  - P5 (EvalReport.faithfulness_by_qtype + total_queries_judged)
  - P4 (EvalReport.recall_by_qtype + k)
  - P0.5 (EvalPack.thresholds.defaults)
---

# P6a — Threshold gating

## Acceptance criteria (test list)

1. `test_gate_passes_when_all_metrics_above_thresholds` — `validate_eval_report`
   on a report whose per-qtype recall and faithfulness exceed every declared
   floor returns `GateResult(passed=True, failures=())`.
2. `test_gate_fails_on_recall_below_threshold` — factoid recall_at_5 actual=0.7
   vs floor 0.8 → passed False; failures has exactly one entry with
   qtype="factoid", metric="recall_at_5", expected=0.8, actual=0.7.
3. `test_gate_fails_on_faithfulness_below_threshold` — factoid faithfulness
   actual=0.4 vs floor 0.6 → one failure with metric="faithfulness".
4. `test_gate_skips_qtype_absent_from_thresholds` — report has qtype "messy",
   thresholds has no entry for "messy" → no failure raised for messy.
5. `test_gate_skips_recall_check_when_threshold_key_absent` — thresholds.defaults
   for factoid has only "faithfulness", no `recall_at_K` matching report.k →
   recall not gated; faithfulness still gated normally.
6. `test_gate_skips_all_faithfulness_when_total_queries_judged_is_zero` —
   anti-gaming: total_queries_judged=0 even with faithfulness_by_qtype entries
   → no faithfulness failures (recall still gated).
7. `test_gate_aggregates_multiple_failures_across_qtypes` — two qtypes fail
   recall + one fails faithfulness across qtypes → failures tuple has 3
   entries, not short-circuited.
8. `test_gate_result_is_frozen` — partner: `GateResult` and `GateFailure` are
   frozen; assigning to a field raises `FrozenInstanceError`.
