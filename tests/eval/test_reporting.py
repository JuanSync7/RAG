# @summary
# P8b — tests for baseline diff + reporting + promote-baseline CLI.
# Covers the pure diff engine (compute_baseline_diff: regression/improvement/
# unchanged bucketing, strict epsilon boundary, new/dropped qtypes, gate flip),
# the deterministic markdown snapshot (render_baseline_diff_markdown reuses
# render_ci_summary + appends a regression callout), and the promote-baseline
# command (writes evals/packs/<pack>/baseline.json, discovers latest report,
# error exit codes).
# Mutation A target: test_compute_diff_epsilon_boundary
# Mutation B target: test_compute_diff_regression_vs_improvement_vs_unchanged
# Mutation C target: test_markdown_snapshot
# Mutation D target: test_promote_baseline_writes_to_pack_dir
# Mutation E target: test_promote_baseline_discovers_latest_report
# Exports: test_markdown_snapshot,
#          test_compute_diff_regression_vs_improvement_vs_unchanged,
#          test_compute_diff_epsilon_boundary,
#          test_compute_diff_new_and_dropped_qtypes,
#          test_compute_diff_gate_flip,
#          test_compute_diff_gate_not_flipped_when_both_pass,
#          test_promote_baseline_writes_to_pack_dir,
#          test_promote_baseline_missing_pack_exits_2,
#          test_promote_baseline_no_report_found_exits_2,
#          test_promote_baseline_discovers_latest_report
# Deps: src.eval.runner.baseline, src.eval.runner.reporting, src.eval.cli
# @end-summary
"""P8b — baseline diff + reporting + promote-baseline tests."""
from __future__ import annotations

import io
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixed JSON payload fixtures (control the snapshot exactly)
# ---------------------------------------------------------------------------


def _baseline_payload() -> dict:
    """Baseline report: PASSING, qtypes {factoid, dropme}."""
    return {
        "pack_name": "synpack",
        "timestamp": "2026-05-01T00:00:00+00:00",
        "collection_name": "test_coll",
        "k": 5,
        "passed": True,
        "failures": [],
        "recall_by_qtype": {"factoid": 0.90, "dropme": 0.80},
        "mrr_by_qtype": {"factoid": 0.70},
        "faithfulness_by_qtype": {"factoid": 0.80},
        "per_query_recall": {},
        "per_query_mrr": {},
        "per_query_faithfulness": {},
        "total_queries_scored": 10,
        "total_queries_skipped": 1,
        "total_queries_judged": 5,
    }


def _current_payload() -> dict:
    """Current report: FAILING (gate flip), qtypes {factoid, newbie}.

    Designed so exactly:
      * factoid recall 0.90 -> 0.80  (delta -0.10) => REGRESSION
      * factoid mrr    0.70 -> 0.85  (delta +0.15) => IMPROVEMENT
      * factoid faith  0.80 -> 0.78  (delta -0.02) => UNCHANGED (within eps)
      * newbie  => NEW qtype (absent from baseline)
      * dropme  => DROPPED qtype (absent from current)
    """
    return {
        "pack_name": "synpack",
        "timestamp": "2026-05-30T00:00:00+00:00",
        "collection_name": "test_coll",
        "k": 5,
        "passed": False,
        "failures": [
            {
                "qtype": "factoid",
                "metric": "recall_at_5",
                "expected": 0.90,
                "actual": 0.80,
                "qid": "",
            }
        ],
        "recall_by_qtype": {"factoid": 0.80, "newbie": 0.95},
        "mrr_by_qtype": {"factoid": 0.85},
        "faithfulness_by_qtype": {"factoid": 0.78},
        "per_query_recall": {},
        "per_query_mrr": {},
        "per_query_faithfulness": {},
        "total_queries_scored": 10,
        "total_queries_skipped": 1,
        "total_queries_judged": 5,
    }


# ---------------------------------------------------------------------------
# 1. Headline snapshot
# ---------------------------------------------------------------------------


EXPECTED_MARKDOWN = """## Eval Gate Summary

**Status: FAIL**

| qtype | recall@5 | mrr | faithfulness | status |
| --- | --- | --- | --- | --- |
| factoid | 0.800 | 0.850 | 0.780 | FAIL |
| newbie | 0.950 | — | — | PASS |

### Failures

| qtype | qid | metric | expected | actual |
| --- | --- | --- | --- | --- |
| factoid | — | recall_at_5 | 0.900 | 0.800 |

_Collection `test_coll` · k=5 · scored=10 · judged=5 · skipped=1_

### Regressions vs Baseline

Baseline `synpack` @ 2026-05-01T00:00:00+00:00 → current @ 2026-05-30T00:00:00+00:00

| qtype | metric | baseline | current | delta |
| --- | --- | --- | --- | --- |
| factoid | recall | 0.900 | 0.800 | -0.100 |

Gate flipped PASS → FAIL vs baseline.

New qtypes (no baseline): newbie
Dropped qtypes (in baseline, absent now): dropme"""


def test_markdown_snapshot() -> None:
    """render_baseline_diff_markdown equals the explicit expected snapshot."""
    from src.eval.runner.reporting import render_baseline_diff_markdown

    out = render_baseline_diff_markdown(_current_payload(), _baseline_payload())
    assert out == EXPECTED_MARKDOWN


# ---------------------------------------------------------------------------
# 2. Regression vs improvement vs unchanged (mutation-aware, distinct values)
# ---------------------------------------------------------------------------


def test_compute_diff_regression_vs_improvement_vs_unchanged() -> None:
    """Each metric lands in the right bucket with the right delta."""
    from src.eval.runner.baseline import compute_baseline_diff

    baseline = {
        "pack_name": "p",
        "timestamp": "t0",
        "passed": True,
        "recall_by_qtype": {"factoid": 0.90},
        "mrr_by_qtype": {"factoid": 0.70},
        "faithfulness_by_qtype": {"factoid": 0.80},
    }
    current = {
        "pack_name": "p",
        "timestamp": "t1",
        "passed": True,
        "recall_by_qtype": {"factoid": 0.80},  # -0.10 regression
        "mrr_by_qtype": {"factoid": 0.85},  # +0.15 improvement
        "faithfulness_by_qtype": {"factoid": 0.78},  # -0.02 unchanged
    }
    diff = compute_baseline_diff(baseline, current)

    reg = {(d.qtype, d.metric): d for d in diff.regressions}
    imp = {(d.qtype, d.metric): d for d in diff.improvements}
    unc = {(d.qtype, d.metric): d for d in diff.unchanged}

    assert set(reg) == {("factoid", "recall")}
    assert set(imp) == {("factoid", "mrr")}
    assert set(unc) == {("factoid", "faithfulness")}

    assert reg[("factoid", "recall")].regressed is True
    assert abs(reg[("factoid", "recall")].delta - (-0.10)) < 1e-9
    assert abs(imp[("factoid", "mrr")].delta - 0.15) < 1e-9
    assert imp[("factoid", "mrr")].regressed is False
    assert abs(unc[("factoid", "faithfulness")].delta - (-0.02)) < 1e-9
    assert unc[("factoid", "faithfulness")].regressed is False


# ---------------------------------------------------------------------------
# 3. Epsilon boundary (discriminating pair: -0.05 NOT regressed, -0.06 regressed)
# ---------------------------------------------------------------------------


def test_compute_diff_epsilon_boundary() -> None:
    """Strict ``current < baseline - epsilon``: -0.05 passes, -0.06 regresses."""
    from src.eval.runner.baseline import compute_baseline_diff

    baseline = {
        "pack_name": "p",
        "timestamp": "t0",
        "passed": True,
        "recall_by_qtype": {"edge": 0.80, "over": 0.80},
        "mrr_by_qtype": {},
        "faithfulness_by_qtype": {},
    }
    current = {
        "pack_name": "p",
        "timestamp": "t1",
        "passed": True,
        "recall_by_qtype": {"edge": 0.75, "over": 0.74},  # -0.05 / -0.06
        "mrr_by_qtype": {},
        "faithfulness_by_qtype": {},
    }
    diff = compute_baseline_diff(baseline, current, epsilon=0.05)

    regressed = {(d.qtype, d.metric) for d in diff.regressions}
    # -0.06 regresses; -0.05 does NOT (boundary partner just one step below).
    assert ("over", "recall") in regressed
    assert ("edge", "recall") not in regressed
    # The -0.05 one is within tolerance (|delta| <= eps) -> unchanged bucket.
    unchanged = {(d.qtype, d.metric) for d in diff.unchanged}
    assert ("edge", "recall") in unchanged


# ---------------------------------------------------------------------------
# 4. New + dropped qtypes do not pollute regressions
# ---------------------------------------------------------------------------


def test_compute_diff_new_and_dropped_qtypes() -> None:
    """One-sided qtypes feed new_qtypes/dropped_qtypes, not regressions."""
    from src.eval.runner.baseline import compute_baseline_diff

    baseline = {
        "pack_name": "p",
        "timestamp": "t0",
        "passed": True,
        "recall_by_qtype": {"gone": 0.90},
        "mrr_by_qtype": {},
        "faithfulness_by_qtype": {},
    }
    current = {
        "pack_name": "p",
        "timestamp": "t1",
        "passed": True,
        "recall_by_qtype": {"fresh": 0.10},  # low, but new -> not a regression
        "mrr_by_qtype": {},
        "faithfulness_by_qtype": {},
    }
    diff = compute_baseline_diff(baseline, current)

    assert diff.new_qtypes == ("fresh",)
    assert diff.dropped_qtypes == ("gone",)
    assert diff.regressions == ()


# ---------------------------------------------------------------------------
# 5. Gate flip
# ---------------------------------------------------------------------------


def test_compute_diff_gate_flip() -> None:
    """baseline passed + current failed => gate_flipped_to_fail True."""
    from src.eval.runner.baseline import compute_baseline_diff

    base = {
        "pack_name": "p", "timestamp": "t0", "passed": True,
        "recall_by_qtype": {}, "mrr_by_qtype": {}, "faithfulness_by_qtype": {},
    }
    cur = {
        "pack_name": "p", "timestamp": "t1", "passed": False,
        "recall_by_qtype": {}, "mrr_by_qtype": {}, "faithfulness_by_qtype": {},
    }
    diff = compute_baseline_diff(base, cur)
    assert diff.gate_flipped_to_fail is True
    assert diff.passed_baseline is True
    assert diff.passed_current is False


def test_compute_diff_gate_not_flipped_when_both_pass() -> None:
    """Partner: both pass => gate_flipped_to_fail False."""
    from src.eval.runner.baseline import compute_baseline_diff

    base = {
        "pack_name": "p", "timestamp": "t0", "passed": True,
        "recall_by_qtype": {}, "mrr_by_qtype": {}, "faithfulness_by_qtype": {},
    }
    cur = {
        "pack_name": "p", "timestamp": "t1", "passed": True,
        "recall_by_qtype": {}, "mrr_by_qtype": {}, "faithfulness_by_qtype": {},
    }
    diff = compute_baseline_diff(base, cur)
    assert diff.gate_flipped_to_fail is False


# ---------------------------------------------------------------------------
# 6. promote-baseline writes to pack dir
# ---------------------------------------------------------------------------


def _write_fake_pack(packs_root: Path, name: str) -> Path:
    pack_dir = packs_root / name
    pack_dir.mkdir(parents=True)
    (pack_dir / "pack.yaml").write_text("name: " + name + "\n", encoding="utf-8")
    return pack_dir


def _write_report(reports_dir: Path, fname: str, payload: dict) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / fname
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_promote_baseline_writes_to_pack_dir(tmp_path: Path) -> None:
    """promote_baseline writes <pack_dir>/baseline.json equal to the report."""
    from src.eval.cli import promote_baseline

    packs_root = tmp_path / "packs"
    reports_dir = tmp_path / "reports"
    pack_dir = _write_fake_pack(packs_root, "synpack")
    payload = _current_payload()
    report_path = _write_report(reports_dir, "synpack_2026.json", payload)

    out = io.StringIO()
    rc = promote_baseline(
        "synpack",
        report_json=str(report_path),
        packs_root=str(packs_root),
        reports_dir=str(reports_dir),
        stdout=out,
    )
    assert rc == 0
    baseline_path = pack_dir / "baseline.json"
    assert baseline_path.exists()
    assert json.loads(baseline_path.read_text(encoding="utf-8")) == payload
    assert "Promoted baseline" in out.getvalue()


def test_promote_baseline_missing_pack_exits_2(tmp_path: Path) -> None:
    """Nonexistent pack dir => exit 2."""
    from src.eval.cli import promote_baseline

    packs_root = tmp_path / "packs"
    packs_root.mkdir()
    err = io.StringIO()
    rc = promote_baseline(
        "ghost", packs_root=str(packs_root), reports_dir=str(tmp_path / "r"),
        stderr=err,
    )
    assert rc == 2


def test_promote_baseline_no_report_found_exits_2(tmp_path: Path) -> None:
    """No matching report file => exit 2."""
    from src.eval.cli import promote_baseline

    packs_root = tmp_path / "packs"
    reports_dir = tmp_path / "reports"
    _write_fake_pack(packs_root, "synpack")
    reports_dir.mkdir(parents=True)
    err = io.StringIO()
    rc = promote_baseline(
        "synpack", packs_root=str(packs_root), reports_dir=str(reports_dir),
        stderr=err,
    )
    assert rc == 2


# ---------------------------------------------------------------------------
# 7. Latest-report discovery
# ---------------------------------------------------------------------------


def test_promote_baseline_discovers_latest_report(tmp_path: Path) -> None:
    """With no --report-json, the LATEST report (by sorted filename) is promoted."""
    from src.eval.cli import promote_baseline

    packs_root = tmp_path / "packs"
    reports_dir = tmp_path / "reports"
    pack_dir = _write_fake_pack(packs_root, "synpack")

    older = _baseline_payload()
    older["timestamp"] = "2026-05-01T00:00:00+00:00"
    newer = _current_payload()
    newer["timestamp"] = "2026-05-30T00:00:00+00:00"
    # Timestamp embedded in name; sorted ascending => last is newest.
    _write_report(reports_dir, "synpack_2026-05-01T000000+0000.json", older)
    _write_report(reports_dir, "synpack_2026-05-30T000000+0000.json", newer)

    out = io.StringIO()
    rc = promote_baseline(
        "synpack", packs_root=str(packs_root), reports_dir=str(reports_dir),
        stdout=out,
    )
    assert rc == 0
    promoted = json.loads((pack_dir / "baseline.json").read_text(encoding="utf-8"))
    assert promoted == newer
    assert promoted["timestamp"] == "2026-05-30T00:00:00+00:00"
