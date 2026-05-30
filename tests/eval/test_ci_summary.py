# @summary
# P7d — tests for the CI eval-gate markdown summary renderer + CLI write seam.
# Covers render_ci_summary (PASS/FAIL status, per-qtype table with actuals,
# faithfulness "—" for absent qtypes, sorted determinism, valid GFM tables,
# failures section) and the run_cli --summary-file / $GITHUB_STEP_SUMMARY
# write seam (pass path, env-var path, neither-set no-op, write-on-gate-fail).
# Mutation A target: test_render_fail_status
# Mutation B target: test_cli_writes_summary_via_env_var
# Mutation C target: test_render_failures_section_lists_each
# Mutation D target: test_cli_writes_summary_on_gate_fail
# Exports: test_render_pass_status, test_render_pass_no_failures_rows,
#          test_render_fail_status, test_render_failures_section_lists_each,
#          test_render_per_qtype_status_pass_and_fail,
#          test_render_faithfulness_absent_renders_dash,
#          test_render_deterministic_sorted, test_render_valid_markdown_tables,
#          test_cli_writes_summary_via_file_arg,
#          test_cli_writes_summary_via_env_var,
#          test_cli_no_summary_when_neither_set,
#          test_cli_writes_summary_on_gate_fail
# Deps: src.eval.runner.ci, src.eval.runner.report, src.eval.runner.gate,
#       src.eval.cli; tests.eval.test_cli (shared synthetic-pack + chain stubs)
# @end-summary
"""P7d — CI summary renderer + CLI write-seam tests."""
from __future__ import annotations

from pathlib import Path

from tests.eval.test_cli import _make_synthetic_pack, _patch_full_chain


# ---------------------------------------------------------------------------
# Fixtures: small in-memory EvalReport + GateResult builders
# ---------------------------------------------------------------------------


def _report(
    *,
    recall=None,
    faithfulness=None,
    k: int = 5,
    collection: str = "test_coll",
    scored: int = 7,
    judged: int = 4,
    skipped: int = 2,
):
    from src.eval.runner.report import EvalReport

    return EvalReport(
        collection_name=collection,
        k=k,
        per_query_recall={},
        recall_by_qtype=recall or {},
        total_queries_scored=scored,
        total_queries_skipped=skipped,
        faithfulness_by_qtype=faithfulness or {},
        total_queries_judged=judged,
    )


def _gate(*failures):
    from src.eval.runner.gate import GateResult

    failures_t = tuple(failures)
    return GateResult(passed=not failures_t, failures=failures_t)


def _failure(qtype: str, metric: str, expected: float, actual: float):
    from src.eval.runner.gate import GateFailure

    return GateFailure(qtype=qtype, metric=metric, expected=expected, actual=actual)


# ---------------------------------------------------------------------------
# render_ci_summary — PASS path
# ---------------------------------------------------------------------------


def test_render_pass_status() -> None:
    """A passing gate renders a PASS status line."""
    from src.eval.runner.ci import render_ci_summary

    report = _report(recall={"factoid": 0.875}, faithfulness={"factoid": 0.912})
    md = render_ci_summary(report, _gate())
    assert "**Status: PASS**" in md
    assert "FAIL" not in md  # no failures section, no fail status


def test_render_pass_no_failures_rows() -> None:
    """On pass: per-qtype actuals present, and no failures expected/actual rows."""
    from src.eval.runner.ci import render_ci_summary

    report = _report(recall={"factoid": 0.875}, faithfulness={"factoid": 0.912})
    md = render_ci_summary(report, _gate())
    assert "factoid" in md
    assert "0.875" in md  # recall actual
    assert "0.912" in md  # faithfulness actual
    # No expected-vs-actual content: the failures section states none.
    assert "expected" not in md.lower() or "none" in md.lower()


# ---------------------------------------------------------------------------
# render_ci_summary — FAIL path
# ---------------------------------------------------------------------------


def test_render_fail_status() -> None:
    """A failing gate renders a FAIL status line.

    MUTATION A TARGET: emitting PASS regardless of gate.passed reds this.
    """
    from src.eval.runner.ci import render_ci_summary

    report = _report(recall={"factoid": 0.400}, faithfulness={"factoid": 0.912})
    gate = _gate(_failure("factoid", "recall_at_5", 0.700, 0.400))
    md = render_ci_summary(report, gate)
    # Assert the STATUS line specifically (not merely "FAIL" anywhere — the
    # failures table also contains FAIL, which would insulate a status mutation).
    assert "**Status: FAIL**" in md
    assert "**Status: PASS**" not in md


def test_render_failures_section_lists_each() -> None:
    """Each gate failure appears with its distinct expected + actual values.

    MUTATION C TARGET: dropping the failure rows reds this.
    """
    from src.eval.runner.ci import render_ci_summary

    # Distinct values, no algebraic coincidences.
    report = _report(
        recall={"factoid": 0.400, "multi_hop": 0.250},
        faithfulness={"factoid": 0.912, "multi_hop": 0.330},
    )
    gate = _gate(
        _failure("factoid", "recall_at_5", 0.700, 0.400),
        _failure("multi_hop", "faithfulness", 0.600, 0.330),
    )
    md = render_ci_summary(report, gate)
    # Both expected and both actual values rendered in the failures section.
    assert "0.700" in md  # factoid recall expected
    assert "0.400" in md  # factoid recall actual
    assert "0.600" in md  # multi_hop faithfulness expected
    assert "0.330" in md  # multi_hop faithfulness actual
    assert "recall_at_5" in md
    assert "faithfulness" in md


def test_render_per_qtype_status_pass_and_fail() -> None:
    """The offending qtype shows FAIL; a clean qtype shows PASS in the table."""
    from src.eval.runner.ci import render_ci_summary

    report = _report(
        recall={"factoid": 0.400, "clean": 0.950},
        faithfulness={"factoid": 0.912, "clean": 0.880},
    )
    gate = _gate(_failure("factoid", "recall_at_5", 0.700, 0.400))
    md = render_ci_summary(report, gate)
    lines = md.splitlines()
    factoid_line = next(l for l in lines if l.lstrip("| ").startswith("factoid"))
    clean_line = next(l for l in lines if l.lstrip("| ").startswith("clean"))
    assert "FAIL" in factoid_line
    assert "PASS" in clean_line


def test_render_faithfulness_absent_renders_dash() -> None:
    """A qtype with recall but no faithfulness renders an em-dash, not 0.000."""
    from src.eval.runner.ci import render_ci_summary

    report = _report(recall={"factoid": 0.875}, faithfulness={})
    md = render_ci_summary(report, _gate())
    lines = md.splitlines()
    factoid_line = next(l for l in lines if l.lstrip("| ").startswith("factoid"))
    assert "—" in factoid_line
    assert "0.000" not in factoid_line


# ---------------------------------------------------------------------------
# render_ci_summary — determinism + valid markdown
# ---------------------------------------------------------------------------


def test_render_deterministic_sorted() -> None:
    """qtype rows + failure rows are sorted; two calls yield identical strings."""
    from src.eval.runner.ci import render_ci_summary

    report = _report(
        recall={"zeta": 0.500, "alpha": 0.600, "mu": 0.700},
        faithfulness={"zeta": 0.800, "alpha": 0.810},
    )
    gate = _gate(
        _failure("zeta", "recall_at_5", 0.900, 0.500),
        _failure("alpha", "faithfulness", 0.950, 0.810),
    )
    md1 = render_ci_summary(report, gate)
    md2 = render_ci_summary(report, gate)
    assert md1 == md2

    lines = md1.splitlines()
    # qtype rows appear in sorted order: alpha < mu < zeta.
    qrows = [l for l in lines if l.lstrip("| ").split(" ")[0].split("|")[0].strip()
             in ("alpha", "mu", "zeta")]
    qnames = [l.lstrip("| ").split("|")[0].strip() for l in qrows]
    # Filter to just the metrics-table rows (each qtype once before failures).
    first_seen: list[str] = []
    for n in qnames:
        if n not in first_seen:
            first_seen.append(n)
    assert first_seen[:3] == ["alpha", "mu", "zeta"]


def test_render_valid_markdown_tables() -> None:
    """Tables include a header separator row (---|---) so GFM renders them."""
    from src.eval.runner.ci import render_ci_summary

    report = _report(recall={"factoid": 0.400}, faithfulness={"factoid": 0.912})
    gate = _gate(_failure("factoid", "recall_at_5", 0.700, 0.400))
    md = render_ci_summary(report, gate)
    assert "---" in md
    assert "|" in md
    # Separator row: a line composed only of pipes, dashes, colons, spaces.
    sep_lines = [
        l for l in md.splitlines()
        if l.strip() and set(l.strip()) <= set("|-: ")
    ]
    assert sep_lines, "no markdown table separator row found"


# ---------------------------------------------------------------------------
# CLI write seam
# ---------------------------------------------------------------------------


def test_cli_writes_summary_via_file_arg(tmp_path: Path, monkeypatch) -> None:
    """--summary-file PATH writes the rendered markdown on gate-pass."""
    from src.eval.cli import run_cli

    pack_dir = _make_synthetic_pack(tmp_path, recall_floor=0.5, faithfulness_floor=0.5)
    _patch_full_chain(monkeypatch, judge_score=0.9)
    summary = tmp_path / "summary.md"
    # Ensure env var does not leak into this test.
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    rc = run_cli(str(pack_dir), summary_file=str(summary))
    assert rc == 0
    assert summary.exists()
    text = summary.read_text()
    assert "factoid" in text
    assert "PASS" in text


def test_cli_writes_summary_via_env_var(tmp_path: Path, monkeypatch) -> None:
    """$GITHUB_STEP_SUMMARY is auto-detected when --summary-file is absent.

    MUTATION B TARGET: honoring only --summary-file (ignoring env) reds this.
    """
    from src.eval.cli import run_cli

    pack_dir = _make_synthetic_pack(tmp_path, recall_floor=0.5, faithfulness_floor=0.5)
    _patch_full_chain(monkeypatch, judge_score=0.9)
    summary = tmp_path / "step_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    rc = run_cli(str(pack_dir), summary_file=None)
    assert rc == 0
    assert summary.exists()
    assert "factoid" in summary.read_text()


def test_cli_no_summary_when_neither_set(tmp_path: Path, monkeypatch) -> None:
    """Neither --summary-file nor env set → no crash, no file written."""
    from src.eval.cli import run_cli

    pack_dir = _make_synthetic_pack(tmp_path, recall_floor=0.5, faithfulness_floor=0.5)
    _patch_full_chain(monkeypatch, judge_score=0.9)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    rc = run_cli(str(pack_dir), summary_file=None)
    assert rc == 0
    # No stray summary files in tmp_path beyond the synthetic pack dir.
    extras = [p for p in tmp_path.iterdir() if p.name != "synthetic_pack"]
    assert extras == [], f"unexpected files written: {extras}"


def test_cli_writes_summary_on_gate_fail(tmp_path: Path, monkeypatch) -> None:
    """On gate-FAIL (exit 1) the summary is STILL written.

    MUTATION D TARGET: skipping the write on fail reds this.
    """
    from src.eval.cli import run_cli

    pack_dir = _make_synthetic_pack(tmp_path, recall_floor=0.5, faithfulness_floor=0.5)
    # Retrieval misses → recall 0.0 < 0.5 floor → gate fails.
    _patch_full_chain(monkeypatch, retrieved_source="other.md", judge_score=0.9)
    summary = tmp_path / "fail_summary.md"
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    rc = run_cli(str(pack_dir), summary_file=str(summary))
    assert rc == 1
    assert summary.exists()
    text = summary.read_text()
    assert "FAIL" in text
    assert "factoid" in text
