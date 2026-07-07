# @summary
# Tests for the read-only `show-trend` eval CLI subcommand + its pure
# render_trend_table formatter. Drives the public CLI surface (main([...]))
# with hand-seeded <pack>.history.jsonl files (no pack load / ingest / judge),
# asserting: a sustained-regressing (qtype, metric) is flagged YES; a transient
# dip is NOT; missing/empty history degrades to a friendly message at rc 0;
# --runs limits the displayed delta columns to the most-recent runs; and the
# formatter is pure (input list[dict] -> markdown string).
# Exports: test_show_trend_flags_sustained,
#          test_show_trend_missing_history_is_graceful,
#          test_show_trend_no_sustained_all_no,
#          test_render_trend_table_pure_no_sustained_vs_sustained,
#          test_show_trend_runs_limits_columns,
#          test_show_trend_runs_differs_from_window_consistent_basis,
#          test_show_trend_main_dispatch_returns_int
# Deps: src.eval.cli, src.eval.runner.show_trend
# @end-summary
"""Tests for the read-only ``show-trend`` subcommand (run-history inspection)."""
from __future__ import annotations

import json
from pathlib import Path


def _write_history(tmp_path: Path, pack: str, entries: list[dict]) -> None:
    """Hand-seed a ``<pack>.history.jsonl`` index (one JSON line per entry)."""
    path = tmp_path / f"{pack}.history.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")


def _entry(ts: str, per_qtype_deltas: dict) -> dict:
    """A minimal history entry carrying only what show-trend reads."""
    return {
        "timestamp": ts,
        "pack_name": "p",
        "k": 5,
        "gate_passed": True,
        "per_query_deltas": {},
        "per_qtype_deltas": per_qtype_deltas,
    }


# ---------------------------------------------------------------------------
# Headline RED test — drives the public CLI surface.
# ---------------------------------------------------------------------------


def test_show_trend_flags_sustained(tmp_path: Path, capsys) -> None:
    """A (qtype, metric) regressing in >= min_regressed of the window is YES.

    factoid/recall regresses (delta < -epsilon) in 2 of the last 3 runs →
    sustained YES; lookup/mrr stays flat → NO. Read-only: rc 0, markdown table.
    """
    from src.eval.cli import main

    pack = "p"
    _write_history(
        tmp_path,
        pack,
        [
            _entry("2026-01-01T00:00:00", {"factoid": {"recall": 0.01},
                                           "lookup": {"mrr": 0.0}}),
            _entry("2026-01-02T00:00:00", {"factoid": {"recall": -0.20},
                                           "lookup": {"mrr": 0.01}}),
            _entry("2026-01-03T00:00:00", {"factoid": {"recall": -0.30},
                                           "lookup": {"mrr": 0.0}}),
        ],
    )

    rc = main(["show-trend", pack, "--reports-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    # Markdown table present.
    assert "| qtype | metric |" in out
    # factoid/recall row flagged sustained YES.
    factoid_row = [ln for ln in out.splitlines()
                   if ln.startswith("| factoid | recall |")]
    assert factoid_row, out
    assert "YES" in factoid_row[0]
    # lookup/mrr row present and NOT sustained.
    lookup_row = [ln for ln in out.splitlines()
                  if ln.startswith("| lookup | mrr |")]
    assert lookup_row, out
    assert "NO" in lookup_row[0]
    assert "YES" not in lookup_row[0]


def test_show_trend_missing_history_is_graceful(tmp_path: Path, capsys) -> None:
    """A pack with no history index → friendly message, rc 0 (never a trace)."""
    from src.eval.cli import main

    rc = main(["show-trend", "neverran", "--reports-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "no run history" in out.lower()
    assert "Traceback" not in out


def test_show_trend_no_sustained_all_no(tmp_path: Path, capsys) -> None:
    """History where nothing regressed → every row SUSTAINED=NO, rc 0."""
    from src.eval.cli import main

    pack = "p"
    _write_history(
        tmp_path,
        pack,
        [
            _entry("2026-01-01T00:00:00", {"factoid": {"recall": 0.10}}),
            _entry("2026-01-02T00:00:00", {"factoid": {"recall": 0.02}}),
            _entry("2026-01-03T00:00:00", {"factoid": {"recall": 0.05}}),
        ],
    )

    rc = main(["show-trend", pack, "--reports-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    data_rows = [ln for ln in out.splitlines()
                 if ln.startswith("| factoid |")]
    assert data_rows
    for row in data_rows:
        assert "NO" in row
        assert "YES" not in row


def test_render_trend_table_pure_no_sustained_vs_sustained() -> None:
    """Direct unit test on render_trend_table — no CLI, no I/O.

    factoid/recall is sustained (2 of 3 regressed) → YES; lookup/mrr is a single
    transient dip (1 of 3) → NO.
    """
    from src.eval.runner.show_trend import render_trend_table

    history = [
        _entry("2026-01-01T00:00:00", {"factoid": {"recall": -0.30},
                                       "lookup": {"mrr": 0.0}}),
        _entry("2026-01-02T00:00:00", {"factoid": {"recall": 0.0},
                                       "lookup": {"mrr": -0.40}}),
        _entry("2026-01-03T00:00:00", {"factoid": {"recall": -0.20},
                                       "lookup": {"mrr": 0.0}}),
    ]
    table = render_trend_table(history)
    assert isinstance(table, str)
    assert "| qtype | metric |" in table

    factoid_row = next(ln for ln in table.splitlines()
                       if ln.startswith("| factoid | recall |"))
    assert "YES" in factoid_row

    lookup_row = next(ln for ln in table.splitlines()
                      if ln.startswith("| lookup | mrr |"))
    assert "NO" in lookup_row
    assert "YES" not in lookup_row


def test_show_trend_runs_limits_columns(tmp_path: Path, capsys) -> None:
    """--runs 2 over a 4-entry history shows only the 2 most-recent runs.

    The two most-recent timestamps appear as delta columns; the two oldest do
    not. Pins recency ordering + the runs limit (distinct from --window).
    """
    from src.eval.cli import main

    pack = "p"
    _write_history(
        tmp_path,
        pack,
        [
            _entry("2026-01-01T00:00:00", {"factoid": {"recall": 0.01}}),
            _entry("2026-01-02T00:00:00", {"factoid": {"recall": 0.01}}),
            _entry("2026-01-03T00:00:00", {"factoid": {"recall": 0.01}}),
            _entry("2026-01-04T00:00:00", {"factoid": {"recall": 0.01}}),
        ],
    )

    rc = main(["show-trend", pack, "--reports-dir", str(tmp_path), "--runs", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    # Header carries exactly the 2 most-recent run timestamps.
    assert "2026-01-04T00:00:00" in out
    assert "2026-01-03T00:00:00" in out
    assert "2026-01-02T00:00:00" not in out
    assert "2026-01-01T00:00:00" not in out


def test_show_trend_runs_differs_from_window_consistent_basis() -> None:
    """With runs != window the shown count ties to displayed runs, flag to window.

    5-entry history: the OLDEST 2 runs regress factoid/recall, the NEWEST 3 are
    clean. With --runs 2 (display newest 2 = clean) and --window 5 / min 2 the
    detector flags the pair SUSTAINED (2 regressed over its window). The row must
    show `0/2` (count over the SHOWN runs — never a phantom 2/5 referencing
    rows not displayed) AND sustained YES (the detector's wider-window verdict).
    Pins the unified displayed-basis count against the detector-basis flag.
    """
    from src.eval.runner.show_trend import render_trend_table

    regress = {"factoid": {"recall": -0.30}}
    clean = {"factoid": {"recall": 0.0}}
    history = [
        _entry("2026-01-01T00:00:00", regress),
        _entry("2026-01-02T00:00:00", regress),
        _entry("2026-01-03T00:00:00", clean),
        _entry("2026-01-04T00:00:00", clean),
        _entry("2026-01-05T00:00:00", clean),
    ]
    table = render_trend_table(history, runs=2, window=5, min_regressed=2)

    # Only the 2 newest runs are shown as columns.
    assert "2026-01-05T00:00:00" in table
    assert "2026-01-04T00:00:00" in table
    assert "2026-01-02T00:00:00" not in table

    row = next(ln for ln in table.splitlines()
               if ln.startswith("| factoid | recall |"))
    # Count is over the SHOWN runs (0 of 2 clean) — NOT the detector's 2/5.
    assert "0/2" in row, row
    assert "2/5" not in row, row
    # Flag is the detector's wider-window verdict.
    assert "YES" in row, row


def test_show_trend_main_dispatch_returns_int(tmp_path: Path) -> None:
    """main(['show-trend', ...]) returns an int exit code (argv dispatch)."""
    from src.eval.cli import main

    rc = main(["show-trend", "neverran", "--reports-dir", str(tmp_path)])
    assert isinstance(rc, int)
    assert rc == 0
