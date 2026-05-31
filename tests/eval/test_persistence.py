# @summary
# P8a.0 — tests for persist_eval_report: writes a parseable JSON report file
# whose eval payload matches _emit_json's shape (single-source proof), plus
# top-level pack_name + timestamp, with a deterministic injected-now filename.
# Reuses _make_synthetic_pack / _patch_full_chain from tests.eval.test_cli.
# Exports: test_persist_writes_parseable_json_with_full_shape,
#          test_persist_path_uses_pack_name_and_injected_timestamp,
#          test_persist_payload_matches_emit_json_shape
# Deps: src.eval.run (run_eval); src.eval.runner (persist_eval_report);
#       src.eval.cli (_emit_json); tests.eval.test_cli
# @end-summary
"""P8a.0 — persist_eval_report tests."""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path

from tests.eval.test_cli import _make_synthetic_pack, _patch_full_chain


def _run(tmp_path: Path, monkeypatch, *, judge_score: float = 0.9):
    from src.eval import run_eval

    pack_dir = _make_synthetic_pack(
        tmp_path, recall_floor=0.5, faithfulness_floor=0.5
    )
    _patch_full_chain(monkeypatch, judge_score=judge_score)
    return run_eval(str(pack_dir))


def test_persist_writes_parseable_json_with_full_shape(
    tmp_path: Path, monkeypatch
) -> None:
    """persist_eval_report writes a parseable JSON file carrying the full key
    set INCLUDING pack_name, timestamp, mrr_by_qtype, the three per_query maps,
    and failures[].qid."""
    from src.eval.runner import persist_eval_report

    result = _run(tmp_path, monkeypatch)
    out_dir = tmp_path / "reports"
    path = persist_eval_report(result, out_dir)

    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))

    for key in (
        "pack_name",
        "timestamp",
        "collection_name",
        "k",
        "passed",
        "failures",
        "recall_by_qtype",
        "faithfulness_by_qtype",
        "mrr_by_qtype",
        "per_query_recall",
        "per_query_mrr",
        "per_query_faithfulness",
        "total_queries_scored",
        "total_queries_skipped",
        "total_queries_judged",
    ):
        assert key in payload, f"persisted payload missing key {key!r}: {payload}"

    # failures entries must carry a qid field (empty for qtype-level failures).
    for failure in payload["failures"]:
        assert "qid" in failure


def test_persist_path_uses_pack_name_and_injected_timestamp(
    tmp_path: Path, monkeypatch
) -> None:
    """An injected now=<fixed datetime> makes the filename deterministic and
    embeds the pack name."""
    from src.eval.runner import persist_eval_report

    result = _run(tmp_path, monkeypatch)
    out_dir = tmp_path / "reports"
    fixed = datetime(2026, 5, 30, 12, 34, 56, tzinfo=timezone.utc)

    path = persist_eval_report(result, out_dir, now=fixed)

    # pack name from the synthetic pack is "synpack".
    assert "synpack" in path.name
    # The fixed timestamp is embedded in a filesafe form (no colons).
    assert "2026" in path.name
    assert ":" not in path.name
    # Deterministic: a second call with the same now lands on the same path.
    path2 = persist_eval_report(result, out_dir, now=fixed)
    assert path2 == path


def test_persist_payload_matches_emit_json_shape(
    tmp_path: Path, monkeypatch
) -> None:
    """The persisted eval payload's keys equal _emit_json's payload keys (minus
    the persistence-only pack_name/timestamp). Single-source drift guard."""
    from src.eval.cli import _emit_json
    from src.eval.runner import persist_eval_report

    result = _run(tmp_path, monkeypatch)

    out_dir = tmp_path / "reports"
    path = persist_eval_report(result, out_dir)
    persisted = json.loads(path.read_text(encoding="utf-8"))

    buf = io.StringIO()
    _emit_json(result.report, result.gate, stdout=buf)
    emitted = json.loads(buf.getvalue())

    persisted_eval_keys = set(persisted) - {"pack_name", "timestamp"}
    assert persisted_eval_keys == set(emitted)
