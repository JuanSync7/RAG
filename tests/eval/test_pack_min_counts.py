"""P2.0 — per-qtype golden quantity gates.

Tests that ``Thresholds.min_goldens_per_qtype`` drives a deficit-listing
``PackValidationError`` in ``validate_pack`` while leaving the
backward-compat default (`{}`) inert.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from src.eval.pack import (
    PackValidationError,
    Thresholds,
    load_pack,
    validate_pack,
)

EXAMPLE_PACK = Path(__file__).resolve().parents[2] / "evals" / "packs" / "_example_"

DECLARED_MINS = {
    "factoid": 20,
    "qfs": 10,
    "multi_topic": 8,
    "multi_aspect": 8,
    "adversarial": 5,
    "out_of_corpus": 5,
    "messy": 5,
}


def _copy_pack(tmp_path: Path) -> Path:
    dst = tmp_path / "pack"
    shutil.copytree(EXAMPLE_PACK, dst)
    return dst


def _write_thresholds(pack_root: Path, *, min_goldens_per_qtype: dict[str, int]) -> None:
    th_path = pack_root / "thresholds.yaml"
    data = yaml.safe_load(th_path.read_text())
    data["min_goldens_per_qtype"] = min_goldens_per_qtype
    th_path.write_text(yaml.safe_dump(data, sort_keys=False))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_validator_rejects_undersized_pack(tmp_path: Path) -> None:
    pack = _copy_pack(tmp_path)
    _write_thresholds(pack, min_goldens_per_qtype=DECLARED_MINS)

    # One valid row per qtype, each in its own JSONL named after qtype.
    goldens_dir = pack / "goldens"
    # factoid already exists with f001; rewrite to be sure it has exactly 1 row.
    _write_jsonl(
        goldens_dir / "factoid.jsonl",
        [
            {
                "qid": "f001",
                "qtype": "factoid",
                "query": "What is the clock frequency?",
                "expected_answer_span": "160 MHz",
                "expected_source_docs": ["docs/intro.md"],
            }
        ],
    )
    for qtype in ("qfs", "multi_topic", "multi_aspect", "adversarial", "out_of_corpus", "messy"):
        _write_jsonl(
            goldens_dir / f"{qtype}.jsonl",
            [{"qid": f"{qtype[0]}001", "qtype": qtype, "query": f"sample {qtype} query"}],
        )

    with pytest.raises(PackValidationError) as exc_info:
        validate_pack(pack)

    msg = str(exc_info.value)
    assert "min_goldens_per_qtype" in msg
    for qtype, declared in DECLARED_MINS.items():
        assert qtype in msg, f"deficit message missing qtype {qtype!r}: {msg}"
        # Each qtype actual count is 1, declared as above.
        assert f"declared {declared}" in msg, f"deficit message missing 'declared {declared}': {msg}"
    assert "found 1" in msg


def test_validator_accepts_when_min_counts_unset() -> None:
    # The repo's _example_ pack does NOT declare min_goldens_per_qtype.
    pack = load_pack(EXAMPLE_PACK)
    assert pack.thresholds.min_goldens_per_qtype == {}


def test_validator_accepts_when_counts_met(tmp_path: Path) -> None:
    pack = _copy_pack(tmp_path)
    _write_thresholds(pack, min_goldens_per_qtype={"factoid": 3})

    _write_jsonl(
        pack / "goldens" / "factoid.jsonl",
        [
            {
                "qid": f"f{i:03d}",
                "qtype": "factoid",
                "query": f"Q{i}",
                "expected_answer_span": "160 MHz",
                "expected_source_docs": ["docs/intro.md"],
            }
            for i in range(1, 5)  # 4 rows, > declared 3
        ],
    )

    loaded = load_pack(pack)
    assert len(loaded.goldens["factoid"]) == 4
    assert loaded.thresholds.min_goldens_per_qtype == {"factoid": 3}


def test_thresholds_schema_exposes_min_goldens_field() -> None:
    t = Thresholds(profile="generic", min_goldens_per_qtype={"factoid": 20})
    assert t.min_goldens_per_qtype == {"factoid": 20}

    t_default = Thresholds(profile="generic")
    assert t_default.min_goldens_per_qtype == {}
