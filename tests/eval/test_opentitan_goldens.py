# @summary
# P2 tests — OpenTitan goldens authoring contract.
# Exports: test_pack_validates_under_count_gate, test_per_qtype_min_counts_pinned,
#   test_actual_goldens_meet_or_exceed_minimums,
#   test_factoid_spans_are_verbatim_in_source_docs,
#   test_all_expected_source_docs_resolve_to_manifest,
#   test_qids_unique_within_each_qtype.
# Deps: pytest, pyyaml, src.eval.pack.
# @end-summary
"""Goldens authoring tests for the opentitan_riscv reference pack.

These tests verify the slice DoD for P2 — that the seven goldens files
exist, meet the declared per-qtype minimums, are grounded against the
real corpus docs, and resolve through the manifest. They use only the
public ``validate_pack`` / ``PackValidationError`` API; everything else
reads the JSONL / YAML / Markdown by hand so failure messages can pin
the offending qid and path.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from src.eval.pack import PackValidationError, validate_pack

PACK_ROOT = Path("evals/packs/opentitan_riscv")
GOLDENS_DIR = PACK_ROOT / "goldens"
DOCS_DIR = PACK_ROOT / "corpus"
THRESHOLDS_PATH = PACK_ROOT / "thresholds.yaml"
MANIFEST_PATH = PACK_ROOT / "corpus" / "manifest.json"

QTYPES = (
    "factoid",
    "qfs",
    "multi_topic",
    "multi_aspect",
    "adversarial",
    "out_of_corpus",
    "messy",
)

EXPECTED_MIN_COUNTS = {
    "factoid": 20,
    "qfs": 10,
    "multi_topic": 8,
    "multi_aspect": 8,
    "adversarial": 5,
    "out_of_corpus": 5,
    "messy": 5,
}


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _manifest_paths() -> set[str]:
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {entry["path"] for entry in data["entries"]}


def test_pack_validates_under_count_gate() -> None:
    """validate_pack returns None once thresholds + counts are satisfied."""
    try:
        result = validate_pack(str(PACK_ROOT))
    except PackValidationError as exc:  # pragma: no cover - failure path
        pytest.fail(f"validate_pack raised PackValidationError: {exc}")
    assert result is None


def test_per_qtype_min_counts_pinned() -> None:
    """thresholds.yaml pins the exact P2.0 per-qtype minimums."""
    data = yaml.safe_load(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    assert "min_goldens_per_qtype" in data, "thresholds.yaml missing min_goldens_per_qtype block"
    assert data["min_goldens_per_qtype"] == EXPECTED_MIN_COUNTS, (
        f"min_goldens_per_qtype mismatch: got {data['min_goldens_per_qtype']!r}, "
        f"expected {EXPECTED_MIN_COUNTS!r}"
    )


def test_actual_goldens_meet_or_exceed_minimums() -> None:
    """Each goldens/<qtype>.jsonl has >= declared minimum non-blank rows."""
    for qtype, minimum in EXPECTED_MIN_COUNTS.items():
        path = GOLDENS_DIR / f"{qtype}.jsonl"
        assert path.exists(), f"missing goldens file: {path}"
        non_blank = sum(
            1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        assert non_blank >= minimum, (
            f"goldens/{qtype}.jsonl has {non_blank} rows, need >= {minimum}"
        )


def test_factoid_spans_are_verbatim_in_source_docs() -> None:
    """Every factoid's expected_answer_span is a verbatim substring of doc[0]."""
    rows = _read_jsonl(GOLDENS_DIR / "factoid.jsonl")
    assert rows, "factoid.jsonl is empty"
    for row in rows:
        qid = row["qid"]
        span = row["expected_answer_span"]
        docs = row.get("expected_source_docs") or []
        assert docs, f"qid={qid}: factoid has empty expected_source_docs"
        doc_rel = docs[0]
        doc_path = PACK_ROOT / "corpus" / doc_rel
        assert doc_path.exists(), f"qid={qid}: source doc not found at {doc_path}"
        doc_content = doc_path.read_text(encoding="utf-8")
        if span not in doc_content:
            pytest.fail(
                f"qid={qid}: expected_answer_span not verbatim in {doc_rel}\n"
                f"  span: {span!r}"
            )


def test_all_expected_source_docs_resolve_to_manifest() -> None:
    """Every expected_source_docs entry resolves to a manifest path."""
    allowlist = _manifest_paths()
    for qtype in QTYPES:
        path = GOLDENS_DIR / f"{qtype}.jsonl"
        if not path.exists():
            continue
        for row in _read_jsonl(path):
            for doc_rel in row.get("expected_source_docs", []) or []:
                if doc_rel not in allowlist:
                    pytest.fail(
                        f"qid={row['qid']} ({qtype}): expected_source_docs entry "
                        f"{doc_rel!r} is not in corpus/manifest.json"
                    )


def test_qids_unique_within_each_qtype() -> None:
    """No duplicate qid values within any single goldens/<qtype>.jsonl."""
    for qtype in QTYPES:
        path = GOLDENS_DIR / f"{qtype}.jsonl"
        if not path.exists():
            continue
        qids = [row["qid"] for row in _read_jsonl(path)]
        dupes = [qid for qid, count in Counter(qids).items() if count > 1]
        assert not dupes, f"duplicate qids in {qtype}.jsonl: {dupes}"
