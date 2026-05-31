# @summary
# P5.0 — calibration ground-truth fixture loader contract tests (FAST, offline).
# Exports: test_load_calibration_fixture_returns_examples,
#          test_sha_guard_rejects_tampered_fixture,
#          test_loader_rejects_non_binary_reference_label,
#          test_to_judge_question_roundtrips,
#          test_calibration_example_is_frozen.
# Deps: src.eval.runner (CalibrationExample, load_calibration_fixture,
#       DEFAULT_CALIBRATION_FIXTURE_PATH, CALIBRATION_FIXTURE_SHA, JudgeQuestion).
# @end-summary
"""P5.0 — calibration ground-truth fixture loader contract tests."""
from __future__ import annotations

import dataclasses
import hashlib

import pytest

from src.eval.runner import (
    CALIBRATION_FIXTURE_SHA,
    DEFAULT_CALIBRATION_FIXTURE_PATH,
    CalibrationExample,
    JudgeQuestion,
    load_calibration_fixture,
)

_VALID_QTYPES = {
    "factoid",
    "qfs",
    "multi_topic",
    "multi_aspect",
    "adversarial",
    "out_of_corpus",
    "messy",
}


def test_load_calibration_fixture_returns_examples():
    """The default fixture loads into a healthy pass/fail mix of examples."""
    examples = load_calibration_fixture(DEFAULT_CALIBRATION_FIXTURE_PATH)

    assert len(examples) >= 10
    labels = [ex.reference_label for ex in examples]
    assert all(lbl in {"pass", "fail"} for lbl in labels)
    # Discriminating mix: an all-one-label fixture would red here.
    assert "pass" in labels
    assert "fail" in labels
    assert labels.count("fail") >= 4
    assert labels.count("pass") >= 4
    assert all(ex.qtype in _VALID_QTYPES for ex in examples)
    # qids must be unique (calibration math pairs by position; dup ids are a smell).
    qids = [ex.qid for ex in examples]
    assert len(qids) == len(set(qids))


def test_sha_guard_rejects_tampered_fixture(tmp_path):
    """The sha256 guard fires on mutated bytes and passes on faithful bytes."""
    raw = DEFAULT_CALIBRATION_FIXTURE_PATH.read_bytes()

    # Faithful copy at its own path loads fine (guard passes on identical bytes).
    faithful = tmp_path / "faithful.jsonl"
    faithful.write_bytes(raw)
    assert load_calibration_fixture(faithful)  # non-empty -> success

    # Tampered copy: append a byte -> sha mismatch -> ValueError.
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_bytes(raw + b"\n")
    with pytest.raises(ValueError) as excinfo:
        load_calibration_fixture(tampered)
    assert str(tampered) in str(excinfo.value)


def test_loader_rejects_non_binary_reference_label(tmp_path):
    """A reference_label outside {pass, fail} fails fast naming the bad qid+label."""
    bad = tmp_path / "bad.jsonl"
    line = (
        '{"qid": "bad-1", "qtype": "factoid", "query": "q?", '
        '"expected_answer_span": "x", "chunk_texts": ["c"], '
        '"reference_label": "partial"}\n'
    )
    bad.write_text(line, encoding="utf-8")
    expected_sha = hashlib.sha256(bad.read_bytes()).hexdigest()

    with pytest.raises(ValueError) as excinfo:
        load_calibration_fixture(bad, expected_sha=expected_sha)
    msg = str(excinfo.value)
    assert "bad-1" in msg
    assert "partial" in msg


def test_to_judge_question_roundtrips():
    """to_judge_question() carries qid/qtype/query/span and a tuple chunk_texts."""
    example = load_calibration_fixture(DEFAULT_CALIBRATION_FIXTURE_PATH)[0]
    jq = example.to_judge_question()

    assert isinstance(jq, JudgeQuestion)
    assert jq.qid == example.qid
    assert jq.qtype == example.qtype
    assert jq.query == example.query
    assert jq.expected_answer_span == example.expected_answer_span
    assert isinstance(jq.chunk_texts, tuple)
    assert jq.chunk_texts == example.chunk_texts


def test_calibration_example_is_frozen():
    """CalibrationExample is immutable (frozen dataclass)."""
    example = load_calibration_fixture(DEFAULT_CALIBRATION_FIXTURE_PATH)[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        example.reference_label = "pass"  # type: ignore[misc]
