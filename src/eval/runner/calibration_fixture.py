# @summary
# P5.0 — versioned, tamper-evident ground-truth calibration fixture + loader.
# Ships a frozen CalibrationExample dataclass and load_calibration_fixture, which
# verifies the fixture's raw bytes against a pinned sha256 (CALIBRATION_FIXTURE_SHA)
# BEFORE parsing, then parses JSONL into examples and enforces that every
# reference_label is exactly "pass"/"fail" (the binary contract consumed by
# compute_calibration). to_judge_question() adapts an example into a JudgeQuestion.
# Exports: CalibrationExample, load_calibration_fixture,
#          DEFAULT_CALIBRATION_FIXTURE_PATH, CALIBRATION_FIXTURE_SHA.
# Deps: stdlib only (dataclasses, hashlib, json, pathlib);
#       .judge (JudgeQuestion contract).
# @end-summary
"""P5.0 — ground-truth calibration fixture + loader.

The judge-calibration loop (:mod:`src.eval.runner.calibration`) measures the
automated judge against a categorical ``"pass"``/``"fail"`` reference. This
module supplies that reference as a *versioned, tamper-evident* fixture: a
hand-authored JSONL file whose raw bytes are pinned by a sha256 digest. The
loader verifies the digest before parsing so a tampered (or accidentally
re-saved) fixture fails fast rather than silently shifting the ground truth.

Each line yields a frozen :class:`CalibrationExample`. :meth:`to_judge_question`
adapts an example into a :class:`~src.eval.runner.judge.JudgeQuestion` for the
scoring path (which this module does not itself invoke).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .judge import JudgeQuestion

# Repo root: this file is at src/eval/runner/calibration_fixture.py, so
# parents[0]=runner, [1]=eval, [2]=src, [3]=repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CALIBRATION_FIXTURE_PATH = (
    _REPO_ROOT / "evals" / "calibration" / "calibration_v1.jsonl"
)

# Regenerate after editing the fixture: sha256sum evals/calibration/calibration_v1.jsonl
CALIBRATION_FIXTURE_SHA = (
    "f739d6cf3a2a6718657f1ec184bcd834e7a96c4b5e56117fafbcf054399b126f"
)

# The binary reference vocabulary consumed by compute_calibration; any other
# label is rejected at load time (fail fast on a corrupted ground truth).
_VALID_REFERENCE_LABELS = frozenset({"pass", "fail"})


@dataclass(frozen=True)
class CalibrationExample:
    """A single hand-authored calibration ground-truth example.

    Frozen (and ``chunk_texts`` is a tuple) so examples are hashable and the
    pinned ground truth cannot be mutated in place after load.

    :param qid: stable identifier for the example.
    :param qtype: the eval question type (e.g. ``"factoid"``, ``"adversarial"``).
    :param query: the user query posed against the retrieved chunks.
    :param expected_answer_span: the verbatim grounding span, or ``None`` when
        the example has no single extractable span (e.g. summary/out-of-corpus).
    :param chunk_texts: the retrieved chunk texts presented to the judge.
    :param reference_label: the human reference verdict, exactly ``"pass"`` or
        ``"fail"``.
    """

    qid: str
    qtype: str
    query: str
    expected_answer_span: str | None
    chunk_texts: tuple[str, ...]
    reference_label: str

    def to_judge_question(self) -> JudgeQuestion:
        """Adapt this example into a :class:`JudgeQuestion` for the scoring path.

        The ``reference_label`` is intentionally dropped: it is the ground truth
        the judge is graded *against*, not an input to the judge.
        """
        return JudgeQuestion(
            qid=self.qid,
            qtype=self.qtype,
            query=self.query,
            expected_answer_span=self.expected_answer_span,
            chunk_texts=self.chunk_texts,
        )


def load_calibration_fixture(
    fixture_path: Path | str = DEFAULT_CALIBRATION_FIXTURE_PATH,
    *,
    expected_sha: str | None = None,
) -> list[CalibrationExample]:
    """Load, integrity-check, and parse the calibration fixture JSONL.

    The sha256 of the file's *raw bytes* is verified against ``expected_sha``
    (default: the pinned :data:`CALIBRATION_FIXTURE_SHA`) BEFORE any parsing, so
    a tampered or re-saved fixture fails fast. Each non-blank line is parsed into
    a :class:`CalibrationExample`; every ``reference_label`` must be exactly
    ``"pass"``/``"fail"``.

    :param fixture_path: path to the JSONL fixture. Defaults to the pinned
        :data:`DEFAULT_CALIBRATION_FIXTURE_PATH`.
    :param expected_sha: sha256 hex digest the file's bytes must match. Defaults
        to :data:`CALIBRATION_FIXTURE_SHA`; tests may pass the digest of a
        temporary fixture to exercise the parse path without weakening the
        production default.
    :returns: the parsed examples, in file order.
    :raises ValueError: if the sha256 does not match, or if any line carries a
        ``reference_label`` outside ``{"pass", "fail"}``.
    """
    path = Path(fixture_path)
    pinned_sha = CALIBRATION_FIXTURE_SHA if expected_sha is None else expected_sha

    # 1) Integrity guard FIRST — a tampered file must fail before we trust it.
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha != pinned_sha:
        raise ValueError(
            f"calibration fixture sha256 mismatch for {path}: "
            f"expected {pinned_sha}, got {actual_sha}. "
            "Regenerate after editing: sha256sum "
            "evals/calibration/calibration_v1.jsonl"
        )

    # 2) Parse JSONL into examples.
    examples: list[CalibrationExample] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        record = json.loads(line)

        # 3) Enforce the binary reference vocabulary, fail fast naming the row.
        label = record["reference_label"]
        if label not in _VALID_REFERENCE_LABELS:
            raise ValueError(
                f"calibration fixture {path}: example {record.get('qid')!r} has "
                f"invalid reference_label {label!r}; expected 'pass' or 'fail'"
            )

        examples.append(
            CalibrationExample(
                qid=record["qid"],
                qtype=record["qtype"],
                query=record["query"],
                expected_answer_span=record["expected_answer_span"],
                # JSON gives a list; tuple keeps the dataclass hashable/frozen.
                chunk_texts=tuple(record["chunk_texts"]),
                reference_label=label,
            )
        )

    return examples
