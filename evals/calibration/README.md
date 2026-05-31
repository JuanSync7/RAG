<!-- @summary
Versioned ground-truth fixtures for judge calibration. Holds the hand-authored
"pass"/"fail" reference set the calibration loop grades the automated judge against.
@end-summary -->

# evals/calibration/

Ground-truth fixtures for **judge calibration** (P5.0). The automated eval judge
(`src/eval/runner/judge.py`) emits continuous faithfulness scores; periodically we
re-check it against a categorical human/Tier-3 reference. This directory holds that
reference as a *versioned, tamper-evident* fixture.

## Contents

| Path | Purpose |
| --- | --- |
| `calibration_v1.jsonl` | Hand-authored calibration examples (one JSON object per line): `qid`, `qtype`, `query`, `expected_answer_span`, `chunk_texts`, and a `reference_label` of exactly `"pass"` or `"fail"`. |

## How it is consumed

`src/eval/runner/calibration_fixture.py` loads this file via
`load_calibration_fixture(...)`. The loader verifies the file's raw bytes against a
pinned sha256 (`CALIBRATION_FIXTURE_SHA`) **before** parsing, so a tampered or
accidentally re-saved fixture fails fast rather than silently shifting the ground
truth. Each line becomes a frozen `CalibrationExample`; `to_judge_question()` adapts
it into a `JudgeQuestion` for the scoring path.

## Editing the fixture

The reference set is pinned by digest. After editing `calibration_v1.jsonl`,
regenerate the pin and update `CALIBRATION_FIXTURE_SHA` in
`src/eval/runner/calibration_fixture.py`:

```sh
sha256sum evals/calibration/calibration_v1.jsonl
```

Keep `reference_label` values restricted to `"pass"`/`"fail"` — any other value is
rejected at load time (it is the binary contract consumed by `compute_calibration`).
