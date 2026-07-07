# @summary
# Real-Docling integration test: parses an actual datasheet PDF and asserts
# that DoclingParser surfaces figures with at least one parsed caption_label.
# Skips gracefully when the fixture is absent or Docling models cannot load.
# Exports: test_real_docling_figures
# Deps: docling (real), src.ingest.support.docling, pytest
# @end-summary

"""Real-Docling figure-extraction smoke test.

Mirrors ``test_real_docling_table_smoke.py``. Mocked-sys.modules tests miss
extraction-quality bugs (memory ``feedback_real_pipeline_test_per_feature``)
so this test boots real Docling end-to-end on an ESP32-S3 datasheet — the TI
datasheets show low caption coverage, but ESP32 has clear ``Figure N-N``
captions which is what we need to validate the regex on real text.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# Search order — fall back so the test runs both in the documented fixture
# location and against the existing local datasheet copy.
_CANDIDATE_FIXTURES = (
    Path("tests/fixtures/datasheets/esp32-s3_datasheet_en.pdf"),
    Path("data/datasheets/esp32-s3_datasheet.pdf"),
)


def _resolve_fixture() -> Path | None:
    for rel in _CANDIDATE_FIXTURES:
        # Resolve relative to repo root (the worktree CWD when pytest runs).
        abs_path = Path.cwd() / rel
        if abs_path.exists() and abs_path.stat().st_size > 0:
            return abs_path
    return None


@pytest.mark.slow
@pytest.mark.integration
def test_real_docling_figures() -> None:
    fixture = _resolve_fixture()
    if fixture is None:
        pytest.skip(
            "No ESP32-S3 datasheet fixture found at any of: "
            + ", ".join(str(p) for p in _CANDIDATE_FIXTURES)
        )

    from src.ingest.common.types import IngestionConfig
    from src.ingest.support.docling import DoclingParser

    config = IngestionConfig()
    try:
        config.docling_auto_download = True
    except Exception:
        pass

    try:
        DoclingParser.ensure_ready(config)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Docling models unavailable: {exc!r}")

    parser = DoclingParser()
    parse_result = parser.parse(fixture, config)

    figures = list(getattr(parse_result, "figures", []) or [])
    assert figures, (
        "Expected DoclingParser to surface figures from the ESP32-S3 "
        "datasheet; got an empty list."
    )

    labelled = [f for f in figures if f.caption_label]
    assert labelled, (
        "Expected at least one figure with a parsed caption_label "
        "(e.g. 'Figure 1' or 'Figure 4-1') from the ESP32-S3 datasheet. "
        f"Captured {len(figures)} figures; first 3 captions: "
        f"{[f.caption[:120] for f in figures[:3]]!r}"
    )

    # Spot-check: document_id should be the file stem.
    assert figures[0].document_id == fixture.stem
