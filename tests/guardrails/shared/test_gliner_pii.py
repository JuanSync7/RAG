"""Tests for src.guardrails.shared.gliner_pii.

Covers the pure ``merge_detections`` helper, the ``GLiNERPIIDetector.detect``
prediction-mapping logic (wired against a fake model, no real ML), the module
constants, and the ``__init__`` import-gating contract.

No real GLiNER model is loaded:
* ``detect`` is exercised on a bare ``__new__`` instance with a ``FakeModel``.
* ``merge_detections`` is pure and uses real ``PIIDetection`` values.
* ``__init__`` is branched on whether ``gliner`` is importable in this venv.
"""
from __future__ import annotations

import importlib.util

import pytest

from src.guardrails.shared import gliner_pii
from src.guardrails.shared.gliner_pii import GLiNERPIIDetector, merge_detections
from src.guardrails.shared.pii import PIIDetection

HAS_GLINER = importlib.util.find_spec("gliner") is not None


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeModel:
    """Stand-in for a loaded GLiNER model.

    Records the arguments forwarded to ``predict_entities`` and returns a
    canned prediction list (or raises if ``raise_exc`` is set).
    """

    def __init__(self, predictions=None, raise_exc: Exception | None = None):
        self.predictions = predictions or []
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    def predict_entities(self, text, labels, threshold):
        self.calls.append(
            {"text": text, "labels": labels, "threshold": threshold}
        )
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.predictions


def _make_detector(predictions=None, raise_exc=None, threshold=0.5):
    """Build a GLiNERPIIDetector without loading a real model."""
    det = GLiNERPIIDetector.__new__(GLiNERPIIDetector)
    det._model = FakeModel(predictions=predictions, raise_exc=raise_exc)
    det._threshold = threshold
    det._labels = list(gliner_pii._PII_LABELS)
    return det


# ---------------------------------------------------------------------------
# merge_detections
# ---------------------------------------------------------------------------


def test_merge_empty_supplementary_returns_primary():
    """No supplementary detections -> primary returned unchanged."""
    primary = [
        PIIDetection("PERSON", 0, 4, "[PERSON_REDACTED]"),
        PIIDetection("EMAIL", 10, 20, "[EMAIL_REDACTED]"),
    ]
    result = merge_detections(primary, [])
    assert result is primary


def test_merge_empty_primary_returns_supplementary_reverse_sorted():
    """No primary -> supplementary returned reverse-sorted by start."""
    supplementary = [
        PIIDetection("PERSON", 0, 4, "[PERSON_REDACTED]"),
        PIIDetection("LOCATION", 30, 38, "[LOCATION_REDACTED]"),
        PIIDetection("ORGANIZATION", 10, 18, "[ORGANIZATION_REDACTED]"),
    ]
    result = merge_detections([], supplementary)
    assert [d.start for d in result] == [30, 10, 0]
    # Original input must not be reordered in place expectations: copy returned.
    assert result is not supplementary


def test_merge_non_overlapping_supplementary_added_and_reverse_sorted():
    """A non-overlapping supplementary detection is added; result reverse-sorted."""
    primary = [PIIDetection("PERSON", 0, 5, "[PERSON_REDACTED]")]
    supplementary = [PIIDetection("LOCATION", 20, 28, "[LOCATION_REDACTED]")]
    result = merge_detections(primary, supplementary)
    assert [d.start for d in result] == [20, 0]
    assert [d.pii_type for d in result] == ["LOCATION", "PERSON"]


def test_merge_overlapping_supplementary_dropped():
    """Supplementary detection sharing any char with primary span is dropped."""
    primary = [PIIDetection("PERSON", 5, 15, "[PERSON_REDACTED]")]
    # Touches by one char (start 14 < p_end 15): overlap -> dropped.
    supplementary = [PIIDetection("LOCATION", 14, 25, "[LOCATION_REDACTED]")]
    result = merge_detections(primary, supplementary)
    assert [d.start for d in result] == [5]
    assert result[0].pii_type == "PERSON"


def test_merge_adjacent_not_overlapping_kept():
    """Adjacent supplementary (det.start == p_end) is NOT overlap -> kept."""
    primary = [PIIDetection("PERSON", 5, 15, "[PERSON_REDACTED]")]
    # start (15) == p_end (15): 15 < 15 is False -> no overlap -> kept.
    supplementary = [PIIDetection("LOCATION", 15, 25, "[LOCATION_REDACTED]")]
    result = merge_detections(primary, supplementary)
    assert [d.start for d in result] == [15, 5]
    assert [d.pii_type for d in result] == ["LOCATION", "PERSON"]


def test_merge_adjacent_left_boundary_kept():
    """Supplementary ending exactly at primary start (det.end == p_start) kept."""
    primary = [PIIDetection("PERSON", 15, 25, "[PERSON_REDACTED]")]
    # end (15) == p_start (15): 15 > 15 is False -> no overlap -> kept.
    supplementary = [PIIDetection("LOCATION", 5, 15, "[LOCATION_REDACTED]")]
    result = merge_detections(primary, supplementary)
    assert [d.start for d in result] == [15, 5]


def test_merge_mixed_overlapping_and_non_overlapping():
    """Mixed supplementary: only non-overlapping added; final reverse-by-start."""
    primary = [
        PIIDetection("PERSON", 0, 5, "[PERSON_REDACTED]"),
        PIIDetection("EMAIL", 40, 50, "[EMAIL_REDACTED]"),
    ]
    supplementary = [
        PIIDetection("LOCATION", 3, 8, "[LOCATION_REDACTED]"),  # overlaps PERSON -> drop
        PIIDetection("ORGANIZATION", 20, 30, "[ORGANIZATION_REDACTED]"),  # keep
        PIIDetection("ADDRESS", 45, 55, "[ADDRESS_REDACTED]"),  # overlaps EMAIL -> drop
        PIIDetection("PERSON", 60, 64, "[PERSON_REDACTED]"),  # keep
    ]
    result = merge_detections(primary, supplementary)
    assert [d.start for d in result] == [60, 40, 20, 0]
    assert [d.pii_type for d in result] == [
        "PERSON",
        "EMAIL",
        "ORGANIZATION",
        "PERSON",
    ]


# ---------------------------------------------------------------------------
# GLiNERPIIDetector.detect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "\t\n  "])
def test_detect_empty_or_whitespace_returns_empty_no_model_call(text):
    """Empty/whitespace text -> [] and the model is never invoked."""
    det = _make_detector(predictions=[{"label": "person", "start": 0, "end": 3}])
    result = det.detect(text)
    assert result == []
    assert det._model.calls == []


def test_detect_happy_path_two_entities_mapped_and_reverse_sorted():
    """Two predictions -> 2 mapped PIIDetections, reverse-sorted by start."""
    text = "John lives in Springfield town"
    preds = [
        {"label": "person", "start": 0, "end": 4, "score": 0.9},
        {"label": "location", "start": 14, "end": 25, "score": 0.8},
    ]
    det = _make_detector(predictions=preds)
    result = det.detect(text)
    assert len(result) == 2
    # reverse-sorted: location (start 14) first, then person (start 0)
    assert [d.start for d in result] == [14, 0]
    loc, person = result
    assert loc.pii_type == "LOCATION"
    assert loc.placeholder == "[LOCATION_REDACTED]"
    assert (loc.start, loc.end) == (14, 25)
    assert person.pii_type == "PERSON"
    assert person.placeholder == "[PERSON_REDACTED]"
    assert (person.start, person.end) == (0, 4)


def test_detect_unknown_label_maps_to_entity():
    """A label not in _LABEL_TO_PII_TYPE -> 'ENTITY' with [ENTITY_REDACTED]."""
    text = "some unknown thing here"
    preds = [{"label": "gibberish", "start": 5, "end": 12, "score": 0.7}]
    det = _make_detector(predictions=preds)
    result = det.detect(text)
    assert len(result) == 1
    assert result[0].pii_type == "ENTITY"
    assert result[0].placeholder == "[ENTITY_REDACTED]"


def test_detect_skips_start_ge_end_and_out_of_bounds():
    """start>=end, end>len(text), and start<0 predictions are all skipped."""
    text = "0123456789"  # len 10
    preds = [
        {"label": "person", "start": 5, "end": 5},  # start == end -> skip
        {"label": "person", "start": 6, "end": 4},  # start > end -> skip
        {"label": "person", "start": 8, "end": 12},  # end > len(text) -> skip
        {"label": "person", "start": -1, "end": 3},  # start < 0 -> skip
        {"label": "location", "start": 2, "end": 6},  # valid -> kept
    ]
    det = _make_detector(predictions=preds)
    result = det.detect(text)
    assert len(result) == 1
    assert result[0].pii_type == "LOCATION"
    assert (result[0].start, result[0].end) == (2, 6)


def test_detect_model_exception_returns_empty():
    """A model exception is swallowed -> []."""
    det = _make_detector(raise_exc=RuntimeError("boom"))
    result = det.detect("John lives somewhere")
    assert result == []
    # The model was actually called (exception path exercised).
    assert len(det._model.calls) == 1


def test_detect_forwards_text_labels_and_threshold():
    """detect forwards text, self._labels, and threshold=self._threshold."""
    text = "Acme Corp is in Berlin"
    preds = [{"label": "organization", "start": 0, "end": 9}]
    det = _make_detector(predictions=preds, threshold=0.37)
    det.detect(text)
    assert len(det._model.calls) == 1
    call = det._model.calls[0]
    assert call["text"] == text
    assert call["labels"] == list(gliner_pii._PII_LABELS)
    assert call["labels"] is not det._labels or call["labels"] == det._labels
    assert call["threshold"] == 0.37


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_pii_labels_exact_contents():
    """_PII_LABELS has the exact expected ordered contents."""
    assert gliner_pii._PII_LABELS == ["person", "organization", "location", "address"]


def test_label_to_pii_type_exact_mapping():
    """_LABEL_TO_PII_TYPE maps each lowercase label to its uppercase PII type."""
    assert gliner_pii._LABEL_TO_PII_TYPE == {
        "person": "PERSON",
        "organization": "ORGANIZATION",
        "location": "LOCATION",
        "address": "ADDRESS",
    }


# ---------------------------------------------------------------------------
# __init__ import gating
# ---------------------------------------------------------------------------


@pytest.mark.skipif(HAS_GLINER, reason="gliner is installed in this venv")
def test_init_raises_import_error_when_gliner_missing():
    """Without the gliner package, constructing the detector raises ImportError."""
    with pytest.raises(ImportError):
        GLiNERPIIDetector()


@pytest.mark.skipif(not HAS_GLINER, reason="gliner not installed in this venv")
def test_init_wires_model_threshold_and_labels(monkeypatch):
    """With gliner present, __init__ wires _model/_threshold/_labels correctly."""
    import gliner as gliner_mod

    captured: dict = {}

    def fake_from_pretrained(path, local_files_only):
        captured["path"] = path
        captured["local_files_only"] = local_files_only
        return FakeModel()

    monkeypatch.setattr(gliner_mod.GLiNER, "from_pretrained", staticmethod(fake_from_pretrained))

    # Explicit model_path override path.
    det = GLiNERPIIDetector(model_path="/tmp/custom-model", threshold=0.42)
    assert captured["path"] == "/tmp/custom-model"
    assert captured["local_files_only"] is True
    assert isinstance(det._model, FakeModel)
    assert det._threshold == 0.42
    assert det._labels == list(gliner_pii._PII_LABELS)

    # Default model_path path falls back to GLINER_MODEL_PATH.
    import config.settings as settings_mod
    det2 = GLiNERPIIDetector()
    assert captured["path"] == settings_mod.GLINER_MODEL_PATH
    assert det2._threshold == 0.5
