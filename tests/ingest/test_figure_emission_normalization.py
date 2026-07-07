# @summary
# Tests that ingest-side xref stamping normalises figure ref values into the
# canonical ``Figure N`` form the resolver filter expects (``Fig.`` /
# ``Fig`` → ``Figure``). Without this normalisation the round-trip would
# fail — emission would stamp "Fig. 4-1" while the resolver filters on
# "Figure 4-1".
# Exports: (pytest test functions)
# Deps: pytest, json, src.ingest.support.docling
# @end-summary
"""Figure-ref normalisation in ``_stamp_xref_targets``."""
from __future__ import annotations

import json

from src.ingest.support.docling import _stamp_xref_targets


def _stamp(text: str, monkeypatch, enabled: bool = True) -> list[dict]:
    from config import settings as _settings

    monkeypatch.setattr(_settings, "RAG_XREF_EXTRACT_FIGURE_REFS", enabled)
    meta: dict = {}
    _stamp_xref_targets(meta, text)
    return json.loads(meta["xref_targets"])


def test_fig_dot_normalised_to_figure(monkeypatch):
    refs = _stamp("see Fig. 4-1 for details", monkeypatch)
    figures = [r for r in refs if r["type"] == "figure"]
    assert figures, refs
    assert figures[0]["value"] == "Figure 4-1"


def test_fig_no_dot_normalised(monkeypatch):
    refs = _stamp("see Fig 2 below", monkeypatch)
    figures = [r for r in refs if r["type"] == "figure"]
    assert figures
    assert figures[0]["value"] == "Figure 2"


def test_figure_already_canonical(monkeypatch):
    refs = _stamp("see Figure 10.3", monkeypatch)
    figures = [r for r in refs if r["type"] == "figure"]
    assert figures
    assert figures[0]["value"] == "Figure 10.3"


def test_figure_lowercase_normalised(monkeypatch):
    refs = _stamp("see figure 7-2", monkeypatch)
    figures = [r for r in refs if r["type"] == "figure"]
    assert figures
    assert figures[0]["value"] == "Figure 7-2"


def test_figure_refs_dropped_when_flag_off(monkeypatch):
    refs = _stamp("see Fig. 4-1", monkeypatch, enabled=False)
    assert not [r for r in refs if r["type"] == "figure"]
