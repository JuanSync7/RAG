# @summary
# Tests that ``RAG_XREF_EXTRACT_FIGURE_REFS`` cleanly gates the figure-ref
# round-trip: when False (default), no figure refs reach the resolver; when
# True, an end-to-end stamp → resolver call succeeds.
# Exports: (pytest test functions)
# Deps: pytest, unittest.mock, json, src.ingest.support.docling,
#       src.retrieval.xref_expansion
# @end-summary
"""End-to-end gating of figure xref refs."""
from __future__ import annotations

import json
from unittest.mock import MagicMock
from typing import Any

from src.ingest.support.docling import _stamp_xref_targets
from src.retrieval.xref_expansion import expand_xref_hits


def _wv_obj(props: dict, uuid: str) -> Any:
    obj = MagicMock()
    obj.properties = props
    obj.uuid = uuid
    return obj


def _patch_figure_filter(monkeypatch):
    from src.retrieval import xref_expansion as mod

    def _fake(label: str, doc_id: str):
        f = MagicMock()
        f._fig_label = label
        f._fig_doc_id = doc_id
        return f

    monkeypatch.setattr(mod, "_filter_for_figure", _fake)


def test_flag_off_no_figure_refs_stamped(monkeypatch):
    from config import settings as _settings

    monkeypatch.setattr(_settings, "RAG_XREF_EXTRACT_FIGURE_REFS", False)
    meta: dict = {}
    _stamp_xref_targets(meta, "see Figure 4-1")
    decoded = json.loads(meta["xref_targets"])
    assert not [r for r in decoded if r["type"] == "figure"]


def test_flag_on_round_trip(monkeypatch):
    from config import settings as _settings

    monkeypatch.setattr(_settings, "RAG_XREF_EXTRACT_FIGURE_REFS", True)

    # 1) ingest stamps the canonical form
    meta: dict = {"document_id": "ds_a"}
    _stamp_xref_targets(meta, "see Fig. 4-1")
    decoded = json.loads(meta["xref_targets"])
    fig_refs = [r for r in decoded if r["type"] == "figure"]
    assert fig_refs
    assert fig_refs[0]["value"] == "Figure 4-1"

    # 2) resolver picks up the stamped value, normalises, fetches
    target = _wv_obj(
        {
            "text": "fig body",
            "chunk_type": "figure",
            "chunk_id": "c-f41",
            "caption_label": "Figure 4-1",
            "document_id": "ds_a",
        },
        uuid="c-f41",
    )
    client = MagicMock()
    col = MagicMock()
    client.collections.get.return_value = col
    client.collections.exists.return_value = True

    def _fetch(*, filters=None, limit=None, **_kw):
        resp = MagicMock()
        resp.objects = [target]
        col._last_filter = filters
        return resp

    col.query.fetch_objects.side_effect = _fetch
    _patch_figure_filter(monkeypatch)

    hits = [
        {
            "text": "see Fig. 4-1",
            "score": 1.0,
            "uuid": "src",
            "metadata": {
                "chunk_id": "src",
                "document_id": "ds_a",
                "xref_targets": meta["xref_targets"],
            },
        }
    ]
    out = expand_xref_hits(hits, client=client, collection="C", enabled=True)
    assert len(out) == 2
    assert out[1]["metadata"]["chunk_id"] == "c-f41"
