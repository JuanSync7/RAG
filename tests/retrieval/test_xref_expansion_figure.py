# @summary
# Tests for ``figure`` ref resolution in ``expand_xref_hits``. Mirrors the
# table resolver pattern: document-scoped exact match on ``caption_label``
# AND ``chunk_type == "figure"``. Covers Fig./Fig → Figure normalisation
# on the resolver side, document-scoping, and false-positive boundaries.
# Exports: (pytest test functions)
# Deps: pytest, unittest.mock, json, src.retrieval.xref_expansion
# @end-summary
"""Figure xref resolution end-to-end."""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from src.retrieval.xref_expansion import (
    _normalize_figure_label,
    expand_xref_hits,
    _fetch_figure_chunks,
)


def _wv_obj(props: dict, uuid: str) -> Any:
    obj = MagicMock()
    obj.properties = props
    obj.uuid = uuid
    return obj


def _wv_fig(*, label: str, doc_id: str, chunk_id: str, text: str = "fig body") -> Any:
    return _wv_obj(
        {
            "text": text,
            "chunk_type": "figure",
            "chunk_id": chunk_id,
            "caption_label": label,
            "document_id": doc_id,
            "section_path": "",
        },
        uuid=chunk_id,
    )


def _patch_figure_filter(monkeypatch):
    from src.retrieval import xref_expansion as mod

    def _fake(label: str, doc_id: str):
        f = MagicMock()
        f._fig_label = label
        f._fig_doc_id = doc_id
        return f

    monkeypatch.setattr(mod, "_filter_for_figure", _fake)


def _make_fig_client(matches: list[Any]):
    client = MagicMock()
    col = MagicMock()
    client.collections.get.return_value = col
    client.collections.exists.return_value = True

    def _fetch(*, filters=None, limit=None, **_kw):
        response = MagicMock()
        response.objects = list(matches)[: (limit or 100)]
        col._last_filter = filters
        return response

    col.query.fetch_objects.side_effect = _fetch
    return client, col


def _hit_with_doc(*, chunk_id: str, text: str, document_id: str,
                  xref_targets: list[dict]) -> dict:
    return {
        "text": text,
        "score": 1.0,
        "uuid": chunk_id,
        "metadata": {
            "chunk_id": chunk_id,
            "chunk_type": "text",
            "section_path": "Doc > Overview",
            "document_id": document_id,
            "xref_targets": json.dumps(xref_targets),
        },
    }


# -- _normalize_figure_label ---------------------------------------------


class TestNormalizeFigureLabel:
    def test_canonical(self):
        assert _normalize_figure_label("Figure 4-1") == "Figure 4-1"

    def test_fig_dot(self):
        assert _normalize_figure_label("Fig. 4-1") == "Figure 4-1"

    def test_fig_no_dot(self):
        assert _normalize_figure_label("Fig 2") == "Figure 2"

    def test_lowercase(self):
        assert _normalize_figure_label("figure 7-2") == "Figure 7-2"

    def test_dot_separator(self):
        assert _normalize_figure_label("Figure 10.3") == "Figure 10.3"

    def test_no_match_returns_empty(self):
        # The boundary-anchored regex MUST drop bogus prefix like "Refigure".
        assert _normalize_figure_label("Refigure 4-1") == ""

    def test_empty(self):
        assert _normalize_figure_label("") == ""


# -- _fetch_figure_chunks ------------------------------------------------


class TestFetchFigureChunks:
    def test_returns_objects_on_match(self, monkeypatch):
        obj = _wv_fig(label="Figure 4-1", doc_id="ds_a", chunk_id="c-f41")
        client, col = _make_fig_client([obj])
        _patch_figure_filter(monkeypatch)

        out = _fetch_figure_chunks(
            client=client, collection="C",
            label="Figure 4-1", document_id="ds_a", limit=5,
        )
        assert out == [obj]
        assert col._last_filter._fig_label == "Figure 4-1"
        assert col._last_filter._fig_doc_id == "ds_a"

    def test_returns_empty_on_no_match(self, monkeypatch):
        client, _ = _make_fig_client([])
        _patch_figure_filter(monkeypatch)
        out = _fetch_figure_chunks(
            client=client, collection="C",
            label="Figure 99", document_id="ds_a", limit=5,
        )
        assert out == []

    def test_swallows_exceptions(self, monkeypatch):
        client = MagicMock()
        client.collections.get.side_effect = RuntimeError("boom")
        _patch_figure_filter(monkeypatch)
        out = _fetch_figure_chunks(
            client=client, collection="C",
            label="Figure 4-1", document_id="ds_a", limit=5,
        )
        assert out == []


# -- end-to-end -----------------------------------------------------------


def test_figure_ref_resolves_within_document(monkeypatch):
    target = _wv_fig(label="Figure 4-1", doc_id="ds_a", chunk_id="c-f41")
    client, col = _make_fig_client([target])
    _patch_figure_filter(monkeypatch)

    hits = [
        _hit_with_doc(
            chunk_id="src", text="see Figure 4-1",
            document_id="ds_a",
            xref_targets=[{"type": "figure", "value": "Figure 4-1"}],
        )
    ]
    out = expand_xref_hits(hits, client=client, collection="C", enabled=True)
    assert len(out) == 2
    inserted = out[1]
    assert inserted["metadata"]["chunk_id"] == "c-f41"
    assert inserted["metadata"]["expanded_from"] == "xref:figure:Figure 4-1"


def test_figure_ref_fig_dot_normalises_then_resolves(monkeypatch):
    """Even if the stamped xref value slipped through as "Fig. 4-1", the
    resolver normalises before filtering so the round-trip still works."""
    target = _wv_fig(label="Figure 4-1", doc_id="ds_a", chunk_id="c-f41")
    client, col = _make_fig_client([target])
    _patch_figure_filter(monkeypatch)

    hits = [
        _hit_with_doc(
            chunk_id="src", text="see Fig. 4-1",
            document_id="ds_a",
            xref_targets=[{"type": "figure", "value": "Fig. 4-1"}],
        )
    ]
    out = expand_xref_hits(hits, client=client, collection="C", enabled=True)
    assert len(out) == 2
    assert col._last_filter._fig_label == "Figure 4-1"


def test_figure_ref_without_document_id_passes_through(monkeypatch):
    client = MagicMock()
    client.collections.exists.return_value = True
    col = MagicMock()
    client.collections.get.return_value = col
    _patch_figure_filter(monkeypatch)

    hits = [
        {
            "text": "see Figure 4-1",
            "score": 1.0,
            "uuid": "src",
            "metadata": {
                "chunk_id": "src",
                "xref_targets": json.dumps(
                    [{"type": "figure", "value": "Figure 4-1"}]
                ),
            },
        }
    ]
    out = expand_xref_hits(hits, client=client, collection="C", enabled=True)
    assert len(out) == 1
    col.query.fetch_objects.assert_not_called()


def test_figure_ref_boundary_refigure_is_rejected(monkeypatch):
    """``Refigure 4-1`` must NOT route to figure resolution. Achieved either
    by ingest (cross_refs regex word-boundary) or by resolver normalisation
    returning "" — either layer is acceptable as long as no fetch fires."""
    client = MagicMock()
    client.collections.exists.return_value = True
    col = MagicMock()
    client.collections.get.return_value = col
    _patch_figure_filter(monkeypatch)

    hits = [
        _hit_with_doc(
            chunk_id="src", text="Refigure 4-1 is unrelated",
            document_id="ds_a",
            xref_targets=[{"type": "figure", "value": "Refigure 4-1"}],
        )
    ]
    out = expand_xref_hits(hits, client=client, collection="C", enabled=True)
    assert len(out) == 1
    col.query.fetch_objects.assert_not_called()
