# @summary
# Tests for one-hop chunk-to-chunk xref expansion at retrieval time. Covers
# section/section_symbol resolution against a mocked Weaviate store, budget
# caps, the disabled-flag short-circuit, and the TODO-scope behaviour for
# table/figure/appendix refs (no resolver yet — must pass through).
# Exports: (pytest test functions)
# Deps: pytest, unittest.mock, src.retrieval.xref_expansion
# @end-summary
"""Tests for ``expand_xref_hits`` (MVP scope: section/section_symbol)."""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from src.retrieval.xref_expansion import (
    _section_value_matches_path,
    expand_xref_hits,
    _fetch_table_chunks,
)
from src.ingest.support.docling import _extract_caption_label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hit(
    *,
    text: str = "",
    chunk_id: str = "",
    section_path: str = "",
    xref_targets: list[dict] | None = None,
) -> dict:
    return {
        "text": text,
        "score": 1.0,
        "uuid": chunk_id or f"uuid-{text}",
        "metadata": {
            "chunk_id": chunk_id or f"uuid-{text}",
            "chunk_type": "text",
            "section_path": section_path,
            "xref_targets": json.dumps(xref_targets or []),
        },
    }


def _wv_obj(props: dict, uuid: str) -> Any:
    obj = MagicMock()
    obj.properties = props
    obj.uuid = uuid
    return obj


def _make_client(by_section: dict[str, list[Any]]):
    """Mock client. The xref helper builds filters via a section-path
    contains-substring; we route by the ``_section_value`` attr that the
    patched filter builder stashes on the filter object."""
    client = MagicMock()
    col = MagicMock()
    client.collections.get.return_value = col
    client.collections.exists.return_value = True

    def _fetch(*, filters=None, limit=None, **_kw):
        sec = getattr(filters, "_section_value", None)
        response = MagicMock()
        response.objects = list(by_section.get(sec, []))[: (limit or 100)]
        return response

    col.query.fetch_objects.side_effect = _fetch
    return client


def _patch_filter(monkeypatch):
    from src.retrieval import xref_expansion as mod

    monkeypatch.setattr(
        mod, "_filter_for_section",
        lambda v: MagicMock(_section_value=v),
    )


# ---------------------------------------------------------------------------
# R1: section_symbol ref resolves to a chunk whose section_path matches.
# ---------------------------------------------------------------------------


def test_r1_section_symbol_ref_resolves(monkeypatch):
    target = _wv_obj(
        {
            "text": "the §3.1 chunk body",
            "chunk_type": "text",
            "chunk_id": "c-3.1",
            "section_path": "Doc > 3.1 Subsection",
            "section": "3.1",
        },
        uuid="c-3.1",
    )
    client = _make_client({"3.1": [target]})
    _patch_filter(monkeypatch)

    hits = [
        _hit(
            text="see §3.1",
            chunk_id="src",
            section_path="Doc > Overview",
            xref_targets=[{"type": "section_symbol", "value": "§3.1"}],
        )
    ]
    out = expand_xref_hits(
        hits, client=client, collection="C", enabled=True,
    )

    assert len(out) == 2
    inserted = out[1]
    assert inserted["metadata"]["chunk_id"] == "c-3.1"
    assert inserted["metadata"]["expanded_from"] == "xref:section_symbol:§3.1"


# ---------------------------------------------------------------------------
# R2: table refs require a source-hit document_id to scope against.
# Without one, resolver passes through (we cannot disambiguate "Table 5-2"
# across datasheets). figure/appendix remain unconditionally pass-through.
# ---------------------------------------------------------------------------


def test_r2_table_ref_without_source_document_id_passes_through(monkeypatch):
    client = MagicMock()
    client.collections.exists.return_value = True
    col = MagicMock()
    client.collections.get.return_value = col
    _patch_filter(monkeypatch)

    # Source hit has NO document_id → table ref cannot be safely scoped.
    hits = [
        _hit(
            text="see Table 5-2",
            chunk_id="src",
            xref_targets=[{"type": "table", "value": "Table 5-2"}],
        )
    ]
    out = expand_xref_hits(hits, client=client, collection="C", enabled=True)
    # No expansion — only the original hit.
    assert len(out) == 1
    # Resolver MUST NOT hit the store when scoping is unavailable.
    col.query.fetch_objects.assert_not_called()


def test_r2b_figure_and_appendix_refs_still_passthrough(monkeypatch):
    client = MagicMock()
    client.collections.exists.return_value = True
    col = MagicMock()
    client.collections.get.return_value = col
    _patch_filter(monkeypatch)

    hits = [
        _hit(
            text="see Figure 3-1 and Appendix A",
            chunk_id="src",
            xref_targets=[
                {"type": "figure", "value": "Figure 3-1"},
                {"type": "appendix", "value": "Appendix A"},
            ],
        )
    ]
    out = expand_xref_hits(hits, client=client, collection="C", enabled=True)
    assert len(out) == 1
    col.query.fetch_objects.assert_not_called()


# ---------------------------------------------------------------------------
# R3: budget cap on total expansions across hits.
# ---------------------------------------------------------------------------


def test_r3_budget_cap_max_total(monkeypatch):
    # 5 hits, each with 3 section refs. Each ref resolves to a unique chunk.
    by_section: dict[str, list[Any]] = {}
    for i in range(5):
        for j in range(3):
            sec = f"{i}.{j}"
            by_section[sec] = [
                _wv_obj(
                    {
                        "text": f"body-{sec}",
                        "chunk_type": "text",
                        "chunk_id": f"c-{sec}",
                        "section_path": f"Doc > {sec} Sub",
                        "section": sec,
                    },
                    uuid=f"c-{sec}",
                )
            ]
    client = _make_client(by_section)
    _patch_filter(monkeypatch)

    hits = [
        _hit(
            text=f"hit-{i}",
            chunk_id=f"src-{i}",
            xref_targets=[
                {"type": "section_symbol", "value": f"§{i}.{j}"}
                for j in range(3)
            ],
        )
    ]
    # Stretch the hits list:
    hits = [
        _hit(
            text=f"hit-{i}",
            chunk_id=f"src-{i}",
            xref_targets=[
                {"type": "section_symbol", "value": f"§{i}.{j}"}
                for j in range(3)
            ],
        )
        for i in range(5)
    ]
    out = expand_xref_hits(
        hits, client=client, collection="C",
        enabled=True, max_per_hit=3, max_total=4,
    )
    expanded = [
        h for h in out
        if h.get("metadata", {}).get("expanded_from", "").startswith("xref:")
    ]
    assert len(expanded) == 4


# ---------------------------------------------------------------------------
# R4: per-hit cap.
# ---------------------------------------------------------------------------


def test_r4_max_per_hit(monkeypatch):
    by_section: dict[str, list[Any]] = {}
    for j in range(5):
        sec = f"4.{j}"
        by_section[sec] = [
            _wv_obj(
                {
                    "text": f"b-{sec}",
                    "chunk_type": "text",
                    "chunk_id": f"c-{sec}",
                    "section_path": f"Doc > {sec}",
                    "section": sec,
                },
                uuid=f"c-{sec}",
            )
        ]
    client = _make_client(by_section)
    _patch_filter(monkeypatch)

    hits = [
        _hit(
            text="src",
            chunk_id="src",
            xref_targets=[
                {"type": "section_symbol", "value": f"§4.{j}"} for j in range(5)
            ],
        )
    ]
    out = expand_xref_hits(
        hits, client=client, collection="C",
        enabled=True, max_per_hit=2, max_total=100,
    )
    expanded = [
        h for h in out
        if h.get("metadata", {}).get("expanded_from", "").startswith("xref:")
    ]
    assert len(expanded) == 2


# ---------------------------------------------------------------------------
# R5: disabled flag short-circuits — returns input unchanged.
# ---------------------------------------------------------------------------


def test_r5_disabled_flag_passthrough(monkeypatch):
    client = MagicMock()
    col = MagicMock()
    client.collections.get.return_value = col
    _patch_filter(monkeypatch)

    hits = [
        _hit(
            text="see §3.1",
            chunk_id="src",
            xref_targets=[{"type": "section_symbol", "value": "§3.1"}],
        )
    ]
    out = expand_xref_hits(
        hits, client=client, collection="C", enabled=False,
    )
    assert out == hits
    col.query.fetch_objects.assert_not_called()


# ---------------------------------------------------------------------------
# R6: dedup — refs that resolve to chunks already in the hit list are skipped.
# ---------------------------------------------------------------------------


def test_r6_dedup_against_existing(monkeypatch):
    target = _wv_obj(
        {
            "text": "the §3.1 body",
            "chunk_type": "text",
            "chunk_id": "c-3.1",
            "section_path": "Doc > 3.1",
        },
        uuid="c-3.1",
    )
    client = _make_client({"3.1": [target]})
    _patch_filter(monkeypatch)

    hits = [
        _hit(
            text="see §3.1",
            chunk_id="src",
            xref_targets=[{"type": "section_symbol", "value": "§3.1"}],
        ),
        _hit(text="already here", chunk_id="c-3.1"),
    ]
    out = expand_xref_hits(hits, client=client, collection="C", enabled=True)
    # No new chunk should be inserted — dedup hit.
    assert len(out) == 2


# ---------------------------------------------------------------------------
# R7: malformed xref_targets payload is tolerated (no raise).
# ---------------------------------------------------------------------------


class TestSectionBoundaryPostFilter:
    """Boundary-aware regex post-filter so ``"3.1"`` does not glob-match
    ``"3.10"`` / ``"13.1"`` / ``"3.11"`` while still matching prefix-style
    breadcrumbs (``"3.1.4 ..."``)."""

    def test_kept_exact_token_in_breadcrumb(self):
        assert _section_value_matches_path("3.1", "Chapter 3 > 3.1 Overview")

    def test_kept_prefix_match_dot_bounded(self):
        # "3.1" is a valid prefix of "3.1.4" — the trailing "." is allowed.
        assert _section_value_matches_path("3.1", "3.1.4 Subsection")

    def test_filtered_longer_number_3_10(self):
        assert not _section_value_matches_path("3.1", "3.10 Other")

    def test_filtered_left_extension_13_1(self):
        assert not _section_value_matches_path("3.1", "13.1 Foo")

    def test_filtered_right_extension_3_11_in_breadcrumb(self):
        assert not _section_value_matches_path(
            "3.1", "App > 3.11 > Detail"
        )

    def test_kept_value_3_is_prefix_of_3_1(self):
        # "3" bounded by "." on the right — valid section prefix match.
        assert _section_value_matches_path("3", "3.1 Foo")

    def test_filtered_value_3_vs_33(self):
        assert not _section_value_matches_path("3", "33 Foo")

    def test_kept_full_value_3_1_4(self):
        assert _section_value_matches_path("3.1.4", "App > 3.1.4 Leaf")


def test_r8_e2e_boundary_filters_false_match(monkeypatch):
    """End-to-end: a ref value of ``3.1`` must NOT pull a chunk whose
    section_path only contains the longer number ``3.10`` even though the
    coarse Weaviate ``like`` prefilter returned both."""
    true_match = _wv_obj(
        {
            "text": "real 3.1 body",
            "chunk_type": "text",
            "chunk_id": "c-3.1",
            "section_path": "Doc > 3.1 Overview",
        },
        uuid="c-3.1",
    )
    false_match = _wv_obj(
        {
            "text": "decoy 3.10 body",
            "chunk_type": "text",
            "chunk_id": "c-3.10",
            "section_path": "Doc > 3.10 Other",
        },
        uuid="c-3.10",
    )
    # The mock client returns BOTH for the substring filter on "3.1" —
    # the Python post-filter must drop the 3.10 decoy.
    client = _make_client({"3.1": [false_match, true_match]})
    _patch_filter(monkeypatch)

    hits = [
        _hit(
            text="see §3.1",
            chunk_id="src",
            section_path="Doc > Overview",
            xref_targets=[{"type": "section_symbol", "value": "§3.1"}],
        )
    ]
    out = expand_xref_hits(
        hits, client=client, collection="C", enabled=True,
    )
    expanded_ids = [
        h["metadata"]["chunk_id"]
        for h in out
        if h.get("metadata", {}).get("expanded_from", "").startswith("xref:")
    ]
    assert expanded_ids == ["c-3.1"]


# ---------------------------------------------------------------------------
# Caption label extraction unit tests.
# ---------------------------------------------------------------------------


class TestExtractCaptionLabel:
    def test_label_with_dash(self):
        assert _extract_caption_label("Table 5-2: GPIO matrix") == "Table 5-2"

    def test_label_bare_number(self):
        assert _extract_caption_label("Table 12") == "Table 12"

    def test_label_with_dot(self):
        assert _extract_caption_label("Table 3.1 — power modes") == "Table 3.1"

    def test_no_prefix_returns_empty(self):
        assert _extract_caption_label("GPIO matrix") == ""

    def test_empty_returns_empty(self):
        assert _extract_caption_label("") == ""

    def test_leading_whitespace_and_extra_spaces(self):
        # Multi-space + trailing punctuation should still produce the
        # canonical label form.
        assert _extract_caption_label("  Table  5-2  : Foo") == "Table 5-2"

    def test_lowercase_keyword_titlecased(self):
        assert _extract_caption_label("table 7-4: clocks") == "Table 7-4"


# ---------------------------------------------------------------------------
# _fetch_table_chunks — Weaviate mock pattern.
# ---------------------------------------------------------------------------


def _wv_table_obj(*, label: str, doc_id: str, chunk_id: str, text: str = "body") -> Any:
    return _wv_obj(
        {
            "text": text,
            "chunk_type": "table_summary",
            "chunk_id": chunk_id,
            "caption_label": label,
            "document_id": doc_id,
            "section_path": "",
        },
        uuid=chunk_id,
    )


def _make_table_client(matches: list[Any]):
    """Mock client whose fetch_objects records the filter handed in and
    returns ``matches`` only when the AND-Filter mock matches the requested
    (label, document_id) tuple."""
    client = MagicMock()
    col = MagicMock()
    client.collections.get.return_value = col
    client.collections.exists.return_value = True

    def _fetch(*, filters=None, limit=None, **_kw):
        response = MagicMock()
        response.objects = list(matches)[: (limit or 100)]
        # stash the filter so the test can inspect it
        col._last_filter = filters
        return response

    col.query.fetch_objects.side_effect = _fetch
    return client, col


def _patch_table_filter(monkeypatch):
    """Replace ``_filter_for_table`` with a probe that records (label, doc_id)
    on the returned MagicMock."""
    from src.retrieval import xref_expansion as mod

    def _fake_table_filter(label: str, doc_id: str):
        f = MagicMock()
        f._table_label = label
        f._table_doc_id = doc_id
        return f

    monkeypatch.setattr(mod, "_filter_for_table", _fake_table_filter)


class TestFetchTableChunks:
    def test_returns_objects_on_match(self, monkeypatch):
        obj = _wv_table_obj(label="Table 5-2", doc_id="ds_a", chunk_id="c-t52")
        client, col = _make_table_client([obj])
        _patch_table_filter(monkeypatch)

        out = _fetch_table_chunks(
            client=client, collection="C",
            label="Table 5-2", document_id="ds_a", limit=5,
        )
        assert out == [obj]
        # The filter must carry BOTH label and document_id (AND scoping).
        assert col._last_filter._table_label == "Table 5-2"
        assert col._last_filter._table_doc_id == "ds_a"

    def test_returns_empty_on_no_match(self, monkeypatch):
        client, _ = _make_table_client([])
        _patch_table_filter(monkeypatch)
        out = _fetch_table_chunks(
            client=client, collection="C",
            label="Table 99", document_id="ds_a", limit=5,
        )
        assert out == []

    def test_swallows_exceptions(self, monkeypatch):
        client = MagicMock()
        client.collections.get.side_effect = RuntimeError("boom")
        _patch_table_filter(monkeypatch)
        out = _fetch_table_chunks(
            client=client, collection="C",
            label="Table 5-2", document_id="ds_a", limit=5,
        )
        assert out == []


# ---------------------------------------------------------------------------
# End-to-end table ref resolution + negative document scoping.
# ---------------------------------------------------------------------------


def _hit_with_doc(
    *, chunk_id: str, text: str, document_id: str, xref_targets: list[dict],
) -> dict:
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


def test_r9_table_ref_e2e_resolves_within_document(monkeypatch):
    target = _wv_table_obj(
        label="Table 5-2", doc_id="ds_a", chunk_id="c-t52",
        text="GPIO matrix table body",
    )
    client, col = _make_table_client([target])
    _patch_table_filter(monkeypatch)

    hits = [
        _hit_with_doc(
            chunk_id="src", text="see Table 5-2",
            document_id="ds_a",
            xref_targets=[{"type": "table", "value": "Table 5-2"}],
        )
    ]
    out = expand_xref_hits(hits, client=client, collection="C", enabled=True)
    assert len(out) == 2
    inserted = out[1]
    assert inserted["metadata"]["chunk_id"] == "c-t52"
    assert inserted["metadata"]["expanded_from"] == "xref:table:Table 5-2"
    # Filter must have scoped to the source doc.
    assert col._last_filter._table_label == "Table 5-2"
    assert col._last_filter._table_doc_id == "ds_a"


def test_r10_table_ref_negative_scoping_different_document(monkeypatch):
    # Source hit lives in ds_a; the table chunk we can find lives in ds_b.
    # The Filter mock simulates Weaviate's AND-equal — the stub returns
    # nothing because we don't pre-load any ds_a match.
    stranger = _wv_table_obj(
        label="Table 5-2", doc_id="ds_b", chunk_id="c-t52-other",
    )

    # Build a client that ONLY returns the stranger when the filter's
    # doc_id matches "ds_b" — i.e. simulate Weaviate honouring the AND.
    client = MagicMock()
    col = MagicMock()
    client.collections.get.return_value = col
    client.collections.exists.return_value = True

    def _fetch(*, filters=None, limit=None, **_kw):
        response = MagicMock()
        if (
            getattr(filters, "_table_label", None) == "Table 5-2"
            and getattr(filters, "_table_doc_id", None) == "ds_b"
        ):
            response.objects = [stranger]
        else:
            response.objects = []
        col._last_filter = filters
        return response

    col.query.fetch_objects.side_effect = _fetch
    _patch_table_filter(monkeypatch)

    hits = [
        _hit_with_doc(
            chunk_id="src", text="see Table 5-2",
            document_id="ds_a",
            xref_targets=[{"type": "table", "value": "Table 5-2"}],
        )
    ]
    out = expand_xref_hits(hits, client=client, collection="C", enabled=True)
    # No expansion — Filter doc_id was ds_a, not ds_b.
    assert len(out) == 1
    assert col._last_filter._table_doc_id == "ds_a"


def test_r7_malformed_payload_tolerated(monkeypatch):
    client = MagicMock()
    col = MagicMock()
    client.collections.get.return_value = col
    _patch_filter(monkeypatch)

    hits = [
        {
            "text": "x",
            "score": 1.0,
            "uuid": "u",
            "metadata": {"chunk_id": "u", "xref_targets": "not-json"},
        }
    ]
    out = expand_xref_hits(hits, client=client, collection="C", enabled=True)
    assert out == hits
