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

from src.retrieval.xref_expansion import expand_xref_hits


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
# R2: table refs in MVP scope are TODO — no expansion (passes through).
# ---------------------------------------------------------------------------


def test_r2_table_refs_are_passthrough_in_mvp(monkeypatch):
    client = MagicMock()
    client.collections.exists.return_value = True
    col = MagicMock()
    client.collections.get.return_value = col
    _patch_filter(monkeypatch)

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
    # Resolver MUST NOT hit the store for table refs.
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
