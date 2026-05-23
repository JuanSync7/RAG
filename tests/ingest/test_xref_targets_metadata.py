# @summary
# Tests that ingestion stamps cross-reference targets onto chunk metadata
# (``xref_targets``) as a JSON-encoded list. Both the prose chunk loop and
# the table-chunk emission must populate the field so retrieval-time
# expansion can walk the edges.
# Exports: (pytest test functions)
# Deps: pytest, json, src.ingest.common.shared
# @end-summary
"""Tests for the ingest-side ``xref_targets`` metadata edge storage."""
from __future__ import annotations

import json

from src.ingest.common.shared import cross_refs


def _encode_xrefs(text: str) -> str:
    """Helper mirroring the ingest-side encoding (kept here so tests are
    pinned to the canonical encoding even if the call site moves)."""
    return json.dumps(cross_refs(text))


def test_cross_refs_section_and_table_are_encoded_together():
    """Text with both a §-section and a Table ref must produce both entries."""
    payload = json.loads(_encode_xrefs("see §3.1 and Table 5-2"))
    types = {(r["type"], r["value"]) for r in payload}
    assert ("section_symbol", "§3.1") in types
    assert any(t == "table" for t, _ in types)


def test_empty_text_is_encoded_as_empty_list():
    """Empty/refless text round-trips to ``[]`` (not ``null``)."""
    assert _encode_xrefs("") == "[]"
    assert _encode_xrefs("just plain words with no refs") == "[]"


def test_encoded_payload_is_json_decodable_list_of_dicts():
    """The on-disk payload must be JSON-decodable to a list[dict[str,str]]."""
    raw = _encode_xrefs("Refer to Section 4 and Appendix B.2.")
    decoded = json.loads(raw)
    assert isinstance(decoded, list)
    for entry in decoded:
        assert set(entry.keys()) == {"type", "value"}


def test_prose_chunk_metadata_has_xref_targets_after_chunking():
    """Smoke: invoke the ingest-side stamping function on a fake chunk dict
    and confirm xref_targets is populated.

    We reach into the small helper that ingest uses to populate metadata,
    not the full DoclingParser pipeline (which requires real docs).
    """
    from src.ingest.support.docling import _stamp_xref_targets

    meta: dict = {}
    _stamp_xref_targets(meta, "see §3.1 and Figure 7-1")
    assert "xref_targets" in meta
    decoded = json.loads(meta["xref_targets"])
    assert any(r["type"] == "section_symbol" for r in decoded)
    assert any(r["type"] == "figure" for r in decoded)


def test_xref_stamp_on_empty_text_writes_empty_list():
    from src.ingest.support.docling import _stamp_xref_targets

    meta: dict = {}
    _stamp_xref_targets(meta, "")
    assert meta["xref_targets"] == "[]"
