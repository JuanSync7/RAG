# @summary
# Tests for citation provenance: page_no + decoded page_bbox surface through
# server.routes.query._source_refs into the citation/source-ref payload.
# Exports: tests only
# Deps: pytest, server.routes.query
# @end-summary

"""Tests that table-chunk citations expose page_no + decoded page_bbox.

The Weaviate store persists ``page_bbox`` as a JSON-encoded TEXT property
(see ``src/vector_db/weaviate/store.py``). The citation surface must decode
that string back into a structured value at read-time, never crash on
malformed JSON, and gracefully omit the bbox when absent.
"""

from __future__ import annotations

from server.routes.query import _source_refs


def _hit(meta: dict) -> dict:
    return {"metadata": meta, "score": 0.5, "text": "row text"}


def test_source_refs_table_chunk_exposes_page_no_and_decoded_bbox():
    """Table chunk hit yields int page_no and dict-shaped decoded page_bbox."""
    hit = _hit(
        {
            "source": "doc.pdf",
            "chunk_type": "table",
            "page_no": 7,
            "page_bbox": '{"l":12.5,"t":34,"r":120,"b":78}',
        }
    )
    refs = _source_refs([hit])
    assert len(refs) == 1
    ref = refs[0]
    assert ref["page_no"] == 7
    assert isinstance(ref["page_no"], int)
    assert ref["page_bbox"] == {"l": 12.5, "t": 34, "r": 120, "b": 78}
    assert isinstance(ref["page_bbox"], dict)


def test_source_refs_table_chunk_with_list_bbox_decodes_to_dict():
    """When stored as JSON list [x0,y0,x1,y1], decoder maps to {l,t,r,b}."""
    hit = _hit(
        {
            "source": "doc.pdf",
            "chunk_type": "table",
            "page_no": 3,
            "page_bbox": "[10.0, 20.0, 110.0, 220.0]",
        }
    )
    refs = _source_refs([hit])
    ref = refs[0]
    assert ref["page_no"] == 3
    assert ref["page_bbox"] == {"l": 10.0, "t": 20.0, "r": 110.0, "b": 220.0}


def test_source_refs_prose_chunk_has_page_no_only():
    """Non-table chunk with page_no but no page_bbox yields None bbox."""
    hit = _hit(
        {
            "source": "doc.pdf",
            "chunk_type": "text",
            "page_no": 2,
            "page_bbox": "",
        }
    )
    refs = _source_refs([hit])
    ref = refs[0]
    assert ref["page_no"] == 2
    assert ref["page_bbox"] is None


def test_source_refs_missing_provenance_omits_bbox_cleanly():
    """Hit with no page_no/page_bbox at all yields None bbox and no page_no."""
    hit = _hit({"source": "doc.pdf", "chunk_type": "text"})
    refs = _source_refs([hit])
    ref = refs[0]
    assert ref["page_bbox"] is None
    assert "page_no" not in ref or ref["page_no"] is None


def test_source_refs_malformed_bbox_does_not_crash(caplog):
    """Malformed JSON in page_bbox must yield page_bbox=None, not raise."""
    hit = _hit(
        {
            "source": "doc.pdf",
            "chunk_type": "table",
            "page_no": 5,
            "page_bbox": "{not-json",
        }
    )
    refs = _source_refs([hit])
    ref = refs[0]
    assert ref["page_no"] == 5
    assert ref["page_bbox"] is None
