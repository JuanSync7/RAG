"""Unit tests for the pure helper functions in ``server/routes/query.py``.

These helpers are deterministic, side-effect-light building blocks used by the
query/streaming routes:

- ``_sse``: format a Server-Sent-Events frame.
- ``_derive_title_from_query``: condense a first query into a short title.
- ``_autotitle_if_new``: set a conversation title iff the conversation is new.
- ``_decode_page_bbox``: decode a stored ``page_bbox`` into ``{l,t,r,b}`` or None.
- ``_source_refs``: extract lightweight source references from RAG results.
- ``_aggregate_stage_totals``: roll stage timings up into per-bucket totals.

Each test exercises one behavior with exact-value assertions and distinct data
so the assertions have teeth against mutation.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from server.routes.query import (
    _aggregate_stage_totals,
    _autotitle_if_new,
    _decode_page_bbox,
    _derive_title_from_query,
    _source_refs,
    _sse,
)


# ---------------------------------------------------------------------------
# _sse
# ---------------------------------------------------------------------------
def test_sse_emits_exact_frame_with_blank_line_terminator():
    """``_sse`` produces ``event: <e>\\ndata: <json>\\n\\n`` with the JSON body."""
    frame = _sse("foo", {"a": 1})
    assert frame == 'event: foo\ndata: {"a":1}\n\n'


def test_sse_data_line_is_json_of_the_dict():
    """The data line carries a parseable JSON encoding of the payload dict."""
    frame = _sse("update", {"b": 2, "c": "x"})
    assert frame.startswith("event: update\ndata: ")
    assert frame.endswith("\n\n")
    data_line = frame.split("\ndata: ", 1)[1][:-2]  # strip trailing blank line
    assert json.loads(data_line) == {"b": 2, "c": "x"}


# ---------------------------------------------------------------------------
# _derive_title_from_query
# ---------------------------------------------------------------------------
def test_derive_title_empty_query_returns_empty_string():
    """An empty query yields the empty string."""
    assert _derive_title_from_query("") == ""


def test_derive_title_whitespace_only_returns_empty_string():
    """A whitespace-only query collapses to the empty string."""
    assert _derive_title_from_query("   \t\n  ") == ""


def test_derive_title_collapses_internal_whitespace():
    """Runs of internal whitespace collapse to single spaces."""
    assert _derive_title_from_query("hello   world\t\nthere") == "hello world there"


def test_derive_title_at_max_len_returned_as_is():
    """A query exactly ``max_len`` long is returned verbatim (boundary)."""
    text = "a" * 60
    assert len(text) == 60
    assert _derive_title_from_query(text) == text


def test_derive_title_over_max_len_cuts_at_word_boundary_strips_punct_appends_ellipsis():
    """An over-length query cuts at the last word, strips trailing punctuation, adds an ellipsis."""
    # The last word boundary within 60 chars keeps a token ending in ".".
    text = "alpha beta gamma delta epsilon zeta eta theta question. moreword tail extra"
    result = _derive_title_from_query(text)
    assert result == "alpha beta gamma delta epsilon zeta eta theta question…"
    assert result.endswith("…")
    assert "question." not in result


def test_derive_title_long_single_word_hard_cuts_at_max_len():
    """A single word longer than ``max_len`` is hard-cut at ``max_len`` plus an ellipsis."""
    text = "x" * 80
    result = _derive_title_from_query(text)
    assert result == "x" * 60 + "…"
    assert len(result) == 61


# ---------------------------------------------------------------------------
# _autotitle_if_new
# ---------------------------------------------------------------------------
def _make_conv(*, message_count=0, title="New conversation", conversation_id="conv-1"):
    conv = MagicMock(name="conv")
    conv.message_count = message_count
    conv.title = title
    conv.conversation_id = conversation_id
    return conv


def _make_principal():
    principal = MagicMock(name="principal")
    principal.subject = "user-42"
    principal.project_id = "proj-7"
    return principal


def test_autotitle_existing_messages_returns_conv_unchanged_no_memory_call():
    """A conversation with prior messages is never retitled."""
    memory = MagicMock()
    conv = _make_conv(message_count=3, title="New conversation")
    result = _autotitle_if_new(
        memory, tenant_id="t1", principal=_make_principal(), conv=conv, query="hi there"
    )
    assert result is conv
    memory.update_conversation_title.assert_not_called()


def test_autotitle_existing_meaningful_title_returns_conv_unchanged():
    """A new conversation that already has a real title is left alone."""
    memory = MagicMock()
    conv = _make_conv(message_count=0, title="My deliberate title")
    result = _autotitle_if_new(
        memory, tenant_id="t1", principal=_make_principal(), conv=conv, query="something"
    )
    assert result is conv
    memory.update_conversation_title.assert_not_called()


def test_autotitle_new_conversation_sets_derived_title_and_returns_update_result():
    """A brand-new placeholder-titled conversation gets the derived title persisted."""
    memory = MagicMock()
    updated_sentinel = object()
    memory.update_conversation_title.return_value = updated_sentinel
    conv = _make_conv(message_count=0, title="New conversation", conversation_id="conv-9")
    principal = _make_principal()
    result = _autotitle_if_new(
        memory, tenant_id="tenant-x", principal=principal, conv=conv, query="how do I reset?"
    )
    memory.update_conversation_title.assert_called_once_with(
        tenant_id="tenant-x",
        subject="user-42",
        project_id="proj-7",
        conversation_id="conv-9",
        title="how do I reset?",
    )
    assert result is updated_sentinel


def test_autotitle_empty_query_returns_conv_unchanged():
    """An empty derived title is a no-op even for a new conversation."""
    memory = MagicMock()
    conv = _make_conv(message_count=0, title="New conversation")
    result = _autotitle_if_new(
        memory, tenant_id="t1", principal=_make_principal(), conv=conv, query="   "
    )
    assert result is conv
    memory.update_conversation_title.assert_not_called()


def test_autotitle_falsy_update_result_falls_back_to_conv():
    """If the memory update returns falsy, the original conv is returned."""
    memory = MagicMock()
    memory.update_conversation_title.return_value = None
    conv = _make_conv(message_count=0, title="New conversation")
    result = _autotitle_if_new(
        memory, tenant_id="t1", principal=_make_principal(), conv=conv, query="real query"
    )
    assert result is conv


# ---------------------------------------------------------------------------
# _decode_page_bbox
# ---------------------------------------------------------------------------
def test_decode_bbox_none_returns_none():
    """``None`` decodes to ``None``."""
    assert _decode_page_bbox(None) is None


def test_decode_bbox_empty_string_returns_none():
    """The empty string (Weaviate's "absent") decodes to ``None``."""
    assert _decode_page_bbox("") is None


def test_decode_bbox_dict_maps_to_float_ltrb():
    """A dict with l/t/r/b keys is float-coerced into the canonical shape."""
    assert _decode_page_bbox({"l": 1, "t": 2, "r": 3, "b": 4}) == {
        "l": 1.0,
        "t": 2.0,
        "r": 3.0,
        "b": 4.0,
    }


def test_decode_bbox_dict_missing_key_returns_none():
    """A dict missing one of the required keys decodes to ``None``."""
    assert _decode_page_bbox({"l": 1, "t": 2, "r": 3}) is None


def test_decode_bbox_list_of_four_maps_to_ltrb():
    """A 4-element list maps positionally to l/t/r/b floats."""
    assert _decode_page_bbox([1, 2, 3, 4]) == {"l": 1.0, "t": 2.0, "r": 3.0, "b": 4.0}


def test_decode_bbox_list_wrong_length_returns_none():
    """A list with a length other than 4 decodes to ``None``."""
    assert _decode_page_bbox([1, 2, 3]) is None


def test_decode_bbox_list_non_numeric_returns_none():
    """A 4-element list with a non-numeric entry decodes to ``None``."""
    assert _decode_page_bbox([1, 2, "nope", 4]) is None


def test_decode_bbox_json_string_list_decodes_to_dict():
    """A JSON-string list is parsed and mapped to the canonical dict."""
    assert _decode_page_bbox("[1, 2, 3, 4]") == {"l": 1.0, "t": 2.0, "r": 3.0, "b": 4.0}


def test_decode_bbox_json_string_object_decodes_to_dict():
    """A JSON-string object is parsed and float-coerced."""
    assert _decode_page_bbox('{"l": 5, "t": 6, "r": 7, "b": 8}') == {
        "l": 5.0,
        "t": 6.0,
        "r": 7.0,
        "b": 8.0,
    }


def test_decode_bbox_malformed_json_returns_none():
    """Malformed JSON is swallowed and decodes to ``None`` (never crashes)."""
    assert _decode_page_bbox("{bad") is None


def test_decode_bbox_non_str_non_collection_returns_none():
    """A bare int (not str/list/dict) decodes to ``None``."""
    assert _decode_page_bbox(42) is None


# ---------------------------------------------------------------------------
# _source_refs
# ---------------------------------------------------------------------------
def test_source_refs_dict_shaped_result_builds_ref():
    """A dict-shaped result populates the full ref envelope."""
    result = {
        "metadata": {
            "source": "doc.pdf",
            "source_uri": "file://doc.pdf",
            "source_key": "k1",
            "document_id": "d1",
            "section": "Intro",
        },
        "score": 0.75,
        "text": "hello",
    }
    refs = _source_refs([result])
    assert refs == [
        {
            "source": "doc.pdf",
            "source_uri": "file://doc.pdf",
            "source_key": "k1",
            "document_id": "d1",
            "section": "Intro",
            "score": 0.75,
            "text": "hello",
            "page_bbox": None,
        }
    ]


def test_source_refs_object_shaped_result_uses_attributes():
    """An object-shaped result is read via attribute (duck-typed) access."""
    obj = MagicMock()
    obj.metadata = {"source": "obj.pdf"}
    obj.score = 0.5
    obj.text = "body"
    # Force isinstance(r, dict) to be False by using a non-dict object.
    refs = _source_refs([obj])
    assert refs[0]["source"] == "obj.pdf"
    assert refs[0]["score"] == 0.5
    assert refs[0]["text"] == "body"


def test_source_refs_text_truncated_to_400_chars():
    """The ref text is truncated to at most 400 characters."""
    long_text = "z" * 500
    refs = _source_refs([{"metadata": {}, "score": 0.0, "text": long_text}])
    assert len(refs[0]["text"]) == 400
    assert refs[0]["text"] == "z" * 400


def test_source_refs_char_start_zero_is_included():
    """``original_char_start`` of 0 IS emitted (0 is not None — not falsy-dropped)."""
    refs = _source_refs(
        [{"metadata": {"original_char_start": 0, "original_char_end": 12}, "score": 0.0, "text": "t"}]
    )
    assert refs[0]["original_char_start"] == 0
    assert refs[0]["original_char_end"] == 12


def test_source_refs_char_offsets_absent_when_none():
    """When the offsets are missing the keys are not present at all."""
    refs = _source_refs([{"metadata": {}, "score": 0.0, "text": "t"}])
    assert "original_char_start" not in refs[0]
    assert "original_char_end" not in refs[0]


def test_source_refs_page_no_string_coerced_to_int():
    """A string ``page_no`` is coerced to an int."""
    refs = _source_refs([{"metadata": {"page_no": "3"}, "score": 0.0, "text": "t"}])
    assert refs[0]["page_no"] == 3


def test_source_refs_page_no_non_numeric_key_absent():
    """A non-numeric ``page_no`` is dropped rather than crashing."""
    refs = _source_refs([{"metadata": {"page_no": "x"}, "score": 0.0, "text": "t"}])
    assert "page_no" not in refs[0]


def test_source_refs_section_falls_back_to_heading():
    """When ``section`` is missing the ``heading`` value is used instead."""
    refs = _source_refs([{"metadata": {"heading": "Chapter 2"}, "score": 0.0, "text": "t"}])
    assert refs[0]["section"] == "Chapter 2"


def test_source_refs_page_bbox_decoded_via_decoder():
    """``page_bbox`` is run through ``_decode_page_bbox``."""
    refs = _source_refs(
        [{"metadata": {"page_bbox": "[1, 2, 3, 4]"}, "score": 0.0, "text": "t"}]
    )
    assert refs[0]["page_bbox"] == {"l": 1.0, "t": 2.0, "r": 3.0, "b": 4.0}


# ---------------------------------------------------------------------------
# _aggregate_stage_totals
# ---------------------------------------------------------------------------
def test_aggregate_same_bucket_sums_not_overwrites():
    """Two stages in the same bucket have their ms summed."""
    totals = _aggregate_stage_totals(
        [{"bucket": "retrieve", "ms": 10.0}, {"bucket": "retrieve", "ms": 5.0}]
    )
    assert totals["retrieve_ms"] == 15.0
    assert totals["total_ms"] == 15.0


def test_aggregate_distinct_buckets_emit_separate_keys():
    """Distinct buckets each get their own ``<bucket>_ms`` key plus a total."""
    totals = _aggregate_stage_totals(
        [{"bucket": "retrieve", "ms": 10.0}, {"bucket": "generate", "ms": 4.0}]
    )
    assert totals == {"retrieve_ms": 10.0, "generate_ms": 4.0, "total_ms": 14.0}


def test_aggregate_missing_bucket_defaults_to_other():
    """A stage without a ``bucket`` is attributed to ``other``."""
    totals = _aggregate_stage_totals([{"ms": 7.0}])
    assert totals["other_ms"] == 7.0
    assert totals["total_ms"] == 7.0


def test_aggregate_missing_ms_defaults_to_zero():
    """A stage without ``ms`` contributes 0.0."""
    totals = _aggregate_stage_totals([{"bucket": "embed"}])
    assert totals["embed_ms"] == 0.0
    assert totals["total_ms"] == 0.0


def test_aggregate_rounds_to_one_decimal():
    """Per-bucket and total values are rounded to one decimal place."""
    totals = _aggregate_stage_totals([{"bucket": "retrieve", "ms": 1.246}])
    assert totals["retrieve_ms"] == 1.2
    assert totals["total_ms"] == 1.2
