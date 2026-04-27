# @summary
# Tests for the EditLog opcode-based offset mapper between original and
# refactored text, plus integration with map_chunk_provenance.
# Exports: (pytest test functions)
# Deps: src.ingest.common.edit_log, src.ingest.common.shared
# @end-summary
"""Tests for EditLog and its integration with map_chunk_provenance."""

from __future__ import annotations

from src.ingest.common.edit_log import EditLog
from src.ingest.common.shared import map_chunk_provenance


class TestEditLogIdentity:
    def test_identical_texts_full_confidence(self):
        text = "Hello world. This is a test."
        log = EditLog.from_diff(text, text)
        s, e, conf = log.map_ref_to_orig(6, 11)
        assert (s, e) == (6, 11)
        assert conf == 1.0

    def test_empty_strings(self):
        log = EditLog.from_diff("", "")
        s, e, conf = log.map_ref_to_orig(0, 0)
        assert s == 0 and e == 0


class TestEditLogInsertion:
    def test_pure_insertion_in_middle(self):
        # Refactor inserts ", world" after "Hello".
        original = "Hello!"
        refactored = "Hello, world!"
        log = EditLog.from_diff(original, refactored)
        # "Hello" — both endpoints in equal blocks → confidence 1.0.
        s, e, conf = log.map_ref_to_orig(0, 5)
        assert original[s:e] == "Hello"
        assert conf == 1.0
        # "!" at end of refactored → maps to "!" at end of original.
        s, e, conf = log.map_ref_to_orig(12, 13)
        assert original[s:e] == "!"
        assert conf == 1.0

    def test_span_inside_inserted_text_returns_block_extent(self):
        original = "AB"
        refactored = "A-INSERTED-B"
        log = EditLog.from_diff(original, refactored)
        # Span "INSERTED" lies entirely inside the inserted block.
        ref_start = refactored.find("INSERTED")
        s, e, conf = log.map_ref_to_orig(ref_start, ref_start + len("INSERTED"))
        # Inserted text has no original counterpart; mapper returns the
        # zero-width gap between A and B with confidence < 1.0.
        assert conf < 1.0
        assert 0 <= s <= e <= len(original)


class TestEditLogReplacement:
    def test_span_inside_replacement_block(self):
        original = "The quick brown fox"
        refactored = "The slow red fox"
        log = EditLog.from_diff(original, refactored)
        # "slow red" replaces "quick brown"; chunk landing inside replacement
        # gets best-effort highlight (the replaced original block).
        ref_start = refactored.find("slow red")
        s, e, conf = log.map_ref_to_orig(ref_start, ref_start + len("slow red"))
        assert conf < 1.0
        # Should land somewhere overlapping "quick brown".
        assert original[s:e].strip() != ""

    def test_anchored_span_across_replacement(self):
        original = "The quick brown fox jumped"
        refactored = "The slow red fox jumped"
        log = EditLog.from_diff(original, refactored)
        # Span "fox jumped" — both endpoints anchored in equal blocks.
        ref_start = refactored.find("fox jumped")
        s, e, conf = log.map_ref_to_orig(ref_start, ref_start + len("fox jumped"))
        assert original[s:e] == "fox jumped"
        assert conf == 1.0


class TestMapChunkProvenanceWithEditLog:
    def test_edit_log_path_preferred_when_chunk_in_refactored(self):
        original = "The quick brown fox jumped over the lazy dog."
        refactored = "The slow red fox jumped over the tired dog."
        log = EditLog.from_diff(original, refactored)
        chunk = "fox jumped over the"
        prov, _, _ = map_chunk_provenance(
            chunk,
            original_text=original,
            refactored_text=refactored,
            original_cursor=0,
            refactored_cursor=0,
            edit_log=log,
        )
        # Chunk is found verbatim in both, so substring path also works; but
        # method should be edit_log when log is supplied and ref-side resolved.
        assert prov["provenance_method"].endswith("edit_log") or "edit_log" in prov["provenance_method"]
        assert prov["provenance_confidence"] == 1.0
        s = int(prov["original_char_start"])
        e = int(prov["original_char_end"])
        assert original[s:e] == "fox jumped over the"

    def test_edit_log_resolves_chunk_only_in_refactored(self):
        # Chunk text contains LLM-rewritten content not present verbatim in original.
        original = "Clock must remain below 800MHz to avoid overheating."
        refactored = "The clock frequency stays under 800MHz to prevent thermal issues."
        log = EditLog.from_diff(original, refactored)
        chunk = "clock frequency stays under 800MHz"
        prov, _, _ = map_chunk_provenance(
            chunk,
            original_text=original,
            refactored_text=refactored,
            original_cursor=0,
            refactored_cursor=0,
            edit_log=log,
        )
        # Without edit_log, this would fall through to fuzzy paragraph match
        # (confidence ≤ 0.79). With edit_log, we should at least have non-zero
        # original offsets pointing somewhere overlapping the original sentence.
        assert prov["original_char_start"] >= 0
        assert prov["original_char_end"] > prov["original_char_start"]

    def test_no_edit_log_preserves_old_behavior(self):
        text = "Alpha. Beta. Gamma."
        prov, _, _ = map_chunk_provenance(
            "Beta.",
            original_text=text,
            refactored_text=text,
            original_cursor=0,
            refactored_cursor=0,
        )
        # Old code path: substring exact match, confidence 1.0, method does not mention edit_log.
        assert prov["provenance_confidence"] == 1.0
        assert "edit_log" not in str(prov["provenance_method"])
