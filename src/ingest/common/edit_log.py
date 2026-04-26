# @summary
# Edit log for mapping refactored-text offsets back to original-text offsets.
# Built post-hoc via difflib.SequenceMatcher; the LLM refactor step is opaque,
# so per-transform tracking is not possible — we diff the endpoints instead.
# Exports: EditLog
# Deps: difflib
# @end-summary

"""Bidirectional offset mapper between original and refactored text.

The ingest pipeline rewrites ``raw_text`` into ``refactored_text`` via a
mechanical-clean stage followed by an LLM rewrite. The LLM is a black box, so
we recover the original<->refactored coordinate mapping by diffing the two
strings with :class:`difflib.SequenceMatcher` and indexing the resulting
opcodes.

A chunk's offsets in refactored space can then be projected back to original
space exactly when the chunk lies within an ``equal`` block, and approximately
(by interpolation across edits) when it straddles edits.
"""

from __future__ import annotations

import difflib
from bisect import bisect_right
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class _Op:
    """One opcode block from SequenceMatcher.

    ``tag`` is one of ``equal``, ``replace``, ``insert``, ``delete``.
    Offsets are half-open ranges in original/refactored text.
    """

    tag: str
    orig_start: int
    orig_end: int
    ref_start: int
    ref_end: int


class EditLog:
    """Maps spans between original and refactored text via diff opcodes.

    Build once per document with :meth:`from_diff`, then reuse for every chunk
    in that document. ``map_ref_to_orig`` returns ``(orig_start, orig_end,
    confidence)`` where confidence is ``1.0`` for spans inside ``equal`` blocks
    and lower when the span crosses edit boundaries.
    """

    def __init__(self, ops: List[_Op]) -> None:
        self._ops = ops
        # Sorted ref_start values for bisect lookup.
        self._ref_starts = [op.ref_start for op in ops]

    @classmethod
    def from_diff(cls, original: str, refactored: str) -> "EditLog":
        """Build an edit log by diffing two strings.

        Uses :class:`difflib.SequenceMatcher` with ``autojunk=False`` so junk
        heuristics do not skip frequent characters in long documents.
        """
        if original == refactored:
            # Fast path: identity mapping.
            n = len(original)
            return cls([_Op("equal", 0, n, 0, n)])
        sm = difflib.SequenceMatcher(a=original, b=refactored, autojunk=False)
        ops: List[_Op] = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            ops.append(_Op(tag, i1, i2, j1, j2))
        return cls(ops)

    def _find_op(self, ref_pos: int) -> Optional[_Op]:
        """Return the opcode block containing ``ref_pos`` (or None if past end)."""
        if not self._ops:
            return None
        # bisect_right gives the first op whose ref_start > ref_pos; the
        # containing op is the one before it.
        idx = bisect_right(self._ref_starts, ref_pos) - 1
        if idx < 0:
            return self._ops[0]
        op = self._ops[idx]
        if ref_pos >= op.ref_end and idx + 1 < len(self._ops):
            return self._ops[idx + 1]
        return op

    def map_ref_to_orig(
        self,
        ref_start: int,
        ref_end: int,
    ) -> Tuple[int, int, float]:
        """Project a refactored-text span back to original-text coordinates.

        Returns ``(orig_start, orig_end, confidence)``:
            * confidence ``1.0`` — span lies entirely inside an ``equal`` block;
              offsets are exact.
            * confidence ``0.9`` — span lies inside a single ``replace``/``insert``/
              ``delete`` block; we return the corresponding original block extents
              (best-effort highlight).
            * confidence ``0.75`` — span straddles multiple blocks; we return the
              union of original extents for those blocks.
        """
        if not self._ops or ref_end < ref_start:
            return -1, -1, 0.0
        if ref_start == ref_end:
            op = self._find_op(ref_start)
            if op is None:
                return -1, -1, 0.0
            offset = ref_start - op.ref_start if op.tag == "equal" else 0
            pos = op.orig_start + offset if op.tag == "equal" else op.orig_start
            return pos, pos, 1.0 if op.tag == "equal" else 0.9

        start_op = self._find_op(ref_start)
        # For end, use ref_end - 1 to find the op containing the last char.
        end_op = self._find_op(max(ref_end - 1, ref_start))
        if start_op is None or end_op is None:
            return -1, -1, 0.0

        if start_op is end_op:
            if start_op.tag == "equal":
                orig_s = start_op.orig_start + (ref_start - start_op.ref_start)
                orig_e = start_op.orig_start + (ref_end - start_op.ref_start)
                return orig_s, orig_e, 1.0
            # Whole span inside one non-equal block: return that block's original extent.
            return start_op.orig_start, start_op.orig_end, 0.9

        # Span crosses block boundaries. Compute exact endpoints when possible.
        if start_op.tag == "equal":
            orig_s = start_op.orig_start + (ref_start - start_op.ref_start)
        else:
            orig_s = start_op.orig_start
        if end_op.tag == "equal":
            orig_e = end_op.orig_start + (ref_end - end_op.ref_start)
        else:
            orig_e = end_op.orig_end
        # Confidence: 1.0 if both endpoints anchored in equal blocks (only edits
        # in between are deletions/insertions we span over cleanly), else 0.75.
        both_anchored = start_op.tag == "equal" and end_op.tag == "equal"
        return orig_s, orig_e, 1.0 if both_anchored else 0.75


__all__ = ["EditLog"]
