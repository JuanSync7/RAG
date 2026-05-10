# @summary
# Data contracts for entity resolution operations.
# Exports: MergeCandidate, ResolutionReport
# Deps: dataclasses
# @end-summary
"""Data contracts for entity resolution operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

__all__ = ["MergeCandidate", "ResolutionReport"]


@dataclass
class MergeCandidate:
    """A pair of entities identified for merging.

    Attributes:
        canonical: Name of the entity to keep.
        duplicate: Name of the entity to absorb.
        similarity: Cosine similarity score (1.0 for alias matches).
        reason: Human-readable merge reason.
    """

    canonical: str
    duplicate: str
    similarity: float
    reason: str


@dataclass
class ResolutionReport:
    """Results from a full entity resolution pass.

    Attributes:
        merges: Ordered list of merge operations performed.
        total_merged: Count of entities absorbed.
    """

    merges: List[MergeCandidate] = field(default_factory=list)
    total_merged: int = 0
