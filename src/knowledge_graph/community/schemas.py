# @summary
# Data contracts for community detection and summarization.
# Exports: CommunitySummary, CommunityDiff
# Deps: dataclasses, typing
# @end-summary
"""Data contracts for community detection and summarization.

Provides ``CommunitySummary`` (per-community LLM summary) and
``CommunityDiff`` (membership changes between detection runs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set

__all__ = ["CommunitySummary", "CommunityDiff"]


@dataclass
class CommunitySummary:
    """Summarised representation of a single community cluster.

    Attributes:
        community_id: Integer identifier assigned by Leiden.
        summary_text: LLM-generated thematic summary (2-4 sentences).
        member_count: Number of entities in this community.
        member_names: Canonical names of member entities.
        generated_at: ISO 8601 timestamp of summary generation.
    """

    community_id: int
    summary_text: str
    member_count: int
    member_names: List[str] = field(default_factory=list)
    generated_at: str = ""


@dataclass
class CommunityDiff:
    """Diff between two consecutive community detection runs.

    The union of all four sets covers every community ID that appeared
    in either the old or new partition.

    Attributes:
        new_communities: IDs present in new partition but not old.
        removed_communities: IDs present in old partition but not new.
        changed_communities: IDs present in both but with different member sets.
        unchanged_communities: IDs present in both with identical member sets.
    """

    new_communities: Set[int] = field(default_factory=set)
    removed_communities: Set[int] = field(default_factory=set)
    changed_communities: Set[int] = field(default_factory=set)
    unchanged_communities: Set[int] = field(default_factory=set)
