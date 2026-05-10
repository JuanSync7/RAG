# @summary
# Entity description accumulation and token-budget management.
# Exports: DescriptionManager
# Deps: src.knowledge_graph.common.schemas
# @end-summary
"""Entity description accumulation and management.

Handles appending new mentions, enforcing token budgets,
and building summary text for retrieval. Implements REQ-KG-400
through REQ-KG-405. Backend-agnostic — operates on dataclasses only.
"""

from __future__ import annotations

import logging
from typing import List

from src.knowledge_graph.common.schemas import EntityDescription

__all__ = ["DescriptionManager"]

logger = logging.getLogger("rag.knowledge_graph")


class DescriptionManager:
    """Manages entity description accumulation with token budget control.

    Strategy (per REQ-KG-400):
    1. Append new mentions with source attribution.
    2. Track approximate token count (word count as proxy).
    3. When exceeding budget, trim oldest mentions (keep most recent).
    4. Build concatenated summary for retrieval use.

    Phase 1b will add LLM-based summarisation when over budget.
    """

    def __init__(self, token_budget: int = 512) -> None:
        self.token_budget = token_budget

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_mention(
        self,
        raw_mentions: List[EntityDescription],
        text: str,
        source: str,
        chunk_id: str = "",
    ) -> List[EntityDescription]:
        """Append a new mention to an entity's description list.

        Deduplicates by (text, source). If the total token count exceeds
        the budget after appending, the oldest mentions are trimmed.

        Args:
            raw_mentions: Current mention list for the entity.
            text: New mention text.
            source: Document source of the mention.
            chunk_id: Optional chunk identifier.

        Returns:
            Updated mention list.
        """
        # Deduplicate — skip if exact same text from same source
        for existing in raw_mentions:
            if existing.text == text and existing.source == source:
                return raw_mentions

        new_mention = EntityDescription(text=text, source=source, chunk_id=chunk_id)
        updated = list(raw_mentions)
        updated.append(new_mention)

        # Check token budget
        total_tokens = self._count_tokens(updated)
        if total_tokens > self.token_budget:
            updated = self._trim_to_budget(updated)

        return updated

    def build_summary(self, raw_mentions: List[EntityDescription]) -> str:
        """Build a concatenated summary from mentions for retrieval.

        Phase 1: simple concatenation with source attribution.
        Phase 1b: LLM summarisation when over budget.

        Args:
            raw_mentions: Entity mention list.

        Returns:
            Semicolon-separated summary text.
        """
        if not raw_mentions:
            return ""

        parts: list[str] = []
        seen_sources: set[str] = set()

        for mention in raw_mentions:
            tag = f"[{mention.source}]" if mention.source else ""
            if mention.source and mention.source not in seen_sources:
                parts.append(f"{tag} {mention.text}")
                seen_sources.add(mention.source)
            else:
                parts.append(mention.text)

        return " | ".join(parts)

    def get_retrieval_text(
        self,
        current_summary: str,
        raw_mentions: List[EntityDescription],
    ) -> str:
        """Return the best text for retrieval context.

        Prefers ``current_summary``; falls back to joining raw mentions.

        Args:
            current_summary: Pre-built summary (may be empty).
            raw_mentions: Raw mention list.

        Returns:
            Text suitable for injection into retrieval context.
        """
        if current_summary:
            return current_summary
        if raw_mentions:
            return " ".join(m.text for m in raw_mentions)
        return ""

    def count_tokens(self, raw_mentions: List[EntityDescription]) -> int:
        """Public token-count accessor."""
        return self._count_tokens(raw_mentions)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _count_tokens(mentions: List[EntityDescription]) -> int:
        """Approximate token count using word splitting."""
        return sum(len(m.text.split()) for m in mentions)

    def _trim_to_budget(
        self, mentions: List[EntityDescription]
    ) -> List[EntityDescription]:
        """Trim oldest mentions, keeping most recent within budget."""
        result: list[EntityDescription] = []
        total = 0
        for mention in reversed(mentions):
            mention_tokens = len(mention.text.split())
            if total + mention_tokens <= self.token_budget:
                result.append(mention)
                total += mention_tokens
            else:
                break
        result.reverse()
        return result
