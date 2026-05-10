# @summary
# LLM-based community summarization for knowledge graph communities.
# Builds prompts from entity descriptions, calls LLM, manages parallel execution.
# Exports: CommunitySummarizer
# Deps: logging, datetime, concurrent.futures, src.knowledge_graph.common.types,
#        src.knowledge_graph.common.schemas, src.knowledge_graph.community.schemas,
#        src.knowledge_graph.backend, src.platform.llm.provider
# @end-summary
"""LLM-based community summarization.

Generates concise thematic summaries for entity communities detected by
the Leiden algorithm.  Each community's member entities and their descriptions
are assembled into an LLM prompt; the response is stored as a
``CommunitySummary``.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.knowledge_graph.backend import GraphStorageBackend
from src.knowledge_graph.common import Entity
from src.knowledge_graph.common import KGConfig
from src.knowledge_graph.community.schemas import CommunityDiff, CommunitySummary

logger = logging.getLogger(__name__)

__all__ = ["CommunitySummarizer"]

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = (
    "You are a knowledge graph analyst. Given a list of entities and their "
    "descriptions from a knowledge graph community, write a concise thematic "
    "summary (2-4 sentences) that identifies: (1) the primary topic or theme "
    "of this cluster, (2) the key entities and their roles, (3) the main "
    "relationships between entities. Be specific and technical. Do not use "
    "filler phrases."
)


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------


class CommunitySummarizer:
    """Generate LLM-based summaries for knowledge graph communities.

    Parameters
    ----------
    config:
        KG runtime configuration (token budgets, temperature, worker count).
    llm_provider:
        Optional LLMProvider instance.  Falls back to the process-wide
        singleton via ``get_llm_provider()`` when ``None``.
    """

    def __init__(
        self,
        config: KGConfig,
        llm_provider: Optional[Any] = None,
    ) -> None:
        self._config = config

        if llm_provider is not None:
            self._provider = llm_provider
        else:
            from src.platform.llm import get_llm_provider
            self._provider = get_llm_provider()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize_community(
        self,
        community_id: int,
        members: List[str],
        backend: GraphStorageBackend,
    ) -> CommunitySummary:
        """Summarize a single community.

        Args:
            community_id: Leiden-assigned community identifier.
            members: Canonical entity names belonging to this community.
            backend: Graph storage backend for entity lookups.

        Returns:
            A ``CommunitySummary`` for the community.
        """
        messages = self._build_prompt(members, backend)
        summary_text = self._call_llm(messages)
        return CommunitySummary(
            community_id=community_id,
            summary_text=summary_text,
            member_count=len(members),
            member_names=list(members),
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    def summarize_all(
        self,
        communities: Dict[int, List[str]],
        backend: GraphStorageBackend,
    ) -> Dict[int, CommunitySummary]:
        """Summarize all communities in parallel, skipping the misc bucket (-1).

        Args:
            communities: Mapping of community_id to member entity names.
            backend: Graph storage backend for entity lookups.

        Returns:
            Mapping of community_id to ``CommunitySummary`` (failures excluded).
        """
        results: Dict[int, CommunitySummary] = {}
        targets = {
            cid: members
            for cid, members in communities.items()
            if cid != -1
        }

        if not targets:
            return results

        max_workers = self._config.community_summary_max_workers
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self.summarize_community, cid, members, backend): cid
                for cid, members in targets.items()
            }
            for future in as_completed(futures):
                cid = futures[future]
                try:
                    results[cid] = future.result()
                except Exception:
                    logger.warning(
                        "Community %d summarization failed, skipping", cid,
                        exc_info=True,
                    )

        return results

    def refresh(
        self,
        diff: CommunityDiff,
        communities: Dict[int, List[str]],
        backend: GraphStorageBackend,
        existing_summaries: Dict[int, CommunitySummary],
    ) -> Dict[int, CommunitySummary]:
        """Incrementally refresh summaries based on a community diff.

        Re-summarizes new and changed communities, carries forward unchanged
        ones, and discards removed communities.

        Args:
            diff: Membership changes between detection runs.
            communities: Current community partition (id -> member names).
            backend: Graph storage backend for entity lookups.
            existing_summaries: Previously generated summaries.

        Returns:
            Updated mapping of community_id to ``CommunitySummary``.
        """
        results: Dict[int, CommunitySummary] = {}

        # Carry forward unchanged summaries
        for cid in diff.unchanged_communities:
            if cid in existing_summaries:
                results[cid] = existing_summaries[cid]

        # Re-summarize new and changed communities
        to_resummarize = {
            cid: communities[cid]
            for cid in diff.new_communities | diff.changed_communities
            if cid in communities
        }
        if to_resummarize:
            fresh = self.summarize_all(to_resummarize, backend)
            results.update(fresh)

        # diff.removed_communities are simply not included
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        members: List[str],
        backend: GraphStorageBackend,
    ) -> List[Dict[str, str]]:
        """Assemble the LLM prompt from member entity descriptions.

        Args:
            members: Canonical entity names in the community.
            backend: Graph storage backend for entity lookups.

        Returns:
            OpenAI-style messages list (system + user).
        """
        entities_with_desc: List[tuple] = []

        for name in members:
            entity: Optional[Entity] = backend.get_entity(name)
            if entity is None:
                continue

            if entity.current_summary:
                desc = entity.current_summary
            elif entity.raw_mentions:
                desc = " ".join(m.text for m in entity.raw_mentions)
            else:
                desc = name

            entities_with_desc.append((name, desc, entity.mention_count))

        formatted = self._truncate_descriptions(
            entities_with_desc,
            self._config.community_summary_input_max_tokens,
        )

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": formatted},
        ]

    def _truncate_descriptions(
        self,
        entities_with_desc: List[tuple],
        max_tokens: int,
    ) -> str:
        """Truncate entity descriptions to fit within the token budget.

        Uses a simple heuristic: ``word_count * 1.3`` as token estimate.
        Entities with the fewest mentions are dropped first.

        Args:
            entities_with_desc: List of (name, description, mention_count) tuples.
            max_tokens: Maximum token budget for the concatenated output.

        Returns:
            Formatted string of entity descriptions.
        """
        # Sort by mention_count ascending (least important first)
        sorted_entities = sorted(entities_with_desc, key=lambda t: t[2])

        def _estimate_tokens(text: str) -> float:
            return len(text.split()) * 1.3

        # Remove from front (lowest mention count) until budget is met
        while sorted_entities:
            total = sum(
                _estimate_tokens(f"Entity: {name}\nDescription: {desc}\n\n")
                for name, desc, _ in sorted_entities
            )
            if total <= max_tokens:
                break
            sorted_entities.pop(0)

        lines: List[str] = []
        for name, desc, _ in sorted_entities:
            lines.append(f"Entity: {name}\nDescription: {desc}\n")

        return "\n".join(lines)

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call the LLM provider and return the response content.

        Args:
            messages: OpenAI-style chat messages.

        Returns:
            Generated summary text.

        Raises:
            RuntimeError: Wraps any provider exception with model/config context.
        """
        try:
            response = self._provider.generate(
                messages=messages,
                max_tokens=self._config.community_summary_output_max_tokens,
                temperature=self._config.community_summary_temperature,
            )
            return response.content
        except Exception as exc:
            logger.error(
                "LLM provider call failed (max_tokens=%d, temperature=%s): %s",
                self._config.community_summary_output_max_tokens,
                self._config.community_summary_temperature,
                exc,
                exc_info=True,
            )
            raise RuntimeError(
                f"Community summarization LLM call failed: {exc}"
            ) from exc
