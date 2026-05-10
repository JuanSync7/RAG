"""Community detection sub-package for the KG subsystem.

Provides Leiden-based community detection, LLM community summarization,
and data contracts (CommunitySummary, CommunityDiff).
"""

from src.knowledge_graph.community.schemas import CommunitySummary, CommunityDiff

__all__ = ["CommunitySummary", "CommunityDiff"]

# --- Auto-generated re-exports (fix_encapsulation.py) ---
from src.knowledge_graph.community.detector import CommunityDetector
from src.knowledge_graph.community.summarizer import CommunitySummarizer
