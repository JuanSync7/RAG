# @summary
# LangGraph node that emits structural section nodes alongside leaf chunks.
# For each unique heading_path prefix in state["chunks"], constructs one
# section node carrying heading + truncated child snippets so the existing
# embedding/storage path indexes it as a tree-internal node.
# No-op when config.enable_tree_retrieval_ingest is False.
# Exports: tree_node_synthesis_node
# Deps: src.ingest.common, src.ingest.embedding.state, src.vector_db
# @end-summary

"""Tree-node synthesis (TREE_RETRIEVAL_DESIGN.md §3.2).

Emits one section node per unique ``heading_path`` prefix observed in the
leaf chunks. Section text = heading + truncated head of each direct child
(or `caption + first markdown row` when the child is a table chunk). No
LLM calls — RAPTOR-style summarisation is deferred to Appendix B.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.ingest.common import ProcessedChunk, append_processing_log
from src.ingest.common.observability import node_span
from src.ingest.embedding.state import EmbeddingPipelineState
from src.vector_db.weaviate.store import (
    build_parent_section_id,
    build_section_node_id,
)

logger = logging.getLogger("rag.ingest.embedding.tree_node_synthesis")


def _table_snippet(meta: dict[str, Any], cap: int) -> str:
    """Build a table-aware snippet: caption + first non-separator row."""
    caption = (meta.get("table_caption") or "").strip()
    cells = meta.get("table_cells") or []
    first_row = ""
    if isinstance(cells, list) and cells:
        head = cells[0]
        if isinstance(head, list):
            first_row = " | ".join(str(c) for c in head if c is not None)
        elif isinstance(head, str):
            first_row = head
    parts = [p for p in (caption, first_row) if p]
    snippet = " — ".join(parts)
    return snippet[:cap]


def _child_snippet(text: str, meta: dict[str, Any], cap: int) -> str:
    """Pick the right snippet style for a child chunk."""
    if (meta.get("chunk_type") or "").lower() == "table":
        snip = _table_snippet(meta, cap)
        if snip:
            return snip
    return (text or "")[:cap]


def _section_id_for(document_id: str, heading_path: list[str]) -> str:
    """Deterministic section node id (document-scoped)."""
    return build_section_node_id(document_id, heading_path)


@node_span("tree_node_synthesis")
def tree_node_synthesis_node(state: EmbeddingPipelineState) -> dict[str, Any]:
    """Append one section node per unique heading_path prefix.

    Section nodes share the same metadata shape as leaves; downstream nodes
    (embedding_storage, commit) handle them uniformly. ``node_kind="section"``
    distinguishes them at retrieval time.

    Args:
        state: Embedding pipeline state.

    Returns:
        Partial state update with augmented ``chunks`` and ``processing_log``.
        When the feature flag is off, the chunks list is returned unchanged.
    """
    t0 = time.monotonic()
    config = state["runtime"].config
    if not getattr(config, "enable_tree_retrieval_ingest", False):
        return {
            "processing_log": append_processing_log(state, "tree_node_synthesis:skip"),
        }

    leaves: list[ProcessedChunk] = list(state.get("chunks") or [])
    if not leaves:
        return {
            "processing_log": append_processing_log(state, "tree_node_synthesis:empty"),
        }

    cap = int(getattr(config, "tree_section_child_snippet_chars", 200))
    max_chars = int(getattr(config, "tree_section_text_max_chars", 1500))

    # Group leaves by their *parent* heading_path so each prefix gets its
    # direct children, not transitive descendants.
    children_by_prefix: dict[tuple[str, ...], list[tuple[str, dict[str, Any]]]] = {}
    all_prefixes: set[tuple[str, ...]] = set()
    document_id = ""
    base_meta_template: dict[str, Any] = {}

    for ch in leaves:
        meta = dict(ch.metadata or {})
        document_id = document_id or str(meta.get("document_id", "") or "")
        if not base_meta_template:
            base_meta_template = {
                k: meta.get(k)
                for k in (
                    "source",
                    "source_uri",
                    "source_key",
                    "source_id",
                    "connector",
                    "source_version",
                    "tenant_id",
                    "document_id",
                    "title",
                    "author",
                    "date",
                    "tags",
                )
                if meta.get(k) is not None
            }
        path = [str(h) for h in (meta.get("heading_path") or []) if h]
        # Record every ancestor prefix as a section to emit.
        for depth in range(1, len(path) + 1):
            all_prefixes.add(tuple(path[:depth]))
        # A leaf is a direct child of the section bearing its exact heading
        # path. Section "Ch1 > 1.1" contains leaves with heading_path == ["Ch1","1.1"];
        # leaves under deeper subsections attach to *those* sections, not to "Ch1 > 1.1".
        leaf_prefix = tuple(path)
        children_by_prefix.setdefault(leaf_prefix, []).append((ch.text or "", meta))

    if not all_prefixes:
        # Document has no headings — degenerate case (TREE_RETRIEVAL_DESIGN §9.4).
        return {
            "processing_log": append_processing_log(
                state, "tree_node_synthesis:no_headings"
            ),
        }

    # Stable ordering: by depth, then by the path itself.
    ordered = sorted(all_prefixes, key=lambda p: (len(p), p))

    sections: list[ProcessedChunk] = []
    for prefix in ordered:
        prefix_list = list(prefix)
        heading = prefix_list[-1]
        # Direct children are leaves whose parent prefix matches.
        direct_children = children_by_prefix.get(prefix, [])
        # Sub-section prefixes immediately under this one.
        sub_prefixes = [
            p for p in ordered if len(p) == len(prefix) + 1 and p[:-1] == prefix
        ]
        child_count = len(direct_children) + len(sub_prefixes)

        # Assemble section text: heading line + child snippets, capped.
        parts: list[str] = [f"{heading}"]
        running = len(parts[0])
        for sub in sub_prefixes:
            sub_heading = sub[-1]
            piece = f"\n## {sub_heading}"
            if running + len(piece) > max_chars:
                break
            parts.append(piece)
            running += len(piece)
        for text, meta in direct_children:
            piece = "\n" + _child_snippet(text, meta, cap)
            if running + len(piece) > max_chars:
                break
            parts.append(piece)
            running += len(piece)
        section_text = "".join(parts).strip()
        if not section_text:
            continue

        section_meta: dict[str, Any] = {
            **base_meta_template,
            "section_path": " > ".join(prefix_list),
            "heading": heading,
            "heading_level": len(prefix_list),
            "heading_path": prefix_list,
            "node_kind": "section",
            "tree_level": len(prefix_list),
            # Section's parent: the section one level up (or "" at root).
            # Used by Stage 4c lift to find sibling sections.
            "parent_section_id": build_section_node_id(
                document_id, prefix_list[:-1]
            ),
            "child_count": child_count,
            # chunk_index / total_chunks are leaf-domain — sections sit outside
            # that numbering. We use a synthetic ID below; index/total stay 0.
            "chunk_index": 0,
            "total_chunks": 0,
            "chunk_type": "section",
            "chunk_id": _section_id_for(document_id, prefix_list),
        }
        sections.append(ProcessedChunk(text=section_text, metadata=section_meta))

    augmented = leaves + sections
    logger.info(
        "tree_node_synthesis: emitted %d section nodes from %d leaves (doc=%s)",
        len(sections),
        len(leaves),
        document_id or "<unknown>",
    )
    logger.debug("tree_node_synthesis_node completed in %.3fs", time.monotonic() - t0)
    return {
        "chunks": augmented,
        "processing_log": append_processing_log(
            state, f"tree_node_synthesis:emitted={len(sections)}"
        ),
    }
