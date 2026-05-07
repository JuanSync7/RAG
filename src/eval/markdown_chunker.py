"""Markdown→chunks converter for eval fixtures.

Splits a markdown document into paragraph-level chunks, tracking the H1..H6
heading stack at each chunk's position. Code-fence blocks become single
chunks (no paragraph splitting inside). Section nodes can then be derived
from the unique heading-path prefixes — same model as
``ingest/embedding/nodes/tree_node_synthesis.py``, but standalone (no
LangGraph state, no LLM, no Weaviate).

Used by:
- ``scripts/build_opentitan_fixture.py`` (one-shot fixture builders)
- eval-only paths that don't need the full ingest pipeline
"""
# @summary
# Standalone markdown→chunks/sections converter for eval fixtures.
# Tracks heading stack, splits paragraphs, preserves code fences whole.
# Exports: chunks_from_markdown, emit_section_nodes
# Deps: hashlib (stdlib)
# @end-summary
from __future__ import annotations

import hashlib
import re
from typing import Iterable

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_CODE_FENCE_RE = re.compile(r"^```")


def _chunk_id(document_id: str, index: int, text: str) -> str:
    payload = f"{document_id}:{index}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def chunks_from_markdown(text: str, *, document_id: str) -> list[dict]:
    """Split ``text`` into paragraph leaves, tracking the heading stack.

    Returns chunk dicts shaped for the eval fixture loader: each carries
    ``chunk_id``, ``text``, ``document_id``, ``heading_path``, ``node_kind``.
    """
    lines = text.splitlines()
    heading_stack: list[tuple[int, str]] = []  # [(level, title), ...]
    chunks: list[dict] = []

    buf: list[str] = []
    in_code_fence = False
    code_fence_buf: list[str] = []

    def _flush_paragraph() -> None:
        if not buf:
            return
        body = "\n".join(buf).strip()
        if body:
            path = [title for _lvl, title in heading_stack]
            idx = len(chunks)
            chunks.append({
                "chunk_id": _chunk_id(document_id, idx, body),
                "text": body,
                "document_id": document_id,
                "heading_path": path,
                "node_kind": "chunk",
            })
        buf.clear()

    def _flush_code_fence() -> None:
        if not code_fence_buf:
            return
        body = "\n".join(code_fence_buf).strip()
        if body:
            path = [title for _lvl, title in heading_stack]
            idx = len(chunks)
            chunks.append({
                "chunk_id": _chunk_id(document_id, idx, body),
                "text": body,
                "document_id": document_id,
                "heading_path": path,
                "node_kind": "chunk",
            })
        code_fence_buf.clear()

    for line in lines:
        if _CODE_FENCE_RE.match(line):
            if in_code_fence:
                code_fence_buf.append(line)
                _flush_code_fence()
                in_code_fence = False
            else:
                _flush_paragraph()
                in_code_fence = True
                code_fence_buf.append(line)
            continue

        if in_code_fence:
            code_fence_buf.append(line)
            continue

        m = _HEADING_RE.match(line)
        if m:
            _flush_paragraph()
            level = len(m.group(1))
            title = m.group(2).strip()
            # Pop deeper-or-equal levels off the stack, then push.
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            continue

        if line.strip() == "":
            _flush_paragraph()
            continue

        buf.append(line)

    _flush_paragraph()
    if in_code_fence:
        _flush_code_fence()

    return chunks


def emit_section_nodes(
    leaves: Iterable[dict],
    *,
    document_id: str,
    snippet_chars: int = 200,
    section_max_chars: int = 1500,
) -> list[dict]:
    """Derive one section node per unique heading-path prefix in ``leaves``.

    Section text = heading + truncated child snippets, mirroring the
    ``tree_node_synthesis`` ingest node's deterministic v1 strategy.
    """
    leaves = list(leaves)
    # Group leaves by every prefix of their heading_path (1..len)
    by_prefix: dict[tuple[str, ...], list[dict]] = {}
    for leaf in leaves:
        path = tuple(leaf.get("heading_path") or [])
        for i in range(1, len(path) + 1):
            by_prefix.setdefault(path[:i], []).append(leaf)

    sections: list[dict] = []
    for prefix, members in by_prefix.items():
        heading = prefix[-1]
        snippets: list[str] = []
        for leaf in members:
            snippet = (leaf.get("text") or "")[:snippet_chars]
            if snippet:
                snippets.append(snippet)
        section_text = (
            f"{heading}\n\n" + "\n\n".join(snippets)
        )[:section_max_chars]

        idx = len(sections)
        synthetic_text_for_id = "::section::" + ">".join(prefix)
        sections.append({
            "chunk_id": _chunk_id(document_id, 100_000 + idx, synthetic_text_for_id),
            "text": section_text,
            "document_id": document_id,
            "heading_path": list(prefix),
            "node_kind": "section",
            "child_count": len(members),
        })
    return sections
