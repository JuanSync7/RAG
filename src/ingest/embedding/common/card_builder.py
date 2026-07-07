# @summary
# Pure, deterministic builder for §6.4 baseline document "cards" used by the
# RAPTOR-lite document-routing feature. Turns one document's chunks into a
# card-record dict (title + ordered/deduped section headings + compact
# card_text) suitable for embedding into the card index. No LLM, no I/O.
# Exports: build_document_card
# Deps: typing (stdlib only)
# @end-summary
"""Pure document-card construction (DOCUMENT_ROUTING_DESIGN.md §6.4, §11).

A "card" is a tiny per-document record whose ``card_text`` is embedded into a
small card index for Stage-1 document routing. The baseline card is built
purely from structural metadata already present on chunks (title + section
headings) — no LLM and no re-ingest required.

This module is intentionally side-effect-free and deterministic: the same
input chunks always yield byte-identical output. It performs no network, file,
clock, or random access. LLM summaries (the RAPTOR-style upgrade) are passed in
by the caller via ``summary`` and are never generated here.

Chunk inputs are accessed through a small accessor so the builder tolerates both
``ProcessedChunk``-like objects (``.text`` / ``.metadata`` attributes) and plain
dicts carrying a ``"metadata"`` key.
"""

from __future__ import annotations

from typing import Any


def _chunk_metadata(chunk: Any) -> dict[str, Any]:
    """Return the metadata mapping for a chunk, tolerating objects or dicts.

    Accepts a ``ProcessedChunk``-like object exposing a ``metadata`` attribute,
    or a plain mapping with a ``"metadata"`` key. Returns an empty dict when no
    metadata is present.
    """
    meta = getattr(chunk, "metadata", None)
    if meta is None and isinstance(chunk, dict):
        meta = chunk.get("metadata")
    if isinstance(meta, dict):
        return meta
    return {}


def _heading_for_chunk(meta: dict[str, Any]) -> str:
    """Derive the single best section heading for one chunk's metadata.

    Precedence (richest first):

    1. ``heading_path`` — a list; use its last non-empty element (the most
       specific heading in the breadcrumb).
    2. ``heading`` — a scalar heading field.
    3. ``section_path`` — a ``" > "``-joined breadcrumb; use its terminal
       segment.

    Returns the trimmed heading, or ``""`` when none is available.
    """
    path = meta.get("heading_path")
    if isinstance(path, (list, tuple)):
        for part in reversed(path):
            text = str(part).strip()
            if text:
                return text

    heading = meta.get("heading")
    if heading is not None:
        text = str(heading).strip()
        if text:
            return text

    section_path = meta.get("section_path")
    if section_path:
        # Terminal segment of a "Top > Mid > Leaf" breadcrumb.
        text = str(section_path).split(">")[-1].strip()
        if text:
            return text

    return ""


def _is_leaf(meta: dict[str, Any]) -> bool:
    """True when the chunk is a leaf (not a synthesised section node).

    Section nodes carry ``node_kind == "section"`` (see
    ``tree_node_synthesis``); everything else (``"chunk"`` or absent) is a leaf.
    """
    return str(meta.get("node_kind", "") or "") != "section"


def build_document_card(
    chunks: list[Any],
    *,
    max_headings: int,
    with_summary: bool = False,
    summary: str = "",
) -> dict:
    """Build the §6.4 baseline document card for ONE document's chunks.

    The card is a routing-only record (never sent to the LLM). It is derived
    purely and deterministically from the chunks' structural metadata.

    Args:
        chunks: All chunk objects for a single document. Each may be a
            ``ProcessedChunk``-like object (``.text`` / ``.metadata``) or a
            plain dict with a ``"metadata"`` key. May include leaf chunks
            (``node_kind == "chunk"`` or absent) and synthesised section nodes
            (``node_kind == "section"``).
        max_headings: Maximum number of section headings retained in the card
            (the §11 ``.xlsx`` guard — a spreadsheet with hundreds of
            sheet/column headings must not blow up the card). Excess headings
            beyond the cap are dropped, never an error.
        with_summary: When True and ``summary`` is non-empty, the summary is
            appended to ``card_text`` (the embedded routing text). When False,
            the summary is excluded from ``card_text``.
        summary: An optional pre-computed document summary. It is ALWAYS stored
            in the returned ``"summary"`` field regardless of ``with_summary``;
            ``with_summary`` only controls whether it also appears in
            ``card_text``.

    Returns:
        A card-record dict with exactly these keys (the contract shared with the
        card store and the emission node):

        - ``document_id`` (str): from ``metadata["document_id"]`` (or ``""``).
        - ``title`` (str): see precedence below.
        - ``source`` (str): ``metadata["source"]`` or ``metadata["source_key"]``
          (whichever is present first), else ``""``.
        - ``section_headings`` (list[str]): deduped section headings in
          first-seen document order, capped at ``max_headings``.
        - ``card_text`` (str): compact deterministic embedding text — the title
          line, then each heading on its own ``"## "``-prefixed line, then the
          summary when ``with_summary`` and ``summary`` is non-empty.
        - ``summary`` (str): the passed-in ``summary`` (``""`` for the baseline).
        - ``num_chunks`` (int): count of LEAF chunks (``node_kind != "section"``).

    Title precedence (first non-empty wins):
        1. ``metadata["title"]``
        2. the first (topmost) derived section heading
        3. ``metadata["source"]`` / ``metadata["source_key"]`` /
           ``metadata["source_name"]``
        4. ``""``
    """
    document_id = ""
    title_meta = ""
    source = ""
    headings: list[str] = []
    seen_headings: set[str] = set()
    num_chunks = 0

    for chunk in chunks:
        meta = _chunk_metadata(chunk)

        # First-seen, non-empty document-level fields. Chunks of one document
        # share these; we take the first present value.
        if not document_id:
            document_id = str(meta.get("document_id", "") or "")
        if not title_meta:
            candidate = str(meta.get("title", "") or "").strip()
            if candidate:
                title_meta = candidate
        if not source:
            for key in ("source", "source_key", "source_name"):
                candidate = str(meta.get(key, "") or "").strip()
                if candidate:
                    source = candidate
                    break

        if _is_leaf(meta):
            num_chunks += 1

        # Headings contribute in document order; dedupe case-sensitively but
        # trimmed. Section nodes may still contribute headings.
        heading = _heading_for_chunk(meta)
        if heading and heading not in seen_headings:
            seen_headings.add(heading)
            headings.append(heading)

    # §11 xlsx guard: cap the heading list (dropping excess is not an error).
    capped_headings = headings[: max(0, int(max_headings))]

    # Title precedence: explicit title -> first heading -> source -> "".
    title = title_meta or (capped_headings[0] if capped_headings else "") or source

    # Compose the compact, deterministic embedding text.
    lines: list[str] = []
    if title:
        lines.append(title)
    lines.extend(f"## {h}" for h in capped_headings)
    card_text = "\n".join(lines)
    if with_summary and summary:
        card_text = f"{card_text}\n{summary}" if card_text else summary

    return {
        "document_id": document_id,
        "title": title,
        "source": source,
        "section_headings": capped_headings,
        "card_text": card_text,
        "summary": summary,
        "num_chunks": num_chunks,
    }
