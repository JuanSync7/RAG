"""Fixture loader for retrieval evaluation.

A fixture lives in a directory containing three JSON files:
- ``corpus.json``  — {documents: [{document_id, chunks: [chunk_dict]}]}
- ``queries.json`` — [{qid, text, qtype}]
- ``gold.json``    — {qid: [chunk_id, ...]}

The loader fills derived fields on every chunk:
- ``parent_section_id`` from ``build_parent_section_id(document_id, heading_path)``
- ``tree_level`` from ``len(heading_path)``
- ``document_id`` propagated from the parent doc
- ``source`` set to the document_id (for filter/diversity bookkeeping)
"""
# @summary
# JSON fixture loader for retrieval eval; derives parent_section_id and
# tree_level so JSON authors only specify heading_path.
# Exports: Query, Fixture, load_fixture
# Deps: src.vector_db.weaviate.store.build_parent_section_id
# @end-summary
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from src.vector_db.weaviate.store import build_section_node_id


@dataclass
class Query:
    qid: str
    text: str
    qtype: str  # "qfs" | "factoid"


@dataclass
class Fixture:
    documents: list[dict]
    queries: list[Query]
    gold: dict[str, set[str]]

    def all_chunks(self) -> Iterator[dict]:
        for doc in self.documents:
            for chunk in doc["chunks"]:
                yield chunk


def _enrich_chunk(chunk: dict, document_id: str) -> dict:
    heading_path = list(chunk.get("heading_path") or [])
    enriched = dict(chunk)
    enriched["document_id"] = document_id
    enriched.setdefault("source", document_id)
    enriched["heading_path"] = heading_path
    enriched["tree_level"] = len(heading_path)
    enriched.setdefault("node_kind", "chunk")
    # Per TREE_RETRIEVAL_DESIGN.md §4.1: a leaf's parent_section_id points
    # to the chunk_id of its containing section node. For section nodes
    # themselves, parent_section_id points one level UP in the tree (used
    # by Stage 4c lift to gather sibling sections), and chunk_id is
    # canonicalised to the deterministic section-node id so descent's
    # ``section.chunk_id`` lookup matches leaf.parent_section_id.
    if enriched["node_kind"] == "section":
        enriched["chunk_id"] = build_section_node_id(document_id, heading_path)
        enriched["parent_section_id"] = build_section_node_id(
            document_id, heading_path[:-1]
        )
    else:
        enriched["parent_section_id"] = build_section_node_id(
            document_id, heading_path
        )
    return enriched


def load_fixture(path: str | Path) -> Fixture:
    """Load a fixture from ``path`` and return a hydrated ``Fixture``."""
    root = Path(path)
    corpus = json.loads((root / "corpus.json").read_text())
    queries_raw = json.loads((root / "queries.json").read_text())
    gold_raw = json.loads((root / "gold.json").read_text())

    documents: list[dict] = []
    for doc in corpus["documents"]:
        document_id = doc["document_id"]
        chunks = [_enrich_chunk(c, document_id) for c in doc["chunks"]]
        documents.append({"document_id": document_id, "chunks": chunks})

    queries = [Query(**q) for q in queries_raw]
    gold = {qid: set(ids) for qid, ids in gold_raw.items()}
    return Fixture(documents=documents, queries=queries, gold=gold)
