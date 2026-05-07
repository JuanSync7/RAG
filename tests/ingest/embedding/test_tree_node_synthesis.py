# @summary
# Tests for src/ingest/embedding/nodes/tree_node_synthesis.py.
# Covers: feature-flag short-circuit (regression gate), section emission for
#         each unique heading_path prefix, parent_section_id stability, table
#         child snippet handling, and degenerate no-headings doc.
# @end-summary
"""Tests for tree_node_synthesis_node (TREE_RETRIEVAL_DESIGN.md §3.2)."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.ingest.common.schemas import ProcessedChunk
from src.ingest.common.types import IngestionConfig, Runtime
from src.ingest.embedding.nodes.tree_node_synthesis import tree_node_synthesis_node


def _leaf(text: str, heading_path: list[str], **extra) -> ProcessedChunk:
    meta = {
        "document_id": "doc-A",
        "source": "doc.md",
        "heading_path": heading_path,
        "heading": heading_path[-1] if heading_path else "",
        "section_path": " > ".join(heading_path),
        "heading_level": len(heading_path),
        "tenant_id": "default",
        "chunk_type": "text",
        **extra,
    }
    return ProcessedChunk(text=text, metadata=meta)


def _state(chunks, *, enable: bool) -> dict:
    config = IngestionConfig()
    config.enable_tree_retrieval_ingest = enable
    runtime = Runtime(
        config=config,
        embedder=MagicMock(),
        weaviate_client=MagicMock(),
    )
    return {
        "chunks": chunks,
        "source_name": "doc.md",
        "source_key": "local_fs:doc:1",
        "errors": [],
        "processing_log": [],
        "runtime": runtime,
    }


class TestFeatureFlag:
    """Regression gate: disabled path must not mutate the chunks list."""

    def test_disabled_returns_no_chunks_update(self):
        leaves = [_leaf("a", ["Ch1", "1.1"]), _leaf("b", ["Ch1", "1.2"])]
        state = _state(leaves, enable=False)
        result = tree_node_synthesis_node(state)
        assert "chunks" not in result, "disabled path must not rewrite chunks"
        assert any("tree_node_synthesis:skip" in entry
                   for entry in result.get("processing_log", []))

    def test_enabled_emits_sections(self):
        leaves = [_leaf("a", ["Ch1", "1.1"]), _leaf("b", ["Ch1", "1.2"])]
        state = _state(leaves, enable=True)
        result = tree_node_synthesis_node(state)
        out = result["chunks"]
        assert len(out) > len(leaves)
        sections = [c for c in out if c.metadata.get("node_kind") == "section"]
        # Three sections: ["Ch1"], ["Ch1","1.1"], ["Ch1","1.2"].
        assert len(sections) == 3


class TestSectionEmission:
    def test_emits_one_section_per_unique_prefix(self):
        leaves = [
            _leaf("intro", ["Ch1", "1.1"]),
            _leaf("body", ["Ch1", "1.1"]),
            _leaf("other", ["Ch1", "1.2"]),
            _leaf("deep", ["Ch2", "2.1", "2.1.1"]),
        ]
        state = _state(leaves, enable=True)
        result = tree_node_synthesis_node(state)
        sections = [c for c in result["chunks"] if c.metadata.get("node_kind") == "section"]
        prefixes = {tuple(s.metadata["heading_path"]) for s in sections}
        assert prefixes == {
            ("Ch1",),
            ("Ch1", "1.1"),
            ("Ch1", "1.2"),
            ("Ch2",),
            ("Ch2", "2.1"),
            ("Ch2", "2.1", "2.1.1"),
        }

    def test_parent_section_id_links_siblings(self):
        leaves = [
            _leaf("a", ["Ch1", "1.1"]),
            _leaf("b", ["Ch1", "1.2"]),
        ]
        state = _state(leaves, enable=True)
        result = tree_node_synthesis_node(state)
        sections = {tuple(s.metadata["heading_path"]): s
                    for s in result["chunks"]
                    if s.metadata.get("node_kind") == "section"}
        # The two leaf sections share the same parent_section_id (Ch1).
        ps_11 = sections[("Ch1", "1.1")].metadata["parent_section_id"]
        ps_12 = sections[("Ch1", "1.2")].metadata["parent_section_id"]
        assert ps_11 and ps_11 == ps_12

    def test_child_count_is_set(self):
        leaves = [
            _leaf("a", ["Ch1", "1.1"]),
            _leaf("b", ["Ch1", "1.1"]),
            _leaf("c", ["Ch1", "1.2"]),
        ]
        state = _state(leaves, enable=True)
        result = tree_node_synthesis_node(state)
        ch1 = next(c for c in result["chunks"]
                   if c.metadata.get("node_kind") == "section"
                   and tuple(c.metadata["heading_path"]) == ("Ch1",))
        # Ch1 has two direct sub-sections (1.1 and 1.2), no direct leaves.
        assert ch1.metadata["child_count"] == 2


class TestTableSnippets:
    def test_table_child_uses_caption_and_first_row(self):
        table_leaf = _leaf(
            "raw markdown table text",
            ["Ch1", "1.1"],
            chunk_type="table",
            table_caption="Register map",
            table_cells=[["Field", "Bits", "Reset"], ["MODE", "[1:0]", "0x0"]],
        )
        state = _state([table_leaf], enable=True)
        result = tree_node_synthesis_node(state)
        sec_11 = next(c for c in result["chunks"]
                      if c.metadata.get("node_kind") == "section"
                      and tuple(c.metadata["heading_path"]) == ("Ch1", "1.1"))
        assert "Register map" in sec_11.text
        assert "Field" in sec_11.text


class TestDegenerateCases:
    def test_no_headings_emits_no_sections(self):
        leaves = [_leaf("plain", []), _leaf("more", [])]
        state = _state(leaves, enable=True)
        result = tree_node_synthesis_node(state)
        # No "chunks" key on the no-headings short-circuit.
        assert "chunks" not in result
        assert any("no_headings" in e for e in result.get("processing_log", []))

    def test_empty_chunks(self):
        state = _state([], enable=True)
        result = tree_node_synthesis_node(state)
        assert "chunks" not in result
        assert any("empty" in e for e in result.get("processing_log", []))
