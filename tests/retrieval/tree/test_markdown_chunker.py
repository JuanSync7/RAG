"""TDD tests for the markdown→fixture chunker used to build real-data
eval fixtures from OpenTitan IP docs (and similarly structured markdown).

The chunker is intentionally simple: each non-empty paragraph becomes one
chunk, code-fence blocks become a single chunk, and ``heading_path``
tracks the H1..H6 stack at the chunk's position.
"""
from __future__ import annotations


SAMPLE = """# Theory of Operation

## Block Diagram

A high-level block diagram of the UART.

## Design Details

### Serial interface

The TX/RX serial lines are high when idle.

Data starts with a START bit followed by 8 data bits.

### Transmission

A write to WDATA enqueues a data byte into the FIFO.

```c
uart_tx_byte(0x42);
```

#### tx_done

The tx_done interrupt fires when the FIFO drains.
"""


class TestChunksFromMarkdown:
    def test_emits_chunk_per_paragraph(self):
        from src.eval.markdown_chunker import chunks_from_markdown

        chunks = chunks_from_markdown(SAMPLE, document_id="uart_theory")
        # 5 paragraph leaves + 1 code block = 6 chunks
        assert len(chunks) == 6, [c["text"][:40] for c in chunks]

    def test_heading_path_tracks_stack(self):
        from src.eval.markdown_chunker import chunks_from_markdown

        chunks = chunks_from_markdown(SAMPLE, document_id="uart_theory")

        # First leaf is under "## Block Diagram"
        assert chunks[0]["heading_path"] == ["Theory of Operation", "Block Diagram"]

        # Serial-interface paragraphs are under H3
        serial_chunks = [
            c for c in chunks
            if c["heading_path"][-1] == "Serial interface"
        ]
        assert len(serial_chunks) == 2
        assert all(
            c["heading_path"]
            == ["Theory of Operation", "Design Details", "Serial interface"]
            for c in serial_chunks
        )

        # tx_done is H4 — full stack must include all four levels
        tx_done_chunks = [
            c for c in chunks if c["heading_path"][-1] == "tx_done"
        ]
        assert len(tx_done_chunks) == 1
        assert tx_done_chunks[0]["heading_path"] == [
            "Theory of Operation",
            "Design Details",
            "Transmission",
            "tx_done",
        ]

    def test_code_fence_kept_as_single_chunk(self):
        from src.eval.markdown_chunker import chunks_from_markdown

        chunks = chunks_from_markdown(SAMPLE, document_id="uart_theory")
        code_chunks = [c for c in chunks if "uart_tx_byte" in c["text"]]
        assert len(code_chunks) == 1
        # The whole code body lives in one chunk (no paragraph splitting)
        assert "```" in code_chunks[0]["text"]

    def test_chunks_carry_document_id_and_node_kind(self):
        from src.eval.markdown_chunker import chunks_from_markdown

        chunks = chunks_from_markdown(SAMPLE, document_id="uart_theory")
        for c in chunks:
            assert c["document_id"] == "uart_theory"
            assert c["node_kind"] == "chunk"
            # chunk_id must be unique and stable
            assert c["chunk_id"]
        ids = [c["chunk_id"] for c in chunks]
        assert len(set(ids)) == len(ids)


class TestEmitSectionNodes:
    def test_section_nodes_emitted_per_unique_heading_path(self):
        from src.eval.markdown_chunker import (
            chunks_from_markdown,
            emit_section_nodes,
        )

        leaves = chunks_from_markdown(SAMPLE, document_id="uart_theory")
        sections = emit_section_nodes(leaves, document_id="uart_theory")

        # Unique heading paths from the leaves:
        # Block Diagram, Serial interface, Transmission, tx_done
        # Plus their ancestors: Theory of Operation, Design Details
        # 6 unique non-empty prefixes
        paths = {tuple(s["heading_path"]) for s in sections}
        assert ("Theory of Operation",) in paths
        assert ("Theory of Operation", "Design Details") in paths
        assert (
            "Theory of Operation",
            "Design Details",
            "Serial interface",
        ) in paths
        assert (
            "Theory of Operation",
            "Design Details",
            "Transmission",
            "tx_done",
        ) in paths

    def test_section_text_includes_heading_and_child_snippet(self):
        from src.eval.markdown_chunker import (
            chunks_from_markdown,
            emit_section_nodes,
        )

        leaves = chunks_from_markdown(SAMPLE, document_id="uart_theory")
        sections = emit_section_nodes(leaves, document_id="uart_theory")

        serial = next(
            s for s in sections
            if s["heading_path"][-1] == "Serial interface"
        )
        assert "Serial interface" in serial["text"]
        # At least one of the two paragraph snippets bubbles up
        assert "TX/RX" in serial["text"] or "START bit" in serial["text"]
