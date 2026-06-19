# @summary
# Unit tests for the pure document-card builder (slice A1):
# src/ingest/embedding/common/card_builder.py::build_document_card.
# Covers exact card_text/heading derivation, title fallback precedence,
# heading dedupe+order, the §11 xlsx max_headings cap, with_summary behaviour,
# leaf-only num_chunks counting, dict-tolerant accessors, and determinism.
# Exports: TestBuildDocumentCard
# Deps: pytest, src.ingest.common.ProcessedChunk,
#       src.ingest.embedding.common.card_builder
# @end-summary
"""Tests for ``build_document_card`` — the §6.4 baseline card builder.

The builder is PURE (no LLM, no I/O, deterministic). These tests exercise it
with synthetic ``ProcessedChunk`` objects and with plain dicts (the accessor
must tolerate both), asserting exact card output.
"""

from __future__ import annotations

from src.ingest.common import ProcessedChunk
from src.ingest.embedding.common.card_builder import build_document_card


def _chunk(text: str = "body", **meta) -> ProcessedChunk:
    """Construct a ProcessedChunk with the given metadata kwargs."""
    return ProcessedChunk(text=text, metadata=dict(meta))


class TestBuildDocumentCard:
    """Behaviour of the pure baseline document-card builder."""

    # ---- happy path: exact card from synthetic chunks -------------------

    def test_basic_card_exact_fields(self):
        chunks = [
            _chunk(
                document_id="doc-1",
                title="AXI Protocol Spec",
                source="specs/axi.pdf",
                heading_path=["Introduction"],
                node_kind="chunk",
            ),
            _chunk(
                document_id="doc-1",
                title="AXI Protocol Spec",
                source="specs/axi.pdf",
                heading_path=["Signals", "Write Channel"],
                node_kind="chunk",
            ),
            _chunk(
                document_id="doc-1",
                title="AXI Protocol Spec",
                source="specs/axi.pdf",
                heading_path=["Signals", "Read Channel"],
                node_kind="chunk",
            ),
        ]

        card = build_document_card(chunks, max_headings=60)

        assert card["document_id"] == "doc-1"
        assert card["title"] == "AXI Protocol Spec"
        assert card["source"] == "specs/axi.pdf"
        assert card["summary"] == ""
        assert card["num_chunks"] == 3
        # Deduped headings in first-seen document order. heading_path is the
        # richest source; we take its terminal heading per chunk.
        assert card["section_headings"] == [
            "Introduction",
            "Write Channel",
            "Read Channel",
        ]
        assert card["card_text"] == (
            "AXI Protocol Spec\n"
            "## Introduction\n"
            "## Write Channel\n"
            "## Read Channel"
        )

    def test_exact_dict_contract_keys(self):
        chunks = [_chunk(document_id="d", title="T", source="s", heading="H")]
        card = build_document_card(chunks, max_headings=60)
        assert set(card.keys()) == {
            "document_id",
            "title",
            "source",
            "section_headings",
            "card_text",
            "summary",
            "num_chunks",
        }

    # ---- title fallback precedence --------------------------------------

    def test_title_falls_back_to_first_heading(self):
        chunks = [
            _chunk(document_id="d", source="path/to/file.md", heading_path=["First H"]),
            _chunk(document_id="d", source="path/to/file.md", heading_path=["Second H"]),
        ]
        card = build_document_card(chunks, max_headings=60)
        assert card["title"] == "First H"

    def test_title_falls_back_to_source_when_no_headings(self):
        chunks = [_chunk(document_id="d", source="path/to/file.md")]
        card = build_document_card(chunks, max_headings=60)
        assert card["title"] == "path/to/file.md"
        assert card["section_headings"] == []
        assert card["card_text"] == "path/to/file.md"

    def test_title_falls_back_to_source_key_when_no_source(self):
        chunks = [_chunk(document_id="d", source_key="connector/key.docx")]
        card = build_document_card(chunks, max_headings=60)
        assert card["title"] == "connector/key.docx"
        assert card["source"] == "connector/key.docx"

    def test_title_empty_when_nothing_available(self):
        chunks = [_chunk(document_id="d")]
        card = build_document_card(chunks, max_headings=60)
        assert card["title"] == ""
        assert card["source"] == ""
        assert card["section_headings"] == []
        assert card["card_text"] == ""

    def test_explicit_title_wins_over_heading_and_source(self):
        chunks = [
            _chunk(
                document_id="d",
                title="Explicit Title",
                source="some/source.pdf",
                heading_path=["A Heading"],
            )
        ]
        card = build_document_card(chunks, max_headings=60)
        assert card["title"] == "Explicit Title"

    def test_blank_title_is_treated_as_absent(self):
        chunks = [_chunk(document_id="d", title="   ", heading_path=["Real H"])]
        card = build_document_card(chunks, max_headings=60)
        assert card["title"] == "Real H"

    # ---- heading dedupe + order -----------------------------------------

    def test_heading_dedupe_preserves_first_seen_order(self):
        chunks = [
            _chunk(document_id="d", title="T", heading_path=["Alpha"]),
            _chunk(document_id="d", title="T", heading_path=["Beta"]),
            _chunk(document_id="d", title="T", heading_path=["Alpha"]),  # dup
            _chunk(document_id="d", title="T", heading_path=["Gamma"]),
            _chunk(document_id="d", title="T", heading_path=["Beta"]),  # dup
        ]
        card = build_document_card(chunks, max_headings=60)
        assert card["section_headings"] == ["Alpha", "Beta", "Gamma"]

    def test_headings_trimmed_and_empties_dropped(self):
        chunks = [
            _chunk(document_id="d", title="T", heading_path=["  Spaced  "]),
            _chunk(document_id="d", title="T", heading_path=["", "  "]),
            _chunk(document_id="d", title="T", heading_path=["Spaced"]),  # dup after trim
            _chunk(document_id="d", title="T", heading_path=["Kept"]),
        ]
        card = build_document_card(chunks, max_headings=60)
        assert card["section_headings"] == ["Spaced", "Kept"]

    def test_dedupe_is_case_sensitive(self):
        chunks = [
            _chunk(document_id="d", title="T", heading_path=["Cache"]),
            _chunk(document_id="d", title="T", heading_path=["cache"]),
        ]
        card = build_document_card(chunks, max_headings=60)
        assert card["section_headings"] == ["Cache", "cache"]

    def test_heading_fallback_to_heading_then_section_path(self):
        # No heading_path: prefer `heading`, else terminal of `section_path`.
        chunks = [
            _chunk(document_id="d", title="T", heading="From Heading Field"),
            _chunk(document_id="d", title="T", section_path="Top > Mid > Leaf"),
        ]
        card = build_document_card(chunks, max_headings=60)
        assert card["section_headings"] == ["From Heading Field", "Leaf"]

    def test_heading_path_preferred_over_heading_field(self):
        # heading_path is the richest available — prefer it.
        chunks = [
            _chunk(
                document_id="d",
                title="T",
                heading_path=["Parent", "Path Heading"],
                heading="Other",
            )
        ]
        card = build_document_card(chunks, max_headings=60)
        assert card["section_headings"] == ["Path Heading"]

    # ---- xlsx-like cap (§11) --------------------------------------------

    def test_xlsx_like_many_headings_capped(self):
        chunks = [
            _chunk(document_id="sheet", title="Big Sheet", heading_path=[f"Col {i}"])
            for i in range(200)
        ]
        card = build_document_card(chunks, max_headings=60)
        assert len(card["section_headings"]) == 60
        # First-seen order preserved within the cap.
        assert card["section_headings"][0] == "Col 0"
        assert card["section_headings"][-1] == "Col 59"
        # Card text bounded: title line + 60 heading lines.
        assert card["card_text"].count("\n") == 60
        assert "Col 60" not in card["card_text"]
        # num_chunks still counts all leaves regardless of the cap.
        assert card["num_chunks"] == 200

    def test_cap_does_not_error_when_under_limit(self):
        chunks = [
            _chunk(document_id="d", title="T", heading_path=[f"H{i}"]) for i in range(3)
        ]
        card = build_document_card(chunks, max_headings=60)
        assert card["section_headings"] == ["H0", "H1", "H2"]

    # ---- with_summary behaviour -----------------------------------------

    def test_with_summary_appends_to_card_text(self):
        chunks = [_chunk(document_id="d", title="T", heading_path=["H1"])]
        card = build_document_card(
            chunks, max_headings=60, with_summary=True, summary="A concise summary."
        )
        assert card["summary"] == "A concise summary."
        assert card["card_text"].endswith("A concise summary.")
        assert "A concise summary." in card["card_text"]
        # Title + heading still present.
        assert card["card_text"].startswith("T\n## H1")

    def test_without_summary_excludes_from_card_text_but_stores_value(self):
        # Decision: the dict ALWAYS stores the passed summary; card_text only
        # includes it when with_summary is True.
        chunks = [_chunk(document_id="d", title="T", heading_path=["H1"])]
        card = build_document_card(
            chunks, max_headings=60, with_summary=False, summary="Hidden summary."
        )
        assert card["summary"] == "Hidden summary."
        assert "Hidden summary." not in card["card_text"]
        assert card["card_text"] == "T\n## H1"

    def test_with_summary_true_but_empty_summary_not_appended(self):
        chunks = [_chunk(document_id="d", title="T", heading_path=["H1"])]
        card = build_document_card(
            chunks, max_headings=60, with_summary=True, summary=""
        )
        assert card["summary"] == ""
        assert card["card_text"] == "T\n## H1"

    # ---- num_chunks counts only leaves ----------------------------------

    def test_num_chunks_excludes_section_nodes(self):
        chunks = [
            _chunk(document_id="d", title="T", heading_path=["A"], node_kind="chunk"),
            _chunk(document_id="d", title="T", heading_path=["B"], node_kind="chunk"),
            # A synthesised section node — must NOT count toward num_chunks.
            _chunk(document_id="d", title="T", heading_path=["A"], node_kind="section"),
        ]
        card = build_document_card(chunks, max_headings=60)
        assert card["num_chunks"] == 2

    def test_missing_node_kind_counts_as_leaf(self):
        chunks = [
            _chunk(document_id="d", title="T", heading_path=["A"]),  # no node_kind
            _chunk(document_id="d", title="T", heading_path=["B"]),
        ]
        card = build_document_card(chunks, max_headings=60)
        assert card["num_chunks"] == 2

    def test_section_node_headings_still_contribute(self):
        # Section nodes are excluded from num_chunks but their headings may
        # still enrich the heading list (in document order).
        chunks = [
            _chunk(document_id="d", title="T", heading_path=["Leaf H"], node_kind="chunk"),
            _chunk(
                document_id="d",
                title="T",
                heading_path=["Section Only H"],
                node_kind="section",
            ),
        ]
        card = build_document_card(chunks, max_headings=60)
        assert "Section Only H" in card["section_headings"]
        assert card["num_chunks"] == 1

    # ---- dict-tolerant accessor + determinism ---------------------------

    def test_accepts_plain_dict_chunks(self):
        chunks = [
            {"text": "body", "metadata": {"document_id": "d", "title": "T",
                                          "heading_path": ["H1"], "source": "s"}},
            {"text": "body", "metadata": {"document_id": "d", "title": "T",
                                          "heading_path": ["H2"], "source": "s"}},
        ]
        card = build_document_card(chunks, max_headings=60)
        assert card["document_id"] == "d"
        assert card["title"] == "T"
        assert card["source"] == "s"
        assert card["section_headings"] == ["H1", "H2"]
        assert card["num_chunks"] == 2

    def test_deterministic_same_input_same_output(self):
        chunks = [
            _chunk(document_id="d", title="T", source="s", heading_path=["A"]),
            _chunk(document_id="d", title="T", source="s", heading_path=["B"]),
        ]
        first = build_document_card(chunks, max_headings=60)
        second = build_document_card(chunks, max_headings=60)
        assert first == second

    def test_empty_chunks_returns_empty_card(self):
        card = build_document_card([], max_headings=60)
        assert card["document_id"] == ""
        assert card["title"] == ""
        assert card["source"] == ""
        assert card["section_headings"] == []
        assert card["card_text"] == ""
        assert card["summary"] == ""
        assert card["num_chunks"] == 0
