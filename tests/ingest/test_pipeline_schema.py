import hashlib
from pathlib import Path

import pytest

from src.common.utils import parse_json_object
from src.ingest.common.schemas import ProcessedChunk
from src.ingest.doc_processing.nodes.document_ingestion import document_ingestion_node
from src.ingest.doc_processing.nodes.multimodal_processing import multimodal_processing_node
from src.ingest.doc_processing.nodes.structure_detection import structure_detection_node
from src.ingest.embedding.nodes.chunking import chunking_node
from src.ingest.embedding.nodes.chunk_enrichment import chunk_enrichment_node
from src.ingest.support.docling import DoclingParseResult
from src.ingest.common.types import IngestionConfig, PIPELINE_NODE_NAMES
from src.ingest.impl import verify_core_design
from src.ingest.common.shared import extract_keywords_fallback, map_chunk_provenance


def test_parse_json_object_handles_fenced_payload():
    payload = "```json\n{\"summary\":\"ok\",\"keywords\":[\"timing\",\"dft\"]}\n```"
    parsed = parse_json_object(payload)
    assert parsed["summary"] == "ok"
    assert parsed["keywords"] == ["timing", "dft"]


def test_extract_keywords_fallback_returns_ranked_terms():
    text = "DFT scan chain timing timing timing clock clock setup hold hold hold hold"
    keywords = extract_keywords_fallback(text, max_keywords=3)
    assert keywords[0] == "hold"
    assert len(keywords) == 3


@pytest.mark.skip(reason="skip logic moved to pipeline orchestrator (pipeline/impl.py)")
def test_ingest_source_marks_unchanged_file_as_skip(tmp_path: Path):
    content = "clock domain crossing basics"
    path = tmp_path / "doc.txt"
    path.write_text(content, encoding="utf-8")
    existing_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    state = {
        "source_path": str(path),
        "source_name": path.name,
        "source_uri": path.resolve().as_uri(),
        "source_key": "local_fs:test",
        "source_id": "test",
        "connector": "local_fs",
        "source_version": "1",
        "content_hash": "",
        "existing_hash": existing_hash,
        "existing_source_uri": path.resolve().as_uri(),
        "should_skip": False,
        "errors": [],
        "raw_text": "",
        "cleaned_text": "",
        "chunks": [],
        "stored_count": 0,
        "metadata_summary": "",
        "metadata_keywords": [],
        "processing_log": [],
        "structure": {},
        "multimodal_notes": [],
        "refactored_text": "",
        "cross_references": [],
        "kg_triples": [],
        "runtime": type("R", (), {"config": IngestionConfig()})(),
    }
    result = document_ingestion_node(state)
    assert result["should_skip"] is True


def test_pipeline_exposes_15_named_nodes():
    assert len(PIPELINE_NODE_NAMES) == 15
    assert PIPELINE_NODE_NAMES[0] == "document_ingestion"
    assert PIPELINE_NODE_NAMES[-1] == "knowledge_graph_storage"


def test_verify_core_design_detects_invalid_kg_configuration():
    cfg = IngestionConfig(
        build_kg=False,
        enable_knowledge_graph_extraction=False,
        enable_knowledge_graph_storage=True,
    )
    report = verify_core_design(cfg)
    assert report.ok is False
    assert any(
        "knowledge_graph_storage requires build_kg=True" in error
        for error in report.errors
    )


def test_verify_core_design_requires_docling_model_when_enabled():
    cfg = IngestionConfig(enable_docling_parser=True, docling_model="")
    report = verify_core_design(cfg)
    assert report.ok is False
    assert any("docling parser requires a non-empty docling_model" in e for e in report.errors)


def test_verify_core_design_requires_multimodal_when_vision_enabled():
    cfg = IngestionConfig(
        enable_multimodal_processing=False,
        enable_vision_processing=True,
    )
    report = verify_core_design(cfg)
    assert report.ok is False
    assert any(
        "vision processing requires multimodal_processing to be enabled" in e
        for e in report.errors
    )


def test_map_chunk_provenance_exact_match():
    text = "Alpha section\n\nClock must remain below 800MHz.\n\nBeta section"
    provenance, _, _ = map_chunk_provenance(
        "Clock must remain below 800MHz.",
        original_text=text,
        refactored_text=text,
        original_cursor=0,
        refactored_cursor=0,
    )
    assert provenance["original_char_start"] >= 0
    assert provenance["refactored_char_start"] >= 0
    assert provenance["provenance_confidence"] >= 0.8


def test_chunk_enrichment_sets_source_fields_and_chunk_id():
    state = {
        "source_path": "/tmp/doc.txt",
        "source_name": "doc.txt",
        "source_uri": "file:///tmp/doc.txt",
        "source_key": "local_fs:1:99",
        "source_id": "1:99",
        "connector": "local_fs",
        "source_version": "1",
        "content_hash": "",
        "existing_hash": "",
        "existing_source_uri": "",
        "should_skip": False,
        "errors": [],
        "raw_text": "Clock must remain below 800MHz.",
        "cleaned_text": "Clock must remain below 800MHz.",
        "chunks": [ProcessedChunk(text="Clock must remain below 800MHz.", metadata={})],
        "stored_count": 0,
        "metadata_summary": "",
        "metadata_keywords": [],
        "processing_log": [],
        "structure": {},
        "multimodal_notes": [],
        "refactored_text": "",
        "cross_references": [],
        "kg_triples": [],
        "runtime": type("R", (), {"config": IngestionConfig()})(),
    }

    result = chunk_enrichment_node(state)
    chunk = result["chunks"][0]
    assert chunk.metadata["source"] == "doc.txt"
    assert chunk.metadata["source_key"] == "local_fs:1:99"
    assert chunk.metadata["chunk_id"]


def test_chunking_node_projects_heading_metadata(monkeypatch):
    monkeypatch.setattr(
        "src.ingest.embedding.nodes.chunking.chunk_markdown",
        lambda *args, **kwargs: [
            {
                "text": "Body text here.",
                "header_metadata": {"h1": "Title", "h2": "Clock Domain Crossing"},
            }
        ],
    )
    state = {
        "source_name": "doc.txt",
        "source_uri": "file:///tmp/doc.txt",
        "source_key": "local_fs:1:200",
        "source_id": "1:200",
        "connector": "local_fs",
        "source_version": "1",
        "raw_text": "# Title\n\n## Clock Domain Crossing\n\nBody text here.",
        "cleaned_text": "# Title\n\n## Clock Domain Crossing\n\nBody text here.",
        "refactored_text": "",
        "processing_log": [],
        "runtime": type(
            "R",
            (),
            {"config": IngestionConfig(semantic_chunking=False), "embedder": None},
        )(),
    }
    result = chunking_node(state)
    assert result["chunks"]
    assert result["chunks"][0].metadata["heading"] == "Clock Domain Crossing"
    assert result["chunks"][0].metadata["section_path"] == "Title > Clock Domain Crossing"


def test_structure_detection_uses_docling_output(monkeypatch):
    monkeypatch.setattr(
        "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
        lambda *_args, **_kwargs: DoclingParseResult(
            text_markdown="# Parsed Title\n\nBody from docling parser.",
            has_figures=True,
            figures=["Figure 1"],
            headings=["Parsed Title"],
            parser_model="docling-parse-v2",
        ),
    )
    state = {
        "source_path": "/tmp/doc.txt",
        "source_name": "doc.txt",
        "raw_text": "raw fallback content",
        "processing_log": [],
        "runtime": type(
            "R",
            (),
            {"config": IngestionConfig(enable_docling_parser=True, docling_strict=True)},
        )(),
    }
    result = structure_detection_node(state)
    assert result["raw_text"].startswith("# Parsed Title")
    assert result["structure"]["has_figures"] is True
    assert result["structure"]["heading_count"] == 1
    assert result["structure"]["docling_enabled"] is True


def test_structure_detection_docling_strict_fail_fast(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise RuntimeError("docling missing artifacts")

    monkeypatch.setattr("src.ingest.doc_processing.nodes.structure_detection.parse_with_docling", _boom)
    state = {
        "source_path": "/tmp/doc.txt",
        "source_name": "doc.txt",
        "raw_text": "raw fallback content",
        "processing_log": [],
        "runtime": type(
            "R",
            (),
            {"config": IngestionConfig(enable_docling_parser=True, docling_strict=True)},
        )(),
    }
    result = structure_detection_node(state)
    assert result["should_skip"] is True
    assert result["errors"]
    assert "docling_parse_failed" in result["errors"][0]


def test_multimodal_processing_uses_vision_notes(monkeypatch):
    monkeypatch.setattr(
        "src.ingest.doc_processing.nodes.multimodal_processing.generate_vision_notes",
        lambda *_args, **_kwargs: (["Figure 1: Clock waveform with setup/hold markers"], 1),
    )
    state = {
        "source_path": "/tmp/doc.md",
        "source_name": "doc.md",
        "raw_text": "![fig](image.png)",
        "structure": {"has_figures": True, "figures": ["Figure 1"]},
        "processing_log": [],
        "runtime": type(
            "R",
            (),
            {
                "config": IngestionConfig(
                    enable_multimodal_processing=True,
                    enable_vision_processing=True,
                )
            },
        )(),
    }
    result = multimodal_processing_node(state)
    assert result["multimodal_notes"][0].startswith("Figure 1:")
    assert result["structure"]["vision_described_count"] == 1


def test_docling_preflight_fails_with_bad_artifacts_path():
    try:
        from docling.document_converter import DocumentConverter  # noqa: F401
    except Exception:
        pytest.skip("docling unavailable in this environment (transformers version conflict)")

    from src.ingest.support.docling import ensure_docling_ready

    missing = "/tmp/definitely_missing_docling_artifacts_12345"
    try:
        ensure_docling_ready(
            parser_model="docling-parse-v2",
            artifacts_path=missing,
            auto_download=False,
        )
    except RuntimeError as exc:
        assert "Docling artifacts path is invalid" in str(exc)
        return
    assert False, "Expected RuntimeError for invalid artifacts path"
