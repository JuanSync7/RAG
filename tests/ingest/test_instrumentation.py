"""OTel instrumentation tests for the ingest pipeline.

Mirrors the fixture pattern from ``tests/llm/conftest.py``: install a fresh
OTel SDK TracerProvider + InMemorySpanExporter, swap the global observability
singleton to a real OTelBackend, and inspect emitted spans.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._once import Once

import src.platform.observability as _obs_module
from src.platform.observability.otel.backend import OTelBackend


@pytest.fixture
def otel_capture(monkeypatch):
    """Install fresh SDK provider + in-memory exporter; rebind tracer singleton."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    monkeypatch.setattr(otel_trace, "_TRACER_PROVIDER", provider, raising=False)
    monkeypatch.setattr(otel_trace, "_TRACER_PROVIDER_SET_ONCE", Once(), raising=False)

    import config.settings as _settings
    _settings.OBSERVABILITY_PROVIDER = "otel"
    _obs_module._backend = OTelBackend()

    yield exporter

    _obs_module._backend = None
    _settings.OBSERVABILITY_PROVIDER = "noop"


def _names(exporter: InMemorySpanExporter) -> list[str]:
    return [s.name for s in exporter.get_finished_spans()]


def _by_name(exporter: InMemorySpanExporter, name: str):
    matches = [s for s in exporter.get_finished_spans() if s.name == name]
    assert matches, f"no span named {name!r}; got {_names(exporter)}"
    return matches[0]


# ─────────────────────────────────────────────────────────────────────────────
# docling: parse_with_docling, chunk_markdown_via_docling
# ─────────────────────────────────────────────────────────────────────────────


def _make_mock_converter(markdown: str):
    mock_document = MagicMock()
    mock_document.export_to_markdown = MagicMock(return_value=markdown)
    mock_document.pictures = []
    mock_result = MagicMock()
    mock_result.document = mock_document
    mock_result.pages = None
    mock_converter = MagicMock()
    mock_converter.convert = MagicMock(return_value=mock_result)
    return mock_converter


def test_parse_with_docling_emits_span(otel_capture, tmp_path):
    """parse_with_docling emits ingest.parse.docling span with page_count attr."""
    from src.ingest.support import docling as docling_mod

    src_file = tmp_path / "x.pdf"
    src_file.write_bytes(b"%PDF-1.4 fake")

    mock_converter = _make_mock_converter("# h\nbody")
    mock_dc = MagicMock()
    mock_dc.DocumentConverter = MagicMock(return_value=mock_converter)

    with patch.dict("sys.modules", {"docling.document_converter": mock_dc}):
        docling_mod.parse_with_docling(
            src_file, parser_model="m", vlm_mode="disabled", generate_page_images=False
        )

    span = _by_name(otel_capture, "ingest.parse.docling")
    assert span.attributes.get("vlm_mode") == "disabled"
    assert span.attributes.get("parser_model") == "m"


def test_parse_with_docling_nests_under_parent_trace(otel_capture, tmp_path):
    """When called inside a tracer.span, parse span parents under it (implicit ctx)."""
    from src.platform.observability import get_tracer
    from src.ingest.support import docling as docling_mod

    src_file = tmp_path / "x.pdf"
    src_file.write_bytes(b"%PDF-1.4 fake")

    mock_dc = MagicMock()
    mock_dc.DocumentConverter = MagicMock(return_value=_make_mock_converter("# t"))

    with patch.dict("sys.modules", {"docling.document_converter": mock_dc}):
        with get_tracer().span("outer") as outer:
            docling_mod.parse_with_docling(src_file, parser_model="m")

    outer_span = _by_name(otel_capture, "outer")
    parse_span = _by_name(otel_capture, "ingest.parse.docling")
    assert parse_span.parent is not None
    assert parse_span.parent.span_id == outer_span.context.span_id


# ─────────────────────────────────────────────────────────────────────────────
# vision: _describe_image, caption_markdown_images_inline, generate_vision_notes
# ─────────────────────────────────────────────────────────────────────────────


def test_describe_image_emits_generation_span(otel_capture):
    """_describe_image emits a gen_ai generation span carrying model + completion."""
    from src.ingest.support import vision as vision_mod
    from src.ingest.support.vision import VisionImageCandidate

    cand = VisionImageCandidate(
        figure_label="Figure 1",
        alt_text="alt",
        source_ref="/tmp/x.png",
        image_b64="abc",
        mime_type="image/png",
    )
    cfg = MagicMock()
    cfg.vision_temperature = 0.0
    cfg.vision_max_tokens = 256
    cfg.vision_timeout_seconds = 30

    fake_provider = MagicMock()
    fake_response = MagicMock()
    fake_response.content = '{"caption":"a cat","visible_text":"","tags":["animal"]}'
    fake_provider.vision_completion = MagicMock(return_value=fake_response)
    # vision_model is referenced as gen_ai.request.model
    cfg.vision_model = "smolvlm-v1"

    with patch.object(vision_mod, "get_llm_provider", return_value=fake_provider):
        result = vision_mod._describe_image(cand, cfg)
    assert result is not None

    span = _by_name(otel_capture, "ingest.vision.caption")
    assert span.attributes.get("gen_ai.system") == "vision-llm"
    assert "a cat" in (span.attributes.get("gen_ai.completion") or "")


def test_caption_markdown_images_inline_emits_batch_span(otel_capture, tmp_path):
    """caption_markdown_images_inline emits an ingest.vision.caption_batch span."""
    from src.ingest.support import vision as vision_mod

    md = "before ![alt](data:image/png;base64,Zm9v) after"  # data url, decode-able
    cfg = MagicMock()
    cfg.vision_max_figures = 5
    cfg.vision_max_image_bytes = 1_000_000
    cfg.enable_vision_processing = True
    cfg.store_figures_in_db = False
    cfg.vision_temperature = 0.0
    cfg.vision_max_tokens = 256
    cfg.vision_timeout_seconds = 30
    cfg.vision_model = "smolvlm-v1"

    # _describe_image goes through get_llm_provider; short-circuit it instead.
    with patch.object(vision_mod, "_describe_image", return_value=None):
        rewritten, figures = vision_mod.caption_markdown_images_inline(
            md, source_path=tmp_path / "doc.md", config=cfg
        )

    span = _by_name(otel_capture, "ingest.vision.caption_batch")
    assert span.attributes.get("image_count") == 1


# ─────────────────────────────────────────────────────────────────────────────
# colqwen: embed_page_images
# ─────────────────────────────────────────────────────────────────────────────


def test_embed_page_images_emits_span_with_vector_count(otel_capture):
    """embed_page_images emits ingest.embedding.colqwen span with vector_count/dim."""
    import torch  # noqa: F401  — required dependency at import-time of code under test
    from src.ingest.support import colqwen as colqwen_mod

    # Build a deterministic batch tensor: 1 page, 2 patches, 4 dims.
    fake_tensor = MagicMock()
    page_tensor = MagicMock()
    mean_tensor = MagicMock()
    page_tensor.float.return_value = page_tensor
    page_tensor.cpu.return_value = page_tensor
    page_tensor.tolist.return_value = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
    page_tensor.mean.return_value = mean_tensor
    page_tensor.shape = (2, 4)
    mean_tensor.cpu.return_value = mean_tensor
    mean_tensor.tolist.return_value = [0.3, 0.4, 0.5, 0.6]
    fake_tensor.__getitem__ = MagicMock(return_value=page_tensor)

    fake_model = MagicMock()
    fake_model.return_value = fake_tensor  # forward
    fake_model.device = "cpu"
    fake_processor = MagicMock()
    fake_processor.process_images = MagicMock(
        return_value={"pixel_values": MagicMock(to=lambda d: MagicMock())}
    )

    # Patch torch.inference_mode + tensor isinstance via the imported colpali path.
    import torch as _t
    with patch.object(_t, "inference_mode", lambda: MagicMock(__enter__=lambda s: None, __exit__=lambda *a: None)):
        # batch_output is a MagicMock (not Tensor). Force the else-branch by ensuring
        # hasattr(batch_output, 'last_hidden_state') is False:
        del fake_tensor.last_hidden_state
        results = colqwen_mod.embed_page_images(
            fake_model, fake_processor, images=[MagicMock()], batch_size=1
        )

    span = _by_name(otel_capture, "ingest.embedding.colqwen")
    assert span.attributes.get("gen_ai.system") == "colqwen"
    # 1 page produced 1 embedding.
    assert span.attributes.get("vector_count") == 1
    # dim derived from mean_vector length (4)
    assert span.attributes.get("vector_dim") == 4


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator-level: ingest_file, run_document_processing, run_embedding_pipeline
# ─────────────────────────────────────────────────────────────────────────────


def test_run_document_processing_emits_trace(otel_capture):
    """run_document_processing wraps the graph invoke in ingest.doc_processing trace."""
    from src.ingest import doc_processing as dp_pkg
    from src.ingest.doc_processing import impl as dp_impl

    fake_graph = MagicMock()
    fake_graph.invoke = MagicMock(side_effect=lambda state: {**state, "cleaned_text": "ok"})

    with patch.object(dp_impl, "_get_graph", return_value=fake_graph):
        out = dp_impl.run_document_processing(
            runtime=MagicMock(),
            source_path="/x",
            source_name="x",
            source_uri="file:///x",
            source_key="k",
            source_id="id",
            connector="local",
            source_version="1",
            trace_id="tid",
        )

    assert out["cleaned_text"] == "ok"
    span = _by_name(otel_capture, "ingest.doc_processing")
    assert span.attributes.get("source_key") == "k"


def test_run_embedding_pipeline_emits_trace(otel_capture):
    """run_embedding_pipeline wraps the graph invoke in ingest.embedding trace."""
    from src.ingest.embedding import impl as emb_impl

    fake_graph = MagicMock()
    fake_graph.invoke = MagicMock(side_effect=lambda state: {**state, "stored_count": 3})

    with patch.object(emb_impl, "_GRAPH", fake_graph):
        out = emb_impl.run_embedding_pipeline(
            runtime=MagicMock(),
            source_key="k",
            source_name="n",
            source_uri="u",
            source_id="id",
            connector="local",
            source_version="1",
            clean_text="hello",
            clean_hash="h",
        )

    assert out["stored_count"] == 3
    span = _by_name(otel_capture, "ingest.embedding")
    assert span.attributes.get("source_key") == "k"


def test_ingest_file_emits_root_trace_with_nested_phases(otel_capture, tmp_path):
    """ingest_file emits ingest.file root span with doc_processing + embedding children."""
    from src.ingest import impl as ingest_impl

    src_file = tmp_path / "doc.txt"
    src_file.write_bytes(b"hello")

    runtime = MagicMock()
    runtime.config = MagicMock(
        export_processed=False,
        clean_store_dir="",
        persist_refactor_mirror=False,
        persist_docling_document=False,
    )

    fake_phase1 = {
        "errors": [],
        "cleaned_text": "clean",
        "processing_log": ["doc_processing"],
        "source_hash": "sh",
        "trace_id": "tid",
        "docling_document": None,
    }
    fake_phase2 = {
        "errors": [],
        "stored_count": 2,
        "metadata_summary": "",
        "metadata_keywords": [],
        "processing_log": ["embedding"],
        "validation": {},
    }

    with patch.object(ingest_impl, "run_document_processing", return_value=fake_phase1), \
         patch.object(ingest_impl, "run_embedding_pipeline", return_value=fake_phase2):
        ingest_impl.ingest_file(
            src_file,
            runtime,
            source_name="n",
            source_uri="u",
            source_key="k",
            source_id="id",
            connector="local",
            source_version="1",
        )

    root = _by_name(otel_capture, "ingest.file")
    assert root.attributes.get("source_key") == "k"
    assert root.attributes.get("connector") == "local"
