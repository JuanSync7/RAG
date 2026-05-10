# @summary
# Phase D white-box tests for Docling-native chunking pipeline.
# Covers: DoclingParseResult.docling_document field, parse_with_docling vlm_mode dispatch,
#   warmup_docling_models with_smolvlm flag, and structure_detection_node
#   docling_document propagation + docling_document_available routing flag.
# Exports: TestDoclingParseResult, TestParseWithDoclingHappyPath,
#          TestParseWithDoclingErrorPaths, TestParseWithDoclingBoundary,
#          TestWarmupDoclingModels, TestStructureDetectionDoclingPropagation,
#          TestStructureDetectionDoclingFlag, TestStructureDetectionBoundaryConditions
# Deps: src.ingest.support.docling, src.ingest.doc_processing.nodes.structure_detection,
#       src.ingest.common.types, unittest.mock, pytest
# @end-summary

"""Phase D tests for the Docling-native chunking pipeline.

Tests cover two modules:
  - ``src.ingest.support.docling`` — parse_with_docling vlm_mode dispatch and
    DoclingParseResult.docling_document field behaviour.
  - ``src.ingest.doc_processing.nodes.structure_detection`` — DoclingDocument
    propagation into pipeline state and the docling_document_available routing flag.

All Docling library imports are mocked so these tests run in environments that
do not have Docling installed.

Implementation note: ``parse_with_docling`` uses lazy imports inside the function
body (``from docling.document_converter import DocumentConverter``).  Patching
``src.ingest.support.docling.DocumentConverter`` does not work because the name
is never bound at module level.  The correct approach is to inject mock modules
into ``sys.modules`` before calling the function so that the lazy import resolves
to the mock.  The ``_docling_sys_modules`` context manager helper does this.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from src.ingest.common.types import IngestionConfig, Runtime
from src.ingest.doc_processing.nodes.structure_detection import structure_detection_node
from src.ingest.support.docling import DoclingParseResult


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_mock_docling_doc(markdown: str = "# Heading\n\nParagraph.", pictures=None):
    """Build a minimal mock DoclingDocument.

    Returns:
        mock_docling_doc with ``export_to_markdown``, ``pictures``, and
        ``model_dump_json`` configured.
    """
    mock_doc = MagicMock()
    mock_doc.export_to_markdown.return_value = markdown
    mock_doc.pictures = pictures if pictures is not None else []
    mock_doc.model_dump_json.return_value = '{"body": {"children": []}}'
    return mock_doc


def _make_mock_converter_result(mock_doc):
    """Build a mock convert() result that returns *mock_doc* as .document."""
    mock_result = MagicMock()
    mock_result.document = mock_doc
    return mock_result


def _make_mock_converter(markdown: str = "# Heading\n\nParagraph.", pictures=None):
    """Build a paired (converter_instance, docling_doc) mock.

    Returns:
        Tuple of (mock_converter_instance, mock_docling_doc).
    """
    mock_doc = _make_mock_docling_doc(markdown=markdown, pictures=pictures)
    mock_result = _make_mock_converter_result(mock_doc)
    mock_converter = MagicMock()
    mock_converter.convert.return_value = mock_result
    return mock_converter, mock_doc


@contextmanager
def _docling_sys_modules(
    converter_instance=None,
    pipeline_options_module=None,
    base_models_module=None,
    pdf_format_option_class=None,
) -> Generator[MagicMock, None, None]:
    """Inject minimal ``docling.*`` mocks into ``sys.modules``.

    ``parse_with_docling`` does lazy imports; patching at module level does not
    work.  This helper injects mocks for all sub-modules that the function may
    import so that the ``from docling.xxx import YYY`` calls succeed.

    Args:
        converter_instance: The mock instance that ``DocumentConverter(...)``
            should return.  When None, a fresh MagicMock is used.
        pipeline_options_module: Optional pre-built mock for
            ``docling.datamodel.pipeline_options``.  When None, a default is
            provided that does NOT raise on import.
        base_models_module: Optional pre-built mock for
            ``docling.datamodel.base_models``.
        pdf_format_option_class: Optional class mock for ``PdfFormatOption``.

    Yields:
        The ``MockDocumentConverter`` class mock so callers can inspect calls.
    """
    if converter_instance is None:
        converter_instance, _ = _make_mock_converter()

    MockDocumentConverter = MagicMock(return_value=converter_instance)

    # Build the docling.document_converter module mock.
    mock_dc_module = MagicMock()
    mock_dc_module.DocumentConverter = MockDocumentConverter
    if pdf_format_option_class is not None:
        mock_dc_module.PdfFormatOption = pdf_format_option_class
    else:
        mock_dc_module.PdfFormatOption = MagicMock()

    # Build pipeline_options mock (safe default).
    if pipeline_options_module is None:
        pipeline_options_module = MagicMock()
        pipeline_options_module.PdfPipelineOptions = MagicMock(return_value=MagicMock())
        pipeline_options_module.PictureDescriptionVlmEngineOptions = MagicMock()

    # Build base_models mock (safe default).
    if base_models_module is None:
        base_models_module = MagicMock()
        base_models_module.InputFormat = MagicMock()
        base_models_module.InputFormat.PDF = "PDF"

    injected = {
        "docling": MagicMock(),
        "docling.document_converter": mock_dc_module,
        "docling.datamodel": MagicMock(),
        "docling.datamodel.pipeline_options": pipeline_options_module,
        "docling.datamodel.base_models": base_models_module,
    }

    original = {k: sys.modules.get(k) for k in injected}
    sys.modules.update(injected)
    try:
        yield MockDocumentConverter
    finally:
        for k, v in original.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _make_state(
    raw_text: str = "",
    source_path: str = "/tmp/test.pdf",
    source_name: str = "test.pdf",
    enable_docling: bool = True,
    docling_strict: bool = False,
    vlm_mode: str = "disabled",
    **config_overrides,
) -> dict:
    """Build a minimal DocumentProcessingState for structure_detection_node tests."""
    config = IngestionConfig(
        enable_docling_parser=enable_docling,
        docling_strict=docling_strict,
        vlm_mode=vlm_mode,
        **config_overrides,
    )
    runtime = Runtime(
        config=config,
        embedder=MagicMock(),
        weaviate_client=MagicMock(),
        kg_builder=None,
    )
    return {
        "raw_text": raw_text,
        "source_path": source_path,
        "source_name": source_name,
        "runtime": runtime,
        "errors": [],
        "processing_log": [],
    }


# ---------------------------------------------------------------------------
# Module 1: src.ingest.support.docling
# ---------------------------------------------------------------------------


class TestDoclingParseResult:
    """Tests for the DoclingParseResult dataclass docling_document field."""

    def test_docling_parse_result_has_docling_document_field(self):
        """DoclingParseResult must accept docling_document as a keyword argument."""
        mock_doc = MagicMock()
        result = DoclingParseResult(
            text_markdown="# Hello",
            has_figures=False,
            figures=[],
            headings=["Hello"],
            parser_model="docling-parse-v2",
            docling_document=mock_doc,
        )
        assert result.docling_document is mock_doc

    def test_docling_parse_result_docling_document_defaults_to_none(self):
        """docling_document defaults to None when not supplied."""
        result = DoclingParseResult(
            text_markdown="text",
            has_figures=False,
            figures=[],
            headings=[],
            parser_model="docling-parse-v2",
        )
        assert result.docling_document is None

    def test_docling_parse_result_docling_document_none_in_error_recovery(self):
        """Explicitly setting docling_document=None is valid (error recovery path)."""
        result = DoclingParseResult(
            text_markdown="fallback",
            has_figures=False,
            figures=[],
            headings=[],
            parser_model="docling-parse-v2",
            docling_document=None,
        )
        assert result.docling_document is None

    def test_docling_parse_result_accepts_arbitrary_object_as_document(self):
        """docling_document field is typed Any — any object is accepted."""
        sentinel = object()
        result = DoclingParseResult(
            text_markdown="text",
            has_figures=False,
            figures=[],
            headings=[],
            parser_model="x",
            docling_document=sentinel,
        )
        assert result.docling_document is sentinel


class TestParseWithDoclingHappyPath:
    """Happy path tests for parse_with_docling — vlm_mode dispatch and result population."""

    def test_disabled_vlm_mode_returns_docling_document(self, tmp_path):
        """vlm_mode='disabled' — DocumentConverter built without picture description;
        returned DoclingParseResult has docling_document set to result.document."""
        source_file = tmp_path / "doc.pdf"
        source_file.write_bytes(b"%PDF-1.4 fake")

        mock_converter, mock_doc = _make_mock_converter()

        with _docling_sys_modules(converter_instance=mock_converter) as MockConverter:
            from src.ingest.support.docling import parse_with_docling

            result = parse_with_docling(
                source_file,
                parser_model="docling-parse-v2",
                vlm_mode="disabled",
            )

        assert result.docling_document is mock_doc
        # For disabled mode, DocumentConverter must be created without format_options.
        MockConverter.assert_called_once()
        _, kwargs = MockConverter.call_args
        assert "format_options" not in kwargs

    def test_external_vlm_mode_returns_docling_document(self, tmp_path):
        """vlm_mode='external' — same as disabled at parse time; docling_document populated.

        External VLM enrichment happens post-chunking; parse_with_docling must not
        enable picture description when vlm_mode='external'.
        """
        source_file = tmp_path / "doc.pdf"
        source_file.write_bytes(b"%PDF-1.4 fake")

        mock_converter, mock_doc = _make_mock_converter()

        with _docling_sys_modules(converter_instance=mock_converter) as MockConverter:
            from src.ingest.support.docling import parse_with_docling

            result = parse_with_docling(
                source_file,
                parser_model="docling-parse-v2",
                vlm_mode="external",
            )

        assert result.docling_document is mock_doc
        MockConverter.assert_called_once()
        _, kwargs = MockConverter.call_args
        assert "format_options" not in kwargs

    def test_builtin_vlm_mode_sets_do_picture_description_true(self, tmp_path):
        """vlm_mode='builtin' — PdfPipelineOptions.do_picture_description set to True
        and SmolVLM preset applied before DocumentConverter is constructed."""
        source_file = tmp_path / "doc.pdf"
        source_file.write_bytes(b"%PDF-1.4 fake")

        mock_converter, mock_doc = _make_mock_converter()

        # Build controlled pipeline_options mock to verify attribute assignments.
        mock_pipeline_options = MagicMock()
        MockPdfPipelineOptions = MagicMock(return_value=mock_pipeline_options)

        mock_smolvlm_result = MagicMock()
        MockPictureDescriptionVlmEngineOptions = MagicMock()
        MockPictureDescriptionVlmEngineOptions.from_preset.return_value = mock_smolvlm_result

        mock_pipeline_options_module = MagicMock()
        mock_pipeline_options_module.PdfPipelineOptions = MockPdfPipelineOptions
        mock_pipeline_options_module.PictureDescriptionVlmEngineOptions = (
            MockPictureDescriptionVlmEngineOptions
        )

        mock_pdf_format_option_instance = MagicMock()
        MockPdfFormatOption = MagicMock(return_value=mock_pdf_format_option_instance)

        with _docling_sys_modules(
            converter_instance=mock_converter,
            pipeline_options_module=mock_pipeline_options_module,
            pdf_format_option_class=MockPdfFormatOption,
        ) as MockConverter:
            from src.ingest.support.docling import parse_with_docling

            result = parse_with_docling(
                source_file,
                parser_model="docling-parse-v2",
                vlm_mode="builtin",
            )

        # do_picture_description must have been set to True on the options object.
        assert mock_pipeline_options.do_picture_description is True
        # SmolVLM preset must have been applied via from_preset("smolvlm").
        MockPictureDescriptionVlmEngineOptions.from_preset.assert_called_once_with("smolvlm")
        assert mock_pipeline_options.picture_description_options is mock_smolvlm_result
        # format_options must have been passed to DocumentConverter.
        MockConverter.assert_called_once()
        _, kwargs = MockConverter.call_args
        assert "format_options" in kwargs
        # The result must still have docling_document populated.
        assert result.docling_document is mock_doc

    def test_parse_result_docling_document_is_never_none_on_success(self, tmp_path):
        """On a successful parse for any non-builtin vlm_mode, docling_document is
        always the document object returned by convert()."""
        source_file = tmp_path / "doc.pdf"
        source_file.write_bytes(b"%PDF-1.4 fake")

        for vlm_mode in ("disabled", "external"):
            mock_converter, mock_doc = _make_mock_converter()
            with _docling_sys_modules(converter_instance=mock_converter):
                from src.ingest.support.docling import parse_with_docling

                result = parse_with_docling(
                    source_file,
                    parser_model="docling-parse-v2",
                    vlm_mode=vlm_mode,
                )
            assert result.docling_document is not None, (
                f"docling_document must not be None on success (vlm_mode={vlm_mode!r})"
            )

    def test_parse_result_docling_document_is_exact_result_document(self, tmp_path):
        """docling_document must be the exact object from result.document (identity, not copy)."""
        source_file = tmp_path / "doc.pdf"
        source_file.write_bytes(b"%PDF-1.4 fake")

        mock_converter, mock_doc = _make_mock_converter()

        with _docling_sys_modules(converter_instance=mock_converter):
            from src.ingest.support.docling import parse_with_docling

            result = parse_with_docling(
                source_file,
                parser_model="docling-parse-v2",
                vlm_mode="disabled",
            )

        assert result.docling_document is mock_doc

    def test_parse_result_other_fields_populated_on_success(self, tmp_path):
        """text_markdown, has_figures, figures, headings, parser_model are all set."""
        source_file = tmp_path / "doc.pdf"
        source_file.write_bytes(b"%PDF-1.4 fake")

        mock_figure = MagicMock()
        mock_converter, _ = _make_mock_converter(
            markdown="# Introduction\n\nBody text.",
            pictures=[mock_figure],
        )

        with _docling_sys_modules(converter_instance=mock_converter):
            from src.ingest.support.docling import parse_with_docling

            result = parse_with_docling(
                source_file,
                parser_model="my-model",
                vlm_mode="disabled",
            )

        assert result.text_markdown == "# Introduction\n\nBody text."
        assert result.has_figures is True
        assert result.figures == ["Figure 1"]
        assert result.headings == ["Introduction"]
        assert result.parser_model == "my-model"

    def test_artifacts_path_accepted_but_not_forwarded(self, tmp_path):
        """parse_with_docling accepts artifacts_path for caller compat but no
        longer forwards it to DocumentConverter (newer Docling versions
        removed the constructor kwarg; model location is now controlled by
        warmup_docling_models / HF cache). See src/ingest/support/docling.py
        line 307-309 for the explicit drop comment."""
        source_file = tmp_path / "doc.pdf"
        source_file.write_bytes(b"%PDF-1.4 fake")
        artifacts = str(tmp_path / "artifacts")

        mock_converter, _ = _make_mock_converter()

        with _docling_sys_modules(converter_instance=mock_converter) as MockConverter:
            from src.ingest.support.docling import parse_with_docling

            parse_with_docling(
                source_file,
                parser_model="docling-parse-v2",
                artifacts_path=artifacts,
                vlm_mode="disabled",
            )

        _, kwargs = MockConverter.call_args
        assert "artifacts_path" not in kwargs

    def test_no_artifacts_path_does_not_pass_key_to_converter(self, tmp_path):
        """When artifacts_path is empty, artifacts_path is NOT passed to DocumentConverter."""
        source_file = tmp_path / "doc.pdf"
        source_file.write_bytes(b"%PDF-1.4 fake")

        mock_converter, _ = _make_mock_converter()

        with _docling_sys_modules(converter_instance=mock_converter) as MockConverter:
            from src.ingest.support.docling import parse_with_docling

            parse_with_docling(
                source_file,
                parser_model="docling-parse-v2",
                vlm_mode="disabled",
            )

        _, kwargs = MockConverter.call_args
        assert "artifacts_path" not in kwargs


class TestParseWithDoclingErrorPaths:
    """Error path tests for parse_with_docling."""

    def test_converter_failure_raises_runtime_error(self, tmp_path):
        """When DocumentConverter.convert() raises, parse_with_docling raises RuntimeError."""
        source_file = tmp_path / "bad.pdf"
        source_file.write_bytes(b"corrupted")

        mock_converter = MagicMock()
        mock_converter.convert.side_effect = Exception("corrupt PDF")

        with _docling_sys_modules(converter_instance=mock_converter):
            from src.ingest.support.docling import parse_with_docling

            with pytest.raises(RuntimeError, match="Docling conversion failed"):
                parse_with_docling(
                    source_file,
                    parser_model="docling-parse-v2",
                    vlm_mode="disabled",
                )

    def test_runtime_error_message_includes_source_path(self, tmp_path):
        """RuntimeError message must reference the source path for diagnostics."""
        source_file = tmp_path / "corrupt.pdf"
        source_file.write_bytes(b"bad")

        mock_converter = MagicMock()
        mock_converter.convert.side_effect = Exception("decode error")

        with _docling_sys_modules(converter_instance=mock_converter):
            from src.ingest.support.docling import parse_with_docling

            with pytest.raises(RuntimeError) as exc_info:
                parse_with_docling(
                    source_file,
                    parser_model="docling-parse-v2",
                    vlm_mode="disabled",
                )

        assert str(source_file) in str(exc_info.value)

    def test_missing_document_attribute_raises_runtime_error(self, tmp_path):
        """When result.document is None, RuntimeError is raised (not AttributeError)."""
        source_file = tmp_path / "doc.pdf"
        source_file.write_bytes(b"%PDF")

        mock_result = MagicMock()
        mock_result.document = None
        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        with _docling_sys_modules(converter_instance=mock_converter):
            from src.ingest.support.docling import parse_with_docling

            with pytest.raises(RuntimeError):
                parse_with_docling(
                    source_file,
                    parser_model="docling-parse-v2",
                    vlm_mode="disabled",
                )

    def test_empty_markdown_raises_runtime_error(self, tmp_path):
        """When Docling returns empty markdown, RuntimeError is raised."""
        source_file = tmp_path / "empty.pdf"
        source_file.write_bytes(b"%PDF")

        mock_converter, _ = _make_mock_converter(markdown="")

        with _docling_sys_modules(converter_instance=mock_converter):
            from src.ingest.support.docling import parse_with_docling

            with pytest.raises(RuntimeError, match="empty markdown"):
                parse_with_docling(
                    source_file,
                    parser_model="docling-parse-v2",
                    vlm_mode="disabled",
                )

    def test_builtin_smolvlm_failure_is_caught_and_parse_continues(self, tmp_path):
        """When SmolVLM setup fails during vlm_mode='builtin', a warning is logged
        and parse proceeds — does not raise.

        The test simulates SmolVLM being unavailable by making PdfPipelineOptions
        raise ImportError; the production code catches this and falls back to
        constructing DocumentConverter without format_options.
        """
        source_file = tmp_path / "doc.pdf"
        source_file.write_bytes(b"%PDF-1.4 fake")

        mock_converter, mock_doc = _make_mock_converter()

        # Make PdfPipelineOptions raise so the builtin VLM setup fails.
        broken_pipeline_module = MagicMock()
        broken_pipeline_module.PdfPipelineOptions.side_effect = ImportError(
            "smolvlm not available"
        )

        with _docling_sys_modules(
            converter_instance=mock_converter,
            pipeline_options_module=broken_pipeline_module,
        ):
            from src.ingest.support.docling import parse_with_docling

            # Must not raise — warning is logged, parse continues.
            result = parse_with_docling(
                source_file,
                parser_model="docling-parse-v2",
                vlm_mode="builtin",
            )

        # docling_document must still be populated.
        assert result.docling_document is mock_doc


class TestParseWithDoclingBoundary:
    """Boundary condition tests for parse_with_docling."""

    def test_vlm_mode_disabled_does_not_set_format_options(self, tmp_path):
        """vlm_mode='disabled' → converter constructed without format_options."""
        source_file = tmp_path / "doc.pdf"
        source_file.write_bytes(b"%PDF")

        mock_converter, _ = _make_mock_converter()

        with _docling_sys_modules(converter_instance=mock_converter) as MockConverter:
            from src.ingest.support.docling import parse_with_docling

            parse_with_docling(source_file, parser_model="m", vlm_mode="disabled")

        _, kwargs = MockConverter.call_args
        assert "format_options" not in kwargs

    def test_vlm_mode_external_does_not_set_format_options(self, tmp_path):
        """vlm_mode='external' → converter has no format_options (enrichment is post-chunking)."""
        source_file = tmp_path / "doc.pdf"
        source_file.write_bytes(b"%PDF")

        mock_converter, _ = _make_mock_converter()

        with _docling_sys_modules(converter_instance=mock_converter) as MockConverter:
            from src.ingest.support.docling import parse_with_docling

            parse_with_docling(source_file, parser_model="m", vlm_mode="external")

        _, kwargs = MockConverter.call_args
        assert "format_options" not in kwargs

    def test_no_pictures_produces_empty_figures_list(self, tmp_path):
        """When document.pictures is empty, figures is [] and has_figures is False."""
        source_file = tmp_path / "doc.pdf"
        source_file.write_bytes(b"%PDF")

        mock_converter, _ = _make_mock_converter(pictures=[])

        with _docling_sys_modules(converter_instance=mock_converter):
            from src.ingest.support.docling import parse_with_docling

            result = parse_with_docling(source_file, parser_model="m", vlm_mode="disabled")

        assert result.figures == []
        assert result.has_figures is False

    def test_multiple_pictures_produces_figure_labels(self, tmp_path):
        """Multiple pictures → figures list has one 'Figure N' label per picture."""
        source_file = tmp_path / "doc.pdf"
        source_file.write_bytes(b"%PDF")

        mock_converter, _ = _make_mock_converter(
            pictures=[MagicMock(), MagicMock(), MagicMock()],
        )

        with _docling_sys_modules(converter_instance=mock_converter):
            from src.ingest.support.docling import parse_with_docling

            result = parse_with_docling(source_file, parser_model="m", vlm_mode="disabled")

        assert result.figures == ["Figure 1", "Figure 2", "Figure 3"]
        assert result.has_figures is True

    def test_markdown_headings_extracted_from_output(self, tmp_path):
        """Heading strings are extracted from Docling markdown output."""
        source_file = tmp_path / "doc.pdf"
        source_file.write_bytes(b"%PDF")

        mock_converter, _ = _make_mock_converter(
            markdown="# Alpha\n\nText.\n\n## Beta\n\nMore text."
        )

        with _docling_sys_modules(converter_instance=mock_converter):
            from src.ingest.support.docling import parse_with_docling

            result = parse_with_docling(source_file, parser_model="m", vlm_mode="disabled")

        assert "Alpha" in result.headings
        assert "Beta" in result.headings


class TestWarmupDoclingModels:
    """Tests for warmup_docling_models with_smolvlm flag.

    These tests verify the flag is forwarded correctly to ``download_models``.
    They patch ``sys.modules`` to inject a fake ``docling.utils.model_downloader``
    module because docling is not installed in the test environment.
    """

    def _make_warmup_modules(self, fake_root):
        """Build sys.modules entries needed by warmup_docling_models.

        Creates stub model directories under fake_root to satisfy post-download
        validation checks.
        """
        (fake_root / "layout").mkdir(parents=True, exist_ok=True)
        (fake_root / "tableformer").mkdir(parents=True, exist_ok=True)

        mock_download = MagicMock(return_value=fake_root)

        mock_layout_options_instance = MagicMock()
        mock_layout_options_instance.model_spec.model_repo_folder = "layout"
        MockLayoutOptions = MagicMock(return_value=mock_layout_options_instance)

        mock_table_model = MagicMock()
        mock_table_model._model_repo_folder = "tableformer"

        return mock_download, MockLayoutOptions, mock_table_model

    def test_warmup_without_smolvlm_passes_false_to_download_models(self, tmp_path):
        """warmup_docling_models(with_smolvlm=False) → download_models called with
        with_smolvlm=False."""
        mock_download, MockLayoutOptions, mock_table_model = self._make_warmup_modules(tmp_path)

        injected = {
            "docling": MagicMock(),
            "docling.datamodel": MagicMock(),
            "docling.datamodel.pipeline_options": MagicMock(LayoutOptions=MockLayoutOptions),
            "docling.models": MagicMock(),
            "docling.models.stages": MagicMock(),
            "docling.models.stages.table_structure": MagicMock(),
            "docling.models.stages.table_structure.table_structure_model": MagicMock(
                TableStructureModel=mock_table_model
            ),
            "docling.utils": MagicMock(),
            "docling.utils.model_downloader": MagicMock(download_models=mock_download),
        }
        original = {k: sys.modules.get(k) for k in injected}
        sys.modules.update(injected)
        try:
            import importlib
            import src.ingest.support.docling as docling_mod
            importlib.reload(docling_mod)
            docling_mod.warmup_docling_models(with_smolvlm=False)
        finally:
            for k, v in original.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

        mock_download.assert_called_once()
        _, call_kwargs = mock_download.call_args
        assert call_kwargs.get("with_smolvlm") is False

    def test_warmup_with_smolvlm_passes_true_to_download_models(self, tmp_path):
        """warmup_docling_models(with_smolvlm=True) → download_models called with
        with_smolvlm=True."""
        mock_download, MockLayoutOptions, mock_table_model = self._make_warmup_modules(tmp_path)

        injected = {
            "docling": MagicMock(),
            "docling.datamodel": MagicMock(),
            "docling.datamodel.pipeline_options": MagicMock(LayoutOptions=MockLayoutOptions),
            "docling.models": MagicMock(),
            "docling.models.stages": MagicMock(),
            "docling.models.stages.table_structure": MagicMock(),
            "docling.models.stages.table_structure.table_structure_model": MagicMock(
                TableStructureModel=mock_table_model
            ),
            "docling.utils": MagicMock(),
            "docling.utils.model_downloader": MagicMock(download_models=mock_download),
        }
        original = {k: sys.modules.get(k) for k in injected}
        sys.modules.update(injected)
        try:
            import importlib
            import src.ingest.support.docling as docling_mod
            importlib.reload(docling_mod)
            docling_mod.warmup_docling_models(with_smolvlm=True)
        finally:
            for k, v in original.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

        mock_download.assert_called_once()
        _, call_kwargs = mock_download.call_args
        assert call_kwargs.get("with_smolvlm") is True


# ---------------------------------------------------------------------------
# Module 2: src.ingest.doc_processing.nodes.structure_detection
# ---------------------------------------------------------------------------


class TestStructureDetectionDoclingPropagation:
    """Tests for docling_document propagation into the state update."""

    def test_docling_document_present_in_state_update_on_success(self):
        """When parse_with_docling succeeds, the state update must contain
        'docling_document' set to parsed.docling_document."""
        mock_doc = MagicMock()
        mock_result = MagicMock()
        mock_result.docling_document = mock_doc
        mock_result.text_markdown = "# Heading\n\nParagraph text."
        mock_result.has_figures = False
        mock_result.figures = []
        mock_result.headings = ["Heading"]
        mock_result.parser_model = "docling-parse-v2"

        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            return_value=mock_result,
        ):
            state = _make_state(enable_docling=True)
            update = structure_detection_node(state)

        assert "docling_document" in update
        assert update["docling_document"] is mock_doc

    def test_docling_document_identity_match(self):
        """docling_document in the update is the exact same object as
        parsed.docling_document (identity, not a copy)."""
        sentinel = object()
        mock_result = MagicMock()
        mock_result.docling_document = sentinel
        mock_result.text_markdown = "text"
        mock_result.figures = []
        mock_result.headings = []

        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            return_value=mock_result,
        ):
            state = _make_state(enable_docling=True)
            update = structure_detection_node(state)

        assert update["docling_document"] is sentinel

    def test_docling_document_absent_when_docling_disabled(self):
        """When enable_docling_parser=False, 'docling_document' key must NOT be
        present in the state update — not even as None."""
        state = _make_state(enable_docling=False, raw_text="Some text here.")
        update = structure_detection_node(state)

        assert "docling_document" not in update

    def test_docling_document_absent_on_non_strict_failure(self):
        """When parse_with_docling raises and docling_strict=False, the fallback
        path must not include 'docling_document' in the state update."""
        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            side_effect=RuntimeError("parse error"),
        ):
            state = _make_state(enable_docling=True, docling_strict=False, raw_text="plain")
            update = structure_detection_node(state)

        assert "docling_document" not in update

    def test_docling_document_absent_on_strict_failure(self):
        """When parse_with_docling raises and docling_strict=True, the error payload
        must not include 'docling_document'."""
        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            side_effect=RuntimeError("parse error"),
        ):
            state = _make_state(enable_docling=True, docling_strict=True)
            update = structure_detection_node(state)

        assert "docling_document" not in update
        assert update.get("should_skip") is True

    def test_docling_document_key_absent_not_set_to_none_on_failure(self):
        """On non-strict fallback, 'docling_document' must be absent — not present as None.

        Callers use ``state.get('docling_document')`` to distinguish between
        'available' and 'not set'; setting it to None would be incorrect.
        """
        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            side_effect=RuntimeError("parse error"),
        ):
            state = _make_state(enable_docling=True, docling_strict=False, raw_text="text")
            update = structure_detection_node(state)

        # Must not be present at all, not even as None.
        assert "docling_document" not in update


class TestStructureDetectionDoclingFlag:
    """Tests for structure['docling_document_available'] routing flag."""

    def test_flag_true_on_successful_docling_parse(self):
        """On successful Docling parse, docling_document_available must be True."""
        mock_result = MagicMock()
        mock_result.docling_document = MagicMock()
        mock_result.text_markdown = "# H1\n\nText."
        mock_result.figures = []
        mock_result.headings = ["H1"]

        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            return_value=mock_result,
        ):
            state = _make_state(enable_docling=True)
            update = structure_detection_node(state)

        assert update["structure"]["docling_document_available"] is True

    def test_flag_is_bool_true_not_truthy(self):
        """Flag value must be the Python bool True, not merely a truthy value."""
        mock_result = MagicMock()
        mock_result.docling_document = MagicMock()
        mock_result.text_markdown = "text"
        mock_result.figures = []
        mock_result.headings = []

        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            return_value=mock_result,
        ):
            state = _make_state(enable_docling=True)
            update = structure_detection_node(state)

        flag = update["structure"]["docling_document_available"]
        assert flag is True
        assert isinstance(flag, bool)

    def test_flag_false_when_docling_disabled(self):
        """When enable_docling_parser=False, docling_document_available must be False."""
        state = _make_state(enable_docling=False, raw_text="Text.")
        update = structure_detection_node(state)

        assert update["structure"]["docling_document_available"] is False

    def test_flag_is_bool_false_not_falsy_when_disabled(self):
        """Flag value must be exactly False (bool), not merely falsy (e.g., None, 0)."""
        state = _make_state(enable_docling=False)
        update = structure_detection_node(state)

        flag = update["structure"]["docling_document_available"]
        assert flag is False
        assert isinstance(flag, bool)

    def test_flag_false_on_non_strict_fallback(self):
        """When Docling parse fails non-strict, docling_document_available must be False."""
        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            side_effect=RuntimeError("parse failed"),
        ):
            state = _make_state(
                enable_docling=True,
                docling_strict=False,
                raw_text="See Figure 1.",
            )
            update = structure_detection_node(state)

        assert update["structure"]["docling_document_available"] is False

    def test_structure_dict_includes_docling_document_available_key(self):
        """structure dict must always contain 'docling_document_available' key,
        regardless of whether Docling is enabled or parse succeeded."""
        # Case 1: Docling disabled.
        state = _make_state(enable_docling=False)
        update = structure_detection_node(state)
        assert "docling_document_available" in update["structure"]

        # Case 2: Docling enabled and succeeds.
        mock_result = MagicMock()
        mock_result.docling_document = MagicMock()
        mock_result.text_markdown = "text"
        mock_result.figures = []
        mock_result.headings = []

        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            return_value=mock_result,
        ):
            state = _make_state(enable_docling=True)
            update = structure_detection_node(state)
        assert "docling_document_available" in update["structure"]

        # Case 3: Docling enabled, parse fails non-strict.
        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            side_effect=RuntimeError("fail"),
        ):
            state = _make_state(enable_docling=True, docling_strict=False, raw_text="text")
            update = structure_detection_node(state)
        assert "docling_document_available" in update["structure"]


class TestStructureDetectionBoundaryConditions:
    """Boundary condition and integration point tests for structure_detection_node."""

    def test_vlm_mode_forwarded_to_parse_with_docling(self):
        """The node must pass config.vlm_mode to parse_with_docling as vlm_mode kwarg."""
        mock_result = MagicMock()
        mock_result.docling_document = MagicMock()
        mock_result.text_markdown = "text"
        mock_result.figures = []
        mock_result.headings = []

        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            return_value=mock_result,
        ) as mock_parse:
            state = _make_state(enable_docling=True, vlm_mode="external")
            structure_detection_node(state)

        _, call_kwargs = mock_parse.call_args
        assert call_kwargs.get("vlm_mode") == "external"

    def test_vlm_mode_builtin_forwarded_to_parse_with_docling(self):
        """vlm_mode='builtin' is forwarded verbatim to parse_with_docling."""
        mock_result = MagicMock()
        mock_result.docling_document = MagicMock()
        mock_result.text_markdown = "text"
        mock_result.figures = []
        mock_result.headings = []

        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            return_value=mock_result,
        ) as mock_parse:
            state = _make_state(enable_docling=True, vlm_mode="builtin")
            structure_detection_node(state)

        _, call_kwargs = mock_parse.call_args
        assert call_kwargs.get("vlm_mode") == "builtin"

    def test_vlm_mode_disabled_forwarded_to_parse_with_docling(self):
        """vlm_mode='disabled' is forwarded verbatim to parse_with_docling."""
        mock_result = MagicMock()
        mock_result.docling_document = MagicMock()
        mock_result.text_markdown = "text"
        mock_result.figures = []
        mock_result.headings = []

        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            return_value=mock_result,
        ) as mock_parse:
            state = _make_state(enable_docling=True, vlm_mode="disabled")
            structure_detection_node(state)

        _, call_kwargs = mock_parse.call_args
        assert call_kwargs.get("vlm_mode") == "disabled"

    def test_processing_log_ends_with_ok_on_success(self):
        """processing_log must end with 'structure_detection:ok' on successful parse."""
        mock_result = MagicMock()
        mock_result.docling_document = MagicMock()
        mock_result.text_markdown = "text"
        mock_result.figures = []
        mock_result.headings = []

        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            return_value=mock_result,
        ):
            state = _make_state(enable_docling=True)
            update = structure_detection_node(state)

        assert update["processing_log"][-1].endswith("structure_detection:ok")

    def test_processing_log_ends_with_failed_on_strict_failure(self):
        """processing_log must end with 'structure_detection:failed' on strict failure."""
        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            side_effect=RuntimeError("parse error"),
        ):
            state = _make_state(enable_docling=True, docling_strict=True)
            update = structure_detection_node(state)

        assert update["processing_log"][-1].endswith("structure_detection:failed")

    def test_strict_failure_returns_should_skip_true(self):
        """Strict mode failure must return should_skip=True to short-circuit the DAG."""
        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            side_effect=RuntimeError("parse error"),
        ):
            state = _make_state(enable_docling=True, docling_strict=True)
            update = structure_detection_node(state)

        assert update.get("should_skip") is True

    def test_strict_failure_errors_list_contains_source_name(self):
        """Errors list on strict failure must reference the source name for traceability."""
        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            side_effect=RuntimeError("parse error"),
        ):
            state = _make_state(
                enable_docling=True,
                docling_strict=True,
                source_name="my_document.pdf",
            )
            update = structure_detection_node(state)

        assert update.get("errors"), "errors list must be non-empty"
        assert any("my_document.pdf" in e for e in update["errors"])

    def test_non_strict_fallback_runs_regex_on_raw_text(self):
        """Non-strict fallback must run regex heuristics on raw_text when parse fails."""
        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            side_effect=RuntimeError("parse error"),
        ):
            state = _make_state(
                enable_docling=True,
                docling_strict=False,
                raw_text="Figure 1 and Figure 2 are shown.",
            )
            update = structure_detection_node(state)

        assert update["structure"]["has_figures"] is True
        assert "docling_document" not in update

    def test_parse_with_docling_called_with_source_path_as_path_object(self):
        """parse_with_docling must be called with Path(state['source_path']) as first arg."""
        mock_result = MagicMock()
        mock_result.docling_document = MagicMock()
        mock_result.text_markdown = "text"
        mock_result.figures = []
        mock_result.headings = []

        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            return_value=mock_result,
        ) as mock_parse:
            state = _make_state(
                enable_docling=True,
                source_path="/tmp/my_doc.pdf",
            )
            structure_detection_node(state)

        positional_args, _ = mock_parse.call_args
        assert positional_args[0] == Path("/tmp/my_doc.pdf")

    def test_docling_parse_not_called_when_docling_disabled(self):
        """parse_with_docling must NOT be called when enable_docling_parser=False."""
        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
        ) as mock_parse:
            state = _make_state(enable_docling=False, raw_text="Text.")
            structure_detection_node(state)

        mock_parse.assert_not_called()

    def test_raw_text_replaced_with_docling_markdown_on_success(self):
        """raw_text in the state update must be replaced with parsed.text_markdown."""
        mock_result = MagicMock()
        mock_result.docling_document = MagicMock()
        mock_result.text_markdown = "# Docling Markdown\n\nContent."
        mock_result.figures = []
        mock_result.headings = ["Docling Markdown"]

        with patch(
            "src.ingest.doc_processing.nodes.structure_detection.parse_with_docling",
            return_value=mock_result,
        ):
            state = _make_state(
                enable_docling=True,
                raw_text="original raw text",
            )
            update = structure_detection_node(state)

        assert update["raw_text"] == "# Docling Markdown\n\nContent."
