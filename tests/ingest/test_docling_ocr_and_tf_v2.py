"""Tests for TableFormer v2 + OCR auto-trigger wiring in Docling integration.

Covers Workstream A:
- warmup_docling_models() default flags now request TF v2 and RapidOCR artifacts.
- warmup_docling_models() still wires the smolvlm flag through.
- parse_with_docling() builds a PdfPipelineOptions with OCR auto-trigger and
  TF v2 (TableStructureOptions.mode=ACCURATE) when configured.
- parse_with_docling() respects enable_ocr=False (do_ocr=False).
- tableformer_mode "fast" maps to TableFormerMode.FAST.
- DoclingParser.parse() forwards new IngestionConfig flags to
  parse_with_docling().

All tests mock the docling library at the seam — they assert on the kwargs/
attributes passed to docling APIs, not on real model behavior.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_docling_config(**overrides):
    """Build a duck-typed config object covering all knobs parse() reads."""
    defaults = dict(
        docling_model="ds4sd/docling-models",
        docling_artifacts_path="",
        docling_auto_download=False,
        vlm_mode="disabled",
        hybrid_chunker_max_tokens=512,
        generate_page_images=False,
        enable_ocr=True,
        ocr_engine="rapidocr",
        tableformer_mode="accurate",
        tableformer_do_cell_matching=True,
    )
    defaults.update(overrides)
    return type("DoclingConfig", (), defaults)()


def _make_mock_converter(markdown="# T\nbody"):
    mock_document = MagicMock()
    mock_document.export_to_markdown = MagicMock(return_value=markdown)
    mock_document.pictures = []
    mock_document.pages = []
    mock_result = MagicMock()
    mock_result.document = mock_document
    mock_result.pages = None
    mock_converter = MagicMock()
    mock_converter.convert = MagicMock(return_value=mock_result)
    return mock_converter


# ---------------------------------------------------------------------------
# warmup_docling_models()
# ---------------------------------------------------------------------------


class TestWarmupTableFormerV2AndRapidOcr:
    def _patch_warmup_imports(self, mock_download):
        """Patch the lazy imports inside warmup_docling_models."""
        import builtins
        real_import = builtins.__import__

        mock_layout_options_instance = MagicMock()
        mock_layout_options_instance.model_spec.model_repo_folder = "layout"

        # model_root / "layout" and model_root / "table" — return existing dirs
        mock_model_root = MagicMock(spec=Path)
        mock_layout_dir = MagicMock()
        mock_layout_dir.exists.return_value = True
        mock_table_dir = MagicMock()
        mock_table_dir.exists.return_value = True
        mock_model_root.__truediv__ = MagicMock(
            side_effect=[mock_layout_dir, mock_table_dir]
        )
        mock_download.return_value = mock_model_root

        def fake_import(name, *args, **kwargs):
            if name == "docling.datamodel.pipeline_options":
                mod = MagicMock()
                mod.LayoutOptions = MagicMock(return_value=mock_layout_options_instance)
                return mod
            if name == "docling.models.stages.table_structure.table_structure_model":
                mod = MagicMock()
                mod.TableStructureModel = MagicMock()
                mod.TableStructureModel._model_repo_folder = "table"
                return mod
            if name == "docling.utils.model_downloader":
                mod = MagicMock()
                mod.download_models = mock_download
                return mod
            return real_import(name, *args, **kwargs)

        return fake_import

    def test_warmup_defaults_request_tf_v2_and_rapidocr(self):
        """Default warmup must request TF v2 and RapidOCR artifacts."""
        from src.ingest.support import docling as docling_mod

        mock_download = MagicMock()
        fake_import = self._patch_warmup_imports(mock_download)

        with patch("builtins.__import__", side_effect=fake_import):
            docling_mod.warmup_docling_models()

        kwargs = mock_download.call_args.kwargs
        assert kwargs["with_tableformer_v2"] is True
        assert kwargs["with_rapidocr"] is True

    def test_warmup_smolvlm_flag_threaded_through(self):
        """warmup_docling_models(with_smolvlm=True) must pass with_smolvlm=True."""
        from src.ingest.support import docling as docling_mod

        mock_download = MagicMock()
        fake_import = self._patch_warmup_imports(mock_download)

        with patch("builtins.__import__", side_effect=fake_import):
            docling_mod.warmup_docling_models(with_smolvlm=True)

        kwargs = mock_download.call_args.kwargs
        assert kwargs["with_smolvlm"] is True

    def test_warmup_can_disable_tf_v2_and_rapidocr(self):
        """Booleans must be tunable via kwargs (backward compat surface)."""
        from src.ingest.support import docling as docling_mod

        mock_download = MagicMock()
        fake_import = self._patch_warmup_imports(mock_download)

        with patch("builtins.__import__", side_effect=fake_import):
            docling_mod.warmup_docling_models(
                with_tableformer_v2=False, with_rapidocr=False
            )

        kwargs = mock_download.call_args.kwargs
        assert kwargs["with_tableformer_v2"] is False
        assert kwargs["with_rapidocr"] is False


# ---------------------------------------------------------------------------
# parse_with_docling() — PdfPipelineOptions wiring
# ---------------------------------------------------------------------------


class TestParseWithDoclingOcrAndTfV2:
    def _capture_pipeline_options(self, tmp_path: Path, **parse_kwargs):
        """Run parse_with_docling against mocked DocumentConverter; return the
        PdfPipelineOptions that was wired into PdfFormatOption."""
        from src.ingest.support import docling as docling_mod
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        captured: dict = {}

        original_pdf_options = PdfPipelineOptions

        def spy_pdf_options(*a, **kw):
            opts = original_pdf_options(*a, **kw)
            captured["opts"] = opts
            return opts

        mock_converter = _make_mock_converter()
        mock_dc_module = MagicMock()
        mock_dc_module.DocumentConverter = MagicMock(return_value=mock_converter)
        # Also need PdfFormatOption available on this module if parse_with_docling
        # imports it from there; use the real one.
        from docling.document_converter import PdfFormatOption as RealPdfFormatOption
        mock_dc_module.PdfFormatOption = RealPdfFormatOption

        with patch.dict("sys.modules", {"docling.document_converter": mock_dc_module}):
            with patch(
                "docling.datamodel.pipeline_options.PdfPipelineOptions",
                side_effect=spy_pdf_options,
            ):
                docling_mod.parse_with_docling(pdf, parser_model="m", **parse_kwargs)

        return captured.get("opts")

    def test_ocr_enabled_sets_do_ocr_and_rapidocr_options(self, tmp_path: Path):
        """enable_ocr=True must build PdfPipelineOptions with do_ocr=True and
        RapidOcrOptions(force_full_page_ocr=False)."""
        from docling.datamodel.pipeline_options import RapidOcrOptions

        opts = self._capture_pipeline_options(
            tmp_path,
            vlm_mode="disabled",
            enable_ocr=True,
            tableformer_mode="accurate",
        )
        assert opts is not None
        assert opts.do_ocr is True
        assert isinstance(opts.ocr_options, RapidOcrOptions)
        assert opts.ocr_options.force_full_page_ocr is False

    def test_ocr_disabled_sets_do_ocr_false(self, tmp_path: Path):
        """enable_ocr=False must result in do_ocr=False on PdfPipelineOptions."""
        opts = self._capture_pipeline_options(
            tmp_path,
            vlm_mode="disabled",
            enable_ocr=False,
            tableformer_mode="accurate",
        )
        assert opts is not None
        assert opts.do_ocr is False

    def test_tableformer_mode_accurate_maps_to_tf_v2(self, tmp_path: Path):
        """tableformer_mode='accurate' → TableStructureOptions.mode == ACCURATE."""
        from docling.datamodel.pipeline_options import TableFormerMode

        opts = self._capture_pipeline_options(
            tmp_path,
            vlm_mode="disabled",
            enable_ocr=True,
            tableformer_mode="accurate",
        )
        assert opts is not None
        assert opts.table_structure_options.mode == TableFormerMode.ACCURATE

    def test_tableformer_mode_fast_maps_to_fast(self, tmp_path: Path):
        """tableformer_mode='fast' → TableStructureOptions.mode == FAST."""
        from docling.datamodel.pipeline_options import TableFormerMode

        opts = self._capture_pipeline_options(
            tmp_path,
            vlm_mode="disabled",
            enable_ocr=False,
            tableformer_mode="fast",
        )
        assert opts is not None
        assert opts.table_structure_options.mode == TableFormerMode.FAST

    def test_tableformer_do_cell_matching_forwarded(self, tmp_path: Path):
        """tableformer_do_cell_matching is wired onto TableStructureOptions."""
        opts = self._capture_pipeline_options(
            tmp_path,
            vlm_mode="disabled",
            enable_ocr=False,
            tableformer_mode="accurate",
            tableformer_do_cell_matching=False,
        )
        assert opts is not None
        assert opts.table_structure_options.do_cell_matching is False


# ---------------------------------------------------------------------------
# DoclingParser.parse() — threads config through
# ---------------------------------------------------------------------------


class TestDoclingParserConfigThreading:
    def test_parse_forwards_ocr_and_tf_config_to_parse_with_docling(
        self, tmp_path: Path
    ):
        """DoclingParser.parse() must pass IngestionConfig.enable_ocr and
        tableformer_mode through to parse_with_docling()."""
        from src.ingest.support import docling as docling_mod
        from src.ingest.support.docling import DoclingParser

        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        cfg = _make_docling_config(
            enable_ocr=False,
            tableformer_mode="fast",
            tableformer_do_cell_matching=False,
        )

        mock_result = MagicMock()
        mock_result.text_markdown = "# T"
        mock_result.headings = ["T"]
        mock_result.has_figures = False
        mock_result.page_count = 0
        mock_result.page_images = []
        mock_result.docling_document = MagicMock()

        with patch.object(
            docling_mod, "parse_with_docling", return_value=mock_result
        ) as mock_pwd:
            DoclingParser().parse(pdf, cfg)

        kwargs = mock_pwd.call_args.kwargs
        assert kwargs["enable_ocr"] is False
        assert kwargs["tableformer_mode"] == "fast"
        assert kwargs["tableformer_do_cell_matching"] is False
