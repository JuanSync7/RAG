 # @summary
 # Docling integration for ingestion parsing into markdown.
 # Exports: DoclingParseResult, warmup_docling_models, ensure_docling_ready, parse_with_docling, DoclingParser
 # Deps: dataclasses, pathlib, typing, src.ingest.support.parser_base
 # vlm_mode="builtin" activates SmolVLM picture description at parse time via PdfPipelineOptions.
 # generate_page_images=True extracts PIL.Image (RGB) page images from the converted document.
 # page_count reflects total pages in source; page_images is empty on extraction failure (no error raised).
 # DoclingParser wraps standalone functions into the DocumentParser protocol (FR-3221, FR-3223, FR-3224).
 # @end-summary
"""Docling integration for ingestion parsing.

This module provides a minimal adapter around Docling to parse source documents
into markdown for downstream ingestion steps (chunking, metadata extraction,
and optional multimodal processing).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DoclingParseResult:
    """Docling parsing output normalized for ingestion nodes.

    Attributes:
        text_markdown: Parsed markdown text.
        has_figures: Whether Docling detected any figures/pictures.
        figures: Lightweight figure identifiers for telemetry/UI.
        headings: Extracted heading text in document order.
        parser_model: Parser model identifier used for telemetry/debugging.
        docling_document: Native DoclingDocument object for HybridChunker.
            When vlm_mode="builtin", figure descriptions are already embedded
            in this document by Docling's picture description pipeline.
            None only when produced by error recovery paths.
        page_images: List of PIL.Image objects, one per extracted page. FR-201, FR-204
        page_count: Total number of pages in the source document. FR-205
    """

    text_markdown: str
    has_figures: bool
    figures: list[str]
    headings: list[str]
    parser_model: str
    docling_document: Any = None  # docling_core.types.doc.DoclingDocument
    page_images: list[Any] = field(default_factory=list)
    """List of PIL.Image objects, one per extracted page. FR-201, FR-204"""
    page_count: int = 0
    """Total number of pages in the source document. FR-205"""


def warmup_docling_models(*, artifacts_path: str = "", with_smolvlm: bool = False) -> Path:
    """Download and validate core Docling models used by ingestion.

    Args:
        artifacts_path: Optional directory to store downloaded artifacts. When
            empty, Docling's default cache location is used.
        with_smolvlm: If True, also download SmolVLM model artifacts.
            Must be True when vlm_mode is "builtin".

    Returns:
        The resolved Docling model root directory.

    Raises:
        RuntimeError: If Docling's downloader is unavailable or required models
            are missing after download.
    """
    try:
        from docling.datamodel.pipeline_options import LayoutOptions
        from docling.models.stages.table_structure.table_structure_model import (
            TableStructureModel,
        )
        from docling.utils.model_downloader import download_models
    except Exception as exc:  # pragma: no cover - import path depends on runtime env
        raise RuntimeError("Docling model downloader is unavailable") from exc

    output_dir = None
    if artifacts_path:
        output_dir = Path(artifacts_path)
        output_dir.mkdir(parents=True, exist_ok=True)

    model_root = download_models(
        output_dir=output_dir,
        force=False,
        progress=False,
        with_layout=True,
        with_tableformer=True,
        with_tableformer_v2=False,
        with_code_formula=False,
        with_picture_classifier=False,
        with_smolvlm=with_smolvlm,
        with_granitedocling=False,
        with_granitedocling_mlx=False,
        with_smoldocling=False,
        with_smoldocling_mlx=False,
        with_granite_vision=False,
        with_granite_chart_extraction=False,
        with_rapidocr=False,
        with_easyocr=False,
    )
    layout_repo_dir = model_root / LayoutOptions().model_spec.model_repo_folder
    tableformer_repo_dir = model_root / TableStructureModel._model_repo_folder
    if not layout_repo_dir.exists():
        raise RuntimeError(f"Docling Heron/Layout model not found in: {layout_repo_dir}")
    if not tableformer_repo_dir.exists():
        raise RuntimeError(f"Docling TableFormer model not found in: {tableformer_repo_dir}")
    return model_root


def ensure_docling_ready(
    *,
    parser_model: str,
    artifacts_path: str = "",
    auto_download: bool = True,
) -> None:
    """Validate Docling runtime setup before ingestion starts.

    This function performs a lightweight import check and, optionally, ensures
    the required models are present by triggering a download.

    Args:
        parser_model: Parser model identifier used for telemetry and validation.
        artifacts_path: Optional directory containing Docling artifacts.
        auto_download: Whether to automatically download missing artifacts.

    Raises:
        RuntimeError: If Docling is unavailable, configuration is invalid, or
            artifacts cannot be prepared.
    """
    if not str(parser_model).strip():
        raise RuntimeError("Docling parser model is empty")
    try:
        from docling.document_converter import DocumentConverter
    except Exception as exc:  # pragma: no cover - import path depends on runtime env
        raise RuntimeError(
            "Docling is required but not installed. Install with: uv add docling"
        ) from exc

    prepared_artifacts_path = artifacts_path
    if auto_download:
        model_root = warmup_docling_models(artifacts_path=artifacts_path)
        if not prepared_artifacts_path:
            prepared_artifacts_path = str(model_root)

    if prepared_artifacts_path:
        artifacts = Path(prepared_artifacts_path)
        if not artifacts.exists() or not artifacts.is_dir():
            raise RuntimeError(f"Docling artifacts path is invalid: {prepared_artifacts_path}")
    # Smoke-test: verify DocumentConverter can be instantiated.
    DocumentConverter()


def _extract_headings_from_markdown(text: str) -> list[str]:
    """Extract heading text from markdown.

    Args:
        text: Markdown content.

    Returns:
        Heading text in appearance order.
    """
    headings: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            if heading:
                headings.append(heading)
    return headings


def _extract_page_images_from_result(conv_result: Any) -> tuple[list[Any], int]:
    """Extract PIL.Image (RGB) page images from a Docling ConversionResult.

    Tries multiple access patterns to accommodate different Docling versions:
    1. ``conv_result.pages`` — ConversionResult page list with image attached.
    2. ``conv_result.document.pages`` — DoclingDocument pages dict/list.

    Each page image is converted to RGB to normalise colour space (FR-204).
    ``page_count`` reflects total pages regardless of partial extraction
    failures (FR-205).

    Args:
        conv_result: A Docling ``ConversionResult`` object returned by
            ``DocumentConverter.convert()``.

    Returns:
        A ``(page_images, page_count)`` tuple.  ``page_images`` is an empty
        list when extraction fails entirely; individual missing pages are
        silently skipped.
    """
    page_images: list[Any] = []
    page_count: int = 0

    # --- Strategy 1: ConversionResult.pages (preferred, richer API) ----------
    conv_pages = getattr(conv_result, "pages", None)
    if conv_pages is not None:
        pages_iter = conv_pages.values() if hasattr(conv_pages, "values") else conv_pages
        pages_list = list(pages_iter)
        page_count = len(pages_list)
        for page in pages_list:
            try:
                img = None
                # Try .image.pil_image (newer Docling page image API)
                page_image_obj = getattr(page, "image", None)
                if page_image_obj is not None:
                    img = getattr(page_image_obj, "pil_image", None)
                # Fallback: page.get_image() callable
                if img is None and callable(getattr(page, "get_image", None)):
                    img = page.get_image()
                if img is not None:
                    page_images.append(img.convert("RGB"))
                else:
                    logger.warning("Page has no extractable image; skipping.")
            except Exception as exc:
                # Skip individual page; do not block the pipeline.
                logger.warning("Failed to extract image from page: %s — skipping.", exc)
        if page_count > 0:
            return page_images, page_count

    # --- Strategy 2: DoclingDocument.pages -----------------------------------
    document = getattr(conv_result, "document", None)
    if document is None:
        return page_images, page_count

    doc_pages = getattr(document, "pages", None)
    if doc_pages is None:
        return page_images, page_count

    pages_iter = doc_pages.values() if hasattr(doc_pages, "values") else doc_pages
    pages_list = list(pages_iter)
    page_count = len(pages_list)
    for page in pages_list:
        try:
            img = None
            page_image_obj = getattr(page, "image", None)
            if page_image_obj is not None:
                img = getattr(page_image_obj, "pil_image", None)
            if img is None and callable(getattr(page, "get_image", None)):
                img = page.get_image()
            if img is not None:
                page_images.append(img.convert("RGB"))
            else:
                logger.warning("Page has no extractable image; skipping.")
        except Exception as exc:
            logger.warning("Failed to extract image from page: %s — skipping.", exc)

    return page_images, page_count


def parse_with_docling(
    source_path: Path,
    *,
    parser_model: str,
    artifacts_path: str = "",
    vlm_mode: str = "disabled",
    generate_page_images: bool = False,
) -> DoclingParseResult:
    """Parse a source document into markdown using local Docling runtime.

    When vlm_mode="builtin", configures DocumentConverter to run SmolVLM on
    figure images during conversion. Figure descriptions are baked into the
    returned DoclingDocument — no post-chunking VLM step is required.

    When vlm_mode="external" or vlm_mode="disabled", do_picture_description is
    False (existing behavior). External VLM enrichment happens post-chunking via
    vlm_enrichment_node.

    When generate_page_images=True, page images are extracted from the
    ConversionResult as PIL.Image (RGB) objects and stored in
    ``DoclingParseResult.page_images``.  Extraction failures are logged as
    warnings and never block the text-track pipeline (FR-107, FR-201).

    Args:
        source_path: Path to the source document to parse.
        parser_model: Parser model identifier used for telemetry/debugging.
        artifacts_path: Optional directory containing Docling artifacts.
        vlm_mode: "builtin" activates Docling's SmolVLM picture description at
            parse time. "external" and "disabled" leave do_picture_description=False.
        generate_page_images: When True, extract per-page PIL.Image objects
            (RGB) from the conversion result.  Defaults to False. FR-107.

    Returns:
        A normalized `DoclingParseResult` with docling_document populated from
        result.document.  When generate_page_images=True, page_images and
        page_count are also populated.

    Raises:
        RuntimeError: If Docling is unavailable, conversion fails, or the output
            is empty/unsupported.
    """
    import logging

    try:
        # Import lazily to keep module import cheap and explicit.
        from docling.document_converter import DocumentConverter
    except Exception as exc:  # pragma: no cover - import path depends on runtime env
        raise RuntimeError(
            "Docling is required but not installed. Install with: uv add docling"
        ) from exc

    converter_kwargs: dict[str, Any] = {}
    # Note: artifacts_path is accepted for caller compat but no longer passed
    # to DocumentConverter (removed in newer Docling versions). Model location
    # is controlled by warmup_docling_models / HF cache.

    if vlm_mode == "builtin":
        # Lazy import to keep module-level import cheap.
        _builtin_vlm_configured = False
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                PictureDescriptionVlmEngineOptions,
            )
            from docling.document_converter import PdfFormatOption

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_picture_description = True
            pipeline_options.picture_description_options = (
                PictureDescriptionVlmEngineOptions.from_preset("smolvlm")
            )
            converter_kwargs["format_options"] = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
            _builtin_vlm_configured = True
        except (ImportError, Exception) as exc:
            logging.getLogger(__name__).warning(
                "vlm_mode='builtin' requested but SmolVLM setup failed (%s); "
                "proceeding without picture description.",
                exc,
            )
        converter = DocumentConverter(**converter_kwargs)
        _ = _builtin_vlm_configured  # noqa: F841 — reserved for telemetry
    else:
        converter = DocumentConverter(**converter_kwargs)

    try:
        result = converter.convert(str(source_path))
    except Exception as exc:
        raise RuntimeError(
            f"Docling conversion failed for {source_path}: {exc}"
        ) from exc
    document = getattr(result, "document", None)
    if document is None:
        raise RuntimeError("Docling conversion did not return a document object")

    if not hasattr(document, "export_to_markdown"):
        raise RuntimeError("Docling document object does not support markdown export")
    markdown = str(document.export_to_markdown() or "").strip()
    if not markdown:
        raise RuntimeError("Docling returned empty markdown output")

    pictures = list(getattr(document, "pictures", []) or [])
    figures = [f"Figure {idx + 1}" for idx, _ in enumerate(pictures)]
    headings = _extract_headings_from_markdown(markdown)

    # --- Page image extraction (FR-107, FR-201, FR-204, FR-205) --------------
    page_images: list[Any] = []
    page_count: int = 0
    if generate_page_images:
        try:
            page_images, page_count = _extract_page_images_from_result(result)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Page image extraction failed for %s (%s); page_images will be empty.",
                source_path,
                exc,
            )
            page_images = []

    return DoclingParseResult(
        text_markdown=markdown,
        has_figures=bool(figures),
        figures=figures,
        headings=headings,
        parser_model=parser_model,
        docling_document=document,
        page_images=page_images,
        page_count=page_count,
    )


def chunk_markdown_via_docling(
    markdown_text: str,
    *,
    max_tokens: int = 512,
    source_path: Path | None = None,
    config: Any = None,
    source_key: str | None = None,
    doc_store_client: Any = None,
) -> list:
    """Caption images inline (if enabled), build a DoclingDocument, chunk via HybridChunker.

    Pipeline:
      1. (optional) VLM-caption every ``![alt](src)`` in-place. Local files and
         data-URLs are sent to the configured VLM; HTTP URLs are skipped with a
         warning (alt text preserved). On VLM failure, alt text stands in.
      2. Build a DoclingDocument from the rewritten markdown via the MD backend
         (~0.2s, no models).
      3. Run HybridChunker — preserves lists, tables, and requirement blocks
         as atomic units.

    Captioning runs only when both ``source_path`` and ``config`` are provided
    AND ``config.enable_vision_processing`` is True. Without those, the function
    behaves as a pure structural chunker (no VLM calls).

    Each returned Chunk carries a ``figures`` list in ``extra_metadata`` with
    one entry per image reference encountered (label, src, alt, caption, …)
    so citation UIs can link the chunk back to the original image.

    Returns a list of Chunk objects (the existing parser_base contract).
    Raises any exception from Docling — callers should catch and fall back.
    """
    from io import BytesIO

    from docling.datamodel.base_models import DocumentStream, InputFormat
    from docling.document_converter import DocumentConverter
    from docling_core.transforms.chunker import HybridChunker

    from src.ingest.support.parser_base import Chunk

    figures: list[dict] = []
    text_for_chunking = markdown_text
    enable_vision = bool(getattr(config, "enable_vision_processing", False))
    if source_path is not None and config is not None and enable_vision:
        from src.ingest.support.vision import caption_markdown_images_inline

        text_for_chunking, figures = caption_markdown_images_inline(
            markdown_text,
            source_path=source_path,
            config=config,
            source_key=source_key,
            doc_store_client=doc_store_client,
        )

    converter = DocumentConverter(allowed_formats=[InputFormat.MD])
    conv_result = converter.convert(
        source=DocumentStream(
            name="input.md", stream=BytesIO(text_for_chunking.encode("utf-8"))
        )
    )
    docling_document = conv_result.document

    chunker = HybridChunker(max_tokens=max_tokens, merge_peers=True)
    raw_chunks = list(chunker.chunk(dl_doc=docling_document))

    figures_payload = figures if figures else []
    chunks: list = []
    for idx, raw in enumerate(raw_chunks):
        headings: list[str] = []
        meta = getattr(raw, "meta", None)
        if meta is not None:
            headings = list(getattr(meta, "headings", None) or [])
        # Only attach figures whose label appears in this chunk's text. Cheap
        # substring check — keeps the citation linkback specific to chunks
        # that actually reference a figure.
        chunk_figures = [
            fig for fig in figures_payload if fig.get("label", "") in raw.text
        ]
        extra: dict = {}
        if chunk_figures:
            extra["figures"] = chunk_figures
        chunks.append(
            Chunk(
                text=raw.text,
                section_path=" > ".join(headings),
                heading=headings[-1] if headings else "",
                heading_level=len(headings),
                chunk_index=idx,
                extra_metadata=extra,
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# DoclingParser — DocumentParser protocol implementation (FR-3221–FR-3224)
# ---------------------------------------------------------------------------

class DoclingParser:
    """Docling-based document parser implementing DocumentParser protocol.

    Wraps existing parse_with_docling(), ensure_docling_ready(), and
    warmup_docling_models() into a class with per-document instance lifecycle.

    Internal state:
        _docling_document: DoclingDocument retained between parse() and chunk().
            Never exposed via ParseResult or any public API. FR-3205.
        _vlm_mode: VLM mode from config, used during parse(). FR-3224.
        _max_tokens: HybridChunker max tokens from config.
    """

    def __init__(self) -> None:
        self._docling_document: Any = None
        self._vlm_mode: str = "disabled"
        self._max_tokens: int = 512

    def parse(self, file_path: Path, config: Any) -> "ParseResult":
        """Parse a document using Docling. FR-3221.

        Calls existing parse_with_docling() internally. Stores the
        DoclingDocument in self._docling_document for use by chunk().
        Returns a ParseResult with no DoclingDocument attribute.

        Args:
            file_path: Path to the source document.
            config: IngestionConfig instance.

        Returns:
            ParseResult with markdown, headings, has_figures, page_count.
        """
        from src.ingest.support.parser_base import ParseResult

        self._vlm_mode = getattr(config, "vlm_mode", "disabled")
        self._max_tokens = getattr(config, "hybrid_chunker_max_tokens", 512)

        result = parse_with_docling(
            file_path,
            parser_model=config.docling_model,
            artifacts_path=config.docling_artifacts_path,
            vlm_mode=self._vlm_mode,
            generate_page_images=config.generate_page_images,
        )

        # Encapsulate DoclingDocument — FR-3205
        self._docling_document = result.docling_document

        return ParseResult(
            markdown=result.text_markdown,
            headings=result.headings,
            has_figures=result.has_figures,
            page_count=result.page_count,
        )

    def chunk(self, parse_result: Any) -> list:
        """Chunk using Docling's HybridChunker. FR-3223.

        Operates on self._docling_document (internal state from parse()).
        Maps HybridChunker output to Chunk dataclass with section_path
        derived from meta.headings.

        Args:
            parse_result: ParseResult from a prior parse() call.

        Returns:
            List of Chunk objects with heading hierarchy metadata.

        Raises:
            RuntimeError: If called before parse() (no DoclingDocument).
        """
        from src.ingest.support.parser_base import Chunk

        if self._docling_document is None:
            raise RuntimeError(
                "DoclingParser.chunk() called before parse(). "
                "Call parse() first to populate internal DoclingDocument."
            )

        from docling_core.transforms.chunker import HybridChunker

        chunker = HybridChunker(
            max_tokens=self._max_tokens,
            merge_peers=True,
        )
        chunk_iter = chunker.chunk(dl_doc=self._docling_document)
        raw_chunks = list(chunk_iter)

        chunks: list[Chunk] = []
        for idx, raw in enumerate(raw_chunks):
            # Extract heading hierarchy from HybridChunker metadata
            headings: list[str] = []
            meta = getattr(raw, "meta", None)
            if meta is not None:
                headings = list(getattr(meta, "headings", None) or [])

            heading = headings[-1] if headings else ""
            section_path = " > ".join(headings)
            heading_level = len(headings)

            chunks.append(
                Chunk(
                    text=raw.text,
                    section_path=section_path,
                    heading=heading,
                    heading_level=heading_level,
                    chunk_index=idx,
                    extra_metadata={},
                )
            )
        return chunks

    @classmethod
    def ensure_ready(cls, config: Any) -> None:
        """Validate Docling runtime. Delegates to ensure_docling_ready(). FR-3204."""
        ensure_docling_ready(
            parser_model=config.docling_model,
            artifacts_path=config.docling_artifacts_path,
            auto_download=config.docling_auto_download,
        )

    @classmethod
    def warmup(cls, config: Any) -> None:
        """Download Docling models. Delegates to warmup_docling_models(). FR-3207."""
        warmup_docling_models(
            artifacts_path=config.docling_artifacts_path,
            with_smolvlm=(getattr(config, "vlm_mode", "disabled") == "builtin"),
        )


# DEPRECATED standalone functions below — preserved for backward compatibility.
# Use DoclingParser class for new code.
# parse_with_docling()       — still available
# ensure_docling_ready()     — still available
# warmup_docling_models()    — still available
# DoclingParseResult         — still available
