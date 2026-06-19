 # @summary
 # Docling integration for ingestion parsing into markdown.
 # Exports: DoclingParseResult, warmup_docling_models, ensure_docling_ready, parse_with_docling, DoclingParser
 # Deps: dataclasses, pathlib, typing, src.ingest.support.parser_base
 # vlm_mode="builtin" activates SmolVLM picture description at parse time via PdfPipelineOptions.
 # generate_page_images=True extracts PIL.Image (RGB) page images from the converted document.
 # page_count reflects total pages in source; page_images is empty on extraction failure (no error raised).
 # DoclingParser wraps standalone functions into the DocumentParser protocol (FR-3221, FR-3223, FR-3224).
 # parse_with_docling builds a PdfPipelineOptions on every call: TableFormer mode
 # (accurate=TF v2 / fast=TF v1), OCR auto-trigger via RapidOcrOptions(force_full_page_ocr=False),
 # and optional built-in SmolVLM picture description.
 # warmup_docling_models defaults to fetching TableFormer v2 and RapidOCR artifacts.
 # @end-summary
"""Docling integration for ingestion parsing.

This module provides a minimal adapter around Docling to parse source documents
into markdown for downstream ingestion steps (chunking, metadata extraction,
and optional multimodal processing).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


import re as _re_caption  # local alias to avoid colliding with downstream `re` usage


_CAPTION_LABEL_RE = _re_caption.compile(
    r"^\s*(Table)\s+(\d+(?:[.-]\d+)?)",
    _re_caption.IGNORECASE,
)

# Figure-side variant. ``Fig.`` and ``Fig`` are tolerated, but a word boundary
# after the keyword prevents matches like ``figured`` / ``configure`` /
# ``Refigure`` that share the prefix.
# Match ``Figure``/``Fig.``/``Fig`` at the start of the caption only when
# followed by whitespace and a digit. Anchoring at ``^\s*`` plus the digit
# lookahead is what excludes ``figured``/``configure``/``Refigure`` — those
# either fail the prefix anchor or fail the trailing-digit requirement.
_FIGURE_CAPTION_LABEL_RE = _re_caption.compile(
    r"^\s*(?:Figure|Fig\.?)\s+(\d+(?:[.-]\d+)?)",
    _re_caption.IGNORECASE,
)


def _extract_caption_label(caption: str) -> str:
    """Pull the canonical ``Table N`` / ``Table N-N`` / ``Table N.N`` label out
    of a caption string. Returns ``""`` when no label prefix can be parsed.

    Preserves the user's separator (``-`` or ``.``) and titlecases the
    ``Table`` keyword. Examples:

    - ``"Table 5-2: GPIO matrix"`` → ``"Table 5-2"``
    - ``"Table 12"``               → ``"Table 12"``
    - ``"Table 3.1 — power modes"``→ ``"Table 3.1"``
    - ``"GPIO matrix"``            → ``""``
    """
    if not caption:
        return ""
    m = _CAPTION_LABEL_RE.match(caption)
    if not m:
        return ""
    return f"Table {m.group(2)}"


def _extract_figure_caption_label(caption: str) -> str:
    """Pull the canonical ``Figure N`` / ``Figure N-N`` / ``Figure N.N`` label
    out of a figure caption string. Returns ``""`` when no label prefix can be
    parsed.

    Accepts ``Figure``, ``Fig``, and ``Fig.`` (case-insensitive) and normalises
    the keyword to ``Figure``. Preserves the user's separator (``-`` or ``.``).
    Examples:

    - ``"Figure 4-1: SoC block diagram"`` → ``"Figure 4-1"``
    - ``"Fig. 2 — pin layout"``           → ``"Figure 2"``
    - ``"FIGURE 10.3 power tree"``        → ``"Figure 10.3"``
    - ``"figured out the wiring"``        → ``""``
    """
    if not caption:
        return ""
    m = _FIGURE_CAPTION_LABEL_RE.match(caption)
    if not m:
        return ""
    return f"Figure {m.group(1)}"


def _normalize_figure_ref_value(value: str) -> str:
    """Coerce a raw figure ref match (e.g. ``"Fig. 4-1"``) to the canonical
    ``"Figure N"`` form the resolver filter expects.

    Returns the input unchanged when no figure prefix matches — callers
    decide whether to keep or drop. We anchor against the same regex
    :data:`_FIGURE_CAPTION_LABEL_RE` uses so the round-trip (emission ↔
    resolution) is symmetric.
    """
    if not value:
        return value
    m = _FIGURE_CAPTION_LABEL_RE.match(value)
    if not m:
        return value
    return f"Figure {m.group(1)}"


def _stamp_xref_targets(meta: dict, text: str) -> None:
    """Populate ``meta["xref_targets"]`` from ``cross_refs(text)``.

    The payload is a JSON-encoded ``list[{type, value}]`` (Weaviate's TEXT
    column doesn't support list-of-struct natively; the retrieval-side
    expander decodes it back).  Empty text yields ``"[]"``.

    ``figure``-type refs are filtered when
    ``RAG_XREF_EXTRACT_FIGURE_REFS`` is False — we don't have a
    FigureArtifact registry to resolve against yet, so emitting them is
    pure downstream noise.  Read the flag fresh on each call so tests can
    monkeypatch ``config.settings`` without re-importing this module.
    """
    # Imported here to avoid widening the module's top-of-file dep graph.
    from src.ingest.common.shared import cross_refs

    try:
        refs = cross_refs(text or "")
    except Exception:  # pragma: no cover - defensive
        refs = []

    try:
        from config import settings as _settings

        emit_figures = bool(getattr(_settings, "RAG_XREF_EXTRACT_FIGURE_REFS", False))
    except Exception:  # pragma: no cover - defensive
        emit_figures = False
    if not emit_figures:
        refs = [r for r in refs if str((r or {}).get("type") or "") != "figure"]
    else:
        # Normalise figure ref values (``"Fig. 4-1"`` → ``"Figure 4-1"``) so
        # the value stamped at ingest matches the resolver-side filter that
        # FIG-2's ``_filter_for_figure`` issues against ``caption_label``.
        # Anything that doesn't match the canonical figure-prefix regex is
        # dropped — that path catches false positives like ``"Refigure 4-1"``
        # that the loose ``cross_refs`` pattern would otherwise surface.
        normalised: list[dict] = []
        for r in refs:
            if not isinstance(r, dict):
                continue
            if str(r.get("type") or "") == "figure":
                norm_v = _normalize_figure_ref_value(str(r.get("value") or ""))
                if not norm_v.startswith("Figure "):
                    continue
                normalised.append({"type": "figure", "value": norm_v})
            else:
                normalised.append(r)
        refs = normalised

    meta["xref_targets"] = json.dumps(refs)

try:
    from config.settings import EMBEDDING_MODEL_PATH, RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS
except Exception:  # pragma: no cover — defensive; settings should always import
    EMBEDDING_MODEL_PATH = ""
    RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS = 1024

from src.platform.observability import get_tracer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HybridChunker tokenizer resolution + module-level cache
# ---------------------------------------------------------------------------

_TOKENIZER_CACHE: dict[tuple[str, int], Any] = {}


def _resolve_tokenizer_model_id(config: Any) -> str:
    """Resolve which HF tokenizer model id to use for HybridChunker.

    Resolution order:
      1. ``config.hybrid_chunker_tokenizer_model`` if non-empty.
      2. ``EMBEDDING_MODEL_PATH`` (the embedder repo / local path) if non-empty,
         so token counts match the embedding model — but ONLY when it is a bare
         HF repo id or a local path that actually exists.
      3. Hardcoded final fallback ``"BAAI/bge-m3"`` (downloaded on demand).

    The existence guard matters: deployments that embed remotely via TEI/vLLM set
    ``EMBEDDING_MODEL_PATH`` to a local dir (e.g. ``/root/models/baai/bge-m3``)
    that is never materialised in the ingest container. Passing that missing path
    to ``AutoTokenizer.from_pretrained`` makes it treat the path as a repo id and
    raise ``HFValidationError``, which breaks native HybridChunker chunking. When
    the path is absent we fall through to the downloadable HF default instead.
    """
    cfg_model = getattr(config, "hybrid_chunker_tokenizer_model", None) if config is not None else None
    if cfg_model:
        return cfg_model
    if EMBEDDING_MODEL_PATH:
        _looks_local = EMBEDDING_MODEL_PATH.startswith(("/", "./", "../", "~"))
        if not _looks_local:
            return EMBEDDING_MODEL_PATH  # bare HF repo id — use as-is
        try:
            _exists = Path(EMBEDDING_MODEL_PATH).expanduser().exists()
        except OSError:
            # An unreadable parent (e.g. /root for a non-root process) raises
            # PermissionError rather than returning False — treat as absent.
            _exists = False
        if _exists:
            return EMBEDDING_MODEL_PATH
        logger.warning(
            "EMBEDDING_MODEL_PATH=%s does not exist locally; falling back to "
            "'BAAI/bge-m3' for the HybridChunker tokenizer (set "
            "config.hybrid_chunker_tokenizer_model to override).",
            EMBEDDING_MODEL_PATH,
        )
    return "BAAI/bge-m3"


def _build_hf_tokenizer(model_id: str, max_tokens: int) -> Any:
    """Build a docling-core HuggingFaceTokenizer for ``model_id``.

    Isolated for monkeypatching in tests. Raises any underlying exception so the
    caller (``_get_or_build_tokenizer``) can decide on fallback behavior.
    """
    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )
    from transformers import AutoTokenizer

    return HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(model_id),
        max_tokens=max_tokens,
    )


def _get_or_build_tokenizer(model_id: str, *, max_tokens: int) -> Any:
    """Return a cached HuggingFaceTokenizer or build (and cache) a new one.

    Cache key is ``(model_id, max_tokens)`` since max_tokens is baked into the
    tokenizer instance.
    """
    key = (model_id, int(max_tokens))
    cached = _TOKENIZER_CACHE.get(key)
    if cached is not None:
        return cached
    tokenizer = _build_hf_tokenizer(model_id, max_tokens)
    _TOKENIZER_CACHE[key] = tokenizer
    # Surface the resolved tokenizer choice once per (model, max_tokens) pair —
    # silent auto-tracking is hard to debug otherwise.
    logger.info(
        "HybridChunker tokenizer resolved: model_id=%s max_tokens=%d",
        model_id,
        max_tokens,
    )
    return tokenizer


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


def warmup_docling_models(
    *,
    artifacts_path: str = "",
    with_smolvlm: bool = False,
    with_tableformer_v2: bool = True,
    with_rapidocr: bool = True,
    with_easyocr: bool = False,
) -> Path:
    """Download and validate core Docling models used by ingestion.

    Args:
        artifacts_path: Optional directory to store downloaded artifacts. When
            empty, Docling's default cache location is used.
        with_smolvlm: If True, also download SmolVLM model artifacts.
            Must be True when vlm_mode is "builtin".
        with_tableformer_v2: If True (default), also fetch TableFormer v2
            artifacts so ``tableformer_mode="accurate"`` works offline.
        with_rapidocr: If True (default), also fetch RapidOCR artifacts so OCR
            auto-trigger works on image-only pages offline.
        with_easyocr: If True, fetch EasyOCR artifacts. Default False — RapidOCR
            is the wired engine.

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
        with_tableformer_v2=with_tableformer_v2,
        with_code_formula=False,
        with_picture_classifier=False,
        with_smolvlm=with_smolvlm,
        with_granitedocling=False,
        with_granitedocling_mlx=False,
        with_smoldocling=False,
        with_smoldocling_mlx=False,
        with_granite_vision=False,
        with_granite_chart_extraction=False,
        with_rapidocr=with_rapidocr,
        with_easyocr=with_easyocr,
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
    enable_ocr: bool = True,
    tableformer_mode: str = "accurate",
    tableformer_do_cell_matching: bool = True,
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

    tracer = get_tracer()
    with tracer.span(
        "ingest.parse.docling",
        {
            "parser_model": parser_model,
            "vlm_mode": vlm_mode,
            "generate_page_images": generate_page_images,
            "source_path": str(source_path),
        },
    ) as _span:
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

        # Always build PdfPipelineOptions so OCR auto-trigger and TableFormer v2
        # selection apply regardless of vlm_mode. PdfFormatOption is wired into
        # converter_kwargs only when the import surface is available — keeps the
        # function robust against pruned docling installs.
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                RapidOcrOptions,
                TableFormerMode,
            )
            from docling.document_converter import PdfFormatOption

            pipeline_options = PdfPipelineOptions()

            # --- TableFormer mode (v2 = ACCURATE, v1 = FAST) -----------------
            mode = (
                TableFormerMode.ACCURATE
                if str(tableformer_mode).lower() == "accurate"
                else TableFormerMode.FAST
            )
            pipeline_options.table_structure_options.mode = mode
            pipeline_options.table_structure_options.do_cell_matching = bool(
                tableformer_do_cell_matching
            )

            # --- OCR auto-trigger (image-only pages) -------------------------
            pipeline_options.do_ocr = bool(enable_ocr)
            if enable_ocr:
                pipeline_options.ocr_options = RapidOcrOptions(
                    force_full_page_ocr=False
                )

            # --- Built-in SmolVLM picture description ------------------------
            if vlm_mode == "builtin":
                try:
                    from docling.datamodel.pipeline_options import (
                        PictureDescriptionVlmEngineOptions,
                    )

                    pipeline_options.do_picture_description = True
                    pipeline_options.picture_description_options = (
                        PictureDescriptionVlmEngineOptions.from_preset("smolvlm")
                    )
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "vlm_mode='builtin' requested but SmolVLM setup failed "
                        "(%s); proceeding without picture description.",
                        exc,
                    )

            converter_kwargs["format_options"] = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        except Exception as exc:
            # If the docling pipeline_options surface is unavailable, fall back to
            # default DocumentConverter — preserves backward compatibility.
            logging.getLogger(__name__).warning(
                "Docling PdfPipelineOptions unavailable (%s); using defaults.", exc
            )

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

        _span.set_attribute("page_count", page_count)
        _span.set_attribute("figure_count", len(figures))
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


def _contextualized_text(chunker: Any, raw: Any) -> str:
    """Embedded text for a HybridChunker chunk WITH its heading breadcrumb.

    Docling's ``HybridChunker.contextualize()`` prepends the section heading
    path to the chunk body, so the EMBEDDED text carries its context. Without
    it (``text=raw.text``) body chunks embed heading-less and lose dense
    retrieval to short title/heading-only chunks on topical queries. Falls back
    to the raw text if contextualize is unavailable or yields nothing.
    """
    try:
        ctx = chunker.contextualize(chunk=raw)
        if isinstance(ctx, str) and ctx.strip():
            return ctx
    except Exception:  # noqa: BLE001 — never fail ingest over contextualization
        pass
    return raw.text


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
      3. Run HybridChunker with an explicit HuggingFaceTokenizer so
         ``merge_peers=True`` consolidates sentence-per-line markdown into
         token-budgeted chunks. The tokenizer model is resolved in this order:
           1. ``config.hybrid_chunker_tokenizer_model`` if set.
           2. ``EMBEDDING_MODEL_PATH`` (so the chunker's token accounting
              matches the embedder).
           3. Final fallback ``"BAAI/bge-m3"``.

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

    tracer = get_tracer()
    with tracer.span(
        "ingest.chunk.docling",
        {
            "strategy": "hybrid",
            "max_tokens": max_tokens,
            "input_chars": len(markdown_text),
        },
    ) as _span:
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

      model_id = _resolve_tokenizer_model_id(config)
      try:
          tokenizer = _get_or_build_tokenizer(model_id, max_tokens=max_tokens)
          chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
      except Exception as exc:  # pragma: no cover — env-dependent
          # Tokenizer load failed (no network, model not in HF cache, etc.).
          # Re-raise so the caller's existing fallback path
          # (parser_text.chunk → chunk_with_markdown) can handle it.
          logger.warning(
              "HybridChunker tokenizer load failed for model_id=%s: %s; "
              "caller should fall back to char-splitter chunking.",
              model_id, exc,
          )
          raise
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
                  text=_contextualized_text(chunker, raw),
                  section_path=" > ".join(headings),
                  heading=headings[-1] if headings else "",
                  heading_level=len(headings),
                  chunk_index=idx,
                  extra_metadata=extra,
                  heading_path=list(headings),
                  page_ref=_page_ref_from_chunk_meta(meta),
              )
          )
      # Apply the native min-body coalesce floor (HybridChunker's merge_peers
      # has none) so sub-floor prose stubs are folded into a same-ancestor
      # neighbour rather than emitted as heading-dominated tiny chunks. Re-index
      # afterwards since merging removes chunks.
      chunks = _coalesce_native_chunks(
          chunks,
          min_chars=int(getattr(config, "native_min_chunk_chars", 512) or 0),
          max_chars=int(getattr(config, "chunk_size", 3500) or 3500),
      )
      for _i, _c in enumerate(chunks):
          _c.chunk_index = _i
      _span.set_attribute("chunk_count", len(chunks))
      _span.set_attribute("figure_count", len(figures))
      return chunks


# ---------------------------------------------------------------------------
# DoclingParser — DocumentParser protocol implementation (FR-3221–FR-3224)
# ---------------------------------------------------------------------------

def _page_ref_from_chunk_meta(meta: Any) -> Any:
    """Derive a PageRef from a HybridChunker chunk's meta.doc_items provenance.

    Uses the first doc_item's first provenance entry as the chunk's primary
    page (HybridChunker preserves document order). Returns None when meta is
    missing or no provenance is attached.
    """
    from src.ingest.support.parser_base import PageRef

    if meta is None:
        return None
    doc_items = getattr(meta, "doc_items", None) or []
    for item in doc_items:
        prov = getattr(item, "prov", None) or []
        for p in prov:
            page_no = getattr(p, "page_no", None)
            if page_no is None:
                continue
            return PageRef(
                page_no=int(page_no),
                page_label="",
                bbox=_decode_docling_bbox(getattr(p, "bbox", None)),
            )
    return None


def _decode_docling_bbox(bbox_obj: Any) -> tuple[float, float, float, float] | None:
    """Decode a Docling BoundingBox into a ``(x0, y0, x1, y1)`` tuple.

    Tolerates both the ``l/t/r/b`` and ``x0/y0/x1/y1`` attribute shapes that
    different Docling versions expose. Returns ``None`` when the object is
    missing or decoding raises.
    """
    if bbox_obj is None:
        return None
    try:
        return (
            float(getattr(bbox_obj, "l", getattr(bbox_obj, "x0", 0.0))),
            float(getattr(bbox_obj, "t", getattr(bbox_obj, "y0", 0.0))),
            float(getattr(bbox_obj, "r", getattr(bbox_obj, "x1", 0.0))),
            float(getattr(bbox_obj, "b", getattr(bbox_obj, "y1", 0.0))),
        )
    except Exception:
        return None


def _page_ref_from_table_item(tbl: Any) -> Any:
    """Derive a PageRef (including bbox) from a Docling TableItem's provenance.

    Returns ``None`` when no provenance entry carries a page number. The bbox
    is decoded via :func:`_decode_docling_bbox` so table-summary chunks carry
    the same citation provenance as text chunks (defect from the real-datasheet
    soak: pre-fix this hardcoded ``bbox=None``).
    """
    from src.ingest.support.parser_base import PageRef

    prov = getattr(tbl, "prov", None) or []
    for p in prov:
        page_no = getattr(p, "page_no", None)
        if page_no is None:
            continue
        return PageRef(
            page_no=int(page_no),
            page_label="",
            bbox=_decode_docling_bbox(getattr(p, "bbox", None)),
        )
    return None


_HEADING_LABEL_VALUES = frozenset({"section_header", "title"})

# Labels whose self_ref → section_path mapping the heading-stack walker
# records. Tables and figures (pictures, charts) both need section provenance
# resolved by replaying iterate_items() because Docling stores headings and
# floats as flat siblings under ``body``.
_CAPTIONABLE_LABEL_VALUES = frozenset({"table", "picture", "chart"})


def _label_value(item: Any) -> str:
    """Return the lowercase string value of an item's ``label`` attribute.

    Works whether ``label`` is a Docling ``DocItemLabel`` enum, a plain string,
    or absent. Returns ``""`` when no label is present.
    """
    label = getattr(item, "label", None)
    if label is None:
        return ""
    value = getattr(label, "value", label)
    try:
        return str(value).lower()
    except Exception:  # pragma: no cover - defensive
        return ""


def _resolve_table_section_paths(docling_document: Any) -> dict[str, str]:
    """Map each table's ``self_ref`` to its ancestor heading breadcrumb.

    Walks ``docling_document.iterate_items()`` in document order, maintaining
    a heading stack keyed by the heading's ``level`` (``TitleItem`` is treated
    as level 0). When a ``TableItem`` is encountered, the current stack is
    joined with ``" > "`` and recorded against the table's ``self_ref``.

    Returns an empty mapping when ``iterate_items`` is unavailable or raises —
    callers must fall back to ``section_path=""`` in that case.
    """
    paths: dict[str, str] = {}
    iterate = getattr(docling_document, "iterate_items", None)
    if not callable(iterate):
        return paths

    # Materialize ``iterate_items()`` once so we can take two passes without
    # double-consuming a generator (and to keep test fixtures that hand back a
    # single iterator working). Real Docling produces O(N) items per document
    # so the memory cost is bounded by the parse itself.
    try:
        entries = list(iterate())
    except Exception:  # pragma: no cover - defensive
        return paths

    # First pass: collect distinct heading levels and total heading count to
    # detect "flat outline" documents (real-world datasheets often emit dozens
    # of chapter headings all at level=1). On flat outlines we disable the
    # ``had_content`` implicit-nesting heuristic and treat same-level headings
    # as siblings unconditionally — otherwise the heuristic builds a 71-deep
    # ancestor chain out of unrelated chapters.
    flat_outline = False
    try:
        levels_seen: set[int] = set()
        heading_count = 0
        for entry in entries:
            item = entry[0] if isinstance(entry, tuple) and len(entry) >= 1 else entry
            if _label_value(item) in _HEADING_LABEL_VALUES:
                lvl = int(getattr(item, "level", 0) or 0)
                if _label_value(item) == "title":
                    lvl = 0
                levels_seen.add(lvl)
                heading_count += 1
        # Flat-outline gate: ≥4 headings AND at most one distinct .level.
        # Below 4 headings we keep W's implicit-nesting heuristic — that's
        # the regime where Docling's font-size level quirk shows up in
        # crafted/short documents (W's existing tests cover it).
        if heading_count >= 4 and len(levels_seen) <= 1:
            flat_outline = True
    except Exception:  # pragma: no cover - defensive
        flat_outline = False

    # Stack entries: (level:int, text:str, had_content:bool). Lower level ==
    # shallower heading. ``had_content`` tracks whether any non-heading body
    # item appeared after this heading was pushed — used to disambiguate
    # logical parent/child vs sibling-replacement when Docling assigns the
    # same ``.level`` to both.
    stack: list[list[Any]] = []
    walker = entries

    try:
        for entry in walker:
            # iterate_items yields (node, depth). Tolerate alternate shapes.
            if isinstance(entry, tuple) and len(entry) >= 1:
                item = entry[0]
            else:
                item = entry

            label = _label_value(item)
            if label in _HEADING_LABEL_VALUES:
                # TitleItem has no .level; treat as level 0 (root).
                level = int(getattr(item, "level", 0) or 0)
                if label == "title":
                    level = 0
                text = str(getattr(item, "text", "") or "")
                if not text:
                    continue
                # Pop entries strictly deeper than the new heading. A deeper
                # heading existing under a parent implies the parent "had
                # content" (its sub-section), so mark the next surviving
                # frame accordingly.
                popped_deeper = False
                while stack and stack[-1][0] > level:
                    stack.pop()
                    popped_deeper = True
                if popped_deeper and stack:
                    stack[-1][2] = True
                # Same-level handling. Two competing realities:
                #   1. Docling sometimes assigns identical ``.level`` to a
                #      logical parent/child pair (font-size based heading
                #      detection). When no content intervenes we tolerate
                #      ONE such implicit nesting — see project memory
                #      ``project_docling_section_path_collapse``.
                #   2. On flat-outline PDFs (datasheets), dozens of unrelated
                #      chapter headings carry the same ``.level`` with no body
                #      items between them — page breaks, captions, etc. don't
                #      register as content. These are siblings, not a 71-deep
                #      ancestor chain. Cap implicit nesting at one frame.
                if stack and stack[-1][0] == level:
                    same_level_run = sum(1 for f in stack if f[0] == level)
                    top_had_content = stack[-1][2]
                    # Flat-outline docs: unconditional sibling replace.
                    # Otherwise: replace if true sibling (had content) OR we
                    # have already implicitly nested once at this level
                    # (cap implicit nesting at one frame).
                    if flat_outline or top_had_content or same_level_run >= 2:
                        stack.pop()
                stack.append([level, text, False])
                # Belt-and-suspenders sanity cap: keep the deepest frames
                # (most specific context wins). Catches pathological inputs
                # the heuristics above don't anticipate.
                _MAX_SECTION_DEPTH = 8
                if len(stack) > _MAX_SECTION_DEPTH:
                    del stack[: len(stack) - _MAX_SECTION_DEPTH]
            elif label in _CAPTIONABLE_LABEL_VALUES:
                # Tables, pictures, and charts all need section_path resolved
                # against the live heading stack. They also count as body
                # content for their enclosing heading(s).
                self_ref = getattr(item, "self_ref", None)
                if self_ref:
                    paths[str(self_ref)] = " > ".join(t for _, t, _ in stack)
                for frame in stack:
                    frame[2] = True
            else:
                # Any other body item (paragraph, list, code, figure, ...)
                # counts as content for the active heading stack.
                if label:
                    for frame in stack:
                        frame[2] = True
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("section_path walker aborted: %s", exc)

    return paths


def _extract_table_artifacts(docling_document: Any, document_id: str = "") -> list:
    """Extract structured table artifacts from a DoclingDocument. FR-3211.

    Walks ``docling_document.tables`` (when present) and converts each entry
    into a ``TableArtifact`` with markdown, row-major cell grid, and section
    breadcrumbs derived from the table's parent heading chain.

    The section breadcrumb is resolved by replaying ``iterate_items()`` once
    and snapshotting the active heading stack at each table's position — the
    Docling object graph stores headings and tables as flat siblings under
    ``body``, so a structural parent walk would not surface the breadcrumb.

    Returns an empty list when the document has no tables or when extraction
    of any individual table fails — table extraction must never break parsing.
    """
    from src.ingest.support.parser_base import TableArtifact

    if docling_document is None:
        return []

    raw_tables = getattr(docling_document, "tables", None) or []
    section_paths = _resolve_table_section_paths(docling_document)

    artifacts: list[TableArtifact] = []
    for idx, tbl in enumerate(raw_tables):
        try:
            cells = _table_to_cells(tbl)
            if not cells:
                continue
            num_rows = len(cells)
            num_cols = max((len(row) for row in cells), default=0)
            try:
                md = tbl.export_to_markdown(doc=docling_document)
            except TypeError:
                # Older docling signatures
                md = tbl.export_to_markdown()
            except Exception:
                md = _cells_to_markdown(cells)
            caption = ""
            try:
                cap_text = tbl.caption_text(doc=docling_document)
                caption = str(cap_text or "")
            except Exception:
                caption = ""
            self_ref = getattr(tbl, "self_ref", None)
            section_path = section_paths.get(str(self_ref), "") if self_ref else ""
            artifacts.append(
                TableArtifact(
                    table_id=f"table-{idx + 1}",
                    markdown=str(md or "").strip(),
                    cells=cells,
                    num_rows=num_rows,
                    num_cols=num_cols,
                    has_header=_detect_header_row(tbl),
                    section_path=section_path,
                    caption=caption.strip(),
                    caption_label=_extract_caption_label(caption),
                    page_ref=_page_ref_from_table_item(tbl),
                    self_ref=str(self_ref or ""),
                    document_id=document_id,
                )
            )
        # Narrow catch: Docling grid/heading/export internals raise these on malformed input; AssertionError + BaseException must propagate so our invariant bugs surface.
        except (AttributeError, KeyError, TypeError, IndexError, ValueError, RuntimeError) as exc:
            table_id = f"table-{idx + 1}"
            self_ref = getattr(tbl, "self_ref", None)
            logger.warning(
                "table extraction skipped table_id=%s self_ref=%s: %s",
                table_id, self_ref, exc,
            )
            continue
    return artifacts


def _figure_page_no(pic: Any) -> int:
    """Return the 1-based page number from a Docling PictureItem's provenance.

    Returns ``0`` when no provenance entry carries a page number. We don't
    surface a full PageRef here (figures' bbox typically frames the image
    region, which downstream callers don't yet consume) — page_no is the
    minimum useful citation provenance for xref expansion.
    """
    prov = getattr(pic, "prov", None) or []
    for p in prov:
        page_no = getattr(p, "page_no", None)
        if page_no is None:
            continue
        try:
            return int(page_no)
        except (TypeError, ValueError):
            continue
    return 0


def _figure_image_uri(pic: Any) -> str:
    """Best-effort image URI from a Docling PictureItem's ``image`` ref.

    Docling 2.82.0 attaches an :class:`ImageRef` whose ``uri`` is either a
    ``data:`` URI (when ``generate_picture_images=True``) or a ``file://``
    path. Returns ``""`` when no image ref is present — common on parses
    that didn't enable picture image generation.
    """
    image = getattr(pic, "image", None)
    if image is None:
        return ""
    uri = getattr(image, "uri", None)
    if uri is None:
        return ""
    try:
        return str(uri)
    except Exception:  # pragma: no cover - defensive
        return ""


def _build_unbound_caption_fallback(doc: Any) -> dict[str, str]:
    """Map ``picture.self_ref`` → forward-walked caption text when Docling
    failed to bind a caption to the PictureItem.

    Docling 2.82.0 frequently parses a figure's caption as an ordinary
    TextItem in the body stream rather than binding it via ``pic.captions``.
    Triage of the 2026-05-24 ESP32-S3 soak found 4 of 5 unresolvable prose
    figure refs were caused by this binding gap (memory:
    ``project_docling_heading_provenance`` documents the table-side variant
    of the same quirk).

    Strategy: walk ``iterate_items()`` in document order. For each
    PictureItem with empty ``captions``, scan forward until either
      * we hit a TextItem on the same page whose text matches the canonical
        figure-caption regex (``^\\s*(Figure|Fig\\.?)\\s+N(-N|.N)?``) — adopt
        it; or
      * we leave the page or hit another PictureItem — abort (no fallback).

    Stays page-scoped so we never invent cross-page provenance. Returns an
    empty mapping when ``iterate_items`` is unavailable.
    """
    fallback: dict[str, str] = {}
    iterate = getattr(doc, "iterate_items", None)
    if not callable(iterate):
        return fallback
    try:
        entries = list(iterate())
    except Exception:  # pragma: no cover - defensive
        return fallback

    # Flatten (item, level) -> item with index for forward scans.
    items: list[Any] = []
    for entry in entries:
        if isinstance(entry, tuple) and len(entry) >= 1:
            items.append(entry[0])
        else:
            items.append(entry)

    def _page_of(it: Any) -> int:
        prov = getattr(it, "prov", None) or []
        for p in prov:
            page_no = getattr(p, "page_no", None)
            if page_no is None:
                continue
            try:
                return int(page_no)
            except (TypeError, ValueError):
                continue
        return 0

    for i, item in enumerate(items):
        if _label_value(item) != "picture":
            continue
        if getattr(item, "captions", None):
            continue  # Docling bound a caption; trust it.
        self_ref = getattr(item, "self_ref", None)
        if not self_ref:
            continue
        pic_page = _page_of(item)
        if pic_page == 0:
            continue  # no provenance to scope by; bail rather than guess
        # Forward scan, bounded by page change or next PictureItem.
        for j in range(i + 1, len(items)):
            nxt = items[j]
            nxt_label = _label_value(nxt)
            if nxt_label == "picture":
                break
            nxt_page = _page_of(nxt)
            if nxt_page and nxt_page != pic_page:
                break
            txt = str(getattr(nxt, "text", "") or "").strip()
            if not txt:
                continue
            if _FIGURE_CAPTION_LABEL_RE.match(txt):
                fallback[str(self_ref)] = txt
                break
    return fallback


def _extract_figure_artifacts(doc: Any, document_id: str = "") -> list:
    """Extract structured figure artifacts from a DoclingDocument.

    Walks ``docling_document.pictures`` (Docling 2.82.0 surfaces both
    PICTURE and CHART items under this attribute) and converts each entry
    into a :class:`FigureArtifact` with caption text, normalised caption
    label, section breadcrumb, page number, and best-effort image URI.

    The section breadcrumb is resolved by reusing the same heading-stack
    replay as ``_extract_table_artifacts`` — Docling stores headings and
    pictures as flat siblings under ``body``, so a structural parent walk
    would surface the body group rather than the enclosing heading
    (see memory ``project_docling_heading_provenance``).

    Returns an empty list when the document has no pictures or when
    extraction of any individual picture fails — figure extraction must
    never break parsing.
    """
    from src.ingest.support.parser_base import FigureArtifact

    if doc is None:
        return []

    raw_pictures = getattr(doc, "pictures", None) or []
    # Reuses the captionable-label walker; the returned dict carries entries
    # for tables, pictures, AND charts so the same call serves both extractors.
    section_paths = _resolve_table_section_paths(doc)
    # Forward-walked fallback for PictureItems whose ``pic.captions`` is
    # empty even though a "Figure N-N. ..." TextItem sits on the same page.
    # Docling 2.82.0 leaves these unbound for ~80% of figures on real-world
    # PDFs (see FIG-4 triage). Empty dict when iterate_items is unavailable
    # — extractor degrades to the pre-FIG-4 behaviour in that case.
    caption_fallback = _build_unbound_caption_fallback(doc)

    artifacts: list[FigureArtifact] = []
    for idx, pic in enumerate(raw_pictures):
        try:
            caption = ""
            try:
                # Defensive: Docling 2.82.0 has known API quirks; tolerate
                # captions= absence/empty by routing through caption_text.
                if getattr(pic, "captions", None):
                    cap_text = pic.caption_text(doc=doc)
                    caption = str(cap_text or "")
            except Exception:
                caption = ""
            self_ref = getattr(pic, "self_ref", None)
            # Forward-walk fallback when Docling didn't bind a caption.
            if not caption and self_ref:
                caption = caption_fallback.get(str(self_ref), "")
            section_path = section_paths.get(str(self_ref), "") if self_ref else ""
            artifacts.append(
                FigureArtifact(
                    document_id=document_id,
                    caption=caption.strip(),
                    caption_label=_extract_figure_caption_label(caption),
                    section_path=section_path,
                    page_no=_figure_page_no(pic),
                    self_ref=str(self_ref or ""),
                    image_uri=_figure_image_uri(pic),
                )
            )
        except (AttributeError, KeyError, TypeError, IndexError, ValueError, RuntimeError) as exc:
            figure_id = f"figure-{idx + 1}"
            self_ref = getattr(pic, "self_ref", None)
            logger.warning(
                "figure extraction skipped figure_id=%s self_ref=%s: %s",
                figure_id, self_ref, exc,
            )
            continue
    return artifacts


def _figure_artifacts_to_chunks(figures: list) -> list:
    """Turn :class:`FigureArtifact` rows into ``chunk_type="figure"`` chunks.

    Mirrors :func:`_apply_adaptive_table_chunking`'s table-side transformer:
    each chunk's ``extra_metadata`` carries the provenance the resolver
    filter and citation UI need (``document_id``, ``caption_label``,
    ``section_path``, ``page_no``, ``self_ref``, ``figure_image_uri``).

    Rules:
      * Skip figures with neither caption nor caption_label — there's no
        useful text to embed.
      * ``image_uri`` starting with ``data:`` is coerced to ``""`` so we
        don't bloat Weaviate with base64 payloads (memory
        ``project_weaviate_schema_drop`` reminds us the schema property
        list is the gate, but a multi-MB ``data:`` URI is still wasteful).
      * ``chunk_id`` is derived from ``(document_id, self_ref)`` via
        :func:`build_figure_chunk_id` so re-ingesting overwrites in place.
        Falls back to the parser's own chunk_id derivation when
        ``self_ref`` is empty (rare; defensive).
    """
    from src.ingest.support.parser_base import Chunk, PageRef
    from src.vector_db.weaviate.store import build_figure_chunk_id

    out: list = []
    for fig in figures or []:
        caption = (getattr(fig, "caption", "") or "").strip()
        label = (getattr(fig, "caption_label", "") or "").strip()
        if not caption and not label:
            continue
        if caption and label and caption.lower().startswith(label.lower()):
            # Caption already starts with the label (typical Docling case);
            # keep the caption verbatim so we don't double-stamp.
            text = caption
        elif caption and label:
            text = f"{label}: {caption}"
        elif label:
            text = label
        else:
            text = caption

        section_path = getattr(fig, "section_path", "") or ""
        heading_path = [h for h in section_path.split(" > ") if h]
        heading = heading_path[-1] if heading_path else ""

        image_uri = getattr(fig, "image_uri", "") or ""
        if image_uri.startswith("data:"):
            image_uri = ""

        document_id = getattr(fig, "document_id", "") or ""
        self_ref = getattr(fig, "self_ref", "") or ""
        if document_id and self_ref:
            chunk_id = build_figure_chunk_id(document_id, self_ref)
        else:
            chunk_id = ""

        page_no = int(getattr(fig, "page_no", 0) or 0)
        page_ref = PageRef(page_no=page_no, page_label="", bbox=None) if page_no else None

        meta: dict = {
            "chunk_type": "figure",
            "document_id": document_id,
            "caption_label": label,
            "self_ref": self_ref,
            "page_no": page_no,
            "figure_image_uri": image_uri,
        }
        if chunk_id:
            meta["chunk_id"] = chunk_id
        # Stamp cross-refs detected in the figure caption text (rare but
        # possible: "Figure 4-1: see Section 3.2 for the matrix").
        _stamp_xref_targets(meta, text)

        out.append(
            Chunk(
                text=text,
                section_path=section_path,
                heading=heading,
                heading_level=len(heading_path),
                chunk_index=0,
                extra_metadata=meta,
                heading_path=list(heading_path),
                page_ref=page_ref,
            )
        )
    return out


def _table_to_cells(tbl: Any) -> list[list[str]]:
    """Convert a Docling TableItem into a row-major list of strings."""
    data = getattr(tbl, "data", None)
    if data is None:
        return []
    grid = getattr(data, "grid", None)
    if not grid:
        return []
    rows: list[list[str]] = []
    for row in grid:
        rows.append([str(getattr(cell, "text", "") or "") for cell in row])
    return rows


def _detect_header_row(tbl: Any) -> bool:
    data = getattr(tbl, "data", None)
    grid = getattr(data, "grid", None) if data is not None else None
    if not grid:
        return False
    first_row = grid[0]
    for cell in first_row:
        if getattr(cell, "column_header", False) or getattr(cell, "row_header", False):
            return True
    return False


def _cells_to_markdown(cells: list[list[str]]) -> str:
    if not cells:
        return ""
    width = max(len(r) for r in cells)
    norm = [r + [""] * (width - len(r)) for r in cells]
    header = "| " + " | ".join(norm[0]) + " |"
    sep = "| " + " | ".join(["---"] * width) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in norm[1:])
    return "\n".join([header, sep, body]).strip()


def _table_signature(markdown: str) -> str:
    """Return the first non-empty, non-separator row of a markdown table.

    Mirrors the heuristic used by ``embedding.nodes.chunking._match_table_artifact``;
    duplicated locally to avoid an import dependency on the embedding node.
    """
    if not markdown:
        return ""
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("|---") or set(stripped) <= {"|", "-", " "}:
            continue
        return stripped
    return ""


def _is_table_dominant(chunk_text: str, table_md: str, signature: str) -> bool:
    """A chunk is "table-dominant" iff it contains the signature row AND the
    table markdown forms >=60% of the chunk text length.
    """
    if not signature or signature not in chunk_text:
        return False
    text_len = len(chunk_text)
    if text_len == 0:
        return False
    return (len(table_md) / text_len) >= 0.6


def _truncate_table_summary_text(text: str, max_chars: int) -> str:
    """Cap embedded ``table_summary`` text at ``max_chars``, appending a marker.

    Guards the embedder against pathologically wide table summaries (200+ row
    datasheets, hundreds of column headers) without altering the full
    ``table_markdown`` stored on metadata. Deterministic: same input ⇒ same
    output, so re-ingest stays idempotent and chunk-id formulas relying on text
    hashes do not drift.

    A non-positive ``max_chars`` disables truncation. When the input already
    fits, it is returned unchanged byte-for-byte (no marker appended).
    """
    if max_chars is None or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    # Marker template is fixed-length once the dropped-char count is known.
    # Reserve room so the final string length is exactly <= max_chars.
    marker_template = " … [truncated {n} chars]"
    # Iterate at most twice: marker length depends on the digit count of n.
    keep = max_chars - len(marker_template.format(n=len(text)))
    if keep < 0:
        # Cap so tiny it cannot fit even the marker — return marker-only,
        # truncated to max_chars. Edge case; preserves the contract.
        return marker_template.format(n=len(text))[:max_chars].rstrip()
    dropped = len(text) - keep
    marker = marker_template.format(n=dropped)
    return text[:keep] + marker


def _build_table_summary_text(tbl: Any) -> str:
    headers = tbl.cells[0] if tbl.has_header and tbl.cells else []
    body_rows = max(0, tbl.num_rows - (1 if tbl.has_header else 0))
    lines: list[str] = []
    caption = (tbl.caption or "").strip()
    if caption:
        lines.append(f"Table: {caption}")
    lines.append(f"Columns: {' | '.join(headers)}")
    lines.append(f"Rows: {body_rows}")
    return "\n".join(lines)


def _classify_table_for_row_chunks(
    tbl: Any, max_rows: int, max_cols: int
) -> str:
    """Classify a table's eligibility for row-chunk emission.

    Returns one of:
        ``"row_chunks"``  — table passes both gates; row chunks should be emitted.
        ``"small_gate"``  — table failed the small/shape gate (no header, empty,
                            too many rows/cols, or out-of-range col count).
        ``"uniform_gate"`` — table cleared the small gate but failed the uniform
                            gate (body rows do not match header width).

    The two gates are kept independent on purpose (see
    ``feedback_heuristic_gates_independent.md``) so observability can attribute
    summary-only outcomes to the correct failure mode.
    """
    if not tbl.has_header or not tbl.cells:
        return "small_gate"
    body = tbl.cells[1:]
    body_rows = len(body)
    if body_rows == 0:
        return "small_gate"
    if body_rows > max_rows:
        return "small_gate"
    if tbl.num_cols < 2 or tbl.num_cols > max_cols:
        return "small_gate"
    header_len = len(tbl.cells[0])
    if not all(len(r) == header_len for r in body):
        return "uniform_gate"
    return "row_chunks"


def _table_is_small_uniform(tbl: Any, max_rows: int, max_cols: int) -> bool:
    """Backward-compatible bool wrapper around :func:`_classify_table_for_row_chunks`."""
    return _classify_table_for_row_chunks(tbl, max_rows, max_cols) == "row_chunks"


def _apply_adaptive_table_chunking(
    chunks: list, tables: list, cfg: Any
) -> list:
    """Drop table-dominant chunks; emit summary + (optional) row chunks per table.

    Inserts emitted chunks at the position where the first table-dominant chunk
    appeared so document order is preserved. Falls back to appending when the
    table-dominant chunk is absent (e.g., HybridChunker did not surface it).
    """
    from src.ingest.support.parser_base import Chunk
    from src.ingest.support.table_metrics import (
        increment_row_chunks_emitted,
        increment_summary_only_small,
        increment_summary_only_uniform,
        increment_summary_text_truncated,
    )

    max_rows = int(getattr(cfg, "max_table_rows_for_row_chunks", 32))
    max_cols = int(getattr(cfg, "max_table_cols_for_row_chunks", 12))
    # Per-summary-text cap (chars). 0 / unset disables truncation.
    summary_max_chars = int(getattr(cfg, "table_summary_max_chars", 0) or 0)

    # Compute table signatures once.
    table_meta: list[tuple[Any, str]] = []
    for tbl in tables:
        md = getattr(tbl, "markdown", "") or ""
        sig = _table_signature(md)
        table_meta.append((tbl, sig))

    # Mark indices for removal and compute insertion anchor per table.
    drop_indices: set[int] = set()
    insertion_anchor: dict[str, int] = {}
    for tbl, sig in table_meta:
        if not sig:
            continue
        md = tbl.markdown or ""
        for i, c in enumerate(chunks):
            if i in drop_indices:
                continue
            if _is_table_dominant(c.text, md, sig):
                drop_indices.add(i)
                insertion_anchor.setdefault(tbl.table_id, i)
                # Mark only the first match per table; let other matches remain
                # so chained tables on a page still get their own anchors.
                break

    def _make_table_chunks(tbl: Any) -> list:
        out: list = []
        heading_path = [h for h in (tbl.section_path or "").split(" > ") if h]
        heading = heading_path[-1] if heading_path else ""
        # Stable within-document group key so retrieval can join a winning row
        # chunk back to its summary (and vice versa) without fuzzy-matching on
        # heading_path + page. Prefer parser self_ref; fall back to table_id.
        group_id = tbl.self_ref or tbl.table_id
        common_meta_summary = {
            "chunk_type": "table_summary",
            "table_id": tbl.table_id,
            "table_group_id": group_id,
            "document_id": getattr(tbl, "document_id", "") or "",
            "caption_label": getattr(tbl, "caption_label", "") or "",
            "table_num_rows": tbl.num_rows,
            "table_num_cols": tbl.num_cols,
            "table_has_header": tbl.has_header,
            # Stash full markdown on the summary chunk so retrieval can
            # reconstruct the table without re-parsing the source PDF.
            "table_markdown": getattr(tbl, "markdown", "") or "",
        }
        if tbl.caption:
            common_meta_summary["table_caption"] = tbl.caption
        raw_summary = _build_table_summary_text(tbl)
        truncated_summary = _truncate_table_summary_text(raw_summary, summary_max_chars)
        if truncated_summary != raw_summary:
            increment_summary_text_truncated()
        # Stamp cross-refs detected in the table summary text (e.g. a
        # caption like "Table 5-2 (see §3.1)" carries a §-section ref).
        _stamp_xref_targets(common_meta_summary, truncated_summary)
        out.append(
            Chunk(
                text=truncated_summary,
                section_path=tbl.section_path or "",
                heading=heading,
                heading_level=len(heading_path),
                chunk_index=0,
                extra_metadata=common_meta_summary,
                heading_path=heading_path,
                page_ref=tbl.page_ref,
            )
        )

        gate = _classify_table_for_row_chunks(tbl, max_rows, max_cols)
        if gate == "small_gate":
            increment_summary_only_small()
        elif gate == "uniform_gate":
            increment_summary_only_uniform()
        else:
            increment_row_chunks_emitted()

        if gate == "row_chunks":
            headers = tbl.cells[0]
            caption_prefix = (tbl.caption or "").strip()
            for body_idx, row in enumerate(tbl.cells[1:]):
                pairs = " | ".join(f"{h}: {v}" for h, v in zip(headers, row))
                text = f"{caption_prefix}\n{pairs}" if caption_prefix else pairs
                row_meta = {
                    "chunk_type": "table_row",
                    "table_id": tbl.table_id,
                    "table_group_id": group_id,
                    "document_id": getattr(tbl, "document_id", "") or "",
                    "caption_label": getattr(tbl, "caption_label", "") or "",
                    "table_row_index": body_idx,
                    "table_num_rows": tbl.num_rows,
                    "table_num_cols": tbl.num_cols,
                }
                # Row-level xref edges: most row bodies have none, but the
                # caption-prefixed text occasionally references a section.
                _stamp_xref_targets(row_meta, text)
                out.append(
                    Chunk(
                        text=text,
                        section_path=tbl.section_path or "",
                        heading=heading,
                        heading_level=len(heading_path),
                        chunk_index=0,
                        extra_metadata=row_meta,
                        heading_path=list(heading_path),
                        page_ref=tbl.page_ref,
                    )
                )
        return out

    # Assemble result preserving order: walk original chunks, drop marked ones,
    # and insert per-table emissions at the first occurrence anchor.
    inserted_for: set[str] = set()
    result: list = []
    for i, c in enumerate(chunks):
        if i in drop_indices:
            # Find which table anchored here and emit its chunks once.
            for tbl, _sig in table_meta:
                if (
                    tbl.table_id not in inserted_for
                    and insertion_anchor.get(tbl.table_id) == i
                ):
                    result.extend(_make_table_chunks(tbl))
                    inserted_for.add(tbl.table_id)
            continue
        result.append(c)

    # Tables that never matched a chunk — append their emissions at the end.
    for tbl, _sig in table_meta:
        if tbl.table_id not in inserted_for:
            result.extend(_make_table_chunks(tbl))
            inserted_for.add(tbl.table_id)

    return result


def _shared_heading_prefix(a: list, b: list) -> list:
    """Return the longest common leading run of two heading paths.

    ``["Ch3","3.1"]`` & ``["Ch3","3.2"]`` -> ``["Ch3"]`` (sibling sections share
    their chapter); ``["Ch3"]`` & ``["Ch4"]`` -> ``[]`` (different chapters).
    Identical paths return the full path (no specificity lost on intra-section
    merges).
    """
    out: list = []
    for x, y in zip(a or [], b or []):
        if x == y:
            out.append(x)
        else:
            break
    return out


def _coalesce_native_chunks(chunks: list, min_chars: int, max_chars: int) -> list:
    """Merge sub-floor HybridChunker chunks into a same-ancestor neighbour.

    HybridChunker's ``merge_peers`` only merges chunks sharing the SAME heading
    path and only up to the token budget — it has no minimum-size floor. So a
    leaf section with a tiny body (a one-line cross-pointer, a stub paragraph)
    is emitted as its own heading-dominated chunk that pollutes retrieval (the
    "title-only chunk" pathology). The chunks that survive small are precisely
    those whose neighbour has a DIFFERENT heading (merge_peers stops at heading
    boundaries), so coalescing here must merge ACROSS heading boundaries.

    Rule: walk in document order; when the current chunk OR the last emitted
    chunk has a sub-``min_chars`` BODY (breadcrumb stripped via
    :func:`chunk_body_text`), and the two share a non-empty heading-path prefix,
    and the merged body stays within ``max_chars``, fold the current chunk into
    the previous one. The merged chunk is re-attributed to the shared ancestor
    breadcrumb (honest when content spans sections; a no-op when the paths are
    identical), keeps the earlier chunk's page provenance, and its xref edges
    are re-stamped from the merged body. Chunks with no shared ancestor (a stub
    under a different chapter) are left intact for the body-aware DROP backstop
    in ``quality_validation``.

    Merges PROSE chunks only — call AFTER :func:`_apply_adaptive_table_chunking`
    and figure-chunk emission. table_summary/table_row/figure chunks carry
    structured metadata that string concatenation would destroy, so they are
    skipped (neither merged nor merged-into); a table/figure chunk sitting
    between two prose chunks therefore also blocks them from merging across it,
    preserving document structure. Running after table chunking is essential —
    coalescing before it could fold a small table-dominant chunk (or its
    adjacent prose) into a neighbour and break or drop the table emission.
    ``min_chars <= 0`` disables.
    """
    from src.ingest.common.shared import chunk_body_text

    if min_chars <= 0 or len(chunks) < 2:
        return chunks

    def _body(c: Any) -> str:
        return chunk_body_text(c.text, list(getattr(c, "heading_path", []) or []))

    def _is_prose(c: Any) -> bool:
        # Prose chunks carry no chunk_type yet (it is set later in chunking_node)
        # or an explicit "text"; table_summary/table_row/figure chunks must not
        # be merged or merged-into.
        ct = (getattr(c, "extra_metadata", None) or {}).get("chunk_type", "text")
        return ct in ("text", "", None)

    out: list = []
    for ch in chunks:
        if out and _is_prose(ch) and _is_prose(out[-1]):
            prev = out[-1]
            prev_body = _body(prev)
            cur_body = _body(ch)
            if (
                len(prev_body.strip()) < min_chars
                or len(cur_body.strip()) < min_chars
            ):
                shared = _shared_heading_prefix(
                    list(getattr(prev, "heading_path", []) or []),
                    list(getattr(ch, "heading_path", []) or []),
                )
                merged_body = (
                    f"{prev_body}\n{cur_body}"
                    if prev_body and cur_body
                    else (prev_body or cur_body)
                )
                if shared and len(merged_body) <= max_chars:
                    prefix = "\n".join(shared) + "\n" if shared else ""
                    prev.text = prefix + merged_body
                    prev.section_path = " > ".join(shared)
                    prev.heading = shared[-1] if shared else ""
                    prev.heading_level = len(shared)
                    prev.heading_path = list(shared)
                    extra: dict = {}
                    _stamp_xref_targets(extra, merged_body)
                    figs = list(prev.extra_metadata.get("figures", [])) + list(
                        ch.extra_metadata.get("figures", [])
                    )
                    if figs:
                        extra["figures"] = figs
                    prev.extra_metadata = extra
                    continue
        out.append(ch)
    return out


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
        # Aligns with parse-time fallback (line ~751) and the package-wide default
        # (config.settings.RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS = 1024). Used
        # only on the rare path where chunk() runs before parse() (in tests).
        self._max_tokens: int = RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS
        self._config: Any = None

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
        self._max_tokens = getattr(config, "hybrid_chunker_max_tokens", 1024)
        self._config = config

        result = parse_with_docling(
            file_path,
            parser_model=config.docling_model,
            artifacts_path=config.docling_artifacts_path,
            vlm_mode=self._vlm_mode,
            generate_page_images=config.generate_page_images,
            enable_ocr=getattr(config, "enable_ocr", True),
            tableformer_mode=getattr(config, "tableformer_mode", "accurate"),
            tableformer_do_cell_matching=getattr(
                config, "tableformer_do_cell_matching", True
            ),
        )

        # Encapsulate DoclingDocument — FR-3205
        self._docling_document = result.docling_document

        tables = _extract_table_artifacts(
            self._docling_document, document_id=file_path.stem
        )
        figures = _extract_figure_artifacts(
            self._docling_document, document_id=file_path.stem
        )

        return ParseResult(
            markdown=result.text_markdown,
            headings=result.headings,
            has_figures=result.has_figures,
            page_count=result.page_count,
            tables=tables,
            figures=figures,
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

        model_id = _resolve_tokenizer_model_id(getattr(self, "_config", None))
        try:
            tokenizer = _get_or_build_tokenizer(
                model_id, max_tokens=self._max_tokens
            )
            chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
        except Exception as exc:  # pragma: no cover — env-dependent
            logger.warning(
                "HybridChunker tokenizer load failed (model_id=%s): %s; "
                "raising for caller fallback.",
                model_id, exc,
            )
            raise
        chunk_iter = chunker.chunk(dl_doc=self._docling_document)
        raw_chunks = list(chunk_iter)

        chunks: list[Chunk] = []
        for raw in raw_chunks:
            # Extract heading hierarchy from HybridChunker metadata
            headings: list[str] = []
            meta = getattr(raw, "meta", None)
            if meta is not None:
                headings = list(getattr(meta, "headings", None) or [])

            heading = headings[-1] if headings else ""
            section_path = " > ".join(headings)
            heading_level = len(headings)
            page_ref = _page_ref_from_chunk_meta(meta)

            extra_metadata: dict = {}
            # Stamp cross-reference edges so retrieval-time expansion can
            # walk them without re-scanning the body text. JSON-encoded
            # list[{type,value}]; empty list when no refs present.
            _stamp_xref_targets(extra_metadata, raw.text)

            chunks.append(
                Chunk(
                    text=_contextualized_text(chunker, raw),
                    section_path=section_path,
                    heading=heading,
                    heading_level=heading_level,
                    chunk_index=0,  # re-sequenced below
                    extra_metadata=extra_metadata,
                    heading_path=headings,
                    page_ref=page_ref,
                )
            )

        cfg = getattr(self, "_config", None)
        tables = list(getattr(parse_result, "tables", []) or [])
        if (
            tables
            and getattr(cfg, "enable_adaptive_table_chunking", True)
        ):
            chunks = _apply_adaptive_table_chunking(chunks, tables, cfg)

        # Append figure chunks. We don't try to insert them at a position in
        # the prose stream — figures lack the row-fingerprint that lets
        # ``_apply_adaptive_table_chunking`` anchor on a matching text chunk.
        # Defensive ``getattr`` so ParseResult shapes that pre-date FIG-1
        # don't trip us (memory ``project_rag_chain_defensive_init``).
        figures = list(getattr(parse_result, "figures", []) or [])
        if figures:
            chunks = list(chunks) + _figure_artifacts_to_chunks(figures)

        # Coalesce sub-floor PROSE chunks AFTER table/figure emission. The pass
        # skips table_summary/table_row/figure chunks (so a small table-dominant
        # chunk can never be folded into a neighbour and lost), and merges only
        # adjacent prose stubs that HybridChunker's merge_peers left undersized
        # at heading boundaries — the "title-only chunk" floor that merge_peers
        # lacks. ``native_min_chunk_chars`` (default 512) gates it; 0 disables.
        chunks = _coalesce_native_chunks(
            chunks,
            min_chars=int(getattr(cfg, "native_min_chunk_chars", 512) or 0),
            max_chars=int(getattr(cfg, "chunk_size", 3500) or 3500),
        )

        for i, c in enumerate(chunks):
            c.chunk_index = i
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
            with_tableformer_v2=(
                str(getattr(config, "tableformer_mode", "accurate")).lower()
                == "accurate"
            ),
            with_rapidocr=bool(getattr(config, "enable_ocr", True)),
        )


# DEPRECATED standalone functions below — preserved for backward compatibility.
# Use DoclingParser class for new code.
# parse_with_docling()       — still available
# ensure_docling_ready()     — still available
# warmup_docling_models()    — still available
# DoclingParseResult         — still available
