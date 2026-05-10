#!/usr/bin/env python3
# @summary
# Developer CLI for ingesting documents into the RAG system.
# Exports: ingest, main
# Deps: config.settings, src.ingest, src.platform.validation
# @end-summary
"""Ingest documents into Weaviate and optionally build a knowledge graph."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import (
    DOCUMENTS_DIR,
    PROJECT_ROOT,
)
from src.ingest import (
    IngestionConfig,
    ingest_directory,
)
from src.platform import validate_documents_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("rag.ingest")


def ingest(
    documents_dir: Path = DOCUMENTS_DIR,
    fresh: bool = True,
    update: bool = False,
    build_kg: bool = True,
    obsidian_export: bool = False,
    semantic_chunking: bool = True,
    export_processed: bool = False,
    selected_file: Optional[Path] = None,
    verbose_stages: Optional[bool] = None,
    persist_refactor_mirror: Optional[bool] = None,
    docling_enabled: Optional[bool] = None,
    docling_model: Optional[str] = None,
    docling_artifacts_path: Optional[str] = None,
    docling_strict: Optional[bool] = None,
    docling_auto_download: Optional[bool] = None,
    vision_enabled: Optional[bool] = None,
    vision_provider: Optional[str] = None,
    vision_model: Optional[str] = None,
    vision_api_base_url: Optional[str] = None,
    vision_timeout_seconds: Optional[int] = None,
    vision_max_figures: Optional[int] = None,
    vision_auto_pull: Optional[bool] = None,
    vision_strict: Optional[bool] = None,
) -> None:
    """Ingest documents from a directory or a single selected file.

    Args:
        documents_dir: Path to the directory containing documents.
        fresh: If True, delete existing collection before ingesting.
        build_kg: If True, build a knowledge graph from the chunks.
        obsidian_export: If True, export KG as Obsidian-compatible markdown files.
        semantic_chunking: If True, use semantic similarity for chunk splitting.
        export_processed: If True, save cleaned docs and chunks to processed/ dir.
        selected_file: Optional single file to ingest instead of all files.
        verbose_stages: Optional override for stage-by-stage progress logs.
        persist_refactor_mirror: Optional override for refactor mirror artifacts.
    """
    documents_dir = validate_documents_dir(documents_dir, PROJECT_ROOT)
    cfg_kwargs = {
        "semantic_chunking": semantic_chunking,
        "build_kg": build_kg,
        "export_processed": export_processed,
        "enable_knowledge_graph_extraction": build_kg,
        "enable_knowledge_graph_storage": build_kg,
    }
    if verbose_stages is not None:
        cfg_kwargs["verbose_stage_logs"] = verbose_stages
    if persist_refactor_mirror is not None:
        cfg_kwargs["persist_refactor_mirror"] = persist_refactor_mirror
    if docling_enabled is not None:
        cfg_kwargs["enable_docling_parser"] = docling_enabled
    if docling_model:
        cfg_kwargs["docling_model"] = docling_model
    if docling_artifacts_path is not None:
        cfg_kwargs["docling_artifacts_path"] = docling_artifacts_path
    if docling_strict is not None:
        cfg_kwargs["docling_strict"] = docling_strict
    if docling_auto_download is not None:
        cfg_kwargs["docling_auto_download"] = docling_auto_download
    if vision_enabled is not None:
        cfg_kwargs["enable_vision_processing"] = vision_enabled
    if vision_provider:
        cfg_kwargs["vision_provider"] = vision_provider
    if vision_model:
        cfg_kwargs["vision_model"] = vision_model
    if vision_api_base_url is not None:
        cfg_kwargs["vision_api_base_url"] = vision_api_base_url
    if vision_timeout_seconds is not None:
        cfg_kwargs["vision_timeout_seconds"] = vision_timeout_seconds
    if vision_max_figures is not None:
        cfg_kwargs["vision_max_figures"] = vision_max_figures
    if vision_auto_pull is not None:
        cfg_kwargs["vision_auto_pull"] = vision_auto_pull
    if vision_strict is not None:
        cfg_kwargs["vision_strict"] = vision_strict

    cfg = IngestionConfig(
        **cfg_kwargs,
    )
    selected_sources = None
    if selected_file is not None:
        selected_sources = [selected_file.resolve()]

    summary = ingest_directory(
        documents_dir=documents_dir,
        config=cfg,
        fresh=fresh,
        update=update,
        obsidian_export=obsidian_export,
        selected_sources=selected_sources,
    )
    logger.info(
        "Ingestion complete. processed=%d skipped=%d failed=%d stored_chunks=%d removed_sources=%d",
        summary.processed,
        summary.skipped,
        summary.failed,
        summary.stored_chunks,
        summary.removed_sources,
    )
    for err in summary.errors:
        logger.warning("Ingestion error: %s", err)
    for warning in summary.design_warnings:
        logger.warning("Ingestion design warning: %s", warning)


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser for ingestion."""
    parser = argparse.ArgumentParser(description="Ingest documents into RAG system")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--file",
        type=Path,
        help="Ingest a single document file only",
    )
    source_group.add_argument(
        "--dir",
        type=Path,
        help="Ingest all supported documents in a specific directory",
    )
    parser.add_argument(
        "--no-kg", action="store_true", help="Skip knowledge graph building"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Incremental update mode (changed docs only; idempotent writes)",
    )
    parser.add_argument(
        "--export-obsidian",
        action="store_true",
        help="Export knowledge graph as Obsidian markdown files",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable semantic chunking (use character splitting only)",
    )
    parser.add_argument(
        "--export-processed",
        action="store_true",
        help="Export cleaned documents and chunks to processed/ directory",
    )
    # Tri-state by design: None -> use config default, True/False -> explicit override.
    parser.add_argument(
        "--verbose-stages",
        dest="verbose_stages",
        action="store_true",
        help="Print stage-by-stage ingestion progress for each source",
    )
    parser.add_argument(
        "--no-verbose-stages",
        dest="verbose_stages",
        action="store_false",
        help="Disable stage-by-stage ingestion progress for this run",
    )
    parser.set_defaults(verbose_stages=None)
    parser.add_argument(
        "--no-refactor-mirror",
        action="store_true",
        help="Disable writing original/refactored mirror artifacts and mappings",
    )
    parser.add_argument(
        "--no-docling",
        action="store_true",
        help="Disable Docling parser in structure detection stage",
    )
    parser.add_argument(
        "--docling-model",
        type=str,
        default=None,
        help="Docling parsing model identifier used in stage-2 parsing",
    )
    parser.add_argument(
        "--docling-artifacts-path",
        type=str,
        default=None,
        help="Optional local path for Docling model artifacts/cache",
    )
    parser.add_argument(
        "--docling-non-strict",
        action="store_true",
        help="Do not fail ingestion when Docling parsing errors occur",
    )
    parser.add_argument(
        "--vision",
        dest="vision_enabled",
        action="store_true",
        help="Enable vision-based figure caption/OCR enrichment",
    )
    parser.add_argument(
        "--no-vision",
        dest="vision_enabled",
        action="store_false",
        help="Disable vision-based figure caption/OCR enrichment for this run",
    )
    parser.add_argument(
        "--vision-provider",
        type=str,
        default=None,
        choices=("ollama", "openai_compatible"),
        help="Vision backend provider",
    )
    parser.add_argument(
        "--vision-model",
        type=str,
        default=None,
        help="Vision model identifier for the selected provider",
    )
    parser.add_argument(
        "--vision-api-base-url",
        type=str,
        default=None,
        help="Base URL for openai_compatible provider (e.g. http://my-vision-server:8009).",
    )
    parser.add_argument(
        "--vision-timeout-seconds",
        type=int,
        default=None,
        help="Timeout per vision model request in seconds",
    )
    parser.add_argument(
        "--vision-max-figures",
        type=int,
        default=None,
        help="Maximum number of extracted figures to describe per document",
    )
    parser.add_argument(
        "--no-vision-auto-pull",
        action="store_true",
        help="Disable auto-pull of missing vision model in Ollama preflight",
    )
    parser.add_argument(
        "--vision-strict",
        action="store_true",
        help="Fail the document if vision analysis errors occur",
    )
    parser.set_defaults(vision_enabled=None)
    parser.add_argument(
        "--no-docling-auto-download",
        action="store_true",
        help="Disable Docling model auto-download preflight",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    selected_file: Optional[Path] = None
    target_documents_dir = DOCUMENTS_DIR
    if args.file is not None:
        selected_file = args.file.resolve()
        if not selected_file.exists() or not selected_file.is_file():
            parser.error(f"--file must point to an existing file: {selected_file}")
        target_documents_dir = selected_file.parent
    elif args.dir is not None:
        target_documents_dir = validate_documents_dir(args.dir, PROJECT_ROOT)

    ingest(
        documents_dir=target_documents_dir,
        fresh=not args.update,
        update=args.update,
        build_kg=not args.no_kg,
        obsidian_export=args.export_obsidian,
        semantic_chunking=not args.no_semantic,
        export_processed=args.export_processed,
        selected_file=selected_file,
        verbose_stages=args.verbose_stages,
        persist_refactor_mirror=not args.no_refactor_mirror,
        docling_enabled=not args.no_docling,
        docling_model=args.docling_model,
        docling_artifacts_path=args.docling_artifacts_path,
        docling_strict=not args.docling_non_strict,
        docling_auto_download=not args.no_docling_auto_download,
        vision_enabled=args.vision_enabled,
        vision_provider=args.vision_provider,
        vision_model=args.vision_model,
        vision_api_base_url=args.vision_api_base_url,
        vision_timeout_seconds=args.vision_timeout_seconds,
        vision_max_figures=args.vision_max_figures,
        vision_auto_pull=not args.no_vision_auto_pull,
        vision_strict=args.vision_strict,
    )


if __name__ == "__main__":
    main()
