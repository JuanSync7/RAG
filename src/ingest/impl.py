# @summary
# Ingestion pipeline orchestrator: source discovery, idempotency, two-phase ingest.
# Exports: ingest_directory, ingest_file, verify_core_design, IngestionConfig, Runtime
# Deps: src.vector_db, src.core.embeddings, src.ingest.embedding,
#       src.ingest.doc_processing, src.ingest.support.parser_registry
# verify_core_design calls _check_docling_chunking_config (Task 4.2) which validates
#   vlm_mode values, builtin-requires-docling, and hybrid_chunker_max_tokens > 8192 limit.
# verify_core_design also calls _check_visual_embedding_config (Task 1.1) which validates
#   enable_visual_embedding requires enable_docling_parser, colqwen_batch_size 1-32,
#   page_image_quality 1-100, and page_image_max_dimension 256-4096.
# verify_core_design also calls _check_parser_abstraction_config (T9) which validates
#   parser_strategy, chunker, VLM mutual exclusion, and VLM+code incompatibility.
# verify_core_design also calls _check_embedding_batch_config (FR-1211) which validates
#   embedding_batch_size range [1, 2048].
# ParserRegistry is instantiated in ingest_directory and attached to Runtime (T9).
# @end-summary

"""Ingestion pipeline runtime orchestration and public implementation API.

This module provides the primary entrypoints for running the ingestion workflow
over a file or directory, including:

- Source discovery and stable identity generation
- Idempotency checks using a persisted manifest (incremental updates)
- Orchestration of the compiled LangGraph workflow
- Persistence of processed outputs (vector store, optional mirrors/artifacts)
"""

from __future__ import annotations

import hashlib
import orjson
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Optional

_UNSAFE_CHARS = re.compile(r'[/\\:*?"<>|]')

from config.settings import (
    LLM_ROUTER_CONFIG,
    PROCESSED_DIR,
    RAG_INGESTION_MIRROR_DIR,
    RAG_INGESTION_EXPORT_EXTENSIONS,
)
from src.core.embeddings import get_embedding_provider
from src.vector_db import (
    delete_collection,
    delete_by_source_key,
    ensure_collection,
    get_client,
)
from src.ingest.support import ensure_docling_ready
from src.ingest.support import ensure_vision_ready
from src.ingest.support.parser_registry import ParserRegistry
from src.ingest.common import (
    ManifestEntry,
    SourceIdentity,
)
from src.ingest.common.schemas import PIPELINE_SCHEMA_VERSION
from src.ingest.common import (
    load_manifest,
    save_manifest,
    sha256_bytes,
    sha256_path,
)
from src.ingest.common import (
    IngestFileResult,
    IngestionConfig,
    IngestionDesignCheck,
    IngestionRunSummary,
    PIPELINE_NODE_NAMES,
    Runtime,
)
from src.ingest.common import CleanDocumentStore
from src.ingest.doc_processing import run_document_processing
from src.ingest.embedding import run_embedding_pipeline

logger = logging.getLogger("rag.ingest")
_LOCAL_CONNECTOR = "local_fs"


def _safe_relative(path: Path, root: Path) -> str:
    """Return a stable display path relative to a root when possible.

    Args:
        path: Path to render.
        root: Root directory to attempt making `path` relative to.

    Returns:
        Relative path string when `path` is within `root`, otherwise the resolved
        absolute path string.
    """
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _local_source_identity(path: Path, documents_root: Path) -> SourceIdentity:
    """Build stable source identity fields for local filesystem ingestion.

    Args:
        path: Path to the source file.
        documents_root: Root directory used for display names.

    Returns:
        A `SourceIdentity` mapping containing stable keys suitable for
        idempotency checks and manifest indexing.
    """
    resolved = path.resolve()
    stat = resolved.stat()
    source_id = f"{stat.st_dev}:{stat.st_ino}"
    source_key = f"{_LOCAL_CONNECTOR}:{source_id}"
    return {
        "source_path": str(resolved),
        "source_name": _safe_relative(resolved, documents_root),
        "source_uri": resolved.as_uri(),
        "source_id": source_id,
        "source_key": source_key,
        "connector": _LOCAL_CONNECTOR,
        "source_version": str(stat.st_mtime_ns),
    }


def _mirror_file_stem(source_name: str, source_key: str) -> str:
    """Return a stable mirror filename stem for a source.

    Args:
        source_name: Display name for the source (often relative path).
        source_key: Stable source key used for hashing.

    Returns:
        A filesystem-safe stem value.
    """
    safe_name = _UNSAFE_CHARS.sub("_", source_name)
    suffix = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:8]
    return f"{safe_name}.{suffix}"


def _write_refactor_mirror_artifacts(
    source: SourceIdentity,
    result: dict,
    config: IngestionConfig,
) -> None:
    """Persist original/refactored mirrors plus chunk-level provenance mapping.

    Args:
        source: Source identity payload.
        result: Ingestion graph result payload.
        config: Ingestion configuration controlling mirror directory.
    """
    mirror_dir = Path(config.mirror_output_dir or str(RAG_INGESTION_MIRROR_DIR))
    mirror_dir.mkdir(parents=True, exist_ok=True)
    stem = _mirror_file_stem(source["source_name"], source["source_key"])
    original_path = mirror_dir / f"{stem}.original.md"
    refactored_path = mirror_dir / f"{stem}.refactored.md"
    mapping_path = mirror_dir / f"{stem}.mapping.json"

    original_path.write_text(str(result.get("raw_text", "")), encoding="utf-8")
    # NOTE: legacy mirror file naming retained for back-compat; the refactoring
    # stage was removed in PR1, so this file now contains cleaned_text (the
    # chunker's input) rather than an LLM-rewritten variant.
    refactored_path.write_text(str(result.get("cleaned_text", "")), encoding="utf-8")

    mapping_payload = {
        "source": source["source_name"],
        "source_uri": source["source_uri"],
        "source_key": source["source_key"],
        "source_id": source["source_id"],
        "connector": source["connector"],
        "source_version": source["source_version"],
        "original_mirror_path": str(original_path),
        "refactored_mirror_path": str(refactored_path),
        "chunks": [
            {
                "chunk_index": int(chunk.metadata.get("chunk_index", idx)),
                "chunk_id": str(chunk.metadata.get("chunk_id", "")),
                "retrieval_text_origin": str(chunk.metadata.get("retrieval_text_origin", "")),
                "original_char_start": int(chunk.metadata.get("original_char_start", -1)),
                "original_char_end": int(chunk.metadata.get("original_char_end", -1)),
                "refactored_char_start": int(chunk.metadata.get("refactored_char_start", -1)),
                "refactored_char_end": int(chunk.metadata.get("refactored_char_end", -1)),
                "provenance_method": str(chunk.metadata.get("provenance_method", "")),
                "provenance_confidence": float(chunk.metadata.get("provenance_confidence", 0.0)),
            }
            for idx, chunk in enumerate(result.get("chunks", []))
        ],
    }
    mapping_path.write_bytes(orjson.dumps(mapping_payload, option=orjson.OPT_INDENT_2))


def _normalize_manifest_entries(
    manifest: dict[str, Any],
) -> dict[str, ManifestEntry]:
    """Normalize old/new manifest formats into a source_key-indexed mapping.

    Args:
        manifest: Raw manifest mapping loaded from disk.

    Returns:
        Normalized mapping keyed by `source_key`.
    """
    normalized: dict[str, ManifestEntry] = {}
    for raw_key, raw_entry in manifest.items():
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        source_key = str(entry.get("source_key", "")).strip()
        if not source_key:
            key_text = str(raw_key)
            if key_text.startswith(f"{_LOCAL_CONNECTOR}:"):
                source_key = key_text
            else:
                source_key = f"legacy_name:{key_text}"
                entry.setdefault("legacy_name", key_text)
        entry["source_key"] = source_key
        # Ensure lifecycle fields have safe defaults for pre-1.0.0 manifests (FR-3114).
        entry.setdefault("schema_version", "0.0.0")
        entry.setdefault("trace_id", "")
        entry.setdefault("batch_id", "")
        entry.setdefault("deleted", False)
        entry.setdefault("deleted_at", "")
        entry.setdefault("validation", {})
        entry.setdefault("clean_hash", "")
        normalized[source_key] = entry
    return normalized


def _find_manifest_entry(
    manifest: dict[str, ManifestEntry],
    source: SourceIdentity,
) -> tuple[Optional[str], ManifestEntry]:
    """Find the best manifest match for a discovered source identity.

    Matching proceeds from most-stable to least-stable identifiers:
    ``source_key`` → ``source_id`` → ``source_uri`` → legacy filename.

    Args:
        manifest: Normalized manifest mapping keyed by `source_key`.
        source: Discovered source identity.

    Returns:
        Tuple of ``(matched_key, entry)``. When no match is found, returns
        ``(None, ManifestEntry())``.
    """
    direct = manifest.get(source["source_key"])
    if direct is not None:
        return source["source_key"], direct

    for key, entry in manifest.items():
        if entry.get("source_id") == source["source_id"]:
            return key, entry
    for key, entry in manifest.items():
        if entry.get("source_uri") == source["source_uri"]:
            return key, entry

    leaf_name = Path(source["source_path"]).name
    for key, entry in manifest.items():
        if entry.get("legacy_name") == leaf_name:
            return key, entry
    return None, ManifestEntry()


def _check_docling_chunking_config(
    config: IngestionConfig,
) -> tuple[list[str], list[str]]:
    """Validate Docling-native chunking configuration.

    Checks three contradiction patterns:
    1. vlm_mode=builtin without docling installed → fatal error
    2. vlm_mode=external without LiteLLM vision model configured → warning
    3. hybrid_chunker_max_tokens > 8192 (bge-m3 limit) → warning

    Args:
        config: IngestionConfig to validate.

    Returns:
        Tuple of (errors, warnings). Errors block pipeline start.
        Warnings are logged but do not halt processing.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Rule 0: vlm_mode must be one of the accepted values.
    _VALID_VLM_MODES = {"disabled", "builtin", "external"}
    if config.vlm_mode not in _VALID_VLM_MODES:
        errors.append(
            f"vlm_mode={config.vlm_mode!r} is not valid;"
            f" must be one of {sorted(_VALID_VLM_MODES)}"
        )

    # Rule A: vlm_mode=builtin requires docling to be installed.
    if config.vlm_mode == "builtin":
        try:
            import docling.document_converter  # noqa: F401
        except ImportError:
            errors.append(
                "vlm_mode=builtin requires docling to be installed (uv add docling)"
            )

    # Rule B: vlm_mode=external without a vision model configured.
    if config.vlm_mode == "external":
        if not config.vision_model and not LLM_ROUTER_CONFIG:
            warnings.append(
                "vlm_mode=external is set but no vision model is configured;"
                " VLM enrichment will be skipped at runtime"
            )

    # Rule C: hybrid_chunker_max_tokens exceeds bge-m3 maximum input (8192).
    # BGE-M3 supports up to 8192 tokens. Past ~1500–2000 the embedding loses
    # specificity, but truncation only kicks in above 8192.
    if config.hybrid_chunker_max_tokens > 8192:
        warnings.append(
            f"hybrid_chunker_max_tokens ({config.hybrid_chunker_max_tokens})"
            " exceeds bge-m3 maximum input (8192);"
            " chunks will be silently truncated during embedding"
        )

    return errors, warnings


def _check_visual_embedding_config(
    config: IngestionConfig,
) -> tuple[list[str], list[str]]:
    """Validate visual embedding configuration.

    Checks:
    1. enable_visual_embedding=True requires enable_docling_parser=True (fatal)
    2. colqwen_batch_size range 1-32 (fatal)
    3. page_image_quality range 1-100 (fatal)
    4. page_image_max_dimension range 256-4096 (fatal)

    Args:
        config: IngestionConfig to validate.

    Returns:
        Tuple of (errors, warnings). Errors block pipeline start.
    """
    if not config.enable_visual_embedding:
        return [], []

    errors: list[str] = []
    warnings: list[str] = []

    if not config.enable_docling_parser:
        errors.append(
            "enable_visual_embedding=True requires enable_docling_parser=True;"
            " Docling is needed to render page images for ColQwen2 embedding"
        )

    if not (1 <= config.colqwen_batch_size <= 32):
        errors.append(
            f"colqwen_batch_size={config.colqwen_batch_size} is out of range;"
            " must be between 1 and 32 (inclusive)"
        )

    if not (1 <= config.page_image_quality <= 100):
        errors.append(
            f"page_image_quality={config.page_image_quality} is out of range;"
            " must be between 1 and 100 (inclusive)"
        )

    if not (256 <= config.page_image_max_dimension <= 4096):
        errors.append(
            f"page_image_max_dimension={config.page_image_max_dimension} is out of range;"
            " must be between 256 and 4096 (inclusive)"
        )

    return errors, warnings


def _check_embedding_batch_config(
    config: IngestionConfig,
) -> tuple[list[str], list[str]]:
    """Validate embedding batch configuration. FR-1211.

    Checks:
    1. embedding_batch_size range [1, 2048] (fatal)

    Args:
        config: IngestionConfig to validate.

    Returns:
        Tuple of (errors, warnings). Errors block pipeline start.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not (1 <= config.embedding_batch_size <= 2048):
        errors.append(
            f"embedding_batch_size={config.embedding_batch_size} is out of range;"
            " must be between 1 and 2048 (inclusive)"
        )

    return errors, warnings


def _check_parser_abstraction_config(
    config: IngestionConfig,
) -> tuple[list[str], list[str]]:
    """Validate parser abstraction configuration fields. FR-3301, FR-3320, FR-3340.

    Checks:
    1. ``parser_strategy`` must be one of the accepted values (FR-3301 AC 3).
    2. ``chunker`` must be one of the accepted values (FR-3322).
    3. ``chunker="markdown"`` emits a warning (FR-3323).
    4. VLM mutual exclusion: ``vlm_mode="builtin"`` AND ``enable_multimodal_processing=True``
       is an error (FR-3340, FR-3341).
    5. ``vlm_mode="external"`` AND ``enable_multimodal_processing=True`` is a warning (FR-3342).
    6. ``enable_vlm_enrichment=True`` AND ``parser_strategy="code"`` is an error (FR-3341).

    Args:
        config: IngestionConfig to validate.

    Returns:
        Tuple of (errors, warnings). Errors block pipeline start.
    """
    errors: list[str] = []
    warnings_list: list[str] = []

    # Rule 1: parser_strategy must be a recognised value.
    _VALID_STRATEGIES = {"auto", "document", "code", "text"}
    if config.parser_strategy not in _VALID_STRATEGIES:
        errors.append(
            f"parser_strategy must be 'auto', 'document', 'code', or 'text', "
            f"got '{config.parser_strategy}'."
        )

    # Rule 2: chunker must be a recognised value.
    _VALID_CHUNKERS = {"native", "markdown"}
    if config.chunker not in _VALID_CHUNKERS:
        errors.append(
            f"chunker must be 'native' or 'markdown', got '{config.chunker}'."
        )

    # Rule 3: chunker override warning.
    if config.chunker == "markdown":
        warnings_list.append(
            "Chunker override active: all parsers will use markdown-based chunking. "
            "Native chunking (with richer heading metadata from HybridChunker / "
            "AST-aware code splitting) is disabled."
        )

    # Rule 4: VLM mutual exclusion (builtin + multimodal).
    if config.vlm_mode == "builtin" and config.enable_multimodal_processing:
        errors.append(
            "vlm_mode='builtin' and enable_multimodal_processing=True are mutually "
            "exclusive. vlm_mode='builtin' describes figures at parse time via Docling "
            "SmolVLM. enable_multimodal_processing describes figures in the Phase 1 "
            "multimodal node via vision.py. Disable one to prevent double VLM "
            "processing of figure images."
        )

    # Rule 5: VLM coexistence info (external + multimodal).
    if config.vlm_mode == "external" and config.enable_multimodal_processing:
        warnings_list.append(
            "vlm_mode='external' and enable_multimodal_processing are both active. "
            "Phase 1 multimodal node will process figures pre-chunking; "
            "vlm_mode='external' will enrich chunks post-chunking. Both being active "
            "is valid but means figures are processed at two pipeline stages."
        )

    # Rule 6: VLM enrichment incompatible with code parser strategy.
    enable_vlm = getattr(config, "enable_vlm_enrichment", False)
    if enable_vlm and config.parser_strategy == "code":
        errors.append(
            "VLM enrichment is incompatible with code parser strategy. "
            "Code parser chunks contain raw source; VLM figure enrichment "
            "is meaningless for code content. "
            "Set parser_strategy to 'auto', 'document', or 'text' to use VLM enrichment."
        )

    return errors, warnings_list


def verify_core_design(config: IngestionConfig) -> IngestionDesignCheck:
    """Validate ingestion configuration compatibility and return actionable feedback.

    Args:
        config: Ingestion configuration to validate.

    Returns:
        An `IngestionDesignCheck` containing errors and warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []
    if config.chunk_overlap >= config.chunk_size:
        errors.append("chunk_overlap must be < chunk_size")
    if config.enable_docling_parser and not str(config.docling_model).strip():
        errors.append("docling parser requires a non-empty docling_model")
    if config.enable_vision_processing:
        if not config.enable_multimodal_processing:
            errors.append("vision processing requires multimodal_processing to be enabled")

    # Docling-native chunking validation (Task 4.2).
    dc_errors, dc_warnings = _check_docling_chunking_config(config)
    errors.extend(dc_errors)
    warnings.extend(dc_warnings)

    # Visual embedding pipeline validation (Task 1.1).
    ve_errors, ve_warnings = _check_visual_embedding_config(config)
    errors.extend(ve_errors)
    warnings.extend(ve_warnings)

    # Parser abstraction validation (Task T9, FR-3301, FR-3320, FR-3340-FR-3342).
    pa_errors, pa_warnings = _check_parser_abstraction_config(config)
    errors.extend(pa_errors)
    warnings.extend(pa_warnings)

    # Batch embedding validation (FR-1211).
    eb_errors, eb_warnings = _check_embedding_batch_config(config)
    errors.extend(eb_errors)
    warnings.extend(eb_warnings)

    return IngestionDesignCheck(ok=not errors, errors=errors, warnings=warnings)


def ingest_file(
    source_path: Path,
    runtime: Runtime,
    source_name: str,
    source_uri: str,
    source_key: str,
    source_id: str,
    connector: str,
    source_version: str,
    existing_hash: str = "",
    existing_source_uri: str = "",
    batch_id: str = "",
    raw_bytes: bytes | None = None,
) -> IngestFileResult:
    """Run the two-phase ingestion pipeline for a single source file.

    Phase 1 (Document Processing) extracts and cleans the document.
    Phase 2 (Embedding Pipeline) chunks, embeds, and stores vectors.
    The CleanDocumentStore persists Phase 1 output as the boundary.

    Args:
        source_path: Source file path.
        runtime: Runtime container with shared dependencies.
        source_name: Display name for the source.
        source_uri: Stable URI for the source.
        source_key: Stable source key used for idempotency.
        source_id: Stable identity for the source.
        connector: Connector identifier.
        source_version: Source version string.
        existing_hash: Previously stored content hash (for incremental updates).
        existing_source_uri: Previously stored URI (for incremental updates).
        batch_id: Optional batch grouping ID (FR-3053). Empty string when not
            part of a named batch run.
        raw_bytes: Pre-read file bytes. When provided, ``document_ingestion_node``
            skips the disk read, avoiding a double-read when the caller already
            read the file for the idempotency hash check.

    Returns:
        ``IngestFileResult`` describing errors, stored chunk count, metadata,
        processing log, content hashes, and trace_id for the ingestion run.
    """
    config = runtime.config

    # ── Trace ID generation (FR-3050) ─────────────────────────────────────
    trace_id = str(uuid.uuid4())
    logger.info(
        "ingest_file_start trace_id=%s source_key=%s batch_id=%s",
        trace_id,
        source_key,
        batch_id,
    )

    # ── Phase 1 ──────────────────────────────────────────────────────────
    phase1 = run_document_processing(
        runtime=runtime,
        source_path=str(source_path),
        source_name=source_name,
        source_uri=source_uri,
        source_key=source_key,
        source_id=source_id,
        connector=connector,
        source_version=source_version,
        trace_id=trace_id,
        raw_bytes=raw_bytes,
    )

    # run_document_processing always returns a DocumentProcessingState TypedDict (never None).
    if phase1.get("errors"):
        return IngestFileResult(
            errors=phase1["errors"],
            stored_count=0,
            metadata_summary="",
            metadata_keywords=[],
            processing_log=phase1.get("processing_log", []),
            source_hash=phase1.get("source_hash", ""),
            clean_hash="",
            trace_id=trace_id,
        )

    # Determine final clean text
    clean_text: str = phase1.get("cleaned_text", "")
    clean_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()

    # ── Debug export (opt-in via export_processed) ────────────────────────
    if config.export_processed and config.clean_store_dir:
        _debug_store = CleanDocumentStore(Path(config.clean_store_dir))
        meta = {
            "source_key": source_key,
            "source_name": source_name,
            "source_uri": source_uri,
            "source_id": source_id,
            "connector": connector,
            "source_version": source_version,
            "source_hash": phase1.get("source_hash", ""),
        }
        _debug_store.write(
            source_key,
            clean_text,
            meta,
            docling_document=phase1.get("docling_document") if config.persist_docling_document else None,
        )

    # ── Write mirror artifacts (optional) ─────────────────────────────────
    if config.persist_refactor_mirror:
        source_identity = {
            "source_path": str(source_path),
            "source_name": source_name,
            "source_uri": source_uri,
            "source_key": source_key,
            "source_id": source_id,
            "connector": connector,
            "source_version": source_version,
        }
        _write_refactor_mirror_artifacts(source_identity, phase1, config)

    # ── Phase 2 (DoclingDocument passed in-memory from Phase 1) ──────────
    # Propagate trace_id from Phase 1 state to Phase 2 (FR-3052).
    # phase1.get("trace_id") will be the same value we injected above, but we
    # read it from the state to stay consistent with the contract that Phase 2
    # receives trace_id via state, not as a free variable.
    phase2 = run_embedding_pipeline(
        runtime=runtime,
        source_key=source_key,
        source_name=source_name,
        source_uri=source_uri,
        source_id=source_id,
        connector=connector,
        source_version=source_version,
        clean_text=clean_text,
        clean_hash=clean_hash,
        docling_document=phase1.get("docling_document"),
        trace_id=phase1.get("trace_id", trace_id),
        batch_id=batch_id,
    )

    return IngestFileResult(
        errors=phase2.get("errors", []),
        stored_count=phase2.get("stored_count", 0),
        metadata_summary=phase2.get("metadata_summary", ""),
        metadata_keywords=phase2.get("metadata_keywords", []),
        processing_log=phase1.get("processing_log", []) + phase2.get("processing_log", []),
        source_hash=phase1.get("source_hash", ""),
        clean_hash=clean_hash,
        trace_id=trace_id,
    )


def ingest_directory(
    documents_dir: Path,
    config: Optional[IngestionConfig] = None,
    fresh: bool = False,
    update: bool = True,
    selected_sources: Optional[list[Path]] = None,
    batch_id: str = "",
) -> IngestionRunSummary:
    """Ingest a directory of documents and persist vector artifacts.

    Args:
        documents_dir: Directory containing source documents.
        config: Optional ingestion configuration. When omitted, defaults are used.
        fresh: Whether to start from a fresh vector store collection.
        update: Whether to run in incremental mode using the manifest.
        selected_sources: Optional explicit list of files to ingest.
        batch_id: Optional batch grouping ID (FR-3053). When provided, all files in
            this run share the same batch_id in their manifests and Weaviate metadata.
            Empty string (default) means no batch grouping.

    Returns:
        An `IngestionRunSummary` describing the run outcome.

    Raises:
        ValueError: If configuration validation fails.
        RuntimeError: If required optional dependencies (e.g., Docling or vision)
            are enabled but not available.
    """
    config = config or IngestionConfig()
    config.update_mode = update
    design = verify_core_design(config)
    if not design.ok:
        raise ValueError("Invalid ingestion config: " + "; ".join(design.errors))
    if config.enable_docling_parser:
        ensure_docling_ready(
            parser_model=config.docling_model,
            artifacts_path=config.docling_artifacts_path,
            auto_download=config.docling_auto_download,
        )
    if config.enable_vision_processing:
        ensure_vision_ready(config)

    manifest = _normalize_manifest_entries(load_manifest())
    errors: list[str] = []
    processed = skipped = failed = stored_chunks = 0

    patterns = [
        pattern.strip()
        for pattern in RAG_INGESTION_EXPORT_EXTENSIONS.split(",")
        if pattern.strip()
    ]
    allowed_suffixes = {pattern.lower() for pattern in patterns}
    if selected_sources is None:
        files = sorted(
            {path.resolve() for p in patterns for path in documents_dir.rglob(f"*{p}")}
        )
    else:
        files = sorted(
            {
                path.resolve()
                for path in selected_sources
                if path.is_file() and path.suffix.lower() in allowed_suffixes
            }
        )
    if not files:
        return IngestionRunSummary(0, 0, 0, 0, 0, [], design.warnings)

    sources = [_local_source_identity(path, documents_dir) for path in files]
    source_keys = {source["source_key"] for source in sources}

    removed_sources = (
        sorted(set(manifest.keys()) - source_keys)
        if update and selected_sources is None
        else []
    )

    with get_client() as client:
        if fresh:
            delete_collection(client)
            manifest = {}
        ensure_collection(client)

        for source in removed_sources:
            delete_by_source_key(
                client,
                source,
                legacy_source=str(manifest.get(source, {}).get("source", "")),
            )
            # Clean up debug export artifacts if they exist.
            if config.clean_store_dir:
                CleanDocumentStore(Path(config.clean_store_dir)).delete(source)
            manifest.pop(source, None)

        _db_client = None
        if config.store_documents:
            from src.db import create_persistent_client as _db_create_client, ensure_bucket as _db_ensure_bucket
            _db_client = _db_create_client()
            _db_ensure_bucket(_db_client, config.target_bucket or None)

        # Instantiate parser registry and validate readiness (T9 / FR-3303).
        try:
            _parser_registry = ParserRegistry(config)
            _parser_registry.ensure_all_ready(config)
        except Exception as _preg_exc:
            logger.warning(
                "ParserRegistry initialisation failed: %s — "
                "structure_detection_node will use legacy Docling fallback.",
                _preg_exc,
            )
            _parser_registry = None

        runtime = Runtime(
            config=config,
            embedder=get_embedding_provider(tier="ingest"),
            weaviate_client=client,
            db_client=_db_client,
            parser_registry=_parser_registry,
        )

        if config.export_processed:
            PROCESSED_DIR.mkdir(exist_ok=True)

        for source in sources:
            source_path = Path(source["source_path"])
            logger.info(
                "ingestion_start source=%s source_key=%s",
                source["source_name"],
                source["source_key"],
            )
            matched_key, matched_entry = _find_manifest_entry(manifest, source)
            previous_hash = matched_entry.get("content_hash", "") if update else ""
            previous_uri = matched_entry.get("source_uri", "") if update else ""
            # Idempotency check: skip if source unchanged (hash match in manifest).
            # Read bytes once here so we can reuse them in ingest_file, avoiding
            # a second disk read inside document_ingestion_node.
            raw_bytes = source_path.read_bytes()
            current_hash = sha256_bytes(raw_bytes)
            if update and previous_hash:
                if current_hash == previous_hash:
                    skipped += 1
                    if matched_key and matched_key != source["source_key"]:
                        manifest.pop(matched_key, None)
                    manifest[source["source_key"]] = {
                        **matched_entry,
                        "source": source["source_name"],
                        "source_uri": source["source_uri"],
                        "source_id": source["source_id"],
                        "source_key": source["source_key"],
                        "connector": source["connector"],
                        "source_version": source["source_version"],
                        "content_hash": previous_hash,
                    }
                    logger.info(
                        "ingestion_skipped source=%s source_key=%s reason=unchanged",
                        source["source_name"],
                        source["source_key"],
                    )
                    continue
            try:
                result = ingest_file(
                    source_path,
                    runtime,
                    source_name=source["source_name"],
                    source_uri=source["source_uri"],
                    source_key=source["source_key"],
                    source_id=source["source_id"],
                    connector=source["connector"],
                    source_version=source["source_version"],
                    existing_hash=previous_hash if update else "",
                    existing_source_uri=previous_uri if update else "",
                    batch_id=batch_id,
                    raw_bytes=raw_bytes,
                )
                if result.errors:
                    failed += 1
                    errors.extend(result.errors)
                    logger.error(
                        "ingestion_failed source=%s source_key=%s errors=%s",
                        source["source_name"],
                        source["source_key"],
                        "; ".join(result.errors),
                    )
                    continue

                processed += 1
                stored_chunks += result.stored_count
                logger.info(
                    "ingestion_done source=%s source_key=%s chunks=%d stored=%d stages=%s",
                    source["source_name"],
                    source["source_key"],
                    result.stored_count,
                    result.stored_count,
                    " > ".join(result.processing_log),
                )
                if matched_key and matched_key != source["source_key"]:
                    manifest.pop(matched_key, None)
                stem = _mirror_file_stem(source["source_name"], source["source_key"])
                manifest[source["source_key"]] = {
                    "source": source["source_name"],
                    "source_uri": source["source_uri"],
                    "source_id": source["source_id"],
                    "source_key": source["source_key"],
                    "connector": source["connector"],
                    "source_version": source["source_version"],
                    "content_hash": result.source_hash,
                    "clean_hash": result.clean_hash,
                    "chunk_count": result.stored_count,
                    "summary": result.metadata_summary,
                    "keywords": result.metadata_keywords,
                    "processing_log": result.processing_log[-12:],
                    "mirror_stem": stem,
                    # -- Data Lifecycle fields (FR-3050, FR-3053, FR-3100) --
                    "schema_version": PIPELINE_SCHEMA_VERSION,
                    "trace_id": result.trace_id,
                    "batch_id": batch_id,
                    "deleted": False,
                    "deleted_at": "",
                    "validation": result.validation,
                }
                save_manifest(manifest)

                # Debug export is handled inside ingest_file when export_processed=True.
            except (OSError, ValueError, RuntimeError) as exc:
                failed += 1
                logger.exception(
                    "ingestion_unhandled_error source=%s error=%s",
                    source.get("source_name", "unknown"),
                    exc,
                )
                errors.append(f"unhandled:{source.get('source_name', 'unknown')}:{exc}")
                continue

        # KG persistence owned by KGWeave: no graph save/export here. The
        # CLI ``--obsidian-export`` switch is now a no-op for the in-process
        # path; export is performed by the KGWeave worker on its own backend.

        if _db_client is not None:
            from src.db import close_client as _db_close_client
            _db_close_client(_db_client)

    save_manifest(manifest)
    return IngestionRunSummary(
        processed=processed,
        skipped=skipped,
        failed=failed,
        stored_chunks=stored_chunks,
        removed_sources=len(removed_sources),
        errors=errors,
        design_warnings=design.warnings,
    )


__all__ = [
    "PIPELINE_NODE_NAMES",
    "IngestionConfig",
    "IngestionDesignCheck",
    "IngestionRunSummary",
    "Runtime",
    "ingest_directory",
    "ingest_file",
    "verify_core_design",
]
