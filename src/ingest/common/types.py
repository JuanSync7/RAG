# @summary
# Shared ingestion pipeline dataclasses, typed state schema, runtime container, and node-name registry.
# Exports: IngestionConfig, IngestionDesignCheck, IngestFileResult, IngestionRunSummary, Runtime, PIPELINE_NODE_NAMES
# Deps: config.settings, src.core.embeddings, src.core.knowledge_graph, src.ingest.common.schemas
# IngestionConfig new fields (Task 1.1): vlm_mode (str), hybrid_chunker_max_tokens (int), persist_docling_document (bool),
#   enable_visual_embedding (bool), visual_target_collection (str), colqwen_model_name (str),
#   colqwen_batch_size (int), page_image_quality (int), page_image_max_dimension (int)
# IngestionConfig new fields (Data Lifecycle T1, T4): clean_store_bucket (str), gc_mode (str),
#   gc_retention_days (int), gc_schedule (str)
# IngestionConfig new fields (Parser Abstraction Phase 2.2): parser_strategy (str), chunker (str)
# PIPELINE_NODE_NAMES includes "vlm_enrichment" between "chunking" and "chunk_enrichment"
# PIPELINE_NODE_NAMES includes "visual_embedding" between "embedding_storage" and "knowledge_graph_storage" (FR-604)
# IngestFileResult new field (Task 4.1): visual_stored_count (int, default 0, FR-605)
# IngestFileResult new fields (Data Lifecycle T3, T6): trace_id (str, default ""), validation (dict, default {})
# IngestionRunSummary new fields (Data Lifecycle T4): gc_soft_deleted, gc_hard_deleted, gc_retention_purged (int, default 0)
# Runtime new field (Phase 2.2): parser_registry (Any, typed as Any to avoid circular import)
# @end-summary

"""Shared ingestion pipeline types, configuration, and state contracts.

This module defines the ingestion pipeline's primary configuration dataclass,
runtime dependency container, and the shared LangGraph state schema used across
pipeline nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict

from config.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    OLLAMA_BASE_URL,
    VECTOR_COLLECTION_DEFAULT,
    RAG_INGESTION_ENABLE_CROSS_REFERENCE_EXTRACTION,
    RAG_INGESTION_DOCLING_ARTIFACTS_PATH,
    RAG_INGESTION_DOCLING_AUTO_DOWNLOAD,
    RAG_INGESTION_DOCLING_ENABLED,
    RAG_INGESTION_DOCLING_MODEL,
    RAG_INGESTION_DOCLING_STRICT,
    RAG_INGESTION_ENABLE_DOCUMENT_REFACTORING,
    RAG_INGESTION_ENABLE_KNOWLEDGE_GRAPH_EXTRACTION,
    RAG_INGESTION_ENABLE_KNOWLEDGE_GRAPH_STORAGE,
    RAG_INGESTION_ENABLE_MULTIMODAL_PROCESSING,
    RAG_INGESTION_ENABLE_QUALITY_VALIDATION,
    RAG_INGESTION_VISION_AUTO_PULL,
    RAG_INGESTION_VISION_ENABLED,
    RAG_INGESTION_VISION_MAX_FIGURES,
    RAG_INGESTION_VISION_MAX_IMAGE_BYTES,
    RAG_INGESTION_VISION_MAX_TOKENS,
    RAG_INGESTION_VISION_MODEL,
    RAG_INGESTION_VISION_PROVIDER,
    RAG_INGESTION_VISION_API_BASE_URL,
    RAG_INGESTION_VISION_API_KEY,
    RAG_INGESTION_VISION_API_PATH,
    RAG_INGESTION_VISION_STRICT,
    RAG_INGESTION_VISION_TEMPERATURE,
    RAG_INGESTION_VISION_TIMEOUT_SECONDS,
    RAG_INGESTION_LLM_ENABLED,
    RAG_INGESTION_LLM_MAX_KEYWORDS,
    RAG_INGESTION_LLM_MODEL,
    RAG_INGESTION_LLM_TEMPERATURE,
    RAG_INGESTION_LLM_TIMEOUT_SECONDS,
    RAG_INGESTION_MIRROR_DIR,
    RAG_INGESTION_PERSIST_REFACTOR_MIRROR,
    RAG_INGESTION_VERBOSE_STAGE_LOGS,
    SEMANTIC_CHUNKING_ENABLED,
    RAG_INGESTION_VLM_MODE,
    RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS,
    RAG_INGESTION_PERSIST_DOCLING_DOCUMENT,
    RAG_INGESTION_STORE_FIGURES_IN_DB,
    RAG_INGESTION_USE_DOCLING_CHUNKER_FOR_MARKDOWN,
    RAG_INGESTION_ENABLE_VISUAL_EMBEDDING,
    RAG_INGESTION_VISUAL_TARGET_COLLECTION,
    RAG_INGESTION_COLQWEN_MODEL,
    RAG_INGESTION_COLQWEN_BATCH_SIZE,
    RAG_INGESTION_PAGE_IMAGE_QUALITY,
    RAG_INGESTION_PAGE_IMAGE_MAX_DIMENSION,
    RAG_INGESTION_EMBEDDING_BATCH_SIZE,
)
from src.core import LocalBGEEmbeddings
from src.core import KnowledgeGraphBuilder
from src.ingest.common.schemas import ProcessedChunk

PIPELINE_NODE_NAMES = [
    "document_ingestion",
    "structure_detection",
    "multimodal_processing",
    "text_cleaning",
    "document_refactoring",
    "chunking",
    "vlm_enrichment",
    "chunk_enrichment",
    "metadata_generation",
    "cross_reference_extraction",
    "knowledge_graph_extraction",
    "quality_validation",
    "embedding_storage",
    "visual_embedding",
    "knowledge_graph_storage",
]


@dataclass
class IngestionConfig:
    """Configuration knobs controlling ingestion behavior and optional stages.

    LLM routing (model selection, provider, retries, fallbacks) is handled by
    ``src.platform.llm.LLMProvider`` via LiteLLM Router.  Fields like
    ``llm_model``, ``ollama_url``, ``vision_provider``, ``vision_api_base_url``
    are retained for backward compatibility and metadata logging but are **not**
    used for HTTP calls — LiteLLM handles that via ``RAG_LLM_*`` env vars or
    the Router YAML config.
    """

    # ── LLM behavioral flags (still active) ────────────────────────────
    llm_temperature: float = RAG_INGESTION_LLM_TEMPERATURE
    llm_timeout_seconds: int = RAG_INGESTION_LLM_TIMEOUT_SECONDS
    max_keywords: int = RAG_INGESTION_LLM_MAX_KEYWORDS
    enable_llm_metadata: bool = RAG_INGESTION_LLM_ENABLED
    # Retained for metadata logging; routing handled by LiteLLM Router.
    llm_model: str = RAG_INGESTION_LLM_MODEL
    enable_docling_parser: bool = RAG_INGESTION_DOCLING_ENABLED
    docling_model: str = RAG_INGESTION_DOCLING_MODEL
    docling_artifacts_path: str = RAG_INGESTION_DOCLING_ARTIFACTS_PATH
    docling_strict: bool = RAG_INGESTION_DOCLING_STRICT
    docling_auto_download: bool = RAG_INGESTION_DOCLING_AUTO_DOWNLOAD
    semantic_chunking: bool = SEMANTIC_CHUNKING_ENABLED
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    enable_multimodal_processing: bool = RAG_INGESTION_ENABLE_MULTIMODAL_PROCESSING
    # ── Vision behavioral flags (still active) ────────────────────────
    enable_vision_processing: bool = RAG_INGESTION_VISION_ENABLED
    vision_timeout_seconds: int = RAG_INGESTION_VISION_TIMEOUT_SECONDS
    vision_max_figures: int = RAG_INGESTION_VISION_MAX_FIGURES
    vision_max_image_bytes: int = RAG_INGESTION_VISION_MAX_IMAGE_BYTES
    vision_temperature: float = RAG_INGESTION_VISION_TEMPERATURE
    vision_max_tokens: int = RAG_INGESTION_VISION_MAX_TOKENS
    vision_auto_pull: bool = RAG_INGESTION_VISION_AUTO_PULL
    vision_strict: bool = RAG_INGESTION_VISION_STRICT
    # Retained for metadata logging; routing handled by LiteLLM Router.
    vision_provider: str = RAG_INGESTION_VISION_PROVIDER
    vision_model: str = RAG_INGESTION_VISION_MODEL
    vision_api_base_url: str = RAG_INGESTION_VISION_API_BASE_URL
    vision_api_key: str = RAG_INGESTION_VISION_API_KEY
    vision_api_path: str = RAG_INGESTION_VISION_API_PATH
    enable_document_refactoring: bool = RAG_INGESTION_ENABLE_DOCUMENT_REFACTORING
    enable_cross_reference_extraction: bool = (
        RAG_INGESTION_ENABLE_CROSS_REFERENCE_EXTRACTION
    )
    enable_knowledge_graph_extraction: bool = (
        RAG_INGESTION_ENABLE_KNOWLEDGE_GRAPH_EXTRACTION
    )
    enable_quality_validation: bool = RAG_INGESTION_ENABLE_QUALITY_VALIDATION
    enable_knowledge_graph_storage: bool = RAG_INGESTION_ENABLE_KNOWLEDGE_GRAPH_STORAGE
    min_chunk_chars: int = 40
    min_quality_score: float = 0.45
    build_kg: bool = True
    export_processed: bool = False
    update_mode: bool = False
    verbose_stage_logs: bool = RAG_INGESTION_VERBOSE_STAGE_LOGS
    persist_refactor_mirror: bool = RAG_INGESTION_PERSIST_REFACTOR_MIRROR
    mirror_output_dir: str = str(RAG_INGESTION_MIRROR_DIR)
    clean_store_dir: str = "data/clean_store"  # Directory for CleanDocumentStore. Empty string disables persistent store.
    # Target vector store collection for embedding storage. Defaults to VECTOR_COLLECTION_DEFAULT.
    target_collection: str = VECTOR_COLLECTION_DEFAULT
    store_documents: bool = True  # Whether to persist clean markdown to the document store (MinIO).
    target_bucket: str = ""  # MinIO bucket for document storage. Empty string uses MINIO_BUCKET default.
    # Retained for backward compat; routing handled by LiteLLM Router.
    ollama_url: str = OLLAMA_BASE_URL
    vlm_mode: str = RAG_INGESTION_VLM_MODE
    """VLM mode: "disabled" | "builtin" | "external". Default: "disabled"."""
    hybrid_chunker_max_tokens: int = RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS
    """Max tokens per HybridChunker chunk. Default: 512 (bge-m3 limit)."""
    persist_docling_document: bool = RAG_INGESTION_PERSIST_DOCLING_DOCUMENT
    """If True, persist DoclingDocument JSON to CleanDocumentStore. Default: True."""
    use_docling_chunker_for_markdown: bool = RAG_INGESTION_USE_DOCLING_CHUNKER_FOR_MARKDOWN
    """If True, .md/.html/.rst use Docling HybridChunker (structure-aware) for
    chunking — preserves lists, tables, requirement blocks. Default: True.
    The parse step stays on the lightweight PlainTextParser path."""
    store_figures_in_db: bool = RAG_INGESTION_STORE_FIGURES_IN_DB
    """If True, upload markdown image bytes to the document store (backend-
    agnostic) and record a ``store_key`` in chunk metadata. Default: False —
    chunks carry only the relative ``src``; citation UI resolves against the
    original source location."""

    # -- Visual embedding pipeline (FR-101 through FR-109) --
    enable_visual_embedding: bool = RAG_INGESTION_ENABLE_VISUAL_EMBEDDING
    """Enable dual-track visual embedding pipeline. Default: False. FR-101"""
    visual_target_collection: str = RAG_INGESTION_VISUAL_TARGET_COLLECTION
    """Weaviate collection name for visual page objects. Default: 'RAGVisualPages'. FR-102"""
    colqwen_model_name: str = RAG_INGESTION_COLQWEN_MODEL
    """ColQwen2 model identifier for visual embedding. Default: 'vidore/colqwen2-v1.0'. FR-103"""
    colqwen_batch_size: int = RAG_INGESTION_COLQWEN_BATCH_SIZE
    """Batch size for ColQwen2 inference. Range: 1-32. Default: 4. FR-104"""
    page_image_quality: int = RAG_INGESTION_PAGE_IMAGE_QUALITY
    """JPEG compression quality for page images. Range: 1-100. Default: 85. FR-105"""
    page_image_max_dimension: int = RAG_INGESTION_PAGE_IMAGE_MAX_DIMENSION
    """Max pixel dimension (longer edge) for page images. Range: 256-4096. Default: 1024. FR-106"""

    # -- Batch embedding (FR-1211) --
    embedding_batch_size: int = RAG_INGESTION_EMBEDDING_BATCH_SIZE
    """Number of chunks per embedding batch. Range: 1-2048. Default: 64. FR-1211"""

    # -- Data Lifecycle: MinIO clean store (Task 1) --
    clean_store_bucket: str = ""
    """MinIO bucket for clean store objects. Empty string reuses target_bucket."""

    # -- Data Lifecycle: GC / Lifecycle (Task 4) --
    gc_mode: str = "soft"
    """Default GC delete mode: "soft" (default) or "hard"."""
    gc_retention_days: int = 30
    """Retention period in days for soft-deleted data before hard deletion."""
    gc_schedule: str = ""
    """Cron expression for scheduled GC runs. Empty string disables."""

    # -- Cross-document deduplication (FR-3460) --
    enable_cross_document_dedup: bool = True
    """Enable cross-document chunk deduplication. Default: True. FR-3402, FR-3460."""
    enable_fuzzy_dedup: bool = False
    """Enable Tier 2 MinHash fuzzy dedup (requires enable_cross_document_dedup=True). Default: False. FR-3420."""
    fuzzy_similarity_threshold: float = 0.95
    """Jaccard similarity threshold for fuzzy match. Range: [0.0, 1.0]. Default: 0.95. FR-3422."""
    fuzzy_shingle_size: int = 3
    """Word-level n-gram size for MinHash shingles. Must be >= 1. Default: 3. FR-3421."""
    fuzzy_num_hashes: int = 128
    """Number of MinHash permutations. Must be >= 16. Default: 128. FR-3421."""
    dedup_override_sources: list[str] = field(default_factory=list)
    """List of source_keys exempt from dedup lookup (per-run). Default: []. FR-3450."""

    # -- Parser Abstraction (Phase 2.2, FR-3301, FR-3320) --
    parser_strategy: str = "auto"
    """Parser selection: "auto" (extension-based routing), "document" (Docling),
    "code" (tree-sitter), "text" (PlainTextParser). Default: "auto". FR-3301.
    Env var: RAG_INGESTION_PARSER_STRATEGY"""
    chunker: str = "native"
    """Chunker selection: "native" (each parser's internal chunker) or
    "markdown" (force heading-aware markdown splitting for all parsers).
    Default: "native". FR-3320.
    Env var: RAG_INGESTION_CHUNKER"""

    @property
    def generate_page_images(self) -> bool:
        """Derived flag: True when visual embedding is enabled. FR-107"""
        return self.enable_visual_embedding


@dataclass
class IngestionDesignCheck:
    """Validation report for ingestion configuration design constraints.

    Attributes:
        ok: Whether the configuration passed validation.
        errors: Human-readable errors that should fail fast.
        warnings: Non-fatal issues that may affect quality or performance.
    """
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class IngestFileResult:
    """Result of a single-file ingestion run.

    Attributes:
        errors: Human-readable error messages from either phase.
        stored_count: Number of chunks successfully stored in the vector store.
        metadata_summary: LLM-generated summary of the document (if enabled).
        metadata_keywords: LLM-extracted keywords from the document (if enabled).
        processing_log: Ordered list of completed pipeline stage names.
        source_hash: SHA-256 of the original source file.
        clean_hash: SHA-256 of the cleaned/refactored text written to CleanDocumentStore.
    """
    errors: list[str]
    stored_count: int
    metadata_summary: str
    metadata_keywords: list[str]
    processing_log: list[str]
    source_hash: str
    clean_hash: str
    # -- Visual embedding extension (FR-605) --
    visual_stored_count: int = 0  # FR-605: number of visual page objects stored
    # -- Data Lifecycle additions (Task 3, Task 6) --
    trace_id: str = ""                               # UUID v4 for manifest recording (FR-3050)
    validation: dict = field(default_factory=dict)   # E2E validation result (FR-3061)


@dataclass
class IngestionRunSummary:
    """Result summary for a directory-level ingestion run.

    Attributes:
        processed: Number of sources processed (including those later failed).
        skipped: Number of sources skipped due to idempotency checks.
        failed: Number of sources that failed during processing.
        stored_chunks: Total number of chunks successfully stored.
        removed_sources: Number of sources removed from storage (when applicable).
        errors: Human-readable error messages for failures.
        design_warnings: Non-fatal configuration warnings surfaced during the run.
    """
    processed: int
    skipped: int
    failed: int
    stored_chunks: int
    removed_sources: int
    errors: list[str]
    design_warnings: list[str] = field(default_factory=list)
    # -- Data Lifecycle additions (Task 4) --
    gc_soft_deleted: int = 0       # Number of sources soft-deleted during this run
    gc_hard_deleted: int = 0       # Number of sources hard-deleted during this run
    gc_retention_purged: int = 0   # Number of expired soft-deleted entries purged


@dataclass
class Runtime:
    """Runtime container holding expensive shared ingestion dependencies.

    Attributes:
        config: Effective ingestion configuration.
        embedder: Embedding model wrapper used by storage stages.
        weaviate_client: Client used for vector and (optionally) metadata storage.
        kg_builder: Optional knowledge graph builder used by KG stages.
        db_client: Optional database client for metadata storage.
        parser_registry: Optional ParserRegistry instance. Typed as Any to avoid
            a circular import between types.py and parser_registry.py. Populated
            during pipeline initialisation in impl.py (Phase 3.2). FR-3303.
    """
    config: IngestionConfig
    embedder: LocalBGEEmbeddings
    weaviate_client: Any
    kg_builder: Optional[KnowledgeGraphBuilder]
    db_client: Optional[Any] = None
    parser_registry: Optional[Any] = None  # ParserRegistry; typed Any to avoid circular import


