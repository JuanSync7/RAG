# @summary
# Centralizes configuration settings for a RAG (Retrieval-Augmented Generation) system.
# Exports: PROJECT_ROOT, DOCUMENTS_DIR, PROCESSED_DIR, EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH, VECTOR_DB_BACKEND, VECTOR_COLLECTION_DEFAULT, WEAVIATE_COLLECTION_NAME, DATABASE_BACKEND, MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET, MINIO_SECURE, RAG_WEAVIATE_MODE, RAG_WEAVIATE_HOST, RAG_WEAVIATE_HTTP_PORT, RAG_WEAVIATE_GRPC_PORT, HYBRID_SEARCH_ALPHA, SEARCH_LIMIT, RERANK_TOP_K, CHUNK_SIZE, CHUNK_OVERLAP, QUERY_CONFIDENCE_THRESHOLD, MAX_SANITIZATION_ITERATIONS, QUERY_PROCESSING_MODEL, QUERY_PROCESSING_TEMPERATURE, QUERY_LOG_DIR, PROMPTS_DIR, DOMAIN_DESCRIPTION, KG_ENABLED, SEMANTIC_CHUNKING_ENABLED, GENERATION_ENABLED, RAG_CONFIDENCE_ROUTING_ENABLED, RAG_DOCUMENT_FORMATTING_ENABLED, RAG_NEMO_PII_GLINER_ENABLED, RAG_INGESTION_VLM_MODE, RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS, RAG_INGESTION_PERSIST_DOCLING_DOCUMENT, RAG_INGESTION_ENABLE_VISUAL_EMBEDDING, RAG_INGESTION_VISUAL_TARGET_COLLECTION, RAG_INGESTION_COLQWEN_MODEL, RAG_INGESTION_COLQWEN_BATCH_SIZE, RAG_INGESTION_PAGE_IMAGE_QUALITY, RAG_INGESTION_PAGE_IMAGE_MAX_DIMENSION, RAG_VISUAL_RETRIEVAL_ENABLED, RAG_VISUAL_RETRIEVAL_LIMIT, RAG_VISUAL_RETRIEVAL_MIN_SCORE, RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS, RAG_STAGE_BUDGET_VISUAL_RETRIEVAL_MS, validate_visual_retrieval_config, VALID_MODEL_PRECISIONS, EMBEDDING_PRECISION_QUERY, EMBEDDING_PRECISION_INGEST, RERANKER_PRECISION, VISUAL_RETRIEVAL_PRECISION, GENERATION_PRECISION, RAG_DOCUMENT_ROUTING_ENABLED, RAG_DOCUMENT_ROUTING_TOP_N, RAG_DOCUMENT_ROUTING_MIN_SCORE, RAG_DOCUMENT_ROUTING_PER_DOC_LEAVES, RAG_DOCUMENT_ROUTING_MAX_CANDIDATES, RAG_DOCUMENT_ROUTING_BOOST, RAG_DOCUMENT_CARD_COLLECTION, RAG_DECOMPOSITION_ENABLED, RAG_DECOMPOSITION_LLM_PRIMARY, RAG_DECOMPOSITION_LLM_TIMEOUT_SECONDS, RAG_DECOMPOSITION_MIN_SUBQUERIES, RAG_DECOMPOSITION_MAX_SUBQUERIES, RAG_INGESTION_BUILD_DOCUMENT_CARDS, RAG_INGESTION_CARD_LLM_SUMMARY, RAG_INGESTION_CARD_MAX_HEADINGS, validate_document_routing_config
# Deps: os, pathlib, logging, dotenv, json
# @end-summary
"""Centralized configuration for the RAG system."""

import logging
import os
from pathlib import Path

# --- Project Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
PROCESSED_DIR = PROJECT_ROOT / "processed"
RUNTIME_DIR = PROJECT_ROOT / ".runtime"

# --- Model Paths (Local BAAI Models) ---
EMBEDDING_MODEL_PATH = os.environ.get(
    "RAG_EMBEDDING_MODEL",
    os.path.expanduser("~/models/baai/bge-m3"),
)
RERANKER_MODEL_PATH = os.environ.get(
    "RAG_RERANKER_MODEL",
    os.path.expanduser("~/models/baai/bge-reranker-v2-m3"),
)

# --- Vector DB ---
# VECTOR_DB_BACKEND selects the active vector store implementation.
# Valid values: "weaviate" (default).
VECTOR_DB_BACKEND: str = os.environ.get("VECTOR_DB_BACKEND", "weaviate")

# --- Document DB (MinIO) ---
# DATABASE_BACKEND selects the active document store implementation.
# Valid values: "minio" (default).
DATABASE_BACKEND: str = os.environ.get("DATABASE_BACKEND", "minio")

MINIO_ENDPOINT: str = os.environ.get("RAG_MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY: str = os.environ.get("RAG_MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY: str = os.environ.get("RAG_MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET: str = os.environ.get("RAG_MINIO_BUCKET", "rag-documents")
MINIO_SECURE: bool = os.environ.get("RAG_MINIO_SECURE", "false").lower() in ("true", "1", "yes")

# --- Weaviate ---
WEAVIATE_COLLECTION_NAME = "RAGDocuments"
# VECTOR_COLLECTION_DEFAULT is the collection used when no explicit collection
# is specified. Override with RAG_VECTOR_COLLECTION_DEFAULT.
VECTOR_COLLECTION_DEFAULT: str = os.environ.get(
    "RAG_VECTOR_COLLECTION_DEFAULT", WEAVIATE_COLLECTION_NAME
)
# Weaviate embedded mode stores data here
WEAVIATE_DATA_DIR = os.environ.get(
    "RAG_WEAVIATE_DATA_DIR",
    str(PROJECT_ROOT / ".weaviate_data"),
)

# Connection mode: "embedded" runs Weaviate in-process (dev default, CI stubs
# rely on this). "networked" connects to a shared Weaviate container.
RAG_WEAVIATE_MODE: str = os.environ.get("RAG_WEAVIATE_MODE", "embedded").lower()
RAG_WEAVIATE_HOST: str = os.environ.get("RAG_WEAVIATE_HOST", "localhost")
RAG_WEAVIATE_HTTP_PORT: int = int(os.environ.get("RAG_WEAVIATE_HTTP_PORT", "8090"))
RAG_WEAVIATE_GRPC_PORT: int = int(os.environ.get("RAG_WEAVIATE_GRPC_PORT", "50051"))
if RAG_WEAVIATE_MODE not in ("embedded", "networked"):
    raise ValueError(
        f"RAG_WEAVIATE_MODE must be 'embedded' or 'networked', got {RAG_WEAVIATE_MODE!r}"
    )

# --- Hybrid Search ---
# Alpha: 0.0 = pure BM25, 1.0 = pure vector, 0.5 = balanced
HYBRID_SEARCH_ALPHA = 0.5
# Single source of truth for retrieval sizing. Env-driven so a deployment can
# raise them; server request schemas (server/schemas.py) derive their defaults
# from these, and rag_chain floors client-supplied values up to them.
SEARCH_LIMIT = int(os.environ.get("RAG_SEARCH_LIMIT", "10"))
RERANK_TOP_K = int(os.environ.get("RAG_RERANK_TOP_K", "5"))
# Drop thin / heading-only candidates (titles, ToC leaders) shorter than this
# many chars before rerank. 0 = disabled. These chunks win dense cosine on
# topical queries but carry no answer content; see rag_chain thin-chunk filter.
RERANK_MIN_CHARS = int(os.environ.get("RAG_RERANK_MIN_CHARS", "0"))
# Drop *navigational* candidates before rerank even when they exceed
# RERANK_MIN_CHARS: table-of-contents / index lines (dotted ".... page" leaders)
# and chapter/part front-matter stubs ("Read this chapter for a description
# of ..."). These carry no answer content but win both dense similarity and the
# cross-encoder reranker on topical queries (observed: an ARM spec ToC stub
# scored 0.95 while the actual answer chunk scored 0.51), starving the LLM of
# real body chunks. The candidate floor still guarantees >= rerank_top_k items.
RERANK_DROP_NAVIGATIONAL = os.environ.get(
    "RAG_RERANK_DROP_NAVIGATIONAL", "true"
).lower() in ("true", "1", "yes")
# A front-matter pointer stub is only treated as navigational when its total
# length is <= this many chars (real body chunks that merely *mention* "see
# chapter X" are far longer and are kept). ToC dotted-leader detection is
# length-independent.
RERANK_NAV_MAX_CHARS = int(os.environ.get("RAG_RERANK_NAV_MAX_CHARS", "320"))
# Floor client-supplied search_limit/rerank_top_k up to the configured
# SEARCH_LIMIT/RERANK_TOP_K so UI clients that hardcode small values (e.g. the
# console presets 10/5) still retrieve enough candidates for the reranker to
# surface body chunks that rank beyond a small top-K. No-op at repo defaults
# (10/5); active when a deployment sets SEARCH_LIMIT/RERANK_TOP_K higher.
RETRIEVAL_ENFORCE_CONFIG_FLOOR = os.environ.get(
    "RAG_RETRIEVAL_ENFORCE_CONFIG_FLOOR", "true"
).lower() in ("true", "1", "yes")

# --- Document Processing ---
# CHUNK_SIZE / CHUNK_OVERLAP apply to the **LangChain fallback** chunker
# (RecursiveCharacterTextSplitter). They are CHARACTER counts, not token
# counts. Kept roughly equivalent in tokens to the Docling HybridChunker
# default RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS=1024 at ~3.5 chars/token
# for English (BGE-M3 tokenizer). If you change one, check the other.
# Docling's HybridChunker has no overlap parameter — it splits at structural
# boundaries and uses heading breadcrumbs to bridge context, so overlap only
# matters for the char-window fallback path.
CHUNK_SIZE = 3500
CHUNK_OVERLAP = 350
# Minimum chunk size (chars) for the markdown/semantic chunker's coalescing
# pass. Semantic chunking splits a section wherever consecutive-sentence cosine
# drops below RAG_SEMANTIC_THRESHOLD; on heterogeneous, table-dense content
# (e.g. ARM spec coherency tables) nearly every adjacent pair is dissimilar, so
# without a floor it emits one tiny chunk per sentence — one CHI spec section
# exploded to 13,758 ~90-char fragments vs 1,488 clean chunks non-semantic.
# Adjacent sub-min segments are merged (up to CHUNK_SIZE) so no sub-min chunk is
# emitted unless it is the only content of its section.
MIN_CHUNK_CHARS = int(os.environ.get("RAG_MIN_CHUNK_CHARS", "512"))

# --- Query Processing (LangGraph) ---
QUERY_CONFIDENCE_THRESHOLD = float(
    os.environ.get("RAG_QUERY_CONFIDENCE_THRESHOLD", "0.6")
)
MAX_SANITIZATION_ITERATIONS = int(
    os.environ.get("RAG_MAX_QUERY_ITERATIONS", "3")
)
QUERY_PROCESSING_MODEL = os.environ.get("RAG_QUERY_MODEL", None)  # defaults to OLLAMA_MODEL (set below)
QUERY_PROCESSING_TEMPERATURE = float(
    os.environ.get("RAG_QUERY_TEMPERATURE", "0.15")
)
QUERY_PROCESSING_TIMEOUT = int(
    os.environ.get("RAG_QUERY_PROCESSING_TIMEOUT", "30")
)
QUERY_LOG_DIR = PROJECT_ROOT / "logs"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
DOMAIN_DESCRIPTION = os.environ.get(
    "RAG_DOMAIN_DESCRIPTION",
    "This knowledge base covers information retrieval, NLP, machine learning, "
    "embeddings, vector search, language models, and related AI/ML topics. "
    "Interpret all acronyms and abbreviations in this domain context.",
)

# --- Knowledge Graph (retrieval-side flag) ---
# Set to False to disable KG query expansion at retrieval time (pure hybrid search only).
# Ingest-side KG behavior lives in KGWeave; see RAG_INGESTION_ENABLE_KG_PHASE2B for the
# RagWeave→KGWeave Temporal handoff toggle.
KG_ENABLED = os.environ.get("RAG_KG_ENABLED", "true").lower() in ("true", "1", "yes")

# --- Semantic Chunking ---
SEMANTIC_CHUNKING_ENABLED = os.environ.get(
    "RAG_SEMANTIC_CHUNKING", "true"
).lower() in ("true", "1", "yes")
SEMANTIC_SIMILARITY_THRESHOLD = float(
    os.environ.get("RAG_SEMANTIC_THRESHOLD", "0.75")
)

# --- GLiNER (PII detector) ---
# Used only by the NeMo Guardrails PII shim (src/guardrails/shared/gliner_pii.py).
GLINER_MODEL_PATH = os.environ.get(
    "RAG_GLINER_MODEL",
    os.path.expanduser("~/models/gliner/gliner_medium-v2.1"),
)

# --- Service Ports (canonical defaults — override via .env) ---
_OLLAMA_PORT = os.environ.get("RAG_OLLAMA_PORT", "11434")
_REDIS_PORT = os.environ.get("RAG_REDIS_PORT", "6379")
_TEMPORAL_PORT = os.environ.get("RAG_TEMPORAL_PORT", "7233")

# --- LLM Generation (Ollama — legacy) ---
GENERATION_ENABLED = os.environ.get(
    "RAG_GENERATION_ENABLED", "true"
).lower() in ("true", "1", "yes")
OLLAMA_BASE_URL = os.environ.get("RAG_OLLAMA_URL", f"http://localhost:{_OLLAMA_PORT}")
OLLAMA_MODEL = os.environ.get("RAG_OLLAMA_MODEL", "qwen2.5:3b")
GENERATION_MAX_TOKENS = int(os.environ.get("RAG_GENERATION_MAX_TOKENS", "1024"))
GENERATION_TEMPERATURE = float(os.environ.get("RAG_GENERATION_TEMPERATURE", "0.3"))
# When false, do NOT send a JSON response_format to the LLM. Reasoning models
# (e.g. qwopus) emit a degenerate "{}" under guided-JSON decoding; free-text
# generation + free-text answer parsing is the reliable path.
GENERATION_STRUCTURED_OUTPUT = os.environ.get(
    "RAG_GENERATION_STRUCTURED_OUTPUT", "true"
).lower() in ("true", "1", "yes")

# --- LLM Provider (LiteLLM) ─────────────────────────────────────────
# Unified config. Falls back to legacy Ollama vars for backward compat.
_legacy_ollama_model = os.environ.get("RAG_OLLAMA_MODEL", "qwen2.5:3b")
_legacy_ollama_url = os.environ.get("RAG_OLLAMA_URL", f"http://localhost:{_OLLAMA_PORT}")

LLM_MODEL = os.environ.get(
    "RAG_LLM_MODEL",
    f"ollama/{_legacy_ollama_model}",
)
LLM_API_BASE = os.environ.get("RAG_LLM_API_BASE", _legacy_ollama_url)
LLM_API_KEY = os.environ.get("RAG_LLM_API_KEY", "")
LLM_MAX_TOKENS = int(os.environ.get("RAG_LLM_MAX_TOKENS", str(GENERATION_MAX_TOKENS)))
LLM_TEMPERATURE = float(os.environ.get("RAG_LLM_TEMPERATURE", str(GENERATION_TEMPERATURE)))
LLM_NUM_RETRIES = int(os.environ.get("RAG_LLM_NUM_RETRIES", "3"))
LLM_FALLBACK_MODELS = [
    m.strip()
    for m in os.environ.get("RAG_LLM_FALLBACK_MODELS", "").split(",")
    if m.strip()
]
_legacy_vision_model = os.environ.get("RAG_INGESTION_VISION_MODEL", "qwen2.5vl:3b")
LLM_VISION_MODEL = os.environ.get(
    "RAG_LLM_VISION_MODEL",
    f"ollama/{_legacy_vision_model}",
)
LLM_ROUTER_CONFIG = os.environ.get("RAG_LLM_ROUTER_CONFIG", "")

# --- Token Budget ---
TOKEN_BUDGET_DEFAULT_CONTEXT_LENGTH = int(
    os.environ.get("RAG_TOKEN_BUDGET_DEFAULT_CONTEXT_LENGTH", "2048")
)
TOKEN_BUDGET_CHARS_PER_TOKEN = int(
    os.environ.get("RAG_TOKEN_BUDGET_CHARS_PER_TOKEN", "4")
)
TOKEN_BUDGET_WARN_PERCENT = int(
    os.environ.get("RAG_TOKEN_BUDGET_WARN_PERCENT", "70")
)
TOKEN_BUDGET_CRITICAL_PERCENT = int(
    os.environ.get("RAG_TOKEN_BUDGET_CRITICAL_PERCENT", "90")
)

# Query-processing model alias — defaults to the primary LLM model.
# Used by the retrieval query processor for reformulation/evaluation.
_legacy_query_model = os.environ.get("RAG_QUERY_MODEL", None)
LLM_QUERY_MODEL = os.environ.get(
    "RAG_LLM_QUERY_MODEL",
    f"ollama/{_legacy_query_model}" if _legacy_query_model else LLM_MODEL,
)

# Resolve forward reference: query processing model defaults to generation model
if QUERY_PROCESSING_MODEL is None:
    QUERY_PROCESSING_MODEL = OLLAMA_MODEL

# --- Reliability / Retry ---
RETRY_PROVIDER = os.environ.get("RAG_RETRY_PROVIDER", "local")
RETRY_MAX_ATTEMPTS = int(os.environ.get("RAG_RETRY_MAX_ATTEMPTS", "3"))
RETRY_INITIAL_BACKOFF_SECONDS = float(
    os.environ.get("RAG_RETRY_INITIAL_BACKOFF_SECONDS", "0.5")
)
RETRY_MAX_BACKOFF_SECONDS = float(
    os.environ.get("RAG_RETRY_MAX_BACKOFF_SECONDS", "5.0")
)
RETRY_BACKOFF_MULTIPLIER = float(
    os.environ.get("RAG_RETRY_BACKOFF_MULTIPLIER", "2.0")
)

# --- Temporal (optional) ---
TEMPORAL_TARGET_HOST = os.environ.get("RAG_TEMPORAL_TARGET_HOST", f"localhost:{_TEMPORAL_PORT}")
TEMPORAL_TASK_QUEUE = os.environ.get("RAG_TEMPORAL_TASK_QUEUE", "rag-reliability")
RAG_INGEST_TEMPORAL_DOC_TIMEOUT_MIN = int(os.environ.get("RAG_INGEST_TEMPORAL_DOC_TIMEOUT_MIN", "15"))
RAG_INGEST_TEMPORAL_EMB_TIMEOUT_MIN = int(os.environ.get("RAG_INGEST_TEMPORAL_EMB_TIMEOUT_MIN", "30"))
RAG_INGEST_TEMPORAL_RETRY_MAX = int(os.environ.get("RAG_INGEST_TEMPORAL_RETRY_MAX", "3"))
RAG_INGEST_TEMPORAL_RETRY_INTERVAL_S = int(os.environ.get("RAG_INGEST_TEMPORAL_RETRY_INTERVAL_S", "5"))

# --- Server ---
RAG_API_PORT = int(os.environ.get("RAG_API_PORT", "8000"))
RAG_API_URL = os.environ.get("RAG_API_URL", f"http://localhost:{RAG_API_PORT}")
RAG_WORKER_CONCURRENCY = int(os.environ.get("RAG_WORKER_CONCURRENCY", "4"))
RAG_API_MAX_INFLIGHT_REQUESTS = int(os.environ.get("RAG_API_MAX_INFLIGHT_REQUESTS", "64"))
RAG_API_OVERLOAD_QUEUE_TIMEOUT_MS = int(
    os.environ.get("RAG_API_OVERLOAD_QUEUE_TIMEOUT_MS", "250")
)
RAG_API_CORS_ORIGINS: list[str] = [
    o.strip()
    for o in os.environ.get("RAG_API_CORS_ORIGINS", "*").split(",")
    if o.strip()
] or ["*"]
RAG_WORKFLOW_DEFAULT_TIMEOUT_MS = int(
    os.environ.get("RAG_WORKFLOW_DEFAULT_TIMEOUT_MS", "120000")
)

# --- Auth / tenancy ---
AUTH_API_KEYS_REQUIRED = os.environ.get("RAG_AUTH_API_KEYS_REQUIRED", "false").lower() in (
    "true",
    "1",
    "yes",
)
AUTH_API_KEYS_JSON = os.environ.get("RAG_AUTH_API_KEYS_JSON", "{}")
AUTH_JWT_ENABLED = os.environ.get("RAG_AUTH_JWT_ENABLED", "false").lower() in (
    "true",
    "1",
    "yes",
)
AUTH_JWT_HS256_SECRET = os.environ.get("RAG_AUTH_JWT_HS256_SECRET", "")
DEFAULT_TENANT_ID = os.environ.get("RAG_DEFAULT_TENANT_ID", "default")
AUTH_OIDC_ENABLED = os.environ.get("RAG_AUTH_OIDC_ENABLED", "false").lower() in (
    "true",
    "1",
    "yes",
)
AUTH_OIDC_ISSUER = os.environ.get("RAG_AUTH_OIDC_ISSUER", "")
AUTH_OIDC_AUDIENCE = os.environ.get("RAG_AUTH_OIDC_AUDIENCE", "")
AUTH_OIDC_JWKS_URL = os.environ.get("RAG_AUTH_OIDC_JWKS_URL", "")
AUTH_OIDC_ROLES_CLAIM = os.environ.get("RAG_AUTH_OIDC_ROLES_CLAIM", "roles")
AUTH_OIDC_TENANT_CLAIM = os.environ.get("RAG_AUTH_OIDC_TENANT_CLAIM", "tenant_id")
AUTH_OIDC_SUBJECT_CLAIM = os.environ.get("RAG_AUTH_OIDC_SUBJECT_CLAIM", "sub")

# --- Rate limit / quotas ---
RATE_LIMIT_ENABLED = os.environ.get("RAG_RATE_LIMIT_ENABLED", "true").lower() in (
    "true",
    "1",
    "yes",
)
RATE_LIMIT_REQUESTS_PER_MINUTE = int(
    os.environ.get("RAG_RATE_LIMIT_REQUESTS_PER_MINUTE", "60")
)
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("RAG_RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_DEFAULT_TENANT_RPM = int(
    os.environ.get("RAG_RATE_LIMIT_DEFAULT_TENANT_RPM", str(RATE_LIMIT_REQUESTS_PER_MINUTE))
)
RATE_LIMIT_DEFAULT_PROJECT_RPM = int(
    os.environ.get("RAG_RATE_LIMIT_DEFAULT_PROJECT_RPM", str(RATE_LIMIT_REQUESTS_PER_MINUTE))
)

# --- API key + quota persistence ---
AUTH_API_KEYS_STORE_PATH = Path(
    os.environ.get("RAG_AUTH_API_KEYS_STORE_PATH", str(RUNTIME_DIR / "security" / "api_keys.json"))
)
AUTH_QUOTAS_STORE_PATH = Path(
    os.environ.get("RAG_AUTH_QUOTAS_STORE_PATH", str(RUNTIME_DIR / "security" / "quotas.json"))
)

# --- Caching ---
CACHE_ENABLED = os.environ.get("RAG_CACHE_ENABLED", "true").lower() in ("true", "1", "yes")
CACHE_PROVIDER = os.environ.get("RAG_CACHE_PROVIDER", "memory")
CACHE_TTL_SECONDS = int(os.environ.get("RAG_CACHE_TTL_SECONDS", "120"))
CACHE_REDIS_URL = os.environ.get("RAG_CACHE_REDIS_URL", f"redis://localhost:{_REDIS_PORT}/0")

# --- Conversation memory ---
MEMORY_ENABLED = os.environ.get("RAG_MEMORY_ENABLED", "true").lower() in ("true", "1", "yes")
MEMORY_PROVIDER = os.environ.get("RAG_MEMORY_PROVIDER", "redis").strip().lower()
MEMORY_REDIS_URL = os.environ.get("RAG_MEMORY_REDIS_URL", CACHE_REDIS_URL)
MEMORY_REDIS_PREFIX = os.environ.get("RAG_MEMORY_REDIS_PREFIX", "rag:memory")
# Timeout (seconds) for the connectivity ping issued when constructing the
# Redis-backed memory provider. If the ping fails or times out, the factory
# falls back to the no-op provider so request handlers do not 500.
MEMORY_REDIS_CONNECT_TIMEOUT_S = float(
    os.environ.get("RAG_MEMORY_REDIS_CONNECT_TIMEOUT_S", "1.0")
)
MEMORY_MAX_RECENT_TURNS = int(os.environ.get("RAG_MEMORY_MAX_RECENT_TURNS", "8"))
MEMORY_MAX_CONTEXT_TOKENS_ESTIMATE = int(
    os.environ.get("RAG_MEMORY_MAX_CONTEXT_TOKENS_ESTIMATE", "1400")
)
MEMORY_SUMMARY_TRIGGER_TURNS = int(os.environ.get("RAG_MEMORY_SUMMARY_TRIGGER_TURNS", "12"))
MEMORY_SUMMARY_MAX_SOURCE_TURNS = int(
    os.environ.get("RAG_MEMORY_SUMMARY_MAX_SOURCE_TURNS", "40")
)

# --- Retrieval controls ---
RAG_DEFAULT_FAST_PATH = os.environ.get("RAG_DEFAULT_FAST_PATH", "false").lower() in (
    "true",
    "1",
    "yes",
)
RAG_RETRIEVAL_TIMEOUT_MS = int(os.environ.get("RAG_RETRIEVAL_TIMEOUT_MS", "30000"))
RAG_STAGE_BUDGET_QUERY_PROCESSING_MS = int(
    os.environ.get("RAG_STAGE_BUDGET_QUERY_PROCESSING_MS", "12000")
)
RAG_STAGE_BUDGET_KG_EXPANSION_MS = int(
    os.environ.get("RAG_STAGE_BUDGET_KG_EXPANSION_MS", "1000")
)
RAG_STAGE_BUDGET_EMBEDDING_MS = int(os.environ.get("RAG_STAGE_BUDGET_EMBEDDING_MS", "1500"))
RAG_STAGE_BUDGET_HYBRID_SEARCH_MS = int(
    os.environ.get("RAG_STAGE_BUDGET_HYBRID_SEARCH_MS", "5000")
)
RAG_STAGE_BUDGET_RERANKING_MS = int(os.environ.get("RAG_STAGE_BUDGET_RERANKING_MS", "5000"))
RAG_STAGE_BUDGET_GENERATION_MS = int(os.environ.get("RAG_STAGE_BUDGET_GENERATION_MS", "60000"))

# --- Observability ---
OBSERVABILITY_PROVIDER = os.environ.get("RAG_OBSERVABILITY_PROVIDER", "noop")
OBSERVABILITY_SCHEMA_VERSION = "1.0"

# --- Incremental ingestion ---
INGESTION_MANIFEST_PATH = PROCESSED_DIR / "ingestion_manifest.json"

# --- Ingestion pipeline (LangGraph) ---
RAG_INGESTION_LLM_ENABLED = os.environ.get("RAG_INGESTION_LLM_ENABLED", "true").lower() in (
    "true",
    "1",
    "yes",
)
RAG_INGESTION_LLM_MODEL = os.environ.get("RAG_INGESTION_LLM_MODEL", "qwen2.5:3b")
RAG_INGESTION_LLM_TEMPERATURE = float(os.environ.get("RAG_INGESTION_LLM_TEMPERATURE", "0.1"))
RAG_INGESTION_LLM_TIMEOUT_SECONDS = int(
    os.environ.get("RAG_INGESTION_LLM_TIMEOUT_SECONDS", "45")
)
RAG_INGESTION_LLM_MAX_KEYWORDS = int(os.environ.get("RAG_INGESTION_LLM_MAX_KEYWORDS", "12"))
RAG_INGESTION_DOCLING_ENABLED = os.environ.get(
    "RAG_INGESTION_DOCLING_ENABLED", "true"
).lower() in ("true", "1", "yes")
RAG_INGESTION_DOCLING_MODEL = os.environ.get(
    "RAG_INGESTION_DOCLING_MODEL",
    "docling-parse-v2",
)
RAG_INGESTION_DOCLING_ARTIFACTS_PATH = os.environ.get(
    "RAG_INGESTION_DOCLING_ARTIFACTS_PATH",
    "",
)
RAG_INGESTION_DOCLING_STRICT = os.environ.get(
    "RAG_INGESTION_DOCLING_STRICT", "true"
).lower() in ("true", "1", "yes")
RAG_INGESTION_DOCLING_AUTO_DOWNLOAD = os.environ.get(
    "RAG_INGESTION_DOCLING_AUTO_DOWNLOAD", "true"
).lower() in ("true", "1", "yes")
RAG_INGESTION_ENABLE_OCR: bool = os.environ.get(
    "RAG_INGESTION_ENABLE_OCR", "true"
).lower() in ("true", "1", "yes")
"""Enable Docling OCR auto-trigger for image-only pages. Default: True.
Uses RapidOCR with force_full_page_ocr=False so text-native pages are not
re-OCR'd. Set False to disable OCR entirely (faster, but image-only pages
yield empty text)."""
RAG_INGESTION_OCR_ENGINE: str = os.environ.get(
    "RAG_INGESTION_OCR_ENGINE", "rapidocr"
)
"""OCR engine identifier. Only "rapidocr" is wired today; reserved for
forward-compat with easyocr/tesseract. Informational."""
RAG_INGESTION_TABLEFORMER_MODE: str = os.environ.get(
    "RAG_INGESTION_TABLEFORMER_MODE", "accurate"
)
"""TableFormer mode: "accurate" (TF v2, higher quality) or "fast" (TF v1)."""
RAG_INGESTION_TABLEFORMER_DO_CELL_MATCHING: bool = os.environ.get(
    "RAG_INGESTION_TABLEFORMER_DO_CELL_MATCHING", "true"
).lower() in ("true", "1", "yes")
"""When True, TableFormer matches PDF text cells to detected table cells
(higher fidelity). Disable to fall back to OCR-only cell content."""
RAG_INGESTION_EXPORT_EXTENSIONS = os.environ.get(
    "RAG_INGESTION_EXPORT_EXTENSIONS",
    ".txt,.md,.markdown,.rst,.html,.htm,.pdf,.docx,.pptx",
)
RAG_INGESTION_ENABLE_MULTIMODAL_PROCESSING = os.environ.get(
    "RAG_INGESTION_ENABLE_MULTIMODAL_PROCESSING", "false"
).lower() in ("true", "1", "yes")
RAG_INGESTION_VISION_ENABLED = os.environ.get(
    "RAG_INGESTION_VISION_ENABLED", "false"
).lower() in ("true", "1", "yes")
RAG_INGESTION_VISION_PROVIDER = os.environ.get(
    "RAG_INGESTION_VISION_PROVIDER",
    "ollama",
).strip()
RAG_INGESTION_VISION_MODEL = os.environ.get(
    "RAG_INGESTION_VISION_MODEL",
    "qwen2.5vl:3b",
).strip()
RAG_INGESTION_VISION_TIMEOUT_SECONDS = int(
    os.environ.get("RAG_INGESTION_VISION_TIMEOUT_SECONDS", "60")
)
RAG_INGESTION_VISION_MAX_FIGURES = int(
    os.environ.get("RAG_INGESTION_VISION_MAX_FIGURES", "4")
)
RAG_INGESTION_VISION_MAX_IMAGE_BYTES = int(
    os.environ.get("RAG_INGESTION_VISION_MAX_IMAGE_BYTES", "3145728")
)
RAG_INGESTION_VISION_TEMPERATURE = float(
    os.environ.get("RAG_INGESTION_VISION_TEMPERATURE", "0.1")
)
RAG_INGESTION_VISION_MAX_TOKENS = int(
    os.environ.get("RAG_INGESTION_VISION_MAX_TOKENS", "220")
)
RAG_INGESTION_VISION_AUTO_PULL = os.environ.get(
    "RAG_INGESTION_VISION_AUTO_PULL", "true"
).lower() in ("true", "1", "yes")
RAG_INGESTION_VISION_STRICT = os.environ.get(
    "RAG_INGESTION_VISION_STRICT", "false"
).lower() in ("true", "1", "yes")
RAG_INGESTION_VISION_API_BASE_URL = os.environ.get(
    "RAG_INGESTION_VISION_API_BASE_URL",
    "",
).strip()
RAG_INGESTION_VISION_API_KEY = os.environ.get(
    "RAG_INGESTION_VISION_API_KEY",
    "",
).strip()
RAG_INGESTION_VISION_API_PATH = os.environ.get(
    "RAG_INGESTION_VISION_API_PATH",
    "/v1/chat/completions",
).strip()
RAG_INGESTION_ENABLE_CROSS_REFERENCE_EXTRACTION = os.environ.get(
    "RAG_INGESTION_ENABLE_CROSS_REFERENCE_EXTRACTION", "true"
).lower() in ("true", "1", "yes")
RAG_INGESTION_ENABLE_QUALITY_VALIDATION = os.environ.get(
    "RAG_INGESTION_ENABLE_QUALITY_VALIDATION", "true"
).lower() in ("true", "1", "yes")
# Lossless verification: after chunking, check the emitted chunks losslessly
# cover the golden source (parsed Docling markdown). Warn-by-default; STRICT
# escalates a coverage shortfall to a hard failure of the document.
RAG_INGESTION_VERIFY_LOSSLESS = os.environ.get(
    "RAG_INGESTION_VERIFY_LOSSLESS", "true"
).lower() in ("true", "1", "yes")
RAG_INGESTION_LOSSLESS_MIN_COVERAGE = float(
    os.environ.get("RAG_INGESTION_LOSSLESS_MIN_COVERAGE", "0.98")
)
RAG_INGESTION_LOSSLESS_STRICT = os.environ.get(
    "RAG_INGESTION_LOSSLESS_STRICT", "false"
).lower() in ("true", "1", "yes")
# Production default: KG runs via Temporal handoff (KGWeave worker fleet on
# KG_TASK_QUEUE). Set to false to skip the KG phase entirely — used by the
# offline CLI ingest and benchmark scripts that don't run a Temporal cluster.
RAG_INGESTION_ENABLE_KG_PHASE2B = os.environ.get(
    "RAG_INGESTION_ENABLE_KG_PHASE2B", "true"
).lower() in ("true", "1", "yes")
RAG_INGESTION_VERBOSE_STAGE_LOGS = os.environ.get(
    "RAG_INGESTION_VERBOSE_STAGE_LOGS", "false"
).lower() in ("true", "1", "yes")
RAG_INGESTION_PERSIST_REFACTOR_MIRROR = os.environ.get(
    "RAG_INGESTION_PERSIST_REFACTOR_MIRROR", "true"
).lower() in ("true", "1", "yes")
RAG_INGESTION_MIRROR_DIR = PROCESSED_DIR / "refactor_mirror"

# --- Docling-Native Chunking Pipeline ---

RAG_INGESTION_VLM_MODE: str = os.environ.get(
    "RAG_INGESTION_VLM_MODE", "disabled"
)
"""VLM mode for figure image description.
Valid values: "disabled", "builtin", "external".
"builtin" runs SmolVLM at parse time inside DocumentConverter.
"external" calls LiteLLM-routed vision model post-chunking.
"""

RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS: int = int(
    os.environ.get("RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS", "1024")
)
"""Maximum token count per chunk for HybridChunker.

Default 1024 — empirical sweet spot for technical-doc retrieval. BGE-M3
itself accepts inputs up to 8192 tokens, but past ~1500–2000 the embedding
loses specificity (a single dense vector covers too many subtopics) and
single-sentence chunks (under ~50 tokens) lack contextual grounding.
1024 keeps natural section units whole while staying comfortably below
the dilution threshold."""

RAG_INGESTION_CHUNKER: str = os.environ.get("RAG_INGESTION_CHUNKER", "native").lower()
"""Selects the chunking backend used by the embedding pipeline (chunking_node).

- ``"native"`` (default): each parser's own structure-aware chunker — for
  documents this is Docling's HybridChunker (``merge_peers=True``, token-bounded)
  plus adaptive table chunking (``table_summary``/``table_row`` with captions)
  and heading-breadcrumb ``contextualize()``. Requires ``parse_result`` +
  ``parser_instance`` to be threaded from Phase 1 into Phase 2.
- ``"markdown"``: parser-abstraction markdown chunker over ``parse_result``.
- ``"legacy"``: the pre-parser-abstraction MarkdownHeaderTextSplitter +
  semantic/character fallback on ``cleaned_text`` (no table awareness, no
  contextualize). The historical default-by-accident — used whenever
  ``parse_result`` was absent.

NOTE: until the Phase-1->Phase-2 handoff was fixed to forward
``parse_result``/``parser_instance``, ``"native"``/``"markdown"`` were
unreachable and every document silently fell back to ``"legacy"``. Set this to
``"legacy"`` to restore the old behavior."""

RAG_INGESTION_TABLE_EMBED_PREPEND_SECTION_PATH: bool = os.environ.get(
    "RAG_INGESTION_TABLE_EMBED_PREPEND_SECTION_PATH", "true"
).lower() in ("true", "1", "yes")
"""If True (default), prepend the section-heading breadcrumb to the EMBEDDED text
of table_summary / table_row / figure chunks, mirroring what HybridChunker's
contextualize() already does for prose chunks. Without it these structured chunks
embed heading-blind (only ``Columns:.../Rows:N`` or a bare caption), so on
topical/heading queries (e.g. "GPIO pin multiplexing") the prose paragraph — which
carries its embedded breadcrumb — outranks the table that actually holds the
answer, at BOTH the dense and reranker stages. ``table_markdown`` metadata is left
unchanged. Changes embedded text -> chunk_id, so flipping this requires a
re-ingest / collection recreate."""

RAG_INGESTION_DROP_NAVIGATIONAL: bool = os.environ.get(
    "RAG_INGESTION_DROP_NAVIGATIONAL", "true"
).lower() in ("true", "1", "yes")
"""If True (default), drop table-of-contents / index / front-matter pointer chunks
at INGEST time (not just at query time). ~17% of the spec corpus is navigational
ToC/front-matter that out-scores real answer chunks on both dense cosine and the
reranker (a ToC stub scored 0.95 vs the answer's 0.51), causing "I cannot answer"
refusals. The query-time rerank filter (RAG_RERANK_DROP_NAVIGATIONAL) is kept as
defense-in-depth, but stripping at ingest is the single-source-of-truth fix: it
shrinks the index, cuts embedding cost, and protects consumers that bypass the
query filter (deep-research path, eval harnesses, alternate rerankers). Uses the
shared ``is_navigational`` predicate so emission and filtering stay symmetric. A
per-document over-prune guard never removes a document's entire chunk set."""

RAG_INGESTION_NAV_MAX_CHARS: int = int(
    os.environ.get("RAG_INGESTION_NAV_MAX_CHARS", "320")
)
"""Length cap (chars) below which the front-matter pointer-phrase branch of the
navigational test fires at ingest. Mirrors the query-side RERANK_NAV_MAX_CHARS so
both gates treat the same chunks as navigational. The dotted-leader (ToC) branch is
length-independent and unaffected by this cap."""

# Adaptive table-chunking knobs (env-wired so they can be tuned on the live
# Temporal worker; previously hardcoded dataclass defaults — audit
# F6-adaptive-table-knobs-no-env-wiring).
RAG_INGESTION_ENABLE_ADAPTIVE_TABLE_CHUNKING: bool = os.environ.get(
    "RAG_INGESTION_ENABLE_ADAPTIVE_TABLE_CHUNKING", "true"
).lower() in ("true", "1", "yes")
"""If True (default), DoclingParser.chunk() replaces the raw table-dominant chunk
with token-budget row-block chunks: each chunk restates the header and packs as
many whole rows as fit ``RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS``, so every row is
captured (no row/col gate, no summary cap, no truncation)."""

RAG_INGESTION_PERSIST_DOCLING_DOCUMENT: bool = os.environ.get(
    "RAG_INGESTION_PERSIST_DOCLING_DOCUMENT", "true"
).lower() in ("true", "1", "yes")
"""If True (default), persist DoclingDocument JSON to CleanDocumentStore.
Set to false to trade storage for compute (re-parse in Phase 2)."""

RAG_INGESTION_USE_DOCLING_CHUNKER_FOR_MARKDOWN: bool = os.environ.get(
    "RAG_INGESTION_USE_DOCLING_CHUNKER_FOR_MARKDOWN", "true"
).lower() in ("true", "1", "yes")
"""If True (default), .md/.html/.rst files use Docling's HybridChunker
(structure-aware: preserves lists, tables, and requirement blocks as atomic
units). Set to false to fall back to the legacy LangChain char-splitter."""

RAG_INGESTION_STORE_FIGURES_IN_DB: bool = os.environ.get(
    "RAG_INGESTION_STORE_FIGURES_IN_DB", "false"
).lower() in ("true", "1", "yes")
"""If True, copy markdown image bytes to the configured document store
(currently MinIO) at ingest time and record a stable ``store_key`` in chunk
metadata so citation UIs can presign and render even if the original source
moves. Default False — chunks carry only the relative ``src`` path; UI
resolves against the original source location.

Backend-agnostic flag — actual upload destination is whatever document store
the pipeline is configured for. Flip on when the doc store should act as the
canonical figure store."""

# --- Ingestion Hardening: Document Parsing Abstraction (FR-3301, FR-3320) ---
RAG_INGESTION_PARSER_STRATEGY: str = os.environ.get(
    "RAG_INGESTION_PARSER_STRATEGY", "auto"
)
"""Parser selection strategy: "auto" | "document" | "code" | "text".
"auto" routes by file extension via ParserRegistry. FR-3301."""

# NOTE: RAG_INGESTION_CHUNKER is defined once, above (with ``.lower()`` so an
# uppercase env value like "Native" normalizes instead of failing
# verify_core_design's value check). A second, lower-casing-less definition used
# to live here and silently won by being later in the module — removed (audit
# F3-chunker-double-def-no-lower).

# --- Ingestion Hardening: Data Lifecycle — MinIO Clean Store + GC ---
RAG_INGESTION_CLEAN_STORE_BUCKET: str = os.environ.get(
    "RAG_INGESTION_CLEAN_STORE_BUCKET", ""
)
"""MinIO bucket for the clean store. Empty string reuses the primary
target_bucket. Data Lifecycle T1."""

RAG_GC_MODE: str = os.environ.get("RAG_GC_MODE", "soft")
"""Default GC delete mode: "soft" (default) or "hard". FR-3020."""

RAG_GC_RETENTION_DAYS: int = int(os.environ.get("RAG_GC_RETENTION_DAYS", "30"))
"""Days before soft-deleted entries are eligible for hard purge. FR-3020."""

RAG_GC_SCHEDULE: str = os.environ.get("RAG_GC_SCHEDULE", "")
"""Cron expression for scheduled GC runs. Empty string disables scheduling."""

# --- Ingestion Hardening: Cross-Document Dedup (FR-3400-3433) ---
RAGWEAVE_ENABLE_CROSS_DOC_DEDUP: bool = os.environ.get(
    "RAGWEAVE_ENABLE_CROSS_DOC_DEDUP", "true"
).lower() in ("true", "1", "yes")
"""Master toggle for cross-document dedup (Tier 1 exact + optional Tier 2).
Default True. FR-3400."""

RAGWEAVE_ENABLE_FUZZY_DEDUP: bool = os.environ.get(
    "RAGWEAVE_ENABLE_FUZZY_DEDUP", "false"
).lower() in ("true", "1", "yes")
"""Enable Tier 2 MinHash fuzzy matching. Default False (Tier 1 only).
Requires the datasketch package. FR-3420."""

RAGWEAVE_FUZZY_THRESHOLD: float = float(
    os.environ.get("RAGWEAVE_FUZZY_THRESHOLD", "0.95")
)
"""Jaccard similarity threshold for Tier 2 fuzzy dedup. Default 0.95. FR-3421."""

RAGWEAVE_FUZZY_SHINGLE_SIZE: int = int(
    os.environ.get("RAGWEAVE_FUZZY_SHINGLE_SIZE", "3")
)
"""Word n-gram size for MinHash shingles. Default 3."""

RAGWEAVE_FUZZY_NUM_HASHES: int = int(
    os.environ.get("RAGWEAVE_FUZZY_NUM_HASHES", "128")
)
"""MinHash permutation count. Higher = better accuracy, more RAM. Default 128."""

# --- Ingestion Hardening: Orchestration Dual Queue + Priority ---
RAG_INGEST_USER_TASK_QUEUE: str = os.environ.get(
    "RAG_INGEST_USER_TASK_QUEUE", ""
)
"""Task queue name for user-triggered ingestion. Empty string + empty
RAG_INGEST_BACKGROUND_TASK_QUEUE = single-queue legacy mode.
Typical production value: "ingest-user"."""

RAG_INGEST_BACKGROUND_TASK_QUEUE: str = os.environ.get(
    "RAG_INGEST_BACKGROUND_TASK_QUEUE", ""
)
"""Task queue name for background/batch ingestion.
Typical production value: "ingest-background"."""

RAG_INGEST_USER_SLOTS: int = int(os.environ.get("RAG_INGEST_USER_SLOTS", "3"))
"""Max concurrent activities for the user-queue worker. Default 3."""

RAG_INGEST_BACKGROUND_SLOTS: int = int(
    os.environ.get("RAG_INGEST_BACKGROUND_SLOTS", "1")
)
"""Max concurrent activities for the background-queue worker. Default 1."""

RAG_INGEST_WORKER_CONCURRENCY: int = int(
    os.environ.get("RAG_INGEST_WORKER_CONCURRENCY", "4")
)
"""Legacy total worker concurrency. Used to derive 75/25 user/background
split when RAG_INGEST_USER_SLOTS / RAG_INGEST_BACKGROUND_SLOTS are unset."""

RAG_INGEST_PRIORITY_HIGH: int = int(os.environ.get("RAG_INGEST_PRIORITY_HIGH", "1"))
"""High priority value for trigger_type=single (user interactive)."""

RAG_INGEST_PRIORITY_MEDIUM: int = int(os.environ.get("RAG_INGEST_PRIORITY_MEDIUM", "2"))
"""Medium priority value for trigger_type=batch."""

RAG_INGEST_PRIORITY_LOW: int = int(os.environ.get("RAG_INGEST_PRIORITY_LOW", "3"))
"""Low priority value for trigger_type=background (bulk/scheduled)."""

# --- Guardrails ---
# GUARDRAIL_BACKEND selects the active guardrail implementation.
# Valid values: "nemo" (default), "" or "none" (disabled).
# RAG_NEMO_ENABLED is deprecated — use GUARDRAIL_BACKEND instead.
GUARDRAIL_BACKEND: str = os.environ.get("GUARDRAIL_BACKEND", "nemo" if os.environ.get("RAG_NEMO_ENABLED", "false").lower() in ("true", "1", "yes") else "")

# --- NeMo Guardrails ---
RAG_NEMO_ENABLED = os.environ.get(
    "RAG_NEMO_ENABLED", "false"
).lower() in ("true", "1", "yes")
RAG_NEMO_CONFIG_DIR = os.environ.get(
    "RAG_NEMO_CONFIG_DIR",
    str(PROJECT_ROOT / "config" / "guardrails"),
)
RAG_NEMO_INJECTION_ENABLED = os.environ.get(
    "RAG_NEMO_INJECTION_ENABLED", "true"
).lower() in ("true", "1", "yes")
RAG_NEMO_INJECTION_SENSITIVITY = os.environ.get(
    "RAG_NEMO_INJECTION_SENSITIVITY", "balanced"
)
RAG_NEMO_PII_ENABLED = os.environ.get(
    "RAG_NEMO_PII_ENABLED", "true"
).lower() in ("true", "1", "yes")
RAG_NEMO_PII_EXTENDED = os.environ.get(
    "RAG_NEMO_PII_EXTENDED", "false"
).lower() in ("true", "1", "yes")
RAG_NEMO_TOXICITY_ENABLED = os.environ.get(
    "RAG_NEMO_TOXICITY_ENABLED", "true"
).lower() in ("true", "1", "yes")
RAG_NEMO_TOXICITY_THRESHOLD = float(
    os.environ.get("RAG_NEMO_TOXICITY_THRESHOLD", "0.5")
)
RAG_NEMO_FAITHFULNESS_ENABLED = os.environ.get(
    "RAG_NEMO_FAITHFULNESS_ENABLED", "true"
).lower() in ("true", "1", "yes")
RAG_NEMO_FAITHFULNESS_THRESHOLD = float(
    os.environ.get("RAG_NEMO_FAITHFULNESS_THRESHOLD", "0.5")
)
RAG_NEMO_FAITHFULNESS_ACTION = os.environ.get(
    "RAG_NEMO_FAITHFULNESS_ACTION", "flag"
)
RAG_NEMO_OUTPUT_PII_ENABLED = os.environ.get(
    "RAG_NEMO_OUTPUT_PII_ENABLED", "true"
).lower() in ("true", "1", "yes")
RAG_NEMO_OUTPUT_TOXICITY_ENABLED = os.environ.get(
    "RAG_NEMO_OUTPUT_TOXICITY_ENABLED", "true"
).lower() in ("true", "1", "yes")
RAG_NEMO_INTENT_CONFIDENCE_THRESHOLD = float(
    os.environ.get("RAG_NEMO_INTENT_CONFIDENCE_THRESHOLD", "0.5")
)
RAG_NEMO_RAIL_TIMEOUT_SECONDS = int(
    os.environ.get("RAG_NEMO_RAIL_TIMEOUT_SECONDS", "10")
)
# Injection: NeMo jailbreak heuristics (perplexity-based)
RAG_NEMO_INJECTION_PERPLEXITY_ENABLED = os.environ.get(
    "RAG_NEMO_INJECTION_PERPLEXITY_ENABLED", "true"
).lower() in ("true", "1", "yes")
RAG_NEMO_INJECTION_MODEL_ENABLED = os.environ.get(
    "RAG_NEMO_INJECTION_MODEL_ENABLED", "true"
).lower() in ("true", "1", "yes")
RAG_NEMO_INJECTION_LP_THRESHOLD = float(
    os.environ.get("RAG_NEMO_INJECTION_LP_THRESHOLD", "89.79")
)
RAG_NEMO_INJECTION_PS_PPL_THRESHOLD = float(
    os.environ.get("RAG_NEMO_INJECTION_PS_PPL_THRESHOLD", "1845.65")
)
# PII: Presidio score threshold
RAG_NEMO_PII_SCORE_THRESHOLD = float(
    os.environ.get("RAG_NEMO_PII_SCORE_THRESHOLD", "0.4")
)
# Topic safety: LLM-based on/off-topic detection
RAG_NEMO_TOPIC_SAFETY_ENABLED = os.environ.get(
    "RAG_NEMO_TOPIC_SAFETY_ENABLED", "true"
).lower() in ("true", "1", "yes")
RAG_NEMO_TOPIC_SAFETY_INSTRUCTIONS = os.environ.get(
    "RAG_NEMO_TOPIC_SAFETY_INSTRUCTIONS", ""
)
# Faithfulness: NeMo self-check-facts approach
RAG_NEMO_FAITHFULNESS_SELF_CHECK = os.environ.get(
    "RAG_NEMO_FAITHFULNESS_SELF_CHECK", "true"
).lower() in ("true", "1", "yes")
# GLiNER supplementary PII detection (entity-based NER: PERSON, ORG, LOCATION)
RAG_NEMO_PII_GLINER_ENABLED = os.environ.get(
    "RAG_NEMO_PII_GLINER_ENABLED", "false"
).lower() in ("true", "1", "yes")

# --- Guardian classifier (shared judge model used by multiple rails) ---
# RAG_GUARDIAN_PROVIDER picks the judge model. Valid: "granite", "self_check", "".
# Granite Guardian is preferred in production; self_check uses the project LLM.
RAG_GUARDIAN_ENABLED = os.environ.get(
    "RAG_GUARDIAN_ENABLED", "false"
).lower() in ("true", "1", "yes")
RAG_GUARDIAN_PROVIDER = os.environ.get("RAG_GUARDIAN_PROVIDER", "granite")
# "transformers" loads the model locally; "vllm" calls an OpenAI-compat endpoint.
RAG_GUARDIAN_MODE = os.environ.get("RAG_GUARDIAN_MODE", "vllm")
RAG_GUARDIAN_MODEL_ID = os.environ.get(
    "RAG_GUARDIAN_MODEL_ID", "ibm-granite/granite-guardian-3.2-5b"
)
RAG_GUARDIAN_ENDPOINT = os.environ.get("RAG_GUARDIAN_ENDPOINT", "")
RAG_GUARDIAN_API_KEY = os.environ.get("RAG_GUARDIAN_API_KEY", "")
RAG_GUARDIAN_TIMEOUT_S = float(os.environ.get("RAG_GUARDIAN_TIMEOUT_S", "5.0"))
RAG_GUARDIAN_THRESHOLD = float(os.environ.get("RAG_GUARDIAN_THRESHOLD", "0.5"))

# --- Composite Confidence Routing ---
RAG_CONFIDENCE_ROUTING_ENABLED = os.environ.get(
    "RAG_CONFIDENCE_ROUTING_ENABLED", "false"
).lower() in ("true", "1", "yes")
RAG_CONFIDENCE_HIGH_THRESHOLD = float(
    os.environ.get("RAG_CONFIDENCE_HIGH_THRESHOLD", "0.70")
)
RAG_CONFIDENCE_LOW_THRESHOLD = float(
    os.environ.get("RAG_CONFIDENCE_LOW_THRESHOLD", "0.50")
)
RAG_CONFIDENCE_RETRIEVAL_WEIGHT = float(
    os.environ.get("RAG_CONFIDENCE_RETRIEVAL_WEIGHT", "0.50")
)
RAG_CONFIDENCE_LLM_WEIGHT = float(
    os.environ.get("RAG_CONFIDENCE_LLM_WEIGHT", "0.25")
)
RAG_CONFIDENCE_CITATION_WEIGHT = float(
    os.environ.get("RAG_CONFIDENCE_CITATION_WEIGHT", "0.25")
)
RAG_CONFIDENCE_RE_RETRIEVE_MAX_RETRIES = int(
    os.environ.get("RAG_CONFIDENCE_RE_RETRIEVE_MAX_RETRIES", "1")
)

# --- Retrieval quality classification thresholds ---
# Reranker best-score thresholds that map to the four retrieval_quality
# labels ("strong", "moderate", "weak", "insufficient").
RAG_RETRIEVAL_QUALITY_STRONG_THRESHOLD = float(
    os.environ.get("RAG_RETRIEVAL_QUALITY_STRONG_THRESHOLD", "0.75")
)
RAG_RETRIEVAL_QUALITY_MODERATE_THRESHOLD = float(
    os.environ.get("RAG_RETRIEVAL_QUALITY_MODERATE_THRESHOLD", "0.50")
)
RAG_RETRIEVAL_QUALITY_WEAK_THRESHOLD = float(
    os.environ.get("RAG_RETRIEVAL_QUALITY_WEAK_THRESHOLD", "0.30")
)

# --- Inference Backend ---
# "local": BGE models run in-process (dev venv path; requires the `local-embed`
#          pyproject extra — torch + sentence-transformers + transformers).
# "tei":   BGE models served by separate TEI containers (rag-embed / rag-rerank)
#          via direct HTTP. Compose default. Worker image does NOT ship torch.
INFERENCE_BACKEND: str = os.environ.get("RAG_INFERENCE_BACKEND", "local").strip().lower()
_VALID_INFERENCE_BACKENDS = {"local", "tei"}
if INFERENCE_BACKEND not in _VALID_INFERENCE_BACKENDS:
    raise ValueError(
        f"RAG_INFERENCE_BACKEND={INFERENCE_BACKEND!r} is not valid; "
        f"must be one of {sorted(_VALID_INFERENCE_BACKENDS)}"
    )

# TEI service URLs — used when INFERENCE_BACKEND="tei".
# Default routes through rag-nginx so replicas can be scaled horizontally
# (`docker compose up -d --scale rag-embed=N`). Override to direct service DNS
# (http://rag-embed:80) for single-replica dev or to bypass the LB.
TEI_EMBED_URL: str = os.environ.get("RAG_TEI_EMBED_URL", "http://rag-nginx:8081")
TEI_RERANK_URL: str = os.environ.get("RAG_TEI_RERANK_URL", "http://rag-nginx:8082")

# Model IDs loaded by the TEI containers (HuggingFace repo IDs). BGE-M3 gives
# mature multilingual retrieval; BGE-reranker-v2-m3 is the matching cross-encoder.
TEI_EMBEDDING_MODEL: str = os.environ.get("RAG_TEI_EMBEDDING_MODEL", "BAAI/bge-m3")
TEI_RERANKER_MODEL: str = os.environ.get("RAG_TEI_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# Timeout for HTTP inference calls (embed + rerank). Should exceed the
# model's p99 latency, including GPU warm-up on first call.
TEI_TIMEOUT_SECONDS: int = int(os.environ.get("RAG_TEI_TIMEOUT_SECONDS", "30"))

# --- Reranker ---
RERANKER_MAX_LENGTH = int(os.environ.get("RAG_RERANKER_MAX_LENGTH", "512"))
# Maximum number of (query, document) pairs per tokenizer/model forward pass.
# Reduce to lower peak VRAM usage when SEARCH_LIMIT is large.
RERANKER_BATCH_SIZE = int(os.environ.get("RAG_RERANKER_BATCH_SIZE", "32"))

# --- Model precision modes (per-model-surface) ---
# Supported precisions for neural model surfaces in the pipeline. Each key
# selects the numeric dtype (and, for int8/int4, the quantization scheme) used
# for that model's weights and activations. Defaults are "fp32" to preserve
# baseline behavior; iterations that flip a key to a lower precision MUST
# pass the regression guard before being accepted.
#
# Mapping per surface:
#   EMBEDDING_PRECISION_QUERY     — BGE query-time embedder (LocalBGEEmbeddings)
#   EMBEDDING_PRECISION_INGEST    — BGE ingest-time embedder (same model class,
#                                    separate config so ingest can be fp32 while
#                                    query is fp16, or vice versa)
#   RERANKER_PRECISION            — BGE reranker (LocalBGEReranker)
#   VISUAL_RETRIEVAL_PRECISION    — ColQwen2 visual embedder (ingest and query)
#   GENERATION_PRECISION          — advisory. Generation goes through the
#                                    Ollama HTTP API, so actual precision is
#                                    baked into the Ollama model tag (e.g.,
#                                    "qwen2.5:3b-instruct-q4_K_M"). This key
#                                    documents the expected precision for
#                                    observability/SLO reasoning; it does not
#                                    reconfigure the remote model.
#
# Allowed values (case-insensitive, lowered at validation):
#   "fp32"  — float32 (default, maximum accuracy)
#   "fp16"  — float16 (NVIDIA Tensor Core fast path)
#   "bf16"  — bfloat16 (Ampere+; trades range for stability vs fp16)
#   "int8"  — 8-bit integer quantization (bitsandbytes / torch.quantization)
#   "int4"  — 4-bit integer quantization (bitsandbytes / GPTQ)
VALID_MODEL_PRECISIONS = frozenset({"fp32", "fp16", "bf16", "int8", "int4"})


def _read_precision_env(env_key: str, default: str = "fp32") -> str:
    """Read a precision setting from the environment, validate, and return normalized value.

    Returns the lowercased value if valid; otherwise logs a warning and
    falls back to ``default``. Raising would break imports for operators
    who set a typo — silent fallback + startup warning is friendlier.
    """
    raw = os.environ.get(env_key, default)
    value = raw.strip().lower()
    if value not in VALID_MODEL_PRECISIONS:
        import warnings
        warnings.warn(
            f"{env_key}={raw!r} is not a valid precision mode "
            f"(allowed: {sorted(VALID_MODEL_PRECISIONS)}). Falling back to {default!r}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return default
    return value


EMBEDDING_PRECISION_QUERY = _read_precision_env("RAG_EMBEDDING_PRECISION_QUERY")
EMBEDDING_PRECISION_INGEST = _read_precision_env("RAG_EMBEDDING_PRECISION_INGEST")
# iter 005: reranker flipped to fp16 after iter 004 landed SDPA. BGE
# reranker-v2-m3 tolerates fp16 well (documented by BAAI); scores differ
# from fp32 in the ~3rd decimal but top-K ordering is stable on
# well-separated KG queries. Drift-guarded by the regression check.
RERANKER_PRECISION = _read_precision_env("RAG_RERANKER_PRECISION", default="fp16")
VISUAL_RETRIEVAL_PRECISION = _read_precision_env("RAG_VISUAL_RETRIEVAL_PRECISION")
GENERATION_PRECISION = _read_precision_env("RAG_GENERATION_PRECISION")  # advisory

# --- Document Formatting ---
RAG_DOCUMENT_FORMATTING_ENABLED = os.environ.get(
    "RAG_DOCUMENT_FORMATTING_ENABLED", "false"
).lower() in ("true", "1", "yes")

# --- Cross-reference (xref) expansion (MVP, default-off) ---
# One-hop expansion that resolves chunk-text cross-references (e.g. "§3.1",
# "see Section 4") into adjacent chunks at retrieval time. Mirrors the
# table-group expansion architecture (src.retrieval.xref_expansion).
# Scoped MVP: only ``section`` / ``section_symbol`` ref types are resolved;
# table/figure/appendix refs require a caption registry and remain TODO.
#
# Default flipped to ON 2026-05-23 after the ESP32-S3 datasheet soak:
# table refs resolved 98.8% (559/566), confirming the expander is safe to
# enable by default. Operators can still disable it via env var.
RAG_XREF_EXPANSION_ENABLED: bool = os.environ.get(
    "RAG_XREF_EXPANSION_ENABLED", "true"
).lower() in ("true", "1", "yes")
RAG_XREF_MAX_PER_HIT: int = int(os.environ.get("RAG_XREF_MAX_PER_HIT", "2"))
RAG_XREF_MAX_TOTAL: int = int(os.environ.get("RAG_XREF_MAX_TOTAL", "6"))

# Whether ingest-time xref extraction emits ``figure`` refs. FIG-4
# (2026-05-24) flipped the default to "true" after the forward-walked
# caption-binding fallback (``_build_unbound_caption_fallback``) lifted
# the ESP32-S3 resolvable_rate from 28.57% to 85.71% (well above the 30%
# bar). The remaining unresolved cases are prose section-references like
# "Figure 4.1.2 …" miscategorised by ``cross_refs`` — tracked as a
# follow-up in ``docs/soak/figure_artifacts_esp32_2026-05-24_triage.md``.
# Operators can still disable via env var.
RAG_XREF_EXTRACT_FIGURE_REFS: bool = os.environ.get(
    "RAG_XREF_EXTRACT_FIGURE_REFS", "true"
).lower() in ("true", "1", "yes")

# --- Visual Embedding Pipeline ---
RAG_INGESTION_ENABLE_VISUAL_EMBEDDING: bool = os.environ.get(
    "RAG_INGESTION_ENABLE_VISUAL_EMBEDDING", "false"
) in ("true", "1", "yes")  # FR-101, FR-109

RAG_INGESTION_VISUAL_TARGET_COLLECTION: str = os.environ.get(
    "RAG_INGESTION_VISUAL_TARGET_COLLECTION", "RAGVisualPages"
)  # FR-102, FR-109

RAG_INGESTION_COLQWEN_MODEL: str = os.environ.get(
    "RAG_INGESTION_COLQWEN_MODEL", "vidore/colqwen2-v1.0"
)  # FR-103, FR-109

RAG_INGESTION_COLQWEN_BATCH_SIZE: int = int(os.environ.get(
    "RAG_INGESTION_COLQWEN_BATCH_SIZE", "4"
))  # FR-104, FR-109

RAG_INGESTION_EMBEDDING_BATCH_SIZE: int = int(os.environ.get(
    "RAGWEAVE_EMBEDDING_BATCH_SIZE", "64"
))  # FR-1211

RAG_INGESTION_PAGE_IMAGE_QUALITY: int = int(os.environ.get(
    "RAG_INGESTION_PAGE_IMAGE_QUALITY", "85"
))  # FR-105, FR-109

RAG_INGESTION_PAGE_IMAGE_MAX_DIMENSION: int = int(os.environ.get(
    "RAG_INGESTION_PAGE_IMAGE_MAX_DIMENSION", "1024"
))  # FR-106, FR-109

# --- Visual Retrieval Pipeline (retrieval-side) ---

_visual_retrieval_logger = logging.getLogger(__name__)

RAG_VISUAL_RETRIEVAL_ENABLED: bool = os.environ.get(
    "RAG_VISUAL_RETRIEVAL_ENABLED", "false"
).lower() in ("true", "1", "yes")  # FR-101

_raw_visual_limit = int(os.environ.get("RAG_VISUAL_RETRIEVAL_LIMIT", "5"))
if _raw_visual_limit < 1 or _raw_visual_limit > 50:
    _visual_retrieval_logger.warning(
        "RAG_VISUAL_RETRIEVAL_LIMIT=%d out of range [1, 50]; clamping.",
        _raw_visual_limit,
    )
RAG_VISUAL_RETRIEVAL_LIMIT: int = max(1, min(50, _raw_visual_limit))  # FR-103

RAG_VISUAL_RETRIEVAL_MIN_SCORE: float = float(
    os.environ.get("RAG_VISUAL_RETRIEVAL_MIN_SCORE", "0.3")
)  # FR-105

RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS: int = int(
    os.environ.get("RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS", "3600")
)  # FR-107

RAG_STAGE_BUDGET_VISUAL_RETRIEVAL_MS: int = int(
    os.environ.get("RAG_STAGE_BUDGET_VISUAL_RETRIEVAL_MS", "10000")
)  # FR-617

# ---------------------------------------------------------------------------
# Deep research (recursive topic-grouped retrieval) — opt-in alternative to
# the linear kg_expand → embed → hybrid_search → rerank stages. See
# src/retrieval/pipeline/deep_research.py for the orchestrator.
# ---------------------------------------------------------------------------

RAG_STAGE_BUDGET_DEEP_RESEARCH_MS: int = int(
    os.environ.get("RAG_STAGE_BUDGET_DEEP_RESEARCH_MS", "30000")
)
RAG_DEEP_RESEARCH_MAX_NODES: int = int(
    os.environ.get("RAG_DEEP_RESEARCH_MAX_NODES", "12")
)
RAG_DEEP_RESEARCH_MAX_LLM_CALLS: int = int(
    os.environ.get("RAG_DEEP_RESEARCH_MAX_LLM_CALLS", "24")
)
RAG_DEEP_RESEARCH_MAX_DEPTH: int = int(
    os.environ.get("RAG_DEEP_RESEARCH_MAX_DEPTH", "3")
)
RAG_DEEP_RESEARCH_MAX_TOPICS: int = int(
    os.environ.get("RAG_DEEP_RESEARCH_MAX_TOPICS", "3")
)
RAG_DEEP_RESEARCH_PER_TOPIC_QUESTIONS: int = int(
    os.environ.get("RAG_DEEP_RESEARCH_PER_TOPIC_QUESTIONS", "3")
)
RAG_DEEP_RESEARCH_GRAPH_CONTEXT_MAX_CHARS: int = int(
    os.environ.get("RAG_DEEP_RESEARCH_GRAPH_CONTEXT_MAX_CHARS", "4000")
)
RAG_DEEP_RESEARCH_KB_TOP_PER_NODE: int = int(
    os.environ.get("RAG_DEEP_RESEARCH_KB_TOP_PER_NODE", "10")
)
RAG_DEEP_RESEARCH_WALL_CLOCK_MS: int = int(
    os.environ.get("RAG_DEEP_RESEARCH_WALL_CLOCK_MS", "30000")
)
RAG_DEEP_RESEARCH_EARLY_STOP_MIN_CHUNKS: int = int(
    os.environ.get("RAG_DEEP_RESEARCH_EARLY_STOP_MIN_CHUNKS", "3")
)
RAG_DEEP_RESEARCH_EARLY_STOP_MIN_SOURCES: int = int(
    os.environ.get("RAG_DEEP_RESEARCH_EARLY_STOP_MIN_SOURCES", "2")
)
RAG_DEEP_RESEARCH_PER_TOPIC_RERANK_MIN_CONFIDENCE: float = float(
    os.environ.get("RAG_DEEP_RESEARCH_PER_TOPIC_RERANK_MIN_CONFIDENCE", "0.6")
)
RAG_DEEP_RESEARCH_PER_TOPIC_TOP_K: int = int(
    os.environ.get("RAG_DEEP_RESEARCH_PER_TOPIC_TOP_K", "3")
)

# --- Ingest: Embedding pipeline ---
RAG_INGEST_EMBEDDING_BATCH_MAX_RETRIES: int = int(
    os.environ.get("RAG_INGEST_EMBEDDING_BATCH_MAX_RETRIES", "3")
)
RAG_INGEST_EMBEDDING_BATCH_RETRY_DELAY_S: float = float(
    os.environ.get("RAG_INGEST_EMBEDDING_BATCH_RETRY_DELAY_S", "0.3")
)

# --- Ingest: LLM metadata extraction ---
RAG_INGEST_LLM_METADATA_MAX_CHARS: int = int(
    os.environ.get("RAG_INGEST_LLM_METADATA_MAX_CHARS", "10000")
)
RAG_INGEST_LLM_SUMMARY_MAX_LEN: int = int(
    os.environ.get("RAG_INGEST_LLM_SUMMARY_MAX_LEN", "240")
)

# --- Ingest: Document structure detection ---
RAG_INGEST_STRUCTURE_MAX_FIGURES: int = int(
    os.environ.get("RAG_INGEST_STRUCTURE_MAX_FIGURES", "32")
)

# --- Ingest: Quality scoring heuristic ---
RAG_INGEST_QUALITY_BASE: float = float(
    os.environ.get("RAG_INGEST_QUALITY_BASE", "0.4")
)
RAG_INGEST_QUALITY_BONUS: float = float(
    os.environ.get("RAG_INGEST_QUALITY_BONUS", "0.2")
)
RAG_INGEST_QUALITY_MIN_BONUS_LEN: int = int(
    os.environ.get("RAG_INGEST_QUALITY_MIN_BONUS_LEN", "120")
)
RAG_INGEST_QUALITY_PER_CHAR: float = float(
    os.environ.get("RAG_INGEST_QUALITY_PER_CHAR", "0.01")
)
RAG_INGEST_QUALITY_MAX: float = float(
    os.environ.get("RAG_INGEST_QUALITY_MAX", "1.0")
)
RAG_INGEST_ANCHOR_PREVIEW_LEN: int = int(
    os.environ.get("RAG_INGEST_ANCHOR_PREVIEW_LEN", "220")
)
RAG_INGEST_PARA_PREVIEW_LEN: int = int(
    os.environ.get("RAG_INGEST_PARA_PREVIEW_LEN", "600")
)
RAG_INGEST_MIN_PARA_SIMILARITY: float = float(
    os.environ.get("RAG_INGEST_MIN_PARA_SIMILARITY", "0.45")
)

# --- Ingest: Clean store ---
RAG_INGEST_CLEAN_STORE_MAX_ATTEMPTS: int = int(
    os.environ.get("RAG_INGEST_CLEAN_STORE_MAX_ATTEMPTS", "10")
)

# --- Ingest: Fuzzy dedup (MinHash) ---
RAG_INGEST_FUZZY_SHINGLE_SIZE: int = int(
    os.environ.get("RAG_INGEST_FUZZY_SHINGLE_SIZE", "3")
)
RAG_INGEST_FUZZY_NUM_HASHES: int = int(
    os.environ.get("RAG_INGEST_FUZZY_NUM_HASHES", "128")
)
RAG_INGEST_FUZZY_SIMILARITY_THRESHOLD: float = float(
    os.environ.get("RAG_INGEST_FUZZY_SIMILARITY_THRESHOLD", "0.95")
)

# --- Ingest: Chunk quality gates ---
RAG_INGEST_MIN_CHUNK_CHARS: int = int(
    os.environ.get("RAG_INGEST_MIN_CHUNK_CHARS", "40")
)
RAG_INGEST_MIN_QUALITY_SCORE: float = float(
    os.environ.get("RAG_INGEST_MIN_QUALITY_SCORE", "0.45")
)
# Native (Docling HybridChunker) min-chunk COALESCE floor (chars). HybridChunker's
# merge_peers pass only merges chunks that share the SAME heading path and only up
# to the token budget — it has no minimum-size floor, so a leaf section with a tiny
# body (a one-line cross-pointer, a stub paragraph, an isolated sentence) is emitted
# as its own heading-dominated chunk that pollutes retrieval (the "title-only chunk"
# pathology). After chunking, DoclingParser merges adjacent sub-floor BODIES into the
# neighbour sharing the longest heading-path prefix (same chapter/parent), capped at
# CHUNK_SIZE chars, preserving the host chunk's breadcrumb/page/xref provenance.
# 0 disables the pass. Distinct from RAG_INGEST_MIN_CHUNK_CHARS (a post-coalesce DROP
# backstop for truly-orphan stubs, measured on the body) and RAG_MIN_CHUNK_CHARS
# (the legacy markdown/semantic-path coalesce floor).
RAG_INGEST_NATIVE_MIN_CHUNK_CHARS: int = int(
    os.environ.get("RAG_INGEST_NATIVE_MIN_CHUNK_CHARS", "512")
)

# --- Ingest: Temporal workflow (KG phase 2b) ---
RAG_INGEST_TEMPORAL_KG_TIMEOUT_MIN: int = int(
    os.environ.get("RAG_INGEST_TEMPORAL_KG_TIMEOUT_MIN", "10")
)
RAG_INGEST_TEMPORAL_KG_RETRY_MAX: int = int(
    os.environ.get("RAG_INGEST_TEMPORAL_KG_RETRY_MAX", "5")
)
RAG_INGEST_TEMPORAL_KG_RETRY_INTERVAL_S: int = int(
    os.environ.get("RAG_INGEST_TEMPORAL_KG_RETRY_INTERVAL_S", "10")
)

# --- Retrieval: Query processor (rewriter / evaluator) ---
RAG_RETRIEVAL_REWRITER_MAX_TOKENS: int = int(
    os.environ.get("RAG_RETRIEVAL_REWRITER_MAX_TOKENS", "400")
)
RAG_QUERY_PRONOUN_DENSITY_THRESHOLD: float = float(
    os.environ.get("RAG_QUERY_PRONOUN_DENSITY_THRESHOLD", "0.15")
)
RAG_QUERY_EVALUATOR_MAX_TOKENS: int = int(
    os.environ.get("RAG_QUERY_EVALUATOR_MAX_TOKENS", "256")
)

# --- Retrieval: Generation / sanitization ---
RAG_GENERATION_MIN_FRAGMENT_LEN: int = int(
    os.environ.get("RAG_GENERATION_MIN_FRAGMENT_LEN", "40")
)

# --- Retrieval: Query suggestion heuristic ---
RAG_QUERY_SUGGESTION_MIN_UNIQUE_SOURCES: int = int(
    os.environ.get("RAG_QUERY_SUGGESTION_MIN_UNIQUE_SOURCES", "2")
)

# --- Core: Embedding batch sizes ---
RAG_EMBEDDING_BATCH_SIZE_DOCUMENTS: int = int(
    os.environ.get("RAG_EMBEDDING_BATCH_SIZE_DOCUMENTS", "32")
)
RAG_EMBEDDING_BATCH_SIZE_SEMANTIC_CHUNKING: int = int(
    os.environ.get("RAG_EMBEDDING_BATCH_SIZE_SEMANTIC_CHUNKING", "64")
)

# --- Platform: LLM defaults & probes ---
RAG_LLM_DEFAULT_MAX_TOKENS: int = int(
    os.environ.get("RAG_LLM_DEFAULT_MAX_TOKENS", "2048")
)
RAG_LLM_DEFAULT_TEMPERATURE: float = float(
    os.environ.get("RAG_LLM_DEFAULT_TEMPERATURE", "0.2")
)
RAG_LLM_DEFAULT_NUM_RETRIES: int = int(
    os.environ.get("RAG_LLM_DEFAULT_NUM_RETRIES", "3")
)
RAG_LLM_RATE_LIMIT_RETRY_DELAY_S: int = int(
    os.environ.get("RAG_LLM_RATE_LIMIT_RETRY_DELAY_S", "5")
)
RAG_LLM_TOKEN_COUNTING_PROBE_TOKENS: int = int(
    os.environ.get("RAG_LLM_TOKEN_COUNTING_PROBE_TOKENS", "1")
)
RAG_LLM_ONESHOT_MAX_TOKENS: int = int(
    os.environ.get("RAG_LLM_ONESHOT_MAX_TOKENS", "256")
)

# --- Platform: Memory text limits ---
RAG_MEMORY_SUMMARY_MAX_CHARS: int = int(
    os.environ.get("RAG_MEMORY_SUMMARY_MAX_CHARS", "2400")
)
RAG_MEMORY_TURN_MAX_CHARS: int = int(
    os.environ.get("RAG_MEMORY_TURN_MAX_CHARS", "1200")
)
RAG_MEMORY_SNIPPET_MAX_CHARS: int = int(
    os.environ.get("RAG_MEMORY_SNIPPET_MAX_CHARS", "220")
)

# --- Platform: Observability ---
RAG_OBSERVABILITY_MAX_CAPTURE_LEN: int = int(
    os.environ.get("RAG_OBSERVABILITY_MAX_CAPTURE_LEN", "500")
)

# --- Platform: Reliability / retry defaults ---
RAG_RELIABILITY_DEFAULT_MAX_ATTEMPTS: int = int(
    os.environ.get("RAG_RELIABILITY_DEFAULT_MAX_ATTEMPTS", "3")
)
RAG_RELIABILITY_INITIAL_BACKOFF_S: float = float(
    os.environ.get("RAG_RELIABILITY_INITIAL_BACKOFF_S", "0.5")
)
RAG_RELIABILITY_MAX_BACKOFF_S: float = float(
    os.environ.get("RAG_RELIABILITY_MAX_BACKOFF_S", "5.0")
)
RAG_RELIABILITY_BACKOFF_MULTIPLIER: float = float(
    os.environ.get("RAG_RELIABILITY_BACKOFF_MULTIPLIER", "2.0")
)

# --- DB: MinIO ---
RAG_MINIO_PRESIGNED_URL_EXPIRY_SECONDS: int = int(
    os.environ.get("RAG_MINIO_PRESIGNED_URL_EXPIRY_SECONDS", "3600")
)
RAG_MINIO_LIST_DEFAULT_LIMIT: int = int(
    os.environ.get("RAG_MINIO_LIST_DEFAULT_LIMIT", "1000")
)
RAG_MINIO_LIST_DEFAULT_OFFSET: int = int(
    os.environ.get("RAG_MINIO_LIST_DEFAULT_OFFSET", "0")
)
RAG_INGEST_PAGE_IMAGE_JPEG_QUALITY: int = int(
    os.environ.get("RAG_INGEST_PAGE_IMAGE_JPEG_QUALITY", "85")
)

# --- Guardrails ---
RAG_GUARDRAILS_PII_SCORE_THRESHOLD: float = float(
    os.environ.get("RAG_GUARDRAILS_PII_SCORE_THRESHOLD", "0.4")
)
RAG_GUARDRAILS_INTENT_CONFIDENCE_THRESHOLD: float = float(
    os.environ.get("RAG_GUARDRAILS_INTENT_CONFIDENCE_THRESHOLD", "0.5")
)
RAG_GUARDRAILS_SELF_CHECK_THRESHOLD: float = float(
    os.environ.get("RAG_GUARDRAILS_SELF_CHECK_THRESHOLD", "0.5")
)
RAG_GUARDRAILS_INTENT_ASYNC_TIMEOUT_S: int = int(
    os.environ.get("RAG_GUARDRAILS_INTENT_ASYNC_TIMEOUT_S", "10")
)
RAG_GUARDRAILS_INPUT_RAIL_POOL_MAX_WORKERS: int = int(
    os.environ.get("RAG_GUARDRAILS_INPUT_RAIL_POOL_MAX_WORKERS", "5")
)
RAG_GUARDRAILS_OUTPUT_RAIL_POOL_MAX_WORKERS: int = int(
    os.environ.get("RAG_GUARDRAILS_OUTPUT_RAIL_POOL_MAX_WORKERS", "3")
)
RAG_GUARDRAILS_FAITHFULNESS_POOL_MAX_WORKERS: int = int(
    os.environ.get("RAG_GUARDRAILS_FAITHFULNESS_POOL_MAX_WORKERS", "2")
)
RAG_GUARDRAILS_INJECTION_LP_THRESHOLD: float = float(
    os.environ.get("RAG_GUARDRAILS_INJECTION_LP_THRESHOLD", "89.79")
)
RAG_GUARDRAILS_INJECTION_PS_PPL_THRESHOLD: float = float(
    os.environ.get("RAG_GUARDRAILS_INJECTION_PS_PPL_THRESHOLD", "1845.65")
)

# --- Ingest: misc support helpers ---
RAG_INGEST_STRIP_TRAILING_SHORT_LINES_MAX_WORDS: int = int(
    os.environ.get("RAG_INGEST_STRIP_TRAILING_SHORT_LINES_MAX_WORDS", "4")
)
RAG_INGEST_LLM_DEFAULT_MAX_TOKENS: int = int(
    os.environ.get("RAG_INGEST_LLM_DEFAULT_MAX_TOKENS", "300")
)
RAG_INGEST_SCORER_PYTEST_TIMEOUT_S: int = int(
    os.environ.get("RAG_INGEST_SCORER_PYTEST_TIMEOUT_S", "300")
)

# --- Platform: Memory (extended) ---
RAG_MEMORY_SANITIZE_DEFAULT_MAX_CHARS: int = int(
    os.environ.get("RAG_MEMORY_SANITIZE_DEFAULT_MAX_CHARS", "1600")
)
RAG_MEMORY_HEURISTIC_SUMMARY_MAX_CHARS: int = int(
    os.environ.get("RAG_MEMORY_HEURISTIC_SUMMARY_MAX_CHARS", "1800")
)
RAG_MEMORY_CONTEXT_MAX_CHARS: int = int(
    os.environ.get("RAG_MEMORY_CONTEXT_MAX_CHARS", "5000")
)
RAG_MEMORY_LLM_SUMMARIZER_MAX_TOKENS: int = int(
    os.environ.get("RAG_MEMORY_LLM_SUMMARIZER_MAX_TOKENS", "512")
)
RAG_MEMORY_LLM_SUMMARY_SANITIZED_MAX_CHARS: int = int(
    os.environ.get("RAG_MEMORY_LLM_SUMMARY_SANITIZED_MAX_CHARS", "2600")
)
RAG_MEMORY_CONVERSATION_TITLE_MAX_CHARS: int = int(
    os.environ.get("RAG_MEMORY_CONVERSATION_TITLE_MAX_CHARS", "200")
)
RAG_MEMORY_GET_TURNS_DEFAULT_LIMIT: int = int(
    os.environ.get("RAG_MEMORY_GET_TURNS_DEFAULT_LIMIT", "100")
)

# --- Platform: Token budget ---
RAG_TOKEN_BUDGET_URLOPEN_TIMEOUT_S: int = int(
    os.environ.get("RAG_TOKEN_BUDGET_URLOPEN_TIMEOUT_S", "5")
)

# --- Platform: LLM cache ---
RAG_LLM_CACHE_DEFAULT_TTL_S: int = int(
    os.environ.get("RAG_LLM_CACHE_DEFAULT_TTL_S", "3600")
)

# --- Retrieval: query/pipeline (extended) ---
RAG_QUERY_KG_MATCH_MAX_TERMS: int = int(
    os.environ.get("RAG_QUERY_KG_MATCH_MAX_TERMS", "20")
)
RAG_DEEP_RESEARCH_EVIDENCE_TEXT_MAX_CHARS: int = int(
    os.environ.get("RAG_DEEP_RESEARCH_EVIDENCE_TEXT_MAX_CHARS", "8000")
)
RAG_RETRIEVAL_INIT_POOL_MAX_WORKERS: int = int(
    os.environ.get("RAG_RETRIEVAL_INIT_POOL_MAX_WORKERS", "3")
)
RAG_RETRIEVAL_STAGE1_POOL_MAX_WORKERS: int = int(
    os.environ.get("RAG_RETRIEVAL_STAGE1_POOL_MAX_WORKERS", "2")
)
RAG_RETRIEVAL_EMBEDDING_CACHE_MAX_SIZE: int = int(
    os.environ.get("RAG_RETRIEVAL_EMBEDDING_CACHE_MAX_SIZE", "128")
)

# NOTE: RAG_INGESTION_COLQWEN_MODEL is reused for retrieval-time model
# selection (FR-109). No separate retrieval model key exists.

# ─── Tree retrieval (TREE_RETRIEVAL_DESIGN.md §5) ─────────────────────────
RAG_TREE_RETRIEVAL_ENABLED: bool = os.environ.get(
    "RAG_TREE_RETRIEVAL_ENABLED", "false"
).lower() in ("true", "1", "yes")
"""Enable Stage 4b (descent) and Stage 4c (lift) sub-stages. When False
(default), retrieval behaviour matches pre-1.2.0 — leaf-only hybrid search."""

RAG_TREE_DESCENT_TOP_K: int = max(1, int(
    os.environ.get("RAG_TREE_DESCENT_TOP_K", "5")
))
"""Top-K section nodes returned by Stage 4b descent search."""

RAG_TREE_DESCENT_LEAVES_PER_SECTION: int = max(1, int(
    os.environ.get("RAG_TREE_DESCENT_LEAVES_PER_SECTION", "3")
))
"""Number of leaf chunks pulled from each top-K section during descent expansion."""

RAG_TREE_DESCENT_DOC_DIVERSITY_TOP_PER_DOC: int = max(1, int(
    os.environ.get("RAG_TREE_DESCENT_DOC_DIVERSITY_TOP_PER_DOC", "2")
))
"""Per-document cap applied to descent section hits before expansion. Prevents
one verbose document from dominating cross-document descent."""

RAG_TREE_LIFT_SEED_K: int = max(1, int(
    os.environ.get("RAG_TREE_LIFT_SEED_K", "5")
))
"""Number of top Stage 4 chunk hits used as seeds for Stage 4c sibling lift."""

RAG_TREE_LIFT_SIBLINGS: int = max(1, int(
    os.environ.get("RAG_TREE_LIFT_SIBLINGS", "2")
))
"""Per-seed cap on sibling chunks fetched during lift. Chip-design profile
recommends 4 (see TREE_RETRIEVAL_DESIGN.md §4.2.1)."""

RAG_STAGE_BUDGET_TREE_DESCENT_MS: int = int(
    os.environ.get("RAG_STAGE_BUDGET_TREE_DESCENT_MS", "200")
)
RAG_STAGE_BUDGET_TREE_LIFT_MS: int = int(
    os.environ.get("RAG_STAGE_BUDGET_TREE_LIFT_MS", "200")
)

# ─── Rerank fusion (R1 — BM25-RRF + heading-aware + anchor confidence) ─────
# These knobs control the post-CE fusion layer in
# ``src.retrieval.query.nodes.rerank_fusion``. Disabling fusion (or setting
# every weight to 0) returns the rerank stage to pure cross-encoder behavior.

RAG_RERANK_FUSION_ENABLED: bool = os.environ.get(
    "RAG_RERANK_FUSION_ENABLED", "true"
).lower() in ("true", "1", "yes")
"""Master switch for rerank-time fusion. When False, the rerank stage uses
pure cross-encoder scores (legacy behavior). Defaults True."""

RAG_RERANK_DIVERSITY_ENABLED: bool = os.environ.get(
    "RAG_RERANK_DIVERSITY_ENABLED", "true"
).lower() in ("true", "1", "yes")
"""Master switch for the per-document diversity cap applied to the final
reranked top-K. When True (default), no single ``document_id`` may occupy more
than ``ceil(rerank_top_k * RAG_RERANK_MAX_DOC_FRACTION)`` of the returned
chunks; the freed slots go to the next-best chunks from OTHER documents.
Greedy-by-score with backfill, so a genuinely single-source answer is never
starved (over-cap chunks are re-added when no diverse candidates remain).

Motivation: a single multi-sheet spreadsheet chunked into many near-duplicate
rows could win every reranked slot for a topical query, evicting the prose
chunk that actually answered it (one document crowding out all others). The
ingest-side bounded-block table chunking is the primary cure; this cap is the
query-side safety net so no document can ever monopolise retrieval again."""

RAG_RERANK_MAX_DOC_FRACTION: float = float(
    os.environ.get("RAG_RERANK_MAX_DOC_FRACTION", "0.75")
)
"""Maximum fraction of the reranked top-K that one ``document_id`` may occupy
when ``RAG_RERANK_DIVERSITY_ENABLED`` is True. 0.75 → at most 9 of 12 (or 3 of
5) from a single document: enough cross-document diversity to surface a second
source (validated to recover questions whose answer lived in a non-dominant
doc) while still letting a strongly-relevant document supply most of the
context (a tighter 0.6 starved depth-heavy comparison answers). Set to 1.0 to
disable the cap without touching the master switch."""

RAG_RERANK_RRF_K: int = max(1, int(
    os.environ.get("RAG_RERANK_RRF_K", "60")
))
"""Reciprocal Rank Fusion dampening constant ``k``. 60 is the canonical default."""

RAG_RERANK_RRF_LAMBDA: float = float(
    os.environ.get("RAG_RERANK_RRF_LAMBDA", "1.0")
)
"""Weight applied to the BM25-RRF component when fusion is enabled."""

RAG_RERANK_HEADING_LAMBDA: float = float(
    os.environ.get("RAG_RERANK_HEADING_LAMBDA", "0.15")
)
"""Weight applied to heading-match score (lambda_heading)."""

RAG_RERANK_ANCHOR_LAMBDA: float = float(
    os.environ.get("RAG_RERANK_ANCHOR_LAMBDA", "0.10")
)
"""Weight applied to anchor-confidence (tree-mode only). Ignored when
the candidate is not a tree-retrieved leaf (``_anchor_rank is None``)."""

RAG_RERANK_ANCHOR_K: int = max(1, int(
    os.environ.get("RAG_RERANK_ANCHOR_K", "10")
))
"""Anchor-confidence dampening constant: ``confidence = 1 / (k + rank)``."""


# ─── Document Routing (RAPTOR-lite) — DOCUMENT_ROUTING_DESIGN.md §6, §7 ───
# Stage-1 card-index routing. All OFF / no-op by default: when
# RAG_DOCUMENT_ROUTING_ENABLED is False, retrieval behaviour is byte-identical
# to pre-routing. Routing is SOFT (top-N, never top-1; never a hard filter) and
# falls back to pure flat retrieval on low confidence (design §6.2, §7).

RAG_DOCUMENT_ROUTING_ENABLED: bool = os.environ.get(
    "RAG_DOCUMENT_ROUTING_ENABLED", "false"
).lower() in ("true", "1", "yes")
"""Master switch for Stage-1 document routing. When False (default), the router
is never invoked and retrieval is unchanged."""

RAG_DOCUMENT_ROUTING_TOP_N: int = int(
    os.environ.get("RAG_DOCUMENT_ROUTING_TOP_N", "6")
)
"""Number of routed documents per query (design §7: never top-1).
Validated >= 2 by ``validate_document_routing_config``."""

RAG_DOCUMENT_ROUTING_MIN_SCORE: float = float(
    os.environ.get("RAG_DOCUMENT_ROUTING_MIN_SCORE", "0.0")
)
"""Card-similarity floor below which routing yields an empty set (→ pure flat
retrieval). Default 0.0 = accept any card hit."""

RAG_DOCUMENT_ROUTING_PER_DOC_LEAVES: int = int(
    os.environ.get("RAG_DOCUMENT_ROUTING_PER_DOC_LEAVES", "5")
)
"""Chunks fetched per routed document during the routed-doc candidate union."""

RAG_DOCUMENT_ROUTING_MAX_CANDIDATES: int = int(
    os.environ.get("RAG_DOCUMENT_ROUTING_MAX_CANDIDATES", "60")
)
"""Upper bound on the unioned candidate set (design §6.2). Validated
>= RAG_DOCUMENT_ROUTING_PER_DOC_LEAVES by ``validate_document_routing_config``."""

RAG_DOCUMENT_ROUTING_BOOST: float = float(
    os.environ.get("RAG_DOCUMENT_ROUTING_BOOST", "0.0")
)
"""Tiny optional tie-break boost for routed-doc candidates. Default 0.0 = no
score influence (rerank decides). Soft only — never a hard filter."""

RAG_DOCUMENT_CARD_COLLECTION: str = os.environ.get(
    "RAG_DOCUMENT_CARD_COLLECTION", "RAGDocumentCards"
)
"""Vector collection holding routing-only document cards (NEVER sent to the
LLM). Shared between ingest (card emission) and retrieval (the router)."""

# ─── Comparison Decomposition — DOCUMENT_ROUTING_DESIGN.md §8 ─────────────
# Tier-1 regex + Tier-2 glossary-LLM query decomposition for comparison
# queries. OFF by default; identity transform (query unchanged) when disabled.

RAG_DECOMPOSITION_ENABLED: bool = os.environ.get(
    "RAG_DECOMPOSITION_ENABLED", "false"
).lower() in ("true", "1", "yes")
"""Master switch for comparison-query decomposition. When False (default),
``decompose_query`` is an identity transform (returns ``[query]``)."""

RAG_DECOMPOSITION_LLM_PRIMARY: bool = os.environ.get(
    "RAG_DECOMPOSITION_LLM_PRIMARY", "true"
).lower() in ("true", "1", "yes")
"""When True (default) and decomposition is enabled, Tier-2 glossary-LLM
decomposition is the primary path, with Tier-1 regex as fallback."""

RAG_DECOMPOSITION_LLM_TIMEOUT_SECONDS: int = int(
    os.environ.get("RAG_DECOMPOSITION_LLM_TIMEOUT_SECONDS", "8")
)
"""Per-call timeout for the Tier-2 decomposition LLM call."""

RAG_DECOMPOSITION_MIN_SUBQUERIES: int = int(
    os.environ.get("RAG_DECOMPOSITION_MIN_SUBQUERIES", "2")
)
"""Minimum sub-queries a valid decomposition must yield. Validated >= 2 and
<= RAG_DECOMPOSITION_MAX_SUBQUERIES by ``validate_document_routing_config``."""

RAG_DECOMPOSITION_MAX_SUBQUERIES: int = int(
    os.environ.get("RAG_DECOMPOSITION_MAX_SUBQUERIES", "5")
)
"""Maximum sub-queries a valid decomposition may yield."""

# ─── Document Cards (ingest) — DOCUMENT_ROUTING_DESIGN.md §11 ─────────────
# Ingest-side card emission. Mirrored onto IngestionConfig fields in
# src/ingest/common/types.py. OFF by default (no card collection written).

RAG_INGESTION_BUILD_DOCUMENT_CARDS: bool = os.environ.get(
    "RAG_INGESTION_BUILD_DOCUMENT_CARDS", "false"
).lower() in ("true", "1", "yes")
"""When True, ingest emits one routing-only document card per document into
RAG_DOCUMENT_CARD_COLLECTION. Default False = no card collection written."""

RAG_INGESTION_CARD_LLM_SUMMARY: bool = os.environ.get(
    "RAG_INGESTION_CARD_LLM_SUMMARY", "false"
).lower() in ("true", "1", "yes")
"""When True, card text includes an LLM-generated summary. Default False =
baseline card text is title + section headings only (no LLM call)."""

RAG_INGESTION_CARD_MAX_HEADINGS: int = int(
    os.environ.get("RAG_INGESTION_CARD_MAX_HEADINGS", "60")
)
"""Cap on the number of section headings included in a card. Guards against
xlsx-style documents with hundreds of header rows (design §11)."""


def validate_document_routing_config() -> None:
    """Validate document-routing / decomposition / card configuration.

    Checks for contradictory or unsafe settings and raises ``ValueError`` with
    a descriptive message naming the offending key(s). Invoked lazily by
    callers (the router / decomposition orchestrator) rather than at import,
    matching ``validate_visual_retrieval_config``'s fail-fast-at-use pattern —
    importing ``config.settings`` with default env never raises.

    Raises:
        ValueError: If ``RAG_DOCUMENT_ROUTING_TOP_N < 2`` (design §7: never
            top-1); if ``RAG_DECOMPOSITION_MIN_SUBQUERIES`` is < 2 or exceeds
            ``RAG_DECOMPOSITION_MAX_SUBQUERIES``; or if
            ``RAG_DOCUMENT_ROUTING_MAX_CANDIDATES`` is below
            ``RAG_DOCUMENT_ROUTING_PER_DOC_LEAVES``.
    """
    if RAG_DOCUMENT_ROUTING_TOP_N < 2:
        raise ValueError(
            f"RAG_DOCUMENT_ROUTING_TOP_N={RAG_DOCUMENT_ROUTING_TOP_N} must be "
            ">= 2 (design §7: route to top-N documents, never top-1)"
        )
    if RAG_DECOMPOSITION_MIN_SUBQUERIES < 2:
        raise ValueError(
            f"RAG_DECOMPOSITION_MIN_SUBQUERIES={RAG_DECOMPOSITION_MIN_SUBQUERIES} "
            "must be >= 2 (a decomposition needs at least two sub-queries)"
        )
    if RAG_DECOMPOSITION_MIN_SUBQUERIES > RAG_DECOMPOSITION_MAX_SUBQUERIES:
        raise ValueError(
            f"RAG_DECOMPOSITION_MIN_SUBQUERIES={RAG_DECOMPOSITION_MIN_SUBQUERIES} "
            "must be <= RAG_DECOMPOSITION_MAX_SUBQUERIES="
            f"{RAG_DECOMPOSITION_MAX_SUBQUERIES}"
        )
    if RAG_DOCUMENT_ROUTING_MAX_CANDIDATES < RAG_DOCUMENT_ROUTING_PER_DOC_LEAVES:
        raise ValueError(
            f"RAG_DOCUMENT_ROUTING_MAX_CANDIDATES={RAG_DOCUMENT_ROUTING_MAX_CANDIDATES} "
            "must be >= RAG_DOCUMENT_ROUTING_PER_DOC_LEAVES="
            f"{RAG_DOCUMENT_ROUTING_PER_DOC_LEAVES} (the union bound cannot be "
            "smaller than the per-document leaf fetch)"
        )


def validate_visual_retrieval_config() -> None:
    """Validate visual retrieval configuration at startup.

    Checks for contradictory or out-of-range settings and raises
    ValueError with a descriptive message identifying the conflicting keys.

    Called during RAGChain initialization when RAG_VISUAL_RETRIEVAL_ENABLED
    is True.

    Raises:
        ValueError: If visual retrieval is enabled but the visual target
            collection is empty, or if score threshold is out of [0.0, 1.0],
            or if URL expiry is out of [60, 86400].
    """
    if not RAG_INGESTION_VISUAL_TARGET_COLLECTION:
        raise ValueError(
            "RAG_VISUAL_RETRIEVAL_ENABLED=true but "
            "RAG_INGESTION_VISUAL_TARGET_COLLECTION is empty"
        )
    if RAG_VISUAL_RETRIEVAL_MIN_SCORE < 0.0 or RAG_VISUAL_RETRIEVAL_MIN_SCORE > 1.0:
        raise ValueError(
            f"RAG_VISUAL_RETRIEVAL_MIN_SCORE={RAG_VISUAL_RETRIEVAL_MIN_SCORE} "
            "out of range [0.0, 1.0]"
        )
    if RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS < 60 or RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS > 86400:
        raise ValueError(
            f"RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS={RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS} "
            "out of range [60, 86400]"
        )


def validate_all_config() -> None:
    """Validate all critical configuration at startup.

    Collects all validation errors and raises a single ValueError
    with all messages for easy diagnosis.
    """
    errors: list[str] = []

    # --- Timeouts must be > 0 ---
    for name, val in [
        ("RAG_WORKFLOW_DEFAULT_TIMEOUT_MS", RAG_WORKFLOW_DEFAULT_TIMEOUT_MS),
        ("RAG_RETRIEVAL_TIMEOUT_MS", RAG_RETRIEVAL_TIMEOUT_MS),
        ("QUERY_PROCESSING_TIMEOUT", QUERY_PROCESSING_TIMEOUT),
        ("RAG_INGESTION_LLM_TIMEOUT_SECONDS", RAG_INGESTION_LLM_TIMEOUT_SECONDS),
        ("RAG_INGESTION_VISION_TIMEOUT_SECONDS", RAG_INGESTION_VISION_TIMEOUT_SECONDS),
        ("RAG_NEMO_RAIL_TIMEOUT_SECONDS", RAG_NEMO_RAIL_TIMEOUT_SECONDS),
    ]:
        if val <= 0:
            errors.append(f"{name}={val} must be > 0")

    # --- Thresholds must be in [0.0, 1.0] ---
    for name, val in [
        ("QUERY_CONFIDENCE_THRESHOLD", QUERY_CONFIDENCE_THRESHOLD),
        ("SEMANTIC_SIMILARITY_THRESHOLD", SEMANTIC_SIMILARITY_THRESHOLD),
        ("RAG_NEMO_TOXICITY_THRESHOLD", RAG_NEMO_TOXICITY_THRESHOLD),
        ("RAG_NEMO_FAITHFULNESS_THRESHOLD", RAG_NEMO_FAITHFULNESS_THRESHOLD),
        ("RAG_NEMO_INTENT_CONFIDENCE_THRESHOLD", RAG_NEMO_INTENT_CONFIDENCE_THRESHOLD),
        ("RAG_NEMO_PII_SCORE_THRESHOLD", RAG_NEMO_PII_SCORE_THRESHOLD),
        ("RAG_CONFIDENCE_HIGH_THRESHOLD", RAG_CONFIDENCE_HIGH_THRESHOLD),
        ("RAG_CONFIDENCE_LOW_THRESHOLD", RAG_CONFIDENCE_LOW_THRESHOLD),
        ("RAG_CONFIDENCE_RETRIEVAL_WEIGHT", RAG_CONFIDENCE_RETRIEVAL_WEIGHT),
        ("RAG_CONFIDENCE_LLM_WEIGHT", RAG_CONFIDENCE_LLM_WEIGHT),
        ("RAG_CONFIDENCE_CITATION_WEIGHT", RAG_CONFIDENCE_CITATION_WEIGHT),
    ]:
        if not (0.0 <= val <= 1.0):
            errors.append(f"{name}={val} must be in [0.0, 1.0]")

    # --- Threshold ordering ---
    if RAG_CONFIDENCE_HIGH_THRESHOLD < RAG_CONFIDENCE_LOW_THRESHOLD:
        errors.append(
            f"RAG_CONFIDENCE_HIGH_THRESHOLD ({RAG_CONFIDENCE_HIGH_THRESHOLD}) "
            f"must be >= RAG_CONFIDENCE_LOW_THRESHOLD ({RAG_CONFIDENCE_LOW_THRESHOLD})"
        )
    if not (
        RAG_RETRIEVAL_QUALITY_STRONG_THRESHOLD
        >= RAG_RETRIEVAL_QUALITY_MODERATE_THRESHOLD
        >= RAG_RETRIEVAL_QUALITY_WEAK_THRESHOLD
    ):
        errors.append(
            "Retrieval quality thresholds must satisfy: "
            "STRONG >= MODERATE >= WEAK"
        )

    # --- Positive integers ---
    for name, val in [
        ("GENERATION_MAX_TOKENS", GENERATION_MAX_TOKENS),
        ("LLM_MAX_TOKENS", LLM_MAX_TOKENS),
        ("RAG_WORKER_CONCURRENCY", RAG_WORKER_CONCURRENCY),
        ("RATE_LIMIT_WINDOW_SECONDS", RATE_LIMIT_WINDOW_SECONDS),
        ("RERANKER_BATCH_SIZE", RERANKER_BATCH_SIZE),
        ("RAG_INGESTION_LLM_MAX_KEYWORDS", RAG_INGESTION_LLM_MAX_KEYWORDS),
    ]:
        if val <= 0:
            errors.append(f"{name}={val} must be > 0")

    # --- Port range ---
    if not (1 <= RAG_API_PORT <= 65535):
        errors.append(f"RAG_API_PORT={RAG_API_PORT} must be in [1, 65535]")

    # --- Feature dependencies ---
    if RAG_INGESTION_VISION_ENABLED:
        if not RAG_INGESTION_VISION_PROVIDER:
            errors.append(
                "RAG_INGESTION_VISION_ENABLED=true requires "
                "RAG_INGESTION_VISION_PROVIDER to be set"
            )
        if not RAG_INGESTION_VISION_MODEL:
            errors.append(
                "RAG_INGESTION_VISION_ENABLED=true requires "
                "RAG_INGESTION_VISION_MODEL to be set"
            )

    if errors:
        raise ValueError(
            "Configuration validation failed:\n  " + "\n  ".join(errors)
        )
