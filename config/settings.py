# @summary
# Centralizes configuration settings for a RAG (Retrieval-Augmented Generation) system.
# Exports: PROJECT_ROOT, DOCUMENTS_DIR, PROCESSED_DIR, EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH, VECTOR_DB_BACKEND, VECTOR_COLLECTION_DEFAULT, WEAVIATE_COLLECTION_NAME, DATABASE_BACKEND, MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET, MINIO_SECURE, RAG_WEAVIATE_MODE, RAG_WEAVIATE_HOST, RAG_WEAVIATE_HTTP_PORT, RAG_WEAVIATE_GRPC_PORT, HYBRID_SEARCH_ALPHA, SEARCH_LIMIT, RERANK_TOP_K, CHUNK_SIZE, CHUNK_OVERLAP, QUERY_CONFIDENCE_THRESHOLD, MAX_SANITIZATION_ITERATIONS, QUERY_PROCESSING_MODEL, QUERY_MAX_LENGTH, QUERY_PROCESSING_TEMPERATURE, QUERY_LOG_DIR, PROMPTS_DIR, DOMAIN_DESCRIPTION, KG_ENABLED, KG_PATH, SEMANTIC_CHUNKING_ENABLED, GLINER_ENABLED, GENERATION_ENABLED, RAG_CONFIDENCE_ROUTING_ENABLED, RAG_DOCUMENT_FORMATTING_ENABLED, RAG_NEMO_PII_GLINER_ENABLED, RAG_INGESTION_VLM_MODE, RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS, RAG_INGESTION_PERSIST_DOCLING_DOCUMENT, RAG_INGESTION_ENABLE_VISUAL_EMBEDDING, RAG_INGESTION_VISUAL_TARGET_COLLECTION, RAG_INGESTION_COLQWEN_MODEL, RAG_INGESTION_COLQWEN_BATCH_SIZE, RAG_INGESTION_PAGE_IMAGE_QUALITY, RAG_INGESTION_PAGE_IMAGE_MAX_DIMENSION, RAG_VISUAL_RETRIEVAL_ENABLED, RAG_VISUAL_RETRIEVAL_LIMIT, RAG_VISUAL_RETRIEVAL_MIN_SCORE, RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS, RAG_STAGE_BUDGET_VISUAL_RETRIEVAL_MS, validate_visual_retrieval_config, VALID_MODEL_PRECISIONS, EMBEDDING_PRECISION_QUERY, EMBEDDING_PRECISION_INGEST, RERANKER_PRECISION, VISUAL_RETRIEVAL_PRECISION, GENERATION_PRECISION
# Exports (retrieval): RAG_KG_RETRIEVAL_EDGE_TYPES, RAG_KG_RETRIEVAL_PATH_PATTERNS,
#   RAG_KG_GRAPH_CONTEXT_TOKEN_BUDGET, RAG_KG_ENABLE_GRAPH_CONTEXT_INJECTION
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
SEARCH_LIMIT = 10
RERANK_TOP_K = 5

# --- Document Processing ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

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
QUERY_MAX_LENGTH = int(os.environ.get("RAG_QUERY_MAX_LENGTH", "500"))
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

# --- Knowledge Graph ---
# Set to False to disable KG query expansion (pure hybrid search only)
KG_ENABLED = os.environ.get("RAG_KG_ENABLED", "true").lower() in ("true", "1", "yes")
KG_PATH = PROJECT_ROOT / ".knowledge_graph.json"
KG_OBSIDIAN_EXPORT_DIR = PROJECT_ROOT / "obsidian_graph"

# KG Retrieval enhancements (REQ-KG-1200..1206)
# Edge type whitelist for typed traversal. Comma-separated in env; empty = untyped.
RAG_KG_RETRIEVAL_EDGE_TYPES: list[str] = [
    e.strip()
    for e in os.environ.get("RAG_KG_RETRIEVAL_EDGE_TYPES", "").split(",")
    if e.strip()
]
# Path patterns: JSON-encoded list of edge-type sequences, e.g. '[["instantiates","specified_by"]]'.
# Empty list = no pattern matching.
import json as _json
try:
    RAG_KG_RETRIEVAL_PATH_PATTERNS: list[list[str]] = _json.loads(
        os.environ.get("RAG_KG_RETRIEVAL_PATH_PATTERNS", "[]")
    )
except (ValueError, TypeError):
    logging.getLogger(__name__).warning(
        "RAG_KG_RETRIEVAL_PATH_PATTERNS is not valid JSON; defaulting to []"
    )
    RAG_KG_RETRIEVAL_PATH_PATTERNS: list[list[str]] = []
# Max tokens for the graph context block injected into the generation prompt.
RAG_KG_GRAPH_CONTEXT_TOKEN_BUDGET: int = int(
    os.environ.get("RAG_KG_GRAPH_CONTEXT_TOKEN_BUDGET", "500")
)
# Master toggle for graph context injection. False = skip all retrieval enhancements.
RAG_KG_ENABLE_GRAPH_CONTEXT_INJECTION: bool = os.environ.get(
    "RAG_KG_ENABLE_GRAPH_CONTEXT_INJECTION", "false"
).lower() in ("true", "1", "yes")
# Community context token budget (REQ-KG-1320). 0 = community context disabled.
RAG_KG_COMMUNITY_CONTEXT_TOKEN_BUDGET: int = int(
    os.environ.get("RAG_KG_COMMUNITY_CONTEXT_TOKEN_BUDGET", "200")
)
# Graph context section marker style (REQ-KG-1322). "markdown" | "xml" | "plain".
RAG_KG_GRAPH_CONTEXT_MARKER_STYLE: str = os.environ.get(
    "RAG_KG_GRAPH_CONTEXT_MARKER_STYLE", "markdown"
)
# Max entities explored per hop in path pattern evaluation (REQ-KG-1324).
RAG_KG_MAX_HOP_FANOUT: int = int(
    os.environ.get("RAG_KG_MAX_HOP_FANOUT", "50")
)
# LLM extraction rate-limit retry policy
RAG_KG_LLM_RATE_LIMIT_RETRIES = int(os.environ.get("RAG_KG_LLM_RATE_LIMIT_RETRIES", "3"))
RAG_KG_LLM_RATE_LIMIT_BACKOFF_S = float(os.environ.get("RAG_KG_LLM_RATE_LIMIT_BACKOFF_S", "1.0"))
# GLiNER entity prediction confidence threshold
RAG_KG_GLINER_THRESHOLD = float(os.environ.get("RAG_KG_GLINER_THRESHOLD", "0.5"))

# --- Semantic Chunking ---
SEMANTIC_CHUNKING_ENABLED = os.environ.get(
    "RAG_SEMANTIC_CHUNKING", "true"
).lower() in ("true", "1", "yes")
SEMANTIC_SIMILARITY_THRESHOLD = float(
    os.environ.get("RAG_SEMANTIC_THRESHOLD", "0.75")
)

# --- GLiNER Entity Extraction ---
GLINER_ENABLED = os.environ.get(
    "RAG_GLINER_ENABLED", "true"
).lower() in ("true", "1", "yes")
GLINER_MODEL_PATH = os.environ.get(
    "RAG_GLINER_MODEL",
    os.path.expanduser("~/models/gliner/gliner_medium-v2.1"),
)
GLINER_ENTITY_LABELS = [
    "technology", "algorithm", "framework", "concept",
    "programming language", "data structure",
]

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
RAG_INGESTION_ENABLE_DOCUMENT_REFACTORING = os.environ.get(
    "RAG_INGESTION_ENABLE_DOCUMENT_REFACTORING", "false"
).lower() in ("true", "1", "yes")
RAG_INGESTION_ENABLE_CROSS_REFERENCE_EXTRACTION = os.environ.get(
    "RAG_INGESTION_ENABLE_CROSS_REFERENCE_EXTRACTION", "true"
).lower() in ("true", "1", "yes")
RAG_INGESTION_ENABLE_KNOWLEDGE_GRAPH_EXTRACTION = os.environ.get(
    "RAG_INGESTION_ENABLE_KNOWLEDGE_GRAPH_EXTRACTION", "true"
).lower() in ("true", "1", "yes")
RAG_INGESTION_ENABLE_QUALITY_VALIDATION = os.environ.get(
    "RAG_INGESTION_ENABLE_QUALITY_VALIDATION", "true"
).lower() in ("true", "1", "yes")
RAG_INGESTION_ENABLE_KNOWLEDGE_GRAPH_STORAGE = os.environ.get(
    "RAG_INGESTION_ENABLE_KNOWLEDGE_GRAPH_STORAGE", "true"
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
    os.environ.get("RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS", "512")
)
"""Maximum token count per chunk for HybridChunker (bge-m3 limit is 512)."""

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

RAG_INGESTION_CHUNKER: str = os.environ.get(
    "RAG_INGESTION_CHUNKER", "native"
)
"""Chunker override: "native" | "markdown".
"native" uses each parser's own chunker (DoclingParser HybridChunker,
CodeParser tree-sitter, PlainTextParser markdown). "markdown" forces
the shared chunk_with_markdown() fallback for all parsers. FR-3320."""

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
# Inside the compose network these resolve to rag-embed / rag-rerank.
TEI_EMBED_URL: str = os.environ.get("RAG_TEI_EMBED_URL", "http://rag-embed:80")
TEI_RERANK_URL: str = os.environ.get("RAG_TEI_RERANK_URL", "http://rag-rerank:80")

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

# NOTE: RAG_INGESTION_COLQWEN_MODEL is reused for retrieval-time model
# selection (FR-109). No separate retrieval model key exists.


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
