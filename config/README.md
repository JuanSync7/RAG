<!-- @summary
This directory contains configuration settings for a RAG system, including paths to directories and models used in the system's operation.
@end-summary -->

# config

## Overview
This directory houses all the configuration settings necessary for a Retrieval-Augmented Generation (RAG) system. It includes various paths to directories where documents are stored, processed, or indexed, as well as paths to embedding and reranking models.

## Files
| File | Purpose | Key Exports |
|------|---------|--------------|
| settings.py | Centralizes configuration settings for the RAG system | PROJECT_ROOT, DOCUMENTS_DIR, PROCESSED_DIR, EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH, WEAVIATE_COLLECTION_NAME, HYBRID_SEARCH_ALPHA, SEARCH_LIMIT, RERANK_TOP_K, CHUNK_SIZE, CHUNK_OVERLAP, QUERY_CONFIDENCE_THRESHOLD, MAX_SANITIZATION_ITERATIONS, QUERY_PROCESSING_MODEL, QUERY_MAX_LENGTH, QUERY_PROCESSING_TEMPERATURE, QUERY_LOG_DIR, PROMPTS_DIR, DOMAIN_DESCRIPTION, KG_ENABLED, KG_PATH, SEMANTIC_CHUNKING_ENABLED, GLINER_ENABLED, GENERATION_ENABLED |

## Internal Dependencies
- settings.py depends on the `os`, `pathlib`, and `dotenv` libraries to manage configuration files.

## Subdirectories
None