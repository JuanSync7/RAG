<!-- @summary
This directory contains core modules for embedding management, knowledge graph construction, and vector store integration in a project.
@end-summary -->

# core

## Overview
This directory includes core modules responsible for managing embeddings, constructing knowledge graphs, and integrating vector stores, essential components for text-based data processing and retrieval.

## Files
| File | Purpose | Key Exports |
|------|---------|--------------|
| embeddings.py | Provides a wrapper for embedding local documents using the BAAI/bge-m3 model compatible with LangChain. | LocalBGEEmbeddings |
| knowledge_graph.py | Constructs a NetworkX DiGraph from document chunks and exports it as .md files, supporting entity extraction and querying. | KnowledgeGraphBuilder, GraphQueryExpander, export_obsidian |
| vector_store.py | Manages a Weaviate vector store with hybrid search capabilities (BM25 + dense vectors). Provides functions for client setup, document addition, and retrieval. | get_weaviate_client, ensure_collection, add_documents, hybrid_search |

## Internal Dependencies
The files within this directory do not rely on each other directly.

## Subdirectories
None