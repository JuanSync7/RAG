<!-- @summary
This directory contains modules for retrieval-based query processing using LangGraph for confidence-based routing and LLM generation via Ollama.
@end-summary -->

# retrieval

## Overview
This directory includes modules for retrieval-based query processing utilizing LangGraph for confidence-based routing, along with an LLM generator module.

## Files
| File | Purpose | Key Exports |
|------|---------|--------------|
| generator.py | Provides a generator for synthesizing RAG answers using Ollama. | OllamaGenerator |
| query_processor.py | Handles retrieval-based query processing using LangGraph for confidence-based routing and user-friendly exports. | process_query, QueryResult, QueryAction |
| rag_chain.py | Orchestrates the end-to-end RAG pipeline including KG expansion, hybrid search, and reranking. | RAGChain, RAGResponse |
| reranker.py | Wraps a local BAAI bge-reranker-v2-m3 for reranking search results. | LocalBGEReranker, RankedResult |

## Internal Dependencies
- generator.py depends on config.settings.
- query_processor.py relies on logging, json, re, state_graph, and _COMPILED_GRAPH.
- rag_chain.py imports LocalBGEEmbeddings, LocalBGEReranker, KnowledgeGraphBuilder, OllamaGenerator, Filter, get_weaviate_client, ensure_collection, hybrid_search.

## Subdirectories
None