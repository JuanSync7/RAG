<!-- @summary
Pipeline stage nodes for RAG query processing and result reranking. Handles query sanitization,
LLM-driven reformulation and confidence routing via LangGraph, and cross-encoder reranking of
vector search results using local or vLLM-backed backends.
@end-summary -->

# retrieval/query/nodes

This package contains the two core stage nodes that prepare queries for search and score
retrieved results before they reach the generation stage.

## Contents

| Path | Purpose |
| --- | --- |
| `query_processor.py` | `process_query` — LangGraph-based pipeline that sanitizes input, reformulates the query for searchability (creation agent), evaluates reformulation confidence (verification agent), and routes to SEARCH, ASK_USER, or loops for retry; falls back to a word-count heuristic when the LLM is unavailable |
| `reranker.py` | `LocalBGEReranker` and `TEIReranker` — cross-encoder reranker implementations backed by a local BAAI/bge-reranker-v2-m3 model (via transformers) or a TEI container (via direct HTTP); `get_reranker_provider` selects the active backend from config; `RankedResult` carries the reranked chunk with its sigmoid-normalized score |
| `__init__.py` | Package facade re-exporting `process_query`, `warm_up_ollama`, `LocalBGEReranker`, `TEIReranker`, `RankedResult`, `get_reranker_provider`, `QueryAction`, `QueryResult`, and `QueryState` |
