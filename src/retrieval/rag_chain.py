# @summary
# End-to-end RAG pipeline for query processing, KG expansion, hybrid search, and reranking. Main classes: RAGChain, RAGResponse. Key imports: LocalBGEEmbeddings, LocalBGEReranker, KnowledgeGraphBuilder, OllamaGenerator, Filter, get_weaviate_client, ensure_collection, hybrid_search.
# @end-summary
"""Main RAG chain that orchestrates the full retrieval pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional

from src.core.embeddings import LocalBGEEmbeddings
from src.retrieval.reranker import LocalBGEReranker, RankedResult
from src.retrieval.query_processor import process_query, QueryAction, QueryResult
from src.core.knowledge_graph import KnowledgeGraphBuilder, GraphQueryExpander
from src.core.vector_store import get_weaviate_client, ensure_collection, hybrid_search

from weaviate.classes.query import Filter
from src.retrieval.generator import OllamaGenerator
from config.settings import (
    HYBRID_SEARCH_ALPHA, SEARCH_LIMIT, RERANK_TOP_K,
    KG_PATH, KG_ENABLED, GENERATION_ENABLED,
)


@dataclass
class RAGResponse:
    """Complete response from the RAG pipeline."""
    query: str
    processed_query: str
    query_confidence: float
    action: str  # "search" or "ask_user"
    results: List[RankedResult] = field(default_factory=list)
    clarification_message: Optional[str] = None
    kg_expanded_terms: Optional[List[str]] = None
    generated_answer: Optional[str] = None


class RAGChain:
    """End-to-end RAG pipeline: query processing -> KG expansion -> hybrid search -> reranking."""

    def __init__(self):
        print("Loading embedding model...")
        self.embeddings = LocalBGEEmbeddings()
        print("Loading reranker model...")
        self.reranker = LocalBGEReranker()

        # Load LLM generator if enabled (graceful fallback)
        self._generator: Optional[OllamaGenerator] = None
        if GENERATION_ENABLED:
            gen = OllamaGenerator()
            if gen.is_available():
                self._generator = gen
                print(f"Ollama generator ready (model: {gen.model}).")
            else:
                print("Warning: Ollama not available. Generation disabled.")
        else:
            print("Generation disabled (set RAG_GENERATION_ENABLED=true to enable).")

        # Load knowledge graph if enabled and available (graceful fallback)
        self._kg_expander: Optional[GraphQueryExpander] = None
        if KG_ENABLED and KG_PATH.exists():
            try:
                builder = KnowledgeGraphBuilder.load(KG_PATH)
                self._kg_expander = GraphQueryExpander(builder.graph)
                stats = builder.stats()
                print(f"Knowledge graph loaded: {stats['nodes']} nodes, {stats['edges']} edges")
            except Exception as e:
                print(f"Warning: Could not load knowledge graph: {e}")
        elif not KG_ENABLED:
            print("Knowledge graph disabled (set RAG_KG_ENABLED=true to enable).")
        else:
            print("No knowledge graph found (run ingest.py first to build it).")

        print("RAG chain ready.")

    def run(
        self,
        query: str,
        alpha: float = HYBRID_SEARCH_ALPHA,
        search_limit: int = SEARCH_LIMIT,
        rerank_top_k: int = RERANK_TOP_K,
        source_filter: Optional[str] = None,
        heading_filter: Optional[str] = None,
    ) -> RAGResponse:
        """Execute the full RAG pipeline.

        Steps:
            1. Process and sanitize the query
            2. If confident: KG expand -> embed query -> hybrid search -> rerank -> return results
            3. If not confident: return clarification request

        Args:
            query: Raw user query.
            alpha: Hybrid search balance (0=BM25, 1=vector).
            search_limit: Number of results from hybrid search.
            rerank_top_k: Number of top results after reranking.
            source_filter: Optional source filename to filter results by.
            heading_filter: Optional section heading to filter results by.

        Returns:
            RAGResponse with results or clarification message.
        """
        # Step 1: Query processing
        query_result: QueryResult = process_query(query)

        if query_result.action == QueryAction.ASK_USER:
            return RAGResponse(
                query=query,
                processed_query=query_result.processed_query,
                query_confidence=query_result.confidence,
                action="ask_user",
                clarification_message=query_result.clarification_message,
            )

        processed_query = query_result.processed_query

        # Step 2: KG expansion — augment BM25 query with related entity terms
        kg_expanded_terms = []
        if self._kg_expander:
            kg_expanded_terms = self._kg_expander.expand(processed_query, depth=1)

        # BM25 query gets expanded terms for broader keyword matching
        if kg_expanded_terms:
            bm25_query = processed_query + " " + " ".join(kg_expanded_terms[:3])
        else:
            bm25_query = processed_query

        # Step 3: Embed the ORIGINAL query (not augmented) for vector search
        # This keeps the semantic intent pure for the dense vector component
        query_embedding = self.embeddings.embed_query(processed_query)

        # Step 4: Build metadata filters and run hybrid search
        wv_filter = None
        if source_filter:
            wv_filter = Filter.by_property("source").equal(source_filter)
        if heading_filter:
            hf = Filter.by_property("heading").equal(heading_filter)
            wv_filter = wv_filter & hf if wv_filter else hf

        with get_weaviate_client() as client:
            ensure_collection(client)
            search_results = hybrid_search(
                client=client,
                query=bm25_query,
                query_embedding=query_embedding,
                alpha=alpha,
                limit=search_limit,
                filters=wv_filter,
            )

        if not search_results:
            return RAGResponse(
                query=query,
                processed_query=processed_query,
                query_confidence=query_result.confidence,
                action="search",
                results=[],
                kg_expanded_terms=kg_expanded_terms or None,
            )

        # Step 5: Rerank against original query for relevance accuracy
        reranked = self.reranker.rerank(
            query=processed_query,
            documents=search_results,
            top_k=rerank_top_k,
        )

        # Step 6: Generate answer from reranked chunks
        generated_answer = None
        if self._generator and reranked:
            context_chunks = [r.text for r in reranked]
            scores = [r.score for r in reranked]
            generated_answer = self._generator.generate(
                query=processed_query,
                context_chunks=context_chunks,
                scores=scores,
            )

        return RAGResponse(
            query=query,
            processed_query=processed_query,
            query_confidence=query_result.confidence,
            action="search",
            results=reranked,
            kg_expanded_terms=kg_expanded_terms or None,
            generated_answer=generated_answer,
        )
