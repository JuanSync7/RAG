# @summary
# LangGraph StateGraph for the 12-node Embedding Pipeline (Phase 3.3 + Issue #42).
# Exports: build_embedding_graph
# Deps: langgraph.graph, src.ingest.embedding.nodes.*, src.ingest.embedding.state
# Node order: document_storage → chunking → vlm_enrichment → chunk_enrichment →
#   metadata_generation → [cross_reference_extraction →] knowledge_graph_extraction
#   → quality_validation → [cross_document_dedup →] embedding_storage
#   → visual_embedding → [knowledge_graph_storage →] commit → END
# vlm_enrichment_node is always present; it short-circuits internally for
# vlm_mode != "external".
# visual_embedding_node is always present (between embedding_storage and
# knowledge_graph_storage); it short-circuits internally when no visual chunks
# are present.
# cross_document_dedup_node: conditional — only runs when
#   config.enable_cross_document_dedup=True; skips to embedding_storage otherwise.
# commit_node: new terminal node; flushes MinIO + Weaviate + KG atomically (Issue #42).
# @end-summary

"""Phase 2 / Phase 3.3 LangGraph workflow for embedding and storage."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.ingest.embedding.nodes import chunk_enrichment_node
from src.ingest.embedding.nodes import chunking_node
from src.ingest.embedding.nodes import commit_node
from src.ingest.embedding.nodes import vlm_enrichment_node
from src.ingest.embedding.nodes import document_storage_node
from src.ingest.embedding.nodes import cross_reference_extraction_node
from src.ingest.embedding.nodes import embedding_storage_node
from src.ingest.embedding.nodes import knowledge_graph_extraction_node
from src.ingest.embedding.nodes import knowledge_graph_storage_node
from src.ingest.embedding.nodes import metadata_generation_node
from src.ingest.embedding.nodes import quality_validation_node
from src.ingest.embedding.nodes import visual_embedding_node
from src.ingest.embedding.nodes.cross_document_dedup import cross_document_dedup_node
from src.ingest.embedding.state import EmbeddingPipelineState


def build_embedding_graph(config=None):
    """Compile the Phase 3.3 + Issue #42 Embedding Pipeline StateGraph.

    Node order:
        document_storage → chunking → vlm_enrichment → chunk_enrichment
        → metadata_generation → [cross_reference_extraction →]
        knowledge_graph_extraction → quality_validation
        → [cross_document_dedup →] embedding_storage
        → visual_embedding → [knowledge_graph_storage →] commit → END

    Routing:
    - vlm_enrichment: always in graph; short-circuits internally when
      ``config.vlm_mode != "external"`` (logs ``vlm_enrichment:skipped``).
    - cross_reference_extraction: only if config.enable_cross_reference_extraction.
    - knowledge_graph_extraction: always runs (handles disabled state internally).
    - cross_document_dedup: conditional — only entered when
      ``config.enable_cross_document_dedup=True``; otherwise routes directly to
      ``embedding_storage``. When disabled, the node is still registered but
      never invoked, preserving backward-compatible behaviour.
    - visual_embedding: always in graph; short-circuits internally when no
      visual chunks are present or visual embedding is not configured.
    - knowledge_graph_storage: only if config.enable_knowledge_graph_storage.
    - commit: terminal node; always present. Flushes MinIO + Weaviate + KG
      atomically. On failure, rolls back Weaviate (by staging_batch_id) and
      MinIO (by document_id). See Issue #42.

    Returns:
        Compiled LangGraph graph accepting ``EmbeddingPipelineState``.
    """
    graph = StateGraph(EmbeddingPipelineState)
    graph.add_node("document_storage", document_storage_node)
    graph.add_node("chunking", chunking_node)
    graph.add_node("vlm_enrichment", vlm_enrichment_node)
    graph.add_node("chunk_enrichment", chunk_enrichment_node)
    graph.add_node("metadata_generation", metadata_generation_node)
    graph.add_node("cross_reference_extraction", cross_reference_extraction_node)
    graph.add_node("knowledge_graph_extraction", knowledge_graph_extraction_node)
    graph.add_node("quality_validation", quality_validation_node)
    graph.add_node("cross_document_dedup", cross_document_dedup_node)
    graph.add_node("embedding_storage", embedding_storage_node)
    graph.add_node("visual_embedding", visual_embedding_node)
    graph.add_node("knowledge_graph_storage", knowledge_graph_storage_node)
    graph.add_node("commit", commit_node)

    graph.set_entry_point("document_storage")
    graph.add_edge("document_storage", "chunking")
    graph.add_edge("chunking", "vlm_enrichment")
    graph.add_edge("vlm_enrichment", "chunk_enrichment")
    graph.add_edge("chunk_enrichment", "metadata_generation")
    graph.add_conditional_edges(
        "metadata_generation",
        lambda state: (
            "cross_reference_extraction"
            if state["runtime"].config.enable_cross_reference_extraction
            else "knowledge_graph_extraction"
        ),
        {
            "cross_reference_extraction": "cross_reference_extraction",
            "knowledge_graph_extraction": "knowledge_graph_extraction",
        },
    )
    graph.add_edge("cross_reference_extraction", "knowledge_graph_extraction")
    # knowledge_graph_extraction_node checks config.enable_knowledge_graph_extraction
    # internally and returns early if disabled — always run the node, no conditional edge.
    graph.add_edge("knowledge_graph_extraction", "quality_validation")
    # Phase 3.3: conditional dedup edge (FR-3402).
    # When disabled, routes directly to embedding_storage, preserving pre-3.3 behaviour.
    graph.add_conditional_edges(
        "quality_validation",
        lambda state: (
            "cross_document_dedup"
            if getattr(state["runtime"].config, "enable_cross_document_dedup", True)
            else "embedding_storage"
        ),
        {
            "cross_document_dedup": "cross_document_dedup",
            "embedding_storage": "embedding_storage",
        },
    )
    graph.add_edge("cross_document_dedup", "embedding_storage")
    graph.add_edge("embedding_storage", "visual_embedding")
    # Issue #42: knowledge_graph_storage and the direct path both converge on commit.
    # visual_embedding routes to knowledge_graph_storage (when enabled) or directly
    # to commit; knowledge_graph_storage always routes to commit.
    graph.add_conditional_edges(
        "visual_embedding",
        lambda state: (
            "knowledge_graph_storage"
            if state["runtime"].config.enable_knowledge_graph_storage
            else "commit"
        ),
        {"knowledge_graph_storage": "knowledge_graph_storage", "commit": "commit"},
    )
    graph.add_edge("knowledge_graph_storage", "commit")
    graph.add_edge("commit", END)
    return graph.compile()
