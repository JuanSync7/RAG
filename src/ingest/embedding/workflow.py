# @summary
# LangGraph StateGraph for the Embedding Pipeline.
# Exports: build_embedding_graph
# Deps: langgraph.graph, src.ingest.embedding.nodes.*, src.ingest.embedding.state
# Node order: document_storage → chunking → vlm_enrichment → chunk_enrichment →
#   tree_node_synthesis → document_card_emission → metadata_generation →
#   [cross_reference_extraction →] quality_validation → lossless_verification →
#   [cross_document_dedup →] embedding_storage → visual_embedding → commit → END
# KG ingest is owned by KGWeave: RagWeave dispatches Phase 2b on KG_TASK_QUEUE
# from the per-document Temporal workflow (see src/ingest/temporal/workflows.py).
# In-process KG nodes were removed when KGWeave became the canonical KG owner.
# @end-summary

"""Embedding Pipeline LangGraph workflow."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.ingest.embedding.nodes import chunk_enrichment_node
from src.ingest.embedding.nodes import chunking_node
from src.ingest.embedding.nodes import commit_node
from src.ingest.embedding.nodes import vlm_enrichment_node
from src.ingest.embedding.nodes import document_storage_node
from src.ingest.embedding.nodes import cross_reference_extraction_node
from src.ingest.embedding.nodes import embedding_storage_node
from src.ingest.embedding.nodes.document_card import document_card_emission_node
from src.ingest.embedding.nodes import metadata_generation_node
from src.ingest.embedding.nodes import quality_validation_node
from src.ingest.embedding.nodes import lossless_verification_node
from src.ingest.embedding.nodes import tree_node_synthesis_node
from src.ingest.embedding.nodes import visual_embedding_node
from src.ingest.embedding.nodes.cross_document_dedup import cross_document_dedup_node
from src.ingest.embedding.state import EmbeddingPipelineState


def build_embedding_graph(config=None):
    """Compile the Embedding Pipeline StateGraph.

    Node order:
        document_storage → chunking → vlm_enrichment → chunk_enrichment
        → tree_node_synthesis → document_card_emission → metadata_generation
        → [cross_reference_extraction →] quality_validation → lossless_verification
        → [cross_document_dedup →] embedding_storage → visual_embedding
        → commit → END

    Routing:
    - document_card_emission: always in graph; short-circuits internally
      (strict no-op) when ``config.build_document_cards`` is False (default).
    - vlm_enrichment: always in graph; short-circuits internally when
      ``config.vlm_mode != "external"``.
    - cross_reference_extraction: conditional on ``config.enable_cross_reference_extraction``.
    - cross_document_dedup: conditional on ``config.enable_cross_document_dedup``.
    - visual_embedding: always in graph; short-circuits internally when no
      visual chunks are present or visual embedding is not configured.
    - commit: terminal node. Flushes MinIO + Weaviate atomically. On failure,
      rolls back Weaviate (by staging_batch_id) and MinIO (by document_id).
    - KG ingest is NOT a node here. Phase 2b is dispatched by name on
      ``KG_TASK_QUEUE`` from ``IngestDocumentWorkflow._run_kg_phase2b``.

    Returns:
        Compiled LangGraph graph accepting ``EmbeddingPipelineState``.
    """
    graph = StateGraph(EmbeddingPipelineState)
    graph.add_node("document_storage", document_storage_node)
    graph.add_node("chunking", chunking_node)
    graph.add_node("vlm_enrichment", vlm_enrichment_node)
    graph.add_node("chunk_enrichment", chunk_enrichment_node)
    graph.add_node("tree_node_synthesis", tree_node_synthesis_node)
    graph.add_node("document_card_emission", document_card_emission_node)
    graph.add_node("metadata_generation", metadata_generation_node)
    graph.add_node("cross_reference_extraction", cross_reference_extraction_node)
    graph.add_node("quality_validation", quality_validation_node)
    graph.add_node("lossless_verification", lossless_verification_node)
    graph.add_node("cross_document_dedup", cross_document_dedup_node)
    graph.add_node("embedding_storage", embedding_storage_node)
    graph.add_node("visual_embedding", visual_embedding_node)
    graph.add_node("commit", commit_node)

    graph.set_entry_point("document_storage")
    graph.add_edge("document_storage", "chunking")
    graph.add_edge("chunking", "vlm_enrichment")
    graph.add_edge("vlm_enrichment", "chunk_enrichment")
    # tree_node_synthesis is unconditional in the graph; the node itself
    # short-circuits when config.enable_tree_retrieval_ingest is False, so
    # the disabled-path produces byte-identical chunks to pre-1.2.0 behaviour.
    graph.add_edge("chunk_enrichment", "tree_node_synthesis")
    # document_card_emission is unconditional in the graph; the node itself
    # short-circuits (strict no-op) when config.build_document_cards is False,
    # so the disabled-path is byte-identical to pre-feature behaviour.
    graph.add_edge("tree_node_synthesis", "document_card_emission")
    graph.add_edge("document_card_emission", "metadata_generation")
    graph.add_conditional_edges(
        "metadata_generation",
        lambda state: (
            "cross_reference_extraction"
            if state["runtime"].config.enable_cross_reference_extraction
            else "quality_validation"
        ),
        {
            "cross_reference_extraction": "cross_reference_extraction",
            "quality_validation": "quality_validation",
        },
    )
    graph.add_edge("cross_reference_extraction", "quality_validation")
    # Lossless verification runs on the finalized chunk set (post quality gate)
    # before storage; it is itself config-gated (no-op when verify_lossless off).
    graph.add_edge("quality_validation", "lossless_verification")
    graph.add_conditional_edges(
        "lossless_verification",
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
    graph.add_edge("visual_embedding", "commit")
    graph.add_edge("commit", END)
    return graph.compile()
