# @summary
# LangGraph StateGraph for the 5-node Document Processing Pipeline (Phase 1).
# Exports: build_document_processing_graph
# Deps: langgraph.graph, src.ingest.doc_processing.nodes.*, src.ingest.doc_processing.state
# @end-summary

"""Phase 1 LangGraph workflow for document processing."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.ingest.doc_processing.nodes import document_ingestion_node
from src.ingest.doc_processing.nodes import structure_detection_node
from src.ingest.doc_processing.nodes import multimodal_processing_node
from src.ingest.doc_processing.nodes import text_cleaning_node
from src.ingest.doc_processing.nodes import document_refactoring_node
from src.ingest.doc_processing.state import DocumentProcessingState


def build_document_processing_graph():
    """Compile the Phase 1 Document Processing StateGraph.

    Routing:
    - After document_ingestion: short-circuit to END on errors.
    - After structure_detection: multimodal_processing if enabled + has_figures, else text_cleaning.
    - After text_cleaning: document_refactoring if enabled, else END.

    Returns:
        Compiled LangGraph graph accepting ``DocumentProcessingState``.
    """
    graph = StateGraph(DocumentProcessingState)
    graph.add_node("document_ingestion", document_ingestion_node)
    graph.add_node("structure_detection", structure_detection_node)
    graph.add_node("multimodal_processing", multimodal_processing_node)
    graph.add_node("text_cleaning", text_cleaning_node)
    graph.add_node("document_refactoring", document_refactoring_node)

    graph.set_entry_point("document_ingestion")
    graph.add_conditional_edges(
        "document_ingestion",
        lambda state: "end" if state.get("errors") else "structure_detection",
        {"structure_detection": "structure_detection", "end": END},
    )
    graph.add_conditional_edges(
        "structure_detection",
        lambda state: "end" if state.get("errors") else (
            "multimodal_processing"
            if (
                state["runtime"].config.enable_multimodal_processing
                and state.get("structure", {}).get("has_figures")
            )
            else "text_cleaning"
        ),
        {"multimodal_processing": "multimodal_processing", "text_cleaning": "text_cleaning", "end": END},
    )
    graph.add_edge("multimodal_processing", "text_cleaning")
    graph.add_conditional_edges(
        "text_cleaning",
        lambda state: (
            "document_refactoring"
            if state["runtime"].config.enable_document_refactoring
            else "end"
        ),
        {"document_refactoring": "document_refactoring", "end": END},
    )
    graph.add_edge("document_refactoring", END)
    return graph.compile()
