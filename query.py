#!/usr/bin/env python3
# @summary
# Interactive query interface for RAG system. Main exports: parse_filters, display_results, main. Deps: re, sys, pathlib, src.retrieval.rag_chain
# @end-summary
"""Interactive query interface for the RAG system."""

import re
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.retrieval.rag_chain import RAGChain


# Filter prefix patterns: "source:filename.txt" or "section:Heading"
_FILTER_PAT = re.compile(
    r"\b(source|section):(\S+)\s*", re.IGNORECASE
)


def parse_filters(raw_query: str) -> tuple:
    """Extract filter prefixes from query, return (clean_query, filters_dict).

    Supported filters:
        source:<filename>   — filter by source document
        section:<heading>   — filter by section heading

    Example:
        "source:sample_doc_3.txt what is RAG?"
        → ("what is RAG?", {"source_filter": "sample_doc_3.txt"})
    """
    filters = {}
    def _replace(m):
        key = m.group(1).lower()
        value = m.group(2)
        if key == "source":
            filters["source_filter"] = value
        elif key == "section":
            filters["heading_filter"] = value
        return ""

    clean = _FILTER_PAT.sub(_replace, raw_query).strip()
    return clean, filters


def display_results(response) -> None:
    """Pretty-print RAG results."""
    print(f"\n{'='*60}")
    print(f"Query:            {response.query}")
    print(f"Processed query:  {response.processed_query}")
    print(f"Confidence:       {response.query_confidence:.0%}")
    print(f"Action:           {response.action}")
    if response.kg_expanded_terms:
        print(f"KG expansion:     {', '.join(response.kg_expanded_terms[:5])}")
    print(f"{'='*60}")

    if response.action == "ask_user":
        print(f"\n{response.clarification_message}")
        return

    # Show generated answer prominently
    if response.generated_answer:
        print(f"\n{'- '*30}")
        print("ANSWER:")
        print(f"{'- '*30}")
        print(response.generated_answer)
        print(f"{'- '*30}")

    if not response.results:
        print("\nNo results found.")
        return

    # Raw chunks below for development/debugging
    print(f"\nTop {len(response.results)} retrieved chunks:\n")
    for i, result in enumerate(response.results, 1):
        print(f"--- Chunk {i} (reranker: {result.score:.4f}) ---")
        print(f"Source: {result.metadata.get('source', 'unknown')}")
        section = result.metadata.get("section_path", "")
        if section:
            print(f"Section: {section}")
        print(f"Text:   {result.text[:200]}{'...' if len(result.text) > 200 else ''}")
        print()


def main() -> None:
    """Run the interactive query loop."""
    print("Initializing RAG system...")
    rag = RAGChain()

    print("\nRAG system ready. Type 'quit' or 'exit' to stop.")
    print("Filters: source:<file> section:<heading>  (e.g. source:sample_doc_3.txt what is RAG?)\n")

    while True:
        try:
            raw_input = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not raw_input:
            continue
        if raw_input.lower() in ("quit", "exit"):
            print("Goodbye.")
            break

        query, filters = parse_filters(raw_input)
        if not query:
            print("Please provide a query (not just filters).")
            continue

        if filters:
            print(f"  Filters: {filters}")

        response = rag.run(query, **filters)
        display_results(response)

        # Handle clarification loop
        if response.action == "ask_user":
            print("(Please rephrase your query or provide more detail.)\n")


if __name__ == "__main__":
    main()
