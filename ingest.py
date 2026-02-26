#!/usr/bin/env python3
# @summary
# Ingest documents from documents/ into Weaviate. Main exports: ingest, KnowledgeGraphBuilder, LocalBGEEmbeddings. Deps: pathlib, json, src.core.embeddings, src.ingest.markdown_processor, src.ingest.document_processor, src.core.knowledge_graph, src.core.vector_store, config.settings
# @end-summary
"""Ingest documents from the documents/ folder into Weaviate and build knowledge graph."""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent))

import json

from config.settings import DOCUMENTS_DIR, KG_PATH, KG_OBSIDIAN_EXPORT_DIR, GLINER_ENABLED, PROCESSED_DIR
from src.core.embeddings import LocalBGEEmbeddings
from src.ingest.markdown_processor import process_document_markdown, clean_document
from src.ingest.document_processor import extract_metadata, metadata_to_dict
from src.core.knowledge_graph import KnowledgeGraphBuilder, export_obsidian
from src.core.vector_store import get_weaviate_client, ensure_collection, add_documents, delete_collection


def ingest(
    documents_dir: Path = DOCUMENTS_DIR,
    fresh: bool = True,
    build_kg: bool = True,
    obsidian_export: bool = False,
    semantic_chunking: bool = True,
    export_processed: bool = False,
) -> None:
    """Ingest all text files from the documents directory.

    Args:
        documents_dir: Path to the directory containing documents.
        fresh: If True, delete existing collection before ingesting.
        build_kg: If True, build a knowledge graph from the chunks.
        obsidian_export: If True, export KG as Obsidian-compatible markdown files.
        semantic_chunking: If True, use semantic similarity for chunk splitting.
        export_processed: If True, save cleaned docs and chunks to processed/ dir.
    """
    # Gather document files
    doc_files = sorted(documents_dir.glob("*.txt"))
    if not doc_files:
        print(f"No .txt files found in {documents_dir}")
        return

    print(f"Found {len(doc_files)} document(s) to ingest.")

    # Load embedder FIRST so it can be shared for semantic chunking + final embedding
    print("Loading embedding model...")
    embedder = LocalBGEEmbeddings()

    # Process documents into chunks
    all_chunks = []
    kg_builder = KnowledgeGraphBuilder(use_gliner=GLINER_ENABLED) if build_kg else None

    if export_processed:
        PROCESSED_DIR.mkdir(exist_ok=True)

    for doc_file in doc_files:
        print(f"  Processing: {doc_file.name}")
        raw_text = doc_file.read_text(encoding="utf-8")
        chunks = process_document_markdown(
            raw_text,
            source=doc_file.name,
            embedder=embedder if semantic_chunking else None,
        )
        all_chunks.extend(chunks)

        # Feed chunks to knowledge graph builder
        if kg_builder:
            for chunk in chunks:
                kg_builder.add_chunk(chunk.text, source=doc_file.name)

        # Export processed document for inspection
        if export_processed:
            stem = doc_file.stem
            # Save cleaned document (post-processing, pre-chunking)
            cleaned_text = clean_document(raw_text)
            doc_metadata = extract_metadata(raw_text, doc_file.name)
            (PROCESSED_DIR / f"{stem}.cleaned.md").write_text(
                cleaned_text, encoding="utf-8"
            )
            # Save chunks as JSON with metadata
            chunks_data = [
                {"chunk_index": i, "text": c.text, "metadata": c.metadata}
                for i, c in enumerate(chunks)
            ]
            (PROCESSED_DIR / f"{stem}.chunks.json").write_text(
                json.dumps(chunks_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    print(f"Total chunks after processing: {len(all_chunks)}")
    if export_processed:
        print(f"Processed documents exported to: {PROCESSED_DIR}")

    # Generate embeddings (reuses same embedder instance — model already loaded)
    print("Generating embeddings...")
    texts = [c.text for c in all_chunks]
    metadatas = [c.metadata for c in all_chunks]
    embeddings = embedder.embed_documents(texts)

    # Store in Weaviate
    print("Storing in Weaviate...")
    with get_weaviate_client() as client:
        if fresh:
            print("  Deleting existing collection for fresh ingestion...")
            delete_collection(client)
        ensure_collection(client)
        count = add_documents(client, texts, embeddings, metadatas)
        print(f"  Stored {count} chunks in Weaviate.")

    # Save knowledge graph
    if kg_builder:
        kg_builder.save(KG_PATH)
        stats = kg_builder.stats()
        print(f"Knowledge graph: {stats['nodes']} nodes, {stats['edges']} edges -> {KG_PATH}")

        if obsidian_export:
            n = export_obsidian(kg_builder.graph, KG_OBSIDIAN_EXPORT_DIR)
            print(f"Obsidian export: {n} files -> {KG_OBSIDIAN_EXPORT_DIR}")

    print("Ingestion complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents into RAG system")
    parser.add_argument(
        "--no-kg", action="store_true", help="Skip knowledge graph building"
    )
    parser.add_argument(
        "--export-obsidian",
        action="store_true",
        help="Export knowledge graph as Obsidian markdown files",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable semantic chunking (use character splitting only)",
    )
    parser.add_argument(
        "--export-processed",
        action="store_true",
        help="Export cleaned documents and chunks to processed/ directory",
    )
    args = parser.parse_args()

    ingest(
        build_kg=not args.no_kg,
        obsidian_export=args.export_obsidian,
        semantic_chunking=not args.no_semantic,
        export_processed=args.export_processed,
    )
