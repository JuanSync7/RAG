<!-- @summary
Documentation for the Embedding Pipeline phase of the RagWeave ingestion system: spec, design, implementation guide, engineering guide, and test docs for both the core embedding pipeline and the cross-document deduplication extension.
@end-summary -->

# docs/ingestion/embedding

Documentation for the Embedding Pipeline — the second major phase of RagWeave ingestion. This phase reads clean Markdown documents from the Clean Document Store and transforms them into vector embeddings stored in the vector database and knowledge graph triples stored in the graph store. It also covers the cross-document deduplication extension that eliminates near-duplicate chunks before embedding.

## Contents

| Path | Purpose |
| --- | --- |
| `EMBEDDING_PIPELINE_SPEC.md` | Formal requirements spec for the embedding phase (FR-600–FR-1304), including chunking, embedding, batch optimisation, and storage |
| `EMBEDDING_PIPELINE_SPEC_SUMMARY.md` | Concise summary of the embedding pipeline spec |
| `EMBEDDING_PIPELINE_DESIGN.md` | Technical design document for the embedding pipeline |
| `EMBEDDING_PIPELINE_IMPLEMENTATION.md` | Implementation guide for the embedding pipeline |
| `EMBEDDING_PIPELINE_ENGINEERING_GUIDE.md` | Post-implementation engineering guide |
| `EMBEDDING_PIPELINE_TEST_DOCS.md` | Test planning document for the embedding pipeline |
| `CROSS_DOCUMENT_DEDUP_SPEC.md` | Formal requirements spec for cross-document deduplication (Tier 1 exact + Tier 2 fuzzy, back-reference model) |
| `CROSS_DOCUMENT_DEDUP_SPEC_SUMMARY.md` | Concise summary of the deduplication spec |
| `CROSS_DOCUMENT_DEDUP_DESIGN.md` | Technical design document for cross-document deduplication |
| `CROSS_DOCUMENT_DEDUP_IMPLEMENTATION.md` | Implementation guide for cross-document deduplication |
| `CROSS_DOCUMENT_DEDUP_ENGINEERING_GUIDE.md` | Post-implementation engineering guide for deduplication |
| `CROSS_DOCUMENT_DEDUP_TEST_DOCS.md` | Test planning document for cross-document deduplication |
