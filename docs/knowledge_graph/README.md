<!-- @summary
Documentation for the RagWeave Knowledge Graph subsystem: specs, design docs, implementation guides, engineering guide, build and eval plans, retrieval specs across two phases, and design sketches covering entity extraction, graph storage, query expansion, and retrieval integration.
@end-summary -->

# docs/knowledge_graph

Documentation for the RagWeave Knowledge Graph (KG) subsystem — a modular package (`src/knowledge_graph/`) responsible for entity extraction, typed-edge graph storage, query expansion, and entity description management. The subsystem integrates with the Embedding Pipeline during ingestion and with the Retrieval Pipeline at query time.

## Contents

| Path | Purpose |
| --- | --- |
| `KNOWLEDGE_GRAPH_SPEC.md` | Formal requirements spec for the KG subsystem (schema, extraction, storage, query, community detection) |
| `KNOWLEDGE_GRAPH_SPEC_SUMMARY.md` | Concise summary of the KG subsystem spec |
| `KNOWLEDGE_GRAPH_DESIGN.md` | Technical design document for the KG subsystem |
| `KNOWLEDGE_GRAPH_IMPLEMENTATION.md` | Implementation guide for the KG subsystem |
| `KNOWLEDGE_GRAPH_ENGINEERING_GUIDE.md` | Post-implementation engineering guide |
| `KNOWLEDGE_GRAPH_BUILD_PLAN.md` | Execution plan for building the KG subsystem |
| `KNOWLEDGE_GRAPH_TEST_PLAN.md` | Test planning document for the KG subsystem |
| `KNOWLEDGE_GRAPH_EVAL_PLAN.md` | Evaluation framework for measuring KG quality |
| `KNOWLEDGE_GRAPH_RETRIEVAL_SPEC.md` | Phase 1 retrieval spec: typed edge traversal, path pattern queries, graph-context prompt injection |
| `KNOWLEDGE_GRAPH_RETRIEVAL_SPEC_SUMMARY.md` | Concise summary of the Phase 1 retrieval spec |
| `KNOWLEDGE_GRAPH_RETRIEVAL_SPEC_P2.md` | Phase 2 retrieval spec extending typed traversal and multi-hop reasoning |
| `KNOWLEDGE_GRAPH_RETRIEVAL_SPEC_SUMMARY_P2.md` | Concise summary of the Phase 2 retrieval spec |
| `KNOWLEDGE_GRAPH_RETRIEVAL_DESIGN.md` | Technical design for Phase 1 KG retrieval |
| `KNOWLEDGE_GRAPH_RETRIEVAL_DESIGN_P2.md` | Technical design for Phase 2 KG retrieval |
| `KNOWLEDGE_GRAPH_RETRIEVAL_IMPLEMENTATION_DOCS.md` | Implementation guide for Phase 1 KG retrieval |
| `KNOWLEDGE_GRAPH_RETRIEVAL_IMPLEMENTATION_DOCS_P2.md` | Implementation guide for Phase 2 KG retrieval |
| `KNOWLEDGE_GRAPH_RETRIEVAL_ENGINEERING_GUIDE.md` | Post-implementation engineering guide for KG retrieval |
| `KNOWLEDGE_GRAPH_RETRIEVAL_TEST_DOCS.md` | Test planning document for KG retrieval |
| `KNOWLEDGE_GRAPH_PHASE1B_DESIGN.md` | Design document for Phase 1B KG work |
| `KNOWLEDGE_GRAPH_PHASE2_DESIGN.md` | Design document for Phase 2 KG work |
| `KNOWLEDGE_GRAPH_PHASE3_DESIGN.md` | Design document for Phase 3 KG work |
| `2026-04-08-kg-subsystem-sketch.md` | Initial brainstorm/pre-spec design sketch for the KG subsystem |
| `2026-04-08-kg-phase1b-sketch.md` | Design sketch for Phase 1B |
| `2026-04-08-kg-phase2-sketch.md` | Design sketch for Phase 2 |
| `2026-04-09-kg-phase3-sketch.md` | Design sketch for Phase 3 |
| `DOC_KG_EXPLORATION.md` | Design exploration notes from a chat discussion covering KG architecture and retrieval patterns |
