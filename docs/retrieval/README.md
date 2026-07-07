<!-- @summary
Retrieval pipeline documentation: spec summary and onboarding checklist at the top level; full sub-pipeline docs (spec, design, implementation, engineering guide, tests) split into query/ and generation/.
@end-summary -->

# docs/retrieval

## Overview

Engineering documentation for the AION RAG retrieval pipeline. The pipeline is split into two sub-pipelines, each with its own spec, design, implementation plan, engineering guide, and module tests.

## Subdirectories

| Directory | Covers |
| --- | --- |
| [`query/`](query/README.md) | Query processing, conversation memory, pre-retrieval guardrails, hybrid search, reranking |
| [`generation/`](generation/README.md) | Document formatting, LLM generation, post-generation guardrails, observability, NFR |

## Cross-Cutting Files

| File | Purpose |
| --- | --- |
| `RETRIEVAL_SPEC_SUMMARY.md` | Concise summary spanning both query and generation specs |
| `RETRIEVAL_NEW_ENGINEER_ONBOARDING_CHECKLIST.md` | One-page onboarding checklist for new engineers |
| `DOCUMENT_ROUTING_DESIGN.md` | RAPTOR-lite document-routing design / rationale (implemented, gated OFF by default) |
| `DOCUMENT_ROUTING_ENGINEERING_GUIDE.md` | As-built document-routing guide: architecture, flow, config keys, how-to-enable, extension, troubleshooting |
| `DOCUMENT_ROUTING_IMPLEMENTATION_PLAN.md` | The slice plan the routing feature was built from |
| `CHUNK_ROLE_AND_METADATA.md` | Chunk-role taxonomy (`content`/`navigation`/`boilerplate`), the single-source vocabulary (`RAG_CHUNK_ROLES`), the ingest→store→query→backfill mechanism, which chunk metadata drives retrieval, and the metadata roadmap |

## Key Starting Points

- **New to the codebase?** Start with `RETRIEVAL_NEW_ENGINEER_ONBOARDING_CHECKLIST.md`
- **Understanding requirements?** Read `RETRIEVAL_SPEC_SUMMARY.md`, then drill into `query/RETRIEVAL_QUERY_SPEC.md` or `generation/RETRIEVAL_GENERATION_SPEC.md`
- **Implementation details?** See `query/RETRIEVAL_QUERY_ENGINEERING_GUIDE.md` or `generation/RETRIEVAL_GENERATION_ENGINEERING_GUIDE.md`
- **Guardrails setup?** See `docs/guardrails/nemo_guardrails/`
