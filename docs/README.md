<!-- @summary
Engineering documentation for the RAG platform, organized by subsystem: ingestion, retrieval, server API, UI/console, platform services, operations, performance, LLM integration, open shell, and features.
@end-summary -->

# docs

## Overview

This directory contains engineering specifications, design documents, implementation guides, and operations runbooks for the RAG platform.

## Subdirectories

| Directory | Contents |
| --- | --- |
| `eval/` | Offline eval loop engineering guide (pack format, runner stages, judge, gate, CLI) |
| `ingestion/` | Ingestion pipeline spec, design docs, implementation guides, engineering guide, onboarding checklist |
| `retrieval/` | Retrieval pipeline spec, design docs, implementation guides, NeMo Guardrails, engineering guide, onboarding checklist |
| `server/` | FastAPI server API spec, implementation guide, platform services spec |
| `ui/` | CLI and web console spec, design, and implementation docs; token budget spec/implementation |
| `platform/` | LiteLLM multi-tenant setup guide |
| `operations/` | Operations platform spec/implementation, Podman migration, 100-user execution plan |
| `performance/` | Retrieval and ingestion performance specifications |
| `llm/` | LiteLLM SDK integration guide and spec |
| `openshell/` | Open Shell specification and implementation guide |
| `features/` | Feature specs (feedback loop, user contribution) |
| `superpowers/` | Implementation plans, design specs, and skill reference documents |

## Key Starting Points

| Goal | Document |
| --- | --- |
| Understand eval loop | `eval/EVAL_LOOP_ENGINEERING_GUIDE.md` |
| Understand ingestion pipeline | `ingestion/INGESTION_PIPELINE_ENGINEERING_GUIDE.md` |
| Understand retrieval pipeline | `retrieval/RETRIEVAL_ENGINEERING_GUIDE.md` |
| Understand server API | `server/SERVER_API_SPEC.md` |
| Set up web console | `ui/WEB_CONSOLE_SPEC.md` |
| Deploy / operate | `operations/OPERATIONS_PLATFORM_SPEC.md` |
| Configure LLMs | `llm/LITELLM_INTEGRATION.md` |
