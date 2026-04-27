<!-- @summary
OpenShell sandbox and inference routing policy files for the RAG stack. Each file restricts a specific workload to the filesystem paths, network destinations, and credentials it needs.
@end-summary -->

# openshell/policies/

Each file in this directory is an OpenShell policy applied to one sandbox or routing layer. Policies are referenced by name in `workspace.yaml` and loaded by the OpenShell runtime at container startup.

## Contents

| Path | Purpose |
| --- | --- |
| `rag-worker.yaml` | `SandboxPolicy` for the Temporal worker — restricts filesystem access to `/app`, `/models`, and per-worker Weaviate data; allows network access to Temporal, Redis, Ollama, and Langfuse; enables inference routing via `privacy-router` |
| `mcp-adapter.yaml` | `SandboxPolicy` for the MCP adapter — most restrictive policy; stdio-only with a single allowed network destination (`rag-api:8000`); requires human approval for all admin tool invocations |
| `privacy-router.yaml` | `InferencePolicy` shared by sandboxes that enable inference routing — classifies outbound LLM calls and routes PII or regulated-tenant queries to the local Nemotron model instead of cloud providers |
