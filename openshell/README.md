<!-- @summary
OpenShell sandbox configuration for the RAG stack. Contains the workspace definition that manages sandbox lifecycle for the Temporal worker and MCP adapter, plus sandbox and inference routing policies.
@end-summary -->

# openshell/

This directory contains the OpenShell configuration for running RAG stack workloads inside sandboxed containers. The workspace definition controls which agent processes are launched, their container images, resource limits, and policy bindings. Policies themselves live in `policies/`.

## Contents

| Path | Purpose |
| --- | --- |
| `workspace.yaml` | OpenShell `Workspace` manifest — defines the `rag-worker` and `mcp-adapter` sandboxes, shared defaults, and the credential store configuration |
| `policies/` | Per-sandbox `SandboxPolicy` and inference routing policy files |
