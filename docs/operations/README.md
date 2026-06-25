<!-- @summary
Operations documentation: deployment topology (uk-ai03 prod+dev), platform
spec/implementation, Podman migration guide and architecture, and 100-user
execution plan.
@end-summary -->

# docs/operations

## Overview

Engineering documentation for deploying, operating, and scaling the RAG platform in production and development environments.

## Files

| File | Purpose |
| --- | --- |
| `DEPLOYMENT_TOPOLOGY.md` | **Start here** — what runs where on uk-ai03: the base+ai03+env compose layering, the prod and dev stacks (live vs target), the uk-ai01 vLLM inference path, how to operate each, and the prod cutover plan |
| `OPERATIONS_PLATFORM_SPEC.md` | Operations platform specification (deployment profiles, scaling, monitoring, DR, CI/CD) |
| `OPERATIONS_PLATFORM_IMPLEMENTATION.md` | Operations platform implementation guide |
| `PODMAN_SPEC.md` | Podman migration specification (rootless containers, drop-in Docker replacement) |
| `PODMAN_ARCHITECTURE.md` | Podman architecture overview and design decisions |
| `PODMAN_IMPLEMENTATION.md` | Podman implementation guide |
| `PODMAN_MIGRATION_GUIDE.md` | Step-by-step guide for migrating from Docker to Podman |
| `PODMAN_TEST_SPEC.md` | Podman migration test specification |
| `RAG_100_USER_EXECUTION_PLAN.md` | Execution plan for supporting 100 concurrent users (load, scaling, SLOs) |
