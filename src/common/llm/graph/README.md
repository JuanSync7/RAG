<!-- @summary
LangGraph workflow primitives: a fluent StateGraph builder, checkpoint backend factory, and human-in-the-loop gate. Used by pipeline stages that need state-machine execution with optional persistence or human review.
@end-summary -->

# src/common/llm/graph

Thin wrappers around LangGraph that expose a stable, project-level API for building and running state-machine workflows. Callers use `workflow(state_schema)` to assemble a graph, optionally attach a checkpointer, and compile it into a `CompiledWorkflow` that supports synchronous, asynchronous, and streaming execution.

## Contents

| Path | Purpose |
| --- | --- |
| `__init__.py` | Public re-exports: `workflow`, `WorkflowBuilder`, `CompiledWorkflow`, `get_checkpointer`, `human_gate` |
| `workflow.py` | `WorkflowBuilder` fluent API and `CompiledWorkflow` executor wrapping `StateGraph` |
| `checkpoint.py` | `get_checkpointer()` factory — returns a `MemorySaver` or `SqliteSaver` based on backend config |
| `interrupt.py` | `human_gate()` — pauses a running workflow for human input via `langgraph.types.interrupt`, with provisional auto-approve fallback |
