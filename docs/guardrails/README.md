<!-- @summary
Documentation for the Colang 2.0 Guardrails subsystem: spec, design, implementation plan, engineering guide, and design reference for the declarative policy layer that wraps the RAG pipeline. Also contains the NeMo Guardrails runtime integration docs in the nested `nemo_guardrails/` subdirectory.
@end-summary -->

# docs/guardrails

Documentation for the RagWeave guardrails subsystem — a dual-layer safety and policy system built on Colang 2.0 and NeMo Guardrails. The declarative Colang layer (33 flows across 5 `.co` files) expresses what to block, hedge, or escalate; the Python executor layer (26 `@action()` wrappers) performs the heavy computation. The two layers are joined through a single `generate_async()` call to the NeMo runtime.

## Contents

| Path | Purpose |
| --- | --- |
| `COLANG_DESIGN_GUIDE.md` | Colang 2.0 syntax reference and design principles used throughout the flow files |
| `COLANG_GUARDRAILS_SPEC.md` | Formal requirements specification for the Colang declarative policy layer (COLANG-101–COLANG-911) |
| `COLANG_GUARDRAILS_SPEC_SUMMARY.md` | Concise summary of the Colang guardrails spec |
| `COLANG_GUARDRAILS_DESIGN.md` | Technical design document: dual-layer architecture, flow ordering, tunable knobs |
| `COLANG_GUARDRAILS_IMPLEMENTATION.md` | Six-phase retroactive implementation plan (preserved for traceability) |
| `COLANG_GUARDRAILS_ENGINEERING_GUIDE.md` | Post-implementation engineering guide: how to run, extend, and troubleshoot the subsystem |
| `nemo_guardrails/` | NeMo Guardrails runtime integration — spec, spec summary, and implementation guide |
