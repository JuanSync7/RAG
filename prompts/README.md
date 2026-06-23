<!-- @summary
Prompt templates used by retrieval query-processing stages, including legacy
split prompts and the combined reformulate+evaluate prompt.
@end-summary -->

# prompts

## Overview

This directory contains prompt files consumed by `src/retrieval/query_processor.py`.
The runtime can use a combined reformulate+evaluate call while retaining split
prompt templates for compatibility and experimentation.

## Files

| File | Purpose |
| --- | --- |
| `rag_system.md` | Answer-generation system prompt (loaded by `src/retrieval/generation/nodes/generator.py`). |
| `query_reformulate_and_evaluate.md` | Primary combined prompt for reformulation + evaluation JSON output. |
| `query_reformulator.md` | Legacy reformulation-only prompt template. |
| `query_evaluator.md` | Legacy evaluation-only prompt template. |
| `retrieval_query_rewriter.md` | Retrieval-mode query rewriter. |
| `deep_research_multi_queries_gen.md` | Deep-research sub-query generation. |
| `deep_research_sufficiency_check.md` | Deep-research stopping/sufficiency check. |

## Design notes

### Grounding is intentionally *soft* (not strict)

`rag_system.md` tells the model to *"Ground every claim in the context and do not
invent facts it does not support"* and to cite with `[N]`, but it deliberately does
**not** restrict the answer to the context **only** (contrast the minimal fallback
prompt in `generator.py`, which does say *"using ONLY the provided context"*). It
also explicitly authorizes *clearly-supported inferences* and discourages refusing
when the answer is implicit/spread across chunks. Consequence: a `[N]` marker means
"this chunk supports this claim", **not** "this sentence is verbatim from chunk N",
and the model may blend genuine background knowledge. This is a tuning choice that
trades a little faithfulness for far fewer false "context doesn't cover this"
refusals — keep it in mind before tightening.

### Specificity directives (added to fight generic/high-level drift)

- `rag_system.md` carries a *"Be specific, not generic"* directive: when the context
  has an exact value / name / identifier / signal-or-register name / version / number,
  state it verbatim rather than paraphrasing it into a generality, and do not
  substitute general background knowledge for specifics the context provides.
- `query_reformulate_and_evaluate.md`'s **"Preserve specificity"** rule dominates the
  reformulation rules: an already-specific query (concrete terms, proper nouns,
  identifiers, signal/register names, versions, numbers) is returned essentially
  unchanged — no synonym expansion — because over-broadening dilutes the search
  vector and surfaces generic overview material instead of the precise chunk.

## Internal Dependencies

- `rag_system.md` loaded by `src/retrieval/generation/nodes/generator.py`.
- Query-processing prompts loaded by `src/retrieval/query_processor.py` through prompt loader helpers.

## Subdirectories

None
