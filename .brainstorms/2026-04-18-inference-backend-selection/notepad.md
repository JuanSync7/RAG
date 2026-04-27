# Brainstorm Notes — Inference Backend Selection

## Status

- **Phase:** B (final circle-back)
- **Type:** decision
- **Anticipated shape:** Decision
- **Turn count:** 3
- **Selected lenses:** Stakeholder, Alternative (universal) + Reversibility, Failure-mode (decision type-specific)

## Context

- Hardware: GPU exists, WSL2 doesn't expose it. Native Linux = full GPU.
- Model pair: Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B (decided)
- Goal: config-driven, consistent, maintainable

## Threads (indexed)

### T1+T7: Hardware constraint — WSL2 vs native Linux
**Status:** resolved
**Resolution:** GPU exists. WSL2 is the constraint, not hardware. Backend must work on CPU (WSL2 dev) and GPU (prod). vLLM satisfies both.

### T2: Model support — Qwen3 pair
**Status:** resolved
**Resolution:** Qwen3 pair confirmed. Narrows to vLLM or Xinference.

### T3: In-process vs separate service
**Status:** resolved
**Lenses applied:** Stakeholder ✓ | Alternative ✓ | Failure-mode ✓
**Resolution:** User correctly identified that in-process failure = same outcome as separate service failure. Failure-mode distinction doesn't favor in-process. Separate service wins for config flexibility — swap model by changing URL + model name, no worker rebuild. Agreed.

### T4: Latency — in-process vs HTTP
**Status:** resolved
**Lenses applied:** Stakeholder ✓ | Alternative ✓
**Resolution:** 5-10ms acceptable. GPU inference is fast (sub-100ms for embed/rerank), so HTTP overhead is negligible in the overall budget. Non-issue.

### T5: CPU/GPU auto-detection + timeout calibration
**Status:** resolved
**Resolution:** Backend handles CPU/GPU automatically. Client doesn't need to know. Simple approach: set conservative timeout at startup (worst case = CPU). Query /health at startup if dynamic calibration needed later. No per-response metadata needed.

### T6: vLLM direct vs Xinference
**Status:** resolved
**Lenses applied:** Stakeholder ✓ | Alternative ✓ | Reversibility ✓ | Failure-mode ✓
**Resolution:**
- Stakeholder ✓: single operator, 2 models — Xinference management UI solves a problem that doesn't exist yet
- Alternative ✓: considered Xinference (management layer), vLLM direct, Ollama (eliminated — no cross-encoder). No better option missed.
- Reversibility ✓: both expose OpenAI-compatible API. BYOM abstraction layer means switching is a config change regardless. Low lock-in.
- Failure-mode ✓: vLLM direct = fewer layers, clearer logs, easier debugging. Xinference adds a layer between you and vLLM's error output.
- User confirmed: management layer not needed for current scale. Consistency is the goal.
**Decision: vLLM direct.**

### T8: Is migration necessary now?
**Status:** resolved
**Resolution:** Config-driven architecture confirmed as goal. Short-term fix (RAG_RERANKER_MAX_LENGTH=128) lands immediately as unblock. BYOM abstraction is a separate track.

## Connections

- T3 resolved → T6 is the only real decision remaining (which separate service)
- T6 resolved → vLLM. T1 confirms vLLM works on both CPU (dev) and GPU (prod).

## Resolution Log

- T1+T7 resolved (turn 2) — GPU exists, WSL2 constraint only
- T2 resolved (turn 2) — Qwen3 pair confirmed
- T8 resolved (turn 2) — config-driven goal confirmed
- T3 resolved (turn 3) — user's failure-mode point correct; separate service wins
- T4 resolved (turn 3) — HTTP overhead negligible
- T5 resolved (turn 3) — conservative timeout sufficient
- T6 resolved (turn 3) — vLLM direct, Stakeholder ✓ Alternative ✓ Reversibility ✓ Failure-mode ✓

## Key Insights

- The failure-mode argument for in-process doesn't hold — if the process dies, inference dies regardless
- Xinference's value is multi-model management at scale; not relevant for 2 models, 1 operator
- vLLM's OpenAI-compatible API means the BYOM abstraction layer is trivial — just a base URL + model name
- Short-term fix and BYOM architecture are independent tracks; don't block one on the other

## Tensions

(none remaining)

## Discarded Candidates

- TGI: maintenance mode
- fastembed: fixed model list
- Ollama for embed/rerank: no cross-encoder
- TEI: uncertain Qwen3 support
- Xinference: management overhead not justified at current scale; revisit if multi-agent, multi-model scenario materializes

## Phase A Misses

(none)
