# Decision Memo — Inference Backend Selection

**Shape:** Decision
**Type:** decision
**Session:** `2026-04-18-inference-backend-selection`

---

### Question

Which inference backend should serve embeddings and reranking in RagWeave — Ollama, vLLM, Xinference, or TEI?

### Options considered

- **vLLM direct** — GPU-first inference server, OpenAI-compatible API, explicit support for Qwen3-Embedding and Qwen3-Reranker. Single process, minimal layers.
- **Xinference** — management layer wrapping vLLM (GPU) and llama.cpp (CPU). Adds UI, model registry, multi-backend routing.
- **Ollama** — already running for generation. CPU-native, quantized GGUF. No cross-encoder reranking support.
- **TEI** — HuggingFace Text Embeddings Inference. Uncertain Qwen3 support. Not a universal cross-encoder server.

### Tradeoffs (per lens)

| Lens | vLLM direct | Xinference | Ollama (embed/rerank) |
|------|-------------|------------|----------------------|
| Stakeholder | Single operator, 2 models — direct control, clear logs | Management UI solves a problem that doesn't exist at this scale | Already running but can't do cross-encoder reranking |
| Alternative | No better open option found that supports Qwen3 pair natively | Adds value at multi-model, multi-agent scale — not now | Eliminated on capability gap |
| Reversibility | OpenAI-compatible API → switching backends is a config change | Same API surface, equally reversible | N/A |
| Failure-mode | One process, direct error output, easy to debug | Extra layer between operator and vLLM logs — harder to diagnose | N/A |

### Recommendation

**vLLM direct**, running as a separate Docker Compose service, serving both Qwen3-Embedding-0.6B and Qwen3-Reranker-0.6B via OpenAI-compatible endpoints.

### Rationale

The Qwen3 embedding + reranker pair is the only complete, coherent pair where both models are explicitly vLLM-supported and outperform the current BGE setup. vLLM is the right backend to serve them — not because it's the most featureful, but because it's the most direct path: one service, clear ownership, OpenAI-compatible API.

Xinference was the main alternative considered. Its value proposition is multi-backend routing (llama.cpp on CPU, vLLM on GPU) under a single management interface. That matters when you're running many models across many agents with different hardware. It does not matter for two models on one node with one operator. The management layer would add debugging complexity with no operational return.

The GPU hardware concern resolved itself: the user has a GPU, the constraint is WSL2 (dev environment only). On native Linux, vLLM runs at full GPU throughput. On WSL2/CPU during development, vLLM runs in CPU mode — slower, but functional for integration testing. The BYOM abstraction layer means the application code never sees the difference.

### Risks accepted

- **CPU mode is slow on vLLM.** Dev/WSL2 inference will be noticeably slower than production. Accepted — the alternative (keeping BGE in-process) also had 82s latency on CPU. Short-term fix (`RAG_RERANKER_MAX_LENGTH=128`) addresses the immediate blocker without waiting for the architecture work.
- **Separate service is a hard dependency.** If vLLM goes down, inference fails. No in-process fallback. Accepted — this is the same failure mode as in-process worker death, and the config-driven architecture goal requires a separate service.
- **Xinference not chosen now.** If the system evolves to many agents each using different models, Xinference becomes the right answer. The decision is reversible — the BYOM abstraction layer isolates this to a config change.

---

### What surfaced along the way

- The "in-process vs separate service" failure-mode argument was a false distinction — worker death kills in-process inference too. The real argument for separate service is config-driven model switching without worker rebuilds.
- Cross-encoder reranking is architecturally different from generation and bi-encoder embedding. This is why BGE-reranker-v2-m3 is not supported in many tools — most inference servers are built for decoder-only generation, not cross-encoder scoring.
- Embedding and reranking models must share the same family (same tokenizer, same representation space). Rerankers don't need to match the embedding model — they operate on raw text, not vectors.

### Open questions

- Does vLLM CPU mode run acceptably for dev iteration, or does it need ONNX Runtime as a CPU-specific path? Worth testing empirically once vLLM is added to the compose stack.
- At what scale does Xinference become worth adding? A rough heuristic: when you have 3+ distinct models or 2+ agents that need different model backends simultaneously.

### You might consider next

The natural next step is implementing the BYOM abstraction layer in RagWeave: define `EmbeddingProvider` and `RerankerProvider` protocols, wrap existing BGE classes behind them, add an `HTTPProvider` pointing at vLLM, and add vLLM as a Docker Compose service. This is a concrete implementation task — a good candidate for direct execution rather than further brainstorming.

Before starting that, the short-term fix (`RAG_RERANKER_MAX_LENGTH=128` + `RAG_RERANKER_PRECISION=fp32` in `.env` → `make restart-worker`) should land first to unblock the current 82s reranking problem.

### Artifacts

- Notepad: `.brainstorms/2026-04-18-inference-backend-selection/notepad.md`
- Meta: `.brainstorms/2026-04-18-inference-backend-selection/meta.yaml`
