# Ingest Throughput Baseline — CPU vs GPU

**Date**: 2026-04-24
**Host**: WSL2 (Linux 6.6.87.2-microsoft-standard-WSL2), NVIDIA GeForce RTX 2060 (6GB VRAM)
**Weaviate**: `semitechnologies/weaviate:1.28.0` container, shared, fresh (no prior data)
**Workload**: 5 random `.md` files from `./docs/` (seed=42), ~239 KB total, produces 619 chunks

## Results

| Metric | CPU | GPU | GPU speedup |
|---|---|---|---|
| **Wall-clock** | 532.4 s | 53.8 s | **9.9×** |
| **chunks/sec** | 1.16 | 11.50 | **9.9×** |
| **docs/sec** | 0.009 | 0.093 | **10.3×** |
| **kB/sec** | 0.45 | 4.44 | 9.9× |
| stored_chunks | 619 | 619 | (deterministic) |

## Pipeline config

All optional stages disabled for a clean embedding-bound measurement:
- `enable_llm_metadata=False` (no Ollama/LiteLLM calls for summaries/keywords)
- `enable_docling_parser=False` (pure `.md` input, no Docling)
- `enable_vision_processing=False`
- `enable_knowledge_graph_*=False`
- `enable_quality_validation=False`
- `store_documents=False` (no MinIO writes)
- `semantic_chunking=False`
- Default chunk_size=512, chunk_overlap=50, embedding_batch_size=64

## Where the time goes (GPU)

53.8 s total / 619 chunks:
- Embedding on GPU: ~1–3 s per sentence-transformer batch of 32 inputs → ~30–40 s total embedding.
- Model load: ~10–15 s (cold).
- Weaviate insert + pipeline overhead: ~5–10 s.

## Where the time goes (CPU)

532 s / 619 chunks:
- Embedding on CPU: ~20–30 s per batch of 32 → ~450–500 s total embedding (dominant).
- Everything else roughly the same as GPU.

## Observations

1. **GPU is 10× faster end-to-end**; the embedding step is where all the gain lives.
2. **Batch internal to sentence-transformers is hardcoded at 32** (`src/core/embeddings.py:42`). With RTX 2060's 6GB VRAM, this is likely under-utilizing the GPU — room to tune up.
3. **`embedding_batch_size=64`** (IngestionConfig) controls how many chunks are handed to one `embed_documents` call. Each call then re-splits at 32. Increasing both together should help on GPU.
4. **Semantic chunking (disabled here)** would add more embedding work; its CPU-vs-GPU sensitivity is untested.
5. **Weaviate insert is not the bottleneck** at this throughput — batch.dynamic() handles 619 objects in seconds.

## Reproduce

```bash
# Weaviate must be up.
./scripts/compose.sh up -d rag-weaviate

# Export LD_LIBRARY_PATH (torch cuda libs) — required for both CPU and GPU.
export NV_LIBS=$(find .venv/lib/python3.12/site-packages/nvidia -name "lib" -type d | paste -sd: -)
export LD_LIBRARY_PATH=$NV_LIBS
export OMP_NUM_THREADS=4
export MKL_THREADING_LAYER=GNU

# CPU
uv run --no-sync python scripts/bench_ingest.py --device cpu --limit 5 --docs-dir docs

# GPU
uv run --no-sync python scripts/bench_ingest.py --device gpu --limit 5 --docs-dir docs
```

Results land in `scripts/.bench_results/<device>-<epoch>.json`.
