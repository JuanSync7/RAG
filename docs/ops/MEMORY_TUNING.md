<!-- @summary
Memory tuning for RagWeave on RAM-constrained hosts: the durable, box-agnostic
Weaviate GOMEMLIMIT mechanism (ops/weaviate/memwrap.sh + RAG_WEAVIATE_MEM_FRACTION),
why cgroup mem-limits don't work everywhere, the soft-limit caveat, and the
structural fix (vector quantization). Also lists the dev-only band-aids to retire.
@end-summary -->

# Memory Tuning (Weaviate + ingestion)

## Problem

Weaviate keeps its **HNSW vector index in RAM by design** (graph + float32 vectors).
For a collection of `N` chunks at `D` dims that is roughly `N * D * 4 bytes` for the
vectors alone, plus graph links and inverted index. On a small single host (e.g. the
15 GB, **no-swap** ai03 dev box) this competes with the ingestion worker (Docling +
embedding batches need 2–4 GB for large PDFs) and the API/query services. When the
sum exceeds RAM, the **kernel OOM-killer** kills a process — usually mid-ingest — and
on a no-swap box this is abrupt and can cascade into a watchdog reboot.

Two failure modes seen in practice:
- Weaviate RSS **ballooning** during heavy ops (segment compaction, full-corpus
  `Like`/`Aggregate` scans) — transient spikes well above the steady-state working set.
- Large documents (multi-MB Synopsys databook PDFs) spiking the ingest worker and
  tipping the box over the edge.

## The durable mechanism: self-computing `GOMEMLIMIT`

Weaviate is a Go binary, so `GOMEMLIMIT` (a soft heap target) bounds the balloon:
as the heap approaches the limit, the Go GC runs more aggressively instead of the
process being OOM-killed.

The **box-agnostic** way to set it is `ops/weaviate/memwrap.sh`, used as the
container entrypoint. At startup it:
1. detects the memory the container can actually use — cgroup v2 `memory.max`, else
   cgroup v1 `memory.limit_in_bytes`, else host `/proc/meminfo` `MemTotal`;
2. computes `GOMEMLIMIT = detected_bytes * RAG_WEAVIATE_MEM_FRACTION / 100`;
3. `exec`s the real Weaviate binary with the original args.

So there is **one knob** — `RAG_WEAVIATE_MEM_FRACTION` (percent, default `50`) — and
it self-adapts to whatever box/cgroup it lands on. Wired in `docker-compose.yml`
(`rag-weaviate`) and the dev compose (`rag-weaviate-dev`):

```yaml
environment:
  RAG_WEAVIATE_MEM_FRACTION: "50"   # weaviate may use ~50% of visible RAM
entrypoint: /memwrap.sh
command: ["--host", "0.0.0.0", "--port", "8080", "--scheme", "http"]
volumes:
  - ./ops/weaviate/memwrap.sh:/memwrap.sh:ro
```

Startup log line to confirm it took effect:
```
[weaviate-mem] GOMEMLIMIT=8066988032B GOGC=75 (frac=50% of detected 16133976064B)
```

### Why not just a cgroup `mem_limit` + `LIMIT_RESOURCES=true`?

That is the "standard" approach and Weaviate supports it — **but it does not work on
ai03**, which is **cgroups v1 + rootless podman**, where `--memory`/`mem_limit` are
*silently ignored* (`"Resource limits are not supported and ignored on cgroups V1
rootless systems"`). `LIMIT_RESOURCES=true` would then read the host's full RAM and
target ~80% of it — no headroom left for ingest. The self-computing wrapper avoids
this by reading whatever limit *is* enforced and falling back to host RAM otherwise,
so the same config behaves sensibly on cgroup v1 rootless, cgroup v2, and k8s.

## Critical caveat: `GOMEMLIMIT` is SOFT

It bounds **spikes and transients above the live heap**. It **cannot shrink a working
set that is genuinely larger than the cap** — if the reachable HNSW data needs 11 GB
and the cap is 8 GB, the GC simply runs hot (CPU burn) and RSS stays ~11 GB anyway.

So `RAG_WEAVIATE_MEM_FRACTION` must be set **above** the steady-state working set. If
the working set itself exceeds a healthy fraction of host RAM, the cap is the wrong
tool — see the structural fix below.

### Picking the fraction
- Dedicated Weaviate box: `70–80`.
- Shared box (Weaviate + ingest + query + services, e.g. ai03): `50` leaves room,
  *provided* the working set fits under it. Measure steady-state RSS and set the
  fraction a little above it; if you can't (working set too big), quantize.

## Structural fix: vector quantization

The durable way to make a large index *fit* (not just bound spikes) is to shrink the
vectors themselves at the HNSW level:
- **SQ** (scalar / int8): ~4× smaller vectors, small recall loss (recoverable with a
  reranker, which RagWeave already runs).
- **BQ** (binary): ~32× smaller, larger recall loss.

This requires creating the collection with quantization enabled and re-indexing
(re-ingest). For a corpus dominated by exploded spreadsheet chunks, **also reconsider
the chunking** — a single sheet becoming thousands of chunks inflates the index for
little retrieval value (see the spreadsheet-chunking discussion in the ingestion docs).

## Dev-only band-aids (NOT durable — retire once the above is in place)

These kept the RAM-tight dev box limping during bring-up; they are **operational
crutches, not fixes**, and should not be relied on in prod:
- `mem_guard.sh` — host script that SIGKILLs the ingest CLI under pressure.
- Manually `podman restart rag-weaviate-dev` to reclaim ballooned RSS.
- Stopping the prod stack to free RAM.
- Quarantining large source PDFs to stop ingest thrashing on them.

The durable replacements are: the `GOMEMLIMIT` wrapper (spikes), quantization
(working-set size), and host **swap** (a real OOM safety net — ai03 currently has
none, which is why kernel OOM-kills are so abrupt).
