# Tree Retrieval — Design Sketch

**Status:** Draft (v0)
**Branch:** `feat/raptor-treerag-research`
**Inspiration:** TreeRAG (Tao et al., ACL 2025 Findings) — "Tree-Chunking + Bidirectional Traversal Retrieval" for long-document QFS.
**Scope:** Retrieval-side enhancement that exploits document hierarchy. **Does not** add LLM summarization at ingest (that is a separate RAPTOR follow-up — see Appendix B).

---

## 1. Why this and not RAPTOR/GraphRAG

| Question | Answer |
|---|---|
| Does RagWeave already have a tree of documents? | **Yes, implicitly.** `chunking_node` computes `heading_path: list[str]` for every chunk. The structural tree exists in chunk metadata; we just don't use it during retrieval. |
| Is this GraphRAG / "TreeRAG-as-communities"? | No — that's KGWeave's job. This is the document's *natural* heading hierarchy (Title → §1 → §1.2 → ¶), not an LLM-derived community tree. |
| Does this require RAPTOR-style summarization? | No. Tree-Retrieval v1 uses heading text + concatenated leaf text for internal nodes. RAPTOR-style LLM summaries are an optional v2 (Appendix B). |
| Cost shape | **Cheap.** Ingest cost ≈ +1 small embedding per non-leaf node (~5–20% extra vectors per doc). No new LLM calls in v1. Retrieval cost ≈ +1 hybrid search per query. |

The win target: **Query-Focused Summarization on long structured docs** (legal contracts, policy PDFs, technical specs, medical guidelines) where the right answer spans a whole section and flat hybrid+rerank shreds it.

---

## 2. What changes — bird's-eye view

```
              ┌─────────────────────────────────────────────────┐
              │                INGESTION                         │
              │                                                  │
  Docling →   chunking_node (existing)                          │
              │   • produces leaf chunks with heading_path      │
              │   • [NEW] also emits section nodes (one per    │
              │     unique heading_path prefix in the doc)      │
              │                                                  │
              ▼                                                  │
  embedding_storage_node                                        │
              │   • [NEW] writes both kinds with node_kind tag │
              └─────────────────────────────────────────────────┘

              ┌─────────────────────────────────────────────────┐
              │                RETRIEVAL                         │
              │                                                  │
              │  rag_chain.run(...)                              │
              │    Stage 4 (existing): hybrid_search on chunks   │
              │    [NEW] Stage 4b: descent_search on sections   │
              │        → expand top-K sections to their chunks │
              │    [NEW] Stage 4c: context_lift on Stage 4 hits│
              │        → pull sibling chunks at heading_path[:k]│
              │    Stage 5 (existing): rerank merged candidates │
              └─────────────────────────────────────────────────┘
```

Existing behaviour is preserved when `RAG_TREE_RETRIEVAL_ENABLED=false` (default).

---

## 3. Ingest-side changes

### 3.1 Chunk schema additions (Weaviate)

In `src/vector_db/weaviate/store.py`:

| Field | Type | Purpose |
|---|---|---|
| `node_kind` | `TEXT` | `"chunk"` (leaf) \| `"section"` (internal) |
| `tree_level` | `INT` | 0 = root/title, 1 = §, 2 = §§, …; leaves = `len(heading_path)` |
| `heading_path` | `TEXT_ARRAY` | Already computed in chunking_node; not yet persisted. Persist it. |
| `heading_path_str` | `TEXT` | `" > ".join(heading_path)` — already stored as `section_path`; reuse. |
| `parent_section_id` | `TEXT` | Stable hash of parent's `(document_id, heading_path[:-1])`. Allows leaf→parent lookup without string filter gymnastics. |
| `child_count` | `INT` | (sections only) Number of direct children. Skip-routing signal: prune sections with `child_count==0`. |

**Migration path:** new fields are optional; existing chunks get `node_kind="chunk"`, `tree_level = len(heading_path)`, `parent_section_id = hash(doc_id, heading_path[:-1])`. Run as part of `src/ingest/lifecycle/migration.py` (existing migration framework). No re-embedding required.

### 3.2 New ingest node: `tree_node_synthesis`

Location: `src/ingest/embedding/nodes/tree_node_synthesis.py`
Position in graph (`workflow.py`): between `chunk_enrichment` and `metadata_generation`.

Responsibility:
1. From the leaf chunks already in state, derive the set of unique `heading_path` prefixes that exist in the document (e.g. for chunks under `["Ch3", "3.2", "3.2.1"]`, emit section nodes for `["Ch3"]`, `["Ch3", "3.2"]`, `["Ch3", "3.2", "3.2.1"]`).
2. For each section node, build its **section text** = the heading itself + the first ~N tokens of each direct child (truncated to `chunk_size`). This is what gets embedded.
   - **No LLM call.** That's RAPTOR territory (Appendix B).
   - **Tables are first-class children.** When a child chunk has `chunk_type=="table"`, contribute `table_caption` (if present) + the first non-separator markdown row instead of a head-truncated slice. Chip-design specs are table-heavy (register maps, signal lists); a head-truncated table row reads as `"| Field | Bits"` and embeds poorly, while caption + header row carries the actual semantics.
3. Append the section nodes to `state["chunks"]` so downstream nodes (`embedding_storage`, `commit`) handle them uniformly. `node_kind="section"` flag distinguishes them.

Toggleable via `config.enable_tree_retrieval_ingest`. When false, the node is a no-op.

### 3.3 Embedding

`embedding_storage_node` is unchanged — it embeds whatever is in `state["chunks"]`. Section nodes flow through automatically. Same BGE-M3 model, same vector size.

**Ingest cost delta:** for a typical PDF with ~200 chunks across ~30 sections, that's +30 embeddings (+15%), one Weaviate batch insert. Negligible.

---

## 4. Retrieval-side changes

All in `src/retrieval/pipeline/rag_chain.py`. Gated by `RAG_TREE_RETRIEVAL_ENABLED`.

### 4.1 Stage 4b — Descent (root → leaves)

After the existing chunk-restricted hybrid search returns `search_results`:

```python
if RAG_TREE_RETRIEVAL_ENABLED:
    section_filter = SearchFilter(property="node_kind", operator="eq", value="section")
    section_hits = self._do_search(
        bm25_query, query_embedding,
        alpha=alpha,
        limit=RAG_TREE_DESCENT_TOP_K,           # default: 5
        filters=(filters or []) + [section_filter],
    )
    # Expand each top section to its leaf chunks (one filter per parent_section_id, batched)
    expanded_chunks = self._expand_sections_to_leaves(
        [s.metadata["chunk_id"] for s in section_hits],
        per_section_limit=RAG_TREE_DESCENT_LEAVES_PER_SECTION,  # default: 3
        filters=filters,
    )
```

Retrieval ranks **section** nodes against the query (descent). Then for each high-scoring section we fetch its leaf chunks. The intuition: a section that scores high overall is likely to contain the answer somewhere in its leaves, even if no single leaf scored high on its own.

### 4.2 Stage 4c — Lift (leaf → root)

For each top-N from Stage 4 (chunk hits), fetch sibling chunks under the same parent section, capped:

```python
sibling_groups = group_by(search_results[:RAG_TREE_LIFT_SEED_K],
                         key=lambda r: r.metadata["parent_section_id"])
lifted_chunks = self._fetch_siblings(
    sibling_groups,
    per_group_limit=RAG_TREE_LIFT_SIBLINGS,  # default: 2
    filters=filters,
)
```

This restores adjacency that flat retrieval shreds. The intuition: if leaf `3.2.1.b` is the best hit, leaves `3.2.1.a` and `3.2.1.c` are likely co-relevant.

### 4.2.1 Chip-design KB profile

For chip-design corpora (datasheets, AMBA-style protocol specs, IP block specs), defaults shift:

| Knob | General default | Chip-design profile |
|---|---|---|
| `RAG_TREE_LIFT_SIBLINGS` | 2 | **4** |
| `RAG_TREE_DESCENT_LEAVES_PER_SECTION` | 3 | 3 (unchanged) |
| `RAG_TREE_DESCENT_TOP_K` | 5 | 5 (unchanged) |

Reason: chip-design sections (e.g. `§5.3.2 AXI Read Response Channel`) are denser and more cross-referencing. The "best leaf" almost always needs adjacent leaves to be useful (signal definition + timing + protocol rule are typically split across 3-5 sibling chunks). Carry these as `KBProfile.CHIP_DESIGN` in `IngestionConfig`; the retrieval layer reads them per-KB.

### 4.5 Cross-document descent

Descent stage is global by default — `node_kind="section"` filter is not joined to a `document_id` filter unless the caller passes `source_filter`. This makes a query like *"find the cache-coherence section across all our SoC specs"* work for free: top-K sections come from a global pool, the reranker sorts.

To prevent one verbose spec from dominating results, add doc-diversity in the candidate pool **before** rerank (not in rerank itself — the reranker stays cross-encoder pure):

```python
RAG_TREE_DESCENT_DOC_DIVERSITY_TOP_PER_DOC = env_int(..., 2)  # cap section hits per document
```

Cross-document **synthesis** (RAPTOR-style cross-doc clusters) remains out of scope for v1; see Appendix A. Cross-document **entity joins** (`AXI_RDATA` mentioned in spec A and testbench B) are KGWeave's job and already wired via Stage 2 KG expansion — independent of tree retrieval.

### 4.3 Merge + rerank

```python
candidates = dedup_by_chunk_id(
    search_results + expanded_chunks + lifted_chunks
)
# Existing reranker handles the rest — just gets more candidates
reranked = self.reranker.rerank(query=processed_query,
                                documents=candidates,
                                top_k=rerank_top_k)
```

The reranker is the great equalizer. We don't need to combine descent and lift scores manually — feed both candidate sets in and let the cross-encoder sort it out. This is the lazy-but-effective move the paper's bidirectional-score-merge dances around.

### 4.4 Bookkeeping

- Add timing entries: `tree_descent`, `tree_lift` to the `TimingPool`.
- Add stage budgets: `RAG_STAGE_BUDGET_TREE_DESCENT_MS`, `RAG_STAGE_BUDGET_TREE_LIFT_MS` (default 200ms each — these are filtered hybrid searches, fast).
- Add response field `tree_retrieval_used: bool` for observability.
- Visual retrieval, KG expansion, confidence routing, fallback retrieval, re-retrieval — **all unchanged**. Tree retrieval slots in before rerank; everything downstream sees a richer candidate pool but the same shape.

---

## 5. Config

Add to `config/settings.py`:

```python
RAG_TREE_RETRIEVAL_ENABLED = env_bool("RAG_TREE_RETRIEVAL_ENABLED", False)
RAG_TREE_DESCENT_TOP_K = env_int("RAG_TREE_DESCENT_TOP_K", 5)
RAG_TREE_DESCENT_LEAVES_PER_SECTION = env_int("RAG_TREE_DESCENT_LEAVES_PER_SECTION", 3)
RAG_TREE_LIFT_SEED_K = env_int("RAG_TREE_LIFT_SEED_K", 5)
RAG_TREE_LIFT_SIBLINGS = env_int("RAG_TREE_LIFT_SIBLINGS", 2)
RAG_STAGE_BUDGET_TREE_DESCENT_MS = env_int("RAG_STAGE_BUDGET_TREE_DESCENT_MS", 200)
RAG_STAGE_BUDGET_TREE_LIFT_MS = env_int("RAG_STAGE_BUDGET_TREE_LIFT_MS", 200)

# Ingest-side
class IngestionConfig:
    enable_tree_retrieval_ingest: bool = False  # mirrors RAG_TREE_RETRIEVAL_ENABLED
```

Validation: ingest-side flag must be true if retrieval flag is true. Add to `config/validate.py` (existing fail-fast pattern).

---

## 6. CLI/UI parity

Per project rule: any user-facing knob added on one surface must be reflected on the other.

- **CLI (`query.py`):** add `--tree / --no-tree` flag mapped to `mode` settings on `RAGRequest`.
- **Web console:** add a "Hierarchical retrieval" toggle in the retrieval-options panel (`server/console/`) wired to the same request field.
- **API (`RAGRequest` schema in `src/retrieval/common/schemas.py`):** add `tree_retrieval: Optional[bool] = None` (None = use config default).

---

## 7. Tests

Co-located under `tests/retrieval/tree/` and `tests/ingest/tree/`:

| Test | What it asserts |
|---|---|
| `test_section_node_emission.py` | Chunking emits exactly the prefixes present in the doc; `tree_level` and `parent_section_id` are stable across re-ingest. |
| `test_section_node_text.py` | Section text = heading + child snippets, deterministic, ≤ chunk_size. |
| `test_descent_filter.py` | Stage 4b `node_kind="section"` filter actually restricts results (no leaf chunks leak into descent hits). |
| `test_lift_siblings.py` | Lift fetches only siblings under the same `parent_section_id`; respects `RAG_TREE_LIFT_SIBLINGS` cap. |
| `test_dedup_merge.py` | `chunk_id` dedup eliminates overlap when descent expansion and Stage 4 hit the same leaf. |
| `test_disabled_path_unchanged.py` | With `RAG_TREE_RETRIEVAL_ENABLED=false`, `rag_chain.run()` produces byte-identical results to current behaviour on a snapshot dataset. **This is the regression gate.** |
| `test_long_doc_qfs.py` (integration) | On a structured PDF (e.g. a policy doc fixture in `documents/`), tree retrieval beats flat retrieval on a held-out QFS query set. Score: `recall@5` of section-spanning answers. |

---

## 8. Rollout

| Phase | Action | Gate |
|---|---|---|
| P0 | Schema migration (add 4 fields, default-fill from existing data) | `migration` test passes; no re-embed needed |
| P1 | Ingest node behind config flag, off by default | Idempotency + re-ingest tests pass |
| P2 | Retrieval changes behind config flag, off by default | Disabled-path-unchanged test passes |
| P3 | Eval pass on policy/legal doc fixtures | recall@5 / nDCG@10 ≥ flat baseline by ≥ 5pp on QFS, ≤ baseline on factoid |
| P4 | Default-on for KBs with `is_long_structured=true` (per-KB toggle) | — |

---

## 9. Open questions

1. **Section-text construction**: heading + first-N-tokens of children, or heading + concatenation of children up to `chunk_size`? The latter overlaps content already in leaves — possibly redundant, possibly helpful for BM25. **Default to heading + concatenation, capped.** Revisit after eval.
2. **Should sections contribute to confidence routing?** A section hit isn't really "evidence" — it's a pointer. Probably exclude `node_kind=="section"` from the citation-coverage signal in `compute_composite_confidence`. Filed as a v0 implementation detail.
3. **Multi-document descent**: descent currently spans documents. For QFS-on-one-doc use cases, callers pass `source_filter`. Cross-doc QFS is a real use case but more naturally answered by RAPTOR-style summary trees (Appendix B). v1 stays single-doc-aware via existing filters.
4. **Cold-start docs without headings** (plain `.txt`, raw OCR): degrade gracefully — `heading_path == []`, so they only ever contribute `node_kind="chunk"` rows, and behave exactly like today. No section nodes get emitted. Verified by `test_section_node_emission.py`.
5. **Numbered-but-untitled headings** (chip-design degenerate case: `"5.3.2"` with no descriptive label, common in old datasheets and auto-extracted PDFs). Section text becomes just the number + child snippets, which embeds weakly. v1 accepts this — child snippets still carry the topic; descent will favor sections with descriptive headings naturally. **Defer LLM heading enrichment to v2** (cheap one-shot summarize-this-section-into-a-heading pass), keep the v1 ingest deterministic. Tracked here as known limitation, not a blocker.

---

## Appendix A — What we're explicitly **not** doing in v1

- LLM summarization of internal nodes (RAPTOR's signature move).
- Cross-document community detection (KGWeave's territory).
- Bidirectional score-fusion math from the TreeRAG paper. We delegate the merge to the existing reranker — simpler, and the reranker has a cross-encoder, which beats any heuristic score combiner.
- Tree-aware reranking. Standard rerank treats every candidate equally. Optional v2.

## Appendix B — RAPTOR follow-up (separate proposal)

If this design lands and we still see "the answer requires synthesis across siblings" failures, the next step is RAPTOR-style summarization layered **on top of** the existing tree:

- For each section node, replace the concatenated-children text with an **LLM-generated abstractive summary**.
- Recurse: cluster sibling sections, summarize, embed.
- Retrieval is unchanged — collapsed-tree, the new summary nodes just live in the same index.

Cost: order-of-magnitude more LLM tokens at ingest. Defer until v1 evaluation says we need it.

---

## R1 — Rerank Fusion (BM25-RRF + heading + anchor)

After the v1 ship, the eval surfaced two systematic miss classes:

1. **Lexical-heavy factoid queries** that the cross-encoder under-ranks
   (e.g. queries naming a specific register field, where the gold leaf
   is *all* about that field but doesn't paraphrase it).
2. **Heading-mention QFS queries** (`q_intr_handling_qfs`,
   `q_fifo_ctrl_qfs`) where the section's heading lexically matches the
   query but the body text doesn't.

R1 layers a thin scoring stage on top of the cross-encoder rerank that
fuses three signals:

```
final = ce_score
      + λ_rrf  · (1 / (k + ce_rank) + 1 / (k + bm25_rank))
      + λ_head · heading_match_score(query, heading_path)
      + λ_anch · 1 / (anchor_k + anchor_rank)        # tree-leaves only
```

Implemented in `src/retrieval/query/nodes/rerank_fusion.py` as four pure
functions (`compute_bm25_rrf`, `heading_match_score`, `anchor_confidence`,
`fuse_scores`). Wired into the rerank stage of `rag_chain.py` after the CE
call. Toggleable via `RAG_RERANK_FUSION_ENABLED`.

### Key implementation choices

- **BM25 plumbing**: a parallel `alpha=0.0` Weaviate hybrid call alongside
  the main hybrid search. Weaviate v4's hybrid query exposes only the
  fused score (and a free-text `explain_score`); a second BM25-only call
  is cheap (BM25 hits the inverted index, no vector compute) and gives a
  clean, deterministic rank list. Failure of the BM25 call degrades
  gracefully to "all `None`", so RRF contributes only the CE side.
- **Heading match** uses substring matching after stopword filtering — no
  external NLP dependency. Fraction of content tokens that appear in any
  heading segment.
- **Anchor confidence** is private metadata (`_anchor_rank` with a leading
  underscore). Tagged in `_run_tree_descent` and `_run_tree_lift` from
  the rank of the section/seed that supplied each leaf. Stage-4 leaves
  carry `None` and contribute zero to this signal.

### Default weights (per `config/settings.py`)

| Key | Default | Notes |
| --- | ---: | --- |
| `RAG_RERANK_FUSION_ENABLED` | `True` | Master switch — `False` = legacy CE-only |
| `RAG_RERANK_RRF_K` | `60` | RRF dampening — canonical default |
| `RAG_RERANK_RRF_LAMBDA` | `1.0` | RRF weight |
| `RAG_RERANK_HEADING_LAMBDA` | `0.15` | Heading-match weight |
| `RAG_RERANK_ANCHOR_LAMBDA` | `0.10` | Anchor-confidence weight (tree-only) |
| `RAG_RERANK_ANCHOR_K` | `10` | Anchor dampening |

### Eval gate (OpenTitan UART fixture, simulator)

| Phase | QFS R@5 (off→on) | QFS nDCG@10 | Factoid R@5 (off→on) | Factoid nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Pre-R1 baseline | 0.304 → 0.304 | 0.402 → 0.494 | 0.000 → 0.000 | 0.000 → 0.105 |
| R1.1 (RRF only) | 0.304 → 0.304 | 0.402 → 0.494 | 0.000 → 0.000 | 0.000 → 0.105 |
| R1.1 + R1.2 | 0.304 → 0.304 | 0.402 → 0.494 | 0.000 → **0.333** | 0.000 → **0.129** |
| R1 default | 0.304 → **0.346** | 0.402 → **0.495** | 0.000 → **0.333** | 0.000 → 0.129 |

R1.1 alone shows zero delta in this lexical simulator because the
CE-stand-in and the BM25 signal both reduce to text-jaccard — RRF can't
reorder identical rank lists. In production with a real cross-encoder
the two signals are independent, and RRF is expected to recover lexical
hits the CE drops. R1.2 lifts factoid recall@5 +33pp on tree-on (the
designed targets), and R1.3 adds a further +4pp on QFS recall@5.

## References

- TreeRAG (Tao et al., ACL Findings 2025): <https://aclanthology.org/2025.findings-acl.20/>
- RAPTOR (Sarthi et al., 2024): <https://arxiv.org/abs/2401.18059>
- RagFlow GraphRAG (for contrast): <https://ragflow.io/blog/ragflow-support-graphrag>
- Reciprocal Rank Fusion (Cormack, Clarke, Buettcher 2009): <https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf>
