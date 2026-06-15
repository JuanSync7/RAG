# Document-Level Routing for RagWeave Retrieval (RAPTOR-lite)

> Status: **Implemented** on branch `feat/raptor-lite-doc-routing` — **gated OFF
> by default** (all `RAG_DOCUMENT_ROUTING_*`, `RAG_DECOMPOSITION_*`, and
> `RAG_INGESTION_*_CARD*` flags default to a no-op; with them unset, retrieval is
> byte-identical to pre-feature behaviour). This document remains the **rationale
> and design intent**; for the as-built architecture, module layout, config knobs,
> how-to-enable, extension, and troubleshooting see the engineering guide:
> [`DOCUMENT_ROUTING_ENGINEERING_GUIDE.md`](DOCUMENT_ROUTING_ENGINEERING_GUIDE.md).
> Author: drafted with Claude during the 2026-06 dev retrieval-quality investigation.

## 1. Problem

RagWeave's retrieval is **flat chunk similarity**: embed the query, find the
nearest chunks across the *entire* corpus, rerank, generate. This fails a
recurring class of query — **broad / procedural / comparison** questions that
should pull together a few *authoritative* documents:

| Query | What the console retrieved | What it *should* have used |
|---|---|---|
| "Compare AMBA AXI4 / AXI5 / CHI…" | glossary lines, project perf docs | ARM AXI spec (IHI0022E) + CHI spec (IHI0098) |
| "Steps before inserting MBIST for a block" | MBIST.pdf, project TRMs | `DFT FLOWS.pdf` "## Infeeds" chunk |
| "Step-by-step guide to set up regression" | MBIST.pdf, svn-book, **Disposal Policy** ("Step by Step Procedure") | Genregress / Gensim / Genbuild / Verification Overview |

In every case **the right content was in the corpus** — it just ranked below
keyword-overlapping tangents. Flat retrieval matches generic words ("step",
"setup", "project", "features") against *any* document containing them.

**Why Claude Sonnet did better on the same corpus:** it worked *from a document
index* — it reasoned at the **document level** first ("regression → Genregress,
Gensim, Genbuild, Verification Overview"), then read those docs. RagWeave has no
document-level routing step.

This is a **recall/ranking** problem, not a coverage gap, and not (any longer) a
parameter-starvation problem — see the related fixes:
- server-side retrieval-depth floor (console can't starve retrieval),
- acronym/term expansion on the retrieval query,
- comparison decomposition ("A compared to B"),
- heading/TOC chunk filter,
- `.xlsx` ingestion (spreadsheets were excluded by the extension allow-list).

Those help, but none of them make retrieval **pick the right documents** for a
broad query. That is what this design adds.

## 2. What tree RAG does today — and the missing layer

The current "tree RAG" (`_run_tree_descent` / `_run_tree_lift` in
`src/retrieval/pipeline/rag_chain.py`) navigates the **section hierarchy *inside*
a document** via `parent_section_id`:

- flat hybrid search produces **seed** chunks,
- **descend**: from a seed's section → its leaf chunks,
- **lift**: sibling chunks under the same parent section,
- `top_per_doc` caps hits per `document_id`.

So tree RAG built the **bottom half** of a hierarchy: `document → section → chunk`.
Its *root is a section within a doc*, and its entry point is **still flat chunk
similarity**. There is **no per-document summary** and **no "which document?"
decision**. If a flat seed lands in the Disposal Policy, tree descent faithfully
explores the Disposal Policy's sections — it never asks whether that's even the
right document.

**Missing top half:** `corpus → document`. That is exactly the layer Sonnet's
doc index provided, and the layer a user intuitively expects "tree RAG" to have.

### Note on the earlier "tree RAG polluted the context"
The pollution was **thin heading/TOC nodes being retrieved *as answer content*** —
they are keyword-dense and short, so the reranker over-ranked them, crowding out
real prose. That was fixed with the substantive-body chunk filter
(`_drop_thin_chunks`). Document routing does **not** reintroduce this: routing
summaries are used **only to select documents** and are **never fed to the LLM**
(see §4 and §6.1).

## 3. Proposed design: 2-stage hierarchical retrieval (RAPTOR-lite)

```
INGEST (once per doc):
    doc ─▶ build a "document card" (title + section headings, optionally an
           LLM summary) ─▶ EMBED ─▶ small doc-card index

QUERY:
    Stage 1 — ROUTE:   embed query ─▶ search the CARD index ─▶ top-N doc IDs
                       (output = a soft document filter, NOT context)

    Stage 2 — RETRIEVE: (a) hybrid + tree-descent over chunks, BOOSTED toward the
                            routed docs
                        (b) PLUS flat hybrid over the whole corpus (unchanged)
                        ─▶ union + dedup ─▶ ONE rerank pass ─▶ top-K ─▶ LLM
```

Stage 1 answers **"where do I look?"** (cheap, summary-level). Stage 2 answers
**"what exactly do I quote?"**. Only Stage 2's chunks become answer context.

## 4. Relationship to RAPTOR

Same family (summary nodes above chunks, hierarchical retrieval), but a
deliberately **simplified and more conservative** variant:

| | RAPTOR (full) | This design (doc routing) |
|---|---|---|
| Structure | Recursive, multi-level summary tree (chunks → cluster summaries → … → root) | **One** summary level: a card **per document** |
| Summaries used as… | **Retrieved context** ("collapsed tree" returns summary nodes to the LLM) | **Routing filter only** — never sent to the LLM |
| Grouping | Semantic clusters *across* documents | Natural document boundaries |
| LLM summarization | Required at every level | **Optional** (title+headings baseline; LLM upgrade) |

We can grow toward full RAPTOR later (cross-doc clusters, multi-level), but
per-document routing is the targeted fix for "find the right guide" and avoids
RAPTOR's main pollution risk (summaries leaking into context).

## 5. Summary-level vs chunk-level: keep BOTH

Summary routing and chunk retrieval each win a *different* class of query:

- **Broad / thematic / procedural / comparison** → summary routing wins (it
  captures what a doc is *about*; abstraction beats keyword overlap).
- **Specific needle facts** → chunk retrieval wins. A value like the reset value
  of `Filters_N_Status` at offset `0x0064` lives in **one chunk** and **no doc
  summary would mention it**. Summary-only routing, if it hard-filtered, would
  *miss the very document* containing the answer.

Therefore **do not replace chunk retrieval — blend**. Run both, union, and let
the reranker arbitrate. This is precisely why RAPTOR's collapsed-tree retrieval
queries leaf chunks and summaries *simultaneously*: a specific chunk can beat a
summary when the query is specific, and a summary can win when it's abstract.

## 6. Key design decisions (the clarifications)

### 6.1 Do tree-RAG-generated nodes go into the general candidate pool?
**No — not as answer content.** Heading/section-tree nodes stay **out** of the
chunk pool that reaches the LLM (the `_drop_thin_chunks` filter remains). Their
text is *repurposed*, not discarded:

- **as raw material for the document card** (title + section headings are a free,
  no-LLM routing summary), and
- **as navigation** for tree descent/lift (assembling coherent local context once
  a *real* prose chunk is seeded).

So the very nodes that used to pollute the context move to where they belong —
the **routing/navigation layer** — and never appear in the answer context.

### 6.2 Rerank each source separately, or union then rerank once?
**Union first, then ONE rerank pass over the deduped pool.** Do **not** rerank
routed-doc chunks separately and then merge.

- Gather candidates from both sources (routed-doc search + flat search),
- **dedup** (a chunk found by both → one entry),
- rerank the **combined** pool **once** against the query,
- take top-K.

**Does a bigger pool dilute?** No — the reranker is precisely what prevents
dilution: it re-scores *every* candidate against the query on one scale and
returns the best K, regardless of how many it was given. More candidates → better
recall; the rerank → precision. The only cost is reranking more items, so we
**bound the union** (e.g. ≤ 40–60 candidates; the GPU reranker handles that
easily). Dilution would only happen if we merged by *raw hybrid scores* (which
are not comparable across the two retrieval modes) — the single rerank pass is
exactly what makes the union safe.

> Routing is expressed as **candidate-pool composition** (ensure routed docs are
> well represented) plus at most a *tiny* tie-break boost — **not** a heavy
> post-rerank score override. Keep the reranker as the final authority on query
> relevance.

### 6.3 Cross-document questions — wouldn't the summary be "weird"?
Two sub-cases:

1. **Cross-doc *comparison*** ("compare AXI4 vs CHI"): handled cleanly by
   **routing to top-N docs (N≈5–8, never 1)** *and* by the existing **comparison
   decomposition**. Decompose "AXI4 vs CHI" → sub-queries "AXI4…", "CHI…", and
   **route each sub-query independently**: "AXI4" routes to the AXI spec, "CHI" to
   the CHI spec, then union. Decomposition makes each routing query single-topic,
   so the summary match is *not* weird — it's sharp. (Decomposition feeds routing.)

2. **Cross-doc *scattered answer*** ("everywhere X is configured", no single doc
   is "about" X): here your intuition is right — summary routing is weak, because
   no doc card strongly matches. This is exactly why we **keep flat chunk
   retrieval in the union** and use a **higher N / low-confidence fallback**. The
   blended flat path carries these queries; routing simply adds nothing (and is
   prevented from *removing* anything by the soft-boost rule).

### 6.5 Decomposition: LLM-primary (glossary-grounded), regex as fallback
Decomposition feeds routing (§6.3), so it must split a comparison into the
*right* entities. There are three tiers — and crucially, **comparisons never need
Tier 3 (deep research, ~24 LLM calls, 5–9 min, disabled for being too slow).**

| Tier | Cost | Role |
|---|---|---|
| 1. Regex heuristic | µs, deterministic | **fallback only** |
| 2. Glossary-grounded LLM | one call (~1–2 s) | **primary** for comparison-intent queries |
| 3. Deep research | recursive, ≤24 calls | **not used for comparisons** |

**Why not regex as primary:** regex on natural language is brittle. A *miss* is
safe (it returns the original query → normal retrieval), but a **mis-split is
actively harmful** — it shreds a query into nonsense sub-queries that retrieve
garbage. Observed mis-splits: aspect text leaking into an entity
(`"AHB for a high-performance SoC"` became an "entity"), and splitting on bare
`and`. Regex also can't map informal phrasing to real entities
(`"hub-based"` → CHI, `"extension-based"` → ACE), so it silently fails to
decompose implicit comparisons (verified: `"…the hub-based approach or the
extension-based one?"` → not split).

**Design:**
- **Gate on comparison-intent** (already detected lexically) so the LLM only fires
  on a minority of queries — cost stays negligible in aggregate.
- **Primary = glossary-grounded LLM**: prompt the model with the shared
  `PROTOCOL_GLOSSARY` (and later the doc-card index) and ask for one standalone
  sub-query per compared entity, as JSON. Grounding is what makes it *safe* — it
  splits into **real corpus entities**, not invented ones, and maps informal
  terms to canonical ones.
- **Fallback = regex**, used only if the LLM is unavailable / times out / returns
  unparseable output (the dev vLLM is shared and contended, so a deterministic
  fallback preserves resilience). If both fail → single-query retrieval (today's
  safe baseline).
- **Validate LLM output**: 2–5 items, each a non-empty short string; otherwise
  treat as failure and fall back.

> Latency note: the dev generation model (qwopus / deepseek_r1) is a *reasoning*
> model, so even a simple decomposition call incurs "thinking" overhead. Mitigate
> with a tight prompt + capped output; longer-term, route decomposition to a
> smaller/faster model if one is available. Gating on comparison-intent keeps the
> blast radius small.

### 6.4 LLM summaries: optional
- **Baseline (no LLM, start here):** card = title + section headings (+ first
  paragraph). Tree RAG already stores heading nodes, so cards can be built from
  data **already in the index — no re-ingest**.
- **Upgrade (RAPTOR-style):** one concise LLM summary per doc at ingest (~hundreds
  of one-time calls). Better semantic routing. (This is what the user's
  `wiki_agent/build_hierarchy.py` already does — card per doc.)

## 7. Safety / no-regression

Routing must be **soft**, so it can only *help* recall, never silently exclude:
- route to **top-N documents, never top-1**;
- **soft boost**, not a hard filter — a strongly-matching chunk from a non-routed
  doc can still make the final top-K (via the flat path in the union);
- **fall back to pure flat retrieval** when routing confidence is low or the
  routed set is small;
- ordinary single-doc / project queries are unaffected — they simply route to
  their own document.

This is the same "scoped, never blunt" philosophy applied at the document layer.

## 8. Reuse of existing machinery
- `SearchFilter(property="source"/"document_id")` in `_do_search` → Stage-2
  document filtering/boosting is already supported.
- `top_per_doc` (per-document cap) already exists.
- Tree descent/lift already assembles intra-doc context.
- Comparison decomposition already exists (`_decompose_comparison`).
- GPU reranker already does the final ordering.

New pieces only: **(a)** document-card construction + a small card index, and
**(b)** the Stage-1 routing query + union/blend wiring in `rag_chain.run`.

## 9. Rollout plan (incremental, measure first)
1. **Prototype, no re-ingest:** build doc cards from existing title+heading nodes,
   embed into a card index, add Stage-1 routing as a **soft boost** blended with
   today's flat retrieval, single rerank over the union. Measure on the
   regression / MBIST / CHI queries **and** on needle + ordinary project queries
   (no-regression check).
2. **If routing needs more punch:** add LLM doc summaries at ingest.
3. **If broad recall still short:** consider multi-level (toward full RAPTOR) or
   cross-doc clustering.

## 10. Validation set (must check both directions)
- **Should improve:** "compare AXI4/CHI", "steps before MBIST", "set up regression
  for a new project" → right authoritative docs surface, grounded answer.
- **Must NOT regress:** a needle fact (register offset/reset value), and a
  single-document project question (e.g. "Vitec PMU power domains") → identical or
  better than today.

## 11. Open questions / risks
- Card quality without LLM (title+headings) may be too thin for some docs →
  measure, upgrade selectively.
- Choosing N (routed docs) and the blend ratio (routed vs flat candidates) —
  tune empirically; expose as config (`RAG_*`), default conservative.
- Spreadsheet (`.xlsx`) docs now ingesting produce huge tabular chunks; their
  cards (title+sheet/headers) need care so routing isn't dominated by giant
  tables.
- Cost: Stage-1 adds one small vector search per query (and per sub-query when
  decomposed) — negligible vs the reranker/generation.
