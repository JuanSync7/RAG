# Document Knowledge Graph — Design Exploration

> **Document type:** Design exploration / notes
> **Context:** Chat discussion covering KG architecture, doc parsing, entity extraction, retrieval patterns
> **Date:** 2026-04-17

---

## 1. The Core Problem

Years of documents accumulated across formats (specs, runbooks, PPTs, Excel, code). No structure connecting them. Goal: consolidate into a queryable knowledge graph that can:

- Answer "which documents discuss retry behavior?"
- Detect contradictions across documents
- Track how thinking evolved over time
- Surface relationships between code and docs

---

## 2. Two Types of Documents

The critical distinction is whether documents have **explicit structure**.

### Layer 1 — Structured (deterministic, no LLM)

New engineering docs authored with a standard format:

```markdown
> **Document type:** Authoritative requirements specification (Layer 3)
> **Upstream:** DOCUMENT_PROCESSING_IMPLEMENTATION.md
> **Downstream:** DOCUMENT_PROCESSING_MODULE_TESTS.md
> **Companion spec:** DOCUMENT_PROCESSING_SPEC.md
> **Source location:** src/ingest/doc_processing/
```

Plus FR-IDs in body (`FR-1213`), section hierarchy (`## 2. Pipeline Overview`), backtick file paths (`` `src/ingest/chunker.py` ``).

Parser extracts these deterministically — no LLM. The format IS the schema.

### Layer 2 — Unstructured (LLM extraction needed)

Old docs, PowerPoints, Excel, code without docstrings. Concepts buried in prose:
*"the retry mechanism limits attempts to 3 before failing the batch..."*

No explicit markup. LLM must extract entities, claims, relationships.

**Real-world corpora are always a mix.** Apply Layer 1 where structure exists, Layer 2 fills the gaps. Sometimes both on the same document (structured metadata + semantic content).

### What determinism covers — and what it doesn't

Structuring a document reduces LLM work. It does not eliminate it.

| What becomes deterministic (no LLM) | What still requires LLM |
|---|---|
| Doc metadata (type, upstream, downstream, companion spec) | Concept entity extraction from body text |
| Explicit requirement IDs (FR-IDs named in the doc) | Implicit relationship inference |
| Doc-to-doc links from blockquote header fields | Embedding generation for similarity-based edges |
| Source paths from backtick references | Whether two entities across docs are the same concept |

The right framing: structured docs give a **deterministic skeleton** (cheap, reliable, always correct). LLM fills in the **semantic tissue** — still necessary, but now scoped to a smaller surface with higher signal because the structural scaffold is already in place. You cannot infer `RetryMechanism IMPLEMENTS embedding_storage._embed_batches` from document structure alone — that inference requires reading and understanding the body text.

---

## 3. Why a Standard Format = Deterministic Parsing

Most tools that build document KGs (GraphRAG, LightRAG, LlamaIndex KG) use LLM extraction because they face unstructured prose. That extraction is probabilistic and expensive.

If documents follow a standard format (FR-IDs, typed frontmatter, explicit cross-references), you get the same information deterministically:

| gitnexus (code KG)         | Doc KG with standard format          |
|----------------------------|--------------------------------------|
| Python/JS grammar          | Your doc format standard             |
| AST parser                 | Frontmatter + FR-ID + heading parser |
| `Function`, `Class` nodes  | `Requirement`, `Section`, `Document` |
| `CALLS`, `IMPORTS` edges   | `IMPLEMENTS`, `REFERENCES`, `CONTAINS` |
| 4-tier type resolution     | Not needed — types are explicit      |
| Confidence scores          | Not needed — edges are binary        |

The format standard is the moat. Teams with unstructured docs pay LLM cost for every extraction. You pay authoring discipline once.

### What the parser extracts (Layer 1)

```python
Document(id="EMBEDDING_PIPELINE_SPEC", type="SPEC", domain="ingestion/embedding")
Section(id="EMBEDDING_PIPELINE_SPEC#FR-1213", heading="Batch Retry Isolation")
Requirement(id="FR-1213", doc="EMBEDDING_PIPELINE_SPEC")
Module(id="src.ingest.embedding.nodes.embedding_storage")

# Edges
CONTAINS:    EMBEDDING_PIPELINE_SPEC → FR-1213
UPSTREAM:    ENGINEERING_GUIDE → IMPLEMENTATION_DOCS
COMPANION:   SPEC_SUMMARY → SPEC
COVERS:      ENGINEERING_GUIDE → src.ingest.embedding.nodes.embedding_storage
DEFINED_IN:  FR-1213 → EMBEDDING_PIPELINE_SPEC
```

### Code KG: the same principle with a closed grammar

For code, the standard format analogy goes further — the grammar is completely closed (finite, unambiguous, language-defined). No custom format standard needed: tree-sitter parses any language with a published grammar and yields a fully deterministic AST.

```
tree-sitter (Python/JS/Go/Rust grammar)
    → Function, Class, Method, Variable, Import nodes
    → CALLS, DEFINES, IMPORTS, CONTAINS edges
    → no LLM at any stage
```

Every node type and edge is explicit in the AST — there are no implicit relationships to infer. The only design work is mapping AST node types to the KG schema (e.g., `function_definition` in the Python grammar → `Function` KG node with `signature`, `docstring` properties). This is a pure implementation task: one mapping per language grammar. The result integrates directly with the doc KG through the same `node_id` bridge — `FR-1213 IMPLEMENTS embedding_storage._embed_batches` becomes a standard KG edge once both sides exist.

### Format inconsistency in existing docs

Real-world finding: ingestion docs use blockquote headers (`> **Upstream:** ...`), retrieval docs use pipe tables, observability docs use different key names. The blockquote format is best — position-stable (before the title), one field per line, trivial regex pattern. Normalize older docs to this format for full graph coverage.

### Document ID namespace

For the KG to work, every document and every requirement needs a stable, unique, human-readable ID. This requires a **centralized namespace** — the same reason libraries use catalog systems.

**Proposed scheme:** `{DOMAIN}-{TYPE}-{SEQ}`

```
INGEST-SPEC-001       → EMBEDDING_PIPELINE_SPEC.md
INGEST-ENG-001        → EMBEDDING_PIPELINE_ENGINEERING_GUIDE.md
INGEST-FR-1001        → first requirement in that spec
KG-SPEC-001           → doc KG spec (once written)
RETRIEVAL-SPEC-001    → RETRIEVAL_QUERY_SPEC.md
```

**Why three parts:**
- `DOMAIN` — coarse subject area (INGEST, KG, RETRIEVAL, OBSERVABILITY, LLM). Maps to docs directory structure.
- `TYPE` — document role (SPEC, DESIGN, ENG, TEST, FR, REQ). FR/REQ are for individual requirements; others for whole documents.
- `SEQ` — centralized monotonic counter per `(DOMAIN, TYPE)` pair.

**Centralized counter matters:** Without it, two people independently write `INGEST-FR-001` for different requirements — namespace collision breaks graph identity. One registry owns the sequence.

**For old corpus:** assign IDs retroactively as docs are structured. High-centrality docs first (the KG itself tells you which docs matter most — use PageRank to prioritize the cataloging queue).

**For new docs:** pass through the ID intake (doc-authoring suite can request the next available ID from the registry before creating a file). Faster and cleaner, but still requires the same registration step.

---

## 4. Architecture — KG + Weaviate, No Text Duplication

### The split

```
KG (graph DB):      nodes + edges + lightweight metadata only
Weaviate:           full text + vector embeddings + BM25 index
Bridge:             node_id field on every Weaviate chunk ↔ weaviate_id on every KG node
```

**KG does NOT store full document text.** Text lives in Weaviate. KG stores structure.

For code nodes specifically: store signature + docstring on the KG node (lightweight, ~50 tokens), full body in Weaviate. This makes path narratives self-contained without a round trip.

```python
# KG node — no full text
Node(
    id="embedding_storage._embed_batches",
    type="Function",
    signature="def _embed_batches(embedder, text_batches, max_retries=3) -> tuple[...]",
    docstring="Embed text batches with per-batch retry isolation.",
    weaviate_id="uuid-xxxx"
)

# Weaviate chunk — no graph structure  
{
    "text": "def _embed_batches(...): ... [full body] ...",
    "vector": [...],
    "node_id": "embedding_storage._embed_batches"
}
```

### Why not store text in KG?

The round trip (graph traversal → fetch text from Weaviate) is a direct key lookup — milliseconds. Not a search. Weaviate is already there for docs (docs need semantic search which graph DBs don't provide natively). So text duplication buys almost nothing at real maintenance cost.

### Code chunking must be AST-aligned

For docs: arbitrary text windows work (prose survives a cut).
For code: arbitrary windows break functions. Chunk boundaries must align with AST nodes (function, class, method). The KG node IS the natural chunk boundary for code — one Weaviate chunk per function body, keyed by the same qualified symbol name used in the KG.

---

## 5. Node Types and Entities

### Everything is a node

This is the foundational mental model. Documents, code files, functions, sections, requirements, abstract concepts, claims, tests — **all of them are nodes**. There is no privileged node type. What makes them meaningful is the edges between them.

```
EMBEDDING_PIPELINE_SPEC.md       (Document node)
        ↓ CONTAINS
      FR-1213                    (Requirement node)
        ↓ IMPLEMENTS
  embedding_storage._embed_batches  (Function node)
        ↓ DEFINED_IN
  embedding_storage.py           (File node)

RetryMechanism                   (Concept node)
        ↑ REFERENCES ── EMBEDDING_PIPELINE_SPEC.md
        ↑ REFERENCES ── Q3_RUNBOOK.pdf
        ↑ IMPLEMENTS ── embedding_storage._embed_batches
        ↑ TESTED_BY  ── test_embedding_storage_batching.py
        ↑ CLAIM["max 3 attempts"] ── Q2_RUNBOOK.pdf
        ↑ CLAIM["max 5 attempts"] ── Q3_SPEC.md  → CONTRADICTS
```

The query "which documents, code files, and tests are all about retry behavior?" becomes:
find `RetryMechanism` node → traverse all incoming edges → get Doc A, Doc B, the `.py` file,
and the test file in a single hop. No text search needed.

The full node vocabulary:

```
Node type     Examples
──────────────────────────────────────────────────────
Document    → EMBEDDING_PIPELINE_SPEC.md, Q3_RUNBOOK.pdf
Section     → EMBEDDING_PIPELINE_SPEC#§4.2
Requirement → FR-1213, REQ-101
Function    → embedding_storage._embed_batches
File        → src/ingest/embedding/nodes/embedding_storage.py
Concept     → RetryMechanism, ExponentialBackoff
Claim       → "retry limit = 3", "batch size = 64"
Test        → test_embedding_storage_batching.py
```

Documents are nodes. The `.py` file is a node. The function inside it is a node. The spec
requirement the function implements is a node. The concept they all share is a node. They
are all first-class citizens connected by typed edges — none more "central" than the others
except by graph structure (centrality, PageRank).

### Entity vs node

- **Entity**: the concept being represented ("retry mechanism" — the idea)
- **Node**: the data structure in the graph that represents an entity (has id, type, properties, edges)

An entity becomes a node at graph build time. People use them interchangeably but they differ.

### Node types (schema) vs instances

```
Types (~10-20):         Instances (thousands):
  Behavior        →       RetryMechanism, ExponentialBackoff
  Component       →       EmbeddingPipeline, VectorStore
  Requirement     →       FR-1213, FR-201, REQ-101
  Configuration   →       BatchSize, RetryLimit
  Document        →       EMBEDDING_SPEC.md, Q3_RUNBOOK.pdf
  Section         →       EMBEDDING_SPEC#§4.2
  Claim           →       "retry limit = 3", "batch size = 64"
  Module          →       src.ingest.embedding.nodes.embedding_storage
```

Small schema, unbounded instances.

### Multiple documents → one entity node

Documents don't "live inside" a node. They point TO it via typed edges. The entity node is the convergence point:

```
RetryMechanism (node)
    ↑ REFERENCES ── Q2_RUNBOOK.pdf
    ↑ REFERENCES ── Q3_SPEC.md
    ↑ IMPLEMENTS ── embedding_storage._embed_batches
    ↑ TESTED_BY  ── test_retry.py
    ↑ CLAIM["max 3 attempts"] ── Q2_RUNBOOK.pdf
    ↑ CLAIM["max 5 attempts"] ── Q3_SPEC.md   → CONTRADICTS edge
```

Multiple documents having edges to the same entity node IS the cross-document relationship. The graph connects documents written years apart, by different people, with different vocabulary — all through shared entity nodes.

---

## 6. Retrieval Patterns

### Parallel is not universal

Running KG and vector in parallel is a safe default but not universally correct. The right entry point depends on query type:

| Query type | Example | Right approach |
|---|---|---|
| **Named entity** (anchor present) | "tell me about sync()" | Parallel — entity is shared anchor for both modalities |
| **Pure semantic** (no anchor) | "explain retry behavior" | Vector first → graph walk from hit |
| **Pure structural** | "what calls sync()?" | Graph only |
| **Cross-modal** | "which spec covers sync()?" | Graph first → vector for content |
| **Broad/overview** | "explain the embedding pipeline" | Community summary retrieval |

**Named entity queries are the sweet spot for parallel.** When the query contains a known entity, both vector (what it says) and graph (how it relates) answer different facets simultaneously. The entity name is the join key.

### Full query flow with both systems

```
Query: "tell me about sync()"
          │
          ├─ Weaviate hybrid (vector+BM25) → top chunks with node_ids [A, B, C]
          │
          ├─ KG: walk from nodes A, B, C → neighbor node_ids [D, E, F, G]
          │      (no text fetched — graph traversal only)
          │
          ├─ Weaviate: WHERE node_id IN [A,B,C,D,E,F,G] + vector+BM25 ranked
          │            (subgraph candidates compete with initial results)
          │
          └─ Reranker → final ranked list → LLM
```

KG adds candidates (D, E, F, G) that initial vector search missed. Those compete in the same Weaviate + reranker pipeline.

### How KG contributes to the reranker

Graph results and vector results have incompatible score types (cosine similarity vs hop distance). Options:

1. **Convert to text, cross-encode uniformly** — graph nodes → fetch text → (query, text) pairs for cross-encoder. Simple but loses structural signal.
2. **Graph as boost signal** — vector candidates boosted by graph connectivity to anchor. 1-hop IMPLEMENTS = ×1.5, 1-hop COMPANION = ×1.2. Keeps structural signal without competing score types.
3. **Context injection** — inject top graph edge into chunk text before cross-encoder: `"[FR-1213 | IMPLEMENTS: embedding_storage_node] Failed batches are excluded via success_mask..."`. Cross-encoder uses structural context without needing a separate score.
4. **RRF** — rank graph results by `edge_type_weight / hop_distance × edge_confidence`, fuse ranks with vector via Reciprocal Rank Fusion.

Recommended: Option 2 + 3 combined. Graph connectivity as boost, top edge injected as context.

### What graph retrieval actually returns

Unlike vector (ranked text chunks), graph retrieval returns different shapes depending on query:

**Single-hop** — direct neighbors:
```
sync() → CALLS → flush(), acquire_lock()
sync() → IMPLEMENTS → FR-201, FR-202
sync() → COVERED_BY → ENGINEERING_GUIDE#§3.1
```

**Multi-hop** — answering relational questions:
```
"is sync() tested?"
sync() → IMPLEMENTS → FR-201 → TESTED_BY → test_sync_flush, test_sync_idempotent
Answer: YES, and exactly where. Vector can't do this.
```

**Path narrative** — the path IS the answer:
```
sync() → FR-1213 → test_batch_retry
"sync() implements FR-1213 which is tested by test_batch_retry"
```

**Subgraph context** — connected neighborhood for broad queries:
```
EMBEDDING_PIPELINE_SPEC
  ├── FR-1210...FR-1214 (requirements)
  │     └── IMPLEMENTED_BY → embedding_storage.py functions
  │     └── TESTED_BY      → test_embedding_storage_batching.py
  ├── COMPANION → ENGINEERING_GUIDE
  └── UPSTREAM  → DOCUMENT_PROCESSING_SPEC
```

**Impact analysis** — unique to graph:
```
"what breaks if I change sync()'s return type?"
sync() ← CALLED_BY ← orchestrate_batch() ← CALLED_BY ← run_pipeline()
sync() ← REFERENCED_BY ← ENGINEERING_GUIDE#§3.1
sync() ← TESTED_BY ← test_sync_flush, test_sync_idempotent
```

### Entry point problem

Subgraph context and path narratives retain all relational information — but you still need to find the starting node. For unanchored queries ("explain retry behavior"), you can't traverse without an anchor. That's what Weaviate hybrid search provides. For technical domains with precise terminology, BM25 is often sufficient as the entry point mechanism.

---

## 7. Graph Properties

### Communities

Clusters of densely-connected nodes — detected by Leiden/Louvain algorithms. Run nightly when usage is low; recompute after significant graph changes.

A community in a combined doc+code graph crosses doc/code boundaries:
```
Community: "Embedding Batch Processing"
  FR-1210–FR-1214, _form_batches(), _embed_batches(),
  EMBEDDING_SPEC#§4, test_embedding_storage_batching.py
```

**Use for:** broad/overview queries ("explain the embedding pipeline architecture") — return pre-computed community summary instead of 50 scattered chunks. This is where pure vector search fails and communities win.

### Centrality and PageRank

- **Centrality** (degree): how many edges a node has. High-centrality nodes are hubs — `INGESTION_PLATFORM_SPEC` referenced by 8 other specs has high centrality.
- **PageRank**: a node is important if important nodes point to it. Identifies authoritative documents.

**Use for:** ranking tie-breaking. When multiple candidates tie in relevance, prefer higher-centrality/PageRank nodes. Also for identifying load-bearing docs — changing a high-centrality spec has large blast radius.

### Shortest path

Graph distance between two nodes. Answers "how does X relate to Y?" directly. Also useful for query routing: anchor nodes far apart → query spans subsystems → return community summaries rather than local neighborhoods.

---

## 8. Old Document Corpus — Entity Extraction and Resolution

### The pipeline

```
Raw documents (all types, all ages)
        ↓
[Extraction — per doc, parallelizable]
  Layer 1: format parser (structured docs)
  Layer 2: LLM with domain taxonomy (unstructured)
        ↓
Raw entity strings per doc {name, type, context_sentence}
        ↓
[Embed all entities] → vectors (semantic values)
        ↓
[Cluster by embedding similarity] → groups of semantically similar entities
  (NOT in the graph — pure ML in vector space)
        ↓
[LLM resolution within each cluster] → SAME / DIFFERENT / UNCERTAIN
  Cluster of 15 entities → 105 pairs (not 1.25B all-pairs)
        ↓
[Human review queue] → UNCERTAIN resolutions
        ↓
[Build graph] → canonical entity nodes + edges
        ↓
[Post-ingestion passes]
  → CONTRADICTS detection (conflicting claims on same entity)
  → SUPERSEDES candidates (coverage overlap + temporal ordering)
  → Human confirms candidates
```

### Why all-pairs LLM is impossible

```
1,000 docs × 50 entities = 50,000 entities
All-pairs: 50,000 × 49,999 / 2 = 1.25 billion LLM calls
```

Clustering first reduces this to thousands of within-cluster pairs.

### Why context sentence matters for embeddings

Short entity names embed poorly:
```
"batch" (processing context) vs "batch" (script context) → same embedding, wrong cluster
```

Always embed with context:
```
"retry mechanism [context: limits failed embedding attempts before excluding batch]"
→ disambiguated embedding → correct cluster
```

The `context_sentence` field in extraction output is not optional — it's load-bearing for resolution quality.

### Similarity threshold design

Embeddings give distance, not SAME/DIFFERENT:

```
"retry mechanism" vs "batch retry logic" → cosine 0.97 → same cluster → LLM: SAME → merge
"cuboid" vs "rectangle"                  → cosine 0.89 → same cluster → LLM: DIFFERENT → two nodes + RELATED_TO
"circuit breaker" vs "retry mechanism"   → cosine 0.61 → different clusters → no LLM needed
```

Cluster aggressively (low threshold ~0.70) to avoid missing pairs. LLM decides precisely within clusters. False positives in clustering (same cluster, LLM says DIFFERENT) are cheap. False negatives (same concept, different clusters) are expensive — entity fragmentation.

### Entity fragmentation

If the same concept becomes two canonical nodes, cross-document connections break. Docs pointing to node A never connect to docs pointing to node B.

**Prevention:** define entity type taxonomy upfront — constrains extraction so similar concepts land in the same typed cluster.

**Detection/repair:** periodic merge pass — re-embed all canonical nodes, find high-similarity pairs, LLM confirms merge, surface to human review.

**Soft links:** `MAY_BE_SAME_AS` edge for close-but-unconfirmed pairs.

### Corpus migration — the library analogy

Think of the old document corpus as a library where books were shelved without a catalog system. The goal is to make every book findable and cross-referenced. The library analogy is precise:

| Library | Document corpus |
|---|---|
| Unindexed book stacks | Old freeform docs (no IDs, no structured format) |
| Cataloging (assigning Dewey/ISBN) | Structuring docs into standard format + assigning DOMAIN-TYPE-SEQ IDs |
| Indexing (subject headings, cross-refs) | Entity extraction + node creation + relationship edges |
| New acquisitions | New docs authored in structured format from day 1 |

**Key insight:** new books still require indexing. Structured intake just means the catalog card is pre-filled (doc metadata, explicit FR-IDs, doc-to-doc links). The librarian still reads the book to extract subject headings — that's entity extraction, and it still requires LLM. Structuring the doc reduces the surface and improves quality; it does not eliminate the indexing step.

**Three-layer migration path:**

```
Layer 1 — Quick KG population (do immediately):
  All old docs → LLM entity extraction → KG nodes (probabilistic)
  Goal: get the graph populated quickly, even imperfectly

Layer 2 — Doc structuring (ongoing, prioritized by centrality):
  High-centrality old docs → LLM-assisted draft → human review → structured format + IDs
  Use PageRank from Layer 1 KG to pick which docs to structure first
  Goal: replace probabilistic Layer 1 entries with deterministic ones progressively

Layer 3 — Deterministic KG (from structured docs):
  Structured docs → parser (no LLM for skeleton) + LLM (semantic tissue only)
  Goal: stable, typed, fully canonical graph entries
```

Layer 1 populates the graph fast. Layer 2 + 3 replace Layer 1 entries as docs get structured — the transition is progressive, not a big-bang migration. At any point the graph is queryable; it just improves in precision over time.

**Prioritization signal:** after Layer 1 runs, run PageRank. The top-50 highest-centrality document nodes are your first structuring targets — they have the most incoming edges, meaning the most other content references them. Structuring them first maximizes the precision improvement per unit of work.

### Incremental ingestion (ongoing, post-bulk)

After the bulk corpus is processed, new documents are cheap:
```
New doc → extract entities → resolve against existing canonical nodes
         (ANN search against ~50k existing, not fresh all-pairs)
         → LLM confirmation for close matches only
         → add to graph
```

The entity space exists — resolution is a lookup, not a rebuild.

### How documents relate through the graph

Doc A (Monday) extracts "RetryMechanism" → creates canonical node
Doc B (Friday) extracts "batch retry logic" → resolves to same canonical node
→ both docs now connected through shared node, even though neither knew about the other

**The entity node is the consolidation point.** Documents ingested months apart, written by different people, using different vocabulary, all converge through shared entity nodes.

---

## 9. CONTRADICTS and SUPERSEDES Detection

### CONTRADICTS

Detected post-ingestion by comparing claims on the same entity:

| Claim type | Detection | Coverage |
|---|---|---|
| Numeric/exact values | Deterministic comparison | ~99% |
| Boolean flags | Deterministic | ~99% |
| Semantic prose | LLM comparison | ~80% |
| Implicit contradictions | Requires human | Low |

Run exact-value pass automatically. Run semantic LLM pass periodically.

### SUPERSEDES

Document-level relationship detected by:
1. Temporal: Doc B created after Doc A (from `created_at` node property)
2. Coverage overlap: Doc B's canonical entities overlap significantly with Doc A's
3. Optional explicit signal: "this replaces..." in text

→ Candidate flagged automatically → human confirms.

**Never auto-apply SUPERSEDES.** The contradiction might be intentional (different environments, versions).

### Source docs — don't edit

Old docs are historical artifacts. Don't restructure them. The graph IS the refactored view.
One narrow exception: confirmed SUPERSEDES can get a one-line annotation added to the old doc:
```markdown
> ⚠ Superseded by: EMBEDDING_PIPELINE_SPEC_V2.md (2024-03-15)
```

---

## 10. Closing the Extraction Gap — Approaching 99%

No single LLM pass reaches 99%. Strategies to get close:

**Define a domain taxonomy before extraction:**
```
Types: system_component, behavior, configuration_parameter,
       data_model, api_contract, error_condition, constraint
```

**Multiple passes with different lenses:**
```
Pass 1: system components and behaviors
Pass 2: configuration parameters and constraints
Pass 3: error conditions and failure modes
Pass 4: "given this doc and these entities, what did we miss?"
Union the results.
```

**Document-type-specific prompts:**

| Doc type | Strategy |
|---|---|
| PowerPoint | Slide titles + bullet points |
| Excel | Table headers → entities; cell values → claims |
| Code (no docstrings) | AST: function names + signatures; LLM: comments |
| Code (with @summary) | Layer 1 — already structured |
| Prose | LLM with taxonomy prompt |

**Structured output:** JSON schema enforces consistent entity format, reduces hallucination.

**Human sampling:** review 5% of docs, measure extraction quality, refine prompts iteratively.

### Bootstrap — cold start loop

The taxonomy-first vs. extract-first dilemma: without a taxonomy, extraction quality is poor. Without extraction results, you don't know what types to put in the taxonomy.

**Practical approach — seed taxonomy + graded iteration:**

```
Step 1: Domain experts define ~10-15 seed entity types
        (system_component, behavior, configuration_parameter,
         data_model, api_contract, error_condition, constraint)
        These don't need to be complete — just enough to guide the first pass.

Step 2: Run extraction on a small representative sample (~10-20 docs,
        covering different doc types: spec, runbook, design, meeting note)

Step 3: LLM grades its own output:
        "For each extracted entity, does it fit a known type? Or is it a new type?"
        → Entities that don't fit any type form a "residual" pile

Step 4: Inspect residual pile → identify missing types → extend taxonomy

Step 5: Re-run extraction on same sample with updated taxonomy
        Repeat until residual pile is small and entity types feel stable

Step 6: Full corpus extraction with final taxonomy
```

The grading loop (Step 3) is the key mechanism — LLM judges its own extraction quality against the taxonomy. The residual pile is the signal: a large residual means the taxonomy is incomplete, not that extraction is failing.

**Confidence signal for the taxonomy:** when the same entity type appears consistently across unrelated docs (design doc + runbook + spec all produce `system_component` entities), the type is load-bearing. Types that only appear in one doc type are candidates for merging or dropping.

### Evaluation — open questions

No settled framework yet. The core challenge: ground truth is expensive to create (human-labeled entity sets), and domain-specific metrics don't transfer across corpora.

**Metrics worth tracking:**
- **Entity extraction recall** — of known entities in a human-labeled sample, how many were found?
- **Edge precision** — of extracted edges, how many are correct (not hallucinated relationships)?
- **Contradiction detection accuracy** — of known contradictions in a seeded test set, how many flagged?
- **Entity resolution quality** — merge precision (same-entity pairs correctly merged) + fragmentation rate (same-entity pairs missed)
- **Coverage** — % of documents with at least one canonical node edge (rough health indicator)

**Open questions:**
- What test corpus is representative enough to measure against without being too expensive to label?
- How do you measure implicit relationship recall (you don't know what you missed)?
- Is entity fragmentation rate measurable without full ground truth?

---

## 11. Doc Freshness and Drift

### The drift hierarchy

| Layer | Drift posture | Sync strategy |
|---|---|---|
| Scope, Architecture | Rarely drift (snapshot decisions) | Annotate when superseded |
| Spec (authoritative) | High cost if stale — it's the contract | Hard-sync |
| Spec Summary | Derivative of spec | Regenerate from spec |
| Design, Impl Plan | Snapshot of pre-build intent | Freeze; don't back-edit |
| Engineering Guide | Drifts with every code change | Hard-sync, bias toward auto-generation |
| Test Docs | Drift with code | Enforce via FR coverage |

### Freshness ledger

Add to each authoritative doc:
```yaml
last_synced_commit: <sha>
covers_paths: [src/ingestion/chunker/**, src/ingestion/schemas.py]
```

CI check: if any `covers_paths` file changed since `last_synced_commit` and the doc wasn't updated → fail (or prompt).

### /commit gate

When commit touches `src/X/**` without touching `docs/X/**`:
- Prompt: update docs now / defer with TODO-DRIFT marker / confirm doc-invisible change
- Three-way choice, not a block — blocks get bypassed, choices get recorded

### Auto-generation for engineering guides

Engineering guide module sections are largely derivable from `@summary` blocks + public API signatures. Auto-regenerate those sections from source. Keep architecture decisions and data flow narrative as human-authored. Cuts drift surface by ~60%.

---

## 12. Tool Landscape

### No gitnexus equivalent for docs

| Tool | Approach | Key gap |
|---|---|---|
| Microsoft GraphRAG | LLM entity extraction + community detection | Untyped edges, no structured doc support, no MCP |
| LightRAG | LLM extraction + hybrid retrieval | Same — probabilistic, no schema |
| LlamaIndex KG | LLM (subject, predicate, object) triples | Free-form predicates, RAG-flavored queries |
| Doorstop | Manual YAML requirements with UID links | No extraction, no graph query |
| Obsidian/Logseq | Manual wikilinks | No typed edges, no query language |

**Gap:** no tool does automatic, deterministic, typed graph extraction from engineering documentation. The closest approaches either use LLM (probabilistic) or require manual authoring.

Your standard doc format closes this gap — deterministic extraction from structure, no LLM needed for the load-bearing edges.

### Gitnexus constraints

- **License:** PolyForm-Noncommercial-1.0.0 — personal use only
- **Never commit** any gitnexus reference into tracked files (no Makefile targets, no CLAUDE.md mentions, no CI integration)
- **Use for:** personal exploration, impact analysis before refactors, MCP-powered doc authoring in-session
- **Do not use for:** CI checks, commit gates, anything requiring the graph in fresh clones

---

## 13. Skill Pipeline vs This Architecture

For reference, how the existing skills relate to this design:

| Skill | Purpose | Where this design fits |
|---|---|---|
| `feature-dev` | End-to-end feature ship | Code changes → freshness ledger check |
| `/brainstorm` | Deliberated design → spec | Spec becomes a graph node (Layer 1) |
| `doc-authoring` suite | 7-layer doc governance | All produced docs are parseable Layer 1 |
| `/sync-docs` (proposed) | Post-feature doc update | Not worth building standalone; use freshness ledger + /commit gate |

The doc KG described here is a natural extension of the `doc-authoring` governance layer — the graph is the queryable representation of what the skill suite produces.
