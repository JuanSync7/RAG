# RAG Document Embedding Pipeline — Architecture Summary & Discussion Guide

**Companion document to:** RAG_embedding_pipeline_arch.md (v1.0.0)
**Purpose:** Implementation-level summary for developer review sessions, plus structured discussion agenda.
**See also:** RAG_embedding_pipeline_spec_summary.md for the requirements-level summary (scope, security, deployment, phasing).

---

## 1. What This System Does (One Paragraph)

The AION RAG Document Embedding Pipeline transforms engineering documents (PDFs, DOCX, PPTX, XLSX, Markdown, HTML, RST, plain text) into semantically searchable vector embeddings stored in Weaviate, with an optional knowledge graph capturing relationships between documents, concepts, and specifications. Each document passes through a 13-node LangGraph processing pipeline that extracts structure, describes figures via VLM, cleans text, resolves domain abbreviations, optionally refactors content for self-containedness, creates semantically coherent chunks, enriches them with metadata, extracts cross-references and knowledge graph triples, validates quality, and stores the final embeddings with full provenance. The system is designed for mission-critical ASIC engineering environments where incorrect retrieval could propagate into design errors.

---

## 2. Architecture Wireframe

### 2.1 End-to-End System Context

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            EXTERNAL INPUTS                                  │
│                                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │   PDFs   │  │   DOCX   │  │   PPTX   │  │   XLSX   │  │ MD/HTML/ │       │
│  │          │  │          │  │          │  │          │  │ RST/TXT  │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       └─────────────┴─────────────┼─────────────┴─────────────┘             │
│                                   │                                         │
│                                   ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                     CLI / Programmatic API                             │ │
│  │                          (main.py)                                     │ │
│  │  python main.py doc.pdf --domain dft --review-tier partially_reviewed  │ │
│  └────────────────────────────────┬───────────────────────────────────────┘ │
└───────────────────────────────────┼─────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│                    EMBEDDING PIPELINE (LangGraph DAG)                      │
│                                                                            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           │
│  │ Node 1  │─>│ Node 2  │─>│ Node 3  │─>│ Node 4  │─>│ Node 5  │           │
│  │ Ingest  │  │Structure│  │   VLM   │  │ Clean   │  │Refactor │           │
│  │ + Hash  │  │ Detect  │  │ Figures │  │  Text   │  │(agentic)│           │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘           │
│       │            │         optional        │         optional            │
│       │            │            │            │            │                │
│       ▼            ▼            ▼            ▼            ▼                │
│  ┌──────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
│  │ Node 6   │─>│ Node 7  │─>│ Node 8  │─>│ Node 9  │─>│ Node 10 │          │
│  │ Chunk    │  │ Enrich  │  │Metadata │  │ X-Refs  │  │   KG    │          │
│  │(semantic)│  │+boundary│  │Keywords │  │ Extract │  │ Triples │          │
│  └────┬─────┘  └─────────┘  └─────────┘  └────┬────┘  └────┬────┘          │
│       │                                   optional     optional            │
│       │                                      │            │                │
│       ▼                                      ▼            ▼                │
│  ┌─────────┐                             ┌─────────┐  ┌─────────┐          │
│  │ Node 11 │────────────────────────────>│ Node 12 │─>│ Node 13 │          │
│  │ Quality │                             │ Embed & │  │KG Store │          │
│  │Validate │                             │ Store   │  │         │          │
│  └─────────┘                             └────┬────┘  └────┬────┘          │
│                                               │            │               │
└───────────────────────────────────────────────┼────────────┼───────────────┘
                                                │            │
                        ┌───────────────────────┘            │
                        │                                    │
                        ▼                                    ▼
┌───────────────────────────────────┐  ┌──────────────────────────────────────┐
│         WEAVIATE VECTOR STORE     │  │        KNOWLEDGE GRAPH STORE         │
│                                   │  │                                      │
│  ┌─────────────────────────────┐  │  │  v1: Weaviate cross-references       │
│  │ engineering_documents       │  │  │  v2: Neo4j (optional)                │
│  │                             │  │  │                                      │
│  │  HNSW Index (vectors)       │  │  │  ┌────────┐     ┌────────┐           │
│  │  BM25 Index (keywords)      │  │  │  │Document│────>│Concept │           │
│  │  32 metadata properties     │  │  │  └────────┘     └────┬───┘           │
│  │  Review tier filtering      │  │  │       │              │               │
│  └─────────────────────────────┘  │  │       ▼              ▼               │
│                                   │  │  ┌────────┐     ┌────────┐           │
│  Hybrid Search (vector + BM25)    │  │  │ Entity │────>│SpecVal │           │
│  Retrieval-time tier filtering    │  │  └────────┘     └────────┘           │
└───────────────┬───────────────────┘  └──────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│        RETRIEVAL LAYER            │
│    (downstream, not this system)  │
│                                   │
│  Query → Embed → Hybrid Search    │
│  → Rerank → Answer Generation     │
└───────────────────────────────────┘
```

### 2.2 Configuration & Support Systems

```
┌──────────────────────────────────────────────────────────────────────┐
│                    CONFIGURATION LAYER                               │
│                                                                      │
│  ┌───────────────────┐  ┌─────────────────┐  ┌────────────────────┐  │
│  │  PipelineConfig   │  │ model_registry  │  │ domain_vocabulary  │  │
│  │  (13 sub-configs) │  │    .yaml        │  │      .yaml         │  │
│  │                   │  │                 │  │                    │  │
│  │  LLM provider     │  │ BGE-large: 1024 │  │ DFT: Design for    │  │
│  │  Embedding model  │  │ E5-large: 1024  │  │   Testability      │  │
│  │  Weaviate URL     │  │ Jina-v2: 768    │  │ STA: Static Timing │  │
│  │  Chunk sizes      │  │ ...             │  │   Analysis         │  │
│  │  Skip flags       │  │                 │  │ CDR: [ambiguous]   │  │
│  │  Review defaults  │  │ Cross-validated │  │ ...                │  │
│  │  KG settings      │  │ at startup      │  │                    │  │
│  └────────┬──────────┘  └───────┬─────────┘  └─────────┬──────────┘  │
│           │                     │                      │             │
│           └─────────────────────┼──────────────────────┘             │
│                                 │                                    │
│                                 ▼                                    │
│                    ┌───────────────────────┐                         │
│                    │  config_validator.py  │                         │
│                    │  (startup checks)     │                         │
│                    │                       │                         │
│                    │  dim matches model?   │                         │
│                    │  prefix matches model?│                         │
│                    │  chunk < model limit? │                         │
│                    │  contradictory flags? │                         │
│                    └───────────────────────┘                         │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                    QUALITY & EVALUATION LAYER                        │
│                                                                      │
│  ┌───────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │ verify_pipeline   │  │ eval_dataset     │  │ evaluate_retrieval│  │
│  │    .py            │  │    .json         │  │       .py         │  │
│  │                   │  │                  │  │                   │  │
│  │ Unit tests        │  │ 50-100 queries   │  │ Recall@5, @10     │  │
│  │ Integration tests │  │ Ground truth     │  │ MRR               │  │
│  │ Node contracts    │  │ Per-domain       │  │ Precision@10      │  │
│  └───────────────────┘  └──────────────────┘  └───────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.3 Data Flow Through a Single Chunk

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  LIFECYCLE OF A CHUNK: from raw document to Weaviate object                  │
│                                                                              │
│  Raw PDF page text                                                           │
│  "The DFT scan chain operates at 100MHz. See Section 3.2 for timing."        │
│       │                                                                      │
│       ▼ [Node 2: Structure Detection]                                        │
│  Section: "Power Domains > DFT > Scan Chain Timing"                          │
│  Content type: TEXT                                                          │
│       │                                                                      │
│       ▼ [Node 4: Text Cleaning]                                              │
│  Boilerplate removed, whitespace normalised                                  │
│       │                                                                      │
│       ▼ [Node 5: Refactoring]                                                │
│  "The DFT (Design for Testability) scan chain operates at 100MHz clock       │
│   frequency. The timing constraints for scan chain operation are defined     │
│   in Section 3.2 of the DFT Scan Chain Specification (DOC-042)."             │
│       │                                                                      │
│       ▼ [Node 6: Chunking]                                                   │
│  chunk_id: "a1b2c3d4-..."  (deterministic from doc_id + index + hash)        │
│  chunk_index: 5                                                              │
│  previous_chunk_id: "..."   next_chunk_id: "..."                             │
│  content_type: TEXT                                                          │
│  chunking_method: "llm_semantic"                                             │
│       │                                                                      │
│       ▼ [Node 7: Enrichment]                                                 │
│  context_header: "[Context: dft | specification | DFT Guide | Power > DFT]"  │
│  enriched_content: "[Previous context: ...2 sentences...]\n{chunk content}"  │
│       │                                                                      │
│       ▼ [Node 8: Metadata]                                                   │
│  keywords: ["scan chain", "100MHz", "timing constraints"] (validated)        │
│  entities: ["DFT Scan Chain Specification", "DOC-042"]                       │
│       │                                                                      │
│       ▼ [Node 11: Quality]                                                   │
│  quality_score: 0.78 (tech terms: +0.08, numbers: +0.06, good length: +0.0)  │
│       │                                                                      │
│       ▼ [Node 12: Embed & Store]                                             │
│  vector: [0.023, -0.145, 0.089, ...] (1024 dimensions, BGE-large)            │
│       │                                                                      │
│       ▼ WEAVIATE OBJECT                                                      │
│  ┌──────────────────────────────────────────────────────────────────┐        │
│  │  uuid: a1b2c3d4-...                                              │        │
│  │  content: "The DFT (Design for Testability) scan chain..."       │        │
│  │  enriched_content: "[Previous context: ...]..."                  │        │
│  │  context_header: "[Context: dft | specification | ...]"          │        │
│  │  keywords: ["scan chain", "100MHz", "timing constraints"]        │        │
│  │  entities: ["DFT Scan Chain Specification", "DOC-042"]           │        │
│  │  section_path: "Power Domains > DFT > Scan Chain Timing"         │        │
│  │  document_domain: "dft"                                          │        │
│  │  review_tier: "partially_reviewed"                               │        │
│  │  quality_score: 0.78                                             │        │
│  │  vector: [0.023, -0.145, 0.089, ...]                             │        │
│  │  ... (32 total properties)                                       │        │
│  └──────────────────────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Re-ingestion Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  RE-INGESTION DECISION TREE                                     │
│                                                                 │
│  Document arrives ──> Compute SHA-256 hash                      │
│                       │                                         │
│                       ▼                                         │
│                  Query Weaviate: "Does document_id exist?"      │
│                       │                                         │
│               ┌───────┴──────────┐                              │
│               │                  │                              │
│          [NO: new doc]    [YES: existing]                       │
│               │                  │                              │
│               ▼                  ▼                              │
│          Normal pipeline    Compare content_hash                │
│                              │                                  │
│                     ┌────────┴────────┐                         │
│                     │                 │                         │
│              [SAME hash]       [DIFFERENT hash]                 │
│                     │                 │                         │
│                     ▼                 ▼                         │
│              strategy?         is_reingestion = True            │
│              │                 Auto-demote review tier          │
│     ┌────────┴──────┐         Process through pipeline          │
│     │               │                 │                         │
│  [skip_unchanged] [delete_*]          ▼                         │
│     │               │         Node 12: Delete old chunks        │
│     ▼               ▼         ──> If 0 new chunks + errors:     │
│   SKIP          Continue             ABORT (preserve old data)  │
│   (no-op)       to pipeline   ──> If cleanup fails:             │
│                                      HALT (no corrupt state)    │
│                               ──> If cleanup succeeds:          │
│                                      INSERT new chunks          │
│                                      Node 13: Clean + insert KG │
└─────────────────────────────────────────────────────────────────┘
```

### 2.5 Review Tier System

```
┌──────────────────────────────────────────────────────────────────────────┐
│  REVIEW TIER LIFECYCLE & RETRIEVAL VISIBILITY                            │
│                                                                          │
│  ┌──────────────────┐    peer review    ┌────────────────────┐           │
│  │  SELF_REVIEWED   │ ──────────────>   │ PARTIALLY_REVIEWED │           │
│  │  (Tier 3)        │ <──────────────   │ (Tier 2)           │           │
│  │                  │   major revision  │                    │           │
│  │  Utility scripts │                   │  Draft specs       │           │
│  │  Personal notes  │                   │  WIP processes     │           │
│  │  Tribal knowledge│                   │  Active projects   │           │
│  └──────────────────┘                   └─────────┬──────────┘           │
│                                                   │                      │
│                                          domain lead sign-off            │
│                                                   │                      │
│                                                   ▼                      │
│                                        ┌───────────────────┐             │
│                                        │  FULLY_REVIEWED   │             │
│                                        │  (Tier 1)         │             │
│                                        │                   │             │
│                                        │  Approved specs   │             │
│                                        │  Signed-off docs  │             │
│                                        │  Authoritative    │             │
│                                        └─────────┬─────────┘             │
│                                                  │                       │
│                                          doc re-ingested                 │
│                                          with changes                    │
│                                                  │                       │
│                                                  ▼                       │
│                                          Auto-demote to                  │
│                                          PARTIALLY_REVIEWED              │
│                                                                          │
│  RETRIEVAL SEARCH SPACES:                                                │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │ DEFAULT:  [Tier 1 only]           ──  Authoritative results      │    │
│  │ EXPANDED: [Tier 1 + Tier 2]       ──  + Draft/WIP results        │    │
│  │ FULL:     [Tier 1 + Tier 2 + Tier 3] ── All knowledge            │    │
│  └──────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Key Design Decisions (Quick Reference)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Orchestration | LangGraph StateGraph | Typed state, conditional routing, compilable DAG |
| Embedding approach | BYOM (Bring Your Own Model) | Model control, prefix handling, offline deployment |
| Default embedding model | BAAI/bge-large-en-v1.5 (1024d) | Best MTEB general-purpose; requires query prefix |
| Chunk target size | 450 tokens | BGE-large max (512) minus boundary context overhead (~60) |
| Re-ingestion strategy | Delete-and-reinsert | Chunk boundaries change; no stable update target |
| ID generation | Deterministic (SHA-256 based) | Idempotent re-ingestion; identical content = identical IDs |
| Context headers | Metadata only, NOT embedded | Preserves full embedding capacity for semantic signal |
| Boundary context | Embedded (last 2 sentences of prev chunk) | Improves recall at embedding time; adjacency links for retrieval-time |
| BM25 enrichment | Validated LLM keywords + TF-IDF fallback | Closes vocabulary gap; validation prevents hallucinated keywords |
| Document-level keywords | Filterable, NOT BM25-indexed | Prevents irrelevant chunk pollution in keyword search |
| KG storage (v1) | Weaviate cross-references | Simpler deployment; Neo4j as v2 upgrade path |
| Error handling | Fail-safe (fallback) except re-ingestion cleanup (fail-hard) | One bad LLM call must not kill a 500-doc batch |

---

## 4. Discussion Points for Group Review

### 4.1 Architecture Decisions Requiring Team Input

| # | Topic | Question | Impact | Reference |
|---|-------|----------|--------|-----------|
| 1 | **Token budget defaults** | `target_chunk_size` is now 450 (safe for BGE-large with boundary context). Should we detect this automatically from model_registry.yaml instead of hardcoding? | Config correctness | Arch doc Section 3.2, 4.7 |
| 2 | **Refactoring cost vs value** | 3 LLM calls/section (refactor + fact-check + completeness). A 20-section doc = 60 LLM calls. Should refactoring be auto-skipped for self_reviewed docs or non-spec doc types? | Cost, latency | Arch doc Section 4.5 |
| 3 | **Re-ingestion safety** | If upstream nodes fail during re-ingestion, old data is now preserved (not deleted). But this means a corrupted-but-runnable document will keep stale data silently. Should we surface this more prominently (e.g., Weaviate property `stale_reingestion_failed: true`)? | Data integrity | Arch doc Section 10.1 |
| 4 | **KG in Weaviate xrefs — is it worth it?** | Weaviate cross-refs don't support graph traversal or path queries. Is the xref KG useful enough for v1, or should we skip KG entirely until Neo4j is ready? | Complexity vs value | Arch doc Section 4.13 |
| 5 | **Evaluation dataset ownership** | The eval dataset (50-100 queries with ground truth) is the most impactful artifact. Who builds it? Who maintains it as docs change? Needs a named owner and a process for updates. | Quality assurance | Arch doc Section 15.3 |
| 6 | **Schema evolution strategy** | Adding Weaviate properties is additive (no re-ingestion). But model swaps, index changes, and type changes require full re-ingestion. Is the blue-green collection strategy (Section 11.4) sufficient, or do we need a formal migration tool? | Operational | Arch doc Section 11.4 |
| 7 | **LLM cost estimation** | For a 500-doc batch with refactoring enabled: ~60 LLM calls/doc × 500 = 30,000 LLM calls. At GPT-4o-mini pricing (~$0.15/1K input tokens), estimate $50-150 per batch. With Claude Sonnet, significantly more. Is this acceptable? Should we provide a cost estimator? | Budget | Arch doc Sections 4.5, 4.6, 4.8 |
| 8 | **Shared KG nodes on re-ingestion** | Neo4j cleanup now uses two-phase (edges first, then orphaned nodes). But the "orphan check" query may be expensive at scale. Is this acceptable, or should we use reference counting? | Performance at scale | Arch doc Section 4.13 |

### 4.2 Implementation Prioritisation

Suggested phased implementation order for team discussion:

```
PHASE 1 — Core Pipeline (MVP, ~3 weeks)
├── Nodes 1, 2, 4, 6, 7, 11, 12     (ingest, structure, clean, chunk, enrich, quality, store)
├── Config system + validator
├── CLI (single file + batch)
├── Deterministic IDs + re-ingestion
└── Recursive fallback chunking (no LLM dependency for MVP)

PHASE 2 — LLM Enhancement (~2 weeks)
├── Node 6 LLM chunking (upgrade from fallback)
├── Node 8 metadata generation + keyword validation
├── Node 5 refactoring (optional, behind skip flag)
└── Domain vocabulary system

PHASE 3 — Extended Features (~2 weeks)
├── Node 3 multimodal processing (VLM)
├── Node 9 cross-reference extraction
├── Nodes 10 + 13 knowledge graph
├── Review tier system + API
└── PPTX / XLSX format extractors

PHASE 4 — Quality & Operations (~1 week)
├── Evaluation framework + dataset
├── Langfuse observability integration
├── Batch processing hardening
└── Schema migration tooling
```

### 4.3 Open Questions Not Covered in Architecture Document

1. **Deployment topology.** Is this a single-machine batch job, a Kubernetes service, or a serverless function? Affects how Weaviate, Ollama, and Neo4j are deployed.

2. **Access control.** The review tier system controls *visibility* in search, but who can *change* a review tier? Is there an auth layer, or is it trust-based?

3. **Monitoring and alerting.** Beyond Langfuse traces, what operational monitoring is needed? Weaviate health, embedding model memory, LLM API rate limits?

4. **Document deletion.** The architecture covers ingestion and re-ingestion. What about removing a document entirely from the system? Is it a manual Weaviate operation, or should the pipeline support a `--delete` flag?

5. **Multi-tenancy.** If multiple teams share the Weaviate instance, is collection-per-team or filter-per-team the right model?

6. **Backup and recovery.** If Weaviate data is lost, can it be reconstructed by re-running the pipeline on the source documents? What about KG data in Neo4j?

---

## 5. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| LLM API cost overrun on large batches | Medium | Medium | Cost estimator in CLI; refactoring skip for non-critical docs |
| Silent embedding truncation from config mismatch | High (was default) | High | Fixed: target_chunk_size now accounts for boundary overhead; config validator checks at startup |
| Evaluation dataset goes stale | High | High | Named owner; auto-eval after batch; dataset versioning |
| Weaviate unavailable during re-ingestion | Low | High | Old data preserved on failure; blue-green for production |
| LLM generates hallucinated keywords in BM25 index | Medium | Medium | Keyword validation gate (Section 14.3); TF-IDF fallback |
| Neo4j shared node deletion on re-ingestion | Medium | Medium | Fixed: two-phase cleanup preserves shared nodes |
| Abbreviation disambiguation failure (CDR ambiguity) | Medium | Medium | Domain-aware vocab; confidence scoring; manual review flag |

---

## 6. Glossary of Key Abstractions

For team members new to the codebase:

| Abstraction | What it is | Where defined |
|-------------|-----------|---------------|
| `PipelineDocument` | The state object that flows through all 13 nodes, accumulating data | `pipeline/models.py` |
| `BaseNode` | Abstract base class — every node inherits from this; never override `__call__` | `pipeline/nodes/base.py` |
| `PipelineConfig` | Master config with 13 sub-configs; three-layer precedence (defaults → JSON → CLI) | `pipeline/config.py` |
| `Chunk` | The atom of retrieval — gets embedded, stored in Weaviate, returned in search | `pipeline/models.py` |
| `KGTriple` | Subject-predicate-object relationship for the knowledge graph | `pipeline/models.py` |
| `ReviewTier` | Coarse trust level (FULLY / PARTIALLY / SELF_REVIEWED) — controls search visibility | `pipeline/models.py` |
| `ReviewStatus` | Fine-grained workflow state (DRAFT → SUBMITTED → IN_REVIEW → APPROVED/REJECTED) | `pipeline/models.py` |
| `deterministic_id()` | SHA-256-based UUID generator — same inputs always produce same ID | `pipeline/id_generation.py` |
| `enriched_content` | Chunk content + boundary context — this is what gets embedded (NOT context_header) | `pipeline/nodes/chunk_enrichment.py` |
| `context_header` | Metadata string for display — stored in Weaviate but NOT embedded | `pipeline/nodes/chunk_enrichment.py` |
