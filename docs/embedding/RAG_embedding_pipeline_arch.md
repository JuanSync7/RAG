# RAG Document Embedding Pipeline — Architecture & Design Document (v1.0.0)

## Document Information

| Field | Value |
|-------|-------|
| System | AION RAG Document Embedding Pipeline |
| Language | Python 3.12 |
| Orchestration | LangGraph (StateGraph) |
| Vector Store | Weaviate (BYOM mode) |
| Knowledge Graph | Weaviate cross-references (v1), Neo4j (v2) (optional) |
| Codebase | 21 files (estimated) |
| Version | 1.0.0 |
| Revision | None |

---

## Abbreviations Used in This Document

The following abbreviations appear throughout this architecture document. These are **document-level terminology** — not to be confused with the domain-specific abbreviations that the pipeline itself processes (e.g., DFT, OCV, SerDes).

| Abbreviation | Expansion |
|---|---|
| API | Application Programming Interface |
| ASIC | Application-Specific Integrated Circuit |
| AST | Abstract Syntax Tree |
| BGE | Beijing Academy of AI General Embedding (BAAI embedding model family) |
| BM25 | Best Matching 25 — probabilistic ranking function for keyword search |
| BYOM | Bring Your Own Model (Weaviate mode where embeddings are computed externally) |
| CLI | Command Line Interface |
| DAG | Directed Acyclic Graph |
| E5 | EmbEddings from bidirEctional Encoder rEpresentations (Microsoft embedding model family) |
| EDA | Electronic Design Automation |
| HNSW | Hierarchical Navigable Small World (approximate nearest-neighbour graph index) |
| HTML | HyperText Markup Language |
| JSON | JavaScript Object Notation |
| KG | Knowledge Graph |
| LLM | Large Language Model |
| MRR | Mean Reciprocal Rank (retrieval evaluation metric) |
| MTEB | Massive Text Embedding Benchmark |
| OCR | Optical Character Recognition |
| PDF | Portable Document Format |
| RAG | Retrieval-Augmented Generation |
| RSS | Resident Set Size (peak memory usage metric) |
| RST | reStructuredText (markup language) |
| SHA | Secure Hash Algorithm (SHA-256 used for content hashing) |
| TF-IDF | Term Frequency–Inverse Document Frequency |
| UTC | Coordinated Universal Time |
| UUID | Universally Unique Identifier |
| VLM | Vision Language Model (multimodal model for image understanding) |
| YAML | YAML Ain't Markup Language (configuration file format) |

---

## 1. System Purpose and Context

### 1.1 Problem Statement

ASIC design organisations accumulate critical engineering knowledge across hundreds of documents: specifications, design guides, runbooks, standard operating procedures, and project reports. This knowledge is fragmented across file servers, SharePoint, and individual workstations. When engineers leave, their contextual understanding of why certain design decisions were made, how specific tools were configured, or which voltage specifications apply to which power domains leaves with them.

Existing search tools (file name search, full-text search) fail for engineering documentation because the same technical term carries different meaning across domains. A search for "clock domain" returns hundreds of results across front-end design, DFT, verification, and physical design — with no way to distinguish which clock domain specification applies to which context.

Additionally, knowledge exists at varying levels of maturity — from formally approved specifications to an individual engineer's utility script. There is no mechanism to surface informal knowledge while clearly distinguishing it from authoritative sources.

### 1.2 Solution

The RAG Document Embedding Pipeline transforms engineering documents into semantically searchable, context-aware vector embeddings stored in Weaviate, with a complementary knowledge graph that captures relationships between documents, concepts, entities, and specifications. Each document passes through a 13-node processing graph that extracts structure, generates figure descriptions, cleans text, resolves domain abbreviations, refactors content for self-containedness, creates semantically coherent chunks, enriches chunks with hierarchical context, generates searchable metadata, extracts cross-references, builds knowledge graph triples, validates quality, and stores the final embeddings with full provenance.

The pipeline is designed for a mission-critical engineering environment where incorrect retrieval (e.g., returning a 1.2V specification when the query is about a 1.8V domain) could propagate into design errors. This drives the zero-hallucination constraint and multi-layer quality validation architecture.

Documents are tagged with a **review tier** (fully reviewed, partially reviewed, self-reviewed) that controls their visibility in search results, enabling the organisation to surface informal knowledge without misrepresenting it as authoritative.

The pipeline supports **idempotent re-ingestion**: when a document is updated and re-processed, all previous chunks and knowledge graph entries for that document are removed before new data is inserted, preventing stale or duplicated content.

### 1.3 Design Principles

**Swappability over lock-in.** Every external dependency — LLM provider, embedding model, vector store, structure detector — is behind a configuration interface. Changing from OpenAI to Anthropic, or from Weaviate to Pinecone, requires configuration changes, not code changes.

**Fail-safe over fail-fast.** When an LLM call fails, the pipeline falls back to deterministic alternatives (regex-based extraction, recursive splitting, TF-IDF keywords) rather than halting. In a batch run of 500 documents, one flawed LLM response must not kill the entire job.

**Context preservation over compression.** Unlike general-purpose RAG systems that aggressively compress content, this pipeline preserves every numerical value, specification, and procedural step. The refactoring node is constrained to restructure for clarity, never to summarise.

**Configuration-driven behaviour.** Pipeline behaviour (skip multimodal processing, disable refactoring, enable dry run) is controlled via a single PipelineConfig dataclass with 12 sub-configurations. Runtime configuration overrides are supported via JSON file and CLI arguments.

**Idempotency by construction.** Every node produces the same output given the same input. Identifiers are derived deterministically from content hashes. Write operations use upsert semantics. Re-ingesting a document produces an identical result to first-time ingestion — with all previous data cleanly replaced. LLM-dependent nodes are documented as "approximately idempotent" (semantically equivalent, not bit-identical) with deterministic fallbacks.

**Controlled access over restriction.** Knowledge at all maturity levels is ingested and searchable. A three-tier review system (fully reviewed, partially reviewed, self-reviewed) controls visibility at retrieval time without restricting what enters the system. Engineers can find an informal script, but the system clearly communicates that it has not been formally reviewed.

---

## 2. System Architecture

### 2.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI / Programmatic API                   │
│                         (main.py)                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  LangGraph StateGraph                       │
│                      (graph.py)                             │
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │  Node 1  │──>│  Node 2  │──>│  Node 3  │──>│  Node 4  │  │
│  │ Ingestion│   │ Structure│   │Multimodal│   │ Cleaning │  │
│  └────┬─────┘   └────┬─────┘   └──────────┘   └────┬─────┘  │
│       │              │ (conditional)               │        │
│       ▼              ▼                             ▼        │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │  Node 5  │──>│  Node 6  │──>│  Node 7  │──>│  Node 8  │  │
│  │ Refactor │   │ Chunking │   │Enrichment│   │ Metadata │  │
│  └──────────┘   └──────────┘   └──────────┘   └────┬─────┘  │
│  (conditional)                                     │        │
│                                                    ▼        │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │  Node 9  │──>│ Node 10  │──>│  Node 11 │──>│  Node 12 │  │
│  │Cross-Refs│   │ KG Build │   │ Quality  │   │ Storage  │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘  │
│  (conditional)  (conditional)                   (cleanup +  │
│                                                  upsert)    │
│                                                    │        │
│                                                    ▼        │
│                                               ┌──────────┐  │
│                                               │  Node 13 │  │
│                                               │  KG Store│  │
│                                               └──────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌──────────────────┐   ┌───────────────────┐
│  Weaviate Vector │   │  Knowledge Graph  │
│      Store       │   │  (Weaviate xrefs  │
│ (HNSW, BYOM)     │   │   / Neo4j v2)     │
└──────────────────┘   └───────────────────┘
```

### 2.2 Processing Flow

The pipeline processes a single document through 13 processing nodes in a directed acyclic graph, plus routing decision points that control conditional paths. The "13 nodes" count refers to processing nodes only — routing decision points (e.g., `kg_decision_point`) are pure conditional functions that inspect state and route to the next processing node without modifying the document. Four conditional routing points exist: multimodal processing (skipped if no figures detected), document refactoring (skippable via config), cross-reference extraction (skippable via config), and knowledge graph extraction (skippable via config).

```
document_ingestion (computes content hash, detects re-ingestion)
       │
       ▼
structure_detection
       │
       ├─  [has figures?] ──> multimodal_processing ──┐
       │                                              │
       └── [no figures] ──────────────────────────────┤
                                                      ▼
                                               text_cleaning
                                                      │
       ┌── [skip refactoring?] ───────────────────────┤
       │                                              │
       │                                              ▼
       │                                     document_refactoring
       │                                              │
       └──────────────────────────────────────────────┤
                                                      ▼
                                                llm_chunking
                                          (deterministic IDs,
                                           adjacency links)
                                                      │
                                                      ▼
                                              chunk_enrichment
                                                      │
                                                      ▼
                                            metadata_generation
                                                      │
       ┌── [skip cross-refs?] ────────────────────────┤
       │                                              │
       │                                              ▼
       │                              cross_reference_extraction
       │                                              │
       └──────────────────────────────────────────────┤
                                                      ▼
       ┌── [skip kg?] ───────────────────────────┐    │
       │                                         │    │
       │                                         ▼    │
       │                          knowledge_graph_extraction
       │                                         │    │
       └─────────────────────────────────────────┤────┘
                                                 ▼
                                          quality_validation
                                                 │
                                                 ▼
                                          embedding_storage
                                   (cleanup old data + upsert)
                                                 │
                                                 ▼
                                         kg_storage (conditional)
                                                 │
                                                 ▼
                                               [END]
```

### 2.3 State Management

All nodes operate on a single shared state object: PipelineDocument. This dataclass accumulates data as the document flows through the graph. Each node reads from the fields populated by upstream nodes and writes to its own output fields.

```
PipelineDocument (state object flowing through all nodes)
├── metadata: DocumentMetadata      ← populated by Node 1, enriched by Node 8
├── raw_content: str                ← populated by Node 1
├── raw_bytes: bytes                ← populated by Node 1 (binary formats)
├── content_hash: str               ← populated by Node 1 (SHA-256 of content)
├── is_reingestion: bool            ← populated by Node 1 (True if doc already exists)
├── previous_document_version: str  ← populated by Node 1 (previous content_hash if re-ingestion)
├── structure: StructureAnalysis    ← populated by Node 2
├── processed_figures: list         ← populated by Node 3
├── cleaned_content: str            ← populated by Node 4
├── refactored_content: str         ← populated by Node 5
├── chunks: list[Chunk]             ← populated by Node 6, enriched by Nodes 7, 8, 11
│   └── each Chunk now carries:
│       ├── chunk_id: str           ← deterministic: hash(document_id + chunk_index + content_hash)
│       ├── previous_chunk_id: str  ← set by Node 6
│       └── next_chunk_id: str      ← set by Node 6
├── cross_references: list          ← populated by Node 9
├── kg_triples: list[KGTriple]     ← populated by Node 10
├── review: ReviewMetadata          ← set by Node 1 from config/CLI, managed by review API
├── abbreviation_context: dict      ← loaded from domain vocabulary, injected into LLM prompts
├── processing_log: list[dict]      ← accumulated by all nodes
└── errors: list[str]               ← accumulated on failures
```

The LangGraph framework wraps this in a PipelineState TypedDict:

```python
class PipelineState(TypedDict):
    document: PipelineDocument
```

Every node's `__call__` receives `{"document": PipelineDocument}` and returns `{"document": PipelineDocument}`. This is enforced by the `BaseNode.__call__` method, which is never overridden by subclasses.

---

## 3. Component Design

### 3.1 BaseNode Abstract Class

File: `pipeline/nodes/base.py`

The BaseNode is the architectural contract that makes every node swappable. It is an Abstract Base Class with two abstract methods and one concrete method.

```
BaseNode (ABC)
├── __init__(config: PipelineConfig)     # Stores config, creates logger
├── process(document) → document         # ABSTRACT — node-specific logic
├── validate_input(document) → bool      # ABSTRACT — input guards
└── __call__(state) → state              # CONCRETE — LangGraph adapter
```

The `__call__` method is the critical integration point. It performs four operations in sequence:

1. **Input validation.** Calls `validate_input()`. If the node's prerequisites are not met (e.g., no chunks to enrich), the node is skipped with a log entry rather than raising an error.
2. **Logging.** Writes a "started" entry to `document.processing_log` with a UTC timestamp and active configuration parameters for this node.
3. **Processing.** Calls `process()` inside a try/except block. If the node raises an exception, the error message is appended to `document.errors` and a "failed" log entry is recorded.
4. **Completion.** Writes a "completed" entry to the processing log including metrics (items processed, duration, LLM call count).

No subclass overrides `__call__`. This guarantees uniform error handling, logging, and state wrapping across all 13 nodes. To replace a node, you implement a new subclass with its own `process()` and `validate_input()`, instantiate it in `graph.py`, and register it under the same node name.

### 3.2 Configuration System

File: `pipeline/config.py`

The configuration is a hierarchy of 13 dataclasses. The master PipelineConfig contains 12 sub-configurations and 6 pipeline-level flags.

```
PipelineConfig
├── llm: LLMProviderConfig
│   ├── provider: str = "openai"            # "openai" | "anthropic" | "ollama"
│   ├── model_name: str = "gpt-4o-mini"
│   ├── temperature: float = 0.0            # Deterministic for reproducibility
│   ├── api_key: Optional[str] = None       # None = environment variable
│   ├── base_url: Optional[str] = None      # For Ollama or custom endpoints
│   ├── max_tokens: int = 4096
│   └── timeout: int = 120
│
├── vlm: VLMProviderConfig
│   ├── provider: str = "ollama"
│   ├── model_name: str = "llava:13b"
│   ├── base_url: str = "http://localhost:8888"
│   └── timeout: int = 180                  # VLMs are slow
│
├── embedding: EmbeddingProviderConfig
│   ├── provider: str = "huggingface"       # "huggingface" | "openai" | "cohere"
│   ├── model_name: str = "BAAI/bge-large-en-v1.5"
│   ├── dimension: int = 1024               # Must match model output
│   ├── query_prefix: str = "Represent this sentence for searching relevant passages: "
│   │                                       # Model-specific query instruction prefix
│   │                                       # BGE: "Represent this sentence for searching relevant passages: "
│   │                                       # E5: "query: "
│   │                                       # Models without prefix: ""
│   │
│   │   WHY ASYMMETRIC PREFIXES EXIST: Many embedding models (BGE, E5, Instructor)
│   │   are trained with asymmetric objectives — the model learns that queries and
│   │   documents are structurally different. A query like "What is the clock frequency?"
│   │   is short and interrogative; the matching document passage is long and declarative.
│   │   The prefix tells the model which "role" the input plays, activating different
│   │   internal representations. Dropping the prefix silently degrades retrieval quality
│   │   by 10-15% (measured on MTEB benchmarks) because the model treats queries as
│   │   documents, placing them in the wrong region of the embedding space. Some newer
│   │   models (Jina v2, OpenAI text-embedding-3) do not require prefixes — they handle
│   │   both roles with a single encoding path. Always check the model card.
│   │
│   ├── document_prefix: str = ""           # Most models embed documents as-is
│   ├── batch_size: int = 32
│   ├── normalize: bool = True
│   ├── device: str = "cpu"                 # "cpu" | "cuda" | "mps"
│   ├── validate_dimension: bool = True     # Check first embedding matches config
│   └── use_model_tokeniser: bool = True    # Use actual model tokeniser for token counts
│
├── weaviate: WeaviateConfig
│   ├── url: str = "http://localhost:8080"
│   ├── collection_name: str = "engineering_documents"
│   ├── use_byom: bool = True               # Bring Your Own Model
│   ├── ef_construction: int = 128          # HNSW build parameter
│   ├── max_connections: int = 64           # HNSW connectivity
│   ├── ef: int = -1                        # HNSW search parameter (-1 = dynamic)
│   └── distance_metric: str = "cosine"
│
├── structure_detector: StructureDetectorConfig
│   ├── provider: str = "docling"
│   ├── ocr_enabled: bool = True
│   ├── table_extraction: bool = True
│   ├── figure_extraction: bool = True
│   ├── export_figures: bool = True
│   ├── quality_check: bool = True          # Flag low-confidence extractions
│   └── min_extraction_confidence: float = 0.5  # Below this → flag for manual review
│
├── chunking: ChunkingConfig
│   ├── strategy: str = "llm_driven"
│   ├── target_chunk_size: int = 450        # Target tokens (default accounts for ~60-token
│   │                                       # boundary context overhead from Node 7 — see note below)
│   │
│   │   TOKEN BUDGET NOTE: The enrichment node (Node 7) appends boundary context
│   │   (~2 sentences, 30-60 tokens) to each chunk's embedding input. The embedding
│   │   model's max_input_tokens (e.g., 512 for BGE-large) is the hard ceiling.
│   │   target_chunk_size MUST be set to max_input_tokens - boundary_overhead to
│   │   prevent silent truncation. Formula: target = max_input_tokens - (boundary_
│   │   context_sentences * ~30 tokens/sentence). For BGE-large: 512 - 60 = ~450.
│   │   For long-context models (Jina v2: 8192), this overhead is negligible and
│   │   target_chunk_size can be set higher. The config_validator (Section 16.3)
│   │   checks this constraint at startup.
│   │
│   ├── min_chunk_size: int = 100
│   ├── max_chunk_size: int = 1024
│   ├── overlap_tokens: int = 50
│   ├── prepend_section_path: bool = True
│   ├── table_atomic: bool = True           # Tables as indivisible units, header prepended on split
│   ├── section_size_limit_factor: int = 3  # Pre-split sections > max_chunk_size * this
│   └── boundary_context_sentences: int = 2 # Sentences from prev chunk for boundary awareness
│
├── quality: QualityConfig
│   ├── min_chunk_tokens: int = 20
│   ├── max_duplicate_similarity: float = 0.95
│   ├── enable_deduplication: bool = True
│   └── boilerplate_patterns: list[str]     # Regex patterns for page headers etc.
│
├── refactoring: RefactoringConfig
│   ├── max_iterations: int = 3
│   ├── enable_fact_check: bool = True
│   ├── enable_completeness_check: bool = True
│   └── confidence_threshold: float = 0.9
│
├── review: ReviewConfig
│   ├── default_tier: str = "self_reviewed"          # Default for new documents
│   ├── demote_on_reingestion: bool = True           # Auto-demote to partially_reviewed
│   ├── require_approval_for_promotion: bool = True  # Tier promotion needs reviewer
│   └── allowed_tiers: list[str] = ["fully_reviewed", "partially_reviewed", "self_reviewed"]
│
├── knowledge_graph: KnowledgeGraphConfig
│   ├── enabled: bool = True
│   ├── provider: str = "weaviate_xref"     # "weaviate_xref" | "neo4j"
│   ├── neo4j_url: Optional[str] = None
│   ├── neo4j_auth: Optional[tuple] = None
│   ├── extract_spec_values: bool = True    # Extract numerical specifications as nodes
│   ├── extract_relationships: bool = True  # LLM-based relationship extraction
│   └── max_triples_per_chunk: int = 20
│
├── vocabulary: VocabularyConfig
│   ├── dictionary_path: Optional[str] = None  # Path to domain_vocabulary.yaml
│   ├── auto_detect_abbreviations: bool = True # Extract new abbreviations from docs
│   ├── inject_into_prompts: bool = True       # Add vocabulary context to all LLM calls
│   └── max_prompt_terms: int = 50             # Limit vocabulary injection size
│
├── reingestion: ReingestionConfig
│   ├── strategy: str = "delete_and_reinsert"  # "delete_and_reinsert" | "skip_unchanged"
│   ├── hash_algorithm: str = "sha256"
│   ├── cleanup_vectors: bool = True           # Delete old chunks from Weaviate
│   ├── cleanup_kg: bool = True                # Delete old KG triples
│   └── preserve_review_tier: bool = False     # If False, auto-demote on change
│
├── observability: ObservabilityConfig
│   ├── enable_langfuse: bool = False
│   └── log_level: str = "INFO"
│
├── evaluation: EvaluationConfig
│   ├── enabled: bool = False
│   ├── dataset_path: Optional[str] = None  # Path to eval dataset JSON
│   ├── run_after_batch: bool = False        # Auto-run eval after batch ingestion
│   ├── metrics: list[str] = ["recall_at_5", "recall_at_10", "mrr"]
│   └── min_recall_at_10: float = 0.85      # Alert threshold
│
├── config_validation: ConfigValidationConfig
│   ├── validate_on_startup: bool = True     # Cross-check all config consistency
│   └── model_registry_path: Optional[str] = None  # Path to model_registry.yaml
│
├── skip_multimodal: bool = False
├── skip_refactoring: bool = False
├── skip_cross_references: bool = False
├── skip_knowledge_graph: bool = False
├── parallel_processing: bool = False
└── dry_run: bool = False
```

Configuration loading follows a three-layer precedence: dataclass defaults → JSON config file → CLI arguments. CLI arguments always win.

### 3.3 Data Models

File: `pipeline/models.py`

The data model hierarchy mirrors the processing pipeline. Each model represents a stage's output or an intermediate representation.

#### Enumerations

**DocumentFormat** defines the supported input formats across two phases:

Phase 1 : PDF, DOCX, HTML, MARKDOWN, PLAIN_TEXT, RST, PPTX, XLSX, and UNKNOWN.
Phase 2 (planned): VISIO, IMAGE (PNG/JPG for OCR), SYSTEMVERILOG (AST-based extraction).

Format detection is file-extension-based during ingestion, with a fallback to UNKNOWN.

Format-specific handling philosophy: **convert everything to text before the pipeline processes it.** Binary formats (PDF, DOCX, PPTX, XLSX) go through format-specific text extractors in Node 1 that produce `raw_content` as clean text. The rest of the pipeline operates uniformly on text regardless of source format.

Note on code and log files:
- **SystemVerilog/Verilog (`.sv`, `.v`):** Planned for Phase 2 using AST-based extraction (e.g., `pyverilog`). AST parsing extracts module hierarchy, port declarations, parameters, and assertions as structured data — far more useful than raw code as text, because it understands language constructs.
- **TCL scripts (`.tcl`):** Ingested as plain text (PLAIN_TEXT format). TCL EDA scripts are human-readable keyword-value commands that chunk naturally.
- **Tool logs (`.log`):** Excluded from the vector embedding pipeline. Logs are keyword-dense with minimal semantic content — they are better served by direct BM25 keyword search, not vector similarity. Logs should be indexed separately in a text search engine (Weaviate BM25-only collection or Elasticsearch).

**ProcessingStatus** tracks the lifecycle: PENDING → IN_PROGRESS → COMPLETED (or FAILED, SKIPPED). This status is set by the ingestion node (IN_PROGRESS) and the storage node (COMPLETED).

**ContentType** classifies chunk content: TEXT, TABLE, FIGURE, CODE, EQUATION, LIST, or HEADING. This tag travels from structure detection through chunking to the Weaviate schema, enabling content-type-aware retrieval (e.g., "show me all specification tables in the DFT domain").

**ReviewTier** classifies document maturity: FULLY_REVIEWED, PARTIALLY_REVIEWED, SELF_REVIEWED. Controls retrieval-time search space filtering. ReviewTier is the **coarse-grained trust level** — it determines which search space a document appears in.

**ReviewStatus** tracks the review lifecycle: DRAFT → SUBMITTED → IN_REVIEW → APPROVED → REJECTED. ReviewStatus is the **fine-grained workflow state** — it tracks where a document is in the review process within a tier. The two enums work together as follows:

```
ReviewTier = SELF_REVIEWED
  └── ReviewStatus: DRAFT → SUBMITTED → IN_REVIEW
      └── If approved → Tier promoted to PARTIALLY_REVIEWED, Status = APPROVED
      └── If rejected → Tier stays SELF_REVIEWED, Status = REJECTED (with feedback)

ReviewTier = PARTIALLY_REVIEWED
  └── ReviewStatus: SUBMITTED → IN_REVIEW
      └── If domain lead approves → Tier promoted to FULLY_REVIEWED, Status = APPROVED
      └── If major revision needed → Tier demoted to SELF_REVIEWED, Status = REJECTED

ReviewTier = FULLY_REVIEWED
  └── ReviewStatus: APPROVED
      └── If document re-ingested with changes → Tier demoted to PARTIALLY_REVIEWED,
          Status = DRAFT, auto_demoted = True
```

A document's retrieval visibility is governed by ReviewTier (the filter in Weaviate queries). ReviewStatus is informational — it tells reviewers what action is needed but does not affect search results.

**KGNodeType** classifies knowledge graph vertices: DOCUMENT, CHUNK, CONCEPT, ENTITY, DOMAIN, PERSON, ABBREVIATION, SPEC_VALUE.

**KGEdgeType** classifies knowledge graph edges: CONTAINS, REFERENCES, DEPENDS_ON, MENTIONS, BELONGS_TO, RELATED_TO, ABBREVIATION_OF, SPECIFIES, AUTHORED_BY, SUPERSEDES, NEXT_CHUNK.

#### Structure Detection Output

BoundingBox stores page coordinates for detected elements. DetectedFigure carries the full lifecycle of a figure: detection coordinates from Docling, exported image bytes/path for VLM processing, and VLM-generated description with confidence score. DetectedTable stores parsed rows, headers, and a markdown representation used for text integration. DocumentSection is a recursive tree node with level, title, content, content type, and child sections. StructureAnalysis is the root container holding the section tree, figure list, table list, page count, and routing flags (has_figures, has_tables).

#### Core Retrieval Unit

Chunk is the atom of the retrieval system — the unit that gets embedded and stored in Weaviate. It carries:

- **Content:** content (raw text), enriched_content (with context headers)
- **Positional context:** section_path, page_numbers, chunk_index
- **Adjacency links:** previous_chunk_id, next_chunk_id (for context expansion at retrieval time)
- **Metadata:** keywords, entities, content_type
- **Linked multimodal references:** figure_ids, table_ids
- **Quality metrics:** token_count, is_duplicate, quality_score
- **Identity:** chunk_id (deterministic, see Section 4.6), content_hash

#### Knowledge Graph Triple

KGTriple represents a single subject-predicate-object relationship extracted from the document:

```python
@dataclass
class KGTriple:
    triple_id: str              # deterministic: hash(document_id + subject + predicate + object)
    subject: str                # Entity or concept name
    subject_type: KGNodeType    # CONCEPT, ENTITY, SPEC_VALUE, etc.
    predicate: str              # Relationship label
    predicate_type: KGEdgeType  # REFERENCES, DEPENDS_ON, SPECIFIES, etc.
    object: str                 # Target entity or concept
    object_type: KGNodeType
    source_chunk_id: str        # Provenance — which chunk this was extracted from
    source_document_id: str     # Parent document
    confidence: float           # 0.0–1.0
```

#### Abbreviation Entry

```python
@dataclass
class AbbreviationEntry:
    abbreviation: str           # "DFT"
    expansion: str              # "Design for Testability"
    domain: str                 # "dft"
    context: Optional[str]      # "ASIC test methodology"
    related: list[str]          # ["ATPG", "scan chain"]
    source: str                 # "dictionary" | "auto_detected" | "document:{doc_id}"
```

#### Review Metadata

```python
@dataclass
class ReviewMetadata:
    tier: ReviewTier = ReviewTier.SELF_REVIEWED
    status: ReviewStatus = ReviewStatus.DRAFT
    reviewed_by: list[str] = field(default_factory=list)
    review_date: Optional[str] = None
    review_notes: Optional[str] = None
    auto_demoted: bool = False   # True if tier was lowered due to re-ingestion
```

#### Relationship Model

CrossReference represents a detected link between documents or sections. It stores the source document, the raw reference text ("see Section 3.2 of DFT Guide"), an optional resolved target, the reference type (explicit, implicit, standard, version, dependency), and a confidence score. Target resolution (mapping "DFT Guide" to a specific document_id) is deferred to the graph layer — not the pipeline itself.

#### Document-Level Metadata

DocumentMetadata captures identity (document_id as UUID, source_path), filesystem metadata (authors, dates, version), domain classification (domain, doc_type), generated metadata (summary, keywords, entities, topics), processing tracking (timestamp, status, pipeline version), and content integrity (content_hash for re-ingestion detection).

#### Deterministic ID Generation

All IDs are now deterministic, derived from content, to ensure idempotent re-ingestion.

```python
import hashlib

def deterministic_id(components: list[str]) -> str:
    """Generate a deterministic UUID-format ID from input components."""
    combined = "|".join(str(c) for c in components)
    hash_hex = hashlib.sha256(combined.encode()).hexdigest()
    # Format as UUID: 8-4-4-4-12
    return f"{hash_hex[:8]}-{hash_hex[8:12]}-{hash_hex[12:16]}-{hash_hex[16:20]}-{hash_hex[20:32]}"

# Document ID: based on source_path (stable across re-ingestion)
document_id = deterministic_id([source_path])

# Chunk ID: based on parent document + position + content
chunk_id = deterministic_id([document_id, chunk_index, content_hash])

# KG Triple ID: based on document + relationship
triple_id = deterministic_id([document_id, subject, predicate, object_value])
```

This means re-ingesting the same document with identical content produces identical IDs. If the content changes (different chunk boundaries, updated text), new IDs are generated and the old ones are cleaned up by the re-ingestion process (see Section 4.12).

**Important: chunk IDs do not survive across content changes, even for unchanged paragraphs.** Because `chunk_index` is a positional component, if an earlier section changes and shifts chunk boundaries, every subsequent chunk receives a new `chunk_id` — even if its content is byte-identical to the previous version. This is correct and intentional for the delete-and-reinsert re-ingestion strategy: the old IDs are bulk-deleted and new IDs bulk-inserted. There is no concept of "this chunk survived across document versions." If cross-version chunk tracking is needed in the future (e.g., for retrieval feedback persistence), a content-only hash (without `chunk_index`) could serve as a stable secondary identifier.

---

## 4. Node Specifications

### 4.1 Node 1 — Document Ingestion

File: `pipeline/nodes/document_ingestion.py`

**Purpose:** Reads the source file from disk, detects its format, computes a content hash for re-ingestion detection, loads the domain vocabulary, checks for existing documents in the vector store, and populates initial metadata including review tier.

**Input validation:** Requires `metadata.source_path` to be non-empty and point to an existing file.

**Processing logic:**

1. **Format detection.** Maps file extension to DocumentFormat via an EXTENSION_TO_FORMAT dictionary. Unknown extensions get `DocumentFormat.UNKNOWN`.

2. **Binary/text routing and format-specific text extraction.** The core principle is: convert everything to text before downstream processing.
   - Binary formats (PDF, DOCX) are read as `raw_bytes` with `raw_content` left empty — Docling will extract text in Node 2.
   - Text formats (Markdown, HTML, TXT, RST, TCL) are read as `raw_content` with encoding fallback.
   - **PPTX (PowerPoint):** Extracted via `python-pptx`. Iterates over slides in order, extracting text from all text frames, speaker notes, and table cells. Each slide is separated by a slide marker (`[SLIDE N: {title}]`) to preserve presentation structure. Slide titles become section-level headings for the structure tree.
   - **XLSX (Excel):** Extracted via `openpyxl`. Each sheet is processed as a named section. Tables are detected by header rows (first row with content) and converted to markdown table format with headers preserved. Named ranges and cell references are preserved as-is. Charts are noted as `[CHART: {title}]` placeholders (chart data extraction is Phase 2).

3. **Encoding fallback chain.** Text files are tried in order: UTF-8 → Latin-1 → CP1252 → UTF-8 with replacement characters. This handles legacy EDA tool outputs that frequently use Windows-1252 encoding.

4. **Content hash computation.** Computes SHA-256 hash of the file bytes (`raw_bytes` for binary, `raw_content.encode()` for text). This hash is the primary mechanism for detecting whether a document has changed since last ingestion.

5. **Re-ingestion detection.** Queries Weaviate for any chunks with a matching `document_id` (derived deterministically from `source_path`). If found:
   - Sets `is_reingestion = True`
   - Stores the previous `content_hash` as `previous_document_version`
   - If the content hash matches the existing document AND `reingestion.strategy == "skip_unchanged"`, marks the document as SKIPPED and short-circuits the pipeline
   - If `review.demote_on_reingestion` is True and the content hash differs, sets `review.tier` to PARTIALLY_REVIEWED and `review.auto_demoted = True`

6. **Domain vocabulary loading.** If `config.vocabulary.dictionary_path` is set, loads the YAML abbreviation dictionary into `document.abbreviation_context`. This dictionary is injected into all downstream LLM prompts.

7. **Review tier initialisation.** Sets `review.tier` from CLI override, metadata override, or `config.review.default_tier` (in that precedence order).

8. **Metadata population.** Sets format, title (derived from filename), modified_date (from filesystem), content_hash, document_id (deterministic from source_path), and transitions `processing_status` to IN_PROGRESS.

**Output:** PipelineDocument with `raw_content` (text formats) or `raw_bytes` (binary formats), `content_hash`, `is_reingestion` flag, `abbreviation_context`, `review` metadata, and initial metadata populated.

### 4.2 Node 2 — Structure Detection

File: `pipeline/nodes/structure_detection.py`

**Purpose:** Parses the document into a hierarchical section tree, extracts figures and tables, and exports figure images for VLM processing.

**Input validation:** Requires either `raw_content` or `raw_bytes` to be non-empty.

**Processing logic (Docling provider):**

1. **Document conversion.** Passes the source file path to Docling's DocumentConverter, which handles PDF, DOCX, and HTML natively with layout analysis.
2. **Text extraction.** Exports the Docling result to markdown. If `raw_content` was empty (binary format), it is populated with the exported markdown.
3. **Section tree construction.** Walks the Docling document tree, creating DocumentSection nodes for each heading-delimited section. The hierarchy preserves heading levels (H1 → level 1, H2 → level 2, etc.) with parent-child relationships.
4. **Table extraction.** Converts Docling's table model into DetectedTable objects with headers, rows, and a markdown representation. Tables are linked to their parent section.
5. **Figure extraction.** Converts Docling's picture model into DetectedFigure objects with bounding boxes, captions, and surrounding text context. If `export_figures` is enabled, figure images are saved to disk for VLM processing.
6. **Abbreviation auto-detection.** Scans the document text for abbreviation definition patterns (e.g., "Design for Testability (DFT)", "DFT: Design for Testability", or abbreviation tables). Detected abbreviations are merged into `document.abbreviation_context` with `source: "document:{document_id}"`.
7. **Structure extraction quality check.** Validates the quality of Docling's extraction by computing an `extraction_confidence` score (0.0–1.0) based on:
   - **Section tree depth:** A document with >5 pages but a flat section tree (depth ≤ 1) scores low — it likely had headings that Docling missed.
   - **Table completeness:** Tables where extracted rows are empty or headers are missing score low.
   - **Text coherence:** Large gaps in page coverage (e.g., pages 3-5 produced no text) indicate extraction failure on those pages.
   - **Character density:** Very low character-per-page ratio compared to expected engineering document density flags scanned/image-only pages.

   If `extraction_confidence` falls below `config.structure_detector.min_extraction_confidence` (default 0.5), the document is flagged in the processing log with status `"warning"` and details `"low_extraction_confidence: {score}"`. The document still proceeds through the pipeline, but the flag can be used by the batch processor to generate a manual review report. Chunks from low-confidence documents inherit a quality penalty in Node 11.
8. **Routing flag computation.** Sets `has_figures` and `has_tables` based on whether any figures/tables were detected. `has_figures` drives the conditional routing to Node 3.

**Swappability:** The `process()` method dispatches on `config.structure_detector.provider`. Adding a new provider (e.g., Unstructured) requires implementing a new `_process_with_unstructured()` method that populates the same StructureAnalysis output model.

**Output:** PipelineDocument with `structure` populated as a complete StructureAnalysis, and `abbreviation_context` enriched with any auto-detected abbreviations.

### 4.3 Node 3 — Multimodal Processing (Conditional)

File: `pipeline/nodes/multimodal_processing.py`

**Purpose:** Converts detected figures into text descriptions using a Vision-Language Model, making visual content searchable via text embeddings.

**Routing condition:** This node is only reached if `structure.has_figures` is True and at least one figure exists. Otherwise, the graph routes directly to Node 4 (Text Cleaning).

**Input validation:** Requires `structure` to be non-None and `has_figures` to be True.

**Processing logic:**

1. **Figure iteration.** Processes each DetectedFigure in `structure.figures` independently.
2. **Image loading.** Loads the figure from `image_data` (in-memory bytes) or `image_path` (exported file), base64-encodes it for the VLM API.
3. **Prompt construction.** Builds a multi-part message with the image, surrounding document text, caption context, and the domain abbreviation dictionary (from `abbreviation_context`). The prompt instructs the VLM to describe diagram type, all visible labels and values, key relationships, and numerical specifications — while prohibiting speculation.
4. **VLM invocation.** Sends the image+text message to the configured VLM (default: Ollama/LLaVA:13b).
5. **Confidence estimation.** Scores the VLM response using a heuristic: empty responses score 0.0, short responses (~0.5), and long responses with technical indicators (numbers, units, signal names, engineering terms) score higher. The 150-character threshold separates "meaningful description" from "generic caption".
6. **Result storage.** Updates the DetectedFigure with description, confidence, and adds it to `document.processed_figures`.

**Swappability:** VLM provider is configurable (Ollama, OpenAI GPT-4V, HuggingFace). Changing provider requires only `config.vlm.provider` and `config.vlm.model_name` changes.

**Output:** PipelineDocument with `processed_figures` containing VLM-generated descriptions and confidence scores.

### 4.4 Node 4 — Text Cleaning

File: `pipeline/nodes/text_cleaning.py`

**Purpose:** Normalises raw document text, removes boilerplate artefacts, and integrates multimodal content (figure descriptions, table markdown) into the text stream.

**Input validation:** Requires `raw_content` to be non-empty.

**Processing logic:**

1. **Whitespace normalisation.** Collapses multiple spaces to single spaces and limits consecutive newlines to a maximum of two.
2. **Boilerplate removal.** Applies regex patterns from `config.quality.boilerplate_patterns` to filter page headers ("Page X of Y"), confidentiality notices, and stray page numbers.
3. **Repeated header removal.** Detects lines that appear more than 3 times in the document (typically page headers/footers in PDFs) and reduces them to a single occurrence.
4. **Figure description integration.** For each figure in `processed_figures`, inserts a formatted block containing the figure ID marker (`[FIGURE_ID: fig01]`), the VLM-generated description, and the original caption. This makes figure content searchable within the text.
5. **Table markdown integration.** For each table in `structure.tables`, inserts the table ID marker (`[TABLE_ID: tbl01]`), the markdown representation, and the caption. This preserves tabular data (voltage specifications, pin tables) in a text-searchable format.

**Output:** PipelineDocument with `cleaned_content` populated.

### 4.5 Node 5 — Document Refactoring (Conditional)

File: `pipeline/nodes/document_refactoring.py`

**Purpose:** Restructures document text to make each paragraph self-contained for RAG retrieval, resolving implicit references ("as mentioned above", "the same voltage") to explicit references ("the 1.8V core voltage specified in Section 1.2").

**Why self-containedness matters for RAG:** In a retrieval system, each chunk is returned to the user (or to the answer-generation LLM) in isolation — stripped from its surrounding document. If a chunk says "the same voltage as described above," the reader has no idea what voltage is being referenced. The original document relied on sequential reading order to make this clear; RAG retrieval breaks that assumption. Without refactoring, the retrieval system returns technically correct chunks that are practically useless — the right chunk is found, but its content is ambiguous without the paragraphs that preceded it. This is especially dangerous in an engineering context: "the same voltage" could be 1.2V or 1.8V, and returning a chunk without that specificity could propagate into a design error. The refactoring node pays the cost of 3 LLM calls per section to eliminate this class of retrieval failure at ingestion time, rather than forcing every query to fetch and reassemble surrounding context at retrieval time.

**Routing condition:** Skippable via `config.skip_refactoring` or processing log flag.

**Input validation:** Requires `cleaned_content` or `raw_content` with at least 50 characters.

**Processing logic — agentic loop:**

The refactoring node implements a self-correcting agentic loop with up to `max_iterations` (default 3) attempts per section. The domain abbreviation dictionary from `abbreviation_context` is injected into the refactoring prompt to ensure abbreviations are expanded correctly.

```
for each section:
    for iteration in 1..max_iterations:
        1. REFACTOR: LLM rewrites section for self-containedness
           (with abbreviation dictionary in prompt context)
        2. FACT-CHECK: Second LLM call validates no hallucination
           ├── Checks for: added information, missing content,
           │   meaning distortion, numerical errors
           └── Returns JSON with passed/failed and detailed findings
        3. COMPLETENESS-CHECK: Third LLM call validates no content loss
           ├── Enumerates every fact in the original
           └── Reports completeness ratio (0.0–1.0)
        4. If both checks pass → accept refactored text
        5. If fact-check fails → feed errors back as correction prompt
        6. If completeness < 80% → reject, return original text

    If all iterations fail → return original text (fail-safe)
```

**Prompt engineering:**

The refactoring prompt constrains the LLM with 8 absolute rules: never add information, never remove content, never change meaning, may restructure for clarity, may expand abbreviations (only if the expansion exists in the document or the abbreviation dictionary), may add section context, must convert implicit references to explicit ones, and must preserve all numerical values exactly.

The fact-check prompt returns a structured JSON with specific lists of hallucinations, missing content, distortions, and numerical errors — enabling targeted correction in subsequent iterations.

**Output:** PipelineDocument with `refactored_content` populated (or `cleaned_content` preserved if refactoring was skipped or failed).

### 4.6 Node 6 — LLM-Driven Chunking

File: `pipeline/nodes/llm_chunking.py`

**Purpose:** Splits document text into semantically coherent chunks that respect topic boundaries, specification blocks, and procedure sequences. Assigns deterministic IDs and adjacency links.

**Input validation:** Requires content (refactored, cleaned, or raw) to be non-empty.

**Processing logic:**

1. **Section-based windowing with size limits.** If structure analysis is available, the document is split into processing windows aligned to section boundaries. Each window carries its section path (e.g., "Power Domains > Core Voltage > Timing"). Without structure, the full text is processed as a single window. **Section-size pre-split:** If any section exceeds `max_chunk_size * section_size_limit_factor` tokens (default: `1024 * 3 = 3072` tokens), it is pre-split at paragraph boundaries into sub-windows, each carrying the parent section path. This ensures the LLM receives windows within its context limits while preserving structural context.

2. **Table atomic chunking.** When `config.chunking.table_atomic` is True (default), tables detected in the structure analysis are treated as indivisible chunking units. Tables are chunked separately from surrounding text:
   - If a table fits within `max_chunk_size`, it becomes a single chunk with `content_type = TABLE`.
   - If a table exceeds `max_chunk_size`, it is split **row-wise** with the header row prepended to every chunk fragment. This ensures each fragment is self-describing — a search for "register CTRL_STATUS" will match the fragment containing that row, and the column headers ("Register", "Offset", "Reset Value") are present for context.
   - Tables are removed from their containing section window before LLM chunking to prevent double-processing.

3. **LLM chunking call.** Each window is sent to the LLM with a domain-specific prompt (including abbreviation dictionary) that instructs: one topic per chunk, target 450 tokens (configurable — set to embedding model's max_input_tokens minus boundary context overhead), never split specification blocks or procedure sequences, preserve all content, and return JSON with chunk content, topic label, content type, and linked figures/tables.

4. **JSON response parsing.** The LLM response is parsed as a JSON array. Each element becomes a Chunk object with content type mapping and token count.

5. **Deterministic ID assignment.** Each chunk receives a deterministic `chunk_id`:
   ```python
   chunk.content_hash = hashlib.sha256(chunk.content.encode()).hexdigest()[:16]
   chunk.chunk_id = deterministic_id([document_id, chunk_index, chunk.content_hash])
   ```
   This ensures re-ingesting the same document with identical content produces identical chunk IDs. Changed content produces new IDs, and the old IDs are cleaned up during re-ingestion (see Node 12).

6. **Adjacency link assignment.** After all chunks are created and ordered:
   ```python
   for i, chunk in enumerate(chunks):
       chunk.previous_chunk_id = chunks[i-1].chunk_id if i > 0 else None
       chunk.next_chunk_id = chunks[i+1].chunk_id if i < len(chunks)-1 else None
   ```
   These links enable "expand context" at retrieval time — when a user retrieves chunk #5, the retrieval layer can immediately fetch chunks #4 and #6 without re-reading the source document.

7. **Chunking method tag.** Each chunk is tagged with its production method:
   - `chunking_method = "llm_semantic"` for LLM-produced chunks
   - `chunking_method = "fallback_recursive"` for fallback-produced chunks
   - `chunking_method = "table_atomic"` for table-specific chunks

   This tag is stored in Weaviate, enabling monitoring (if 30% of documents fall back, the LLM pipeline needs attention) and optional retrieval-time quality weighting.

8. **Fallback recursive splitting.** If the LLM call fails (timeout, rate limit, malformed JSON), the system falls back to a deterministic recursive character splitter. This splitter splits on paragraph boundaries (`\n\n`), then sentence boundaries, targeting the configured chunk size with overlap. Fallback chunks get `ContentType.TEXT`, the section path from the processing window, and `chunking_method = "fallback_recursive"`. Deterministic IDs and adjacency links are applied identically to fallback chunks.

9. **Content type mapping.** String labels from the LLM ("text", "table", "code", "figure", "specification") are mapped to ContentType enum values. Unknown types default to TEXT.

10. **Accurate token counting.** When `config.embedding.use_model_tokeniser` is True (default), uses the actual tokeniser from the configured embedding model to count tokens per chunk. The tokeniser is loaded once lazily and reused. This correctly handles ASIC terminology where a single "word" like `JTAG_TDI_ACTIVE_LOW` may tokenise into 7+ subword tokens. When the flag is False, falls back to the conservative approximation `word_count * 1.3` for environments where the tokeniser is unavailable.

**Output:** PipelineDocument with `chunks` populated as a list of Chunk objects, each with deterministic IDs and adjacency links.

### 4.7 Node 7 — Chunk Enrichment

File: `pipeline/nodes/chunk_enrichment.py`

**Purpose:** Builds metadata context headers for retrieval-time filtering and display, appends boundary context from adjacent chunks, and prepares the embedding input. Context headers are stored as Weaviate metadata properties for filtering — they are NOT embedded into the vector. The raw chunk content is what gets embedded, keeping the full embedding capacity for semantic signal.

**Input validation:** Requires at least one chunk in `document.chunks`.

**Processing logic:**

**A. Context header construction** (for metadata, not embedding):

For each chunk, builds a `context_header` string stored as a separate field:
1. **Context line.** Format: `[Context: {domain} | {doc_type} | {title} | {section_path}]`.
2. **Content type tag.** For non-TEXT chunks, adds `[Type: {content_type}]`.
3. **Review tier tag.** Adds `[Review: {tier}]`.
4. **Linked content references.** If the chunk references figures or tables, adds `[References: fig01, tbl03]`.

This header is stored in the Weaviate `context_header` property for display in search results and can be used for filtering. It is NOT prepended to the embedding input.

**Rationale for not embedding headers:** Context headers consume 20-30 tokens of embedding capacity (BGE-large has a 512-token input limit) with information that is better served by Weaviate metadata filters. Domain disambiguation is handled by filtering on `document_domain`, content type by `content_type`, and review tier by `review_tier`. Embedding the raw content maximises semantic signal for retrieval. This decision should be validated with the evaluation dataset (see Section 15) — if testing shows headers improve retrieval, this can be reversed via config.

**B. Boundary context injection:**

For each chunk (except the first), appends the last N sentences from the previous chunk as a faded boundary context:

```
[Previous context: ...last 2-3 sentences of preceding chunk...]
{original chunk content}
```

The number of sentences is controlled by `config.chunking.boundary_context_sentences` (default 2). This gives the embedding awareness of the topic transition at the chunk boundary without requiring explicit overlap or always fetching neighbours at retrieval time.

**Why boundary context AND adjacency links (they solve different problems):** Adjacency links (`previous_chunk_id` / `next_chunk_id`) are a retrieval-time mechanism — after a chunk is found, the retrieval layer can fetch its neighbours to give the user or the answer-generation LLM more context. But adjacency links do nothing to help the chunk get *found* in the first place. Boundary context injection is an *embedding-time* mechanism — by including the last 2 sentences of the preceding chunk in the embedding input, the vector representation captures the topic transition. Consider a chunk that begins "These timing constraints apply to all scan chains in the design." Without boundary context, the embedding has no signal about *which* timing constraints — the preceding chunk's closing sentences ("The DFT insertion tool configures hold-fix cells at 1.2V...") give the embedding model the semantic bridge it needs. In short: boundary context improves *recall* (the right chunk gets found), adjacency links improve *comprehension* (the user sees enough context to understand it).

**C. Embedding content assembly:**

Sets `chunk.enriched_content` to: boundary context (if applicable) + raw `chunk.content`. This is what gets embedded. The raw `content` is preserved separately for display in retrieval results.

**Token budget interaction with boundary context:** Node 6 (Chunking) counts tokens for each chunk and targets `max_chunk_size` (default 1024 tokens). Node 7 then adds boundary context (typically 2 sentences, ~30-60 tokens) on top. This means the final `enriched_content` that gets embedded can exceed the chunk's `token_count` field by the boundary context size. For models with generous input limits (Jina v2: 8192 tokens), this is irrelevant. For models with tight limits (BGE-large: 512 tokens), this matters: a chunk at exactly 512 tokens plus 50 tokens of boundary context produces 562 tokens, and the embedding model will silently truncate the trailing tokens — losing the chunk's final sentences from the embedding.

**Implementation guidance:** The enrichment node should recount tokens on `enriched_content` after assembly and store the updated count. If the enriched token count exceeds the embedding model's `max_input_tokens` (from the model registry), the boundary context should be trimmed (reduce sentences) rather than the chunk content. The `token_count` field stored in Weaviate should reflect the raw `content` length (for user-facing metrics), while a separate check ensures `enriched_content` fits the model. The chunking node's `target_chunk_size` default (450 tokens) already accounts for this overhead for the default BGE-large model (512 max input - ~60 token boundary context = ~450). For long-context models (Jina v2: 8192 tokens), `target_chunk_size` can be set much higher.

**Output:** PipelineDocument with each chunk's `enriched_content` (for embedding) and `context_header` (for metadata/display) populated.

### 4.8 Node 8 — Metadata Generation

File: `pipeline/nodes/metadata_generation.py`

**Purpose:** Generates searchable metadata at both document level and chunk level to support hybrid retrieval (BM25 keyword search + vector similarity).

**Input validation:** Requires content and at least one chunk.

**Processing logic — two levels:**

**Document-level metadata:** Sends the first 3,000 characters of the document to the LLM with the abbreviation dictionary and a prompt requesting: 2–3 sentence summary, 10–20 technical keywords, named entities (tools, standards, IPs), 3–5 topic categories, domain classification (one of: frontend_design, verification, dft, physical_design, siops, analog, general), and document type classification (specification, guide, runbook, etc.). The LLM-generated metadata is merged with any existing metadata from ingestion (file-derived title, dates) without overwriting user-provided overrides.

**Chunk-level metadata:** For each chunk, sends the content with its section path and abbreviation dictionary to the LLM requesting 5–10 keywords and named entities. These keywords enable BM25 exact-match search alongside vector similarity.

**Keyword and entity validation (critical for BM25 quality):** All LLM-generated keywords and entities are passed through a validation gate before being stored in BM25-indexed fields. The validation ensures that every keyword is *grounded* in the chunk content — either the term appears directly in the text, is an abbreviation expansion of something in the text, or shares significant word overlap with the text. Keywords that the LLM inferred by association (e.g., tagging a "scan chain insertion" chunk with "ATPG" because the LLM knows they're related, even though ATPG isn't discussed) are discarded. See Section 14.3 for the full validation algorithm and rules.

This validation is the primary defence against BM25 keyword hallucination — the risk that LLM-generated keywords introduce false positives into the inverted index, causing irrelevant chunks to appear in search results. See Section 14.2 for a detailed risk assessment of each enrichment action.

**Document-level keyword scoping:** Document-level keywords (`document_keywords`) are stored as a filterable property on each chunk but are NOT BM25-indexed. This prevents a document-level keyword like "ATPG" from making every chunk in the document (including a project timeline chunk) appear in BM25 results for "ATPG". Document-level keywords are used for pre-filtering ("find documents about ATPG") at the document level, not for chunk-level ranking. See Section 14.2 for the rationale.

**Fallback keyword extraction:** If the LLM call fails, a deterministic TF-IDF-style extractor runs. It tokenises the text, filters stop words and short tokens, counts frequencies, and returns the top terms. This ensures every chunk has at least some keywords even without LLM access. TF-IDF keywords are the safest BM25 signal — purely statistical, no hallucination risk.

**Output:** PipelineDocument with `metadata` enriched (summary, keywords, entities, domain, doc_type) and each chunk's keywords and entities populated (validated against content).

### 4.9 Node 9 — Cross-Reference Extraction (Conditional)

File: `pipeline/nodes/cross_reference_extraction.py`

**Purpose:** Detects cross-references between documents, sections, and external standards to feed the knowledge graph for relationship-aware retrieval.

**Routing condition:** Skippable via `config.skip_cross_references` or processing log flag.

**Processing logic — two-pass approach:**

**Pass 1: Regex extraction** (fast, deterministic). Five compiled regex patterns scan the document text for common reference formats:

- Section/chapter/appendix citations: "see Section 3.2", "refer to Appendix A"
- Named document references: "described in the DFT Implementation Guide"
- Standards citations: "per IEEE 802.3", "according to JEDEC JESD79", "ISO 9001"
- Bracketed references: "[REF-001]", "(DOC-042)"
- Version references: "version 2.1 of the Clock Domain Spec"

Each match produces a CrossReference with confidence 0.8 and ref_type "explicit".

**Pass 2: LLM extraction** (deep, contextual). Sends the first 5,000 characters to the LLM with the abbreviation dictionary, requesting JSON-structured references including raw text, target document, target section, reference type (explicit, standard, version, dependency, implicit), and confidence. This pass catches implicit references that regex cannot detect, such as "requires completion of floorplan" (a dependency reference) or "the same methodology" (an implicit reference to another document).

**Merge and deduplication.** LLM references take priority. References are deduplicated by normalising `target_reference` to lowercase and checking for exact matches. When a duplicate is found, the LLM version is kept (higher confidence, richer metadata).

**Output:** PipelineDocument with `cross_references` populated.

### 4.10 Node 10 — Knowledge Graph Extraction (Conditional — NEW)

File: `pipeline/nodes/knowledge_graph_extraction.py`

**Purpose:** Extracts structured subject-predicate-object triples from document content, entities, and cross-references to build a knowledge graph that enables relationship-aware retrieval and impact analysis.

**Routing condition:** Skippable via `config.skip_knowledge_graph` or processing log flag.

**Input validation:** Requires at least one chunk with keywords/entities (populated by Node 8). Falls back to cross-reference-only graph if metadata generation failed.

**Processing logic — three stages:**

**Stage 1: Structural triples** (deterministic, no LLM). Generates triples from the pipeline's own structural knowledge:

```
(document_id, CONTAINS, chunk_id)          — for each chunk
(chunk_id, NEXT_CHUNK, next_chunk_id)      — for adjacency
(document_id, BELONGS_TO, domain)          — from metadata
(document_id, AUTHORED_BY, author)         — from metadata, if available
(abbreviation, ABBREVIATION_OF, expansion) — from abbreviation_context
```

For each cross-reference detected by Node 9:
```
(document_id, REFERENCES, target_reference)       — explicit refs
(document_id, DEPENDS_ON, target_reference)        — dependency refs
(document_id, CITES_STANDARD, standard_reference)  — standards refs
(document_id, SUPERSEDES, previous_version)         — version refs
```

**Stage 2: Entity consolidation** (deterministic + fuzzy matching). Merges entities from all chunks:

- Exact-match deduplication (case-insensitive)
- Abbreviation resolution: if an entity matches a known abbreviation, both forms are linked via an ABBREVIATION_OF triple
- Fuzzy matching for near-duplicates: "Synopsys DFT Compiler" ≈ "DFTC" ≈ "DFT Compiler" (using the abbreviation dictionary and Jaccard similarity on token sets)
- Each unique entity becomes a graph node with a type (ENTITY, CONCEPT, SPEC_VALUE, etc.)

**Stage 3: LLM-based relationship extraction** (optional, per `config.knowledge_graph.extract_relationships`). For each chunk, sends the content with entities and abbreviation dictionary to the LLM requesting subject-predicate-object triples:

```
Prompt: Given this engineering text and the extracted entities, identify
relationships between concepts. Return JSON array of triples.

Rules:
- Subject and object must be entities, concepts, or specification values
  found in or directly implied by the text
- Predicates should use controlled vocabulary: requires, specifies,
  configures, validates, implements, constrains, depends_on, input_to,
  output_of, measured_by, defined_in
- Include numerical specification relationships (e.g., "core_voltage
  SPECIFIES 1.8V")
- Do not infer relationships not supported by the text
- Maximum {max_triples_per_chunk} triples per chunk
```

The LLM response is parsed as JSON. Each triple is validated (subject and object must appear in the chunk or entity list) and assigned a confidence score. Invalid triples are discarded.

**Deterministic fallback:** If the LLM fails, Stage 3 is skipped entirely. The knowledge graph is built from structural triples (Stage 1) and entity consolidation (Stage 2) only. This produces a useful but less rich graph.

**Triple ID generation:** All triple IDs are deterministic:
```python
triple_id = deterministic_id([document_id, subject, predicate, object_value])
```

**Output:** PipelineDocument with `kg_triples` populated as a list of KGTriple objects.

### 4.11 Node 11 — Quality Validation

File: `pipeline/nodes/quality_validation.py`

**Purpose:** Filters out low-quality, duplicate, and boilerplate chunks before they consume embedding compute and storage.

**Input validation:** Always passes (defensive — quality should run even on marginal input).

**Processing logic — four stages:**

**Stage 1: Minimum token filtering.** Chunks with `token_count` below `config.quality.min_chunk_tokens` (default 20) are removed. This catches stray page numbers, single-word fragments, and empty chunks from splitting artefacts.

**Stage 2: Near-duplicate detection.** Uses 3-word shingling and Jaccard similarity to identify near-duplicate chunks:

- For each chunk, creates a set of 3-word shingles (sliding window across the tokenised content).
- Compares each chunk's shingle set against all previously seen sets.
- If Jaccard similarity exceeds `config.quality.max_duplicate_similarity` (default 0.95), the chunk is flagged as `is_duplicate = True`.
- Duplicates are removed from the chunk list. The first occurrence is always kept.

**Stage 3: Quality scoring.** Each surviving chunk receives a `quality_score` between 0.0 and 1.0 based on a weighted heuristic over content signals. The score is stored in the chunk and carried through to Weaviate, enabling retrieval-time quality weighting.

```python
def compute_quality_score(chunk: Chunk, extraction_confidence: float) -> float:
    """
    Heuristic quality score. All weights are tunable — start with these
    defaults and adjust based on evaluation results (Section 15).
    """
    score = 0.5  # Baseline — an average chunk

    tokens = chunk.token_count
    text = chunk.content

    # --- Positive signals (reward content-rich chunks) ---

    # Technical term density: count matches against a curated set of
    # engineering indicators (voltage units, frequencies, tool names, etc.)
    tech_terms = len(re.findall(
        r'\b(?:MHz|GHz|ns|ps|mV|μA|Vdd|Vss|JTAG|BIST|ATPG|DRC|LVS|OCV|PVT)\b',
        text, re.IGNORECASE
    ))
    score += min(tech_terms * 0.02, 0.15)  # Cap at +0.15

    # Numerical density: specifications, measurements, register values
    numbers = len(re.findall(r'\b\d+\.?\d*\s*(?:V|mA|MHz|nm|°C|Ω|%)\b', text))
    score += min(numbers * 0.03, 0.15)  # Cap at +0.15

    # Hex addresses / register offsets (strong signal for ASIC docs)
    hex_values = len(re.findall(r'0x[0-9A-Fa-f]{2,8}', text))
    score += min(hex_values * 0.02, 0.10)  # Cap at +0.10

    # Structured content bonus: tables, numbered lists, code blocks
    if chunk.content_type in (ContentType.TABLE, ContentType.CODE):
        score += 0.10

    # --- Negative signals (penalise low-information chunks) ---

    # Very short chunks (likely fragments or artefacts)
    if tokens < 50:
        score -= 0.20
    elif tokens < 100:
        score -= 0.10

    # Excessive whitespace ratio (indicates extraction artefacts)
    whitespace_ratio = len(re.findall(r'\s', text)) / max(len(text), 1)
    if whitespace_ratio > 0.5:
        score -= 0.15

    # Boilerplate match (page headers, confidentiality notices)
    for pattern in boilerplate_patterns:
        if re.search(pattern, text):
            score -= 0.10
            break  # One penalty is enough

    # --- External signal: structure extraction confidence ---
    # Chunks from documents where Docling struggled get a penalty
    if extraction_confidence < 0.5:
        score -= 0.10

    return max(0.0, min(1.0, score))  # Clamp to [0.0, 1.0]
```

These weights are starting points, not gospel. The evaluation framework (Section 15) should be used to validate: if low-scoring chunks are frequently the correct retrieval result, the scoring formula needs recalibration.

**Stage 4: Adjacency link repair.** After removing low-quality and duplicate chunks, re-links `previous_chunk_id` and `next_chunk_id` to skip over removed chunks:
```python
surviving = [c for c in chunks if not c.is_duplicate and c.token_count >= min_tokens]
for i, chunk in enumerate(surviving):
    chunk.previous_chunk_id = surviving[i-1].chunk_id if i > 0 else None
    chunk.next_chunk_id = surviving[i+1].chunk_id if i < len(surviving)-1 else None
```

**Output:** PipelineDocument with low-quality and duplicate chunks removed, `quality_score` assigned, and adjacency links repaired.

### 4.12 Node 12 — Embedding & Storage (with Re-ingestion Cleanup)

File: `pipeline/nodes/embedding_storage.py`

**Purpose:** Cleans up previous document data on re-ingestion, validates embedding dimensions, generates vector embeddings for all chunks, and upserts them into Weaviate with full metadata.

**Input validation:** Requires at least one chunk.

**Processing logic:**

1. **Dry-run check.** If `config.dry_run` is True, skips all embedding and storage, sets status to COMPLETED, and returns.

2. **Re-ingestion cleanup.** If `document.is_reingestion` is True and `config.reingestion.cleanup_vectors` is True:
   ```
   CLEANUP SEQUENCE (fail-safe — cleanup must complete before insert proceeds):

   a. Query Weaviate for ALL objects where document_id == current document_id
   b. Collect all object UUIDs from the query result
   c. Batch-delete all collected UUIDs from Weaviate
   d. Log: "Deleted {count} previous chunks for document {document_id}"
   e. If cleanup_kg is also True, delete KG triples (deferred to Node 13)

   If cleanup fails:
   - Log the error
   - Set document.errors += "Re-ingestion cleanup failed: {reason}"
   - HALT this node (do not upsert new data on top of stale data)
   - This is the ONE case where the pipeline fails hard rather than continuing,
     because inserting new chunks alongside stale chunks creates data corruption
   ```

   **Important: this sequence is NOT truly atomic.** Between the delete (step c) and
   the subsequent insert (step 6 below), there is a window where the document has no
   data in Weaviate. For a single-user batch job this is acceptable — no queries are
   running during ingestion. For production deployments with concurrent queries, this
   window means a query during re-ingestion may return zero results for the affected
   document. See Section 7.3 for production mitigations (blue-green versioning,
   staging collections). The "fail-safe" guarantee here is narrower: if cleanup
   succeeds but insert fails, the document is absent (detectable and recoverable via
   re-run) rather than corrupted (stale + new chunks coexisting, undetectable).

   The delete-then-reinsert strategy is chosen over update-in-place because:
   - Chunk boundaries change when content changes — there's no stable chunk ID to update
   - A document that previously produced 15 chunks may now produce 12 — the 3 orphans must be removed
   - Deterministic IDs help when content is unchanged (same ID = same chunk), but when content changes, the old IDs become orphans

3. **Embedding dimension validation.** On the first batch, checks that the actual embedding vector dimension matches `config.embedding.dimension`:
   ```python
   first_embedding = embed(chunks[0].enriched_content)
   if len(first_embedding) != config.embedding.dimension:
       raise EmbeddingDimensionMismatch(
           f"Model produced {len(first_embedding)}d vectors, "
           f"config expects {config.embedding.dimension}d. "
           f"Update config.embedding.dimension or change model."
       )
   ```
   This prevents silently storing mismatched vectors when someone changes the embedding model without updating the config. The error is fatal for this node — mismatched dimensions corrupt the vector index.

4. **Embedding generation.** Iterates over chunks in batches of `config.embedding.batch_size` (default 32). For each chunk, the `enriched_content` is embedded — this contains the raw content with boundary context but WITHOUT metadata headers (see Node 7). The document prefix (`config.embedding.document_prefix`, default empty for BGE) is prepended if configured. The BYOM (Bring Your Own Model) approach means the pipeline generates embeddings using a local or API model (default: BAAI/bge-large-en-v1.5 via HuggingFace sentence-transformers) and passes pre-computed vectors to Weaviate, rather than using Weaviate's built-in vectoriser modules.

   **Why BYOM over Weaviate's built-in vectorisers:** Weaviate offers built-in vectoriser modules (e.g., `text2vec-transformers`, `text2vec-openai`) that embed content automatically on insert and query. BYOM was chosen for three reasons: (1) **Model control** — built-in modules pin you to the models Weaviate bundles or the API it integrates with; BYOM lets us use any model from HuggingFace, swap models without redeploying Weaviate, and version-lock the exact model checkpoint. (2) **Prefix handling** — asymmetric models (BGE, E5) require different prefixes for documents vs queries; BYOM gives the pipeline full control over what text is actually embedded, including boundary context composition and prefix injection. Weaviate's built-in modules embed the stored text property as-is, with no hook to prepend prefixes or inject boundary context. (3) **Offline/air-gapped deployment** — ASIC design environments often have restricted internet access; BYOM with a local HuggingFace model works entirely offline, whereas Weaviate's API-based vectorisers require outbound connectivity. The tradeoff is implementation complexity: the pipeline must manage model loading, batching, and the embedding dimension must be manually kept in sync with the config (hence the dimension validation in step 3).

   Note: the `query_prefix` field is used by the retrieval layer (not this pipeline) when embedding user queries.

5. **Collection creation.** On first run, creates the Weaviate collection with the full property schema (see below), HNSW vector index configuration, AND hybrid search configuration. The `vectorizer_config` is set to `none()` (BYOM mode). BM25 indexing is enabled on `keywords`, `entities`, `document_keywords`, and `content` fields to support hybrid search at retrieval time. The HNSW parameters (`ef_construction=128`, `max_connections=64`) are configurable.

6. **Batch upsert.** Uses Weaviate's dynamic batch API to insert chunks with their properties and pre-computed vectors. Each chunk is identified by its deterministic UUID (`chunk_id`). Because IDs are deterministic, re-inserting an identical chunk is a no-op (upsert semantics).

7. **Status update.** Sets `processing_status` to COMPLETED.

**Weaviate schema (32 properties):**

| Property | Type | Source |
|----------|------|--------|
| content | TEXT | Chunk raw content (BM25 indexed) |
| enriched_content | TEXT | Content with boundary context (embedded) |
| context_header | TEXT | Metadata header for display (NOT embedded) |
| chunk_index | INT | Position in document |
| content_type | TEXT | text, table, code, figure, etc. |
| chunking_method | TEXT | llm_semantic / fallback_recursive / table_atomic |
| token_count | INT | Actual token count (from model tokeniser) |
| quality_score | NUMBER | 0.0–1.0 quality metric |
| content_hash | TEXT | SHA-256 hash of chunk content |
| section_path | TEXT | "Section A > Subsection B" |
| page_numbers | INT_ARRAY | Source page numbers |
| keywords | TEXT_ARRAY | Chunk-level keywords (BM25 indexed) |
| entities | TEXT_ARRAY | Named entities (BM25 indexed) |
| linked_figures | TEXT_ARRAY | Figure IDs referenced |
| linked_tables | TEXT_ARRAY | Table IDs referenced |
| previous_chunk_id | TEXT | Adjacent chunk (before) |
| next_chunk_id | TEXT | Adjacent chunk (after) |
| document_id | TEXT | Parent document UUID |
| document_title | TEXT | Document title |
| document_domain | TEXT | dft, verification, etc. |
| document_type | TEXT | specification, guide, etc. |
| source_path | TEXT | Original file path |
| source_format | TEXT | pdf, docx, pptx, xlsx, etc. |
| document_keywords | TEXT_ARRAY | Document-level keywords (filterable, NOT BM25 indexed — see Section 14.2) |
| document_summary | TEXT | LLM-generated summary |
| document_content_hash | TEXT | SHA-256 of entire document |
| extraction_confidence | NUMBER | 0.0–1.0 from structure quality check |
| review_tier | TEXT | fully_reviewed / partially_reviewed / self_reviewed |
| review_status | TEXT | approved / in_review / draft / submitted |
| reviewed_by | TEXT_ARRAY | List of reviewer identifiers |
| review_date | TEXT | ISO 8601 timestamp of last review |
| retrieval_feedback_score | NUMBER | 0.0–1.0, updated by retrieval layer (default null) |
| ingestion_timestamp | TEXT | ISO 8601 timestamp |
| pipeline_version | TEXT | "2.1.0" |

**Embedding provider swappability:** The `_get_embedder()` factory method dispatches on `config.embedding.provider`. Supported providers: HuggingFace (local), OpenAI (API), Cohere (API). Adding a new provider requires one `elif` block.

**Vector store swappability:** To replace Weaviate with Pinecone or Qdrant, implement a new storage node that inherits from BaseNode, implements `process()` and `validate_input()`, and register it in `graph.py` under the "embedding_storage" node name.

**Output:** PipelineDocument with `processing_status = COMPLETED`, chunks stored in Weaviate, old data cleaned up on re-ingestion.

### 4.13 Node 13 — Knowledge Graph Storage (Conditional — NEW)

File: `pipeline/nodes/kg_storage.py`

**Purpose:** Persists knowledge graph triples to the graph store and cleans up previous triples on re-ingestion.

**Routing condition:** Runs only if `config.knowledge_graph.enabled` is True and `kg_triples` is non-empty.

**Input validation:** Requires at least one triple in `document.kg_triples`.

**Processing logic:**

1. **Dry-run check.** If `config.dry_run`, log triple count and return.

2. **Re-ingestion KG cleanup.** If `document.is_reingestion` and `config.reingestion.cleanup_kg`:
   ```
   CLEANUP SEQUENCE:
   
   Weaviate xref provider:
     a. Query for all KG objects where source_document_id == document_id
     b. Batch-delete all found objects
     c. Remove all cross-references pointing FROM deleted objects
   
   Neo4j provider (two-phase cleanup to protect shared nodes):
     a. Phase 1 — Delete edges owned by this document:
        MATCH ()-[r]->() WHERE r.source_document_id = $document_id DELETE r
        This removes all relationships originating from this document while
        preserving shared concept nodes that other documents reference.
     b. Phase 2 — Garbage-collect orphaned nodes:
        MATCH (n) WHERE n.source_document_id = $document_id
        AND NOT EXISTS { MATCH (n)<-[r]-() WHERE r.source_document_id <> $document_id }
        AND NOT EXISTS { MATCH (n)-[r]->() WHERE r.source_document_id <> $document_id }
        DELETE n
        This deletes nodes that were ONLY referenced by this document. Shared
        nodes (e.g., a "DFT" concept also referenced by other documents) are
        preserved because they still have edges from other documents.
   
   If cleanup fails → halt (same rationale as vector cleanup — no stale triples)
   ```

3. **Triple storage.** Dispatches on `config.knowledge_graph.provider`:

   **Weaviate cross-reference mode:**
   - Creates a `kg_triples` collection with properties: `triple_id`, `subject`, `subject_type`, `predicate`, `predicate_type`, `object`, `object_type`, `source_chunk_id`, `source_document_id`, `confidence`
   - Each triple is stored as a Weaviate object with deterministic UUID
   - Cross-references are created between triple objects and chunk objects for provenance
   
   **Neo4j mode:**
   - Creates nodes for subjects and objects with labels matching their KGNodeType
   - Creates typed edges matching the predicate
   - Each edge carries `source_document_id`, `source_chunk_id`, and `confidence` as properties
   - Uses MERGE (not CREATE) to avoid duplicating shared nodes across documents

4. **Status logging.** Records triple count, node count, and edge count.

**Output:** PipelineDocument unchanged (triples already stored externally). Processing log updated.

---

## 5. Review Tier System

### 5.1 Tier Definitions

```
TIER 1: FULLY REVIEWED
├── Content: Mature engineering processes, approved specifications, signed-off runbooks
├── Review: Formal review + domain lead sign-off
├── Retrieval: Default search space — always included
└── Trust: Authoritative — can be cited in design decisions

TIER 2: PARTIALLY REVIEWED
├── Content: Architecture diagrams in active projects, draft specs, WIP processes
├── Review: At least 1 peer review, not yet signed off
├── Retrieval: Included with visual "[DRAFT]" indicator in results
└── Trust: Informational — verify before using in designs

TIER 3: SELF-REVIEWED
├── Content: Utility scripts, personal notes, boilerplate solutions, tribal knowledge
├── Review: Author self-certifies
├── Retrieval: Opt-in search space — user must expand search
└── Trust: Community — use at own discretion, not design-authoritative
```

### 5.2 Review Lifecycle

```
SELF_REVIEWED ──[submit for review]──▶ IN_REVIEW ──[peer approves]──▶ PARTIALLY_REVIEWED
                                                  ──[peer rejects]──▶ SELF_REVIEWED
                                                                       (with feedback)

PARTIALLY_REVIEWED ──[domain lead sign-off]──▶ FULLY_REVIEWED
                   ──[major revision]──▶ SELF_REVIEWED (re-enter cycle)

FULLY_REVIEWED ──[document re-ingested with changes]──▶ PARTIALLY_REVIEWED
                 (auto-demote, review.auto_demoted = True)
```

Auto-demotion on re-ingestion is critical: when a fully-reviewed document is modified and re-ingested, its review tier drops to `partially_reviewed` until re-approved. This prevents stale approvals on changed content. The `auto_demoted` flag signals to reviewers that this was a system action, not a manual demotion.

### 5.3 Review Tier Update API

Review tier changes do NOT require re-ingestion. They are property updates on existing Weaviate objects:

```python
def update_review_tier(document_id: str, new_tier: ReviewTier, 
                       reviewer: str, notes: str = ""):
    """Update review tier for all chunks of a document without re-embedding."""
    chunks = collection.query(
        filters=Filter.by_property("document_id").equal(document_id),
        return_properties=["chunk_id"]
    )
    for chunk in chunks:
        collection.data.update(
            uuid=chunk.uuid,
            properties={
                "review_tier": new_tier.value,
                "review_status": "approved" if new_tier == "fully_reviewed" else "in_review",
                "reviewed_by": [...existing, reviewer],
                "review_date": datetime.utcnow().isoformat(),
            }
        )
```

### 5.4 Retrieval-Time Search Spaces

```python
# Default — fully reviewed only (authoritative results)
results = collection.query(
    vector=query_embedding,
    filters=Filter.by_property("review_tier").equal("fully_reviewed"),
    limit=10
)

# Expanded — include drafts (informational + authoritative)
results = collection.query(
    vector=query_embedding,
    filters=Filter.by_property("review_tier").contains_any(
        ["fully_reviewed", "partially_reviewed"]
    ),
    limit=10
)

# Full corpus — user explicitly opts in (all knowledge)
results = collection.query(
    vector=query_embedding,
    limit=10  # no filter
)
```

---

## 6. Domain Vocabulary System

### 6.1 Abbreviation Dictionary Format

```yaml
# domain_vocabulary.yaml
abbreviations:
  DFT:
    expansion: "Design for Testability"
    domain: "dft"
    context: "ASIC test methodology"
    related: ["ATPG", "scan chain", "BIST"]
  STA:
    expansion: "Static Timing Analysis"
    domain: "physical_design"
    related: ["timing closure", "setup time", "hold time"]
  OCV:
    expansion: "On-Chip Variation"
    domain: "physical_design"
    related: ["PVT", "AOCV", "POCV"]
  CDR:
    - expansion: "Critical Design Review"
      domain: "general"
      context: "project milestone"
    - expansion: "Clock Data Recovery"
      domain: "analog"
      context: "SerDes circuits"
  IP:
    expansion: "Intellectual Property"
    domain: "general"
    context: "reusable design block, not Internet Protocol"

compound_terms:
  - "clock domain crossing"
  - "scan chain"
  - "power domain"
  - "timing closure"
  - "design rule check"
  - "electromigration"
  - "signal integrity"
```

### 6.2 Vocabulary Injection into LLM Prompts

When `config.vocabulary.inject_into_prompts` is True, every LLM call in the pipeline (Nodes 3, 5, 6, 8, 9, 10) prepends a vocabulary context block to the system prompt:

```
DOMAIN VOCABULARY (use these definitions when interpreting abbreviations):
- DFT: Design for Testability (ASIC test methodology)
- STA: Static Timing Analysis
- OCV: On-Chip Variation
- IP: Intellectual Property (reusable design block)
- CDR: Critical Design Review (project milestone) OR Clock Data Recovery (SerDes)
  → Disambiguate using document domain: if domain=analog, use Clock Data Recovery

COMPOUND TERMS (do not split these across chunk boundaries):
- clock domain crossing, scan chain, power domain, timing closure
```

The vocabulary is truncated to `config.vocabulary.max_prompt_terms` (default 50) most relevant terms, ranked by frequency in the current document.

### 6.3 Auto-Detection of New Abbreviations

Node 2 (Structure Detection) scans for abbreviation patterns:

```python
# Pattern 1: "Full Name (ABBR)"
re.compile(r'([A-Z][a-z]+(?: [A-Z][a-z]+)+)\s*\(([A-Z]{2,6})\)')

# Pattern 2: "ABBR: Full Name" or "ABBR — Full Name"  
re.compile(r'([A-Z]{2,6})\s*[:\—–-]\s*([A-Z][a-z]+(?: [A-Za-z]+)+)')

# Pattern 3: Abbreviation tables (markdown table with "Abbreviation" header)
```

**Regex limitations and tuning guidance:** The auto-detection patterns are starting points optimised for common engineering document conventions. Known limitations:
- Pattern 1 requires each word to start uppercase followed by lowercase, so it misses hyphenated expansions like "On-Chip Variation (OCV)" and all-caps terms like "AUTOMATIC TEST PATTERN GENERATION (ATPG)".
- Pattern 2 assumes 2-6 uppercase characters, missing longer abbreviations (e.g., "SERDES", "DFTMAX").
- Neither pattern handles abbreviations defined in table cells without an "Abbreviation" header.

Recommended improvement: supplement regex patterns with a fuzzy matching pass that compares candidate uppercase sequences against the domain vocabulary dictionary for near-matches. The auto-detection results should be reviewed during the first batch ingestion and the patterns tuned for the specific corpus.

Auto-detected abbreviations are added to `document.abbreviation_context` with `source: "document:{document_id}"`. They are available for the current pipeline run. Optionally, they can be written back to the master vocabulary file for future runs (controlled by a config flag).

---

## 7. Re-ingestion Strategy

### 7.1 Design Goals

1. **No stale data.** When a document is re-ingested, every trace of the previous version (chunks, embeddings, KG triples, cross-references) is removed before new data is inserted.
2. **No duplicates.** Re-ingesting an unchanged document produces no new data and no side effects.
3. **Review tier awareness.** Changed documents are auto-demoted; unchanged documents preserve their review tier.
4. **Atomic cleanup.** Either all old data is removed and all new data is inserted, or the operation fails cleanly with no partial state.

### 7.2 Re-ingestion Flow

```
Document arrives for processing
       │
       ▼
Node 1: Compute content_hash (SHA-256 of file bytes)
       │
       ▼
Node 1: Derive document_id = deterministic_id(source_path)
       │
       ▼
Node 1: Query Weaviate for existing chunks with this document_id
       │
       ├── [no existing chunks] ──▶ First ingestion (is_reingestion = False)
       │                            Proceed normally
       │
       └── [existing chunks found]
            │
            ├── [content_hash matches AND strategy = "skip_unchanged"]
            │   └──▶ SKIP — document unchanged, preserve all existing data
            │         Set status = SKIPPED, return
            │
            └── [content_hash differs OR strategy = "delete_and_reinsert"]
                │
                ├── Set is_reingestion = True
                ├── Auto-demote review tier (if configured)
                └── Proceed through pipeline normally
                     │
                     ... (Nodes 2-11 process as usual) ...
                     │
                     ▼
               Node 12: CLEANUP PHASE
                     │
                     ├── Delete ALL old chunks from Weaviate where
                     │   document_id matches (batch delete by UUID)
                     │
                     ├── If cleanup fails → HALT (no partial state)
                     │
                     └── INSERT new chunks (upsert by deterministic ID)
                          │
                          ▼
               Node 13: KG CLEANUP PHASE
                     │
                     ├── Delete ALL old KG triples from graph store
                     │   where source_document_id matches
                     │
                     ├── If cleanup fails → HALT
                     │
                     └── INSERT new KG triples
                          │
                          ▼
                        [END]
```

### 7.3 Cleanup Safety

The delete-then-reinsert strategy has a risk window: between the delete and the insert, the document's data is absent from the store. For a single-user batch job this is acceptable. For a production system with concurrent queries, two mitigations:

1. **Blue-green approach.** Insert new chunks with a `version` tag, then delete old chunks, then update the `version` tag to "current". Queries always filter on `version = "current"`.
2. **Staging collection.** Insert into a staging Weaviate collection, then swap (rename collections). This is atomic at the collection level.

For initial version, the simple delete-then-reinsert is sufficient. The blue-green approach is documented as an enhancement for production deployments with high query concurrency.

---

## 8. Workflow Orchestration

### 8.1 LangGraph Integration

File: `pipeline/graph.py`

The pipeline uses LangGraph's StateGraph to define the processing DAG. LangGraph provides typed state passing between nodes, conditional routing based on state inspection, compilation to an executable workflow, and automatic state merging for parallel execution paths.

The `build_pipeline_graph()` function instantiates all 13 nodes with the pipeline configuration, registers them in the graph, defines edges (9 direct, 4 conditional), and sets the entry point.

### 8.2 Conditional Routing Functions

Four pure functions inspect the pipeline state and return the name of the next node:

**Why routing checks the processing log, not the config directly:** The routing functions read `document.processing_log` for skip flags rather than checking `config.skip_*` booleans directly. This is intentional indirection for two reasons: (1) **Runtime skip decisions.** A node can dynamically add a skip flag to the processing log based on what it discovers. For example, if Node 1 (Ingestion) detects that a document is a one-page plain-text file with no cross-references, it could add `"skip_cross_references"` to the log even though the config didn't request it. This enables data-driven routing that the static config alone cannot express. (2) **Auditability.** The processing log is the single source of truth for what happened during a run. If routing decisions are derived from the log, the log is self-explanatory — you can reconstruct the full execution path from it alone, without also needing the config that was active at the time. The `BaseNode.__call__` method writes the config-driven skip flags into the processing log at pipeline start (during the "started" log entry for Node 1), so the initial config flags are always present in the log for routing functions to find.

**should_process_multimodal(state)** — Checks `document.structure.has_figures` and the presence of at least one figure. Returns `"multimodal_processing"` if figures exist, otherwise `"text_cleaning"`. Handles the case where structure analysis failed entirely (no structure object) by defaulting to text cleaning.

**should_refactor(state)** — Scans `document.processing_log` for a config entry with `details == "skip_refactoring"`. If found, routes directly to `"llm_chunking"`. Otherwise routes to `"document_refactoring"`.

**should_extract_xrefs(state)** — Same pattern as refactoring: scans for `"skip_cross_references"` in the processing log. Routes to `"knowledge_graph_extraction"` or `"quality_validation"` accordingly.

**should_build_kg(state)** — Scans for `"skip_knowledge_graph"` in the processing log. Routes to `"quality_validation"` (skip) or `"knowledge_graph_extraction"` (process).

### 8.3 Edge Topology

```
document_ingestion ──────────────────────▶ structure_detection
structure_detection ──[conditional]──────▶ multimodal_processing | text_cleaning
multimodal_processing ───────────────────▶ text_cleaning
text_cleaning ──[conditional]────────────▶ document_refactoring | llm_chunking
document_refactoring ────────────────────▶ llm_chunking
llm_chunking ───────────────────────────▶ chunk_enrichment
chunk_enrichment ───────────────────────▶ metadata_generation
metadata_generation ──[conditional]──────▶ cross_reference_extraction | kg_decision_point (conditional)
cross_reference_extraction ──[conditional]▶ knowledge_graph_extraction | quality_validation
kg_decision_point ──[conditional]────────▶ knowledge_graph_extraction | quality_validation
knowledge_graph_extraction ──────────────▶ quality_validation
quality_validation ─────────────────────▶ embedding_storage
embedding_storage ──────────────────────▶ kg_storage
kg_storage ─────────────────────────────▶ END
```

**Double-skip scenario:** When both cross-references AND knowledge graph are skipped, the routing chain resolves as follows:
1. `metadata_generation` → `should_extract_xrefs()` finds `"skip_cross_references"` → routes to `should_build_kg()` (NOT directly to `knowledge_graph_extraction`)
2. `should_build_kg()` finds `"skip_knowledge_graph"` → routes to `quality_validation`

This means the conditional edge from `metadata_generation` does NOT route directly to `knowledge_graph_extraction` as a node — it routes to the *second conditional function* (`should_build_kg`), which then decides between `knowledge_graph_extraction` and `quality_validation`. In LangGraph terms, the implementation looks like:

```python
# After metadata_generation, the first conditional checks xrefs
graph.add_conditional_edges("metadata_generation", should_extract_xrefs, {
    "cross_reference_extraction": "cross_reference_extraction",
    "skip_to_kg_decision": "kg_decision_point",  # NOT directly to kg_extraction
})

# The KG decision point is a separate conditional (not a processing node)
graph.add_conditional_edges("kg_decision_point", should_build_kg, {
    "knowledge_graph_extraction": "knowledge_graph_extraction",
    "quality_validation": "quality_validation",
})

# Cross-ref extraction also feeds into the KG decision
graph.add_conditional_edges("cross_reference_extraction", should_build_kg, {
    "knowledge_graph_extraction": "knowledge_graph_extraction",
    "quality_validation": "quality_validation",
})
```

This ensures every combination of skip flags produces a valid path to `quality_validation`.

---

## 9. Entry Points and CLI

### 9.1 Command-Line Interface

File: `main.py`

```bash
# Single file
python main.py /path/to/document.pdf

# Batch processing (recursive directory scan)
python main.py /path/to/docs/ --batch

# Dry run (full pipeline, no Weaviate/KG writes)
python main.py /path/to/document.pdf --dry-run

# Skip expensive steps
python main.py /path/to/document.pdf --skip-refactoring --skip-multimodal --skip-xrefs --skip-kg

# Custom config + metadata override
python main.py /path/to/document.pdf --config config.json --domain dft --doc-type runbook

# Review tier override
python main.py /path/to/document.pdf --review-tier fully_reviewed

# Re-ingestion (force, even if content unchanged)
python main.py /path/to/document.pdf --force-reingestion

# Domain vocabulary
python main.py /path/to/document.pdf --vocabulary domain_vocabulary.yaml

# Debug logging
python main.py /path/to/document.pdf --log-level DEBUG
```

### 9.2 Programmatic API

```python
from pipeline.config import PipelineConfig
from pipeline.models import PipelineDocument, DocumentMetadata, ReviewMetadata, ReviewTier
from pipeline.graph import compile_pipeline

config = PipelineConfig()
config.llm.provider = "anthropic"
config.llm.model_name = "claude-sonnet-4-20250514"
config.vocabulary.dictionary_path = "domain_vocabulary.yaml"
config.knowledge_graph.enabled = True

pipeline = compile_pipeline(config)

doc = PipelineDocument(
    metadata=DocumentMetadata(
        source_path="/data/specs/dft_scan_chain.pdf",
        domain="dft",
        doc_type="specification",
    ),
    review=ReviewMetadata(
        tier=ReviewTier.PARTIALLY_REVIEWED,
    )
)

result = pipeline.invoke({"document": doc})
final = result["document"]

print(f"Chunks: {len(final.chunks)}")
print(f"KG Triples: {len(final.kg_triples)}")
print(f"Cross-refs: {len(final.cross_references)}")
print(f"Re-ingestion: {final.is_reingestion}")
print(f"Review tier: {final.review.tier}")
print(f"Errors: {final.errors}")
```

### 9.3 Batch Processing

The `process_batch()` function recursively scans a directory for files with matching extensions (.pdf, .docx, .md, .html, .txt, .rst, .pptx, .xlsx, .tcl), processes each through the full pipeline independently, and reports a summary of successes, failures, skips (unchanged documents), re-ingestions, and low-confidence extractions flagged for manual review. Individual file failures do not halt the batch.

### 9.4 Review Tier Management (non-pipeline)

```python
from pipeline.review import update_review_tier, get_review_history

# Promote a document without re-ingesting
update_review_tier(
    document_id="abc-123-def",
    new_tier=ReviewTier.FULLY_REVIEWED,
    reviewer="jane.smith",
    notes="Approved after DFT team review"
)

# Check review history
history = get_review_history(document_id="abc-123-def")
```

---

## 10. Error Handling Strategy

### 10.1 Node-Level Error Isolation

Every node's execution is wrapped in the `BaseNode.__call__` try/except block. If a node raises any exception, the pipeline does not crash. Instead, the error message is appended to `document.errors`, a "failed" log entry is recorded, and the document continues to the next node with whatever state it had before the failure.

**Exception 1: Re-ingestion cleanup failure.** If Node 12 (Embedding & Storage) fails to delete old chunks during re-ingestion, the node halts and does NOT proceed to insert new data. This is the one case where fail-hard is correct — inserting new chunks alongside stale chunks creates data corruption that is worse than a failed ingestion. The document arrives at END with `processing_status = FAILED` and a clear error message.

**Exception 2: Re-ingestion with upstream failure (data loss prevention).** If `document.is_reingestion` is True but the document arrives at Node 12 with zero chunks (because an upstream node like chunking failed), Node 12 MUST NOT proceed with cleanup. Deleting old chunks and then having nothing to insert would remove the document from the vector store entirely — a data loss scenario that is worse than keeping the stale version.

Node 12's input validation implements this guard:
```python
if document.is_reingestion and len(document.chunks) == 0 and len(document.errors) > 0:
    # Upstream failure during re-ingestion — preserve existing data
    log.error("Re-ingestion aborted: upstream failures produced 0 chunks. "
              "Existing data preserved. Errors: {document.errors}")
    document.processing_status = ProcessingStatus.FAILED
    return document  # Skip cleanup AND insert — old data stays intact
```

This ensures that a re-ingestion run with processing failures is a no-op against the vector store, preserving the previous (stale but functional) version of the document.

This means a document can complete the pipeline with partial results. For example, if the LLM chunking node fails, the document will have no chunks, which causes downstream nodes (enrichment, metadata, quality) to skip via their input validation, and the storage node will also skip (no chunks to store). The document arrives at END with errors recorded and `processing_status = FAILED`.

### 10.2 LLM Call Fallbacks

Every node that makes LLM calls has a deterministic fallback:

| Node | LLM Path | Fallback |
|------|----------|----------|
| LLM Chunking | Semantic chunking via JSON prompt | Recursive character splitter on paragraph/sentence boundaries |
| Document Refactoring | Multi-pass agentic refactoring | Return original text unchanged |
| Metadata Generation | LLM keyword/entity extraction | TF-IDF frequency-based keyword extraction with stop word filtering |
| Cross-Reference Extraction | LLM implicit reference detection | Regex-only extraction (5 compiled patterns) |
| Multimodal Processing | VLM image-to-text | Figure recorded without description (confidence = 0.0) |
| Knowledge Graph Extraction | LLM relationship extraction | Structural triples only (deterministic, no LLM) |

### 10.3 JSON Response Reliability

Getting LLMs to return valid, parseable JSON consistently requires a multi-layer strategy. The pipeline addresses this at both the **prompt engineering** level (prevent bad JSON) and the **parsing** level (survive bad JSON):

**Prompt-level strategies (prevention):**

1. **Structured output modes (preferred).** When using OpenAI models, use `response_format={"type": "json_object"}` (JSON mode) or function calling with a Pydantic schema. When using Anthropic, use tool use with an input schema. These modes constrain the model's output to valid JSON at the token sampling level, making malformed responses nearly impossible. LangChain exposes this via `model.with_structured_output(schema)`.

2. **Few-shot examples in prompts.** For Ollama/local models that don't support structured output modes, include 1-2 concrete JSON examples in the prompt. Show the exact format expected, with realistic engineering content — not placeholder text. The model mirrors the structure it sees.

3. **Schema specification in system prompt.** State the expected JSON schema explicitly: field names, types, and constraints. Example: `"Return a JSON array where each element has: content (string), topic (string), content_type (one of: text, table, code, figure)."` Avoid ambiguous descriptions like "return the chunks as JSON."

4. **"JSON only" instruction.** End the prompt with an explicit constraint: `"Return ONLY the JSON array. Do not include any text before or after the JSON."` This reduces the frequency of models wrapping JSON in conversational text.

**Parsing-level strategies (survival):**

All LLM responses that expect JSON are parsed through a defensive parser that:
1. Strips markdown code fences (`` ```json ... ``` ``) — models frequently wrap JSON in code blocks even when told not to
2. Strips any leading/trailing non-JSON text (find first `[` or `{`, last `]` or `}`)
3. Attempts `json.loads()`
4. On failure, returns a node-specific safe default (`{"passed": False}` for fact-checks, `{"references": []}` for cross-refs, `{"triples": []}` for KG extraction, empty dict for metadata)
5. Logs the raw response and parse error for debugging

This prevents a single malformed LLM response from crashing a node. The fallback defaults are designed to trigger the node's deterministic fallback path (e.g., a failed chunking JSON parse triggers recursive splitting).

### 10.4 Processing Log

Every node writes timestamped entries to `document.processing_log`:

```json
{"node": "LLMChunkingNode", "status": "started", "details": "strategy=llm_driven, target_size=512, model=gpt-4o-mini", "timestamp": "2025-01-15T10:30:00.000000"}
{"node": "LLMChunkingNode", "status": "completed", "details": "chunks_produced=12, llm_calls=3, fallback_used=false, duration_ms=5123", "timestamp": "2025-01-15T10:30:05.123456"}
```

Possible status values: "started", "completed", "skipped" (input validation failed), "failed" (exception caught), "info" (configuration flags). This log is the primary debugging tool for pipeline issues and provides a complete timeline of every node's execution.

---

## 11. Non-Functional Requirements

### 11.1 Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Single document (10-page PDF, no refactoring) | < 60 seconds | End-to-end pipeline time |
| Single document (10-page PDF, with refactoring) | < 180 seconds | End-to-end pipeline time |
| Batch throughput | ≥ 20 documents/hour | Sequential processing |
| Embedding generation (32-chunk batch) | < 5 seconds | HuggingFace local, CPU |
| Weaviate upsert (50 chunks) | < 2 seconds | Localhost Weaviate |
| Re-ingestion cleanup (100 old chunks) | < 3 seconds | Batch delete |
| Pipeline startup (graph compilation) | < 1 second | No external service init |
| Embedding model first-load (cold start) | 10–30 seconds | One-time cost on first document. HuggingFace models are downloaded on first use (~1.3 GB for BGE-large) and loaded into memory. Subsequent documents reuse the cached model. First-run timing should exclude this cold start. If the model is pre-cached (offline deployment), first-load reduces to 5-10 seconds (memory allocation only). |
| Memory usage per document | < 2 GB | Peak RSS during processing |

### 11.2 Scalability Constraints

- **Concurrent documents:** The pipeline processes documents sequentially by default. `parallel_processing` flag is reserved for future multi-document parallelism but is not implemented.
- **Maximum document size:** Limited by LLM context window for refactoring/chunking nodes. Documents larger than ~100 pages should be split into section-level sub-documents before ingestion.
- **Weaviate collection size:** Tested up to 500,000 chunks. For larger deployments, consider Weaviate sharding or multiple collections per domain.
- **Knowledge graph size:** Weaviate cross-reference mode is practical up to ~1M triples. Neo4j mode scales to billions.

### 11.3 Acceptance Criteria

| Criterion | Threshold |
|-----------|-----------|
| Chunks produced per 10-page document | 8–30 (depending on density) |
| Quality score distribution | > 80% of chunks score ≥ 0.5 |
| Near-duplicate detection | 0 duplicate chunks in Weaviate after re-ingestion |
| Re-ingestion cleanup completeness | 0 orphaned chunks from previous version |
| Cross-reference detection | ≥ 90% of explicit references ("see Section X") detected |
| Abbreviation resolution | ≥ 95% of dictionary abbreviations correctly expanded |
| LLM fallback coverage | 100% of LLM nodes have deterministic fallback |
| Pipeline crash rate | 0 crashes (all errors caught and logged) |

### 11.4 Weaviate Schema Versioning

The Weaviate collection schema (32 properties currently) will evolve as features are added. Weaviate supports additive schema changes (adding new properties) without requiring re-ingestion — new properties are simply `null` on existing objects. However, the following schema changes DO require re-ingestion of affected documents:

- **Property removal.** Weaviate does not support removing properties from an existing collection. Workaround: the property remains in the schema but is no longer populated by new ingestions. A full re-ingestion is needed to clean it from all objects.
- **Property type change.** Changing a property's type (e.g., `TEXT` to `TEXT_ARRAY`) requires creating a new collection and migrating data.
- **Vector index configuration change.** HNSW parameters (`ef_construction`, `max_connections`, `distance_metric`) are set at collection creation and cannot be changed. Changing these requires creating a new collection and re-ingesting.
- **Embedding model change.** Changing the embedding model produces vectors in a different space. All existing vectors become incompatible. Full re-ingestion is required.

**Schema migration strategy:**

```
ADDITIVE CHANGE (new property, e.g., adding "reranker_score"):
  1. Update PipelineConfig and models.py with the new field
  2. Update Node 12 schema creation to include the new property
  3. New ingestions populate the field; existing objects have null
  4. No re-ingestion needed (null is handled by retrieval layer)

BREAKING CHANGE (model swap, index change, type change):
  1. Create new collection with suffix: engineering_documents_v2
  2. Run batch re-ingestion into new collection
  3. Validate with evaluation framework (Section 15)
  4. Swap active collection in retrieval layer config
  5. Delete old collection after validation period
```

The `pipeline_version` property stored on every chunk enables identifying which schema version produced the data.

---

## 12. External Dependencies

### 12.1 Runtime Dependencies

| Package | Version | Purpose | Required |
|---------|---------|---------|----------|
| langchain | ≥0.3.0 | LLM abstraction layer | Yes |
| langchain-core | ≥0.3.0 | Message types, parsers | Yes |
| langgraph | ≥0.2.0 | Workflow orchestration | Yes |
| langchain-openai | ≥0.2.0 | OpenAI provider | If using OpenAI |
| langchain-anthropic | ≥0.3.0 | Anthropic provider | If using Anthropic |
| langchain-ollama | ≥0.2.0 | Ollama provider | If using Ollama |
| langchain-huggingface | ≥0.1.0 | HuggingFace embeddings | If using HF embeddings |
| sentence-transformers | ≥3.0.0 | Local embedding models | If using HF embeddings |
| tokenizers | ≥0.20.0 | Accurate token counting from embedding model tokeniser | Yes |
| weaviate-client | ≥4.9.0 | Vector store client | Yes |
| docling | ≥2.0.0 | Document structure detection | Yes |
| python-pptx | ≥1.0.0 | PowerPoint text extraction | Yes |
| openpyxl | ≥3.1.0 | Excel spreadsheet extraction with table awareness | Yes |
| pydantic | ≥2.0 | Data validation | Yes |
| pyyaml | ≥6.0 | Vocabulary dictionary + model registry loading | Yes |
| neo4j | ≥5.0 | Graph database client | If using Neo4j KG |

### 12.2 Service Dependencies

| Service | Default Configuration | Purpose |
|---------|----------------------|---------|
| Weaviate | http://localhost:8080 | Vector storage, HNSW index, hybrid search, KG cross-refs |
| Ollama | http://localhost:11434 | Local LLM (default) and VLM (LLaVA) |
| Neo4j | bolt://localhost:7687 | Knowledge graph storage (optional, v2+) |

### 12.3 Lazy Initialisation

All external services are initialised lazily. The Weaviate client, embedding model, LLM instances, VLM instances, and Neo4j driver are created on first use within their respective nodes, not at pipeline construction time. This means pipeline construction and graph compilation are fast and dependency-free — you can build and inspect the graph without any external services running.

---

## 13. File Inventory

```
rag_embedding_pipeline/
├── main.py                                    # CLI entry point, batch processing
├── requirements.txt                           # Python dependencies
├── domain_vocabulary.yaml                     # Domain abbreviation dictionary
├── model_registry.yaml                        # Embedding/LLM model configuration registry 
├── eval_dataset.json                          # Evaluation dataset: queries + ground truth 
├── verify_pipeline.py                         # Verification test suite
├── evaluate_retrieval.py                      # Evaluation runner: Recall@K, MRR 
├── PIPELINE_PLAN.md                           # Original planning document
├── README.md                                  # Usage documentation
├── VERIFICATION_TEST_PLAN.md                  # Test plan documentation
└── pipeline/
    ├── __init__.py                            # Package init
    ├── config.py                              # PipelineConfig + 14 sub-configs 
    ├── config_validator.py                    # Startup config cross-validation
    ├── graph.py                               # LangGraph workflow definition
    ├── models.py                              # Data models (18+ dataclasses, 7 enums)
    ├── vocabulary.py                          # Vocabulary loading, auto-detection, prompt injection
    ├── review.py                              # Review tier management API (non-pipeline)
    ├── id_generation.py                       # Deterministic ID utilities
    ├── format_extractors/                     # Format-specific text extraction
    │   ├── __init__.py
    │   ├── pptx_extractor.py                  # PowerPoint → text (python-pptx)
    │   ├── xlsx_extractor.py                  # Excel → text with tables (openpyxl)
    │   ├── base_extractor.py                  # Extractor interface
    │   └── ...                                # ... more types of extractor
    └── nodes/
        ├── __init__.py                        # Node exports
        ├── base.py                            # BaseNode ABC
        ├── document_ingestion.py              # Node 1: File reading, hash, format extraction
        ├── structure_detection.py             # Node 2: Docling structure + quality check
        ├── multimodal_processing.py           # Node 3: VLM figure-to-text
        ├── text_cleaning.py                   # Node 4: Normalisation, integration
        ├── document_refactoring.py            # Node 5: Agentic refactoring loop
        ├── llm_chunking.py                    # Node 6: Semantic + table atomic chunking
        ├── chunk_enrichment.py                # Node 7: Boundary context, metadata headers
        ├── metadata_generation.py             # Node 8: Keywords, entities, summary
        ├── cross_reference_extraction.py      # Node 9: Regex + LLM reference detection
        ├── knowledge_graph_extraction.py      # Node 10: KG triple extraction
        ├── quality_validation.py              # Node 11: Dedup, scoring, filtering, link repair
        ├── embedding_storage.py               # Node 12: BYOM embedding, cleanup, Weaviate upsert
        └── kg_storage.py                      # Node 13: KG persistence + cleanup
```

---

## 14. BM25 Keyword Index Strategy

### 14.1 How the Pipeline Strengthens the Inverted Index

Weaviate automatically builds a BM25 inverted index on any property declared as `TEXT` or `TEXT_ARRAY`. No explicit configuration is needed — the index is populated on insert. However, relying solely on the raw chunk text for BM25 produces poor results for engineering documentation. Engineering text is dense with abbreviations, implicit context, and domain jargon that users search for using different surface forms than what appears in the document. The pipeline performs deliberate enrichment to close this vocabulary gap.

**The pipeline's BM25 enrichment chain:**

```
Raw chunk text (baseline BM25 signal)
   │
   ├── Node 5 (Refactoring): Abbreviation expansion inline
   │   "DFT" in original → "Design for Testability (DFT)" in chunk text
   │   BM25 now matches both "DFT" and "Design for Testability"
   │
   ├── Node 8 (Metadata): LLM-generated chunk-level keywords (5-10 terms)
   │   Chunk about "hold violation fix" → keywords: ["hold time", "timing closure", "STA"]
   │   BM25 now matches domain terms that don't appear verbatim in the text
   │
   ├── Node 8 (Metadata): Named entity extraction per chunk
   │   Extracts tool names ("Synopsys PrimeTime"), standards ("IEEE 1149.1"), IP blocks
   │   BM25 exact-matches on proper nouns that engineers search for
   │
   ├── Node 8 (Fallback): TF-IDF frequency-based keywords
   │   Purely statistical — no hallucination risk, always available
   │   Ensures every chunk has at least some BM25 signal even without LLM access
   │
   └── Section 6 (Vocabulary): Domain dictionary injected into LLM prompts
       Ensures LLM-generated keywords use canonical domain terms
       ("timing closure" not "timing convergence") for consistent BM25 matching
```

**Which Weaviate properties carry BM25 signal:**

| Property | BM25 Indexed | Content Source | Purpose |
|----------|-------------|----------------|---------|
| `content` | Yes | Raw chunk text | Baseline — matches terms that appear verbatim |
| `keywords` | Yes | LLM-generated chunk-level keywords | Vocabulary expansion — matches domain terms not in raw text |
| `entities` | Yes | LLM-extracted named entities | Proper noun matching — tools, standards, IP blocks |
| `document_keywords` | **No** (filterable only) | LLM-generated document-level keywords | Document-level filtering only — see Section 14.2 for rationale |

### 14.2 BM25 Enrichment Risks and Mitigations

Enriching the keyword index is not risk-free. Every term added to a chunk's BM25-indexed fields that is not actually discussed in the chunk creates a false positive — a search that returns an irrelevant result. In a mission-critical engineering context, false positives are dangerous: an engineer searching for "ATPG coverage" who gets a chunk about project timelines wastes time and loses trust in the system.

**Risk assessment by enrichment action:**

```
RISK LEVEL: HIGH
┌─────────────────────────────────────────────────────────────────────┐
│ Document-level keywords applied to chunk BM25 index                 │
│                                                                     │
│ Problem: A DFT document has document-level keywords ["scan chain",  │
│ "ATPG", "BIST", "fault coverage"]. The document also contains a     │
│ chunk about "project timeline and milestones". If document_keywords │
│ is BM25-indexed on every chunk, a search for "ATPG" returns the     │
│ timeline chunk. This is actively harmful — it pollutes precision    │
│ without helping recall (the actual ATPG chunks already have the     │
│ term in their content or chunk-level keywords).                     │
│                                                                     │
│ Mitigation: document_keywords is stored as a filterable property    │
│ but NOT BM25-indexed. Use it for document-level pre-filtering       │
│ ("find documents about ATPG") not chunk-level ranking.              │
└─────────────────────────────────────────────────────────────────────┘

RISK LEVEL: MODERATE
┌─────────────────────────────────────────────────────────────────────┐
│ LLM-generated chunk-level keywords                                  │
│                                                                     │
│ Problem: The LLM infers related terms that aren't discussed in the  │
│ chunk. A chunk about "scan chain insertion flow" gets tagged with   │
│ "ATPG", "BIST", "fault coverage" because the LLM knows these        │
│ concepts are related — but the chunk doesn't discuss them.          │
│                                                                     │
│ Mitigation: Keyword validation (see Section 14.3). Only retain      │
│ keywords that can be traced to the chunk content or the             │
│ abbreviation dictionary.                                            │
│                                                                     │
│ Problem: Over-broad keywords. If the LLM tags every DFT chunk with  │
│ "ASIC", "testing", "design", those terms lose discriminative power  │
│ in BM25 — they match everything and rank nothing.                   │
│                                                                     │
│ Mitigation: Reject keywords that appear in more than 30% of chunks  │
│ within a batch (corpus-level IDF filtering). If a keyword appears   │
│ everywhere, it's not useful for BM25 discrimination.                │
└─────────────────────────────────────────────────────────────────────┘

RISK LEVEL: MODERATE
┌─────────────────────────────────────────────────────────────────────┐
│ Named entity extraction                                             │
│                                                                     │
│ Problem: LLM extracts entities not mentioned in the chunk. A chunk  │
│ that says "the DFT tool" gets entity "Synopsys DFT Compiler" — an   │
│ inference, not a fact. BM25 now returns this chunk for a search     │
│ about DFT Compiler when the chunk may be about a different tool.    │
│                                                                     │
│ Mitigation: Same validation rule as keywords — entity must appear   │
│ in the chunk text or be a direct abbreviation expansion of          │
│ something in the text. Inferred entities are discarded.             │
└─────────────────────────────────────────────────────────────────────┘

RISK LEVEL: LOW
┌─────────────────────────────────────────────────────────────────────┐
│ Abbreviation expansion in refactoring (Node 5)                      │
│                                                                     │
│ Benefit: Converting "DFT" → "Design for Testability (DFT)" means    │
│ BM25 matches both forms. The expansion is inline in the text,       │
│ verifiable, and doesn't create phantom terms.                       │
│                                                                     │
│ Risk: Wrong expansion for ambiguous abbreviations (CDR = "Clock     │
│ Data Recovery" vs "Critical Design Review"). Mitigated by the       │
│ domain-aware disambiguation in the vocabulary system (Section 6).   │
│                                                                     │
│ Assessment: High reward, low risk. Keep as-is.                      │
└─────────────────────────────────────────────────────────────────────┘

RISK LEVEL: LOWEST
┌─────────────────────────────────────────────────────────────────────┐
│ TF-IDF fallback keywords                                            │
│                                                                     │
│ Purely statistical — no hallucination possible. Only risk is        │
│ surfacing frequent-but-uninformative terms ("section", "document"), │
│ handled by stop-word filtering. Safest BM25 signal.                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 14.3 Keyword Validation Rules

All LLM-generated keywords and entities must pass a validation gate before being stored in BM25-indexed fields. The validation is applied in Node 8 (Metadata Generation) after the LLM response is parsed:

```python
def validate_chunk_keywords(
    keywords: list[str],
    chunk_content: str,
    abbreviation_context: dict,
    content_type: ContentType
) -> list[str]:
    """
    Retain only keywords that are grounded in the chunk content.
    Reject hallucinated or over-broad terms.
    """
    validated = []
    content_lower = chunk_content.lower()

    for keyword in keywords:
        keyword_lower = keyword.lower()

        # Rule 1: Direct presence — keyword (or a stemmed variant) appears in chunk text
        if keyword_lower in content_lower:
            validated.append(keyword)
            continue

        # Rule 2: Abbreviation bridge — keyword is the expansion of an abbreviation
        # that appears in the chunk, or vice versa
        for abbr, entry in abbreviation_context.items():
            expansion = entry.get("expansion", "").lower() if isinstance(entry, dict) else str(entry).lower()
            if (keyword_lower == expansion and abbr.lower() in content_lower) or \
               (keyword_lower == abbr.lower() and expansion in content_lower):
                validated.append(keyword)
                break
        else:
            # Rule 3: Compound term overlap — for multi-word keywords, at least
            # 2 of the constituent words must appear in the chunk
            words = keyword_lower.split()
            if len(words) >= 2:
                matches = sum(1 for w in words if w in content_lower)
                if matches >= 2:
                    validated.append(keyword)
                    continue

            # Keyword is not grounded in the chunk — discard
            # Log for monitoring: track rejection rate to tune LLM prompts

    return validated
```

The same validation applies to named entities. This validation reduces keyword hallucination at the cost of occasionally dropping a genuinely useful associative term. The evaluation framework (Section 15) should measure the impact: if Recall@10 drops significantly with validation enabled, the rules may be too strict and Rule 3 thresholds can be relaxed.

### 14.4 Measuring BM25 Enrichment Quality

The evaluation framework (Section 15) must measure BM25 enrichment impact in isolation, not just as part of hybrid search. Recommended evaluation protocol:

```
For each enrichment action (keywords, entities, abbreviation expansion):

1. Run eval with enrichment ENABLED  → measure Recall@10, Precision@10, MRR
2. Run eval with enrichment DISABLED → measure Recall@10, Precision@10, MRR
3. Compare: if Precision@10 drops more than Recall@10 gains, the enrichment
   is net-negative and should be disabled or its validation tightened

Run BM25-only search (alpha=1.0) to isolate BM25 signal from vector signal:
- If BM25-only Precision@10 is below 0.50, keyword enrichment is introducing
  too much noise
- If BM25-only Recall@10 is below 0.60, keyword enrichment is insufficient
  and the vocabulary gap is not being closed
```

---

## 15. Evaluation Framework

### 15.1 Purpose

The evaluation framework measures end-to-end retrieval quality — the pipeline's entire purpose is to produce chunks that are found when engineers search. Without evaluation, every design decision (embedding model, chunk size, enrichment strategy, header embedding vs metadata filters) is guesswork.

The evaluation framework is NOT a unit test suite (that's `verify_pipeline.py`). It measures whether the right chunks are retrieved for real engineering queries against ground truth judgments.

### 15.2 Evaluation Dataset Structure

File: `eval_dataset.json`

```json
{
  "metadata": {
    "version": "1.0",
    "created_by": "domain_experts",
    "domain_coverage": ["dft", "verification", "physical_design"],
    "last_updated": "2025-MM-DD"
  },
  "queries": [
    {
      "query_id": "q001",
      "query_text": "What is the clock frequency for the DFT scan chain?",
      "domain": "dft",
      "intent": "specification_lookup",
      "ground_truth_chunks": [
        {
          "document_id": "doc-abc-123",
          "chunk_id": "chunk-def-456",
          "relevance": "primary",
          "rationale": "Contains the scan chain clock spec in Section 3.2"
        },
        {
          "document_id": "doc-abc-123",
          "chunk_id": "chunk-ghi-789",
          "relevance": "supporting",
          "rationale": "Contains related timing constraints"
        }
      ],
      "expected_abbreviation_expansion": "Design for Testability scan chain"
    },
    {
      "query_id": "q002",
      "query_text": "How do I configure OCV derating for signoff?",
      "domain": "physical_design",
      "intent": "procedural_howto",
      "ground_truth_chunks": [...]
    }
  ]
}
```

**Query intent taxonomy:**
- `specification_lookup` — "What is the value of X?" (exact answer expected)
- `procedural_howto` — "How do I do X?" (step-by-step procedure expected)
- `conceptual_explanation` — "What is X and why?" (background/context expected)
- `troubleshooting` — "X is failing, what could be wrong?" (diagnostic expected)
- `comparison` — "What's the difference between X and Y?" (multi-source expected)

**Relevance levels:**
- `primary` — This chunk directly answers the query. Must appear in results.
- `supporting` — This chunk provides useful context. Should appear but not mandatory.

### 15.3 Building the Evaluation Dataset

**This should be built collaboratively with domain experts.** Recommended approach:

1. **Query collection.** Ask 5-10 engineers across different domains to write 10 queries each that represent real questions they've had to answer by searching documentation. Target: 50-100 queries covering all domains. Emphasise diversity of intent (spec lookups, how-tos, troubleshooting, comparisons).

2. **Document ingestion.** Run the pipeline on a representative document corpus (20-50 documents covering the domains in the query set).

3. **Ground truth annotation.** For each query, domain experts identify which chunks in Weaviate correctly answer it. Use the Weaviate console or a simple annotation UI to browse chunks and tag relevance. Each query needs at minimum 1 primary chunk and 0-3 supporting chunks.

4. **Review and balance.** Ensure the dataset covers all domains roughly equally and includes queries that are easy (direct keyword match), medium (requires semantic understanding), and hard (abbreviation-heavy, cross-document, implicit).

**Minimal viable dataset:** 50 queries, 3 ground truth chunks per query average. This is sufficient to detect gross retrieval failures and compare model/config alternatives.

### 15.4 Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| Recall@5 | Fraction of primary ground truth chunks found in top 5 results | ≥ 0.75 |
| Recall@10 | Fraction of primary ground truth chunks found in top 10 results | ≥ 0.85 |
| Precision@10 | Fraction of top 10 results that are relevant (primary or supporting) | ≥ 0.50 |
| MRR (Mean Reciprocal Rank) | Average of 1/rank of first correct result across all queries | ≥ 0.60 |
| Abbreviation Hit Rate | Fraction of queries with abbreviations where expansion was correct | ≥ 0.95 |
| Fallback Chunk Retrieval Rate | Fraction of retrieved chunks that were `fallback_recursive` | Monitor (no target — lower is better) |
| BM25 Keyword Noise Rate | Fraction of BM25-only results (alpha=1.0) that are irrelevant | Monitor (< 0.40 target — see Section 14.4) |

### 15.5 Evaluation Runner

File: `evaluate_retrieval.py`

The evaluation runner:
1. Loads the evaluation dataset
2. For each query: expands abbreviations via `domain_vocabulary.yaml`, embeds the query with the configured model and `query_prefix`, runs hybrid search in Weaviate, collects top-K results
3. Compares retrieved chunk IDs against ground truth
4. Computes Recall@5, Recall@10, MRR, and per-query breakdown
5. Outputs a summary report with pass/fail against targets and per-domain breakdown

**Integration points:**
- `--eval` CLI flag runs evaluation after batch ingestion
- `config.evaluation.run_after_batch` triggers auto-eval
- Results logged to Langfuse (if enabled) for trend tracking

### 15.6 What the Evaluation Framework Validates

| Decision | How eval validates it |
|----------|----------------------|
| Embedding model choice (BGE vs E5 vs Jina) | Run eval with each model, compare Recall@10 |
| Chunk size target (256 vs 512 vs 1024) | Re-ingest with different sizes, compare MRR |
| Boundary context (on vs off) | Run eval with/without, compare Recall@5 |
| Header embedding vs metadata filter | Run eval both ways, compare Recall@10 |
| Hybrid search alpha (BM25 vs vector balance) | Sweep alpha 0.0–1.0, find optimal |
| Query abbreviation expansion | Check Abbreviation Hit Rate metric |
| Reranking (on vs off) | Run eval with/without cross-encoder, compare MRR |

---

## 16. Configuration Validation and Model Registry

### 16.1 Problem

Configuration errors are the most common cause of silent quality degradation. Changing the embedding model without updating the dimension, switching to a model that requires a query prefix without setting it, or configuring a chunk size that exceeds the model's input limit — these all produce a pipeline that runs without errors but produces bad results.

### 16.2 Model Registry

File: `model_registry.yaml`

A declarative registry of known model configurations. The pipeline loads this at startup and cross-validates the active config against it.

```yaml
embedding_models:
  "BAAI/bge-large-en-v1.5":
    dimension: 1024
    max_input_tokens: 512
    query_prefix: "Represent this sentence for searching relevant passages: "
    document_prefix: ""
    normalize: true
    notes: "Best general-purpose model. Requires query prefix for optimal retrieval."

  "BAAI/bge-base-en-v1.5":
    dimension: 768
    max_input_tokens: 512
    query_prefix: "Represent this sentence for searching relevant passages: "
    document_prefix: ""
    normalize: true
    notes: "Smaller, faster. ~2% lower retrieval quality than large."

  "intfloat/e5-large-v2":
    dimension: 1024
    max_input_tokens: 512
    query_prefix: "query: "
    document_prefix: "passage: "
    normalize: true
    notes: "Microsoft E5. Requires both query and document prefixes."

  "jinaai/jina-embeddings-v2-base-en":
    dimension: 768
    max_input_tokens: 8192
    query_prefix: ""
    document_prefix: ""
    normalize: true
    notes: "Long-context model. No prefixes required. Good for large chunks."

  "text-embedding-3-large":
    dimension: 3072
    max_input_tokens: 8191
    query_prefix: ""
    document_prefix: ""
    normalize: false
    notes: "OpenAI model. High dimension, API-only."

llm_models:
  "gpt-4o-mini":
    provider: "openai"
    max_context_tokens: 128000
    max_output_tokens: 16384
    notes: "Default. Good balance of quality and cost."

  "claude-sonnet-4-20250514":
    provider: "anthropic"
    max_context_tokens: 200000
    max_output_tokens: 8192
    notes: "Higher quality for refactoring and KG extraction."

  "llama3.1:8b":
    provider: "ollama"
    max_context_tokens: 128000
    max_output_tokens: 4096
    notes: "Local model. No API costs. Lower quality."

reranker_models:
  "BAAI/bge-reranker-large":
    max_input_tokens: 512
    notes: "Cross-encoder reranker. Use for retrieval-time reranking."
```

### 16.3 Startup Config Validation

File: `pipeline/config_validator.py`

When `config.config_validation.validate_on_startup` is True (default), the pipeline runs cross-validation checks before processing any documents:

**Embedding config checks:**
1. If `model_name` exists in the registry → validate `dimension` matches registry
2. If `model_name` exists in the registry → warn if `query_prefix` doesn't match registry (common misconfiguration)
3. If `target_chunk_size > model.max_input_tokens` → error: chunks will exceed model input limit
4. If `max_chunk_size > model.max_input_tokens` → error: even single chunks may not fit

**LLM config checks:**
1. If `model_name` exists in the registry → validate `provider` matches registry
2. If `max_tokens > model.max_output_tokens` → warn: may cause truncation

**Cross-config checks:**
1. If `chunking.target_chunk_size` is set but `embedding.use_model_tokeniser` is False → warn: token estimation may be inaccurate for engineering text
2. If `knowledge_graph.enabled` is True but `skip_knowledge_graph` is True → warn: contradictory config
3. If `review.demote_on_reingestion` is True but `reingestion.preserve_review_tier` is True → error: contradictory config
4. If `vocabulary.inject_into_prompts` is True but `vocabulary.dictionary_path` is None → warn: no vocabulary loaded
5. If `chunking.target_chunk_size + (chunking.boundary_context_sentences * 30) > model.max_input_tokens` → error: enriched chunks will exceed embedding model input limit, causing silent truncation. Suggest reducing `target_chunk_size` to `max_input_tokens - (boundary_context_sentences * 30)`
6. If `parallel_processing` is True → warn: parallel processing is reserved for future implementation (v3.0+) and has no effect in the current version

**Validation output:** Errors halt pipeline startup with clear messages. Warnings are logged but do not block. The validation report is included in the first processing log entry.

### 16.4 Design Principle

This validation pattern should extend to all future configuration additions. Every config field that has a dependency on another field (dimension depends on model, prefix depends on model, chunk size depends on model input limit) must have a corresponding cross-validation check. The model registry is the single source of truth for model-specific parameters — when someone changes the embedding model, the registry tells them exactly what else needs to change.