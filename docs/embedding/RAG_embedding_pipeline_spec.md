# RAG Document Embedding Pipeline — System Specification (v2.0.0)

## Document Information

| Field | Value |
|-------|-------|
| System | AION RAG Document Embedding Pipeline |
| Document Type | System Specification |
| Companion Documents | RAG_embedding_pipeline_arch.md (Detailed Design), RAG_embedding_pipeline_summary.md (Executive Summary) |
| Version | 2.0.0 |
| Status | Draft |

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | — | — | Initial specification |
| 2.0.0 | 2026-03-10 | — | Restructured to align with write-spec skill: added per-requirement rationale and acceptance criteria, terminology table, assumptions & constraints, requirement format section, entry/exit points, requirements traceability matrix |

---

## 1. Purpose and Scope

### 1.1 Problem Statement

ASIC design organisations accumulate critical engineering knowledge across hundreds of documents: specifications, design guides, runbooks, standard operating procedures, and project reports. This knowledge is fragmented across file servers, SharePoint, and individual workstations. When engineers leave, their contextual understanding leaves with them.

Existing search tools fail for engineering documentation because:

1. The same technical term carries different meaning across domains (e.g., "clock domain" spans front-end design, DFT, verification, and physical design).
2. Knowledge exists at varying levels of maturity — from formally approved specifications to an individual engineer's utility script — with no mechanism to distinguish authoritative from informal sources.
3. Documents rely on sequential reading order; isolated paragraphs lose meaning without surrounding context (e.g., "the same voltage" is ambiguous outside its original section).

### 1.2 Scope

The system SHALL transform engineering documents into semantically searchable, context-aware vector embeddings stored in a vector database, with a complementary knowledge graph capturing relationships between documents, concepts, entities, and specifications.

The system is designed for a mission-critical engineering environment where incorrect retrieval (e.g., returning a 4nm specification when the query is about a 12nm specification) could propagate into design errors.

**Entry point:** Source document file (PDF, DOCX, PPTX, XLSX, Markdown, HTML, RST, plain text) on local filesystem.

**Exit point:** Vector embeddings + metadata stored in vector database; knowledge graph triples stored in graph store.

**In scope:**
- Document ingestion, processing, and embedding
- Vector storage with hybrid search capability (vector + keyword)
- Knowledge graph construction and storage
- Document review tier management
- Re-ingestion of updated documents
- Batch processing of document directories
- Retrieval quality evaluation framework

### 1.3 Terminology

The following terms are used throughout this specification. A complete glossary is provided in Section 17.

| Term | Definition |
|------|-----------|
| Chunk | The atomic unit of the retrieval system; a segment of document text that is individually embedded and stored |
| RAG | Retrieval-Augmented Generation — a pattern where retrieved context is provided to an LLM for answer generation |
| Knowledge Graph | A graph of entities and relationships extracted from documents |
| Triple | A subject-predicate-object relationship in the knowledge graph |
| Hybrid Search | Combined vector similarity search and BM25 keyword search |
| BM25 | Best Matching 25 — a probabilistic ranking function for keyword-based text search |
| BYOM | Bring Your Own Model — mode where embeddings are computed externally and passed as pre-computed vectors to the vector store |
| Embedding | A dense vector representation of text, used for semantic similarity search |
| Review Tier | A trust classification (Fully/Partially/Self Reviewed) controlling a document's visibility in search results |
| Re-ingestion | Processing a previously ingested document again, cleaning up old data and inserting new data |
| DAG | Directed Acyclic Graph — the processing pipeline topology |
| VLM | Vision-Language Model — a multimodal model that can process both images and text |
| Deterministic ID | An identifier derived from content via cryptographic hashing, ensuring the same input always produces the same ID |
| HNSW | Hierarchical Navigable Small World — an approximate nearest-neighbour graph index algorithm |
| Idempotent | An operation that produces the same result whether applied once or multiple times |

### 1.4 Requirement Priority Levels

This specification uses the key words defined in [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119) to indicate requirement levels:

| Keyword | Meaning |
|---------|---------|
| **MUST** / **SHALL** | Absolute requirement. The system cannot be considered conformant without implementing this. |
| **SHOULD** / **RECOMMENDED** | There may be valid reasons to omit this in particular circumstances, but the full implications must be understood and carefully weighed. |
| **MAY** / **OPTIONAL** | Truly optional. The system is conformant whether or not this is implemented. |

### 1.5 Requirement Format

Requirements in this specification follow a structured ID convention:

| Prefix | Meaning |
|--------|---------|
| **FR-** | Functional Requirement |
| **NFR-** | Non-Functional Requirement |
| **SC-** | Security / Compliance Requirement |

Requirement ID ranges are allocated to sections as follows:

| ID Range | Section |
|----------|---------|
| FR-100–FR-199 | 3.1 Document Ingestion |
| FR-200–FR-299 | 3.2 Structure Detection |
| FR-300–FR-399 | 3.3 Multimodal Processing |
| FR-400–FR-499 | 3.4 Text Cleaning |
| FR-500–FR-599 | 3.5 Document Refactoring |
| FR-600–FR-699 | 3.6 Chunking |
| FR-700–FR-799 | 3.7 Chunk Enrichment |
| FR-800–FR-899 | 3.8 Metadata Generation |
| FR-900–FR-999 | 3.9 Cross-Reference Extraction |
| FR-1000–FR-1099 | 3.10 Knowledge Graph Extraction |
| FR-1100–FR-1199 | 3.11 Quality Validation |
| FR-1200–FR-1299 | 3.12 Embedding & Storage |
| FR-1300–FR-1399 | 3.13 Knowledge Graph Storage |
| FR-1400–FR-1499 | Re-ingestion |
| FR-1500–FR-1599 | Review Tiers |
| FR-1600–FR-1699 | Domain Vocabulary |
| FR-1700–FR-1799 | Error Handling & Fallbacks |
| FR-1800–FR-1899 | Configuration |
| FR-1900–FR-1999 | CLI / API Interface |
| FR-2000–FR-2199 | Evaluation & Feedback |
| NFR-100–NFR-699 | Non-Functional Requirements |
| SC-100–SC-199 | Security & Compliance |

### 1.6 Assumptions & Constraints

| ID | Assumption / Constraint | Impact if Violated |
|----|------------------------|--------------------|
| A-1 | Python 3.11+ runtime | Type hint syntax and match statements fail |
| A-2 | LangGraph/LangChain available | Pipeline DAG orchestration unavailable |
| A-3 | Vector database (e.g., Weaviate) accessible at configured URL | Embedding storage and retrieval fail |
| A-4 | LLM provider accessible (local or API) | LLM-dependent stages fall back to deterministic alternatives |
| A-5 | Sequential document processing (no concurrent ingestion of same document) | Race conditions on re-ingestion cleanup |
| A-6 | Documents ≤ ~100 pages | Larger documents should be pre-split; memory budget is < 2 GB peak RSS |
| A-7 | Embedding model input token limit accommodates target chunk size + boundary context | Embeddings will be truncated or fail |

### 1.7 Design Principles

The following principles SHALL guide all design and implementation decisions:

| Principle | Description |
|-----------|-------------|
| **Swappability over lock-in** | Every external dependency (LLM provider, embedding model, vector store, structure detector) SHALL be behind a configuration interface. Changing providers SHALL require configuration changes, not code changes. |
| **Fail-safe over fail-fast** | When an LLM call fails, the pipeline SHALL fall back to deterministic alternatives rather than halting. A single flawed LLM response SHALL NOT halt a batch job. |
| **Context preservation over compression** | The pipeline SHALL preserve every numerical value, specification, and procedural step. Content restructuring SHALL NOT summarise or remove information. |
| **Configuration-driven behaviour** | Pipeline behaviour (skip/enable processing stages, dry run mode) SHALL be controlled via a single configuration system with runtime overrides. |
| **Idempotency by construction** | Every processing stage SHALL produce the same output given the same input. Identifiers SHALL be derived deterministically from content. Re-ingesting a document SHALL produce an equivalent result to first-time ingestion. |
| **Controlled access over restriction** | Knowledge at all maturity levels SHALL be ingested and searchable. A tiered review system SHALL control visibility at retrieval time without restricting what enters the system. |

### 1.8 Out of Scope

**Out of scope:**
- Query processing, reranking, and answer generation (downstream retrieval layer)
- User authentication and access control
- Document authoring or editing
- Real-time document change detection (push-based ingestion)

**Out of scope — broader platform components (see Strategic Proposal):**

The RAG Document Embedding Pipeline is one component of a larger AI-Enabled Knowledge Management Platform. The following platform components are covered by the Strategic Proposal document and are NOT part of this specification:

- Skills repository and anti-skills validation framework
- AI-assisted documentation agents (background monitoring, draft generation)
- Code standardisation and Python migration initiative
- Web dashboard UI (planned for Phase 3 deployment)
- Reranker model deployment and tuning (downstream retrieval layer)
- Adoption strategy, change management, and gamification

---

## 2. Processing Pipeline Overview

### 2.1 High-Level Architecture

The system SHALL process documents through a directed acyclic graph (DAG) of processing stages, implemented using a graph-based orchestration framework (LangGraph, which is part of the LangChain ecosystem). Each document SHALL flow through the following stages in order:

```
Source Document (filesystem path)
    │
    ▼
┌──────────────────────────────────────┐
│ [1] DOCUMENT INGESTION               │
│     Read file, detect format,        │
│     compute hash, detect re-ingest   │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ [2] STRUCTURE DETECTION              │
│     Parse hierarchical section tree, │
│     extract figures and tables       │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ [3] MULTIMODAL PROCESSING [optional] │
│     Convert figures to text via VLM  │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ [4] TEXT CLEANING                    │
│     Normalise text, remove           │
│     boilerplate, integrate figures   │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ [5] DOCUMENT REFACTORING [optional]  │
│     Restructure for self-contained   │
│     paragraphs, resolve references   │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ [6] CHUNKING                         │
│     Split into semantic chunks       │
│     with deterministic IDs           │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ [7] CHUNK ENRICHMENT                 │
│     Add boundary context,            │
│     build metadata headers           │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ [8] METADATA GENERATION              │
│     Generate keywords, entities,     │
│     summaries at doc & chunk level   │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ [9] CROSS-REFERENCE EXTRACTION [opt] │
│     Detect inter-document refs       │
│     and standard citations           │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ [10] KNOWLEDGE GRAPH EXTRACTION [opt]│
│      Extract structured triples      │
│      for relationship-aware retrieval│
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ [11] QUALITY VALIDATION              │
│      Filter low-quality/duplicates,  │
│      assign quality scores           │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ [12] EMBEDDING & STORAGE             │
│      Generate vectors, clean up      │
│      old data, store in vector DB    │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ [13] KNOWLEDGE GRAPH STORAGE [opt]   │
│      Persist triples to graph store  │
└──────────────────────────────────────┘
               │
               ▼
Vector Database + Knowledge Graph Store
```

Stages marked with `[optional]` or `[opt]` are conditional and may be skipped based on configuration or document characteristics.

### 2.2 Stage Descriptions

| # | Stage | Purpose | Conditional |
|---|-------|---------|-------------|
| 1 | Document Ingestion | Read source file, detect format, compute content hash, detect re-ingestion | No |
| 2 | Structure Detection | Parse document into hierarchical section tree, extract figures and tables | No |
| 3 | Multimodal Processing | Convert detected figures into text descriptions using a vision-language model | Yes — only if figures detected (tentatively convert figures to text first) |
| 4 | Text Cleaning | Normalise text, remove boilerplate, integrate figure/table descriptions | No |
| 5 | Document Refactoring | Restructure content for self-containedness, resolve implicit references | Yes — skippable via config |
| 6 | Chunking | Split document into semantically coherent chunks with deterministic IDs | No |
| 7 | Chunk Enrichment | Add boundary context from adjacent chunks, build metadata headers | No |
| 8 | Metadata Generation | Generate searchable keywords and entities at document and chunk level | No |
| 9 | Cross-Reference Extraction | Detect references between documents, sections, and standards | Yes — skippable via config |
| 10 | Knowledge Graph Extraction | Extract structured triples for relationship-aware retrieval | Yes — skippable via config |
| 11 | Quality Validation | Filter low-quality and duplicate chunks, assign quality scores | No |
| 12 | Embedding & Storage | Generate vector embeddings, clean up old data on re-ingestion, store in vector database | No |
| 13 | Knowledge Graph (KG) Storage | Persist knowledge graph triples to graph store | Yes — only if KG enabled |

### 2.3 Processing Stage Independence

Each processing stage SHALL:
- Operate on a shared document state object
- Read only from fields populated by upstream stages
- Write only to its own designated output fields
- Be individually replaceable without affecting other stages
- Handle its own errors without crashing the pipeline
## 3. Functional Requirements

### 3.1 Document Ingestion (FR-100)

> **FR-101** | Priority: MUST
> **Description:** The system MUST support the following input formats: PDF, DOCX, PPTX, XLSX, Markdown, HTML, reStructuredText (RST), and plain text.
> **Rationale:** Engineering knowledge is spread across multiple document formats; without broad format coverage, significant portions of organisational knowledge remain unsearchable, directly contributing to the knowledge fragmentation problem.
> **Acceptance Criteria:** Given a test corpus containing one file of each supported format (PDF spec sheet, DOCX design guide, PPTX review deck, XLSX pin table, Markdown runbook, HTML datasheet, RST procedure, plain-text README), the system successfully ingests all eight files and produces non-empty text output for each. Given a `.wav` file, the system does not attempt format-specific extraction and classifies the format as UNKNOWN.

> **FR-102** | Priority: MUST
> **Description:** The system MUST detect document format from file extension with a fallback to UNKNOWN for unrecognised extensions.
> **Rationale:** Reliable format detection is a prerequisite for routing documents to the correct text extractor. Falling back to UNKNOWN rather than guessing prevents misinterpretation of binary data. Supports the fail-safe-over-fail-fast principle.
> **Acceptance Criteria:** Given a file named `clock_tree_spec.pdf`, the system detects format as PDF. Given a file named `notes.xyz`, the system assigns format UNKNOWN. Given a file with no extension, the system assigns format UNKNOWN.

> **FR-103** | Priority: MUST
> **Description:** The system MUST convert all input formats to text before downstream processing. Binary formats (PDF, DOCX, PPTX, XLSX) MUST go through format-specific text extractors.
> **Rationale:** A uniform text representation is the contract between ingestion and all downstream stages. Without this normalisation, every downstream stage would need format-specific logic, violating stage independence.
> **Acceptance Criteria:** Given a PDF containing a voltage specification table, the text extractor produces a text output that preserves the table content. Given a DOCX with embedded images and headings, the extractor outputs text with heading structure intact. The output for all formats is a plain text or markdown string — no binary content remains.

> **FR-104** | Priority: MUST
> **Description:** The system MUST extract text from PowerPoint files including text frames, speaker notes, and table cells, preserving slide order and structure.
> **Rationale:** Engineering review presentations often contain critical design decisions in speaker notes and specification data in table cells. Omitting these would lose knowledge that exists nowhere else. Supports the context-preservation principle.
> **Acceptance Criteria:** Given a PPTX with 5 slides where slide 3 contains a table of clock frequencies and slide 4 has speaker notes explaining a design trade-off, the extracted text includes both the table content from slide 3 and the speaker notes from slide 4 in correct slide order. Given a PPTX with an empty slide, that slide produces no text but does not cause an error.

> **FR-105** | Priority: MUST
> **Description:** The system MUST extract text from Excel files including all sheets, with table detection and markdown conversion. Named ranges and cell references MUST be preserved.
> **Rationale:** Excel files in ASIC organisations commonly contain pin tables, power budgets, and register maps across multiple sheets. Losing sheet context or cell references would make the extracted data ambiguous. Supports the context-preservation principle.
> **Acceptance Criteria:** Given an XLSX with sheets "Power_Budget" and "Pin_Table", the system extracts text from both sheets with sheet names as section headers. Given a sheet containing a table with headers `[Pin, Direction, Voltage, Description]`, the output contains a markdown table preserving all columns and rows. Given a named range "VDD_CORE_SPECS", the name appears in the extracted text.

> **FR-106** | Priority: MUST
> **Description:** The system MUST compute a SHA-256 content hash of the source file for change detection and re-ingestion purposes.
> **Rationale:** Content hashing is the foundation of the re-ingestion mechanism; without it the system cannot detect whether a document has changed, leading to either unnecessary reprocessing or stale data. Supports the idempotency-by-construction principle.
> **Acceptance Criteria:** Given the same file ingested twice, the system produces identical SHA-256 hashes both times. Given a file where a single character is changed, the system produces a different hash. The hash is a 64-character lowercase hexadecimal string.

> **FR-107** | Priority: MUST
> **Description:** The system MUST generate deterministic document identifiers derived from the source file path, ensuring the same file always produces the same document ID.
> **Rationale:** Deterministic IDs are essential for re-ingestion — the system must be able to locate previously ingested data for the same document. Supports the idempotency-by-construction principle.
> **Acceptance Criteria:** Given the file path `/data/specs/clock_tree_spec_v2.pdf` ingested twice, the system produces the same document ID both times. Given two different file paths, the system produces different document IDs.

> **FR-108** | Priority: MUST
> **Description:** The system MUST handle text encoding variations with a fallback chain (UTF-8 → Latin-1 → CP1252 → UTF-8 with replacement).
> **Rationale:** Legacy engineering documents frequently use non-UTF-8 encodings (e.g., Windows-1252 from older Word exports). Without a fallback chain, these documents would fail ingestion entirely, losing valuable knowledge. Supports the fail-safe-over-fail-fast principle.
> **Acceptance Criteria:** Given a Latin-1 encoded text file containing the character `µ` (micro sign), the system successfully decodes and preserves the character. Given a file with invalid byte sequences in all encodings, the system falls back to UTF-8 with replacement characters rather than raising an exception.

> **FR-109** | Priority: MUST
> **Description:** The system MUST load and make available a configurable domain vocabulary (abbreviation dictionary) for use by downstream stages.
> **Rationale:** ASIC/semiconductor terminology is highly domain-specific (e.g., DFT, PVT, CDC). A shared vocabulary ensures consistent abbreviation expansion across all stages and prevents misinterpretation of terms that differ across domains. Addresses the problem of term ambiguity across domains.
> **Acceptance Criteria:** Given a domain vocabulary file containing `{"DFT": "Design for Testability", "CDC": "Clock Domain Crossing"}`, the system loads the vocabulary and makes it accessible to downstream stages. Given no vocabulary file configured, the system starts with an empty vocabulary without error.

> **FR-110** | Priority: SHOULD
> **Description:** The system SHOULD support future addition of new input formats (e.g., SystemVerilog via AST-based extraction, Visio, images via OCR) without architectural changes.
> **Rationale:** Engineering toolchains evolve and new document types emerge; the ingestion layer must be extensible without requiring pipeline redesign. Supports the swappability-over-lock-in principle.
> **Acceptance Criteria:** A new format handler can be added by implementing a defined extractor interface and registering it in configuration, without modifying existing extractor code or pipeline orchestration logic. No existing unit tests break when the new handler is added.

> **FR-111** | Priority: MUST
> **Description:** Tool log files (.log) MUST be excluded from the vector embedding pipeline. Logs are better served by direct keyword search.
> **Rationale:** Log files are high-volume, low-semantic-density content that would pollute the vector space with noise, degrading retrieval quality for actual engineering documents.
> **Acceptance Criteria:** Given a directory containing `synthesis_run.log` and `clock_spec.pdf`, a batch ingestion run processes `clock_spec.pdf` and skips `synthesis_run.log`. The system logs a message indicating the `.log` file was excluded.

> **FR-112** | Priority: MUST
> **Description:** The system MUST support ingestion from local filesystem paths (absolute and relative).
> **Rationale:** Local filesystem is the primary document source for engineering teams; this is the baseline ingestion capability required before any remote integrations.
> **Acceptance Criteria:** Given an absolute path `/data/specs/clock_spec.pdf`, the system reads and ingests the file. Given a relative path `./specs/clock_spec.pdf`, the system resolves it against the working directory and ingests the file. Given a non-existent path, the system raises a clear error indicating the file was not found.

> **FR-113** | Priority: SHOULD
> **Description:** The system SHOULD support future integration with SharePoint document libraries as a document source without architectural changes. SharePoint integration is planned for Phase 3.
> **Rationale:** Many organisations store engineering documents in SharePoint; supporting it as a source expands coverage of fragmented knowledge. Supports the swappability-over-lock-in principle.
> **Acceptance Criteria:** The document source interface is abstracted such that a SharePoint adapter can be implemented without modifying the ingestion pipeline. The adapter would need to implement a defined source interface (e.g., `list_documents()`, `download_document()`) and be selectable via configuration.

### 3.2 Structure Detection (FR-200)

> **FR-201** | Priority: MUST
> **Description:** The system MUST parse documents into a hierarchical section tree preserving heading levels and parent-child relationships.
> **Rationale:** Engineering documents derive meaning from their hierarchical structure — a voltage value under "3.3V IO Domain" means something different than under "1.0V Core Domain". Losing this hierarchy causes the context-loss problem identified in the problem statement.
> **Acceptance Criteria:** Given a document with headings H1 "Power Specifications" → H2 "Core Domain" → H3 "Voltage Limits", the system produces a tree where "Voltage Limits" is a child of "Core Domain", which is a child of "Power Specifications". Given a flat document with no headings, the system produces a single root node containing all content.

> **FR-202** | Priority: MUST
> **Description:** The system MUST extract tables with headers, rows, and a text representation suitable for downstream processing.
> **Rationale:** Tables in ASIC specifications carry dense, high-value data (pin tables, timing parameters, register maps). A text representation ensures this data is searchable and preserved through the pipeline. Supports the context-preservation principle.
> **Acceptance Criteria:** Given a table with headers `[Parameter, Min, Typ, Max, Unit]` and a row `[VDD_CORE, 0.95, 1.0, 1.05, V]`, the system extracts both headers and row data and produces a markdown table representation. Given a table with merged cells, the system produces a best-effort extraction with no data loss.

> **FR-203** | Priority: MUST
> **Description:** The system MUST detect and extract figures with bounding boxes, captions, and surrounding text context.
> **Rationale:** Figures (block diagrams, timing diagrams, floorplans) convey information not present in the text. Capturing bounding boxes enables VLM processing; captions and surrounding text provide semantic anchoring for the generated description.
> **Acceptance Criteria:** Given a PDF page containing a block diagram with caption "Figure 3: Clock Distribution Network", the system extracts the figure region, associates the caption text, and captures at least the preceding and following paragraphs as surrounding context. Given a page with no figures, the figures list for that page is empty.

> **FR-204** | Priority: MUST
> **Description:** The system MUST export detected figure images to disk for multimodal processing when configured.
> **Rationale:** The VLM stage requires image files as input; exporting figures to disk decouples structure detection from multimodal processing, allowing each stage to operate independently.
> **Acceptance Criteria:** Given a document with 3 detected figures and figure export enabled in configuration, the system writes 3 image files (PNG or similar) to the configured output directory. File names include the document ID and figure index for traceability. Given figure export disabled in configuration, no image files are written.

> **FR-205** | Priority: MUST
> **Description:** The system MUST auto-detect abbreviation definitions in the document text (e.g., "Design for Testability (DFT)", abbreviation tables) and merge them with the domain vocabulary.
> **Rationale:** Documents often define their own abbreviations that may not be in the global domain vocabulary. Auto-detection ensures downstream stages (refactoring, enrichment) can resolve document-specific terms. Addresses term ambiguity across domains.
> **Acceptance Criteria:** Given text containing "Clock Domain Crossing (CDC) analysis ensures...", the system detects `CDC → Clock Domain Crossing` and adds it to the vocabulary. Given an abbreviation table with rows `DFT | Design for Testability`, the system extracts all entries. Given a conflict with the domain vocabulary (e.g., document defines "PD" as "Power Domain" but vocabulary has "PD" as "Physical Design"), the document-local definition takes precedence within that document's processing.

> **FR-206** | Priority: MUST
> **Description:** The system MUST compute an extraction confidence score (0.0–1.0) based on section tree depth, table completeness, text coherence, and character density.
> **Rationale:** Extraction quality varies significantly by document format and layout complexity. A confidence score enables downstream stages and operators to identify documents that may have lost structure during extraction.
> **Acceptance Criteria:** Given a well-structured Markdown document with clear headings and tables, the confidence score is above 0.8. Given a scanned PDF with OCR artefacts and no detectable headings, the confidence score is below 0.5. The score is a float in the range [0.0, 1.0].

> **FR-207** | Priority: MUST
> **Description:** Documents with extraction confidence below a configurable threshold MUST be flagged for manual review but MUST continue through the pipeline.
> **Rationale:** Stopping the pipeline on low-confidence documents would block batch processing. Flagging ensures operators are alerted without halting the pipeline. Supports the fail-safe-over-fail-fast principle.
> **Acceptance Criteria:** Given a confidence threshold configured at 0.6 and a document scoring 0.45, the system sets a `requires_manual_review` flag to true and continues processing. Given a document scoring 0.75 with the same threshold, the flag is false. The flagged document appears in pipeline output/logs with a review warning.

> **FR-208** | Priority: MUST
> **Description:** The structure detection provider MUST be swappable via configuration.
> **Rationale:** Different structure detection tools (e.g., Docling, Unstructured, custom parsers) have different strengths; the system must allow switching without code changes. Supports the swappability-over-lock-in principle.
> **Acceptance Criteria:** Given configuration specifying provider A, the system uses provider A for structure detection. Changing the configuration to provider B and re-running uses provider B with no code changes. Both providers implement the same interface and produce compatible output structures.

### 3.3 Multimodal Processing (FR-300)

> **FR-301** | Priority: MUST
> **Description:** The system MUST convert detected figures into text descriptions using a vision-language model (VLM), making visual content searchable via text embeddings.
> **Rationale:** Figures (block diagrams, timing diagrams, layout views) contain critical engineering information that would be entirely lost from the search index without text conversion. This directly addresses knowledge fragmentation by making visual knowledge retrievable.
> **Acceptance Criteria:** Given an extracted figure image of a clock distribution block diagram, the system sends it to the VLM and receives a text description. The description is stored associated with the figure's ID. Given a document with no figures, this stage produces no descriptions.

> **FR-302** | Priority: MUST
> **Description:** This stage MUST only execute when figures are detected in the document.
> **Rationale:** VLM calls are expensive (latency and cost). Skipping the stage entirely when no figures exist avoids unnecessary API calls and processing time.
> **Acceptance Criteria:** Given a Markdown document with no figures, the multimodal processing stage is not invoked (no VLM API calls are made). Given a PDF with 2 detected figures, the stage executes and processes both figures.

> **FR-303** | Priority: MUST
> **Description:** VLM descriptions MUST include diagram type, visible labels and values, key relationships, and numerical specifications.
> **Rationale:** Generic descriptions (e.g., "a block diagram") are useless for engineering retrieval. Descriptions must capture the specific technical content visible in the figure to enable precise search. Supports the context-preservation principle.
> **Acceptance Criteria:** Given a timing diagram showing setup time = 0.5ns and hold time = 0.2ns for a flip-flop, the VLM description includes the diagram type ("timing diagram"), the numerical values (0.5ns, 0.2ns), and the relationship (setup/hold time for flip-flop). The description does not merely say "a timing diagram".

> **FR-304** | Priority: MUST
> **Description:** VLM descriptions MUST NOT contain speculative or inferred information not visible in the figure.
> **Rationale:** In a mission-critical engineering environment, fabricated details in a figure description could propagate into design decisions. The system must never present LLM hallucinations as document content. Supports the context-preservation principle.
> **Acceptance Criteria:** Given a block diagram showing modules A, B, and C connected in series, the description does not mention module D or speculate about internal implementation of any module. A validation check can compare entities in the description against entities visible in the figure caption and surrounding text.

> **FR-305** | Priority: MUST
> **Description:** Each VLM-generated description MUST have an associated confidence score.
> **Rationale:** VLM output quality varies with figure complexity and clarity. A confidence score allows downstream stages and retrieval to weight or filter descriptions appropriately.
> **Acceptance Criteria:** Given a VLM description, the system stores a confidence score as a float in [0.0, 1.0] alongside the description. Given a clear, well-labelled schematic, the confidence score is higher than for a blurry or hand-drawn sketch.

> **FR-306** | Priority: MUST
> **Description:** The VLM provider MUST be swappable via configuration.
> **Rationale:** VLM technology is rapidly evolving; the system must allow switching providers (e.g., GPT-4V to Claude Vision) without code changes. Supports the swappability-over-lock-in principle.
> **Acceptance Criteria:** Given configuration specifying VLM provider "openai/gpt-4o", the system uses that provider. Changing the configuration to "anthropic/claude-sonnet" and re-running uses the new provider with no code changes. Both providers are accessed through the same interface.

> **FR-307** | Priority: MUST
> **Description:** If the VLM call fails, the figure MUST be recorded without a description (confidence = 0.0).
> **Rationale:** A single failed VLM call must not halt processing of the entire document. The figure is still recorded so it can be reprocessed later. Supports the fail-safe-over-fail-fast principle.
> **Acceptance Criteria:** Given a VLM API timeout or error for figure 2 of 5, the system records figure 2 with an empty description and confidence 0.0, then continues to process figures 3–5. The pipeline does not raise an exception. Pipeline logs record the failure with the figure ID and error details.

### 3.4 Text Cleaning (FR-400)

> **FR-401** | Priority: MUST
> **Description:** The system MUST normalise whitespace (collapse multiple spaces, limit consecutive newlines).
> **Rationale:** PDF and DOCX extraction frequently produces irregular whitespace that wastes token budget and degrades embedding quality. Normalisation ensures consistent input for chunking and embedding.
> **Acceptance Criteria:** Given text with 5 consecutive spaces between words, the output contains a single space. Given text with 6 consecutive newlines, the output contains no more than the configured maximum (e.g., 2). Given text with tabs mixed with spaces, the output uses consistent spacing.

> **FR-402** | Priority: MUST
> **Description:** The system MUST remove boilerplate artefacts (page headers/footers, confidentiality notices, stray page numbers) using configurable patterns.
> **Rationale:** Boilerplate text (e.g., "CONFIDENTIAL — Company Internal" repeated on every page) adds noise to embeddings and wastes token capacity without contributing searchable knowledge. Supports the configuration-driven-behaviour principle.
> **Acceptance Criteria:** Given a configurable pattern list including `"CONFIDENTIAL.*INTERNAL"` and text containing "CONFIDENTIAL — Company Internal" on 20 pages, all 20 occurrences are removed. Given a pattern `"^Page \\d+$"`, stray page numbers like "Page 42" on their own line are removed. Custom patterns can be added via configuration without code changes.

> **FR-403** | Priority: MUST
> **Description:** The system MUST detect and reduce repeated headers/footers (lines appearing more than 3 times) to a single occurrence.
> **Rationale:** Even without explicit pattern configuration, frequently repeated lines are almost certainly boilerplate. Automatic detection catches document-specific headers/footers that configurable patterns might miss.
> **Acceptance Criteria:** Given a document where the line "ASIC Design Specification Rev 2.1" appears 15 times (once per page), the output contains this line exactly once. Given a line that appears exactly 3 times in a document, it is preserved (threshold is "more than 3"). Given a line that appears 4 times, it is reduced to 1 occurrence.

> **FR-404** | Priority: MUST
> **Description:** The system MUST integrate VLM-generated figure descriptions into the text stream with figure ID markers.
> **Rationale:** Figure descriptions must be embedded in-line at the figure's original position so that chunking preserves the spatial relationship between figures and surrounding text. Supports the context-preservation principle.
> **Acceptance Criteria:** Given a figure description for figure ID `fig_003` with text "Block diagram showing clock distribution from PLL to 4 clock domains", the text stream contains a marker such as `[FIGURE fig_003]` followed by the description at the position where the figure originally appeared. Given a figure with no description (confidence = 0.0), only the marker is inserted with no description text.

> **FR-405** | Priority: MUST
> **Description:** The system MUST integrate table markdown representations into the text stream with table ID markers, preserving tabular data (e.g., voltage specifications, pin tables) in a searchable format.
> **Rationale:** Tables carry dense, high-value engineering data. Integrating them as markdown preserves structure while keeping the data in the text stream for chunking and embedding. Supports the context-preservation principle.
> **Acceptance Criteria:** Given a table with ID `tbl_002` containing voltage specifications, the text stream includes a marker such as `[TABLE tbl_002]` followed by the markdown table representation. The markdown preserves all headers, rows, and cell values from the original table. Given a pin table with 50 rows, all 50 rows appear in the markdown output.

### 3.5 Document Refactoring (FR-500)

> **FR-501** | Priority: MUST
> **Description:** The system MUST support optional restructuring of document text to make each paragraph self-contained for retrieval, resolving implicit references (e.g., "as mentioned above") to explicit references (e.g., "the 1.8V core voltage specified in Section 1.2").
> **Rationale:** Isolated paragraphs lose meaning without surrounding context — this is a core problem identified in the problem statement (e.g., "the same voltage" is ambiguous outside its original section). Refactoring resolves this at ingestion time rather than burdening retrieval. Supports the context-preservation principle.
> **Acceptance Criteria:** Given a paragraph containing "This voltage should not exceed the limit specified above", the refactored output replaces the implicit reference with the explicit value, e.g., "This voltage should not exceed the 1.05V maximum specified in Section 2.3 Core Power". Given a paragraph with no implicit references, the text is returned unchanged.

> **FR-502** | Priority: MUST
> **Description:** Refactoring MUST be skippable via configuration.
> **Rationale:** Refactoring uses LLM calls which add latency and cost. Some deployments may prefer raw text. Supports the configuration-driven-behaviour principle.
> **Acceptance Criteria:** Given configuration with `refactoring.enabled = false`, the refactoring stage is skipped entirely and the text passes through unchanged. Given `refactoring.enabled = true`, the stage executes. No LLM calls are made when the stage is disabled.

> **FR-503** | Priority: MUST
> **Description:** The refactoring process MUST use a self-correcting loop with configurable maximum iterations per section.
> **Rationale:** A single LLM pass may introduce errors or miss references. Iterative correction with a bounded loop prevents runaway processing while improving output quality.
> **Acceptance Criteria:** Given `refactoring.max_iterations = 3`, the system performs at most 3 refactoring-validation cycles per section. If iteration 2 passes validation, iteration 3 is not executed. Given `max_iterations = 1`, only one pass is performed regardless of validation outcome.

> **FR-504** | Priority: MUST
> **Description:** Each refactoring iteration MUST include a fact-check validation that no information was added, removed, or distorted.
> **Rationale:** LLMs can hallucinate or subtly alter technical facts. In a mission-critical environment, a refactored paragraph claiming "1.2V" when the source says "1.0V" could cause a design error. Supports the context-preservation principle.
> **Acceptance Criteria:** Given a source paragraph stating "VDD_CORE = 1.0V" and a refactored version that changes it to "VDD_CORE = 1.2V", the fact-check validation fails and the iteration is rejected. Given a refactored version that only adds context ("VDD_CORE = 1.0V, as specified in Section 2.1"), the fact-check passes.

> **FR-505** | Priority: MUST
> **Description:** Each refactoring iteration MUST include a completeness check that no content was lost.
> **Rationale:** Refactoring must restructure, not summarise. Dropped sentences or values represent irreversible knowledge loss. Supports the context-preservation principle.
> **Acceptance Criteria:** Given a source paragraph with 4 sentences and a refactored version with only 3 sentences (one dropped), the completeness check fails. Given a refactored version that rephrases all 4 sentences but preserves all information, the completeness check passes.

> **FR-506** | Priority: MUST
> **Description:** Refactoring MUST adhere to the following constraints: never add information not in the source, never remove content, never change meaning, preserve all numerical values exactly, and only expand abbreviations found in the document or the domain vocabulary.
> **Rationale:** These constraints are the safety boundary for LLM-based content manipulation in a mission-critical environment. Violating any one of them could propagate incorrect information into engineering decisions. Supports the context-preservation principle.
> **Acceptance Criteria:** Given source text "The PVT corners are nominal", the refactored text may expand to "The Process-Voltage-Temperature (PVT) corners are nominal" if PVT is in the vocabulary, but must not add "at 25°C" unless that temperature appears in the source. Given a numerical value "3.3V ± 5%", the refactored text preserves "3.3V ± 5%" exactly.

> **FR-507** | Priority: MUST
> **Description:** If all refactoring iterations fail validation, the system MUST return the original text unchanged (fail-safe).
> **Rationale:** A failed refactoring attempt must never result in corrupted or partially-refactored text entering the pipeline. The original text is always a safe fallback. Supports the fail-safe-over-fail-fast principle.
> **Acceptance Criteria:** Given 3 iterations where each fails fact-check or completeness validation, the system returns the exact original text byte-for-byte. The system logs a warning indicating refactoring failed for the section. No exception is raised.

> **FR-508** | Priority: MUST
> **Description:** If completeness falls below 80%, the system MUST reject the refactored text and return the original.
> **Rationale:** A completeness score below 80% indicates significant content loss, making the refactored text unreliable. The 80% threshold provides a concrete, measurable safety net.
> **Acceptance Criteria:** Given a completeness score of 0.75 (75%), the system rejects the refactored text and returns the original. Given a completeness score of 0.85 (85%), the system accepts the refactored text. The completeness score and accept/reject decision are logged.

### 3.6 Chunking (FR-600)

> **FR-601** | Priority: MUST
> **Description:** The system MUST split document text into semantically coherent chunks that respect topic boundaries, specification blocks, and procedure sequences.
> **Rationale:** Naive fixed-size splitting breaks mid-sentence or mid-specification, producing chunks that lose meaning in isolation. Semantic chunking preserves the coherence needed for accurate retrieval. Addresses the context-loss problem from isolated paragraphs.
> **Acceptance Criteria:** Given a section containing a 3-step procedure (Step 1, Step 2, Step 3), the chunker keeps all 3 steps in the same chunk if they fit within the size limit. Given two topically distinct subsections ("Clock Distribution" and "Power Grid"), they are placed in separate chunks. Given a specification block with related parameters, the parameters are not split across chunks.

> **FR-602** | Priority: MUST
> **Description:** Chunk size MUST be configurable with target, minimum, and maximum token limits.
> **Rationale:** Optimal chunk size depends on the embedding model, retrieval strategy, and document characteristics. Configurable limits allow tuning without code changes. Supports the configuration-driven-behaviour principle.
> **Acceptance Criteria:** Given configuration `chunk_size.target = 512, chunk_size.min = 100, chunk_size.max = 1024`, chunks are produced targeting ~512 tokens, with no chunk below 100 tokens (except the final chunk of a section) and no chunk exceeding 1024 tokens. Changing these values in configuration changes the chunk sizes produced.

> **FR-603** | Priority: MUST
> **Description:** The target chunk size MUST account for downstream enrichment overhead (boundary context) to prevent exceeding the embedding model's input token limit.
> **Rationale:** Boundary context (FR-703) adds tokens to the embedding input. If the chunk itself already uses the full token budget, the enriched input will exceed the model's limit, causing truncation or errors.
> **Acceptance Criteria:** Given an embedding model with 8192 token limit and boundary context configured to add up to 200 tokens, the effective maximum chunk size is at most 8192 - 200 = 7992 tokens. No enriched chunk (chunk + boundary context) exceeds the embedding model's input token limit.

> **FR-604** | Priority: MUST
> **Description:** Tables MUST be treated as indivisible chunking units by default (configurable). Tables exceeding maximum chunk size MUST be split row-wise with the header row prepended to every fragment.
> **Rationale:** Splitting a table mid-row produces meaningless fragments. Keeping tables atomic preserves data integrity. When a table is too large, prepending headers to each fragment ensures every fragment is interpretable on its own. Supports the context-preservation principle.
> **Acceptance Criteria:** Given a 10-row pin table within the maximum chunk size, it appears as a single chunk. Given a 200-row register map exceeding the maximum chunk size, it is split into multiple chunks where each chunk starts with the original header row (`| Register | Address | Width | Description |`) followed by a subset of data rows. Given `tables.keep_atomic = false` in configuration, tables may be split at any boundary.

> **FR-605** | Priority: MUST
> **Description:** Each chunk MUST receive a deterministic identifier derived from the document ID, chunk position, and content hash. Re-ingesting the same document with identical content MUST produce identical chunk IDs.
> **Rationale:** Deterministic chunk IDs enable the re-ingestion mechanism to identify and replace specific chunks. Without determinism, every re-ingestion would appear as entirely new content. Supports the idempotency-by-construction principle.
> **Acceptance Criteria:** Given the same document ingested twice with no content changes, all chunk IDs are identical across both runs. Given a document where only section 5 changes, chunks derived from unchanged sections produce the same IDs. The chunk ID is a string that encodes document ID, position index, and a content-derived hash component.

> **FR-606** | Priority: MUST
> **Description:** Each chunk MUST carry adjacency links (previous/next chunk IDs) to enable context expansion at retrieval time.
> **Rationale:** Retrieval sometimes returns a chunk that is only partially relevant; adjacency links let the retrieval layer fetch surrounding chunks for additional context without re-processing the document. Supports the context-preservation principle.
> **Acceptance Criteria:** Given a document split into chunks C1, C2, C3, chunk C2 has `previous_chunk_id = C1.id` and `next_chunk_id = C3.id`. Chunk C1 has `previous_chunk_id = null` and chunk C3 has `next_chunk_id = null`. After quality validation removes a chunk, adjacency links are repaired (see FR-1105).

> **FR-607** | Priority: MUST
> **Description:** Each chunk MUST be tagged with its production method (e.g., LLM-driven, deterministic fallback, table-specific).
> **Rationale:** Knowing how a chunk was produced enables quality analysis and debugging. If retrieval quality degrades, operators can correlate with production method to identify whether fallback chunking is the cause.
> **Acceptance Criteria:** Given a chunk produced by the LLM chunker, it has `production_method = "llm"`. Given a chunk produced by the deterministic fallback, it has `production_method = "deterministic_fallback"`. Given a table chunk, it has `production_method = "table"`. The tag is stored as chunk metadata.

> **FR-608** | Priority: MUST
> **Description:** If the primary (LLM-driven) chunking fails, the system MUST fall back to a deterministic recursive splitter on paragraph/sentence boundaries.
> **Rationale:** LLM calls can fail due to rate limits, timeouts, or provider outages. A deterministic fallback ensures the pipeline always produces chunks rather than halting. Supports the fail-safe-over-fail-fast principle.
> **Acceptance Criteria:** Given an LLM API failure during chunking, the system switches to the deterministic splitter and produces valid chunks. The fallback chunks respect paragraph and sentence boundaries (no mid-sentence splits except when a single sentence exceeds the maximum chunk size). The pipeline logs a warning indicating the fallback was used.

> **FR-609** | Priority: MUST
> **Description:** Each chunk MUST carry a content type tag (text, table, figure, code, equation, list, heading).
> **Rationale:** Content type enables type-specific retrieval filtering (e.g., "show me only tables") and type-aware ranking at retrieval time. Different content types may also require different embedding strategies.
> **Acceptance Criteria:** Given a chunk containing a markdown table, it has `content_type = "table"`. Given a chunk containing a code block, it has `content_type = "code"`. Given a chunk containing narrative text, it has `content_type = "text"`. Given a chunk containing a figure description, it has `content_type = "figure"`.

> **FR-610** | Priority: MUST
> **Description:** Sections exceeding a configurable size limit MUST be pre-split at paragraph boundaries before chunking to keep processing windows within manageable limits.
> **Rationale:** Very large sections sent to an LLM chunker in a single call may exceed context window limits or produce poor chunking decisions. Pre-splitting reduces the problem size while preserving paragraph integrity.
> **Acceptance Criteria:** Given a section of 50,000 tokens and a pre-split limit of 10,000 tokens, the section is split into approximately 5 segments at paragraph boundaries before being sent to the chunker. Given a section of 5,000 tokens with the same limit, no pre-splitting occurs. Pre-splits never break mid-paragraph.

> **FR-611** | Priority: MUST
> **Description:** Token counting MUST use the actual tokeniser from the configured embedding model to correctly handle domain-specific terminology. A conservative approximation MUST be available as a fallback.
> **Rationale:** Different tokenisers produce different token counts for the same text, especially for domain-specific terms (e.g., "SystemVerilog" may be 1 or 3 tokens depending on the tokeniser). Using the wrong tokeniser could cause chunks to exceed the model's input limit. Supports the swappability-over-lock-in principle.
> **Acceptance Criteria:** Given an embedding model with a specific tokeniser, the system uses that tokeniser for all token counts. Given a term like "clock_domain_crossing_check", the count matches the model's actual tokenisation. If the model tokeniser is unavailable, the fallback uses a conservative ratio (e.g., 1 token per 3.5 characters) that overestimates rather than underestimates token count.

### 3.7 Chunk Enrichment (FR-700)

> **FR-701** | Priority: MUST
> **Description:** The system MUST construct metadata context headers for each chunk containing: domain, document type, title, section path, content type, review tier, and linked content references.
> **Rationale:** Context headers provide the structured metadata needed for filtering, display, and retrieval-time decisions. Without them, chunks are anonymous text fragments with no organisational context. Supports the context-preservation principle.
> **Acceptance Criteria:** Given a chunk from section "3.2 Core Power" in document "ASIC_Power_Spec_v2.pdf" classified as domain "ASIC Design" and review tier "Approved", the context header includes all of: `domain: "ASIC Design"`, `document_type: "Specification"`, `title: "ASIC_Power_Spec_v2"`, `section_path: "Power Specifications > Core Power"`, `content_type: "text"`, `review_tier: "Approved"`. No fields are omitted.

> **FR-702** | Priority: MUST
> **Description:** Context headers MUST be stored as metadata properties for filtering and display. Context headers MUST NOT be embedded into the vector by default (configurable).
> **Rationale:** Embedding metadata into the vector dilutes the semantic signal of the actual content. Storing metadata as filterable properties enables precise pre-filtering (e.g., "only Approved documents in ASIC Design domain") without compromising embedding quality. Supports the configuration-driven-behaviour principle.
> **Acceptance Criteria:** By default, the vector embedding input contains only the chunk content and boundary context — not the context header fields. The metadata fields are stored as separate filterable properties in the vector store. Given configuration `enrichment.embed_headers = true`, the context header is prepended to the embedding input. A retrieval query filtering on `domain = "ASIC Design"` uses the metadata property, not vector similarity.

> **FR-703** | Priority: MUST
> **Description:** The system MUST inject boundary context (last N sentences from the preceding chunk) into each chunk's embedding input to capture topic transitions. The number of sentences MUST be configurable.
> **Rationale:** Chunk boundaries are artificial — a topic may span two chunks. Injecting trailing sentences from the previous chunk into the embedding input captures continuity that would otherwise be lost, improving retrieval recall for queries that span chunk boundaries.
> **Acceptance Criteria:** Given `enrichment.boundary_sentences = 3` and a preceding chunk ending with sentences S1, S2, S3, the current chunk's embedding input is prepended with S1, S2, S3. Given the first chunk in a document (no predecessor), no boundary context is added. Given `boundary_sentences = 0`, no boundary context is injected for any chunk.

> **FR-704** | Priority: MUST
> **Description:** Boundary context MUST improve embedding recall without consuming excessive embedding capacity. If the enriched content exceeds the embedding model's input limit, boundary context MUST be trimmed rather than chunk content.
> **Rationale:** The chunk's own content is the primary semantic signal and must never be truncated. Boundary context is supplementary; trimming it preserves the core content integrity. Supports the context-preservation principle.
> **Acceptance Criteria:** Given a chunk of 7,800 tokens, boundary context of 500 tokens, and an embedding model limit of 8,192 tokens, the boundary context is trimmed to 392 tokens (or fewer) so the total does not exceed 8,192. The chunk's own content (7,800 tokens) is never truncated. Given a chunk of 4,000 tokens with 500 tokens of boundary context and the same model limit, no trimming occurs.

> **FR-705** | Priority: MUST
> **Description:** The raw chunk content MUST be preserved separately from the enriched embedding input for display in retrieval results.
> **Rationale:** Users viewing retrieval results should see the original chunk text, not text polluted with boundary context from adjacent chunks. Separating raw and enriched versions serves both embedding quality and user experience.
> **Acceptance Criteria:** Given a chunk with raw content "The VDD_CORE supply is 1.0V nominal" and enriched content that includes boundary context prepended, both versions are stored. The retrieval API returns the raw content for display. The enriched content is used only for generating the embedding vector. Modifying the boundary context configuration and re-ingesting does not alter the stored raw content.
### 3.8 Metadata Generation (FR-800)

> **FR-801** | Priority: MUST
> **Description:** The system MUST generate document-level metadata: summary (2-3 sentences), technical keywords (10-20), named entities, topic categories, domain classification, and document type classification.
> **Rationale:** Document-level metadata enables faceted filtering and retrieval-time ranking. Without structured metadata, queries cannot distinguish between documents by domain, type, or topic, worsening knowledge fragmentation across engineering teams.
> **Acceptance Criteria:** Given a design specification document for a 5nm ASIC clock distribution network, the system produces: a 2-3 sentence summary mentioning clock distribution; between 10 and 20 keywords including domain terms (e.g., "clock tree synthesis", "skew budgeting"); named entities (e.g., "TSMC N5"); a domain classification (e.g., "Physical Design"); and a document type (e.g., "Specification"). A document with fewer than 10 extractable keywords still produces at least 10 keywords or explicitly flags the shortfall.

> **FR-802** | Priority: MUST
> **Description:** The system MUST generate chunk-level metadata: keywords (5-10) and named entities for each chunk.
> **Rationale:** Chunk-level metadata supports fine-grained retrieval filtering, ensuring search results can be narrowed to chunks that specifically discuss the queried concept rather than returning entire documents. This directly addresses the term ambiguity problem where "clock domain" has different meanings in different contexts.
> **Acceptance Criteria:** Given a chunk discussing DFT scan chain insertion for a 7nm design, the system produces 5-10 keywords (e.g., "scan chain", "DFT", "ATPG", "fault coverage", "test compression") and named entities (e.g., "Synopsys DFT Compiler"). A chunk with only 3 extractable keywords produces exactly those 3 without padding with unrelated terms.

> **FR-803** | Priority: MUST
> **Description:** All LLM-generated keywords and entities MUST pass a validation gate ensuring they are grounded in the chunk content (direct presence, abbreviation bridge, or compound term overlap). Keywords inferred by association but not discussed in the chunk MUST be discarded.
> **Rationale:** Supports the context-preservation design principle. LLMs may hallucinate plausible-sounding keywords (e.g., generating "power integrity" for a chunk that only discusses signal integrity), which would cause false-positive retrievals and erode trust in search results.
> **Acceptance Criteria:** Given a chunk that discusses "CTS" (clock tree synthesis) but never mentions "OCV" (on-chip variation), an LLM-generated keyword "OCV" is discarded because it has no direct presence, abbreviation bridge, or compound term overlap in the chunk. A keyword "clock tree synthesis" is retained via abbreviation bridge to "CTS" present in the chunk text.

> **FR-804** | Priority: MUST
> **Description:** Document-level keywords MUST be stored as filterable properties but MUST NOT be keyword-indexed at the chunk level to prevent false positives.
> **Rationale:** Indexing document-level keywords at the chunk level would cause every chunk in a document to match queries for terms discussed only in other sections. This directly causes the term ambiguity problem described in the problem statement, where a search for "power grid" returns chunks about unrelated topics simply because the parent document mentions power grids elsewhere.
> **Acceptance Criteria:** Given a 50-page SoC integration guide where only Section 4 discusses "thermal management", a BM25 keyword search for "thermal management" returns only chunks from Section 4, not chunks from other sections. Document-level metadata for the guide includes "thermal management" as a filterable property for document-level queries.

> **FR-805** | Priority: MUST
> **Description:** If the LLM call fails, the system MUST fall back to a deterministic frequency-based keyword extraction (TF-IDF).
> **Rationale:** Supports the fail-safe-over-fail-fast design principle. Metadata generation must not block the pipeline; TF-IDF provides acceptable keyword extraction quality to ensure every chunk has searchable metadata even when the LLM is unavailable.
> **Acceptance Criteria:** Given an LLM timeout during metadata generation for a chunk containing "The scan chain length was optimised from 1500 to 800 flip-flops to reduce test time", the TF-IDF fallback produces keywords including "scan chain", "flip-flops", and "test time". The pipeline continues without halting, and the chunk's metadata includes a flag indicating fallback extraction was used.

> **FR-806** | Priority: MUST
> **Description:** The system MUST support domain vocabulary injection into metadata generation prompts to ensure canonical domain terminology is used.
> **Rationale:** Supports configuration-driven-behaviour. Without vocabulary injection, the LLM may generate inconsistent terminology (e.g., "DFT" vs "Design for Test" vs "Design for Testability"), fragmenting search results for the same concept across different keyword forms.
> **Acceptance Criteria:** Given a domain vocabulary entry mapping "DFT" to "Design for Testability", metadata generation for a chunk mentioning "DFT" produces the canonical keyword "Design for Testability" (or both "DFT" and "Design for Testability") rather than a non-canonical variant like "Design for Test". Removing the vocabulary entry results in the LLM choosing its own expansion.

---

### 3.9 Cross-Reference Extraction (FR-900)

> **FR-901** | Priority: MUST
> **Description:** The system MUST detect cross-references between documents, sections, and external standards.
> **Rationale:** Engineering documents form a web of dependencies; a specification may reference a design guide, which references an IEEE standard. Without cross-reference extraction, these relationships are invisible to retrieval, and the system cannot surface related documents or perform impact analysis when a referenced standard is updated.
> **Acceptance Criteria:** Given a document containing "Refer to Section 3.2 of the Clock Domain Crossing Guidelines (CDC_GUIDE_v2.1)" and "per IEEE 1149.1", the system extracts at least two cross-references: one internal reference to CDC_GUIDE_v2.1 Section 3.2, and one external standard reference to IEEE 1149.1. Each reference includes type, target identifier, and source location.

> **FR-902** | Priority: MUST
> **Description:** Cross-reference extraction MUST be skippable via configuration.
> **Rationale:** Supports configuration-driven-behaviour. Some ingestion runs (e.g., quick re-indexing of a single updated document) may not need cross-reference extraction, and skipping it reduces processing time and LLM costs.
> **Acceptance Criteria:** Given a pipeline configuration with `cross_reference_extraction.enabled: false`, the cross-reference extraction stage is skipped entirely (no LLM calls, no regex matching). The pipeline log records the stage as "skipped". Setting `cross_reference_extraction.enabled: true` enables the stage and produces cross-reference output.

> **FR-903** | Priority: MUST
> **Description:** The system MUST support both deterministic pattern matching (regex) and LLM-based extraction for implicit references.
> **Rationale:** Regex reliably captures explicit, well-formatted references (e.g., "IEEE 1149.1", "Section 3.2"), while LLM-based extraction catches implicit references (e.g., "the timing constraints discussed in the companion document") that no regex can match. Supporting both maximises recall while ensuring a deterministic baseline per the fail-safe-over-fail-fast principle.
> **Acceptance Criteria:** Given text containing both "per JEDEC JESD79-4" (explicit) and "as described in the power delivery analysis performed last quarter" (implicit), the regex path extracts the JEDEC reference, and the LLM path identifies the implicit reference to a power delivery analysis document. When the LLM is unavailable, only the regex-extracted references are returned and the implicit reference is not extracted.

> **FR-904** | Priority: MUST
> **Description:** Detected reference types MUST include: explicit (section/document citations), standard (IEEE, JEDEC, ISO), version, dependency, and implicit references.
> **Rationale:** Different reference types serve different retrieval and impact analysis needs. Version references enable traceability across document revisions; dependency references enable impact analysis when upstream documents change; standard references connect internal documents to external normative sources.
> **Acceptance Criteria:** Given a document containing "See Section 5 of SPEC-CLK-001 v2.3", "compliant with IEEE 1801-2015 (UPF)", "This design depends on the PDN model from PWR-ANALYSIS-003", and "using the approach outlined earlier", the system produces references of types: explicit (Section 5 of SPEC-CLK-001), standard (IEEE 1801-2015), version (v2.3), dependency (PWR-ANALYSIS-003), and implicit ("the approach outlined earlier"). Each reference carries a `type` field matching one of the five enumerated types.

> **FR-905** | Priority: MUST
> **Description:** Duplicate references MUST be merged, with LLM-extracted versions taking priority.
> **Rationale:** The same reference may be detected by both regex and LLM extraction. Merging avoids inflating the reference count and cluttering the knowledge graph, while prioritising the LLM version ensures richer metadata (e.g., the LLM may resolve an implicit reference target that regex cannot).
> **Acceptance Criteria:** Given that regex extracts a reference to "IEEE 1149.1" with type "standard" and the LLM also extracts a reference to "IEEE 1149.1 (JTAG boundary scan standard)" with type "standard" and additional context, the system produces a single merged reference to "IEEE 1149.1" retaining the LLM's enriched description. The final reference list contains no duplicate target identifiers.

---

### 3.10 Knowledge Graph Extraction (FR-1000)

> **FR-1001** | Priority: MUST
> **Description:** The system MUST extract structured subject-predicate-object triples from document content to build a knowledge graph enabling relationship-aware retrieval and impact analysis.
> **Rationale:** Vector similarity search alone cannot answer relational queries such as "which specifications constrain this design block?" or "what changes if we update the PDN model?". A knowledge graph captures these structural relationships, directly addressing the knowledge fragmentation problem by making inter-document dependencies explicit and queryable.
> **Acceptance Criteria:** Given a chunk stating "The clock distribution network uses a H-tree topology and must meet the skew budget defined in SPEC-CLK-001", the system extracts at least two triples: ("clock distribution network", "uses", "H-tree topology") and ("clock distribution network", "constrained_by", "SPEC-CLK-001"). Triples are stored as structured objects with subject, predicate, and object fields.

> **FR-1002** | Priority: MUST
> **Description:** Knowledge graph extraction MUST be skippable via configuration.
> **Rationale:** Supports configuration-driven-behaviour. Knowledge graph extraction involves significant LLM cost and processing time; teams that do not need relationship-aware retrieval can disable it to reduce cost and latency without affecting core vector search functionality.
> **Acceptance Criteria:** Given a pipeline configuration with `knowledge_graph.enabled: false`, the KG extraction stage is skipped (no LLM calls, no structural triple generation). The pipeline log records the stage as "skipped". Downstream stages (Quality Validation, Embedding & Storage) proceed normally without KG data.

> **FR-1003** | Priority: MUST
> **Description:** The system MUST generate structural triples deterministically (document-chunk containment, adjacency, domain membership, authorship, abbreviation mappings, cross-references).
> **Rationale:** Supports idempotency-by-construction. Structural triples derived from document hierarchy are always available regardless of LLM availability, providing a guaranteed baseline knowledge graph. Deterministic generation ensures re-ingesting the same document always produces the same structural triples.
> **Acceptance Criteria:** Given a document "SPEC-CLK-001" by author "J. Smith" in domain "Physical Design" with chunks C1 and C2 (adjacent), the system produces triples including: ("SPEC-CLK-001", "contains", "C1"), ("SPEC-CLK-001", "contains", "C2"), ("C1", "adjacent_to", "C2"), ("SPEC-CLK-001", "belongs_to_domain", "Physical Design"), ("SPEC-CLK-001", "authored_by", "J. Smith"). Re-ingesting the same document produces identical triples.

> **FR-1004** | Priority: MUST
> **Description:** The system MUST consolidate entities across chunks via exact-match deduplication, abbreviation resolution, and fuzzy matching.
> **Rationale:** The same entity often appears in different forms across chunks (e.g., "Clock Data Recovery" in one chunk, "CDR" in another, "clock data recovery circuit" in a third). Without consolidation, the knowledge graph fragments into disconnected nodes representing the same real-world entity, defeating the purpose of relationship-aware retrieval.
> **Acceptance Criteria:** Given three chunks where Chunk A mentions "Clock Data Recovery", Chunk B mentions "CDR", and Chunk C mentions "clock data recovery", all three are consolidated into a single entity node "Clock Data Recovery" with alias "CDR". The knowledge graph contains one node, not three. Fuzzy matching merges "clock data recovery circuit" with "Clock Data Recovery" but does not merge "CDR" (Clock Data Recovery) with "CDR" (Critical Design Review) when domain context disambiguates them.

> **FR-1005** | Priority: MUST
> **Description:** The system MUST optionally use LLM-based relationship extraction with a controlled predicate vocabulary.
> **Rationale:** A controlled predicate vocabulary (e.g., "uses", "constrained_by", "implements", "depends_on") ensures the knowledge graph is queryable with consistent predicates rather than free-form natural language, while LLM extraction captures semantic relationships that structural analysis cannot detect.
> **Acceptance Criteria:** Given a controlled vocabulary of 15 predicates and a chunk stating "The OCV derating factors are applied during static timing analysis", the LLM extracts a triple ("OCV derating factors", "applied_during", "static timing analysis") where "applied_during" is from the controlled vocabulary. If the LLM proposes a predicate not in the vocabulary (e.g., "are used in"), the system maps it to the nearest controlled predicate or discards the triple.

> **FR-1006** | Priority: MUST
> **Description:** All LLM-extracted triples MUST be validated: subjects and objects must appear in the chunk content or entity list. Invalid triples MUST be discarded.
> **Rationale:** Supports context-preservation. LLMs may hallucinate plausible relationships involving entities not actually discussed in the source chunk. Storing ungrounded triples would introduce false relationships into the knowledge graph, leading to incorrect impact analysis and misleading retrieval results.
> **Acceptance Criteria:** Given a chunk that discusses "scan chain insertion" and "DFT Compiler" but not "formal verification", an LLM-extracted triple ("DFT Compiler", "supports", "formal verification") is discarded because "formal verification" does not appear in the chunk content or consolidated entity list. A triple ("DFT Compiler", "performs", "scan chain insertion") is retained because both subject and object are grounded.

> **FR-1007** | Priority: MUST
> **Description:** If LLM extraction fails, the system MUST build the knowledge graph from structural triples and entity consolidation only.
> **Rationale:** Supports fail-safe-over-fail-fast. A knowledge graph built from structural triples alone (containment, adjacency, domain membership) still provides value for document navigation and basic impact analysis, ensuring the pipeline produces useful output even when the LLM is unavailable.
> **Acceptance Criteria:** Given an LLM failure during KG extraction for a 30-page document with 45 chunks, the system produces structural triples (containment, adjacency, domain, authorship, abbreviation mappings) and consolidated entities without any LLM-derived relationship triples. The pipeline log records the LLM failure and indicates fallback to structural-only mode. The resulting knowledge graph is still navigable from document to chunks.

> **FR-1008** | Priority: MUST
> **Description:** Triple identifiers MUST be deterministic, derived from document ID, subject, predicate, and object.
> **Rationale:** Supports idempotency-by-construction. Deterministic triple IDs ensure that re-ingesting the same document produces triples with the same identifiers, enabling clean upsert operations during re-ingestion and preventing duplicate triples from accumulating in the graph store.
> **Acceptance Criteria:** Given a triple ("SPEC-CLK-001", "contains", "chunk_abc123") extracted from document "SPEC-CLK-001", the triple ID is a deterministic hash of (document_id, subject, predicate, object). Re-ingesting the same document produces a triple with the identical ID. Changing the object to "chunk_def456" produces a different triple ID.

> **FR-1009** | Priority: MUST
> **Description:** The maximum number of triples per chunk MUST be configurable.
> **Rationale:** Supports configuration-driven-behaviour. Dense technical chunks can produce hundreds of triples, inflating storage costs and slowing graph queries. A configurable cap lets teams balance knowledge graph richness against storage and performance constraints.
> **Acceptance Criteria:** Given a configuration `knowledge_graph.max_triples_per_chunk: 20` and a chunk from which the LLM extracts 35 triples, only the top 20 triples (ranked by validation confidence or extraction order) are retained. Changing the configuration to `max_triples_per_chunk: 50` retains all 35 triples. Setting it to `0` disables the cap.

---

### 3.11 Quality Validation (FR-1100)

> **FR-1101** | Priority: MUST
> **Description:** The system MUST remove chunks below a configurable minimum token count.
> **Rationale:** Very short chunks (e.g., a section header with no body text) produce low-quality embeddings that pollute search results with near-meaningless matches. Removing them improves retrieval precision without losing substantive content.
> **Acceptance Criteria:** Given a configuration `quality.min_token_count: 50` and a chunk containing only "3.2 Clock Tree Results" (5 tokens), the chunk is removed from the pipeline. A chunk with 51 tokens is retained. Changing the threshold to `min_token_count: 3` retains the 5-token chunk.

> **FR-1102** | Priority: MUST
> **Description:** The system MUST detect and remove near-duplicate chunks based on content similarity exceeding a configurable threshold.
> **Rationale:** Document boilerplate (e.g., repeated disclaimers, identical headers across sections) produces near-duplicate chunks that waste storage and dilute retrieval results by returning effectively the same content multiple times.
> **Acceptance Criteria:** Given a threshold of `quality.dedup_similarity_threshold: 0.95` and two chunks with 97% content similarity (e.g., identical disclaimer paragraphs from two sections), one chunk is removed as a near-duplicate. Two chunks with 90% similarity (e.g., similar but distinct design rule descriptions for different metal layers) are both retained. The removed chunk is logged with the ID of the chunk it duplicates.

> **FR-1103** | Priority: MUST
> **Description:** The system MUST assign a quality score (0.0-1.0) to each surviving chunk based on content signals (technical term density, numerical density, structured content, chunk length, whitespace ratio, boilerplate presence, extraction confidence).
> **Rationale:** Not all chunks carry equal informational value. Quality scores enable retrieval-time weighting so that dense, information-rich chunks (e.g., a timing constraints table) rank higher than sparse, boilerplate-heavy chunks, improving answer quality without discarding low-scoring content entirely.
> **Acceptance Criteria:** Given a chunk containing a timing constraints table with 12 numerical values and 8 technical terms, the quality score is above 0.7. Given a chunk containing a generic project introduction with no numerical values and 1 technical term, the quality score is below 0.4. Scores are floating-point values in the range [0.0, 1.0] with at least two decimal places of precision.

> **FR-1104** | Priority: MUST
> **Description:** Quality scores MUST be stored with the chunk and available for retrieval-time quality weighting.
> **Rationale:** Supports controlled-access-over-restriction. Rather than discarding low-quality chunks outright, storing quality scores allows the retrieval layer to apply configurable weighting, ensuring that even low-quality chunks remain searchable when a user broadens their search.
> **Acceptance Criteria:** Given a stored chunk in the vector database, the chunk object includes a `quality_score` field of type float. A retrieval query can filter or boost results using the quality score (e.g., `quality_score >= 0.5`). The quality score is present on 100% of stored chunks.

> **FR-1105** | Priority: MUST
> **Description:** After removing low-quality and duplicate chunks, the system MUST repair adjacency links to skip over removed chunks.
> **Rationale:** Supports context-preservation. Adjacency links enable the retrieval layer to fetch surrounding chunks for context. If a removed chunk breaks the adjacency chain, the retrieval layer loses the ability to reconstruct reading order, reintroducing the context loss problem described in the problem statement.
> **Acceptance Criteria:** Given chunks [C1, C2, C3, C4] where C2 is removed as a near-duplicate and C3 is removed for being below minimum token count, the adjacency links are repaired so that C1.next = C4 and C4.previous = C1. Before repair, C1.next = C2. After repair, no adjacency link points to a removed chunk.

---

### 3.12 Embedding & Storage (FR-1200)

> **FR-1201** | Priority: MUST
> **Description:** The system MUST generate vector embeddings for all surviving chunks using the configured embedding model.
> **Rationale:** Vector embeddings are the core output of the pipeline, enabling semantic similarity search. Without embeddings, chunks cannot be retrieved by meaning, and the system fails its primary purpose of transforming documents into semantically searchable representations.
> **Acceptance Criteria:** Given 45 surviving chunks after quality validation, the system produces exactly 45 embedding vectors, one per chunk. Each vector has the dimensionality specified by the configured embedding model (e.g., 1024 for a model configured with `embedding.dimension: 1024`). No chunk is stored without an embedding.

> **FR-1202** | Priority: MUST
> **Description:** The embedding model, dimension, and provider MUST be configurable.
> **Rationale:** Supports swappability-over-lock-in. Embedding models evolve rapidly; teams must be able to switch from one provider (e.g., OpenAI) to another (e.g., Cohere, a local model) without code changes, enabling cost optimisation and adaptation to domain-specific models as they become available.
> **Acceptance Criteria:** Given a configuration change from `embedding.model: "text-embedding-3-large"` and `embedding.provider: "openai"` to `embedding.model: "embed-english-v3.0"` and `embedding.provider: "cohere"`, the system uses the Cohere model without any code changes. The embedding dimension updates accordingly. An invalid provider name produces a clear configuration error at startup, not at embedding time.

> **FR-1203** | Priority: MUST
> **Description:** The system MUST validate that the actual embedding dimension matches the configured dimension on the first batch. Mismatched dimensions MUST halt the stage.
> **Rationale:** A dimension mismatch between the embedding model and the vector store schema causes silent data corruption: vectors are stored but produce meaningless similarity scores. Halting immediately prevents an entire batch of documents from being stored with unusable embeddings.
> **Acceptance Criteria:** Given a configuration `embedding.dimension: 1024` but an embedding model that produces 768-dimensional vectors, the system detects the mismatch on the first batch of embeddings, logs an error including both expected (1024) and actual (768) dimensions, and halts the embedding stage. No chunks are written to the vector store. When dimensions match (both 1024), the stage proceeds normally.

> **FR-1204** | Priority: MUST
> **Description:** The system MUST store chunks with their embeddings, metadata (32+ properties), and pre-computed vectors in the vector database.
> **Rationale:** Rich metadata stored alongside embeddings enables hybrid retrieval (vector + metadata filtering) without requiring separate lookups, which is essential for scoping queries to specific domains, document types, review tiers, or technology nodes in a semiconductor engineering context.
> **Acceptance Criteria:** Given a chunk from a 5nm standard cell library specification, the stored vector database object includes: the embedding vector, the chunk text, and at least 32 metadata properties including document_id, chunk_id, section_hierarchy, domain, document_type, review_tier, quality_score, keywords, named_entities, source_file, and content_hash. All 32+ properties are queryable as filters.

> **FR-1205** | Priority: MUST
> **Description:** The system MUST use Bring Your Own Model (BYOM) mode, computing embeddings externally and passing pre-computed vectors to the vector store.
> **Rationale:** Supports swappability-over-lock-in. BYOM decouples the embedding model from the vector store, allowing the team to use any embedding model (including fine-tuned or locally hosted models) regardless of the vector store provider's native model support.
> **Acceptance Criteria:** Given a locally hosted embedding model that produces 1024-dimensional vectors, the system computes embeddings using the local model and stores the pre-computed vectors in the vector database (e.g., Weaviate) without invoking any vectoriser module built into the vector store. The vector store is configured in BYOM/none-vectoriser mode.

> **FR-1206** | Priority: MUST
> **Description:** The system MUST support asymmetric embedding prefixes (different prefixes for documents vs queries) as required by certain embedding models.
> **Rationale:** Some embedding models (e.g., E5, BGE) require different input prefixes for documents ("passage:") and queries ("query:") to produce embeddings in a shared space. Without prefix support, these models produce misaligned embeddings that degrade retrieval quality.
> **Acceptance Criteria:** Given a configuration `embedding.document_prefix: "passage: "` and `embedding.query_prefix: "query: "`, document chunks are embedded with the "passage: " prefix prepended to the text. The query prefix is stored in configuration for use by the downstream retrieval layer. When prefix configuration is empty, no prefix is prepended.

> **FR-1207** | Priority: MUST
> **Description:** The system MUST support hybrid search (vector similarity + BM25 keyword search) in the vector store.
> **Rationale:** Pure vector search struggles with exact term matching (e.g., searching for a specific specification ID like "SPEC-CLK-001"), while pure keyword search misses semantic matches. Hybrid search combines both strengths, which is critical in engineering documentation where both precise identifiers and conceptual queries are common.
> **Acceptance Criteria:** Given a vector store configured with hybrid search, a query for "SPEC-CLK-001 timing constraints" returns results ranked by a fusion of vector similarity (semantic match on "timing constraints") and BM25 keyword match (exact match on "SPEC-CLK-001"). A chunk containing the exact string "SPEC-CLK-001" ranks higher than a semantically similar chunk that does not contain the identifier.

> **FR-1208** | Priority: MUST
> **Description:** The system MUST support a dry-run mode that executes the full pipeline without writing to external stores.
> **Rationale:** Supports configuration-driven-behaviour. Dry-run mode enables pipeline validation, cost estimation, and debugging without side effects, which is essential when tuning chunking strategies or testing new embedding models before committing to production data changes.
> **Acceptance Criteria:** Given a configuration `pipeline.dry_run: true`, the pipeline processes a document through all stages (ingestion, chunking, embedding) but writes zero records to the vector store and zero triples to the graph store. The pipeline log reports the number of chunks, embeddings, and triples that would have been written. Setting `pipeline.dry_run: false` enables writes.

> **FR-1209** | Priority: MUST
> **Description:** The embedding provider and vector store MUST be swappable via configuration.
> **Rationale:** Supports swappability-over-lock-in. The organisation must be able to migrate from one vector store (e.g., Weaviate) to another (e.g., Qdrant, Pinecone) or switch embedding providers without modifying pipeline code, protecting against vendor lock-in and enabling competitive evaluation.
> **Acceptance Criteria:** Given a configuration change from `vector_store.provider: "weaviate"` to `vector_store.provider: "qdrant"`, the system stores chunks in Qdrant without code changes. Both providers support the same set of metadata filters and hybrid search. An unsupported provider name produces a clear error at startup.

---

### 3.13 Knowledge Graph Storage (FR-1300)

> **FR-1301** | Priority: MUST
> **Description:** The system MUST persist knowledge graph triples to a configurable graph store.
> **Rationale:** Extracted triples are only useful if persisted for query-time retrieval. A configurable store ensures the KG backend can evolve independently of the extraction logic, consistent with swappability-over-lock-in.
> **Acceptance Criteria:** Given 200 extracted triples for a document, all 200 triples are persisted to the configured graph store and are queryable after ingestion completes. A query for all triples with subject "SPEC-CLK-001" returns the expected containment, authorship, and relationship triples.

> **FR-1302** | Priority: MUST
> **Description:** The system MUST support at least two graph storage backends: vector store cross-references and a dedicated graph database.
> **Rationale:** Supports swappability-over-lock-in. Teams without a dedicated graph database can store cross-references as properties in the vector store (lower capability but zero additional infrastructure), while teams needing full graph traversal can use a dedicated graph database (e.g., Neo4j). This graduated approach avoids forcing infrastructure requirements on all deployments.
> **Acceptance Criteria:** Given a configuration `graph_store.backend: "vector_store_refs"`, triples are stored as cross-reference properties on vector store objects. Given `graph_store.backend: "neo4j"`, triples are stored as nodes and edges in Neo4j. Both backends support triple insertion, deletion by document ID, and lookup by subject. Switching between backends requires only a configuration change.

> **FR-1303** | Priority: MUST
> **Description:** The graph storage provider MUST be swappable via configuration.
> **Rationale:** Supports swappability-over-lock-in. As the knowledge graph matures, the team may need to migrate from a lightweight backend to a full graph database, and this migration must not require code changes.
> **Acceptance Criteria:** Given a configuration change from `graph_store.provider: "weaviate_refs"` to `graph_store.provider: "neo4j"`, the system persists triples to Neo4j without code changes. The pipeline produces identical triples regardless of the configured backend. An invalid provider name produces a clear error at startup.

> **FR-1304** | Priority: MUST
> **Description:** The system MUST support a dry-run mode for KG storage.
> **Rationale:** Supports configuration-driven-behaviour. Dry-run mode for KG storage enables teams to preview the knowledge graph that would be generated (triple count, entity count, relationship types) without committing data, which is essential for validating KG extraction configuration before production runs.
> **Acceptance Criteria:** Given a configuration `pipeline.dry_run: true`, the KG storage stage logs the number of triples and entities that would be persisted but writes zero records to the graph store. The triples are available in the pipeline's in-memory state for inspection. Setting `pipeline.dry_run: false` enables writes.

---

## 4. Re-ingestion Requirements (FR-1400)

> **FR-1401** | Priority: MUST
> **Description:** The system MUST detect when a document has been previously ingested by querying the vector store for existing chunks with the same document ID.
> **Rationale:** Re-ingestion detection is the prerequisite for idempotent document updates. Without it, the system would blindly insert duplicate chunks on every run, inflating the vector store and returning duplicate results for every query.
> **Acceptance Criteria:** Given a document "SPEC-CLK-001" that was previously ingested (45 chunks in the vector store), the system queries the vector store by document_id, detects the existing 45 chunks, and flags the document as "previously ingested" in the pipeline state. For a never-ingested document "SPEC-NEW-001", the query returns zero results and the document is flagged as "new".

> **FR-1402** | Priority: MUST
> **Description:** The system MUST compare content hashes to determine whether the document has changed since last ingestion.
> **Rationale:** Supports idempotency-by-construction. Content hash comparison provides a reliable, fast mechanism to determine whether re-processing is needed, avoiding unnecessary LLM calls and compute costs for unchanged documents while ensuring changed documents are always re-processed.
> **Acceptance Criteria:** Given a previously ingested document whose stored content hash is "abc123" and the current file's computed hash is "abc123", the system determines the document is unchanged. If the current hash is "def456", the system determines the document has changed. The hash algorithm is deterministic: the same file content always produces the same hash.

> **FR-1403** | Priority: MUST
> **Description:** If the document is unchanged and the strategy is "skip unchanged", the system MUST skip processing entirely (no-op).
> **Rationale:** Supports idempotency-by-construction. Re-processing an unchanged document wastes compute, incurs LLM costs, and risks introducing non-determinism. Skipping unchanged documents is essential for efficient batch re-ingestion of document directories where most documents have not changed.
> **Acceptance Criteria:** Given a batch of 100 documents where 95 are unchanged and the strategy is `reingestion.strategy: "skip_unchanged"`, the system processes only 5 documents. The 95 unchanged documents produce zero LLM calls, zero new chunks, and zero vector store writes. The pipeline log records each skipped document with reason "unchanged (hash match)".

> **FR-1404** | Priority: MUST
> **Description:** If the document has changed, the system MUST process it through the full pipeline and then clean up all previous data (chunks, embeddings, KG triples) before inserting new data.
> **Rationale:** Stale data from a previous version of a document must be removed to prevent contradictory information from coexisting in the vector store (e.g., old timing constraints alongside new ones). Processing before cleanup ensures new data is ready before old data is removed, minimising the window of data unavailability.
> **Acceptance Criteria:** Given document "SPEC-CLK-001 v2" (changed from v1, which had 45 chunks), the system: (1) processes v2 through the full pipeline producing 50 new chunks, (2) deletes all 45 old chunks and their embeddings from the vector store, (3) deletes all KG triples owned by the old version, (4) inserts the 50 new chunks. After completion, the vector store contains exactly 50 chunks for "SPEC-CLK-001", all from v2.

> **FR-1405** | Priority: MUST
> **Description:** Re-ingestion cleanup MUST be fail-safe: if cleanup of old data fails, the system MUST halt and NOT insert new data on top of stale data. Partial state (stale + new data coexisting) MUST NOT occur.
> **Rationale:** Supports fail-safe-over-fail-fast. Partial state (old and new chunks coexisting for the same document) would cause the retrieval layer to return contradictory results — e.g., both the old 0.9V supply voltage and the new 0.75V supply voltage for the same design block — which could propagate into design errors.
> **Acceptance Criteria:** Given a re-ingestion where cleanup of old chunks fails (e.g., vector store connection timeout during deletion), the system halts and does not insert the new chunks. The vector store retains only the old 45 chunks (consistent state). The pipeline log records the failure with the error detail. A subsequent retry re-attempts the full cleanup-then-insert sequence.

> **FR-1406** | Priority: MUST
> **Description:** If re-ingestion processing fails upstream (zero new chunks produced due to errors), the system MUST abort cleanup and preserve existing data. Data loss (deleting old data with nothing to replace it) MUST NOT occur.
> **Rationale:** Supports fail-safe-over-fail-fast. If the pipeline fails to produce new chunks (e.g., due to a parsing error in the updated document), deleting the old data would leave the document with zero searchable content — a worse outcome than retaining the stale but functional previous version.
> **Acceptance Criteria:** Given a re-ingestion of "SPEC-CLK-001 v2" where structure detection fails and produces zero chunks, the system does not delete the existing 45 chunks from v1. The pipeline log records "re-ingestion aborted: zero new chunks produced, preserving existing data". The vector store retains the 45 v1 chunks unchanged.

> **FR-1407** | Priority: MUST
> **Description:** Re-ingesting an unchanged document MUST produce no new data and no side effects (idempotent).
> **Rationale:** Supports idempotency-by-construction. Running the pipeline twice on the same unchanged document must be indistinguishable from running it once, ensuring that batch re-ingestion scripts can be safely re-run without corrupting or duplicating data.
> **Acceptance Criteria:** Given document "SPEC-CLK-001" ingested at time T1, re-ingesting the same unchanged file at T2 produces: zero new chunks, zero deleted chunks, zero new KG triples, zero deleted KG triples, zero LLM calls (beyond the initial hash check). The vector store state at T2 is byte-identical to T1. The pipeline log records "skipped: unchanged".

> **FR-1408** | Priority: MUST
> **Description:** The system MUST support two re-ingestion strategies: "skip unchanged" and "delete and reinsert".
> **Rationale:** Supports configuration-driven-behaviour. "Skip unchanged" is optimal for routine batch updates where most documents are stable. "Delete and reinsert" is needed when the pipeline configuration has changed (e.g., new chunking strategy or embedding model) and all documents must be reprocessed regardless of content changes.
> **Acceptance Criteria:** Given `reingestion.strategy: "skip_unchanged"`, an unchanged document is skipped. Given `reingestion.strategy: "delete_and_reinsert"`, the same unchanged document is fully reprocessed: old chunks are deleted and new chunks (produced by the current pipeline configuration) are inserted. Both strategies are selectable via configuration without code changes.

> **FR-1409** | Priority: MUST
> **Description:** KG cleanup for shared graph nodes MUST use a two-phase approach: delete edges owned by the document first, then garbage-collect nodes only referenced by that document. Shared nodes referenced by other documents MUST be preserved.
> **Rationale:** Knowledge graph nodes may be shared across documents (e.g., the entity "TSMC N5" appears in many specifications). Naively deleting all nodes associated with a re-ingested document would destroy shared entities and break triples from other documents, corrupting the knowledge graph.
> **Acceptance Criteria:** Given documents A and B that both reference entity "TSMC N5", re-ingesting document A: (1) deletes all edges owned by document A (e.g., ("SPEC-A", "targets", "TSMC N5")), (2) checks whether "TSMC N5" is referenced by any other document, (3) finds that document B still references "TSMC N5" and preserves the node. If document B is subsequently deleted and "TSMC N5" has no remaining references, the garbage collector removes the orphaned node. The node reference count is verified before and after cleanup.
## 5. Review Tier Requirements (FR-1500)

### 5.1 Tier Definitions

> **FR-1501** | Priority: MUST
> **Description:** The system MUST implement a three-tier review system: Fully Reviewed (Tier 1), Partially Reviewed (Tier 2), and Self-Reviewed (Tier 3).
> **Rationale:** Engineering knowledge exists at varying maturity levels (controlled-access-over-restriction). A tiered system prevents conflating an approved specification with an engineer's personal notes, directly addressing the problem of indistinguishable authority levels.
> **Acceptance Criteria:** Given the system is initialised, when the review tier enumeration is inspected, then exactly three tiers exist: Fully Reviewed (Tier 1), Partially Reviewed (Tier 2), and Self-Reviewed (Tier 3). Negative: attempting to assign a tier value outside these three (e.g., "Tier 0" or "Unreviewed") is rejected.

> **FR-1502** | Priority: MUST
> **Description:** **Tier 1 — Fully Reviewed:** Formally reviewed documents with domain lead sign-off. MUST always be included in default search results. Represents authoritative, design-decision-grade content.
> **Rationale:** In ASIC design, relying on unverified content for design decisions (e.g., voltage specifications, timing constraints) can propagate errors into silicon. Tier 1 ensures default search surfaces only sign-off-grade content (controlled-access-over-restriction).
> **Acceptance Criteria:** Given a Tier 1 document (e.g., a signed-off 7nm PDK specification), when a user performs a default search, then Tier 1 results appear. Given a Tier 2 or Tier 3 document, when a user performs a default search, then those results do not appear.

> **FR-1503** | Priority: MUST
> **Description:** **Tier 2 — Partially Reviewed:** Documents with at least one peer review but not yet signed off. MUST be included in expanded search results with a visual indicator. Represents informational content.
> **Rationale:** Peer-reviewed but unsigned content (e.g., a DFT methodology guide reviewed by a colleague) has value but must be visually distinguished from authoritative sources to prevent accidental reliance on non-final content (controlled-access-over-restriction).
> **Acceptance Criteria:** Given a Tier 2 document (e.g., a peer-reviewed clock domain crossing guide), when a user performs an expanded search, then the document appears with a visual indicator distinguishing it from Tier 1 results. Negative: Tier 2 results do not appear in default (Tier 1-only) searches.

> **FR-1504** | Priority: MUST
> **Description:** **Tier 3 — Self-Reviewed:** Documents where the author self-certifies. MUST only be included when the user explicitly expands the search space. Represents community/informal knowledge.
> **Rationale:** Informal knowledge (e.g., an engineer's personal runbook for analog simulation setup) should be searchable but never surfaced alongside authoritative specs unless explicitly requested (controlled-access-over-restriction).
> **Acceptance Criteria:** Given a Tier 3 document (e.g., a self-certified personal simulation script guide), when a user performs a full search, then the document appears. Negative: Tier 3 results do not appear in default or expanded searches.

### 5.2 Tier Lifecycle

> **FR-1510** | Priority: MUST
> **Description:** The system MUST support tier promotion: Self-Reviewed → Partially Reviewed (via peer review) → Fully Reviewed (via domain lead sign-off).
> **Rationale:** Documents mature over time; a personal runbook may be peer-reviewed and eventually formally approved. The system must support this natural lifecycle (controlled-access-over-restriction).
> **Acceptance Criteria:** Given a Tier 3 document, when a peer review is recorded, then the document is promoted to Tier 2. Given a Tier 2 document, when a domain lead sign-off is recorded, then the document is promoted to Tier 1. Negative: attempting to promote directly from Tier 3 to Tier 1 (skipping peer review) is rejected.

> **FR-1511** | Priority: MUST
> **Description:** The system MUST support tier demotion: Fully Reviewed → Partially Reviewed (via major revision or re-ingestion with changes).
> **Rationale:** A previously approved document that undergoes major revision is no longer verified as authoritative until re-reviewed. Failing to demote risks surfacing stale-approved content (fail-safe-over-fail-fast).
> **Acceptance Criteria:** Given a Tier 1 document, when a major revision triggers demotion, then the document becomes Tier 2. Negative: demotion below Tier 2 (e.g., directly to Tier 3) does not occur via this mechanism.

> **FR-1512** | Priority: MUST
> **Description:** When a Fully Reviewed document is re-ingested with content changes, the system MUST auto-demote it to Partially Reviewed and flag the demotion as automatic.
> **Rationale:** If a signed-off ASIC power specification is re-ingested with changed voltage values, it is no longer the same approved document. Auto-demotion prevents stale approvals from persisting (fail-safe-over-fail-fast, idempotency-by-construction).
> **Acceptance Criteria:** Given a Tier 1 document with content hash H1, when re-ingested with a different content hash H2, then the document is demoted to Tier 2 and the demotion is flagged as "automatic". Given a Tier 1 document re-ingested with unchanged content, then no demotion occurs.

> **FR-1513** | Priority: MUST
> **Description:** Review tier changes MUST NOT require re-ingestion. Tier updates MUST be property updates on existing stored objects.
> **Rationale:** Re-ingesting a document just to change its review status would be wasteful and could alter chunk boundaries. Tier is an administrative property, not a content property (configuration-driven-behaviour).
> **Acceptance Criteria:** Given a stored document with 15 chunks at Tier 3, when the tier is promoted to Tier 2, then all 15 chunks reflect the new tier without re-running the pipeline. The chunk content, IDs, and embeddings remain unchanged.

> **FR-1514** | Priority: MUST
> **Description:** The default review tier for new documents MUST be configurable (default: Self-Reviewed).
> **Rationale:** Different organisations may have different trust baselines. A team that pre-reviews all documents before ingestion may want Tier 2 as default (configuration-driven-behaviour).
> **Acceptance Criteria:** Given default configuration, when a new document is ingested without specifying a tier, then it is assigned Tier 3 (Self-Reviewed). Given configuration overriding default tier to Tier 2, when a new document is ingested, then it is assigned Tier 2.

### 5.3 Retrieval-Time Filtering

> **FR-1520** | Priority: MUST
> **Description:** The system MUST support three search spaces at retrieval time: Default (Tier 1 only), Expanded (Tier 1 + Tier 2), and Full (all tiers).
> **Rationale:** Engineers need the ability to widen or narrow their search depending on context — design-critical decisions use Default, exploratory research uses Full (controlled-access-over-restriction).
> **Acceptance Criteria:** Given documents across all three tiers, when a Default search is executed, then only Tier 1 results are returned. When an Expanded search is executed, then Tier 1 and Tier 2 results are returned. When a Full search is executed, then all tiers are returned.

> **FR-1521** | Priority: MUST
> **Description:** Review tier filtering MUST be applied at query time, not at ingestion time. All documents MUST be stored regardless of tier.
> **Rationale:** Filtering at ingestion time would require re-ingestion to change visibility. Storing everything and filtering at query time supports tier promotion without re-processing (controlled-access-over-restriction, idempotency-by-construction).
> **Acceptance Criteria:** Given a Tier 3 document is ingested, when the vector store is inspected, then all chunks from that document are present. When a Default search is executed, then those chunks are excluded by the query filter, not by absence from the store.

---

## 6. Domain Vocabulary Requirements (FR-1600)

> **FR-1601** | Priority: MUST
> **Description:** The system MUST support a domain vocabulary dictionary in a structured format (e.g., YAML) containing abbreviations, expansions, domains, context notes, related terms, and compound terms.
> **Rationale:** ASIC/semiconductor engineering uses dense abbreviations (DFT, CDC, PVT, LVS) where meaning depends on context. A structured vocabulary is the foundation for consistent term handling across the pipeline (context-preservation).
> **Acceptance Criteria:** Given a YAML vocabulary file containing an entry for "DFT" with expansion "Design for Testability", domain "verification", and related terms ["scan", "ATPG"], when loaded, then all fields are accessible to downstream stages. Negative: a vocabulary file missing the required schema fields (e.g., no "expansion" key) is rejected with a validation error.

> **FR-1602** | Priority: MUST
> **Description:** The vocabulary MUST support ambiguous abbreviations with multiple expansions disambiguated by domain context (e.g., "CDR" = "Critical Design Review" in general context, "Clock Data Recovery" in analog context).
> **Rationale:** Term ambiguity is a core problem (problem statement item 1). "CDR" in a project management document means something entirely different from "CDR" in a SerDes design guide. Domain-aware disambiguation prevents incorrect expansion (context-preservation).
> **Acceptance Criteria:** Given a vocabulary entry for "CDR" with two expansions — "Critical Design Review" (domain: project_management) and "Clock Data Recovery" (domain: analog) — when processing a document classified as "analog", then "CDR" resolves to "Clock Data Recovery". When processing a document classified as "project_management", then "CDR" resolves to "Critical Design Review".

> **FR-1603** | Priority: MUST
> **Description:** The vocabulary MUST be injectable into all LLM prompts across the pipeline to ensure consistent abbreviation handling.
> **Rationale:** Without vocabulary injection, each LLM call independently interprets domain abbreviations, leading to inconsistent expansions across stages (context-preservation, configuration-driven-behaviour).
> **Acceptance Criteria:** Given a vocabulary with 50 terms, when the chunking stage constructs its LLM prompt, then the relevant vocabulary terms are included. When the metadata generation stage constructs its prompt, then vocabulary terms are also included. Negative: no LLM-calling stage omits vocabulary injection.

> **FR-1604** | Priority: MUST
> **Description:** The number of vocabulary terms injected into prompts MUST be configurable to manage prompt size.
> **Rationale:** Injecting the full vocabulary (potentially hundreds of terms) into every prompt wastes tokens and may exceed context windows. Configurability balances coverage against cost (configuration-driven-behaviour).
> **Acceptance Criteria:** Given a vocabulary with 200 terms and max_prompt_terms configured to 50, when an LLM prompt is constructed, then at most 50 vocabulary terms are included. Given max_prompt_terms set to 0, then no terms are injected.

> **FR-1605** | Priority: MUST
> **Description:** The system MUST auto-detect abbreviation definitions within documents (e.g., "Design for Testability (DFT)", abbreviation tables) and merge them with the domain vocabulary for the current processing run.
> **Rationale:** Documents often define their own abbreviations inline or in glossary tables. Auto-detection captures document-specific terms that may not exist in the master vocabulary, improving downstream expansion accuracy (context-preservation).
> **Acceptance Criteria:** Given a document containing the text "Phase-Locked Loop (PLL)" where "PLL" is not in the domain vocabulary, when structure detection completes, then "PLL" → "Phase-Locked Loop" is available in the merged vocabulary for subsequent stages. Given an abbreviation table in the document listing "ESD" → "Electrostatic Discharge", then this mapping is also merged.

> **FR-1606** | Priority: MUST
> **Description:** The compound terms list MUST inform chunking to avoid splitting multi-word domain terms across chunk boundaries.
> **Rationale:** Splitting "clock domain crossing" across two chunks degrades both chunks — one has "clock domain" without "crossing", the other has "crossing" without context. Compound term awareness preserves term integrity (context-preservation).
> **Acceptance Criteria:** Given a compound term "clock domain crossing" in the vocabulary, when chunking a document, then this three-word term is never split across chunk boundaries. Negative: if "clock domain crossing" is not in the compound terms list, the chunker is not obligated to keep it together.

---

## 7. Error Handling Requirements (FR-1700)

> **FR-1701** | Priority: MUST
> **Description:** Processing stage failures MUST NOT crash the pipeline. Errors MUST be captured, logged, and the document MUST continue to the next stage with whatever state it had before the failure.
> **Rationale:** In batch processing of hundreds of engineering documents, a single parsing failure (e.g., a corrupted PDF table) must not halt the entire job. The pipeline must be resilient (fail-safe-over-fail-fast).
> **Acceptance Criteria:** Given a document where the metadata generation stage throws an exception, when the pipeline continues, then the document proceeds to the next stage with metadata fields empty/default. The error is logged with stage name, document ID, and exception details. Negative: the pipeline does not terminate or skip the entire document.

> **FR-1702** | Priority: MUST
> **Description:** Every stage that makes LLM calls MUST have a deterministic fallback that produces a usable (if lower quality) result.
> **Rationale:** LLM services are inherently unreliable (rate limits, timeouts, malformed responses). A deterministic fallback ensures the pipeline always produces output, even if degraded (fail-safe-over-fail-fast).
> **Acceptance Criteria:** Given that the LLM service is unavailable, when the chunking stage executes, then it falls back to the recursive character splitter and produces valid chunks. This applies to all six LLM-dependent stages (see 7.1 LLM Fallback Matrix). Negative: no LLM-dependent stage exists without a corresponding fallback implementation.

> **FR-1703** | Priority: MUST
> **Description:** The system MUST record a processing log with timestamped entries for every stage (started, completed, skipped, failed) with relevant metrics.
> **Rationale:** Without a processing log, diagnosing why a particular document produced poor-quality chunks requires re-running the pipeline with debug logging. Structured logs enable post-hoc analysis and auditing (fail-safe-over-fail-fast).
> **Acceptance Criteria:** Given a document processed through the full pipeline, when the processing log is inspected, then it contains one entry per stage with: stage name, status (started/completed/skipped/failed), timestamp, and stage-specific metrics (e.g., chunk count for chunking, triple count for KG extraction). Negative: no stage completes without writing a log entry.

> **FR-1704** | Priority: MUST
> **Description:** A document MUST be able to complete the pipeline with partial results. Missing data from failed stages MUST cause downstream stages to skip gracefully via input validation.
> **Rationale:** If cross-reference extraction fails, the document should still be chunked, embedded, and stored — just without cross-reference metadata. Partial results are better than no results (fail-safe-over-fail-fast).
> **Acceptance Criteria:** Given a document where the refactoring stage fails and returns original text, when chunking executes, then it operates on the original text and produces valid chunks. Given a document where KG extraction fails, when embedding and storage executes, then chunks are stored successfully without KG triples.

> **FR-1705** | Priority: MUST
> **Description:** LLM responses expected to be structured (JSON) MUST be parsed through a defensive parser that handles common LLM response formatting issues (code fences, leading/trailing text).
> **Rationale:** LLMs frequently wrap JSON in markdown code fences (```json ... ```) or prepend conversational text. A rigid JSON parser would fail on valid content due to formatting artifacts (fail-safe-over-fail-fast).
> **Acceptance Criteria:** Given an LLM response containing `"```json\n{\"keywords\": [\"PVT\", \"corner\"]}\n```"`, when parsed, then the JSON object is extracted successfully. Given an LLM response containing `"Here is the result: {\"keywords\": [\"LVS\"]}"`, when parsed, then the JSON object is extracted. Negative: given a response containing no valid JSON, then the parser returns a parse failure (not a crash).

> **FR-1706** | Priority: MUST
> **Description:** On JSON parse failure, the system MUST use stage-specific safe defaults that trigger the deterministic fallback path.
> **Rationale:** When the LLM returns unparseable output (e.g., truncated JSON from a timeout), the stage must not crash or produce corrupt data. Safe defaults activate the fallback, ensuring continuity (fail-safe-over-fail-fast).
> **Acceptance Criteria:** Given a chunking stage where the LLM returns invalid JSON, when the parse fails, then the stage returns a safe default that triggers the recursive character splitter fallback. Given a metadata stage where the LLM returns invalid JSON, then the stage falls back to TF-IDF keyword extraction.

### 7.1 LLM Fallback Matrix

| Stage | Primary (LLM) | Fallback (Deterministic) |
|-------|---------------|--------------------------|
| Chunking | Semantic chunking via LLM | Recursive character splitter on paragraph/sentence boundaries |
| Refactoring | Multi-pass agentic refactoring | Return original text unchanged |
| Metadata Generation | LLM keyword/entity extraction | TF-IDF frequency-based keyword extraction |
| Cross-Reference Extraction | LLM implicit reference detection | Regex-only extraction |
| Multimodal Processing | VLM image-to-text | Figure recorded without description |
| Knowledge Graph Extraction | LLM relationship extraction | Structural triples only |

---

## 8. Configuration Requirements (FR-1800)

### 8.1 General Configuration

> **FR-1801** | Priority: MUST
> **Description:** All pipeline behaviour MUST be driven by a single hierarchical configuration system.
> **Rationale:** Scattering configuration across environment variables, code constants, and config files creates inconsistency and makes reproducibility impossible. A single system ensures one source of truth (configuration-driven-behaviour).
> **Acceptance Criteria:** Given the pipeline is started, when configuration is loaded, then all configurable parameters (LLM provider, chunk sizes, skip flags, etc.) are resolved from the same configuration hierarchy. Negative: no pipeline behaviour is controlled by hard-coded constants that cannot be overridden via configuration.

> **FR-1802** | Priority: MUST
> **Description:** Configuration MUST support three-layer precedence: defaults → configuration file → command-line arguments. Command-line arguments MUST always take priority.
> **Rationale:** Defaults provide sensible baselines, config files capture team/project settings, and CLI arguments enable per-run overrides (e.g., `--dry-run` or `--skip-refactoring` for a quick test). This layered approach is standard practice (configuration-driven-behaviour).
> **Acceptance Criteria:** Given a default chunk size of 512, a config file setting chunk size to 768, and a CLI argument `--chunk-size 1024`, when the pipeline resolves configuration, then chunk size is 1024. Given only the config file (no CLI override), then chunk size is 768. Given no config file and no CLI override, then chunk size is 512.

> **FR-1803** | Priority: MUST
> **Description:** The system MUST support a configuration file format (e.g., JSON) for persistent configuration.
> **Rationale:** Persistent configuration files enable version-controlled, reproducible pipeline settings shared across team members (configuration-driven-behaviour).
> **Acceptance Criteria:** Given a JSON configuration file specifying LLM provider, embedding model, and chunking parameters, when the pipeline is started with `--config path/to/config.json`, then all specified parameters are loaded. Negative: a malformed JSON file produces a clear validation error at startup, not a runtime crash.

### 8.2 Configurable Components

The following components MUST be configurable:

| Category | Configurable Aspects |
|----------|---------------------|
| LLM Provider | Provider (OpenAI, Anthropic, Ollama, etc.), model name, temperature, API key, base URL, max tokens, timeout |
| VLM Provider | Provider, model name, base URL, timeout |
| Embedding Model | Provider (HuggingFace, OpenAI, Cohere, etc.), model name, dimension, query/document prefixes, batch size, normalisation, device |
| Vector Store | URL, collection name, BYOM mode, index parameters, distance metric |
| Structure Detector | Provider, OCR enablement, table/figure extraction, quality check threshold |
| Chunking | Strategy, target/min/max chunk size, overlap, section path prepending, table atomicity, boundary context sentences |
| Quality | Minimum chunk tokens, duplicate similarity threshold, deduplication enablement, boilerplate patterns |
| Refactoring | Maximum iterations, fact-check enablement, completeness-check enablement, confidence threshold |
| Review | Default tier, auto-demotion on re-ingestion, approval requirement for promotion |
| Knowledge Graph | Enablement, provider (vector store cross-refs or graph database), spec value extraction, relationship extraction, max triples per chunk |
| Vocabulary | Dictionary path, auto-detection, prompt injection, max prompt terms |
| Re-ingestion | Strategy (skip unchanged / delete and reinsert), hash algorithm, vector/KG cleanup flags |
| Observability | Tracing enablement, log level |
| Evaluation | Enablement, dataset path, auto-run after batch, metrics list, alert thresholds |

### 8.3 Pipeline-Level Flags

| Flag | Purpose |
|------|---------|
| Skip multimodal | Bypass VLM processing |
| Skip refactoring | Bypass document refactoring |
| Skip cross-references | Bypass cross-reference extraction |
| Skip knowledge graph | Bypass KG extraction and storage |
| Dry run | Execute full pipeline without writing to external stores |

### 8.4 Configuration Validation

> **FR-1840** | Priority: MUST
> **Description:** The system MUST cross-validate configuration at startup before processing any documents.
> **Rationale:** Detecting invalid configuration after processing 50 documents wastes compute and time. Fail-fast at startup prevents wasted work (configuration-driven-behaviour).
> **Acceptance Criteria:** Given a configuration with embedding dimension set to 384 but the model registry declares the configured model outputs 768 dimensions, when the pipeline starts, then a validation error is raised before any document is processed. Negative: no document processing begins if configuration validation fails.

> **FR-1841** | Priority: MUST
> **Description:** The system MUST validate that embedding dimension matches the configured model's known dimension.
> **Rationale:** A dimension mismatch (e.g., configuring 384 dimensions for a model that outputs 768) would produce embeddings that fail at storage or return nonsensical search results (fail-safe-over-fail-fast).
> **Acceptance Criteria:** Given configuration specifying `all-MiniLM-L6-v2` with dimension 768, when the model registry declares this model outputs 384 dimensions, then a validation error is raised: "Configured dimension 768 does not match model dimension 384". Given a correct dimension of 384, then validation passes.

> **FR-1842** | Priority: MUST
> **Description:** The system MUST validate that embedding prefixes match the configured model's requirements.
> **Rationale:** Some embedding models (e.g., E5, BGE) require specific prefixes like "query:" and "passage:" for asymmetric retrieval. Missing prefixes silently degrade search quality (fail-safe-over-fail-fast).
> **Acceptance Criteria:** Given a model that requires prefixes "query:" and "passage:" but configuration specifies no prefixes, when validation runs, then a warning is raised. Given a model that requires no prefixes but configuration specifies them, then a warning is raised.

> **FR-1843** | Priority: MUST
> **Description:** The system MUST validate that target chunk size plus boundary context overhead does not exceed the embedding model's maximum input tokens.
> **Rationale:** If enriched chunks exceed the embedding model's context window (e.g., 512 tokens), they will be silently truncated, losing critical tail content like concluding specifications (context-preservation).
> **Acceptance Criteria:** Given target chunk size of 480 tokens, boundary context of 3 sentences (~50 tokens), and embedding model max input of 512 tokens, when validation runs, then a warning is raised: total estimated input (530) exceeds model limit (512). Given target chunk size of 400 with the same overhead, then validation passes.

> **FR-1844** | Priority: MUST
> **Description:** Contradictory configuration (e.g., KG enabled but KG skipped; demote on re-ingestion but preserve review tier) MUST be detected and reported as errors.
> **Rationale:** Contradictory settings create ambiguous behaviour — does KG run or not? Detecting contradictions at startup eliminates this class of bugs (configuration-driven-behaviour).
> **Acceptance Criteria:** Given configuration with `knowledge_graph.enabled = true` and `skip_knowledge_graph = true`, when validation runs, then an error is raised: "Contradictory configuration: KG enabled but KG skip flag set". Given non-contradictory configuration, then validation passes.

> **FR-1845** | Priority: MUST
> **Description:** The system MUST support a model registry that declares known model configurations (dimensions, max tokens, required prefixes) for automated validation.
> **Rationale:** Without a registry, validation requires manual lookup of each model's specifications. A registry automates this and prevents misconfiguration when switching models (swappability-over-lock-in).
> **Acceptance Criteria:** Given a model registry containing entries for "all-MiniLM-L6-v2" (dimension: 384, max_tokens: 256) and "text-embedding-3-small" (dimension: 1536, max_tokens: 8191), when the configured model is "all-MiniLM-L6-v2", then validation uses dimension 384 and max_tokens 256. Given a model not in the registry, then validation logs a warning and skips model-specific checks.

> **FR-1846** | Priority: MUST
> **Description:** Configuration errors MUST halt pipeline startup. Configuration warnings MUST be logged but not block processing.
> **Rationale:** Errors (dimension mismatch, contradictions) would cause failures or corrupt data downstream — halting is the safe choice. Warnings (suboptimal settings) inform the operator without blocking legitimate runs (fail-safe-over-fail-fast).
> **Acceptance Criteria:** Given a configuration error (e.g., dimension mismatch), when the pipeline starts, then it exits with a non-zero status and an error message. Given a configuration warning (e.g., unknown model not in registry), when the pipeline starts, then it logs the warning and proceeds to process documents.

---

## 9. Interface Requirements

### 9.1 Command-Line Interface (FR-1900)

> **FR-1901** | Priority: MUST
> **Description:** The system MUST provide a CLI for single-file processing.
> **Rationale:** Single-file processing is the fundamental operation — engineers need to ingest individual documents during authoring and review cycles (configuration-driven-behaviour).
> **Acceptance Criteria:** Given a single PDF file `power_spec_7nm.pdf`, when the CLI is invoked with `pipeline ingest power_spec_7nm.pdf`, then the document is processed through the full pipeline and stored. The CLI returns exit code 0 on success. Negative: invoking the CLI without a file path produces a usage error.

> **FR-1902** | Priority: MUST
> **Description:** The system MUST provide a CLI for batch processing (recursive directory scan) with configurable file extension filters.
> **Rationale:** Initial corpus ingestion requires processing hundreds of documents across nested directory structures. Manual single-file invocation is impractical at scale (configuration-driven-behaviour).
> **Acceptance Criteria:** Given a directory `/docs/project_alpha/` containing 50 files (.pdf, .docx, .md, .log), when the CLI is invoked with `pipeline ingest-dir /docs/project_alpha/ --extensions .pdf,.docx,.md`, then all matching files are processed recursively and .log files are excluded. The CLI reports a summary of processed/skipped/failed counts.

> **FR-1903** | Priority: MUST
> **Description:** The CLI MUST support the following options: config file path, domain override, document type override, review tier override, skip flags (multimodal, refactoring, cross-refs, KG), dry run, force re-ingestion, vocabulary path, and log level.
> **Rationale:** CLI options provide per-run overrides that take precedence over configuration file settings, enabling quick experimentation without editing config files (configuration-driven-behaviour).
> **Acceptance Criteria:** Given the CLI invoked with `pipeline ingest spec.pdf --domain analog --tier partially_reviewed --skip-refactoring --dry-run --log-level DEBUG`, then the domain is set to "analog", tier is set to Tier 2, refactoring is skipped, no writes to external stores occur, and log level is DEBUG. Each flag is independent and combinable.

> **FR-1904** | Priority: MUST
> **Description:** Individual file failures in batch mode MUST NOT halt the batch. The system MUST report a summary of successes, failures, skips, and flags.
> **Rationale:** A corrupted PDF in a batch of 200 documents should not prevent the remaining 199 from being processed. The summary enables operators to address failures without re-running the full batch (fail-safe-over-fail-fast).
> **Acceptance Criteria:** Given a batch of 10 documents where document 3 fails (corrupted PDF) and document 7 is skipped (unchanged), when the batch completes, then the summary reports: 8 succeeded, 1 failed (document 3 with error detail), 1 skipped (document 7, reason: unchanged). Exit code is non-zero if any failures occurred.

### 9.2 Programmatic API (FR-1950)

> **FR-1951** | Priority: MUST
> **Description:** The system MUST provide a programmatic API for pipeline configuration, document creation, and pipeline invocation.
> **Rationale:** Downstream systems (e.g., a web dashboard, CI/CD integration, or automated ingestion service) need to invoke the pipeline programmatically without shelling out to CLI commands (swappability-over-lock-in).
> **Acceptance Criteria:** Given a Python script, when using the API to create a PipelineConfig, construct a PipelineDocument from a file path, and invoke the pipeline, then the document is processed identically to a CLI invocation. The API returns a result object with processing status, chunk count, and any errors.

> **FR-1952** | Priority: MUST
> **Description:** The system MUST provide a non-pipeline API for review tier management (promote/demote without re-ingestion).
> **Rationale:** Tier changes are administrative operations that should not require re-processing the document through the pipeline. A dedicated API enables lightweight tier management (controlled-access-over-restriction).
> **Acceptance Criteria:** Given a stored document at Tier 3, when the API call `promote_tier(doc_id, new_tier=PARTIALLY_REVIEWED, reviewer="john.doe")` is invoked, then all chunks for that document are updated to Tier 2 without re-ingestion. The review metadata records the reviewer and timestamp. Negative: attempting to promote to an invalid tier value raises a validation error.

---

## 10. Data Model Requirements

### 10.1 Key Entities

The system MUST define the following key data entities:

| Entity | Description |
|--------|-------------|
| **PipelineDocument** | The shared state object flowing through all processing stages. Accumulates data from each stage. |
| **DocumentMetadata** | Document identity (ID, path), filesystem metadata (authors, dates), domain classification, generated metadata (summary, keywords), processing tracking, and content integrity hash. |
| **StructureAnalysis** | Section tree, figure list, table list, page count, and routing flags. |
| **Chunk** | The atom of the retrieval system — the unit that gets embedded and stored. Carries content, positional context, adjacency links, metadata, quality metrics, and deterministic identity. |
| **KGTriple** | A single subject-predicate-object relationship with typed nodes/edges, provenance (source chunk/document), and confidence score. |
| **CrossReference** | A detected link between documents/sections with reference type, confidence, and optional resolved target. |
| **ReviewMetadata** | Review tier, review status, reviewers, review date, notes, and auto-demotion flag. |
| **AbbreviationEntry** | Abbreviation, expansion, domain, context, related terms, and source (dictionary or auto-detected). |

### 10.2 Enumerations

| Enumeration | Values |
|-------------|--------|
| **DocumentFormat** | PDF, DOCX, HTML, MARKDOWN, PLAIN_TEXT, RST, PPTX, XLSX, UNKNOWN (Phase 2: VISIO, IMAGE, SYSTEMVERILOG) |
| **ProcessingStatus** | PENDING, IN_PROGRESS, COMPLETED, FAILED, SKIPPED |
| **ContentType** | TEXT, TABLE, FIGURE, CODE, EQUATION, LIST, HEADING |
| **ReviewTier** | FULLY_REVIEWED, PARTIALLY_REVIEWED, SELF_REVIEWED |
| **ReviewStatus** | DRAFT, SUBMITTED, IN_REVIEW, APPROVED, REJECTED |
| **KGNodeType** | DOCUMENT, CHUNK, CONCEPT, ENTITY, DOMAIN, PERSON, ABBREVIATION, SPEC_VALUE |
| **KGEdgeType** | CONTAINS, REFERENCES, DEPENDS_ON, MENTIONS, BELONGS_TO, RELATED_TO, ABBREVIATION_OF, SPECIFIES, AUTHORED_BY, SUPERSEDES, NEXT_CHUNK |

### 10.3 Deterministic Identity

> **FR-1030** | Priority: MUST
> **Description:** All identifiers (document, chunk, triple) MUST be deterministic, derived from content via cryptographic hashing.
> **Rationale:** Deterministic IDs are the foundation of idempotent re-ingestion. If the same input always produces the same IDs, the system can detect duplicates and unchanged content without external state (idempotency-by-construction).
> **Acceptance Criteria:** Given the same document file processed twice, when chunk IDs are compared, then they are identical. Given two different documents, when their document IDs are compared, then they are different. Negative: no identifier contains random components (e.g., UUIDs, timestamps).

> **FR-1031** | Priority: MUST
> **Description:** Document IDs MUST be derived from the source file path (stable across re-ingestion).
> **Rationale:** Path-based IDs ensure the same file always maps to the same document ID, enabling the system to detect re-ingestion and perform cleanup of previous versions (idempotency-by-construction).
> **Acceptance Criteria:** Given a file at `/docs/specs/power_spec_7nm.pdf`, when ingested twice (even with different content), then the document ID is identical both times. Given a file moved to `/docs/archive/power_spec_7nm.pdf`, then it receives a different document ID.

> **FR-1032** | Priority: MUST
> **Description:** Chunk IDs MUST be derived from parent document ID, chunk position, and content hash.
> **Rationale:** Including the content hash ensures that chunks with changed content receive new IDs, enabling clean delete-and-reinsert re-ingestion. Including position ensures ordering is captured (idempotency-by-construction).
> **Acceptance Criteria:** Given a document producing 10 chunks, when chunk 5 has content "The core voltage is 0.9V", then its ID is derived from (document_id, 5, SHA256("The core voltage is 0.9V")). Given the same document with chunk 5 content changed to "The core voltage is 0.85V", then chunk 5 receives a different ID.

> **FR-1033** | Priority: MUST
> **Description:** Triple IDs MUST be derived from document ID, subject, predicate, and object.
> **Rationale:** Deterministic triple IDs enable deduplication of knowledge graph relationships across re-ingestion runs and prevent duplicate edges (idempotency-by-construction).
> **Acceptance Criteria:** Given a triple ("7nm_PDK", "SPECIFIES", "core_voltage_0.9V") from document D1, when the same triple is extracted on re-ingestion, then it receives the same triple ID. Given a different triple ("7nm_PDK", "SPECIFIES", "core_voltage_0.85V"), then it receives a different ID.

> **FR-1034** | Priority: MUST
> **Description:** Chunk IDs MUST NOT survive across content changes. When earlier content shifts chunk boundaries, all downstream chunks receive new IDs. This is intentional for the delete-and-reinsert re-ingestion strategy.
> **Rationale:** If a paragraph is inserted at the beginning of a document, all subsequent chunk boundaries shift. New IDs for all affected chunks ensure the delete-and-reinsert strategy cleanly replaces stale data (idempotency-by-construction).
> **Acceptance Criteria:** Given a document with 10 chunks, when a new paragraph is inserted before chunk 3 causing chunks 3-10 to shift, then chunks 3-10 all receive new IDs. The re-ingestion strategy deletes old chunk IDs 3-10 and inserts the new ones. Negative: old chunk IDs for positions 3-10 do not persist in the vector store after re-ingestion.

---

## 11. Storage Schema Requirements

### 11.1 Vector Store Schema

The vector store collection MUST support the following property categories:

| Category | Properties |
|----------|-----------|
| **Chunk content** | Raw content (keyword-indexed), enriched content (embedded), context header (display only) |
| **Chunk metadata** | Chunk index, content type, chunking method, token count, quality score, content hash |
| **Structural context** | Section path, page numbers |
| **Searchable metadata** | Chunk-level keywords (keyword-indexed), entities (keyword-indexed), linked figures, linked tables |
| **Navigation** | Previous/next chunk IDs |
| **Document identity** | Document ID, title, domain, type, source path, source format |
| **Document metadata** | Document-level keywords (filterable, NOT keyword-indexed), summary, content hash, extraction confidence |
| **Review** | Review tier, review status, reviewed by, review date |
| **Operational** | Retrieval feedback score, ingestion timestamp, pipeline version |

### 11.2 Vector Index Requirements

> **FR-1120** | Priority: MUST
> **Description:** The vector index MUST support approximate nearest-neighbour search.
> **Rationale:** Exact nearest-neighbour search does not scale beyond small corpora. Approximate methods (e.g., HNSW) provide sub-linear search time necessary for a target of 1.5M chunks (configuration-driven-behaviour).
> **Acceptance Criteria:** Given a vector store with 100,000 chunks, when a similarity search is executed, then results are returned using an approximate nearest-neighbour algorithm (e.g., HNSW). Negative: exact brute-force search is not used as the default search method.

> **FR-1121** | Priority: MUST
> **Description:** The vector index parameters (construction quality, search quality, connectivity) MUST be configurable.
> **Rationale:** Different deployments have different accuracy/speed trade-offs. A small team prioritises accuracy; a large deployment prioritises speed. Configurable index parameters support both (configuration-driven-behaviour).
> **Acceptance Criteria:** Given configuration specifying HNSW parameters `ef_construction=200`, `ef=100`, `max_connections=32`, when the vector index is created, then these parameters are applied. Given different parameters, then the index reflects the new settings.

> **FR-1122** | Priority: MUST
> **Description:** The system MUST support hybrid search combining vector similarity and BM25 keyword matching.
> **Rationale:** Pure vector search struggles with exact technical identifiers (e.g., "TSMC N7" or "IEEE 1149.1"). BM25 keyword matching complements semantic search by handling exact-match queries that embeddings may not capture (context-preservation).
> **Acceptance Criteria:** Given a query "JEDEC JESD79-4 DDR4 timing", when hybrid search is executed, then results include chunks matched by BM25 on "JESD79-4" even if the embedding similarity is moderate. Given a conceptual query "memory interface signal integrity", then vector similarity dominates the ranking.

> **FR-1123** | Priority: MUST
> **Description:** BM25 indexing MUST be applied to chunk content, chunk-level keywords, and entities. BM25 MUST NOT be applied to document-level keywords (to prevent cross-chunk pollution).
> **Rationale:** Document-level keywords (e.g., "power management") apply to the entire document but may not be relevant to every chunk. BM25-indexing them at the chunk level would cause irrelevant chunks to match keyword queries, reducing precision (context-preservation).
> **Acceptance Criteria:** Given a document about "power management" with 15 chunks, where chunk 7 discusses "clock gating" and chunk 12 discusses "voltage scaling", when a BM25 search for "voltage scaling" is executed, then chunk 12 matches (term in chunk content) but chunk 7 does not match solely because "voltage scaling" is a document-level keyword. Negative: document-level keywords are stored as filterable properties but are not BM25-indexed.

### 11.3 Schema Versioning

> **FR-1130** | Priority: MUST
> **Description:** Additive schema changes (new properties) MUST NOT require re-ingestion. New properties MUST be null on existing objects.
> **Rationale:** Adding a new metadata field (e.g., "compliance_standard") should not force re-processing of the entire corpus. Null defaults allow gradual population (configuration-driven-behaviour).
> **Acceptance Criteria:** Given 10,000 existing chunks in the vector store, when a new property "compliance_standard" is added to the schema, then existing chunks have `compliance_standard = null` and remain searchable. Newly ingested chunks populate the field. No re-ingestion is required.

> **FR-1131** | Priority: MUST
> **Description:** Breaking schema changes (property removal, type change, index configuration change, embedding model change) MUST require creating a new collection and re-ingesting.
> **Rationale:** Changing the embedding model produces vectors in a different semantic space — mixing old and new embeddings in the same collection produces meaningless similarity scores. A clean collection ensures consistency (idempotency-by-construction).
> **Acceptance Criteria:** Given a schema change from embedding model "all-MiniLM-L6-v2" (384 dims) to "text-embedding-3-small" (1536 dims), when the migration is performed, then a new collection is created, all documents are re-ingested with the new model, and the old collection is retained until validation completes. Negative: old and new embeddings are never mixed in the same collection.

> **FR-1132** | Priority: MUST
> **Description:** A pipeline version identifier MUST be stored on every chunk to enable identifying the schema version that produced the data.
> **Rationale:** When investigating retrieval quality issues, knowing which pipeline version produced a chunk enables targeted re-ingestion of affected documents (idempotency-by-construction).
> **Acceptance Criteria:** Given pipeline version "1.2.0", when a document is ingested, then every stored chunk has `pipeline_version = "1.2.0"`. When the pipeline is upgraded to "1.3.0" and new documents are ingested, then new chunks have `pipeline_version = "1.3.0"` while old chunks retain "1.2.0".

> **FR-1133** | Priority: MUST
> **Description:** The system MUST support a migration strategy for breaking changes: create new collection → batch re-ingest → validate → swap active collection → delete old collection.
> **Rationale:** A structured migration strategy prevents data loss during schema transitions and enables rollback if validation fails. Swapping only after validation ensures zero-downtime migration (fail-safe-over-fail-fast).
> **Acceptance Criteria:** Given a breaking schema change, when the migration is executed, then: (1) a new collection is created, (2) all documents are re-ingested into the new collection, (3) validation confirms chunk counts and search quality meet thresholds, (4) the active collection pointer is swapped, (5) the old collection is deleted only after successful swap. Negative: if validation fails at step 3, the old collection remains active and the new collection is discarded.
## 12. Non-Functional Requirements

### 12.1 Performance (NFR-100)

> **NFR-101** | Priority: MUST
> **Description:** Single document processing (10-page PDF, no refactoring) MUST complete in less than 60 seconds.
> **Rationale:** Engineers need timely feedback when ingesting individual documents. A sub-minute target ensures the pipeline is practical for interactive single-document workflows without requiring batch scheduling.
> **Acceptance Criteria:** Given a 10-page PDF document with refactoring disabled, when processed through the full pipeline, then wall-clock time from invocation to completion is < 60 seconds on the minimum deployment environment (20 GB RAM, 4 vCPU).

> **NFR-102** | Priority: MUST
> **Description:** Single document processing (10-page PDF, with refactoring) MUST complete in less than 180 seconds.
> **Rationale:** Refactoring involves multi-pass LLM calls with fact-check and completeness validation loops, which are inherently slower. The 180-second budget accommodates this while remaining practical for interactive use.
> **Acceptance Criteria:** Given a 10-page PDF document with refactoring enabled (max 3 iterations), when processed through the full pipeline, then wall-clock time from invocation to completion is < 180 seconds on the minimum deployment environment.

> **NFR-103** | Priority: MUST
> **Description:** Batch throughput (sequential processing) MUST achieve at least 20 documents per hour.
> **Rationale:** The target corpus is 100,000 documents. At 20 docs/hour, initial ingestion of a 500-document pilot set completes in ~25 hours — a reasonable overnight batch window. Falling below this rate makes initial corpus ingestion impractical.
> **Acceptance Criteria:** Given a batch of 20 mixed-format documents (PDFs, DOCX, Markdown) averaging 10 pages each, when processed sequentially, then all 20 complete within 60 minutes.

> **NFR-104** | Priority: MUST
> **Description:** Embedding generation for a 32-chunk batch on local CPU MUST complete in less than 5 seconds.
> **Rationale:** Embedding is a per-document bottleneck. With an average of 15 chunks per document, a 32-chunk batch covers ~2 documents. Exceeding 5 seconds would dominate the per-document processing budget.
> **Acceptance Criteria:** Given 32 chunks of typical length (300–500 tokens each), when embeddings are generated using the configured local CPU embedding model, then the batch completes in < 5 seconds.

> **NFR-105** | Priority: MUST
> **Description:** Vector store upsert of 50 chunks to localhost MUST complete in less than 2 seconds.
> **Rationale:** Storage should not be a bottleneck relative to the compute-intensive stages (embedding, LLM calls). A 2-second cap ensures storage is a minor fraction of per-document time.
> **Acceptance Criteria:** Given 50 chunks with embeddings and full metadata (32+ properties), when upserted to a localhost vector store instance, then the operation completes in < 2 seconds.

> **NFR-106** | Priority: MUST
> **Description:** Re-ingestion cleanup of 100 old chunks MUST complete in less than 3 seconds.
> **Rationale:** Re-ingestion cleanup (deleting previous chunks before inserting new ones) must be fast to keep the re-ingestion path comparable to first-time ingestion. Slow cleanup would discourage document updates (idempotency-by-construction).
> **Acceptance Criteria:** Given a document with 100 previously stored chunks, when re-ingestion cleanup is triggered, then all 100 old chunks and their embeddings are deleted in < 3 seconds.

> **NFR-107** | Priority: MUST
> **Description:** Pipeline startup (graph compilation) MUST complete in less than 1 second.
> **Rationale:** The LangGraph DAG compilation is a cold-start cost paid on every CLI invocation. Exceeding 1 second would make the tool feel sluggish for single-document interactive use.
> **Acceptance Criteria:** Given the pipeline is invoked via CLI, when the DAG is compiled and ready to accept a document, then the startup phase completes in < 1 second (excluding embedding model loading).

> **NFR-108** | Priority: MUST
> **Description:** Embedding model cold start (first-time load) MUST complete within 10–30 seconds (one-time cost).
> **Rationale:** Local embedding models require loading weights into memory on first use. This is an acceptable one-time cost per session but must be bounded to prevent perceived hangs.
> **Acceptance Criteria:** Given the embedding model has not been loaded in the current process, when the first embedding request is made, then the model loads and produces embeddings within 30 seconds. Subsequent requests in the same session incur no load penalty.

> **NFR-109** | Priority: MUST
> **Description:** Memory usage per document MUST remain below 2 GB peak RSS.
> **Rationale:** The minimum deployment environment has 20 GB RAM, shared with the vector store, graph database, and OS. A 2 GB per-document cap ensures the pipeline does not crowd out co-located services.
> **Acceptance Criteria:** Given a 100-page document (the maximum supported size per NFR-202), when processed through the full pipeline, then peak resident set size (RSS) does not exceed 2 GB as measured by process monitoring.

**Performance Targets Summary:**

| ID | Operation | Target |
|----|-----------|--------|
| NFR-101 | Single document (10pp, no refactoring) | < 60 seconds |
| NFR-102 | Single document (10pp, with refactoring) | < 180 seconds |
| NFR-103 | Batch throughput (sequential) | ≥ 20 documents/hour |
| NFR-104 | Embedding generation (32-chunk batch, local CPU) | < 5 seconds |
| NFR-105 | Vector store upsert (50 chunks, localhost) | < 2 seconds |
| NFR-106 | Re-ingestion cleanup (100 old chunks) | < 3 seconds |
| NFR-107 | Pipeline startup (graph compilation) | < 1 second |
| NFR-108 | Embedding model cold start (first-time load) | 10–30 seconds (one-time) |
| NFR-109 | Memory usage per document | < 2 GB peak RSS |

### 12.2 Scalability (NFR-200)

> **NFR-201** | Priority: MUST
> **Description:** The system MUST process documents sequentially by default. Parallel document processing is reserved for future versions.
> **Rationale:** Sequential processing simplifies state management, error handling, and resource accounting for the initial deployment. Parallelism introduces concurrency risks (e.g., KG node conflicts, vector store race conditions) that are deferred to a later phase.
> **Acceptance Criteria:** Given a batch of 10 documents, when processed, then documents are processed one at a time in order. No two documents are in-flight simultaneously. The processing log shows sequential start/end timestamps with no overlap.

> **NFR-202** | Priority: MUST
> **Description:** The system MUST support documents up to ~100 pages. Larger documents SHOULD be split before ingestion.
> **Rationale:** A 100-page limit bounds memory consumption (NFR-109) and processing time. Engineering documents rarely exceed this — those that do (e.g., 500-page combined specs) are better split into logical sub-documents for retrieval quality.
> **Acceptance Criteria:** Given a 100-page PDF, when ingested, then the pipeline processes it successfully within the NFR-109 memory limit. Given a 150-page PDF, when ingested, then the system logs a warning recommending the document be split, but continues processing.

> **NFR-203** | Priority: MUST
> **Description:** The vector store MUST support up to 1,500,000 chunks (target corpus: 100,000 documents at ~15 chunks per document). For larger deployments, sharding or domain-partitioned collections SHOULD be used.
> **Rationale:** The target corpus of 100,000 engineering documents at ~15 chunks each yields ~1.5M chunks. The vector store must handle this scale without degraded search latency. Beyond this, domain-partitioned collections provide both performance and organisational benefits.
> **Acceptance Criteria:** Given a vector store collection containing 1,500,000 chunks with embeddings, when a hybrid search query is executed, then results are returned within acceptable latency (< 500ms). The collection remains stable under continuous upsert/delete operations.

> **NFR-204** | Priority: MUST
> **Description:** The knowledge graph (graph database mode) MUST scale to billions of triples. Vector store cross-reference mode MUST be practical up to ~1M triples.
> **Rationale:** A dedicated graph database (e.g., Neo4j) is designed for graph-scale data. The vector store cross-reference fallback is a simpler alternative with inherent scale limits. Both modes must handle their expected workloads.
> **Acceptance Criteria:** Given a graph database backend, when loaded with 1 billion triples, then traversal queries (e.g., "find all documents referencing specification X") complete within acceptable latency. Given a vector store cross-reference backend with 1M triples, then cross-reference lookups remain responsive.

### 12.3 Reliability (NFR-300)

> **NFR-301** | Priority: MUST
> **Description:** The pipeline crash rate MUST be zero. All errors MUST be caught and logged, not propagated as unhandled exceptions.
> **Rationale:** In a batch processing environment, an unhandled crash in one document aborts the entire batch and potentially leaves the vector store in an inconsistent state. Zero-crash design ensures every document produces either a result or a logged error (fail-safe-over-fail-fast).
> **Acceptance Criteria:** Given a batch of 100 documents including 5 with corrupt content, 3 with unsupported formats, and 2 that trigger LLM timeouts, when processed, then the pipeline completes without any unhandled exceptions. All 10 problem documents have logged errors. The remaining 90 documents are processed successfully.

> **NFR-302** | Priority: MUST
> **Description:** Every LLM-dependent stage MUST have a deterministic fallback (100% coverage).
> **Rationale:** LLM services are inherently unreliable — they can timeout, return malformed responses, or be unavailable entirely. Without fallbacks, LLM outages would halt the entire pipeline (fail-safe-over-fail-fast).
> **Acceptance Criteria:** Given the LLM provider is unreachable, when a document is processed through the full pipeline, then every LLM-dependent stage (chunking, refactoring, metadata, cross-references, multimodal, KG extraction) activates its deterministic fallback. The document completes with lower-quality but usable results. The processing log records each fallback activation.

> **NFR-303** | Priority: MUST
> **Description:** External services (vector store, LLM, embedding model, graph database) MUST be initialised lazily on first use, not at pipeline construction time. Pipeline construction MUST succeed without external services running.
> **Rationale:** Lazy initialisation enables dry-run mode, unit testing, and pipeline configuration validation without requiring all services to be online. It also improves startup time and makes the system more resilient to transient service unavailability.
> **Acceptance Criteria:** Given the vector store and LLM provider are offline, when the pipeline is constructed (DAG compiled), then construction succeeds without errors. When a document is processed in dry-run mode, then no connection attempts are made to external services. When a document is processed in normal mode, then connections are established on first use of each service.

### 12.4 Maintainability (NFR-400)

> **NFR-401** | Priority: MUST
> **Description:** Each processing stage MUST conform to a common abstract interface with uniform error handling, logging, and state wrapping.
> **Rationale:** A uniform interface ensures that any developer can understand, debug, and modify any stage without learning stage-specific patterns. It also enables the pipeline orchestrator to treat all stages generically (swappability-over-lock-in).
> **Acceptance Criteria:** Given the abstract stage interface, when a new processing stage is implemented, then it must implement the standard methods (process, validate input, handle error). Given any existing stage, when inspected, then it conforms to the same interface with no stage-specific error handling patterns outside the interface contract.

> **NFR-402** | Priority: MUST
> **Description:** Replacing a processing stage MUST require implementing the interface and registering the new component; no other code changes.
> **Rationale:** The swappability principle requires that replacing a stage (e.g., swapping the chunking algorithm) is a localised change. If replacing a stage requires modifying orchestration code, routing logic, or other stages, the architecture has failed (swappability-over-lock-in).
> **Acceptance Criteria:** Given a new chunking implementation that conforms to the stage interface, when registered as the active chunking stage, then the pipeline uses it without any changes to other stages, the orchestrator, or the configuration schema (beyond the stage registration).

> **NFR-403** | Priority: MUST
> **Description:** The routing logic MUST derive routing decisions from the processing log (auditable) rather than directly from configuration.
> **Rationale:** When debugging why a stage was skipped or executed, the processing log provides an auditable trail. If routing reads configuration directly, the decision rationale is implicit and harder to trace (fail-safe-over-fail-fast).
> **Acceptance Criteria:** Given a document processed with cross-reference extraction skipped via configuration, when the processing log is inspected, then it contains an explicit entry such as "cross-reference extraction: SKIPPED (reason: disabled in configuration)". The routing decision is traceable to a log entry, not inferred from configuration state.

### 12.5 Security & Compliance (SC-100)

> **SC-101** | Priority: MUST
> **Description:** All pipeline operations (document ingestion, re-ingestion, deletion, tier changes) MUST produce timestamped audit trail entries suitable for compliance review.
> **Rationale:** Engineering organisations operating under ISO 9001 or similar quality frameworks require traceability of document processing actions. Audit trails enable compliance review and incident investigation.
> **Acceptance Criteria:** Given a document is ingested, when the audit log is inspected, then it contains a timestamped entry with: operation type (ingestion), document ID, source path, user/service identity, and outcome (success/failure). Given a tier change, then a separate audit entry records the old tier, new tier, and reason.

> **SC-102** | Priority: MUST
> **Description:** All document processing and data storage MUST occur within the configured deployment boundary (VPC, on-premise server, or local machine). No data MUST leave the deployment boundary unless the LLM or embedding provider is explicitly configured as an external API.
> **Rationale:** Engineering documents may contain proprietary design information (e.g., process node details, circuit architectures). Data sovereignty within the deployment boundary is a baseline security requirement.
> **Acceptance Criteria:** Given a deployment configured with local LLM and local embedding model, when a document is processed, then network monitoring confirms zero outbound data transfers beyond the deployment boundary. Given a deployment configured with an external LLM API, when a document is processed, then only LLM requests are sent externally, and the audit log records which documents were sent to which endpoint (per SC-103).

> **SC-103** | Priority: MUST
> **Description:** The system MUST NOT transmit document content to any service not explicitly listed in the pipeline configuration. When using external LLM APIs, the system MUST log which documents were sent to which external endpoint.
> **Rationale:** Prevents accidental data exfiltration via misconfigured or undeclared services. Logging external transmissions enables security audit and data lineage tracking.
> **Acceptance Criteria:** Given the pipeline configuration lists only "Ollama at localhost:11434" as the LLM provider, when a stage attempts to call a different endpoint (e.g., api.openai.com), then the request is blocked or rejected. Given an external LLM API is configured, when a document is processed, then the audit log contains entries listing each external API call with the document ID and endpoint URL.

> **SC-104** | Priority: MUST
> **Description:** Stored chunks, embeddings, and KG triples MUST support configurable retention policies. The system MUST support expiry-based cleanup of data older than a configurable retention period.
> **Rationale:** Engineering documents may have lifecycle constraints (e.g., project-specific specs that become irrelevant after tapeout). Retention policies prevent unbounded data accumulation and support data governance requirements.
> **Acceptance Criteria:** Given a retention policy of 365 days is configured, when a cleanup job runs, then all chunks, embeddings, and KG triples with an ingestion timestamp older than 365 days are identified for deletion. Given a document re-ingested within the retention period, then its timestamp is refreshed and it is not flagged for cleanup.

> **SC-105** | Priority: MUST
> **Description:** API access to the vector store and graph database MUST use configurable authentication credentials. Credentials MUST NOT be stored in plain text in configuration files; environment variable or secrets manager references MUST be supported.
> **Rationale:** Plain-text credentials in configuration files are a common security vulnerability, especially when configuration files are committed to version control. Environment variables and secrets managers are standard secure credential management approaches.
> **Acceptance Criteria:** Given a configuration file referencing a vector store API key as `${WEAVIATE_API_KEY}`, when the pipeline starts, then the key is resolved from the environment variable. Given a configuration file containing a plain-text API key (e.g., `api_key: "sk-abc123"`), when validated, then a warning is logged recommending environment variable or secrets manager usage.

> **SC-106** | Priority: MUST
> **Description:** The pipeline MUST NOT index or store personally identifiable information (PII) beyond what exists in source documents. No PII MUST be generated or inferred by pipeline processing stages.
> **Rationale:** The pipeline processes engineering documents, not personnel records. Any PII present in source documents (e.g., author names) passes through, but the pipeline must not synthesise new PII (e.g., inferring employee IDs from naming patterns).
> **Acceptance Criteria:** Given a document containing an author name "John Smith" in its header, when processed, then the author name is preserved as-is in metadata. Given a document with no PII, when metadata generation runs, then no PII is generated (e.g., no inferred author identities, no email addresses synthesised from name patterns).

### 12.6 Deployment (NFR-600)

> **NFR-601** | Priority: MUST
> **Description:** The minimum deployment environment MUST be: 20 GB RAM, 50 GB storage, 4 vCPU.
> **Rationale:** This specification sets the baseline hardware requirement to ensure consistent performance targets (NFR-100 series) are achievable. The 20 GB RAM accommodates the embedding model (~4 GB), vector store, and pipeline processing concurrently.
> **Acceptance Criteria:** Given a server with exactly 20 GB RAM, 50 GB storage, and 4 vCPU, when the full pipeline (with local embedding model and local vector store) is deployed and a 10-page PDF is processed, then all performance targets (NFR-101 through NFR-109) are met.

> **NFR-602** | Priority: MUST
> **Description:** GPU access MUST be optional. GPU is required only for local embedding model inference at scale. CPU-only mode MUST be fully supported for all pipeline operations.
> **Rationale:** Many engineering servers lack GPUs. The pipeline must be deployable on commodity hardware. GPU acceleration is a performance optimisation, not a functional requirement (configuration-driven-behaviour).
> **Acceptance Criteria:** Given a server with no GPU, when the pipeline is deployed and configured with a CPU-compatible embedding model, then all pipeline operations complete successfully. Embedding generation uses CPU inference. No errors or warnings related to missing GPU hardware appear.

> **NFR-603** | Priority: MUST
> **Description:** The system MUST support containerised deployment (Docker). A Dockerfile and docker-compose configuration MUST be provided.
> **Rationale:** Containerisation ensures reproducible deployment environments and simplifies dependency management. Docker Compose enables single-command deployment of the pipeline with its co-located services (vector store, graph database).
> **Acceptance Criteria:** Given a clean Docker host, when `docker-compose up` is executed, then the pipeline, vector store, and graph database services start successfully. When a document is ingested via CLI within the container, then it is processed and stored correctly.

> **NFR-604** | Priority: MUST
> **Description:** The system MUST support deployment within an AWS VPC with no public internet access when configured with local LLM, embedding, and VLM providers.
> **Rationale:** Engineering organisations often operate in air-gapped or VPC-isolated environments for IP protection. The pipeline must function without internet when all providers are local (fail-safe-over-fail-fast).
> **Acceptance Criteria:** Given a VPC with no internet gateway and all providers configured as local (Ollama LLM, local embedding model, local VLM), when a document is processed, then no outbound internet requests are attempted and the document is processed successfully.

> **NFR-605** | Priority: MUST
> **Description:** The system MUST support local/on-premise deployment on Linux servers (the same infrastructure used for existing batch job submission).
> **Rationale:** Engineering teams already have Linux batch job infrastructure. Deploying on existing servers avoids provisioning new hardware and leverages familiar operational workflows.
> **Acceptance Criteria:** Given a Linux server (e.g., RHEL 8/9 or Ubuntu 20.04+) with the minimum hardware requirements (NFR-601), when the pipeline is installed via standard Python packaging, then all pipeline operations function correctly without containerisation.

---

## 13. System-Level Acceptance Criteria

| Criterion | Threshold | Related Requirements |
|-----------|-----------|----------------------|
| Chunks produced per 10-page document | 8–30 (depending on density) | FR-601, FR-602 |
| Quality score distribution | > 80% of chunks score ≥ 0.5 | FR-1103, FR-1104 |
| Near-duplicate detection | 0 duplicate chunks in vector store after re-ingestion | FR-1102 |
| Re-ingestion cleanup completeness | 0 orphaned chunks from previous version | FR-1404, FR-1405 |
| Cross-reference detection | ≥ 90% of explicit references ("see Section X") detected | FR-901 |
| Abbreviation resolution | ≥ 95% of dictionary abbreviations correctly expanded | FR-1601, FR-1605 |
| LLM fallback coverage | 100% of LLM stages have deterministic fallback | FR-1702 |
| Pipeline crash rate | 0 unhandled crashes | FR-1701, NFR-301 |

---

## 14. Evaluation Framework Requirements (FR-2000)

> **FR-2001** | Priority: MUST
> **Description:** The system MUST include an evaluation framework that measures end-to-end retrieval quality against a ground-truth dataset.
> **Rationale:** Without objective measurement, there is no way to determine whether pipeline changes (e.g., new chunking strategy, different embedding model) improve or degrade retrieval quality. Evaluation closes the feedback loop between pipeline engineering and retrieval outcomes.
> **Acceptance Criteria:** Given a ground-truth dataset with queries and expected chunks, when the evaluation framework is executed, then it retrieves chunks for each query and compares results against ground truth, producing metric scores (Recall, Precision, MRR).

> **FR-2002** | Priority: MUST
> **Description:** The evaluation dataset MUST contain queries with associated ground-truth chunks, relevance levels (primary/supporting), and query intent classification.
> **Rationale:** Relevance levels distinguish "the exact answer chunk" from "a helpful context chunk." Intent classification (lookup vs. how-to vs. troubleshooting) ensures evaluation covers the diversity of real engineering queries.
> **Acceptance Criteria:** Given the evaluation dataset, when inspected, then each query entry contains: query text, a list of ground-truth chunk IDs with relevance level (primary or supporting), and an intent classification (e.g., specification_lookup, procedural_howto, conceptual_explanation, troubleshooting, comparison).

> **FR-2003** | Priority: MUST
> **Description:** The evaluation dataset MUST be built collaboratively with domain experts, covering multiple domains and query intents (specification lookup, procedural how-to, conceptual explanation, troubleshooting, comparison).
> **Rationale:** An evaluation dataset built without domain expert input would not reflect real engineering queries. Coverage across multiple domains and intents prevents optimisation for one query type at the expense of others.
> **Acceptance Criteria:** Given the evaluation dataset, when analysed, then it contains queries from at least 3 engineering domains (e.g., front-end design, DFT, physical design) and at least 4 of the 5 intent types (specification lookup, procedural how-to, conceptual explanation, troubleshooting, comparison).

> **FR-2004** | Priority: MUST
> **Description:** The minimum viable evaluation dataset MUST contain 50 queries with an average of 3 ground-truth chunks per query.
> **Rationale:** 50 queries with ~3 ground-truth chunks each provides sufficient statistical power to detect meaningful differences between pipeline configurations. Fewer queries risk noisy metrics that cannot distinguish real improvements from variance.
> **Acceptance Criteria:** Given the evaluation dataset, when counted, then it contains at least 50 queries. The total number of ground-truth chunk associations divided by the number of queries is at least 3.0.

> **FR-2005** | Priority: MUST
> **Description:** The system MUST compute the following metrics: Recall@5 (target ≥ 0.75), Recall@10 (target ≥ 0.85), Precision@10 (target ≥ 0.50), MRR (target ≥ 0.60), Abbreviation Hit Rate (target ≥ 0.95).
> **Rationale:** These metrics cover complementary aspects of retrieval quality — Recall measures completeness, Precision measures noise, MRR measures ranking quality, and Abbreviation Hit Rate measures domain-specific term handling. Targets are calibrated for engineering documentation retrieval.
> **Acceptance Criteria:** Given the evaluation framework is run against the ground-truth dataset, when results are produced, then all five metrics are computed and reported. For a passing evaluation: Recall@5 ≥ 0.75, Recall@10 ≥ 0.85, Precision@10 ≥ 0.50, MRR ≥ 0.60, Abbreviation Hit Rate ≥ 0.95.

> **FR-2006** | Priority: MUST
> **Description:** The system MUST support measuring BM25 enrichment impact in isolation (keyword-only search mode) to validate that keyword enrichment is net-positive.
> **Rationale:** BM25 keyword enrichment adds complexity and storage overhead. If keyword search does not improve retrieval over vector-only search, the enrichment is wasted effort. Isolated measurement validates the investment.
> **Acceptance Criteria:** Given the evaluation framework, when run in keyword-only mode (BM25 without vector similarity), then metrics are computed. When compared against vector-only mode, then the impact of BM25 enrichment is quantified (positive or negative delta for each metric).

> **FR-2007** | Priority: MUST
> **Description:** The evaluation framework MUST support A/B comparison of pipeline configurations (e.g., different embedding models, chunk sizes, enrichment strategies).
> **Rationale:** Iterative pipeline improvement requires comparing configurations objectively. Without A/B comparison, decisions about embedding models or chunk sizes would be based on intuition rather than measured retrieval quality.
> **Acceptance Criteria:** Given two pipeline configurations (A: 512-token chunks with BGE-large, B: 256-token chunks with BGE-M3), when both are evaluated against the same ground-truth dataset, then the framework produces a side-by-side comparison report showing metric deltas (e.g., Recall@10: A=0.82, B=0.87, delta=+0.05).

> **FR-2008** | Priority: MUST
> **Description:** The evaluation runner MUST be invocable via CLI and optionally triggered automatically after batch ingestion.
> **Rationale:** CLI invocation enables manual evaluation runs during development. Automatic post-batch evaluation catches regressions immediately when new documents are ingested (configuration-driven-behaviour).
> **Acceptance Criteria:** Given the CLI, when `pipeline evaluate --dataset eval.json` is executed, then the evaluation runs and reports results. Given the configuration `evaluation.auto_run_after_batch: true`, when a batch ingestion completes, then the evaluation framework runs automatically and results are logged.

---

## 15. Feedback & Continuous Improvement Requirements (FR-2100)

> **FR-2101** | Priority: MUST
> **Description:** The system MUST store retrieval feedback scores (per-chunk user ratings) as a mutable property on stored chunk objects in the vector store.
> **Rationale:** Feedback scores enable retrieval-time quality weighting — boosting chunks that users find helpful and penalising those that are not. Storing on the chunk object avoids a separate feedback store and keeps the signal co-located with the data it describes.
> **Acceptance Criteria:** Given a chunk stored in the vector store, when a feedback score (e.g., 4 out of 5) is recorded, then the chunk's `retrieval_feedback_score` property is updated in-place. When the chunk is subsequently retrieved, the feedback score is available for weighting.

> **FR-2102** | Priority: MUST
> **Description:** The system MUST provide a feedback ingestion API for recording user ratings (e.g., thumbs up/down, 1–5 scale) on retrieved chunks, linking each rating to the chunk ID and query context.
> **Rationale:** Structured feedback collection (chunk ID + query context + rating) enables analysis of which chunks perform well for which query types, supporting targeted pipeline improvements.
> **Acceptance Criteria:** Given the feedback API, when a rating of "thumbs down" is submitted for chunk ID "chunk-abc123" with query context "What is the clock frequency for the 7nm block?", then the rating is stored with the chunk ID and query text. When feedback records are queried, then the entry is retrievable by chunk ID or query text.

> **FR-2103** | Priority: MUST
> **Description:** Feedback scores MUST be available as a retrieval-time weighting signal, allowing the retrieval layer to boost or penalise chunks based on accumulated user feedback.
> **Rationale:** User feedback directly reflects retrieval quality from the consumer's perspective. Incorporating it as a weighting signal creates a self-improving retrieval system where frequently-helpful chunks are surfaced more prominently.
> **Acceptance Criteria:** Given two chunks with identical vector similarity scores but different feedback scores (chunk A: 4.5, chunk B: 1.2), when the retrieval layer applies feedback weighting, then chunk A is ranked higher than chunk B in the final results.

> **FR-2104** | Priority: MUST
> **Description:** The system MUST support periodic feedback analysis to identify consistently low-rated chunks or documents, flagging them as candidates for re-processing, review tier demotion, or manual review.
> **Rationale:** Chunks that consistently receive poor feedback may indicate extraction errors, stale content, or poor chunking decisions. Periodic analysis surfaces these systematically rather than relying on manual inspection.
> **Acceptance Criteria:** Given a feedback analysis job is run, when chunks with an average feedback score below a configurable threshold (e.g., < 2.0 over 10+ ratings) are identified, then they are flagged in a report listing chunk ID, document ID, average score, and rating count. The report recommends actions: re-process, demote review tier, or flag for manual review.

---

## 16. External Dependencies

### 16.1 Required Services

| Service | Purpose |
|---------|---------|
| Vector database (e.g., Weaviate) | Vector storage, approximate nearest-neighbour search, hybrid search, metadata filtering |
| LLM provider (e.g., OpenAI, Anthropic, Ollama) | Semantic chunking, refactoring, metadata generation, cross-reference extraction, KG extraction |

### 16.2 Optional Services

| Service | Purpose |
|---------|---------|
| VLM provider (e.g., Ollama/LLaVA) | Figure-to-text conversion |
| Graph database (e.g., Neo4j) | Dedicated knowledge graph storage (alternative to vector store cross-references) |
| Observability platform (e.g., Langfuse) | Pipeline tracing and monitoring |

### 16.3 Downstream Dependencies (Outside This System)

| Service | Purpose | Interface Contract |
|---------|---------|-------------------|
| Reranker model (e.g., BGE-Reranker-v2-m3) | Re-scores retrieved chunks for relevance before answer generation | Consumes chunk content + query text; the embedding pipeline SHALL produce chunks with sufficient standalone context for effective reranking |
| Answer generation LLM | Generates answers from retrieved context | Consumes ranked chunks with metadata; the pipeline SHALL store both raw content (for display) and enriched content (for embedding) to support flexible downstream formatting |

### 16.4 Deployment Constraints

> **NFR-501** | Priority: MUST
> **Description:** The system MUST support offline/air-gapped deployment using local models (local embedding model, local LLM via Ollama).
> **Rationale:** Engineering environments handling proprietary ASIC designs often operate in air-gapped networks. The pipeline must be fully functional without any internet connectivity when configured with local providers (fail-safe-over-fail-fast).
> **Acceptance Criteria:** Given a server with no network connectivity beyond localhost, when the pipeline is configured with a local embedding model and Ollama LLM, then a document is processed end-to-end with no network errors. No DNS lookups or outbound connection attempts are made.

> **NFR-502** | Priority: MUST
> **Description:** The system MUST NOT require outbound internet connectivity when configured with local providers.
> **Rationale:** This is the contrapositive of NFR-501 — ensuring that no hidden dependency (e.g., telemetry, model update checks, license validation) forces internet access when the deployment is configured for local operation.
> **Acceptance Criteria:** Given a deployment with all providers configured as local and a firewall blocking all outbound traffic, when a batch of 10 documents is processed, then all 10 complete successfully. Firewall logs show zero blocked outbound connection attempts from the pipeline process.

---

## Appendix A. Glossary

| Term | Definition |
|------|-----------|
| BM25 | Best Matching 25 — a probabilistic ranking function for keyword-based text search |
| BYOM | Bring Your Own Model — mode where embeddings are computed externally and passed as pre-computed vectors to the vector store |
| Chunk | The atomic unit of the retrieval system; a segment of document text that is individually embedded and stored |
| DAG | Directed Acyclic Graph — the processing pipeline topology |
| Deterministic ID | An identifier derived from content via cryptographic hashing, ensuring the same input always produces the same ID |
| HNSW | Hierarchical Navigable Small World — an approximate nearest-neighbour graph index algorithm |
| Hybrid Search | Combined vector similarity search and BM25 keyword search |
| Idempotent | An operation that produces the same result whether applied once or multiple times |
| Knowledge Graph | A graph of entities and relationships extracted from documents |
| LangGraph | A graph-based orchestration framework (part of the LangChain ecosystem) used to define the processing pipeline DAG |
| PII | Personally Identifiable Information — data that could identify a specific individual |
| RAG | Retrieval-Augmented Generation — a pattern where retrieved context is provided to an LLM for answer generation |
| Reranker | A cross-encoder model that re-scores retrieved chunks for relevance given a specific query, improving precision over embedding-only retrieval |
| Re-ingestion | Processing a previously ingested document again, cleaning up old data and inserting new data |
| Review Tier | A trust classification (Fully/Partially/Self Reviewed) controlling a document's visibility in search results |
| Triple | A subject-predicate-object relationship in the knowledge graph |
| VLM | Vision-Language Model — a multimodal model that can process both images and text |

---

## Appendix B. Document References

| Document | Purpose |
|----------|---------|
| RAG_embedding_pipeline_arch.md | Detailed design and architecture — implementation-level specification including code snippets, algorithms, data structures, and library-specific details |
| RAG_embedding_pipeline_spec_summary.md | Specification summary — requirements overview, scope, security/deployment, phasing, and open questions for stakeholders and team leads |
| RAG_embedding_pipeline_summary.md | Architecture summary — wireframe diagrams, code abstractions, implementation discussion points, and risk register for developers |
| Strategic Proposal: AI-Enabled Knowledge Management Platform | Business case, adoption strategy, infrastructure requirements, and phased rollout plan. This spec is a sub-component of the platform described in the proposal. |

---

## Appendix C. Implementation Phasing

This section maps specification requirements to the implementation phases defined in the Strategic Proposal. Requirements not listed are included in Phase 1 by default.

### Phase 1 — Pilot (Year 1, H1)

**Objective:** Functional RAG pipeline on a document subset with baseline evaluation.

| Scope | Requirements |
|-------|-------------|
| Core pipeline stages | FR-100 (Ingestion), FR-200 (Structure), FR-400 (Cleaning), FR-600 (Chunking), FR-700 (Enrichment), FR-800 (Metadata), FR-1100 (Quality), FR-1200 (Embedding & Storage) |
| CLI interface | FR-1900 |
| Configuration system | FR-1800 (all) |
| Re-ingestion | FR-1400 (all) |
| Review tiers | FR-1500 (all) |
| Error handling & fallbacks | FR-1700 (all) |
| Basic evaluation | FR-2001–FR-2005 (core metrics on 50-query dataset) |
| Local deployment | NFR-501, NFR-502, NFR-601–NFR-605 |
| Data model & schema | FR-1030–FR-1034, FR-1120–FR-1133 |

**Success criteria:** RAG system operational on document subset; retrieval accuracy > 80% on pilot golden question set.

### Phase 2 — Core Development (Year 1, H2)

**Objective:** Full pipeline with advanced features, end-to-end automated evaluation.

| Scope | Requirements |
|-------|-------------|
| Document refactoring | FR-500 (all) |
| Multimodal processing | FR-300 (all) |
| Cross-reference extraction | FR-900 (all) |
| Knowledge graph extraction & storage | FR-1000 (all), FR-1300 (all) |
| Domain vocabulary | FR-1600 (all) |
| A/B evaluation | FR-2006–FR-2008 |
| Security foundations | SC-101–SC-103 |

**Success criteria:** End-to-end workflow operational; automated evaluation against human-validated baseline.

### Phase 3 — Production Deployment (Year 2, H1)

**Objective:** User-facing deployment with UI, feedback collection, and expanded document coverage.

| Scope | Requirements |
|-------|-------------|
| Feedback collection & analysis | FR-2100 (all) |
| Programmatic API | FR-1950 (all) |
| SharePoint integration | FR-113 |
| Observability (Langfuse) | Observability configuration (FR-1800) |
| Full security & compliance | SC-104–SC-106 |
| Web dashboard UI | Out of scope for this spec — see Strategic Proposal |

**Success criteria:** 80–90% of target documents indexed; feedback collection operational; GUI deployed.

### Phase 4 — Optimisation (Year 2, H2+)

**Objective:** Production maturity, parameter optimisation, self-service, LLM migration.

| Scope | Requirements |
|-------|-------------|
| AI-assisted parameter tuning | Feedback analysis (FR-2104) driving configuration adjustments |
| Self-service document embedding | Simplified ingestion workflow (FR-1901 + FR-1951) |
| LLM migration (cloud → self-hosted) | Swappability requirements (FR-1209, FR-306, FR-208) |
| 95% accuracy target | FR-2005 metric targets at full corpus scale |

**Success criteria:** 95% retrieval accuracy; self-service embedding functional; daily automated testing operational.

---

## Appendix D. Open Questions

The following questions SHALL be resolved before finalising this specification:

1. **Deployment model:** Containerised (Docker) or bare-metal? Single-node or distributed? *(Partially addressed by NFR-603–NFR-605; final decision needed.)*
2. **Authentication:** How do users authenticate to the vector store and graph database? How are autonomous agent service accounts provisioned? What command/operation guardrails need to be established for agent access? *(See Strategic Proposal — IT Security & Access Control questions.)*
3. **Monitoring:** What operational monitoring and alerting is required beyond the processing log? What Langfuse dashboards are needed for production?
4. **Document deletion:** Should the system support explicit document deletion (remove all chunks and KG triples for a document)?
5. **Multi-tenancy:** Should different teams/projects have separate collections or share a single collection with metadata-based isolation? *(The Strategic Proposal implies domain-level separation — Front-end, DFT, Physical Design, Verification — confirm if this maps to separate collections or metadata filters.)*
6. **Backup and recovery:** What is the backup strategy for the vector store and knowledge graph?
7. **Concurrent re-ingestion:** Should the system support concurrent re-ingestion of different documents, or is sequential processing sufficient?
8. **Retention policy:** Should documents/chunks have a configurable retention period or expiry? *(SC-104 establishes the requirement; exact retention periods need definition.)*
9. **LLM migration strategy:** The Strategic Proposal outlines a migration path from Claude API (Months 1–6) → Llama 70B via PrivateLink (Month 6+) → Llama 405B (Year 2+). What triggers the migration decision? What acceptance criteria must the self-hosted model meet before replacing the cloud API?
10. **Embedding model selection:** The Strategic Proposal specifies BGE-M3 (multi-lingual, multi-granularity); the architecture document uses BGE-large (1024d) as the reference model. Confirm the target embedding model for Phase 1 deployment.
11. **Business KPI mapping:** The Strategic Proposal targets "80% retrieval accuracy on pilot golden question set" and "30%+ productivity improvement." How do these map to the technical metrics in FR-2005 (Recall@5 ≥ 0.75, Recall@10 ≥ 0.85, MRR ≥ 0.60)? Define the translation between technical evaluation metrics and business-reported accuracy figures.

---

## Requirements Traceability Matrix

| REQ ID | Section | Priority | Component/Stage |
|--------|---------|----------|-----------------|
| FR-101 | 3.1 | MUST | Document Ingestion |
| FR-102 | 3.1 | MUST | Document Ingestion |
| FR-103 | 3.1 | MUST | Document Ingestion |
| FR-104 | 3.1 | MUST | Document Ingestion |
| FR-105 | 3.1 | MUST | Document Ingestion |
| FR-106 | 3.1 | MUST | Document Ingestion |
| FR-107 | 3.1 | MUST | Document Ingestion |
| FR-108 | 3.1 | MUST | Document Ingestion |
| FR-109 | 3.1 | MUST | Document Ingestion |
| FR-110 | 3.1 | SHOULD | Document Ingestion |
| FR-111 | 3.1 | MUST | Document Ingestion |
| FR-112 | 3.1 | MUST | Document Ingestion |
| FR-113 | 3.1 | SHOULD | Document Ingestion |
| FR-201 | 3.2 | MUST | Structure Detection |
| FR-202 | 3.2 | MUST | Structure Detection |
| FR-203 | 3.2 | MUST | Structure Detection |
| FR-204 | 3.2 | MUST | Structure Detection |
| FR-205 | 3.2 | MUST | Structure Detection |
| FR-206 | 3.2 | MUST | Structure Detection |
| FR-207 | 3.2 | MUST | Structure Detection |
| FR-208 | 3.2 | MUST | Structure Detection |
| FR-301 | 3.3 | MUST | Multimodal Processing |
| FR-302 | 3.3 | MUST | Multimodal Processing |
| FR-303 | 3.3 | MUST | Multimodal Processing |
| FR-304 | 3.3 | MUST | Multimodal Processing |
| FR-305 | 3.3 | MUST | Multimodal Processing |
| FR-306 | 3.3 | MUST | Multimodal Processing |
| FR-307 | 3.3 | MUST | Multimodal Processing |
| FR-401 | 3.4 | MUST | Text Cleaning |
| FR-402 | 3.4 | MUST | Text Cleaning |
| FR-403 | 3.4 | MUST | Text Cleaning |
| FR-404 | 3.4 | MUST | Text Cleaning |
| FR-405 | 3.4 | MUST | Text Cleaning |
| FR-501 | 3.5 | MUST | Document Refactoring |
| FR-502 | 3.5 | MUST | Document Refactoring |
| FR-503 | 3.5 | MUST | Document Refactoring |
| FR-504 | 3.5 | MUST | Document Refactoring |
| FR-505 | 3.5 | MUST | Document Refactoring |
| FR-506 | 3.5 | MUST | Document Refactoring |
| FR-507 | 3.5 | MUST | Document Refactoring |
| FR-508 | 3.5 | MUST | Document Refactoring |
| FR-601 | 3.6 | MUST | Chunking |
| FR-602 | 3.6 | MUST | Chunking |
| FR-603 | 3.6 | MUST | Chunking |
| FR-604 | 3.6 | MUST | Chunking |
| FR-605 | 3.6 | MUST | Chunking |
| FR-606 | 3.6 | MUST | Chunking |
| FR-607 | 3.6 | MUST | Chunking |
| FR-608 | 3.6 | MUST | Chunking |
| FR-609 | 3.6 | MUST | Chunking |
| FR-610 | 3.6 | MUST | Chunking |
| FR-611 | 3.6 | MUST | Chunking |
| FR-701 | 3.7 | MUST | Chunk Enrichment |
| FR-702 | 3.7 | MUST | Chunk Enrichment |
| FR-703 | 3.7 | MUST | Chunk Enrichment |
| FR-704 | 3.7 | MUST | Chunk Enrichment |
| FR-705 | 3.7 | MUST | Chunk Enrichment |
| FR-801 | 3.8 | MUST | Metadata Generation |
| FR-802 | 3.8 | MUST | Metadata Generation |
| FR-803 | 3.8 | MUST | Metadata Generation |
| FR-804 | 3.8 | MUST | Metadata Generation |
| FR-805 | 3.8 | MUST | Metadata Generation |
| FR-806 | 3.8 | MUST | Metadata Generation |
| FR-901 | 3.9 | MUST | Cross-Reference Extraction |
| FR-902 | 3.9 | MUST | Cross-Reference Extraction |
| FR-903 | 3.9 | MUST | Cross-Reference Extraction |
| FR-904 | 3.9 | MUST | Cross-Reference Extraction |
| FR-905 | 3.9 | MUST | Cross-Reference Extraction |
| FR-1001 | 3.10 | MUST | Knowledge Graph Extraction |
| FR-1002 | 3.10 | MUST | Knowledge Graph Extraction |
| FR-1003 | 3.10 | MUST | Knowledge Graph Extraction |
| FR-1004 | 3.10 | MUST | Knowledge Graph Extraction |
| FR-1005 | 3.10 | MUST | Knowledge Graph Extraction |
| FR-1006 | 3.10 | MUST | Knowledge Graph Extraction |
| FR-1007 | 3.10 | MUST | Knowledge Graph Extraction |
| FR-1008 | 3.10 | MUST | Knowledge Graph Extraction |
| FR-1009 | 3.10 | MUST | Knowledge Graph Extraction |
| FR-1101 | 3.11 | MUST | Quality Validation |
| FR-1102 | 3.11 | MUST | Quality Validation |
| FR-1103 | 3.11 | MUST | Quality Validation |
| FR-1104 | 3.11 | MUST | Quality Validation |
| FR-1105 | 3.11 | MUST | Quality Validation |
| FR-1201 | 3.12 | MUST | Embedding & Storage |
| FR-1202 | 3.12 | MUST | Embedding & Storage |
| FR-1203 | 3.12 | MUST | Embedding & Storage |
| FR-1204 | 3.12 | MUST | Embedding & Storage |
| FR-1205 | 3.12 | MUST | Embedding & Storage |
| FR-1206 | 3.12 | MUST | Embedding & Storage |
| FR-1207 | 3.12 | MUST | Embedding & Storage |
| FR-1208 | 3.12 | MUST | Embedding & Storage |
| FR-1209 | 3.12 | MUST | Embedding & Storage |
| FR-1301 | 3.13 | MUST | Knowledge Graph Storage |
| FR-1302 | 3.13 | MUST | Knowledge Graph Storage |
| FR-1303 | 3.13 | MUST | Knowledge Graph Storage |
| FR-1304 | 3.13 | MUST | Knowledge Graph Storage |
| FR-1401 | 4 | MUST | Re-ingestion |
| FR-1402 | 4 | MUST | Re-ingestion |
| FR-1403 | 4 | MUST | Re-ingestion |
| FR-1404 | 4 | MUST | Re-ingestion |
| FR-1405 | 4 | MUST | Re-ingestion |
| FR-1406 | 4 | MUST | Re-ingestion |
| FR-1407 | 4 | MUST | Re-ingestion |
| FR-1408 | 4 | MUST | Re-ingestion |
| FR-1409 | 4 | MUST | Re-ingestion |
| FR-1501 | 5.1 | MUST | Review Tiers |
| FR-1502 | 5.1 | MUST | Review Tiers |
| FR-1503 | 5.1 | MUST | Review Tiers |
| FR-1504 | 5.1 | MUST | Review Tiers |
| FR-1510 | 5.2 | MUST | Review Tiers |
| FR-1511 | 5.2 | MUST | Review Tiers |
| FR-1512 | 5.2 | MUST | Review Tiers |
| FR-1513 | 5.2 | MUST | Review Tiers |
| FR-1514 | 5.2 | MUST | Review Tiers |
| FR-1520 | 5.3 | MUST | Review Tiers |
| FR-1521 | 5.3 | MUST | Review Tiers |
| FR-1601 | 6 | MUST | Domain Vocabulary |
| FR-1602 | 6 | MUST | Domain Vocabulary |
| FR-1603 | 6 | MUST | Domain Vocabulary |
| FR-1604 | 6 | MUST | Domain Vocabulary |
| FR-1605 | 6 | MUST | Domain Vocabulary |
| FR-1606 | 6 | MUST | Domain Vocabulary |
| FR-1701 | 7 | MUST | Error Handling |
| FR-1702 | 7 | MUST | Error Handling |
| FR-1703 | 7 | MUST | Error Handling |
| FR-1704 | 7 | MUST | Error Handling |
| FR-1705 | 7 | MUST | Error Handling |
| FR-1706 | 7 | MUST | Error Handling |
| FR-1801 | 8.1 | MUST | Configuration |
| FR-1802 | 8.1 | MUST | Configuration |
| FR-1803 | 8.1 | MUST | Configuration |
| FR-1840 | 8.4 | MUST | Configuration |
| FR-1841 | 8.4 | MUST | Configuration |
| FR-1842 | 8.4 | MUST | Configuration |
| FR-1843 | 8.4 | MUST | Configuration |
| FR-1844 | 8.4 | MUST | Configuration |
| FR-1845 | 8.4 | MUST | Configuration |
| FR-1846 | 8.4 | MUST | Configuration |
| FR-1901 | 9.1 | MUST | Interface |
| FR-1902 | 9.1 | MUST | Interface |
| FR-1903 | 9.1 | MUST | Interface |
| FR-1904 | 9.1 | MUST | Interface |
| FR-1951 | 9.2 | MUST | Interface |
| FR-1952 | 9.2 | MUST | Interface |
| FR-1030 | 10.3 | MUST | Deterministic Identity |
| FR-1031 | 10.3 | MUST | Deterministic Identity |
| FR-1032 | 10.3 | MUST | Deterministic Identity |
| FR-1033 | 10.3 | MUST | Deterministic Identity |
| FR-1034 | 10.3 | MUST | Deterministic Identity |
| FR-1120 | 11.2 | MUST | Vector Index |
| FR-1121 | 11.2 | MUST | Vector Index |
| FR-1122 | 11.2 | MUST | Vector Index |
| FR-1123 | 11.2 | MUST | Vector Index |
| FR-1130 | 11.3 | MUST | Schema Versioning |
| FR-1131 | 11.3 | MUST | Schema Versioning |
| FR-1132 | 11.3 | MUST | Schema Versioning |
| FR-1133 | 11.3 | MUST | Schema Versioning |
| NFR-101 | 12.1 | MUST | Performance |
| NFR-102 | 12.1 | MUST | Performance |
| NFR-103 | 12.1 | MUST | Performance |
| NFR-104 | 12.1 | MUST | Performance |
| NFR-105 | 12.1 | MUST | Performance |
| NFR-106 | 12.1 | MUST | Performance |
| NFR-107 | 12.1 | MUST | Performance |
| NFR-108 | 12.1 | MUST | Performance |
| NFR-109 | 12.1 | MUST | Performance |
| NFR-201 | 12.2 | MUST | Scalability |
| NFR-202 | 12.2 | MUST | Scalability |
| NFR-203 | 12.2 | MUST | Scalability |
| NFR-204 | 12.2 | MUST | Scalability |
| NFR-301 | 12.3 | MUST | Reliability |
| NFR-302 | 12.3 | MUST | Reliability |
| NFR-303 | 12.3 | MUST | Reliability |
| NFR-401 | 12.4 | MUST | Maintainability |
| NFR-402 | 12.4 | MUST | Maintainability |
| NFR-403 | 12.4 | MUST | Maintainability |
| NFR-501 | 16.4 | MUST | Deployment Constraints |
| NFR-502 | 16.4 | MUST | Deployment Constraints |
| NFR-601 | 12.6 | MUST | Deployment |
| NFR-602 | 12.6 | MUST | Deployment |
| NFR-603 | 12.6 | MUST | Deployment |
| NFR-604 | 12.6 | MUST | Deployment |
| NFR-605 | 12.6 | MUST | Deployment |
| SC-101 | 12.5 | MUST | Security & Compliance |
| SC-102 | 12.5 | MUST | Security & Compliance |
| SC-103 | 12.5 | MUST | Security & Compliance |
| SC-104 | 12.5 | MUST | Security & Compliance |
| SC-105 | 12.5 | MUST | Security & Compliance |
| SC-106 | 12.5 | MUST | Security & Compliance |
| FR-2001 | 14 | MUST | Evaluation |
| FR-2002 | 14 | MUST | Evaluation |
| FR-2003 | 14 | MUST | Evaluation |
| FR-2004 | 14 | MUST | Evaluation |
| FR-2005 | 14 | MUST | Evaluation |
| FR-2006 | 14 | MUST | Evaluation |
| FR-2007 | 14 | MUST | Evaluation |
| FR-2008 | 14 | MUST | Evaluation |
| FR-2101 | 15 | MUST | Feedback |
| FR-2102 | 15 | MUST | Feedback |
| FR-2103 | 15 | MUST | Feedback |
| FR-2104 | 15 | MUST | Feedback |

**Total Requirements: 200**
- MUST: 198
- SHOULD: 2
- MAY: 0