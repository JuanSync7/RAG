# Retrieval Pipeline Specification

**AION Knowledge Management Platform**
Version: 1.0 | Status: Draft | Domain: Retrieval Pipeline

---

## 1. Scope & Definitions

### 1.1 Scope

This specification defines the requirements for the **retrieval pipeline** of the AION RAG system. The pipeline boundary is:

- **Entry point:** User submits a natural language query
- **Exit point:** User receives a generated answer with source citations and confidence metadata

Everything between these two points is in scope. Document ingestion, embedding pipeline, and offline evaluation infrastructure are explicitly **out of scope**.

### 1.2 Terminology

| Term | Definition |
|------|-----------|
| **Hybrid Search** | A retrieval strategy combining dense vector similarity search (semantic) with BM25 keyword search (lexical) using a configurable fusion weight |
| **Reranking** | A second-pass scoring step using a cross-encoder model to re-score each (query, document) pair for fine-grained relevance |
| **Guardrail** | A validation checkpoint in the pipeline that inspects inputs or outputs and may reject, redact, or flag content before it proceeds |
| **Confidence Score** | A composite numerical score (0.0–1.0) derived from multiple independent signals indicating the reliability of the generated answer |
| **PII** | Personally Identifiable Information — names, email addresses, phone numbers, employee IDs, physical addresses, and other data that can identify an individual |
| **Risk Level** | A classification (HIGH, MEDIUM, LOW) assigned to a query based on the potential consequences of an incorrect answer |
| **Coreference Resolution** | The process of resolving pronouns and references ("it", "that", "the previous one") against prior conversation context |
| **Knowledge Graph (KG)** | A directed graph of domain entities and their relationships, used optionally to expand queries with related terms |
| **Fusion** | The method by which BM25 and vector search scores are combined into a single ranked list |

### 1.3 Requirement Priority Levels

This document uses RFC 2119 language:

- **MUST** — Absolute requirement. The system is non-conformant without it.
- **SHOULD** — Recommended. May be omitted only with documented justification.
- **MAY** — Optional. Included at the implementor's discretion.

### 1.4 Requirement Format

Each requirement follows this structure:

> **REQ-xxx** | Priority: MUST/SHOULD/MAY
> **Description:** What the system shall do.
> **Rationale:** Why this requirement exists.
> **Acceptance Criteria:** How to verify conformance.

---

## 2. Pipeline Overview

### 2.1 Pipeline Stages

```
User Query (natural language input)
    │
    ▼
┌──────────────────────────────────────┐
│ [1] QUERY PROCESSING                 │
│     Reformulate, score confidence,   │
│     resolve multi-turn references    │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ [2] PRE-RETRIEVAL GUARDRAIL          │
│     Input validation, injection      │
│     detection, risk classification,  │
│     PII filtering (external LLM)     │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ [3] RETRIEVAL                        │
│     Vector search + BM25 hybrid      │
│     Optional KG expansion            │
│     Metadata filtering               │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ [4] RERANKING                        │
│     Cross-encoder rescoring          │
│     Score normalization & thresholds │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ [5] DOCUMENT FORMATTING              │
│     Structured metadata attachment   │
│     Version conflict detection       │
│     Context preparation for LLM      │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ [6] GENERATION                       │
│     Anti-hallucination prompt        │
│     Source citation                  │
│     LLM confidence self-report       │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ [7] POST-GENERATION GUARDRAIL        │
│     3-signal confidence scoring      │
│     Hallucination detection          │
│     PII filtering, output sanitize   │
│     Confidence routing               │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ [8] ANSWER DELIVERY                  │
│     Return answer with sources,      │
│     confidence, risk-based display   │
└──────────────────────────────────────┘
```

### 2.2 Data Flow Summary

| Stage | Input | Output |
|-------|-------|--------|
| Query Processing | Raw user query, conversation history | Processed query, confidence, action (search/ask_user) |
| Pre-Retrieval Guardrail | Processed query, risk taxonomy, PII patterns | Validated query, risk level, or rejection |
| Retrieval | Validated query, query embedding, filters | Ranked candidate documents (top-N) |
| Reranking | Query, candidate documents | Re-scored documents (top-K), relevance scores |
| Document Formatting | Re-scored documents, metadata | Formatted context string, version conflict flags |
| Generation | Formatted context, query, system prompt | Generated answer with citations, LLM confidence |
| Post-Generation Guardrail | Answer, retrieved docs, scores, risk level | Validated answer or re-retrieval trigger or escalation |
| Answer Delivery | Validated answer, sources, confidence, risk | Final response to user |

---

## 3. Query Processing

> **REQ-101** | Priority: MUST
> **Description:** The system MUST reformulate user queries into precise retrieval queries using an LLM. The reformulation MUST add domain context, remove ambiguity, and generate up to 2 alternative phrasings for ambiguous queries.
> **Rationale:** Vague queries produce poor retrieval. A query like "what's the timing thing for USB" must be transformed into "USB controller timing constraints specification" before hitting the vector database.
> **Acceptance Criteria:** Given a vague query, the system produces a primary reformulated query and at least one alternative phrasing. Reformulated queries retrieve more relevant documents than the original query in A/B evaluation.

> **REQ-102** | Priority: MUST
> **Description:** The system MUST score each query for confidence (0.0–1.0) and route based on a configurable threshold. Queries above the threshold proceed to search. Queries below the threshold MUST be returned to the user with a clarification request.
> **Rationale:** Sending low-quality queries to retrieval wastes compute and returns poor results. It is better to ask the user for clarification.
> **Acceptance Criteria:** Queries like "asdf jkl" are routed to ask_user. Queries like "what is the supply voltage for USB power domain" are routed to search. The confidence threshold is configurable without code changes.

> **REQ-103** | Priority: SHOULD
> **Description:** The system SHOULD maintain conversation history (last N turns) and resolve coreferences in follow-up queries. Pronouns and references ("it", "that", "the previous one", "tell me more") SHOULD be resolved against recent conversation context before reformulation.
> **Rationale:** Without multi-turn context, users cannot have natural follow-up conversations. Every query is treated as independent, which limits usability in interactive deployments.
> **Acceptance Criteria:** After answering "What is the USB power domain voltage?", a follow-up query "What about the clock frequency?" is resolved to "What is the USB controller clock frequency?" using conversation context.

> **REQ-104** | Priority: MUST
> **Description:** The system MUST implement an iterative refinement loop for query processing. If the confidence score is below the threshold after reformulation, the system MUST retry reformulation up to a configurable maximum number of iterations before routing to ask_user.
> **Rationale:** A single reformulation attempt may not be sufficient for highly ambiguous queries. Iterative refinement gives the system multiple chances to improve query quality.
> **Acceptance Criteria:** The system attempts up to N reformulations (configurable, default 3) before giving up. Each iteration produces a measurably different query. The loop terminates early if confidence exceeds the threshold.

---

## 4. Pre-Retrieval Guardrail

> **REQ-201** | Priority: MUST
> **Description:** The system MUST validate all query inputs before retrieval. Validation MUST include:
> - Query length within configurable bounds (min and max characters)
> - Search parameters (`alpha`, `search_limit`, `rerank_top_k`) within valid ranges
> - Metadata filter values (`source_filter`, `heading_filter`) sanitized against injection
> **Rationale:** Unvalidated inputs passed directly to the vector database or LLM can cause errors, unexpected behavior, or security vulnerabilities.
> **Acceptance Criteria:** Out-of-range parameters are rejected with descriptive error messages. Filter values containing Weaviate query language injection are sanitized or rejected. Empty queries are rejected.

> **REQ-202** | Priority: MUST
> **Description:** The system MUST detect and reject prompt injection attempts. Injection patterns MUST be defined in an external configuration file (not hardcoded) so they can be updated without code changes.
> **Rationale:** Hardcoded patterns cannot be updated without redeployment. An external config file allows rapid response to new injection techniques.
> **Acceptance Criteria:** Injection patterns are loaded from a YAML/JSON config file at startup. Queries matching any pattern are rejected with a generic message (no information leakage about which pattern matched). New patterns can be added to the config file and take effect on restart.

> **REQ-203** | Priority: MUST
> **Description:** The system MUST classify each query's risk level as HIGH, MEDIUM, or LOW using a deterministic keyword taxonomy. The risk level MUST be computed before generation and attached to the pipeline state for downstream use.
> **Rationale:** Incorrect answers in certain domains (electrical specifications, timing constraints, safety compliance) have real consequences. Risk classification enables proportional verification friction — engineers only encounter verification overhead when the stakes justify it.
> **Acceptance Criteria:** Queries containing terms like "voltage", "timing constraint", "iso26262", "safety" are classified as HIGH. Queries containing "procedure", "guideline", "checklist" are classified as MEDIUM. All others default to LOW. Classification is deterministic and auditable. The keyword taxonomy is externalized to a config file.

> **REQ-204** | Priority: SHOULD
> **Description:** When the retrieval pipeline uses an external (non-local) LLM for query processing or generation, the system SHOULD detect and redact PII from the query before it is sent to the external API. PII types include: person names, email addresses, phone numbers, employee IDs, and physical addresses.
> **Rationale:** Sending PII to external LLM providers constitutes a data leak. This guardrail is conditional — it is not needed when using a local LLM (e.g., Ollama) where data stays on-premise.
> **Acceptance Criteria:** When external LLM mode is enabled, PII entities are detected and replaced with typed placeholders (e.g., `[PERSON]`, `[EMAIL]`) before the query leaves the system. The original query with PII is preserved internally for retrieval but never sent externally. PII detection covers at minimum: email regex, phone regex, and named entity recognition for person names.

> **REQ-205** | Priority: MUST
> **Description:** The system MUST reject queries that fail validation or injection detection. Rejected queries MUST return a structured error response (not an exception) with a user-safe message that does not reveal internal system details.
> **Rationale:** Revealing which specific pattern was matched or which validation rule failed gives attackers information to refine their injection attempts.
> **Acceptance Criteria:** Rejected queries return a response object with action="rejected", a generic user-facing message, and an internal log entry with the specific rejection reason.

---

## 5. Retrieval

> **REQ-301** | Priority: MUST
> **Description:** The system MUST support dense vector similarity search using pre-computed document embeddings. The query MUST be embedded at query time using the same embedding model used for document ingestion.
> **Rationale:** Semantic search captures conceptual similarity — queries like "how do I fix a setup violation" will match documents about timing closure even if the exact words differ.
> **Acceptance Criteria:** Vector search returns documents ranked by cosine similarity (or equivalent via L2-normalized dot product). The embedding model is configurable. Query embedding dimensions match document embedding dimensions.

> **REQ-302** | Priority: MUST
> **Description:** The system MUST support BM25 keyword search over the document corpus. BM25 search MUST operate on the full text content of each document chunk.
> **Rationale:** Semantic search alone misses exact terminology — part numbers, specification IDs, tool names, and acronyms. BM25 captures exact lexical matches that vector search may miss.
> **Acceptance Criteria:** A query for "DO-254 DAL-A compliance checklist" returns documents containing those exact terms, even if semantic similarity is low.

> **REQ-303** | Priority: MUST
> **Description:** The system MUST combine vector search and BM25 results using a hybrid fusion strategy. The fusion weight (alpha) MUST be configurable:
> - alpha = 0.0: pure BM25
> - alpha = 1.0: pure vector search
> - alpha = 0.5: equal weight (default)
>
> The fusion method MUST normalize scores from both sources to a common scale before combining.
> **Rationale:** Neither search modality is sufficient alone. Hybrid search captures both conceptual and lexical relevance.
> **Acceptance Criteria:** Changing alpha measurably shifts results toward keyword or semantic matches. The default alpha (0.5) produces results that include both exact-term and conceptual matches in the top-N.

> **REQ-304** | Priority: MAY
> **Description:** The system MAY expand queries using a knowledge graph before retrieval. When enabled, KG expansion MUST:
> - Match entities in the query against the KG
> - Traverse up to a configurable depth (default 1 hop)
> - Append related terms to the BM25 query (not the vector query)
> - Limit expansion to a configurable number of terms (default 3)
>
> KG expansion MUST be independently toggleable via configuration.
> **Rationale:** KG expansion improves recall for acronyms and related concepts (e.g., expanding "RAG" to "Retrieval-Augmented Generation"). However, over-expansion can introduce noise. This feature is optional.
> **Acceptance Criteria:** When enabled, KG-expanded terms appear in search results that would not have been retrieved without expansion. When disabled, the pipeline operates identically without KG. The feature toggle does not require code changes.

> **REQ-305** | Priority: MUST
> **Description:** The system MUST support metadata filtering at retrieval time. Supported filter dimensions MUST include at minimum: document source (filename), domain, document type, and version. Filters MUST be applied as pre-filtering (before scoring), not post-filtering.
> **Rationale:** Engineers often know which document or domain they want to search within. Pre-filtering reduces noise and improves relevance.
> **Acceptance Criteria:** A query with `source_filter="Power_Spec_v3.pdf"` only returns chunks from that document. Filters can be combined (e.g., domain + version). Filters with no matches return an empty result set (not an error).

> **REQ-306** | Priority: SHOULD
> **Description:** The system SHOULD cache query embeddings using an LRU (Least Recently Used) cache. Repeated or identical queries SHOULD return cached embeddings without recomputation.
> **Rationale:** Embedding computation is the most expensive per-query operation. RAG workloads frequently have repeated queries. Caching eliminates redundant computation.
> **Acceptance Criteria:** A query submitted twice in succession uses the cached embedding on the second call (measurable by latency reduction). The cache has a configurable maximum size. Cache eviction follows LRU policy.

> **REQ-307** | Priority: SHOULD
> **Description:** The system SHOULD maintain a persistent connection pool to the vector database rather than creating a new connection per query. The connection pool SHOULD include health checks on startup and periodic liveness checks.
> **Rationale:** Creating a new database connection per query adds latency and resource overhead. Connection pooling amortizes connection setup cost across queries.
> **Acceptance Criteria:** Multiple concurrent queries reuse connections from the pool. A health check failure on startup prevents the system from accepting queries (fail-fast). Connection failures during operation trigger reconnection without crashing.

> **REQ-308** | Priority: SHOULD
> **Description:** The system SHOULD cache full query results (search + rerank + generation) keyed by `(processed_query, filters)` with a configurable TTL (time-to-live). Cache hits SHOULD bypass all downstream pipeline stages.
> **Rationale:** Most RAG workloads have repeat queries. Caching the full response saves all expensive computation (search, reranking, generation).
> **Acceptance Criteria:** An identical query with identical filters returns a cached response within the TTL window. Responses are evicted after TTL expiry. Cache can be disabled via configuration. Cache keys are normalized (e.g., whitespace-insensitive).

---

## 6. Reranking

> **REQ-401** | Priority: MUST
> **Description:** The system MUST rerank retrieved documents using a cross-encoder model. The reranker MUST score each `(query, document)` pair independently and output a relevance score normalized to the range 0.0–1.0 (via sigmoid).
> **Rationale:** Initial retrieval (BM25 + vector) uses fast but approximate scoring. Cross-encoder reranking provides fine-grained relevance assessment by attending to the full query-document interaction.
> **Acceptance Criteria:** Reranked order differs from initial retrieval order in at least some cases (the reranker is not a no-op). Scores are in the range [0.0, 1.0]. The reranker model is configurable.

> **REQ-402** | Priority: MUST
> **Description:** The reranker MUST return a configurable top-K subset of documents after reranking. The default top-K MUST be configurable without code changes.
> **Rationale:** Only the highest-relevance documents should proceed to generation. Passing too many documents dilutes context quality and increases generation cost.
> **Acceptance Criteria:** Changing `rerank_top_k` from 5 to 3 results in only 3 documents being passed to generation. The parameter is configurable via settings.

> **REQ-403** | Priority: MUST
> **Description:** The system MUST define and enforce reranker score thresholds for interpretation:
>
> | Score Range | Interpretation | Action |
> |-------------|---------------|--------|
> | > 0.75 | Strong match | High retrieval confidence |
> | 0.50–0.75 | Moderate match | Proceed with caution |
> | 0.30–0.50 | Weak match | Flag, consider re-retrieval |
> | < 0.30 | No relevant match | Do not generate |
>
> Documents scoring below the minimum threshold (default 0.30) MUST be excluded from generation context.
> **Rationale:** Generating answers from irrelevant documents is the primary source of hallucination. A hard floor prevents the LLM from receiving junk context.
> **Acceptance Criteria:** No document with a reranker score below 0.30 appears in the generation context. If all documents score below the threshold, the system returns "Insufficient documentation found" instead of generating an answer. Thresholds are configurable.

---

## 7. Document Formatting

> **REQ-501** | Priority: MUST
> **Description:** Every document chunk injected into the LLM context MUST carry structured metadata. The metadata MUST include at minimum:
> - Filename (source document name)
> - Version (document version identifier)
> - Date (document date, ISO format)
> - Domain (e.g., physical_design, verification, dft, frontend)
> - Section (section heading or number within the document)
> - Spec ID (stable identifier across versions, if available)
>
> Metadata MUST be formatted in a consistent, parseable structure within the context string.
> **Rationale:** Structured metadata enables the LLM to cite accurately (filename, version, section). It also enables the post-generation guardrail to verify citations against source metadata.
> **Acceptance Criteria:** Each document chunk in the LLM context includes all required metadata fields. A missing field is populated with "unknown" rather than omitted. The format is consistent across all chunks.

> **REQ-502** | Priority: MUST
> **Description:** The system MUST detect version conflicts before generation. When two or more retrieved documents share the same specification ID (or filename stem) but differ in version, the system MUST:
> 1. Flag the conflict in the pipeline state
> 2. Include the conflict information in the LLM prompt
> 3. Surface the conflict to the user in the final response
>
> The LLM MUST NOT silently resolve version conflicts.
> **Rationale:** In engineering workflows, specifications evolve across tape-out cycles. Silently choosing one version over another when both are retrieved can produce answers based on outdated or incorrect parameters.
> **Acceptance Criteria:** When documents from Power_Spec_v2.pdf and Power_Spec_v3.pdf are both retrieved, the system flags a version conflict. The generated answer includes a note about the conflict. The user sees which versions were found.

> **REQ-503** | Priority: MUST
> **Description:** The system MUST prepare a formatted context string for LLM injection that includes all retrieved chunks with their metadata and any version conflict flags. The format MUST be deterministic (same input produces same output).
> **Rationale:** A well-structured context string improves LLM citation accuracy and makes hallucination detection more reliable.
> **Acceptance Criteria:** The context string follows a documented format specification. Chunks are numbered sequentially. Metadata precedes content for each chunk. Version conflicts appear as explicit warnings in the context string.

---

## 8. Generation

> **REQ-601** | Priority: MUST
> **Description:** The system MUST use an anti-hallucination system prompt that instructs the LLM to:
> 1. Answer ONLY from the provided retrieved documents
> 2. Never use training data or prior knowledge
> 3. Cite sources using a specified format (e.g., `[Filename, Version, Section]`)
> 4. Explicitly state when information is insufficient rather than guessing
> 5. Report its confidence level (high, medium, low) as part of the response
> **Rationale:** Without explicit anti-hallucination instructions, LLMs default to generating plausible-sounding text from training data. In engineering contexts, a hallucinated specification is a design risk.
> **Acceptance Criteria:** The system prompt is stored as a separate file (not inline code). It contains all five instruction categories above. Changing the prompt does not require code changes.

> **REQ-602** | Priority: SHOULD
> **Description:** The system SHOULD use a template engine for prompt construction that safely handles variable injection. The template MUST NOT interpret content within retrieved documents as template variables (e.g., curly braces `{}` in JSON snippets within documents MUST NOT be treated as template placeholders).
> **Rationale:** Retrieved documents frequently contain JSON, YAML, code snippets, and other content with curly braces. Naive string formatting (f-strings, `.format()`) will break or produce incorrect prompts when documents contain `{key}` patterns.
> **Acceptance Criteria:** A retrieved document containing `{"voltage": "1.8V"}` does not cause a template error or variable substitution. Only explicitly declared template variables are substituted.

> **REQ-603** | Priority: MUST
> **Description:** The system MUST enforce a source citation format in generated answers. Each claim in the answer MUST reference a specific retrieved document using an identifier that the user can trace back to the source. The citation format MUST include at minimum: filename and section.
> **Rationale:** Citations make hallucination detectable. An engineer can verify a claim by checking the cited source. Without citations, there is no way to distinguish grounded claims from hallucinated ones.
> **Acceptance Criteria:** Every factual claim in a generated answer includes a citation. Citations are traceable to a specific retrieved document chunk. Answers without citations are flagged by the post-generation guardrail.

> **REQ-604** | Priority: MUST
> **Description:** The system MUST extract a self-reported confidence level from the LLM's response. The LLM is instructed to report confidence as "high", "medium", or "low". This value MUST be parsed from the response and mapped to a numerical score for use in the composite confidence calculation.
> **Rationale:** LLM self-reported confidence is one of three signals in the composite confidence score. While LLMs tend toward overconfidence, this signal provides useful information when combined with objective signals.
> **Acceptance Criteria:** The confidence level is reliably extracted from LLM responses. A downward correction is applied to account for overconfidence bias (e.g., "high" maps to 0.85, not 1.0). Parsing failures default to a neutral score (0.5).

> **REQ-605** | Priority: MUST
> **Description:** The system MUST implement retry logic with exponential backoff for all external LLM calls (generation and query processing). The retry configuration MUST include:
> - Maximum number of retries (configurable, default 3)
> - Base backoff interval (configurable)
> - Maximum backoff cap
> - Graceful fallback when all retries are exhausted
> **Rationale:** A single network hiccup or LLM provider timeout should not kill the entire query. Retry logic with backoff handles transient failures without overwhelming the service.
> **Acceptance Criteria:** A transient timeout on the first attempt is retried automatically. Backoff intervals increase between retries. After all retries are exhausted, the system returns a graceful error (not a crash). Retry count and intervals are configurable.

---

## 9. Post-Generation Guardrail

> **REQ-701** | Priority: MUST
> **Description:** The system MUST compute a composite confidence score from three independent signals:
>
> | Signal | Weight | Source | Objectivity |
> |--------|--------|--------|-------------|
> | Retrieval Confidence | 0.50 | Top-3 reranker score average | Objective |
> | LLM Self-Reported Confidence | 0.25 | Extracted from LLM response | Subjective |
> | Citation Coverage | 0.25 | Fraction of answer sentences grounded in retrieved quotes | Structural |
>
> The weights MUST be configurable. The combined score MUST be in the range 0.0–1.0.
> **Rationale:** No single signal is reliable alone. Reranker scores measure retrieval quality but not generation quality. LLM confidence is useful but biased. Citation coverage measures structural grounding. Combining all three produces a more robust confidence estimate.
> **Acceptance Criteria:** The composite score is computed for every generated answer. Changing the weights in configuration changes the composite score. Each individual signal is logged alongside the composite for debugging.

> **REQ-702** | Priority: MUST
> **Description:** The system MUST verify that claims in the generated answer are grounded in the retrieved documents. Every factual sentence in the answer SHOULD be traceable to a specific retrieved chunk. Sentences that cannot be traced to any retrieved document MUST be flagged.
> **Rationale:** This is the primary hallucination detection mechanism. If the LLM generates content not present in the retrieved context, it is hallucinating from training data.
> **Acceptance Criteria:** The system computes citation coverage (fraction of answer sentences with matching retrieved content). Answers with citation coverage below a configurable threshold are flagged. Flagged answers trigger the confidence routing logic (re-retrieve or escalate).

> **REQ-703** | Priority: MUST
> **Description:** The system MUST detect and redact PII from generated answers before they reach the user. PII detection MUST cover at minimum:
> - Email addresses (regex)
> - Phone numbers (regex)
> - Social security numbers / employee IDs (regex)
> - Person names (named entity recognition)
>
> Detected PII MUST be replaced with typed placeholders (e.g., `[EMAIL]`, `[PHONE]`, `[PERSON]`).
> **Rationale:** Retrieved documents may contain PII (author names, contact information, employee references). The LLM may surface this PII in generated answers. PII in answers is a data handling violation regardless of whether the source documents are internal.
> **Acceptance Criteria:** An answer containing "Contact john.smith@company.com for details" is redacted to "Contact [EMAIL] for details". PII detection runs on every generated answer. Redactions are logged for audit.

> **REQ-704** | Priority: MUST
> **Description:** The system MUST sanitize generated output to remove:
> - Leaked system prompt fragments
> - Internal metadata or formatting artifacts
> - Template variable names or placeholders that were not substituted
> - Raw document chunk boundaries or markers
> **Rationale:** LLMs occasionally echo system prompt content or internal formatting in their output. Leaking the system prompt reveals anti-hallucination instructions and injection patterns.
> **Acceptance Criteria:** The generated answer does not contain any substring from the system prompt. Internal markers (e.g., `--- Document 3 ---`) are stripped from the final output.

> **REQ-705** | Priority: SHOULD
> **Description:** For queries classified as HIGH risk (per REQ-203), the system SHOULD apply additional output filtering:
> - Numerical values (voltages, frequencies, temperatures, timing values) that do not have an exact quote match in a retrieved document SHOULD be flagged or suppressed
> - The answer SHOULD include a verification warning (e.g., "VERIFY BEFORE IMPLEMENTATION")
> **Rationale:** In HIGH risk domains (electrical specifications, safety compliance), an incorrect number is not a UX problem — it is a design risk. Requiring exact quote matches for numerical claims adds a safety margin.
> **Acceptance Criteria:** A HIGH risk answer containing "The voltage is 3.3V" where no retrieved document contains "3.3V" is flagged. The flag is visible to the user. LOW risk answers are not subject to this additional filtering.

> **REQ-706** | Priority: MUST
> **Description:** The system MUST route based on the composite confidence score and risk level:
>
> | Composite Score | Risk Level | Action |
> |----------------|------------|--------|
> | > 0.70 | LOW or MEDIUM | Return answer to user |
> | > 0.70 | HIGH | Return answer with verification warning |
> | 0.50–0.70 | Any | Re-retrieve with broader parameters (one retry) |
> | < 0.50 | Any | Do not return generated answer; return "Insufficient documentation found" |
>
> Re-retrieval MUST be attempted at most once. If re-retrieval does not improve confidence above the threshold, the system MUST escalate (flag for review or return insufficient documentation message).
> **Rationale:** This routing logic ensures that low-confidence answers are never silently returned to the user. The single retry with broader parameters gives the system one chance to recover before giving up.
> **Acceptance Criteria:** An answer with composite confidence 0.45 triggers re-retrieval. If re-retrieval produces confidence 0.72, the answer is returned. If re-retrieval produces confidence 0.48, the system returns "Insufficient documentation found". HIGH risk answers always include a verification warning regardless of confidence.

---

## 10. Observability

> **REQ-801** | Priority: MUST
> **Description:** The system MUST produce an end-to-end trace for every query processed. The trace MUST include a unique trace ID, the risk level, and a record for each pipeline stage.
> **Rationale:** Without tracing, there is no way to diagnose why a specific query produced a bad answer. Was it poor retrieval? Bad reranking? LLM hallucination? Tracing makes the failure mode visible.
> **Acceptance Criteria:** Every query response includes a trace ID. The trace is retrievable by ID and shows all pipeline stages with their inputs, outputs, and timing.

> **REQ-802** | Priority: MUST
> **Description:** The system MUST capture per-stage metrics for every query:
>
> | Stage | Metrics |
> |-------|---------|
> | Query Processing | Reformulation count, confidence score, action taken |
> | Pre-Retrieval Guardrail | Validation pass/fail, risk classification, PII detections |
> | Retrieval | Search latency, result count, filter hit rate, KG expansion terms |
> | Reranking | Score distribution (min, max, mean), top-1 score |
> | Document Formatting | Chunk count, version conflicts detected |
> | Generation | Generation latency, LLM confidence, token count |
> | Post-Generation Guardrail | Composite confidence, citation coverage, PII redactions, routing action |
>
> **Rationale:** These metrics enable trend analysis, regression detection, and targeted optimization.
> **Acceptance Criteria:** All listed metrics are captured and available for querying/dashboarding. Metrics are structured (key-value, not embedded in log messages).

> **REQ-803** | Priority: SHOULD
> **Description:** The system SHOULD define alerting thresholds for key metrics and trigger alerts when thresholds are breached:
> - Average composite confidence drops below 0.60
> - End-to-end latency exceeds configurable target
> - PII detection rate exceeds baseline (potential data quality issue)
> - Re-retrieval rate exceeds 30% (retrieval quality degradation)
> **Rationale:** Automated alerts surface systemic problems before users report them.
> **Acceptance Criteria:** Alerts are triggered when thresholds are breached. Alerts include the metric name, current value, threshold, and time window.

---

## 11. Non-Functional Requirements

> **REQ-901** | Priority: SHOULD
> **Description:** The system SHOULD meet the following latency targets for each pipeline stage under standard load:
>
> | Stage | Target Latency |
> |-------|---------------|
> | Query Processing | < 2s |
> | Pre-Retrieval Guardrail | < 100ms |
> | Retrieval (search) | < 500ms |
> | Reranking | < 1s |
> | Document Formatting | < 50ms |
> | Generation | < 5s |
> | Post-Generation Guardrail | < 500ms |
> | **Total end-to-end** | **< 10s** |
>
> **Rationale:** Interactive usage requires responsive answers. Each stage has a latency budget proportional to its computational complexity.
> **Acceptance Criteria:** Median latency per stage is measured and compared against targets. P95 latency is also tracked.

> **REQ-902** | Priority: MUST
> **Description:** The system MUST degrade gracefully when optional components are unavailable:
>
> | Component Unavailable | Degraded Behavior |
> |----------------------|-------------------|
> | External LLM (generation) | Return retrieved documents without synthesis; skip generation |
> | External LLM (query processing) | Use heuristic confidence scoring; skip reformulation |
> | Knowledge Graph | Skip KG expansion; use original query for BM25 |
> | Embedding cache | Recompute embeddings (slower, but functional) |
> | Query result cache | Recompute full pipeline (slower, but functional) |
>
> The system MUST NOT crash or return an unhandled error when any single optional component is unavailable.
> **Rationale:** In production, components fail. The pipeline must continue operating in a degraded but functional state rather than failing entirely.
> **Acceptance Criteria:** Each degradation scenario is tested. The system logs a warning when operating in degraded mode. The response indicates which components were unavailable.

> **REQ-903** | Priority: MUST
> **Description:** All configurable thresholds, weights, patterns, and parameters MUST be externalized to configuration files (not hardcoded in source code). Changes to configuration MUST take effect on restart without code changes.
>
> Configuration categories:
> - Search parameters (alpha, search_limit, rerank_top_k)
> - Confidence thresholds and weights
> - Risk classification taxonomy
> - Injection detection patterns
> - PII detection patterns
> - Retry parameters (count, backoff intervals)
> - Latency targets
> - Cache sizes and TTLs
> **Rationale:** Hardcoded values require code changes and redeployment to tune. Externalized configuration enables rapid iteration on retrieval quality without engineering cycles.
> **Acceptance Criteria:** Every threshold, weight, and pattern referenced in this specification is loaded from a configuration file. The configuration file format is documented. Missing configuration values fall back to documented defaults.

---

## 12. Requirements Traceability Matrix

| REQ ID | Section | Priority | Pipeline Stage |
|--------|---------|----------|---------------|
| REQ-101 | 3 | MUST | Query Processing |
| REQ-102 | 3 | MUST | Query Processing |
| REQ-103 | 3 | SHOULD | Query Processing |
| REQ-104 | 3 | MUST | Query Processing |
| REQ-201 | 4 | MUST | Pre-Retrieval Guardrail |
| REQ-202 | 4 | MUST | Pre-Retrieval Guardrail |
| REQ-203 | 4 | MUST | Pre-Retrieval Guardrail |
| REQ-204 | 4 | SHOULD | Pre-Retrieval Guardrail |
| REQ-205 | 4 | MUST | Pre-Retrieval Guardrail |
| REQ-301 | 5 | MUST | Retrieval |
| REQ-302 | 5 | MUST | Retrieval |
| REQ-303 | 5 | MUST | Retrieval |
| REQ-304 | 5 | MAY | Retrieval |
| REQ-305 | 5 | MUST | Retrieval |
| REQ-306 | 5 | SHOULD | Retrieval |
| REQ-307 | 5 | SHOULD | Retrieval |
| REQ-308 | 5 | SHOULD | Retrieval |
| REQ-401 | 6 | MUST | Reranking |
| REQ-402 | 6 | MUST | Reranking |
| REQ-403 | 6 | MUST | Reranking |
| REQ-501 | 7 | MUST | Document Formatting |
| REQ-502 | 7 | MUST | Document Formatting |
| REQ-503 | 7 | MUST | Document Formatting |
| REQ-601 | 8 | MUST | Generation |
| REQ-602 | 8 | SHOULD | Generation |
| REQ-603 | 8 | MUST | Generation |
| REQ-604 | 8 | MUST | Generation |
| REQ-605 | 8 | MUST | Generation |
| REQ-701 | 9 | MUST | Post-Generation Guardrail |
| REQ-702 | 9 | MUST | Post-Generation Guardrail |
| REQ-703 | 9 | MUST | Post-Generation Guardrail |
| REQ-704 | 9 | MUST | Post-Generation Guardrail |
| REQ-705 | 9 | SHOULD | Post-Generation Guardrail |
| REQ-706 | 9 | MUST | Post-Generation Guardrail |
| REQ-801 | 10 | MUST | Observability |
| REQ-802 | 10 | MUST | Observability |
| REQ-803 | 10 | SHOULD | Observability |
| REQ-901 | 11 | SHOULD | Non-Functional |
| REQ-902 | 11 | MUST | Non-Functional |
| REQ-903 | 11 | MUST | Non-Functional |

**Total Requirements: 39**
- MUST: 28
- SHOULD: 9
- MAY: 1
