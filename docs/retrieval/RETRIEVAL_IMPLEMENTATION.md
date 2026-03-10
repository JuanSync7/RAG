# Retrieval Pipeline — Implementation Guide

**AION Knowledge Management Platform**
Version: 1.0 | Status: Draft | Domain: Retrieval Pipeline

This document provides a phased implementation plan and detailed code appendix for the retrieval pipeline specified in `RETRIEVAL_SPEC.md`. Every task references the requirements it satisfies.

---

# Part A: Task-Oriented Overview

## Phase 1 — Core Pipeline Hardening

Foundation work: guardrails, validation, and resilience. These tasks make the pipeline safe before adding new capabilities.

### Task 1.1: Pre-Retrieval Guardrail Layer

**Description:** Build a guardrail module that sits between query processing and retrieval. It validates inputs, detects injection, classifies risk, and optionally filters PII from the query.

**Requirements Covered:** REQ-201, REQ-202, REQ-203, REQ-204, REQ-205, REQ-903

**Dependencies:** None — this is a new module.

**Complexity:** M

**Subtasks:**
1. Define a `GuardrailResult` data structure (pass/reject, risk level, sanitized query, PII detections)
2. Implement input validation (query length, parameter ranges, filter sanitization)
3. Load injection patterns from external YAML config file
4. Implement risk classifier with externalized keyword taxonomy
5. Implement conditional PII filtering (gated on `EXTERNAL_LLM_MODE` config flag)
6. Wire into the pipeline between query processing and retrieval

---

### Task 1.2: Post-Generation Guardrail Layer

**Description:** Build a guardrail module that sits between generation and answer delivery. It computes confidence, detects hallucination, filters PII from the answer, sanitizes output, and routes based on confidence + risk.

**Requirements Covered:** REQ-701, REQ-702, REQ-703, REQ-704, REQ-705, REQ-706

**Dependencies:** Task 2.1 (confidence scoring logic), Task 2.3 (risk classification — shared with Task 1.1)

**Complexity:** L

**Subtasks:**
1. Define a `PostGuardrailResult` data structure (action: return/re-retrieve/flag, redacted answer, confidence breakdown)
2. Implement PII detection on generated answer (regex + optional NER)
3. Implement output sanitization (system prompt leak detection, artifact stripping)
4. Implement risk-based output filtering for HIGH risk queries (numerical claim verification)
5. Implement confidence routing logic (threshold-based, with single re-retrieval retry)
6. Wire into the pipeline between generation and answer delivery

---

### Task 1.3: Retry Logic for External LLM Calls

**Description:** Add exponential backoff retry wrapper for all external LLM calls (query reformulation, confidence evaluation, answer generation). Include graceful fallback when all retries are exhausted.

**Requirements Covered:** REQ-605, REQ-902

**Dependencies:** None.

**Complexity:** S

**Subtasks:**
1. Implement a generic retry decorator/wrapper with configurable max retries, base interval, and max backoff
2. Apply to query processing LLM calls (reformulation, evaluation)
3. Apply to generation LLM call
4. Define fallback behavior for each call site (heuristic confidence, skip generation, etc.)
5. Externalize retry parameters to config (REQ-903)

---

### Task 1.4: Input Validation at System Boundaries

**Description:** Add validation for all external inputs that flow into the retrieval pipeline: search parameters, metadata filters, and query content.

**Requirements Covered:** REQ-201, REQ-903

**Dependencies:** None. Can be implemented within Task 1.1 or as a standalone utility.

**Complexity:** S

**Subtasks:**
1. Define valid ranges for `alpha` (0.0–1.0), `search_limit` (1–100), `rerank_top_k` (1–50)
2. Sanitize metadata filter values (prevent Weaviate query language injection)
3. Validate query length (min/max configurable)
4. Return structured error responses for invalid inputs (not exceptions)

---

## Phase 2 — Confidence & Routing

The intelligence layer: composite scoring, risk-aware routing, and full-pipeline orchestration.

### Task 2.1: 3-Signal Confidence Scoring

**Description:** Implement the composite confidence score that combines retrieval confidence (reranker scores), LLM self-reported confidence, and citation coverage into a single 0.0–1.0 score.

**Requirements Covered:** REQ-701, REQ-604

**Dependencies:** REQ-403 (reranker score thresholds — already exists), REQ-604 (LLM confidence extraction)

**Complexity:** M

**Subtasks:**
1. Implement retrieval confidence calculation (top-3 reranker score average)
2. Implement LLM confidence extraction and parsing (high/medium/low → numerical with downward correction)
3. Implement citation coverage calculation (sentence-level grounding check)
4. Implement weighted combination with configurable weights
5. Externalize weights and threshold values to config (REQ-903)

---

### Task 2.2: Full-Pipeline LangGraph Routing

**Description:** Extend the existing LangGraph workflow from query-processing-only to wrap the entire pipeline. Add conditional routing based on composite confidence: return answer, re-retrieve with broader parameters, or flag for review.

**Requirements Covered:** REQ-706, REQ-902

**Dependencies:** Task 2.1 (confidence scoring), Task 1.1 (pre-retrieval guardrail), Task 1.2 (post-generation guardrail)

**Complexity:** L

**Subtasks:**
1. Define `RAGPipelineState` TypedDict covering all pipeline stages (query, transformed query, retrieved docs, ranked docs, formatted context, answer, confidence, risk level, retry count)
2. Define graph nodes for each pipeline stage (transform → guardrail → retrieve → rerank → format → generate → evaluate → route)
3. Implement conditional edges from evaluate node (return / re-retrieve / flag)
4. Implement re-retrieval with broader parameters (increased top-k, lower alpha toward BM25, relaxed filters)
5. Enforce single retry limit (retry_count < 1)
6. Wire human review / escalation node for flagged queries

---

### Task 2.3: Risk Classification

**Description:** Implement deterministic keyword-based risk classification for queries. HIGH risk queries trigger additional verification in the post-generation guardrail.

**Requirements Covered:** REQ-203, REQ-705, REQ-903

**Dependencies:** None — standalone classifier.

**Complexity:** S

**Subtasks:**
1. Define risk taxonomy (HIGH/MEDIUM/LOW keyword lists) in external config file
2. Implement classifier: scan query for keyword matches, return highest matching level
3. Attach risk level to pipeline state for downstream use
4. Ensure taxonomy config is reloadable on restart

---

## Phase 3 — Retrieval Quality

Improve the quality of what gets retrieved and how it's presented to the LLM.

### Task 3.1: Structured Document Formatter

**Description:** Replace the current simple chunk formatting with structured metadata attachment. Each chunk injected into the LLM context must carry filename, version, date, domain, section, and spec ID.

**Requirements Covered:** REQ-501, REQ-503

**Dependencies:** Metadata schema must be populated at ingestion time (out of scope, but formatter must handle missing fields gracefully).

**Complexity:** S

**Subtasks:**
1. Define the formatted chunk template (metadata header + content body)
2. Implement formatter function that maps document metadata to the template
3. Handle missing metadata fields (default to "unknown")
4. Number chunks sequentially in the context string
5. Ensure format is deterministic

---

### Task 3.2: Version Conflict Detection

**Description:** Detect when retrieved documents share the same specification ID or filename stem but differ in version. Flag conflicts before generation.

**Requirements Covered:** REQ-502

**Dependencies:** Task 3.1 (document formatter — conflicts must be included in formatted context)

**Complexity:** M

**Subtasks:**
1. Group retrieved documents by spec_id or filename stem
2. Within each group, detect version mismatches
3. Build conflict records (spec, version A, version B, dates)
4. Inject conflict warnings into the LLM context string
5. Include conflict information in the final response to the user

---

### Task 3.3: PromptTemplate Integration

**Description:** Replace raw string formatting for prompt construction with a template engine that safely handles variable injection. Documents containing curly braces (JSON, code) must not be interpreted as template variables.

**Requirements Covered:** REQ-601, REQ-602

**Dependencies:** None.

**Complexity:** S

**Subtasks:**
1. Store system prompt in a separate markdown file (no variables — static rules only)
2. Store human turn template in a separate file with named placeholders (`{documents}`, `{question}`)
3. Use a template engine (e.g., LangChain ChatPromptTemplate) that only substitutes declared variables
4. Verify that document content with `{curly_braces}` is passed through safely
5. Ensure prompt changes take effect on restart without code changes

---

### Task 3.4: Multi-Turn Context / Coreference Resolution

**Description:** Add conversation history tracking and coreference resolution to the query processing stage. Enable follow-up queries that reference prior turns.

**Requirements Covered:** REQ-103

**Dependencies:** None, but integrates with Task 2.2 (pipeline state must carry conversation history).

**Complexity:** M

**Subtasks:**
1. Define a conversation buffer (last N turns, configurable)
2. Thread conversation history through the pipeline state
3. Modify the query reformulation prompt to include recent conversation context
4. Implement coreference resolution: detect pronouns/references and resolve against prior turns
5. Optionally persist conversation state (JSON file or in-memory with TTL)

---

## Phase 4 — Performance & Observability

Make the pipeline fast and debuggable.

### Task 4.1: Connection Pooling for Vector Database

**Description:** Replace per-query connection creation with a persistent connection pool. Add health checks on startup.

**Requirements Covered:** REQ-307

**Dependencies:** None.

**Complexity:** S

**Subtasks:**
1. Create a connection pool manager (singleton or module-level)
2. Support external vector database via URL configuration (not just embedded)
3. Implement startup health check (fail-fast if DB is unreachable)
4. Implement periodic liveness checks during operation
5. Handle connection failures with reconnection logic

---

### Task 4.2: Embedding Cache (LRU)

**Description:** Add an LRU cache around the query embedding function. Identical queries return cached embeddings without recomputation.

**Requirements Covered:** REQ-306

**Dependencies:** None.

**Complexity:** S

**Subtasks:**
1. Wrap the embed_query function with an LRU cache (configurable max size)
2. Use the raw query string as cache key (normalize whitespace)
3. Ensure cache is thread-safe if concurrent queries are supported
4. Add cache hit/miss metrics for observability (REQ-802)
5. Externalize cache size to config (REQ-903)

---

### Task 4.3: Query Result Cache (TTL)

**Description:** Cache full pipeline responses keyed by `(processed_query, filters)` with a configurable TTL. Cache hits bypass all downstream stages.

**Requirements Covered:** REQ-308

**Dependencies:** None, but should be placed early in the pipeline (after query processing, before retrieval).

**Complexity:** S

**Subtasks:**
1. Define cache key: normalized `(processed_query, source_filter, heading_filter, alpha)` tuple
2. Implement TTL-based cache (dict + timestamps, or `cachetools.TTLCache`)
3. On cache hit, return cached response immediately
4. On cache miss, proceed through pipeline and store result
5. Add cache bypass flag for debugging
6. Externalize TTL and max cache size to config (REQ-903)

---

### Task 4.4: Observability Instrumentation

**Description:** Add end-to-end tracing and per-stage metrics capture for every query processed.

**Requirements Covered:** REQ-801, REQ-802, REQ-803

**Dependencies:** All pipeline stages must exist (Phases 1–3).

**Complexity:** M

**Subtasks:**
1. Choose observability backend (e.g., Langfuse, OpenTelemetry, or structured JSON logging)
2. Generate unique trace ID per query
3. Implement trace decorator/wrapper for each pipeline stage
4. Capture all metrics defined in REQ-802 (latency, scores, counts per stage)
5. Implement alerting thresholds (REQ-803) with configurable values
6. Ensure trace ID is included in the final response for debugging

---

## Phase 5 — Security

Harden the pipeline against data leakage and injection.

### Task 5.1: Externalize Injection Patterns

**Description:** Move hardcoded injection detection patterns to an external config file. Patterns are loaded at startup and applied during pre-retrieval guardrail processing.

**Requirements Covered:** REQ-202, REQ-903

**Dependencies:** Task 1.1 (pre-retrieval guardrail — this is a subtask of 1.1, broken out for clarity).

**Complexity:** S

**Subtasks:**
1. Create a YAML/JSON config file for injection patterns
2. Migrate existing hardcoded patterns to the config file
3. Load patterns at startup, compile to regex
4. Log pattern count on load for verification
5. Document the pattern format for maintainers

---

### Task 5.2: Pre-Retrieval PII Filtering

**Description:** Detect and redact PII from user queries before they are sent to external LLM APIs. Conditional on `EXTERNAL_LLM_MODE` configuration flag.

**Requirements Covered:** REQ-204

**Dependencies:** Task 1.1 (pre-retrieval guardrail — this is a subtask of 1.1, broken out for clarity).

**Complexity:** M

**Subtasks:**
1. Implement regex-based PII detection (email, phone, SSN/employee ID patterns)
2. Implement NER-based person name detection (optional, using existing entity extraction or a lightweight model)
3. Replace detected PII with typed placeholders (`[PERSON]`, `[EMAIL]`, `[PHONE]`)
4. Preserve original query internally (for retrieval against local DB — PII is fine locally)
5. Only activate when `EXTERNAL_LLM_MODE=true` in config
6. Log PII detection events (count, types) without logging the actual PII values

---

### Task 5.3: Post-Generation PII Filtering

**Description:** Detect and redact PII from generated answers before they reach the user. This runs regardless of LLM deployment mode.

**Requirements Covered:** REQ-703

**Dependencies:** Task 1.2 (post-generation guardrail — this is a subtask of 1.2, broken out for clarity).

**Complexity:** M

**Subtasks:**
1. Reuse PII detection logic from Task 5.2 (shared module)
2. Apply to the generated answer text
3. Replace PII with typed placeholders
4. Log redaction events for audit (count and type, not the PII values)
5. This filter runs on every generated answer (not conditional)

---

## Task Dependency Graph

```
Phase 1 (Foundation)
├── Task 1.1: Pre-Retrieval Guardrail
├── Task 1.3: Retry Logic                                     
├── Task 1.4: Input Validation (can merge into 1.1)           
│                                                             
Phase 2 (Confidence & Routing)                                
├── Task 2.3: Risk Classification ◄────Task 1.1
├── Task 2.1: 3-Signal Confidence                             
│                                                             
├── Task 1.2: Post-Generation Guardrail ◄── Task 2.1, 2.3     
│                                                             
└── Task 2.2: Full-Pipeline LangGraph ◄── Task 1.1, 1.2, 2.1  
                                                              
Phase 3 (Retrieval Quality)                                   
├── Task 3.1: Document Formatter                              
├── Task 3.2: Version Conflict Detection ◄── Task 3.1         
├── Task 3.3: PromptTemplate Integration                      
└── Task 3.4: Multi-Turn Context                              
                                                              
Phase 4 (Performance & Observability)                         
├── Task 4.1: Connection Pooling                              
├── Task 4.2: Embedding Cache                                 
├── Task 4.3: Query Result Cache                              
└── Task 4.4: Observability ◄── All pipeline stages           
                                                              
Phase 5 (Security)                                            
├── Task 5.1: Externalize Injection Patterns ◄── Task 1.1
├── Task 5.2: Pre-Retrieval PII Filtering ◄── Task 1.1
└── Task 5.3: Post-Generation PII Filtering ◄── Task 1.2
```

---

## Task-to-Requirement Mapping

| Task | Requirements Covered |
|------|---------------------|
| 1.1 Pre-Retrieval Guardrail | REQ-201, REQ-202, REQ-203, REQ-204, REQ-205, REQ-903 |
| 1.2 Post-Generation Guardrail | REQ-701, REQ-702, REQ-703, REQ-704, REQ-705, REQ-706 |
| 1.3 Retry Logic | REQ-605, REQ-902 |
| 1.4 Input Validation | REQ-201, REQ-903 |
| 2.1 3-Signal Confidence | REQ-701, REQ-604 |
| 2.2 LangGraph Routing | REQ-706, REQ-902 |
| 2.3 Risk Classification | REQ-203, REQ-705, REQ-903 |
| 3.1 Document Formatter | REQ-501, REQ-503 |
| 3.2 Version Conflict Detection | REQ-502 |
| 3.3 PromptTemplate Integration | REQ-601, REQ-602 |
| 3.4 Multi-Turn Context | REQ-103 |
| 4.1 Connection Pooling | REQ-307 |
| 4.2 Embedding Cache | REQ-306 |
| 4.3 Query Result Cache | REQ-308 |
| 4.4 Observability | REQ-801, REQ-802, REQ-803 |
| 5.1 Externalize Injection Patterns | REQ-202, REQ-903 |
| 5.2 Pre-Retrieval PII Filtering | REQ-204 |
| 5.3 Post-Generation PII Filtering | REQ-703 |

---

# Part B: Code Appendix

## B.1 Pre-Retrieval Guardrail

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import re
import yaml


class RiskLevel(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class GuardrailAction(Enum):
    PASS = "pass"
    REJECT = "reject"


@dataclass
class GuardrailResult:
    action: GuardrailAction
    risk_level: RiskLevel
    sanitized_query: str
    rejection_reason: Optional[str] = None      # Internal log only
    user_message: Optional[str] = None           # User-safe message
    pii_detections: list[dict] = field(default_factory=list)


class PreRetrievalGuardrail:
    def __init__(self, config_path: str = "config/guardrails.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.max_query_length = config.get("max_query_length", 500)
        self.min_query_length = config.get("min_query_length", 2)
        self.injection_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in config.get("injection_patterns", [])
        ]
        self.risk_taxonomy = config.get("risk_taxonomy", {})
        self.param_ranges = config.get("parameter_ranges", {
            "alpha": {"min": 0.0, "max": 1.0},
            "search_limit": {"min": 1, "max": 100},
            "rerank_top_k": {"min": 1, "max": 50},
        })
        self.external_llm_mode = config.get("external_llm_mode", False)

    def validate(
        self,
        query: str,
        alpha: float = 0.5,
        search_limit: int = 10,
        rerank_top_k: int = 5,
        source_filter: Optional[str] = None,
        heading_filter: Optional[str] = None,
    ) -> GuardrailResult:
        # 1. Length validation
        if len(query.strip()) < self.min_query_length:
            return GuardrailResult(
                action=GuardrailAction.REJECT,
                risk_level=RiskLevel.LOW,
                sanitized_query=query,
                rejection_reason="Query too short",
                user_message="Please provide a more detailed query.",
            )

        if len(query) > self.max_query_length:
            return GuardrailResult(
                action=GuardrailAction.REJECT,
                risk_level=RiskLevel.LOW,
                sanitized_query=query,
                rejection_reason=f"Query exceeds {self.max_query_length} chars",
                user_message="Your query is too long. Please shorten it.",
            )

        # 2. Parameter range validation
        param_errors = self._validate_params(alpha, search_limit, rerank_top_k)
        if param_errors:
            return GuardrailResult(
                action=GuardrailAction.REJECT,
                risk_level=RiskLevel.LOW,
                sanitized_query=query,
                rejection_reason=f"Invalid parameters: {param_errors}",
                user_message="Invalid search parameters provided.",
            )

        # 3. Injection detection
        for pattern in self.injection_patterns:
            if pattern.search(query):
                return GuardrailResult(
                    action=GuardrailAction.REJECT,
                    risk_level=RiskLevel.LOW,
                    sanitized_query=query,
                    rejection_reason="Injection pattern detected",
                    user_message="Your query could not be processed.",
                )

        # 4. Risk classification
        risk_level = self._classify_risk(query)

        # 5. PII filtering (conditional)
        sanitized_query = query
        pii_detections = []
        if self.external_llm_mode:
            sanitized_query, pii_detections = self._filter_pii(query)

        return GuardrailResult(
            action=GuardrailAction.PASS,
            risk_level=risk_level,
            sanitized_query=sanitized_query,
            pii_detections=pii_detections,
        )

    def _validate_params(
        self, alpha: float, search_limit: int, rerank_top_k: int
    ) -> list[str]:
        errors = []
        ranges = self.param_ranges
        if not (ranges["alpha"]["min"] <= alpha <= ranges["alpha"]["max"]):
            errors.append(f"alpha must be {ranges['alpha']['min']}-{ranges['alpha']['max']}")
        if not (ranges["search_limit"]["min"] <= search_limit <= ranges["search_limit"]["max"]):
            errors.append(f"search_limit must be {ranges['search_limit']['min']}-{ranges['search_limit']['max']}")
        if not (ranges["rerank_top_k"]["min"] <= rerank_top_k <= ranges["rerank_top_k"]["max"]):
            errors.append(f"rerank_top_k must be {ranges['rerank_top_k']['min']}-{ranges['rerank_top_k']['max']}")
        return errors

    def _classify_risk(self, query: str) -> RiskLevel:
        query_lower = query.lower()
        for keyword in self.risk_taxonomy.get("HIGH", []):
            if keyword in query_lower:
                return RiskLevel.HIGH
        for keyword in self.risk_taxonomy.get("MEDIUM", []):
            if keyword in query_lower:
                return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _filter_pii(self, query: str) -> tuple[str, list[dict]]:
        detections = []
        filtered = query

        # Email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, filtered):
            detections.append({"type": "EMAIL", "position": match.span()})
        filtered = re.sub(email_pattern, "[EMAIL]", filtered)

        # Phone (basic international patterns)
        phone_pattern = r'\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b'
        for match in re.finditer(phone_pattern, filtered):
            detections.append({"type": "PHONE", "position": match.span()})
        filtered = re.sub(phone_pattern, "[PHONE]", filtered)

        # Employee ID (alphanumeric patterns like EMP-12345, E12345)
        emp_pattern = r'\b(?:EMP[-.]?\d{4,8}|E\d{5,8})\b'
        for match in re.finditer(emp_pattern, filtered, re.IGNORECASE):
            detections.append({"type": "EMPLOYEE_ID", "position": match.span()})
        filtered = re.sub(emp_pattern, "[EMPLOYEE_ID]", filtered, flags=re.IGNORECASE)

        return filtered, detections
```

---

## B.2 Risk Classification Config

```yaml
# config/guardrails.yaml

max_query_length: 500
min_query_length: 2
external_llm_mode: false

parameter_ranges:
  alpha:
    min: 0.0
    max: 1.0
  search_limit:
    min: 1
    max: 100
  rerank_top_k:
    min: 1
    max: 50

risk_taxonomy:
  HIGH:
    # Electrical
    - "voltage"
    - "current"
    - "power domain"
    - "supply rail"
    - "vdd"
    - "vss"
    # Timing
    - "timing constraint"
    - "setup time"
    - "hold time"
    - "clock frequency"
    - "propagation delay"
    - "skew"
    - "jitter"
    # Safety/Compliance
    - "iso26262"
    - "do-254"
    - "safety"
    - "compliance"
    - "functional safety"
    - "hazard"
    - "fault"
    - "asil"
    # Critical specs
    - "threshold"
    - "limit"
    - "maximum"
    - "minimum"
    - "specification"
    - "temperature range"
    - "operating condition"
  MEDIUM:
    - "procedure"
    - "guideline"
    - "checklist"
    - "review"
    - "signoff"
    - "flow"
    - "methodology"
    - "constraint file"
    - "sdc"
    - "upf"

injection_patterns:
  - "ignore.*(all|previous|prior|above).*instructions"
  - "you are now"
  - "^system:\\s"
  - "<\\/?[a-z]+>"
  - "\\[INST\\]"
  - "forget.*(everything|all|previous)"
  - "(sudo|admin|root)\\s+(access|mode|command)"
  - "disregard.*prompt"
  - "override.*safety"
```

---

## B.3 3-Signal Confidence Scoring

```python
from dataclasses import dataclass


@dataclass
class ConfidenceBreakdown:
    retrieval_score: float       # From reranker (objective)
    llm_score: float             # From LLM self-report (subjective)
    citation_score: float        # From citation coverage (structural)
    composite: float             # Weighted combination
    retrieval_weight: float
    llm_weight: float
    citation_weight: float


# Downward correction map for LLM overconfidence
LLM_CONFIDENCE_MAP = {
    "high": 0.85,      # Not 1.0 — LLMs overestimate
    "medium": 0.55,    # Not 0.6
    "low": 0.25,       # Not 0.3
}


def compute_retrieval_confidence(reranker_scores: list[float], top_n: int = 3) -> float:
    """Average of top-N reranker scores. Objective signal."""
    if not reranker_scores:
        return 0.0
    scores = sorted(reranker_scores, reverse=True)[:top_n]
    return sum(scores) / len(scores)


def parse_llm_confidence(llm_confidence_text: str) -> float:
    """Map LLM self-reported confidence to numerical score with downward correction."""
    return LLM_CONFIDENCE_MAP.get(llm_confidence_text.strip().lower(), 0.5)


def compute_citation_coverage(answer: str, retrieved_quotes: list[str]) -> float:
    """Fraction of answer sentences grounded in retrieved content."""
    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0

    covered = 0
    for sentence in sentences:
        sentence_lower = sentence.lower().strip()
        if len(sentence_lower) < 10:
            # Skip very short sentences (e.g., "Yes.", "No.")
            covered += 1
            continue
        for quote in retrieved_quotes:
            if _has_substantial_overlap(sentence_lower, quote.lower()):
                covered += 1
                break

    return covered / len(sentences)


def compute_composite_confidence(
    reranker_scores: list[float],
    llm_confidence_text: str,
    answer: str,
    retrieved_quotes: list[str],
    retrieval_weight: float = 0.50,
    llm_weight: float = 0.25,
    citation_weight: float = 0.25,
) -> ConfidenceBreakdown:
    """Compute composite confidence from 3 independent signals."""
    retrieval = compute_retrieval_confidence(reranker_scores)
    llm = parse_llm_confidence(llm_confidence_text)
    citation = compute_citation_coverage(answer, retrieved_quotes)

    composite = (
        retrieval * retrieval_weight +
        llm * llm_weight +
        citation * citation_weight
    )

    return ConfidenceBreakdown(
        retrieval_score=round(retrieval, 3),
        llm_score=round(llm, 3),
        citation_score=round(citation, 3),
        composite=round(composite, 3),
        retrieval_weight=retrieval_weight,
        llm_weight=llm_weight,
        citation_weight=citation_weight,
    )


def _split_sentences(text: str) -> list[str]:
    """Basic sentence splitting. Replace with spaCy/nltk for production."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]


def _has_substantial_overlap(sentence: str, quote: str, threshold: int = 5) -> bool:
    """Check if sentence shares at least `threshold` consecutive words with a quote."""
    sentence_words = sentence.split()
    quote_words = quote.split()
    for i in range(len(sentence_words) - threshold + 1):
        ngram = " ".join(sentence_words[i:i + threshold])
        if ngram in " ".join(quote_words):
            return True
    return False
```

---

## B.4 Post-Generation Guardrail

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import re


class PostGuardrailAction(Enum):
    RETURN = "return"             # Answer is good — deliver to user
    RE_RETRIEVE = "re_retrieve"  # Low confidence — try again with broader params
    FLAG = "flag"                 # Escalate to human review
    BLOCK = "block"              # Do not return answer


@dataclass
class PostGuardrailResult:
    action: PostGuardrailAction
    answer: str                              # Possibly redacted/sanitized
    confidence: ConfidenceBreakdown
    risk_level: RiskLevel
    pii_redactions: list[dict] = field(default_factory=list)
    hallucination_flags: list[str] = field(default_factory=list)
    verification_warning: Optional[str] = None


class PostGenerationGuardrail:
    def __init__(self, config_path: str = "config/guardrails.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        post_config = config.get("post_generation", {})
        self.high_confidence_threshold = post_config.get("high_confidence_threshold", 0.70)
        self.low_confidence_threshold = post_config.get("low_confidence_threshold", 0.50)
        self.system_prompt_fragments = self._load_system_prompt_fragments(
            post_config.get("system_prompt_path", "prompts/rag_system.md")
        )

    def evaluate(
        self,
        answer: str,
        confidence: ConfidenceBreakdown,
        risk_level: RiskLevel,
        retrieved_docs: list[dict],
        retry_count: int = 0,
    ) -> PostGuardrailResult:
        # 1. PII filtering (always runs)
        answer, pii_redactions = self._filter_pii(answer)

        # 2. Output sanitization
        answer = self._sanitize_output(answer)

        # 3. Hallucination detection
        hallucination_flags = self._detect_hallucination(answer, retrieved_docs, risk_level)

        # 4. Risk-based output filtering (HIGH risk only)
        verification_warning = None
        if risk_level == RiskLevel.HIGH:
            answer, verification_warning = self._apply_risk_filtering(
                answer, retrieved_docs
            )

        # 5. Confidence routing
        action = self._route(confidence.composite, risk_level, retry_count)

        return PostGuardrailResult(
            action=action,
            answer=answer,
            confidence=confidence,
            risk_level=risk_level,
            pii_redactions=pii_redactions,
            hallucination_flags=hallucination_flags,
            verification_warning=verification_warning,
        )

    def _route(
        self, composite_score: float, risk_level: RiskLevel, retry_count: int
    ) -> PostGuardrailAction:
        if composite_score < self.low_confidence_threshold:
            if retry_count < 1:
                return PostGuardrailAction.RE_RETRIEVE
            return PostGuardrailAction.BLOCK

        if composite_score < self.high_confidence_threshold:
            if retry_count < 1:
                return PostGuardrailAction.RE_RETRIEVE
            return PostGuardrailAction.FLAG

        # composite >= high threshold
        if risk_level == RiskLevel.HIGH:
            return PostGuardrailAction.RETURN  # With verification warning attached
        return PostGuardrailAction.RETURN

    def _filter_pii(self, text: str) -> tuple[str, list[dict]]:
        redactions = []
        # Email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            redactions.append({"type": "EMAIL", "position": match.span()})
        text = re.sub(email_pattern, "[EMAIL]", text)

        # Phone
        phone_pattern = r'\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b'
        for match in re.finditer(phone_pattern, text):
            redactions.append({"type": "PHONE", "position": match.span()})
        text = re.sub(phone_pattern, "[PHONE]", text)

        # Person names would use NER here — placeholder
        # names = ner_model.detect_persons(text)
        # for name in names:
        #     redactions.append({"type": "PERSON", ...})
        #     text = text.replace(name, "[PERSON]")

        return text, redactions

    def _sanitize_output(self, answer: str) -> str:
        # Remove any system prompt fragments that leaked
        for fragment in self.system_prompt_fragments:
            if fragment in answer:
                answer = answer.replace(fragment, "")

        # Remove internal document markers
        answer = re.sub(r'---\s*Document\s*\d+\s*---', '', answer)

        # Remove template variable artifacts
        answer = re.sub(r'\{(?:documents|question|context)\}', '', answer)

        return answer.strip()

    def _detect_hallucination(
        self, answer: str, retrieved_docs: list[dict], risk_level: RiskLevel
    ) -> list[str]:
        flags = []
        sentences = _split_sentences(answer)
        doc_texts = [doc.get("text", "") for doc in retrieved_docs]

        for sentence in sentences:
            if len(sentence.strip()) < 10:
                continue
            grounded = any(
                _has_substantial_overlap(sentence.lower(), doc.lower())
                for doc in doc_texts
            )
            if not grounded:
                flags.append(sentence)

        return flags

    def _apply_risk_filtering(
        self, answer: str, retrieved_docs: list[dict]
    ) -> tuple[str, str]:
        """For HIGH risk: flag numerical claims not found in source docs."""
        warning = "VERIFY BEFORE IMPLEMENTATION — This answer addresses a high-risk domain."

        # Find numerical values in the answer
        numerical_pattern = r'\b\d+\.?\d*\s*(?:V|mV|A|mA|MHz|GHz|kHz|ns|ps|us|ms|°C|K)\b'
        numbers_in_answer = re.findall(numerical_pattern, answer, re.IGNORECASE)

        doc_text = " ".join(doc.get("text", "") for doc in retrieved_docs)
        for num in numbers_in_answer:
            if num not in doc_text:
                warning += f"\n  - Value '{num}' not found verbatim in source documents."

        return answer, warning

    def _load_system_prompt_fragments(self, path: str) -> list[str]:
        """Load system prompt and extract key fragments for leak detection."""
        try:
            with open(path) as f:
                content = f.read()
            # Extract distinctive phrases (not common words)
            lines = [
                line.strip() for line in content.split("\n")
                if len(line.strip()) > 30
            ]
            return lines[:20]  # Top 20 distinctive lines
        except FileNotFoundError:
            return []
```

---

## B.5 Version Conflict Detection

```python
from dataclasses import dataclass
from typing import Optional
import re


@dataclass
class VersionConflict:
    spec_identifier: str           # Spec ID or filename stem
    version_a: str
    date_a: str
    version_b: str
    date_b: str


def detect_version_conflicts(documents: list[dict]) -> list[VersionConflict]:
    """Detect when retrieved docs share a spec ID / filename stem but differ in version."""
    version_map: dict[str, dict] = {}
    conflicts: list[VersionConflict] = []

    for doc in documents:
        metadata = doc.get("metadata", {})

        # Prefer spec_id; fall back to filename stem
        key = metadata.get("spec_id") or _extract_filename_stem(
            metadata.get("filename", "")
        )
        if not key:
            continue

        version = metadata.get("version", "unknown")
        date = metadata.get("date", "unknown")

        if key in version_map:
            existing = version_map[key]
            if existing["version"] != version:
                conflicts.append(VersionConflict(
                    spec_identifier=key,
                    version_a=existing["version"],
                    date_a=existing["date"],
                    version_b=version,
                    date_b=date,
                ))
        else:
            version_map[key] = {"version": version, "date": date}

    return conflicts


def format_conflict_warning(conflicts: list[VersionConflict]) -> str:
    """Format conflicts for injection into LLM context."""
    if not conflicts:
        return ""

    lines = ["WARNING — Version conflicts detected in retrieved documents:"]
    for c in conflicts:
        lines.append(
            f"  - {c.spec_identifier}: {c.version_a} ({c.date_a}) vs "
            f"{c.version_b} ({c.date_b})"
        )
    lines.append(
        "You MUST note this conflict in your answer. "
        "Do NOT silently choose one version over another."
    )
    return "\n".join(lines)


def _extract_filename_stem(filename: str) -> Optional[str]:
    """Extract base name without version suffix. E.g., 'Power_Spec_v3.pdf' → 'Power_Spec'."""
    if not filename:
        return None
    # Remove extension
    name = re.sub(r'\.[^.]+$', '', filename)
    # Remove version suffix patterns (_v3, _v2.1, _rev3, etc.)
    name = re.sub(r'[_-]?v\d+(\.\d+)?$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[_-]?rev\d+$', '', name, flags=re.IGNORECASE)
    return name if name else None
```

---

## B.6 Structured Document Formatter

```python
def format_retrieved_docs(
    documents: list[dict],
    conflicts: list[VersionConflict],
) -> str:
    """Format retrieved documents with structured metadata for LLM context injection."""
    formatted_chunks = []

    for i, doc in enumerate(documents, 1):
        metadata = doc.get("metadata", {})
        chunk = f"""--- Document {i} ---
Filename    : {metadata.get('filename', 'unknown')}
Version     : {metadata.get('version', 'unknown')}
Date        : {metadata.get('date', 'unknown')}
Domain      : {metadata.get('domain', 'unknown')}
Section     : {metadata.get('section', 'unknown')}
Spec ID     : {metadata.get('spec_id', 'N/A')}
Content:
{doc.get('text', '')}
"""
        formatted_chunks.append(chunk)

    # Prepend version conflict warnings if any
    conflict_warning = format_conflict_warning(conflicts)

    parts = []
    if conflict_warning:
        parts.append(conflict_warning)
    parts.append("\n".join(formatted_chunks))

    return "\n".join(parts)
```

---

## B.7 Retry Logic Wrapper

```python
import time
import logging
from functools import wraps
from typing import TypeVar, Callable, Optional

T = TypeVar("T")
logger = logging.getLogger(__name__)


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    fallback: Optional[Callable[..., T]] = None,
    exceptions: tuple = (Exception,),
):
    """Decorator: retry with exponential backoff. Falls back if all retries fail."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(
                            base_delay * (backoff_factor ** attempt),
                            max_delay
                        )
                        logger.warning(
                            f"{func.__name__} attempt {attempt + 1}/{max_retries + 1} "
                            f"failed: {e}. Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} exhausted all {max_retries + 1} attempts. "
                            f"Last error: {e}"
                        )

            # All retries exhausted — use fallback or raise
            if fallback is not None:
                logger.info(f"{func.__name__} using fallback.")
                return fallback(*args, **kwargs)
            raise last_exception

        return wrapper
    return decorator


# Usage example:
#
# def _heuristic_confidence(query: str) -> float:
#     word_count = len(query.split())
#     if word_count <= 2: return 0.4
#     if word_count <= 5: return 0.7
#     return 0.85
#
# @with_retry(max_retries=3, base_delay=1.0, fallback=_heuristic_confidence)
# def evaluate_query_confidence(query: str) -> float:
#     return call_ollama_for_confidence(query)
```

---

## B.8 Embedding Cache

```python
from functools import lru_cache
from typing import Protocol


class EmbeddingModel(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


class CachedEmbeddings:
    """LRU-cached wrapper around any embedding model."""

    def __init__(self, model: EmbeddingModel, cache_size: int = 256):
        self._model = model
        self._cache_size = cache_size
        # Create a bound cached function
        self._cached_embed = lru_cache(maxsize=cache_size)(self._embed)

    def embed_query(self, text: str) -> list[float]:
        # Normalize whitespace for consistent cache keys
        normalized = " ".join(text.split())
        return self._cached_embed(normalized)

    def _embed(self, text: str) -> tuple[float, ...]:
        """Internal — returns tuple for hashability (lru_cache requirement)."""
        result = self._model.embed_query(text)
        return tuple(result)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Documents are not cached — typically one-time ingestion."""
        return self._model.embed_documents(texts)

    @property
    def cache_info(self):
        return self._cached_embed.cache_info()

    def clear_cache(self):
        self._cached_embed.cache_clear()
```

---

## B.9 Query Result Cache

```python
import time
import hashlib
from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class CacheEntry:
    response: Any
    timestamp: float
    query_key: str


class QueryResultCache:
    """TTL-based cache for full pipeline responses."""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 100):
        self._cache: dict[str, CacheEntry] = {}
        self._ttl = ttl_seconds
        self._max_size = max_size

    def get(
        self,
        processed_query: str,
        source_filter: Optional[str] = None,
        heading_filter: Optional[str] = None,
        alpha: float = 0.5,
    ) -> Optional[Any]:
        key = self._make_key(processed_query, source_filter, heading_filter, alpha)
        entry = self._cache.get(key)

        if entry is None:
            return None

        if time.time() - entry.timestamp > self._ttl:
            del self._cache[key]
            return None

        return entry.response

    def put(
        self,
        response: Any,
        processed_query: str,
        source_filter: Optional[str] = None,
        heading_filter: Optional[str] = None,
        alpha: float = 0.5,
    ) -> None:
        # Evict oldest if at capacity
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._cache, key=lambda k: self._cache[k].timestamp)
            del self._cache[oldest_key]

        key = self._make_key(processed_query, source_filter, heading_filter, alpha)
        self._cache[key] = CacheEntry(
            response=response,
            timestamp=time.time(),
            query_key=key,
        )

    def _make_key(
        self,
        query: str,
        source_filter: Optional[str],
        heading_filter: Optional[str],
        alpha: float,
    ) -> str:
        normalized_query = " ".join(query.lower().split())
        raw = f"{normalized_query}|{source_filter}|{heading_filter}|{alpha}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def clear(self) -> None:
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)
```

---

## B.10 Connection Pool Manager

```python
import logging
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class VectorDBPool:
    """Persistent connection pool for the vector database."""

    def __init__(
        self,
        db_url: Optional[str] = None,
        data_path: str = ".weaviate_data",
        health_check_interval: int = 60,
    ):
        self._db_url = db_url
        self._data_path = data_path
        self._client = None
        self._health_check_interval = health_check_interval

    def connect(self) -> None:
        """Initialize the connection. Call once at startup."""
        if self._db_url:
            # External vector DB
            self._client = self._connect_external(self._db_url)
        else:
            # Embedded vector DB
            self._client = self._connect_embedded(self._data_path)

        # Fail-fast health check
        if not self._health_check():
            raise ConnectionError("Vector database health check failed on startup.")

        logger.info("Vector database connection established.")

    def get_client(self):
        """Return the persistent client. Reconnect if needed."""
        if self._client is None:
            self.connect()

        if not self._health_check():
            logger.warning("Vector database connection lost. Reconnecting...")
            self.connect()

        return self._client

    def close(self) -> None:
        """Close the connection pool."""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("Vector database connection closed.")

    def _health_check(self) -> bool:
        """Check if the vector database is reachable."""
        try:
            self._client.is_ready()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def _connect_external(self, url: str):
        """Connect to an external vector database instance."""
        import weaviate
        return weaviate.connect_to_custom(
            http_host=url.split("://")[1].split(":")[0],
            http_port=int(url.split(":")[-1]),
            http_secure=url.startswith("https"),
        )

    def _connect_embedded(self, data_path: str):
        """Connect to an embedded vector database instance."""
        import weaviate
        return weaviate.connect_to_embedded(
            persistence_data_path=data_path
        )
```

---

## B.11 PromptTemplate Integration

```python
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path


def load_rag_prompt(
    system_prompt_path: str = "prompts/rag_system.md",
    human_template_path: str = "prompts/rag_human.md",
) -> ChatPromptTemplate:
    """
    Load prompt from markdown files.

    System prompt: static rules, no variables.
    Human template: contains {documents} and {question} placeholders.

    LangChain's ChatPromptTemplate handles curly braces in document content
    safely — only explicitly declared variables are substituted.
    """
    system_prompt = Path(system_prompt_path).read_text()
    human_template = Path(human_template_path).read_text()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_template),
    ])

    return prompt


# Usage:
#
# prompt = load_rag_prompt()
# chain = prompt | llm | output_parser
#
# response = chain.invoke({
#     "documents": format_retrieved_docs(ranked_docs, conflicts),
#     "question": user_query,
# })
```

---

## B.12 Full-Pipeline LangGraph Definition

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional


class RAGPipelineState(TypedDict):
    # Input
    question: str
    conversation_history: list[dict]

    # Query processing
    processed_query: str
    query_confidence: float
    query_action: str                  # "search" or "ask_user"
    clarification_message: Optional[str]

    # Pre-retrieval guardrail
    risk_level: str                    # "HIGH", "MEDIUM", "LOW"
    sanitized_query: str
    guardrail_passed: bool

    # Retrieval
    retrieved_docs: list[dict]
    kg_expanded_terms: list[str]

    # Reranking
    ranked_docs: list[dict]
    reranker_scores: list[float]

    # Document formatting
    formatted_context: str
    version_conflicts: list[dict]

    # Generation
    answer: str
    llm_confidence: str                # "high", "medium", "low"

    # Post-generation guardrail
    composite_confidence: float
    confidence_breakdown: dict
    post_guardrail_action: str         # "return", "re_retrieve", "flag", "block"
    final_answer: str
    verification_warning: Optional[str]

    # Control
    retry_count: int


def build_rag_pipeline() -> StateGraph:
    graph = StateGraph(RAGPipelineState)

    # Define nodes
    graph.add_node("process_query",           process_query_node)
    graph.add_node("pre_retrieval_guardrail",  pre_guardrail_node)
    graph.add_node("retrieve",                 retrieve_node)
    graph.add_node("rerank",                   rerank_node)
    graph.add_node("format_docs",              format_docs_node)
    graph.add_node("generate",                 generate_node)
    graph.add_node("post_generation_guardrail", post_guardrail_node)
    graph.add_node("return_answer",            return_answer_node)
    graph.add_node("ask_user",                 ask_user_node)
    graph.add_node("block_answer",             block_answer_node)
    graph.add_node("flag_for_review",          flag_for_review_node)

    # Entry point
    graph.set_entry_point("process_query")

    # Query processing → route on action
    graph.add_conditional_edges("process_query", route_after_query_processing, {
        "search": "pre_retrieval_guardrail",
        "ask_user": "ask_user",
    })

    # Pre-retrieval guardrail → route on pass/reject
    graph.add_conditional_edges("pre_retrieval_guardrail", route_after_guardrail, {
        "pass": "retrieve",
        "reject": "block_answer",
    })

    # Linear pipeline: retrieve → rerank → format → generate → evaluate
    graph.add_edge("retrieve",    "rerank")
    graph.add_edge("rerank",      "format_docs")
    graph.add_edge("format_docs", "generate")
    graph.add_edge("generate",    "post_generation_guardrail")

    # Post-generation guardrail → route on confidence
    graph.add_conditional_edges("post_generation_guardrail", route_after_post_guardrail, {
        "return":       "return_answer",
        "re_retrieve":  "retrieve",        # Loop back with broader params
        "flag":         "flag_for_review",
        "block":        "block_answer",
    })

    # Terminal nodes
    graph.add_edge("return_answer",    END)
    graph.add_edge("ask_user",         END)
    graph.add_edge("block_answer",     END)
    graph.add_edge("flag_for_review",  END)

    return graph.compile()


# --- Routing functions ---

def route_after_query_processing(state: RAGPipelineState) -> str:
    return state["query_action"]


def route_after_guardrail(state: RAGPipelineState) -> str:
    return "pass" if state["guardrail_passed"] else "reject"


def route_after_post_guardrail(state: RAGPipelineState) -> str:
    return state["post_guardrail_action"]


# --- Node implementations (signatures) ---

def process_query_node(state: RAGPipelineState) -> dict:
    """Reformulate query, score confidence, resolve multi-turn references."""
    # Implementation: use existing query processor + conversation history
    ...

def pre_guardrail_node(state: RAGPipelineState) -> dict:
    """Run pre-retrieval guardrail: validate, detect injection, classify risk, PII filter."""
    # Implementation: see B.1 PreRetrievalGuardrail
    ...

def retrieve_node(state: RAGPipelineState) -> dict:
    """Execute hybrid search (vector + BM25) with optional KG expansion."""
    # On retry: broaden params (increase search_limit, shift alpha toward BM25)
    search_limit = 10 if state["retry_count"] == 0 else 20
    alpha = 0.5 if state["retry_count"] == 0 else 0.3
    ...

def rerank_node(state: RAGPipelineState) -> dict:
    """Cross-encoder reranking with score thresholds."""
    # Filter out documents below minimum score threshold (0.30)
    ...

def format_docs_node(state: RAGPipelineState) -> dict:
    """Attach structured metadata, detect version conflicts, format context."""
    # Implementation: see B.5 and B.6
    ...

def generate_node(state: RAGPipelineState) -> dict:
    """Generate answer using anti-hallucination prompt with retry logic."""
    # Implementation: see B.7 retry wrapper + B.11 PromptTemplate
    ...

def post_guardrail_node(state: RAGPipelineState) -> dict:
    """Compute confidence, detect hallucination, filter PII, route."""
    # Implementation: see B.3 and B.4
    ...

def return_answer_node(state: RAGPipelineState) -> dict:
    """Package final answer with sources, confidence, and risk metadata."""
    ...

def ask_user_node(state: RAGPipelineState) -> dict:
    """Return clarification request to user."""
    ...

def block_answer_node(state: RAGPipelineState) -> dict:
    """Return 'Insufficient documentation found' message."""
    ...

def flag_for_review_node(state: RAGPipelineState) -> dict:
    """Escalate to human review queue."""
    ...
```

---

## B.13 Observability Wrapper

```python
import time
import uuid
import logging
from functools import wraps
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StageMetrics:
    stage_name: str
    latency_ms: float
    metadata: dict = field(default_factory=dict)


@dataclass
class QueryTrace:
    trace_id: str
    risk_level: str
    stages: list[StageMetrics] = field(default_factory=list)
    total_latency_ms: float = 0.0

    def add_stage(self, stage: StageMetrics):
        self.stages.append(stage)
        self.total_latency_ms = sum(s.latency_ms for s in self.stages)


# Global trace context (per-request — use contextvars in async)
_current_trace: QueryTrace | None = None


def start_trace(risk_level: str = "LOW") -> QueryTrace:
    global _current_trace
    _current_trace = QueryTrace(
        trace_id=str(uuid.uuid4()),
        risk_level=risk_level,
    )
    return _current_trace


def get_current_trace() -> QueryTrace | None:
    return _current_trace


def traced(stage_name: str):
    """Decorator to capture timing and metadata for a pipeline stage."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000

            trace = get_current_trace()
            if trace:
                # Extract metrics from result if it's a dict
                metadata = {}
                if isinstance(result, dict):
                    metadata = {
                        k: v for k, v in result.items()
                        if isinstance(v, (int, float, str, bool))
                    }

                trace.add_stage(StageMetrics(
                    stage_name=stage_name,
                    latency_ms=round(elapsed_ms, 2),
                    metadata=metadata,
                ))

                logger.info(
                    f"[{trace.trace_id}] {stage_name}: {elapsed_ms:.1f}ms",
                    extra={"trace_id": trace.trace_id, "stage": stage_name},
                )

            return result
        return wrapper
    return decorator


# Usage:
#
# @traced("retrieval")
# def retrieve_node(state):
#     results = hybrid_search(...)
#     return {"retrieved_docs": results, "result_count": len(results)}
#
# The decorator automatically captures:
# - Latency (elapsed time)
# - Metadata (result_count and other scalar values from the return dict)
# - Logs with trace ID for correlation
```

---

## B.14 Multi-Turn Conversation State

```python
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConversationTurn:
    query: str
    processed_query: str
    answer: Optional[str]


@dataclass
class ConversationState:
    turns: list[ConversationTurn] = field(default_factory=list)
    max_turns: int = 5

    def add_turn(self, query: str, processed_query: str, answer: Optional[str] = None):
        self.turns.append(ConversationTurn(
            query=query,
            processed_query=processed_query,
            answer=answer,
        ))
        # Keep only last N turns
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def get_context_for_reformulation(self) -> str:
        """Format recent turns as context for the query reformulator."""
        if not self.turns:
            return ""

        lines = ["Recent conversation context:"]
        for i, turn in enumerate(self.turns[-3:], 1):  # Last 3 turns
            lines.append(f"  Turn {i}:")
            lines.append(f"    User: {turn.query}")
            if turn.answer:
                # Truncate long answers
                answer_preview = turn.answer[:200] + "..." if len(turn.answer) > 200 else turn.answer
                lines.append(f"    Assistant: {answer_preview}")

        return "\n".join(lines)

    def resolve_coreferences(self, query: str) -> str:
        """
        Basic coreference resolution against recent conversation context.

        Detects pronouns and references that likely refer to prior turns.
        For production, consider using a dedicated coreference resolution model.
        """
        if not self.turns:
            return query

        # Patterns that suggest a follow-up question
        followup_indicators = [
            "tell me more",
            "what about",
            "how about",
            "and the",
            "what else",
            "can you elaborate",
            "more details",
            "expand on",
        ]

        query_lower = query.lower()
        is_followup = any(indicator in query_lower for indicator in followup_indicators)

        # Check for pronoun-heavy queries with little context
        pronoun_heavy = query_lower.strip().startswith(("it ", "its ", "that ", "this ", "they ", "those "))

        if is_followup or pronoun_heavy:
            # Prepend context from the last turn
            last_turn = self.turns[-1]
            context_prefix = f"Regarding '{last_turn.processed_query}': "
            return context_prefix + query

        return query
```
