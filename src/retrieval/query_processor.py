# @summary
# One-sentence description of what the file does:
# A retrieval-based query processing module using LangGraph for confidence-based routing with user-friendly exports and dependency management.
#
# Key exports and dependencies:
# Exports process_query, QueryResult, QueryAction; relies on logging, json, re, state_graph, _COMPILED_GRAPH.
# @end-summary
"""
LangGraph-based query processing pipeline.

Implements a creation/verification loop:
  - Sanitize: clean input, check guardrails (injection, length)
  - Reformulate (creation agent): LLM reformulates query for searchability
  - Evaluate (verification agent): LLM scores query confidence with reasoning
  - Route: confident → SEARCH, exhausted → ASK_USER, else → loop

Falls back to word-count heuristic if Ollama is unavailable.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
from typing import Dict, List, Optional, TypedDict
from urllib.error import URLError
from urllib.request import Request, urlopen

from langgraph.graph import END, StateGraph

from config.settings import (
    DOMAIN_DESCRIPTION,
    KG_PATH,
    MAX_SANITIZATION_ITERATIONS,
    OLLAMA_BASE_URL,
    PROMPTS_DIR,
    QUERY_CONFIDENCE_THRESHOLD,
    QUERY_LOG_DIR,
    QUERY_MAX_LENGTH,
    QUERY_PROCESSING_MODEL,
    QUERY_PROCESSING_TEMPERATURE,
    QUERY_PROCESSING_TIMEOUT,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("rag.query_processor")

if not logger.handlers:
    logger.setLevel(logging.INFO)
    os.makedirs(QUERY_LOG_DIR, exist_ok=True)
    _file_handler = logging.FileHandler(QUERY_LOG_DIR / "query_processor.log")
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(_file_handler)


# ---------------------------------------------------------------------------
# Public API — backward-compatible with rag_chain.py
# ---------------------------------------------------------------------------


class QueryAction(Enum):
    """Action to take after query processing."""

    SEARCH = "search"
    ASK_USER = "ask_user"


@dataclass
class QueryResult:
    """Result of the query processing pipeline."""

    processed_query: str
    confidence: float
    action: QueryAction
    clarification_message: Optional[str] = None
    iterations: int = 0


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------


class QueryState(TypedDict):
    original_query: str
    current_query: str
    confidence: float
    reasoning: str
    iteration: int
    max_iterations: int
    confidence_threshold: float
    action: str  # "search" | "ask_user" | ""
    clarification_message: str
    ollama_available: bool


# ---------------------------------------------------------------------------
# Prompt loading (cached at module level)
# ---------------------------------------------------------------------------


def _load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


_REFORMULATOR_PROMPT: Optional[str] = None
_EVALUATOR_PROMPT: Optional[str] = None


def _get_reformulator_prompt() -> str:
    global _REFORMULATOR_PROMPT
    if _REFORMULATOR_PROMPT is None:
        _REFORMULATOR_PROMPT = _load_prompt("query_reformulator.md")
    return _REFORMULATOR_PROMPT


def _get_evaluator_prompt() -> str:
    global _EVALUATOR_PROMPT
    if _EVALUATOR_PROMPT is None:
        _EVALUATOR_PROMPT = _load_prompt("query_evaluator.md")
    return _EVALUATOR_PROMPT


# ---------------------------------------------------------------------------
# Knowledge graph vocabulary (loaded once, used for reformulation context)
# ---------------------------------------------------------------------------

_KG_TERMS: Optional[List[str]] = None
_KG_WORD_INDEX: Optional[Dict[str, List[str]]] = None


def _get_kg_terms() -> tuple:
    """Load entity names from the knowledge graph JSON (if available).

    Returns (terms_list, word_index) where word_index maps lowercase words
    to the terms containing them. Both are built once and cached.
    """
    global _KG_TERMS, _KG_WORD_INDEX
    if _KG_TERMS is not None:
        return _KG_TERMS, _KG_WORD_INDEX

    _KG_TERMS = []
    _KG_WORD_INDEX = defaultdict(list)

    if not KG_PATH.exists():
        return _KG_TERMS, _KG_WORD_INDEX

    try:
        with open(KG_PATH, encoding="utf-8") as f:
            kg_data = json.load(f)
        nodes = kg_data.get("nodes", [])
        # Sort by mention count, filter out noisy short/long entries
        valid = [
            n for n in nodes
            if 2 <= len(n.get("id", "")) <= 60 and n.get("mention_count", 0) >= 1
        ]
        valid.sort(key=lambda n: n.get("mention_count", 0), reverse=True)
        _KG_TERMS = [n["id"] for n in valid]

        # Build inverted index: word -> [term1, term2, ...]
        for term in _KG_TERMS:
            for word in term.lower().split():
                if len(word) >= 3:
                    _KG_WORD_INDEX[word].append(term)

        logger.info(
            "Loaded %d KG terms (%d index keys) for reformulation context",
            len(_KG_TERMS), len(_KG_WORD_INDEX),
        )
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Failed to load KG terms: %s", e)

    return _KG_TERMS, _KG_WORD_INDEX


# ---------------------------------------------------------------------------
# Guardrails — prompt injection detection
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    re.compile(
        r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts)", re.I
    ),
    re.compile(r"you\s+are\s+now\s+", re.I),
    re.compile(r"system\s*:\s*", re.I),
    re.compile(r"<\s*/?\s*(system|prompt|instruction)", re.I),
    re.compile(r"\b(ADMIN|ROOT)\s*:", re.I),
    re.compile(r"do\s+not\s+follow\s+(your|the)\s+(rules|instructions)", re.I),
    re.compile(r"forget\s+(everything|your\s+instructions)", re.I),
    re.compile(r"\[\s*INST\s*\]", re.I),
    re.compile(r"```\s*(system|instruction)", re.I),
]


def _detect_injection(query: str) -> bool:
    """Check for common prompt injection patterns."""
    return any(p.search(query) for p in _INJECTION_PATTERNS)


# ---------------------------------------------------------------------------
# Ollama helper (matches generator.py pattern — raw urllib)
# ---------------------------------------------------------------------------


def _call_ollama(prompt: str, system: str = "") -> Optional[str]:
    """Call Ollama chat API. Returns response text or None on failure."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": QUERY_PROCESSING_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": QUERY_PROCESSING_TEMPERATURE,
            "num_predict": 256,
        },
    }
    try:
        req = Request(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=QUERY_PROCESSING_TIMEOUT) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("message", {}).get("content")
    except (URLError, json.JSONDecodeError, KeyError) as e:
        logger.warning("Ollama call failed: %s", e)
        return None


def _check_ollama_available() -> bool:
    """Check if Ollama API is reachable."""
    try:
        req = Request(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", method="GET")
        with urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fallback heuristic (preserves old behavior when Ollama is unavailable)
# ---------------------------------------------------------------------------


def _heuristic_confidence(query: str) -> float:
    """Word-count-based confidence heuristic."""
    word_count = len(query.split())
    if word_count == 0:
        return 0.0
    elif word_count <= 2:
        return 0.4
    elif word_count <= 5:
        return 0.7
    else:
        return 0.85


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def sanitize_node(state: QueryState) -> dict:
    """Clean input and check guardrails."""
    query = state["current_query"].strip()
    query = " ".join(query.split())  # collapse whitespace

    # Empty check
    if not query:
        logger.info("Query rejected: empty")
        return {
            "current_query": query,
            "confidence": 0.0,
            "action": "ask_user",
            "clarification_message": (
                "Your query appears to be empty. Please enter a question."
            ),
        }

    # Length check
    if len(query) > QUERY_MAX_LENGTH:
        query = query[:QUERY_MAX_LENGTH]
        logger.info("Query truncated to %d characters", QUERY_MAX_LENGTH)

    # Injection detection
    if _detect_injection(query):
        logger.warning("Potential prompt injection detected: %s", query[:80])
        return {
            "current_query": query,
            "confidence": 0.0,
            "action": "ask_user",
            "clarification_message": (
                "Your query could not be processed. Please rephrase."
            ),
        }

    return {"current_query": query}


def _match_kg_terms(query: str, max_terms: int = 20) -> str:
    """Find KG terms relevant to the query using inverted index lookup.

    Returns a formatted string for injection into the reformulator prompt.
    Uses pre-built word→terms index for O(query_words) lookup instead of
    scanning all terms. Scales to 10k+ KG nodes with <1ms lookup.
    """
    kg_terms, word_index = _get_kg_terms()
    if not kg_terms:
        return ""

    query_words = {w.lower() for w in query.split() if len(w) >= 3}
    if not query_words:
        return ""

    # Collect candidate terms from index, preserving mention-count order
    seen = set()
    matched = []
    for word in query_words:
        for term in word_index.get(word, []):
            if term not in seen:
                seen.add(term)
                matched.append(term)
                if len(matched) >= max_terms:
                    break
        if len(matched) >= max_terms:
            break

    if not matched:
        # No direct matches — fall back to top terms by mention count
        matched = kg_terms[:max_terms]

    return "Known terms in the knowledge base: " + ", ".join(matched)


def reformulate_node(state: QueryState) -> dict:
    """LLM-based query reformulation (creation agent)."""
    new_iteration = state["iteration"] + 1

    if not state["ollama_available"]:
        return {"iteration": new_iteration}

    previous_feedback = ""
    if state["reasoning"]:
        previous_feedback = (
            f"Previous evaluator feedback: {state['reasoning']}"
        )

    kg_terms = _match_kg_terms(state["original_query"])

    prompt = _get_reformulator_prompt().format(
        original_query=state["original_query"],
        iteration=new_iteration,
        max_iterations=state["max_iterations"],
        previous_feedback=previous_feedback,
        kg_terms=kg_terms,
        domain_description=DOMAIN_DESCRIPTION,
    )

    result = _call_ollama(prompt)

    if result:
        reformulated = result.strip().strip('"').strip("'")
        # Avoid empty reformulations
        if reformulated:
            logger.info(
                "Iteration %d: reformulated '%s' -> '%s'",
                new_iteration,
                state["current_query"],
                reformulated,
            )
            return {"current_query": reformulated, "iteration": new_iteration}

    logger.warning("Iteration %d: reformulation failed, keeping current query", new_iteration)
    return {"iteration": new_iteration}


def evaluate_node(state: QueryState) -> dict:
    """LLM-based query quality evaluation (verification agent)."""
    if not state["ollama_available"]:
        conf = _heuristic_confidence(state["current_query"])
        logger.info(
            "Iteration %d: heuristic confidence=%.2f (Ollama unavailable)",
            state["iteration"],
            conf,
        )
        return {
            "confidence": conf,
            "reasoning": "fallback heuristic (Ollama unavailable)",
        }

    prompt = _get_evaluator_prompt().format(query=state["current_query"])
    result = _call_ollama(prompt)

    if result:
        try:
            # Strip markdown code fences if present
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", result.strip())
            parsed = json.loads(cleaned)
            confidence = float(parsed.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
            reasoning = str(parsed.get("reasoning", ""))
            logger.info(
                "Iteration %d: confidence=%.2f reasoning='%s'",
                state["iteration"],
                confidence,
                reasoning,
            )
            return {"confidence": confidence, "reasoning": reasoning}
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(
                "Failed to parse evaluator JSON: %s. Raw: %s", e, result[:200]
            )

    # Fallback on parse failure
    logger.warning("Iteration %d: evaluation failed, using default confidence", state["iteration"])
    return {"confidence": 0.5, "reasoning": "evaluation parse failed, using default"}


def exhaust_node(state: QueryState) -> dict:
    """Generate clarification message when iterations are exhausted."""
    query = state["current_query"]
    confidence = state["confidence"]

    if len(query.split()) <= 2:
        msg = (
            f'Your query "{query}" seems quite short. '
            "Could you provide more context or detail about what you're looking for?"
        )
    else:
        msg = (
            f'I\'m not fully confident I understand your query "{query}" '
            f"(confidence: {confidence:.0%}). Could you rephrase or add more detail?"
        )

    logger.info("Iterations exhausted. Action: ask_user. Confidence: %.2f", confidence)
    return {"action": "ask_user", "clarification_message": msg}


# ---------------------------------------------------------------------------
# Routers (conditional edges)
# ---------------------------------------------------------------------------


def _route_after_sanitize(state: QueryState) -> str:
    if state.get("action") == "ask_user":
        return "end"
    return "reformulate"


def _route_after_evaluate(state: QueryState) -> str:
    if state["confidence"] >= state["confidence_threshold"]:
        return "end"
    if state["iteration"] >= state["max_iterations"]:
        return "exhaust"
    return "reformulate"


# ---------------------------------------------------------------------------
# Graph assembly (compiled once at module level)
# ---------------------------------------------------------------------------


def _build_graph():
    graph = StateGraph(QueryState)

    graph.add_node("sanitize", sanitize_node)
    graph.add_node("reformulate", reformulate_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("exhaust", exhaust_node)

    graph.set_entry_point("sanitize")

    graph.add_conditional_edges(
        "sanitize",
        _route_after_sanitize,
        {"end": END, "reformulate": "reformulate"},
    )
    graph.add_edge("reformulate", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        _route_after_evaluate,
        {"end": END, "reformulate": "reformulate", "exhaust": "exhaust"},
    )
    graph.add_edge("exhaust", END)

    return graph.compile()


_COMPILED_GRAPH = _build_graph()


# ---------------------------------------------------------------------------
# Public entry point — backward-compatible with rag_chain.py
# ---------------------------------------------------------------------------


def process_query(
    raw_query: str,
    confidence_threshold: float = QUERY_CONFIDENCE_THRESHOLD,
    max_iterations: int = MAX_SANITIZATION_ITERATIONS,
) -> QueryResult:
    """Query processing loop with confidence-based routing.

    Sanitizes the query, then iteratively reformulates and evaluates it
    using LLM agents via LangGraph. If confidence meets the threshold,
    routes to SEARCH. Otherwise, routes to ASK_USER for clarification.

    Falls back to word-count heuristic if Ollama is unavailable.

    Args:
        raw_query: The raw user query.
        confidence_threshold: Minimum confidence to proceed with search.
        max_iterations: Max reformulation attempts before asking user.

    Returns:
        QueryResult with the processed query and recommended action.
    """
    ollama_available = _check_ollama_available()
    if not ollama_available:
        logger.warning("Ollama unavailable; falling back to heuristic mode")

    initial_state: QueryState = {
        "original_query": raw_query,
        "current_query": raw_query,
        "confidence": 0.0,
        "reasoning": "",
        "iteration": 0,
        "max_iterations": max_iterations,
        "confidence_threshold": confidence_threshold,
        "action": "",
        "clarification_message": "",
        "ollama_available": ollama_available,
    }

    logger.info("Processing query: '%s'", raw_query[:100])
    final_state = _COMPILED_GRAPH.invoke(initial_state)

    action = (
        QueryAction.ASK_USER
        if final_state["action"] == "ask_user"
        else QueryAction.SEARCH
    )
    clarification = final_state["clarification_message"] or None

    logger.info(
        "Query processing complete: action=%s confidence=%.2f iterations=%d query='%s'",
        action.value,
        final_state["confidence"],
        final_state["iteration"],
        final_state["current_query"][:100],
    )

    return QueryResult(
        processed_query=final_state["current_query"],
        confidence=final_state["confidence"],
        action=action,
        clarification_message=clarification,
        iterations=final_state["iteration"],
    )
