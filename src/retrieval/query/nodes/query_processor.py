# @summary
# LangGraph-based query processing with confidence routing, backed by LiteLLM Router.
# Exports: process_query, QueryResult, QueryAction, warm_up_ollama
# Deps: langgraph, config.settings, src.platform.llm, src.retrieval.query.schemas, src.retrieval.common.utils
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
from __future__ import annotations


import orjson
import logging
import os
import re
import time
from typing import Optional

from langgraph.graph import END, StateGraph

from config.settings import (
    DOMAIN_DESCRIPTION,
    MAX_SANITIZATION_ITERATIONS,
    PROMPTS_DIR,
    QUERY_CONFIDENCE_THRESHOLD,
    QUERY_LOG_DIR,
    QUERY_PROCESSING_TEMPERATURE,
    RAG_QUERY_PRONOUN_DENSITY_THRESHOLD,
    RAG_QUERY_EVALUATOR_MAX_TOKENS,
    RAG_RETRIEVAL_REWRITER_MAX_TOKENS,
    RAG_QUERY_KG_MATCH_MAX_TERMS,
)
from src.platform.llm import call_oneshot, get_llm_provider
from src.platform.observability import get_tracer
from src.retrieval.query.schemas import QueryAction, QueryResult, QueryState
from src.retrieval.common import parse_json_object

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("rag.query_processor")

# File logging is set up lazily on first use so that importing this module
# (e.g., during test collection) does not create directories or file handles.
_file_logging_ready = False


def _ensure_file_logging() -> None:
    """Attach a rotating file handler the first time it is needed."""
    global _file_logging_ready
    if _file_logging_ready or logger.handlers:
        return
    _file_logging_ready = True
    try:
        os.makedirs(QUERY_LOG_DIR, exist_ok=True)
        handler = logging.FileHandler(QUERY_LOG_DIR / "query_processor.log")
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
    except OSError as exc:
        logger.warning(
            "Could not attach file logger to %s: %s — continuing with console only",
            QUERY_LOG_DIR,
            exc,
        )


# ---------------------------------------------------------------------------
# Public API — backward-compatible with rag_chain.py
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Prompt loading (cached at module level)
# ---------------------------------------------------------------------------


def _load_prompt(filename: str) -> str:
    _t0 = time.perf_counter()
    try:
        path = PROMPTS_DIR / filename
        if not path.exists():
            logger.error("Prompt file not found: %s", path)
            raise FileNotFoundError(f"Prompt file not found: {path}")
        text = path.read_text(encoding="utf-8")
        logger.debug(
            "_load_prompt: loaded %s (%d chars) in %.2fms",
            filename, len(text), (time.perf_counter() - _t0) * 1000,
        )
        return text
    except FileNotFoundError:
        raise
    except OSError as exc:
        logger.error("_load_prompt: failed to read %s: %s", filename, exc)
        raise


_REFORMULATOR_PROMPT: Optional[str] = None
_EVALUATOR_PROMPT: Optional[str] = None
_COMBINED_PROMPT: Optional[str] = None


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


def _get_combined_prompt() -> str:
    global _COMBINED_PROMPT
    if _COMBINED_PROMPT is None:
        _COMBINED_PROMPT = _load_prompt("query_reformulate_and_evaluate.md")
    return _COMBINED_PROMPT


_RETRIEVAL_REWRITER_PROMPT: Optional[str] = None


def _get_retrieval_rewriter_prompt() -> str:
    global _RETRIEVAL_REWRITER_PROMPT
    if _RETRIEVAL_REWRITER_PROMPT is None:
        _RETRIEVAL_REWRITER_PROMPT = _load_prompt("retrieval_query_rewriter.md")
    return _RETRIEVAL_REWRITER_PROMPT


_VALID_HISTORY_DECISIONS = {"use_as_is", "partial_history", "full_history"}


def _retrieval_auto_rewrite(
    user_query: str,
    memory_context: str,
) -> tuple[str, str, int]:
    """Run the retrieval-mode rewriter LLM and return (processed_query, decision, turns_used).

    Falls back to ``use_as_is`` with the literal query if the LLM is unavailable
    or returns malformed output. Never raises — Auto mode must always produce
    a usable query.
    """
    try:
        provider = get_llm_provider()
        if not _check_llm_available():
            return user_query, "use_as_is", 0
        prompt = _get_retrieval_rewriter_prompt()
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"USER_QUERY: {user_query}\n\n"
                    f"CONVERSATION_HISTORY:\n{memory_context or '(none)'}"
                ),
            },
        ]
        response = provider.generate(
            messages,
            model_alias="default",
            temperature=QUERY_PROCESSING_TEMPERATURE,
            max_tokens=RAG_RETRIEVAL_REWRITER_MAX_TOKENS,
        )
        raw = (response.content or "").strip()
        parsed = parse_json_object(raw) or {}
        decision = str(parsed.get("decision", "")).strip().lower()
        if decision not in _VALID_HISTORY_DECISIONS:
            decision = "use_as_is"
        processed = str(parsed.get("processed_query", "") or "").strip() or user_query
        try:
            turns_used = int(parsed.get("history_turns_used", 0) or 0)
        except (TypeError, ValueError):
            turns_used = 0
        if decision == "use_as_is":
            turns_used = 0
        return processed, decision, max(0, turns_used)
    except Exception as exc:
        logger.warning("Retrieval rewriter failed, using literal query: %s", exc)
        return user_query, "use_as_is", 0


# ---------------------------------------------------------------------------
# Knowledge graph vocabulary (loaded once, used for reformulation context)
# ---------------------------------------------------------------------------

def _kg_query_match(query: str, max_terms: int) -> list[str]:
    """Return KG terms relevant to *query* via the kgweave client facade.

    Delegates word-level matching + top-N fallback to the KGWeave service
    (in-process by default; HTTP when ``KGWEAVE_API_URL`` is set). Returns
    an empty list when the graph is unavailable or empty.
    """
    from kgweave.client import get_client  # noqa: PLC0415

    try:
        result = get_client().match_kg_query(query, max_terms=max_terms)
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("kg match_kg_query unavailable: %s", exc)
        return []
    return list(result.matched)


# ---------------------------------------------------------------------------
# Guardrails — prompt injection detection (fallback when NeMo is disabled)
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: Optional[list[re.Pattern]] = None


def _load_injection_patterns() -> list[re.Pattern]:
    """Load injection patterns from config/injection_patterns.yaml.

    Patterns are loaded once and cached. Falls back to a minimal
    hardcoded set if the YAML file is missing or unreadable.
    """
    global _INJECTION_PATTERNS
    if _INJECTION_PATTERNS is not None:
        return _INJECTION_PATTERNS

    _t0 = time.perf_counter()
    try:
        import yaml
        patterns_path = PROMPTS_DIR.parent / "config" / "injection_patterns.yaml"
        if not patterns_path.exists():
            # File absent — expected in minimal environments; use fallback silently.
            pass
        else:
            with open(patterns_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            raw_patterns = data.get("patterns", [])
            _INJECTION_PATTERNS = [re.compile(p, re.I) for p in raw_patterns]
            logger.info(
                "Loaded %d injection patterns from %s in %.1fms",
                len(_INJECTION_PATTERNS),
                patterns_path,
                (time.perf_counter() - _t0) * 1000,
            )
            return _INJECTION_PATTERNS
    except Exception as e:
        # The file exists but could not be parsed — this is an operator error.
        # Log at ERROR so it surfaces in production alerting. Degraded injection
        # detection (3 fallback patterns) is active until the file is fixed.
        logger.error(
            "Injection patterns file could not be loaded (%s) — "
            "falling back to minimal pattern set. Injection detection is degraded. "
            "Fix %s to restore full coverage.",
            e,
            PROMPTS_DIR.parent / "config" / "injection_patterns.yaml",
        )

    # Minimal fallback if file is missing
    _INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts)", re.I),
        re.compile(r"you\s+are\s+now\s+", re.I),
        re.compile(r"forget\s+(everything|your\s+instructions)", re.I),
    ]
    return _INJECTION_PATTERNS


def _detect_injection(query: str) -> bool:
    """Check for prompt injection patterns (fallback for when NeMo is disabled).

    When NeMo Guardrails is enabled, its 4-layer injection detection
    (regex + perplexity + model + LLM) handles this. This function
    serves as a lightweight fallback for environments without NeMo.
    """
    _t0 = time.perf_counter()
    try:
        patterns = _load_injection_patterns()
        hit = any(p.search(query) for p in patterns)
        if hit:
            logger.warning(
                "_detect_injection: match in %.2fms (query_len=%d)",
                (time.perf_counter() - _t0) * 1000, len(query),
            )
        else:
            logger.debug(
                "_detect_injection: clean in %.2fms (query_len=%d)",
                (time.perf_counter() - _t0) * 1000, len(query),
            )
        return hit
    except re.error as exc:
        logger.error("_detect_injection: regex error %s — treating as clean", exc)
        return False


# ---------------------------------------------------------------------------
# Backward-reference and context-reset detection (REQ-1103, REQ-1105)
# ---------------------------------------------------------------------------

# Backward-reference markers (REQ-1103)
_BACKWARD_REF_PATTERNS = [
    re.compile(
        r"\b(the above|you said|you mentioned|previously|tell me more|elaborate"
        r"|based on what we discussed|regarding what you mentioned|as we discussed"
        r"|from earlier|what about that)\b",
        re.IGNORECASE,
    ),
]

# Context-reset markers (REQ-1105)
_CONTEXT_RESET_PATTERNS = [
    re.compile(
        r"\b(forget about (the )?(past |previous )?(conversation|convo|chat|discussion)"
        r"|ignore (the )?(previous|prior|past)"
        r"|new topic|start fresh|fresh start"
        r"|disregard what we discussed)\b",
        re.IGNORECASE,
    ),
]

# Pronouns for backward-ref density check
_PRONOUNS = re.compile(r"\b(it|its|that|those|this|these|them)\b", re.IGNORECASE)


def _has_backward_reference(query: str) -> bool:
    """Detect backward-reference signals in a query (REQ-1103).

    Returns True if the query contains explicit backward-reference markers
    or has high pronoun density without resolution targets.
    """
    _t0 = time.perf_counter()
    try:
        if any(p.search(query) for p in _BACKWARD_REF_PATTERNS):
            logger.debug(
                "_has_backward_reference: explicit marker in %.2fms",
                (time.perf_counter() - _t0) * 1000,
            )
            return True
        words = query.split()
        if words:
            pronoun_count = len(_PRONOUNS.findall(query))
            if pronoun_count / len(words) >= RAG_QUERY_PRONOUN_DENSITY_THRESHOLD:
                logger.debug(
                    "_has_backward_reference: high pronoun density (%d/%d) in %.2fms",
                    pronoun_count, len(words), (time.perf_counter() - _t0) * 1000,
                )
                return True
        return False
    except re.error as exc:
        logger.error("_has_backward_reference: regex error %s — treating as absent", exc)
        return False


def _detect_suppress_memory(query: str) -> bool:
    """Detect explicit context-reset signals in a query (REQ-1105).

    Returns True if the user explicitly asks to ignore conversation history.
    """
    _t0 = time.perf_counter()
    try:
        hit = any(p.search(query) for p in _CONTEXT_RESET_PATTERNS)
        logger.debug(
            "_detect_suppress_memory: %s in %.2fms",
            "hit" if hit else "clean", (time.perf_counter() - _t0) * 1000,
        )
        return hit
    except re.error as exc:
        logger.error("_detect_suppress_memory: regex error %s — treating as absent", exc)
        return False


# ---------------------------------------------------------------------------
# LLM helper (backed by LiteLLM Router via LLMProvider)
# ---------------------------------------------------------------------------


def _call_llm(prompt: str, system: str = "") -> Optional[str]:
    """Call LLM via the platform-layer one-shot helper, wrapped in a tracing span."""
    with get_tracer().span("query_processor.call_llm", {"model_alias": "query"}):
        return call_oneshot(
            prompt,
            system=system,
            model_alias="query",
            temperature=QUERY_PROCESSING_TEMPERATURE,
            max_tokens=RAG_QUERY_EVALUATOR_MAX_TOKENS,
        )


def _check_llm_available() -> bool:
    """Check if the LLM provider is reachable."""
    with get_tracer().span("query_processor.llm_healthcheck"):
        try:
            provider = get_llm_provider()
            available = provider.is_available(model_alias="query")
            logger.debug("_check_llm_available: %s", available)
            return available
        except Exception as exc:
            logger.warning("_check_llm_available: provider error: %s", exc)
            return False


# Backward-compatible alias
_check_ollama_available = _check_llm_available


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

    if state.get("fast_path", False):
        return {
            "current_query": query,
            "confidence": 1.0,
            "reasoning": "fast_path enabled",
        }

    return {"current_query": query}


def _match_kg_terms(query: str, max_terms: int = RAG_QUERY_KG_MATCH_MAX_TERMS) -> str:
    """Find KG terms relevant to the query and format them for prompt injection.

    Match logic (word-level lookup + top-N fallback) lives in
    ``KGQueryService.match_kg_query`` so it can serve both in-process and
    remote callers identically.
    """
    matched = _kg_query_match(query, max_terms=max_terms)
    if not matched:
        return ""
    return "Known terms in the knowledge base: " + ", ".join(matched)


def reformulate_and_evaluate_node(state: QueryState) -> dict:
    """Combined LLM reformulation + evaluation in a single Ollama call.

    Halves the number of LLM round-trips compared to separate nodes.
    """
    new_iteration = state["iteration"] + 1

    if not state["ollama_available"]:
        conf = _heuristic_confidence(state["current_query"])
        logger.info(
            "Iteration %d: heuristic confidence=%.2f (Ollama unavailable)",
            new_iteration,
            conf,
        )
        return {
            "current_query": state["current_query"],
            "iteration": new_iteration,
            "confidence": conf,
            "reasoning": "fallback heuristic (Ollama unavailable)",
        }

    previous_feedback = ""
    if state["reasoning"]:
        # state["reasoning"] originates from the previous LLM response and
        # could contain adversarially crafted content if the user's original
        # query was designed to elicit it. Check the FULL text for injection
        # patterns first, then truncate the clean version to a safe length.
        if _detect_injection(state["reasoning"]):
            logger.warning(
                "Injection-like content detected in LLM reasoning field — "
                "omitting from next prompt iteration"
            )
            safe_reasoning = ""
        else:
            safe_reasoning = state["reasoning"][:200]
        if safe_reasoning:
            previous_feedback = f"Previous evaluator feedback: {safe_reasoning}"

    kg_terms = _match_kg_terms(state["original_query"])

    prompt = _get_combined_prompt().format(
        original_query=state["original_query"],
        iteration=new_iteration,
        max_iterations=state["max_iterations"],
        previous_feedback=previous_feedback,
        kg_terms=kg_terms,
        domain_description=DOMAIN_DESCRIPTION,
    )

    result = _call_llm(prompt)

    if result:
        try:
            parsed = parse_json_object(result)

            # Support new dual-query format (processed_query + standalone_query) and
            # backward-compatible legacy format (reformulated_query).
            processed = (
                str(parsed.get("processed_query", "")).strip()
                or str(parsed.get("reformulated_query", "")).strip()
            )
            standalone = str(parsed.get("standalone_query", "")).strip()
            confidence = float(parsed.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
            reasoning = str(parsed.get("reasoning", ""))

            if processed:
                logger.info(
                    "Iteration %d: '%s' -> '%s' (confidence=%.2f, reasoning='%s')",
                    new_iteration,
                    state["current_query"],
                    processed,
                    confidence,
                    reasoning,
                )
            else:
                processed = state["current_query"]
                logger.info(
                    "Iteration %d: reformulation empty, keeping current query (confidence=%.2f, reasoning='%s')",
                    new_iteration,
                    confidence,
                    reasoning,
                )

            # When no memory context is present, standalone_query must equal
            # processed_query (no conversation to separate out).
            if not state.get("has_memory_context", False) or not standalone:
                standalone = processed

            return {
                "current_query": processed,
                "standalone_query": standalone,
                "iteration": new_iteration,
                "confidence": confidence,
                "reasoning": reasoning,
            }
        except (orjson.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(
                "Failed to parse combined JSON: %s. Raw: %s", e, result[:200]
            )
            # Fallback: treat raw response as processed_query; standalone mirrors it.
            fallback_query = state["current_query"]
            return {
                "current_query": fallback_query,
                "standalone_query": fallback_query,
                "iteration": new_iteration,
                "confidence": 0.5,
                "reasoning": "parse failed",
            }

    logger.warning(
        "Iteration %d: combined reformulate+evaluate returned no content, keeping query",
        new_iteration,
    )
    return {
        "current_query": state["current_query"],
        "standalone_query": state["current_query"],
        "iteration": new_iteration,
        "confidence": 0.5,
        "reasoning": "empty response",
    }


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
    return "reformulate_and_evaluate"


def _route_after_combined(state: QueryState) -> str:
    if state["confidence"] >= state["confidence_threshold"]:
        return "end"
    if state["iteration"] >= state["max_iterations"]:
        return "exhaust"
    return "reformulate_and_evaluate"


# ---------------------------------------------------------------------------
# Graph assembly (compiled lazily on first invocation)
# ---------------------------------------------------------------------------


def _build_graph():
    graph = StateGraph(QueryState)

    graph.add_node("sanitize", sanitize_node)
    graph.add_node("reformulate_and_evaluate", reformulate_and_evaluate_node)
    graph.add_node("exhaust", exhaust_node)

    graph.set_entry_point("sanitize")

    graph.add_conditional_edges(
        "sanitize",
        _route_after_sanitize,
        {"end": END, "reformulate_and_evaluate": "reformulate_and_evaluate"},
    )
    graph.add_conditional_edges(
        "reformulate_and_evaluate",
        _route_after_combined,
        {"end": END, "reformulate_and_evaluate": "reformulate_and_evaluate", "exhaust": "exhaust"},
    )
    graph.add_edge("exhaust", END)

    return graph.compile()


# Compiled graph is initialised on first call to process_query() rather than
# at import time. This avoids LangGraph compilation overhead during test
# collection and prevents StateGraph errors if langgraph is not fully
# initialised when the module is first imported.
_compiled_graph_instance = None


def _get_compiled_graph():
    global _compiled_graph_instance
    if _compiled_graph_instance is None:
        _compiled_graph_instance = _build_graph()
    return _compiled_graph_instance


# ---------------------------------------------------------------------------
# Public entry point — backward-compatible with rag_chain.py
# ---------------------------------------------------------------------------


def warm_up_ollama() -> None:
    """Send a tiny request to Ollama to pre-load the query processing model.

    Call this at worker startup so the first real query doesn't pay the
    cold-start cost (~5-10s model load).
    """
    _ensure_file_logging()
    if not _check_llm_available():
        logger.warning("LLM not reachable — skipping warm-up")
        return

    logger.info("Warming up query LLM model...")
    t0 = time.perf_counter()
    _call_llm("ping", system="Reply with 'ok' only.")
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("LLM warm-up done in %.0fms", elapsed)


def process_query(
    raw_query: str,
    confidence_threshold: float = QUERY_CONFIDENCE_THRESHOLD,
    max_iterations: int = MAX_SANITIZATION_ITERATIONS,
    fast_path: bool = False,
    memory_context: Optional[str] = None,
    user_query: Optional[str] = None,
    mode: str = "query",
    retrieval_sub_mode: str = "auto",
) -> QueryResult:
    """Query processing loop with confidence-based routing.

    Sanitizes the query, then iteratively reformulates and evaluates it
    using LLM agents via LangGraph. If confidence meets the threshold,
    routes to SEARCH. Otherwise, routes to ASK_USER for clarification.

    Falls back to word-count heuristic if Ollama is unavailable.

    When ``memory_context`` is provided (non-empty), the graph state is
    flagged with ``has_memory_context=True`` so the reformulation node
    can produce a distinct ``standalone_query`` (current-turn only) in
    addition to the context-enriched ``processed_query``. When memory
    context is absent, ``standalone_query`` mirrors ``processed_query``.

    Args:
        raw_query: The raw user query (may already include prepended
            conversation context when called from rag_chain).
        confidence_threshold: Minimum confidence to proceed with search.
        max_iterations: Max reformulation attempts before asking user.
        fast_path: Skip LLM reformulation when True.
        memory_context: The raw conversation memory string (used only to
            set the ``has_memory_context`` flag; the caller is responsible
            for prepending it to ``raw_query`` before this call).
        user_query: The bare user query without any memory prepending. When
            provided, detection functions (_has_backward_reference,
            _detect_suppress_memory) operate on this string instead of
            ``raw_query`` to avoid false positives from memory context
            containing reference phrases (review bug B2).

    Returns:
        QueryResult with the processed query, standalone query, and
        recommended action.
    """
    _ensure_file_logging()
    # Retrieval mode short-circuits the LangGraph creation/verification loop.
    # Hard sub-mode never calls the LLM; Auto sub-mode runs a single
    # retrieval-tuned rewriter pass. Both produce a usable QueryResult with
    # ``action=SEARCH`` because retrieval never asks the user for clarification —
    # the user always sees ranked docs, even if the rewriter's output is poor.
    if mode == "retrieval":
        bare_user_query = (user_query if user_query is not None else raw_query) or ""
        if retrieval_sub_mode == "hard":
            return QueryResult(
                processed_query=bare_user_query,
                standalone_query=bare_user_query,
                confidence=1.0,
                action=QueryAction.SEARCH,
                clarification_message=None,
                iterations=0,
                has_backward_reference=False,
                suppress_memory=True,
                history_decision="hard_query",
                history_turns_used=0,
            )
        # Auto sub-mode.
        processed, decision, turns_used = _retrieval_auto_rewrite(
            bare_user_query, memory_context or ""
        )
        return QueryResult(
            processed_query=processed,
            standalone_query=processed,
            confidence=1.0,
            action=QueryAction.SEARCH,
            clarification_message=None,
            iterations=1,
            has_backward_reference=(decision != "use_as_is"),
            suppress_memory=False,
            history_decision=decision,
            history_turns_used=turns_used,
        )

    with get_tracer().span("query_processor.process_query", {"raw_query_len": len(raw_query)}) as root_span:
        ollama_available = _check_llm_available()
        if not ollama_available:
            logger.warning("LLM unavailable; falling back to heuristic mode")

        has_memory_context = bool(memory_context and memory_context.strip())

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
            "fast_path": fast_path,
            "standalone_query": "",
            "suppress_memory": False,
            "has_backward_reference": False,
            "has_memory_context": has_memory_context,
        }

        logger.info("Processing query: '%s'", raw_query[:100])
        final_state = _get_compiled_graph().invoke(initial_state)

        action = (
            QueryAction.ASK_USER
            if final_state["action"] == "ask_user"
            else QueryAction.SEARCH
        )
        clarification = final_state["clarification_message"] or None

        # Resolve standalone_query: fall back to processed_query when absent
        # or when there was no memory context to separate out.
        processed_query = final_state["current_query"]
        standalone_query = final_state.get("standalone_query") or ""
        if not has_memory_context or not standalone_query:
            standalone_query = processed_query

        logger.info(
            "Query processing complete: action=%s confidence=%.2f iterations=%d "
            "query='%s' standalone='%s'",
            action.value,
            final_state["confidence"],
            final_state["iteration"],
            processed_query[:100],
            standalone_query[:100],
        )

        # Detect conversational routing signals (REQ-1103, REQ-1105).
        # Use bare user query for signal detection to avoid false positives
        # from memory context containing reference phrases (review bug B2).
        detection_query = user_query if user_query is not None else raw_query
        has_backward_ref = _has_backward_reference(detection_query)
        suppress_mem = _detect_suppress_memory(detection_query)

        result = QueryResult(
            processed_query=processed_query,
            standalone_query=standalone_query,
            confidence=final_state["confidence"],
            action=action,
            clarification_message=clarification,
            iterations=final_state["iteration"],
            has_backward_reference=has_backward_ref,
            suppress_memory=suppress_mem,
        )
        root_span.set_attribute("action", result.action.value)
        root_span.set_attribute("confidence", result.confidence)
        return result


__all__ = [
    "QueryAction",
    "QueryResult",
    "QueryState",
    "process_query",
    "warm_up_ollama",
]
