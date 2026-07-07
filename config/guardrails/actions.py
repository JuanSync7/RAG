# @summary
# NeMo-discoverable Colang action wrappers. Thin wrappers around existing
# guardrail rail classes and new lightweight policy actions.
# Auto-discovered by NeMo Guardrails when placed in the config directory.
# Exports: all @action()-decorated functions
# Deps: nemoguardrails, src.guardrails.*, langdetect, re, logging
# @end-summary
"""NeMo Guardrails Colang action wrappers.

Each function decorated with @action() is auto-discovered by the NeMo runtime
when this file lives inside the guardrails config directory. Actions are thin
wrappers — they delegate to existing rail classes or implement lightweight
deterministic checks. All actions return dicts for Colang variable assignment.
"""

from __future__ import annotations

import functools
import logging
import re
from typing import Any, Dict

# Conditional import: NeMo may not be installed when tests run with RAG_NEMO_ENABLED=false
try:
    from nemoguardrails.actions import action
except ImportError:
    def action():
        """No-op decorator when nemoguardrails is not installed."""
        def decorator(fn):
            return fn
        return decorator

logger = logging.getLogger("rag.guardrails.actions")

# ── Session state (in-memory, keyed by session_id) ──
_jailbreak_session_state: Dict[str, int] = {}
_abuse_session_state: Dict[str, list] = {}

# ── Lazy-initialized rail class singletons ──
_rail_instances: Dict[str, Any] = {}


def _fail_open(default: dict):
    """Decorator: catch exceptions and return default (fail-open)."""
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                logger.warning("Action %s failed: %s — returning default", fn.__name__, e)
                return default
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════
# 2.2 — Lightweight deterministic actions
# ═══════════════════════════════════════════════════════════════════════

@action()
@_fail_open({"valid": True, "length": 0, "reason": ""})
async def check_query_length(query: str) -> dict:
    """Validate query length: min/max chars (operator-tunable, defaults 3/2000)."""
    from config.settings import (
        RAG_GUARDRAILS_QUERY_MIN_CHARS,
        RAG_GUARDRAILS_QUERY_MAX_CHARS,
    )
    length = len(query.strip())
    if length < RAG_GUARDRAILS_QUERY_MIN_CHARS:
        return {"valid": False, "length": length, "reason": "Query too short. Please provide at least a few words."}
    if length > RAG_GUARDRAILS_QUERY_MAX_CHARS:
        return {"valid": False, "length": length, "reason": f"Query too long. Please keep your question under {RAG_GUARDRAILS_QUERY_MAX_CHARS} characters."}
    return {"valid": True, "length": length, "reason": ""}


@action()
@_fail_open({"language": "en", "supported": True})
async def detect_language(query: str) -> dict:
    """Detect query language using langdetect. Only English is supported."""
    try:
        from langdetect import detect
        lang = detect(query)
    except Exception:
        return {"language": "unknown", "supported": True}
    return {"language": lang, "supported": lang == "en"}


@action()
@_fail_open({"clear": True, "suggestion": ""})
async def check_query_clarity(query: str) -> dict:
    """Heuristic clarity check: reject very short or all-stopword queries."""
    words = query.strip().split()
    stopwords = {"a", "an", "the", "it", "is", "was", "are", "to", "of", "in", "on", "and", "or", "for", "that", "this", "what"}
    if len(words) < 2:
        return {"clear": False, "suggestion": f"Your query '{query}' is too vague. Could you provide more detail about what you're looking for?"}
    non_stop = [w for w in words if w.lower() not in stopwords]
    if not non_stop:
        return {"clear": False, "suggestion": "Your query doesn't contain specific terms. Please include keywords related to what you want to find."}
    return {"clear": True, "suggestion": ""}


@action()
@_fail_open({"abusive": False, "reason": ""})
async def check_abuse_pattern(query: str, context: dict = None) -> dict:
    """Track query rate per session. Flag if too many queries in rapid succession (operator-tunable, default > 20 within 60s)."""
    import time
    from config.settings import (
        RAG_NEMO_ABUSE_WINDOW_SECONDS,
        RAG_NEMO_ABUSE_MAX_QUERIES,
    )
    session_id = context.get("session_id", "default") if context else "default"
    now = time.time()
    window = RAG_NEMO_ABUSE_WINDOW_SECONDS  # rate-limit window in seconds
    if session_id not in _abuse_session_state:
        _abuse_session_state[session_id] = []
    _abuse_session_state[session_id] = [t for t in _abuse_session_state[session_id] if now - t < window]
    _abuse_session_state[session_id].append(now)
    if len(_abuse_session_state[session_id]) > RAG_NEMO_ABUSE_MAX_QUERIES:
        return {"abusive": True, "reason": "Rate limit exceeded"}
    return {"abusive": False, "reason": ""}


@action()
@_fail_open({"has_citations": True})
async def check_citations(answer: str) -> dict:
    """Check if answer contains citation patterns like [Source: ...] or [1]."""
    patterns = [
        r'\[Source:.*?\]',
        r'\[\d+\]',
        r'\(Source:.*?\)',
        r'According to',
        r'Based on the document',
    ]
    for pattern in patterns:
        if re.search(pattern, answer, re.IGNORECASE):
            return {"has_citations": True}
    return {"has_citations": False}


@action()
@_fail_open({"answer": ""})
async def add_citation_reminder(answer: str) -> dict:
    """Append citation reminder to answer."""
    return {"answer": f"{answer}\n\nNote: Sources are available in the response metadata."}


@action()
@_fail_open({"answer": ""})
async def prepend_hedge(answer: str) -> dict:
    """Prepend hedge language for low-confidence answers."""
    return {"answer": f"Based on limited information in the knowledge base: {answer}"}


@action()
@_fail_open({"answer": ""})
async def prepend_text(text: str, answer: str) -> dict:
    """Prepend arbitrary text (e.g., disclaimer) to answer."""
    return {"answer": f"{text}\n\n{answer}"}


@action()
@_fail_open({"answer": ""})
async def prepend_low_confidence_note(answer: str) -> dict:
    """Prepend low-confidence note."""
    return {"answer": f"Note: The following answer is based on limited matches in the knowledge base.\n\n{answer}"}


@action()
@_fail_open({"valid": True, "reason": ""})
async def check_answer_length(answer: str) -> dict:
    """Validate answer length: min/max chars (operator-tunable, defaults 20/5000)."""
    from config.settings import (
        RAG_GUARDRAILS_ANSWER_MIN_CHARS,
        RAG_GUARDRAILS_ANSWER_MAX_CHARS,
    )
    length = len(answer.strip())
    if length < RAG_GUARDRAILS_ANSWER_MIN_CHARS:
        return {"valid": False, "reason": "too short"}
    if length > RAG_GUARDRAILS_ANSWER_MAX_CHARS:
        return {"valid": False, "reason": "too long"}
    return {"valid": True, "reason": ""}


@action()
@_fail_open({"answer": ""})
async def adjust_answer_length(answer: str, reason: str) -> dict:
    """Truncate overly long answers or flag terse ones."""
    from config.settings import RAG_GUARDRAILS_ANSWER_MAX_CHARS
    if reason == "too long":
        return {"answer": answer[:RAG_GUARDRAILS_ANSWER_MAX_CHARS] + "..."}
    return {"answer": answer}


@action()
@_fail_open({"sensitive": False, "disclaimer": "", "domain": ""})
async def check_sensitive_topic(query: str) -> dict:
    """Keyword + regex check for medical/legal/financial sensitive topics."""
    q_lower = query.lower()
    domains = {
        "medical": {
            "keywords": ["medication", "dosage", "symptom", "diagnosis", "prescription", "treatment", "disease", "medical advice", "drug interaction"],
            "disclaimer": "Note: This information is from the knowledge base and is not medical advice. Please consult a healthcare professional.",
        },
        "legal": {
            "keywords": ["legal advice", "lawsuit", "attorney", "court", "liability", "legal rights", "sue", "legal action"],
            "disclaimer": "Note: This is informational only and does not constitute legal advice. Please consult a qualified attorney.",
        },
        "financial": {
            "keywords": ["investment advice", "stock pick", "buy or sell", "financial advice", "portfolio", "tax advice"],
            "disclaimer": "Note: This is not financial advice. Please consult a qualified financial professional.",
        },
    }
    for domain, info in domains.items():
        for kw in info["keywords"]:
            if kw in q_lower:
                return {"sensitive": True, "disclaimer": info["disclaimer"], "domain": domain}
    return {"sensitive": False, "disclaimer": "", "domain": ""}


@action()
@_fail_open({"attempt": False, "pattern": ""})
async def check_exfiltration(query: str) -> dict:
    """Detect bulk data extraction patterns."""
    patterns = [
        r"list\s+all\s+(documents|records|entries|files|data)",
        r"dump\s+(everything|all|the\s+database)",
        r"show\s+me\s+(all|every)\s+(records?|documents?|entries?)",
        r"export\s+(the\s+)?(database|data|everything)",
        r"give\s+me\s+every(thing|\s+entry|\s+record|\s+document)",
        r"download\s+all",
        r"extract\s+all",
    ]
    for pattern in patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return {"attempt": True, "pattern": pattern}
    return {"attempt": False, "pattern": ""}


@action()
@_fail_open({"violation": False})
async def check_role_boundary(query: str) -> dict:
    """Detect role-play and instruction-override patterns."""
    patterns = [
        r"you\s+are\s+now\s+a",
        r"ignore\s+(previous|all|your)\s+instructions",
        r"pretend\s+(you\s+are|to\s+be)",
        r"act\s+as\s+if",
        r"forget\s+(everything|your\s+rules|your\s+instructions)",
        r"disregard\s+(your|all)\s+(rules|guidelines|instructions)",
        r"you\s+have\s+no\s+restrictions",
        r"jailbreak",
        r"DAN\s+mode",
    ]
    for pattern in patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return {"violation": True}
    return {"violation": False}


@action()
@_fail_open({"escalation_level": "none"})
async def check_jailbreak_escalation(query: str, context: dict = None) -> dict:
    """Track jailbreak attempt count per session. Thresholds operator-tunable (defaults: warn at 1, block at 3)."""
    from config.settings import (
        RAG_NEMO_JAILBREAK_BLOCK_THRESHOLD,
        RAG_NEMO_JAILBREAK_WARN_THRESHOLD,
    )
    session_id = context.get("session_id", "default") if context else "default"
    violation_patterns = [
        r"ignore\s+(previous|all|your)\s+instructions",
        r"you\s+are\s+now\s+a",
        r"pretend\s+(you\s+are|to\s+be)",
        r"jailbreak",
        r"DAN\s+mode",
    ]
    is_violation = any(re.search(p, query, re.IGNORECASE) for p in violation_patterns)
    if is_violation:
        _jailbreak_session_state[session_id] = _jailbreak_session_state.get(session_id, 0) + 1

    count = _jailbreak_session_state.get(session_id, 0)
    if count >= RAG_NEMO_JAILBREAK_BLOCK_THRESHOLD:
        return {"escalation_level": "block"}
    elif count >= RAG_NEMO_JAILBREAK_WARN_THRESHOLD:
        return {"escalation_level": "warn"}
    return {"escalation_level": "none"}


@action()
@_fail_open({"has_context": False, "augmented_query": ""})
async def handle_follow_up(query: str) -> dict:
    """Check for prior conversation context (stub)."""
    return {"has_context": False, "augmented_query": query}


@action()
@_fail_open({"drifted": False})
async def check_topic_drift(query: str) -> dict:
    """Topic drift detection (stub)."""
    return {"drifted": False}


@action()
@_fail_open({"confidence": "high"})
async def check_response_confidence(answer: str) -> dict:
    """Read retrieval confidence from context (stub)."""
    if not answer or answer.strip() == "":
        return {"confidence": "none"}
    return {"confidence": "high"}


@action()
@_fail_open({"has_results": True, "count": 1, "avg_confidence": 1.0})
async def check_retrieval_results(answer: str) -> dict:
    """Check retrieval results quality (stub)."""
    if not answer or answer.strip() == "":
        return {"has_results": False, "count": 0, "avg_confidence": 0.0}
    return {"has_results": True, "count": 1, "avg_confidence": 0.8}


@action()
@_fail_open({"in_scope": True})
async def check_source_scope(answer: str) -> dict:
    """Source scope check (stub)."""
    return {"in_scope": True}


@action()
@_fail_open({"ambiguous": False, "disambiguation_prompt": ""})
async def check_query_ambiguity(query: str) -> dict:
    """Query ambiguity check (stub)."""
    return {"ambiguous": False, "disambiguation_prompt": ""}


@action()
@_fail_open({"summary": "This knowledge base contains documents about various topics."})
async def get_knowledge_base_summary() -> dict:
    """Return a static summary of the knowledge base contents."""
    return {"summary": "I have access to documents in the knowledge base covering various technical topics. You can ask questions about any topic covered by the ingested documents. Try asking about specific concepts, architectures, or techniques."}


# ═══════════════════════════════════════════════════════════════════════
# 2.1 — Lazy-init helpers for executor singletons
# ═══════════════════════════════════════════════════════════════════════

def _get_input_executor():
    """Lazy-initialize InputRailExecutor with config from env vars."""
    if "input_executor" not in _rail_instances:
        from config.settings import (
            RAG_NEMO_INJECTION_ENABLED,
            RAG_NEMO_INJECTION_SENSITIVITY,
            RAG_NEMO_INJECTION_PERPLEXITY_ENABLED,
            RAG_NEMO_INJECTION_MODEL_ENABLED,
            RAG_NEMO_INJECTION_LP_THRESHOLD,
            RAG_NEMO_INJECTION_PS_PPL_THRESHOLD,
            RAG_NEMO_INTENT_CONFIDENCE_THRESHOLD,
            RAG_NEMO_PII_ENABLED,
            RAG_NEMO_PII_EXTENDED,
            RAG_NEMO_PII_SCORE_THRESHOLD,
            RAG_NEMO_TOXICITY_ENABLED,
            RAG_NEMO_TOXICITY_THRESHOLD,
            RAG_NEMO_TOPIC_SAFETY_ENABLED,
            RAG_NEMO_TOPIC_SAFETY_INSTRUCTIONS,
            RAG_NEMO_RAIL_TIMEOUT_SECONDS,
        )
        from src.guardrails.nemo_guardrails import InputRailExecutor
        from src.guardrails.shared import (
            InjectionDetector,
            IntentClassifier,
            PIIDetector,
            TopicSafetyChecker,
            ToxicityFilter,
        )

        intent_classifier = IntentClassifier(
            confidence_threshold=RAG_NEMO_INTENT_CONFIDENCE_THRESHOLD,
        )
        injection_detector = (
            InjectionDetector(
                sensitivity=RAG_NEMO_INJECTION_SENSITIVITY,
                enable_perplexity=RAG_NEMO_INJECTION_PERPLEXITY_ENABLED,
                enable_model_classifier=RAG_NEMO_INJECTION_MODEL_ENABLED,
                lp_threshold=RAG_NEMO_INJECTION_LP_THRESHOLD,
                ps_ppl_threshold=RAG_NEMO_INJECTION_PS_PPL_THRESHOLD,
            )
            if RAG_NEMO_INJECTION_ENABLED
            else None
        )
        from config.settings import RAG_NEMO_PII_GLINER_ENABLED
        pii_detector = (
            PIIDetector(
                extended=RAG_NEMO_PII_EXTENDED,
                score_threshold=RAG_NEMO_PII_SCORE_THRESHOLD,
                use_gliner=RAG_NEMO_PII_GLINER_ENABLED,
            )
            if RAG_NEMO_PII_ENABLED
            else None
        )
        toxicity_filter = (
            ToxicityFilter(threshold=RAG_NEMO_TOXICITY_THRESHOLD)
            if RAG_NEMO_TOXICITY_ENABLED
            else None
        )
        topic_safety_checker = (
            TopicSafetyChecker(
                custom_instructions=RAG_NEMO_TOPIC_SAFETY_INSTRUCTIONS,
            )
            if RAG_NEMO_TOPIC_SAFETY_ENABLED
            else None
        )

        _rail_instances["input_executor"] = InputRailExecutor(
            intent_classifier=intent_classifier,
            injection_detector=injection_detector,
            pii_detector=pii_detector,
            toxicity_filter=toxicity_filter,
            topic_safety_checker=topic_safety_checker,
            timeout_seconds=RAG_NEMO_RAIL_TIMEOUT_SECONDS,
        )
        from src.guardrails.common import RailMergeGate
        _rail_instances["merge_gate"] = RailMergeGate()
        # Store shared instances for output executor
        _rail_instances["_pii"] = pii_detector
        _rail_instances["_toxicity"] = toxicity_filter

    return _rail_instances["input_executor"]


def _get_output_executor():
    """Lazy-initialize OutputRailExecutor with config from env vars."""
    if "output_executor" not in _rail_instances:
        from config.settings import (
            RAG_NEMO_FAITHFULNESS_ENABLED,
            RAG_NEMO_FAITHFULNESS_THRESHOLD,
            RAG_NEMO_FAITHFULNESS_ACTION,
            RAG_NEMO_FAITHFULNESS_SELF_CHECK,
            RAG_NEMO_OUTPUT_PII_ENABLED,
            RAG_NEMO_OUTPUT_TOXICITY_ENABLED,
            RAG_NEMO_RAIL_TIMEOUT_SECONDS,
        )
        from src.guardrails.nemo_guardrails import OutputRailExecutor
        from src.guardrails.shared import FaithfulnessChecker

        # Ensure input executor is initialized first (for shared instances)
        _get_input_executor()

        faithfulness_checker = (
            FaithfulnessChecker(
                threshold=RAG_NEMO_FAITHFULNESS_THRESHOLD,
                action=RAG_NEMO_FAITHFULNESS_ACTION,
                use_self_check=RAG_NEMO_FAITHFULNESS_SELF_CHECK,
            )
            if RAG_NEMO_FAITHFULNESS_ENABLED
            else None
        )
        output_pii = _rail_instances.get("_pii") if RAG_NEMO_OUTPUT_PII_ENABLED else None
        output_toxicity = _rail_instances.get("_toxicity") if RAG_NEMO_OUTPUT_TOXICITY_ENABLED else None

        _rail_instances["output_executor"] = OutputRailExecutor(
            faithfulness_checker=faithfulness_checker,
            pii_detector=output_pii,
            toxicity_filter=output_toxicity,
            timeout_seconds=RAG_NEMO_RAIL_TIMEOUT_SECONDS,
        )

    return _rail_instances["output_executor"]


# ═══════════════════════════════════════════════════════════════════════
# 2.1 — Actions wrapping existing rail classes (stubs — Task 5 fills in)
# ═══════════════════════════════════════════════════════════════════════

@action()
@_fail_open({"verdict": "pass", "method": "none", "confidence": 0.0})
async def check_injection(query: str) -> dict:
    """Wraps InjectionDetector. Stub — filled in Task 5."""
    return {"verdict": "pass", "method": "none", "confidence": 0.0}


@action()
@_fail_open({"found": False, "entities": [], "redacted_text": ""})
async def detect_pii(text: str, direction: str = "input") -> dict:
    """Wraps PIIDetector. Stub — filled in Task 5."""
    return {"found": False, "entities": [], "redacted_text": text}


@action()
@_fail_open({"verdict": "pass", "score": 0.0})
async def check_toxicity(text: str, direction: str = "input") -> dict:
    """Wraps ToxicityFilter. Stub — filled in Task 5."""
    return {"verdict": "pass", "score": 0.0}


@action()
@_fail_open({"on_topic": True, "confidence": 1.0})
async def check_topic_safety(query: str) -> dict:
    """Wraps TopicSafetyChecker. Stub — filled in Task 5."""
    return {"on_topic": True, "confidence": 1.0}


@action()
@_fail_open({"verdict": "pass", "score": 1.0, "claim_scores": []})
async def check_faithfulness(answer: str, context_chunks: list = None) -> dict:
    """Wraps FaithfulnessChecker. Stub — filled in Task 5."""
    return {"verdict": "pass", "score": 1.0, "claim_scores": []}


@action()
@_fail_open({"action": "pass", "intent": "rag_search", "redacted_query": "", "reject_message": "", "metadata": {}})
async def run_input_rails(query: str) -> dict:
    """Run all Python input rails via InputRailExecutor + RailMergeGate.

    Returns dict with: action ('pass'|'reject'|'modify'), intent, redacted_query,
    reject_message (if rejected), metadata.
    """
    import asyncio
    executor = _get_input_executor()
    merge_gate = _rail_instances.get("merge_gate")

    # InputRailExecutor.execute() is synchronous — run in thread
    loop = asyncio.get_event_loop()
    rail_result = await loop.run_in_executor(None, executor.execute, query)

    # Build a minimal query_result-like object for the merge gate
    class _QueryResult:
        def __init__(self, q):
            self.processed_query = q

    if merge_gate is not None:
        merge_decision = merge_gate.merge(_QueryResult(query), rail_result)
        action_str = merge_decision.get("action", "search")
        if action_str in ("reject", "canned"):
            return {
                "action": "reject",
                "intent": rail_result.intent or "unknown",
                "redacted_query": query,
                "reject_message": merge_decision.get("message", "Your query could not be processed."),
                "metadata": {},
            }
        return {
            "action": "modify" if rail_result.redacted_query else "pass",
            "intent": rail_result.intent or "rag_search",
            "redacted_query": merge_decision.get("query", query),
            "reject_message": "",
            "metadata": {},
        }

    # No merge gate — just pass through
    return {
        "action": "pass",
        "intent": rail_result.intent or "rag_search",
        "redacted_query": rail_result.redacted_query or query,
        "reject_message": "",
        "metadata": {},
    }


@action()
@_fail_open({"action": "pass", "redacted_answer": "", "reject_message": "", "metadata": {}})
async def run_output_rails(answer: str) -> dict:
    """Run all Python output rails via OutputRailExecutor.

    Returns dict with: action ('pass'|'reject'|'modify'), redacted_answer,
    reject_message (if rejected), metadata.
    """
    import asyncio
    executor = _get_output_executor()

    # OutputRailExecutor.execute() is synchronous — run in thread
    # Note: context_chunks would ideally come from NeMo context, but for now pass empty
    loop = asyncio.get_event_loop()
    rail_result = await loop.run_in_executor(None, executor.execute, answer, [])

    if rail_result.final_answer != answer:
        # Check if it was a rejection (faithfulness fail → fallback message)
        from src.guardrails.common import RailVerdict
        if rail_result.faithfulness_verdict == RailVerdict.REJECT:
            return {
                "action": "reject",
                "redacted_answer": "",
                "reject_message": rail_result.final_answer,
                "metadata": {},
            }
        return {
            "action": "modify",
            "redacted_answer": rail_result.final_answer,
            "reject_message": "",
            "metadata": {},
        }

    return {
        "action": "pass",
        "redacted_answer": answer,
        "reject_message": "",
        "metadata": {},
    }


# ── RAG chain reference (set by rag_chain.py during init) ──
_rag_chain_ref = None


def set_rag_chain(chain) -> None:
    """Called by rag_chain.py during _init_guardrails to provide the chain reference.

    This enables the rag_retrieve_and_generate action to call back into
    the RAG pipeline for retrieval + generation.
    """
    global _rag_chain_ref
    _rag_chain_ref = chain


@action()
@_fail_open({"answer": "", "sources": [], "confidence": 0.0})
async def rag_retrieve_and_generate(query: str) -> dict:
    """Execute the RAG retrieval+generation pipeline.

    Calls RAGChain.run() in a thread executor since run() is synchronous.
    The _nemo_bypass flag prevents infinite recursion (run() would otherwise
    re-enter the NeMo path).
    """
    if _rag_chain_ref is None:
        logger.warning("rag_retrieve_and_generate called but no RAG chain reference set")
        return {"answer": "", "sources": [], "confidence": 0.0}

    import asyncio

    # RAGChain.run() is synchronous — run in thread to avoid blocking NeMo's event loop
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            None,
            lambda: _rag_chain_ref.run(query=query),
        )
        return {
            "answer": response.answer or "",
            "sources": [s.get("source", "") for s in (response.sources or [])],
            "confidence": response.confidence_score if hasattr(response, "confidence_score") else 0.0,
        }
    except Exception as e:
        logger.warning("rag_retrieve_and_generate failed: %s", e)
        return {"answer": "", "sources": [], "confidence": 0.0}
