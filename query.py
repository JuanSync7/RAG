#!/usr/bin/env python3
# @summary
# Core RAG query logic: logging setup, filter parsing, result display, output suppression.
# Main exports: parse_filters, display_results, _quiet_output, _detect_vector_backend, _setup_logging.
# Deps: re, sys, os, contextlib, logging, pathlib, src.retrieval.rag_chain, src.platform.validation
# @end-summary
"""Core query logic for the RAG system — logging, filters, display, output suppression."""

import contextlib
import logging
import os
import re
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.platform.cli_log_formatting import (
    build_level_badges,
    build_logger_style,
    style_log_message,
)
from src.platform.validation import validate_filter_value

# ── ANSI colors ──────────────────────────────────────────────────────────────
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
RED = "\033[31m"
WHITE = "\033[97m"

B_CYAN = f"{BOLD}{CYAN}"
B_GREEN = f"{BOLD}{GREEN}"
B_YELLOW = f"{BOLD}{YELLOW}"
B_BLUE = f"{BOLD}{BLUE}"
B_MAGENTA = f"{BOLD}{MAGENTA}"
B_RED = f"{BOLD}{RED}"
B_WHITE = f"{BOLD}{WHITE}"

# ── Logging setup ────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent / "logs"
LOG_FILE = LOG_DIR / "rag_query.log"
_verbose_mode = False

_PALETTE = {
    "RESET": RESET,
    "DIM": DIM,
    "CYAN": CYAN,
    "GREEN": GREEN,
    "YELLOW": YELLOW,
    "BLUE": BLUE,
    "MAGENTA": MAGENTA,
    "RED": RED,
    "WHITE": WHITE,
    "B_CYAN": B_CYAN,
    "B_GREEN": B_GREEN,
    "B_YELLOW": B_YELLOW,
    "B_BLUE": B_BLUE,
    "B_MAGENTA": B_MAGENTA,
    "B_RED": B_RED,
    "B_WHITE": B_WHITE,
}
_LOGGER_STYLE = build_logger_style(_PALETTE)
_LEVEL_BADGE = build_level_badges(_PALETTE)


class StyledConsoleHandler(logging.Handler):
    """Pretty-prints rag.* logs to the console. Silences everything else."""

    def emit(self, record):
        # Only show rag.* namespace on console (unless verbose)
        if not record.name.startswith("rag.") and not _verbose_mode:
            return
        # In non-verbose mode, skip DEBUG
        if not _verbose_mode and record.levelno < logging.INFO:
            return

        badge = _LEVEL_BADGE.get(record.levelname, f"{DIM}?{RESET}")
        label, color = _LOGGER_STYLE.get(
            record.name, (f"{DIM}⟡ {record.name}{RESET}", "")
        )
        msg = record.getMessage()

        # Parse out useful info for common patterns
        msg = style_log_message(record.name, msg, _PALETTE)

        print(f"    {badge}  {label}  {DIM}{msg}{RESET}")

    def _verbose_emit(self, record):
        """Show all loggers in verbose mode."""
        badge = _LEVEL_BADGE.get(record.levelname, f"{DIM}?{RESET}")
        msg = record.getMessage()
        print(f"    {badge}  {DIM}{record.name}{RESET}  {msg}")

def _setup_logging():
    """Configure logging: file gets everything, console gets styled rag.* only."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Root logger captures all
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Remove any existing handlers (from basicConfig etc.)
    root.handlers.clear()

    # File handler — captures everything for debugging
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    ))
    root.addHandler(file_handler)

    # Styled console handler — only rag.* namespace by default
    console_handler = StyledConsoleHandler()
    console_handler.setLevel(logging.DEBUG)
    root.addHandler(console_handler)


_setup_logging()
logger = logging.getLogger("rag.query_cli")


@contextlib.contextmanager
def _quiet_output():
    """Suppress stdout/stderr at OS fd level to catch subprocess output (Weaviate Go JSON, tqdm).

    Python-level sys.stderr redirect does NOT catch output from child processes
    that write directly to inherited file descriptors. os.dup2 redirects the
    underlying fd so even Go/C subprocesses writing to fd 1/2 are silenced.
    Python's own sys.stdout is re-pointed to a copy of the original fd so our
    print() calls still reach the terminal.
    """
    import warnings

    # Save original OS-level file descriptors
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    # Create a Python file object from the saved stdout fd for our own output
    saved_stdout = os.fdopen(os.dup(saved_stdout_fd), "w")
    old_sys_stdout = sys.stdout
    old_sys_stderr = sys.stderr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # Redirect OS-level fds to /dev/null — catches subprocess output
            os.dup2(devnull_fd, 1)
            os.dup2(devnull_fd, 2)
            # Point Python's sys.stdout to the saved copy so print() still works
            sys.stdout = saved_stdout
            sys.stderr = open(os.devnull, "w")
            yield
        finally:
            # Restore OS-level fds
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            sys.stdout = old_sys_stdout
            sys.stderr = old_sys_stderr
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
            os.close(devnull_fd)
            saved_stdout.close()


def _detect_vector_backend() -> str:
    """Auto-detect the vector store backend from installed packages."""
    try:
        import weaviate
        return "Weaviate"
    except ImportError:
        pass
    try:
        import chromadb
        return "ChromaDB"
    except ImportError:
        pass
    try:
        import qdrant_client
        return "Qdrant"
    except ImportError:
        pass
    try:
        import pinecone
        return "Pinecone"
    except ImportError:
        pass
    return "Vector DB"


# ── Filter parsing ───────────────────────────────────────────────────────────

# Filter prefix patterns:
# - source:filename.txt
# - section:Heading
# - source:"path with spaces/file.md"
# - section:"Clock Domain Crossing"
_FILTER_PAT = re.compile(
    r'\b(source|section|deep):(?:"([^"]+)"|(\S+))\s*',
    re.IGNORECASE,
)


def parse_filters(raw_query: str) -> tuple:
    """Extract filter prefixes from query, return (clean_query, filters_dict).

    Supported filters:
        source:<filename>   — filter by source document
        section:<heading>   — filter by section heading
        deep:<bool>         — toggle deep_research (true/false/yes/no/1/0)

    Example:
        "source:sample_doc_3.txt what is RAG?"
        → ("what is RAG?", {"source_filter": "sample_doc_3.txt"})
        "deep:true compare X to Y"
        → ("compare X to Y", {"deep_research": True})
    """
    filters = {}

    def _replace(m):
        key = m.group(1).lower()
        value = m.group(2) if m.group(2) is not None else m.group(3)
        if key == "source":
            filters["source_filter"] = value
        elif key == "section":
            filters["heading_filter"] = value
        elif key == "deep":
            filters["deep_research"] = value.lower() in ("1", "true", "yes", "on")
        return ""

    clean = _FILTER_PAT.sub(_replace, raw_query).strip()
    if "source_filter" in filters:
        filters["source_filter"] = validate_filter_value("source_filter", filters["source_filter"])
    if "heading_filter" in filters:
        filters["heading_filter"] = validate_filter_value("heading_filter", filters["heading_filter"])
    return clean, filters


def _truncate(text: str, max_len: int) -> str:
    """Truncate text for display."""
    text = text.replace("\n", " ")
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _display_hyde_round(r: dict, idx: int) -> None:
    """Print one HyDE round (hypothetical + terms + fallback badge), indented."""
    rn = r.get("round", idx)
    aspect = (r.get("target_aspect") or "").strip()
    fb = f" {B_YELLOW}[literal fallback]{RESET}" if r.get("fell_back") else ""
    head = f"Round {rn}" + (f" — {aspect}" if aspect else "")
    print(f"    {DIM}{head}{RESET}{fb}")
    hyp = (r.get("hypothetical_answer") or "").strip()
    for line in hyp.split("\n"):
        if line.strip():
            print(f"      {DIM}{line}{RESET}")
    terms = r.get("search_terms") or []
    if terms:
        print(f"      {DIM}search terms: {', '.join(str(t) for t in terms)}{RESET}")


def _display_query_processing(response) -> None:
    """CLI parity with the web 'Query processing' panel: show the HyDE / deep-
    research detail the retriever produced BEFORE generation. Reads the same
    ``response.metadata`` the web panel consumes. Silent when nothing to show."""
    md = getattr(response, "metadata", None) or {}
    agentic = md.get("agentic_retrieval") or {}
    dr = md.get("deep_research") or {}
    rounds = agentic.get("hyde_rounds") or []
    tried = agentic.get("tried_hyde") or []
    has_agentic = bool(agentic.get("rounds_run") or rounds or tried)
    has_dr = bool(dr.get("decomposed") or dr.get("topic_count"))
    if not has_agentic and not has_dr:
        return

    print(f"\n  {B_WHITE}Query processing{RESET}")
    if has_agentic:
        stats = []
        if agentic.get("rounds_run"):
            stats.append(f"{agentic['rounds_run']} round(s)")
        if agentic.get("hyde_variants_tried"):
            stats.append(f"{agentic['hyde_variants_tried']} variant(s)")
        if agentic.get("kept_count") is not None:
            stats.append(f"{agentic['kept_count']} kept")
        if agentic.get("backfilled"):
            stats.append(f"{agentic['backfilled']} backfilled")
        if agentic.get("ranker"):
            stats.append(f"ranker: {agentic['ranker']}")
        if agentic.get("stop_reason"):
            stats.append(f"stop: {agentic['stop_reason']}")
        if agentic.get("elapsed_ms"):
            stats.append(f"{agentic['elapsed_ms'] / 1000:.1f}s")
        print(f"    {B_CYAN}HyDE{RESET} {DIM}{' · '.join(stats)}{RESET}")
        hf = agentic.get("hyde_failures") or 0
        if hf:
            print(
                f"    {B_YELLOW}⚠ HyDE generation failed on {hf} round(s) — "
                f"fell back to literal-query retrieval{RESET}"
            )
        if rounds:
            for i, r in enumerate(rounds, 1):
                _display_hyde_round(r if isinstance(r, dict) else {}, i)
        elif tried:
            for i, t in enumerate(tried, 1):
                _display_hyde_round({"hypothetical_answer": t}, i)
    if has_dr:
        stats = ["decomposed" if dr.get("decomposed") else "unified"]
        if dr.get("topic_count"):
            stats.append(f"{dr['topic_count']} topic(s)")
        if dr.get("iteration_count"):
            stats.append(f"{dr['iteration_count']} iteration(s)")
        if dr.get("node_count"):
            stats.append(f"{dr['node_count']} node(s)")
        if dr.get("llm_call_count"):
            stats.append(f"{dr['llm_call_count']} LLM call(s)")
        print(f"    {B_CYAN}Deep research{RESET} {DIM}{' · '.join(stats)}{RESET}")


def display_results(response, elapsed: float) -> None:
    """Pretty-print RAG results."""
    print()
    print(f"  {DIM}{'─' * 72}{RESET}")

    # Query metadata
    print(f"  {B_WHITE}Query{RESET}         {response.query}")
    print(f"  {DIM}Processed{RESET}     {response.processed_query}")

    # Confidence with color coding
    conf = response.query_confidence
    if conf >= 0.7:
        conf_color = B_GREEN
    elif conf >= 0.4:
        conf_color = B_YELLOW
    else:
        conf_color = B_RED
    print(f"  {DIM}Confidence{RESET}    {conf_color}{conf:.0%}{RESET}")

    # Action badge
    action_colors = {
        "answer": B_GREEN,
        "ask_user": B_YELLOW,
        "search": B_CYAN,
    }
    ac = action_colors.get(response.action, DIM)
    print(f"  {DIM}Action{RESET}        {ac}{response.action}{RESET}")

    if response.kg_expanded_terms:
        terms = ", ".join(response.kg_expanded_terms[:5])
        print(f"  {DIM}KG expansion{RESET}  {B_BLUE}{terms}{RESET}")

    # Token budget / context window usage
    tb = getattr(response, "token_budget", None)
    if tb:
        pct = tb.usage_percent if hasattr(tb, "usage_percent") else (tb.get("usage_percent", 0) if isinstance(tb, dict) else 0)
        inp = tb.input_tokens if hasattr(tb, "input_tokens") else (tb.get("input_tokens", 0) if isinstance(tb, dict) else 0)
        ctx = tb.context_length if hasattr(tb, "context_length") else (tb.get("context_length", 0) if isinstance(tb, dict) else 0)
        mdl = tb.model_name if hasattr(tb, "model_name") else (tb.get("model_name", "") if isinstance(tb, dict) else "")

        if pct >= 90:
            pct_color = B_RED
        elif pct >= 70:
            pct_color = B_YELLOW
        else:
            pct_color = B_GREEN
        print(f"  {DIM}Context{RESET}       {pct_color}{pct:.0f}%{RESET} {DIM}({inp}/{ctx} tokens, {mdl}){RESET}")

        # Detailed breakdown
        bd = tb.breakdown if hasattr(tb, "breakdown") else (tb.get("breakdown") if isinstance(tb, dict) else None)
        if bd:
            sp = bd.system_prompt if hasattr(bd, "system_prompt") else (bd.get("system_prompt", 0) if isinstance(bd, dict) else 0)
            mem = bd.memory_context if hasattr(bd, "memory_context") else (bd.get("memory_context", 0) if isinstance(bd, dict) else 0)
            chk = bd.retrieval_chunks if hasattr(bd, "retrieval_chunks") else (bd.get("retrieval_chunks", 0) if isinstance(bd, dict) else 0)
            qry = bd.user_query if hasattr(bd, "user_query") else (bd.get("user_query", 0) if isinstance(bd, dict) else 0)
            oh = bd.template_overhead if hasattr(bd, "template_overhead") else (bd.get("template_overhead", 0) if isinstance(bd, dict) else 0)
            print(f"                {DIM}system:{sp}  memory:{mem}  chunks:{chk}  query:{qry}  overhead:{oh}{RESET}")

        # Actual tokens from LLM response
        apt = tb.actual_prompt_tokens if hasattr(tb, "actual_prompt_tokens") else (tb.get("actual_prompt_tokens", 0) if isinstance(tb, dict) else 0)
        act = tb.actual_completion_tokens if hasattr(tb, "actual_completion_tokens") else (tb.get("actual_completion_tokens", 0) if isinstance(tb, dict) else 0)
        if apt:
            print(f"  {DIM}Tokens{RESET}        {DIM}actual: {apt} in + {act} out = {apt + act} total{RESET}")

    print(f"  {DIM}{'─' * 72}{RESET}")

    # Query-processing detail (HyDE / deep-research) BEFORE the answer — CLI
    # parity with the web console's "Query processing" panel.
    _display_query_processing(response)

    if response.action == "ask_user":
        # Per-reason hint badge (CLI parity with the web console). The
        # reason enum is additive to ``action``: legacy code that only
        # checks the action string keeps working; reason-aware UIs can
        # show a typed prefix.
        reason = getattr(response, "ask_user_reason", None)
        reason_labels = {
            "sanitizer_reject": "empty or invalid query",
            "injection_blocked": "query blocked by safety rails",
            "vague_query": "query too vague to retrieve",
            "budget_exhausted": "retrieval timeout",
            "no_results": "no matching documents",
        }
        label = reason_labels.get(reason or "", "ask_user")
        print(
            f"\n  {B_YELLOW}?{RESET} [{label}] "
            f"{response.clarification_message}"
        )
        return

    # Show generated answer prominently
    if response.generated_answer:
        print()
        print(f"  {B_GREEN}✦ Answer{RESET} {DIM}({elapsed:.1f}s){RESET}\n")
        # Indent each line of the answer
        for line in response.generated_answer.split("\n"):
            print(f"  {line}")
        print()
        # Always show LLM confidence indicator
        llm_conf = getattr(response, "llm_confidence", None)
        if llm_conf:
            if llm_conf == "high":
                conf_color = B_GREEN
            elif llm_conf == "medium":
                conf_color = B_YELLOW
            else:
                conf_color = B_RED
            print(f"  {DIM}{'─' * 72}{RESET}")
            print(f"  {conf_color}Confidence: {llm_conf}{RESET}")
        print(f"  {DIM}{'─' * 72}{RESET}")

        # CLI/UI parity: surface the same advisory the web chip shows.
        dr_sugg = getattr(response, "dr_suggestion", None)
        if isinstance(dr_sugg, dict) and dr_sugg.get("suggest"):
            print(
                f"  {B_YELLOW}💡 Multi-topic question detected — "
                f"try `deep:true` for richer coverage.{RESET}"
            )

        # CLI/UI parity: the same "you might also ask…" follow-up questions the
        # web console renders as clickable chips.
        follow_ups = getattr(response, "suggested_questions", None)
        if follow_ups:
            print(f"\n  {B_WHITE}You might also ask:{RESET}")
            for q in follow_ups:
                print(f"    {B_CYAN}›{RESET} {q}")

    if not response.results:
        print(f"\n  {B_YELLOW}⚠{RESET} No results found.\n")
        return

    # Retrieved chunks
    print(f"\n  {B_WHITE}Top {len(response.results)} retrieved chunks{RESET}\n")
    for i, result in enumerate(response.results, 1):
        score_color = B_GREEN if result.score >= 0.5 else (B_YELLOW if result.score >= 0.2 else DIM)
        print(f"  {B_CYAN}#{i}{RESET}  {score_color}score: {result.score:.4f}{RESET}  {DIM}│{RESET}  {B_MAGENTA}{result.metadata.get('source', 'unknown')}{RESET}")
        source_uri = result.metadata.get("citation_source_uri") or result.metadata.get("source_uri", "")
        if source_uri:
            print(f"      {DIM}location:{RESET} {source_uri}")
        origin = result.metadata.get("retrieval_text_origin", "")
        if origin:
            print(f"      {DIM}retrieval_text:{RESET} {origin}")
        section = result.metadata.get("section_path", "")
        if section:
            print(f"      {DIM}section:{RESET} {section}")
        print(f"      {DIM}{_truncate(result.text, 200)}{RESET}")
        print()
