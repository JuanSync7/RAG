#!/usr/bin/env python3
# @summary
# Thin CLI client that connects to the RAG API server. Same REPL experience
# as cli.py, but no model loading — queries go over HTTP to the server.
# Renders the turn-loop SSE activity log (turn_action/hyde_query/
# retrieve_result/judge_verdict/deep_study/llm_call/draft/gate/clarify) as
# compact one-liners with dimmed draft streaming, and carries the tri-state
# turn_loop request override (RAG_CLI_TURN_LOOP env / /turn-loop toggle).
# Exports: main
# Deps: urllib.request, orjson, sys, src.platform (command runtime),
#       src.retrieval.pipeline.turn_loop (event-name constants only)
# @end-summary
"""CLI client for the RAG API server.

Same interactive experience as the local CLI, but queries are sent over
HTTP to the FastAPI server → Temporal → preloaded worker. No torch,
no transformers, no GPU needed in this process. Starts instantly.

Usage:
    python -m server.cli_client
    python -m server.cli_client --server http://localhost:8000
"""

import orjson
import logging
import os
import re
import readline
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.platform import (
    get_input_with_menu,
    setup_tab_completion,
)
from src.platform import (
    MODE_SERVER_CLI,
    list_command_specs,
)
from src.platform import dispatch_slash_command
# Single source of truth for the turn-loop SSE event vocabulary (schema-only
# import, no side effects) — keeps the CLI renderer 1:1 with the server.
from src.retrieval.pipeline.turn_loop import TurnEventType

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("rag.cli_client")
logger.setLevel(logging.DEBUG)
_fh = logging.FileHandler(_LOG_DIR / "cli_client.log")
_fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_fh)

# ANSI colors
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
_BG_SEL = "\033[48;5;237m"


_API_PORT = os.environ.get("RAG_API_PORT", "8000")
DEFAULT_SERVER = os.environ.get("RAG_API_URL", f"http://localhost:{_API_PORT}")
API_KEY = os.environ.get("RAG_API_KEY", "").strip()
BEARER_TOKEN = os.environ.get("RAG_BEARER_TOKEN", "").strip()
_FILTER_PAT = re.compile(r"\b(source|section):(\S+)\s*", re.IGNORECASE)
_CURRENT_SERVER = DEFAULT_SERVER
_verbose_mode = False
_ALLOW_LOCAL_ADMIN_COMMANDS = False
_CURRENT_CONVERSATION_ID: str | None = None
_MEMORY_ENABLED = True


def _parse_tristate_env(raw: str) -> bool | None:
    """Parse an env override into a tri-state flag (''/unset -> None)."""
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


# Tri-state per-request override of RAG_TURN_LOOP_ENABLED (CLI side of the
# buildQueryBody parity contract): None = server default, True/False = force.
_TURN_LOOP_MODE: bool | None = _parse_tristate_env(
    os.environ.get("RAG_CLI_TURN_LOOP", "")
)

# Command registry for parity with cli.py UX
# tuple layout: (handler, description, hidden, admin_only)
_COMMANDS: dict[str, tuple[Callable[[str], None], str, bool, bool]] = {}
_SERVER_SPECS = {
    spec.name: spec
    for spec in list_command_specs(
        MODE_SERVER_CLI,
        include_hidden=True,
        allow_admin=True,
    )
}


def _register_command(
    name: str,
    description: str,
    *,
    hidden: bool = False,
    admin_only: bool = False,
):
    def decorator(func):
        _COMMANDS[name] = (func, description, hidden, admin_only)
        return func
    return decorator


def _visible_registry() -> dict[str, tuple]:
    """Commands shown in / menu and tab completion."""
    return {
        name: (handler, description)
        for name, (handler, description, hidden, admin_only) in _COMMANDS.items()
        if not hidden and (not admin_only or _ALLOW_LOCAL_ADMIN_COMMANDS)
    }


_BOX_W = 50


def _get_menu_items(filter_text: str = "") -> list[tuple[str, str]]:
    visible = _visible_registry()
    ft = filter_text.lower()
    return [
        (name, desc)
        for name, (_, desc) in visible.items()
        if name.lower().startswith(ft)
    ]


def _get_input(prompt: str) -> str:
    return get_input_with_menu(
        prompt,
        _visible_registry(),
        box_width=_BOX_W,
        dim=DIM,
        reset=RESET,
        bold_cyan=B_CYAN,
        bg_sel=_BG_SEL,
    )


def _setup_tab_completion():
    setup_tab_completion(_visible_registry())


def _server_command_handlers() -> dict[str, Callable[[str], None]]:
    """Expose server CLI command handlers for shared dispatch."""
    return {
        name: handler
        for name, (handler, _, _, admin_only) in _COMMANDS.items()
        if not admin_only or _ALLOW_LOCAL_ADMIN_COMMANDS
    }


def parse_filters(raw_query: str) -> tuple[str, dict]:
    """Extract source:/section: filters from query text."""
    filters = {}

    def _replace(m):
        key = m.group(1).lower()
        value = m.group(2)
        if key == "source":
            filters["source_filter"] = value
        elif key == "section":
            filters["heading_filter"] = value
        return ""

    clean = _FILTER_PAT.sub(_replace, raw_query).strip()
    return clean, filters


def _truncate(text: str, max_len: int) -> str:
    text = text.replace("\n", " ")
    return text[:max_len - 3] + "..." if len(text) > max_len else text


def send_query_stream(server: str, query: str, filters: dict):
    """Stream a query via SSE. Yields (event_type, data_dict) tuples."""
    payload = {
        "query": query,
        **filters,
        "conversation_id": _CURRENT_CONVERSATION_ID,
        "memory_enabled": _MEMORY_ENABLED,
    }
    if _TURN_LOOP_MODE is not None:
        payload["turn_loop"] = _TURN_LOOP_MODE
    data = orjson.dumps(payload)
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    elif BEARER_TOKEN:
        headers["Authorization"] = f"Bearer {BEARER_TOKEN}"
    req = urllib.request.Request(
        f"{server}/query/stream",
        data=data,
        headers=headers,
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=120)
    current_event = None
    try:
        for raw_line in resp:
            line = raw_line.decode("utf-8").rstrip("\n")
            if line.startswith("event: "):
                current_event = line[7:]
            elif line.startswith("data: ") and current_event:
                yield current_event, orjson.loads(line[6:])
                current_event = None
    finally:
        resp.close()


def check_server(server: str, retries: int = 5, backoff: float = 1.0) -> dict | None:
    """Check if the API server is reachable, retrying on failure.

    Returns the parsed health JSON on success, or None if completely unreachable.
    """
    for attempt in range(retries):
        try:
            req = urllib.request.Request(f"{server}/health", method="GET")
            if API_KEY:
                req.add_header("X-API-Key", API_KEY)
            elif BEARER_TOKEN:
                req.add_header("Authorization", f"Bearer {BEARER_TOKEN}")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = orjson.loads(resp.read())
                logger.info(
                    "Health check OK (attempt %d/%d): %s", attempt + 1, retries, data
                )
                return data
        except Exception as exc:
            logger.debug(
                "Health check attempt %d/%d failed: %s", attempt + 1, retries, exc
            )
            if attempt < retries - 1:
                time.sleep(backoff)
    logger.warning("Server unreachable after %d attempts", retries)
    return None


def _request_server_json(server: str, method: str, path: str, payload: dict | None = None) -> dict:
    data = orjson.dumps(payload) if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    elif BEARER_TOKEN:
        headers["Authorization"] = f"Bearer {BEARER_TOKEN}"
    req = urllib.request.Request(
        f"{server.rstrip('/')}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
        return orjson.loads(raw) if raw.strip() else {}


def _list_conversations(server: str, limit: int = 20) -> list[dict]:
    return _request_server_json(server, "GET", f"/conversations?limit={max(1, min(limit, 100))}")


def _new_conversation(server: str, title: str = "New conversation") -> dict:
    return _request_server_json(server, "POST", "/conversations/new", {"title": title})


def _conversation_history(server: str, conversation_id: str, limit: int = 20) -> dict:
    return _request_server_json(
        server,
        "GET",
        f"/conversations/{conversation_id}/history?limit={max(1, min(limit, 200))}",
    )


def _compact_conversation(server: str, conversation_id: str) -> dict:
    return _request_server_json(
        server,
        "POST",
        f"/conversations/{conversation_id}/compact",
        {"conversation_id": conversation_id},
    )


def _delete_conversation(server: str, conversation_id: str) -> dict:
    return _request_server_json(
        server,
        "DELETE",
        f"/conversations/{conversation_id}",
    )


def display_retrieval(response: dict) -> None:
    """Display retrieval metadata and chunks (no generated answer — that streams separately)."""
    print()
    print(f"  {DIM}{'─' * 72}{RESET}")
    print(f"  {B_WHITE}Query{RESET}         {response['query']}")
    print(f"  {DIM}Processed{RESET}     {response['processed_query']}")

    conf = response["query_confidence"]
    if conf >= 0.7:
        cc = B_GREEN
    elif conf >= 0.4:
        cc = B_YELLOW
    else:
        cc = B_RED
    print(f"  {DIM}Confidence{RESET}    {cc}{conf:.0%}{RESET}")

    action_colors = {"answer": B_GREEN, "ask_user": B_YELLOW, "search": B_CYAN}
    ac = action_colors.get(response["action"], DIM)
    print(f"  {DIM}Action{RESET}        {ac}{response['action']}{RESET}")

    if response.get("kg_expanded_terms"):
        terms = ", ".join(response["kg_expanded_terms"][:5])
        print(f"  {DIM}KG expansion{RESET}  {B_BLUE}{terms}{RESET}")

    latency = response.get("latency_ms")
    if latency is not None:
        print(f"  {DIM}Retrieval{RESET}     {latency:.0f}ms")

    wf_id = response.get("workflow_id")
    if wf_id:
        logger.debug("Query handled by workflow %s", wf_id)
    conv_id = response.get("conversation_id")
    if conv_id:
        print(f"  {DIM}Conversation{RESET}  {conv_id}")

    # Token budget / context window usage
    tb = response.get("token_budget")
    if tb and isinstance(tb, dict):
        pct = tb.get("usage_percent", 0)
        inp = tb.get("input_tokens", 0)
        ctx = tb.get("context_length", 0)
        mdl = tb.get("model_name", "")

        if pct >= 90:
            pct_color = B_RED
        elif pct >= 70:
            pct_color = B_YELLOW
        else:
            pct_color = B_GREEN
        print(f"  {DIM}Context{RESET}       {pct_color}{pct:.0f}%{RESET} {DIM}({inp}/{ctx} tokens, {mdl}){RESET}")

        bd = tb.get("breakdown")
        if bd and isinstance(bd, dict):
            sp = bd.get("system_prompt", 0)
            mem = bd.get("memory_context", 0)
            chk = bd.get("retrieval_chunks", 0)
            qry = bd.get("user_query", 0)
            oh = bd.get("template_overhead", 0)
            print(f"                {DIM}system:{sp}  memory:{mem}  chunks:{chk}  query:{qry}  overhead:{oh}{RESET}")

        apt = tb.get("actual_prompt_tokens", 0)
        act = tb.get("actual_completion_tokens", 0)
        if apt:
            print(f"  {DIM}Tokens{RESET}        {DIM}actual: {apt} in + {act} out = {apt + act} total{RESET}")

    print(f"  {DIM}{'─' * 72}{RESET}")

    if response["action"] == "ask_user":
        # Reason-typed hint (parity with query.py CLI and the web console).
        reason = response.get("ask_user_reason")
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
            f"{response.get('clarification_message', '')}"
        )
        return

    results = response.get("results", [])
    if not results:
        print(f"\n  {B_YELLOW}⚠{RESET} No results found.\n")
        return

    # Concise "Sources:" summary — the distinct documents that fed the answer
    # (CLI/UI parity with the web references panel's Sources header). Basename
    # only, deduped, in first-seen order.
    _seen: set[str] = set()
    _used: list[str] = []
    for r in results:
        name = str((r.get("metadata", {}) or {}).get("source") or "unknown").split("/")[-1]
        if name not in _seen:
            _seen.add(name)
            _used.append(name)
    if _used:
        print(f"\n  {B_GREEN}Sources:{RESET} {B_MAGENTA}{', '.join(_used)}{RESET}")

    print(f"\n  {B_WHITE}Top {len(results)} retrieved chunks{RESET}\n")
    for i, r in enumerate(results, 1):
        score = r["score"]
        sc = B_GREEN if score >= 0.5 else (B_YELLOW if score >= 0.2 else DIM)
        source = r.get("metadata", {}).get("source", "unknown")
        print(f"  {B_CYAN}#{i}{RESET}  {sc}score: {score:.4f}{RESET}  {DIM}│{RESET}  {B_MAGENTA}{source}{RESET}")
        source_uri = (
            r.get("metadata", {}).get("citation_source_uri")
            or r.get("metadata", {}).get("source_uri", "")
        )
        if source_uri:
            print(f"      {DIM}location:{RESET} {source_uri}")
        origin = r.get("metadata", {}).get("retrieval_text_origin", "")
        if origin:
            print(f"      {DIM}retrieval_text:{RESET} {origin}")
        section = r.get("metadata", {}).get("section_path", "")
        if section:
            print(f"      {DIM}section:{RESET} {section}")
        print(f"      {DIM}{_truncate(r['text'], 200)}{RESET}")
        print()


def _close_turn_draft(state: dict) -> None:
    """Close an open dimmed draft run so subsequent output renders normally."""
    if state.get("draft_open"):
        sys.stdout.write(f"{RESET}\n")
        sys.stdout.flush()
        state["draft_open"] = False


def _render_turn_activity(event_type: str, data: dict, state: dict) -> None:
    """Render one turn-loop SSE event as a compact activity-log line.

    CLI side of the console activity-log parity contract (design §8): every
    typed loop event gets a dimmed one-liner; draft deltas stream dimmed under
    a per-attempt header; clarify renders the question with its hint /
    scoping-question lists.
    """
    if not state.get("started"):
        # Clear the "Querying server..." spinner and open the activity log.
        print(f"\r  {B_CYAN}⟳ Turn loop{RESET}                              ")
        state["started"] = True

    if event_type == TurnEventType.DRAFT:
        attempt = data.get("attempt", 0)
        delta = str(data.get("text_delta") or data.get("reasoning_delta") or "")
        if not state.get("draft_open") or state.get("draft_attempt") != attempt:
            _close_turn_draft(state)
            print(f"    {DIM}✍ draft {attempt}{RESET}")
            sys.stdout.write(f"    {DIM}")
            state["draft_open"] = True
            state["draft_attempt"] = attempt
        sys.stdout.write(delta.replace("\n", "\n    "))
        sys.stdout.flush()
        return

    _close_turn_draft(state)

    if event_type == TurnEventType.TURN_ACTION:
        line = (
            f"[turn {data.get('index', 0)}] {data.get('action', '?')}"
            f" — {_truncate(str(data.get('reason', '')), 80)}"
            f" (confidence {float(data.get('confidence', 0.0)):.2f})"
        )
    elif event_type == TurnEventType.HYDE_QUERY:
        aspect = data.get("target_aspect")
        line = (
            f"[round {data.get('round', 0)}] hyde: "
            f"\"{_truncate(str(data.get('hypothetical_answer', '')), 70)}\""
            + (f" (aspect: {aspect})" if aspect else "")
        )
    elif event_type == TurnEventType.RETRIEVE_RESULT:
        line = (
            f"[round {data.get('round', 0)}] retrieve → "
            f"+{data.get('added', 0)} chunks "
            f"({data.get('dup', 0)} dup, pool {data.get('pool_size', 0)})"
        )
    elif event_type == TurnEventType.JUDGE_VERDICT:
        # missing_information is free text on the wire (JudgeVerdictEvent);
        # tolerate a legacy list shape by joining it.
        missing = data.get("missing_information") or ""
        if isinstance(missing, (list, tuple)):
            missing = "; ".join(str(gap) for gap in missing)
        line = (
            f"[round {data.get('round', 0)}] judge: kept {data.get('kept', 0)}, "
            f"confidence {float(data.get('confidence', 0.0)):.2f}, "
            + ("sufficient" if data.get("sufficient") else "insufficient")
            + (f" — missing: {_truncate(str(missing), 60)}" if missing else "")
        )
    elif event_type == TurnEventType.DEEP_STUDY:
        line = (
            f"deep-study {data.get('title') or data.get('document_id', '?')} "
            f"window {data.get('window', 0)}/{data.get('of_windows', 0)}: "
            f"{_truncate(str(data.get('notes_preview', '')), 60)}"
        )
    elif event_type == TurnEventType.LLM_CALL:
        tokens = ""
        if data.get("prompt_tokens") or data.get("completion_tokens"):
            tokens = f" ({data.get('prompt_tokens', 0)}+{data.get('completion_tokens', 0)} tok)"
        line = f"llm {data.get('purpose', '?')} [{data.get('alias', '?')}] {data.get('ms', 0)}ms{tokens}"
    elif event_type == TurnEventType.GATE:
        verdict = "PASS" if data.get("passed") else "RETRY"
        weakest = data.get("weakest")
        line = (
            f"gate attempt {data.get('attempt', 0)}: "
            f"{float(data.get('score', 0.0)):.2f} vs {float(data.get('threshold', 0.0)):.2f}"
            f" → {verdict}" + (f" (weakest: {weakest})" if weakest else "")
        )
    elif event_type == TurnEventType.CLARIFY:
        print(f"    {B_YELLOW}? {data.get('question', '')}{RESET}")
        for i, hint in enumerate(data.get("hints") or [], start=1):
            print(f"      {DIM}{i}) {hint}{RESET}")
        for sq in data.get("scoping_questions") or []:
            print(f"      {DIM}or ask: {sq}{RESET}")
        return
    else:
        line = f"{event_type}: {_truncate(str(data), 80)}"
    print(f"    {DIM}{line}{RESET}")


def _display_turn_loop_summary(done_data: dict | None) -> None:
    """Print the compact turn-loop summary carried in done.metadata.turn_loop."""
    tl = ((done_data or {}).get("metadata") or {}).get("turn_loop")
    if not isinstance(tl, dict):
        return
    print(
        f"  {DIM}turn loop: {tl.get('actions', 0)} actions │ "
        f"{tl.get('llm_calls', 0)} llm calls │ "
        f"confidence {float(tl.get('confidence', 0.0)):.2f} │ "
        f"stop: {tl.get('stop_reason', '?')}{RESET}"
    )


def _display_stage_timings(done_data: dict | None, retrieval_data: dict | None) -> None:
    """Verbose stage-level timing split for retrieval/generation."""
    if not _verbose_mode:
        return

    stage_timings = []
    if done_data and done_data.get("stage_timings"):
        stage_timings = done_data.get("stage_timings", [])
    elif retrieval_data and retrieval_data.get("stage_timings"):
        stage_timings = retrieval_data.get("stage_timings", [])

    if not stage_timings:
        print()
        print(f"  {B_YELLOW}Stage timings (verbose){RESET}")
        print(f"  {DIM}No per-stage timing data returned by backend for this query.{RESET}")
        print(f"  {DIM}Tip: restart API/worker processes to ensure latest code is running.{RESET}")
        return

    print()
    print(f"  {B_WHITE}Stage timings (verbose){RESET}")
    print(f"  {DIM}{'-' * 72}{RESET}")
    for stage in stage_timings:
        bucket = str(stage.get("bucket", "other"))
        name = str(stage.get("stage", "unknown"))
        ms = float(stage.get("ms", 0.0))
        bucket_color = B_CYAN if bucket == "retrieval" else (B_GREEN if bucket == "generation" else DIM)
        print(f"  {bucket_color}{bucket:10}{RESET} {DIM}{name:24}{RESET} {ms:7.1f}ms")

    totals = {}
    if done_data and done_data.get("timing_totals"):
        totals = done_data.get("timing_totals", {})
    if not totals:
        bucket_totals: dict[str, float] = {}
        for stage in stage_timings:
            bucket = str(stage.get("bucket", "other"))
            bucket_totals[bucket] = bucket_totals.get(bucket, 0.0) + float(stage.get("ms", 0.0))
        totals = {f"{bucket}_ms": round(ms, 1) for bucket, ms in bucket_totals.items()}
        totals["total_ms"] = round(sum(bucket_totals.values()), 1)
    if totals:
        retrieval_total = float(totals.get("retrieval_ms", 0.0))
        generation_total = float(totals.get("generation_ms", 0.0))
        total = float(totals.get("total_ms", retrieval_total + generation_total))
        print(f"  {DIM}{'-' * 72}{RESET}")
        print(
            f"  {DIM}stage totals:{RESET} "
            f"{B_CYAN}retrieval {retrieval_total:.1f}ms{RESET} {DIM}|{RESET} "
            f"{B_GREEN}generation {generation_total:.1f}ms{RESET} {DIM}|{RESET} "
            f"{B_WHITE}total {total:.1f}ms{RESET}"
        )


def clear_screen():
    os.system("clear" if os.name != "nt" else "cls")


def print_banner(server: str):
    clear_screen()
    print()
    chip_lines = [
        "                    ██      ██      ██      ██      ██      ██      ██",
        "                    ║║      ║║      ║║      ║║      ║║      ║║      ║║",
        "           ┏━━━━━━━━╩╩━━━━━━╩╩━━━━━━╩╩━━━━━━╩╩━━━━━━╩╩━━━━━━╩╩━━━━━━╩╩━━━━━━━━━┓",
        "           ┃░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░┃│",
        "   ██━━━━━━┨░░ · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ░░┠━━━━━━██",
        "           ┃░░    ┏╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍┓   ░░┃│",
        "   ██━━━━━━┨░░    ╏            ██████╗   █████╗   ██████╗                ╏   ░░┠━━━━━━██",
        "           ┃░░    ╏            ██╔══██╗ ██╔══██╗ ██╔════╝                ╏   ░░┃│",
        "   ██━━━━━━┨░░    ╏            ██████╔╝ ███████║ ██║  ███╗               ╏   ░░┠━━━━━━██",
        "           ┃░░    ╏            ██╔══██╗ ██╔══██║ ██║   ██║               ╏   ░░┃│",
        "   ██━━━━━━┨░░    ╏            ██║  ██║ ██║  ██║ ╚██████╔╝               ╏   ░░┠━━━━━━██",
        "           ┃░░    ╏            ╚═╝  ╚═╝ ╚═╝  ╚═╝  ╚═════╝                ╏   ░░┃│",
        "   ██━━━━━━┨░░    ┗╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍┛   ░░┠━━━━━━██",
        "           ┃░░ · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ░░┃│",
        "           ┃░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░┃│",
        "           ┗━━━━━━━━╦╦━━━━━━╦╦━━━━━━╦╦━━━━━━╦╦━━━━━━╦╦━━━━━━╦╦━━━━━━╦╦━━━━━━━━━┛│",
        "            └───────║║──────║║──────║║──────║║──────║║──────║║──────║║──────────┘",
        "                    ║║      ║║      ║║      ║║      ║║      ║║      ║║",
        "                    ██      ██      ██      ██      ██      ██      ██",
    ]
    min_col = min((j for line in chip_lines for j, ch in enumerate(line) if ch != " "), default=0)
    max_col = max((j for line in chip_lines for j, ch in enumerate(line) if ch != " "), default=1)
    col_span = max(max_col - min_col, 1)
    for line in chip_lines:
        colored = []
        for j, ch in enumerate(line):
            t = max(0, j - min_col) / col_span
            r = int(255 * t)
            g = int(255 * (1 - t))
            colored.append(f"\033[1;38;2;{r};{g};255m{ch}")
        print("".join(colored) + RESET)
    print(f"{RESET}")
    print(f"{DIM}    ─────────────────────────────────────────────────────────────────────────────{RESET}")
    print(f"{B_WHITE}    RAG Query Engine{RESET}  {DIM}powered by{RESET} {B_MAGENTA}Claude{RESET} {DIM}+{RESET} {B_CYAN}Server Mode{RESET}")
    print(f"{DIM}    ─────────────────────────────────────────────────────────────────────────────{RESET}")
    print()
    print(f"  {DIM}Server:{RESET}  {B_WHITE}{server}{RESET}")
    print(f"  {DIM}Models loaded remotely — queries execute in ~100-500ms{RESET}")
    print()
    print(f"  {B_MAGENTA}[ Retrieval ]{RESET}  {B_BLUE}[ Reranker ]{RESET}  {B_GREEN}[ KG Expansion ]{RESET}  {B_YELLOW}[ Generation ]{RESET}")
    print()
    print(f"  {DIM}Filters: {RESET}{B_WHITE}source:<file>{RESET}{DIM} · {RESET}{B_WHITE}section:<heading>{RESET}")
    print(f"  {DIM}Type {RESET}{B_WHITE}/{RESET} {DIM}to see all commands{RESET}")
    print(f"{DIM}    ─────────────────────────────────────────────────────────────────────────────{RESET}")
    print()


def print_help():
    print(f"""
{B_WHITE}  Commands:{RESET}  {DIM}(type {RESET}{B_WHITE}/{RESET}{DIM} to open menu){RESET}
""")
    for name, (_, desc) in _visible_registry().items():
        padded = f"/{name}".ljust(14)
        print(f"    {B_CYAN}{padded}{RESET} {DIM}{desc}{RESET}")
    if _ALLOW_LOCAL_ADMIN_COMMANDS:
        print(f"\n  {B_YELLOW}Local Admin Commands Enabled{RESET} {DIM}(debug/maintenance){RESET}")
    print(f"""
{B_WHITE}  Filter Syntax:{RESET}
    {B_YELLOW}source:<file>{RESET}      Filter by source document
    {B_YELLOW}section:<heading>{RESET}  Filter by section heading

{B_WHITE}  Logs:{RESET}
    {DIM}Client logs are saved to{RESET} {B_WHITE}{_LOG_DIR / "cli_client.log"}{RESET}

{B_WHITE}  Example Queries:{RESET}
    {DIM}•{RESET} What is retrieval-augmented generation?
    {DIM}•{RESET} source:sample_doc_3.txt what is RAG?
    {DIM}•{RESET} section:Introduction explain the main concepts
    {DIM}•{RESET} How does the embedding pipeline work?
""")


def _print_health_summary(server: str, health: dict | None) -> None:
    """Render compact server health status."""
    if health is None:
        print(f"  {B_RED}✗{RESET} Server unreachable: {B_WHITE}{server}{RESET}")
        print(f"    {DIM}Tip: /set-server <url> to switch endpoint, /health to retry.{RESET}\n")
        return
    temporal_ok = health.get("temporal_connected", False)
    worker_ok = health.get("worker_available", False)
    status = health.get("status", "unknown")
    temporal = f"{B_GREEN}ok{RESET}" if temporal_ok else f"{B_YELLOW}starting{RESET}"
    worker = f"{B_GREEN}ok{RESET}" if worker_ok else f"{B_YELLOW}warming{RESET}"
    print(
        f"  {B_WHITE}{server}{RESET} {DIM}|{RESET} status {B_CYAN}{status}{RESET} "
        f"{DIM}|{RESET} temporal {temporal} {DIM}|{RESET} worker {worker}"
    )
    print()


@_register_command("help", _SERVER_SPECS["help"].description)
def _cmd_help(_: str = ""):
    print_help()


@_register_command("clear", _SERVER_SPECS["clear"].description)
def _cmd_clear(_: str = ""):
    print_banner(_CURRENT_SERVER)


@_register_command("verbose", _SERVER_SPECS["verbose"].description)
def _cmd_verbose(_: str = ""):
    global _verbose_mode
    _verbose_mode = not _verbose_mode
    state = f"{B_GREEN}ON{RESET}" if _verbose_mode else f"{DIM}OFF{RESET}"
    print(f"  {B_CYAN}⟡{RESET} Verbose logging: {state}")
    print()


@_register_command("health", _SERVER_SPECS["health"].description)
def _cmd_health(_: str = ""):
    health = check_server(_CURRENT_SERVER, retries=2, backoff=0.5)
    _print_health_summary(_CURRENT_SERVER, health)


@_register_command("server", _SERVER_SPECS["server"].description)
def _cmd_server(_: str = ""):
    print(f"  {B_WHITE}Current server:{RESET} {_CURRENT_SERVER}\n")


@_register_command(
    "set-server",
    _SERVER_SPECS["set-server"].description,
    hidden=True,
    admin_only=True,
)
def _cmd_set_server(arg: str = ""):
    global _CURRENT_SERVER
    if not arg:
        print(f"  {B_YELLOW}⚠{RESET} Usage: /set-server http://host:port\n")
        return
    candidate = arg.strip().rstrip("/")
    if not (candidate.startswith("http://") or candidate.startswith("https://")):
        print(f"  {B_YELLOW}⚠{RESET} Server URL must start with http:// or https://\n")
        return
    _CURRENT_SERVER = candidate
    print(f"  {B_GREEN}✓{RESET} Server updated to {_CURRENT_SERVER}")
    health = check_server(_CURRENT_SERVER, retries=2, backoff=0.5)
    _print_health_summary(_CURRENT_SERVER, health)


@_register_command("auth", _SERVER_SPECS["auth"].description)
def _cmd_auth(_: str = ""):
    if API_KEY:
        masked = f"{API_KEY[:4]}...{API_KEY[-4:]}" if len(API_KEY) >= 8 else "***"
        print(f"  {B_WHITE}Auth:{RESET} API key {DIM}({masked}){RESET}\n")
        return
    if BEARER_TOKEN:
        masked = (
            f"{BEARER_TOKEN[:6]}...{BEARER_TOKEN[-6:]}"
            if len(BEARER_TOKEN) >= 12
            else "***"
        )
        print(f"  {B_WHITE}Auth:{RESET} Bearer token {DIM}({masked}){RESET}\n")
        return
    print(f"  {B_WHITE}Auth:{RESET} none configured {DIM}(public mode){RESET}\n")


@_register_command(
    "_raw-health",
    _SERVER_SPECS["_raw-health"].description,
    hidden=True,
    admin_only=True,
)
def _cmd_raw_health(_: str = ""):
    health = check_server(_CURRENT_SERVER, retries=1, backoff=0.2)
    if health is None:
        print(f"  {B_RED}✗{RESET} No health payload available (server unreachable)\n")
        return
    print(orjson.dumps(health, option=orjson.OPT_INDENT_2).decode())
    print()


@_register_command("quit", _SERVER_SPECS["quit"].description)
def _cmd_quit(_: str = ""):
    print(f"\n  {DIM}Goodbye! 👋{RESET}\n")
    os._exit(0)


@_register_command("status", _SERVER_SPECS["status"].description)
def _cmd_status(_: str = ""):
    conv = _CURRENT_CONVERSATION_ID or "(auto/new on first query)"
    mem = "on" if _MEMORY_ENABLED else "off"
    loop = "server default" if _TURN_LOOP_MODE is None else ("on" if _TURN_LOOP_MODE else "off")
    print(f"  {B_WHITE}Server:{RESET} {_CURRENT_SERVER}")
    print(f"  {B_WHITE}Conversation:{RESET} {conv}")
    print(f"  {B_WHITE}Memory:{RESET} {mem}")
    print(f"  {B_WHITE}Turn loop:{RESET} {loop}\n")


def _cmd_turn_loop(arg: str = ""):
    """Toggle the per-request turn-loop override (on / off / auto)."""
    global _TURN_LOOP_MODE
    value = arg.strip().lower()
    if value in {"on", "true", "1"}:
        _TURN_LOOP_MODE = True
    elif value in {"off", "false", "0"}:
        _TURN_LOOP_MODE = False
    elif value in {"auto", "default", ""}:
        # Bare /turn-loop cycles: default -> on -> off -> default.
        if not value:
            _TURN_LOOP_MODE = {None: True, True: False, False: None}[_TURN_LOOP_MODE]
        else:
            _TURN_LOOP_MODE = None
    else:
        print(f"  {B_YELLOW}⚠{RESET} Usage: /turn-loop [on|off|auto]\n")
        return
    state = (
        f"{DIM}server default{RESET}"
        if _TURN_LOOP_MODE is None
        else (f"{B_GREEN}ON{RESET}" if _TURN_LOOP_MODE else f"{DIM}OFF{RESET}")
    )
    print(f"  {B_CYAN}⟳{RESET} Turn loop: {state}")
    print()


# The shared command catalog (src/platform/command_catalog.py) is the single
# source of truth for slash commands across CLI/console; register the handler
# only once the "turn-loop" spec exists there so the dispatcher accepts it.
if "turn-loop" in _SERVER_SPECS:
    _register_command("turn-loop", _SERVER_SPECS["turn-loop"].description)(_cmd_turn_loop)


@_register_command("new-chat", _SERVER_SPECS["new-chat"].description)
def _cmd_new_chat(arg: str = ""):
    global _CURRENT_CONVERSATION_ID
    title = arg.strip() or "New conversation"
    try:
        created = _new_conversation(_CURRENT_SERVER, title=title)
        _CURRENT_CONVERSATION_ID = created.get("conversation_id")
        print(f"  {B_GREEN}✓{RESET} New conversation: {_CURRENT_CONVERSATION_ID}\n")
    except Exception as exc:
        print(f"  {B_RED}✗{RESET} Could not create conversation: {exc}\n")


@_register_command("conversations", _SERVER_SPECS["conversations"].description)
def _cmd_conversations(_: str = ""):
    try:
        items = _list_conversations(_CURRENT_SERVER, limit=20)
    except Exception as exc:
        print(f"  {B_RED}✗{RESET} Could not load conversations: {exc}\n")
        return
    if not items:
        print(f"  {DIM}No conversations yet.{RESET}\n")
        return
    print(f"  {B_WHITE}Recent Conversations{RESET}")
    for item in items:
        cid = item.get("conversation_id", "")
        title = item.get("title", "New conversation")
        updated = item.get("updated_at_ms", 0)
        marker = "*" if cid == _CURRENT_CONVERSATION_ID else " "
        print(f"  {marker} {B_CYAN}{cid}{RESET}  {DIM}{title}{RESET}  {DIM}(updated {updated}){RESET}")
    print()


@_register_command("switch", _SERVER_SPECS["switch"].description)
def _cmd_switch(arg: str = ""):
    global _CURRENT_CONVERSATION_ID
    cid = arg.strip()
    if not cid:
        print(f"  {B_YELLOW}⚠{RESET} Usage: /switch <conversation_id>\n")
        return
    _CURRENT_CONVERSATION_ID = cid
    print(f"  {B_GREEN}✓{RESET} Active conversation set to {cid}\n")


@_register_command("history", _SERVER_SPECS["history"].description)
def _cmd_history(arg: str = ""):
    if not _CURRENT_CONVERSATION_ID:
        print(f"  {B_YELLOW}⚠{RESET} No active conversation. Run /new-chat first.\n")
        return
    limit = 12
    if arg.strip().isdigit():
        limit = int(arg.strip())
    try:
        payload = _conversation_history(_CURRENT_SERVER, _CURRENT_CONVERSATION_ID, limit=limit)
    except Exception as exc:
        print(f"  {B_RED}✗{RESET} Could not load history: {exc}\n")
        return
    turns = payload.get("turns", [])
    if not turns:
        print(f"  {DIM}No turns yet in this conversation.{RESET}\n")
        return
    print(f"  {B_WHITE}Conversation History{RESET} {DIM}({_CURRENT_CONVERSATION_ID}){RESET}")
    for turn in turns:
        role = str(turn.get("role", "user"))
        content = _truncate(str(turn.get("content", "")), 180)
        role_color = B_CYAN if role == "user" else B_GREEN
        print(f"  {role_color}{role:9}{RESET} {content}")
    print()


@_register_command("compact", _SERVER_SPECS["compact"].description)
def _cmd_compact(_: str = ""):
    if not _CURRENT_CONVERSATION_ID:
        print(f"  {B_YELLOW}⚠{RESET} No active conversation. Run /new-chat first.\n")
        return
    try:
        payload = _compact_conversation(_CURRENT_SERVER, _CURRENT_CONVERSATION_ID)
        summary = str(payload.get("summary", "")).strip()
        if summary:
            print(f"  {B_GREEN}✓{RESET} Conversation compacted.")
            print(f"  {DIM}{_truncate(summary, 260)}{RESET}\n")
        else:
            print(f"  {B_GREEN}✓{RESET} Conversation compacted.\n")
    except Exception as exc:
        print(f"  {B_RED}✗{RESET} Could not compact conversation: {exc}\n")


@_register_command("delete", _SERVER_SPECS["delete"].description)
def _cmd_delete(arg: str = ""):
    global _CURRENT_CONVERSATION_ID
    target = arg.strip() or (_CURRENT_CONVERSATION_ID or "")
    if not target:
        print(f"  {B_YELLOW}⚠{RESET} No conversation to delete. Use /delete <conversation_id> or /switch first.\n")
        return
    try:
        payload = _delete_conversation(_CURRENT_SERVER, target)
        deleted = payload.get("deleted", False)
        if deleted:
            if _CURRENT_CONVERSATION_ID == target:
                _CURRENT_CONVERSATION_ID = None
            print(f"  {B_GREEN}✓{RESET} Conversation {target} deleted.\n")
        else:
            print(f"  {B_YELLOW}⚠{RESET} Conversation {target} not found (may already be deleted).\n")
    except Exception as exc:
        print(f"  {B_RED}✗{RESET} Could not delete conversation: {exc}\n")


def main() -> None:
    import argparse
    global _CURRENT_SERVER, _ALLOW_LOCAL_ADMIN_COMMANDS, _CURRENT_CONVERSATION_ID
    parser = argparse.ArgumentParser(description="RAG CLI client (server mode)")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="API server URL")
    parser.add_argument(
        "--allow-local-admin-commands",
        action="store_true",
        help="Enable hidden local admin commands (/set-server, /_raw-health)",
    )
    args = parser.parse_args()

    server = args.server.rstrip("/")
    _CURRENT_SERVER = server
    _ALLOW_LOCAL_ADMIN_COMMANDS = args.allow_local_admin_commands or (
        os.environ.get("RAG_CLI_LOCAL_ADMIN", "").strip().lower() in {"1", "true", "yes", "on"}
    )
    _setup_tab_completion()
    print_banner(server)
    if _ALLOW_LOCAL_ADMIN_COMMANDS:
        print(
            f"  {B_YELLOW}⚠{RESET} {DIM}Local admin commands enabled "
            f"(/set-server, /_raw-health).{RESET}"
        )
        print()

    print(f"  {B_CYAN}⟡{RESET} {DIM}Connecting to {server}...{RESET}", end="", flush=True)
    health = check_server(server, retries=6, backoff=1.5)
    if health is not None:
        temporal_ok = health.get("temporal_connected", False)
        worker_ok = health.get("worker_available", False)
        logger.info(
            "Connected — temporal=%s, worker=%s, status=%s",
            temporal_ok, worker_ok, health.get("status"),
        )
        if temporal_ok and worker_ok:
            print(f"\r  {B_GREEN}✓{RESET} Connected — ready to query                 ")
        elif temporal_ok:
            print(f"\r  {B_GREEN}✓{RESET} Connected — worker still loading models     ")
        else:
            print(f"\r  {B_GREEN}✓{RESET} Connected — backend services starting       ")
            print(f"    {DIM}Queries will work once all services are ready.{RESET}")
    else:
        print(f"\r  {B_YELLOW}⚠{RESET} Could not reach {server}              ")
        print(f"    {DIM}Make sure the server stack is running:{RESET}")
        print(f"    {B_WHITE}1.{RESET} {DIM}./scripts/compose.sh up -d{RESET}")
        print(f"    {B_WHITE}2.{RESET} {DIM}.venv/bin/python -m server.worker{RESET}")
        print(f"    {B_WHITE}3.{RESET} {DIM}.venv/bin/uvicorn server.api:app --port 8000{RESET}")
        print()
        print(f"    {DIM}You can still type queries — they'll work once the server is up.{RESET}")
    print()

    PROMPT = f"  {B_CYAN}❯{RESET} "

    while True:
        try:
            raw_input_str = _get_input(PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n  {DIM}Goodbye! 👋{RESET}\n")
            os._exit(0)

        if not raw_input_str:
            continue

        handled, error = dispatch_slash_command(
            raw=raw_input_str,
            mode=MODE_SERVER_CLI,
            handlers=_server_command_handlers(),
            allow_admin=_ALLOW_LOCAL_ADMIN_COMMANDS,
        )
        if handled:
            if error == "UNKNOWN_COMMAND":
                print(f"  {B_YELLOW}⚠{RESET} Unknown command: {B_WHITE}{raw_input_str}{RESET}")
                print(f"    {DIM}Type{RESET} {B_WHITE}/{RESET} {DIM}to see available commands{RESET}")
                print()
            elif error in ("FORBIDDEN_COMMAND", "UNSUPPORTED_COMMAND"):
                # This usually means local-admin command requested without permission.
                cmd_parts = raw_input_str[1:].split(maxsplit=1)
                cmd_name = cmd_parts[0].lower() if cmd_parts else ""
                print(
                    f"  {B_YELLOW}⚠{RESET} Command /{cmd_name} is local-admin only."
                )
                print(
                    f"    {DIM}Run with --allow-local-admin-commands or "
                    f"RAG_CLI_LOCAL_ADMIN=1 to enable it.{RESET}"
                )
                print()
            elif error == "EMPTY_COMMAND":
                print(f"  {B_YELLOW}⚠{RESET} Please enter a command name after '/'.")
                print()
            continue

        query_text, filters = parse_filters(raw_input_str)
        if not query_text:
            print(f"  {B_YELLOW}⚠{RESET} Please provide a query (not just filters).")
            continue

        if filters:
            for k, v in filters.items():
                print(f"    {B_YELLOW}⬡{RESET} {DIM}{k}:{RESET} {v}")

        print()
        print(f"  {B_CYAN}⟡{RESET} {DIM}Querying server...{RESET}", end="", flush=True)
        start = time.time()

        try:
            retrieval_data = None
            generated_tokens = []
            done_data = None
            answer_started = False
            thinking_started = False
            turn_ui: dict = {"started": False, "draft_open": False, "draft_attempt": None}

            for event_type, data in send_query_stream(_CURRENT_SERVER, query_text, filters):
                if event_type in TurnEventType.ALL:
                    # Turn-loop activity log — compact one-liners / dimmed drafts.
                    _render_turn_activity(event_type, data or {}, turn_ui)

                elif event_type == "retrieval":
                    _close_turn_draft(turn_ui)
                    retrieval_data = data
                    conv_id = data.get("conversation_id")
                    if isinstance(conv_id, str) and conv_id.strip():
                        _CURRENT_CONVERSATION_ID = conv_id.strip()
                    elapsed = time.time() - start
                    print(f"\r  {B_GREEN}✓{RESET} Retrieved {DIM}({elapsed:.1f}s){RESET}        ")
                    display_retrieval(retrieval_data)
                    # The "✦ Answer" header is printed lazily on the first content
                    # token, so any live reasoning (below) renders between them.

                elif event_type == "reasoning":
                    # Live chain-of-thought from a reasoning model — shown dimmed
                    # under a "Thinking…" header, before the answer. Not the answer.
                    _close_turn_draft(turn_ui)
                    text = data.get("text", "")
                    if not thinking_started:
                        print(f"\n  {DIM}💭 Thinking…{RESET}\n")
                        sys.stdout.write(f"  {DIM}")
                        thinking_started = True
                    sys.stdout.write(text.replace("\n", "\n  "))
                    sys.stdout.flush()

                elif event_type == "token":
                    _close_turn_draft(turn_ui)
                    token = data.get("token", "")
                    if not answer_started:
                        if thinking_started:
                            sys.stdout.write(f"{RESET}\n")
                        print(f"\n  {B_GREEN}✦ Answer{RESET}\n")
                        sys.stdout.write("  ")
                        answer_started = True
                    generated_tokens.append(token)
                    if token == "\n":
                        sys.stdout.write(f"\n  ")
                    else:
                        sys.stdout.write(token)
                    sys.stdout.flush()

                elif event_type == "done":
                    done_data = data

                elif event_type == "error":
                    _close_turn_draft(turn_ui)
                    if thinking_started and not answer_started:
                        sys.stdout.write(f"{RESET}\n")
                    print(f"\n  {B_RED}✗{RESET} {data.get('message', 'Unknown error')}")

            # Reasoning streamed but no answer followed (e.g. model spent its whole
            # budget thinking) — close the dim run so the prompt isn't left dimmed.
            _close_turn_draft(turn_ui)
            if thinking_started and not answer_started:
                sys.stdout.write(f"{RESET}\n")
                sys.stdout.flush()

            if generated_tokens:
                print()
                total = done_data.get("latency_ms", 0) if done_data else (time.time() - start) * 1000
                ret_ms = done_data.get("retrieval_ms", 0) if done_data else 0
                gen_ms = done_data.get("generation_ms", 0) if done_data else 0
                tok_count = done_data.get("token_count", len(generated_tokens)) if done_data else len(generated_tokens)
                print()
                print(f"  {DIM}retrieval: {ret_ms:.0f}ms │ generation: {gen_ms:.0f}ms ({tok_count} tokens) │ total: {total:.0f}ms{RESET}")
                _display_turn_loop_summary(done_data)
                _display_stage_timings(done_data, retrieval_data)
            elif retrieval_data:
                # ask_user / no-generation path
                _display_turn_loop_summary(done_data)
                _display_stage_timings(done_data, retrieval_data)

            if retrieval_data and retrieval_data.get("action") == "ask_user":
                print(f"  {DIM}(Please rephrase your query or provide more detail.){RESET}\n")

        except urllib.error.HTTPError as exc:
            elapsed = time.time() - start
            print(f"\r  {B_RED}✗{RESET} HTTP {exc.code} {DIM}({elapsed:.1f}s){RESET}        ")
            try:
                body = orjson.loads(exc.read())
                print(f"    {DIM}{body.get('detail', str(exc))}{RESET}")
            except Exception:
                print(f"    {DIM}{exc}{RESET}")
            print()
            continue
        except urllib.error.URLError as exc:
            print(f"\r  {B_RED}✗{RESET} Connection failed                   ")
            print(f"    {DIM}{exc.reason}{RESET}")
            print(f"    {DIM}Is the server running?{RESET}")
            print()
            continue
        except Exception as exc:
            print(f"\r  {B_RED}✗{RESET} Error: {exc}                   ")
            logger.exception("Query failed")
            print()
            continue

        print(f"\n  {DIM}{'─' * 72}{RESET}\n")


if __name__ == "__main__":
    main()
