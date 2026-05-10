#!/usr/bin/env python3
# @summary
# CLI entry point with styled landing page, REPL loop, command registry,
# interactive /command menu (live-filtering), and tab completion.
# Main exports: main. Deps: os, sys, time, select, termios, tty, readline, query
# @end-summary
"""Styled CLI entry point for the RAG query engine."""

import argparse
import os
import readline
import sys
import termios
import time
import tty
import warnings
from pathlib import Path

# Suppress noisy cleanup warnings (ResourceWarning from tempfile, DeprecationWarning from swig)
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress TensorFlow C++ logging before it gets imported transitively
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent))

from query import (
    RESET, BOLD, DIM, CYAN, GREEN, YELLOW, BLUE, MAGENTA, RED, WHITE,
    B_CYAN, B_GREEN, B_YELLOW, B_BLUE, B_MAGENTA, B_RED, B_WHITE,
    LOG_FILE,
    _verbose_mode, _quiet_output, _detect_vector_backend,
    _setup_logging, parse_filters, display_results,
    logger,
)
from src.platform.cli_interactive import get_input_with_menu, setup_tab_completion
from src.platform.command_catalog import (
    MODE_INGEST_CLI,
    MODE_QUERY_CLI,
    build_registry,
    list_command_specs,
)
from src.platform.command_runtime import dispatch_slash_command

# RAGChain import deferred to main() — avoids loading torch/transformers before banner shows

# Highlight background for the selected menu row
_BG_SEL = "\033[48;5;237m"


# ── Command registry ─────────────────────────────────────────────────────────
# Each entry: name -> (handler_function, short_description)
# Handlers receive no args and return None.  Add new /commands here.
_COMMANDS: dict[str, tuple] = {}
_QUERY_SPECS = {spec.name: spec for spec in list_command_specs(MODE_QUERY_CLI)}
_INGEST_SPECS = list_command_specs(MODE_INGEST_CLI)


def _query_command_handlers() -> dict[str, callable]:
    """Adapt query command registry to shared dispatch signature."""
    return {name: (lambda _arg, fn=fn: fn()) for name, (fn, _) in _COMMANDS.items()}


def _register_command(name: str, description: str):
    """Decorator to register a / command."""
    def decorator(func):
        _COMMANDS[name] = (func, description)
        return func
    return decorator


# ── Interactive command menu ─────────────────────────────────────────────────

# Menu box width (visible characters between the │ borders).
_BOX_W = 50


def _get_menu_items(
    filter_text: str = "",
    registry: dict[str, tuple] | None = None,
) -> list[tuple[str, str]]:
    """Return filtered (name, description) pairs for the command menu.

    This is the single source of menu content.  Override or extend it to
    add custom default items in the future.
    """
    registry = registry or _COMMANDS
    ft = filter_text.lower()
    return [
        (name, desc)
        for name, (_, desc) in registry.items()
        if name.lower().startswith(ft)
    ]


def _redraw_menu(prompt: str, buf: str, items: list, sel: int) -> None:
    """Redraw prompt + typed buffer + the command dropdown below it."""
    w = _BOX_W
    # Clear from start of current line to end of screen, then rewrite
    sys.stdout.write(f"\r\033[J{prompt}{buf}")
    sys.stdout.write("\033[s")  # save cursor position

    if items:
        sys.stdout.write(f"\n  {DIM}┌{'─' * w}┐{RESET}")
        for i, (name, desc) in enumerate(items):
            tag = f"/{name}"
            # Pad: 1 + tag(14) + 1 + desc(remaining) + 1 = w
            desc_w = w - 17  # 1 leading + 14 tag + 1 space + 1 trailing
            cell_name = tag.ljust(14)
            cell_desc = desc[:desc_w].ljust(desc_w)
            if i == sel:
                sys.stdout.write(
                    f"\n  {DIM}│{_BG_SEL} {B_CYAN}{cell_name}"
                    f"{RESET}{_BG_SEL} {cell_desc} {RESET}{DIM}│{RESET}"
                )
            else:
                sys.stdout.write(
                    f"\n  {DIM}│{RESET} {B_CYAN}{cell_name}"
                    f"{RESET} {DIM}{cell_desc}{RESET} {DIM}│{RESET}"
                )
        sys.stdout.write(f"\n  {DIM}└{'─' * w}┘{RESET}")
    else:
        sys.stdout.write(f"\n    {DIM}No matching commands{RESET}")

    sys.stdout.write("\033[u")  # restore cursor position
    sys.stdout.flush()


def _read_key(fd: int) -> str:
    """Read a single keypress, handling multi-byte escape sequences.

    Returns special strings for arrow keys: 'UP', 'DOWN', 'LEFT', 'RIGHT'.
    Returns the raw character for everything else.
    """
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        # Try to read the rest of an escape sequence without relying on
        # select() timeouts which can be unreliable on some terminals.
        # Switch to non-blocking reads to drain the sequence.
        old_flags = termios.tcgetattr(fd)
        try:
            # Set non-blocking by using os.read with O_NONBLOCK
            import fcntl
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            try:
                seq = os.read(fd, 10).decode("utf-8", errors="replace")
            except (BlockingIOError, OSError):
                seq = ""
            finally:
                fcntl.fcntl(fd, fcntl.F_SETFL, flags)
        except Exception:
            seq = ""

        if seq.startswith("[A"):
            return "UP"
        elif seq.startswith("[B"):
            return "DOWN"
        elif seq.startswith("[C"):
            return "RIGHT"
        elif seq.startswith("[D"):
            return "LEFT"
        # Plain Escape (no sequence followed)
        return "ESC"
    return ch


def _interactive_command_select(
    prompt: str,
    registry: dict[str, tuple] | None = None,
) -> str | None:
    """Live-filtering command selector.  Returns a command name or None."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    buf = "/"
    sel = 0
    items = _get_menu_items(registry=registry)

    try:
        tty.setcbreak(fd)
        _redraw_menu(prompt, buf, items, sel)

        while True:
            key = _read_key(fd)

            if key in ("\r", "\n"):
                # Accept selection
                sys.stdout.write(f"\r\033[J")
                if items:
                    chosen = items[sel][0]
                    sys.stdout.write(f"{prompt}/{chosen}\n")
                    sys.stdout.flush()
                    return chosen
                sys.stdout.flush()
                return None

            elif key == "ESC":
                # Plain Escape → cancel
                sys.stdout.write(f"\r\033[J")
                sys.stdout.flush()
                return None

            elif key == "UP":
                sel = max(0, sel - 1)

            elif key == "DOWN":
                sel = min(len(items) - 1, sel + 1) if items else 0

            elif key in ("\x7f", "\x08"):
                # Backspace
                if len(buf) > 1:
                    buf = buf[:-1]
                    sel = 0
                else:
                    # Backspaced past "/" → cancel
                    sys.stdout.write(f"\r\033[J")
                    sys.stdout.flush()
                    return None

            elif key == "\x03":
                sys.stdout.write(f"\r\033[J")
                sys.stdout.flush()
                raise KeyboardInterrupt

            elif key == "\x04":
                sys.stdout.write(f"\r\033[J")
                sys.stdout.flush()
                raise EOFError

            elif len(key) == 1 and key.isprintable():
                buf += key
                sel = 0

            else:
                continue

            # Re-filter and clamp selection
            items = _get_menu_items(buf[1:], registry=registry)
            if sel >= len(items):
                sel = max(0, len(items) - 1)
            _redraw_menu(prompt, buf, items, sel)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _get_input(
    prompt: str,
    registry: dict[str, tuple] | None = None,
) -> str:
    """Read input with shared slash-command interactive menu."""
    effective_registry = registry or _COMMANDS
    return get_input_with_menu(
        prompt,
        effective_registry,
        box_width=_BOX_W,
        dim=DIM,
        reset=RESET,
        bold_cyan=B_CYAN,
        bg_sel=_BG_SEL,
    )


# ── Tab completion (fallback for non-interactive terminals) ──────────────────

def _setup_tab_completion(registry: dict[str, tuple] | None = None):
    """Configure / command completion via shared interactive module."""
    setup_tab_completion(registry or _COMMANDS)


# ── Helpers ──────────────────────────────────────────────────────────────────

def clear_screen():
    os.system("clear" if os.name != "nt" else "cls")


def print_banner():
    clear_screen()

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

    print()
    min_col = min(
        (j for line in chip_lines for j, ch in enumerate(line) if ch != " "),
        default=0,
    )
    max_col = max(
        (j for line in chip_lines for j, ch in enumerate(line) if ch != " "),
        default=1,
    )
    col_span = max(max_col - min_col, 1)
    for i, line in enumerate(chip_lines):
        # Horizontal gradient: cyan (0,255,255) → magenta (255,0,255)
        colored = []
        for j, ch in enumerate(line):
            t = max(0, j - min_col) / col_span
            r = int(255 * t)
            g = int(255 * (1 - t))
            colored.append(f"\033[1;38;2;{r};{g};255m{ch}")
        print("".join(colored) + RESET)
    print(f"{RESET}")
    print(f"{DIM}    ─────────────────────────────────────────────────────────────────────────────{RESET}")
    _backend = _detect_vector_backend()
    print(f"{B_WHITE}    RAG Query Engine{RESET}  {DIM}powered by{RESET} {B_MAGENTA}Claude{RESET} {DIM}+{RESET} {B_CYAN}{_backend}{RESET}")
    print(f"{DIM}    ─────────────────────────────────────────────────────────────────────────────{RESET}")
    print()

    # Tool badges
    print(f"    {B_MAGENTA}[ Retrieval ]{RESET}  {B_BLUE}[ Reranker ]{RESET}  {B_GREEN}[ KG Expansion ]{RESET}  {B_YELLOW}[ Generation ]{RESET}")
    print()
    print(f"    {DIM}Ask me anything — I'll retrieve and synthesize from your documents.{RESET}")
    print(f"    {DIM}Filters: {RESET}{B_WHITE}source:<file>{RESET}{DIM} · {RESET}{B_WHITE}section:<heading>{RESET}")
    print(f"    {DIM}Type {RESET}{B_WHITE}/{RESET} {DIM}to see all commands{RESET}")
    print(f"{DIM}    ─────────────────────────────────────────────────────────────────────────────{RESET}")
    print()


def print_help():
    print(f"""
{B_WHITE}  Commands:{RESET}  {DIM}(type {RESET}{B_WHITE}/{RESET}{DIM} to open menu){RESET}
""")
    for name, (_, desc) in _COMMANDS.items():
        padded = f"/{name}".ljust(14)
        print(f"    {B_CYAN}{padded}{RESET} {DIM}{desc}{RESET}")
    print(f"""
{B_WHITE}  Filter Syntax:{RESET}
    {B_YELLOW}source:<file>{RESET}      Filter by source document
    {B_YELLOW}section:<heading>{RESET}  Filter by section heading

{B_WHITE}  Logs:{RESET}
    {DIM}All logs are saved to{RESET} {B_WHITE}{LOG_FILE}{RESET}
    {DIM}Use{RESET} {B_CYAN}/verbose{RESET} {DIM}to show all library logs in the console{RESET}

{B_WHITE}  Example Queries:{RESET}
    {DIM}•{RESET} What is retrieval-augmented generation?
    {DIM}•{RESET} source:sample_doc_3.txt what is RAG?
    {DIM}•{RESET} section:Introduction explain the main concepts
    {DIM}•{RESET} How does the embedding pipeline work?
""")


# ── Register / commands ───────────────────────────────────────────────────────
# NOTE: Registration order = display order in the menu.

@_register_command("help", _QUERY_SPECS["help"].description)
def _cmd_help():
    print_help()


@_register_command("clear", _QUERY_SPECS["clear"].description)
def _cmd_clear():
    print_banner()


@_register_command("verbose", _QUERY_SPECS["verbose"].description)
def _cmd_verbose():
    import query
    query._verbose_mode = not query._verbose_mode
    state = f"{B_GREEN}ON{RESET}" if query._verbose_mode else f"{DIM}OFF{RESET}"
    print(f"  {B_CYAN}⟡{RESET} Verbose logging: {state}")
    if query._verbose_mode:
        print(f"    {DIM}All library logs will now appear in the console{RESET}")
    else:
        print(f"    {DIM}Only RAG pipeline logs shown. Full logs at {B_WHITE}{LOG_FILE}{RESET}")
    print()


@_register_command("quit", _QUERY_SPECS["quit"].description)
def _cmd_quit():
    print(f"\n  {DIM}Goodbye! 👋{RESET}\n")
    os._exit(0)


@_register_command("compact", _QUERY_SPECS["compact"].description)
def _cmd_compact():
    print(f"  {B_YELLOW}⚠{RESET} /compact is only available in server mode.")
    print(f"    {DIM}Run: python -m server.cli_client{RESET}\n")


@_register_command("delete", _QUERY_SPECS["delete"].description)
def _cmd_delete(_arg: str = ""):
    print(f"  {B_YELLOW}⚠{RESET} /delete is only available in server mode.")
    print(f"    {DIM}Run: python -m server.cli_client{RESET}\n")


# ── Main REPL ────────────────────────────────────────────────────────────────

def run_query_cli() -> None:
    """Run the interactive query loop."""
    import threading
    import query

    _setup_tab_completion()
    print_banner()

    # Show spinner while loading heavy deps + models
    _SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    start = time.time()
    loading_done = threading.Event()

    def _spin():
        i = 0
        while not loading_done.is_set():
            elapsed = time.time() - start
            frame = _SPINNER[i % len(_SPINNER)]
            sys.stdout.write(f"\r  {B_CYAN}{frame}{RESET} {DIM}Loading models... ({elapsed:.0f}s){RESET}  ")
            sys.stdout.flush()
            loading_done.wait(0.1)
            i += 1

    spin_thread = threading.Thread(target=_spin, daemon=True)
    spin_thread.start()

    # Heavy import — pulls in torch, transformers, sentence_transformers (~8s)
    from src.retrieval.pipeline.rag_chain import RAGChain

    # Init RAG pipeline (models load in parallel inside RAGChain)
    try:
        with _quiet_output():
            rag = RAGChain()
    except Exception as e:
        loading_done.set()
        spin_thread.join()
        sys.stdout.write(f"\r\033[K")
        print(f"  {B_RED}✗{RESET} Failed to initialize RAG system: {e}")
        os._exit(1)

    loading_done.set()
    spin_thread.join()
    elapsed = time.time() - start
    sys.stdout.write(f"\r\033[K")
    print(f"  {B_GREEN}✓{RESET} RAG system ready {DIM}({elapsed:.1f}s){RESET}")
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

        # Dispatch /commands from registry
        handled, error = dispatch_slash_command(
            raw=raw_input_str,
            mode=MODE_QUERY_CLI,
            handlers=_query_command_handlers(),
            allow_admin=False,
        )
        if handled:
            if error == "UNKNOWN_COMMAND":
                print(f"  {B_YELLOW}⚠{RESET} Unknown command: {B_WHITE}{raw_input_str}{RESET}")
                print(f"    {DIM}Type{RESET} {B_WHITE}/{RESET} {DIM}to see available commands{RESET}")
                print()
            elif error in ("EMPTY_COMMAND", "UNSUPPORTED_COMMAND"):
                print(f"  {B_YELLOW}⚠{RESET} Invalid command: {B_WHITE}{raw_input_str}{RESET}")
                print(f"    {DIM}Type{RESET} {B_WHITE}/{RESET} {DIM}to see available commands{RESET}")
                print()
            continue

        try:
            query_text, filters = parse_filters(raw_input_str)
            if not query_text:
                print(f"  {B_YELLOW}⚠{RESET} Please provide a query (not just filters).")
                continue

            if filters:
                for k, v in filters.items():
                    print(f"    {B_YELLOW}⬡{RESET} {DIM}{k}:{RESET} {v}")

            print()
            _backend = _detect_vector_backend()
            print(f"  {B_CYAN}⟡{RESET} {DIM}Connecting to {_backend}...{RESET}", end="", flush=True)
            start = time.time()
            if query._verbose_mode:
                response = rag.run(query_text, **filters)
            else:
                with _quiet_output():
                    response = rag.run(query_text, **filters)
            elapsed = time.time() - start
            print(f"\r  {B_GREEN}✓{RESET} {_backend} query complete {DIM}({elapsed:.1f}s){RESET}        ")
        except ValueError as exc:
            logger.warning("Invalid query input: %s", exc)
            print(f"  {B_RED}✗{RESET} Invalid input: {exc}")
            continue
        except Exception as exc:
            print(f"  {B_RED}✗ Error:{RESET} {exc}")
            continue
        display_results(response, elapsed)

        # Handle clarification loop
        if response.action == "ask_user":
            print(f"  {DIM}(Please rephrase your query or provide more detail.){RESET}\n")

        print(f"  {DIM}{'─' * 72}{RESET}\n")


def _print_ingest_banner() -> None:
    """Render a compact banner for ingestion mode."""
    clear_screen()
    print()
    print(f"    {B_WHITE}RAG Ingestion Console{RESET}")
    print(f"    {DIM}Run ingestion with configurable scope and stage options.{RESET}")
    print(f"    {DIM}{'─' * 76}{RESET}")
    print()


def _print_ingest_help() -> None:
    """Show commands available in the ingestion console."""
    print(f"{B_WHITE}  Ingest Commands:{RESET}")
    for spec in _INGEST_SPECS:
        args = f" {spec.args_hint}" if spec.args_hint else ""
        label = f"/{spec.name}{args}".ljust(32)
        print(f"    {B_CYAN}{label}{RESET} {DIM}{spec.description}{RESET}")
    print()


def _print_ingest_status(state: dict) -> None:
    """Pretty-print current ingestion runtime options."""
    file_scope = str(state["selected_file"]) if state["selected_file"] else "(none)"
    print(f"  {B_WHITE}Current Ingestion Config{RESET}")
    print(f"    {DIM}documents_dir:{RESET} {state['documents_dir']}")
    print(f"    {DIM}selected_file:{RESET} {file_scope}")
    print(f"    {DIM}update_mode:{RESET} {state['update']}")
    print(f"    {DIM}build_kg:{RESET} {state['build_kg']}")
    print(f"    {DIM}semantic_chunking:{RESET} {state['semantic_chunking']}")
    print(f"    {DIM}export_processed:{RESET} {state['export_processed']}")
    print(f"    {DIM}verbose_stage_logs:{RESET} {state['verbose_stages']}")
    print(f"    {DIM}persist_refactor_mirror:{RESET} {state['persist_refactor_mirror']}")
    print(f"    {DIM}vision_enabled:{RESET} {state['vision_enabled']}")
    print(f"    {DIM}vision_provider:{RESET} {state['vision_provider']}")
    print(f"    {DIM}vision_model:{RESET} {state['vision_model']}")
    print(f"    {DIM}vision_api_base_url:{RESET} {state['vision_api_base_url']}")
    print()


def _print_ingest_overview(state: dict) -> None:
    """Redraw full ingest dashboard (banner + commands + settings)."""
    _print_ingest_banner()
    _print_ingest_help()
    _print_ingest_status(state)


def _execute_ingest_run(state: dict) -> None:
    """Execute one ingestion run using the current interactive state."""
    from ingest import ingest as run_ingest

    if not state["update"]:
        # Destructive path: require typed confirmation.
        from config.settings import WEAVIATE_COLLECTION_NAME
        print(
            f"\n  {B_RED}⚠ FRESH MODE{RESET}: this will DELETE the entire "
            f"Weaviate collection '{WEAVIATE_COLLECTION_NAME}' before re-ingesting."
        )
        try:
            answer = input(
                f"  Type the collection name to confirm "
                f"(or anything else to abort): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer != WEAVIATE_COLLECTION_NAME:
            print(f"  {B_RED}✗ Aborted{RESET} (fresh-mode confirmation failed)\n")
            return

    print(f"  {B_CYAN}⟡{RESET} {DIM}Running ingestion...{RESET}")
    start = time.time()
    try:
        with _quiet_output():
            run_ingest(
                documents_dir=state["documents_dir"],
                fresh=not state["update"],
                update=state["update"],
                build_kg=state["build_kg"],
                semantic_chunking=state["semantic_chunking"],
                export_processed=state["export_processed"],
                selected_file=state["selected_file"],
                verbose_stages=state["verbose_stages"],
                persist_refactor_mirror=state["persist_refactor_mirror"],
                vision_enabled=state["vision_enabled"],
                vision_provider=state["vision_provider"],
                vision_model=state["vision_model"],
                vision_api_base_url=state["vision_api_base_url"],
            )
    except Exception as exc:
        print(f"  {B_RED}✗ Ingestion failed:{RESET} {exc}\n")
        return
    elapsed = time.time() - start
    print(f"  {B_GREEN}✓{RESET} Ingestion complete {DIM}({elapsed:.1f}s){RESET}\n")


def run_ingest_cli() -> None:
    """Run interactive ingestion console mode."""
    from config.settings import (
        DOCUMENTS_DIR,
        PROJECT_ROOT,
        RAG_INGESTION_PERSIST_REFACTOR_MIRROR,
        RAG_INGESTION_VISION_ENABLED,
        RAG_INGESTION_VISION_MODEL,
        RAG_INGESTION_VISION_PROVIDER,
        RAG_INGESTION_VISION_API_BASE_URL,
    )
    from src.platform.validation import validate_documents_dir

    ingest_registry = build_registry(MODE_INGEST_CLI)
    _setup_tab_completion(ingest_registry)

    state = {
        "documents_dir": validate_documents_dir(DOCUMENTS_DIR, PROJECT_ROOT),
        "selected_file": None,
        "update": True,
        "build_kg": True,
        "semantic_chunking": True,
        "export_processed": False,
        "verbose_stages": False,
        "persist_refactor_mirror": RAG_INGESTION_PERSIST_REFACTOR_MIRROR,
        "vision_enabled": RAG_INGESTION_VISION_ENABLED,
        "vision_provider": RAG_INGESTION_VISION_PROVIDER,
        "vision_model": RAG_INGESTION_VISION_MODEL,
        "vision_api_base_url": RAG_INGESTION_VISION_API_BASE_URL,
    }
    _print_ingest_overview(state)

    prompt = f"  {B_GREEN}ingest{RESET}{DIM}>{RESET} "
    while True:
        try:
            raw = _get_input(prompt, registry=ingest_registry).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n  {DIM}Goodbye! 👋{RESET}\n")
            return
        if not raw:
            continue

        if not raw.startswith("/") and raw.lower() == "run":
            raw = "/run"

        command = ""
        arg = ""
        if raw.startswith("/"):
            parts = raw.split(maxsplit=1)
            command = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

        if command == "/help":
            _print_ingest_help()
        elif command == "/status":
            _print_ingest_status(state)
        elif command == "/info":
            _print_ingest_overview(state)
        elif command == "/set-dir":
            if not arg:
                print(f"  {B_YELLOW}⚠{RESET} Usage: /set-dir <path>\n")
                continue
            try:
                new_dir = validate_documents_dir(Path(arg), PROJECT_ROOT)
            except Exception as exc:
                print(f"  {B_RED}✗ Invalid directory:{RESET} {exc}\n")
                continue
            state["documents_dir"] = new_dir
            print(f"  {B_GREEN}✓{RESET} documents_dir set to {new_dir}\n")
        elif command == "/set-file":
            if not arg:
                print(f"  {B_YELLOW}⚠{RESET} Usage: /set-file <path>\n")
                continue
            candidate = Path(arg).resolve()
            if not candidate.exists() or not candidate.is_file():
                print(f"  {B_RED}✗ Invalid file:{RESET} {candidate}\n")
                continue
            try:
                validate_documents_dir(candidate.parent, PROJECT_ROOT)
            except Exception as exc:
                print(f"  {B_RED}✗ Invalid file path:{RESET} {exc}\n")
                continue
            state["selected_file"] = candidate
            state["documents_dir"] = candidate.parent
            print(f"  {B_GREEN}✓{RESET} selected_file set to {candidate}\n")
        elif command == "/clear-file":
            state["selected_file"] = None
            print(f"  {B_GREEN}✓{RESET} selected_file cleared\n")
        elif command == "/toggle-update":
            state["update"] = not state["update"]
            print(f"  {B_CYAN}⟡{RESET} update_mode={state['update']}\n")
        elif command == "/toggle-kg":
            state["build_kg"] = not state["build_kg"]
            print(f"  {B_CYAN}⟡{RESET} build_kg={state['build_kg']}\n")
        elif command == "/toggle-semantic":
            state["semantic_chunking"] = not state["semantic_chunking"]
            print(f"  {B_CYAN}⟡{RESET} semantic_chunking={state['semantic_chunking']}\n")
        elif command == "/toggle-export":
            state["export_processed"] = not state["export_processed"]
            print(f"  {B_CYAN}⟡{RESET} export_processed={state['export_processed']}\n")
        elif command == "/toggle-stages":
            state["verbose_stages"] = not state["verbose_stages"]
            print(f"  {B_CYAN}⟡{RESET} verbose_stage_logs={state['verbose_stages']}\n")
        elif command == "/toggle-mirror":
            state["persist_refactor_mirror"] = not state["persist_refactor_mirror"]
            print(
                f"  {B_CYAN}⟡{RESET} "
                f"persist_refactor_mirror={state['persist_refactor_mirror']}\n"
            )
        elif command == "/toggle-vision":
            state["vision_enabled"] = not state["vision_enabled"]
            print(f"  {B_CYAN}⟡{RESET} vision_enabled={state['vision_enabled']}\n")
        elif command == "/set-vision-model":
            if not arg:
                print(f"  {B_YELLOW}⚠{RESET} Usage: /set-vision-model <model>\n")
                continue
            state["vision_model"] = arg
            print(f"  {B_GREEN}✓{RESET} vision_model set to {arg}\n")
        elif command == "/set-vision-provider":
            if arg not in ("ollama", "openai_compatible"):
                print(
                    f"  {B_YELLOW}⚠{RESET} Usage: /set-vision-provider "
                    "ollama|openai_compatible\n"
                )
                continue
            state["vision_provider"] = arg
            print(f"  {B_GREEN}✓{RESET} vision_provider set to {arg}\n")
        elif command == "/set-vision-api":
            state["vision_api_base_url"] = arg
            print(f"  {B_GREEN}✓{RESET} vision_api_base_url set to {arg or '(empty)'}\n")
        elif command == "/clear":
            _print_ingest_overview(state)
        elif command in ("/run", "run"):
            _execute_ingest_run(state)
        elif command in ("/quit", "quit", "/exit", "exit"):
            print(f"\n  {DIM}Goodbye! 👋{RESET}\n")
            return
        else:
            print(f"  {B_YELLOW}⚠{RESET} Unknown command: {B_WHITE}{raw}{RESET}")
            print(f"    {DIM}Type /help for ingestion commands.{RESET}\n")


def main() -> None:
    """Run unified CLI in query or ingest mode."""
    parser = argparse.ArgumentParser(description="Unified RAG CLI")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("query", "ingest"),
        default="query",
        help="Run query console or ingestion console",
    )
    args = parser.parse_args()
    if args.mode == "query":
        run_query_cli()
    else:
        run_ingest_cli()


if __name__ == "__main__":
    main()
