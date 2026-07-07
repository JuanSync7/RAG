"""Unit tests for the terminal interactive slash-command helpers.

Covers :mod:`src.platform.cli_interactive`, which provides a live-filtering
slash-command dropdown for terminal CLIs plus readline tab completion.

Strategy:

* :func:`get_menu_items` is pure and is exercised directly.
* :func:`read_key` is driven by monkeypatching ``sys.stdin.read`` (to emit the
  ESC lead byte) and ``os.read`` (to emit the trailing escape sequence), with
  ``fcntl.fcntl`` stubbed to a no-op so the real escape-decode branch runs.
* :func:`redraw_menu` writes to ``sys.stdout`` and is asserted via ``capsys``.
* :func:`interactive_command_select` is driven by monkeypatching the
  terminal-mode calls (``termios``/``tty``) to no-ops and replacing
  :func:`read_key` with a scripted iterator so the key loop is deterministic.
* :func:`get_input_with_menu` is exercised across its tty/non-tty branches by
  monkeypatching ``sys.stdin.isatty``, ``builtins.input``, the first-char read,
  and :func:`interactive_command_select`.
* :func:`setup_tab_completion` is tested by capturing the registered completer
  via a monkeypatched ``readline.set_completer`` and invoking it directly.

One behaviour per test, exact-value assertions, distinct data per case.
"""

from __future__ import annotations

import pytest

from src.platform import cli_interactive


# A registry: name -> (handler, description). Handlers are inert sentinels.
def _registry() -> dict[str, tuple]:
    return {
        "help": (object(), "Show help"),
        "hello": (object(), "Greet"),
        "status": (object(), "Show status"),
    }


# Style kwargs for redraw/select; sentinels make styling assertions exact.
STYLE = dict(
    box_width=40,
    dim="<DIM>",
    reset="<RESET>",
    bold_cyan="<CYAN>",
    bg_sel="<BGSEL>",
)


# --------------------------------------------------------------------------
# get_menu_items (pure)
# --------------------------------------------------------------------------
def test_get_menu_items_no_filter_returns_all_in_order():
    """Empty filter yields every item, preserving registry insertion order."""
    items = cli_interactive.get_menu_items(_registry())
    assert items == [
        ("help", "Show help"),
        ("hello", "Greet"),
        ("status", "Show status"),
    ]


def test_get_menu_items_prefix_filter_matches_only_prefixed_names():
    """A prefix filter keeps only names starting with that prefix."""
    items = cli_interactive.get_menu_items(_registry(), "he")
    assert items == [("help", "Show help"), ("hello", "Greet")]


def test_get_menu_items_filter_is_case_insensitive():
    """An uppercase filter matches lowercase names (case-insensitive prefix)."""
    items = cli_interactive.get_menu_items(_registry(), "HE")
    assert items == [("help", "Show help"), ("hello", "Greet")]


def test_get_menu_items_no_match_returns_empty():
    """A filter matching no name yields an empty list."""
    assert cli_interactive.get_menu_items(_registry(), "zzz") == []


def test_get_menu_items_filter_is_prefix_not_substring():
    """A non-prefix substring ('elp' inside 'help') does NOT match."""
    assert cli_interactive.get_menu_items(_registry(), "elp") == []


# --------------------------------------------------------------------------
# read_key
# --------------------------------------------------------------------------
def _arm_escape(monkeypatch, seq_bytes: bytes):
    """Arm read_key to see ESC then ``seq_bytes`` from os.read."""
    monkeypatch.setattr(cli_interactive.sys.stdin, "read", lambda n: "\x1b")
    monkeypatch.setattr(cli_interactive.os, "read", lambda fd, n: seq_bytes)
    import fcntl

    monkeypatch.setattr(fcntl, "fcntl", lambda *a, **k: 0)


@pytest.mark.parametrize(
    "seq,expected",
    [
        (b"[A", "UP"),
        (b"[B", "DOWN"),
        (b"[C", "RIGHT"),
        (b"[D", "LEFT"),
        (b"[Z", "ESC"),
    ],
)
def test_read_key_escape_sequences(monkeypatch, seq, expected):
    """ESC + arrow sequences map to symbolic names; unknown seq is ESC."""
    _arm_escape(monkeypatch, seq)
    assert cli_interactive.read_key(0) == expected


def test_read_key_plain_char_returns_char(monkeypatch):
    """A non-ESC keypress is returned verbatim."""
    monkeypatch.setattr(cli_interactive.sys.stdin, "read", lambda n: "a")
    assert cli_interactive.read_key(0) == "a"


# --------------------------------------------------------------------------
# redraw_menu (capsys)
# --------------------------------------------------------------------------
def test_redraw_menu_with_items_renders_tags_and_border(capsys):
    """Each item's /name tag and the box border chars are written."""
    items = [("help", "Show help"), ("hello", "Greet")]
    cli_interactive.redraw_menu("> ", "/he", items, 0, **STYLE)
    out = capsys.readouterr().out
    assert "/help" in out
    assert "/hello" in out
    assert "┌" in out and "┐" in out
    assert "└" in out and "┘" in out


def test_redraw_menu_no_items_renders_no_matching(capsys):
    """An empty item list renders the 'No matching commands' notice."""
    cli_interactive.redraw_menu("> ", "/zzz", [], 0, **STYLE)
    out = capsys.readouterr().out
    assert "No matching commands" in out


def test_redraw_menu_selected_row_uses_bg_sel(capsys):
    """The selected row is styled with the bg_sel sentinel; others are not."""
    items = [("help", "Show help"), ("hello", "Greet")]
    cli_interactive.redraw_menu("> ", "/he", items, 1, **STYLE)
    out = capsys.readouterr().out
    # bg_sel sentinel appears (selected row styling) and the selected tag is /hello.
    assert "<BGSEL>" in out
    assert "/hello" in out


# --------------------------------------------------------------------------
# interactive_command_select (scripted read_key)
# --------------------------------------------------------------------------
def _patch_terminal(monkeypatch):
    """Stub termios/tty so raw-mode setup is harmless under a fake fd."""
    monkeypatch.setattr(cli_interactive.sys.stdin, "fileno", lambda: 0)
    monkeypatch.setattr(cli_interactive.termios, "tcgetattr", lambda fd: [])
    monkeypatch.setattr(cli_interactive.termios, "tcsetattr", lambda fd, w, o: None)
    monkeypatch.setattr(cli_interactive.tty, "setcbreak", lambda fd: None)


def _script_keys(monkeypatch, keys):
    """Replace read_key with a scripted iterator over ``keys``."""
    it = iter(keys)
    monkeypatch.setattr(cli_interactive, "read_key", lambda fd: next(it))


def test_select_enter_returns_first_item(monkeypatch, capsys):
    """Pressing Enter immediately returns the first (top) item."""
    _patch_terminal(monkeypatch)
    _script_keys(monkeypatch, ["\r"])
    result = cli_interactive.interactive_command_select("> ", _registry(), **STYLE)
    assert result == "help"


def test_select_down_then_enter_returns_second_item(monkeypatch, capsys):
    """DOWN then Enter selects the second item."""
    _patch_terminal(monkeypatch)
    _script_keys(monkeypatch, ["DOWN", "\r"])
    result = cli_interactive.interactive_command_select("> ", _registry(), **STYLE)
    assert result == "hello"


def test_select_up_at_top_is_clamped(monkeypatch, capsys):
    """UP from the top row stays clamped at index 0 (returns first item)."""
    _patch_terminal(monkeypatch)
    _script_keys(monkeypatch, ["UP", "\r"])
    result = cli_interactive.interactive_command_select("> ", _registry(), **STYLE)
    assert result == "help"


def test_select_down_past_end_is_clamped(monkeypatch, capsys):
    """DOWN beyond the last row stays clamped on the last item."""
    _patch_terminal(monkeypatch)
    # 3 items; 5 DOWNs would overshoot if unclamped.
    _script_keys(monkeypatch, ["DOWN", "DOWN", "DOWN", "DOWN", "DOWN", "\r"])
    result = cli_interactive.interactive_command_select("> ", _registry(), **STYLE)
    assert result == "status"


def test_select_esc_returns_none(monkeypatch, capsys):
    """ESC cancels selection and returns None."""
    _patch_terminal(monkeypatch)
    _script_keys(monkeypatch, ["ESC"])
    result = cli_interactive.interactive_command_select("> ", _registry(), **STYLE)
    assert result is None


def test_select_typed_char_filters_then_enter_selects(monkeypatch, capsys):
    """Typing 's' filters to 'status', then Enter selects it."""
    _patch_terminal(monkeypatch)
    _script_keys(monkeypatch, ["s", "\r"])
    result = cli_interactive.interactive_command_select("> ", _registry(), **STYLE)
    assert result == "status"


def test_select_backspace_at_root_buffer_returns_none(monkeypatch, capsys):
    """Backspace when the buffer is just '/' cancels and returns None."""
    _patch_terminal(monkeypatch)
    _script_keys(monkeypatch, ["\x7f"])
    result = cli_interactive.interactive_command_select("> ", _registry(), **STYLE)
    assert result is None


def test_select_ctrl_c_raises_keyboard_interrupt(monkeypatch, capsys):
    """Ctrl-C raises KeyboardInterrupt."""
    _patch_terminal(monkeypatch)
    _script_keys(monkeypatch, ["\x03"])
    with pytest.raises(KeyboardInterrupt):
        cli_interactive.interactive_command_select("> ", _registry(), **STYLE)


def test_select_ctrl_d_raises_eof(monkeypatch, capsys):
    """Ctrl-D raises EOFError."""
    _patch_terminal(monkeypatch)
    _script_keys(monkeypatch, ["\x04"])
    with pytest.raises(EOFError):
        cli_interactive.interactive_command_select("> ", _registry(), **STYLE)


# --------------------------------------------------------------------------
# get_input_with_menu
# --------------------------------------------------------------------------
def test_get_input_non_tty_passthrough_to_input(monkeypatch):
    """When stdin is not a tty, input(prompt) is returned and no menu opens."""
    monkeypatch.setattr(cli_interactive.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda prompt="": "typed line")

    def _boom(*a, **k):
        raise AssertionError("menu should not open in non-tty mode")

    monkeypatch.setattr(cli_interactive, "interactive_command_select", _boom)
    result = cli_interactive.get_input_with_menu("> ", _registry(), **STYLE)
    assert result == "typed line"


def test_get_input_tty_slash_opens_menu_and_returns_slash_cmd(monkeypatch):
    """A leading '/' in tty mode opens the menu and returns '/<cmd>'."""
    monkeypatch.setattr(cli_interactive.sys.stdin, "isatty", lambda: True)
    _patch_terminal(monkeypatch)
    monkeypatch.setattr(cli_interactive.sys.stdin, "read", lambda n: "/")
    monkeypatch.setattr(
        cli_interactive, "interactive_command_select", lambda *a, **k: "help"
    )
    result = cli_interactive.get_input_with_menu("> ", _registry(), **STYLE)
    assert result == "/help"


def test_get_input_tty_slash_cancelled_returns_empty(monkeypatch):
    """A '/' that is cancelled (menu returns None) yields an empty string."""
    monkeypatch.setattr(cli_interactive.sys.stdin, "isatty", lambda: True)
    _patch_terminal(monkeypatch)
    monkeypatch.setattr(cli_interactive.sys.stdin, "read", lambda n: "/")
    monkeypatch.setattr(
        cli_interactive, "interactive_command_select", lambda *a, **k: None
    )
    result = cli_interactive.get_input_with_menu("> ", _registry(), **STYLE)
    assert result == ""


def test_get_input_tty_enter_returns_empty(monkeypatch):
    """Pressing Enter as the first char in tty mode returns an empty string."""
    monkeypatch.setattr(cli_interactive.sys.stdin, "isatty", lambda: True)
    _patch_terminal(monkeypatch)
    monkeypatch.setattr(cli_interactive.sys.stdin, "read", lambda n: "\r")
    result = cli_interactive.get_input_with_menu("> ", _registry(), **STYLE)
    assert result == ""


# --------------------------------------------------------------------------
# setup_tab_completion
# --------------------------------------------------------------------------
def _capture_completer(monkeypatch):
    """Register no-op readline hooks and capture the completer callable."""
    captured = {}
    monkeypatch.setattr(
        cli_interactive.readline,
        "set_completer",
        lambda fn: captured.__setitem__("fn", fn),
    )
    monkeypatch.setattr(
        cli_interactive.readline, "set_completer_delims", lambda d: None
    )
    monkeypatch.setattr(cli_interactive.readline, "parse_and_bind", lambda s: None)
    return captured


def test_setup_tab_completion_matches_slash_prefix(monkeypatch):
    """The completer returns matching /name options for a slash prefix."""
    captured = _capture_completer(monkeypatch)
    cli_interactive.setup_tab_completion(_registry())
    completer = captured["fn"]
    # "/he" matches "/help" and "/hello"; first option at state 0.
    assert completer("/he", 0) == "/help"
    assert completer("/he", 1) == "/hello"
    # Past the option count -> None.
    assert completer("/he", 2) is None


def test_setup_tab_completion_non_slash_text_has_no_options(monkeypatch):
    """Non-slash text produces no completion options (None at state 0)."""
    captured = _capture_completer(monkeypatch)
    cli_interactive.setup_tab_completion(_registry())
    completer = captured["fn"]
    assert completer("help", 0) is None
