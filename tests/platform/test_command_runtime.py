"""Unit tests for the shared slash-command parsing and dispatch runtime.

Covers :func:`parse_slash_command` (tokenization, case-folding, whitespace
handling) and :func:`dispatch_slash_command` (all six dispatch outcomes plus
the ``allow_admin`` plumbing into the catalog lookups).

A mix of strategies is used: a couple of tests run against the *real*
``src.platform.command_catalog`` (mode ``server_cli``, where ``help`` is a
normal command and ``set-server`` is hidden/admin-only) to exercise the true
integration of units, while the precise branch-control tests monkeypatch
``src.platform.command_runtime.get_command_spec`` so each dispatch branch can
be driven deterministically regardless of future catalog edits.
"""

from __future__ import annotations

import pytest

from src.platform import command_runtime
from src.platform.command_catalog import CommandSpec, MODE_SERVER_CLI
from src.platform.command_runtime import (
    ParsedSlashCommand,
    dispatch_slash_command,
    parse_slash_command,
)


# --------------------------------------------------------------------------
# parse_slash_command
# --------------------------------------------------------------------------


def test_parse_non_slash_returns_none():
    """Input not starting with `/` is not a slash command -> None."""
    assert parse_slash_command("hello world") is None


def test_parse_bare_slash_yields_empty_name_and_arg():
    """A bare `/` parses to an empty-name, empty-arg command."""
    parsed = parse_slash_command("/")
    assert parsed == ParsedSlashCommand(name="", arg="")


def test_parse_slash_with_only_whitespace_yields_empty_name_and_arg():
    """`/` followed by whitespace still yields an empty command."""
    parsed = parse_slash_command("/   ")
    assert parsed == ParsedSlashCommand(name="", arg="")


def test_parse_command_without_arg():
    """`/help` parses to name='help' with an empty arg."""
    parsed = parse_slash_command("/help")
    assert parsed == ParsedSlashCommand(name="help", arg="")


def test_parse_command_name_is_lowercased():
    """Command name is case-folded to lowercase (`/Help` -> 'help')."""
    parsed = parse_slash_command("/Help")
    assert parsed is not None
    assert parsed.name == "help"


def test_parse_command_with_arg():
    """`/ask what is x` splits into name='ask' and arg='what is x'."""
    parsed = parse_slash_command("/ask what is x")
    assert parsed == ParsedSlashCommand(name="ask", arg="what is x")


def test_parse_strips_surrounding_whitespace_in_raw():
    """Leading/trailing whitespace in the raw input is stripped before parsing."""
    parsed = parse_slash_command("   /ask hello   ")
    assert parsed == ParsedSlashCommand(name="ask", arg="hello")


def test_parse_preserves_internal_arg_spacing_after_first_split():
    """Only the first whitespace run splits name from arg; arg internal spacing
    survives (interior double-space is kept after the leading split)."""
    parsed = parse_slash_command("/ask  what   is  x")
    assert parsed is not None
    assert parsed.name == "ask"
    # split(maxsplit=1) consumes the run after the name; remaining interior
    # spacing inside the arg is preserved (then outer-stripped).
    assert parsed.arg == "what   is  x"


# --------------------------------------------------------------------------
# dispatch_slash_command  -- helpers
# --------------------------------------------------------------------------


def _spec(name: str, *, admin_only: bool = False) -> CommandSpec:
    """Build a minimal CommandSpec for monkeypatched lookups."""
    return CommandSpec(
        name=name,
        description="test",
        modes=(MODE_SERVER_CLI,),
        admin_only=admin_only,
    )


class _RecordingHandler:
    """Callable spy that records each arg it was invoked with."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, arg: str) -> None:
        self.calls.append(arg)


# --------------------------------------------------------------------------
# dispatch_slash_command  -- six outcomes
# --------------------------------------------------------------------------


def test_dispatch_non_slash_returns_unhandled_and_skips_handler():
    """(a) Non-slash input -> (False, None) and the handler is never called."""
    handler = _RecordingHandler()
    handled, error = dispatch_slash_command(
        raw="just chatting",
        mode=MODE_SERVER_CLI,
        handlers={"help": handler},
    )
    assert (handled, error) == (False, None)
    assert handler.calls == []


def test_dispatch_empty_command_returns_empty_command_error():
    """(b) A bare `/` -> (True, 'EMPTY_COMMAND')."""
    handler = _RecordingHandler()
    handled, error = dispatch_slash_command(
        raw="/",
        mode=MODE_SERVER_CLI,
        handlers={"help": handler},
    )
    assert (handled, error) == (True, "EMPTY_COMMAND")
    assert handler.calls == []


def test_dispatch_allowed_command_invokes_handler_with_arg(monkeypatch):
    """(c) Allowed command with a registered handler -> handler called exactly
    once with the parsed arg; returns (True, None)."""
    monkeypatch.setattr(
        command_runtime,
        "get_command_spec",
        lambda mode, name, **kwargs: _spec("ask"),
    )
    handler = _RecordingHandler()
    handled, error = dispatch_slash_command(
        raw="/ask what is rag",
        mode=MODE_SERVER_CLI,
        handlers={"ask": handler},
    )
    assert (handled, error) == (True, None)
    # Real teeth: exact arg value, exactly one call.
    assert handler.calls == ["what is rag"]


def test_dispatch_allowed_command_without_handler_is_unsupported(monkeypatch):
    """(d) Allowed by the catalog but no handler registered -> (True,
    'UNSUPPORTED_COMMAND')."""
    monkeypatch.setattr(
        command_runtime,
        "get_command_spec",
        lambda mode, name, **kwargs: _spec("ask"),
    )
    handled, error = dispatch_slash_command(
        raw="/ask hi",
        mode=MODE_SERVER_CLI,
        handlers={},
    )
    assert (handled, error) == (True, "UNSUPPORTED_COMMAND")


def test_dispatch_unknown_command_returns_unknown(monkeypatch):
    """(e) Catalog returns None for both lookups -> (True, 'UNKNOWN_COMMAND')."""
    monkeypatch.setattr(
        command_runtime,
        "get_command_spec",
        lambda mode, name, **kwargs: None,
    )
    handler = _RecordingHandler()
    handled, error = dispatch_slash_command(
        raw="/nope",
        mode=MODE_SERVER_CLI,
        handlers={"nope": handler},
    )
    assert (handled, error) == (True, "UNKNOWN_COMMAND")
    assert handler.calls == []


def test_dispatch_admin_only_command_forbidden_without_admin(monkeypatch):
    """(f) Known admin-only command with allow_admin=False -> (True,
    'FORBIDDEN_COMMAND').

    First lookup (non-admin) returns None; second lookup (include_hidden +
    allow_admin) returns the admin-only spec, so the dispatcher distinguishes
    forbidden from unknown.
    """

    def fake_get_command_spec(mode, name, *, include_hidden=False, allow_admin=False):
        if allow_admin:
            return _spec("set-server", admin_only=True)
        return None

    monkeypatch.setattr(command_runtime, "get_command_spec", fake_get_command_spec)
    handler = _RecordingHandler()
    handled, error = dispatch_slash_command(
        raw="/set-server http://x",
        mode=MODE_SERVER_CLI,
        handlers={"set-server": handler},
        allow_admin=False,
    )
    assert (handled, error) == (True, "FORBIDDEN_COMMAND")
    assert handler.calls == []


def test_dispatch_admin_only_command_dispatches_with_admin(monkeypatch):
    """(f, partner) The SAME admin-only command with allow_admin=True is allowed
    and the handler runs -> (True, None)."""

    def fake_get_command_spec(mode, name, *, include_hidden=False, allow_admin=False):
        # With admin granted, the first (gated) lookup succeeds.
        if allow_admin:
            return _spec("set-server", admin_only=True)
        return None

    monkeypatch.setattr(command_runtime, "get_command_spec", fake_get_command_spec)
    handler = _RecordingHandler()
    handled, error = dispatch_slash_command(
        raw="/set-server http://x",
        mode=MODE_SERVER_CLI,
        handlers={"set-server": handler},
        allow_admin=True,
    )
    assert (handled, error) == (True, None)
    assert handler.calls == ["http://x"]


# --------------------------------------------------------------------------
# dispatch_slash_command  -- allow_admin plumbing has teeth
# --------------------------------------------------------------------------


def test_dispatch_threads_allow_admin_false_into_first_lookup(monkeypatch):
    """The first catalog lookup must receive include_hidden/allow_admin matching
    allow_admin (here False)."""
    seen: list[dict] = []

    def spy(mode, name, *, include_hidden=False, allow_admin=False):
        seen.append({"include_hidden": include_hidden, "allow_admin": allow_admin})
        return _spec("ask")  # short-circuit; only the first call matters

    monkeypatch.setattr(command_runtime, "get_command_spec", spy)
    dispatch_slash_command(
        raw="/ask q",
        mode=MODE_SERVER_CLI,
        handlers={"ask": _RecordingHandler()},
        allow_admin=False,
    )
    assert seen == [{"include_hidden": False, "allow_admin": False}]


def test_dispatch_threads_allow_admin_true_into_first_lookup(monkeypatch):
    """With allow_admin=True the first lookup must request hidden+admin commands."""
    seen: list[dict] = []

    def spy(mode, name, *, include_hidden=False, allow_admin=False):
        seen.append({"include_hidden": include_hidden, "allow_admin": allow_admin})
        return _spec("ask")

    monkeypatch.setattr(command_runtime, "get_command_spec", spy)
    dispatch_slash_command(
        raw="/ask q",
        mode=MODE_SERVER_CLI,
        handlers={"ask": _RecordingHandler()},
        allow_admin=True,
    )
    assert seen == [{"include_hidden": True, "allow_admin": True}]


def test_dispatch_second_lookup_uses_full_catalog_view(monkeypatch):
    """When the first lookup misses, the disambiguation lookup must use
    include_hidden=True and allow_admin=True to detect known-but-restricted
    commands."""
    seen: list[dict] = []

    def spy(mode, name, *, include_hidden=False, allow_admin=False):
        seen.append({"include_hidden": include_hidden, "allow_admin": allow_admin})
        return None  # both lookups miss -> UNKNOWN

    monkeypatch.setattr(command_runtime, "get_command_spec", spy)
    dispatch_slash_command(
        raw="/mystery",
        mode=MODE_SERVER_CLI,
        handlers={},
        allow_admin=False,
    )
    assert seen == [
        {"include_hidden": False, "allow_admin": False},
        {"include_hidden": True, "allow_admin": True},
    ]


# --------------------------------------------------------------------------
# dispatch_slash_command  -- integration against the REAL catalog
# --------------------------------------------------------------------------


def test_dispatch_real_catalog_known_command_dispatches():
    """Integration of units: `/help` is a real server_cli command, so with a
    registered handler it dispatches cleanly."""
    handler = _RecordingHandler()
    handled, error = dispatch_slash_command(
        raw="/help",
        mode=MODE_SERVER_CLI,
        handlers={"help": handler},
    )
    assert (handled, error) == (True, None)
    assert handler.calls == [""]


def test_dispatch_real_catalog_admin_command_forbidden_without_admin():
    """Integration of units: `/set-server` is hidden+admin_only in the real
    catalog, so without admin it is reported FORBIDDEN (not UNKNOWN)."""
    handler = _RecordingHandler()
    handled, error = dispatch_slash_command(
        raw="/set-server http://example",
        mode=MODE_SERVER_CLI,
        handlers={"set-server": handler},
        allow_admin=False,
    )
    assert (handled, error) == (True, "FORBIDDEN_COMMAND")
    assert handler.calls == []
