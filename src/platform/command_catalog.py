# @summary
# Single source of truth for user-visible slash commands across CLI and console surfaces.
# Exports: CommandSpec, list_command_specs, get_command_spec, build_registry, to_payload,
#          MODE_QUERY_CLI, MODE_INGEST_CLI, MODE_SERVER_CLI, MODE_CONSOLE_QUERY, MODE_CONSOLE_INGEST
# Deps: dataclasses
# @end-summary
"""Slash-command catalog shared across CLI and console surfaces.

This module is the single source of truth for user-visible slash commands.
Each runtime surface (terminal CLI, web console, SDK adapters) can consume the
same command metadata, then attach surface-specific handlers/formatters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

MODE_QUERY_CLI = "query_cli"
MODE_INGEST_CLI = "ingest_cli"
MODE_SERVER_CLI = "server_cli"
MODE_CONSOLE_QUERY = "console_query"
MODE_CONSOLE_INGEST = "console_ingest"


@dataclass(frozen=True)
class CommandSpec:
    """Shared metadata for a slash command."""

    name: str
    description: str
    modes: tuple[str, ...]
    args_hint: str = ""
    hidden: bool = False
    admin_only: bool = False
    intent: str = ""


_COMMAND_SPECS: tuple[CommandSpec, ...] = (
    CommandSpec(
        name="help",
        description="Show command help",
        modes=(
            MODE_QUERY_CLI,
            MODE_INGEST_CLI,
            MODE_SERVER_CLI,
            MODE_CONSOLE_QUERY,
            MODE_CONSOLE_INGEST,
        ),
        intent="show_help",
    ),
    CommandSpec(
        name="clear",
        description="Clear output area / redraw view",
        modes=(
            MODE_QUERY_CLI,
            MODE_INGEST_CLI,
            MODE_SERVER_CLI,
            MODE_CONSOLE_QUERY,
            MODE_CONSOLE_INGEST,
        ),
        intent="clear_view",
    ),
    CommandSpec(
        name="verbose",
        description="Toggle verbose mode",
        modes=(MODE_QUERY_CLI, MODE_SERVER_CLI),
        intent="toggle_verbose",
    ),
    CommandSpec(
        name="quit",
        description="Exit current interactive session",
        modes=(MODE_QUERY_CLI, MODE_INGEST_CLI, MODE_SERVER_CLI),
        intent="quit",
    ),
    CommandSpec(
        name="run",
        description="Execute action in current tab",
        modes=(MODE_INGEST_CLI, MODE_CONSOLE_QUERY, MODE_CONSOLE_INGEST),
        intent="run",
    ),
    CommandSpec(
        name="run-non-stream",
        description="Run non-stream query",
        modes=(MODE_CONSOLE_QUERY,),
        intent="run_non_stream",
    ),
    CommandSpec(
        name="new-chat",
        description="Create and switch to a new conversation",
        modes=(MODE_SERVER_CLI, MODE_CONSOLE_QUERY),
        args_hint="[title]",
        intent="new_conversation",
    ),
    CommandSpec(
        name="conversations",
        description="List recent conversations",
        modes=(MODE_SERVER_CLI, MODE_CONSOLE_QUERY),
        intent="list_conversations",
    ),
    CommandSpec(
        name="switch",
        description="Switch active conversation",
        modes=(MODE_SERVER_CLI, MODE_CONSOLE_QUERY),
        args_hint="<conversation_id>",
        intent="switch_conversation",
    ),
    CommandSpec(
        name="history",
        description="Show conversation history",
        modes=(MODE_SERVER_CLI, MODE_CONSOLE_QUERY),
        args_hint="[limit]",
        intent="show_history",
    ),
    CommandSpec(
        name="compact",
        description="Compact conversation summary now",
        modes=(MODE_QUERY_CLI, MODE_SERVER_CLI, MODE_CONSOLE_QUERY),
        intent="compact_conversation",
    ),
    CommandSpec(
        name="delete",
        description="Delete a conversation",
        modes=(MODE_QUERY_CLI, MODE_SERVER_CLI, MODE_CONSOLE_QUERY),
        args_hint="<conversation_id>",
        intent="delete_conversation",
    ),
    CommandSpec(
        name="status",
        description="Show current settings/status",
        modes=(MODE_INGEST_CLI, MODE_CONSOLE_QUERY, MODE_CONSOLE_INGEST, MODE_SERVER_CLI),
        intent="show_status",
    ),
    CommandSpec(
        name="info",
        description="Redraw banner/commands/settings",
        modes=(MODE_INGEST_CLI,),
        intent="show_info",
    ),
    CommandSpec(
        name="set-dir",
        description="Set documents directory scope",
        modes=(MODE_INGEST_CLI,),
        args_hint="<path>",
        intent="set_documents_dir",
    ),
    CommandSpec(
        name="set-file",
        description="Set single-file ingestion scope",
        modes=(MODE_INGEST_CLI,),
        args_hint="<path>",
        intent="set_selected_file",
    ),
    CommandSpec(
        name="clear-file",
        description="Clear single-file ingestion scope",
        modes=(MODE_INGEST_CLI,),
        intent="clear_selected_file",
    ),
    CommandSpec(
        name="toggle-update",
        description="Toggle incremental update mode",
        modes=(MODE_INGEST_CLI,),
        intent="toggle_update_mode",
    ),
    CommandSpec(
        name="toggle-kg",
        description="Toggle KG extraction/storage",
        modes=(MODE_INGEST_CLI,),
        intent="toggle_build_kg",
    ),
    CommandSpec(
        name="toggle-semantic",
        description="Toggle semantic chunking",
        modes=(MODE_INGEST_CLI,),
        intent="toggle_semantic_chunking",
    ),
    CommandSpec(
        name="toggle-export",
        description="Toggle processed artifact export",
        modes=(MODE_INGEST_CLI,),
        intent="toggle_export_processed",
    ),
    CommandSpec(
        name="toggle-stages",
        description="Toggle per-stage progress logs",
        modes=(MODE_INGEST_CLI,),
        intent="toggle_verbose_stages",
    ),
    CommandSpec(
        name="toggle-mirror",
        description="Toggle original/refactor mirror artifacts",
        modes=(MODE_INGEST_CLI,),
        intent="toggle_refactor_mirror",
    ),
    CommandSpec(
        name="toggle-vision",
        description="Toggle vision figure analysis",
        modes=(MODE_INGEST_CLI,),
        intent="toggle_vision_enabled",
    ),
    CommandSpec(
        name="set-vision-provider",
        description="Set vision provider",
        modes=(MODE_INGEST_CLI,),
        args_hint="<ollama|openai_compatible>",
        intent="set_vision_provider",
    ),
    CommandSpec(
        name="set-vision-model",
        description="Set vision model",
        modes=(MODE_INGEST_CLI,),
        args_hint="<model>",
        intent="set_vision_model",
    ),
    CommandSpec(
        name="set-vision-api",
        description="Set OpenAI-compatible vision API URL",
        modes=(MODE_INGEST_CLI,),
        args_hint="<url>",
        intent="set_vision_api",
    ),
    CommandSpec(
        name="health",
        description="Show backend + worker readiness",
        modes=(MODE_SERVER_CLI, MODE_CONSOLE_QUERY),
        intent="show_health",
    ),
    CommandSpec(
        name="server",
        description="Show current API server endpoint",
        modes=(MODE_SERVER_CLI,),
        intent="show_server",
    ),
    CommandSpec(
        name="set-server",
        description="Switch API server endpoint",
        modes=(MODE_SERVER_CLI,),
        args_hint="<url>",
        hidden=True,
        admin_only=True,
        intent="set_server",
    ),
    CommandSpec(
        name="auth",
        description="Show auth mode (masked)",
        modes=(MODE_SERVER_CLI,),
        intent="show_auth",
    ),
    CommandSpec(
        name="_raw-health",
        description="Dump raw health payload",
        modes=(MODE_SERVER_CLI,),
        hidden=True,
        admin_only=True,
        intent="dump_raw_health",
    ),
)


def list_command_specs(
    mode: str,
    *,
    include_hidden: bool = False,
    allow_admin: bool = False,
) -> list[CommandSpec]:
    """List command specs available for a given interaction mode.

    Args:
        mode: Interaction mode identifier.
        include_hidden: Whether to include hidden commands.
        allow_admin: Whether to include admin-only commands.

    Returns:
        A list of command specs available in the given mode.
    """

    result: list[CommandSpec] = []
    for spec in _COMMAND_SPECS:
        if mode not in spec.modes:
            continue
        if spec.hidden and not include_hidden:
            continue
        if spec.admin_only and not allow_admin:
            continue
        result.append(spec)
    return result


def get_command_spec(
    mode: str,
    name: str,
    *,
    include_hidden: bool = False,
    allow_admin: bool = False,
) -> CommandSpec | None:
    """Look up a single command by name for the given mode.

    Args:
        mode: Interaction mode identifier.
        name: Command name (with or without leading `/`).
        include_hidden: Whether to include hidden commands.
        allow_admin: Whether to include admin-only commands.

    Returns:
        The matching command spec, or None if not available for the mode.
    """

    target = name.strip().lower()
    for spec in list_command_specs(
        mode,
        include_hidden=include_hidden,
        allow_admin=allow_admin,
    ):
        if spec.name == target:
            return spec
    return None


def build_registry(
    mode: str,
    *,
    include_hidden: bool = False,
    allow_admin: bool = False,
) -> dict[str, tuple[None, str]]:
    """Build a menu/tab-completion friendly command registry.

    Args:
        mode: Interaction mode identifier.
        include_hidden: Whether to include hidden commands.
        allow_admin: Whether to include admin-only commands.

    Returns:
        Mapping of command name to `(handler_placeholder, description)` tuple.
    """

    return {
        spec.name: (None, spec.description)
        for spec in list_command_specs(
            mode, include_hidden=include_hidden, allow_admin=allow_admin
        )
    }


def to_payload(specs: Iterable[CommandSpec]) -> list[dict]:
    """Serialize command specs for API/JSON transport.

    Args:
        specs: Iterable of command specs.

    Returns:
        List of JSON-serializable dictionaries.
    """

    return [asdict(spec) for spec in specs]


__all__ = [
    "MODE_CONSOLE_INGEST",
    "MODE_CONSOLE_QUERY",
    "MODE_INGEST_CLI",
    "MODE_QUERY_CLI",
    "MODE_SERVER_CLI",
    "CommandSpec",
    "build_registry",
    "get_command_spec",
    "list_command_specs",
    "to_payload",
]
