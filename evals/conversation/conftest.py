# @summary
# Pytest fixtures for the multi-turn conversation eval suite: fixture loading
# with the skip-if-unpopulated convention (mirrors the deep_research
# _load_or_skip), a skip-if-unimportable loader for the real run_turn_loop
# orchestrator entry point, and a session-scoped offline suite run shared by
# every test — the default run needs NO live infrastructure.
# Exports: golden_conversations, run_turn_loop_entry, offline_report (fixtures)
# Deps: pytest, evals.conftest.load_json_fixture, evals.conversation.harness
# @end-summary
"""Fixtures for the multi-turn conversation eval suite (offline by default)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

import pytest

from evals.conftest import load_json_fixture

_FIXTURES_ROOT = Path(__file__).parent / "fixtures"


def _load_or_skip(name: str) -> Dict[str, Any]:
    """Load a conversation fixture, skipping when missing or unpopulated."""
    path = _FIXTURES_ROOT / name
    if not path.exists():
        pytest.skip(f"Conversation fixture not populated: {path}")
    data = load_json_fixture(path)
    if not data.get("conversations"):
        pytest.skip(f"Conversation fixture {name} is empty")
    return data


@pytest.fixture(scope="session")
def golden_conversations() -> Dict[str, Any]:
    """The golden multi-turn conversation set (versioned, design section 10)."""
    return _load_or_skip("golden_conversations.json")


@pytest.fixture(scope="session")
def run_turn_loop_entry() -> Callable[..., Any]:
    """The REAL orchestrator entry point, or skip with the exact import error.

    Imported through the package facade (PEP 562 lazy resolution) so the eval
    exercises the same import path production callers use. Skipping — not
    failing — keeps the suite honest while the orchestrator track lands.
    """
    try:
        from src.retrieval.pipeline.turn_loop import run_turn_loop
    except Exception as exc:  # noqa: BLE001 — surface the exact import error
        pytest.skip(f"run_turn_loop not importable yet: {exc!r}")
    return run_turn_loop


@pytest.fixture(scope="session")
def offline_report(
    run_turn_loop_entry: Callable[..., Any],
    golden_conversations: Dict[str, Any],
):
    """Run the whole golden suite ONCE offline and share the report.

    A synchronous session fixture (no running event loop) so
    ``asyncio.run`` inside the harness wrapper is safe; every test then
    asserts against the same deterministic run.
    """
    from evals.conversation.harness import run_suite_offline_sync

    return run_suite_offline_sync(run_turn_loop_entry, golden_conversations)
