# @summary
# Session-level state + cycle helper for the per-request tree-retrieval
# toggle (TREE_RETRIEVAL_DESIGN.md §6). Keeps a single source of truth
# shared by CLI, web console, and any other interactive surfaces so they
# behave identically.
# Exports: get_tree_state, set_tree_state, cycle_tree_state, label_tree_state
# Deps: stdlib only
# @end-summary
"""Session-level toggle state for tree retrieval.

State values:
    None  → use config default (``RAG_TREE_RETRIEVAL_ENABLED``)
    True  → force tree retrieval on for this session
    False → force tree retrieval off for this session

The state is process-local; one CLI session is one Python process. Web
sessions hold their own state in the request layer.
"""
from __future__ import annotations

from typing import Optional

_state: Optional[bool] = None


def get_tree_state() -> Optional[bool]:
    """Return the current session tree-retrieval override (None/True/False)."""
    return _state


def set_tree_state(value: Optional[bool]) -> Optional[bool]:
    """Set the session tree-retrieval override and return the new value."""
    global _state
    if value is not None and not isinstance(value, bool):
        raise TypeError("tree session state must be None, True, or False")
    _state = value
    return _state


def cycle_tree_state(current: Optional[bool]) -> Optional[bool]:
    """Cycle: default (None) → on (True) → off (False) → default (None)."""
    if current is None:
        return True
    if current is True:
        return False
    return None


def label_tree_state(value: Optional[bool]) -> str:
    """Human-readable label for a tree state value."""
    if value is None:
        return "default"
    return "on" if value else "off"


__all__ = [
    "get_tree_state",
    "set_tree_state",
    "cycle_tree_state",
    "label_tree_state",
]
