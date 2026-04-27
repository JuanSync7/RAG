"""Tests for src.ingest.temporal.constants — pure-logic routing helpers.

All tests use monkeypatch + importlib.reload to exercise the module-level
env-var resolution paths.
"""

from __future__ import annotations

import importlib
import sys


def _reload_constants():
    """Force a clean reload of the constants module so env-var defaults are re-evaluated."""
    mod_name = "src.ingest.temporal.constants"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


class TestTriggerStringConstants:
    """Module-level trigger-type string constants."""

    def test_trigger_single_value(self):
        from src.ingest.temporal.constants import TRIGGER_SINGLE
        assert TRIGGER_SINGLE == "single"

    def test_trigger_batch_value(self):
        from src.ingest.temporal.constants import TRIGGER_BATCH
        assert TRIGGER_BATCH == "batch"

    def test_trigger_background_value(self):
        from src.ingest.temporal.constants import TRIGGER_BACKGROUND
        assert TRIGGER_BACKGROUND == "background"


class TestTriggerToPriority:
    """trigger_to_priority routing function."""

    def test_single_returns_high(self):
        from src.ingest.temporal.constants import (
            PRIORITY_HIGH,
            TRIGGER_SINGLE,
            trigger_to_priority,
        )
        assert trigger_to_priority(TRIGGER_SINGLE) == PRIORITY_HIGH

    def test_batch_returns_medium(self):
        from src.ingest.temporal.constants import (
            PRIORITY_MEDIUM,
            TRIGGER_BATCH,
            trigger_to_priority,
        )
        assert trigger_to_priority(TRIGGER_BATCH) == PRIORITY_MEDIUM

    def test_background_returns_low(self):
        from src.ingest.temporal.constants import (
            PRIORITY_LOW,
            TRIGGER_BACKGROUND,
            trigger_to_priority,
        )
        assert trigger_to_priority(TRIGGER_BACKGROUND) == PRIORITY_LOW

    def test_unknown_falls_back_to_low(self):
        from src.ingest.temporal.constants import PRIORITY_LOW, trigger_to_priority
        assert trigger_to_priority("unknown_trigger") == PRIORITY_LOW

    def test_empty_string_falls_back_to_low(self):
        from src.ingest.temporal.constants import PRIORITY_LOW, trigger_to_priority
        assert trigger_to_priority("") == PRIORITY_LOW


class TestTriggerToQueue:
    """trigger_to_queue routing function."""

    def test_single_returns_user_queue(self):
        from src.ingest.temporal.constants import (
            QUEUE_USER,
            TRIGGER_SINGLE,
            trigger_to_queue,
        )
        assert trigger_to_queue(TRIGGER_SINGLE) == QUEUE_USER

    def test_batch_returns_user_queue(self):
        from src.ingest.temporal.constants import (
            QUEUE_USER,
            TRIGGER_BATCH,
            trigger_to_queue,
        )
        assert trigger_to_queue(TRIGGER_BATCH) == QUEUE_USER

    def test_background_returns_background_queue(self):
        from src.ingest.temporal.constants import (
            QUEUE_BACKGROUND,
            TRIGGER_BACKGROUND,
            trigger_to_queue,
        )
        assert trigger_to_queue(TRIGGER_BACKGROUND) == QUEUE_BACKGROUND

    def test_unknown_falls_back_to_background_queue(self):
        from src.ingest.temporal.constants import QUEUE_BACKGROUND, trigger_to_queue
        assert trigger_to_queue("totally_unknown") == QUEUE_BACKGROUND


class TestPriorityEnvVarOverrides:
    """Priority constants resolved from env vars at import time (FR-3572)."""

    def test_default_priority_values(self):
        from src.ingest.temporal.constants import (
            PRIORITY_HIGH,
            PRIORITY_LOW,
            PRIORITY_MEDIUM,
        )
        assert PRIORITY_HIGH == 1
        assert PRIORITY_MEDIUM == 2
        assert PRIORITY_LOW == 3

    def test_priority_high_override(self, monkeypatch):
        monkeypatch.setenv("RAG_INGEST_PRIORITY_HIGH", "5")
        monkeypatch.setenv("RAG_INGEST_PRIORITY_MEDIUM", "10")
        monkeypatch.setenv("RAG_INGEST_PRIORITY_LOW", "20")
        mod = _reload_constants()
        assert mod.PRIORITY_HIGH == 5
        assert mod.PRIORITY_MEDIUM == 10
        assert mod.PRIORITY_LOW == 20

    def test_priority_high_env_var_only(self, monkeypatch):
        monkeypatch.setenv("RAG_INGEST_PRIORITY_HIGH", "7")
        monkeypatch.delenv("RAG_INGEST_PRIORITY_MEDIUM", raising=False)
        monkeypatch.delenv("RAG_INGEST_PRIORITY_LOW", raising=False)
        mod = _reload_constants()
        assert mod.PRIORITY_HIGH == 7
        # defaults unchanged
        assert mod.PRIORITY_MEDIUM == 2
        assert mod.PRIORITY_LOW == 3


class TestQueueEnvVarResolution:
    """Queue name resolution from env vars with legacy fallback."""

    def test_dual_queue_env_vars_set(self, monkeypatch):
        monkeypatch.setenv("RAG_INGEST_USER_TASK_QUEUE", "ingest-user")
        monkeypatch.setenv("RAG_INGEST_BACKGROUND_TASK_QUEUE", "ingest-bg")
        mod = _reload_constants()
        assert mod.QUEUE_USER == "ingest-user"
        assert mod.QUEUE_BACKGROUND == "ingest-bg"

    def test_legacy_fallback_when_env_vars_unset(self, monkeypatch):
        monkeypatch.delenv("RAG_INGEST_USER_TASK_QUEUE", raising=False)
        monkeypatch.delenv("RAG_INGEST_BACKGROUND_TASK_QUEUE", raising=False)
        mod = _reload_constants()
        # Falls back to the legacy TEMPORAL_TASK_QUEUE from config.settings
        from config.settings import TEMPORAL_TASK_QUEUE
        assert mod.QUEUE_USER == TEMPORAL_TASK_QUEUE
        assert mod.QUEUE_BACKGROUND == TEMPORAL_TASK_QUEUE

    def test_only_user_queue_set_uses_legacy_for_bg(self, monkeypatch):
        monkeypatch.setenv("RAG_INGEST_USER_TASK_QUEUE", "my-user-q")
        monkeypatch.delenv("RAG_INGEST_BACKGROUND_TASK_QUEUE", raising=False)
        mod = _reload_constants()
        assert mod.QUEUE_USER == "my-user-q"
        from config.settings import TEMPORAL_TASK_QUEUE
        assert mod.QUEUE_BACKGROUND == TEMPORAL_TASK_QUEUE
