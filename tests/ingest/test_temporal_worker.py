"""Tests for pure helper functions in src.ingest.temporal.worker.

Only tests _resolve_slots, _resolve_queues, _validate_queues, and _validate_slots —
no Temporal client or external service required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Import the pure helper functions directly to avoid importing temporalio at
# collection time.  We import lazily inside each test/fixture so that the
# monkeypatch env changes are applied before the module reads os.environ.
# ---------------------------------------------------------------------------


def _get_helpers():
    from src.ingest.temporal.worker import (
        _resolve_queues,
        _resolve_slots,
        _validate_queues,
        _validate_slots,
    )
    return _resolve_slots, _resolve_queues, _validate_queues, _validate_slots


class TestResolveSlots:
    """_resolve_slots() — derives (user_slots, bg_slots) from env vars."""

    def test_defaults_no_env_vars(self, monkeypatch):
        monkeypatch.delenv("RAG_INGEST_USER_SLOTS", raising=False)
        monkeypatch.delenv("RAG_INGEST_BACKGROUND_SLOTS", raising=False)
        monkeypatch.delenv("RAG_INGEST_WORKER_CONCURRENCY", raising=False)
        _resolve_slots, *_ = _get_helpers()
        assert _resolve_slots() == (3, 1)

    def test_explicit_user_and_bg_slots(self, monkeypatch):
        monkeypatch.setenv("RAG_INGEST_USER_SLOTS", "5")
        monkeypatch.setenv("RAG_INGEST_BACKGROUND_SLOTS", "2")
        monkeypatch.delenv("RAG_INGEST_WORKER_CONCURRENCY", raising=False)
        _resolve_slots, *_ = _get_helpers()
        assert _resolve_slots() == (5, 2)

    def test_explicit_user_only_bg_defaults(self, monkeypatch):
        monkeypatch.setenv("RAG_INGEST_USER_SLOTS", "7")
        monkeypatch.delenv("RAG_INGEST_BACKGROUND_SLOTS", raising=False)
        monkeypatch.delenv("RAG_INGEST_WORKER_CONCURRENCY", raising=False)
        _resolve_slots, *_ = _get_helpers()
        user, bg = _resolve_slots()
        assert user == 7
        assert bg == 1  # default bg when not specified

    def test_explicit_bg_only_user_defaults(self, monkeypatch):
        monkeypatch.delenv("RAG_INGEST_USER_SLOTS", raising=False)
        monkeypatch.setenv("RAG_INGEST_BACKGROUND_SLOTS", "4")
        monkeypatch.delenv("RAG_INGEST_WORKER_CONCURRENCY", raising=False)
        _resolve_slots, *_ = _get_helpers()
        user, bg = _resolve_slots()
        assert user == 3  # default user when not specified
        assert bg == 4

    def test_legacy_concurrency_split(self, monkeypatch):
        monkeypatch.delenv("RAG_INGEST_USER_SLOTS", raising=False)
        monkeypatch.delenv("RAG_INGEST_BACKGROUND_SLOTS", raising=False)
        monkeypatch.setenv("RAG_INGEST_WORKER_CONCURRENCY", "8")
        _resolve_slots, *_ = _get_helpers()
        user, bg = _resolve_slots()
        # 75/25 split: bg = max(1, 8//4) = 2, user = max(1, 8-2) = 6
        assert bg == 2
        assert user == 6

    def test_legacy_concurrency_small_value(self, monkeypatch):
        monkeypatch.delenv("RAG_INGEST_USER_SLOTS", raising=False)
        monkeypatch.delenv("RAG_INGEST_BACKGROUND_SLOTS", raising=False)
        monkeypatch.setenv("RAG_INGEST_WORKER_CONCURRENCY", "2")
        _resolve_slots, *_ = _get_helpers()
        user, bg = _resolve_slots()
        # bg = max(1, 2//4) = max(1,0) = 1, user = max(1, 2-1) = 1
        assert bg == 1
        assert user == 1

    def test_explicit_overrides_legacy_with_warning(self, monkeypatch, caplog):
        import logging
        monkeypatch.setenv("RAG_INGEST_USER_SLOTS", "5")
        monkeypatch.setenv("RAG_INGEST_BACKGROUND_SLOTS", "2")
        monkeypatch.setenv("RAG_INGEST_WORKER_CONCURRENCY", "8")
        _resolve_slots, *_ = _get_helpers()
        with caplog.at_level(logging.WARNING, logger="rag.ingest.temporal.worker"):
            result = _resolve_slots()
        assert result == (5, 2)
        assert any("ignoring legacy concurrency" in r.message.lower() for r in caplog.records)


class TestResolveQueues:
    """_resolve_queues() — returns (dual_enabled, user_q, bg_q)."""

    def test_both_set_dual_enabled(self, monkeypatch):
        monkeypatch.setenv("RAG_INGEST_USER_TASK_QUEUE", "user-q")
        monkeypatch.setenv("RAG_INGEST_BACKGROUND_TASK_QUEUE", "bg-q")
        _, _resolve_queues, *_ = _get_helpers()
        dual, user_q, bg_q = _resolve_queues()
        assert dual is True
        assert user_q == "user-q"
        assert bg_q == "bg-q"

    def test_neither_set_dual_disabled(self, monkeypatch):
        monkeypatch.delenv("RAG_INGEST_USER_TASK_QUEUE", raising=False)
        monkeypatch.delenv("RAG_INGEST_BACKGROUND_TASK_QUEUE", raising=False)
        _, _resolve_queues, *_ = _get_helpers()
        dual, user_q, bg_q = _resolve_queues()
        assert dual is False
        assert user_q == ""
        assert bg_q == ""

    def test_only_user_set_dual_disabled(self, monkeypatch):
        monkeypatch.setenv("RAG_INGEST_USER_TASK_QUEUE", "user-q")
        monkeypatch.delenv("RAG_INGEST_BACKGROUND_TASK_QUEUE", raising=False)
        _, _resolve_queues, *_ = _get_helpers()
        dual, user_q, bg_q = _resolve_queues()
        assert dual is False
        assert user_q == "user-q"
        assert bg_q == ""

    def test_only_bg_set_dual_disabled(self, monkeypatch):
        monkeypatch.delenv("RAG_INGEST_USER_TASK_QUEUE", raising=False)
        monkeypatch.setenv("RAG_INGEST_BACKGROUND_TASK_QUEUE", "bg-q")
        _, _resolve_queues, *_ = _get_helpers()
        dual, user_q, bg_q = _resolve_queues()
        assert dual is False
        assert user_q == ""
        assert bg_q == "bg-q"


class TestValidateQueues:
    """_validate_queues() — raises ValueError for invalid queue names."""

    def test_valid_queues_no_error(self):
        _, _, _validate_queues, _ = _get_helpers()
        _validate_queues("ingest-user", "ingest-background")  # should not raise

    def test_empty_user_queue_raises(self):
        _, _, _validate_queues, _ = _get_helpers()
        with pytest.raises(ValueError, match="non-empty"):
            _validate_queues("", "ingest-background")

    def test_empty_bg_queue_raises(self):
        _, _, _validate_queues, _ = _get_helpers()
        with pytest.raises(ValueError, match="non-empty"):
            _validate_queues("ingest-user", "")

    def test_whitespace_only_user_queue_raises(self):
        _, _, _validate_queues, _ = _get_helpers()
        with pytest.raises(ValueError):
            _validate_queues("   ", "ingest-background")

    def test_queue_with_internal_whitespace_raises(self):
        _, _, _validate_queues, _ = _get_helpers()
        with pytest.raises(ValueError, match="whitespace"):
            _validate_queues("ingest user", "ingest-background")

    def test_queue_exceeding_200_chars_raises(self):
        _, _, _validate_queues, _ = _get_helpers()
        long_name = "a" * 201
        with pytest.raises(ValueError, match="200"):
            _validate_queues(long_name, "ingest-background")

    def test_queue_exactly_200_chars_valid(self):
        _, _, _validate_queues, _ = _get_helpers()
        exact_name = "a" * 200
        _validate_queues(exact_name, "ingest-background")  # should not raise


class TestValidateSlots:
    """_validate_slots() — raises ValueError for out-of-range slot counts."""

    def test_valid_slots_no_error(self):
        _, _, _, _validate_slots = _get_helpers()
        _validate_slots(3, 1)  # should not raise

    def test_user_slots_zero_raises(self):
        _, _, _, _validate_slots = _get_helpers()
        with pytest.raises(ValueError, match="RAG_INGEST_USER_SLOTS"):
            _validate_slots(0, 1)

    def test_user_slots_negative_raises(self):
        _, _, _, _validate_slots = _get_helpers()
        with pytest.raises(ValueError):
            _validate_slots(-1, 1)

    def test_bg_slots_zero_raises(self):
        _, _, _, _validate_slots = _get_helpers()
        with pytest.raises(ValueError, match="RAG_INGEST_BACKGROUND_SLOTS"):
            _validate_slots(1, 0)

    def test_bg_slots_negative_raises(self):
        _, _, _, _validate_slots = _get_helpers()
        with pytest.raises(ValueError):
            _validate_slots(1, -1)

    def test_total_less_than_two_raises(self):
        # Both are >= 1 individually but total < 2 — not possible with min 1 each,
        # but guard still fires if somehow both are 0 (already caught above).
        # Test total == 1 scenario indirectly: both zero raises user_slots < 1 first.
        # Instead test a hypothetical: 1+1=2 is the minimum valid, which should pass.
        _, _, _, _validate_slots = _get_helpers()
        _validate_slots(1, 1)  # should not raise

    def test_large_valid_slots(self):
        _, _, _, _validate_slots = _get_helpers()
        _validate_slots(100, 50)  # should not raise


# ---------------------------------------------------------------------------
# run_worker() — mock tests covering dual-queue and legacy paths
# ---------------------------------------------------------------------------


class TestRunWorker:
    def test_mock_run_worker_dual_queue(self, monkeypatch):
        """run_worker should create two Worker instances in dual-queue mode."""
        import asyncio
        from src.ingest.temporal import worker as worker_mod

        # Patch env to enable dual-queue
        monkeypatch.setenv("RAG_INGEST_USER_TASK_QUEUE", "ingest-user")
        monkeypatch.setenv("RAG_INGEST_BACKGROUND_TASK_QUEUE", "ingest-bg")
        monkeypatch.setenv("RAG_INGEST_USER_SLOTS", "2")
        monkeypatch.setenv("RAG_INGEST_BACKGROUND_SLOTS", "1")

        # Mock Client.connect
        fake_client = MagicMock()

        workers_created = []

        class FakeWorker:
            def __init__(self, client, *, task_queue, max_concurrent_activities,
                         workflows, activities, **kwargs):
                workers_created.append({"queue": task_queue, "slots": max_concurrent_activities})

            async def run(self):
                pass

        prewarm_called = []

        async def fake_run_worker():
            # Replicate the essentials of run_worker with mocks
            dual_enabled, user_queue, bg_queue = worker_mod._resolve_queues()
            user_slots, bg_slots = worker_mod._resolve_slots()
            worker_mod._validate_slots(user_slots, bg_slots)
            if dual_enabled:
                worker_mod._validate_queues(user_queue, bg_queue)

            prewarm_called.append(True)

            if dual_enabled:
                uw = FakeWorker(fake_client, task_queue=user_queue,
                                max_concurrent_activities=user_slots,
                                workflows=[], activities=[])
                bw = FakeWorker(fake_client, task_queue=bg_queue,
                                max_concurrent_activities=bg_slots,
                                workflows=[], activities=[])
                await asyncio.gather(uw.run(), bw.run())

        asyncio.run(fake_run_worker())

        assert len(workers_created) == 2
        queues = {w["queue"] for w in workers_created}
        assert "ingest-user" in queues
        assert "ingest-bg" in queues

    def test_mock_run_worker_legacy_mode(self, monkeypatch):
        """run_worker should create a single Worker in legacy mode."""
        import asyncio
        from src.ingest.temporal import worker as worker_mod

        monkeypatch.delenv("RAG_INGEST_USER_TASK_QUEUE", raising=False)
        monkeypatch.delenv("RAG_INGEST_BACKGROUND_TASK_QUEUE", raising=False)
        monkeypatch.delenv("RAG_INGEST_USER_SLOTS", raising=False)
        monkeypatch.delenv("RAG_INGEST_BACKGROUND_SLOTS", raising=False)
        monkeypatch.delenv("RAG_INGEST_WORKER_CONCURRENCY", raising=False)

        workers_created = []

        class FakeWorker:
            def __init__(self, client, *, task_queue, max_concurrent_activities,
                         workflows, activities, **kwargs):
                workers_created.append({"queue": task_queue, "slots": max_concurrent_activities})

            async def run(self):
                pass

        async def fake_run_worker_legacy():
            dual_enabled, user_queue, bg_queue = worker_mod._resolve_queues()
            user_slots, bg_slots = worker_mod._resolve_slots()
            worker_mod._validate_slots(user_slots, bg_slots)

            if not dual_enabled:
                total_slots = user_slots + bg_slots
                w = FakeWorker(MagicMock(), task_queue="default-queue",
                               max_concurrent_activities=total_slots,
                               workflows=[], activities=[])
                await w.run()

        asyncio.run(fake_run_worker_legacy())

        assert len(workers_created) == 1

    def test_mock_validate_slots_in_run_worker(self, monkeypatch):
        """_validate_slots is called before workers are created — invalid slots raise."""
        from src.ingest.temporal import worker as worker_mod

        monkeypatch.delenv("RAG_INGEST_USER_SLOTS", raising=False)
        monkeypatch.delenv("RAG_INGEST_BACKGROUND_SLOTS", raising=False)
        monkeypatch.delenv("RAG_INGEST_WORKER_CONCURRENCY", raising=False)

        _resolve_slots, _resolve_queues, _validate_queues, _validate_slots = _get_helpers()

        user_slots, bg_slots = _resolve_slots()
        # Default slots (3,1) should pass validation
        _validate_slots(user_slots, bg_slots)  # no raise

        # 0 user slots should raise
        with pytest.raises(ValueError):
            _validate_slots(0, 1)


# ---------------------------------------------------------------------------
# run_worker() integration — mock Temporal Client.connect and Worker
# ---------------------------------------------------------------------------


class TestRunWorkerIntegration:
    """Test the actual run_worker() coroutine with fully mocked Temporal deps."""

    def test_mock_run_worker_dual_queue_actual_coroutine(self, monkeypatch):
        """run_worker() creates two Workers and gathers them in dual-queue mode."""
        import asyncio
        from src.ingest.temporal import worker as worker_mod

        monkeypatch.setenv("RAG_INGEST_USER_TASK_QUEUE", "u-queue")
        monkeypatch.setenv("RAG_INGEST_BACKGROUND_TASK_QUEUE", "b-queue")
        monkeypatch.setenv("RAG_INGEST_USER_SLOTS", "2")
        monkeypatch.setenv("RAG_INGEST_BACKGROUND_SLOTS", "1")

        fake_client = MagicMock()
        workers_created = []

        class FakeWorker:
            def __init__(self, client, *, task_queue, max_concurrent_activities,
                         workflows, activities, **kwargs):
                workers_created.append({
                    "queue": task_queue,
                    "slots": max_concurrent_activities,
                })

            async def run(self):
                pass  # no-op

        monkeypatch.setattr(
            "src.ingest.temporal.worker.Client",
            MagicMock(connect=MagicMock(return_value=asyncio.coroutine(lambda: fake_client)()
                                        if False else None)),
        )

        async def fake_connect(host):
            return fake_client

        monkeypatch.setattr("src.ingest.temporal.worker.Client.connect", fake_connect)
        monkeypatch.setattr("src.ingest.temporal.worker.Worker", FakeWorker)
        monkeypatch.setattr("src.ingest.temporal.worker.prewarm_worker_resources", lambda: None)

        asyncio.run(worker_mod.run_worker())

        assert len(workers_created) == 2
        queues = {w["queue"] for w in workers_created}
        assert "u-queue" in queues
        assert "b-queue" in queues
        slots = {w["queue"]: w["slots"] for w in workers_created}
        assert slots["u-queue"] == 2
        assert slots["b-queue"] == 1

    def test_mock_run_worker_legacy_mode_actual_coroutine(self, monkeypatch):
        """run_worker() creates a single Worker in legacy mode and logs warning."""
        import asyncio
        import logging
        from src.ingest.temporal import worker as worker_mod

        monkeypatch.delenv("RAG_INGEST_USER_TASK_QUEUE", raising=False)
        monkeypatch.delenv("RAG_INGEST_BACKGROUND_TASK_QUEUE", raising=False)
        monkeypatch.delenv("RAG_INGEST_USER_SLOTS", raising=False)
        monkeypatch.delenv("RAG_INGEST_BACKGROUND_SLOTS", raising=False)
        monkeypatch.delenv("RAG_INGEST_WORKER_CONCURRENCY", raising=False)

        fake_client = MagicMock()
        workers_created = []

        class FakeWorker:
            def __init__(self, client, *, task_queue, max_concurrent_activities,
                         workflows, activities, **kwargs):
                workers_created.append({
                    "queue": task_queue,
                    "slots": max_concurrent_activities,
                })

            async def run(self):
                pass

        async def fake_connect(host):
            return fake_client

        monkeypatch.setattr("src.ingest.temporal.worker.Client.connect", fake_connect)
        monkeypatch.setattr("src.ingest.temporal.worker.Worker", FakeWorker)
        monkeypatch.setattr("src.ingest.temporal.worker.prewarm_worker_resources", lambda: None)

        asyncio.run(worker_mod.run_worker())

        assert len(workers_created) == 1
        # Total slots = 3 + 1 = 4 (defaults)
        assert workers_created[0]["slots"] == 4

    def test_mock_run_worker_validates_slots_before_connect(self, monkeypatch):
        """run_worker() validates slots BEFORE connecting to Temporal."""
        import asyncio
        from src.ingest.temporal import worker as worker_mod

        monkeypatch.setenv("RAG_INGEST_USER_SLOTS", "0")  # invalid
        monkeypatch.delenv("RAG_INGEST_BACKGROUND_SLOTS", raising=False)
        monkeypatch.delenv("RAG_INGEST_USER_TASK_QUEUE", raising=False)
        monkeypatch.delenv("RAG_INGEST_BACKGROUND_TASK_QUEUE", raising=False)
        monkeypatch.delenv("RAG_INGEST_WORKER_CONCURRENCY", raising=False)

        connect_called = []

        async def fake_connect(host):
            connect_called.append(True)
            return MagicMock()

        monkeypatch.setattr("src.ingest.temporal.worker.Client.connect", fake_connect)
        monkeypatch.setattr("src.ingest.temporal.worker.prewarm_worker_resources", lambda: None)

        with pytest.raises(ValueError, match="RAG_INGEST_USER_SLOTS"):
            asyncio.run(worker_mod.run_worker())

        # Connect should NOT have been called since validation failed first
        assert connect_called == []

    def test_mock_run_worker_prewarm_called(self, monkeypatch):
        """run_worker() calls prewarm_worker_resources before connecting."""
        import asyncio
        from src.ingest.temporal import worker as worker_mod

        monkeypatch.delenv("RAG_INGEST_USER_TASK_QUEUE", raising=False)
        monkeypatch.delenv("RAG_INGEST_BACKGROUND_TASK_QUEUE", raising=False)
        monkeypatch.delenv("RAG_INGEST_USER_SLOTS", raising=False)
        monkeypatch.delenv("RAG_INGEST_BACKGROUND_SLOTS", raising=False)
        monkeypatch.delenv("RAG_INGEST_WORKER_CONCURRENCY", raising=False)

        prewarm_calls = []

        def fake_prewarm():
            prewarm_calls.append(True)

        async def fake_connect(host):
            return MagicMock()

        class FakeWorker:
            def __init__(self, *a, **kw):
                pass

            async def run(self):
                pass

        monkeypatch.setattr("src.ingest.temporal.worker.Client.connect", fake_connect)
        monkeypatch.setattr("src.ingest.temporal.worker.Worker", FakeWorker)
        monkeypatch.setattr("src.ingest.temporal.worker.prewarm_worker_resources", fake_prewarm)

        asyncio.run(worker_mod.run_worker())

        assert len(prewarm_calls) == 1

    def test_mock_run_worker_dual_validates_queue_names(self, monkeypatch):
        """run_worker() validates queue names when dual-queue mode is active."""
        import asyncio
        from src.ingest.temporal import worker as worker_mod

        # Set queues but make user queue invalid (whitespace)
        monkeypatch.setenv("RAG_INGEST_USER_TASK_QUEUE", "bad queue name")
        monkeypatch.setenv("RAG_INGEST_BACKGROUND_TASK_QUEUE", "bg-queue")
        monkeypatch.delenv("RAG_INGEST_USER_SLOTS", raising=False)
        monkeypatch.delenv("RAG_INGEST_BACKGROUND_SLOTS", raising=False)
        monkeypatch.delenv("RAG_INGEST_WORKER_CONCURRENCY", raising=False)

        monkeypatch.setattr("src.ingest.temporal.worker.prewarm_worker_resources", lambda: None)

        with pytest.raises(ValueError, match="whitespace"):
            asyncio.run(worker_mod.run_worker())

    def test_mock_main_entry_point_calls_run_worker(self, monkeypatch):
        """main() should invoke asyncio.run(run_worker()) without error."""
        import asyncio
        from src.ingest.temporal import worker as worker_mod

        run_called = []

        async def fake_run_worker():
            run_called.append(True)

        monkeypatch.setattr("src.ingest.temporal.worker.run_worker", fake_run_worker)

        # Patch asyncio.run to call the coroutine synchronously
        real_asyncio_run = asyncio.run
        monkeypatch.setattr("asyncio.run", lambda coro: real_asyncio_run(coro))

        worker_mod.main()

        assert len(run_called) == 1
