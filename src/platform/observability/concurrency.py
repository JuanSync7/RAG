# @summary
# Context-preserving submit helper for ThreadPoolExecutor.
# Captures the caller's contextvars snapshot (incl. OTel ambient span ctx)
# and runs the callable inside ctx.run(...) on the worker thread.
# Exports: submit_with_context
# Deps: contextvars, concurrent.futures
# @end-summary
"""Concurrency helpers that preserve observability context across threads.

OpenTelemetry's ambient span context lives in :mod:`contextvars`. Python's
:class:`concurrent.futures.ThreadPoolExecutor` does *not* propagate
contextvars to worker threads — workers start with an empty context. As a
result, any ``tracer.span(...)`` opened inside a worker becomes a fresh
trace root rather than a child of the caller's span.

:func:`submit_with_context` is a drop-in replacement for
``executor.submit(fn, *args, **kwargs)`` that captures the current
contextvars :class:`~contextvars.Context` on the caller side and runs the
callable inside ``ctx.run(...)`` on the worker. The OTel ambient span,
request-id contextvars, and any other contextvar state are preserved.
"""
from __future__ import annotations

import contextvars
from concurrent.futures import Executor, Future
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def submit_with_context(
    executor: Executor,
    fn: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> "Future[T]":
    """Submit ``fn`` to ``executor`` with the caller's contextvars preserved.

    Captures :func:`contextvars.copy_context` on the caller side and runs
    ``fn(*args, **kwargs)`` inside ``ctx.run(...)`` on the worker thread.
    This preserves the OpenTelemetry ambient span context, so any
    ``tracer.span(...)`` opened inside ``fn`` nests under the caller's
    current span. Returns a normal :class:`~concurrent.futures.Future`.

    Note: ``ctx.run`` runs the callable inside a *copy* of the caller's
    context. Mutations to contextvars inside ``fn`` are isolated to the
    worker and do not leak back to the caller (this matches the semantics
    of submitting via a fresh thread).
    """
    ctx = contextvars.copy_context()
    return executor.submit(ctx.run, fn, *args, **kwargs)


__all__ = ["submit_with_context"]
