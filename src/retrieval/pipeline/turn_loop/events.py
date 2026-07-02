# @summary
# Typed event emission for the turn loop: TurnEventEmitter appends every
# TurnEvent to the TurnState trace, forwards it to the injected async sink
# (honoring RAG_TURN_LOOP_STREAM_EVENTS), and swallows-and-logs sink errors so
# observability can never kill the loop. Also hosts the charged LLM-call
# wrapper (ledger charge + timing + llm_call event around every
# provider.agenerate) and the latest-event trace lookup used by the gate and
# the controller digest.
# Exports: TurnEventEmitter, latest_event_payload
# Deps: asyncio-free stdlib (logging, time),
#       src.retrieval.pipeline.turn_loop.schemas (config.settings lazily)
# @end-summary
"""Typed event emitter + instrumented LLM-call wrapper for the turn loop.

Every loop event flows through :class:`TurnEventEmitter`: it is appended to
``TurnState.events`` unconditionally (the trace is the result contract) and
forwarded to ``TurnLoopDeps.emit`` only when streaming is enabled
(``RAG_TURN_LOOP_STREAM_EVENTS``). A failing sink is logged and swallowed —
observability must never kill the loop (design §8).

The charged-call wrapper lives here too because the two concerns are one
invariant: *every* provider call must be charged to the single
``TurnBudget.max_llm_calls`` ledger AND produce an ``llm_call`` event.
Co-locating the wrapper with the emitter makes that invariant structural
instead of a per-stage convention.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from src.retrieval.pipeline.turn_loop.schemas import (
    TurnBudget,
    TurnEvent,
    TurnEventType,
    TurnLoopDeps,
    TurnState,
)

logger = logging.getLogger(__name__)


def latest_event_payload(state: TurnState, event_type: str) -> Optional[dict]:
    """Return the payload of the most recent event of ``event_type``.

    The trace doubles as loop memory for cross-stage signals that the frozen
    :class:`TurnState` has no dedicated field for (e.g. the latest
    ``judge_verdict`` confidence consumed by the answer gate, design §5).

    Args:
        state: The per-turn state whose event trace is scanned.
        event_type: One of the :class:`TurnEventType` constants.

    Returns:
        The newest matching payload dict, or ``None`` when no event of that
        type has been emitted this turn.
    """
    for event in reversed(state.events):
        if event.type == event_type:
            return event.payload
    return None


class TurnEventEmitter:
    """Per-turn event emitter bound to one ``(deps, state, budget)`` triple.

    Instantiated fresh inside ``run_turn_loop`` and discarded with the turn
    (same no-global rule as :class:`TurnState`).
    """

    def __init__(
        self,
        *,
        deps: TurnLoopDeps,
        state: TurnState,
        budget: TurnBudget,
        stream_events: Optional[bool] = None,
    ) -> None:
        """Bind the emitter to the turn's dependency seam and state.

        Args:
            deps: Injected capabilities; only ``deps.emit`` and
                ``deps.llm_provider`` are used here.
            state: The turn's mutable state (trace + LLM ledger).
            budget: The frozen per-turn budget (ledger cap + wall clock, used
                to derive per-call timeouts).
            stream_events: Overrides ``RAG_TURN_LOOP_STREAM_EVENTS`` when not
                ``None`` (tests / non-streaming callers); when ``None`` the
                setting is read lazily at construction.
        """
        self._deps = deps
        self._state = state
        self._budget = budget
        if stream_events is None:
            from config import settings

            stream_events = bool(
                getattr(settings, "RAG_TURN_LOOP_STREAM_EVENTS", True)
            )
        self._stream_events = stream_events

    async def emit(self, event_type: str, payload: dict) -> TurnEvent:
        """Record one typed event: append to the trace, forward to the sink.

        The trace append always happens (non-streaming callers read the trace
        from ``TurnLoopResult``); the sink forward is gated by the streaming
        flag and guarded so an emitter/SSE failure degrades to a log line —
        never an exception into the loop.

        Args:
            event_type: One of the :class:`TurnEventType` constants.
            payload: JSON-serializable event body (design §8 shapes).

        Returns:
            The recorded :class:`TurnEvent`.
        """
        event = TurnEvent.now(event_type, payload)
        self._state.events.append(event)
        if self._stream_events:
            try:
                await self._deps.emit(event)
            except Exception as exc:  # noqa: BLE001 — observability never kills the loop
                logger.warning(
                    "turn loop event sink failed for %s event: %s",
                    event_type,
                    exc,
                )
        return event

    async def emit_llm_call(
        self,
        *,
        alias: str,
        purpose: str,
        ms: int,
        response: Optional[Any] = None,
    ) -> None:
        """Emit one ``llm_call`` observability event.

        Token counts are included when the call produced an ``LLMResponse``
        (streamed drafts and failed calls carry timing only).

        Args:
            alias: Router model alias the call was made with.
            purpose: Loop-stage vocabulary — ``controller`` / ``judge`` /
                ``deep_study_read`` / ``clarify`` / ``draft`` / ``self_score``.
            ms: Wall-clock call duration in milliseconds.
            response: The ``LLMResponse`` when available (``prompt_tokens`` /
                ``completion_tokens`` are read via ``getattr``).
        """
        payload: dict = {"alias": alias, "purpose": purpose, "ms": int(ms)}
        if response is not None:
            payload["prompt_tokens"] = int(getattr(response, "prompt_tokens", 0) or 0)
            payload["completion_tokens"] = int(
                getattr(response, "completion_tokens", 0) or 0
            )
        await self.emit(TurnEventType.LLM_CALL, payload)

    def remaining_timeout_s(self) -> int:
        """Seconds of wall-clock budget left, floored at 1 (per-call timeout)."""
        remaining_ms = self._budget.wall_clock_ms - self._state.elapsed_ms()
        return max(1, int(remaining_ms / 1000))

    async def charged_call(
        self,
        *,
        alias: str,
        purpose: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Optional[Any]:
        """Run one non-streaming provider call under the turn's single ledger.

        Charges ``TurnState.llm_calls`` BEFORE the call (an attempted call
        consumes budget even when the provider fails, so a flaky endpoint can
        never spin the loop), derives the timeout from the remaining wall
        clock, and always emits an ``llm_call`` event. Fail-open: any provider
        exception (and an exhausted ledger) returns ``None`` — the caller
        applies its stage-specific safe default (design §6).

        Args:
            alias: Router model alias for the call.
            purpose: ``llm_call`` purpose vocabulary (see
                :meth:`emit_llm_call`).
            prompt: Single-user-message prompt text.
            temperature: Sampling temperature (decision calls use 0.0).
            max_tokens: Optional completion-token cap.

        Returns:
            The provider ``LLMResponse``, or ``None`` when the ledger is
            exhausted or the call failed.
        """
        if self._state.llm_calls >= self._budget.max_llm_calls:
            logger.warning(
                "turn loop LLM budget exhausted (%d/%d) — skipping %s call",
                self._state.llm_calls,
                self._budget.max_llm_calls,
                purpose,
            )
            return None
        self._state.charge_llm_call()
        kwargs: dict = {
            "model_alias": alias,
            "temperature": temperature,
            "timeout": self.remaining_timeout_s(),
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        started = time.perf_counter()
        response: Optional[Any] = None
        try:
            response = await self._deps.llm_provider.agenerate(
                [{"role": "user", "content": prompt}],
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 — fail open to the stage default
            logger.warning("turn loop %s LLM call failed: %s", purpose, exc)
        ms = int((time.perf_counter() - started) * 1000)
        await self.emit_llm_call(alias=alias, purpose=purpose, ms=ms, response=response)
        return response


__all__ = ["TurnEventEmitter", "latest_event_payload"]
