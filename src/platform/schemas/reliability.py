"""Typed schemas for reliability and retry behavior.

These dataclasses define retry policies and operation metadata shared across
retry provider implementations.
"""
from __future__ import annotations


from dataclasses import dataclass
from typing import Callable, Optional, Type

from config.settings import (
    RAG_RELIABILITY_BACKOFF_MULTIPLIER,
    RAG_RELIABILITY_DEFAULT_MAX_ATTEMPTS,
    RAG_RELIABILITY_INITIAL_BACKOFF_S,
    RAG_RELIABILITY_MAX_BACKOFF_S,
)


@dataclass(frozen=True)
class RetryPolicy:
    """Retry policy for external service operations."""

    max_attempts: int = RAG_RELIABILITY_DEFAULT_MAX_ATTEMPTS
    initial_backoff_seconds: float = RAG_RELIABILITY_INITIAL_BACKOFF_S
    max_backoff_seconds: float = RAG_RELIABILITY_MAX_BACKOFF_S
    backoff_multiplier: float = RAG_RELIABILITY_BACKOFF_MULTIPLIER
    retryable_exceptions: tuple[Type[BaseException], ...] = (Exception,)


@dataclass(frozen=True)
class RetryOperation:
    """Operation metadata used by retry providers."""

    operation_name: str
    idempotency_key: Optional[str]
    timeout_seconds: Optional[float] = None
    policy: Optional[RetryPolicy] = None


RetryCallable = Callable[[], object]
