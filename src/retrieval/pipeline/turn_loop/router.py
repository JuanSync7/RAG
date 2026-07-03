# @summary
# Pre-flight confidence router for the turn loop (design §"route itself"): a
# pure decision over cheap, already-computed query signals that seeds the
# turn's first action and effort as an ADVISORY hint — never a hard override.
# Compound queries seed DECOMPOSE; a short, high-confidence, self-contained
# factoid takes the fast lane (skip the first controller LLM call, go straight
# RETRIEVE->ANSWER); everything else gets no seed and the controller decides as
# today. Fail-open by construction: a disabled router (or neutral signals)
# returns an empty hint that degrades the loop to its exact pre-router baseline.
# Zero heavy imports (stdlib + config.settings lazily) so the pure loop package
# stays infrastructure-free; the runner gathers the signals and calls route().
# Exports: RouteEffort, RouteSignals, RouteConfig, RouteHint, route
# Deps: dataclasses, typing (config.settings lazily inside
#       RouteConfig.from_settings only); src.retrieval.pipeline.turn_loop.schemas
# @end-summary
"""Pre-flight confidence router: seed the turn's first action + effort.

One stage per file (CLAUDE.md §2). The router is a *pure function* of typed
signals (:class:`RouteSignals`) and typed config (:class:`RouteConfig`) — it
runs no LLM, touches no infrastructure, and re-implements no classifier: the
signals are gathered by the runner from the heuristics the codebase already
owns (``has_compound_marker`` / ``heuristic_confidence`` /
``has_backward_reference``), so identical signals always produce an identical
hint (evalable/cachable).

The output is a :class:`RouteHint` the orchestrator treats as ADVICE:

- ``initial_action`` is rendered into the first controller prompt as a
  suggestion the controller may override (fail-open — a wrong hint costs one
  ignored line, never a wrong turn).
- ``fast_lane`` is the one harder move: it lets the loop skip the first
  controller LLM call for a query where RETRIEVE->ANSWER is near-certainly
  right, and even then it degrades safely — a fast-lane answer that fails the
  gate hands control straight back to the controller.
- ``effort`` selects the :class:`~...schemas.TurnBudget` scale.

Class solved (CLAUDE.md §0): "route the query by its intrinsic shape", using
domain-neutral lexical/structural properties (compound-ness, self-containment,
length, heuristic confidence) — never a vendor/phrase/corpus match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# RouteEffort lives in schemas.py (the canonical contract module) so TurnBudget
# can reference it without importing the router; re-exported here for callers
# that reach for it alongside the router types.
from src.retrieval.pipeline.turn_loop.schemas import RouteEffort, TurnAction


@dataclass(frozen=True)
class RouteSignals:
    """Cheap pre-flight signals over the raw turn query + cross-turn context.

    All fields are pure, sub-millisecond lexical/structural properties the
    codebase already computes elsewhere (the runner is the one place that
    calls those heuristics; see ``turn_loop_runner.build_route_signals``). No
    LLM, no retrieval — the router must stay off the latency critical path.
    """

    query_confidence: float
    """Heuristic [0,1] confidence the query is well-formed / answerable as
    written (``query_processor.heuristic_confidence`` — word-count based)."""

    word_count: int
    """Whitespace token count of the raw query (length proxy)."""

    is_compound: bool
    """The query names several facets at once (``query_shape.has_compound_marker``:
    an "and"/"vs"/"compare" marker or 2+ question marks)."""

    has_backward_reference: bool
    """The query leans on conversation memory to resolve (pronoun density or an
    explicit back-reference marker, ``query_processor.has_backward_reference``)
    — such a query needs the controller's context reasoning, not a literal
    verbatim retrieval, so it is never fast-laned."""


@dataclass(frozen=True)
class RouteConfig:
    """Typed router thresholds (behavior config, CLAUDE.md §3).

    Built from ``RAG_TURN_LOOP_*`` via :meth:`from_settings`; hand-built in
    tests. Kept a frozen value object so the pure :func:`route` never reads
    settings itself (import-free, deterministic).
    """

    router_enabled: bool = True
    """Master gate. False → :func:`route` returns the empty hint (no seed,
    balanced effort, no fast lane) → the loop runs exactly as pre-router."""

    fast_lane_enabled: bool = False
    """Whether the fast lane may fire. Defaults False: skipping the controller
    is the router's one non-advisory move, so it stays opt-in until the routing
    eval confirms fast-lane p50/p95 matches linear (blueprint Phase 3 gate)."""

    fast_lane_max_words: int = 24
    """Upper length bound for the fast lane — excludes long, likely multi-part
    questions even when their heuristic confidence is high."""

    fast_lane_min_confidence: float = 0.8
    """Minimum heuristic confidence for the fast lane. At the default the
    ``heuristic_confidence`` word-count buckets admit only well-formed
    (>5-word) questions and exclude ultra-short ambiguous ones (<=2 words →
    0.4)."""

    decompose_on_compound: bool = True
    """Whether a compound query seeds the DECOMPOSE action (advisory)."""

    @classmethod
    def from_settings(cls) -> "RouteConfig":
        """Build the router config from the ``RAG_TURN_LOOP_*`` settings block.

        Lazy ``config.settings`` import so importing this module has zero
        config side effects (the pure-contract rule the whole package follows).
        """
        from config import settings

        return cls(
            router_enabled=bool(
                getattr(settings, "RAG_TURN_LOOP_ROUTER_ENABLED", True)
            ),
            fast_lane_enabled=bool(
                getattr(settings, "RAG_TURN_LOOP_FAST_LANE_ENABLED", False)
            ),
            fast_lane_max_words=int(
                getattr(settings, "RAG_TURN_LOOP_FAST_LANE_MAX_WORDS", 24)
            ),
            fast_lane_min_confidence=float(
                getattr(settings, "RAG_TURN_LOOP_FAST_LANE_MIN_CONFIDENCE", 0.8)
            ),
            decompose_on_compound=bool(
                getattr(
                    settings, "RAG_TURN_LOOP_ROUTER_DECOMPOSE_ON_COMPOUND", True
                )
            ),
        )


@dataclass(frozen=True)
class RouteHint:
    """The router's advisory verdict for one turn.

    Consumed by the orchestrator (``run_turn_loop(route_hint=...)``) and
    surfaced in ``metadata.turn_loop.router``. The empty hint
    (:meth:`neutral`) is the fail-open default — no seed, balanced effort, no
    fast lane — so ``route_hint is None`` and a neutral hint are equivalent to
    the pre-router loop.
    """

    initial_action: Optional[str] = None
    """A :class:`TurnAction` to suggest for the first iteration, or ``None``
    (let the controller open the turn). Advisory unless :attr:`fast_lane`."""

    effort: str = RouteEffort.BALANCED
    """The :class:`RouteEffort` level selecting the budget scale."""

    fast_lane: bool = False
    """When True the loop skips the first controller LLM call and runs a
    deterministic RETRIEVE->ANSWER, re-engaging the controller only if that
    answer fails the gate. Implies ``initial_action == RETRIEVE``."""

    reason: str = ""
    """Short human-readable justification (rendered into the controller prompt
    hint and the trace)."""

    @classmethod
    def neutral(cls, reason: str = "router disabled") -> "RouteHint":
        """The fail-open empty hint (no seed, balanced, no fast lane)."""
        return cls(
            initial_action=None,
            effort=RouteEffort.BALANCED,
            fast_lane=False,
            reason=reason,
        )


def route(signals: RouteSignals, config: RouteConfig) -> RouteHint:
    """Choose the turn's opening action + effort from cheap query signals.

    Precedence (most specific first):

    1. **Router off** → the neutral hint (pre-router baseline).
    2. **Compound query** (and ``decompose_on_compound``) → seed ``DECOMPOSE``,
       balanced effort. A multi-facet question is never fast-laned — its
       facets need the parallel fan-out, not a single verbatim retrieval.
    3. **Fast lane** (enabled; single-facet; self-contained — no backward
       reference; within ``fast_lane_max_words``; confidence at/above
       ``fast_lane_min_confidence``) → seed ``RETRIEVE`` with ``fast_lane`` and
       fast effort. This is the only hint that skips the controller.
    4. **Otherwise** → no seed, balanced effort (the controller decides, as
       today), but the loop still runs under the router (metadata records it).

    Pure function: no I/O, no settings read, no mutation of the inputs.

    Args:
        signals: The cheap pre-flight signals for this turn.
        config: The typed router thresholds.

    Returns:
        A :class:`RouteHint`. Always safe to ignore (fail-open); a wrong hint
        degrades to the controller's own judgment.
    """
    if not config.router_enabled:
        return RouteHint.neutral()

    # (2) Compound → DECOMPOSE seed (advisory). Takes priority over the fast
    # lane: a genuinely multi-facet question must not be short-circuited.
    if signals.is_compound and config.decompose_on_compound:
        return RouteHint(
            initial_action=TurnAction.DECOMPOSE,
            effort=RouteEffort.BALANCED,
            fast_lane=False,
            reason="compound query (multiple facets) — seed DECOMPOSE fan-out",
        )

    # (3) Fast lane: a short, self-contained, high-confidence single-facet
    # question where RETRIEVE->ANSWER is near-certainly the whole turn.
    if (
        config.fast_lane_enabled
        and not signals.is_compound
        and not signals.has_backward_reference
        and signals.word_count <= config.fast_lane_max_words
        and signals.query_confidence >= config.fast_lane_min_confidence
    ):
        return RouteHint(
            initial_action=TurnAction.RETRIEVE,
            effort=RouteEffort.FAST,
            fast_lane=True,
            reason=(
                "high-confidence self-contained factoid — fast lane "
                "(retrieve then answer, controller skipped)"
            ),
        )

    # (4) Default: no seed; the controller opens the turn as it does today.
    return RouteHint(
        initial_action=None,
        effort=RouteEffort.BALANCED,
        fast_lane=False,
        reason="no strong shape signal — controller decides",
    )


__all__ = [
    "RouteEffort",
    "RouteSignals",
    "RouteConfig",
    "RouteHint",
    "route",
]
