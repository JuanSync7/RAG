# @summary
# Tests for the pure pre-flight router (turn_loop/router.py): the precedence
# ladder (router-off -> neutral; compound -> DECOMPOSE seed; fast-lane -> RETRIEVE
# + fast_lane; else -> no seed), every fast-lane guard (disabled / compound /
# backward-reference / too-long / low-confidence each block it), the neutral
# fail-open hint, and RouteConfig.from_settings reading the RAG_TURN_LOOP_* keys.
# All pure: hand-built RouteSignals/RouteConfig, zero infrastructure.
# @end-summary
"""Tests for ``turn_loop/router.py`` (pure routing decision)."""

from __future__ import annotations

from src.retrieval.pipeline.turn_loop.router import (
    RouteConfig,
    RouteEffort,
    RouteHint,
    RouteSignals,
    route,
)
from src.retrieval.pipeline.turn_loop.schemas import TurnAction


def _signals(**overrides) -> RouteSignals:
    values = dict(
        query_confidence=0.85,
        word_count=8,
        is_compound=False,
        has_backward_reference=False,
    )
    values.update(overrides)
    return RouteSignals(**values)


def _config(**overrides) -> RouteConfig:
    values = dict(
        router_enabled=True,
        fast_lane_enabled=True,
        fast_lane_max_words=24,
        fast_lane_min_confidence=0.8,
    )
    values.update(overrides)
    return RouteConfig(**values)


# ── router master gate ────────────────────────────────────────────────────────

def test_router_disabled_returns_neutral_hint():
    # Even a textbook fast-lane query gets no seed when the router is off.
    hint = route(_signals(), _config(router_enabled=False))
    assert hint.initial_action is None
    assert hint.effort == RouteEffort.BALANCED
    assert hint.fast_lane is False


def test_neutral_classmethod_is_fail_open_shape():
    hint = RouteHint.neutral()
    assert hint.initial_action is None
    assert hint.effort == RouteEffort.BALANCED
    assert hint.fast_lane is False


# ── compound handling (LLM-driven, not a router regex seed) ───────────────────
# The router no longer seeds DECOMPOSE from a keyword-regex compound marker —
# the controller now classifies query_shape and coerces the opening move
# (test_controller). is_compound survives ONLY as a conservative fast-lane
# exclusion (never skip the controller for a possibly-multi-facet query).

def test_compound_no_longer_seeds_decompose():
    """A compound signal produces no router seed — the controller's query_shape
    classification owns the DECOMPOSE decision now (regex→LLM, CLAUDE.md §0)."""
    hint = route(_signals(is_compound=True), _config())
    assert hint.initial_action is None
    assert hint.fast_lane is False
    assert hint.effort == RouteEffort.BALANCED


def test_compound_still_blocks_the_fast_lane():
    """A short, high-confidence query that WOULD fast-lane is held back when it
    looks compound — the fast lane skips the controller, so it must never
    short-circuit a possibly-multi-facet question past the query_shape check."""
    hint = route(
        _signals(is_compound=True, word_count=6, query_confidence=0.95), _config()
    )
    assert hint.fast_lane is False
    assert hint.initial_action is None


# ── fast lane ─────────────────────────────────────────────────────────────────

def test_fast_lane_fires_for_short_confident_selfcontained_query():
    hint = route(_signals(word_count=8, query_confidence=0.85), _config())
    assert hint.initial_action == TurnAction.RETRIEVE
    assert hint.fast_lane is True
    assert hint.effort == RouteEffort.FAST


def test_fast_lane_blocked_when_disabled():
    hint = route(_signals(), _config(fast_lane_enabled=False))
    assert hint.fast_lane is False
    assert hint.initial_action is None  # default, no seed


def test_fast_lane_blocked_by_backward_reference():
    # "how does it compare?" needs memory resolution -> controller, not verbatim.
    hint = route(_signals(has_backward_reference=True), _config())
    assert hint.fast_lane is False
    assert hint.initial_action is None


def test_fast_lane_blocked_by_low_confidence():
    hint = route(_signals(query_confidence=0.4), _config())
    assert hint.fast_lane is False


def test_fast_lane_blocked_when_too_long():
    hint = route(_signals(word_count=40), _config(fast_lane_max_words=24))
    assert hint.fast_lane is False


def test_fast_lane_boundary_inclusive_on_words_and_confidence():
    # Exactly at both thresholds still qualifies (>= / <=).
    hint = route(
        _signals(word_count=24, query_confidence=0.8),
        _config(fast_lane_max_words=24, fast_lane_min_confidence=0.8),
    )
    assert hint.fast_lane is True


# ── default ───────────────────────────────────────────────────────────────────

def test_default_no_seed_for_midconfidence_singlefacet():
    # 3-5 word factoid (heuristic 0.7) is below the fast-lane bar -> controller.
    hint = route(_signals(word_count=4, query_confidence=0.7), _config())
    assert hint.initial_action is None
    assert hint.fast_lane is False
    assert hint.effort == RouteEffort.BALANCED


# ── config from settings ──────────────────────────────────────────────────────

def test_config_from_settings_reads_defaults(monkeypatch):
    import config.settings as settings

    for key, val in [
        ("RAG_TURN_LOOP_ROUTER_ENABLED", True),
        ("RAG_TURN_LOOP_FAST_LANE_ENABLED", True),
        ("RAG_TURN_LOOP_FAST_LANE_MAX_WORDS", 30),
        ("RAG_TURN_LOOP_FAST_LANE_MIN_CONFIDENCE", 0.75),
    ]:
        monkeypatch.setattr(settings, key, val, raising=False)

    cfg = RouteConfig.from_settings()
    assert cfg.router_enabled is True
    assert cfg.fast_lane_enabled is True
    assert cfg.fast_lane_max_words == 30
    assert cfg.fast_lane_min_confidence == 0.75
