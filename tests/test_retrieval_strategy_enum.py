"""Contract tests for the `retrieval_strategy` enum waypoint (Phase 0b).

The enum collapses the former pairwise "X and Y cannot both be forced" validators
into ONE structural choice, while keeping the legacy booleans accepted and mapping
them <-> the enum so the downstream dispatch (which still reads the booleans) is
unchanged. These tests pin that two-way mapping and the preserved cross-field
contradictions, for BOTH request models (CLI/UI parity — one contract).
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from server.schemas import ConsoleQueryRequest, QueryRequest

MODELS = [QueryRequest, ConsoleQueryRequest]


def _mk(model, **kw):
    return model(query="x", **kw)


@pytest.mark.parametrize("model", MODELS)
def test_default_is_auto_booleans_untouched(model):
    req = _mk(model)
    assert req.retrieval_strategy == "auto"
    assert req.deep_research is False
    assert req.agentic_retrieval is None
    assert req.tree_retrieval is None
    assert req.turn_loop is None


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"deep_research": True}, "deep_research"),
        ({"agentic_retrieval": True}, "agentic"),
        ({"tree_retrieval": True}, "tree"),
        ({"turn_loop": True}, "turn_loop"),
    ],
)
def test_legacy_boolean_resolves_to_strategy(model, kwargs, expected):
    """A forced legacy boolean is reflected in retrieval_strategy."""
    req = _mk(model, **kwargs)
    assert req.retrieval_strategy == expected


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "kwargs,attr",
    [
        ({"retrieval_strategy": "deep_research"}, "deep_research"),
        ({"retrieval_strategy": "agentic"}, "agentic_retrieval"),
        ({"retrieval_strategy": "tree"}, "tree_retrieval"),
        ({"retrieval_strategy": "turn_loop"}, "turn_loop"),
    ],
)
def test_enum_maps_onto_boolean_for_dispatch(model, kwargs, attr):
    """An explicit enum choice sets the matching boolean True (dispatch reads it)
    and forces the competing booleans off (explicit strategy overrides config)."""
    req = _mk(model, **kwargs)
    assert getattr(req, attr) is True
    others = {"deep_research", "agentic_retrieval", "tree_retrieval", "turn_loop"} - {attr}
    for o in others:
        assert getattr(req, o) in (False, None) and getattr(req, o) is not True


@pytest.mark.parametrize("model", MODELS)
def test_linear_forces_all_orchestrators_off(model):
    req = _mk(model, retrieval_strategy="linear")
    assert req.retrieval_strategy == "linear"
    assert req.deep_research is False
    assert req.agentic_retrieval is False
    assert req.tree_retrieval is False
    assert req.turn_loop is False


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "kwargs",
    [
        {"deep_research": True, "agentic_retrieval": True},
        {"deep_research": True, "tree_retrieval": True},   # newly caught (was latent)
        {"agentic_retrieval": True, "turn_loop": True},
        {"tree_retrieval": True, "turn_loop": True},
    ],
)
def test_two_forced_orchestrators_rejected(model, kwargs):
    """Mutual exclusion is preserved (and now covers every pair structurally)."""
    with pytest.raises(ValidationError):
        _mk(model, **kwargs)


@pytest.mark.parametrize("model", MODELS)
def test_enum_conflicting_with_boolean_rejected(model):
    with pytest.raises(ValidationError):
        _mk(model, retrieval_strategy="agentic", deep_research=True)


@pytest.mark.parametrize("model", MODELS)
def test_linear_conflicting_with_forced_boolean_rejected(model):
    with pytest.raises(ValidationError):
        _mk(model, retrieval_strategy="linear", turn_loop=True)


@pytest.mark.parametrize("model", MODELS)
def test_turn_loop_incompatible_with_retrieval_mode(model):
    """Preserved: turn_loop's terminal action IS generation; retrieval mode skips it."""
    with pytest.raises(ValidationError):
        _mk(model, turn_loop=True, mode="retrieval")
    with pytest.raises(ValidationError):
        _mk(model, retrieval_strategy="turn_loop", mode="retrieval")


@pytest.mark.parametrize("model", MODELS)
def test_fast_path_contradiction_preserved(model):
    """Preserved: fast_path forces a single agentic round."""
    with pytest.raises(ValidationError):
        _mk(model, agentic_retrieval=True, fast_path=True, max_agentic_rounds=2)


@pytest.mark.parametrize("model", MODELS)
def test_unknown_strategy_rejected(model):
    with pytest.raises(ValidationError):
        _mk(model, retrieval_strategy="quantum")


@pytest.mark.parametrize("model", MODELS)
def test_redundant_enum_and_matching_boolean_ok(model):
    """Sending the enum AND its matching boolean is consistent, not a conflict."""
    req = _mk(model, retrieval_strategy="turn_loop", turn_loop=True)
    assert req.retrieval_strategy == "turn_loop"
    assert req.turn_loop is True
