# @summary
# Tests for the turn-loop controller stage: valid decision parsing (action +
# typed args), the design-§6 fail-open ladder (unparseable JSON / provider
# error / exhausted ledger -> RETRIEVE-verbatim on iteration 1, ANSWER later),
# empty-query normalization, full prompt-variable substitution, and the
# deterministic evidence digest (chunk one-liners carry document_id/source_key
# verbatim; judge missing_information surfaces from the trace).
# @end-summary
"""Tests for ``turn_loop/controller.py``."""

from __future__ import annotations

from src.retrieval.pipeline.turn_loop.controller import (
    build_evidence_digest,
    decide,
)
from src.retrieval.pipeline.turn_loop.events import TurnEventEmitter
from src.retrieval.pipeline.turn_loop.router import RouteHint
from src.retrieval.pipeline.turn_loop.schemas import (
    DecomposeArgs,
    DeepStudyArgs,
    QueryShape,
    RetrieveArgs,
    TurnAction,
    TurnDecision,
    TurnEvent,
    TurnEventType,
    TurnState,
)

from tests.retrieval.turn_loop.conftest import (
    FakeProvider,
    decision_json,
    make_budget,
    make_chunk,
    make_deps,
)


def _emitter(provider, state, budget):
    deps, events = make_deps(provider)
    return (
        TurnEventEmitter(deps=deps, state=state, budget=budget, stream_events=True),
        events,
    )


async def test_valid_decision_parses_action_and_args(empty_context):
    provider = FakeProvider(
        responses=[
            decision_json(
                "DEEP_STUDY",
                question="what are the timing values?",
                document_id="doc-9",
                source_key="docs/doc-9.md",
            )
        ]
    )
    state, budget = TurnState(), make_budget()
    emitter, _ = _emitter(provider, state, budget)

    decision = await decide(
        query="q", context=empty_context, state=state, budget=budget, emitter=emitter
    )

    assert decision.action == TurnAction.DEEP_STUDY
    assert isinstance(decision.args, DeepStudyArgs)
    assert decision.args.document_id == "doc-9"
    assert decision.args.source_key == "docs/doc-9.md"


async def test_parse_failure_fails_open_to_retrieve_verbatim_on_iteration_one(
    empty_context,
):
    """Unusable controller JSON on the first iteration -> RETRIEVE the user query."""
    provider = FakeProvider(responses=["this is not json at all"])
    state, budget = TurnState(), make_budget()
    emitter, _ = _emitter(provider, state, budget)

    decision = await decide(
        query="how do I set up X?",
        context=empty_context,
        state=state,
        budget=budget,
        emitter=emitter,
    )

    assert decision.action == TurnAction.RETRIEVE
    assert isinstance(decision.args, RetrieveArgs)
    assert decision.args.query_text == "how do I set up X?"


async def test_parse_failure_fails_open_to_answer_after_iteration_one(empty_context):
    """Unusable controller JSON after actions were dispatched -> ANSWER attempt."""
    provider = FakeProvider(responses=['{"action": "LAUNCH_ROCKETS"}'])
    state, budget = TurnState(), make_budget()
    state.iteration = 2
    emitter, _ = _emitter(provider, state, budget)

    decision = await decide(
        query="q", context=empty_context, state=state, budget=budget, emitter=emitter
    )

    assert decision.action == TurnAction.ANSWER


async def test_provider_error_fails_open(empty_context):
    provider = FakeProvider(responses=[RuntimeError("endpoint down")])
    state, budget = TurnState(), make_budget()
    emitter, _ = _emitter(provider, state, budget)

    decision = await decide(
        query="q", context=empty_context, state=state, budget=budget, emitter=emitter
    )

    assert decision.action == TurnAction.RETRIEVE
    # The attempted call still consumed ledger budget (charge-before-call).
    assert state.llm_calls == 1


async def test_exhausted_ledger_fails_open_without_provider_call(empty_context):
    provider = FakeProvider(responses=[decision_json("ANSWER")])
    state, budget = TurnState(), make_budget(max_llm_calls=1)
    state.llm_calls = 1
    emitter, _ = _emitter(provider, state, budget)

    decision = await decide(
        query="q", context=empty_context, state=state, budget=budget, emitter=emitter
    )

    assert decision.action == TurnAction.RETRIEVE
    assert provider.calls == []  # never reached the provider


async def test_empty_retrieve_query_normalized_to_verbatim_query(empty_context):
    provider = FakeProvider(responses=[decision_json("RETRIEVE", query_text="")])
    state, budget = TurnState(), make_budget()
    emitter, _ = _emitter(provider, state, budget)

    decision = await decide(
        query="the real question",
        context=empty_context,
        state=state,
        budget=budget,
        emitter=emitter,
    )

    assert isinstance(decision.args, RetrieveArgs)
    assert decision.args.query_text == "the real question"


async def test_prompt_substitutes_every_template_variable(empty_context):
    provider = FakeProvider(responses=[decision_json("ANSWER")])
    state, budget = TurnState(), make_budget()
    emitter, _ = _emitter(provider, state, budget)

    await decide(
        query="q", context=empty_context, state=state, budget=budget, emitter=emitter
    )

    _, messages, _ = provider.calls[0]
    prompt = messages[0]["content"]
    assert "{{" not in prompt  # every {{ var }} token was rendered
    assert "q" in prompt


async def test_prompt_asks_for_query_shape_classification(empty_context):
    """Contract guard: the controller prompt must instruct the model to emit
    query_shape with the compound/comparison guidance the coercion depends on
    (a live-validated wording — losing it silently regresses the c002 fix, which
    unit tests can't catch since they script query_shape directly)."""
    provider = FakeProvider(responses=[decision_json("ANSWER")])
    state, budget = TurnState(), make_budget()
    emitter, _ = _emitter(provider, state, budget)

    await decide(
        query="q", context=empty_context, state=state, budget=budget, emitter=emitter
    )

    prompt = provider.calls[0][1][0]["content"]
    assert "query_shape" in prompt
    for token in ("compound", "single_facet"):
        assert token in prompt
    # the comparison-is-compound rule (the exact class the c002 model missed)
    assert "comparison" in prompt.lower()


async def test_route_hint_renders_into_first_controller_prompt(empty_context):
    """An advisory (non-fast-lane) seed is rendered into the iteration-0 prompt;
    the controller still decides freely (fail-open)."""
    provider = FakeProvider(responses=[decision_json("RETRIEVE", query_text="x")])
    state, budget = TurnState(), make_budget()
    emitter, _ = _emitter(provider, state, budget)
    hint = RouteHint(
        initial_action=TurnAction.DECOMPOSE,
        effort="balanced",
        fast_lane=False,
        reason="compound query (multiple facets)",
    )

    await decide(
        query="a and b",
        context=empty_context,
        state=state,
        budget=budget,
        emitter=emitter,
        route_hint=hint,
    )

    prompt = provider.calls[0][1][0]["content"]
    assert "DECOMPOSE" in prompt
    assert "compound query (multiple facets)" in prompt
    assert "advice" in prompt  # framed as advisory, not an instruction


async def test_route_hint_not_rendered_after_first_iteration(empty_context):
    """The hint seeds only the OPENING move — later iterations render (none)."""
    provider = FakeProvider(responses=[decision_json("ANSWER")])
    state, budget = TurnState(), make_budget()
    state.iteration = 2
    emitter, _ = _emitter(provider, state, budget)
    hint = RouteHint(
        initial_action=TurnAction.DECOMPOSE, effort="balanced", reason="x"
    )

    await decide(
        query="q", context=empty_context, state=state, budget=budget,
        emitter=emitter, route_hint=hint,
    )

    prompt = provider.calls[0][1][0]["content"]
    assert "DECOMPOSE because" not in prompt  # the hint sentence is absent
    assert "{{" not in prompt  # router_hint slot still rendered (as "(none)")


def test_evidence_digest_carries_verbatim_refs_and_judge_gaps():
    """Digest lines expose document_id/source_key (DEEP_STUDY copy source) and
    the latest judge verdict's missing_information from the trace."""
    state = TurnState()
    state.pool.append(make_chunk("c1", document_id="doc-7", source_key="docs/d7.md"))
    state.tried_queries.append("first query")
    state.events.append(
        TurnEvent.now(
            TurnEventType.JUDGE_VERDICT,
            {
                "round": 0,
                "kept": 1,
                "sufficient": False,
                "confidence": 0.4,
                "missing_information": "the frequency limits table",
            },
        )
    )

    digest = build_evidence_digest(state)

    assert "document_id=doc-7" in digest
    assert "source_key=docs/d7.md" in digest
    assert "first query" in digest
    assert "the frequency limits table" in digest


def test_evidence_digest_is_deterministic_and_handles_empty_pool():
    state = TurnState()
    assert build_evidence_digest(state) == build_evidence_digest(state)
    assert "no evidence" in build_evidence_digest(state)


# ── query_shape parsing (LLM-driven compound classification) ──────────────────
# The controller emits an intrinsic-shape label alongside its action; the loop
# reads it structurally instead of re-deriving compound-ness from a keyword
# regex (COMPOUND_MARKERS). Parsing must be tolerant (case/separator) and reject
# anything outside the closed vocabulary (CLAUDE.md §0 — no literal matching).

def test_from_llm_json_parses_query_shape():
    decision = TurnDecision.from_llm_json(
        {
            "action": "RETRIEVE",
            "reason": "r",
            "confidence": 0.5,
            "query_shape": "compound",
            "args": {"query_text": "x"},
        }
    )
    assert decision is not None
    assert decision.query_shape == QueryShape.COMPOUND


def test_from_llm_json_query_shape_is_case_and_separator_insensitive():
    decision = TurnDecision.from_llm_json(
        {"action": "ANSWER", "query_shape": "Single-Facet"}
    )
    assert decision is not None
    assert decision.query_shape == QueryShape.SINGLE_FACET


def test_from_llm_json_unknown_query_shape_is_none():
    decision = TurnDecision.from_llm_json({"action": "ANSWER", "query_shape": "banana"})
    assert decision is not None
    assert decision.query_shape is None


def test_from_llm_json_missing_query_shape_is_none():
    decision = TurnDecision.from_llm_json({"action": "ANSWER"})
    assert decision is not None
    assert decision.query_shape is None


# ── shape-driven DECOMPOSE coercion (replaces the router's compound regex) ────
# When the controller labels the OPENING query compound but still chose a single
# RETRIEVE (the c002 pathology — it recognised the comparison in its own reason
# yet retrieved), the opening move is coerced to DECOMPOSE so the facets fan out
# in parallel. Safe only because DECOMPOSE is now additive (the raw-query anchor
# leg makes the fan-out a superset of the RETRIEVE it replaces).

async def test_shape_compound_coerces_opening_retrieve_to_decompose(empty_context):
    provider = FakeProvider(
        responses=[
            decision_json("RETRIEVE", query_text="just one facet", query_shape="compound")
        ]
    )
    state, budget = TurnState(), make_budget()
    emitter, _ = _emitter(provider, state, budget)

    decision = await decide(
        query="compare A and B",
        context=empty_context,
        state=state,
        budget=budget,
        emitter=emitter,
    )

    assert decision.action == TurnAction.DECOMPOSE
    assert isinstance(decision.args, DecomposeArgs)
    # The whole user question is what gets split, not the controller's narrow query.
    assert decision.args.question == "compare A and B"


async def test_shape_compound_does_not_coerce_deep_study(empty_context):
    """A deliberate DEEP_STUDY is a considered deep move — never overridden."""
    provider = FakeProvider(
        responses=[
            decision_json(
                "DEEP_STUDY",
                question="q",
                document_id="d1",
                query_shape="compound",
            )
        ]
    )
    state, budget = TurnState(), make_budget()
    emitter, _ = _emitter(provider, state, budget)

    decision = await decide(
        query="compare A and B",
        context=empty_context,
        state=state,
        budget=budget,
        emitter=emitter,
    )

    assert decision.action == TurnAction.DEEP_STUDY


async def test_shape_compound_decompose_action_passes_through_unchanged(empty_context):
    """When the controller ALREADY chose DECOMPOSE, coercion is a no-op — its own
    question is preserved (the coercion only rewrites an opening RETRIEVE)."""
    provider = FakeProvider(
        responses=[
            decision_json(
                "DECOMPOSE", question="the controller's own split", query_shape="compound"
            )
        ]
    )
    state, budget = TurnState(), make_budget()
    emitter, _ = _emitter(provider, state, budget)

    decision = await decide(
        query="compare A and B",
        context=empty_context,
        state=state,
        budget=budget,
        emitter=emitter,
    )

    assert decision.action == TurnAction.DECOMPOSE
    assert isinstance(decision.args, DecomposeArgs)
    assert decision.args.question == "the controller's own split"


async def test_shape_single_facet_retrieve_is_not_coerced(empty_context):
    provider = FakeProvider(
        responses=[decision_json("RETRIEVE", query_text="x", query_shape="single_facet")]
    )
    state, budget = TurnState(), make_budget()
    emitter, _ = _emitter(provider, state, budget)

    decision = await decide(
        query="what is X?",
        context=empty_context,
        state=state,
        budget=budget,
        emitter=emitter,
    )

    assert decision.action == TurnAction.RETRIEVE


async def test_shape_compound_only_coerces_the_opening_iteration(empty_context):
    """Mid-loop the controller reasons from evidence/gate state — coercing there
    could re-trigger the DECOMPOSE spiral, so only iteration 0 is coerced."""
    provider = FakeProvider(
        responses=[decision_json("RETRIEVE", query_text="x", query_shape="compound")]
    )
    state, budget = TurnState(), make_budget()
    state.iteration = 2
    emitter, _ = _emitter(provider, state, budget)

    decision = await decide(
        query="compare A and B",
        context=empty_context,
        state=state,
        budget=budget,
        emitter=emitter,
    )

    assert decision.action == TurnAction.RETRIEVE


async def test_shape_decompose_disabled_suppresses_coercion(empty_context):
    provider = FakeProvider(
        responses=[decision_json("RETRIEVE", query_text="x", query_shape="compound")]
    )
    state, budget = TurnState(), make_budget(shape_decompose_enabled=False)
    emitter, _ = _emitter(provider, state, budget)

    decision = await decide(
        query="compare A and B",
        context=empty_context,
        state=state,
        budget=budget,
        emitter=emitter,
    )

    assert decision.action == TurnAction.RETRIEVE


async def test_shape_coercion_requires_additive_decompose(empty_context):
    """Invariant guard (CLAUDE.md §3, adversarial-review finding): coercing
    RETRIEVE->DECOMPOSE is justified ONLY because DECOMPOSE is additive (the
    raw-query anchor leg makes the fan-out a superset of the RETRIEVE it
    replaces). With decompose_anchor_raw=False the fan-out is substitutive —
    the sub-query rewrite can steer retrieval OFF a doc the raw RETRIEVE matched
    (mode-D variance). So the coercion must NOT fire when the anchor is off,
    even if shape_decompose_enabled is True (the two flags are independent env
    booleans). Enforced at the decision point, not left to a prose 'safe
    because' claim."""
    provider = FakeProvider(
        responses=[decision_json("RETRIEVE", query_text="x", query_shape="compound")]
    )
    state, budget = TurnState(), make_budget(
        shape_decompose_enabled=True, decompose_anchor_raw=False
    )
    emitter, _ = _emitter(provider, state, budget)

    decision = await decide(
        query="compare A and B",
        context=empty_context,
        state=state,
        budget=budget,
        emitter=emitter,
    )

    assert decision.action == TurnAction.RETRIEVE
