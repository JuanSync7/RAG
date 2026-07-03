import pytest
from pydantic import ValidationError

from server.schemas import ConsoleQueryRequest, QueryRequest


def test_query_request_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        QueryRequest(query="hello", unexpected=True)


def test_query_request_rejects_unknown_stage_budget_key():
    with pytest.raises(ValidationError):
        QueryRequest(
            query="hello",
            stage_budget_overrides={"unknown_stage": 1000},
        )


def test_query_request_accepts_valid_stage_budget_overrides():
    req = QueryRequest(
        query="hello",
        stage_budget_overrides={
            "query_processing": 1200,
            "hybrid_search": 900,
        },
    )
    assert req.stage_budget_overrides["query_processing"] == 1200


def test_query_request_accepts_memory_fields():
    req = QueryRequest(
        query="follow-up question",
        conversation_id="conv_123abc",
        memory_enabled=True,
        memory_turn_window=8,
        compact_now=False,
    )
    assert req.conversation_id == "conv_123abc"
    assert req.memory_enabled is True
    assert req.memory_turn_window == 8


def test_query_request_missing_required_field_raises():
    """A QueryRequest without the required `query` field raises ValidationError."""
    with pytest.raises(ValidationError):
        QueryRequest()  # type: ignore[call-arg]


def test_query_request_out_of_range_alpha_raises():
    """alpha must be between 0.0 and 1.0 inclusive."""
    with pytest.raises(ValidationError):
        QueryRequest(query="hello", alpha=2.5)


def test_query_request_out_of_range_search_limit_raises():
    """search_limit must be between 1 and 100."""
    with pytest.raises(ValidationError):
        QueryRequest(query="hello", search_limit=0)

    with pytest.raises(ValidationError):
        QueryRequest(query="hello", search_limit=101)


def test_query_request_out_of_range_rerank_top_k_raises():
    """rerank_top_k must be between 1 and 50."""
    with pytest.raises(ValidationError):
        QueryRequest(query="hello", rerank_top_k=0)

    with pytest.raises(ValidationError):
        QueryRequest(query="hello", rerank_top_k=51)


def test_query_request_valid_minimal():
    """A QueryRequest with only the required field should be accepted."""
    req = QueryRequest(query="what is RAG?")
    assert req.query == "what is RAG?"
    assert req.alpha == 0.5
    assert req.search_limit == 10


# ---------------------------------------------------------------------------
# Query length bounds — guards against regression of the 500/2000-char limits
# ---------------------------------------------------------------------------


QUERY_MAX = 32000


def test_query_request_accepts_long_query_up_to_cap():
    """Long-form queries (error logs, multi-paragraph context) up to 32000 chars."""
    long_query = "a" * QUERY_MAX
    req = QueryRequest(query=long_query)
    assert len(req.query) == QUERY_MAX


def test_query_request_accepts_query_above_old_2000_cap():
    """Regression guard: 5000-char queries used to 422; they must now pass."""
    req = QueryRequest(query="x" * 5000)
    assert len(req.query) == 5000


def test_query_request_rejects_query_over_cap():
    with pytest.raises(ValidationError):
        QueryRequest(query="a" * (QUERY_MAX + 1))


def test_query_request_rejects_empty_query():
    with pytest.raises(ValidationError):
        QueryRequest(query="")


def test_console_query_request_accepts_long_query_up_to_cap():
    """ConsoleQueryRequest must share the same 32000-char ceiling as QueryRequest."""
    req = ConsoleQueryRequest(query="a" * QUERY_MAX)
    assert len(req.query) == QUERY_MAX


def test_console_query_request_rejects_query_over_cap():
    with pytest.raises(ValidationError):
        ConsoleQueryRequest(query="a" * (QUERY_MAX + 1))


# ---------------------------------------------------------------------------
# Turn-loop mutual exclusion (TURN_LOOP_DESIGN.md §5): a request may carry at
# most ONE per-turn orchestrator; retrieval-only mode skips the generation the
# loop's terminal action IS. Enforced identically on both request surfaces
# (CLI/UI parity contract).
# ---------------------------------------------------------------------------
TURN_LOOP_COMPETING_FIELDS = [
    {"deep_research": True},
    {"agentic_retrieval": True},
    {"tree_retrieval": True},
    {"mode": "retrieval"},
]


@pytest.mark.parametrize("extra", TURN_LOOP_COMPETING_FIELDS)
def test_query_request_rejects_turn_loop_with_competing_orchestrator(extra):
    with pytest.raises(ValidationError):
        QueryRequest(query="q", turn_loop=True, **extra)


@pytest.mark.parametrize("extra", TURN_LOOP_COMPETING_FIELDS)
def test_console_query_request_rejects_turn_loop_with_competing_orchestrator(extra):
    """The console surface enforces the same contract as the API surface."""
    with pytest.raises(ValidationError):
        ConsoleQueryRequest(query="q", turn_loop=True, **extra)


@pytest.mark.parametrize("extra", TURN_LOOP_COMPETING_FIELDS)
def test_query_request_allows_competing_flag_when_turn_loop_not_forced(extra):
    """turn_loop=None (config default) + an explicit competing flag is VALID
    at the schema layer — the server-side resolver, not the validator, owns
    that precedence (the env default yields to the explicit request)."""
    req = QueryRequest(query="q", **extra)
    assert req.turn_loop is None


def test_query_request_rejects_turn_loop_off_conflicts_nowhere():
    """turn_loop=False never conflicts (it forces the classic path)."""
    req = QueryRequest(query="q", turn_loop=False, deep_research=True)
    assert req.turn_loop is False


def test_console_query_request_carries_turn_loop_override():
    """Parity: the console request model exposes the same tri-state override."""
    assert ConsoleQueryRequest(query="q").turn_loop is None
    assert ConsoleQueryRequest(query="q", turn_loop=True).turn_loop is True
