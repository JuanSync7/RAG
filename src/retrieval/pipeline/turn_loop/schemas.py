# @summary
# Typed contracts for the turn-level agentic conversation loop: the action
# vocabulary, per-action argument dataclasses, the controller decision (with a
# tolerant LLM-JSON coercion classmethod), the frozen per-turn budget, the
# evidence-pool unit, the answer-gate feedback record, typed stream events, the
# mutable per-turn state accumulator, the clarification/final results, the
# injected-dependency seam (DI for testability), and the cross-turn TurnContext
# digest.
# Exports: TurnAction, RouteEffort, QueryShape, RetrieveArgs, DecomposeArgs,
#          DeepStudyArgs, ClarifyArgs, AnswerArgs,
#          TurnActionArgs, TurnDecision, TurnBudget, EvidenceChunk, GateFeedback,
#          TurnEventType, TurnEvent, StudiedDoc, FacetCoverage, TurnState,
#          ClarificationOut, TurnLoopResult, TurnLoopDeps, TurnContext
# Deps: dataclasses, time, typing (config.settings lazily inside
#       TurnBudget.from_settings only)
# @end-summary
"""Typed schema/contract dataclasses for the turn-level conversation loop.

Single canonical contract module for ``src/retrieval/pipeline/turn_loop``
(CLAUDE.md §2: shared type/state/config contracts live in one module). Like the
agentic loop's ``state.py``, this module is deliberately free of config imports
at module scope — :meth:`TurnBudget.from_settings` imports ``config.settings``
lazily, so importing this module has zero side effects and unit tests can build
budgets by hand. See ``docs/retrieval/TURN_LOOP_DESIGN.md``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, ClassVar, Optional, Union


# ---------------------------------------------------------------------------
# Action vocabulary
# ---------------------------------------------------------------------------


class TurnAction:
    """Str-valued constants for the controller's action space.

    Plain ``str`` constants (not an enum) so decisions/events serialize to
    JSON/SSE without adapters and compare directly against LLM output after
    coercion. ``ALL`` is the closed set used by
    :meth:`TurnDecision.from_llm_json` to reject unknown actions.
    """

    RETRIEVE: str = "RETRIEVE"
    DECOMPOSE: str = "DECOMPOSE"
    DEEP_STUDY: str = "DEEP_STUDY"
    CLARIFY: str = "CLARIFY"
    ANSWER: str = "ANSWER"

    ALL: frozenset[str] = frozenset({RETRIEVE, DECOMPOSE, DEEP_STUDY, CLARIFY, ANSWER})


class RouteEffort:
    """Str-valued effort levels selecting a :class:`TurnBudget` scale.

    The canonical home (CLAUDE.md §2: shared contracts in one module) for the
    effort levels the pre-flight router (:mod:`turn_loop.router`) picks and
    :meth:`TurnBudget.from_settings` consumes — kept here (not in ``router.py``)
    so ``TurnBudget`` can reference the constants without importing the router
    (which would be a cycle: the router imports this module). ``BALANCED`` is
    the neutral level whose budget is byte-for-byte ``from_settings()`` at
    scale 1.0. Plain ``str`` constants so a hint serializes without adapters.
    """

    FAST: str = "fast"
    BALANCED: str = "balanced"
    THOROUGH: str = "thorough"


class QueryShape:
    """Str-valued intrinsic-shape labels the controller assigns to a query.

    The LLM-driven replacement for the ``query_shape.COMPOUND_MARKERS`` keyword
    regex (CLAUDE.md §0 — classify by the model that is already reasoning about
    the query, not by literal ``" and "`` / ``" vs "`` matching): the controller
    emits one of these on every decision, and the loop coerces the OPENING move
    to DECOMPOSE when the shape is ``COMPOUND`` (see ``controller.decide``). A
    property of the question itself, independent of the chosen action. Plain
    ``str`` constants so a decision serializes to JSON without adapters; ``ALL``
    is the closed set :meth:`TurnDecision.from_llm_json` validates against
    (anything else coerces to ``None`` — an absent/unusable shape simply skips
    the coercion, never mis-fires it).
    """

    SINGLE_FACET: str = "single_facet"
    """One thing is asked about — a plain RETRIEVE covers it."""

    COMPOUND: str = "compound"
    """Several distinct facets at once (a comparison, a multi-part "X, Y and Z",
    a broad "summarise the whole flow") — needs the DECOMPOSE parallel fan-out."""

    VAGUE: str = "vague"
    """Too underspecified to retrieve well as written (informational for now —
    no coercion; a future rung could route it to CLARIFY)."""

    ALL: frozenset[str] = frozenset({SINGLE_FACET, COMPOUND, VAGUE})


# ---------------------------------------------------------------------------
# Per-action argument contracts
# ---------------------------------------------------------------------------


@dataclass
class RetrieveArgs:
    """Arguments for a RETRIEVE action: one hybrid-search round.

    ``query_text`` anchors the lexical/BM25 side; ``hypothetical_answer`` (HyDE,
    answer-space) is embedded when present; ``target_aspect`` names the gap this
    retrieval round is trying to cover (steers dedup/diversification).
    """

    query_text: str
    hypothetical_answer: Optional[str] = None
    target_aspect: Optional[str] = None


@dataclass
class DecomposeArgs:
    """Arguments for a DECOMPOSE action: fan a broad/compound question out into a
    few focused sub-queries retrieved in parallel into the same flat pool.

    ``question`` is the compound question to split (defaults to the turn query).
    ``missing_information`` optionally carries the judge's named gap so the split
    targets what is still uncovered rather than re-deriving the whole question.
    """

    question: str
    missing_information: Optional[str] = None


@dataclass
class DeepStudyArgs:
    """Arguments for a DEEP_STUDY action: read one full source document.

    At least one of ``document_id`` / ``source_key`` must identify the target —
    both are copied verbatim from evidence/context refs (the controller prompt
    forbids inventing names). ``question`` is the focused question the windowed
    read tries to answer.
    """

    question: str
    document_id: Optional[str] = None
    source_key: Optional[str] = None


@dataclass
class ClarifyArgs:
    """Arguments for a CLARIFY action (terminal).

    Empty in v1 — the clarification content is authored by a dedicated LLM call
    (``turn_clarify_generate.md``), not by the controller decision. Reserved as
    a typed slot so future fields (e.g. a suggested focus) are non-breaking.
    """


@dataclass
class AnswerArgs:
    """Arguments for an ANSWER action (terminal attempt).

    Empty in v1 — drafting consumes the evidence pool directly. Reserved as a
    typed slot so future fields (e.g. a requested answer style) are non-breaking.
    """


TurnActionArgs = Union[RetrieveArgs, DecomposeArgs, DeepStudyArgs, ClarifyArgs, AnswerArgs]
"""Union of the per-action argument payloads carried by a :class:`TurnDecision`."""


# ---------------------------------------------------------------------------
# Tolerant coercion helpers (module-private; dict/str ops only — no regex)
# ---------------------------------------------------------------------------


def _coerce_str(value: Any, default: str = "") -> str:
    """Best-effort string coercion: ``None`` → ``default``, else stripped str."""
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _coerce_opt_str(value: Any) -> Optional[str]:
    """Best-effort optional-string coercion: empty/``None`` → ``None``."""
    text = _coerce_str(value)
    return text or None


def _coerce_confidence(value: Any, default: float = 0.0) -> float:
    """Best-effort [0,1] float coercion (tolerates numeric strings)."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, num))


def _coerce_query_shape(value: Any) -> Optional[str]:
    """Coerce a raw ``query_shape`` token to the closed :class:`QueryShape` set.

    Case- and separator-insensitive (``"Single-Facet"`` / ``"single facet"`` ->
    ``SINGLE_FACET``), mirroring the action-token normalization. Anything outside
    the vocabulary — including a blank/missing value — returns ``None`` so a
    sloppy or absent shape simply skips the shape coercion rather than mis-firing
    it. Pure dict/str ops, no regex (CLAUDE.md §0)."""
    shape = _coerce_str(value).lower().replace(" ", "_").replace("-", "_")
    return shape if shape in QueryShape.ALL else None


@dataclass
class TurnDecision:
    """One controller verdict: the chosen action plus its typed arguments.

    Produced once per loop iteration from the salvage-parsed JSON of the
    ``turn_controller_decide.md`` call via :meth:`from_llm_json`.
    """

    action: str
    reason: str
    confidence: float
    args: Optional[TurnActionArgs] = None
    query_shape: Optional[str] = None
    """The controller's intrinsic-shape label for the query (:class:`QueryShape`
    value, or ``None`` when absent/unrecognized). Drives the shape-based
    DECOMPOSE coercion in ``controller.decide`` — a top-level property of the
    query, not of ``args``."""

    @classmethod
    def from_llm_json(cls, payload: dict) -> Optional["TurnDecision"]:
        """Coerce a salvage-parsed LLM JSON object into a decision.

        Tolerant by design (the model may flatten ``args`` to the top level,
        emit lowercase/spaced action names, or return confidence as a string);
        strict only on the action vocabulary: anything outside
        :attr:`TurnAction.ALL` returns ``None`` so the orchestrator applies its
        fail-open default. Pure dict/str coercion — no regex (CLAUDE.md §0).

        Args:
            payload: The parsed JSON object from the controller call.

        Returns:
            A :class:`TurnDecision`, or ``None`` when ``payload`` is not a dict
            or names an unknown action.
        """
        if not isinstance(payload, dict):
            return None
        # Normalize the action token: case- and separator-insensitive so
        # "deep study" / "deep-study" / "Deep_Study" all resolve to DEEP_STUDY.
        action = (
            _coerce_str(payload.get("action"))
            .upper()
            .replace(" ", "_")
            .replace("-", "_")
        )
        if action not in TurnAction.ALL:
            return None
        # Models sometimes flatten args to the top level — fall back to the
        # payload itself as the args source when "args" is absent/not a dict.
        raw_args = payload.get("args")
        if not isinstance(raw_args, dict):
            raw_args = payload
        args: Optional[TurnActionArgs]
        if action == TurnAction.RETRIEVE:
            args = RetrieveArgs(
                query_text=_coerce_str(raw_args.get("query_text")),
                hypothetical_answer=_coerce_opt_str(
                    raw_args.get("hypothetical_answer")
                ),
                target_aspect=_coerce_opt_str(raw_args.get("target_aspect")),
            )
        elif action == TurnAction.DECOMPOSE:
            args = DecomposeArgs(
                question=_coerce_str(raw_args.get("question")),
                missing_information=_coerce_opt_str(
                    raw_args.get("missing_information")
                ),
            )
        elif action == TurnAction.DEEP_STUDY:
            args = DeepStudyArgs(
                question=_coerce_str(raw_args.get("question")),
                document_id=_coerce_opt_str(raw_args.get("document_id")),
                source_key=_coerce_opt_str(raw_args.get("source_key")),
            )
        elif action == TurnAction.CLARIFY:
            args = ClarifyArgs()
        else:  # TurnAction.ANSWER
            args = AnswerArgs()
        return cls(
            action=action,
            reason=_coerce_str(payload.get("reason")),
            confidence=_coerce_confidence(payload.get("confidence")),
            args=args,
            # query_shape is a TOP-LEVEL property of the query (never inside args,
            # even when the model flattened everything else there).
            query_shape=_coerce_query_shape(payload.get("query_shape")),
        )


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnBudget:
    """Immutable per-turn caps, sourced from ``RAG_TURN_LOOP_*`` by the caller.

    Frozen so an iteration cannot mutate the budget mid-loop (sibling of
    ``AgenticBudget`` / ``DeepResearchBudget``). All LLM calls made by the loop
    — controller, HyDE, judge, deep-study reads, clarify, draft, self-score —
    are charged against the ONE ``max_llm_calls`` ledger (no double-budgeting
    with the agentic loop's own budget; the loop composes agentic primitives
    directly).
    """

    max_actions: int
    """Hard cap on controller iterations (actions dispatched) per turn."""

    max_llm_calls: int
    """Combined LLM-call budget across the whole turn (single ledger)."""

    wall_clock_ms: int
    """Turn wall-clock budget in ms; must stay below the workflow timeout."""

    max_answer_attempts: int
    """How many gated ANSWER drafts may be attempted before best-effort exit."""

    deep_study_max_docs: int
    """Max distinct documents a single turn may DEEP_STUDY."""

    deep_study_window_chars: int
    """Character size of each deep-study read window."""

    deep_study_window_overlap_chars: int
    """Character overlap between consecutive deep-study windows."""

    deep_study_max_windows: int
    """Max windows read per studied document."""

    clarify_max_hints: int
    """Max clickable hint chips a CLARIFY response may carry."""

    retrieve_top_k: int
    """Ranked-chunk count requested from each retrieve_ranked call."""

    answer_confidence_threshold: float
    """Answer-gate pass threshold in (0, 1]."""

    gate_weight_judge: float
    """Answer-gate weight for the judge pool confidence component."""

    gate_weight_self: float
    """Answer-gate weight for the LLM self-score component."""

    gate_weight_citation: float
    """Answer-gate weight for the citation-coverage component."""

    llm_max_tokens: int = 512
    """Default completion-token cap for loop LLM calls that set no explicit
    cap. An uncapped request makes litellm ask the endpoint for its full
    context as output tokens (vLLM rejects it with a 400)."""

    min_call_budget_ms: int = 8000
    """Minimum remaining wall clock required to start another LLM call or
    loop action; below it the loop goes straight to its best-effort exit
    instead of firing calls with near-zero timeouts."""

    max_no_progress_rounds: int = 2
    """Consecutive evidence-gathering rounds (RETRIEVE / DECOMPOSE / DEEP_STUDY)
    that may add ZERO new chunks before the loop forces an ANSWER attempt from
    the pool gathered so far. Guards the observed pathology where a weak
    controller keeps re-retrieving the same chunks after the judge already
    marked the pool sufficient — wasting actions and latency, and risking a
    max_actions best-effort exit instead of a clean gated answer. 0 disables
    the guard."""

    fallback_pool_size: int = 12
    """Target GENERATION-pool size: how many chunks the ANSWER stage feeds the
    generator. The ANSWER fills the pool toward this size from the reservoir when
    the judged pool is empty OR thin (kept chunks stay first). Matches
    RAG_AGENTIC_FINAL_MAX_CHUNKS so turn_loop feeds generation as much
    cross-document context as the agentic default. 0 disables floor + fill."""

    reservoir_size: int = 40
    """Cross-round retention cap for the raw ``TurnState.fallback_chunks``
    reservoir the ANSWER fill draws from — SEPARATE from and much wider than
    ``fallback_pool_size`` (the generation target). The fill is source-diversity-
    first, so a WIDE reservoir is what lets it reach the OTHER documents a
    multi-part / cross-document question needs. Matches RAG_AGENTIC_JUDGE_POOL_MAX
    so turn_loop's breadth equals the agentic default's. 0 disables retention."""

    baseline_floor_k: int = 4
    """How many raw-query retrieval chunks are carried as a PERMANENT grounding
    floor into the ANSWER pool — present even when the judged pool is already
    FULL (unlike ``fallback_pool_size``'s fill, which only tops up a THIN pool).
    Makes the turn genuinely ADDITIVE: a DECOMPOSE rewrite / query drift / an
    over-strict judge can never DROP a document the plain query found. Source-
    diverse (each floor chunk adds a NEW document not already pooled). 0 disables
    (fill-only, pre-fix behavior)."""

    citation_target: int = 5
    """Distinct-source count at which the answer gate's citation-coverage
    component saturates to 1.0 (denominator = min(pool_size, target)). Keeps a
    grounded refusal from failing the gate just because the pool was filled with
    context chunks it need not all cite."""

    facet_commit_enabled: bool = True
    """Whether the deterministic facet-commit guard is armed. When True, once a
    multi-way DECOMPOSE has covered every facet (each with >=1 judge-kept chunk)
    the loop forces an ANSWER instead of gathering further — the DECOMPOSE-spiral
    fix, complementary to ``max_no_progress_rounds`` (which fires on the opposite
    signal, gathering that stopped producing new evidence). False disables it (the
    controller alone decides when to answer)."""

    decompose_anchor_raw: bool = True
    """Whether a DECOMPOSE round retrieves the RAW turn query as an additive
    anchor leg alongside its sub-queries (RRF/union). When True, the pool is the
    UNION of raw-query and sub-query hits, so a decomposition can only ADD
    candidates — never DROP a doc the raw query matched (the query-transform
    variance fix; mirrors the agentic HyDE asymmetry). False reverts to
    sub-queries-only (substitutive)."""

    shape_decompose_enabled: bool = True
    """Whether the controller's ``query_shape=compound`` label coerces the
    OPENING move to DECOMPOSE (``controller.decide``). The LLM-driven replacement
    for the router's compound-marker regex seed: when True, a query the controller
    itself calls compound but opens with a single RETRIEVE is rewritten to the
    DECOMPOSE parallel fan-out (the c002 fix). Safe only because DECOMPOSE is
    additive (``decompose_anchor_raw``) — the fan-out is a superset of the
    RETRIEVE it replaces — so the coercion self-disables when ``decompose_anchor_raw``
    is off (it would otherwise force a substitutive fan-out). False leaves the
    controller's opening action as-is."""

    @staticmethod
    def _effort_scale(effort: str, settings: Any) -> float:
        """Multiplier the router's ``effort`` applies to the work budgets.

        ``balanced`` (and any unknown level) → 1.0, so the default budget is
        byte-for-byte today's. ``fast`` shrinks and ``thorough`` grows the
        action/LLM-call ceilings via ``RAG_TURN_LOOP_EFFORT_*_SCALE`` (the
        wall clock is deliberately NOT scaled — it is a fixed safety ceiling
        the retrieve timeout is derived from, not a work target)."""
        if effort == RouteEffort.FAST:
            return float(getattr(settings, "RAG_TURN_LOOP_EFFORT_FAST_SCALE", 0.5))
        if effort == RouteEffort.THOROUGH:
            return float(
                getattr(settings, "RAG_TURN_LOOP_EFFORT_THOROUGH_SCALE", 1.5)
            )
        return 1.0

    @classmethod
    def from_settings(cls, effort: str = "balanced") -> "TurnBudget":
        """Build a budget from the ``RAG_TURN_LOOP_*`` settings block.

        Imports ``config.settings`` lazily so this module stays config-free at
        import time (pure contract surface; tests construct budgets directly).

        Args:
            effort: The router-selected effort level (``fast``/``balanced``/
                ``thorough``). Scales ``max_actions`` and ``max_llm_calls``
                only; ``balanced`` (default) leaves every field at its setting.
        """
        from config import settings

        weights = settings.RAG_TURN_LOOP_ANSWER_GATE_WEIGHTS
        scale = cls._effort_scale(effort, settings)
        return cls(
            max_actions=max(1, round(settings.RAG_TURN_LOOP_MAX_ACTIONS * scale)),
            max_llm_calls=max(
                2, round(settings.RAG_TURN_LOOP_MAX_LLM_CALLS * scale)
            ),
            wall_clock_ms=settings.RAG_TURN_LOOP_WALL_CLOCK_MS,
            max_answer_attempts=settings.RAG_TURN_LOOP_MAX_ANSWER_ATTEMPTS,
            deep_study_max_docs=settings.RAG_TURN_LOOP_DEEP_STUDY_MAX_DOCS,
            deep_study_window_chars=settings.RAG_TURN_LOOP_DEEP_STUDY_WINDOW_CHARS,
            deep_study_window_overlap_chars=(
                settings.RAG_TURN_LOOP_DEEP_STUDY_WINDOW_OVERLAP_CHARS
            ),
            deep_study_max_windows=settings.RAG_TURN_LOOP_DEEP_STUDY_MAX_WINDOWS,
            clarify_max_hints=settings.RAG_TURN_LOOP_CLARIFY_MAX_HINTS,
            retrieve_top_k=settings.RAG_TURN_LOOP_RETRIEVE_TOP_K,
            answer_confidence_threshold=(
                settings.RAG_TURN_LOOP_ANSWER_CONFIDENCE_THRESHOLD
            ),
            gate_weight_judge=weights[0],
            gate_weight_self=weights[1],
            gate_weight_citation=weights[2],
            llm_max_tokens=settings.RAG_TURN_LOOP_LLM_MAX_TOKENS,
            min_call_budget_ms=settings.RAG_TURN_LOOP_MIN_CALL_BUDGET_MS,
            max_no_progress_rounds=settings.RAG_TURN_LOOP_MAX_NO_PROGRESS_ROUNDS,
            fallback_pool_size=settings.RAG_TURN_LOOP_FALLBACK_POOL_SIZE,
            reservoir_size=settings.RAG_TURN_LOOP_RESERVOIR_SIZE,
            baseline_floor_k=settings.RAG_TURN_LOOP_BASELINE_FLOOR_K,
            citation_target=settings.RAG_TURN_LOOP_CITATION_TARGET,
            facet_commit_enabled=settings.RAG_TURN_LOOP_FACET_COMMIT_ENABLED,
            decompose_anchor_raw=settings.RAG_TURN_LOOP_DECOMPOSE_ANCHOR_RAW,
            shape_decompose_enabled=settings.RAG_TURN_LOOP_SHAPE_DECOMPOSE_ENABLED,
        )


# ---------------------------------------------------------------------------
# Evidence pool
# ---------------------------------------------------------------------------


@dataclass
class EvidenceChunk:
    """The loop's evidence-pool unit: one grounded, citable text span.

    Both RETRIEVE (ranked hybrid-search chunks) and DEEP_STUDY (extracted
    window findings) contribute pool entries in this one shape so drafting,
    gating and TurnContext persistence never branch on origin — ``provenance``
    records it for telemetry/trace only.
    """

    PROVENANCE_RETRIEVE: ClassVar[str] = "retrieve"
    PROVENANCE_DEEP_STUDY: ClassVar[str] = "deep_study"

    chunk_id: str
    """Stable chunk identity (Weaviate uuid for retrieved chunks) — the pool
    dedup key via ``TurnState.seen_chunk_ids``."""

    document_id: str
    """Owning document id (deep-study anchor + citation grouping)."""

    source_key: str
    """MinIO object key of the source document (deep-study fetch handle)."""

    source: str
    """Human-readable source name (citation display)."""

    heading: str
    """Section heading the span sits under."""

    text: str
    """The evidence text fed to drafting."""

    score: float
    """Ranking score at pool-entry time (raw server scale; loop-internal)."""

    refactored_char_start: int
    """Span start offset in the refactored source markdown (-1 sentinel = no
    anchor; deep-study falls back to heading match or window 0)."""

    refactored_char_end: int
    """Span end offset in the refactored source markdown (-1 = no anchor)."""

    provenance: str
    """How the chunk entered the pool: ``'retrieve'`` or ``'deep_study'``."""

    round_added: int
    """Loop iteration (0-based) on which the chunk entered the pool."""


# ---------------------------------------------------------------------------
# Answer gate feedback
# ---------------------------------------------------------------------------


@dataclass
class GateFeedback:
    """Outcome of one answer-confidence gate evaluation.

    On failure this is fed back to the controller (rendered into
    ``{{ gate_feedback }}``) so the next action targets the weakest component
    rather than blindly retrying.
    """

    score: float
    """Weighted composite score (judge/self/citation per TurnBudget weights)."""

    threshold: float
    """The pass threshold the score was compared against."""

    passed: bool
    """Whether the draft cleared the gate."""

    weakest_component: str
    """Name of the lowest-scoring component (``judge``/``self``/``citation``)."""

    unsupported_claims: list[str] = field(default_factory=list)
    """Draft claims the self-score call flagged as ungrounded."""

    missing_information: list[str] = field(default_factory=list)
    """Judge-named information gaps steering the next RETRIEVE."""


# ---------------------------------------------------------------------------
# Stream events
# ---------------------------------------------------------------------------


class TurnEventType:
    """Str constants for typed loop events — names are 1:1 with the SSE event
    vocabulary in the design (§8) and the OTel span vocabulary, so a trace
    entry, an SSE frame and a span all speak the same names.
    """

    TURN_ACTION: str = "turn_action"
    HYDE_QUERY: str = "hyde_query"
    RETRIEVE_RESULT: str = "retrieve_result"
    JUDGE_VERDICT: str = "judge_verdict"
    DEEP_STUDY: str = "deep_study"
    LLM_CALL: str = "llm_call"
    DRAFT: str = "draft"
    GATE: str = "gate"
    CLARIFY: str = "clarify"

    ALL: frozenset[str] = frozenset(
        {
            TURN_ACTION,
            HYDE_QUERY,
            RETRIEVE_RESULT,
            JUDGE_VERDICT,
            DEEP_STUDY,
            LLM_CALL,
            DRAFT,
            GATE,
            CLARIFY,
        }
    )


@dataclass
class TurnEvent:
    """One typed observability event: appended to the trace and (when
    streaming) emitted as an SSE frame via ``TurnLoopDeps.emit``."""

    type: str
    """Event name — one of the :class:`TurnEventType` constants."""

    payload: dict
    """JSON-serializable event body (shape per design §8 event table)."""

    ts_ms: int
    """Wall-clock emission time, epoch milliseconds."""

    @classmethod
    def now(cls, type: str, payload: dict) -> "TurnEvent":
        """Build an event stamped with the current wall-clock time."""
        return cls(type=type, payload=payload, ts_ms=int(time.time() * 1000))


# ---------------------------------------------------------------------------
# Mutable per-turn state
# ---------------------------------------------------------------------------


@dataclass
class FacetCoverage:
    """One decomposed sub-question and whether the turn has gathered supporting
    evidence for it.

    Populated by a multi-way DECOMPOSE round (``run_decompose``): each sub-query
    the split produced becomes a facet, marked ``covered`` once at least one
    judge-KEPT chunk was retrieved for it. Drives the deterministic
    facet-commit guard (``orchestrator._facet_commit_decision``) — once every
    facet is covered the loop can synthesize the answer and must stop gathering.
    Coverage is monotonic (a facet once covered stays covered).
    """

    question: str
    """The decomposed sub-question text (facet identity, deduped case-insensitively)."""

    covered: bool = False
    """True once >=1 judge-kept chunk has been retrieved for this facet."""


@dataclass
class StudiedDoc:
    """Accumulated deep-study progress for one document within a turn."""

    document_id: str
    """Identity of the studied document."""

    title: str = ""
    """Human-readable title (for events/context rendering)."""

    windows_read: list[int] = field(default_factory=list)
    """Indices of the windows already read (dedup + budget check)."""

    notes: str = ""
    """Accumulated extraction notes across windows."""

    conclusion: str = ""
    """Final takeaway once the study of this document ends."""


@dataclass
class TurnState:
    """Mutable per-turn loop state. Instantiated fresh per turn inside the
    orchestrator and discarded — NEVER a module/class global (same
    cross-request-bleed rule as ``AgenticState``)."""

    iteration: int = 0
    """Controller iterations dispatched so far (charged vs max_actions)."""

    llm_calls: int = 0
    """LLM calls charged so far (single ledger, all purposes)."""

    started_monotonic: float = field(default_factory=time.monotonic)
    """``time.monotonic()`` at turn start — basis for :meth:`elapsed_ms`."""

    pool: list[EvidenceChunk] = field(default_factory=list)
    """Accumulated evidence; ordering = pool order fed to drafting."""

    fallback_chunks: list[EvidenceChunk] = field(default_factory=list)
    """Best-scored RAW retrieved chunks (judge-INDEPENDENT), retained across the
    turn as a grounding floor. Populated every RETRIEVE round from the raw
    candidates BEFORE the judge; consumed only when ``pool`` is empty — an ANSWER
    then grounds on these instead of "(no evidence retrieved)". Guards the class
    where a judge rejecting an entire fresh batch strands the turn with nothing
    to cite (never a query/content match — CLAUDE.md §0)."""

    baseline_floor: list[EvidenceChunk] = field(default_factory=list)
    """The RAW user-query retrieval's top-k (no HyDE, no decomposition), seeded
    ONCE per turn and carried as a PERMANENT grounding floor into the ANSWER pool
    — present even when the judged pool is full. This is the union-not-replace
    guarantee: a HyDE-drifted RETRIEVE round, a DECOMPOSE rewrite, or an
    over-strict judge can never DROP a document the plain query found. Distinct
    from ``fallback_chunks`` (which on a RETRIEVE turn holds only the HyDE-driven
    candidates — never the raw baseline). Empty when ``baseline_floor_k`` is 0."""

    baseline_seeded: bool = False
    """Whether the raw-query ``baseline_floor`` retrieval has been attempted this
    turn (guards the once-per-turn seed against answer-attempt retries; set True
    even on a failed/empty retrieval so it is never retried)."""

    seen_chunk_ids: set[str] = field(default_factory=set)
    """Stable chunk ids ever pooled — cross-action dedup."""

    tried_queries: list[str] = field(default_factory=list)
    """Literal query texts already retrieved with (anti-repeat)."""

    tried_hyde: list[str] = field(default_factory=list)
    """HyDE hypothetical answers already tried (diversification)."""

    studied: dict[str, StudiedDoc] = field(default_factory=dict)
    """Deep-study progress keyed by document_id."""

    facets: list[FacetCoverage] = field(default_factory=list)
    """Decomposed sub-questions accumulated across the turn, each with its
    per-facet coverage (see :class:`FacetCoverage`). Empty until a multi-way
    DECOMPOSE runs; drives the deterministic facet-commit guard."""

    answer_attempts: int = 0
    """Gated ANSWER drafts attempted so far (vs max_answer_attempts)."""

    last_gate: Optional[GateFeedback] = None
    """Feedback from the most recent failed answer gate (None before any)."""

    events: list[TurnEvent] = field(default_factory=list)
    """Full ordered event trace — becomes the result trace."""

    def charge_llm_call(self) -> int:
        """Charge one LLM call to the turn ledger; returns the new total."""
        self.llm_calls += 1
        return self.llm_calls

    def record_facet(self, question: str, covered: bool) -> None:
        """Register a decomposed facet, or upgrade an existing one's coverage.

        Deduped by case-insensitive question text — a later DECOMPOSE naming the
        same facet updates the existing entry rather than duplicating it.
        Coverage is monotonic: a facet once covered stays covered even if a
        subsequent round names it again with no fresh evidence. Blank questions
        are ignored.
        """
        key = question.strip().lower()
        if not key:
            return
        for facet in self.facets:
            if facet.question.strip().lower() == key:
                facet.covered = facet.covered or covered
                return
        self.facets.append(FacetCoverage(question=question.strip(), covered=covered))

    def facets_fully_covered(self) -> bool:
        """True when at least one facet is tracked and every one is covered.

        The commit signal for the facet guard: a decomposed question whose
        every facet now has supporting evidence can be synthesized, so the loop
        should answer rather than keep gathering. False when no DECOMPOSE has
        run yet (no facets) — the guard then stays out of the way.
        """
        return bool(self.facets) and all(facet.covered for facet in self.facets)

    def remaining_actions(self, budget: TurnBudget) -> int:
        """Controller iterations still available under ``budget`` (floor 0)."""
        return max(0, budget.max_actions - self.iteration)

    def elapsed_ms(self) -> int:
        """Milliseconds elapsed since the turn started (monotonic clock)."""
        return int((time.monotonic() - self.started_monotonic) * 1000)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class ClarificationOut:
    """LLM-authored clarification returned to the user (terminal CLARIFY)."""

    question: str
    """The clarification question shown to the user."""

    hints: list[str] = field(default_factory=list)
    """Clickable answer-hint chips (capped by clarify_max_hints)."""

    scoping_questions: list[str] = field(default_factory=list)
    """Alternative narrower questions the user could ask instead."""


@dataclass
class TurnLoopResult:
    """What ``run_turn_loop`` hands back to the route layer for one turn."""

    ACTION_ANSWERED: ClassVar[str] = "answered"
    ACTION_ASK_USER: ClassVar[str] = "ask_user"

    answer: str
    """Final answer text (empty when ``action='ask_user'``)."""

    action: str
    """Terminal outcome: ``'answered'`` or ``'ask_user'``."""

    clarification: Optional[ClarificationOut] = None
    """The clarification payload when ``action='ask_user'``; else None."""

    confidence: float = 0.0
    """Final answer-gate composite score (0.0 when no draft passed)."""

    pool: list[EvidenceChunk] = field(default_factory=list)
    """Final evidence pool — source refs + memory persistence input."""

    trace: list[TurnEvent] = field(default_factory=list)
    """Ordered event trace for metadata.turn_loop.trace."""

    stop_reason: str = ""
    """Why the loop ended (gate_passed / budget name / clarify / error)."""

    llm_calls: int = 0
    """Total LLM calls charged this turn."""

    actions_taken: int = 0
    """Controller iterations dispatched this turn."""

    elapsed_ms: int = 0
    """Total turn wall-clock in milliseconds."""


# ---------------------------------------------------------------------------
# Dependency-injection seam
# ---------------------------------------------------------------------------


RetrieveRankedFn = Callable[[str, Optional[str], int], Awaitable["list[EvidenceChunk]"]]
"""``retrieve_ranked(query_text, hyde_text, top_k) -> list[EvidenceChunk]``."""

FetchDocumentFn = Callable[[Optional[str], Optional[str]], Awaitable[Optional[object]]]
"""``fetch_document(document_id, source_key) -> StoredDocument | None``."""

EmitFn = Callable[[TurnEvent], Awaitable[None]]
"""``emit(event) -> None`` — async sink for typed loop events."""


@dataclass
class TurnLoopDeps:
    """Injected async capabilities the orchestrator runs against.

    The orchestrator itself is pure control flow over these seams (the
    ``src/eval`` smoke DI style): endpoint code injects the real Temporal
    activity / MinIO / LLMProvider / SSE bridge, while tests inject fakes and
    drive the loop with zero infrastructure.
    """

    retrieve_ranked: RetrieveRankedFn
    """Ranked hybrid retrieval (the ``retrieve_ranked`` Temporal activity):
    embeds ``hyde_text`` when given (else ``query_text``), runs hybrid search +
    rerank + thin/nav filter, and returns up to ``top_k`` chunks already mapped
    to :class:`EvidenceChunk`. Idempotent; no LLM calls inside."""

    fetch_document: FetchDocumentFn
    """Full-document fetch for DEEP_STUDY: resolves ``document_id`` and/or
    ``source_key`` to the stored refactored markdown (MinIO-backed,
    conversation-cached by the injector). Returns ``None`` when unresolvable —
    the loop then falls back instead of raising."""

    llm_provider: Any
    """The platform LLMProvider handle (``agenerate``/``generate_stream``).
    Every loop LLM call — controller, HyDE, judge, deep-study read, clarify,
    draft, self-score — goes through this one handle so calls are charged to
    the single ``TurnBudget.max_llm_calls`` ledger."""

    emit: EmitFn
    """Async event sink. The orchestrator awaits this for every
    :class:`TurnEvent`; the endpoint bridges it to SSE frames (or a no-op
    collector when ``RAG_TURN_LOOP_STREAM_EVENTS`` is off / non-streaming)."""


# ---------------------------------------------------------------------------
# Cross-turn context
# ---------------------------------------------------------------------------


@dataclass
class TurnContext:
    """The typed cross-turn memory contract assembled from conversation memory.

    What one turn hands the next (design §7): prior prose Q&A, served-chunk
    references (so deepening into a prior document is a first-class move — no
    hard doc suppression), documents already studied, and any clarification
    still pending an answer.
    """

    conversation_id: str
    """Owning conversation identity."""

    rolling_summary: str = ""
    """Compact LLM rolling summary of older turns (prose)."""

    recent_turns: list[dict] = field(default_factory=list)
    """Recent prose turns (dicts with ``query``/``answer`` keys)."""

    chunk_refs: list[dict] = field(default_factory=list)
    """Served-chunk references (list[dict]: chunk_id, document_id, source_key,
    heading, score, preview) — the only legal DEEP_STUDY target namespace."""

    docs_studied: list[dict] = field(default_factory=list)
    """Documents already deep-studied (list[dict]: document_id, windows_read,
    sections, conclusion, ts)."""

    pending_clarification: Optional[dict] = None
    """The previous turn's unanswered clarification (``question``/``hints``)
    so a reply like "the second one" resolves against the offered hints."""

    def render_for_prompt(self, max_chars: int) -> str:
        """Render the deterministic controller-prompt digest.

        Pure string assembly over the typed fields — no LLM, no regex — so the
        same context always renders the same digest (evalable/cachable).

        Sections are ordered by decision-criticality, most-critical first,
        because the result is head-truncated to ``max_chars``: the pending
        clarification (needed to resolve replies like "the second one"), the
        rolling summary, and the recent turns must survive truncation, so the
        bulky served-evidence catalogue (up to ``RAG_TURN_CONTEXT_MAX_CHUNK_REFS``
        ref lines) renders LAST and is the first thing dropped when the digest
        is over budget.

        Args:
            max_chars: Hard character cap for the digest (<= 0 means no cap).

        Returns:
            The prompt-ready digest string.
        """
        lines: list[str] = []
        # Highest priority: a pending clarification anchors the current reply.
        if self.pending_clarification:
            question = _coerce_str(self.pending_clarification.get("question"))
            hints = self.pending_clarification.get("hints") or []
            lines.append("Pending clarification from the previous turn:")
            lines.append(f"Question: {question}")
            for index, hint in enumerate(hints, start=1):
                lines.append(f"Hint {index}: {_coerce_str(hint)}")
            lines.append("")
        if self.rolling_summary:
            lines.append("Conversation summary:")
            lines.append(self.rolling_summary)
            lines.append("")
        if self.recent_turns:
            lines.append("Recent turns:")
            for turn in self.recent_turns:
                query = _coerce_str(turn.get("query")) if isinstance(turn, dict) else ""
                answer = _coerce_str(turn.get("answer")) if isinstance(turn, dict) else ""
                if query:
                    lines.append(f"Q: {query}")
                if answer:
                    lines.append(f"A: {answer}")
            lines.append("")
        if self.docs_studied:
            lines.append("Documents already deep-studied:")
            for doc in self.docs_studied:
                if not isinstance(doc, dict):
                    continue
                document_id = _coerce_str(doc.get("document_id"))
                conclusion = _coerce_str(doc.get("conclusion"))
                lines.append(f"- document_id={document_id}: {conclusion}")
            lines.append("")
        # Lowest priority (bulky): the served-evidence catalogue renders last
        # so head-truncation sheds it before the anchor sections above.
        if self.chunk_refs:
            lines.append(
                "Evidence already served this conversation "
                "(valid DEEP_STUDY targets):"
            )
            for ref in self.chunk_refs:
                if not isinstance(ref, dict):
                    continue
                document_id = _coerce_str(ref.get("document_id"))
                source_key = _coerce_str(ref.get("source_key"))
                heading = _coerce_str(ref.get("heading"))
                preview = _coerce_str(ref.get("preview"))
                lines.append(
                    f"- document_id={document_id} source_key={source_key} "
                    f"heading={heading}: {preview}"
                )
            lines.append("")
        digest = "\n".join(lines).strip()
        if max_chars > 0 and len(digest) > max_chars:
            digest = digest[:max_chars]
        return digest


__all__ = [
    "TurnAction",
    "RouteEffort",
    "QueryShape",
    "RetrieveArgs",
    "DecomposeArgs",
    "DeepStudyArgs",
    "ClarifyArgs",
    "AnswerArgs",
    "TurnActionArgs",
    "TurnDecision",
    "TurnBudget",
    "EvidenceChunk",
    "GateFeedback",
    "TurnEventType",
    "TurnEvent",
    "StudiedDoc",
    "TurnState",
    "ClarificationOut",
    "TurnLoopResult",
    "TurnLoopDeps",
    "TurnContext",
]
