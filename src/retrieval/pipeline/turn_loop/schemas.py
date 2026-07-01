# @summary
# Typed contracts for the turn-level agentic conversation loop: the action
# vocabulary, per-action argument dataclasses, the controller decision (with a
# tolerant LLM-JSON coercion classmethod), the frozen per-turn budget, the
# evidence-pool unit, the answer-gate feedback record, typed stream events, the
# mutable per-turn state accumulator, the clarification/final results, the
# injected-dependency seam (DI for testability), and the cross-turn TurnContext
# digest.
# Exports: TurnAction, RetrieveArgs, DeepStudyArgs, ClarifyArgs, AnswerArgs,
#          TurnActionArgs, TurnDecision, TurnBudget, EvidenceChunk, GateFeedback,
#          TurnEventType, TurnEvent, StudiedDoc, TurnState, ClarificationOut,
#          TurnLoopResult, TurnLoopDeps, TurnContext
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
    DEEP_STUDY: str = "DEEP_STUDY"
    CLARIFY: str = "CLARIFY"
    ANSWER: str = "ANSWER"

    ALL: frozenset[str] = frozenset({RETRIEVE, DEEP_STUDY, CLARIFY, ANSWER})


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


TurnActionArgs = Union[RetrieveArgs, DeepStudyArgs, ClarifyArgs, AnswerArgs]
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

    @classmethod
    def from_settings(cls) -> "TurnBudget":
        """Build a budget from the ``RAG_TURN_LOOP_*`` settings block.

        Imports ``config.settings`` lazily so this module stays config-free at
        import time (pure contract surface; tests construct budgets directly).
        """
        from config import settings

        weights = settings.RAG_TURN_LOOP_ANSWER_GATE_WEIGHTS
        return cls(
            max_actions=settings.RAG_TURN_LOOP_MAX_ACTIONS,
            max_llm_calls=settings.RAG_TURN_LOOP_MAX_LLM_CALLS,
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

    seen_chunk_ids: set[str] = field(default_factory=set)
    """Stable chunk ids ever pooled — cross-action dedup."""

    tried_queries: list[str] = field(default_factory=list)
    """Literal query texts already retrieved with (anti-repeat)."""

    tried_hyde: list[str] = field(default_factory=list)
    """HyDE hypothetical answers already tried (diversification)."""

    studied: dict[str, StudiedDoc] = field(default_factory=dict)
    """Deep-study progress keyed by document_id."""

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
        same context always renders the same digest (evalable/cachable). The
        result is head-truncated to ``max_chars`` (older material is already
        compacted into the rolling summary, so the head carries the signal).

        Args:
            max_chars: Hard character cap for the digest (<= 0 means no cap).

        Returns:
            The prompt-ready digest string.
        """
        lines: list[str] = []
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
        if self.docs_studied:
            lines.append("Documents already deep-studied:")
            for doc in self.docs_studied:
                if not isinstance(doc, dict):
                    continue
                document_id = _coerce_str(doc.get("document_id"))
                conclusion = _coerce_str(doc.get("conclusion"))
                lines.append(f"- document_id={document_id}: {conclusion}")
            lines.append("")
        if self.pending_clarification:
            question = _coerce_str(self.pending_clarification.get("question"))
            hints = self.pending_clarification.get("hints") or []
            lines.append("Pending clarification from the previous turn:")
            lines.append(f"Question: {question}")
            for index, hint in enumerate(hints, start=1):
                lines.append(f"Hint {index}: {_coerce_str(hint)}")
            lines.append("")
        digest = "\n".join(lines).strip()
        if max_chars > 0 and len(digest) > max_chars:
            digest = digest[:max_chars]
        return digest


__all__ = [
    "TurnAction",
    "RetrieveArgs",
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
