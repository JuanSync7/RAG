# @summary
# Unit tests for the agentic retrieval loop (INC-1, single round): the generic
# model-driven judge (keep/drop by threshold + fail-open keep-all), HyDE
# generation fail-open, and the orchestrator's one-round flow (HyDE-keyed
# retrieval, rerank-to-original-question, accumulate approved, anti-refusal
# fallback, telemetry). No live models — provider/reranker/retrieve are stubs.
# @end-summary
"""Tests for the agentic HyDE/controller/judge retrieval loop."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Optional

import pytest

from src.retrieval.common.schemas import RankedResult
from src.retrieval.pipeline.agentic import (
    AgenticBudget,
    AgenticRetrieval,
    generate_hyde,
    judge_chunks,
)
from src.vector_db.common.schemas import SearchResult


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _LLMResp:
    content: str
    model: str = "fake"
    prompt_tokens: int = 0
    completion_tokens: int = 0


class RoutingProvider:
    """Stub provider that returns a HyDE payload or a judge payload depending on
    which prompt it sees (the orchestrator calls HyDE first, then judge)."""

    def __init__(self, hyde: Any = "__default__", judge: Any = "__default__") -> None:
        self._hyde = hyde
        self._judge = judge
        self.hyde_prompts: list[str] = []
        self.judge_prompts: list[str] = []

    async def agenerate(self, messages, **kwargs):
        content = str(messages[-1].get("content", ""))
        if "Chunk Judge" in content:
            self.judge_prompts.append(content)
            payload = self._judge
        else:
            self.hyde_prompts.append(content)
            payload = self._hyde
        if payload == "__default__":
            payload = {}
        if isinstance(payload, str):
            return _LLMResp(content=payload)
        return _LLMResp(content=json.dumps(payload))


class FakeReranker:
    """Records the query it was asked to rerank and returns the documents as
    RankedResults with descending scores in input order, truncated to top_k."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def rerank(self, query: str, documents: list, top_k: int) -> list[RankedResult]:
        self.queries.append(query)
        out: list[RankedResult] = []
        n = len(documents)
        for i, d in enumerate(documents[:top_k]):
            out.append(RankedResult(text=d.text, score=float(n - i), metadata=dict(d.metadata)))
        return out


def _sr(idx: int, text: str, source: str = "doc.md") -> SearchResult:
    return SearchResult(
        text=text,
        score=0.5,
        metadata={"source": source, "heading": "h", "chunk_index": idx, "document_id": source},
        object_id=f"oid-{idx}",
        collection="RagDocuments",
    )


def _sr_role(idx: int, text: str, role: Optional[str], source: str = "doc.md") -> SearchResult:
    """A SearchResult carrying a ``chunk_role`` in metadata (None = untagged/legacy)."""
    md = {"source": source, "heading": "h", "chunk_index": idx, "document_id": source}
    if role is not None:
        md["chunk_role"] = role
    return SearchResult(
        text=text, score=0.5, metadata=md,
        object_id=f"oid-{idx}", collection="RagDocuments",
    )


def _budget(**over: Any) -> AgenticBudget:
    base = dict(
        max_rounds=1, max_llm_calls=10, wall_clock_ms=45000,
        keep_top_k_per_round=8, final_max_chunks=8, min_kept_chunks=3, min_sources=1,
        relevance_threshold=0.5, faithfulness_threshold=0.5, sufficiency_target=0.7,
        hyde_diversity_max_cosine=0.92,
    )
    base.update(over)
    return AgenticBudget(**base)


def _orch(provider, retrieve, *, budget: AgenticBudget, reranker=None,
          thin_filter=None, final_top_k: int = 5,
          judge_concise: bool = False, fill_mode: str = "hybrid",
          role_backstop: bool = False,
          excluded_roles=("navigation", "boilerplate")) -> AgenticRetrieval:
    return AgenticRetrieval(
        provider=provider,
        retrieve=retrieve,
        reranker=reranker or FakeReranker(),
        thin_filter=thin_filter or (lambda items: items),
        doc_diversity=lambda ranked, k: ranked[:k],
        original_question="What are the major coherency features of protocol Z?",
        processed_query="protocol Z coherency features",
        budget=budget,
        controller_alias="controller",
        judge_alias="judge",
        hyde_max_tokens=256,
        hyde_temperature=0.4,
        llm_timeout_s=30,
        final_top_k=final_top_k,
        judge_concise=judge_concise,
        fill_mode=fill_mode,
        role_backstop=role_backstop,
        excluded_roles=list(excluded_roles),
    )


# ---------------------------------------------------------------------------
# Judge — generic, threshold-driven, fail-open
# ---------------------------------------------------------------------------


def test_judge_keeps_relevant_faithful_drops_low():
    """Model-driven keep/drop on a GENERIC information-content property — a
    different instance of the 'low-value chunk' class than the ToC/nav regex:
    here a low-relevance chunk from an arbitrary domain is dropped purely by the
    judge's scores, with no pattern/vendor match (CLAUDE.md §0)."""
    cands = [RankedResult(text=f"c{i}", score=1.0, metadata={}) for i in range(3)]
    provider = RoutingProvider(judge={
        "chunks": [
            {"i": 0, "relevance": 0.9, "faithfulness": 0.9, "keep": True, "reason": "on-topic"},
            {"i": 1, "relevance": 0.2, "faithfulness": 0.8, "keep": False, "reason": "off-topic"},
            {"i": 2, "relevance": 0.8, "faithfulness": 0.1, "keep": False, "reason": "fragment"},
        ],
        "pool": {"sufficient": False, "confidence": 0.4, "missing_information": "aspect B", "covered_aspects": ["A"]},
    })
    verdicts, pool = asyncio.run(judge_chunks(
        provider, model_alias="judge",
        original_question="q", candidates=cands, timeout_s=30,
    ))
    assert [v.keep for v in verdicts] == [True, False, False]
    assert pool is not None and pool.sufficient is False
    assert pool.missing_information == "aspect B"


def test_judge_fail_open_keep_all_on_empty_json():
    """qwopus emits '{}' under guided JSON — the judge must KEEP ALL, never drop
    the whole round to empty (the load-bearing judge-model decision)."""
    cands = [RankedResult(text=f"c{i}", score=1.0, metadata={}) for i in range(4)]
    for bad in ("{}", "", "not json at all", '{"unexpected": 1}'):
        provider = RoutingProvider(judge=bad)
        verdicts, pool = asyncio.run(judge_chunks(
            provider, model_alias="judge",
            original_question="q", candidates=cands, timeout_s=30,
        ))
        assert len(verdicts) == 4
        assert all(v.keep for v in verdicts), bad
        assert pool is None


def test_judge_strips_reasoning_and_salvages_free_form():
    """A reasoning model emits a <think> block (with stray braces) then the JSON.
    With json_mode off, the judge must strip the reasoning and salvage the verdicts
    + ranking — the path that lets the deployed qwopus model judge without the
    guided-JSON constraint that makes it emit '{}'."""
    cands = [RankedResult(text=f"c{i}", score=1.0, metadata={}) for i in range(2)]
    raw = (
        "<think>chunk {0} looks relevant, chunk {1} less so; set is ok</think>\n"
        "```json\n"
        '{"chunks": [{"i": 0, "relevance": 0.9, "faithfulness": 0.9, "keep": true},'
        '{"i": 1, "relevance": 0.2, "faithfulness": 0.8, "keep": false}],'
        '"ranking": [0], "pool": {"sufficient": true, "confidence": 0.8}}\n```'
    )
    provider = RoutingProvider(judge=raw)
    verdicts, pool = asyncio.run(judge_chunks(
        provider, model_alias="judge", original_question="q",
        candidates=cands, timeout_s=30, json_mode=False,
    ))
    assert [v.keep for v in verdicts] == [True, False]
    assert verdicts[0].rank == 0          # listwise ranking survived the salvage
    assert pool is not None and pool.sufficient is True


def test_judge_json_mode_toggle_controls_response_format():
    """json_mode=False must NOT send response_format (the constraint that breaks a
    reasoning model); json_mode=True must send it."""
    seen: dict = {}

    class _CaptureProvider:
        async def agenerate(self, messages, **kwargs):
            seen["response_format"] = kwargs.get("response_format", "ABSENT")
            return _LLMResp(content='{"chunks": [], "ranking": [], '
                                    '"pool": {"sufficient": false, "confidence": 0.0}}')

    cands = [RankedResult(text="c0", score=1.0, metadata={})]
    asyncio.run(judge_chunks(_CaptureProvider(), model_alias="judge",
                             original_question="q", candidates=cands,
                             timeout_s=30, json_mode=False))
    assert seen["response_format"] == "ABSENT"
    asyncio.run(judge_chunks(_CaptureProvider(), model_alias="judge",
                             original_question="q", candidates=cands,
                             timeout_s=30, json_mode=True))
    assert seen["response_format"] == {"type": "json_object"}


def test_judge_concise_mode_ranking_is_keep_and_order():
    """Concise judge: the model reasons then emits ONLY a ranked id-list (+
    sufficiency). Listed indices are KEPT in that order; unlisted indices are
    dropped — the tiny-output design that cuts latency. No per-chunk scores."""
    cands = [RankedResult(text=f"c{i}", score=1.0, metadata={}) for i in range(3)]
    provider = RoutingProvider(judge={"ranking": [2, 0], "sufficient": True, "confidence": 0.8})
    verdicts, pool = asyncio.run(judge_chunks(
        provider, model_alias="judge", original_question="q",
        candidates=cands, timeout_s=30, concise=True,
    ))
    assert [v.keep for v in verdicts] == [True, False, True]   # 2 and 0 ranked, 1 dropped
    assert verdicts[2].rank == 0 and verdicts[0].rank == 1     # order from the list
    assert verdicts[1].rank == -1
    assert pool is not None and pool.sufficient is True and pool.confidence == 0.8


def test_judge_concise_empty_ranking_drops_all():
    """A concise ranking of [] is a valid 'nothing relevant' verdict (drop all),
    distinct from fail-open."""
    cands = [RankedResult(text=f"c{i}", score=1.0, metadata={}) for i in range(3)]
    provider = RoutingProvider(judge={"ranking": [], "sufficient": False, "confidence": 0.1})
    verdicts, pool = asyncio.run(judge_chunks(
        provider, model_alias="judge", original_question="q",
        candidates=cands, timeout_s=30, concise=True,
    ))
    assert all(not v.keep for v in verdicts)
    assert pool is not None and pool.sufficient is False


def test_judge_concise_fail_open_keep_all():
    """Concise mode with no parseable ranking → fail-open keep-all (rank -1), so a
    flaky judge never drops the round and ordering falls back to hybrid."""
    cands = [RankedResult(text=f"c{i}", score=1.0, metadata={}) for i in range(3)]
    for bad in ("{}", "", "garbage", {"no_ranking": 1}):
        provider = RoutingProvider(judge=bad)
        verdicts, pool = asyncio.run(judge_chunks(
            provider, model_alias="judge", original_question="q",
            candidates=cands, timeout_s=30, concise=True,
        ))
        assert all(v.keep for v in verdicts), bad
        assert all(v.rank == -1 for v in verdicts), bad
        assert pool is None


def test_judge_omitted_index_is_kept_not_dropped():
    """A candidate the model forgot to score is kept (fail-open per chunk),
    never dropped by omission."""
    cands = [RankedResult(text=f"c{i}", score=1.0, metadata={}) for i in range(3)]
    provider = RoutingProvider(judge={
        "chunks": [{"i": 0, "relevance": 0.9, "faithfulness": 0.9, "keep": True, "reason": "x"}],
        "pool": {"sufficient": True, "confidence": 0.9},
    })
    verdicts, _ = asyncio.run(judge_chunks(
        provider, model_alias="judge", original_question="q",
        candidates=cands, timeout_s=30,
    ))
    assert [v.keep for v in verdicts] == [True, True, True]


# ---------------------------------------------------------------------------
# HyDE — fail-open
# ---------------------------------------------------------------------------


def test_generate_hyde_parses_variant():
    provider = RoutingProvider(hyde={
        "hypothetical_answer": "Protocol Z adds cache coherency and snoop transactions.",
        "search_terms": ["coherency", "snoop"],
        "target_aspect": "coherency",
    })
    v = asyncio.run(generate_hyde(
        provider, model_alias="controller", original_question="q",
        max_tokens=256, temperature=0.4, timeout_s=30,
    ))
    assert v is not None
    assert "coherency" in v.hypothetical_answer.lower()
    assert v.search_terms == ["coherency", "snoop"]


def test_generate_hyde_injects_domain_into_prompt():
    """The corpus domain is rendered into the HyDE prompt so the controller
    resolves domain-ambiguous acronyms in-corpus (fixes wrong-domain confabulation,
    e.g. 'DFT' → Design-for-Test not Density Functional Theory)."""
    provider = RoutingProvider(hyde={
        "hypothetical_answer": "x", "search_terms": ["a"], "target_aspect": "t",
    })
    asyncio.run(generate_hyde(
        provider, model_alias="controller", original_question="Describe the DFT flow.",
        max_tokens=256, temperature=0.4, timeout_s=30,
        domain="Silicon/SoC design: AMBA protocols, DFT (Design-for-Test), MBIST.",
    ))
    assert provider.hyde_prompts, "expected a HyDE prompt to be captured"
    assert "Design-for-Test" in provider.hyde_prompts[0]  # domain reached the prompt


def test_generate_hyde_fail_open_returns_none():
    for bad in ("{}", "", "garbage", {"hypothetical_answer": ""}):
        provider = RoutingProvider(hyde=bad)
        v = asyncio.run(generate_hyde(
            provider, model_alias="controller", original_question="q",
            max_tokens=256, temperature=0.4, timeout_s=30,
        ))
        assert v is None, bad


# ---------------------------------------------------------------------------
# Orchestrator — single round
# ---------------------------------------------------------------------------


def test_single_round_hyde_keyed_retrieval_and_anchor_rerank():
    """ranker="cross_encoder" (INC-1 path): the retrieve tool is called with the
    HyDE ANSWER (not the processed query), and the cross-encoder is anchored to
    the ORIGINAL question."""
    received = {}

    async def retrieve(hyde_answer, search_terms):
        received["hyde"] = hyde_answer
        received["terms"] = list(search_terms)
        return [_sr(i, f"body chunk {i}") for i in range(4)]

    reranker = FakeReranker()
    provider = RoutingProvider(
        hyde={"hypothetical_answer": "ZHYDE answer text", "search_terms": ["zc"]},
        judge={"chunks": [{"i": i, "relevance": 0.9, "faithfulness": 0.9, "keep": True}
                          for i in range(4)],
               "pool": {"sufficient": True, "confidence": 0.9}},
    )
    orch = _orch(provider, retrieve,
                 budget=_budget(min_kept_chunks=1, ranker="cross_encoder"), reranker=reranker)
    result = asyncio.run(orch.run())

    assert received["hyde"] == "ZHYDE answer text"   # HyDE-keyed, not processed_query
    assert reranker.queries == ["What are the major coherency features of protocol Z?"]
    assert result.rounds_run == 1
    assert result.llm_calls == 2 and result.judge_calls == 1
    assert result.kept_count == 4
    assert len(result.reranked) == 4


def test_judge_mode_bypasses_cross_encoder():
    """Default ranker="judge": the cross-encoder is NOT called — the LLM judge
    ranks the RAW hybrid pool directly (the measured-best path)."""
    async def retrieve(hyde_answer, search_terms):
        return [_sr(i, f"body chunk {i}") for i in range(4)]

    reranker = FakeReranker()
    provider = RoutingProvider(
        hyde={"hypothetical_answer": "h", "search_terms": []},
        judge={"chunks": [{"i": i, "relevance": 0.9, "faithfulness": 0.9, "keep": True}
                          for i in range(4)],
               "pool": {"sufficient": True, "confidence": 0.9}},
    )
    orch = _orch(provider, retrieve, budget=_budget(min_kept_chunks=1), reranker=reranker)
    result = asyncio.run(orch.run())

    assert reranker.queries == []          # cross-encoder bypassed
    assert result.ranker == "judge"
    assert result.kept_count == 4
    assert {r.text for r in result.reranked} == {f"body chunk {i}" for i in range(4)}


def test_judge_listwise_ranking_orders_output():
    """The judge's listwise `ranking` (best->worst) — NOT the pointwise relevance
    — sets the order fed to generation. Pointwise relevance is deliberately FLAT
    (all 0.9) so any ordering must come from `ranking`."""
    async def retrieve(hyde_answer, search_terms):
        return [_sr(i, f"chunk {i}") for i in range(4)]

    provider = RoutingProvider(
        hyde={"hypothetical_answer": "h", "search_terms": []},
        judge={"chunks": [{"i": i, "relevance": 0.9, "faithfulness": 0.9, "keep": True}
                          for i in range(4)],
               "ranking": [2, 0, 3, 1],          # best -> worst, not the input order
               "pool": {"sufficient": True, "confidence": 0.9}},
    )
    orch = _orch(provider, retrieve, budget=_budget(min_kept_chunks=1), final_top_k=4)
    result = asyncio.run(orch.run())
    assert [r.text for r in result.reranked] == ["chunk 2", "chunk 0", "chunk 3", "chunk 1"]


def test_finalize_scores_obey_unit_relevance_contract():
    """Regression (class: a result `.score` rendered as a relevance % must stay in
    [0,1]). `_finalize` stamps a strictly-decreasing ORDER key onto `.score`; that
    key must be normalized into (0,1] — an un-normalized ordinal (total-pos) made
    the console render rank-7 as '700%' and pegged the retrieval-quality gate to
    'strong'. The cross-encoder emits sigmoid(logit) in (0,1), so the judge path
    must honor the same contract. Six kept chunks (> the typical bad %) so the
    bug, if reintroduced, shows as scores of 6.0, 5.0, ... = 600%, 500%."""
    async def retrieve(hyde_answer, search_terms):
        return [_sr(i, f"chunk {i}") for i in range(6)]

    provider = RoutingProvider(
        hyde={"hypothetical_answer": "h", "search_terms": []},
        judge={"chunks": [{"i": i, "relevance": 0.9, "faithfulness": 0.9, "keep": True}
                          for i in range(6)],
               "ranking": [0, 1, 2, 3, 4, 5],
               "pool": {"sufficient": True, "confidence": 0.9}},
    )
    orch = _orch(provider, retrieve, budget=_budget(min_kept_chunks=1), final_top_k=6)
    result = asyncio.run(orch.run())

    scores = [r.score for r in result.reranked]
    assert scores, "expected a non-empty curated set"
    # Bounded into the [0,1] relevance contract (so score*100 <= 100%).
    assert all(0.0 < s <= 1.0 for s in scores), scores
    assert scores[0] == pytest.approx(1.0)        # top result = full relevance
    # Strictly decreasing: the order key the diversity sort relies on is intact.
    assert all(a > b for a, b in zip(scores, scores[1:])), scores


def test_judge_fail_open_preserves_hybrid_order_not_flat():
    """Inv-3 pin: when the judge fails open (no parseable scores/ranking), the
    order MUST be the raw-hybrid input order — never a flat-1.0 collapse. Hybrid
    score descends with index here, so the output must equal the input order."""
    async def retrieve(hyde_answer, search_terms):
        # Descending hybrid scores so a correct hybrid-order fallback is testable.
        return [SearchResult(text=f"chunk {i}", score=1.0 - i * 0.1,
                             metadata={"source": "d.md", "heading": "h",
                                       "chunk_index": i, "document_id": "d.md"},
                             object_id=f"oid-{i}", collection="c") for i in range(4)]

    provider = RoutingProvider(
        hyde={"hypothetical_answer": "h", "search_terms": []},
        judge="{}",   # fail-open keep-all, no ranking
    )
    orch = _orch(provider, retrieve, budget=_budget(min_kept_chunks=1), final_top_k=4)
    result = asyncio.run(orch.run())
    assert [r.text for r in result.reranked] == ["chunk 0", "chunk 1", "chunk 2", "chunk 3"]


def test_hyde_failure_is_counted_in_telemetry():
    """Regression (silent-degradation class): when the controller HyDE returns
    nothing parseable, the loop falls back to the LITERAL query — and that MUST be
    observable. ``hyde_failures`` counts it so a mis-provisioned controller (e.g. a
    reasoning model emitting empty content under the HyDE token budget) cannot hide
    as 'working'. This is the exact failure that went undiagnosed in deployment."""
    async def retrieve(hyde_answer, search_terms):
        return [_sr(i, f"chunk {i}") for i in range(3)]

    provider = RoutingProvider(
        hyde={},  # empty -> no hypothetical_answer -> generate_hyde returns None
        judge={"chunks": [{"i": i, "relevance": 0.9, "faithfulness": 0.9, "keep": True}
                          for i in range(3)],
               "ranking": [0, 1, 2], "pool": {"sufficient": True, "confidence": 0.9}},
    )
    orch = _orch(provider, retrieve, budget=_budget(min_kept_chunks=1), final_top_k=3)
    result = asyncio.run(orch.run())

    assert result.hyde_failures == 1
    assert result.telemetry()["hyde_failures"] == 1
    # The fallback still retrieved + kept (degraded, never worse than baseline).
    assert result.kept_count == 3


def test_tried_hyde_texts_are_exported_in_telemetry():
    """The actual per-round HyDE hypothetical texts are observable in telemetry —
    a count alone can't distinguish an on-topic-but-weak HyDE from a strong one, so
    validating/judging HyDE quality requires the generated text itself."""
    async def retrieve(hyde_answer, search_terms):
        return [_sr(i, f"body chunk {i}") for i in range(3)]

    provider = RoutingProvider(
        hyde={"hypothetical_answer": "ZHYDE answer text", "search_terms": ["zc"]},
        judge={"chunks": [{"i": i, "relevance": 0.9, "faithfulness": 0.9, "keep": True}
                          for i in range(3)],
               "pool": {"sufficient": True, "confidence": 0.9}},
    )
    orch = _orch(provider, retrieve, budget=_budget(min_kept_chunks=1), final_top_k=3)
    result = asyncio.run(orch.run())

    tried = result.telemetry()["tried_hyde"]
    assert isinstance(tried, list)
    assert "ZHYDE answer text" in tried


def test_hyde_rounds_export_full_variant_in_telemetry():
    """The UI query-processing panel needs the FULL per-round HyDE variant — the
    hypothetical answer PLUS the lexical anchor terms and target aspect — not just
    the answer text. telemetry()['hyde_rounds'] carries that structured record."""
    async def retrieve(hyde_answer, search_terms):
        return [_sr(i, f"body chunk {i}") for i in range(3)]

    provider = RoutingProvider(
        hyde={"hypothetical_answer": "ZHYDE answer text",
              "search_terms": ["zc", "zd"], "target_aspect": "the Z aspect"},
        judge={"chunks": [{"i": i, "relevance": 0.9, "faithfulness": 0.9, "keep": True}
                          for i in range(3)],
               "pool": {"sufficient": True, "confidence": 0.9}},
    )
    orch = _orch(provider, retrieve, budget=_budget(min_kept_chunks=1), final_top_k=3)
    result = asyncio.run(orch.run())

    rounds = result.telemetry()["hyde_rounds"]
    assert isinstance(rounds, list) and len(rounds) == 1
    r = rounds[0]
    assert r["round"] == 1
    assert r["hypothetical_answer"] == "ZHYDE answer text"
    assert r["search_terms"] == ["zc", "zd"]
    assert r["target_aspect"] == "the Z aspect"
    assert r["fell_back"] is False


def test_hyde_rounds_mark_literal_fallback():
    """When HyDE generation fails and the loop embeds the literal query, that
    round's hyde_rounds record is flagged fell_back=True (the alarm the panel
    surfaces) — parity with the hyde_failures counter."""
    async def retrieve(hyde_answer, search_terms):
        return [_sr(i, f"body chunk {i}") for i in range(3)]

    provider = RoutingProvider(
        hyde=None,  # controller returns nothing → literal-query fallback
        judge={"chunks": [{"i": i, "relevance": 0.9, "faithfulness": 0.9, "keep": True}
                          for i in range(3)],
               "pool": {"sufficient": True, "confidence": 0.9}},
    )
    orch = _orch(provider, retrieve, budget=_budget(min_kept_chunks=1), final_top_k=3)
    result = asyncio.run(orch.run())

    tel = result.telemetry()
    assert tel["hyde_failures"] == 1
    assert tel["hyde_rounds"][0]["fell_back"] is True


def test_single_round_drops_unfaithful_chunk():
    async def retrieve(hyde_answer, search_terms):
        return [_sr(i, f"body chunk {i}") for i in range(3)]

    provider = RoutingProvider(
        hyde={"hypothetical_answer": "h", "search_terms": []},
        judge={"chunks": [
            {"i": 0, "relevance": 0.9, "faithfulness": 0.9, "keep": True},
            {"i": 1, "relevance": 0.1, "faithfulness": 0.9, "keep": False},
            {"i": 2, "relevance": 0.9, "faithfulness": 0.9, "keep": True},
        ], "pool": {"sufficient": True, "confidence": 0.8}},
    )
    orch = _orch(provider, retrieve, budget=_budget(min_kept_chunks=1), fill_mode="none")
    result = asyncio.run(orch.run())
    assert result.kept_count == 2
    texts = {r.text for r in result.reranked}
    assert texts == {"body chunk 0", "body chunk 2"}


def test_anti_refusal_fallback_when_judge_keeps_too_few():
    """If the judge keeps fewer than min_kept_chunks, backfill from the round's
    best reranked candidates — never return empty context on a non-empty
    retrieval (guards the 'I cannot answer' regression)."""
    async def retrieve(hyde_answer, search_terms):
        return [_sr(i, f"body chunk {i}") for i in range(5)]

    provider = RoutingProvider(
        hyde={"hypothetical_answer": "h", "search_terms": []},
        judge={"chunks": [{"i": i, "relevance": 0.0, "faithfulness": 0.0, "keep": False}
                          for i in range(5)],
               "pool": {"sufficient": False, "confidence": 0.0, "missing_information": "everything"}},
    )
    orch = _orch(provider, retrieve, budget=_budget(min_kept_chunks=3), final_top_k=5)
    result = asyncio.run(orch.run())
    assert result.kept_count == 0          # judge approved nothing
    assert len(result.reranked) >= 3       # but we still return context
    assert result.stop_reason == "single_round"


def test_orchestrator_concise_judge_orders_by_ranking():
    """End-to-end through the orchestrator with the CONCISE judge: only the ranked
    indices reach generation, in the judge's order (gate + rank in one tiny output)."""
    async def retrieve(hyde_answer, search_terms):
        return [_sr(i, f"chunk {i}") for i in range(4)]

    provider = RoutingProvider(
        hyde={"hypothetical_answer": "h"},
        judge={"ranking": [3, 1], "sufficient": True, "confidence": 0.9},
    )
    orch = _orch(provider, retrieve, budget=_budget(min_kept_chunks=1),
                 judge_concise=True, final_top_k=4, fill_mode="none")
    result = asyncio.run(orch.run())
    assert [r.text for r in result.reranked] == ["chunk 3", "chunk 1"]
    assert result.kept_count == 2


def test_promote_fill_judge_picks_on_top_then_hybrid_fill():
    """fill_mode='hybrid' (default): the judge's picks are promoted to the top, and
    the remaining top-K slots are filled from the hybrid pool (the chunks the
    cautious judge didn't pick) — recovering recall without losing the judge's
    precision. Hybrid scores descend with index, so the fill order is predictable."""
    async def retrieve(hyde_answer, search_terms):
        return [SearchResult(text=f"chunk {i}", score=1.0 - i * 0.1,
                             metadata={"source": "d.md", "heading": "h",
                                       "chunk_index": i, "document_id": "d.md"},
                             object_id=f"oid-{i}", collection="c") for i in range(6)]

    provider = RoutingProvider(
        hyde={"hypothetical_answer": "h"},
        judge={"ranking": [4, 1], "sufficient": True, "confidence": 0.9},  # keep only 4,1
    )
    orch = _orch(provider, retrieve, budget=_budget(min_kept_chunks=1),
                 judge_concise=True, final_top_k=5, fill_mode="hybrid")
    result = asyncio.run(orch.run())
    texts = [r.text for r in result.reranked]
    assert texts[:2] == ["chunk 4", "chunk 1"]        # judge picks promoted on top
    assert len(texts) == 5                             # filled to final_top_k
    # the fill is the highest-hybrid-score chunks the judge didn't pick (0,2,3)
    assert set(texts[2:]) == {"chunk 0", "chunk 2", "chunk 3"}
    assert result.kept_count == 2 and result.backfilled == 3


def test_fill_mode_none_trusts_judge_no_hybrid_fill():
    """fill_mode='none': only the judge's picks reach generation (+ the anti-refusal
    floor) — the strong-judge / Design-A path."""
    async def retrieve(hyde_answer, search_terms):
        return [_sr(i, f"chunk {i}") for i in range(6)]

    provider = RoutingProvider(
        hyde={"hypothetical_answer": "h"},
        judge={"ranking": [4, 1], "sufficient": True, "confidence": 0.9},
    )
    orch = _orch(provider, retrieve, budget=_budget(min_kept_chunks=1),
                 judge_concise=True, final_top_k=5, fill_mode="none")
    result = asyncio.run(orch.run())
    assert [r.text for r in result.reranked] == ["chunk 4", "chunk 1"]   # no fill
    assert result.backfilled == 0


def test_fill_mode_adaptive_gates_on_judge_confidence():
    """fill_mode='adaptive': trust the judge (no fill) when it reports sufficient +
    confident; fill from hybrid when it does not."""
    async def retrieve(hyde_answer, search_terms):
        return [SearchResult(text=f"chunk {i}", score=1.0 - i * 0.1,
                             metadata={"source": "d.md", "heading": "h",
                                       "chunk_index": i, "document_id": "d.md"},
                             object_id=f"oid-{i}", collection="c") for i in range(6)]

    # confident -> behaves like 'none'
    prov_conf = RoutingProvider(hyde={"hypothetical_answer": "h"},
        judge={"ranking": [4, 1], "sufficient": True, "confidence": 0.9})
    r1 = asyncio.run(_orch(prov_conf, retrieve, budget=_budget(min_kept_chunks=1),
                           judge_concise=True, final_top_k=5, fill_mode="adaptive").run())
    assert [r.text for r in r1.reranked] == ["chunk 4", "chunk 1"]
    assert r1.backfilled == 0

    # NOT confident -> fills from hybrid like 'hybrid'
    prov_unsure = RoutingProvider(hyde={"hypothetical_answer": "h"},
        judge={"ranking": [4, 1], "sufficient": False, "confidence": 0.2})
    r2 = asyncio.run(_orch(prov_unsure, retrieve, budget=_budget(min_kept_chunks=1),
                           judge_concise=True, final_top_k=5, fill_mode="adaptive").run())
    assert [r.text for r in r2.reranked][:2] == ["chunk 4", "chunk 1"]
    assert len(r2.reranked) == 5 and r2.backfilled == 3


def test_empty_retrieval_returns_empty_no_crash():
    async def retrieve(hyde_answer, search_terms):
        return []

    provider = RoutingProvider(hyde={"hypothetical_answer": "h"})
    orch = _orch(provider, retrieve, budget=_budget())
    result = asyncio.run(orch.run())
    assert result.reranked == []
    assert result.rounds_run == 1


def test_hyde_failure_falls_back_to_processed_query_search():
    """When HyDE generation fails, retrieval still runs on the processed query
    (degrades to one standard hybrid search — never worse than baseline)."""
    received = {}

    async def retrieve(hyde_answer, search_terms):
        received["hyde"] = hyde_answer
        return [_sr(0, "body chunk 0")]

    provider = RoutingProvider(
        hyde="{}",  # unparseable -> generate_hyde returns None
        judge={"chunks": [{"i": 0, "relevance": 0.9, "faithfulness": 0.9, "keep": True}],
               "pool": {"sufficient": True, "confidence": 0.9}},
    )
    orch = _orch(provider, retrieve, budget=_budget(min_kept_chunks=1))
    result = asyncio.run(orch.run())
    assert received["hyde"] == "protocol Z coherency features"  # processed_query fallback
    assert result.kept_count == 1


# ---------------------------------------------------------------------------
# Orchestrator — multi-round (INC-2)
# ---------------------------------------------------------------------------


class SequencedProvider:
    """Returns queued HyDE / judge payloads in order across rounds (clamping to
    the last entry once a queue is exhausted, e.g. the finalize re-rank call)."""

    def __init__(self, hyde_seq, judge_seq):
        self._hyde = list(hyde_seq)
        self._judge = list(judge_seq)
        self.hyde_calls = 0
        self.judge_calls = 0

    async def agenerate(self, messages, **kwargs):
        content = str(messages[-1].get("content", ""))
        if "Chunk Judge" in content:
            payload = self._judge[min(self.judge_calls, len(self._judge) - 1)]
            self.judge_calls += 1
        else:
            payload = self._hyde[min(self.hyde_calls, len(self._hyde) - 1)]
            self.hyde_calls += 1
        if isinstance(payload, str):
            return _LLMResp(content=payload)
        return _LLMResp(content=json.dumps(payload))


def _round_retrieve(rounds):
    """Build a retrieve() that returns ``rounds[i]`` on the i-th call (clamping)."""
    state = {"i": 0}

    async def retrieve(hyde_answer, search_terms):
        i = min(state["i"], len(rounds) - 1)
        state["i"] += 1
        return list(rounds[i])

    return retrieve


def test_multi_round_loops_until_judge_says_sufficient():
    """Round 1 is judged insufficient (names a gap) so the loop continues; round 2
    is judged sufficient, stopping the loop. QFS-style convergence driven by the
    judge's sufficiency verdict, not a surface classifier."""
    retrieve = _round_retrieve([
        [_sr(i, f"r1 chunk {i}") for i in range(3)],     # round 1 pool
        [_sr(i + 10, f"r2 chunk {i}") for i in range(3)],  # round 2 pool (fresh ids)
    ])
    provider = SequencedProvider(
        hyde_seq=[{"hypothetical_answer": "h1"}, {"hypothetical_answer": "h2"}],
        judge_seq=[
            {"chunks": [{"i": 0, "relevance": 0.9, "faithfulness": 0.9, "keep": True},
                        {"i": 1, "relevance": 0.1, "faithfulness": 0.9, "keep": False},
                        {"i": 2, "relevance": 0.1, "faithfulness": 0.9, "keep": False}],
             "ranking": [0],
             "pool": {"sufficient": False, "confidence": 0.3, "missing_information": "aspect B"}},
            {"chunks": [{"i": 0, "relevance": 0.9, "faithfulness": 0.9, "keep": True},
                        {"i": 1, "relevance": 0.1, "faithfulness": 0.9, "keep": False},
                        {"i": 2, "relevance": 0.1, "faithfulness": 0.9, "keep": False}],
             "ranking": [0],
             "pool": {"sufficient": True, "confidence": 0.9}},
            "{}",  # finalize cross-round re-rank → fail-open → hybrid order
        ],
    )
    orch = _orch(provider, retrieve, budget=_budget(max_rounds=3, min_kept_chunks=1))
    result = asyncio.run(orch.run())
    assert result.rounds_run == 2
    assert result.stop_reason == "sufficient"
    assert result.hyde_variants_tried == 2     # a DIFFERENT HyDE each round
    assert result.kept_count == 2              # one kept per round, accumulated


def test_multi_round_cross_round_backfill_never_empty():
    """Inv-2 blocker fix: round 1 retrieves a NON-EMPTY pool but the judge keeps
    0; round 2's HyDE recycles into already-seen chunks (fresh=[]), ending the
    loop via no_progress. finalize must still return context, backfilled from the
    CROSS-ROUND reservoir — never empty on a non-empty retrieval."""
    round1 = [_sr(i, f"chunk {i}") for i in range(3)]
    retrieve = _round_retrieve([round1, round1])   # round 2 returns the SAME (seen) pool
    provider = SequencedProvider(
        hyde_seq=[{"hypothetical_answer": "h1"}, {"hypothetical_answer": "h2"}],
        judge_seq=[
            {"chunks": [{"i": i, "relevance": 0.0, "faithfulness": 0.0, "keep": False}
                        for i in range(3)],
             "ranking": [],
             "pool": {"sufficient": False, "confidence": 0.0, "missing_information": "everything"}},
        ],
    )
    orch = _orch(provider, retrieve, budget=_budget(max_rounds=2, min_kept_chunks=2))
    result = asyncio.run(orch.run())
    assert result.rounds_run == 2
    assert result.stop_reason == "no_progress"
    assert result.kept_count == 0
    assert len(result.reranked) >= 2           # backfilled from the reservoir
    assert result.backfilled >= 2


def test_judge_pool_max_does_not_burn_deep_pool():
    """judge_pool_max truncates the pool shown to the judge, but the un-judged
    TAIL must NOT be marked seen — so a chunk buried past the cap can still
    surface and be judged in a later round (deep-rank-burial recall guard)."""
    pool = [_sr(i, f"chunk {i}") for i in range(5)]
    retrieve = _round_retrieve([pool, pool])   # same pool both rounds
    provider = SequencedProvider(
        hyde_seq=[{"hypothetical_answer": "h1"}, {"hypothetical_answer": "h2"}],
        judge_seq=[
            # round 1 judges only the first 2 (judge_pool_max=2); keeps neither's
            # gold but is insufficient so the loop continues.
            {"chunks": [{"i": i, "relevance": 0.4, "faithfulness": 0.9, "keep": False}
                        for i in range(2)],
             "ranking": [],
             "pool": {"sufficient": False, "confidence": 0.2, "missing_information": "the rest"}},
            # round 2 sees the FRESH tail (chunks 2,3) — proof they weren't burned
            # — and keeps chunk 2.
            {"chunks": [{"i": 0, "relevance": 0.9, "faithfulness": 0.9, "keep": True},
                        {"i": 1, "relevance": 0.2, "faithfulness": 0.9, "keep": False}],
             "ranking": [0],
             "pool": {"sufficient": True, "confidence": 0.9}},
            "{}",
        ],
    )
    orch = _orch(provider, retrieve,
                 budget=_budget(max_rounds=2, min_kept_chunks=1, judge_pool_max=2))
    result = asyncio.run(orch.run())
    assert result.rounds_run == 2
    texts = {r.text for r in result.reranked}
    assert "chunk 2" in texts                  # the tail chunk survived to round 2


def test_qfs_off_single_round_reproduces_inc1_reason():
    """max_rounds=1 (QFS routing off) → exactly one round, stop_reason
    'single_round', regardless of the judge verdict."""
    async def retrieve(hyde_answer, search_terms):
        return [_sr(i, f"chunk {i}") for i in range(3)]

    provider = RoutingProvider(
        hyde={"hypothetical_answer": "h"},
        judge={"chunks": [{"i": 0, "relevance": 0.9, "faithfulness": 0.9, "keep": True}],
               "ranking": [0],
               "pool": {"sufficient": False, "confidence": 0.1, "missing_information": "lots"}},
    )
    orch = _orch(provider, retrieve, budget=_budget(max_rounds=1, min_kept_chunks=1))
    result = asyncio.run(orch.run())
    assert result.rounds_run == 1
    assert result.stop_reason == "single_round"


def test_cross_encoder_mode_does_not_burn_reranker_tail():
    """ranker="cross_encoder", multi-round: the cross-encoder drops the tail
    beyond keep_top_k, but those chunks must NOT be marked seen — so they can
    resurface in a later round. FakeReranker COPIES metadata, so this also pins
    the robust text-based source link (not metadata identity)."""
    pool = [_sr(i, f"chunk {i}") for i in range(4)]
    retrieve = _round_retrieve([pool, pool])     # same pool both rounds
    provider = SequencedProvider(
        hyde_seq=[{"hypothetical_answer": "h1"}, {"hypothetical_answer": "h2"}],
        judge_seq=[
            # round 1 sees the reranker's top-2 (chunks 0,1); keep neither, not yet
            # sufficient → continue.
            {"chunks": [{"i": 0, "relevance": 0.3, "faithfulness": 0.9, "keep": False},
                        {"i": 1, "relevance": 0.3, "faithfulness": 0.9, "keep": False}],
             "ranking": [],
             "pool": {"sufficient": False, "confidence": 0.2, "missing_information": "rest"}},
            # round 2 sees the FRESH tail (chunks 2,3) — proof the tail was not
            # burned — and keeps chunk 2.
            {"chunks": [{"i": 0, "relevance": 0.9, "faithfulness": 0.9, "keep": True},
                        {"i": 1, "relevance": 0.2, "faithfulness": 0.9, "keep": False}],
             "ranking": [0],
             "pool": {"sufficient": True, "confidence": 0.9}},
            "{}",
        ],
    )
    orch = _orch(provider, retrieve,
                 budget=_budget(max_rounds=2, min_kept_chunks=1,
                                keep_top_k_per_round=2, ranker="cross_encoder"))
    result = asyncio.run(orch.run())
    assert result.rounds_run == 2
    assert "chunk 2" in {r.text for r in result.reranked}   # tail survived to round 2


# ---------------------------------------------------------------------------
# Run-time role backstop (Slice E) — metadata fast-path + judge-prompt awareness
# ---------------------------------------------------------------------------


def _nav_backstop_provider() -> RoutingProvider:
    """Keep-all judge so the ONLY thing that can drop a chunk is the metadata
    backstop (the judge never drops here)."""
    return RoutingProvider(
        hyde={"hypothetical_answer": "h", "search_terms": []},
        judge={"chunks": [{"i": i, "relevance": 0.9, "faithfulness": 0.9, "keep": True}
                          for i in range(8)],
               "ranking": [0, 1, 2, 3, 4, 5, 6, 7],
               "pool": {"sufficient": True, "confidence": 0.9}},
    )


def test_role_backstop_drops_nav_chunk_pre_judge_when_on():
    """A retrieved candidate TAGGED chunk_role='navigation' is dropped BEFORE the
    judge when the metadata backstop is on — even though the (keep-all) judge would
    have kept it. Defense for a tagged-nav chunk that slipped the query filter.
    The chunk never reaches the judge prompt, and never reaches generation."""
    async def retrieve(hyde_answer, search_terms):
        return [
            _sr_role(0, "real content chunk", role="content"),
            _sr_role(1, "TOC navigation chunk", role="navigation"),
            _sr_role(2, "another content chunk", role="content"),
        ]

    provider = _nav_backstop_provider()
    orch = _orch(provider, retrieve, budget=_budget(min_kept_chunks=1),
                 final_top_k=5, fill_mode="none", role_backstop=True)
    result = asyncio.run(orch.run())

    texts = {r.text for r in result.reranked}
    assert texts == {"real content chunk", "another content chunk"}
    assert "TOC navigation chunk" not in texts
    # The dropped chunk never reached the judge (not in the rendered judge prompt).
    assert provider.judge_prompts, "judge should still run on the surviving chunks"
    assert "TOC navigation chunk" not in provider.judge_prompts[0]
    assert "real content chunk" in provider.judge_prompts[0]


def test_role_backstop_keeps_nav_chunk_when_off():
    """With the backstop OFF, a tagged-nav chunk is NOT dropped pre-judge — the
    config flag genuinely gates the behavior (so default-off is a no-op and the
    feature is fully toggleable, CLAUDE.md §3)."""
    async def retrieve(hyde_answer, search_terms):
        return [
            _sr_role(0, "real content chunk", role="content"),
            _sr_role(1, "TOC navigation chunk", role="navigation"),
            _sr_role(2, "another content chunk", role="content"),
        ]

    provider = _nav_backstop_provider()
    orch = _orch(provider, retrieve, budget=_budget(min_kept_chunks=1),
                 final_top_k=5, fill_mode="none", role_backstop=False)
    result = asyncio.run(orch.run())

    texts = {r.text for r in result.reranked}
    assert "TOC navigation chunk" in texts          # NOT dropped when off
    assert provider.judge_prompts
    assert "TOC navigation chunk" in provider.judge_prompts[0]   # it reached the judge


def test_role_backstop_drops_boilerplate_too():
    """The backstop excludes EVERY role in excluded_roles, not just navigation —
    a boilerplate-tagged chunk is dropped as well."""
    async def retrieve(hyde_answer, search_terms):
        return [
            _sr_role(0, "real content chunk", role="content"),
            _sr_role(1, "copyright notice chunk", role="boilerplate"),
        ]

    provider = _nav_backstop_provider()
    orch = _orch(provider, retrieve, budget=_budget(min_kept_chunks=1),
                 final_top_k=5, fill_mode="none", role_backstop=True,
                 excluded_roles=("navigation", "boilerplate"))
    result = asyncio.run(orch.run())
    texts = {r.text for r in result.reranked}
    assert texts == {"real content chunk"}


def test_role_backstop_fail_open_keeps_untagged_and_content():
    """FAIL-OPEN: a chunk with NO chunk_role (NULL/legacy) and a chunk tagged
    'content' are both KEPT by the backstop — only an explicitly EXCLUDED role is
    dropped. Never drop a chunk on a missing/unknown role (invariant #4)."""
    async def retrieve(hyde_answer, search_terms):
        return [
            _sr_role(0, "untagged legacy chunk", role=None),       # no chunk_role key
            _sr_role(1, "content-tagged chunk", role="content"),
            _sr_role(2, "unknown-role chunk", role="mystery"),     # not in excluded set
            _sr_role(3, "nav chunk", role="navigation"),
        ]

    provider = _nav_backstop_provider()
    orch = _orch(provider, retrieve, budget=_budget(min_kept_chunks=1),
                 final_top_k=5, fill_mode="none", role_backstop=True)
    result = asyncio.run(orch.run())
    texts = {r.text for r in result.reranked}
    assert "untagged legacy chunk" in texts
    assert "content-tagged chunk" in texts
    assert "unknown-role chunk" in texts          # unknown role is not excluded → kept
    assert "nav chunk" not in texts               # only the explicit excluded role drops


def test_role_backstop_all_nav_does_not_yield_empty():
    """Anti-refusal: if the backstop would drop every retrieved chunk (all tagged
    nav), the round simply judges nothing — never crashes — and the loop returns
    without context for that round rather than raising."""
    async def retrieve(hyde_answer, search_terms):
        return [
            _sr_role(0, "nav a", role="navigation"),
            _sr_role(1, "nav b", role="navigation"),
        ]

    provider = _nav_backstop_provider()
    orch = _orch(provider, retrieve, budget=_budget(min_kept_chunks=1),
                 final_top_k=5, fill_mode="none", role_backstop=True)
    result = asyncio.run(orch.run())          # must not raise
    assert result.rounds_run == 1
    texts = {r.text for r in result.reranked}
    assert "nav a" not in texts and "nav b" not in texts


def test_concise_judge_prompt_has_nav_awareness_line():
    """The concise judge prompt must instruct the model — by FUNCTION, no
    regex/keywords — to treat a table-of-contents / index / cross-reference /
    copyright / title-page chunk as NON-answer-bearing and not rank it. This is the
    LLM backstop for a leaked nav chunk that lacks the metadata tag."""
    from pathlib import Path
    import config.settings as cfg

    prompt = (Path(cfg.PROMPTS_DIR) / "agentic_chunk_judge_concise.md").read_text(
        encoding="utf-8"
    ).lower()
    # Function-based fingerprints the judge must be told to NOT rank.
    assert "table of contents" in prompt or "table-of-contents" in prompt
    assert "index" in prompt
    assert "cross-reference" in prompt or "cross reference" in prompt
    assert "copyright" in prompt
    assert "title page" in prompt or "title-page" in prompt
    # And the instruction is to NOT rank / treat as non-answer-bearing.
    assert "not rank" in prompt or "do not rank" in prompt or "non-answer-bearing" in prompt


def test_verbose_judge_prompt_has_nav_awareness_line():
    """Parity: the verbose judge prompt carries the same FUNCTION-based nav
    awareness instruction (both clients of one judging contract)."""
    from pathlib import Path
    import config.settings as cfg

    prompt = (Path(cfg.PROMPTS_DIR) / "agentic_chunk_judge.md").read_text(
        encoding="utf-8"
    ).lower()
    assert "table of contents" in prompt or "table-of-contents" in prompt
    assert "index" in prompt
    assert "cross-reference" in prompt or "cross reference" in prompt
    assert "copyright" in prompt
    assert "title page" in prompt or "title-page" in prompt
    assert "not rank" in prompt or "do not rank" in prompt or "non-answer-bearing" in prompt
