# @summary
# End-to-end integration test (slice E1) for RAPTOR-lite document routing. It
# exercises the REAL routing chain — build_document_card (A1) -> card collection
# -> decompose_query (C3, regex tier, no LLM) -> route_documents (B1) ->
# RAGChain._collect_candidates routed union (B2) -> RAGChain.run() Stage-1 wiring
# (B3) -> a single rerank pass — against IN-MEMORY FAKE backends (no live
# weaviate / vLLM / LLM). Only the EXTERNAL backends are faked; every routing
# function under test runs for real.
#
# Fakes: (1) a deterministic topic-vocabulary embedder (cosine is meaningful);
# (2) one FakeVectorSearch backing BOTH the chunk collection (RAGDocuments) and
# the card collection (RAGDocumentCards), honouring SearchFilter eq/in/not_in;
# (3) an identity-by-input-order fake reranker. The fake search is patched in at
# the TWO seams the code resolves it from: ``src.retrieval.pipeline.rag_chain.search``
# (used by _do_search) and ``src.retrieval.routing.router.search`` (used by the
# B1 router) — both are independent ``from src.vector_db import search`` binds.
#
# Asserts the design §10 validation set in BOTH directions: SHOULD-improve
# ("compare AXI4 vs CHI" surfaces BOTH authoritative spec docs) and MUST-NOT-
# regress (a needle register fact, and a single-document project question, both
# returned with routing ON and OFF). Fully deterministic: fixed fake vectors, no
# network, no randomness.
# Exports: (pytest test functions)
# Deps: pytest, unittest.mock, src.retrieval.pipeline.rag_chain,
#   src.retrieval.routing.router, src.ingest.embedding.common.card_builder,
#   src.vector_db.common.schemas, config.settings
# @end-summary
"""End-to-end test for RAPTOR-lite document routing (design §10 validation set).

This is the E1 slice. Unlike the B1/B2/B3 unit tests — which stub the routing
functions themselves — here ``decompose_query``, ``route_documents`` and
``_collect_candidates`` are the REAL implementations. Only the external backends
(vector search, embeddings, reranker, LLM) are faked, so the test verifies the
*wired* behaviour of the whole chain end to end.

What layer is driven
--------------------
The **full** :meth:`RAGChain.run` is driven (``skip_generation=True``), exactly
as the B3 wiring test drives it: a real :class:`RAGChain` is constructed with the
heavy dependencies mocked at ``__init__`` time, then the embedder / reranker /
vector-search seams are replaced with deterministic in-memory fakes. This is the
most faithful integration possible without live services — Stage-1 routing
(decompose -> route per sub-query -> union doc_ids), the Stage-4 routed candidate
union, and the single Stage-5 rerank all run for real over the seeded corpus.

Fake embedder topic scheme
--------------------------
``FakeEmbedder`` embeds text as an L2-normalised bag-of-topic-keywords vector
over a small fixed vocabulary (see ``TOPIC_VOCAB``). Each topic term is matched
case-insensitively as a whole word (with a couple of substring aliases for spec
IDs and the needle symbol). The component for a topic is 1.0 when its term is
present, else 0.0; the vector is then L2-normalised. Cosine similarity between
two such vectors therefore measures shared-topic overlap: "AXI4 vs CHI" embeds
near both AXI material and CHI material; the needle query embeds near the needle
chunk; the project query embeds near the project doc. This makes routing +
retrieval ordering deterministic and inspectable.

Why decomposition uses the regex tier
--------------------------------------
``RAG_DECOMPOSITION_LLM_PRIMARY`` is forced ``False`` for the comparison test so
the REAL deterministic ``regex_decompose`` splits "Compare AMBA AXI4 vs CHI
coherency" into ["AXI4", "CHI coherency"] with no LLM call. ``decompose_query``
(C3) is otherwise unmodified — the gate, comparison-intent detection and tier
selection all run for real.
"""
from __future__ import annotations

import math
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.ingest.embedding.common.card_builder import build_document_card
from src.retrieval.common.schemas import RankedResult
from src.vector_db.common.schemas import SearchFilter, SearchResult

pytestmark = pytest.mark.integration


# ─── Collection names (must match config defaults the code resolves) ─────────

CHUNK_COLLECTION = "RAGDocuments"          # == settings.VECTOR_COLLECTION_DEFAULT
CARD_COLLECTION = "RAGDocumentCards"       # == settings.RAG_DOCUMENT_CARD_COLLECTION

# §11 heading cap used when building cards (mirrors the ingest default).
_CARD_MAX_HEADINGS = 60


# ─── Deterministic topic-vocabulary fake embedder ───────────────────────────

# Fixed topic vocabulary. ``embed(text)`` -> L2-normalised vector whose i-th
# component is 1.0 iff topic ``TOPIC_VOCAB[i]`` is present in the text. Cosine
# similarity then measures shared-topic overlap (the property routing relies on).
TOPIC_VOCAB: tuple[str, ...] = (
    "axi",
    "axi4",
    "axi5",
    "chi",
    "ahb",
    "apb",
    "mbist",
    "regression",
    "pmu",
    "reset_value",
    "filters_status",
)

# Substring/keyword aliases mapped onto a topic dimension. Matched
# case-insensitively as plain substrings so spec IDs and the needle symbol pull
# the right topic even when the canonical token is absent.
_TOPIC_ALIASES: dict[str, tuple[str, ...]] = {
    "axi": ("ihi0022e", "advanced extensible interface", "axi3"),
    "chi": ("ihi0098", "coherent hub interface", "coherency", "coherent"),
    "reset_value": ("reset value", "offset 0x0064", "0x0064", "0x0"),
    "filters_status": ("filters_n_status", "filters_n", "filters status"),
    "pmu": ("power domain", "power management"),
    "regression": ("regression run", "regression suite"),
}


def _topic_present(text_lower: str, topic: str) -> bool:
    """True when *topic* (or one of its aliases) appears in lower-cased text.

    The canonical topic token is matched on word/segment boundaries (so ``axi``
    does not match inside ``axis``); aliases are matched as plain substrings.
    """
    # Whole-word-ish match for the canonical token: surround the text with
    # spaces and require a non-alphanumeric boundary on both sides.
    token = topic.replace("_", " ")
    padded = f" {text_lower} "
    canonical_words = token.split()
    if all(_word_in(padded, w) for w in canonical_words):
        return True
    for alias in _TOPIC_ALIASES.get(topic, ()):  # plain substring aliases
        if alias in text_lower:
            return True
    return False


def _word_in(padded_lower: str, word: str) -> bool:
    """Whole-word containment check inside a space-padded lower-cased string."""
    idx = padded_lower.find(word)
    while idx != -1:
        before = padded_lower[idx - 1] if idx > 0 else " "
        after_pos = idx + len(word)
        after = padded_lower[after_pos] if after_pos < len(padded_lower) else " "
        if not before.isalnum() and not after.isalnum():
            return True
        idx = padded_lower.find(word, idx + 1)
    return False


class FakeEmbedder:
    """Deterministic topic-vocabulary embedder (see module docstring).

    Exposes the single method the chain uses: :meth:`embed_query`. The returned
    vector is L2-normalised so cosine similarity is a clean topic-overlap score.
    A text with no recognised topic yields the zero vector (cosine 0 with
    everything) — that never happens for the seeded corpus or the test queries.
    """

    def embed(self, text: str) -> list[float]:
        lowered = (text or "").lower()
        vec = [1.0 if _topic_present(lowered, t) else 0.0 for t in TOPIC_VOCAB]
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    # The RAGChain calls ``self.embeddings.embed_query(text)``.
    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)

    # The card builder pipeline (here, the test seeding) embeds documents.
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def _cosine(a: list[float], b: list[float]) -> float:
    """Plain cosine similarity of two equal-length vectors (0 on a zero vector)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# ─── In-memory fake vector search (chunk + card collections) ─────────────────


class _Record:
    """One stored (text, vector, metadata) row in a fake collection."""

    __slots__ = ("text", "vector", "metadata")

    def __init__(self, text: str, vector: list[float], metadata: dict[str, Any]):
        self.text = text
        self.vector = vector
        self.metadata = metadata


def _filter_matches(meta: dict[str, Any], flt: SearchFilter) -> bool:
    """Evaluate ONE SearchFilter clause against a record's metadata.

    Honours the operators the routing chain emits: ``eq`` (leaf-only
    ``node_kind == "chunk"`` and tenant/source pins), ``in`` (the B2 routed
    ``document_id`` ``contains_any``), and ``not_in`` (ignored docs). Unknown
    operators conservatively match (the fakes never emit them).
    """
    actual = meta.get(flt.property)
    op = flt.operator
    if op == "eq":
        return actual == flt.value
    if op == "ne":
        return actual != flt.value
    if op == "in":
        return actual in (flt.value or [])
    if op == "not_in":
        return actual not in (flt.value or [])
    return True


class FakeVectorSearch:
    """In-memory hybrid-search stand-in backing every collection by name.

    Stores ``_Record`` rows per collection. :meth:`__call__` matches the
    ``src.vector_db.search`` facade signature; on each call it computes cosine
    similarity between ``query_embedding`` and the stored vectors of the named
    collection, applies the AND-combined ``SearchFilter`` list, sorts by score
    (descending, stable) and returns the top-``limit`` :class:`SearchResult`
    objects. ``alpha`` and ``query`` (the keyword component) are accepted and
    ignored — scoring is pure cosine, which is all the deterministic fakes need.
    """

    def __init__(self) -> None:
        self._by_collection: dict[str, list[_Record]] = {}
        self.calls: list[dict[str, Any]] = []

    def add(self, collection: str, text: str, vector: list[float],
            metadata: dict[str, Any]) -> None:
        self._by_collection.setdefault(collection, []).append(
            _Record(text, vector, metadata)
        )

    def __call__(
        self,
        client: Any = None,
        query: str = "",
        query_embedding: Optional[list[float]] = None,
        alpha: float = 0.5,
        limit: int = 10,
        filters: Optional[list[SearchFilter]] = None,
        collection: Optional[str] = None,
    ) -> list[SearchResult]:
        self.calls.append(
            {"collection": collection, "limit": limit, "filters": filters,
             "query": query, "alpha": alpha}
        )
        records = self._by_collection.get(collection or CHUNK_COLLECTION, [])
        flt_list = list(filters or [])
        qv = query_embedding or []

        scored: list[tuple[float, _Record]] = []
        for rec in records:
            if not all(_filter_matches(rec.metadata, f) for f in flt_list):
                continue
            scored.append((_cosine(qv, rec.vector), rec))

        # Stable sort by descending score (Python's sort is stable, so ties keep
        # insertion order — keeps the fake deterministic).
        scored.sort(key=lambda pair: pair[0], reverse=True)

        out: list[SearchResult] = []
        for score, rec in scored[: max(0, int(limit))]:
            out.append(
                SearchResult(
                    text=rec.text,
                    score=score,
                    metadata=dict(rec.metadata),
                    object_id=rec.metadata.get("chunk_id"),
                    collection=collection,
                )
            )
        return out


# ─── Identity-order fake reranker ────────────────────────────────────────────


def _fake_rerank(query: str, documents: list[Any], top_k: int = 10) -> list[RankedResult]:
    """Cosine-by-topic reranker over the candidate pool.

    Scores each candidate by topic-cosine to the query (deterministic, mirrors
    the embedder scheme), sorts descending and returns the top-``top_k`` as
    :class:`RankedResult`. This keeps the SINGLE-rerank contract: the reranker is
    the final authority over the (routed) candidate union, exactly as in
    production — it just uses a transparent scoring function instead of a model.
    """
    embedder = FakeEmbedder()
    qv = embedder.embed(query)
    ranked = [
        RankedResult(
            text=getattr(doc, "text", ""),
            score=_cosine(qv, embedder.embed(getattr(doc, "text", ""))),
            metadata=getattr(doc, "metadata", {}) or {},
        )
        for doc in documents
    ]
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked[: max(0, int(top_k))]


# ─── Synthetic corpus seeding ────────────────────────────────────────────────


def _chunk(document_id: str, title: str, source: str, heading_path: list[str],
           text: str) -> dict[str, Any]:
    """Build a synthetic leaf-chunk dict (card_builder + fake-search shape).

    Carries the metadata the real card_builder reads (``document_id`` / ``title``
    / ``source`` / ``heading_path``) and the leaf marker ``node_kind="chunk"``
    so it survives the chain's leaf-only filter. ``chunk_id`` keys dedup.
    """
    return {
        "text": text,
        "metadata": {
            "document_id": document_id,
            "title": title,
            "source": source,
            "heading_path": list(heading_path),
            "node_kind": "chunk",
            "chunk_id": f"{document_id}:{len(heading_path)}:{abs(hash(text)) % 10_000}",
        },
    }


# The seeded corpus (design §10): two authoritative specs, two keyword-overlapping
# distractors, one needle doc, one single-document project doc.
def _build_corpus() -> list[dict[str, Any]]:
    corpus: list[dict[str, Any]] = []

    # doc "axi_spec" (IHI0022E-like) — AXI/AXI4/AXI5 read/write channels.
    corpus += [
        _chunk("axi_spec", "AMBA AXI Protocol Specification (IHI0022E)",
               "ihi0022e.pdf", ["AXI", "Read channels"],
               "AXI read address channel and read data channel handshake. AXI4 "
               "burst-based reads. IHI0022E."),
        _chunk("axi_spec", "AMBA AXI Protocol Specification (IHI0022E)",
               "ihi0022e.pdf", ["AXI", "Write channels"],
               "AXI write address, write data and write response channels. AXI4 "
               "and AXI5 write transactions."),
        _chunk("axi_spec", "AMBA AXI Protocol Specification (IHI0022E)",
               "ihi0022e.pdf", ["AXI5", "Atomic transactions"],
               "AXI5 atomic transaction extensions and coherency alignment."),
    ]

    # doc "chi_spec" (IHI0098-like) — CHI coherency / hub.
    corpus += [
        _chunk("chi_spec", "AMBA CHI Architecture Specification (IHI0098)",
               "ihi0098.pdf", ["CHI", "Coherency model"],
               "CHI coherent hub interface coherency model. Snoop transactions "
               "across the hub. IHI0098."),
        _chunk("chi_spec", "AMBA CHI Architecture Specification (IHI0098)",
               "ihi0098.pdf", ["CHI", "Request nodes"],
               "CHI request nodes, home nodes and the coherent hub topology."),
    ]

    # doc "glossary" — distractor: keyword-overlapping but NOT authoritative.
    # Several AXI-heavy chunks so a NARROW flat top-k is dominated by AXI-topic
    # distractors, crowding the (single) CHI spec out of the flat window. This
    # is the §10 "improve" failure mode routing is meant to fix: the dedicated
    # CHI sub-query routes to chi_spec and the per-doc union pulls a CHI chunk in
    # regardless of how AXI-heavy the flat pool is.
    corpus += [
        _chunk("glossary", "Project Glossary", "glossary.md",
               ["Glossary", "AXI acronyms"],
               "Glossary: AXI means Advanced eXtensible Interface. AXI4 and AXI5 "
               "are AXI generations. AXI AXI AXI definitions."),
        _chunk("glossary", "Project Glossary", "glossary.md",
               ["Glossary", "AXI4 acronyms"],
               "Glossary: AXI4 is the fourth-generation AXI protocol. AXI4 "
               "AXI4 AXI4 acronym expansion only, not authoritative."),
    ]

    # doc "proj_perf" — distractor: more AXI-flavoured perf chunks.
    corpus += [
        _chunk("proj_perf", "SoC Performance Notes", "perf_notes.md",
               ["Performance", "AXI tuning"],
               "Performance notes: AXI fabric throughput tuning. AXI4 burst "
               "sizing. AXI AXI4 AXI4 latency knobs (not a spec)."),
        _chunk("proj_perf", "SoC Performance Notes", "perf_notes.md",
               ["Performance", "AXI5 tuning"],
               "Performance notes: AXI5 atomic-op tuning. AXI AXI4 AXI5 "
               "bandwidth observations (throughput doc, not authoritative)."),
    ]

    # doc "reg_needle" — the needle. Its CARD (title + heading) must NOT obviously
    # match a needle query about a specific register reset value; the needle fact
    # lives only in the chunk body (so it must arrive via the flat path).
    corpus += [
        _chunk("reg_needle", "Peripheral Register Map", "regmap.md",
               ["Registers", "Status registers"],
               "Filters_N_Status reset value at offset 0x0064 is 0x0. This "
               "register reports per-channel filter status after MBIST."),
    ]

    # doc "vitec_pmu" — single-document project doc.
    corpus += [
        _chunk("vitec_pmu", "Vitec PMU Integration Guide", "vitec_pmu.md",
               ["Vitec PMU", "Power domains"],
               "Vitec PMU power domains: the power management unit gates the "
               "core, fabric and peripheral power domains for the Vitec SoC."),
    ]

    return corpus


def _seed(fake_search: FakeVectorSearch, embedder: FakeEmbedder) -> None:
    """Embed every chunk into the chunk collection and every doc CARD into the
    card collection, using the REAL ``build_document_card`` for the cards."""
    corpus = _build_corpus()

    # 1) Chunk collection: one record per chunk, embedded by its body text.
    for ch in corpus:
        fake_search.add(
            CHUNK_COLLECTION,
            text=ch["text"],
            vector=embedder.embed(ch["text"]),
            metadata=dict(ch["metadata"]),
        )

    # 2) Card collection: group chunks by document_id, build the real card, embed
    #    its ``card_text`` (title + headings — routing-only, no chunk bodies).
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for ch in corpus:
        by_doc.setdefault(ch["metadata"]["document_id"], []).append(ch)

    for document_id, chunks in by_doc.items():
        card = build_document_card(chunks, max_headings=_CARD_MAX_HEADINGS)
        fake_search.add(
            CARD_COLLECTION,
            text=card["card_text"],
            vector=embedder.embed(card["card_text"]),
            metadata={
                "document_id": card["document_id"],
                "title": card["title"],
                "source": card["source"],
            },
        )


# ─── RAGChain factory (real chain, heavy deps mocked) ────────────────────────


def _build_chain(fake_search: FakeVectorSearch, embedder: FakeEmbedder):
    """Construct a run()-drivable RAGChain whose external backends are the fakes.

    Mirrors the B3 wiring harness: every heavy dependency is mocked at __init__
    time, then the embedder / reranker seams are replaced with the deterministic
    fakes. The vector-search seams are patched by the caller (two namespaces).
    """
    with patch("src.retrieval.pipeline.rag_chain.create_persistent_client"), \
         patch("src.retrieval.pipeline.rag_chain.ensure_collection"), \
         patch("src.retrieval.pipeline.rag_chain.get_embedding_provider") as gep, \
         patch("src.retrieval.pipeline.rag_chain.get_reranker_provider") as grp, \
         patch("src.retrieval.pipeline.rag_chain.OllamaGenerator") as gen_cls, \
         patch("src.retrieval.pipeline.rag_chain._get_kg_client") as gkc, \
         patch("src.retrieval.pipeline.rag_chain.KG_ENABLED", False), \
         patch("src.retrieval.pipeline.rag_chain.GENERATION_ENABLED", False):
        gep.return_value = MagicMock()
        grp.return_value = MagicMock()
        gen_cls.return_value = MagicMock()
        gkc.return_value = None

        from src.retrieval.pipeline.rag_chain import RAGChain  # noqa: PLC0415
        chain = RAGChain(persistent_weaviate=False)

    chain._generator = None
    # Replace the external backends with the deterministic fakes.
    chain.embeddings.embed_query = embedder.embed_query
    chain.reranker.rerank = _fake_rerank
    return chain


def _stub_process_query(processed: str):
    """Patch context for process_query yielding a simple search-action result."""
    proc = patch("src.retrieval.pipeline.rag_chain.process_query")
    return proc, MagicMock(
        processed_query=processed,
        standalone_query=processed,
        confidence=0.9,
        action=MagicMock(value="search"),
        has_backward_reference=False,
        suppress_memory=False,
    )


def _run(chain, fake_search: FakeVectorSearch, query: str, *,
         routing_enabled: bool, decomposition_enabled: bool = False,
         search_limit: int = 10, rerank_top_k: int = 5):
    """Drive RAGChain.run() with the fakes patched in at BOTH search seams.

    Patches the search facade as imported into the rag_chain module (used by
    _do_search) AND as imported into the router module (used by route_documents)
    — they are independent ``from src.vector_db import search`` binds. Forces the
    regex decomposition tier (no LLM) and disables rerank-fusion so the single
    cosine rerank is the sole authority over the union.

    ``search_limit`` is the FLAT-path window; a narrow window lets distractors
    crowd an authoritative doc out of the OFF baseline while routing's per-doc
    union still pulls it in (the §10 "improve" scenario).
    """
    proc_patch, proc_val = _stub_process_query(query)
    with proc_patch as proc, \
         patch("src.retrieval.pipeline.rag_chain.search", fake_search), \
         patch("src.retrieval.routing.router.search", fake_search), \
         patch("src.retrieval.pipeline.rag_chain.RAG_DOCUMENT_ROUTING_ENABLED",
               routing_enabled), \
         patch("src.retrieval.pipeline.rag_chain.RAG_RERANK_FUSION_ENABLED", False), \
         patch("config.settings.RAG_DECOMPOSITION_ENABLED", decomposition_enabled), \
         patch("config.settings.RAG_DECOMPOSITION_LLM_PRIMARY", False):
        proc.return_value = proc_val
        return chain.run(
            query=query, skip_generation=True,
            search_limit=search_limit, rerank_top_k=rerank_top_k,
        )


def _result_doc_ids(response) -> list[str]:
    """The document_ids represented in a RAGResponse's reranked results."""
    return [
        (r.metadata or {}).get("document_id")
        for r in (response.results or [])
    ]


# ─── SHOULD-IMPROVE: comparison surfaces BOTH authoritative specs ────────────


def test_should_improve_comparison_surfaces_both_specs():
    """Design §10 (improve): with routing + decomposition ON, "compare AXI4 vs
    CHI" must surface chunks from BOTH the AXI spec AND the CHI spec.

    The REAL decompose_query (regex tier) splits the query per entity; each
    sub-query routes over the card index (B1); the union feeds the routed
    candidate union (B2); the single rerank (B3) is the final authority.

    To make the improvement load-bearing (not vacuously satisfied by a wide flat
    pool), the flat window is narrowed (``search_limit=2``) and the corpus seeds
    several AXI-flavoured distractor chunks. With routing OFF the narrow flat
    window is crowded and only ONE authoritative spec survives; with routing ON
    the dedicated per-entity sub-queries route to both specs and the per-doc
    union pulls BOTH in. The required bar (ON surfaces both) is asserted; the OFF
    comparison additionally demonstrates the regression routing fixes.
    """
    embedder = FakeEmbedder()
    both = {"axi_spec", "chi_spec"}
    query = "Compare AMBA AXI4 vs CHI coherency"

    # Routing ON (+ decomposition) — the under-test path. Narrow flat window.
    fs_on = FakeVectorSearch()
    _seed(fs_on, embedder)
    chain_on = _build_chain(fs_on, embedder)
    resp_on = _run(chain_on, fs_on, query, routing_enabled=True,
                   decomposition_enabled=True, search_limit=2, rerank_top_k=5)
    docs_on = set(_result_doc_ids(resp_on))

    # Required bar: ON surfaces BOTH authoritative specs.
    assert both <= docs_on, (
        f"routing ON must surface BOTH the AXI and CHI specs; got {docs_on}"
    )

    # Routing OFF baseline, same narrow flat window — distractors crowd one spec
    # out, so OFF covers strictly fewer of the two specs than ON.
    fs_off = FakeVectorSearch()
    _seed(fs_off, embedder)
    chain_off = _build_chain(fs_off, embedder)
    resp_off = _run(chain_off, fs_off, query, routing_enabled=False,
                    search_limit=2, rerank_top_k=5)
    docs_off = set(_result_doc_ids(resp_off))

    assert (both & docs_off) != both, (
        "test corpus precondition: OFF must NOT already cover both specs in the "
        f"narrow flat window (else routing's improvement is untestable); got {docs_off}"
    )
    assert len(both & docs_on) > len(both & docs_off), (
        "routing ON must cover MORE authoritative specs than OFF "
        f"(ON={both & docs_on}, OFF={both & docs_off})"
    )


def test_should_improve_routing_unions_both_spec_docids():
    """The REAL Stage-1 routing unions BOTH spec document_ids for the comparison.

    Drives ``_route_documents_stage1`` (via run's wiring) over the seeded card
    index and asserts the union of routed doc_ids contains both authoritative
    specs — proving the route_documents (B1) + per-sub-query union (B3) chain
    runs for real, not just that the reranker happened to pick them.
    """
    embedder = FakeEmbedder()
    fake_search = FakeVectorSearch()
    _seed(fake_search, embedder)

    chain = _build_chain(fake_search, embedder)
    seen: list[Optional[list[str]]] = []
    real_collect = chain._collect_candidates

    def _spy(*args, **kwargs):
        seen.append(kwargs.get("routed_doc_ids"))
        return real_collect(*args, **kwargs)

    chain._collect_candidates = _spy  # type: ignore[assignment]

    _run(chain, fake_search, "Compare AMBA AXI4 vs CHI coherency",
         routing_enabled=True, decomposition_enabled=True)

    assert seen, "collect_candidates must be called"
    routed = seen[0] or []
    assert "axi_spec" in routed and "chi_spec" in routed, (
        f"Stage-1 routing must union both spec doc_ids; got {routed}"
    )


# ─── MUST-NOT-REGRESS: needle fact survives routing ON and OFF ───────────────


def test_must_not_regress_needle_fact_returned_both_modes():
    """Design §10 (no regress): the needle register fact is returned with routing
    ON and OFF. Its CARD (title "Peripheral Register Map" + headings) does NOT
    match a needle query, so it can only arrive via the FLAT path — proving
    routing's soft union never drops a flat-path needle."""
    embedder = FakeEmbedder()
    fake_search = FakeVectorSearch()
    _seed(fake_search, embedder)

    query = "reset value of Filters_N_Status at offset 0x0064"

    chain_on = _build_chain(fake_search, embedder)
    resp_on = _run(chain_on, fake_search, query, routing_enabled=True)
    chain_off = _build_chain(fake_search, embedder)
    resp_off = _run(chain_off, fake_search, query, routing_enabled=False)

    assert "reg_needle" in _result_doc_ids(resp_on), (
        "needle chunk must be returned with routing ON (flat path)"
    )
    assert "reg_needle" in _result_doc_ids(resp_off), (
        "needle chunk must be returned with routing OFF (today's path)"
    )
    # The needle chunk should rank first in both modes (it is the only chunk
    # carrying the reset_value/filters_status/offset topics).
    assert _result_doc_ids(resp_on)[0] == "reg_needle"
    assert _result_doc_ids(resp_off)[0] == "reg_needle"


# ─── MUST-NOT-REGRESS: single-document project query unaffected ──────────────


def test_must_not_regress_single_doc_project_query():
    """Design §10 (no regress): "Vitec PMU power domains" returns the vitec_pmu
    doc with routing ON and OFF, and routing does not degrade its rank."""
    embedder = FakeEmbedder()
    fake_search = FakeVectorSearch()
    _seed(fake_search, embedder)

    query = "Vitec PMU power domains"

    chain_on = _build_chain(fake_search, embedder)
    resp_on = _run(chain_on, fake_search, query, routing_enabled=True)
    chain_off = _build_chain(fake_search, embedder)
    resp_off = _run(chain_off, fake_search, query, routing_enabled=False)

    assert "vitec_pmu" in _result_doc_ids(resp_on), (
        "single-doc project content must be returned with routing ON"
    )
    assert "vitec_pmu" in _result_doc_ids(resp_off), (
        "single-doc project content must be returned with routing OFF"
    )
    # Top hit is the project doc in both modes (no degradation from routing).
    assert _result_doc_ids(resp_on)[0] == "vitec_pmu"
    assert _result_doc_ids(resp_off)[0] == "vitec_pmu"


# ─── Determinism: identical inputs -> identical outputs, no services ─────────


def test_e2e_is_deterministic_and_offline():
    """Two identical routing-ON runs of the comparison query return byte-identical
    ordered (text, document_id) result tuples — confirming fixed fake vectors,
    no randomness and no network dependency."""
    embedder = FakeEmbedder()

    def _once() -> list[tuple[str, Optional[str]]]:
        fs = FakeVectorSearch()
        _seed(fs, embedder)
        chain = _build_chain(fs, embedder)
        resp = _run(chain, fs, "Compare AMBA AXI4 vs CHI coherency",
                    routing_enabled=True, decomposition_enabled=True)
        return [(r.text, (r.metadata or {}).get("document_id"))
                for r in (resp.results or [])]

    assert _once() == _once()
