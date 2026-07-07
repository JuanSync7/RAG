"""Verify the rag_chain KG expansion call site uses kgweave.client.

Covers the migration from direct GraphQueryExpander construction to the
KGQueryService facade. The chain should:

  1. Resolve the read-side service via kgweave.client.get_client()
  2. Probe health at startup, log backend stats
  3. Call service.expand(...) at retrieval time, returning ExpandResponse
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from kgweave.contracts.http import ExpandResponse, HealthResponse


def _fake_service(*, terms=None, graph_context: str = "") -> MagicMock:
    service = MagicMock()
    service.health.return_value = HealthResponse(
        ok=True, contract_version="v1", backend="FakeBackend",
        stats={"nodes": 10, "edges": 7},
    )
    service.expand.return_value = ExpandResponse(
        terms=list(terms or []),
        graph_context=graph_context,
    )
    return service


def test_load_kg_uses_get_client_and_returns_service(tmp_path, monkeypatch):
    """`_load_kg` should call kgweave.client.get_client() when KG enabled+present."""
    fake = _fake_service(terms=["alpha"])
    fake_path = tmp_path / "graph.json"
    fake_path.write_text("{}")

    with patch("src.retrieval.pipeline.rag_chain.create_persistent_client"), \
         patch("src.retrieval.pipeline.rag_chain.ensure_collection"), \
         patch("src.retrieval.pipeline.rag_chain.get_embedding_provider"), \
         patch("src.retrieval.pipeline.rag_chain.get_reranker_provider"), \
         patch("src.retrieval.pipeline.rag_chain.OllamaGenerator"), \
         patch("src.retrieval.pipeline.rag_chain._get_kg_client", return_value=fake), \
         patch("src.retrieval.pipeline.rag_chain.KG_ENABLED", True):
        from src.retrieval.pipeline.rag_chain import RAGChain  # noqa: PLC0415

        chain = RAGChain(persistent_weaviate=False)

    assert chain._kg_expander is fake
    fake.health.assert_called_once()


def test_load_kg_returns_none_when_disabled(tmp_path):
    with patch("src.retrieval.pipeline.rag_chain.create_persistent_client"), \
         patch("src.retrieval.pipeline.rag_chain.ensure_collection"), \
         patch("src.retrieval.pipeline.rag_chain.get_embedding_provider"), \
         patch("src.retrieval.pipeline.rag_chain.get_reranker_provider"), \
         patch("src.retrieval.pipeline.rag_chain.OllamaGenerator"), \
         patch("src.retrieval.pipeline.rag_chain._get_kg_client") as get_client, \
         patch("src.retrieval.pipeline.rag_chain.KG_ENABLED", False):
        from src.retrieval.pipeline.rag_chain import RAGChain  # noqa: PLC0415

        chain = RAGChain(persistent_weaviate=False)

    assert chain._kg_expander is None
    get_client.assert_not_called()


def test_load_kg_returns_none_when_health_raises(tmp_path):
    fake = MagicMock()
    fake.health.side_effect = RuntimeError("backend offline")
    fake_path = tmp_path / "graph.json"
    fake_path.write_text("{}")

    with patch("src.retrieval.pipeline.rag_chain.create_persistent_client"), \
         patch("src.retrieval.pipeline.rag_chain.ensure_collection"), \
         patch("src.retrieval.pipeline.rag_chain.get_embedding_provider"), \
         patch("src.retrieval.pipeline.rag_chain.get_reranker_provider"), \
         patch("src.retrieval.pipeline.rag_chain.OllamaGenerator"), \
         patch("src.retrieval.pipeline.rag_chain._get_kg_client", return_value=fake), \
         patch("src.retrieval.pipeline.rag_chain.KG_ENABLED", True):
        from src.retrieval.pipeline.rag_chain import RAGChain  # noqa: PLC0415

        chain = RAGChain(persistent_weaviate=False)

    assert chain._kg_expander is None
