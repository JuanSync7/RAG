"""Verify _match_kg_terms uses the kgweave.client facade."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from kgweave.contracts.http import KGQueryMatchResponse


def test_match_kg_terms_calls_client_facade() -> None:
    fake = MagicMock()
    fake.match_kg_query.return_value = KGQueryMatchResponse(
        matched=["alpha-mod", "beta-mod"], used_fallback=False,
    )
    with patch("kgweave.client.get_client", return_value=fake):
        from src.retrieval.query.nodes.query_processor import _match_kg_terms  # noqa: PLC0415

        out = _match_kg_terms("alpha and beta")

    assert out == "Known terms in the knowledge base: alpha-mod, beta-mod"
    fake.match_kg_query.assert_called_once()
    call = fake.match_kg_query.call_args
    assert call.args[0] == "alpha and beta"
    assert call.kwargs["max_terms"] == 20


def test_match_kg_terms_returns_empty_string_when_client_returns_nothing() -> None:
    fake = MagicMock()
    fake.match_kg_query.return_value = KGQueryMatchResponse(
        matched=[], used_fallback=False,
    )
    with patch("kgweave.client.get_client", return_value=fake):
        from src.retrieval.query.nodes.query_processor import _match_kg_terms  # noqa: PLC0415

        assert _match_kg_terms("nothing matches") == ""


def test_match_kg_terms_swallows_client_failures() -> None:
    fake = MagicMock()
    fake.match_kg_query.side_effect = RuntimeError("backend down")
    with patch("kgweave.client.get_client", return_value=fake):
        from src.retrieval.query.nodes.query_processor import _match_kg_terms  # noqa: PLC0415

        assert _match_kg_terms("anything") == ""
